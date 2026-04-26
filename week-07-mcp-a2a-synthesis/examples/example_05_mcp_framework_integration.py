import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

"""
Example 5: MCP Integration with Orchestration Frameworks (LangGraph & ADK)
===========================================================================
Topic 5 — How to plug MCP tools into the frameworks you already know.

The BIG IDEA (Feynman):
  Think of MCP as a universal power adapter.  Your LangGraph or ADK agent
  is a laptop.  MCP tools are appliances in a foreign country.  This example
  builds the "adapter" that lets your existing agents use ANY MCP tool
  without rewriting the agent itself.

Previously covered (review):
  - LangGraph tool agents (Week 2-3)
  - ADK tool agents (Week 2-3)
  - MCP client basics (example_03)
  - MCP server basics (example_04)

NEW in this example:
  - MCPToolNode: wraps MCP client calls as a LangGraph tool node
  - mcp_to_langchain_tool: converts MCP tool metadata → LangChain tool
  - Async MCP client lifecycle management inside a graph
  - Phoenix tracing at the MCP call boundary

Run: python week-07-mcp-a2a-synthesis/examples/example_05_mcp_framework_integration.py
"""

import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, Any
from dataclasses import dataclass, field
from datetime import datetime


# ================================================================
# LLM Setup (same pattern as all weeks)
# ================================================================

def get_llm(temperature=0.3):
    """Create LLM based on provider setting."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )


# ================================================================
# Phoenix Tracing (optional — graceful degradation)
# ================================================================

try:
    from config.phoenix_config import setup_tracing
    setup_tracing()
    PHOENIX_AVAILABLE = True
except Exception:
    PHOENIX_AVAILABLE = False


# ================================================================
# PART 1: THE ADAPTER PATTERN (Feynman explanation)
# ================================================================
# Without an adapter, you'd write code like this for EVERY MCP tool:
#
#   @tool
#   def get_weather(city: str) -> str:
#       # manually connect to MCP server
#       # manually call the tool
#       # manually parse the result
#       return result
#
# That defeats the purpose of MCP!  Instead, we build ONE adapter
# that can wrap ANY MCP tool as a LangChain/ADK tool automatically.
#
# Analogy: Instead of buying a separate charger for each device,
# you buy one universal adapter and plug anything into it.

print("=" * 70)
print("PART 1: MCP → LangChain Tool Adapter (The Universal Adapter)")
print("=" * 70)


# ================================================================
# MCP Client Manager — handles connection lifecycle
# ================================================================

@dataclass
class MCPToolInfo:
    """Metadata about an MCP tool, extracted from the server."""
    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's input


class MCPClientManager:
    """Manages the lifecycle of an MCP client connection.

    Think of this as the "plug" that stays in the wall socket —
    you connect once and can use any tool through it.
    """

    def __init__(self, server_command: str, server_args: list[str]):
        self.server_command = server_command
        self.server_args = server_args
        self._session = None
        self._read = None
        self._write = None
        self._client_ctx = None
        self._session_ctx = None
        self.tools: list[MCPToolInfo] = []

    async def connect(self):
        """Establish connection to the MCP server."""
        try:
            from mcp.client.stdio import stdio_client
            from mcp.client.session import ClientSession
        except ImportError:
            print("⚠️  mcp library not installed. Run: pip install mcp>=1.0.0")
            return False

        try:
            from mcp.client.stdio import StdioServerParameters
        except ImportError:
            from mcp import StdioServerParameters

        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
        )

        # Open the connection (this launches the server subprocess)
        self._client_ctx = stdio_client(server_params)
        self._read, self._write = await self._client_ctx.__aenter__()

        # Create a session (handles the JSON-RPC protocol)
        self._session_ctx = ClientSession(self._read, self._write)
        self._session = await self._session_ctx.__aenter__()

        # Initialize — capability handshake
        await self._session.initialize()

        # Discover available tools
        tools_result = await self._session.list_tools()
        self.tools = [
            MCPToolInfo(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema if hasattr(t, 'inputSchema') else {},
            )
            for t in tools_result.tools
        ]

        print(f"  ✅ Connected to MCP server with {len(self.tools)} tools:")
        for t in self.tools:
            print(f"     • {t.name}: {t.description[:60]}...")

        return True

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call an MCP tool and return the text result."""
        if not self._session:
            return "Error: Not connected to MCP server"

        result = await self._session.call_tool(name, arguments)

        # Extract text from the result
        if result.content:
            texts = [c.text for c in result.content if hasattr(c, 'text')]
            return "\n".join(texts) if texts else str(result.content)
        return "No result returned"

    async def disconnect(self):
        """Clean up the connection."""
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        if self._client_ctx:
            await self._client_ctx.__aexit__(None, None, None)
        print("  🔌 Disconnected from MCP server")


# ================================================================
# PART 2: LangGraph Integration — MCP tools as graph nodes
# ================================================================

print("\n" + "=" * 70)
print("PART 2: LangGraph + MCP Integration")
print("=" * 70)
print("""
The strategy:
  1. Connect to MCP server → discover tools
  2. Convert each MCP tool → LangChain @tool function
  3. Bind tools to the LLM (llm.bind_tools)
  4. Build a standard LangGraph ReAct agent

The agent doesn't know or care that the tools come from MCP.
It just sees normal LangChain tools!
""")


def mcp_to_langchain_tools(mcp_manager: MCPClientManager):
    """Convert MCP tools to LangChain tool functions.

    This is the "universal adapter" — it reads the MCP tool metadata
    (name, description, parameters) and generates LangChain-compatible
    tool functions that call through to the MCP server.
    """
    from langchain_core.tools import StructuredTool

    lc_tools = []
    for mcp_tool in mcp_manager.tools:
        # Build an async function that calls the MCP tool.
        # The MCP session is bound to the current event loop, so we must
        # await it on the same loop — not spin up a new loop in a thread.
        def make_async_caller(tool_name):
            async def call_mcp_tool(**kwargs) -> str:
                return await mcp_manager.call_tool(tool_name, kwargs)
            return call_mcp_tool

        lc_tool = StructuredTool.from_function(
            coroutine=make_async_caller(mcp_tool.name),
            name=mcp_tool.name,
            description=mcp_tool.description,
        )
        lc_tools.append(lc_tool)

    return lc_tools


# ================================================================
# PART 3: Build a LangGraph agent with MCP tools
# ================================================================

async def demo_langgraph_mcp():
    """Demonstrate a LangGraph agent using MCP tools."""
    from langgraph.graph import StateGraph, END, add_messages
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    # Step 1: Connect to MCP server
    server_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mcp-servers", "research_history_server.py"
    )

    print(f"\n  📡 Connecting to research history MCP server...")
    manager = MCPClientManager("python", [server_path])

    if not await manager.connect():
        print("  ❌ Failed to connect. Skipping LangGraph demo.")
        return

    try:
        # Step 2: Convert MCP tools to LangChain tools
        lc_tools = mcp_to_langchain_tools(manager)
        print(f"\n  🔧 Converted {len(lc_tools)} MCP tools to LangChain tools")

        # Step 3: Create LLM with tools bound
        llm = get_llm(temperature=0)
        llm_with_tools = llm.bind_tools(lc_tools)

        # Step 4: Build the graph
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]

        async def agent_node(state: AgentState) -> dict:
            """LLM decides what to do next."""
            response = await llm_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        async def tool_node(state: AgentState) -> dict:
            """Execute tool calls from the LLM (async — MCP session is loop-bound)."""
            last_msg = state["messages"][-1]
            results = []

            for tool_call in last_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"  🔨 Calling MCP tool: {tool_name}({tool_args})")

                # Find the matching LangChain tool and invoke it
                for lc_tool in lc_tools:
                    if lc_tool.name == tool_name:
                        result = await lc_tool.ainvoke(tool_args)
                        from langchain_core.messages import ToolMessage
                        results.append(
                            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                        )
                        break

            return {"messages": results}

        def should_continue(state: AgentState) -> str:
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"
            return "end"

        # Wire the graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        app = graph.compile()

        # Step 5: Run the agent
        print("\n  🤖 Running LangGraph agent with MCP tools...\n")

        # First, save some research
        result = await manager.call_tool("save_research", {
            "topic": "AI in Healthcare",
            "summary": "AI is transforming healthcare through diagnostic imaging, drug discovery, and personalized treatment plans. Deep learning models can detect diseases in medical images with accuracy matching or exceeding human radiologists.",
            "sources": '["https://nature.com/ai-health", "https://nejm.org/ai-review"]'
        })
        print(f"  Pre-loaded research:\n  {result}\n")

        # Now ask the agent to search
        messages = [
            SystemMessage(content=(
                "You are a research assistant with MCP tools. "
                "Use each tool AT MOST ONCE. After you have the results, write a final "
                "answer summarizing what you found — do NOT call the same tool again."
            )),
            HumanMessage(content="Search for existing research on 'AI in healthcare' and then show the database stats. Then summarize both in one reply.")
        ]

        # Cap graph steps so a looping LLM can't run forever.
        final_state = await app.ainvoke(
            {"messages": messages},
            config={"recursion_limit": 8},
        )

        # Print the conversation
        print("\n  📜 Agent conversation:")
        for msg in final_state["messages"]:
            role = msg.__class__.__name__.replace("Message", "")
            content = str(msg.content)[:200]
            if content.strip():
                print(f"    [{role}] {content}")

    finally:
        await manager.disconnect()


# ================================================================
# PART 4: ADK Integration Pattern (Conceptual)
# ================================================================

print("\n" + "=" * 70)
print("PART 4: ADK Integration Pattern")
print("=" * 70)
print("""
For Google ADK, the pattern is similar but uses ADK's FunctionTool:

  1. Connect to MCP server → discover tools
  2. For each MCP tool, create a Python function with matching signature
  3. Wrap each function with google.adk.tools.FunctionTool
  4. Pass the tools to LlmAgent(tools=[...])

The key difference: ADK uses plain Python functions (not LangChain tools),
so the adapter converts MCP tool metadata → Python functions → FunctionTool.

Here's the conceptual pattern (simplified):
""")


# Conceptual ADK adapter (doesn't require ADK installed)
@dataclass
class ADKMCPAdapter:
    """Conceptual adapter showing how MCP tools become ADK tools.

    In real ADK code, you'd import:
      from google.adk.agents import LlmAgent
      from google.adk.tools import FunctionTool
    """

    mcp_tools: list[MCPToolInfo] = field(default_factory=list)

    def to_adk_tools(self):
        """Convert MCP tools to ADK FunctionTool format."""
        adk_tools = []
        for mcp_tool in self.mcp_tools:
            # Each MCP tool becomes a Python function
            tool_spec = {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.parameters,
                # In real code: FunctionTool(func=wrapper_function)
            }
            adk_tools.append(tool_spec)
            print(f"  🔧 ADK tool: {tool_spec['name']}")
            print(f"     Description: {tool_spec['description'][:60]}")
        return adk_tools


print("\nADK adapter converts MCP tools like this:")
sample_tools = [
    MCPToolInfo("get_weather", "Get current weather for a city", {
        "properties": {"city": {"type": "string"}}, "required": ["city"]
    }),
    MCPToolInfo("search_research", "Search past research by keyword", {
        "properties": {"query": {"type": "string"}}, "required": ["query"]
    }),
]
adapter = ADKMCPAdapter(mcp_tools=sample_tools)
adapter.to_adk_tools()


# ================================================================
# PART 5: Key Takeaways
# ================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. MCP tools can be wrapped as LangChain/ADK tools with a simple adapter
2. The agent doesn't know it's using MCP — it just sees normal tools
3. This means you can swap MCP servers without changing agent code
4. The adapter pattern works for ANY MCP server, not just specific ones
5. Phoenix tracing can be added at the adapter boundary for observability

Architecture:

  ┌─────────────────────────────────────────┐
  │            Your Agent (LangGraph/ADK)    │
  │  ┌──────────────────────────────────┐   │
  │  │     LLM with bound tools         │   │
  │  └──────────┬───────────────────────┘   │
  │             │ calls tools               │
  │  ┌──────────▼───────────────────────┐   │
  │  │   MCP Tool Adapter               │   │
  │  │   (converts tool calls to MCP)    │   │
  │  └──────────┬───────────────────────┘   │
  └─────────────┼───────────────────────────┘
                │ stdio / SSE
  ┌─────────────▼───────────────────────┐
  │   MCP Server (weather, research,    │
  │   database, code exec, etc.)        │
  └─────────────────────────────────────┘
""")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LIVE DEMO: LangGraph Agent + MCP Research History Server")
    print("=" * 70)

    try:
        asyncio.run(demo_langgraph_mcp())
    except KeyboardInterrupt:
        print("\n  ⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n  ⚠️  Demo error: {e}")
        print("  This is expected if dependencies aren't installed.")
        print("  Install with: pip install mcp langchain-groq langgraph")

    print("\n✅ Example 05 complete!")
