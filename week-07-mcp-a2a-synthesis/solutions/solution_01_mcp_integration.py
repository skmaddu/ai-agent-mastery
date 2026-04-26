import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 1: MCP Client Integration with LLM Tool Selection
=============================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 2 hours

Complete solution for Exercise 1.  Demonstrates:
  - Connecting to an MCP server via stdio transport
  - Discovering available tools at runtime
  - Using an LLM to select the right tool for a user query
  - Calling MCP tools and displaying results
  - Basic cost tracking throughout the session

Run: python week-07-mcp-a2a-synthesis/solutions/solution_01_mcp_integration.py
"""

import os
import sys
import json
import asyncio
import time
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# LLM Setup
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
# Cost Tracker (simple version)
# ================================================================

class SimpleCostTracker:
    """Track LLM calls and estimated costs."""

    def __init__(self):
        self.calls = []
        self.total_tokens = 0

    def log_call(self, description: str, tokens: int = 0):
        self.calls.append({"description": description, "tokens": tokens, "time": time.time()})
        self.total_tokens += tokens

    def report(self):
        print(f"\n{'=' * 50}")
        print(f"Cost Report:")
        print(f"  Total calls: {len(self.calls)}")
        print(f"  Total tokens (est.): {self.total_tokens}")
        for call in self.calls:
            print(f"  - {call['description']} ({call['tokens']} tokens)")
        print(f"{'=' * 50}")


cost_tracker = SimpleCostTracker()


# ================================================================
# TODO 1: Define MCP Server Parameters (COMPLETED)
# ================================================================

from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
try:
    from mcp.client.stdio import StdioServerParameters
except ImportError:
    from mcp import StdioServerParameters

SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "mcp-servers", "research_history_server.py"
)

server_params = StdioServerParameters(
    command="python",
    args=[SERVER_PATH],
)


# ================================================================
# TODO 2: Implement connect_to_server() (COMPLETED)
# ================================================================

async def connect_to_server():
    """Connect to the MCP server and return the session.

    Returns:
        tuple: (session, client_ctx, session_ctx) for cleanup
    """
    # Open the stdio transport to the MCP server process
    client_ctx = stdio_client(server_params)
    read, write = await client_ctx.__aenter__()

    # Create a ClientSession over the transport
    session_ctx = ClientSession(read, write)
    session = await session_ctx.__aenter__()

    # Perform the MCP capability handshake
    await session.initialize()

    print("  Connected to MCP server successfully!")
    return session, client_ctx, session_ctx


# ================================================================
# TODO 3: Implement discover_tools() (COMPLETED)
# ================================================================

async def discover_tools(session) -> list[dict]:
    """Discover available tools from the MCP server.

    Args:
        session: The MCP ClientSession

    Returns:
        List of dicts with 'name', 'description', 'input_schema' for each tool
    """
    tools_result = await session.list_tools()
    tools_info = []

    print("\n  Available MCP Tools:")
    print("  " + "-" * 46)

    for tool in tools_result.tools:
        info = {
            "name": tool.name,
            "description": tool.description or "No description",
            "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
        }
        tools_info.append(info)
        print(f"    - {tool.name}: {info['description'][:80]}")

    print(f"\n  Total tools discovered: {len(tools_info)}")
    cost_tracker.log_call("Tool discovery (MCP list_tools)", tokens=0)
    return tools_info


# ================================================================
# TODO 4: Implement call_mcp_tool() (COMPLETED)
# ================================================================

async def call_mcp_tool(session, tool_name: str, arguments: dict) -> str:
    """Call an MCP tool and return the text result.

    Args:
        session: The MCP ClientSession
        tool_name: Name of the tool to call
        arguments: Dict of arguments to pass to the tool

    Returns:
        The tool's text response
    """
    try:
        result = await session.call_tool(tool_name, arguments)

        # Extract text from the content list
        text_parts = []
        for content in result.content:
            if hasattr(content, "text"):
                text_parts.append(content.text)

        response_text = "\n".join(text_parts) if text_parts else "(no text content)"

        # Log the MCP call
        est_tokens = int(len(response_text.split()) * 1.3)
        cost_tracker.log_call(
            f"MCP call: {tool_name}({json.dumps(arguments)[:60]})",
            tokens=est_tokens,
        )

        return response_text

    except Exception as e:
        return f"Error calling {tool_name}: {e}"


# ================================================================
# TODO 5: Implement llm_select_tool() (COMPLETED)
# ================================================================

def llm_select_tool(user_query: str, available_tools: list[dict]) -> dict:
    """Use the LLM to select the best tool for the user's query.

    Args:
        user_query: The user's natural language query
        available_tools: List of tool info dicts from discover_tools()

    Returns:
        Dict with 'tool_name' and 'arguments' keys
    """
    # Build the tool descriptions for the prompt
    tools_text = ""
    for i, tool in enumerate(available_tools, 1):
        schema = tool.get("input_schema", {})
        params = schema.get("properties", {})
        param_names = list(params.keys())
        required = schema.get("required", [])
        param_desc = ", ".join(
            f"{p} ({'required' if p in required else 'optional'})"
            for p in param_names
        ) or "none"
        tools_text += f"  {i}. {tool['name']}: {tool['description']}\n"
        tools_text += f"     Parameters: {param_desc}\n"

    system_prompt = f"""You are a tool-selection assistant. Given a user query and a list of available tools, select the single best tool and provide the arguments.

Available tools:
{tools_text}

Respond ONLY with a JSON object. No explanation, no markdown, no extra text.
The JSON must have exactly two keys: "tool_name" (string) and "arguments" (object).

Example:
User: "Search for research about climate change"
{{"tool_name": "search_research", "arguments": {{"query": "climate change"}}}}

Example:
User: "Save research about AI: 'Artificial intelligence is transforming industries.'"
{{"tool_name": "save_research", "arguments": {{"topic": "AI", "summary": "Artificial intelligence is transforming industries.", "sources": "[]"}}}}

Example:
User: "Show me the last 5 entries"
{{"tool_name": "list_recent", "arguments": {{"limit": 5}}}}

Example:
User: "How many entries are in the database?"
{{"tool_name": "get_stats", "arguments": {{}}}}
"""

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ])

    response_text = response.content if hasattr(response, "content") else str(response)

    # Estimate tokens for cost tracking
    prompt_text = system_prompt + user_query
    est_tokens = int((len(prompt_text.split()) + len(response_text.split())) * 1.3)
    cost_tracker.log_call(
        f"LLM tool selection for: '{user_query[:50]}...'",
        tokens=est_tokens,
    )

    return parse_llm_response(response_text)


# ================================================================
# TODO 6: Implement parse_llm_response() (COMPLETED)
# ================================================================

def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's tool selection response.

    Args:
        response_text: Raw text from the LLM

    Returns:
        Dict with 'tool_name' (str) and 'arguments' (dict)
    """
    text = response_text.strip()

    # Strip markdown code block markers if present
    # Handles ```json ... ``` and ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)

        # Validate required keys
        if "tool_name" not in parsed:
            print("    [!] LLM response missing 'tool_name', defaulting to get_stats")
            return {"tool_name": "get_stats", "arguments": {}}

        if "arguments" not in parsed:
            parsed["arguments"] = {}

        # Ensure arguments is a dict
        if not isinstance(parsed["arguments"], dict):
            parsed["arguments"] = {}

        return parsed

    except json.JSONDecodeError as e:
        print(f"    [!] Failed to parse LLM response as JSON: {e}")
        print(f"    [!] Raw response: {response_text[:200]}")
        return {"tool_name": "get_stats", "arguments": {}}


# ================================================================
# TODO 7: Implement the main interaction loop (COMPLETED)
# ================================================================

SAMPLE_QUERIES = [
    "Save research about quantum computing: 'Quantum computers use qubits to perform parallel computations. IBM and Google are leading the race to quantum advantage.'",
    "Search for any research we have about quantum computing",
    "Show me the most recent 3 research entries",
    "What are the statistics of our research database?",
]


async def main():
    """Main interaction loop -- connects, discovers, and processes queries."""
    print("=" * 70)
    print("MCP Client with LLM Tool Selection")
    print("=" * 70)

    # 1. Connect to MCP server
    print("\n[Step 1] Connecting to MCP server...")
    session, client_ctx, session_ctx = await connect_to_server()

    try:
        # 2. Discover tools
        print("\n[Step 2] Discovering available tools...")
        available_tools = await discover_tools(session)

        # 3. Process each sample query
        print("\n[Step 3] Processing sample queries...\n")
        for i, query in enumerate(SAMPLE_QUERIES, 1):
            print(f"{'=' * 70}")
            print(f"  Query {i}: {query}")
            print(f"{'=' * 70}")

            # a. Ask LLM to select the best tool
            print(f"\n  >> Asking LLM to select a tool...")
            selection = llm_select_tool(query, available_tools)
            print(f"  >> LLM selected: {selection['tool_name']}")
            print(f"  >> Arguments: {json.dumps(selection['arguments'], indent=2)}")

            # b. Call the selected MCP tool
            print(f"\n  >> Calling MCP tool: {selection['tool_name']}...")
            result = await call_mcp_tool(session, selection["tool_name"], selection["arguments"])

            # c. Print the result
            print(f"\n  Result:\n{result}")
            print()

    finally:
        # 4. Clean up the connection
        print("\n[Step 4] Cleaning up connection...")
        await session_ctx.__aexit__(None, None, None)
        await client_ctx.__aexit__(None, None, None)
        print("  Connection closed.")

    # 5. Print cost report (TODO 8)
    cost_tracker.report()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("\nSolution 1: MCP Client with LLM Tool Selection")
    print("Connects to research_history_server via MCP and uses LLM to route queries.\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Interrupted")
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Make sure you've installed: pip install mcp langchain-groq")
