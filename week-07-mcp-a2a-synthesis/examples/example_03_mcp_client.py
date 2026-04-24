import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3: Building an MCP Client — The Universal Remote for AI Tools
======================================================================
A REAL, functional MCP client that connects to any MCP server via stdio.

WHY THIS MATTERS (Feynman Explanation):
  Think of an MCP client as a UNIVERSAL REMOTE CONTROL. Your TV remote
  only works with your TV. But a universal remote can talk to ANY device
  — it first asks "what buttons do you have?" (list_tools), then presses
  the right button (call_tool). The MCP client does exactly this: it
  connects to ANY MCP server, discovers what tools it offers, and calls
  them — without knowing the server's internals in advance.

  The magic is the PROTOCOL. Just like USB lets any device talk to any
  computer, MCP lets any AI client talk to any tool server. The client
  doesn't need custom code for each server — it uses the same handshake
  every time: connect → initialize → list_tools → call_tool.

What this example demonstrates:
  1. Launch an MCP server (weather_server.py) as a subprocess via stdio
  2. Perform the capability handshake (initialize)
  3. Discover available tools dynamically (list_tools)
  4. Call tools directly: get_weather, compare_weather
  5. LLM-driven tool selection: give the LLM the tool list, let it decide

Transport: stdio — the client spawns the server as a child process and
communicates via stdin/stdout using JSON-RPC messages. No HTTP, no ports,
no network config. Just pipes.

Run: python week-07-mcp-a2a-synthesis/examples/example_03_mcp_client.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

import asyncio
import json
from typing import Any

# ================================================================
# MCP Client Imports
# ================================================================
# The `mcp` library provides both server AND client components.
# - Server side: FastMCP (we used this in weather_server.py)
# - Client side: stdio_client + ClientSession
#
# stdio_client handles the subprocess plumbing (spawn, pipe stdin/stdout).
# ClientSession handles the JSON-RPC protocol (initialize, list_tools, call_tool).

from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

# StdioServerParameters tells the client HOW to launch the server.
# The import location varies between mcp library versions, so we
# use a try/except pattern for compatibility.
try:
    from mcp.client.stdio import StdioServerParameters
except ImportError:
    from mcp import StdioServerParameters

# ================================================================
# Phoenix Tracing (optional — works without it)
# ================================================================
# Phoenix gives you a visual trace of every MCP call, which is
# invaluable for debugging tool-call chains in production.

try:
    from config.phoenix_config import setup_tracing
    setup_tracing()
    PHOENIX_AVAILABLE = True
except Exception:
    PHOENIX_AVAILABLE = False

# ================================================================
# LLM Setup
# ================================================================

def get_llm(temperature=0.3):
    """Create LLM based on provider setting.

    WHY temperature=0.3?
    For tool selection, we want the LLM to be fairly deterministic —
    it should pick the RIGHT tool, not a creative one. Low temperature
    keeps it focused.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )


# ================================================================
# PART 1: DIRECT MCP CLIENT — Connect, Discover, Call
# ================================================================
# This is the fundamental MCP client pattern. Every MCP interaction
# follows the same three steps:
#
#   1. CONNECT — Launch the server subprocess, establish stdio pipes
#   2. INITIALIZE — Capability handshake ("I speak MCP v1, you?")
#   3. USE — list_tools, call_tool, etc.
#
# It's like a phone call: dial (connect), say hello (initialize),
# then have the conversation (use tools).

async def run_direct_client():
    """Connect to the weather server and call tools directly."""

    print("=" * 70)
    print("PART 1: DIRECT MCP CLIENT")
    print("=" * 70)

    # ── Step 1: Define how to launch the server ──────────────────
    # StdioServerParameters tells the client:
    #   - command: what executable to run (python)
    #   - args: script path (our weather server)
    #
    # The client will spawn this as a subprocess and connect via pipes.
    # This is the SIMPLEST transport — no ports, no URLs, just a process.

    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mcp-servers", "weather_server.py"
    )
    server_script = os.path.normpath(server_script)

    if not os.path.exists(server_script):
        print(f"\n  ERROR: Weather server not found at: {server_script}")
        print("  Make sure weather_server.py exists in week-07-mcp-a2a-synthesis/mcp-servers/")
        return None

    print(f"\n  Server script: {server_script}")

    server_params = StdioServerParameters(
        command=sys.executable,  # Use the same Python that's running us
        args=[server_script],
    )

    # ── Step 2: Connect and initialize ───────────────────────────
    # stdio_client() is an async context manager that:
    #   a) Spawns the subprocess
    #   b) Gives us (read_stream, write_stream) for communication
    #
    # ClientSession wraps those streams with the MCP protocol layer:
    #   - session.initialize() does the capability handshake
    #   - session.list_tools() discovers available tools
    #   - session.call_tool() invokes a specific tool

    print("\n  Connecting to weather server via stdio...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Initialize: exchange capabilities with the server
            # This is like the TCP handshake but for MCP — both sides
            # agree on protocol version and supported features.
            await session.initialize()
            print("  Handshake complete! Connection established.")

            # ── Step 3: Discover tools ───────────────────────────
            # list_tools() returns everything the server offers.
            # This is the "universal remote" magic — we didn't hardcode
            # any tool names. We DISCOVER them at runtime.

            print("\n  " + "-" * 60)
            print("  DISCOVERING TOOLS (list_tools)")
            print("  " + "-" * 60)

            tools_result = await session.list_tools()
            tools = tools_result.tools

            print(f"\n  Found {len(tools)} tools:\n")

            tool_info = []
            for tool in tools:
                # Each tool has: name, description, inputSchema (JSON Schema)
                schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                params = schema.get("properties", {})
                required = schema.get("required", [])

                param_strs = []
                for pname, pinfo in params.items():
                    req_marker = " (required)" if pname in required else " (optional)"
                    param_strs.append(f"{pname}: {pinfo.get('type', '?')}{req_marker}")

                tool_info.append({
                    "name": tool.name,
                    "description": tool.description or "No description",
                    "params": params,
                    "required": required,
                })

                print(f"    {tool.name}")
                print(f"      Description: {tool.description or 'N/A'}")
                if param_strs:
                    print(f"      Parameters:  {', '.join(param_strs)}")
                print()

            # ── Step 4: Call get_weather directly ────────────────
            print("  " + "-" * 60)
            print("  CALLING TOOLS DIRECTLY")
            print("  " + "-" * 60)

            print("\n  Calling get_weather('London')...")
            result = await session.call_tool("get_weather", {"city": "London"})

            # The result contains a list of content blocks.
            # For text tools, there's typically one TextContent block.
            if result.content:
                for block in result.content:
                    text = block.text if hasattr(block, 'text') else str(block)
                    print(f"\n{text}")
            else:
                print("  (No content returned)")

            # ── Step 5: Call compare_weather ──────────────────────
            print("\n\n  Calling compare_weather(['Tokyo', 'Sydney', 'Berlin'])...")
            result = await session.call_tool(
                "compare_weather",
                {"cities": ["Tokyo", "Sydney", "Berlin"]}
            )

            if result.content:
                for block in result.content:
                    text = block.text if hasattr(block, 'text') else str(block)
                    print(f"\n{text}")

            print("\n  " + "-" * 60)
            print("  Direct client demo complete!")
            print("  " + "-" * 60)

            return tool_info


# ================================================================
# PART 2: LLM-DRIVEN TOOL SELECTION
# ================================================================
# In Part 1, WE decided which tool to call. But the real power of
# MCP + LLMs is letting the LLM decide. Here's the flow:
#
#   1. Get tool list from MCP server (same as Part 1)
#   2. Format tool descriptions for the LLM
#   3. Give the LLM a user query + tool descriptions
#   4. LLM outputs: which tool to call + arguments
#   5. We call that tool via MCP and return the result
#
# This is EXACTLY how ChatGPT's function calling works — but with
# MCP, the tools come from an external server instead of being
# hardcoded in the application.

async def run_llm_driven_client():
    """Let the LLM choose which MCP tool to call based on a user query."""

    print("\n" + "=" * 70)
    print("PART 2: LLM-DRIVEN TOOL SELECTION")
    print("=" * 70)

    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mcp-servers", "weather_server.py"
    )
    server_script = os.path.normpath(server_script)

    if not os.path.exists(server_script):
        print(f"\n  ERROR: Weather server not found at: {server_script}")
        return

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("\n  Connected to weather server.")

            # ── Get available tools ──────────────────────────────
            tools_result = await session.list_tools()
            tools = tools_result.tools

            # ── Format tools for the LLM ─────────────────────────
            # We convert MCP tool schemas into a text description the
            # LLM can understand. This is the bridge between MCP's
            # machine-readable format and the LLM's natural language.

            tool_descriptions = []
            for tool in tools:
                schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                params = schema.get("properties", {})
                required = schema.get("required", [])

                param_lines = []
                for pname, pinfo in params.items():
                    req = "required" if pname in required else "optional"
                    ptype = pinfo.get("type", "string")
                    desc = pinfo.get("description", "")
                    param_lines.append(f"    - {pname} ({ptype}, {req}): {desc}")

                tool_desc = (
                    f"Tool: {tool.name}\n"
                    f"  Description: {tool.description or 'N/A'}\n"
                    f"  Parameters:\n" + "\n".join(param_lines)
                )
                tool_descriptions.append(tool_desc)

            tools_text = "\n\n".join(tool_descriptions)

            # ── User queries to process ──────────────────────────
            queries = [
                "What's the weather like in Paris right now?",
                "Compare the weather in New York, London, and Tokyo",
                "Will it rain in Mumbai in the next 3 days?",
            ]

            try:
                llm = get_llm(temperature=0.1)
            except Exception as e:
                print(f"\n  Could not initialize LLM: {e}")
                print("  Skipping LLM-driven demo (set LLM_PROVIDER and API keys in config/.env)")
                return

            for query in queries:
                print(f"\n  {'=' * 55}")
                print(f"  USER QUERY: {query}")
                print(f"  {'=' * 55}")

                # ── Ask LLM to pick a tool ───────────────────────
                # We give the LLM:
                #   1. The available tools (from MCP)
                #   2. The user's question
                #   3. Instructions to output JSON with tool_name + arguments
                #
                # WHY JSON output? We need to PARSE the LLM's decision
                # programmatically. Natural language is ambiguous; JSON isn't.

                system_prompt = f"""You are a tool-calling assistant. You have access to these tools:

{tools_text}

Given a user query, decide which tool to call and with what arguments.
Respond ONLY with a JSON object (no markdown, no explanation):
{{
    "tool_name": "the_tool_to_call",
    "arguments": {{"param1": "value1", "param2": "value2"}}
}}

Pick the BEST tool for the query. Use exact parameter names from the tool schema."""

                try:
                    from langchain_core.messages import HumanMessage, SystemMessage
                    response = llm.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=query),
                    ])

                    # Parse LLM's tool selection
                    response_text = response.content.strip()
                    # Strip markdown code fences if present
                    if response_text.startswith("```"):
                        lines = response_text.split("\n")
                        response_text = "\n".join(
                            l for l in lines if not l.strip().startswith("```")
                        )

                    tool_call = json.loads(response_text)
                    tool_name = tool_call["tool_name"]
                    arguments = tool_call["arguments"]

                    print(f"  LLM chose: {tool_name}")
                    print(f"  Arguments: {json.dumps(arguments, indent=4)}")

                    # ── Execute the tool via MCP ─────────────────
                    print(f"\n  Calling {tool_name} via MCP...")
                    result = await session.call_tool(tool_name, arguments)

                    if result.content:
                        for block in result.content:
                            text = block.text if hasattr(block, 'text') else str(block)
                            print(f"\n{text}")
                    else:
                        print("  (No content returned)")

                except json.JSONDecodeError:
                    print(f"  LLM returned unparseable response: {response_text[:200]}")
                except KeyError as e:
                    print(f"  LLM response missing key: {e}")
                except Exception as e:
                    print(f"  Error during LLM tool call: {e}")

    print("\n  LLM-driven demo complete!")


# ================================================================
# PART 3: ERROR HANDLING PATTERNS
# ================================================================
# Production MCP clients need to handle:
#   - Server not found (bad path)
#   - Server crashes mid-call
#   - Invalid tool names
#   - Malformed arguments
#
# WHY THIS MATTERS: In production, MCP servers are often on different
# machines or managed by different teams. Things WILL break. Your
# client needs to fail gracefully.

async def demo_error_handling():
    """Demonstrate graceful error handling for common MCP failures."""

    print("\n" + "=" * 70)
    print("PART 3: ERROR HANDLING PATTERNS")
    print("=" * 70)

    # ── Case 1: Server not found ─────────────────────────────────
    print("\n  Case 1: Server not found")
    try:
        bad_params = StdioServerParameters(
            command=sys.executable,
            args=["nonexistent_server_12345.py"],
        )
        async with stdio_client(bad_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
    except Exception as e:
        error_type = type(e).__name__
        print(f"  Caught {error_type}: {str(e)[:100]}")
        print("  LESSON: Always check server path exists before connecting.\n")

    # ── Case 2: Invalid tool name ────────────────────────────────
    print("  Case 2: Invalid tool name")
    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mcp-servers", "weather_server.py"
    )
    server_script = os.path.normpath(server_script)

    if os.path.exists(server_script):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
        )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Try calling a tool that doesn't exist
                    result = await session.call_tool("nonexistent_tool", {"x": 1})
                    print(f"  Result: {result}")
        except Exception as e:
            error_type = type(e).__name__
            print(f"  Caught {error_type}: {str(e)[:100]}")
            print("  LESSON: Validate tool names against list_tools() before calling.\n")
    else:
        print("  (Skipped — weather server not found)\n")

    # ── Case 3: Malformed arguments ──────────────────────────────
    print("  Case 3: Malformed arguments")
    if os.path.exists(server_script):
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Call with wrong argument types
                    result = await session.call_tool("get_weather", {})
                    if result.content:
                        for block in result.content:
                            text = block.text if hasattr(block, 'text') else str(block)
                            if "error" in text.lower():
                                print(f"  Server returned error: {text[:100]}")
                            else:
                                print(f"  Unexpected success: {text[:100]}")
                    print("  LESSON: Validate arguments match inputSchema before calling.\n")
        except Exception as e:
            error_type = type(e).__name__
            print(f"  Caught {error_type}: {str(e)[:100]}")
            print("  LESSON: Validate arguments match inputSchema before calling.\n")
    else:
        print("  (Skipped — weather server not found)\n")

    print("  Error handling demo complete!")


# ================================================================
# MAIN: Run all demos
# ================================================================

async def main():
    """Run all MCP client demonstrations."""

    print("\n" + "=" * 70)
    print("  WEEK 7, EXAMPLE 3: MCP CLIENT — THE UNIVERSAL REMOTE")
    print("=" * 70)

    if PHOENIX_AVAILABLE:
        print("\n  Phoenix tracing: ENABLED (check localhost:6006)")
    else:
        print("\n  Phoenix tracing: not available (install arize-phoenix for traces)")

    # Part 1: Direct tool calls
    print()
    try:
        tool_info = await run_direct_client()
    except Exception as e:
        print(f"\n  Part 1 failed: {type(e).__name__}: {e}")
        tool_info = None

    # Part 2: LLM-driven tool selection
    try:
        await run_llm_driven_client()
    except Exception as e:
        print(f"\n  Part 2 failed: {type(e).__name__}: {e}")

    # Part 3: Error handling patterns
    try:
        await demo_error_handling()
    except Exception as e:
        print(f"\n  Part 3 failed: {type(e).__name__}: {e}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: WHAT YOU LEARNED")
    print("=" * 70)
    print("""
  1. MCP CLIENT PATTERN: connect → initialize → list_tools → call_tool
     This same pattern works with ANY MCP server, regardless of what
     tools it provides. That's the power of a standard protocol.

  2. STDIO TRANSPORT: The client spawns the server as a subprocess.
     Communication happens over stdin/stdout pipes using JSON-RPC.
     No HTTP servers, no ports, no CORS — just processes and pipes.

  3. LLM-DRIVEN TOOL SELECTION: Give the LLM the tool descriptions
     from list_tools(), and it picks the right tool + arguments.
     This is how you build truly dynamic agents that adapt to whatever
     tools are available.

  4. ERROR HANDLING: Always validate before calling — check that the
     server exists, the tool name is valid, and arguments match the
     schema. In production, add timeouts and circuit breakers.

  NEXT: Example 4 — Build your own MCP server with SQLite persistence
""")


if __name__ == "__main__":
    asyncio.run(main())
