import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 1: MCP Client Integration with LLM Tool Selection
=============================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 2 hours

Task:
Build an MCP client that connects to the Research History MCP server,
discovers available tools, and lets an LLM decide which tools to call
based on user queries.

The system should:
  1. Connect to the research_history_server.py via stdio transport
  2. Discover all available tools from the server
  3. Present a user query to the LLM along with available tools
  4. Parse the LLM's tool selection and call the MCP tool
  5. Return the formatted result to the user
  6. Track costs throughout the session

Graph:
  User Query → LLM (selects tool) → MCP Client → MCP Server → Result

Instructions:
  Complete the 8 TODOs below to build a working MCP client with
  LLM-driven tool selection.

Hints:
  - Study example_03_mcp_client.py for MCP client connection patterns
  - Study example_04_mcp_server_sql.py for understanding the server tools
  - The MCP server exposes: save_research, search_research, list_recent, get_stats
  - Use asyncio.run() to run the async main function
  - The LLM should output JSON with "tool_name" and "arguments" keys

Run: python week-07-mcp-a2a-synthesis/exercises/exercise_01_mcp_integration.py
"""

import os
import sys
import json
import asyncio
import time
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
        print(f"\n💰 Cost Report:")
        print(f"  Total LLM calls: {len(self.calls)}")
        print(f"  Total tokens (est.): {self.total_tokens}")
        for call in self.calls:
            print(f"  • {call['description']} ({call['tokens']} tokens)")


cost_tracker = SimpleCostTracker()


# ================================================================
# TODO 1: Define MCP Server Parameters
# ================================================================
# Create the server parameters for connecting to the research_history_server.
# The server is at: ../mcp-servers/research_history_server.py (relative to this file)
#
# Hint: Use StdioServerParameters with command="python" and
#       args=[path_to_server]

# --- YOUR CODE HERE ---
# Import the necessary MCP client modules:
#   from mcp.client.stdio import stdio_client
#   from mcp.client.session import ClientSession
#   try:
#       from mcp.client.stdio import StdioServerParameters
#   except ImportError:
#       from mcp import StdioServerParameters
#
# SERVER_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "..", "mcp-servers", "research_history_server.py"
# )
#
# server_params = StdioServerParameters(
#     command="python",
#     args=[SERVER_PATH],
# )
# --- END YOUR CODE ---


# ================================================================
# TODO 2: Implement connect_to_server()
# ================================================================
# Create an async function that:
#   1. Opens a stdio_client connection with the server params
#   2. Creates a ClientSession
#   3. Initializes the session (capability handshake)
#   4. Returns the session, and context managers for cleanup
#
# Hint: Use the pattern from example_03:
#   async with stdio_client(server_params) as (read, write):
#       async with ClientSession(read, write) as session:
#           await session.initialize()

async def connect_to_server():
    """Connect to the MCP server and return the session.

    Returns:
        tuple: (session, client_ctx, session_ctx) for cleanup
    """
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 3: Implement discover_tools()
# ================================================================
# Create an async function that:
#   1. Calls session.list_tools() to get available tools
#   2. Prints each tool's name and description
#   3. Returns a list of tool info dicts with name, description, and input schema
#
# Hint: tools_result = await session.list_tools()
#       tools_result.tools is a list of Tool objects

async def discover_tools(session) -> list[dict]:
    """Discover available tools from the MCP server.

    Args:
        session: The MCP ClientSession

    Returns:
        List of dicts with 'name', 'description', 'input_schema' for each tool
    """
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 4: Implement call_mcp_tool()
# ================================================================
# Create an async function that:
#   1. Calls session.call_tool(tool_name, arguments)
#   2. Extracts text content from the result
#   3. Returns the text as a string
#   4. Handles errors gracefully
#
# Hint: result = await session.call_tool(name, args)
#       result.content is a list of content objects with .text attribute

async def call_mcp_tool(session, tool_name: str, arguments: dict) -> str:
    """Call an MCP tool and return the text result.

    Args:
        session: The MCP ClientSession
        tool_name: Name of the tool to call
        arguments: Dict of arguments to pass to the tool

    Returns:
        The tool's text response
    """
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 5: Implement llm_select_tool()
# ================================================================
# Create a function that:
#   1. Takes the user query and list of available tools
#   2. Builds a prompt asking the LLM to select the best tool
#   3. The LLM should respond with JSON: {"tool_name": "...", "arguments": {...}}
#   4. Calls the LLM and returns the parsed JSON
#
# Hint: Format the tools as a numbered list in the prompt:
#   "Available tools:
#    1. save_research: Save a research finding... Parameters: topic, summary, sources
#    2. search_research: Search past research... Parameters: query, max_results
#    ..."
#
# The system prompt should instruct the LLM to ONLY respond with JSON.

def llm_select_tool(user_query: str, available_tools: list[dict]) -> dict:
    """Use the LLM to select the best tool for the user's query.

    Args:
        user_query: The user's natural language query
        available_tools: List of tool info dicts from discover_tools()

    Returns:
        Dict with 'tool_name' and 'arguments' keys
    """
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 6: Implement parse_llm_response()
# ================================================================
# Create a function that:
#   1. Takes the raw LLM response string
#   2. Extracts JSON from it (the LLM might include markdown code blocks)
#   3. Parses the JSON into a dict
#   4. Validates it has 'tool_name' and 'arguments' keys
#   5. Returns the parsed dict, or a sensible default on failure
#
# Hint: The LLM might respond with:
#   ```json
#   {"tool_name": "search_research", "arguments": {"query": "AI"}}
#   ```
#   You need to strip the markdown code block markers.

def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's tool selection response.

    Args:
        response_text: Raw text from the LLM

    Returns:
        Dict with 'tool_name' (str) and 'arguments' (dict)
    """
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 7: Implement the main interaction loop
# ================================================================
# Create an async function that:
#   1. Connects to the MCP server (TODO 2)
#   2. Discovers available tools (TODO 3)
#   3. Runs a loop with sample queries (provided below)
#   4. For each query: LLM selects tool (TODO 5) → call MCP tool (TODO 4)
#   5. Prints the results
#   6. Cleans up the connection when done
#
# Sample queries to process:
SAMPLE_QUERIES = [
    "Save research about quantum computing: 'Quantum computers use qubits to perform parallel computations. IBM and Google are leading the race to quantum advantage.'",
    "Search for any research we have about quantum computing",
    "Show me the most recent 3 research entries",
    "What are the statistics of our research database?",
]

async def main():
    """Main interaction loop — connects, discovers, and processes queries."""
    print("=" * 70)
    print("MCP Client with LLM Tool Selection")
    print("=" * 70)

    # --- YOUR CODE HERE ---
    # 1. Connect to MCP server
    # 2. Discover tools
    # 3. Loop through SAMPLE_QUERIES:
    #    a. Print the query
    #    b. Ask LLM to select a tool
    #    c. Call the selected MCP tool
    #    d. Print the result
    # 4. Clean up
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 8: Add cost tracking
# ================================================================
# Integrate the SimpleCostTracker:
#   1. Log each LLM call in llm_select_tool() with cost_tracker.log_call()
#   2. Log each MCP tool call in call_mcp_tool()
#   3. Print the cost report at the end of main()
#
# Hint: Estimate tokens as len(text.split()) * 1.3 (rough approximation)

# --- YOUR CODE HERE ---
# Add cost_tracker.log_call() calls in the appropriate functions above
# Add cost_tracker.report() at the end of main()
# --- END YOUR CODE ---


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("\n⚠️  Exercise 1: Complete the 8 TODOs to build a working MCP client!")
    print("    Study example_03 and example_04 for reference.\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  ⏹️  Interrupted")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        print("  Make sure you've completed all TODOs and installed: pip install mcp langchain-groq")
