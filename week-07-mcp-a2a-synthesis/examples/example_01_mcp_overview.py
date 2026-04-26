import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Example 01 — Model Context Protocol (MCP): Overview
=====================================================

This example teaches MCP from first principles using a fully simulated
message exchange. No real servers or LLM calls are needed — everything
runs in pure Python so you can study the protocol mechanics directly.

Key takeaway:
    MCP is like a USB port for AI agents — any tool that speaks the
    protocol can plug in. Just as USB freed you from needing a different
    cable for every peripheral, MCP frees AI applications from needing
    a custom adapter for every external tool.

What you will learn:
    1. What a protocol is and why AI agents need one
    2. The JSON-RPC 2.0 message format that MCP uses
    3. The full lifecycle: initialize → discover tools → call tools
    4. Client vs server roles
    5. Transport types (stdio, SSE, Streamable HTTP)
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import json

# ================================================================
# SECTION 1 — First Principles: What Is a Protocol?
# ================================================================
#
# A protocol is simply a set of rules two parties agree to follow
# so they can communicate without confusion.
#
# Real-world analogy:
#   When you call a restaurant to order food, there is an implicit
#   protocol: you greet them, they ask what you'd like, you state
#   your order, they confirm and give a total, you say thanks.
#   Both sides know the steps. That shared understanding IS the
#   protocol.
#
# In software, protocols specify:
#   • Message format  (what does a request look like?)
#   • Message flow    (who speaks first? what comes next?)
#   • Error handling  (what if something goes wrong?)
#
# MCP defines all three for the specific problem of
# "an AI agent wants to use an external tool."

# ================================================================
# SECTION 2 — Why Do AI Agents Need a Standard Protocol?
# ================================================================
#
# Without MCP, every AI application has to write custom glue code
# for every tool it wants to use:
#
#   App A ──custom code──▶ Tool 1
#   App A ──custom code──▶ Tool 2
#   App B ──custom code──▶ Tool 1   (duplicated effort!)
#   App B ──custom code──▶ Tool 2
#
# With MCP, each side implements the protocol once:
#
#   App A ──MCP──▶ ┌─────────┐ ──MCP──▶ Tool 1
#   App B ──MCP──▶ │ standard│ ──MCP──▶ Tool 2
#                  │ protocol│ ──MCP──▶ Tool 3
#                  └─────────┘
#
# This is the same insight that made USB successful: instead of
# N devices × M peripherals = N×M cables, you get N+M adapters.

# ================================================================
# SECTION 3 — Architecture Overview (ASCII Diagram)
# ================================================================
#
#  ┌──────────────────────┐          ┌──────────────────────┐
#  │     MCP CLIENT       │          │     MCP SERVER        │
#  │  (AI app / agent)    │          │  (tool provider)      │
#  │                      │          │                       │
#  │  • Sends requests    │ JSON-RPC │  • Exposes tools      │
#  │  • Receives results  │◄────────►│  • Exposes resources  │
#  │  • Drives the flow   │  over    │  • Exposes prompts    │
#  │                      │ transport│  • Executes actions    │
#  └──────────────────────┘          └──────────────────────┘
#
#  Transport options:
#    1. stdio   — client spawns server as a subprocess,
#                 communicates via stdin/stdout
#    2. SSE     — server runs as HTTP server, streams
#                 responses via Server-Sent Events
#    3. Streamable HTTP — newest option, single HTTP
#                 endpoint, supports bidirectional streaming
#
#  The protocol itself is the SAME regardless of transport.
#  Think of transport as the "cable" and the protocol as the
#  "language spoken over the cable."

# ================================================================
# SECTION 4 — JSON-RPC 2.0: The Message Format
# ================================================================
#
# MCP messages follow JSON-RPC 2.0, which defines three kinds:
#
#   Request:       { "jsonrpc": "2.0", "id": 1, "method": "...", "params": {...} }
#   Response:      { "jsonrpc": "2.0", "id": 1, "result": {...} }
#   Notification:  { "jsonrpc": "2.0", "method": "...", "params": {...} }
#                  (no "id" — no response expected)
#
# The "id" links a response back to its request.
# This is the same format used by Language Server Protocol (LSP)
# for code editors — MCP borrowed a proven design.


@dataclass
class JsonRpcRequest:
    """A JSON-RPC 2.0 request message."""
    method: str
    id: int
    params: dict = field(default_factory=dict)
    jsonrpc: str = "2.0"


@dataclass
class JsonRpcResponse:
    """A JSON-RPC 2.0 response message."""
    id: int
    result: dict = field(default_factory=dict)
    jsonrpc: str = "2.0"


def pretty(label: str, obj: Any) -> None:
    """Pretty-print a message with a label."""
    if hasattr(obj, '__dataclass_fields__'):
        obj = asdict(obj)
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# ================================================================
# SECTION 5 — Simulated MCP Server
# ================================================================
#
# In real life the server is a separate process. Here we simulate
# it as a Python class so you can see exactly what happens on the
# server side when it receives each message.

class SimulatedMCPServer:
    """
    A simulated MCP server that exposes two tools:
      • get_weather  — returns mock weather data
      • web_search   — returns mock search results

    This is NOT a real MCP server. It demonstrates the protocol
    message flow in pure Python.
    """

    SERVER_INFO = {
        "name": "demo-tool-server",
        "version": "1.0.0",
    }

    CAPABILITIES = {
        "tools": {"listChanged": True},     # server supports tool discovery
        "resources": {"subscribe": False},   # server supports resources (no subscriptions)
        "prompts": {"listChanged": False},   # server supports prompt templates
    }

    TOOLS = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'London'"
                    }
                },
                "required": ["city"]
            }
        },
        {
            "name": "web_search",
            "description": "Search the web for a query",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        },
    ]

    def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """
        Route an incoming request to the correct handler.

        In a real MCP server, the transport layer (stdio/SSE/HTTP)
        would deserialize the JSON, call this handler, then serialize
        the response back. We skip the serialization step here.
        """
        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        handler = handlers.get(request.method)
        if handler is None:
            return JsonRpcResponse(
                id=request.id,
                result={"error": f"Unknown method: {request.method}"}
            )
        return handler(request)

    # ── Step 1: Initialize ──────────────────────────────────────
    def _handle_initialize(self, req: JsonRpcRequest) -> JsonRpcResponse:
        """
        The very first message in any MCP session.

        The client says "hello, here is who I am and what I support."
        The server responds with its own identity and capabilities.

        This is called "capability negotiation" — both sides learn
        what the other can do before any real work begins.
        """
        return JsonRpcResponse(
            id=req.id,
            result={
                "protocolVersion": "2025-03-26",
                "serverInfo": self.SERVER_INFO,
                "capabilities": self.CAPABILITIES,
            }
        )

    # ── Step 2: List Tools ──────────────────────────────────────
    def _handle_tools_list(self, req: JsonRpcRequest) -> JsonRpcResponse:
        """
        The client asks "what tools do you have?"

        The server responds with a list of tool definitions, each
        including a name, description, and a JSON Schema describing
        the expected input. This is how the AI agent learns what
        tools are available without any hardcoding.
        """
        return JsonRpcResponse(
            id=req.id,
            result={"tools": self.TOOLS}
        )

    # ── Step 3: Call a Tool ─────────────────────────────────────
    def _handle_tools_call(self, req: JsonRpcRequest) -> JsonRpcResponse:
        """
        The client says "please run this tool with these arguments."

        The server executes the tool and returns the result.
        In real MCP, the result uses a 'content' array that can
        hold text, images, or other media types.
        """
        tool_name = req.params.get("name", "")
        arguments = req.params.get("arguments", {})

        # Simulated tool implementations
        if tool_name == "get_weather":
            city = arguments.get("city", "Unknown")
            result_text = f"Weather in {city}: 22°C, partly cloudy, humidity 65%"
        elif tool_name == "web_search":
            query = arguments.get("query", "")
            result_text = (
                f"Search results for '{query}':\n"
                f"  1. Introduction to {query} — Wikipedia\n"
                f"  2. Latest developments in {query} — Nature\n"
                f"  3. {query} explained simply — Khan Academy"
            )
        else:
            result_text = f"Error: Unknown tool '{tool_name}'"

        return JsonRpcResponse(
            id=req.id,
            result={
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }
        )


# ================================================================
# SECTION 6 — Simulated MCP Client
# ================================================================
#
# The client is the AI application (or agent framework like
# LangGraph or ADK). It drives the conversation:
#   1. Initialize the session
#   2. Discover available tools
#   3. Call tools as needed

class SimulatedMCPClient:
    """
    A simulated MCP client that connects to a simulated server.

    In real MCP:
      • The client would spawn the server process (stdio transport)
        or connect via HTTP (SSE/Streamable HTTP transport).
      • Messages would be serialized to JSON and sent over the wire.

    Here we call the server directly so you can trace the flow.
    """

    def __init__(self, server: SimulatedMCPServer):
        self.server = server
        self.request_id = 0
        self.available_tools: list[dict] = []

    def _next_id(self) -> int:
        """Each request gets a unique incrementing ID."""
        self.request_id += 1
        return self.request_id

    def initialize(self) -> dict:
        """
        Step 1: Start the session.

        The client sends its info and learns about the server's
        capabilities. This MUST be the first message.
        """
        request = JsonRpcRequest(
            method="initialize",
            id=self._next_id(),
            params={
                "protocolVersion": "2025-03-26",
                "clientInfo": {
                    "name": "demo-ai-agent",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "sampling": {}  # client can provide LLM sampling
                }
            }
        )
        pretty("CLIENT → SERVER  [initialize request]", request)

        response = self.server.handle_request(request)
        pretty("SERVER → CLIENT  [initialize response]", response)
        return response.result

    def list_tools(self) -> list[dict]:
        """
        Step 2: Discover available tools.

        After initialization, the client asks what tools the server
        provides. The response includes JSON Schema for each tool's
        inputs — this is what gets passed to the LLM as the function
        definitions for tool calling.
        """
        request = JsonRpcRequest(
            method="tools/list",
            id=self._next_id(),
        )
        pretty("CLIENT → SERVER  [tools/list request]", request)

        response = self.server.handle_request(request)
        pretty("SERVER → CLIENT  [tools/list response]", response)

        self.available_tools = response.result.get("tools", [])
        return self.available_tools

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Step 3: Execute a tool.

        The client (on behalf of the AI agent) asks the server to
        run a specific tool with specific arguments. The server
        executes it and returns the result.
        """
        request = JsonRpcRequest(
            method="tools/call",
            id=self._next_id(),
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        pretty(f"CLIENT → SERVER  [tools/call '{tool_name}']", request)

        response = self.server.handle_request(request)
        pretty(f"SERVER → CLIENT  [tools/call result]", response)
        return response.result


# ================================================================
# SECTION 7 — Transport Types Explained
# ================================================================

def explain_transports() -> None:
    """
    MCP supports multiple transport types. The protocol messages
    are identical — only the delivery mechanism changes.
    """
    print("\n" + "=" * 60)
    print("  TRANSPORT TYPES")
    print("=" * 60)

    transports = {
        "stdio": {
            "how_it_works": (
                "Client spawns the server as a child process.\n"
                "    Messages flow through stdin (client→server)\n"
                "    and stdout (server→client)."
            ),
            "best_for": "Local tools, CLI integrations, desktop apps",
            "example": "claude desktop → python mcp_server.py",
            "pros": "Simple, no network needed, secure (local only)",
            "cons": "Single client only, no remote access",
        },
        "SSE (Server-Sent Events)": {
            "how_it_works": (
                "Server runs as an HTTP server.\n"
                "    Client sends requests via POST.\n"
                "    Server streams responses via SSE endpoint."
            ),
            "best_for": "Remote servers, shared tool services",
            "example": "agent app → https://tools.example.com/sse",
            "pros": "Works over network, multiple clients",
            "cons": "Two endpoints needed (POST + SSE), being replaced",
        },
        "Streamable HTTP": {
            "how_it_works": (
                "Single HTTP endpoint handles everything.\n"
                "    Requests via POST, responses streamed back.\n"
                "    Supports bidirectional streaming."
            ),
            "best_for": "Production deployments (newest, recommended)",
            "example": "agent app → https://tools.example.com/mcp",
            "pros": "Single endpoint, full streaming, stateless option",
            "cons": "Newest — some SDKs still adding support",
        },
    }

    for name, info in transports.items():
        print(f"\n  ┌─ {name} ─────────────────────────────")
        print(f"  │ How:      {info['how_it_works']}")
        print(f"  │ Best for: {info['best_for']}")
        print(f"  │ Example:  {info['example']}")
        print(f"  │ Pros:     {info['pros']}")
        print(f"  │ Cons:     {info['cons']}")
        print(f"  └{'─' * 50}")


# ================================================================
# SECTION 8 — Client vs Server Roles
# ================================================================

def explain_roles() -> None:
    """Explain who does what in MCP."""
    print("\n" + "=" * 60)
    print("  CLIENT vs SERVER ROLES")
    print("=" * 60)
    print("""
  CLIENT (the AI application)          SERVER (the tool provider)
  ─────────────────────────            ──────────────────────────
  • Initiates the connection           • Waits for connections
  • Sends initialize request           • Responds with capabilities
  • Asks for tool/resource lists       • Provides tool/resource lists
  • Decides WHEN to call a tool        • Executes tool calls
  • Passes results to the LLM          • Returns structured results
  • Manages the conversation           • Stateless between calls (usually)
  • One client can connect to          • One server can serve
    MULTIPLE servers                     MULTIPLE tools

  Examples of clients:                 Examples of servers:
  • Claude Desktop                     • File system MCP server
  • Cursor IDE                         • GitHub MCP server
  • Your LangGraph agent               • Database query server
  • Your ADK agent                     • Web search server
    """)


# ================================================================
# SECTION 9 — Run the Full Simulation
# ================================================================

def main() -> None:
    print("=" * 60)
    print("  MCP OVERVIEW — Simulated Protocol Exchange")
    print("=" * 60)
    print()
    print("  Think of MCP like USB for AI agents:")
    print("  Any tool that speaks the protocol can plug right in.")
    print("  No custom adapters needed.")
    print()
    print("  We will now simulate a complete MCP session:")
    print("  1. Client initializes the connection")
    print("  2. Client discovers available tools")
    print("  3. Client calls a tool and gets results")

    # Create server and client
    server = SimulatedMCPServer()
    client = SimulatedMCPClient(server)

    # ── Step 1: Initialize ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Initialize Session")
    print("=" * 60)
    print("  The client introduces itself and learns what the")
    print("  server supports (capability negotiation).")
    init_result = client.initialize()

    # ── Step 2: Discover Tools ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Discover Available Tools")
    print("=" * 60)
    print("  The client asks the server what tools it offers.")
    print("  Each tool comes with a JSON Schema describing its inputs.")
    print("  This schema is what gets passed to the LLM so it knows")
    print("  how to format tool calls.")
    tools = client.list_tools()

    print(f"\n  ✓ Discovered {len(tools)} tools:")
    for tool in tools:
        print(f"    • {tool['name']}: {tool['description']}")

    # ── Step 3: Call Tools ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Call a Tool")
    print("=" * 60)
    print("  The AI agent decides to call 'get_weather' for London.")
    print("  In a real system, the LLM would generate this tool call")
    print("  based on the user's question and the available tool schemas.")
    result = client.call_tool("get_weather", {"city": "London"})

    print("\n  Now calling 'web_search'...")
    result2 = client.call_tool("web_search", {"query": "Model Context Protocol"})

    # ── Explain transports and roles ────────────────────────────
    explain_transports()
    explain_roles()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY — The MCP Lifecycle")
    print("=" * 60)
    print("""
  1. INITIALIZE    Client and server exchange identities and
                   capabilities (what features each supports).

  2. DISCOVER      Client asks for available tools, resources,
                   and prompts. Gets back schemas it can pass
                   to the LLM for function calling.

  3. USE           Client calls tools/reads resources as needed.
                   Each call is a JSON-RPC request/response pair.

  4. CLOSE         Client closes the connection (not shown here).

  Key insight: The AI agent (LLM) never talks to the MCP server
  directly. The CLIENT application sits in between — it takes the
  LLM's tool-call decisions and translates them into MCP requests.

  ┌─────┐    tool call    ┌────────┐    MCP request    ┌────────┐
  │ LLM │ ──────────────▶ │ Client │ ────────────────▶ │ Server │
  │     │ ◀────────────── │  App   │ ◀──────────────── │        │
  └─────┘    tool result   └────────┘    MCP response   └────────┘
    """)
    print("  Done! You've seen the complete MCP message flow.")


if __name__ == "__main__":
    main()
