import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Example 02 — MCP: Purpose and Core Ideas
==========================================

Before MCP, every AI app needed a custom adapter for every tool —
like needing a different charger for every phone. MCP is the
universal standard that fixes this.

This example teaches:
    1. The N×M integration problem and how MCP solves it
    2. The four core primitives: Resources, Tools, Prompts, Sampling
    3. Transport types and when to use each
    4. Capability negotiation — how client and server discover
       each other's features
    5. Pre-MCP vs Post-MCP comparison

All concepts are demonstrated with runnable Python code using
Pydantic models. No external dependencies beyond stdlib + pydantic.
"""

from dataclasses import dataclass
from typing import Any
from enum import Enum
import json

# We use Pydantic to model MCP primitives — this mirrors how real
# MCP SDKs define their types.
try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Pydantic is required. Install with: pip install pydantic")
    sys.exit(1)


# ================================================================
# SECTION 1 — The N×M Integration Problem
# ================================================================
#
# Feynman explanation:
#   Before MCP, every AI app needed a custom adapter for every tool
#   — like needing a different charger for every phone. If you had
#   3 phones and 4 chargers, you might need up to 12 different
#   cable combinations. MCP is the USB-C of the AI world: one
#   standard plug that works everywhere.
#
# The math:
#   Without MCP:  N agents × M tools = N×M custom integrations
#   With MCP:     N agents + M tools = N+M implementations
#
#   For 5 agents and 10 tools:
#     Without MCP: 5 × 10 = 50 custom integrations
#     With MCP:    5 + 10 = 15 implementations (70% less work!)

def demonstrate_integration_problem() -> None:
    """Show the N×M problem with concrete numbers."""
    print("=" * 60)
    print("  THE N×M INTEGRATION PROBLEM")
    print("=" * 60)

    agents = ["Claude Desktop", "Cursor IDE", "LangGraph App", "ADK Agent", "Custom Bot"]
    tools = [
        "File System", "GitHub", "Slack", "Database",
        "Web Search", "Email", "Calendar", "Jira",
        "AWS S3", "Stripe API"
    ]

    n = len(agents)
    m = len(tools)

    print(f"\n  Agents ({n}): {', '.join(agents)}")
    print(f"  Tools  ({m}): {', '.join(tools)}")

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  WITHOUT MCP (custom adapters)                  │
  │                                                 │
  │  Each agent needs a custom integration for      │
  │  each tool. That's {n} × {m} = {n * m} integrations.     │
  │                                                 │
  │  Agent A ──adapter──▶ Tool 1                    │
  │  Agent A ──adapter──▶ Tool 2                    │
  │  Agent A ──adapter──▶ Tool 3                    │
  │  Agent B ──adapter──▶ Tool 1  (duplicated!)     │
  │  Agent B ──adapter──▶ Tool 2  (duplicated!)     │
  │  ... {n * m} total adapters to build and maintain   │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │  WITH MCP (standard protocol)                   │
  │                                                 │
  │  Each agent implements MCP client once.          │
  │  Each tool implements MCP server once.           │
  │  That's {n} + {m} = {n + m} implementations.             │
  │                                                 │
  │  Agent A ──MCP──┐                               │
  │  Agent B ──MCP──┤     ┌──MCP──▶ Tool 1          │
  │  Agent C ──MCP──┼─────┼──MCP──▶ Tool 2          │
  │  Agent D ──MCP──┤     └──MCP──▶ Tool 3          │
  │  Agent E ──MCP──┘                               │
  │                                                 │
  │  Savings: {n * m} → {n + m} = {((n * m - (n + m)) / (n * m) * 100):.0f}% less work!             │
  └─────────────────────────────────────────────────┘
    """)


# ================================================================
# SECTION 2 — Core Primitive: Resources
# ================================================================
#
# Resources are READ-ONLY data that the server exposes.
# Think of them like files you can look at but not change.
#
# Examples: file contents, database rows, API responses,
#           configuration data, log entries
#
# The key idea: resources let the AI agent ACCESS information
# without taking any action. They are safe and side-effect-free.

class ResourceContent(BaseModel):
    """Content within a resource — can be text or binary."""
    uri: str = Field(description="Unique identifier for this resource")
    mimeType: str = Field(default="text/plain", description="MIME type of content")
    text: str | None = Field(default=None, description="Text content (if text-based)")


class Resource(BaseModel):
    """
    An MCP Resource — read-only data exposed by the server.

    Resources are identified by URIs (like web URLs but for data).
    The URI scheme tells you what kind of resource it is:
      file:///path/to/file     — a file on disk
      db://table/row           — a database row
      api://service/endpoint   — data from an API
    """
    uri: str = Field(description="Resource URI (unique identifier)")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="What this resource contains")
    mimeType: str = Field(default="text/plain", description="Content type")


def demonstrate_resources() -> None:
    """Show how MCP Resources work."""
    print("=" * 60)
    print("  CORE PRIMITIVE 1: Resources (Read-Only Data)")
    print("=" * 60)
    print()
    print("  Resources are like files you can read but not modify.")
    print("  They let the AI agent gather information safely.")
    print()

    # Example resources a file-system MCP server might expose
    resources = [
        Resource(
            uri="file:///project/README.md",
            name="Project README",
            description="Main documentation file for the project",
            mimeType="text/markdown",
        ),
        Resource(
            uri="db://users/active-count",
            name="Active User Count",
            description="Current number of active users in the database",
            mimeType="application/json",
        ),
        Resource(
            uri="api://weather/london",
            name="London Weather",
            description="Current weather conditions in London",
            mimeType="application/json",
        ),
    ]

    for r in resources:
        print(f"  Resource: {r.name}")
        print(f"    URI:         {r.uri}")
        print(f"    MIME type:   {r.mimeType}")
        print(f"    Description: {r.description}")
        print()


# ================================================================
# SECTION 3 — Core Primitive: Tools
# ================================================================
#
# Tools are ACTIONS the server can perform. Unlike resources,
# tools DO have side effects — they change things in the world.
#
# Examples: send an email, create a GitHub issue, run a database
#           query, execute code, call an external API
#
# Each tool has a JSON Schema for its inputs, so the LLM knows
# exactly what arguments to provide.

class ToolParameter(BaseModel):
    """A single parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True


class MCPTool(BaseModel):
    """
    An MCP Tool — an action the server can perform.

    Tools are the most commonly used primitive. They map directly
    to "function calling" in LLM APIs. The server describes
    available tools with JSON Schema, and the LLM generates
    structured calls to them.
    """
    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="What the tool does (shown to LLM)")
    parameters: list[ToolParameter] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Result returned after executing a tool."""
    tool_name: str
    success: bool
    content_type: str = "text"
    content: str = ""


def demonstrate_tools() -> None:
    """Show how MCP Tools work."""
    print("=" * 60)
    print("  CORE PRIMITIVE 2: Tools (Actions)")
    print("=" * 60)
    print()
    print("  Tools let the AI agent DO things — they have side effects.")
    print("  Each tool defines its inputs via JSON Schema so the LLM")
    print("  knows exactly how to call it.")
    print()

    tools = [
        MCPTool(
            name="send_email",
            description="Send an email to a recipient",
            parameters=[
                ToolParameter(name="to", type="string", description="Recipient email address"),
                ToolParameter(name="subject", type="string", description="Email subject line"),
                ToolParameter(name="body", type="string", description="Email body text"),
            ]
        ),
        MCPTool(
            name="create_github_issue",
            description="Create a new issue in a GitHub repository",
            parameters=[
                ToolParameter(name="repo", type="string", description="Repository (owner/name)"),
                ToolParameter(name="title", type="string", description="Issue title"),
                ToolParameter(name="body", type="string", description="Issue description"),
                ToolParameter(name="labels", type="array", description="Labels to apply", required=False),
            ]
        ),
    ]

    for tool in tools:
        print(f"  Tool: {tool.name}")
        print(f"    Description: {tool.description}")
        print(f"    Parameters:")
        for p in tool.parameters:
            req = "required" if p.required else "optional"
            print(f"      • {p.name} ({p.type}, {req}): {p.description}")
        print()

    # Simulate a tool call
    print("  --- Simulated Tool Call ---")
    call = {"tool": "send_email", "arguments": {
        "to": "team@example.com",
        "subject": "Weekly Report",
        "body": "Here are this week's highlights..."
    }}
    result = ToolResult(
        tool_name="send_email",
        success=True,
        content="Email sent successfully to team@example.com"
    )
    print(f"  Call:   {json.dumps(call, indent=2)}")
    print(f"  Result: {result.content}")
    print()


# ================================================================
# SECTION 4 — Core Primitive: Prompts
# ================================================================
#
# Prompts are REUSABLE TEMPLATES that the server provides.
# Think of them as pre-written instructions the AI can use.
#
# Examples: "Summarize this document", "Review this code",
#           "Translate to French"
#
# Prompts are optional — many servers only expose tools. But
# prompts are useful for standardizing common operations.

class PromptArgument(BaseModel):
    """An argument that can be filled into a prompt template."""
    name: str
    description: str
    required: bool = True


class MCPPrompt(BaseModel):
    """
    An MCP Prompt — a reusable prompt template.

    Prompts are like fill-in-the-blank templates. The server
    defines the template with placeholders, and the client fills
    in the values before sending to the LLM. This ensures
    consistent, well-crafted prompts across different applications.
    """
    name: str = Field(description="Prompt identifier")
    description: str = Field(description="What this prompt does")
    arguments: list[PromptArgument] = Field(default_factory=list)


def demonstrate_prompts() -> None:
    """Show how MCP Prompts work."""
    print("=" * 60)
    print("  CORE PRIMITIVE 3: Prompts (Reusable Templates)")
    print("=" * 60)
    print()
    print("  Prompts are like fill-in-the-blank templates.")
    print("  The server defines them, the client fills in values.")
    print()

    prompts = [
        MCPPrompt(
            name="code_review",
            description="Review code for bugs, style, and improvements",
            arguments=[
                PromptArgument(name="code", description="The code to review"),
                PromptArgument(name="language", description="Programming language"),
                PromptArgument(name="focus", description="What to focus on", required=False),
            ]
        ),
        MCPPrompt(
            name="summarize_document",
            description="Create a concise summary of a document",
            arguments=[
                PromptArgument(name="document", description="The text to summarize"),
                PromptArgument(name="max_length", description="Maximum summary length", required=False),
            ]
        ),
    ]

    for prompt in prompts:
        print(f"  Prompt: {prompt.name}")
        print(f"    Description: {prompt.description}")
        print(f"    Arguments:")
        for a in prompt.arguments:
            req = "required" if a.required else "optional"
            print(f"      • {a.name} ({req}): {a.description}")
        print()

    # Show what a filled-in prompt looks like
    print("  --- Filled-In Prompt Example ---")
    print("  Template: code_review")
    print("  Arguments: {code: 'def add(a,b): return a+b', language: 'python'}")
    print("  Result sent to LLM:")
    print("    'Review this python code for bugs, style, and improvements:")
    print("     def add(a,b): return a+b'")
    print()


# ================================================================
# SECTION 5 — Core Primitive: Sampling
# ================================================================
#
# Sampling is the most unusual primitive — it reverses the flow.
# Normally the client calls the server. With sampling, the SERVER
# asks the CLIENT's LLM to generate text.
#
# Why? Sometimes a tool server needs AI help to complete its work.
# For example, a code analysis server might need the LLM to
# interpret results before returning them.
#
# Important: The client (human) must approve sampling requests.
# The server cannot use the LLM without permission.

class SamplingRequest(BaseModel):
    """
    A request from the SERVER to the CLIENT's LLM.

    This is the reverse of normal flow — the server is asking
    the client for help. The client can approve, modify, or
    deny the request.
    """
    messages: list[dict] = Field(description="Messages to send to the LLM")
    model_preferences: dict = Field(
        default_factory=dict,
        description="Hints about what model to use (client decides)"
    )
    max_tokens: int = Field(default=500, description="Maximum tokens in response")
    purpose: str = Field(default="", description="Why the server needs LLM help")


def demonstrate_sampling() -> None:
    """Show how MCP Sampling works."""
    print("=" * 60)
    print("  CORE PRIMITIVE 4: Sampling (Server Asks Client's LLM)")
    print("=" * 60)
    print()
    print("  Normally: Client calls Server (agent uses a tool)")
    print("  Sampling: Server calls Client (tool asks the LLM for help)")
    print()
    print("  This is like a mechanic (server/tool) asking you (client)")
    print("  to call an expert friend (LLM) for a second opinion.")
    print()

    # Simulated sampling request
    request = SamplingRequest(
        messages=[
            {"role": "user", "content": (
                "I analyzed the codebase and found 3 potential security issues. "
                "Please categorize each by severity (low/medium/high):\n"
                "1. SQL query uses string concatenation instead of parameterized queries\n"
                "2. API key is stored in a comment in the source code\n"
                "3. Error messages expose internal file paths"
            )}
        ],
        model_preferences={"hints": [{"name": "claude-sonnet"}]},
        max_tokens=300,
        purpose="Categorize security findings by severity"
    )

    print("  --- Simulated Sampling Flow ---")
    print(f"  Server → Client: 'I need LLM help to {request.purpose}'")
    print(f"  Messages: {json.dumps(request.messages, indent=2)}")
    print()
    print("  Client checks: Is this request appropriate? [APPROVED]")
    print("  Client → LLM: (forwards the request)")
    print("  LLM → Client: 'Issue 1: HIGH, Issue 2: HIGH, Issue 3: MEDIUM'")
    print("  Client → Server: (forwards the response)")
    print()
    print("  Key point: The CLIENT controls access to the LLM.")
    print("  The server cannot bypass the client's approval.")
    print()


# ================================================================
# SECTION 6 — Transport Types
# ================================================================

class TransportType(Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class TransportInfo:
    name: str
    mechanism: str
    direction: str
    use_case: str
    status: str


def demonstrate_transports() -> None:
    """Explain the three transport types."""
    print("=" * 60)
    print("  TRANSPORT TYPES")
    print("=" * 60)
    print()
    print("  Transport = HOW messages travel between client and server.")
    print("  The MCP messages themselves are the same regardless of transport.")
    print()

    transports = [
        TransportInfo(
            name="stdio (Standard I/O)",
            mechanism="Client spawns server as subprocess, uses stdin/stdout",
            direction="Bidirectional via pipes",
            use_case="Local tools (file system, git, local databases)",
            status="Stable, widely supported",
        ),
        TransportInfo(
            name="SSE (Server-Sent Events)",
            mechanism="HTTP POST for requests, SSE stream for responses",
            direction="Client→Server: POST | Server→Client: SSE stream",
            use_case="Remote servers, web-based tool providers",
            status="Stable but being superseded by Streamable HTTP",
        ),
        TransportInfo(
            name="Streamable HTTP",
            mechanism="Single HTTP endpoint, bidirectional streaming",
            direction="Both directions via single endpoint",
            use_case="Production deployments (recommended for new servers)",
            status="Newest transport, actively being adopted",
        ),
    ]

    for t in transports:
        print(f"  {t.name}")
        print(f"    Mechanism: {t.mechanism}")
        print(f"    Direction: {t.direction}")
        print(f"    Use case:  {t.use_case}")
        print(f"    Status:    {t.status}")
        print()

    print("  How to choose:")
    print("    • Building a local tool? → Use stdio")
    print("    • Building a remote service? → Use Streamable HTTP")
    print("    • Need backward compat? → Support SSE + Streamable HTTP")
    print()


# ================================================================
# SECTION 7 — Capability Negotiation
# ================================================================

class ServerCapabilities(BaseModel):
    """What the server supports — declared during initialization."""
    tools: bool = Field(default=False, description="Server exposes tools")
    resources: bool = Field(default=False, description="Server exposes resources")
    prompts: bool = Field(default=False, description="Server exposes prompt templates")
    sampling: bool = Field(default=False, description="Server may request LLM sampling")
    logging: bool = Field(default=False, description="Server emits log messages")


class ClientCapabilities(BaseModel):
    """What the client supports — declared during initialization."""
    sampling: bool = Field(default=False, description="Client can handle sampling requests")
    roots: bool = Field(default=False, description="Client can provide filesystem roots")


def demonstrate_capability_negotiation() -> None:
    """Show how client and server discover each other's features."""
    print("=" * 60)
    print("  CAPABILITY NEGOTIATION")
    print("=" * 60)
    print()
    print("  During initialization, client and server exchange")
    print("  capabilities. This way each side knows what the other")
    print("  supports — no guessing, no trial-and-error.")
    print()

    # Simulate different server configurations
    servers = [
        ("File System Server", ServerCapabilities(
            tools=True, resources=True, prompts=False, sampling=False
        )),
        ("Code Analysis Server", ServerCapabilities(
            tools=True, resources=True, prompts=True, sampling=True
        )),
        ("Weather Server", ServerCapabilities(
            tools=True, resources=False, prompts=False, sampling=False
        )),
    ]

    client_caps = ClientCapabilities(sampling=True, roots=True)

    print(f"  Client capabilities:")
    print(f"    • Sampling support: {client_caps.sampling}")
    print(f"    • Filesystem roots: {client_caps.roots}")
    print()

    for name, caps in servers:
        print(f"  {name}:")
        features = []
        if caps.tools:
            features.append("Tools")
        if caps.resources:
            features.append("Resources")
        if caps.prompts:
            features.append("Prompts")
        if caps.sampling:
            features.append("Sampling")
        print(f"    Supports: {', '.join(features) if features else 'None'}")

        # Check compatibility
        if caps.sampling and not client_caps.sampling:
            print(f"    Warning: Server wants sampling but client doesn't support it")
        if caps.sampling and client_caps.sampling:
            print(f"    Match: Both support sampling — server can request LLM help")
        print()

    print("  The negotiation ensures graceful degradation:")
    print("  if the server offers something the client can't handle,")
    print("  that feature is simply not used (no crashes).")
    print()


# ================================================================
# SECTION 8 — Pre-MCP vs Post-MCP Comparison
# ================================================================

def demonstrate_comparison() -> None:
    """Side-by-side comparison of the old way vs MCP."""
    print("=" * 60)
    print("  PRE-MCP vs POST-MCP WORLD")
    print("=" * 60)
    print()

    comparisons = [
        (
            "Adding a new tool",
            "Write custom adapter code for each AI app that needs it",
            "Implement one MCP server — all MCP clients can use it instantly"
        ),
        (
            "Supporting a new AI app",
            "Write custom integrations for each tool it needs",
            "Implement one MCP client — all MCP servers are accessible"
        ),
        (
            "Tool discovery",
            "Hardcoded tool definitions in each application",
            "Dynamic discovery via tools/list — tools self-describe"
        ),
        (
            "Security model",
            "Each integration has its own auth approach",
            "Standardized capability negotiation and approval flows"
        ),
        (
            "Transport flexibility",
            "Each tool decides its own communication method",
            "Standard transports (stdio, SSE, Streamable HTTP)"
        ),
        (
            "Error handling",
            "Different error formats per integration",
            "Consistent JSON-RPC error codes and messages"
        ),
        (
            "Ecosystem growth",
            "Linear — each new tool requires work from each app",
            "Network effect — each new server benefits all clients"
        ),
    ]

    # Print as a formatted table
    print(f"  {'Aspect':<24} {'Pre-MCP':<34} {'Post-MCP'}")
    print(f"  {'─' * 24} {'─' * 34} {'─' * 38}")
    for aspect, pre, post in comparisons:
        # Wrap long lines
        print(f"\n  {aspect}")
        print(f"    Before: {pre}")
        print(f"    After:  {post}")

    print()


# ================================================================
# SECTION 9 — Putting It All Together
# ================================================================

def demonstrate_full_picture() -> None:
    """Show how all four primitives work together in a real scenario."""
    print("=" * 60)
    print("  ALL FOUR PRIMITIVES IN ACTION")
    print("=" * 60)
    print()
    print("  Scenario: An AI agent helps a developer review code.")
    print("  The agent connects to a 'Code Assistant' MCP server.")
    print()

    steps = [
        (
            "RESOURCE",
            "Agent reads the source code file",
            "resources/read → file:///src/main.py → returns file contents"
        ),
        (
            "TOOL",
            "Agent runs the linter on the code",
            "tools/call → run_linter(file='main.py') → returns lint warnings"
        ),
        (
            "PROMPT",
            "Agent uses the server's review template",
            "prompts/get → code_review(code=..., language='python') → formatted prompt"
        ),
        (
            "SAMPLING",
            "Server asks LLM to categorize findings",
            "sampling/createMessage → LLM categorizes bugs by severity → returns categories"
        ),
    ]

    for primitive, action, detail in steps:
        print(f"  Step [{primitive}]: {action}")
        print(f"    MCP call: {detail}")
        print()

    print("  Result: The agent produces a comprehensive code review")
    print("  by combining read-only data (Resource), actions (Tool),")
    print("  standardized instructions (Prompt), and AI assistance")
    print("  within the tool itself (Sampling).")
    print()


# ================================================================
# SECTION 10 — Run Everything
# ================================================================

def main() -> None:
    print("=" * 60)
    print("  MCP: PURPOSE AND CORE IDEAS")
    print("=" * 60)
    print()
    print("  Before MCP, every AI app needed a custom adapter for")
    print("  every tool — like needing a different charger for every")
    print("  phone. MCP is the universal charging standard for AI.")
    print()

    demonstrate_integration_problem()
    demonstrate_resources()
    demonstrate_tools()
    demonstrate_prompts()
    demonstrate_sampling()
    demonstrate_transports()
    demonstrate_capability_negotiation()
    demonstrate_comparison()
    demonstrate_full_picture()

    print("=" * 60)
    print("  KEY TAKEAWAYS")
    print("=" * 60)
    print("""
  1. MCP solves the N×M integration problem by providing a
     standard protocol — build once, connect everywhere.

  2. Four core primitives:
     • Resources — read-only data (safe, no side effects)
     • Tools     — actions (side effects, needs approval)
     • Prompts   — reusable templates (consistent quality)
     • Sampling  — reverse flow (server asks client's LLM)

  3. Three transport types:
     • stdio          — local tools (simple, secure)
     • SSE            — remote servers (being replaced)
     • Streamable HTTP — production (recommended for new work)

  4. Capability negotiation ensures client and server only use
     features both sides support — graceful, no crashes.

  5. MCP creates a network effect: every new server benefits
     ALL existing clients, and vice versa.
    """)
    print("  Done! You now understand why MCP exists and how it works.")


if __name__ == "__main__":
    main()
