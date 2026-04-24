import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 7: A2A vs MCP — Complementary Standards
=================================================
Topic 7 — MCP gives agents hands (tools). A2A gives agents voices (communication).

The BIG IDEA (Feynman):
  Think of a hospital.  MCP is like the medical equipment — stethoscopes,
  X-ray machines, lab equipment.  Any doctor can use them.  A2A is like
  the communication system — how the ER doctor talks to the surgeon,
  how the surgeon talks to the anesthesiologist.

  You need BOTH.  Equipment without communication = chaos.
  Communication without equipment = just talking, no action.

First Principles:
  MCP answers: "How does an agent USE a tool?"
  A2A answers: "How does an agent TALK TO another agent?"

  They're different layers of the stack:
    Layer 3: Agent-to-Agent   (A2A)  ← coordination
    Layer 2: Agent-to-Tool    (MCP)  ← capability
    Layer 1: LLM              (API)  ← intelligence

Run: python week-07-mcp-a2a-synthesis/examples/example_07_a2a_vs_mcp.py
"""

from dataclasses import dataclass, field
from typing import Optional
import json


# ================================================================
# PART 1: SIDE-BY-SIDE COMPARISON
# ================================================================

print("=" * 70)
print("PART 1: MCP vs A2A — Feature Comparison")
print("=" * 70)

comparison = [
    ("Purpose",          "Connect agents to TOOLS",          "Connect agents to AGENTS"),
    ("Analogy",          "USB port (plug in any device)",    "Phone network (call any agent)"),
    ("Relationship",     "Agent ↔ Tool",                     "Agent ↔ Agent"),
    ("Discovery",        "Server declares tools",            "Agent Card at /.well-known/"),
    ("Communication",    "JSON-RPC (stdio/SSE)",             "JSON-RPC (HTTP)"),
    ("Unit of work",     "Tool call (request/response)",     "Task (state machine)"),
    ("Statefulness",     "Stateless (each call independent)","Stateful (task lifecycle)"),
    ("Who runs it",      "Tool server (passive)",            "Agent server (active, has LLM)"),
    ("Authentication",   "Per-server config",                "In Agent Card (OAuth2, API key)"),
    ("Streaming",        "SSE transport",                    "Streaming task responses"),
    ("Spec owner",       "Anthropic (open standard)",        "Google (open standard)"),
    ("Key concept",      "Tools, Resources, Prompts",        "Agent Cards, Tasks, Artifacts"),
]

# Print as a formatted table
print(f"\n  {'Feature':<20} {'MCP':<35} {'A2A':<35}")
print(f"  {'─' * 20} {'─' * 35} {'─' * 35}")
for feature, mcp_val, a2a_val in comparison:
    print(f"  {feature:<20} {mcp_val:<35} {a2a_val:<35}")


# ================================================================
# PART 2: WHEN TO USE WHICH
# ================================================================

print("\n" + "=" * 70)
print("PART 2: When to Use Which")
print("=" * 70)
print("""
Use MCP when:
  ✅ You need to call a database, API, or external service
  ✅ The "other side" is a tool (not an AI agent)
  ✅ You want stateless request/response
  ✅ You're adding capabilities to a single agent

Use A2A when:
  ✅ You need two AI agents to collaborate on a task
  ✅ The "other side" has its own LLM and decision-making
  ✅ You need a task lifecycle (submitted → working → completed)
  ✅ You're building a multi-agent system across services

Use BOTH when:
  ✅ Agent A uses MCP to access tools (databases, APIs)
  ✅ Agent A communicates results to Agent B via A2A
  ✅ Agent B uses its own MCP tools to process the results
  ✅ This is the HYBRID pattern — and it's the most powerful!
""")


# ================================================================
# PART 3: THE LAYERED ARCHITECTURE
# ================================================================

print("=" * 70)
print("PART 3: The Layered Architecture")
print("=" * 70)
print("""
How MCP and A2A fit together in a real system:

  ┌─────────────────────────────────────────────────────────────┐
  │                    USER / APPLICATION                        │
  │                  "Research AI safety"                        │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │              LAYER 3: A2A (Agent Coordination)               │
  │                                                              │
  │   ┌─────────────┐    A2A     ┌─────────────┐               │
  │   │  Research    │◄──────────►│   Writer    │               │
  │   │  Agent      │   Tasks    │   Agent     │               │
  │   └──────┬──────┘            └──────┬──────┘               │
  │          │                          │                       │
  └──────────┼──────────────────────────┼───────────────────────┘
             │                          │
  ┌──────────▼──────────────────────────▼───────────────────────┐
  │              LAYER 2: MCP (Tool Access)                      │
  │                                                              │
  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                │
  │   │ Web      │  │ Database │  │ File     │                 │
  │   │ Search   │  │ Server   │  │ System   │                 │
  │   │ (MCP)    │  │ (MCP)    │  │ (MCP)    │                 │
  │   └──────────┘  └──────────┘  └──────────┘                │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
             │                          │
  ┌──────────▼──────────────────────────▼───────────────────────┐
  │              LAYER 1: LLM APIs                               │
  │                                                              │
  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                │
  │   │  Groq    │  │  OpenAI  │  │  Gemini  │                │
  │   └──────────┘  └──────────┘  └──────────┘                │
  └──────────────────────────────────────────────────────────────┘

Key insight: Each layer is independent. You can swap LLMs without
changing MCP tools, or swap MCP tools without changing A2A agents.
""")


# ================================================================
# PART 4: HYBRID SCENARIO WALKTHROUGH
# ================================================================

print("=" * 70)
print("PART 4: Hybrid Scenario — Research + Write Article")
print("=" * 70)
print("""
Let's trace a complete request through both protocols:

User: "Write a blog post about quantum computing advances"
""")


@dataclass
class HybridStep:
    """One step in the hybrid MCP + A2A flow."""
    step_num: int
    protocol: str  # "A2A" or "MCP" or "Internal"
    actor: str     # Who is doing this step
    action: str    # What they're doing
    detail: str    # The actual data being exchanged


# Trace through the hybrid flow
steps = [
    HybridStep(1, "A2A", "User → Planner",
        "Send task via A2A",
        "POST /tasks/send {message: 'Write a blog post about quantum computing'}"),

    HybridStep(2, "Internal", "Planner Agent",
        "Decompose into sub-tasks",
        "Sub-task 1: Research quantum computing\nSub-task 2: Write article"),

    HybridStep(3, "A2A", "Planner → Researcher",
        "Delegate research via A2A",
        "POST /tasks/send {message: 'Research quantum computing advances 2025-2026'}"),

    HybridStep(4, "MCP", "Researcher → Web Search MCP",
        "Search the web via MCP tool",
        "call_tool('web_search', {query: 'quantum computing breakthroughs 2026'})"),

    HybridStep(5, "MCP", "Researcher → Database MCP",
        "Check past research via MCP tool",
        "call_tool('search_research', {query: 'quantum computing'})"),

    HybridStep(6, "MCP", "Researcher → Database MCP",
        "Save new research via MCP tool",
        "call_tool('save_research', {topic: 'quantum computing', summary: '...'})"),

    HybridStep(7, "A2A", "Researcher → Planner",
        "Return research results via A2A",
        "Task completed with artifact: {name: 'research_notes', content: '...'}"),

    HybridStep(8, "A2A", "Planner → Writer",
        "Send research to writer via A2A",
        "POST /tasks/send {message: 'Write blog post', context: research_notes}"),

    HybridStep(9, "MCP", "Writer → File System MCP",
        "Save draft via MCP tool",
        "call_tool('write_file', {path: 'blog_post.md', content: '...'})"),

    HybridStep(10, "A2A", "Writer → Planner",
        "Return finished article via A2A",
        "Task completed with artifact: {name: 'blog_post', format: 'markdown'}"),

    HybridStep(11, "A2A", "Planner → User",
        "Return final result via A2A",
        "Task completed with artifact: blog_post.md"),
]

for step in steps:
    protocol_emoji = {"A2A": "🤝", "MCP": "🔧", "Internal": "🧠"}
    emoji = protocol_emoji.get(step.protocol, "❓")

    print(f"\n  Step {step.step_num}: [{step.protocol}] {emoji} {step.actor}")
    print(f"  Action: {step.action}")
    print(f"  Data: {step.detail[:80]}{'...' if len(step.detail) > 80 else ''}")


# ================================================================
# PART 5: PROTOCOL INTERACTION PATTERNS
# ================================================================

print("\n\n" + "=" * 70)
print("PART 5: Common Integration Patterns")
print("=" * 70)

patterns = {
    "Pattern 1: MCP-Only (Single Agent + Tools)": {
        "description": "One agent uses MCP to access multiple tools",
        "when": "Simple tasks that don't need agent collaboration",
        "flow": "User → Agent → [MCP Tool 1, MCP Tool 2, ...] → User",
        "example": "A weather bot that checks weather + traffic + news",
    },
    "Pattern 2: A2A-Only (Agent Delegation)": {
        "description": "Agents delegate tasks to specialists without tools",
        "when": "Pure LLM tasks (writing, analysis, translation)",
        "flow": "User → Agent A → Agent B → Agent A → User",
        "example": "Translate + summarize: Translator agent → Summarizer agent",
    },
    "Pattern 3: Hybrid Sequential": {
        "description": "Agent uses MCP for data, then sends to another agent via A2A",
        "when": "Research-then-write, gather-then-analyze",
        "flow": "User → Agent A → [MCP tools] → Agent B (via A2A) → User",
        "example": "Research agent gathers data → Writer agent produces report",
    },
    "Pattern 4: Hybrid Parallel": {
        "description": "Multiple agents work simultaneously, each with their own MCP tools",
        "when": "Complex tasks that can be parallelized",
        "flow": "User → Supervisor → [Agent A + MCP, Agent B + MCP] → Supervisor → User",
        "example": "Parallel research on different sub-topics, then merge",
    },
    "Pattern 5: Hierarchical Hybrid": {
        "description": "A supervisor coordinates sub-agents, each with MCP tools",
        "when": "Large, multi-step projects",
        "flow": "User → Planner → [Researcher+MCP, Writer+MCP, Reviewer] → User",
        "example": "Full content pipeline: plan → research → write → review → publish",
    },
}

for name, details in patterns.items():
    print(f"\n  📋 {name}")
    print(f"     {details['description']}")
    print(f"     When: {details['when']}")
    print(f"     Flow: {details['flow']}")
    print(f"     Example: {details['example']}")


# ================================================================
# PART 6: DECISION MATRIX
# ================================================================

print("\n\n" + "=" * 70)
print("PART 6: Decision Matrix — Which Protocol Do I Need?")
print("=" * 70)

print("""
  Ask yourself these questions:

  Q1: Does the other side have an LLM / make decisions?
      YES → A2A (it's an agent)
      NO  → MCP (it's a tool)

  Q2: Do I need a multi-step conversation?
      YES → A2A (tasks have lifecycles)
      NO  → MCP (single request/response)

  Q3: Does the other side need to be discoverable by anyone?
      YES → A2A (Agent Cards for discovery)
      NO  → MCP (configured per-client)

  Q4: Is the interaction stateless (fire-and-forget)?
      YES → MCP (tool calls are independent)
      NO  → A2A (tasks maintain state)

  Q5: Am I building a single agent or a multi-agent system?
      Single → MCP (add tools to your agent)
      Multi  → A2A + MCP (agents coordinate via A2A, use tools via MCP)
""")


# ================================================================
# KEY TAKEAWAYS
# ================================================================

print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. MCP = agent-to-TOOL protocol (like a USB port)
2. A2A = agent-to-AGENT protocol (like a phone network)
3. They're COMPLEMENTARY, not competing — use both!
4. MCP is at the tool layer, A2A is at the coordination layer
5. The hybrid pattern (MCP + A2A) is the most powerful for real systems
6. Decision matrix: Does the other side think? A2A. Is it a tool? MCP.

Coming up: example_08 builds a REAL A2A agent with FastAPI!
""")

print("✅ Example 07 complete!")
