import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5d: Multi-Agent Communication Patterns with LLM Agents
================================================================
Example 04 introduced three communication patterns conceptually.
This example implements all three with REAL LLM agents solving the
same problem, so you can compare how information flows in each.

The three patterns:
  1. SHARED STATE — all agents read/write a common state dictionary
     (like a shared Google Doc)
  2. MESSAGE PASSING — agents send typed messages to specific recipients
     (like email with inboxes)
  3. BLACKBOARD — structured slots that agents read/write based on role
     (like a shared whiteboard with assigned sections)

All three solve the same task: analyze a business proposal.
  - Researcher gathers market data
  - Analyst evaluates financials
  - Strategist recommends action

Same agents, same task, different communication mechanics.

Run: python week-04-advanced-patterns/examples/example_05d_communication_patterns.py
"""

import os
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================
# LLM Setup
# ==============================================================

def get_llm(temperature=0.5):
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


llm = get_llm()

TASK = (
    "Analyze this business proposal: Launch a premium dog food subscription "
    "service targeting urban millennials, priced at $49/month, with organic "
    "and locally-sourced ingredients."
)


def call_agent(role: str, instruction: str, context: str) -> str:
    """Call an LLM agent with a specific role and context."""
    response = llm.invoke([
        SystemMessage(content=f"You are a {role}. {instruction} Keep to 80 words."),
        HumanMessage(content=context),
    ])
    return response.content


# ================================================================
# PATTERN 1: SHARED STATE
# ================================================================
# All agents read from and write to the SAME state dictionary.
# Simple and easy to implement. This is what LangGraph uses by default.
#
# Pros: Simple, every agent sees everything, easy to debug
# Cons: Agents can overwrite each other, no access control,
#       state grows large as agents add data

def demo_shared_state():
    """All agents read/write a shared dictionary."""
    print(f"\n{'='*60}")
    print("  PATTERN 1: SHARED STATE")
    print(f"{'='*60}")
    print("  All agents read from and write to the same state dict.")
    print("  Like: a shared Google Doc everyone edits\n")

    # The shared state — everyone can read everything
    state = {"task": TASK, "market_data": "", "financial_analysis": "", "recommendation": ""}

    # Agent 1: Researcher — reads task, writes market_data
    print("  [RESEARCHER] Reading state['task'], writing state['market_data']...")
    state["market_data"] = call_agent(
        "market researcher",
        "Analyze market potential for this business. Focus on target demographic, "
        "market size, and competition.",
        f"Task: {state['task']}",
    )
    print(f"    → {state['market_data'][:100]}...\n")

    # Agent 2: Analyst — reads task + market_data, writes financial_analysis
    print("  [ANALYST] Reading state['task'] + state['market_data'], writing state['financial_analysis']...")
    state["financial_analysis"] = call_agent(
        "financial analyst",
        "Evaluate the financial viability. Consider pricing, margins, and unit economics.",
        f"Task: {state['task']}\nMarket Data: {state['market_data']}",
    )
    print(f"    → {state['financial_analysis'][:100]}...\n")

    # Agent 3: Strategist — reads EVERYTHING, writes recommendation
    print("  [STRATEGIST] Reading ALL state, writing state['recommendation']...")
    state["recommendation"] = call_agent(
        "business strategist",
        "Based on market data and financial analysis, give a clear go/no-go "
        "recommendation with top 3 action items.",
        f"Task: {state['task']}\n"
        f"Market Data: {state['market_data']}\n"
        f"Financial Analysis: {state['financial_analysis']}",
    )
    print(f"    → {state['recommendation'][:100]}...\n")

    print(f"  SHARED STATE RESULT:\n  {state['recommendation']}\n")
    print("  Characteristics:")
    print("    + Simple: just a dictionary")
    print("    + Transparent: every agent sees everything")
    print("    - No access control: researcher could overwrite financial_analysis")
    print("    - State bloat: grows with every agent's output")
    return state


# ================================================================
# PATTERN 2: MESSAGE PASSING
# ================================================================
# Agents send typed messages to specific recipients via inboxes.
# Each agent has its own inbox and can only read messages sent TO it.
#
# Pros: Explicit communication, clear sender/receiver, auditable
# Cons: More complex, agents must know who to message,
#       can miss info if wrong recipient

@dataclass
class Message:
    """A typed message between agents."""
    sender: str
    receiver: str
    content: str
    msg_type: str = "info"  # info, request, result


class MessageBus:
    """Simple message bus with per-agent inboxes."""

    def __init__(self):
        self.inboxes: dict = defaultdict(list)
        self.all_messages: list = []

    def send(self, msg: Message):
        """Send a message to a specific agent's inbox."""
        self.inboxes[msg.receiver].append(msg)
        self.all_messages.append(msg)
        print(f"    [MSG] {msg.sender} → {msg.receiver}: "
              f"{msg.content[:60]}... [{msg.msg_type}]")

    def read_inbox(self, agent_name: str) -> list:
        """Read all messages in an agent's inbox."""
        return self.inboxes.get(agent_name, [])

    def get_context(self, agent_name: str) -> str:
        """Format inbox messages as context for an LLM call."""
        messages = self.read_inbox(agent_name)
        if not messages:
            return "No messages received."
        parts = []
        for m in messages:
            parts.append(f"From {m.sender} ({m.msg_type}): {m.content}")
        return "\n".join(parts)


def demo_message_passing():
    """Agents communicate through typed messages with explicit routing."""
    print(f"\n{'='*60}")
    print("  PATTERN 2: MESSAGE PASSING")
    print(f"{'='*60}")
    print("  Agents send messages to specific recipients via inboxes.")
    print("  Like: email with explicit To/From fields\n")

    bus = MessageBus()

    # Coordinator sends task to researcher
    bus.send(Message("coordinator", "researcher", TASK, "request"))

    # Researcher processes and sends results to analyst
    print("\n  [RESEARCHER] Reading inbox, sending results to analyst...")
    context = bus.get_context("researcher")
    research = call_agent(
        "market researcher",
        "Analyze market potential. Focus on demographics, size, and competition.",
        context,
    )
    bus.send(Message("researcher", "analyst", research, "result"))

    # Analyst processes and sends results to strategist
    print("\n  [ANALYST] Reading inbox, sending results to strategist...")
    context = bus.get_context("analyst")
    analysis = call_agent(
        "financial analyst",
        "Evaluate financial viability based on the research you received.",
        context,
    )
    bus.send(Message("analyst", "strategist", analysis, "result"))
    # Researcher also sends directly to strategist
    bus.send(Message("researcher", "strategist", research, "result"))

    # Strategist reads messages from BOTH researcher and analyst
    print("\n  [STRATEGIST] Reading inbox (messages from researcher + analyst)...")
    context = bus.get_context("strategist")
    recommendation = call_agent(
        "business strategist",
        "Give a go/no-go recommendation with top 3 action items based on messages received.",
        context,
    )

    print(f"\n  MESSAGE PASSING RESULT:\n  {recommendation}\n")
    print(f"  Total messages sent: {len(bus.all_messages)}")
    print("  Characteristics:")
    print("    + Explicit: clear who sends what to whom")
    print("    + Auditable: full message log for debugging")
    print("    + Access control: agents only see their inbox")
    print("    - Complex: agents must know routing (who to send to)")
    print("    - Can miss info: if researcher forgets to CC strategist")
    return recommendation


# ================================================================
# PATTERN 3: BLACKBOARD
# ================================================================
# A structured shared workspace with named SLOTS. Each agent has
# READ access to some slots and WRITE access to others based on role.
#
# This is the middle ground: structured like shared state,
# controlled like message passing.
#
# Pros: Structured, role-based access, clear responsibilities
# Cons: Rigid slot structure, needs upfront design

class Blackboard:
    """Structured workspace with role-based access control."""

    def __init__(self):
        self.slots = {
            "task": "",
            "market_data": "",
            "financial_analysis": "",
            "recommendation": "",
        }
        # Define who can write to which slots
        self.write_access = {
            "coordinator": ["task"],
            "researcher": ["market_data"],
            "analyst": ["financial_analysis"],
            "strategist": ["recommendation"],
        }
        # Define who can read which slots
        self.read_access = {
            "researcher": ["task"],
            "analyst": ["task", "market_data"],
            "strategist": ["task", "market_data", "financial_analysis"],
        }

    def write(self, agent: str, slot: str, value: str):
        """Write to a slot if the agent has permission."""
        if slot not in self.write_access.get(agent, []):
            print(f"    [DENIED] {agent} cannot write to '{slot}'")
            return False
        self.slots[slot] = value
        print(f"    [WRITE] {agent} → {slot}: {value[:60]}...")
        return True

    def read(self, agent: str) -> str:
        """Read all slots the agent has access to."""
        readable = self.read_access.get(agent, [])
        parts = []
        for slot in readable:
            if self.slots[slot]:
                parts.append(f"{slot}: {self.slots[slot]}")
        return "\n".join(parts) if parts else "No data available."


def demo_blackboard():
    """Agents interact through a structured blackboard with access control."""
    print(f"\n{'='*60}")
    print("  PATTERN 3: BLACKBOARD")
    print(f"{'='*60}")
    print("  Structured workspace with role-based read/write access.")
    print("  Like: a whiteboard where each person has an assigned section\n")

    board = Blackboard()

    # Coordinator posts the task
    board.write("coordinator", "task", TASK)

    # Researcher reads task slot, writes market_data slot
    print("\n  [RESEARCHER] Reads: [task] | Writes: [market_data]")
    context = board.read("researcher")
    research = call_agent(
        "market researcher",
        "Analyze market potential. Focus on demographics, size, and competition.",
        context,
    )
    board.write("researcher", "market_data", research)

    # Show access control: researcher CANNOT write to financial_analysis
    print("\n  Demonstrating access control:")
    board.write("researcher", "financial_analysis", "I shouldn't be able to write here")

    # Analyst reads task + market_data, writes financial_analysis
    print("\n  [ANALYST] Reads: [task, market_data] | Writes: [financial_analysis]")
    context = board.read("analyst")
    analysis = call_agent(
        "financial analyst",
        "Evaluate financial viability based on market data.",
        context,
    )
    board.write("analyst", "financial_analysis", analysis)

    # Strategist reads ALL data slots, writes recommendation
    print("\n  [STRATEGIST] Reads: [task, market_data, financial_analysis] | Writes: [recommendation]")
    context = board.read("strategist")
    recommendation = call_agent(
        "business strategist",
        "Give a go/no-go recommendation with top 3 action items.",
        context,
    )
    board.write("strategist", "recommendation", recommendation)

    print(f"\n  BLACKBOARD RESULT:\n  {recommendation}\n")
    print("  Characteristics:")
    print("    + Structured: clear slots for each type of data")
    print("    + Access control: agents can only write their own slots")
    print("    + Scalable: add new slots/agents without breaking others")
    print("    - Rigid: slot structure must be designed upfront")
    print("    - Less flexible than free-form message passing")
    return recommendation


# ================================================================
# Main — Run all three and compare
# ================================================================

if __name__ == "__main__":
    print("Example 5d: Multi-Agent Communication Patterns")
    print("=" * 60)
    print(f"Task: {TASK[:70]}...")
    print("Same agents, same task, three different communication styles.")
    print("=" * 60)

    demo_shared_state()
    demo_message_passing()
    demo_blackboard()

    print(f"\n{'='*60}")
    print("  COMPARISON: When to Use Which Pattern")
    print(f"{'='*60}")
    print("""
  SHARED STATE (LangGraph default):
    Best for: Simple pipelines, 2-4 agents, prototyping
    Example:  LangGraph TypedDict state passed between nodes
    Risk:     State bloat, accidental overwrites
    Use when: "Just get it working"

  MESSAGE PASSING (ADK / custom):
    Best for: Complex workflows, many agents, audit requirements
    Example:  ADK's event-based communication, Slack-like channels
    Risk:     Routing complexity, missed messages
    Use when: "I need to know exactly who said what to whom"

  BLACKBOARD (hybrid):
    Best for: Structured analysis, role-based teams, access control
    Example:  Shared workspace with assigned sections per role
    Risk:     Rigid structure, upfront design required
    Use when: "Different agents own different data sections"

  Progression in practice:
    1. Start with SHARED STATE (simplest)
    2. If agents overwrite each other → switch to BLACKBOARD
    3. If you need audit trails / complex routing → MESSAGE PASSING
""")
    print(f"{'='*60}")
