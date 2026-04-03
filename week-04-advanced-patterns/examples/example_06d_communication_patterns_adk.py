import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6d: Multi-Agent Communication Patterns using Google ADK
================================================================
The ADK counterpart of Example 5d. Implements all three multi-agent
communication patterns using Google ADK LlmAgents instead of LangChain.

The three patterns:
  1. SHARED STATE -- all agents read/write a common state dictionary
     (like a shared Google Doc)
  2. MESSAGE PASSING -- agents send typed messages to specific recipients
     (like email with inboxes)
  3. BLACKBOARD -- structured slots with role-based read/write access
     (like a shared whiteboard with assigned sections)

All three solve the same task: analyze a business proposal
(researcher -> analyst -> strategist). ADK's async, event-based
model naturally aligns with message passing, making Pattern 2
particularly idiomatic in ADK.

Run: python week-04-advanced-patterns/examples/example_06d_communication_patterns_adk.py
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Configuration
# ==============================================================

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")

TASK = (
    "Analyze this business proposal: Launch a premium dog food subscription "
    "service targeting urban millennials, priced at $49/month, with organic "
    "and locally-sourced ingredients."
)


# ==============================================================
# ADK Agent Definitions
# ==============================================================
# Each agent is an LlmAgent with a focused role and instruction.
# We create them once and reuse across all three patterns.

researcher_agent = LlmAgent(
    name="researcher",
    model=MODEL,
    instruction=(
        "You are a market researcher. Analyze market potential for a business "
        "proposal. Focus on target demographic, market size, and competition. "
        "Keep your response to 80 words."
    ),
    tools=[],
    description="Analyzes market potential for business proposals.",
)

analyst_agent = LlmAgent(
    name="analyst",
    model=MODEL,
    instruction=(
        "You are a financial analyst. Evaluate the financial viability of a "
        "business proposal. Consider pricing, margins, and unit economics. "
        "Keep your response to 80 words."
    ),
    tools=[],
    description="Evaluates financial viability of business proposals.",
)

strategist_agent = LlmAgent(
    name="strategist",
    model=MODEL,
    instruction=(
        "You are a business strategist. Based on market data and financial "
        "analysis, give a clear go/no-go recommendation with your top 3 "
        "action items. Keep your response to 80 words."
    ),
    tools=[],
    description="Provides strategic recommendations for business proposals.",
)


# ==============================================================
# Helper: Run an ADK Agent with Context
# ==============================================================

async def call_adk_agent(agent: LlmAgent, context: str, retries: int = 5) -> str:
    """Run an ADK agent with a context string and return the response text.
    Each call creates a fresh session -- we control info flow explicitly.
    Includes retry logic for transient API errors (503, rate limits).
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="comm_patterns_demo",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="comm_patterns_demo",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=context)],
                ),
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        result_text = event.content.parts[0].text

            return result_text
        except Exception as e:
            if attempt < retries:
                wait = attempt * 10
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [ERROR] All {retries} attempts failed: {e}")
                return f"[Error: API temporarily unavailable after {retries} retries]"


# ================================================================
# PATTERN 1: SHARED STATE
# ================================================================
# All agents read/write the SAME state dictionary. Simple, like LangGraph default.
# Pros: Simple, every agent sees everything, easy to debug
# Cons: No access control, state grows large

async def demo_shared_state():
    """All agents read/write a shared dictionary."""
    print(f"\n{'='*60}")
    print("  PATTERN 1: SHARED STATE (ADK)")
    print(f"{'='*60}")
    print("  All agents read from and write to the same state dict.")
    print("  Like: a shared Google Doc everyone edits\n")

    # The shared state -- everyone can read everything
    state = {
        "task": TASK,
        "market_data": "",
        "financial_analysis": "",
        "recommendation": "",
    }

    # Agent 1: Researcher -- reads task, writes market_data
    print("  [RESEARCHER] Reading state['task'], writing state['market_data']...")
    state["market_data"] = await call_adk_agent(
        researcher_agent,
        f"Task: {state['task']}",
    )
    print(f"    -> {state['market_data'][:100]}...\n")

    # Agent 2: Analyst -- reads task + market_data, writes financial_analysis
    print("  [ANALYST] Reading state['task'] + state['market_data'], writing state['financial_analysis']...")
    state["financial_analysis"] = await call_adk_agent(
        analyst_agent,
        f"Task: {state['task']}\nMarket Data: {state['market_data']}",
    )
    print(f"    -> {state['financial_analysis'][:100]}...\n")

    # Agent 3: Strategist -- reads EVERYTHING, writes recommendation
    print("  [STRATEGIST] Reading ALL state, writing state['recommendation']...")
    state["recommendation"] = await call_adk_agent(
        strategist_agent,
        f"Task: {state['task']}\n"
        f"Market Data: {state['market_data']}\n"
        f"Financial Analysis: {state['financial_analysis']}",
    )
    print(f"    -> {state['recommendation'][:100]}...\n")

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
# ADK's event-based, async architecture naturally fits this pattern.
# Pros: Explicit communication, clear sender/receiver, auditable
# Cons: More complex, agents must know who to message

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
        print(f"    [MSG] {msg.sender} -> {msg.receiver}: "
              f"{msg.content[:60]}... [{msg.msg_type}]")

    def read_inbox(self, agent_name: str) -> list:
        """Read all messages in an agent's inbox."""
        return self.inboxes.get(agent_name, [])

    def get_context(self, agent_name: str) -> str:
        """Format inbox messages as context for an agent call."""
        messages = self.read_inbox(agent_name)
        if not messages:
            return "No messages received."
        parts = []
        for m in messages:
            parts.append(f"From {m.sender} ({m.msg_type}): {m.content}")
        return "\n".join(parts)


async def demo_message_passing():
    """Agents communicate through typed messages with explicit routing."""
    print(f"\n{'='*60}")
    print("  PATTERN 2: MESSAGE PASSING (ADK)")
    print(f"{'='*60}")
    print("  Agents send messages to specific recipients via inboxes.")
    print("  Like: email with explicit To/From fields")
    print("  ADK's event-based model naturally aligns with this pattern.\n")

    bus = MessageBus()

    # Coordinator sends task to researcher
    bus.send(Message("coordinator", "researcher", TASK, "request"))

    # Researcher processes and sends results to analyst
    print("\n  [RESEARCHER] Reading inbox, sending results to analyst...")
    context = bus.get_context("researcher")
    research = await call_adk_agent(researcher_agent, context)
    bus.send(Message("researcher", "analyst", research, "result"))

    # Analyst processes and sends results to strategist
    print("\n  [ANALYST] Reading inbox, sending results to strategist...")
    context = bus.get_context("analyst")
    analysis = await call_adk_agent(analyst_agent, context)
    bus.send(Message("analyst", "strategist", analysis, "result"))
    # Researcher also sends directly to strategist
    bus.send(Message("researcher", "strategist", research, "result"))

    # Strategist reads messages from BOTH researcher and analyst
    print("\n  [STRATEGIST] Reading inbox (messages from researcher + analyst)...")
    context = bus.get_context("strategist")
    recommendation = await call_adk_agent(strategist_agent, context)

    print(f"\n  MESSAGE PASSING RESULT:\n  {recommendation}\n")
    print(f"  Total messages sent: {len(bus.all_messages)}")
    print("  Characteristics:")
    print("    + Explicit: clear who sends what to whom")
    print("    + Auditable: full message log for debugging")
    print("    + Access control: agents only see their inbox")
    print("    + Natural fit: ADK events work like messages between agents")
    print("    - Complex: agents must know routing (who to send to)")
    print("    - Can miss info: if researcher forgets to CC strategist")
    return recommendation


# ================================================================
# PATTERN 3: BLACKBOARD
# ================================================================
# Structured shared workspace with named SLOTS and role-based access.
# Middle ground: structured like shared state, controlled like message passing.
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
        print(f"    [WRITE] {agent} -> {slot}: {value[:60]}...")
        return True

    def read(self, agent: str) -> str:
        """Read all slots the agent has access to."""
        readable = self.read_access.get(agent, [])
        parts = []
        for slot in readable:
            if self.slots[slot]:
                parts.append(f"{slot}: {self.slots[slot]}")
        return "\n".join(parts) if parts else "No data available."


async def demo_blackboard():
    """Agents interact through a structured blackboard with access control."""
    print(f"\n{'='*60}")
    print("  PATTERN 3: BLACKBOARD (ADK)")
    print(f"{'='*60}")
    print("  Structured workspace with role-based read/write access.")
    print("  Like: a whiteboard where each person has an assigned section\n")

    board = Blackboard()

    # Coordinator posts the task
    board.write("coordinator", "task", TASK)

    # Researcher reads task slot, writes market_data slot
    print("\n  [RESEARCHER] Reads: [task] | Writes: [market_data]")
    context = board.read("researcher")
    research = await call_adk_agent(researcher_agent, context)
    board.write("researcher", "market_data", research)

    # Show access control: researcher CANNOT write to financial_analysis
    print("\n  Demonstrating access control:")
    board.write("researcher", "financial_analysis", "I shouldn't be able to write here")

    # Analyst reads task + market_data, writes financial_analysis
    print("\n  [ANALYST] Reads: [task, market_data] | Writes: [financial_analysis]")
    context = board.read("analyst")
    analysis = await call_adk_agent(analyst_agent, context)
    board.write("analyst", "financial_analysis", analysis)

    # Strategist reads ALL data slots, writes recommendation
    print("\n  [STRATEGIST] Reads: [task, market_data, financial_analysis] | Writes: [recommendation]")
    context = board.read("strategist")
    recommendation = await call_adk_agent(strategist_agent, context)
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
# Main -- Run all three patterns and compare
# ================================================================

async def main():
    """Run all three communication patterns and compare results."""
    print("Example 6d: Multi-Agent Communication Patterns (ADK)")
    print("=" * 60)
    print(f"Task: {TASK[:70]}...")
    print("Same agents, same task, three different communication styles.")
    print("Using Google ADK LlmAgents with async execution.")
    print("=" * 60)

    await demo_shared_state()
    await demo_message_passing()
    await demo_blackboard()

    # ============================================================
    # Comparison: When to Use Which Pattern
    # ============================================================
    print(f"\n{'='*60}")
    print("  COMPARISON: When to Use Which Pattern")
    print(f"{'='*60}")
    print("""
  SHARED STATE (LangGraph default):
    Best for: Simple pipelines, 2-4 agents, prototyping
    Risk:     State bloat, accidental overwrites
    Use when: "Just get it working"
  MESSAGE PASSING (ADK natural fit):
    Best for: Complex workflows, many agents, audit requirements
    Risk:     Routing complexity, missed messages
    Use when: "I need to know exactly who said what to whom"
  BLACKBOARD (hybrid):
    Best for: Structured analysis, role-based teams, access control
    Risk:     Rigid structure, upfront design required
    Use when: "Different agents own different data sections"

  Progression:
    1. Start with SHARED STATE (simplest)
    2. If agents overwrite each other -> switch to BLACKBOARD
    3. If you need audit trails / complex routing -> MESSAGE PASSING
""")

    # ============================================================
    # ADK-Specific Insights
    # ============================================================
    print(f"{'='*60}")
    print("  ADK vs LangChain: Communication Pattern Fit")
    print(f"{'='*60}")
    print("""
  SHARED STATE:   Both frameworks handle equally well.
  MESSAGE PASSING: ADK has a natural advantage -- Runner.run_async
                   yields events like a message stream, and
                   InMemorySessionService acts like a built-in inbox.
  BLACKBOARD:     Both frameworks handle equally well.

  Key Takeaway:
    - LangGraph excels at SHARED STATE (it's the default graph state)
    - ADK excels at MESSAGE PASSING (events are messages)
    - BLACKBOARD works equally well in both frameworks
    - Choose the pattern that fits your problem, then pick the
      framework that makes that pattern easiest.
""")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
