import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 4: Multi-Agent Collaboration — Concepts (No LLM Required)
==================================================================
How do multiple agents work together to solve problems no single agent
could handle alone?

Think of a newspaper. A single journalist COULD write, photograph, edit,
and print an entire paper — but the result would be mediocre at everything.
Real newspapers have specialized roles: reporters gather facts, photographers
capture images, editors refine prose, and printers handle production.
Each person is excellent at ONE thing, and they coordinate through
well-defined handoffs.

Multi-agent systems work exactly the same way:
  1. TOPOLOGIES define who talks to whom (pipeline, hierarchy, debate)
  2. COMMUNICATION PATTERNS define how they share information
  3. SPECIALIZATION means each agent has focused expertise

This example covers:
  - Three multi-agent topologies (sequential, supervisor/worker, debate)
  - Three communication patterns (shared state, message passing, blackboard)
  - Why multi-agent scales complexity (specialization, modularity, speed)
  - When NOT to use multi-agent (the "multi-agent trap")

Key Concepts (Topics 2, 11, 13 of Week 4):
  - Supervisor/Worker architecture
  - Agent communication and coordination
  - When multi-agent helps vs. hurts

Run: python week-04-advanced-patterns/examples/example_04_multi_agent_concepts.py
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List


# ================================================================
# PART 1: Three Multi-Agent Topologies
# ================================================================
# Before building multi-agent systems, you need to decide HOW agents
# relate to each other. There are three fundamental topologies:
#
# 1. SEQUENTIAL PIPELINE — assembly line, each agent transforms
#    output and passes it to the next.
#
# 2. HIERARCHICAL (SUPERVISOR/WORKER) — one boss delegates to
#    specialist workers and combines their results.
#
# 3. DEBATE/COMMITTEE — agents argue different perspectives,
#    a judge synthesizes the best answer.

def demo_topologies():
    """Visualize and explain the three fundamental multi-agent topologies."""
    print("=" * 60)
    print("PART 1: Three Multi-Agent Topologies")
    print("=" * 60)

    # ---- Topology 1: Sequential Pipeline ----
    print("\n--- Topology 1: Sequential Pipeline ---")
    print("""
    Think of a newspaper production line:

      Reporter --> Photographer --> Editor --> Printer
      (gathers     (adds           (refines    (formats
       facts)       visuals)        prose)      output)

    In an AI agent pipeline:

      ┌────────────┐    ┌──────────┐    ┌────────┐
      │ Researcher │───>│ Analyst  │───>│ Writer │
      │ (finds     │    │ (spots   │    │ (draft │
      │  data)     │    │  trends) │    │ report)│
      └────────────┘    └──────────┘    └────────┘

    Each agent receives the PREVIOUS agent's output as input.
    Simple, predictable, easy to debug — but no parallelism.
    """)
    print("  Best for: linear workflows where order matters")
    print("  Example:  research -> analyze -> write report")
    print("  Weakness: bottleneck at the slowest agent")

    # ---- Topology 2: Hierarchical (Supervisor/Worker) ----
    print("\n--- Topology 2: Hierarchical (Supervisor/Worker) ---")
    print("""
    Think of an editor-in-chief with specialized reporters:

                     ┌──────────────┐
                     │  Supervisor  │
                     │  (assigns    │
                     │   tasks,     │
                     │   combines)  │
                     └──┬───┬───┬───┘
                        │   │   │
               ┌────────┘   │   └────────┐
               ▼            ▼            ▼
        ┌────────────┐ ┌──────────┐ ┌────────┐
        │ Researcher │ │ Analyst  │ │ Writer │
        │ (facts)    │ │ (trends) │ │ (prose)│
        └────────────┘ └──────────┘ └────────┘

    The supervisor decides WHICH worker to call and WHEN.
    Workers report back to the supervisor, not to each other.
    """)
    print("  Best for: tasks requiring different expertise")
    print("  Example:  supervisor routes 'AI ethics' to researcher,")
    print("            routes findings to analyst, routes analysis to writer")
    print("  Weakness: supervisor is a single point of failure")

    # ---- Topology 3: Debate/Committee ----
    print("\n--- Topology 3: Debate/Committee ---")
    print("""
    Think of a panel of experts debating a topic:

        ┌──────────────┐   ┌──────────────┐
        │  Optimist    │   │  Skeptic     │
        │  Agent       │   │  Agent       │
        │  "AI will    │   │  "AI risks   │
        │   help!"     │   │   are real!" │
        └──────┬───────┘   └───────┬──────┘
               │                   │
               └─────────┬─────────┘
                         ▼
                 ┌──────────────┐
                 │    Judge     │
                 │  (weighs     │
                 │   arguments, │
                 │   decides)   │
                 └──────────────┘

    Multiple agents present DIFFERENT perspectives on the same
    question. A judge agent evaluates arguments and synthesizes.
    """)
    print("  Best for: nuanced decisions with trade-offs")
    print("  Example:  'Should we deploy AI in healthcare?'")
    print("  Weakness: expensive (multiple agents per question)")
    print()


# ================================================================
# PART 2: Communication Pattern 1 — Shared State
# ================================================================
# The simplest pattern: all agents read from and write to a
# shared dictionary. Like a team writing on the same whiteboard.
#
# Pros: simple, easy to debug, agents see all context
# Cons: no privacy, ordering matters, race conditions in async

class SharedStateDemo:
    """Agents communicate by reading/writing a shared dictionary.

    Think of a shared Google Doc: any team member can read what
    others wrote and add their own section.
    """

    def __init__(self):
        self.state = {
            "topic": "",
            "findings": [],
            "analysis": "",
            "report": "",
        }

    def researcher(self, topic: str):
        """Gather facts about the topic and store in shared state."""
        print(f"  [Researcher] Investigating: {topic}")
        self.state["topic"] = topic
        self.state["findings"] = [
            "AI ethics guidelines exist in 40+ countries",
            "Key concerns: bias, privacy, accountability, transparency",
            "UNESCO adopted global AI ethics recommendation in 2021",
            "EU AI Act classifies AI systems by risk level",
        ]
        print(f"  [Researcher] Added {len(self.state['findings'])} findings to shared state")

    def analyst(self):
        """Read findings from shared state, write analysis."""
        findings = self.state["findings"]
        print(f"  [Analyst] Reading {len(findings)} findings from shared state")
        self.state["analysis"] = (
            f"Analysis of {self.state['topic']}: "
            f"Global regulatory momentum is strong ({len(findings)} key data points). "
            "The field is moving from voluntary guidelines toward binding legislation. "
            "The EU AI Act represents the most comprehensive framework to date."
        )
        print(f"  [Analyst] Wrote analysis ({len(self.state['analysis'])} chars) to shared state")

    def writer(self):
        """Read analysis from shared state, write final report."""
        analysis = self.state["analysis"]
        print(f"  [Writer] Reading analysis from shared state")
        self.state["report"] = (
            f"REPORT: {self.state['topic'].upper()}\n"
            f"{'=' * 40}\n"
            f"Key Findings:\n"
        )
        for i, finding in enumerate(self.state["findings"], 1):
            self.state["report"] += f"  {i}. {finding}\n"
        self.state["report"] += (
            f"\nExpert Analysis:\n"
            f"  {analysis}\n"
            f"\nConclusion: AI ethics governance is rapidly maturing worldwide."
        )
        print(f"  [Writer] Wrote final report ({len(self.state['report'])} chars) to shared state")


def demo_shared_state():
    """Run the shared state communication pattern."""
    print("=" * 60)
    print("PART 2a: Communication Pattern — Shared State")
    print("=" * 60)
    print("\nAll agents read/write the SAME dictionary.")
    print("Like a team sharing one whiteboard.\n")

    demo = SharedStateDemo()

    # Sequential pipeline using shared state
    print("Step 1: Researcher gathers facts")
    demo.researcher("AI ethics")

    print("\nStep 2: Analyst processes findings")
    demo.analyst()

    print("\nStep 3: Writer creates report")
    demo.writer()

    print("\n--- Final Shared State ---")
    print(f"  Topic:    {demo.state['topic']}")
    print(f"  Findings: {len(demo.state['findings'])} items")
    print(f"  Analysis: {len(demo.state['analysis'])} chars")
    print(f"  Report:   {len(demo.state['report'])} chars")
    print(f"\n--- Generated Report ---\n{demo.state['report']}")
    print()


# ================================================================
# PART 2b: Communication Pattern 2 — Message Passing
# ================================================================
# Agents send typed messages to each other through an inbox system.
# Like email: each agent has its own inbox and sends explicit messages.
#
# Pros: clear data flow, agents are decoupled, easy to trace
# Cons: more boilerplate, agents must know who to send to

@dataclass
class AgentMessage:
    """A typed message between agents.

    Fields:
      sender:   who sent it
      receiver: who should read it
      content:  the actual data
      msg_type: category of message (findings, analysis, report)
    """
    sender: str
    receiver: str
    content: str
    msg_type: str  # "findings", "analysis", "report"


class MessagePassingDemo:
    """Agents communicate by sending explicit messages to each other.

    Think of email: each agent has an inbox, and messages are
    explicitly addressed to a recipient.
    """

    def __init__(self):
        self.inbox: dict = defaultdict(list)
        self.message_log: list = []

    def send(self, msg: AgentMessage):
        """Deliver a message to the receiver's inbox."""
        self.inbox[msg.receiver].append(msg)
        self.message_log.append(msg)
        print(f"  [MessageBus] {msg.sender} -> {msg.receiver}: "
              f"({msg.msg_type}) {msg.content[:60]}...")

    def receive(self, agent_name: str) -> List[AgentMessage]:
        """Retrieve all messages for an agent (clears the inbox)."""
        messages = self.inbox[agent_name]
        self.inbox[agent_name] = []
        return messages

    def researcher(self, topic: str):
        """Research a topic and send findings to the analyst."""
        print(f"  [Researcher] Investigating: {topic}")
        findings = [
            "AI ethics guidelines exist in 40+ countries",
            "Key concerns: bias, privacy, accountability, transparency",
            "UNESCO adopted global AI ethics recommendation in 2021",
            "EU AI Act classifies AI systems by risk level",
        ]
        # Send each finding as a separate message
        for finding in findings:
            self.send(AgentMessage(
                sender="researcher",
                receiver="analyst",
                content=finding,
                msg_type="findings",
            ))
        # Notify analyst that research is complete
        self.send(AgentMessage(
            sender="researcher",
            receiver="analyst",
            content="Research complete. All findings sent.",
            msg_type="control",
        ))

    def analyst(self):
        """Read messages from researcher, send analysis to writer."""
        messages = self.receive("analyst")
        findings = [m.content for m in messages if m.msg_type == "findings"]
        print(f"  [Analyst] Received {len(findings)} findings from inbox")

        analysis = (
            f"Analysis based on {len(findings)} data points: "
            "Global regulatory momentum is strong. "
            "The field is moving from voluntary guidelines toward binding legislation. "
            "The EU AI Act represents the most comprehensive framework to date."
        )
        self.send(AgentMessage(
            sender="analyst",
            receiver="writer",
            content=analysis,
            msg_type="analysis",
        ))
        # Also forward the raw findings for the writer
        self.send(AgentMessage(
            sender="analyst",
            receiver="writer",
            content=" | ".join(findings),
            msg_type="findings_summary",
        ))

    def writer(self):
        """Read messages from analyst, produce the final report."""
        messages = self.receive("writer")
        analysis = next((m.content for m in messages if m.msg_type == "analysis"), "")
        findings_str = next((m.content for m in messages if m.msg_type == "findings_summary"), "")
        findings = findings_str.split(" | ") if findings_str else []
        print(f"  [Writer] Received analysis + {len(findings)} findings from inbox")

        report = "REPORT: AI ETHICS\n" + "=" * 40 + "\n"
        report += "Key Findings:\n"
        for i, f in enumerate(findings, 1):
            report += f"  {i}. {f}\n"
        report += f"\nExpert Analysis:\n  {analysis}\n"
        report += "\nConclusion: AI ethics governance is rapidly maturing worldwide."

        self.send(AgentMessage(
            sender="writer",
            receiver="output",
            content=report,
            msg_type="report",
        ))


def demo_message_passing():
    """Run the message passing communication pattern."""
    print("=" * 60)
    print("PART 2b: Communication Pattern — Message Passing")
    print("=" * 60)
    print("\nAgents send explicit typed messages to each other.")
    print("Like email: each agent has an inbox.\n")

    demo = MessagePassingDemo()

    print("Step 1: Researcher sends findings to Analyst")
    demo.researcher("AI ethics")

    print("\nStep 2: Analyst reads inbox, sends analysis to Writer")
    demo.analyst()

    print("\nStep 3: Writer reads inbox, produces report")
    demo.writer()

    # Show the full message trace
    print(f"\n--- Message Trace ({len(demo.message_log)} messages) ---")
    for i, msg in enumerate(demo.message_log, 1):
        print(f"  {i}. [{msg.msg_type:>16}] {msg.sender:>10} -> {msg.receiver:<10}")

    # Show the final report
    final = demo.receive("output")
    if final:
        print(f"\n--- Generated Report ---\n{final[0].content}")
    print()


# ================================================================
# PART 2c: Communication Pattern 3 — Blackboard
# ================================================================
# A structured workspace with TYPED SLOTS. Agents contribute to
# specific sections. Like a lab notebook with labeled sections
# that different scientists fill in.
#
# Pros: structured, clear responsibilities, prevents conflicts
# Cons: rigid structure, must be designed upfront

class BlackboardDemo:
    """Agents communicate through a structured blackboard with typed slots.

    Think of a lab notebook with labeled sections:
      - "Hypotheses" section (researcher writes here)
      - "Evidence" section (researcher writes here)
      - "Analysis" section (analyst writes here)
      - "Conclusion" section (writer writes here)

    Each agent knows which slots it READS from and WRITES to.
    """

    def __init__(self):
        self.board = {
            "hypotheses": [],
            "evidence": [],
            "analysis": "",
            "conclusion": "",
        }
        self.access_log: list = []

    def _log_access(self, agent: str, slot: str, action: str):
        """Track who accessed what slot and how."""
        self.access_log.append((agent, slot, action))
        print(f"  [Blackboard] {agent} {action} slot '{slot}'")

    def researcher(self, topic: str):
        """Post hypotheses and evidence to the blackboard."""
        print(f"  [Researcher] Investigating: {topic}")

        # Write hypotheses
        self.board["hypotheses"] = [
            "Global AI regulation is accelerating",
            "Ethical AI requires both technical and governance solutions",
        ]
        self._log_access("researcher", "hypotheses", "WROTE")

        # Write evidence
        self.board["evidence"] = [
            "AI ethics guidelines exist in 40+ countries",
            "Key concerns: bias, privacy, accountability, transparency",
            "UNESCO adopted global AI ethics recommendation in 2021",
            "EU AI Act classifies AI systems by risk level",
        ]
        self._log_access("researcher", "evidence", "WROTE")

    def analyst(self):
        """Read hypotheses and evidence, write analysis."""
        hypotheses = self.board["hypotheses"]
        evidence = self.board["evidence"]
        self._log_access("analyst", "hypotheses", "READ")
        self._log_access("analyst", "evidence", "READ")

        print(f"  [Analyst] Evaluating {len(hypotheses)} hypotheses "
              f"against {len(evidence)} pieces of evidence")

        self.board["analysis"] = (
            f"Both hypotheses are SUPPORTED by evidence. "
            f"Hypothesis 1 ('regulation accelerating') is backed by "
            f"{len(evidence)} data points showing global legislative action. "
            f"Hypothesis 2 ('technical + governance needed') is supported by "
            f"the breadth of concerns spanning technical (bias) and "
            f"governance (accountability) domains."
        )
        self._log_access("analyst", "analysis", "WROTE")

    def writer(self):
        """Read all slots, write conclusion."""
        self._log_access("writer", "hypotheses", "READ")
        self._log_access("writer", "evidence", "READ")
        self._log_access("writer", "analysis", "READ")

        report = "REPORT: AI ETHICS\n" + "=" * 40 + "\n"
        report += "Hypotheses:\n"
        for i, h in enumerate(self.board["hypotheses"], 1):
            report += f"  H{i}: {h}\n"
        report += "\nEvidence:\n"
        for i, e in enumerate(self.board["evidence"], 1):
            report += f"  E{i}: {e}\n"
        report += f"\nAnalysis:\n  {self.board['analysis']}\n"
        report += "\nConclusion: AI ethics governance is rapidly maturing worldwide."

        self.board["conclusion"] = report
        self._log_access("writer", "conclusion", "WROTE")


def demo_blackboard():
    """Run the blackboard communication pattern."""
    print("=" * 60)
    print("PART 2c: Communication Pattern — Blackboard")
    print("=" * 60)
    print("\nA structured workspace with typed slots.")
    print("Like a lab notebook with labeled sections.\n")

    demo = BlackboardDemo()

    print("Step 1: Researcher posts hypotheses and evidence")
    demo.researcher("AI ethics")

    print("\nStep 2: Analyst reads hypotheses + evidence, writes analysis")
    demo.analyst()

    print("\nStep 3: Writer reads everything, writes conclusion")
    demo.writer()

    # Show access pattern
    print(f"\n--- Blackboard Access Log ({len(demo.access_log)} accesses) ---")
    for agent, slot, action in demo.access_log:
        marker = ">>" if action == "WROTE" else "<<"
        print(f"  {marker} {agent:>10} {action:>5} '{slot}'")

    # Show final board state
    print("\n--- Final Blackboard Slots ---")
    for slot, value in demo.board.items():
        if isinstance(value, list):
            print(f"  {slot}: [{len(value)} items]")
        else:
            print(f"  {slot}: [{len(value)} chars]")

    print(f"\n--- Generated Report ---\n{demo.board['conclusion']}")
    print()


# ================================================================
# PART 3: Why Multi-Agent Scales Complexity
# ================================================================
# Multi-agent systems help along three axes. Each axis addresses
# a different scaling challenge.

def demo_why_multi_agent():
    """Show the three axes where multi-agent systems excel."""
    print("=" * 60)
    print("PART 3: Why Multi-Agent Scales Complexity")
    print("=" * 60)

    # ---- Axis 1: Specialization (Quality) ----
    print("\n--- Axis 1: Specialization -> Better Quality ---")
    print("""
    A single generalist prompt:
      "Research AI ethics, analyze the findings, and write a report"

    vs. three specialist prompts:
      Researcher: "Find the 5 most important facts about AI ethics.
                   Focus on recent legislation and official frameworks."
      Analyst:    "Given these facts, identify patterns, evaluate
                   strength of evidence, and note any contradictions."
      Writer:     "Given this analysis, write a 200-word executive
                   summary for a non-technical audience."

    The specialist prompts are:
      - More focused (each agent knows its exact job)
      - Easier to test (check researcher output independently)
      - Higher quality (an agent with one job does it better)
    """)

    # ---- Axis 2: Modularity (Maintainability) ----
    print("--- Axis 2: Modularity -> Easier Maintenance ---")
    print("""
    Need a better researcher? Swap just the researcher agent.
    The analyst and writer don't change at all.

      Before:  Researcher_v1 -> Analyst -> Writer
      After:   Researcher_v2 -> Analyst -> Writer
                  ^
                  Only this changed!

    In a monolithic single-agent system, improving one capability
    means rewriting the entire prompt and retesting everything.
    """)

    # ---- Axis 3: Parallelism (Speed) ----
    print("--- Axis 3: Parallelism -> Faster Execution ---")
    print("""
    Some sub-tasks are INDEPENDENT and can run simultaneously:

      Sequential (slow):
        Research AI ethics -> Research AI safety -> Research AI regulation
        Time: 3 * T = 3T

      Parallel (fast):
        Research AI ethics  ──┐
        Research AI safety  ──┼──> Combine -> Analyze -> Write
        Research AI regulation┘
        Time: T + T_combine + T_analyze + T_write << 3T

    The supervisor topology enables this naturally: the supervisor
    dispatches multiple workers in parallel and waits for all results.
    """)
    print()


# ================================================================
# PART 4: When NOT to Use Multi-Agent
# ================================================================
# The "multi-agent trap": using multi-agent for problems that
# don't need it. More agents = more complexity, more cost,
# more points of failure.

def demo_when_not_to_use():
    """Show the multi-agent trap and rules of thumb."""
    print("=" * 60)
    print("PART 4: When NOT to Use Multi-Agent")
    print("=" * 60)

    print("""
    THE MULTI-AGENT TRAP
    ====================

    Bad idea: using 3 agents to answer "What is 2 + 2?"

      ┌────────┐    ┌──────────┐    ┌──────────┐
      │ Parser │───>│ Calculator│───>│ Formatter│
      │ "2+2"  │    │ "= 4"    │    │ "Four."  │
      └────────┘    └──────────┘    └──────────┘

      Total cost: 3 LLM calls, 3x latency, 3x failure points
      A single agent could answer this in one call.

    RULE OF THUMB: Use multi-agent when you have:
      1. Genuinely DIFFERENT expertise needs
         (research vs. analysis vs. writing = different skills)
      2. Parallelizable sub-tasks
         (research 3 topics at the same time)
      3. Quality requirements that need checks and balances
         (draft + critique + revise loop)

    DO NOT use multi-agent when:
      1. A single prompt can handle the task well
      2. The "agents" would just pass data through unchanged
      3. The coordination overhead exceeds the quality gain
    """)

    # Concrete decision examples
    print("--- Decision Examples ---")
    decisions = [
        ("Summarize a 1-page email",
         "SINGLE agent",
         "Simple task, one skill needed"),
        ("Research a topic, analyze trends, write a report",
         "MULTI agent",
         "Three distinct skills, each benefits from specialization"),
        ("Translate English to French",
         "SINGLE agent",
         "One skill, no decomposition needed"),
        ("Build a research paper with literature review, methodology, results",
         "MULTI agent",
         "Different expertise per section, parallelizable"),
        ("Answer a factual question from a database",
         "SINGLE agent",
         "One tool call, no analysis chain needed"),
        ("Evaluate a business proposal from financial, legal, and market angles",
         "MULTI agent",
         "Three genuinely different expertise areas"),
    ]

    for task, verdict, reason in decisions:
        icon = "[MULTI]" if "MULTI" in verdict else "[SINGLE]"
        print(f"  {icon:>8}  {task}")
        print(f"           Reason: {reason}")
    print()


# ================================================================
# MAIN: Run All Demos
# ================================================================

if __name__ == "__main__":
    print()
    print("*" * 60)
    print("  MULTI-AGENT COLLABORATION CONCEPTS")
    print("  Pure Python — No LLM Required")
    print("*" * 60)
    print()

    # Part 1: Topologies
    demo_topologies()

    # Part 2: Communication Patterns
    demo_shared_state()
    demo_message_passing()
    demo_blackboard()

    # Part 3: Why it works
    demo_why_multi_agent()

    # Part 4: When not to use it
    demo_when_not_to_use()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. TOPOLOGIES define structure:
       - Sequential: simple pipeline, easy to debug
       - Supervisor/Worker: flexible delegation, enables parallelism
       - Debate/Committee: multiple perspectives, best for nuanced decisions

    2. COMMUNICATION defines data flow:
       - Shared State: simplest, all agents see everything
       - Message Passing: explicit, traceable, decoupled
       - Blackboard: structured slots, clear responsibilities

    3. Multi-agent SCALES along three axes:
       - Specialization (quality) — focused agents do better work
       - Modularity (maintainability) — swap one agent, keep the rest
       - Parallelism (speed) — independent tasks run concurrently

    4. Avoid the MULTI-AGENT TRAP:
       - Don't use 3 agents for a job 1 agent can handle
       - The coordination overhead must be worth the quality gain

    Next Examples:
      example_05  — Supervisor/Worker in LangGraph
      example_05b — Intent Routing (4 approaches: rule, embedding, LLM, cascading)
      example_05c — Pipeline & Debate topologies with LLM agents
      example_05d — Communication patterns (shared state, message passing, blackboard)
      example_06  — Supervisor/Worker in ADK
    """)
