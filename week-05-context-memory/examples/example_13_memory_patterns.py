import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 13: Memory Patterns — Hierarchical Memory & Agent Buffers
====================================================================
Python + LangGraph demonstration of production memory patterns.

Covers:
  1. Hierarchical Memory (Summary → Detail Layers)
  2. Agent-Specific Buffers & Shared vs Private Memory
  3. Multi-Agent Memory Patterns

These patterns go beyond single-strategy memory (buffer, summary) to
composable, multi-layer memory architectures used in production agents.

Run: python week-05-context-memory/examples/example_13_memory_patterns.py
"""

import os
import sys
import textwrap
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypedDict
from datetime import datetime

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# 1. HIERARCHICAL MEMORY (L1 → L2 → L3)
# ================================================================
# Inspired by CPU cache hierarchy (L1/L2/L3), hierarchical memory
# provides graduated detail levels:
#
#   L1 (Hot Cache):  Last N messages, full text, fast access
#   L2 (Summary):    Summarized blocks of older messages
#   L3 (Archive):    Key facts and decisions extracted from history
#
# As messages age, they flow from L1 → L2 → L3, getting more
# compressed at each level.  This preserves ALL information while
# keeping the active context small.

@dataclass
class L1Cache:
    """
    L1: Hot cache — most recent messages in full text.

    Analogy: CPU L1 cache — fastest, smallest, most recent.
    """
    messages: List[Dict] = field(default_factory=list)
    max_messages: int = 10

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role, "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def is_full(self) -> bool:
        return len(self.messages) >= self.max_messages

    def drain(self, keep: int = 4) -> List[Dict]:
        """Remove old messages, keeping the most recent `keep`."""
        if len(self.messages) <= keep:
            return []
        drained = self.messages[:-keep]
        self.messages = self.messages[-keep:]
        return drained

    @property
    def token_estimate(self) -> int:
        return sum(len(m["content"].split()) for m in self.messages)


@dataclass
class L2Summary:
    """
    L2: Summary layer — compressed blocks of older messages.

    Each block is a time-bounded summary of a conversation segment.
    """
    summaries: List[Dict] = field(default_factory=list)
    max_summaries: int = 10

    def add_summary(self, text: str, message_count: int):
        self.summaries.append({
            "summary": text,
            "message_count": message_count,
            "created_at": datetime.now().isoformat(),
        })

    def is_full(self) -> bool:
        return len(self.summaries) >= self.max_summaries

    def drain(self) -> List[Dict]:
        """Drain old summaries for L3 archival."""
        if len(self.summaries) <= 2:
            return []
        drained = self.summaries[:-2]
        self.summaries = self.summaries[-2:]
        return drained


@dataclass
class L3Archive:
    """
    L3: Archive — extracted key facts and decisions.

    The most compressed form: just the essential facts that should
    never be forgotten (user preferences, critical decisions, etc.)
    """
    facts: List[str] = field(default_factory=list)

    def add_facts(self, new_facts: List[str]):
        for fact in new_facts:
            if fact not in self.facts:
                self.facts.append(fact)


class HierarchicalMemory:
    """
    Three-tier memory system: L1 (full) → L2 (summary) → L3 (facts).

    Flow:
      New messages → L1 (full text)
      When L1 full → summarize old messages → L2 (summary)
      When L2 full → extract key facts → L3 (archive)

    Context building: L3 facts + L2 summaries + L1 recent messages
    """

    def __init__(self, l1_size: int = 10, l2_size: int = 10):
        self.l1 = L1Cache(max_messages=l1_size)
        self.l2 = L2Summary(max_summaries=l2_size)
        self.l3 = L3Archive()

    def add_message(self, role: str, content: str):
        """Add a message, triggering compaction if needed."""
        self.l1.add(role, content)

        # L1 → L2 compaction
        if self.l1.is_full():
            drained = self.l1.drain(keep=4)
            if drained:
                # Simulate summarization (production: LLM call)
                summary = "Summary: " + " | ".join(
                    m["content"][:40] for m in drained
                )
                self.l2.add_summary(summary, len(drained))
                print(f"    [L1→L2] Compacted {len(drained)} messages → summary")

        # L2 → L3 compaction
        if self.l2.is_full():
            drained = self.l2.drain()
            if drained:
                # Extract facts (production: LLM extraction)
                facts = [s["summary"][:60] + "..." for s in drained]
                self.l3.add_facts(facts)
                print(f"    [L2→L3] Archived {len(drained)} summaries → {len(facts)} facts")

    def build_context(self) -> str:
        """Build context from all three layers."""
        parts = []

        # L3: Core facts (always included)
        if self.l3.facts:
            parts.append("Key facts:\n" + "\n".join(f"  • {f}" for f in self.l3.facts))

        # L2: Recent summaries
        if self.l2.summaries:
            parts.append("Conversation summaries:\n" + "\n".join(
                f"  {s['summary']}" for s in self.l2.summaries[-3:]
            ))

        # L1: Recent messages (full text)
        if self.l1.messages:
            parts.append("Recent messages:\n" + "\n".join(
                f"  {m['role']}: {m['content']}" for m in self.l1.messages
            ))

        return "\n\n".join(parts)

    def stats(self) -> Dict:
        return {
            "L1 messages": len(self.l1.messages),
            "L2 summaries": len(self.l2.summaries),
            "L3 facts": len(self.l3.facts),
        }


def demo_hierarchical_memory():
    """Demonstrate the L1→L2→L3 compaction flow."""

    print("=" * 65)
    print("  HIERARCHICAL MEMORY (L1 → L2 → L3)")
    print("=" * 65)

    mem = HierarchicalMemory(l1_size=6, l2_size=4)

    # Simulate a long conversation
    messages = [
        ("user", "I want to plan a trip to Japan"),
        ("assistant", "Japan is great! When are you thinking of going?"),
        ("user", "Next spring, I have a budget of $3000"),
        ("assistant", "Spring is perfect for cherry blossoms. $3000 is workable."),
        ("user", "I'm vegetarian with a nut allergy"),
        ("assistant", "Good to know. Japan has great vegetarian options."),
        ("user", "Should I get a Japan Rail Pass?"),
        ("assistant", "Yes, the 14-day JR Pass is ¥50,000 and covers most trains."),
        ("user", "What about hotels in Kyoto?"),
        ("assistant", "Budget hotels in Kyoto range from $40-80/night."),
        ("user", "Can you suggest a 3-day itinerary for Kyoto?"),
        ("assistant", "Day 1: Fushimi Inari, Day 2: Arashiyama, Day 3: Eastern temples."),
        ("user", "What about Osaka day trips?"),
        ("assistant", "Osaka is 15 min from Kyoto by train. Visit Dotonbori and castle."),
    ]

    for role, content in messages:
        mem.add_message(role, content)
        print(f"  Added: {role}: {content[:50]}...")

    print(f"\n  Stats: {mem.stats()}")
    print(f"\n  Built context:")
    print(f"  {mem.build_context()}")


# ================================================================
# 2 & 3. AGENT-SPECIFIC BUFFERS & MULTI-AGENT MEMORY PATTERNS
# ================================================================

def demo_multi_agent_memory():
    """Show shared vs private memory in multi-agent systems."""

    print("\n" + "=" * 65)
    print("  MULTI-AGENT MEMORY PATTERNS")
    print("=" * 65)

    print("""
  In multi-agent systems, memory can be:

  PRIVATE (agent-specific buffer):
  ┌────────────────────────────────────────┐
  │  Researcher Agent                       │
  │  ┌─────────────────────┐               │
  │  │ Private Memory:      │               │
  │  │ - Search history     │               │
  │  │ - Source evaluations │               │
  │  │ - Draft notes        │               │
  │  └─────────────────────┘               │
  └────────────────────────────────────────┘

  SHARED (accessible by all agents):
  ┌────────────────────────────────────────┐
  │  Shared Workspace                       │
  │  ┌─────────────────────┐               │
  │  │ Shared Memory:       │               │
  │  │ - User preferences   │               │
  │  │ - Task status         │               │
  │  │ - Final outputs      │               │
  │  └─────────────────────┘               │
  │  Used by: Researcher, Writer, Reviewer │
  └────────────────────────────────────────┘

  PATTERNS:

  1. BLACKBOARD PATTERN:
     All agents write to a shared "blackboard" state.
     Each agent reads what others wrote and adds its contribution.
     → Use LangGraph shared state (TypedDict with all fields)

  2. MESSAGE PASSING:
     Agents communicate by sending messages through a supervisor.
     Each agent has private memory; shared info goes through messages.
     → Use LangGraph with supervisor node routing messages

  3. HIERARCHICAL:
     Supervisor has full memory access; workers get scoped views.
     Workers can't see each other's private memory.
     → Use state fields with access control in node logic""")

    # Simulated multi-agent state
    print(f"\n  ── Simulated Multi-Agent State ──")

    shared_state = {
        "task": "Research AI safety regulations",
        "user_preferences": {"detail_level": "high", "format": "report"},
        "status": "in_progress",
    }

    agent_buffers = {
        "researcher": {
            "search_queries": ["EU AI Act 2025", "US AI Executive Order"],
            "sources_found": 5,
            "quality_scores": [0.9, 0.85, 0.78, 0.72, 0.65],
        },
        "writer": {
            "outline": ["Intro", "EU Regulations", "US Regulations", "Comparison"],
            "current_section": "EU Regulations",
            "word_count": 450,
        },
        "reviewer": {
            "feedback_items": ["Add more specific dates", "Missing citation for claim X"],
            "overall_score": 7.5,
        },
    }

    print(f"\n  Shared State (all agents can read/write):")
    for k, v in shared_state.items():
        print(f"    {k}: {v}")

    for agent, buffer in agent_buffers.items():
        print(f"\n  {agent.title()} Private Buffer:")
        for k, v in buffer.items():
            print(f"    {k}: {v}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 13: Memory Patterns                        ║")
    print("╚" + "═" * 63 + "╝")

    demo_hierarchical_memory()
    demo_multi_agent_memory()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Hierarchical memory (L1→L2→L3) provides graduated compression:
       recent messages in full, older ones as summaries, oldest as facts.

    2. The compaction flow is automatic: when L1 fills, it drains to L2;
       when L2 fills, it drains to L3.  No manual intervention needed.

    3. Multi-agent systems need both SHARED memory (task state, user
       prefs) and PRIVATE buffers (agent-specific working data).

    4. Three multi-agent memory patterns: Blackboard (shared state),
       Message Passing (supervisor routes), Hierarchical (scoped access).

    5. In LangGraph, implement these patterns using TypedDict state
       with fields for each memory layer and each agent's buffer.
    """))
