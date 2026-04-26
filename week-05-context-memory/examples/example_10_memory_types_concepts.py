import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 10: Memory Types — Short-term (Session) vs Long-term (Persistent)
==========================================================================
Pure-Python concept demonstration.

Covers:
  1. Short-term (Session) vs Long-term (Persistent) Memory
  2. Vector Store Trade-offs & Production Choices

Agents need memory to maintain context across turns and sessions.
Without memory, every turn is a fresh start — the agent can't recall
user preferences, past decisions, or learned facts.

Run: python week-05-context-memory/examples/example_10_memory_types_concepts.py
"""

import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta


# ================================================================
# 1. SHORT-TERM (SESSION) vs LONG-TERM (PERSISTENT) MEMORY
# ================================================================
# SHORT-TERM MEMORY:
#   - Lives within a single conversation session
#   - Stores: recent messages, tool results, working variables
#   - Implemented as: in-memory list/dict, session state
#   - Analogy: your working memory while solving a math problem
#
# LONG-TERM MEMORY:
#   - Persists across sessions (days, weeks, months)
#   - Stores: user preferences, learned facts, past interactions
#   - Implemented as: database, vector store, JSON file
#   - Analogy: your long-term knowledge and experiences

@dataclass
class MemoryEntry:
    """A single memory entry with metadata for management."""
    content: str
    memory_type: str              # "short_term" or "long_term"
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5       # 0-1 score (higher = more important)
    source: str = "conversation"  # Where this memory came from
    tags: List[str] = field(default_factory=list)

    def access(self):
        """Record an access to this memory (updates recency and count)."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def staleness_seconds(self) -> float:
        return (datetime.now() - self.last_accessed).total_seconds()


class ShortTermMemory:
    """
    Session-scoped memory that lives only during one conversation.

    Key properties:
    - Fast (in-memory, O(1) access)
    - Limited capacity (typically 10-50 items)
    - FIFO eviction when capacity is reached
    - No persistence — lost when session ends

    Production use: conversation buffer, tool result cache,
    working variables for multi-step reasoning.
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.entries: List[MemoryEntry] = []

    def add(self, content: str, importance: float = 0.5,
            tags: Optional[List[str]] = None) -> None:
        """Add a memory entry, evicting oldest if at capacity."""
        entry = MemoryEntry(
            content=content,
            memory_type="short_term",
            importance=importance,
            tags=tags or [],
        )
        if len(self.entries) >= self.capacity:
            # Evict: prefer low-importance, then oldest
            self.entries.sort(key=lambda e: (e.importance, -e.age_seconds))
            evicted = self.entries.pop(0)
            print(f"    [STM] Evicted: '{evicted.content[:40]}...' "
                  f"(importance={evicted.importance})")
        self.entries.append(entry)

    def recall(self, query: str = "", top_k: int = 5) -> List[MemoryEntry]:
        """Recall recent memories, optionally filtered by query."""
        results = self.entries
        if query:
            query_words = set(query.lower().split())
            results = [e for e in results
                       if query_words & set(e.content.lower().split())]
        # Return most recent first
        results = sorted(results, key=lambda e: e.created_at, reverse=True)
        for e in results[:top_k]:
            e.access()
        return results[:top_k]

    def clear(self):
        """Clear all short-term memory (session end)."""
        self.entries.clear()

    def stats(self) -> Dict:
        return {
            "count": len(self.entries),
            "capacity": self.capacity,
            "utilization": f"{len(self.entries) / self.capacity:.0%}",
        }


class LongTermMemory:
    """
    Persistent memory that survives across sessions.

    Key properties:
    - Slower access (database/file I/O)
    - Large capacity (thousands of entries)
    - Importance-based retention (forgetting curve)
    - Survives restarts

    Production use: user preferences, learned facts, past research
    results, agent skills and procedures.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.entries: List[MemoryEntry] = []
        self.storage_path = storage_path  # JSON file for persistence

    def store(self, content: str, importance: float = 0.5,
              tags: Optional[List[str]] = None, source: str = "conversation") -> None:
        """Store a memory for long-term retention."""
        # Check for duplicates before storing
        for existing in self.entries:
            overlap = set(content.lower().split()) & set(existing.content.lower().split())
            if len(overlap) / max(len(content.split()), 1) > 0.8:
                print(f"    [LTM] Duplicate detected, updating existing entry")
                existing.content = content
                existing.last_accessed = datetime.now()
                existing.importance = max(existing.importance, importance)
                return

        entry = MemoryEntry(
            content=content,
            memory_type="long_term",
            importance=importance,
            tags=tags or [],
            source=source,
        )
        self.entries.append(entry)

    def recall(self, query: str = "", top_k: int = 5) -> List[MemoryEntry]:
        """Recall memories relevant to the query."""
        if not query:
            # Return most important memories
            ranked = sorted(self.entries, key=lambda e: e.importance, reverse=True)
        else:
            # Simple keyword matching (production: use vector similarity)
            query_words = set(query.lower().split())
            ranked = []
            for e in self.entries:
                content_words = set(e.content.lower().split())
                overlap = len(query_words & content_words)
                relevance = overlap / max(len(query_words), 1)
                ranked.append((relevance, e))
            ranked = [e for _, e in sorted(ranked, key=lambda x: x[0], reverse=True)]

        for e in ranked[:top_k]:
            e.access()
        return ranked[:top_k]

    def forget(self, min_importance: float = 0.0,
               max_age_days: Optional[int] = None) -> int:
        """
        Forget low-importance or old memories.

        This implements a FORGETTING POLICY — essential for preventing
        memory bloat.  Without forgetting, long-term memory grows
        unbounded and retrieval quality degrades.
        """
        before = len(self.entries)
        now = datetime.now()

        self.entries = [
            e for e in self.entries
            if e.importance >= min_importance and (
                max_age_days is None or
                (now - e.created_at).days <= max_age_days
            )
        ]
        forgotten = before - len(self.entries)
        return forgotten

    def stats(self) -> Dict:
        avg_importance = (sum(e.importance for e in self.entries) / len(self.entries)
                          if self.entries else 0)
        return {
            "count": len(self.entries),
            "avg_importance": f"{avg_importance:.2f}",
            "sources": list(set(e.source for e in self.entries)),
        }


def demo_memory_types():
    """Demonstrate short-term vs long-term memory."""

    print("=" * 65)
    print("  SHORT-TERM vs LONG-TERM MEMORY")
    print("=" * 65)

    # Short-term memory demo
    print("\n  ── Short-Term Memory (session-scoped) ──")
    stm = ShortTermMemory(capacity=5)

    messages = [
        ("I'm planning a trip to Japan", 0.7, ["travel", "japan"]),
        ("Budget is $3000 for 2 weeks", 0.8, ["budget", "travel"]),
        ("I prefer vegetarian food", 0.9, ["preferences", "food"]),
        ("Hotels over hostels please", 0.6, ["preferences", "lodging"]),
        ("I speak some Japanese", 0.4, ["skills"]),
        ("I want to visit Kyoto temples", 0.7, ["destination", "kyoto"]),
    ]

    for content, importance, tags in messages:
        stm.add(content, importance, tags)
        print(f"    Added: '{content}' (importance={importance})")

    print(f"\n    STM Stats: {stm.stats()}")
    print(f"    Recalling 'food preferences':")
    for m in stm.recall("food preferences", top_k=2):
        print(f"      → '{m.content}' (importance={m.importance})")

    # Long-term memory demo
    print(f"\n  ── Long-Term Memory (persistent) ──")
    ltm = LongTermMemory()

    facts = [
        ("User prefers vegetarian restaurants", 0.9, ["preferences"]),
        ("User's budget is typically $2000-3000", 0.8, ["budget"]),
        ("User visited Paris in 2024", 0.5, ["history"]),
        ("User has a nut allergy", 0.95, ["health", "critical"]),
        ("User likes historical sites", 0.7, ["preferences"]),
    ]

    for content, importance, tags in facts:
        ltm.store(content, importance, tags, source="conversation")
        print(f"    Stored: '{content}' (importance={importance})")

    print(f"\n    LTM Stats: {ltm.stats()}")
    print(f"    Recalling 'food dietary':")
    for m in ltm.recall("food dietary", top_k=2):
        print(f"      → '{m.content}' (importance={m.importance})")


# ================================================================
# 2. VECTOR STORE TRADE-OFFS & PRODUCTION CHOICES
# ================================================================

def demo_vector_store_tradeoffs():
    """Compare memory storage options for production."""

    print("\n" + "=" * 65)
    print("  VECTOR STORE TRADE-OFFS & PRODUCTION CHOICES")
    print("=" * 65)

    print("""
  For SHORT-TERM memory:
  ┌──────────────────┬──────────┬───────────┬──────────────────┐
  │ Option           │ Speed    │ Capacity  │ Best For         │
  ├──────────────────┼──────────┼───────────┼──────────────────┤
  │ Python list/dict │ <1ms     │ ~1000     │ Prototypes       │
  │ Redis            │ <5ms     │ ~100K     │ Multi-process    │
  │ LangGraph state  │ <1ms     │ ~1000     │ Graph workflows  │
  │ ADK session      │ <1ms     │ ~1000     │ ADK agents       │
  └──────────────────┴──────────┴───────────┴──────────────────┘

  For LONG-TERM memory:
  ┌──────────────────┬──────────┬───────────┬──────────────────┐
  │ Option           │ Speed    │ Capacity  │ Best For         │
  ├──────────────────┼──────────┼───────────┼──────────────────┤
  │ JSON file        │ ~50ms    │ ~10K      │ Single-user      │
  │ SQLite           │ ~10ms    │ ~1M       │ Local apps       │
  │ FAISS + pickle   │ ~5ms     │ ~10M      │ Vector search    │
  │ ChromaDB         │ ~20ms    │ ~5M       │ Small production │
  │ PostgreSQL       │ ~15ms    │ Unlimited │ Multi-user prod  │
  │ Pinecone         │ ~50ms    │ Billions  │ Enterprise scale │
  └──────────────────┴──────────┴───────────┴───────────────���──┘

  Decision framework:
    1. Single user, prototype → JSON file or Python dict
    2. Single user, production → SQLite or ChromaDB
    3. Multi-user, production → PostgreSQL + pgvector
    4. Enterprise scale → Pinecone or Weaviate

  KEY PRINCIPLE: Start simple.  Most agents don't need a cloud vector
  database.  A JSON file with importance scoring and a forgetting
  policy handles thousands of memories effectively.
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 10: Memory Types                           ║")
    print("╚" + "═" * 63 + "╝")

    demo_memory_types()
    demo_vector_store_tradeoffs()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Short-term memory = within-session buffer.  Fast, limited,
       automatically cleared.  Use for conversation context and
       tool result caching.

    2. Long-term memory = cross-session persistence.  Slower, larger,
       needs explicit forgetting policies.  Use for user preferences,
       learned facts, and past interaction summaries.

    3. Both types need IMPORTANCE SCORING to decide what to keep and
       what to evict.  Not all memories are equal.

    4. Deduplication prevents memory bloat — check before storing.

    5. Start with simple storage (JSON/dict).  Scale to vector stores
       or databases only when you need semantic search or multi-user
       access.
    """))
