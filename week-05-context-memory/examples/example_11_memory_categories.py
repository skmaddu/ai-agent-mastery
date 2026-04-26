import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 11: Episodic vs Semantic vs Procedural Memory
=======================================================
Pure-Python concept demonstration.

Covers:
  1. Episodic Memory (conversation history & events)
  2. Semantic Memory (facts & knowledge)
  3. Procedural Memory (skills, instructions, best practices)
  4. How to Store & Retrieve Each Type

Inspired by cognitive science, agent memory can be categorized into
three types — each with different storage, retrieval, and decay patterns.

Run: python week-05-context-memory/examples/example_11_memory_categories.py
"""

import textwrap
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# ================================================================
# 1. EPISODIC MEMORY (Conversation History & Events)
# ================================================================
# Episodic memory stores EVENTS and EXPERIENCES with temporal context.
# "What happened" — complete with who, what, when, where.
#
# Examples:
#   - "User asked about Japan hotels on March 5 and preferred Kyoto"
#   - "The API call to weather service failed at 3:42 PM"
#   - "User approved the meal plan after 2 revision rounds"
#
# Key properties:
#   - Timestamped (temporal order matters)
#   - Contextual (includes surrounding circumstances)
#   - Decays naturally (older episodes are less relevant)

@dataclass
class Episode:
    """A single episodic memory — a recorded event."""
    event: str                         # What happened
    context: str                       # Surrounding circumstances
    timestamp: datetime = field(default_factory=datetime.now)
    participants: List[str] = field(default_factory=list)  # Who was involved
    outcome: str = ""                  # How it ended
    emotional_valence: float = 0.0     # -1 (negative) to +1 (positive)
    importance: float = 0.5


class EpisodicMemory:
    """
    Stores and retrieves episodic memories (events and experiences).

    Retrieval strategies:
      - TEMPORAL: most recent episodes first
      - SIMILARITY: episodes similar to a query
      - EMOTIONAL: episodes with strong emotional valence
    """

    def __init__(self, max_episodes: int = 100):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes

    def record(self, event: str, context: str = "", participants: List[str] = None,
               outcome: str = "", importance: float = 0.5) -> None:
        """Record a new episodic memory."""
        episode = Episode(
            event=event, context=context,
            participants=participants or [],
            outcome=outcome, importance=importance,
        )
        self.episodes.append(episode)

        # Evict oldest low-importance episodes if over capacity
        if len(self.episodes) > self.max_episodes:
            self.episodes.sort(key=lambda e: e.importance)
            self.episodes.pop(0)

    def recall_recent(self, n: int = 5) -> List[Episode]:
        """Recall the N most recent episodes."""
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def recall_similar(self, query: str, n: int = 5) -> List[Episode]:
        """Recall episodes similar to a query (simple keyword matching)."""
        query_words = set(query.lower().split())
        scored = []
        for ep in self.episodes:
            words = set((ep.event + " " + ep.context).lower().split())
            overlap = len(query_words & words)
            scored.append((overlap, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n]]

    def to_context_string(self, n: int = 5) -> str:
        """Format recent episodes for injection into an LLM prompt."""
        recent = self.recall_recent(n)
        lines = []
        for ep in recent:
            ts = ep.timestamp.strftime("%H:%M")
            lines.append(f"[{ts}] {ep.event}")
            if ep.outcome:
                lines.append(f"  → Outcome: {ep.outcome}")
        return "\n".join(lines)


# ================================================================
# 2. SEMANTIC MEMORY (Facts & Knowledge)
# ================================================================
# Semantic memory stores FACTS and KNOWLEDGE without temporal context.
# "What is true" — general knowledge and learned information.
#
# Examples:
#   - "The user is allergic to nuts"
#   - "Japan Rail Pass costs ¥50,000 for 14 days"
#   - "RLHF stands for Reinforcement Learning from Human Feedback"
#
# Key properties:
#   - Timeless (facts don't have timestamps per se)
#   - Structured (often as subject-predicate-object triples)
#   - Updated, not appended (correct outdated facts)

@dataclass
class Fact:
    """A semantic memory — a known fact."""
    subject: str
    predicate: str
    obj: str                   # 'object' is a Python builtin, so 'obj'
    confidence: float = 0.8    # 0-1 confidence in this fact
    source: str = "conversation"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def triple(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"

    @property
    def natural(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}"


class SemanticMemory:
    """
    Stores and retrieves factual knowledge as subject-predicate-object triples.

    Supports CRUD operations on facts:
      - CREATE: add new facts
      - READ: query by subject, predicate, or keyword
      - UPDATE: modify existing facts when new info arrives
      - DELETE: remove incorrect facts
    """

    def __init__(self):
        self.facts: List[Fact] = []

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 0.8, source: str = "conversation") -> None:
        """Add or update a fact."""
        # Check for existing fact with same subject + predicate
        for existing in self.facts:
            if (existing.subject.lower() == subject.lower() and
                    existing.predicate.lower() == predicate.lower()):
                # Update existing fact
                existing.obj = obj
                existing.confidence = max(existing.confidence, confidence)
                existing.updated_at = datetime.now()
                existing.source = source
                print(f"    [SEM] Updated: {existing.triple}")
                return

        fact = Fact(subject=subject, predicate=predicate, obj=obj,
                    confidence=confidence, source=source)
        self.facts.append(fact)

    def query(self, subject: str = "", keyword: str = "") -> List[Fact]:
        """Query facts by subject or keyword."""
        results = self.facts
        if subject:
            results = [f for f in results if subject.lower() in f.subject.lower()]
        if keyword:
            kw = keyword.lower()
            results = [f for f in results
                       if kw in f.subject.lower() or kw in f.predicate.lower()
                       or kw in f.obj.lower()]
        return sorted(results, key=lambda f: f.confidence, reverse=True)

    def remove_fact(self, subject: str, predicate: str) -> bool:
        """Remove a fact (e.g., when it's proven incorrect)."""
        before = len(self.facts)
        self.facts = [
            f for f in self.facts
            if not (f.subject.lower() == subject.lower() and
                    f.predicate.lower() == predicate.lower())
        ]
        return len(self.facts) < before

    def to_context_string(self) -> str:
        """Format all facts for injection into an LLM prompt."""
        lines = []
        for f in sorted(self.facts, key=lambda f: f.confidence, reverse=True):
            lines.append(f"• {f.natural} (confidence: {f.confidence:.0%})")
        return "\n".join(lines)


# ================================================================
# 3. PROCEDURAL MEMORY (Skills, Instructions, Best Practices)
# ================================================================
# Procedural memory stores HOW TO DO things — skills and procedures.
# "How to" — step-by-step instructions and learned workflows.
#
# Examples:
#   - "To search for flights, use Skyscanner API with these params..."
#   - "When the user is frustrated, acknowledge their feelings first"
#   - "For budget calculations, always convert to USD using xe.com"
#
# Key properties:
#   - Instruction-based (steps, rules, patterns)
#   - Skill-like (improves with use and feedback)
#   - Rarely expires (procedures change slowly)

@dataclass
class Procedure:
    """A procedural memory — a learned skill or workflow."""
    name: str
    description: str
    steps: List[str]
    trigger: str                        # When to use this procedure
    success_count: int = 0              # Times this procedure succeeded
    failure_count: int = 0              # Times it failed
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def record_outcome(self, success: bool):
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used = datetime.now()


class ProceduralMemory:
    """
    Stores and retrieves procedural knowledge (skills and workflows).

    Procedures are matched by TRIGGER conditions and ranked by
    success rate.  Failed procedures get deprioritized over time.
    """

    def __init__(self):
        self.procedures: List[Procedure] = []

    def learn(self, name: str, description: str, steps: List[str],
              trigger: str) -> None:
        """Learn a new procedure."""
        # Update if procedure with same name exists
        for existing in self.procedures:
            if existing.name.lower() == name.lower():
                existing.steps = steps
                existing.description = description
                existing.trigger = trigger
                return
        self.procedures.append(
            Procedure(name=name, description=description,
                      steps=steps, trigger=trigger)
        )

    def match(self, situation: str) -> List[Procedure]:
        """Find procedures that match the current situation."""
        situation_words = set(situation.lower().split())
        matched = []
        for proc in self.procedures:
            trigger_words = set(proc.trigger.lower().split())
            if trigger_words & situation_words:
                matched.append(proc)
        # Rank by success rate
        return sorted(matched, key=lambda p: p.success_rate, reverse=True)

    def to_context_string(self, situation: str = "") -> str:
        """Format matching procedures for injection into an LLM prompt."""
        procs = self.match(situation) if situation else self.procedures
        lines = []
        for p in procs:
            lines.append(f"Procedure: {p.name} (success rate: {p.success_rate:.0%})")
            for i, step in enumerate(p.steps, 1):
                lines.append(f"  {i}. {step}")
        return "\n".join(lines)


# ================================================================
# 4. HOW TO STORE & RETRIEVE EACH TYPE
# ================================================================

class UnifiedMemoryStore:
    """
    Combines all three memory types into a single interface.

    This is the pattern used in production agents: one memory
    manager that routes storage/retrieval to the right subsystem.
    """

    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()

    def build_context(self, query: str = "", max_tokens: int = 500) -> str:
        """
        Build a context string from all memory types.

        This is what gets injected into the LLM's system prompt.
        Each memory type contributes a section.
        """
        parts = []

        # Episodic: recent events
        episodic_ctx = self.episodic.to_context_string(n=3)
        if episodic_ctx:
            parts.append(f"Recent events:\n{episodic_ctx}")

        # Semantic: relevant facts
        semantic_ctx = self.semantic.to_context_string()
        if semantic_ctx:
            parts.append(f"Known facts:\n{semantic_ctx}")

        # Procedural: relevant procedures
        if query:
            proc_ctx = self.procedural.to_context_string(query)
            if proc_ctx:
                parts.append(f"Available procedures:\n{proc_ctx}")

        context = "\n\n".join(parts)

        # Simple token budget enforcement
        words = context.split()
        if len(words) > max_tokens * 0.75:  # Approximate tokens
            context = " ".join(words[:int(max_tokens * 0.75)]) + "\n[Memory truncated]"

        return context


def demo_unified_memory():
    """Demonstrate all three memory types working together."""

    print("=" * 65)
    print("  EPISODIC, SEMANTIC & PROCEDURAL MEMORY")
    print("=" * 65)

    mem = UnifiedMemoryStore()

    # Populate episodic memory (events)
    print("\n  ── Populating Episodic Memory ──")
    mem.episodic.record(
        "User asked about Japan trip planning",
        context="First message in session, user seems excited",
        participants=["user", "agent"],
        outcome="Provided city recommendations",
        importance=0.7,
    )
    mem.episodic.record(
        "User specified budget of $3000 for 2 weeks",
        context="Budget discussion during planning",
        participants=["user"],
        importance=0.8,
    )
    print(f"    Recorded 2 episodes")

    # Populate semantic memory (facts)
    print("\n  ── Populating Semantic Memory ──")
    mem.semantic.add_fact("user", "has_allergy", "nuts", confidence=0.95)
    mem.semantic.add_fact("user", "prefers", "vegetarian food", confidence=0.9)
    mem.semantic.add_fact("Japan Rail Pass", "costs", "¥50,000 for 14 days", confidence=0.85)
    mem.semantic.add_fact("user", "speaks", "some Japanese", confidence=0.7)
    print(f"    Stored 4 facts")

    # Populate procedural memory (skills)
    print("\n  ── Populating Procedural Memory ──")
    mem.procedural.learn(
        name="trip_budget_calculation",
        description="Calculate trip budget with breakdown",
        steps=[
            "Get total budget and trip duration",
            "Estimate: 40% accommodation, 30% food, 20% transport, 10% activities",
            "Convert to local currency using current rates",
            "Add 10% buffer for unexpected expenses",
            "Present itemized breakdown to user",
        ],
        trigger="budget calculate cost expense money",
    )
    mem.procedural.learn(
        name="restaurant_recommendation",
        description="Recommend restaurants with dietary awareness",
        steps=[
            "Check user's dietary restrictions from semantic memory",
            "Search for restaurants matching restrictions",
            "Filter by budget range",
            "Prioritize highly-rated options",
            "Present top 3 with prices and allergen info",
        ],
        trigger="restaurant food eat dining meal",
    )
    print(f"    Learned 2 procedures")

    # Build unified context
    print(f"\n  ── Building Unified Context ──")
    context = mem.build_context(query="budget food restaurant")
    print(f"\n{context}")

    # Show how each type serves a different purpose
    print(f"\n  ── Memory Type Summary ──")
    print(f"  {'Type':<15} {'Purpose':<25} {'Retrieval By':<20} {'Example'}")
    print(f"  {'─' * 15} {'─' * 25} {'─' * 20} {'─' * 30}")
    print(f"  {'Episodic':<15} {'What happened':<25} {'Time, similarity':<20} {'User asked about Japan'}")
    print(f"  {'Semantic':<15} {'What is true':<25} {'Subject, keyword':<20} {'User is allergic to nuts'}")
    print(f"  {'Procedural':<15} {'How to do it':<25} {'Trigger matching':<20} {'Budget calculation steps'}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 11: Memory Categories                      ║")
    print("╚" + "═" * 63 + "╝")

    demo_unified_memory()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Episodic memory stores EVENTS (what happened, when, with whom).
       Best for maintaining conversation continuity and learning from
       past interactions.

    2. Semantic memory stores FACTS (what is true). Best for user
       preferences, domain knowledge, and learned information.
       Supports CRUD operations — facts are UPDATED, not just appended.

    3. Procedural memory stores SKILLS (how to do things). Best for
       learned workflows and best practices. Ranked by success rate
       to prefer procedures that have worked before.

    4. A UnifiedMemoryStore combines all three types and builds a
       context string that includes relevant entries from each.

    5. Each type has different storage, retrieval, and decay patterns.
       Don't treat all memory the same — categorize and manage each
       type appropriately.
    """))
