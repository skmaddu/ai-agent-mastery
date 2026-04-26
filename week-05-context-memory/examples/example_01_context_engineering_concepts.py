import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 1: Context Engineering Overview — Managing Token Limits & Efficiency
=============================================================================
Pure-Python concept demonstration (no LLM calls required).

Covers:
  1. Why Context Engineering Replaced Prompt Engineering in 2026
  2. Context as "Agent RAM" – The Living Playbook (ACE Framework)
  3. Token Limits, Cost, Latency & Accuracy Trade-offs
  4. Recency Bias, Distraction & The "Lost in the Middle" Problem

Key insight: In 2026 the bottleneck shifted from *writing good prompts* to
*managing what enters the context window*.  A perfectly-worded prompt buried
under 50 k tokens of noise will still produce garbage.  Context engineering
treats the context window as a scarce, shared resource — like RAM in an
operating system — and applies deliberate strategies to fill it with the
right information at the right time.

Run: python week-05-context-memory/examples/example_01_context_engineering_concepts.py
"""

import math
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ================================================================
# 1. WHY CONTEXT ENGINEERING REPLACED PROMPT ENGINEERING IN 2026
# ================================================================
# Before 2026, most AI tutorials focused on prompt engineering:
#   "Write clearer instructions and you'll get better output."
#
# That advice still holds, but it solves only ~20% of the problem.
# Modern agents maintain LONG conversations, call tools, retrieve
# documents, and juggle multiple sub-tasks — all within one context
# window.  The dominant failure mode is no longer "bad prompt" but
# "wrong stuff in the window" (stale data, irrelevant docs, leaked
# secrets, duplicated information, or simply too many tokens).
#
# Context engineering = deciding WHAT goes in, WHEN it enters, HOW
# MUCH space it gets, and WHEN it leaves.

def demo_prompt_vs_context_engineering():
    """Compare the scope of prompt engineering vs. context engineering."""

    print("=" * 65)
    print("  PROMPT ENGINEERING  vs.  CONTEXT ENGINEERING")
    print("=" * 65)

    comparison = [
        ("Scope",
         "Single user ↔ LLM turn",
         "Entire agent lifecycle"),
        ("Primary concern",
         "Wording & instruction clarity",
         "What fills the window & when"),
        ("Failure mode",
         "Vague / ambiguous instructions",
         "Noise, overflow, stale data"),
        ("Typical artefact",
         "System prompt template",
         "Context pipeline + memory mgr"),
        ("Analogy",
         "Writing a good exam question",
         "Managing a computer's RAM"),
    ]

    for dimension, prompt_eng, context_eng in comparison:
        print(f"\n  {dimension}:")
        print(f"    Prompt Eng  → {prompt_eng}")
        print(f"    Context Eng → {context_eng}")

    print("\n" + "-" * 65)
    print("  Takeaway: Prompt engineering is a SUBSET of context engineering.")
    print("  You still need clear prompts, but you also need to control the")
    print("  entire information pipeline feeding into your agent's window.")
    print("-" * 65)


# ================================================================
# 2. CONTEXT AS "AGENT RAM" — THE LIVING PLAYBOOK (ACE FRAMEWORK)
# ================================================================
# The ACE (Agent Context Engineering) framework treats the context
# window as having four zones, much like memory segments in an OS:
#
#   ┌─────────────────────────────────────────────┐
#   │  SYSTEM / INSTRUCTIONS   (static)           │  ← "ROM"
#   ├─────────────────────────────────────────────┤
#   │  KNOWLEDGE / RETRIEVED DOCS  (semi-static)  │  ← "Heap"
#   ├─────────────────────────────────────────────┤
#   │  CONVERSATION HISTORY  (growing)            │  ← "Stack"
#   ├─────────────────────────────────────────────┤
#   │  TOOL RESULTS / SCRATCH PAD  (volatile)     │  ← "Registers"
#   └─────────────────────────────────────────────┘
#
# Each zone competes for the same finite token budget.  As the
# conversation grows, the "stack" pushes against the "heap", and
# you must decide what to evict — just like an OS does with pages.

@dataclass
class ContextZone:
    """Represents one zone of the context window with a token budget."""
    name: str
    description: str
    max_tokens: int
    current_tokens: int = 0
    priority: int = 1          # Higher = harder to evict

    @property
    def utilization(self) -> float:
        """How full this zone is, as a percentage."""
        if self.max_tokens == 0:
            return 0.0
        return (self.current_tokens / self.max_tokens) * 100

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.current_tokens)


@dataclass
class ContextWindow:
    """
    Models an LLM context window as a set of competing zones.

    This is the core abstraction of the ACE framework: treat the
    window as a resource to be MANAGED, not just a place to dump text.
    """
    total_tokens: int
    zones: List[ContextZone] = field(default_factory=list)

    def add_zone(self, zone: ContextZone) -> None:
        self.zones.append(zone)

    def allocated(self) -> int:
        return sum(z.current_tokens for z in self.zones)

    def free(self) -> int:
        return self.total_tokens - self.allocated()

    def report(self) -> None:
        """Print a visual report of context window utilization."""
        print(f"\n  Context Window: {self.total_tokens:,} tokens total")
        print(f"  Allocated: {self.allocated():,} | Free: {self.free():,}")
        print(f"  {'─' * 50}")

        for z in self.zones:
            bar_len = 30
            filled = int((z.current_tokens / self.total_tokens) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{bar}] {z.name:20s} "
                  f"{z.current_tokens:>6,}/{z.max_tokens:>6,} "
                  f"({z.utilization:5.1f}%)")

        # Warn if over-allocated (a real problem in production!)
        if self.allocated() > self.total_tokens:
            overage = self.allocated() - self.total_tokens
            print(f"\n  ⚠ OVER-ALLOCATED by {overage:,} tokens!  "
                  "Eviction or summarization required.")


def demo_context_as_ram():
    """Show how context zones compete for the same finite window."""

    print("\n" + "=" * 65)
    print("  CONTEXT AS AGENT RAM — ACE FRAMEWORK DEMO")
    print("=" * 65)

    # Simulate a 128k-token window (e.g. GPT-4o or Claude 3.5)
    window = ContextWindow(total_tokens=128_000)

    # Define the four standard zones with realistic budgets
    window.add_zone(ContextZone(
        name="System/Instructions",
        description="System prompt, persona, rules",
        max_tokens=4_000,
        current_tokens=2_500,
        priority=5,   # Never evict the system prompt
    ))
    window.add_zone(ContextZone(
        name="Knowledge/RAG",
        description="Retrieved documents, facts",
        max_tokens=40_000,
        current_tokens=35_000,
        priority=3,
    ))
    window.add_zone(ContextZone(
        name="Conversation",
        description="User + assistant messages",
        max_tokens=60_000,
        current_tokens=45_000,
        priority=2,   # Can be summarized
    ))
    window.add_zone(ContextZone(
        name="Tool Results",
        description="Recent tool call outputs",
        max_tokens=24_000,
        current_tokens=18_000,
        priority=1,   # Most volatile — evict first
    ))

    window.report()

    # Simulate a scenario where a new RAG retrieval pushes us over
    print("\n  → New RAG retrieval adds 30,000 tokens...")
    window.zones[1].current_tokens += 30_000
    window.report()

    print("\n  → Strategy: Summarize conversation (45k → 8k) and "
          "trim old tool results (18k → 2k).")
    window.zones[2].current_tokens = 8_000
    window.zones[3].current_tokens = 2_000
    window.report()

    print("\n  Key lesson: context management is an ONGOING process,")
    print("  not a one-time setup.  Every new message or retrieval")
    print("  requires a budget check and possible eviction.")


# ================================================================
# 3. TOKEN LIMITS, COST, LATENCY & ACCURACY TRADE-OFFS
# ================================================================
# Using more of the context window isn't free.  There are four
# interrelated costs:
#
#   1. MONETARY COST — tokens are priced per-million.  A 100k
#      context costs 10–50× more than a 10k context.
#   2. LATENCY — time-to-first-token grows with input length,
#      often super-linearly due to attention complexity.
#   3. ACCURACY — more context ≠ better answers.  Irrelevant
#      tokens DILUTE the signal (the "distraction" effect).
#   4. RELIABILITY — very long contexts increase the chance of
#      the model hallucinating or ignoring instructions.

# 2026 pricing (approximate, for illustration)
MODEL_PRICING_PER_M_TOKENS = {
    "groq-llama-3.3-70b":       {"input": 0.59, "output": 0.79},
    "gpt-4o":                    {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":               {"input": 0.15, "output": 0.60},
    "claude-4-sonnet":           {"input": 3.00, "output": 15.00},
    "gemini-2.5-flash":          {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro":            {"input": 1.25, "output": 10.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the dollar cost of a single LLM call."""
    pricing = MODEL_PRICING_PER_M_TOKENS.get(model)
    if not pricing:
        return 0.0
    return (input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000)


def demo_token_cost_tradeoffs():
    """Show how context size affects cost, latency, and accuracy."""

    print("\n" + "=" * 65)
    print("  TOKEN LIMITS, COST, LATENCY & ACCURACY TRADE-OFFS")
    print("=" * 65)

    # Scenario: same question, different amounts of context stuffed in
    scenarios = [
        ("Minimal (2k in)",    2_000,   500),
        ("Moderate (20k in)",  20_000,  500),
        ("Large (80k in)",     80_000,  500),
        ("Maximum (120k in)", 120_000,  500),
    ]

    print(f"\n  {'Scenario':<22} {'Input':>8} {'Output':>8} ", end="")
    for model in ["gpt-4o-mini", "gpt-4o", "claude-4-sonnet"]:
        print(f" {model:>16}", end="")
    print()
    print(f"  {'─' * 22} {'─' * 8} {'─' * 8}", end="")
    for _ in range(3):
        print(f" {'─' * 16}", end="")
    print()

    for label, inp, out in scenarios:
        print(f"  {label:<22} {inp:>7,} {out:>7,} ", end="")
        for model in ["gpt-4o-mini", "gpt-4o", "claude-4-sonnet"]:
            cost = estimate_cost(model, inp, out)
            print(f" ${cost:>14.4f}", end="")
        print()

    # Latency model (simplified: base + per-1k-token overhead)
    print(f"\n  Estimated Latency (simplified model):")
    print(f"  {'Context Size':<18} {'Time to First Token':>20} {'Quality Impact':>18}")
    print(f"  {'─' * 18} {'─' * 20} {'─' * 18}")

    latency_data = [
        ("2k tokens",    "~0.3s",  "May lack info"),
        ("20k tokens",   "~0.8s",  "Sweet spot ✓"),
        ("80k tokens",   "~2.5s",  "Some dilution"),
        ("120k tokens",  "~4.0s",  "High dilution risk"),
    ]
    for size, ttft, quality in latency_data:
        print(f"  {size:<18} {ttft:>20} {quality:>18}")

    print("\n  Rule of thumb: use the MINIMUM context needed for the task.")
    print("  Adding 'just in case' context is like malloc-ing RAM you")
    print("  never use — except here you PAY for every unused byte.")


# ================================================================
# 4. RECENCY BIAS, DISTRACTION & THE "LOST IN THE MIDDLE" PROBLEM
# ================================================================
# Research (Liu et al., 2023 & follow-ups through 2025) shows that
# LLMs pay MORE attention to:
#   - The BEGINNING of the context (primacy effect)
#   - The END of the context (recency effect)
# And LESS attention to the MIDDLE.  This is the "Lost in the
# Middle" problem.  It means:
#   - Critical instructions should be at the START (system prompt)
#     or REPEATED at the END (just before the final user turn).
#   - Retrieved documents should be RANKED, with the most relevant
#     chunks placed at the beginning and end — not randomly.

def demo_lost_in_the_middle():
    """Simulate the attention distribution across context positions."""

    print("\n" + "=" * 65)
    print("  RECENCY BIAS, DISTRACTION & LOST IN THE MIDDLE")
    print("=" * 65)

    # Simulate a simplified attention curve over 20 context positions
    # Real attention is more complex, but this captures the U-shape
    n_positions = 20
    positions = list(range(1, n_positions + 1))

    # U-shaped attention: high at start, dips in middle, rises at end
    # Using a simple quadratic model for illustration
    mid = (n_positions + 1) / 2
    attention_scores = []
    for p in positions:
        # Quadratic U-shape centered at the middle
        distance_from_mid = abs(p - mid)
        score = 0.4 + 0.6 * (distance_from_mid / mid) ** 1.5
        attention_scores.append(min(1.0, score))

    print("\n  Simulated Attention Distribution Across Context Positions:")
    print(f"  (1 = beginning, {n_positions} = end of context)\n")

    max_bar = 40
    for i, (pos, score) in enumerate(zip(positions, attention_scores)):
        bar = "█" * int(score * max_bar)
        zone = ""
        if pos <= 3:
            zone = " ← System prompt (HIGH attention)"
        elif pos >= n_positions - 2:
            zone = " ← Recent turns (HIGH attention)"
        elif pos == n_positions // 2:
            zone = " ← MIDDLE (LOW attention — lost here!)"
        print(f"  Pos {pos:>2}: [{bar:<{max_bar}}] {score:.2f}{zone}")

    # Practical strategies
    print(f"\n  {'─' * 60}")
    print("  STRATEGIES TO COMBAT 'LOST IN THE MIDDLE':")
    print(f"  {'─' * 60}")

    strategies = [
        ("1. Primacy placement",
         "Put critical rules in the system prompt (position 1-3)"),
        ("2. Recency reinforcement",
         "Repeat key instructions just before the user's query"),
        ("3. Ranked retrieval",
         "Place most-relevant docs at the START of the context"),
        ("4. Chunked context",
         "Split long docs into smaller chunks, retrieve only the best"),
        ("5. Summarize the middle",
         "Replace verbose middle sections with concise summaries"),
    ]
    for title, desc in strategies:
        print(f"\n  {title}")
        print(f"    → {desc}")


def demo_distraction_effect():
    """Show how irrelevant context can degrade answer quality."""

    print("\n" + "=" * 65)
    print("  THE DISTRACTION EFFECT — MORE CONTEXT ≠ BETTER ANSWERS")
    print("=" * 65)

    # Simulate retrieval quality vs. number of documents
    print("\n  Simulated experiment: 'What is the capital of France?'")
    print("  We stuff N documents into context, only 1 is relevant.\n")

    print(f"  {'Docs in context':>16} {'Relevant':>10} {'Irrelevant':>12} "
          f"{'Accuracy':>10} {'Risk':>15}")
    print(f"  {'─' * 16} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 15}")

    data = [
        (1,  1, 0,  "99%", "Under-informed"),
        (3,  1, 2,  "97%", "Low"),
        (5,  1, 4,  "95%", "Acceptable"),
        (10, 1, 9,  "88%", "Moderate"),
        (20, 1, 19, "75%", "High dilution"),
        (50, 1, 49, "60%", "Severe noise"),
    ]
    for total, rel, irrel, acc, risk in data:
        print(f"  {total:>16} {rel:>10} {irrel:>12} {acc:>10} {risk:>15}")

    print("\n  Lesson: retrieval PRECISION matters more than recall.")
    print("  Returning 3 highly relevant chunks beats 20 'maybe relevant' ones.")
    print("  This is why reranking (Week 5, Topic 5) is critical.")


# ================================================================
# MAIN — Run all demos
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 1: Context Engineering Overview            ║")
    print("╚" + "═" * 63 + "╝")

    demo_prompt_vs_context_engineering()
    demo_context_as_ram()
    demo_token_cost_tradeoffs()
    demo_lost_in_the_middle()
    demo_distraction_effect()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Context engineering is the 2026 evolution of prompt engineering.
       It treats the context window as a MANAGED RESOURCE, not a dumping ground.

    2. The ACE framework divides context into four zones (instructions,
       knowledge, conversation, tool results) that compete for space.

    3. More context is NOT free: it increases cost, latency, and dilution.
       Use the MINIMUM context needed for each task.

    4. The "Lost in the Middle" problem means LLMs pay less attention to
       middle positions.  Place critical information at the START and END.

    5. Retrieval precision > recall.  Three relevant chunks beat twenty
       vaguely-related ones.

    6. Context management is CONTINUOUS — every new turn or retrieval
       requires a budget check and possible eviction/summarization.
    """))
