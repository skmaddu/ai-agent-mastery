import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 2: Recent Advancements — Infinite Context (Gemini 2.x / Claude 4.x)
=============================================================================
Pure-Python concept demonstration (no LLM calls required).

Covers:
  1. Current Real Limits of "Infinite" Context (cost, latency, quality)
  2. Gemini 2.x & Claude 4.x Practical Benchmarks
  3. Why Engineering Is Still Required Even with Long Contexts

Key insight: Models now accept 1M–10M tokens, but "accepts" ≠ "uses well".
Effective utilization drops sharply in the middle, costs scale linearly (or
worse), and latency makes interactive use painful beyond ~200k tokens.
Context engineering is MORE important with long contexts, not less.

Run: python week-05-context-memory/examples/example_02_infinite_context_techniques.py
"""

import textwrap
from dataclasses import dataclass
from typing import List


# ================================================================
# 1. CURRENT REAL LIMITS OF "INFINITE" CONTEXT
# ================================================================
# Marketing says "1 million tokens".  Reality is more nuanced:
#
#   • COST: 1M input tokens on Claude 4 Sonnet ≈ $3.  That's per
#     single call.  An agent making 20 calls/session = $60/session.
#   • LATENCY: Time-to-first-token at 1M tokens can be 15-45 seconds
#     depending on the provider.  Users won't wait that long.
#   • QUALITY: The "needle in a haystack" test shows >95% recall for
#     most models, but real-world tasks with COMPETING information
#     show 60-80% accuracy at 500k+ tokens (distraction effect).
#   • RATE LIMITS: Providers throttle large-context requests more
#     aggressively.  You may hit TPM limits after just a few calls.

@dataclass
class ModelContextSpec:
    """Specification for a model's context capabilities."""
    name: str
    max_context: int           # Maximum tokens accepted
    effective_context: int     # Tokens where quality remains high
    input_cost_per_m: float    # $ per million input tokens
    output_cost_per_m: float   # $ per million output tokens
    ttft_at_100k: str          # Time to first token at 100k input
    ttft_at_1m: str            # Time to first token at 1M input
    needle_accuracy_1m: str    # Needle-in-haystack accuracy at 1M


# 2026-era model specifications (approximate benchmarks)
MODELS_2026 = [
    ModelContextSpec(
        name="Gemini 2.5 Pro",
        max_context=1_048_576,
        effective_context=500_000,
        input_cost_per_m=1.25,
        output_cost_per_m=10.00,
        ttft_at_100k="~1.5s",
        ttft_at_1m="~12s",
        needle_accuracy_1m="96%",
    ),
    ModelContextSpec(
        name="Gemini 2.5 Flash",
        max_context=1_048_576,
        effective_context=400_000,
        input_cost_per_m=0.15,
        output_cost_per_m=0.60,
        ttft_at_100k="~0.5s",
        ttft_at_1m="~6s",
        needle_accuracy_1m="92%",
    ),
    ModelContextSpec(
        name="Claude 4 Opus",
        max_context=1_048_576,
        effective_context=600_000,
        input_cost_per_m=15.00,
        output_cost_per_m=75.00,
        ttft_at_100k="~2.0s",
        ttft_at_1m="~20s",
        needle_accuracy_1m="97%",
    ),
    ModelContextSpec(
        name="Claude 4 Sonnet",
        max_context=1_048_576,
        effective_context=500_000,
        input_cost_per_m=3.00,
        output_cost_per_m=15.00,
        ttft_at_100k="~1.0s",
        ttft_at_1m="~15s",
        needle_accuracy_1m="95%",
    ),
    ModelContextSpec(
        name="GPT-4o",
        max_context=128_000,
        effective_context=80_000,
        input_cost_per_m=2.50,
        output_cost_per_m=10.00,
        ttft_at_100k="~1.2s",
        ttft_at_1m="N/A",
        needle_accuracy_1m="N/A (128k max)",
    ),
    ModelContextSpec(
        name="Llama 3.3 70B (Groq)",
        max_context=128_000,
        effective_context=64_000,
        input_cost_per_m=0.59,
        output_cost_per_m=0.79,
        ttft_at_100k="~0.3s",
        ttft_at_1m="N/A",
        needle_accuracy_1m="N/A (128k max)",
    ),
]


def demo_real_limits():
    """Show the gap between marketed and effective context windows."""

    print("=" * 75)
    print("  REAL LIMITS OF 'INFINITE' CONTEXT WINDOWS")
    print("=" * 75)

    print(f"\n  {'Model':<22} {'Max':>10} {'Effective':>10} "
          f"{'$/M in':>8} {'TTFT@1M':>10} {'Needle@1M':>12}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 12}")

    for m in MODELS_2026:
        print(f"  {m.name:<22} {m.max_context:>10,} {m.effective_context:>10,} "
              f"${m.input_cost_per_m:>6.2f} {m.ttft_at_1m:>10} {m.needle_accuracy_1m:>12}")

    print("\n  Key observations:")
    print("  • 'Effective context' is typically 40-60% of the advertised maximum.")
    print("  • Even cheap models become expensive at scale: Flash at 1M = $0.15/call.")
    print("  • TTFT at 1M tokens makes interactive chat feel sluggish (6-20s).")


# ================================================================
# 2. GEMINI 2.x & CLAUDE 4.x PRACTICAL BENCHMARKS
# ================================================================
# The "Needle in a Haystack" (NIAH) test is the standard benchmark
# for long-context quality.  A fact is hidden at a random position
# in a large document, and the model must find it.
#
# But NIAH is TOO EASY — real tasks are harder because:
#   1. There are MULTIPLE competing facts (not just one needle)
#   2. The model must SYNTHESIZE information, not just find it
#   3. Context contains CONTRADICTORY statements
#   4. The answer isn't a verbatim quote but requires reasoning
#
# More realistic benchmarks: RULER, LongBench v2, BABILong, InfiniteBench

@dataclass
class BenchmarkResult:
    """Result from a long-context benchmark suite."""
    model: str
    benchmark: str
    context_size: str
    score: float       # 0-100 accuracy
    notes: str


BENCHMARK_DATA = [
    # Needle-in-a-haystack (easy — nearly all models ace this)
    BenchmarkResult("Gemini 2.5 Pro", "NIAH (single)", "1M", 98.5, "Near-perfect recall"),
    BenchmarkResult("Claude 4 Sonnet", "NIAH (single)", "1M", 97.2, "Near-perfect recall"),
    BenchmarkResult("GPT-4o", "NIAH (single)", "128k", 96.8, "Strong within window"),

    # Multi-needle (harder — must find multiple facts)
    BenchmarkResult("Gemini 2.5 Pro", "NIAH (multi-10)", "1M", 85.3, "Drops with more needles"),
    BenchmarkResult("Claude 4 Sonnet", "NIAH (multi-10)", "1M", 82.1, "Similar degradation"),
    BenchmarkResult("GPT-4o", "NIAH (multi-10)", "128k", 88.5, "Shorter context helps"),

    # Synthesis tasks (hardest — real-world proxy)
    BenchmarkResult("Gemini 2.5 Pro", "LongBench v2", "256k", 72.4, "Synthesis is harder"),
    BenchmarkResult("Claude 4 Sonnet", "LongBench v2", "256k", 74.8, "Slightly better at reasoning"),
    BenchmarkResult("GPT-4o", "LongBench v2", "128k", 71.2, "Competitive at shorter length"),

    # Very long context degradation
    BenchmarkResult("Gemini 2.5 Pro", "RULER", "1M", 65.0, "Significant quality drop"),
    BenchmarkResult("Claude 4 Sonnet", "RULER", "500k", 68.3, "Better quality/length ratio"),
]


def demo_practical_benchmarks():
    """Display benchmark comparisons showing the reality gap."""

    print("\n" + "=" * 75)
    print("  GEMINI 2.x & CLAUDE 4.x — PRACTICAL BENCHMARKS (2026)")
    print("=" * 75)

    print(f"\n  {'Model':<22} {'Benchmark':<18} {'Context':>8} "
          f"{'Score':>7} {'Notes'}")
    print(f"  {'─' * 22} {'─' * 18} {'─' * 8} {'─' * 7} {'─' * 30}")

    for b in BENCHMARK_DATA:
        print(f"  {b.model:<22} {b.benchmark:<18} {b.context_size:>8} "
              f"{b.score:>6.1f}% {b.notes}")

    print("\n  Interpretation:")
    print("  • Single-needle NIAH is solved (>95% for all modern models).")
    print("  • Multi-fact retrieval drops 10-15% — the 'distraction' effect.")
    print("  • Synthesis tasks (the real job of agents) score 65-75% even")
    print("    on the best models. THIS is why context engineering matters.")
    print("  • GPT-4o with 128k often matches 1M-context models on real tasks")
    print("    because shorter, curated context avoids dilution.")


# ================================================================
# 3. WHY ENGINEERING IS STILL REQUIRED EVEN WITH LONG CONTEXTS
# ================================================================
# "But if the model can handle 1M tokens, why not just dump
#  everything in?" — This is the #1 misconception in 2026.
#
# Five concrete reasons:
#   1. COST EXPLOSION: 1M tokens × 20 calls/session × $3/call = $60
#   2. LATENCY TAX: Users expect <2s responses, not 15s
#   3. DILUTION: Irrelevant content actively harms accuracy
#   4. STALE DATA: Old context may contradict current state
#   5. SECURITY: Larger windows = larger attack surface for injection

def demo_why_engineering_still_needed():
    """Demonstrate the five reasons why long context doesn't save you."""

    print("\n" + "=" * 75)
    print("  WHY CONTEXT ENGINEERING IS STILL REQUIRED WITH LONG CONTEXTS")
    print("=" * 75)

    # Cost comparison: engineered vs. "dump everything"
    print("\n  ── COST COMPARISON: Engineered vs. 'Dump Everything' ──\n")

    scenarios = [
        {
            "name": "Dump Everything",
            "tokens_per_call": 500_000,
            "calls_per_session": 15,
            "model": "claude-4-sonnet",
            "cost_per_m": 3.00,
        },
        {
            "name": "Engineered Context",
            "tokens_per_call": 20_000,
            "calls_per_session": 15,
            "model": "claude-4-sonnet",
            "cost_per_m": 3.00,
        },
    ]

    for s in scenarios:
        session_cost = (s["tokens_per_call"] * s["calls_per_session"]
                        * s["cost_per_m"] / 1_000_000)
        monthly_cost = session_cost * 100  # 100 sessions/month
        print(f"  Strategy: {s['name']}")
        print(f"    Tokens/call: {s['tokens_per_call']:>10,}")
        print(f"    Calls/session: {s['calls_per_session']:>8}")
        print(f"    Cost/session: ${session_cost:>10.2f}")
        print(f"    Monthly (100 sessions): ${monthly_cost:>10.2f}")
        print()

    savings_pct = (1 - 20_000 / 500_000) * 100
    print(f"  → Engineered context saves {savings_pct:.0f}% on cost alone.")

    # The five pillars
    print(f"\n  ── FIVE REASONS YOU STILL NEED CONTEXT ENGINEERING ──\n")

    reasons = [
        ("1. Cost Explosion",
         "1M tokens × 20 calls × $3/M = $60/session → $6,000/mo at scale.",
         "Budget: allocate tokens per zone, evict low-priority content."),
        ("2. Latency Tax",
         "TTFT at 1M tokens is 12-20s. Users expect <2s for chat UX.",
         "Strategy: keep interactive calls under 30k tokens."),
        ("3. Dilution / Distraction",
         "Irrelevant tokens actively compete for attention, reducing accuracy.",
         "Strategy: retrieve fewer, higher-quality chunks (precision > recall)."),
        ("4. Stale Data / Context Rot",
         "Old tool results or outdated facts contradict current state.",
         "Strategy: TTL (time-to-live) on context entries, freshness checks."),
        ("5. Security / Attack Surface",
         "Larger context = more room for indirect prompt injection.",
         "Strategy: sanitize all retrieved content, validate tool outputs."),
    ]

    for title, problem, solution in reasons:
        print(f"  {title}")
        print(f"    Problem:  {problem}")
        print(f"    Solution: {solution}")
        print()


# ================================================================
# BONUS: Context Length Decision Framework
# ================================================================

def demo_context_length_decision():
    """Provide a practical decision framework for choosing context size."""

    print("=" * 75)
    print("  DECISION FRAMEWORK: HOW MUCH CONTEXT DO YOU ACTUALLY NEED?")
    print("=" * 75)

    print("""
  Ask these questions in order:

  Q1: Is this a RETRIEVAL task (find a specific fact) or SYNTHESIS
      (combine multiple sources)?
      → Retrieval: 5-20k tokens is usually sufficient.
      → Synthesis: 20-80k tokens, carefully curated.

  Q2: How many SOURCES does the answer depend on?
      → 1-3 sources: include them directly (5-15k tokens).
      → 4-10 sources: summarize each, include summaries (10-30k).
      → 10+ sources: use hierarchical summarization or RAG pipeline.

  Q3: Is LATENCY critical (interactive chat vs. batch processing)?
      → Interactive (<2s): keep under 30k tokens.
      → Background processing: up to 200k is fine.

  Q4: What's your BUDGET per session?
      → Use the cost table from Section 1 to calculate.
      → Rule: if cost/session > $1, you're probably over-stuffing.

  Q5: Does the context contain CONTRADICTORY information?
      → If yes: filter and deduplicate BEFORE sending to the model.
      → Contradictions in context cause unpredictable outputs.

  ┌─────────────────────────────────────────────────────────┐
  │  GOLDEN RULE: Start small (10-20k), measure quality,   │
  │  add context ONLY when quality improves.  Never start   │
  │  by dumping everything in and hoping for the best.      │
  └─────────────────────────────────────────────────────────┘
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 73 + "╗")
    print("║  WEEK 5 — EXAMPLE 2: Infinite Context — Reality vs. Marketing        ║")
    print("╚" + "═" * 73 + "╝")

    demo_real_limits()
    demo_practical_benchmarks()
    demo_why_engineering_still_needed()
    demo_context_length_decision()

    print("\n" + "=" * 75)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 75)
    print(textwrap.dedent("""
    1. "1M token context" means the model ACCEPTS 1M tokens, not that it
       USES them well.  Effective context is 40-60% of the maximum.

    2. Needle-in-a-haystack is solved; real-world synthesis tasks score
       65-75% even on frontier models.  Don't trust marketing benchmarks.

    3. Cost scales linearly with context.  An unoptimized agent can cost
       25× more than an engineered one with identical quality.

    4. Latency at long contexts (6-20s TTFT) makes interactive UX painful.
       Keep interactive calls under 30k tokens.

    5. The golden rule: start with MINIMAL context, add more only when
       measured quality improves.  "Dump everything" is almost always wrong.
    """))
