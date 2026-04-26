import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3: Techniques — Summarization, Compression, Windowing
==============================================================
Pure-Python concept demonstration (no LLM calls required).

Covers:
  1. Map-Reduce, Refine & Hierarchical Summarization
  2. Compression Libraries (LLMLingua-style)
  3. Sliding Windows vs Hierarchical Windowing
  4. The 4 Pillars of Context Engineering (Write, Select, Compress, Isolate)

These techniques are the TOOLKIT you use to manage context.  Think of
them as garbage-collection strategies for your context window: each
has different trade-offs between information preservation and token
savings.

Run: python week-05-context-memory/examples/example_03_context_techniques_concepts.py
"""

import textwrap
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ================================================================
# SAMPLE DATA — We'll use these paragraphs throughout the demos
# ================================================================

SAMPLE_DOCUMENTS = [
    "Artificial intelligence has transformed healthcare by enabling early disease "
    "detection through medical imaging analysis. Deep learning models can now identify "
    "tumors in X-rays and MRIs with accuracy comparable to senior radiologists. "
    "However, regulatory approval remains a bottleneck, and many AI diagnostic tools "
    "are limited to research settings.",

    "Natural language processing has revolutionized customer service through chatbots "
    "and virtual assistants. Modern LLMs can handle complex multi-turn conversations, "
    "resolve billing disputes, and even detect customer frustration through sentiment "
    "analysis. The key challenge is maintaining context across long conversations.",

    "Autonomous vehicles use a combination of computer vision, LiDAR, and reinforcement "
    "learning to navigate roads safely. While Level 4 autonomy has been achieved in "
    "geo-fenced areas, full Level 5 autonomy remains elusive due to edge cases like "
    "construction zones and aggressive human drivers.",

    "AI in education enables personalized learning paths that adapt to each student's "
    "pace and learning style. Intelligent tutoring systems can identify knowledge gaps "
    "and provide targeted practice problems. Critics argue that over-reliance on AI "
    "tutors may reduce critical thinking and social learning skills.",

    "Generative AI has disrupted creative industries, from art and music to writing "
    "and video production. While these tools democratize creativity, they raise "
    "concerns about copyright, authenticity, and the displacement of human artists. "
    "The legal landscape is still evolving rapidly.",
]


# ================================================================
# 1. MAP-REDUCE, REFINE & HIERARCHICAL SUMMARIZATION
# ================================================================
# These are three strategies for summarizing content that exceeds
# the context window.  Each trades off differently between quality,
# cost, and parallelizability.
#
#   MAP-REDUCE:
#     Step 1 (Map):   Summarize each chunk independently (parallelizable)
#     Step 2 (Reduce): Combine all summaries into one final summary
#     Pros: Fast (parallel), cheap (small contexts per call)
#     Cons: Loses cross-chunk connections
#
#   REFINE (Sequential):
#     Step 1: Summarize chunk 1
#     Step 2: Summarize chunk 2 + previous summary → updated summary
#     Step N: Continue until all chunks processed
#     Pros: Preserves cross-chunk context, better coherence
#     Cons: Slow (sequential), more expensive (growing context)
#
#   HIERARCHICAL:
#     Level 0: Raw chunks
#     Level 1: Summarize pairs of chunks
#     Level 2: Summarize pairs of level-1 summaries
#     Continue until one summary remains (like a tournament bracket)
#     Pros: Balanced quality/speed, O(log N) depth
#     Cons: More complex to implement

def simulate_word_count(text: str) -> int:
    """Approximate token count (1 token ≈ 0.75 words)."""
    return int(len(text.split()) / 0.75)


def simulate_summarize(text: str, target_ratio: float = 0.3) -> str:
    """
    Simulate summarization by keeping the first N words.
    In production, this would be an LLM call.
    """
    words = text.split()
    keep = max(5, int(len(words) * target_ratio))
    return " ".join(words[:keep]) + "..."


def demo_map_reduce():
    """Demonstrate the Map-Reduce summarization pattern."""

    print("=" * 65)
    print("  MAP-REDUCE SUMMARIZATION")
    print("=" * 65)
    print("\n  Strategy: Summarize each chunk independently, then combine.\n")

    # MAP phase: summarize each document independently
    print("  ── MAP PHASE (parallelizable) ──")
    chunk_summaries = []
    total_input_tokens = 0

    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        tokens = simulate_word_count(doc)
        total_input_tokens += tokens
        summary = simulate_summarize(doc, 0.3)
        chunk_summaries.append(summary)
        print(f"    Chunk {i + 1}: {tokens:>3} tokens → {simulate_word_count(summary):>3} tokens")

    # REDUCE phase: combine summaries
    print(f"\n  ── REDUCE PHASE ──")
    combined = " ".join(chunk_summaries)
    final_summary = simulate_summarize(combined, 0.5)
    final_tokens = simulate_word_count(final_summary)
    print(f"    Combined summaries: {simulate_word_count(combined)} tokens")
    print(f"    Final summary: {final_tokens} tokens")
    print(f"\n    Compression: {total_input_tokens} → {final_tokens} tokens "
          f"({final_tokens / total_input_tokens * 100:.0f}% of original)")

    print(f"\n    LLM calls: {len(SAMPLE_DOCUMENTS)} (map) + 1 (reduce) = "
          f"{len(SAMPLE_DOCUMENTS) + 1} total")
    print(f"    Parallelizable: YES (map phase)")


def demo_refine():
    """Demonstrate the Refine (sequential) summarization pattern."""

    print("\n" + "=" * 65)
    print("  REFINE (SEQUENTIAL) SUMMARIZATION")
    print("=" * 65)
    print("\n  Strategy: Each chunk refines the running summary.\n")

    running_summary = ""
    total_input_tokens = 0

    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        tokens = simulate_word_count(doc)
        total_input_tokens += tokens

        if i == 0:
            running_summary = simulate_summarize(doc, 0.4)
            context_size = tokens
        else:
            combined = running_summary + " " + doc
            context_size = simulate_word_count(combined)
            running_summary = simulate_summarize(combined, 0.3)

        print(f"    Step {i + 1}: context={context_size:>3} tokens → "
              f"summary={simulate_word_count(running_summary):>3} tokens")

    final_tokens = simulate_word_count(running_summary)
    print(f"\n    Final: {total_input_tokens} → {final_tokens} tokens")
    print(f"    LLM calls: {len(SAMPLE_DOCUMENTS)} (sequential)")
    print(f"    Parallelizable: NO — each step depends on the previous")
    print(f"    Quality: HIGHER — cross-chunk context preserved")


def demo_hierarchical():
    """Demonstrate the Hierarchical (tournament) summarization pattern."""

    print("\n" + "=" * 65)
    print("  HIERARCHICAL (TOURNAMENT) SUMMARIZATION")
    print("=" * 65)
    print("\n  Strategy: Pair-wise summarization in tree levels.\n")

    current_level = list(SAMPLE_DOCUMENTS)
    level = 0
    total_calls = 0

    while len(current_level) > 1:
        print(f"  Level {level}: {len(current_level)} items")
        next_level = []

        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                combined = current_level[i] + " " + current_level[i + 1]
                summary = simulate_summarize(combined, 0.4)
                total_calls += 1
                print(f"    Pair ({i + 1},{i + 2}) → summary "
                      f"({simulate_word_count(summary)} tokens)")
            else:
                summary = current_level[i]  # Odd one out, pass through
                print(f"    Item {i + 1} → passed through (unpaired)")
            next_level.append(summary)

        current_level = next_level
        level += 1

    final_tokens = simulate_word_count(current_level[0])
    depth = math.ceil(math.log2(max(len(SAMPLE_DOCUMENTS), 2)))
    print(f"\n    Final summary: {final_tokens} tokens")
    print(f"    Tree depth: {depth} levels")
    print(f"    LLM calls: {total_calls}")
    print(f"    Partially parallelizable: YES (within each level)")


# ================================================================
# 2. COMPRESSION LIBRARIES (LLMLingua-style)
# ================================================================
# LLMLingua (Microsoft, 2023-2024) and its successors use a small
# LLM to identify which tokens in the input are LEAST important
# and remove them.  This achieves 2-10× compression while preserving
# meaning.
#
# The core idea: not all tokens carry equal information.  Stop words,
# repeated phrases, and boilerplate can be removed without hurting
# the downstream task.

@dataclass
class CompressionResult:
    """Result of a compression pass."""
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    method: str

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


def simulate_llmlingua_compression(text: str) -> CompressionResult:
    """
    Simulate LLMLingua-style compression by removing low-information words.

    In production, you would use the actual LLMLingua library:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
        result = compressor.compress_prompt(text, rate=0.5)

    Here we simulate it with a simple stop-word removal + deduplication.
    """
    # Low-information words (simplified — real LLMLingua uses perplexity)
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'that', 'which', 'who', 'whom', 'this', 'these', 'those',
        'very', 'really', 'just', 'quite', 'rather', 'somewhat',
        'however', 'moreover', 'furthermore', 'additionally',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    }

    words = text.split()
    original_count = len(words)

    # Keep content words, remove stop words (but keep some for readability)
    compressed_words = []
    for i, w in enumerate(words):
        clean = w.lower().strip('.,;:!?')
        # Keep every 3rd stop word for readability, remove the rest
        if clean in stop_words:
            if i % 3 == 0:
                compressed_words.append(w)
        else:
            compressed_words.append(w)

    compressed_text = " ".join(compressed_words)
    return CompressionResult(
        original=text,
        compressed=compressed_text,
        original_tokens=int(original_count / 0.75),
        compressed_tokens=int(len(compressed_words) / 0.75),
        method="LLMLingua-style (simulated)",
    )


def demo_compression():
    """Show compression in action and discuss trade-offs."""

    print("\n" + "=" * 65)
    print("  COMPRESSION LIBRARIES (LLMLingua-style)")
    print("=" * 65)

    print("\n  Compressing sample documents:\n")

    total_orig = 0
    total_comp = 0

    for i, doc in enumerate(SAMPLE_DOCUMENTS[:3]):
        result = simulate_llmlingua_compression(doc)
        total_orig += result.original_tokens
        total_comp += result.compressed_tokens

        print(f"  Doc {i + 1}: {result.original_tokens} → {result.compressed_tokens} "
              f"tokens ({result.ratio:.0%})")
        print(f"    Original:   {doc[:80]}...")
        print(f"    Compressed: {result.compressed[:80]}...")
        print()

    print(f"  Total: {total_orig} → {total_comp} tokens "
          f"({total_comp / total_orig:.0%})")

    print(f"\n  {'─' * 55}")
    print("  COMPRESSION TRADE-OFFS:")
    print(f"  {'─' * 55}")
    print("  • 2× compression: safe, minimal quality loss")
    print("  • 5× compression: noticeable grammar degradation")
    print("  • 10× compression: meaning may be altered (risky)")
    print("  • Best for: boilerplate-heavy text (legal, medical records)")
    print("  • Worst for: dense technical content, code, math")


# ================================================================
# 3. SLIDING WINDOWS vs HIERARCHICAL WINDOWING
# ================================================================
# When conversation history grows beyond the token budget, you need
# a WINDOWING strategy to decide what stays and what goes.
#
#   SLIDING WINDOW: Keep the most recent N messages.
#     Pros: Simple, fast, preserves recency
#     Cons: Loses ALL information older than the window
#
#   SLIDING WINDOW WITH OVERLAP: Keep N messages, but also include
#     a summary of the evicted messages.
#     Pros: Preserves some historical context
#     Cons: Summary quality varies
#
#   HIERARCHICAL WINDOWING:
#     Layer 0: Last 10 messages (full text)
#     Layer 1: Messages 11-50 (summarized to 1 paragraph each)
#     Layer 2: Messages 51+ (one summary for the entire block)
#     Pros: Best quality/token ratio, preserves old context
#     Cons: More complex, requires periodic re-summarization

@dataclass
class Message:
    role: str
    content: str
    turn: int
    tokens: int = 0

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = simulate_word_count(self.content)


def demo_sliding_window():
    """Compare sliding window strategies."""

    print("\n" + "=" * 65)
    print("  SLIDING WINDOWS vs HIERARCHICAL WINDOWING")
    print("=" * 65)

    # Create a 20-turn conversation
    messages = []
    for i in range(1, 21):
        messages.append(Message("user", f"User message about topic {i} with some details " * 3, i))
        messages.append(Message("assistant", f"Assistant response to topic {i} with analysis " * 3, i))

    total_tokens = sum(m.tokens for m in messages)
    print(f"\n  Conversation: {len(messages)} messages, {total_tokens} tokens total")
    print(f"  Token budget: 500 tokens\n")

    # Strategy 1: Simple sliding window
    print("  ── Strategy 1: Simple Sliding Window (last N messages) ──")
    budget = 500
    window = []
    for msg in reversed(messages):
        if sum(m.tokens for m in window) + msg.tokens <= budget:
            window.insert(0, msg)
        else:
            break
    print(f"    Kept: {len(window)} messages (turns {window[0].turn}-{window[-1].turn})")
    print(f"    Lost: {len(messages) - len(window)} messages (turns 1-{window[0].turn - 1})")
    print(f"    Info loss: TOTAL for turns 1-{window[0].turn - 1}")

    # Strategy 2: Sliding window + summary
    print(f"\n  ── Strategy 2: Sliding Window + Summary ──")
    summary_budget = 100  # Reserve 100 tokens for summary
    remaining_budget = budget - summary_budget
    window2 = []
    for msg in reversed(messages):
        if sum(m.tokens for m in window2) + msg.tokens <= remaining_budget:
            window2.insert(0, msg)
        else:
            break
    evicted = len(messages) - len(window2)
    print(f"    Summary of turns 1-{window2[0].turn - 1}: ~{summary_budget} tokens")
    print(f"    Recent messages: {len(window2)} (turns {window2[0].turn}-{window2[-1].turn})")
    print(f"    Info loss: PARTIAL (summary retains key points)")

    # Strategy 3: Hierarchical
    print(f"\n  ── Strategy 3: Hierarchical Windowing ──")
    print(f"    Layer 0 (full text):   Last 4 messages   (~200 tokens)")
    print(f"    Layer 1 (1-line each): Messages 5-12     (~100 tokens)")
    print(f"    Layer 2 (one block):   Messages 13-40    (~100 tokens)")
    print(f"    Total: ~400 tokens — ALL turns represented!")
    print(f"    Info loss: MINIMAL (graduated compression)")

    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │  RECOMMENDATION: Use hierarchical windowing for    │")
    print(f"  │  production agents.  Simple sliding windows are    │")
    print(f"  │  fine for prototypes but lose critical early info. │")
    print(f"  └────────────────────────────────────────────────────┘")


# ================================================================
# 4. THE 4 PILLARS OF CONTEXT ENGINEERING
# ================================================================
# Every context engineering strategy falls into one of four pillars:
#
#   WRITE   — What goes INTO the context (system prompts, RAG, tools)
#   SELECT  — What to KEEP when space runs out (relevance ranking)
#   COMPRESS — How to SHRINK content (summarization, compression)
#   ISOLATE — How to SEPARATE concerns (sub-agents, tool sandboxes)
#
# A well-designed agent uses ALL four pillars together.

def demo_four_pillars():
    """Present the 4 pillars with examples for each."""

    print("\n" + "=" * 65)
    print("  THE 4 PILLARS OF CONTEXT ENGINEERING")
    print("=" * 65)

    pillars = [
        {
            "name": "WRITE",
            "icon": "✎",
            "question": "What information enters the context?",
            "techniques": [
                "System prompt design (clear, concise instructions)",
                "RAG pipeline (retrieve only relevant documents)",
                "Tool output formatting (structured, not verbose)",
                "Few-shot examples (carefully chosen, not random)",
            ],
            "anti_pattern": "Dumping entire documents 'just in case'",
        },
        {
            "name": "SELECT",
            "icon": "⊕",
            "question": "What stays when space runs out?",
            "techniques": [
                "Relevance scoring (rank by similarity to current task)",
                "Recency weighting (prefer recent over old)",
                "Importance tagging (mark critical vs. nice-to-have)",
                "Deduplication (remove repeated information)",
            ],
            "anti_pattern": "FIFO eviction (losing important old context)",
        },
        {
            "name": "COMPRESS",
            "icon": "⊘",
            "question": "How to preserve information in fewer tokens?",
            "techniques": [
                "Conversation summarization (history → bullet points)",
                "Document compression (LLMLingua, extractive summary)",
                "Structured output (JSON/tables vs. prose)",
                "Hierarchical windowing (full → summary → archive)",
            ],
            "anti_pattern": "Summarizing EVERYTHING (lose precision on recent items)",
        },
        {
            "name": "ISOLATE",
            "icon": "☐",
            "question": "How to separate concerns across sub-contexts?",
            "techniques": [
                "Sub-agent delegation (each agent gets its own window)",
                "Tool sandboxing (tool results filtered before injection)",
                "Scoped retrieval (RAG namespace per topic/user)",
                "Memory tiers (L1 fast cache, L2 summary, L3 archive)",
            ],
            "anti_pattern": "One giant context for everything (no separation)",
        },
    ]

    for p in pillars:
        print(f"\n  {'─' * 55}")
        print(f"  {p['icon']} Pillar: {p['name']}")
        print(f"    Question: {p['question']}")
        print(f"    Techniques:")
        for t in p["techniques"]:
            print(f"      • {t}")
        print(f"    Anti-pattern: {p['anti_pattern']}")

    # Decision matrix
    print(f"\n  {'─' * 55}")
    print(f"  WHEN TO USE WHICH PILLAR:")
    print(f"  {'─' * 55}")
    print(f"  • Context is EMPTY → WRITE (fill it with the right stuff)")
    print(f"  • Context is FULL  → SELECT (choose what stays)")
    print(f"  • Content is LONG  → COMPRESS (make it shorter)")
    print(f"  • Task is COMPLEX  → ISOLATE (split across sub-agents)")
    print(f"\n  Production agents apply all 4 pillars continuously.")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 3: Summarization, Compression, Windowing   ║")
    print("╚" + "═" * 63 + "╝")

    demo_map_reduce()
    demo_refine()
    demo_hierarchical()
    demo_compression()
    demo_sliding_window()
    demo_four_pillars()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Three summarization strategies: Map-Reduce (fast, parallel),
       Refine (sequential, better quality), Hierarchical (balanced).
       Choose based on latency vs. quality requirements.

    2. Compression (LLMLingua-style) achieves 2-5× savings safely.
       10× is risky.  Best for boilerplate-heavy content.

    3. Simple sliding windows lose ALL old context.  Hierarchical
       windowing preserves graduated detail across the full history.

    4. The 4 Pillars (Write, Select, Compress, Isolate) cover every
       context engineering technique.  Production agents use all four.

    5. Anti-pattern: "dump everything and hope" — always CURATE what
       enters the context window, just like you'd curate RAM allocation.
    """))
