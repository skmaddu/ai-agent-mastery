"""
Example 11: Why Patterns Improve Agent Intelligence
=====================================================
A single LLM call is like asking someone to answer a question without
thinking. Patterns add STRUCTURE to how agents think, making them
dramatically better at complex tasks.

This example demonstrates the difference through side-by-side
comparisons using a real LLM:
  1. Single-shot vs Reflection (quality improvement)
  2. No tools vs Tool-use (factual accuracy)
  3. No routing vs Smart routing (efficiency)

Run: python week-03-basic-patterns/examples/example_11_why_patterns_matter.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================
# Setup
# ==============================================================

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.7)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7)


llm = get_llm()


# ==============================================================
# COMPARISON 1: Single-Shot vs Reflection
# ==============================================================

def compare_single_vs_reflection():
    """Show how reflection improves output quality."""

    print("=" * 60)
    print("COMPARISON 1: Single-Shot vs Reflection")
    print("=" * 60)

    topic = "The future of renewable energy"

    # -- Single-shot: one LLM call ----------------------------
    print(f"\n  A) Single-shot (1 LLM call):")
    single = llm.invoke([
        SystemMessage(content="Write a concise, informative paragraph about the given topic."),
        HumanMessage(content=topic),
    ])
    print(f"     \"{single.content[:200]}...\"" if len(single.content) > 200 else f"     \"{single.content}\"")

    # -- Reflection: generate + critique + refine --------------
    print(f"\n  B) With Reflection (3 LLM calls):")

    # Step 1: Generate
    draft = llm.invoke([
        SystemMessage(content="Write a concise, informative paragraph about the given topic."),
        HumanMessage(content=topic),
    ]).content

    # Step 2: Critique
    critique = llm.invoke([
        SystemMessage(content=(
            "Critique this text. List specific weaknesses: missing facts, "
            "vague language, unsupported claims. Be strict."
        )),
        HumanMessage(content=draft),
    ]).content

    # Step 3: Refine
    refined = llm.invoke([
        SystemMessage(content=(
            "Rewrite the text addressing ALL issues in the critique. "
            "Keep strengths, fix weaknesses. Output only the revised paragraph."
        )),
        HumanMessage(content=f"Original:\n{draft}\n\nCritique:\n{critique}\n\nRevise:"),
    ]).content

    print(f"     \"{refined[:200]}...\"" if len(refined) > 200 else f"     \"{refined}\"")

    print(f"\n  [TIP] Reflection uses 3x the tokens but typically produces")
    print(f"     significantly better output — more specific, fewer gaps,")
    print(f"     better structure. The LLM catches its own mistakes.")


# ==============================================================
# COMPARISON 2: No Tools vs Tool Use
# ==============================================================

def compare_no_tools_vs_tools():
    """Show how tools improve factual accuracy."""

    print(f"\n{'='*60}")
    print("COMPARISON 2: No Tools vs With Tools")
    print("=" * 60)

    question = "What percentage of global electricity comes from renewable sources?"

    # -- Without tools: LLM relies on training data -----------
    print(f"\n  A) Without tools (LLM memory only):")
    no_tools = llm.invoke([
        SystemMessage(content="Answer factual questions. If unsure, say so."),
        HumanMessage(content=question),
    ])
    print(f"     \"{no_tools.content[:200]}\"")

    # -- With tools: LLM gets real data ------------------------
    print(f"\n  B) With tools (search result injected):")

    # Simulate a tool providing current data
    tool_result = "Renewable energy now accounts for 35% of global electricity generation (2026 data)."

    with_tools = llm.invoke([
        SystemMessage(content="Answer using the provided search results. Cite the data."),
        HumanMessage(content=f"Question: {question}\n\nSearch result: {tool_result}"),
    ])
    print(f"     \"{with_tools.content[:200]}\"")

    print(f"\n  [TIP] Without tools, the LLM relies on training data (may be outdated).")
    print(f"     With tools, it accesses current information and can cite sources.")
    print(f"     Tools turn agents from 'reasoners' into 'researchers'.")


# ==============================================================
# COMPARISON 3: The Compound Effect
# ==============================================================

def demonstrate_compound_effect():
    """Show how combining patterns multiplies the improvement."""

    print(f"\n{'='*60}")
    print("COMPARISON 3: The Compound Effect")
    print("=" * 60)

    print("""
  Patterns don't just add up — they MULTIPLY.

  Single LLM call:
    Quality: ##........ 3/10
    The LLM does its best in one shot.

  + Reflection pattern:
    Quality: ######.... 6/10
    The LLM catches and fixes its own mistakes.

  + Tool use:
    Quality: ########.. 8/10
    The LLM accesses real data instead of guessing.

  + Reflection + Tools:
    Quality: #########. 9/10
    Real data + self-critique = publication-quality output.

  + HITL approval:
    Quality: ########## 10/10
    Human catches the remaining edge cases.

  This is why patterns matter. Each one addresses a DIFFERENT
  weakness of raw LLM output:
    - Reflection   -> fixes quality and accuracy
    - Tool use     -> fixes knowledge gaps
    - Routing      -> fixes efficiency (right tool for right task)
    - HITL         -> fixes trust and edge cases
    - Evaluation   -> fixes blind spots (measures what matters)
""")


# ==============================================================
# The Pattern Decision Tree
# ==============================================================

def pattern_decision_tree():
    """Help students choose which pattern to use when."""

    print(f"{'='*60}")
    print("Pattern Decision Tree: Which Pattern Do I Need?")
    print("=" * 60)

    print("""
  Start here: What's wrong with my agent's output?

  "Output quality is inconsistent"
    -> Use REFLECTION (self-critique + refine loop)

  "Agent makes up facts or uses outdated information"
    -> Use TOOL USE (give it search, database, API access)

  "Agent wastes time/tokens on simple queries"
    -> Use ROUTING (classify first, then dispatch to handlers)

  "I can't trust the agent's decisions for important actions"
    -> Use HITL (approval gate, confidence threshold, escalation)

  "I don't know if my changes improve the agent"
    -> Use EVALS (LLM-as-judge, eval pipeline, regression tests)

  "All of the above"
    -> Combine patterns! Start with tool use + reflection,
      add HITL for high-stakes actions, eval to measure progress.
""")


if __name__ == "__main__":
    print("Example 11: Why Patterns Improve Agent Intelligence")
    print("=" * 60)

    compare_single_vs_reflection()
    compare_no_tools_vs_tools()
    demonstrate_compound_effect()
    pattern_decision_tree()
