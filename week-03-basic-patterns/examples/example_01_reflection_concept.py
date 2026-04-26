"""
Example 1: The Reflection Pattern — Self-Critique Without an LLM
=================================================================
The Reflection pattern is one of the most powerful agentic patterns.
Instead of accepting the first output, the agent CRITIQUES its own
work and REFINES it iteratively — just like a human writer who
drafts, reviews, and revises.

This example uses NO LLM — just pure Python — so you can understand
the mechanics of the pattern before adding AI.

The Pattern:
  Generate -> Critique -> Refine -> Critique -> ... -> Good Enough -> Done

Key Concepts:
  - Generator: produces an initial draft
  - Critic: evaluates the draft against criteria
  - Refiner: improves the draft based on critique
  - Quality gate: decides when output is "good enough" to stop

Why This Matters:
  A single LLM call often produces mediocre output. Adding a reflection
  loop dramatically improves quality — the same LLM can catch its own
  mistakes when asked to review its work separately.

Run: python week-03-basic-patterns/examples/example_01_reflection_concept.py
"""


# ==============================================================
# PART 1: Simple Reflection Loop (Pure Python)
# ==============================================================
# Let's simulate the reflection pattern with a text processing task.
# The "generator" writes a summary, the "critic" checks for issues,
# and the "refiner" fixes them.

def generate_summary(topic: str) -> str:
    """Simulate generating a first draft (like an LLM would)."""
    # Intentionally imperfect output — missing details, too short
    return f"{topic} is an important subject that affects many people."


def critique_summary(summary: str, topic: str) -> dict:
    """Simulate critiquing the summary against quality criteria.

    Returns a dict with:
      - passed: bool — whether the summary meets all criteria
      - issues: list[str] — specific problems found
      - score: int — quality score out of 10
    """
    issues = []

    # Criterion 1: Length — summary should be at least 100 characters
    if len(summary) < 100:
        issues.append(f"Too short ({len(summary)} chars, need 100+)")

    # Criterion 2: Must mention specific details (not just vague statements)
    vague_words = ["important", "many", "various", "several"]
    vague_count = sum(1 for word in vague_words if word in summary.lower())
    if vague_count >= 2:
        issues.append(f"Too vague — uses {vague_count} vague words: replace with specifics")

    # Criterion 3: Should have multiple sentences
    sentence_count = summary.count(".") + summary.count("!") + summary.count("?")
    if sentence_count < 3:
        issues.append(f"Only {sentence_count} sentence(s) — need at least 3")

    # Criterion 4: Should reference the topic explicitly
    if topic.lower() not in summary.lower():
        issues.append(f"Does not mention the topic '{topic}'")

    # Calculate a score
    max_possible_issues = 4
    score = max(1, 10 - (len(issues) * 2))

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "score": score,
    }


def refine_summary(summary: str, issues: list) -> str:
    """Simulate refining the summary based on critique feedback.

    In a real agent, the LLM would receive the original text + the
    critique and produce an improved version. Here we simulate it.
    """
    refined = summary

    # Fix: too short — add more content
    if any("Too short" in issue for issue in issues):
        refined += (
            " Research shows significant developments in recent years. "
            "Experts have identified key trends that shape the current landscape. "
            "Understanding these dynamics is crucial for informed decision-making."
        )

    # Fix: too vague — replace vague words with specifics
    if any("vague" in issue.lower() for issue in issues):
        refined = refined.replace("many people", "millions of people worldwide")
        refined = refined.replace("important subject", "critical and evolving field")

    # Fix: missing topic reference — add it back
    # (In practice the LLM handles this naturally)

    return refined


def run_reflection_loop(topic: str, max_iterations: int = 5) -> str:
    """Run the full reflection loop: generate -> critique -> refine -> repeat.

    Args:
        topic: The subject to summarize
        max_iterations: Safety limit to prevent infinite loops

    Returns:
        The final refined summary
    """
    print(f"\n{'='*60}")
    print(f"Reflection Loop: '{topic}'")
    print(f"{'='*60}")

    # Step 1: Generate initial draft
    draft = generate_summary(topic)
    print(f"\n[DRAFT] Initial Draft:\n   \"{draft}\"")

    # Step 2: Iterate — critique and refine until good enough
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Critique the current draft
        critique = critique_summary(draft, topic)
        print(f"   Score: {critique['score']}/10")

        if critique["passed"]:
            print(f"   [PASS] All criteria met! Stopping after {iteration} iteration(s).")
            break

        # Show what needs fixing
        print(f"   Issues found ({len(critique['issues'])}):")
        for issue in critique["issues"]:
            print(f"     - {issue}")

        # Refine based on critique
        draft = refine_summary(draft, critique["issues"])
        print(f"   Refined:\n   \"{draft[:120]}...\"" if len(draft) > 120 else f"   Refined:\n   \"{draft}\"")

    else:
        # max_iterations reached without passing
        print(f"\n   [WARN] Max iterations ({max_iterations}) reached. Returning best effort.")

    return draft


# ==============================================================
# PART 2: Reflection as a State Machine
# ==============================================================
# The reflection pattern can be modeled as a state machine with
# clear transitions. This is exactly how we'll implement it in
# LangGraph (Example 2) and ADK (Example 3).

def demonstrate_state_machine():
    """Show the reflection pattern as explicit states and transitions."""

    print(f"\n{'='*60}")
    print("Reflection as a State Machine")
    print(f"{'='*60}")

    # The states our agent passes through
    states = {
        "generate": "Create initial output",
        "critique": "Evaluate against criteria",
        "refine":   "Improve based on feedback",
        "done":     "Output meets quality bar",
    }

    # The transitions between states
    transitions = [
        ("generate", "critique", "Always — every draft gets reviewed"),
        ("critique", "done", "If score >= 8 (quality gate passed)"),
        ("critique", "refine", "If score < 8 (needs improvement)"),
        ("refine", "critique", "Always — re-evaluate after changes"),
    ]

    print("\nStates:")
    for state, description in states.items():
        print(f"  [{state}] -> {description}")

    print("\nTransitions:")
    for from_state, to_state, condition in transitions:
        print(f"  {from_state} -> {to_state}  (when: {condition})")

    print("\nTypical flow:")
    print("  generate -> critique -> refine -> critique -> refine -> critique -> done")
    print("           \\____________________loop____________________/")

    # Key insight
    print("\n[TIP] Key Insight:")
    print("   The reflection loop is essentially the same graph pattern as")
    print("   Week 2's agent->tools->agent loop, but instead of calling tools,")
    print("   we call a CRITIC. The conditional edge checks quality instead")
    print("   of checking for tool_calls.")


# ==============================================================
# PART 3: Comparing With and Without Reflection
# ==============================================================

def compare_with_without():
    """Show the difference reflection makes."""

    print(f"\n{'='*60}")
    print("Comparison: With vs Without Reflection")
    print(f"{'='*60}")

    topic = "Climate Change"

    # Without reflection — single pass
    single_pass = generate_summary(topic)
    critique_single = critique_summary(single_pass, topic)

    print(f"\n[FAIL] Without Reflection (single pass):")
    print(f"   Output: \"{single_pass}\"")
    print(f"   Score:  {critique_single['score']}/10")
    print(f"   Issues: {len(critique_single['issues'])}")

    # With reflection — multiple passes
    reflected = run_reflection_loop(topic, max_iterations=3)
    critique_reflected = critique_summary(reflected, topic)

    print(f"\n[PASS] With Reflection (after loop):")
    print(f"   Output: \"{reflected[:150]}...\"" if len(reflected) > 150 else f"   Output: \"{reflected}\"")
    print(f"   Score:  {critique_reflected['score']}/10")
    print(f"   Issues: {len(critique_reflected['issues'])}")

    print(f"\n[TIP] Key Takeaway:")
    print(f"   Same generator, same content — but reflection improved the")
    print(f"   score from {critique_single['score']}/10 to {critique_reflected['score']}/10.")
    print(f"   In real agents, the LLM acts as both generator AND critic,")
    print(f"   catching mistakes it wouldn't catch in a single pass.")


# ==============================================================
# Run all demonstrations
# ==============================================================

if __name__ == "__main__":
    # Part 1: Basic reflection loop
    result = run_reflection_loop("Artificial Intelligence", max_iterations=5)

    # Part 2: State machine view
    demonstrate_state_machine()

    # Part 3: With vs without comparison
    compare_with_without()

    print(f"\n{'='*60}")
    print("Next: See example_02 for reflection with a real LLM in LangGraph")
    print(f"{'='*60}")
