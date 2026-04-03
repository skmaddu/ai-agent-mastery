import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 7: Evaluator-Optimizer Pattern & Ralph Wiggum Loop — Concepts
======================================================================
Two powerful meta-patterns that drive quality in production agents:

1. EVALUATOR-OPTIMIZER: Separate the "creator" from the "judge."
   One agent generates output, another scores it, and if the score
   is too low, the first agent revises. This is NOT the same as
   Week 3's reflection (where one agent critiques itself) — here
   the evaluator and optimizer are separate, with different prompts,
   potentially different models, and independent perspectives.

2. RALPH WIGGUM LOOP: A continuous iteration pattern where a simple
   outer loop spawns a FRESH agent each cycle. Progress lives in
   files and test results, NOT in the LLM's context window. Each
   iteration starts clean — no conversation history carried over.

   "Ralph is a Bash loop." — The agent keeps trying, guided by specs
   and test results, like a persistent (if sometimes confused) intern.

Key Concepts (Sections 3, 17 of the Research Bible):
  - Evaluator-Optimizer as a quality assurance pipeline
  - Score tracking and plateau detection
  - Three escape strategies for stuck loops
  - Ralph Wiggum Loop: fresh context, external persistence

Run: python week-04-advanced-patterns/examples/example_07_evaluator_optimizer_concept.py
"""

import random


# ================================================================
# PART 1: Evaluator-Optimizer Pattern
# ================================================================
# The pattern separates creation from judgment:
#
#   ┌───────────┐     ┌───────────┐
#   │ Optimizer  │────>│ Evaluator │
#   │ (creates/ │     │ (scores)  │
#   │  revises) │     └─────┬─────┘
#   └─────┬─────┘           │
#         ^                 │
#         │    score < threshold
#         └─────────────────┘
#                           │ score >= threshold
#                           v
#                     ┌──────────┐
#                     │  Done!   │
#                     └──────────┘
#
# Why separate? Because:
# 1. Different skills: creating and judging are different tasks
# 2. Less bias: self-critique is inherently biased
# 3. Different models: use a cheap model to create, expensive to judge
# 4. Independent prompts: optimizer focuses on quality, evaluator on rubric

def evaluate_text(text: str) -> dict:
    """Simulated evaluator: scores text on three dimensions.

    In a real system, this would be an LLM call with a rubric prompt.
    Here we use simple heuristics to show the mechanics.
    """
    scores = {}

    # Correctness: does it contain specific facts? (not just vague claims)
    specific_indicators = ["percent", "%", "million", "billion", "study", "research",
                          "according to", "data shows", "evidence"]
    fact_count = sum(1 for indicator in specific_indicators if indicator in text.lower())
    scores["correctness"] = min(5, 1 + fact_count)

    # Completeness: does it cover multiple aspects?
    aspect_indicators = ["however", "additionally", "furthermore", "on the other hand",
                        "challenges", "benefits", "risks", "opportunities"]
    aspect_count = sum(1 for a in aspect_indicators if a in text.lower())
    scores["completeness"] = min(5, 1 + aspect_count)

    # Clarity: is it well-structured? (sentences, not too long)
    sentences = text.split(".")
    sentence_count = len([s for s in sentences if len(s.strip()) > 10])
    if 3 <= sentence_count <= 8:
        scores["clarity"] = 4
    elif sentence_count > 0:
        scores["clarity"] = 2
    else:
        scores["clarity"] = 1

    avg = sum(scores.values()) / len(scores)
    return {
        "scores": scores,
        "average": round(avg, 1),
        "feedback": generate_feedback(scores),
    }


def generate_feedback(scores: dict) -> str:
    """Generate specific improvement suggestions based on scores."""
    suggestions = []
    if scores.get("correctness", 0) < 3:
        suggestions.append("Add specific statistics, percentages, or named studies")
    if scores.get("completeness", 0) < 3:
        suggestions.append("Discuss both benefits AND challenges/risks")
    if scores.get("clarity", 0) < 3:
        suggestions.append("Break into 4-6 well-structured sentences")
    if not suggestions:
        suggestions.append("Minor polish needed — text is solid overall")
    return "; ".join(suggestions)


def optimize_text(text: str, feedback: str, iteration: int) -> str:
    """Simulated optimizer: improves text based on feedback.

    In a real system, this would be an LLM call:
      "Revise this text based on the feedback: {feedback}"

    Here we simulate progressive improvement to show the loop mechanics.
    """
    improvements = {
        1: (
            "AI is transforming healthcare in significant ways. "
            "According to research, AI diagnostics can detect diseases with 94% accuracy. "
            "However, challenges remain in data privacy and algorithmic bias."
        ),
        2: (
            "AI is transforming healthcare with measurable impact. "
            "A 2025 study found AI diagnostics detect certain cancers with 94% accuracy, "
            "reducing false negatives by 30%. Additionally, AI-driven drug discovery "
            "has cut development timelines from 10 years to 3. However, challenges remain: "
            "data privacy concerns affect 67% of healthcare institutions, and algorithmic "
            "bias risks unequal care for underrepresented populations."
        ),
        3: (
            "AI is fundamentally transforming healthcare across diagnostics, treatment, "
            "and drug discovery. Research from Stanford (2025) shows AI diagnostics "
            "detect certain cancers with 94% accuracy, reducing false negatives by 30%. "
            "Furthermore, AI-driven drug discovery has compressed development timelines "
            "from 10 years to approximately 3 years, according to McKinsey data. "
            "On the other hand, significant challenges persist: 67% of healthcare "
            "institutions report data privacy concerns, and evidence shows algorithmic "
            "bias can lead to unequal care quality for underrepresented populations. "
            "Despite these risks, the opportunities are substantial — the global AI "
            "healthcare market is projected to reach $187 billion by 2030."
        ),
    }
    return improvements.get(iteration, improvements[3])


def demo_evaluator_optimizer():
    """Run the evaluator-optimizer loop."""
    print("=" * 60)
    print("PART 1: Evaluator-Optimizer Pattern")
    print("=" * 60)

    THRESHOLD = 3.5
    MAX_ITERATIONS = 5

    # Start with a weak first draft
    current_text = "AI is changing healthcare. It helps doctors. This is important."
    print(f"\nGoal: Write about AI in healthcare (threshold: {THRESHOLD}/5)")
    print(f"Initial draft: {current_text}")
    print()

    scores_history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        # EVALUATE
        evaluation = evaluate_text(current_text)
        scores_history.append(evaluation["average"])

        print(f"--- Iteration {iteration} ---")
        print(f"  Scores: {evaluation['scores']}")
        print(f"  Average: {evaluation['average']}/5")
        print(f"  Feedback: {evaluation['feedback']}")

        # CHECK: good enough?
        if evaluation["average"] >= THRESHOLD:
            print(f"  ✓ Score {evaluation['average']} >= threshold {THRESHOLD} — Done!")
            break

        # OPTIMIZE: revise based on feedback
        current_text = optimize_text(current_text, evaluation["feedback"], iteration)
        print(f"  Revised text: {current_text[:80]}...")
        print()

    print(f"\nFinal text ({len(current_text)} chars):")
    print(f"  {current_text}")
    print(f"\nScore progression: {scores_history}")
    print()


# ================================================================
# PART 2: Plateau Detection — The "Ralph Wiggum Problem"
# ================================================================
# Sometimes the loop gets stuck: the score stops improving but
# hasn't reached the threshold. This is the "Ralph Wiggum problem"
# — the agent keeps trying the same approach expecting different
# results.
#
# Detection: if score doesn't improve for N consecutive iterations,
# the agent is stuck and needs a strategy change.

def detect_plateau(scores: list, window: int = 2) -> bool:
    """Detect if scores have plateaued.

    Returns True if the last `window` scores show no improvement.
    """
    if len(scores) < window + 1:
        return False
    recent = scores[-window:]
    previous = scores[-(window + 1)]
    # Plateau = no score in recent window is better than the one before
    return all(s <= previous for s in recent)


def demo_plateau_detection():
    """Show plateau detection and escape strategies."""
    print("=" * 60)
    print("PART 2: Plateau Detection")
    print("=" * 60)

    # Simulate a stuck loop
    scores_stuck = [2.0, 3.0, 3.5, 3.5, 3.5, 3.5]
    scores_improving = [2.0, 3.0, 3.5, 3.8, 4.0, 4.2]

    print("\nScenario 1 (stuck):", scores_stuck)
    for i in range(2, len(scores_stuck)):
        is_plateau = detect_plateau(scores_stuck[:i+1], window=2)
        if is_plateau:
            print(f"  After score {scores_stuck[i]}: PLATEAU DETECTED at iteration {i+1}")
            break

    print("\nScenario 2 (improving):", scores_improving)
    plateau_found = False
    for i in range(2, len(scores_improving)):
        if detect_plateau(scores_improving[:i+1], window=2):
            plateau_found = True
            break
    if not plateau_found:
        print("  No plateau — scores keep improving ✓")

    print("""
Three Escape Strategies for Plateaus:

  1. INCREASE TEMPERATURE — make the LLM more creative
     (0.7 → 0.9). More randomness = different solutions.

  2. CHANGE THE PROMPT — add "Try a completely different approach"
     or "Ignore your previous attempts and start fresh."

  3. ESCALATE — if 2+ strategy changes fail, give up gracefully
     and return the best result so far with a note that it didn't
     fully meet the threshold.

Trade-off (Topic 16): More iterations = better quality but higher
cost. Set a max_iterations cap AND a budget cap.
""")


# ================================================================
# PART 3: The Ralph Wiggum Loop
# ================================================================
# The Ralph Wiggum Loop is fundamentally different from an in-context
# improvement loop:
#
# REGULAR LOOP (in-context):
#   Agent remembers all previous attempts in its context window.
#   Problem: context grows → model gets confused → quality degrades.
#
# RALPH LOOP (fresh context):
#   Each iteration starts a FRESH agent with NO conversation history.
#   Progress lives in FILES (code, test results, specs), not in the LLM.
#
#   ┌─────────────────────────────────────────────┐
#   │  while not done and iterations < max:       │
#   │    1. Read: spec + current files + test log │
#   │    2. Invoke: fresh agent (NO history)      │
#   │    3. Agent modifies files                  │
#   │    4. Run: tests / quality gates            │
#   │    5. If pass → done                        │
#   │    6. Else → log failures, loop back        │
#   └─────────────────────────────────────────────┘
#
# Key insight: the agent never sees its PREVIOUS ATTEMPTS — only
# the CURRENT STATE of the artifacts. This prevents context rot.

class SimulatedFile:
    """Simulates a file on disk for the Ralph Loop demo."""
    def __init__(self, content: str = ""):
        self.content = content
        self.version = 0


def simulated_quality_gate(content: str) -> dict:
    """Simulated test suite / quality check.

    In a real Ralph Loop, this would be pytest, ruff, or an LLM judge.
    """
    checks = {
        "has_intro": "introduction" in content.lower() or "overview" in content.lower(),
        "has_data": any(c.isdigit() for c in content),
        "has_conclusion": "conclusion" in content.lower() or "summary" in content.lower(),
        "min_length": len(content) > 200,
        "has_sections": content.count("\n\n") >= 2,
    }
    passed = sum(checks.values())
    total = len(checks)
    return {
        "passed": passed == total,
        "score": f"{passed}/{total}",
        "checks": checks,
        "failures": [k for k, v in checks.items() if not v],
    }


def simulated_agent_iteration(spec: str, current_content: str, iteration: int) -> str:
    """Simulates a fresh agent improving content based on spec + current state.

    In a real Ralph Loop, this would be an LLM call with NO history:
      "Here is the spec and current file. Improve it."
    """
    # Each iteration adds something the quality gate checks for
    versions = [
        # Iteration 1: basic content
        "AI in Healthcare\n\nAI is changing how we approach medicine.",
        # Iteration 2: add introduction and data
        (
            "AI in Healthcare: An Overview\n\n"
            "Introduction: Artificial intelligence is revolutionizing healthcare "
            "across diagnostics, treatment, and research. Studies show a 40% "
            "improvement in diagnostic accuracy when AI assists doctors.\n\n"
            "Current applications span radiology, pathology, and drug discovery."
        ),
        # Iteration 3: add conclusion and sections
        (
            "AI in Healthcare: An Overview\n\n"
            "Introduction: Artificial intelligence is revolutionizing healthcare "
            "across diagnostics, treatment, and research.\n\n"
            "Key findings: Studies show a 40% improvement in diagnostic accuracy "
            "when AI assists doctors. Over 500 AI medical devices have received "
            "FDA approval as of 2025. Drug discovery timelines have been reduced "
            "by up to 60% using AI-driven molecular analysis.\n\n"
            "Challenges include data privacy, algorithmic bias, and the need for "
            "clinician training on AI-augmented workflows.\n\n"
            "Conclusion: AI represents a transformative force in healthcare, with "
            "measurable benefits already visible. Summary: the technology is ready, "
            "but governance and education must catch up."
        ),
    ]
    idx = min(iteration - 1, len(versions) - 1)
    return versions[idx]


def demo_ralph_loop():
    """Demonstrate the Ralph Wiggum Loop pattern."""
    print("=" * 60)
    print("PART 3: The Ralph Wiggum Loop")
    print("=" * 60)

    spec = "Write a report on AI in healthcare with intro, data, and conclusion"
    output_file = SimulatedFile()
    MAX_ITERATIONS = 5

    print(f"\nSpec: {spec}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'='*40}")
        print(f"  RALPH ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*40}")

        # Step 1: Read current state (spec + file)
        print(f"  Reading: spec + current file (v{output_file.version})")

        # Step 2: Fresh agent (no history from previous iterations!)
        new_content = simulated_agent_iteration(
            spec, output_file.content, iteration
        )
        output_file.content = new_content
        output_file.version += 1
        print(f"  Agent wrote {len(new_content)} chars (v{output_file.version})")

        # Step 3: Run quality gates
        gate_result = simulated_quality_gate(new_content)
        print(f"  Quality gate: {gate_result['score']}")

        if gate_result["passed"]:
            print(f"  ✓ ALL GATES PASSED — Ralph is done!")
            break
        else:
            print(f"  ✗ Failures: {gate_result['failures']}")
            print(f"  → Will retry with fresh agent next iteration")
        print()

    print(f"\nFinal output (v{output_file.version}):")
    print(f"  {output_file.content[:120]}...")
    print()


# ================================================================
# PART 4: Evaluator-Optimizer vs. Ralph Loop — When to Use Which
# ================================================================

def demo_comparison():
    """Compare the two patterns."""
    print("=" * 60)
    print("PART 4: Evaluator-Optimizer vs. Ralph Loop")
    print("=" * 60)
    print("""
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Dimension           │ Evaluator-Optimizer  │ Ralph Wiggum Loop    │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Context             │ In-context (shared   │ Fresh each iteration │
│                     │ conversation)        │ (no history)         │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Memory              │ LLM context window   │ Files, git, DB       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Progress tracking   │ State in graph/loop  │ Test results, specs  │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Context rot risk    │ HIGH (grows with     │ NONE (fresh each     │
│                     │ each iteration)      │ time)                │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Best for            │ Short tasks (2-5     │ Long tasks (code     │
│                     │ iterations)          │ gen, large docs)     │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Quality gate        │ LLM-as-judge score   │ Tests, linters, or   │
│                     │                      │ LLM judge            │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Cost model          │ Incremental (each    │ Flat per iteration   │
│                     │ round adds context)  │ (no growing context) │
└─────────────────────┴──────────────────────┴──────────────────────┘

Decision guide:
  - Need quick refinement of a paragraph? → Evaluator-Optimizer
  - Building a whole codebase overnight?  → Ralph Wiggum Loop
  - Worried about context window limits?  → Ralph Wiggum Loop
  - Want LLM to remember its mistakes?    → Evaluator-Optimizer
  - Want deterministic context each time? → Ralph Wiggum Loop

Common misconception: "Ralph is just a retry loop."
  NO. A retry loop re-runs the SAME thing hoping for different results.
  Ralph reads UPDATED ARTIFACTS each iteration, so it has genuinely
  new information. It's convergent, not repetitive.
""")


# ================================================================
# Main: Run all demos
# ================================================================

if __name__ == "__main__":
    print()
    print("Example 7: Evaluator-Optimizer & Ralph Wiggum Loop — Concepts")
    print("=" * 60)
    print("No LLM required — pure Python demonstration")
    print()

    demo_evaluator_optimizer()
    demo_plateau_detection()
    demo_ralph_loop()
    demo_comparison()

    print("=" * 60)
    print("Next: See example_08 for LangGraph implementation")
    print("      and example_09 for ADK implementation.")
    print("=" * 60)
