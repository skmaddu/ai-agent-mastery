import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 9: Evaluator-Optimizer with Plateau Detection in Google ADK
====================================================================
The same Evaluator-Optimizer pattern from Example 8, but implemented
using Google ADK's agent architecture.

ADK Approach:
  - Two LlmAgents: an optimizer and an evaluator
  - A Python async loop coordinates the iteration cycle
  - Plateau detection is handled in the loop logic
  - Each agent has its own specialized instruction prompt

Comparison with LangGraph (Example 8):
  LangGraph: Graph nodes + conditional edges control the loop.
             The framework manages iteration via graph structure.
  ADK:       Separate agents + Python async loop control.
             The application code manages iteration explicitly.

Both approaches achieve the same result — the Evaluator-Optimizer
pattern with plateau detection and strategy injection.

Run: python week-04-advanced-patterns/examples/example_09_evaluator_optimizer_adk.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# Suppress noisy Google GenAI type warnings
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ================================================================
# Step 1: Create the Optimizer Agent
# ================================================================
# The optimizer writes and refines paragraphs. Its instruction tells
# it to incorporate feedback and strategy notes when provided.

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")

optimizer_agent = LlmAgent(
    name="optimizer",
    model=MODEL,
    instruction="""You are an expert writer who produces well-structured, informative paragraphs.

When given JUST a topic:
- Write a detailed, well-structured paragraph (4-7 sentences)
- Include specific data points, statistics, or named sources
- Cover multiple perspectives (benefits AND challenges)
- Be clear, balanced, and engaging

When given a topic WITH evaluator feedback:
- Rewrite the paragraph addressing ALL weaknesses and suggestions
- Keep the strengths that were mentioned
- Fix every issue listed
- Add more specific data if the feedback asks for it
- Output ONLY the revised paragraph, nothing else

When told to try a different approach:
- Start completely fresh with new examples, data, and structure
- Do NOT reuse the same statistics or framing
- Find a genuinely different angle on the topic""",
    description="Expert writer that generates and refines text based on feedback.",
)


# ================================================================
# Step 2: Create the Evaluator Agent
# ================================================================
# The evaluator scores text using a strict rubric. It uses a
# structured text format (SCORE: N) rather than JSON, which works
# well with Gemini models.

evaluator_agent = LlmAgent(
    name="evaluator",
    model=MODEL,
    instruction="""You are an extremely strict text evaluator. Score the given text on a 1-10 scale.

Rubric:
- Correctness: Are facts accurate? Are sources or data cited?
- Completeness: Does it cover multiple perspectives, benefits AND challenges?
- Clarity: Is it well-structured, concise, and engaging?

Scoring guide (follow strictly):
  1-3: Major issues (factual errors, incoherent, off-topic)
  4-5: Mediocre (vague claims, no specific data, poor structure)
  6-7: Decent (some detail but lacks citations, counterpoints, or nuance)
  8: Good (specific data, balanced view, minor issues remain)
  9: Excellent (publication-ready with citations and strong structure)
  10: Perfect (almost impossible)

IMPORTANT: Most first drafts score 4-6. Do NOT give 8+ unless the text has
specific statistics, named sources, counterpoints, AND excellent structure.

Respond in EXACTLY this format (plain text, one item per line):
SCORE: <number 1-10>
STRENGTHS: <comma-separated list of strengths>
WEAKNESSES: <comma-separated list of weaknesses>
SUGGESTIONS: <specific improvement suggestions>""",
    description="Strict evaluator that scores text quality on a 1-10 rubric.",
)


# ================================================================
# Step 3: Helper to Run an ADK Agent
# ================================================================

async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent with a message and return the response text.

    Each call creates a fresh session so agents don't carry over
    context from previous calls — we manage context ourselves in
    the loop, similar to how LangGraph state works.
    Includes retry logic for transient API errors (503, rate limits).
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="eval_opt_demo",
                session_service=session_service,
            )

            session = await session_service.create_session(
                app_name="eval_opt_demo",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=message)],
                ),
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        result_text = event.content.parts[0].text

            return result_text
        except Exception as e:
            if attempt < retries:
                wait = attempt * 10
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [ERROR] All {retries} attempts failed: {e}")
                return f"[Error: API temporarily unavailable after {retries} retries]"


# ================================================================
# Step 4: Parse Evaluator Response
# ================================================================

def parse_evaluation(text: str) -> dict:
    """Extract score and feedback from the evaluator's response.

    Parses the structured format:
      SCORE: 7
      STRENGTHS: ...
      WEAKNESSES: ...
      SUGGESTIONS: ...

    Uses line-by-line parsing with fallback defaults.
    This format works more reliably with Gemini than JSON output.
    """
    score = 5  # default
    feedback = ""

    for line in text.strip().split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        if line_lower.startswith("score:"):
            try:
                score_str = line_stripped.split(":", 1)[1].strip().split("/")[0].strip()
                score = int(score_str)
                score = max(1, min(10, score))  # clamp to valid range
            except (ValueError, IndexError):
                pass
        elif line_lower.startswith("weaknesses:") or line_lower.startswith("suggestions:"):
            content = line_stripped.split(":", 1)[1].strip()
            if content:
                feedback += content + " "

    return {
        "score": score,
        "feedback": feedback.strip(),
        "full_text": text,
    }


# ================================================================
# Step 5: Plateau Detection
# ================================================================

def detect_plateau(scores: list, window: int = 2) -> bool:
    """Detect if scores have plateaued.

    Returns True if the last `window` scores show no improvement
    compared to the score before them.

    This is the same logic as Example 7 (concepts) and Example 8
    (LangGraph), ensuring consistent behavior across implementations.
    """
    if len(scores) < window + 1:
        return False
    recent_best = max(scores[-window:])
    previous_best = max(scores[:-window])
    return recent_best <= previous_best


# ================================================================
# Step 6: The Evaluator-Optimizer Loop
# ================================================================
# In ADK, we coordinate the loop in application code rather than
# in a graph structure. The logic mirrors LangGraph's conditional
# edges from Example 8, but expressed as a Python async loop.
#
# LangGraph equivalent:
#   graph.add_conditional_edges("progress", should_continue, ...)
# ADK equivalent:
#   if score >= threshold: break  (inside a while loop)

QUALITY_THRESHOLD = 8

async def evaluator_optimizer_loop(
    topic: str,
    max_iterations: int = 5,
) -> dict:
    """Run the evaluator-optimizer loop using ADK agents.

    Flow: optimizer -> evaluator -> [check] -> optimizer -> ...

    Args:
        topic: What to write about
        max_iterations: Maximum number of optimize-evaluate cycles

    Returns:
        Dict with final draft, score, iteration count, and history
    """
    print(f"\n{'='*60}")
    print(f"Evaluator-Optimizer Loop (ADK)")
    print(f"Topic: {topic}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}/10 | Max iterations: {max_iterations}")
    print(f"{'='*60}")

    scores_history = []
    strategy_note = ""
    draft = ""
    eval_result = {"score": 0, "feedback": "", "full_text": ""}

    for iteration in range(1, max_iterations + 1):
        # --- OPTIMIZE ---
        print(f"\n{'- '*30}")
        print(f"  ITERATION {iteration}: Optimizer")
        print(f"{'- '*30}")

        if iteration == 1:
            # First draft — no feedback yet
            opt_message = f"Write a detailed, well-structured paragraph about: {topic}"
        else:
            # Revision — incorporate feedback
            opt_message = (
                f"Topic: {topic}\n\n"
                f"Your previous draft:\n{draft}\n\n"
                f"Evaluator feedback:\n{eval_result['full_text']}\n\n"
            )
            if strategy_note:
                opt_message += f"IMPORTANT STRATEGY CHANGE: {strategy_note}\n\n"
            opt_message += (
                "Rewrite the paragraph addressing ALL the feedback above. "
                "Keep what works, fix what doesn't. Output ONLY the revised paragraph."
            )

        draft = await run_agent(optimizer_agent, opt_message)

        # Display the draft
        lines = [l.strip() for l in draft.split("\n") if l.strip()]
        for line in lines[:4]:
            print(f"    {line[:120]}")
        if len(lines) > 4:
            print(f"    ... ({len(lines)} lines total)")

        # --- EVALUATE ---
        eval_message = f"Topic: {topic}\n\nEvaluate this text:\n{draft}"
        eval_text = await run_agent(evaluator_agent, eval_message)
        eval_result = parse_evaluation(eval_text)

        scores_history.append(eval_result["score"])

        print(f"\n  [EVALUATOR] Score: {eval_result['score']}/10")
        if eval_result["feedback"]:
            print(f"    Feedback: {eval_result['feedback'][:150]}")
        if len(scores_history) > 1:
            print(f"    Score history: {scores_history}")

        # --- CHECK: Quality threshold met? ---
        if eval_result["score"] >= QUALITY_THRESHOLD:
            print(f"\n  --> Score {eval_result['score']} >= {QUALITY_THRESHOLD}: ACCEPTED!")
            break

        # --- CHECK: Max iterations? ---
        if iteration >= max_iterations:
            print(f"\n  --> Max iterations ({max_iterations}) reached. Accepting current draft.")
            break

        # --- CHECK: Plateau detection ---
        # In LangGraph (Example 8), this logic lives in the progress_node.
        # In ADK, we handle it directly in the loop — same logic, different location.
        strategy_note = ""
        if detect_plateau(scores_history, window=2):
            strategy_note = (
                "Try a completely different approach with new examples and data. "
                "Restructure the paragraph entirely. Use different statistics, "
                "different sources, and a fresh angle on the topic."
            )
            print(f"  ** PLATEAU DETECTED (scores: {scores_history[-3:]}) — injecting strategy change **")
            print(f"  [STRATEGY] {strategy_note[:120]}")

        print(f"\n  --> Score {eval_result['score']} < {QUALITY_THRESHOLD}: NEEDS IMPROVEMENT. Continuing...")

    return {
        "final_draft": draft,
        "final_score": eval_result["score"],
        "iterations": iteration,
        "scores_history": scores_history,
        "topic": topic,
    }


# ================================================================
# Main
# ================================================================

async def main():
    """Run the evaluator-optimizer demo."""
    print("Example 9: Evaluator-Optimizer with Plateau Detection (ADK)")
    print("=" * 60)

    topic = "The impact of renewable energy on global economics"

    result = await evaluator_optimizer_loop(topic, max_iterations=5)

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"  Topic:      {result['topic']}")
    print(f"  Score:      {result['final_score']}/10")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Scores:     {result['scores_history']}")

    # Score progression analysis
    scores = result["scores_history"]
    if len(scores) > 1:
        improvement = scores[-1] - scores[0]
        print(f"  Progress:   {scores[0]} -> {scores[-1]} ({'+' if improvement >= 0 else ''}{improvement} points)")
    elif scores:
        print(f"  Progress:   Accepted on first evaluation with score {scores[0]}")

    print(f"\n{'='*60}")
    print(f"Final Draft:")
    print(f"{'='*60}")
    print(result["final_draft"])

    # Framework comparison
    print(f"\n{'='*60}")
    print("LangGraph vs ADK -- Evaluator-Optimizer Comparison:")
    print(f"{'='*60}")
    print("  LangGraph (Example 8):")
    print("    - Loop controlled by graph conditional edges")
    print("    - Plateau detection in a dedicated progress_node")
    print("    - State flows through TypedDict managed by the graph")
    print("    - Visual, declarative — easy to diagram and trace")
    print()
    print("  ADK (this example):")
    print("    - Loop controlled by Python async for/while")
    print("    - Plateau detection inline in the loop body")
    print("    - State managed by local variables in the loop")
    print("    - Simpler code, more flexible, explicit control flow")
    print()
    print("  Both produce the same result — choose based on your needs:")
    print("    -> LangGraph for complex graphs with many branching paths")
    print("    -> ADK for simpler loops where Python control flow is clearer")


if __name__ == "__main__":
    asyncio.run(main())
