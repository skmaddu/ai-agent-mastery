import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 8: Evaluator-Optimizer with Plateau Detection in LangGraph
===================================================================
The Evaluator-Optimizer pattern from Example 7, now implemented as a
real LangGraph graph with actual LLM calls.

Two separate LLM roles:
  - Optimizer: writes/rewrites a paragraph on a given topic
  - Evaluator: scores the draft on a strict rubric (1-10)

The loop runs until the evaluator gives a score >= 8, a plateau is
detected (score stuck for 2 rounds), or max iterations are reached.

Plateau Detection ("Ralph Wiggum Problem"):
  If the score doesn't improve for 2 consecutive rounds, the optimizer
  is stuck. We inject a strategy_note telling it to try a completely
  different approach — this is one of the three escape strategies
  from Example 7.

Graph:
  START -> optimizer -> evaluator -> progress
                         ^              |
                         |              | (needs improvement)
                         +--------------+
                                        | (done)
                                        v
                                       END

Run: python week-04-advanced-patterns/examples/example_08_evaluator_optimizer_langgraph.py
"""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# Step 1: LLM Factory
# ================================================================

def get_llm(temperature=0.7):
    """Get the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


# ================================================================
# Step 2: State Definition
# ================================================================
# TypedDict defines the data that flows through the graph.
# Every node reads from and writes to this shared state.

class EvalOptState(TypedDict):
    topic: str                # What to write about
    current_draft: str        # The latest version of the text
    eval_score: int           # Most recent evaluation score (1-10)
    eval_feedback: str        # Evaluator's improvement suggestions
    scores_history: list      # All scores so far — used for plateau detection
    iteration: int            # Current iteration number
    max_iterations: int       # Stop after this many iterations
    strategy_note: str        # Injected when plateau detected — tells optimizer to change approach


# ================================================================
# Step 3: JSON Parsing Helper
# ================================================================

def parse_eval_json(text: str) -> dict:
    """Parse the evaluator's JSON response with fallback handling.

    The evaluator is instructed to return JSON, but LLMs sometimes
    wrap it in markdown code blocks or produce malformed output.
    This function handles those cases gracefully.
    """
    # Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        score = int(parsed.get("score", 5))
        # Clamp to valid range
        score = max(1, min(10, score))
        strengths = parsed.get("strengths", [])
        weaknesses = parsed.get("weaknesses", [])
        suggestion = parsed.get("suggestion", "")
        return {
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestion": suggestion,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: try to extract a score from the raw text
        score = 5  # default
        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        if score_match:
            score = max(1, min(10, int(score_match.group(1))))
        return {
            "score": score,
            "strengths": [],
            "weaknesses": [],
            "suggestion": text[:200],
        }


# ================================================================
# Step 4: Graph Nodes
# ================================================================
# Each node is a function that takes state and returns a partial
# state update. LangGraph merges the update into the full state.

def optimizer_node(state: EvalOptState) -> dict:
    """Generate or revise a draft based on topic, feedback, and strategy.

    First iteration: write a fresh paragraph.
    Later iterations: rewrite incorporating evaluator feedback.
    If a strategy_note is present (plateau detected), the optimizer
    is told to try a completely different approach.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(temperature=0.7)

    system_prompt = (
        "You are an expert writer who produces well-structured, informative paragraphs. "
        "Include specific data points, statistics, named sources, or concrete examples. "
        "Your writing should be clear, balanced, and demonstrate depth of knowledge."
    )

    if state["iteration"] == 0:
        # First draft — no feedback yet
        human_msg = f"Write a detailed, well-structured paragraph about: {state['topic']}"
    else:
        # Revision — incorporate feedback
        human_msg = (
            f"Topic: {state['topic']}\n\n"
            f"Your previous draft:\n{state['current_draft']}\n\n"
            f"Evaluator feedback:\n{state['eval_feedback']}\n\n"
        )
        if state.get("strategy_note"):
            human_msg += (
                f"IMPORTANT STRATEGY CHANGE: {state['strategy_note']}\n\n"
            )
        human_msg += (
            "Rewrite the paragraph addressing ALL the feedback above. "
            "Keep what works, fix what doesn't. Output ONLY the revised paragraph."
        )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]
    response = llm.invoke(messages)
    draft = response.content.strip()

    return {
        "current_draft": draft,
        "iteration": state["iteration"] + 1,
    }


def evaluator_node(state: EvalOptState) -> dict:
    """Score the current draft using a strict rubric.

    Uses temperature=0 for consistent, deterministic evaluation.
    Returns structured JSON with score, strengths, weaknesses, suggestion.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    # Use temperature=0 for consistent evaluation
    llm = get_llm(temperature=0)

    system_prompt = (
        "You are an extremely strict text evaluator. Score the given text on a 1-10 scale.\n\n"
        "Rubric:\n"
        "- Correctness: Are facts accurate? Are sources/data cited?\n"
        "- Completeness: Does it cover multiple perspectives, benefits AND challenges?\n"
        "- Clarity: Is it well-structured, concise, and engaging?\n\n"
        "Scoring guide (follow strictly):\n"
        "  1-3: Major issues (factual errors, incoherent, off-topic)\n"
        "  4-5: Mediocre (vague claims, no specific data, poor structure)\n"
        "  6-7: Decent (some detail but lacks citations, counterpoints, or nuance)\n"
        "  8: Good (specific data, balanced view, minor issues remain)\n"
        "  9: Excellent (publication-ready with citations and strong structure)\n"
        "  10: Perfect (almost impossible)\n\n"
        "IMPORTANT: Most first drafts score 4-6. Do NOT give 8+ unless the text has "
        "specific statistics, named sources, counterpoints, AND excellent structure.\n\n"
        "Respond with ONLY valid JSON in this format:\n"
        '{"score": <number>, "strengths": ["..."], "weaknesses": ["..."], "suggestion": "..."}'
    )

    human_msg = (
        f"Topic: {state['topic']}\n\n"
        f"Text to evaluate:\n{state['current_draft']}"
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]
    response = llm.invoke(messages)

    parsed = parse_eval_json(response.content)

    # Build feedback string for the optimizer
    feedback_parts = []
    if parsed["strengths"]:
        feedback_parts.append(f"Strengths: {', '.join(parsed['strengths'])}")
    if parsed["weaknesses"]:
        feedback_parts.append(f"Weaknesses: {', '.join(parsed['weaknesses'])}")
    if parsed["suggestion"]:
        feedback_parts.append(f"Suggestion: {parsed['suggestion']}")
    feedback_str = "\n".join(feedback_parts) if feedback_parts else "No specific feedback provided."

    # Append new score to history
    updated_history = list(state.get("scores_history", [])) + [parsed["score"]]

    return {
        "eval_score": parsed["score"],
        "eval_feedback": feedback_str,
        "scores_history": updated_history,
    }


def progress_node(state: EvalOptState) -> dict:
    """Check progress and detect plateaus.

    Three possible outcomes:
      1. Score >= 8         -> done (quality met)
      2. Plateau detected   -> inject strategy_note and continue
      3. Max iterations hit -> done (give up gracefully)
      4. Otherwise          -> continue improving

    Plateau detection: if the score hasn't improved for 2 consecutive
    rounds, the optimizer is stuck and needs a strategy change.
    """
    scores = state.get("scores_history", [])
    iteration = state["iteration"]
    max_iter = state["max_iterations"]

    # Check for plateau: score didn't improve for 2 rounds
    strategy_note = ""
    if len(scores) >= 3:
        # Compare last 2 scores against the one before them
        recent_best = max(scores[-2:])
        previous_best = max(scores[:-2])
        if recent_best <= previous_best:
            strategy_note = (
                "Try a completely different approach with new examples and data. "
                "Restructure the paragraph entirely. Use different statistics, "
                "different sources, and a fresh angle on the topic."
            )
            print(f"  ** PLATEAU DETECTED (scores: {scores[-3:]}) — injecting strategy change **")

    return {
        "strategy_note": strategy_note,
    }


# ================================================================
# Step 5: Routing Function
# ================================================================

def should_continue(state: EvalOptState) -> str:
    """Decide whether to continue the loop or finish.

    This is the conditional edge after the progress node.
    """
    if state["eval_score"] >= 8:
        return "done"
    if state["iteration"] >= state["max_iterations"]:
        return "done"
    return "improve"


# ================================================================
# Step 6: Build the Graph
# ================================================================

def build_eval_opt_graph():
    """Construct the evaluator-optimizer LangGraph.

    Graph flow:
      START -> optimizer -> evaluator -> progress -> [improve|done]
                              ^                        |
                              |     (improve)          |
                              +------------------------+
                                       (done) -> END
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(EvalOptState)

    # Add nodes
    graph.add_node("optimizer", optimizer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("progress", progress_node)

    # Set entry point
    graph.set_entry_point("optimizer")

    # Add edges
    graph.add_edge("optimizer", "evaluator")
    graph.add_edge("evaluator", "progress")

    # Conditional edge from progress: continue or stop
    graph.add_conditional_edges(
        "progress",
        should_continue,
        {
            "improve": "optimizer",  # Loop back for another round
            "done": END,             # Quality met or max iterations reached
        },
    )

    return graph.compile()


# ================================================================
# Step 7: Run the Evaluator-Optimizer Loop
# ================================================================

def run_evaluator_optimizer(topic: str, max_iterations: int = 5):
    """Run the evaluator-optimizer graph on a topic.

    Args:
        topic: What the optimizer should write about
        max_iterations: Maximum number of optimize-evaluate cycles

    Returns:
        Final state with draft, score, and history
    """
    print(f"\n{'='*60}")
    print(f"Evaluator-Optimizer Loop (LangGraph)")
    print(f"Topic: {topic}")
    print(f"Quality threshold: 8/10 | Max iterations: {max_iterations}")
    print(f"{'='*60}")

    graph = build_eval_opt_graph()

    # Initial state
    initial_state: EvalOptState = {
        "topic": topic,
        "current_draft": "",
        "eval_score": 0,
        "eval_feedback": "",
        "scores_history": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "strategy_note": "",
    }

    # Run the graph — LangGraph handles the loop via conditional edges
    # We stream to see each step as it happens
    final_state = None
    current_iteration = 0

    for step in graph.stream(initial_state, {"recursion_limit": max_iterations * 3 + 5}):
        # Each step is a dict with one key (the node name) and value (state update)
        for node_name, node_output in step.items():
            if node_name == "optimizer":
                current_iteration = node_output.get("iteration", current_iteration)
                draft = node_output.get("current_draft", "")
                print(f"\n{'- '*30}")
                print(f"  ITERATION {current_iteration}: Optimizer Output")
                print(f"{'- '*30}")
                # Show first few lines of the draft
                lines = [l.strip() for l in draft.split("\n") if l.strip()]
                for line in lines[:4]:
                    print(f"    {line[:120]}")
                if len(lines) > 4:
                    print(f"    ... ({len(lines)} lines total)")

            elif node_name == "evaluator":
                score = node_output.get("eval_score", 0)
                feedback = node_output.get("eval_feedback", "")
                scores = node_output.get("scores_history", [])
                print(f"\n  [EVALUATOR] Score: {score}/10")
                for fb_line in feedback.split("\n"):
                    if fb_line.strip():
                        print(f"    {fb_line.strip()[:150]}")
                if len(scores) > 1:
                    print(f"    Score history: {scores}")

            elif node_name == "progress":
                strategy = node_output.get("strategy_note", "")
                if strategy:
                    print(f"  [STRATEGY] {strategy[:120]}")

        final_state = step

    # Extract final state from the last step
    # The graph.stream returns partial updates, so we need to invoke for full state
    result = graph.invoke(initial_state, {"recursion_limit": max_iterations * 3 + 5})
    return result


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("Example 8: Evaluator-Optimizer with Plateau Detection (LangGraph)")
    print("=" * 60)

    topic = "The impact of renewable energy on global economics"

    result = run_evaluator_optimizer(topic, max_iterations=5)

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"  Topic:      {result['topic']}")
    print(f"  Score:      {result['eval_score']}/10")
    print(f"  Iterations: {result['iteration']}")
    print(f"  Scores:     {result['scores_history']}")

    if result.get("strategy_note"):
        print(f"  Strategy:   Plateau escape was triggered")

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
    print(result["current_draft"])

    print(f"\n{'='*60}")
    print("Key Takeaways:")
    print(f"{'='*60}")
    print("  1. Optimizer and evaluator use DIFFERENT prompts and temperatures")
    print("     -> Optimizer (temp=0.7): creative, generates diverse text")
    print("     -> Evaluator (temp=0):  consistent, strict scoring")
    print("  2. Plateau detection prevents infinite loops with no progress")
    print("  3. Strategy injection (strategy_note) escapes local optima")
    print("  4. LangGraph conditional edges handle the loop/exit decision")
    print(f"\nNext: See example_09 for the same pattern in Google ADK.")
