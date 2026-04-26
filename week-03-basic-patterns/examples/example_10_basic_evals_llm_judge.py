import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 10: Basic Evals — LLM-as-Judge
========================================
How do you know if your agent's output is GOOD? You can't manually
review thousands of outputs. The solution: use another LLM as a judge.

LLM-as-Judge Pattern:
  Agent generates output -> Judge LLM scores it against criteria

This example covers:
  1. Simple scoring (1-10 with criteria)
  2. Pairwise comparison (which output is better?)
  3. Multi-criteria evaluation (score on multiple dimensions)
  4. Building an eval pipeline to test your agent systematically

Run: python week-03-basic-patterns/examples/example_10_basic_evals_llm_judge.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================
# Setup
# ==============================================================

def get_llm():
    """Create LLM for judging."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


llm = get_llm()


# ==============================================================
# EVAL 1: Simple Scoring (1-10)
# ==============================================================
# The judge scores a single output against defined criteria.
# This is the simplest and most common eval approach.

def simple_score(text: str, criteria: str) -> dict:
    """Score text on a 1-10 scale using LLM-as-judge.

    Args:
        text: The text to evaluate
        criteria: What to evaluate it against

    Returns:
        Dict with score (int) and reasoning (str)
    """
    messages = [
        SystemMessage(content=(
            "You are an evaluation judge. Score the given text on a scale "
            "of 1-10 based on the specified criteria.\n\n"
            "Respond in this EXACT format (plain text):\n"
            "SCORE: <number>\n"
            "REASONING: <one sentence explaining the score>"
        )),
        HumanMessage(content=(
            f"Criteria: {criteria}\n\n"
            f"Text to evaluate:\n{text}"
        )),
    ]

    response = llm.invoke(messages)
    result = response.content.strip()

    # Parse response
    score = 5
    reasoning = result
    for line in result.split("\n"):
        if line.strip().upper().startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip().split("/")[0].strip())
            except ValueError:
                pass
        elif line.strip().upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {"score": score, "reasoning": reasoning}


def demo_simple_scoring():
    """Demonstrate simple 1-10 scoring."""

    print("=" * 60)
    print("EVAL 1: Simple Scoring (1-10)")
    print("=" * 60)

    # Two outputs to evaluate — one good, one bad
    good_output = (
        "Artificial intelligence in healthcare has reduced diagnostic errors by 30% "
        "in radiology departments using AI-assisted imaging. The global AI healthcare "
        "market reached $15.1 billion in 2025, with applications spanning drug discovery, "
        "patient monitoring, and administrative automation."
    )

    bad_output = (
        "AI is really cool and helpful in healthcare. It does lots of important stuff "
        "that helps many people. Everyone agrees it's the future."
    )

    criteria = "Factual accuracy, specific details (numbers/examples), and clarity"

    print(f"\n  Criteria: {criteria}")

    print(f"\n  Output A (detailed):")
    print(f"    \"{good_output[:100]}...\"")
    result_a = simple_score(good_output, criteria)
    print(f"    Score: {result_a['score']}/10")
    print(f"    Reasoning: {result_a['reasoning'][:150]}")

    print(f"\n  Output B (vague):")
    print(f"    \"{bad_output[:100]}...\"")
    result_b = simple_score(bad_output, criteria)
    print(f"    Score: {result_b['score']}/10")
    print(f"    Reasoning: {result_b['reasoning'][:150]}")


# ==============================================================
# EVAL 2: Pairwise Comparison
# ==============================================================
# Instead of absolute scoring, compare two outputs side by side.
# More reliable than absolute scoring (humans do this naturally).

def pairwise_compare(text_a: str, text_b: str, criteria: str) -> dict:
    """Compare two texts and pick the better one.

    Args:
        text_a: First text
        text_b: Second text
        criteria: What to compare them on

    Returns:
        Dict with winner ("A" or "B") and reasoning
    """
    messages = [
        SystemMessage(content=(
            "You are a fair judge. Compare two texts and decide which is better "
            "based on the given criteria.\n\n"
            "Respond in this EXACT format:\n"
            "WINNER: A or B\n"
            "REASONING: <one sentence explaining why>"
        )),
        HumanMessage(content=(
            f"Criteria: {criteria}\n\n"
            f"Text A:\n{text_a}\n\n"
            f"Text B:\n{text_b}"
        )),
    ]

    response = llm.invoke(messages)
    result = response.content.strip()

    winner = "A"
    reasoning = result
    for line in result.split("\n"):
        if line.strip().upper().startswith("WINNER:"):
            w = line.split(":", 1)[1].strip().upper()
            winner = "A" if "A" in w else "B"
        elif line.strip().upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {"winner": winner, "reasoning": reasoning}


def demo_pairwise():
    """Demonstrate pairwise comparison."""

    print(f"\n{'='*60}")
    print("EVAL 2: Pairwise Comparison")
    print("=" * 60)

    text_a = (
        "The reflection pattern improves agent output by having the LLM critique "
        "and revise its own work iteratively, similar to how a writer drafts and edits."
    )
    text_b = (
        "Reflection is an agentic pattern where a generator produces output, a critic "
        "evaluates it against specific criteria (accuracy, completeness, clarity), and "
        "a refiner addresses identified weaknesses. Typical improvement: 2-3 points on "
        "a 10-point scale within 3 iterations."
    )

    criteria = "Technical depth and practical usefulness"

    print(f"\n  Text A: \"{text_a[:80]}...\"")
    print(f"  Text B: \"{text_b[:80]}...\"")

    result = pairwise_compare(text_a, text_b, criteria)
    print(f"\n  Winner: Text {result['winner']}")
    print(f"  Reasoning: {result['reasoning'][:200]}")


# ==============================================================
# EVAL 3: Multi-Criteria Evaluation
# ==============================================================
# Score on multiple dimensions separately — gives more actionable
# feedback than a single overall score.

def multi_criteria_eval(text: str, criteria: list) -> dict:
    """Evaluate text on multiple criteria independently.

    Args:
        text: Text to evaluate
        criteria: List of criterion names

    Returns:
        Dict mapping each criterion to a score and note
    """
    criteria_str = "\n".join(f"- {c}" for c in criteria)

    messages = [
        SystemMessage(content=(
            "You are an evaluation judge. Score the text on EACH criterion "
            "separately (1-10). Respond in this EXACT format, one line per criterion:\n"
            "criterion_name: score - brief note\n\n"
            f"Criteria to evaluate:\n{criteria_str}"
        )),
        HumanMessage(content=f"Text to evaluate:\n{text}"),
    ]

    response = llm.invoke(messages)
    result = response.content.strip()

    # Parse multi-criteria response
    scores = {}
    for line in result.split("\n"):
        line = line.strip()
        if ":" in line and any(c.lower() in line.lower() for c in criteria):
            parts = line.split(":", 1)
            name = parts[0].strip().lower()
            try:
                score_part = parts[1].strip()
                score = int(score_part.split("-")[0].split("/")[0].strip())
                note = score_part.split("-", 1)[1].strip() if "-" in score_part else ""
                # Match to the closest criterion
                for c in criteria:
                    if c.lower() in name or name in c.lower():
                        scores[c] = {"score": score, "note": note}
                        break
            except (ValueError, IndexError):
                pass

    return scores


def demo_multi_criteria():
    """Demonstrate multi-criteria evaluation."""

    print(f"\n{'='*60}")
    print("EVAL 3: Multi-Criteria Evaluation")
    print("=" * 60)

    text = (
        "Climate change is caused by greenhouse gas emissions from human activities. "
        "Global temperatures have risen 1.2°C above pre-industrial levels. The Paris "
        "Agreement aims to limit warming to 1.5°C. Renewable energy now provides 35% "
        "of global electricity, up from 22% in 2015."
    )

    criteria = ["Accuracy", "Completeness", "Clarity", "Specificity"]

    print(f"\n  Text: \"{text[:100]}...\"")
    print(f"\n  Scores:")

    scores = multi_criteria_eval(text, criteria)
    for criterion in criteria:
        if criterion in scores:
            s = scores[criterion]
            bar = "#" * s["score"] + "." * (10 - s["score"])
            print(f"    {criterion:<15} [{bar}] {s['score']}/10  {s['note'][:60]}")
        else:
            print(f"    {criterion:<15} [not scored]")

    print(f"\n  [TIP] Multi-criteria evals help you find SPECIFIC weaknesses.")
    print(f"     A text might score 9/10 on accuracy but 4/10 on completeness.")


# ==============================================================
# EVAL 4: Building an Eval Pipeline
# ==============================================================

def demo_eval_pipeline():
    """Show how to build a systematic eval pipeline.

    A REAL eval pipeline has two steps:
      Step 1: The AGENT generates an answer (LLM call #1)
      Step 2: The JUDGE scores that answer   (LLM call #2)

    This is end-to-end: you test the agent's actual output, not
    hardcoded strings. Run this after every change to your agent
    to catch regressions.
    """

    print(f"\n{'='*60}")
    print("EVAL 4: Building an Eval Pipeline")
    print("=" * 60)

    # ----- Test cases: input question + what a good answer should contain -----
    # The "expected" field tells the JUDGE what to look for, not what the
    # exact answer should be. This lets the agent answer freely while still
    # being evaluated against specific criteria.
    test_cases = [
        {
            "input": "What is machine learning?",
            "expected": "Should explain ML clearly with at least one concrete example",
        },
        {
            "input": "Explain quantum computing in simple terms",
            "expected": "Should explain qubits or superposition, mention at least one practical use case",
        },
        {
            "input": "What are the pros and cons of remote work?",
            "expected": "Should list at least 2 pros and 2 cons with specific reasons",
        },
    ]

    print(f"\n  Running eval pipeline on {len(test_cases)} test cases...")
    print(f"  Each test: LLM generates answer -> LLM judges quality\n")

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"  Test case {i}: \"{tc['input']}\"")

        # ----- Step 1: AGENT generates an answer (this is the LLM call being tested) -----
        agent_response = llm.invoke([
            SystemMessage(content="You are a helpful assistant. Give clear, concise answers."),
            HumanMessage(content=tc["input"]),
        ])
        agent_output = agent_response.content.strip()
        print(f"    Agent output: \"{agent_output[:120]}...\"")

        # ----- Step 2: JUDGE scores the answer (separate LLM call) -----
        score = simple_score(agent_output, tc["expected"])
        results.append(score)
        status = "PASS" if score["score"] >= 7 else "FAIL"
        print(f"    Judge score:  {score['score']}/10 [{status}]")
        print(f"    Reasoning:    {score['reasoning'][:120]}")
        print()

    # ----- Summary: aggregate metrics across all test cases -----
    avg_score = sum(r["score"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["score"] >= 7) / len(results)
    print(f"  Pipeline Results:")
    print(f"    Test cases:    {len(results)}")
    print(f"    Average score: {avg_score:.1f}/10")
    print(f"    Pass rate:     {pass_rate:.0%} (threshold: 7/10)")
    print(f"    LLM calls:     {len(results) * 2} ({len(results)} agent + {len(results)} judge)")

    print(f"\n  [TIP] Run this pipeline after every change to your agent.")
    print(f"     If the average score drops, your change made things worse.")


# ==============================================================
# Run all demos
# ==============================================================

if __name__ == "__main__":
    print("Example 10: Basic Evals — LLM-as-Judge")
    print("=" * 60)

    demo_simple_scoring()
    demo_pairwise()
    demo_multi_criteria()
    demo_eval_pipeline()

    print(f"\n{'='*60}")
    print("Key Takeaways:")
    print("  1. Simple scoring: quick quality check (most common)")
    print("  2. Pairwise: reliable comparisons between agent versions")
    print("  3. Multi-criteria: find specific weaknesses")
    print("  4. Eval pipeline: systematic testing with known test cases")
    print(f"{'='*60}")
