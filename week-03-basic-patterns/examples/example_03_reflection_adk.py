import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3: Reflection Pattern in Google ADK
=============================================
The same reflection pattern from Example 2, but implemented using
Google ADK's agent architecture.

ADK Approach:
  - Use TWO agents: a Writer agent and a Critic agent
  - A coordinator function runs the reflection loop
  - Each agent has its own specialized instruction prompt
  - ADK's Runner handles the LLM calls

This shows how ADK's declarative style handles the same pattern
differently from LangGraph's explicit graph.

Comparison:
  LangGraph: Explicit graph with nodes + conditional edges
  ADK:       Separate agents coordinated by application logic

Run: python week-03-basic-patterns/examples/example_03_reflection_adk.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Step 1: Create the Writer Agent
# ==============================================================
# The Writer agent generates and refines text. Its instruction
# tells it to incorporate feedback when provided.

writer_agent = LlmAgent(
    name="writer",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    instruction="""You are an expert writer who creates clear, detailed content.

When given JUST a topic:
- Write a well-structured paragraph (3-5 sentences)
- Include specific facts, examples, or data points
- Be clear and engaging

When given a topic WITH critique feedback:
- Rewrite the paragraph addressing ALL issues in the critique
- Keep the strengths that were mentioned
- Fix every weakness listed
- Output ONLY the revised paragraph, nothing else""",
    description="Expert writer that generates and refines text based on feedback.",
)


# ==============================================================
# Step 2: Create the Critic Agent
# ==============================================================
# The Critic agent evaluates text quality. It uses a DIFFERENT
# prompt to ensure independent evaluation.

critic_agent = LlmAgent(
    name="critic",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    instruction="""You are an extremely strict editor. Your job is to find EVERY flaw.
A score of 10 means absolutely perfect -- almost nothing deserves a 10.

Respond in EXACTLY this format (plain text, not JSON):
SCORE: <number 1-10>
STRENGTHS: <comma-separated list>
WEAKNESSES: <comma-separated list>
SUGGESTIONS: <comma-separated list>

STRICT scoring guide (follow this exactly):
- 1-3: Major issues (factual errors, incoherent, off-topic)
- 4-5: Mediocre (vague claims, lacks specific numbers/data, poor structure)
- 6-7: Decent (has some detail but missing citations, counterpoints, or nuance)
- 8: Good (specific data, balanced view, but still has minor issues)
- 9: Excellent (publication-ready with citations, counterpoints, and strong structure)
- 10: Perfect (almost impossible -- flawless in every dimension)

IMPORTANT: Most first drafts should score 5-7. Do NOT give 8+ unless
the text has specific statistics, named sources, counterpoints/limitations,
and excellent structure. Be harsh but constructive.""",
    description="Strict editor that evaluates and scores text quality.",
)


# ==============================================================
# Step 3: Helper to Run an ADK Agent
# ==============================================================

async def run_agent(agent: LlmAgent, message: str) -> str:
    """Run an ADK agent with a message and return the response text.

    Each call creates a fresh session so agents don't carry over
    context from previous calls (we manage context ourselves).
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="reflection_demo",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="reflection_demo",
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
            result_text = event.content.parts[0].text

    return result_text


# ==============================================================
# Step 4: Parse Critique Response
# ==============================================================

def parse_critique(critique_text: str) -> dict:
    """Extract score and feedback from the critic's response.

    Handles the structured format:
      SCORE: 7
      STRENGTHS: ...
      WEAKNESSES: ...
      SUGGESTIONS: ...
    """
    score = 5  # default
    weaknesses = ""
    suggestions = ""

    for line in critique_text.strip().split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("score:"):
            try:
                # Extract number from "SCORE: 7" or "SCORE: 7/10"
                score_str = line.split(":", 1)[1].strip().split("/")[0].strip()
                score = int(score_str)
            except (ValueError, IndexError):
                pass
        elif line_lower.startswith("weaknesses:"):
            weaknesses = line.split(":", 1)[1].strip()
        elif line_lower.startswith("suggestions:"):
            suggestions = line.split(":", 1)[1].strip()

    return {
        "score": score,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "full_critique": critique_text,
    }


# ==============================================================
# Step 5: The Reflection Loop
# ==============================================================
# In ADK, we coordinate the loop in our application code rather
# than in a graph structure. The logic is the same as LangGraph's
# conditional edges, but expressed as a Python while loop.

QUALITY_THRESHOLD = 8  # Strict critic + threshold 8 = usually 2-3 iterations before passing

async def reflection_loop(topic: str, max_iterations: int = 3) -> dict:
    """Run the reflection loop using ADK agents.

    Flow: writer -> critic -> [writer -> critic]* -> done

    Args:
        topic: What to write about
        max_iterations: Maximum refinement cycles

    Returns:
        Dict with final draft, score, and history
    """
    print(f"\n{'='*60}")
    print(f"ADK Reflection: '{topic}'")
    print(f"{'='*60}")

    history = []

    # Step 1: Generate initial draft
    print(f"\n{'- '*30}")
    print(f"  ITERATION 1: Initial Draft")
    print(f"{'- '*30}")
    draft = await run_agent(writer_agent, f"Write about: {topic}")

    print(f"\n  [GENERATE] Draft (v1):")
    for line in draft.split("\n"):
        if line.strip():
            print(f"    {line.strip()[:120]}")
    history.append({"type": "draft", "content": draft})

    # Step 2: Critique-refine loop
    for iteration in range(1, max_iterations + 1):
        # Get critique
        critique_text = await run_agent(
            critic_agent,
            f"Topic: {topic}\n\nEvaluate this text:\n{draft}"
        )
        critique = parse_critique(critique_text)

        print(f"\n  [CRITIQUE] Score: {critique['score']}/10  (threshold: {QUALITY_THRESHOLD}/10)")
        if critique["weaknesses"]:
            print(f"    Weaknesses: {critique['weaknesses'][:150]}")
        if critique["suggestions"]:
            print(f"    Suggestions: {critique['suggestions'][:150]}")
        history.append({"type": "critique", "score": critique["score"]})

        # Quality gate -- same logic as LangGraph's should_continue
        if critique["score"] >= QUALITY_THRESHOLD:
            print(f"\n  --> Score {critique['score']} >= {QUALITY_THRESHOLD}: ACCEPTED! Done.")
            break

        if iteration >= max_iterations:
            print(f"\n  --> Max iterations ({max_iterations}) reached. Accepting current draft.")
            break

        # Score too low -- refine
        print(f"\n  --> Score {critique['score']} < {QUALITY_THRESHOLD}: NEEDS IMPROVEMENT. Refining...")

        print(f"\n{'- '*30}")
        print(f"  ITERATION {iteration + 1}: Refining Draft")
        print(f"{'- '*30}")

        draft = await run_agent(
            writer_agent,
            f"Topic: {topic}\n\n"
            f"Previous draft:\n{draft}\n\n"
            f"Critique:\n{critique['full_critique']}\n\n"
            f"Rewrite addressing all weaknesses and suggestions:"
        )

        print(f"\n  [GENERATE] Draft (v{iteration + 1}):")
        for line in draft.split("\n"):
            if line.strip():
                print(f"    {line.strip()[:120]}")
        history.append({"type": "draft", "content": draft})

    return {
        "final_draft": draft,
        "final_score": critique["score"],
        "iterations": iteration,
        "history": history,
    }


# ==============================================================
# Run the demo
# ==============================================================

if __name__ == "__main__":
    print("Example 3: Reflection Pattern in ADK")
    print("=" * 60)

    result = asyncio.run(
        reflection_loop("The impact of artificial intelligence on healthcare", max_iterations=3)
    )

    # Show score progression
    scores = [e["score"] for e in result["history"] if e["type"] == "critique"]
    if len(scores) > 1:
        print(f"\nScore Progression: {' -> '.join(str(s) for s in scores)}")
        improvement = scores[-1] - scores[0]
        print(f"Total Improvement: {'+' if improvement >= 0 else ''}{improvement} points over {len(scores)} iterations")
    elif scores:
        print(f"\nScore: {scores[0]}/10 (accepted on first iteration)")

    print(f"\n{'='*60}")
    print(f"Final Output (Score: {result['final_score']}/10, {result['iterations']} iterations):")
    print(f"{'='*60}")
    print(result["final_draft"])

    # Show comparison with LangGraph approach
    print(f"\n{'='*60}")
    print("LangGraph vs ADK -- Reflection Pattern Comparison:")
    print(f"{'='*60}")
    print("  LangGraph: Graph nodes + conditional edges control the loop")
    print("             -> Visual, declarative, easy to trace")
    print("  ADK:       Separate agents + application loop control")
    print("             -> Simpler code, more flexible, agent-per-role")
    print("\n  Both produce the same result -- choose based on your needs.")
