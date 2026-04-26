import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 2: Reflection Pattern in LangGraph — LLM Self-Critique
================================================================
Now we implement the reflection pattern with a real LLM in LangGraph.
The agent generates a response, a CRITIC node evaluates it, and if
the quality is insufficient, it loops back to refine.

Graph:
  generate -> critique -> [refine -> critique]* -> done -> END

This is the same pattern from Example 1, but now the LLM does the
generating, critiquing, and refining — making the output genuinely
improve with each iteration.

Key Concepts:
  - Separate LLM calls for generation vs critique (different prompts)
  - Quality scoring with a threshold to control the loop
  - Max iterations as a safety valve
  - State tracks the full history: drafts, critiques, scores

Run: python week-03-basic-patterns/examples/example_02_reflection_langgraph.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional


# ==============================================================
# Step 1: Define the State
# ==============================================================
# We track drafts, critiques, and iteration count in the state.
# This gives full visibility into the reflection process.

class ReflectionState(TypedDict):
    topic: str                    # The subject to write about
    current_draft: str            # Latest version of the output
    critique: str                 # Latest critique feedback
    quality_score: int            # Score from 1-10
    iteration: int                # Current iteration count
    max_iterations: int           # Safety limit
    history: list                 # Log of all drafts and critiques


# ==============================================================
# Step 2: Set up the LLM
# ==============================================================

def get_llm():
    """Create LLM based on provider setting."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.7,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
        )


llm = get_llm()


# ==============================================================
# Step 3: Define the Nodes
# ==============================================================

# -- Generator Node --------------------------------------------
# On first call: generates initial draft from the topic.
# On subsequent calls: refines the draft using critique feedback.

def generate_node(state: ReflectionState) -> dict:
    """Generate or refine a draft based on the current state."""
    iteration = state.get("iteration", 0)

    if iteration == 0:
        # First call -- generate from scratch
        messages = [
            SystemMessage(content=(
                "You are an expert writer. Write a clear, detailed paragraph "
                "about the given topic. Include specific facts, examples, and "
                "data points. The output should be 3-5 sentences."
            )),
            HumanMessage(content=f"Write about: {state['topic']}"),
        ]
        print(f"\n{'- '*30}")
        print(f"  ITERATION {iteration + 1}: Initial Draft")
        print(f"{'- '*30}")
    else:
        # Subsequent calls -- refine based on critique
        messages = [
            SystemMessage(content=(
                "You are an expert writer. You will receive your previous draft "
                "and a critique. Rewrite the draft to address ALL issues raised "
                "in the critique. Keep the good parts, fix the problems. "
                "Output ONLY the revised paragraph, nothing else."
            )),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n\n"
                f"Previous draft:\n{state['current_draft']}\n\n"
                f"Critique:\n{state['critique']}\n\n"
                f"Rewrite the draft addressing all issues above:"
            )),
        ]
        print(f"\n{'- '*30}")
        print(f"  ITERATION {iteration + 1}: Refining Draft")
        print(f"{'- '*30}")

    response = llm.invoke(messages)
    draft = response.content

    print(f"\n  [GENERATE] Draft (v{iteration + 1}):")
    # Print the full draft wrapped for readability
    for line in draft.split("\n"):
        if line.strip():
            print(f"    {line.strip()[:120]}")

    return {
        "current_draft": draft,
        "iteration": iteration + 1,
        "history": state.get("history", []) + [{"type": "draft", "content": draft}],
    }


# -- Critic Node -----------------------------------------------
# Evaluates the draft against specific criteria and returns a
# score + detailed feedback. Uses a DIFFERENT system prompt than
# the generator — this separation is key to effective reflection.

def critique_node(state: ReflectionState) -> dict:
    """Critique the current draft and assign a quality score."""
    messages = [
        SystemMessage(content=(
            "You are a strict but fair editor. Evaluate the given text and "
            "respond in this EXACT JSON format (no markdown, no code blocks):\n"
            '{"score": <1-10>, "strengths": ["..."], "weaknesses": ["..."], '
            '"suggestions": ["..."]}\n\n'
            "Scoring guide:\n"
            "- 1-3: Major issues (factual errors, incoherent, off-topic)\n"
            "- 4-6: Needs improvement (vague, lacks detail, poor structure)\n"
            "- 7-8: Good (clear, detailed, well-structured, minor issues)\n"
            "- 9-10: Excellent (specific facts, compelling, publication-ready)"
        )),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"Text to evaluate:\n{state['current_draft']}"
        )),
    ]

    response = llm.invoke(messages)

    # Parse the critique — handle both clean JSON and markdown-wrapped JSON
    critique_text = response.content.strip()
    # Strip markdown code blocks if present
    if critique_text.startswith("```"):
        critique_text = critique_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        critique_data = json.loads(critique_text)
        score = int(critique_data.get("score", 5))
        # Build readable critique string from structured data
        critique_summary = []
        if critique_data.get("weaknesses"):
            critique_summary.append("Weaknesses: " + "; ".join(critique_data["weaknesses"]))
        if critique_data.get("suggestions"):
            critique_summary.append("Suggestions: " + "; ".join(critique_data["suggestions"]))
        critique_str = "\n".join(critique_summary) if critique_summary else "No specific issues."
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, use the raw text and a middle score
        score = 5
        critique_str = critique_text

    print(f"\n  [CRITIQUE] Score: {score}/10  (threshold: {QUALITY_THRESHOLD}/10)")
    # Show weaknesses/suggestions clearly
    for line in critique_str.split("\n"):
        if line.strip():
            print(f"    {line.strip()[:120]}")

    # Show decision
    if score >= QUALITY_THRESHOLD:
        print(f"\n  --> Score {score} >= {QUALITY_THRESHOLD}: ACCEPTED! Moving to done.")
    elif state.get("iteration", 0) >= state.get("max_iterations", 3):
        print(f"\n  --> Max iterations reached. Accepting current draft.")
    else:
        print(f"\n  --> Score {score} < {QUALITY_THRESHOLD}: NEEDS IMPROVEMENT. Refining...")

    return {
        "critique": critique_str,
        "quality_score": score,
        "history": state.get("history", []) + [
            {"type": "critique", "score": score, "feedback": critique_str}
        ],
    }


# -- Done Node -------------------------------------------------
# Simply passes through — the final draft is already in state.

def done_node(state: ReflectionState) -> dict:
    """Finalize the output."""
    print(f"\n{'='*60}")
    print(f"  DONE! Final score: {state['quality_score']}/10 after {state['iteration']} iteration(s)")
    print(f"{'='*60}")
    return {}


# ==============================================================
# Step 4: Routing Function (Quality Gate)
# ==============================================================
# This is the KEY decision point. After each critique, we decide:
#   - Score >= 7: Good enough -> go to "done"
#   - Score < 7 AND iterations left: Needs work -> go to "generate" (refine)
#   - Max iterations hit: Stop anyway -> go to "done"

QUALITY_THRESHOLD = 9  # Minimum score to accept (set high to force multiple iterations)

def should_continue(state: ReflectionState) -> str:
    """Decide whether to refine further or accept the current draft."""
    score = state.get("quality_score", 0)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if score >= QUALITY_THRESHOLD:
        return "done"       # Quality gate passed!
    elif iteration >= max_iter:
        print(f"   [WARN] Max iterations ({max_iter}) reached. Accepting current draft.")
        return "done"       # Safety valve — stop the loop
    else:
        return "refine"     # Loop back to generator with critique feedback


# ==============================================================
# Step 5: Build the Graph
# ==============================================================
# Flow: generate -> critique -> [generate -> critique]* -> done -> END
#
# This is the same structure as the agent->tools loop from Week 2,
# but instead of calling tools, we call a CRITIC.

def build_reflection_graph():
    """Build the reflection graph."""
    graph = StateGraph(ReflectionState)

    # Add nodes
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("done", done_node)

    # Entry point: always start by generating
    graph.set_entry_point("generate")

    # After generating, always critique
    graph.add_edge("generate", "critique")

    # After critique, decide: refine or accept
    graph.add_conditional_edges(
        "critique",
        should_continue,
        {
            "refine": "generate",  # Loop back to refine
            "done": "done",        # Accept the output
        },
    )

    # After done, end
    graph.add_edge("done", END)

    return graph.compile()


# ==============================================================
# Step 6: Run the Reflection Agent
# ==============================================================

def run_reflection(topic: str, max_iterations: int = 3) -> dict:
    """Run the reflection agent on a topic.

    Args:
        topic: What to write about
        max_iterations: Maximum refinement loops (default 3)

    Returns:
        The final state with draft, score, and history
    """
    app = build_reflection_graph()

    result = app.invoke({
        "topic": topic,
        "current_draft": "",
        "critique": "",
        "quality_score": 0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "history": [],
    })

    return result


if __name__ == "__main__":
    print("Example 2: Reflection Pattern in LangGraph")
    print("=" * 60)

    # Run reflection on a topic
    result = run_reflection("The impact of artificial intelligence on healthcare", max_iterations=3)

    # Show score progression summary
    scores = [e["score"] for e in result["history"] if e["type"] == "critique"]
    if len(scores) > 1:
        print(f"\nScore Progression: {' -> '.join(str(s) for s in scores)}")
        improvement = scores[-1] - scores[0]
        print(f"Total Improvement: +{improvement} points over {len(scores)} iterations")
    elif scores:
        print(f"\nScore: {scores[0]}/10 (accepted on first iteration)")

    print(f"\n{'='*60}")
    print(f"Final Output (Score: {result['quality_score']}/10):")
    print(f"{'='*60}")
    print(result["current_draft"])
