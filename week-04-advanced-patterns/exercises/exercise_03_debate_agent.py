import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 3: Multi-Agent Debate System with Judge
==================================================
Difficulty: Advanced | Time: 2 hours

Task:
Build a debate system with three agents using LangGraph:
  1. Pro Debater: argues IN FAVOR of a given position
  2. Con Debater: argues AGAINST the position
  3. Judge/Moderator: organizes debate, scores arguments, produces synthesis

The debate runs for 3 rounds:
  - Round 1: Opening statements (each debater presents initial arguments)
  - Round 2: Rebuttals (each debater responds to the other's arguments)
  - Round 3: Closing statements (final arguments)

After 3 rounds, the judge:
  - Scores each debater on: argument strength, evidence quality,
    logical consistency (each 1-10)
  - Produces a neutral, balanced synthesis incorporating both sides

Instructions:
1. Complete the DebateState TypedDict with all required fields
2. Implement pro_debater_node — generates arguments for the position
3. Implement con_debater_node — generates arguments against the position
4. Implement judge_node — scores both sides and produces a synthesis
5. Implement should_continue — route to next round or to judge
6. Wire the LangGraph StateGraph with correct edges

Hints:
- Look at example_05_multi_agent_langgraph.py for multi-agent patterns
- Look at example_08_evaluator_optimizer_langgraph.py for graph routing
- Each debater should REFERENCE the other's arguments in rounds 2-3
  (pass them in the prompt so they actually rebut, not just repeat)
- The judge prompt should ask for JSON output with scores + synthesis
- Use json.loads with a try/except fallback for parsing judge output
- Track round_number to know which round instructions to give debaters

Run: python week-04-advanced-patterns/exercises/exercise_03_debate_agent.py
"""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ==============================================================
# LLM Factory (provided)
# ==============================================================

def get_llm(temperature=0.7):
    """Get the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


# ==============================================================
# TODO 1: Complete the DebateState TypedDict
# ==============================================================
# Define a TypedDict with these fields:
#   - topic: str              (the debate topic/proposition)
#   - pro_position: str       (brief label for the pro side)
#   - con_position: str       (brief label for the con side)
#   - pro_arguments: list     (list of pro arguments, one per round)
#   - con_arguments: list     (list of con arguments, one per round)
#   - round_number: int       (current round: 1, 2, or 3)
#   - max_rounds: int         (always 3)
#   - judge_scores: dict      (scoring dict from the judge)
#   - synthesis: str           (final neutral synthesis from the judge)

# class DebateState(TypedDict):
#     topic: str
#     pro_position: str
#     con_position: str
#     pro_arguments: list
#     con_arguments: list
#     round_number: int
#     max_rounds: int
#     judge_scores: dict
#     synthesis: str


# ==============================================================
# TODO 2: Implement the Pro Debater Node
# ==============================================================
# Create pro_debater_node(state) that:
#   - Gets the current round number from state
#   - Builds a system message: "You are debating IN FAVOR of: {topic}"
#   - Round 1 (opening): "Present 3 strong arguments WITH evidence
#     supporting your position."
#   - Round 2 (rebuttal): "Your opponent argued: {con_arguments[-1]}.
#     Counter their arguments specifically, then reinforce your position."
#   - Round 3 (closing): "Your opponent argued: {con_arguments[-1]}.
#     Give a compelling closing statement addressing the strongest
#     counter-arguments."
#   - Calls the LLM and appends the response to pro_arguments
#   - Returns dict with updated pro_arguments
#
# Key: In rounds 2-3, you MUST include the opponent's arguments in
# the prompt so the debater actually engages with them.

# def pro_debater_node(state: DebateState) -> dict:
#     llm = get_llm(temperature=0.7)
#     round_num = state["round_number"]
#     topic = state["topic"]
#     ...


# ==============================================================
# TODO 3: Implement the Con Debater Node
# ==============================================================
# Create con_debater_node(state) that:
#   - Same structure as pro_debater_node, but argues AGAINST
#   - System message: "You are debating AGAINST: {topic}"
#   - Round 1: Present 3 strong counter-arguments with evidence
#   - Round 2: Rebut the PRO debater's arguments (pro_arguments[-1])
#   - Round 3: Closing statement referencing pro_arguments[-1]
#   - Appends to con_arguments list

# def con_debater_node(state: DebateState) -> dict:
#     llm = get_llm(temperature=0.7)
#     round_num = state["round_number"]
#     topic = state["topic"]
#     ...


# ==============================================================
# TODO 4: Implement the Judge Node
# ==============================================================
# Create judge_node(state) that:
#   - Only runs AFTER all 3 rounds are complete
#   - Sends ALL arguments from BOTH sides to the LLM
#   - Asks the judge to score each debater on:
#       argument_strength (1-10)
#       evidence_quality (1-10)
#       logical_consistency (1-10)
#   - Asks for a neutral synthesis (3-5 sentences)
#   - Requests JSON output in this format:
#     {
#       "pro_scores": {"argument_strength": N, "evidence_quality": N, "logical_consistency": N},
#       "con_scores": {"argument_strength": N, "evidence_quality": N, "logical_consistency": N},
#       "pro_total": N,
#       "con_total": N,
#       "winner": "pro" or "con" or "tie",
#       "justification": "...",
#       "synthesis": "..."
#     }
#   - Parse the JSON (with fallback if parsing fails)
#   - Return dict with judge_scores and synthesis

# def judge_node(state: DebateState) -> dict:
#     llm = get_llm(temperature=0.3)  # Lower temperature for consistent judging
#     ...


# ==============================================================
# TODO 5: Implement the Routing Function
# ==============================================================
# Create advance_round(state) that:
#   - Increments round_number by 1
#   - Returns the updated round_number

# def advance_round(state: DebateState) -> dict:
#     ...

# Create should_continue(state) that returns:
#   - "judge" if round_number > max_rounds (all rounds done)
#   - "pro_debater" otherwise (continue to next round)

# def should_continue(state: DebateState) -> str:
#     ...


# ==============================================================
# TODO 6: Wire the Graph
# ==============================================================
# Build the StateGraph:
#   1. Add nodes: "pro_debater", "con_debater", "advance_round", "judge"
#   2. Set entry point to "pro_debater"
#   3. Edge: pro_debater -> con_debater (always)
#   4. Edge: con_debater -> advance_round (always)
#   5. Conditional edge from advance_round:
#      - "pro_debater" -> loop back for next round
#      - "judge" -> go to judge node
#   6. Edge: judge -> END
#   7. Compile the graph
#
# Graph flow:
#   START -> pro_debater -> con_debater -> advance_round
#                                             |
#                             +---------------+
#                             | more rounds   | done
#                             v               v
#                         pro_debater      judge -> END

# from langgraph.graph import StateGraph, END
# graph = StateGraph(DebateState)
# ... add nodes, edges ...
# app = graph.compile()


# ==============================================================
# Run Function (provided)
# ==============================================================

def run_debate(topic: str, pro_position: str = "In Favor", con_position: str = "Against"):
    """Run a 3-round debate on the given topic."""
    print(f"\n{'='*70}")
    print(f"DEBATE: {topic}")
    print(f"{'='*70}")
    print(f"  PRO ({pro_position}) vs CON ({con_position})")
    print(f"  Format: 3 rounds (Opening -> Rebuttal -> Closing)")
    print(f"{'='*70}")

    initial_state = {
        "topic": topic,
        "pro_position": pro_position,
        "con_position": con_position,
        "pro_arguments": [],
        "con_arguments": [],
        "round_number": 1,
        "max_rounds": 3,
        "judge_scores": {},
        "synthesis": "",
    }

    # Uncomment after implementing:
    # result = app.invoke(initial_state)
    #
    # # Print results
    # print(f"\n{'='*70}")
    # print("JUDGE'S DECISION")
    # print(f"{'='*70}")
    #
    # if result.get("judge_scores"):
    #     scores = result["judge_scores"]
    #     print(f"\n  Pro Scores: {scores.get('pro_scores', 'N/A')}")
    #     print(f"  Con Scores: {scores.get('con_scores', 'N/A')}")
    #     print(f"  Pro Total:  {scores.get('pro_total', 'N/A')}/30")
    #     print(f"  Con Total:  {scores.get('con_total', 'N/A')}/30")
    #     print(f"  Winner:     {scores.get('winner', 'N/A').upper()}")
    #     print(f"\n  Justification: {scores.get('justification', 'N/A')}")
    #
    # if result.get("synthesis"):
    #     print(f"\n  Synthesis:\n  {result['synthesis']}")
    #
    # return result

    print("\n(Uncomment the test code above after implementing all TODOs!)")
    return initial_state


# ==============================================================
# Test your implementation
# ==============================================================

if __name__ == "__main__":
    print("Exercise 3: Multi-Agent Debate System with Judge")
    print("=" * 70)

    # Test debate
    result = run_debate(
        topic="AI will create more jobs than it destroys in the next decade",
        pro_position="AI Creates Jobs",
        con_position="AI Destroys Jobs",
    )

    print(f"\n{'='*70}")
    print("Success Criteria:")
    print("  - Debaters reference and rebut each other's specific arguments")
    print("  - Judge scoring uses rubric (argument_strength, evidence_quality,")
    print("    logical_consistency) with scores 1-10 for each")
    print("  - Neutral synthesis incorporates points from both sides")
    print("  - Total cost stays under $0.50")
    print(f"{'='*70}")
