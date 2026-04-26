import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 3: Multi-Agent Debate System with Judge
==================================================
Complete solution for exercise_03_debate_agent.py

A LangGraph-based 3-agent debate system:
  - Pro Debater: argues IN FAVOR of a given position
  - Con Debater: argues AGAINST the position
  - Judge: scores both debaters and produces a neutral synthesis

The debate runs for 3 rounds:
  Round 1 — Opening statements (initial arguments)
  Round 2 — Rebuttals (respond to opponent's arguments)
  Round 3 — Closing statements (final arguments)

Graph:
  START -> pro_debater -> con_debater -> advance_round
                                            |
                            +---------------+
                            | more rounds   | done
                            v               v
                        pro_debater      judge -> END

Run: python week-04-advanced-patterns/solutions/solution_03_debate_agent.py
"""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================
# LLM Factory
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
# Step 1: DebateState TypedDict
# ==============================================================

class DebateState(TypedDict):
    topic: str                # The debate proposition
    pro_position: str         # Label for the pro side
    con_position: str         # Label for the con side
    pro_arguments: list       # List of pro arguments (one string per round)
    con_arguments: list       # List of con arguments (one string per round)
    round_number: int         # Current round (1, 2, or 3)
    max_rounds: int           # Always 3
    judge_scores: dict        # Scoring dict from the judge
    synthesis: str            # Final neutral synthesis


# ==============================================================
# Step 2: Pro Debater Node
# ==============================================================

ROUND_NAMES = {1: "Opening Statement", 2: "Rebuttal", 3: "Closing Statement"}


def pro_debater_node(state: DebateState) -> dict:
    """Generate arguments IN FAVOR of the topic."""
    llm = get_llm(temperature=0.7)
    round_num = state["round_number"]
    topic = state["topic"]
    round_name = ROUND_NAMES.get(round_num, f"Round {round_num}")

    print(f"\n  [Round {round_num} - {round_name}] Pro Debater ({state['pro_position']})...")

    system_prompt = (
        f"You are an expert debater arguing IN FAVOR of the following proposition:\n"
        f"\"{topic}\"\n\n"
        f"You are articulate, evidence-based, and persuasive. "
        f"Always support claims with reasoning, data, or real-world examples."
    )

    if round_num == 1:
        # Opening statement: present initial arguments
        user_prompt = (
            f"This is Round 1: Opening Statement.\n\n"
            f"Present 3 strong arguments IN FAVOR of: \"{topic}\"\n\n"
            f"For each argument:\n"
            f"- State the argument clearly\n"
            f"- Provide supporting evidence or reasoning\n"
            f"- Explain why this matters\n\n"
            f"Be concise but compelling. Keep your total response under 400 words."
        )
    elif round_num == 2:
        # Rebuttal: respond to opponent's arguments
        opponent_args = state["con_arguments"][-1] if state["con_arguments"] else "No arguments yet."
        user_prompt = (
            f"This is Round 2: Rebuttal.\n\n"
            f"Your opponent (arguing AGAINST) made these arguments:\n"
            f"---\n{opponent_args}\n---\n\n"
            f"Your tasks:\n"
            f"1. Address and counter your opponent's SPECIFIC arguments (reference them directly)\n"
            f"2. Point out weaknesses in their reasoning\n"
            f"3. Reinforce your strongest points with additional evidence\n\n"
            f"Be specific — quote or paraphrase their arguments before countering them. "
            f"Keep your response under 400 words."
        )
    else:
        # Closing statement
        opponent_args = state["con_arguments"][-1] if state["con_arguments"] else "No arguments yet."
        user_prompt = (
            f"This is Round 3: Closing Statement.\n\n"
            f"Your opponent's most recent arguments:\n"
            f"---\n{opponent_args}\n---\n\n"
            f"Give a compelling closing statement that:\n"
            f"1. Acknowledges the strongest counter-argument and explains why your position still holds\n"
            f"2. Summarizes your most powerful evidence\n"
            f"3. Ends with a memorable concluding point\n\n"
            f"Keep your response under 300 words."
        )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    argument = response.content

    # Print a preview
    preview = argument[:150].replace("\n", " ")
    print(f"    Pro: {preview}...")

    updated_args = list(state["pro_arguments"]) + [argument]
    return {"pro_arguments": updated_args}


# ==============================================================
# Step 3: Con Debater Node
# ==============================================================

def con_debater_node(state: DebateState) -> dict:
    """Generate arguments AGAINST the topic."""
    llm = get_llm(temperature=0.7)
    round_num = state["round_number"]
    topic = state["topic"]
    round_name = ROUND_NAMES.get(round_num, f"Round {round_num}")

    print(f"  [Round {round_num} - {round_name}] Con Debater ({state['con_position']})...")

    system_prompt = (
        f"You are an expert debater arguing AGAINST the following proposition:\n"
        f"\"{topic}\"\n\n"
        f"You are articulate, evidence-based, and persuasive. "
        f"Always support claims with reasoning, data, or real-world examples."
    )

    # The pro debater always goes first, so we always have their latest argument
    pro_args = state["pro_arguments"][-1] if state["pro_arguments"] else "No arguments yet."

    if round_num == 1:
        # Opening statement: present counter-arguments
        user_prompt = (
            f"This is Round 1: Opening Statement.\n\n"
            f"Your opponent (arguing IN FAVOR) made these arguments:\n"
            f"---\n{pro_args}\n---\n\n"
            f"Present 3 strong arguments AGAINST: \"{topic}\"\n\n"
            f"For each argument:\n"
            f"- State the argument clearly\n"
            f"- Provide supporting evidence or reasoning\n"
            f"- Explain why this matters\n\n"
            f"You may also address weaknesses in your opponent's opening. "
            f"Be concise but compelling. Keep your total response under 400 words."
        )
    elif round_num == 2:
        # Rebuttal: respond to pro's arguments
        user_prompt = (
            f"This is Round 2: Rebuttal.\n\n"
            f"Your opponent (arguing IN FAVOR) made these arguments:\n"
            f"---\n{pro_args}\n---\n\n"
            f"Your tasks:\n"
            f"1. Address and counter your opponent's SPECIFIC arguments (reference them directly)\n"
            f"2. Point out weaknesses in their reasoning\n"
            f"3. Reinforce your strongest points with additional evidence\n\n"
            f"Be specific — quote or paraphrase their arguments before countering them. "
            f"Keep your response under 400 words."
        )
    else:
        # Closing statement
        user_prompt = (
            f"This is Round 3: Closing Statement.\n\n"
            f"Your opponent's most recent arguments:\n"
            f"---\n{pro_args}\n---\n\n"
            f"Give a compelling closing statement that:\n"
            f"1. Acknowledges the strongest counter-argument and explains why your position still holds\n"
            f"2. Summarizes your most powerful evidence\n"
            f"3. Ends with a memorable concluding point\n\n"
            f"Keep your response under 300 words."
        )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    argument = response.content

    # Print a preview
    preview = argument[:150].replace("\n", " ")
    print(f"    Con: {preview}...")

    updated_args = list(state["con_arguments"]) + [argument]
    return {"con_arguments": updated_args}


# ==============================================================
# Step 4: Judge Node
# ==============================================================

def judge_node(state: DebateState) -> dict:
    """Score both debaters and produce a neutral synthesis."""
    llm = get_llm(temperature=0.3)  # Lower temperature for consistent judging
    topic = state["topic"]

    print(f"\n  [Judge] Evaluating the debate...")

    # Format all arguments for the judge
    pro_summary = ""
    con_summary = ""
    for i, (pro_arg, con_arg) in enumerate(zip(state["pro_arguments"], state["con_arguments"]), 1):
        round_name = ROUND_NAMES.get(i, f"Round {i}")
        pro_summary += f"\n--- Round {i} ({round_name}) ---\n{pro_arg}\n"
        con_summary += f"\n--- Round {i} ({round_name}) ---\n{con_arg}\n"

    system_prompt = (
        "You are an impartial debate judge. You evaluate arguments fairly based on "
        "their merit, regardless of which side you personally agree with.\n\n"
        "Score each debater on three criteria (1-10 each):\n"
        "  - argument_strength: How compelling and well-structured are the arguments?\n"
        "    (1=weak/vague, 5=adequate, 10=exceptional/irrefutable)\n"
        "  - evidence_quality: How well-supported are claims with data, examples, or logic?\n"
        "    (1=no evidence, 5=some support, 10=thoroughly evidenced)\n"
        "  - logical_consistency: Are arguments internally consistent and free of fallacies?\n"
        "    (1=contradictory/fallacious, 5=mostly sound, 10=flawless logic)\n\n"
        "Then produce a neutral synthesis that incorporates the strongest points from BOTH sides."
    )

    user_prompt = (
        f"Debate topic: \"{topic}\"\n\n"
        f"=== PRO DEBATER (arguing IN FAVOR) ===\n{pro_summary}\n\n"
        f"=== CON DEBATER (arguing AGAINST) ===\n{con_summary}\n\n"
        f"Please respond with ONLY valid JSON in this exact format:\n"
        f"{{\n"
        f"  \"pro_scores\": {{\n"
        f"    \"argument_strength\": <1-10>,\n"
        f"    \"evidence_quality\": <1-10>,\n"
        f"    \"logical_consistency\": <1-10>\n"
        f"  }},\n"
        f"  \"con_scores\": {{\n"
        f"    \"argument_strength\": <1-10>,\n"
        f"    \"evidence_quality\": <1-10>,\n"
        f"    \"logical_consistency\": <1-10>\n"
        f"  }},\n"
        f"  \"pro_total\": <sum of pro scores>,\n"
        f"  \"con_total\": <sum of con scores>,\n"
        f"  \"winner\": \"pro\" or \"con\" or \"tie\",\n"
        f"  \"justification\": \"<2-3 sentences explaining the decision>\",\n"
        f"  \"synthesis\": \"<3-5 sentence neutral synthesis incorporating both sides>\"\n"
        f"}}\n\n"
        f"Respond with ONLY the JSON object, no markdown formatting or extra text."
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    raw_output = response.content

    # Parse JSON from the judge's response
    judge_scores, synthesis = _parse_judge_output(raw_output)

    return {
        "judge_scores": judge_scores,
        "synthesis": synthesis,
    }


def _parse_judge_output(raw_output: str) -> tuple:
    """Parse the judge's JSON output with fallback handling."""
    # Try direct JSON parse
    try:
        data = json.loads(raw_output)
        synthesis = data.pop("synthesis", "No synthesis provided.")
        return data, synthesis
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            synthesis = data.pop("synthesis", "No synthesis provided.")
            return data, synthesis
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in the text
    brace_match = re.search(r"\{[\s\S]*\}", raw_output)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            synthesis = data.pop("synthesis", "No synthesis provided.")
            return data, synthesis
        except json.JSONDecodeError:
            pass

    # Fallback: return raw output as synthesis with default scores
    print("    [Warning] Could not parse judge JSON, using fallback scores")
    fallback_scores = {
        "pro_scores": {"argument_strength": 5, "evidence_quality": 5, "logical_consistency": 5},
        "con_scores": {"argument_strength": 5, "evidence_quality": 5, "logical_consistency": 5},
        "pro_total": 15,
        "con_total": 15,
        "winner": "tie",
        "justification": "Could not parse structured scores from judge output.",
    }
    return fallback_scores, raw_output[:500]


# ==============================================================
# Step 5: Round Advancement and Routing
# ==============================================================

def advance_round(state: DebateState) -> dict:
    """Increment the round number after both debaters have spoken."""
    new_round = state["round_number"] + 1
    print(f"\n  --- Round {state['round_number']} complete ---")
    return {"round_number": new_round}


def should_continue(state: DebateState) -> str:
    """Route to next round or to the judge."""
    if state["round_number"] > state["max_rounds"]:
        print(f"\n  All {state['max_rounds']} rounds complete. Sending to judge...")
        return "judge"
    else:
        print(f"\n  Proceeding to Round {state['round_number']}...")
        return "pro_debater"


# ==============================================================
# Step 6: Wire the Graph
# ==============================================================

graph = StateGraph(DebateState)

# Add nodes
graph.add_node("pro_debater", pro_debater_node)
graph.add_node("con_debater", con_debater_node)
graph.add_node("advance_round", advance_round)
graph.add_node("judge", judge_node)

# Set entry point
graph.set_entry_point("pro_debater")

# Edges
graph.add_edge("pro_debater", "con_debater")        # Pro always goes first, then con
graph.add_edge("con_debater", "advance_round")       # After both speak, advance round
graph.add_conditional_edges(
    "advance_round",
    should_continue,
    {
        "pro_debater": "pro_debater",   # More rounds -> back to pro
        "judge": "judge",               # All rounds done -> judge
    },
)
graph.add_edge("judge", END)

app = graph.compile()


# ==============================================================
# Run Function
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

    result = app.invoke(initial_state)

    # Print results
    print(f"\n{'='*70}")
    print("JUDGE'S DECISION")
    print(f"{'='*70}")

    if result.get("judge_scores"):
        scores = result["judge_scores"]

        # Pro scores
        pro_s = scores.get("pro_scores", {})
        print(f"\n  PRO Scores ({pro_position}):")
        print(f"    Argument Strength:    {pro_s.get('argument_strength', 'N/A')}/10")
        print(f"    Evidence Quality:     {pro_s.get('evidence_quality', 'N/A')}/10")
        print(f"    Logical Consistency:  {pro_s.get('logical_consistency', 'N/A')}/10")
        print(f"    TOTAL:                {scores.get('pro_total', 'N/A')}/30")

        # Con scores
        con_s = scores.get("con_scores", {})
        print(f"\n  CON Scores ({con_position}):")
        print(f"    Argument Strength:    {con_s.get('argument_strength', 'N/A')}/10")
        print(f"    Evidence Quality:     {con_s.get('evidence_quality', 'N/A')}/10")
        print(f"    Logical Consistency:  {con_s.get('logical_consistency', 'N/A')}/10")
        print(f"    TOTAL:                {scores.get('con_total', 'N/A')}/30")

        # Winner
        winner = scores.get("winner", "N/A")
        print(f"\n  Winner: {winner.upper()}")
        print(f"  Justification: {scores.get('justification', 'N/A')}")

    if result.get("synthesis"):
        print(f"\n  NEUTRAL SYNTHESIS:")
        print(f"  {result['synthesis']}")

    # Print full debate transcript
    print(f"\n{'='*70}")
    print("FULL DEBATE TRANSCRIPT")
    print(f"{'='*70}")
    for i in range(len(result["pro_arguments"])):
        round_name = ROUND_NAMES.get(i + 1, f"Round {i + 1}")
        print(f"\n  --- Round {i + 1}: {round_name} ---")
        print(f"\n  PRO:\n  {result['pro_arguments'][i][:300]}...")
        print(f"\n  CON:\n  {result['con_arguments'][i][:300]}...")

    return result


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    print("Solution 3: Multi-Agent Debate System with Judge")
    print("=" * 70)

    result = run_debate(
        topic="AI will create more jobs than it destroys in the next decade",
        pro_position="AI Creates Jobs",
        con_position="AI Destroys Jobs",
    )

    print(f"\n{'='*70}")
    print("Verification:")
    print(f"  Rounds completed:    {len(result['pro_arguments'])}")
    print(f"  Pro arguments:       {len(result['pro_arguments'])} entries")
    print(f"  Con arguments:       {len(result['con_arguments'])} entries")
    print(f"  Judge scores parsed: {'Yes' if result.get('judge_scores') else 'No'}")
    print(f"  Synthesis produced:  {'Yes' if result.get('synthesis') else 'No'}")
    print(f"{'='*70}")
