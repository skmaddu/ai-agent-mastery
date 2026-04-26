import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 2: Hierarchical Memory Agent
========================================
Difficulty: ⭐⭐⭐ Advanced | Time: 3 hours

Complete implementation of a LangGraph conversational agent with hierarchical memory:
  - L1 Buffer: Last N messages in full text
  - L2 Summary: Summarized blocks of older messages
  - L3 Facts: Extracted key facts that never expire

Graph:
  START → extract_facts → update_l1 → check_memory_pressure
                                           |
                                 (L1 full) → summarize_to_l2
                                 (L2 full) → archive_to_l3
                                           |
                                           v
                                     respond_with_context → END

Run: python week-05-context-memory/solutions/solution_02_memory_agent.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END


# ================================================================
# LLM Setup
# ================================================================

def get_llm(temperature=0.7):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )


# Memory capacity constants
L1_CAPACITY = 8    # Max messages in L1 buffer
L2_CAPACITY = 5    # Max summaries in L2
L1_KEEP = 4        # Messages to keep when draining L1


# ================================================================
# TODO 1: Define MemoryAgentState (SOLVED)
# ================================================================

class MemoryAgentState(TypedDict):
    user_input: str                 # current user message
    l1_messages: List[Dict]         # recent messages [{role, content}]
    l2_summaries: List[str]         # summarized blocks of older messages
    l3_facts: List[str]             # extracted key facts (permanent)
    new_facts: List[str]            # facts extracted this turn
    memory_action: str              # "none", "l1_to_l2", "l2_to_l3"
    response: str                   # agent's response


# ================================================================
# TODO 2: Implement extract_facts_node (SOLVED)
# ================================================================

FACT_KEYWORDS = ["budget", "allergic", "vegetarian", "prefer", "vegan",
                 "like", "don't like", "need", "want"]

def extract_facts_node(state: MemoryAgentState) -> dict:
    """Extract facts from the user's current message."""
    user_input = state["user_input"].lower()
    new_facts = []

    for keyword in FACT_KEYWORDS:
        if keyword in user_input:
            # Use the original (non-lowered) input for the fact
            fact = f"User said (re: {keyword}): {state['user_input']}"
            # Avoid duplicate facts
            if fact not in state.get("l3_facts", []):
                new_facts.append(fact)

    if new_facts:
        print(f"    [Facts] Extracted {len(new_facts)} fact(s)")

    return {"new_facts": new_facts}


# ================================================================
# TODO 3: Implement update_l1_node (SOLVED)
# ================================================================

def update_l1_node(state: MemoryAgentState) -> dict:
    """Add current message to L1 and merge new facts to L3."""
    # 1. Append the new user message to L1
    l1 = list(state.get("l1_messages", []))
    l1.append({"role": "user", "content": state["user_input"]})

    # 2. Merge new_facts into l3_facts (avoid duplicates)
    l3 = list(state.get("l3_facts", []))
    for fact in state.get("new_facts", []):
        if fact not in l3:
            l3.append(fact)

    print(f"    [L1] Now has {len(l1)} messages")
    return {"l1_messages": l1, "l3_facts": l3}


# ================================================================
# TODO 4: Implement check_memory_pressure (SOLVED)
# ================================================================

def check_memory_pressure(state: MemoryAgentState) -> str:
    """Route based on memory layer fullness."""
    l1_count = len(state.get("l1_messages", []))
    l2_count = len(state.get("l2_summaries", []))

    if l1_count > L1_CAPACITY:
        print(f"    [Pressure] L1 over capacity ({l1_count}/{L1_CAPACITY}) → summarize to L2")
        return "l1_to_l2"
    if l2_count > L2_CAPACITY:
        print(f"    [Pressure] L2 over capacity ({l2_count}/{L2_CAPACITY}) → archive to L3")
        return "l2_to_l3"
    return "respond"


# ================================================================
# TODO 5: Implement summarize_to_l2_node (SOLVED)
# ================================================================

def summarize_to_l2_node(state: MemoryAgentState) -> dict:
    """Summarize old L1 messages and move to L2."""
    l1 = list(state.get("l1_messages", []))

    # Split: drain old messages, keep the most recent L1_KEEP
    old_messages = l1[:-L1_KEEP]
    keep_messages = l1[-L1_KEEP:]

    # Create a summary from the drained messages
    summary_parts = []
    for msg in old_messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        # Truncate long messages for the summary
        if len(content) > 100:
            content = content[:100] + "..."
        summary_parts.append(f"{role}: {content}")

    summary = "Summary of conversation: " + " | ".join(summary_parts)

    l2 = list(state.get("l2_summaries", []))
    l2.append(summary)

    print(f"    [L1→L2] Drained {len(old_messages)} messages, created summary. L1: {len(keep_messages)}, L2: {len(l2)}")
    return {"l1_messages": keep_messages, "l2_summaries": l2}


# ================================================================
# TODO 6: Implement archive_to_l3_node (SOLVED)
# ================================================================

def archive_to_l3_node(state: MemoryAgentState) -> dict:
    """Archive old L2 summaries as L3 facts."""
    l2 = list(state.get("l2_summaries", []))

    # Keep the 2 most recent summaries, archive the rest
    old_summaries = l2[:-2]
    keep_summaries = l2[-2:]

    # Convert old summaries to brief fact strings
    l3 = list(state.get("l3_facts", []))
    for summary in old_summaries:
        fact = f"Historical: {summary[:150]}"
        if fact not in l3:
            l3.append(fact)

    print(f"    [L2→L3] Archived {len(old_summaries)} summaries as facts. L2: {len(keep_summaries)}, L3: {len(l3)}")
    return {"l2_summaries": keep_summaries, "l3_facts": l3}


# ================================================================
# TODO 7: Implement respond_with_context_node (SOLVED)
# ================================================================

llm = get_llm(temperature=0.7)

def respond_with_context_node(state: MemoryAgentState) -> dict:
    """Generate response using all memory layers as context."""
    # Build system prompt from all three memory layers
    context_parts = []

    # L3: permanent facts
    l3_facts = state.get("l3_facts", [])
    if l3_facts:
        context_parts.append("KNOWN FACTS (permanent memory):\n" + "\n".join(f"- {f}" for f in l3_facts))

    # L2: conversation summaries
    l2_summaries = state.get("l2_summaries", [])
    if l2_summaries:
        context_parts.append("CONVERSATION HISTORY (summaries):\n" + "\n".join(f"- {s}" for s in l2_summaries))

    # L1: recent messages as conversation context
    l1_messages = state.get("l1_messages", [])
    if l1_messages:
        recent = []
        for msg in l1_messages:
            recent.append(f"{msg['role'].capitalize()}: {msg['content']}")
        context_parts.append("RECENT MESSAGES:\n" + "\n".join(recent))

    system_content = (
        "You are a helpful travel assistant with hierarchical memory. "
        "Use the context below to provide informed, personalized responses. "
        "Remember the user's preferences and past conversation topics.\n\n"
        + "\n\n".join(context_parts)
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=state["user_input"]),
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    # Add assistant response to L1
    l1 = list(state.get("l1_messages", []))
    l1.append({"role": "assistant", "content": answer})

    return {"response": answer, "l1_messages": l1}


# ================================================================
# Build the graph
# ================================================================

def build_memory_agent():
    graph = StateGraph(MemoryAgentState)

    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("update_l1", update_l1_node)
    graph.add_node("summarize_to_l2", summarize_to_l2_node)
    graph.add_node("archive_to_l3", archive_to_l3_node)
    graph.add_node("respond", respond_with_context_node)

    graph.set_entry_point("extract_facts")
    graph.add_edge("extract_facts", "update_l1")

    graph.add_conditional_edges("update_l1", check_memory_pressure, {
        "l1_to_l2": "summarize_to_l2",
        "l2_to_l3": "archive_to_l3",
        "respond": "respond",
    })

    graph.add_edge("summarize_to_l2", "respond")
    graph.add_edge("archive_to_l3", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SOLUTION 2: Hierarchical Memory Agent")
    print("=" * 65)

    app = build_memory_agent()

    turns = [
        "Hi! I'm planning a trip to Japan. My budget is $3000.",
        "I'm vegetarian and allergic to nuts.",
        "What cities should I visit in 2 weeks?",
        "Tell me about Kyoto's temples.",
        "What food should I try in Osaka?",
        "How does the Japan Rail Pass work?",
        "Can you recommend budget hotels?",
        "What's the weather like in spring?",
        "Any tips for navigating Tokyo subway?",
        "Summarize what we've discussed so far.",
        "Do you remember my dietary restrictions?",
    ]

    state = {
        "user_input": "",
        "l1_messages": [],
        "l2_summaries": [],
        "l3_facts": [],
        "new_facts": [],
        "memory_action": "none",
        "response": "",
    }

    for i, turn in enumerate(turns):
        print(f"\n{'━' * 65}")
        print(f"  Turn {i + 1}: {turn}")
        print(f"{'━' * 65}")

        state["user_input"] = turn
        result = app.invoke(state)

        # Carry state forward
        state["l1_messages"] = result.get("l1_messages", [])
        state["l2_summaries"] = result.get("l2_summaries", [])
        state["l3_facts"] = result.get("l3_facts", [])

        print(f"  Response: {result.get('response', '[no response]')[:200]}")
        print(f"  L1: {len(state['l1_messages'])} msgs | "
              f"L2: {len(state['l2_summaries'])} summaries | "
              f"L3: {len(state['l3_facts'])} facts")
