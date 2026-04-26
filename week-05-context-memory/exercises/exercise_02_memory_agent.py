import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 2: Hierarchical Memory Agent
========================================
Difficulty: ⭐⭐⭐ Advanced | Time: 3 hours

Task:
Build a LangGraph conversational agent with hierarchical memory:
  - L1 Buffer: Last N messages in full text
  - L2 Summary: Summarized blocks of older messages
  - L3 Facts: Extracted key facts that never expire

The agent should:
  1. Extract facts from every user message
  2. Store messages in L1 buffer
  3. When L1 is full, summarize old messages to L2
  4. When L2 is full, extract key facts to L3
  5. Build response context from all three layers

Graph:
  START → extract_facts → update_l1 → check_memory_pressure
                                           |
                                 (L1 full) → summarize_to_l2
                                 (L2 full) → archive_to_l3
                                           |
                                           v
                                     respond_with_context → END

Instructions:
1. Define MemoryAgentState with L1, L2, L3 fields (TODO 1)
2. Implement extract_facts_node (TODO 2)
3. Implement update_l1_node (TODO 3)
4. Implement check_memory_pressure routing (TODO 4)
5. Implement summarize_to_l2_node (TODO 5)
6. Implement archive_to_l3_node (TODO 6)
7. Implement respond_with_context_node (TODO 7)

Hints:
- Study example_13_memory_patterns.py for hierarchical memory
- Study example_14_langgraph_state_memory.py for LangGraph state
- L1 capacity: 8 messages.  L2 capacity: 5 summaries.
- Use simple keyword matching for fact extraction (no LLM needed)
- The routing function should check L1 → L2 → L3 in order

Run: python week-05-context-memory/exercises/exercise_02_memory_agent.py
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
# TODO 1: Define MemoryAgentState
# ================================================================
# Fields:
#   - user_input: str                — current user message
#   - l1_messages: List[Dict]        — recent messages [{role, content, timestamp}]
#   - l2_summaries: List[str]        — summarized blocks of older messages
#   - l3_facts: List[str]            — extracted key facts (permanent)
#   - new_facts: List[str]           — facts extracted this turn
#   - memory_action: str             — "none", "l1_to_l2", "l2_to_l3"
#   - response: str                  — agent's response

class MemoryAgentState(TypedDict):
    # TODO: Define all fields
    pass


# ================================================================
# TODO 2: Implement extract_facts_node
# ================================================================
# Extract factual statements from the user's message.
# Look for keywords: "budget", "allergic", "vegetarian", "prefer",
# "vegan", "like", "don't like", "need", "want"
# For each keyword match, create a fact string.

def extract_facts_node(state: MemoryAgentState) -> dict:
    """Extract facts from the user's current message."""
    # TODO: Implement fact extraction using keyword matching
    # Return: {"new_facts": [list of extracted fact strings]}
    pass


# ================================================================
# TODO 3: Implement update_l1_node
# ================================================================
# Add the current user message to L1 buffer.
# Also merge any new_facts into l3_facts (facts are permanent).

def update_l1_node(state: MemoryAgentState) -> dict:
    """Add current message to L1 and merge new facts to L3."""
    # TODO:
    # 1. Append {"role": "user", "content": user_input} to l1_messages
    # 2. Merge new_facts into l3_facts (avoid duplicates)
    # 3. Return updated l1_messages and l3_facts
    pass


# ================================================================
# TODO 4: Implement check_memory_pressure
# ================================================================
# This is a routing function that returns one of:
#   "l1_to_l2" — if L1 has more messages than L1_CAPACITY
#   "l2_to_l3" — if L2 has more summaries than L2_CAPACITY
#   "respond"  — if no memory pressure

def check_memory_pressure(state: MemoryAgentState) -> str:
    """Route based on memory layer fullness."""
    # TODO: Check L1 size first, then L2 size, default to "respond"
    pass


# ================================================================
# TODO 5: Implement summarize_to_l2_node
# ================================================================
# When L1 is over capacity:
#   1. Take the oldest messages (keep the last L1_KEEP)
#   2. Create a summary of the drained messages
#   3. Add the summary to L2
# Use a simple concatenation summary (no LLM needed for exercise)

def summarize_to_l2_node(state: MemoryAgentState) -> dict:
    """Summarize old L1 messages and move to L2."""
    # TODO:
    # 1. Split l1_messages: old = [:-L1_KEEP], keep = [-L1_KEEP:]
    # 2. Create summary string from old messages
    # 3. Return updated l1_messages (just keep) and l2_summaries (appended)
    pass


# ================================================================
# TODO 6: Implement archive_to_l3_node
# ================================================================
# When L2 is over capacity:
#   1. Take the oldest summaries (keep the last 2)
#   2. Extract key phrases as facts
#   3. Add to L3 facts

def archive_to_l3_node(state: MemoryAgentState) -> dict:
    """Archive old L2 summaries as L3 facts."""
    # TODO:
    # 1. Split l2_summaries: old = [:-2], keep = [-2:]
    # 2. Convert old summaries to brief fact strings
    # 3. Return updated l2_summaries and l3_facts
    pass


# ================================================================
# TODO 7: Implement respond_with_context_node
# ================================================================
# Build context from all three memory layers and generate response.

llm = get_llm(temperature=0.7)

def respond_with_context_node(state: MemoryAgentState) -> dict:
    """Generate response using all memory layers as context."""
    # TODO:
    # 1. Build system prompt with:
    #    - L3 facts as "Known facts"
    #    - L2 summaries as "Conversation history"
    #    - L1 recent messages as conversation context
    # 2. Add current user_input as HumanMessage
    # 3. Invoke LLM and return response
    # 4. Add assistant response to l1_messages
    pass


# ================================================================
# Build the graph (partially provided)
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
    print("╔" + "═" * 63 + "╗")
    print("║  EXERCISE 2: Hierarchical Memory Agent                        ║")
    print("╚" + "═" * 63 + "╝")

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
