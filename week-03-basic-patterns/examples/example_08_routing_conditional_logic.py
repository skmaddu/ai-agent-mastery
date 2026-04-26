import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 8: Routing & Conditional Logic in Agents
==================================================
Routing is how agents decide WHAT to do next. Instead of following
a fixed sequence, the agent evaluates the current state and chooses
a path — like a traffic controller directing cars.

This example covers 3 routing strategies:
  1. Content-Based Routing: Route based on what the input contains
  2. LLM-Based Routing: Let the LLM classify and choose the path
  3. Multi-Path Routing: Fan out to multiple handlers, merge results

All implemented in LangGraph to show how conditional edges work
in practice.

Run: python week-03-basic-patterns/examples/example_08_routing_conditional_logic.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict


# ==============================================================
# STRATEGY 1: Content-Based Routing (No LLM)
# ==============================================================
# Route based on keywords or patterns in the input.
# Fast and cheap — no LLM call needed for routing.

class ContentRouterState(TypedDict):
    query: str         # User's input
    category: str      # Detected category
    response: str      # Handler's response


def classify_by_content(state: ContentRouterState) -> dict:
    """Classify the query based on keywords — no LLM needed."""
    query = state["query"].lower()

    # Simple keyword-based classification
    if any(word in query for word in ["weather", "temperature", "rain", "forecast"]):
        category = "weather"
    elif any(word in query for word in ["calculate", "math", "compute", "sum", "add", "multiply"]):
        category = "math"
    elif any(word in query for word in ["define", "meaning", "synonym", "word"]):
        category = "language"
    else:
        category = "general"

    print(f"  [Router] Classified as: {category}")
    return {"category": category}


def handle_weather(state: ContentRouterState) -> dict:
    """Handle weather-related queries."""
    return {"response": f"[Weather Handler] Looking up weather for: {state['query']}"}


def handle_math(state: ContentRouterState) -> dict:
    """Handle math-related queries."""
    return {"response": f"[Math Handler] Calculating: {state['query']}"}


def handle_language(state: ContentRouterState) -> dict:
    """Handle language/vocabulary queries."""
    return {"response": f"[Language Handler] Looking up: {state['query']}"}


def handle_general(state: ContentRouterState) -> dict:
    """Handle queries that don't match specific categories."""
    return {"response": f"[General Handler] Answering: {state['query']}"}


def route_by_category(state: ContentRouterState) -> str:
    """Conditional edge function — returns the next node name."""
    return state["category"]


def demo_content_routing():
    """Demonstrate content-based routing."""

    print("=" * 60)
    print("STRATEGY 1: Content-Based Routing (No LLM)")
    print("=" * 60)

    # Build the routing graph
    graph = StateGraph(ContentRouterState)

    # Add nodes
    graph.add_node("classify", classify_by_content)
    graph.add_node("weather", handle_weather)
    graph.add_node("math", handle_math)
    graph.add_node("language", handle_language)
    graph.add_node("general", handle_general)

    # Entry point -> classify
    graph.set_entry_point("classify")

    # Conditional routing based on classification result
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "weather": "weather",
            "math": "math",
            "language": "language",
            "general": "general",
        },
    )

    # All handlers -> END
    graph.add_edge("weather", END)
    graph.add_edge("math", END)
    graph.add_edge("language", END)
    graph.add_edge("general", END)

    app = graph.compile()

    # Test with different queries
    queries = [
        "What's the weather in London?",
        "Calculate 15 * 7 + 3",
        "Define the word 'ephemeral'",
        "Tell me about the history of computers",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        result = app.invoke({"query": query, "category": "", "response": ""})
        print(f"  Result: {result['response']}")


# ==============================================================
# STRATEGY 2: LLM-Based Routing
# ==============================================================
# Let the LLM classify the input — handles ambiguous cases better
# than keyword matching, but costs an API call.

class LLMRouterState(TypedDict):
    query: str
    category: str
    response: str


def get_llm():
    """Create LLM for routing."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def llm_classify(state: LLMRouterState) -> dict:
    """Use the LLM to classify the query into a category."""
    llm = get_llm()

    messages = [
        SystemMessage(content=(
            "Classify the user's query into exactly ONE category. "
            "Respond with ONLY the category name, nothing else.\n"
            "Categories: weather, math, language, general"
        )),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    category = response.content.strip().lower()

    # Validate — fall back to "general" if LLM returns unexpected value
    valid_categories = {"weather", "math", "language", "general"}
    if category not in valid_categories:
        print(f"  [Router] LLM returned '{category}', defaulting to 'general'")
        category = "general"

    print(f"  [LLM Router] Classified as: {category}")
    return {"category": category}


def demo_llm_routing():
    """Demonstrate LLM-based routing."""

    print(f"\n{'='*60}")
    print("STRATEGY 2: LLM-Based Routing")
    print("=" * 60)

    graph = StateGraph(LLMRouterState)

    graph.add_node("classify", llm_classify)
    graph.add_node("weather", handle_weather)
    graph.add_node("math", handle_math)
    graph.add_node("language", handle_language)
    graph.add_node("general", handle_general)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_by_category,
        {"weather": "weather", "math": "math", "language": "language", "general": "general"},
    )
    graph.add_edge("weather", END)
    graph.add_edge("math", END)
    graph.add_edge("language", END)
    graph.add_edge("general", END)

    app = graph.compile()

    # Test with ambiguous queries that keyword matching would get wrong
    queries = [
        "Is it going to be cold tomorrow?",          # Weather — no keyword "weather"
        "How much would three dozen eggs cost at $4 each?",  # Math — no keyword "calculate"
        "What's a fancy way to say 'short-lived'?",  # Language — no keyword "define"
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        result = app.invoke({"query": query, "category": "", "response": ""})
        print(f"  Result: {result['response']}")

    print(f"\n  [TIP] Notice: LLM routing handles ambiguous queries that keyword")
    print(f"     matching misses. Tradeoff: costs an extra LLM call per request.")


# ==============================================================
# STRATEGY 3: Multi-Path Routing (Fan-Out / Fan-In)
# ==============================================================
# Sometimes you want to process the SAME input through MULTIPLE
# handlers and merge the results. This is the fan-out/fan-in pattern.

class MultiPathState(TypedDict):
    query: str
    factual_response: str
    creative_response: str
    combined_response: str


def factual_handler(state: MultiPathState) -> dict:
    """Handle with a factual approach."""
    return {"factual_response": f"[Factual] Based on data: '{state['query']}' has documented evidence..."}


def creative_handler(state: MultiPathState) -> dict:
    """Handle with a creative approach."""
    return {"creative_response": f"[Creative] Imagine: '{state['query']}' opens up exciting possibilities..."}


def merge_responses(state: MultiPathState) -> dict:
    """Merge results from multiple handlers."""
    combined = (
        f"Factual view: {state['factual_response']}\n"
        f"Creative view: {state['creative_response']}"
    )
    return {"combined_response": combined}


def demo_multipath_routing():
    """Demonstrate fan-out / fan-in routing."""

    print(f"\n{'='*60}")
    print("STRATEGY 3: Multi-Path Routing (Fan-Out / Fan-In)")
    print("=" * 60)
    print("Same input processed by multiple handlers, results merged.\n")

    graph = StateGraph(MultiPathState)

    graph.add_node("factual", factual_handler)
    graph.add_node("creative", creative_handler)
    graph.add_node("merge", merge_responses)

    graph.set_entry_point("factual")
    # Note: LangGraph doesn't natively parallelize nodes in basic mode.
    # We simulate fan-out by running handlers sequentially then merging.
    graph.add_edge("factual", "creative")
    graph.add_edge("creative", "merge")
    graph.add_edge("merge", END)

    app = graph.compile()

    query = "The future of space exploration"
    print(f"  Query: \"{query}\"")
    result = app.invoke({
        "query": query,
        "factual_response": "", "creative_response": "", "combined_response": "",
    })
    print(f"\n  Combined Result:")
    print(f"    {result['combined_response']}")

    print(f"\n  [TIP] Fan-out/fan-in is useful when you want multiple perspectives,")
    print(f"     consensus (agree/disagree), or redundancy (if one fails, others work).")


# ==============================================================
# Summary
# ==============================================================

def summary():
    """Compare routing strategies."""

    print(f"\n{'='*60}")
    print("Summary: Choosing a Routing Strategy")
    print("=" * 60)

    print(f"\n  {'Strategy':<25} {'Cost':<15} {'Accuracy':<15} {'Best For'}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*25}")
    print(f"  {'Content-Based':<25} {'Free':<15} {'Good':<15} {'Clear keyword patterns'}")
    print(f"  {'LLM-Based':<25} {'1 API call':<15} {'Excellent':<15} {'Ambiguous inputs'}")
    print(f"  {'Multi-Path':<25} {'Varies':<15} {'N/A':<15} {'Multiple perspectives'}")


if __name__ == "__main__":
    print("Example 8: Routing & Conditional Logic")
    print("=" * 60)

    # Strategy 1: No LLM needed
    demo_content_routing()

    # Strategy 2: Needs LLM
    demo_llm_routing()

    # Strategy 3: Multiple handlers
    demo_multipath_routing()

    summary()
