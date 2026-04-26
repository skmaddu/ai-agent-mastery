"""
Web Search Tool
================
Provides web search capability for the research agent.
Add your preferred search API key to config/.env.

Week 2: Mock search with realistic structure (real API in Week 3).
Week 3+: Replace with real search API (Tavily, SerpAPI, or Brave).
"""

import requests
from langchain_core.tools import tool


# Mock search results for common research topics
# This gives the agent realistic data to work with before we add a real API
MOCK_RESULTS = {
    "ai agents": [
        "AI agents are autonomous software systems that perceive their environment and take actions to achieve goals.",
        "Key components of AI agents include: perception, reasoning, planning, and action execution.",
        "Modern AI agents use large language models (LLMs) as their reasoning engine.",
    ],
    "machine learning": [
        "Machine learning is a subset of AI where systems learn from data without explicit programming.",
        "Common ML paradigms: supervised learning, unsupervised learning, and reinforcement learning.",
        "Deep learning uses neural networks with many layers to learn complex patterns.",
    ],
    "langgraph": [
        "LangGraph is a framework for building stateful, multi-actor LLM applications as graphs.",
        "LangGraph uses StateGraph with nodes (functions) and edges (connections) to define agent workflows.",
        "Key features: conditional routing, state persistence, and human-in-the-loop support.",
    ],
    "google adk": [
        "Google ADK (Agent Development Kit) is a framework for building AI agents with Gemini models.",
        "ADK uses declarative agent configuration with LlmAgent, Runner, and SessionService.",
        "ADK handles the tool-calling loop automatically, unlike LangGraph's explicit graph approach.",
    ],
}


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query string
        max_results: Maximum results to return (default: 5)

    Returns:
        Search results as formatted text
    """
    # Week 2: Return mock results matching the query topic
    # Week 3: Replace with real search API
    query_lower = query.lower()

    # Try to match a known topic
    for topic, results in MOCK_RESULTS.items():
        if topic in query_lower:
            limited = results[:max_results]
            formatted = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(limited))
            return f"Search results for '{query}':\n{formatted}"

    # Default fallback for unknown topics
    return (
        f"Search results for '{query}':\n"
        f"  1. '{query}' is an active area of research and development.\n"
        f"  2. Key aspects include methodology, applications, and current trends.\n"
        f"  3. Further research recommended for detailed information."
    )


def search_web_plain(query: str, max_results: int = 5) -> str:
    """Plain function version of search_web for ADK compatibility.

    ADK requires plain functions (no @tool decorator).
    This calls the same logic as the LangGraph version.

    Args:
        query: The search query string
        max_results: Maximum results to return (default: 5)

    Returns:
        Search results as formatted text
    """
    query_lower = query.lower()

    for topic, results in MOCK_RESULTS.items():
        if topic in query_lower:
            limited = results[:max_results]
            formatted = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(limited))
            return f"Search results for '{query}':\n{formatted}"

    return (
        f"Search results for '{query}':\n"
        f"  1. '{query}' is an active area of research and development.\n"
        f"  2. Key aspects include methodology, applications, and current trends.\n"
        f"  3. Further research recommended for detailed information."
    )
