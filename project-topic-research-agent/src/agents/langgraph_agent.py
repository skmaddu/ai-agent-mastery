"""
LangGraph Implementation — Topic Research Agent
=================================================
Week 1: Basic LLM summarizer with structured output (Pydantic) + Phoenix tracing
        (tracing is enabled from main.py via LangChain instrumentor).

Later weeks extend this graph with tools, loops, and multi-agent flows.
"""

from __future__ import annotations

import os
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from schemas.research_schemas import ResearchReport


def _make_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
    )


class ResearchState(TypedDict):
    """Minimal state for Week 1 — grows in later weeks."""

    topic: str
    structured_output: Optional[dict]
    iteration: int


def create_research_agent():
    """Single-node graph: one structured LLM call for the research report."""

    llm = _make_llm()
    structured_llm = llm.with_structured_output(ResearchReport)

    def research_node(state: ResearchState) -> dict:
        topic = state["topic"]
        messages = [
            SystemMessage(
                content=(
                    "You are an expert research analyst. "
                    "Produce an accurate, balanced report. "
                    "If you did not use real external sources, set sources to null or empty."
                )
            ),
            HumanMessage(
                content=(
                    f"Research the following topic and fill the structured report schema.\n\n"
                    f"Topic: {topic}"
                )
            ),
        ]
        report: ResearchReport = structured_llm.invoke(messages)
        return {
            "structured_output": report.model_dump(),
            "iteration": state.get("iteration", 0) + 1,
        }

    graph = StateGraph(ResearchState)
    graph.add_node("research", research_node)
    graph.set_entry_point("research")
    graph.add_edge("research", END)

    return graph.compile()


def run_research(topic: str) -> dict:
    """Run the LangGraph research agent on a given topic.

    Args:
        topic: The research topic (already validated / sanitized by caller)

    Returns:
        Structured research output dict (ResearchReport fields)
    """
    agent = create_research_agent()
    result = agent.invoke(
        {
            "topic": topic,
            "structured_output": None,
            "iteration": 0,
        }
    )
    report = result.get("structured_output") or {}
    return {
        "topic": topic,
        "report": report,
        "iteration": result.get("iteration", 0),
    }
