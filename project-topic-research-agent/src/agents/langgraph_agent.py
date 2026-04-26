"""
LangGraph Implementation — Topic Research Agent
=================================================
Week 2: Added tool integration, conditional routing, error handling.

The agent now:
  1. Uses tools (search_web, calculate) to gather information
  2. Routes conditionally: agent → tools loop → format → END
  3. Has error handling with retry logic and fallback responses
  4. Tracks iterations to prevent infinite loops

(Evolves each week — see git history for progression)
"""

import os
import sys

# Add project source to path for tool imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import add_messages

# Import project tools
from tools.search import search_web
from tools.calculator import calculate


class ResearchState(TypedDict):
    """Minimal state for Week 1 — grows in later weeks."""

    topic: str
    structured_output: Optional[dict]
    iteration: int
    error_count: int


# Configuration
MAX_ITERATIONS = 8    # Prevent infinite tool-calling loops
MAX_LLM_RETRIES = 3  # Retries for malformed tool calls


def create_research_agent():
    """Build and return the LangGraph research agent with tools."""

    tools = [search_web, calculate]

    # LLM setup with provider flexibility
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.7,
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
        )

    llm_with_tools = llm.bind_tools(tools)

    # ── Agent node: calls the LLM with tools bound ────────
    def agent_node(state: ResearchState) -> dict:
        """Call the LLM — it may request tool calls or give a direct answer."""
        iteration = state.get("iteration", 0) + 1

        # Build system context with the research topic
        topic = state.get("topic", "")
        if topic and not any(
            topic.lower() in msg.content.lower()
            for msg in state["messages"]
            if hasattr(msg, "content") and isinstance(msg.content, str)
        ):
            # First call: inject the research prompt
            research_prompt = HumanMessage(
                content=f"""You are a research analyst. Research this topic thoroughly: {topic}

Use the search_web tool to find information. Use calculate for any numerical analysis.
After gathering information, provide a structured summary with:
- Title
- Key findings (3-5 points)
- Follow-up questions for deeper research"""
            )
            messages = state["messages"] + [research_prompt]
        else:
            messages = state["messages"]

        # Retry logic for malformed tool calls (common with Groq/Llama)
        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = llm_with_tools.invoke(messages)
                return {
                    "messages": [response],
                    "iteration": iteration,
                }
            except Exception as e:
                if attempt < MAX_LLM_RETRIES - 1:
                    continue
                # All retries failed — return a fallback message
                fallback = AIMessage(
                    content=f"I encountered an error researching '{topic}'. "
                            f"Error: {type(e).__name__}. Please try again."
                )
                return {
                    "messages": [fallback],
                    "iteration": iteration,
                    "error_count": state.get("error_count", 0) + 1,
                }

    # ── Routing: tools, format, or end? ────────────────────
    def should_continue(state: ResearchState) -> str:
        """Route based on LLM response and safety limits."""
        # Safety: prevent infinite loops
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "format"

        # Too many errors: go to format with what we have
        if state.get("error_count", 0) >= 3:
            return "format"

        # Check if LLM wants to call tools
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"

        # No tool calls: LLM gave a direct answer → format it
        return "format"

    # ── Format node: structure the final output ────────────
    def format_node(state: ResearchState) -> dict:
        """Format raw research into structured output."""
        # Get the last text content from the agent
        raw_research = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                raw_research = msg.content
                break

        return {
            "raw_research": raw_research,
            "structured_output": {
                "topic": state["topic"],
                "research": raw_research,
                "iteration": state.get("iteration", 0),
                "errors": state.get("error_count", 0),
            },
        }

    # ── Build the graph ────────────────────────────────────
    # Flow: agent → [tools → agent]* → format → END
    graph = StateGraph(ResearchState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("format", format_node)

    graph.set_entry_point("agent")

    # Conditional routing from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",     # LLM wants to use a tool
            "format": "format",   # LLM gave a direct answer
        },
    )

    # After tools execute, go back to agent
    graph.add_edge("tools", "agent")

    # After format, we're done
    graph.add_edge("format", END)

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
        "raw_research": "",
        "structured_output": None,
        "iteration": 0,
        "error_count": 0,
    })

    return result.get("structured_output", {})
