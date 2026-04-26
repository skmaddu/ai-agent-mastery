"""
ADK Implementation — Topic Research Agent
===========================================
Week 2: Added tool integration and error handling.

The agent now:
  1. Uses tools (search, calculate) for research
  2. Has error handling in the run_research function
  3. Uses an expanded instruction prompt with tool guidance

(Evolves each week — see git history for progression)
"""

import logging
import os
import sys

# Add project source to path for tool imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import plain-function version of search (ADK needs no @tool decorator)
from tools.search import search_web_plain
from tools.calculator import _safe_eval


# ── ADK-compatible tool wrappers ─────────────────────────────
# ADK needs plain functions with type hints and docstrings.
# These wrap the project's existing tool logic.

def search_topic(query: str, max_results: int = 5) -> str:
    """Search for information about a research topic.

    Args:
        query: What to search for
        max_results: Maximum number of results (default: 5)

    Returns:
        Search results as formatted text
    """
    return search_web_plain(query, max_results)


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Supports: +, -, *, /, ** (power), parentheses.
    Examples: '2 + 3 * 4', '100 / 7', '2 ** 10'
    """
    import ast
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def create_research_agent():
    """Build and return the ADK research agent with tools."""

    agent = LlmAgent(
        name="topic_researcher",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are an expert research analyst with access to tools.

Available tools:
- search_topic: Search for information about a topic. Use this to gather facts.
- calculate: Evaluate math expressions for any numerical analysis.

When given a topic to research:
1. Use search_topic to find key information about the topic
2. Use calculate if any numerical analysis is needed
3. Provide a comprehensive summary with:
   - A clear title
   - 3-5 key findings, each with its importance
   - 2-3 follow-up questions for deeper research
   - Any areas of uncertainty or debate

If a tool returns an error, acknowledge it and try a different approach.
Be factual and balanced in your analysis.""",
        tools=[search_topic, calculate],
        description="Expert research analyst that investigates topics using search and analysis tools.",
    )

    return agent


async def run_research(topic: str) -> str:
    """Run the ADK research agent on a given topic.

    Args:
        topic: The research topic to investigate

    Returns:
        Research output as text
    """
    agent = create_research_agent()
    session_service = InMemorySessionService()

    runner = Runner(
        agent=agent,
        app_name="topic_research",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="topic_research",
        user_id="researcher",
    )

    result_text = ""
    try:
        async for event in runner.run_async(
            user_id="researcher",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=f"Research this topic thoroughly: {topic}")],
            ),
        ):
            if event.is_final_response():
                result_text = event.content.parts[0].text
    except Exception as e:
        result_text = (
            f"Error during research: {type(e).__name__}: {e}\n"
            f"The agent encountered an issue while researching '{topic}'. "
            f"Please check your GOOGLE_API_KEY and try again."
        )

    return result_text
