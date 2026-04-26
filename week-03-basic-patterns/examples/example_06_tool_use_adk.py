import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6: Tool Use Pattern in ADK — Multi-Tool Agent
=======================================================
The same multi-tool agent from Example 5, but using Google ADK.

ADK Differences:
  - Tools are PLAIN functions (no @tool decorator)
  - Type hints + docstrings are the schema (ADK reads them automatically)
  - The Runner manages the tool-calling loop internally
  - No explicit graph building — ADK handles routing

This example shows the same 3 tools (search, calculate, word info)
implemented the ADK way.

Run: python week-03-basic-patterns/examples/example_06_tool_use_adk.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Step 1: Define Tools as Plain Functions
# ==============================================================
# ADK reads the function name, docstring, and type hints to create
# the tool schema automatically. No decorator needed.

def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for current information about any topic.

    Use this when you need facts, statistics, news, or information
    that might not be in your training data.

    Args:
        query: Search query (e.g., 'population of Japan 2026')
        max_results: Number of results to return (1-5, default 3)

    Returns:
        Search results as formatted text
    """
    results_db = {
        "population": [
            "World Population 2026: approximately 8.1 billion people globally.",
            "India surpassed China as the most populous country in 2023.",
            "Japan's population has declined to approximately 122 million in 2026.",
        ],
        "ai": [
            "The global AI market is projected to reach $300 billion by 2027.",
            "Large Language Models have become the fastest-adopted technology in history.",
            "AI agents are transforming customer service, coding, and research workflows.",
        ],
        "climate": [
            "Global temperatures have risen 1.2°C above pre-industrial levels.",
            "Renewable energy now accounts for 35% of global electricity generation.",
            "The Paris Agreement targets limiting warming to 1.5°C above pre-industrial levels.",
        ],
    }

    for keyword, results in results_db.items():
        if keyword in query.lower():
            selected = results[:max_results]
            return "\n".join(f"  {i+1}. {r}" for i, r in enumerate(selected))

    return f"No results found for '{query}'. Try different keywords."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this for arithmetic, percentages, or numerical analysis.
    Supports: +, -, *, /, ** (power), parentheses.

    Args:
        expression: Math expression (e.g., '300 * 0.35', '8.1 - 1.4')

    Returns:
        The calculation result as a string
    """
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return f"Error: Invalid characters in '{expression}'."

    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error: Could not evaluate '{expression}': {e}"


def get_word_info(word: str) -> str:
    """Get the definition, synonyms, and usage example for a word.

    Use this when the user asks about word meanings, definitions,
    synonyms, or how to use a word.

    Args:
        word: The word to look up (e.g., 'resilient', 'ubiquitous')

    Returns:
        Word information including definition, synonyms, and example
    """
    dictionary = {
        "resilient": {
            "definition": "Able to recover quickly from difficulties; tough",
            "synonyms": ["robust", "adaptable", "flexible", "hardy"],
            "example": "The resilient community rebuilt after the disaster.",
        },
        "ubiquitous": {
            "definition": "Present, appearing, or found everywhere",
            "synonyms": ["omnipresent", "pervasive", "widespread"],
            "example": "Smartphones have become ubiquitous in modern life.",
        },
        "ephemeral": {
            "definition": "Lasting for a very short time",
            "synonyms": ["fleeting", "transient", "brief", "momentary"],
            "example": "The ephemeral beauty of cherry blossoms draws millions.",
        },
    }

    info = dictionary.get(word.lower())
    if info:
        return (
            f"Word: {word}\n"
            f"Definition: {info['definition']}\n"
            f"Synonyms: {', '.join(info['synonyms'])}\n"
            f"Example: {info['example']}"
        )
    return f"Word '{word}' not found. Try: resilient, ubiquitous, ephemeral."


# ==============================================================
# Step 2: Create the ADK Agent
# ==============================================================
# In ADK, you pass tools directly to LlmAgent. The agent's
# instruction prompt guides HOW it uses the tools.

agent = LlmAgent(
    name="multi_tool_assistant",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    instruction="""You are a helpful assistant with access to tools.

Available tools:
- search_web: Search for current facts, news, statistics
- calculate: Evaluate math expressions
- get_word_info: Look up word definitions and synonyms

Guidelines:
1. Use search_web when you need factual information
2. Use calculate for any numerical operations
3. Use get_word_info for language/vocabulary questions
4. You can chain tools — e.g., search first, then calculate
5. For simple questions you already know, respond directly
6. Always cite the tool results in your response""",
    tools=[search_web, calculate, get_word_info],
    description="Multi-tool assistant for research, math, and language queries.",
)


# ==============================================================
# Step 3: Run the Agent
# ==============================================================

async def ask(question: str):
    """Ask the ADK agent a question."""
    print(f"\n{'-'*60}")
    print(f"Question: {question}")
    print(f"{'-'*60}")

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="tool_use_demo",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="tool_use_demo",
        user_id="student",
    )

    result_text = ""
    try:
        async for event in runner.run_async(
            user_id="student",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=question)],
            ),
        ):
            if event.is_final_response():
                result_text = event.content.parts[0].text
    except Exception as e:
        result_text = f"Error: {type(e).__name__}: {e}"

    print(f"\n  Answer: {result_text[:500]}")


async def main():
    """Run all test questions."""
    print("Example 6: Multi-Tool Agent in ADK")
    print("=" * 60)

    # Test 1: Single tool (search)
    await ask("What is the current world population?")

    # Test 2: Single tool (calculate)
    await ask("If renewable energy is 35% of 29,000 TWh, how many TWh is that?")

    # Test 3: Multi-tool chaining
    await ask("Search for the AI market size, then calculate 15% of it.")

    # Test 4: Word info tool
    await ask("What does 'ephemeral' mean?")

    print(f"\n{'='*60}")
    print("LangGraph vs ADK — Tool Use Comparison:")
    print(f"{'='*60}")
    print("  LangGraph:")
    print("    - @tool decorator creates schemas")
    print("    - Explicit ToolNode + conditional edges")
    print("    - You control the loop (should_continue)")
    print("    - Full visibility into each step")
    print("\n  ADK:")
    print("    - Plain functions with docstrings")
    print("    - Runner handles tool loop internally")
    print("    - Less code, less control")
    print("    - Simpler for straightforward agents")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
