"""
Solution: Exercise 2 — Multi-Tool ADK Agent
=============================================
An ADK agent with calculate, to_uppercase, and convert_temperature
tools. Handles multi-step queries by chaining tool calls automatically.

Requires: GOOGLE_API_KEY in your environment.

Run: python week-02-framework-basics/solutions/solution_02_multi_tool_adk_agent.py
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


# ── Tools (plain functions — no decorators) ──────────────────

def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Supports: +, -, *, /, ** (power), parentheses.
    Examples: '25 * 4', '(100 - 32) * 5 / 9', '2 ** 8'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in '{expression}'. Use only numbers and +-*/()"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def to_uppercase(text: str) -> str:
    """Convert text to uppercase.
    Use this when the user asks to capitalize or uppercase text.
    """
    return text.upper()


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.
    from_unit and to_unit should be 'celsius' or 'fahrenheit'.

    Formulas:
    - Celsius to Fahrenheit: F = C * 9/5 + 32
    - Fahrenheit to Celsius: C = (F - 32) * 5/9
    """
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    if from_unit == to_unit:
        return f"{value}° {from_unit} is already in {to_unit}"

    if from_unit == "celsius" and to_unit == "fahrenheit":
        result = value * 9 / 5 + 32
        return f"{value}°C = {result:.2f}°F"
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.2f}°C"
    else:
        return f"Error: Unknown units. Use 'celsius' or 'fahrenheit'. Got: {from_unit} -> {to_unit}"


# ── Agent setup ──────────────────────────────────────────────

agent = LlmAgent(
    name="multi_tool_agent",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    instruction="""You are a helpful assistant with access to these tools:
    - calculate: Evaluate math expressions (use Python syntax)
    - to_uppercase: Convert text to uppercase
    - convert_temperature: Convert between Celsius and Fahrenheit
      (use 'celsius' or 'fahrenheit' for the unit parameters)

    For multi-step problems, use tools one at a time and build on results.
    Always include the tool results in your final answer.""",
    tools=[calculate, to_uppercase, convert_temperature],
)

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="multi_tool_app",
    session_service=session_service,
)


# ── Helper function ──────────────────────────────────────────

async def ask_agent(runner, session_service, session_id: str, query: str) -> str:
    """Send a query to the ADK agent and return the response."""
    response_text = ""
    async for event in runner.run_async(
        user_id="student",
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)],
        ),
    ):
        if event.is_final_response():
            response_text = event.content.parts[0].text
    return response_text


# ── Run tests ────────────────────────────────────────────────

async def run_tests():
    """Run all test queries."""
    session = await session_service.create_session(
        app_name="multi_tool_app", user_id="student"
    )

    test_queries = [
        ("Simple calculation", "Calculate 25 * 4"),
        ("Temperature conversion", "Convert 212 degrees Fahrenheit to Celsius"),
        ("Multi-step", "Calculate 15 + 27, then convert that result from Celsius to Fahrenheit"),
        ("Uppercase", "Convert 'hello world' to uppercase"),
    ]

    for label, query in test_queries:
        print(f"\n{'-' * 50}")
        print(f"Test: {label}")
        print(f"Query: {query}")
        try:
            result = await ask_agent(runner, session_service, session.id, query)
            print(f"Agent: {result}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    print(f"Solution 2: Multi-Tool ADK Agent (model: {model})")
    print("=" * 60)
    asyncio.run(run_tests())
