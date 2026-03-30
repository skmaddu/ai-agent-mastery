"""
Exercise 2: Multi-Tool ADK Agent
==================================
Difficulty: Beginner-Intermediate | Time: 2 hours

Task:
Build an ADK agent with 3 tools that can handle multi-step queries.
The agent should be able to chain tool calls to answer complex
questions like "Convert 100°F to Celsius, then round to nearest integer".

Instructions:
1. Complete the 3 tool functions: calculate, to_uppercase, convert_temperature
2. Create an LlmAgent with clear instructions mentioning each tool
3. Set up Runner and InMemorySessionService
4. Create the ask_agent helper function
5. Test with all 3 queries below
6. Bonus: Show session persistence across turns

Hints:
- ADK tools are plain functions — NO @tool decorator needed
- ADK reads your type hints and docstring — make them clear!
- For temperature conversion: C = (F - 32) * 5/9, F = C * 9/5 + 32
- Look at example_04_adk_tool_agent.py for the pattern

Requires: GOOGLE_API_KEY in your environment.

Run: python week-02-framework-basics/exercises/exercise_02_multi_tool_adk_agent.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ── Step 1: Define tools as plain Python functions ──────────
# Remember: ADK tools need type hints AND docstrings.
# ADK uses these to generate the tool schema for the LLM.

def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Supports: +, -, *, /, ** (power), parentheses.
    Examples: '25 * 4', '(100 - 32) * 5 / 9', '2 ** 8'
    """
    # TODO: Implement safe math evaluation
    # 1. Validate that expression only contains safe characters (digits, operators, spaces, parens, dots)
    # 2. Use eval() to compute the result
    # 3. Return a string like "25 * 4 = 100"
    # 4. Handle errors (return error message, don't crash!)
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression. Use only numbers and +-*/()"
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
    # TODO: Convert text to uppercase and return it
    return text.upper()


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.
    from_unit and to_unit should be 'celsius' or 'fahrenheit'.

    Formulas:
    - Celsius to Fahrenheit: F = C * 9/5 + 32
    - Fahrenheit to Celsius: C = (F - 32) * 5/9
    """
    # TODO: Implement temperature conversion
    # 1. Normalize from_unit and to_unit to lowercase
    # 2. If from_unit == to_unit, return the value as-is
    # 3. Convert between celsius and fahrenheit
    # 4. Return a string like "100°F = 37.78°C"
    # 5. Handle invalid units (return error message)
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit == to_unit:
        return f"{value}°{from_unit[0].upper()} = {value}°{to_unit[0].upper()}"
    
    if from_unit == "celsius" and to_unit == "fahrenheit":
        result = value * 9/5 + 32
        return f"{value}°C = {result:.2f}°F"
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        result = (value - 32) * 5/9
        return f"{value}°F = {result:.2f}°C"
    else:
        return f"Error: Invalid units. Use 'celsius' or 'fahrenheit'"


# ── Step 2: Create the ADK agent ───────────────────────────
# TODO: Create an LlmAgent with:
#   - name: "multi_tool_agent"
#   - model: from GOOGLE_MODEL env var (default "gemini-2.0-flash")
#   - instruction: Tell the agent about each tool and when to use it
#   - tools: list of your 3 tool functions

agent = LlmAgent(
    name="multi_tool_agent",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    instruction="""You are a helpful assistant with access to three tools:
    1. calculate: For arithmetic operations and math expressions
    2. to_uppercase: For converting text to uppercase
    3. convert_temperature: For converting between Celsius and Fahrenheit
    
    Always use the appropriate tool when the user asks for calculations,
    text conversion, or temperature conversion. Show the tool results clearly
    in your final answer.""",
    tools=[calculate, to_uppercase, convert_temperature],
)


# ── Step 3: Set up Runner and SessionService ────────────────
# TODO: Create InMemorySessionService and Runner

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="multi_tool_app",
    session_service=session_service,
)


# ── Step 4: Create the ask_agent helper ─────────────────────
# TODO: Implement this async function that sends a query and returns the response

async def ask_agent(runner, session_service, session_id: str, query: str) -> str:
    """Send a query to the ADK agent and return the response.

    Args:
        runner: The ADK Runner instance
        session_service: The session service
        session_id: The session ID (for multi-turn conversations)
        query: The user's question

    Returns:
        The agent's response as a string
    """
    # TODO: Implement this function
    # 1. Call runner.run_async() with the query
    # 2. Iterate over events with async for
    # 3. Return the text from the final response event
    async for event in runner.run_async(
        user_id="student",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
    return ""


# ── Test your implementation ────────────────────────────────

async def run_tests():
    """Run test queries against the agent."""
    # TODO: Uncomment after implementing

    session = await session_service.create_session(
        app_name="multi_tool_app", user_id="student"
    )

    # Test 1: Simple calculation
    print("Test 1: Calculate 25 * 4")
    result = await ask_agent(runner, session_service, session.id, "Calculate 25 * 4")
    print(f"Agent: {result}")

    # Test 2: Temperature conversion
    print("\nTest 2: Convert 212 degrees Fahrenheit to Celsius")
    result = await ask_agent(runner, session_service, session.id,
        "Convert 212 degrees Fahrenheit to Celsius")
    print(f"Agent: {result}")

    # Test 3: Multi-step query (chains multiple tools)
    print("\nTest 3: Calculate 15 + 27, then convert that from Celsius to Fahrenheit")
    result = await ask_agent(runner, session_service, session.id,
        "Calculate 15 + 27, then convert that result from Celsius to Fahrenheit")
    print(f"Agent: {result}")

    # Bonus Test: Uppercase
    print("\nBonus: Convert 'hello world' to uppercase")
    result = await ask_agent(runner, session_service, session.id,
        "Convert 'hello world' to uppercase")
    print(f"Agent: {result}")


if __name__ == "__main__":
    print("Exercise 2: Multi-Tool ADK Agent")
    print("=" * 50)
    asyncio.run(run_tests())
