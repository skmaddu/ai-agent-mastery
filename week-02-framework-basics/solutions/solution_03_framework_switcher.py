"""
Solution: Exercise 3 — Framework Switcher
===========================================
Same tools, same query, switchable between LangGraph and ADK.
Demonstrates framework-agnostic tool design and side-by-side comparison.

Run: python week-02-framework-basics/solutions/solution_03_framework_switcher.py
"""

import asyncio
import logging
import os
import time
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════
# Shared tool logic (framework-agnostic)
# ══════════════════════════════════════════════════════════════

def calculate_logic(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero"
    except Exception as e:
        return f"Error: {e}"


def reverse_text_logic(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


# ══════════════════════════════════════════════════════════════
# LangGraph builder
# ══════════════════════════════════════════════════════════════

def build_langgraph_agent():
    """Build a LangGraph agent using the shared tools."""
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    # Wrap shared logic with @tool
    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '15 * 7', '2 ** 10'"""
        return calculate_logic(expression)

    @tool
    def reverse_text(text: str) -> str:
        """Reverse a string. Example: reverse_text('hello') returns 'olleh'"""
        return reverse_text_logic(text)

    tools = [calculate, reverse_text]

    # LLM setup
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    llm_with_tools = llm.bind_tools(tools)

    MAX_ITERATIONS = 10  # Safety valve: prevent infinite loops

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        iteration: int

    def agent_node(state):
        for attempt in range(3):
            try:
                return {
                    "messages": [llm_with_tools.invoke(state["messages"])],
                    "iteration": state.get("iteration", 0) + 1,
                }
            except Exception:
                if attempt < 2:
                    continue
                raise

    def should_continue(state):
        # Safety valve: stop if too many iterations
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "end"
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ══════════════════════════════════════════════════════════════
# ADK builder
# ══════════════════════════════════════════════════════════════

def build_adk_agent():
    """Build an ADK agent using the shared tools."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    # ADK tools — call shared logic directly
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '15 * 7', '2 ** 10'"""
        return calculate_logic(expression)

    def reverse_text(text: str) -> str:
        """Reverse a string. Example: reverse_text('hello') returns 'olleh'"""
        return reverse_text_logic(text)

    agent = LlmAgent(
        name="switcher_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="You have calculate and reverse_text tools. Use them to answer questions.",
        tools=[calculate, reverse_text],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="switcher", session_service=session_service)
    return runner, session_service


# ══════════════════════════════════════════════════════════════
# Framework dispatcher
# ══════════════════════════════════════════════════════════════

def run_with_framework(framework: str, query: str) -> dict:
    """Run a query using the specified framework.

    Args:
        framework: "langgraph" or "adk"
        query: The user's question

    Returns:
        Dict with result, time_seconds, and framework name
    """
    try:
        if framework == "langgraph":
            from langchain_core.messages import HumanMessage
            app = build_langgraph_agent()
            start = time.time()
            result = app.invoke({"messages": [HumanMessage(content=query)], "iteration": 0})
            elapsed = time.time() - start
            return {
                "result": result["messages"][-1].content,
                "time_seconds": elapsed,
                "framework": "LangGraph",
            }

        elif framework == "adk":
            from google.genai import types

            async def _run_adk():
                runner, session_service = build_adk_agent()
                session = await session_service.create_session(
                    app_name="switcher", user_id="user1"
                )
                answer = ""
                async for event in runner.run_async(
                    user_id="user1",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=query)]
                    ),
                ):
                    if event.is_final_response():
                        answer = event.content.parts[0].text
                return answer

            start = time.time()
            answer = asyncio.run(_run_adk())
            elapsed = time.time() - start
            return {
                "result": answer,
                "time_seconds": elapsed,
                "framework": "ADK",
            }
        else:
            return {
                "result": f"Error: Unknown framework '{framework}'",
                "time_seconds": 0,
                "framework": framework,
            }

    except Exception as e:
        return {
            "result": f"Error: {type(e).__name__}: {e}",
            "time_seconds": 0,
            "framework": framework,
        }


# ══════════════════════════════════════════════════════════════
# Comparison
# ══════════════════════════════════════════════════════════════

def compare_frameworks(query: str):
    """Run the same query on both frameworks and print comparison."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}")

    # Run LangGraph
    print("\n  Running LangGraph...", end=" ")
    lg = run_with_framework("langgraph", query)
    print(f"Done ({lg['time_seconds']:.2f}s)")

    # Run ADK
    print("  Running ADK...", end=" ")
    adk = run_with_framework("adk", query)
    print(f"Done ({adk['time_seconds']:.2f}s)")

    # Print results
    print(f"\n  {'Framework':<12} {'Time':>8}  Result")
    print(f"  {'-' * 12} {'-' * 8}  {'-' * 35}")
    print(f"  {'LangGraph':<12} {lg['time_seconds']:>7.2f}s  {lg['result'][:60]}")
    print(f"  {'ADK':<12} {adk['time_seconds']:>7.2f}s  {adk['result'][:60]}")


# ══════════════════════════════════════════════════════════════
# Run tests
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Solution 3: Framework Switcher")
    print("=" * 60)

    compare_frameworks("What is 15 * 7?")
    compare_frameworks("Reverse the word 'framework'")
    compare_frameworks("Calculate 2 to the power of 10, then reverse that number as a string")

    print(f"\n{'=' * 60}")
    print("OBSERVATIONS:")
    print("  - Both frameworks use the SAME tool logic -- no duplication")
    print("  -LangGraph needs more setup code but gives more control")
    print("  -ADK is simpler but limited to Gemini models")
    print("  -Execution times may vary based on API latency")
    print(f"{'=' * 60}")
