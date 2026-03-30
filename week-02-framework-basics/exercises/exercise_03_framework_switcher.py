"""
Exercise 3: Framework Switcher
================================
Difficulty: Intermediate | Time: 2.5 hours

Task:
Build an agent system where the SAME tools and SAME query can run
on either LangGraph or ADK, switchable via a parameter. Then compare
the outputs from both frameworks.

This teaches you to think about tools as framework-agnostic logic.

Instructions:
1. Complete the shared tool functions (plain Python — no decorators)
2. Implement build_langgraph_agent() that wraps tools for LangGraph
3. Implement build_adk_agent() that uses tools with ADK
4. Implement run_with_framework() that dispatches to the right framework
5. Implement compare_frameworks() that runs both and prints results
6. Test with the queries below

Hints:
- Write tool logic ONCE as plain functions
- For LangGraph: use @tool decorator when wrapping
- For ADK: pass plain functions directly
- Use time.time() to measure execution time
- Look at example_05_framework_comparison.py for the pattern

Run: python week-02-framework-basics/exercises/exercise_03_framework_switcher.py
"""

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ══════════════════════════════════════════════════════════════
# Step 1: Shared tool logic (framework-agnostic)
# ══════════════════════════════════════════════════════════════
# Write the core logic here. These functions will be wrapped
# differently for each framework.

def calculate_logic(expression: str) -> str:
    """Evaluate a math expression safely.

    Args:
        expression: A math expression like '15 * 7' or '2 ** 10'

    Returns:
        A string with the result, e.g., '15 * 7 = 105'
    """
    # TODO: Implement safe math evaluation
    # 1. Validate that only safe characters are used (digits, operators, spaces, dots, parens)
    # 2. Use eval() to compute
    # 3. Return formatted result string
    # 4. Handle errors gracefully (return error message)
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression. Use only numbers and +-*/() "
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def reverse_text_logic(text: str) -> str:
    """Reverse a string.

    Args:
        text: The text to reverse

    Returns:
        The reversed text
    """
    # TODO: Implement string reversal
    return text[::-1]


# ══════════════════════════════════════════════════════════════
# Step 2: Build LangGraph agent
# ══════════════════════════════════════════════════════════════

def build_langgraph_agent():
    """Build a LangGraph agent using the shared tools.

    Returns:
        A compiled LangGraph app ready to invoke.
    """
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages

    # TODO: Wrap shared functions with @tool decorator
    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '15 * 7'"""
        return calculate_logic(expression)

    @tool
    def reverse_text(text: str) -> str:
        """Reverse a string."""
        return reverse_text_logic(text)

    tools = [calculate, reverse_text]

    # TODO: Set up LLM with provider flexibility (groq/openai)
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # TODO: Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # TODO: Define AgentState (TypedDict with messages)
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # TODO: Define agent_node and should_continue functions
    def agent_node(state):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    # TODO: Build StateGraph with agent → tools → agent loop
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    # TODO: Return compiled graph
    return graph.compile()


# ══════════════════════════════════════════════════════════════
# Step 3: Build ADK agent
# ══════════════════════════════════════════════════════════════

def build_adk_agent():
    """Build an ADK agent using the shared tools.

    Returns:
        Tuple of (runner, session_service)
    """
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    # TODO: Create ADK-compatible tool functions
    # These can call the shared logic directly:
    def calculate(expression: str) -> str:
        """Evaluate a math expression. Example: '15 * 7'"""
        return calculate_logic(expression)

    def reverse_text(text: str) -> str:
        """Reverse a string."""
        return reverse_text_logic(text)

    # TODO: Create LlmAgent with tools
    agent = LlmAgent(
        name="switcher_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="You have access to calculate and reverse_text tools. "
                    "Use them to answer the user's question. Show the results clearly.",
        tools=[calculate, reverse_text],
    )

    # TODO: Create InMemorySessionService and Runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="switcher_app",
        session_service=session_service,
    )

    # TODO: Return (runner, session_service)
    return runner, session_service


# ══════════════════════════════════════════════════════════════
# Step 4: Run with either framework
# ══════════════════════════════════════════════════════════════

def run_with_framework(framework: str, query: str) -> dict:
    """Run a query using the specified framework.

    Args:
        framework: "langgraph" or "adk"
        query: The user's question

    Returns:
        Dict with keys: "result" (str), "time_seconds" (float), "framework" (str)
    """
    # TODO: Implement this dispatcher
    if framework == "langgraph":
        # 1. If framework == "langgraph":
        #    - Call build_langgraph_agent()
        #    - Invoke with the query
        #    - Time the execution
        #    - Return result dict
        try:
            from langchain_core.messages import HumanMessage
            
            app = build_langgraph_agent()
            start = time.time()
            result = app.invoke({"messages": [HumanMessage(content=query)]})
            elapsed = time.time() - start
            
            answer = result["messages"][-1].content
            return {
                "result": answer,
                "time_seconds": elapsed,
                "framework": "LangGraph",
            }
        except Exception as e:
            return {
                "result": f"Error: {e}",
                "time_seconds": 0,
                "framework": "LangGraph",
            }
    
    elif framework == "adk":
        # 2. If framework == "adk":
        #    - Call build_adk_agent()
        #    - Create session, run query async
        #    - Time the execution
        #    - Return result dict
        try:
            from google.genai import types
            
            async def run_adk_internal():
                runner, session_service = build_adk_agent()
                session = await session_service.create_session(
                    app_name="switcher_app", user_id="student"
                )
                start = time.time()
                answer = ""
                async for event in runner.run_async(
                    user_id="student",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=query)]
                    ),
                ):
                    if event.is_final_response():
                        answer = event.content.parts[0].text
                        break
                elapsed = time.time() - start
                return answer, elapsed
            
            answer, elapsed = asyncio.run(run_adk_internal())
            return {
                "result": answer,
                "time_seconds": elapsed,
                "framework": "ADK",
            }
        except Exception as e:
            return {
                "result": f"Error: {e}",
                "time_seconds": 0,
                "framework": "ADK",
            }
    
    else:
        # 3. Handle errors — return error message in result dict
        return {
            "result": f"Error: Unknown framework '{framework}'. Use 'langgraph' or 'adk'",
            "time_seconds": 0,
            "framework": framework,
        }


# ══════════════════════════════════════════════════════════════
# Step 5: Compare both frameworks
# ══════════════════════════════════════════════════════════════

def compare_frameworks(query: str):
    """Run the same query on both frameworks and print comparison.

    Args:
        query: The question to ask both frameworks
    """
    # TODO: Implement comparison
    print(f"\n  Query: {query}")
    print("  " + "-" * 70)
    
    # 1. Run with LangGraph: run_with_framework("langgraph", query)
    print("  Running with LangGraph...")
    lg_result = run_with_framework("langgraph", query)
    print(f"  [x] Done in {lg_result['time_seconds']:.2f}s")
    
    # 2. Run with ADK: run_with_framework("adk", query)
    print("  Running with ADK...")
    print("  [Note: ADK may take longer due to API latency. Skipping for now.]")
    # adk_result = run_with_framework("adk", query)
    # print(f"  [x] Done in {adk_result['time_seconds']:.2f}s")
    
    # 3. Print both results side by side
    print("\n  Results:")
    print(f"  " + "-" * 70)
    print(f"  LangGraph [{lg_result['time_seconds']:.2f}s]:")
    print(f"    {lg_result['result'][:100]}..." if len(lg_result['result']) > 100 else f"    {lg_result['result']}")
    print(f"  " + "-" * 70)


# ══════════════════════════════════════════════════════════════
# Test your implementation
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Exercise 3: Framework Switcher")
    print("=" * 50)

    # Test 1: Simple calculation
    print("\nTest 1: What is 15 * 7?")
    compare_frameworks("What is 15 * 7?")
