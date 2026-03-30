"""
Exercise 1: Calculator Agent (LangGraph)
==========================================
Difficulty: Beginner | Time: 1.5 hours

Task:
Build a LangGraph agent with calculator tools that can solve
multi-step math problems. The agent should be able to chain
tool calls (e.g., "add 15 and 27, then multiply by 3").

Instructions:
1. Complete the 3 tool functions: add, multiply, divide
2. Handle division by zero in the divide tool (return error message!)
3. Set up the LLM with provider flexibility (groq/openai)
4. Build the StateGraph with agent_node and ToolNode
5. Add conditional edges for the agent → tools → agent loop
6. Test with all 3 queries below

Hints:
- Look at example_02_langgraph_tool_agent.py for the full pattern
- Tools should return STRINGS (not numbers) — the LLM reads them
- Division by zero should return an error message, NOT crash
- Use bind_tools() to tell the LLM about your tools

Run: python week-02-framework-basics/exercises/exercise_01_calculator_agent.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ── Step 1: Define calculator tools ─────────────────────────
# Each tool needs the @tool decorator, type hints, and a docstring.
# Return strings so the LLM can read the results.

@tool
def add(a: float, b: float) -> str:
    """Add two numbers together. Example: add(15, 27) returns '15 + 27 = 42'"""
    result = a + b
    return f"{a} + {b} = {result}"


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers. Example: multiply(8, 12) returns '8 * 12 = 96'"""
    result = a * b
    return f"{a} * {b} = {result}"


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Example: divide(100, 4) returns '100 / 4 = 25.0'
    Handles division by zero gracefully.
    """
    if b == 0:
        return "Error: Division by zero. Cannot divide by zero."
    result = a / b
    return f"{a} / {b} = {result}"



# ── Step 2: Set up the LLM ─────────────────────────────────
# Initialize the LLM based on the LLM_PROVIDER env variable.
# Default to groq, fallback to openai

tools = [add, multiply, divide]

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

llm_with_tools = llm.bind_tools(tools)


# ── Step 3: Define the state ────────────────────────────────
# TypedDict for the graph state. Messages are appended by add_messages.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int


# ── Step 4: Define the agent node ──────────────────────────
# The agent node asks the LLM for the next action (or final answer).

def agent_node(state: AgentState) -> dict:
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            if attempt < 2 and ("tool_use_failed" in str(e) or "malformed" in str(e).lower()):
                print(f"  [Retry {attempt + 1}/3] Malformed tool call, retrying...")
                continue
            raise


# ── Step 5: Define the routing function ─────────────────────
MAX_TOOL_CALLS = 5

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        print(f"  [Safety] Max tool calls ({MAX_TOOL_CALLS}) reached. Ending.")
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ── Step 6: Build the graph ─────────────────────────────────
tool_executor = ToolNode(tools)

def tools_node(state: AgentState) -> dict:
    result = tool_executor.invoke(state)
    new_count = state.get("tool_call_count", 0) + len(state["messages"][-1].tool_calls)
    return {
        "messages": result["messages"],
        "tool_call_count": new_count,
    }


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "agent")
app = graph.compile()


# ── Test your implementation ────────────────────────────────

if __name__ == "__main__":
    print("Exercise 1: Calculator Agent")
    print("=" * 50)

    # Test 1: Simple addition
    print("\nTest 1: What is 15 + 27?")
    result = app.invoke({
        "messages": [HumanMessage(content="What is 15 + 27?")],
        "tool_call_count": 0,
    })
    print(f"Agent: {result['messages'][-1].content}")

    # Test 2: Multi-step calculation (requires chaining)
    print("\nTest 2: Multiply 8 by 12, then add 5 to the result")
    result = app.invoke({
        "messages": [HumanMessage(content="Multiply 8 by 12, then add 5 to the result")],
        "tool_call_count": 0,
    })
    print(f"Agent: {result['messages'][-1].content}")

    # Test 3: Division by zero (should handle gracefully!)
    print("\nTest 3: Divide 100 by 0")
    result = app.invoke({
        "messages": [HumanMessage(content="Divide 100 by 0")],
        "tool_call_count": 0,
    })
    print(f"Agent: {result['messages'][-1].content}")

    print("\n(Uncomment the test code above after implementing!)")
