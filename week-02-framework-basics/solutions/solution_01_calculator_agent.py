"""
Solution: Exercise 1 — Calculator Agent (LangGraph)
=====================================================
A LangGraph agent with add, multiply, and divide tools.
Handles division by zero gracefully and can chain tool calls
for multi-step math problems.

Run: python week-02-framework-basics/solutions/solution_01_calculator_agent.py
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


# ── Tools ────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> str:
    """Add two numbers together. Example: add(15, 27) returns '15.0 + 27.0 = 42.0'"""
    result = a + b
    return f"{a} + {b} = {result}"


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers. Example: multiply(8, 12) returns '8.0 * 12.0 = 96.0'"""
    result = a * b
    return f"{a} * {b} = {result}"


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Handles division by zero gracefully."""
    if b == 0:
        return f"Error: Cannot divide {a} by zero. Division by zero is undefined."
    result = a / b
    return f"{a} / {b} = {result}"


# ── LLM setup ───────────────────────────────────────────────

tools = [add, multiply, divide]

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

llm_with_tools = llm.bind_tools(tools)


# ── State ────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Nodes and routing ────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Call the LLM with retry logic for malformed tool calls."""
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            if attempt < 2:
                continue
            raise


def should_continue(state: AgentState) -> str:
    """Route to tools if LLM made a tool call, else end."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


# ── Build graph ──────────────────────────────────────────────

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()


# ── Run tests ────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Solution 1: Calculator Agent (provider: {provider})")
    print("=" * 60)

    test_queries = [
        ("Simple addition", "What is 15 + 27?"),
        ("Multi-step", "Multiply 8 by 12, then add 5 to the result"),
        ("Division by zero", "Divide 100 by 0"),
        ("Normal division", "Divide 100 by 4"),
    ]

    for label, query in test_queries:
        print(f"\n{'-' * 50}")
        print(f"Test: {label}")
        print(f"Query: {query}")
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=query)],
            })
            print(f"Agent: {result['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
