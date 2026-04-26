import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5: Tool Use Pattern in LangGraph -- Multi-Tool Agent
=============================================================
Building on Example 4's tool design principles, this example creates
a LangGraph agent with MULTIPLE tools that it can chain together
to answer complex questions.

The agent has 3 tools:
  - search_web: Find information online (simulated)
  - calculate: Do math
  - get_word_info: Get definitions and synonyms

The key learning: the agent decides ON ITS OWN which tools to call,
in what order, and how to combine results. This is the Tool Use pattern.

Pattern Flow:
  User Question -> Agent (LLM) -> [Tool Call -> Result -> Agent]* -> Final Answer

Run: python week-03-basic-patterns/examples/example_05_tool_use_langgraph.py
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


# ==============================================================
# Step 1: Define Multiple Tools (Following the 5 Rules)
# ==============================================================

@tool
def search_web(query: str) -> str:
    """Search the web for current information about any topic.

    Use this when you need facts, statistics, news, or information
    that might not be in your training data. Always search before
    making factual claims.

    Args:
        query: Search query (e.g., 'population of Japan 2026')
    """
    # Simulated search results for teaching purposes
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

    # Find matching results
    for keyword, results in results_db.items():
        if keyword in query.lower():
            return "\n".join(f"  {i+1}. {r}" for i, r in enumerate(results))

    return f"No results found for '{query}'. Try different keywords."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this for any arithmetic, percentage calculations, or numerical
    comparisons. Supports: +, -, *, /, ** (power), parentheses.

    Args:
        expression: Math expression (e.g., '300 * 0.35', '8.1 - 1.4')
    """
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return f"Error: Invalid characters in '{expression}'. Use only numbers and +,-,*,/,**,()"

    try:
        result = eval(expression)  # Safe here because we validated characters
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error: Could not evaluate '{expression}': {e}"


@tool
def get_word_info(word: str) -> str:
    """Get the definition, synonyms, and usage examples for a word.

    Use this when the user asks about word meanings, synonyms,
    antonyms, or how to use a word in context.

    Args:
        word: The word to look up (e.g., 'resilient', 'ubiquitous')
    """
    # Simulated dictionary
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
            "example": "The ephemeral beauty of cherry blossoms draws millions of visitors.",
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
    return f"Word '{word}' not found in dictionary. Try: resilient, ubiquitous, ephemeral."


# ==============================================================
# Step 2: Set Up the LLM with Tools Bound
# ==============================================================

tools = [search_web, calculate, get_word_info]

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "groq":
    from langchain_groq import ChatGroq
    # NOTE: llama-3.3-70b-versatile has broken tool calling via langchain_groq
    # (generates malformed XML-style tool calls). qwen/qwen3-32b works reliably.
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,  # Low temperature for consistent tool selection
    )
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

# bind_tools tells the LLM about our tools — it sees their schemas
llm_with_tools = llm.bind_tools(tools)


# ==============================================================
# Step 3: Build the Agent Graph
# ==============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    tool_call_count: int                      # Safety counter


# Maximum tool calls before forcing a response (prevents infinite loops)
MAX_TOOL_CALLS = 8


def agent_node(state: AgentState) -> dict:
    """Call the LLM — it decides whether to use tools or respond directly."""
    response = llm_with_tools.invoke(state["messages"])

    # Count tool calls for safety tracking
    new_count = state.get("tool_call_count", 0)
    if hasattr(response, "tool_calls") and response.tool_calls:
        new_count += len(response.tool_calls)
        # Print which tools the agent chose (for learning)
        for tc in response.tool_calls:
            print(f"  [TOOL] Tool call: {tc['name']}({tc['args']})")

    return {
        "messages": [response],
        "tool_call_count": new_count,
    }


def should_continue(state: AgentState) -> str:
    """Route: call tools, or give final answer?

    The LLM signals its decision through tool_calls:
      - Has tool_calls -> route to "tools" node
      - No tool_calls  -> LLM is done, route to END
      - Too many calls -> safety stop
    """
    # Safety valve: prevent runaway tool loops
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        print(f"  [WARN] Safety limit: {MAX_TOOL_CALLS} tool calls reached")
        return "end"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# Build the graph: agent -> [tools -> agent]* -> END
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()


# ==============================================================
# Step 4: Run the Agent with Different Questions
# ==============================================================

def ask(question: str):
    """Ask the agent a question and display the process."""
    print(f"\n{'-'*60}")
    print(f"Question: {question}")
    print(f"{'-'*60}")

    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "tool_call_count": 0,
    })

    answer = result["messages"][-1].content
    tool_count = result.get("tool_call_count", 0)
    print(f"\n  Answer ({tool_count} tool call(s)):")
    print(f"  {answer[:500]}")


if __name__ == "__main__":
    print("Example 5: Multi-Tool Agent in LangGraph")
    print("=" * 60)

    # Test 1: Single tool (search)
    ask("What is the current world population?")

    # Test 2: Single tool (calculate)
    ask("If renewable energy is 35% of global electricity and total generation is 29,000 TWh, how many TWh is renewable?")

    # Test 3: Multi-tool chaining (search + calculate)
    ask("Search for the AI market size projection, then calculate what 15% of that would be.")

    # Test 4: Different tool (word info)
    ask("What does 'ephemeral' mean? Give me synonyms too.")

    # Test 5: No tool needed (direct answer)
    ask("What is 2 + 2?")

    print(f"\n{'='*60}")
    print("Key Observations:")
    print("  1. The agent CHOOSES which tool(s) to call based on the question")
    print("  2. It can CHAIN tools — search first, then calculate")
    print("  3. For simple questions, it may skip tools entirely")
    print("  4. The tool_call_count safety limit prevents infinite loops")
    print(f"{'='*60}")
