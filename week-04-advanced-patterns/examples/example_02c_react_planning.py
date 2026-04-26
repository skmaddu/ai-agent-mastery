import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 2c: ReAct Planning Pattern in LangGraph
=================================================
Example 01 introduced three planning flavors. Examples 02/03 covered
Plan-Execute. This example implements the REACT pattern.

ReAct = Reasoning + Acting (Yao et al., 2022)

The agent interleaves THINKING and ACTING in a tight loop:
  1. THOUGHT — reason about what to do next ("I need to check dietary info...")
  2. ACTION — call a tool to gather information
  3. OBSERVATION — process the tool result
  4. Repeat until the agent has enough info to answer

Key differences from Plan-Execute:
  Plan-Execute: Create full plan upfront → execute all steps → done
  ReAct:        No upfront plan. Think one step at a time, react to what
                you learn. Each observation informs the next thought.

When to use ReAct vs Plan-Execute:
  ReAct:         1-5 step tasks, exploratory/research, unknown number of steps
  Plan-Execute:  5+ step tasks, well-structured goals, parallelizable sub-tasks

This example shows ReAct with EXPLICIT reasoning traces so you can
see the Think→Act→Observe loop clearly in the output.

Graph:
  START → react_agent → has_tool_calls?
                              |
                     yes: tool_executor → react_agent
                     no:  END (final answer)

Run: python week-04-advanced-patterns/examples/example_02c_react_planning.py
"""

import os
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ==============================================================
# Step 1: LLM Setup
# ==============================================================

def get_llm(temperature=0.3):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )


llm = get_llm(temperature=0.3)


# ==============================================================
# Step 2: Tools for a Research Scenario
# ==============================================================
# ReAct shines in EXPLORATORY tasks where the agent doesn't know
# upfront how many steps it needs. Research is a perfect fit.

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Use this to find facts, statistics, recent developments, or
    general information about any topic.

    Args:
        query: Search query (e.g., 'renewable energy trends 2024')
    """
    search_db = {
        "solar energy cost": (
            "Search results for 'solar energy cost':\n"
            "  1. Solar panel costs have dropped 90% since 2010 (IRENA 2024)\n"
            "  2. Average residential solar: $2.75/watt installed in 2024\n"
            "  3. Solar is now the cheapest electricity source in most countries\n"
            "  4. Utility-scale solar LCOE: $30-50/MWh globally"
        ),
        "wind energy": (
            "Search results for 'wind energy':\n"
            "  1. Global wind capacity reached 906 GW in 2023 (GWEC)\n"
            "  2. Offshore wind growing fastest: 64 GW installed globally\n"
            "  3. Wind LCOE: $25-50/MWh onshore, $50-100/MWh offshore\n"
            "  4. Major growth in US, China, and Europe"
        ),
        "battery storage": (
            "Search results for 'battery storage':\n"
            "  1. Lithium-ion battery costs fell to $139/kWh in 2023 (BloombergNEF)\n"
            "  2. Grid-scale storage deployments up 130% year-over-year\n"
            "  3. Sodium-ion batteries emerging as cheaper alternative\n"
            "  4. Tesla Megapack and BYD dominate utility-scale market"
        ),
        "renewable energy challenge": (
            "Search results for 'renewable energy challenges':\n"
            "  1. Intermittency: solar/wind don't produce 24/7\n"
            "  2. Grid infrastructure needs $4.5 trillion upgrade globally\n"
            "  3. Supply chain constraints for critical minerals (lithium, cobalt)\n"
            "  4. Permitting and land-use conflicts slow deployment"
        ),
        "energy policy": (
            "Search results for 'energy policy':\n"
            "  1. US Inflation Reduction Act: $369B for clean energy\n"
            "  2. EU Green Deal targets 42.5% renewables by 2030\n"
            "  3. China leads in renewable deployment but still burns most coal\n"
            "  4. 130+ countries have net-zero pledges"
        ),
        "renewable market": (
            "Search results for 'renewable energy market':\n"
            "  1. Global renewable investment: $500B+ in 2023\n"
            "  2. Renewables generated 30% of global electricity in 2023\n"
            "  3. Solar + wind additions exceeded fossil fuel additions first time\n"
            "  4. Market projected to reach $2.5 trillion by 2030"
        ),
    }

    query_lower = query.lower()
    for key, result in search_db.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return result

    return (
        f"Search results for '{query}':\n"
        "  1. General information: This is a developing topic\n"
        "  2. Multiple perspectives exist on this subject\n"
        "  3. Try more specific queries for detailed results\n"
        f"  Tip: Try searching for specific aspects like 'cost', 'policy', or 'challenges'"
    )


@tool
def fact_check(claim: str) -> str:
    """Verify a specific claim or statistic.

    Use this to double-check facts, numbers, or assertions before
    including them in your final answer.

    Args:
        claim: The specific claim to verify (e.g., 'solar costs dropped 90% since 2010')
    """
    verified = {
        "solar": "VERIFIED: Solar costs dropped ~89% from 2010-2024 (IRENA data). "
                 "The 90% figure is commonly cited and approximately correct.",
        "wind": "VERIFIED: Global wind capacity ~906 GW as of 2023 (GWEC). "
                "Offshore wind is the fastest-growing segment.",
        "battery": "VERIFIED: Li-ion costs at ~$139/kWh in 2023 (BloombergNEF). "
                   "Down from $1,100/kWh in 2010.",
        "investment": "VERIFIED: Global clean energy investment exceeded $500B in 2023. "
                      "Some sources report up to $620B including all clean tech.",
        "30%": "VERIFIED: Renewables generated approximately 30% of global electricity "
               "in 2023, up from 28% in 2022.",
        "grid": "PARTIALLY VERIFIED: Grid upgrade estimates vary. $4.5 trillion is "
                "one estimate; IEA suggests $600B/year needed through 2030.",
    }

    claim_lower = claim.lower()
    for key, result in verified.items():
        if key in claim_lower:
            return result

    return f"UNVERIFIED: Could not find verification for '{claim}'. Consider this unconfirmed."


@tool
def compare_data(topic_a: str, topic_b: str) -> str:
    """Compare two energy sources or technologies side by side.

    Use this to create comparative analysis between different
    renewable energy options.

    Args:
        topic_a: First topic to compare (e.g., 'solar')
        topic_b: Second topic to compare (e.g., 'wind')
    """
    comparisons = {
        ("solar", "wind"): (
            "Solar vs Wind Comparison:\n"
            "  Cost (LCOE): Solar $30-50/MWh vs Wind $25-50/MWh (onshore)\n"
            "  Capacity factor: Solar 15-25% vs Wind 25-45%\n"
            "  Growth rate: Solar faster (30%+ annual) vs Wind (12% annual)\n"
            "  Land use: Solar needs more area per MW\n"
            "  Intermittency: Solar predictable (daytime) vs Wind less predictable\n"
            "  Maturity: Both mature, solar declining in cost faster"
        ),
        ("solar", "battery"): (
            "Solar + Battery Storage:\n"
            "  Combined LCOE: $50-80/MWh (solar + 4hr battery)\n"
            "  Key synergy: Battery stores daytime solar for evening peak\n"
            "  Trend: Solar-plus-storage now cheaper than new gas plants\n"
            "  Challenge: Battery degradation over 10-15 year lifetime"
        ),
        ("wind", "battery"): (
            "Wind + Battery Storage:\n"
            "  Combination addresses wind intermittency\n"
            "  Offshore wind + storage emerging for baseload replacement\n"
            "  Economics improving but still more expensive than solar+storage\n"
            "  Best suited for high-wind regions (coast, plains)"
        ),
    }

    a, b = topic_a.lower().strip(), topic_b.lower().strip()
    for (ka, kb), result in comparisons.items():
        if (ka in a and kb in b) or (kb in a and ka in b):
            return result

    return f"No direct comparison data available for '{topic_a}' vs '{topic_b}'."


# ==============================================================
# Step 3: State
# ==============================================================

class ReActState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    max_iterations: int


# ==============================================================
# Step 4: Nodes
# ==============================================================

tools = [web_search, fact_check, compare_data]
llm_with_tools = llm.bind_tools(tools)

# The REACT_SYSTEM_PROMPT is what makes this ReAct rather than
# plain tool-calling. It instructs the LLM to show its reasoning
# before each action, creating the Think→Act→Observe pattern.

REACT_SYSTEM_PROMPT = """You are a research agent using the ReAct (Reasoning + Acting) pattern.

For EVERY response, you MUST follow this format:

**Thought:** [Your reasoning about what you know so far and what you need to find out next]
**Action:** [Call a tool to gather information]

OR, if you have enough information to answer:

**Thought:** [Summarize what you've learned and why you can now answer]
**Final Answer:** [Your comprehensive answer based on all gathered evidence]

Rules:
1. ALWAYS start with a Thought before taking any action
2. Each thought should reference what you learned from previous observations
3. Don't repeat searches you've already done
4. After 3-4 tool calls, you should have enough info to give a final answer
5. Your final answer should cite specific facts and numbers from your research

Available tools:
- web_search: Search for information on any topic
- fact_check: Verify specific claims or statistics
- compare_data: Compare two technologies side by side"""


def react_agent_node(state: ReActState) -> dict:
    """The ReAct agent: thinks, then acts (or gives final answer).

    This is fundamentally different from Plan-Execute:
      - No upfront plan — the agent decides what to do based on
        what it's learned so far
      - Each tool call is informed by previous observations
      - The agent can change direction mid-task based on findings
    """
    iteration = state.get("iteration", 0) + 1
    print(f"\n{'='*60}")
    print(f"  REACT AGENT — Iteration {iteration}")
    print(f"{'='*60}")

    # Build messages: system prompt + conversation history
    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + list(state["messages"])

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        # Groq/Llama models sometimes generate malformed tool calls.
        # Fall back to a no-tools call so the agent can still reason.
        print(f"  [WARN] Tool call failed: {str(e)[:80]}. Retrying without tools...")
        response = llm.invoke(messages)

    # Print the agent's reasoning (the "Thought" part)
    if response.content:
        for line in response.content.strip().split("\n"):
            if line.strip():
                print(f"  {line.strip()[:120]}")

    # Print tool calls if any (the "Action" part)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f"\n  ACTION: {tc['name']}({tc['args']})")

    return {
        "messages": [response],
        "iteration": iteration,
    }


def tool_executor_node(state: ReActState) -> dict:
    """Execute tool calls and return observations.

    The tool results become "Observations" in the ReAct loop.
    The agent will see these results and think about what to do next.
    """
    tool_node = ToolNode(tools)
    try:
        result = tool_node.invoke({"messages": state["messages"]})
    except Exception as e:
        # Handle malformed tool calls from Groq/Llama models
        print(f"  [WARN] Tool execution failed: {str(e)[:80]}")
        last_msg = state["messages"][-1]
        error_messages = []
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                error_messages.append(ToolMessage(
                    content="Tool error: could not execute. Please answer without this tool.",
                    tool_call_id=tc["id"],
                ))
        return {"messages": error_messages}

    # Print observations
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"\n  OBSERVATION: {msg.content[:200]}")

    return {"messages": result["messages"]}


# ==============================================================
# Step 5: Routing — Continue or Stop
# ==============================================================

def should_continue(state: ReActState) -> str:
    """Decide: does the agent want to use more tools, or is it done?

    The ReAct loop continues as long as:
      1. The agent made tool calls (wants more information)
      2. We haven't hit the iteration limit

    It stops when:
      1. No tool calls (agent gave a Final Answer)
      2. Iteration limit reached (safety)
    """
    # Safety limit
    if state.get("iteration", 0) >= state.get("max_iterations", 8):
        print(f"\n  [SAFETY] Max iterations reached. Stopping.")
        return "end"

    # Check last message for tool calls
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return "end"


# ==============================================================
# Step 6: Build the ReAct Graph
# ==============================================================
# This is much simpler than Plan-Execute because there's no
# separate planner, executor, or synthesizer. It's just:
#   agent → tools → agent → tools → ... → agent (final answer)
#
#   START → react_agent → should_continue?
#                              |
#                     "tools": tool_executor → react_agent
#                     "end":   END

def build_react_graph():
    graph = StateGraph(ReActState)

    graph.add_node("react_agent", react_agent_node)
    graph.add_node("tool_executor", tool_executor_node)

    graph.set_entry_point("react_agent")

    graph.add_conditional_edges("react_agent", should_continue, {
        "tools": "tool_executor",
        "end": END,
    })

    graph.add_edge("tool_executor", "react_agent")

    return graph.compile()


# ==============================================================
# Step 7: Run
# ==============================================================

def run_react_agent(question: str, max_iterations: int = 8) -> dict:
    app = build_react_graph()
    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
        "max_iterations": max_iterations,
    })
    return result


if __name__ == "__main__":
    print("Example 2c: ReAct Planning Pattern in LangGraph")
    print("=" * 60)
    print("The agent THINKs about what it knows, ACTs to gather info,")
    print("OBSERVEs the result, and repeats until it can answer.")
    print("No upfront plan — pure reactive reasoning.")
    print("=" * 60)

    question = (
        "What is the current state of renewable energy? Compare solar and wind "
        "costs, and explain the role of battery storage. Include specific numbers."
    )

    print(f"\nQuestion: {question}")
    result = run_react_agent(question, max_iterations=8)

    # Extract the final answer from the last AI message
    final_answer = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            final_answer = msg.content
            break
        if hasattr(msg, "content") and msg.content and hasattr(msg, "tool_calls") and not msg.tool_calls:
            final_answer = msg.content
            break

    print(f"\n\n{'#'*60}")
    print("  FINAL ANSWER")
    print(f"{'#'*60}")
    print(f"\n{final_answer}")

    print(f"\n{'='*60}")
    print("ReAct vs Plan-Execute Comparison:")
    print(f"{'='*60}")
    print("  ReAct (this example):")
    print("    - No upfront plan; decides next step based on observations")
    print("    - Natural for exploration: 'search → learn → search more'")
    print("    - Flexible: can change direction based on what it finds")
    print("    - Simpler graph: just agent ⇄ tools")
    print()
    print("  Plan-Execute (example 02):")
    print("    - Creates full plan before executing anything")
    print("    - Better for structured tasks with known sub-steps")
    print("    - Enables parallel execution of independent steps")
    print("    - More complex graph: planner → executor → progress → ...")
    print()
    print("  Rule of thumb:")
    print("    - Use ReAct for research, Q&A, exploratory tasks")
    print("    - Use Plan-Execute for multi-part goals with structure")
    print("    - Use Decompose+Delegate for tasks with parallel sub-goals")
    print("    - See example_02d for the Decompose+Delegate pattern")
    print(f"{'='*60}")
