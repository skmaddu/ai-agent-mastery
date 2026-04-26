import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 2: Plan-Execute Pattern in LangGraph -- Real LLM Planning
==================================================================
Building on Example 1's planning concepts, this example implements a full
Plan-Execute-Replan agent using LangGraph with a real LLM.

The agent receives a complex goal (plan a dinner party) and:
  1. PLANS — The LLM decomposes the goal into 3-5 ordered steps
  2. EXECUTES — Each step is handled by the LLM with access to tools
  3. CHECKS PROGRESS — Advances to the next step or finishes
  4. SYNTHESIZES — Combines all step results into a final answer

Graph:
  START -> planner -> executor -> progress_check
                        ^              |
                        |              | (steps remain)
                        +──────────────+
                                       | (all done)
                                       v
                                    synthesize -> END

Key Concepts (Section 1 of the Research Bible):
  - Plan-Execute separation: planning LLM call uses temperature=0 for
    deterministic decomposition; executor uses temperature=0.7 for creative
    tool use
  - Task decomposition: the planner breaks the goal into verifiable sub-steps
  - Safety limits: max_iterations prevents infinite loops
  - Tool-augmented execution: the executor can call simulated tools

Run: python week-04-advanced-patterns/examples/example_02_planning_langgraph.py
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ==============================================================
# Step 1: Set Up the LLM
# ==============================================================

def get_llm(temperature=0.7):
    """Create LLM based on provider setting."""
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


# Two LLM instances: deterministic planner, creative executor
planner_llm = get_llm(temperature=0)
executor_llm = get_llm(temperature=0.7)


# ==============================================================
# Step 2: Define Simulated Tools
# ==============================================================
# These tools return hardcoded data so the example runs without
# external APIs. The LLM decides WHICH tool to call and HOW,
# which is the interesting part.

@tool
def recipe_search(cuisine_type: str) -> str:
    """Search a recipe database for dishes matching a cuisine type.

    Use this to find recipe ideas, ingredient lists, and serving sizes
    for a specific type of cuisine.

    Args:
        cuisine_type: Type of cuisine to search for (e.g., 'italian', 'mexican', 'asian')
    """
    recipes_db = {
        "italian": (
            "Found 3 Italian recipes:\n"
            "  1. Pasta Primavera (serves 8) - penne, seasonal vegetables, olive oil, parmesan. "
            "Prep: 15min, Cook: 20min. Can be made gluten-free with GF pasta.\n"
            "  2. Chicken Piccata (serves 8) - chicken breast, capers, lemon, butter. "
            "Prep: 10min, Cook: 25min. Naturally gluten-free if served without breading.\n"
            "  3. Tiramisu (serves 10) - mascarpone, espresso, ladyfingers, cocoa. "
            "Prep: 30min, Chill: 4hrs. Contains gluten in ladyfingers."
        ),
        "mexican": (
            "Found 3 Mexican recipes:\n"
            "  1. Chicken Tacos (serves 8) - corn tortillas, chicken, salsa, avocado. "
            "Prep: 20min, Cook: 15min. Corn tortillas are naturally gluten-free.\n"
            "  2. Black Bean Soup (serves 8) - black beans, cumin, tomatoes, cilantro. "
            "Prep: 10min, Cook: 30min. Naturally gluten-free and vegan.\n"
            "  3. Churros (serves 8) - flour, sugar, cinnamon, chocolate sauce. "
            "Prep: 15min, Cook: 10min. Contains gluten."
        ),
        "asian": (
            "Found 3 Asian recipes:\n"
            "  1. Chicken Stir-Fry (serves 8) - chicken, broccoli, soy sauce, rice. "
            "Prep: 15min, Cook: 10min. Use tamari for gluten-free version.\n"
            "  2. Vegetable Pad Thai (serves 8) - rice noodles, tofu, peanuts, lime. "
            "Prep: 20min, Cook: 10min. Rice noodles are gluten-free.\n"
            "  3. Mango Sticky Rice (serves 8) - glutinous rice, coconut milk, mango. "
            "Prep: 10min, Cook: 30min. Naturally gluten-free despite the name."
        ),
    }

    # Search for matching cuisine
    for key, recipes in recipes_db.items():
        if key in cuisine_type.lower():
            return recipes

    return (
        f"No recipes found for '{cuisine_type}'. "
        f"Available cuisines: italian, mexican, asian. Try one of these."
    )


@tool
def nutrition_check(meal_name: str) -> str:
    """Check nutrition information for a meal or dish.

    Use this to verify calorie counts, allergen info, and dietary
    compatibility for specific dishes.

    Args:
        meal_name: Name of the meal to check (e.g., 'pasta primavera', 'chicken tacos')
    """
    nutrition_db = {
        "pasta": "Pasta Primavera: 380 cal/serving, 12g protein, 52g carbs, 14g fat. "
                 "Allergens: gluten (wheat pasta), dairy (parmesan). GF option available.",
        "chicken piccata": "Chicken Piccata: 320 cal/serving, 35g protein, 8g carbs, 16g fat. "
                          "Allergens: dairy (butter). Gluten-free if unbreaded.",
        "tiramisu": "Tiramisu: 450 cal/serving, 7g protein, 38g carbs, 28g fat. "
                    "Allergens: gluten (ladyfingers), dairy, eggs, caffeine.",
        "tacos": "Chicken Tacos: 290 cal/serving, 24g protein, 28g carbs, 10g fat. "
                 "Allergens: none with corn tortillas. Naturally gluten-free.",
        "black bean": "Black Bean Soup: 220 cal/serving, 14g protein, 40g carbs, 2g fat. "
                      "Allergens: none. Vegan and gluten-free.",
        "stir-fry": "Chicken Stir-Fry: 340 cal/serving, 28g protein, 32g carbs, 12g fat. "
                    "Allergens: soy. Use tamari for gluten-free.",
        "pad thai": "Vegetable Pad Thai: 310 cal/serving, 12g protein, 48g carbs, 8g fat. "
                    "Allergens: peanuts, soy. Rice noodles are gluten-free.",
    }

    for key, info in nutrition_db.items():
        if key in meal_name.lower():
            return info

    return f"No nutrition data found for '{meal_name}'. Try a more common dish name."


@tool
def budget_calculator(items: str) -> str:
    """Calculate the total cost of a list of grocery items.

    Use this to estimate the budget for a meal or shopping list.
    Provide items as a comma-separated list.

    Args:
        items: Comma-separated list of items (e.g., 'chicken breast, rice, vegetables, spices')
    """
    price_db = {
        "chicken": 12.00, "beef": 18.00, "tofu": 4.00, "fish": 15.00,
        "pasta": 3.00, "rice": 4.00, "noodle": 3.50, "bread": 3.50,
        "vegetable": 8.00, "broccoli": 3.00, "tomato": 4.00, "avocado": 5.00,
        "cheese": 6.00, "parmesan": 7.00, "butter": 4.50, "cream": 5.00,
        "mascarpone": 6.00, "coconut": 3.00, "mango": 4.00,
        "olive oil": 8.00, "oil": 6.00, "spice": 5.00, "cumin": 2.50,
        "soy sauce": 3.00, "tamari": 4.00, "salsa": 4.00,
        "bean": 3.00, "peanut": 3.50, "lime": 1.50, "lemon": 1.50,
        "caper": 4.00, "cilantro": 1.50, "cocoa": 3.00, "espresso": 5.00,
        "sugar": 2.50, "cinnamon": 2.00, "chocolate": 4.00, "flour": 2.50,
        "tortilla": 4.00, "egg": 4.00, "ladyfinger": 5.00,
    }

    item_list = [item.strip().lower() for item in items.split(",")]
    total = 0.0
    breakdown = []

    for item in item_list:
        matched = False
        for key, price in price_db.items():
            if key in item:
                total += price
                breakdown.append(f"  {item}: ${price:.2f}")
                matched = True
                break
        if not matched:
            estimate = 5.00
            total += estimate
            breakdown.append(f"  {item}: ~${estimate:.2f} (estimated)")

    result = "Budget Breakdown:\n" + "\n".join(breakdown)
    result += f"\n\nTotal: ${total:.2f}"
    if total > 100:
        result += f"\n  WARNING: Over $100 budget by ${total - 100:.2f}!"
    else:
        result += f"\n  Remaining budget: ${100 - total:.2f}"

    return result


# ==============================================================
# Step 3: Define the State
# ==============================================================
# The state tracks the full plan-execute lifecycle:
#   - goal: what we're trying to achieve
#   - plan: list of steps the planner creates
#   - current_step: which step we're executing
#   - step_results: results from each completed step
#   - final_output: the synthesized final answer
#   - iteration / max_iterations: safety counters

class PlanExecuteState(TypedDict):
    goal: str                # The complex goal to achieve
    plan: list               # List of step descriptions
    current_step: int        # Index of current step being executed
    step_results: list       # Results from completed steps
    final_output: str        # Final synthesized result
    iteration: int           # Safety counter
    max_iterations: int      # Safety limit
    messages: Annotated[list, add_messages]  # Messages for tool-calling executor


# ==============================================================
# Step 4: Define the Nodes
# ==============================================================

# Collect tools and bind to executor LLM
tools = [recipe_search, nutrition_check, budget_calculator]
executor_llm_with_tools = executor_llm.bind_tools(tools)


# -- Planner Node -----------------------------------------------
# The planner decomposes the goal into a numbered list of steps.
# Uses temperature=0 for deterministic, consistent planning.

def planner_node(state: PlanExecuteState) -> dict:
    """Decompose the goal into an ordered list of 3-5 steps."""
    print(f"\n{'='*60}")
    print("  PLANNER: Decomposing goal into steps...")
    print(f"{'='*60}")

    messages = [
        SystemMessage(content=(
            "You are a planning agent. Given a complex goal, break it down into "
            "3-5 concrete, ordered steps. Each step should be actionable and specific.\n\n"
            "Output ONLY a numbered list, one step per line, like:\n"
            "1. First step description\n"
            "2. Second step description\n"
            "3. Third step description\n\n"
            "Do NOT include any other text, explanations, or formatting."
        )),
        HumanMessage(content=f"Goal: {state['goal']}"),
    ]

    response = planner_llm.invoke(messages)
    raw_plan = response.content.strip()

    # Parse the numbered list into a Python list
    # Handle formats like "1. Step" or "1) Step" or "- Step"
    steps = []
    for line in raw_plan.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering prefixes: "1. ", "1) ", "- ", "* "
        cleaned = re.sub(r'^(\d+[\.\)]\s*|[-*]\s*)', '', line).strip()
        if cleaned:
            steps.append(cleaned)

    # Safety: ensure we have at least 1 step and at most 7
    if not steps:
        steps = ["Research the topic and provide a comprehensive answer"]
    steps = steps[:7]

    print(f"\n  Plan ({len(steps)} steps):")
    for i, step in enumerate(steps):
        print(f"    {i+1}. {step}")

    return {
        "plan": steps,
        "current_step": 0,
        "step_results": [],
        "iteration": 0,
        "messages": [],
    }


# -- Executor Node -----------------------------------------------
# Executes the current step using the LLM with tools bound.
# The LLM decides which tool(s) to call for each step.

def executor_node(state: PlanExecuteState) -> dict:
    """Execute the current step using the LLM with tools."""
    step_idx = state["current_step"]
    step = state["plan"][step_idx]
    iteration = state.get("iteration", 0) + 1

    print(f"\n{'- '*30}")
    print(f"  EXECUTOR: Step {step_idx + 1}/{len(state['plan'])}")
    print(f"  Task: {step}")
    print(f"{'- '*30}")

    # Build context for the executor: the goal, the full plan, and
    # results from previous steps so the LLM can build on them.
    context_parts = [f"Overall Goal: {state['goal']}", f"\nCurrent Task: {step}"]

    if state["step_results"]:
        context_parts.append("\nResults from previous steps:")
        for i, result in enumerate(state["step_results"]):
            context_parts.append(f"  Step {i+1}: {result[:200]}")

    context = "\n".join(context_parts)

    # Use fresh messages for each step execution so tool history
    # doesn't accumulate across steps (cleaner context)
    step_messages = [
        SystemMessage(content=(
            "You are an execution agent with access to tools. Complete the given task "
            "using the available tools. Use the tools to gather real information, then "
            "provide a clear, concise summary of what you found.\n\n"
            "Available tools:\n"
            "  - recipe_search: Find recipes by cuisine type\n"
            "  - nutrition_check: Check nutrition and allergen info for a dish\n"
            "  - budget_calculator: Calculate costs for a list of items\n\n"
            "After using tools, provide a clear summary of the results."
        )),
        HumanMessage(content=context),
    ]

    # Execute with tool-calling loop (max 5 tool calls per step)
    max_tool_calls = 5
    tool_call_count = 0
    tool_node = ToolNode(tools)

    for _ in range(max_tool_calls):
        response = executor_llm_with_tools.invoke(step_messages)
        step_messages.append(response)

        # If the LLM wants to call tools, execute them
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call_count += len(response.tool_calls)
            for tc in response.tool_calls:
                print(f"    [TOOL] {tc['name']}({tc['args']})")

            # Use ToolNode to execute the tool calls
            tool_result = tool_node.invoke({"messages": step_messages})
            tool_messages = tool_result["messages"]
            step_messages.extend(tool_messages)
        else:
            # No more tool calls — LLM is done with this step
            break

    # Extract the final response text for this step
    result_text = response.content if response.content else "Step completed (no text output)."

    print(f"\n    Result: {result_text[:300]}...")
    if tool_call_count > 0:
        print(f"    ({tool_call_count} tool call(s) made)")

    # Append result to step_results
    updated_results = list(state["step_results"]) + [result_text]

    return {
        "step_results": updated_results,
        "iteration": iteration,
        "messages": [],  # Reset messages for next step
    }


# -- Progress Check Node -------------------------------------------
# Advances current_step. Routes to executor (more steps) or synthesize (done).

def progress_check_node(state: PlanExecuteState) -> dict:
    """Advance to the next step."""
    next_step = state["current_step"] + 1
    total_steps = len(state["plan"])

    if next_step < total_steps:
        print(f"\n  PROGRESS: Step {next_step}/{total_steps} complete. "
              f"Moving to step {next_step + 1}.")
    else:
        print(f"\n  PROGRESS: All {total_steps} steps complete! Moving to synthesis.")

    return {"current_step": next_step}


# -- Synthesize Node -----------------------------------------------
# Combines all step results into a cohesive final answer.

def synthesize_node(state: PlanExecuteState) -> dict:
    """Synthesize all step results into a final output."""
    print(f"\n{'='*60}")
    print("  SYNTHESIZER: Combining all results...")
    print(f"{'='*60}")

    # Build the synthesis prompt with all step results
    results_summary = []
    for i, (step, result) in enumerate(zip(state["plan"], state["step_results"])):
        results_summary.append(f"Step {i+1} ({step}):\n{result}")

    messages = [
        SystemMessage(content=(
            "You are a synthesis agent. You receive results from multiple completed "
            "steps of a plan. Combine them into a single, clear, well-organized "
            "final answer that addresses the original goal. Include specific details "
            "from each step. Format it nicely with sections."
        )),
        HumanMessage(content=(
            f"Original Goal: {state['goal']}\n\n"
            f"Completed Step Results:\n\n"
            + "\n\n".join(results_summary)
            + "\n\nProvide a comprehensive final answer addressing the goal:"
        )),
    ]

    response = executor_llm.invoke(messages)
    final_output = response.content

    print(f"\n  Final output generated ({len(final_output)} characters)")

    return {"final_output": final_output}


# ==============================================================
# Step 5: Routing Function
# ==============================================================

def should_continue(state: PlanExecuteState) -> str:
    """Decide whether to execute the next step or synthesize.

    Routes to:
      - "executor": more steps to execute
      - "synthesize": all steps done
      - "synthesize": safety limit reached (max_iterations)
    """
    # Safety valve: prevent runaway loops
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        print(f"  [WARN] Safety limit ({state['max_iterations']}) reached. "
              f"Forcing synthesis.")
        return "synthesize"

    # Check if all steps are done
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"

    return "executor"


# ==============================================================
# Step 6: Build the Graph
# ==============================================================
# Flow: planner -> executor -> progress_check -> [executor -> ...]* -> synthesize -> END
#
# This separates PLANNING (what to do) from EXECUTION (how to do it),
# which is the core insight of the Plan-Execute pattern.

def build_plan_execute_graph():
    """Build the Plan-Execute graph."""
    graph = StateGraph(PlanExecuteState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("progress_check", progress_check_node)
    graph.add_node("synthesize", synthesize_node)

    # Entry point: always start by planning
    graph.set_entry_point("planner")

    # After planning, go to executor
    graph.add_edge("planner", "executor")

    # After executing, check progress
    graph.add_edge("executor", "progress_check")

    # After progress check, decide: more steps or synthesize
    graph.add_conditional_edges(
        "progress_check",
        should_continue,
        {
            "executor": "executor",      # More steps to do
            "synthesize": "synthesize",   # All done
        },
    )

    # After synthesis, end
    graph.add_edge("synthesize", END)

    return graph.compile()


# ==============================================================
# Step 7: Run the Plan-Execute Agent
# ==============================================================

def run_plan_execute(goal: str, max_iterations: int = 10) -> dict:
    """Run the Plan-Execute agent on a complex goal.

    Args:
        goal: The complex goal to achieve
        max_iterations: Safety limit for total execution steps (default 10)

    Returns:
        The final state with plan, step results, and final output
    """
    app = build_plan_execute_graph()

    result = app.invoke({
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "final_output": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "messages": [],
    })

    return result


if __name__ == "__main__":
    print("Example 2: Plan-Execute Pattern in LangGraph")
    print("=" * 60)
    print("The LLM will decompose a complex goal into steps,")
    print("execute each step with tools, then synthesize the results.")
    print("=" * 60)

    goal = "Plan a dinner party for 8 people with a $100 budget, one guest is gluten-free"

    result = run_plan_execute(goal, max_iterations=10)

    # Display the plan
    print(f"\n\n{'#'*60}")
    print(f"  EXECUTION SUMMARY")
    print(f"{'#'*60}")

    print(f"\n  Goal: {result['goal']}")
    print(f"\n  Plan ({len(result['plan'])} steps):")
    for i, step in enumerate(result["plan"]):
        print(f"    {i+1}. {step}")

    print(f"\n  Step Results:")
    for i, step_result in enumerate(result["step_results"]):
        print(f"\n    --- Step {i+1} ---")
        # Truncate long results for readability
        for line in step_result[:500].split("\n"):
            if line.strip():
                print(f"    {line.strip()[:120]}")
        if len(step_result) > 500:
            print(f"    ... ({len(step_result) - 500} more characters)")

    print(f"\n{'#'*60}")
    print(f"  FINAL OUTPUT")
    print(f"{'#'*60}")
    print(f"\n{result['final_output']}")

    print(f"\n{'='*60}")
    print("Key Takeaways:")
    print("  1. SEPARATION OF CONCERNS: Planner thinks about WHAT, executor about HOW")
    print("  2. TOOL AUTONOMY: The executor decides which tools to call per step")
    print("  3. CONTEXT ACCUMULATION: Each step can build on previous results")
    print("  4. SAFETY LIMITS: max_iterations prevents infinite execution loops")
    print("  5. COMPOSABILITY: Planning + tool-use patterns work together")
    print(f"{'='*60}")
