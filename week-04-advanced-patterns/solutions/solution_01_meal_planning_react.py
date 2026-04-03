import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 1: Plan-Execute-Replan Pattern -- Meal Planning Agent with HITL
=========================================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 3 hours

This is the complete solution for exercise_01_meal_planning_react.py.

The agent:
  1. Plans -- Decomposes the meal planning goal into 5 ordered steps
  2. Executes -- Handles each step using simulated tools
  3. Synthesizes -- Combines results into a 7-day meal plan
  4. HITL -- Asks the user for approval
  5. Replans -- If rejected, incorporates feedback and starts over

Run: python week-04-advanced-patterns/solutions/solution_01_meal_planning_react.py
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
# Step 2: Simulated Tools
# ==============================================================

@tool
def recipe_search(query: str) -> str:
    """Search for recipes matching a query. Use this to find meal ideas
    based on cuisine type, dietary restriction, or ingredient.

    Args:
        query: Search query (e.g., 'vegetarian dinner', 'gluten-free lunch', 'budget chicken')
    """
    recipes_db = {
        "vegetarian": (
            "Vegetarian Recipes Found:\n"
            "  1. Lentil Curry (serves 4) - red lentils, coconut milk, spinach, rice. "
            "Cost: $8. Prep: 10min, Cook: 25min. Allergens: none.\n"
            "  2. Caprese Pasta (serves 4) - penne, tomatoes, mozzarella, basil. "
            "Cost: $10. Prep: 10min, Cook: 15min. Allergens: gluten, dairy.\n"
            "  3. Black Bean Tacos (serves 4) - corn tortillas, black beans, avocado, salsa. "
            "Cost: $7. Prep: 15min, Cook: 10min. Allergens: none.\n"
            "  4. Mushroom Risotto (serves 4) - arborio rice, mushrooms, parmesan, broth. "
            "Cost: $12. Prep: 10min, Cook: 30min. Allergens: dairy."
        ),
        "chicken": (
            "Chicken Recipes Found:\n"
            "  1. Grilled Chicken Salad (serves 4) - chicken breast, mixed greens, vinaigrette. "
            "Cost: $9. Prep: 15min, Cook: 20min. Allergens: none.\n"
            "  2. Chicken Stir-Fry (serves 4) - chicken, broccoli, bell pepper, soy sauce, rice. "
            "Cost: $10. Prep: 15min, Cook: 15min. Allergens: soy.\n"
            "  3. Chicken Fajitas (serves 4) - chicken, peppers, onions, tortillas, guacamole. "
            "Cost: $11. Prep: 20min, Cook: 15min. Allergens: gluten (flour tortillas).\n"
            "  4. Lemon Herb Chicken (serves 4) - chicken thighs, lemon, herbs, potatoes. "
            "Cost: $10. Prep: 10min, Cook: 35min. Allergens: none."
        ),
        "fish": (
            "Fish Recipes Found:\n"
            "  1. Baked Salmon (serves 4) - salmon fillets, lemon, dill, asparagus. "
            "Cost: $16. Prep: 10min, Cook: 20min. Allergens: fish.\n"
            "  2. Fish Tacos (serves 4) - white fish, cabbage slaw, corn tortillas, lime. "
            "Cost: $12. Prep: 15min, Cook: 10min. Allergens: fish.\n"
            "  3. Tuna Pasta (serves 4) - canned tuna, penne, olive oil, capers, tomatoes. "
            "Cost: $8. Prep: 10min, Cook: 15min. Allergens: fish, gluten."
        ),
        "budget": (
            "Budget-Friendly Recipes Found:\n"
            "  1. Rice and Beans (serves 4) - rice, pinto beans, onion, cumin, salsa. "
            "Cost: $4. Prep: 5min, Cook: 20min. Allergens: none.\n"
            "  2. Egg Fried Rice (serves 4) - rice, eggs, frozen vegetables, soy sauce. "
            "Cost: $5. Prep: 5min, Cook: 10min. Allergens: eggs, soy.\n"
            "  3. Spaghetti Marinara (serves 4) - spaghetti, canned tomatoes, garlic, basil. "
            "Cost: $5. Prep: 5min, Cook: 15min. Allergens: gluten.\n"
            "  4. Potato Soup (serves 4) - potatoes, onion, broth, cream, chives. "
            "Cost: $6. Prep: 10min, Cook: 25min. Allergens: dairy."
        ),
        "gluten-free": (
            "Gluten-Free Recipes Found:\n"
            "  1. Grilled Chicken with Quinoa (serves 4) - chicken, quinoa, roasted vegetables. "
            "Cost: $11. Prep: 10min, Cook: 25min. Allergens: none.\n"
            "  2. Shrimp and Rice Bowl (serves 4) - shrimp, jasmine rice, edamame, teriyaki. "
            "Cost: $13. Prep: 10min, Cook: 15min. Allergens: shellfish, soy.\n"
            "  3. Turkey Lettuce Wraps (serves 4) - ground turkey, lettuce, water chestnuts. "
            "Cost: $9. Prep: 15min, Cook: 10min. Allergens: none.\n"
            "  4. Stuffed Bell Peppers (serves 4) - bell peppers, ground beef, rice, cheese. "
            "Cost: $10. Prep: 15min, Cook: 30min. Allergens: dairy."
        ),
        "breakfast": (
            "Breakfast Recipes Found:\n"
            "  1. Oatmeal with Fruit (serves 4) - oats, banana, berries, honey. "
            "Cost: $4. Prep: 5min, Cook: 5min. Allergens: gluten (oats).\n"
            "  2. Veggie Omelette (serves 4) - eggs, spinach, tomato, cheese. "
            "Cost: $6. Prep: 10min, Cook: 10min. Allergens: eggs, dairy.\n"
            "  3. Smoothie Bowl (serves 4) - frozen berries, banana, yogurt, granola. "
            "Cost: $7. Prep: 10min, Cook: 0min. Allergens: dairy, gluten."
        ),
    }

    for key, recipes in recipes_db.items():
        if key in query.lower():
            return recipes

    return (
        f"No exact match for '{query}'. Try: vegetarian, chicken, fish, "
        f"budget, gluten-free, or breakfast."
    )


@tool
def nutrition_check(meal_name: str) -> str:
    """Check nutrition information for a specific meal or dish.
    Returns calories, protein, fiber, and dietary notes.

    Args:
        meal_name: Name of the meal (e.g., 'lentil curry', 'chicken stir-fry')
    """
    nutrition_db = {
        "lentil curry": "Lentil Curry: 380 cal, 18g protein, 12g fiber. High in iron and folate. Vegan-friendly.",
        "caprese pasta": "Caprese Pasta: 420 cal, 14g protein, 3g fiber. Contains gluten and dairy.",
        "black bean tacos": "Black Bean Tacos: 310 cal, 15g protein, 11g fiber. High fiber, vegan option.",
        "mushroom risotto": "Mushroom Risotto: 450 cal, 10g protein, 2g fiber. Rich in B vitamins. Contains dairy.",
        "grilled chicken salad": "Grilled Chicken Salad: 280 cal, 32g protein, 4g fiber. Low carb, high protein.",
        "chicken stir-fry": "Chicken Stir-Fry: 350 cal, 28g protein, 4g fiber. Good balance of macros.",
        "chicken fajitas": "Chicken Fajitas: 400 cal, 30g protein, 3g fiber. Contains gluten (tortillas).",
        "lemon herb chicken": "Lemon Herb Chicken: 380 cal, 35g protein, 3g fiber. High protein, GF-friendly.",
        "baked salmon": "Baked Salmon: 420 cal, 38g protein, 2g fiber. Rich in omega-3 fatty acids.",
        "fish tacos": "Fish Tacos: 320 cal, 24g protein, 3g fiber. Good omega-3 source. GF with corn tortillas.",
        "rice and beans": "Rice and Beans: 340 cal, 12g protein, 8g fiber. Complete protein when combined.",
        "egg fried rice": "Egg Fried Rice: 380 cal, 14g protein, 2g fiber. Quick and affordable.",
        "spaghetti marinara": "Spaghetti Marinara: 360 cal, 10g protein, 4g fiber. Contains gluten.",
        "potato soup": "Potato Soup: 300 cal, 6g protein, 3g fiber. Comfort food, contains dairy.",
        "oatmeal": "Oatmeal with Fruit: 280 cal, 8g protein, 5g fiber. Heart-healthy whole grains.",
        "omelette": "Veggie Omelette: 320 cal, 22g protein, 2g fiber. High protein breakfast.",
        "smoothie": "Smoothie Bowl: 290 cal, 10g protein, 4g fiber. Rich in antioxidants.",
        "quinoa": "Grilled Chicken with Quinoa: 400 cal, 36g protein, 5g fiber. Complete protein.",
        "shrimp": "Shrimp and Rice Bowl: 370 cal, 28g protein, 2g fiber. Low fat, high protein.",
        "lettuce wrap": "Turkey Lettuce Wraps: 260 cal, 26g protein, 2g fiber. Low carb option.",
        "stuffed pepper": "Stuffed Bell Peppers: 380 cal, 24g protein, 4g fiber. Balanced macros.",
    }

    for key, info in nutrition_db.items():
        if key in meal_name.lower():
            return info

    return f"No nutrition data for '{meal_name}'. Try a common dish name like 'chicken stir-fry' or 'lentil curry'."


@tool
def budget_calculator(items: str) -> str:
    """Calculate the total cost of grocery items for meal planning.
    Provide items as a comma-separated list with optional quantities.

    Args:
        items: Comma-separated grocery items (e.g., 'chicken breast, rice, broccoli, soy sauce')
    """
    price_db = {
        "chicken": 12.00, "turkey": 10.00, "beef": 18.00, "salmon": 16.00,
        "fish": 14.00, "shrimp": 15.00, "tuna": 5.00, "egg": 4.00,
        "rice": 4.00, "pasta": 3.00, "spaghetti": 3.00, "quinoa": 6.00,
        "oat": 3.50, "bread": 3.50, "tortilla": 4.00, "penne": 3.00,
        "lentil": 3.00, "bean": 2.50, "chickpea": 3.00,
        "broccoli": 3.00, "spinach": 3.50, "lettuce": 2.50, "tomato": 3.50,
        "onion": 2.00, "potato": 4.00, "pepper": 3.50, "mushroom": 4.00,
        "avocado": 5.00, "asparagus": 4.50, "cabbage": 2.50, "carrot": 2.00,
        "banana": 2.00, "berries": 5.00, "lemon": 1.50, "lime": 1.50,
        "mango": 3.00, "frozen vegetable": 3.00,
        "cheese": 6.00, "mozzarella": 5.00, "parmesan": 7.00, "cream": 4.00,
        "yogurt": 4.00, "butter": 4.50, "milk": 3.50, "coconut milk": 3.00,
        "olive oil": 8.00, "soy sauce": 3.00, "salsa": 3.50, "vinaigrette": 4.00,
        "cumin": 2.50, "herb": 3.00, "basil": 2.00, "dill": 2.00,
        "garlic": 1.50, "honey": 4.00, "granola": 4.50, "broth": 3.00,
        "caper": 4.00, "chive": 2.00, "water chestnut": 3.00,
        "guacamole": 5.00, "edamame": 4.00,
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
    return result


# ==============================================================
# Step 3: Define the State (SOLUTION for TODO 1)
# ==============================================================
# The state tracks the full plan-execute-replan lifecycle:
#   - goal/constraints: what we're planning for
#   - plan/current_step/step_results: plan-execute tracking
#   - meal_plan: the synthesized output
#   - approved/feedback: HITL approval state
#   - iteration/max_iterations: safety counters
#   - messages: for tool-calling in the executor

class MealPlanState(TypedDict):
    goal: str                                   # The meal planning request
    constraints: dict                           # Budget, dietary, family_size
    plan: list                                  # List of step descriptions
    current_step: int                           # Index of step being executed
    step_results: list                          # Results from each completed step
    meal_plan: str                              # Synthesized 7-day meal plan
    approved: bool                              # Whether user approved
    feedback: str                               # User feedback if rejected
    iteration: int                              # Current iteration (safety)
    max_iterations: int                         # Max replan attempts
    messages: Annotated[list, add_messages]      # Messages for tool-calling


# ==============================================================
# Step 4: Bind Tools to LLM
# ==============================================================

tools = [recipe_search, nutrition_check, budget_calculator]
executor_llm_with_tools = executor_llm.bind_tools(tools)


# ==============================================================
# Step 5: Planner Node (SOLUTION for TODO 2)
# ==============================================================
# The planner decomposes the meal planning goal into ordered steps.
# If the user rejected a previous plan and gave feedback, the planner
# incorporates that feedback into the new plan.

def planner_node(state: MealPlanState) -> dict:
    """Decompose the meal planning goal into 5 ordered steps."""
    print(f"\n{'='*60}")
    print("  PLANNER: Creating meal planning steps...")
    print(f"{'='*60}")

    constraints = state.get("constraints", {})
    feedback = state.get("feedback", "")

    # Build the system prompt
    system_content = (
        "You are a meal planning agent. Given a goal and constraints, break the task "
        "into exactly 5 concrete, ordered steps for creating a weekly meal plan.\n\n"
        "The steps should cover:\n"
        "  1. Reviewing dietary restrictions and family requirements\n"
        "  2. Searching for suitable recipes\n"
        "  3. Checking nutritional balance\n"
        "  4. Organizing into a 7-day schedule\n"
        "  5. Calculating the grocery budget\n\n"
        "Output ONLY a numbered list, one step per line:\n"
        "1. First step\n"
        "2. Second step\n"
        "...\n\n"
        "Do NOT include any other text."
    )

    # Build the human message with goal, constraints, and any feedback
    human_content = (
        f"Goal: {state['goal']}\n\n"
        f"Constraints:\n"
        f"  Budget: ${constraints.get('budget', 100)}/week\n"
        f"  Dietary: {constraints.get('dietary', 'none')}\n"
        f"  Family Size: {constraints.get('family_size', 4)}\n"
    )

    # If the user rejected a previous plan, include their feedback
    if feedback:
        human_content += (
            f"\nIMPORTANT - Previous plan was REJECTED. User feedback:\n"
            f"  \"{feedback}\"\n"
            f"Adjust your plan to address this feedback."
        )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    response = planner_llm.invoke(messages)
    raw_plan = response.content.strip()

    # Parse the numbered list into a Python list
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
        steps = [
            "Review dietary restrictions and family size requirements",
            "Search for recipes matching the constraints",
            "Check nutritional balance of selected meals",
            "Create a 7-day meal schedule",
            "Calculate total grocery budget and shopping list",
        ]
    steps = steps[:7]

    print(f"\n  Plan ({len(steps)} steps):")
    for i, step in enumerate(steps):
        print(f"    {i+1}. {step}")

    if feedback:
        print(f"\n  (Replanning based on feedback: \"{feedback[:80]}...\")")

    return {
        "plan": steps,
        "current_step": 0,
        "step_results": [],
        "iteration": state.get("iteration", 0),
        "messages": [],
    }


# ==============================================================
# Step 6: Executor Node (SOLUTION for TODO 3)
# ==============================================================
# Executes the current step using the LLM with bound tools.
# The LLM autonomously decides which tool(s) to call for each step.

def executor_node(state: MealPlanState) -> dict:
    """Execute the current step using the LLM with tools."""
    step_idx = state["current_step"]
    step = state["plan"][step_idx]
    iteration = state.get("iteration", 0) + 1

    print(f"\n{'- '*30}")
    print(f"  EXECUTOR: Step {step_idx + 1}/{len(state['plan'])}")
    print(f"  Task: {step}")
    print(f"{'- '*30}")

    # Build context: goal, constraints, and previous step results
    constraints = state.get("constraints", {})
    context_parts = [
        f"Overall Goal: {state['goal']}",
        f"Constraints: Budget=${constraints.get('budget', 100)}/week, "
        f"Dietary={constraints.get('dietary', 'none')}, "
        f"Family Size={constraints.get('family_size', 4)}",
        f"\nCurrent Task: {step}",
    ]

    if state["step_results"]:
        context_parts.append("\nResults from previous steps:")
        for i, result in enumerate(state["step_results"]):
            context_parts.append(f"  Step {i+1}: {result[:300]}")

    context = "\n".join(context_parts)

    # Fresh messages for each step (avoid accumulating tool history)
    step_messages = [
        SystemMessage(content=(
            "You are a meal planning execution agent with access to tools. "
            "Complete the given task using the available tools.\n\n"
            "Available tools:\n"
            "  - recipe_search: Find recipes by cuisine, dietary restriction, or type\n"
            "  - nutrition_check: Check calories, protein, fiber for a dish\n"
            "  - budget_calculator: Calculate costs for a comma-separated list of items\n\n"
            "Use the tools to gather real information, then provide a clear summary."
        )),
        HumanMessage(content=context),
    ]

    # Tool-calling loop: let the LLM call tools up to 5 times per step
    max_tool_calls = 5
    tool_call_count = 0
    tool_node = ToolNode(tools)

    for _ in range(max_tool_calls):
        response = executor_llm_with_tools.invoke(step_messages)
        step_messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call_count += len(response.tool_calls)
            for tc in response.tool_calls:
                print(f"    [TOOL] {tc['name']}({tc['args']})")

            # Execute the tool calls
            tool_result = tool_node.invoke({"messages": step_messages})
            tool_messages = tool_result["messages"]
            step_messages.extend(tool_messages)
        else:
            # No more tool calls -- LLM is done with this step
            break

    # Extract the final response text
    result_text = response.content if response.content else "Step completed (no text output)."

    print(f"\n    Result: {result_text[:300]}...")
    if tool_call_count > 0:
        print(f"    ({tool_call_count} tool call(s) made)")

    updated_results = list(state["step_results"]) + [result_text]

    return {
        "step_results": updated_results,
        "iteration": iteration,
        "messages": [],
    }


# ==============================================================
# Step 7: Progress Check Node
# ==============================================================

def progress_check_node(state: MealPlanState) -> dict:
    """Advance to the next step."""
    next_step = state["current_step"] + 1
    total_steps = len(state["plan"])

    if next_step < total_steps:
        print(f"\n  PROGRESS: Step {next_step}/{total_steps} complete. "
              f"Moving to step {next_step + 1}.")
    else:
        print(f"\n  PROGRESS: All {total_steps} steps complete! Moving to synthesis.")

    return {"current_step": next_step}


# ==============================================================
# Step 8: Synthesize Node
# ==============================================================

def synthesize_node(state: MealPlanState) -> dict:
    """Combine all step results into a 7-day meal plan."""
    print(f"\n{'='*60}")
    print("  SYNTHESIZER: Creating 7-day meal plan...")
    print(f"{'='*60}")

    results_summary = []
    for i, (step, result) in enumerate(zip(state["plan"], state["step_results"])):
        results_summary.append(f"Step {i+1} ({step}):\n{result}")

    constraints = state.get("constraints", {})
    messages = [
        SystemMessage(content=(
            "You are a meal planning expert. Using the research results from each step, "
            "create a complete 7-day meal plan. Format it clearly with:\n"
            "- Day-by-day breakdown (Mon-Sun)\n"
            "- Breakfast, Lunch, and Dinner for each day\n"
            "- Estimated daily cost\n"
            "- Total weekly cost\n"
            "- A consolidated shopping list at the end\n\n"
            "Ensure the plan respects all dietary restrictions and stays within budget."
        )),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            f"Constraints:\n"
            f"  Budget: ${constraints.get('budget', 100)}/week\n"
            f"  Dietary: {constraints.get('dietary', 'none')}\n"
            f"  Family Size: {constraints.get('family_size', 4)}\n\n"
            f"Research Results:\n\n"
            + "\n\n".join(results_summary)
            + "\n\nCreate the 7-day meal plan:"
        )),
    ]

    response = executor_llm.invoke(messages)
    meal_plan = response.content

    print(f"\n  Meal plan generated ({len(meal_plan)} characters)")

    return {"meal_plan": meal_plan}


# ==============================================================
# Step 9: HITL Checkpoint (SOLUTION for TODO 4)
# ==============================================================
# This node pauses execution and presents the meal plan to the user.
# The user can approve it (continue to END) or reject it with feedback
# (loop back to the planner for replanning).

def hitl_checkpoint(state: MealPlanState) -> dict:
    """Pause and ask the user to approve the meal plan."""
    print(f"\n{'='*60}")
    print("  HITL CHECKPOINT: Review the Meal Plan")
    print(f"{'='*60}")
    print(f"\n{state['meal_plan']}")
    print(f"\n{'='*60}")

    # Ask for user approval. In non-interactive environments (CI, piped input),
    # default to "yes" so the agent doesn't hang.
    try:
        approval = input("\nDo you approve this meal plan? (yes/no) [yes]: ").strip().lower()
        if not approval:
            approval = "yes"
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment -- auto-approve
        print("\n  [AUTO] Non-interactive mode detected. Auto-approving.")
        approval = "yes"

    if approval in ("yes", "y"):
        print("\n  [APPROVED] Meal plan accepted!")
        return {
            "approved": True,
            "feedback": "",
        }
    else:
        # Ask for feedback on what to change
        try:
            feedback = input("What would you like changed? ").strip()
            if not feedback:
                feedback = "Please adjust the meal variety and costs."
        except (EOFError, KeyboardInterrupt):
            feedback = "Please adjust the meal plan."

        print(f"\n  [REJECTED] Will replan with feedback: \"{feedback}\"")
        return {
            "approved": False,
            "feedback": feedback,
            "iteration": state.get("iteration", 0) + 1,
        }


# ==============================================================
# Step 10: Routing Functions (SOLUTION for TODO 5)
# ==============================================================

def should_continue(state: MealPlanState) -> str:
    """Route after progress_check: more steps or synthesize.

    Returns:
      - "executor": more steps remain to execute
      - "synthesize": all steps done or safety limit reached
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


def should_replan(state: MealPlanState) -> str:
    """Route after hitl_checkpoint: done or replan.

    Returns:
      - "done": user approved OR max iterations reached
      - "replan": user rejected and replanning attempts remain
    """
    if state.get("approved", False):
        return "done"

    # Check if we still have replanning attempts left
    if state.get("iteration", 0) < state.get("max_iterations", 3):
        print(f"\n  REPLANNING: Attempt {state['iteration']}/{state['max_iterations']}")
        return "replan"

    # Max iterations reached -- force accept
    print(f"\n  [WARN] Max replanning attempts reached. Accepting current plan.")
    return "done"


# ==============================================================
# Step 11: Build the Graph (SOLUTION for TODO 6)
# ==============================================================
# Flow:
#   planner -> executor -> progress_check -> [executor]*
#                                         -> synthesize -> hitl_checkpoint
#                                                          -> END (approved)
#                                                          -> planner (rejected)

def build_meal_planning_graph():
    """Build the Plan-Execute-Replan graph with HITL checkpoint."""
    graph = StateGraph(MealPlanState)

    # Add all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("progress_check", progress_check_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("hitl_checkpoint", hitl_checkpoint)

    # Entry point: always start by planning
    graph.set_entry_point("planner")

    # After planning, start executing
    graph.add_edge("planner", "executor")

    # After executing a step, check progress
    graph.add_edge("executor", "progress_check")

    # After progress check: more steps or synthesize
    graph.add_conditional_edges(
        "progress_check",
        should_continue,
        {
            "executor": "executor",
            "synthesize": "synthesize",
        },
    )

    # After synthesis, go to HITL checkpoint
    graph.add_edge("synthesize", "hitl_checkpoint")

    # After HITL: done (END) or replan (back to planner)
    graph.add_conditional_edges(
        "hitl_checkpoint",
        should_replan,
        {
            "done": END,
            "replan": "planner",
        },
    )

    return graph.compile()


# ==============================================================
# Run Function
# ==============================================================

def run_meal_planner(budget=75, dietary="none", family_size=4, max_iterations=3):
    """Run the meal planning agent.

    Args:
        budget: Weekly grocery budget in dollars
        dietary: Dietary restrictions (e.g., 'vegetarian', 'gluten-free', 'none')
        family_size: Number of people to plan meals for
        max_iterations: Max replan attempts if user rejects

    Returns:
        The final state with the approved (or forced) meal plan
    """
    goal = (
        f"Create a 7-day meal plan for a family of {family_size} "
        f"with a ${budget}/week grocery budget"
    )
    if dietary != "none":
        goal += f" following a {dietary} diet"

    constraints = {
        "budget": budget,
        "dietary": dietary,
        "family_size": family_size,
    }

    print(f"\nGoal: {goal}")
    print(f"Constraints: {json.dumps(constraints, indent=2)}")
    print(f"Max replanning iterations: {max_iterations}")

    app = build_meal_planning_graph()

    result = app.invoke({
        "goal": goal,
        "constraints": constraints,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "meal_plan": "",
        "approved": False,
        "feedback": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "messages": [],
    })

    return result


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    print("Solution 1: Meal Planning Agent with Plan-Execute-Replan + HITL")
    print("=" * 65)
    print("The agent will:")
    print("  1. Plan the meal planning task into steps")
    print("  2. Execute each step with tools (recipe search, nutrition, budget)")
    print("  3. Synthesize a 7-day meal plan")
    print("  4. Ask for your approval (HITL checkpoint)")
    print("  5. Replan if you reject (with your feedback)")
    print("=" * 65)

    # Test 1: Gluten-free family meal plan
    print("\n\nTest 1: Gluten-free family meal plan ($75, family of 4)")
    print("-" * 50)
    result = run_meal_planner(budget=75, dietary="gluten-free", family_size=4)

    print(f"\n\n{'#'*60}")
    print("  FINAL RESULT")
    print(f"{'#'*60}")
    print(f"\n  Approved: {result.get('approved', 'N/A')}")
    print(f"  Iterations: {result.get('iteration', 'N/A')}")
    print(f"  Plan steps: {len(result.get('plan', []))}")
    print(f"\n  Meal Plan:\n{result.get('meal_plan', 'No plan generated')[:1000]}")

    # Uncomment for additional tests:
    # print("\n\nTest 2: Vegetarian meal plan ($80, couple)")
    # print("-" * 50)
    # result = run_meal_planner(budget=80, dietary="vegetarian", family_size=2)
    # print(f"\n  Meal Plan:\n{result.get('meal_plan', '')[:1000]}")

    # print("\n\nTest 3: Gluten-free family ($100, family of 6)")
    # print("-" * 50)
    # result = run_meal_planner(budget=100, dietary="gluten-free", family_size=6)
    # print(f"\n  Meal Plan:\n{result.get('meal_plan', '')[:1000]}")

    print(f"\n{'='*60}")
    print("Key Takeaways:")
    print("  1. PLAN-EXECUTE: Planner decomposes, executor handles each step with tools")
    print("  2. HITL APPROVAL: Human reviews and can reject with feedback")
    print("  3. REPLANNING: Agent adapts the plan based on user feedback")
    print("  4. SAFETY LIMITS: max_iterations prevents infinite replan loops")
    print("  5. TOOL AUTONOMY: LLM decides which tools to call per step")
    print(f"{'='*60}")
