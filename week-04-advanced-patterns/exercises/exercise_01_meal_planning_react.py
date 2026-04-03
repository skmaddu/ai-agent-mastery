import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 1: Plan-Execute-Replan Pattern -- Meal Planning Agent with HITL
=========================================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 3 hours

Task:
Build a LangGraph meal planning agent that uses the Plan-Execute-Replan pattern
with Human-in-the-Loop (HITL) approval. The agent accepts dietary constraints
(budget, dietary restrictions, family size) and creates a 7-day meal plan.

The agent follows this flow:
  1. PLAN   -- Decompose the goal into 5 ordered steps
  2. EXECUTE -- Handle each step using tools (recipe search, nutrition, budget)
  3. SYNTHESIZE -- Combine results into a 7-day meal plan
  4. HITL CHECKPOINT -- Present the plan to the user for approval
  5. REPLAN -- If rejected, incorporate feedback and replan

Graph:
  START -> planner -> executor -> progress_check
                        ^              |
                        |              | (steps remain)
                        +──────────────+
                                       | (all done)
                                       v
                                    synthesize -> hitl_checkpoint
                                       ^                |
                                       |    (rejected)  |
                                       +────────────────+
                                                        | (approved)
                                                        v
                                                       END

Instructions:
1. Define the MealPlanState TypedDict with all required fields (TODO 1)
2. Implement planner_node to decompose the meal planning goal (TODO 2)
3. Implement executor_node to execute each step with tools (TODO 3)
4. Implement hitl_checkpoint to pause and ask user for approval (TODO 4)
5. Implement should_continue routing function (TODO 5)
6. Wire all graph edges together (TODO 6)

Hints:
- Study example_02_planning_langgraph.py for the Plan-Execute pattern
- The planner should create 5 steps: check restrictions -> find recipes ->
  check nutrition -> create shopping list -> verify budget
- Use SystemMessage for LLM instructions, HumanMessage for the task
- The HITL checkpoint should print the meal plan and use input() to ask
  for approval. Default to "yes" for non-interactive testing.
- If the user rejects, ask for feedback and re-invoke the planner

Run: python week-04-advanced-patterns/exercises/exercise_01_meal_planning_react.py
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


planner_llm = get_llm(temperature=0)
executor_llm = get_llm(temperature=0.7)


# ==============================================================
# Step 2: Simulated Tools (PROVIDED -- do not modify)
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
# Step 3: Define the State (TODO 1)
# ==============================================================
# TODO 1: Define MealPlanState TypedDict with these fields:
#   - goal: str               (the meal planning request)
#   - constraints: dict       (budget, dietary restrictions, family size)
#   - plan: list              (list of step descriptions from planner)
#   - current_step: int       (index of step being executed)
#   - step_results: list      (results from each completed step)
#   - meal_plan: str          (the synthesized 7-day meal plan)
#   - approved: bool          (whether user approved the plan)
#   - feedback: str           (user feedback if rejected)
#   - iteration: int          (current iteration for safety)
#   - max_iterations: int     (safety limit)
#   - messages: Annotated[list, add_messages]  (for tool-calling)
#
# Hint: Look at PlanExecuteState in example_02_planning_langgraph.py
# Hint: The 'constraints' dict holds keys like 'budget', 'dietary', 'family_size'

# class MealPlanState(TypedDict):
#     pass  # Replace with your field definitions


# ==============================================================
# Step 4: Bind Tools to LLM
# ==============================================================

tools = [recipe_search, nutrition_check, budget_calculator]
executor_llm_with_tools = executor_llm.bind_tools(tools)


# ==============================================================
# Step 5: Implement the Planner Node (TODO 2)
# ==============================================================
# TODO 2: Implement the planner node
# Hint: Use SystemMessage to instruct the LLM to create a numbered plan
# Hint: The plan should have ~5 steps for meal planning:
#   1. Review dietary restrictions and family size
#   2. Search for recipes matching restrictions
#   3. Check nutrition balance across meals
#   4. Create a 7-day meal schedule
#   5. Calculate total grocery budget
# Hint: Parse the LLM response into a list of step strings
# Hint: Include state['constraints'] in the HumanMessage so the LLM knows the requirements
# Hint: If user provided feedback from a rejected plan, include it in the prompt
# See: example_02_planning_langgraph.py planner_node for reference

# def planner_node(state: MealPlanState) -> dict:
#     print(f"\n{'='*60}")
#     print("  PLANNER: Creating meal planning steps...")
#     print(f"{'='*60}")
#
#     # Build the prompt with goal, constraints, and any feedback
#     # ...
#
#     # Parse the numbered list into steps
#     # ...
#
#     return {
#         "plan": steps,
#         "current_step": 0,
#         "step_results": [],
#         "iteration": 0,
#         "messages": [],
#     }


# ==============================================================
# Step 6: Implement the Executor Node (TODO 3)
# ==============================================================
# TODO 3: Implement the executor node
# Hint: Get the current step from state['plan'][state['current_step']]
# Hint: Build context with the goal, constraints, and previous step results
# Hint: Use executor_llm_with_tools to let the LLM call tools
# Hint: Loop up to 5 times for tool calls, break when LLM stops calling tools
# Hint: Use ToolNode(tools) to execute tool calls
# See: example_02_planning_langgraph.py executor_node for the tool-calling loop

# def executor_node(state: MealPlanState) -> dict:
#     step_idx = state["current_step"]
#     step = state["plan"][step_idx]
#     iteration = state.get("iteration", 0) + 1
#
#     print(f"\n{'- '*30}")
#     print(f"  EXECUTOR: Step {step_idx + 1}/{len(state['plan'])}")
#     print(f"  Task: {step}")
#     print(f"{'- '*30}")
#
#     # Build context and execute with tool-calling loop
#     # ...
#
#     return {
#         "step_results": updated_results,
#         "iteration": iteration,
#         "messages": [],
#     }


# ==============================================================
# Step 7: Progress Check Node (PROVIDED)
# ==============================================================

def progress_check_node(state) -> dict:
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
# Step 8: Synthesize Node (PROVIDED)
# ==============================================================

def synthesize_node(state) -> dict:
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
# Step 9: Implement HITL Checkpoint (TODO 4)
# ==============================================================
# TODO 4: Implement the human-in-the-loop checkpoint
# Hint: Print the current meal_plan from state
# Hint: Use input() to ask the user: "Do you approve this meal plan? (yes/no): "
# Hint: If the user types "no", ask for feedback with another input()
# Hint: Handle non-interactive environments by defaulting to "yes" (use try/except on input)
# Hint: Return dict updating 'approved' (bool) and 'feedback' (str)
# Hint: If approved, set feedback to empty string
# Hint: If rejected, increment iteration and store the feedback for the replanner

# def hitl_checkpoint(state: MealPlanState) -> dict:
#     print(f"\n{'='*60}")
#     print("  HITL CHECKPOINT: Review the Meal Plan")
#     print(f"{'='*60}")
#     print(f"\n{state['meal_plan']}")
#     print(f"\n{'='*60}")
#
#     # Ask for user approval
#     # Handle non-interactive mode with try/except
#     # ...
#
#     return {
#         "approved": ...,
#         "feedback": ...,
#     }


# ==============================================================
# Step 10: Implement Routing Functions (TODO 5)
# ==============================================================
# TODO 5a: Implement should_continue for progress_check -> executor or synthesize
# Hint: Return "executor" if current_step < len(plan)
# Hint: Return "synthesize" if all steps done
# Hint: Return "synthesize" if iteration >= max_iterations (safety)

# def should_continue(state) -> str:
#     pass  # Your implementation here

# TODO 5b: Implement should_replan for hitl_checkpoint -> END or planner
# Hint: Return "done" if state['approved'] is True
# Hint: Return "replan" if not approved AND iteration < max_iterations
# Hint: Return "done" if max_iterations reached (force accept)

# def should_replan(state) -> str:
#     pass  # Your implementation here


# ==============================================================
# Step 11: Build the Graph (TODO 6)
# ==============================================================
# TODO 6: Wire up the complete graph
# Hint: Nodes: planner, executor, progress_check, synthesize, hitl_checkpoint
# Hint: Entry point: planner
# Hint: Edges:
#   planner -> executor (always)
#   executor -> progress_check (always)
#   progress_check -> should_continue -> {"executor": executor, "synthesize": synthesize}
#   synthesize -> hitl_checkpoint (always)
#   hitl_checkpoint -> should_replan -> {"done": END, "replan": planner}
# See: example_02_planning_langgraph.py build_plan_execute_graph for reference

# def build_meal_planning_graph():
#     graph = StateGraph(MealPlanState)
#     # Add nodes...
#     # Set entry point...
#     # Add edges...
#     # Add conditional edges...
#     return graph.compile()


# ==============================================================
# Run Function (PROVIDED)
# ==============================================================

def run_meal_planner(budget=75, dietary="none", family_size=4, max_iterations=3):
    """Run the meal planning agent.

    Args:
        budget: Weekly grocery budget in dollars
        dietary: Dietary restrictions (e.g., 'vegetarian', 'gluten-free', 'none')
        family_size: Number of people to plan meals for
        max_iterations: Max replan attempts if user rejects
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

    # Uncomment after implementing TODO 6:
    # app = build_meal_planning_graph()
    # result = app.invoke({
    #     "goal": goal,
    #     "constraints": constraints,
    #     "plan": [],
    #     "current_step": 0,
    #     "step_results": [],
    #     "meal_plan": "",
    #     "approved": False,
    #     "feedback": "",
    #     "iteration": 0,
    #     "max_iterations": max_iterations,
    #     "messages": [],
    # })
    # return result


# ==============================================================
# Test your implementation
# ==============================================================

if __name__ == "__main__":
    print("Exercise 1: Meal Planning Agent with Plan-Execute-Replan + HITL")
    print("=" * 65)

    # Test 1: Basic family meal plan
    print("\nTest 1: Budget family meal plan")
    print("-" * 40)
    run_meal_planner(budget=75, dietary="none", family_size=4)

    # Test 2: Vegetarian meal plan
    # print("\nTest 2: Vegetarian meal plan")
    # print("-" * 40)
    # run_meal_planner(budget=80, dietary="vegetarian", family_size=2)

    # Test 3: Gluten-free with larger family
    # print("\nTest 3: Gluten-free family meal plan")
    # print("-" * 40)
    # run_meal_planner(budget=100, dietary="gluten-free", family_size=6)

    print("\n(Uncomment the graph code and tests above after implementing!)")
    print("\nExpected behavior:")
    print("  - Planner creates 5 steps for the meal planning task")
    print("  - Executor uses tools to research recipes, nutrition, and budget")
    print("  - Synthesizer creates a formatted 7-day meal plan")
    print("  - HITL checkpoint asks for your approval")
    print("  - If rejected, the agent replans with your feedback")

    print("\nSuccess Criteria:")
    print("  - Plan covers 7 days with breakfast, lunch, dinner")
    print("  - Budget stays within the specified constraint")
    print("  - Dietary restrictions are respected")
    print("  - At least 1 HITL checkpoint works (approve or reject)")
