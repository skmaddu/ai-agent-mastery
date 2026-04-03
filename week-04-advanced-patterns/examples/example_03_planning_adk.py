import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3: Plan-Execute Pattern in Google ADK
===============================================
The same dinner party planning scenario from Example 2, but implemented
using Google ADK's agent architecture.

ADK Approach:
  - Two LlmAgents: a Planner and an Executor
  - The Planner decomposes a goal into numbered steps
  - The Executor has tools bound and handles individual steps
  - Coordination lives in a Python async loop, NOT in a graph

Comparison:
  LangGraph: Explicit StateGraph with planner/executor nodes + conditional
             edges. The graph IS the control flow. Replanning is a graph edge.
  ADK:       Separate LlmAgents coordinated by application code (async loop).
             More flexible — you control the loop with plain Python.
             Simpler for linear plan-execute, but you manage state yourself.

Key Insight: In ADK, the "graph" is your Python code. You decide when to
call which agent, how to pass context, and when to stop. This is more
explicit but gives you full control over the planning loop.

Run: python week-04-advanced-patterns/examples/example_03_planning_adk.py
"""

import asyncio
import logging
import os
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Step 1: Define Tools as Plain Functions
# ==============================================================
# ADK reads function name, docstring, and type hints to create
# the tool schema automatically. No @tool decorator needed.
# These are the same simulated tools from Example 1's concepts,
# now callable by the executor agent.

def recipe_search(cuisine_type: str) -> str:
    """Search for recipes by cuisine type or course category.

    Use this to find recipe options for a specific course or cuisine.

    Args:
        cuisine_type: The type of cuisine or course to search for
                      (e.g., 'appetizer', 'main', 'dessert', 'italian',
                       'gluten-free', 'vegetarian')

    Returns:
        Available recipes with cost and dietary information
    """
    recipes = {
        "appetizer": [
            "Bruschetta - $12 (serves 8, contains gluten, 20 min prep)",
            "Hummus Platter - $10 (serves 10, contains sesame, 15 min prep)",
            "Caprese Salad - $14 (serves 8, contains dairy, 10 min prep)",
            "Vegetable Spring Rolls - $11 (serves 8, gluten-free, 25 min prep)",
        ],
        "main": [
            "Chicken Stir-Fry - $28 (serves 8, gluten-free, 45 min prep)",
            "Pasta Primavera - $22 (serves 8, contains gluten, 35 min prep)",
            "Grilled Salmon - $38 (serves 8, gluten-free, 40 min prep)",
            "Rice Bowl Bar - $25 (serves 8, gluten-free, 30 min prep)",
        ],
        "dessert": [
            "Fruit Salad - $10 (serves 10, gluten-free & dairy-free, 15 min prep)",
            "Chocolate Cake - $15 (serves 10, contains gluten & dairy, 60 min prep)",
            "Ice Cream Sundae Bar - $12 (serves 8, gluten-free, contains dairy, 10 min prep)",
            "Coconut Rice Pudding - $9 (serves 8, gluten-free & dairy-free, 25 min prep)",
        ],
    }

    # Match by course type or keyword
    query = cuisine_type.lower()
    for category, items in recipes.items():
        if category in query or query in category:
            return f"Recipes for '{cuisine_type}':\n" + "\n".join(f"  - {r}" for r in items)

    # Keyword search across all recipes
    matches = []
    for category, items in recipes.items():
        for item in items:
            if query in item.lower():
                matches.append(f"  - [{category}] {item}")
    if matches:
        return f"Recipes matching '{cuisine_type}':\n" + "\n".join(matches)

    return f"No recipes found for '{cuisine_type}'. Try: appetizer, main, dessert, gluten-free."


def nutrition_check(meal_name: str) -> str:
    """Check nutritional information and dietary compatibility for a meal.

    Use this to verify if a meal meets dietary requirements like
    gluten-free, dairy-free, or vegetarian.

    Args:
        meal_name: Name of the meal to check (e.g., 'Chicken Stir-Fry')

    Returns:
        Nutritional info including calories, allergens, and dietary flags
    """
    nutrition_db = {
        "bruschetta": "Calories: 180/serving | Allergens: gluten, dairy | NOT gluten-free",
        "hummus platter": "Calories: 150/serving | Allergens: sesame | Gluten-free, dairy-free",
        "caprese salad": "Calories: 200/serving | Allergens: dairy | Gluten-free, vegetarian",
        "vegetable spring rolls": "Calories: 160/serving | Allergens: none | Gluten-free, vegan",
        "chicken stir-fry": "Calories: 350/serving | Allergens: soy | Gluten-free (use tamari)",
        "pasta primavera": "Calories: 400/serving | Allergens: gluten, dairy | NOT gluten-free",
        "grilled salmon": "Calories: 380/serving | Allergens: fish | Gluten-free, dairy-free",
        "rice bowl bar": "Calories: 320/serving | Allergens: soy | Gluten-free, customizable",
        "fruit salad": "Calories: 120/serving | Allergens: none | Gluten-free, dairy-free, vegan",
        "chocolate cake": "Calories: 450/serving | Allergens: gluten, dairy, eggs | NOT gluten-free",
        "ice cream sundae bar": "Calories: 300/serving | Allergens: dairy | Gluten-free",
        "coconut rice pudding": "Calories: 220/serving | Allergens: none | Gluten-free, dairy-free, vegan",
    }

    key = meal_name.lower().strip()
    for name, info in nutrition_db.items():
        if name in key or key in name:
            return f"Nutrition for '{meal_name}': {info}"

    return f"No nutrition data found for '{meal_name}'."


def budget_calculator(items: str) -> str:
    """Calculate the total cost of menu items and check against a budget.

    Use this to add up costs and see if the total fits within a budget.

    Args:
        items: Comma-separated list of items with costs,
               e.g., 'Chicken Stir-Fry $28, Fruit Salad $10, Hummus $10'

    Returns:
        Itemized breakdown with total cost and budget assessment
    """
    # Parse items and extract costs
    total = 0.0
    parsed = []
    for item in items.split(","):
        item = item.strip()
        # Extract dollar amounts using regex
        price_match = re.search(r'\$(\d+(?:\.\d{2})?)', item)
        if price_match:
            cost = float(price_match.group(1))
            total += cost
            parsed.append(f"  - {item}")
        elif item:
            parsed.append(f"  - {item} (no price found)")

    result = "Budget Calculation:\n"
    result += "\n".join(parsed) if parsed else "  No items parsed"
    result += f"\n  TOTAL: ${total:.2f}"
    result += f"\n  For 8 guests: ${total/8:.2f} per person"

    if total <= 100:
        result += f"\n  STATUS: Within $100 budget (${100 - total:.2f} remaining)"
    else:
        result += f"\n  STATUS: OVER $100 budget by ${total - 100:.2f}!"

    return result


# ==============================================================
# Step 2: Create the Planner Agent
# ==============================================================
# The planner's job is ONLY to decompose goals into numbered steps.
# It does NOT have tools — it just plans.
#
# Compare to LangGraph: In LangGraph, this would be a graph node
# that writes to state["plan"]. Here, it's a standalone agent.

planner_agent = LlmAgent(
    name="planner",
    model=os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview"),
    instruction="""You are a planning specialist. Your ONLY job is to break down
goals into clear, numbered steps.

Rules:
1. Output a numbered list (1. 2. 3. etc.) of concrete steps
2. Each step should be a single actionable task
3. Keep to 3-5 steps (no more than 5)
4. Consider constraints mentioned in the goal (budget, dietary needs, etc.)
5. Steps should be in logical order (dependencies first)
6. When asked to synthesize results, create a clear summary instead

For dinner party planning, consider:
- Dietary restrictions must be checked first
- Menu selection depends on dietary info
- Budget verification comes after menu selection
- Timeline planning comes last""",
    description="Decomposes complex goals into ordered, actionable steps.",
)


# ==============================================================
# Step 3: Create the Executor Agent
# ==============================================================
# The executor has tools bound and handles individual steps.
# It receives one step at a time, plus context from previous steps.
#
# Compare to LangGraph: In LangGraph, the executor node calls
# tools via ToolNode with conditional edges. Here, ADK's Runner
# handles the tool-calling loop internally.

executor_agent = LlmAgent(
    name="executor",
    model=os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview"),
    instruction="""You are a task executor with access to planning tools.

For each step you receive:
1. Read the step description carefully
2. Use the appropriate tool(s) to complete it
3. Consider any previous results provided as context
4. Return a clear, concise result

Available tools:
- recipe_search: Find recipes by course type or cuisine
- nutrition_check: Verify dietary compatibility of a meal
- budget_calculator: Calculate costs and check budget

When selecting menu items, prefer options that are:
- Within budget constraints
- Compatible with dietary restrictions mentioned
- Practical for the number of guests

Always explain your reasoning briefly.""",
    tools=[recipe_search, nutrition_check, budget_calculator],
    description="Executes individual plan steps using available tools.",
)


# ==============================================================
# Step 4: Helper to Run an ADK Agent
# ==============================================================

async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent with a message and return the response text.

    Each call creates a fresh session so agents don't carry over
    context from previous calls (we manage context ourselves in
    the plan-execute loop).

    Includes retry logic for transient API errors (503, rate limits).

    Compare to LangGraph: In LangGraph, state flows through the
    graph automatically. Here, we pass context manually via the
    message string — more explicit, more control.
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="planning_demo",
                session_service=session_service,
            )

            session = await session_service.create_session(
                app_name="planning_demo",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=message)],
                ),
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        result_text = event.content.parts[0].text

            return result_text
        except Exception as e:
            if attempt < retries:
                wait = attempt * 10
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [ERROR] All {retries} attempts failed: {e}")
                return f"[Error: API temporarily unavailable after {retries} retries]"


# ==============================================================
# Step 5: Parse Plan into Steps
# ==============================================================

def parse_plan(plan_text: str) -> list:
    """Extract numbered steps from the planner's output.

    Handles formats like:
      1. Do something
      2. Do another thing
      1) Alternative format
      Step 1: Yet another format
    """
    steps = []
    for line in plan_text.strip().split("\n"):
        line = line.strip()
        # Match "1. ...", "1) ...", "Step 1: ...", "- ..."
        match = re.match(r'^(?:\d+[\.\)]\s*|Step\s+\d+[:\.\)]\s*|-\s+)(.*)', line)
        if match:
            step_text = match.group(1).strip()
            if step_text and len(step_text) > 5:  # Skip very short lines
                steps.append(step_text)

    if not steps:
        # Fallback: treat non-empty lines as steps
        steps = [line.strip() for line in plan_text.strip().split("\n")
                 if line.strip() and len(line.strip()) > 5]

    return steps


# ==============================================================
# Step 6: The Plan-Execute Loop
# ==============================================================
# This is where ADK differs most from LangGraph:
#   LangGraph: The loop is encoded as graph edges (planner -> executor
#              -> should_continue -> planner or end)
#   ADK:       The loop is a Python for-loop. You call agents
#              explicitly and manage the results list yourself.
#
# Both approaches work. ADK is more "code-first" while LangGraph
# is more "graph-first".

MAX_STEPS = 5  # Safety limit to prevent runaway plans


async def main():
    """Run the Plan-Execute pattern with ADK agents."""
    print("Example 3: Plan-Execute Pattern in ADK")
    print("=" * 60)

    goal = "Plan a dinner party for 8 people with a $100 budget, one guest is gluten-free"

    # ----------------------------------------------------------
    # Phase 1: PLAN — Ask the planner to decompose the goal
    # ----------------------------------------------------------
    # In LangGraph, this would be the planner node writing to
    # state["plan"]. Here, we just call the planner agent.
    print(f"\nGoal: {goal}")
    print(f"\n{'- '*30}")
    print("  PHASE 1: Planning")
    print(f"{'- '*30}")

    plan_text = await run_agent(planner_agent, f"Break this goal into steps: {goal}")

    print(f"\n  Planner output:")
    for line in plan_text.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

    steps = parse_plan(plan_text)

    # Safety limit
    if len(steps) > MAX_STEPS:
        print(f"\n  [SAFETY] Truncating plan from {len(steps)} to {MAX_STEPS} steps")
        steps = steps[:MAX_STEPS]

    print(f"\n  Parsed {len(steps)} steps from plan")

    # ----------------------------------------------------------
    # Phase 2: EXECUTE — Run each step through the executor
    # ----------------------------------------------------------
    # In LangGraph, this would be the executor node + tool nodes
    # with conditional edges. Here, it's a simple for-loop.
    print(f"\n{'- '*30}")
    print("  PHASE 2: Execution")
    print(f"{'- '*30}")

    results = []

    for i, step in enumerate(steps):
        print(f"\n  --- Step {i + 1}/{len(steps)}: {step[:80]} ---")

        # Build context from previous results
        # This is what LangGraph does automatically via state —
        # in ADK, we build the context string ourselves.
        context_parts = [f"You are executing step {i + 1} of a plan."]
        context_parts.append(f"Step: {step}")
        context_parts.append(f"Overall goal: {goal}")

        if results:
            context_parts.append("\nPrevious step results:")
            for j, prev in enumerate(results):
                # Include a summary of each previous result
                summary = prev[:200] if len(prev) > 200 else prev
                context_parts.append(f"  Step {j + 1}: {summary}")

        context_message = "\n".join(context_parts)

        # Call the executor agent with tools
        result = await run_agent(executor_agent, context_message)
        results.append(result)

        # Print result (truncated for readability)
        for line in result.strip().split("\n"):
            if line.strip():
                print(f"    {line.strip()[:120]}")

    # ----------------------------------------------------------
    # Phase 3: SYNTHESIZE — Ask the planner to create final output
    # ----------------------------------------------------------
    # In LangGraph, this could be a final node that reads from
    # state. Here, we call the planner again with all results.
    print(f"\n{'- '*30}")
    print("  PHASE 3: Synthesis")
    print(f"{'- '*30}")

    synthesis_prompt = (
        f"Original goal: {goal}\n\n"
        f"All step results:\n"
    )
    for i, (step, result) in enumerate(zip(steps, results)):
        synthesis_prompt += f"\nStep {i + 1} ({step}):\n{result}\n"
    synthesis_prompt += (
        "\nCreate a final comprehensive dinner party plan based on these results. "
        "Include: menu with costs, dietary notes, and a brief cooking timeline."
    )

    synthesis = await run_agent(planner_agent, synthesis_prompt)

    print(f"\n  Final Plan:")
    print(f"  {'='*50}")
    for line in synthesis.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")
    print(f"  {'='*50}")

    # ----------------------------------------------------------
    # Framework Comparison
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("LangGraph vs ADK -- Plan-Execute Comparison:")
    print(f"{'='*60}")
    print("  LangGraph:")
    print("    - Plan lives in graph state (state['plan'], state['results'])")
    print("    - Execution loop is a conditional edge (steps remain?)")
    print("    - Replanning is another edge back to the planner node")
    print("    - Visual graph structure, easy to trace in Phoenix")
    print()
    print("  ADK:")
    print("    - Plan lives in Python variables (plan_text, results list)")
    print("    - Execution loop is a Python for-loop")
    print("    - Replanning is just another agent call")
    print("    - More flexible — any Python logic can control the flow")
    print()
    print("  When to choose which:")
    print("    - LangGraph: When you want visual debugging, checkpointing,")
    print("                 or complex branching (the graph helps)")
    print("    - ADK:       When you want simplicity and full Python control")
    print("                 (the code IS the control flow)")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
