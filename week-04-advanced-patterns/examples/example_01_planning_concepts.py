import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 1: Planning Patterns — Concepts (No LLM Required)
==========================================================
Why does a single-shot LLM call fail on complex tasks?

Imagine asking someone to "plan a dinner party for 8 people" and expecting
the perfect answer in one breath. You wouldn't — you'd break it down:
check dietary needs, pick recipes, make a shopping list, plan the timeline.

Planning patterns do exactly this for LLM agents:
  1. DECOMPOSE the goal into ordered sub-tasks
  2. EXECUTE each sub-task (with tools if needed)
  3. CHECK progress and REPLAN if necessary

This example demonstrates planning concepts in pure Python — no LLM needed.
It shows WHY planning matters before we implement it with real LLMs.

Key Concepts (Section 1 of the Research Bible):
  - Plan-Execute pattern: separate planning from execution
  - Task decomposition: break complex goals into verifiable sub-steps
  - Dependency ordering: some tasks depend on others
  - Replanning: adjust the plan when things change

Run: python week-04-advanced-patterns/examples/example_01_planning_concepts.py
"""

from dataclasses import dataclass, field
from typing import Optional


# ================================================================
# PART 1: Why Planning Matters
# ================================================================
# Without planning, an agent tries to solve everything in one shot.
# This leads to three fundamental problems:
#
# 1. COMPOSITIONAL COMPLEXITY — complex tasks require combining
#    multiple pieces of information in the right order.
#
# 2. ERROR COMPOUNDING — without intermediate checks, small errors
#    in step 1 become catastrophic by step 10.
#
# 3. VERIFICATION IS EASIER THAN GENERATION — it's easier to check
#    "is this meal within budget?" than to generate a budget-optimal
#    meal from scratch.
#
# Planning exploits this asymmetry: generate small steps, verify each.

def demo_why_planning_matters():
    """Compare single-shot vs. planned approach to a complex task."""
    print("=" * 60)
    print("PART 1: Why Planning Matters")
    print("=" * 60)

    # The task: "Plan a dinner party for 8 with a $100 budget"
    goal = "Plan a dinner party for 8 people with a $100 budget"

    # ---- Single-shot approach (what a naive agent does) ----
    print(f"\nGoal: {goal}")
    print("\n--- Single-Shot Approach (No Planning) ---")
    print("Agent tries to answer everything at once:")
    single_shot_result = (
        "For a dinner party for 8, I'd suggest making pasta with "
        "garlic bread and a salad. Buy pasta ($5), sauce ($3), "
        "garlic bread ($4), and salad mix ($6). Total: about $18."
    )
    print(f"  Result: {single_shot_result}")
    print("  Problems:")
    print("    - No dietary restrictions considered")
    print("    - No timeline for cooking")
    print("    - No drink plan")
    print("    - Budget not fully used (wasteful simplicity)")
    print("    - Single point of failure — if anything is wrong,")
    print("      the whole plan is bad")

    # ---- Planned approach ----
    print("\n--- Planned Approach (Decomposition) ---")
    print("Agent breaks the goal into verifiable sub-tasks:")
    steps = [
        "1. Check dietary restrictions for all 8 guests",
        "2. Plan appetizer course (budget: $20)",
        "3. Plan main course (budget: $40)",
        "4. Plan dessert (budget: $15)",
        "5. Plan drinks (budget: $15)",
        "6. Create shopping list from all recipes",
        "7. Verify total is within $100 budget",
        "8. Create cooking timeline (what to start when)",
    ]
    for step in steps:
        print(f"    {step}")
    print("\n  Benefits:")
    print("    - Each step is small enough to verify")
    print("    - Errors caught early (step 7 checks budget)")
    print("    - Can replan if a step fails (e.g., allergies)")
    print("    - Parallel execution possible (steps 2-5)")
    print()


# ================================================================
# PART 2: Task Data Structure
# ================================================================
# A Task represents one step in a plan. It has:
#   - A unique ID for tracking
#   - A description of what to do
#   - Dependencies (which tasks must finish first)
#   - Status and result fields
#
# This is the "scratchpad" the agent carries through execution.

@dataclass
class Task:
    """One step in a plan.

    Dependencies define ordering: a task can only execute after
    all its dependencies are completed. This naturally creates
    a DAG (directed acyclic graph) of execution.
    """
    id: str
    description: str
    dependencies: list = field(default_factory=list)
    status: str = "pending"      # pending | running | completed | failed
    result: Optional[str] = None

    def is_ready(self, completed_ids: set) -> bool:
        """A task is ready when all its dependencies are completed."""
        return all(dep in completed_ids for dep in self.dependencies)


def demo_task_structure():
    """Show how tasks form a dependency graph."""
    print("=" * 60)
    print("PART 2: Task Data Structure")
    print("=" * 60)

    # Create a simple plan with dependencies
    tasks = [
        Task(id="check_diet", description="Check dietary restrictions"),
        Task(id="plan_appetizer", description="Plan appetizer course",
             dependencies=["check_diet"]),
        Task(id="plan_main", description="Plan main course",
             dependencies=["check_diet"]),
        Task(id="plan_dessert", description="Plan dessert",
             dependencies=["check_diet"]),
        Task(id="shopping_list", description="Create shopping list",
             dependencies=["plan_appetizer", "plan_main", "plan_dessert"]),
        Task(id="verify_budget", description="Verify total within budget",
             dependencies=["shopping_list"]),
    ]

    print("\nTask Dependency Graph:")
    print("  check_diet")
    print("    ├── plan_appetizer ──┐")
    print("    ├── plan_main ───────┼── shopping_list ── verify_budget")
    print("    └── plan_dessert ────┘")
    print()

    # Show which tasks are ready at start
    completed = set()
    print("Initially ready tasks:", end=" ")
    ready = [t for t in tasks if t.is_ready(completed)]
    print(", ".join(t.id for t in ready))

    # Simulate completing check_diet
    completed.add("check_diet")
    print("After check_diet completes:", end=" ")
    ready = [t for t in tasks if t.is_ready(completed) and t.id not in completed]
    print(", ".join(t.id for t in ready))

    # These three can run in PARALLEL — a key advantage of planning
    print("  ^ These 3 tasks can run in PARALLEL!")
    print()


# ================================================================
# PART 3: Goal Decomposition
# ================================================================
# Decomposition is the planner's main job: take a high-level goal
# and break it into concrete, ordered sub-tasks.
#
# In a real agent, an LLM does this. Here we simulate it to show
# the mechanics clearly.

# Simulated knowledge base for our "tools"
RECIPE_DB = {
    "appetizer": {
        "bruschetta": {"servings": 8, "cost": 12, "prep_time": 20, "allergens": ["gluten"]},
        "hummus_platter": {"servings": 10, "cost": 10, "prep_time": 15, "allergens": ["sesame"]},
        "caprese_salad": {"servings": 8, "cost": 14, "prep_time": 10, "allergens": ["dairy"]},
    },
    "main": {
        "chicken_stir_fry": {"servings": 8, "cost": 28, "prep_time": 45, "allergens": []},
        "pasta_primavera": {"servings": 8, "cost": 22, "prep_time": 35, "allergens": ["gluten"]},
        "grilled_salmon": {"servings": 8, "cost": 38, "prep_time": 40, "allergens": ["fish"]},
    },
    "dessert": {
        "fruit_salad": {"servings": 10, "cost": 10, "prep_time": 15, "allergens": []},
        "chocolate_cake": {"servings": 10, "cost": 15, "prep_time": 60, "allergens": ["gluten", "dairy"]},
        "ice_cream_sundae": {"servings": 8, "cost": 12, "prep_time": 10, "allergens": ["dairy"]},
    },
}

GUEST_RESTRICTIONS = {
    "Alice": [],
    "Bob": ["gluten"],
    "Carol": [],
    "Dave": ["dairy"],
    "Eve": [],
    "Frank": [],
    "Grace": ["fish"],
    "Hank": [],
}


def tool_check_dietary(guests: dict) -> dict:
    """Simulated tool: Check all dietary restrictions."""
    restrictions = set()
    for name, allergies in guests.items():
        restrictions.update(allergies)
    return {
        "total_guests": len(guests),
        "restrictions": list(restrictions),
        "details": {name: allergies for name, allergies in guests.items() if allergies},
    }


def tool_find_recipes(course: str, avoid_allergens: list, budget: float) -> list:
    """Simulated tool: Find recipes that fit constraints."""
    recipes = RECIPE_DB.get(course, {})
    suitable = []
    for name, info in recipes.items():
        # Check allergens
        if any(a in info["allergens"] for a in avoid_allergens):
            continue
        # Check budget
        if info["cost"] <= budget:
            suitable.append({"name": name, **info})
    return suitable


def tool_calculate_budget(items: list) -> dict:
    """Simulated tool: Calculate total cost and check budget."""
    total = sum(item.get("cost", 0) for item in items)
    return {"total_cost": total, "items": len(items)}


def decompose_goal(goal: str) -> list:
    """Simulate an LLM decomposing a goal into tasks.

    In a real agent, this would be an LLM call like:
      "Break this goal into 5-7 concrete steps: {goal}"

    Here we return a pre-defined plan to show the mechanics.
    """
    return [
        Task(id="check_diet",
             description="Check dietary restrictions for all guests"),
        Task(id="find_appetizer",
             description="Find appetizer recipes (budget: $15)",
             dependencies=["check_diet"]),
        Task(id="find_main",
             description="Find main course recipes (budget: $35)",
             dependencies=["check_diet"]),
        Task(id="find_dessert",
             description="Find dessert recipes (budget: $12)",
             dependencies=["check_diet"]),
        Task(id="verify_budget",
             description="Verify total cost within $100 budget",
             dependencies=["find_appetizer", "find_main", "find_dessert"]),
        Task(id="create_timeline",
             description="Create cooking timeline",
             dependencies=["verify_budget"]),
    ]


# ================================================================
# PART 4: Plan Executor
# ================================================================
# The executor processes tasks in dependency order, running each
# task's "tool" and recording the result. If a task fails, it can
# trigger replanning.
#
# This is the core of the Plan-Execute pattern:
#
#   ┌─────────┐     ┌──────────┐     ┌──────────────┐
#   │ Planner │────>│ Executor │────>│ Progress     │
#   │ (decomp)│     │ (run one)│     │ Check        │
#   └─────────┘     └──────────┘     └──────┬───────┘
#        ^                                   │
#        │              ┌────────────────────┤
#        │              │ steps remain       │ all done
#        └──────────────┘                    v
#                                      ┌──────────┐
#                                      │ Final    │
#                                      │ Output   │
#                                      └──────────┘

class PlanExecutor:
    """Executes a plan respecting task dependencies.

    This is a simplified version of what LangGraph and ADK do
    under the hood when you build planning agents.
    """

    def __init__(self, tasks: list, budget: float = 100.0):
        self.tasks = {t.id: t for t in tasks}
        self.completed_ids = set()
        self.context = {"budget": budget, "allergens": [], "selected_recipes": []}
        self.execution_log = []

    def execute_task(self, task: Task) -> str:
        """Execute a single task using the appropriate tool."""
        task.status = "running"
        self.execution_log.append(f"  [{task.id}] Starting: {task.description}")

        if task.id == "check_diet":
            result = tool_check_dietary(GUEST_RESTRICTIONS)
            self.context["allergens"] = result["restrictions"]
            output = f"Found restrictions: {result['restrictions']} for {result['total_guests']} guests"

        elif task.id.startswith("find_"):
            course = task.id.replace("find_", "")
            budget_map = {"appetizer": 15, "main": 35, "dessert": 12}
            budget = budget_map.get(course, 20)
            recipes = tool_find_recipes(course, self.context["allergens"], budget)
            if recipes:
                chosen = recipes[0]  # Pick first suitable recipe
                self.context["selected_recipes"].append(chosen)
                output = f"Selected: {chosen['name']} (${chosen['cost']}, {chosen['prep_time']}min)"
            else:
                output = f"No suitable {course} found within constraints!"

        elif task.id == "verify_budget":
            budget_result = tool_calculate_budget(self.context["selected_recipes"])
            remaining = self.context["budget"] - budget_result["total_cost"]
            output = f"Total: ${budget_result['total_cost']} / ${self.context['budget']} (${remaining} remaining)"
            if remaining < 0:
                output += " — OVER BUDGET! Replanning needed."

        elif task.id == "create_timeline":
            recipes = self.context["selected_recipes"]
            # Sort by prep time (longest first)
            sorted_recipes = sorted(recipes, key=lambda r: r["prep_time"], reverse=True)
            timeline = []
            time_offset = 0
            for r in sorted_recipes:
                timeline.append(f"T-{90 - time_offset}min: Start {r['name']} ({r['prep_time']}min)")
                time_offset += r["prep_time"] + 5  # 5 min buffer between tasks
            output = "Timeline: " + " → ".join(timeline)

        else:
            output = f"Unknown task: {task.id}"

        task.status = "completed"
        task.result = output
        self.completed_ids.add(task.id)
        self.execution_log.append(f"  [{task.id}] Result: {output}")
        return output

    def run(self) -> dict:
        """Execute all tasks in dependency order."""
        max_rounds = 20  # Safety limit
        round_count = 0

        while len(self.completed_ids) < len(self.tasks) and round_count < max_rounds:
            round_count += 1
            # Find all tasks that are ready to run
            ready_tasks = [
                t for t in self.tasks.values()
                if t.status == "pending" and t.is_ready(self.completed_ids)
            ]

            if not ready_tasks:
                self.execution_log.append("  [WARN] No tasks ready — possible deadlock!")
                break

            # Execute ready tasks (in a real system, these could run in parallel)
            for task in ready_tasks:
                self.execute_task(task)

        return {
            "completed": len(self.completed_ids),
            "total": len(self.tasks),
            "recipes": self.context["selected_recipes"],
            "log": self.execution_log,
        }


def demo_plan_execution():
    """Run the full plan-execute pattern."""
    print("=" * 60)
    print("PART 3 & 4: Goal Decomposition + Plan Execution")
    print("=" * 60)

    goal = "Plan a dinner party for 8 people with a $100 budget"
    print(f"\nGoal: {goal}")

    # Step 1: Decompose
    print("\n--- Step 1: Decompose Goal into Tasks ---")
    tasks = decompose_goal(goal)
    for t in tasks:
        deps = f" (after: {', '.join(t.dependencies)})" if t.dependencies else ""
        print(f"  [{t.id}] {t.description}{deps}")

    # Step 2: Execute
    print("\n--- Step 2: Execute Plan ---")
    executor = PlanExecutor(tasks, budget=100.0)
    result = executor.run()

    for log_line in result["log"]:
        print(log_line)

    print(f"\n--- Result: {result['completed']}/{result['total']} tasks completed ---")
    if result["recipes"]:
        print("\nFinal Menu:")
        total_cost = 0
        for r in result["recipes"]:
            print(f"  - {r['name']}: ${r['cost']} ({r['prep_time']} min prep)")
            total_cost += r["cost"]
        print(f"  Total: ${total_cost}")
    print()


# ================================================================
# PART 5: Replanning — What Happens When Things Change
# ================================================================
# Real-world plans need adjustment. Maybe a recipe is out of stock,
# a guest cancels, or the budget changes. The replan step checks
# progress and updates the plan accordingly.

def demo_replanning():
    """Show how replanning works when constraints change."""
    print("=" * 60)
    print("PART 5: Replanning When Things Change")
    print("=" * 60)

    print("\n--- Scenario: Budget drops from $100 to $50 mid-plan ---")
    print()

    # First plan with full budget
    tasks = decompose_goal("dinner party")
    executor = PlanExecutor(tasks, budget=50.0)  # Tighter budget!
    result = executor.run()

    for log_line in result["log"]:
        print(log_line)

    # Check if we're over budget
    total_cost = sum(r["cost"] for r in result["recipes"])
    if total_cost > 50:
        print(f"\n  [REPLAN] Over budget (${total_cost} > $50)!")
        print("  In a real agent, the replanner would:")
        print("    1. Check which recipes can be swapped for cheaper ones")
        print("    2. Re-run the find_* tasks with lower budget limits")
        print("    3. Verify the new total is within budget")
    else:
        print(f"\n  Budget OK: ${total_cost} <= $50")

    print()
    print("Key Insight: Replanning is just decomposition + execution again,")
    print("but with updated context (what's already done + new constraints).")
    print()


# ================================================================
# PART 6: The Three Flavors of Planning
# ================================================================

def demo_planning_flavors():
    """Compare ReAct, Plan-Execute, and Decomposition+Delegation."""
    print("=" * 60)
    print("PART 6: Three Flavors of Planning")
    print("=" * 60)

    print("""
┌──────────────────────────────────────────────────────────┐
│                    PLANNING PATTERNS                     │
├──────────────┬───────────────────────────────────────────┤
│ ReAct        │ Think → Act → Observe → Repeat           │
│              │ Best for: 1-5 step tasks                  │
│              │ Like: A chef who tastes as they cook       │
├──────────────┼───────────────────────────────────────────┤
│ Plan-Execute │ Plan all steps → Execute one by one       │
│              │ Best for: 5+ step tasks needing structure  │
│              │ Like: A head chef writing the menu first   │
├──────────────┼───────────────────────────────────────────┤
│ Decompose +  │ Break into independent sub-goals          │
│ Delegate     │ Best for: tasks with parallel sub-tasks   │
│              │ Like: A catering manager with teams        │
└──────────────┴───────────────────────────────────────────┘

When to use which:
  - Simple lookup/calculation → Single-shot (no planning needed)
  - Research question → ReAct (search, read, search more)
  - "Write a report on X" → Plan-Execute (outline, then fill in)
  - "Organize an event" → Decompose + Delegate (venue team,
    catering team, entertainment team work in parallel)

Rule of thumb: Start with the SIMPLEST pattern that works.
Planning is not overhead — it's the single highest-leverage
investment for improving agent reliability.
""")


# ================================================================
# Main: Run all demos
# ================================================================

if __name__ == "__main__":
    print()
    print("Example 1: Planning Patterns — Concepts")
    print("=" * 60)
    print("No LLM required — pure Python demonstration")
    print()

    demo_why_planning_matters()
    demo_task_structure()
    demo_plan_execution()
    demo_replanning()
    demo_planning_flavors()

    print("=" * 60)
    print("Next Examples:")
    print("  example_02  — Plan-Execute in LangGraph (basic)")
    print("  example_02b — Plan-Execute-REPLAN (evaluation + replan loop)")
    print("  example_02c — ReAct pattern (Think → Act → Observe)")
    print("  example_02d — Decompose + Delegate (parallel specialists)")
    print("  example_03  — Plan-Execute in ADK")
    print("=" * 60)
