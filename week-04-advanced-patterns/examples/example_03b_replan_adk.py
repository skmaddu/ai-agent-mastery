import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3b: Plan-Execute-REPLAN Pattern in Google ADK
=====================================================
The same Plan-Execute-REPLAN pattern from Example 2b (LangGraph), but
implemented using Google ADK's agent architecture.

Example 2b showed how LangGraph encodes the replan loop as GRAPH EDGES:
  planner -> executor -> evaluator -> (replan edge) -> planner

This example shows the ADK approach: the replan loop is PYTHON CODE.
Four separate LlmAgents are coordinated by an async orchestration
function — no graph, no state machine, just explicit Python control flow.

ADK vs LangGraph for Replanning:
  LangGraph: Replan loop is a conditional GRAPH EDGE. Evaluator writes
    score to state; routing function decides replan vs synthesize.
    Visual, checkpointable, but the graph gets complex.
  ADK: Replan loop is a Python WHILE LOOP. Evaluator returns text;
    we parse the score and use if/else. Explicit, flexible, but less visual.

Key Insight: For REPLAN specifically, ADK is often cleaner because the
loop logic (score < 7 AND replans left) is natural Python. LangGraph
requires routing functions + graph edges but is easier to checkpoint.

Flow:
  1. PLANNER agent decomposes goal into 3 steps
  2. EXECUTOR agent (with tools) executes each step sequentially
  3. EVALUATOR agent scores results (1-10) and identifies issues
  4. If score < 7 and replans remain:
       REPLANNER creates improved plan using feedback → back to step 2
  5. If score >= 7 or no replans left:
       SYNTHESIZER creates final comprehensive output

Run: python week-04-advanced-patterns/examples/example_03b_replan_adk.py
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
# ADK reads function name, docstring, and type hints to build
# the tool schema. No @tool decorator needed (unlike LangGraph).
# These are the same simulated tools from Example 2b, adapted
# to ADK's plain-function convention.

def recipe_search(cuisine_type: str) -> str:
    """Search for recipes by cuisine type or course category.

    Args:
        cuisine_type: Type of cuisine or course (e.g., 'appetizer', 'main', 'dessert', 'budget')

    Returns:
        Available recipes with cost, serving size, and dietary information
    """
    recipes = {
        "appetizer": ("Appetizer options:\n"
            "  1. Bruschetta (serves 8) - $12, contains gluten\n"
            "  2. Hummus Platter (serves 10) - $10, contains sesame\n"
            "  3. Caprese Salad (serves 8) - $14, contains dairy\n"
            "  4. Spring Rolls (serves 8) - $11, gluten-free"),
        "main": ("Main course options:\n"
            "  1. Chicken Stir-Fry (serves 8) - $28, gluten-free\n"
            "  2. Pasta Primavera (serves 8) - $22, contains gluten\n"
            "  3. Grilled Salmon (serves 8) - $38, gluten-free but expensive\n"
            "  4. Rice Bowl Bar (serves 8) - $25, gluten-free"),
        "dessert": ("Dessert options:\n"
            "  1. Fruit Salad (serves 10) - $10, gluten-free, dairy-free\n"
            "  2. Chocolate Cake (serves 10) - $15, contains gluten & dairy\n"
            "  3. Coconut Rice Pudding (serves 8) - $9, gluten-free"),
        "budget": ("Budget-friendly options across all courses:\n"
            "  - Hummus Platter: $10 (appetizer)\n"
            "  - Pasta Primavera: $22 (main, but has gluten)\n"
            "  - Rice Bowl Bar: $25 (main, gluten-free)\n"
            "  - Coconut Rice Pudding: $9 (dessert)\n"
            "  - Fruit Salad: $10 (dessert)"),
    }
    for key, result in recipes.items():
        if key in cuisine_type.lower():
            return result
    return f"No recipes for '{cuisine_type}'. Try: appetizer, main, dessert, budget."


def budget_calculator(items: str) -> str:
    """Calculate total cost of menu items and check against budget.

    Args:
        items: Comma-separated list of items with costs, e.g., 'Stir-Fry $28, Hummus $10'

    Returns:
        Itemized breakdown with total cost and budget status
    """
    total = 0.0
    parsed = []
    for item in items.split(","):
        item = item.strip()
        match = re.search(r'\$(\d+(?:\.\d{2})?)', item)
        if match:
            cost = float(match.group(1))
            total += cost
            parsed.append(f"  - {item}")
        elif item:
            parsed.append(f"  - {item} (no price found)")

    result = "Budget Calculation:\n" + "\n".join(parsed) if parsed else "No items"
    result += f"\n  TOTAL: ${total:.2f}"
    if total <= 60:
        result += f"\n  STATUS: Within $60 budget (${60 - total:.2f} remaining)"
    else:
        result += f"\n  STATUS: OVER BUDGET by ${total - 60:.2f}! Must choose cheaper options."
    return result


def dietary_check(meal_name: str) -> str:
    """Check if a meal meets dietary restrictions (gluten-free, dairy-free, etc.).

    Args:
        meal_name: Name of the meal to check (e.g., 'Chicken Stir-Fry')

    Returns:
        Dietary compatibility information for the meal
    """
    db = {
        "bruschetta": "Contains GLUTEN. Not suitable for gluten-free guests.",
        "hummus": "Contains sesame. Gluten-free and dairy-free.",
        "caprese": "Contains DAIRY. Gluten-free.",
        "spring rolls": "Gluten-free and dairy-free. Safe for all restrictions.",
        "stir-fry": "Gluten-free (use tamari instead of soy sauce). Dairy-free.",
        "pasta": "Contains GLUTEN. Not suitable for gluten-free guests.",
        "salmon": "Gluten-free and dairy-free. Contains fish allergen.",
        "rice bowl": "Gluten-free and dairy-free. Safe for all restrictions.",
        "fruit salad": "Gluten-free, dairy-free, vegan. Safe for all restrictions.",
        "chocolate cake": "Contains GLUTEN and DAIRY. Not suitable for restricted guests.",
        "coconut rice": "Gluten-free and dairy-free. Safe for all restrictions.",
    }
    for key, info in db.items():
        if key in meal_name.lower():
            return f"Dietary check for '{meal_name}': {info}"
    return f"No dietary data for '{meal_name}'."


# ==============================================================
# Step 2: Create Specialized ADK Agents
# ==============================================================
# Each agent has a focused role. Unlike LangGraph where all roles
# are nodes in one graph sharing TypedDict state, ADK agents are
# independent — we pass context between them as text strings.

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")

# --- PLANNER: Decomposes goal into steps (no tools) ---
planner_agent = LlmAgent(
    name="planner",
    model=MODEL,
    instruction="""You are a planning specialist. Break down goals into exactly 3
concrete, ordered steps. Output ONLY a numbered list:

1. First step
2. Second step
3. Third step

No other text. Maximum 3 steps.
Focus on finding the most impressive menu possible.
Prioritize taste and presentation over other concerns.""",
    tools=[],
    description="Decomposes complex goals into ordered, actionable steps.",
)

# --- EXECUTOR: Runs individual steps with tools ---
executor_agent = LlmAgent(
    name="executor",
    model=MODEL,
    instruction="""You are a task executor. For each step:
1. Use the appropriate tool(s) to complete it
2. Consider previous results provided as context
3. Return a clear, concise result

Tools: recipe_search (find recipes), budget_calculator (check costs),
dietary_check (verify gluten-free/dairy-free compatibility).
Use at most 2 tool calls per step, then summarize findings.""",
    tools=[recipe_search, budget_calculator, dietary_check],
    description="Executes individual plan steps using available tools.",
)

# --- EVALUATOR: Scores results strictly (no tools) ---
# KEY agent for replanning — checks if results satisfy constraints.
evaluator_agent = LlmAgent(
    name="evaluator",
    model=MODEL,
    instruction="""You are a STRICT quality evaluator. Check ALL of these:
1. BUDGET: Was total cost calculated AND within budget? No calculator = score 5 max.
2. DIETARY: Was EACH item checked? Violations = score 4 max. No checks = score 5 max.
3. COMPLETENESS: All 3 courses (appetizer + main + dessert)? Missing = score 3 max.
4. SPECIFICITY: Were actual recipes selected with tools? Vague ideas = score 5 max.

Only score 7+ if budget AND dietary were explicitly verified with tools.

Output format (EXACTLY this, one per line):
SCORE: <number>
ISSUES: <list of specific problems found>
SUGGESTION: <concrete action to fix the issues>""",
    tools=[],
    description="Strictly evaluates execution results and scores quality.",
)

# --- SYNTHESIZER: Creates final output (no tools) ---
synthesizer_agent = LlmAgent(
    name="synthesizer",
    model=MODEL,
    instruction="""Combine all step results into a clear, well-organized final
answer for a dinner party plan. Include:
- Selected menu items with costs
- Dietary compatibility notes for each item
- Total budget breakdown
- Brief cooking timeline or preparation notes

Be specific and reference the actual tool results provided.""",
    tools=[],
    description="Synthesizes all results into a comprehensive final plan.",
)


# ==============================================================
# Step 3: Helper to Run an ADK Agent
# ==============================================================
# Each call creates a fresh session. Unlike LangGraph where state
# flows through the graph automatically, we pass context manually
# via the message string.

async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent with a message and return the response text.

    Creates a fresh session each call. Includes retry logic with
    exponential backoff for transient API errors.
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="replan_demo",
                session_service=session_service,
            )

            session = await session_service.create_session(
                app_name="replan_demo",
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
                wait = attempt * 10  # Backoff: 10s, 20s, 30s, 40s
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [ERROR] All {retries} attempts failed: {e}")
                return f"[Error: API temporarily unavailable after {retries} retries]"


# ==============================================================
# Step 4: Parsing Helpers
# ==============================================================

def parse_plan(plan_text: str) -> list:
    """Extract numbered steps from planner output."""
    steps = []
    for line in plan_text.strip().split("\n"):
        match = re.match(r'^(?:\d+[\.\)]\s*|Step\s+\d+[:\.\)]\s*|-\s+)(.*)', line.strip())
        if match:
            step_text = match.group(1).strip()
            if step_text and len(step_text) > 5:
                steps.append(step_text)
    if not steps:
        steps = [l.strip() for l in plan_text.strip().split("\n")
                 if l.strip() and len(l.strip()) > 5]
    return steps


def parse_evaluation(eval_text: str) -> tuple:
    """Parse evaluator's SCORE/ISSUES/SUGGESTION into (score, feedback)."""
    score = 7
    score_match = re.search(r'SCORE:\s*(\d+)', eval_text)
    if score_match:
        score = min(10, max(1, int(score_match.group(1))))
    feedback_parts = []
    issues_match = re.search(r'ISSUES:\s*(.+?)(?:\n|$)', eval_text, re.IGNORECASE)
    suggestion_match = re.search(r'SUGGESTION:\s*(.+?)(?:\n|$)', eval_text, re.IGNORECASE)
    if issues_match and issues_match.group(1).strip().lower() != "none":
        feedback_parts.append(f"Issues: {issues_match.group(1).strip()}")
    if suggestion_match and suggestion_match.group(1).strip().lower() != "none":
        feedback_parts.append(f"Suggestion: {suggestion_match.group(1).strip()}")
    return score, "\n".join(feedback_parts)


# ==============================================================
# Step 5: The Plan-Execute-REPLAN Orchestration Loop
# ==============================================================
# ADK: replan loop = Python while loop (explicit, flexible)
# LangGraph: replan loop = conditional graph edges (visual, checkpointable)


async def run_replan_pipeline(goal: str, max_iterations: int = 8, max_replans: int = 1):
    """Run the full Plan-Execute-REPLAN pipeline with four coordinated agents."""
    iteration = 0
    replan_count = 0
    current_plan = []
    step_results = []
    evaluation_text = ""
    evaluation_score = 0
    replan_feedback = ""

    # Phase 1: INITIAL PLANNING
    print(f"\n{'='*60}")
    print(f"  PLANNER: Decomposing goal into steps...")
    print(f"{'='*60}")

    plan_text = await run_agent(planner_agent, f"Break this goal into steps:\n{goal}")
    current_plan = parse_plan(plan_text)[:4] or ["Research recipes and create a meal plan"]

    print(f"\n  Plan ({len(current_plan)} steps):")
    for i, step in enumerate(current_plan):
        print(f"    {i+1}. {step}")

    # Main Replan Loop (while loop = ADK's equivalent of graph edges)
    while True:
        # Phase 2: EXECUTE each step
        print(f"\n{'='*60}")
        print(f"  EXECUTOR: Running {len(current_plan)} steps...")
        print(f"{'='*60}")
        step_results = []

        for i, step in enumerate(current_plan):
            iteration += 1
            if iteration > max_iterations:
                print(f"  [SAFETY] Max iterations ({max_iterations}) reached.")
                break
            print(f"\n  Step {i+1}/{len(current_plan)}: {step[:80]}")

            # Build context (in LangGraph, state carries this automatically)
            context_parts = [f"Overall Goal: {goal}", f"Current Task: {step}"]
            if step_results:
                context_parts.append("\nPrevious step results:")
                for j, prev in enumerate(step_results):
                    context_parts.append(f"  Step {j+1}: {prev[:200]}")

            result = await run_agent(executor_agent, "\n".join(context_parts))
            step_results.append(result)
            for line in result.strip().split("\n")[:6]:
                if line.strip():
                    print(f"    {line.strip()[:120]}")

        # Phase 3: EVALUATE results
        print(f"\n{'='*60}")
        print(f"  EVALUATOR: Checking quality of results...")
        print(f"{'='*60}")
        results_text = ""
        for i, (step, res) in enumerate(zip(current_plan, step_results)):
            results_text += f"Step {i+1} ({step}):\n{res[:300]}\n\n"

        evaluation_text = await run_agent(evaluator_agent,
            f"Goal: {goal}\n\nExecution Results:\n{results_text}\nEvaluate strictly:")
        evaluation_score, replan_feedback = parse_evaluation(evaluation_text)

        print(f"\n  Score: {evaluation_score}/10")
        print(f"  Evaluation: {evaluation_text[:200]}")
        if replan_feedback:
            print(f"  Feedback: {replan_feedback[:150]}")

        # Phase 4: REPLAN DECISION (if/else instead of routing function)
        if evaluation_score >= 7:
            print(f"\n  DECISION: Score {evaluation_score}/10 >= 7 -> synthesizing")
            break
        elif replan_count < max_replans:
            replan_count += 1
            print(f"\n  DECISION: Score {evaluation_score}/10 < 7, "
                  f"{max_replans - replan_count} replans left -> REPLANNING")
            print(f"\n{'='*60}")
            print(f"  REPLANNER (attempt {replan_count + 1}): Adjusting plan...")
            print(f"{'='*60}")

            replan_prompt = (
                f"You previously created a plan for this goal: {goal}\n\n"
                f"The evaluator found problems:\n{replan_feedback}\n\n"
                f"Previous step results:\n")
            for i, r in enumerate(step_results):
                replan_prompt += f"  Step {i+1}: {r[:150]}\n"
            replan_prompt += ("\nCreate an IMPROVED plan (exactly 3 numbered steps) "
                "that fixes these issues. Include dietary verification and budget checking.")

            new_plan_text = await run_agent(planner_agent, replan_prompt)
            current_plan = parse_plan(new_plan_text)[:4] or ["Search recipes, verify dietary needs, check budget"]
            print(f"\n  New Plan ({len(current_plan)} steps):")
            for i, step in enumerate(current_plan):
                print(f"    {i+1}. {step}")
            continue
        else:
            print(f"\n  DECISION: Score {evaluation_score}/10 < 7 but no replans left")
            break

    # Phase 5: SYNTHESIZE final output
    print(f"\n{'='*60}")
    print(f"  SYNTHESIZER: Creating final output...")
    print(f"{'='*60}")
    synthesis_prompt = f"Original goal: {goal}\n\nAll step results:\n"
    for i, (step, res) in enumerate(zip(current_plan, step_results)):
        synthesis_prompt += f"\nStep {i+1} ({step}):\n{res}\n"
    synthesis_prompt += (f"\nEvaluation: {evaluation_text}\nReplan count: {replan_count}\n\n"
        "Provide the final comprehensive dinner party plan:")

    final_output = await run_agent(synthesizer_agent, synthesis_prompt)
    print(f"  Final output: {len(final_output)} characters")

    return {
        "goal": goal,
        "plan": current_plan,
        "step_results": step_results,
        "evaluation_score": evaluation_score,
        "evaluation": evaluation_text,
        "replan_count": replan_count,
        "iterations_used": iteration,
        "final_output": final_output,
    }


# ==============================================================
# Step 6: Main Entry Point
# ==============================================================

async def main():
    """Run the Plan-Execute-REPLAN pattern with ADK agents."""
    print("Example 3b: Plan-Execute-REPLAN Pattern in ADK")
    print("=" * 60)
    print("The agent plans, executes, EVALUATES results, and replans")
    print("if the evaluation score is below 7/10.")
    print("=" * 60)

    # Same tricky goal as Example 2b — deliberately triggers replan
    goal = (
        "Plan a dinner party for 8 people with a $60 budget. "
        "Two guests are gluten-free and one is dairy-free. "
        "Must include appetizer, main course, and dessert. "
        "Every item must be verified for dietary compatibility."
    )

    result = await run_replan_pipeline(goal, max_iterations=8, max_replans=1)

    print(f"\n\n{'#'*60}")
    print("  EXECUTION SUMMARY")
    print(f"{'#'*60}")
    print(f"  Goal: {result['goal']}")
    print(f"  Replan count: {result['replan_count']}")
    print(f"  Total iterations: {result['iterations_used']}")
    print(f"  Evaluation score: {result['evaluation_score']}/10")
    print(f"  Final Plan ({len(result['plan'])} steps):")
    for i, step in enumerate(result["plan"]):
        print(f"    {i+1}. {step}")
    print(f"\n{'#'*60}")
    print("  FINAL OUTPUT")
    print(f"{'#'*60}")
    print(f"\n{result['final_output']}")

    # ----------------------------------------------------------
    # Framework Comparison
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("LangGraph vs ADK -- Plan-Execute-REPLAN Comparison:")
    print(f"{'='*60}")
    print("  LangGraph (Example 2b):")
    print("    - Replan loop = conditional GRAPH EDGES")
    print("    - evaluator -> should_replan() -> replan_increment -> planner")
    print("    - State flows through TypedDict automatically")
    print("    - 5 nodes + 2 routing functions + 6 edges")
    print("    - Visual, checkpointable, but complex graph")
    print()
    print("  ADK (This Example):")
    print("    - Replan loop = Python WHILE LOOP with if/else")
    print("    - score < 7 and replans_used < max_replans -> continue")
    print("    - Context passed as formatted strings between agents")
    print("    - 4 agents + 1 orchestration function")
    print("    - Simpler to read, but less visual and no checkpointing")
    print()
    print("  When to Choose Which:")
    print("    - LangGraph: Checkpointing, visual debugging, team clarity")
    print("    - ADK: Simplicity, rapid iteration, dynamic replan logic")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
