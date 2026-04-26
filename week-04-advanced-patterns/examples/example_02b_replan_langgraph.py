import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 2b: Plan-Execute-REPLAN Pattern in LangGraph
=====================================================
Example 02 showed basic Plan-Execute: plan steps, execute them, synthesize.
But what happens when execution reveals the plan was WRONG?

Real-world agents need REPLANNING — the ability to:
  1. PLAN initial steps
  2. EXECUTE each step
  3. EVALUATE results (did we actually achieve the goal?)
  4. REPLAN if the evaluation fails (adjust strategy based on what we learned)

This is the key difference from Example 02:
  Example 02: plan → execute all → synthesize (no feedback loop)
  This example: plan → execute all → EVALUATE → replan if needed → repeat

The replan loop is what makes agents robust. Without it, a plan that
looks good on paper but fails in practice has no recovery path.

Graph:
  START → planner → executor → progress_check ──┐
                      ↑              |           |
                      | steps remain |           |
                      +--------------+           |
                                          all done
                                                 ↓
                                           evaluator
                                                 |
                                    ┌────────────┴────────────┐
                                    | quality < 7             | quality >= 7
                                    | AND replans left        | OR no replans left
                                    ↓                         ↓
                                 replanner              synthesize → END
                                    |
                                    └──→ planner (new plan)

Run: python week-04-advanced-patterns/examples/example_02b_replan_langgraph.py
"""

import os
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ==============================================================
# Step 1: LLM Setup
# ==============================================================

def get_llm(temperature=0.7):
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
# Step 2: Simulated Tools (same domain as examples 02/03)
# ==============================================================

@tool
def recipe_search(cuisine_type: str) -> str:
    """Search for recipes by cuisine type or course category.

    Args:
        cuisine_type: Type of cuisine or course (e.g., 'appetizer', 'main', 'dessert', 'italian')
    """
    recipes = {
        "appetizer": (
            "Appetizer options:\n"
            "  1. Bruschetta (serves 8) - $12, contains gluten\n"
            "  2. Hummus Platter (serves 10) - $10, contains sesame\n"
            "  3. Caprese Salad (serves 8) - $14, contains dairy\n"
            "  4. Spring Rolls (serves 8) - $11, gluten-free"
        ),
        "main": (
            "Main course options:\n"
            "  1. Chicken Stir-Fry (serves 8) - $28, gluten-free\n"
            "  2. Pasta Primavera (serves 8) - $22, contains gluten\n"
            "  3. Grilled Salmon (serves 8) - $38, gluten-free but expensive\n"
            "  4. Rice Bowl Bar (serves 8) - $25, gluten-free"
        ),
        "dessert": (
            "Dessert options:\n"
            "  1. Fruit Salad (serves 10) - $10, gluten-free, dairy-free\n"
            "  2. Chocolate Cake (serves 10) - $15, contains gluten & dairy\n"
            "  3. Coconut Rice Pudding (serves 8) - $9, gluten-free"
        ),
        "budget": (
            "Budget-friendly options across all courses:\n"
            "  - Hummus Platter: $10 (appetizer)\n"
            "  - Pasta Primavera: $22 (main, but has gluten)\n"
            "  - Rice Bowl Bar: $25 (main, gluten-free)\n"
            "  - Coconut Rice Pudding: $9 (dessert)\n"
            "  - Fruit Salad: $10 (dessert)"
        ),
    }
    for key, result in recipes.items():
        if key in cuisine_type.lower():
            return result
    return f"No recipes for '{cuisine_type}'. Try: appetizer, main, dessert, budget."


@tool
def budget_calculator(items: str) -> str:
    """Calculate total cost of menu items and check against budget.

    Args:
        items: Comma-separated list of items with costs, e.g., 'Stir-Fry $28, Hummus $10'
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
    if total <= 100:
        result += f"\n  STATUS: Within $100 budget (${100 - total:.2f} remaining)"
    else:
        result += f"\n  STATUS: OVER BUDGET by ${total - 100:.2f}! Must choose cheaper options."
    return result


@tool
def dietary_check(meal_name: str) -> str:
    """Check if a meal meets dietary restrictions (gluten-free, dairy-free, etc.).

    Args:
        meal_name: Name of the meal to check
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
# Step 3: State — Extends basic Plan-Execute with replan fields
# ==============================================================

class ReplanState(TypedDict):
    goal: str
    plan: list                # Current plan steps
    current_step: int         # Which step we're executing
    step_results: list        # Results from completed steps
    final_output: str         # Final synthesized answer
    iteration: int            # Total execution iterations
    max_iterations: int       # Safety limit
    replan_count: int         # How many times we've replanned
    max_replans: int          # Max replans allowed (prevent infinite loops)
    evaluation: str           # Evaluator's assessment
    evaluation_score: int     # Numeric quality score (1-10)
    replan_feedback: str      # What to fix in the next plan
    messages: Annotated[list, add_messages]


# ==============================================================
# Step 4: Nodes
# ==============================================================

tools = [recipe_search, budget_calculator, dietary_check]
executor_llm_with_tools = executor_llm.bind_tools(tools)


def planner_node(state: ReplanState) -> dict:
    """Decompose goal into steps. If replanning, incorporate feedback."""
    replan_count = state.get("replan_count", 0)
    feedback = state.get("replan_feedback", "")

    if replan_count > 0 and feedback:
        print(f"\n{'='*60}")
        print(f"  REPLANNER (attempt {replan_count + 1}): Adjusting plan...")
        print(f"  Feedback: {feedback[:100]}")
        print(f"{'='*60}")

        prompt = (
            f"You previously created a plan for this goal: {state['goal']}\n\n"
            f"The plan was executed but the evaluator found problems:\n"
            f"{feedback}\n\n"
            f"Previous step results:\n"
        )
        for i, r in enumerate(state.get("step_results", [])):
            prompt += f"  Step {i+1}: {r[:150]}\n"
        prompt += (
            "\nCreate an IMPROVED plan (3-5 numbered steps) that fixes these issues. "
            "Focus specifically on what went wrong."
        )
    else:
        print(f"\n{'='*60}")
        print(f"  PLANNER: Decomposing goal into steps...")
        print(f"{'='*60}")
        prompt = f"Goal: {state['goal']}"

    # On the FIRST attempt, the planner prompt is generic — it doesn't
    # emphasize checking dietary restrictions or verifying the budget.
    # This makes it likely to produce a plan that the evaluator will reject,
    # triggering a replan. On subsequent attempts, the replan_feedback
    # from the evaluator guides the planner to fix the specific issues.
    messages = [
        SystemMessage(content=(
            "You are a planning agent. Break the goal into exactly 3 concrete, ordered steps.\n"
            "Output ONLY a numbered list:\n1. First step\n2. Second step\n3. Third step\n"
            "No other text. Maximum 3 steps.\n"
            "Focus on finding the most impressive menu possible. "
            "Prioritize taste and presentation over other concerns."
        )),
        HumanMessage(content=prompt),
    ]

    response = planner_llm.invoke(messages)
    steps = []
    for line in response.content.strip().split("\n"):
        cleaned = re.sub(r'^(\d+[\.\)]\s*|[-*]\s*)', '', line.strip()).strip()
        if cleaned:
            steps.append(cleaned)
    steps = steps[:4] or ["Research the topic and provide a comprehensive answer"]

    print(f"\n  Plan ({len(steps)} steps):")
    for i, step in enumerate(steps):
        print(f"    {i+1}. {step}")

    return {
        "plan": steps,
        "current_step": 0,
        "step_results": [],
        "iteration": state.get("iteration", 0),
        "messages": [],
    }


def executor_node(state: ReplanState) -> dict:
    """Execute the current step with tools."""
    step_idx = state["current_step"]
    step = state["plan"][step_idx]
    iteration = state.get("iteration", 0) + 1

    print(f"\n  EXECUTOR: Step {step_idx + 1}/{len(state['plan'])} — {step[:80]}")

    context_parts = [f"Overall Goal: {state['goal']}", f"Current Task: {step}"]
    if state["step_results"]:
        context_parts.append("Previous results:")
        for i, r in enumerate(state["step_results"]):
            context_parts.append(f"  Step {i+1}: {r[:200]}")

    step_messages = [
        SystemMessage(content=(
            "Complete the given task using available tools. Tools:\n"
            "  - recipe_search: Find recipes by cuisine/course\n"
            "  - budget_calculator: Calculate costs\n"
            "  - dietary_check: Check dietary compatibility\n"
            "Use at most 2 tool calls, then summarize your results. "
            "If the task doesn't need tools, just answer directly."
        )),
        HumanMessage(content="\n".join(context_parts)),
    ]

    tool_node = ToolNode(tools)
    for _ in range(2):  # Max 2 tool rounds per step to keep execution fast
        try:
            response = executor_llm_with_tools.invoke(step_messages)
        except Exception as e:
            # Groq/Llama models sometimes generate malformed tool calls
            # (XML format instead of JSON). Fall back to a no-tools call.
            print(f"    [WARN] Tool call failed: {str(e)[:80]}. Retrying without tools...")
            response = executor_llm.invoke(step_messages)
            step_messages.append(response)
            break
        step_messages.append(response)
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    [TOOL] {tc['name']}({tc['args']})")
            try:
                tool_result = tool_node.invoke({"messages": step_messages})
                step_messages.extend(tool_result["messages"])
            except Exception as e:
                print(f"    [WARN] Tool execution failed: {str(e)[:80]}. Continuing...")
                for tc in response.tool_calls:
                    step_messages.append(ToolMessage(
                        content=f"Tool error: could not execute. Please answer without this tool.",
                        tool_call_id=tc["id"],
                    ))
        else:
            break

    result_text = response.content or "Step completed."
    print(f"    Result: {result_text[:200]}...")

    return {
        "step_results": list(state["step_results"]) + [result_text],
        "iteration": iteration,
        "messages": [],
    }


def progress_check_node(state: ReplanState) -> dict:
    """Advance to next step."""
    next_step = state["current_step"] + 1
    if next_step < len(state["plan"]):
        print(f"  PROGRESS: Step {next_step}/{len(state['plan'])} done. Next step.")
    else:
        print(f"  PROGRESS: All {len(state['plan'])} steps done. Moving to evaluation.")
    return {"current_step": next_step}


# -- NEW: Evaluator Node (not in example 02) -----------------------
# This is the key addition. After all steps execute, the evaluator
# checks whether the results actually satisfy the goal's constraints.

def evaluator_node(state: ReplanState) -> dict:
    """Evaluate whether execution results satisfy the goal."""
    print(f"\n{'='*60}")
    print(f"  EVALUATOR: Checking quality of results...")
    print(f"{'='*60}")

    results_text = ""
    for i, (step, result) in enumerate(zip(state["plan"], state["step_results"])):
        results_text += f"Step {i+1} ({step}):\n{result[:300]}\n\n"

    messages = [
        SystemMessage(content=(
            "You are a STRICT quality evaluator. Review the execution results against "
            "the original goal. You must check ALL of the following:\n"
            "  1. BUDGET: Was the total cost calculated AND confirmed within budget?\n"
            "     If no budget_calculator was used or total exceeds the limit → score 5 max.\n"
            "  2. DIETARY: Was EACH menu item checked for dietary compatibility?\n"
            "     If any item violates gluten-free or dairy-free needs → score 4 max.\n"
            "     If dietary_check was never used → score 5 max.\n"
            "  3. COMPLETENESS: Are all 3 courses present (appetizer + main + dessert)?\n"
            "     If any course is missing → score 3 max.\n"
            "  4. SPECIFICITY: Were actual recipes from the recipe database selected?\n"
            "     If the plan only describes vague ideas without tool results → score 5 max.\n\n"
            "Be strict. A plan that 'sounds good' but wasn't verified with tools should "
            "score 5 or below. Only score 7+ if budget AND dietary constraints were "
            "explicitly verified using tools.\n\n"
            "Output format (EXACTLY this, one per line):\n"
            "SCORE: <number>\n"
            "ISSUES: <list of specific problems found>\n"
            "SUGGESTION: <concrete action to fix the issues>"
        )),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            f"Execution Results:\n{results_text}\n"
            "Evaluate these results strictly:"
        )),
    ]

    response = planner_llm.invoke(messages)
    eval_text = response.content.strip()

    # Parse score
    score = 7  # default
    score_match = re.search(r'SCORE:\s*(\d+)', eval_text)
    if score_match:
        score = min(10, max(1, int(score_match.group(1))))

    # Parse issues/suggestions for replan feedback
    feedback = ""
    issues_match = re.search(r'ISSUES:\s*(.+?)(?:\n|$)', eval_text, re.IGNORECASE)
    suggestion_match = re.search(r'SUGGESTION:\s*(.+?)(?:\n|$)', eval_text, re.IGNORECASE)
    if issues_match and issues_match.group(1).strip().lower() != "none":
        feedback += f"Issues: {issues_match.group(1).strip()}\n"
    if suggestion_match and suggestion_match.group(1).strip().lower() != "none":
        feedback += f"Suggestion: {suggestion_match.group(1).strip()}"

    print(f"\n  Score: {score}/10")
    print(f"  Evaluation: {eval_text[:200]}")
    if feedback:
        print(f"  Replan feedback: {feedback[:150]}")

    return {
        "evaluation": eval_text,
        "evaluation_score": score,
        "replan_feedback": feedback,
    }


def synthesize_node(state: ReplanState) -> dict:
    """Combine all results into final output."""
    print(f"\n{'='*60}")
    print(f"  SYNTHESIZER: Creating final output...")
    print(f"{'='*60}")

    results_text = ""
    for i, (step, result) in enumerate(zip(state["plan"], state["step_results"])):
        results_text += f"Step {i+1} ({step}):\n{result}\n\n"

    messages = [
        SystemMessage(content=(
            "Combine all step results into a clear, well-organized final answer. "
            "Include specific details from each step."
        )),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            f"Results:\n{results_text}\n"
            f"Evaluation: {state.get('evaluation', 'N/A')}\n\n"
            f"Replan count: {state.get('replan_count', 0)}\n\n"
            "Provide the final comprehensive answer:"
        )),
    ]

    response = executor_llm.invoke(messages)
    print(f"  Final output: {len(response.content)} characters")
    return {"final_output": response.content}


# ==============================================================
# Step 5: Routing Functions
# ==============================================================

def should_continue_executing(state: ReplanState) -> str:
    """Route after progress check: more steps or evaluate."""
    if state.get("iteration", 0) >= state.get("max_iterations", 8):
        print(f"  [SAFETY] Max iterations reached. Forcing evaluation.")
        return "evaluator"
    if state["current_step"] >= len(state["plan"]):
        return "evaluator"
    return "executor"


def should_replan(state: ReplanState) -> str:
    """Route after evaluation: replan or synthesize.

    This is the KEY routing decision for replanning:
      - Score < 7 AND replans remaining → replan (go back to planner)
      - Score >= 7 OR no replans left → synthesize (finish)
    """
    score = state.get("evaluation_score", 7)
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("max_replans", 2)

    if score < 7 and replan_count < max_replans:
        print(f"\n  DECISION: Score {score}/10 < 7 and {max_replans - replan_count} "
              f"replans left → REPLANNING")
        return "replan"
    elif score < 7:
        print(f"\n  DECISION: Score {score}/10 < 7 but no replans left → "
              f"synthesizing with best effort")
        return "synthesize"
    else:
        print(f"\n  DECISION: Score {score}/10 >= 7 → DONE, synthesizing")
        return "synthesize"


def replan_increment(state: ReplanState) -> dict:
    """Increment replan counter before going back to planner."""
    new_count = state.get("replan_count", 0) + 1
    print(f"\n  REPLAN #{new_count}: Going back to planner with feedback...")
    return {"replan_count": new_count}


# ==============================================================
# Step 6: Build the Graph
# ==============================================================
# The graph extends Example 02's Plan-Execute with:
#   - An evaluator node after execution completes
#   - A conditional replan edge back to the planner
#   - A replan counter to prevent infinite replan loops
#
#   planner → executor ⇄ progress_check → evaluator
#      ↑                                      |
#      |              replan_increment ←───── (score < 7)
#      |                    |                  |
#      +────────────────────+           (score >= 7)
#                                             |
#                                        synthesize → END

def build_replan_graph():
    graph = StateGraph(ReplanState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("progress_check", progress_check_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("replan_increment", replan_increment)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "progress_check")

    graph.add_conditional_edges("progress_check", should_continue_executing, {
        "executor": "executor",
        "evaluator": "evaluator",
    })

    graph.add_conditional_edges("evaluator", should_replan, {
        "replan": "replan_increment",
        "synthesize": "synthesize",
    })

    # After incrementing replan count, go back to planner
    graph.add_edge("replan_increment", "planner")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ==============================================================
# Step 7: Run
# ==============================================================

def run_replan_agent(goal: str, max_iterations: int = 8, max_replans: int = 1):
    app = build_replan_graph()
    result = app.invoke(
        {
            "goal": goal,
            "plan": [],
            "current_step": 0,
            "step_results": [],
            "final_output": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "replan_count": 0,
            "max_replans": max_replans,
            "evaluation": "",
            "evaluation_score": 0,
            "replan_feedback": "",
            "messages": [],
        },
        {"recursion_limit": 50},
    )
    return result


if __name__ == "__main__":
    print("Example 2b: Plan-Execute-REPLAN Pattern in LangGraph")
    print("=" * 60)
    print("The agent plans, executes, EVALUATES results, and replans")
    print("if the evaluation score is below 7/10.")
    print("=" * 60)

    # This goal is deliberately tricky:
    #   - The planner prompt emphasizes "impressive" food, so the first
    #     plan will likely pick expensive items (salmon $38, cake $15, etc.)
    #   - Gluten-free + dairy-free constraints eliminate many options
    #   - The strict evaluator will reject plans that skip dietary checks
    #   - This triggers a replan with feedback to fix the specific issues
    goal = (
        "Plan a dinner party for 8 people with a $60 budget. "
        "Two guests are gluten-free and one is dairy-free. "
        "Must include appetizer, main course, and dessert. "
        "Every item must be verified for dietary compatibility."
    )

    result = run_replan_agent(goal, max_iterations=8, max_replans=1)

    print(f"\n\n{'#'*60}")
    print("  EXECUTION SUMMARY")
    print(f"{'#'*60}")
    print(f"\n  Goal: {result['goal']}")
    print(f"  Replan count: {result['replan_count']}")
    print(f"  Evaluation score: {result['evaluation_score']}/10")
    print(f"\n  Final Plan ({len(result['plan'])} steps):")
    for i, step in enumerate(result["plan"]):
        print(f"    {i+1}. {step}")

    print(f"\n{'#'*60}")
    print("  FINAL OUTPUT")
    print(f"{'#'*60}")
    print(f"\n{result['final_output']}")

    print(f"\n{'='*60}")
    print("Key Takeaways (vs Example 02):")
    print("  1. EVALUATION: After execution, an evaluator scores the results")
    print("  2. REPLAN LOOP: If score < 7, the planner gets feedback and retries")
    print("  3. SAFETY: max_replans prevents infinite replan loops")
    print("  4. CONTEXT: Replanner sees what went wrong + previous results")
    print("  5. CONVERGENCE: Each replan should improve on the last attempt")
    print(f"{'='*60}")
