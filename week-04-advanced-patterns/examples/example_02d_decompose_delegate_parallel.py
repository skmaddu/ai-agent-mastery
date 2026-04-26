import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 2d: Decompose + Delegate with Parallel Agents
======================================================
Example 01 introduced three planning flavors. Examples 02/03 covered
Plan-Execute, 02b added Replanning, 02c showed ReAct. This example
implements the third pattern: DECOMPOSE + DELEGATE.

The key idea: break a goal into INDEPENDENT sub-goals, assign each
to a SPECIALIST agent, and run them IN PARALLEL. A final aggregator
combines the results.

This is fundamentally different from Plan-Execute:
  Plan-Execute:        Sequential steps, one executor, shared context
  Decompose+Delegate:  Independent sub-goals, multiple specialists, PARALLEL

Real-world analogy: A catering manager doesn't cook everything
themselves in order. They assign the appetizer team, the main course
team, and the dessert team to work SIMULTANEOUSLY, then combine.

Architecture:
  1. DECOMPOSER — breaks goal into independent sub-goals
  2. SPECIALISTS — separate agents with focused expertise, run in parallel:
     - Venue Specialist (finds and evaluates venues)
     - Menu Specialist (plans food within budget and dietary constraints)
     - Entertainment Specialist (plans activities and music)
  3. AGGREGATOR — combines all specialist outputs into a cohesive plan

Graph:
  START → decomposer → parallel_dispatch
                           |
              ┌────────────┼────────────┐
              ↓            ↓            ↓
          venue_agent  menu_agent  entertainment_agent
              |            |            |
              └────────────┼────────────┘
                           ↓
                       aggregator → END

Run: python week-04-advanced-patterns/examples/example_02d_decompose_delegate_parallel.py
"""

import os
import re
import concurrent.futures
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


decomposer_llm = get_llm(temperature=0)
specialist_llm = get_llm(temperature=0.7)


# ==============================================================
# Step 2: Specialist Tools
# ==============================================================
# Each specialist has access to domain-specific tools.
# This is a key feature of Decompose+Delegate: specialists
# have DIFFERENT capabilities (tools), not just different prompts.

# --- Venue Tools ---
@tool
def search_venues(event_type: str) -> str:
    """Search for event venues by type.

    Args:
        event_type: Type of event (e.g., 'birthday party', 'corporate dinner', 'outdoor gathering')
    """
    venues = {
        "indoor": [
            "City Banquet Hall — capacity 50, $500 rental, full kitchen, AV system",
            "The Loft Space — capacity 30, $300 rental, open plan, BYO catering",
            "Community Center — capacity 80, $150 rental, basic kitchen, parking",
        ],
        "outdoor": [
            "Riverside Park Pavilion — capacity 40, $100 rental, covered, grills available",
            "Rooftop Garden — capacity 25, $400 rental, skyline views, no kitchen",
            "Botanical Gardens — capacity 60, $350 rental, photography friendly",
        ],
        "restaurant": [
            "Luigi's Private Room — capacity 20, $0 rental (min spend $800), Italian cuisine",
            "The Modern Bistro — capacity 15, $200 rental, prix fixe $45/person",
            "Harbor View — capacity 30, $0 rental (min spend $500), seafood focused",
        ],
    }
    query = event_type.lower()
    for key, items in venues.items():
        if key in query:
            return f"Venues for '{event_type}':\n" + "\n".join(f"  - {v}" for v in items)

    all_venues = []
    for category, items in venues.items():
        all_venues.extend(f"  - [{category}] {v}" for v in items)
    return f"All available venues:\n" + "\n".join(all_venues)


@tool
def check_venue_availability(venue_name: str) -> str:
    """Check if a specific venue is available and get details.

    Args:
        venue_name: Name of the venue to check
    """
    details = {
        "community center": "AVAILABLE. $150 rental. Includes tables/chairs for 80, "
                           "basic kitchen, ample parking. Alcohol allowed with permit ($50).",
        "riverside": "AVAILABLE. $100 rental. Covered pavilion, 3 BBQ grills, "
                    "picnic tables for 40. No alcohol. Closes at 9 PM.",
        "loft": "AVAILABLE. $300 rental. Open-plan space, bare walls (decorating OK), "
               "no kitchen but catering delivery area. 10 PM curfew.",
        "banquet": "AVAILABLE. $500 rental. Full service kitchen, round tables "
                  "with linens, sound system, projector. Late hours OK.",
        "botanical": "AVAILABLE. $350 rental. Beautiful garden setting, guest max 60, "
                    "photos allowed, catering must use approved vendor list.",
    }
    for key, info in details.items():
        if key in venue_name.lower():
            return f"Venue check for '{venue_name}': {info}"
    return f"Venue '{venue_name}' not found in our database."


# --- Menu Tools ---
@tool
def search_caterers(cuisine_type: str) -> str:
    """Search for catering options by cuisine type.

    Args:
        cuisine_type: Type of food or cuisine (e.g., 'italian', 'bbq', 'asian', 'mixed')
    """
    caterers = {
        "italian": (
            "Italian Catering Options:\n"
            "  1. Mama Rosa's — $18/person, family-style pasta + salad + bread\n"
            "  2. Tuscan Table — $25/person, antipasto + main + dessert, GF options\n"
            "  3. Pizza Party Co — $12/person, assorted pizzas + sides"
        ),
        "bbq": (
            "BBQ Catering Options:\n"
            "  1. Smokey Joe's — $20/person, brisket + ribs + 3 sides\n"
            "  2. Backyard BBQ Co — $15/person, burgers + hot dogs + sides\n"
            "  3. Grill Masters — $22/person, premium meats + vegetarian options"
        ),
        "asian": (
            "Asian Catering Options:\n"
            "  1. Golden Dragon — $16/person, stir-fry buffet, all GF\n"
            "  2. Sushi Fresh — $28/person, sushi platters + hot dishes\n"
            "  3. Thai Express — $14/person, curry + noodles + rice, vegan available"
        ),
        "mixed": (
            "Mixed/General Catering:\n"
            "  1. All Occasions — $20/person, customizable menu, GF/vegan OK\n"
            "  2. Budget Bites — $10/person, sandwich + soup + salad bar\n"
            "  3. Elegant Eats — $35/person, 3-course plated dinner"
        ),
    }
    for key, result in caterers.items():
        if key in cuisine_type.lower():
            return result
    return f"No caterers found for '{cuisine_type}'. Try: italian, bbq, asian, mixed."


@tool
def calculate_food_budget(per_person_cost: float, guest_count: int, extras: str) -> str:
    """Calculate total food budget including extras.

    Args:
        per_person_cost: Cost per person for catering
        guest_count: Number of guests
        extras: Comma-separated extras like 'drinks, cake, ice'
    """
    food_total = per_person_cost * guest_count

    extra_costs = {
        "drink": 3.0, "drinks": 3.0 * guest_count,
        "cake": 40.0, "ice": 15.0, "napkins": 10.0,
        "plates": 12.0, "cups": 8.0, "utensils": 10.0,
        "decorations": 25.0, "flowers": 30.0,
    }

    extras_total = 0
    extras_breakdown = []
    for extra in extras.split(","):
        extra = extra.strip().lower()
        for key, cost in extra_costs.items():
            if key in extra:
                extras_total += cost
                extras_breakdown.append(f"  - {extra}: ${cost:.2f}")
                break

    total = food_total + extras_total
    result = (
        f"Food Budget Calculation:\n"
        f"  Catering: ${per_person_cost:.2f} x {guest_count} guests = ${food_total:.2f}\n"
    )
    if extras_breakdown:
        result += "  Extras:\n" + "\n".join(extras_breakdown) + "\n"
    result += f"  TOTAL: ${total:.2f}"
    return result


# --- Entertainment Tools ---
@tool
def search_entertainment(event_type: str) -> str:
    """Search for entertainment options for an event.

    Args:
        event_type: Type of event or entertainment preference
                    (e.g., 'birthday', 'casual', 'formal', 'outdoor')
    """
    options = {
        "birthday": (
            "Birthday Entertainment Options:\n"
            "  1. DJ Package — $200, 3 hours, brings own speakers\n"
            "  2. Photo Booth — $150, 2 hours, props + prints included\n"
            "  3. Karaoke Setup — $100, 4 hours, 2 mics + screen\n"
            "  4. Lawn Games Kit — $50, cornhole + giant Jenga + bocce"
        ),
        "casual": (
            "Casual Party Entertainment:\n"
            "  1. Spotify Playlist + Bluetooth Speaker — $0 (DIY)\n"
            "  2. Board Game Collection — $30 to rent assorted games\n"
            "  3. Trivia Night Kit — $40, hosted trivia with prizes\n"
            "  4. Movie Projector Rental — $75, outdoor movie setup"
        ),
        "formal": (
            "Formal Event Entertainment:\n"
            "  1. Live Jazz Trio — $500, 3 hours\n"
            "  2. String Quartet — $600, 2 hours\n"
            "  3. Professional MC — $300, keeps event flowing\n"
            "  4. Magician/Close-up Magic — $250, 1.5 hours"
        ),
        "outdoor": (
            "Outdoor Event Entertainment:\n"
            "  1. Lawn Games Kit — $50, cornhole + giant Jenga + bocce\n"
            "  2. Portable Speaker + Playlist — $0 (DIY)\n"
            "  3. Outdoor Movie Setup — $75, projector + screen + blankets\n"
            "  4. Scavenger Hunt Kit — $25, pre-made clues for park"
        ),
    }
    for key, result in options.items():
        if key in event_type.lower():
            return result
    return "Entertainment for '" + event_type + "'. Try: birthday, casual, formal, outdoor."


@tool
def check_entertainment_cost(items: str) -> str:
    """Calculate total entertainment budget.

    Args:
        items: Comma-separated entertainment items with costs
    """
    total = 0.0
    parsed = []
    for item in items.split(","):
        match = re.search(r'\$(\d+)', item)
        if match:
            cost = float(match.group(1))
            total += cost
            parsed.append(f"  - {item.strip()}")
    result = "Entertainment Budget:\n" + "\n".join(parsed) + f"\n  TOTAL: ${total:.2f}"
    return result


# ==============================================================
# Step 3: State
# ==============================================================

class DecomposeState(TypedDict):
    goal: str
    sub_goals: list            # List of independent sub-goals
    venue_result: str          # Output from venue specialist
    menu_result: str           # Output from menu specialist
    entertainment_result: str  # Output from entertainment specialist
    final_output: str
    messages: Annotated[list, add_messages]


# ==============================================================
# Step 4: Nodes
# ==============================================================

venue_tools = [search_venues, check_venue_availability]
menu_tools = [search_caterers, calculate_food_budget]
entertainment_tools = [search_entertainment, check_entertainment_cost]


def run_specialist(llm, tools_list, system_prompt: str, task: str, goal: str) -> str:
    """Run a specialist agent with its tools until it gives a final answer.

    This helper encapsulates the tool-calling loop for one specialist.
    Each specialist has its OWN set of tools — a key differentiator
    from Plan-Execute where one executor shares all tools.
    """
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Overall event goal: {goal}\n\nYour specific task: {task}"),
    ]

    for _ in range(5):  # Max 5 tool calls per specialist
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"    [WARN] Tool call failed: {str(e)[:80]}. Retrying without tools...")
            response = llm.invoke(messages)
            messages.append(response)
            break
        messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    [TOOL] {tc['name']}({tc['args']})")
            try:
                tool_result = tool_node.invoke({"messages": messages})
                messages.extend(tool_result["messages"])
            except Exception as e:
                print(f"    [WARN] Tool execution failed: {str(e)[:80]}. Continuing...")
                for tc in response.tool_calls:
                    messages.append(ToolMessage(
                        content="Tool error: could not execute. Please answer without this tool.",
                        tool_call_id=tc["id"],
                    ))
        else:
            break

    return response.content or "No result from specialist."


def decomposer_node(state: DecomposeState) -> dict:
    """Break the goal into independent sub-goals for parallel delegation.

    This is the DECOMPOSER — it identifies which parts of the goal
    can be handled independently by different specialists.
    """
    print(f"\n{'='*60}")
    print(f"  DECOMPOSER: Breaking goal into independent sub-goals...")
    print(f"{'='*60}")

    messages = [
        SystemMessage(content=(
            "You are a goal decomposition specialist. Break the event planning goal "
            "into exactly 3 independent sub-goals that can be worked on IN PARALLEL "
            "by different specialists:\n"
            "  1. VENUE — finding and evaluating the event location\n"
            "  2. MENU — planning food, drinks, and dietary accommodations\n"
            "  3. ENTERTAINMENT — planning activities, music, and fun\n\n"
            "For each sub-goal, include relevant constraints from the original goal "
            "(budget allocation, guest count, preferences, etc.).\n\n"
            "Output format (exactly 3 lines):\n"
            "VENUE: <venue sub-goal with constraints>\n"
            "MENU: <menu sub-goal with constraints>\n"
            "ENTERTAINMENT: <entertainment sub-goal with constraints>"
        )),
        HumanMessage(content=f"Goal: {state['goal']}"),
    ]

    response = decomposer_llm.invoke(messages)
    raw = response.content.strip()

    # Parse sub-goals
    sub_goals = {"venue": "", "menu": "", "entertainment": ""}
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("VENUE"):
            sub_goals["venue"] = re.sub(r'^VENUE[:\s]*', '', line, flags=re.IGNORECASE).strip()
        elif line.upper().startswith("MENU"):
            sub_goals["menu"] = re.sub(r'^MENU[:\s]*', '', line, flags=re.IGNORECASE).strip()
        elif line.upper().startswith("ENTERTAINMENT"):
            sub_goals["entertainment"] = re.sub(r'^ENTERTAINMENT[:\s]*', '', line, flags=re.IGNORECASE).strip()

    # Fallbacks if parsing fails
    if not sub_goals["venue"]:
        sub_goals["venue"] = "Find a suitable venue for the event within budget"
    if not sub_goals["menu"]:
        sub_goals["menu"] = "Plan a menu with catering within the food budget"
    if not sub_goals["entertainment"]:
        sub_goals["entertainment"] = "Plan entertainment activities for guests"

    print(f"\n  Sub-goals (each assigned to a specialist):")
    for role, task in sub_goals.items():
        print(f"    [{role.upper()}] {task[:100]}")

    return {
        "sub_goals": [sub_goals["venue"], sub_goals["menu"], sub_goals["entertainment"]],
    }


def parallel_specialists_node(state: DecomposeState) -> dict:
    """Run all three specialists IN PARALLEL using ThreadPoolExecutor.

    This is the DELEGATE step — each specialist works independently
    on its sub-goal with its own tools. They don't share state or
    wait for each other.

    We use ThreadPoolExecutor for true parallelism. In production,
    you'd use async or distributed workers.
    """
    print(f"\n{'='*60}")
    print(f"  PARALLEL DISPATCH: Running 3 specialists simultaneously...")
    print(f"{'='*60}")

    venue_task = state["sub_goals"][0] if len(state["sub_goals"]) > 0 else "Find a venue"
    menu_task = state["sub_goals"][1] if len(state["sub_goals"]) > 1 else "Plan a menu"
    entertainment_task = state["sub_goals"][2] if len(state["sub_goals"]) > 2 else "Plan entertainment"

    goal = state["goal"]

    # Define specialist configurations
    specialists = [
        {
            "name": "VENUE SPECIALIST",
            "tools": venue_tools,
            "prompt": (
                "You are a venue specialist. Find and recommend the best venue for "
                "the event. Consider: capacity, cost, amenities, and restrictions. "
                "Use your tools to search venues and check availability. "
                "Provide a clear recommendation with reasoning."
            ),
            "task": venue_task,
        },
        {
            "name": "MENU SPECIALIST",
            "tools": menu_tools,
            "prompt": (
                "You are a menu and catering specialist. Plan the food for the event. "
                "Consider: budget per person, dietary restrictions, cuisine preferences, "
                "and practical logistics. Use your tools to search caterers and "
                "calculate costs. Provide a detailed food plan with budget breakdown."
            ),
            "task": menu_task,
        },
        {
            "name": "ENTERTAINMENT SPECIALIST",
            "tools": entertainment_tools,
            "prompt": (
                "You are an entertainment specialist. Plan activities and entertainment "
                "for the event. Consider: event type, guest demographics, venue "
                "constraints, and budget. Use your tools to find options and "
                "calculate costs. Provide a fun activity plan."
            ),
            "task": entertainment_task,
        },
    ]

    # Run all three specialists in PARALLEL
    # This is the key advantage of Decompose+Delegate over Plan-Execute:
    # independent sub-tasks run simultaneously, cutting wall-clock time.
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {}
        for spec in specialists:
            print(f"\n  >> Launching {spec['name']}...")
            future = executor.submit(
                run_specialist,
                specialist_llm,
                spec["tools"],
                spec["prompt"],
                spec["task"],
                goal,
            )
            future_to_name[future] = spec["name"]

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                results[name] = result
                print(f"\n  << {name} completed ({len(result)} chars)")
                for line in result.strip().split("\n")[:5]:
                    if line.strip():
                        print(f"     {line.strip()[:100]}")
                if len(result.strip().split("\n")) > 5:
                    print(f"     ... ({len(result.strip().split(chr(10))) - 5} more lines)")
            except Exception as e:
                results[name] = f"Error: {e}"
                print(f"\n  << {name} FAILED: {e}")

    return {
        "venue_result": results.get("VENUE SPECIALIST", "No venue result"),
        "menu_result": results.get("MENU SPECIALIST", "No menu result"),
        "entertainment_result": results.get("ENTERTAINMENT SPECIALIST", "No entertainment result"),
    }


def aggregator_node(state: DecomposeState) -> dict:
    """Combine all specialist results into a cohesive event plan.

    The aggregator's job is to:
      1. Check for conflicts between specialist outputs
         (e.g., menu requires kitchen but venue has none)
      2. Ensure the combined plan fits the overall budget
      3. Create a unified, well-organized final plan
    """
    print(f"\n{'='*60}")
    print(f"  AGGREGATOR: Combining specialist results...")
    print(f"{'='*60}")

    messages = [
        SystemMessage(content=(
            "You are an event planning aggregator. You receive results from three "
            "specialist agents (venue, menu, entertainment) who worked independently "
            "and in parallel.\n\n"
            "Your job:\n"
            "1. Combine their outputs into ONE cohesive event plan\n"
            "2. Check for CONFLICTS (e.g., venue has no kitchen but menu needs one)\n"
            "3. Calculate the TOTAL budget across all areas\n"
            "4. Flag any issues or missing items\n"
            "5. Present a clear, organized final plan\n\n"
            "Format your output with clear sections: Venue, Menu, Entertainment, "
            "Budget Summary, and any Notes/Conflicts."
        )),
        HumanMessage(content=(
            f"Original Goal: {state['goal']}\n\n"
            f"--- VENUE SPECIALIST REPORT ---\n{state['venue_result']}\n\n"
            f"--- MENU SPECIALIST REPORT ---\n{state['menu_result']}\n\n"
            f"--- ENTERTAINMENT SPECIALIST REPORT ---\n{state['entertainment_result']}\n\n"
            "Create the combined event plan:"
        )),
    ]

    response = specialist_llm.invoke(messages)
    print(f"  Final plan: {len(response.content)} characters")

    return {"final_output": response.content}


# ==============================================================
# Step 5: Build the Graph
# ==============================================================
#
# This graph has a FAN-OUT / FAN-IN structure:
#
#   decomposer → parallel_specialists → aggregator → END
#                  (runs 3 agents
#                   in parallel
#                   internally)
#
# The parallelism happens INSIDE the parallel_specialists_node
# using ThreadPoolExecutor. In a production system, you could
# also use LangGraph's Send() API for graph-level fan-out.

def build_decompose_delegate_graph():
    graph = StateGraph(DecomposeState)

    graph.add_node("decomposer", decomposer_node)
    graph.add_node("parallel_specialists", parallel_specialists_node)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("decomposer")
    graph.add_edge("decomposer", "parallel_specialists")
    graph.add_edge("parallel_specialists", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ==============================================================
# Step 6: Run
# ==============================================================

def run_decompose_delegate(goal: str) -> dict:
    app = build_decompose_delegate_graph()
    result = app.invoke({
        "goal": goal,
        "sub_goals": [],
        "venue_result": "",
        "menu_result": "",
        "entertainment_result": "",
        "final_output": "",
        "messages": [],
    })
    return result


if __name__ == "__main__":
    print("Example 2d: Decompose + Delegate with Parallel Agents")
    print("=" * 60)
    print("Three specialist agents work IN PARALLEL on independent")
    print("sub-goals, then an aggregator combines their results.")
    print("=" * 60)

    goal = (
        "Plan a birthday party for 20 people with a total budget of $800. "
        "The birthday person loves Italian food and outdoor activities. "
        "Two guests are vegetarian and one is gluten-free. "
        "The party is on a Saturday afternoon."
    )

    print(f"\nGoal: {goal}")
    result = run_decompose_delegate(goal)

    print(f"\n\n{'#'*60}")
    print("  COMBINED EVENT PLAN")
    print(f"{'#'*60}")
    print(f"\n{result['final_output']}")

    print(f"\n{'='*60}")
    print("Decompose + Delegate vs Other Patterns:")
    print(f"{'='*60}")
    print()
    print("  Decompose+Delegate (this example):")
    print("    - Breaks goal into INDEPENDENT sub-goals")
    print("    - Each specialist has DIFFERENT tools and expertise")
    print("    - Specialists run IN PARALLEL (faster wall-clock time)")
    print("    - Aggregator checks for conflicts between specialist outputs")
    print("    - Best for: tasks with clearly separable concerns")
    print()
    print("  Plan-Execute (example 02):")
    print("    - Single agent executes SEQUENTIAL steps")
    print("    - Each step can build on previous results")
    print("    - Best for: tasks where steps depend on each other")
    print()
    print("  ReAct (example 02c):")
    print("    - Single agent thinks and acts step by step")
    print("    - No upfront plan — reacts to observations")
    print("    - Best for: exploratory research, unknown scope")
    print()
    print("  When to use Decompose+Delegate:")
    print("    - Sub-tasks are genuinely independent (no data dependencies)")
    print("    - Different sub-tasks need different expertise/tools")
    print("    - You want speed (parallel execution)")
    print("    - Example: event planning, report writing (sections in parallel),")
    print("               multi-market analysis, polyglot code generation")
    print(f"{'='*60}")
