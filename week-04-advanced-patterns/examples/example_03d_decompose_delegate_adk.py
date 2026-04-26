import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3d: Decompose + Delegate with Parallel Agents (ADK)
=============================================================
ADK counterpart of example_02d_decompose_delegate_parallel.py.

Same pattern — DECOMPOSE a goal into independent sub-goals, DELEGATE
each to a specialist agent with its own tools, run them IN PARALLEL,
then AGGREGATE the results — but using Google ADK instead of LangGraph.

Key ADK advantage: agents are async-native, so true parallelism comes
from asyncio.gather() instead of ThreadPoolExecutor.

Architecture: DECOMPOSER → 3 SPECIALISTS in parallel → AGGREGATOR

Run: python week-04-advanced-patterns/examples/example_03d_decompose_delegate_adk.py
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

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")

# Use 3 different Gemini models for parallel specialists to avoid
# 503 "high demand" errors when hitting the same model endpoint
# simultaneously. The decomposer and aggregator run sequentially
# so they can share one model.
SPECIALIST_MODELS = [
    os.getenv("GOOGLE_MODEL_1", "gemini-3-flash-preview"),
    os.getenv("GOOGLE_MODEL_2", "gemini-3.1-flash-lite-preview"),
    os.getenv("GOOGLE_MODEL_3", "gemini-3.1-flash-lite-preview"),
]


# ==============================================================
# Step 1: Specialist Tools (plain functions for ADK)
# ==============================================================
# ADK auto-generates tool schemas from name, docstring, type hints.
def search_venues(event_type: str) -> str:
    """Search for event venues by type.
    Args:
        event_type: Type of event (e.g., 'indoor', 'outdoor', 'restaurant')
    """
    venues = {
        "indoor": [
            "City Banquet Hall — capacity 50, $500 rental, full kitchen, AV system",
            "The Loft Space — capacity 30, $300 rental, open plan, BYO catering",
            "Community Center — capacity 80, $150 rental, basic kitchen, parking",
        ],
        "outdoor": [
            "Riverside Park Pavilion — capacity 40, $100 rental, covered, grills",
            "Rooftop Garden — capacity 25, $400 rental, skyline views, no kitchen",
            "Botanical Gardens — capacity 60, $350 rental, photography friendly",
        ],
        "restaurant": [
            "Luigi's Private Room — capacity 20, $0 rental (min spend $800), Italian",
            "The Modern Bistro — capacity 15, $200 rental, prix fixe $45/person",
            "Harbor View — capacity 30, $0 rental (min spend $500), seafood",
        ],
    }
    query = event_type.lower()
    for key, items in venues.items():
        if key in query:
            return f"Venues for '{event_type}':\n" + "\n".join(f"  - {v}" for v in items)
    all_venues = []
    for category, items in venues.items():
        all_venues.extend(f"  - [{category}] {v}" for v in items)
    return "All available venues:\n" + "\n".join(all_venues)

def check_venue_availability(venue_name: str) -> str:
    """Check if a specific venue is available and get details.
    Args:
        venue_name: Name of the venue to check
    """
    details = {
        "community center": "AVAILABLE. $150 rental. Tables/chairs for 80, basic kitchen, parking.",
        "riverside": "AVAILABLE. $100 rental. Covered pavilion, 3 BBQ grills, picnic tables for 40.",
        "loft": "AVAILABLE. $300 rental. Open-plan, decorating OK, no kitchen, 10 PM curfew.",
        "banquet": "AVAILABLE. $500 rental. Full kitchen, round tables, sound system, late hours OK.",
        "botanical": "AVAILABLE. $350 rental. Garden setting, max 60, approved vendor list only.",
    }
    for key, info in details.items():
        if key in venue_name.lower():
            return f"Venue '{venue_name}': {info}"
    return f"Venue '{venue_name}' not found in our database."

def search_caterers(cuisine_type: str) -> str:
    """Search for catering options by cuisine type.
    Args:
        cuisine_type: Type of cuisine (e.g., 'italian', 'bbq', 'asian', 'mixed')
    """
    caterers = {
        "italian": "Italian Catering:\n  1. Mama Rosa's — $18/person, family-style pasta + salad + bread\n  2. Tuscan Table — $25/person, antipasto + main + dessert, GF options\n  3. Pizza Party Co — $12/person, assorted pizzas + sides",
        "bbq": "BBQ Catering:\n  1. Smokey Joe's — $20/person, brisket + ribs + 3 sides\n  2. Backyard BBQ Co — $15/person, burgers + hot dogs + sides\n  3. Grill Masters — $22/person, premium meats + vegetarian options",
        "asian": "Asian Catering:\n  1. Golden Dragon — $16/person, stir-fry buffet, all GF\n  2. Sushi Fresh — $28/person, sushi platters\n  3. Thai Express — $14/person, curry + noodles, vegan available",
        "mixed": "Mixed Catering:\n  1. All Occasions — $20/person, customizable, GF/vegan OK\n  2. Budget Bites — $10/person, sandwich + soup + salad bar\n  3. Elegant Eats — $35/person, 3-course plated dinner",
    }
    for key, result in caterers.items():
        if key in cuisine_type.lower():
            return result
    return f"No caterers found for '{cuisine_type}'. Try: italian, bbq, asian, mixed."

def calculate_food_budget(per_person_cost: float, guest_count: int, extras: str) -> str:
    """Calculate total food budget including extras.
    Args:
        per_person_cost: Cost per person for catering
        guest_count: Number of guests
        extras: Comma-separated extras like 'drinks, cake, ice'
    """
    food_total = per_person_cost * guest_count
    extra_costs = {
        "drink": 3.0, "drinks": 3.0 * guest_count, "cake": 40.0,
        "ice": 15.0, "napkins": 10.0, "plates": 12.0, "cups": 8.0,
        "utensils": 10.0, "decorations": 25.0, "flowers": 30.0,
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
    result = f"Food Budget Calculation:\n  Catering: ${per_person_cost:.2f} x {guest_count} = ${food_total:.2f}\n"
    if extras_breakdown:
        result += "  Extras:\n" + "\n".join(extras_breakdown) + "\n"
    result += f"  TOTAL: ${total:.2f}"
    return result

def search_entertainment(event_type: str) -> str:
    """Search for entertainment options for an event.
    Args:
        event_type: Type of event (e.g., 'birthday', 'casual', 'formal', 'outdoor')
    """
    options = {
        "birthday": "Birthday Entertainment:\n  1. DJ Package — $200, 3hrs\n  2. Photo Booth — $150, 2hrs, props+prints\n  3. Karaoke Setup — $100, 4hrs\n  4. Lawn Games Kit — $50, cornhole+Jenga+bocce",
        "outdoor": "Outdoor Entertainment:\n  1. Lawn Games Kit — $50, cornhole+Jenga+bocce\n  2. Portable Speaker+Playlist — $0 (DIY)\n  3. Outdoor Movie Setup — $75, projector+screen\n  4. Scavenger Hunt Kit — $25",
        "casual": "Casual Entertainment:\n  1. Spotify Playlist+Speaker — $0 (DIY)\n  2. Board Games — $30 rental\n  3. Trivia Night Kit — $40\n  4. Movie Projector — $75",
        "formal": "Formal Entertainment:\n  1. Live Jazz Trio — $500, 3hrs\n  2. String Quartet — $600, 2hrs\n  3. Professional MC — $300\n  4. Magician — $250, 1.5hrs",
    }
    for key, result in options.items():
        if key in event_type.lower():
            return result
    return "Entertainment for '" + event_type + "'. Try: birthday, casual, formal, outdoor."

def check_entertainment_cost(items: str) -> str:
    """Calculate total entertainment budget from a list of items.
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
    return "Entertainment Budget:\n" + "\n".join(parsed) + f"\n  TOTAL: ${total:.2f}"

# ==============================================================
# Step 2: Create Specialist Agents (each with its OWN tools)
# ==============================================================

decomposer_agent = LlmAgent(
    name="decomposer",
    model=MODEL,
    instruction=(
        "You are a goal decomposition specialist. Break the event planning goal "
        "into exactly 3 independent sub-goals for parallel work:\n"
        "  1. VENUE — finding and evaluating the event location\n"
        "  2. MENU — planning food, drinks, and dietary accommodations\n"
        "  3. ENTERTAINMENT — planning activities, music, and fun\n\n"
        "Include relevant constraints (budget, guest count, preferences).\n\n"
        "Output format (exactly 3 lines):\n"
        "VENUE: <sub-goal with constraints>\n"
        "MENU: <sub-goal with constraints>\n"
        "ENTERTAINMENT: <sub-goal with constraints>"
    ),
    description="Breaks a complex goal into 3 independent sub-goals.",
)

venue_agent = LlmAgent(
    name="venue_specialist",
    model=SPECIALIST_MODELS[0],
    instruction=(
        "You are a venue specialist. Search venues and check availability. "
        "Consider capacity, cost, amenities, restrictions. "
        "Provide a clear recommendation with reasoning and cost."
    ),
    tools=[search_venues, check_venue_availability],
    description="Finds and evaluates event venues.",
)

menu_agent = LlmAgent(
    name="menu_specialist",
    model=SPECIALIST_MODELS[1],
    instruction=(
        "You are a catering specialist. Search caterers and calculate costs. "
        "Consider dietary restrictions (vegetarian, gluten-free), cuisine "
        "preferences, and budget. Provide a food plan with budget breakdown."
    ),
    tools=[search_caterers, calculate_food_budget],
    description="Plans menus and catering with budget calculations.",
)

entertainment_agent = LlmAgent(
    name="entertainment_specialist",
    model=SPECIALIST_MODELS[2],
    instruction=(
        "You are an entertainment specialist. Search entertainment options and "
        "calculate costs. Consider event type, guest preferences, venue type, "
        "and budget. Provide a fun activity plan with cost breakdown."
    ),
    tools=[search_entertainment, check_entertainment_cost],
    description="Plans activities and entertainment.",
)

aggregator_agent = LlmAgent(
    name="aggregator",
    model=MODEL,
    instruction=(
        "You are an event planning aggregator. Combine results from three "
        "specialist agents (venue, menu, entertainment) into ONE cohesive plan.\n\n"
        "Your job:\n"
        "1. Combine outputs into a unified plan\n"
        "2. Check for CONFLICTS (e.g., venue has no kitchen but menu needs one)\n"
        "3. Calculate TOTAL budget across all areas\n"
        "4. Flag issues or missing items\n\n"
        "Format: Venue, Menu, Entertainment, Budget Summary, Notes/Conflicts."
    ),
    description="Combines specialist outputs into a unified event plan.",
)

# ==============================================================
# Step 3: Async Agent Runner Helper
# ==============================================================
async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent and return its final text response.
    Fresh session per call. Retries on transient errors.
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="decompose_delegate_demo",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="decompose_delegate_demo",
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
                return f"[Error: API unavailable after {retries} retries]"

# ==============================================================
# Step 4: Parse Sub-Goals from Decomposer Output
# ==============================================================
def parse_sub_goals(raw_text: str) -> dict:
    """Extract VENUE/MENU/ENTERTAINMENT sub-goals from decomposer output."""
    sub_goals = {"venue": "", "menu": "", "entertainment": ""}
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("VENUE"):
            sub_goals["venue"] = re.sub(r'^VENUE[:\s]*', '', line, flags=re.IGNORECASE).strip()
        elif line.upper().startswith("MENU"):
            sub_goals["menu"] = re.sub(r'^MENU[:\s]*', '', line, flags=re.IGNORECASE).strip()
        elif line.upper().startswith("ENTERTAINMENT"):
            sub_goals["entertainment"] = re.sub(
                r'^ENTERTAINMENT[:\s]*', '', line, flags=re.IGNORECASE
            ).strip()
    # Fallbacks
    if not sub_goals["venue"]:
        sub_goals["venue"] = "Find a suitable venue for the event within budget"
    if not sub_goals["menu"]:
        sub_goals["menu"] = "Plan a menu with catering within the food budget"
    if not sub_goals["entertainment"]:
        sub_goals["entertainment"] = "Plan entertainment activities for guests"
    return sub_goals

# ==============================================================
# Step 5: Run a Single Specialist (for asyncio.gather)
# ==============================================================
async def run_specialist(agent: LlmAgent, name: str, task: str, goal: str) -> tuple:
    """Run one specialist and return (name, result) for identification."""
    print(f"\n  >> Launching {name}...")
    message = (
        f"Overall event goal: {goal}\n\n"
        f"Your specific task: {task}\n\n"
        f"Use your tools to research options, then provide a clear "
        f"recommendation with cost breakdown."
    )
    result = await run_agent(agent, message)
    lines = result.strip().split("\n")
    print(f"\n  << {name} completed ({len(result)} chars)")
    for line in lines[:5]:
        if line.strip():
            print(f"     {line.strip()[:100]}")
    if len(lines) > 5:
        print(f"     ... ({len(lines) - 5} more lines)")
    return (name, result)

# ==============================================================
# Step 6: Main — Decompose, Delegate in Parallel, Aggregate
# ==============================================================
async def main():
    """Run the full Decompose + Delegate with Parallel Agents pattern."""
    print("Example 3d: Decompose + Delegate with Parallel Agents (ADK)")
    print("=" * 60)
    print("Three specialist agents work IN PARALLEL via asyncio.gather(),")
    print("then an aggregator combines their results.")
    print("=" * 60)

    goal = (
        "Plan a birthday party for 20 people with a total budget of $800. "
        "The birthday person loves Italian food and outdoor activities. "
        "Two guests are vegetarian and one is gluten-free. "
        "The party is on a Saturday afternoon."
    )
    print(f"\nGoal: {goal}")

    # Phase 1: DECOMPOSE
    print(f"\n{'='*60}")
    print("  PHASE 1: DECOMPOSE — Breaking goal into sub-goals...")
    print(f"{'='*60}")

    decompose_text = await run_agent(
        decomposer_agent,
        f"Break this event planning goal into sub-goals: {goal}",
    )
    sub_goals = parse_sub_goals(decompose_text)

    print(f"\n  Sub-goals (each assigned to a specialist):")
    for role, task in sub_goals.items():
        print(f"    [{role.upper()}] {task[:100]}")

    # Phase 2: DELEGATE — Run 3 specialists IN PARALLEL
    # Key difference from LangGraph (example_02d): asyncio.gather vs ThreadPoolExecutor.
    # ADK agents are async-native, so gather() is more natural — no threads, no GIL.
    print(f"\n{'='*60}")
    print("  PHASE 2: DELEGATE — Running 3 specialists in parallel...")
    print(f"{'='*60}")

    results = await asyncio.gather(
        run_specialist(venue_agent, "VENUE SPECIALIST", sub_goals["venue"], goal),
        run_specialist(menu_agent, "MENU SPECIALIST", sub_goals["menu"], goal),
        run_specialist(entertainment_agent, "ENTERTAINMENT SPECIALIST",
                       sub_goals["entertainment"], goal),
    )

    specialist_results = {name: result for name, result in results}
    venue_result = specialist_results.get("VENUE SPECIALIST", "No venue result")
    menu_result = specialist_results.get("MENU SPECIALIST", "No menu result")
    entertainment_result = specialist_results.get("ENTERTAINMENT SPECIALIST", "No result")

    # Phase 3: AGGREGATE
    print(f"\n{'='*60}")
    print("  PHASE 3: AGGREGATE — Combining specialist results...")
    print(f"{'='*60}")

    aggregate_message = (
        f"Original Goal: {goal}\n\n"
        f"--- VENUE SPECIALIST REPORT ---\n{venue_result}\n\n"
        f"--- MENU SPECIALIST REPORT ---\n{menu_result}\n\n"
        f"--- ENTERTAINMENT SPECIALIST REPORT ---\n{entertainment_result}\n\n"
        f"Create the combined event plan. Check for conflicts and verify "
        f"the total stays within the $800 budget."
    )
    final_plan = await run_agent(aggregator_agent, aggregate_message)
    print(f"\n  Final plan: {len(final_plan)} characters")

    # Print Final Plan
    print(f"\n\n{'#'*60}")
    print("  COMBINED EVENT PLAN")
    print(f"{'#'*60}")
    print(f"\n{final_plan}")

    # Framework Comparison
    print(f"\n{'='*60}")
    print("Decompose + Delegate: LangGraph vs ADK")
    print(f"{'='*60}")
    print()
    print("  LangGraph (example_02d):")
    print("    - Parallelism: ThreadPoolExecutor (thread-based)")
    print("    - State: TypedDict + StateGraph with reducers")
    print("    - Tools: Manual ToolNode + message wiring per specialist")
    print("    - Structure: decomposer -> parallel_node -> aggregator -> END")
    print()
    print("  ADK (this example):")
    print("    - Parallelism: asyncio.gather() (async-native, no threads)")
    print("    - State: Plain Python variables (simpler, less structured)")
    print("    - Tools: Runner auto-manages tool call loop internally")
    print("    - Structure: Python code — decompose -> gather -> aggregate")
    print()
    print("  When to choose which:")
    print("    - LangGraph: Graph visualization, checkpointing, complex branching")
    print("    - ADK: Async-native parallelism, less boilerplate, Python control")
    print()
    print("  Decompose + Delegate pattern recap:")
    print("    - Breaks goal into INDEPENDENT sub-goals (not sequential steps)")
    print("    - Specialists have DIFFERENT tools and expertise")
    print("    - Parallel execution cuts wall-clock time")
    print("    - Aggregator resolves conflicts between specialist outputs")
    print("    - Best for: event planning, reports, multi-market analysis")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
