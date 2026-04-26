import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 15: ADK Workflows for Memory
=======================================
Google ADK implementation of memory-aware agent workflows.

Covers:
  1. YAML-Configured Workflows & Memory Injection
  2. Callback-Based Memory Handling in ADK

ADK manages memory through sessions and state.  Unlike LangGraph's
explicit TypedDict state, ADK uses InMemorySessionService for
within-session memory and custom tools for cross-session persistence.

Run: python week-05-context-memory/examples/example_15_adk_memory_workflows.py
"""

import os
import sys
import json
import asyncio
import textwrap
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# ── Phoenix ────────────────────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        register(project_name="week5-adk-memory")
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None

# ── ADK ────────────────────────────────────────────────────────
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("[ADK] Not installed. pip install google-adk")


# ================================================================
# 1. PERSISTENT MEMORY STORE (JSON-based)
# ================================================================
# ADK's InMemorySessionService is ephemeral — it dies when the
# process exits.  For cross-session memory, we use a simple JSON
# file store.  The agent interacts with it through tools.

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "memory_store.json")


def _load_memory() -> Dict:
    """Load memory from JSON file."""
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"facts": [], "preferences": {}, "history": []}


def _save_memory(data: Dict):
    """Save memory to JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(MEMORY_FILE) or ".", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# Need this import for type hints in tools
from typing import Dict, Optional


# ================================================================
# ADK TOOLS FOR MEMORY
# ================================================================

def remember_fact(fact: str, importance: str = "medium") -> str:
    """
    Store a fact about the user in long-term memory.

    Call this whenever you learn something important about the user
    that should be remembered across sessions (preferences, constraints,
    personal info, past decisions).

    Args:
        fact: The fact to remember (e.g., "User is vegetarian").
        importance: How important this is: "low", "medium", or "high".

    Returns:
        Confirmation that the fact was stored.
    """
    data = _load_memory()

    # Deduplication check
    for existing in data["facts"]:
        if isinstance(existing, dict) and existing.get("text", "").lower() == fact.lower():
            return f"Already remembered: '{fact}'"

    data["facts"].append({
        "text": fact,
        "importance": importance,
        "created_at": datetime.now().isoformat(),
    })
    _save_memory(data)
    print(f"  [TOOL] remember_fact('{fact}', importance={importance})")
    return f"Remembered: '{fact}' (importance: {importance})"


def recall_facts(topic: str = "") -> str:
    """
    Recall stored facts from long-term memory.

    Call this at the start of each conversation to load relevant
    context about the user, and whenever you need to reference
    past information.

    Args:
        topic: Optional topic filter. If provided, only returns facts
               related to this topic. If empty, returns all facts.

    Returns:
        JSON string with all matching facts from memory.
    """
    data = _load_memory()
    facts = data.get("facts", [])

    if topic:
        topic_lower = topic.lower()
        facts = [f for f in facts
                 if isinstance(f, dict) and topic_lower in f.get("text", "").lower()]

    print(f"  [TOOL] recall_facts('{topic}') → {len(facts)} facts")
    return json.dumps({"facts": facts, "total": len(facts)})


def update_preference(key: str, value: str) -> str:
    """
    Update a user preference in long-term memory.

    Preferences are key-value pairs that persist across sessions.
    Examples: diet="vegetarian", budget_style="moderate", language="Japanese basics".

    Args:
        key: The preference name (e.g., "diet", "budget", "travel_style").
        value: The preference value.

    Returns:
        Confirmation of the update.
    """
    data = _load_memory()
    old_value = data["preferences"].get(key)
    data["preferences"][key] = value
    _save_memory(data)
    print(f"  [TOOL] update_preference({key}={value})")
    if old_value:
        return f"Updated preference: {key} changed from '{old_value}' to '{value}'"
    return f"Set preference: {key} = '{value}'"


def get_preferences() -> str:
    """
    Get all stored user preferences.

    Call this at the start of conversations to load the user's
    known preferences and personalize your responses.

    Returns:
        JSON string with all stored preferences.
    """
    data = _load_memory()
    prefs = data.get("preferences", {})
    print(f"  [TOOL] get_preferences() → {len(prefs)} preferences")
    return json.dumps(prefs)


# ================================================================
# 2. ADK AGENT WITH MEMORY TOOLS
# ================================================================

def build_memory_agent():
    """
    Build an ADK agent with persistent memory capabilities.

    The agent's instruction includes a MEMORY PROTOCOL that tells
    it when and how to use memory tools.
    """
    agent = LlmAgent(
        name="memory_travel_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are a travel planning assistant with persistent memory.

MEMORY PROTOCOL (follow this every turn):
1. At the START of each conversation, call recall_facts() and get_preferences()
   to load what you know about the user.
2. When the user mentions a preference or personal fact, call remember_fact()
   to store it, and update_preference() if it's a clear preference.
3. Use recalled facts and preferences to personalize your responses.
4. Reference specific memories naturally: "I remember you mentioned..."

RULES:
- Always acknowledge when you remember something from a previous session.
- If preferences conflict with a new request, ask for clarification.
- Store important facts with appropriate importance levels:
  high: allergies, medical conditions, hard constraints
  medium: preferences, budget, travel style
  low: casual mentions, nice-to-haves""",
        tools=[remember_fact, recall_facts, update_preference, get_preferences],
        description="Travel assistant with persistent cross-session memory.",
    )
    return agent


# ================================================================
# RUNNER
# ================================================================

async def run_agent_turn(agent, message: str, retries: int = 5) -> str:
    """Run one turn of conversation with retry logic for Gemini API errors."""
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="week5_adk_memory",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="week5_adk_memory",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=message)]
                ),
            ):
                if event.is_final_response() and event.content and event.content.parts:
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

    return "[Error: unexpected]"


async def run_demo():
    """Demonstrate ADK memory across simulated sessions."""
    if not ADK_AVAILABLE:
        print("[SKIP] ADK not available.")
        return

    # Clean up any previous memory for clean demo
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    agent = build_memory_agent()

    # Session 1: User provides preferences
    print("\n" + "=" * 65)
    print("  SESSION 1: Learning User Preferences")
    print("=" * 65)

    session1_turns = [
        "Hi! I'm planning trips for this year. I'm vegetarian and allergic to shellfish.",
        "My travel budget is usually around $3000. I prefer boutique hotels over hostels.",
    ]

    for turn in session1_turns:
        print(f"\n  User: {turn}")
        response = await run_agent_turn(agent, turn)
        print(f"  Agent: {response[:200]}")

    # Session 2: Agent should recall preferences
    print("\n" + "=" * 65)
    print("  SESSION 2: Recalling from Memory")
    print("=" * 65)

    session2_turns = [
        "I'm thinking about a trip to Japan. What should I know?",
        "Can you remind me of any dietary restrictions you have on file for me?",
    ]

    for turn in session2_turns:
        print(f"\n  User: {turn}")
        response = await run_agent_turn(agent, turn)
        print(f"  Agent: {response[:200]}")

    # Show final memory state
    print(f"\n{'=' * 65}")
    print(f"  FINAL MEMORY STATE")
    print(f"{'=' * 65}")
    data = _load_memory()
    print(f"  Facts ({len(data.get('facts', []))}):")
    for f in data.get("facts", []):
        if isinstance(f, dict):
            print(f"    • {f.get('text', '')} (importance: {f.get('importance', '?')})")
    print(f"  Preferences ({len(data.get('preferences', {}))}):")
    for k, v in data.get("preferences", {}).items():
        print(f"    • {k}: {v}")

    # Cleanup
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 15: ADK Memory Workflows                   ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()

    if ADK_AVAILABLE:
        asyncio.run(run_demo())
    else:
        print("\n  Install Google ADK: pip install google-adk")

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. ADK manages within-session memory via InMemorySessionService.
       For cross-session persistence, add tools that read/write to
       external storage (JSON, database, vector store).

    2. The MEMORY PROTOCOL in the agent instruction is crucial — it
       tells the agent WHEN to load, store, and reference memories.

    3. JSON file storage works for single-user prototypes.  For
       production, use SQLite or PostgreSQL.

    4. Importance levels (high/medium/low) help prioritize which
       memories to recall when context space is limited.

    5. ADK vs LangGraph for memory:
       - LangGraph: Memory is STATE — explicit, checkpointed, typed
       - ADK: Memory is TOOLS — agent-driven, flexible, external storage
    """))
