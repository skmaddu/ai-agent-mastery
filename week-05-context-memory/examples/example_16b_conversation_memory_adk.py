import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 16b: Conversation Chains with Memory — Google ADK
============================================================
ADK agent with multi-turn conversation memory using session state
and external storage tools.

Run: python week-05-context-memory/examples/example_16b_conversation_memory_adk.py
"""

import os
import sys
import json
import asyncio
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# ── Phoenix ────────────────────────────────────────────────────
try:
    import phoenix as px
    from phoenix.otel import register
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        register(project_name="week5-conversation-adk")
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
# IN-MEMORY CONVERSATION STORE
# ================================================================
# Simulates persistent memory using a module-level dict.
# In production, replace with a database or key-value store.

conversation_store = {
    "facts": [],
    "preferences": {},
    "turn_count": 0,
    "history_summary": "",
}


def save_user_info(key: str, value: str) -> str:
    """
    Save user information to persistent memory.

    Use this when the user shares preferences, constraints, or
    personal info that should be remembered across the conversation.

    Args:
        key: Category (e.g., "diet", "budget", "allergy", "preference").
        value: The information to save.

    Returns:
        Confirmation message.
    """
    conversation_store["preferences"][key] = value
    conversation_store["facts"].append(f"{key}: {value}")
    print(f"  [TOOL] save_user_info({key}={value})")
    return f"Saved to memory: {key} = {value}"


def load_user_info() -> str:
    """
    Load all saved user information from persistent memory.

    Call this at the start of each response to check what you
    already know about the user.

    Returns:
        JSON with all saved facts and preferences.
    """
    result = {
        "facts": conversation_store["facts"],
        "preferences": conversation_store["preferences"],
        "turns_so_far": conversation_store["turn_count"],
        "summary": conversation_store.get("history_summary", ""),
    }
    print(f"  [TOOL] load_user_info() → {len(result['facts'])} facts")
    return json.dumps(result)


def save_conversation_summary(summary: str) -> str:
    """
    Save a conversation summary for future reference.

    Call this periodically (every 4-5 turns) to summarize the
    conversation so far. This helps maintain context even when
    the full history isn't available.

    Args:
        summary: A concise summary of the conversation so far.

    Returns:
        Confirmation message.
    """
    conversation_store["history_summary"] = summary
    print(f"  [TOOL] save_conversation_summary({summary[:50]}...)")
    return f"Summary saved ({len(summary.split())} words)"


# ================================================================
# AGENT
# ================================================================

def build_conversation_agent():
    agent = LlmAgent(
        name="conversation_memory_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are a helpful restaurant recommendation assistant with persistent memory.

MEMORY PROTOCOL:
1. Start EVERY response by calling load_user_info() to check what you know.
2. When the user shares info (diet, budget, allergies, preferences), call save_user_info().
3. Every 4-5 turns, call save_conversation_summary() with a brief summary.
4. Reference your memory naturally: "I remember you mentioned..."
5. If you find contradictions, ask the user to clarify.

Be concise, helpful, and personalized based on memory.""",
        tools=[save_user_info, load_user_info, save_conversation_summary],
        description="Restaurant assistant with conversation memory.",
    )
    return agent


# ================================================================
# RUNNER
# ================================================================

async def run_turn(agent, message: str, retries: int = 5) -> str:
    """Run one conversation turn with retry logic for Gemini API errors."""
    conversation_store["turn_count"] += 1
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(agent=agent, app_name="week5_conv_adk", session_service=session_service)
            session = await session_service.create_session(app_name="week5_conv_adk", user_id="user1")

            result = ""
            async for event in runner.run_async(
                user_id="user1", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    result = event.content.parts[0].text

            return result
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
    if not ADK_AVAILABLE:
        print("[SKIP] ADK not available.")
        return

    # Reset store
    conversation_store.update({"facts": [], "preferences": {}, "turn_count": 0, "history_summary": ""})

    agent = build_conversation_agent()

    turns = [
        "Hi! Looking for restaurant recommendations in Tokyo. I'm vegetarian.",
        "My budget is moderate, around $30-50 per meal.",
        "What about in Shibuya area?",
        "Any places with good desserts too?",
        "Do you remember my dietary restrictions?",
    ]

    print("\n" + "=" * 65)
    print("  ADK CONVERSATION WITH MEMORY")
    print("=" * 65)

    for i, turn in enumerate(turns):
        print(f"\n{'━' * 65}")
        print(f"  Turn {i + 1}: {turn}")
        print(f"{'━' * 65}")
        response = await run_turn(agent, turn)
        print(f"  Agent: {response[:200]}")

    print(f"\n{'=' * 65}")
    print(f"  FINAL MEMORY STATE")
    print(f"{'=' * 65}")
    print(f"  Facts: {conversation_store['facts']}")
    print(f"  Preferences: {conversation_store['preferences']}")
    print(f"  Turns: {conversation_store['turn_count']}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 16b: Conversation Memory (ADK)             ║")
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
    1. ADK conversation memory uses tools for external storage and
       the instruction's MEMORY PROTOCOL to guide when to save/load.

    2. Each session is stateless from ADK's perspective — memory
       persistence comes from the external store (module dict, JSON, DB).

    3. The "do you remember?" test verifies that the agent properly
       loads and references stored information.

    4. Periodic conversation summaries keep context available even
       when full history is lost between sessions.

    5. Both LangGraph and ADK can achieve the same memory behavior;
       they differ in WHERE state lives (graph state vs. external tools).
    """))
