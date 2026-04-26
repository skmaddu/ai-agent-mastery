import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 14: LangGraph State for Memory -- Persistent SQLite Checkpointing
===========================================================================
LangGraph implementation of stateful memory with REAL database persistence.

Covers:
  1. TypedDict State + Reducers
  2. SQLite Persistence (SqliteSaver -- survives process restarts)
  3. Cross-Session Memory (stop and resume conversations)
  4. Time-Travel Debugging (inspect any past state)
  5. PostgreSQL Migration Path (for production)

This is THE production pattern for memory in LangGraph: use TypedDict
state as your memory store, SqliteSaver for persistence, and
reducers for controlled state updates.

UPGRADE FROM MemorySaver:
  MemorySaver = in-memory only, lost on process restart
  SqliteSaver = persisted to disk, survives restarts (THIS EXAMPLE)
  PostgresSaver = production-grade, multi-process, cloud-ready

Requirements:
  pip install langgraph-checkpoint-sqlite

Phoenix tracing: YES

Run: python week-05-context-memory/examples/example_14_langgraph_state_memory.py
"""

import os
import sys
import sqlite3
import textwrap
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages

# ── Phoenix ────────────────────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week5-langgraph-state")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None


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


# ================================================================
# 1. TypedDict STATE + REDUCERS
# ================================================================
# LangGraph state serves as the agent's memory.  Each field can
# have a REDUCER that controls how updates are merged.
#
# The add_messages reducer APPENDS new messages to the existing list.
# Without a reducer, updates REPLACE the entire field.
#
# This gives you fine-grained control:
#   - messages: append (add_messages reducer) -- conversation grows
#   - summary: replace (no reducer) -- latest summary overwrites
#   - facts: append (custom reducer) -- facts accumulate
#   - preferences: replace -- latest preferences win

def merge_facts(existing: List[str], new: List[str]) -> List[str]:
    """
    Custom reducer for facts: append new facts, deduplicate.

    Reducers receive the existing value and the new value,
    and return the merged result.  This is called automatically
    by LangGraph when a node returns a partial state update.
    """
    combined = list(existing)
    for fact in new:
        if fact not in combined:
            combined.append(fact)
    return combined


class MemoryState(TypedDict):
    # Conversation messages — appended via add_messages reducer
    messages: Annotated[list, add_messages]

    # Running summary of older conversation (replaced on each update)
    summary: str

    # Accumulated facts about the user (appended, deduplicated)
    facts: Annotated[List[str], merge_facts]

    # User preferences (replaced with latest)
    preferences: Dict[str, str]

    # Current user input (replaced each turn)
    user_input: str


# ================================================================
# 2. GRAPH NODES
# ================================================================

llm = get_llm(temperature=0.7)
fact_extractor_llm = get_llm(temperature=0)


def extract_facts_node(state: MemoryState) -> dict:
    """
    Extract factual statements from the user's message.

    This node runs on every turn and pulls out persistent facts:
    preferences, constraints, personal info, etc.
    """
    user_input = state["user_input"]

    # Simple heuristic extraction (production: use LLM)
    new_facts = []
    keywords_to_facts = {
        "budget": f"User mentioned budget: '{user_input[:60]}'",
        "allergic": f"User has allergy: '{user_input[:60]}'",
        "vegetarian": "User is vegetarian",
        "vegan": "User is vegan",
        "prefer": f"User preference: '{user_input[:60]}'",
    }
    for keyword, fact in keywords_to_facts.items():
        if keyword in user_input.lower():
            new_facts.append(fact)

    if new_facts:
        print(f"  [FACTS] Extracted: {new_facts}")

    return {"facts": new_facts}


def respond_node(state: MemoryState) -> dict:
    """
    Generate a response using all memory layers.

    The prompt includes:
      1. System instructions
      2. Known facts about the user
      3. User preferences
      4. Conversation summary (if any)
      5. Recent messages (via add_messages)
      6. Current user input
    """
    user_input = state["user_input"]
    facts = state.get("facts", [])
    preferences = state.get("preferences", {})
    summary = state.get("summary", "")

    # Build system prompt with memory context
    system_parts = ["You are a helpful travel planning assistant."]

    if facts:
        system_parts.append(f"\nKnown facts about the user:\n" +
                           "\n".join(f"  • {f}" for f in facts))
    if preferences:
        system_parts.append(f"\nUser preferences:\n" +
                           "\n".join(f"  • {k}: {v}" for k, v in preferences.items()))
    if summary:
        system_parts.append(f"\nConversation summary:\n  {summary}")

    system_parts.append("\nUse the above context to personalize your response.")

    prompt = [SystemMessage(content="\n".join(system_parts))]

    # Add recent messages from state
    for msg in state.get("messages", [])[-6:]:  # Last 3 pairs
        prompt.append(msg)

    prompt.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [RESPOND] {answer[:100]}...")

    return {
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ],
    }


# ================================================================
# 3. GRAPH WITH CHECKPOINTING
# ================================================================

# ================================================================
# SQLITE DATABASE PATH
# ================================================================
# The SQLite database file stores ALL conversation state persistently.
# It survives process restarts -- you can stop the script, come back
# hours later, and resume the exact same conversation.

DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "langgraph_memory.db"


def build_memory_graph(use_sqlite: bool = True):
    """
    Build a stateful graph with PERSISTENT SQLite checkpointing.

    SqliteSaver stores the COMPLETE state in a real SQLite database.
    This enables:
      - Cross-turn persistence (state carries across invocations)
      - Cross-SESSION persistence (survives process restarts!)
      - Time-travel debugging (inspect state at any past step)
      - Session restore (continue where you left off)

    Persistence hierarchy:
      MemorySaver   -- in-memory only, lost on restart (dev/testing)
      SqliteSaver   -- file-based, survives restarts (THIS EXAMPLE)
      PostgresSaver -- server-based, multi-process (production)

    Args:
        use_sqlite: If True, use SQLite (persistent). If False, use
                    MemorySaver (in-memory, for testing).
    """
    graph = StateGraph(MemoryState)

    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("extract_facts")
    graph.add_edge("extract_facts", "respond")
    graph.add_edge("respond", END)

    if use_sqlite:
        from langgraph.checkpoint.sqlite import SqliteSaver
        # SqliteSaver takes a sqlite3 connection object.
        # The database file persists across process restarts.
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        print(f"  [DB] Using SQLite: {DB_PATH}")
        print(f"  [DB] State persists across process restarts!")
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        print(f"  [DB] Using in-memory (state lost on restart)")

    return graph.compile(checkpointer=checkpointer)


# ================================================================
# DEMO: Multi-turn with persistence
# ================================================================

def run_demo():
    """
    Demonstrate stateful multi-turn conversation with SQLite persistence.

    KEY: The thread_id ties multiple invocations to the same conversation.
    LangGraph automatically loads the previous state from SQLite when
    you invoke with the same thread_id -- even across process restarts!
    """
    app = build_memory_graph(use_sqlite=True)

    # Use a FIXED thread_id so you can resume across process restarts.
    # Try this: run the script once, then run it again -- the second
    # run will pick up where the first left off because the state
    # is persisted in SQLite.
    thread_id = "japan-trip-session-001"
    config = {"configurable": {"thread_id": thread_id}}

    # Check if we have existing state from a previous run
    existing_state = None
    try:
        existing_state = app.get_state(config)
        if existing_state and existing_state.values.get("messages"):
            prev_facts = existing_state.values.get("facts", [])
            prev_msgs = existing_state.values.get("messages", [])
            print(f"\n  [DB] RESUMED existing session from SQLite!")
            print(f"  [DB] Previous state: {len(prev_msgs)} messages, "
                  f"{len(prev_facts)} facts")
            if prev_facts:
                print(f"  [DB] Known facts from last session:")
                for f in prev_facts:
                    print(f"         - {f}")
    except Exception:
        pass

    user_turns = [
        "Hi! I'm planning a trip to Japan next spring.",
        "My budget is about $3000 for 2 weeks. I'm vegetarian.",
        "What cities should I visit?",
        "Do you remember my dietary requirements?",  # Tests memory recall
    ]

    print(f"\n{'=' * 65}")
    print(f"  LANGGRAPH STATE + SQLITE PERSISTENCE DEMO")
    print(f"{'=' * 65}")
    print(f"  Thread ID: {thread_id}")
    print(f"  Database:  {DB_PATH}")

    for i, user_input in enumerate(user_turns):
        print(f"\n{'-' * 65}")
        print(f"  Turn {i + 1}: {user_input}")
        print(f"{'-' * 65}")

        result = app.invoke(
            {
                "user_input": user_input,
                "messages": [],
                "summary": "",
                "facts": [],
                "preferences": {},
            },
            config,
        )

        # Show accumulated facts
        facts = result.get("facts", [])
        if facts:
            print(f"  [MEMORY] Known facts: {facts}")

        # Show last AI response
        messages = result.get("messages", [])
        if messages:
            last_ai = [m for m in messages if isinstance(m, AIMessage)]
            if last_ai:
                print(f"  [AI] {last_ai[-1].content[:200]}")

    # === TIME-TRAVEL DEBUGGING ===
    print(f"\n{'=' * 65}")
    print(f"  TIME-TRAVEL: Inspecting historical state from SQLite")
    print(f"{'=' * 65}")

    try:
        checkpoints = list(app.checkpointer.list(config))
        print(f"  Found {len(checkpoints)} checkpoints in SQLite for '{thread_id}'")
        for i, cp in enumerate(checkpoints[:5]):
            step = cp.metadata.get("step", "?") if cp.metadata else "?"
            source = cp.metadata.get("source", "?") if cp.metadata else "?"
            print(f"    Checkpoint {i}: step={step}, source={source}")
    except Exception as e:
        print(f"  [Note] Checkpoint listing: {e}")

    # === SQLITE DATABASE INSPECTION ===
    print(f"\n{'=' * 65}")
    print(f"  SQLITE DATABASE INSPECTION")
    print(f"{'=' * 65}")

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Show tables created by SqliteSaver
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"  Tables in {DB_PATH.name}: {tables}")

        # Show row counts
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cursor.fetchone()[0]
            print(f"    {table}: {count} rows")

        # Show database file size
        db_size = DB_PATH.stat().st_size
        print(f"\n  Database file size: {db_size / 1024:.1f} KB")
        print(f"  Location: {DB_PATH.absolute()}")

        conn.close()
    except Exception as e:
        print(f"  [Note] DB inspection: {e}")

    # === CROSS-SESSION PERSISTENCE DEMO ===
    print(f"\n{'=' * 65}")
    print(f"  CROSS-SESSION PERSISTENCE")
    print(f"{'=' * 65}")
    print(f"""
  The state is now saved in SQLite. To prove persistence:

  1. This script just saved the conversation to:
     {DB_PATH}

  2. Run this script AGAIN -- it will detect the existing session
     and show "RESUMED existing session from SQLite!" at the top.

  3. To start fresh, delete the database:
     rm {DB_PATH}

  This is the key difference from MemorySaver:
    MemorySaver  = state lost when process exits
    SqliteSaver  = state persists across restarts (THIS EXAMPLE)
    PostgresSaver = state persists + multi-process + cloud-ready
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 14: LangGraph State Memory (SQLite)")
    print("=" * 65)

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. LangGraph state IS memory.  TypedDict fields store conversation
       messages, facts, preferences, and summaries -- all explicit.

    2. Reducers control how state updates merge: add_messages APPENDS,
       custom reducers can deduplicate, and no reducer REPLACES.

    3. SqliteSaver persists state to a REAL database file.  Use the same
       thread_id to resume conversations even after process restarts.

    4. Time-travel debugging lets you inspect state at any past step --
       invaluable for debugging why an agent made a wrong decision.

    5. Persistence hierarchy for different deployment scenarios:
         MemorySaver   -- dev/testing (in-memory, lost on restart)
         SqliteSaver   -- single-user (file-based, THIS EXAMPLE)
         PostgresSaver -- production (server-based, multi-process)

    6. To use PostgreSQL in production:
         pip install langgraph-checkpoint-postgres
         from langgraph.checkpoint.postgres import PostgresSaver
         checkpointer = PostgresSaver(conn_string="postgresql://...")
    """))
