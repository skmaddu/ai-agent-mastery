import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 16: Conversation Chains with Memory — LangGraph
=========================================================
End-to-end LangGraph conversation agent with full memory.

Covers:
  1. End-to-End Conversation Chain Patterns
  2. Real-World Use Cases (Research Agent, Customer Support)

Combines all memory techniques from previous examples:
  - Message buffer with add_messages reducer
  - Fact extraction on every turn
  - Automatic summarization when context grows
  - SqliteSaver checkpointing for real database persistence

Run: python week-05-context-memory/examples/example_16_conversation_memory_langgraph.py
"""

import os
import sys
import sqlite3
import textwrap
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages

# -- SQLite persistence path --
DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "conversation_memory.db"

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
        tracer_provider = register(project_name="week5-conversation-memory")
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
# STATE — Full-featured conversation state
# ================================================================

def merge_unique(existing: List[str], new: List[str]) -> List[str]:
    combined = list(existing)
    for item in new:
        if item not in combined:
            combined.append(item)
    return combined


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    facts: Annotated[List[str], merge_unique]
    summary: str
    turn_count: int
    token_estimate: int


# ================================================================
# GRAPH NODES
# ================================================================

llm = get_llm(temperature=0.7)
summarizer = get_llm(temperature=0)

SUMMARIZE_THRESHOLD = 400  # Trigger summarization above this token estimate


def extract_and_count_node(state: ConversationState) -> dict:
    """Extract facts from user input and estimate token count."""
    user_input = state["user_input"]
    messages = state.get("messages", [])

    # Simple fact extraction
    new_facts = []
    for keyword, pattern in [
        ("budget", "budget"), ("vegetarian", "diet:vegetarian"),
        ("allergic", "allergy"), ("prefer", "preference"),
        ("vegan", "diet:vegan"), ("$", "budget"),
    ]:
        if keyword in user_input.lower():
            new_facts.append(f"{pattern}: {user_input[:60]}")

    # Token estimation
    total = len(state.get("summary", "").split())
    total += sum(len(str(m.content if hasattr(m, 'content') else m).split())
                 for m in messages)
    total += len(user_input.split())
    token_est = int(total / 0.75)

    turn = state.get("turn_count", 0) + 1
    print(f"  [TURN {turn}] Tokens: ~{token_est}, Facts: +{len(new_facts)}")

    return {
        "facts": new_facts,
        "turn_count": turn,
        "token_estimate": token_est,
    }


def should_summarize(state: ConversationState) -> str:
    """Check if summarization is needed before responding."""
    if state.get("token_estimate", 0) > SUMMARIZE_THRESHOLD and len(state.get("messages", [])) > 6:
        print(f"  [DECISION] Token estimate {state['token_estimate']} > {SUMMARIZE_THRESHOLD} → SUMMARIZE")
        return "summarize"
    return "respond"


def summarize_node(state: ConversationState) -> dict:
    """Summarize older messages to free context space."""
    messages = state.get("messages", [])
    old_summary = state.get("summary", "")

    # Keep last 4 messages, summarize the rest
    if len(messages) <= 4:
        return {}

    old_msgs = messages[:-4]
    old_text = ""
    if old_summary:
        old_text = f"Previous summary: {old_summary}\n"
    old_text += "\n".join(
        str(m.content if hasattr(m, 'content') else m)[:100]
        for m in old_msgs
    )

    try:
        response = summarizer.invoke([
            SystemMessage(content="Summarize this conversation concisely. Preserve key facts, decisions, and preferences. Output ONLY the summary."),
            HumanMessage(content=old_text),
        ])
        new_summary = response.content.strip()
    except Exception as e:
        new_summary = old_summary + " " + " | ".join(
            str(m.content if hasattr(m, 'content') else m)[:30]
            for m in old_msgs
        )

    print(f"  [SUMMARIZE] {len(old_msgs)} old msgs → {len(new_summary.split())} word summary")

    # We can't remove messages from add_messages reducer, but we can
    # update the summary for the next respond_node to use
    return {"summary": new_summary}


def respond_node(state: ConversationState) -> dict:
    """Generate response with full memory context."""
    user_input = state["user_input"]
    facts = state.get("facts", [])
    summary = state.get("summary", "")

    system = "You are a helpful assistant with memory."
    if facts:
        system += "\n\nKnown facts:\n" + "\n".join(f"• {f}" for f in facts[-10:])
    if summary:
        system += f"\n\nConversation summary:\n{summary}"

    prompt = [SystemMessage(content=system)]

    # Add recent messages (last 6)
    for msg in state.get("messages", [])[-6:]:
        prompt.append(msg)

    prompt.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    return {
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ],
    }


# ================================================================
# GRAPH
# ================================================================

def build_conversation_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("extract_and_count", extract_and_count_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("extract_and_count")
    graph.add_conditional_edges("extract_and_count", should_summarize, {
        "summarize": "summarize",
        "respond": "respond",
    })
    graph.add_edge("summarize", "respond")
    graph.add_edge("respond", END)

    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    print(f"  [DB] SQLite persistence: {DB_PATH}")
    return graph.compile(checkpointer=checkpointer)


# ================================================================
# DEMO
# ================================================================

def run_demo():
    app = build_conversation_graph()
    thread_id = "conversation-memory-session-001"
    config = {"configurable": {"thread_id": thread_id}}

    turns = [
        "Hi! I'm looking for restaurant recommendations in Tokyo.",
        "I'm vegetarian and allergic to soy. Budget is moderate.",
        "What about in the Shinjuku area specifically?",
        "Any good ramen shops with vegetarian options there?",
        "How about dessert places nearby?",
        "Can you summarize what we've discussed so far?",
        "One more thing — do you remember my dietary restrictions?",
    ]

    print("\n" + "=" * 65)
    print("  END-TO-END CONVERSATION WITH MEMORY")
    print("=" * 65)

    for i, user_input in enumerate(turns):
        print(f"\n{'━' * 65}")
        print(f"  User: {user_input}")
        print(f"{'━' * 65}")

        result = app.invoke(
            {"user_input": user_input, "messages": [], "facts": [],
             "summary": "", "turn_count": 0, "token_estimate": 0},
            config,
        )

        messages = result.get("messages", [])
        last_ai = [m for m in messages if isinstance(m, AIMessage)]
        if last_ai:
            print(f"  Agent: {last_ai[-1].content[:200]}")

    # Final state report
    print(f"\n{'=' * 65}")
    print(f"  MEMORY STATE REPORT")
    print(f"{'=' * 65}")
    print(f"  Turns: {result.get('turn_count', 0)}")
    print(f"  Facts: {result.get('facts', [])}")
    if result.get('summary'):
        print(f"  Summary: {result['summary'][:200]}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 16: Conversation Memory (LangGraph + SQLite)")
    print("=" * 65)

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Production conversation agents combine multiple memory techniques:
       message buffer + fact extraction + auto-summarization + checkpointing.

    2. Fact extraction runs on EVERY turn to catch user preferences and
       constraints.  These persist even when messages are summarized.

    3. Auto-summarization triggers based on token estimates to keep
       context within budget while preserving key information.

    4. SqliteSaver with thread_id enables session continuity -- the agent
       picks up where it left off even after process restart.

    5. The agent should be able to answer "do you remember?" questions
       by referencing facts and summaries — test this explicitly.
    """))
