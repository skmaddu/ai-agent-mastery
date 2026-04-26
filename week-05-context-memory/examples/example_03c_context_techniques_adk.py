import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3c: Context Techniques — ADK Agent with Summarization Tool
===================================================================
Google ADK implementation of context-aware conversation management.

The ADK agent has a `summarize_history` tool that it can invoke when
the conversation grows long.  Unlike the LangGraph version (which uses
graph edges to automate summarization), here the AGENT DECIDES when
to summarize — a more agentic approach.

Phoenix tracing: YES — observe the agent's tool-calling behavior.

Run: python week-05-context-memory/examples/example_03c_context_techniques_adk.py
"""

import os
import sys
import json
import asyncio
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# ── Phoenix Tracing Setup ──────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        print("[Phoenix] Not available.")
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week5-context-techniques-adk")
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception as e:
        print(f"[Phoenix] Setup failed: {e}")
        return None

# ── ADK Imports ────────────────────────────────────────────────
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("[ADK] Google ADK not installed. Install with: pip install google-adk")


# ================================================================
# CONVERSATION MEMORY STORE
# ================================================================
# We maintain a simple in-memory history that the agent's tool can
# access.  In production, this would be a database or session store.

conversation_history: list = []
conversation_summary: str = ""


def get_conversation_stats() -> str:
    """
    Get statistics about the current conversation.

    The agent calls this to decide whether summarization is needed.
    This is the ADK equivalent of the token-counting node in LangGraph.

    Returns:
        JSON string with message count, approximate token count, and
        whether summarization is recommended.
    """
    global conversation_history
    total_tokens = sum(
        max(1, int(len(msg["content"].split()) / 0.75))
        for msg in conversation_history
    )
    needs_summary = total_tokens > 400 and len(conversation_history) > 6

    stats = {
        "message_count": len(conversation_history),
        "approximate_tokens": total_tokens,
        "has_summary": bool(conversation_summary),
        "summary_length_tokens": max(1, int(len(conversation_summary.split()) / 0.75)) if conversation_summary else 0,
        "needs_summarization": needs_summary,
    }
    print(f"  [STATS] Messages: {stats['message_count']}, "
          f"Tokens: ~{stats['approximate_tokens']}, "
          f"Needs summary: {stats['needs_summarization']}")
    return json.dumps(stats)


def summarize_old_messages(keep_recent: int = 4) -> str:
    """
    Summarize old conversation messages to save context space.

    Keeps the most recent `keep_recent` message pairs in full text
    and creates a concise summary of everything older.  This mirrors
    the hierarchical windowing pattern.

    Args:
        keep_recent: Number of recent message pairs to keep in full.

    Returns:
        Confirmation message with the new summary and token savings.
    """
    global conversation_history, conversation_summary

    if len(conversation_history) <= keep_recent * 2:
        return "Not enough messages to summarize. Keep chatting!"

    # Split into old and recent
    cutoff = len(conversation_history) - (keep_recent * 2)
    old_messages = conversation_history[:cutoff]
    recent_messages = conversation_history[cutoff:]

    # Build summary (in production, use an LLM for this)
    # Here we simulate with extractive summarization
    old_topics = []
    for msg in old_messages:
        if msg["role"] == "user":
            # Extract first 30 words as topic indicator
            words = msg["content"].split()[:30]
            old_topics.append(" ".join(words))

    new_summary_parts = []
    if conversation_summary:
        new_summary_parts.append(f"Earlier: {conversation_summary}")
    new_summary_parts.append(f"Recent topics: {' | '.join(old_topics)}")
    conversation_summary = " ".join(new_summary_parts)

    # Update history to only keep recent messages
    old_count = len(conversation_history)
    conversation_history = recent_messages

    old_tokens = sum(int(len(m["content"].split()) / 0.75) for m in old_messages)
    summary_tokens = max(1, int(len(conversation_summary.split()) / 0.75))

    result = (
        f"Summarized {len(old_messages)} old messages into {summary_tokens} tokens. "
        f"Saved ~{old_tokens - summary_tokens} tokens. "
        f"Keeping {len(recent_messages)} recent messages in full."
    )
    print(f"  [SUMMARIZE] {result}")
    return result


def add_to_history(role: str, content: str) -> str:
    """
    Add a message to the conversation history.

    Args:
        role: Either 'user' or 'assistant'.
        content: The message content.

    Returns:
        Confirmation that the message was stored.
    """
    global conversation_history
    conversation_history.append({"role": role, "content": content})
    return f"Added {role} message to history (total: {len(conversation_history)} messages)"


# ================================================================
# ADK AGENT WITH CONTEXT MANAGEMENT
# ================================================================

def build_context_agent():
    """
    Build an ADK agent that manages its own conversation context.

    KEY DIFFERENCE FROM LANGGRAPH: In the LangGraph version, the graph
    structure FORCES summarization when tokens exceed a threshold.
    Here, the agent CHOOSES when to summarize by calling tools.  This
    is more flexible but less predictable.
    """
    agent = LlmAgent(
        name="context_manager",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-latest"),
        instruction="""You are a helpful travel planning assistant with context management abilities.

IMPORTANT RULES:
1. At the START of each response, call get_conversation_stats to check context health.
2. If needs_summarization is true, call summarize_old_messages BEFORE responding.
3. After you respond to the user, call add_to_history to record both the user's message and your response.
4. When a conversation summary exists, use it as context for your responses.

Current conversation summary (if any): {conversation_summary}

You help users plan trips, suggest destinations, and provide travel advice.
Always be helpful, concise, and reference earlier conversation context when relevant.""".format(
            conversation_summary=conversation_summary or "None yet."
        ),
        tools=[
            get_conversation_stats,
            summarize_old_messages,
            add_to_history,
        ],
        description="Travel assistant with automatic context management.",
    )
    return agent


# ================================================================
# ASYNC RUNNER
# ================================================================

async def run_agent_turn(agent: LlmAgent, message: str, turn: int,
                         retries: int = 5) -> str:
    """
    Run one conversation turn with the ADK agent.

    Each turn creates a new session (stateless from ADK's perspective).
    The context management is handled by our external history store,
    which the agent accesses through tools.

    Includes retry logic for transient Gemini API errors (503, 500).
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="week5_context_demo",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="week5_context_demo",
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

    return "[Error: unexpected]"


async def run_conversation():
    """Run a multi-turn conversation demonstrating context management."""

    if not ADK_AVAILABLE:
        print("[SKIP] ADK not available.")
        return

    agent = build_context_agent()

    user_turns = [
        "Hi! I'm planning a trip to Japan next spring.",
        "Tell me about Kyoto's temples.",
        "What food should I try in Osaka?",
        "How does the Japan Rail Pass work?",
        "Budget tips for a 2-week trip?",
        "Day trips from Tokyo?",
        "Weather in April?",
        "Tips for Tokyo subway?",
    ]

    print("\n" + "=" * 65)
    print("  ADK MULTI-TURN CONVERSATION WITH CONTEXT MANAGEMENT")
    print("=" * 65)

    for i, user_input in enumerate(user_turns):
        print(f"\n{'━' * 65}")
        print(f"  Turn {i + 1}/{len(user_turns)}")
        print(f"  User: {user_input}")
        print(f"{'━' * 65}")

        response = await run_agent_turn(agent, user_input, i + 1)
        print(f"\n  Assistant: {response[:200]}")

    # Final report
    print(f"\n{'=' * 65}")
    print(f"  FINAL STATE")
    print(f"{'=' * 65}")
    print(f"  Messages in history: {len(conversation_history)}")
    print(f"  Summary exists: {bool(conversation_summary)}")
    if conversation_summary:
        print(f"  Summary: {conversation_summary[:200]}...")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 3c: Context Techniques (Google ADK)        ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()

    if ADK_AVAILABLE:
        asyncio.run(run_conversation())
    else:
        print("\n  [SKIP] Install Google ADK to run this example:")
        print("  pip install google-adk")

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. ADK agents can manage their own context via TOOLS — the agent
       decides when to summarize, not the graph structure.

    2. This is more flexible than LangGraph's edge-based approach but
       less deterministic: the agent might forget to summarize.

    3. The external history store pattern decouples memory from the
       session — useful when sessions are short-lived or stateless.

    4. In production, combine BOTH approaches: use tools for agent-
       initiated summarization AND a hard limit that forces eviction
       if the agent neglects to manage context.

    5. LangGraph vs ADK for context management:
       - LangGraph: Deterministic, graph-enforced policies
       - ADK: Agent-driven, more flexible, requires good instructions
    """))
