import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3b: Context Techniques — LangGraph Auto-Summarization
==============================================================
LangGraph implementation of automatic context summarization.

When the conversation exceeds a token threshold, this graph:
  1. Counts tokens in the current context
  2. Decides whether summarization is needed
  3. Summarizes old messages into a compact summary
  4. Responds using the summarized + recent context

This is a PRODUCTION PATTERN: without auto-summarization, long
conversations either crash (exceed window) or silently degrade
(the model ignores early messages).

Phoenix tracing: YES — observe summarization triggers in the dashboard.

Run: python week-05-context-memory/examples/example_03b_context_techniques_langgraph.py
"""

import os
import sys
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

# ── Phoenix Tracing Setup ──────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    """Launch Phoenix dashboard and instrument LangChain."""
    if not PHOENIX_AVAILABLE:
        print("[Phoenix] Not available — install phoenix and openinference.")
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week5-context-techniques")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception as e:
        print(f"[Phoenix] Setup failed: {e}")
        return None


# ── LLM Setup ──────────────────────────────────────────────────

def get_llm(temperature=0.7):
    """Create LLM based on provider setting (Groq default)."""
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
# STATE DEFINITION
# ================================================================
# The state tracks:
#   - messages:       Full conversation history (grows unbounded)
#   - summary:        Running summary of old messages (replaces them)
#   - token_count:    Approximate token count of the current context
#   - summary_count:  How many times we've summarized (for tracking)
#
# KEY DESIGN CHOICE: We use a SIMPLE message list (not Annotated
# with add_messages reducer) because we need fine-grained control
# over which messages to keep/evict.  The add_messages reducer
# just appends — we need to REPLACE old messages with a summary.

class ContextState(TypedDict):
    messages: List[dict]       # {"role": str, "content": str}
    summary: str               # Running summary of evicted messages
    token_count: int           # Approximate current context tokens
    summary_count: int         # Number of summarizations performed
    user_input: str            # Current user input to process
    response: str              # Agent's response


# ================================================================
# TOKEN COUNTING
# ================================================================
# In production, use tiktoken for exact counts.  Here we use a
# simple word-based approximation (1 token ≈ 0.75 words).

TOKEN_THRESHOLD = 500   # Trigger summarization above this
KEEP_RECENT = 4         # Keep the N most recent message pairs


def estimate_tokens(text: str) -> int:
    """Approximate token count from text."""
    return max(1, int(len(text.split()) / 0.75))


def count_context_tokens(state: ContextState) -> int:
    """Count total tokens across summary + messages + system prompt."""
    total = estimate_tokens(state.get("summary", ""))
    for msg in state.get("messages", []):
        total += estimate_tokens(msg.get("content", ""))
    total += 100  # System prompt overhead estimate
    return total


# ================================================================
# GRAPH NODES
# ================================================================

# LLM instances: one for summarization (deterministic), one for chat
summarizer_llm = get_llm(temperature=0)
chat_llm = get_llm(temperature=0.7)


def count_tokens_node(state: ContextState) -> dict:
    """
    Node 1: Count the current context size.

    This runs BEFORE every response to check if we're approaching
    the token budget.  In production, you'd also factor in the
    expected output tokens (leave room for the response).
    """
    token_count = count_context_tokens(state)
    print(f"\n  [TOKEN COUNT] Current context: ~{token_count} tokens "
          f"(threshold: {TOKEN_THRESHOLD})")
    return {"token_count": token_count}


def should_summarize(state: ContextState) -> str:
    """
    Conditional edge: decide whether to summarize or respond directly.

    We summarize when the context exceeds the threshold AND there are
    enough messages to make summarization worthwhile.
    """
    if (state["token_count"] > TOKEN_THRESHOLD
            and len(state["messages"]) > KEEP_RECENT * 2):
        print(f"  [DECISION] Token count {state['token_count']} > {TOKEN_THRESHOLD} "
              f"→ SUMMARIZE old messages")
        return "summarize"
    else:
        print(f"  [DECISION] Token count within budget → RESPOND directly")
        return "respond"


def summarize_node(state: ContextState) -> dict:
    """
    Node 2: Summarize old messages to free up context space.

    Strategy:
      1. Keep the KEEP_RECENT most recent message pairs (full text)
      2. Summarize everything older into a running summary
      3. Merge with any existing summary

    This implements the "hierarchical windowing" pattern from Example 3:
    recent messages get full fidelity, older ones get compressed.
    """
    messages = state["messages"]
    old_summary = state.get("summary", "")

    # Split: old messages (to summarize) vs recent (to keep)
    cutoff = len(messages) - (KEEP_RECENT * 2)
    old_messages = messages[:cutoff]
    recent_messages = messages[cutoff:]

    print(f"  [SUMMARIZE] Condensing {len(old_messages)} old messages...")
    print(f"  [SUMMARIZE] Keeping {len(recent_messages)} recent messages in full")

    # Build the text to summarize
    old_text = ""
    if old_summary:
        old_text += f"Previous summary: {old_summary}\n\n"
    old_text += "Messages to summarize:\n"
    for msg in old_messages:
        old_text += f"  {msg['role']}: {msg['content']}\n"

    # Use LLM to create a concise summary
    summary_prompt = [
        SystemMessage(content=(
            "You are a conversation summarizer. Create a concise summary of the "
            "conversation below. Preserve: key topics discussed, user preferences "
            "mentioned, any decisions made, and important facts. Be brief but complete. "
            "Output ONLY the summary, no preamble."
        )),
        HumanMessage(content=old_text),
    ]

    try:
        response = summarizer_llm.invoke(summary_prompt)
        new_summary = response.content.strip()
    except Exception as e:
        # Fallback: just concatenate first lines of old messages
        print(f"  [SUMMARIZE] LLM failed ({e}), using fallback")
        new_summary = old_summary + " " + " | ".join(
            m["content"][:50] for m in old_messages
        )

    new_token_count = count_context_tokens({
        "messages": recent_messages,
        "summary": new_summary,
    })

    print(f"  [SUMMARIZE] New summary: {estimate_tokens(new_summary)} tokens")
    print(f"  [SUMMARIZE] Total context now: ~{new_token_count} tokens")

    return {
        "messages": recent_messages,
        "summary": new_summary,
        "token_count": new_token_count,
        "summary_count": state.get("summary_count", 0) + 1,
    }


def respond_node(state: ContextState) -> dict:
    """
    Node 3: Generate a response using the current context.

    The context is assembled from:
      1. System prompt (static instructions)
      2. Summary of old conversation (if any)
      3. Recent messages (full text)
      4. Current user input

    This is the RECOMBINATION step — we stitch together the different
    context layers into a coherent prompt.
    """
    user_input = state["user_input"]
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    # Build the full prompt with context layers
    system_content = (
        "You are a helpful assistant. Answer the user's question concisely. "
        "Use the conversation context below to maintain continuity."
    )

    if summary:
        system_content += f"\n\nConversation summary (older context):\n{summary}"

    prompt_messages = [SystemMessage(content=system_content)]

    # Add recent conversation history
    for msg in messages:
        if msg["role"] == "user":
            prompt_messages.append(HumanMessage(content=msg["content"]))
        else:
            prompt_messages.append(AIMessage(content=msg["content"]))

    # Add current user input
    prompt_messages.append(HumanMessage(content=user_input))

    print(f"\n  [RESPOND] Generating response...")
    print(f"    Context: {len(prompt_messages)} messages "
          f"({'with' if summary else 'no'} summary)")

    try:
        response = chat_llm.invoke(prompt_messages)
        response_text = response.content.strip()
    except Exception as e:
        response_text = f"[Error generating response: {e}]"

    # Update messages with the new exchange
    updated_messages = list(messages) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response_text},
    ]

    print(f"  [RESPOND] Response: {response_text[:100]}...")

    return {
        "messages": updated_messages,
        "response": response_text,
    }


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================
#
# Flow:
#   START → count_tokens → [should_summarize?]
#                              ├─ "summarize" → summarize → respond → END
#                              └─ "respond"   → respond → END

def build_graph():
    """
    Build the auto-summarization conversation graph.

    This graph is invoked ONCE PER USER TURN.  Between turns, the
    state persists (in production, via checkpointing).
    """
    graph = StateGraph(ContextState)

    graph.add_node("count_tokens", count_tokens_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("count_tokens")

    # Conditional: summarize if over budget, else respond directly
    graph.add_conditional_edges(
        "count_tokens",
        should_summarize,
        {
            "summarize": "summarize",
            "respond": "respond",
        }
    )

    graph.add_edge("summarize", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# ================================================================
# MULTI-TURN CONVERSATION DEMO
# ================================================================

def run_conversation():
    """
    Simulate a multi-turn conversation that triggers summarization.

    We send enough messages to exceed the token threshold, then
    observe how the graph automatically summarizes old messages
    while preserving recent context.
    """
    app = build_graph()

    # Conversation turns — enough to trigger at least one summarization
    user_turns = [
        "Hi! I'm planning a trip to Japan next spring. What cities should I visit?",
        "Tell me more about Kyoto. I love temples and traditional culture.",
        "What about food in Osaka? I've heard it's the food capital.",
        "How should I get around? Is the Japan Rail Pass worth it?",
        "I have a budget of about $3000 for 2 weeks. Is that realistic?",
        "Can you also suggest some day trips from Tokyo?",
        "What's the weather like in April? Should I pack for rain?",
        "Last question — any tips for navigating Tokyo's subway system?",
    ]

    # Persistent state across turns
    state = {
        "messages": [],
        "summary": "",
        "token_count": 0,
        "summary_count": 0,
        "user_input": "",
        "response": "",
    }

    print("\n" + "=" * 65)
    print("  MULTI-TURN CONVERSATION WITH AUTO-SUMMARIZATION")
    print("=" * 65)

    for i, user_input in enumerate(user_turns):
        print(f"\n{'━' * 65}")
        print(f"  Turn {i + 1}/{len(user_turns)}")
        print(f"  User: {user_input}")
        print(f"{'━' * 65}")

        state["user_input"] = user_input

        # Invoke the graph for this turn
        result = app.invoke(state, {"run_name": f"turn-{i + 1}"})

        # Carry state forward to next turn
        state["messages"] = result["messages"]
        state["summary"] = result.get("summary", state["summary"])
        state["summary_count"] = result.get("summary_count", state["summary_count"])
        state["response"] = result["response"]

        print(f"\n  Assistant: {result['response'][:200]}")

    # Final report
    print(f"\n{'=' * 65}")
    print(f"  CONVERSATION SUMMARY REPORT")
    print(f"{'=' * 65}")
    print(f"  Total turns: {len(user_turns)}")
    print(f"  Summarizations triggered: {state['summary_count']}")
    print(f"  Messages in current window: {len(state['messages'])}")
    print(f"  Running summary length: {len(state.get('summary', '').split())} words")
    if state.get("summary"):
        print(f"\n  Running summary:\n    {state['summary'][:300]}...")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 3b: Auto-Summarization (LangGraph)         ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()
    run_conversation()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Auto-summarization keeps context within budget WITHOUT losing
       information.  Old messages become a concise summary; recent ones
       stay in full.

    2. The summarization threshold should be set BELOW the model's max
       context — leave room for the system prompt, tool results, and
       the model's response.

    3. Use a deterministic (temperature=0) LLM for summarization to
       ensure consistent, factual summaries.

    4. In production, use tiktoken for exact token counting instead of
       the word-based approximation shown here.

    5. Phoenix tracing lets you see WHEN summarization triggers and
       HOW MUCH context was saved — essential for tuning thresholds.
    """))
