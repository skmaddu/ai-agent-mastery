import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 12: LangChain Memory Modules — Overview & Migration Guide
====================================================================
Demonstrates legacy LangChain memory modules and explains why modern
agents should use native LangGraph state instead.

Covers:
  1. Legacy LangChain Memory Modules Overview
  2. Pros, Cons & Why to Move to Native Framework State

IMPORTANT CONTEXT: LangChain's memory classes (ConversationBufferMemory,
ConversationSummaryMemory, etc.) were the standard in 2023-2024.  In
2025-2026, LangGraph's native state management + checkpointing has
largely replaced them.  This example teaches both so you understand
legacy code AND the modern approach.

Run: python week-05-context-memory/examples/example_12_langchain_memory_modules.py
"""

import os
import sys
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

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
        tracer_provider = register(project_name="week5-memory-modules")
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
# 1. LEGACY LANGCHAIN MEMORY MODULES OVERVIEW
# ================================================================
# LangChain provided several memory classes, each with a different
# strategy for managing conversation history:
#
#   ConversationBufferMemory:
#     - Stores ALL messages verbatim
#     - Simple but unbounded growth
#     - Will eventually exceed context window
#
#   ConversationBufferWindowMemory:
#     - Stores only the last K message pairs
#     - Bounded but loses old context completely
#
#   ConversationSummaryMemory:
#     - Summarizes old messages using an LLM
#     - Bounded growth, preserves key info
#     - Extra LLM call per turn for summarization
#
#   ConversationSummaryBufferMemory:
#     - Hybrid: keeps recent messages verbatim, summarizes older ones
#     - Best of both worlds
#     - Most complex to configure
#
#   VectorStoreRetrieverMemory:
#     - Stores messages in a vector store
#     - Retrieves relevant past messages by similarity
#     - Good for long conversations where topic revisiting is common

def demo_legacy_modules():
    """Show each legacy memory module's behavior."""

    print("=" * 65)
    print("  LEGACY LANGCHAIN MEMORY MODULES OVERVIEW")
    print("=" * 65)

    # Simulate conversation to show each module's behavior
    conversation = [
        ("user", "Hi, I'm planning a trip to Japan next spring."),
        ("ai", "Japan in spring is wonderful! Cherry blossom season..."),
        ("user", "My budget is $3000 for 2 weeks."),
        ("ai", "That's a good budget. I'd suggest..."),
        ("user", "I'm vegetarian and allergic to nuts."),
        ("ai", "Important to note! In Japan, many dishes contain..."),
        ("user", "What about the Japan Rail Pass?"),
        ("ai", "The JR Pass is excellent for tourists..."),
        ("user", "Can you recommend hotels in Kyoto?"),
        ("ai", "Here are some great options in Kyoto..."),
    ]

    # --- ConversationBufferMemory (simulated) ---
    print("\n  ── ConversationBufferMemory ──")
    print("  Stores: ALL messages verbatim")
    buffer = []
    total_tokens = 0
    for role, msg in conversation:
        buffer.append(f"{role}: {msg}")
        total_tokens += len(msg.split())

    print(f"  After {len(conversation)} messages: {total_tokens} words (~{int(total_tokens / 0.75)} tokens)")
    print(f"  Growth: UNBOUNDED — will eventually exceed context window")
    print(f"  Memory: {buffer[0][:50]}...")
    print(f"          ... ({len(buffer)} total messages)")

    # --- ConversationBufferWindowMemory (simulated, k=3) ---
    print("\n  ── ConversationBufferWindowMemory (k=3 pairs) ──")
    print("  Stores: Only the last K message pairs")
    window = buffer[-6:]  # Last 3 pairs = 6 messages
    print(f"  After {len(conversation)} messages: keeping {len(window)} messages")
    print(f"  LOST: Budget info, dietary restrictions (from early turns)!")
    print(f"  Kept: {window[0][:50]}...")

    # --- ConversationSummaryMemory (simulated) ---
    print("\n  ── ConversationSummaryMemory ──")
    print("  Stores: LLM-generated summary of all messages")
    summary = ("The user is planning a 2-week Japan trip in spring with "
               "$3000 budget. They are vegetarian with a nut allergy. "
               "Discussed cherry blossoms, JR Pass, and Kyoto hotels.")
    print(f"  Summary ({len(summary.split())} words): {summary}")
    print(f"  Compression: {total_tokens} → {len(summary.split())} words "
          f"({len(summary.split()) / total_tokens:.0%})")
    print(f"  Trade-off: Extra LLM call per turn for summarization")

    # --- ConversationSummaryBufferMemory (simulated) ---
    print("\n  ── ConversationSummaryBufferMemory (best hybrid) ──")
    print("  Stores: Summary of old messages + recent messages verbatim")
    old_summary = "User planning 2-week Japan trip, $3000 budget, vegetarian, nut allergy."
    recent = buffer[-4:]  # Last 2 pairs
    print(f"  Summary: {old_summary}")
    print(f"  Recent: {recent[0][:50]}...")
    print(f"          {recent[1][:50]}...")
    print(f"  Total: ~{len(old_summary.split()) + sum(len(m.split()) for m in recent)} words")

    # Comparison table
    print(f"\n  {'─' * 55}")
    print(f"  COMPARISON:")
    print(f"  {'─' * 55}")
    print(f"  {'Module':<30} {'Growth':^10} {'Info Loss':^12} {'Cost':^8}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 12} {'─' * 8}")
    print(f"  {'ConversationBuffer':<30} {'O(n)':^10} {'None':^12} {'Free':^8}")
    print(f"  {'ConversationBufferWindow':<30} {'O(1)':^10} {'High':^12} {'Free':^8}")
    print(f"  {'ConversationSummary':<30} {'O(1)':^10} {'Low':^12} {'$$':^8}")
    print(f"  {'ConversationSummaryBuffer':<30} {'O(1)':^10} {'Very Low':^12} {'$':^8}")
    print(f"  {'VectorStoreRetriever':<30} {'O(n)*':^10} {'Selective':^12} {'$':^8}")
    print(f"  * Vector store scales well but requires separate infra")


# ================================================================
# 2. PROS, CONS & WHY TO MOVE TO NATIVE FRAMEWORK STATE
# ================================================================

def demo_migration_to_langgraph():
    """Show why LangGraph state is preferred over legacy memory."""

    print("\n" + "=" * 65)
    print("  WHY MOVE TO NATIVE LANGGRAPH STATE")
    print("=" * 65)

    print("""
  PROBLEMS WITH LEGACY MEMORY MODULES:

  1. IMPLICIT STATE: Memory is hidden inside the chain object.
     You can't inspect, modify, or checkpoint it easily.

  2. SINGLE STRATEGY: Each module uses ONE memory strategy.
     Real agents need different strategies for different data.

  3. NO CHECKPOINTING: Can't save/restore conversation state.
     If the process crashes, all memory is lost.

  4. POOR TESTABILITY: Hard to test because memory is coupled
     to the chain.  Can't inject test state easily.

  5. FRAMEWORK LOCK-IN: Memory modules are LangChain-specific.
     Migrating to another framework means rewriting memory logic.

  LANGGRAPH STATE ADVANTAGES:

  1. EXPLICIT STATE: Memory is a TypedDict — visible, inspectable,
     and type-checked.  You SEE everything in the state.

  2. COMPOSABLE: Different fields can use different strategies
     (e.g., messages as buffer, facts as semantic store).

  3. CHECKPOINTING: Built-in MemorySaver persists state across
     turns and sessions.  Supports time-travel debugging.

  4. TESTABLE: Just pass a state dict to test any node.
     No hidden state to mock or configure.

  5. FRAMEWORK-AGNOSTIC: State is just a dict — works with any
     LLM provider, any storage backend.""")

    # Migration example
    print(f"""
  ── MIGRATION EXAMPLE ──

  BEFORE (Legacy LangChain):
  ┌─────────────────────────────────────────────────┐
  │ from langchain.memory import                    ���
  │     ConversationSummaryBufferMemory             │
  │                                                 │
  │ memory = ConversationSummaryBufferMemory(        │
  │     llm=llm, max_token_limit=1000)             │
  │ chain = ConversationChain(llm=llm, memory=mem) │
  │ response = chain.predict(input="Hello")         │
  │ # Memory is HIDDEN inside the chain object     │
  └─────────────────────────────────────────────────┘

  AFTER (LangGraph State):
  ┌─────────────────────────────────────────────────┐
  │ class ChatState(TypedDict):                     │
  │     messages: Annotated[list, add_messages]     │
  │     summary: str  # Explicit summary field      │
  │     facts: List[str]  # Explicit fact storage   │
  │                                                 │
  │ graph = StateGraph(ChatState)                   │
  │ graph.add_node("chat", chat_node)               │
  │ graph.add_node("summarize", summarize_node)     │
  │ app = graph.compile(checkpointer=MemorySaver()) │
  │ # State is EXPLICIT, checkpointed, inspectable │
  └─────────────────────────────────────────────────┘

  The LangGraph version is more verbose but MUCH more controllable.
  You decide exactly when to summarize, what to store, and how to
  persist — instead of relying on magic inside a memory class.
    """)


# ================================================================
# BONUS: Quick Reference — When to Use What
# ================================================================

def demo_decision_guide():
    print(f"\n  {'─' * 55}")
    print(f"  DECISION GUIDE: Legacy Module → Modern Replacement")
    print(f"  {'─' * 55}")
    print(f"  {'Legacy Module':<30} {'Modern Replacement':<30}")
    print(f"  {'─' * 30} {'─' * 30}")
    print(f"  {'ConversationBuffer':<30} {'LangGraph messages + add_messages':<30}")
    print(f"  {'ConversationBufferWindow':<30} {'Trim messages in graph node':<30}")
    print(f"  {'ConversationSummary':<30} {'Summarize node + summary field':<30}")
    print(f"  {'ConversationSummaryBuffer':<30} {'Example 3b auto-summarization':<30}")
    print(f"  {'VectorStoreRetriever':<30} {'RAG pipeline (Examples 5-9)':<30}")
    print(f"\n  Rule: Use LangGraph state for NEW projects.")
    print(f"  Only use legacy modules when maintaining existing code.")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 12: LangChain Memory Modules               ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()
    demo_legacy_modules()
    demo_migration_to_langgraph()
    demo_decision_guide()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. LangChain memory modules (Buffer, Window, Summary) were useful
       but have been superseded by LangGraph's explicit state management.

    2. ConversationSummaryBufferMemory was the best legacy option —
       it combines recent verbatim messages with summarized history.

    3. LangGraph state is preferred because it's explicit, checkpointed,
       composable, and testable.  No hidden state, no magic.

    4. Migration path: replace memory classes with TypedDict state fields
       and graph nodes that implement the same strategies.

    5. Use legacy modules only for maintaining existing code.  All new
       agents should use LangGraph state (or ADK sessions).
    """))
