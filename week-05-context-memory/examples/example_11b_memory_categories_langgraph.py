import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 11b: Memory Categories in LangGraph -- Episodic, Semantic, Procedural
==============================================================================
LangGraph implementation showing how to implement ALL THREE memory categories
from cognitive science (Example 11) as explicit state fields with reducers.

Covers:
  1. Episodic Memory  -> messages + summary fields (what happened)
  2. Semantic Memory   -> facts field with dedup reducer (what is true)
  3. Procedural Memory -> procedures field (how to do things)
  4. Memory-Aware Generation (uses all three in the prompt)
  5. Cross-Turn Persistence with MemorySaver checkpointing

Architecture:
  Each turn flows through:
    START -> extract_facts -> extract_procedures -> maybe_summarize -> respond -> END

  State fields and their reducers:
    messages:    Annotated[list, add_messages]     -- APPEND (episodic)
    summary:     str                                -- REPLACE (compressed episodic)
    facts:       Annotated[List[str], merge_facts]  -- APPEND + DEDUP (semantic)
    procedures:  Annotated[List[Dict], merge_procs] -- APPEND + DEDUP (procedural)
    preferences: Dict[str, str]                     -- REPLACE (semantic shortcuts)

Phoenix tracing: YES

Run: python week-05-context-memory/examples/example_11b_memory_categories_langgraph.py
"""

import os
import sys
import uuid
import textwrap
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

# -- Phoenix ----------------------------------------------------
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
        tracer_provider = register(project_name="week5-memory-categories-langgraph")
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
# CUSTOM REDUCERS
# ================================================================
# Reducers control HOW state updates merge with existing state.
# Without a reducer, new values REPLACE old values.
# With a reducer, you can APPEND, DEDUPLICATE, or apply custom logic.

def merge_facts(existing: List[str], new: List[str]) -> List[str]:
    """
    Reducer for semantic memory (facts).

    Appends new facts while deduplicating.  Facts are timeless --
    once learned, they persist until explicitly corrected.

    Example:
      existing: ["User is vegetarian", "Budget is $3000"]
      new:      ["User is vegetarian", "User allergic to nuts"]
      result:   ["User is vegetarian", "Budget is $3000", "User allergic to nuts"]
    """
    combined = list(existing)
    for fact in new:
        # Case-insensitive dedup
        if not any(fact.lower() == f.lower() for f in combined):
            combined.append(fact)
    return combined


def merge_procedures(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """
    Reducer for procedural memory (skills/workflows).

    Appends new procedures, updates existing ones by name.
    Procedures are learned from successful interactions and
    refined over time.

    Example:
      existing: [{"name": "book_flight", "steps": [...], "success_count": 2}]
      new:      [{"name": "book_flight", "steps": [...], "success_count": 3}]
      result:   [{"name": "book_flight", "steps": [...], "success_count": 3}]  (updated)
    """
    combined = {p["name"]: p for p in existing}
    for proc in new:
        name = proc["name"]
        if name in combined:
            # Update existing: merge success counts, update steps if provided
            old = combined[name]
            combined[name] = {
                **old,
                **proc,
                "success_count": old.get("success_count", 0) + proc.get("success_count", 0),
                "failure_count": old.get("failure_count", 0) + proc.get("failure_count", 0),
            }
        else:
            combined[name] = proc
    return list(combined.values())


# ================================================================
# STATE -- All Three Memory Categories as TypedDict Fields
# ================================================================

class ThreeMemoryState(TypedDict):
    # -- Current input -----------------------------------------
    user_input: str                     # Current user message

    # -- EPISODIC MEMORY (what happened) -----------------------
    # Stores the full conversation history with timestamps.
    # add_messages reducer APPENDS new messages to the list.
    # This is the "raw" episodic memory -- every event recorded.
    messages: Annotated[list, add_messages]

    # Compressed episodic memory: LLM-generated summary of older
    # messages.  Replaces on each update (no reducer = REPLACE).
    # This is how we handle context window limits -- summarize old
    # episodes instead of dropping them completely.
    summary: str

    # -- SEMANTIC MEMORY (what is true) ------------------------
    # Timeless facts about the user, learned across turns.
    # merge_facts reducer APPENDS + DEDUPLICATES.
    # Facts persist even when old messages get summarized away.
    facts: Annotated[List[str], merge_facts]

    # Quick-access semantic shortcuts (latest value wins).
    # These are frequently-accessed facts stored as key-value pairs.
    preferences: Dict[str, str]

    # -- PROCEDURAL MEMORY (how to do things) ------------------
    # Learned workflows and skills, refined by experience.
    # merge_procedures reducer APPENDS new, UPDATES existing.
    procedures: Annotated[List[Dict], merge_procedures]


# ================================================================
# GRAPH NODES
# ================================================================

llm = get_llm(temperature=0.3)


def extract_facts_node(state: ThreeMemoryState) -> dict:
    """
    SEMANTIC MEMORY: Extract facts from the user's message.

    This node runs on every turn and pulls out persistent facts:
    preferences, constraints, personal info -- anything that's TRUE
    about the user regardless of conversation context.

    In production, use an LLM for extraction:
        facts = llm.invoke("Extract facts from: {message}")

    Here we use keyword heuristics + an LLM call for richer extraction.
    """
    user_input = state["user_input"]
    existing_facts = state.get("facts", [])

    # --- Heuristic extraction (fast, catches obvious facts) ---
    new_facts = []
    preferences = {}

    fact_patterns = {
        "vegetarian": "User is vegetarian",
        "vegan": "User is vegan",
        "allergic": f"User has allergy mentioned in: {user_input[:80]}",
        "gluten-free": "User needs gluten-free food",
        "halal": "User requires halal food",
        "kosher": "User requires kosher food",
    }

    for keyword, fact in fact_patterns.items():
        if keyword in user_input.lower():
            new_facts.append(fact)

    # Extract budget as a preference
    if "budget" in user_input.lower():
        # Try to find a number
        import re
        amounts = re.findall(r'\$[\d,]+|\d+\s*(?:dollars|usd|eur)', user_input.lower())
        if amounts:
            preferences["budget"] = amounts[0]
            new_facts.append(f"User's budget is {amounts[0]}")

    # Extract destination preferences
    destinations = ["japan", "tokyo", "kyoto", "osaka", "paris", "london",
                    "rome", "bangkok", "bali", "new york"]
    for dest in destinations:
        if dest in user_input.lower():
            preferences["destination"] = dest.title()
            new_facts.append(f"User is interested in {dest.title()}")

    # --- LLM extraction (catches nuanced facts) ---
    if not new_facts and len(user_input.split()) > 5:
        try:
            extract_prompt = [
                SystemMessage(content=(
                    "Extract factual statements about the user from their message. "
                    "Only extract PERMANENT facts (preferences, constraints, personal info). "
                    "Do NOT extract questions or temporary requests. "
                    "Return each fact on a new line, or 'NONE' if no facts found. "
                    "Keep each fact under 15 words."
                )),
                HumanMessage(content=f"User message: {user_input}"),
            ]
            response = llm.invoke(extract_prompt)
            extracted = response.content.strip()
            if extracted.upper() != "NONE":
                for line in extracted.split("\n"):
                    line = line.strip().strip("- *")
                    if line and len(line) > 3:
                        new_facts.append(line)
        except Exception:
            pass  # Heuristic extraction is sufficient as fallback

    if new_facts:
        print(f"  [SEMANTIC] Extracted facts: {new_facts}")
    if preferences:
        print(f"  [SEMANTIC] Updated preferences: {preferences}")

    result = {"facts": new_facts}
    if preferences:
        # Merge with existing preferences
        merged_prefs = {**state.get("preferences", {}), **preferences}
        result["preferences"] = merged_prefs

    return result


def extract_procedures_node(state: ThreeMemoryState) -> dict:
    """
    PROCEDURAL MEMORY: Learn workflows from successful interactions.

    This node detects when a user is asking for a TASK and checks
    if we have a learned procedure for it.  If a task succeeds,
    the procedure's success count increases.

    Procedural memory is "how to do things" -- the agent gets
    BETTER at repeated tasks because it remembers what worked.
    """
    user_input = state["user_input"].lower()
    new_procedures = []

    # Detect task patterns and record/update procedures
    task_triggers = {
        "book": {
            "name": "booking_workflow",
            "description": "Help user book travel (flights, hotels, activities)",
            "steps": [
                "Confirm destination and dates from semantic memory",
                "Check budget constraints from preferences",
                "Search for options within budget",
                "Present top 3 options with prices",
                "Confirm user choice and proceed",
            ],
            "trigger": "book reserve flight hotel",
        },
        "restaurant": {
            "name": "restaurant_recommendation",
            "description": "Recommend restaurants with dietary awareness",
            "steps": [
                "Check dietary restrictions from semantic memory (facts)",
                "Identify cuisine preferences from past interactions",
                "Search for restaurants matching all constraints",
                "Filter by budget from preferences",
                "Present top 3 with allergen info and prices",
            ],
            "trigger": "restaurant food eat dining meal",
        },
        "itinerary": {
            "name": "itinerary_planning",
            "description": "Plan a day-by-day travel itinerary",
            "steps": [
                "Get trip duration and destination from semantic memory",
                "Identify must-see attractions and user interests",
                "Allocate days to areas/neighborhoods",
                "Add meal recommendations (check dietary facts)",
                "Include transport info and time estimates",
                "Present day-by-day plan with map references",
            ],
            "trigger": "itinerary plan schedule day trip",
        },
        "budget": {
            "name": "budget_calculation",
            "description": "Calculate and break down trip budget",
            "steps": [
                "Get total budget from preferences",
                "Estimate split: 40% accommodation, 25% food, 20% transport, 15% activities",
                "Convert to local currency",
                "Add 10% buffer for unexpected expenses",
                "Check against known costs from semantic memory",
                "Present itemized breakdown",
            ],
            "trigger": "budget cost expense money calculate",
        },
    }

    for keyword, proc_template in task_triggers.items():
        if keyword in user_input:
            # Check if this procedure already exists
            existing = [p for p in state.get("procedures", [])
                        if p["name"] == proc_template["name"]]
            if existing:
                # Record a "use" of this procedure (implicit success)
                new_procedures.append({
                    "name": proc_template["name"],
                    "success_count": 1,
                    "failure_count": 0,
                })
                print(f"  [PROCEDURAL] Reusing procedure: {proc_template['name']} "
                      f"(total uses: {existing[0].get('success_count', 0) + 1})")
            else:
                # Learn new procedure
                new_procedures.append({
                    **proc_template,
                    "success_count": 1,
                    "failure_count": 0,
                    "learned_at": datetime.now().isoformat(),
                })
                print(f"  [PROCEDURAL] Learned new procedure: {proc_template['name']}")

    return {"procedures": new_procedures}


def maybe_summarize_node(state: ThreeMemoryState) -> dict:
    """
    EPISODIC MEMORY (compression): Summarize old messages when history grows.

    When the conversation exceeds a threshold, older messages are
    summarized into a compact form.  This is how episodic memory
    transitions from "raw events" to "compressed episodes".

    The summary preserves KEY information while discarding details.
    Combined with semantic memory (facts), nothing important is lost.
    """
    messages = state.get("messages", [])

    # Only summarize if we have more than 6 messages (3 turns)
    if len(messages) <= 6:
        return {}

    old_summary = state.get("summary", "")

    # Take messages that will be summarized (all except last 4)
    to_summarize = messages[:-4]

    if not to_summarize:
        return {}

    # Build summary with LLM
    summary_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:100]}"
        for m in to_summarize
    )

    try:
        summary_prompt = [
            SystemMessage(content=(
                "Summarize this conversation segment in 2-3 sentences. "
                "Focus on: decisions made, tasks requested, and outcomes. "
                "Do NOT include facts about the user (those are stored separately)."
            )),
            HumanMessage(content=(
                f"Previous summary: {old_summary}\n\n"
                f"New messages to summarize:\n{summary_text}"
            )),
        ]
        response = llm.invoke(summary_prompt)
        new_summary = response.content.strip()
        print(f"  [EPISODIC] Summarized {len(to_summarize)} old messages")
        print(f"  [EPISODIC] Summary: {new_summary[:120]}...")
    except Exception:
        new_summary = old_summary  # Keep old summary on error

    return {"summary": new_summary}


def respond_node(state: ThreeMemoryState) -> dict:
    """
    Generate a response using ALL THREE memory categories.

    The system prompt is constructed from:
      1. EPISODIC: conversation summary + recent messages
      2. SEMANTIC: known facts + preferences
      3. PROCEDURAL: relevant procedures for the current task

    This is the key insight: each memory type contributes different
    context, and together they give the agent a complete picture.
    """
    user_input = state["user_input"]
    facts = state.get("facts", [])
    preferences = state.get("preferences", {})
    summary = state.get("summary", "")
    procedures = state.get("procedures", [])

    # -- Build system prompt with all three memory types -------
    system_parts = [
        "You are a helpful travel planning assistant with excellent memory.",
        "Use ALL the context below to personalize your responses.",
    ]

    # SEMANTIC MEMORY -> Known facts
    if facts:
        facts_str = "\n".join(f"  - {f}" for f in facts)
        system_parts.append(
            f"\n[SEMANTIC MEMORY -- Known facts about the user]\n{facts_str}"
        )

    # SEMANTIC MEMORY -> Preferences
    if preferences:
        pref_str = "\n".join(f"  - {k}: {v}" for k, v in preferences.items())
        system_parts.append(
            f"\n[SEMANTIC MEMORY -- User preferences]\n{pref_str}"
        )

    # EPISODIC MEMORY -> Conversation summary
    if summary:
        system_parts.append(
            f"\n[EPISODIC MEMORY -- Conversation summary]\n  {summary}"
        )

    # PROCEDURAL MEMORY -> Relevant procedures
    if procedures:
        # Find procedures relevant to current input
        input_words = set(user_input.lower().split())
        relevant = []
        for proc in procedures:
            trigger_words = set(proc.get("trigger", "").lower().split())
            if trigger_words & input_words:
                relevant.append(proc)

        if relevant:
            proc_strs = []
            for proc in relevant:
                steps = proc.get("steps", [])
                success = proc.get("success_count", 0)
                proc_strs.append(
                    f"  Procedure: {proc['name']} (used {success} times)\n" +
                    "\n".join(f"    {i+1}. {s}" for i, s in enumerate(steps))
                )
            system_parts.append(
                f"\n[PROCEDURAL MEMORY -- Relevant workflows]\n" +
                "\n".join(proc_strs) +
                "\n  Follow these steps when applicable."
            )

    system_parts.append(
        "\nIMPORTANT: Reference what you know about the user naturally. "
        "Don't list facts mechanically -- weave them into your response."
    )

    # -- Build full prompt -------------------------------------
    prompt = [SystemMessage(content="\n".join(system_parts))]

    # Add recent messages (episodic -- raw recent history)
    for msg in state.get("messages", [])[-6:]:  # Last 3 turns
        prompt.append(msg)

    prompt.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [RESPOND] {answer[:120]}...")

    return {
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ],
    }


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_memory_graph():
    """
    Build the three-memory-category agent.

    Flow per turn:
      extract_facts (semantic) -> extract_procedures (procedural)
        -> maybe_summarize (episodic compression) -> respond (uses all three)

    MemorySaver checkpoints state after each step, enabling:
      - Cross-turn persistence (same thread_id = same conversation)
      - Time-travel debugging (inspect any past state)
    """
    graph = StateGraph(ThreeMemoryState)

    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("extract_procedures", extract_procedures_node)
    graph.add_node("maybe_summarize", maybe_summarize_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("extract_facts")
    graph.add_edge("extract_facts", "extract_procedures")
    graph.add_edge("extract_procedures", "maybe_summarize")
    graph.add_edge("maybe_summarize", "respond")
    graph.add_edge("respond", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ================================================================
# DEMO -- Multi-turn conversation showing all three memory types
# ================================================================

def run_demo():
    app = build_memory_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    conversations = [
        # Turn 1: Establish facts (semantic memory)
        "Hi! I'm planning a trip to Japan next spring. "
        "My budget is $3000 for 2 weeks. I'm vegetarian and allergic to nuts.",

        # Turn 2: Task triggers procedural memory
        "Can you recommend some restaurants in Tokyo?",

        # Turn 3: More facts + procedural memory
        "I'd also love to see cherry blossoms. Can you plan a 3-day itinerary for Kyoto?",

        # Turn 4: Tests if agent remembers everything
        "Wait, remind me -- do you remember my dietary restrictions and budget?",

        # Turn 5: Another procedure, should reference accumulated knowledge
        "Great! Now help me calculate a detailed budget breakdown for the full trip.",

        # Turn 6: Tests episodic memory (should recall earlier conversation)
        "What was the first thing I told you about this trip?",
    ]

    print("\n" + "=" * 65)
    print("  THREE MEMORY CATEGORIES IN LANGGRAPH -- LIVE DEMO")
    print("=" * 65)
    print(f"  Thread ID: {thread_id[:8]}...")

    for i, user_input in enumerate(conversations):
        print(f"\n{'-' * 65}")
        print(f"  Turn {i + 1}: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
        print(f"{'-' * 65}")

        result = app.invoke(
            {
                "user_input": user_input,
                "messages": [],
                "summary": "",
                "facts": [],
                "preferences": {},
                "procedures": [],
            },
            config,
        )

        # Show memory state after each turn
        facts = result.get("facts", [])
        prefs = result.get("preferences", {})
        procs = result.get("procedures", [])
        summary = result.get("summary", "")
        msgs = result.get("messages", [])

        print(f"\n  -- Memory State After Turn {i + 1} --")
        print(f"  EPISODIC:    {len(msgs)} messages"
              f"{f', summary: {summary[:60]}...' if summary else ''}")
        print(f"  SEMANTIC:    {len(facts)} facts, {len(prefs)} preferences")
        if facts:
            for f in facts[-3:]:  # Show last 3 facts
                print(f"               - {f}")
        print(f"  PROCEDURAL:  {len(procs)} learned procedures")
        for p in procs:
            print(f"               - {p['name']} (used {p.get('success_count', 0)}x)")

        # Show AI response
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            print(f"\n  AI: {ai_msgs[-1].content[:300]}")

    # === FINAL MEMORY SUMMARY ===
    print(f"\n{'=' * 65}")
    print(f"  FINAL MEMORY STATE SUMMARY")
    print(f"{'=' * 65}")

    final_state = app.get_state(config)
    state_values = final_state.values

    print(f"\n  EPISODIC MEMORY (what happened):")
    print(f"    Messages: {len(state_values.get('messages', []))}")
    print(f"    Summary: {state_values.get('summary', 'None')[:100]}")

    print(f"\n  SEMANTIC MEMORY (what is true):")
    for f in state_values.get("facts", []):
        print(f"    - {f}")
    for k, v in state_values.get("preferences", {}).items():
        print(f"    - {k}: {v}")

    print(f"\n  PROCEDURAL MEMORY (how to do things):")
    for p in state_values.get("procedures", []):
        print(f"    - {p['name']}: {len(p.get('steps', []))} steps, "
              f"used {p.get('success_count', 0)}x")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 11b: Memory Categories in LangGraph")
    print("=" * 65)

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS")
    print("=" * 65)
    print(textwrap.dedent("""
    HOW EACH MEMORY CATEGORY MAPS TO LANGGRAPH:

    1. EPISODIC MEMORY (what happened):
       - Field: messages (Annotated[list, add_messages])
       - Field: summary (str, replaced on each compression)
       - Reducer: add_messages APPENDS new messages
       - Compression: maybe_summarize_node condenses old messages
       - Persisted via MemorySaver checkpointing

    2. SEMANTIC MEMORY (what is true):
       - Field: facts (Annotated[List[str], merge_facts])
       - Field: preferences (Dict[str, str])
       - Reducer: merge_facts APPENDS + DEDUPLICATES
       - Extraction: extract_facts_node runs on every turn
       - Facts survive even when old messages are summarized

    3. PROCEDURAL MEMORY (how to do things):
       - Field: procedures (Annotated[List[Dict], merge_procedures])
       - Reducer: merge_procedures APPENDS new, UPDATES existing
       - Learning: extract_procedures_node detects task patterns
       - Procedures track success_count to rank by reliability
       - Referenced in respond_node when trigger words match

    KEY DESIGN DECISIONS:
    - Each memory type has its own REDUCER (append vs replace vs merge)
    - Facts are EXTRACTED from messages and stored separately --
      they persist even when old messages get summarized away
    - Procedures are LEARNED from interactions and IMPROVE over time
    - The respond node COMBINES all three types in the system prompt
    - MemorySaver enables cross-turn persistence (same thread_id)

    PRODUCTION UPGRADES:
    - Replace MemorySaver -> PostgresSaver for persistence across restarts
    - Replace fact extraction heuristics -> fine-tuned classifier
    - Add vector store for semantic memory similarity search
    - Add procedure versioning and A/B testing
    """))
