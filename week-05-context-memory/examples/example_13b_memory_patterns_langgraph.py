import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 13b: Memory Patterns in LangGraph -- Real Implementation
==================================================================
LangGraph implementation of the three memory patterns from Example 13:

  1. Hierarchical Memory (L1 hot cache -> L2 summary -> L3 facts)
     -- Real LLM-based summarization and fact extraction
     -- Automatic compaction triggered by token budget

  2. Shared + Private Memory in Multi-Agent Systems
     -- Blackboard pattern: all agents read/write shared state
     -- Each agent has its own private working buffer

  3. Multi-Agent Memory with Supervisor Routing
     -- Supervisor sees full state; workers see scoped views
     -- Workers contribute to shared output; supervisor merges

All three patterns are implemented as runnable LangGraph graphs with
real LLM calls, MemorySaver checkpointing, and Phoenix tracing.

Run: python week-05-context-memory/examples/example_13b_memory_patterns_langgraph.py
"""

import os
import sys
import uuid
import textwrap
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

# -- Phoenix ----------------------------------------------------------
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
        tracer_provider = register(project_name="week5-memory-patterns-langgraph")
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


llm = get_llm(temperature=0.3)


# =====================================================================
# PATTERN 1: HIERARCHICAL MEMORY (L1 -> L2 -> L3) IN LANGGRAPH
# =====================================================================
#
# Inspired by CPU cache hierarchy:
#   L1 (Hot Cache):  Last N messages, full text       ~fast, detailed
#   L2 (Summary):    LLM-summarized blocks             ~medium
#   L3 (Archive):    LLM-extracted key facts            ~compact, permanent
#
# When L1 exceeds a token budget, old messages are summarized (L2).
# When L2 has too many summaries, key facts are extracted (L3).
#
# In LangGraph, each layer is a STATE FIELD with its own reducer.
# Compaction is a GRAPH NODE that runs conditionally.

def merge_unique_strings(existing: List[str], new: List[str]) -> List[str]:
    """Reducer: append + deduplicate strings."""
    combined = list(existing)
    for item in new:
        if item not in combined:
            combined.append(item)
    return combined


class HierarchicalMemoryState(TypedDict):
    user_input: str

    # L1: Full recent messages (add_messages appends)
    messages: Annotated[list, add_messages]

    # L2: Summaries of older message blocks (append + dedup)
    summaries: Annotated[List[str], merge_unique_strings]

    # L3: Permanent facts extracted from summaries (append + dedup)
    archived_facts: Annotated[List[str], merge_unique_strings]

    # Config
    l1_max_messages: int    # When to trigger L1 -> L2 compaction
    l2_max_summaries: int   # When to trigger L2 -> L3 compaction


# -- Nodes -------------------------------------------------------------

def respond_node(state: HierarchicalMemoryState) -> dict:
    """Generate a response using all three memory layers as context."""
    user_input = state["user_input"]

    # Build context from all layers (L3 first -- most compressed, always relevant)
    system_parts = ["You are a helpful travel assistant with excellent memory."]

    archived = state.get("archived_facts", [])
    if archived:
        system_parts.append(
            "\n[ARCHIVED FACTS -- permanent knowledge]\n" +
            "\n".join(f"  - {f}" for f in archived)
        )

    summaries = state.get("summaries", [])
    if summaries:
        system_parts.append(
            "\n[CONVERSATION SUMMARIES -- older context]\n" +
            "\n".join(f"  - {s}" for s in summaries[-3:])  # Last 3 summaries
        )

    system_parts.append(
        "\nUse all the context above to give personalized responses. "
        "Reference what you know naturally."
    )

    prompt = [SystemMessage(content="\n".join(system_parts))]

    # L1: Add recent messages (most recent, full detail)
    for msg in state.get("messages", [])[-8:]:
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


def compact_l1_to_l2(state: HierarchicalMemoryState) -> dict:
    """
    L1 -> L2: Summarize old messages when L1 exceeds budget.

    Takes the oldest messages from L1, summarizes them with the LLM,
    and stores the summary in L2.  Recent messages stay in L1 (full text).
    """
    messages = state.get("messages", [])
    max_msgs = state.get("l1_max_messages", 8)

    if len(messages) <= max_msgs:
        return {}

    # Messages to summarize (all except the most recent ones to keep)
    keep_recent = 4
    to_summarize = messages[:-keep_recent]

    if not to_summarize:
        return {}

    # Format messages for summarization
    msg_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
        f"{m.content[:150]}"
        for m in to_summarize
    )

    try:
        summary_prompt = [
            SystemMessage(content=(
                "Summarize this conversation segment in 1-2 sentences. "
                "Focus on: topics discussed, decisions made, and key information exchanged. "
                "Be concise but preserve important details."
            )),
            HumanMessage(content=f"Conversation to summarize:\n{msg_text}"),
        ]
        response = llm.invoke(summary_prompt)
        summary = response.content.strip()
    except Exception as e:
        summary = f"Earlier conversation about: {msg_text[:100]}"

    print(f"  [L1->L2] Summarized {len(to_summarize)} messages:")
    print(f"           \"{summary[:80]}...\"")

    return {"summaries": [summary]}


def compact_l2_to_l3(state: HierarchicalMemoryState) -> dict:
    """
    L2 -> L3: Extract permanent facts when L2 has too many summaries.

    Takes older summaries, uses the LLM to extract key facts, and
    stores them in L3 (archive).  These facts NEVER expire.
    """
    summaries = state.get("summaries", [])
    max_summaries = state.get("l2_max_summaries", 5)

    if len(summaries) <= max_summaries:
        return {}

    # Summarize the oldest summaries into facts
    to_archive = summaries[:-2]  # Keep 2 most recent summaries

    archive_text = "\n".join(f"- {s}" for s in to_archive)

    try:
        extract_prompt = [
            SystemMessage(content=(
                "Extract permanent facts from these conversation summaries. "
                "Only extract facts that should ALWAYS be remembered: "
                "user preferences, constraints, personal info, key decisions. "
                "Return each fact on its own line. Keep each under 15 words."
            )),
            HumanMessage(content=f"Summaries:\n{archive_text}"),
        ]
        response = llm.invoke(extract_prompt)
        facts = [
            line.strip().strip("- *")
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 3
        ]
    except Exception:
        facts = [f"Archived: {s[:60]}" for s in to_archive]

    print(f"  [L2->L3] Archived {len(to_archive)} summaries -> {len(facts)} facts:")
    for f in facts[:5]:
        print(f"           - {f}")

    # Clear archived summaries (keep only recent ones)
    remaining_summaries = summaries[-2:]

    return {"archived_facts": facts, "summaries": remaining_summaries}


def should_compact_l1(state: HierarchicalMemoryState) -> str:
    """Check if L1 needs compaction."""
    messages = state.get("messages", [])
    max_msgs = state.get("l1_max_messages", 8)
    if len(messages) > max_msgs:
        return "compact_l1"
    return "skip_compact"


def should_compact_l2(state: HierarchicalMemoryState) -> str:
    """Check if L2 needs compaction."""
    summaries = state.get("summaries", [])
    max_summaries = state.get("l2_max_summaries", 5)
    if len(summaries) > max_summaries:
        return "compact_l2"
    return "skip_l2"


def build_hierarchical_memory_graph():
    """
    Graph flow:
      respond -> [L1 full?] -> compact_l1 -> [L2 full?] -> compact_l2 -> END
                     |                            |
                     +-> END                      +-> END
    """
    graph = StateGraph(HierarchicalMemoryState)

    graph.add_node("respond", respond_node)
    graph.add_node("compact_l1", compact_l1_to_l2)
    graph.add_node("compact_l2", compact_l2_to_l3)

    graph.set_entry_point("respond")

    graph.add_conditional_edges("respond", should_compact_l1, {
        "compact_l1": "compact_l1",
        "skip_compact": END,
    })

    graph.add_conditional_edges("compact_l1", should_compact_l2, {
        "compact_l2": "compact_l2",
        "skip_l2": END,
    })

    graph.add_edge("compact_l2", END)

    return graph.compile(checkpointer=MemorySaver())


def demo_hierarchical_memory():
    """Demo: watch messages flow from L1 -> L2 -> L3 over many turns."""

    print("=" * 65)
    print("  PATTERN 1: HIERARCHICAL MEMORY (L1 -> L2 -> L3)")
    print("=" * 65)

    app = build_hierarchical_memory_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Use small limits so compaction triggers quickly in the demo
    base_state = {
        "user_input": "",
        "messages": [],
        "summaries": [],
        "archived_facts": [],
        "l1_max_messages": 6,    # Compact L1 after 6 messages (3 turns)
        "l2_max_summaries": 3,   # Compact L2 after 3 summaries
    }

    turns = [
        "I'm planning a trip to Japan. Budget is $3000 for 2 weeks.",
        "I'm vegetarian and allergic to nuts. What food should I try?",
        "Should I get the Japan Rail Pass?",
        "What are the best neighborhoods to stay in Tokyo?",
        "Can you suggest a 3-day Kyoto itinerary?",
        "How about day trips from Osaka?",
        "What's the weather like in April?",
        "Any tips for cherry blossom viewing spots?",
    ]

    for i, user_input in enumerate(turns):
        print(f"\n{'=' * 65}")
        print(f"  Turn {i + 1}: {user_input}")
        print(f"{'=' * 65}")

        result = app.invoke(
            {**base_state, "user_input": user_input},
            config,
        )

        # Show memory state
        msgs = result.get("messages", [])
        sums = result.get("summaries", [])
        facts = result.get("archived_facts", [])

        print(f"\n  Memory State:")
        print(f"    L1 (messages):  {len(msgs)} messages")
        print(f"    L2 (summaries): {len(sums)} summaries")
        if sums:
            for s in sums[-2:]:
                print(f"      \"{s[:70]}...\"")
        print(f"    L3 (facts):     {len(facts)} archived facts")
        if facts:
            for f in facts:
                print(f"      - {f}")

        # Show AI response (truncated)
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            print(f"\n  AI: {ai_msgs[-1].content[:200]}")


# =====================================================================
# PATTERN 2: SHARED + PRIVATE MEMORY (BLACKBOARD PATTERN)
# =====================================================================
#
# Multi-agent system where:
#   - SHARED state: task info, user prefs, final outputs (all agents see)
#   - PRIVATE buffers: per-agent working data (only that agent sees)
#
# In LangGraph, this is implemented as a single TypedDict where:
#   - Shared fields are read/written by all nodes
#   - Private fields are prefixed with the agent name
#   - Each node only reads/writes its own private fields + shared fields

def merge_strings(existing: List[str], new: List[str]) -> List[str]:
    """Reducer: append strings."""
    return list(existing) + list(new)


class BlackboardState(TypedDict):
    # -- SHARED (all agents read/write) --------------------------------
    task: str                                    # What needs to be done
    user_preferences: Dict[str, str]             # User's requirements
    status: str                                  # Current workflow status
    final_output: str                            # Assembled final result

    # -- RESEARCHER PRIVATE BUFFER -------------------------------------
    researcher_sources: Annotated[List[str], merge_strings]   # Found sources
    researcher_notes: str                        # Working notes

    # -- WRITER PRIVATE BUFFER -----------------------------------------
    writer_outline: List[str]                    # Document outline
    writer_draft: str                            # Current draft

    # -- REVIEWER PRIVATE BUFFER ---------------------------------------
    reviewer_feedback: Annotated[List[str], merge_strings]  # Feedback items
    reviewer_score: float                        # Quality score


def researcher_node(state: BlackboardState) -> dict:
    """
    Researcher agent: searches for information based on the task.

    READS: task, user_preferences (shared)
    WRITES: researcher_sources, researcher_notes (private), status (shared)
    """
    task = state["task"]
    prefs = state.get("user_preferences", {})

    prompt = [
        SystemMessage(content=(
            "You are a research agent. Given a task, identify 3-4 key points "
            "that should be covered. For each point, write a brief finding "
            "(1-2 sentences). Format as a numbered list."
        )),
        HumanMessage(content=(
            f"Task: {task}\n"
            f"User preferences: {prefs}\n"
            f"Research and list key findings:"
        )),
    ]

    try:
        response = llm.invoke(prompt)
        notes = response.content.strip()
    except Exception as e:
        notes = f"[Research error: {e}]"

    # Extract individual findings as sources
    sources = [
        line.strip().strip("0123456789.)-")
        for line in notes.split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    print(f"  [RESEARCHER] Found {len(sources)} points")
    for s in sources[:3]:
        print(f"    - {s[:70]}")

    return {
        "researcher_sources": sources,
        "researcher_notes": notes,
        "status": "research_complete",
    }


def writer_node(state: BlackboardState) -> dict:
    """
    Writer agent: drafts content based on research.

    READS: task, user_preferences (shared), researcher_notes (cross-agent)
    WRITES: writer_outline, writer_draft (private), status (shared)
    """
    task = state["task"]
    prefs = state.get("user_preferences", {})
    research = state.get("researcher_notes", "")

    detail_level = prefs.get("detail_level", "medium")

    prompt = [
        SystemMessage(content=(
            f"You are a writer agent. Based on the research provided, write a "
            f"concise {detail_level}-detail response. Use clear structure with "
            f"sections. Keep it under 200 words."
        )),
        HumanMessage(content=(
            f"Task: {task}\n"
            f"Research notes:\n{research}\n\n"
            f"Write a well-structured response:"
        )),
    ]

    try:
        response = llm.invoke(prompt)
        draft = response.content.strip()
    except Exception as e:
        draft = f"[Draft error: {e}]"

    # Extract outline from the draft (section headers)
    outline = [
        line.strip().strip("#").strip()
        for line in draft.split("\n")
        if line.strip().startswith("#") or line.strip().startswith("**")
    ]
    if not outline:
        outline = ["Introduction", "Key Points", "Conclusion"]

    print(f"  [WRITER] Drafted {len(draft.split())} words, {len(outline)} sections")

    return {
        "writer_outline": outline,
        "writer_draft": draft,
        "status": "draft_complete",
    }


def reviewer_node(state: BlackboardState) -> dict:
    """
    Reviewer agent: evaluates the draft and provides feedback.

    READS: task, user_preferences (shared), writer_draft (cross-agent)
    WRITES: reviewer_feedback, reviewer_score (private), final_output, status (shared)
    """
    task = state["task"]
    draft = state.get("writer_draft", "")

    prompt = [
        SystemMessage(content=(
            "You are a reviewer agent. Evaluate this draft for:\n"
            "1. Completeness (covers the topic?)\n"
            "2. Accuracy (claims are reasonable?)\n"
            "3. Clarity (well-structured and readable?)\n\n"
            "Give a score (1-10) and 2-3 specific feedback items. "
            "Format: Score: X/10 then feedback as a numbered list."
        )),
        HumanMessage(content=(
            f"Task: {task}\n"
            f"Draft to review:\n{draft}"
        )),
    ]

    try:
        response = llm.invoke(prompt)
        review = response.content.strip()
    except Exception as e:
        review = f"Score: 5/10\n1. [Review error: {e}]"

    # Parse score
    score = 7.0
    for line in review.split("\n"):
        if "score" in line.lower() and "/" in line:
            try:
                num = line.split("/")[0]
                score = float("".join(c for c in num if c.isdigit() or c == "."))
            except ValueError:
                pass

    # Parse feedback items
    feedback = [
        line.strip().strip("0123456789.)-")
        for line in review.split("\n")
        if line.strip() and not line.lower().startswith("score")
        and len(line.strip()) > 10
    ]

    print(f"  [REVIEWER] Score: {score}/10, {len(feedback)} feedback items")
    for f in feedback[:3]:
        print(f"    - {f[:70]}")

    return {
        "reviewer_feedback": feedback,
        "reviewer_score": score,
        "final_output": draft,  # In a real system, writer would revise based on feedback
        "status": "review_complete",
    }


def build_blackboard_graph():
    """
    Blackboard pattern: researcher -> writer -> reviewer.

    All agents share the same state (blackboard). Each agent reads
    shared fields + other agents' outputs, and writes to its own
    private buffer + shared status.
    """
    graph = StateGraph(BlackboardState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


def demo_blackboard_pattern():
    """Demo: three agents collaborating via shared blackboard state."""

    print("\n" + "=" * 65)
    print("  PATTERN 2: SHARED + PRIVATE MEMORY (BLACKBOARD)")
    print("=" * 65)

    app = build_blackboard_graph()

    initial_state = {
        "task": "Explain the key differences between RLHF and DPO for AI alignment",
        "user_preferences": {"detail_level": "high", "format": "structured report"},
        "status": "starting",
        "final_output": "",
        "researcher_sources": [],
        "researcher_notes": "",
        "writer_outline": [],
        "writer_draft": "",
        "reviewer_feedback": [],
        "reviewer_score": 0.0,
    }

    print(f"\n  Task: {initial_state['task']}")
    print(f"  Preferences: {initial_state['user_preferences']}")

    result = app.invoke(initial_state, {"run_name": "blackboard-demo"})

    print(f"\n  {'=' * 55}")
    print(f"  FINAL STATE:")
    print(f"  {'=' * 55}")
    print(f"  Status: {result['status']}")
    print(f"  Researcher: {len(result['researcher_sources'])} sources found")
    print(f"  Writer: {len(result['writer_draft'].split())} words, "
          f"{len(result['writer_outline'])} sections")
    print(f"  Reviewer: {result['reviewer_score']}/10, "
          f"{len(result['reviewer_feedback'])} feedback items")
    print(f"\n  Final Output:\n  {result['final_output'][:400]}")


# =====================================================================
# PATTERN 3: SUPERVISOR + SCOPED WORKER MEMORY
# =====================================================================
#
# The supervisor has FULL state access and routes tasks to workers.
# Workers only see their SCOPED view (task + relevant context).
# This prevents information leakage and keeps worker prompts focused.
#
# In LangGraph, scoping is done in node logic: each worker node
# only reads the fields it needs from the full state.

class SupervisorState(TypedDict):
    # -- Supervisor has full access to everything --
    task: str
    subtasks: List[str]
    current_subtask_index: int
    worker_results: Annotated[List[Dict], merge_strings]

    # -- Scoped context for workers --
    current_worker_input: str        # What the current worker should do
    current_worker_context: str      # Relevant context for current worker

    # -- Accumulated output --
    final_report: str
    status: str


def supervisor_plan_node(state: SupervisorState) -> dict:
    """
    Supervisor: decomposes the task into subtasks.

    Has FULL state access. Decides which subtasks to assign and
    what context each worker needs.
    """
    task = state["task"]

    prompt = [
        SystemMessage(content=(
            "You are a supervisor agent. Break down this task into 2-3 subtasks. "
            "Each subtask should be a focused, self-contained research question. "
            "Return ONLY the subtasks, one per line, numbered."
        )),
        HumanMessage(content=f"Task: {task}"),
    ]

    try:
        response = llm.invoke(prompt)
        subtask_lines = [
            line.strip().strip("0123456789.)-").strip()
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
    except Exception:
        subtask_lines = [f"Research: {task}"]

    print(f"  [SUPERVISOR] Planned {len(subtask_lines)} subtasks:")
    for i, st in enumerate(subtask_lines):
        print(f"    {i + 1}. {st[:70]}")

    return {
        "subtasks": subtask_lines,
        "current_subtask_index": 0,
        "status": "planned",
    }


def supervisor_assign_node(state: SupervisorState) -> dict:
    """
    Supervisor: assigns the next subtask to a worker with SCOPED context.

    The worker only sees:
      - Its specific subtask (not the full plan)
      - Relevant context (not all previous results)
    """
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx >= len(subtasks):
        return {"status": "all_assigned"}

    current_subtask = subtasks[idx]

    # Scope the context: worker only sees results from previous subtasks
    # that are relevant (not ALL results)
    prev_results = state.get("worker_results", [])
    context = ""
    if prev_results:
        context = "Previous findings:\n" + "\n".join(
            f"- {r}" if isinstance(r, str) else f"- {r.get('result', '')[:100]}"
            for r in prev_results[-2:]  # Only last 2 results
        )

    print(f"  [SUPERVISOR] Assigning subtask {idx + 1}: {current_subtask[:60]}")

    return {
        "current_worker_input": current_subtask,
        "current_worker_context": context,
        "status": f"assigned_subtask_{idx + 1}",
    }


def worker_node(state: SupervisorState) -> dict:
    """
    Worker: executes a subtask with SCOPED memory.

    Can only see:
      - current_worker_input (its specific subtask)
      - current_worker_context (relevant prior results)

    Cannot see: full task, other subtasks, supervisor's plan.
    """
    subtask = state.get("current_worker_input", "")
    context = state.get("current_worker_context", "")

    prompt_parts = [
        SystemMessage(content=(
            "You are a focused research worker. Answer the question concisely "
            "in 2-3 sentences based on your knowledge."
        )),
    ]

    if context:
        prompt_parts.append(
            HumanMessage(content=f"Context from prior research:\n{context}\n\n"
                        f"Your task: {subtask}")
        )
    else:
        prompt_parts.append(HumanMessage(content=f"Your task: {subtask}"))

    try:
        response = llm.invoke(prompt_parts)
        result = response.content.strip()
    except Exception as e:
        result = f"[Worker error: {e}]"

    print(f"  [WORKER] Completed: {result[:80]}...")

    idx = state.get("current_subtask_index", 0)

    return {
        "worker_results": [{"subtask": subtask, "result": result}],
        "current_subtask_index": idx + 1,
    }


def should_assign_more(state: SupervisorState) -> str:
    """Check if there are more subtasks to assign."""
    idx = state.get("current_subtask_index", 0)
    subtasks = state.get("subtasks", [])
    if idx < len(subtasks):
        return "assign_next"
    return "synthesize"


def supervisor_synthesize_node(state: SupervisorState) -> dict:
    """
    Supervisor: merges all worker results into a final report.

    Has FULL access to all worker results and the original task.
    """
    task = state["task"]
    results = state.get("worker_results", [])

    results_text = "\n\n".join(
        f"Subtask: {r['subtask'] if isinstance(r, dict) else 'unknown'}\n"
        f"Finding: {r['result'] if isinstance(r, dict) else str(r)}"
        for r in results
    )

    prompt = [
        SystemMessage(content=(
            "You are a supervisor. Synthesize these research findings into a "
            "coherent final report. Be concise (under 150 words). Use clear structure."
        )),
        HumanMessage(content=(
            f"Original task: {task}\n\n"
            f"Worker findings:\n{results_text}\n\n"
            f"Synthesized report:"
        )),
    ]

    try:
        response = llm.invoke(prompt)
        report = response.content.strip()
    except Exception as e:
        report = f"[Synthesis error: {e}]"

    print(f"  [SUPERVISOR] Synthesized {len(results)} findings into report")

    return {
        "final_report": report,
        "status": "complete",
    }


def build_supervisor_graph():
    """
    Supervisor pattern:
      plan -> assign -> worker -> [more subtasks?] -> assign (loop)
                                                   -> synthesize -> END
    """
    graph = StateGraph(SupervisorState)

    graph.add_node("plan", supervisor_plan_node)
    graph.add_node("assign", supervisor_assign_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesize", supervisor_synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "assign")
    graph.add_edge("assign", "worker")

    graph.add_conditional_edges("worker", should_assign_more, {
        "assign_next": "assign",
        "synthesize": "synthesize",
    })

    graph.add_edge("synthesize", END)

    return graph.compile()


def demo_supervisor_pattern():
    """Demo: supervisor decomposes, assigns to workers, synthesizes."""

    print("\n" + "=" * 65)
    print("  PATTERN 3: SUPERVISOR + SCOPED WORKER MEMORY")
    print("=" * 65)

    app = build_supervisor_graph()

    initial_state = {
        "task": "Compare the EU AI Act and US AI Executive Order: scope, requirements, and enforcement",
        "subtasks": [],
        "current_subtask_index": 0,
        "worker_results": [],
        "current_worker_input": "",
        "current_worker_context": "",
        "final_report": "",
        "status": "starting",
    }

    print(f"\n  Task: {initial_state['task']}")

    result = app.invoke(initial_state, {"run_name": "supervisor-demo"})

    print(f"\n  {'=' * 55}")
    print(f"  FINAL REPORT:")
    print(f"  {'=' * 55}")
    print(f"  {result['final_report'][:500]}")
    print(f"\n  Status: {result['status']}")
    print(f"  Subtasks completed: {len(result.get('worker_results', []))}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 13b: Memory Patterns in LangGraph")
    print("=" * 65)

    setup_phoenix()

    # Run all three pattern demos
    demo_hierarchical_memory()
    demo_blackboard_pattern()
    demo_supervisor_pattern()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS")
    print("=" * 65)
    print(textwrap.dedent("""
    PATTERN 1 -- HIERARCHICAL MEMORY (L1 -> L2 -> L3):
      - L1 (messages): Full recent messages, add_messages reducer
      - L2 (summaries): LLM-summarized blocks, triggered by token budget
      - L3 (archived_facts): LLM-extracted permanent facts
      - Compaction is conditional graph edges: respond -> [L1 full?] -> compact
      - MemorySaver persists all layers across turns

    PATTERN 2 -- BLACKBOARD (SHARED + PRIVATE MEMORY):
      - One TypedDict holds BOTH shared and private fields
      - Shared: task, user_preferences, status, final_output
      - Private: researcher_sources, writer_draft, reviewer_feedback
      - All agents read shared state; each writes its own private buffer
      - LangGraph enforces this through node logic, not access control

    PATTERN 3 -- SUPERVISOR + SCOPED WORKERS:
      - Supervisor has FULL state access (plans, assigns, synthesizes)
      - Workers get SCOPED views (only their subtask + relevant context)
      - Prevents information overload and keeps worker prompts focused
      - Supervisor assign node controls what each worker can see
      - Worker loop: assign -> work -> [more?] -> assign / synthesize

    WHEN TO USE EACH:
      Hierarchical: Long conversations that exceed context window
      Blackboard:   Collaborative agents working on the same artifact
      Supervisor:   Complex tasks that decompose into independent subtasks
    """))
