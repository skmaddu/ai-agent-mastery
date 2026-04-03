import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5c: Pipeline & Debate Topologies with LLM Agents
==========================================================
Example 04 introduced three multi-agent topologies conceptually.
Example 05 implemented Supervisor/Worker. This example implements the
other two topologies with real LLM agents:

  1. SEQUENTIAL PIPELINE — assembly line, each agent transforms output
     and passes it to the next. Like a newspaper: reporter → editor → publisher.

  2. DEBATE/COMMITTEE — agents argue different perspectives, a judge
     synthesizes the best answer. Like a panel discussion.

These two topologies solve fundamentally different problems:
  Pipeline:  Each stage REFINES the output (quality improves at each step)
  Debate:    Multiple PERSPECTIVES on the same question (breadth of analysis)

Graph Structures:
  Pipeline: START → researcher → writer → editor → END
  Debate:   START → [optimist, skeptic, pragmatist] → judge → END

Run: python week-04-advanced-patterns/examples/example_05c_pipeline_debate.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict


# ==============================================================
# LLM Setup
# ==============================================================

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
# TOPOLOGY 1: SEQUENTIAL PIPELINE
# ================================================================
# Each agent takes the previous agent's output as input and refines it.
#
#   researcher → writer → editor → END
#   (facts)      (draft)   (polish)
#
# Key insight: Pipeline topology is ideal when each stage has a
# DIFFERENT skill (research ≠ writing ≠ editing) and the output
# quality improves at each stage.

class PipelineState(TypedDict):
    topic: str
    research: str       # Stage 1 output: raw facts
    draft: str          # Stage 2 output: written article
    final_article: str  # Stage 3 output: edited and polished


def researcher_node(state: PipelineState) -> dict:
    """Stage 1: Gather facts and key points on the topic."""
    print(f"\n  STAGE 1 — RESEARCHER: Gathering facts...")
    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research specialist. Given a topic, provide 5-7 key facts "
            "and data points. Output as a numbered list. Be specific with numbers "
            "and sources. Keep it under 200 words."
        )),
        HumanMessage(content=f"Research this topic: {state['topic']}"),
    ])
    result = response.content
    print(f"    Output: {result[:150]}...")
    return {"research": result}


def writer_node(state: PipelineState) -> dict:
    """Stage 2: Transform research into a well-structured article."""
    print(f"\n  STAGE 2 — WRITER: Drafting article from research...")
    llm = get_llm(temperature=0.7)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a skilled writer. Given research notes, write a clear, "
            "engaging 2-paragraph article. Use the facts provided but make it "
            "readable and compelling. Do NOT add facts that aren't in the research. "
            "Keep it under 200 words."
        )),
        HumanMessage(content=f"Topic: {state['topic']}\n\nResearch notes:\n{state['research']}"),
    ])
    result = response.content
    print(f"    Output: {result[:150]}...")
    return {"draft": result}


def editor_node(state: PipelineState) -> dict:
    """Stage 3: Polish the draft — fix grammar, improve flow, tighten prose."""
    print(f"\n  STAGE 3 — EDITOR: Polishing draft...")
    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior editor. Review and improve this article draft. "
            "Fix any grammar issues, improve sentence flow, cut unnecessary words, "
            "and ensure the opening hooks the reader. Keep the same length. "
            "Output ONLY the improved article, no commentary."
        )),
        HumanMessage(content=f"Draft to edit:\n\n{state['draft']}"),
    ])
    result = response.content
    print(f"    Output: {result[:150]}...")
    return {"final_article": result}


def build_pipeline():
    """Build the sequential pipeline graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    return graph.compile()


def run_pipeline(topic: str):
    """Run the pipeline and show how output improves at each stage."""
    print(f"\n{'='*60}")
    print(f"  PIPELINE TOPOLOGY: {topic}")
    print(f"{'='*60}")
    print("  Flow: Researcher → Writer → Editor")

    app = build_pipeline()
    result = app.invoke({
        "topic": topic,
        "research": "",
        "draft": "",
        "final_article": "",
    })

    print(f"\n  {'- '*30}")
    print(f"  PIPELINE RESULT:")
    print(f"  {'- '*30}")
    print(f"\n{result['final_article']}")
    return result


# ================================================================
# TOPOLOGY 2: DEBATE / COMMITTEE
# ================================================================
# Multiple agents analyze the SAME question from different angles.
# A judge synthesizes their perspectives into a balanced conclusion.
#
#   START → optimist  ─┐
#         → skeptic   ─┼─→ judge → END
#         → pragmatist─┘
#
# Key insight: Debate topology is ideal when a question has multiple
# valid perspectives and you want a BALANCED analysis, not just one view.

class DebateState(TypedDict):
    question: str
    optimist_view: str      # Pro / opportunities perspective
    skeptic_view: str       # Con / risks perspective
    pragmatist_view: str    # Practical / implementation perspective
    synthesis: str          # Judge's balanced conclusion


def optimist_node(state: DebateState) -> dict:
    """Argue the optimistic / pro perspective."""
    print(f"\n  OPTIMIST: Arguing the positive case...")
    llm = get_llm(temperature=0.7)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an OPTIMIST debater. Given a topic, argue strongly for its "
            "benefits, opportunities, and positive potential. Be specific with "
            "examples and data. Present the best possible case. Keep to 100 words."
        )),
        HumanMessage(content=f"Topic: {state['question']}"),
    ])
    result = response.content
    print(f"    View: {result[:120]}...")
    return {"optimist_view": result}


def skeptic_node(state: DebateState) -> dict:
    """Argue the skeptical / con perspective."""
    print(f"\n  SKEPTIC: Arguing the critical case...")
    llm = get_llm(temperature=0.7)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a SKEPTIC debater. Given a topic, argue against it — focus on "
            "risks, downsides, hidden costs, and potential failures. Be specific "
            "with examples and data. Present the strongest critique. Keep to 100 words."
        )),
        HumanMessage(content=f"Topic: {state['question']}"),
    ])
    result = response.content
    print(f"    View: {result[:120]}...")
    return {"skeptic_view": result}


def pragmatist_node(state: DebateState) -> dict:
    """Argue the practical / implementation perspective."""
    print(f"\n  PRAGMATIST: Arguing the practical case...")
    llm = get_llm(temperature=0.5)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a PRAGMATIST. Given a topic, focus on practical implementation: "
            "what would it actually take? What are the realistic timelines, costs, "
            "and prerequisites? Skip the hype and doom — focus on what works. "
            "Keep to 100 words."
        )),
        HumanMessage(content=f"Topic: {state['question']}"),
    ])
    result = response.content
    print(f"    View: {result[:120]}...")
    return {"pragmatist_view": result}


def judge_node(state: DebateState) -> dict:
    """Synthesize all perspectives into a balanced conclusion."""
    print(f"\n  JUDGE: Synthesizing all perspectives...")
    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a neutral JUDGE synthesizing a debate. You received three "
            "perspectives: optimist, skeptic, and pragmatist. Your job:\n"
            "1. Acknowledge the strongest point from EACH perspective\n"
            "2. Identify where they agree and disagree\n"
            "3. Provide a balanced, nuanced conclusion\n"
            "Keep to 150 words. Be fair to all sides."
        )),
        HumanMessage(content=(
            f"Topic: {state['question']}\n\n"
            f"OPTIMIST:\n{state['optimist_view']}\n\n"
            f"SKEPTIC:\n{state['skeptic_view']}\n\n"
            f"PRAGMATIST:\n{state['pragmatist_view']}\n\n"
            "Synthesize a balanced conclusion:"
        )),
    ])
    result = response.content
    print(f"    Synthesis: {result[:150]}...")
    return {"synthesis": result}


def build_debate():
    """Build the debate graph.

    Note: The three debaters run SEQUENTIALLY in this implementation
    because LangGraph processes nodes one at a time. For true parallel
    execution, see example_02d (ThreadPoolExecutor approach).
    """
    graph = StateGraph(DebateState)

    graph.add_node("optimist", optimist_node)
    graph.add_node("skeptic", skeptic_node)
    graph.add_node("pragmatist", pragmatist_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("optimist")
    graph.add_edge("optimist", "skeptic")
    graph.add_edge("skeptic", "pragmatist")
    graph.add_edge("pragmatist", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def run_debate(question: str):
    """Run the debate and show each perspective + synthesis."""
    print(f"\n{'='*60}")
    print(f"  DEBATE TOPOLOGY: {question}")
    print(f"{'='*60}")
    print("  Flow: Optimist → Skeptic → Pragmatist → Judge")

    app = build_debate()
    result = app.invoke({
        "question": question,
        "optimist_view": "",
        "skeptic_view": "",
        "pragmatist_view": "",
        "synthesis": "",
    })

    print(f"\n  {'- '*30}")
    print(f"  DEBATE SYNTHESIS:")
    print(f"  {'- '*30}")
    print(f"\n{result['synthesis']}")
    return result


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("Example 5c: Pipeline & Debate Topologies")
    print("=" * 60)

    # --- Pipeline Demo ---
    pipeline_result = run_pipeline("The impact of artificial intelligence on healthcare")

    # --- Debate Demo ---
    debate_result = run_debate("Should companies adopt AI agents for customer service?")

    # --- Comparison ---
    print(f"\n\n{'='*60}")
    print("  TOPOLOGY COMPARISON")
    print(f"{'='*60}")
    print("""
  SEQUENTIAL PIPELINE:
    Flow:    A → B → C (each stage refines)
    Best for: Content creation, data processing, ETL
    Example: Research → Write → Edit
    Agents:  Different skills, same data flowing through
    Output:  Single refined result

  DEBATE / COMMITTEE:
    Flow:    [A, B, C] → Judge (parallel perspectives)
    Best for: Analysis, decision-making, risk assessment
    Example: Optimist + Skeptic + Pragmatist → Judge
    Agents:  Same skills, different viewpoints
    Output:  Balanced synthesis of perspectives

  SUPERVISOR/WORKER (Example 05):
    Flow:    Boss → [Workers] → Boss (delegated sub-tasks)
    Best for: Complex research, multi-part tasks
    Example: Supervisor → Researcher + Analyst → Judge
    Agents:  Different skills, different sub-tasks
    Output:  Combined results from parallel work

  When to use which:
    "Process this data through stages"      → Pipeline
    "Analyze this from multiple angles"     → Debate
    "Break this big task into pieces"       → Supervisor/Worker
    "Route this to the right specialist"    → Intent Router (05b)
""")
    print(f"{'='*60}")
