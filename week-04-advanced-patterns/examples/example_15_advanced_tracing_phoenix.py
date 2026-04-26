import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 15: Advanced Tracing with Phoenix for Multi-Agent Systems
==================================================================
Phoenix is the "X-ray machine" for your agent harness. When your
agent produces bad output, takes too long, or costs too much, you
don't guess — you look at the trace.

This example runs 4 REAL LangGraph workflows, each demonstrating a
different trace pattern you'll encounter in production:

  Trace 1: HOT NODE — one node dominates total latency
  Trace 2: LOOP THAT WON'T DIE — evaluate-optimize stuck in a loop
  Trace 3: COST SPIKE — context grows each iteration, token count explodes
  Trace 4: MULTI-AGENT RESEARCH — supervisor → researcher (tools) → analyst → judge

All 4 appear as separate traces in Phoenix at http://localhost:6006.
Open the dashboard to compare span durations, token counts, and structure.

Run: python week-04-advanced-patterns/examples/example_15_advanced_tracing_phoenix.py
"""

import os
import time
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# Phoenix Setup — MUST happen before any LangChain/LangGraph imports
# ================================================================

PHOENIX_AVAILABLE = False

try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    PHOENIX_AVAILABLE = True
    print("[Phoenix] Available — traces will appear at http://localhost:6006")
except ImportError:
    print("[Phoenix] Not installed.")
    print("  Install: pip install arize-phoenix openinference-instrumentation-langchain")
    print("  Exiting — this example requires Phoenix.")
    sys.exit(1)


def setup_phoenix():
    """Launch Phoenix and instrument LangChain before any imports."""
    try:
        # use_temp_dir=False avoids Windows PermissionError on cleanup
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week4-multi-agent")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"[Phoenix] Dashboard: {session.url}")
        return session
    except Exception as e:
        print(f"[Phoenix] Setup failed: {e}")
        return None


# ================================================================
# LLM Setup
# ================================================================

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
# TRACE 1: THE HOT NODE
# ================================================================
# One node does heavy work (multiple LLM calls), others are fast.
# In Phoenix: one span will dominate the total duration.
#
#   Graph: quick_setup → heavy_researcher → quick_formatter → END
#   Expected: heavy_researcher takes 80%+ of total time

def run_trace_1_hot_node():
    """Trace 1: One node dominates latency."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n{'='*60}")
    print("  TRACE 1: THE HOT NODE")
    print(f"{'='*60}")
    print("  Graph: quick_setup → heavy_researcher → quick_formatter → END")
    print("  Watch: heavy_researcher span will dominate in Phoenix\n")

    llm = get_llm(temperature=0.3)

    class HotNodeState(TypedDict):
        topic: str
        setup_output: str
        research_output: str
        final_output: str

    def quick_setup_node(state: HotNodeState) -> dict:
        """Fast node — just reformats the topic. ~1 LLM call."""
        print("  [quick_setup] Formatting topic...")
        response = llm.invoke([
            SystemMessage(content="Rephrase this topic as a research question. One sentence only."),
            HumanMessage(content=state["topic"]),
        ])
        print(f"    Done: {response.content[:80]}...")
        return {"setup_output": response.content}

    def heavy_researcher_node(state: HotNodeState) -> dict:
        """HOT NODE — makes 3 sequential LLM calls. Takes 3x longer."""
        print("  [heavy_researcher] Making 3 LLM calls (this is the hot node)...")

        # Call 1: Find facts
        r1 = llm.invoke([
            SystemMessage(content="List 3 key facts about this topic. Be specific with numbers."),
            HumanMessage(content=state["setup_output"]),
        ])
        print(f"    Call 1 done: {r1.content[:60]}...")

        # Call 2: Find challenges
        r2 = llm.invoke([
            SystemMessage(content="List 3 challenges or risks for this topic. Be specific."),
            HumanMessage(content=state["setup_output"]),
        ])
        print(f"    Call 2 done: {r2.content[:60]}...")

        # Call 3: Combine
        r3 = llm.invoke([
            SystemMessage(content="Combine these facts and challenges into a 50-word summary."),
            HumanMessage(content=f"Facts:\n{r1.content}\n\nChallenges:\n{r2.content}"),
        ])
        print(f"    Call 3 done: {r3.content[:60]}...")

        return {"research_output": r3.content}

    def quick_formatter_node(state: HotNodeState) -> dict:
        """Fast node — formats the output. ~1 LLM call."""
        print("  [quick_formatter] Formatting final output...")
        response = llm.invoke([
            SystemMessage(content="Add a title and format this as a brief report. Under 80 words."),
            HumanMessage(content=state["research_output"]),
        ])
        print(f"    Done: {response.content[:80]}...")
        return {"final_output": response.content}

    graph = StateGraph(HotNodeState)
    graph.add_node("quick_setup", quick_setup_node)
    graph.add_node("heavy_researcher", heavy_researcher_node)
    graph.add_node("quick_formatter", quick_formatter_node)
    graph.set_entry_point("quick_setup")
    graph.add_edge("quick_setup", "heavy_researcher")
    graph.add_edge("heavy_researcher", "quick_formatter")
    graph.add_edge("quick_formatter", END)

    app = graph.compile()
    result = app.invoke(
        {
            "topic": "AI in healthcare",
            "setup_output": "",
            "research_output": "",
            "final_output": "",
        },
        {"run_name": "trace1_hot_node"},
    )

    print(f"\n  Phoenix: Look for 'trace1_hot_node' — heavy_researcher is 3x longer")
    print(f"  than quick_setup and quick_formatter because it makes 3 LLM calls.")


# ================================================================
# TRACE 2: THE LOOP THAT WON'T DIE
# ================================================================
# Evaluator-optimizer loop that runs multiple iterations.
# In Phoenix: you'll see repeated evaluate/optimize span pairs.
#
#   Graph: writer → evaluator → (score < 9 → optimizer → evaluator) → END
#   Expected: evaluator and optimizer spans repeat 3+ times

def run_trace_2_stuck_loop():
    """Trace 2: Evaluate-optimize loop runs multiple times."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n{'='*60}")
    print("  TRACE 2: THE LOOP THAT WON'T DIE")
    print(f"{'='*60}")
    print("  Graph: writer → evaluator ⇄ optimizer (loops until score >= 9 or 3 iterations)")
    print("  Watch: evaluator + optimizer spans repeat in Phoenix\n")

    llm = get_llm(temperature=0.7)

    class LoopState(TypedDict):
        task: str
        draft: str
        score: int
        feedback: str
        iteration: int

    def writer_node(state: LoopState) -> dict:
        """Writes the initial draft."""
        print("  [writer] Creating initial draft...")
        response = llm.invoke([
            SystemMessage(content="Write a 2-sentence product description. Be creative but brief."),
            HumanMessage(content=state["task"]),
        ])
        print(f"    Draft: {response.content[:80]}...")
        return {"draft": response.content, "iteration": 0}

    def evaluator_node(state: LoopState) -> dict:
        """Scores the draft — always gives 6-7 on first tries to force looping."""
        iteration = state.get("iteration", 0) + 1
        print(f"  [evaluator] Scoring draft (iteration {iteration})...")
        response = llm.invoke([
            SystemMessage(content=(
                "Score this product description 1-10 on creativity and clarity. "
                "Be strict — only score 9+ if it's truly exceptional. "
                "Output format: SCORE: <number>\nFEEDBACK: <what to improve>"
            )),
            HumanMessage(content=f"Draft:\n{state['draft']}"),
        ])
        score_match = re.search(r'SCORE:\s*(\d+)', response.content)
        score = int(score_match.group(1)) if score_match else 6
        score = min(10, max(1, score))
        feedback = response.content
        print(f"    Score: {score}/10 (iteration {iteration})")
        return {"score": score, "feedback": feedback, "iteration": iteration}

    def optimizer_node(state: LoopState) -> dict:
        """Rewrites the draft based on feedback."""
        print(f"  [optimizer] Rewriting based on feedback...")
        response = llm.invoke([
            SystemMessage(content=(
                "Improve this product description based on the feedback. "
                "Make it more creative and compelling. Keep it to 2 sentences."
            )),
            HumanMessage(content=(
                f"Current draft:\n{state['draft']}\n\n"
                f"Feedback:\n{state['feedback']}"
            )),
        ])
        print(f"    New draft: {response.content[:80]}...")
        return {"draft": response.content}

    def should_loop(state: LoopState) -> str:
        if state.get("score", 0) >= 9 or state.get("iteration", 0) >= 3:
            return "done"
        return "optimize"

    graph = StateGraph(LoopState)
    graph.add_node("writer", writer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("optimizer", optimizer_node)
    graph.set_entry_point("writer")
    graph.add_edge("writer", "evaluator")
    graph.add_conditional_edges("evaluator", should_loop, {
        "optimize": "optimizer",
        "done": END,
    })
    graph.add_edge("optimizer", "evaluator")

    app = graph.compile()
    result = app.invoke(
        {
            "task": "An AI-powered smart water bottle that tracks hydration",
            "draft": "",
            "score": 0,
            "feedback": "",
            "iteration": 0,
        },
        {"recursion_limit": 20, "run_name": "trace2_stuck_loop"},
    )

    print(f"\n  Final score: {result['score']}/10 after {result['iteration']} iterations")
    print(f"  Phoenix: Look for 'trace2_stuck_loop' — evaluator/optimizer spans repeat.")


# ================================================================
# TRACE 3: THE COST SPIKE
# ================================================================
# Each iteration appends to a growing context (no summarization).
# In Phoenix: token counts increase with each span.
#
#   Graph: researcher → accumulator → (loop 3 times) → END
#   Expected: each accumulator call uses more tokens than the last

def run_trace_3_cost_spike():
    """Trace 3: Context grows each iteration, token count spikes."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n{'='*60}")
    print("  TRACE 3: THE COST SPIKE")
    print(f"{'='*60}")
    print("  Graph: researcher → accumulator (loops 3x, context grows each time)")
    print("  Watch: token counts increase per span in Phoenix\n")

    llm = get_llm(temperature=0.5)

    class CostState(TypedDict):
        topic: str
        accumulated_context: str
        iteration: int
        final_output: str

    SUBTOPICS = ["current applications", "future predictions", "ethical concerns"]

    def researcher_node(state: CostState) -> dict:
        """Researches the next subtopic and APPENDS to growing context."""
        iteration = state.get("iteration", 0)
        subtopic = SUBTOPICS[iteration] if iteration < len(SUBTOPICS) else "summary"

        print(f"  [researcher] Iteration {iteration + 1}: researching '{subtopic}'...")

        # Each call includes ALL previous context (this is the cost spike!)
        context = state.get("accumulated_context", "")
        prompt = f"Previous research:\n{context}\n\n" if context else ""
        prompt += f"Now research: {state['topic']} — specifically about {subtopic}. Write 3-4 sentences."

        response = llm.invoke([
            SystemMessage(content="You are a researcher. Add new findings to the existing research."),
            HumanMessage(content=prompt),
        ])

        # Append (never summarize — this causes the cost spike!)
        new_context = context + f"\n\n--- {subtopic.upper()} ---\n{response.content}"
        print(f"    Context size: {len(context)} → {len(new_context)} chars")

        return {
            "accumulated_context": new_context,
            "iteration": iteration + 1,
        }

    def formatter_node(state: CostState) -> dict:
        """Formats the final output — receives the full bloated context."""
        print(f"  [formatter] Formatting (context: {len(state['accumulated_context'])} chars)...")
        response = llm.invoke([
            SystemMessage(content="Summarize all this research into 3 bullet points."),
            HumanMessage(content=state["accumulated_context"]),
        ])
        return {"final_output": response.content}

    def should_continue(state: CostState) -> str:
        if state.get("iteration", 0) >= 3:
            return "format"
        return "research"

    graph = StateGraph(CostState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("formatter", formatter_node)
    graph.set_entry_point("researcher")
    graph.add_conditional_edges("researcher", should_continue, {
        "research": "researcher",
        "format": "formatter",
    })
    graph.add_edge("formatter", END)

    app = graph.compile()
    result = app.invoke(
        {
            "topic": "AI in healthcare",
            "accumulated_context": "",
            "iteration": 0,
            "final_output": "",
        },
        {"recursion_limit": 20, "run_name": "trace3_cost_spike"},
    )

    print(f"\n  Phoenix: Look for 'trace3_cost_spike' — token counts grow each iteration.")
    print(f"  Fix: Add summarization middleware between iterations.")


# ================================================================
# TRACE 4: MULTI-AGENT RESEARCH (full pipeline)
# ================================================================
# The complete multi-agent workflow with tools, showing how a
# real production trace looks in Phoenix.
#
#   Graph: supervisor → researcher (tools) → analyst → judge → END
#   Expected: researcher span has child tool spans

def run_trace_4_multi_agent():
    """Trace 4: Full multi-agent pipeline with tool calls."""
    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, END, add_messages
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    print(f"\n{'='*60}")
    print("  TRACE 4: MULTI-AGENT RESEARCH PIPELINE")
    print(f"{'='*60}")
    print("  Graph: supervisor → researcher (tools) → analyst → judge → END")
    print("  Watch: researcher span has child tool spans in Phoenix\n")

    llm = get_llm(temperature=0.3)

    @tool
    def search_facts(topic: str) -> str:
        """Search for key facts about a topic."""
        time.sleep(0.1)  # Simulate API latency (visible in Phoenix)
        return (
            f"Facts about '{topic}': AI diagnostics achieve 94% accuracy in radiology. "
            "Drug discovery time reduced by 60%. Market projected at $45B by 2030. "
            "FDA approved 500+ AI medical devices."
        )

    @tool
    def search_statistics(topic: str) -> str:
        """Search for statistics and numbers about a topic."""
        time.sleep(0.1)  # Simulate API latency
        return (
            f"Statistics for '{topic}': Global AI healthcare market $15.1B (2024), "
            "projected $45.2B by 2030 (CAGR 20.9%). "
            "87% of healthcare orgs adopted or plan to adopt AI. "
            "AI reduces diagnostic errors by 30-40%."
        )

    tools_list = [search_facts, search_statistics]
    llm_with_tools = llm.bind_tools(tools_list)

    class ResearchState(TypedDict):
        messages: Annotated[list, add_messages]
        topic: str
        research: str
        analysis: str
        judgment: str

    def supervisor_node(state: ResearchState) -> dict:
        """Decomposes the research question."""
        print("  [supervisor] Decomposing question...")
        response = llm.invoke([
            SystemMessage(content="List 2 specific sub-questions to research this topic. Be brief."),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ])
        print(f"    Sub-questions: {response.content[:80]}...")
        return {"messages": [response]}

    def researcher_node(state: ResearchState) -> dict:
        """Gathers facts using tools — creates child tool spans."""
        print("  [researcher] Calling tools...")
        tool_node = ToolNode(tools_list)

        messages = [
            SystemMessage(content=(
                "Use search_facts AND search_statistics tools to research the topic. "
                "Call BOTH tools."
            )),
            HumanMessage(content=f"Research: {state['topic']}"),
        ]

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    [TOOL] {tc['name']}({tc['args']})")
            try:
                tool_result = tool_node.invoke({"messages": messages})
                messages.extend(tool_result["messages"])
                response = llm.invoke(messages)
            except Exception as e:
                print(f"    [WARN] Tool error: {e}")
                response = llm.invoke(messages)

        print(f"    Research: {response.content[:80]}...")
        return {"research": response.content, "messages": [response]}

    def analyst_node(state: ResearchState) -> dict:
        """Analyzes the research findings."""
        print("  [analyst] Analyzing findings...")
        response = llm.invoke([
            SystemMessage(content="Identify top 3 insights and 2 concerns. Under 80 words."),
            HumanMessage(content=f"Research:\n{state.get('research', '')}"),
        ])
        print(f"    Analysis: {response.content[:80]}...")
        return {"analysis": response.content, "messages": [response]}

    def judge_node(state: ResearchState) -> dict:
        """Final judgment — synthesizes everything."""
        print("  [judge] Synthesizing final output...")
        response = llm.invoke([
            SystemMessage(content="Give a balanced 3-sentence conclusion based on research and analysis."),
            HumanMessage(content=(
                f"Research:\n{state.get('research', '')}\n\n"
                f"Analysis:\n{state.get('analysis', '')}"
            )),
        ])
        print(f"    Judgment: {response.content[:80]}...")
        return {"judgment": response.content, "messages": [response]}

    graph = StateGraph(ResearchState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("judge", judge_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "judge")
    graph.add_edge("judge", END)

    app = graph.compile()
    result = app.invoke(
        {
            "messages": [],
            "topic": "AI in healthcare",
            "research": "",
            "analysis": "",
            "judgment": "",
        },
        {"run_name": "trace4_multi_agent"},
    )

    print(f"\n  Phoenix: Look for 4 node spans + child tool spans under 'researcher'.")
    print(f"  Compare duration/tokens: supervisor (light) vs researcher (heavy with tools).")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print()
    print("Example 15: Advanced Tracing with Phoenix — 4 Trace Patterns")
    print("=" * 60)
    print()

    # Launch Phoenix BEFORE any LangChain imports
    phoenix_session = setup_phoenix()
    if not phoenix_session:
        print("Phoenix setup failed. Exiting.")
        sys.exit(1)

    print()
    print("Running 4 workflows — each creates a separate trace in Phoenix:")
    print("  1. Hot Node — one span dominates latency")
    print("  2. Loop That Won't Die — evaluate/optimize repeats")
    print("  3. Cost Spike — token count grows each iteration")
    print("  4. Multi-Agent Research — supervisor → researcher (tools) → analyst → judge")
    print()

    # Run with 60s gaps between traces to avoid Groq rate limits.
    # Free tier: 6000 TPM. Each trace uses ~2000-4000 tokens.
    traces = [
        ("Trace 1: Hot Node", run_trace_1_hot_node),
        ("Trace 2: Stuck Loop", run_trace_2_stuck_loop),
        ("Trace 3: Cost Spike", run_trace_3_cost_spike),
        ("Trace 4: Multi-Agent", run_trace_4_multi_agent),
    ]

    for i, (name, fn) in enumerate(traces):
        if i > 0:
            wait = 60
            print(f"\n  Waiting {wait}s before next trace (Groq rate limit cooldown)...")
            time.sleep(wait)
        try:
            fn()
        except Exception as e:
            print(f"\n  [ERROR] {name} failed: {e}")
            print(f"  Waiting 60s and continuing...")
            time.sleep(60)

    print(f"\n{'='*60}")
    print("  ALL 4 TRACES COMPLETE — Check Phoenix Dashboard")
    print(f"{'='*60}")
    print("""
  Open http://localhost:6006 and look for 4 traces:

  Trace 1 (Hot Node):
    quick_setup → heavy_researcher → quick_formatter
    → heavy_researcher span is 3x longer (3 LLM calls)

  Trace 2 (Stuck Loop):
    writer → evaluator → optimizer → evaluator → optimizer → ...
    → evaluator/optimizer spans repeat 2-3 times

  Trace 3 (Cost Spike):
    researcher (x3) → formatter
    → each researcher span uses more tokens than the last

  Trace 4 (Multi-Agent):
    supervisor → researcher → analyst → judge
    → researcher has child tool spans (search_facts, search_statistics)
""")

    print("Phoenix dashboard: http://localhost:6006")
    input("\nPress Enter to exit...")
