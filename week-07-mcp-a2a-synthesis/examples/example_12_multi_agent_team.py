import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 12: Multi-Agent Team — Planner + Researcher + Writer
=============================================================
Topic 12 — Building a coordinated team of specialized agents
using LangGraph, with MCP-backed research capabilities.

The BIG IDEA (Feynman):
  A research team works like a newspaper: the editor (Planner) decides
  what stories to cover, reporters (Researchers) go out and gather facts,
  and writers (Writers) turn those facts into articles.

  No single person does everything. The editor doesn't do interviews,
  the reporter doesn't write the final article, and the writer doesn't
  decide what stories to pursue. Each agent has a clear role, and they
  coordinate through a shared workflow.

  This example builds exactly that:
    - PlannerAgent: decomposes a query into 2-3 research sub-tasks
    - ResearcherAgent: investigates each sub-task using MCP tools
    - WriterAgent: synthesizes all findings into a structured report
    - TeamSupervisor: LangGraph StateGraph that orchestrates the flow

  Flow: START -> plan -> research (for each sub-task) -> write -> END

Previously covered:
  - MCP client/server (examples 03-05)
  - LangGraph orchestration (Week 2-6)
  - Multi-agent patterns (Week 4)

Run: python week-07-mcp-a2a-synthesis/examples/example_12_multi_agent_team.py
"""

import os
import json
import asyncio
import time
from typing import TypedDict, Any
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# LLM Setup
# ================================================================
# Each agent uses a different temperature depending on its role.
# The Planner needs to be precise (temp=0), the Writer can be
# creative (temp=0.7), and the Researcher sits in between.

def get_llm(temperature=0.3):
    """Create LLM instance based on provider setting."""
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
# Phoenix Tracing (optional)
# ================================================================
# Phoenix traces every LLM call and MCP interaction, giving you
# a visual timeline of how the team coordinates. Invaluable for
# debugging multi-agent workflows.

try:
    from config.phoenix_config import setup_tracing
    setup_tracing()
    PHOENIX_AVAILABLE = True
except Exception:
    PHOENIX_AVAILABLE = False


# ================================================================
# PART 1: TEAM STATE DEFINITION
# ================================================================
# In a multi-agent system, state is the SHARED WHITEBOARD. Every
# agent reads from it and writes to it. The state flows through
# the graph like a baton in a relay race.
#
# Think of it like a project folder on a shared drive:
#   - The Planner writes the task list
#   - The Researcher adds findings for each task
#   - The Writer reads everything and produces the final report

print("=" * 70)
print("PART 1: Team State & Agent Definitions")
print("=" * 70)
print("""
  TeamState is the shared memory for all agents:
    - messages: conversation history (for context)
    - plan: list of sub-tasks from the Planner
    - research_results: findings from the Researcher (one per sub-task)
    - final_report: the Writer's synthesized output
    - cost_tracker: running total of tokens and estimated cost

  Each agent reads what it needs and writes its output back to state.
""")


class TeamState(TypedDict):
    """Shared state flowing through the multi-agent team graph.

    WHY TypedDict?
    LangGraph uses TypedDict for state because it's simple, type-checked,
    and serializable. Each key is a "channel" that agents read/write.
    """
    messages: list[dict]              # Conversation history
    plan: list[str]                   # Sub-tasks from PlannerAgent
    research_results: list[dict]      # Findings from ResearcherAgent
    final_report: str                 # Synthesized report from WriterAgent
    cost_tracker: dict                # {total_tokens, estimated_cost}


# ================================================================
# PART 2: PLANNER AGENT
# ================================================================
# The Planner is like the newspaper editor. Given a broad topic,
# it breaks it down into specific, researchable questions.
#
# WHY temperature=0?
# Planning needs to be DETERMINISTIC. We want the same query to
# produce the same plan every time. Creativity here would mean
# inconsistent results — bad for reproducibility.

print("\n" + "=" * 70)
print("PART 2: PlannerAgent — Decompose Query into Sub-Tasks")
print("=" * 70)


async def planner_node(state: TeamState) -> dict:
    """PlannerAgent: decompose a user query into 2-3 research sub-tasks.

    This agent uses the LLM with temperature=0 to produce a deterministic
    plan. The output is a JSON list of sub-task strings.
    """
    print("\n  [PlannerAgent] Analyzing query and creating research plan...")

    # Extract the user query from messages
    user_query = ""
    for msg in state.get("messages", []):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        print("  [PlannerAgent] WARNING: No user query found in messages")
        return {"plan": ["Research the given topic broadly"]}

    # Use LLM with temperature=0 for deterministic planning
    llm = get_llm(temperature=0)

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research planner. Given a topic, decompose it into "
            "exactly 2-3 specific, focused research sub-tasks. Each sub-task "
            "should be a clear question or investigation area.\n\n"
            "Respond ONLY with a JSON array of strings. Example:\n"
            '["What is the current state of X?", "What are the main challenges of X?", "What are future trends in X?"]\n\n'
            "No explanation, no markdown — just the JSON array."
        )),
        HumanMessage(content=f"Topic to research: {user_query}"),
    ])

    # Track tokens
    cost = state.get("cost_tracker", {"total_tokens": 0, "estimated_cost": 0.0})
    tokens_used = getattr(response, "usage_metadata", {})
    if isinstance(tokens_used, dict):
        cost["total_tokens"] += tokens_used.get("total_tokens", 0)
        cost["estimated_cost"] += tokens_used.get("total_tokens", 0) * 0.000001
    else:
        # Estimate from response length if metadata unavailable
        estimated = len(response.content) // 4 + len(user_query) // 4
        cost["total_tokens"] += estimated
        cost["estimated_cost"] += estimated * 0.000001

    # Parse the JSON response
    response_text = response.content.strip()
    # Strip markdown code fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        )

    try:
        plan = json.loads(response_text)
        if not isinstance(plan, list):
            plan = [str(plan)]
    except json.JSONDecodeError:
        # Fallback: split by newlines or use as single task
        print("  [PlannerAgent] Could not parse JSON, falling back to text split")
        plan = [line.strip("- ").strip() for line in response_text.split("\n") if line.strip()]
        if not plan:
            plan = [f"Research: {user_query}"]

    # Limit to 3 sub-tasks max
    plan = plan[:3]

    print(f"  [PlannerAgent] Created {len(plan)} sub-tasks:")
    for i, task in enumerate(plan, 1):
        print(f"    {i}. {task}")

    return {"plan": plan, "cost_tracker": cost}


# ================================================================
# PART 3: RESEARCHER AGENT
# ================================================================
# The Researcher is like a newspaper reporter. For each sub-task,
# it goes out and gathers information. It tries to use MCP to access
# the research history database, and falls back to LLM-only if MCP
# is unavailable.
#
# WHY MCP FALLBACK?
# In production, external services fail. A good agent degrades
# gracefully — if the database is down, it still produces results
# using its built-in knowledge. This is the "circuit breaker" pattern.

print("\n" + "=" * 70)
print("PART 3: ResearcherAgent — Investigate Sub-Tasks via MCP")
print("=" * 70)
print("""
  The Researcher tries to connect to the research_history MCP server.
  If connected, it saves and retrieves findings from the database.
  If MCP fails, it falls back to LLM-only research.

  MCP connection pattern:
    1. Launch research_history_server.py via stdio
    2. Initialize session
    3. Call tools: save_research, search_research
    4. Close connection when done
""")


async def research_with_mcp(sub_task: str) -> dict:
    """Attempt research using the MCP research history server.

    Returns a dict with 'source' ('mcp' or 'llm') and 'findings'.
    """
    from mcp.client.stdio import stdio_client
    from mcp.client.session import ClientSession
    try:
        from mcp.client.stdio import StdioServerParameters
    except ImportError:
        from mcp import StdioServerParameters

    server_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mcp-servers", "research_history_server.py"
    )
    server_path = os.path.normpath(server_path)

    if not os.path.exists(server_path):
        raise FileNotFoundError(f"MCP server not found: {server_path}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Search for existing research on this sub-task
            search_result = await session.call_tool(
                "search_research",
                {"query": sub_task}
            )
            existing = ""
            if search_result.content:
                existing = "\n".join(
                    c.text for c in search_result.content if hasattr(c, "text")
                )

            # Use LLM to research the sub-task (informed by existing data)
            llm = get_llm(temperature=0.3)
            from langchain_core.messages import HumanMessage, SystemMessage

            context_note = ""
            if existing and "0 results" not in existing.lower():
                context_note = f"\n\nExisting research found in database:\n{existing[:500]}"

            response = llm.invoke([
                SystemMessage(content=(
                    "You are a thorough researcher. Given a research task, provide "
                    "detailed findings in 2-3 paragraphs. Include specific facts, "
                    "data points, or trends where possible. Be concise but comprehensive."
                    f"{context_note}"
                )),
                HumanMessage(content=f"Research task: {sub_task}"),
            ])

            findings = response.content

            # Save findings to the MCP database for future reference
            try:
                await session.call_tool(
                    "save_research",
                    {
                        "topic": sub_task,
                        "summary": findings[:1000],
                        "sources": '["llm-research"]',
                    }
                )
            except Exception:
                pass  # Non-critical: saving to DB is best-effort

            return {
                "source": "mcp",
                "sub_task": sub_task,
                "findings": findings,
                "tokens_estimate": len(findings) // 4 + len(sub_task) // 4,
            }


async def research_llm_only(sub_task: str) -> dict:
    """Fallback: research using LLM only (no MCP)."""
    llm = get_llm(temperature=0.3)
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([
        SystemMessage(content=(
            "You are a thorough researcher. Given a research task, provide "
            "detailed findings in 2-3 paragraphs. Include specific facts, "
            "data points, or trends where possible. Be concise but comprehensive."
        )),
        HumanMessage(content=f"Research task: {sub_task}"),
    ])

    return {
        "source": "llm-only",
        "sub_task": sub_task,
        "findings": response.content,
        "tokens_estimate": len(response.content) // 4 + len(sub_task) // 4,
    }


async def researcher_node(state: TeamState) -> dict:
    """ResearcherAgent: investigate each sub-task from the plan.

    For each sub-task:
      1. Try MCP-backed research (database + LLM)
      2. If MCP fails, fall back to LLM-only research
      3. Continue with remaining sub-tasks even if one fails

    This is the "graceful degradation" pattern: the team keeps working
    even when individual components fail.
    """
    print("\n  [ResearcherAgent] Starting research on all sub-tasks...")

    plan = state.get("plan", [])
    cost = state.get("cost_tracker", {"total_tokens": 0, "estimated_cost": 0.0})
    results = []

    for i, sub_task in enumerate(plan, 1):
        print(f"\n  [ResearcherAgent] Sub-task {i}/{len(plan)}: {sub_task[:80]}...")

        try:
            # Try MCP-backed research first
            result = await research_with_mcp(sub_task)
            print(f"    Source: MCP (database-backed)")
        except Exception as e:
            # MCP failed — fall back to LLM-only
            print(f"    MCP unavailable ({type(e).__name__}), falling back to LLM-only")
            try:
                result = await research_llm_only(sub_task)
                print(f"    Source: LLM-only (fallback)")
            except Exception as e2:
                # Even LLM failed — record the error but continue
                print(f"    ERROR: Research failed completely: {e2}")
                result = {
                    "source": "error",
                    "sub_task": sub_task,
                    "findings": f"Research failed: {e2}",
                    "tokens_estimate": 0,
                }

        results.append(result)

        # Update cost tracker
        tokens = result.get("tokens_estimate", 0)
        cost["total_tokens"] += tokens
        cost["estimated_cost"] += tokens * 0.000001

        # Show preview of findings
        findings_preview = result["findings"][:150].replace("\n", " ")
        print(f"    Preview: {findings_preview}...")

    print(f"\n  [ResearcherAgent] Completed {len(results)} research tasks")

    return {"research_results": results, "cost_tracker": cost}


# ================================================================
# PART 4: WRITER AGENT
# ================================================================
# The Writer is like the newspaper writer. It takes all the raw
# research findings and crafts them into a polished, structured
# report. It uses temperature=0.7 for creative, engaging prose.
#
# WHY temperature=0.7?
# Writing is inherently creative. A bit of randomness makes the
# output more natural and varied. Too high (>0.9) would make it
# incoherent; too low (<0.3) would make it robotic and repetitive.

print("\n" + "=" * 70)
print("PART 4: WriterAgent — Synthesize Research into Report")
print("=" * 70)


async def writer_node(state: TeamState) -> dict:
    """WriterAgent: synthesize all research findings into a structured report.

    Takes the research_results from state, passes them to an LLM with
    creative temperature, and produces a report with:
      - Introduction
      - Key sections (one per sub-task)
      - Key points / highlights
      - Conclusion
    """
    print("\n  [WriterAgent] Synthesizing research into final report...")

    research_results = state.get("research_results", [])
    cost = state.get("cost_tracker", {"total_tokens": 0, "estimated_cost": 0.0})

    if not research_results:
        return {
            "final_report": "No research findings available to write about.",
            "cost_tracker": cost,
        }

    # Format all findings for the writer
    findings_text = ""
    for i, result in enumerate(research_results, 1):
        findings_text += f"\n--- Research Finding {i} ---\n"
        findings_text += f"Task: {result.get('sub_task', 'Unknown')}\n"
        findings_text += f"Source: {result.get('source', 'Unknown')}\n"
        findings_text += f"Findings:\n{result.get('findings', 'No findings')}\n"

    # Use LLM with temperature=0.7 for creative writing
    llm = get_llm(temperature=0.7)
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert technical writer. Given research findings on "
            "multiple sub-topics, synthesize them into a well-structured report.\n\n"
            "Your report MUST include:\n"
            "1. A brief Introduction (2-3 sentences setting context)\n"
            "2. Main sections — one for each research finding, with a clear heading\n"
            "3. Key Points — a bulleted list of 3-5 most important takeaways\n"
            "4. Conclusion — 2-3 sentences summarizing the overall picture\n\n"
            "Write in a professional but accessible style. Use markdown formatting."
        )),
        HumanMessage(content=f"Research findings to synthesize:\n{findings_text}"),
    ])

    final_report = response.content

    # Update cost tracker
    tokens_used = getattr(response, "usage_metadata", {})
    if isinstance(tokens_used, dict):
        cost["total_tokens"] += tokens_used.get("total_tokens", 0)
        cost["estimated_cost"] += tokens_used.get("total_tokens", 0) * 0.000001
    else:
        estimated = len(final_report) // 4 + len(findings_text) // 4
        cost["total_tokens"] += estimated
        cost["estimated_cost"] += estimated * 0.000001

    print(f"  [WriterAgent] Report generated ({len(final_report)} chars)")

    return {"final_report": final_report, "cost_tracker": cost}


# ================================================================
# PART 5: TEAM SUPERVISOR (LangGraph StateGraph)
# ================================================================
# The Supervisor is the overall workflow coordinator. It doesn't do
# any work itself — it just defines the ORDER in which agents run.
#
# Think of it as the production schedule at a newspaper:
#   1. Morning meeting: Editor assigns stories (Planner)
#   2. Daytime: Reporters gather facts (Researcher)
#   3. Evening: Writers produce articles (Writer)
#   4. Done: Paper goes to print (END)
#
# LangGraph makes this explicit as a graph:
#   START -> plan -> research -> write -> END

print("\n" + "=" * 70)
print("PART 5: Team Supervisor (LangGraph Orchestration)")
print("=" * 70)
print("""
  The supervisor is a LangGraph StateGraph with 3 nodes:

    START
      |
      v
    [plan] ---- PlannerAgent (decompose query)
      |
      v
    [research] - ResearcherAgent (investigate sub-tasks)
      |
      v
    [write] ---- WriterAgent (synthesize report)
      |
      v
     END

  State flows through each node, accumulating results.
""")


def build_team_graph():
    """Build the multi-agent team as a LangGraph StateGraph.

    WHY LangGraph for orchestration?
    LangGraph gives us:
      - Explicit, visible workflow (not hidden in function calls)
      - State management (each node reads/writes shared state)
      - Checkpointing (resume from any point if something fails)
      - Tracing (Phoenix shows every node execution)

    It's like a flowchart that actually RUNS.
    """
    from langgraph.graph import StateGraph, START, END

    # Create the graph with our TeamState schema
    graph = StateGraph(TeamState)

    # Add nodes — each is an async function that takes state and returns updates
    graph.add_node("plan", planner_node)
    graph.add_node("research", researcher_node)
    graph.add_node("write", writer_node)

    # Define edges — the order of execution
    # START -> plan: first, create the research plan
    graph.add_edge(START, "plan")

    # plan -> research: then, investigate each sub-task
    graph.add_edge("plan", "research")

    # research -> write: finally, synthesize into a report
    graph.add_edge("research", "write")

    # write -> END: we're done
    graph.add_edge("write", END)

    # Compile the graph into a runnable
    return graph.compile()


# ================================================================
# PART 6: DEMO — Run the Team
# ================================================================

print("\n" + "=" * 70)
print("PART 6: Running the Multi-Agent Team")
print("=" * 70)


async def run_team_demo():
    """Run the multi-agent team on a sample research query."""

    query = "Research the impact of AI on education in 2026"

    print(f"\n  Query: \"{query}\"")
    print("  " + "-" * 60)

    # Build the team graph
    team = build_team_graph()

    # Initialize the state with the user query
    initial_state: TeamState = {
        "messages": [{"role": "user", "content": query}],
        "plan": [],
        "research_results": [],
        "final_report": "",
        "cost_tracker": {"total_tokens": 0, "estimated_cost": 0.0},
    }

    # Run the team
    start_time = time.time()

    try:
        result = await team.ainvoke(initial_state)
    except Exception as e:
        print(f"\n  Team execution failed: {type(e).__name__}: {e}")
        print("\n  This usually means LLM API keys are not configured.")
        print("  Set GROQ_API_KEY or OPENAI_API_KEY in config/.env")
        return

    elapsed = time.time() - start_time

    # ── Display results ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Plan
    print("\n  PLAN (from PlannerAgent):")
    for i, task in enumerate(result.get("plan", []), 1):
        print(f"    {i}. {task}")

    # Research results
    print("\n  RESEARCH FINDINGS (from ResearcherAgent):")
    for i, res in enumerate(result.get("research_results", []), 1):
        source = res.get("source", "unknown")
        findings = res.get("findings", "No findings")[:200].replace("\n", " ")
        print(f"\n    Finding {i} [source: {source}]:")
        print(f"    {findings}...")

    # Final report (truncated to 500 chars for display)
    print("\n  FINAL REPORT (from WriterAgent):")
    print("  " + "-" * 60)
    report = result.get("final_report", "No report generated")
    report_display = report[:500]
    for line in report_display.split("\n"):
        print(f"  {line}")
    if len(report) > 500:
        print(f"\n  ... [truncated, full report is {len(report)} chars]")

    # Cost summary
    print("\n  " + "-" * 60)
    cost = result.get("cost_tracker", {})
    print(f"  COST SUMMARY:")
    print(f"    Total tokens (estimated): {cost.get('total_tokens', 0):,}")
    print(f"    Estimated cost: ${cost.get('estimated_cost', 0):.6f}")
    print(f"    Elapsed time: {elapsed:.1f}s")


# ================================================================
# KEY TAKEAWAYS
# ================================================================

print("""
Key Takeaways:
  1. DECOMPOSITION: The Planner breaks complex queries into manageable pieces
  2. SPECIALIZATION: Each agent has one job and does it well
  3. GRACEFUL DEGRADATION: If MCP fails, the Researcher falls back to LLM-only
  4. SHARED STATE: All agents read/write to the same TeamState
  5. COST TRACKING: Monitor token usage across the entire team
  6. ORCHESTRATION: LangGraph makes the workflow explicit and traceable
""")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  WEEK 7, EXAMPLE 12: MULTI-AGENT TEAM")
    print("  Planner + Researcher + Writer with MCP Integration")
    print("=" * 70)

    if PHOENIX_AVAILABLE:
        print("\n  Phoenix tracing: ENABLED (check localhost:6006)")
    else:
        print("\n  Phoenix tracing: not available (install arize-phoenix for traces)")

    try:
        asyncio.run(run_team_demo())
    except KeyboardInterrupt:
        print("\n  Demo interrupted")
    except Exception as e:
        print(f"\n  Demo error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Install: pip install langgraph mcp langchain-groq langchain-openai python-dotenv")

    print("\n  Example 12 complete!")
