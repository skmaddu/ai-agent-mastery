import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 3: Capstone — Multi-Agent Research System
====================================================
Difficulty: ⭐⭐⭐⭐ Advanced | Time: 5 hours

Complete solution combining MCP + A2A + LangGraph into a unified
multi-agent research system.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │                LangGraph Supervisor                   │
  │                                                       │
  │  START → [Planner] → [Researcher] → [Writer] → END  │
  │              │            │              │            │
  │              │        MCP Client      A2A Client      │
  │              │            │              │            │
  │              │     ┌──────▼──────┐ ┌────▼─────┐      │
  │              │     │ Research DB │ │ Writer   │      │
  │              │     │ MCP Server  │ │ A2A Agent│      │
  │              │     └─────────────┘ └──────────┘      │
  └──────────────────────────────────────────────────────┘

The system:
  1. Planner decomposes the topic into 2-3 sub-tasks (LLM)
  2. Researcher uses MCP to search/save research, LLM to generate findings
  3. Quality check loops researcher if fewer than 2 results (max 2 iterations)
  4. Writer A2A agent synthesizes findings into a structured report
  5. Budget tracker enforces $0.50 limit
  6. Phoenix-style timing spans at each node

Run: python week-07-mcp-a2a-synthesis/solutions/solution_03_capstone_synthesis.py
"""

import os
import json
import uuid
import asyncio
import threading
import time
from typing import TypedDict, Annotated, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from pydantic import BaseModel, Field


# ================================================================
# LLM Setup
# ================================================================

def get_llm(temperature=0.3):
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

try:
    from config.phoenix_config import setup_tracing
    setup_tracing()
    PHOENIX_AVAILABLE = True
except Exception:
    PHOENIX_AVAILABLE = False


# ================================================================
# TODO 1: Define the Research State (COMPLETED)
# ================================================================

class ResearchState(TypedDict):
    topic: str
    plan: list[str]
    research_results: list[dict]
    final_report: str
    current_step: str
    cost: dict  # {"total_tokens": 0, "estimated_cost": 0.0}
    errors: list[str]


# ================================================================
# TODO 9: Cost Tracking and Budget Enforcement (COMPLETED)
# ================================================================

BUDGET_LIMIT = 0.50  # $0.50 max per session


class BudgetTracker:
    """Track costs and enforce budget."""

    def __init__(self, limit: float = BUDGET_LIMIT):
        self.limit = limit
        self.total_tokens = 0
        self.total_cost = 0.0
        self.calls = []

    def check_budget(self) -> bool:
        """Returns True if we're within budget."""
        return self.total_cost < self.limit

    def log(self, description: str, tokens: int):
        """Log a call and update cost."""
        cost = tokens * 0.001 / 1000  # $0.001 per 1K tokens
        self.total_tokens += tokens
        self.total_cost += cost
        self.calls.append({"desc": description, "tokens": tokens, "cost": cost})

    def report(self) -> str:
        lines = [f"\n💰 Cost Report (Budget: ${self.limit:.2f})"]
        for c in self.calls:
            lines.append(f"  • {c['desc']}: {c['tokens']} tokens (${c['cost']:.4f})")
        lines.append(f"  Total: {self.total_tokens} tokens (${self.total_cost:.4f})")
        remaining = self.limit - self.total_cost
        lines.append(f"  Remaining budget: ${remaining:.4f}")
        return "\n".join(lines)


budget = BudgetTracker()

# Iteration counter for quality loop
_researcher_iterations = 0


# ================================================================
# TODO 2: Implement the Planner Node (COMPLETED)
# ================================================================

def planner_node(state: dict) -> dict:
    """Decompose the research topic into sub-tasks."""
    start_time = time.time()

    topic = state["topic"]
    errors = list(state.get("errors", []))

    if not budget.check_budget():
        errors.append("Budget exceeded in planner")
        duration = (time.time() - start_time) * 1000
        print(f"  📊 Span [planner]: {duration:.0f}ms (SKIPPED — over budget)")
        return {"plan": [], "current_step": "planning", "errors": errors}

    llm = get_llm(temperature=0)
    prompt = (
        f"You are a research planner. Decompose the following topic into 2-3 "
        f"specific research sub-tasks. Return ONLY a JSON array of strings, "
        f"nothing else.\n\n"
        f"Topic: {topic}\n\n"
        f"Example output: [\"Research the history of X\", \"Analyze current trends in X\", \"Identify future challenges for X\"]\n\n"
        f"Your JSON array:"
    )

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Estimate tokens and log cost
    tokens = int(len((prompt + response_text).split()) * 1.3)
    budget.log("planner", tokens)

    # Parse JSON array from response
    try:
        # Try to extract JSON array from response
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        plan = json.loads(response_text)
        if not isinstance(plan, list):
            plan = [str(plan)]
    except json.JSONDecodeError:
        # Fallback: split by newlines or use as single task
        plan = [line.strip().lstrip("0123456789.-) ") for line in response_text.split("\n") if line.strip()]
        if not plan:
            plan = [f"Research: {topic}"]

    duration = (time.time() - start_time) * 1000
    print(f"  📊 Span [planner]: {duration:.0f}ms")

    return {
        "plan": plan,
        "current_step": "planning",
        "cost": {"total_tokens": budget.total_tokens, "estimated_cost": budget.total_cost},
    }


# ================================================================
# TODO 3: Implement the MCP Research Tools (COMPLETED)
# ================================================================

async def mcp_search_research(query: str) -> str:
    """Search existing research via MCP."""
    try:
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

        server_params = StdioServerParameters(command="python", args=[server_path])
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("search_research", {"query": query})
                if result.content:
                    return "\n".join(c.text for c in result.content if hasattr(c, 'text'))
                return "No results found"
    except Exception as e:
        print(f"    ⚠️  MCP search failed (graceful fallback): {e}")
        return f"No existing research found for '{query}' (MCP unavailable)"


async def mcp_save_research(topic: str, summary: str) -> str:
    """Save new research via MCP."""
    try:
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

        server_params = StdioServerParameters(command="python", args=[server_path])
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "save_research",
                    {"topic": topic, "summary": summary, "sources": "[]"}
                )
                if result.content:
                    return "\n".join(c.text for c in result.content if hasattr(c, 'text'))
                return "Saved (no confirmation)"
    except Exception as e:
        print(f"    ⚠️  MCP save failed (graceful fallback): {e}")
        return f"Research saved locally (MCP unavailable)"


# ================================================================
# TODO 4: Implement the Researcher Node (COMPLETED)
# ================================================================

def researcher_node(state: dict) -> dict:
    """Research each sub-task using MCP tools and LLM."""
    global _researcher_iterations
    start_time = time.time()

    plan = state.get("plan", [])
    results = list(state.get("research_results", []))
    errors = list(state.get("errors", []))
    _researcher_iterations += 1

    for i, sub_task in enumerate(plan):
        print(f"    🔍 Researching sub-task {i + 1}/{len(plan)}: {sub_task[:60]}...")

        try:
            # Step A: Search existing research via MCP
            existing = asyncio.get_event_loop().run_until_complete(
                mcp_search_research(sub_task)
            )
            print(f"      MCP search: {existing[:80]}...")

            # Step B: Check budget before LLM call
            if not budget.check_budget():
                errors.append(f"Budget exceeded researching: {sub_task[:40]}")
                print(f"      ⚠️  Budget exceeded, skipping LLM call")
                continue

            # Step C: Generate new findings with LLM
            llm = get_llm(temperature=0)
            from langchain_core.messages import HumanMessage, SystemMessage
            prompt = (
                f"You are a thorough researcher. Research the following sub-task and provide "
                f"detailed findings in 100-200 words.\n\n"
                f"Sub-task: {sub_task}\n\n"
                f"Existing research context: {existing[:300]}\n\n"
                f"Provide your findings:"
            )
            response = llm.invoke([
                SystemMessage(content="You are a research assistant. Provide factual, well-structured findings."),
                HumanMessage(content=prompt),
            ])
            findings = response.content.strip()

            # Track cost
            tokens = int(len((prompt + findings).split()) * 1.3)
            budget.log(f"researcher[{sub_task[:30]}]", tokens)

            # Step D: Save findings via MCP
            save_result = asyncio.get_event_loop().run_until_complete(
                mcp_save_research(sub_task, findings)
            )
            print(f"      MCP save: {save_result[:80]}...")

            results.append({
                "sub_task": sub_task,
                "findings": findings,
                "existing_research": existing[:200],
            })

        except Exception as e:
            error_msg = f"Error on sub-task '{sub_task[:40]}': {e}"
            errors.append(error_msg)
            print(f"      ❌ {error_msg}")
            continue

    duration = (time.time() - start_time) * 1000
    print(f"  📊 Span [researcher]: {duration:.0f}ms")

    return {
        "research_results": results,
        "current_step": "researching",
        "cost": {"total_tokens": budget.total_tokens, "estimated_cost": budget.total_cost},
        "errors": errors,
    }


# ================================================================
# TODO 8: Quality Check — Conditional Edge (COMPLETED)
# ================================================================

def check_research_quality(state: dict) -> str:
    """Check if research quality is sufficient to proceed to writer."""
    results = state.get("research_results", [])
    if len(results) < 2 and _researcher_iterations < 2:
        print(f"    🔄 Quality check: only {len(results)} results, looping back (iteration {_researcher_iterations})")
        return "researcher"
    print(f"    ✅ Quality check: {len(results)} results, proceeding to writer")
    return "writer"


# ================================================================
# TODO 6: Create the Writer A2A Agent (COMPLETED)
# ================================================================

def create_writer_a2a_server(port: int = 8011):
    """Create a Writer Agent A2A server."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Writer Agent A2A")

    agent_card = {
        "name": "Writer Agent",
        "description": "Synthesizes research findings into well-structured reports.",
        "url": f"http://localhost:{port}",
        "version": "1.0.0",
        "skills": [
            {
                "id": "write-report",
                "name": "Report Writing",
                "description": "Write structured reports from research notes",
                "tags": ["writing", "reports", "synthesis"],
            }
        ],
        "capabilities": {"streaming": False, "pushNotifications": False},
    }

    @app.get("/.well-known/agent.json")
    async def get_card():
        return agent_card

    @app.post("/tasks/send")
    async def handle_task(request: Request):
        body = await request.json()
        message_parts = body.get("message", {}).get("parts", [])
        user_text = " ".join(p.get("text", "") for p in message_parts if p.get("text"))

        try:
            llm = get_llm(temperature=0.7)
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=(
                    "You are an expert report writer. Transform the provided research notes "
                    "into a well-structured research report. Use clear sections with headers. "
                    "Include an executive summary, main findings, and conclusion."
                )),
                HumanMessage(content=user_text),
            ])

            return JSONResponse(content={
                "id": body.get("id", str(uuid.uuid4())),
                "status": {"state": "completed"},
                "artifacts": [
                    {
                        "name": "report",
                        "parts": [{"type": "text", "text": response.content}],
                        "mime_type": "text/markdown",
                    }
                ],
            })
        except Exception as e:
            return JSONResponse(content={
                "id": body.get("id", str(uuid.uuid4())),
                "status": {"state": "failed", "message": str(e)},
                "artifacts": [],
            })

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "Writer Agent"}

    return app


# ================================================================
# TODO 5: Implement the Writer Node (COMPLETED)
# ================================================================

def writer_node(state: dict) -> dict:
    """Send research to Writer Agent via A2A for synthesis."""
    start_time = time.time()

    results = state.get("research_results", [])
    errors = list(state.get("errors", []))

    # Format research notes for the writer
    research_notes = "Research findings to synthesize into a report:\n\n"
    for i, r in enumerate(results, 1):
        research_notes += f"## Sub-task {i}: {r['sub_task']}\n{r['findings']}\n\n"

    report = ""

    # Try A2A first
    try:
        import httpx

        a2a_payload = {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": research_notes}],
            },
        }

        response = httpx.post(
            "http://localhost:8011/tasks/send",
            json=a2a_payload,
            timeout=60.0,
        )
        data = response.json()

        if data.get("status", {}).get("state") == "completed":
            artifacts = data.get("artifacts", [])
            if artifacts:
                parts = artifacts[0].get("parts", [])
                if parts:
                    report = parts[0].get("text", "")
                    print(f"    ✅ Writer A2A agent produced report ({len(report)} chars)")

                    # Estimate cost for the A2A LLM call
                    tokens = int(len((research_notes + report).split()) * 1.3)
                    budget.log("writer_a2a", tokens)
        else:
            raise Exception(f"A2A task failed: {data.get('status', {}).get('message', 'unknown')}")

    except Exception as e:
        print(f"    ⚠️  A2A failed ({e}), falling back to direct LLM call...")

        # Fallback: direct LLM call
        if budget.check_budget():
            llm = get_llm(temperature=0.7)
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=(
                    "You are an expert report writer. Transform the provided research notes "
                    "into a well-structured research report with an executive summary, "
                    "main findings, and conclusion."
                )),
                HumanMessage(content=research_notes),
            ])
            report = response.content.strip()
            tokens = int(len((research_notes + report).split()) * 1.3)
            budget.log("writer_fallback", tokens)
            print(f"    ✅ Fallback LLM produced report ({len(report)} chars)")
        else:
            errors.append("Budget exceeded in writer node")
            report = "Report generation skipped — budget exceeded."

    duration = (time.time() - start_time) * 1000
    print(f"  📊 Span [writer]: {duration:.0f}ms")

    return {
        "final_report": report,
        "current_step": "writing",
        "cost": {"total_tokens": budget.total_tokens, "estimated_cost": budget.total_cost},
        "errors": errors,
    }


# ================================================================
# TODO 7: Wire the LangGraph (COMPLETED)
# ================================================================

def build_research_graph():
    """Build the LangGraph research pipeline."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")

    # TODO 8: Conditional edge for quality loop
    graph.add_conditional_edges(
        "researcher",
        check_research_quality,
        {"researcher": "researcher", "writer": "writer"},
    )

    graph.add_edge("writer", END)

    return graph.compile()


# ================================================================
# TODO 11 & 12: Demo Runner (COMPLETED)
# ================================================================

async def run_demo():
    """Run the complete multi-agent research system."""
    global _researcher_iterations

    print("=" * 70)
    print("Capstone: Multi-Agent Research System")
    print("MCP (tools) + A2A (agents) + LangGraph (orchestration)")
    print("=" * 70)

    # Step 1: Start Writer A2A server in background thread
    print("\n  🚀 Starting Writer A2A server on port 8011...")
    import uvicorn

    writer_app = create_writer_a2a_server(port=8011)
    writer_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": writer_app, "host": "localhost", "port": 8011, "log_level": "warning"},
        daemon=True,
    )
    writer_thread.start()
    await asyncio.sleep(2)  # Wait for server to start
    print("  ✅ Writer A2A server running")

    # Step 2: Build graph
    print("\n  🔧 Building LangGraph research pipeline...")
    app = build_research_graph()
    print("  ✅ Graph compiled: planner → researcher → (quality check) → writer")

    # Step 3: Run the graph
    topic = "The impact of AI agents on software development in 2026"
    print(f"\n  🎯 Topic: {topic}")
    print("  " + "─" * 60)

    _researcher_iterations = 0  # Reset iteration counter

    initial_state = {
        "topic": topic,
        "plan": [],
        "research_results": [],
        "final_report": "",
        "current_step": "start",
        "cost": {"total_tokens": 0, "estimated_cost": 0.0},
        "errors": [],
    }

    print("\n  ▶ Running graph...\n")
    result = app.invoke(initial_state)

    # Step 4: Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Plan
    print(f"\n  📋 Plan ({len(result.get('plan', []))} sub-tasks):")
    for i, task in enumerate(result.get("plan", []), 1):
        print(f"    {i}. {task}")

    # Research findings
    print(f"\n  🔬 Research Findings ({len(result.get('research_results', []))} results):")
    for i, r in enumerate(result.get("research_results", []), 1):
        findings_preview = r.get("findings", "")[:150]
        print(f"    [{i}] {r.get('sub_task', 'Unknown')}")
        print(f"        {findings_preview}...")

    # Final report
    report = result.get("final_report", "No report generated")
    print(f"\n  📰 Final Report ({len(report)} chars):")
    print("  " + "─" * 50)
    for line in report[:500].split("\n"):
        print(f"    {line}")
    if len(report) > 500:
        print("    ...")

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n  ⚠️  Errors ({len(errors)}):")
        for err in errors:
            print(f"    • {err}")

    # Step 5: Cost report
    print(budget.report())

    print("\n" + "=" * 70)
    print("Capstone complete!")
    print("=" * 70)


# ================================================================
# Main Entry Point
# ================================================================

if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n  ⏹️  Interrupted")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Install: pip install mcp fastapi uvicorn httpx langgraph langchain-groq")
