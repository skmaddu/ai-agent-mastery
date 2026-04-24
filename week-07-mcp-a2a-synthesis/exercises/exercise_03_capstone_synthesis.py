import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 3: Capstone — Multi-Agent Research System (MCP + A2A + LangGraph)
============================================================================
Difficulty: ⭐⭐⭐⭐ Advanced | Time: 5 hours

Task:
Build a complete multi-agent research system that combines:
  - MCP for tool access (research database)
  - A2A for inter-agent communication (researcher ↔ writer)
  - LangGraph for orchestration (planner → researcher → writer)
  - Middleware for logging, cost tracking, and safety

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

The system should:
  1. Accept a research topic from the user
  2. Planner decomposes it into 2-3 sub-tasks
  3. Researcher uses MCP to search existing research and save new findings
  4. Writer (A2A agent) synthesizes findings into a report
  5. Track costs and enforce a $0.50 budget
  6. Trace all operations with Phoenix spans

Instructions:
  Complete the 12 TODOs below.

Hints:
  - Study example_05 for MCP + LangGraph integration
  - Study example_08 for A2A server/client
  - Study example_09 for the hybrid pattern
  - Study example_12 for the multi-agent team structure

Run: python week-07-mcp-a2a-synthesis/exercises/exercise_03_capstone_synthesis.py
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
# TODO 1: Define the Research State (TypedDict for LangGraph)
# ================================================================
# Create a TypedDict with:
#   - topic: str — the research topic
#   - plan: list[str] — list of sub-tasks from the planner
#   - research_results: list[dict] — findings from the researcher
#   - final_report: str — the writer's output
#   - current_step: str — tracks which step we're on
#   - cost: dict — {"total_tokens": 0, "estimated_cost": 0.0}
#   - errors: list[str] — any errors encountered
#
# Hint: from typing import TypedDict
#       class ResearchState(TypedDict):
#           topic: str
#           ...

# --- YOUR CODE HERE ---
# class ResearchState(TypedDict):
#     ...
# --- END YOUR CODE ---


# ================================================================
# TODO 2: Implement the Planner Node
# ================================================================
# Create a function that:
#   1. Takes the state (ResearchState)
#   2. Uses the LLM to decompose the topic into 2-3 research sub-tasks
#   3. Returns {"plan": [...sub-tasks...], "current_step": "planning"}
#
# The LLM prompt should ask for a JSON list of sub-task strings.
# Example output: ["Research history of topic", "Find recent developments", "Identify key challenges"]
#
# Hint: Use temperature=0 for deterministic planning

def planner_node(state: dict) -> dict:
    """Decompose the research topic into sub-tasks."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 3: Implement the MCP Research Tool
# ================================================================
# Create an async function that:
#   1. Connects to the research_history_server.py via MCP
#   2. Searches for existing research on the topic
#   3. Returns any found results
#
# If MCP connection fails, fall back to returning an empty result.
#
# Hint: Use the pattern from example_05:
#   from mcp.client.stdio import stdio_client
#   from mcp.client.session import ClientSession

async def mcp_search_research(query: str) -> str:
    """Search existing research via MCP."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


async def mcp_save_research(topic: str, summary: str) -> str:
    """Save new research via MCP."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 4: Implement the Researcher Node
# ================================================================
# Create a function that:
#   1. Takes the state
#   2. For each sub-task in the plan:
#      a. Search existing research via MCP (TODO 3)
#      b. Use the LLM to generate research findings
#      c. Save findings via MCP (TODO 3)
#   3. Returns {"research_results": [...], "current_step": "researching"}
#
# Handle errors: if one sub-task fails, continue with others.

def researcher_node(state: dict) -> dict:
    """Research each sub-task using MCP tools and LLM."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 5: Implement the Writer Node
# ================================================================
# Create a function that:
#   1. Takes the state (with research_results populated)
#   2. Sends all research to the Writer A2A agent (see TODO 6 for server)
#   3. Returns {"final_report": "...", "current_step": "writing"}
#
# If A2A fails, fall back to using the LLM directly.
#
# Hint: Use httpx to POST to http://localhost:8011 with tasks/send

def writer_node(state: dict) -> dict:
    """Send research to Writer Agent via A2A for synthesis."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 6: Create the Writer A2A Agent
# ================================================================
# Create a FastAPI app that:
#   1. Serves an Agent Card at /.well-known/agent.json
#   2. Handles tasks/send — takes research notes, produces a report
#   3. Uses LLM with temperature=0.7 for creative writing
#
# Hint: Study example_08 and example_09 for the server pattern.

def create_writer_a2a_server(port: int = 8011):
    """Create a Writer Agent A2A server."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 7: Wire the LangGraph
# ================================================================
# Create a StateGraph with:
#   - Nodes: planner, researcher, writer
#   - Edges: START → planner → researcher → writer → END
#   - Compile the graph
#
# Hint: from langgraph.graph import StateGraph, END

def build_research_graph():
    """Build the LangGraph research pipeline."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 8: Add Conditional Edge (Quality Loop)
# ================================================================
# After the researcher node, add a conditional edge:
#   - If research quality is LOW (fewer than 2 results or very short),
#     loop back to researcher for another round (max 2 iterations)
#   - If quality is GOOD, proceed to writer
#
# Hint: Use a conditional edge function:
#   def check_research_quality(state) -> str:
#       results = state.get("research_results", [])
#       if len(results) < 2:
#           return "researcher"  # loop back
#       return "writer"  # proceed

# --- YOUR CODE HERE ---
# Add conditional edge to the graph from TODO 7
# --- END YOUR CODE ---


# ================================================================
# TODO 9: Add Cost Tracking and Budget Enforcement
# ================================================================
# Add a cost tracking mechanism:
#   1. Track tokens at each LLM call (estimate: len(text.split()) * 1.3)
#   2. Track cumulative cost (use $0.001 per 1000 tokens as estimate)
#   3. Before each LLM call, check if budget ($0.50) is exceeded
#   4. If over budget, skip the LLM call and return a budget error
#
# Hint: Add budget checking to planner_node, researcher_node, writer_node

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

# --- YOUR CODE HERE ---
# Integrate budget.check_budget() and budget.log() into the nodes above
# --- END YOUR CODE ---


# ================================================================
# TODO 10: Add Phoenix Tracing at Graph Boundaries
# ================================================================
# Add timing spans at each graph node:
#   - planner_node: trace planning time
#   - researcher_node: trace research time + MCP calls
#   - writer_node: trace writing time + A2A calls
#
# Format: print(f"  📊 Span [node_name]: {duration_ms:.0f}ms")

# --- YOUR CODE HERE ---
# Add timing/tracing to each node above
# --- END YOUR CODE ---


# ================================================================
# TODO 11: Implement the Demo Runner
# ================================================================
# Create an async function that:
#   1. Starts the Writer A2A server (TODO 6) in a background thread
#   2. Builds the LangGraph (TODO 7)
#   3. Runs the graph with a sample topic
#   4. Prints the results: plan, research findings, final report
#   5. Prints the cost report

async def run_demo():
    """Run the complete multi-agent research system."""
    print("=" * 70)
    print("Capstone: Multi-Agent Research System")
    print("MCP (tools) + A2A (agents) + LangGraph (orchestration)")
    print("=" * 70)

    # --- YOUR CODE HERE ---
    # 1. Start Writer A2A server
    # 2. Build graph
    # 3. Run with topic: "The impact of AI agents on software development in 2026"
    # 4. Print results
    # 5. Print cost report
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 12: Run the Demo
# ================================================================

if __name__ == "__main__":
    print("\n⚠️  Exercise 3 (Capstone): Complete all 12 TODOs!")
    print("    This combines MCP + A2A + LangGraph into one system.")
    print("    Study examples 05, 08, 09, and 12 for reference.\n")

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n  ⏹️  Interrupted")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Install: pip install mcp fastapi uvicorn httpx langgraph langchain-groq")
