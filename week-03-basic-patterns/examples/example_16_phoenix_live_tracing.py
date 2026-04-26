import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 16: Live Phoenix Tracing -- See Real LLM Traces
=========================================================
This example makes REAL LLM calls and captures traces in Phoenix.
When you run it, open http://localhost:6006 in your browser to see:

  - Every LLM call (prompt, response, tokens, latency)
  - Every tool call (arguments, return value, duration)
  - The full execution flow as a visual timeline

HOW IT WORKS (3 steps):
  1. Start Phoenix (local tracing dashboard)
  2. Instrument LangChain (auto-captures all LLM/tool calls)
  3. Run your agent normally -- traces appear in Phoenix automatically

WHAT YOU'LL SEE IN PHOENIX:
  The dashboard shows "traces" -- each trace is one complete agent run.
  Click a trace to expand it into "spans" (individual steps):

    Trace: "What is the world population?"
    |-- Span: LLM Call (input prompt, output, 450ms, 200 tokens)
    |-- Span: Tool Call - search_web (args, result, 5ms)
    |-- Span: LLM Call (final answer, 300ms, 150 tokens)

  This lets you debug WHY an agent gave a wrong answer, WHERE time
  is spent, and HOW MANY tokens each step costs.

Prerequisites:
  pip install arize-phoenix openinference-instrumentation-langchain

Run: python week-03-basic-patterns/examples/example_16_phoenix_live_tracing.py
"""

import os
import warnings
import tempfile

from dotenv import load_dotenv
load_dotenv("config/.env")
load_dotenv()

# Suppress noisy warnings from Phoenix's SQLAlchemy internals.
# These are harmless "unsupported index" warnings from the Phoenix DB.
warnings.filterwarnings("ignore", message=".*Skipped unsupported.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Windows fix: Phoenix stores traces in a temp SQLite DB. On exit,
# Python tries to delete it while Phoenix still has the file locked.
# This patch makes the cleanup silently skip locked files.
_orig_rmtree = getattr(tempfile.TemporaryDirectory, '_rmtree', None)
if _orig_rmtree:
    @classmethod
    def _safe_rmtree(cls, name, ignore_errors=False):
        try:
            _orig_rmtree.__func__(cls, name, ignore_errors=True)
        except Exception:
            pass
    tempfile.TemporaryDirectory._rmtree = _safe_rmtree


# ==============================================================
# STEP 1: Start Phoenix
# ==============================================================
# Phoenix is a local web app that collects and displays traces.
# launch_app() starts it on http://localhost:6006.
# It stores traces in a local SQLite database (temp directory).

print("Example 16: Live Phoenix Tracing")
print("=" * 60)

try:
    import phoenix as px
    import logging

    # Suppress noisy gRPC and Phoenix server logs. The gRPC collector
    # may fail to bind port 4317 (already in use), but the HTTP dashboard
    # at port 6006 still works fine. These errors are harmless.
    logging.getLogger("phoenix").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.CRITICAL)

    # Start the Phoenix dashboard. After this, http://localhost:6006 is live.
    px.launch_app()
    print("  [OK] Phoenix is running at http://localhost:6006")
    phoenix_active = True
except ImportError:
    print("  [FAIL] Phoenix not installed.")
    print("  Run: pip install arize-phoenix openinference-instrumentation-langchain")
    print("  Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"  [FAIL] Phoenix failed to start: {e}")
    print("  Exiting.")
    sys.exit(1)


# ==============================================================
# STEP 2: Connect OpenTelemetry to Phoenix
# ==============================================================
# Phoenix collects traces via OpenTelemetry (OTel). We need TWO things:
#
#   A) A TracerProvider that knows WHERE to send spans (-> Phoenix)
#      phoenix.otel.register() creates this and points it at Phoenix's
#      collector endpoint (http://localhost:6006/v1/traces by default).
#
#   B) An Instrumentor that knows WHAT to capture (-> LangChain calls)
#      LangChainInstrumentor().instrument() hooks into LangChain so that
#      every llm.invoke(), tool call, and agent.invoke() is auto-captured.
#
# Without (A), the instrumentor captures spans but has nowhere to send them.
# Without (B), the tracer provider is ready but nothing generates spans.
# You need BOTH for traces to show up in Phoenix.

try:
    # (A) Register a TracerProvider that exports spans to Phoenix
    from phoenix.otel import register
    tracer_provider = register(
        project_name="week-03-tracing-demo",  # Shows as project name in Phoenix UI
        endpoint="http://localhost:6006/v1/traces",  # Phoenix's collector endpoint
    )
    print("  [OK] OpenTelemetry TracerProvider connected to Phoenix")

    # (B) Instrument LangChain -- auto-captures all LLM and tool calls
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("  [OK] LangChain instrumentation active -- all LLM calls will be traced")
except ImportError:
    print("  [FAIL] Required packages not installed.")
    print("  Run: pip install arize-phoenix-otel openinference-instrumentation-langchain")
    sys.exit(1)


# ==============================================================
# STEP 3: Set up LLM and Tools
# ==============================================================
# This is a normal LangGraph agent setup -- nothing special for tracing.
# The instrumentor from Step 2 captures everything automatically.

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# -- Tools (same simulated tools as example_05) ------------------

@tool
def search_facts(query: str) -> str:
    """Search for factual information about a topic.

    Args:
        query: What to search for (e.g., 'world population', 'AI market')
    """
    facts_db = {
        "population": "World population is approximately 8.1 billion (2026). India is the most populous country.",
        "ai": "The global AI market is projected to reach $300 billion by 2027, growing at 35% annually.",
        "climate": "Global temperatures have risen 1.2C above pre-industrial levels. Renewables are 35% of electricity.",
        "space": "SpaceX Starship completed its first orbital flight in 2024. Artemis targets Moon return.",
    }
    for keyword, fact in facts_db.items():
        if keyword in query.lower():
            return fact
    return f"No results found for '{query}'."


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, parentheses.

    Args:
        expression: Math expression like '300 * 0.35' or '8.1 * 1000000000'
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: Invalid characters in '{expression}'"
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# -- LLM setup ---------------------------------------------------
# Using qwen/qwen3-32b because llama-3.3-70b has broken tool calling
# via langchain_groq (see example_05 comments for details).

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

tools = [search_facts, calculate]
llm_with_tools = llm.bind_tools(tools)


# -- Agent graph --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    """LLM decides to call a tool or respond directly."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if the LLM made tool calls, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# Build the graph: agent -> [tools -> agent]* -> END
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()


# ==============================================================
# STEP 4: Run the Agent -- Traces are Captured Automatically
# ==============================================================
# Each app.invoke() call creates ONE trace in Phoenix.
# The trace contains child spans for every LLM call and tool call.

def ask(question: str):
    """Run one agent query. This creates one trace in Phoenix."""
    print(f"\n{'-'*60}")
    print(f"  Question: {question}")
    print(f"{'-'*60}")

    result = app.invoke({
        "messages": [HumanMessage(content=question)],
    })

    # Get the final answer (last message content)
    answer = result["messages"][-1].content
    # Count how many tool calls were made (from intermediate messages)
    tool_calls = sum(
        len(m.tool_calls) for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    print(f"  Tools used: {tool_calls}")
    # Truncate very long answers (qwen3 can be verbose with <think> tags)
    display = answer
    if "<think>" in display:
        # Strip qwen3's thinking tags for cleaner display
        import re
        display = re.sub(r'<think>.*?</think>', '', display, flags=re.DOTALL).strip()
    print(f"  Answer: {display[:300]}")
    print(f"  >> This created 1 trace in Phoenix -- check localhost:6006!")


# ==============================================================
# STEP 5: Run Test Queries
# ==============================================================
# Each query below creates a separate trace in Phoenix.
# After running, open http://localhost:6006 and click on each
# trace to see the full execution breakdown.

print(f"\n{'='*60}")
print("Running test queries (each creates a trace in Phoenix)...")
print("=" * 60)

# Query 1: Simple search -- should call search_facts once
# In Phoenix: you'll see 1 LLM span + 1 tool span + 1 LLM span
ask("What is the current world population?")

# Query 2: Math only -- should call calculate once
# In Phoenix: you'll see 1 LLM span + 1 tool span + 1 LLM span
ask("What is 8.1 billion times 365?")

# Query 3: Multi-step -- should call search_facts then calculate
# In Phoenix: you'll see multiple LLM + tool spans chained together
ask("What is 15% of the global AI market value?")

# Query 4: No tools needed -- LLM answers directly
# In Phoenix: you'll see just 1 LLM span (no tool spans)
ask("What is 2 + 2?")


# ==============================================================
# STEP 6: Keep Phoenix Running So You Can Explore
# ==============================================================
# Phoenix is a web server. It needs to stay running for you to
# browse traces at localhost:6006. We wait for the user to finish
# exploring before shutting down.

print(f"\n{'='*60}")
print("DONE! All queries complete. Traces are in Phoenix.")
print("=" * 60)
print()
print("  Open http://localhost:6006 in your browser now.")
print()
print("  WHAT TO LOOK FOR:")
print("    1. Click 'Traces' in the left sidebar")
print("    2. You should see 4 traces (one per query above)")
print("    3. Click a trace to expand its spans (LLM calls, tool calls)")
print("    4. Each span shows: input, output, latency, token count")
print()
print("  TRY THIS:")
print("    - Compare trace #1 (search) vs trace #4 (no tools)")
print("    - Look at trace #3 (multi-step) to see tool chaining")
print("    - Check token counts -- which query cost the most?")
print()

try:
    input("Press Enter to shut down Phoenix and exit...")
except KeyboardInterrupt:
    pass

print("Shutting down Phoenix...")
try:
    px.close_app()
except Exception:
    pass
print("Done.")
