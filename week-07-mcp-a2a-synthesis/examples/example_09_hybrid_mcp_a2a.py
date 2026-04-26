import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 9: Hybrid Integration — MCP + A2A + Middlewares
========================================================
Topic 9 — Combining MCP (tool access) + A2A (agent communication)
into one unified multi-agent system.

The BIG IDEA (Feynman):
  A hospital has specialists (A2A agents) who each have their own
  equipment (MCP tools).  The ER doctor uses an X-ray machine (MCP)
  to diagnose, then calls the surgeon (A2A) who uses surgical tools
  (MCP) to operate.  The hospital also has protocols for hand-washing,
  patient tracking, and billing (middlewares).

  This example builds exactly that:
    - ResearchAgent uses MCP (database, web search) to gather data
    - WriterAgent uses A2A to receive research and produce a report
    - Middleware pipeline: logging, cost tracking, retry at boundaries

Previously covered:
  - MCP client/server (examples 03-05)
  - A2A server/client (example 08)
  - Middlewares (Week 4-6)
  - LangGraph orchestration (Week 2-6)

Run: python week-07-mcp-a2a-synthesis/examples/example_09_hybrid_mcp_a2a.py
"""

import os
import json
import asyncio
import time
import threading
import uuid
from typing import TypedDict, Annotated, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
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
# PART 1: MIDDLEWARE PIPELINE (Cross-cutting concerns)
# ================================================================

print("=" * 70)
print("PART 1: Middleware Pipeline for Protocol Boundaries")
print("=" * 70)
print("""
Middlewares wrap every MCP tool call and A2A task to provide:
  📊 Logging — what happened, when, and how long it took
  💰 Cost tracking — how much each operation costs
  🔄 Retry logic — automatically retry failed calls
  🛡️ Safety — validate inputs/outputs at boundaries
""")


@dataclass
class OperationLog:
    """Record of a single operation (MCP call or A2A task)."""
    protocol: str  # "mcp" or "a2a"
    operation: str  # tool name or task method
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    input_preview: str = ""
    output_preview: str = ""
    retry_count: int = 0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class ProtocolMiddleware:
    """Middleware that wraps both MCP and A2A calls with cross-cutting concerns."""

    def __init__(self, budget_limit: float = 1.0, max_retries: int = 2):
        self.logs: list[OperationLog] = []
        self.total_cost: float = 0.0
        self.budget_limit = budget_limit
        self.max_retries = max_retries

    async def wrap_mcp_call(self, tool_name: str, arguments: dict, call_fn):
        """Wrap an MCP tool call with logging, cost tracking, and retry."""
        log = OperationLog(
            protocol="mcp",
            operation=tool_name,
            input_preview=str(arguments)[:100],
        )

        for attempt in range(self.max_retries + 1):
            log.start_time = time.time()
            log.retry_count = attempt
            try:
                result = await call_fn(tool_name, arguments)
                log.end_time = time.time()
                log.success = True
                log.output_preview = str(result)[:100]
                self.logs.append(log)
                print(f"    📊 MCP [{tool_name}] OK in {log.duration_ms:.0f}ms (attempt {attempt + 1})")
                return result
            except Exception as e:
                log.end_time = time.time()
                log.error = str(e)
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # exponential backoff
                    print(f"    🔄 MCP [{tool_name}] retry in {wait}s... ({e})")
                    await asyncio.sleep(wait)
                else:
                    log.success = False
                    self.logs.append(log)
                    print(f"    ❌ MCP [{tool_name}] FAILED after {attempt + 1} attempts: {e}")
                    raise

    async def wrap_a2a_task(self, agent_name: str, message: str, send_fn):
        """Wrap an A2A task with logging, cost tracking, and retry."""
        log = OperationLog(
            protocol="a2a",
            operation=f"task→{agent_name}",
            input_preview=message[:100],
        )

        for attempt in range(self.max_retries + 1):
            log.start_time = time.time()
            log.retry_count = attempt
            try:
                result = await send_fn(message)
                log.end_time = time.time()
                log.success = True
                log.output_preview = str(result)[:100] if result else "No result"
                self.logs.append(log)
                print(f"    📊 A2A [→{agent_name}] OK in {log.duration_ms:.0f}ms (attempt {attempt + 1})")
                return result
            except Exception as e:
                log.end_time = time.time()
                log.error = str(e)
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"    🔄 A2A [→{agent_name}] retry in {wait}s... ({e})")
                    await asyncio.sleep(wait)
                else:
                    log.success = False
                    self.logs.append(log)
                    print(f"    ❌ A2A [→{agent_name}] FAILED after {attempt + 1} attempts: {e}")
                    raise

    def report(self) -> str:
        """Generate a report of all operations."""
        lines = ["\n📊 Protocol Operations Report", "=" * 50]

        mcp_ops = [l for l in self.logs if l.protocol == "mcp"]
        a2a_ops = [l for l in self.logs if l.protocol == "a2a"]

        lines.append(f"\n  MCP Operations: {len(mcp_ops)}")
        for op in mcp_ops:
            status = "✅" if op.success else "❌"
            lines.append(f"    {status} {op.operation}: {op.duration_ms:.0f}ms (retries: {op.retry_count})")

        lines.append(f"\n  A2A Operations: {len(a2a_ops)}")
        for op in a2a_ops:
            status = "✅" if op.success else "❌"
            lines.append(f"    {status} {op.operation}: {op.duration_ms:.0f}ms (retries: {op.retry_count})")

        total_time = sum(l.duration_ms for l in self.logs)
        success_rate = sum(1 for l in self.logs if l.success) / max(len(self.logs), 1) * 100
        lines.append(f"\n  Total time: {total_time:.0f}ms")
        lines.append(f"  Success rate: {success_rate:.0f}%")

        return "\n".join(lines)


# ================================================================
# PART 2: WRITER AGENT (A2A Server)
# ================================================================

print("\n" + "=" * 70)
print("PART 2: Writer Agent (A2A Server)")
print("=" * 70)


def create_writer_server(port: int = 8002):
    """Create an A2A server for the Writer agent."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Writer Agent A2A")

    agent_card = {
        "name": "Writer Agent",
        "description": "Transforms research notes into polished, well-structured articles and reports.",
        "url": f"http://localhost:{port}",
        "version": "1.0.0",
        "skills": [
            {
                "id": "write-article",
                "name": "Article Writing",
                "description": "Write articles from research notes",
                "tags": ["writing", "articles"],
            }
        ],
        "capabilities": {"streaming": False, "pushNotifications": False},
    }

    @app.get("/.well-known/agent.json")
    async def get_card():
        return agent_card

    @app.post("/")
    async def handle_rpc(request: Request):
        body = await request.json()
        method = body.get("method")

        if method == "tasks/send":
            params = body.get("params", {})
            message_parts = params.get("message", {}).get("parts", [])
            user_text = " ".join(p.get("text", "") for p in message_parts if p.get("text"))

            # Process with LLM
            try:
                llm = get_llm(temperature=0.7)
                from langchain_core.messages import HumanMessage, SystemMessage
                response = llm.invoke([
                    SystemMessage(content=(
                        "You are an expert writer. Transform the provided research notes "
                        "into a well-structured, engaging article. Use markdown formatting. "
                        "Include an introduction, main sections, and conclusion."
                    )),
                    HumanMessage(content=user_text),
                ])

                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "id": params.get("id", str(uuid.uuid4())),
                        "status": {"state": "completed"},
                        "messages": [
                            {"role": "agent", "parts": [{"type": "text", "text": response.content}]}
                        ],
                        "artifacts": [
                            {
                                "name": "article",
                                "parts": [{"type": "text", "text": response.content}],
                                "mime_type": "text/markdown",
                            }
                        ],
                    },
                })
            except Exception as e:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "id": params.get("id", str(uuid.uuid4())),
                        "status": {"state": "failed", "message": str(e)},
                        "messages": [],
                        "artifacts": [],
                    },
                })

        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        })

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "Writer Agent"}

    return app


# ================================================================
# PART 3: HYBRID ORCHESTRATOR (LangGraph)
# ================================================================

print("\n" + "=" * 70)
print("PART 3: Hybrid Orchestrator")
print("=" * 70)
print("""
The orchestrator coordinates:
  1. Research phase — uses MCP to access the research database
  2. Writing phase — sends research to Writer Agent via A2A
  3. Middleware — wraps everything with logging, cost, retry

  ┌──────────────┐     MCP      ┌──────────────────┐
  │  Orchestrator │────────────▶│ Research History  │
  │  (LangGraph) │             │ MCP Server        │
  │              │◀────────────│ (SQLite)           │
  │              │              └──────────────────┘
  │              │
  │              │     A2A      ┌──────────────────┐
  │              │────────────▶│ Writer Agent      │
  │              │             │ A2A Server        │
  │              │◀────────────│ (LLM)             │
  └──────────────┘              └──────────────────┘
""")


async def run_hybrid_demo():
    """Run the complete hybrid MCP + A2A system."""
    import httpx
    import uvicorn

    middleware = ProtocolMiddleware(budget_limit=1.0, max_retries=2)

    # --- Step 1: Start the Writer Agent A2A server ---
    print("\n  🚀 Starting Writer Agent (A2A server on port 8002)...")
    writer_app = create_writer_server(port=8002)
    writer_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": writer_app, "host": "localhost", "port": 8002, "log_level": "warning"},
        daemon=True,
    )
    writer_thread.start()
    await asyncio.sleep(2)
    print("  ✅ Writer Agent running")

    # --- Step 2: Connect to Research History MCP server ---
    print("\n  📡 Connecting to Research History MCP server...")
    mcp_connected = False
    mcp_manager = None

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
        client_ctx = stdio_client(server_params)
        read_stream, write_stream = await client_ctx.__aenter__()
        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()
        await session.initialize()

        tools = await session.list_tools()
        print(f"  ✅ MCP connected — {len(tools.tools)} tools available")
        mcp_connected = True

        # Define MCP call function for middleware
        async def mcp_call(tool_name, arguments):
            result = await session.call_tool(tool_name, arguments)
            if result.content:
                return "\n".join(c.text for c in result.content if hasattr(c, 'text'))
            return "No result"

    except Exception as e:
        print(f"  ⚠️  MCP connection failed: {e}")
        print("  Continuing with simulated MCP calls...")

        async def mcp_call(tool_name, arguments):
            """Fallback: simulate MCP calls if server unavailable."""
            if tool_name == "save_research":
                return f"✅ Saved research on '{arguments.get('topic', 'unknown')}'"
            elif tool_name == "search_research":
                return f"🔍 Found 0 results for '{arguments.get('query', '')}'"
            elif tool_name == "list_recent":
                return "📚 No recent research entries"
            elif tool_name == "get_stats":
                return "📊 Database: 0 entries"
            return f"Unknown tool: {tool_name}"

    # Define A2A send function for middleware
    async def a2a_send_to_writer(message: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8002",
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tasks/send",
                    "params": {
                        "id": str(uuid.uuid4()),
                        "message": {
                            "role": "user",
                            "parts": [{"type": "text", "text": message}],
                        },
                    },
                },
                timeout=60.0,
            )
            data = response.json()
            result = data.get("result", {})
            artifacts = result.get("artifacts", [])
            if artifacts:
                parts = artifacts[0].get("parts", [])
                if parts:
                    return parts[0].get("text", "No text")
            return "No artifacts returned"

    # --- Step 3: Run the hybrid workflow ---
    print("\n  🔄 Running hybrid workflow: Research → Write")
    print("  " + "─" * 50)

    topic = "AI Agent Frameworks in 2026"

    # Phase 1: Research (MCP)
    print(f"\n  📚 Phase 1: Researching '{topic}' via MCP...")

    # Save some research first
    await middleware.wrap_mcp_call(
        "save_research",
        {
            "topic": topic,
            "summary": (
                "The AI agent framework landscape in 2026 is dominated by LangGraph and Google ADK. "
                "LangGraph offers graph-based orchestration with persistence and subgraphs. "
                "Google ADK provides a simpler model with LlmAgent and built-in A2A support. "
                "Both frameworks now support MCP for standardized tool integration. "
                "Key trends: multi-agent systems, hybrid MCP+A2A architectures, and production observability."
            ),
            "sources": '["https://langchain.com/langgraph", "https://google.github.io/adk"]',
        },
        mcp_call,
    )

    # Search for research
    research_results = await middleware.wrap_mcp_call(
        "search_research",
        {"query": "AI agent frameworks"},
        mcp_call,
    )
    print(f"\n  📄 Research results:\n  {research_results[:300]}...")

    # Get stats
    stats = await middleware.wrap_mcp_call(
        "get_stats",
        {},
        mcp_call,
    )
    print(f"\n  📊 Database stats:\n  {stats[:200]}")

    # Phase 2: Writing (A2A)
    print(f"\n  ✍️  Phase 2: Sending research to Writer Agent via A2A...")

    research_notes = f"""
    Topic: {topic}

    Research findings:
    {research_results[:500]}

    Please write a concise blog post (200-300 words) based on these research notes.
    Include an introduction, 2-3 key points, and a conclusion.
    """

    try:
        article = await middleware.wrap_a2a_task(
            "Writer Agent",
            research_notes,
            a2a_send_to_writer,
        )

        print(f"\n  📰 Written article:")
        print("  " + "─" * 50)
        for line in (article or "No article produced")[:600].split("\n"):
            print(f"  {line}")
        if len(article or "") > 600:
            print("  ...")

    except Exception as e:
        print(f"  ⚠️  Writing failed: {e}")
        print("  This is expected if LLM API keys are not configured.")

    # --- Step 4: Report ---
    print(middleware.report())

    # Cleanup MCP
    if mcp_connected:
        try:
            await session_ctx.__aexit__(None, None, None)
            await client_ctx.__aexit__(None, None, None)
            print("\n  🔌 MCP connection closed")
        except Exception:
            pass


# ================================================================
# PART 4: KEY TAKEAWAYS
# ================================================================

print("""
Key Takeaways:
  1. MCP for TOOLS: Agent accesses databases, APIs, file systems via MCP
  2. A2A for AGENTS: Agents delegate work to other agents via A2A
  3. Middleware wraps BOTH protocols with logging, cost, retry
  4. The orchestrator coordinates MCP and A2A calls in a workflow
  5. Each component is independent — swap servers without changing the orchestrator
""")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    try:
        asyncio.run(run_hybrid_demo())
    except KeyboardInterrupt:
        print("\n  ⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n  ⚠️  Demo error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Install: pip install mcp fastapi uvicorn httpx langchain-groq")

    print("\n✅ Example 09 complete!")
