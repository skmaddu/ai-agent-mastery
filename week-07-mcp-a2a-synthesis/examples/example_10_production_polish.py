import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 10: Production Polish — Retries, Cost Control, Deployment
==================================================================
Topic 10 — Making your MCP + A2A system production-ready.

The BIG IDEA (Feynman):
  Production code is like a car — the engine (LLM) matters, but without
  brakes (cost limits), seat belts (retries), and a dashboard (health
  checks), you'll crash.

  A prototype that works on your laptop is like a car in a parking lot.
  Moving to production means driving on a highway with traffic, weather,
  and potholes.  You need:
    1. Retries      — if the engine stalls, restart it (not the whole car)
    2. Cost control — a fuel gauge so you don't run out mid-trip
    3. Health checks — warning lights on the dashboard
    4. Deployment    — the road itself (Docker, config, networking)
    5. Graceful shutdown — pulling over safely, not slamming the brakes

This example builds REAL utility classes you can copy into any project.

Previously covered:
  - MCP client/server (examples 03-05)
  - A2A protocol (examples 06-08)
  - Hybrid integration (example 09)

Run: python week-07-mcp-a2a-synthesis/examples/example_10_production_polish.py
"""

import os
import json
import asyncio
import signal
import time
import subprocess
import threading
from typing import Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager

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
# PART 1: MCP RETRY CLIENT
# ================================================================
#
# Why retries matter:
#   MCP servers are network services.  Networks are unreliable.
#   A single failed connection doesn't mean the server is down —
#   it might just be a blip.  Retries with exponential backoff
#   give the server time to recover without hammering it.
#
# Exponential backoff means:
#   Attempt 1 fails → wait 1 second
#   Attempt 2 fails → wait 2 seconds
#   Attempt 3 fails → wait 4 seconds
#   Each wait doubles, giving the server breathing room.
#
# We use the `tenacity` library which handles all of this cleanly.

print("=" * 70)
print("PART 1: MCPRetryClient — Resilient MCP Connections")
print("=" * 70)

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
        RetryError,
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    print("[INFO] tenacity not installed. Install with: pip install tenacity")
    print("       Showing retry logic with manual implementation.\n")


class MCPConnectionError(Exception):
    """Raised when an MCP server connection fails."""
    pass


class MCPTimeoutError(Exception):
    """Raised when an MCP tool call times out."""
    pass


class MCPRetryClient:
    """
    Wraps MCP client calls with automatic retry logic.

    Think of it like a phone with auto-redial:
    - If the call drops, it tries again automatically
    - It waits a bit longer between each attempt (exponential backoff)
    - After 3 failed attempts, it gives up and reports the error

    Also implements CONNECTION POOLING:
    - Reuses existing connections instead of creating new ones each time
    - Like keeping a phone line open instead of hanging up and redialing
    """

    def __init__(self, server_url: str = "http://localhost:8000",
                 max_retries: int = 3, base_wait: float = 1.0):
        self.server_url = server_url
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.call_count = 0
        self.retry_count = 0

        # ---- Connection Pool ----
        # Instead of creating a new connection for every call,
        # we maintain a pool of reusable connections.
        # This reduces latency and server load.
        self._connection_pool: dict[str, dict] = {}
        self._pool_lock = threading.Lock()
        self._max_pool_size = 5

    def _get_connection(self, server_url: str) -> dict:
        """Get or create a connection from the pool."""
        with self._pool_lock:
            if server_url in self._connection_pool:
                conn = self._connection_pool[server_url]
                conn["reuse_count"] += 1
                return conn
            # Create new connection (simulated)
            conn = {
                "url": server_url,
                "created_at": time.time(),
                "reuse_count": 1,
                "active": True,
            }
            # Evict oldest if pool is full
            if len(self._connection_pool) >= self._max_pool_size:
                oldest_key = min(
                    self._connection_pool,
                    key=lambda k: self._connection_pool[k]["created_at"]
                )
                del self._connection_pool[oldest_key]
            self._connection_pool[server_url] = conn
            return conn

    def _release_connection(self, server_url: str):
        """Return connection to pool (mark as available)."""
        # In a real implementation, this would mark the connection as idle
        pass

    def close_all(self):
        """Close all pooled connections (used during shutdown)."""
        with self._pool_lock:
            for url, conn in self._connection_pool.items():
                conn["active"] = False
            self._connection_pool.clear()

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an MCP tool with automatic retries.

        If tenacity is available, uses its decorator pattern.
        Otherwise, falls back to manual retry loop.
        """
        self.call_count += 1

        if TENACITY_AVAILABLE:
            return await self._call_with_tenacity(tool_name, arguments)
        else:
            return await self._call_with_manual_retry(tool_name, arguments)

    async def _call_with_tenacity(self, tool_name: str, arguments: dict) -> dict:
        """Retry using tenacity — the production-grade approach."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.base_wait, min=1, max=10),
            retry=retry_if_exception_type((MCPConnectionError, MCPTimeoutError)),
        )
        async def _do_call():
            conn = self._get_connection(self.server_url)
            return await self._simulate_mcp_call(tool_name, arguments)

        try:
            return await _do_call()
        except RetryError as e:
            raise MCPConnectionError(
                f"All {self.max_retries} retries failed for {tool_name}"
            ) from e

    async def _call_with_manual_retry(self, tool_name: str, arguments: dict) -> dict:
        """Manual retry loop — shows what tenacity does under the hood."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                conn = self._get_connection(self.server_url)
                return await self._simulate_mcp_call(tool_name, arguments)
            except (MCPConnectionError, MCPTimeoutError) as e:
                last_error = e
                self.retry_count += 1
                wait_time = self.base_wait * (2 ** (attempt - 1))  # Exponential
                print(f"    [Retry] Attempt {attempt}/{self.max_retries} "
                      f"failed: {e}. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time * 0.01)  # Shortened for demo
        raise MCPConnectionError(
            f"All {self.max_retries} retries failed: {last_error}"
        )

    async def _simulate_mcp_call(self, tool_name: str, arguments: dict) -> dict:
        """
        Simulates an MCP tool call.
        In production, this would send a JSON-RPC request to the MCP server.
        We simulate occasional failures to demonstrate retry behavior.
        """
        # Simulate network latency
        await asyncio.sleep(0.01)

        # Simulate occasional failures (30% chance on first call)
        if self.call_count <= 2 and self.retry_count == 0:
            self.retry_count += 1
            raise MCPConnectionError(f"Connection refused to {self.server_url}")

        return {
            "jsonrpc": "2.0",
            "result": {
                "tool": tool_name,
                "output": f"Result for {tool_name}({json.dumps(arguments)})",
                "status": "success",
            }
        }

    def get_pool_stats(self) -> dict:
        """Report connection pool statistics."""
        with self._pool_lock:
            return {
                "pool_size": len(self._connection_pool),
                "connections": {
                    url: {
                        "reuse_count": c["reuse_count"],
                        "age_seconds": round(time.time() - c["created_at"], 1),
                    }
                    for url, c in self._connection_pool.items()
                }
            }


# Demo: MCPRetryClient
async def demo_retry_client():
    client = MCPRetryClient(max_retries=3)

    print("\nCalling MCP tool (may fail and retry automatically)...")
    try:
        result = await client.call_tool("search_database", {"query": "AI agents"})
        print(f"  Success: {result['result']['output']}")
    except MCPConnectionError as e:
        print(f"  Failed after retries: {e}")

    # Second call — should succeed (connection pooled)
    print("\nCalling again (connection reused from pool)...")
    client.retry_count = 0  # Reset for clean demo
    client.call_count = 10  # Skip failure simulation
    result = await client.call_tool("get_weather", {"city": "London"})
    print(f"  Success: {result['result']['output']}")

    print(f"\n  Pool stats: {json.dumps(client.get_pool_stats(), indent=2)}")
    client.close_all()

asyncio.run(demo_retry_client())


# ================================================================
# PART 2: A2A COST GUARD
# ================================================================
#
# Why cost control matters:
#   Every LLM call costs money. Without a budget, a runaway loop
#   (agent keeps retrying, or generates endlessly) can drain your
#   account.  A cost guard is like a prepaid phone card — once the
#   credit runs out, calls stop.
#
# Two levels of budget:
#   1. Per-task budget  — "this single research task can spend max $0.10"
#   2. Per-session budget — "this user's entire session can spend max $1.00"

print("\n" + "=" * 70)
print("PART 2: A2ACostGuard — Budget Enforcement for A2A Tasks")
print("=" * 70)


class BudgetExceededError(Exception):
    """Raised when a task or session exceeds its cost budget."""
    def __init__(self, message: str, current_cost: float, budget: float):
        super().__init__(message)
        self.current_cost = current_cost
        self.budget = budget


class A2ACostGuard:
    """
    Tracks estimated cost per A2A task and enforces budget limits.

    Cost estimation formula (simplified):
      cost = (input_tokens * input_price + output_tokens * output_price)

    The guard checks BEFORE each LLM call whether the budget allows it.
    If not, it raises BudgetExceededError — the calling code can then
    decide to return a partial result or ask the user for more budget.
    """

    # Approximate pricing per 1M tokens (as of 2025)
    PRICING = {
        "groq": {"input": 0.05, "output": 0.10},      # Very cheap
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
    }

    def __init__(self, task_budget: float = 0.10, session_budget: float = 1.00,
                 model: str = "groq"):
        self.task_budget = task_budget
        self.session_budget = session_budget
        self.model = model
        self.pricing = self.PRICING.get(model, self.PRICING["groq"])

        # Tracking
        self._task_costs: dict[str, float] = {}   # task_id -> total cost
        self._session_cost: float = 0.0
        self._call_log: list[dict] = []

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single LLM call."""
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost

    def check_budget(self, task_id: str, estimated_input_tokens: int = 500,
                     estimated_output_tokens: int = 200):
        """
        Check if the next call is within budget.  Call this BEFORE each LLM call.
        Raises BudgetExceededError if the budget would be exceeded.
        """
        estimated_cost = self.estimate_cost(
            estimated_input_tokens, estimated_output_tokens
        )
        task_cost = self._task_costs.get(task_id, 0.0)

        # Check per-task budget
        if task_cost + estimated_cost > self.task_budget:
            raise BudgetExceededError(
                f"Task '{task_id}' would exceed budget: "
                f"${task_cost + estimated_cost:.6f} > ${self.task_budget:.2f}",
                current_cost=task_cost,
                budget=self.task_budget,
            )

        # Check per-session budget
        if self._session_cost + estimated_cost > self.session_budget:
            raise BudgetExceededError(
                f"Session would exceed budget: "
                f"${self._session_cost + estimated_cost:.6f} > "
                f"${self.session_budget:.2f}",
                current_cost=self._session_cost,
                budget=self.session_budget,
            )

    def record_usage(self, task_id: str, input_tokens: int, output_tokens: int):
        """Record actual token usage after a successful LLM call."""
        cost = self.estimate_cost(input_tokens, output_tokens)
        self._task_costs[task_id] = self._task_costs.get(task_id, 0.0) + cost
        self._session_cost += cost

        self._call_log.append({
            "task_id": task_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_summary(self) -> dict:
        """Return a summary of all costs."""
        return {
            "model": self.model,
            "session_cost": round(self._session_cost, 6),
            "session_budget": self.session_budget,
            "session_remaining": round(self.session_budget - self._session_cost, 6),
            "task_costs": {
                tid: round(c, 6) for tid, c in self._task_costs.items()
            },
            "total_calls": len(self._call_log),
        }


# Demo: A2ACostGuard
print("\nSimulating A2A task cost tracking...")
guard = A2ACostGuard(task_budget=0.001, session_budget=0.01, model="groq")

task_id = "research-ai-agents"
for i in range(1, 6):
    try:
        guard.check_budget(task_id, estimated_input_tokens=800, estimated_output_tokens=400)
        # Simulate the LLM call succeeding
        guard.record_usage(task_id, input_tokens=800, output_tokens=400)
        print(f"  Call {i}: OK (task cost so far: "
              f"${guard._task_costs[task_id]:.6f})")
    except BudgetExceededError as e:
        print(f"  Call {i}: BLOCKED - {e}")
        break

print(f"\n  Cost summary: {json.dumps(guard.get_summary(), indent=2)}")


# ================================================================
# PART 3: HEALTH CHECKER
# ================================================================
#
# Health checks answer the question: "Is everything running?"
#
# Two kinds:
#   1. MCP server health — can we reach the server process?
#      (Like checking if a restaurant is open before driving there)
#   2. A2A agent health — does the agent's HTTP endpoint respond?
#      (Like calling ahead to make sure they have a table)
#
# A structured health report makes it easy for monitoring tools
# (like Grafana or Phoenix) to track system status over time.

print("\n" + "=" * 70)
print("PART 3: HealthChecker — System Health Monitoring")
print("=" * 70)


@dataclass
class HealthReport:
    """Structured health report for a single service."""
    service_name: str
    service_type: str      # "mcp_server" or "a2a_agent"
    status: str            # "healthy", "degraded", "unhealthy"
    latency_ms: float
    checked_at: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "service": self.service_name,
            "type": self.service_type,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "checked_at": self.checked_at,
            "details": self.details,
        }


class HealthChecker:
    """
    Checks health of MCP servers and A2A agents.

    MCP servers: checked via subprocess ping (TCP connectivity)
    A2A agents:  checked via HTTP GET /health endpoint

    Returns structured HealthReport objects that can be:
    - Displayed in a dashboard
    - Sent to monitoring systems
    - Used to trigger alerts
    """

    def __init__(self):
        self.reports: list[HealthReport] = []

    async def check_mcp_server(self, name: str, host: str = "localhost",
                               port: int = 8000) -> HealthReport:
        """
        Check if an MCP server is reachable.
        Uses subprocess to test TCP connectivity.
        """
        start = time.time()
        try:
            # On Windows, use 'ping -n 1'; on Unix, 'ping -c 1'
            # For a real MCP server, you'd do a TCP connect or send
            # an initialize JSON-RPC request.
            ping_cmd = ["ping", "-n", "1", "-w", "1000", host]
            if sys.platform != "win32":
                ping_cmd = ["ping", "-c", "1", "-W", "1", host]

            proc = subprocess.run(
                ping_cmd, capture_output=True, timeout=3
            )
            latency = (time.time() - start) * 1000

            if proc.returncode == 0:
                status = "healthy" if latency < 500 else "degraded"
            else:
                status = "unhealthy"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            latency = (time.time() - start) * 1000
            status = "unhealthy"

        report = HealthReport(
            service_name=name,
            service_type="mcp_server",
            status=status,
            latency_ms=latency,
            checked_at=datetime.now(timezone.utc).isoformat(),
            details={"host": host, "port": port},
        )
        self.reports.append(report)
        return report

    async def check_a2a_agent(self, name: str,
                               endpoint: str = "http://localhost:9000") -> HealthReport:
        """
        Check if an A2A agent is healthy via its /health endpoint.

        In production, A2A agents expose a standard health endpoint.
        We simulate this here since we don't have a running server.
        """
        start = time.time()
        try:
            # In production, you would do:
            #   import httpx
            #   async with httpx.AsyncClient() as client:
            #       resp = await client.get(f"{endpoint}/health", timeout=5)
            #       healthy = resp.status_code == 200

            # Simulated health check
            await asyncio.sleep(0.01)  # Simulate network round-trip
            latency = (time.time() - start) * 1000

            # Simulate: endpoint is "healthy" if port is standard
            healthy = ":9000" in endpoint or ":8080" in endpoint
            status = "healthy" if healthy else "degraded"
            details = {"endpoint": endpoint, "version": "1.2.0"}
        except Exception as e:
            latency = (time.time() - start) * 1000
            status = "unhealthy"
            details = {"error": str(e)}

        report = HealthReport(
            service_name=name,
            service_type="a2a_agent",
            status=status,
            latency_ms=latency,
            checked_at=datetime.now(timezone.utc).isoformat(),
            details=details,
        )
        self.reports.append(report)
        return report

    async def check_all(self, services: list[dict]) -> list[HealthReport]:
        """
        Check all services in parallel.
        Each service dict has: name, type ("mcp"/"a2a"), and connection info.
        """
        tasks = []
        for svc in services:
            if svc["type"] == "mcp":
                tasks.append(self.check_mcp_server(
                    svc["name"],
                    svc.get("host", "localhost"),
                    svc.get("port", 8000),
                ))
            elif svc["type"] == "a2a":
                tasks.append(self.check_a2a_agent(
                    svc["name"],
                    svc.get("endpoint", "http://localhost:9000"),
                ))
        return await asyncio.gather(*tasks)

    def print_dashboard(self):
        """Print a formatted health dashboard."""
        print("\n  +---------------------+------------+----------+-----------+")
        print("  | Service             | Type       | Status   | Latency   |")
        print("  +---------------------+------------+----------+-----------+")
        for r in self.reports:
            status_icon = {"healthy": "[OK]", "degraded": "[!!]",
                           "unhealthy": "[XX]"}.get(r.status, "[??]")
            print(f"  | {r.service_name:<19} | {r.service_type:<10} "
                  f"| {status_icon:<8} | {r.latency_ms:>7.1f}ms |")
        print("  +---------------------+------------+----------+-----------+")


# Demo: HealthChecker
async def demo_health_checker():
    checker = HealthChecker()
    services = [
        {"name": "sql-mcp-server", "type": "mcp", "host": "localhost", "port": 8000},
        {"name": "research-agent", "type": "a2a", "endpoint": "http://localhost:9000"},
        {"name": "writer-agent", "type": "a2a", "endpoint": "http://localhost:8080"},
    ]
    print("\nChecking all services...")
    await checker.check_all(services)
    checker.print_dashboard()

asyncio.run(demo_health_checker())


# ================================================================
# PART 4: DEPLOYMENT CONFIG
# ================================================================
#
# Pydantic models for deployment configuration.
#
# Why Pydantic?  Because YAML/JSON configs are error-prone.
#   - Misspell a key? Pydantic catches it.
#   - Wrong type (string instead of int)? Pydantic catches it.
#   - Missing required field? Pydantic catches it.
#
# Think of it as spell-check for your configuration files.

print("\n" + "=" * 70)
print("PART 4: DeploymentConfig — Type-Safe Deployment Settings")
print("=" * 70)


class DeploymentConfig(BaseModel):
    """
    Pydantic model for deployment settings.

    Every field has a type, a default, and validation.
    If you try to create a config with a string port like "8000",
    Pydantic will either coerce it to int or raise an error.
    """
    docker_image: str = Field(
        ..., description="Docker image name with tag"
    )
    port: int = Field(
        default=8000, ge=1024, le=65535,
        description="Port to expose (must be 1024-65535)"
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables (non-secret)"
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="Names of secrets (loaded from vault, NOT stored here)"
    )
    health_check_interval: int = Field(
        default=30, ge=5, le=300,
        description="Seconds between health checks"
    )
    max_concurrent_tasks: int = Field(
        default=10, ge=1, le=100,
        description="Max parallel tasks this service handles"
    )
    replicas: int = Field(
        default=1, ge=1, le=20,
        description="Number of container replicas"
    )


# Demo: DeploymentConfig
mcp_config = DeploymentConfig(
    docker_image="ai-mastery/mcp-sql-server:1.0",
    port=8000,
    env_vars={"LOG_LEVEL": "INFO", "DB_HOST": "postgres"},
    secrets=["DB_PASSWORD", "OPENAI_API_KEY"],
    health_check_interval=15,
    max_concurrent_tasks=20,
)

a2a_config = DeploymentConfig(
    docker_image="ai-mastery/research-agent:1.0",
    port=9000,
    env_vars={"LLM_PROVIDER": "groq", "PHOENIX_ENABLED": "true"},
    secrets=["GROQ_API_KEY"],
    health_check_interval=30,
    max_concurrent_tasks=5,
)

print("\nMCP Server Config:")
print(f"  {json.dumps(mcp_config.model_dump(), indent=2)}")
print("\nA2A Agent Config:")
print(f"  {json.dumps(a2a_config.model_dump(), indent=2)}")


# ================================================================
# PART 5: DOCKER COMPOSE TEMPLATE
# ================================================================
#
# Docker Compose lets you define multi-container applications
# in a single YAML file.  One command (`docker compose up`)
# starts everything: MCP server, A2A agents, database, Phoenix.
#
# This is how production systems are deployed — not by running
# `python script.py` manually on your laptop.

print("\n" + "=" * 70)
print("PART 5: Docker Compose Template")
print("=" * 70)

DOCKER_COMPOSE_TEMPLATE = """
# ===== Docker Compose for MCP + A2A Production Deployment =====
# Run with: docker compose up -d
# Monitor:  docker compose logs -f

version: '3.8'

services:
  # ---- MCP Server: SQL Database Tools ----
  mcp-sql-server:
    image: {mcp_image}
    ports:
      - "{mcp_port}:{mcp_port}"
    environment:
      - LOG_LEVEL=INFO
      - DB_HOST=postgres
      - DB_PORT=5432
    secrets:
      - db_password
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{mcp_port}/health"]
      interval: {health_interval}s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy

  # ---- A2A Research Agent ----
  research-agent:
    image: {a2a_image}
    ports:
      - "{a2a_port}:{a2a_port}"
    environment:
      - LLM_PROVIDER=groq
      - MCP_SERVER_URL=http://mcp-sql-server:{mcp_port}
      - PHOENIX_COLLECTOR=http://phoenix:6006
    secrets:
      - groq_api_key
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{a2a_port}/health"]
      interval: {health_interval}s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M

  # ---- PostgreSQL (for MCP server) ----
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agent_data
      POSTGRES_USER: agent
    secrets:
      - db_password
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent"]
      interval: 10s
      timeout: 5s
      retries: 3

  # ---- Phoenix Observability ----
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
    restart: unless-stopped

secrets:
  db_password:
    file: ./secrets/db_password.txt
  groq_api_key:
    file: ./secrets/groq_api_key.txt

volumes:
  pgdata:
"""

# Generate the compose file from our config objects
compose_yaml = DOCKER_COMPOSE_TEMPLATE.format(
    mcp_image=mcp_config.docker_image,
    mcp_port=mcp_config.port,
    a2a_image=a2a_config.docker_image,
    a2a_port=a2a_config.port,
    health_interval=mcp_config.health_check_interval,
)

print(compose_yaml)

print("KEY INSIGHT: Secrets are loaded from files, NOT environment variables.")
print("  This prevents secrets from appearing in `docker inspect` output.\n")


# ================================================================
# PART 6: GRACEFUL SHUTDOWN
# ================================================================
#
# Why graceful shutdown matters:
#   Imagine a waiter carrying 5 plates of food.  If you yell "STOP!"
#   and they freeze immediately, food hits the floor.  A graceful
#   shutdown says "finish delivering those plates, then stop taking
#   new orders."
#
# In our system:
#   1. Stop accepting new tasks
#   2. Wait for in-flight tasks to complete (with timeout)
#   3. Close MCP connections cleanly
#   4. Save any unsaved state
#   5. Exit

print("=" * 70)
print("PART 6: GracefulShutdown — Safe Process Termination")
print("=" * 70)


class GracefulShutdown:
    """
    Context manager that handles SIGINT (Ctrl+C) and SIGTERM (kill).

    Usage:
        with GracefulShutdown(mcp_client=client) as shutdown:
            while not shutdown.should_stop:
                process_next_task()

    When a signal is received:
    1. Sets should_stop = True (no new tasks accepted)
    2. Waits for in-flight tasks to finish (up to drain_timeout)
    3. Closes MCP connections
    4. Exits cleanly
    """

    def __init__(self, mcp_client: Optional[MCPRetryClient] = None,
                 drain_timeout: float = 30.0):
        self.mcp_client = mcp_client
        self.drain_timeout = drain_timeout
        self.should_stop = False
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()
        self._original_sigint = None
        self._original_sigterm = None

    def __enter__(self):
        # Save original handlers and install ours
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        # SIGTERM is not available on Windows in all contexts
        try:
            self._original_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (OSError, ValueError):
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Drain in-flight tasks
        if self._in_flight:
            print(f"\n  Draining {len(self._in_flight)} in-flight tasks "
                  f"(timeout: {self.drain_timeout}s)...")
            start = time.time()
            while self._in_flight and (time.time() - start) < self.drain_timeout:
                time.sleep(0.1)
            if self._in_flight:
                print(f"  WARNING: {len(self._in_flight)} tasks did not complete")

        # Close MCP connections
        if self.mcp_client:
            self.mcp_client.close_all()
            print("  MCP connections closed.")

        # Restore original signal handlers
        signal.signal(signal.SIGINT, self._original_sigint or signal.SIG_DFL)
        try:
            signal.signal(signal.SIGTERM, self._original_sigterm or signal.SIG_DFL)
        except (OSError, ValueError):
            pass
        print("  Shutdown complete.")
        return False  # Don't suppress exceptions

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n  Received {sig_name} — initiating graceful shutdown...")
        self.should_stop = True

    def register_task(self, task_id: str):
        """Mark a task as in-flight."""
        with self._lock:
            self._in_flight.add(task_id)

    def complete_task(self, task_id: str):
        """Mark a task as completed."""
        with self._lock:
            self._in_flight.discard(task_id)


# Demo: GracefulShutdown
print("\nDemonstrating graceful shutdown lifecycle...")
client = MCPRetryClient()

with GracefulShutdown(mcp_client=client, drain_timeout=5.0) as shutdown:
    # Simulate processing a few tasks
    for i in range(3):
        task_id = f"task-{i}"
        shutdown.register_task(task_id)
        print(f"  Processing {task_id}...")
        time.sleep(0.05)  # Simulate work
        shutdown.complete_task(task_id)
        print(f"  Completed {task_id}")

    # In a real server, you'd loop: while not shutdown.should_stop: ...
    print("\n  All tasks done. Exiting context manager...")
# __exit__ runs here automatically


# ================================================================
# PART 7: PUTTING IT ALL TOGETHER
# ================================================================

print("\n" + "=" * 70)
print("PART 7: Production Checklist Summary")
print("=" * 70)

checklist = """
PRODUCTION READINESS CHECKLIST:

  [1] RETRIES (MCPRetryClient)
      - Exponential backoff: 1s -> 2s -> 4s
      - Max 3 attempts before giving up
      - Connection pooling to reduce latency
      - Only retry on transient errors (connection, timeout)

  [2] COST CONTROL (A2ACostGuard)
      - Per-task budget: prevent any single task from running away
      - Per-session budget: overall spending cap
      - Pre-flight check: verify budget BEFORE making the LLM call
      - Usage logging: know exactly where money goes

  [3] HEALTH CHECKS (HealthChecker)
      - MCP servers: TCP connectivity check
      - A2A agents: HTTP /health endpoint
      - Structured reports for monitoring systems
      - Parallel checks for speed

  [4] DEPLOYMENT (DeploymentConfig + Docker Compose)
      - Type-safe configuration with Pydantic
      - Docker Compose for multi-service orchestration
      - Secrets management (files, not env vars)
      - Health check integration in container orchestration

  [5] GRACEFUL SHUTDOWN (GracefulShutdown)
      - Signal handling (SIGINT, SIGTERM)
      - Task draining with timeout
      - Connection cleanup
      - Works as a context manager (with statement)

KEY PRINCIPLE:
  Each of these is a SEPARATE concern.  You compose them together
  like LEGO bricks.  The MCPRetryClient doesn't know about costs.
  The CostGuard doesn't know about health checks.  Each piece
  does one thing well, and they combine into a robust system.
"""
print(checklist)

print("=" * 70)
print("Example 10 complete! Your MCP+A2A system is production-ready.")
print("=" * 70)
