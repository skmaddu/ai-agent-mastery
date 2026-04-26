import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 11: Evaluation — Phoenix for End-to-End Evals
======================================================
Topic 11 — Measuring and improving your MCP + A2A system with
structured evaluation, LLM-as-judge scoring, and Phoenix tracing.

The BIG IDEA (Feynman):
  You can't improve what you can't measure.  Phoenix is like a medical
  checkup for your AI system — it shows you exactly where the problems
  are.

  Imagine you bake cookies but never taste them.  You might think
  they're perfect, but your guests spit them out.  Evaluation is
  tasting your own cookies — systematically, with a scorecard:
    - How long did they take to bake? (latency)
    - Did they come out right? (quality)
    - How much flour did you use? (cost)
    - Did any burn? (error rate)

  This example builds a complete evaluation framework and shows how
  to use Phoenix to track everything.

Previously covered:
  - MCP client/server (examples 03-05)
  - A2A protocol (examples 06-08)
  - Production polish (example 10)

Run: python week-07-mcp-a2a-synthesis/examples/example_11_evaluation_phoenix.py
"""

import os
import json
import asyncio
import time
import statistics
from typing import Optional, Any
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
# Phoenix Integration (optional — works without it)
# ================================================================
#
# Phoenix provides:
#   1. Trace visualization — see every LLM call, tool call, latency
#   2. Eval storage — store quality scores alongside traces
#   3. Dashboards — aggregate metrics over time
#
# If Phoenix isn't installed, we still collect all the same data
# internally and display it in text tables.

try:
    import phoenix as px
    from opentelemetry import trace
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

print("=" * 70)
print("Example 11: Evaluation Framework with Phoenix")
print("=" * 70)
print(f"\nPhoenix available: {PHOENIX_AVAILABLE}")
if not PHOENIX_AVAILABLE:
    print("  (Install with: pip install arize-phoenix opentelemetry-api)")
    print("  All eval features work without Phoenix — just no dashboard.\n")
else:
    print("  Phoenix dashboard: http://localhost:6006\n")


# ================================================================
# PART 1: SPAN TRACKERS — Recording What Happened
# ================================================================
#
# A "span" is a single unit of work with a start time and end time.
# Just like a stopwatch for each step of your pipeline.
#
# We track TWO kinds of spans:
#   1. MCP spans — tool calls to MCP servers
#   2. A2A spans — task delegations between agents
#
# Together, they form a complete picture of what your system did
# and how long each piece took.

print("=" * 70)
print("PART 1: Span Trackers — Stopwatch for Every Operation")
print("=" * 70)


@dataclass
class MCPSpan:
    """
    Records a single MCP tool call.

    Like a receipt from a store: what you bought (tool_name),
    when (timestamps), whether it worked (success), how long
    it took (latency), and how much it cost (token_count).
    """
    tool_name: str
    start_time: float
    end_time: float = 0.0
    success: bool = True
    error_message: str = ""
    token_count: int = 0
    request_payload: dict = field(default_factory=dict)
    response_preview: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class MCPSpanTracker:
    """
    Creates and manages OpenTelemetry-style span records for MCP tool calls.

    Usage:
        tracker = MCPSpanTracker()
        span = tracker.start_span("search_db", {"query": "AI"})
        # ... do the work ...
        tracker.end_span(span, success=True, tokens=150)
    """

    def __init__(self):
        self.spans: list[MCPSpan] = []
        self._otel_tracer = None

        # If Phoenix is available, create a real OTel tracer
        if PHOENIX_AVAILABLE:
            try:
                self._otel_tracer = trace.get_tracer("mcp-eval")
            except Exception:
                pass

    def start_span(self, tool_name: str, payload: dict = None) -> MCPSpan:
        """Begin tracking an MCP tool call."""
        span = MCPSpan(
            tool_name=tool_name,
            start_time=time.time(),
            request_payload=payload or {},
        )
        self.spans.append(span)
        return span

    def end_span(self, span: MCPSpan, success: bool = True,
                 tokens: int = 0, response: str = "",
                 error: str = ""):
        """Finish tracking an MCP tool call."""
        span.end_time = time.time()
        span.success = success
        span.token_count = tokens
        span.response_preview = response[:200]
        span.error_message = error

        # Also record in Phoenix if available
        if self._otel_tracer:
            try:
                with self._otel_tracer.start_as_current_span(
                    f"mcp.{span.tool_name}"
                ) as otel_span:
                    otel_span.set_attribute("mcp.tool_name", span.tool_name)
                    otel_span.set_attribute("mcp.success", success)
                    otel_span.set_attribute("mcp.latency_ms", span.latency_ms)
                    otel_span.set_attribute("mcp.tokens", tokens)
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Aggregate statistics across all recorded spans."""
        if not self.spans:
            return {"count": 0}

        latencies = [s.latency_ms for s in self.spans]
        successes = sum(1 for s in self.spans if s.success)
        total_tokens = sum(s.token_count for s in self.spans)

        return {
            "count": len(self.spans),
            "success_rate": round(successes / len(self.spans) * 100, 1),
            "total_tokens": total_tokens,
            "latency_p50": round(statistics.median(latencies), 1),
            "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
                           if len(latencies) >= 2 else round(latencies[0], 1),
            "latency_p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 1)
                           if len(latencies) >= 2 else round(latencies[0], 1),
        }


@dataclass
class A2ASpan:
    """
    Records a single A2A task operation.

    Tracks the full lifecycle: submitted -> working -> completed/failed
    Each state transition is recorded with a timestamp.
    """
    task_id: str
    agent_name: str
    start_time: float
    end_time: float = 0.0
    state_transitions: list = field(default_factory=list)
    success: bool = True
    token_count: int = 0
    error_message: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def transition(self, new_state: str):
        """Record a state transition with timestamp."""
        self.state_transitions.append({
            "state": new_state,
            "timestamp": time.time(),
        })


class A2ASpanTracker:
    """
    Creates and manages span records for A2A task operations.

    Tracks the full A2A task lifecycle including state transitions
    (submitted -> working -> completed/failed).
    """

    def __init__(self):
        self.spans: list[A2ASpan] = []
        self._otel_tracer = None

        if PHOENIX_AVAILABLE:
            try:
                self._otel_tracer = trace.get_tracer("a2a-eval")
            except Exception:
                pass

    def start_span(self, task_id: str, agent_name: str) -> A2ASpan:
        """Begin tracking an A2A task."""
        span = A2ASpan(
            task_id=task_id,
            agent_name=agent_name,
            start_time=time.time(),
        )
        span.transition("submitted")
        self.spans.append(span)
        return span

    def end_span(self, span: A2ASpan, success: bool = True,
                 tokens: int = 0, error: str = ""):
        """Finish tracking an A2A task."""
        span.end_time = time.time()
        span.success = success
        span.token_count = tokens
        span.error_message = error
        final_state = "completed" if success else "failed"
        span.transition(final_state)

        if self._otel_tracer:
            try:
                with self._otel_tracer.start_as_current_span(
                    f"a2a.{span.agent_name}"
                ) as otel_span:
                    otel_span.set_attribute("a2a.task_id", span.task_id)
                    otel_span.set_attribute("a2a.agent", span.agent_name)
                    otel_span.set_attribute("a2a.success", success)
                    otel_span.set_attribute("a2a.latency_ms", span.latency_ms)
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Aggregate statistics across all A2A spans."""
        if not self.spans:
            return {"count": 0}

        latencies = [s.latency_ms for s in self.spans]
        successes = sum(1 for s in self.spans if s.success)
        total_tokens = sum(s.token_count for s in self.spans)

        return {
            "count": len(self.spans),
            "success_rate": round(successes / len(self.spans) * 100, 1),
            "total_tokens": total_tokens,
            "latency_p50": round(statistics.median(latencies), 1),
            "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
                           if len(latencies) >= 2 else round(latencies[0], 1),
        }


# Demo: Span Trackers
print("\nRecording MCP spans...")
mcp_tracker = MCPSpanTracker()

for tool in ["search_db", "get_weather", "translate"]:
    span = mcp_tracker.start_span(tool, {"input": "demo"})
    time.sleep(0.02)  # Simulate work
    mcp_tracker.end_span(
        span,
        success=(tool != "translate"),  # translate "fails" for demo
        tokens=150 if tool != "translate" else 0,
        response=f"Result from {tool}" if tool != "translate" else "",
        error="Timeout" if tool == "translate" else "",
    )

print(f"  MCP stats: {json.dumps(mcp_tracker.get_stats(), indent=2)}")

print("\nRecording A2A spans...")
a2a_tracker = A2ASpanTracker()

for agent in ["researcher", "writer"]:
    span = a2a_tracker.start_span(f"task-{agent}", agent)
    span.transition("working")
    time.sleep(0.03)
    a2a_tracker.end_span(span, success=True, tokens=300)

print(f"  A2A stats: {json.dumps(a2a_tracker.get_stats(), indent=2)}")


# ================================================================
# PART 2: EVAL TEST CASES & SUITE
# ================================================================
#
# Evaluation = running your system on known inputs and checking
# whether the outputs are good enough.
#
# Each test case defines:
#   - A query (what to ask the system)
#   - Expected keywords (what should appear in the answer)
#   - Max latency (how fast it should respond)
#   - Min quality score (how good the answer should be, 0-10)
#
# The eval suite runs ALL test cases and collects metrics.

print("\n" + "=" * 70)
print("PART 2: Evaluation Test Cases & Suite")
print("=" * 70)


class EvalTestCase(BaseModel):
    """
    A single evaluation test case.

    Think of it like a quiz question with an answer key:
    - The query is the question
    - Expected keywords are the answer key
    - Max latency and min quality are the grading rubric
    """
    id: str = Field(..., description="Unique test case identifier")
    query: str = Field(..., description="Input query to the system")
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that should appear in the response"
    )
    max_latency_ms: float = Field(
        default=5000.0, description="Maximum acceptable latency in ms"
    )
    min_quality_score: float = Field(
        default=7.0, ge=0, le=10,
        description="Minimum acceptable quality score (0-10)"
    )
    category: str = Field(
        default="general", description="Test category for grouping"
    )


class EvalResult(BaseModel):
    """Result of running a single test case."""
    test_id: str
    query: str
    response: str = ""
    latency_ms: float = 0.0
    quality_score: float = 0.0
    keyword_hits: int = 0
    keyword_total: int = 0
    passed: bool = False
    failure_reasons: list[str] = Field(default_factory=list)


class QualityScorer:
    """
    Uses LLM-as-judge to score response quality on a 0-10 scale.

    The idea: ask a separate LLM to evaluate whether the response
    is good.  This is like having a teacher grade a student's exam.

    The judge LLM gets:
      - The original question
      - The system's response
      - A rubric (what makes a good answer)

    And returns a score from 0 (terrible) to 10 (excellent).
    """

    JUDGE_PROMPT = """You are an evaluation judge. Score the following response
on a scale of 0-10 based on these criteria:
- Relevance: Does it answer the question? (0-3 points)
- Accuracy: Is the information correct? (0-3 points)
- Completeness: Does it cover the key aspects? (0-2 points)
- Clarity: Is it well-written and easy to understand? (0-2 points)

Question: {query}

Response: {response}

Return ONLY a JSON object: {{"score": <number>, "reasoning": "<brief explanation>"}}"""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    async def score(self, query: str, response: str) -> tuple[float, str]:
        """
        Score a response. Returns (score, reasoning).
        Falls back to keyword heuristic if LLM is unavailable.
        """
        if self.use_llm:
            try:
                return await self._llm_score(query, response)
            except Exception as e:
                print(f"    [Scorer] LLM judge failed ({e}), using heuristic")
                return self._heuristic_score(query, response)
        return self._heuristic_score(query, response)

    async def _llm_score(self, query: str, response: str) -> tuple[float, str]:
        """Use an LLM to score the response."""
        llm = self._get_llm()
        prompt = self.JUDGE_PROMPT.format(query=query, response=response)
        result = await llm.ainvoke(prompt)
        content = result.content.strip()

        # Parse JSON from the LLM response
        # Handle cases where LLM wraps in markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 5))
            reasoning = parsed.get("reasoning", "No reasoning provided")
            return min(10.0, max(0.0, score)), reasoning
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, try to extract a number
            import re
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
            if numbers:
                score = float(numbers[0])
                return min(10.0, max(0.0, score)), content[:100]
            return 5.0, "Could not parse LLM judge response"

    def _heuristic_score(self, query: str, response: str) -> tuple[float, str]:
        """
        Simple heuristic scoring when LLM is not available.
        Checks response length, query word overlap, and structure.
        """
        score = 0.0
        reasons = []

        # Length check (longer responses tend to be more complete)
        if len(response) > 200:
            score += 3.0
            reasons.append("Good length")
        elif len(response) > 50:
            score += 1.5
            reasons.append("Moderate length")
        else:
            reasons.append("Too short")

        # Query relevance (do query words appear in response?)
        query_words = set(query.lower().split())
        response_lower = response.lower()
        overlap = sum(1 for w in query_words if w in response_lower)
        relevance = overlap / max(len(query_words), 1)
        score += relevance * 4.0
        reasons.append(f"Relevance: {relevance:.0%}")

        # Structure check (has sentences, not just fragments)
        if "." in response and len(response.split(".")) > 2:
            score += 2.0
            reasons.append("Well-structured")

        # Cap at 10
        score = min(10.0, score)
        return round(score, 1), "; ".join(reasons)


class EvalSuite:
    """
    Runs a list of test cases and collects metrics.

    This is the main evaluation engine.  It:
    1. Runs each test case through a provided system function
    2. Scores the response with QualityScorer
    3. Checks keyword presence
    4. Checks latency
    5. Determines pass/fail
    6. Collects all results for reporting
    """

    def __init__(self, test_cases: list[EvalTestCase],
                 scorer: Optional[QualityScorer] = None):
        self.test_cases = test_cases
        self.scorer = scorer or QualityScorer(use_llm=False)
        self.results: list[EvalResult] = []

    async def run(self, system_fn) -> list[EvalResult]:
        """
        Run all test cases through the system function.

        Args:
            system_fn: async function that takes a query string
                       and returns a response string.
        """
        self.results = []
        for tc in self.test_cases:
            print(f"\n  Running test '{tc.id}'...")
            result = await self._run_single(tc, system_fn)
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"    {status} | Quality: {result.quality_score:.1f}/10 "
                  f"| Latency: {result.latency_ms:.0f}ms "
                  f"| Keywords: {result.keyword_hits}/{result.keyword_total}")
        return self.results

    async def _run_single(self, tc: EvalTestCase, system_fn) -> EvalResult:
        """Run a single test case."""
        result = EvalResult(
            test_id=tc.id,
            query=tc.query,
            keyword_total=len(tc.expected_keywords),
        )

        # Run the system and measure latency
        start = time.time()
        try:
            response = await system_fn(tc.query)
            result.response = response
        except Exception as e:
            result.response = f"ERROR: {e}"
            result.failure_reasons.append(f"System error: {e}")
        result.latency_ms = (time.time() - start) * 1000

        # Check keywords
        response_lower = result.response.lower()
        result.keyword_hits = sum(
            1 for kw in tc.expected_keywords
            if kw.lower() in response_lower
        )

        # Score quality
        result.quality_score, reasoning = await self.scorer.score(
            tc.query, result.response
        )

        # Determine pass/fail
        failures = []
        if result.latency_ms > tc.max_latency_ms:
            failures.append(
                f"Latency {result.latency_ms:.0f}ms > {tc.max_latency_ms:.0f}ms"
            )
        if result.quality_score < tc.min_quality_score:
            failures.append(
                f"Quality {result.quality_score:.1f} < {tc.min_quality_score:.1f}"
            )
        if tc.expected_keywords and result.keyword_hits == 0:
            failures.append("No expected keywords found")

        result.failure_reasons = failures
        result.passed = len(failures) == 0
        return result


# ================================================================
# PART 3: EVAL REPORT — Making Sense of the Numbers
# ================================================================
#
# Raw numbers are useless without context.  The EvalReport turns
# a list of results into actionable insights:
#   - Latency percentiles: p50 (typical), p95 (slow), p99 (worst)
#   - Success rate: what % of test cases passed?
#   - Quality distribution: how many scored 8+, 5-7, <5?
#   - Cost breakdown: how much did the eval run cost?

print("\n" + "=" * 70)
print("PART 3: Evaluation Report Generation")
print("=" * 70)


class EvalReport:
    """
    Generates a comprehensive evaluation report.

    Like a doctor reading blood test results:
    - What's healthy (passed tests)
    - What needs attention (failed tests)
    - What's the trend (aggregate metrics)
    """

    def __init__(self, results: list[EvalResult],
                 cost_per_1k_tokens: float = 0.00005):
        self.results = results
        self.cost_per_1k_tokens = cost_per_1k_tokens

    def generate(self) -> dict:
        """Generate the full report as a dictionary."""
        if not self.results:
            return {"error": "No results to report"}

        latencies = [r.latency_ms for r in self.results]
        qualities = [r.quality_score for r in self.results]
        passed = sum(1 for r in self.results if r.passed)

        # Latency percentiles
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        latency_stats = {
            "p50": round(sorted_lat[int(n * 0.50)], 1) if n > 0 else 0,
            "p95": round(sorted_lat[min(int(n * 0.95), n - 1)], 1) if n > 0 else 0,
            "p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 1) if n > 0 else 0,
            "mean": round(statistics.mean(latencies), 1),
        }

        # Quality distribution
        quality_dist = {
            "excellent (8-10)": sum(1 for q in qualities if q >= 8),
            "good (5-7.9)": sum(1 for q in qualities if 5 <= q < 8),
            "poor (<5)": sum(1 for q in qualities if q < 5),
        }

        # Estimated cost (based on response lengths as token proxy)
        est_tokens = sum(len(r.response.split()) * 1.3 for r in self.results)
        est_cost = (est_tokens / 1000) * self.cost_per_1k_tokens

        return {
            "summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "failed": len(self.results) - passed,
                "pass_rate": f"{passed / len(self.results) * 100:.1f}%",
            },
            "latency": latency_stats,
            "quality": {
                "mean": round(statistics.mean(qualities), 1),
                "min": round(min(qualities), 1),
                "max": round(max(qualities), 1),
                "distribution": quality_dist,
            },
            "cost": {
                "estimated_tokens": int(est_tokens),
                "estimated_cost_usd": round(est_cost, 6),
            },
            "failures": [
                {
                    "test_id": r.test_id,
                    "reasons": r.failure_reasons,
                    "quality_score": r.quality_score,
                }
                for r in self.results if not r.passed
            ],
        }

    def print_report(self):
        """Print a formatted text report."""
        report = self.generate()
        s = report["summary"]
        lat = report["latency"]
        q = report["quality"]
        c = report["cost"]

        print("\n  ============================================")
        print("  EVALUATION REPORT")
        print("  ============================================")

        # Summary
        print(f"\n  Tests: {s['total_tests']}  |  "
              f"Passed: {s['passed']}  |  "
              f"Failed: {s['failed']}  |  "
              f"Pass Rate: {s['pass_rate']}")

        # Latency table
        print("\n  --- Latency (ms) ---")
        print(f"  {'Metric':<10} {'Value':>10}")
        print(f"  {'p50':<10} {lat['p50']:>10.1f}")
        print(f"  {'p95':<10} {lat['p95']:>10.1f}")
        print(f"  {'p99':<10} {lat['p99']:>10.1f}")
        print(f"  {'mean':<10} {lat['mean']:>10.1f}")

        # Quality table
        print("\n  --- Quality Scores ---")
        print(f"  Mean: {q['mean']:.1f}  |  Min: {q['min']:.1f}  |  Max: {q['max']:.1f}")
        for bucket, count in q["distribution"].items():
            bar = "#" * (count * 4)
            print(f"  {bucket:<18} {count:>3}  {bar}")

        # Cost
        print(f"\n  --- Cost ---")
        print(f"  Estimated tokens: {c['estimated_tokens']}")
        print(f"  Estimated cost:   ${c['estimated_cost_usd']:.6f}")

        # Per-test detail
        print("\n  --- Per-Test Results ---")
        print(f"  {'ID':<20} {'Status':<6} {'Quality':>8} {'Latency':>10} {'Keywords':>10}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            kw = f"{r.keyword_hits}/{r.keyword_total}"
            print(f"  {r.test_id:<20} {status:<6} {r.quality_score:>8.1f} "
                  f"{r.latency_ms:>8.0f}ms {kw:>10}")

        # Failures
        if report["failures"]:
            print("\n  --- Failure Details ---")
            for f in report["failures"]:
                print(f"  {f['test_id']}: {', '.join(f['reasons'])}")

        # Iteration guidance
        print("\n  --- Iteration Guidance ---")
        if q["mean"] < 7:
            print("  [ACTION] Quality < 7: Adjust system prompt for better responses")
            print("           Consider: more specific instructions, examples in prompt,")
            print("           or switching to a more capable model")
        if lat["p95"] > 5000:
            print("  [ACTION] p95 latency > 5s: Check MCP server performance")
            print("           Consider: connection pooling, caching, server scaling")
        if q["mean"] >= 7 and lat["p95"] <= 5000:
            print("  [OK] System is performing well! Monitor for regression.")

        print("\n  ============================================")


# ================================================================
# PART 4: FULL DEMO — Run Eval Suite End-to-End
# ================================================================
#
# We'll create:
#   1. A simulated "system" that answers queries
#   2. Five test cases with different expectations
#   3. Run the suite and generate a report
#
# This demonstrates the complete evaluation workflow:
#   Define test cases -> Run system -> Score responses ->
#   Generate report -> Decide what to improve

print("\n" + "=" * 70)
print("PART 4: Running the Full Evaluation")
print("=" * 70)

# Define test cases
TEST_CASES = [
    EvalTestCase(
        id="mcp-basics",
        query="What is the Model Context Protocol (MCP) and why is it useful?",
        expected_keywords=["protocol", "tools", "server", "client"],
        max_latency_ms=3000,
        min_quality_score=6.0,
        category="knowledge",
    ),
    EvalTestCase(
        id="a2a-basics",
        query="Explain the Agent-to-Agent (A2A) protocol and its key concepts.",
        expected_keywords=["agent", "task", "communication", "protocol"],
        max_latency_ms=3000,
        min_quality_score=6.0,
        category="knowledge",
    ),
    EvalTestCase(
        id="mcp-vs-a2a",
        query="What is the difference between MCP and A2A protocols?",
        expected_keywords=["tools", "agents", "different"],
        max_latency_ms=2000,
        min_quality_score=7.0,
        category="comparison",
    ),
    EvalTestCase(
        id="retry-pattern",
        query="How should you handle failures in MCP tool calls?",
        expected_keywords=["retry", "error", "backoff"],
        max_latency_ms=2000,
        min_quality_score=5.0,
        category="patterns",
    ),
    EvalTestCase(
        id="cost-control",
        query="What strategies exist for controlling LLM costs in production?",
        expected_keywords=["budget", "token", "limit"],
        max_latency_ms=2500,
        min_quality_score=6.0,
        category="production",
    ),
]


# Simulated system function (in a real eval, this calls your actual system)
async def simulated_system(query: str) -> str:
    """
    Simulates a system response.

    In a real evaluation, this would call your actual MCP+A2A pipeline.
    We simulate it here to demonstrate the eval framework without
    requiring API keys or running servers.
    """
    # Simulate variable latency
    await asyncio.sleep(0.05 + (hash(query) % 100) / 1000)

    # Pre-built responses for our test cases
    responses = {
        "mcp": (
            "The Model Context Protocol (MCP) is an open standard that defines "
            "how AI applications communicate with external tools and data sources. "
            "It uses a client-server architecture where the AI application is the "
            "client and tools are exposed through MCP servers. This protocol "
            "standardizes tool discovery, invocation, and response handling, "
            "making it easy to plug different tools into any AI agent. "
            "Think of it like USB for AI — any tool that speaks MCP can connect "
            "to any agent that speaks MCP."
        ),
        "a2a": (
            "The Agent-to-Agent (A2A) protocol enables communication between "
            "AI agents. Its key concepts include: Agent Cards (capability "
            "descriptions), Tasks (units of work with state machines), and "
            "message-based communication. A2A lets specialized agents "
            "collaborate — a research agent can delegate writing to a writer "
            "agent without knowing its internal implementation. The protocol "
            "defines standard task states: submitted, working, completed, failed."
        ),
        "difference": (
            "MCP and A2A serve different purposes. MCP connects agents to tools "
            "(like a database or API), while A2A connects agents to other agents. "
            "MCP is vertical (agent-to-tool), A2A is horizontal (agent-to-agent). "
            "They are complementary: an agent uses MCP to access tools and A2A "
            "to collaborate with other agents."
        ),
        "failure": (
            "MCP tool call failures should be handled with retry logic using "
            "exponential backoff. Start with a 1-second wait, then double it "
            "on each retry attempt, up to a maximum of 3 retries. Only retry "
            "on transient errors like connection timeouts or server errors "
            "(5xx). Permanent errors (4xx) should not be retried. Use circuit "
            "breakers to stop calling a consistently failing server."
        ),
        "cost": (
            "Key strategies for controlling LLM costs include: setting a token "
            "budget per task and per session, checking the budget before each "
            "LLM call, using cheaper models for simple tasks (like Groq for "
            "drafts), caching frequent responses to avoid duplicate calls, "
            "and limiting output token counts. Monitor actual spend vs budget "
            "and set up alerts when approaching the limit."
        ),
    }

    # Match query to response
    query_lower = query.lower()
    if "mcp" in query_lower and "a2a" in query_lower:
        return responses["difference"]
    elif "mcp" in query_lower:
        return responses["mcp"]
    elif "a2a" in query_lower:
        return responses["a2a"]
    elif "failure" in query_lower or "handle" in query_lower:
        return responses["failure"]
    elif "cost" in query_lower:
        return responses["cost"]
    else:
        return "I don't have information about that topic."


async def run_full_eval():
    """Run the complete evaluation pipeline."""
    print("\nInitializing evaluation suite with 5 test cases...")

    # Use heuristic scorer (no API key needed for demo)
    scorer = QualityScorer(use_llm=False)
    suite = EvalSuite(TEST_CASES, scorer)

    # Run all test cases
    results = await suite.run(simulated_system)

    # Generate and print report
    print("\n\nGenerating evaluation report...")
    report = EvalReport(results)
    report.print_report()

    return report

eval_report = asyncio.run(run_full_eval())


# ================================================================
# PART 5: ITERATION STRATEGY — What To Do With Eval Results
# ================================================================
#
# Evaluation is not a one-time thing.  It's a loop:
#   Run eval -> Find problems -> Fix them -> Run eval again
#
# Common fixes based on eval results:

print("\n" + "=" * 70)
print("PART 5: Iteration Strategy — Turning Numbers Into Improvements")
print("=" * 70)

iteration_guide = """
HOW TO USE EVAL RESULTS TO IMPROVE YOUR SYSTEM:

  +------------------+--------------------------------------+
  | Problem          | Fix                                  |
  +------------------+--------------------------------------+
  | Quality < 7      | Adjust system prompt:                |
  |                  |   - Add specific instructions         |
  |                  |   - Include few-shot examples         |
  |                  |   - Use a more capable model          |
  +------------------+--------------------------------------+
  | Latency > 5s     | Check MCP server:                    |
  |                  |   - Enable connection pooling         |
  |                  |   - Add response caching              |
  |                  |   - Scale server horizontally         |
  +------------------+--------------------------------------+
  | Keywords missing | Improve retrieval:                    |
  |                  |   - Better MCP tool descriptions      |
  |                  |   - Add more relevant tools           |
  |                  |   - Improve prompt grounding           |
  +------------------+--------------------------------------+
  | High error rate  | Fix reliability:                      |
  |                  |   - Add retry logic (example 10)      |
  |                  |   - Improve error handling             |
  |                  |   - Add health checks before calls    |
  +------------------+--------------------------------------+
  | Cost too high    | Reduce token usage:                   |
  |                  |   - Shorter prompts                    |
  |                  |   - Limit output tokens                |
  |                  |   - Cache frequent queries             |
  |                  |   - Use cheaper model for simple tasks |
  +------------------+--------------------------------------+

EVAL LOOP:
  1. Define test cases (once, then expand)
  2. Run eval suite
  3. Read the report
  4. Pick the worst metric
  5. Apply the fix from the table above
  6. Run eval suite again
  7. Compare: did the metric improve?
  8. Repeat until all metrics meet your targets

GOLDEN RULE:
  Never deploy without running your eval suite.
  It's your safety net — like running unit tests before a release.
"""
print(iteration_guide)


# ================================================================
# PART 6: PHOENIX INTEGRATION (when available)
# ================================================================
#
# If Phoenix is installed and running, we can push our eval results
# to the Phoenix dashboard for visualization and historical tracking.

print("=" * 70)
print("PART 6: Phoenix Integration")
print("=" * 70)

if PHOENIX_AVAILABLE:
    print("""
  Phoenix is available! To use it with this eval framework:

  1. Start Phoenix:
       import phoenix as px
       px.launch_app()

  2. Set up tracing in your system code:
       from opentelemetry import trace
       tracer = trace.get_tracer("my-agent")

  3. Wrap your eval suite calls with spans:
       with tracer.start_as_current_span("eval-run"):
           results = await suite.run(system_fn)

  4. View results at: http://localhost:6006
     - Traces tab: see every LLM call, tool call, latency
     - Evals tab: quality scores over time
     - Compare runs to see if changes improved metrics
""")
else:
    print("""
  Phoenix is NOT installed, but that's OK!

  All evaluation features work without Phoenix:
  - Span tracking (MCPSpanTracker, A2ASpanTracker)
  - Quality scoring (QualityScorer)
  - Test suite (EvalSuite)
  - Reports (EvalReport)

  Phoenix adds visualization and historical tracking.
  Install it when you're ready:
    pip install arize-phoenix opentelemetry-api opentelemetry-sdk
""")


# ================================================================
# SUMMARY
# ================================================================

print("=" * 70)
print("SUMMARY: Evaluation Framework Components")
print("=" * 70)
print("""
  What we built:

  1. MCPSpanTracker — Records every MCP tool call with latency & tokens
  2. A2ASpanTracker — Records every A2A task with state transitions
  3. EvalTestCase   — Defines what to test (query, expected keywords,
                      latency limits, quality threshold)
  4. EvalSuite      — Runs all test cases and collects results
  5. QualityScorer  — LLM-as-judge scoring (with heuristic fallback)
  6. EvalReport     — Latency percentiles, quality distribution,
                      cost breakdown, failure analysis

  How they connect:

    [Test Cases] -> [EvalSuite] -> [System Function] -> [Responses]
                         |                                    |
                    [QualityScorer]  <----- score ------------|
                         |
                    [EvalReport] -> print or push to Phoenix

  Next steps:
    - Replace simulated_system with your real MCP+A2A pipeline
    - Add more test cases (aim for 20+ covering edge cases)
    - Run evals in CI/CD to catch regressions automatically
    - Use Phoenix dashboard for historical trend analysis
""")

print("=" * 70)
print("Example 11 complete! You now have a full evaluation framework.")
print("=" * 70)
