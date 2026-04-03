import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 13: Failure Recovery — Circuit Breakers & Fallback Chains
==================================================================
In production, tools and APIs WILL fail. The question isn't "if"
but "when." This example shows how to build resilient agents that
handle failures gracefully instead of crashing.

Two key patterns:
  1. CIRCUIT BREAKER — detects repeated failures and temporarily
     disables a component, preventing cascading failures.
  2. FALLBACK CHAIN — tries backup tools/strategies when the
     primary one fails.

Think of a power grid: when a plant goes offline, the grid
reroutes electricity from other plants. If a whole region fails,
it isolates that region and activates backup generators.

Key Concepts (Section 14 of the Research Bible):
  - Circuit breaker states: CLOSED → OPEN → HALF_OPEN
  - Fallback chains with priority ordering
  - Graceful degradation vs. hard failure

Run: python week-04-advanced-patterns/examples/example_13_failure_recovery.py
"""

import os
import time
import json
from functools import wraps
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# PART 1: Circuit Breaker Pattern (Pure Python)
# ================================================================
# A circuit breaker has three states:
#
#   CLOSED (normal)     → requests flow through, failures counted
#   OPEN (tripped)      → all requests blocked, use fallback
#   HALF_OPEN (testing) → one test request allowed through
#
#   ┌────────┐  failures >= threshold  ┌────────┐
#   │ CLOSED │───────────────────────>│  OPEN  │
#   │(normal)│                        │(blocked)│
#   └────────┘                        └────┬───┘
#       ^                                  │
#       │ test succeeds                    │ timeout elapsed
#       │                            ┌────┴────┐
#       └─────────────────────────── │HALF_OPEN│
#                                    │(testing)│
#                test fails ──────>  └─────────┘
#                (back to OPEN)

class CircuitBreaker:
    """Prevents cascading failures from a repeatedly failing component.

    Usage:
        breaker = CircuitBreaker(max_failures=3, reset_timeout=10)
        result = breaker.call(my_tool, "query", fallback=backup_tool)
    """

    def __init__(self, name: str = "default", max_failures: int = 3,
                 reset_timeout: float = 10.0):
        self.name = name
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0

    def call(self, func, *args, fallback=None, **kwargs):
        """Call a function through the circuit breaker.

        If the circuit is open, uses the fallback instead.
        If no fallback, raises CircuitOpenError.
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                print(f"    [BREAKER:{self.name}] HALF_OPEN — testing recovery")
            else:
                print(f"    [BREAKER:{self.name}] OPEN — using fallback")
                if fallback:
                    return fallback(*args, **kwargs)
                raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN (no fallback)")

        try:
            result = func(*args, **kwargs)
            # Success: reset counter, close circuit
            if self.state == "HALF_OPEN":
                print(f"    [BREAKER:{self.name}] Test passed — CLOSED")
            self.failure_count = 0
            self.state = "CLOSED"
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            print(f"    [BREAKER:{self.name}] Failure #{self.failure_count}: {e}")

            if self.failure_count >= self.max_failures:
                self.state = "OPEN"
                print(f"    [BREAKER:{self.name}] → OPEN (threshold reached)")

            if fallback:
                print(f"    [BREAKER:{self.name}] Using fallback")
                return fallback(*args, **kwargs)
            raise

    def status(self) -> str:
        return f"CircuitBreaker({self.name}): state={self.state}, failures={self.failure_count}/{self.max_failures}"


def demo_circuit_breaker():
    """Show the circuit breaker in action."""
    print("=" * 60)
    print("PART 1: Circuit Breaker Pattern")
    print("=" * 60)

    # Simulated unreliable tool
    call_count = 0

    def unreliable_search(query: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ConnectionError(f"API timeout (attempt {call_count})")
        return f"Results for '{query}': AI is transforming..."

    def backup_search(query: str) -> str:
        return f"[CACHED] Basic results for '{query}': General information about the topic."

    breaker = CircuitBreaker(name="search_api", max_failures=3, reset_timeout=1)

    # Call 1-3: Failures, counting up
    print("\n--- Calls 1-3: Primary API fails ---")
    for i in range(3):
        result = breaker.call(unreliable_search, "AI healthcare", fallback=backup_search)
        print(f"  Call {i+1}: {result[:60]}...")
        print(f"  Status: {breaker.status()}")
        print()

    # Call 4: Circuit is OPEN, goes straight to fallback
    print("--- Call 4: Circuit OPEN, using fallback directly ---")
    result = breaker.call(unreliable_search, "AI healthcare", fallback=backup_search)
    print(f"  Call 4: {result[:60]}...")
    print(f"  Status: {breaker.status()}")
    print()

    # Wait for reset timeout, then test recovery
    print("--- Waiting for reset timeout... ---")
    time.sleep(1.1)

    # Call 5: Circuit transitions to HALF_OPEN, tests one request
    print("--- Call 5: Testing recovery (HALF_OPEN) ---")
    call_count = 10  # Make API "recover"
    result = breaker.call(unreliable_search, "AI healthcare", fallback=backup_search)
    print(f"  Call 5: {result[:60]}...")
    print(f"  Status: {breaker.status()}")
    print()


# ================================================================
# PART 2: Fallback Chain Pattern (Pure Python)
# ================================================================
# A fallback chain defines a priority-ordered list of alternatives.
# Try the best option first; if it fails, try the next one.
#
#   Primary Tool ──┐
#                  │ fails
#   Backup Tool 1 ─┤
#                  │ fails
#   Backup Tool 2 ─┤
#                  │ fails
#   Cached Results ┘
#                  │ all fail
#   Graceful Error Message

class FallbackChain:
    """Tries tools in priority order until one succeeds.

    Usage:
        chain = FallbackChain()
        chain.add("primary", primary_search)
        chain.add("backup", backup_search)
        chain.add("cache", cache_lookup)
        result = chain.execute("query")
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.tools = []  # List of (name, function) tuples

    def add(self, tool_name: str, func):
        """Add a tool to the chain (order matters — first is highest priority)."""
        self.tools.append((tool_name, func))
        return self  # Allow chaining

    def execute(self, *args, **kwargs) -> dict:
        """Try each tool until one succeeds."""
        for tool_name, func in self.tools:
            try:
                result = func(*args, **kwargs)
                if result:  # Non-empty result
                    return {
                        "success": True,
                        "tool_used": tool_name,
                        "result": result,
                    }
            except Exception as e:
                print(f"    [{tool_name}] Failed: {e}")

        return {
            "success": False,
            "tool_used": None,
            "result": "All tools in the fallback chain failed.",
        }


def demo_fallback_chain():
    """Show the fallback chain in action."""
    print("=" * 60)
    print("PART 2: Fallback Chain Pattern")
    print("=" * 60)

    # Define tools with different reliability
    def premium_search(query: str) -> str:
        raise ConnectionError("Premium API is down")

    def free_search(query: str) -> str:
        if "complex" in query.lower():
            raise ValueError("Free API can't handle complex queries")
        return f"[FREE] Results for '{query}': AI technology overview..."

    def cache_lookup(query: str) -> str:
        cache = {
            "AI healthcare": "[CACHED] AI is used in diagnostics, treatment planning...",
            "renewable energy": "[CACHED] Solar and wind power have grown 300%...",
        }
        for key, value in cache.items():
            if key.lower() in query.lower():
                return value
        return ""  # Cache miss

    chain = FallbackChain("search")
    chain.add("premium_api", premium_search)
    chain.add("free_api", free_search)
    chain.add("local_cache", cache_lookup)

    # Test 1: Premium fails, free succeeds
    print("\n--- Test 1: Simple query ---")
    result = chain.execute("AI healthcare")
    print(f"  Tool used: {result['tool_used']}")
    print(f"  Result: {result['result'][:60]}...")

    # Test 2: Premium and free fail, cache succeeds
    print("\n--- Test 2: Complex query (free API can't handle) ---")
    result = chain.execute("complex AI healthcare analysis")
    print(f"  Tool used: {result['tool_used']}")
    print(f"  Result: {result['result'][:60]}...")

    # Test 3: All fail
    print("\n--- Test 3: Unknown topic (all fail) ---")
    result = chain.execute("complex quantum computing")
    print(f"  Tool used: {result['tool_used']}")
    print(f"  Result: {result['result'][:60]}...")
    print()


# ================================================================
# PART 3: LangGraph Agent with Fallback Tools
# ================================================================
# Shows how to integrate failure recovery into a LangGraph agent.
# Uses simulated tools — no real API calls.

def demo_langgraph_fallback():
    """Show failure recovery in a LangGraph-style agent flow."""
    print("=" * 60)
    print("PART 3: Failure Recovery in Agent Flow")
    print("=" * 60)

    # Simulated agent state
    state = {
        "query": "What is the impact of AI on healthcare?",
        "tool_failures": {},
        "results": [],
    }

    # Tools with simulated reliability
    def tavily_search(query):
        raise ConnectionError("Tavily API rate limit exceeded")

    def brave_search(query):
        return f"[Brave] AI healthcare: diagnostic accuracy improved 40%, drug discovery 60% faster"

    def duckduckgo_search(query):
        return f"[DDG] AI in healthcare includes diagnostics, treatment, and research"

    # Fallback chain for search
    search_chain = FallbackChain("search")
    search_chain.add("tavily", tavily_search)
    search_chain.add("brave", brave_search)
    search_chain.add("duckduckgo", duckduckgo_search)

    # Agent execution with fallback
    print(f"\n  Query: {state['query']}")
    print()

    # Step 1: Try to search
    print("  Step 1: Search for information")
    search_result = search_chain.execute(state["query"])
    if search_result["success"]:
        state["results"].append(search_result["result"])
        print(f"  ✓ Got results from {search_result['tool_used']}")
    else:
        print(f"  ✗ All search tools failed — returning graceful error")
        state["results"].append("Unable to search. Answering from general knowledge.")

    # Step 2: Synthesize (simulated)
    print("\n  Step 2: Synthesize results")
    synthesis = f"Based on research: {' '.join(state['results'])}"
    print(f"  ✓ Final output: {synthesis[:80]}...")
    print()


# ================================================================
# PART 4: Pattern Trade-offs
# ================================================================

def demo_tradeoffs():
    """Discuss when to use each pattern."""
    print("=" * 60)
    print("PART 4: Pattern Trade-offs (Topic 16)")
    print("=" * 60)
    print("""
┌──────────────────┬────────────────────┬──────────────────────┐
│ Pattern          │ Best For           │ Overhead             │
├──────────────────┼────────────────────┼──────────────────────┤
│ Circuit Breaker  │ APIs with known    │ Low (counter +       │
│                  │ failure modes      │ timer per component) │
├──────────────────┼────────────────────┼──────────────────────┤
│ Fallback Chain   │ Multiple providers │ Medium (need to      │
│                  │ for same capability│ maintain alternatives)│
├──────────────────┼────────────────────┼──────────────────────┤
│ Simple Retry     │ Transient errors   │ Very Low (just a     │
│                  │ (timeouts, 503s)   │ loop with backoff)   │
├──────────────────┼────────────────────┼──────────────────────┤
│ Graceful         │ Non-critical       │ Low (return partial  │
│ Degradation      │ features           │ result with warning) │
└──────────────────┴────────────────────┴──────────────────────┘

Decision guide:
  - Single API that sometimes times out? → Simple retry (3x with backoff)
  - Critical API with known outage patterns? → Circuit breaker
  - Multiple equivalent data sources? → Fallback chain
  - Nice-to-have feature (e.g., image generation)? → Graceful degradation

Rule of thumb: Start with simple retry. Add circuit breaker when you
see repeated failures in Phoenix traces. Add fallback chain when you
have alternative data sources.
""")


# ================================================================
# Main: Run all demos
# ================================================================

if __name__ == "__main__":
    print()
    print("Example 13: Failure Recovery — Circuit Breakers & Fallback Chains")
    print("=" * 60)
    print()

    demo_circuit_breaker()
    demo_fallback_chain()
    demo_langgraph_fallback()
    demo_tradeoffs()

    print("=" * 60)
    print("Key Takeaway: Design for failure from the start.")
    print("Circuit breakers prevent cascading failures.")
    print("Fallback chains ensure graceful degradation.")
    print("=" * 60)
