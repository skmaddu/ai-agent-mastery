import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5b: Intent Routing — Directing Traffic to the Right Agent
==================================================================
Before specialist agents can do their job, SOMETHING must decide which
specialist to call. That "something" is the intent router — the triage
nurse of a multi-agent system.

Intent routing is the "H" (Handoff) layer of the HARNESS framework:
the very first decision in the pipeline. Without it, every request
goes to a single generalist agent, negating the benefits of specialization.

This example implements FOUR routing approaches, from simplest to most
sophisticated, all solving the same problem: given a user query, pick
the right specialist agent (research, code, summarize).

Approach         | How It Works                      | Pros              | Cons
-----------------+-----------------------------------+-------------------+--------------------
1. Rule-based    | if "SQL" in query -> SQL agent     | Fast, zero cost   | Brittle
2. Keyword/Embed | Semantic similarity to clusters    | More robust       | Needs embeddings
3. LLM Classify  | Ask LLM: "research/code/summary?" | Most flexible     | Adds latency/cost
4. Cascading     | Try cheapest first, escalate       | Cost-efficient    | More complex

WARNING — Router Misclassification Cascades:
  If the router sends a code question to the research agent, the bad
  output wastes tokens in the evaluator-optimizer trying to fix unfixable
  work. Invest heavily in router quality: use few-shot examples, test
  edge cases, and monitor misclassification rate in Phoenix traces.

Run: python week-04-advanced-patterns/examples/example_05b_intent_routing.py
"""

import os
import re
import time
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ==============================================================
# Specialist Agents (simulated)
# ==============================================================
# In a real system, each of these would be a full LangGraph/ADK agent
# with its own tools. Here we simulate them to focus on the ROUTING.

def research_agent(query: str) -> str:
    """Specialist: handles research and fact-finding questions."""
    return (
        f"[RESEARCH AGENT] Researching: '{query}'\n"
        "  Found 3 peer-reviewed sources. Key findings:\n"
        "  - Topic is well-studied with consensus on main points\n"
        "  - Recent developments in 2024 show emerging trends\n"
        "  - Recommended reading: 2 survey papers"
    )


def code_agent(query: str) -> str:
    """Specialist: handles coding, debugging, and SQL questions."""
    return (
        f"[CODE AGENT] Generating code for: '{query}'\n"
        "  ```python\n"
        "  # Solution generated\n"
        "  def solution():\n"
        "      return 'Implementation here'\n"
        "  ```\n"
        "  Tested: all edge cases pass."
    )


def summarize_agent(query: str) -> str:
    """Specialist: handles summarization and writing tasks."""
    return (
        f"[SUMMARIZE AGENT] Summarizing: '{query}'\n"
        "  Executive Summary (3 bullets):\n"
        "  - Main point identified and condensed\n"
        "  - Supporting details organized by importance\n"
        "  - Action items extracted"
    )


def general_agent(query: str) -> str:
    """Fallback: handles anything that doesn't fit a specialist."""
    return (
        f"[GENERAL AGENT] Handling: '{query}'\n"
        "  Processed as general question."
    )


AGENTS = {
    "research": research_agent,
    "code": code_agent,
    "summarize": summarize_agent,
    "general": general_agent,
}


# ==============================================================
# Test Queries (with expected correct routing)
# ==============================================================

TEST_QUERIES = [
    ("What are the latest findings on CRISPR gene therapy?", "research"),
    ("Write a Python function to merge two sorted lists", "code"),
    ("Summarize this 10-page report on climate change", "summarize"),
    ("Fix this SQL query: SELECT * FORM users WHERE id = 1", "code"),
    ("What is the capital of France?", "research"),
    ("Condense these meeting notes into action items", "summarize"),
    ("Debug this error: IndexError: list index out of range", "code"),
    ("Compare the economic policies of two countries", "research"),
]


def evaluate_router(name: str, route_fn, queries=TEST_QUERIES):
    """Run a router against test queries and report accuracy."""
    print(f"\n{'- '*30}")
    print(f"  Testing: {name}")
    print(f"{'- '*30}")

    correct = 0
    total = len(queries)
    results = []

    for query, expected in queries:
        start = time.time()
        predicted = route_fn(query)
        elapsed = time.time() - start

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        marker = "OK" if is_correct else "MISS"

        results.append((query[:55], expected, predicted, marker, elapsed))
        print(f"  [{marker}] '{query[:55]}...' → {predicted} "
              f"(expected: {expected}) [{elapsed:.3f}s]")

    accuracy = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    return accuracy


# ==============================================================
# APPROACH 1: Rule-Based Router
# ==============================================================
# Simplest possible router: keyword matching with if/elif chains.
# Fast and deterministic, but brittle — misses anything not in the rules.

def rule_based_router(query: str) -> str:
    """Route based on keyword matching.

    This is the starting point for any routing system. It works
    surprisingly well for narrow, well-defined domains.
    """
    q = query.lower()

    # Code indicators
    code_keywords = ["python", "function", "code", "sql", "debug", "error",
                     "fix", "write a", "implement", "bug", "indexerror",
                     "typeerror", "syntax"]
    if any(kw in q for kw in code_keywords):
        return "code"

    # Summarize indicators
    summarize_keywords = ["summarize", "summary", "condense", "shorten",
                         "brief", "tldr", "action items", "meeting notes",
                         "digest", "recap"]
    if any(kw in q for kw in summarize_keywords):
        return "summarize"

    # Research indicators
    research_keywords = ["research", "findings", "compare", "analyze",
                        "what is", "what are", "explain", "study",
                        "latest", "history of", "economic", "scientific"]
    if any(kw in q for kw in research_keywords):
        return "research"

    return "general"


# ==============================================================
# APPROACH 2: Keyword Embedding Router (Simulated)
# ==============================================================
# Uses semantic similarity between the query and category centroids.
# More robust than rules because "fix my SQL" matches "code" even
# without the exact keyword "code" in the query.
#
# In production, you'd use sentence-transformers or OpenAI embeddings.
# Here we simulate with keyword overlap scoring.

# Category "centroids" — representative terms for each category
CATEGORY_CENTROIDS = {
    "research": ["research", "study", "findings", "analysis", "compare",
                 "investigate", "discover", "evidence", "data", "trends",
                 "scientific", "academic", "capital", "history", "economic",
                 "policy", "latest", "what"],
    "code": ["code", "python", "function", "debug", "error", "sql",
             "implement", "fix", "bug", "syntax", "programming", "script",
             "query", "class", "method", "write", "merge", "sorted",
             "indexerror", "typeerror", "form"],
    "summarize": ["summarize", "summary", "condense", "brief", "shorten",
                  "digest", "recap", "notes", "action items", "report",
                  "key points", "meeting", "highlights", "tldr", "overview"],
}


def embedding_router(query: str) -> str:
    """Route based on simulated semantic similarity.

    In production, replace this with:
      query_embedding = model.encode(query)
      best = max(centroids, key=lambda c: cosine_sim(query_embedding, c))
    """
    q_words = set(query.lower().split())

    scores = {}
    for category, centroid_words in CATEGORY_CENTROIDS.items():
        # Simulate cosine similarity with word overlap
        centroid_set = set(centroid_words)
        overlap = len(q_words & centroid_set)
        # Bonus for substring matches (catches "SQL" in "Fix this SQL query")
        substring_bonus = sum(
            1 for cw in centroid_words
            if cw in query.lower() and cw not in q_words
        )
        scores[category] = overlap + substring_bonus * 0.5

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


# ==============================================================
# APPROACH 3: LLM Classification Router
# ==============================================================
# Most flexible — the LLM understands nuance, handles edge cases,
# and can be improved with few-shot examples. But adds latency and cost.

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


CLASSIFICATION_PROMPT = """Classify this user query into exactly ONE category.

Categories:
- research: factual questions, comparisons, analysis, "what is", finding information
- code: writing code, debugging, fixing errors, SQL, programming tasks
- summarize: condensing text, summarizing documents, extracting key points

Few-shot examples:
  "What causes inflation?" → research
  "Write a binary search in Java" → code
  "Give me bullet points from this article" → summarize
  "Fix this TypeError in my React app" → code
  "Compare Tesla vs Ford stock performance" → research
  "Shorten this email to 3 sentences" → summarize

Query: {query}

Respond with ONLY the category name (research, code, or summarize). Nothing else."""


def llm_classification_router(query: str) -> str:
    """Route using LLM classification with few-shot examples."""
    from langchain_core.messages import HumanMessage

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=CLASSIFICATION_PROMPT.format(query=query))])
    result = response.content.strip().lower()

    # Parse — the LLM should return just the category name
    for category in ["research", "code", "summarize"]:
        if category in result:
            return category
    return "general"


# ==============================================================
# APPROACH 4: Cascading Router
# ==============================================================
# Try the cheapest router first. If its confidence is low, escalate
# to a more expensive but accurate router. This balances cost and quality.
#
# Flow: rule-based (free) → embedding (free) → LLM (costs tokens)
#       ↓ confident?          ↓ confident?        ↓ always trust
#       YES → done            YES → done           → done

def cascading_router(query: str) -> str:
    """Route using cheapest method first, escalating if unsure.

    Level 1: Rule-based (zero cost, instant)
    Level 2: Embedding similarity (zero cost, instant)
    Level 3: LLM classification (costs tokens, ~0.5s)

    Escalation happens when Level 1 and 2 DISAGREE — that signals
    the query is ambiguous and needs the LLM to break the tie.
    """
    # Level 1: Rule-based
    rule_result = rule_based_router(query)

    # Level 2: Embedding
    embed_result = embedding_router(query)

    # If both agree, we're confident — no need for LLM
    if rule_result == embed_result:
        return rule_result

    # If rule-based returned "general" but embedding found something, trust embedding
    if rule_result == "general" and embed_result != "general":
        return embed_result

    # Disagreement — escalate to LLM (most expensive but most accurate)
    try:
        llm_result = llm_classification_router(query)
        return llm_result
    except Exception:
        # If LLM fails, fall back to embedding result
        return embed_result


# ==============================================================
# APPROACH 5 (Bonus): LangGraph Intent Router
# ==============================================================
# Shows how to wire intent routing into a LangGraph StateGraph.
# The router is the ENTRY NODE that dispatches to specialist nodes.

def demo_langgraph_router():
    """Build a LangGraph with intent routing as the entry node."""
    from langgraph.graph import StateGraph, END
    from typing import TypedDict

    class RouterState(TypedDict):
        query: str
        route: str
        result: str

    def router_node(state: RouterState) -> dict:
        """Classify the query and set the route."""
        route = cascading_router(state["query"])
        print(f"    Router: '{state['query'][:50]}...' → {route}")
        return {"route": route}

    def research_node(state: RouterState) -> dict:
        return {"result": research_agent(state["query"])}

    def code_node(state: RouterState) -> dict:
        return {"result": code_agent(state["query"])}

    def summarize_node(state: RouterState) -> dict:
        return {"result": summarize_agent(state["query"])}

    def general_node(state: RouterState) -> dict:
        return {"result": general_agent(state["query"])}

    def route_dispatch(state: RouterState) -> str:
        return state["route"]

    graph = StateGraph(RouterState)
    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("code", code_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("general", general_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_dispatch, {
        "research": "research",
        "code": "code",
        "summarize": "summarize",
        "general": "general",
    })
    for node in ["research", "code", "summarize", "general"]:
        graph.add_edge(node, END)

    app = graph.compile()

    print(f"\n{'='*60}")
    print("  BONUS: LangGraph Intent Router")
    print(f"{'='*60}")
    print("  Graph: router → [research | code | summarize | general] → END\n")

    test_queries = [
        "What are the latest findings on CRISPR?",
        "Write a Python function to sort a list",
        "Summarize this 10-page climate report",
    ]

    for q in test_queries:
        result = app.invoke({"query": q, "route": "", "result": ""})
        print(f"    Result: {result['result'][:80]}...")
        print()


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    print("Example 5b: Intent Routing — 4 Approaches")
    print("=" * 60)
    print("Each router classifies queries into: research, code, or summarize")
    print("=" * 60)

    # Test approaches 1-2 (no LLM needed, instant)
    print(f"\n{'='*60}")
    print("  APPROACH 1: Rule-Based Router")
    print(f"{'='*60}")
    print("  How: Keyword matching with if/elif chains")
    print("  Cost: Zero | Latency: <1ms | Brittleness: High")
    acc1 = evaluate_router("Rule-Based", rule_based_router)

    print(f"\n{'='*60}")
    print("  APPROACH 2: Keyword/Embedding Router")
    print(f"{'='*60}")
    print("  How: Semantic similarity to category centroids")
    print("  Cost: Zero | Latency: <1ms | Robustness: Medium")
    acc2 = evaluate_router("Embedding (simulated)", embedding_router)

    # Test approach 3 (requires LLM)
    print(f"\n{'='*60}")
    print("  APPROACH 3: LLM Classification Router")
    print(f"{'='*60}")
    print("  How: Ask LLM with few-shot examples")
    print("  Cost: ~100 tokens/query | Latency: ~0.5s | Accuracy: High")
    try:
        acc3 = evaluate_router("LLM Classification", llm_classification_router)
    except Exception as e:
        print(f"  [SKIP] LLM router failed: {e}")
        acc3 = 0

    # Test approach 4 (cascading)
    print(f"\n{'='*60}")
    print("  APPROACH 4: Cascading Router")
    print(f"{'='*60}")
    print("  How: Rule → Embedding → LLM (escalate on disagreement)")
    print("  Cost: Zero for easy queries, tokens only for ambiguous ones")
    try:
        acc4 = evaluate_router("Cascading", cascading_router)
    except Exception as e:
        print(f"  [SKIP] Cascading router failed: {e}")
        acc4 = 0

    # Summary comparison
    print(f"\n{'='*60}")
    print("  ACCURACY COMPARISON")
    print(f"{'='*60}")
    print(f"  Rule-Based:       {acc1:.0f}%  (zero cost, brittle)")
    print(f"  Embedding:        {acc2:.0f}%  (zero cost, more robust)")
    print(f"  LLM Classify:     {acc3:.0f}%  (token cost, most flexible)")
    print(f"  Cascading:        {acc4:.0f}%  (cost-efficient, best balance)")

    print(f"\n{'='*60}")
    print("  MISCLASSIFICATION CASCADE WARNING")
    print(f"{'='*60}")
    print("  If the router sends a CODE question to RESEARCH agent:")
    print("    1. Research agent wastes tokens searching irrelevant sources")
    print("    2. Evaluator sees bad output, triggers retry/replan")
    print("    3. Retry STILL goes to wrong agent (same router, same mistake)")
    print("    4. Total wasted cost: 3-5x the correct routing cost")
    print()
    print("  Mitigation strategies:")
    print("    - Few-shot examples in LLM router (as shown in Approach 3)")
    print("    - Confidence threshold: if router confidence < 0.8, ask user")
    print("    - Monitor misclassification rate in Phoenix traces")
    print("    - Use cascading (Approach 4) to catch edge cases")

    # Bonus: LangGraph integration
    try:
        demo_langgraph_router()
    except Exception as e:
        print(f"\n  [SKIP] LangGraph demo failed: {e}")

    print(f"\n{'='*60}")
    print("Key Takeaways:")
    print("  1. Start with rule-based routing — it's free and surprisingly good")
    print("  2. Add embedding routing for semantic robustness")
    print("  3. Use LLM routing only for ambiguous cases (cascading)")
    print("  4. Router accuracy is THE most leveraged investment in multi-agent")
    print("  5. Monitor misclassification — bad routing cascades are expensive")
    print(f"{'='*60}")
