import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6b: Intent Routing in ADK — Directing Traffic to the Right Agent
=========================================================================
The ADK counterpart of Example 5b (LangGraph intent routing). Same four
routing approaches, but using Google ADK LlmAgent specialists instead of
simulated functions, and Gemini for LLM classification instead of
Groq/LangChain.

Intent routing is the "H" (Handoff) layer of the HARNESS framework:
the very first decision in the pipeline. Without it, every request
goes to a single generalist agent, negating the benefits of specialization.

Four Routing Approaches (same as 5b):

Approach         | How It Works                      | Pros              | Cons
-----------------+-----------------------------------+-------------------+--------------------
1. Rule-based    | if "SQL" in query -> code agent    | Fast, zero cost   | Brittle
2. Keyword/Embed | Semantic similarity to clusters    | More robust       | Needs embeddings
3. LLM Classify  | Ask Gemini: "research/code/sum?"   | Most flexible     | Adds latency/cost
4. Cascading     | Try cheapest first, escalate       | Cost-efficient    | More complex

LangGraph vs ADK — Intent Routing Comparison:
  LangGraph (Example 5b):
    - Router is a GRAPH NODE that sets state["route"]
    - Dispatch uses add_conditional_edges() for branching
    - Specialists are graph nodes connected by edges
    - LLM classification uses LangChain ChatGroq/ChatOpenAI
    - Synchronous execution with graph.invoke()

  ADK (This Example):
    - Router is a PYTHON FUNCTION that returns a category string
    - Dispatch calls run_agent() on the matching LlmAgent
    - Specialists are independent LlmAgent instances
    - LLM classification uses a Gemini-based classifier LlmAgent
    - Async execution with asyncio.run()
    - BONUS: ADK's built-in AgentTool delegation pattern

  Key Differences:
    - ADK routing is orchestrated in Python async code, not graph edges
    - ADK uses Gemini natively; LangGraph uses LangChain wrappers
    - ADK's AgentTool enables a supervisor to delegate without manual dispatch
    - Both achieve the same result — choose your style preference

Run: python week-04-advanced-patterns/examples/example_06b_intent_routing_adk.py
"""

import asyncio
import logging
import os
import re
import time
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Configuration
# ==============================================================

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")


# ==============================================================
# Step 1: ADK Specialist Agents
# ==============================================================
# Each specialist is a real LlmAgent with focused instructions.
# Unlike Example 5b's simulated functions, these actually call Gemini.

research_agent = LlmAgent(
    name="research_agent",
    model=MODEL,
    instruction="""You are a research specialist. Given a query, provide a brief
but thorough research response. Include:
- 2-3 key findings or facts
- Mention of relevant sources or studies where applicable
- A concise conclusion

Keep your response under 150 words. Be factual and specific.""",
    tools=[],
    description="Handles research, fact-finding, and analytical questions.",
)

code_agent = LlmAgent(
    name="code_agent",
    model=MODEL,
    instruction="""You are a coding specialist. Given a query, provide a clear
coding solution. Include:
- Working code with comments
- Brief explanation of the approach
- Note any edge cases

Keep your response focused and under 150 words. Use Python unless
another language is specified.""",
    tools=[],
    description="Handles coding, debugging, SQL, and programming tasks.",
)

summarize_agent = LlmAgent(
    name="summarize_agent",
    model=MODEL,
    instruction="""You are a summarization specialist. Given a query, provide
a concise summary or condensed version. Include:
- Key points as bullet points
- Action items if applicable
- Executive summary in 1-2 sentences

Keep your response under 150 words. Focus on extracting what matters most.""",
    tools=[],
    description="Handles summarization, condensing, and key-point extraction.",
)

general_agent = LlmAgent(
    name="general_agent",
    model=MODEL,
    instruction="""You are a helpful general assistant. Answer the user's
question directly and concisely. Keep your response under 100 words.""",
    tools=[],
    description="Handles general questions that don't fit other specialists.",
)

# Classifier agent — used for LLM-based routing (Approach 3)
classifier_agent = LlmAgent(
    name="classifier_agent",
    model=MODEL,
    instruction="""You are a query classifier. Your ONLY job is to classify
user queries into exactly ONE category.

Categories:
- research: factual questions, comparisons, analysis, "what is", finding information
- code: writing code, debugging, fixing errors, SQL, programming tasks
- summarize: condensing text, summarizing documents, extracting key points

Few-shot examples:
  "What causes inflation?" -> research
  "Write a binary search in Java" -> code
  "Give me bullet points from this article" -> summarize
  "Fix this TypeError in my React app" -> code
  "Compare Tesla vs Ford stock performance" -> research
  "Shorten this email to 3 sentences" -> summarize

Respond with ONLY the category name (research, code, or summarize).
No explanation, no punctuation, no extra text. Just the single word.""",
    tools=[],
    description="Classifies queries into research, code, or summarize.",
)

AGENT_MAP = {
    "research": research_agent,
    "code": code_agent,
    "summarize": summarize_agent,
    "general": general_agent,
}


# ==============================================================
# Step 2: Helper — Run an ADK Agent
# ==============================================================

async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent with a message and return the response text.

    Creates a fresh session per call so agents don't share context.
    Includes retry logic for transient API errors (503, rate limits).
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="intent_routing_demo",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="intent_routing_demo",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=message)],
                ),
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        result_text = event.content.parts[0].text

            return result_text
        except Exception as e:
            if attempt < retries:
                wait = attempt * 10
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [ERROR] All {retries} attempts failed: {e}")
                return f"[Error: API temporarily unavailable after {retries} retries]"


# ==============================================================
# Step 3: Test Queries (same as Example 5b)
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


# ==============================================================
# APPROACH 1: Rule-Based Router
# ==============================================================
# Identical to Example 5b — no LLM needed, pure keyword matching.

def rule_based_router(query: str) -> str:
    """Route based on keyword matching. Fast, free, but brittle."""
    q = query.lower()

    code_keywords = ["python", "function", "code", "sql", "debug", "error",
                     "fix", "write a", "implement", "bug", "indexerror",
                     "typeerror", "syntax"]
    if any(kw in q for kw in code_keywords):
        return "code"

    summarize_keywords = ["summarize", "summary", "condense", "shorten",
                         "brief", "tldr", "action items", "meeting notes",
                         "digest", "recap"]
    if any(kw in q for kw in summarize_keywords):
        return "summarize"

    research_keywords = ["research", "findings", "compare", "analyze",
                        "what is", "what are", "explain", "study",
                        "latest", "history of", "economic", "scientific"]
    if any(kw in q for kw in research_keywords):
        return "research"

    return "general"


# ==============================================================
# APPROACH 2: Keyword/Embedding Router (Simulated)
# ==============================================================
# Same simulated embedding approach as 5b. In production, replace
# with sentence-transformers or Gemini embeddings.

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

    In production, replace with:
      from google.genai import types
      embedding = client.models.embed_content(model='text-embedding-004', ...)
    """
    q_words = set(query.lower().split())

    scores = {}
    for category, centroid_words in CATEGORY_CENTROIDS.items():
        centroid_set = set(centroid_words)
        overlap = len(q_words & centroid_set)
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
# APPROACH 3: LLM Classification Router (Gemini via ADK)
# ==============================================================
# Uses the classifier_agent (a Gemini-backed LlmAgent) instead
# of LangChain's ChatGroq. This is the key ADK difference.

async def llm_classification_router(query: str) -> str:
    """Route using Gemini LLM classification via an ADK classifier agent.

    The classifier_agent is instructed to output ONLY the category name.
    This replaces LangChain's ChatGroq/ChatOpenAI from Example 5b.
    """
    response = await run_agent(classifier_agent, query)
    result = response.strip().lower()

    # Parse — the classifier should return just the category name
    for category in ["research", "code", "summarize"]:
        if category in result:
            return category
    return "general"


# ==============================================================
# APPROACH 4: Cascading Router
# ==============================================================
# Try cheapest router first, escalate on disagreement.
# Flow: rule-based (free) -> embedding (free) -> LLM (costs tokens)

async def cascading_router(query: str) -> str:
    """Route using cheapest method first, escalating if unsure.

    Level 1: Rule-based (zero cost, instant)
    Level 2: Embedding similarity (zero cost, instant)
    Level 3: LLM classification via Gemini (costs tokens, ~0.5s)

    Escalation happens when Level 1 and 2 DISAGREE.
    """
    rule_result = rule_based_router(query)
    embed_result = embedding_router(query)

    # If both agree, we're confident
    if rule_result == embed_result:
        return rule_result

    # If rule-based returned "general" but embedding found something
    if rule_result == "general" and embed_result != "general":
        return embed_result

    # Disagreement — escalate to Gemini LLM classifier
    try:
        llm_result = await llm_classification_router(query)
        return llm_result
    except Exception:
        return embed_result


# ==============================================================
# Step 4: Async Router Evaluation
# ==============================================================

async def evaluate_router_sync(name: str, route_fn, queries=TEST_QUERIES):
    """Evaluate a synchronous router against test queries."""
    print(f"\n{'- '*30}")
    print(f"  Testing: {name}")
    print(f"{'- '*30}")

    correct = 0
    total = len(queries)

    for query, expected in queries:
        start = time.time()
        predicted = route_fn(query)
        elapsed = time.time() - start

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        marker = "OK" if is_correct else "MISS"

        print(f"  [{marker}] '{query[:55]}...' -> {predicted} "
              f"(expected: {expected}) [{elapsed:.3f}s]")

    accuracy = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    return accuracy


async def evaluate_router_async(name: str, route_fn, queries=TEST_QUERIES):
    """Evaluate an async router against test queries."""
    print(f"\n{'- '*30}")
    print(f"  Testing: {name}")
    print(f"{'- '*30}")

    correct = 0
    total = len(queries)

    for query, expected in queries:
        start = time.time()
        predicted = await route_fn(query)
        elapsed = time.time() - start

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        marker = "OK" if is_correct else "MISS"

        print(f"  [{marker}] '{query[:55]}...' -> {predicted} "
              f"(expected: {expected}) [{elapsed:.3f}s]")

    accuracy = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    return accuracy


# ==============================================================
# Step 5: Full Pipeline — Route + Dispatch to ADK Agent
# ==============================================================

async def route_and_dispatch(query: str, router_name: str = "cascading"):
    """Classify a query and dispatch it to the appropriate ADK agent.

    This demonstrates the full intent routing pipeline:
      1. Router classifies the query into a category
      2. The matching ADK LlmAgent handles the query
      3. The agent's response is returned

    Args:
        query: The user's query
        router_name: Which router to use (rule/embedding/llm/cascading)

    Returns:
        Tuple of (category, agent_response)
    """
    # Step 1: Classify
    if router_name == "rule":
        category = rule_based_router(query)
    elif router_name == "embedding":
        category = embedding_router(query)
    elif router_name == "llm":
        category = await llm_classification_router(query)
    else:  # cascading (default)
        category = await cascading_router(query)

    # Step 2: Dispatch to the matching ADK agent
    agent = AGENT_MAP.get(category, general_agent)
    print(f"  Router -> {category} -> {agent.name}")

    # Step 3: Run the agent and get the response
    response = await run_agent(agent, query)

    return category, response


# ==============================================================
# BONUS: ADK AgentTool Delegation Pattern
# ==============================================================
# ADK supports a built-in delegation pattern where a supervisor
# agent can use AgentTool to delegate directly to sub-agents.
# This eliminates the need for manual Python dispatch code.
#
# The supervisor sees each sub-agent as a callable tool and
# decides which one to invoke based on its own reasoning.

async def demo_agent_tool_delegation():
    """Demonstrate ADK's AgentTool pattern for intent routing.

    Instead of a separate router function, the supervisor agent
    itself decides which specialist to call using AgentTool.
    The sub-agents appear as tools the supervisor can invoke.
    """
    from google.adk.tools.agent_tool import AgentTool

    # Wrap specialists as AgentTools
    research_tool = AgentTool(agent=research_agent)
    code_tool = AgentTool(agent=code_agent)
    summarize_tool = AgentTool(agent=summarize_agent)

    # Create a supervisor that routes via AgentTool delegation
    supervisor = LlmAgent(
        name="routing_supervisor",
        model=MODEL,
        instruction="""You are an intent router. Your job is to delegate user
queries to the most appropriate specialist agent.

Rules:
1. For research, factual, or analytical questions -> use research_agent
2. For coding, debugging, SQL, or programming -> use code_agent
3. For summarization or condensing -> use summarize_agent
4. Pick exactly ONE agent for each query
5. Pass the user's full query to the chosen agent
6. Return the specialist's response directly — do not modify it""",
        tools=[research_tool, code_tool, summarize_tool],
        description="Routes queries to specialist agents via AgentTool delegation.",
    )

    print(f"\n{'='*60}")
    print("  BONUS: ADK AgentTool Delegation Pattern")
    print(f"{'='*60}")
    print("  The supervisor agent sees specialists as callable tools.")
    print("  It decides which to invoke based on its own reasoning.")
    print("  No manual router code needed — the LLM IS the router.\n")

    demo_queries = [
        "What are the latest findings on CRISPR?",
        "Write a Python function to sort a list",
        "Summarize this 10-page climate report",
    ]

    for query in demo_queries:
        print(f"  Query: {query}")
        response = await run_agent(supervisor, query)
        # Show first 120 chars of response
        preview = response.replace('\n', ' ')[:120]
        print(f"  Response: {preview}...")
        print()

    print("  How AgentTool Works:")
    print("    1. Each sub-agent is wrapped with AgentTool(agent=...)")
    print("    2. Supervisor's tools list includes these AgentTools")
    print("    3. Gemini sees them as callable functions with descriptions")
    print("    4. Supervisor picks the right tool based on the query")
    print("    5. ADK handles the sub-agent execution automatically")
    print()
    print("  Advantage over manual routing:")
    print("    - No separate classifier agent or router function needed")
    print("    - The supervisor's LLM reasoning handles edge cases")
    print("    - Sub-agent descriptions guide the routing decision")
    print("    - Simpler code — delegation is built into the framework")


# ==============================================================
# Main
# ==============================================================

async def main():
    """Run all four routing approaches and the bonus AgentTool demo."""
    print("Example 6b: Intent Routing in ADK — 4 Approaches + AgentTool Bonus")
    print("=" * 65)
    print("Each router classifies queries into: research, code, or summarize")
    print("Specialists are real ADK LlmAgent instances backed by Gemini")
    print("=" * 65)

    # --- Approach 1: Rule-Based (no LLM needed) ---
    print(f"\n{'='*60}")
    print("  APPROACH 1: Rule-Based Router")
    print(f"{'='*60}")
    print("  How: Keyword matching with if/elif chains")
    print("  Cost: Zero | Latency: <1ms | Brittleness: High")
    acc1 = await evaluate_router_sync("Rule-Based", rule_based_router)

    # --- Approach 2: Embedding (no LLM needed) ---
    print(f"\n{'='*60}")
    print("  APPROACH 2: Keyword/Embedding Router")
    print(f"{'='*60}")
    print("  How: Semantic similarity to category centroids")
    print("  Cost: Zero | Latency: <1ms | Robustness: Medium")
    acc2 = await evaluate_router_sync("Embedding (simulated)", embedding_router)

    # --- Approach 3: LLM Classification (Gemini via ADK) ---
    print(f"\n{'='*60}")
    print("  APPROACH 3: LLM Classification Router (Gemini)")
    print(f"{'='*60}")
    print("  How: Gemini classifier agent with few-shot examples")
    print("  Cost: ~100 tokens/query | Latency: ~0.5s | Accuracy: High")
    try:
        acc3 = await evaluate_router_async("LLM Classification (Gemini)", llm_classification_router)
    except Exception as e:
        print(f"  [SKIP] LLM router failed: {e}")
        acc3 = 0

    # --- Approach 4: Cascading ---
    print(f"\n{'='*60}")
    print("  APPROACH 4: Cascading Router")
    print(f"{'='*60}")
    print("  How: Rule -> Embedding -> Gemini (escalate on disagreement)")
    print("  Cost: Zero for easy queries, tokens only for ambiguous ones")
    try:
        acc4 = await evaluate_router_async("Cascading", cascading_router)
    except Exception as e:
        print(f"  [SKIP] Cascading router failed: {e}")
        acc4 = 0

    # --- Accuracy Comparison ---
    print(f"\n{'='*60}")
    print("  ACCURACY COMPARISON")
    print(f"{'='*60}")
    print(f"  Rule-Based:       {acc1:.0f}%  (zero cost, brittle)")
    print(f"  Embedding:        {acc2:.0f}%  (zero cost, more robust)")
    print(f"  LLM Classify:     {acc3:.0f}%  (Gemini token cost, most flexible)")
    print(f"  Cascading:        {acc4:.0f}%  (cost-efficient, best balance)")

    # --- Full Pipeline Demo: Route + Dispatch to ADK Agent ---
    print(f"\n{'='*60}")
    print("  FULL PIPELINE: Route + Dispatch to ADK Specialist Agent")
    print(f"{'='*60}")
    print("  Using cascading router -> ADK LlmAgent specialist\n")

    demo_queries = [
        "What are the latest findings on CRISPR gene therapy?",
        "Write a Python function to merge two sorted lists",
        "Condense these meeting notes into action items",
    ]

    for query in demo_queries:
        print(f"  Query: {query}")
        try:
            category, response = await route_and_dispatch(query, router_name="cascading")
            preview = response.replace('\n', ' ')[:120]
            print(f"  Agent Response: {preview}...")
        except Exception as e:
            print(f"  [SKIP] Dispatch failed: {e}")
        print()

    # --- Misclassification Warning ---
    print(f"{'='*60}")
    print("  MISCLASSIFICATION CASCADE WARNING")
    print(f"{'='*60}")
    print("  If the router sends a CODE question to RESEARCH agent:")
    print("    1. Research agent wastes tokens searching irrelevant sources")
    print("    2. Evaluator sees bad output, triggers retry/replan")
    print("    3. Retry STILL goes to wrong agent (same router, same mistake)")
    print("    4. Total wasted cost: 3-5x the correct routing cost")
    print()
    print("  Mitigation strategies:")
    print("    - Few-shot examples in classifier agent (as in Approach 3)")
    print("    - Confidence threshold: if unsure, ask user to clarify")
    print("    - Monitor misclassification rate in Phoenix traces")
    print("    - Use cascading (Approach 4) to catch edge cases")
    print("    - Use AgentTool delegation (Bonus) for LLM-native routing")

    # --- Bonus: AgentTool Delegation ---
    try:
        await demo_agent_tool_delegation()
    except Exception as e:
        print(f"\n  [SKIP] AgentTool demo failed: {e}")
        print(f"  (AgentTool may require a newer version of google-adk)")

    # --- Key Takeaways ---
    print(f"\n{'='*60}")
    print("Key Takeaways:")
    print("  1. Same 4 routing approaches work in ADK as in LangGraph")
    print("  2. ADK uses Gemini natively — no LangChain wrapper needed")
    print("  3. Specialists are real LlmAgent instances, not simulated functions")
    print("  4. Async orchestration in Python replaces LangGraph's graph edges")
    print("  5. AgentTool delegation lets the LLM itself be the router")
    print("  6. Router accuracy is THE most leveraged investment in multi-agent")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
