import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 9b: Agentic RAG — Dynamic Retrieval (Google ADK)
==========================================================
ADK agent that dynamically decides when, where, and how to retrieve.

The agent has three tools:
  - search_research_papers: Academic AI research
  - search_regulations: Legal/regulatory documents
  - check_answer_quality: Verify if the answer is grounded

The agent decides which tools to call based on the query,
implementing the same CRAG pattern as Example 9 but through
agent-driven tool selection rather than graph edges.

Run: python week-05-context-memory/examples/example_09b_agentic_rag_adk.py
"""

import os
import sys
import json
import math
import asyncio
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# ── Phoenix ────────────────────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        register(project_name="week5-agentic-rag-adk")
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None

# ── ADK ────────────────────────────────────────────────────────
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("[ADK] Not installed. pip install google-adk")


# ================================================================
# SIMULATED KNOWLEDGE BASES
# ================================================================

KB_RESEARCH = [
    {"title": "RLHF Mechanism", "content": "RLHF trains a reward model on human preference data then uses PPO to optimize the policy."},
    {"title": "DPO Method", "content": "DPO directly optimizes language model policy using preference pairs without a separate reward model."},
    {"title": "Safety Alignment", "content": "Safety alignment combines RLHF with constitutional principles, red-teaming, and interpretability."},
]

KB_REGULATIONS = [
    {"title": "EU AI Act", "content": "The EU AI Act classifies AI by risk level. High-risk systems need accuracy, robustness, and human oversight."},
    {"title": "US AI EO", "content": "US Executive Order requires frontier model developers to report safety test results to the AI Safety Institute."},
]


def _embed(text):
    vocab = ["ai", "safety", "rlhf", "dpo", "alignment", "model", "policy",
             "reward", "human", "regulation", "eu", "act"]
    words = text.lower().split()
    e = [float(words.count(v)) for v in vocab]
    m = math.sqrt(sum(x * x for x in e))
    return [x / m for x in e] if m > 0 else e


def _search(query, kb, top_k=2):
    q = _embed(query)
    scored = []
    for doc in kb:
        d = _embed(doc["content"])
        score = sum(a * b for a, b in zip(q, d))
        scored.append({**doc, "score": round(score, 3)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ================================================================
# ADK TOOLS
# ================================================================

def search_research_papers(query: str) -> str:
    """
    Search academic AI research papers for technical information.

    Use this tool for questions about AI techniques, algorithms,
    safety methods (RLHF, DPO, alignment), and research findings.

    Args:
        query: Search query for research papers.

    Returns:
        JSON with matching research papers and relevance scores.
    """
    results = _search(query, KB_RESEARCH)
    print(f"  [TOOL] search_research_papers('{query[:40]}') → {len(results)} results")
    return json.dumps({"source": "research_papers", "results": results})


def search_regulations(query: str) -> str:
    """
    Search AI regulations and legal documents.

    Use this tool for questions about AI laws, compliance, EU AI Act,
    US Executive Orders, and regulatory requirements.

    Args:
        query: Search query for regulatory documents.

    Returns:
        JSON with matching regulatory documents and relevance scores.
    """
    results = _search(query, KB_REGULATIONS)
    print(f"  [TOOL] search_regulations('{query[:40]}') → {len(results)} results")
    return json.dumps({"source": "regulations", "results": results})


def check_answer_quality(proposed_answer: str, source_documents: str) -> str:
    """
    Verify that a proposed answer is grounded in source documents.

    Use this AFTER retrieving documents and before giving a final answer.
    It checks whether the answer is supported by the retrieved context.

    Args:
        proposed_answer: The answer you're about to give.
        source_documents: The source documents the answer should be based on.

    Returns:
        JSON with grounding assessment: is_grounded (bool), confidence (0-1),
        and suggestions for improvement if not grounded.
    """
    # Simple heuristic: check word overlap between answer and sources
    answer_words = set(proposed_answer.lower().split())
    source_words = set(source_documents.lower().split())
    overlap = len(answer_words & source_words)
    confidence = min(1.0, overlap / max(len(answer_words), 1))
    is_grounded = confidence > 0.3

    result = {
        "is_grounded": is_grounded,
        "confidence": round(confidence, 2),
        "suggestion": "Answer is well-grounded." if is_grounded else "Consider re-searching with more specific terms.",
    }
    print(f"  [TOOL] check_answer_quality → grounded={is_grounded}, conf={confidence:.2f}")
    return json.dumps(result)


# ================================================================
# AGENT
# ================================================================

def build_agentic_rag_agent():
    agent = LlmAgent(
        name="agentic_rag_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are an AI safety expert with access to research papers and regulatory documents.

WORKFLOW (follow this exactly):
1. Analyze the user's question to determine which knowledge base(s) to search.
2. Search the appropriate knowledge base(s) using the search tools.
3. Before answering, use check_answer_quality to verify your answer is grounded.
4. If not grounded, search again with refined terms.
5. Provide your final answer with [Source: title] citations.

RULES:
- ALWAYS search before answering — never rely on your training data alone.
- Use search_research_papers for technical/scientific questions.
- Use search_regulations for legal/regulatory questions.
- For comparison questions, search BOTH knowledge bases.
- ALWAYS verify your answer with check_answer_quality before responding.
- If verification fails, refine your search and try again (max 2 retries).""",
        tools=[search_research_papers, search_regulations, check_answer_quality],
        description="AI safety expert with agentic RAG capabilities.",
    )
    return agent


# ================================================================
# RUNNER
# ================================================================

async def run_agent(agent, query: str, retries: int = 5) -> str:
    """Run agent with retry logic for transient Gemini API errors."""
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(agent=agent, app_name="week5_crag_adk", session_service=session_service)
            session = await session_service.create_session(app_name="week5_crag_adk", user_id="user1")

            result_text = ""
            async for event in runner.run_async(
                user_id="user1", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=query)]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
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

    return "[Error: unexpected]"


async def run_demo():
    if not ADK_AVAILABLE:
        print("[SKIP] ADK not available.")
        return

    agent = build_agentic_rag_agent()
    queries = [
        "How does RLHF work?",
        "What does the EU AI Act require for high-risk AI?",
        "Compare RLHF and DPO for safety alignment",
    ]

    print("\n" + "=" * 65)
    print("  AGENTIC RAG — GOOGLE ADK")
    print("=" * 65)

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")
        answer = await run_agent(agent, query)
        print(f"\n  Answer: {answer[:300]}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 9b: Agentic RAG (Google ADK)              ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()

    if ADK_AVAILABLE:
        asyncio.run(run_demo())
    else:
        print("\n  Install Google ADK: pip install google-adk")

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. ADK agentic RAG gives the agent TOOLS for different knowledge
       bases and a quality-check tool for grounding verification.

    2. The agent's instruction defines the CRAG workflow — the agent
       follows it through tool calls rather than graph edges.

    3. ADK vs LangGraph for agentic RAG:
       - LangGraph: Deterministic loop with explicit retry edges
       - ADK: Agent-driven, more flexible, relies on instruction following

    4. The check_answer_quality tool is key — it forces the agent to
       verify before responding, reducing hallucination.

    5. Both approaches achieve the same CRAG pattern; choose based on
       whether you need deterministic (LangGraph) or flexible (ADK) control.
    """))
