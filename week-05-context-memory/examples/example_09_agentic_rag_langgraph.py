import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 9: Agentic RAG — CRAG Pattern (LangGraph)
====================================================
LangGraph implementation of the Corrective RAG (CRAG) pattern:
  route → retrieve → check groundedness → [grounded?]
                                             ├─ YES → generate → END
                                             └─ NO  → refine → route (max 3)

The agent:
  1. Routes the query to the best knowledge base
  2. Retrieves documents
  3. Checks if the answer would be GROUNDED in retrieved docs
  4. If not grounded, refines the query and re-retrieves

Phoenix tracing: YES — see the self-correction loop in the dashboard.

Run: python week-05-context-memory/examples/example_09_agentic_rag_langgraph.py
"""

import os
import sys
import math
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Phoenix ────────────────────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week5-agentic-rag")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None

# ── LLM ────────────────────────────────────────────────────────

def get_llm(temperature=0.7):
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
# MULTI-SOURCE KNOWLEDGE BASES
# ================================================================
# We simulate three different knowledge bases to demonstrate
# query routing.  In production, these would be separate vector
# stores, databases, or API endpoints.

KB_RESEARCH = [
    {"title": "RLHF Mechanism", "content": "RLHF trains a reward model on human preference data then uses PPO to optimize the policy. Key steps: collect comparisons, train reward model, optimize policy with RL."},
    {"title": "DPO Method", "content": "DPO directly optimizes language model policy using preference pairs without training a separate reward model. It reformulates RLHF as a classification problem."},
    {"title": "Safety Alignment", "content": "Safety alignment combines RLHF with constitutional principles, red-teaming, and interpretability research to ensure AI systems follow human intentions."},
]

KB_REGULATIONS = [
    {"title": "EU AI Act Summary", "content": "The EU AI Act classifies AI by risk level. High-risk systems must meet standards for accuracy, robustness, and human oversight. Violations face fines up to 35M EUR."},
    {"title": "US AI Executive Order", "content": "US EO on AI Safety requires frontier model developers to report safety test results. Establishes AI Safety Institute for testing and evaluation."},
]

KB_ENGINEERING = [
    {"title": "RAG Best Practices", "content": "Production RAG should use hybrid search (dense + BM25), reranking, and grounding verification. Chunk size 200-500 tokens with recursive splitting."},
    {"title": "LangGraph Memory", "content": "LangGraph supports memory via TypedDict state with MemorySaver checkpointing. State persists across turns with time-travel debugging."},
]

ALL_KBS = {
    "research": KB_RESEARCH,
    "regulations": KB_REGULATIONS,
    "engineering": KB_ENGINEERING,
}


# ================================================================
# RETRIEVAL HELPERS
# ================================================================

def _embed(text):
    vocab = ["ai", "safety", "rlhf", "dpo", "alignment", "model", "policy",
             "reward", "human", "regulation", "eu", "act", "rag", "memory",
             "langgraph", "vector", "search", "preference", "feedback"]
    words = text.lower().split()
    e = [float(words.count(v)) for v in vocab]
    m = math.sqrt(sum(x * x for x in e))
    return [x / m for x in e] if m > 0 else e


def _search(query, kb_docs, top_k=2):
    q = _embed(query)
    scored = []
    for doc in kb_docs:
        d = _embed(doc["content"])
        score = sum(a * b for a, b in zip(q, d))
        scored.append({**doc, "score": round(score, 3)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ================================================================
# STATE
# ================================================================

class CRAGState(TypedDict):
    query: str                          # Original user query
    current_query: str                  # Possibly refined query
    route: str                          # Selected knowledge base
    retrieved_docs: List[Dict]          # Retrieved documents
    is_grounded: bool                   # Whether answer would be grounded
    groundedness_score: float           # 0-1 grounding quality
    retries: int                        # Current retry count
    max_retries: int                    # Safety limit
    context: str                        # Formatted context for generation
    answer: str                         # Final answer
    sources: List[str]                  # Source citations


# ================================================================
# GRAPH NODES
# ================================================================

def route_node(state: CRAGState) -> dict:
    """
    Route the query to the most appropriate knowledge base.

    This is a CLASSIFICATION task — in production, use an LLM:
        "Given these KBs: research, regulations, engineering —
         which is best for: [query]?"
    """
    query = state["current_query"].lower()

    if any(w in query for w in ["regulation", "eu", "act", "legal", "compliance"]):
        route = "regulations"
    elif any(w in query for w in ["rlhf", "dpo", "alignment", "safety", "research"]):
        route = "research"
    else:
        route = "engineering"

    print(f"  [ROUTE] '{state['current_query'][:50]}...' → {route}")
    return {"route": route}


def retrieve_node(state: CRAGState) -> dict:
    """Retrieve documents from the selected knowledge base."""
    route = state["route"]
    query = state["current_query"]
    kb = ALL_KBS.get(route, KB_ENGINEERING)

    docs = _search(query, kb, top_k=2)
    print(f"  [RETRIEVE] From '{route}': {len(docs)} docs")
    for d in docs:
        print(f"    [{d['score']:.3f}] {d['title']}")

    return {"retrieved_docs": docs}


def check_groundedness_node(state: CRAGState) -> dict:
    """
    Check whether the retrieved docs can ground an answer.

    This simulates an LLM grounding check.  In production:
        prompt = "Can this question be answered using ONLY these docs?"
        grounded = llm.invoke(prompt) → "yes" or "no"

    We simulate using the retrieval score as a proxy for groundedness.
    """
    docs = state["retrieved_docs"]
    if not docs:
        score = 0.0
    else:
        score = max(d["score"] for d in docs)

    threshold = 0.3
    is_grounded = score >= threshold

    print(f"  [GROUND] Score={score:.3f}, threshold={threshold}, "
          f"grounded={'YES' if is_grounded else 'NO'}")

    return {"is_grounded": is_grounded, "groundedness_score": score}


def should_continue(state: CRAGState) -> str:
    """Conditional: generate if grounded, refine if not (with retry limit)."""
    if state["is_grounded"]:
        return "generate"
    elif state["retries"] >= state["max_retries"]:
        print(f"  [DECISION] Max retries ({state['max_retries']}) reached — "
              f"generating with best available context")
        return "generate"
    else:
        print(f"  [DECISION] Not grounded — refining query (retry {state['retries'] + 1})")
        return "refine"


def refine_node(state: CRAGState) -> dict:
    """
    Refine the query for better retrieval on the next attempt.

    Strategies:
      - Add domain-specific terms
      - Try a different knowledge base
      - Rephrase the question
    """
    query = state["current_query"]
    retries = state["retries"]

    # On first retry: add specificity
    if retries == 0:
        refined = query + " specific mechanism details technical"
    # On second retry: try broader terms
    elif retries == 1:
        refined = query + " overview explanation fundamentals"
    else:
        refined = query + " comprehensive guide"

    print(f"  [REFINE] '{query[:40]}...' → '{refined[:50]}...'")
    return {"current_query": refined, "retries": retries + 1}


llm = get_llm(temperature=0.3)


def generate_node(state: CRAGState) -> dict:
    """Generate final answer from retrieved context."""
    docs = state["retrieved_docs"]
    context = "\n\n".join(
        f"[Source: {d['title']}]\n{d['content']}" for d in docs
    )
    sources = [d["title"] for d in docs]

    prompt = [
        SystemMessage(content=(
            "Answer based ONLY on the provided context. Cite sources. "
            "If the context is insufficient, say what's missing."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}"),
    ]

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [GENERATE] {answer[:120]}...")
    return {"answer": answer, "sources": sources, "context": context}


# ================================================================
# GRAPH
# ================================================================

def build_crag_graph():
    """
    Build the CRAG (Corrective RAG) graph.

    Flow:
      route → retrieve → check_groundedness → [grounded?]
                                                 ├─ YES → generate → END
                                                 └─ NO  → refine → route
    """
    graph = StateGraph(CRAGState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_groundedness", check_groundedness_node)
    graph.add_node("refine", refine_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "check_groundedness")

    graph.add_conditional_edges("check_groundedness", should_continue, {
        "generate": "generate",
        "refine": "refine",
    })
    graph.add_edge("refine", "route")
    graph.add_edge("generate", END)

    return graph.compile()


# ================================================================
# DEMO
# ================================================================

def run_demo():
    app = build_crag_graph()

    queries = [
        "How does RLHF train AI models?",
        "What are the EU AI Act requirements?",
        "Best practices for production RAG systems",
    ]

    print("\n" + "=" * 65)
    print("  AGENTIC RAG — CRAG PATTERN (LANGGRAPH)")
    print("=" * 65)

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")

        result = app.invoke(
            {
                "query": query, "current_query": query,
                "route": "", "retrieved_docs": [],
                "is_grounded": False, "groundedness_score": 0.0,
                "retries": 0, "max_retries": 2,
                "context": "", "answer": "", "sources": [],
            },
            {"run_name": f"crag-{i + 1}"},
        )

        print(f"\n  Answer: {result['answer'][:250]}")
        print(f"  Sources: {', '.join(result['sources'])}")
        print(f"  Retries: {result['retries']}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 9: Agentic RAG / CRAG (LangGraph)         ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. CRAG adds a SELF-CORRECTION LOOP to RAG: if retrieved docs
       can't ground the answer, the agent refines and re-retrieves.

    2. Query routing sends queries to the RIGHT knowledge base —
       don't search everywhere for everything.

    3. Groundedness checking prevents hallucination by verifying that
       the answer is supported by retrieved context.

    4. Always set a max_retries limit to prevent infinite loops.
       2-3 retries is typically sufficient.

    5. Phoenix tracing shows the full correction loop — essential
       for debugging why a query needed multiple retrieval attempts.
    """))
