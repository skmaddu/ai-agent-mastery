import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 7: Advanced RAG — Hybrid Search with Reranking (LangGraph)
===================================================================
LangGraph implementation of hybrid RAG with query expansion,
BM25 + dense retrieval, RRF fusion, and simulated reranking.

Graph:
  START → expand_query → dense_retrieve → bm25_retrieve → fuse_results
        → rerank → check_quality → [quality OK?]
                                      ├─ YES → generate → END
                                      └─ NO  → expand_query (retry, max 2)

Phoenix tracing: YES — see each retrieval path in the dashboard.

Run: python week-05-context-memory/examples/example_07_advanced_rag_langgraph.py
"""

import os
import sys
import math
import textwrap
from collections import Counter
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
        tracer_provider = register(project_name="week5-advanced-rag")
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
# CORPUS (reuse from Example 6)
# ================================================================

CORPUS = [
    {"id": 0, "title": "RLHF Process", "content": "Reinforcement Learning from Human Feedback trains AI models using human preferences. Annotators rank model outputs, creating a reward model. The policy is then optimized using PPO to maximize the reward signal."},
    {"id": 1, "title": "Constitutional AI", "content": "Constitutional AI uses a set of principles to self-critique and revise outputs. The model generates responses, then evaluates them against constitutional rules. This reduces the need for human annotation."},
    {"id": 2, "title": "Direct Preference Optimization", "content": "DPO simplifies RLHF by directly optimizing the policy on preference data without a separate reward model. It treats the problem as classification between preferred and dispreferred responses."},
    {"id": 3, "title": "Prompt Injection Defense", "content": "Defending against prompt injection requires multiple layers: input sanitization strips dangerous patterns, instruction hierarchy ensures system prompts take precedence, and output filtering catches information leaks."},
    {"id": 4, "title": "AI Red Teaming", "content": "Red teaming uses adversarial techniques to find AI vulnerabilities. Teams employ jailbreak prompts, social engineering, and automated attack generation to stress-test safety guardrails."},
    {"id": 5, "title": "Model Interpretability", "content": "Mechanistic interpretability reverse-engineers neural network computations. Circuit analysis identifies specific neuron pathways responsible for capabilities like factual recall and logical reasoning."},
]


# ================================================================
# RETRIEVAL HELPERS
# ================================================================

VOCAB = ["ai", "safety", "alignment", "model", "rlhf", "feedback", "human",
         "train", "reward", "preference", "attack", "injection", "defense",
         "red", "team", "interpretability", "circuit", "policy", "optimize"]


def embed(text: str) -> List[float]:
    words = text.lower().split()
    e = [float(words.count(v)) for v in VOCAB]
    mag = math.sqrt(sum(x * x for x in e))
    return [x / mag for x in e] if mag > 0 else e


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def bm25(query, doc_content, k1=1.5, b=0.75):
    terms = query.lower().split()
    doc_terms = doc_content.lower().split()
    tf = Counter(doc_terms)
    avg_len = sum(len(d["content"].split()) for d in CORPUS) / len(CORPUS)
    score = 0.0
    for t in terms:
        df = sum(1 for d in CORPUS if t in d["content"].lower())
        if df == 0:
            continue
        idf = math.log((len(CORPUS) - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf.get(t, 0) * (k1 + 1)) / (tf.get(t, 0) + k1 * (1 - b + b * len(doc_terms) / avg_len))
        score += idf * tf_norm
    return score


# ================================================================
# STATE
# ================================================================

class AdvancedRAGState(TypedDict):
    original_query: str
    expanded_query: str
    dense_results: List[Dict]
    bm25_results: List[Dict]
    fused_results: List[Dict]
    reranked_results: List[Dict]
    top_score: float
    quality_ok: bool
    retries: int
    max_retries: int
    context: str
    answer: str
    sources: List[str]


# ================================================================
# GRAPH NODES
# ================================================================

def expand_query_node(state: AdvancedRAGState) -> dict:
    """
    Expand the query with synonyms and related terms.

    On the first pass, this adds domain-specific synonyms.
    On retries, it adds even more context to broaden the search.
    """
    query = state["original_query"]
    retries = state.get("retries", 0)

    expansions = {
        "rlhf": "reinforcement learning human feedback preference reward",
        "dpo": "direct preference optimization classification",
        "safe": "safety alignment guardrails defense",
        "injection": "injection attack adversarial jailbreak prompt",
        "interpret": "interpretability explainability circuit neuron",
    }

    expanded = query
    for key, exp in expansions.items():
        if key in query.lower():
            expanded += " " + exp

    # On retry, add broader context
    if retries > 0:
        expanded += " AI machine learning model training technique"

    print(f"  [EXPAND] '{query}' → '{expanded[:80]}...' (retry={retries})")
    return {"expanded_query": expanded}


def dense_retrieve_node(state: AdvancedRAGState) -> dict:
    """Dense (embedding-based) retrieval using the expanded query."""
    q_emb = embed(state["expanded_query"])
    scored = []
    for doc in CORPUS:
        score = cosine(q_emb, embed(doc["content"]))
        scored.append({"id": doc["id"], "title": doc["title"],
                       "content": doc["content"], "score": round(score, 4),
                       "method": "dense"})
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:5]
    print(f"  [DENSE] Top 3: {', '.join(d['title'] for d in top[:3])}")
    return {"dense_results": top}


def bm25_retrieve_node(state: AdvancedRAGState) -> dict:
    """BM25 (keyword-based) retrieval using the expanded query."""
    scored = []
    for doc in CORPUS:
        score = bm25(state["expanded_query"], doc["content"])
        scored.append({"id": doc["id"], "title": doc["title"],
                       "content": doc["content"], "score": round(score, 4),
                       "method": "bm25"})
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:5]
    print(f"  [BM25]  Top 3: {', '.join(d['title'] for d in top[:3])}")
    return {"bm25_results": top}


def fuse_results_node(state: AdvancedRAGState) -> dict:
    """
    Reciprocal Rank Fusion (RRF) to combine dense and BM25 results.

    RRF formula: score(doc) = sum( 1 / (k + rank) ) across all lists.
    k=60 is the standard parameter from the original RRF paper.
    """
    k = 60
    fused: Dict[int, float] = {}
    doc_map: Dict[int, Dict] = {}

    for ranked_list in [state["dense_results"], state["bm25_results"]]:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            fused[doc_id] = fused.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    result = []
    for doc_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(score, 4)
        result.append(doc)

    print(f"  [FUSE]  RRF top 3: {', '.join(d['title'] for d in result[:3])}")
    return {"fused_results": result[:5]}


def rerank_node(state: AdvancedRAGState) -> dict:
    """
    Simulate cross-encoder reranking on the fused results.

    In production, use a cross-encoder model or Cohere Rerank API.
    """
    query_terms = set(state["original_query"].lower().split())
    reranked = []

    for doc in state["fused_results"]:
        base = doc["rrf_score"]
        # Title match boost
        title_match = len(query_terms & set(doc["title"].lower().split())) * 0.05
        # First sentence boost
        first = doc["content"].split(".")[0].lower()
        first_match = len(query_terms & set(first.split())) * 0.03
        score = round(base + title_match + first_match, 4)
        reranked.append({**doc, "rerank_score": score})

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_score = reranked[0]["rerank_score"] if reranked else 0

    print(f"  [RERANK] Top: {reranked[0]['title']} (score={top_score})")
    return {"reranked_results": reranked, "top_score": top_score}


def check_quality(state: AdvancedRAGState) -> str:
    """
    Conditional edge: is the top result good enough?

    If the top reranked score is below a threshold AND we haven't
    exhausted retries, go back and expand the query further.
    """
    threshold = 0.025
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 2)

    if state["top_score"] >= threshold or retries >= max_retries:
        print(f"  [QUALITY] Score {state['top_score']:.4f} ≥ {threshold} or "
              f"retries={retries} — proceeding to generate")
        return "generate"
    else:
        print(f"  [QUALITY] Score {state['top_score']:.4f} < {threshold} — "
              f"retrying with broader query (attempt {retries + 1})")
        return "retry"


def increment_retry(state: AdvancedRAGState) -> dict:
    """Increment retry counter before re-expanding the query."""
    return {"retries": state.get("retries", 0) + 1}


llm = get_llm(temperature=0.3)


def generate_node(state: AdvancedRAGState) -> dict:
    """Generate answer from the top reranked results."""
    docs = state["reranked_results"][:3]
    context = "\n\n".join(
        f"[Source: {d['title']}]\n{d['content']}" for d in docs
    )
    sources = [d["title"] for d in docs]

    prompt = [
        SystemMessage(content=(
            "Answer the question based ONLY on the provided context. "
            "Cite sources with [Source: title]. Be concise."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['original_query']}"),
    ]

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [GENERATE] {answer[:120]}...")
    return {"answer": answer, "sources": sources, "context": context}


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_advanced_rag_graph():
    graph = StateGraph(AdvancedRAGState)

    graph.add_node("expand_query", expand_query_node)
    graph.add_node("dense_retrieve", dense_retrieve_node)
    graph.add_node("bm25_retrieve", bm25_retrieve_node)
    graph.add_node("fuse_results", fuse_results_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("expand_query")
    graph.add_edge("expand_query", "dense_retrieve")
    graph.add_edge("dense_retrieve", "bm25_retrieve")
    graph.add_edge("bm25_retrieve", "fuse_results")
    graph.add_edge("fuse_results", "rerank")

    graph.add_conditional_edges("rerank", check_quality, {
        "generate": "generate",
        "retry": "increment_retry",
    })
    graph.add_edge("increment_retry", "expand_query")
    graph.add_edge("generate", END)

    return graph.compile()


# ================================================================
# DEMO
# ================================================================

def run_demo():
    app = build_advanced_rag_graph()

    queries = [
        "How does RLHF train AI models?",
        "What defenses exist against prompt injection?",
        "Explain mechanistic interpretability",
    ]

    print("\n" + "=" * 65)
    print("  ADVANCED RAG — HYBRID + RERANKING (LANGGRAPH)")
    print("=" * 65)

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")

        result = app.invoke(
            {
                "original_query": query, "expanded_query": "",
                "dense_results": [], "bm25_results": [],
                "fused_results": [], "reranked_results": [],
                "top_score": 0.0, "quality_ok": False,
                "retries": 0, "max_retries": 2,
                "context": "", "answer": "", "sources": [],
            },
            {"run_name": f"advanced-rag-{i + 1}"},
        )

        print(f"\n  Answer: {result['answer'][:250]}")
        print(f"  Sources: {', '.join(result['sources'])}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 7: Advanced RAG (LangGraph)                ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. The LangGraph advanced RAG pipeline has a RETRY LOOP: if
       reranking quality is low, it broadens the query and retries.

    2. RRF fusion (k=60) is the standard method for combining ranked
       lists.  It's robust and doesn't require score normalization.

    3. Reranking after fusion improves precision significantly — the
       cross-encoder can compare query and document directly.

    4. Query expansion adds domain synonyms to catch vocabulary gaps.
       On retry, it adds even broader terms for better recall.

    5. Phoenix tracing shows each pipeline stage — useful for debugging
       why a particular query gets bad results.
    """))
