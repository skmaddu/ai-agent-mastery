import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 7b: Advanced RAG — Production Pipeline with Real Vector Store
======================================================================
PRODUCTION-GRADE hybrid search with real embeddings, real vector store,
BM25 keyword search, Reciprocal Rank Fusion, and cross-encoder reranking.

This is the REAL implementation of the concepts taught in Examples 6-7.
Every component uses production libraries:

  - ChromaDB: persistent vector store (not toy in-memory lists)
  - sentence-transformers: real semantic embeddings (not word frequency)
  - rank_bm25: real BM25 keyword search (not simplified heuristics)
  - CrossEncoder: real neural reranking (not score heuristics)

Pipeline (LangGraph):
  START → ingest_documents → expand_query → dense_retrieve (ChromaDB)
        → bm25_retrieve → fuse_results (RRF) → rerank (CrossEncoder)
        → check_quality → [quality OK?]
                            ├─ YES → generate (LLM) → END
                            └─ NO  → expand_query (retry, max 2)

Requirements (already in requirements.txt):
  pip install chromadb sentence-transformers rank-bm25

Run: python week-05-context-memory/examples/example_07b_advanced_rag_production.py
"""

import os
import sys
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Phoenix Tracing ───────────────────────────────────────────
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
        tracer_provider = register(project_name="week5-advanced-rag-production")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None

# ── LLM Setup ────────────────────────────────────────────────

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
# REAL EMBEDDING MODEL + VECTOR STORE + BM25 + CROSS-ENCODER
# ================================================================
# This section initializes the REAL production components.
# Compare with Example 7 which used toy word-frequency embeddings.

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# ── Embedding Model ───────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80MB, excellent quality/speed ratio.
# Captures REAL semantic meaning — "car" matches "automobile".
print("[INIT] Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Cross-Encoder Reranker ────────────────────────────────────
# Cross-encoders are MORE ACCURATE than bi-encoders because they
# process (query, document) TOGETHER, allowing cross-attention.
# But they're slower — so we only rerank the top candidates.
#
# ms-marco-MiniLM-L-6-v2: trained on MS MARCO passage ranking,
# specifically designed for search reranking tasks.
print("[INIT] Loading cross-encoder reranker...")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[INIT] Models ready.\n")


# ================================================================
# KNOWLEDGE BASE — Larger, More Realistic Corpus
# ================================================================
# A richer corpus to demonstrate where hybrid search + reranking
# really shines over basic dense-only retrieval.

CORPUS = [
    {
        "id": "doc_0", "title": "RLHF Training Process",
        "content": (
            "Reinforcement Learning from Human Feedback (RLHF) trains AI models "
            "using human preferences. The process has three stages: (1) Supervised "
            "fine-tuning on high-quality demonstrations, (2) Training a reward model "
            "on human preference comparisons where annotators rank model outputs, "
            "(3) Optimizing the policy using Proximal Policy Optimization (PPO) to "
            "maximize the reward signal while staying close to the original model. "
            "RLHF was pioneered by OpenAI and Anthropic for aligning large language models."
        ),
    },
    {
        "id": "doc_1", "title": "Constitutional AI Framework",
        "content": (
            "Constitutional AI (CAI) is an alignment approach developed by Anthropic. "
            "Instead of relying solely on human feedback, CAI uses a set of principles "
            "(a 'constitution') to guide self-improvement. The model generates responses, "
            "then critiques and revises them against constitutional rules. This reduces "
            "annotation costs and scales better than pure RLHF. CAI combines supervised "
            "learning from self-revision with RLHF from AI-generated preference data."
        ),
    },
    {
        "id": "doc_2", "title": "Direct Preference Optimization",
        "content": (
            "DPO (Direct Preference Optimization) simplifies RLHF by eliminating the "
            "need for a separate reward model. It directly optimizes the language model "
            "policy on preference data by reformulating the RLHF objective as a "
            "classification problem: preferred vs. dispreferred responses. DPO is "
            "mathematically equivalent to RLHF under certain assumptions but is simpler "
            "to implement, more stable to train, and computationally cheaper. Published "
            "by Rafailov et al. at Stanford in 2023."
        ),
    },
    {
        "id": "doc_3", "title": "Prompt Injection Defense Strategies",
        "content": (
            "Defending against prompt injection requires a multi-layered approach. "
            "Layer 1: Input sanitization strips dangerous patterns and escape sequences. "
            "Layer 2: Instruction hierarchy ensures system prompts take precedence over "
            "user input. Layer 3: Output filtering catches information leaks and policy "
            "violations. Layer 4: Canary tokens detect when system prompts are extracted. "
            "No single defense is sufficient — defense in depth is essential."
        ),
    },
    {
        "id": "doc_4", "title": "AI Red Teaming Methodology",
        "content": (
            "Red teaming uses adversarial techniques to discover AI vulnerabilities "
            "before deployment. Techniques include: jailbreak prompt engineering, social "
            "engineering attacks, automated attack generation using attacker LLMs, and "
            "multi-turn manipulation strategies. Effective red teams combine automated "
            "fuzzing with human creativity. Best practice: continuous red-teaming "
            "throughout the model lifecycle, not just pre-launch."
        ),
    },
    {
        "id": "doc_5", "title": "Mechanistic Interpretability",
        "content": (
            "Mechanistic interpretability reverse-engineers how neural networks compute "
            "internally. Key techniques: circuit analysis identifies specific neuron "
            "pathways (circuits) responsible for capabilities like factual recall, "
            "induction heads, and logical reasoning. Researchers at Anthropic have "
            "mapped circuits for indirect object identification and multi-step reasoning. "
            "Tools include activation patching, causal tracing, and sparse autoencoders."
        ),
    },
    {
        "id": "doc_6", "title": "AI Safety Benchmarks and Evaluation",
        "content": (
            "Safety benchmarks measure model robustness against harmful outputs. Key "
            "benchmarks: HarmBench tests resistance to adversarial attacks across "
            "categories. TruthfulQA measures factual accuracy and resistance to common "
            "misconceptions. BBQ (Bias Benchmark for QA) evaluates social biases. "
            "SafetyPrompts provides a standardized test suite for safety evaluation. "
            "Benchmarks should be run before and after safety training."
        ),
    },
    {
        "id": "doc_7", "title": "The Alignment Tax Problem",
        "content": (
            "The alignment tax refers to the performance cost of safety training. "
            "RLHF and safety filters can reduce model capability on benign tasks — "
            "for example, making models overly cautious or less creative. Minimizing "
            "the alignment tax while maintaining safety is an active research area. "
            "Techniques to reduce the tax include targeted safety training, capability "
            "evaluations, and Pareto-optimal training schedules."
        ),
    },
    {
        "id": "doc_8", "title": "RAG Pipeline Best Practices",
        "content": (
            "Production RAG pipelines should use hybrid search combining dense "
            "embeddings with BM25 keyword matching. Chunk documents at 200-500 tokens "
            "using recursive text splitting with overlap. Use cross-encoder reranking "
            "on the top 10-20 candidates to improve precision. Always verify answer "
            "groundedness — check that the generated answer is supported by retrieved "
            "context. Monitor retrieval quality with MRR, NDCG, and recall metrics."
        ),
    },
    {
        "id": "doc_9", "title": "EU AI Act Regulatory Framework",
        "content": (
            "The EU AI Act (effective 2025) establishes a risk-based framework for AI "
            "regulation. It classifies AI systems into four risk tiers: unacceptable "
            "(banned), high-risk (strict requirements), limited risk (transparency "
            "obligations), and minimal risk (no requirements). High-risk AI must meet "
            "standards for accuracy, robustness, cybersecurity, and human oversight. "
            "Violations face fines up to 35 million EUR or 7% of global turnover."
        ),
    },
]


# ================================================================
# DOCUMENT INGESTION — ChromaDB + BM25 Index
# ================================================================

def build_indices():
    """
    Build BOTH a dense index (ChromaDB) and a sparse index (BM25).

    This is the key to hybrid search: two complementary indices.

    Dense (ChromaDB + sentence-transformers):
      - Captures SEMANTIC similarity ("car" ≈ "automobile")
      - Great for natural language queries
      - Misses exact keyword matches sometimes

    Sparse (BM25):
      - Captures KEYWORD matches ("DPO" → documents containing "DPO")
      - Great for acronyms, names, technical terms
      - Misses paraphrases and synonyms

    Together, they cover each other's weaknesses.
    """
    # ── ChromaDB Dense Index ──────────────────────────────────
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("advanced_rag_kb")
    except Exception:
        pass

    collection = client.create_collection(
        name="advanced_rag_kb",
        metadata={"description": "AI safety corpus for advanced RAG"},
    )

    contents = [doc["content"] for doc in CORPUS]
    embeddings = EMBED_MODEL.encode(contents).tolist()

    collection.add(
        ids=[doc["id"] for doc in CORPUS],
        documents=contents,
        embeddings=embeddings,
        metadatas=[{"title": doc["title"]} for doc in CORPUS],
    )
    print(f"  [INDEX] ChromaDB: {collection.count()} docs indexed "
          f"(dim={len(embeddings[0])})")

    # ── BM25 Sparse Index ─────────────────────────────────────
    # Tokenize documents for BM25 (simple whitespace + lowercase)
    tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"  [INDEX] BM25: {len(tokenized_corpus)} docs indexed")

    return collection, bm25_index


print("[INDEX] Building dense + sparse indices...")
CHROMA_COLLECTION, BM25_INDEX = build_indices()


# ================================================================
# STATE
# ================================================================

class AdvancedRAGState(TypedDict):
    original_query: str
    expanded_query: str
    dense_results: List[Dict]       # From ChromaDB
    bm25_results: List[Dict]        # From BM25
    fused_results: List[Dict]       # After RRF
    reranked_results: List[Dict]    # After cross-encoder
    top_score: float                # Best reranker score
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
    Expand the query with domain-specific synonyms.

    In a full production system, you'd use the LLM for this:
        expanded = llm.invoke(f"Expand this search query with synonyms: {query}")

    Here we use a deterministic expansion for reproducibility.
    On retries, we broaden further to increase recall.
    """
    query = state["original_query"]
    retries = state.get("retries", 0)

    # Domain-specific expansions
    expansions = {
        "rlhf": "reinforcement learning human feedback preference reward PPO",
        "dpo": "direct preference optimization classification",
        "safe": "safety alignment guardrails defense robustness",
        "injection": "injection attack adversarial jailbreak prompt defense",
        "interpret": "interpretability explainability circuit mechanistic",
        "regulation": "regulation EU AI Act compliance legal framework",
        "benchmark": "benchmark evaluation metrics testing HarmBench TruthfulQA",
    }

    expanded = query
    for key, exp in expansions.items():
        if key in query.lower():
            expanded += " " + exp

    if retries > 0:
        expanded += " AI machine learning model training technique approach"

    print(f"  [EXPAND] '{query[:50]}' → '{expanded[:70]}...' (retry={retries})")
    return {"expanded_query": expanded}


def dense_retrieve_node(state: AdvancedRAGState) -> dict:
    """
    Dense retrieval using ChromaDB with real sentence-transformer embeddings.

    This is SEMANTIC search — it understands meaning, not just keywords.
    "How do you train AI with human preferences?" matches RLHF documents
    even though none of those exact words appear in the query.
    """
    query = state["expanded_query"]
    top_k = 5

    # Embed query with the SAME model used for indexing
    query_embedding = EMBED_MODEL.encode(query).tolist()

    # Query ChromaDB
    results = CHROMA_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    dense_results = []
    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
        # Convert L2 distance to similarity (higher = better)
        similarity = 1.0 / (1.0 + distance)
        dense_results.append({
            "title": metadata["title"],
            "content": doc_text,
            "score": round(similarity, 4),
            "method": "dense",
        })

    print(f"  [DENSE] ChromaDB top 3: "
          f"{', '.join(f'{d['title']}({d['score']:.3f})' for d in dense_results[:3])}")
    return {"dense_results": dense_results}


def bm25_retrieve_node(state: AdvancedRAGState) -> dict:
    """
    Sparse retrieval using real BM25 (rank_bm25 library).

    BM25 is the gold standard for keyword-based retrieval. It excels at:
      - Exact term matching (acronyms like "DPO", "RLHF", "PPO")
      - Rare terms (technical vocabulary gets high IDF weight)
      - When the user knows the exact terminology

    BM25 scores are NOT normalized — they can be any positive number.
    That's why we use Reciprocal Rank Fusion (RRF) to combine with
    dense results, since RRF works on RANKS, not raw scores.
    """
    query = state["expanded_query"]
    top_k = 5

    # Tokenize query the same way we tokenized the corpus
    query_tokens = query.lower().split()

    # Get BM25 scores for all documents
    scores = BM25_INDEX.get_scores(query_tokens)

    # Pair scores with documents and sort
    scored_docs = []
    for idx, score in enumerate(scores):
        doc = CORPUS[idx]
        scored_docs.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": round(float(score), 4),
            "method": "bm25",
        })

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    bm25_results = scored_docs[:top_k]

    print(f"  [BM25]  Top 3: "
          f"{', '.join(f'{d['title']}({d['score']:.2f})' for d in bm25_results[:3])}")
    return {"bm25_results": bm25_results}


def fuse_results_node(state: AdvancedRAGState) -> dict:
    """
    Reciprocal Rank Fusion (RRF) — merge dense and BM25 ranked lists.

    RRF formula: score(doc) = sum( 1 / (k + rank_i) ) for each list i.

    Why RRF instead of weighted average?
      - RRF works on RANKS, not scores — no normalization needed
      - Dense scores (0-1 similarity) and BM25 scores (unbounded)
        are on completely different scales
      - RRF is robust: k=60 (from the original paper) works universally
      - Simple, no hyperparameter tuning beyond k

    This is the same RRF used by Elasticsearch, Pinecone, and Weaviate.
    """
    k = 60  # Standard RRF parameter
    fused_scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict] = {}

    for ranked_list in [state["dense_results"], state["bm25_results"]]:
        for rank, doc in enumerate(ranked_list):
            title = doc["title"]
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[title] = fused_scores.get(title, 0) + rrf_score
            doc_data[title] = doc

    # Sort by fused score
    fused = []
    for title, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_data[title].copy()
        doc["rrf_score"] = round(score, 4)
        fused.append(doc)

    top_fused = fused[:7]  # Keep top 7 for reranking
    print(f"  [FUSE]  RRF top 3: "
          f"{', '.join(f'{d['title']}({d['rrf_score']:.4f})' for d in top_fused[:3])}")
    return {"fused_results": top_fused}


def rerank_node(state: AdvancedRAGState) -> dict:
    """
    Neural reranking with a real cross-encoder model.

    Cross-encoders are MORE ACCURATE than bi-encoders because they
    process query and document TOGETHER with full cross-attention.
    The tradeoff: they're slower (can't pre-compute document embeddings).

    That's why we use a two-stage pipeline:
      Stage 1 (fast): Dense + BM25 retrieves top ~20 candidates
      Stage 2 (accurate): Cross-encoder reranks the top candidates

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Trained on MS MARCO passage ranking dataset
      - Output: relevance score (higher = more relevant)
      - ~50ms per (query, document) pair
    """
    query = state["original_query"]
    fused = state["fused_results"]

    if not fused:
        return {"reranked_results": [], "top_score": 0.0}

    # Prepare (query, document) pairs for the cross-encoder
    pairs = [(query, doc["content"]) for doc in fused]

    # Cross-encoder scores all pairs in one batch
    ce_scores = RERANKER.predict(pairs)

    # Attach scores and sort
    reranked = []
    for doc, ce_score in zip(fused, ce_scores):
        reranked.append({
            **doc,
            "rerank_score": round(float(ce_score), 4),
        })

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_score = reranked[0]["rerank_score"]

    print(f"  [RERANK] Cross-encoder top 3:")
    for i, doc in enumerate(reranked[:3]):
        print(f"    #{i+1} [{doc['rerank_score']:.4f}] {doc['title']}")

    return {"reranked_results": reranked, "top_score": top_score}


def check_quality(state: AdvancedRAGState) -> str:
    """
    Conditional edge: is the top reranked result relevant enough?

    Cross-encoder scores are roughly calibrated:
      > 2.0  = highly relevant
      0 - 2  = somewhat relevant
      < 0    = probably irrelevant

    If the best result scores below threshold and we have retries
    left, broaden the query and try again.
    """
    threshold = 0.5
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", 2)

    if state["top_score"] >= threshold or retries >= max_retries:
        print(f"  [QUALITY] Score {state['top_score']:.4f} >= {threshold} or "
              f"retries={retries} — proceeding to generate")
        return "generate"
    else:
        print(f"  [QUALITY] Score {state['top_score']:.4f} < {threshold} — "
              f"retrying with broader query (attempt {retries + 1})")
        return "retry"


def increment_retry(state: AdvancedRAGState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


llm = get_llm(temperature=0.3)


def generate_node(state: AdvancedRAGState) -> dict:
    """Generate answer from the top reranked results with citations."""
    docs = state["reranked_results"][:3]
    context = "\n\n".join(
        f"[Source: {d['title']}] (relevance: {d['rerank_score']:.2f})\n{d['content']}"
        for d in docs
    )
    sources = [d["title"] for d in docs]

    prompt = [
        SystemMessage(content=(
            "You are an AI safety expert. Answer based ONLY on the provided context. "
            "Cite sources with [Source: title]. Be concise and accurate. "
            "If the context is insufficient, say what's missing."
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
    """
    Build the production advanced RAG pipeline.

    Flow:
      expand_query → dense_retrieve → bm25_retrieve → fuse_results
                   → rerank (cross-encoder) → [quality check]
                                                ├─ OK → generate → END
                                                └─ LOW → retry → expand_query
    """
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
# DEMO — Compare retrieval methods side-by-side
# ================================================================

def compare_retrieval_methods(query: str):
    """
    Show how each stage improves retrieval quality.

    This demonstrates WHY hybrid search + reranking matters:
    dense alone misses keyword matches, BM25 alone misses semantics,
    and without reranking the top results may not be the most relevant.
    """
    print(f"\n  {'─' * 60}")
    print(f"  RETRIEVAL COMPARISON for: '{query}'")
    print(f"  {'─' * 60}")

    # Dense only
    q_emb = EMBED_MODEL.encode(query).tolist()
    dense_raw = CHROMA_COLLECTION.query(
        query_embeddings=[q_emb], n_results=3,
        include=["metadatas", "distances"],
    )
    print(f"\n  Dense only (ChromaDB):")
    for i in range(3):
        title = dense_raw["metadatas"][0][i]["title"]
        dist = dense_raw["distances"][0][i]
        sim = 1.0 / (1.0 + dist)
        print(f"    #{i+1} [{sim:.3f}] {title}")

    # BM25 only
    scores = BM25_INDEX.get_scores(query.lower().split())
    bm25_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  BM25 only (keyword):")
    for rank, (idx, score) in enumerate(bm25_ranked):
        print(f"    #{rank+1} [{score:.2f}] {CORPUS[idx]['title']}")

    # Cross-encoder reranking of top 5 from each
    all_candidates = set()
    for i in range(min(5, len(dense_raw["metadatas"][0]))):
        all_candidates.add(dense_raw["metadatas"][0][i]["title"])
    for idx, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]:
        all_candidates.add(CORPUS[idx]["title"])

    candidate_docs = [doc for doc in CORPUS if doc["title"] in all_candidates]
    pairs = [(query, doc["content"]) for doc in candidate_docs]
    ce_scores = RERANKER.predict(pairs)

    reranked = sorted(
        zip(candidate_docs, ce_scores),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n  After cross-encoder reranking (hybrid + rerank):")
    for rank, (doc, score) in enumerate(reranked[:3]):
        print(f"    #{rank+1} [{score:.4f}] {doc['title']}")


def run_demo():
    app = build_advanced_rag_graph()

    queries = [
        "How does RLHF train AI models using human feedback?",
        "What defenses exist against prompt injection?",
        "DPO vs RLHF — which is simpler?",
        "What are the EU AI Act requirements for high-risk systems?",
    ]

    print("\n" + "=" * 65)
    print("  ADVANCED RAG — PRODUCTION PIPELINE")
    print("  (ChromaDB + BM25 + RRF + Cross-Encoder Reranking)")
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
            {"run_name": f"adv-rag-prod-{i + 1}"},
        )

        print(f"\n  Answer: {result['answer'][:350]}")
        print(f"  Sources: {', '.join(result['sources'])}")

    # Show side-by-side comparison for one query
    compare_retrieval_methods("DPO optimization without reward model")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 7b: Advanced RAG (Production Pipeline)")
    print("=" * 65)

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS — PRODUCTION vs. CONCEPT EXAMPLES")
    print("=" * 65)
    print(textwrap.dedent("""
    WHAT CHANGED from Example 7 (concept) to Example 7b (production):

    1. EMBEDDINGS: Word-frequency vectors → sentence-transformers
       - Real semantic understanding ("car" matches "automobile")
       - 384 dimensions vs. 18-dim toy vectors

    2. VECTOR STORE: Python list + manual cosine → ChromaDB
       - Persistent storage, metadata filtering, HNSW index
       - Scales to millions of documents

    3. BM25: Manual BM25 formula → rank_bm25 library
       - Proper tokenization, IDF computation, TF saturation
       - Battle-tested implementation used in production

    4. RERANKING: Title/first-sentence heuristic → CrossEncoder
       - Neural cross-attention between query and document
       - Trained on MS MARCO (250M+ query-passage pairs)
       - Dramatically improves precision (typically +14-20%)

    5. QUALITY CHECK: Fixed threshold → calibrated cross-encoder scores
       - Cross-encoder scores are roughly interpretable
       - > 2.0 = highly relevant, < 0 = irrelevant

    PRODUCTION DEPLOYMENT CHECKLIST:
    [ ] Switch ChromaDB to PersistentClient or HttpClient
    [ ] Add document chunking (200-500 tokens, recursive split)
    [ ] Add metadata filtering (date, category, source)
    [ ] Monitor retrieval quality (MRR, NDCG, recall@k)
    [ ] Cache embeddings to avoid re-computation
    [ ] Add logging and error handling for each stage
    """))
