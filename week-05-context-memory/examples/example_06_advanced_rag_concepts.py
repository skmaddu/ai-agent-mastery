import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6: Advanced RAG — Reranking, Hybrid Search & Query Rewriting
=====================================================================
Pure-Python concept demonstration (no LLM calls required).

Covers:
  1. Hybrid Search (Dense + Sparse/BM25 + Reciprocal Rank Fusion)
  2. Reranking Techniques (Cross-Encoder, LLM-as-Judge)
  3. Query Rewriting & HyDE (Hypothetical Document Embeddings)
  4. Evaluation Metrics & Accuracy Improvements

Basic RAG (Example 4-5) suffers from three problems:
  • Embedding similarity misses keyword matches (semantic gap)
  • Top-k results may include irrelevant documents (precision)
  • User queries are often vague or poorly formed (query quality)

Advanced RAG solves these with hybrid search, reranking, and
query rewriting.

Run: python week-05-context-memory/examples/example_06_advanced_rag_concepts.py
"""

import math
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ================================================================
# SAMPLE CORPUS
# ================================================================

CORPUS = [
    {"id": 0, "title": "RLHF Process",
     "content": "Reinforcement Learning from Human Feedback trains AI models using human preferences. Annotators rank model outputs, creating a reward model. The policy is then optimized using PPO to maximize the reward signal."},
    {"id": 1, "title": "Constitutional AI",
     "content": "Constitutional AI uses a set of principles to self-critique and revise outputs. The model generates responses, then evaluates them against constitutional rules. This reduces the need for human annotation."},
    {"id": 2, "title": "Direct Preference Optimization",
     "content": "DPO simplifies RLHF by directly optimizing the policy on preference data without a separate reward model. It treats the problem as classification: preferred vs dispreferred responses."},
    {"id": 3, "title": "Prompt Injection Defense",
     "content": "Defending against prompt injection requires multiple layers: input sanitization strips dangerous patterns, instruction hierarchy ensures system prompts take precedence, and output filtering catches information leaks."},
    {"id": 4, "title": "AI Red Teaming",
     "content": "Red teaming uses adversarial techniques to find AI vulnerabilities. Teams employ jailbreak prompts, social engineering, and automated attack generation to stress-test safety guardrails."},
    {"id": 5, "title": "Model Interpretability",
     "content": "Mechanistic interpretability reverse-engineers neural network computations. Circuit analysis identifies specific neuron pathways responsible for capabilities like factual recall and logical reasoning."},
    {"id": 6, "title": "AI Safety Benchmarks",
     "content": "Safety benchmarks evaluate model robustness against adversarial attacks, bias, and harmful outputs. Key benchmarks include HarmBench, TruthfulQA, and BBQ for bias measurement."},
    {"id": 7, "title": "Alignment Tax",
     "content": "The alignment tax refers to the performance cost of safety training. RLHF and safety filters can reduce model capability on benign tasks. Minimizing this tax while maintaining safety is an active research area."},
]


# ================================================================
# 1. HYBRID SEARCH (Dense + BM25 + Reciprocal Rank Fusion)
# ================================================================
# Dense search (embeddings) captures SEMANTIC similarity:
#   "How do you train AI with human preferences?" → matches RLHF
#
# Sparse search (BM25) captures KEYWORD matches:
#   "DPO optimization" → matches DPO article directly
#
# Hybrid search combines both, using Reciprocal Rank Fusion (RRF)
# to merge the ranked lists.  This catches documents that either
# approach might miss.

def bm25_score(query: str, document: str, k1: float = 1.5, b: float = 0.75,
               avg_doc_len: float = 50.0) -> float:
    """
    BM25 scoring function (simplified Okapi BM25).

    BM25 is the gold standard for keyword-based retrieval.  It's what
    search engines use before neural methods.  The formula considers:
      - Term frequency (tf): how often the query term appears in the doc
      - Inverse document frequency (idf): rarer terms are more important
      - Document length normalization: longer docs aren't unfairly boosted

    Args:
        query: Search query
        document: Document text
        k1: Term frequency saturation parameter
        b: Length normalization parameter
        avg_doc_len: Average document length in words
    """
    query_terms = query.lower().split()
    doc_terms = document.lower().split()
    doc_len = len(doc_terms)
    doc_tf = Counter(doc_terms)

    # Simplified IDF (in production, compute across full corpus)
    total_docs = len(CORPUS)
    score = 0.0

    for term in query_terms:
        tf = doc_tf.get(term, 0)
        # Count docs containing this term
        df = sum(1 for d in CORPUS if term in d["content"].lower())
        if df == 0:
            continue

        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm

    return score


def dense_search(query: str, top_k: int = 5) -> List[Dict]:
    """Dense (embedding-based) search using cosine similarity."""
    vocab = ["ai", "safety", "alignment", "model", "rlhf", "feedback",
             "human", "train", "reward", "preference", "attack", "injection",
             "defense", "red", "team", "interpretability", "circuit", "benchmark"]

    def embed(text):
        words = text.lower().split()
        emb = [float(words.count(v)) for v in vocab]
        mag = math.sqrt(sum(x * x for x in emb))
        return [x / mag for x in emb] if mag > 0 else emb

    q_emb = embed(query)
    scored = []
    for doc in CORPUS:
        d_emb = embed(doc["content"])
        score = sum(a * b for a, b in zip(q_emb, d_emb))
        scored.append({"id": doc["id"], "title": doc["title"],
                       "content": doc["content"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def sparse_search(query: str, top_k: int = 5) -> List[Dict]:
    """Sparse (BM25) keyword search."""
    scored = []
    for doc in CORPUS:
        score = bm25_score(query, doc["content"])
        scored.append({"id": doc["id"], "title": doc["title"],
                       "content": doc["content"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def reciprocal_rank_fusion(ranked_lists: List[List[Dict]],
                           k: int = 60) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) — merge multiple ranked lists.

    RRF formula: score(doc) = Σ 1 / (k + rank_i(doc))

    The parameter k controls how much weight top-ranked items get.
    k=60 is the standard value from the original paper.

    This is the SIMPLEST and most robust fusion method.  Unlike
    weighted averages, it doesn't require score normalization and
    works well even when the score scales differ wildly.
    """
    fused_scores: Dict[int, float] = {}
    doc_data: Dict[int, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = doc

    # Sort by fused score
    result = []
    for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = round(score, 4)
        result.append(doc)

    return result


def demo_hybrid_search():
    """Compare dense, sparse, and hybrid search on different queries."""

    print("=" * 65)
    print("  HYBRID SEARCH: Dense + BM25 + Reciprocal Rank Fusion")
    print("=" * 65)

    queries = [
        "How does RLHF work?",                      # Both should find it
        "DPO optimization without reward model",     # Keyword-heavy → BM25 wins
        "making AI listen to people",                # Semantic → Dense wins
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        print(f"  {'─' * 55}")

        dense_results = dense_search(query, top_k=3)
        sparse_results = sparse_search(query, top_k=3)
        hybrid_results = reciprocal_rank_fusion(
            [dense_results, sparse_results], k=60
        )[:3]

        print(f"  {'Method':<10} {'#1':^20} {'#2':^20} {'#3':^20}")
        for method, results in [("Dense", dense_results),
                                 ("BM25", sparse_results),
                                 ("Hybrid", hybrid_results)]:
            titles = [r["title"][:18] for r in results[:3]]
            while len(titles) < 3:
                titles.append("—")
            print(f"  {method:<10} {titles[0]:^20} {titles[1]:^20} {titles[2]:^20}")

    print(f"\n  Takeaway: Hybrid search combines the strengths of both.")
    print(f"  Dense catches semantic matches; BM25 catches exact keywords.")


# ================================================================
# 2. RERANKING TECHNIQUES
# ================================================================
# After initial retrieval (dense, sparse, or hybrid), a RERANKER
# refines the ranking.  The reranker is more accurate but slower
# (it processes each doc individually), so it's applied to a
# smaller candidate set (top 10-20 from initial retrieval).
#
#   CROSS-ENCODER RERANKING:
#     A BERT-like model that takes (query, document) as input and
#     outputs a relevance score.  Much more accurate than embedding
#     similarity because it can attend to both texts simultaneously.
#
#   LLM-AS-JUDGE RERANKING:
#     Use an LLM to score each document's relevance.  More flexible
#     but more expensive than a cross-encoder.

def simulate_cross_encoder_rerank(query: str,
                                   documents: List[Dict]) -> List[Dict]:
    """
    Simulate cross-encoder reranking.

    In production:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, doc["content"]) for doc in documents])

    Here we simulate with a heuristic that boosts documents where
    query terms appear in important positions (title, first sentence).
    """
    reranked = []
    query_terms = set(query.lower().split())

    for doc in documents:
        # Base score from initial retrieval
        base_score = doc.get("rrf_score", doc.get("score", 0))

        # Boost for title match
        title_terms = set(doc["title"].lower().split())
        title_overlap = len(query_terms & title_terms)
        title_boost = title_overlap * 0.1

        # Boost for first-sentence match
        first_sentence = doc["content"].split(".")[0].lower()
        first_terms = set(first_sentence.split())
        first_overlap = len(query_terms & first_terms)
        first_boost = first_overlap * 0.05

        rerank_score = base_score + title_boost + first_boost
        reranked.append({**doc, "rerank_score": round(rerank_score, 4)})

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def demo_reranking():
    """Show how reranking improves retrieval precision."""

    print("\n" + "=" * 65)
    print("  RERANKING: Cross-Encoder & LLM-as-Judge")
    print("=" * 65)

    query = "How to make AI models safer using human feedback?"

    # Initial hybrid retrieval
    dense = dense_search(query, top_k=5)
    sparse = sparse_search(query, top_k=5)
    hybrid = reciprocal_rank_fusion([dense, sparse])[:5]

    # Rerank
    reranked = simulate_cross_encoder_rerank(query, hybrid)

    print(f"\n  Query: '{query}'\n")
    print(f"  Before reranking (hybrid):")
    for i, doc in enumerate(hybrid[:5]):
        print(f"    #{i + 1} [{doc.get('rrf_score', 0):.4f}] {doc['title']}")

    print(f"\n  After reranking (cross-encoder):")
    for i, doc in enumerate(reranked[:5]):
        print(f"    #{i + 1} [{doc['rerank_score']:.4f}] {doc['title']}")

    print(f"\n  Reranking methods comparison:")
    print(f"  {'Method':<20} {'Latency':>10} {'Cost':>10} {'Quality':>10}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(f"  {'Cross-Encoder':<20} {'~50ms':>10} {'Free':>10} {'High':>10}")
    print(f"  {'LLM-as-Judge':<20} {'~500ms':>10} {'$$':>10} {'Highest':>10}")
    print(f"  {'Cohere Rerank API':<20} {'~100ms':>10} {'$':>10} {'High':>10}")


# ================================================================
# 3. QUERY REWRITING & HyDE
# ================================================================
# Users write BAD queries.  They're vague, misspelled, or use
# different vocabulary than the documents.  Query rewriting fixes this.
#
#   QUERY EXPANSION: Add synonyms and related terms
#     "RLHF" → "RLHF reinforcement learning human feedback training"
#
#   QUERY DECOMPOSITION: Split complex queries into sub-queries
#     "Compare RLHF and DPO" → ["What is RLHF?", "What is DPO?"]
#
#   HyDE (Hypothetical Document Embeddings):
#     Generate a HYPOTHETICAL answer, embed THAT instead of the query.
#     The hypothesis is closer to the document space than the question.

def simulate_query_expansion(query: str) -> str:
    """
    Simulate query expansion with synonym injection.

    In production, use an LLM:
        expanded = llm.invoke(f"Expand this query with synonyms: {query}")
    """
    expansion_map = {
        "rlhf": "RLHF reinforcement learning human feedback preference",
        "dpo": "DPO direct preference optimization",
        "safe": "safe safety alignment guardrails",
        "attack": "attack adversarial injection jailbreak",
        "interpret": "interpretability explainability mechanistic circuit",
    }
    expanded = query
    for key, expansion in expansion_map.items():
        if key in query.lower():
            expanded += " " + expansion
    return expanded


def simulate_hyde(query: str) -> str:
    """
    Simulate HyDE — Hypothetical Document Embeddings.

    Instead of embedding the QUESTION, we generate a hypothetical
    ANSWER and embed that.  The hypothesis lives in "document space",
    making it closer to actual documents than the question is.

    In production: hypothesis = llm.invoke(f"Write a short paragraph
    answering: {query}")
    """
    # Simulated hypothetical answers
    hypotheticals = {
        "rlhf": "RLHF works by collecting human preference data where annotators compare model outputs. A reward model is trained on these preferences, then the language model is fine-tuned using PPO to maximize the reward signal.",
        "safety": "AI safety involves multiple approaches including alignment research, red teaming, interpretability, and regulatory compliance. The goal is ensuring AI systems behave as intended without causing harm.",
        "default": f"This document discusses {query}. It covers the key concepts, methods, and best practices related to the topic.",
    }
    for key, hyp in hypotheticals.items():
        if key in query.lower():
            return hyp
    return hypotheticals["default"]


def demo_query_rewriting():
    """Show how query rewriting improves retrieval."""

    print("\n" + "=" * 65)
    print("  QUERY REWRITING & HyDE")
    print("=" * 65)

    query = "How does RLHF work?"

    print(f"\n  Original query: '{query}'")

    # Expansion
    expanded = simulate_query_expansion(query)
    print(f"\n  Expanded query: '{expanded[:80]}...'")

    # HyDE
    hypothesis = simulate_hyde(query)
    print(f"\n  HyDE hypothesis: '{hypothesis[:80]}...'")

    # Compare retrieval with each approach
    print(f"\n  Retrieval comparison:")
    for method, search_query in [("Original", query),
                                  ("Expanded", expanded),
                                  ("HyDE", hypothesis)]:
        results = dense_search(search_query, top_k=3)
        titles = [f"{r['title'][:20]}({r['score']:.2f})" for r in results]
        print(f"    {method:<10}: {', '.join(titles)}")


# ================================================================
# 4. EVALUATION METRICS
# ================================================================

def demo_evaluation_metrics():
    """Show how to measure RAG pipeline quality."""

    print("\n" + "=" * 65)
    print("  EVALUATION METRICS & ACCURACY IMPROVEMENTS")
    print("=" * 65)

    # Simulated evaluation results
    print(f"""
  Standard RAG metrics:

  ┌─────────────────────┬──────────┬───────────────┬──────────────┐
  │ Metric              │ Basic RAG│ + Hybrid      │ + Reranking  │
  ├─────────────────────┼──────────┼────────��──────┼──────────────┤
  │ Recall@5            │   72%    │     85%       │     85%      │
  │ Precision@3         │   55%    │     68%       │     82%      │
  │ MRR (Mean Recip.    │   0.61   │     0.74      │     0.88     │
  │      Rank)          │          │               │              │
  │ NDCG@5              │   0.58   │     0.72      │     0.85     │
  │ Answer Correctness  │   65%    │     75%       │     83%      │
  │ Faithfulness        │   78%    │     82%       │     90%      │
  └─────────────────────┴──────────┴───────────────┴─��────────────┘

  Key metrics explained:
    Recall@k    — % of relevant docs found in top-k results
    Precision@k — % of top-k results that are relevant
    MRR         — 1/rank of first relevant result (higher = better)
    NDCG@k      — Normalized Discounted Cumulative Gain (rank quality)
    Correctness — Does the final answer match ground truth?
    Faithfulness— Is the answer grounded in retrieved context?

  Each technique improves different metrics:
    Hybrid search → Recall (+13%) — finds more relevant documents
    Reranking     → Precision (+14%) — promotes the best documents
    Query rewrite → Recall (+5-10%) — catches synonym mismatches
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 6: Advanced RAG Concepts                   ║")
    print("╚" + "═" * 63 + "╝")

    demo_hybrid_search()
    demo_reranking()
    demo_query_rewriting()
    demo_evaluation_metrics()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Hybrid search (dense + BM25 + RRF) catches both semantic and
       keyword matches.  Use it by default in production RAG.

    2. Reranking with a cross-encoder improves precision by 14%+ over
       initial retrieval.  Always rerank the top 10-20 candidates.

    3. Query rewriting (expansion, decomposition, HyDE) fixes poor
       user queries.  HyDE is especially effective for technical domains.

    4. Measure with the right metrics: recall for coverage, precision
       for relevance, faithfulness for hallucination prevention.

    5. Each technique stacks: hybrid + reranking + query rewriting
       can improve answer correctness from 65% to 83%+.
    """))
