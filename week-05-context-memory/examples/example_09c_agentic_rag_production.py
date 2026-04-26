import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 9c: Agentic RAG — Production CRAG with Real Vector Store
==================================================================
PRODUCTION-GRADE Corrective RAG (CRAG) with real vector stores,
real embeddings, multi-collection routing, cross-encoder grounding
verification, and query refinement using the LLM.

This is the REAL implementation of the concepts taught in Examples 8-9.
Every component uses production libraries:

  - ChromaDB: separate collections per knowledge domain
  - sentence-transformers: real semantic embeddings
  - CrossEncoder: neural grounding verification
  - LLM-based query routing and refinement

The CRAG Pattern (Corrective RAG):
  The agent SELF-CORRECTS its retrieval strategy. Instead of blindly
  passing whatever it retrieves to the LLM, it:
    1. Routes the query to the best knowledge base
    2. Retrieves documents with real semantic search
    3. Verifies that retrieved docs actually answer the query (grounding)
    4. If not grounded, refines the query and re-retrieves
    5. Only generates when it has verified, high-quality context

LangGraph Pipeline:
  START → classify_query → route → retrieve (ChromaDB)
        → verify_grounding (CrossEncoder) → [grounded?]
                                              ├─ YES → generate → END
                                              └─ NO  → refine_query → route (max 3)

Requirements (already in requirements.txt):
  pip install chromadb sentence-transformers

Run: python week-05-context-memory/examples/example_09c_agentic_rag_production.py
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
        tracer_provider = register(project_name="week5-agentic-rag-production")
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
# REAL MODELS
# ================================================================

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

print("[INIT] Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-encoder for grounding verification.
# We use it to check: "Does this document actually answer this question?"
# This replaces the toy heuristic in Example 9.
print("[INIT] Loading cross-encoder for grounding verification...")
GROUNDING_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[INIT] Models ready.\n")


# ================================================================
# MULTI-COLLECTION KNOWLEDGE BASES — ChromaDB
# ================================================================
# In production, different knowledge domains live in SEPARATE
# collections (or even separate vector stores). The agent routes
# queries to the right collection based on topic classification.
#
# This is more realistic than searching one giant collection:
#   - Each collection can have domain-specific chunking
#   - Retrieval quality is higher (smaller, focused index)
#   - You can apply different access controls per collection

KB_RESEARCH = [
    {"id": "r0", "title": "RLHF Training Process",
     "content": "Reinforcement Learning from Human Feedback (RLHF) trains AI models using human preferences. The three-stage process involves supervised fine-tuning, reward model training on human preference comparisons, and policy optimization using PPO to maximize the learned reward signal."},
    {"id": "r1", "title": "DPO Method",
     "content": "Direct Preference Optimization (DPO) simplifies RLHF by directly optimizing the language model on preference pairs without a separate reward model. It reformulates the RL objective as a classification loss, making training simpler, more stable, and computationally cheaper."},
    {"id": "r2", "title": "Constitutional AI",
     "content": "Constitutional AI (CAI) uses a set of written principles to guide model self-improvement. The model generates, critiques, and revises its own outputs against constitutional rules. This reduces dependence on human annotation and scales alignment training."},
    {"id": "r3", "title": "Safety Alignment Research",
     "content": "Safety alignment ensures AI systems pursue intended goals without harmful side effects. Key approaches include RLHF for value learning, interpretability for understanding model internals, and robustness testing through red-teaming and adversarial evaluation."},
    {"id": "r4", "title": "Mechanistic Interpretability",
     "content": "Mechanistic interpretability reverse-engineers neural network computations. Circuit analysis identifies specific neuron pathways responsible for capabilities. Tools include activation patching, causal tracing, and sparse autoencoders for decomposing representations."},
]

KB_REGULATIONS = [
    {"id": "g0", "title": "EU AI Act Framework",
     "content": "The EU AI Act (effective 2025) classifies AI systems into four risk tiers: unacceptable (banned), high-risk (strict requirements including accuracy, robustness, human oversight), limited risk (transparency obligations), and minimal risk (no requirements). Violations face fines up to 35 million EUR."},
    {"id": "g1", "title": "US AI Executive Order",
     "content": "The US Executive Order on AI Safety requires frontier model developers to report safety test results to the government. It establishes the AI Safety Institute for independent testing and evaluation, and mandates watermarking of AI-generated content."},
    {"id": "g2", "title": "AI Compliance Requirements",
     "content": "AI compliance across jurisdictions requires: risk assessment documentation, bias testing and mitigation, data governance policies, human oversight mechanisms, incident reporting procedures, and regular auditing. Companies should implement AI governance frameworks proactively."},
]

KB_ENGINEERING = [
    {"id": "e0", "title": "RAG Pipeline Architecture",
     "content": "Production RAG pipelines combine document chunking (200-500 tokens, recursive splitting with overlap), embedding with sentence-transformers, hybrid search (dense + BM25), cross-encoder reranking, and grounding verification. Monitor with MRR, NDCG, and recall metrics."},
    {"id": "e1", "title": "Vector Store Selection",
     "content": "Choose vector stores by scale: FAISS for prototypes (up to 10M vectors, in-memory), ChromaDB for small production (persistent, zero-infra), Pinecone for scale (managed cloud, billions of vectors). Always benchmark retrieval quality before and after migration."},
    {"id": "e2", "title": "LangGraph State Management",
     "content": "LangGraph manages agent state via TypedDict with MemorySaver checkpointing. State persists across conversation turns with full time-travel debugging. Use reducers for complex state updates and conditional edges for dynamic routing."},
    {"id": "e3", "title": "Agent Memory Patterns",
     "content": "Agent memory types: short-term (conversation buffer within a session), long-term (vector store or database for cross-session facts), episodic (past interaction summaries), and semantic (structured knowledge graphs). Use hierarchical memory with hot cache, summary layer, and archive."},
]

ALL_KB_DOCS = {
    "research": KB_RESEARCH,
    "regulations": KB_REGULATIONS,
    "engineering": KB_ENGINEERING,
}


def build_collections() -> Dict[str, chromadb.Collection]:
    """
    Build separate ChromaDB collections for each knowledge domain.

    Each collection gets its own index — queries are routed to the
    right collection based on topic classification, not searched
    across all documents indiscriminately.
    """
    client = chromadb.EphemeralClient()
    collections = {}

    for domain, docs in ALL_KB_DOCS.items():
        try:
            client.delete_collection(f"crag_{domain}")
        except Exception:
            pass

        collection = client.create_collection(
            name=f"crag_{domain}",
            metadata={"domain": domain},
        )

        contents = [doc["content"] for doc in docs]
        embeddings = EMBED_MODEL.encode(contents).tolist()

        collection.add(
            ids=[doc["id"] for doc in docs],
            documents=contents,
            embeddings=embeddings,
            metadatas=[{"title": doc["title"], "domain": domain} for doc in docs],
        )

        collections[domain] = collection
        print(f"  [INDEX] Collection '{domain}': {collection.count()} docs")

    return collections


print("[INDEX] Building multi-collection knowledge base...")
COLLECTIONS = build_collections()
print()


# ================================================================
# STATE
# ================================================================

class CRAGState(TypedDict):
    query: str                          # Original user query
    current_query: str                  # Possibly refined query
    query_type: str                     # Classification: research/regulations/engineering
    route: str                          # Selected collection
    retrieved_docs: List[Dict]          # Retrieved documents
    grounding_scores: List[float]       # Cross-encoder grounding scores
    is_grounded: bool                   # Whether answer would be grounded
    best_grounding_score: float         # Highest grounding score
    retries: int                        # Current retry count
    max_retries: int                    # Safety limit
    refinement_reason: str              # Why the query was refined
    context: str                        # Formatted context for generation
    answer: str                         # Final answer
    sources: List[str]                  # Source citations


# ================================================================
# GRAPH NODES
# ================================================================

llm = get_llm(temperature=0.3)


def classify_and_route_node(state: CRAGState) -> dict:
    """
    Classify the query and route to the best knowledge collection.

    This uses a lightweight LLM call to classify the query domain.
    In production, you could also use:
      - A fine-tuned classifier (faster, cheaper)
      - Embedding similarity to collection centroids
      - Rule-based routing with keyword fallback

    We use the LLM here because it handles ambiguous queries well
    and can route to MULTIPLE collections for comparison queries.
    """
    query = state["current_query"]

    # Use LLM for classification
    classify_prompt = [
        SystemMessage(content=(
            "Classify this query into exactly ONE category. "
            "Respond with ONLY the category name, nothing else.\n\n"
            "Categories:\n"
            "- research: AI techniques, algorithms, training methods (RLHF, DPO, alignment, interpretability)\n"
            "- regulations: Laws, compliance, EU AI Act, executive orders, governance\n"
            "- engineering: RAG pipelines, vector stores, frameworks, deployment, memory systems\n"
        )),
        HumanMessage(content=f"Query: {query}"),
    ]

    try:
        response = llm.invoke(classify_prompt)
        route = response.content.strip().lower()
        # Validate the route
        if route not in COLLECTIONS:
            route = "research"  # Default fallback
    except Exception:
        # Fallback: keyword-based routing
        q_lower = query.lower()
        if any(w in q_lower for w in ["regulation", "eu", "act", "legal", "compliance"]):
            route = "regulations"
        elif any(w in q_lower for w in ["rag", "vector", "deploy", "memory", "langgraph"]):
            route = "engineering"
        else:
            route = "research"

    print(f"  [ROUTE] '{query[:50]}...' → collection: {route}")
    return {"route": route, "query_type": route}


def retrieve_node(state: CRAGState) -> dict:
    """
    Retrieve documents from the routed ChromaDB collection.

    Uses real sentence-transformer embeddings for semantic search.
    Returns top-k documents with similarity scores.
    """
    route = state["route"]
    query = state["current_query"]
    collection = COLLECTIONS[route]
    top_k = 3

    # Embed query with the same model used for indexing
    query_embedding = EMBED_MODEL.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
        similarity = 1.0 / (1.0 + distance)

        retrieved.append({
            "title": metadata["title"],
            "content": doc_text,
            "domain": metadata.get("domain", route),
            "similarity": round(similarity, 4),
        })

    print(f"  [RETRIEVE] From '{route}' — {len(retrieved)} docs:")
    for doc in retrieved:
        print(f"    [{doc['similarity']:.3f}] {doc['title']}")

    return {"retrieved_docs": retrieved}


def verify_grounding_node(state: CRAGState) -> dict:
    """
    Verify that retrieved documents can ground an answer to the query.

    This is the KEY innovation of CRAG: instead of blindly trusting
    retrieval scores, we use a cross-encoder to check whether each
    document ACTUALLY ANSWERS the question.

    Retrieval similarity != answer quality:
      - A document about "RLHF" might be retrieved for "How does RLHF work?"
        but only discuss RLHF history, not the mechanism → low grounding
      - A document with low embedding similarity might perfectly answer
        the question if it uses different vocabulary → high grounding

    The cross-encoder checks (query, document) relevance directly,
    catching these mismatches that embedding similarity misses.
    """
    query = state["query"]
    docs = state["retrieved_docs"]

    if not docs:
        print(f"  [GROUND] No documents retrieved — NOT grounded")
        return {
            "is_grounded": False,
            "best_grounding_score": 0.0,
            "grounding_scores": [],
        }

    # Cross-encoder scores each (query, document) pair
    pairs = [(query, doc["content"]) for doc in docs]
    scores = GROUNDING_MODEL.predict(pairs)

    grounding_scores = [round(float(s), 4) for s in scores]
    best_score = max(grounding_scores)

    # Grounding threshold: cross-encoder scores > 1.0 are typically relevant
    threshold = 1.0
    is_grounded = best_score >= threshold

    print(f"  [GROUND] Scores: {grounding_scores}")
    print(f"  [GROUND] Best: {best_score:.4f} (threshold: {threshold}) → "
          f"{'GROUNDED' if is_grounded else 'NOT GROUNDED'}")

    return {
        "is_grounded": is_grounded,
        "best_grounding_score": best_score,
        "grounding_scores": grounding_scores,
    }


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


def refine_query_node(state: CRAGState) -> dict:
    """
    Refine the query using the LLM to improve retrieval on the next attempt.

    Unlike Example 9 which appends static keywords, this uses the LLM
    to intelligently reformulate the query based on what was retrieved
    and why it wasn't sufficient.

    Strategies by retry:
      1st retry: Reformulate with more specific terms
      2nd retry: Try a different knowledge domain
      3rd retry: Broaden to a more general query
    """
    query = state["current_query"]
    retries = state["retries"]
    route = state["route"]

    if retries == 0:
        # First retry: ask LLM to reformulate
        refine_prompt = [
            SystemMessage(content=(
                "Reformulate this search query to be more specific and targeted. "
                "Add technical terms and be precise. Return ONLY the reformulated query."
            )),
            HumanMessage(content=f"Original query: {state['query']}"),
        ]
        try:
            response = llm.invoke(refine_prompt)
            refined = response.content.strip()
            reason = "LLM reformulation for specificity"
        except Exception:
            refined = query + " specific mechanism details technical explanation"
            reason = "keyword expansion (LLM fallback)"

    elif retries == 1:
        # Second retry: try a different collection
        alt_routes = [r for r in COLLECTIONS if r != route]
        new_route = alt_routes[0] if alt_routes else route
        refined = state["query"]  # Use original query with new route
        reason = f"switching from '{route}' to '{new_route}' collection"
        return {
            "current_query": refined,
            "route": new_route,
            "retries": retries + 1,
            "refinement_reason": reason,
        }

    else:
        # Third retry: broaden
        refined = state["query"] + " overview explanation fundamentals comprehensive guide"
        reason = "broadened query for maximum recall"

    print(f"  [REFINE] Strategy: {reason}")
    print(f"  [REFINE] '{query[:40]}' → '{refined[:60]}'")

    return {
        "current_query": refined,
        "retries": retries + 1,
        "refinement_reason": reason,
    }


def generate_node(state: CRAGState) -> dict:
    """
    Generate answer from verified, grounded context.

    The answer includes:
      - Source citations for traceability
      - Grounding scores for transparency
      - A confidence indicator based on grounding quality
    """
    docs = state["retrieved_docs"]
    scores = state.get("grounding_scores", [])

    # Pair docs with their grounding scores and sort by relevance
    if scores and len(scores) == len(docs):
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        docs_sorted = [d for d, _ in doc_score_pairs]
        scores_sorted = [s for _, s in doc_score_pairs]
    else:
        docs_sorted = docs
        scores_sorted = [0.0] * len(docs)

    # Build context from top grounded documents
    context_parts = []
    sources = []
    for doc, score in zip(docs_sorted[:3], scores_sorted[:3]):
        context_parts.append(
            f"[Source: {doc['title']}] (grounding: {score:.2f})\n{doc['content']}"
        )
        sources.append(doc["title"])

    context = "\n\n".join(context_parts)

    # Determine confidence level
    best = state.get("best_grounding_score", 0.0)
    if best >= 3.0:
        confidence = "HIGH"
    elif best >= 1.0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    prompt = [
        SystemMessage(content=(
            "You are an AI expert. Answer based ONLY on the provided context. "
            "Cite sources with [Source: title]. Be concise and accurate. "
            "If the context is insufficient to fully answer, explicitly state "
            "what information is missing."
        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}\n\n"
            f"Answer (cite sources):"
        )),
    ]

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [GENERATE] Confidence: {confidence} | {answer[:120]}...")
    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_crag_graph():
    """
    Build the production CRAG (Corrective RAG) pipeline.

    Flow:
      classify_and_route → retrieve (ChromaDB) → verify_grounding
                                                   ├─ GROUNDED → generate → END
                                                   └─ NOT → refine_query → classify_and_route
    """
    graph = StateGraph(CRAGState)

    graph.add_node("classify_and_route", classify_and_route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("verify_grounding", verify_grounding_node)
    graph.add_node("refine_query", refine_query_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("classify_and_route")
    graph.add_edge("classify_and_route", "retrieve")
    graph.add_edge("retrieve", "verify_grounding")

    graph.add_conditional_edges("verify_grounding", should_continue, {
        "generate": "generate",
        "refine": "refine_query",
    })

    # Refine loops back to retrieve (route may have changed)
    graph.add_edge("refine_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# ================================================================
# DEMO
# ================================================================

def run_demo():
    app = build_crag_graph()

    queries = [
        "How does RLHF train AI models?",
        "What are the EU AI Act requirements for high-risk AI?",
        "Best practices for production RAG pipelines",
        "Compare RLHF and DPO — which is better for alignment?",
        "What memory patterns should AI agents use?",
    ]

    print("\n" + "=" * 65)
    print("  AGENTIC RAG — PRODUCTION CRAG PIPELINE")
    print("  (ChromaDB multi-collection + CrossEncoder grounding)")
    print("=" * 65)

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")

        result = app.invoke(
            {
                "query": query, "current_query": query,
                "query_type": "", "route": "",
                "retrieved_docs": [],
                "grounding_scores": [],
                "is_grounded": False, "best_grounding_score": 0.0,
                "retries": 0, "max_retries": 3,
                "refinement_reason": "",
                "context": "", "answer": "", "sources": [],
            },
            {"run_name": f"crag-prod-{i + 1}"},
        )

        print(f"\n  Answer: {result['answer'][:350]}")
        print(f"  Sources: {', '.join(result['sources'])}")
        print(f"  Route: {result.get('route', '?')} | "
              f"Grounding: {result.get('best_grounding_score', 0):.2f} | "
              f"Retries: {result['retries']}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 9c: Agentic RAG / CRAG (Production)")
    print("=" * 65)

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS — PRODUCTION vs. CONCEPT EXAMPLES")
    print("=" * 65)
    print(textwrap.dedent("""
    WHAT CHANGED from Example 9 (concept) to Example 9c (production):

    1. VECTOR STORE: Python dict + manual cosine → ChromaDB collections
       - Separate collections per knowledge domain (research, regs, engineering)
       - Real persistent storage with metadata filtering
       - Scales to millions of documents per collection

    2. EMBEDDINGS: Word-frequency vectors → sentence-transformers
       - Real semantic understanding, 384-dim dense vectors
       - "How to train AI with preferences" matches RLHF documents

    3. GROUNDING CHECK: Max retrieval score heuristic → CrossEncoder
       - Neural model verifies (query, document) relevance directly
       - Catches cases where embedding similarity != answer quality
       - Much more reliable than score thresholds

    4. QUERY ROUTING: Keyword matching → LLM classification
       - Handles ambiguous queries ("compare RLHF and regulations")
       - Falls back to keyword routing if LLM fails

    5. QUERY REFINEMENT: Static keyword append → LLM reformulation
       - Intelligent reformulation based on what was retrieved
       - Multi-strategy: specificity → domain switch → broadening

    PRODUCTION CRAG ARCHITECTURE:
    ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
    │ Classify  │──→│ Retrieve │──→│   Verify     │──→│ Generate │
    │ & Route   │   │ ChromaDB │   │  Grounding   │   │  Answer  │
    │ (LLM)     │   │          │   │ (CrossEnc.)  │   │  (LLM)   │
    └──────────┘   └──────────┘   └──────┬───────┘   └──────────┘
         ^                                │
         │         ┌──────────┐           │ NOT GROUNDED
         └─────────│  Refine  │←──────────┘
                   │  Query   │
                   │  (LLM)   │
                   └──────────┘
    """))
