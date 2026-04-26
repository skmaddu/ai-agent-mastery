import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 3: Agentic RAG with Memory & Context Graphs
=======================================================
Difficulty: ⭐⭐⭐⭐ Expert | Time: 4 hours

PRODUCTION implementation using real vector stores (ChromaDB),
real embeddings (sentence-transformers), and cross-encoder
grounding verification.

Complete agentic RAG system in LangGraph that combines:
  a) Self-corrective retrieval (CRAG pattern) with real ChromaDB
  b) Persistent query memory (remembers past queries & results)
  c) Context graph construction (builds entity relationships)

Graph:
  START → check_memory → route_query → retrieve → check_groundedness
                                          ^               |
                                          |  (not grounded, retries < 3)
                                          +───────────────+
                                                          |
                                                (grounded)
                                                          v
                              update_graph → update_memory → generate → END

Requirements (already in requirements.txt):
  pip install chromadb sentence-transformers

Run: python week-05-context-memory/solutions/solution_03_agentic_rag_memory.py
"""

import os
import sys
import hashlib
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Real Embedding Model + Vector Store ───────────────────────
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

print("[INIT] Loading models...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
GROUNDING_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[INIT] Models ready.")


# ================================================================
# LLM Setup
# ================================================================

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
# Knowledge Bases — Indexed in ChromaDB
# ================================================================

KB_TECHNICAL = [
    {"id": "t0", "title": "RLHF Training", "content": "RLHF trains AI models using human preference rankings. A reward model learns from comparisons, then PPO optimizes the policy to maximize the reward signal."},
    {"id": "t1", "title": "DPO Method", "content": "DPO simplifies RLHF by directly optimizing on preference pairs without a separate reward model, treating alignment as classification."},
    {"id": "t2", "title": "RAG Architecture", "content": "RAG retrieves relevant documents and injects them into the LLM prompt. Hybrid search (dense + BM25) with reranking gives the best retrieval quality."},
]

KB_SAFETY = [
    {"id": "s0", "title": "Prompt Injection", "content": "Prompt injection attacks cause LLMs to follow attacker instructions. Defenses include input sanitization, instruction hierarchy, and output filtering."},
    {"id": "s1", "title": "AI Red Teaming", "content": "Red teaming probes AI for vulnerabilities using adversarial prompts, jailbreaks, and automated attack generation. Continuous red-teaming is best practice."},
]

ALL_KB_DOCS = {"technical": KB_TECHNICAL, "safety": KB_SAFETY}


def build_chromadb_collections() -> Dict[str, chromadb.Collection]:
    """Build separate ChromaDB collections for each knowledge domain."""
    client = chromadb.EphemeralClient()
    collections = {}

    for domain, docs in ALL_KB_DOCS.items():
        try:
            client.delete_collection(f"sol3_{domain}")
        except Exception:
            pass

        collection = client.create_collection(name=f"sol3_{domain}")
        contents = [d["content"] for d in docs]
        embeddings = EMBED_MODEL.encode(contents).tolist()

        collection.add(
            ids=[d["id"] for d in docs],
            documents=contents,
            embeddings=embeddings,
            metadatas=[{"title": d["title"], "domain": domain} for d in docs],
        )
        collections[domain] = collection
        print(f"  [INDEX] Collection '{domain}': {collection.count()} docs")

    return collections


print("[INDEX] Building ChromaDB collections...")
COLLECTIONS = build_chromadb_collections()


def query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]


# Persistent stores (live across invocations in this process)
QUERY_MEMORY: Dict[str, Dict] = {}     # hash → {answer, sources, timestamp}
CONTEXT_GRAPH: Dict[str, List[Dict]] = {}  # entity → [{relation, target, source}]


# ================================================================
# TODO 1: Define AgenticRAGMemoryState (SOLVED)
# ================================================================

class AgenticRAGMemoryState(TypedDict):
    query: str                      # original user query
    current_query: str              # possibly refined query
    route: str                      # selected knowledge base name
    retrieved_docs: List[Dict]      # retrieved documents
    is_grounded: bool               # groundedness check result
    groundedness_score: float       # 0-1 score
    retries: int                    # current retry count
    max_retries: int                # max retries (default 3)
    cache_hit: bool                 # whether query was in memory
    cached_answer: str              # answer from cache (if hit)
    new_entities: List[Dict]        # entities extracted this turn
    context: str                    # formatted context for generation
    answer: str                     # final answer
    sources: List[str]              # source citations


# ================================================================
# TODO 2: Implement check_memory_node (SOLVED)
# ================================================================

def check_memory_node(state: AgenticRAGMemoryState) -> dict:
    """Check query memory for cached results."""
    qh = query_hash(state["query"])

    if qh in QUERY_MEMORY:
        cached = QUERY_MEMORY[qh]
        print(f"    [Memory] Cache HIT for query hash {qh}")
        return {
            "cache_hit": True,
            "cached_answer": cached["answer"],
            "sources": cached.get("sources", []),
        }

    print(f"    [Memory] Cache MISS for query hash {qh}")
    return {"cache_hit": False}


def should_use_cache(state: AgenticRAGMemoryState) -> str:
    """Route: use cache or proceed to retrieval."""
    if state.get("cache_hit"):
        return "use_cache"
    return "route"


# ================================================================
# TODO 3: Implement route_query_node (SOLVED)
# ================================================================

SAFETY_KEYWORDS = ["safety", "injection", "attack", "red team", "red-team",
                   "jailbreak", "adversarial", "security", "vulnerability",
                   "defense", "prompt injection"]

def route_query_node(state: AgenticRAGMemoryState) -> dict:
    """Route query to the appropriate knowledge base."""
    query_lower = state["current_query"].lower()

    for keyword in SAFETY_KEYWORDS:
        if keyword in query_lower:
            print(f"    [Route] Matched safety keyword '{keyword}' → safety KB")
            return {"route": "safety"}

    print(f"    [Route] No safety keywords → technical KB")
    return {"route": "technical"}


# ================================================================
# TODO 4: Implement retrieve_node (SOLVED)
# ================================================================
# PRODUCTION UPGRADE: Uses ChromaDB with real sentence-transformer
# embeddings instead of toy word-frequency vectors.

def retrieve_node(state: AgenticRAGMemoryState) -> dict:
    """Retrieve documents from the routed ChromaDB collection."""
    route = state.get("route", "technical")
    collection = COLLECTIONS.get(route)

    if collection is None:
        print(f"    [Retrieve] Collection '{route}' not found, falling back to technical")
        collection = COLLECTIONS["technical"]

    query = state["current_query"]
    query_embedding = EMBED_MODEL.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(2, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    top_docs = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        similarity = 1.0 / (1.0 + distance)
        top_docs.append({
            "title": results["metadatas"][0][i]["title"],
            "content": results["documents"][0][i],
            "score": round(similarity, 4),
        })

    print(f"    [Retrieve] From '{route}' ChromaDB — top scores: {[f'{d['score']:.3f}' for d in top_docs]}")
    return {"retrieved_docs": top_docs}


# ================================================================
# TODO 5: Implement check_groundedness_node (SOLVED)
# ================================================================
# PRODUCTION UPGRADE: Uses cross-encoder to verify grounding instead
# of using retrieval score as a proxy. This is much more accurate.

def check_groundedness_node(state: AgenticRAGMemoryState) -> dict:
    """Check retrieval quality for groundedness using cross-encoder."""
    docs = state.get("retrieved_docs", [])

    if not docs:
        print(f"    [Grounded] No documents retrieved — NOT grounded")
        return {"is_grounded": False, "groundedness_score": 0.0}

    # Cross-encoder checks (query, document) relevance directly
    query = state["query"]
    pairs = [(query, doc["content"]) for doc in docs]
    scores = GROUNDING_MODEL.predict(pairs)

    max_score = float(max(scores))
    threshold = 1.0  # Cross-encoder scores > 1.0 are typically relevant
    is_grounded = max_score >= threshold

    print(f"    [Grounded] Cross-encoder scores: {[f'{s:.2f}' for s in scores]} "
          f"(threshold: {threshold}) → {'GROUNDED' if is_grounded else 'NOT grounded'}")
    return {"is_grounded": is_grounded, "groundedness_score": max_score}


def should_continue(state: AgenticRAGMemoryState) -> str:
    """Route: generate if grounded, refine if not."""
    if state.get("is_grounded"):
        return "update_graph"
    if state.get("retries", 0) >= state.get("max_retries", 3):
        return "update_graph"
    return "refine"


def refine_node(state: AgenticRAGMemoryState) -> dict:
    """Refine query for re-retrieval."""
    new_query = state["current_query"] + " details explanation"
    print(f"    [Refine] Retry {state.get('retries', 0) + 1}: '{new_query}'")
    return {
        "current_query": new_query,
        "retries": state.get("retries", 0) + 1,
    }


# ================================================================
# TODO 6: Implement update_graph_node (SOLVED)
# ================================================================

def update_graph_node(state: AgenticRAGMemoryState) -> dict:
    """Update the context graph with new entities from retrieval."""
    new_entities = []

    for doc in state.get("retrieved_docs", []):
        entity = doc["title"]
        content = doc["content"]

        # Extract keywords from content as related entities
        # Use simple heuristic: words that appear capitalized or are key terms
        words = content.split()
        keywords = set()
        for word in words:
            # Clean punctuation
            clean = word.strip(".,;:()[]")
            # Collect capitalized words (likely proper nouns/terms) and acronyms
            if clean and (clean[0].isupper() or clean.isupper()) and len(clean) > 2:
                keywords.add(clean)

        # Add edges to CONTEXT_GRAPH
        if entity not in CONTEXT_GRAPH:
            CONTEXT_GRAPH[entity] = []

        for kw in keywords:
            if kw != entity:
                edge = {"relation": "related_to", "target": kw, "source": doc["title"]}
                # Avoid duplicate edges
                if edge not in CONTEXT_GRAPH[entity]:
                    CONTEXT_GRAPH[entity].append(edge)
                    new_entities.append({"entity": entity, "target": kw})

    print(f"    [Graph] Added {len(new_entities)} entity relations. Total entities: {len(CONTEXT_GRAPH)}")
    return {"new_entities": new_entities}


# ================================================================
# TODO 7: Implement update_memory_node (SOLVED)
# ================================================================

def update_memory_node(state: AgenticRAGMemoryState) -> dict:
    """Save query result to memory for future cache hits."""
    import time

    qh = query_hash(state["query"])
    QUERY_MEMORY[qh] = {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "timestamp": time.time(),
    }

    print(f"    [Memory] Saved query hash {qh}. Total cached: {len(QUERY_MEMORY)}")
    return {}


# ================================================================
# Generate node (PROVIDED)
# ================================================================

llm = get_llm(temperature=0.3)

def generate_node(state: AgenticRAGMemoryState) -> dict:
    """Generate answer from retrieved context."""
    # If cache hit, just return the cached answer
    if state.get("cache_hit"):
        return {"answer": state.get("cached_answer", "")}

    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(f"[Source: {d['title']}]\n{d['content']}" for d in docs)

    # Add graph context if available
    graph_context = ""
    for entity, relations in CONTEXT_GRAPH.items():
        if entity.lower() in state["query"].lower():
            for rel in relations[:3]:
                graph_context += f"\n{entity} → {rel['relation']} → {rel['target']}"

    if graph_context:
        context += f"\n\nKnowledge graph context:{graph_context}"

    prompt = [
        SystemMessage(content="Answer based on the provided context. Cite sources with [Source: title]."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}"),
    ]

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    sources = [d["title"] for d in docs]
    return {"answer": answer, "sources": sources, "context": context}


# ================================================================
# TODO 8: Wire the graph (SOLVED)
# ================================================================

def build_agentic_rag_graph():
    """Build the complete agentic RAG + memory + graph system."""
    graph = StateGraph(AgenticRAGMemoryState)

    # Add all nodes
    graph.add_node("check_memory", check_memory_node)
    graph.add_node("route_query", route_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_groundedness", check_groundedness_node)
    graph.add_node("refine", refine_node)
    graph.add_node("update_graph", update_graph_node)
    graph.add_node("update_memory", update_memory_node)
    graph.add_node("generate", generate_node)

    # Set entry point
    graph.set_entry_point("check_memory")

    # Cache check: use_cache → generate, route → route_query
    graph.add_conditional_edges("check_memory", should_use_cache, {
        "use_cache": "generate",
        "route": "route_query",
    })

    # Route → retrieve → check groundedness
    graph.add_edge("route_query", "retrieve")
    graph.add_edge("retrieve", "check_groundedness")

    # Groundedness check: grounded → update_graph, not grounded → refine or give up
    graph.add_conditional_edges("check_groundedness", should_continue, {
        "update_graph": "update_graph",
        "refine": "refine",
    })

    # Refine loops back to retrieve
    graph.add_edge("refine", "retrieve")

    # update_graph → update_memory → generate → END
    graph.add_edge("update_graph", "generate")
    graph.add_edge("generate", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SOLUTION 3: Agentic RAG + Memory + Context Graphs")
    print("  (ChromaDB + sentence-transformers + cross-encoder)")
    print("=" * 65)

    app = build_agentic_rag_graph()

    queries = [
        "How does RLHF work?",
        "What defenses exist against prompt injection?",
        "How does RLHF work?",  # Should be a cache hit!
        "Compare RLHF and DPO approaches",
    ]

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")

        result = app.invoke({
            "query": query, "current_query": query,
            "route": "", "retrieved_docs": [],
            "is_grounded": False, "groundedness_score": 0.0,
            "retries": 0, "max_retries": 3,
            "cache_hit": False, "cached_answer": "",
            "new_entities": [],
            "context": "", "answer": "", "sources": [],
        })

        print(f"  Answer: {result.get('answer', '[no answer]')[:250]}")
        print(f"  Cache hit: {result.get('cache_hit', False)}")
        print(f"  Graph entities: {len(CONTEXT_GRAPH)}")
        print(f"  Memory entries: {len(QUERY_MEMORY)}")
