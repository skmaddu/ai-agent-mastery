import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 3: Agentic RAG with Memory & Context Graphs
=======================================================
Difficulty: ⭐⭐⭐⭐ Expert | Time: 4 hours

Task:
Build an agentic RAG system in LangGraph that combines:
  a) Self-corrective retrieval (CRAG pattern)
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

The agent:
  1. Checks memory for similar past queries (skip retrieval if cached)
  2. Routes to the best knowledge base
  3. Retrieves documents
  4. Checks if results are grounded enough
  5. If not, refines query and re-retrieves (max 3 retries)
  6. Updates the context graph with new entities/relations
  7. Saves the query→result pair to memory for future cache hits
  8. Generates the final answer

Instructions:
1. Define AgenticRAGMemoryState (TODO 1)
2. Implement check_memory_node (TODO 2)
3. Implement route_query_node (TODO 3)
4. Implement retrieve_node (TODO 4)
5. Implement check_groundedness_node (TODO 5)
6. Implement update_graph_node (TODO 6)
7. Implement update_memory_node (TODO 7)
8. Wire the complete graph with conditional edges (TODO 8)

Hints:
- Study example_09_agentic_rag_langgraph.py for the CRAG pattern
- Study example_20_context_graphs_langgraph.py for context graphs
- Study example_14_langgraph_state_memory.py for state memory
- Query memory is a simple dict: {query_hash: {answer, sources, timestamp}}
- Context graph is a dict: {entity: [{relation, target, source_doc}]}

Run: python week-05-context-memory/exercises/exercise_03_agentic_rag_memory.py
"""

import os
import sys
import math
import hashlib
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


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
# Knowledge Bases (PROVIDED)
# ================================================================

KB_TECHNICAL = [
    {"title": "RLHF Training", "content": "RLHF trains AI models using human preference rankings. A reward model learns from comparisons, then PPO optimizes the policy to maximize the reward signal."},
    {"title": "DPO Method", "content": "DPO simplifies RLHF by directly optimizing on preference pairs without a separate reward model, treating alignment as classification."},
    {"title": "RAG Architecture", "content": "RAG retrieves relevant documents and injects them into the LLM prompt. Hybrid search (dense + BM25) with reranking gives the best retrieval quality."},
]

KB_SAFETY = [
    {"title": "Prompt Injection", "content": "Prompt injection attacks cause LLMs to follow attacker instructions. Defenses include input sanitization, instruction hierarchy, and output filtering."},
    {"title": "AI Red Teaming", "content": "Red teaming probes AI for vulnerabilities using adversarial prompts, jailbreaks, and automated attack generation. Continuous red-teaming is best practice."},
]

ALL_KBS = {"technical": KB_TECHNICAL, "safety": KB_SAFETY}


# ================================================================
# Helpers (PROVIDED)
# ================================================================

VOCAB = ["ai", "safety", "rlhf", "dpo", "model", "policy", "reward",
         "human", "attack", "injection", "rag", "retrieval", "search"]


def simple_embed(text: str) -> List[float]:
    words = text.lower().split()
    e = [float(words.count(v)) for v in VOCAB]
    m = math.sqrt(sum(x * x for x in e))
    return [x / m for x in e] if m > 0 else e


def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))


def query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]


# Persistent stores (live across invocations in this process)
QUERY_MEMORY: Dict[str, Dict] = {}     # hash → {answer, sources, timestamp}
CONTEXT_GRAPH: Dict[str, List[Dict]] = {}  # entity → [{relation, target, source}]


# ================================================================
# TODO 1: Define AgenticRAGMemoryState
# ================================================================
# Fields:
#   - query: str                    — original user query
#   - current_query: str            — possibly refined query
#   - route: str                    — selected knowledge base name
#   - retrieved_docs: List[Dict]    — retrieved documents
#   - is_grounded: bool             — groundedness check result
#   - groundedness_score: float     — 0-1 score
#   - retries: int                  — current retry count
#   - max_retries: int              — max retries (default 3)
#   - cache_hit: bool               — whether query was in memory
#   - cached_answer: str            — answer from cache (if hit)
#   - new_entities: List[Dict]      — entities extracted this turn
#   - context: str                  — formatted context for generation
#   - answer: str                   — final answer
#   - sources: List[str]            — source citations

class AgenticRAGMemoryState(TypedDict):
    # TODO: Define all fields
    pass


# ================================================================
# TODO 2: Implement check_memory_node
# ================================================================
# Check if a similar query exists in QUERY_MEMORY.
# If cache hit, set cache_hit=True and cached_answer.
# If miss, set cache_hit=False.

def check_memory_node(state: AgenticRAGMemoryState) -> dict:
    """Check query memory for cached results."""
    # TODO: Hash the query, look up in QUERY_MEMORY
    # If found, return cache_hit=True, cached_answer=..., sources=...
    # If not, return cache_hit=False
    pass


def should_use_cache(state: AgenticRAGMemoryState) -> str:
    """Route: use cache or proceed to retrieval."""
    if state.get("cache_hit"):
        return "use_cache"
    return "route"


# ================================================================
# TODO 3: Implement route_query_node
# ================================================================
# Route to "technical" or "safety" KB based on keywords.

def route_query_node(state: AgenticRAGMemoryState) -> dict:
    """Route query to the appropriate knowledge base."""
    # TODO: Check for safety-related keywords → "safety"
    # Otherwise → "technical"
    pass


# ================================================================
# TODO 4: Implement retrieve_node
# ================================================================
# Search the selected KB using cosine similarity.

def retrieve_node(state: AgenticRAGMemoryState) -> dict:
    """Retrieve documents from the routed knowledge base."""
    # TODO: Get KB docs from ALL_KBS[route], embed query,
    # compute similarity, return top 2 results
    pass


# ================================================================
# TODO 5: Implement check_groundedness_node
# ================================================================
# Check if retrieved docs can ground an answer.

def check_groundedness_node(state: AgenticRAGMemoryState) -> dict:
    """Check retrieval quality for groundedness."""
    # TODO: Use max retrieval score as proxy for groundedness
    # Threshold: 0.3
    pass


def should_continue(state: AgenticRAGMemoryState) -> str:
    """Route: generate if grounded, refine if not."""
    if state.get("is_grounded"):
        return "update_graph"
    if state.get("retries", 0) >= state.get("max_retries", 3):
        return "update_graph"
    return "refine"


def refine_node(state: AgenticRAGMemoryState) -> dict:
    """Refine query for re-retrieval."""
    return {
        "current_query": state["current_query"] + " details explanation",
        "retries": state.get("retries", 0) + 1,
    }


# ================================================================
# TODO 6: Implement update_graph_node
# ================================================================
# Extract entities and relations from retrieved docs, add to CONTEXT_GRAPH.
# Simple approach: extract capitalized phrases as entities, link them
# to the document topic.

def update_graph_node(state: AgenticRAGMemoryState) -> dict:
    """Update the context graph with new entities from retrieval."""
    # TODO: For each retrieved doc:
    #   1. Extract entity = doc title
    #   2. Extract keywords from content as related entities
    #   3. Add edges to CONTEXT_GRAPH
    # Return new_entities list
    pass


# ================================================================
# TODO 7: Implement update_memory_node
# ================================================================
# Save the query→answer pair to QUERY_MEMORY for future cache hits.

def update_memory_node(state: AgenticRAGMemoryState) -> dict:
    """Save query result to memory for future cache hits."""
    # TODO: Store in QUERY_MEMORY using query_hash as key
    # Value: {answer, sources, timestamp}
    pass


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
# TODO 8: Wire the graph
# ================================================================

def build_agentic_rag_graph():
    """Build the complete agentic RAG + memory + graph system."""
    graph = StateGraph(AgenticRAGMemoryState)

    # TODO: Add all nodes
    # TODO: Set entry point to "check_memory"
    # TODO: Add conditional edges for cache check (use_cache → generate, route → route_query)
    # TODO: Add edges: route → retrieve → check_groundedness
    # TODO: Add conditional edges for groundedness (update_graph or refine → route)
    # TODO: Add edges: update_graph → update_memory → generate → END

    pass


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  EXERCISE 3: Agentic RAG + Memory + Context Graphs           ║")
    print("╚" + "═" * 63 + "╝")

    app = build_agentic_rag_graph()
    if app is None:
        print("\n  [!] Complete all TODOs before running.")
        sys.exit(0)

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
