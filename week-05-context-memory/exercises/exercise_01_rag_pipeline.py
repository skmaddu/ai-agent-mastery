import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 1: Build a Complete RAG Pipeline with Citations
==========================================================
Difficulty: ⭐⭐ Intermediate | Time: 2 hours

Task:
Build a LangGraph RAG pipeline that:
  1. Chunks documents into manageable pieces
  2. Creates simple embeddings for each chunk
  3. Retrieves the most relevant chunks for a query
  4. Generates an answer WITH source citations

Graph:
  START → chunk_documents → embed_chunks → retrieve → generate_with_citations → END

Instructions:
1. Define the RAGPipelineState TypedDict with all required fields (TODO 1)
2. Implement chunk_documents_node to split documents into chunks (TODO 2)
3. Implement embed_and_store_node to embed each chunk (TODO 3)
4. Implement retrieve_node with top-k similarity search (TODO 4)
5. Implement generate_with_citations_node that cites sources (TODO 5)
6. Wire all graph edges together (TODO 6)

Hints:
- Study example_05_basic_rag_langgraph.py for the basic RAG pattern
- Use the simple_embed() function provided for embeddings
- Chunk size of ~100 words works well for this exercise
- The generate node should include "[Source: X]" citations
- Use cosine similarity for retrieval (formula provided)

Run: python week-05-context-memory/exercises/exercise_01_rag_pipeline.py
"""

import os
import sys
import math
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ================================================================
# LLM Setup
# ================================================================

def get_llm(temperature=0.7):
    """Create LLM based on provider setting."""
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
# Helper Functions (PROVIDED — do not modify)
# ================================================================

VOCAB = ["ai", "safety", "model", "train", "data", "learning", "agent",
         "memory", "context", "token", "prompt", "retrieval", "vector",
         "search", "embed", "chunk", "document", "query", "answer",
         "knowledge", "graph", "neural", "network", "transformer"]


def simple_embed(text: str) -> List[float]:
    """Create a simple word-frequency embedding vector."""
    words = text.lower().split()
    emb = [float(words.count(v)) for v in VOCAB]
    mag = math.sqrt(sum(x * x for x in emb))
    return [x / mag for x in emb] if mag > 0 else emb


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    return sum(x * y for x, y in zip(a, b))


# Sample documents to index
DOCUMENTS = [
    {
        "title": "AI Agent Memory Systems",
        "content": (
            "AI agents use memory systems to maintain context across conversations. "
            "Short-term memory stores recent messages within a session. Long-term memory "
            "persists facts and preferences across sessions using vector stores or databases. "
            "Hierarchical memory uses multiple layers: a hot cache for recent data, a summary "
            "layer for older context, and an archive for key facts. This approach balances "
            "detail with token efficiency. Memory management is critical for production agents "
            "because without it, every conversation starts from scratch."
        ),
    },
    {
        "title": "RAG Pipeline Architecture",
        "content": (
            "Retrieval-Augmented Generation combines document retrieval with LLM generation. "
            "The pipeline starts by chunking documents into smaller pieces, typically 100-500 "
            "tokens each. These chunks are embedded into vector representations using models "
            "like sentence-transformers. At query time, the user's question is embedded and "
            "compared against stored vectors using cosine similarity. The top-k most similar "
            "chunks are retrieved and injected into the LLM prompt as context. The LLM then "
            "generates an answer grounded in the retrieved documents."
        ),
    },
    {
        "title": "Context Engineering Best Practices",
        "content": (
            "Context engineering treats the LLM context window as a managed resource. "
            "The four pillars are Write (what enters), Select (what stays), Compress "
            "(how to shrink), and Isolate (how to separate concerns). Key practices include "
            "using hierarchical windowing to preserve graduated detail, applying token budgets "
            "to each context zone, and implementing automatic summarization when context grows "
            "too large. The lost-in-the-middle problem means important information should be "
            "placed at the start or end of context, never buried in the middle."
        ),
    },
    {
        "title": "Vector Store Selection Guide",
        "content": (
            "Choosing a vector store depends on scale and requirements. FAISS works well for "
            "prototypes with up to 10 million vectors, running entirely in memory. ChromaDB "
            "offers persistent storage suitable for small production deployments. Pinecone "
            "provides a fully managed cloud service that scales to billions of vectors. For "
            "most AI agent projects, start with a simple in-memory store, then migrate to "
            "ChromaDB when persistence is needed, and Pinecone when scale demands it. Always "
            "benchmark retrieval quality before and after migrating stores."
        ),
    },
]


# ================================================================
# TODO 1: Define the RAGPipelineState
# ================================================================
# Define a TypedDict with these fields:
#   - documents: List[Dict]       — raw input documents
#   - chunks: List[Dict]          — chunked documents with metadata
#   - chunk_embeddings: List[Dict] — chunks with their embeddings
#   - query: str                  — user's question
#   - query_embedding: List[float] — embedded query vector
#   - retrieved_chunks: List[Dict] — top-k retrieved chunks
#   - answer: str                 — generated answer with citations
#   - sources: List[str]          — list of source titles cited

class RAGPipelineState(TypedDict):
    # TODO: Define all the fields listed above
    pass


# ================================================================
# TODO 2: Implement chunk_documents_node
# ================================================================
# Split each document's content into chunks of approximately
# `chunk_size` words.  Each chunk should include metadata:
#   {"text": str, "source": str, "chunk_index": int}
#
# Strategy: Split on sentence boundaries (". ") and group
# sentences until you reach the chunk_size.

def chunk_documents_node(state: RAGPipelineState) -> dict:
    """Split documents into chunks with metadata."""
    # TODO: Implement document chunking
    # Hint: for each document, split content by ". " to get sentences,
    # then group sentences until word count reaches chunk_size (~50 words)
    pass


# ================================================================
# TODO 3: Implement embed_and_store_node
# ================================================================
# Embed each chunk using simple_embed() and store the embedding
# alongside the chunk text and metadata.

def embed_and_store_node(state: RAGPipelineState) -> dict:
    """Embed each chunk and the query."""
    # TODO: Embed each chunk and the query using simple_embed()
    # Return: chunk_embeddings (list of {text, source, chunk_index, embedding})
    #         query_embedding (the embedded query vector)
    pass


# ================================================================
# TODO 4: Implement retrieve_node
# ================================================================
# Find the top-k most similar chunks to the query using cosine
# similarity.  Return the top 3 chunks.

def retrieve_node(state: RAGPipelineState) -> dict:
    """Retrieve top-k chunks by similarity to query."""
    # TODO: Compare query_embedding against each chunk_embedding
    # using cosine_similarity(), sort by score, return top 3
    # Include the similarity score in each result
    pass


# ================================================================
# TODO 5: Implement generate_with_citations_node
# ================================================================
# Generate an answer using the LLM with retrieved context.
# The prompt should instruct the model to cite sources.

llm = get_llm(temperature=0.3)

def generate_with_citations_node(state: RAGPipelineState) -> dict:
    """Generate answer with source citations."""
    # TODO: Build a prompt with:
    #   1. System message instructing citation format [Source: title]
    #   2. Retrieved chunks as context
    #   3. The user's query
    # Invoke the LLM and return the answer and sources list
    pass


# ================================================================
# TODO 6: Build and wire the graph
# ================================================================

def build_rag_pipeline():
    """Build the complete RAG pipeline graph."""
    graph = StateGraph(RAGPipelineState)

    # TODO: Add all nodes to the graph
    # TODO: Set entry point
    # TODO: Add edges: chunk → embed → retrieve → generate → END
    # TODO: Compile and return

    pass


# ================================================================
# Test (runs when you've completed all TODOs)
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  EXERCISE 1: RAG Pipeline with Citations                      ║")
    print("╚" + "═" * 63 + "╝")

    app = build_rag_pipeline()
    if app is None:
        print("\n  [!] Complete all TODOs before running.")
        sys.exit(0)

    queries = [
        "How do AI agents manage memory across conversations?",
        "What is the best vector store for a prototype?",
        "What are the four pillars of context engineering?",
    ]

    for query in queries:
        print(f"\n{'━' * 65}")
        print(f"  Query: {query}")
        print(f"{'━' * 65}")

        result = app.invoke({
            "documents": DOCUMENTS,
            "chunks": [],
            "chunk_embeddings": [],
            "query": query,
            "query_embedding": [],
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
        })

        print(f"\n  Answer: {result.get('answer', '[no answer]')[:300]}")
        print(f"  Sources: {result.get('sources', [])}")
