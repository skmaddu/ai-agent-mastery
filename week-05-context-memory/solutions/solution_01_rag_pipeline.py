import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 1: Build a Complete RAG Pipeline with Citations
==========================================================
Difficulty: ⭐⭐ Intermediate | Time: 2 hours

PRODUCTION implementation using real vector store (ChromaDB) and
real embeddings (sentence-transformers) — not toy word-frequency vectors.

Complete LangGraph RAG pipeline that:
  1. Chunks documents into manageable pieces
  2. Embeds chunks with sentence-transformers and stores in ChromaDB
  3. Retrieves the most relevant chunks using real semantic search
  4. Generates an answer WITH source citations

Graph:
  START → chunk_documents → index_in_chromadb → retrieve (ChromaDB) → generate_with_citations → END

Requirements (already in requirements.txt):
  pip install chromadb sentence-transformers

Run: python week-05-context-memory/solutions/solution_01_rag_pipeline.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Real Embedding Model + Vector Store ───────────────────────
import chromadb
from sentence_transformers import SentenceTransformer

print("[INIT] Loading sentence-transformer model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[INIT] Model ready.")


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
# TODO 1: Define the RAGPipelineState (SOLVED)
# ================================================================

class RAGPipelineState(TypedDict):
    documents: List[Dict]           # raw input documents
    chunks: List[Dict]              # chunked documents with metadata
    collection_name: str            # ChromaDB collection name
    query: str                      # user's question
    retrieved_chunks: List[Dict]    # top-k retrieved chunks with scores
    answer: str                     # generated answer with citations
    sources: List[str]              # list of source titles cited


# ChromaDB client (shared across invocations)
CHROMA_CLIENT = chromadb.EphemeralClient()


# ================================================================
# TODO 2: Implement chunk_documents_node (SOLVED)
# ================================================================

def chunk_documents_node(state: RAGPipelineState) -> dict:
    """Split documents into chunks with metadata."""
    all_chunks = []

    for doc in state["documents"]:
        title = doc["title"]
        content = doc["content"]

        # Split on sentence boundaries
        sentences = content.split(". ")
        # Re-add periods that were removed by splitting
        sentences = [s if s.endswith(".") else s + "." for s in sentences]

        current_chunk_sentences = []
        current_word_count = 0
        chunk_index = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            # If adding this sentence exceeds ~50 words, save current chunk
            if current_word_count + word_count > 50 and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                all_chunks.append({
                    "text": chunk_text,
                    "source": title,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                current_chunk_sentences = []
                current_word_count = 0

            current_chunk_sentences.append(sentence)
            current_word_count += word_count

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            all_chunks.append({
                "text": chunk_text,
                "source": title,
                "chunk_index": chunk_index,
            })

    print(f"  [Chunk] Created {len(all_chunks)} chunks from {len(state['documents'])} documents")
    return {"chunks": all_chunks}


# ================================================================
# TODO 3: Implement index_in_chromadb_node (SOLVED)
# ================================================================
# PRODUCTION UPGRADE: Instead of storing embeddings in a Python list,
# we index them in ChromaDB — a real, persistent vector store.

def index_in_chromadb_node(state: RAGPipelineState) -> dict:
    """Embed chunks with sentence-transformers and index in ChromaDB."""
    chunks = state["chunks"]
    collection_name = "rag_exercise_1"

    # Clean start — delete if exists
    try:
        CHROMA_CLIENT.delete_collection(collection_name)
    except Exception:
        pass

    collection = CHROMA_CLIENT.create_collection(
        name=collection_name,
        metadata={"description": "Exercise 1 RAG pipeline"},
    )

    # Batch embed all chunks with sentence-transformers
    texts = [c["text"] for c in chunks]
    embeddings = EMBED_MODEL.encode(texts).tolist()

    # Index in ChromaDB with metadata
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
    )

    print(f"  [Index] Indexed {collection.count()} chunks in ChromaDB "
          f"(dim={len(embeddings[0])})")
    return {"collection_name": collection_name}


# ================================================================
# TODO 4: Implement retrieve_node (SOLVED)
# ================================================================
# PRODUCTION UPGRADE: Uses ChromaDB semantic search instead of
# manual cosine similarity over a Python list.

def retrieve_node(state: RAGPipelineState) -> dict:
    """Retrieve top-k chunks from ChromaDB by semantic similarity."""
    collection = CHROMA_CLIENT.get_collection(state["collection_name"])
    query = state["query"]
    top_k = 3

    # Embed query with the SAME model used for indexing
    query_embedding = EMBED_MODEL.encode(query).tolist()

    # ChromaDB handles similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    top_chunks = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        similarity = 1.0 / (1.0 + distance)
        top_chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score": round(similarity, 4),
        })

    print(f"  [Retrieve] Top {top_k} chunks (scores: {[f'{c['score']:.3f}' for c in top_chunks]})")
    return {"retrieved_chunks": top_chunks}


# ================================================================
# TODO 5: Implement generate_with_citations_node (SOLVED)
# ================================================================

llm = get_llm(temperature=0.3)

def generate_with_citations_node(state: RAGPipelineState) -> dict:
    """Generate answer with source citations."""
    retrieved = state["retrieved_chunks"]

    # Build context from retrieved chunks
    context_parts = []
    for chunk in retrieved:
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # Build the prompt
    system_msg = SystemMessage(content=(
        "You are a helpful assistant that answers questions based on the provided context. "
        "Always cite your sources using the format [Source: title] after each claim. "
        "If the context doesn't contain enough information, say so."
    ))
    human_msg = HumanMessage(content=(
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        f"Answer with citations:"
    ))

    try:
        response = llm.invoke([system_msg, human_msg])
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error generating answer: {e}]"

    # Collect unique source titles
    sources = list(dict.fromkeys(chunk["source"] for chunk in retrieved))

    return {"answer": answer, "sources": sources}


# ================================================================
# TODO 6: Build and wire the graph (SOLVED)
# ================================================================

def build_rag_pipeline():
    """Build the complete RAG pipeline graph."""
    graph = StateGraph(RAGPipelineState)

    # Add all nodes
    graph.add_node("chunk_documents", chunk_documents_node)
    graph.add_node("index_in_chromadb", index_in_chromadb_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_with_citations", generate_with_citations_node)

    # Set entry point
    graph.set_entry_point("chunk_documents")

    # Wire edges: chunk → index in ChromaDB → retrieve → generate → END
    graph.add_edge("chunk_documents", "index_in_chromadb")
    graph.add_edge("index_in_chromadb", "retrieve")
    graph.add_edge("retrieve", "generate_with_citations")
    graph.add_edge("generate_with_citations", END)

    return graph.compile()


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SOLUTION 1: RAG Pipeline with Citations")
    print("  (ChromaDB + sentence-transformers — production pipeline)")
    print("=" * 65)

    app = build_rag_pipeline()

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
            "collection_name": "",
            "query": query,
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
        })

        print(f"\n  Answer: {result.get('answer', '[no answer]')[:300]}")
        print(f"  Sources: {result.get('sources', [])}")
