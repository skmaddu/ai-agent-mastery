import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5: Basic RAG Pipeline — LangGraph + ChromaDB + Sentence-Transformers
==============================================================================
LangGraph RAG pipeline using REAL vector store (ChromaDB) and REAL
embeddings (sentence-transformers) — production-grade retrieval.

Pipeline:  embed_query → retrieve (ChromaDB) → generate (LLM) → END

This is a significant upgrade from the toy in-memory store in Example 4.
ChromaDB provides persistent vector storage, and sentence-transformers
produce semantic embeddings that capture meaning — not just word frequency.

Requirements (already in requirements.txt):
  pip install chromadb sentence-transformers

Phoenix tracing: YES — observe each pipeline stage in the dashboard.

Run: python week-05-context-memory/examples/example_05_basic_rag_langgraph.py
"""

import os
import sys
import math
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Phoenix Tracing ────────────────────────────────────────────
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
        print("[Phoenix] Not available.")
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week5-basic-rag")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception as e:
        print(f"[Phoenix] Setup failed: {e}")
        return None

# ── Cost Tracking ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from shared.utils.cost_tracker import CostTracker
    cost_tracker = CostTracker(weekly_budget=1.00)
except ImportError:
    cost_tracker = None

# ── LLM Setup ─────────────────────────────────────────────────

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
# CHROMADB VECTOR STORE + SENTENCE-TRANSFORMERS
# ================================================================
# ChromaDB is a lightweight, persistent vector database that runs
# embedded (no server needed).  Combined with sentence-transformers,
# this gives us REAL semantic search — not the toy word-frequency
# approach from Example 4.
#
# Why ChromaDB?
#   - Zero infrastructure: just pip install, no Docker/server
#   - Persistent by default: survives process restarts
#   - Built-in embedding support: can auto-embed with sentence-transformers
#   - Metadata filtering: filter by source, date, category
#   - Production-ready for small-to-medium deployments (~5M vectors)
#
# Why sentence-transformers?
#   - Free, local embeddings (no API costs)
#   - Captures SEMANTIC meaning ("car" ≈ "automobile" ≈ "vehicle")
#   - Model: all-MiniLM-L6-v2 — fast, 384-dim, good quality
#   - Runs on CPU, ~100ms per embedding

import chromadb
from sentence_transformers import SentenceTransformer

# Initialize the embedding model (loaded once, reused across queries)
# all-MiniLM-L6-v2: 384 dimensions, ~80MB, excellent quality/speed ratio
print("[INIT] Loading sentence-transformer model (first run downloads ~80MB)...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[INIT] Embedding model ready.")


# ================================================================
# KNOWLEDGE BASE — AI Safety Corpus
# ================================================================
# Same content as Example 4, but now indexed in a real vector store.

KNOWLEDGE_BASE = [
    {"id": "doc_0", "title": "AI Alignment Overview",
     "content": "AI alignment ensures AI systems act according to human values. Key approaches include RLHF, constitutional AI, and interpretability research. Misalignment ranges from minor annoyances to existential risks."},
    {"id": "doc_1", "title": "Prompt Injection Attacks",
     "content": "Prompt injection causes LLMs to ignore original instructions. Direct injection embeds instructions in user input. Indirect injection hides instructions in retrieved data. Defenses: input sanitization, instruction hierarchy, output filtering."},
    {"id": "doc_2", "title": "Red Teaming for AI",
     "content": "Red teaming probes AI systems for vulnerabilities before deployment. Techniques: adversarial prompting, jailbreak attempts, bias probes, automated attack generation. Best practice: continuous red-teaming, not just pre-launch."},
    {"id": "doc_3", "title": "AI Safety Regulations",
     "content": "EU AI Act (2025) classifies systems by risk level. High-risk systems need accuracy, robustness, human oversight. US Executive Order requires frontier model reporting. Global trend: mandatory risk assessment and incident reporting."},
    {"id": "doc_4", "title": "Interpretability Research",
     "content": "Interpretability aims to understand how neural networks make decisions. Techniques: attention visualization, SHAP, LIME, circuit analysis. Researchers can identify circuits for factual recall and language understanding."},
]


def create_vector_store() -> chromadb.Collection:
    """
    Create a ChromaDB collection and index the knowledge base.

    ChromaDB stores vectors + metadata + documents together.  We use
    an ephemeral client here (in-memory for the demo), but you can
    switch to persistent storage with one line change:

        client = chromadb.PersistentClient(path="./chroma_db")

    This would save the index to disk and survive process restarts —
    no need to re-embed documents every time.
    """
    # EphemeralClient = in-memory (fast for demos)
    # PersistentClient = saves to disk (production)
    client = chromadb.EphemeralClient()

    # Delete collection if it exists (clean start for demo)
    try:
        client.delete_collection("ai_safety_kb")
    except Exception:
        pass

    # Create collection — ChromaDB handles embedding storage internally
    collection = client.create_collection(
        name="ai_safety_kb",
        metadata={"description": "AI safety knowledge base for RAG demo"},
    )

    # Embed and index all documents
    print("\n  [INDEX] Embedding and indexing knowledge base with ChromaDB...")

    # Embed all documents in one batch (much faster than one-by-one)
    contents = [doc["content"] for doc in KNOWLEDGE_BASE]
    embeddings = EMBED_MODEL.encode(contents).tolist()

    # Add to ChromaDB with metadata
    collection.add(
        ids=[doc["id"] for doc in KNOWLEDGE_BASE],
        documents=contents,
        embeddings=embeddings,
        metadatas=[{"title": doc["title"]} for doc in KNOWLEDGE_BASE],
    )

    print(f"  [INDEX] Indexed {collection.count()} documents "
          f"(embedding dim: {len(embeddings[0])})")

    return collection


# Initialize the vector store at module load time
COLLECTION = create_vector_store()


# ================================================================
# STATE DEFINITION
# ================================================================

class RAGState(TypedDict):
    query: str                           # User's question
    retrieved_docs: List[Dict]           # Top-k retrieved documents with scores
    context: str                         # Formatted context for LLM
    answer: str                          # Generated answer
    sources: List[str]                   # Source citations


# ================================================================
# GRAPH NODES
# ================================================================

def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Retrieve the top-k most relevant documents from ChromaDB.

    ChromaDB handles embedding + similarity search in one call.
    We pass the raw query text, and ChromaDB:
      1. Embeds it using our embedding function
      2. Computes cosine similarity against all stored vectors
      3. Returns the top-k results with distances and metadata

    PRODUCTION NOTES:
    - ChromaDB distances are L2 by default; lower = more similar
    - We convert to a similarity score for readability
    - Metadata filtering can narrow search (e.g., by category, date)
    - For large collections, use HNSW index (ChromaDB default) for speed
    """
    query = state["query"]
    top_k = 3

    # Embed the query with the same model used for indexing
    # CRITICAL: always use the SAME embedding model for queries and documents
    query_embedding = EMBED_MODEL.encode(query).tolist()

    # Query ChromaDB — returns documents, distances, and metadata
    results = COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    # Parse ChromaDB results into a clean format
    # ChromaDB returns nested lists: results["documents"][0] = [doc1, doc2, ...]
    retrieved = []
    print(f"\n  [RETRIEVE] Query: '{query}'")
    print(f"  [RETRIEVE] Top {top_k} documents from ChromaDB:")

    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]

        # Convert L2 distance to a similarity score (0-1, higher = better)
        # L2 distance: 0 = identical, larger = more different
        similarity = 1.0 / (1.0 + distance)

        title = metadata.get("title", f"Doc {i}")
        retrieved.append({
            "title": title,
            "content": doc_text,
            "score": round(similarity, 4),
            "distance": round(distance, 4),
        })
        print(f"    #{i + 1} [sim={similarity:.3f}, dist={distance:.3f}] {title}")

    # Format context for the LLM
    context_parts = []
    sources = []
    for doc in retrieved:
        context_parts.append(f"[Source: {doc['title']}]\n{doc['content']}")
        sources.append(doc["title"])

    context = "\n\n".join(context_parts)

    return {
        "retrieved_docs": retrieved,
        "context": context,
        "sources": sources,
    }


llm = get_llm(temperature=0.3)  # Low temperature for factual Q&A


def generate_node(state: RAGState) -> dict:
    """
    Node 2: Generate an answer using the retrieved context.

    The prompt structure is critical for RAG quality:
      1. System message with citation instructions
      2. Retrieved context clearly labeled with source markers
      3. User query at the END (recency position for attention)

    PRODUCTION NOTES:
    - Always instruct the model to cite sources and admit uncertainty
    - Use LOW temperature (0-0.3) for factual Q&A
    - Track token usage for cost monitoring
    - The generate node is the ONLY node that calls the LLM
    """
    query = state["query"]
    context = state["context"]

    prompt = [
        SystemMessage(content=(
            "You are a knowledgeable AI safety expert. Answer the user's question "
            "based ONLY on the provided context. If the context doesn't contain "
            "enough information, say so. Always cite your sources using [Source: title] "
            "format. Be concise and accurate."
        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer (with source citations):"
        )),
    ]

    print(f"\n  [GENERATE] Sending to LLM with {len(context.split())} words of context...")

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Track cost
        if cost_tracker:
            usage = response.response_metadata.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            cost = cost_tracker.log_call(model_name, input_tokens, output_tokens)
            print(f"  [COST] {input_tokens} in + {output_tokens} out = ${cost:.6f}")

    except Exception as e:
        answer = f"[Error generating answer: {e}]"

    print(f"  [GENERATE] Answer: {answer[:150]}...")

    return {"answer": answer}


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================
#
# Pipeline: retrieve (ChromaDB) → generate (LLM) → END
#
# Note: No separate embed_query node needed — ChromaDB handles
# embedding internally when we pass the query text.  This simplifies
# the graph compared to a manual embedding approach.

def build_rag_graph():
    """Build the basic RAG pipeline as a LangGraph StateGraph."""

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ================================================================
# DEMO
# ================================================================

def run_rag_demo():
    """Run several queries through the RAG pipeline."""

    app = build_rag_graph()

    queries = [
        "What are the main approaches to AI alignment?",
        "How can I defend against prompt injection attacks?",
        "What are the key requirements of the EU AI Act?",
        "How do researchers understand what happens inside neural networks?",
    ]

    print("\n" + "=" * 65)
    print("  BASIC RAG PIPELINE — LANGGRAPH + CHROMADB")
    print("=" * 65)

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")

        result = app.invoke(
            {
                "query": query,
                "retrieved_docs": [],
                "context": "",
                "answer": "",
                "sources": [],
            },
            {"run_name": f"rag-query-{i + 1}"},
        )

        print(f"\n  Final Answer:")
        print(f"  {result['answer'][:400]}")
        print(f"\n  Sources: {', '.join(result['sources'])}")

    # Print cost summary
    if cost_tracker:
        print(f"\n{'=' * 65}")
        cost_tracker.report()


# ================================================================
# BONUS: Show ChromaDB collection info
# ================================================================

def show_collection_info():
    """Display ChromaDB collection statistics."""

    print(f"\n{'=' * 65}")
    print("  CHROMADB COLLECTION INFO")
    print(f"{'=' * 65}")
    print(f"  Collection name: {COLLECTION.name}")
    print(f"  Document count:  {COLLECTION.count()}")
    print(f"  Metadata:        {COLLECTION.metadata}")

    # Peek at stored documents
    peek = COLLECTION.peek(limit=3)
    print(f"\n  Sample documents:")
    for i, (doc_id, doc, meta) in enumerate(zip(
            peek["ids"], peek["documents"], peek["metadatas"])):
        print(f"    [{doc_id}] {meta['title']}: {doc[:60]}...")

    # Show what a raw query result looks like
    print(f"\n  Raw ChromaDB query example:")
    raw = COLLECTION.query(
        query_embeddings=[EMBED_MODEL.encode("alignment").tolist()],
        n_results=2,
    )
    print(f"    IDs: {raw['ids'][0]}")
    print(f"    Distances: {[round(d, 3) for d in raw['distances'][0]]}")

    # Production migration notes
    print(f"""
  ── PRODUCTION MIGRATION ──
  Current:    chromadb.EphemeralClient()      (in-memory, lost on restart)
  Production: chromadb.PersistentClient("./db")  (saved to disk)
  Scale up:   chromadb.HttpClient(host="...")     (dedicated server)

  Alternative vector stores:
    FAISS:    pip install faiss-cpu   (in-memory, very fast, no persistence)
    Pinecone: pip install pinecone    (cloud, scales to billions, paid)
    Weaviate: pip install weaviate    (cloud/self-hosted, GraphQL API)
    pgvector: PostgreSQL extension    (if you already use Postgres)
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 5: Basic RAG (LangGraph + ChromaDB)        ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()
    run_rag_demo()
    show_collection_info()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. ChromaDB + sentence-transformers = production-grade RAG with
       zero infrastructure.  Just pip install, no Docker or servers.

    2. sentence-transformers captures SEMANTIC meaning:
       "alignment" matches "human values" and "RLHF" — not just keywords.
       This is a massive upgrade over word-frequency embeddings.

    3. ALWAYS use the SAME embedding model for indexing and querying.
       Mixing models causes distribution mismatch → bad retrieval.

    4. ChromaDB returns L2 distances (lower = more similar).  Convert
       to similarity scores for human-readable output.

    5. For production: switch from EphemeralClient to PersistentClient
       (one line change) to persist the index across restarts.

    6. LangGraph makes the pipeline EXPLICIT and TRACEABLE.
       Each node (retrieve, generate) shows up in Phoenix traces.
    """))
