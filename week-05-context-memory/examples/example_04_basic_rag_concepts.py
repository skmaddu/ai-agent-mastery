import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 4: Basic RAG — Document Retrieval & Q&A Concepts
=========================================================
Pure-Python concept demonstration (no LLM calls required).

Covers:
  1. Naive Retrieve-Then-Generate Pipeline
  2. Chunking Strategies & Common Pitfalls
  3. Vector Store Basics (FAISS / Chroma / Pinecone)

RAG (Retrieval-Augmented Generation) is the #1 technique for giving
an LLM access to knowledge that isn't in its training data.  Instead
of fine-tuning, you RETRIEVE relevant documents at query time and
inject them into the context window.

Run: python week-05-context-memory/examples/example_04_basic_rag_concepts.py
"""

import math
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ================================================================
# SAMPLE KNOWLEDGE BASE
# ================================================================
# A small corpus about AI safety that we'll chunk and retrieve from.

KNOWLEDGE_BASE = [
    {
        "title": "AI Alignment Overview",
        "content": (
            "AI alignment is the field of research aimed at ensuring that artificial "
            "intelligence systems act in accordance with human values and intentions. "
            "The core challenge is that specifying human values precisely enough for a "
            "machine to optimize is extremely difficult. Misalignment can range from "
            "minor annoyances (a chatbot being unhelpful) to existential risks (an "
            "AGI pursuing goals incompatible with human survival). Key approaches "
            "include RLHF (reinforcement learning from human feedback), constitutional "
            "AI, and interpretability research."
        ),
    },
    {
        "title": "Prompt Injection Attacks",
        "content": (
            "Prompt injection is an attack where malicious input causes an LLM to "
            "ignore its original instructions and follow attacker-controlled prompts. "
            "Direct injection embeds instructions in user input. Indirect injection "
            "hides instructions in data the model retrieves (e.g., a web page or "
            "document). Defenses include input sanitization, instruction hierarchy "
            "(system > user > retrieved), output filtering, and sandboxing tool "
            "access. No single defense is sufficient; a layered approach is required."
        ),
    },
    {
        "title": "Red Teaming for AI Systems",
        "content": (
            "Red teaming involves systematically probing an AI system to discover "
            "vulnerabilities, biases, and failure modes before deployment. Techniques "
            "include adversarial prompting, jailbreak attempts, bias probes, and "
            "automated attack generation using attacker LLMs. A red team exercise "
            "typically produces a report with severity-rated findings and recommended "
            "mitigations. Best practice is to red-team continuously, not just before "
            "launch, as new attack vectors emerge regularly."
        ),
    },
    {
        "title": "AI Safety Regulations 2025-2026",
        "content": (
            "The EU AI Act (enforced 2025) classifies AI systems by risk level and "
            "imposes requirements ranging from transparency to prohibited practices. "
            "High-risk systems (medical, hiring, law enforcement) must meet strict "
            "standards for accuracy, robustness, and human oversight. The US Executive "
            "Order on AI Safety (2023) established reporting requirements for frontier "
            "models. China's Generative AI regulations require content labeling and "
            "safety assessments. The global trend is toward mandatory risk assessment "
            "and incident reporting for AI systems."
        ),
    },
    {
        "title": "Interpretability and Explainability",
        "content": (
            "Interpretability research aims to understand HOW neural networks make "
            "decisions. Key techniques include attention visualization, feature "
            "attribution (SHAP, LIME), circuit analysis (mechanistic interpretability), "
            "and probing classifiers. Explainability is the user-facing counterpart: "
            "providing understandable justifications for model outputs. The field has "
            "made significant progress — researchers can now identify specific circuits "
            "responsible for factual recall, language understanding, and even deception "
            "in large language models."
        ),
    },
]


# ================================================================
# 1. NAIVE RETRIEVE-THEN-GENERATE PIPELINE
# ================================================================
# The simplest RAG pipeline has three steps:
#
#   Query → [Embed] → [Retrieve top-k chunks] → [Generate answer]
#
# The "naive" version uses cosine similarity between the query
# embedding and each chunk embedding to find relevant documents.
#
# LIMITATIONS of naive RAG:
#   - Retrieval quality depends entirely on embedding similarity
#   - No query understanding (typos, synonyms, multi-hop questions)
#   - No verification of retrieved chunks' relevance
#   - No handling of contradictory sources

def simple_embed(text: str) -> List[float]:
    """
    Create a simple bag-of-words embedding (for demonstration).

    In production, use sentence-transformers or an embedding API:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text)

    Or via API:
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = response.data[0].embedding

    This simple version uses word frequency as a feature vector.
    """
    # Build vocabulary from the text
    words = text.lower().split()
    # Use a fixed vocabulary for consistency
    vocab = [
        "ai", "safety", "alignment", "model", "attack", "injection",
        "prompt", "risk", "human", "system", "learning", "data",
        "regulation", "red", "team", "research", "bias", "defense",
        "interpretability", "neural", "network", "training", "values",
        "rlhf", "feedback", "eu", "act", "explainability", "circuit",
    ]
    # Count frequency of each vocab word
    embedding = []
    for v in vocab:
        count = words.count(v)
        embedding.append(float(count))

    # Normalize to unit length
    magnitude = math.sqrt(sum(x * x for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_sim(a, b) = (a · b) / (||a|| × ||b||)

    Since our embeddings are already normalized, this simplifies to
    just the dot product.
    """
    return sum(x * y for x, y in zip(a, b))


@dataclass
class RetrievedChunk:
    """A document chunk with its relevance score."""
    title: str
    content: str
    score: float
    chunk_id: int


def naive_retrieve(query: str, chunks: List[Dict], top_k: int = 3) -> List[RetrievedChunk]:
    """
    Retrieve the top-k most similar chunks to the query.

    This is the core retrieval step of naive RAG.  It embeds the
    query, computes similarity with all chunk embeddings, and
    returns the top-k matches.
    """
    query_embedding = simple_embed(query)

    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = simple_embed(chunk["content"])
        score = cosine_similarity(query_embedding, chunk_embedding)
        scored_chunks.append(RetrievedChunk(
            title=chunk["title"],
            content=chunk["content"],
            score=score,
            chunk_id=i,
        ))

    # Sort by score descending and return top-k
    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    return scored_chunks[:top_k]


def demo_naive_rag():
    """Demonstrate the naive retrieve-then-generate pipeline."""

    print("=" * 65)
    print("  NAIVE RETRIEVE-THEN-GENERATE PIPELINE")
    print("=" * 65)

    queries = [
        "What is AI alignment and why does it matter?",
        "How do prompt injection attacks work?",
        "What regulations exist for AI systems?",
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        print(f"  {'─' * 55}")

        results = naive_retrieve(query, KNOWLEDGE_BASE, top_k=2)

        for rank, chunk in enumerate(results, 1):
            print(f"  #{rank} [{chunk.score:.3f}] {chunk.title}")
            print(f"       {chunk.content[:100]}...")

        # In a real pipeline, these chunks would be injected into the LLM prompt:
        # context = "\n\n".join(r.content for r in results)
        # prompt = f"Based on the following context:\n{context}\n\nAnswer: {query}"
        # answer = llm.invoke(prompt)

    print(f"\n  {'─' * 55}")
    print("  Pipeline: Query → Embed → Cosine Similarity → Top-K → LLM")
    print("  This is NAIVE because it has no query rewriting, reranking,")
    print("  or relevance verification.  We'll add those in Examples 6-9.")


# ================================================================
# 2. CHUNKING STRATEGIES & COMMON PITFALLS
# ================================================================
# Before retrieval, documents must be split into chunks.  The
# chunking strategy has a HUGE impact on retrieval quality.
#
# PITFALLS:
#   1. Chunks too small → lose context (sentence fragments)
#   2. Chunks too large → dilute relevance (bury the answer)
#   3. Chunks split mid-sentence → broken semantics
#   4. No overlap → miss information at chunk boundaries

@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int

    @property
    def token_estimate(self) -> int:
        return max(1, int(len(self.text.split()) / 0.75))


def chunk_fixed_size(text: str, source: str, chunk_size: int = 200,
                     overlap: int = 50) -> List[Chunk]:
    """
    Fixed-size chunking with overlap.

    The simplest strategy: split text every N characters with an
    overlap window to preserve context at boundaries.

    Args:
        text: Full document text
        source: Document identifier
        chunk_size: Characters per chunk
        overlap: Characters of overlap between consecutive chunks
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(
            text=text[start:end],
            source=source,
            chunk_index=idx,
            char_start=start,
            char_end=end,
        ))
        start += chunk_size - overlap
        idx += 1
    return chunks


def chunk_by_sentence(text: str, source: str,
                      max_chunk_tokens: int = 100) -> List[Chunk]:
    """
    Sentence-based chunking (respects sentence boundaries).

    Better than fixed-size because it never splits mid-sentence.
    Groups sentences until the chunk reaches the token limit.
    """
    # Simple sentence splitting (production: use spacy or nltk)
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    chunks = []
    current_chunk = ""
    chunk_start = 0
    idx = 0
    char_pos = 0

    for sentence in sentences:
        test_chunk = (current_chunk + " " + sentence).strip()
        test_tokens = max(1, int(len(test_chunk.split()) / 0.75))

        if test_tokens > max_chunk_tokens and current_chunk:
            chunks.append(Chunk(
                text=current_chunk,
                source=source,
                chunk_index=idx,
                char_start=chunk_start,
                char_end=char_pos,
            ))
            idx += 1
            current_chunk = sentence
            chunk_start = char_pos
        else:
            current_chunk = test_chunk

        char_pos += len(sentence) + 1

    if current_chunk:
        chunks.append(Chunk(
            text=current_chunk,
            source=source,
            chunk_index=idx,
            char_start=chunk_start,
            char_end=char_pos,
        ))

    return chunks


def chunk_recursive(text: str, source: str,
                    max_chunk_tokens: int = 100) -> List[Chunk]:
    """
    Recursive character text splitting (LangChain's default strategy).

    Tries to split on paragraph boundaries first, then sentences,
    then words, then characters.  This preserves the most natural
    text boundaries possible.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int = 0) -> List[str]:
        if max(1, int(len(text.split()) / 0.75)) <= max_chunk_tokens:
            return [text]
        if sep_idx >= len(separators):
            # Last resort: hard split at character boundary
            mid = len(text) // 2
            return [text[:mid], text[mid:]]

        sep = separators[sep_idx]
        parts = text.split(sep)
        result = []
        current = ""

        for part in parts:
            test = (current + sep + part).strip() if current else part
            if max(1, int(len(test.split()) / 0.75)) > max_chunk_tokens:
                if current:
                    result.extend(_split(current, sep_idx + 1))
                current = part
            else:
                current = test

        if current:
            result.extend(_split(current, sep_idx + 1))
        return result

    texts = _split(text)
    chunks = []
    char_pos = 0
    for idx, t in enumerate(texts):
        chunks.append(Chunk(
            text=t,
            source=source,
            chunk_index=idx,
            char_start=char_pos,
            char_end=char_pos + len(t),
        ))
        char_pos += len(t)

    return chunks


def demo_chunking_strategies():
    """Compare different chunking approaches on the same document."""

    print("\n" + "=" * 65)
    print("  CHUNKING STRATEGIES & COMMON PITFALLS")
    print("=" * 65)

    # Use a longer document for chunking
    doc = KNOWLEDGE_BASE[0]["content"] + " " + KNOWLEDGE_BASE[1]["content"]
    doc_tokens = max(1, int(len(doc.split()) / 0.75))
    print(f"\n  Document: {doc_tokens} tokens\n")

    strategies = [
        ("Fixed-size (200 chars, 50 overlap)",
         chunk_fixed_size(doc, "test", 200, 50)),
        ("Sentence-based (100 token max)",
         chunk_by_sentence(doc, "test", 100)),
        ("Recursive (100 token max)",
         chunk_recursive(doc, "test", 100)),
    ]

    for name, chunks in strategies:
        print(f"  ── {name} ──")
        print(f"  Produced {len(chunks)} chunks:")
        for c in chunks:
            print(f"    [{c.chunk_index}] ({c.token_estimate:>3} tokens) "
                  f"{c.text[:60]}...")
        print()

    print("  PITFALL CHECKLIST:")
    print("  ✗ Fixed-size can split mid-word/sentence → broken semantics")
    print("  ✓ Sentence-based respects boundaries → better retrieval")
    print("  ✓ Recursive tries natural splits first → best quality")
    print("  ✓ Always add overlap for fixed-size to catch boundary info")


# ================================================================
# 3. VECTOR STORE BASICS (FAISS / Chroma / Pinecone)
# ================================================================
# A vector store indexes embeddings for fast similarity search.
# The three most common options in 2026:
#
#   FAISS (Facebook AI Similarity Search):
#     - In-memory, extremely fast for local use
#     - No server needed, pip install faiss-cpu
#     - Best for: prototypes, single-machine deployments
#
#   ChromaDB:
#     - Lightweight, persistent, good DX
#     - Runs embedded or as a server
#     - Best for: small-to-medium production deployments
#
#   Pinecone:
#     - Fully managed cloud service
#     - Scales to billions of vectors
#     - Best for: large-scale production with SLA requirements

class SimpleVectorStore:
    """
    A minimal vector store implementation for learning.

    In production, replace this with FAISS, Chroma, or Pinecone.
    This implementation uses brute-force search (O(n) per query),
    which is fine for small collections but doesn't scale.
    """

    def __init__(self):
        self.documents: List[Dict] = []      # {text, metadata, embedding}
        self.dimension: Optional[int] = None

    def add(self, text: str, metadata: Optional[Dict] = None) -> int:
        """Add a document to the store. Returns the document ID."""
        embedding = simple_embed(text)
        if self.dimension is None:
            self.dimension = len(embedding)

        doc_id = len(self.documents)
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "embedding": embedding,
        })
        return doc_id

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for the top-k most similar documents."""
        query_embedding = simple_embed(query)
        scored = []
        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc["embedding"])
            scored.append({**doc, "score": score})

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]

    def stats(self) -> Dict:
        """Return store statistics."""
        return {
            "num_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": "brute_force (O(n))",
        }


def demo_vector_store():
    """Demonstrate vector store operations."""

    print("\n" + "=" * 65)
    print("  VECTOR STORE BASICS")
    print("=" * 65)

    store = SimpleVectorStore()

    # Index the knowledge base
    print("\n  Indexing knowledge base...")
    for doc in KNOWLEDGE_BASE:
        doc_id = store.add(doc["content"], {"title": doc["title"]})
        print(f"    Added doc {doc_id}: {doc['title']}")

    stats = store.stats()
    print(f"\n  Store stats: {stats}")

    # Search
    queries = [
        "How to defend against prompt injection?",
        "What is RLHF?",
        "EU AI Act requirements",
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        results = store.search(query, top_k=2)
        for r in results:
            print(f"    [{r['score']:.3f}] {r['metadata']['title']}")

    # Production comparison
    print(f"\n  {'─' * 55}")
    print("  VECTOR STORE COMPARISON (Production):")
    print(f"  {'─' * 55}")

    comparison = [
        ("Feature",         "FAISS",           "ChromaDB",        "Pinecone"),
        ("Hosting",         "In-memory",       "Embedded/Server", "Cloud SaaS"),
        ("Scale",           "~10M vectors",    "~5M vectors",     "Billions"),
        ("Persistence",     "Manual save",     "Built-in",        "Managed"),
        ("Filtering",       "Post-search",     "Metadata filter", "Server-side"),
        ("Setup",           "pip install",     "pip install",     "API key"),
        ("Cost",            "Free",            "Free/Paid",       "$70+/mo"),
        ("Best for",        "Prototypes",      "Small prod",      "Large scale"),
    ]

    for row in comparison:
        print(f"  {row[0]:<16} {row[1]:<18} {row[2]:<18} {row[3]}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 4: Basic RAG Concepts                      ║")
    print("╚" + "═" * 63 + "╝")

    demo_naive_rag()
    demo_chunking_strategies()
    demo_vector_store()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. RAG = Retrieve relevant documents + Generate answer from them.
       It's the primary way to give LLMs access to private/current data.

    2. Chunking strategy matters MORE than embedding model choice.
       Use recursive splitting with sentence boundaries, not fixed-size.

    3. Chunk size sweet spot: 100-500 tokens.  Smaller = precise but
       may lack context.  Larger = more context but dilutes relevance.

    4. Vector stores: FAISS for prototypes, ChromaDB for small production,
       Pinecone for enterprise scale.  All use the same embed→index→search
       pattern.

    5. Naive RAG is a starting point, not a production solution.  It needs
       reranking (Topic 5), query rewriting, and grounding checks (Topic 6).
    """))
