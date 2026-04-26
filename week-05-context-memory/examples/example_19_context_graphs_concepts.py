import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 19: Context Graphs — Concepts & Pure-Python Demo
==========================================================
Pure-Python concept demonstration of context graphs for agents.

Covers:
  1. Context Graphs vs Pure Vector Stores vs GraphRAG
  2. Temporal Knowledge Graphs (Zep Graphiti + Neo4j)
  3. Hybrid Vector + Graph Architecture
  4. Multi-Hop Reasoning & Token Efficiency Benefits
  5. When to Use Context Graphs (vs Vector-Only)

Context graphs represent knowledge as entities and relationships rather
than flat text chunks.  This enables multi-hop reasoning ("Who does
Alice's manager report to?") that vector search alone struggles with,
because the answer spans multiple documents that share no surface-level
similarity.

Run: python week-05-context-memory/examples/example_19_context_graphs_concepts.py
"""

import time
import math
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from collections import deque


# ================================================================
# 1. CONTEXT GRAPHS vs PURE VECTOR STORES vs GRAPHRAG
# ================================================================
# Three main paradigms for giving agents structured knowledge:
#
# PURE VECTOR STORE:
#   - Chunks text into fixed-size pieces, embeds each, stores in FAISS/Chroma.
#   - Retrieval: cosine similarity between query embedding and chunk embeddings.
#   - Strengths: simple, fast, good for "find me something about X".
#   - Weakness: no structural awareness.  Can't answer "how are X and Y related?"
#     without hoping both appear in the same chunk.
#
# GRAPHRAG (Microsoft):
#   - Pre-builds a knowledge graph from documents (entity extraction + clustering).
#   - Queries hit communities/clusters rather than individual chunks.
#   - Strengths: global summarization, theme detection across corpus.
#   - Weakness: expensive build step, slower query, heavy LLM usage.
#
# CONTEXT GRAPHS (Zep Graphiti-style):
#   - Incremental: entities and relations added as conversation proceeds.
#   - Temporal: edges carry timestamps, enabling "what changed since last week?"
#   - Lightweight: no pre-build; grows organically with each interaction.
#   - Strengths: multi-hop reasoning, temporal awareness, token-efficient context.
#   - Weakness: needs entity resolution logic; graph can fragment without care.

def demo_paradigm_comparison():
    """Print a comparison table and trade-off analysis."""
    print("\n" + "=" * 70)
    print("  1. CONTEXT GRAPHS vs VECTOR STORES vs GRAPHRAG")
    print("=" * 70)

    print(textwrap.dedent("""
  ┌───────────────────┬──────────────────┬───────────────────┬──────────────────┐
  │ Dimension         │ Vector Store     │ GraphRAG          │ Context Graph    │
  ├───────────────────┼──────────────────┼───────────────────┼──────────────────┤
  │ Build cost        │ Low (embed once) │ High (LLM calls)  │ Low (incremental)│
  │ Query latency     │ ~10ms            │ ~500ms-2s         │ ~5-50ms          │
  │ Multi-hop Q&A     │ Poor             │ Good (clusters)   │ Excellent        │
  │ Temporal queries  │ None             │ None              │ Native           │
  │ Token efficiency  │ 500-2000 tokens  │ 300-1000 tokens   │ 50-300 tokens    │
  │ Incremental add   │ Easy             │ Hard (rebuild)    │ Easy             │
  │ Global summaries  │ Poor             │ Excellent         │ Moderate         │
  │ Setup complexity  │ Minimal          │ High              │ Moderate         │
  └───────────────────┴──────────────────┴───────────────────┴──────────────────┘

  WHEN EACH APPROACH WINS:

  Vector Store wins when:
    - You have a large, static document corpus (docs, manuals, FAQs).
    - Questions are "find me content about topic X" (single-hop).
    - You need the simplest possible implementation.

  GraphRAG wins when:
    - You need corpus-wide summarization ("What are the main themes?").
    - Your documents are interconnected and you can afford the build cost.
    - Queries are thematic rather than entity-specific.

  Context Graphs win when:
    - Your agent has ongoing conversations with evolving knowledge.
    - Questions require multi-hop reasoning ("Who manages Alice's team?").
    - You need temporal awareness ("What changed since last meeting?").
    - Token budget is tight and you want surgical context injection.
    """))


# ================================================================
# 2. TEMPORAL KNOWLEDGE GRAPHS (Zep Graphiti + Neo4j)
# ================================================================
# Temporal knowledge graphs attach TIME to relationships, not just
# entities.  This matters because facts change:
#
#   "Alice manages the ML team" was true in 2024-01,
#   "Bob manages the ML team" became true in 2024-06.
#
# Zep Graphiti stores temporal edges in Neo4j with:
#   - valid_from: when the relationship became true
#   - valid_until: when it stopped being true (None if still active)
#   - episode_id: which conversation/document introduced this fact
#
# Benefits for agents:
#   - Answer "who managed the ML team in March?" accurately
#   - Detect contradictions: "user said X before but now says Y"
#   - Prioritize recent knowledge over stale knowledge

@dataclass
class TemporalEdge:
    """A relationship between two entities, with time bounds."""
    source: str
    relation: str
    target: str
    valid_from: datetime
    valid_until: Optional[datetime] = None  # None means still active
    confidence: float = 1.0
    episode_id: str = ""  # which conversation introduced this

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    def __repr__(self):
        status = "ACTIVE" if self.is_active else f"until {self.valid_until:%Y-%m-%d}"
        return f"({self.source})-[{self.relation}]->({self.target}) [{self.valid_from:%Y-%m-%d} {status}]"


class TemporalKnowledgeGraph:
    """
    A time-aware knowledge graph that tracks when facts were true.

    In production, this would be backed by Neo4j with Cypher queries.
    Here we use Python dicts to demonstrate the concepts without
    any external dependencies.
    """

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}  # entity_id -> attributes
        self.edges: List[TemporalEdge] = []

    def add_entity(self, entity_id: str, entity_type: str = "unknown", **attributes):
        """Add or update an entity with its attributes."""
        if entity_id not in self.entities:
            self.entities[entity_id] = {"type": entity_type, "created_at": datetime.now()}
        self.entities[entity_id].update(attributes)
        self.entities[entity_id]["type"] = entity_type

    def add_edge(self, source: str, relation: str, target: str,
                 valid_from: Optional[datetime] = None, episode_id: str = ""):
        """
        Add a temporal edge.  If a conflicting active edge exists,
        close it (set valid_until) before adding the new one.

        This is ENTITY RESOLUTION in action: we detect that a new fact
        supersedes an old one, rather than blindly duplicating.
        """
        if valid_from is None:
            valid_from = datetime.now()

        # Auto-create entities if they don't exist
        for eid in [source, target]:
            if eid not in self.entities:
                self.add_entity(eid)

        # Close conflicting active edges (same source + relation)
        # Example: if Alice already "manages" team X, and we add
        # "Bob manages team X", we close Alice's edge.
        for edge in self.edges:
            if (edge.source == source and edge.relation == relation
                    and edge.is_active and edge.target != target):
                edge.valid_until = valid_from
            # Also close if same relation + target but different source
            if (edge.relation == relation and edge.target == target
                    and edge.is_active and edge.source != source):
                edge.valid_until = valid_from

        self.edges.append(TemporalEdge(
            source=source, relation=relation, target=target,
            valid_from=valid_from, episode_id=episode_id,
        ))

    def query_at_time(self, point_in_time: datetime) -> List[TemporalEdge]:
        """Get all facts that were active at a specific point in time."""
        results = []
        for edge in self.edges:
            if edge.valid_from <= point_in_time:
                if edge.valid_until is None or edge.valid_until > point_in_time:
                    results.append(edge)
        return results

    def get_active_edges(self) -> List[TemporalEdge]:
        """Get all currently active relationships."""
        return [e for e in self.edges if e.is_active]

    def get_entity_relations(self, entity_id: str, active_only: bool = True) -> List[TemporalEdge]:
        """Get all edges involving an entity (as source or target)."""
        results = []
        for edge in self.edges:
            if active_only and not edge.is_active:
                continue
            if edge.source == entity_id or edge.target == entity_id:
                results.append(edge)
        return results

    def stats(self) -> Dict[str, int]:
        active = sum(1 for e in self.edges if e.is_active)
        return {
            "entities": len(self.entities),
            "total_edges": len(self.edges),
            "active_edges": active,
            "historical_edges": len(self.edges) - active,
        }


def demo_temporal_knowledge_graph():
    """Demonstrate temporal edges and entity resolution."""
    print("\n" + "=" * 70)
    print("  2. TEMPORAL KNOWLEDGE GRAPHS")
    print("=" * 70)

    tkg = TemporalKnowledgeGraph()

    # Simulate knowledge evolving over time
    jan = datetime(2024, 1, 15)
    mar = datetime(2024, 3, 1)
    jun = datetime(2024, 6, 1)

    # Episode 1: January onboarding
    tkg.add_entity("alice", "person", role="ML engineer")
    tkg.add_entity("bob", "person", role="data scientist")
    tkg.add_entity("ml_team", "team", department="engineering")
    tkg.add_edge("alice", "manages", "ml_team", valid_from=jan, episode_id="ep1")
    tkg.add_edge("bob", "member_of", "ml_team", valid_from=jan, episode_id="ep1")
    tkg.add_edge("ml_team", "works_on", "recommendation_engine", valid_from=jan, episode_id="ep1")

    print("\n  After January onboarding:")
    for edge in tkg.get_active_edges():
        print(f"    {edge}")

    # Episode 2: June reorg — Bob now manages the team
    tkg.add_edge("bob", "manages", "ml_team", valid_from=jun, episode_id="ep2")
    tkg.add_entity("alice", "person", role="IC engineer")  # role change
    tkg.add_edge("ml_team", "works_on", "llm_platform", valid_from=jun, episode_id="ep2")

    print("\n  After June reorg:")
    for edge in tkg.get_active_edges():
        print(f"    {edge}")

    # Time-travel query
    print("\n  Who managed the ML team in February?")
    feb_facts = tkg.query_at_time(datetime(2024, 2, 15))
    manager_facts = [e for e in feb_facts if e.relation == "manages" and e.target == "ml_team"]
    for f in manager_facts:
        print(f"    -> {f.source} (from episode {f.episode_id})")

    print("\n  Who manages the ML team NOW (after June)?")
    for e in tkg.get_active_edges():
        if e.relation == "manages" and e.target == "ml_team":
            print(f"    -> {e.source}")

    print(f"\n  Graph stats: {tkg.stats()}")


# ================================================================
# 3. HYBRID VECTOR + GRAPH ARCHITECTURE
# ================================================================
# The most powerful retrieval combines BOTH paradigms:
#
# VECTOR SEARCH finds relevant text chunks (broad recall).
# GRAPH TRAVERSAL finds structured relationships (precise reasoning).
#
# Pipeline:
#   1. User asks: "What projects does Alice's team work on?"
#   2. Vector search returns chunks mentioning "Alice" and "projects".
#   3. Graph traversal: Alice -> manages -> ml_team -> works_on -> [projects]
#   4. Merge results: graph gives precise answer, vector adds context.
#
# This is more token-efficient than stuffing 10 chunks into the prompt.
# The graph gives you the exact relationship chain; vector adds color.

class SimpleVectorStore:
    """
    Minimal vector store using term-frequency similarity.
    In production, use FAISS, ChromaDB, or Pinecone with real embeddings.
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []

    def add(self, text: str, metadata: Optional[Dict] = None):
        words = set(text.lower().split())
        self.documents.append({
            "text": text,
            "words": words,
            "metadata": metadata or {},
        })

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Keyword overlap similarity (stand-in for cosine similarity)."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            overlap = len(query_words & doc["words"])
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append({"text": doc["text"], "score": score, **doc["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class ContextGraph:
    """
    A pure-Python context graph for agent knowledge management.

    Stores entities and relationships as adjacency lists.
    Supports multi-hop traversal and context string generation
    for efficient LLM prompt injection.
    """

    def __init__(self):
        # Adjacency list: entity -> [(relation, target, metadata)]
        self.adjacency: Dict[str, List[Tuple[str, str, Dict]]] = {}
        self.entity_attrs: Dict[str, Dict[str, Any]] = {}

    def add_entity(self, entity_id: str, **attributes):
        """
        Add an entity (node) to the graph.

        Args:
            entity_id: Unique identifier (e.g., "alice", "ml_team").
            **attributes: Any key-value attributes (type, role, etc.).
        """
        if entity_id not in self.adjacency:
            self.adjacency[entity_id] = []
        self.entity_attrs[entity_id] = attributes

    def add_relation(self, source: str, relation: str, target: str, **metadata):
        """
        Add a directed edge (relationship) between two entities.

        Args:
            source: Source entity id.
            relation: Relationship label (e.g., "manages", "works_on").
            target: Target entity id.
            **metadata: Additional edge properties (confidence, timestamp).
        """
        # Auto-create entities
        for eid in [source, target]:
            if eid not in self.adjacency:
                self.add_entity(eid)

        # Avoid exact duplicates
        for r, t, _ in self.adjacency[source]:
            if r == relation and t == target:
                return  # Already exists

        self.adjacency[source].append((relation, target, metadata))

    def traverse(self, start: str, max_depth: int = 2) -> List[Tuple[str, str, str, int]]:
        """
        Breadth-first traversal from a starting entity.

        Returns a list of (source, relation, target, depth) tuples
        representing all reachable facts within max_depth hops.

        WHY BFS: We want the closest relationships first.  An agent
        asking about Alice cares more about her direct team than about
        a project three hops away.
        """
        if start not in self.adjacency:
            return []

        visited: Set[str] = {start}
        queue: deque = deque([(start, 0)])
        results: List[Tuple[str, str, str, int]] = []

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for relation, target, _ in self.adjacency.get(current, []):
                results.append((current, relation, target, depth + 1))
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))

        return results

    def multi_hop_query(self, start: str, relation_path: List[str]) -> List[str]:
        """
        Follow a specific chain of relations from a start entity.

        Example: multi_hop_query("alice", ["manages", "works_on"])
        -> finds what projects Alice's managed team works on.

        This is WHERE GRAPHS BEAT VECTORS: a single traversal gives
        the precise answer, while vector search would need to retrieve
        and parse multiple documents hoping they overlap.
        """
        current_entities = {start}

        for relation in relation_path:
            next_entities: Set[str] = set()
            for entity in current_entities:
                for r, t, _ in self.adjacency.get(entity, []):
                    if r == relation:
                        next_entities.add(t)
            current_entities = next_entities
            if not current_entities:
                return []  # Dead end — no entities match this hop

        return list(current_entities)

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, str]]:
        """Get direct neighbors: [(relation, target), ...]."""
        return [(r, t) for r, t, _ in self.adjacency.get(entity_id, [])]

    def to_context_string(self, entity_id: str, max_depth: int = 2) -> str:
        """
        Generate a compact text representation of an entity's subgraph.

        This is what gets injected into the LLM prompt instead of full
        document chunks.  Much more token-efficient.

        Example output:
          alice (person, role=ML engineer):
            - manages -> ml_team
            - ml_team works_on -> recommendation_engine
        """
        if entity_id not in self.adjacency:
            return f"No information about '{entity_id}'."

        lines = []
        attrs = self.entity_attrs.get(entity_id, {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
        lines.append(f"{entity_id} ({attr_str}):" if attr_str else f"{entity_id}:")

        facts = self.traverse(entity_id, max_depth)
        for source, relation, target, depth in facts:
            indent = "  " * depth
            target_attrs = self.entity_attrs.get(target, {})
            if target_attrs:
                target_info = f"{target} ({', '.join(f'{k}={v}' for k, v in target_attrs.items())})"
            else:
                target_info = target
            lines.append(f"{indent}- {source} --[{relation}]--> {target_info}")

        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        total_edges = sum(len(edges) for edges in self.adjacency.values())
        return {"entities": len(self.adjacency), "edges": total_edges}


def demo_hybrid_architecture():
    """Show vector search + graph traversal working together."""
    print("\n" + "=" * 70)
    print("  3. HYBRID VECTOR + GRAPH ARCHITECTURE")
    print("=" * 70)

    # Build the graph
    graph = ContextGraph()
    graph.add_entity("alice", type="person", role="ML engineer")
    graph.add_entity("bob", type="person", role="data scientist")
    graph.add_entity("ml_team", type="team", department="engineering")
    graph.add_entity("rec_engine", type="project", status="active")
    graph.add_entity("llm_platform", type="project", status="planning")

    graph.add_relation("alice", "manages", "ml_team")
    graph.add_relation("bob", "member_of", "ml_team")
    graph.add_relation("ml_team", "works_on", "rec_engine")
    graph.add_relation("ml_team", "works_on", "llm_platform")
    graph.add_relation("rec_engine", "uses", "collaborative_filtering")
    graph.add_relation("llm_platform", "uses", "transformer_architecture")

    # Build the vector store
    vector_store = SimpleVectorStore()
    vector_store.add("Alice is leading the ML team's efforts on personalization.")
    vector_store.add("The recommendation engine uses collaborative filtering and handles 10M users.")
    vector_store.add("Bob joined the ML team in January 2024 as a data scientist.")
    vector_store.add("The LLM platform project aims to build internal AI tools using transformers.")
    vector_store.add("The engineering department has three teams: ML, backend, and infra.")

    # Query: "What projects does Alice's team work on?"
    query = "What projects does Alice's team work on?"
    print(f"\n  Query: {query}")

    # Vector-only approach
    print("\n  --- Vector Search Results (broad recall) ---")
    vector_results = vector_store.search(query, top_k=3)
    vector_tokens = 0
    for r in vector_results:
        print(f"    [{r['score']:.2f}] {r['text'][:80]}")
        vector_tokens += len(r["text"].split())
    print(f"    Total tokens (approx): ~{vector_tokens}")

    # Graph-only approach
    print("\n  --- Graph Traversal (precise reasoning) ---")
    projects = graph.multi_hop_query("alice", ["manages", "works_on"])
    graph_context = graph.to_context_string("alice", max_depth=2)
    print(f"    Multi-hop result: alice -> manages -> ? -> works_on -> {projects}")
    print(f"    Context string ({len(graph_context.split())} tokens):")
    for line in graph_context.split("\n"):
        print(f"      {line}")

    # Hybrid approach
    print("\n  --- Hybrid: Graph + Vector (best of both) ---")
    print(f"    Graph answer: {projects}")
    print(f"    Vector enrichment: {vector_results[0]['text'][:80]}...")
    hybrid_tokens = len(graph_context.split()) + len(vector_results[0]["text"].split())
    print(f"    Combined tokens: ~{hybrid_tokens} (vs ~{vector_tokens} vector-only)")
    print(f"    Savings: ~{max(0, vector_tokens - hybrid_tokens)} tokens")


# ================================================================
# 4. MULTI-HOP REASONING & TOKEN EFFICIENCY
# ================================================================
# Multi-hop reasoning = answering questions that require traversing
# multiple relationships.  Graphs excel here because each "hop" is
# a direct edge traversal, not a separate retrieval call.
#
# Example: "What technology does Alice's team's active project use?"
#   Hop 1: alice -> manages -> ml_team
#   Hop 2: ml_team -> works_on -> rec_engine
#   Hop 3: rec_engine -> uses -> collaborative_filtering
#
# With vectors, you'd need to retrieve and parse 3-5 separate chunks
# and hope the LLM can piece them together.  With a graph, you get
# the answer in a single traversal.

def demo_multi_hop_reasoning():
    """Show multi-hop queries and token savings vs naive RAG."""
    print("\n" + "=" * 70)
    print("  4. MULTI-HOP REASONING & TOKEN EFFICIENCY")
    print("=" * 70)

    # Build a richer graph
    graph = ContextGraph()

    # Organization structure
    graph.add_entity("ceo", type="person", name="Carol")
    graph.add_entity("vp_eng", type="person", name="Diana")
    graph.add_entity("alice", type="person", role="team lead")
    graph.add_entity("bob", type="person", role="senior engineer")
    graph.add_entity("ml_team", type="team")
    graph.add_entity("rec_engine", type="project")
    graph.add_entity("collab_filter", type="technology")
    graph.add_entity("python", type="language")

    graph.add_relation("ceo", "oversees", "vp_eng")
    graph.add_relation("vp_eng", "oversees", "alice")
    graph.add_relation("alice", "manages", "ml_team")
    graph.add_relation("bob", "member_of", "ml_team")
    graph.add_relation("ml_team", "works_on", "rec_engine")
    graph.add_relation("rec_engine", "uses", "collab_filter")
    graph.add_relation("rec_engine", "implemented_in", "python")

    # Multi-hop queries
    queries = [
        ("Who oversees Alice's boss?",
         "vp_eng", ["oversees"],
         "1-hop: find who oversees Alice's manager"),
        ("What does Alice's team work on?",
         "alice", ["manages", "works_on"],
         "2-hop: alice -> manages -> team -> works_on -> projects"),
        ("What technology does Alice's team's project use?",
         "alice", ["manages", "works_on", "uses"],
         "3-hop: alice -> team -> project -> uses -> technology"),
    ]

    for question, start, path, explanation in queries:
        result = graph.multi_hop_query(start, path)
        context = graph.to_context_string(start, max_depth=len(path))
        tokens = len(context.split())

        print(f"\n  Q: {question}")
        print(f"  Strategy: {explanation}")
        print(f"  Answer: {result}")
        print(f"  Tokens used: ~{tokens}")

    # Compare token costs
    print(textwrap.dedent("""
  ┌──────────────────────┬────────────┬────────────────────────────────┐
  │ Approach             │ Tokens     │ Notes                          │
  ├──────────────────────┼────────────┼────────────────────────────────┤
  │ Naive RAG (5 chunks) │ ~500-2000  │ Retrieve 5 docs, hope for best│
  │ Graph traversal      │ ~50-150    │ Precise path, minimal tokens   │
  │ Hybrid (graph + 1    │ ~150-400   │ Graph for answer, 1 chunk for  │
  │   enrichment chunk)  │            │ supporting detail              │
  └──────────────────────┴────────────┴────────────────────────────────┘

  TOKEN SAVINGS MATTER because:
    - Lower cost per query ($0.002 vs $0.01 for GPT-4)
    - Faster response (less input to process)
    - More room for actual reasoning in the context window
    - Fewer irrelevant chunks = fewer hallucinations
    """))


# ================================================================
# 5. WHEN TO USE CONTEXT GRAPHS (Decision Framework)
# ================================================================
# Not every agent needs a knowledge graph.  Use this framework:
#
# USE CONTEXT GRAPHS WHEN:
#   [x] Your data has rich relationships (org charts, supply chains,
#       codebases, medical records, research papers).
#   [x] Users ask multi-hop questions ("What does X's team's project use?").
#   [x] Knowledge changes over time and you need temporal awareness.
#   [x] Token budget is tight and you need surgical context injection.
#   [x] You need explainable retrieval ("I found this via: A -> B -> C").
#
# USE VECTOR-ONLY WHEN:
#   [x] Your data is mostly flat text (FAQs, docs, manuals).
#   [x] Questions are single-hop ("Find me info about X").
#   [x] Relationships between items don't matter for answers.
#   [x] You want the simplest possible implementation.
#
# USE BOTH (HYBRID) WHEN:
#   [x] You have structured data AND rich text descriptions.
#   [x] Some queries need precision (graph) and some need recall (vector).
#   [x] You can afford the extra complexity for better results.

def demo_decision_framework():
    """Interactive decision framework for choosing a retrieval strategy."""
    print("\n" + "=" * 70)
    print("  5. WHEN TO USE CONTEXT GRAPHS — Decision Framework")
    print("=" * 70)

    scenarios = [
        {
            "name": "Customer FAQ Bot",
            "has_relationships": False,
            "multi_hop": False,
            "temporal": False,
            "token_sensitive": False,
            "recommendation": "VECTOR-ONLY",
            "reason": "Flat Q&A pairs, single-hop lookups, no relationships needed.",
        },
        {
            "name": "Org Knowledge Assistant",
            "has_relationships": True,
            "multi_hop": True,
            "temporal": True,
            "token_sensitive": True,
            "recommendation": "CONTEXT GRAPH",
            "reason": "Rich org relationships, temporal changes, multi-hop queries about reporting chains.",
        },
        {
            "name": "Code Documentation Agent",
            "has_relationships": True,
            "multi_hop": True,
            "temporal": False,
            "token_sensitive": True,
            "recommendation": "HYBRID",
            "reason": "Functions call other functions (graph), but docstrings need full text (vector).",
        },
        {
            "name": "Research Paper Finder",
            "has_relationships": True,
            "multi_hop": False,
            "temporal": False,
            "token_sensitive": False,
            "recommendation": "VECTOR-ONLY or HYBRID",
            "reason": "Citation graphs exist but most queries are 'find papers about X' (single-hop).",
        },
        {
            "name": "Medical History Agent",
            "has_relationships": True,
            "multi_hop": True,
            "temporal": True,
            "token_sensitive": True,
            "recommendation": "CONTEXT GRAPH",
            "reason": "Patient history evolves over time, drug interactions are multi-hop, accuracy is critical.",
        },
    ]

    print(textwrap.dedent("""
  Decision checklist — score each dimension:
    1. Does your data have rich relationships?     (Y/N)
    2. Do users ask multi-hop questions?           (Y/N)
    3. Does knowledge change over time?            (Y/N)
    4. Is token efficiency critical?               (Y/N)

  If 0-1 "Yes": Vector-only
  If 2-3 "Yes": Hybrid (vector + graph)
  If 3-4 "Yes": Context graph (with optional vector enrichment)
    """))

    for s in scenarios:
        score = sum([s["has_relationships"], s["multi_hop"], s["temporal"], s["token_sensitive"]])
        checks = (
            f"Relationships={'Y' if s['has_relationships'] else 'N'}, "
            f"MultiHop={'Y' if s['multi_hop'] else 'N'}, "
            f"Temporal={'Y' if s['temporal'] else 'N'}, "
            f"TokenSensitive={'Y' if s['token_sensitive'] else 'N'}"
        )
        print(f"  Scenario: {s['name']}")
        print(f"    {checks}  (score: {score}/4)")
        print(f"    -> {s['recommendation']}: {s['reason']}")
        print()


# ================================================================
# PUTTING IT ALL TOGETHER — Full ContextGraph Demo
# ================================================================

def demo_full_context_graph():
    """End-to-end demo of the ContextGraph class."""
    print("\n" + "=" * 70)
    print("  FULL CONTEXT GRAPH DEMO")
    print("=" * 70)

    g = ContextGraph()

    # Build a knowledge graph about a software company
    g.add_entity("acme_corp", type="company", industry="tech")
    g.add_entity("alice", type="person", role="CTO")
    g.add_entity("bob", type="person", role="team lead")
    g.add_entity("carol", type="person", role="engineer")
    g.add_entity("ml_team", type="team", size=5)
    g.add_entity("search_service", type="project", status="production")
    g.add_entity("chatbot", type="project", status="development")
    g.add_entity("pytorch", type="technology")
    g.add_entity("fastapi", type="technology")

    g.add_relation("alice", "works_at", "acme_corp")
    g.add_relation("alice", "oversees", "bob")
    g.add_relation("bob", "manages", "ml_team")
    g.add_relation("carol", "member_of", "ml_team")
    g.add_relation("ml_team", "owns", "search_service")
    g.add_relation("ml_team", "owns", "chatbot")
    g.add_relation("search_service", "built_with", "pytorch")
    g.add_relation("search_service", "built_with", "fastapi")
    g.add_relation("chatbot", "built_with", "pytorch")

    print(f"\n  Graph stats: {g.stats()}")

    # Context string for agent prompt
    print("\n  Context string for 'bob' (2 hops):")
    ctx = g.to_context_string("bob", max_depth=2)
    for line in ctx.split("\n"):
        print(f"    {line}")

    # Multi-hop queries
    print("\n  Multi-hop queries:")
    q1 = g.multi_hop_query("alice", ["oversees", "manages"])
    print(f"    alice -> oversees -> ? -> manages -> ? = {q1}")

    q2 = g.multi_hop_query("bob", ["manages", "owns"])
    print(f"    bob -> manages -> ? -> owns -> ? = {q2}")

    q3 = g.multi_hop_query("bob", ["manages", "owns", "built_with"])
    print(f"    bob -> manages -> ? -> owns -> ? -> built_with -> ? = {q3}")

    # Direct neighbors
    print(f"\n  Bob's direct neighbors: {g.get_neighbors('bob')}")

    # Full traversal
    print("\n  Full traversal from 'alice' (depth 3):")
    for source, rel, target, depth in g.traverse("alice", max_depth=3):
        indent = "    " + "  " * depth
        print(f"{indent}{source} --[{rel}]--> {target}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 19: Context Graphs (Concepts)              ║")
    print("╚" + "═" * 63 + "╝")

    demo_paradigm_comparison()
    demo_temporal_knowledge_graph()
    demo_hybrid_architecture()
    demo_multi_hop_reasoning()
    demo_decision_framework()
    demo_full_context_graph()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Context graphs represent knowledge as entities + relationships,
       enabling multi-hop reasoning that vector stores cannot do alone.

    2. Temporal edges track WHEN facts were true.  This prevents stale
       knowledge from overriding current facts — critical for agents
       that operate over weeks or months.

    3. Hybrid (vector + graph) gives the best results: graphs for
       precise relational queries, vectors for broad text retrieval.

    4. Multi-hop graph traversal uses 5-10x FEWER tokens than naive
       RAG for relational questions, saving cost and reducing noise.

    5. Use the decision framework: if your data has rich relationships,
       temporal changes, or multi-hop queries, invest in a context graph.
       Otherwise, vector-only is simpler and sufficient.
    """))
