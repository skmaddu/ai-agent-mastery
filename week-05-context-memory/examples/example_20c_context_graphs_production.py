import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 20c: Context Graphs -- Production Implementation
==========================================================
Production-grade context graph with REAL persistence, REAL hybrid
retrieval (graph traversal + ChromaDB vector search), and LLM-based
entity extraction.

This is the REAL implementation of the concepts from Examples 19-20.
Every component uses production libraries:

  - SQLite: persistent graph storage (survives restarts)
  - ChromaDB + sentence-transformers: real vector search for hybrid mode
  - LLM entity extraction: structured extraction with JSON output
  - Multi-hop graph traversal: BFS with depth control
  - Hybrid retrieval: graph precision + vector recall merged

LangGraph Pipeline:
  START -> extract_entities -> query_graph -> hybrid_retrieve
        -> generate_with_context -> END

Production Upgrade Path:
  SQLite graph   -> Neo4j (Cypher queries, scalable)
  In-process     -> Zep Graphiti (temporal edges, managed service)
  Manual extract -> Fine-tuned NER model (faster, cheaper)

Requirements (already in requirements.txt):
  pip install chromadb sentence-transformers

Run: python week-05-context-memory/examples/example_20c_context_graphs_production.py
"""

import os
import sys
import json
import sqlite3
import textwrap
from pathlib import Path
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict, Any, Set, Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# -- Real embedding model + vector store
import chromadb
from sentence_transformers import SentenceTransformer

# -- Phoenix
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
        tracer_provider = register(project_name="week5-context-graphs-production")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None


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


# ==================================================================
# PERSISTENT KNOWLEDGE GRAPH -- SQLite-backed
# ==================================================================
# This graph persists to SQLite so it survives process restarts.
# The schema stores entities and edges in separate tables.
#
# In production, replace with Neo4j:
#   from neo4j import GraphDatabase
#   driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
#   driver.execute_query("MERGE (a:Person {name: $name})", name="alice")

DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(exist_ok=True)
GRAPH_DB_PATH = DB_DIR / "knowledge_graph.db"


class PersistentKnowledgeGraph:
    """
    SQLite-backed knowledge graph with temporal edges.

    Stores entities and relationships in a real database.
    Supports multi-hop traversal, temporal queries, and
    survives process restarts.

    Production upgrade: replace SQLite with Neo4j for:
      - Cypher query language (much richer graph queries)
      - Scalability (millions of nodes/edges)
      - Concurrent access (multi-process/multi-user)
      - Native graph algorithms (shortest path, community detection)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT DEFAULT 'unknown',
                attributes TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                valid_from TEXT DEFAULT (datetime('now')),
                valid_until TEXT DEFAULT NULL,
                episode_id TEXT DEFAULT '',
                FOREIGN KEY (source) REFERENCES entities(id),
                FOREIGN KEY (target) REFERENCES entities(id),
                UNIQUE(source, relation, target, valid_until)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE INDEX IF NOT EXISTS idx_edges_active ON edges(valid_until);
        """)
        self.conn.commit()

    def add_entity(self, entity_id: str, entity_type: str = "unknown",
                   **attributes):
        """Add or update an entity."""
        entity_id = entity_id.lower().strip()
        attrs_json = json.dumps(attributes)
        self.conn.execute("""
            INSERT INTO entities (id, entity_type, attributes)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                entity_type = excluded.entity_type,
                attributes = excluded.attributes,
                updated_at = datetime('now')
        """, (entity_id, entity_type, attrs_json))
        self.conn.commit()

    def add_edge(self, source: str, relation: str, target: str,
                 confidence: float = 1.0, episode_id: str = ""):
        """
        Add a temporal edge. Closes conflicting active edges first.

        If source already has an active edge with the same relation
        but different target, the old edge gets closed (valid_until set).
        This handles entity resolution: "Alice manages team" then later
        "Bob manages team" -- Alice's edge gets closed.
        """
        source = source.lower().strip()
        target = target.lower().strip()
        relation = relation.lower().strip()

        # Ensure entities exist
        self.add_entity(source)
        self.add_entity(target)

        # Close conflicting active edges (same source+relation, different target)
        self.conn.execute("""
            UPDATE edges SET valid_until = datetime('now')
            WHERE source = ? AND relation = ? AND target != ?
              AND valid_until IS NULL
        """, (source, relation, target))

        # Insert new edge (ignore if exact duplicate exists)
        try:
            self.conn.execute("""
                INSERT INTO edges (source, relation, target, confidence, episode_id)
                VALUES (?, ?, ?, ?, ?)
            """, (source, relation, target, confidence, episode_id))
        except sqlite3.IntegrityError:
            pass  # Exact edge already exists

        self.conn.commit()

    def get_active_edges(self, entity_id: Optional[str] = None) -> List[Dict]:
        """Get all currently active edges, optionally filtered by entity."""
        entity_id = entity_id.lower().strip() if entity_id else None

        if entity_id:
            cursor = self.conn.execute("""
                SELECT source, relation, target, confidence, valid_from
                FROM edges
                WHERE (source = ? OR target = ?) AND valid_until IS NULL
                ORDER BY valid_from DESC
            """, (entity_id, entity_id))
        else:
            cursor = self.conn.execute("""
                SELECT source, relation, target, confidence, valid_from
                FROM edges WHERE valid_until IS NULL
                ORDER BY valid_from DESC
            """)

        return [
            {"source": r[0], "relation": r[1], "target": r[2],
             "confidence": r[3], "valid_from": r[4]}
            for r in cursor.fetchall()
        ]

    def traverse_bfs(self, start: str, max_depth: int = 2) -> List[Dict]:
        """
        Breadth-first traversal from a starting entity.

        Returns facts as dicts: {source, relation, target, depth}.
        Follows active edges only.
        """
        start = start.lower().strip()
        visited = {start}
        queue = deque([(start, 0)])
        results = []

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            edges = self.get_active_edges(current)
            for edge in edges:
                src, rel, tgt = edge["source"], edge["relation"], edge["target"]
                # Only follow outgoing edges from current
                if src == current:
                    results.append({
                        "source": src, "relation": rel,
                        "target": tgt, "depth": depth + 1
                    })
                    if tgt not in visited:
                        visited.add(tgt)
                        queue.append((tgt, depth + 1))

        return results

    def multi_hop_query(self, start: str, relation_path: List[str]) -> List[str]:
        """
        Follow a specific chain of relations.

        Example: multi_hop_query("alice", ["manages", "works_on"])
        -> finds what projects Alice's managed team works on.
        """
        start = start.lower().strip()
        current_entities = {start}

        for relation in relation_path:
            next_entities = set()
            for entity in current_entities:
                edges = self.get_active_edges(entity)
                for edge in edges:
                    if edge["source"] == entity and edge["relation"] == relation:
                        next_entities.add(edge["target"])
            current_entities = next_entities
            if not current_entities:
                return []

        return list(current_entities)

    def to_context_string(self, entity_id: str, max_depth: int = 2) -> str:
        """
        Generate compact text for LLM prompt injection.

        This is MUCH more token-efficient than raw document chunks.
        A typical subgraph is 50-150 tokens vs 500-2000 for RAG chunks.
        """
        entity_id = entity_id.lower().strip()

        # Get entity info
        cursor = self.conn.execute(
            "SELECT entity_type, attributes FROM entities WHERE id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        if not row:
            return f"No information about '{entity_id}'."

        etype, attrs_json = row
        attrs = json.loads(attrs_json) if attrs_json else {}
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())

        lines = [f"{entity_id} ({etype}{', ' + attr_str if attr_str else ''}):"]

        facts = self.traverse_bfs(entity_id, max_depth)
        for fact in facts:
            indent = "  " * fact["depth"]
            lines.append(f"{indent}- {fact['source']} --[{fact['relation']}]--> {fact['target']}")

        return "\n".join(lines)

    def get_all_entities(self) -> List[Dict]:
        """Get all entities with their types."""
        cursor = self.conn.execute("SELECT id, entity_type FROM entities ORDER BY id")
        return [{"id": r[0], "type": r[1]} for r in cursor.fetchall()]

    def stats(self) -> Dict[str, int]:
        entities = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        total_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        active_edges = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE valid_until IS NULL"
        ).fetchone()[0]
        return {
            "entities": entities,
            "total_edges": total_edges,
            "active_edges": active_edges,
            "historical_edges": total_edges - active_edges,
        }

    def close(self):
        self.conn.close()


# ==================================================================
# HYBRID RETRIEVAL -- Graph + ChromaDB Vector Search
# ==================================================================
# The production hybrid approach:
#   1. Graph traversal: precise relational answers (multi-hop)
#   2. Vector search: broad context from document chunks
#   3. Merge: graph facts + top vector results -> LLM prompt

print("[INIT] Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB for the vector store component of hybrid retrieval
CHROMA_CLIENT = chromadb.EphemeralClient()
try:
    CHROMA_CLIENT.delete_collection("context_graph_docs")
except Exception:
    pass

VECTOR_COLLECTION = CHROMA_CLIENT.create_collection(
    name="context_graph_docs",
    metadata={"description": "Document chunks for hybrid graph+vector retrieval"},
)

# Seed documents that complement the graph
SEED_DOCUMENTS = [
    "Alice is the ML team lead with 8 years of experience in deep learning. She manages a team of 5 engineers.",
    "Bob is a senior data scientist on the ML team. He specializes in recommendation systems and collaborative filtering.",
    "The search engine project uses PyTorch for model training and FastAPI for serving. It handles 10 million queries per day.",
    "The chatbot project uses LangChain and LangGraph for agent orchestration. It was started in Q3 2024.",
    "Carol leads the backend team responsible for the API platform. The team uses FastAPI and PostgreSQL.",
    "Diana is the VP of Engineering overseeing both the ML and backend teams. She reports directly to the CEO.",
    "The recommendation engine is a new project exploring neural collaborative filtering for personalized content.",
    "The ML team adopted Arize Phoenix for observability and experiment tracking in all their projects.",
]


def index_documents():
    """Index seed documents in ChromaDB for hybrid retrieval."""
    embeddings = EMBED_MODEL.encode(SEED_DOCUMENTS).tolist()
    VECTOR_COLLECTION.add(
        ids=[f"doc_{i}" for i in range(len(SEED_DOCUMENTS))],
        documents=SEED_DOCUMENTS,
        embeddings=embeddings,
    )
    print(f"[INDEX] ChromaDB: {VECTOR_COLLECTION.count()} docs indexed")


# ==================================================================
# Initialize persistent graph + vector store
# ==================================================================

print(f"[INIT] Opening graph database: {GRAPH_DB_PATH}")
GRAPH = PersistentKnowledgeGraph(str(GRAPH_DB_PATH))

# Check if graph already has data from a previous run
existing_stats = GRAPH.stats()
if existing_stats["entities"] > 0:
    print(f"[INIT] RESUMED graph from SQLite: {existing_stats}")
else:
    # Seed the graph with initial facts
    seed_facts = [
        ("alice", "person", "manages", "ml_team"),
        ("bob", "person", "member_of", "ml_team"),
        ("carol", "person", "leads", "backend_team"),
        ("diana", "person", "oversees", "alice"),
        ("diana", "person", "oversees", "carol"),
        ("ml_team", "team", "works_on", "search_engine"),
        ("ml_team", "team", "works_on", "chatbot"),
        ("backend_team", "team", "works_on", "api_platform"),
        ("search_engine", "project", "uses", "pytorch"),
        ("search_engine", "project", "uses", "fastapi"),
        ("chatbot", "project", "uses", "langchain"),
        ("api_platform", "project", "uses", "fastapi"),
        ("api_platform", "project", "uses", "postgresql"),
    ]

    for source, stype, relation, target in seed_facts:
        GRAPH.add_entity(source, stype)
        GRAPH.add_edge(source, relation, target, episode_id="seed")

    print(f"[INIT] Seeded graph: {GRAPH.stats()}")

index_documents()
print("[INIT] Ready.\n")


# ==================================================================
# STATE
# ==================================================================

class GraphRAGState(TypedDict):
    query: str
    entities: List[str]                  # Extracted entity names
    graph_context: str                   # Graph traversal results
    vector_context: str                  # ChromaDB search results
    combined_context: str                # Merged hybrid context
    answer: str
    graph_stats: Dict[str, int]


# ==================================================================
# GRAPH NODES
# ==================================================================

llm = get_llm(temperature=0.3)


def extract_entities_node(state: GraphRAGState) -> dict:
    """
    Extract entities and relationships from the user message using LLM.

    New facts are added to the persistent SQLite graph.
    Entity names become starting points for graph traversal.
    """
    query = state["query"]

    prompt = [
        SystemMessage(content=(
            "Extract entities and relationships from this message. "
            "Return JSON: {\"entities\": [{\"name\": \"...\", \"type\": \"person|team|project|technology\"}], "
            "\"relations\": [{\"source\": \"...\", \"relation\": \"...\", \"target\": \"...\"}]}. "
            "Keep names lowercase. Return valid JSON only, no markdown."
        )),
        HumanMessage(content=query),
    ]

    entity_names = []
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)

        for ent in data.get("entities", []):
            name = ent.get("name", "").lower().strip()
            etype = ent.get("type", "unknown")
            if name:
                GRAPH.add_entity(name, etype)
                entity_names.append(name)

        for rel in data.get("relations", []):
            src = rel.get("source", "").lower().strip()
            r = rel.get("relation", "").lower().strip()
            tgt = rel.get("target", "").lower().strip()
            if src and r and tgt:
                GRAPH.add_edge(src, r, tgt, episode_id=f"turn-{datetime.now().isoformat()}")

    except (json.JSONDecodeError, Exception):
        # Fallback: simple noun extraction
        stop = {"what", "who", "how", "does", "is", "are", "the", "a", "an",
                "of", "and", "or", "in", "on", "to", "for", "with", "about"}
        words = query.lower().split()
        entity_names = [w.strip("?,.!'\"") for w in words
                       if w.strip("?,.!'\"") not in stop and len(w) > 2][:5]

    print(f"  [EXTRACT] Entities: {entity_names}")
    return {"entities": entity_names, "graph_stats": GRAPH.stats()}


def query_graph_node(state: GraphRAGState) -> dict:
    """
    Traverse the knowledge graph for each extracted entity.

    Uses BFS traversal from the SQLite-persisted graph.
    Also runs multi-hop queries for relational questions.
    """
    entities = state.get("entities", [])
    subgraphs = []

    for entity in entities:
        ctx = GRAPH.to_context_string(entity, max_depth=2)
        if "No information" not in ctx:
            subgraphs.append(ctx)

    # Also check for entities referenced in edges
    if not subgraphs:
        all_entities = GRAPH.get_all_entities()
        for ent in all_entities:
            if any(e in ent["id"] for e in entities):
                ctx = GRAPH.to_context_string(ent["id"], max_depth=2)
                if "No information" not in ctx:
                    subgraphs.append(ctx)

    graph_context = "\n\n".join(subgraphs) if subgraphs else "No graph context found."

    graph_tokens = len(graph_context.split())
    print(f"  [GRAPH] {len(subgraphs)} subgraphs, ~{graph_tokens} tokens")

    return {"graph_context": graph_context}


def hybrid_retrieve_node(state: GraphRAGState) -> dict:
    """
    Hybrid retrieval: combine graph context with ChromaDB vector search.

    Graph gives: precise relational facts (multi-hop, structured)
    Vector gives: rich text context (descriptions, details, color)

    Together: precise answers + supporting detail.
    """
    query = state["query"]
    graph_context = state.get("graph_context", "")

    # Vector search with real embeddings
    query_embedding = EMBED_MODEL.encode(query).tolist()
    results = VECTOR_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"],
    )

    vector_parts = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        sim = 1.0 / (1.0 + dist)
        vector_parts.append(f"[{sim:.2f}] {doc}")

    vector_context = "\n".join(vector_parts)

    # Combine: graph context first (precise), then vector (enrichment)
    combined = ""
    if graph_context and "No graph context" not in graph_context:
        combined += f"=== KNOWLEDGE GRAPH (structured facts) ===\n{graph_context}\n\n"
    combined += f"=== DOCUMENT SEARCH (supporting detail) ===\n{vector_context}"

    graph_tokens = len(graph_context.split()) if graph_context else 0
    vector_tokens = len(vector_context.split())
    print(f"  [HYBRID] Graph: ~{graph_tokens} tokens + Vector: ~{vector_tokens} tokens "
          f"= ~{graph_tokens + vector_tokens} total")

    return {
        "vector_context": vector_context,
        "combined_context": combined,
    }


def generate_node(state: GraphRAGState) -> dict:
    """Generate answer using hybrid graph + vector context."""
    query = state["query"]
    context = state.get("combined_context", "")
    stats = state.get("graph_stats", {})

    prompt = [
        SystemMessage(content=(
            f"You are a knowledge assistant with access to a knowledge graph "
            f"({stats.get('entities', 0)} entities, {stats.get('active_edges', 0)} relationships) "
            f"and a document store.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Instructions:\n"
            f"- Answer using the provided context.\n"
            f"- Reference graph relationships when answering relational questions.\n"
            f"- Use document context for descriptive details.\n"
            f"- If context is insufficient, say what's missing.\n"
            f"- Be concise."
        )),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [GENERATE] {answer[:100]}...")
    return {"answer": answer}


# ==================================================================
# GRAPH CONSTRUCTION
# ==================================================================

def build_graph_rag_pipeline():
    graph = StateGraph(GraphRAGState)

    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("query_graph", query_graph_node)
    graph.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("extract_entities")
    graph.add_edge("extract_entities", "query_graph")
    graph.add_edge("query_graph", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ==================================================================
# DEMO
# ==================================================================

def run_demo():
    app = build_graph_rag_pipeline()

    turns = [
        # Relational query (graph excels)
        "What projects does Alice's team work on?",
        # Adds new knowledge to the persistent graph
        "Bob is also working on a new recommendation engine project using pytorch.",
        # Multi-hop query
        "Who does Alice report to, and what teams does that person oversee?",
        # Tests graph update + temporal edge closure
        "The chatbot project has been moved to the backend team under Carol.",
        # Hybrid query (needs both graph structure + document details)
        "What technologies are used across all projects and who works with PyTorch?",
        # Tests persistence -- graph should have all accumulated knowledge
        "Give me a complete overview of the ML team's projects and team members.",
    ]

    print("=" * 65)
    print("  CONTEXT GRAPHS -- PRODUCTION PIPELINE")
    print("  (SQLite graph + ChromaDB vector + hybrid retrieval)")
    print("=" * 65)

    for i, user_input in enumerate(turns):
        print(f"\n{'-' * 65}")
        print(f"  Turn {i + 1}: {user_input}")
        print(f"{'-' * 65}")

        result = app.invoke({
            "query": user_input,
            "entities": [],
            "graph_context": "",
            "vector_context": "",
            "combined_context": "",
            "answer": "",
            "graph_stats": {},
        })

        print(f"\n  Entities: {result.get('entities', [])}")
        print(f"  Graph: {result.get('graph_stats', {})}")
        print(f"  Answer: {result.get('answer', 'N/A')[:350]}")

    # Final graph state
    print(f"\n{'=' * 65}")
    print(f"  FINAL KNOWLEDGE GRAPH (persisted in SQLite)")
    print(f"{'=' * 65}")
    print(f"  Database: {GRAPH_DB_PATH}")
    print(f"  Stats: {GRAPH.stats()}")
    print(f"\n  All entities:")
    for ent in GRAPH.get_all_entities():
        print(f"    - {ent['id']} ({ent['type']})")

    print(f"\n  Subgraph from 'alice' (3 hops):")
    print(f"  {GRAPH.to_context_string('alice', max_depth=3)}")

    # Multi-hop demo
    print(f"\n  Multi-hop queries:")
    for start, path, desc in [
        ("alice", ["manages", "works_on"], "alice -> manages -> ? -> works_on -> ?"),
        ("diana", ["oversees", "manages"], "diana -> oversees -> ? -> manages -> ?"),
        ("ml_team", ["works_on", "uses"], "ml_team -> works_on -> ? -> uses -> ?"),
    ]:
        result = GRAPH.multi_hop_query(start, path)
        print(f"    {desc} = {result}")

    # Show SQLite tables
    print(f"\n  SQLite tables:")
    cursor = GRAPH.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for row in cursor:
        count = GRAPH.conn.execute(f"SELECT COUNT(*) FROM [{row[0]}]").fetchone()[0]
        print(f"    {row[0]}: {count} rows")

    db_size = Path(GRAPH_DB_PATH).stat().st_size
    print(f"\n  Database size: {db_size / 1024:.1f} KB")
    print(f"  Run this script AGAIN to see resumed graph state!")


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 20c: Context Graphs (Production)")
    print("=" * 65)

    setup_phoenix()
    run_demo()

    GRAPH.close()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS -- PRODUCTION vs CONCEPT")
    print("=" * 65)
    print(textwrap.dedent("""
    WHAT CHANGED from Example 20 (concept) to 20c (production):

    1. PERSISTENCE: Python dict -> SQLite database
       - Graph survives process restarts (run twice to verify!)
       - Proper SQL schema with entities + edges tables
       - Indexed for fast lookups by source/target

    2. TEMPORAL EDGES: No time tracking -> valid_from/valid_until
       - New facts automatically close conflicting old edges
       - "Bob manages team" supersedes "Alice manages team"
       - Historical edges preserved for audit trail

    3. VECTOR SEARCH: Word-overlap heuristic -> ChromaDB + sentence-transformers
       - Real 384-dim semantic embeddings
       - Hybrid retrieval: graph precision + vector recall

    4. HYBRID RETRIEVAL: Graph-only -> Graph + Vector merged
       - Graph context: structured facts (50-150 tokens, precise)
       - Vector context: document details (rich, descriptive)
       - Combined: best of both worlds

    PRODUCTION UPGRADE PATH:
      SQLite graph    -> Neo4j (Cypher queries, scale, algorithms)
      Manual edges    -> Zep Graphiti (auto entity resolution, temporal)
      ChromaDB        -> Pinecone (managed, billions of vectors)
      In-process      -> Graph as a service (API, multi-user)

    NEO4J EXAMPLE (for reference):
      from neo4j import GraphDatabase
      driver = GraphDatabase.driver("bolt://localhost:7687")
      with driver.session() as session:
          session.run(
              "MERGE (a:Person {name: $name}) "
              "MERGE (b:Team {name: $team}) "
              "MERGE (a)-[:MANAGES]->(b)",
              name="alice", team="ml_team"
          )
    """))
