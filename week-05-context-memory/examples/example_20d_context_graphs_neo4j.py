import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Suppress noisy async cleanup warnings from Graphiti's internal httpx/neo4j drivers.
# These are harmless "Event loop is closed" messages during driver teardown.
import warnings
import logging
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("graphiti_core").setLevel(logging.CRITICAL)

"""
Example 20d: Context Graphs -- Production with Neo4j & Graphiti
================================================================
Full production-grade context graph implementation using:

  1. Neo4j -- Real graph database with Cypher queries
  2. Zep Graphiti -- Automatic entity extraction, temporal edges,
     entity resolution, and semantic search over the graph
  3. ChromaDB -- Vector store for hybrid graph+vector retrieval
  4. FastAPI -- Graph-as-a-service REST API wrapper

This is the PRODUCTION UPGRADE of Example 20c (SQLite).

PREREQUISITES:
  Neo4j must be running locally or in the cloud:
    Option A: Docker (recommended)
      docker run -d --name neo4j -p 7687:7687 -p 7474:7474 \
        -e NEO4J_AUTH=neo4j/password neo4j:latest

    Option B: Neo4j Desktop (download from https://neo4j.com/download/)

    Option C: Neo4j Aura (free cloud tier at https://neo4j.com/aura/)

  Set environment variables in config/.env:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password

  If Neo4j is NOT available, this example falls back to SQLite
  (same as Example 20c) so it always runs.

Requirements (already in requirements.txt):
  pip install neo4j graphiti-core chromadb sentence-transformers fastapi uvicorn

Run: python week-05-context-memory/examples/example_20d_context_graphs_neo4j.py
"""

import os
import sys
import json
import sqlite3
import asyncio
import textwrap
from pathlib import Path
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

import chromadb
from sentence_transformers import SentenceTransformer

# ==================================================================
# NEO4J CONNECTION
# ==================================================================
# Try to connect to Neo4j. If unavailable, fall back to SQLite.

# Suppress "Task exception was never retrieved" tracebacks from Graphiti's
# async driver cleanup. These happen when event loops close before httpx
# connections finish tearing down -- completely harmless but very noisy.
def _silence_async_cleanup(loop, context):
    msg = context.get("message", "")
    if "Task exception was never retrieved" in msg:
        return  # Swallow silently
    loop.default_exception_handler(context)


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

NEO4J_AVAILABLE = False
neo4j_driver = None

try:
    from neo4j import GraphDatabase
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Test the connection
    neo4j_driver.verify_connectivity()
    NEO4J_AVAILABLE = True
    print(f"[OK] Neo4j connected: {NEO4J_URI}")
except Exception as e:
    print(f"[FALLBACK] Neo4j not available ({e})")
    print(f"[FALLBACK] Using SQLite graph (same as Example 20c)")
    print(f"[FALLBACK] To use Neo4j: docker run -d -p 7687:7687 -p 7474:7474 "
          f"-e NEO4J_AUTH=neo4j/password neo4j:latest\n")

# ==================================================================
# GRAPHITI (Zep Temporal Knowledge Graph)
# ==================================================================
# Graphiti automatically:
#   - Extracts entities and relationships from text (LLM-powered)
#   - Resolves entity duplicates ("Alice" = "alice" = "Alice Chen")
#   - Manages temporal edges (new facts supersede old ones)
#   - Enables semantic search over the graph
#
# It requires Neo4j as its backend.

GRAPHITI_AVAILABLE = False
graphiti_client = None

if NEO4J_AVAILABLE:
    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client import OpenAIClient, LLMConfig
        from graphiti_core.search.search_config import (
            SearchConfig, NodeSearchConfig, EdgeSearchConfig,
            NodeSearchMethod, EdgeSearchMethod, SearchResults,
        )

        # Graphiti's internal entity extraction uses json_schema response format,
        # which requires OpenAI models (Groq doesn't support it).
        # So Graphiti ALWAYS uses OpenAI, regardless of LLM_PROVIDER setting.
        llm_config = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model="gpt-4o-mini",
            small_model="gpt-4o-mini",
        )

        graphiti_llm = OpenAIClient(llm_config)

        # Graphiti needs an embedder (defaults to OpenAI embeddings).
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key or openai_key.startswith("sk-your"):
            print("[NOTE] Graphiti requires OPENAI_API_KEY for its embedder.")
            print("[NOTE] Set OPENAI_API_KEY in config/.env for full Graphiti support.")
            print("[NOTE] Using direct Neo4j driver instead (Graphiti disabled).")
        else:
            # Store config for lazy creation in the thread pool.
            # Graphiti uses async Neo4j drivers that are bound to an event loop,
            # so we create fresh instances per-thread to avoid loop conflicts.
            _graphiti_config = {
                "uri": NEO4J_URI,
                "user": NEO4J_USER,
                "password": NEO4J_PASSWORD,
                "llm_config": llm_config,
            }

            # Build schema indexes (one-time setup, idempotent)
            async def _init_schema():
                g = Graphiti(
                    uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD,
                    llm_client=OpenAIClient(llm_config),
                )
                await g.build_indices_and_constraints()
                await g.close()

            _init_loop = asyncio.new_event_loop()
            _init_loop.set_exception_handler(_silence_async_cleanup)
            _init_loop.run_until_complete(_init_schema())
            _init_loop.close()
            GRAPHITI_AVAILABLE = True
            print("[OK] Graphiti schema created + ready for ingestion/search")

    except Exception as e:
        print(f"[NOTE] Graphiti not configured: {e}")
        print(f"[NOTE] Using direct Neo4j driver instead")

try:
    _graphiti_config
except NameError:
    _graphiti_config = {}


# ==================================================================
# GRAPH BACKEND -- Neo4j (primary) or SQLite (fallback)
# ==================================================================
# This abstraction lets the example run with or without Neo4j.
# The same LangGraph pipeline works with either backend.

class GraphBackend:
    """Abstract interface for the knowledge graph backend."""

    def add_entity(self, entity_id: str, entity_type: str = "unknown", **attrs):
        raise NotImplementedError

    def add_edge(self, source: str, relation: str, target: str,
                 confidence: float = 1.0, episode_id: str = ""):
        raise NotImplementedError

    def traverse_bfs(self, start: str, max_depth: int = 2) -> List[Dict]:
        raise NotImplementedError

    def multi_hop_query(self, start: str, relation_path: List[str]) -> List[str]:
        raise NotImplementedError

    def to_context_string(self, entity_id: str, max_depth: int = 2) -> str:
        raise NotImplementedError

    def get_all_entities(self) -> List[Dict]:
        raise NotImplementedError

    def stats(self) -> Dict[str, int]:
        raise NotImplementedError


class Neo4jGraphBackend(GraphBackend):
    """
    Production graph backend using Neo4j with Cypher queries.

    Neo4j advantages over SQLite:
      - Native graph storage (no JOIN overhead for traversals)
      - Cypher query language (expressive pattern matching)
      - Scalable to millions of nodes/edges
      - Built-in graph algorithms (PageRank, community detection)
      - Concurrent multi-user access
      - ACID transactions
    """

    def __init__(self, driver):
        self.driver = driver
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for fast lookups."""
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def add_entity(self, entity_id: str, entity_type: str = "unknown", **attrs):
        entity_id = entity_id.lower().strip()
        attrs_json = json.dumps(attrs)
        with self.driver.session() as session:
            # MERGE: create if not exists, then always update type/attrs.
            # ON CREATE SET must come BEFORE the general SET clause.
            session.run(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET e.created_at = datetime()
                SET e.type = $type,
                    e.attributes = $attrs,
                    e.updated_at = datetime()
                """,
                id=entity_id, type=entity_type, attrs=attrs_json,
            )

    def add_edge(self, source: str, relation: str, target: str,
                 confidence: float = 1.0, episode_id: str = ""):
        source = source.lower().strip()
        target = target.lower().strip()
        relation = relation.upper().strip().replace(" ", "_")

        with self.driver.session() as session:
            # Ensure both entities exist
            session.run(
                "MERGE (e:Entity {id: $id}) ON CREATE SET e.created_at = datetime()",
                id=source,
            )
            session.run(
                "MERGE (e:Entity {id: $id}) ON CREATE SET e.created_at = datetime()",
                id=target,
            )

            # Close conflicting active edges (temporal resolution).
            # We match outgoing edges of the same type from the same source
            # that point to a DIFFERENT target and are still active.
            session.run(
                """
                MATCH (s:Entity {id: $source})-[r]->(t:Entity)
                WHERE type(r) = $relation AND t.id <> $target
                  AND r.valid_until IS NULL
                SET r.valid_until = datetime()
                """,
                source=source, relation=relation, target=target,
            )

            # Create or update the edge with temporal metadata.
            # ON CREATE SET fires only when the edge is new;
            # the general SET always runs (updates confidence etc.).
            session.run(
                f"""
                MATCH (s:Entity {{id: $source}})
                MATCH (t:Entity {{id: $target}})
                MERGE (s)-[r:{relation}]->(t)
                ON CREATE SET r.valid_from = datetime()
                SET r.confidence = $confidence,
                    r.episode_id = $episode_id,
                    r.valid_until = NULL
                """,
                source=source, target=target,
                confidence=confidence, episode_id=episode_id,
            )

    def traverse_bfs(self, start: str, max_depth: int = 2) -> List[Dict]:
        start = start.lower().strip()
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (s:Entity {id: $start})-[r*1..""" + str(max_depth) + """]->(t:Entity)
                WHERE ALL(rel IN relationships(path) WHERE rel.valid_until IS NULL)
                UNWIND range(0, length(path)-1) AS idx
                WITH nodes(path)[idx] AS src, relationships(path)[idx] AS rel, nodes(path)[idx+1] AS tgt, idx+1 AS depth
                RETURN DISTINCT src.id AS source, type(rel) AS relation, tgt.id AS target, depth
                ORDER BY depth, source
                """,
                start=start,
            )
            return [
                {"source": r["source"], "relation": r["relation"].lower(),
                 "target": r["target"], "depth": r["depth"]}
                for r in result
            ]

    def multi_hop_query(self, start: str, relation_path: List[str]) -> List[str]:
        start = start.lower().strip()

        # Build dynamic Cypher pattern: (s)-[:REL1]->()-[:REL2]->(end)
        pattern_parts = [f"(n0:Entity {{id: $start}})"]
        for i, rel in enumerate(relation_path):
            rel_upper = rel.upper().replace(" ", "_")
            pattern_parts.append(f"-[:{rel_upper}]->(n{i+1})")

        pattern = "".join(pattern_parts)
        last_var = f"n{len(relation_path)}"
        query = f"MATCH {pattern} WHERE ALL(r IN relationships((n0){'-[*]->' + '(' + last_var + ')'}) WHERE r.valid_until IS NULL) RETURN DISTINCT {last_var}.id AS result"

        # Simpler approach: iterative traversal
        current = {start}
        with self.driver.session() as session:
            for rel in relation_path:
                rel_upper = rel.upper().replace(" ", "_")
                next_set = set()
                for entity in current:
                    result = session.run(
                        f"MATCH (s:Entity {{id: $id}})-[r:{rel_upper}]->(t:Entity) "
                        f"WHERE r.valid_until IS NULL RETURN t.id AS target",
                        id=entity,
                    )
                    for record in result:
                        next_set.add(record["target"])
                current = next_set
                if not current:
                    return []
        return list(current)

    def to_context_string(self, entity_id: str, max_depth: int = 2) -> str:
        entity_id = entity_id.lower().strip()
        with self.driver.session() as session:
            # Get entity info
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e.type AS type, e.attributes AS attrs",
                id=entity_id,
            )
            record = result.single()
            if not record:
                return f"No information about '{entity_id}'."

            etype = record["type"] or "unknown"
            lines = [f"{entity_id} ({etype}):"]

        facts = self.traverse_bfs(entity_id, max_depth)
        for fact in facts:
            indent = "  " * fact["depth"]
            lines.append(f"{indent}- {fact['source']} --[{fact['relation']}]--> {fact['target']}")

        return "\n".join(lines)

    def get_all_entities(self) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.id IS NOT NULL RETURN e.id AS id, e.type AS type ORDER BY e.id"
            )
            return [{"id": r["id"], "type": r["type"]} for r in result]

    def stats(self) -> Dict[str, int]:
        with self.driver.session() as session:
            entities = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            total = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            active = session.run(
                "MATCH ()-[r]->() WHERE r.valid_until IS NULL RETURN count(r) AS c"
            ).single()["c"]
            return {
                "entities": entities,
                "total_edges": total,
                "active_edges": active,
                "historical_edges": total - active,
                "backend": "neo4j",
            }


class SQLiteGraphBackend(GraphBackend):
    """Fallback graph backend using SQLite (from Example 20c)."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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
                UNIQUE(source, relation, target, valid_until)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self.conn.commit()

    def add_entity(self, entity_id: str, entity_type: str = "unknown", **attrs):
        entity_id = entity_id.lower().strip()
        self.conn.execute(
            "INSERT INTO entities (id, entity_type, attributes) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET entity_type=excluded.entity_type, updated_at=datetime('now')",
            (entity_id, entity_type, json.dumps(attrs)),
        )
        self.conn.commit()

    def add_edge(self, source: str, relation: str, target: str,
                 confidence: float = 1.0, episode_id: str = ""):
        source, target, relation = source.lower().strip(), target.lower().strip(), relation.lower().strip()
        self.add_entity(source)
        self.add_entity(target)
        self.conn.execute(
            "UPDATE edges SET valid_until=datetime('now') "
            "WHERE source=? AND relation=? AND target!=? AND valid_until IS NULL",
            (source, relation, target),
        )
        try:
            self.conn.execute(
                "INSERT INTO edges (source,relation,target,confidence,episode_id) VALUES (?,?,?,?,?)",
                (source, relation, target, confidence, episode_id),
            )
        except sqlite3.IntegrityError:
            pass
        self.conn.commit()

    def _get_active_edges(self, entity_id: str) -> List[Dict]:
        cursor = self.conn.execute(
            "SELECT source, relation, target FROM edges "
            "WHERE (source=? OR target=?) AND valid_until IS NULL",
            (entity_id, entity_id),
        )
        return [{"source": r[0], "relation": r[1], "target": r[2]} for r in cursor]

    def traverse_bfs(self, start: str, max_depth: int = 2) -> List[Dict]:
        start = start.lower().strip()
        visited, queue, results = {start}, deque([(start, 0)]), []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._get_active_edges(current):
                if edge["source"] == current:
                    results.append({**edge, "depth": depth + 1})
                    if edge["target"] not in visited:
                        visited.add(edge["target"])
                        queue.append((edge["target"], depth + 1))
        return results

    def multi_hop_query(self, start: str, relation_path: List[str]) -> List[str]:
        current = {start.lower().strip()}
        for rel in relation_path:
            nxt = set()
            for entity in current:
                cursor = self.conn.execute(
                    "SELECT target FROM edges WHERE source=? AND relation=? AND valid_until IS NULL",
                    (entity, rel),
                )
                for row in cursor:
                    nxt.add(row[0])
            current = nxt
            if not current:
                return []
        return list(current)

    def to_context_string(self, entity_id: str, max_depth: int = 2) -> str:
        entity_id = entity_id.lower().strip()
        cursor = self.conn.execute("SELECT entity_type FROM entities WHERE id=?", (entity_id,))
        row = cursor.fetchone()
        if not row:
            return f"No information about '{entity_id}'."
        lines = [f"{entity_id} ({row[0]}):"]
        for fact in self.traverse_bfs(entity_id, max_depth):
            indent = "  " * fact["depth"]
            lines.append(f"{indent}- {fact['source']} --[{fact['relation']}]--> {fact['target']}")
        return "\n".join(lines)

    def get_all_entities(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT id, entity_type FROM entities ORDER BY id")
        return [{"id": r[0], "type": r[1]} for r in cursor]

    def stats(self) -> Dict[str, int]:
        entities = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        total = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        active = self.conn.execute("SELECT COUNT(*) FROM edges WHERE valid_until IS NULL").fetchone()[0]
        return {"entities": entities, "total_edges": total, "active_edges": active,
                "historical_edges": total - active, "backend": "sqlite"}


# ==================================================================
# Initialize the appropriate backend
# ==================================================================

DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(exist_ok=True)

if NEO4J_AVAILABLE:
    GRAPH = Neo4jGraphBackend(neo4j_driver)
    print("[GRAPH] Using Neo4j backend")
else:
    GRAPH = SQLiteGraphBackend(str(DB_DIR / "knowledge_graph_prod.db"))
    print("[GRAPH] Using SQLite fallback backend")

print(f"[GRAPH] Current state: {GRAPH.stats()}")

# Vector store for hybrid retrieval
print("[INIT] Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHROMA_CLIENT = chromadb.EphemeralClient()
try:
    CHROMA_CLIENT.delete_collection("prod_graph_docs")
except Exception:
    pass
VECTOR_STORE = CHROMA_CLIENT.create_collection(name="prod_graph_docs")

DOCS = [
    "Alice is the ML team lead with 8 years of deep learning experience, managing 5 engineers.",
    "Bob is a senior data scientist specializing in recommendation systems and collaborative filtering.",
    "The search engine uses PyTorch for training and FastAPI for serving, handling 10M daily queries.",
    "The chatbot project uses LangChain and LangGraph for orchestration, started in Q3 2024.",
    "Carol leads the backend team building the API platform with FastAPI and PostgreSQL.",
    "Diana is VP of Engineering overseeing ML and backend teams, reporting to the CEO.",
    "The ML team adopted Arize Phoenix for observability across all their projects.",
]
embeddings = EMBED_MODEL.encode(DOCS).tolist()
VECTOR_STORE.add(
    ids=[f"doc_{i}" for i in range(len(DOCS))],
    documents=DOCS, embeddings=embeddings,
)
print(f"[INDEX] ChromaDB: {VECTOR_STORE.count()} docs\n")


# ==================================================================
# GRAPHITI EPISODE INGESTION (when Neo4j is available)
# ==================================================================
# Graphiti uses async Neo4j drivers internally. Since LangGraph nodes
# are synchronous, we run Graphiti operations in a dedicated thread
# with its own event loop to avoid async driver conflicts.

import threading
import concurrent.futures

_graphiti_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _run_async_in_thread(coro):
    """Run an async coroutine in a dedicated thread with its own event loop."""
    def _run():
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_silence_async_cleanup)
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    future = _graphiti_executor.submit(_run)
    return future.result(timeout=60)


def _make_graphiti():
    """Create a fresh Graphiti client (each needs its own async driver)."""
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIClient, LLMConfig as LC
    cfg = _graphiti_config
    return Graphiti(
        uri=cfg["uri"], user=cfg["user"], password=cfg["password"],
        llm_client=OpenAIClient(cfg["llm_config"]),
    )


def ingest_with_graphiti(text: str, episode_name: str = "conversation"):
    """
    Use Graphiti to automatically extract entities and relationships.

    Graphiti does what our manual LLM extraction does, but better:
      - Automatic entity resolution (deduplication)
      - Temporal edge management
      - Confidence scoring
      - Community detection
    """
    if not GRAPHITI_AVAILABLE or not _graphiti_config:
        return

    try:
        from graphiti_core.nodes import EpisodeType

        async def _ingest():
            g = _make_graphiti()
            try:
                await g.add_episode(
                    name=episode_name,
                    episode_body=text,
                    source_description="Agent conversation turn",
                    reference_time=datetime.now(),
                    source=EpisodeType.text,
                )
            finally:
                await g.close()

        _run_async_in_thread(_ingest())
        print(f"  [GRAPHITI] Ingested episode: {episode_name}")
    except Exception as e:
        print(f"  [GRAPHITI] Ingestion failed: {e}")


def search_with_graphiti(query: str, num_results: int = 5) -> str:
    """Search the Graphiti graph with semantic search."""
    if not GRAPHITI_AVAILABLE or not _graphiti_config:
        return ""

    try:
        async def _search():
            g = _make_graphiti()
            try:
                return await g.search(query=query, num_results=num_results)
            finally:
                await g.close()

        results = _run_async_in_thread(_search())
        if results:
            lines = ["[GRAPHITI SEARCH RESULTS]"]
            if hasattr(results, 'edges') and results.edges:
                for edge in results.edges[:5]:
                    lines.append(
                        f"  - {edge.source_node_name} --[{edge.name}]--> "
                        f"{edge.target_node_name}: {edge.fact}"
                    )
            if hasattr(results, 'nodes') and results.nodes:
                for node in results.nodes[:5]:
                    lines.append(f"  - Entity: {node.name}")
            return "\n".join(lines)
    except Exception as e:
        print(f"  [GRAPHITI] Search failed: {e}")

    return ""


# ==================================================================
# LLM + STATE + LANGGRAPH PIPELINE
# ==================================================================

def get_llm(temperature=0.3):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


llm = get_llm()


class ProdGraphRAGState(TypedDict):
    query: str
    entities: List[str]
    graph_context: str
    graphiti_context: str
    vector_context: str
    combined_context: str
    answer: str
    graph_stats: Dict[str, Any]


def extract_entities_node(state: ProdGraphRAGState) -> dict:
    """Extract entities via LLM and add to the graph backend."""
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
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        for ent in data.get("entities", []):
            name = ent.get("name", "").lower().strip()
            if name:
                GRAPH.add_entity(name, ent.get("type", "unknown"))
                entity_names.append(name)

        for rel in data.get("relations", []):
            src = rel.get("source", "").lower().strip()
            r = rel.get("relation", "").lower().strip()
            tgt = rel.get("target", "").lower().strip()
            if src and r and tgt:
                GRAPH.add_edge(src, r, tgt, episode_id=f"turn-{datetime.now().isoformat()}")

    except (json.JSONDecodeError, Exception):
        stop = {"what","who","how","does","is","are","the","a","an","of","and","or","in","on","to","for","with","about"}
        entity_names = [w.strip("?,.!'\"") for w in query.lower().split()
                       if w.strip("?,.!'\"") not in stop and len(w) > 2][:5]

    # Also ingest into Graphiti if available (sync wrapper handles async)
    if GRAPHITI_AVAILABLE:
        ingest_with_graphiti(query, f"turn-{len(entity_names)}")

    print(f"  [EXTRACT] Entities: {entity_names} (backend: {GRAPH.stats().get('backend', '?')})")
    return {"entities": entity_names, "graph_stats": GRAPH.stats()}


def query_graph_node(state: ProdGraphRAGState) -> dict:
    """Query the graph backend (Neo4j or SQLite)."""
    entities = state.get("entities", [])
    subgraphs = []

    for entity in entities:
        ctx = GRAPH.to_context_string(entity, max_depth=2)
        if "No information" not in ctx:
            subgraphs.append(ctx)

    # Fallback: check if any known entity partially matches
    if not subgraphs:
        all_ents = GRAPH.get_all_entities()
        for ent in all_ents:
            ent_id = ent.get("id") or ""
            if ent_id and any(e in ent_id for e in entities):
                ctx = GRAPH.to_context_string(ent_id, max_depth=2)
                if "No information" not in ctx:
                    subgraphs.append(ctx)

    graph_context = "\n\n".join(subgraphs) if subgraphs else ""

    # Also query Graphiti if available (sync wrapper handles async)
    graphiti_context = ""
    if GRAPHITI_AVAILABLE:
        graphiti_context = search_with_graphiti(state["query"])

    print(f"  [GRAPH] {len(subgraphs)} subgraphs, ~{len(graph_context.split())} tokens")
    if graphiti_context:
        print(f"  [GRAPHITI] Additional context: {len(graphiti_context.split())} tokens")

    return {"graph_context": graph_context, "graphiti_context": graphiti_context}


def hybrid_retrieve_node(state: ProdGraphRAGState) -> dict:
    """Hybrid: graph + Graphiti + ChromaDB vector search."""
    query = state["query"]
    graph_ctx = state.get("graph_context", "")
    graphiti_ctx = state.get("graphiti_context", "")

    # Vector search
    q_emb = EMBED_MODEL.encode(query).tolist()
    results = VECTOR_STORE.query(query_embeddings=[q_emb], n_results=3, include=["documents", "distances"])
    vector_parts = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        vector_parts.append(f"[{1.0/(1.0+dist):.2f}] {doc}")
    vector_ctx = "\n".join(vector_parts)

    # Combine all sources
    parts = []
    if graph_ctx:
        parts.append(f"=== KNOWLEDGE GRAPH ({GRAPH.stats().get('backend','').upper()}) ===\n{graph_ctx}")
    if graphiti_ctx:
        parts.append(f"=== GRAPHITI (temporal/semantic) ===\n{graphiti_ctx}")
    parts.append(f"=== DOCUMENT SEARCH (ChromaDB) ===\n{vector_ctx}")

    combined = "\n\n".join(parts)
    print(f"  [HYBRID] Combined: ~{len(combined.split())} tokens from {len(parts)} sources")

    return {"vector_context": vector_ctx, "combined_context": combined}


def generate_node(state: ProdGraphRAGState) -> dict:
    """Generate answer from hybrid context."""
    stats = state.get("graph_stats", {})
    prompt = [
        SystemMessage(content=(
            f"You are a knowledge assistant backed by a {stats.get('backend','graph')} knowledge graph "
            f"({stats.get('entities',0)} entities, {stats.get('active_edges',0)} relationships) "
            f"and a document store.\n\n"
            f"CONTEXT:\n{state.get('combined_context','')}\n\n"
            f"Answer using the context. Reference graph relationships for relational questions. "
            f"Be concise."
        )),
        HumanMessage(content=state["query"]),
    ]
    try:
        answer = llm.invoke(prompt).content.strip()
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [GENERATE] {answer[:100]}...")
    return {"answer": answer}


def build_pipeline():
    graph = StateGraph(ProdGraphRAGState)
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
# FASTAPI SERVICE (Graph-as-a-Service pattern)
# ==================================================================

def create_api_app():
    """
    Wrap the knowledge graph as a REST API service.

    This is the "Graph as a Service" pattern -- other agents
    and applications can query the graph over HTTP.

    To run as a server:
      uvicorn example_20d_context_graphs_neo4j:api_app --port 8080
    """
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Knowledge Graph API", version="1.0")

    class EntityRequest(BaseModel):
        entity_id: str
        entity_type: str = "unknown"

    class EdgeRequest(BaseModel):
        source: str
        relation: str
        target: str

    class QueryRequest(BaseModel):
        query: str

    @app.get("/stats")
    def get_stats():
        return GRAPH.stats()

    @app.get("/entities")
    def list_entities():
        return GRAPH.get_all_entities()

    @app.get("/entity/{entity_id}")
    def get_entity(entity_id: str, depth: int = 2):
        return {"context": GRAPH.to_context_string(entity_id, depth)}

    @app.post("/entity")
    def add_entity(req: EntityRequest):
        GRAPH.add_entity(req.entity_id, req.entity_type)
        return {"status": "ok", "entity": req.entity_id}

    @app.post("/edge")
    def add_edge(req: EdgeRequest):
        GRAPH.add_edge(req.source, req.relation, req.target)
        return {"status": "ok", "edge": f"{req.source}-[{req.relation}]->{req.target}"}

    @app.post("/multi-hop")
    def multi_hop(start: str, relations: str):
        """relations as comma-separated: manages,works_on"""
        path = [r.strip() for r in relations.split(",")]
        results = GRAPH.multi_hop_query(start, path)
        return {"start": start, "path": path, "results": results}

    @app.post("/query")
    def query_rag(req: QueryRequest):
        pipeline = build_pipeline()
        result = pipeline.invoke({
            "query": req.query, "entities": [], "graph_context": "",
            "graphiti_context": "", "vector_context": "",
            "combined_context": "", "answer": "", "graph_stats": {},
        })
        return {"answer": result["answer"], "entities": result["entities"],
                "graph_stats": result["graph_stats"]}

    return app


# Create the API app (importable for uvicorn)
api_app = create_api_app()


# ==================================================================
# DEMO
# ==================================================================

def clear_graph():
    """Reset the graph to a clean state for a fresh demo run."""
    if NEO4J_AVAILABLE and neo4j_driver:
        with neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("  [RESET] Neo4j cleared")
    elif isinstance(GRAPH, SQLiteGraphBackend):
        GRAPH.conn.executescript("DELETE FROM edges; DELETE FROM entities;")
        GRAPH.conn.commit()
        print("  [RESET] SQLite graph cleared")

    # Re-seed with initial facts
    seed = [
        ("alice", "person", "manages", "ml_team"),
        ("bob", "person", "member_of", "ml_team"),
        ("carol", "person", "leads", "backend_team"),
        ("diana", "person", "oversees", "alice"),
        ("diana", "person", "oversees", "carol"),
        ("ml_team", "team", "works_on", "search_engine"),
        ("ml_team", "team", "works_on", "chatbot"),
        ("backend_team", "team", "works_on", "api_platform"),
        ("search_engine", "project", "uses", "pytorch"),
        ("chatbot", "project", "uses", "langchain"),
        ("api_platform", "project", "uses", "fastapi"),
    ]
    for src, stype, rel, tgt in seed:
        GRAPH.add_entity(src, stype)
        GRAPH.add_edge(src, rel, tgt, episode_id="seed")
    print(f"  [RESET] Re-seeded: {GRAPH.stats()}")


def run_demo():
    # Always start clean so the demo output is consistent.
    # Remove this call if you want to see accumulated knowledge across runs.
    clear_graph()

    app = build_pipeline()

    turns = [
        "What projects does Alice's team work on?",
        "Bob is also working on a new recommendation engine using pytorch.",
        "Who does Alice report to, and what teams does that person oversee?",
        "The chatbot project has been moved to the backend team under Carol.",
        "What technologies are used across all projects?",
    ]

    backend = GRAPH.stats().get("backend", "unknown")
    print("=" * 65)
    print(f"  PRODUCTION CONTEXT GRAPHS ({backend.upper()})")
    print(f"  Neo4j: {'CONNECTED' if NEO4J_AVAILABLE else 'not available (SQLite fallback)'}")
    print(f"  Graphiti: {'ACTIVE' if GRAPHITI_AVAILABLE else 'not configured'}")
    print("=" * 65)

    for i, turn in enumerate(turns):
        print(f"\n{'-' * 65}")
        print(f"  Turn {i+1}: {turn}")
        print(f"{'-' * 65}")

        result = app.invoke({
            "query": turn, "entities": [], "graph_context": "",
            "graphiti_context": "", "vector_context": "",
            "combined_context": "", "answer": "", "graph_stats": {},
        })

        print(f"\n  Answer: {result['answer'][:350]}")

    # Final state
    print(f"\n{'=' * 65}")
    print(f"  FINAL GRAPH STATE ({backend})")
    print(f"{'=' * 65}")
    print(f"  Stats: {GRAPH.stats()}")
    print(f"  Entities: {[e['id'] for e in GRAPH.get_all_entities()]}")

    # Multi-hop demos
    print(f"\n  Multi-hop queries:")
    for start, path in [("alice", ["manages", "works_on"]), ("diana", ["oversees"])]:
        result = GRAPH.multi_hop_query(start, path)
        print(f"    {start} -> {' -> '.join(path)} -> {result}")

    # FastAPI info
    print(f"\n{'=' * 65}")
    print(f"  GRAPH-AS-A-SERVICE (FastAPI)")
    print(f"{'=' * 65}")
    print(f"  To start the API server:")
    print(f"    uvicorn week-05-context-memory.examples.example_20d_context_graphs_neo4j:api_app --port 8080")
    print(f"\n  Endpoints:")
    print(f"    GET  /stats           - Graph statistics")
    print(f"    GET  /entities        - List all entities")
    print(f"    GET  /entity/alice    - Subgraph for an entity")
    print(f"    POST /entity          - Add an entity")
    print(f"    POST /edge            - Add a relationship")
    print(f"    POST /multi-hop       - Multi-hop traversal")
    print(f"    POST /query           - Full RAG query (graph + vector + LLM)")


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  WEEK 5 - EXAMPLE 20d: Context Graphs (Neo4j + Graphiti)")
    print("=" * 65)

    run_demo()

    # Cleanup
    if neo4j_driver:
        neo4j_driver.close()

    print("\n" + "=" * 65)
    print("  PRODUCTION ARCHITECTURE SUMMARY")
    print("=" * 65)
    print(textwrap.dedent("""
    COMPONENT STACK:

    +------------------+     +------------------+     +------------------+
    | LangGraph        |     | Neo4j            |     | ChromaDB         |
    | Pipeline         |---->| Knowledge Graph  |     | Vector Store     |
    | (orchestration)  |     | (relationships)  |     | (documents)      |
    +------------------+     +------------------+     +------------------+
           |                        |                        |
           v                        v                        v
    +------------------+     +------------------+     +------------------+
    | LLM              |     | Graphiti         |     | sentence-        |
    | (extraction +    |     | (auto entity     |     | transformers     |
    |  generation)     |     |  resolution +    |     | (embeddings)     |
    +------------------+     |  temporal edges) |     +------------------+
                             +------------------+
                                    |
                             +------------------+
                             | FastAPI          |
                             | (graph as a      |
                             |  service API)    |
                             +------------------+

    WHEN TO USE EACH BACKEND:

    SQLite (Example 20c):
      - Single user, single process
      - Prototyping and development
      - < 100K entities

    Neo4j (THIS EXAMPLE):
      - Multi-user, multi-process
      - Complex graph queries (Cypher)
      - > 100K entities, need graph algorithms
      - Need ACID transactions

    Graphiti (THIS EXAMPLE, optional):
      - Want automatic entity extraction
      - Need temporal edge management
      - Want semantic search over the graph
      - Don't want to write extraction prompts

    FastAPI wrapper (THIS EXAMPLE):
      - Other services need to query the graph
      - Microservice architecture
      - Need authentication/rate limiting
    """))
