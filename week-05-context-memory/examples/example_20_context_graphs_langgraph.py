import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 20: Context Graphs — LangGraph Implementation
=======================================================
LangGraph agent that builds and queries a knowledge graph during
conversation.  The graph persists across invocations so knowledge
accumulates over multiple turns.

Nodes:
  extract_entities -> query_graph -> expand_subgraph -> generate_with_graph_context -> END

Demonstrates:
  - Entity extraction from user messages
  - Graph-augmented generation (inject subgraph context, not raw chunks)
  - Multi-hop reasoning via graph traversal
  - Persistent knowledge graph that grows with each conversation turn

Run: python week-05-context-memory/examples/example_20_context_graphs_langgraph.py
"""

import os
import sys
import uuid
import json
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, List, Dict, Optional, Any, Set, Tuple
from collections import deque
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END


# ── Phoenix ────────────────────────────────────────────────────
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
        tracer_provider = register(project_name="week5-context-graphs")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None


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
# PERSISTENT KNOWLEDGE GRAPH
# ================================================================
# This graph lives at module level so it persists across invocations.
# In production, you'd back this with Neo4j or a database.
# Each conversation turn adds new entities and relations, building
# a richer knowledge base over time.

class KnowledgeGraph:
    """Module-level knowledge graph that grows across turns."""

    def __init__(self):
        self.adjacency: Dict[str, List[Tuple[str, str]]] = {}
        self.entity_types: Dict[str, str] = {}

    def add_entity(self, entity_id: str, entity_type: str = "unknown"):
        entity_id = entity_id.lower().strip()
        if entity_id not in self.adjacency:
            self.adjacency[entity_id] = []
        self.entity_types[entity_id] = entity_type

    def add_relation(self, source: str, relation: str, target: str):
        source = source.lower().strip()
        target = target.lower().strip()
        relation = relation.lower().strip()
        self.add_entity(source)
        self.add_entity(target)
        # Avoid duplicates
        if (relation, target) not in self.adjacency[source]:
            self.adjacency[source].append((relation, target))

    def get_subgraph(self, entity_id: str, max_depth: int = 2) -> str:
        """BFS traversal returning a compact text representation."""
        entity_id = entity_id.lower().strip()
        if entity_id not in self.adjacency:
            return f"No information about '{entity_id}'."

        visited: Set[str] = {entity_id}
        queue: deque = deque([(entity_id, 0)])
        lines = [f"Knowledge about {entity_id} ({self.entity_types.get(entity_id, 'unknown')}):"]

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for relation, target in self.adjacency.get(current, []):
                indent = "  " * (depth + 1)
                lines.append(f"{indent}{current} --[{relation}]--> {target}")
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))

        return "\n".join(lines) if len(lines) > 1 else f"Entity '{entity_id}' exists but has no relations."

    def get_all_entities(self) -> List[str]:
        return list(self.adjacency.keys())

    def stats(self) -> Dict[str, int]:
        total_edges = sum(len(edges) for edges in self.adjacency.values())
        return {"entities": len(self.adjacency), "edges": total_edges}


# Module-level graph — persists across all invocations
KNOWLEDGE_GRAPH = KnowledgeGraph()


# ================================================================
# STATE
# ================================================================
# GraphRAGState carries all the data flowing through the pipeline.
# The 'graph' field holds a serializable snapshot so we can inspect
# it, but the actual graph is the module-level KNOWLEDGE_GRAPH.

class GraphRAGState(TypedDict):
    query: str                    # User's original question
    entities: List[str]           # Entities extracted from the query
    graph: Dict[str, Any]         # Serialized graph snapshot (for inspection)
    subgraph: str                 # Text context extracted from graph
    answer: str                   # Final generated answer


# ================================================================
# NODE 1: EXTRACT ENTITIES
# ================================================================
# The first step is identifying what entities the user is asking about.
# We use the LLM to extract entity names and types from the query.
# These become the starting points for graph traversal.

def extract_entities(state: GraphRAGState) -> GraphRAGState:
    """Extract entities from the user query using the LLM."""
    llm = get_llm(temperature=0.0)
    query = state["query"]

    prompt = [
        SystemMessage(content=textwrap.dedent("""
            Extract entities and relationships from the user message.
            Return a JSON object with:
            {
                "entities": [{"name": "...", "type": "person|org|project|technology|concept"}],
                "relations": [{"source": "...", "relation": "...", "target": "..."}]
            }
            Only extract what is explicitly stated.  Do not infer.
            Keep entity names lowercase.  Return valid JSON only.
        """)),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(prompt)
        # Parse the JSON from the LLM response
        text = response.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)

        # Add extracted entities and relations to the persistent graph
        entity_names = []
        for ent in data.get("entities", []):
            name = ent.get("name", "").lower().strip()
            etype = ent.get("type", "unknown")
            if name:
                KNOWLEDGE_GRAPH.add_entity(name, etype)
                entity_names.append(name)

        for rel in data.get("relations", []):
            src = rel.get("source", "").lower().strip()
            r = rel.get("relation", "").lower().strip()
            tgt = rel.get("target", "").lower().strip()
            if src and r and tgt:
                KNOWLEDGE_GRAPH.add_relation(src, r, tgt)

        return {**state, "entities": entity_names}

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: extract simple nouns as entities
        words = query.lower().split()
        # Filter common stop words
        stop = {"what", "who", "how", "does", "is", "are", "the", "a", "an", "of",
                "and", "or", "in", "on", "at", "to", "for", "with", "about", "that"}
        entities = [w.strip("?,.'\"!") for w in words if w.strip("?,.'\"!") not in stop and len(w) > 2]
        return {**state, "entities": entities[:5]}


# ================================================================
# NODE 2: QUERY GRAPH
# ================================================================
# Look up each extracted entity in the knowledge graph and pull
# its subgraph.  This is where graph traversal replaces vector search.

def query_graph(state: GraphRAGState) -> GraphRAGState:
    """Query the knowledge graph for each extracted entity."""
    entities = state.get("entities", [])
    subgraphs = []

    for entity in entities:
        sg = KNOWLEDGE_GRAPH.get_subgraph(entity, max_depth=2)
        subgraphs.append(sg)

    # Combine all subgraphs into a single context string
    combined = "\n\n".join(subgraphs) if subgraphs else "No relevant information in the knowledge graph."

    # Snapshot the graph state for inspection
    graph_snapshot = {
        "stats": KNOWLEDGE_GRAPH.stats(),
        "entities": KNOWLEDGE_GRAPH.get_all_entities(),
    }

    return {**state, "subgraph": combined, "graph": graph_snapshot}


# ================================================================
# NODE 3: EXPAND SUBGRAPH
# ================================================================
# If the initial subgraph is thin (few facts), we expand to deeper
# traversal or related entities.  This simulates "follow-up retrieval"
# that agentic RAG does with vector stores, but using graph hops.

def expand_subgraph(state: GraphRAGState) -> GraphRAGState:
    """Expand the subgraph if initial context is too sparse."""
    subgraph = state.get("subgraph", "")
    entities = state.get("entities", [])

    # Count facts in the subgraph (lines with --> are facts)
    fact_count = subgraph.count("-->")

    if fact_count < 3 and entities:
        # Expand: get deeper traversal and include neighbor entities
        expanded_parts = [subgraph]
        for entity in entities:
            entity = entity.lower().strip()
            # Check neighbors at depth 3
            deep_sg = KNOWLEDGE_GRAPH.get_subgraph(entity, max_depth=3)
            if deep_sg not in expanded_parts:
                expanded_parts.append(deep_sg)

            # Also include entities that reference this entity
            for eid, edges in KNOWLEDGE_GRAPH.adjacency.items():
                for rel, target in edges:
                    if target == entity and eid not in entities:
                        reverse_sg = KNOWLEDGE_GRAPH.get_subgraph(eid, max_depth=1)
                        if reverse_sg not in expanded_parts:
                            expanded_parts.append(reverse_sg)

        subgraph = "\n\n".join(expanded_parts)

    return {**state, "subgraph": subgraph}


# ================================================================
# NODE 4: GENERATE WITH GRAPH CONTEXT
# ================================================================
# The final node injects the graph context into the LLM prompt and
# generates an answer.  Notice how the context is structured facts,
# not raw document chunks — this is more token-efficient and precise.

def generate_with_graph_context(state: GraphRAGState) -> GraphRAGState:
    """Generate answer using the graph-derived context."""
    llm = get_llm(temperature=0.7)
    query = state["query"]
    subgraph = state.get("subgraph", "No context available.")
    graph_info = state.get("graph", {})

    system_prompt = textwrap.dedent(f"""
        You are a knowledge assistant powered by a context graph.

        KNOWLEDGE GRAPH CONTEXT:
        {subgraph}

        GRAPH STATS: {graph_info.get('stats', 'N/A')}

        Instructions:
        - Answer the user's question using ONLY the knowledge graph context above.
        - If the graph doesn't contain enough information, say so honestly.
        - Reference specific relationships from the graph to support your answer.
        - Be concise but thorough.
    """)

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"[Error generating answer: {e}]"

    return {**state, "answer": answer}


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_graph_rag_pipeline():
    """
    Build the LangGraph pipeline:
      extract_entities -> query_graph -> expand_subgraph -> generate -> END
    """
    graph = StateGraph(GraphRAGState)

    graph.add_node("extract_entities", extract_entities)
    graph.add_node("query_graph", query_graph)
    graph.add_node("expand_subgraph", expand_subgraph)
    graph.add_node("generate", generate_with_graph_context)

    graph.set_entry_point("extract_entities")
    graph.add_edge("extract_entities", "query_graph")
    graph.add_edge("query_graph", "expand_subgraph")
    graph.add_edge("expand_subgraph", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ================================================================
# DEMO
# ================================================================

def run_demo():
    app = build_graph_rag_pipeline()

    # Seed the knowledge graph with some initial facts.
    # In production, these would come from document ingestion or prior conversations.
    seed_facts = [
        ("alice", "person", "manages", "ml_team"),
        ("bob", "person", "member_of", "ml_team"),
        ("carol", "person", "leads", "backend_team"),
        ("ml_team", "team", "works_on", "search_engine"),
        ("ml_team", "team", "works_on", "chatbot"),
        ("backend_team", "team", "works_on", "api_platform"),
        ("search_engine", "project", "uses", "pytorch"),
        ("chatbot", "project", "uses", "langchain"),
        ("api_platform", "project", "uses", "fastapi"),
        ("alice", "person", "reports_to", "diana"),
        ("carol", "person", "reports_to", "diana"),
        ("diana", "person", "role_is", "vp_engineering"),
    ]

    for source, stype, relation, target in seed_facts:
        KNOWLEDGE_GRAPH.add_entity(source, stype)
        KNOWLEDGE_GRAPH.add_relation(source, relation, target)

    print(f"\n  Seeded graph: {KNOWLEDGE_GRAPH.stats()}")

    # Simulate a multi-turn conversation.
    # Each turn may ADD new knowledge (from the user's message) AND
    # QUERY existing knowledge.  The graph grows over time.
    turns = [
        "What projects does Alice's team work on?",
        "Bob is also working on a new recommendation engine project using pytorch.",
        "Who does Alice report to, and what teams does that person oversee?",
        "The chatbot project has been moved to the backend team under Carol.",
        "What technologies are used across all projects?",
    ]

    print("\n" + "=" * 65)
    print("  GRAPH-AUGMENTED CONVERSATION")
    print("=" * 65)

    for i, user_input in enumerate(turns):
        print(f"\n{'━' * 65}")
        print(f"  Turn {i + 1}: {user_input}")
        print(f"{'━' * 65}")

        result = app.invoke({
            "query": user_input,
            "entities": [],
            "graph": {},
            "subgraph": "",
            "answer": "",
        })

        print(f"  Entities: {result.get('entities', [])}")
        print(f"  Graph: {result.get('graph', {}).get('stats', 'N/A')}")
        print(f"  Answer: {result.get('answer', 'N/A')[:300]}")

    # Show final graph state
    print(f"\n{'=' * 65}")
    print(f"  FINAL KNOWLEDGE GRAPH STATE")
    print(f"{'=' * 65}")
    print(f"  Stats: {KNOWLEDGE_GRAPH.stats()}")
    print(f"  Entities: {KNOWLEDGE_GRAPH.get_all_entities()}")
    print(f"\n  Full subgraph from 'alice':")
    print(f"  {KNOWLEDGE_GRAPH.get_subgraph('alice', max_depth=3)}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 20: Context Graphs (LangGraph)             ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()
    run_demo()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. The knowledge graph PERSISTS across invocations — each conversation
       turn can add new entities and relations, building a richer context
       over time.  This is how agents develop long-term knowledge.

    2. Entity extraction (Node 1) turns unstructured text into structured
       graph updates.  The LLM does the heavy lifting; we just parse JSON.

    3. Graph traversal (Nodes 2-3) replaces vector search for relational
       queries.  The subgraph context is compact and precise — typically
       5-10x fewer tokens than equivalent RAG chunks.

    4. Subgraph expansion (Node 3) handles sparse graphs by going deeper
       or checking reverse edges.  This is analogous to "retrieval with
       follow-up" in agentic RAG.

    5. The pipeline is composable: you could add a vector store node
       between query_graph and generate for hybrid retrieval when the
       graph alone isn't sufficient.
    """))
