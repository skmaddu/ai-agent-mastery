import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 20b: Context Graphs — Google ADK Implementation
=========================================================
ADK agent with tools to build and query a knowledge graph.

Tools:
  - add_to_knowledge_graph: Add entities and relations
  - query_knowledge_graph: Get subgraph context for an entity
  - get_entity_neighbors: List direct connections of an entity

The agent decides WHEN to add knowledge and WHEN to query it,
making the graph a natural part of its reasoning loop.

Run: python week-05-context-memory/examples/example_20b_context_graphs_adk.py
"""

import os
import sys
import json
import asyncio
import textwrap
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Set, Any
from collections import deque

load_dotenv("config/.env")
load_dotenv()

# ── Phoenix ────────────────────────────────────────────────────
try:
    import phoenix as px
    from phoenix.otel import register
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        register(project_name="week5-context-graphs-adk")
        print("[Phoenix] Dashboard: http://localhost:6006")
        return session
    except Exception:
        return None

# ── ADK ────────────────────────────────────────────────────────
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("[ADK] Not installed. pip install google-adk")


# ================================================================
# PERSISTENT KNOWLEDGE GRAPH
# ================================================================
# Module-level graph that persists across conversation turns.
# The ADK agent interacts with this graph via tool functions.
# In production, you'd replace this with Neo4j or a graph database.

class KnowledgeGraph:
    """In-memory knowledge graph with adjacency list storage."""

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
        if (relation, target) not in self.adjacency[source]:
            self.adjacency[source].append((relation, target))

    def get_subgraph(self, entity_id: str, max_depth: int = 2) -> str:
        """BFS traversal returning compact text context."""
        entity_id = entity_id.lower().strip()
        if entity_id not in self.adjacency:
            return f"No information about '{entity_id}' in the knowledge graph."

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

        return "\n".join(lines) if len(lines) > 1 else f"Entity '{entity_id}' exists but has no connections."

    def get_neighbors(self, entity_id: str) -> List[Dict[str, str]]:
        """Direct connections as a list of dicts."""
        entity_id = entity_id.lower().strip()
        if entity_id not in self.adjacency:
            return []
        return [{"relation": r, "target": t} for r, t in self.adjacency[entity_id]]

    def get_all_entities(self) -> List[str]:
        return list(self.adjacency.keys())

    def stats(self) -> Dict[str, int]:
        total_edges = sum(len(edges) for edges in self.adjacency.values())
        return {"entities": len(self.adjacency), "edges": total_edges}


# Module-level graph — persists across all conversation turns
KNOWLEDGE_GRAPH = KnowledgeGraph()


# ================================================================
# ADK TOOL FUNCTIONS
# ================================================================
# These tools let the agent interact with the knowledge graph.
# The agent decides when to call each tool based on its instruction.
# ADK tools are plain functions (not decorated like LangGraph @tool).

def add_to_knowledge_graph(source_entity: str, source_type: str,
                           relation: str, target_entity: str,
                           target_type: str) -> str:
    """
    Add a relationship to the knowledge graph.

    Use this when the user shares new information about entities
    and their relationships (e.g., "Alice manages the ML team",
    "The project uses PyTorch").

    Args:
        source_entity: The source entity name (e.g., "alice").
        source_type: Type of source (person, team, project, technology, concept).
        relation: The relationship (e.g., "manages", "works_on", "uses").
        target_entity: The target entity name (e.g., "ml_team").
        target_type: Type of target (person, team, project, technology, concept).

    Returns:
        Confirmation message with updated graph stats.
    """
    KNOWLEDGE_GRAPH.add_entity(source_entity, source_type)
    KNOWLEDGE_GRAPH.add_entity(target_entity, target_type)
    KNOWLEDGE_GRAPH.add_relation(source_entity, relation, target_entity)
    stats = KNOWLEDGE_GRAPH.stats()
    return (f"Added: ({source_entity}) --[{relation}]--> ({target_entity}). "
            f"Graph now has {stats['entities']} entities, {stats['edges']} edges.")


def query_knowledge_graph(entity: str, max_depth: str = "2") -> str:
    """
    Query the knowledge graph for information about an entity.

    Use this when the user asks about an entity, its relationships,
    or anything that requires looking up stored knowledge.

    Args:
        entity: The entity to look up (e.g., "alice", "ml_team").
        max_depth: How many hops to traverse (1-3). Default "2".

    Returns:
        Text description of the entity's subgraph (relationships and connections).
    """
    depth = min(int(max_depth), 3)
    subgraph = KNOWLEDGE_GRAPH.get_subgraph(entity, max_depth=depth)
    stats = KNOWLEDGE_GRAPH.stats()
    return f"{subgraph}\n\n[Graph: {stats['entities']} entities, {stats['edges']} edges]"


def get_entity_neighbors(entity: str) -> str:
    """
    Get the direct connections (neighbors) of an entity.

    Use this for quick lookups when you need to know what an entity
    is directly connected to, without deep traversal.

    Args:
        entity: The entity to check (e.g., "bob", "search_engine").

    Returns:
        JSON list of direct connections with relation and target.
    """
    neighbors = KNOWLEDGE_GRAPH.get_neighbors(entity)
    if not neighbors:
        available = KNOWLEDGE_GRAPH.get_all_entities()
        return (f"No entity '{entity}' found. "
                f"Available entities: {', '.join(available[:15]) if available else 'none'}")
    return json.dumps(neighbors, indent=2)


# ================================================================
# AGENT CONSTRUCTION
# ================================================================

def build_graph_agent():
    """
    Build an ADK agent with knowledge graph tools.

    The instruction tells the agent HOW and WHEN to use each tool:
    - add_to_knowledge_graph: when user shares new facts
    - query_knowledge_graph: when user asks about entities
    - get_entity_neighbors: for quick direct-connection checks
    """
    instruction = textwrap.dedent("""
        You are a knowledge assistant powered by a context graph.

        KNOWLEDGE GRAPH PROTOCOL:
        1. When the user shares NEW information (facts, relationships),
           use add_to_knowledge_graph to store it. Extract all entities
           and relationships from their message.
        2. When the user ASKS a question, use query_knowledge_graph to
           look up relevant entities before answering.
        3. Use get_entity_neighbors for quick checks on direct connections.
        4. Always reference the graph data in your answers — show the user
           what relationships you found.
        5. If the graph doesn't have enough info, say so honestly.

        ENTITY NAMING: Use lowercase, underscores for spaces (e.g., "ml_team").
        RELATION NAMING: Use active verbs (manages, works_on, uses, reports_to).

        Be helpful, concise, and always ground your answers in graph data.
    """).strip()

    return LlmAgent(
        name="graph_knowledge_agent",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction=instruction,
        tools=[add_to_knowledge_graph, query_knowledge_graph, get_entity_neighbors],
    )


# ================================================================
# RUNNER
# ================================================================

async def run_turn(runner: Runner, session_id: str, message: str,
                   retries: int = 5) -> str:
    """Run one conversation turn with retry logic for Gemini API errors."""
    for attempt in range(1, retries + 1):
        try:
            result = ""
            async for event in runner.run_async(
                user_id="user1",
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    result = event.content.parts[0].text
            return result
        except Exception as e:
            if attempt < retries:
                wait = attempt * 10
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [ERROR] All {retries} attempts failed: {e}")
                return f"[Error: API temporarily unavailable after {retries} retries]"

    return "[Error: unexpected]"


async def run_demo():
    if not ADK_AVAILABLE:
        print("[SKIP] ADK not available.")
        return

    # Seed the knowledge graph with initial facts
    seed_facts = [
        ("alice", "person", "manages", "ml_team", "team"),
        ("bob", "person", "member_of", "ml_team", "team"),
        ("carol", "person", "leads", "backend_team", "team"),
        ("ml_team", "team", "works_on", "search_engine", "project"),
        ("search_engine", "project", "uses", "pytorch", "technology"),
        ("alice", "person", "reports_to", "diana", "person"),
    ]
    for src, stype, rel, tgt, ttype in seed_facts:
        KNOWLEDGE_GRAPH.add_entity(src, stype)
        KNOWLEDGE_GRAPH.add_entity(tgt, ttype)
        KNOWLEDGE_GRAPH.add_relation(src, rel, tgt)

    print(f"\n  Seeded graph: {KNOWLEDGE_GRAPH.stats()}")

    agent = build_graph_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="week5_graph_adk",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="week5_graph_adk", user_id="user1",
    )

    # Multi-turn conversation — the graph grows with each turn
    turns = [
        "What do you know about Alice and her team?",
        "Bob is now also working on a new chatbot project that uses langchain.",
        "What projects is the ML team involved in?",
        "Carol's backend team has started using fastapi for the api_platform project.",
        "Give me a summary of all teams and their projects.",
    ]

    print("\n" + "=" * 65)
    print("  ADK GRAPH-AUGMENTED CONVERSATION")
    print("=" * 65)

    for i, turn in enumerate(turns):
        print(f"\n{'━' * 65}")
        print(f"  Turn {i + 1}: {turn}")
        print(f"{'━' * 65}")
        response = await run_turn(runner, session.id, turn)
        print(f"  Agent: {response[:300]}")
        print(f"  Graph: {KNOWLEDGE_GRAPH.stats()}")

    # Final graph state
    print(f"\n{'=' * 65}")
    print(f"  FINAL KNOWLEDGE GRAPH STATE")
    print(f"{'=' * 65}")
    print(f"  Stats: {KNOWLEDGE_GRAPH.stats()}")
    print(f"  Entities: {KNOWLEDGE_GRAPH.get_all_entities()}")
    print(f"\n  Full context for 'alice':")
    print(f"  {KNOWLEDGE_GRAPH.get_subgraph('alice', max_depth=3)}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 20b: Context Graphs (ADK)                  ║")
    print("╚" + "═" * 63 + "╝")

    setup_phoenix()

    if ADK_AVAILABLE:
        asyncio.run(run_demo())
    else:
        print("\n  Install Google ADK: pip install google-adk")

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. ADK agents interact with knowledge graphs through TOOLS, giving
       the agent autonomy to decide when to store vs. query knowledge.
       The instruction's PROTOCOL section guides this decision.

    2. The knowledge graph persists at module level across turns, so
       each conversation enriches the shared knowledge base.  In
       production, back this with Neo4j or a persistent store.

    3. Tool design matters: add_to_knowledge_graph takes structured
       arguments (source, type, relation, target) so the graph stays
       clean.  Avoid letting the agent dump raw text into the graph.

    4. get_entity_neighbors is a lightweight alternative to full
       subgraph queries — use it when the agent just needs to check
       direct connections without deep traversal.

    5. Both LangGraph (pipeline nodes) and ADK (agent tools) achieve
       the same graph-augmented generation.  LangGraph controls the
       flow explicitly; ADK lets the agent decide the flow via tools.
    """))
