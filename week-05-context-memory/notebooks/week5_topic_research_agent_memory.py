import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Week 5 Integrated Project: Topic Research Agent — RAG + Memory + Context Graphs
=================================================================================
This is the Week 5 evolution of the ongoing Topic Research Agent project.

New capabilities added this week:
  1. RAG Pipeline: Agent retrieves from a local knowledge base before answering
  2. Persistent Memory: Agent remembers past research across sessions (JSON)
  3. Context Graph: Agent builds a knowledge graph of entities & relationships
  4. Context Management: Auto-summarization when conversation grows

Architecture (LangGraph):
  START → load_memory → check_cache → [cache hit?]
                                        ├─ YES → format_cached → END
                                        └─ NO  → retrieve_rag → check_quality
                                                                    |
                                                      [quality OK?]
                                                        ├─ YES → build_graph → generate → save_memory → END
                                                        └─ NO  → refine → retrieve_rag (max 2 retries)

Both LangGraph and ADK implementations included.

Run: python week-05-context-memory/notebooks/week5_topic_research_agent_memory.py
"""

import os
import sys
import json
import math
import hashlib
import textwrap
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
        print("[Phoenix] Not available — install phoenix for tracing.")
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        tracer_provider = register(project_name="week5-research-agent")
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
# PART 1: KNOWLEDGE BASE (simulated domain-specific corpus)
# ================================================================

RESEARCH_KB = [
    {"id": 0, "title": "AI Agent Architectures",
     "content": "Modern AI agents use three primary architectures: ReAct (Reason+Act), Plan-Execute, and Multi-Agent Supervisor. ReAct interleaves thinking and action steps. Plan-Execute creates a full plan then executes steps sequentially. Multi-Agent uses specialized sub-agents coordinated by a supervisor."},
    {"id": 1, "title": "Context Engineering Fundamentals",
     "content": "Context engineering treats the LLM context window as managed RAM. The four pillars are Write (what enters), Select (what stays), Compress (how to shrink), and Isolate (how to separate). The lost-in-the-middle problem means critical info should be at the start or end."},
    {"id": 2, "title": "RAG Pipeline Best Practices",
     "content": "Production RAG uses hybrid search combining dense embeddings with BM25 keyword matching. Reciprocal Rank Fusion merges results. Cross-encoder reranking improves precision by 14%. Chunk sizes of 200-500 tokens with recursive splitting work best."},
    {"id": 3, "title": "Memory Systems for AI Agents",
     "content": "Agent memory has three types: episodic (events), semantic (facts), and procedural (skills). Hierarchical memory uses L1 cache (recent), L2 summary (compressed), and L3 archive (permanent facts). Importance scoring determines what to keep vs forget."},
    {"id": 4, "title": "AI Safety and Alignment",
     "content": "AI safety encompasses alignment (RLHF, DPO, Constitutional AI), red-teaming (adversarial testing), and defensive layers (input sanitization, output filtering). A 6-layer safety stack protects production agents from prompt injection and misuse."},
    {"id": 5, "title": "Knowledge Graphs for Agents",
     "content": "Knowledge graphs store entities and relationships as (subject, predicate, object) triples. Temporal knowledge graphs add time awareness. Graph-based retrieval enables multi-hop reasoning in fewer tokens than naive RAG by traversing entity connections."},
    {"id": 6, "title": "LangGraph Framework",
     "content": "LangGraph uses StateGraph with TypedDict state and conditional edges. MemorySaver enables checkpointing for cross-session persistence. Reducers like add_messages control how state updates merge. Subgraphs enable modular agent composition."},
    {"id": 7, "title": "Google ADK Framework",
     "content": "Google ADK uses LlmAgent with declarative configuration. Agents define tools as plain functions. Runner manages execution with InMemorySessionService for session state. ADK supports YAML configuration and callback-based memory handling."},
]


# ================================================================
# PART 2: EMBEDDING & RETRIEVAL
# ================================================================

VOCAB = ["ai", "agent", "memory", "context", "rag", "retrieval", "graph",
         "knowledge", "safety", "alignment", "langgraph", "adk", "tool",
         "search", "vector", "embedding", "model", "plan", "state",
         "token", "prompt", "architecture", "framework", "pipeline"]


def embed(text: str) -> List[float]:
    words = text.lower().split()
    e = [float(words.count(v)) for v in VOCAB]
    m = math.sqrt(sum(x * x for x in e))
    return [x / m for x in e] if m > 0 else e


KB_EMBEDDINGS = {doc["id"]: embed(doc["content"]) for doc in RESEARCH_KB}


def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    q = embed(query)
    scored = []
    for doc in RESEARCH_KB:
        score = sum(a * b for a, b in zip(q, KB_EMBEDDINGS[doc["id"]]))
        scored.append({**doc, "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ================================================================
# PART 3: PERSISTENT MEMORY (JSON file)
# ================================================================

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "research_memory.json")


def load_memory() -> Dict:
    try:
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"query_cache": {}, "facts": [], "graph": {}}


def save_memory(data: Dict):
    os.makedirs(os.path.dirname(MEMORY_PATH) or ".", exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]


# ================================================================
# PART 4: LANGGRAPH STATE
# ================================================================

class ResearchState(TypedDict):
    topic: str                         # Research topic
    query: str                         # Current query
    cache_hit: bool                    # Whether we found a cached answer
    cached_answer: str                 # The cached answer (if hit)
    retrieved_docs: List[Dict]         # RAG results
    quality_score: float               # Retrieval quality
    retries: int                       # Retry count
    max_retries: int                   # Max retries
    graph_entities: List[Dict]         # New entities for the knowledge graph
    context: str                       # Assembled context
    answer: str                        # Final answer
    sources: List[str]                 # Citations
    research_summary: str              # Running summary of all research


# ================================================================
# PART 5: GRAPH NODES
# ================================================================

research_llm = get_llm(temperature=0.3)


def load_memory_node(state: ResearchState) -> dict:
    """Load persistent memory and check for cached results."""
    memory = load_memory()
    qhash = get_query_hash(state["query"])

    if qhash in memory.get("query_cache", {}):
        cached = memory["query_cache"][qhash]
        print(f"  [CACHE] Hit for '{state['query'][:40]}...'")
        return {
            "cache_hit": True,
            "cached_answer": cached.get("answer", ""),
            "sources": cached.get("sources", []),
        }

    print(f"  [CACHE] Miss for '{state['query'][:40]}...'")
    return {"cache_hit": False}


def should_use_cache(state: ResearchState) -> str:
    return "format_cached" if state.get("cache_hit") else "retrieve"


def format_cached_node(state: ResearchState) -> dict:
    """Format a cached answer."""
    print(f"  [CACHED] Using cached answer")
    return {"answer": f"[From cache] {state['cached_answer']}"}


def retrieve_node(state: ResearchState) -> dict:
    """Retrieve relevant documents from the knowledge base."""
    docs = retrieve(state["query"], top_k=3)
    quality = max(d["score"] for d in docs) if docs else 0

    print(f"  [RETRIEVE] Top docs:")
    for d in docs:
        print(f"    [{d['score']:.3f}] {d['title']}")

    return {
        "retrieved_docs": docs,
        "quality_score": quality,
    }


def check_quality(state: ResearchState) -> str:
    """Route based on retrieval quality."""
    if state["quality_score"] >= 0.25 or state["retries"] >= state["max_retries"]:
        return "build_graph"
    print(f"  [QUALITY] Score {state['quality_score']:.3f} too low — refining")
    return "refine"


def refine_node(state: ResearchState) -> dict:
    return {
        "query": state["query"] + " details explanation overview",
        "retries": state.get("retries", 0) + 1,
    }


def build_graph_node(state: ResearchState) -> dict:
    """Extract entities and update the persistent knowledge graph."""
    memory = load_memory()
    graph = memory.get("graph", {})

    new_entities = []
    for doc in state.get("retrieved_docs", []):
        entity = doc["title"]
        # Extract keywords as related entities
        words = doc["content"].split()
        keywords = [w.strip(".,;:!?()") for w in words
                    if len(w) > 5 and w[0].isupper()][:5]

        for kw in keywords:
            if kw != entity:
                if entity not in graph:
                    graph[entity] = []
                graph[entity].append({
                    "relation": "relates_to",
                    "target": kw,
                    "source": doc["title"],
                })
                new_entities.append({"entity": entity, "related": kw})

    memory["graph"] = graph
    save_memory(memory)

    print(f"  [GRAPH] Added {len(new_entities)} entity relationships")
    return {"graph_entities": new_entities}


def generate_node(state: ResearchState) -> dict:
    """Generate research answer with full context."""
    docs = state.get("retrieved_docs", [])
    context_parts = [f"[Source: {d['title']}]\n{d['content']}" for d in docs]

    # Add graph context if available
    memory = load_memory()
    graph = memory.get("graph", {})
    graph_context = []
    for entity, relations in graph.items():
        if entity.lower() in state["query"].lower() or \
           any(entity.lower() in d["title"].lower() for d in docs):
            for rel in relations[:3]:
                graph_context.append(f"{entity} → {rel['relation']} → {rel['target']}")

    if graph_context:
        context_parts.append("Knowledge graph connections:\n" +
                           "\n".join(graph_context[:10]))

    context = "\n\n".join(context_parts)
    sources = [d["title"] for d in docs]

    prompt = [
        SystemMessage(content=(
            "You are an expert AI research analyst. Provide a comprehensive "
            "analysis of the topic based on the provided context. Structure your "
            "response with: 1) Overview, 2) Key findings, 3) Implications. "
            "Cite sources with [Source: title] format. Be thorough but concise."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nResearch topic: {state['topic']}\nQuery: {state['query']}"),
    ]

    try:
        response = research_llm.invoke(prompt)
        answer = response.content.strip()

        if cost_tracker:
            usage = response.response_metadata.get("token_usage", {})
            inp = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            out = usage.get("completion_tokens", usage.get("output_tokens", 0))
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            cost_tracker.log_call(model, inp, out)
    except Exception as e:
        answer = f"[Error: {e}]"

    print(f"  [GENERATE] Answer: {answer[:120]}...")
    return {"answer": answer, "sources": sources, "context": context}


def save_memory_node(state: ResearchState) -> dict:
    """Save results to persistent memory for future cache hits."""
    memory = load_memory()

    # Cache the query result
    qhash = get_query_hash(state["query"])
    memory["query_cache"][qhash] = {
        "answer": state["answer"],
        "sources": state["sources"],
        "timestamp": datetime.now().isoformat(),
        "topic": state["topic"],
    }

    # Update research facts
    fact = f"Researched '{state['topic']}': {state['answer'][:100]}..."
    if fact not in memory.get("facts", []):
        memory.setdefault("facts", []).append(fact)

    save_memory(memory)
    print(f"  [MEMORY] Saved to cache and facts")
    return {}


# ================================================================
# PART 6: GRAPH CONSTRUCTION
# ================================================================

def build_research_graph():
    """Build the full research agent graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("format_cached", format_cached_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("refine", refine_node)
    graph.add_node("build_graph", build_graph_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")

    graph.add_conditional_edges("load_memory", should_use_cache, {
        "format_cached": "format_cached",
        "retrieve": "retrieve",
    })

    graph.add_edge("format_cached", END)

    graph.add_conditional_edges("retrieve", check_quality, {
        "build_graph": "build_graph",
        "refine": "refine",
    })

    graph.add_edge("refine", "retrieve")
    graph.add_edge("build_graph", "generate")
    graph.add_edge("generate", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# ================================================================
# PART 7: DEMO — Multi-query research session
# ================================================================

def run_research_session():
    """Run a multi-query research session demonstrating all features."""

    # Clean up previous memory for demo
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)

    app = build_research_graph()

    research_queries = [
        ("AI Agent Architectures", "What are the main AI agent architectures and how do they compare?"),
        ("RAG Best Practices", "What are the best practices for building a production RAG pipeline?"),
        ("AI Agent Memory", "How do AI agents implement memory systems?"),
        ("AI Agent Architectures", "What are the main AI agent architectures and how do they compare?"),  # Cache hit!
        ("Knowledge Graphs", "How do knowledge graphs help AI agents with reasoning?"),
    ]

    print("\n" + "=" * 70)
    print("  TOPIC RESEARCH AGENT — WEEK 5 BUILD")
    print("  RAG + Persistent Memory + Context Graphs")
    print("=" * 70)

    for i, (topic, query) in enumerate(research_queries):
        print(f"\n{'━' * 70}")
        print(f"  Research {i + 1}/{len(research_queries)}: {topic}")
        print(f"  Query: {query}")
        print(f"{'━' * 70}")

        result = app.invoke(
            {
                "topic": topic, "query": query,
                "cache_hit": False, "cached_answer": "",
                "retrieved_docs": [], "quality_score": 0.0,
                "retries": 0, "max_retries": 2,
                "graph_entities": [],
                "context": "", "answer": "", "sources": [],
                "research_summary": "",
            },
            {"run_name": f"research-{i + 1}-{topic[:20]}"},
        )

        print(f"\n  {'─' * 60}")
        print(f"  Answer: {result['answer'][:400]}")
        print(f"  Sources: {result.get('sources', [])}")
        if result.get("cache_hit"):
            print(f"  (Served from cache)")

    # Final memory report
    memory = load_memory()
    print(f"\n{'=' * 70}")
    print(f"  SESSION REPORT")
    print(f"{'=' * 70}")
    print(f"  Queries processed: {len(research_queries)}")
    print(f"  Cached results: {len(memory.get('query_cache', {}))}")
    print(f"  Research facts: {len(memory.get('facts', []))}")
    print(f"  Graph entities: {len(memory.get('graph', {}))}")

    if memory.get("graph"):
        print(f"\n  Knowledge Graph Sample:")
        for entity, relations in list(memory["graph"].items())[:3]:
            targets = [r["target"] for r in relations[:3]]
            print(f"    {entity} → {', '.join(targets)}")

    if cost_tracker:
        print()
        cost_tracker.report()

    # Cleanup
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)


# ================================================================
# PART 8: ADK VERSION (simplified)
# ================================================================

async def run_adk_version():
    """Simplified ADK version of the research agent."""
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types
    except ImportError:
        print("\n  [ADK] Not installed — skipping ADK version.")
        return

    # ADK tools reuse the same retrieval and memory functions
    def search_knowledge_base(query: str) -> str:
        """Search the AI research knowledge base."""
        results = retrieve(query, top_k=3)
        return json.dumps({"results": [{"title": r["title"], "content": r["content"],
                                         "score": r["score"]} for r in results]})

    def remember_research(topic: str, finding: str) -> str:
        """Save a research finding to persistent memory."""
        memory = load_memory()
        memory.setdefault("facts", []).append(f"{topic}: {finding[:100]}")
        save_memory(memory)
        return f"Saved finding for '{topic}'"

    def recall_research(topic: str = "") -> str:
        """Recall past research findings from memory."""
        memory = load_memory()
        facts = memory.get("facts", [])
        if topic:
            facts = [f for f in facts if topic.lower() in f.lower()]
        return json.dumps({"findings": facts, "total": len(facts)})

    agent = LlmAgent(
        name="research_agent_adk",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are an AI research analyst. For every question:
1. Search the knowledge base using search_knowledge_base.
2. Check for past research using recall_research.
3. Provide a structured answer with: Overview, Key Findings, Implications.
4. Save important findings using remember_research.
5. Always cite sources with [Source: title].""",
        tools=[search_knowledge_base, remember_research, recall_research],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="week5_research_adk",
                    session_service=session_service)

    print("\n" + "=" * 70)
    print("  ADK VERSION — Topic Research Agent")
    print("=" * 70)

    queries = [
        "What are the main AI agent architectures?",
        "How do knowledge graphs help AI agents?",
    ]

    for query in queries:
        session = await session_service.create_session(
            app_name="week5_research_adk", user_id="researcher")

        print(f"\n  Query: {query}")
        async for event in runner.run_async(
            user_id="researcher", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                print(f"  Answer: {event.content.parts[0].text[:300]}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  WEEK 5 PROJECT: Topic Research Agent — Memory + RAG + Graphs       ║")
    print("╚" + "═" * 68 + "╝")

    setup_phoenix()

    # Run LangGraph version
    run_research_session()

    # Run ADK version
    import asyncio
    asyncio.run(run_adk_version())

    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS — WEEK 5 PROJECT")
    print("=" * 70)
    print(textwrap.dedent("""
    This week's project evolution added three major capabilities:

    1. RAG PIPELINE: The agent retrieves from a knowledge base before
       answering, grounding responses in real data instead of relying
       solely on LLM training data.

    2. PERSISTENT MEMORY: Query results are cached to JSON, enabling
       instant responses for repeated queries and cross-session context.
       Research facts accumulate over time.

    3. CONTEXT GRAPH: Entity relationships are extracted from retrieved
       documents and stored as a graph. This enables richer context
       generation through entity connections.

    4. COST TRACKING: Every LLM call is tracked for token usage and
       cost, keeping the agent within the weekly budget.

    Next week (Week 6): Production-grade observability, streaming,
    and Phoenix evaluation pipelines.
    """))
