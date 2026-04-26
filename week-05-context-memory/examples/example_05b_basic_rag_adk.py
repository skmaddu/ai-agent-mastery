import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5b: Basic RAG Pipeline — Google ADK Implementation
============================================================
ADK agent with a knowledge base search tool for RAG.

In ADK, RAG is implemented by giving the agent a TOOL that searches
the knowledge base.  The agent decides when and how to use it.
Compare this with the LangGraph version (Example 5) where the
pipeline is hardcoded as graph edges.

Run: python week-05-context-memory/examples/example_05b_basic_rag_adk.py
"""

import os
import sys
import json
import math
import asyncio
import textwrap
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

# ── Phoenix ────────────────────────────────────────────────────
PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.otel import register
    PHOENIX_AVAILABLE = True
except ImportError:
    pass

def setup_phoenix():
    if not PHOENIX_AVAILABLE:
        return None
    try:
        session = px.launch_app(use_temp_dir=False)
        register(project_name="week5-basic-rag-adk")
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
# KNOWLEDGE BASE & SIMPLE VECTOR SEARCH
# ================================================================

KNOWLEDGE_BASE = [
    {"id": 0, "title": "AI Alignment Overview",
     "content": "AI alignment ensures AI systems act according to human values. Key approaches include RLHF, constitutional AI, and interpretability research. Misalignment ranges from minor annoyances to existential risks."},
    {"id": 1, "title": "Prompt Injection Attacks",
     "content": "Prompt injection causes LLMs to ignore original instructions. Direct injection embeds instructions in user input. Indirect injection hides instructions in retrieved data. Defenses: input sanitization, instruction hierarchy, output filtering."},
    {"id": 2, "title": "Red Teaming for AI",
     "content": "Red teaming probes AI systems for vulnerabilities before deployment. Techniques: adversarial prompting, jailbreak attempts, bias probes, automated attack generation. Best practice: continuous red-teaming, not just pre-launch."},
    {"id": 3, "title": "AI Safety Regulations",
     "content": "EU AI Act (2025) classifies systems by risk level. High-risk systems need accuracy, robustness, human oversight. US Executive Order requires frontier model reporting. Global trend: mandatory risk assessment and incident reporting."},
    {"id": 4, "title": "Interpretability Research",
     "content": "Interpretability aims to understand how neural networks make decisions. Techniques: attention visualization, SHAP, LIME, circuit analysis. Researchers can identify circuits for factual recall and language understanding."},
]


def _simple_embed(text):
    vocab = ["ai", "safety", "alignment", "model", "attack", "injection",
             "prompt", "risk", "human", "system", "learning", "data",
             "regulation", "red", "team", "research", "bias", "defense",
             "interpretability", "neural", "network", "rlhf", "feedback"]
    words = text.lower().split()
    emb = [float(words.count(v)) for v in vocab]
    mag = math.sqrt(sum(x * x for x in emb))
    return [x / mag for x in emb] if mag > 0 else emb


KB_EMBEDDINGS = {doc["id"]: _simple_embed(doc["content"]) for doc in KNOWLEDGE_BASE}


# ================================================================
# ADK TOOLS
# ================================================================
# In ADK, tools are plain Python functions with docstrings.
# The agent reads the docstring to understand when to use each tool.

def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the AI safety knowledge base for information relevant to a query.

    Use this tool whenever the user asks a question about AI safety,
    alignment, regulations, prompt injection, red teaming, or interpretability.
    Always search before answering to ensure accuracy.

    Args:
        query: The search query describing what information you need.
        top_k: Number of results to return (default 3).

    Returns:
        JSON string with the top matching documents and their relevance scores.
    """
    query_emb = _simple_embed(query)

    scored = []
    for doc in KNOWLEDGE_BASE:
        score = sum(a * b for a, b in zip(query_emb, KB_EMBEDDINGS[doc["id"]]))
        scored.append({"title": doc["title"], "content": doc["content"],
                       "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    print(f"  [TOOL] search_knowledge_base('{query[:50]}...') → {len(results)} results")
    for r in results:
        print(f"    [{r['score']:.3f}] {r['title']}")

    return json.dumps({"results": results, "query": query})


# ================================================================
# AGENT DEFINITION
# ================================================================

def build_rag_agent():
    """
    Build an ADK RAG agent.

    KEY DESIGN CHOICE: The agent's instruction tells it to ALWAYS
    search before answering.  Without this, the agent might try to
    answer from its training data, bypassing RAG entirely.
    """
    agent = LlmAgent(
        name="rag_safety_expert",
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        instruction="""You are an AI safety expert. You MUST use the search_knowledge_base tool
to find relevant information before answering any question.

Rules:
1. ALWAYS search the knowledge base before answering.
2. Base your answer ONLY on the search results, not your training data.
3. Cite sources using [Source: title] format.
4. If the knowledge base doesn't contain relevant information, say so clearly.
5. Be concise — aim for 2-3 sentences with citations.""",
        tools=[search_knowledge_base],
        description="AI safety expert with knowledge base access.",
    )
    return agent


# ================================================================
# RUNNER
# ================================================================

async def run_rag_agent(agent, query: str, retries: int = 5) -> str:
    """Run a single RAG query through the ADK agent with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="week5_rag_adk",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="week5_rag_adk",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=query)],
                ),
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        result_text = event.content.parts[0].text

            return result_text
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
    """Run the RAG demo with several queries."""
    if not ADK_AVAILABLE:
        print("[SKIP] ADK not available.")
        return

    agent = build_rag_agent()

    queries = [
        "What are the main approaches to AI alignment?",
        "How can I defend against prompt injection attacks?",
        "What regulations govern AI systems in Europe?",
    ]

    print("\n" + "=" * 65)
    print("  BASIC RAG — GOOGLE ADK AGENT")
    print("=" * 65)

    for i, query in enumerate(queries):
        print(f"\n{'━' * 65}")
        print(f"  Query {i + 1}: {query}")
        print(f"{'━' * 65}")

        answer = await run_rag_agent(agent, query)
        print(f"\n  Answer: {answer[:300]}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 5b: Basic RAG Pipeline (Google ADK)        ║")
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
    1. ADK RAG = agent + search tool.  The agent decides WHEN to search,
       unlike LangGraph where the pipeline is hardcoded.

    2. The agent instruction MUST say "always search before answering" —
       otherwise it may skip retrieval and hallucinate.

    3. Return search results as JSON for structured parsing by the agent.

    4. LangGraph vs ADK for RAG:
       - LangGraph: Deterministic pipeline, every query goes through
         embed → retrieve → generate.  Predictable, testable.
       - ADK: Agent-driven, may skip search for simple queries.
         More flexible, less predictable.

    5. Both approaches benefit from Phoenix tracing to see tool calls,
       retrieval scores, and generation quality.
    """))
