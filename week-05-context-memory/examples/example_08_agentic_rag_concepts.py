import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 8: Agentic RAG — Dynamic Retrieval with Tools (Concepts)
=================================================================
Pure-Python concept demonstration (no LLM calls required).

Covers:
  1. Self-RAG / Adaptive Retrieval Loops
  2. Agent Decides When/How/What to Retrieve
  3. Plan → Retrieve → Critique → Re-retrieve Pattern
  4. Reflection & Verification in Retrieval

Agentic RAG is the evolution from "pipeline RAG" (fixed retrieve→generate)
to "agent-driven RAG" where the agent DECIDES:
  • WHETHER to retrieve (maybe it already knows the answer)
  • WHAT to retrieve (which knowledge base? what query?)
  • HOW MUCH to retrieve (1 chunk or 10?)
  • WHETHER the results are good enough (self-critique)
  • WHETHER to re-retrieve with a different strategy

This is the CRAG (Corrective RAG) and Self-RAG pattern.

Run: python week-05-context-memory/examples/example_08_agentic_rag_concepts.py
"""

import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ================================================================
# 1. SELF-RAG / ADAPTIVE RETRIEVAL LOOPS
# ================================================================
# Self-RAG (Asai et al., 2023) introduced the idea that the model
# itself should decide when retrieval is helpful.  For factual
# questions, retrieval is critical; for creative writing, it's
# unnecessary.  Wasting retrieval on easy questions adds latency
# and cost without improving quality.
#
# The Self-RAG loop:
#   1. CLASSIFY: Does this query need retrieval? (yes/no)
#   2. RETRIEVE: If yes, fetch relevant documents
#   3. GENERATE: Produce an answer using retrieved context
#   4. CRITIQUE: Is the answer grounded in the retrieved docs?
#   5. LOOP: If not grounded, re-retrieve with a refined query

class RetrievalDecision(Enum):
    NO_RETRIEVAL = "no_retrieval"       # Answer from model knowledge
    SINGLE_SOURCE = "single_source"     # One knowledge base is sufficient
    MULTI_SOURCE = "multi_source"       # Need multiple knowledge bases
    WEB_SEARCH = "web_search"           # Need real-time web information


@dataclass
class AgenticRAGStep:
    """One step in an agentic RAG execution trace."""
    action: str
    input_data: str
    output_data: str
    decision: str
    tokens_used: int = 0


def classify_retrieval_need(query: str) -> RetrievalDecision:
    """
    Simulate the agent's decision about whether to retrieve.

    In production, this is an LLM call with a classification prompt:
        "Given this query, should I: (a) answer from knowledge,
         (b) search one source, (c) search multiple sources,
         (d) search the web?"

    The classifier considers:
      - Query type (factual vs. creative vs. procedural)
      - Temporal sensitivity (needs current data?)
      - Domain specificity (requires specialized knowledge?)
    """
    query_lower = query.lower()

    # Temporal queries need web search
    if any(word in query_lower for word in ["latest", "today", "current", "2026", "news"]):
        return RetrievalDecision.WEB_SEARCH

    # Comparison queries need multiple sources
    if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
        return RetrievalDecision.MULTI_SOURCE

    # Factual questions need single source
    if any(word in query_lower for word in ["what is", "how does", "explain", "define"]):
        return RetrievalDecision.SINGLE_SOURCE

    # Creative or simple queries don't need retrieval
    return RetrievalDecision.NO_RETRIEVAL


def demo_adaptive_retrieval():
    """Show how the agent classifies retrieval needs."""

    print("=" * 65)
    print("  SELF-RAG: ADAPTIVE RETRIEVAL DECISIONS")
    print("=" * 65)

    queries = [
        "What is RLHF?",
        "Write a haiku about programming",
        "Compare RLHF and DPO approaches",
        "What are the latest AI safety regulations in 2026?",
        "How does constitutional AI work?",
        "Tell me a joke about machine learning",
    ]

    print(f"\n  {'Query':<45} {'Decision':<20}")
    print(f"  {'─' * 45} {'─' * 20}")

    for query in queries:
        decision = classify_retrieval_need(query)
        print(f"  {query:<45} {decision.value}")

    print(f"\n  The agent saves cost and latency by skipping retrieval")
    print(f"  when it's not needed (creative tasks, simple questions).")


# ================================================================
# 2. AGENT DECIDES WHEN/HOW/WHAT TO RETRIEVE
# ================================================================
# Beyond the binary "retrieve or not" decision, an agentic RAG
# system routes queries to different knowledge bases based on the
# topic.  This is QUERY ROUTING.

@dataclass
class KnowledgeBase:
    """Represents a searchable knowledge base."""
    name: str
    description: str
    domain: str
    doc_count: int
    avg_latency_ms: int


KNOWLEDGE_BASES = [
    KnowledgeBase("internal_docs", "Company internal documentation", "engineering", 50000, 50),
    KnowledgeBase("research_papers", "Academic AI research papers", "research", 200000, 120),
    KnowledgeBase("legal_regulatory", "AI regulations and legal docs", "legal", 15000, 80),
    KnowledgeBase("customer_support", "Customer tickets and FAQs", "support", 100000, 40),
    KnowledgeBase("web_search", "Real-time web search (Tavily/Brave)", "general", -1, 500),
]


def route_query(query: str) -> List[KnowledgeBase]:
    """
    Route a query to the most appropriate knowledge base(s).

    In production, this is an LLM call or a lightweight classifier:
        "Given these knowledge bases: [...], which should I search
         for the query: [query]? Return 1-3 selections."
    """
    query_lower = query.lower()
    selected = []

    # Rule-based routing (production: use LLM or trained classifier)
    if any(w in query_lower for w in ["regulation", "legal", "eu ai act", "compliance"]):
        selected.append(KNOWLEDGE_BASES[2])  # legal
    if any(w in query_lower for w in ["rlhf", "dpo", "alignment", "research", "paper"]):
        selected.append(KNOWLEDGE_BASES[1])  # research
    if any(w in query_lower for w in ["how to", "setup", "configure", "deploy"]):
        selected.append(KNOWLEDGE_BASES[0])  # internal
    if any(w in query_lower for w in ["customer", "ticket", "issue", "bug"]):
        selected.append(KNOWLEDGE_BASES[3])  # support
    if any(w in query_lower for w in ["latest", "current", "news", "today"]):
        selected.append(KNOWLEDGE_BASES[4])  # web

    if not selected:
        selected.append(KNOWLEDGE_BASES[0])  # default to internal docs

    return selected


def demo_query_routing():
    """Show query routing to different knowledge bases."""

    print("\n" + "=" * 65)
    print("  QUERY ROUTING: Agent Decides WHERE to Search")
    print("=" * 65)

    queries = [
        "What does the EU AI Act require for high-risk systems?",
        "How to deploy our AI model with Docker?",
        "Latest research on RLHF vs DPO comparison",
        "Customer reported a bug in the chatbot",
    ]

    for query in queries:
        targets = route_query(query)
        print(f"\n  Query: {query}")
        for kb in targets:
            print(f"    → {kb.name} ({kb.domain}, ~{kb.avg_latency_ms}ms)")


# ================================================================
# 3. PLAN → RETRIEVE → CRITIQUE → RE-RETRIEVE PATTERN
# ================================================================
# The most sophisticated agentic RAG pattern follows this loop:
#
#   PLAN:     Decompose the query into information needs
#   RETRIEVE: Fetch documents for each need
#   CRITIQUE: Evaluate whether retrieved docs answer the query
#   RE-RETRIEVE: If not, refine the query and try again
#
# This is CRAG (Corrective RAG) — the agent self-corrects its
# retrieval strategy based on the quality of results.

@dataclass
class RAGPlanStep:
    """One step in a retrieval plan."""
    information_need: str
    target_source: str
    query: str
    status: str = "pending"         # pending, retrieved, verified, failed
    retrieved_docs: List[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class CRAGTrace:
    """Full execution trace of a CRAG (Corrective RAG) loop."""
    original_query: str
    plan: List[RAGPlanStep] = field(default_factory=list)
    iterations: int = 0
    final_quality: float = 0.0


def demo_crag_pattern():
    """Walk through a complete CRAG execution trace."""

    print("\n" + "=" * 65)
    print("  PLAN → RETRIEVE → CRITIQUE → RE-RETRIEVE (CRAG)")
    print("=" * 65)

    query = "Compare RLHF and DPO: which is better for safety alignment?"

    print(f"\n  Query: {query}")

    # PLAN phase
    print(f"\n  ── PLAN ──")
    plan = [
        RAGPlanStep("What is RLHF and how does it work?", "research_papers",
                     "RLHF reinforcement learning human feedback mechanism"),
        RAGPlanStep("What is DPO and how does it work?", "research_papers",
                     "DPO direct preference optimization mechanism"),
        RAGPlanStep("Comparison of RLHF vs DPO for safety", "research_papers",
                     "RLHF DPO comparison safety alignment effectiveness"),
    ]
    for i, step in enumerate(plan):
        print(f"    Step {i + 1}: {step.information_need}")
        print(f"      → Source: {step.target_source}, Query: '{step.query[:50]}'")

    # RETRIEVE phase (iteration 1)
    print(f"\n  ── RETRIEVE (iteration 1) ──")
    plan[0].retrieved_docs = ["RLHF paper abstract: trains using PPO with reward model..."]
    plan[0].quality_score = 0.85
    plan[0].status = "verified"
    print(f"    Step 1: Retrieved 1 doc, quality={plan[0].quality_score} ✓")

    plan[1].retrieved_docs = ["DPO paper abstract: direct optimization without reward..."]
    plan[1].quality_score = 0.82
    plan[1].status = "verified"
    print(f"    Step 2: Retrieved 1 doc, quality={plan[1].quality_score} ✓")

    plan[2].retrieved_docs = ["Blog post: RLHF is standard but DPO is simpler..."]
    plan[2].quality_score = 0.45  # Low quality — not a rigorous comparison
    plan[2].status = "failed"
    print(f"    Step 3: Retrieved 1 doc, quality={plan[2].quality_score} ✗ (below threshold)")

    # CRITIQUE phase
    print(f"\n  ── CRITIQUE ──")
    print(f"    Steps 1-2: Sufficient quality ✓")
    print(f"    Step 3: Quality too low ({plan[2].quality_score} < 0.7)")
    print(f"    Decision: RE-RETRIEVE step 3 with refined query")

    # RE-RETRIEVE phase (iteration 2)
    print(f"\n  ── RE-RETRIEVE (iteration 2) ──")
    plan[2].query = "RLHF vs DPO safety alignment comparison empirical results"
    plan[2].retrieved_docs = ["Survey paper: systematic comparison of RLHF, DPO for safety..."]
    plan[2].quality_score = 0.88
    plan[2].status = "verified"
    print(f"    Step 3 (refined): Retrieved 1 doc, quality={plan[2].quality_score} ✓")

    # SYNTHESIZE
    print(f"\n  ── SYNTHESIZE ──")
    print(f"    All 3 information needs satisfied.")
    print(f"    Total iterations: 2 (1 re-retrieval)")
    print(f"    Ready to generate final answer with 3 verified sources.")


# ================================================================
# 4. REFLECTION & VERIFICATION IN RETRIEVAL
# ================================================================

def demo_reflection_verification():
    """Show how agents verify retrieved information."""

    print("\n" + "=" * 65)
    print("  REFLECTION & VERIFICATION IN RETRIEVAL")
    print("=" * 65)

    print("""
  The agent performs THREE types of verification:

  1. RELEVANCE CHECK: "Does this document answer the query?"
     ┌──────────────────────────────────────────────┐
     │ Query: "How does RLHF work?"                 │
     │ Doc: "RLHF trains using human preferences"   │ ← Relevant ✓
     │ Doc: "DPO is an alternative to RLHF"         │ ← Tangential ✗
     └──────────────────────────────────────────────┘

  2. GROUNDING CHECK: "Is the generated answer supported by sources?"
     ┌──────────────────────────────────────────────┐
     │ Answer: "RLHF uses PPO with a reward model"  │
     │ Source: "...optimized using PPO to maximize   │
     │          the reward signal"                   │ ← Grounded ✓
     │                                               │
     │ Answer: "RLHF was invented in 2019"           │
     │ Source: (not mentioned anywhere)              │ ← Hallucination ✗
     └──────────────────────────────────────────────┘

  3. COMPLETENESS CHECK: "Does the answer address ALL parts of the query?"
     ┌──────────────────────────────────────────────┐
     │ Query: "Compare RLHF and DPO for safety"     │
     │ Answer covers: RLHF ✓, DPO ✓, Comparison ✓, │
     │                Safety relevance ✗             │ ← Incomplete
     │ Action: Re-retrieve for safety-specific data  │
     └──────────────────────────────────────────────┘

  Verification adds 1-2 LLM calls but dramatically reduces
  hallucination.  Production systems should ALWAYS verify.
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  WEEK 5 — EXAMPLE 8: Agentic RAG Concepts                    ║")
    print("╚" + "═" * 63 + "╝")

    demo_adaptive_retrieval()
    demo_query_routing()
    demo_crag_pattern()
    demo_reflection_verification()

    print("\n" + "=" * 65)
    print("  KEY TAKEAWAYS & BEST PRACTICES")
    print("=" * 65)
    print(textwrap.dedent("""
    1. Self-RAG lets the agent DECIDE whether retrieval is needed.
       This saves cost/latency on queries that don't need it.

    2. Query routing sends queries to the RIGHT knowledge base.
       Don't search everything — route by domain, topic, and freshness.

    3. The CRAG pattern (Plan→Retrieve→Critique→Re-retrieve) is the
       gold standard for production RAG.  It self-corrects low-quality
       retrievals instead of passing garbage to the generator.

    4. Three verification types: relevance, grounding, completeness.
       All three should be checked in production systems.

    5. Agentic RAG costs more (extra LLM calls for classification,
       critique, and verification) but dramatically improves quality
       and reduces hallucination.
    """))
