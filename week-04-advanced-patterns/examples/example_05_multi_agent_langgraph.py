import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 5: Multi-Agent Supervisor/Worker System in LangGraph
=============================================================
This example builds a complete supervisor/worker multi-agent system
where specialized agents collaborate to research a topic.

The architecture has FOUR roles:
  1. SUPERVISOR — decomposes a research question into sub-questions
  2. RESEARCHER — searches for information on each sub-question
  3. ANALYST — synthesizes all findings into a coherent analysis
  4. JUDGE — scores the analysis quality (1-10) and decides if
     the system should iterate or finish

This is the Hierarchical (Supervisor/Worker) topology from Example 4,
now implemented as a real LangGraph with LLM-powered nodes.

Key Concepts (Week 4, Topics 2-3):
  - Supervisor/Worker architecture with LLM agents
  - State-based coordination between agents
  - Conditional routing (quality gate with retry loop)
  - Simulated tools for reproducible teaching

Graph Flow:
  START -> supervisor -> researcher -> analyst -> judge
                                                   |
                                          score >= 7? -> END
                                          score < 7 AND iterations left? -> supervisor (retry)
                                          else -> END

Run: python week-04-advanced-patterns/examples/example_05_multi_agent_langgraph.py
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict


# ==============================================================
# Step 1: LLM Setup
# ==============================================================

def get_llm(temperature=0.7):
    """Get the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


# ==============================================================
# Step 2: Simulated Research Tools
# ==============================================================
# These tools return hardcoded data so the example works without
# external APIs. In production, these would call real search APIs
# (Tavily, SerpAPI, Brave Search, etc.).

@tool
def search_academic(query: str) -> str:
    """Search academic papers and research publications for a topic.

    Use this to find peer-reviewed research, studies, and scholarly
    analysis on a given topic.

    Args:
        query: Academic search query (e.g., 'AI impact on student learning')
    """
    academic_db = {
        "ai education": (
            "Academic Results for AI in Education:\n"
            "  1. Zhang et al. (2024) — 'Adaptive Learning with AI Tutors': Students using "
            "AI-powered tutoring systems showed 23% improvement in test scores compared to "
            "traditional methods. Study covered 5,000 students across 12 universities.\n"
            "  2. Patel & Williams (2025) — 'LLMs as Teaching Assistants': Large language models "
            "reduced teacher workload by 35% for grading and feedback tasks. Teachers reported "
            "higher job satisfaction but raised concerns about student over-reliance.\n"
            "  3. Chen (2024) — 'Equity in AI-Assisted Education': AI tutoring tools helped "
            "close achievement gaps for disadvantaged students by 15%, but only when schools "
            "provided adequate digital infrastructure."
        ),
        "ai student learning": (
            "Academic Results for AI and Student Learning:\n"
            "  1. Morrison (2025) — 'Personalized Learning Pathways': AI systems that adapt "
            "to individual learning styles improved retention rates by 28%.\n"
            "  2. Kumar et al. (2024) — 'Critical Thinking in the Age of AI': Students who "
            "used AI assistants without guidance showed 12% decline in independent problem-solving."
        ),
        "ai teachers": (
            "Academic Results for AI and Teachers:\n"
            "  1. Davies & Lopez (2025) — 'Teacher Perspectives on AI': 67% of teachers see AI "
            "as a useful tool, but 78% want more training on how to integrate it effectively.\n"
            "  2. Nakamura (2024) — 'AI Grading Accuracy': AI grading matched human grading "
            "with 91% agreement on standardized assessments, but struggled with creative writing."
        ),
    }

    query_lower = query.lower()
    for keyword, results in academic_db.items():
        if any(word in query_lower for word in keyword.split()):
            return results

    return f"No academic results found for '{query}'. Try broader terms like 'ai education'."


@tool
def search_news(query: str) -> str:
    """Search recent news articles and reports about a topic.

    Use this to find current developments, industry trends, and
    real-world examples related to the query.

    Args:
        query: News search query (e.g., 'AI education policy 2025')
    """
    news_db = {
        "ai education": (
            "News Results for AI in Education:\n"
            "  1. [Reuters, Jan 2026] UNESCO releases guidelines for responsible AI use in "
            "classrooms. Recommends human oversight for all AI-generated assessments.\n"
            "  2. [TechCrunch, Dec 2025] Khan Academy's AI tutor 'Khanmigo' reaches 10 million "
            "students worldwide. Reports 40% improvement in math scores for regular users.\n"
            "  3. [BBC, Feb 2026] UK schools pilot AI-assisted lesson planning. Early results "
            "show teachers save an average of 5 hours per week on administrative tasks."
        ),
        "ai policy": (
            "News Results for AI Policy:\n"
            "  1. [NYT, Mar 2026] The EU AI Act's education provisions take effect, requiring "
            "transparency labels on all AI-generated educational content.\n"
            "  2. [WSJ, Jan 2026] US Department of Education allocates $500M for AI literacy "
            "programs in public schools."
        ),
        "ai challenges": (
            "News Results for AI Challenges in Education:\n"
            "  1. [Guardian, Feb 2026] Study finds 45% of university students have used AI to "
            "complete assignments. Universities debate detection vs. integration approaches.\n"
            "  2. [Wired, Dec 2025] Digital divide widens: rural schools lack infrastructure "
            "for AI tools that urban schools increasingly rely on."
        ),
    }

    query_lower = query.lower()
    for keyword, results in news_db.items():
        if any(word in query_lower for word in keyword.split()):
            return results

    return f"No news results found for '{query}'. Try broader terms like 'ai education'."


# ==============================================================
# Step 3: Multi-Agent State
# ==============================================================
# This state is shared across ALL nodes. Each node reads what it
# needs and writes its contribution. This is the "shared blackboard"
# communication pattern from Example 4.

class MultiAgentState(TypedDict):
    query: str                  # Original research question
    sub_questions: list         # Supervisor's decomposition
    research_findings: list     # Researcher's collected results
    analysis: str               # Analyst's synthesis
    quality_score: int          # Judge's quality rating (1-10)
    iteration: int              # Current iteration count
    max_iterations: int         # Maximum retry attempts


# ==============================================================
# Step 4: Agent Nodes
# ==============================================================
# Each node is a function that takes the state, does work (often
# calling an LLM), and returns a state update dict.

def supervisor_node(state: MultiAgentState) -> dict:
    """The Supervisor decomposes the research question into sub-questions.

    Think of a newspaper editor assigning stories to reporters.
    The supervisor doesn't DO the research — it figures out WHAT
    needs to be researched.
    """
    llm = get_llm(temperature=0.3)  # Low temperature for structured output

    iteration = state.get("iteration", 0)
    previous_analysis = state.get("analysis", "")

    if iteration > 0 and previous_analysis:
        # On retry, ask for DIFFERENT sub-questions
        prompt = f"""You are a research supervisor. A previous research attempt on this topic
scored below the quality threshold. Here is the topic and previous analysis:

Topic: {state['query']}
Previous Analysis: {previous_analysis[:500]}

Generate 2-3 NEW and DIFFERENT sub-questions that would improve the research.
Focus on gaps or weak areas in the previous analysis.

Return ONLY a numbered list:
1. [first sub-question]
2. [second sub-question]
3. [optional third sub-question]"""
    else:
        prompt = f"""You are a research supervisor. Break down this research question into
2-3 focused sub-questions that specialized researchers can investigate independently.

Research question: {state['query']}

Return ONLY a numbered list:
1. [first sub-question]
2. [second sub-question]
3. [optional third sub-question]"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse numbered list from response
    lines = response.content.strip().split("\n")
    sub_questions = []
    for line in lines:
        # Match lines starting with a number (e.g., "1.", "1)", "1:")
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line.strip())
        if cleaned and len(cleaned) > 10:  # Skip too-short lines
            sub_questions.append(cleaned)

    # Fallback if parsing fails
    if not sub_questions:
        sub_questions = [
            f"What are the main benefits of {state['query']}?",
            f"What are the challenges and risks of {state['query']}?",
            f"What does recent research say about {state['query']}?",
        ]

    print(f"\n  [Supervisor] Decomposed into {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"    {i}. {q}")

    return {
        "sub_questions": sub_questions,
        "iteration": iteration + 1,
    }


def researcher_node(state: MultiAgentState) -> dict:
    """The Researcher investigates each sub-question using tools.

    Think of a reporter going out to gather facts. The researcher
    calls search tools and collects raw findings.
    """
    findings = []

    for i, question in enumerate(state["sub_questions"], 1):
        print(f"\n  [Researcher] Investigating sub-question {i}: {question[:60]}...")

        # Call both search tools for each sub-question
        academic_result = search_academic.invoke({"query": question})
        news_result = search_news.invoke({"query": question})

        finding = {
            "sub_question": question,
            "academic": academic_result,
            "news": news_result,
        }
        findings.append(finding)

        print(f"    Found academic sources: {'Yes' if 'No academic results' not in academic_result else 'No'}")
        print(f"    Found news sources: {'Yes' if 'No news results' not in news_result else 'No'}")

    return {"research_findings": findings}


def analyst_node(state: MultiAgentState) -> dict:
    """The Analyst synthesizes all research findings into a coherent analysis.

    Think of a senior editor who takes all the reporters' stories
    and weaves them into a comprehensive feature article.
    """
    llm = get_llm(temperature=0.5)

    # Build context from all findings
    findings_text = ""
    for i, finding in enumerate(state["research_findings"], 1):
        findings_text += f"\n--- Sub-question {i}: {finding['sub_question']} ---\n"
        findings_text += f"Academic Sources:\n{finding['academic']}\n"
        findings_text += f"News Sources:\n{finding['news']}\n"

    prompt = f"""You are a research analyst. Synthesize the following research findings
into a clear, well-structured analysis that answers the original question.

Original Question: {state['query']}

Research Findings:
{findings_text}

Write a comprehensive analysis (3-4 paragraphs) that:
1. Summarizes the key findings across all sub-questions
2. Identifies patterns, agreements, and contradictions in the evidence
3. Draws actionable conclusions
4. Notes any gaps or limitations in the research

Write the analysis directly — no preamble or meta-commentary."""

    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content.strip()

    print(f"\n  [Analyst] Synthesized {len(state['research_findings'])} findings into analysis")
    print(f"    Analysis length: {len(analysis)} characters")

    return {"analysis": analysis}


def judge_node(state: MultiAgentState) -> dict:
    """The Judge evaluates the analysis quality and assigns a score.

    Think of a quality control inspector who decides if the product
    meets standards before shipping.
    """
    llm = get_llm(temperature=0.1)  # Very low temperature for consistent scoring

    prompt = f"""You are a research quality judge. Evaluate this analysis and assign
a quality score from 1 to 10.

Original Question: {state['query']}

Analysis to Evaluate:
{state['analysis'][:2000]}

Score the analysis on these criteria:
- Comprehensiveness (does it cover multiple perspectives?)
- Evidence quality (does it cite specific studies/sources?)
- Logical coherence (does the argument flow well?)
- Actionable conclusions (are there clear takeaways?)

Respond with ONLY a JSON object:
{{"score": <number 1-10>, "reason": "<one sentence explanation>"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Parse JSON with fallback for markdown-wrapped responses
    # LLMs sometimes wrap JSON in ```json ... ``` code blocks
    score = 7  # Default fallback
    reason = "Could not parse judge response"

    try:
        # Try direct JSON parse first
        result = json.loads(response_text)
        score = int(result.get("score", 7))
        reason = result.get("reason", "No reason provided")
    except json.JSONDecodeError:
        # Fallback: try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                score = int(result.get("score", 7))
                reason = result.get("reason", "No reason provided")
            except (json.JSONDecodeError, ValueError):
                pass
        else:
            # Last resort: look for a number in the response
            number_match = re.search(r"\b(\d+)\b", response_text)
            if number_match:
                score = min(10, max(1, int(number_match.group(1))))
                reason = "Score extracted from unstructured response"

    # Clamp score to valid range
    score = min(10, max(1, score))

    print(f"\n  [Judge] Quality Score: {score}/10")
    print(f"    Reason: {reason}")

    return {"quality_score": score}


# ==============================================================
# Step 5: Routing Logic
# ==============================================================
# After the judge scores, we decide: accept (END) or retry (supervisor).
# This is the quality gate — the conditional edge that makes the
# graph a loop rather than a straight pipeline.

def should_retry(state: MultiAgentState) -> str:
    """Decide whether to accept the analysis or retry.

    Rules:
      - Score >= 7: Accept the analysis (go to END)
      - Score < 7 AND iterations remaining: Retry (go back to supervisor)
      - Score < 7 AND no iterations left: Accept anyway (go to END)
    """
    score = state.get("quality_score", 7)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)

    if score >= 7:
        print(f"  [Router] Score {score} >= 7. Accepting analysis.")
        return "accept"
    elif iteration < max_iterations:
        print(f"  [Router] Score {score} < 7, iteration {iteration}/{max_iterations}. Retrying...")
        return "retry"
    else:
        print(f"  [Router] Score {score} < 7, but max iterations reached. Accepting anyway.")
        return "accept"


# ==============================================================
# Step 6: Build the LangGraph
# ==============================================================
# This is where the multi-agent topology becomes a real graph.
# Each agent is a node, and edges define the workflow.

def build_multi_agent_graph():
    """Build the supervisor/worker multi-agent graph.

    Graph Structure:
        supervisor -> researcher -> analyst -> judge
                                                 |
                                        retry -> supervisor
                                        accept -> END
    """
    graph = StateGraph(MultiAgentState)

    # Add agent nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("judge", judge_node)

    # Add edges (the workflow)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "judge")

    # Conditional edge from judge: retry or accept
    graph.add_conditional_edges(
        "judge",
        should_retry,
        {
            "retry": "supervisor",
            "accept": END,
        },
    )

    return graph.compile()


# ==============================================================
# Step 7: Run the Multi-Agent System
# ==============================================================

def main():
    """Run the multi-agent research system."""
    print("Example 5: Multi-Agent Supervisor/Worker System in LangGraph")
    print("=" * 65)

    # Build the graph
    app = build_multi_agent_graph()

    # Define the research query
    query = "What is the impact of AI on education?"

    print(f"\nResearch Query: {query}")
    print(f"Max Iterations: 2")
    print("-" * 65)

    # Run the graph with initial state
    initial_state = {
        "query": query,
        "sub_questions": [],
        "research_findings": [],
        "analysis": "",
        "quality_score": 0,
        "iteration": 0,
        "max_iterations": 2,
    }

    result = app.invoke(initial_state)

    # Display results
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(f"\nOriginal Query: {result['query']}")
    print(f"Iterations Used: {result['iteration']}")
    print(f"Final Quality Score: {result['quality_score']}/10")

    print(f"\nSub-Questions Investigated:")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")

    print(f"\nResearch Sources Used: {len(result['research_findings'])} sub-topics")
    for i, finding in enumerate(result["research_findings"], 1):
        has_academic = "No academic results" not in finding["academic"]
        has_news = "No news results" not in finding["news"]
        print(f"  {i}. {finding['sub_question'][:50]}... "
              f"[Academic: {'Yes' if has_academic else 'No'}, News: {'Yes' if has_news else 'No'}]")

    print(f"\nFinal Analysis:")
    print("-" * 65)
    print(result["analysis"])
    print("-" * 65)

    # Architecture summary
    print(f"\n{'='*65}")
    print("Multi-Agent Architecture Summary:")
    print(f"{'='*65}")
    print("  Topology: Hierarchical (Supervisor/Worker)")
    print("  Communication: Shared State (blackboard pattern)")
    print("  Quality Control: Judge node with retry loop")
    print("  Agents:")
    print("    1. Supervisor — decomposes query into sub-questions")
    print("    2. Researcher — gathers evidence using search tools")
    print("    3. Analyst   — synthesizes findings into analysis")
    print("    4. Judge     — scores quality and decides retry/accept")
    print(f"\n  LangGraph Advantage:")
    print("    The entire workflow is defined as a GRAPH with explicit")
    print("    edges and conditional routing. You can visualize it,")
    print("    checkpoint it, and resume from any node.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
