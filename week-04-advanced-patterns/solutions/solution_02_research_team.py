import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 2: Multi-Agent Research Team (Supervisor/Worker)
==========================================================
Complete solution for exercise_02_research_team.py

This implements a multi-agent research system with:
  1. SUPERVISOR   — decomposes the query into 3-4 sub-questions
  2. RESEARCHER   — searches academic and news sources per sub-question
  3. ANALYST      — synthesizes findings into a structured analysis
  4. JUDGE        — scores the analysis (1-10) with behavioral anchors
  5. Conditional loop — retries if score < 7, up to 3 iterations

Run: python week-04-advanced-patterns/solutions/solution_02_research_team.py
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
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
# Hardcoded data so the solution works without external APIs.
# Keywords in the query are matched to return relevant results.

@tool
def search_academic(query: str) -> str:
    """Search academic papers and research publications.

    Use this to find peer-reviewed studies, statistics, and scholarly
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
    """Search recent news articles and industry reports.

    Use this to find current developments, trends, and real-world
    examples related to the query.

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
# Step 3: State Definition
# ==============================================================

class ResearchState(TypedDict):
    query: str                  # Original research question
    sub_questions: list         # Supervisor's decomposition (list of strings)
    research_findings: list     # Researcher's raw results (list of dicts)
    analysis: str               # Analyst's synthesized report
    quality_score: int          # Judge's rating (1-10)
    judge_feedback: str         # Judge's explanation for the score
    iteration: int              # Current loop count
    max_iterations: int         # Max allowed loops


# ==============================================================
# Step 4: Supervisor Node
# ==============================================================

def supervisor_node(state: ResearchState) -> dict:
    """Decompose the research query into focused sub-questions.

    On the first pass, generate fresh sub-questions. On retries,
    generate DIFFERENT sub-questions that address gaps identified
    by the judge in the previous round.
    """
    llm = get_llm(temperature=0.3)

    iteration = state.get("iteration", 0)
    previous_feedback = state.get("judge_feedback", "")
    previous_analysis = state.get("analysis", "")

    if iteration > 0 and previous_analysis:
        # Retry: ask for different sub-questions based on judge feedback
        prompt = (
            "You are a research supervisor. A previous research attempt on this "
            "topic scored below the quality threshold.\n\n"
            f"Topic: {state['query']}\n"
            f"Previous Analysis (summary): {previous_analysis[:500]}\n"
            f"Judge Feedback: {previous_feedback}\n\n"
            "Generate 3-4 NEW and DIFFERENT sub-questions that would improve "
            "the research. Focus on the gaps or weaknesses the judge identified.\n\n"
            "Return ONLY a numbered list:\n"
            "1. [first sub-question]\n"
            "2. [second sub-question]\n"
            "3. [third sub-question]\n"
            "4. [optional fourth sub-question]"
        )
    else:
        # First pass: decompose the query
        prompt = (
            "Break this research question into 3-4 focused sub-questions that "
            "specialized researchers can investigate independently. Each sub-question "
            "should cover a different angle (e.g., benefits, challenges, evidence, "
            "policy implications).\n\n"
            f"Research question: {state['query']}\n\n"
            "Return ONLY a numbered list:\n"
            "1. [first sub-question]\n"
            "2. [second sub-question]\n"
            "3. [third sub-question]\n"
            "4. [optional fourth sub-question]"
        )

    response = llm.invoke([
        SystemMessage(content="You are a research supervisor who breaks complex questions into focused sub-questions."),
        HumanMessage(content=prompt),
    ])

    # Parse numbered list from response
    lines = response.content.strip().split("\n")
    sub_questions = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line.strip())
        if cleaned and len(cleaned) > 10:
            sub_questions.append(cleaned)

    # Fallback if parsing fails
    if not sub_questions:
        sub_questions = [
            f"What are the main benefits of {state['query']}?",
            f"What are the challenges and risks of {state['query']}?",
            f"What does recent research say about {state['query']}?",
        ]

    print(f"\n  [Supervisor] Iteration {iteration + 1}: Decomposed into {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"    {i}. {q}")

    return {
        "sub_questions": sub_questions,
        "iteration": iteration + 1,
    }


# ==============================================================
# Step 5: Researcher Node
# ==============================================================

def researcher_node(state: ResearchState) -> dict:
    """Search academic and news sources for each sub-question.

    Calls both search_academic and search_news for every sub-question,
    collecting all raw findings for the analyst.
    """
    findings = []

    for i, question in enumerate(state["sub_questions"], 1):
        print(f"\n  [Researcher] Investigating sub-question {i}: {question[:60]}...")

        # Call both search tools
        academic_result = search_academic.invoke({"query": question})
        news_result = search_news.invoke({"query": question})

        finding = {
            "sub_question": question,
            "academic": academic_result,
            "news": news_result,
        }
        findings.append(finding)

        has_academic = "No academic results" not in academic_result
        has_news = "No news results" not in news_result
        print(f"    Academic sources: {'Found' if has_academic else 'None'}")
        print(f"    News sources: {'Found' if has_news else 'None'}")

    return {"research_findings": findings}


# ==============================================================
# Step 6: Analyst Node
# ==============================================================

def analyst_node(state: ResearchState) -> dict:
    """Synthesize all research findings into a structured analysis.

    Takes the raw findings from the researcher and produces a coherent,
    evidence-based report that answers the original question.
    """
    llm = get_llm(temperature=0.5)

    # Build context from all findings
    findings_text = ""
    for i, finding in enumerate(state["research_findings"], 1):
        findings_text += f"\n--- Sub-question {i}: {finding['sub_question']} ---\n"
        findings_text += f"Academic Sources:\n{finding['academic']}\n"
        findings_text += f"News Sources:\n{finding['news']}\n"

    prompt = (
        f"Synthesize the following research findings into a clear, well-structured "
        f"analysis that answers the original question.\n\n"
        f"Original Question: {state['query']}\n\n"
        f"Research Findings:\n{findings_text}\n\n"
        f"Write a comprehensive analysis (3-4 paragraphs) that:\n"
        f"1. Summarizes the key findings across all sub-questions\n"
        f"2. Cites specific studies, statistics, and sources as evidence\n"
        f"3. Identifies patterns, agreements, and contradictions\n"
        f"4. Draws actionable conclusions and notes limitations\n\n"
        f"Write the analysis directly — no preamble or meta-commentary."
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data analyst. Synthesize these research findings into a clear, "
            "evidence-based analysis. Cite specific numbers and sources. Be thorough "
            "but concise."
        )),
        HumanMessage(content=prompt),
    ])

    analysis = response.content.strip()

    print(f"\n  [Analyst] Synthesized {len(state['research_findings'])} findings into analysis")
    print(f"    Analysis length: {len(analysis)} characters")

    return {"analysis": analysis}


# ==============================================================
# Step 7: Judge Node
# ==============================================================

def judge_node(state: ResearchState) -> dict:
    """Score the analysis quality using a rubric with behavioral anchors.

    The judge evaluates on three criteria (correctness, completeness,
    clarity) and assigns an overall score from 1-10 with behavioral
    anchors describing what each score range looks like.
    """
    llm = get_llm(temperature=0.1)

    prompt = (
        f"You are a research quality judge. Evaluate this analysis and assign "
        f"a quality score from 1 to 10.\n\n"
        f"Original Question: {state['query']}\n\n"
        f"Analysis to Evaluate:\n{state['analysis'][:2000]}\n\n"
        f"Score using this rubric with behavioral anchors:\n\n"
        f"  1-3 (Poor): Missing evidence, no clear structure, fails to answer "
        f"the query, makes unsupported claims, no sources cited.\n"
        f"  4-6 (Adequate): Some evidence cited, partial coverage of the topic, "
        f"basic structure but gaps remain, conclusions are vague.\n"
        f"  7-8 (Good): Strong evidence with specific citations, covers multiple "
        f"perspectives, clear and logical conclusions, identifies limitations.\n"
        f"  9-10 (Excellent): Exceptional depth and nuance, novel insights, "
        f"actionable recommendations, synthesizes contradictions, publication-ready.\n\n"
        f"Evaluate on these criteria:\n"
        f"  - Correctness: Are claims supported by the research findings?\n"
        f"  - Completeness: Does it address the full scope of the query?\n"
        f"  - Clarity: Is the analysis well-organized and readable?\n\n"
        f"Respond with ONLY a JSON object (no markdown, no explanation outside JSON):\n"
        f"{{\"score\": <number 1-10>, \"feedback\": \"<2-3 sentences explaining the score and any gaps>\"}}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Parse JSON with robust fallbacks
    score = 7
    feedback = "Could not parse judge response"

    try:
        result = json.loads(response_text)
        score = int(result.get("score", 7))
        feedback = result.get("feedback", "No feedback provided")
    except json.JSONDecodeError:
        # Fallback: try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                score = int(result.get("score", 7))
                feedback = result.get("feedback", "No feedback provided")
            except (json.JSONDecodeError, ValueError):
                pass
        else:
            # Last resort: extract a number
            number_match = re.search(r"\b(\d+)\b", response_text)
            if number_match:
                score = min(10, max(1, int(number_match.group(1))))
                feedback = "Score extracted from unstructured response"

    # Clamp to valid range
    score = min(10, max(1, score))

    print(f"\n  [Judge] Quality Score: {score}/10")
    print(f"    Feedback: {feedback}")

    return {
        "quality_score": score,
        "judge_feedback": feedback,
    }


# ==============================================================
# Step 8: Routing Function
# ==============================================================

def should_continue(state: ResearchState) -> str:
    """Route based on quality score and iteration count.

    Rules:
      - score >= 7          -> "accept" (done, go to END)
      - iteration >= max    -> "accept" (tried enough, go to END)
      - otherwise           -> "retry"  (back to supervisor)
    """
    score = state.get("quality_score", 7)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)

    if score >= 7:
        print(f"  [Router] Score {score} >= 7. Accepting analysis.")
        return "accept"
    elif iteration >= max_iterations:
        print(f"  [Router] Score {score} < 7, but max iterations ({max_iterations}) reached. Accepting anyway.")
        return "accept"
    else:
        print(f"  [Router] Score {score} < 7, iteration {iteration}/{max_iterations}. Retrying...")
        return "retry"


# ==============================================================
# Step 9: Build the Graph
# ==============================================================

def build_research_graph():
    """Build the multi-agent research graph.

    Graph Structure:
        START -> supervisor -> researcher -> analyst -> judge
                                                         |
                                                retry -> supervisor
                                                accept -> END
    """
    graph = StateGraph(ResearchState)

    # Add agent nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("judge", judge_node)

    # Linear edges through the pipeline
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "judge")

    # Conditional edge from judge: retry or accept
    graph.add_conditional_edges(
        "judge",
        should_continue,
        {
            "retry": "supervisor",
            "accept": END,
        },
    )

    return graph.compile()


# ==============================================================
# Step 10: Run the Multi-Agent System
# ==============================================================

def main():
    """Run the multi-agent research system."""
    print("Solution 2: Multi-Agent Research Team")
    print("=" * 60)

    # Build the graph
    app = build_research_graph()

    # Test query
    query = "What is the impact of AI on education?"

    print(f"\nResearch Query: {query}")
    print(f"Max Iterations: 3")
    print("-" * 60)

    # Run the graph with initial state
    initial_state = {
        "query": query,
        "sub_questions": [],
        "research_findings": [],
        "analysis": "",
        "quality_score": 0,
        "judge_feedback": "",
        "iteration": 0,
        "max_iterations": 3,
    }

    result = app.invoke(initial_state)

    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"\nOriginal Query: {result['query']}")
    print(f"Iterations Used: {result['iteration']}")
    print(f"Final Score: {result['quality_score']}/10")
    print(f"Judge Feedback: {result['judge_feedback']}")

    print(f"\nSub-Questions Investigated:")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")

    print(f"\nResearch Sources Used: {len(result['research_findings'])} sub-topics")
    for i, finding in enumerate(result["research_findings"], 1):
        has_academic = "No academic results" not in finding["academic"]
        has_news = "No news results" not in finding["news"]
        print(f"  {i}. {finding['sub_question'][:50]}... "
              f"[Academic: {'Yes' if has_academic else 'No'}, "
              f"News: {'Yes' if has_news else 'No'}]")

    print(f"\nFinal Analysis:")
    print("-" * 60)
    print(result["analysis"])
    print("-" * 60)

    # Architecture summary
    print(f"\n{'=' * 60}")
    print("Architecture Summary:")
    print(f"{'=' * 60}")
    print("  Topology: Hierarchical (Supervisor/Worker)")
    print("  Communication: Shared State (blackboard pattern)")
    print("  Quality Control: Judge node with retry loop (max 3)")
    print("  Agents:")
    print("    1. Supervisor  — decomposes query into sub-questions")
    print("    2. Researcher  — gathers evidence using search tools")
    print("    3. Analyst     — synthesizes findings into analysis")
    print("    4. Judge       — scores quality with behavioral rubric")
    print("  Routing: score >= 7 or max iterations -> END, else -> retry")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
