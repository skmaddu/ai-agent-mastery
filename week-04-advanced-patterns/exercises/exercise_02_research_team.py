import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 2: Multi-Agent Research Team (Supervisor/Worker)
==========================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 3 hours

Task:
Build a multi-agent research system using LangGraph's supervisor/worker
pattern. The system has FIVE roles that collaborate to answer a research
question:

  1. SUPERVISOR   — decomposes the research query into 3-5 focused sub-questions
  2. RESEARCHER   — searches academic and news sources for each sub-question
  3. ANALYST      — processes all findings and produces a structured analysis
  4. SYNTHESIZER  — combines the analysis into a final structured report
  5. JUDGE        — scores the report (1-10) on a rubric; if score < 7,
                    the system loops back to the supervisor for another attempt

The system should loop at most 3 times (max_iterations = 3).

Instructions:
1. Complete the ResearchState TypedDict (TODO 1)
2. Implement supervisor_node — use the LLM to decompose the query (TODO 2)
3. Implement researcher_node — call both search tools for each sub-question (TODO 3)
4. Implement analyst_node — use the LLM to synthesize findings (TODO 4)
5. Implement judge_node — use the LLM to score with a rubric, return JSON (TODO 5)
6. Implement should_continue — route based on score and iteration count (TODO 6)
7. Wire the graph with nodes, edges, and conditional routing (TODO 7)

Hints:
- Look at example_05_multi_agent_langgraph.py for the supervisor/worker pattern
- Look at example_08_evaluator_optimizer_langgraph.py for the loop/judge pattern
- The judge prompt should include behavioral anchors (what does a 3 look like vs. an 8?)
- Use low temperature (0.1-0.3) for the judge to get consistent scoring
- Use re.sub() to parse numbered lists from LLM output
- Always have a fallback if JSON parsing fails in the judge node
- The analyst and synthesizer can be combined into a single analyst node
  for simplicity — the key pattern is decompose -> research -> synthesize -> judge

Success Criteria:
- Supervisor correctly decomposes queries into 3-5 sub-questions
- Researcher calls both search_academic and search_news for each sub-question
- Analyst produces a multi-paragraph synthesis with evidence
- Judge rubric has behavioral anchors (not just "rate 1-10")
- Max 3 total iterations
- The graph compiles and runs end-to-end

Run: python week-04-advanced-patterns/exercises/exercise_02_research_team.py
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
# LLM Setup (provided)
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
# Simulated Research Tools (provided)
# ==============================================================
# These return hardcoded data so the exercise works without
# external APIs. In production, you'd call Tavily, SerpAPI, etc.

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
# TODO 1: Complete the State Definition
# ==============================================================
# The state is shared across ALL nodes. Each field is read/written
# by specific nodes. Fill in the type annotations and understand
# what each field is for.
#
# Fields:
#   query            — the original research question (str)
#   sub_questions     — supervisor's decomposition (list of strings)
#   research_findings — researcher's raw results (list of dicts)
#   analysis          — analyst's synthesized report (str)
#   quality_score     — judge's rating 1-10 (int)
#   judge_feedback    — judge's explanation for the score (str)
#   iteration         — current loop count (int)
#   max_iterations    — max allowed loops (int)

class ResearchState(TypedDict):
    # TODO: Add all 8 fields with correct type annotations
    pass


# ==============================================================
# TODO 2: Implement the Supervisor Node
# ==============================================================
# The supervisor decomposes the research query into 3-4 sub-questions.
# On retry (iteration > 0), it should generate DIFFERENT sub-questions
# that address gaps from the previous attempt.
#
# Steps:
#   1. Get the LLM (low temperature for structured output)
#   2. Check if this is a retry (iteration > 0)
#   3. Build a prompt asking for 3-4 numbered sub-questions
#   4. Parse the numbered list from the response
#   5. Return {"sub_questions": [...], "iteration": iteration + 1}
#
# Hint: Use re.sub(r"^\d+[\.\)\:]\s*", "", line.strip()) to strip
#       numbering from each line.

def supervisor_node(state: ResearchState) -> dict:
    """Decompose the research query into focused sub-questions."""
    # TODO: Implement this node
    pass


# ==============================================================
# TODO 3: Implement the Researcher Node
# ==============================================================
# The researcher calls BOTH search tools for each sub-question
# and collects the results.
#
# Steps:
#   1. Loop through state["sub_questions"]
#   2. Call search_academic.invoke({"query": question}) for each
#   3. Call search_news.invoke({"query": question}) for each
#   4. Store results as a list of dicts:
#      {"sub_question": q, "academic": result1, "news": result2}
#   5. Return {"research_findings": findings}

def researcher_node(state: ResearchState) -> dict:
    """Search academic and news sources for each sub-question."""
    # TODO: Implement this node
    pass


# ==============================================================
# TODO 4: Implement the Analyst Node
# ==============================================================
# The analyst takes ALL research findings and synthesizes them
# into a coherent, structured analysis.
#
# Steps:
#   1. Get the LLM
#   2. Build a context string from all findings
#   3. Prompt the LLM to synthesize (3-4 paragraphs)
#   4. Return {"analysis": analysis_text}
#
# Hint: Use a SystemMessage like:
#   "You are a data analyst. Synthesize these research findings
#    into a clear, evidence-based analysis."

def analyst_node(state: ResearchState) -> dict:
    """Synthesize all research findings into a structured analysis."""
    # TODO: Implement this node
    pass


# ==============================================================
# TODO 5: Implement the Judge Node
# ==============================================================
# The judge evaluates the analysis quality on a rubric with
# behavioral anchors and returns a JSON score.
#
# Rubric (include these anchors in your prompt):
#   1-3: Missing evidence, no clear structure, fails to answer query
#   4-6: Some evidence cited, partial coverage, basic structure
#   7-8: Strong evidence, covers multiple perspectives, clear conclusions
#   9-10: Exceptional depth, novel insights, actionable recommendations
#
# Criteria to score:
#   - Correctness: Are claims supported by the research findings?
#   - Completeness: Does it address the full scope of the query?
#   - Clarity: Is the analysis well-organized and readable?
#
# Steps:
#   1. Get the LLM (very low temperature, 0.1)
#   2. Build prompt with rubric and behavioral anchors
#   3. Ask for JSON response: {"score": N, "feedback": "..."}
#   4. Parse JSON (with fallback for markdown-wrapped JSON)
#   5. Return {"quality_score": score, "judge_feedback": feedback}

def judge_node(state: ResearchState) -> dict:
    """Score the analysis quality using a rubric with behavioral anchors."""
    # TODO: Implement this node
    pass


# ==============================================================
# TODO 6: Implement the Routing Function
# ==============================================================
# After the judge scores, decide: accept or retry?
#
# Rules:
#   - score >= 7          -> "accept" (go to END)
#   - iteration >= max    -> "accept" (go to END, we've tried enough)
#   - otherwise           -> "retry"  (go back to supervisor)

def should_continue(state: ResearchState) -> str:
    """Route based on quality score and iteration count."""
    # TODO: Implement routing logic
    pass


# ==============================================================
# TODO 7: Build the Graph
# ==============================================================
# Wire all nodes together with edges and conditional routing.
#
# Graph structure:
#   START -> supervisor -> researcher -> analyst -> judge
#                                                    |
#                                           retry -> supervisor
#                                           accept -> END

def build_research_graph():
    """Build the multi-agent research graph."""
    # TODO: Create StateGraph, add nodes, add edges, compile
    pass


# ==============================================================
# Main (provided)
# ==============================================================

if __name__ == "__main__":
    print("Exercise 2: Multi-Agent Research Team")
    print("=" * 60)

    # Build the graph
    # app = build_research_graph()

    # Test query
    query = "What is the impact of AI on education?"

    print(f"\nResearch Query: {query}")
    print(f"Max Iterations: 3")
    print("-" * 60)

    # Run the graph
    # initial_state = {
    #     "query": query,
    #     "sub_questions": [],
    #     "research_findings": [],
    #     "analysis": "",
    #     "quality_score": 0,
    #     "judge_feedback": "",
    #     "iteration": 0,
    #     "max_iterations": 3,
    # }
    #
    # result = app.invoke(initial_state)
    #
    # # Display results
    # print("\n" + "=" * 60)
    # print("FINAL RESULTS")
    # print("=" * 60)
    # print(f"Iterations Used: {result['iteration']}")
    # print(f"Final Score: {result['quality_score']}/10")
    # print(f"Judge Feedback: {result['judge_feedback']}")
    # print(f"\nSub-Questions:")
    # for i, q in enumerate(result["sub_questions"], 1):
    #     print(f"  {i}. {q}")
    # print(f"\nFinal Analysis:")
    # print("-" * 60)
    # print(result["analysis"])
    # print("-" * 60)

    print("\n(Uncomment the code above after implementing all TODOs!)")
    print("\nExpected behavior:")
    print("  - Supervisor breaks query into 3-4 sub-questions")
    print("  - Researcher searches academic + news for each sub-question")
    print("  - Analyst synthesizes findings into a multi-paragraph report")
    print("  - Judge scores the report; if < 7, loops back to supervisor")
    print("  - System stops after score >= 7 or 3 iterations")
