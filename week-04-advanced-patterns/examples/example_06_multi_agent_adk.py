import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6: Multi-Agent Supervisor/Worker System in ADK
=======================================================
The same multi-agent research system from Example 5, but using Google ADK.

ADK Differences from LangGraph:
  - Each agent is an LlmAgent with its own instruction and tools
  - Coordination happens in PYTHON CODE (async functions), not in a graph
  - No explicit graph edges — you orchestrate agents procedurally
  - Tools are plain functions (no @tool decorator)
  - The Runner handles each agent's tool-calling loop internally

Architecture (same as Example 5):
  1. SUPERVISOR — decomposes research question into sub-questions
  2. RESEARCHER — searches for information (has search tools)
  3. ANALYST — synthesizes findings into coherent analysis

The quality check is done by parsing a score from the analyst's
output, since ADK coordination is in Python rather than a graph loop.

Key Insight: LangGraph puts coordination in the GRAPH STRUCTURE.
ADK puts coordination in PYTHON CODE. Same pattern, different style.

Run: python week-04-advanced-patterns/examples/example_06_multi_agent_adk.py
"""

import asyncio
import json
import logging
import os
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Step 1: Simulated Research Tools (Plain Functions for ADK)
# ==============================================================
# ADK reads function name, docstring, and type hints to create
# the tool schema. No decorator needed — just plain Python.

def search_academic(query: str) -> str:
    """Search academic papers and research publications for a topic.

    Use this to find peer-reviewed research, studies, and scholarly
    analysis on a given topic.

    Args:
        query: Academic search query (e.g., 'AI impact on student learning')

    Returns:
        Academic search results as formatted text
    """
    academic_db = {
        "ai education": (
            "Academic Results for AI in Education:\n"
            "  1. Zhang et al. (2024) -- 'Adaptive Learning with AI Tutors': Students using "
            "AI-powered tutoring systems showed 23% improvement in test scores compared to "
            "traditional methods. Study covered 5,000 students across 12 universities.\n"
            "  2. Patel & Williams (2025) -- 'LLMs as Teaching Assistants': Large language models "
            "reduced teacher workload by 35% for grading and feedback tasks. Teachers reported "
            "higher job satisfaction but raised concerns about student over-reliance.\n"
            "  3. Chen (2024) -- 'Equity in AI-Assisted Education': AI tutoring tools helped "
            "close achievement gaps for disadvantaged students by 15%, but only when schools "
            "provided adequate digital infrastructure."
        ),
        "ai student learning": (
            "Academic Results for AI and Student Learning:\n"
            "  1. Morrison (2025) -- 'Personalized Learning Pathways': AI systems that adapt "
            "to individual learning styles improved retention rates by 28%.\n"
            "  2. Kumar et al. (2024) -- 'Critical Thinking in the Age of AI': Students who "
            "used AI assistants without guidance showed 12% decline in independent problem-solving."
        ),
        "ai teachers": (
            "Academic Results for AI and Teachers:\n"
            "  1. Davies & Lopez (2025) -- 'Teacher Perspectives on AI': 67% of teachers see AI "
            "as a useful tool, but 78% want more training on how to integrate it effectively.\n"
            "  2. Nakamura (2024) -- 'AI Grading Accuracy': AI grading matched human grading "
            "with 91% agreement on standardized assessments, but struggled with creative writing."
        ),
    }

    query_lower = query.lower()
    for keyword, results in academic_db.items():
        if any(word in query_lower for word in keyword.split()):
            return results

    return f"No academic results found for '{query}'. Try broader terms like 'ai education'."


def search_news(query: str) -> str:
    """Search recent news articles and reports about a topic.

    Use this to find current developments, industry trends, and
    real-world examples related to the query.

    Args:
        query: News search query (e.g., 'AI education policy 2025')

    Returns:
        News search results as formatted text
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
# Step 2: Helper to Run an ADK Agent
# ==============================================================
# In ADK, each agent needs a Runner and Session to execute.
# This helper encapsulates that boilerplate.

async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent with a message and return the response text.

    Each call creates a fresh session so agents don't share context
    unless we explicitly pass information between them.
    Includes retry logic for transient API errors (503, rate limits).
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="multi_agent_demo",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="multi_agent_demo",
                user_id="demo_user",
            )

            result_text = ""
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=message)],
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


# ==============================================================
# Step 3: Create Specialized ADK Agents
# ==============================================================
# Each agent has a focused role with specific instructions.
# NOTE: In ADK, tools are passed directly to LlmAgent.
# The researcher gets search tools; others get no tools.

MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

supervisor_agent = LlmAgent(
    name="supervisor",
    model=MODEL,
    instruction="""You are a research supervisor. Your job is to break down
a research question into 2-3 focused sub-questions that can be investigated
independently by specialized researchers.

Rules:
1. Return ONLY a numbered list of sub-questions
2. Each sub-question should be specific and researchable
3. Together, the sub-questions should cover the original topic comprehensively
4. Do NOT answer the questions — just decompose them

Format:
1. [first sub-question]
2. [second sub-question]
3. [optional third sub-question]""",
    tools=[],
    description="Decomposes research questions into focused sub-questions.",
)

# The researcher has search tools — it can gather information
researcher_agent = LlmAgent(
    name="researcher",
    model=MODEL,
    instruction="""You are a research specialist. Given a research question,
use your search tools to find relevant information from both academic
and news sources.

Rules:
1. Use search_academic for scholarly research and studies
2. Use search_news for current developments and real-world examples
3. Summarize the key findings from each source
4. Include specific data points, percentages, and citations
5. Report findings clearly — the analyst will synthesize them later""",
    tools=[search_academic, search_news],
    description="Researches topics using academic and news search tools.",
)

analyst_agent = LlmAgent(
    name="analyst",
    model=MODEL,
    instruction="""You are a research analyst. Your job is to synthesize
research findings into a clear, comprehensive analysis.

Rules:
1. Write 3-4 well-structured paragraphs
2. Identify patterns, agreements, and contradictions across sources
3. Draw actionable conclusions backed by evidence
4. Note any gaps or limitations
5. At the END of your analysis, add a quality self-assessment line:
   "QUALITY_SCORE: X/10" where X is your honest rating of the analysis quality

Write the analysis directly — no preamble or meta-commentary.""",
    tools=[],
    description="Synthesizes research findings into comprehensive analysis.",
)


# ==============================================================
# Step 4: Orchestrator — Coordinates the Agents
# ==============================================================
# In ADK, multi-agent coordination happens in Python code.
# This is the key difference from LangGraph, where coordination
# is defined in graph edges. Here, YOU write the control flow.

async def orchestrate_research(query: str, max_iterations: int = 2):
    """Orchestrate the multi-agent research pipeline.

    Flow:
      1. Supervisor decomposes the query
      2. Researcher investigates each sub-question
      3. Analyst synthesizes all findings
      4. Quality check — retry if score is low

    This function IS the supervisor/worker coordination logic.
    In LangGraph, this would be graph edges and conditional routing.
    In ADK, it's just Python.
    """
    print(f"\nResearch Query: {query}")
    print(f"Max Iterations: {max_iterations}")
    print("-" * 65)

    analysis = ""
    quality_score = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*40} Iteration {iteration} {'='*40}")

        # --- Step 1: Supervisor decomposes the query ---
        if iteration == 1:
            supervisor_prompt = f"Break down this research question into sub-questions:\n{query}"
        else:
            supervisor_prompt = (
                f"A previous research attempt on this topic scored {quality_score}/10. "
                f"Generate 2-3 NEW and DIFFERENT sub-questions to improve the research.\n"
                f"Topic: {query}\n"
                f"Previous analysis summary: {analysis[:300]}"
            )

        supervisor_response = await run_agent(supervisor_agent, supervisor_prompt)
        print(f"\n  [Supervisor] Response:")

        # Parse sub-questions from numbered list
        sub_questions = []
        for line in supervisor_response.strip().split("\n"):
            cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line.strip())
            if cleaned and len(cleaned) > 10:
                sub_questions.append(cleaned)

        # Fallback if parsing fails
        if not sub_questions:
            sub_questions = [
                f"What are the main benefits of {query}?",
                f"What are the challenges and risks of {query}?",
            ]

        for i, q in enumerate(sub_questions, 1):
            print(f"    {i}. {q}")

        # --- Step 2: Researcher investigates each sub-question ---
        all_findings = []
        for i, question in enumerate(sub_questions, 1):
            print(f"\n  [Researcher] Investigating sub-question {i}: {question[:60]}...")
            researcher_prompt = (
                f"Research this question thoroughly using your search tools:\n{question}"
            )
            finding = await run_agent(researcher_agent, researcher_prompt)
            all_findings.append(f"Sub-question {i}: {question}\nFindings: {finding}")
            print(f"    Gathered findings ({len(finding)} chars)")

        # --- Step 3: Analyst synthesizes findings ---
        combined_findings = "\n\n".join(all_findings)
        analyst_prompt = (
            f"Synthesize these research findings into a comprehensive analysis.\n\n"
            f"Original Question: {query}\n\n"
            f"Research Findings:\n{combined_findings}\n\n"
            f"Remember to include QUALITY_SCORE: X/10 at the end."
        )

        analysis = await run_agent(analyst_agent, analyst_prompt)
        print(f"\n  [Analyst] Synthesized {len(all_findings)} findings into analysis")
        print(f"    Analysis length: {len(analysis)} characters")

        # --- Step 4: Parse quality score from analyst output ---
        # The analyst self-assesses by including "QUALITY_SCORE: X/10"
        score_match = re.search(r"QUALITY_SCORE:\s*(\d+)\s*/\s*10", analysis)
        if score_match:
            quality_score = min(10, max(1, int(score_match.group(1))))
        else:
            # Fallback: assume reasonable quality
            quality_score = 7

        print(f"\n  [Quality Check] Score: {quality_score}/10")

        # --- Decide: accept or retry ---
        if quality_score >= 7:
            print(f"  [Quality Check] Score {quality_score} >= 7. Accepting analysis.")
            break
        elif iteration < max_iterations:
            print(f"  [Quality Check] Score {quality_score} < 7. Retrying with new sub-questions...")
        else:
            print(f"  [Quality Check] Score {quality_score} < 7, but max iterations reached. Accepting.")

    return {
        "query": query,
        "sub_questions": sub_questions,
        "analysis": analysis,
        "quality_score": quality_score,
        "iterations_used": iteration,
    }


# ==============================================================
# Step 5: Main — Run the Multi-Agent System
# ==============================================================

async def main():
    """Run the multi-agent research system."""
    print("Example 6: Multi-Agent Supervisor/Worker System in ADK")
    print("=" * 65)

    result = await orchestrate_research(
        query="What is the impact of AI on education?",
        max_iterations=2,
    )

    # Display results
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(f"\nOriginal Query: {result['query']}")
    print(f"Iterations Used: {result['iterations_used']}")
    print(f"Final Quality Score: {result['quality_score']}/10")

    print(f"\nSub-Questions Investigated:")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")

    print(f"\nFinal Analysis:")
    print("-" * 65)
    # Remove the QUALITY_SCORE line from display output
    display_analysis = re.sub(r"\n*QUALITY_SCORE:\s*\d+\s*/\s*10\s*$", "", result["analysis"]).strip()
    print(display_analysis)
    print("-" * 65)

    # Architecture comparison
    print(f"\n{'='*65}")
    print("LangGraph vs ADK — Multi-Agent Comparison:")
    print(f"{'='*65}")
    print("  LangGraph (Example 5):")
    print("    - Coordination defined in GRAPH EDGES")
    print("    - Conditional routing via add_conditional_edges()")
    print("    - Retry loop is a graph cycle (judge -> supervisor)")
    print("    - State is a TypedDict shared across all nodes")
    print("    - Graph can be visualized, checkpointed, resumed")
    print()
    print("  ADK (This Example):")
    print("    - Coordination defined in PYTHON CODE (async functions)")
    print("    - Retry loop is a Python for-loop with break")
    print("    - Each agent runs independently with its own session")
    print("    - Data passed between agents via function arguments")
    print("    - More flexible but less structured than a graph")
    print()
    print("  When to Use Which:")
    print("    - LangGraph: Complex workflows with branching, retries,")
    print("      checkpointing, or workflows that need visualization")
    print("    - ADK: Simpler orchestration, rapid prototyping, or when")
    print("      you want full Python control over agent coordination")
    print(f"{'='*65}")


if __name__ == "__main__":
    asyncio.run(main())
