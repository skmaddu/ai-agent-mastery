import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 16: Evaluating Multi-Agent Systems with LLM-as-Judge
==============================================================
How do you know if your multi-agent system is working well?
Traditional metrics (BLEU, ROUGE) don't work for open-ended tasks.
Instead, use an LLM as a judge to score outputs against rubrics.

This example runs a REAL multi-agent pipeline (researcher → analyst)
and then evaluates EACH agent's output with a separate LLM-as-judge.

What you'll see:
  1. Researcher agent searches for information (with tools)
  2. Analyst agent synthesizes findings into a report
  3. Judge LLM evaluates researcher output (relevance/specificity/coverage)
  4. Judge LLM evaluates analyst output (correctness/completeness/clarity)
  5. Per-agent verdicts identify which agent needs improvement

Key Concepts:
  - LLM-as-judge pattern with separate judge (different temp/prompt)
  - Rubric design with behavioral anchors (what each score means)
  - Per-agent evaluation: diagnose the source of quality issues
  - Verdict logic: PASS (avg >= 4) / MARGINAL (>= 3) / FAIL (< 3)

Run: python week-04-advanced-patterns/examples/example_16_evals_multi_agent.py
"""

import os
import re
import json
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# ================================================================
# Step 1: LLM Setup — Separate models for agents vs judge
# ================================================================

def get_llm(temperature=0.7):
    """LLM for agents (creative, higher temperature)."""
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


# Judge uses temperature=0 for consistent, deterministic scoring
judge_llm = get_llm(temperature=0)
# Agents use temperature=0.7 for creative output
agent_llm = get_llm(temperature=0.7)


# ================================================================
# Step 2: Research Tools
# ================================================================

@tool
def search_academic(query: str) -> str:
    """Search academic papers and research publications.

    Args:
        query: Research topic to search for
    """
    db = {
        "ai healthcare": (
            "Academic sources found:\n"
            "  1. Stanford 2025 study: AI diagnostics detect certain cancers with 94% accuracy\n"
            "  2. Nature Medicine review: AI drug discovery reduces timelines from 10 to 3-4 years\n"
            "  3. WHO report: 67% of healthcare institutions report data privacy concerns with AI\n"
            "  4. Lancet Digital Health: AI-assisted radiology reduced diagnostic errors by 30%"
        ),
        "climate change": (
            "Academic sources found:\n"
            "  1. IPCC 2025: Global temperatures up 1.2C since pre-industrial levels\n"
            "  2. Nature Climate: Arctic ice volume declined 40% since 1980\n"
            "  3. Science: Extreme weather events increased 5x in frequency since 1970\n"
            "  4. Annual Review: Carbon capture tech costs dropped 60% since 2020"
        ),
        "quantum computing": (
            "Academic sources found:\n"
            "  1. IBM Quantum roadmap: 1000+ qubit processors planned for 2025\n"
            "  2. Nature review: Quantum error correction progress accelerating\n"
            "  3. NIST: Post-quantum cryptography standards finalized 2024\n"
            "  4. Google: Quantum supremacy demonstrated for specific optimization problems"
        ),
    }
    for key, result in db.items():
        if key in query.lower():
            return result
    return (
        f"Academic sources for '{query}':\n"
        "  1. Multiple peer-reviewed studies found on this topic\n"
        "  2. Research is ongoing with mixed conclusions\n"
        "  3. Further investigation recommended"
    )


@tool
def search_news(query: str) -> str:
    """Search recent news articles and reports.

    Args:
        query: News topic to search for
    """
    db = {
        "ai healthcare": (
            "Recent news:\n"
            "  1. FDA approved 3 new AI diagnostic tools in Q1 2025\n"
            "  2. Hospital chains report 25% reduction in misdiagnosis with AI\n"
            "  3. Patient privacy debate: new EU regulations proposed for health AI"
        ),
        "climate change": (
            "Recent news:\n"
            "  1. COP30 pledges: 150 countries commit to 50% emissions cut by 2035\n"
            "  2. Record solar installations in 2024: 400GW added globally\n"
            "  3. Insurance costs rising 20% annually due to climate-related disasters"
        ),
        "quantum computing": (
            "Recent news:\n"
            "  1. Microsoft announces topological qubit breakthrough\n"
            "  2. China's quantum computing investment reaches $15B\n"
            "  3. First commercial quantum-secured banking network launched in Europe"
        ),
    }
    for key, result in db.items():
        if key in query.lower():
            return result
    return f"News for '{query}': General coverage found, no specific breaking stories."


# ================================================================
# Step 3: Multi-Agent Pipeline (Researcher → Analyst)
# ================================================================

class PipelineState(TypedDict):
    query: str
    researcher_output: str
    analyst_output: str
    messages: Annotated[list, add_messages]


tools = [search_academic, search_news]
researcher_llm = agent_llm.bind_tools(tools)


def researcher_node(state: PipelineState) -> dict:
    """Researcher agent: uses tools to gather facts and data."""
    print(f"\n  [RESEARCHER] Searching for: {state['query'][:60]}...")

    messages = [
        SystemMessage(content=(
            "You are a research specialist. Use the available tools to find "
            "relevant academic papers and news about the given topic. "
            "Call BOTH search_academic and search_news to get comprehensive coverage. "
            "After searching, provide a concise research summary with specific "
            "data points, sources, and statistics. Keep under 150 words."
        )),
        HumanMessage(content=f"Research this topic: {state['query']}"),
    ]

    tool_node = ToolNode(tools)
    # Tool-calling loop (max 2 rounds)
    for _ in range(2):
        try:
            response = researcher_llm.invoke(messages)
        except Exception as e:
            print(f"    [WARN] Tool call error: {str(e)[:60]}. Using LLM without tools.")
            response = agent_llm.invoke(messages)
            messages.append(response)
            break
        messages.append(response)
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    [TOOL] {tc['name']}({tc['args']})")
            try:
                tool_result = tool_node.invoke({"messages": messages})
                messages.extend(tool_result["messages"])
            except Exception as e:
                print(f"    [WARN] Tool exec error: {str(e)[:60]}")
                from langchain_core.messages import ToolMessage
                for tc in response.tool_calls:
                    messages.append(ToolMessage(
                        content="Tool error. Answer using general knowledge.",
                        tool_call_id=tc["id"],
                    ))
        else:
            break

    result = response.content or "Research completed but no summary generated."
    print(f"    Output: {result[:120]}...")
    return {"researcher_output": result, "messages": []}


def analyst_node(state: PipelineState) -> dict:
    """Analyst agent: synthesizes research into a clear report."""
    print(f"\n  [ANALYST] Synthesizing research findings...")

    messages = [
        SystemMessage(content=(
            "You are an expert analyst. Given research findings, write a clear, "
            "well-structured analysis. Include: (1) key findings with specific numbers, "
            "(2) implications or significance, (3) any caveats or limitations. "
            "Be concise but thorough. Keep under 200 words."
        )),
        HumanMessage(content=(
            f"Query: {state['query']}\n\n"
            f"Research findings:\n{state['researcher_output']}\n\n"
            "Write a comprehensive analysis based on these findings:"
        )),
    ]

    response = agent_llm.invoke(messages)
    result = response.content or "Analysis could not be generated."
    print(f"    Output: {result[:120]}...")
    return {"analyst_output": result}


def build_pipeline():
    """Build researcher → analyst pipeline."""
    graph = StateGraph(PipelineState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", END)
    return graph.compile()


# ================================================================
# Step 4: Evaluation Rubrics with Behavioral Anchors
# ================================================================
# Two separate rubrics: one for the researcher, one for the analyst.
# Each has domain-specific dimensions and clear anchors per score level.

RESEARCHER_RUBRIC = """You are an expert evaluator assessing a RESEARCH agent's output quality.

## Research Evaluation Rubric

### Relevance (1-5)
1: Findings are off-topic or irrelevant to the query
2: Somewhat related but misses the core question
3: Findings are relevant to the main topic
4: Highly relevant with targeted, on-point sources
5: Precisely relevant, addresses all aspects of the query

### Specificity (1-5)
1: Vague claims with no sources or data ("AI is useful")
2: Some claims with weak support ("studies show...")
3: Specific data points mentioned but sources unclear
4: Named sources with specific statistics and dates
5: Multiple named sources with precise data, dates, and methodology

### Coverage (1-5)
1: Only one narrow aspect of the topic addressed
2: Two aspects but misses important angles
3: Main aspects covered, reasonable breadth
4: Comprehensive — covers benefits, challenges, and trends
5: Exhaustive — multiple perspectives, edge cases, future directions

## Input
Query: {query}
Research Output: {output}

## Instructions
Score each dimension 1-5 with a one-sentence justification.
Respond in EXACT JSON format (no markdown, no code blocks):
{{"relevance": {{"score": N, "justification": "..."}}, "specificity": {{"score": N, "justification": "..."}}, "coverage": {{"score": N, "justification": "..."}}}}"""


ANALYST_RUBRIC = """You are an expert evaluator assessing an ANALYST agent's output quality.

## Analysis Evaluation Rubric

### Correctness (1-5)
1: Contains factual errors or hallucinations
2: Mostly correct but has minor inaccuracies
3: Factually correct, no obvious errors
4: Correct with specific evidence or data cited
5: Correct, well-evidenced, acknowledges limitations

### Completeness (1-5)
1: Addresses less than 25% of the question
2: Addresses some parts but misses key aspects
3: Addresses all main parts of the question
4: Comprehensive, covers main points and edge cases
5: Exhaustive, anticipates follow-up questions

### Clarity (1-5)
1: Disorganized, hard to follow
2: Understandable but poorly structured
3: Clear and logical organization
4: Well-structured with good use of examples
5: Exceptionally clear, concise, and well-illustrated

## Input
Query: {query}
Analysis Output: {output}

## Instructions
Score each dimension 1-5 with a one-sentence justification.
Respond in EXACT JSON format (no markdown, no code blocks):
{{"correctness": {{"score": N, "justification": "..."}}, "completeness": {{"score": N, "justification": "..."}}, "clarity": {{"score": N, "justification": "..."}}}}"""


# ================================================================
# Step 5: Judge Functions
# ================================================================

def parse_eval_json(response_text: str, dimensions: list) -> dict:
    """Parse judge response into scores, handling various formats."""
    text = response_text.strip()
    # Strip markdown code blocks
    if "```" in text:
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '').strip()

    try:
        data = json.loads(text)
        scores = {}
        for dim in dimensions:
            if dim in data and isinstance(data[dim], dict):
                scores[dim] = {
                    "score": int(data[dim].get("score", 3)),
                    "justification": str(data[dim].get("justification", "N/A")),
                }
            else:
                scores[dim] = {"score": 3, "justification": "Dimension not evaluated"}
        scores["average"] = round(
            sum(scores[d]["score"] for d in dimensions) / len(dimensions), 1
        )
        return scores
    except (json.JSONDecodeError, ValueError, KeyError):
        result = {d: {"score": 3, "justification": "Parse failed"} for d in dimensions}
        result["average"] = 3.0
        return result


def evaluate_researcher(query: str, researcher_output: str) -> dict:
    """Judge the researcher agent's output on relevance/specificity/coverage."""
    prompt = RESEARCHER_RUBRIC.format(query=query, output=researcher_output)
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return parse_eval_json(response.content, ["relevance", "specificity", "coverage"])


def evaluate_analyst(query: str, analyst_output: str) -> dict:
    """Judge the analyst agent's output on correctness/completeness/clarity."""
    prompt = ANALYST_RUBRIC.format(query=query, output=analyst_output)
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return parse_eval_json(response.content, ["correctness", "completeness", "clarity"])


def get_verdict(avg_score: float) -> str:
    """Determine quality verdict from average score."""
    if avg_score >= 4:
        return "PASS"
    elif avg_score >= 3:
        return "MARGINAL"
    else:
        return "FAIL"


# ================================================================
# Step 6: Run Full Pipeline + Evaluation
# ================================================================

def run_and_evaluate(query: str):
    """Run the multi-agent pipeline, then evaluate each agent."""
    print(f"\n{'#' * 60}")
    print(f"  QUERY: {query}")
    print(f"{'#' * 60}")

    # --- Phase 1: Run the multi-agent pipeline ---
    print(f"\n  PHASE 1: Running Researcher → Analyst Pipeline")
    print(f"  {'─' * 50}")

    pipeline = build_pipeline()
    result = pipeline.invoke({
        "query": query,
        "researcher_output": "",
        "analyst_output": "",
        "messages": [],
    })

    researcher_output = result["researcher_output"]
    analyst_output = result["analyst_output"]

    # --- Phase 2: Judge evaluates each agent ---
    print(f"\n  PHASE 2: LLM-as-Judge Evaluation (temp=0, strict)")
    print(f"  {'─' * 50}")

    # Evaluate researcher
    print(f"\n  [JUDGE → RESEARCHER]")
    r_eval = evaluate_researcher(query, researcher_output)
    for dim in ["relevance", "specificity", "coverage"]:
        s = r_eval[dim]
        print(f"    {dim}: {s['score']}/5 — {s['justification'][:70]}")
    r_verdict = get_verdict(r_eval["average"])
    print(f"    Average: {r_eval['average']}/5 → {r_verdict}")

    # Evaluate analyst
    print(f"\n  [JUDGE → ANALYST]")
    a_eval = evaluate_analyst(query, analyst_output)
    for dim in ["correctness", "completeness", "clarity"]:
        s = a_eval[dim]
        print(f"    {dim}: {s['score']}/5 — {s['justification'][:70]}")
    a_verdict = get_verdict(a_eval["average"])
    print(f"    Average: {a_eval['average']}/5 → {a_verdict}")

    # --- Phase 3: Diagnosis ---
    print(f"\n  PHASE 3: Diagnosis")
    print(f"  {'─' * 50}")

    # Identify which agent is the bottleneck
    if r_eval["average"] < a_eval["average"] - 0.5:
        diagnosis = "RESEARCHER is the bottleneck — improve search prompts or add tools"
    elif a_eval["average"] < r_eval["average"] - 0.5:
        diagnosis = "ANALYST is the bottleneck — improve synthesis prompt or rubric"
    elif r_eval["average"] < 3 and a_eval["average"] < 3:
        diagnosis = "BOTH agents need improvement — start with researcher (garbage in = garbage out)"
    else:
        diagnosis = "Pipeline is balanced — both agents performing at similar quality"

    pipeline_avg = round((r_eval["average"] + a_eval["average"]) / 2, 1)
    pipeline_verdict = get_verdict(pipeline_avg)

    print(f"    Researcher: {r_eval['average']}/5 ({r_verdict})")
    print(f"    Analyst:    {a_eval['average']}/5 ({a_verdict})")
    print(f"    Pipeline:   {pipeline_avg}/5 ({pipeline_verdict})")
    print(f"    Diagnosis:  {diagnosis}")

    return {
        "query": query,
        "researcher_output": researcher_output,
        "analyst_output": analyst_output,
        "researcher_eval": r_eval,
        "analyst_eval": a_eval,
        "pipeline_avg": pipeline_avg,
        "pipeline_verdict": pipeline_verdict,
        "diagnosis": diagnosis,
    }


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print()
    print("Example 16: Multi-Agent Evaluation with LLM-as-Judge")
    print("=" * 60)
    print("Runs a REAL researcher → analyst pipeline, then a separate")
    print("judge LLM (temp=0) evaluates each agent's output against")
    print("rubrics with behavioral anchors.")
    print("=" * 60)

    # Run 3 queries through the pipeline + evaluation
    queries = [
        "What is the impact of AI on healthcare?",
        "How is climate change affecting global economies?",
        "How does quantum computing differ from classical computing?",
    ]

    all_results = []
    for query in queries:
        result = run_and_evaluate(query)
        all_results.append(result)

    # --- Aggregate Report ---
    print(f"\n\n{'=' * 60}")
    print("  AGGREGATE QUALITY REPORT")
    print(f"{'=' * 60}")

    print(f"\n  {'Query':<45} {'Researcher':>10} {'Analyst':>10} {'Pipeline':>10} {'Verdict':>10}")
    print(f"  {'─' * 85}")
    for r in all_results:
        q = r["query"][:42] + "..." if len(r["query"]) > 42 else r["query"]
        print(f"  {q:<45} {r['researcher_eval']['average']:>8}/5 "
              f"{r['analyst_eval']['average']:>8}/5 "
              f"{r['pipeline_avg']:>8}/5 {r['pipeline_verdict']:>10}")

    avg_researcher = round(sum(r["researcher_eval"]["average"] for r in all_results) / len(all_results), 1)
    avg_analyst = round(sum(r["analyst_eval"]["average"] for r in all_results) / len(all_results), 1)
    avg_pipeline = round(sum(r["pipeline_avg"] for r in all_results) / len(all_results), 1)

    print(f"  {'─' * 85}")
    print(f"  {'AVERAGE':<45} {avg_researcher:>8}/5 {avg_analyst:>8}/5 {avg_pipeline:>8}/5")

    print(f"\n{'=' * 60}")
    print("  EVALUATION BEST PRACTICES")
    print(f"{'=' * 60}")
    print("""
  1. SEPARATE JUDGE: Judge LLM uses temp=0 (deterministic), while
     agents use temp=0.7 (creative). Different roles, different settings.

  2. PER-AGENT RUBRICS: Researcher is judged on relevance/specificity/
     coverage. Analyst on correctness/completeness/clarity. Different
     roles need different evaluation criteria.

  3. BEHAVIORAL ANCHORS: "Score 3 = addresses all main parts" is
     better than a vague 1-5 scale. Anchors reduce judge subjectivity.

  4. BOTTLENECK DIAGNOSIS: By evaluating each agent separately, you
     know WHERE to improve. Bad research data? Fix the researcher.
     Good data but poor synthesis? Fix the analyst.

  5. SCORE INFLATION: LLM judges are generous. A 4/5 from the LLM
     often equals 3/5 from a human. Adjust thresholds accordingly.

  6. CALIBRATE WITH HUMANS: Run 20-50 samples through both LLM judge
     and human raters. Target > 0.7 correlation.
""")
    print(f"{'=' * 60}")
