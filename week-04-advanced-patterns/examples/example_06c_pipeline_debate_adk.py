import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6c: Pipeline & Debate Topologies using Google ADK
==========================================================
The ADK counterpart of Example 5c (LangGraph Pipeline & Debate).

This example implements two multi-agent topologies using Google ADK:

  1. SEQUENTIAL PIPELINE -- assembly line, each agent transforms output
     and passes it to the next. Like a newspaper: reporter -> editor -> publisher.

  2. DEBATE/COMMITTEE -- agents argue different perspectives, a judge
     synthesizes the best answer. Like a panel discussion.

ADK Differences from LangGraph (Example 5c):
  - Each agent is an LlmAgent with its own instruction
  - Coordination happens in PYTHON CODE (async functions), not in a graph
  - No explicit graph edges -- you orchestrate agents procedurally
  - Debate agents run with asyncio.gather for TRUE parallel execution
    (LangGraph 5c runs debaters sequentially)
  - Each agent gets a fresh session per call

Architecture:
  Pipeline: researcher_agent -> writer_agent -> editor_agent
  Debate:   [optimist, skeptic, pragmatist] --parallel--> judge

Run: python week-04-advanced-patterns/examples/example_06c_pipeline_debate_adk.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================
# Helper: Run an ADK Agent with Retries
# ==============================================================
# Each call creates a fresh session so agents don't share context
# unless we explicitly pass information between them.

async def run_agent(agent: LlmAgent, message: str, max_retries: int = 5) -> str:
    """Run an ADK agent with a message and return the response text.

    Includes retry logic for transient API failures.
    """
    for attempt in range(1, max_retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="pipeline_debate_demo",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="pipeline_debate_demo",
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

            if result_text:
                return result_text

        except Exception as e:
            if attempt < max_retries:
                wait = attempt * 10
                print(f"    [RETRY] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                return f"Error after {max_retries} attempts: {type(e).__name__}: {e}"

    return "Error: No response received from agent."


# ==============================================================
# Model Configuration
# ==============================================================

MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")


# ================================================================
# TOPOLOGY 1: SEQUENTIAL PIPELINE
# ================================================================
# Each agent takes the previous agent's output as input and refines it.
#
#   researcher -> writer -> editor -> DONE
#   (facts)       (draft)   (polish)
#
# Key insight: Pipeline topology is ideal when each stage has a
# DIFFERENT skill (research != writing != editing) and the output
# quality improves at each stage.
#
# ADK advantage: The sequential flow is natural Python -- just call
# agents one after another, passing output as input. No graph needed.

researcher_agent = LlmAgent(
    name="pipeline_researcher",
    model=MODEL,
    instruction=(
        "You are a research specialist. Given a topic, provide 5-7 key facts "
        "and data points. Output as a numbered list. Be specific with numbers "
        "and sources. Keep it under 200 words."
    ),
    tools=[],
    description="Gathers key facts and data points on a given topic.",
)

writer_agent = LlmAgent(
    name="pipeline_writer",
    model=MODEL,
    instruction=(
        "You are a skilled writer. Given research notes, write a clear, "
        "engaging 2-paragraph article. Use the facts provided but make it "
        "readable and compelling. Do NOT add facts that aren't in the research. "
        "Keep it under 200 words."
    ),
    tools=[],
    description="Transforms research notes into a well-structured article.",
)

editor_agent = LlmAgent(
    name="pipeline_editor",
    model=MODEL,
    instruction=(
        "You are a senior editor. Review and improve this article draft. "
        "Fix any grammar issues, improve sentence flow, cut unnecessary words, "
        "and ensure the opening hooks the reader. Keep the same length. "
        "Output ONLY the improved article, no commentary."
    ),
    tools=[],
    description="Polishes article drafts for grammar, flow, and clarity.",
)


async def run_pipeline(topic: str) -> dict:
    """Run the sequential pipeline: Researcher -> Writer -> Editor.

    Each agent's output becomes the next agent's input.
    This is the simplest multi-agent pattern -- just sequential calls.
    """
    print(f"\n{'='*60}")
    print(f"  PIPELINE TOPOLOGY (ADK): {topic}")
    print(f"{'='*60}")
    print("  Flow: Researcher -> Writer -> Editor")

    # --- Stage 1: Research ---
    print(f"\n  STAGE 1 -- RESEARCHER: Gathering facts...")
    research = await run_agent(
        researcher_agent,
        f"Research this topic: {topic}",
    )
    print(f"    Output: {research[:150]}...")

    # --- Stage 2: Writing ---
    print(f"\n  STAGE 2 -- WRITER: Drafting article from research...")
    draft = await run_agent(
        writer_agent,
        f"Topic: {topic}\n\nResearch notes:\n{research}",
    )
    print(f"    Output: {draft[:150]}...")

    # --- Stage 3: Editing ---
    print(f"\n  STAGE 3 -- EDITOR: Polishing draft...")
    final_article = await run_agent(
        editor_agent,
        f"Draft to edit:\n\n{draft}",
    )
    print(f"    Output: {final_article[:150]}...")

    # --- Display Result ---
    print(f"\n  {'- '*30}")
    print(f"  PIPELINE RESULT:")
    print(f"  {'- '*30}")
    print(f"\n{final_article}")

    return {
        "topic": topic,
        "research": research,
        "draft": draft,
        "final_article": final_article,
    }


# ================================================================
# TOPOLOGY 2: DEBATE / COMMITTEE
# ================================================================
# Multiple agents analyze the SAME question from different angles.
# A judge synthesizes their perspectives into a balanced conclusion.
#
#   START -> optimist   -+
#         -> skeptic    -+-> judge -> DONE
#         -> pragmatist -+
#
# Key insight: Debate topology is ideal when a question has multiple
# valid perspectives and you want a BALANCED analysis, not just one view.
#
# ADK advantage: We use asyncio.gather to run all three debaters
# TRULY in parallel. LangGraph's Example 5c runs them sequentially
# because LangGraph processes nodes one at a time. ADK's async
# approach gives real concurrency for independent agents.

optimist_agent = LlmAgent(
    name="debate_optimist",
    model=MODEL,
    instruction=(
        "You are an OPTIMIST debater. Given a topic, argue strongly for its "
        "benefits, opportunities, and positive potential. Be specific with "
        "examples and data. Present the best possible case. Keep to 100 words."
    ),
    tools=[],
    description="Argues the optimistic/pro perspective in a debate.",
)

skeptic_agent = LlmAgent(
    name="debate_skeptic",
    model=MODEL,
    instruction=(
        "You are a SKEPTIC debater. Given a topic, argue against it -- focus on "
        "risks, downsides, hidden costs, and potential failures. Be specific "
        "with examples and data. Present the strongest critique. Keep to 100 words."
    ),
    tools=[],
    description="Argues the skeptical/con perspective in a debate.",
)

pragmatist_agent = LlmAgent(
    name="debate_pragmatist",
    model=MODEL,
    instruction=(
        "You are a PRAGMATIST. Given a topic, focus on practical implementation: "
        "what would it actually take? What are the realistic timelines, costs, "
        "and prerequisites? Skip the hype and doom -- focus on what works. "
        "Keep to 100 words."
    ),
    tools=[],
    description="Argues the practical/implementation perspective in a debate.",
)

judge_agent = LlmAgent(
    name="debate_judge",
    model=MODEL,
    instruction=(
        "You are a neutral JUDGE synthesizing a debate. You received three "
        "perspectives: optimist, skeptic, and pragmatist. Your job:\n"
        "1. Acknowledge the strongest point from EACH perspective\n"
        "2. Identify where they agree and disagree\n"
        "3. Provide a balanced, nuanced conclusion\n"
        "Keep to 150 words. Be fair to all sides."
    ),
    tools=[],
    description="Synthesizes multiple debate perspectives into a balanced conclusion.",
)


async def run_debate(question: str) -> dict:
    """Run the debate: [Optimist, Skeptic, Pragmatist] in parallel -> Judge.

    The three debaters run concurrently using asyncio.gather -- this is
    a key ADK advantage over LangGraph's sequential node execution.
    The judge then synthesizes all perspectives.
    """
    print(f"\n{'='*60}")
    print(f"  DEBATE TOPOLOGY (ADK): {question}")
    print(f"{'='*60}")
    print("  Flow: [Optimist | Skeptic | Pragmatist] -> Judge")
    print("  (debaters run in PARALLEL via asyncio.gather)")

    # --- Run all three debaters in parallel ---
    print(f"\n  Running 3 debaters in parallel...")
    debater_prompt = f"Topic: {question}"

    optimist_task = run_agent(optimist_agent, debater_prompt)
    skeptic_task = run_agent(skeptic_agent, debater_prompt)
    pragmatist_task = run_agent(pragmatist_agent, debater_prompt)

    # asyncio.gather runs all three concurrently -- real parallelism!
    optimist_view, skeptic_view, pragmatist_view = await asyncio.gather(
        optimist_task,
        skeptic_task,
        pragmatist_task,
    )

    print(f"\n  OPTIMIST perspective:")
    print(f"    {optimist_view[:120]}...")
    print(f"\n  SKEPTIC perspective:")
    print(f"    {skeptic_view[:120]}...")
    print(f"\n  PRAGMATIST perspective:")
    print(f"    {pragmatist_view[:120]}...")

    # --- Judge synthesizes all perspectives ---
    print(f"\n  JUDGE: Synthesizing all perspectives...")
    judge_prompt = (
        f"Topic: {question}\n\n"
        f"OPTIMIST:\n{optimist_view}\n\n"
        f"SKEPTIC:\n{skeptic_view}\n\n"
        f"PRAGMATIST:\n{pragmatist_view}\n\n"
        "Synthesize a balanced conclusion:"
    )
    synthesis = await run_agent(judge_agent, judge_prompt)
    print(f"    Synthesis: {synthesis[:150]}...")

    # --- Display Result ---
    print(f"\n  {'- '*30}")
    print(f"  DEBATE SYNTHESIS:")
    print(f"  {'- '*30}")
    print(f"\n{synthesis}")

    return {
        "question": question,
        "optimist_view": optimist_view,
        "skeptic_view": skeptic_view,
        "pragmatist_view": pragmatist_view,
        "synthesis": synthesis,
    }


# ================================================================
# Main
# ================================================================

async def main():
    """Run both topologies and compare them."""
    print("Example 6c: Pipeline & Debate Topologies (ADK)")
    print("=" * 60)

    # --- Pipeline Demo ---
    pipeline_result = await run_pipeline(
        "The impact of artificial intelligence on healthcare"
    )

    # --- Debate Demo ---
    debate_result = await run_debate(
        "Should companies adopt AI agents for customer service?"
    )

    # --- Topology Comparison ---
    print(f"\n\n{'='*60}")
    print("  TOPOLOGY COMPARISON")
    print(f"{'='*60}")
    print("""
  SEQUENTIAL PIPELINE:
    Flow:    A -> B -> C (each stage refines)
    Best for: Content creation, data processing, ETL
    Example: Research -> Write -> Edit
    Agents:  Different skills, same data flowing through
    Output:  Single refined result

  DEBATE / COMMITTEE:
    Flow:    [A, B, C] -> Judge (parallel perspectives)
    Best for: Analysis, decision-making, risk assessment
    Example: Optimist + Skeptic + Pragmatist -> Judge
    Agents:  Same skills, different viewpoints
    Output:  Balanced synthesis of perspectives

  SUPERVISOR/WORKER (Examples 05/06):
    Flow:    Boss -> [Workers] -> Boss (delegated sub-tasks)
    Best for: Complex research, multi-part tasks
    Example: Supervisor -> Researcher + Analyst -> Judge
    Agents:  Different skills, different sub-tasks
    Output:  Combined results from parallel work

  When to use which:
    "Process this data through stages"      -> Pipeline
    "Analyze this from multiple angles"     -> Debate
    "Break this big task into pieces"       -> Supervisor/Worker
    "Route this to the right specialist"    -> Intent Router (05b/06b)
""")

    # --- LangGraph vs ADK Comparison for these topologies ---
    print(f"{'='*60}")
    print("  LangGraph vs ADK -- Pipeline & Debate Comparison")
    print(f"{'='*60}")
    print("""
  PIPELINE:
    LangGraph (5c): Graph edges define the flow (researcher -> writer -> editor)
                    State is a shared TypedDict; each node reads/writes fields.
                    Graph can be visualized, checkpointed, and resumed mid-flow.

    ADK (this):     Sequential async calls in Python code.
                    Each agent gets a fresh session; output passed as strings.
                    Simpler to write but no built-in checkpointing or replay.

  DEBATE:
    LangGraph (5c): Debaters run SEQUENTIALLY (graph processes one node at a time).
                    All perspectives stored in shared state TypedDict.
                    Judge node reads all fields from state.

    ADK (this):     Debaters run in TRUE PARALLEL via asyncio.gather.
                    Each agent runs independently with its own session.
                    Judge receives perspectives as a combined prompt string.
                    Faster execution when debaters are independent.

  Key Takeaway:
    ADK shines for the Debate topology because asyncio.gather gives
    real concurrent execution -- all three debaters call the LLM at once.
    LangGraph shines for Pipeline because checkpointing lets you resume
    from any stage if something fails mid-pipeline.
""")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
