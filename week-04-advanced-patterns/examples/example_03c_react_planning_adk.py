import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 3c: ReAct Planning Pattern in Google ADK
==================================================
The ADK counterpart of example_02c_react_planning.py. Implements the
ReAct (Reasoning + Acting) pattern using Google ADK.

ReAct = Reasoning + Acting (Yao et al., 2022)

The agent interleaves THINKING and ACTING in a tight loop:
  1. THOUGHT -- reason about what to know and what to do next
  2. ACTION  -- call a tool to gather information
  3. OBSERVATION -- process the tool result
  4. Repeat until the agent has enough info to answer

ADK vs LangGraph -- ReAct Comparison:
  LangGraph:
    - You build an explicit StateGraph: react_agent -> should_continue?
      -> tool_executor -> react_agent (loop) or END
    - State tracks iteration count, messages list, max_iterations
    - The routing function (should_continue) checks for tool_calls
    - You manually wire the Think->Act->Observe loop as graph edges
    - More boilerplate (~120 lines of graph wiring) but fully visible

  ADK:
    - ADK's Runner ALREADY handles the tool-calling loop internally.
      When the LLM returns a tool call, the Runner executes it and
      feeds the result back to the LLM automatically. This IS the
      Think->Act->Observe loop -- it's just built into the Runner.
    - You create ONE agent with tools and a ReAct-style instruction.
      The instruction makes the reasoning EXPLICIT (the LLM prints
      its Thought before each Action), but the loop itself is free.
    - Result: ~60% less code for the same behavior.
    - The key insight: ReAct in ADK is about PROMPTING, not plumbing.
      The Runner gives you the loop; the instruction gives you the
      visible reasoning trace.

  When to use which:
    - ADK ReAct:      Simple research/Q&A tasks where you want the
                      agent to reason step by step. Less code, fast setup.
    - LangGraph ReAct: When you need custom routing logic, iteration
                       limits enforced at the graph level, or complex
                       state transformations between steps.

Same research question and tools as example_02c:
  - web_search: Find information on any topic
  - fact_check: Verify specific claims or statistics
  - compare_data: Compare two technologies side by side

Run: python week-04-advanced-patterns/examples/example_03c_react_planning_adk.py
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
# Step 1: Tools as Plain Functions
# ==============================================================
# ADK reads function name, docstring, and type hints to create
# the tool schema automatically. No @tool decorator needed.
# These are the same simulated tools from example_02c, adapted
# for ADK (plain functions instead of @tool-decorated).

def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Use this to find facts, statistics, recent developments, or
    general information about any topic.

    Args:
        query: Search query (e.g., 'renewable energy trends 2024')

    Returns:
        Search results with relevant facts and statistics
    """
    search_db = {
        "solar energy cost": (
            "Search results for 'solar energy cost':\n"
            "  1. Solar panel costs have dropped 90% since 2010 (IRENA 2024)\n"
            "  2. Average residential solar: $2.75/watt installed in 2024\n"
            "  3. Solar is now the cheapest electricity source in most countries\n"
            "  4. Utility-scale solar LCOE: $30-50/MWh globally"
        ),
        "wind energy": (
            "Search results for 'wind energy':\n"
            "  1. Global wind capacity reached 906 GW in 2023 (GWEC)\n"
            "  2. Offshore wind growing fastest: 64 GW installed globally\n"
            "  3. Wind LCOE: $25-50/MWh onshore, $50-100/MWh offshore\n"
            "  4. Major growth in US, China, and Europe"
        ),
        "battery storage": (
            "Search results for 'battery storage':\n"
            "  1. Lithium-ion battery costs fell to $139/kWh in 2023 (BloombergNEF)\n"
            "  2. Grid-scale storage deployments up 130% year-over-year\n"
            "  3. Sodium-ion batteries emerging as cheaper alternative\n"
            "  4. Tesla Megapack and BYD dominate utility-scale market"
        ),
        "renewable energy challenge": (
            "Search results for 'renewable energy challenges':\n"
            "  1. Intermittency: solar/wind don't produce 24/7\n"
            "  2. Grid infrastructure needs $4.5 trillion upgrade globally\n"
            "  3. Supply chain constraints for critical minerals (lithium, cobalt)\n"
            "  4. Permitting and land-use conflicts slow deployment"
        ),
        "energy policy": (
            "Search results for 'energy policy':\n"
            "  1. US Inflation Reduction Act: $369B for clean energy\n"
            "  2. EU Green Deal targets 42.5% renewables by 2030\n"
            "  3. China leads in renewable deployment but still burns most coal\n"
            "  4. 130+ countries have net-zero pledges"
        ),
        "renewable market": (
            "Search results for 'renewable energy market':\n"
            "  1. Global renewable investment: $500B+ in 2023\n"
            "  2. Renewables generated 30% of global electricity in 2023\n"
            "  3. Solar + wind additions exceeded fossil fuel additions first time\n"
            "  4. Market projected to reach $2.5 trillion by 2030"
        ),
    }

    query_lower = query.lower()
    for key, result in search_db.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return result

    return (
        f"Search results for '{query}':\n"
        "  1. General information: This is a developing topic\n"
        "  2. Multiple perspectives exist on this subject\n"
        "  3. Try more specific queries for detailed results\n"
        f"  Tip: Try searching for specific aspects like 'cost', 'policy', or 'challenges'"
    )


def fact_check(claim: str) -> str:
    """Verify a specific claim or statistic.

    Use this to double-check facts, numbers, or assertions before
    including them in your final answer.

    Args:
        claim: The specific claim to verify (e.g., 'solar costs dropped 90% since 2010')

    Returns:
        Verification result with source information
    """
    verified = {
        "solar": "VERIFIED: Solar costs dropped ~89% from 2010-2024 (IRENA data). "
                 "The 90% figure is commonly cited and approximately correct.",
        "wind": "VERIFIED: Global wind capacity ~906 GW as of 2023 (GWEC). "
                "Offshore wind is the fastest-growing segment.",
        "battery": "VERIFIED: Li-ion costs at ~$139/kWh in 2023 (BloombergNEF). "
                   "Down from $1,100/kWh in 2010.",
        "investment": "VERIFIED: Global clean energy investment exceeded $500B in 2023. "
                      "Some sources report up to $620B including all clean tech.",
        "30%": "VERIFIED: Renewables generated approximately 30% of global electricity "
               "in 2023, up from 28% in 2022.",
        "grid": "PARTIALLY VERIFIED: Grid upgrade estimates vary. $4.5 trillion is "
                "one estimate; IEA suggests $600B/year needed through 2030.",
    }

    claim_lower = claim.lower()
    for key, result in verified.items():
        if key in claim_lower:
            return result

    return f"UNVERIFIED: Could not find verification for '{claim}'. Consider this unconfirmed."


def compare_data(topic_a: str, topic_b: str) -> str:
    """Compare two energy sources or technologies side by side.

    Use this to create comparative analysis between different
    renewable energy options.

    Args:
        topic_a: First topic to compare (e.g., 'solar')
        topic_b: Second topic to compare (e.g., 'wind')

    Returns:
        Side-by-side comparison with key metrics
    """
    comparisons = {
        ("solar", "wind"): (
            "Solar vs Wind Comparison:\n"
            "  Cost (LCOE): Solar $30-50/MWh vs Wind $25-50/MWh (onshore)\n"
            "  Capacity factor: Solar 15-25% vs Wind 25-45%\n"
            "  Growth rate: Solar faster (30%+ annual) vs Wind (12% annual)\n"
            "  Land use: Solar needs more area per MW\n"
            "  Intermittency: Solar predictable (daytime) vs Wind less predictable\n"
            "  Maturity: Both mature, solar declining in cost faster"
        ),
        ("solar", "battery"): (
            "Solar + Battery Storage:\n"
            "  Combined LCOE: $50-80/MWh (solar + 4hr battery)\n"
            "  Key synergy: Battery stores daytime solar for evening peak\n"
            "  Trend: Solar-plus-storage now cheaper than new gas plants\n"
            "  Challenge: Battery degradation over 10-15 year lifetime"
        ),
        ("wind", "battery"): (
            "Wind + Battery Storage:\n"
            "  Combination addresses wind intermittency\n"
            "  Offshore wind + storage emerging for baseload replacement\n"
            "  Economics improving but still more expensive than solar+storage\n"
            "  Best suited for high-wind regions (coast, plains)"
        ),
    }

    a, b = topic_a.lower().strip(), topic_b.lower().strip()
    for (ka, kb), result in comparisons.items():
        if (ka in a and kb in b) or (kb in a and ka in b):
            return result

    return f"No direct comparison data available for '{topic_a}' vs '{topic_b}'."


# ==============================================================
# Step 2: ReAct System Prompt
# ==============================================================
# This is the KEY part of ReAct in ADK. The Runner already handles
# the tool-calling loop (Think->Act->Observe) automatically. But
# without this prompt, the agent's reasoning would be IMPLICIT --
# the LLM would call tools without explaining WHY.
#
# The ReAct prompt makes the reasoning EXPLICIT by forcing the
# agent to write out its Thought before each Action. This gives
# us a visible reasoning trace we can inspect and debug.
#
# Compare to LangGraph: In LangGraph, you put this same prompt
# in a SystemMessage. The prompt content is identical -- the
# difference is only in how the loop is implemented (graph edges
# vs Runner internals).

REACT_INSTRUCTION = """You are a research agent using the ReAct (Reasoning + Acting) pattern.

For EVERY response, you MUST follow this format:

**Thought:** [Your reasoning about what you know so far and what you need to find out next]
**Action:** [Call a tool to gather information]

OR, if you have enough information to answer:

**Thought:** [Summarize what you've learned and why you can now answer]
**Final Answer:** [Your comprehensive answer based on all gathered evidence]

Rules:
1. ALWAYS start with a Thought before taking any action
2. Each thought should reference what you learned from previous observations
3. Don't repeat searches you've already done
4. After 3-4 tool calls, you should have enough info to give a final answer
5. Your final answer should cite specific facts and numbers from your research

Available tools:
- web_search: Search for information on any topic
- fact_check: Verify specific claims or statistics
- compare_data: Compare two technologies side by side"""


# ==============================================================
# Step 3: Create the ReAct Agent
# ==============================================================
# In LangGraph, ReAct requires:
#   - A StateGraph with ReActState
#   - A react_agent_node that calls llm_with_tools
#   - A tool_executor_node with ToolNode
#   - A should_continue routing function
#   - Conditional edges wiring them together
#   (~80 lines of graph construction)
#
# In ADK, it's ONE agent with tools and the right instruction:

react_agent = LlmAgent(
    name="react_researcher",
    model=os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview"),
    instruction=REACT_INSTRUCTION,
    tools=[web_search, fact_check, compare_data],
    description="Research agent that uses ReAct pattern to reason step-by-step.",
)


# ==============================================================
# Step 4: Helper to Run the Agent
# ==============================================================
# Same async runner pattern used in example_03 (Plan-Execute ADK).
# Each call creates a fresh session. Includes retry logic for
# transient API errors (503, rate limits, etc.).

async def run_agent(agent: LlmAgent, message: str, retries: int = 5) -> str:
    """Run an ADK agent and return the final response text.

    Also captures and prints intermediate events (tool calls and
    their results) so we can see the ReAct trace: each Thought,
    Action, and Observation as it happens.

    Args:
        agent: The LlmAgent to run
        message: User message to send
        retries: Number of retry attempts for transient errors

    Returns:
        The agent's final response text
    """
    for attempt in range(1, retries + 1):
        try:
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="react_demo",
                session_service=session_service,
            )

            session = await session_service.create_session(
                app_name="react_demo",
                user_id="demo_user",
            )

            # Track the reasoning trace
            iteration = 0
            result_text = ""

            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=message)],
                ),
            ):
                # --------------------------------------------------
                # Print the ReAct trace as it happens
                # --------------------------------------------------
                # The Runner fires events for each step:
                #   1. Agent produces text + tool_call -> we see the Thought + Action
                #   2. Tool executes -> we see the Observation
                #   3. Agent produces more text + tool_call -> next Thought + Action
                #   4. ...until agent produces text without tool_call -> Final Answer

                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # Text part = agent's reasoning (Thought) or final answer
                        if hasattr(part, "text") and part.text:
                            text = part.text.strip()
                            if text:
                                # Check if this looks like a new reasoning step
                                if "**Thought:**" in text or "**Action:**" in text:
                                    iteration += 1
                                    print(f"\n{'='*60}")
                                    print(f"  REACT AGENT -- Iteration {iteration}")
                                    print(f"{'='*60}")

                                for line in text.split("\n"):
                                    if line.strip():
                                        print(f"  {line.strip()[:120]}")

                        # Function call part = Action (tool invocation)
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            print(f"\n  ACTION: {fc.name}({dict(fc.args) if fc.args else {}})")

                        # Function response part = Observation (tool result)
                        if hasattr(part, "function_response") and part.function_response:
                            fr = part.function_response
                            resp_text = str(fr.response) if fr.response else ""
                            print(f"\n  OBSERVATION: {resp_text[:200]}")

                # Capture the final response
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
# Step 5: Main -- Run the ReAct Research Agent
# ==============================================================

async def main():
    """Run the ReAct agent on a renewable energy research question.

    The question is the same one used in example_02c (LangGraph ReAct)
    so you can directly compare the output and reasoning traces.
    """
    print("Example 3c: ReAct Planning Pattern in ADK")
    print("=" * 60)
    print("The agent THINKs about what it knows, ACTs to gather info,")
    print("OBSERVEs the result, and repeats until it can answer.")
    print("ADK's Runner handles the loop -- we just need the prompt.")
    print("=" * 60)

    question = (
        "What is the current state of renewable energy? Compare solar and wind "
        "costs, and explain the role of battery storage. Include specific numbers."
    )

    print(f"\nQuestion: {question}")

    # ----------------------------------------------------------
    # Run the ReAct agent
    # ----------------------------------------------------------
    # In LangGraph (example_02c), running ReAct requires:
    #   1. Build the graph (StateGraph, nodes, edges, compile)
    #   2. Invoke with initial state (messages, iteration, max_iterations)
    #   3. Extract final answer from result["messages"]
    #
    # In ADK, we just call the agent with the question.
    # The Runner handles Think->Act->Observe automatically.
    # The ReAct instruction makes the reasoning visible.

    print(f"\n{'- '*30}")
    print("  REACT REASONING TRACE")
    print(f"{'- '*30}")

    final_answer = await run_agent(react_agent, question)

    # ----------------------------------------------------------
    # Display the final answer
    # ----------------------------------------------------------
    print(f"\n\n{'#'*60}")
    print("  FINAL ANSWER")
    print(f"{'#'*60}")
    print(f"\n{final_answer}")

    # ----------------------------------------------------------
    # Framework Comparison
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("ReAct: ADK vs LangGraph Comparison")
    print(f"{'='*60}")
    print()
    print("  LangGraph ReAct (example_02c):")
    print("    - Explicit StateGraph with react_agent + tool_executor nodes")
    print("    - should_continue routing function checks for tool_calls")
    print("    - Iteration tracking in graph state (state['iteration'])")
    print("    - ~120 lines of graph wiring code")
    print("    - Full control: custom logic at every decision point")
    print("    - Great for: complex routing, checkpointing, visual debugging")
    print()
    print("  ADK ReAct (this example):")
    print("    - ONE LlmAgent with tools + ReAct instruction")
    print("    - Runner handles the tool loop internally")
    print("    - No graph, no routing function, no state management")
    print("    - ~30 lines of agent setup code")
    print("    - The prompt does the work, not the architecture")
    print("    - Great for: quick setup, simple research tasks, prototyping")
    print()
    print("  Key insight:")
    print("    ADK's Runner IS a ReAct loop. When the LLM returns a tool")
    print("    call, the Runner executes it and feeds the result back.")
    print("    That's Think->Act->Observe. You don't need to build it --")
    print("    you just need the right prompt to make reasoning visible.")
    print()
    print("  When to use which:")
    print("    - ADK ReAct:      Simple research/Q&A, fast prototyping,")
    print("                      when the built-in loop is sufficient")
    print("    - LangGraph ReAct: Custom iteration limits, complex state,")
    print("                       branching logic, production observability")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
