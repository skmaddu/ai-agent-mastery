"""
Example 9: Understanding Traces -- What to Look For (Conceptual)
=================================================================
This is a CONCEPTUAL example. No LLM calls, no Phoenix server.
It runs instantly and teaches you what tracing IS, what trace data
looks like, and how to diagnose problems from traces.

WHY THIS MATTERS:
  When your agent does something wrong, you can't just "print debug"
  an LLM. You need to see the full execution trace:
    - What did the LLM decide at each step?
    - Which tools did it call, with what arguments?
    - How long did each step take? How many tokens?
    - Did the reflection loop actually improve the output?

  Tracing tools like Phoenix capture this data automatically and
  let you visualize it in a web dashboard.

FOR REAL TRACES: See example_16_phoenix_live_tracing.py which makes
  actual LLM calls and shows traces in the Phoenix UI.

Run: python week-03-basic-patterns/examples/example_09_tracing_patterns_phoenix.py
"""


# ==============================================================
# PART 1: What IS a Trace?
# ==============================================================
# A "trace" is a record of everything that happened during one
# agent execution. Think of it like a detailed receipt:
#
#   Trace: "What is 15% of the AI market?"
#   |
#   |-- Span 1: LLM Call (450ms, 150 tokens)
#   |   Decision: "I need to search for the AI market size"
#   |
#   |-- Span 2: Tool Call - search_web (120ms)
#   |   Input:  {"query": "AI market size 2026"}
#   |   Output: "AI market: $300 billion by 2027"
#   |
#   |-- Span 3: LLM Call (380ms, 180 tokens)
#   |   Decision: "Now I need to calculate 15% of $300B"
#   |
#   |-- Span 4: Tool Call - calculate (2ms)
#   |   Input:  {"expression": "300 * 0.15"}
#   |   Output: "45.0"
#   |
#   |-- Span 5: LLM Call (400ms, 120 tokens)
#       Generated final answer: "15% of $300B is $45B"
#
# Each "span" is one step. Phoenix collects all spans and shows
# them in a timeline view so you can see exactly what happened.

def explain_traces():
    """Explain what traces are and why they matter."""

    print("=" * 60)
    print("PART 1: What is a Trace?")
    print("=" * 60)

    print("""
  A trace is a record of ONE complete agent execution.
  It contains "spans" -- one for each step the agent took.

  Example trace for: "What is 15% of the AI market?"

  [Trace Start] ----------------------------------------
  |
  |-- [Span 1] LLM Call          450ms   150 tokens
  |   Decision: "I should search for AI market size"
  |
  |-- [Span 2] Tool: search_web  120ms
  |   Args:   {"query": "AI market size 2026"}
  |   Result: "AI market: $300 billion by 2027"
  |
  |-- [Span 3] LLM Call          380ms   180 tokens
  |   Decision: "Now calculate 15% of 300 billion"
  |
  |-- [Span 4] Tool: calculate   2ms
  |   Args:   {"expression": "300 * 0.15"}
  |   Result: "45.0"
  |
  |-- [Span 5] LLM Call          400ms   120 tokens
  |   Final answer: "15% of the $300B AI market is $45B"
  |
  [Trace End] ------------------------------------------

  Total: 5 spans, 1352ms, 450 tokens, 2 tool calls

  Without tracing, all you see is the final answer.
  With tracing, you see every decision the agent made.""")


# ==============================================================
# PART 2: Tracing a Reflection Loop
# ==============================================================
# A reflection loop trace shows the EVOLUTION of the output:
#   - Did the score actually improve each iteration?
#   - What feedback did the critic give?
#   - How many tokens did each iteration cost?
#   - Did it converge (hit quality gate) or time out (hit max iterations)?

def explain_reflection_traces():
    """Show what a reflection loop trace looks like."""

    print(f"\n{'='*60}")
    print("PART 2: Tracing a Reflection Loop")
    print("=" * 60)

    # This is what the trace data looks like for a 3-iteration reflection loop.
    # In Phoenix, each iteration appears as a nested span group.
    iterations = [
        {"draft_tokens": 200, "critique_tokens": 100, "score": 4,
         "gen_ms": 800, "crit_ms": 400, "issues": 3,
         "feedback": "Lacks specific examples; no statistics cited; too vague"},
        {"draft_tokens": 280, "critique_tokens": 120, "score": 6,
         "gen_ms": 900, "crit_ms": 450, "issues": 2,
         "feedback": "Better examples but statistics need sources; conclusion weak"},
        {"draft_tokens": 350, "critique_tokens": 130, "score": 8,
         "gen_ms": 1000, "crit_ms": 500, "issues": 1,
         "feedback": "Strong overall; minor: could improve opening sentence"},
    ]

    print("""
  Reflection loop trace for: "Write about AI in healthcare"

  What you see in Phoenix (each row is a span):""")

    total_tokens = 0
    total_ms = 0

    for i, it in enumerate(iterations, 1):
        tokens = it["draft_tokens"] + it["critique_tokens"]
        ms = it["gen_ms"] + it["crit_ms"]
        total_tokens += tokens
        total_ms += ms

        print(f"""
  Iteration {i}:
    [Generate]  {it['gen_ms']}ms | {it['draft_tokens']} tokens | Draft v{i}
    [Critique]  {it['crit_ms']}ms | {it['critique_tokens']} tokens | Score: {it['score']}/10
                Issues: {it['issues']} | "{it['feedback']}" """)

    print(f"""
  -------------------------------------------------------
  Summary (visible in Phoenix dashboard):
    Iterations:       {len(iterations)}
    Total tokens:     {total_tokens} (cost tracking!)
    Total time:       {total_ms}ms
    Score progression: {' -> '.join(str(it['score']) for it in iterations)}
    Converged:        Yes (score >= 7 threshold met)

  KEY INSIGHT: If score stays flat (5 -> 5 -> 5), the critique
  prompt is too vague. If tokens explode, the context is growing
  because you're passing full history instead of just the latest draft.""")


# ==============================================================
# PART 3: Tracing Tool Use
# ==============================================================
# A tool use trace shows:
#   - Which tools the LLM chose (was the choice correct?)
#   - What arguments it passed (well-formed? reasonable?)
#   - Tool execution time vs LLM thinking time
#   - Whether tool results were actually used in the answer

def explain_tool_traces():
    """Show what a tool use trace looks like."""

    print(f"\n{'='*60}")
    print("PART 3: Tracing Tool Calls")
    print("=" * 60)

    # This is the trace data for a multi-tool agent execution.
    steps = [
        {"type": "LLM",  "detail": "Decided to call search_web",     "ms": 450, "tokens": 150},
        {"type": "TOOL", "detail": "search_web({'query': 'AI market'})", "ms": 120, "tokens": 0},
        {"type": "LLM",  "detail": "Decided to call calculate",      "ms": 380, "tokens": 180},
        {"type": "TOOL", "detail": "calculate({'expr': '300*0.15'})", "ms": 2,   "tokens": 0},
        {"type": "LLM",  "detail": "Generated final answer",         "ms": 400, "tokens": 120},
    ]

    print(f'\n  Query: "What is 15% of the global AI market value?"')
    print()

    llm_ms = sum(s["ms"] for s in steps if s["type"] == "LLM")
    tool_ms = sum(s["ms"] for s in steps if s["type"] == "TOOL")

    for i, s in enumerate(steps, 1):
        print(f"    Step {i} [{s['type']:<4}] {s['detail']:<45} {s['ms']:>5}ms"
              + (f"  {s['tokens']} tok" if s['tokens'] else ""))

    print(f"""
  -------------------------------------------------------
  Time breakdown (visible in Phoenix):
    LLM thinking:  {llm_ms}ms ({llm_ms*100//(llm_ms+tool_ms)}%)  <-- usually the bottleneck
    Tool execution: {tool_ms}ms ({tool_ms*100//(llm_ms+tool_ms)}%)

  In Phoenix, you can click each span to see:
    - The exact prompt sent to the LLM
    - The exact tool arguments and return values
    - Token count and latency for each step
    - Whether the LLM used the tool result in its answer""")


# ==============================================================
# PART 4: Diagnosing Common Issues from Traces
# ==============================================================
# These are real problems you'll encounter when building agents.
# Each one is diagnosable by looking at trace data in Phoenix.

def explain_diagnosis():
    """Show how to diagnose problems from trace data."""

    print(f"\n{'='*60}")
    print("PART 4: Diagnosing Issues from Traces")
    print("=" * 60)

    issues = [
        {
            "symptom": "Reflection score never improves (5 -> 5 -> 5)",
            "what_trace_shows": "Critique text is generic ('needs improvement') with no specifics",
            "fix": "Make critic prompt list EXACT issues and HOW to fix each one",
        },
        {
            "symptom": "Agent calls same tool repeatedly with same arguments",
            "what_trace_shows": "3x search_web({'query': 'weather'}) with identical results",
            "fix": "Add tool_call_count safety limit; detect repeated calls and break the loop",
        },
        {
            "symptom": "Tool returns error but agent gives confident wrong answer",
            "what_trace_shows": "Tool span shows error, but next LLM span ignores it",
            "fix": "Make tool error messages clear; add system prompt: 'If a tool fails, say so'",
        },
        {
            "symptom": "Score jumps to 9/10 on first iteration (too easy)",
            "what_trace_shows": "Critique span gives high score with vague praise",
            "fix": "Add strict rubric to critic; require specific criteria to score above 7",
        },
        {
            "symptom": "Token count doubles each iteration",
            "what_trace_shows": "Each LLM span's input grows because full history is included",
            "fix": "Only pass latest draft + critique, not entire conversation history",
        },
    ]

    for i, issue in enumerate(issues, 1):
        print(f"\n  Problem #{i}: {issue['symptom']}")
        print(f"    Trace shows: {issue['what_trace_shows']}")
        print(f"    Fix:         {issue['fix']}")


# ==============================================================
# PART 5: How Phoenix Collects Traces (The Setup)
# ==============================================================
# This explains the code you'll see in example_16.

def explain_setup():
    """Explain how to set up Phoenix tracing in your code."""

    print(f"\n{'='*60}")
    print("PART 5: How to Set Up Phoenix Tracing")
    print("=" * 60)

    print("""
  Phoenix tracing requires 3 things:

  1. INSTALL the packages:
     pip install arize-phoenix openinference-instrumentation-langchain

  2. START Phoenix (creates a local web dashboard):
     import phoenix as px
     px.launch_app()      # Opens dashboard at http://localhost:6006

  3. INSTRUMENT LangChain (auto-captures all LLM and tool calls):
     from openinference.instrumentation.langchain import LangChainInstrumentor
     LangChainInstrumentor().instrument()

  That's it! After step 3, every LangChain call (llm.invoke,
  agent.invoke, tool calls) is automatically traced and visible
  in the Phoenix dashboard.

  No code changes needed in your agent -- Phoenix hooks into
  LangChain's internals and captures everything.

  TRY IT: Run example_16_phoenix_live_tracing.py to see real
  traces from actual LLM calls in the Phoenix dashboard.""")


# ==============================================================
# Run all parts
# ==============================================================

if __name__ == "__main__":
    print("Example 9: Understanding Traces -- What to Look For")
    print("=" * 60)
    print("(Conceptual guide -- no LLM calls, runs instantly)")
    print("(For live Phoenix traces, see example_16)")

    explain_traces()
    explain_reflection_traces()
    explain_tool_traces()
    explain_diagnosis()
    explain_setup()

    print(f"\n{'='*60}")
    print("NEXT STEP: Run example_16_phoenix_live_tracing.py to see")
    print("real traces in the Phoenix dashboard at localhost:6006.")
    print(f"{'='*60}")
