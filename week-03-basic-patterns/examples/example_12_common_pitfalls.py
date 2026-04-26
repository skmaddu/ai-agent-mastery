"""
Example 12: Common Pitfalls in Pattern Implementation
======================================================
Every agentic pattern has traps that catch beginners. This example
demonstrates each pitfall with code, shows WHY it fails, and
provides the correct implementation.

Pitfalls covered:
  1. Infinite reflection loops (no max iterations)
  2. Vague critic prompts (score never improves)
  3. Missing tool error handling (agent crashes)
  4. Context window explosion (passing full history)
  5. Tool schema mismatch (LLM sends wrong arguments)
  6. Greedy tool use (calling tools unnecessarily)

Run: python week-03-basic-patterns/examples/example_12_common_pitfalls.py
"""


# ==============================================================
# PITFALL 1: Infinite Reflection Loop
# ==============================================================

def pitfall_infinite_loop():
    """Reflection without a max iteration limit."""

    print("=" * 60)
    print("PITFALL 1: Infinite Reflection Loop")
    print("=" * 60)

    # [BAD] BAD: No stopping condition
    print("""
  [BAD] BAD CODE:
    def should_continue(state):
        if state["score"] >= 9:    # What if the LLM never scores 9?
            return "done"
        return "refine"            # Loops forever!

  What happens:
    - If the LLM critic never gives a score of 9, the loop never stops
    - Each iteration costs tokens -> your budget drains rapidly
    - After 20+ iterations, the context window fills up and errors occur

  [OK] FIX: Always add BOTH a quality threshold AND a max iteration limit.

    MAX_ITERATIONS = 5  # Hard stop after 5 refinements

    def should_continue(state):
        if state["score"] >= 7:           # Quality gate (lowered to realistic level)
            return "done"
        if state["iteration"] >= MAX_ITERATIONS:  # Safety valve
            return "done"
        return "refine"

  [TIP] Rule of thumb: Set max_iterations to 3-5 for most tasks.
     Diminishing returns kick in fast — iteration 4 rarely improves
     much over iteration 3.
""")


# ==============================================================
# PITFALL 2: Vague Critic Prompts
# ==============================================================

def pitfall_vague_critic():
    """Critic that gives unhelpful feedback."""

    print(f"{'='*60}")
    print("PITFALL 2: Vague Critic Prompts")
    print("=" * 60)

    print("""
  [BAD] BAD CRITIC PROMPT:
    "Rate this text from 1-10 and give feedback."

  What happens:
    Critic: "Score: 7. This is pretty good. Could be better."
    -> The generator has NO IDEA what to improve!
    -> Score stays at 7 forever. Loop hits max iterations.

  [OK] FIX: Specific criteria with actionable feedback.

    "Evaluate the text on these specific criteria:
     1. Does it contain at least 3 specific facts or statistics?
     2. Is every claim supported by a source or example?
     3. Is the writing clear and jargon-free?
     4. Does it directly address the given topic?

     For EACH criterion, give a score (1-3) and state exactly
     what is missing or wrong. Be specific: say 'paragraph 2
     lacks a source for the 30% claim' not 'needs more sources'."

  [TIP] The quality of your reflection loop is determined by the
     quality of your critic prompt. Invest time here.
""")


# ==============================================================
# PITFALL 3: Missing Tool Error Handling
# ==============================================================

def pitfall_tool_errors():
    """Tool that crashes instead of returning error messages."""

    print(f"{'='*60}")
    print("PITFALL 3: Missing Tool Error Handling")
    print("=" * 60)

    # [BAD] BAD: Tool crashes on bad input
    def bad_search(query):
        import requests
        response = requests.get(f"https://api.example.com/search?q={query}")
        response.raise_for_status()  # Crashes if API is down!
        return response.json()["results"]

    # [OK] GOOD: Tool returns error message
    def good_search(query: str) -> str:
        """Search the web for information."""
        try:
            # Simulated API call
            if not query or len(query) < 2:
                return f"Error: Query too short ('{query}'). Please provide a more specific query."

            # Simulated result
            return f"Results for '{query}': Found 3 relevant articles..."

        except Exception as e:
            return f"Error: Search failed for '{query}': {type(e).__name__}. Try a different query."

    print("  [BAD] BAD: Tool crashes on API error -> entire agent crashes")
    print("     bad_search('test') -> raise_for_status() -> HTTP 500 -> crash!")

    print(f"\n  [OK] GOOD: Tool returns error message -> agent can adapt")
    print(f"     good_search('') -> '{good_search('')}'")
    print(f"     good_search('AI agents') -> '{good_search('AI agents')}'")

    print(f"\n  [TIP] RULE: Tools should NEVER raise exceptions.")
    print(f"     Always return a string describing the error so the LLM")
    print(f"     can try a different approach.")


# ==============================================================
# PITFALL 4: Context Window Explosion
# ==============================================================

def pitfall_context_explosion():
    """Passing full history causes context to grow unbounded."""

    print(f"\n{'='*60}")
    print("PITFALL 4: Context Window Explosion")
    print("=" * 60)

    # ---- THE PROBLEM EXPLAINED ----
    # In a reflection loop, the LLM generates a draft, a critic reviews it,
    # and the LLM refines the draft. This repeats multiple times.
    #
    # The question is: WHAT do you send to the LLM each iteration?
    #
    # Think of it like editing an essay with a teacher:
    #   - You write a draft, teacher gives feedback, you rewrite.
    #   - For the rewrite, you only need: your latest draft + latest feedback.
    #   - You do NOT hand the teacher every previous version of the essay!
    #   - That would just be a growing pile of paper they'd have to read through.

    print("""
  THE PROBLEM:
  In a reflection loop, you call the LLM multiple times to refine output.
  Each time you call the LLM, you send it a "prompt" (input text).
  The question is: what goes into that prompt?

  ANALOGY: Editing an essay with a teacher
  -----------------------------------------
  You write a draft. Teacher gives feedback. You rewrite.
  For the rewrite, the teacher only needs:
    - Your LATEST draft (what to improve)
    - Their LATEST feedback (what to fix)
    - The original assignment (for context)
  They do NOT need every previous version of your essay!

  Two approaches:
  -----------------------------------------
  [BAD] Send ALL previous drafts + ALL previous critiques every time
        -> The prompt grows bigger each iteration (more tokens = more cost)

  [OK]  Send only the LATEST draft + LATEST critique each time
        -> The prompt stays the same size every iteration (constant cost)
""")

    # ---- SHOW THE MATH ----
    base_prompt_tokens = 200   # System prompt + topic
    draft_tokens = 300         # One draft is ~300 tokens
    critique_tokens = 150      # One critique is ~150 tokens
    per_iteration = draft_tokens + critique_tokens  # 450 tokens added per round

    good_total = base_prompt_tokens + draft_tokens + critique_tokens  # Always 650

    print(f"  Token math (each draft ~{draft_tokens} tokens, each critique ~{critique_tokens} tokens):")
    print()

    for iteration in range(1, 6):
        bad_total = base_prompt_tokens + per_iteration * iteration

        # Show exactly what's in the prompt for each approach
        print(f"    --- Iteration {iteration} ---")

        # BAD: show what's being sent
        bad_contents = f"system prompt({base_prompt_tokens})"
        for j in range(1, iteration + 1):
            bad_contents += f" + draft{j}({draft_tokens}) + critique{j}({critique_tokens})"
        print(f"    [BAD] Prompt contains: {bad_contents}")
        print(f"          Total: {bad_total:,} tokens")

        # GOOD: show what's being sent
        good_contents = (f"system prompt({base_prompt_tokens})"
                         f" + draft{iteration}({draft_tokens})"
                         f" + critique{iteration}({critique_tokens})")
        print(f"    [OK]  Prompt contains: {good_contents}")
        print(f"          Total: {good_total:,} tokens")

        if iteration > 1:
            wasted = bad_total - good_total
            print(f"          Wasted: {wasted:,} tokens on old drafts/critiques the LLM doesn't need!")
        print()

    # ---- VISUAL COMPARISON ----
    print("  Visual comparison (each # = 100 tokens):")
    print()
    for iteration in range(1, 8):
        bad_total = base_prompt_tokens + per_iteration * iteration
        bar_bad = "#" * (bad_total // 100)
        bar_good = "#" * (good_total // 100)

        if iteration == 1:
            print(f"    Iter {iteration}: [BOTH]  {bar_bad}  ({bad_total:,} tokens)")
        else:
            print(f"    Iter {iteration}: [BAD]   {bar_bad}  ({bad_total:,} tokens)")
            print(f"           [OK]    {bar_good}  ({good_total:,} tokens)")

    final_bad = base_prompt_tokens + per_iteration * 7
    cost_bad = final_bad * 0.00001   # Rough cost per token
    cost_good = good_total * 0.00001
    print(f"""
  RESULT BY ITERATION 7:
    [BAD] Full history: {final_bad:,} tokens  (~${cost_bad:.4f} per call)
    [OK]  Latest only:  {good_total} tokens   (~${cost_good:.4f} per call)
    Wasted:             {final_bad - good_total:,} tokens ({(final_bad - good_total) * 100 // final_bad}% overhead!)

  WHY THIS MATTERS:
    1. COST: You pay per token. Sending old drafts = paying for useless text.
    2. SPEED: More tokens = slower LLM response time.
    3. LIMIT: LLMs have a max context window (e.g., 8K-128K tokens).
       A bloated prompt can hit the limit and CRASH your agent.
    4. QUALITY: Irrelevant old text can confuse the LLM, making output WORSE.

  THE FIX:
    Only send the LLM what it needs for the CURRENT iteration:
      - The LATEST draft (what to revise)
      - The LATEST critique (what to fix)
      - The original topic (for context)

    Store old drafts/critiques in your STATE for logging,
    but do NOT include them in the LLM prompt.
""")


# ==============================================================
# PITFALL 5: Tool Schema Mismatch
# ==============================================================

def pitfall_schema_mismatch():
    """LLM sends wrong argument types or names."""

    print(f"{'='*60}")
    print("PITFALL 5: Tool Schema Mismatch")
    print("=" * 60)

    print("""
  [BAD] BAD TOOL DEFINITION:
    @tool
    def search(q):   # No type hint! No docstring!
        return results

  What happens:
    - LLM doesn't know the parameter name (is it "q", "query", "search_term"?)
    - LLM doesn't know the type (string? list? dict?)
    - LLM guesses wrong -> tool receives unexpected input -> error

  [OK] FIX: Full type hints + detailed docstring.

    @tool
    def search_web(query: str, max_results: int = 5) -> str:
        \"\"\"Search the web for information about a topic.

        Use this when you need current facts or data.

        Args:
            query: Search query (e.g., 'AI market size 2026')
            max_results: Number of results (1-10, default 5)
        \"\"\"

  [TIP] The LLM reads: function name, parameter names, types,
     docstring, and examples. Provide ALL of these.
""")


# ==============================================================
# PITFALL 6: Greedy Tool Use
# ==============================================================

def pitfall_greedy_tools():
    """Agent calls tools when it doesn't need to."""

    print(f"{'='*60}")
    print("PITFALL 6: Greedy Tool Use")
    print("=" * 60)

    print("""
  Problem: Agent calls search_web for "What is 2 + 2?"
    -> Wasted API call, slower response, unnecessary cost

  Why it happens:
    - Agent instruction says "always use tools" too aggressively
    - Too many tools confuse the LLM about which to use
    - Tool descriptions are too broad ("use for any question")

  [OK] FIX 1: Better instruction prompt
    "Use tools ONLY when you need information you don't have.
     For common knowledge or simple questions, answer directly."

  [OK] FIX 2: Specific tool descriptions
    [BAD] "Search for anything"
    [OK] "Search for CURRENT facts that may have changed since 2024"

  [OK] FIX 3: Fewer, more focused tools
    Instead of 10 tools -> give 3-4 clearly distinct tools
    The LLM makes better decisions with fewer options

  [TIP] Signs of greedy tool use (check in Phoenix traces):
     - Tool called but result wasn't used in the response
     - Tool called with a query the LLM already knows the answer to
     - Same tool called multiple times with slightly different queries
""")


# ==============================================================
# Summary Checklist
# ==============================================================

def summary_checklist():
    """Pre-flight checklist for pattern implementations."""

    print(f"\n{'='*60}")
    print("Pattern Implementation Checklist")
    print("=" * 60)

    checks = [
        ("Reflection loop has max_iterations?", "Prevents infinite loops and budget drain"),
        ("Critic prompt lists specific criteria?", "Vague feedback = no improvement"),
        ("All tools return strings, never crash?", "Exceptions kill the agent"),
        ("Only latest draft/critique sent to LLM?", "Prevents context window overflow"),
        ("Tools have type hints + docstrings?", "LLM needs schema to call correctly"),
        ("Agent instruction says when NOT to use tools?", "Prevents unnecessary tool calls"),
        ("Quality threshold is realistic (7-8, not 10)?", "Perfection is unreachable"),
        ("Eval pipeline exists to measure changes?", "Can't improve what you can't measure"),
    ]

    for i, (check, why) in enumerate(checks, 1):
        print(f"  [{i}] {check}")
        print(f"      Why: {why}")


if __name__ == "__main__":
    print("Example 12: Common Pitfalls in Pattern Implementation")
    print("=" * 60)

    pitfall_infinite_loop()
    pitfall_vague_critic()
    pitfall_tool_errors()
    pitfall_context_explosion()
    pitfall_schema_mismatch()
    pitfall_greedy_tools()
    summary_checklist()
