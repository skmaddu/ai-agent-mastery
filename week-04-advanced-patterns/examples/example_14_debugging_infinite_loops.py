import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 14: Debugging Infinite Loops in Agent Systems (No LLM)
================================================================
When an agent loops 20+ times before stopping, something is wrong.
This example shows the 5 most common infinite loop anti-patterns
and how to fix each one.

Use the STEP protocol to debug loops:
  S — Set hard caps (iterations, tokens, time)
  T — Trace with Phoenix (visualize the loop)
  E — Ensure state changes each iteration
  P — Probe the exit condition (is it reachable?)

Key Concepts (Section 12 of the Research Bible):
  - Five anti-patterns and their fixes
  - LoopDetector utility class
  - The STEP debugging protocol

Run: python week-04-advanced-patterns/examples/example_14_debugging_infinite_loops.py
"""

import time
from collections import Counter


# ================================================================
# PART 1: The Five Anti-Patterns
# ================================================================

def demo_antipattern_1_no_max():
    """Anti-pattern 1: No max_iterations limit."""
    print("=" * 60)
    print("ANTI-PATTERN 1: No Max Iterations")
    print("=" * 60)

    print("""
BAD CODE:
  def should_continue(state):
      if state["score"] >= 9:    # What if score never reaches 9?
          return "done"
      return "continue"          # Loops forever!

GOOD CODE:
  MAX_ITERATIONS = 5

  def should_continue(state):
      if state["score"] >= 9:
          return "done"
      if state["iteration"] >= MAX_ITERATIONS:
          print("[WARN] Max iterations reached, stopping")
          return "done"          # Graceful exit
      return "continue"

RULE: Every loop MUST have a hard cap. No exceptions.
      Use 5-10 for most agents, 3 for evaluator-optimizer loops.
""")

    # Demonstrate the safe version
    print("--- Safe Loop Demo ---")
    score = 3
    MAX_ITERATIONS = 5
    for i in range(1, 100):  # Outer limit is just safety
        score += 1  # Simulated improvement
        print(f"  Iteration {i}: score = {score}")
        if score >= 9:
            print(f"  ✓ Goal reached at iteration {i}")
            break
        if i >= MAX_ITERATIONS:
            print(f"  ⚠ Max iterations ({MAX_ITERATIONS}) reached, stopping gracefully")
            break
    print()


def demo_antipattern_2_plateau():
    """Anti-pattern 2: Score never improves (Ralph Wiggum problem)."""
    print("=" * 60)
    print("ANTI-PATTERN 2: Score Plateau (Ralph Wiggum Problem)")
    print("=" * 60)

    print("""
BAD CODE:
  # Agent keeps trying with no score improvement check
  while score < threshold:
      output = generate(prompt)     # Same prompt every time
      score = evaluate(output)      # Score stuck at 3.5

GOOD CODE:
  scores = []
  while score < threshold and iteration < max_iter:
      output = generate(prompt)
      score = evaluate(output)
      scores.append(score)
      # Detect plateau: no improvement for 2 rounds
      if len(scores) >= 3 and scores[-1] <= scores[-3]:
          print("[WARN] Score plateau detected!")
          # Strategy change: different prompt or higher temperature
          prompt = prompt + " Try a completely different approach."
          strategy_changes += 1
      if strategy_changes >= 2:
          print("[WARN] Multiple strategy changes failed, accepting best")
          break

RULE: Track score history. If no improvement for 2+ rounds,
      change strategy or stop.
""")

    # Demonstrate plateau detection
    print("--- Plateau Detection Demo ---")
    scores = [3.0, 3.5, 3.5, 3.5]  # Plateau!
    for i in range(len(scores)):
        is_plateau = (i >= 2 and scores[i] <= scores[i-2])
        status = " ← PLATEAU!" if is_plateau else ""
        print(f"  Iteration {i+1}: score = {scores[i]}{status}")
    print()


def demo_antipattern_3_oscillation():
    """Anti-pattern 3: Oscillating decisions."""
    print("=" * 60)
    print("ANTI-PATTERN 3: Oscillating Decisions")
    print("=" * 60)

    print("""
BAD CODE:
  # Agent flip-flops between two options
  Iteration 1: "I'll use approach A" → score 6
  Iteration 2: "Actually, approach B is better" → score 5
  Iteration 3: "No, approach A was right" → score 6
  Iteration 4: "Wait, approach B..." → score 5
  (Loops forever between 5 and 6, never reaching threshold 8)

GOOD CODE:
  # Add hysteresis: only switch if significantly better
  if new_score > best_score + 0.5:  # Minimum improvement margin
      best_output = new_output
      best_score = new_score
  else:
      print("Not enough improvement to switch, keeping current best")

  # Also: track decision history to detect oscillation
  decisions = ["A", "B", "A", "B"]
  if len(set(decisions[-4:])) <= 2 and len(decisions) >= 4:
      print("[WARN] Oscillation detected!")

RULE: Use hysteresis (minimum improvement threshold) to prevent
      flip-flopping. Track decision history.
""")

    # Demonstrate oscillation detection
    print("--- Oscillation Detection Demo ---")
    decisions = []
    scores = [6, 5, 6, 5, 6, 5]
    approaches = ["A", "B", "A", "B", "A", "B"]

    for i in range(len(scores)):
        decisions.append(approaches[i])
        oscillating = (
            len(decisions) >= 4 and
            len(set(decisions[-4:])) <= 2 and
            decisions[-1] != decisions[-2]
        )
        status = " ← OSCILLATION!" if oscillating else ""
        print(f"  Iteration {i+1}: approach {approaches[i]}, score {scores[i]}{status}")
    print()


def demo_antipattern_4_self_feeding():
    """Anti-pattern 4: Self-feeding loop (tool output becomes input)."""
    print("=" * 60)
    print("ANTI-PATTERN 4: Self-Feeding Loop")
    print("=" * 60)

    print("""
BAD CODE:
  # Agent's tool output feeds directly back as its next input
  result = search("What is AI?")
  # Agent sees result and decides to search for part of the result
  result = search("artificial intelligence machine learning")
  # Agent sees that result and searches for part of IT
  result = search("machine learning algorithms neural networks")
  # ... infinite chain of searches

GOOD CODE:
  # Track previous searches to detect loops
  previous_queries = set()

  def safe_search(query):
      if query in previous_queries:
          return "Already searched for this. Use existing results."
      previous_queries.add(query)
      return search(query)

  # Also: limit total tool calls
  MAX_TOOL_CALLS = 5
  if tool_call_count >= MAX_TOOL_CALLS:
      return "Tool call limit reached. Synthesize from existing data."

RULE: Deduplicate tool inputs. Limit total tool calls per run.
""")

    # Demonstrate deduplication
    print("--- Input Deduplication Demo ---")
    queries = ["What is AI?", "artificial intelligence", "What is AI?", "machine learning", "What is AI?"]
    seen = set()
    for q in queries:
        if q in seen:
            print(f"  Query '{q}' → SKIPPED (duplicate)")
        else:
            seen.add(q)
            print(f"  Query '{q}' → Executed ✓")
    print()


def demo_antipattern_5_redelegation():
    """Anti-pattern 5: Supervisor re-delegates the same task."""
    print("=" * 60)
    print("ANTI-PATTERN 5: Supervisor Re-delegation")
    print("=" * 60)

    print("""
BAD CODE:
  # Supervisor keeps sending the same task to the same worker
  Iteration 1: Supervisor → Researcher: "Find AI healthcare data"
  Iteration 2: Supervisor → Researcher: "Find AI healthcare data"
  (Worker returns same result, supervisor is unsatisfied, re-delegates)

GOOD CODE:
  delegation_history = []

  def supervisor_decide(task, worker_results):
      # Check if we already delegated this exact task
      task_key = (task, worker_name)
      if task_key in delegation_history:
          # Either accept the result or try a DIFFERENT worker/approach
          if len([d for d in delegation_history if d == task_key]) >= 2:
              return "accept_current_result"
          else:
              return "try_different_worker"
      delegation_history.append(task_key)
      return "delegate"

RULE: Track delegation history. If same task sent to same worker
      twice, either accept or change strategy.
""")

    # Demonstrate re-delegation detection
    print("--- Re-delegation Detection Demo ---")
    history = []
    delegations = [
        ("research AI", "researcher"),
        ("analyze data", "analyst"),
        ("research AI", "researcher"),  # Repeat!
        ("research AI", "researcher"),  # Third time!
    ]
    for task, worker in delegations:
        key = (task, worker)
        count = history.count(key)
        if count >= 2:
            print(f"  [{worker}] '{task}' → BLOCKED (delegated {count+1} times!)")
            print(f"    → Accept current result or try different worker")
        elif count == 1:
            print(f"  [{worker}] '{task}' → WARNING (second attempt)")
        else:
            print(f"  [{worker}] '{task}' → Delegated ✓")
        history.append(key)
    print()


# ================================================================
# PART 2: The LoopDetector Utility
# ================================================================
# A reusable class you can drop into any agent to detect and
# prevent infinite loops.

class LoopDetector:
    """Detects and prevents infinite loops in agent systems.

    Drop this into any agent's state to monitor loop health.

    Usage:
        detector = LoopDetector(max_iterations=5, plateau_window=2)
        for iteration in range(100):
            score = do_work()
            action = detector.check(score)
            if action == "stop":
                break
            elif action == "change_strategy":
                # Modify approach
    """

    def __init__(self, max_iterations: int = 5, plateau_window: int = 2):
        self.max_iterations = max_iterations
        self.plateau_window = plateau_window
        self.scores = []
        self.decisions = []
        self.iteration = 0
        self.strategy_changes = 0

    def check(self, score: float, decision: str = "") -> str:
        """Check loop health and return recommended action.

        Returns:
            "continue" — loop is healthy, keep going
            "change_strategy" — plateau detected, try different approach
            "stop" — should stop (max iterations or stuck)
        """
        self.iteration += 1
        self.scores.append(score)
        if decision:
            self.decisions.append(decision)

        # Check 1: Max iterations
        if self.iteration >= self.max_iterations:
            return "stop"

        # Check 2: Plateau detection
        if len(self.scores) >= self.plateau_window + 1:
            recent = self.scores[-self.plateau_window:]
            baseline = self.scores[-(self.plateau_window + 1)]
            if all(s <= baseline for s in recent):
                self.strategy_changes += 1
                if self.strategy_changes >= 2:
                    return "stop"  # Multiple strategy changes failed
                return "change_strategy"

        # Check 3: Oscillation detection
        if len(self.decisions) >= 4:
            last_4 = self.decisions[-4:]
            if len(set(last_4)) <= 2 and last_4[0] != last_4[1]:
                return "change_strategy"

        return "continue"

    def report(self) -> str:
        """Generate a summary report of loop health."""
        return (
            f"Iterations: {self.iteration}/{self.max_iterations}\n"
            f"Scores: {self.scores}\n"
            f"Strategy changes: {self.strategy_changes}\n"
            f"Best score: {max(self.scores) if self.scores else 'N/A'}"
        )


def demo_loop_detector():
    """Show the LoopDetector in action."""
    print("=" * 60)
    print("PART 2: LoopDetector Utility")
    print("=" * 60)

    # Scenario: scores plateau at 3.5
    print("\n--- Scenario: Score Plateau ---")
    detector = LoopDetector(max_iterations=8, plateau_window=2)
    simulated_scores = [2.0, 3.0, 3.5, 3.5, 3.5, 3.8, 3.8, 3.8]

    for score in simulated_scores:
        action = detector.check(score)
        print(f"  Iteration {detector.iteration}: score={score}, action={action}")
        if action == "stop":
            print(f"  → Stopping! Best score: {max(detector.scores)}")
            break
        elif action == "change_strategy":
            print(f"  → Changing strategy (attempt #{detector.strategy_changes})")

    print(f"\n  Report:\n  {detector.report()}")
    print()


# ================================================================
# PART 3: The STEP Protocol
# ================================================================

def demo_step_protocol():
    """Explain the STEP debugging protocol."""
    print("=" * 60)
    print("PART 3: The STEP Debugging Protocol")
    print("=" * 60)

    print("""
When your agent loops too many times, follow STEP:

  ╔═══════════════════════════════════════════════════════════╗
  ║  S — SET HARD CAPS                                       ║
  ║      Add max_iterations, max_tokens, max_time            ║
  ║      Every loop needs ALL THREE caps                     ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  T — TRACE WITH PHOENIX                                  ║
  ║      Look for repeated patterns: same node appearing     ║
  ║      10+ times with similar inputs. Phoenix makes this   ║
  ║      visually obvious as a long sequence of spans.       ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  E — ENSURE STATE CHANGES                                ║
  ║      If state looks identical between iterations, the    ║
  ║      agent is stuck. Check: Is feedback being used?      ║
  ║      Are tool results stale/cached? Did summarization    ║
  ║      erase needed information?                           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  P — PROBE THE EXIT CONDITION                            ║
  ║      Is the threshold actually reachable? "The output    ║
  ║      must be perfect" is NOT a valid exit condition.     ║
  ║      "Score 4+ on 3 specific dimensions" IS.             ║
  ╚═══════════════════════════════════════════════════════════╝

Pro tip: When in doubt, lower the threshold. It's better to return
a "good enough" result than to burn API budget in an infinite loop.
""")


# ================================================================
# Main: Run all demos
# ================================================================

if __name__ == "__main__":
    print()
    print("Example 14: Debugging Infinite Loops in Agent Systems")
    print("=" * 60)
    print("No LLM required — pure Python demonstration")
    print()

    demo_antipattern_1_no_max()
    demo_antipattern_2_plateau()
    demo_antipattern_3_oscillation()
    demo_antipattern_4_self_feeding()
    demo_antipattern_5_redelegation()
    demo_loop_detector()
    demo_step_protocol()

    print("=" * 60)
    print("Key Takeaway: Every loop needs max_iterations, score")
    print("tracking, and plateau detection. Use the LoopDetector")
    print("class as a starting point for your own agents.")
    print("=" * 60)
