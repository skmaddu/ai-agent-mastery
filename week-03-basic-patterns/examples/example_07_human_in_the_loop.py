import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 7: Human-in-the-Loop (HITL) Patterns -- Interactive
=============================================================
This example is INTERACTIVE. The LLM proposes actions and YOU
decide whether to approve, reject, or modify them in real time.

It demonstrates 4 HITL patterns with real LLM calls:

  1. Approval Gate: LLM proposes an email, you approve/reject
  2. Confidence Threshold: LLM classifies queries, asks you when unsure
  3. Edit-Before-Execute: LLM generates a research plan, you edit it
  4. Escalation: LLM handles support tickets, escalates hard ones to you

WHY HITL MATTERS:
  Agents with tool access can take REAL actions -- send emails, make
  purchases, delete data. Mistakes are costly and irreversible.
  HITL patterns let humans provide oversight during the trust-building
  phase, or where regulations require human approval.

Run: python week-03-basic-patterns/examples/example_07_human_in_the_loop.py
"""

import os
import re
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================
# Setup LLM
# ==============================================================

provider = os.getenv("LLM_PROVIDER", "groq").lower()
if provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.3)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3)


# ==============================================================
# Helper: Get user input with a default (press Enter to accept)
# ==============================================================

def ask_user(prompt: str, default: str = "") -> str:
    """Prompt the user for input. Shows default in brackets.
    Press Enter to accept the default."""
    if default:
        response = input(f"  >> {prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"  >> {prompt}: ").strip()


# ==============================================================
# PATTERN 1: Approval Gate
# ==============================================================
# The LLM generates an email draft. You review it and decide
# whether to "send" it or reject it. The LLM does NOT send
# anything without your explicit approval.
#
# Real-world use: sending emails, making API calls, database writes,
# deploying code -- any action that's hard to undo.

def approval_gate_demo():
    """LLM drafts an email, human approves or rejects before sending."""

    print("=" * 60)
    print("PATTERN 1: Approval Gate")
    print("=" * 60)
    print("  The LLM will draft an email. You approve or reject it.\n")

    # Step 1: Ask the LLM to draft an email
    topic = ask_user("What should the email be about?", "requesting a day off next Friday")

    print("\n  [LLM] Generating email draft...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are an email assistant. Draft a professional email based on "
            "the user's request. Output ONLY the email with To, Subject, and Body fields. "
            "Keep it concise (under 100 words for the body)."
        )),
        HumanMessage(content=f"Draft an email about: {topic}"),
    ])
    draft = response.content.strip()

    # Step 2: Show the draft to the human
    print(f"\n  --- Email Draft ---")
    for line in draft.split("\n"):
        print(f"  | {line}")
    print(f"  -------------------")

    # Step 3: APPROVAL GATE -- human decides
    print()
    decision = ask_user("Approve this email? (yes/no/edit)", "yes").lower()

    if decision in ("yes", "y"):
        print("\n  [OK] APPROVED -- email would be sent.")
        print("  (In a real agent, this is where send_email() executes)")
    elif decision in ("edit", "e"):
        print("\n  [HITL] You chose to edit. In a real system, you'd modify the")
        print("  draft in a text editor or form, then the agent sends the edited version.")
        feedback = ask_user("What would you change?", "make it more formal")
        print(f"\n  [LLM] Revising with your feedback: '{feedback}'...")
        revised = llm.invoke([
            SystemMessage(content="Revise this email based on the feedback. Output ONLY the revised email."),
            HumanMessage(content=f"Original:\n{draft}\n\nFeedback: {feedback}"),
        ])
        print(f"\n  --- Revised Email ---")
        for line in revised.content.strip().split("\n"):
            print(f"  | {line}")
        print(f"  ----------------------")
        print("  [OK] Revised email would be sent.")
    else:
        print("\n  [BLOCK] REJECTED -- email discarded. No action taken.")

    print("\n  TAKEAWAY: The agent NEVER acts without human approval.")
    print("  This prevents costly mistakes like sending wrong emails.")


# ==============================================================
# PATTERN 2: Confidence Threshold
# ==============================================================
# The LLM classifies user queries. When it's confident (>= 70%),
# it acts automatically. When unsure, it asks YOU for guidance.
#
# Real-world use: email classification, ticket routing, content
# moderation -- where some cases are obvious and others aren't.

def confidence_threshold_demo():
    """LLM classifies queries, asks human only when confidence is low."""

    print(f"\n{'='*60}")
    print("PATTERN 2: Confidence Threshold")
    print("=" * 60)
    print("  The LLM classifies queries. It auto-handles confident ones")
    print("  and asks YOU when it's unsure.\n")

    THRESHOLD = 70  # Below this percentage -> ask human

    # Queries for the LLM to classify
    queries = [
        "How do I reset my password?",
        "I want to cancel my subscription and get a full refund for the last 3 months",
        "What are your business hours?",
        "Your product ruined my carpet and I'm considering legal action",
    ]

    print(f"  Confidence threshold: {THRESHOLD}%")
    print(f"  Categories: billing, account, general, legal, complaint\n")

    for i, query in enumerate(queries, 1):
        print(f"  Query {i}: \"{query}\"")

        # Ask LLM to classify with confidence
        response = llm.invoke([
            SystemMessage(content=(
                "Classify this customer query into ONE category and give a confidence percentage.\n"
                "Categories: billing, account, general, legal, complaint\n\n"
                "Respond in EXACTLY this format (no other text):\n"
                "CATEGORY: <category>\n"
                "CONFIDENCE: <number>%\n"
                "REASON: <brief reason>"
            )),
            HumanMessage(content=query),
        ])

        result = response.content.strip()

        # Parse the LLM's classification
        category = "unknown"
        confidence = 50
        reason = ""
        for line in result.split("\n"):
            line_clean = line.strip()
            if line_clean.upper().startswith("CATEGORY:"):
                category = line_clean.split(":", 1)[1].strip().lower()
            elif line_clean.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = int(re.search(r'\d+', line_clean.split(":", 1)[1]).group())
                except (AttributeError, ValueError):
                    confidence = 50
            elif line_clean.upper().startswith("REASON:"):
                reason = line_clean.split(":", 1)[1].strip()

        if confidence >= THRESHOLD:
            # High confidence -> auto-handle
            print(f"    [AUTO] {category} ({confidence}%) -- {reason}")
            print(f"    -> Auto-routing to {category} team\n")
        else:
            # Low confidence -> ask human
            print(f"    [HITL] {category}? ({confidence}%) -- {reason}")
            human_category = ask_user(
                f"What category is this? (billing/account/general/legal/complaint)",
                category
            )
            print(f"    -> Human says: {human_category}. Routing accordingly.\n")

    print("  TAKEAWAY: The agent handles clear cases (90%+ of volume)")
    print("  automatically, and only bothers you with ambiguous ones.")


# ==============================================================
# PATTERN 3: Edit-Before-Execute
# ==============================================================
# The LLM generates a multi-step research plan. You can review
# each step, add/remove/modify steps, then the agent "executes"
# the plan you approved.
#
# Real-world use: research plans, data pipelines, deployment
# checklists -- any multi-step workflow where context matters.

def edit_before_execute_demo():
    """LLM generates a plan, human reviews and edits before execution."""

    print(f"\n{'='*60}")
    print("PATTERN 3: Edit-Before-Execute")
    print("=" * 60)
    print("  The LLM generates a research plan. You can edit it.\n")

    topic = ask_user("What topic should the agent research?", "impact of AI on jobs")

    # Step 1: LLM generates a plan
    print(f"\n  [LLM] Generating research plan for: '{topic}'...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research planning assistant. Generate a step-by-step research plan.\n"
            "Output EXACTLY 4-5 numbered steps, each on its own line, in this format:\n"
            "1. [action] description\n"
            "Actions can be: [search], [analyze], [compare], [summarize]\n"
            "Keep each step to one line."
        )),
        HumanMessage(content=f"Create a research plan for: {topic}"),
    ])
    plan = response.content.strip()

    # Step 2: Show plan to human
    print(f"\n  --- Proposed Plan ---")
    steps = [line.strip() for line in plan.split("\n") if line.strip() and line.strip()[0].isdigit()]
    for step in steps:
        print(f"  | {step}")
    print(f"  ---------------------")

    # Step 3: Human reviews
    print(f"\n  You can:")
    print(f"    - Press Enter to approve the plan as-is")
    print(f"    - Type 'add <step>' to add a step  (e.g., 'add validation step')")
    print(f"    - Type 'remove <number>' to remove  (e.g., 'remove 3')")
    print(f"    - Type 'done' when finished editing")

    modified = list(steps)
    while True:
        edit = ask_user("Edit (or Enter to approve)", "done").strip()
        edit_lower = edit.lower()
        if edit_lower in ("done", ""):
            break
        elif edit_lower.startswith("add"):
            # Support both "add: step text" and "add step text"
            rest = edit[3:].lstrip(": ").strip()
            if rest:
                modified.append(f"{len(modified)+1}. [search] {rest}")
                print(f"    + Added step {len(modified)}: {rest}")
            else:
                print(f"    (Specify what to add, e.g., 'add validation step')")
        elif edit_lower.startswith("remove"):
            rest = edit[6:].lstrip(": ").strip()
            try:
                num = int(rest) - 1
                if 0 <= num < len(modified):
                    removed = modified.pop(num)
                    removed_text = re.sub(r'^\d+\.?\s*', '', removed)
                    print(f"    - Removed step {num+1}: {removed_text[:80]}")
                else:
                    print(f"    (Invalid step number. Valid: 1-{len(modified)})")
            except ValueError:
                print(f"    (Specify step number, e.g., 'remove 2')")
        else:
            # Treat any other text as a step to add (user-friendly)
            modified.append(f"{len(modified)+1}. [search] {edit}")
            print(f"    + Added step {len(modified)}: {edit}")

    # Step 4: "Execute" the approved plan
    print(f"\n  --- Final Plan (approved) ---")
    for i, step in enumerate(modified, 1):
        # Renumber
        text = re.sub(r'^\d+\.?\s*', '', step)
        print(f"  | {i}. {text}")
    print(f"  -----------------------------")
    print(f"  [OK] Executing plan with {len(modified)} steps...")
    print(f"  (In a real agent, each step would trigger actual tool calls)")

    print(f"\n  TAKEAWAY: The human shapes the plan BEFORE execution.")
    print(f"  The agent does the work, but the human sets the direction.")


# ==============================================================
# PATTERN 4: Escalation
# ==============================================================
# The LLM handles simple customer questions automatically.
# When it encounters something complex, sensitive, or risky,
# it escalates to YOU (the human agent).
#
# Real-world use: customer support, incident response, content
# review -- where some cases need human judgment.

def escalation_demo():
    """LLM handles simple questions, escalates complex ones to human."""

    print(f"\n{'='*60}")
    print("PATTERN 4: Escalation")
    print("=" * 60)
    print("  The LLM handles customer queries. Complex ones go to you.\n")

    tickets = [
        "How do I change my shipping address?",
        "I was charged twice for order #12345 and I need an immediate refund",
        "What's the return policy?",
        "I'm going to report you to the BBB unless someone fixes this NOW",
    ]

    for i, ticket in enumerate(tickets, 1):
        print(f"  Ticket {i}: \"{ticket}\"")

        # LLM decides: handle or escalate?
        response = llm.invoke([
            SystemMessage(content=(
                "You are a customer support triage agent. For each query, decide:\n"
                "- HANDLE: if it's a simple, routine question you can answer\n"
                "- ESCALATE: if it involves money, legal threats, anger, or complexity\n\n"
                "Respond in EXACTLY this format:\n"
                "DECISION: HANDLE or ESCALATE\n"
                "REASON: <brief reason>\n"
                "RESPONSE: <your response if HANDLE, or 'N/A' if ESCALATE>"
            )),
            HumanMessage(content=ticket),
        ])

        result = response.content.strip()

        # Parse decision
        decision = "ESCALATE"
        reason = ""
        bot_response = ""
        for line in result.split("\n"):
            line_clean = line.strip()
            if line_clean.upper().startswith("DECISION:"):
                decision = "ESCALATE" if "ESCALATE" in line_clean.upper() else "HANDLE"
            elif line_clean.upper().startswith("REASON:"):
                reason = line_clean.split(":", 1)[1].strip()
            elif line_clean.upper().startswith("RESPONSE:"):
                bot_response = line_clean.split(":", 1)[1].strip()

        if decision == "HANDLE":
            print(f"    [AUTO] Bot handled: {reason}")
            print(f"    Bot reply: \"{bot_response[:120]}\"")
        else:
            print(f"    [ALERT] ESCALATED to you: {reason}")
            human_response = ask_user("Your response to the customer", "Let me look into this for you")
            print(f"    Human reply: \"{human_response}\"")
        print()

    print("  TAKEAWAY: The agent handles routine queries (80%+ of volume)")
    print("  and only escalates cases that need human judgment or empathy.")


# ==============================================================
# Summary
# ==============================================================

def summary():
    """Compare all four HITL patterns."""

    print(f"\n{'='*60}")
    print("Summary: Choosing the Right HITL Pattern")
    print("=" * 60)

    patterns = [
        ("Approval Gate",       "Every action reviewed",         "Emails, purchases, deletes"),
        ("Confidence Threshold", "Only uncertain cases reviewed", "Classification, routing"),
        ("Edit-Before-Execute",  "Human modifies the plan",      "Research, multi-step tasks"),
        ("Escalation",           "Complex cases go to human",    "Support, incident response"),
    ]

    print(f"\n  {'Pattern':<24} {'When Human Intervenes':<30} {'Best For'}")
    print(f"  {'-'*24} {'-'*30} {'-'*30}")
    for name, scope, best_for in patterns:
        print(f"  {name:<24} {scope:<30} {best_for}")

    print(f"\n  In practice, agents often COMBINE these patterns:")
    print(f"    - Approval gate for sends/deletes + confidence threshold for classification")
    print(f"    - Edit-before-execute for plans + escalation for edge cases")


# ==============================================================
# Run all demos
# ==============================================================

if __name__ == "__main__":
    print("Example 7: Human-in-the-Loop (HITL) Patterns -- Interactive")
    print("=" * 60)
    print("This example is INTERACTIVE -- you'll be prompted for input.\n")

    approval_gate_demo()
    confidence_threshold_demo()
    edit_before_execute_demo()
    escalation_demo()
    summary()

    print(f"\n{'='*60}")
    print("Done! You just experienced all 4 HITL patterns live.")
    print("=" * 60)
