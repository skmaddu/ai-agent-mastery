import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 6: Google A2A Protocol — Agent Cards, Task Lifecycle, Discovery
========================================================================
Topic 6 — How independent AI agents find and talk to each other.

The BIG IDEA (Feynman):
  Imagine a city where every business has a sign outside listing what
  services they offer and how to reach them.  That sign is the "Agent Card."
  When you need a service, you walk down the street reading signs until
  you find the right business.  Then you walk in, describe what you need
  (a "Task"), wait while they work on it, and pick up the result.

  A2A is exactly that — but for AI agents on the internet.

First Principles:
  1. DISCOVERY — How does Agent A find Agent B?  → Agent Cards
  2. COMMUNICATION — How do they exchange messages?  → JSON-RPC over HTTP
  3. LIFECYCLE — How does a task move from request to completion?  → State machine
  4. TRUST — How does Agent A know Agent B is legit?  → Authentication in Agent Card

Previously covered: Multi-agent systems (Week 4), middleware (Week 4-6)

NEW: Standardized inter-agent communication via the A2A protocol (Google, 2025)

Run: python week-07-mcp-a2a-synthesis/examples/example_06_a2a_protocol.py
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
import json
import uuid

# We use Pydantic for the A2A data models (production-grade validation)
from pydantic import BaseModel, Field


# ================================================================
# PART 1: WHY A2A? THE PROBLEM IT SOLVES
# ================================================================
# Without A2A, every multi-agent system is a custom snowflake:
#
#   - Agent A talks to Agent B via a custom REST API
#   - Agent C talks to Agent D via a message queue
#   - Agent E talks to Agent F via shared memory
#
# This is the same N×M problem MCP solved for tools, but for AGENTS.
#
# A2A standardizes:
#   ✅ How agents describe themselves (Agent Cards)
#   ✅ How agents find each other (Discovery via well-known URL)
#   ✅ How agents exchange work (Tasks with state machine)
#   ✅ How agents authenticate (OAuth2, API keys in Agent Card)

print("=" * 70)
print("PART 1: WHY A2A? (The Phone Directory for AI Agents)")
print("=" * 70)
print("""
Analogy: Before phone books, you had to personally know everyone's
number.  The phone book (A2A) lets you look up any business (agent)
by what they do, call them, and get service — without knowing them
personally.

The A2A protocol has 4 core pieces:
  1. Agent Card    — "Here's who I am and what I can do"
  2. Task          — "Here's what I need you to do"
  3. Message       — "Here's information for the task"
  4. Artifact      — "Here's the finished output"
""")


# ================================================================
# PART 2: AGENT CARDS — The Business Card for AI Agents
# ================================================================

print("\n" + "=" * 70)
print("PART 2: AGENT CARDS")
print("=" * 70)


class AgentSkill(BaseModel):
    """One thing an agent can do.

    Like a menu item at a restaurant — it describes one specific
    service with enough detail for another agent to decide if it's
    what they need.
    """
    id: str = Field(description="Unique skill identifier")
    name: str = Field(description="Human-readable skill name")
    description: str = Field(description="What this skill does")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    examples: list[str] = Field(
        default_factory=list,
        description="Example inputs this skill handles"
    )


class AgentAuthentication(BaseModel):
    """How to authenticate with this agent."""
    schemes: list[str] = Field(
        default_factory=lambda: ["none"],
        description="Supported auth schemes: none, api_key, oauth2, bearer"
    )
    credentials_url: Optional[str] = Field(
        default=None,
        description="Where to get credentials (if needed)"
    )


class AgentCapabilities(BaseModel):
    """What protocol features this agent supports."""
    streaming: bool = Field(default=False, description="Supports streaming responses?")
    push_notifications: bool = Field(default=False, description="Can push status updates?")
    state_transition_history: bool = Field(
        default=True,
        description="Keeps history of task state changes?"
    )


class AgentCard(BaseModel):
    """The complete identity of an A2A-compliant agent.

    Served at: GET /.well-known/agent.json

    This is like a business card + resume + menu all in one.
    Any agent can read this and know:
      - WHO you are (name, description, version)
      - WHAT you can do (skills)
      - WHERE to reach you (url)
      - HOW to authenticate (authentication)
      - WHAT protocol features you support (capabilities)
    """
    name: str = Field(description="Agent's display name")
    description: str = Field(description="What this agent does (1-2 sentences)")
    url: str = Field(description="Base URL for A2A communication")
    version: str = Field(default="1.0.0", description="Agent version (semver)")
    documentation_url: Optional[str] = Field(default=None, description="Link to docs")
    provider: Optional[str] = Field(default=None, description="Organization name")
    skills: list[AgentSkill] = Field(default_factory=list)
    authentication: AgentAuthentication = Field(default_factory=AgentAuthentication)
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"],
        description="Accepted input MIME types"
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"],
        description="Output MIME types"
    )


# Create a sample agent card
research_agent_card = AgentCard(
    name="Research Agent",
    description="Researches topics using web search and database sources, producing structured summaries with citations.",
    url="http://localhost:8001",
    version="2.1.0",
    provider="AI Agent Mastery",
    documentation_url="https://github.com/ai-agent-mastery/docs",
    skills=[
        AgentSkill(
            id="topic-research",
            name="Topic Research",
            description="Deep research on any topic with web and academic sources",
            tags=["research", "summarize", "citations"],
            examples=["Research AI in healthcare", "Summarize quantum computing advances"]
        ),
        AgentSkill(
            id="fact-check",
            name="Fact Checking",
            description="Verify claims against reliable sources",
            tags=["fact-check", "verify", "truth"],
            examples=["Is it true that GPT-4 can pass the bar exam?"]
        ),
    ],
    authentication=AgentAuthentication(schemes=["api_key"]),
    capabilities=AgentCapabilities(streaming=True, push_notifications=False),
)

print("📇 Sample Agent Card:")
print(json.dumps(research_agent_card.model_dump(), indent=2))


# ================================================================
# PART 3: TASK LIFECYCLE — The State Machine
# ================================================================

print("\n" + "=" * 70)
print("PART 3: TASK LIFECYCLE (The State Machine)")
print("=" * 70)
print("""
A Task goes through these states:

  ┌──────────┐     ┌──────────┐     ┌───────────┐
  │ submitted │────▶│ working  │────▶│ completed │
  └──────────┘     └────┬─────┘     └───────────┘
                        │
                        │ (needs more info)
                        ▼
                  ┌──────────────┐
                  │input-required│──── (client provides input) ───▶ working
                  └──────────────┘
                        │
                        │ (or)
                        ▼
                  ┌──────────┐
                  │  failed  │
                  └──────────┘

  • submitted:      Task received, queued for processing
  • working:        Agent is actively working on it
  • input-required: Agent needs more info from the client
  • completed:      Task finished successfully (has artifacts)
  • failed:         Task failed (has error message)
  • canceled:       Task was canceled by the client
""")


class TaskState(str, Enum):
    """All possible states for an A2A task."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskMessage(BaseModel):
    """A message within a task conversation.

    Like a chat message — either from the client (role='user')
    or from the agent (role='agent').
    """
    role: str = Field(description="'user' or 'agent'")
    parts: list[dict] = Field(
        description="Message parts — each has 'type' and content"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TaskArtifact(BaseModel):
    """A piece of output produced by the agent.

    Like a deliverable — could be text, a file, structured data, etc.
    """
    name: str = Field(description="Artifact name")
    parts: list[dict] = Field(description="Artifact content parts")
    mime_type: str = Field(default="text/plain")


class Task(BaseModel):
    """An A2A Task — the unit of work between agents.

    The lifecycle:
      1. Client creates a task with a message
      2. Server processes it (state: working)
      3. Server may ask for more input (state: input-required)
      4. Server produces artifacts (state: completed)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = Field(default=TaskState.SUBMITTED)
    messages: list[TaskMessage] = Field(default_factory=list)
    artifacts: list[TaskArtifact] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def transition(self, new_state: TaskState):
        """Move the task to a new state with validation."""
        # Define valid transitions
        valid_transitions = {
            TaskState.SUBMITTED: {TaskState.WORKING, TaskState.CANCELED},
            TaskState.WORKING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED},
            TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELED},
            TaskState.COMPLETED: set(),   # Terminal state
            TaskState.FAILED: set(),      # Terminal state
            TaskState.CANCELED: set(),    # Terminal state
        }

        allowed = valid_transitions.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self.state.value} → {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return f"  📋 Task {self.id[:8]}... : {old_state.value} → {new_state.value}"


# ================================================================
# PART 4: SIMULATED TASK LIFECYCLE
# ================================================================

print("\n" + "=" * 70)
print("PART 4: Simulated Task Lifecycle")
print("=" * 70)

# Create a task
task = Task(
    messages=[
        TaskMessage(
            role="user",
            parts=[{"type": "text", "text": "Research the current state of quantum computing"}]
        )
    ],
    metadata={"skill_id": "topic-research", "priority": "normal"}
)

print(f"\n  📤 Task created: {task.id[:8]}...")
print(f"     State: {task.state.value}")
print(f"     Message: {task.messages[0].parts[0]['text']}")

# Simulate the lifecycle
print(f"\n{task.transition(TaskState.WORKING)}")

# Agent adds a progress message
task.messages.append(TaskMessage(
    role="agent",
    parts=[{"type": "text", "text": "Searching web and academic databases for quantum computing research..."}]
))
print(f"  💬 Agent: Searching web and academic databases...")

# Agent needs more input
print(f"\n{task.transition(TaskState.INPUT_REQUIRED)}")
task.messages.append(TaskMessage(
    role="agent",
    parts=[{"type": "text", "text": "Should I focus on hardware advances or software/algorithm advances?"}]
))
print(f"  ❓ Agent asks: Should I focus on hardware or software advances?")

# Client provides input
task.messages.append(TaskMessage(
    role="user",
    parts=[{"type": "text", "text": "Focus on both, but emphasize practical applications."}]
))
print(f"  💬 User: Focus on both, emphasize practical applications.")

# Back to working
print(f"\n{task.transition(TaskState.WORKING)}")

# Agent completes with an artifact
task.artifacts.append(TaskArtifact(
    name="quantum_computing_summary",
    parts=[{
        "type": "text",
        "text": "# Quantum Computing: Current State (2026)\n\n"
                "## Hardware Advances\n- IBM: 1000+ qubit processors...\n"
                "## Software Advances\n- Quantum error correction breakthroughs...\n"
                "## Practical Applications\n- Drug discovery, financial modeling..."
    }],
    mime_type="text/markdown"
))

print(f"\n{task.transition(TaskState.COMPLETED)}")
print(f"  📦 Artifact: {task.artifacts[0].name}")
print(f"     Content preview: {task.artifacts[0].parts[0]['text'][:100]}...")

# Try an invalid transition (should fail)
print("\n  Testing invalid transition (completed → working):")
try:
    task.transition(TaskState.WORKING)
except ValueError as e:
    print(f"  ❌ Correctly rejected: {e}")


# ================================================================
# PART 5: AGENT DISCOVERY
# ================================================================

print("\n" + "=" * 70)
print("PART 5: Agent Discovery (The Phone Book Lookup)")
print("=" * 70)
print("""
How does Agent A find Agent B?

  1. Agent B hosts its Agent Card at:
     GET https://agent-b.example.com/.well-known/agent.json

  2. Agent A fetches that URL and reads the card.

  3. Agent A checks if Agent B has the skills it needs.

  4. If yes, Agent A sends a task to Agent B's URL.

The ".well-known" convention comes from web standards (RFC 8615).
It's like a standardized address — every agent puts their card
in the same place, so you always know where to look.
""")


class AgentDirectory:
    """A simple agent directory for discovering agents.

    In production, this could be a registry service that agents
    register with, or simply a list of known agent URLs.
    """

    def __init__(self):
        self.agents: dict[str, AgentCard] = {}

    def register(self, card: AgentCard):
        """Register an agent in the directory."""
        self.agents[card.name] = card
        print(f"  📝 Registered: {card.name} at {card.url}")

    def find_by_skill(self, skill_tag: str) -> list[AgentCard]:
        """Find agents that have a specific skill tag."""
        results = []
        for card in self.agents.values():
            for skill in card.skills:
                if skill_tag.lower() in [t.lower() for t in skill.tags]:
                    results.append(card)
                    break
        return results

    def find_by_name(self, name: str) -> Optional[AgentCard]:
        """Find an agent by exact name."""
        return self.agents.get(name)


# Demo the directory
directory = AgentDirectory()

# Register several agents
directory.register(research_agent_card)

directory.register(AgentCard(
    name="Writer Agent",
    description="Produces polished written content from research notes and outlines.",
    url="http://localhost:8002",
    skills=[
        AgentSkill(
            id="write-article", name="Article Writing",
            description="Write articles from research notes",
            tags=["writing", "articles", "content"],
            examples=["Write a blog post about AI trends"]
        ),
    ],
))

directory.register(AgentCard(
    name="Code Agent",
    description="Writes, reviews, and debugs code in multiple languages.",
    url="http://localhost:8003",
    skills=[
        AgentSkill(
            id="code-gen", name="Code Generation",
            description="Generate code from descriptions",
            tags=["code", "programming", "generation"],
            examples=["Write a Python function to sort a list"]
        ),
    ],
))

# Search for agents
print("\n  🔍 Finding agents with 'research' skill:")
results = directory.find_by_skill("research")
for card in results:
    print(f"     ✅ {card.name}: {card.description[:60]}...")

print("\n  🔍 Finding agents with 'writing' skill:")
results = directory.find_by_skill("writing")
for card in results:
    print(f"     ✅ {card.name}: {card.description[:60]}...")


# ================================================================
# PART 6: JSON-RPC MESSAGE FORMAT
# ================================================================

print("\n" + "=" * 70)
print("PART 6: A2A JSON-RPC Message Format")
print("=" * 70)
print("""
A2A uses JSON-RPC 2.0 over HTTP.  Here's what the actual HTTP
requests and responses look like:
""")

# Example: Send a task
send_task_request = {
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "tasks/send",
    "params": {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": "Research AI safety best practices"}
            ]
        }
    }
}

send_task_response = {
    "jsonrpc": "2.0",
    "id": "req-001",
    "result": {
        "id": send_task_request["params"]["id"],
        "state": "completed",
        "messages": [
            {"role": "agent", "parts": [{"type": "text", "text": "Here are the key AI safety practices..."}]}
        ],
        "artifacts": [
            {
                "name": "safety_report",
                "parts": [{"type": "text", "text": "# AI Safety Best Practices\n\n1. Input validation..."}],
                "mimeType": "text/markdown"
            }
        ]
    }
}

print("  📤 Request (POST /tasks/send):")
print(f"  {json.dumps(send_task_request, indent=4)[:500]}")
print("\n  📥 Response:")
print(f"  {json.dumps(send_task_response, indent=4)[:500]}")


# ================================================================
# PART 7: KEY TAKEAWAYS
# ================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. Agent Card = agent's identity + capabilities (served at /.well-known/agent.json)
2. Task = unit of work with a state machine (submitted → working → completed)
3. Messages = conversation within a task (user and agent roles)
4. Artifacts = deliverables produced by the agent
5. Discovery = find agents by reading their Agent Cards
6. JSON-RPC = the wire format for all A2A communication

Coming up in example_08: Building a REAL A2A server with FastAPI!
""")

print("✅ Example 06 complete!")
