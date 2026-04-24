import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Exercise 2: A2A-Compliant Agent Server and Client
====================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 2 hours

Task:
Build an A2A-compliant agent server that serves an Agent Card, accepts
tasks via JSON-RPC, processes them with an LLM, and returns structured
results.  Then build a client that discovers the agent and sends tasks.

The system should:
  1. Serve an Agent Card at GET /.well-known/agent.json
  2. Accept tasks via POST / (JSON-RPC method: tasks/send)
  3. Process tasks using an LLM (research/QA skill)
  4. Track task state transitions (submitted → working → completed/failed)
  5. Return results with artifacts
  6. Include Phoenix tracing for task lifecycle events

Architecture:
  ┌─────────────┐         ┌──────────────────────┐
  │  A2A Client │         │  A2A Server (FastAPI) │
  │             │──GET───▶│  /.well-known/agent   │
  │             │──POST──▶│  /  (JSON-RPC)        │
  │             │◀────────│  LLM Processing       │
  └─────────────┘         └──────────────────────┘

Instructions:
  Complete the 8 TODOs below.

Hints:
  - Study example_06_a2a_protocol.py for the data models
  - Study example_08_a2a_adk_agents.py for the FastAPI implementation
  - The server runs in a background thread so client can test it
  - Use httpx.AsyncClient for the client HTTP calls

Run: python week-07-mcp-a2a-synthesis/exercises/exercise_02_a2a_protocol.py
"""

import os
import json
import uuid
import asyncio
import threading
import time
from typing import Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from pydantic import BaseModel, Field
from enum import Enum


# ================================================================
# LLM Setup
# ================================================================

def get_llm(temperature=0.3):
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


# ================================================================
# Phoenix Tracing (optional)
# ================================================================

try:
    from config.phoenix_config import setup_tracing
    setup_tracing()
    PHOENIX_AVAILABLE = True
except Exception:
    PHOENIX_AVAILABLE = False


# ================================================================
# TODO 1: Define the AgentCard Pydantic Model
# ================================================================
# Create an AgentCard model with:
#   - name: str
#   - description: str
#   - url: str
#   - version: str (default "1.0.0")
#   - skills: list of dicts with id, name, description, tags
#   - capabilities: dict with streaming (bool), pushNotifications (bool)
#
# Hint: Study example_06 for the full Agent Card structure.
#       Keep it simple — just the fields listed above.

class AgentCard(BaseModel):
    """The agent's identity — served at /.well-known/agent.json"""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 2: Define the Task State Machine
# ================================================================
# Create:
#   - TaskState enum with: submitted, working, input_required, completed, failed, canceled
#   - Task model with: id, state, messages (list), artifacts (list), metadata (dict)
#   - A method to validate state transitions (e.g., submitted → working is valid,
#     completed → working is invalid)
#
# Valid transitions:
#   submitted → working, canceled
#   working → completed, failed, input_required
#   input_required → working, canceled
#   completed/failed/canceled → (none, terminal states)

class TaskState(str, Enum):
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


class Task(BaseModel):
    """An A2A task with state machine."""
    # --- YOUR CODE HERE ---
    pass
    # --- END YOUR CODE ---


# ================================================================
# TODO 3: Implement the Agent Card Endpoint
# ================================================================
# Create a FastAPI app with a GET endpoint at /.well-known/agent.json
# that returns the agent card.
#
# The agent should have:
#   - name: "Research Assistant"
#   - description: "Answers questions and researches topics"
#   - url: "http://localhost:8010"
#   - skills: [{id: "qa", name: "Question Answering", ...}]
#
# Hint: Use FastAPI's @app.get("/.well-known/agent.json")

def create_a2a_app(agent_card_data: dict) -> "FastAPI":
    """Create the FastAPI app with A2A endpoints."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="A2A Agent")

    # Store tasks in memory
    tasks: dict[str, dict] = {}

    # --- YOUR CODE HERE ---
    # @app.get("/.well-known/agent.json")
    # async def get_agent_card():
    #     ...
    # --- END YOUR CODE ---

    # ================================================================
    # TODO 4: Implement the Task Handler
    # ================================================================
    # Create a POST endpoint at "/" that handles JSON-RPC requests.
    # It should:
    #   1. Parse the JSON-RPC request body
    #   2. Route to handlers based on method:
    #      - "tasks/send" → create and process a task
    #      - "tasks/get" → retrieve a task by ID
    #   3. Return JSON-RPC response format
    #
    # Hint: The request body looks like:
    #   {"jsonrpc": "2.0", "id": "...", "method": "tasks/send", "params": {...}}

    # --- YOUR CODE HERE ---
    # @app.post("/")
    # async def handle_jsonrpc(request: Request):
    #     body = await request.json()
    #     method = body.get("method")
    #     ...
    # --- END YOUR CODE ---

    # ================================================================
    # TODO 5: Implement Task State Transitions
    # ================================================================
    # When processing a task:
    #   1. Set state to "working"
    #   2. Call the LLM with the user's message
    #   3. On success: set state to "completed", add artifact
    #   4. On failure: set state to "failed", add error message
    #
    # Hint: Extract user text from params.message.parts[0].text
    #       Use get_llm() to create the LLM and invoke it

    # --- YOUR CODE HERE ---
    # async def process_task(task_data: dict, user_text: str) -> dict:
    #     ...
    # --- END YOUR CODE ---

    @app.get("/health")
    async def health():
        return {"status": "healthy", "tasks_count": len(tasks)}

    return app


# ================================================================
# TODO 6: Implement the A2A Client
# ================================================================
# Create a class that:
#   1. discover() — GET /.well-known/agent.json, return the card
#   2. send_task(message_text) — POST / with JSON-RPC tasks/send
#   3. get_task(task_id) — POST / with JSON-RPC tasks/get
#
# Hint: Use httpx.AsyncClient for HTTP calls
#       Study example_08 for the client implementation

class A2AClient:
    """Client for communicating with an A2A agent."""

    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip("/")
        self.agent_card = None

    async def discover(self) -> dict:
        """Discover the agent by fetching its Agent Card."""
        # --- YOUR CODE HERE ---
        pass
        # --- END YOUR CODE ---

    async def send_task(self, message_text: str) -> dict:
        """Send a task to the agent."""
        # --- YOUR CODE HERE ---
        pass
        # --- END YOUR CODE ---

    async def get_task(self, task_id: str) -> dict:
        """Get the status of a task."""
        # --- YOUR CODE HERE ---
        pass
        # --- END YOUR CODE ---


# ================================================================
# TODO 7: Add Error Handling
# ================================================================
# Add error handling throughout:
#   1. In the server: handle LLM call failures gracefully (set task to "failed")
#   2. In the client: handle network errors (connection refused, timeout)
#   3. In the client: handle invalid JSON-RPC responses
#
# Hint: Wrap httpx calls in try/except httpx.RequestError
#       Wrap LLM calls in try/except Exception

# --- YOUR CODE HERE ---
# Add try/except blocks to the functions above
# --- END YOUR CODE ---


# ================================================================
# TODO 8: Add Phoenix Tracing Spans
# ================================================================
# Add tracing at task lifecycle events:
#   1. When a task is created (submitted)
#   2. When processing starts (working)
#   3. When processing completes (completed/failed)
#   4. When the client sends a request
#
# Hint: If PHOENIX_AVAILABLE, use a simple timing wrapper:
#   start = time.time()
#   ... do work ...
#   duration = time.time() - start
#   print(f"  📊 Span [{event_name}]: {duration*1000:.0f}ms")

# --- YOUR CODE HERE ---
# Add tracing calls in the appropriate locations above
# --- END YOUR CODE ---


# ================================================================
# DEMO: Run Server + Client
# ================================================================

async def demo():
    """Run the A2A server and test it with the client."""
    import uvicorn

    print("=" * 70)
    print("A2A Agent Server + Client Demo")
    print("=" * 70)

    # Create agent card data
    agent_card_data = {
        "name": "Research Assistant",
        "description": "Answers questions and researches topics with detailed responses",
        "url": "http://localhost:8010",
        "version": "1.0.0",
        "skills": [
            {"id": "qa", "name": "Question Answering", "description": "Answer questions", "tags": ["qa"]},
            {"id": "research", "name": "Research", "description": "Research topics", "tags": ["research"]},
        ],
        "capabilities": {"streaming": False, "pushNotifications": False},
    }

    # Start server
    print("\n  🚀 Starting A2A server on port 8010...")
    app = create_a2a_app(agent_card_data)

    if app is None:
        print("  ❌ create_a2a_app() returned None — complete the TODOs first!")
        return

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "localhost", "port": 8010, "log_level": "warning"},
        daemon=True,
    )
    server_thread.start()
    await asyncio.sleep(2)

    # Test with client
    client = A2AClient("http://localhost:8010")

    # Step 1: Discover
    print("\n  🔍 Discovering agent...")
    card = await client.discover()
    if card:
        print(f"     Name: {card.get('name', 'Unknown')}")
        print(f"     Skills: {[s.get('name') for s in card.get('skills', [])]}")
    else:
        print("     ❌ Discovery failed — complete TODO 3")

    # Step 2: Send tasks
    test_queries = [
        "What are the three laws of robotics?",
        "Explain the difference between MCP and A2A in two sentences.",
    ]

    for query in test_queries:
        print(f"\n  📤 Sending task: '{query[:50]}...'")
        result = await client.send_task(query)
        if result:
            state = result.get("status", {}).get("state", "unknown")
            print(f"     State: {state}")
            artifacts = result.get("artifacts", [])
            if artifacts:
                text = artifacts[0].get("parts", [{}])[0].get("text", "")
                print(f"     Response: {text[:200]}...")
        else:
            print("     ❌ Task failed — complete TODOs 4-5")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    print("\n⚠️  Exercise 2: Complete the 8 TODOs to build an A2A agent!")
    print("    Study example_06 and example_08 for reference.\n")

    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n  ⏹️  Interrupted")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        print("  Install: pip install fastapi uvicorn httpx langchain-groq")
