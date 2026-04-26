import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Solution 2: A2A-Compliant Agent Server and Client
====================================================
Difficulty: ⭐⭐⭐ Intermediate | Time: 2 hours

Complete solution implementing all 8 TODOs:
  1. AgentCard Pydantic model
  2. Task state machine (TaskState enum + Task model)
  3. Agent Card endpoint (GET /.well-known/agent.json)
  4. Task handler (POST / with JSON-RPC routing)
  5. Task state transitions (submitted -> working -> completed/failed)
  6. A2A Client (discover, send_task, get_task)
  7. Error handling (LLM failures, network errors)
  8. Phoenix tracing spans (timing around lifecycle events)

Run: python week-07-mcp-a2a-synthesis/solutions/solution_02_a2a_protocol.py
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
    """Create LLM based on provider setting."""
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
# TODO 1 (SOLVED): Define the AgentCard Pydantic Model
# ================================================================

class AgentCard(BaseModel):
    """The agent's identity — served at /.well-known/agent.json"""
    name: str = Field(description="Agent's display name")
    description: str = Field(description="What this agent does")
    url: str = Field(description="Base URL for A2A communication")
    version: str = Field(default="1.0.0", description="Agent version")
    skills: list[dict] = Field(
        default_factory=list,
        description="List of skill dicts with id, name, description, tags",
    )
    capabilities: dict = Field(
        default_factory=lambda: {"streaming": False, "pushNotifications": False},
        description="Protocol capabilities",
    )


# ================================================================
# TODO 2 (SOLVED): Define the Task State Machine
# ================================================================

class TaskState(str, Enum):
    """All possible states for an A2A task."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# Valid transitions map
VALID_TRANSITIONS = {
    TaskState.SUBMITTED: {TaskState.WORKING, TaskState.CANCELED},
    TaskState.WORKING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED},
    TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELED},
    TaskState.COMPLETED: set(),    # Terminal
    TaskState.FAILED: set(),       # Terminal
    TaskState.CANCELED: set(),     # Terminal
}


class Task(BaseModel):
    """An A2A task with state machine."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: dict = Field(
        default_factory=lambda: {"state": TaskState.SUBMITTED.value, "message": "Task submitted"},
    )
    messages: list[dict] = Field(default_factory=list)
    artifacts: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def transition(self, new_state: TaskState, message: str = ""):
        """Move the task to a new state with validation."""
        current = TaskState(self.status["state"])
        allowed = VALID_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {current.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        old = current.value
        self.status = {"state": new_state.value, "message": message}
        return f"  Task {self.id[:8]}... : {old} -> {new_state.value}"


# ================================================================
# TODO 3-5 (SOLVED): FastAPI App with Agent Card, Task Handler,
#                     and Task State Transitions
# ================================================================

def create_a2a_app(agent_card_data: dict) -> "FastAPI":
    """Create the FastAPI app with A2A endpoints."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="A2A Agent")

    # Store tasks in memory
    tasks: dict[str, dict] = {}

    # --- TODO 3: Agent Card endpoint ---
    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        """Return the Agent Card — the agent's identity and capabilities."""
        return agent_card_data

    # --- TODO 4: JSON-RPC task handler ---
    @app.post("/")
    async def handle_jsonrpc(request: Request):
        """Handle A2A JSON-RPC requests."""
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            })

        rpc_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

        if method == "tasks/send":
            result = await handle_send_task(params)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": result,
            })

        elif method == "tasks/get":
            task_id = params.get("id")
            if task_id and task_id in tasks:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": tasks[task_id],
                })
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32602, "message": f"Task not found: {task_id}"},
            })

        else:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })

    # --- TODO 5: Task state transitions + LLM processing ---
    async def handle_send_task(params: dict) -> dict:
        """Create a task, process it with the LLM, return the result."""
        task_id = params.get("id", str(uuid.uuid4()))
        message_data = params.get("message", {})
        user_text = ""
        for part in message_data.get("parts", []):
            if part.get("text"):
                user_text += part["text"] + " "
        user_text = user_text.strip()

        # Create task (submitted)
        # --- TODO 8: Phoenix tracing span — task creation ---
        span_start = time.time()

        task = Task(
            id=task_id,
            messages=[message_data] if message_data else [],
            metadata=params.get("metadata", {}),
        )

        creation_ms = (time.time() - span_start) * 1000
        print(f"  Span [task_created]: {creation_ms:.0f}ms")

        # Transition to working
        # --- TODO 8: Phoenix tracing span — processing ---
        proc_start = time.time()
        task.transition(TaskState.WORKING, "Processing with LLM...")

        # --- TODO 7: Error handling around LLM call ---
        try:
            llm = get_llm(temperature=0.3)
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke([
                SystemMessage(content=(
                    f"You are {agent_card_data.get('name', 'Research Assistant')}. "
                    f"{agent_card_data.get('description', '')} "
                    "Provide clear, well-structured responses."
                )),
                HumanMessage(content=user_text),
            ])

            response_text = response.content

            # Add agent message
            task.messages.append({
                "role": "agent",
                "parts": [{"type": "text", "text": response_text}],
            })

            # Add artifact
            task.artifacts.append({
                "name": "response",
                "parts": [{"type": "text", "text": response_text}],
                "mimeType": "text/plain",
            })

            # Transition to completed
            task.transition(TaskState.COMPLETED, "Task completed successfully")

        except Exception as e:
            # On failure, set task to failed state
            task.transition(TaskState.FAILED, f"LLM error: {str(e)}")
            task.messages.append({
                "role": "agent",
                "parts": [{"type": "text", "text": f"Error: {str(e)}"}],
            })

        proc_ms = (time.time() - proc_start) * 1000
        print(f"  Span [task_processing]: {proc_ms:.0f}ms")

        # Store task
        task_dict = task.model_dump()
        tasks[task_id] = task_dict

        # --- TODO 8: Phoenix tracing span — completion ---
        total_ms = (time.time() - span_start) * 1000
        print(f"  Span [task_total]: {total_ms:.0f}ms  state={task.status['state']}")

        return task_dict

    @app.get("/health")
    async def health():
        return {"status": "healthy", "tasks_count": len(tasks)}

    return app


# ================================================================
# TODO 6 (SOLVED): A2A Client
# ================================================================

class A2AClient:
    """Client for communicating with an A2A agent."""

    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip("/")
        self.agent_card = None

    async def discover(self) -> dict:
        """Discover the agent by fetching its Agent Card."""
        import httpx

        # --- TODO 8: Phoenix tracing span — client discover ---
        span_start = time.time()

        # --- TODO 7: Error handling for network errors ---
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.agent_url}/.well-known/agent.json",
                    timeout=10.0,
                )
                response.raise_for_status()
                self.agent_card = response.json()

            disc_ms = (time.time() - span_start) * 1000
            print(f"  Span [client_discover]: {disc_ms:.0f}ms")
            return self.agent_card

        except httpx.RequestError as e:
            disc_ms = (time.time() - span_start) * 1000
            print(f"  Span [client_discover_error]: {disc_ms:.0f}ms")
            print(f"  Network error during discovery: {e}")
            return None

    async def send_task(self, message_text: str) -> dict:
        """Send a task to the agent via JSON-RPC tasks/send."""
        import httpx

        task_id = str(uuid.uuid4())

        # --- TODO 8: Phoenix tracing span — client send ---
        span_start = time.time()

        rpc_request = {
            "jsonrpc": "2.0",
            "id": f"req-{task_id[:8]}",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message_text}],
                },
            },
        }

        # --- TODO 7: Error handling for network / JSON-RPC errors ---
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.agent_url,
                    json=rpc_request,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

            send_ms = (time.time() - span_start) * 1000
            print(f"  Span [client_send_task]: {send_ms:.0f}ms")

            if "error" in data and data["error"]:
                print(f"  JSON-RPC error: {data['error']}")
                return None

            return data.get("result")

        except httpx.RequestError as e:
            send_ms = (time.time() - span_start) * 1000
            print(f"  Span [client_send_error]: {send_ms:.0f}ms")
            print(f"  Network error sending task: {e}")
            return None

    async def get_task(self, task_id: str) -> dict:
        """Get the status of a task via JSON-RPC tasks/get."""
        import httpx

        rpc_request = {
            "jsonrpc": "2.0",
            "id": f"get-{task_id[:8]}",
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        # --- TODO 7: Error handling ---
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.agent_url,
                    json=rpc_request,
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

            if "error" in data and data["error"]:
                print(f"  JSON-RPC error: {data['error']}")
                return None

            return data.get("result")

        except httpx.RequestError as e:
            print(f"  Network error getting task: {e}")
            return None


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
    print("\n  Starting A2A server on port 8010...")
    app = create_a2a_app(agent_card_data)

    if app is None:
        print("  create_a2a_app() returned None — something went wrong!")
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
    print("\n  Discovering agent...")
    card = await client.discover()
    if card:
        print(f"     Name: {card.get('name', 'Unknown')}")
        print(f"     Skills: {[s.get('name') for s in card.get('skills', [])]}")
    else:
        print("     Discovery failed")

    # Step 2: Send tasks
    test_queries = [
        "What are the three laws of robotics?",
        "Explain the difference between MCP and A2A in two sentences.",
    ]

    for query in test_queries:
        print(f"\n  Sending task: '{query[:50]}...'")
        result = await client.send_task(query)
        if result:
            state = result.get("status", {}).get("state", "unknown")
            print(f"     State: {state}")
            artifacts = result.get("artifacts", [])
            if artifacts:
                text = artifacts[0].get("parts", [{}])[0].get("text", "")
                print(f"     Response: {text[:200]}...")
        else:
            print("     Task failed")

    print("\nDemo complete!")


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n  Interrupted")
    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Install: pip install fastapi uvicorn httpx langchain-groq")
