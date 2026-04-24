import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 8: Building A2A-Compliant Agents — REAL Implementation
================================================================
Topic 8 — A REAL A2A server and client using FastAPI.

The BIG IDEA (Feynman):
  We're building a real business that hangs a sign on its door
  (Agent Card), accepts walk-in customers (Tasks), does the work
  (LLM processing), and hands back the result (Artifacts).

  The server is a FastAPI app.  The client is an HTTP client.
  They communicate using the A2A protocol over real HTTP.

Previously covered:
  - A2A concepts (example_06)
  - Agent Cards, Tasks, state machine (example_06)
  - FastAPI basics (Week 6 deployment)

NEW: A real, runnable A2A server + client that you can test!

Run: python week-07-mcp-a2a-synthesis/examples/example_08_a2a_adk_agents.py
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
# PART 1: A2A DATA MODELS (Pydantic)
# ================================================================
# These models define the wire format for A2A communication.
# They match the A2A specification from Google.

print("=" * 70)
print("PART 1: A2A Data Models")
print("=" * 70)


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class MessagePart(BaseModel):
    """A single part of a message (text, file, data, etc.)."""
    type: str = "text"
    text: Optional[str] = None
    data: Optional[dict] = None
    mime_type: Optional[str] = None


class TaskMessage(BaseModel):
    """A message in the task conversation."""
    role: str  # "user" or "agent"
    parts: list[MessagePart]
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TaskArtifact(BaseModel):
    """An output produced by the agent."""
    name: str
    parts: list[MessagePart]
    mime_type: str = "text/plain"


class TaskStatus(BaseModel):
    """Current status of a task."""
    state: TaskState
    message: Optional[str] = None


class Task(BaseModel):
    """An A2A task — the unit of work."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = Field(
        default_factory=lambda: TaskStatus(state=TaskState.SUBMITTED)
    )
    messages: list[TaskMessage] = Field(default_factory=list)
    artifacts: list[TaskArtifact] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    """The agent's identity card — served at /.well-known/agent.json"""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: list[AgentSkill] = Field(default_factory=list)
    capabilities: dict = Field(default_factory=lambda: {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    })
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"]
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"]
    )


# JSON-RPC request/response models
class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: dict = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[dict] = None


print("  ✅ A2A data models defined (Task, AgentCard, JSONRPCRequest/Response)")


# ================================================================
# PART 2: A2A SERVER — A Real FastAPI Application
# ================================================================

print("\n" + "=" * 70)
print("PART 2: Building the A2A Server (FastAPI)")
print("=" * 70)
print("""
The server has 3 endpoints:
  1. GET  /.well-known/agent.json  → Returns the Agent Card
  2. POST /                         → JSON-RPC endpoint for task operations
  3. GET  /health                   → Health check

The JSON-RPC endpoint handles these methods:
  - tasks/send      → Create and process a task synchronously
  - tasks/get       → Get the status of a task by ID
  - tasks/cancel    → Cancel a running task
""")


class A2AServer:
    """A real A2A-compliant agent server.

    This wraps an LLM agent and exposes it via the A2A protocol.
    Think of it as a "storefront" for your AI agent — any A2A client
    can discover it and send it work.
    """

    def __init__(self, name: str, description: str, host: str = "localhost", port: int = 8001):
        self.name = name
        self.description = description
        self.host = host
        self.port = port
        self.tasks: dict[str, Task] = {}  # In-memory task store
        self.app = None
        self._server_thread = None

        # Define the agent card
        self.agent_card = AgentCard(
            name=name,
            description=description,
            url=f"http://{host}:{port}",
            version="1.0.0",
            skills=[
                AgentSkill(
                    id="research",
                    name="Topic Research",
                    description="Research any topic and produce a structured summary",
                    tags=["research", "summarize", "analyze"],
                ),
                AgentSkill(
                    id="qa",
                    name="Question Answering",
                    description="Answer questions based on knowledge and reasoning",
                    tags=["qa", "answer", "explain"],
                ),
            ],
        )

    def _create_app(self):
        """Build the FastAPI application with A2A endpoints."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title=f"A2A: {self.name}", version="1.0.0")

        # Endpoint 1: Agent Card discovery
        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Return the Agent Card — the agent's identity and capabilities."""
            return self.agent_card.model_dump()

        # Endpoint 2: JSON-RPC handler
        @app.post("/")
        async def handle_jsonrpc(request: Request):
            """Handle A2A JSON-RPC requests."""
            try:
                body = await request.json()
                rpc_request = JSONRPCRequest(**body)
            except Exception as e:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                })

            # Route to the appropriate handler based on method
            handlers = {
                "tasks/send": self._handle_send_task,
                "tasks/get": self._handle_get_task,
                "tasks/cancel": self._handle_cancel_task,
            }

            handler = handlers.get(rpc_request.method)
            if not handler:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": rpc_request.id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {rpc_request.method}"
                    },
                })

            result = await handler(rpc_request)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": rpc_request.id,
                "result": result,
            })

        # Endpoint 3: Health check
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "agent": self.name,
                "tasks_count": len(self.tasks),
            }

        return app

    async def _handle_send_task(self, request: JSONRPCRequest) -> dict:
        """Handle tasks/send — create and process a task."""
        params = request.params

        # Create the task
        task_id = params.get("id", str(uuid.uuid4()))
        message_data = params.get("message", {})

        # Build the message
        parts = [MessagePart(**p) for p in message_data.get("parts", [])]
        message = TaskMessage(
            role=message_data.get("role", "user"),
            parts=parts,
        )

        task = Task(
            id=task_id,
            messages=[message],
            metadata=params.get("metadata", {}),
        )

        # Store the task
        self.tasks[task_id] = task

        # Process the task (this is where the LLM does its work)
        await self._process_task(task)

        return task.model_dump()

    async def _handle_get_task(self, request: JSONRPCRequest) -> dict:
        """Handle tasks/get — retrieve a task by ID."""
        task_id = request.params.get("id")
        if not task_id or task_id not in self.tasks:
            return {"error": f"Task not found: {task_id}"}
        return self.tasks[task_id].model_dump()

    async def _handle_cancel_task(self, request: JSONRPCRequest) -> dict:
        """Handle tasks/cancel — cancel a running task."""
        task_id = request.params.get("id")
        if not task_id or task_id not in self.tasks:
            return {"error": f"Task not found: {task_id}"}

        task = self.tasks[task_id]
        if task.status.state in (TaskState.COMPLETED, TaskState.FAILED):
            return {"error": f"Task already in terminal state: {task.status.state.value}"}

        task.status = TaskStatus(state=TaskState.CANCELED, message="Canceled by client")
        return task.model_dump()

    async def _process_task(self, task: Task):
        """Process a task using the LLM.

        This is the BRAIN of the agent. It:
          1. Reads the user's message
          2. Calls the LLM to generate a response
          3. Creates an artifact with the result
          4. Updates the task state
        """
        # Transition to working
        task.status = TaskStatus(state=TaskState.WORKING, message="Processing...")

        try:
            # Extract the user's message text
            user_text = ""
            for msg in task.messages:
                if msg.role == "user":
                    for part in msg.parts:
                        if part.text:
                            user_text += part.text + " "

            if not user_text.strip():
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message="No text content in the task message"
                )
                return

            # Call the LLM
            llm = get_llm(temperature=0.3)

            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=(
                    f"You are {self.name}. {self.description} "
                    "Provide clear, well-structured responses. "
                    "Use markdown formatting for readability."
                )),
                HumanMessage(content=user_text.strip()),
            ])

            # Create the agent's response message
            agent_message = TaskMessage(
                role="agent",
                parts=[MessagePart(type="text", text=response.content)],
            )
            task.messages.append(agent_message)

            # Create an artifact
            task.artifacts.append(TaskArtifact(
                name="response",
                parts=[MessagePart(type="text", text=response.content)],
                mime_type="text/markdown",
            ))

            # Mark as completed
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message="Task completed successfully"
            )

        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=f"Processing error: {str(e)}"
            )
            task.messages.append(TaskMessage(
                role="agent",
                parts=[MessagePart(type="text", text=f"Error: {str(e)}")],
            ))

    def start(self):
        """Start the A2A server in a background thread."""
        import uvicorn

        self.app = self._create_app()

        def run_server():
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",  # Quiet logs
            )

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        time.sleep(2)  # Give the server a moment to start
        print(f"  🚀 A2A server '{self.name}' running at http://{self.host}:{self.port}")
        print(f"  📇 Agent Card: http://{self.host}:{self.port}/.well-known/agent.json")


# ================================================================
# PART 3: A2A CLIENT — Discover and Call the Agent
# ================================================================

print("\n" + "=" * 70)
print("PART 3: Building the A2A Client")
print("=" * 70)
print("""
The client follows this flow:
  1. Discover the agent by fetching its Agent Card
  2. Check if the agent has the skills we need
  3. Send a task via JSON-RPC
  4. Process the response (artifacts, messages)
""")


class A2AClient:
    """A client for communicating with A2A-compliant agents.

    Like calling a business: look them up, verify they do what you need,
    then place your order.
    """

    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip("/")
        self.agent_card: Optional[AgentCard] = None

    async def discover(self) -> AgentCard:
        """Fetch the agent's Agent Card."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.agent_url}/.well-known/agent.json",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            self.agent_card = AgentCard(**data)

        print(f"  📇 Discovered agent: {self.agent_card.name}")
        print(f"     Description: {self.agent_card.description}")
        print(f"     Skills: {[s.name for s in self.agent_card.skills]}")
        print(f"     Version: {self.agent_card.version}")
        return self.agent_card

    async def send_task(self, message_text: str, metadata: dict = None) -> Task:
        """Send a task to the agent and get the result."""
        import httpx

        task_id = str(uuid.uuid4())

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
                "metadata": metadata or {},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.agent_url,
                json=rpc_request,
                timeout=60.0,  # LLM calls can take a while
            )
            response.raise_for_status()
            data = response.json()

        if "error" in data and data["error"]:
            raise Exception(f"A2A error: {data['error']}")

        task = Task(**data["result"])
        return task

    async def get_task(self, task_id: str) -> Task:
        """Get the status of a previously submitted task."""
        import httpx

        rpc_request = {
            "jsonrpc": "2.0",
            "id": f"get-{task_id[:8]}",
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.agent_url,
                json=rpc_request,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        return Task(**data["result"])

    async def check_health(self) -> dict:
        """Check if the agent is healthy."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.agent_url}/health",
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()


# ================================================================
# PART 4: LIVE DEMO — Server + Client
# ================================================================

async def demo_a2a():
    """Run a live demo of the A2A server and client."""

    print("\n" + "=" * 70)
    print("PART 4: LIVE DEMO — A2A Server + Client")
    print("=" * 70)

    # Step 1: Start the server
    print("\n  📡 Starting A2A server...")
    server = A2AServer(
        name="Research Assistant",
        description="An AI research assistant that can research topics and answer questions with detailed, well-structured responses.",
        port=8001,
    )
    server.start()

    # Step 2: Create a client and discover the agent
    print("\n  🔍 Discovering agent...")
    client = A2AClient("http://localhost:8001")

    try:
        card = await client.discover()
    except Exception as e:
        print(f"  ❌ Discovery failed: {e}")
        print("  Make sure the server started correctly.")
        return

    # Step 3: Health check
    print("\n  🏥 Health check...")
    health = await client.check_health()
    print(f"     Status: {health['status']}")
    print(f"     Agent: {health['agent']}")

    # Step 4: Send a task
    print("\n  📤 Sending task: 'What are the key principles of prompt engineering?'")
    try:
        task = await client.send_task(
            "What are the top 5 principles of prompt engineering? Be concise.",
            metadata={"skill_id": "qa", "priority": "normal"},
        )

        print(f"\n  📥 Task result:")
        print(f"     ID: {task.id[:8]}...")
        print(f"     State: {task.status.state.value}")
        print(f"     Messages: {len(task.messages)}")
        print(f"     Artifacts: {len(task.artifacts)}")

        # Show the agent's response
        if task.artifacts:
            content = task.artifacts[0].parts[0].text or ""
            # Truncate for display
            display = content[:500] + ("..." if len(content) > 500 else "")
            print(f"\n  📄 Agent's response:\n")
            for line in display.split("\n"):
                print(f"     {line}")

    except Exception as e:
        print(f"  ❌ Task failed: {e}")
        print("  This is expected if LLM API keys are not configured.")

    # Step 5: Send another task
    print("\n  📤 Sending second task: 'Explain MCP vs A2A in one paragraph'")
    try:
        task2 = await client.send_task(
            "Explain the difference between MCP and A2A protocols in one paragraph.",
            metadata={"skill_id": "qa"},
        )
        print(f"     State: {task2.status.state.value}")
        if task2.artifacts:
            content = task2.artifacts[0].parts[0].text or ""
            display = content[:400] + ("..." if len(content) > 400 else "")
            print(f"\n  📄 Response:\n")
            for line in display.split("\n"):
                print(f"     {line}")

    except Exception as e:
        print(f"  ❌ Task failed: {e}")

    # Step 6: Retrieve a task by ID
    print(f"\n  🔄 Retrieving task {task.id[:8]}... by ID")
    try:
        retrieved = await client.get_task(task.id)
        print(f"     State: {retrieved.status.state.value}")
        print(f"     Messages: {len(retrieved.messages)}")
    except Exception as e:
        print(f"  ❌ Retrieval failed: {e}")


# ================================================================
# PART 5: KEY TAKEAWAYS
# ================================================================

print("""
Architecture:

  ┌───────────────┐                    ┌────────────────────────┐
  │  A2A Client   │                    │   A2A Server (FastAPI) │
  │               │                    │                        │
  │  1. Discover ─┼── GET /agent.json──▶│  Agent Card           │
  │               │                    │                        │
  │  2. Send     ─┼── POST / ─────────▶│  JSON-RPC Router      │
  │     Task      │   (tasks/send)     │    │                   │
  │               │                    │    ▼                   │
  │               │                    │  Process Task          │
  │               │                    │    │                   │
  │               │                    │    ▼                   │
  │  3. Get     ◀─┼── Response ────────│  LLM Agent            │
  │     Result    │   (task + artifacts)│                       │
  └───────────────┘                    └────────────────────────┘
""")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    try:
        asyncio.run(demo_a2a())
    except KeyboardInterrupt:
        print("\n  ⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n  ⚠️  Demo error: {e}")
        print("  Install dependencies: pip install fastapi uvicorn httpx langchain-groq")

    print("\n✅ Example 08 complete!")
