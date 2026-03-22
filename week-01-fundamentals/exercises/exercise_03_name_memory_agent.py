"""
Exercise 3: Name Memory Agent
================================
Difficulty: Intermediate | Time: 2.5 hours

Task:
Build a LangGraph agent that:
1. Greets the user and asks for their name
2. Remembers the name across conversation turns
3. Tracks how many messages have been exchanged
4. Uses the name naturally in all responses

Instructions:
1. Define a ChatState TypedDict with: messages, user_name, message_count
2. Create a chat_node that uses the LLM with state context
3. Build a LangGraph StateGraph with conditional edges
4. Add a simple loop: continue chatting until user says "bye"
5. Bonus: Add Phoenix tracing and observe the conversation flow

Run: python exercise_03_name_memory_agent.py
     (Works from this folder or anywhere: config/.env is found via the script file path.)
"""

import os
import re
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import add_messages

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / "config" / ".env")
load_dotenv()


class ChatState(TypedDict):
    """Conversation state: history, remembered name, and turn count."""

    messages: Annotated[list, add_messages]
    user_name: Optional[str]
    message_count: int


def _make_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.7,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0.7,
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.7,
    )


def _extract_name(text: str, current: Optional[str]) -> Optional[str]:
    """Best-effort name capture from a user message (no extra LLM call)."""
    if current:
        return current
    t = text.strip()
    patterns = [
        r"(?i)\bmy name is\s+([A-Za-z][A-Za-z\s'-]{0,48})",
        r"(?i)\b(?:i'?m|i am)\s+([A-Za-z][A-Za-z'-]{0,47})",
        r"(?i)\bcall me\s+([A-Za-z][A-Za-z'-]{0,47})",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            name = m.group(1).strip()
            name = re.split(r"[\s,!.]+", name)[0]
            return name[:1].upper() + name[1:].lower() if name else None
    return None


def _route_decision(state: ChatState) -> str:
    """Route to goodbye if the latest user message is an exit phrase."""
    msgs = state.get("messages") or []
    if not msgs:
        return "chat"
    last = msgs[-1]
    if isinstance(last, HumanMessage):
        phrase = last.content.strip().lower()
        if phrase in ("bye", "quit", "exit", "goodbye"):
            return "end"
    return "chat"


def _router_node(state: ChatState):
    """No-op node so we can branch from a real graph node."""
    return {}


def _build_chat_node(llm):
    def _chat_node(state: ChatState):
        msgs = state["messages"]
        last = msgs[-1]
        if not isinstance(last, HumanMessage):
            raise ValueError("Expected last message to be from the user.")

        user_name = _extract_name(last.content, state.get("user_name"))
        message_count = int(state.get("message_count") or 0) + 1

        known = user_name or state.get("user_name")
        system = SystemMessage(
            content=(
                "You are a warm, concise assistant.\n"
                f"- User messages exchanged so far (including this one): {message_count}\n"
                f"- User's name (if known): {known or 'not yet known — greet them and ask for their name if it fits the conversation'}\n"
                "- Once you know their name, use it naturally in your replies.\n"
                "- Keep replies short (2–4 sentences) unless the user asks for more."
            )
        )

        response = llm.invoke([system, *msgs])
        return {
            "messages": [response],
            "user_name": user_name or state.get("user_name"),
            "message_count": message_count,
        }

    return _chat_node


def _goodbye_node(state: ChatState):
    name = state.get("user_name")
    text = f"Goodbye{' ' + name if name else ''}! Thanks for chatting."
    return {"messages": [AIMessage(content=text)]}


def create_memory_agent():
    """Build the name-memory agent.

    Returns:
        Compiled LangGraph agent
    """
    llm = _make_llm()
    graph = StateGraph(ChatState)
    graph.add_node("router", _router_node)
    graph.add_node("chat", _build_chat_node(llm))
    graph.add_node("goodbye", _goodbye_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {"chat": "chat", "end": "goodbye"},
    )
    graph.add_edge("chat", END)
    graph.add_edge("goodbye", END)

    return graph.compile()


if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    print("Name Memory Agent")
    print("=" * 40)
    print(f"Provider: {provider}")
    print("Say 'bye', 'quit', or 'exit' to leave.\n")

    agent = create_memory_agent()
    state: ChatState = {
        "messages": [],
        "user_name": None,
        "message_count": 0,
    }

    while True:
        user_input = input("You: ")
        phrase = user_input.strip().lower()
        state = {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_input)],
        }
        state = agent.invoke(state)
        last = state["messages"][-1]
        content = getattr(last, "content", str(last))
        print(f"Assistant: {content}\n")
        if phrase in ("quit", "exit", "bye", "goodbye"):
            break
