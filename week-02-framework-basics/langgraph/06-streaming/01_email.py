# Smart Email Support Assistant - Conditional Routing Example

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


class SupportState(TypedDict):
    email_text: str
    cleaned_text: str
    category: str
    response: str
    summary: str


def preprocess_email(state: SupportState):
    text = state["email_text"].strip().lower()
    return {"cleaned_text": text}


def categorize_email(state: SupportState):
    prompt = f"""
    You are a support triage assistant. 
    Read this email and classify it as exactly one of the following:
    - billing
    - technical
    - general

    Email: {state["cleaned_text"]}

    Respond with only one word: billing, technical, or general.
    """
    predicted_category = llm.invoke(prompt).content.strip().lower()
    if predicted_category not in {"billing", "technical", "general"}:
        predicted_category = "general"
    return {"category": predicted_category}


def route_next(
    state: SupportState,
) -> Literal["billing_node", "tech_node", "clarify_node"]:
    c = state["category"]
    if "bill" in c:
        return "billing_node"
    elif "tech" in c:
        return "tech_node"
    else:
        return "clarify_node"


def billing_node(state):
    return {"response": "Your billing issue has been forwarded to our accounts team."}


def tech_node(state):
    return {"response": "Our technical team will review your issue shortly."}


def clarify_node(state):
    return {"response": "Could you please provide more details about your query?"}


def summarize_interaction(state):
    summary = llm.invoke(f"""
    Summarize this support interaction in one sentence:
    Email: {state["email_text"]}
    Category: {state["category"]}
    Response: {state["response"]}
    """).content
    return {"summary": summary}


graph_builder = StateGraph(SupportState)

graph_builder.add_node("preprocess", preprocess_email)
graph_builder.add_node("categorize", categorize_email)
graph_builder.add_node("billing_node", billing_node)
graph_builder.add_node("tech_node", tech_node)
graph_builder.add_node("clarify_node", clarify_node)
graph_builder.add_node("summarize", summarize_interaction)

graph_builder.add_edge(START, "preprocess")
graph_builder.add_edge("preprocess", "categorize")

graph_builder.add_conditional_edges("categorize", route_next)

graph_builder.add_edge("billing_node", "summarize")
graph_builder.add_edge("tech_node", "summarize")
graph_builder.add_edge("clarify_node", "summarize")
graph_builder.add_edge("summarize", END)

graph = graph_builder.compile()

initial_state = {"email_text": "I was charged twice on my invoice."}

for chunk in graph.stream(initial_state, stream_mode="debug"):
    print(chunk, end="\n\n")
