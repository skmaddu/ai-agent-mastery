# Smart Email Support Assistant - Conditional Routing Example
# Returning labels from routing function

from typing import TypedDict

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
    predicted_category = llm.invoke(prompt).strip().lower()
    if predicted_category not in {"billing", "technical", "general"}:
        predicted_category = "general"
    return {"category": predicted_category}


def route_next(state: SupportState):
    c = state["category"]
    if "bill" in c:
        return "Billing issue"
    elif "tech" in c:
        return "Technical issue"
    else:
        return "Unclear query"


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
    """)
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

graph_builder.add_conditional_edges(
    "categorize",
    route_next,
    {
        "Billing issue": "billing_node",
        "Technical issue": "tech_node",
        "Unclear query": "clarify_node",
    },
)

graph_builder.add_edge("billing_node", "summarize")
graph_builder.add_edge("tech_node", "summarize")
graph_builder.add_edge("clarify_node", "summarize")
graph_builder.add_edge("summarize", END)

graph = graph_builder.compile()

with open("email_routing_labels.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


emails = [
    "I was charged twice on my invoice.",
    "There's a bug in your app that crashes on login.",
    "What is your pricing policy?",
]

for e in emails:
    print("\nInput:", e)
    final_state = graph.invoke({"email_text": e})
    print("Response:", final_state["response"])
    print("Summary:", final_state["summary"])
