# Routing using conditional edges

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal

class State(TypedDict):
    email_text: str
    category: str

def classify_email(state: State):
    text = state["email_text"].lower()

    if "price" in text or "buy" in text:
        category = "sales"
    else:
        category = "support"

    print("Email classified as:", category)
    return {"category": category}

def email_router(state: State) ->Literal["handle_sales", "handle_support"]:
    if state["category"] == "sales":
        return "handle_sales"
    else:
        return "handle_support"

def handle_sales(state: State):
    print("Routing email to Sales team")
    print("State:", state)
    return {}

def handle_support(state: State):
    print("Routing email to Support team")
    print("State:", state)
    return {}

# Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("classify_email", classify_email)
graph_builder.add_node("handle_sales", handle_sales)
graph_builder.add_node("handle_support", handle_support)

graph_builder.add_edge(START, "classify_email")

# Conditional edge handles routing
graph_builder.add_conditional_edges("classify_email", email_router)

graph_builder.add_edge("handle_sales", END)
graph_builder.add_edge("handle_support", END)

graph = graph_builder.compile()

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

graph.invoke({"email_text": "What is the price of your product?"})
