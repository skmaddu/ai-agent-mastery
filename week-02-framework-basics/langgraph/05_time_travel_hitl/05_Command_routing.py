# Routing using Command instead of conditional edges

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

class State(TypedDict):
    email_text: str

# Classification + routing
def classify_and_route(state: State)-> Command[Literal["handle_sales", "handle_support"]]:
    text = state["email_text"].lower()

    if "price" in text or "buy" in text:
        print("Email classified as: sales")
        return Command(goto="handle_sales")
    else:
        print("Email classified as: support")
        return Command(goto="handle_support")

def handle_sales(state: State):
    print("Routing email to Sales team")
    print("State:", state)
    return {}

def handle_support(state: State):
    print("Routing email to Support team")
    print("State:", state)
    return {}

graph_builder = StateGraph(State)

graph_builder.add_node("classify_and_route", classify_and_route)
graph_builder.add_node("handle_sales", handle_sales)
graph_builder.add_node("handle_support", handle_support)

graph_builder.add_edge(START, "classify_and_route")
graph_builder.add_edge("handle_sales", END)
graph_builder.add_edge("handle_support", END)

graph = graph_builder.compile()

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

graph.invoke({"email_text": "What is the price of your product?"})
