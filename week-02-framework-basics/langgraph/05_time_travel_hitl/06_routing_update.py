# Command example combining routing and state update

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing import Literal

class State(TypedDict):
    email_text: str
    assigned_team: str

def classify_and_route_email(state: State) -> Command[Literal["handle_sales", "handle_support"]]:
    text = state["email_text"].lower()

    if "price" in text or "buy" in text:
        print("Assigned team: Sales Team")
        return Command(
            update={"assigned_team": "Sales Team"},
            goto="handle_sales"
        )
    else:
        print("Assigned team: Support Team")
        return Command(
            update={"assigned_team": "Support Team"},
            goto="handle_support"
        )

def handle_sales(state: State):
    print("Routing email to Sales team")
    print("State:", state)
    return {}

def handle_support(state: State):
    print("Routing email to Support team")
    print("State:", state)
    return {}

graph_builder = StateGraph(State)

graph_builder.add_node("classify_and_route_email", classify_and_route_email)
graph_builder.add_node("handle_sales", handle_sales)
graph_builder.add_node("handle_support", handle_support)

graph_builder.set_entry_point("classify_and_route_email")

graph_builder.add_edge("handle_sales", END)
graph_builder.add_edge("handle_support", END)

graph = graph_builder.compile()

graph.invoke({"email_text": "What is the price of your product?"})
