# Static interrupts for debugging using interrupt_before and interrupt_after

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    data: Annotated[str, operator.add]

def node_a(state: State):
    print("Node A executed")
    return {"data": "A "}

def node_b(state: State):
    print("Node B executed")
    return {"data": "B "}

def node_c(state: State):
    print("Node C executed")
    return {"data": "C "}

def node_d(state: State):
    print("Node D executed")
    return {"data": "D "}

def node_e(state: State):
    print("Node E executed")
    return {"data": "E "}


graph_builder = StateGraph(State)

graph_builder.add_node("A", node_a)
graph_builder.add_node("B", node_b)
graph_builder.add_node("C", node_c)
graph_builder.add_node("D", node_d)
graph_builder.add_node("E", node_e)

graph_builder.add_edge(START, "A")
graph_builder.add_edge("A", "B")
graph_builder.add_edge("B", "C")
graph_builder.add_edge("C", "D")
graph_builder.add_edge("D", "E")
graph_builder.add_edge("E", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["B"],     # Pause before node B
    interrupt_after=["C", "D"]  # Pause after nodes C and D
)

config = {"configurable": {"thread_id": "debug-1"}}
graph.invoke({"data": ""},config= config)
