# Sequential Graph with Persistence

from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Build the graph
class State(TypedDict):
    data: Annotated[str, operator.add]   

def node_a(state: State) -> State:
    print("Node A executed")
    return {"data": "A "}

def node_b(state: State) -> State:
    print("Node B executed")
    return {"data": "B "}

def node_c(state: State) -> State:
    print("Node C executed")
    return {"data": "C "}

def node_d(state: State) -> State:
    print("Node D executed")
    return {"data": "D "}

def node_e(state: State) -> State:
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
graph_builder.add_edge("B", "D")
graph_builder.add_edge("C", "E")
graph_builder.add_edge("D", "E")
graph_builder.add_edge("E", END)

#  SQLliteSaver Checkpointer 
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=checkpointer)

# Config with thread_id
config = {"configurable": {"thread_id": "user-2"}}

# Invoke the graph with config
final_state = graph.invoke({"data": "XYZ "}, config=config)
print("Final state:", final_state)
