# Looping Workflow to Process a List of Tasks

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    tasks: list[str]

def task_node(state: State):
    task = state["tasks"][0]
    print(f"Processing task: {task}")
    return {"tasks": state["tasks"][1:]}

# Router function for looping
def should_continue(state: State) ->Literal["loop", "exit"]:
    if state["tasks"]:
        return "loop"   # loop back to task_node
    else:
        return "exit"   # go to END

graph_builder = StateGraph(State)

graph_builder.add_node("task_node", task_node)

# Conditional edges for looping
graph_builder.add_conditional_edges(
    "task_node",
    should_continue,
    {
        "loop": "task_node",
        "exit": END
    }
)

graph_builder.add_edge(START, "task_node")

graph = graph_builder.compile()

with open("loop_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# Invoke the graph with initial state
initial_state = {"tasks": ["Email client", "Write report", "Schedule meeting"]}

graph.invoke(initial_state)