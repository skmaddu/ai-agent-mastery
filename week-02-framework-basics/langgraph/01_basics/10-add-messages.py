# Demonstrating add_messages reducer 
 
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def node1(state: ChatState):
    return {"messages": [HumanMessage(content="Hello there!")]}

def node2(state: ChatState):
    return {"messages": [AIMessage(content="Hi! How can I assist you today?")]}

def node3(state: ChatState):
    return {"messages": ["Hello"]}

def node4(state: ChatState): 
    return {"messages": [{"type": "human", "content": "What can you do?"}]}
def node5(state: ChatState): 
    return {"messages": [{"type": "ai", "content": "I can assist you with various tasks."}]}
graph_builder = StateGraph(ChatState)

graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)

graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")
graph_builder.add_edge("node3", "node4")
graph_builder.add_edge("node4", "node5")
graph_builder.add_edge("node5", END)

graph = graph_builder.compile()

final_state = graph.invoke({"messages": []})

for message in final_state["messages"]:
    message.pretty_print() 