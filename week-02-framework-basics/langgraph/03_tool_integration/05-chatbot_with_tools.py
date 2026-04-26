# Demonstrates a tool-aware chatbot where the LLM requests tools and the workflow executes them

from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph, add_messages

load_dotenv()

tavily_tool = TavilySearch(max_results=2)


@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers and returns the sum.
    Args:
        a: The first number to add
        b: The second number to add
    """
    return a + b


@tool("subtract", description="Subtract b from a and return the result.")
def sub(a: int, b: int) -> int:
    """Return the difference of two numbers."""
    return a - b


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

tools = [tavily_tool, add, sub]
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}


def should_continue(state: ChatState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "invoke tools"
    return "end"


def tool_node(state: ChatState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_output = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
    return {"messages": result}


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tool_node", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot", should_continue, {"invoke tools": "tool_node", "end": END}
)
graph_builder.add_edge("tool_node", END)

graph = graph_builder.compile()

with open("workflow.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["stop", "exit", "end", "bye"]:
        print("Assistant: Goodbye!")
        break
    final_state = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    for message in final_state["messages"]:
        message.pretty_print()
    print()
