# Chatbot with Memory


from langchain_google_genai import ChatGoogleGenerativeAI 


from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END

from langchain_core.messages import HumanMessage, AnyMessage, ToolMessage
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

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
    return { "messages": response}

def should_continue(state: ChatState) :
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
    "chatbot",
    should_continue,
    {
        "invoke tools": "tool_node",
        "end": END,
    }
)
                                
graph_builder.add_edge("tool_node", "chatbot")

graph = graph_builder.compile()

#  SQLliteSaver Checkpointer 
conn = sqlite3.connect("chatbot.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=checkpointer)

# Config with thread_id
config = {"configurable": {"thread_id": "streamlit-session-1"}}

