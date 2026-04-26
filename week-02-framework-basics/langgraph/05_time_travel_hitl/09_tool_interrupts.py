# HITL example with interrupt inside a tool

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt

load_dotenv()


# Tool with interrupt inside the tool (HITL)
@tool
def delete_file(file_name: str):
    """Delete a file from the system (simulated)."""

    # Human input expected as: {"approve": "yes"/"no"}
    human_input = interrupt(
        {
            "action": "delete_file",
            "file_name": file_name,
            "message": "Do you really want to delete this file?",
        }
    )
    print(human_input)

    if human_input.get("approve") == "yes":
        # In a real system, file deletion would happen here
        return f"File '{file_name}' deleted successfully."

    return "File deletion cancelled by user."


# LLM + tool binding
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

tools = [delete_file]
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}


# State
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Chatbot node (ReAct-style)
def chatbot(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}


def should_continue(state: ChatState) -> Literal["invoke tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "invoke tools"
    return "end"


# Tool execution node
def tool_node(state: ChatState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_output = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Build the graph
graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tool_node", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot", should_continue, {"invoke tools": "tool_node", "end": END}
)
graph_builder.add_edge("tool_node", "chatbot")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# Thread config (conversation memory)
config = {"configurable": {"thread_id": "demo-5"}}

# CLI UI loop
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["stop", "exit", "end", "bye"]:
        print("Assistant: Goodbye!")
        break
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config=config
    )

    # ------------------------------------------------
    # HITL UI for tool-level interrupt
    # ------------------------------------------------
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value

        print("\n" + "=" * 50)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 50)

        print("Action:", payload["action"])
        print("File:", payload["file_name"])
        print("Message:", payload["message"])

        print("=" * 50)

        approval = input("Approve? (yes/no): ").strip().lower()

        result = graph.invoke(Command(resume={"approve": approval}), config=config)

    # Final assistant response
    last_message = result["messages"][-1]
    print("Assistant:", last_message.content)
