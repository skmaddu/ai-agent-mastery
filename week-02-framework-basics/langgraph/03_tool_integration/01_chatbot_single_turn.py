# Single-turn chatbot

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph, add_messages

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("You: ")
final_state = graph.invoke({"messages": [HumanMessage(content=user_input)]})
last_message = final_state["messages"][-1]
print("Assistant:", last_message.content)
