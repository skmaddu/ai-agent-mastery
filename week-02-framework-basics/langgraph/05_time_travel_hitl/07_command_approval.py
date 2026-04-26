# Routing control flow after interrupt using Command
# Using Command in the program clip5_1_tweet_approval.py

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt

load_dotenv()


# State (approval removed)
class State(TypedDict):
    topic: str
    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def create_tweet(state: State):
    response = llm.invoke(
        [
            SystemMessage("You are an expert at writing engaging tweets"),
            HumanMessage(f"Write a tweet on {state['topic']}"),
        ]
    )
    return {"messages": response}


def human_review(state: State) -> Command[Literal["post_tweet", "__end__"]]:
    print("Human review node executing....")

    human_input = interrupt(
        {
            "tweet": state["messages"][-1].content,
            "question": "Do you want to post this tweet? (yes/no)",
        }
    )

    # Routing happens here using Command
    if human_input.lower() == "yes":
        return Command(goto="post_tweet")
    else:
        return Command(goto=END)


def post_tweet(state: State):
    print("\nTweet posted\n")
    print(state["messages"][-1].content)
    return {}


graph_builder = StateGraph(State)

graph_builder.add_node("create_tweet", create_tweet)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("post_tweet", post_tweet)

graph_builder.add_edge(START, "create_tweet")
graph_builder.add_edge("create_tweet", "human_review")
graph_builder.add_edge("post_tweet", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# Run the workflow
config = {"configurable": {"thread_id": "demo-3"}}
result = graph.invoke({"topic": "Life"}, config=config)

# -------------------------
# UI simulation
# -------------------------
payload = result["__interrupt__"][0].value

print("\n" + "=" * 50)
print("HUMAN REVIEW SCREEN")
print("=" * 50)

print("\nGenerated Tweet:\n")
print(payload["tweet"])

print("\nQuestion:")
print(payload["question"])

print("\nWaiting for human response...")
print("=" * 50)

# Simulated UI response
human_response = "yes"
print("\nHuman selected:", human_response)
print("Sending response back to workflow...\n")

# Resume workflow
graph.invoke(Command(resume=human_response), config=config)
