# # Human approval loop with feedback

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt

load_dotenv()


class State(TypedDict):
    topic: str
    messages: Annotated[list[AnyMessage], add_messages]
    approval: str
    feedback: str


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def create_tweet(state: State):
    response = llm.invoke(
        [
            SystemMessage("You are an expert at writing engaging tweets"),
            HumanMessage(f"Write a tweet on {state['topic']}"),
        ]
    )
    return {"messages": response}


def human_review(state: State):
    print("Human review node executing...")

    # Human input expected as: {"approval": "yes" | "no", "feedback": str}
    human_input = interrupt(
        {
            "tweet": state["messages"][-1].content,
            "question": "Approve this tweet? (yes / no + feedback)",
        }
    )

    return {
        "approval": human_input["approval"],
        "feedback": human_input.get("feedback", ""),
    }


def review_router(state: State) -> Literal["post_tweet", "incorporate_feedback"]:
    if state["approval"].lower() == "yes":
        return "post_tweet"
    else:
        return "incorporate_feedback"


# Rewrite tweet using feedback
def incorporate_feedback(state: State):
    print("\nIncorporating feedback...\n")

    tweet = state["messages"][-1].content
    response = llm.invoke(
        [
            SystemMessage("You are an expert at writing engaging tweets"),
            HumanMessage(
                f"Rewrite this tweet:\n{tweet}\n\nFeedback: {state['feedback']}"
            ),
        ]
    )
    return {"messages": response}


def post_tweet(state: State):
    print("\nTweet posted\n")
    print(state["messages"][-1].content)
    return {}


graph_builder = StateGraph(State)

graph_builder.add_node("create_tweet", create_tweet)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("incorporate_feedback", incorporate_feedback)
graph_builder.add_node("post_tweet", post_tweet)

graph_builder.add_edge(START, "create_tweet")
graph_builder.add_edge("create_tweet", "human_review")
graph_builder.add_conditional_edges("human_review", review_router)
graph_builder.add_edge("incorporate_feedback", "human_review")
graph_builder.add_edge("post_tweet", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

# Run the workflow
config = {"configurable": {"thread_id": "demo-2"}}
result = graph.invoke({"topic": "Life"}, config=config)


# -------------------------
# UI simulation Loop
# -------------------------
while "__interrupt__" in result:
    payload = result["__interrupt__"][0].value

    print("\n" + "=" * 50)
    print("HUMAN REVIEW SCREEN")
    print("=" * 50)

    print("\nCurrent Tweet:\n")
    print(payload["tweet"])

    print("\nQuestion:")
    print(payload["question"])

    print("=" * 50)

    # --- Simulated UI input ---
    approval = input("\nApprove? (yes/no): ").strip().lower()
    feedback = ""

    if approval == "no":
        feedback = input("Provide feedback: ")

    print("\nSending response back to workflow...\n")

    result = graph.invoke(
        Command(resume={"approval": approval, "feedback": feedback}), config=config
    )
