# Safe Iterative Rewrite Workflow: Auto-Stop After 4 Evaluation Cycles

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


class State(TypedDict):
    topic: str
    post: str
    feedback: str
    quality: Literal["ready to post", "needs rewrite"]
    attempts: int


# ---- Evaluation schema ----
class Review(BaseModel):
    verdict: Literal["ready to post", "needs rewrite"] = Field(
        description="Decide if the LinkedIn post is ready or needs improvement."
    )
    feedback: str = Field(
        description=""" Always provide feedback. 
            "If the post is ready, briefly explain why it is good.
            If rewrite needed, suggest how to improve tone, clarity, or engagement."""
    )


evaluator_llm = llm.with_structured_output(Review)


def post_generator(state: State):
    """LLM creates or refines a LinkedIn post."""

    if state.get("feedback") and state.get("post"):
        prompt = (
            f"Here is your previous LinkedIn post:\n\n{state['post']}\n\n"
            f"Feedback from the reviewer: {state['feedback']}\n\n"
            f"Revise and improve this post about '{state['topic']}' to make it clearer, "
            f"more engaging, and suitable for LinkedIn."
        )
    else:
        prompt = f"Write a very short, engaging, professional LinkedIn post about '{state['topic']}'."

    response = llm.invoke(prompt)
    return {"post": response.content}


def post_evaluator(state: State):
    """Evaluate the LinkedIn post."""
    review = evaluator_llm.invoke(f"Evaluate this LinkedIn post:\n{state['post']}")
    return {
        "quality": review.verdict,
        "feedback": review.feedback,
        "attempts": state.get("attempts", 0) + 1,
    }


# ---- Router ----
def decide_next(state: State):
    attempts = state["attempts"]
    quality = state["quality"]

    # Force at least 2 evaluation cycles (rewrites)
    if attempts < 2:
        return "Revise"

    # After 2 attempts, accept only if quality is good
    if quality == "ready to post":
        return "Accept"

    # Hard stop after 4 attempts (accept even if not perfect)
    if attempts >= 4:
        return "Accept"

    return "Revise"


graph_builder = StateGraph(State)
graph_builder.add_node("post_generator", post_generator)
graph_builder.add_node("post_evaluator", post_evaluator)

graph_builder.add_edge(START, "post_generator")
graph_builder.add_edge("post_generator", "post_evaluator")
graph_builder.add_conditional_edges(
    "post_evaluator",
    decide_next,
    {
        "Accept": END,
        "Revise": "post_generator",
    },
)

graph = graph_builder.compile()

initial_state = {"topic": "Why Consistency Matters More Than Talent", "attempts": 0}

for chunk in graph.stream(initial_state, stream_mode="debug"):
    step_no = chunk["step"]
    event_type = chunk["type"]
    payload = chunk["payload"]
    node_name = payload["name"]

    if event_type == "task":
        print(f"Step {step_no}")
        print(f"Running {node_name}")

    elif event_type == "task_result":
        print(f"Completed   {node_name}")

        # Show state updates (result of this node)
        result = payload.get("result")
        if result:
            print("State update:")
            for k, v in result.items():
                print(f"   {k}: {v}")
        print("-" * 100)
