# Iterative Workflow: Generate, Evaluate, and Improve a LinkedIn Post

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


# ---- Evaluation schema ----
class Review(BaseModel):
    verdict: Literal["ready to post", "needs rewrite"] = Field(
        description="Decide if the LinkedIn post is ready or needs improvement."
    )
    feedback: str = Field(
        description="If rewrite needed, suggest how to improve tone, clarity, or engagement."
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
        prompt = (
            f"Write an engaging, professional LinkedIn post about '{state['topic']}'."
        )

    response = llm.invoke(prompt)
    return {"post": response}


def post_evaluator(state: State):
    """Evaluate the LinkedIn post."""
    review = evaluator_llm.invoke(f"Evaluate this LinkedIn post:\n{state['post']}")
    return {"quality": review.verdict, "feedback": review.feedback}


# ---- Router ----
def decide_next(state: State):
    if state["quality"] == "ready to post":
        return "Accept"
    else:
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

with open("evaluator_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


initial_state = {"topic": "Personal Branding for Creators"}
final_state = graph.invoke(initial_state)

print("Final LinkedIn Post:\n", final_state["post"])
