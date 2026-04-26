# Sequential workflow using an LLM to generate a passage, key points, and quiz questions.

from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()  # Load environment variables from .env file


class WorkflowState(TypedDict):
    topic: str
    passage: str
    key_points: str
    questions: str


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def generate_passage(state: WorkflowState):
    topic = state["topic"]
    response = llm.invoke(f"Write a short informative passage about: {topic}")
    print(response)
    return {"passage": response.content}


def extract_key_points(state: WorkflowState):
    passage = state["passage"]
    response = llm.invoke(f"Extract 5 key points from this passage:\n\n{passage}")
    return {"key_points": response.content}


def generate_questions(state: WorkflowState):
    points = state["key_points"]
    response = llm.invoke(
        f"Based on these key points, generate 5 short quiz-style questions:\n\n{points}"
    )
    return {"questions": response.content}


graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("generate_passage", generate_passage)
graph_builder.add_node("extract_key_points", extract_key_points)
graph_builder.add_node("generate_questions", generate_questions)

graph_builder.add_edge(START, "generate_passage")
graph_builder.add_edge("generate_passage", "extract_key_points")
graph_builder.add_edge("extract_key_points", "generate_questions")
graph_builder.add_edge("generate_questions", END)

graph = graph_builder.compile()

with open("sequential.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

final_state = graph.invoke({"topic": "Artificial Intelligence in Education"})

print("Topic:\n", final_state["topic"])
print("Generated passage:\n", final_state["passage"])
print("\nKey Points:\n", final_state["key_points"])
print("\nQuestions:\n", final_state["questions"])
