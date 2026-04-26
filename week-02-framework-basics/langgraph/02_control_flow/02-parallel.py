# Job Application Assistant – Running Parallel Analyses

from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


class ScreeningState(TypedDict):
    cv_text: str
    job_description: str
    strengths: str
    weaknesses: str
    interview_questions: str
    screening_report: str


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def extract_strengths(state: ScreeningState):
    response = llm.invoke(f"""
    Given the candidate CV:  {state["cv_text"]}
    And the Job Description:  {state["job_description"]}
    Extract the candidate's key strengths relevant for this role.
    """)
    return {"strengths": response.content}


def extract_weaknesses(state: ScreeningState):
    response = llm.invoke(f"""
    Given the candidate CV:{state["cv_text"]}
    And the Job Description:{state["job_description"]}
    Extract the candidate's weaknesses or skill gaps for this role.
    """)
    return {"weaknesses": response.content}


def generate_interview_questions(state: ScreeningState):
    response = llm.invoke(f"""
    Given the candidate CV: {state["cv_text"]}
    And the Job Description:{state["job_description"]}
    Suggest 5 tailored interview questions.
    """)
    return {"interview_questions": response.content}


def create_screening_report(state: ScreeningState):
    response = llm.invoke(f"""
    Combine the following into a structured screening report for the recruiter:
    Candidate Strengths: {state["strengths"]}
    Candidate Weaknesses: {state["weaknesses"]}
    Suggested Interview Questions: {state["interview_questions"]}
    """)
    return {"screening_report": response.content}


graph_builder = StateGraph(ScreeningState)

graph_builder.add_node("strengths", extract_strengths)
graph_builder.add_node("weaknesses", extract_weaknesses)
graph_builder.add_node("questions", generate_interview_questions)
graph_builder.add_node("merge_report", create_screening_report)

graph_builder.add_edge(START, "strengths")
graph_builder.add_edge(START, "weaknesses")
graph_builder.add_edge(START, "questions")
graph_builder.add_edge("strengths", "merge_report")
graph_builder.add_edge("weaknesses", "merge_report")
graph_builder.add_edge("questions", "merge_report")
graph_builder.add_edge("merge_report", END)

graph = graph_builder.compile()
with open("parallel.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

cv_text = "Experienced software engineer with 5 years in backend development, Python, and cloud technologies."
job_description = "Looking for a backend developer skilled in Python, cloud platforms, and scalable system design."

final_state = graph.invoke({"cv_text": cv_text, "job_description": job_description})

print("\nScreening Report:\n", final_state["screening_report"])
