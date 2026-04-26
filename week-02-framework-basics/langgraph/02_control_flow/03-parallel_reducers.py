# Job Application Assistant – Running Parallel Analyses
# Added a candidate score

import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


class ScreeningState(TypedDict):
    cv_text: str
    job_description: str
    strengths: str
    weaknesses: str
    interview_questions: str
    screening_report: str
    candidate_score: Annotated[float, operator.add]


# Pydantic output schemas with descriptions and validation
class StrengthsOutput(BaseModel):
    strengths: str = Field(
        ..., description="Candidate's key strengths relevant to the role"
    )
    candidate_score: float = Field(
        ..., ge=0, le=1, description="Score between 0 and 1 indicating strength level"
    )


class WeaknessesOutput(BaseModel):
    weaknesses: str = Field(..., description="Candidate's key weaknesses or skill gaps")
    candidate_score: float = Field(
        ...,
        ge=-1,
        le=0,
        description="Score between -1 and 0 indicating weakness severity",
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Create structured LLM wrappers
structured_strengths_llm = llm.with_structured_output(StrengthsOutput)
structured_weaknesses_llm = llm.with_structured_output(WeaknessesOutput)


def extract_strengths(state: ScreeningState):
    response = structured_strengths_llm.invoke(f"""
    Given the candidate CV:  {state["cv_text"]}
    And the Job Description:  {state["job_description"]}
    Extract the candidate's key strengths relevant for this role
    and provide an overall candidate score between 0 and 1 reflecting these strengths.
    Respond with fields: strengths (string), candidate_score (float).
    """)
    return {
        "strengths": response.strengths,
        "candidate_score": response.candidate_score,
    }


def extract_weaknesses(state: ScreeningState):
    response = structured_weaknesses_llm.invoke(f"""
    Given the candidate CV:{state["cv_text"]}
    And the Job Description:{state["job_description"]}
    Extract the candidate's weaknesses or skill gaps for this role.
    and provide a candidate score between -1 and 0 reflecting these weaknesses.
    Respond with fields: weaknesses (string), candidate_score (float).
    """)
    return {
        "weaknesses": response.weaknesses,
        "candidate_score": response.candidate_score,
    }


def generate_interview_questions(state: ScreeningState):
    response = llm.invoke(f"""
    Given the candidate CV: {state["cv_text"]}
    And the Job Description:{state["job_description"]}
    Suggest 5 tailored interview questions.
    """)
    return {"interview_questions": response}


def create_screening_report(state: ScreeningState):
    response = llm.invoke(f"""
    Combine the following into a structured screening report for the recruiter:
    Candidate Strengths: {state["strengths"]}
    Candidate Weaknesses: {state["weaknesses"]}
    Candidate Score: {state["candidate_score"]}
    Suggested Interview Questions: {state["interview_questions"]}
    Also include a 1-line summary of what the final candidate score means.
    """)
    return {"screening_report": response}


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

cv_text = "Experienced software engineer with 5 years in backend development, Python, and cloud technologies."
job_description = "Looking for a backend developer skilled in Python, cloud platforms, and scalable system design."

final_state = graph.invoke({"cv_text": cv_text, "job_description": job_description})

print("\nScreening Report:\n", final_state["screening_report"])
