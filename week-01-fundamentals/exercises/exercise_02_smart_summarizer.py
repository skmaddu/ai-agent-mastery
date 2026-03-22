"""
Exercise 2: Smart Summarizer
==============================
Difficulty: Beginner | Time: 2 hours

Task:
Build an LLM-powered text analyzer that returns structured output
with: summary, key_terms, and sentiment.

Instructions:
1. Set up your groq API key
2. Create a Pydantic model for the output schema
3. Use .with_structured_output() for guaranteed schema compliance
4. Test with at least 2 different paragraphs
5. Bonus: Add Phoenix tracing to view the LLM call

Run: python exercise_02_smart_summarizer.py
"""

import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Load config/.env via script location (works from any working directory)
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / "config" / ".env")
load_dotenv()


class TextAnalysis(BaseModel):
    """Structured output for text analysis."""

    summary: str = Field(
        description="A concise 2–4 sentence summary capturing the main ideas."
    )
    key_terms: List[str] = Field(
        description="5–10 important terms or short phrases from the text.",
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall emotional tone of the text.",
    )


def _make_llm():
    """Chat model from LLM_PROVIDER (default: groq)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0.2,
        )
    # Unknown provider: fall back to groq
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
    )


def analyze_text(text: str) -> TextAnalysis:
    """Analyze text and return structured insights.

    Args:
        text: The text to analyze

    Returns:
        TextAnalysis with summary, key_terms, and sentiment
    """
    if text is None or not str(text).strip():
        raise ValueError("Text must be non-empty.")

    llm = _make_llm()
    structured = llm.with_structured_output(TextAnalysis)

    messages = [
        SystemMessage(
            content=(
                "You extract structured analysis from user text. "
                "Key terms must be taken from or clearly implied by the text. "
                "Choose exactly one sentiment: positive, negative, or neutral."
            )
        ),
        HumanMessage(content=f"Analyze this text:\n\n{str(text).strip()}"),
    ]

    return structured.invoke(messages)


if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    print(f"Provider: {provider}\n")

    sample1 = """
    AI agents represent a paradigm shift in software development.
    Unlike traditional programs, agents can reason about their
    environment, use tools autonomously, and improve through
    self-reflection. Companies like Klarna have deployed agents
    that handle millions of customer interactions.
    """

    sample2 = """
    Supply chain delays and rising energy costs have squeezed margins
    for small manufacturers this quarter. Several plant closures were
    announced, and analysts expect a difficult year ahead for the sector.
    """

    print("--- Paragraph 1 ---")
    print("Analyzing text...")
    result = analyze_text(sample1)
    print(f"Summary: {result.summary}")
    print(f"Key Terms: {result.key_terms}")
    print(f"Sentiment: {result.sentiment}")

    print("\n--- Paragraph 2 ---")
    print("Analyzing text...")
    result2 = analyze_text(sample2)
    print(f"Summary: {result2.summary}")
    print(f"Key Terms: {result2.key_terms}")
    print(f"Sentiment: {result2.sentiment}")
