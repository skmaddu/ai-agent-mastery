import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 14: Output Parsing with Validation
============================================
LLMs return free-form text, but agents need STRUCTURED data to
pass between nodes, store in databases, and display in UIs.

This example covers 3 approaches to getting reliable structured
output from LLMs:

  1. JSON mode — ask the LLM to return JSON (simple but fragile)
  2. Pydantic with_structured_output — schema-enforced (recommended)
  3. Retry parsing — handle malformed output gracefully

Run: python week-03-basic-patterns/examples/example_14_output_parsing_validation.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ==============================================================
# Setup
# ==============================================================

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


llm = get_llm()


# ==============================================================
# APPROACH 1: Manual JSON Parsing (Simple but Fragile)
# ==============================================================

def demo_manual_json():
    """Parse JSON from LLM output manually."""

    print("=" * 60)
    print("APPROACH 1: Manual JSON Parsing")
    print("=" * 60)

    messages = [
        SystemMessage(content=(
            "Analyze the given product review and respond in this EXACT JSON format "
            "(no markdown, no code blocks, just raw JSON):\n"
            '{"product": "...", "rating": 1-5, "pros": ["..."], "cons": ["..."], "summary": "..."}'
        )),
        HumanMessage(content=(
            "Review: 'I bought this wireless mouse last month. The battery lasts forever "
            "and the tracking is smooth on any surface. However, the scroll wheel feels "
            "cheap and it disconnects from Bluetooth occasionally. Overall good for the price.'"
        )),
    ]

    response = llm.invoke(messages)
    raw_output = response.content.strip()

    print(f"\n  Raw LLM output:")
    print(f"    {raw_output[:200]}")

    # -- Parse the JSON ----------------------------------------
    # This is fragile — LLMs sometimes wrap JSON in markdown code blocks
    try:
        # Strip markdown code blocks if present
        cleaned = raw_output
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(cleaned)
        print(f"\n  [OK] Parsed successfully:")
        print(f"    Product: {parsed.get('product', 'N/A')}")
        print(f"    Rating:  {parsed.get('rating', 'N/A')}/5")
        print(f"    Pros:    {parsed.get('pros', [])}")
        print(f"    Cons:    {parsed.get('cons', [])}")
    except json.JSONDecodeError as e:
        print(f"\n  [FAIL] JSON parsing failed: {e}")
        print(f"    This is why manual parsing is fragile!")

    print(f"\n  [WARN]  Problems with manual JSON parsing:")
    print(f"     - LLM might wrap in markdown code blocks")
    print(f"     - LLM might add extra text before/after JSON")
    print(f"     - No type validation (rating could be 'great' instead of a number)")
    print(f"     - No field validation (missing fields go unnoticed)")


# ==============================================================
# APPROACH 2: Pydantic + with_structured_output (Recommended)
# ==============================================================

# -- Define your output schema as a Pydantic model -------------
# Pydantic models define the EXACT shape of the output,
# with types, defaults, validation rules, and documentation.

class ProductReview(BaseModel):
    """Structured analysis of a product review."""

    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Rating from 1 (terrible) to 5 (excellent)", ge=1, le=5)
    pros: list[str] = Field(description="List of positive aspects mentioned", min_length=1)
    cons: list[str] = Field(description="List of negative aspects mentioned", default_factory=list)
    summary: str = Field(description="One-sentence summary of the review")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")

    # Custom validation — Pydantic runs this automatically
    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        allowed = {"positive", "negative", "mixed"}
        if v.lower() not in allowed:
            raise ValueError(f"Sentiment must be one of {allowed}, got '{v}'")
        return v.lower()


class ResearchFinding(BaseModel):
    """A structured research finding."""

    title: str = Field(description="Title of the finding")
    key_points: list[str] = Field(description="3-5 key points", min_length=1, max_length=7)
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0",
        ge=0.0, le=1.0,
    )
    sources_needed: bool = Field(
        description="Whether this finding needs additional source verification",
        default=True,
    )
    follow_up_questions: list[str] = Field(
        description="Questions for deeper research",
        default_factory=list,
    )


def demo_pydantic_structured():
    """Demonstrate Pydantic with_structured_output."""

    print(f"\n{'='*60}")
    print("APPROACH 2: Pydantic + with_structured_output")
    print("=" * 60)

    # Bind the Pydantic model to the LLM — it GUARANTEES the output matches
    structured_llm = llm.with_structured_output(ProductReview)

    review_text = (
        "I bought this wireless mouse last month. The battery lasts forever "
        "and the tracking is smooth on any surface. However, the scroll wheel "
        "feels cheap and it disconnects from Bluetooth occasionally. Overall "
        "good value for the price at $25."
    )

    try:
        result = structured_llm.invoke(
            f"Analyze this product review:\n{review_text}"
        )

        print(f"\n  [OK] Structured output (ProductReview):")
        print(f"    Product:   {result.product_name}")
        print(f"    Rating:    {'*' * result.rating} ({result.rating}/5)")
        print(f"    Sentiment: {result.sentiment}")
        print(f"    Pros:      {result.pros}")
        print(f"    Cons:      {result.cons}")
        print(f"    Summary:   {result.summary}")
        print(f"\n    Type of result: {type(result).__name__}")
        print(f"    Access fields:  result.rating -> {result.rating} (int, validated 1-5)")

    except Exception as e:
        print(f"\n  [FAIL] Error: {e}")

    print(f"\n  [OK] Why Pydantic is better than manual JSON:")
    print(f"     - Type enforcement (rating MUST be int 1-5, not a string)")
    print(f"     - Field validation (sentiment MUST be positive/negative/mixed)")
    print(f"     - IDE autocomplete (result.rating, not result['rating'])")
    print(f"     - Default values (cons defaults to empty list if none found)")
    print(f"     - Documentation (Field descriptions help the LLM)")


# -- Demo with ResearchFinding ---------------------------------

def demo_research_finding():
    """Show Pydantic structured output for research."""

    print(f"\n{'-'*60}")
    print("  ResearchFinding structured output:")
    print(f"{'-'*60}")

    structured_llm = llm.with_structured_output(ResearchFinding)

    try:
        result = structured_llm.invoke(
            "Provide a research finding about the impact of AI on education."
        )

        print(f"\n    Title: {result.title}")
        print(f"    Confidence: {result.confidence:.0%}")
        print(f"    Key points:")
        for point in result.key_points:
            print(f"      - {point}")
        print(f"    Needs verification: {result.sources_needed}")
        if result.follow_up_questions:
            print(f"    Follow-up questions:")
            for q in result.follow_up_questions[:3]:
                print(f"      ? {q}")
    except Exception as e:
        print(f"    Error: {e}")


# ==============================================================
# APPROACH 3: Retry Parsing (Handle Failures Gracefully)
# ==============================================================
# Even with_structured_output can fail with some models.
# A retry parser catches failures and tries again.

def parse_with_retry(llm_instance, schema_class, prompt: str, max_retries: int = 3) -> Optional[BaseModel]:
    """Try to get structured output, retrying on failure.

    Args:
        llm_instance: The LLM to call
        schema_class: The Pydantic model class
        prompt: The prompt to send
        max_retries: Maximum attempts

    Returns:
        Parsed Pydantic model or None if all retries fail
    """
    structured_llm = llm_instance.with_structured_output(schema_class)

    for attempt in range(1, max_retries + 1):
        try:
            result = structured_llm.invoke(prompt)
            print(f"    Attempt {attempt}: [OK] Success")
            return result
        except Exception as e:
            print(f"    Attempt {attempt}: [FAIL] Failed ({type(e).__name__}: {str(e)[:80]})")
            if attempt < max_retries:
                # Add more explicit instructions on retry
                prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: You must respond with valid structured data matching "
                    f"the schema exactly. Previous attempt failed."
                )

    print(f"    All {max_retries} attempts failed. Returning None.")
    return None


def demo_retry_parsing():
    """Demonstrate retry parsing."""

    print(f"\n{'='*60}")
    print("APPROACH 3: Retry Parsing")
    print("=" * 60)

    print(f"\n  Attempting to parse with retry logic:")
    result = parse_with_retry(
        llm,
        ProductReview,
        "Analyze: 'Great laptop, fast processor, but heavy and expensive at $2000'",
        max_retries=3,
    )

    if result:
        print(f"\n  Result: {result.product_name} — {result.rating}/5 ({result.sentiment})")
    else:
        print(f"\n  Could not parse after retries. Consider:")
        print(f"  - Simplifying the Pydantic model")
        print(f"  - Using a more capable LLM")
        print(f"  - Falling back to manual JSON parsing")


# ==============================================================
# Summary: Choosing Your Parsing Strategy
# ==============================================================

def summary():
    print(f"\n{'='*60}")
    print("Summary: Choosing Your Parsing Strategy")
    print("=" * 60)
    print(f"""
  {'Approach':<30} {'Reliability':<15} {'Best For'}
  {'-'*30} {'-'*15} {'-'*25}
  {'Manual JSON':<30} {'Low':<15} {'Quick prototypes'}
  {'Pydantic structured_output':<30} {'High':<15} {'Production agents (use this!)'}
  {'Retry parsing':<30} {'Very High':<15} {'Critical data extraction'}

  [TIP] Start with Pydantic + with_structured_output for everything.
     Add retry logic for mission-critical parsing.
     Use manual JSON only for quick experiments.
""")


if __name__ == "__main__":
    print("Example 14: Output Parsing with Validation")
    print("=" * 60)

    demo_manual_json()
    demo_pydantic_structured()
    demo_research_finding()
    demo_retry_parsing()
    summary()
