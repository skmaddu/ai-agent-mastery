"""
Exercise 3: Output Parsing Lab — Structured Data Extraction Agent
==================================================================
Difficulty: Intermediate-Advanced | Time: 2.5 hours

Task:
Build an agent that reads product reviews and extracts structured data
using Pydantic models. The agent should:
  1. Parse reviews into a structured format (product, rating, pros, cons)
  2. Validate the parsed output (reject malformed data)
  3. Handle parsing failures with retry logic
  4. Track success/failure rates across multiple reviews

This combines output parsing (Example 14) with evaluation (Example 10)
and guardrails (Example 13) into a complete data pipeline.

Instructions:
1. Define the ProductReview Pydantic model with validation
2. Build the extraction function using with_structured_output
3. Add retry logic for failed extractions
4. Process all 10 test reviews and track success rate
5. Report extraction quality metrics

Hints:
- Look at example_14_output_parsing_validation.py for Pydantic patterns
- Use field_validator for custom validation (e.g., rating must be 1-5)
- The retry function should add stricter instructions on each retry
- Track: total reviews, successful extractions, failed extractions,
  average rating, most common pros/cons

Run: python week-03-basic-patterns/exercises/exercise_03_data_extraction_agent.py
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional


# ==============================================================
# Step 1: Define the Pydantic Model
# ==============================================================
# TODO: Create a ProductReview Pydantic model with these fields:
#   - product_name: str (required, description for the LLM)
#   - rating: int (required, must be 1-5, use Field with ge=1, le=5)
#   - pros: list[str] (positive aspects, at least 1 item)
#   - cons: list[str] (negative aspects, can be empty)
#   - summary: str (one-sentence summary)
#   - recommend: bool (would the reviewer recommend this product?)
#
# Add a field_validator for 'rating' that ensures it's between 1-5
# Add a field_validator for 'summary' that ensures it's not empty

# class ProductReview(BaseModel):
#     """Structured extraction of a product review."""
#     product_name: str = Field(description="...")
#     rating: int = Field(description="...", ge=1, le=5)
#     pros: list[str] = Field(description="...", min_length=1)
#     cons: list[str] = Field(description="...", default_factory=list)
#     summary: str = Field(description="...")
#     recommend: bool = Field(description="...")
#
#     @field_validator("rating")
#     @classmethod
#     def validate_rating(cls, v):
#         if not 1 <= v <= 5:
#             raise ValueError(f"Rating must be 1-5, got {v}")
#         return v
#
#     @field_validator("summary")
#     @classmethod
#     def validate_summary(cls, v):
#         if not v.strip():
#             raise ValueError("Summary cannot be empty")
#         return v.strip()


# ==============================================================
# Step 2: Set Up the LLM
# ==============================================================
# TODO: Create the LLM with temperature=0 (we want consistent parsing)

# provider = os.getenv("LLM_PROVIDER", "groq").lower()
# if provider == "groq":
#     from langchain_groq import ChatGroq
#     llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
# else:
#     from langchain_openai import ChatOpenAI
#     llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


# ==============================================================
# Step 3: Implement Extraction with Retry
# ==============================================================
# TODO: Create extract_review(review_text, max_retries=3) that:
#   1. Creates a structured LLM using llm.with_structured_output(ProductReview)
#   2. Calls it with the review text
#   3. If it fails, retries with a stricter prompt
#   4. Returns the ProductReview or None if all retries fail
#   5. Prints status for each attempt

# def extract_review(review_text: str, max_retries: int = 3) -> Optional[ProductReview]:
#     """Extract structured data from a product review with retry logic.
#
#     Args:
#         review_text: The raw review text to parse
#         max_retries: Maximum number of extraction attempts
#
#     Returns:
#         ProductReview if successful, None if all retries fail
#     """
#     structured_llm = llm.with_structured_output(ProductReview)
#
#     for attempt in range(1, max_retries + 1):
#         try:
#             # Build the prompt — add more detail on retry
#             if attempt == 1:
#                 prompt = f"Extract structured data from this product review:\n\n{review_text}"
#             else:
#                 prompt = (
#                     f"Extract structured data from this product review. "
#                     f"Previous attempt failed. Be very precise:\n"
#                     f"- rating MUST be an integer from 1 to 5\n"
#                     f"- pros MUST have at least one item\n"
#                     f"- summary MUST be a single sentence\n"
#                     f"- recommend MUST be true or false\n\n"
#                     f"Review:\n{review_text}"
#                 )
#
#             result = structured_llm.invoke(prompt)
#             print(f"    Attempt {attempt}: [OK] Success")
#             return result
#
#         except Exception as e:
#             print(f"    Attempt {attempt}: [FAIL] {type(e).__name__}: {str(e)[:60]}")
#
#     return None


# ==============================================================
# Step 4: Process All Reviews and Track Metrics
# ==============================================================
# TODO: Create process_all_reviews(reviews) that:
#   1. Iterates through all reviews
#   2. Calls extract_review for each one
#   3. Tracks: successful count, failed count, ratings, pros, cons
#   4. Returns a summary report

# def process_all_reviews(reviews: list[str]) -> dict:
#     """Process all reviews and return metrics.
#
#     Returns:
#         dict with total, success_count, fail_count, avg_rating,
#         all_pros, all_cons, results list
#     """
#     results = []
#     successful = []
#
#     for i, review in enumerate(reviews, 1):
#         print(f"\n  Review {i}/{len(reviews)}:")
#         print(f"    \"{review[:80]}...\"" if len(review) > 80 else f"    \"{review}\"")
#
#         extracted = extract_review(review)
#         results.append(extracted)
#
#         if extracted:
#             successful.append(extracted)
#             print(f"    -> {extracted.product_name} | {'*' * extracted.rating} | "
#                   f"{'Recommended' if extracted.recommend else 'Not recommended'}")
#
#     # Calculate metrics
#     # TODO: Calculate success rate, average rating, common pros/cons
#     # ...
#
#     return {
#         "total": len(reviews),
#         "success_count": len(successful),
#         "fail_count": len(reviews) - len(successful),
#         "results": results,
#         # Add: avg_rating, top_pros, top_cons
#     }


# ==============================================================
# Test Reviews (10 reviews to process)
# ==============================================================

TEST_REVIEWS = [
    # 1. Clear positive review
    "I love this wireless keyboard! The keys are clicky and satisfying, battery lasts 6 months, "
    "and the Bluetooth connection is rock solid. Only downside is it's a bit heavy. Highly recommend!",

    # 2. Clear negative review
    "Terrible headphones. Sound quality is muddy, the ear cups are uncomfortable after 30 minutes, "
    "and the noise cancellation is basically non-existent. Returned within a week. Save your money.",

    # 3. Mixed review
    "The smart watch has amazing fitness tracking — heart rate, sleep, and GPS are accurate. "
    "But the battery only lasts 2 days (advertised as 5), and the screen is hard to read in sunlight. "
    "Good for gym use, not great for everyday wear.",

    # 4. Short review
    "Good laptop. Fast, light, great screen. Expensive though. 4/5.",

    # 5. Detailed review
    "This standing desk is fantastic for the price ($350). Smooth motor, goes from 28 to 48 inches, "
    "and remembers 3 height presets. The desktop surface is spacious at 60x30 inches. "
    "Assembly took 45 minutes and instructions were clear. My only complaint is the cable management "
    "tray feels flimsy. I've been using it daily for 3 months with no issues. Would definitely buy again.",

    # 6. Emoji-heavy review
    "Amazing camera!! Best phone photos I've ever taken. Night mode is incredible. "
    "Video stabilization works perfectly. The zoom is just okay though, gets grainy past 10x. "
    "Still the best camera phone I've used.",

    # 7. Comparative review
    "Switched from the previous model to this one. The new version is 30% faster and the screen is "
    "noticeably brighter. However, they removed the headphone jack (annoying) and the price went up $100. "
    "If you have last year's model, probably not worth upgrading.",

    # 8. Minimal review
    "Works as expected. No complaints. 3 stars.",

    # 9. Technical review
    "The router handles 802.11ax (WiFi 6E) beautifully. Got 1.2 Gbps on the 6GHz band at 10 feet, "
    "and 450 Mbps through two walls on 5GHz. Setup through the app took 5 minutes. QoS and parental "
    "controls are intuitive. Mesh capability works great with 2 nodes covering my 2500 sq ft house. "
    "Pricey at $350 but worth it for the speed and coverage.",

    # 10. Sarcastic/difficult review
    "Oh great, another 'smart' device that needs an app, an account, a firmware update, and a "
    "sacrifice to the tech gods before it works. Once I got past the 20-minute setup, the air purifier "
    "actually works well — air quality noticeably improved. Filter replacement is easy. But seriously, "
    "why does an air purifier need WiFi?",
]


# ==============================================================
# Test your implementation
# ==============================================================

if __name__ == "__main__":
    print("Exercise 3: Structured Data Extraction Agent")
    print("=" * 60)

    # Process all reviews
    # report = process_all_reviews(TEST_REVIEWS)

    # Print summary report
    # print(f"\n{'='*60}")
    # print("EXTRACTION REPORT")
    # print(f"{'='*60}")
    # print(f"  Total reviews:     {report['total']}")
    # print(f"  Successful:        {report['success_count']}")
    # print(f"  Failed:            {report['fail_count']}")
    # print(f"  Success rate:      {report['success_count']/report['total']:.0%}")
    # if 'avg_rating' in report:
    #     print(f"  Average rating:    {report['avg_rating']:.1f}/5")

    print("\n(Uncomment the test code above after implementing!)")
    print("\nSuccess criteria:")
    print("  - Parse success rate >= 80% (8 out of 10 reviews)")
    print("  - All ProductReview fields populated correctly")
    print("  - Retry logic handles at least 1 failure gracefully")
    print("  - Metrics report prints successfully")
