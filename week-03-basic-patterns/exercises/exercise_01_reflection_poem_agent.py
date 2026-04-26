"""
Exercise 1: Reflection Pattern — Poetry Refinement Agent
=========================================================
Difficulty: Intermediate | Time: 2 hours

Task:
Build a LangGraph agent that generates a poem, critiques it for
rhyme and structure, and refines it iteratively. The agent should
use the reflection pattern: generate -> critique -> refine -> repeat.

Instructions:
1. Define the ReflectionState with fields for the poem, critique,
   score, iteration count, and history
2. Implement the generate_node that creates/refines a poem
3. Implement the critique_node that scores the poem on:
   - Rhyme quality (do lines rhyme where expected?)
   - Structure (consistent line count and rhythm?)
   - Creativity (interesting word choices and imagery?)
4. Implement the should_continue routing function with:
   - Quality threshold of 7/10 to stop
   - Max iterations of 4 as safety limit
5. Build the StateGraph and connect the nodes
6. Test with all 3 topics below

Hints:
- Look at example_02_reflection_langgraph.py for the full pattern
- The critique prompt should ask for SPECIFIC feedback, not just a score
- Use different system prompts for generation vs critique
- Parse the critic's score from its response (see example_02 for parsing)
- Remember to handle both first-time generation AND refinement in generate_node

Run: python week-03-basic-patterns/exercises/exercise_01_reflection_poem_agent.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict


# ==============================================================
# Step 1: Define the State
# ==============================================================
# TODO: Create a TypedDict called PoemReflectionState with:
#   - topic: str           (the poem subject)
#   - current_poem: str    (the latest version of the poem)
#   - critique: str        (latest critique feedback)
#   - score: int           (quality score 1-10)
#   - iteration: int       (current iteration number)
#   - max_iterations: int  (safety limit)
#   - history: list        (log of poems and critiques)

# class PoemReflectionState(TypedDict):
#     ...


# ==============================================================
# Step 2: Set up the LLM
# ==============================================================
# TODO: Create the LLM based on LLM_PROVIDER environment variable
# Use groq as default, fallback to openai
# Set temperature to 0.8 (slightly creative for poetry)

# provider = os.getenv("LLM_PROVIDER", "groq").lower()
# if provider == "groq":
#     from langchain_groq import ChatGroq
#     llm = ChatGroq(...)
# else:
#     from langchain_openai import ChatOpenAI
#     llm = ChatOpenAI(...)


# ==============================================================
# Step 3: Implement the Generate Node
# ==============================================================
# TODO: Create generate_node(state) that:
#   - On FIRST call (iteration == 0): Generate a new poem about the topic
#     System prompt: "You are a skilled poet. Write a short poem (4-8 lines)
#     about the given topic. Use vivid imagery and try to include rhymes."
#   - On SUBSEQUENT calls: Refine the poem using critique feedback
#     System prompt: "You are a skilled poet. Revise the poem to address
#     the critique. Keep what works, fix what doesn't. Output ONLY the
#     revised poem, nothing else."
#   - Return dict updating: current_poem, iteration, history

# def generate_node(state: PoemReflectionState) -> dict:
#     iteration = state.get("iteration", 0)
#     if iteration == 0:
#         # First call — generate from scratch
#         messages = [...]
#     else:
#         # Refinement — incorporate critique
#         messages = [...]
#     response = llm.invoke(messages)
#     poem = response.content
#     return {
#         "current_poem": poem,
#         "iteration": iteration + 1,
#         "history": state.get("history", []) + [{"type": "poem", "content": poem}],
#     }


# ==============================================================
# Step 4: Implement the Critique Node
# ==============================================================
# TODO: Create critique_node(state) that:
#   - Sends the current poem to the LLM with a CRITIC system prompt
#   - The critic should evaluate on 3 specific criteria:
#     1. Rhyme quality (1-10)
#     2. Structure and rhythm (1-10)
#     3. Creativity and imagery (1-10)
#   - Ask for an overall score and specific suggestions
#   - Parse the score from the response
#   - Return dict updating: critique, score, history
#
# Critic system prompt suggestion:
#   "You are a poetry critic. Evaluate the poem on these criteria:
#    1. Rhyme quality: Do the lines rhyme effectively? (1-10)
#    2. Structure: Is there consistent rhythm and line count? (1-10)
#    3. Creativity: Are the word choices and imagery vivid? (1-10)
#    Respond with:
#    SCORE: <overall 1-10>
#    FEEDBACK: <specific suggestions for improvement>"

# def critique_node(state: PoemReflectionState) -> dict:
#     messages = [...]
#     response = llm.invoke(messages)
#     # Parse score and feedback from response
#     # ...
#     return {
#         "critique": ...,
#         "score": ...,
#         "history": state.get("history", []) + [{"type": "critique", "score": ..., "feedback": ...}],
#     }


# ==============================================================
# Step 5: Implement the Routing Function
# ==============================================================
# TODO: Create should_continue(state) that returns:
#   - "done" if score >= 7 (quality gate passed)
#   - "done" if iteration >= max_iterations (safety limit)
#   - "refine" otherwise (loop back to generator)

QUALITY_THRESHOLD = 7

# def should_continue(state: PoemReflectionState) -> str:
#     ...


# ==============================================================
# Step 6: Build and Compile the Graph
# ==============================================================
# TODO: Create the StateGraph with:
#   1. Three nodes: "generate", "critique", "done"
#   2. Entry point: "generate"
#   3. Edge: generate -> critique (always)
#   4. Conditional edges from critique: "refine" -> generate, "done" -> done
#   5. Edge: done -> END
#   6. The "done" node can simply print the final result and return {}

# def done_node(state: PoemReflectionState) -> dict:
#     print(f"\n[PASS] Final poem (score: {state['score']}/10, {state['iteration']} iterations)")
#     return {}

# graph = StateGraph(PoemReflectionState)
# ... add nodes, edges ...
# app = graph.compile()


# ==============================================================
# Step 7: Run Function
# ==============================================================
# TODO: Create a run_poem_reflection(topic, max_iterations=4) function
# that invokes the graph with proper initial state

# def run_poem_reflection(topic: str, max_iterations: int = 4) -> dict:
#     result = app.invoke({
#         "topic": topic,
#         "current_poem": "",
#         "critique": "",
#         "score": 0,
#         "iteration": 0,
#         "max_iterations": max_iterations,
#         "history": [],
#     })
#     return result


# ==============================================================
# Test your implementation
# ==============================================================

if __name__ == "__main__":
    print("Exercise 1: Reflection Pattern — Poetry Refinement Agent")
    print("=" * 60)

    # Test 1: Nature topic
    print("\nTest 1: 'A sunset over the ocean'")
    # result = run_poem_reflection("A sunset over the ocean")
    # print(f"\nFinal Poem:\n{result['current_poem']}")

    # Test 2: Technology topic
    print("\nTest 2: 'The loneliness of a robot'")
    # result = run_poem_reflection("The loneliness of a robot")
    # print(f"\nFinal Poem:\n{result['current_poem']}")

    # Test 3: Abstract topic
    print("\nTest 3: 'The passage of time'")
    # result = run_poem_reflection("The passage of time")
    # print(f"\nFinal Poem:\n{result['current_poem']}")

    print("\n(Uncomment the test code above after implementing!)")
    print("\nExpected behavior:")
    print("  - Each poem should improve across iterations")
    print("  - Score should increase (or loop stops at max iterations)")
    print("  - History should show the evolution of the poem")
