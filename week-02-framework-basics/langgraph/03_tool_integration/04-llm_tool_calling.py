# Shows how an LLM can decide when to call tools after binding them using bind_tools()

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

tavily_tool = TavilySearch(max_results=2)


@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers and returns the sum.
    Args:
        a: The first number to add
        b: The second number to add
    """
    return a + b


@tool("subtract", description="Subtract b from a and return the result.")
def sub(a: int, b: int) -> int:
    """Return the difference of two numbers."""
    return a - b


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

tools = [tavily_tool, add, sub]
llm_with_tools = llm.bind_tools(tools)

queries = [
    "Give one line definition of photosynthesis",
    "Who won the women's cricket world cup in 2025?",
    "Find the sum of 67 and 450",
    "Decrease 56 by 8",
    "What are the 7 colours in a rainbow",
]

for query in queries:
    print(query)
    response = llm_with_tools.invoke(query)
    print(response, end="\n\n")
