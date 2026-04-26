# Demonstrates defining tools and invoking them manually

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

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

# Inspect tools
print(tavily_tool.name)
print(add.name)
print(sub.name)

print("\n=== Tool Descriptions ===\n")

print(tavily_tool.description)
print(add.description)
print(sub.description)

# Invoke tools
tool_output = tavily_tool.invoke("Who won the women's cricket world cup in 2025?")
print(tool_output)

tool_output = add.invoke({"a": 5, "b": 3})
print(tool_output)

tool_output = sub.invoke({"a": 5, "b": 3})
print(tool_output)
