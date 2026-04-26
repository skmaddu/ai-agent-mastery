# High-level agent example using create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

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

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.Use tools for web search or math calculations.",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find temperature of Bangalore and then add 5 to it",
            }
        ]
    }
)
print(result)

for msg in result["messages"]:
    msg.pretty_print()

# Extract and print only the assistant’s natural-language reply
reply = result["messages"][-1].content
print("\n\nAssistant:", reply)
