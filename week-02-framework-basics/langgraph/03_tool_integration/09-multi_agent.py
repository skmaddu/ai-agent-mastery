# A multi-agent system where a controller agent delegates tasks to specialized math and research agents
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# =======================
# Sub-Agent 1: Math Agent
# =======================


@tool(description="Add two numbers and return the sum.")
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


@tool("subtract", description="Subtract b from a and return the result.")
def sub(a: int, b: int) -> int:
    """Return a - b."""
    return a - b


math_agent = create_agent(
    model=llm,
    tools=[add, sub],
    system_prompt="""You are a math expert. Always use the provided tools for calculations and respond in plain text only.""",
)


@tool("math_helper", description="Use this for arithmetic questions.")
def call_math_agent(query: str) -> str:
    result = math_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# ===========================
# Sub-Agent 2: Research Agent
# ===========================

tavily_tool = TavilySearch(max_results=2)

research_agent = create_agent(
    model=llm,
    tools=[tavily_tool],
    system_prompt="""You are a research specialist. Always use TavilySearch for factual and recent information.""",
)


@tool(
    "search_helper",
    description="Use this to retrieve up-to-date information from the web.",
)
def call_research_agent(query: str) -> str:
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# ======================
# Main Controller Agent
# ======================

controller_agent = create_agent(
    model=llm,
    tools=[call_math_agent, call_research_agent],
    system_prompt="""You are a controller agent. Decide which agent should perform the task. 
Always provide the final answer in plain natural language.""",
)


# ============
# Test Queries
# =============

queries = [
    "What is 56 - 19?",
    "What is the young one of a cat called",
    "What is the capital of France?",
    "What is 123 + 456?",
    "Who won the first nobel prize in physics?",
    "Who won the cricket world cup in 2023?",
]

for q in queries:
    print(f"\nUser: {q}")
    response = controller_agent.invoke({"messages": [{"role": "user", "content": q}]})

    for message in response["messages"]:
        message.pretty_print()
