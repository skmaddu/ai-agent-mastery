"""
Example 4: Tool Use Pattern — What Makes a Good Tool
======================================================
Tools give agents the ability to interact with the real world.
But not all tools are created equal. This example teaches:

1. What makes a GOOD vs BAD tool definition
2. How the LLM decides which tool to call (via schemas)
3. Tool design principles: naming, descriptions, parameters
4. Error handling in tools
5. Tool composition (combining simple tools)

No LLM calls — this focuses on understanding tool DESIGN so that
when you build tools in Examples 5 & 6, you know why they work.

Run: python week-03-basic-patterns/examples/example_04_tool_use_concepts.py
"""

import json


# ==============================================================
# PART 1: Good vs Bad Tool Definitions
# ==============================================================
# The LLM reads your tool's NAME, DESCRIPTION, and PARAMETER SCHEMA
# to decide when and how to call it. Poor definitions = poor decisions.

def demonstrate_good_vs_bad_tools():
    """Show why tool definitions matter for LLM decision-making."""

    print("=" * 60)
    print("PART 1: Good vs Bad Tool Definitions")
    print("=" * 60)

    # -- BAD Tool Definition ----------------------------------
    # The LLM sees this schema and has no idea when to use it:

    bad_tool = {
        "name": "do_thing",              # Vague name — what thing?
        "description": "Does a thing",   # Useless description
        "parameters": {
            "x": "str",                  # What is x? No hint.
            "y": "str",                  # What is y? No hint.
        },
    }

    print("\n[BAD] BAD Tool Definition:")
    print(f"   Name: {bad_tool['name']}")
    print(f"   Description: {bad_tool['description']}")
    print(f"   Parameters: {bad_tool['parameters']}")
    print("   Problem: The LLM cannot figure out WHEN to call this tool")
    print("   or WHAT to pass as arguments.")

    # -- GOOD Tool Definition ---------------------------------
    # Clear name, detailed description, typed parameters with docs:

    good_tool = {
        "name": "search_web",
        "description": (
            "Search the web for current information about a topic. "
            "Use this when you need facts, news, or data that may not "
            "be in your training data. Returns top search results with "
            "titles and snippets."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query (e.g., 'latest AI research 2026')",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5)",
                "default": 5,
            },
        },
    }

    print("\n[OK] GOOD Tool Definition:")
    print(f"   Name: {good_tool['name']}")
    print(f"   Description: {good_tool['description'][:80]}...")
    print(f"   Parameters: query (string), max_results (integer, default 5)")
    print("   Why better: The LLM knows WHEN to call it (needs current info),")
    print("   WHAT to pass (a search query), and WHAT to expect back (results).")


# ==============================================================
# PART 2: The 5 Rules of Tool Design
# ==============================================================

def demonstrate_tool_design_rules():
    """The 5 rules for designing effective agent tools."""

    print(f"\n{'='*60}")
    print("PART 2: The 5 Rules of Tool Design")
    print(f"{'='*60}")

    rules = [
        {
            "rule": "1. Name it like a verb phrase",
            "bad": "weather",
            "good": "get_current_weather",
            "why": "The LLM needs to understand the ACTION. 'weather' could mean "
                   "anything — 'get_current_weather' is unambiguous.",
        },
        {
            "rule": "2. Describe WHEN to use it, not just WHAT it does",
            "bad": "Calculates math",
            "good": "Evaluate a mathematical expression. Use this when the user asks "
                    "for calculations, comparisons, or numerical analysis.",
            "why": "The LLM decides WHICH tool to call based on the description. "
                   "Including 'when to use' helps it choose correctly.",
        },
        {
            "rule": "3. Parameters should be self-documenting",
            "bad": "city: str",
            "good": "city: str — 'City name (e.g., London, New York). Use English names.'",
            "why": "Examples in parameter descriptions reduce ambiguity. The LLM "
                   "passes better arguments when it sees examples.",
        },
        {
            "rule": "4. Return strings, not complex objects",
            "bad": "return {'temp': 22, 'unit': 'C', 'wind': {'speed': 15, 'dir': 'NW'}}",
            "good": "return 'Temperature: 22°C, Wind: 15 km/h NW, Humidity: 65%'",
            "why": "The LLM reads the tool result as text. A formatted string is "
                   "easier for it to incorporate into its response than raw JSON.",
        },
        {
            "rule": "5. Handle errors gracefully — return error messages, don't crash",
            "bad": "raise ValueError('City not found')",
            "good": "return 'Error: City not found. Try a different spelling or use English name.'",
            "why": "If a tool crashes, the agent crashes. Returning an error message "
                   "lets the LLM try a different approach or inform the user.",
        },
    ]

    for rule_info in rules:
        print(f"\n  {rule_info['rule']}")
        print(f"    [BAD] Bad:  {rule_info['bad']}")
        print(f"    [OK] Good: {rule_info['good'][:80]}{'...' if len(rule_info['good']) > 80 else ''}")
        print(f"    Why:  {rule_info['why'][:100]}{'...' if len(rule_info['why']) > 100 else ''}")


# ==============================================================
# PART 3: Implementing Well-Designed Tools
# ==============================================================
# Let's build tools that follow ALL 5 rules.

def get_current_weather(city: str) -> str:
    """Get the current weather for a city.

    Use this when the user asks about weather conditions, temperature,
    or forecasts for a specific location.

    Args:
        city: City name in English (e.g., 'London', 'Tokyo', 'New York')

    Returns:
        Formatted weather information or error message
    """
    # Simulated weather data (real version would call an API)
    weather_data = {
        "london": "London: 15°C, Partly cloudy, Wind: 12 km/h W, Humidity: 72%",
        "tokyo": "Tokyo: 22°C, Clear sky, Wind: 8 km/h SE, Humidity: 55%",
        "new york": "New York: 18°C, Overcast, Wind: 20 km/h NW, Humidity: 68%",
    }

    # Rule 5: Return error message, don't crash
    result = weather_data.get(city.lower())
    if result is None:
        return f"Error: Weather data not available for '{city}'. Try: London, Tokyo, New York."
    return result


def search_knowledge_base(query: str, category: str = "all") -> str:
    """Search the internal knowledge base for information.

    Use this when the user asks about company policies, procedures,
    or internal documentation. For external/public information,
    use search_web instead.

    Args:
        query: What to search for (e.g., 'vacation policy', 'expense limits')
        category: Filter by category — 'hr', 'finance', 'engineering', or 'all'

    Returns:
        Relevant knowledge base entries or 'No results found'
    """
    # Simulated KB
    kb = {
        "vacation": "Vacation Policy: 20 days/year, max 10 consecutive days. "
                    "Submit requests 2 weeks in advance via HR portal.",
        "expenses": "Expense Policy: Meals up to $50/day, flights require manager "
                    "approval, receipts required for amounts over $25.",
    }

    for key, value in kb.items():
        if key in query.lower():
            return value

    return f"No results found for '{query}' in category '{category}'."


def demonstrate_tools():
    """Show the tools in action."""

    print(f"\n{'='*60}")
    print("PART 3: Well-Designed Tools in Action")
    print(f"{'='*60}")

    # Good call
    print("\n  get_current_weather('London'):")
    print(f"    -> {get_current_weather('London')}")

    # Error handling
    print("\n  get_current_weather('Atlantis'):")
    print(f"    -> {get_current_weather('Atlantis')}")

    # KB search
    print("\n  search_knowledge_base('vacation policy'):")
    print(f"    -> {search_knowledge_base('vacation policy')}")

    # No results
    print("\n  search_knowledge_base('quantum physics'):")
    print(f"    -> {search_knowledge_base('quantum physics')}")


# ==============================================================
# PART 4: Tool Composition — Building Complex Tools from Simple Ones
# ==============================================================

def demonstrate_composition():
    """Show how to compose tools for more complex operations."""

    print(f"\n{'='*60}")
    print("PART 4: Tool Composition")
    print(f"{'='*60}")

    # Simple tools
    def convert_celsius_to_fahrenheit(celsius: float) -> str:
        """Convert Celsius to Fahrenheit."""
        f = (celsius * 9/5) + 32
        return f"{celsius}°C = {f}°F"

    def compare_numbers(a: float, b: float) -> str:
        """Compare two numbers and return the relationship."""
        if a > b:
            return f"{a} is greater than {b} (difference: {a - b})"
        elif b > a:
            return f"{b} is greater than {a} (difference: {b - a})"
        return f"{a} and {b} are equal"

    # Composed tool — combines weather lookup + temperature conversion
    def get_weather_in_fahrenheit(city: str) -> str:
        """Get weather with temperature in both Celsius and Fahrenheit.

        Use this when the user asks for weather in Fahrenheit or when
        comparing temperatures across cities for US-based users.
        """
        weather = get_current_weather(city)
        if weather.startswith("Error"):
            return weather

        # Extract temperature (simple parsing for demo)
        try:
            temp_str = weather.split(":")[1].strip().split("°")[0].strip()
            celsius = float(temp_str)
            conversion = convert_celsius_to_fahrenheit(celsius)
            return f"{weather} | {conversion}"
        except (IndexError, ValueError):
            return weather

    print("\n  Simple tool: convert_celsius_to_fahrenheit(22):")
    print(f"    -> {convert_celsius_to_fahrenheit(22)}")

    print("\n  Composed tool: get_weather_in_fahrenheit('Tokyo'):")
    print(f"    -> {get_weather_in_fahrenheit('Tokyo')}")

    print("\n  [TIP] Key Insight:")
    print("     Keep individual tools SIMPLE and FOCUSED.")
    print("     Build complex behavior by giving the agent MULTIPLE tools")
    print("     and letting it chain them — or compose them in code when")
    print("     a common combination is always needed together.")


# ==============================================================
# PART 5: How the LLM Decides Which Tool to Call
# ==============================================================

def demonstrate_tool_selection():
    """Explain how the LLM selects tools based on schemas."""

    print(f"\n{'='*60}")
    print("PART 5: How the LLM Selects Tools")
    print(f"{'='*60}")

    print("""
  When you bind tools to an LLM, here's what happens:

  1. SCHEMA INJECTION: Your tool definitions (name, description,
     parameters) are added to the system prompt as a JSON schema.

  2. USER QUERY: The user asks "What's the weather in Tokyo?"

  3. LLM REASONING: The LLM reads the query + available tool schemas
     and decides:
       "The user wants weather -> I have get_current_weather -> I should
        call it with city='Tokyo'"

  4. TOOL CALL: The LLM outputs a structured tool call:
     {"tool": "get_current_weather", "args": {"city": "Tokyo"}}

  5. EXECUTION: Your code runs the actual function and returns the result.

  6. SYNTHESIS: The LLM reads the tool result and generates a natural
     language response for the user.

  This is why tool descriptions matter so much — the LLM literally
  reads them to make decisions. A bad description = bad decisions.

  [TIP] Key Insight:
     The LLM does NOT see your function's source code.
     It only sees: name + description + parameter schema.
     Everything the LLM needs to know must be in those three things.
""")


# ==============================================================
# Run all parts
# ==============================================================

if __name__ == "__main__":
    print("Example 4: Tool Use Pattern — What Makes a Good Tool")
    demonstrate_good_vs_bad_tools()
    demonstrate_tool_design_rules()
    demonstrate_tools()
    demonstrate_composition()
    demonstrate_tool_selection()

    print(f"\n{'='*60}")
    print("Next: See example_05 for tool-using agents in LangGraph")
    print(f"{'='*60}")
