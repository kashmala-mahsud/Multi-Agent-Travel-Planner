import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# --------------------------------------------------
# MCP CLIENT
# --------------------------------------------------

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },

        "aviationstack": {
            "transport": "stdio",
            "command": r"D:\Multi-Agent-Travel-Planner\aviationstack-mcp\.venv\Scripts\python.exe",
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "cwd": r"D:\Multi-Agent-Travel-Planner\aviationstack-mcp",
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY,
            },
        },

        "weather": {
            "transport": "stdio",
            "command": r"D:\Multi-Agent-Travel-Planner\.venv\Scripts\python.exe",
            "args": [
                r"D:\Multi-Agent-Travel-Planner\custom_weather_mcp_server.py",
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }
)


# --------------------------------------------------
# GLOBAL TOOL VARIABLES
# --------------------------------------------------

search_tool = None
aviation_tools = {}

weather_tool = None
forecast_tool = None


# --------------------------------------------------
# INITIALIZE ALL MCP TOOLS
# --------------------------------------------------

async def initialize_mcp():

    global search_tool
    global aviation_tools
    global weather_tool
    global forecast_tool

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print("-", tool.name)

    # Tavily
    search_tool = next(
        (tool for tool in tools if tool.name == "tavily_search"),
        None
    )

    # AviationStack
    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name not in [
            "tavily_search",
            "get_current_weather",
            "get_forecast",
        ]
    }

    # Weather
    weather_tool = next(
        (tool for tool in tools if tool.name == "get_current_weather"),
        None
    )

    forecast_tool = next(
        (tool for tool in tools if tool.name == "get_forecast"),
        None
    )


# --------------------------------------------------
# TAVILY SEARCH
# --------------------------------------------------

async def tavily_mcp_search(query: str):

    await initialize_mcp()

    if search_tool is None:
        return "Tavily search tool unavailable"

    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )

    return result


# --------------------------------------------------
# AVIATION GENERIC TOOL CALL
# --------------------------------------------------

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict | None = None
):

    await initialize_mcp()

    tool = aviation_tools.get(tool_name)

    if tool is None:
        return f"Aviation tool '{tool_name}' unavailable"

    result = await tool.ainvoke(
        tool_args or {}
    )

    return result


# --------------------------------------------------
# AIRPORTS
# --------------------------------------------------

async def get_airports():

    await initialize_mcp()

    tool = aviation_tools.get("list_airports")

    if not tool:
        return "Airport tool unavailable"

    result = await tool.ainvoke({})

    return result


# --------------------------------------------------
# AIRLINES
# --------------------------------------------------

async def get_airlines():

    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if not tool:
        return "Airline tool unavailable"

    result = await tool.ainvoke({})

    return result


# --------------------------------------------------
# WEATHER
# --------------------------------------------------

async def weather_mcp_search(city: str):

    await initialize_mcp()

    if weather_tool is None:
        return "Weather tool unavailable"

    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


# --------------------------------------------------
# FORECAST
# --------------------------------------------------

async def forecast_mcp_search(city: str):

    await initialize_mcp()

    if forecast_tool is None:
        return "Forecast tool unavailable"

    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# DESTINATION EXTRACTION
# --------------------------------------------------

def extract_destination(query: str):

    prompt = f"""
Extract only the destination city or country from the following travel request.

Query:
{query}

Return only the destination name.
Do not provide any explanation.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

async def main():

    result = await tavily_mcp_search(
        "Best hotels in Japan"
    )

    print("\nTavily Search Result:")
    print(result)

    print("\nAirports:")
    airports = await get_airports()
    print(airports)

    print("\nAirlines:")
    airlines = await get_airlines()
    print(airlines)

    print("\nWeather:")
    weather = await weather_mcp_search("Tokyo")
    print(weather)

    print("\nForecast:")
    forecast = await forecast_mcp_search("Tokyo")
    print(forecast)

    print("\nDestination extraction:")

    destination = extract_destination(
        "I want to travel to Germany for 5 days from Pakistan"
    )

    print(destination)


if __name__ == "__main__":
    asyncio.run(main())