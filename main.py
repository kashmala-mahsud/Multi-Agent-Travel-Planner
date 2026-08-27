import asyncio
import os
from typing import TypedDict, Annotated
import operator
import uuid

import psycopg

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    get_airlines,
    get_airports,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search
)

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# STATE
# --------------------------------------------------

class TravelState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        operator.add
    ]

    user_query: str

    flight_results: str

    hotel_results: str

    itinerary: str

    llm_calls: int

    weather_results: str


# --------------------------------------------------
# FLIGHT PROMPT
# --------------------------------------------------

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.

IMPORTANT:
Do not claim that a specific flight is available unless
the flight data explicitly confirms it.
"""


# --------------------------------------------------
# FLIGHT AGENT
# --------------------------------------------------

def flight_agent(state: TravelState):

    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel flight planner."
                ),
                HumanMessage(
                    content=prompt
                )
            ]
        )

        flight_data = response.content

    except Exception as e:

        flight_data = (
            f"Flight information unavailable: {str(e)}"
        )

    return {
        "flight_results": flight_data,

        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# --------------------------------------------------
# HOTEL AGENT
# --------------------------------------------------

def hotel_agent(state: TravelState):

    query = f"""
Find actual hotels in the destination from this travel request:

{state['user_query']}

Search specifically for:

- Hotel names
- City
- Price per night
- Rating
- Location
- Hotel description
- Booking information

Do not return general travel guides or trip-cost articles.
"""

    hotel_results = asyncio.run(
        tavily_mcp_search(query)
    )

    return {
        "hotel_results": hotel_results,

        "messages": [
            AIMessage(
                content="Hotel information fetched"
            )
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# --------------------------------------------------
# WEATHER AGENT
# --------------------------------------------------

def weather_agent(state: TravelState):

    city = extract_destination(
        state["user_query"]
    )

    print(f"\nWeather destination: {city}\n")

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
Current Weather:

{weather_data}

Forecast:

{forecast_data}
""",

        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }


# --------------------------------------------------
# ITINERARY AGENT
# --------------------------------------------------

def itinerary_agent(state: TravelState):

    prompt = f"""
You are a travel itinerary planner.

Create a travel itinerary using the information
provided by the Flight Agent, Hotel Agent and Weather Agent.

USER REQUEST:
{state['user_query']}

FLIGHT AGENT RESULTS:
{state['flight_results']}

HOTEL AGENT RESULTS:
{state['hotel_results']}

WEATHER INFORMATION:
{state['weather_results']}


STRICT RULES:

1. Do NOT invent flights.
2. Do NOT invent hotels.
3. Do NOT invent prices.
4. Do NOT invent ratings.
5. Do NOT invent availability.
6. Do NOT replace missing tool results with your own knowledge.
7. Clearly distinguish search results from estimates.
8. If suitable flight data is missing, say:
   "No suitable flight data was found."
9. If suitable hotel data is missing, say:
   "No suitable hotel data was found."

Create a useful itinerary based on the
available information.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert travel planner "
                    "who never fabricates booking information."
                )
            ),

            HumanMessage(
                content=prompt
            )
        ]
    )

    return {
        "itinerary": response.content,

        "messages": [
            response
        ],

        "llm_calls": state.get(
            "llm_calls",
            0
        ) + 1
    }


# --------------------------------------------------
# GRAPH
# --------------------------------------------------

graph = StateGraph(TravelState)

graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "weather_agent",
    weather_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)


graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "weather_agent"
)

graph.add_edge(
    "weather_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    END
)


# --------------------------------------------------
# POSTGRES MEMORY
# --------------------------------------------------

_conn = psycopg.connect(
    DATABASE_URL
)

checkpointer = PostgresSaver(
    _conn
)

checkpointer.setup()

app = graph.compile(
    checkpointer=checkpointer
)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    user_input = input(
        "Enter travel request: "
    )

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],

            "user_query": user_input,

            "flight_results": "",

            "hotel_results": "",

            "itinerary": "",

            "weather_results": "",

            "llm_calls": 0
        },

        config=config
    )

    print("\nFINAL RESPONSE:\n")

    # Print the actual itinerary
    print(
        result["itinerary"]
    )