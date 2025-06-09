import os
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
import requests
import json

# Set up environment variables (replace with your actual API keys)
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

# Initialize Open AI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# SerpAPI call for flights
@tool
def search_flights(origin: str, destination: str, date: str, budget: float) -> str:
    """Search for flights using SerpApi's Google Flights API and return multiple options within budget."""
    api_key = os.environ["SERPAPI_API_KEY"]
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "currency": "USD",
        "hl": "en",
        "api_key": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for errors in response
        if data.get("search_metadata", {}).get("status") != "Success":
            return f"Error fetching flights: {data.get('error', 'Unknown error')}"

        # Extract flights
        flights = data.get("best_flights", []) + data.get("other_flights", [])
        if not flights:
            return "No flights found for the given parameters."

        # Filter flights by budget
        filtered_flights = [
            f for f in flights
            if f.get("price", float("inf")) <= budget
        ]
        if not filtered_flights:
            return f"No flights found within budget of ${budget}."

        # Sort flights by price and layovers
        sorted_flights = sorted(
            filtered_flights,
            key=lambda x: (x.get("price", float("inf")), len(x.get("layovers", [])))
        )

        # Prepare flight options
        flight_options = []
        for flight in sorted_flights[:4]:  # Limit to top 4 options
            airline = flight["flights"][0]["airline"]
            price = flight.get("price", "N/A")
            layovers = len(flight.get("layovers", []))
            duration = flight.get("total_duration", "N/A")
            flight_options.append(f"{airline} for ${price}, {layovers} layover(s), {duration} minutes")

        return "Flight Options:\n" + "\n".join(flight_options)
    except Exception as e:
        return f"Error fetching flights: {str(e)}"


# SerpAPI call for hotels
@tool
def search_hotels(destination: str, checkin: str, budget: float, min_rating: float) -> str:
    """Search for hotels using SerpApi's Google Hotels API and return multiple options within budget and rating."""
    api_key = os.environ["SERPAPI_API_KEY"]
    url = "https://serpapi.com/search.json"
    checkin_date = datetime.strptime(checkin, "%Y-%m-%d")
    checkout_date = checkin_date + timedelta(days=1)
    params = {
        "engine": "google_hotels",
        "q": f"hotels in {destination}",
        "check_in_date": checkin_date.strftime("%Y-%m-%d"),
        "check_out_date": checkout_date.strftime("%Y-%m-%d"),
        "currency": "USD",
        "hl": "en",
        "adults": 2,
        "api_key": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for errors in response
        if data.get("search_metadata", {}).get("status") != "Success":
            return f"Error fetching hotels: {data.get('error', 'Unknown error')}"

        # Extract properties
        properties = data.get("properties", [])
        if not properties:
            return "No hotels found for the given parameters."

        # Filter hotels by budget and rating
        filtered_hotels = [
            p for p in properties
            if p.get("rate_per_night", {}).get("extracted_lowest", float("inf")) <= budget
            and p.get("overall_rating", 0) >= min_rating
            and destination.lower() in p.get("address", "").lower()  # Enforce city match
        ]

        if not filtered_hotels:
            return f"No hotels found within budget of ${budget} and minimum rating of {min_rating}."

        # Sort hotels by rating and price
        sorted_hotels = sorted(
            filtered_hotels,
            key=lambda x: (-x.get("overall_rating", 0), x.get("rate_per_night", {}).get("extracted_lowest", float("inf")))
        )

        # Prepare hotel options
        hotel_options = []
        for hotel in sorted_hotels[:4]:  # Limit to top 4 options
            name = hotel.get("name", "N/A")
            price = hotel.get("rate_per_night", {}).get("extracted_lowest", "N/A")
            rating = hotel.get("overall_rating", "N/A")
            address = hotel.get("address", "N/A")
            hotel_options.append(f"{name} - ${price}/night, {rating} stars, located at {address}")

        return "Hotel Options:\n" + "\n".join(hotel_options)
    except Exception as e:
        return f"Error fetching hotels: {str(e)}"



# Itinerary generation (unchanged, as it doesn't use SerpAPI)
@tool
def generate_itinerary(destination: str, days: int) -> str:
    """Generate a dynamic travel itinerary for the destination using SerpAPI Places."""
    api_key = os.environ["SERPAPI_API_KEY"]
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_maps",
        "type": "search",
        "q": f"top tourist attractions in {destination}",
        "hl": "en",
        "api_key": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        places = data.get("local_results", [])[:days]
        itinerary = f"Itinerary for {destination} ({days} days):\n"

        for day in range(1, days + 1):
            place = places[day - 1] if day - 1 < len(places) else {"title": "Explore the city", "address": ""}
            itinerary += f"Day {day}:\n"
            itinerary += f"  - Visit: {place.get('title', 'Explore the city')}, located at {place.get('address', 'N/A')}\n"
            itinerary += f"  - Transport: Recommended local transport (e.g., Metro, Taxi)\n"
            itinerary += f"  - Dine at: Try local cuisines at popular spots\n"
        return itinerary
    except Exception as e:
        return f"Error generating itinerary: {str(e)}"


# Define state schema
class TravelState(TypedDict):
    origin: str
    destination: str
    date: str
    budget: float
    days: int
    min_rating: float
    flight_result: str
    hotel_result: str
    itinerary_result: str
    final_plan: str

# Define nodes
def flight_node(state: TravelState) -> TravelState:
    """Node to search for flights."""
    result = search_flights.invoke({
        "origin": state["origin"],
        "destination": state["destination"],
        "date": state["date"],
        "budget": state["budget"]
    })
    return {"flight_result": result}

def hotel_node(state: TravelState) -> TravelState:
    """Node to search for hotels."""
    result = search_hotels.invoke({
        "destination": state["destination"],
        "checkin": state["date"],
        "budget": state["budget"],
        "min_rating": state["min_rating"]
    })
    return {"hotel_result": result}

def itinerary_node(state: TravelState) -> TravelState:
    """Node to generate itinerary."""
    result = generate_itinerary.invoke({
        "destination": state["destination"],
        "days": state["days"]
    })
    return {"itinerary_result": result}

def coordinator_node(state: TravelState) -> TravelState:
    """Node to compile final plan."""
    prompt = (
        "You are a travel coordinator. Compile a trip plan based on the following details:\n"
        f"Flight: {state['flight_result']}\n\n"
        f"Hotel: {state['hotel_result']}\n\n"
        f"Itinerary: {state['itinerary_result']}\n\n"
        "Format it day-by-day and highlight any issues or mismatched info (e.g., incorrect locations).\n"
        "Make sure the hotel recommendation is actually in the destination city.\n"
    )

    messages = [
        SystemMessage(content="You are a travel coordinator providing concise and clear trip plans."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return {"final_plan": response.content}


# Build LangGraph workflow
workflow = StateGraph(TravelState)
workflow.add_node("flight", flight_node)
workflow.add_node("hotel", hotel_node)
workflow.add_node("itinerary", itinerary_node)
workflow.add_node("coordinator", coordinator_node)
workflow.add_edge("flight", "hotel")
workflow.add_edge("hotel", "itinerary")
workflow.add_edge("itinerary", "coordinator")
workflow.add_edge("coordinator", END)
workflow.set_entry_point("flight")
graph = workflow.compile()

# Streamlit UI
def main():
    st.title("AI Travel Planner")
    st.subheader("Plan your perfect trip with AI-powered recommendations")

    # User inputs
    origin = st.text_input("Departure City", "London")
    destination = st.text_input("Destination City", "Paris")
    date = st.date_input("Travel Date", datetime.now() + timedelta(days=7))
    budget = st.number_input("Budget ($)", min_value=100, value=500, step=50)
    days = st.number_input("Trip Duration (days)", min_value=1, value=3, step=1)
    min_rating = st.slider("Minimum Hotel Rating", 3.0, 5.0, 4.0, 0.1)

    if st.button("Plan My Trip"):
        # Convert date to string
        date_str = date.strftime("%Y-%m-%d")

        # Initialize state
        initial_state = {
            "origin": origin,
            "destination": destination,
            "date": date_str,
            "budget": budget,
            "days": days,
            "min_rating": min_rating,
            "flight_result": "",
            "hotel_result": "",
            "itinerary_result": "",
            "final_plan": ""
        }

        # Run the graph
        try:
            result = graph.invoke(initial_state)
            st.subheader("Your Trip Plan")
            st.write(result["final_plan"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()