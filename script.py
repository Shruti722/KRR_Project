import openai
import os
import requests
import json
import http.client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration (from .env file)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")

MAKCORPS_API_KEY = os.getenv("MAKCORPS_API_KEY")
MAKCORPS_API_URL = "https://api.makcorps.com/booking"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PLACES_API_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

# RapidAPI Connection Test

def test_rapidapi_connection():
    """Test RapidAPI connection with basic endpoint."""
    conn = http.client.HTTPSConnection("booking-com.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    conn.request("GET", "/v1/metadata/exchange-rates?locale=en-us&currency=USD", headers=headers)
    res = conn.getresponse()
    data = res.read()
    print("\n🔍 RapidAPI Connection Test Response:")
    print(data.decode("utf-8"))


# Set up OpenAI client
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Define three different agents with unique system prompts
AGENTS = {
    "1": {
        "description": "Case 1: Giving more than the required number of outputs.",
        "system_message": """You are an AI assistant that follows user instructions precisely.

        - When a user requests a single result, return exactly one and no more.
        - Do not assume the user wants additional recommendations unless explicitly stated.
        - If multiple possible answers exist, choose the most relevant one based on context, instead of listing multiple options.
        - If clarification is needed, ask instead of assuming.
        - Prioritize precision over over-explaining or expanding the response unnecessarily.

        Examples:

        User: "Give me just one affordable hotel in New York."
        AI: "Sure! The most affordable hotel in New York is Hotel A."

        User: "Find a nearby restaurant."
        AI: "I recommend Joe's Diner, a well-rated spot close to your location."

        User: "Give me a hotel in San Francisco under $100."
        AI: "Based on affordability and reviews, I suggest The Budget Inn."

        Do not provide multiple answers unless explicitly asked for.
        Do not add unnecessary options or alternatives.

        """
    },
    "2": {
        "description": "Case 2: Assuming location of the prompt.",
        "system_message": """You are an AI assistant that provides accurate and context-aware responses to user queries. 

        - Never assume missing information such as locations, dates, budgets, or preferences. 
        - If a user request lacks key details (e.g., "give me affordable hotels"), explicitly ask for clarification before proceeding. 
        - If the user has previously provided relevant details, use that information to avoid unnecessary questions.
        - Ensure that your responses are relevant, concise, and aligned with the user's intent.
        - If multiple interpretations exist for a query, offer clarification questions instead of making assumptions.
        - If the user asks for a default recommendation (e.g., "give me some options"), use general best practices or popular choices, but explicitly state that the user can refine their request.

        Examples:

        User: "Give me affordable hotels."
        AI: "Could you specify the location you're looking for? Also, do you have a price range in mind?"

        User: "Find me a hotel under $100."
        AI: "What city or area should I search in?"

        User: "Give me restaurants."
        AI: "Would you like recommendations for a specific city or cuisine type?"

        User: "How far is it?" 
        AI: "Could you clarify which two locations you'd like to compare?"
        """
    },

    "3": {
        "description": "Case 3: Not giving updated outputs.",
        "system_message": "You are a travel assistant. Extract user preferences and fetch real-time hotel prices based on location, budget, and amenities."
    }
}

import re

# def extract_query_parameters(user_query):
#     """Extract structured search parameters using NLP first, then validate with GPT-4o."""
    
#     # Step 1: Extract potential location using regex and keywords
#     location_match = re.search(r"in ([A-Za-z\s]+)", user_query, re.IGNORECASE)
#     budget_match = re.search(r"under \$?(\d+)", user_query, re.IGNORECASE)
#     amenities_match = re.findall(r"(infinity pool|spa|wifi|beachfront|gym|restaurant)", user_query, re.IGNORECASE)

#     location = location_match.group(1).strip() if location_match else None
#     budget = int(budget_match.group(1)) if budget_match else None
#     amenities = [amenity.lower() for amenity in amenities_match] if amenities_match else []

#     # Step 2: Validate extracted data with GPT-4o
#     system_prompt = """
#     You are an AI assistant that validates and refines extracted hotel search parameters.
#     Ensure:
#     - 'location' is a valid city or region (e.g., "Chicago, USA", "New York, USA").
#     - 'budget' is correct (if present).
#     - 'amenities' are valid hotel features.

#     Always return JSON format like this:
#     {"location": "New York, USA", "budget": 150, "amenities": ["infinity pool"]}
#     """

#     functions = [
#         {
#             "name": "validate_hotel_search_parameters",
#             "description": "Validates and refines extracted search parameters.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {"type": "string", "description": "Validated city or region name."},
#                     "budget": {"type": "integer", "description": "Maximum price per night in USD."},
#                     "amenities": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "List of hotel amenities requested by the user."
#                     }
#                 },
#                 "required": ["location"]
#             }
#         }
#     ]

#     gpt_response = client.chat.completions.create(
#         model=AZURE_DEPLOYMENT_NAME,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": json.dumps({"location": location, "budget": budget, "amenities": amenities})}
#         ],
#         functions=functions,
#         function_call={"name": "validate_hotel_search_parameters"}
#     )

#     try:
#         structured_data = json.loads(gpt_response.choices[0].message.function_call.arguments)
#         return structured_data
#     except Exception as e:
#         print("Error parsing GPT-4o function response:", e)
#         return {"location": location, "budget": budget, "amenities": amenities}  # Fallback to extracted data


#     response = client.chat.completions.create(
#         model=AZURE_DEPLOYMENT_NAME,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_query}
#         ],
#         functions=functions,
#         function_call={"name": "extract_hotel_search_parameters"}
#     )

#     try:
#         structured_data = response.choices[0].message.function_call.arguments
#         return json.loads(structured_data)  # Convert JSON response to a Python dictionary
#     except Exception as e:
#         print("Error parsing GPT-4o function response:", e)
#         return None

# Function to let LLM parse user query naturally
def extract_query_details_from_llm(user_query):
    system_prompt = """
    You are an assistant that extracts hotel search details from user requests. 
    Extract location, check-in date, check-out date, and budget if mentioned. 
    Dates must be formatted as YYYY-MM-DD. 
    Respond only with JSON in this format: 
    {"location": "city name", "checkin": "YYYY-MM-DD", "checkout": "YYYY-MM-DD", "budget": 150} 
    If budget is not mentioned, set it as null.
    """
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.2,
        max_tokens=300
    )
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except Exception as e:
        print("Failed to parse LLM response:", content)
        print("Parsing error:", e)
        return None

# Function to get destination ID for RapidAPI hotel search
def get_destination_id(city_name):
    url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    params = {"name": city_name, "locale": "en-us"}
    response = requests.get(url, headers=headers, params=params)
    print(f"\n🔍 Destination ID API Response: {response.text}")
    if response.status_code == 200:
        data = response.json()
        if data:
            dest_id = data[0].get("dest_id")
            print(f"✅ Fetched Destination ID for {city_name}: {dest_id}")
            return dest_id
    return None

def get_hotel_prices_via_rapidapi(hotel_details, adults=2, rooms=1):
    location = hotel_details["location"]
    checkin = hotel_details["checkin"]
    checkout = hotel_details["checkout"]
    budget = hotel_details.get("budget")

    dest_id = get_destination_id(location)
    if not dest_id:
        return "Could not find hotels in that location."

    url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    params = {
        "checkin_date": checkin,
        "checkout_date": checkout,
        "dest_id": dest_id,
        "dest_type": "city",
        "adults_number": adults,
        "room_number": rooms,
        "order_by": "price",
        "currency": "USD",  # Hardcoded
        "locale": "en-us",
        "filter_by_currency": "USD",  # Hardcoded
        "units": "metric"  # Hardcoded as metric
    }
    response = requests.get(url, headers=headers, params=params)
    print(f"\n🔍 Hotel Search API Response: {response.text}")
    if response.status_code != 200:
        return f"Error: API request failed. {response.status_code}: {response.text}"

    data = response.json()
    filtered_hotels = []
    for hotel in data.get("result", []):
        price = hotel.get("min_total_price", float("inf"))
        if budget is None or price <= budget:
            filtered_hotels.append({
                "name": hotel.get("hotel_name"),
                "price": price,
                "address": hotel.get("address", "No address"),
                "rating": hotel.get("review_score", "No rating"),
                "url": hotel.get("url", "No URL")
            })

    return filtered_hotels if filtered_hotels else "No hotels found matching your criteria."

# Main interactive loop
if __name__ == "__main__":
    print("Multi-Agent GPT-4o System (Type 'exit' to quit)")

    # Test connection to RapidAPI at startup
    test_rapidapi_connection()

    while True:
        print("\nSelect an agent:")
        for key, info in AGENTS.items():
            print(f"{key}: {info['description']}")

        agent_choice = input("\nEnter agent number: ").strip()
        if agent_choice.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        if agent_choice not in AGENTS:
            print("Invalid choice. Please select a valid agent.")
            continue

        # Get the selected system message
        agent_info = AGENTS[agent_choice]
        system_prompt = agent_info["system_message"]
        print(f"\nSelected Agent: {agent_info['description']}")

        while True:
            try:
                user_query = input("\nYou: ")
                if user_query.lower() == "exit":
                    break

                if agent_choice == "3":
                    hotel_details = extract_query_details_from_llm(user_query)
                    if not hotel_details:
                        print("❌ Could not extract details. Please try again.")
                        continue
                    print(f"\n📍 Searching hotels in {hotel_details['location']} from {hotel_details['checkin']} to {hotel_details['checkout']}")
                    if hotel_details['budget']:
                        print(f"💰 Budget: ${hotel_details['budget']} per night")
                    hotels = get_hotel_prices_via_rapidapi(hotel_details)
                    print("\n🏨 Hotel Options:")
                    if isinstance(hotels, str):
                        print(hotels)
                    else:
                        for idx, hotel in enumerate(hotels, start=1):
                            print(f"{idx}. {hotel['name']} - ${hotel['price']} - {hotel['rating']} stars")
                            print(f"   Address: {hotel['address']}\n   URL: {hotel['url']}\n")
                else:
                    response = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    print("\nGPT-4o:", response.choices[0].message.content)
            except KeyboardInterrupt:
                print("\nExiting... Goodbye!")
                break
