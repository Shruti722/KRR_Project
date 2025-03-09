import openai
import os
import requests
import json
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

def extract_query_parameters(user_query):
    """Extracts structured search parameters from user input using regex and GPT validation."""
    location_match = re.search(r"in ([A-Za-z\s]+)", user_query, re.IGNORECASE)
    budget_match = re.search(r"under \$?(\d+)", user_query, re.IGNORECASE)
    amenities_match = re.findall(r"(infinity pool|spa|wifi|beachfront|gym|restaurant)", user_query, re.IGNORECASE)

    location = location_match.group(1).strip() if location_match else "Unknown"
    budget = int(budget_match.group(1)) if budget_match else None
    amenities = [amenity.lower() for amenity in amenities_match] if amenities_match else []

    return {"location": location, "budget": budget, "amenities": amenities}


def find_hotels_with_google(location, amenities):
    """Fetch hotels from Google Places API based on GPT-4o extracted location and amenities."""
    search_query = f"hotels in {location}"
    if amenities:
        search_query += f" with {', '.join(amenities)}"

    params = {
        "query": search_query,
        "key": GOOGLE_API_KEY
    }

    response = requests.get(GOOGLE_PLACES_API_URL, params=params)
    data = response.json()

    if "results" not in data or not data["results"]:
        return f"No hotels found matching your criteria in {location}.", []

    hotel_list = []
    for hotel in data["results"][:5]:  # Limit to 5 results
        name = hotel.get("name", "Unknown Hotel")
        place_id = hotel.get("place_id", "N/A")  
        address = hotel.get("formatted_address", "No address available")
        rating = hotel.get("rating", "No rating available")
        hotel_list.append({"name": name, "place_id": place_id, "address": address, "rating": rating})

    return f"Here are some hotels in {location}:", hotel_list


def get_hotel_prices(country, hotel_name, checkin, checkout, currency="USD", adults=2, rooms=1, kids=0):
    """Fetch live hotel prices using Makcorps API."""
    params = {
        "country": country,
        "hotelid": hotel_name,
        "checkin": checkin,
        "checkout": checkout,
        "currency": currency,
        "adults": adults,
        "rooms": rooms,
        "kids": kids,
        "api_key": MAKCORPS_API_KEY
    }

    response = requests.get(MAKCORPS_API_URL, params=params)
    
    if response.status_code != 200:
        return f"Error: API request failed with status code {response.status_code}"

    return response.json()

def chat_with_gpt(prompt, system_message):
    """Sends a prompt to GPT-4o and returns the response."""
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Interactive Mode for Selecting and Using Agents
if __name__ == "__main__":
    print("Multi-Agent GPT-4o System (Type 'exit' to quit)")

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
                user_input = input("\nYou: ")
                if user_input.lower() == "exit":
                    break

                if agent_choice == "3":  # If the Travel Agent is selected
                    city = user_input.strip().title()
                    print("\nSearching for hotels in", city, "...")
                    message, hotels = find_hotels_with_google(city, [])

                    print("\nGPT-4o (Hotels Found):")
                    print(message)
                    for idx, hotel in enumerate(hotels, start=1):
                        print(f"{idx}. {hotel['name']} - {hotel['rating']} stars - {hotel['address']}")

                    if not hotels:
                        continue

                    hotel_choice = int(input("\nEnter the number of the hotel to fetch prices: ")) - 1
                    if hotel_choice < 0 or hotel_choice >= len(hotels):
                        print("Invalid selection.")
                        continue

                    hotel_selected = hotels[hotel_choice]["name"]
                    country = input("Enter country (e.g., us, uk): ").strip().lower()
                    checkin = input("Enter check-in date (YYYY-MM-DD): ").strip()
                    checkout = input("Enter check-out date (YYYY-MM-DD): ").strip()

                    print(f"\nFetching live hotel prices for {hotel_selected}...")
                    hotels_prices = get_hotel_prices(country, hotel_selected, checkin, checkout)
                    print("\nGPT-4o (Hotel Prices):", hotels_prices)
                else:
                    response = chat_with_gpt(user_input, system_prompt)
                    print("\nGPT-4o:", response)

            except KeyboardInterrupt:
                print("\nExiting... Goodbye!")
                break