import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RENTCAST_API_KEY")
BASE_URL = "https://api.rentcast.io/v1"

headers = {"X-Api-Key": API_KEY}

params = {
    "city": "Lubbock",
    "state": "TX",
    "status": "Active",
    "limit": 5
}

response = requests.get(
    f"{BASE_URL}/listings/sale",
    headers=headers,
    params=params
)

print(response.status_code)
print(response.json())
