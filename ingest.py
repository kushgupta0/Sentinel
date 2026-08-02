import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RENTCAST_API_KEY")
BASE_URL = "https://api.rentcast.io/v1"
RAW_DIR = "data/raw"

CITY = "Lubbock"
STATE = "TX"
LIMIT = 500


def fetch_listings():
    headers = {"X-Api-Key": API_KEY}
    params = {
        "city": CITY,
        "state": STATE,
        "status": "Active",
        "limit": LIMIT
    }

    response = requests.get(
        f"{BASE_URL}/listings/sale",
        headers=headers,
        params=params
    )
    response.raise_for_status()
    return response.json()


def save_raw(listings):
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = f"{RAW_DIR}/listings_{stamp}.json"

    with open(path, "w") as f:
        json.dump(listings, f, indent=2)

    return path


if __name__ == "__main__":
    listings = fetch_listings()
    path = save_raw(listings)
    print(f"Pulled {len(listings)} listings")
    print(f"Saved to {path}")
