"""Pull active sale listings for a target market and cache them.

The API allowance is small, so this hits RentCast once per run and
saves the raw response to a dated file. All downstream work reads
from disk, which means the scoring model can be iterated on
indefinitely without spending calls.
"""

import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RENTCAST_API_KEY")
BASE_URL = "https://api.rentcast.io/v1"
RAW_DIR = "data/raw"

DEFAULT_CITY = "Lubbock"
DEFAULT_STATE = "TX"
LIMIT = 500


def fetch_listings(city, state):
    headers = {"X-Api-Key": API_KEY}
    params = {
        "city": city,
        "state": state,
        "status": "Active",
        "limit": LIMIT,
    }

    response = requests.get(
        f"{BASE_URL}/listings/sale",
        headers=headers,
        params=params,
    )
    response.raise_for_status()
    return response.json()


def save_raw(listings, city):
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    slug = city.lower().replace(" ", "-")
    path = f"{RAW_DIR}/{slug}_{stamp}.json"

    with open(path, "w") as f:
        json.dump(listings, f, indent=2)

    return path


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CITY
    state = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_STATE

    print(f"Pulling active listings for {city}, {state}")
    listings = fetch_listings(city, state)
    path = save_raw(listings, city)

    print(f"Pulled {len(listings)} listings")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
