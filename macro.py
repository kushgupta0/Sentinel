"""Capture macro context alongside each snapshot.

National series, constant across listings, so not model features.
Stored as metadata so snapshot comparisons stay interpretable.
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
MACRO_DIR = "data/macro"

SERIES = {
    "MORTGAGE30US": "30-year fixed mortgage rate (weekly, national)",
    "MSACSR": "Months supply of new houses (monthly, national)",
    "HOUST": "Housing starts (monthly, national)",
    "CSUSHPINSA": "Case-Shiller national home price index (monthly)",
    "UNRATE": "Unemployment rate (monthly, national)",
}

# Industry convention, not fitted from this project's data.
SUPPLY_BUYERS_MARKET = 7.0
SUPPLY_SELLERS_MARKET = 4.0


def fetch_series(series_id, limit=12):
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    observations = []
    for obs in data.get("observations", []):
        if obs["value"] == ".":  # FRED uses "." for missing
            continue
        observations.append({
            "date": obs["date"],
            "value": float(obs["value"]),
        })

    return observations


def summarize(series_id, observations):
    if not observations:
        return {"series_id": series_id, "error": "no observations"}

    latest = observations[0]
    summary = {
        "series_id": series_id,
        "description": SERIES.get(series_id, ""),
        "latest_date": latest["date"],
        "latest_value": latest["value"],
        "observations": observations,
    }

    if len(observations) >= 2:
        summary["prior_value"] = observations[1]["value"]
        summary["change"] = round(latest["value"] - observations[1]["value"], 3)

    if len(observations) >= 12:
        summary["year_ago_value"] = observations[11]["value"]
        summary["year_change"] = round(
            latest["value"] - observations[11]["value"], 3
        )

    return summary


def interpret_supply(value):
    if value >= SUPPLY_BUYERS_MARKET:
        return "buyers market (supply >= 7 months)"
    if value <= SUPPLY_SELLERS_MARKET:
        return "sellers market (supply <= 4 months)"
    return "balanced (supply between 4 and 7 months)"


def main():
    if not API_KEY or API_KEY == "paste_your_key_here":
        print("FRED_API_KEY not set in .env")
        return

    results = {}
    for series_id in SERIES:
        try:
            obs = fetch_series(series_id)
            results[series_id] = summarize(series_id, obs)
            print(f"Pulled {series_id}: {len(obs)} observations")
        except Exception as e:
            print(f"Failed {series_id}: {e}")
            results[series_id] = {"series_id": series_id, "error": str(e)}

    os.makedirs(MACRO_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = f"{MACRO_DIR}/macro_{stamp}.json"

    with open(path, "w") as f:
        json.dump({"captured": stamp, "series": results}, f, indent=2)

    print(f"\nSaved to {path}")

    print("\nSnapshot context:")
    for series_id, r in results.items():
        if "error" in r:
            continue
        line = f"  {series_id:>14}: {r['latest_value']:>10,.2f}  ({r['latest_date']})"
        if "change" in r:
            line += f"  prior {r['change']:+.2f}"
        if "year_change" in r:
            line += f"  yr {r['year_change']:+.2f}"
        print(line)

    supply = results.get("MSACSR", {})
    if "latest_value" in supply:
        print(f"\n  Months supply reads as: {interpret_supply(supply['latest_value'])}")
        print("  Note: national figure. Local conditions may differ.")


if __name__ == "__main__":
    main()
