"""Clean a raw snapshot into a usable listing table.

Filters out non-residential listings, stale inventory, and rows
missing the fields the model needs, then derives price per square
foot and price movement history.

Market is passed as an argument so the same pipeline runs against
any city that has been ingested.
"""

import os
import sys
import json
import glob
import csv

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

DEFAULT_MARKET = "lubbock"
MAX_DAYS_ON_MARKET = 365
ALLOWED_TYPES = {"Single Family", "Condo", "Townhouse"}

FIELDS = [
    "id",
    "formattedAddress",
    "zipCode",
    "propertyType",
    "bedrooms",
    "bathrooms",
    "squareFootage",
    "lotSize",
    "yearBuilt",
    "price",
    "daysOnMarket",
]

DERIVED = [
    "pricePerSqft",
    "lotRatio",
    "priceEventCount",
    "totalPriceChangePct",
    "priceCuts",
]


def load_latest_raw(market):
    files = sorted(glob.glob(f"{RAW_DIR}/{market}_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No snapshots for '{market}'. Run: python ingest.py <City> <ST>"
        )
    with open(files[-1]) as f:
        return json.load(f), files[-1]


def is_usable(listing):
    if listing.get("propertyType") not in ALLOWED_TYPES:
        return False
    if not listing.get("price"):
        return False
    if not listing.get("squareFootage"):
        return False
    dom = listing.get("daysOnMarket")
    if dom is None or dom > MAX_DAYS_ON_MARKET:
        return False
    return True


def parse_history(listing):
    history = listing.get("history") or {}

    events = []
    for date, event in history.items():
        price = event.get("price")
        if price:
            events.append((date, price))
    events.sort()

    result = {
        "priceEventCount": len(events),
        "totalPriceChangePct": 0.0,
        "priceCuts": 0,
    }

    if len(events) >= 2:
        first, last = events[0][1], events[-1][1]
        result["totalPriceChangePct"] = round(100 * (last - first) / first, 2)
        result["priceCuts"] = sum(
            1 for i in range(1, len(events))
            if events[i][1] < events[i - 1][1]
        )

    return result


def flatten(listing):
    row = {field: listing.get(field) for field in FIELDS}
    row["pricePerSqft"] = round(row["price"] / row["squareFootage"], 2)

    # Building footprint relative to lot. Low ratio means a large
    # lot for the house, which is where land value shows up.
    lot = row.get("lotSize")
    if lot:
        row["lotRatio"] = round(row["squareFootage"] / lot, 4)
    else:
        row["lotRatio"] = ""

    row.update(parse_history(listing))
    return row


def main():
    market = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MARKET).lower()

    listings, source = load_latest_raw(market)
    print(f"Loaded {len(listings)} listings from {source}")

    usable = [flatten(l) for l in listings if is_usable(l)]
    print(f"{len(usable)} usable after filtering")

    with_cuts = sum(1 for r in usable if r["priceCuts"] > 0)
    multi = sum(1 for r in usable if r["priceEventCount"] > 1)
    print(f"{with_cuts} have at least one price cut")
    print(f"{multi} have more than one price event")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = f"{PROCESSED_DIR}/{market}_clean.csv"

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + DERIVED)
        writer.writeheader()
        writer.writerows(usable)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
