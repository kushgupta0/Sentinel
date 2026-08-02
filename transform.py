import os
import json
import glob
import csv

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

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
    "yearBuilt",
    "price",
    "daysOnMarket",
]


def load_latest_raw():
    files = sorted(glob.glob(f"{RAW_DIR}/listings_*.json"))
    if not files:
        raise FileNotFoundError("No raw files found. Run ingest.py first.")
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


def flatten(listing):
    row = {field: listing.get(field) for field in FIELDS}
    row["pricePerSqft"] = round(row["price"] / row["squareFootage"], 2)
    return row


def main():
    listings, source = load_latest_raw()
    print(f"Loaded {len(listings)} listings from {source}")

    usable = [flatten(l) for l in listings if is_usable(l)]
    print(f"{len(usable)} usable after filtering")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = f"{PROCESSED_DIR}/listings_clean.csv"

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ["pricePerSqft"])
        writer.writeheader()
        writer.writerows(usable)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
