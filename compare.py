"""Diff two dated snapshots to extract listing outcomes.

A listing present in an earlier snapshot but absent later either sold
or was withdrawn. That is the only ground truth available here.
"""

import json
import glob
import csv
import sys
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def list_snapshots():
    files = sorted(glob.glob(f"{RAW_DIR}/listings_*.json"))
    return [(os.path.basename(f).replace("listings_", "").replace(".json", ""), f)
            for f in files]


def load(path):
    with open(path) as f:
        listings = json.load(f)
    return {l["id"]: l for l in listings}


def classify(listing):
    """Infer what happened to a listing that left the active set."""
    status = listing.get("status")
    removed = listing.get("removedDate")

    if status and status.lower() != "active":
        return status
    if removed:
        return "Removed"
    return "Unknown"


def compare(early_date, early_path, late_date, late_path):
    early = load(early_path)
    late = load(late_path)

    early_ids = set(early)
    late_ids = set(late)

    gone = early_ids - late_ids
    new = late_ids - early_ids
    persisting = early_ids & late_ids

    print(f"Comparing {early_date} -> {late_date}")
    print(f"  {len(early_ids)} listings in earlier snapshot")
    print(f"  {len(late_ids)} listings in later snapshot")
    print(f"  {len(gone)} disappeared")
    print(f"  {len(new)} appeared")
    print(f"  {len(persisting)} still listed")

    price_changes = []
    for pid in persisting:
        old_price = early[pid].get("price")
        new_price = late[pid].get("price")
        if old_price and new_price and old_price != new_price:
            price_changes.append({
                "id": pid,
                "address": early[pid].get("formattedAddress"),
                "oldPrice": old_price,
                "newPrice": new_price,
                "changePct": round(100 * (new_price - old_price) / old_price, 2),
            })

    print(f"  {len(price_changes)} changed price while listed")

    outcomes = []
    for pid in gone:
        l = early[pid]
        outcomes.append({
            "id": pid,
            "address": l.get("formattedAddress"),
            "zipCode": l.get("zipCode"),
            "propertyType": l.get("propertyType"),
            "listPrice": l.get("price"),
            "squareFootage": l.get("squareFootage"),
            "daysOnMarketAtSnapshot": l.get("daysOnMarket"),
            "outcome": classify(l),
            "snapshotFrom": early_date,
            "snapshotTo": late_date,
        })

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    out = f"{PROCESSED_DIR}/outcomes_{early_date}_to_{late_date}.csv"
    if outcomes:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(outcomes[0].keys()))
            w.writeheader()
            w.writerows(outcomes)
        print(f"\nWrote {out}")

    if price_changes:
        pc_out = f"{PROCESSED_DIR}/price_changes_{early_date}_to_{late_date}.csv"
        with open(pc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(price_changes[0].keys()))
            w.writeheader()
            w.writerows(price_changes)
        print(f"Wrote {pc_out}")


def main():
    snaps = list_snapshots()

    if not snaps:
        print("No snapshots found. Run ingest.py first.")
        return

    print(f"Found {len(snaps)} snapshot(s):")
    for date, path in snaps:
        print(f"  {date}")

    if len(snaps) < 2:
        print(
            "\nOnly one snapshot exists, so there is nothing to compare yet.\n"
            "Run ingest.py again on a later date (60 days is a reasonable\n"
            "interval for a market this size), then rerun this script."
        )
        return

    if len(sys.argv) == 3:
        wanted = {d: p for d, p in snaps}
        early_date, late_date = sys.argv[1], sys.argv[2]
        compare(early_date, wanted[early_date], late_date, wanted[late_date])
    else:
        (ed, ep), (ld, lp) = snaps[0], snaps[-1]
        compare(ed, ep, ld, lp)


if __name__ == "__main__":
    main()
