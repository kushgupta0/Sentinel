import csv
import statistics
from collections import defaultdict

INPUT = "data/processed/listings_clean.csv"
OUTPUT = "data/processed/listings_scored.csv"

MIN_GROUP_SIZE = 5

W_PPSF = 0.45
W_PRICE = 0.35
W_DOM = 0.20

NUMERIC = ["price", "squareFootage", "daysOnMarket", "pricePerSqft", "yearBuilt"]


def load():
    with open(INPUT) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k]) if r[k] else None
    return rows


def percentile_rank(value, population):
    """Percent of population strictly below value. 0-100."""
    below = sum(1 for v in population if v < value)
    return 100.0 * below / len(population)


def score_group(group):
    ppsf_pop = [r["pricePerSqft"] for r in group]
    price_pop = [r["price"] for r in group]
    dom_pop = [r["daysOnMarket"] for r in group]

    ppsf_median = statistics.median(ppsf_pop)
    price_median = statistics.median(price_pop)

    for r in group:
        # inverted: cheaper than peers = higher score
        ppsf_score = 100 - percentile_rank(r["pricePerSqft"], ppsf_pop)
        price_score = 100 - percentile_rank(r["price"], price_pop)
        # higher DOM = more seller fatigue = higher score
        dom_score = percentile_rank(r["daysOnMarket"], dom_pop)

        r["ppsfScore"] = round(ppsf_score, 1)
        r["priceScore"] = round(price_score, 1)
        r["domScore"] = round(dom_score, 1)

        r["compositeScore"] = round(
            W_PPSF * ppsf_score
            + W_PRICE * price_score
            + W_DOM * dom_score,
            1,
        )

        r["ppsfVsZipMedian"] = round(
            100 * (r["pricePerSqft"] - ppsf_median) / ppsf_median, 1
        )
        r["priceVsZipMedian"] = round(
            100 * (r["price"] - price_median) / price_median, 1
        )
        r["zipCount"] = len(group)

    return group


def main():
    rows = load()
    print(f"Loaded {len(rows)} listings")

    groups = defaultdict(list)
    for r in rows:
        groups[r["zipCode"]].append(r)

    scored = []
    dropped = 0
    for zipcode, group in groups.items():
        if len(group) < MIN_GROUP_SIZE:
            dropped += len(group)
            continue
        scored.extend(score_group(group))

    print(f"Scored {len(scored)} across {len(set(r['zipCode'] for r in scored))} zips")
    print(f"Dropped {dropped} in thin zips (< {MIN_GROUP_SIZE} listings)")

    scored.sort(key=lambda r: r["compositeScore"], reverse=True)

    fields = list(scored[0].keys())
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)

    print(f"\nWrote {OUTPUT}")
    print("\nTop 10:")
    for r in scored[:10]:
        print(
            f"  {r['compositeScore']:5.1f}  "
            f"${int(r['price']):>7,}  "
            f"{int(r['squareFootage']):>5,} sqft  "
            f"${r['pricePerSqft']:>6.0f}/sqft  "
            f"{int(r['daysOnMarket']):>4}d  "
            f"{r['zipCode']}  "
            f"{r['formattedAddress']}"
        )


if __name__ == "__main__":
    main()
