import csv
import statistics
from collections import defaultdict

INPUT = "data/processed/listings_clean.csv"
OUTPUT = "data/processed/listings_scored.csv"
FLAGGED = "data/processed/listings_flagged.csv"

MIN_GROUP_SIZE = 5
OUTLIER_FLOOR = 0.60  # ppsf below this share of zip median = suspect

W_PPSF = 0.60
W_DOM = 0.25
W_CUT = 0.15

NUMERIC = [
    "price",
    "squareFootage",
    "daysOnMarket",
    "pricePerSqft",
    "yearBuilt",
    "priceEventCount",
    "totalPriceChangePct",
    "priceCuts",
]


def load():
    with open(INPUT) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k]) if r[k] else 0.0
    return rows


def percentile_rank(value, population):
    """Percent of population strictly below value. 0-100."""
    below = sum(1 for v in population if v < value)
    return 100.0 * below / len(population)


def flag_outliers(group):
    """Split a zip group into clean listings and suspect ones."""
    ppsf_median = statistics.median(r["pricePerSqft"] for r in group)
    floor = ppsf_median * OUTLIER_FLOOR

    clean, flagged = [], []
    for r in group:
        if r["pricePerSqft"] < floor:
            r["flagReason"] = (
                f"ppsf {r['pricePerSqft']:.0f} below floor {floor:.0f} "
                f"(zip median {ppsf_median:.0f})"
            )
            flagged.append(r)
        else:
            clean.append(r)
    return clean, flagged


def score_group(group):
    ppsf_pop = [r["pricePerSqft"] for r in group]
    dom_pop = [r["daysOnMarket"] for r in group]
    # negate so a bigger cut is a bigger number
    cut_pop = [-r["totalPriceChangePct"] for r in group]

    ppsf_median = statistics.median(ppsf_pop)

    for r in group:
        # cheaper per sqft than peers = higher score
        ppsf_score = 100 - percentile_rank(r["pricePerSqft"], ppsf_pop)
        # longer on market = more seller fatigue = higher score
        dom_score = percentile_rank(r["daysOnMarket"], dom_pop)
        # bigger price cut = more motivated = higher score
        cut_score = percentile_rank(-r["totalPriceChangePct"], cut_pop)

        r["ppsfScore"] = round(ppsf_score, 1)
        r["domScore"] = round(dom_score, 1)
        r["cutScore"] = round(cut_score, 1)

        r["compositeScore"] = round(
            W_PPSF * ppsf_score + W_DOM * dom_score + W_CUT * cut_score, 1
        )

        r["ppsfVsZipMedian"] = round(
            100 * (r["pricePerSqft"] - ppsf_median) / ppsf_median, 1
        )
        r["zipCount"] = len(group)

    return group


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load()
    print(f"Loaded {len(rows)} listings")

    groups = defaultdict(list)
    for r in rows:
        groups[r["zipCode"]].append(r)

    scored, all_flagged = [], []
    thin = 0

    for zipcode, group in groups.items():
        if len(group) < MIN_GROUP_SIZE:
            thin += len(group)
            continue
        clean, flagged = flag_outliers(group)
        all_flagged.extend(flagged)
        if len(clean) < MIN_GROUP_SIZE:
            thin += len(clean)
            continue
        scored.extend(score_group(clean))

    zips = len(set(r["zipCode"] for r in scored))
    print(f"Scored {len(scored)} across {zips} zips")
    print(f"Flagged {len(all_flagged)} as price outliers")
    print(f"Dropped {thin} in thin zips (< {MIN_GROUP_SIZE})")

    scored.sort(key=lambda r: r["compositeScore"], reverse=True)

    fields = list(scored[0].keys())
    write(OUTPUT, scored, fields)

    if all_flagged:
        all_flagged.sort(key=lambda r: r["pricePerSqft"])
        flag_fields = [
            "formattedAddress", "zipCode", "price", "squareFootage",
            "pricePerSqft", "yearBuilt", "daysOnMarket", "flagReason",
        ]
        write(FLAGGED, all_flagged, flag_fields)

    print(f"\nWrote {OUTPUT}")
    print(f"Wrote {FLAGGED}")

    print("\nTop 10:")
    for r in scored[:10]:
        print(
            f"  {r['compositeScore']:5.1f}  "
            f"${int(r['price']):>7,}  "
            f"{int(r['squareFootage']):>5,}sf  "
            f"${r['pricePerSqft']:>5.0f}/sf  "
            f"{r['ppsfVsZipMedian']:>6.1f}%  "
            f"{int(r['daysOnMarket']):>4}d  "
            f"cut {r['totalPriceChangePct']:>6.1f}%  "
            f"{int(r['yearBuilt']) if r['yearBuilt'] else '????'}  "
            f"{r['formattedAddress']}"
        )

    print("\nFlagged as outliers:")
    for r in all_flagged[:10]:
        print(
            f"  ${int(r['price']):>7,}  "
            f"${r['pricePerSqft']:>5.0f}/sf  "
            f"{int(r['yearBuilt']) if r['yearBuilt'] else '????'}  "
            f"{r['formattedAddress']}"
        )


if __name__ == "__main__":
    main()
