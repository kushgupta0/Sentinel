import csv
import sys
import statistics
from collections import defaultdict

PROCESSED_DIR = "data/processed"
DEFAULT_MARKET = "lubbock"

MIN_GROUP_SIZE = 5
OUTLIER_FLOOR = 0.60  # ppsf below this share of zip median = suspect

# Market regime determines how much weight seller motivation carries.
# Soft market: buyers are scarce, so motivated sellers reveal themselves
# through time on market and price cuts. Those signals get more weight.
# Hot market: everything moves fast, so DOM and cuts carry little
# information and raw relative value dominates.
#
# Set to "soft" based on FRED MSACSR (months supply of new houses),
# which read 9.30 as of 2026-06-01, against a conventional buyers-market
# threshold of 7.0. Mortgage rates (MORTGAGE30US) were 6.66% as of
# 2026-07-30, up 0.30 year over year. See macro.py and data/macro/.
#
# This is a national figure standing in for local conditions. Proper
# regime detection requires a historical baseline for each market,
# which needs multiple snapshots per city.
REGIME = "soft"

WEIGHTS = {
    "soft": {"ppsf": 0.45, "dom": 0.35, "cut": 0.20},
    "hot":  {"ppsf": 0.70, "dom": 0.20, "cut": 0.10},
}

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


def load(market):
    path = f"{PROCESSED_DIR}/{market}_clean.csv"
    with open(path) as f:
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
    """Split a zip group into clean listings and suspect ones.

    A listing priced far below its zip's median per square foot is
    usually a condition problem, not a bargain. We set those aside
    rather than deleting them so the exclusions stay auditable.
    """
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


def score_group(group, weights):
    ppsf_pop = [r["pricePerSqft"] for r in group]
    dom_pop = [r["daysOnMarket"] for r in group]
    cut_pop = [-r["totalPriceChangePct"] for r in group]

    ppsf_median = statistics.median(ppsf_pop)

    for r in group:
        ppsf_score = 100 - percentile_rank(r["pricePerSqft"], ppsf_pop)
        dom_score = percentile_rank(r["daysOnMarket"], dom_pop)
        cut_score = percentile_rank(-r["totalPriceChangePct"], cut_pop)

        r["ppsfScore"] = round(ppsf_score, 1)
        r["domScore"] = round(dom_score, 1)
        r["cutScore"] = round(cut_score, 1)

        r["compositeScore"] = round(
            weights["ppsf"] * ppsf_score
            + weights["dom"] * dom_score
            + weights["cut"] * cut_score,
            1,
        )

        r["ppsfVsZipMedian"] = round(
            100 * (r["pricePerSqft"] - ppsf_median) / ppsf_median, 1
        )
        r["zipCount"] = len(group)

    return group


def build_groups(rows):
    """Group by zip, drop thin zips, split off outliers."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["zipCode"]].append(r)

    usable, all_flagged, thin = [], [], 0

    for zipcode, group in groups.items():
        if len(group) < MIN_GROUP_SIZE:
            thin += len(group)
            continue
        clean, flagged = flag_outliers(group)
        all_flagged.extend(flagged)
        if len(clean) < MIN_GROUP_SIZE:
            thin += len(clean)
            continue
        usable.append(clean)

    return usable, all_flagged, thin


def run_regime(row_groups, regime):
    """Score every group under one weight set. Returns a sorted list."""
    weights = WEIGHTS[regime]
    scored = []
    for group in row_groups:
        scored.extend(score_group(group, weights))
    scored.sort(key=lambda r: r["compositeScore"], reverse=True)
    return scored


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, limit=10):
    for i, r in enumerate(rows[:limit], 1):
        year = int(r["yearBuilt"]) if r["yearBuilt"] else "????"
        print(
            f"  {i:>2}. {r['compositeScore']:5.1f}  "
            f"${int(r['price']):>7,}  "
            f"{int(r['squareFootage']):>5,}sf  "
            f"${r['pricePerSqft']:>5.0f}/sf  "
            f"{r['ppsfVsZipMedian']:>6.1f}%  "
            f"{int(r['daysOnMarket']):>4}d  "
            f"cut {r['totalPriceChangePct']:>6.1f}%  "
            f"{year}  "
            f"{r['formattedAddress']}"
        )


def compare_regimes(row_groups):
    """Run both weight sets and report how much the ranking moves."""
    results = {}
    for regime in WEIGHTS:
        scored = run_regime(row_groups, regime)
        results[regime] = [r["id"] for r in scored]
        print(f"\nTop 10 under '{regime}' weights {WEIGHTS[regime]}:")
        print_table(scored)

    a, b = list(WEIGHTS)
    top_a, top_b = set(results[a][:10]), set(results[b][:10])
    overlap = len(top_a & top_b)

    ranks_a = {pid: i for i, pid in enumerate(results[a])}
    ranks_b = {pid: i for i, pid in enumerate(results[b])}
    shifts = [abs(ranks_a[p] - ranks_b[p]) for p in ranks_a]
    avg_shift = sum(shifts) / len(shifts)

    print(f"\nRegime sensitivity:")
    print(f"  Top 10 overlap: {overlap}/10 properties")
    print(f"  Mean rank shift across all listings: {avg_shift:.1f} positions")
    print(f"  Max rank shift: {max(shifts)} positions")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    market = (args[0] if args else DEFAULT_MARKET).lower()
    compare = "--compare" in sys.argv

    print(f"Market: {market}")
    rows = load(market)
    print(f"Loaded {len(rows)} listings")

    row_groups, all_flagged, thin = build_groups(rows)
    total = sum(len(g) for g in row_groups)

    print(f"Scoring {total} across {len(row_groups)} zips")
    print(f"Flagged {len(all_flagged)} as price outliers")
    print(f"Dropped {thin} in thin zips (< {MIN_GROUP_SIZE})")

    if compare:
        compare_regimes(row_groups)
        return

    scored = run_regime(row_groups, REGIME)

    output = f"{PROCESSED_DIR}/{market}_scored.csv"
    flagged_path = f"{PROCESSED_DIR}/{market}_flagged.csv"

    fields = list(scored[0].keys())
    write(output, scored, fields)

    if all_flagged:
        all_flagged.sort(key=lambda r: r["pricePerSqft"])
        write(flagged_path, all_flagged, [
            "formattedAddress", "zipCode", "price", "squareFootage",
            "pricePerSqft", "yearBuilt", "daysOnMarket", "flagReason",
        ])

    print(f"\nWrote {output}")
    print(f"Wrote {flagged_path}")
    print(f"\nTop 10 under '{REGIME}' weights {WEIGHTS[REGIME]}:")
    print_table(scored)


if __name__ == "__main__":
    main()
