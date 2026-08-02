# SENTINEL

A decision-support tool for real estate listings. Pulls active listings
for a target market, scores each against comparable nearby properties,
and ranks them so a human can prioritize which to investigate further.

It does not predict sale price or profit. It narrows a few hundred
listings down to a shortlist worth a closer look, using only publicly
available data.

## Pipeline

1. `ingest.py` — pulls active listings from RentCast, saves a dated
   raw snapshot. One API call per run.
2. `transform.py` — filters to usable residential listings, derives
   price per square foot and price movement history.
3. `score.py` — groups by zip, flags price outliers, ranks the
   remainder on a weighted composite.
4. `compare.py` — diffs two dated snapshots to extract outcomes.

## Scoring model

Each factor is converted to a percentile rank within the listing's zip
code, then weighted. Percentile rather than z-score because price
distributions are right-skewed and a single high-end outlier would
distort a mean-based measure.

Weights depend on an assumed market regime:

- Soft market: ppsf 45%, days on market 35%, price cut 20%
- Hot market:  ppsf 70%, days on market 20%, price cut 10%

Seller-motivation signals carry more weight in a soft market, where
listings that sit are more likely to reflect seller circumstance than
scarcity of buyers.

Run `python score.py --compare` to see how much the regime choice
moves the ranking.

## Filters

- Listings over 365 days on market are excluded as stale.
- Land and listings missing square footage are excluded.
- Zips with fewer than 5 comparable listings are excluded; a median
  computed from three properties is not meaningful.
- Listings priced below 60% of their zip median per square foot are
  flagged and set aside rather than ranked. Extreme relative
  cheapness usually reflects something the data does not capture.
  These are written to a separate file so exclusions stay auditable.

## Known limitations

- **No ground truth.** No listing outcomes have been observed yet, so
  the weights are assumptions rather than fitted parameters and the
  model's accuracy is untested. `compare.py` exists to address this.
- **Zip is a coarse peer group.** Zip codes are mail routes. A single
  zip can span multiple submarkets and build eras.
- **Price per square foot favors larger homes.** Fixed costs (kitchen,
  HVAC, lot) are spread over more square footage in a large house, so
  smaller homes carry a structurally higher ppsf.
- **Motivation signals are proxies.** Days on market and price cuts
  are consistent with a motivated seller, but also with a bad layout,
  a poor location, or a condition problem the listing does not show.
- **Missing variables.** Condition, street quality, school zone, and
  renovation history are not in the data and are not inferable from it.
  **Requires inventory volume.** The method needs enough active
  listings per zip to form meaningful peer groups and estimate
  neighborhood effects. Highland Park, TX returned 32 total active
  listings across 3 zips, with a price range from $395K to $21.5M.
  No parameter choice makes that modelable. Small, exclusive, or
  low-turnover markets fall outside the tool's domain.
- **Assumes structure drives price.** Size and age are the primary
  inputs, which holds where the house is the asset. In markets where
  buyers are primarily purchasing land, scarcity, or access, and the
  structure is incidental or slated for teardown, the model is
  measuring the wrong thing. An age penalty is actively wrong where
  age carries a premium.

## Stack

Python 3.9, RentCast API. No external dependencies beyond `requests`
and `python-dotenv`.

## Setup
