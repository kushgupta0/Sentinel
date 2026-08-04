# SENTINEL

**Live demo:** https://sentinel-aqcvbwifrdzreksrqpasef.streamlit.app/

A decision-support tool for real estate listings. Pulls active listings for a
market, scores each against comparable properties two independent ways, and
ranks them so a human can prioritize which to investigate.

It does not predict sale price or profit. It narrows a few hundred listings to
a shortlist using only public data, and is explicit about where its own
assumptions break.

---

## Pipeline

```bash
python ingest.py Arlington TX     # one API call, dated raw snapshot
python transform.py arlington     # filter, derive ppsf and price history
python score.py arlington         # percentile composite within zip
python model.py arlington         # residual against hedonic regression
python compare.py                 # diff snapshots to extract outcomes
python macro.py                   # FRED context for the snapshot date
streamlit run app.py              # viewer
```

The API allowance is small, so `ingest.py` hits RentCast once per run and
caches the raw response. Everything downstream reads from disk, which means the
model can be iterated on indefinitely without spending calls.

---

## Two ranking methods

### Percentile composite

Each listing is ranked 0–100 within its zip code on price per square foot, days
on market, and price cut history, then weighted and summed.

Percentile rather than z-score because price distributions are right-skewed and
a single high-end listing distorts a mean-based measure while barely moving a
rank.

Weights depend on an assumed market regime:

| Regime | ppsf | days on market | price cut |
|---|---|---|---|
| Soft | 45% | 35% | 20% |
| Hot | 70% | 20% | 10% |

Set to **soft** based on FRED `MSACSR` (months supply of new houses), which read
9.30 as of June 2026 against a conventional buyers-market threshold of 7.0.
Mortgage rates (`MORTGAGE30US`) were 6.66%, up 0.30 year over year. Both are
national figures standing in for local conditions; proper regime detection
requires a historical baseline per market.

Run `python score.py <market> --compare` to measure how much the choice matters.
Top-10 overlap is 7 of 10 across regimes in both Lubbock and Frisco, with a mean
rank shift of roughly 10% of the list.

### Log-linear hedonic regression

Predicts `log(price)` from square footage, age, lot size, and zip dummies, then
ranks by residual.

Log because the untransformed model produced **negative predicted prices** for
small older homes. Log makes non-positive predictions structurally impossible
and imposes multiplicative rather than additive effects, which matches how
housing premiums actually behave.

Candidate specifications are compared on 5-fold cross-validated R², and the
simplest is kept unless a richer one beats it by more than 0.02. Picking the
maximum when the difference sits inside the cross-validation standard deviation
is not a real selection.

Sample is capped per market by square footage and price. The log specification
compounds, so out-of-support extrapolation is severe: an 11,373 sqft Lubbock
listing drew an $11.5M prediction before caps were added.

**Where the two methods agree, the signal is stronger than either alone.**

---

## Findings across six markets

| Market | Listings | Zips | Median price | Median year | CV R² | Median abs residual |
|---|---|---|---|---|---|---|
| Lubbock | 298 | 12 | $205K | 1987 | 0.792 | 13.8% |
| Frisco | 471 | 4 | $700K | 2012 | 0.791 | 11.4% |
| Anna | 456 | 1 | $374K | 2022 | 0.644 | 7.1% |
| Celina | 475 | 3 | $549K | 2021 | 0.809 | 9.5% |
| Prosper | 483 | 1 | $872K | 2018 | 0.843 | 10.1% |
| Arlington | 456 | 13 | $380K | 1985 | 0.834 | 7.6% |

### Lot size matters where land is scarce

Adding lot size improves held-out fit by 0.003 in Lubbock, where it is rejected
as noise, and by 0.108 in Prosper. The coefficient tracks how built out each
market is:

| Market | Δ CV R² | logLot coefficient |
|---|---|---|
| Lubbock | +0.003 | rejected |
| Anna | +0.108 | +15.1% |
| Celina | +0.072 | +22.9% |
| Arlington | +0.073 | +23.1% |
| Frisco | +0.026 | +26.9% |
| Prosper | +0.108 | +39.9% |

Lubbock can expand in any direction. Prosper cannot. The model recovers that
without being told.

### Condos corrupt the model and are excluded

Their reported lot size is the shared parcel rather than the unit's land. Frisco
condos report a median lot of **175,242 square feet**, roughly four acres each.
They also carry monthly assessments the data does not expose.

Before exclusion they held 4 of Arlington's top 10 underpriced slots, and two
adjacent units in the same building landed at opposite ends of the ranking.

Removing them cut cross-validation standard deviation from 0.046 to 0.011 in
Arlington and from 0.053 to 0.017 in Frisco, and revealed that Frisco's true lot
coefficient was 50% higher than previously measured. Lower R², substantially
lower error, far more stable.

### Townhouses are kept, on evidence

Their price per square foot tracks single family within a few percent in every
market (Arlington 191 vs 177, Lubbock 127 vs 129, Frisco 237 vs 234), and they
report genuine smaller lots that the lot size term already handles.

### New construction does not distort the ranking

Even at 52% of the market. In Anna it is 52% of listings but only 20% of the top
50, with a median rank of 282 of 456. Same pattern in Celina (41%), Prosper
(34%), and Arlington (1%). Builders price to the market, so the model correctly
reads these as fairly priced.

### Bathrooms adds nothing beyond square footage

It correlates with area at 0.78 to 0.87 and changes cross-validated R² by −0.002
in Lubbock and +0.008 in Frisco, both inside the noise band.

Bedrooms was dropped outright after producing a **−$42,830** coefficient in the
first linear specification, a collinearity artifact offsetting an inflated area
term, and the direct cause of the negative price predictions.

---

## Limitations

**No ground truth.** No listing in this data has sold. The model is fit to
asking prices, so it learns what sellers request, not what buyers pay. Weights
are assumptions rather than fitted parameters and accuracy is untested.
`compare.py` exists to address this; the first outcome comparison is scheduled
for October 2026, sixty days after the August 2 baseline.

**Sale prices are unavailable.** Texas is a non-disclosure state and the data
source records only listing events, so the eventual outcome variable is binary:
the listing sold or was withdrawn.

**Condition is invisible.** Roof, foundation, layout, and renovation history are
not in the data. A listing priced far below its peers is usually cheap for a
reason the listing does not state. Listings under 60% of their zip median per
square foot are set aside rather than ranked; every one caught in Lubbock was
built between 1929 and 1959.

**Distressed sales are only partially identifiable.** The source labels short
sales but not foreclosures or REO. A verified foreclosure in the Frisco sample
was labeled "Standard" and ranked first by residual.

**Property type labels are unreliable for attached units.** After excluding
condos, one Arlington listing typed as Townhouse is an apartment by address.

**Zip is a coarse peer group.** Zip codes are mail routes. Anna and Prosper have
one each, so the percentile ranking has no group finer than the whole city and
the regression has no location term at all.

**Structure is assumed to be the asset.** Size, age, and lot drive the model.
Where buyers are primarily purchasing land, scarcity, or access, and the
structure is incidental or slated for teardown, the age penalty has the wrong
sign entirely. Zip dummies shift the intercept but cannot vary the slope.

**Requires inventory volume.** Highland Park, TX returned 32 active listings
across 3 zips, priced from $395K to $21.5M. No parameter choice makes that
modelable, and the tool should not try.

---

## Stack

Python · RentCast API · FRED API · scikit-learn · pandas · Altair · Streamlit

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

echo "RENTCAST_API_KEY=your_key" > .env
echo "FRED_API_KEY=your_key" >> .env

python ingest.py Lubbock TX
python transform.py lubbock
python score.py lubbock
python model.py lubbock
```

Data snapshots captured 2026-08-02 (Lubbock, Frisco) and 2026-08-04 (Anna,
Celina, Prosper, Arlington).
