## Two ranking methods

**Percentile composite.** Each listing is ranked 0-100 within its zip
on price per square foot, days on market, and price cut history, then
weighted. Percentile rather than z-score because price distributions
are right-skewed and one high-end listing distorts a mean.

Weights depend on an assumed market regime. Set to soft based on FRED
MSACSR (months supply of new houses), which read 9.30 in June 2026
against a conventional buyers-market threshold of 7.0. Run
`python score.py <market> --compare` to see how much the choice moves
the ranking: top-10 overlap is 7 of 10 across regimes in both Lubbock
and Frisco.

**Log-linear hedonic regression.** Predicts log price from square
footage, age, lot size, and zip, then ranks by residual. Log because
the untransformed model produced negative predicted prices for small
older homes. Specifications are compared on cross-validated R² and the
simplest is kept unless a richer one beats it by more than 0.02.

Where the two methods agree, the signal is stronger than either alone.

## Findings across six markets

| Market | Listings | Zips | Median price | CV R² | Median abs residual |
|---|---|---|---|---|---|
| Lubbock | 298 | 12 | $205K | 0.792 | 13.8% |
| Frisco | 471 | 4 | $700K | 0.791 | 11.4% |
| Anna | 456 | 1 | $374K | 0.644 | 7.1% |
| Celina | 475 | 3 | $549K | 0.809 | 9.5% |
| Prosper | 483 | 1 | $872K | 0.843 | 10.1% |
| Arlington | 456 | 13 | $380K | 0.834 | 7.6% |

**Lot size matters where land is scarce.** Adding it improves held-out
fit by 0.003 in Lubbock, where it is rejected as noise, and by 0.108 in
Prosper. The coefficient tracks how built out each market is: +15% per
log unit in Anna, +27% in Frisco, +40% in Prosper. Lubbock can expand
in any direction; Prosper cannot.

**Condos corrupt the model and were excluded.** Their reported lot size
is the shared parcel rather than the unit's land. Frisco condos report
a median lot of 175,242 square feet, roughly four acres each. Before
exclusion they held 4 of Arlington's top 10 underpriced slots, and two
adjacent units in one building landed at opposite ends of the ranking.
Removing them cut cross-validation standard deviation from 0.046 to
0.011 in Arlington and revealed Frisco's true lot coefficient was 50%
higher than measured.

**Townhouses were kept**, on evidence rather than assumption. Their
price per square foot tracks single family within a few percent in all
six markets (Arlington 191 vs 177, Lubbock 127 vs 129), and they report
genuine smaller lots the model already handles.

**New construction does not distort the ranking**, even at 52% of the
market. In Anna it is 52% of listings but 20% of the top 50, with a
median rank of 282 of 456. Builders price to the market, so the model
correctly reads these as fairly priced.

**Bathrooms adds nothing beyond square footage.** It correlates with
area at 0.78 to 0.87 and changes cross-validated R² by -0.002 in
Lubbock. Bedrooms was dropped outright after producing a -$42,830
coefficient, a collinearity artifact offsetting an inflated area term.

## Limitations

**No ground truth.** No listing in this data has sold. The model is fit
to asking prices, so it learns what sellers request, not what buyers
pay. Weights are assumptions rather than fitted parameters and accuracy
is untested. `compare.py` exists to address this; the first outcome
comparison is scheduled for October 2026, sixty days after the baseline.

**Sale prices are unavailable.** Texas is a non-disclosure state and
the data source records only listing events, so the eventual outcome
variable is binary: the listing sold or was withdrawn.

**Condition is invisible.** Roof, foundation, layout, and renovation
history are not in the data. A listing priced far below its peers is
usually cheap for a reason the listing does not state. Listings under
60% of their zip median per square foot are set aside rather than
ranked; every one caught in Lubbock was built between 1929 and 1959.

**Distressed sales are only partially identifiable.** The source labels
short sales but not foreclosures or REO. A verified foreclosure in the
Frisco sample was labeled Standard and ranked first by residual.

**Property type labels are unreliable for attached units.** After
excluding condos, one Arlington listing typed as Townhouse is an
apartment by address.

**Zip is a coarse peer group.** Zip codes are mail routes. Anna and
Prosper have one each, so the percentile ranking has no group finer
than the whole city and the regression has no location term.

**Structure is assumed to be the asset.** Size, age, and lot drive the
model. Where buyers are purchasing land, scarcity, or access and the
structure is incidental, the age penalty has the wrong sign entirely.

**Requires inventory volume.** Highland Park, TX returned 32 active
listings across 3 zips, priced from $395K to $21.5M. No parameter
choice makes that modelable.

## Stack

Python, RentCast API, FRED API, scikit-learn, Streamlit.

## Setup
