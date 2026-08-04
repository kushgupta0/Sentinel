"""Bare-bones viewer for SENTINEL output.

Reads the scored and residual CSVs produced by the pipeline. Does not
call any API or recompute anything. Run the pipeline first, then this.
"""

import glob
import os
import altair as alt
import pandas as pd
import streamlit as st

PROCESSED_DIR = "data/processed"

st.set_page_config(page_title="SENTINEL", layout="wide")


def available_markets():
    files = glob.glob(f"{PROCESSED_DIR}/*_scored.csv")
    return sorted(os.path.basename(f).replace("_scored.csv", "") for f in files)


@st.cache_data
def load(market, kind):
    path = f"{PROCESSED_DIR}/{market}_{kind}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


st.title("SENTINEL")
st.caption(
    "Ranks active listings by how far their asking price sits below "
    "comparable properties. Not a valuation. A shortlist for a human."
)

markets = available_markets()
if not markets:
    st.error("No scored data found. Run the pipeline first.")
    st.stop()

market = st.selectbox("Market", markets)

scored = load(market, "scored")
residual = load(market, "residual")
flagged = load(market, "flagged")

col1, col2, col3 = st.columns(3)
col1.metric("Listings scored", len(scored))
col2.metric("Zip codes", scored["zipCode"].nunique())
col3.metric("Flagged as outliers", len(flagged) if flagged is not None else 0)

tab1, tab2, tab3, tab5, tab4 = st.tabs(
    ["Ranked", "Model residuals", "Flagged", "Across markets",
     "Limitations"]
)

with tab1:
    st.subheader("Percentile composite ranking")
    st.caption(
        "Each listing scored against others in its zip on price per "
        "square foot, days on market, and price cut history."
    )

    n = st.slider("Show top N", 10, 100, 20, key="n_scored")

    cols = [
        "formattedAddress", "compositeScore", "price", "squareFootage",
        "pricePerSqft", "ppsfVsZipMedian", "daysOnMarket",
        "totalPriceChangePct", "yearBuilt", "zipCode", "listingType",
    ]
    cols = [c for c in cols if c in scored.columns]

    st.dataframe(
        scored[cols].head(n),
        width='stretch',
        hide_index=True,
    )

with tab2:
    if residual is None:
        st.info("No residual file. Run model.py for this market.")
    else:
        st.subheader("Regression residuals")
        st.caption(
            "Predicted price from square footage, age, and zip. "
            "Positive residual means listed below the estimate."
        )

        pts = residual[[
            "price", "predictedPrice", "residualPct",
            "formattedAddress", "squareFootage", "yearBuilt",
        ]].copy()

        lo = float(min(pts["price"].min(), pts["predictedPrice"].min()))
        hi = float(max(pts["price"].max(), pts["predictedPrice"].max()))

        # Reference line where predicted equals actual.
        diag = pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})

        line = (
            alt.Chart(diag)
            .mark_line(color="#c85a5a", strokeDash=[6, 4])
            .encode(x="x:Q", y="y:Q")
        )

        dots = (
            alt.Chart(pts)
            .mark_circle(size=55, opacity=0.55)
            .encode(
                x=alt.X("predictedPrice:Q", title="Model estimate"),
                y=alt.Y("price:Q", title="Asking price"),
                # Domain clamped to +/- 40 percent. Beyond that the
                # scale is dominated by a handful of extreme listings
                # and the ordinary range loses all contrast.
                color=alt.Color(
                    "residualPct:Q",
                    title="Residual %",
                    scale=alt.Scale(
                        scheme="redyellowblue",
                        reverse=True,
                        domain=[-40, 40],
                        clamp=True,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("formattedAddress:N", title="Address"),
                    alt.Tooltip("price:Q", title="Asking", format="$,.0f"),
                    alt.Tooltip("predictedPrice:Q", title="Estimate",
                                format="$,.0f"),
                    alt.Tooltip("residualPct:Q", title="Residual %",
                                format=".1f"),
                    alt.Tooltip("squareFootage:Q", title="Sqft", format=","),
                    alt.Tooltip("yearBuilt:Q", title="Built"),
                ],
            )
        )

        st.altair_chart(
            (line + dots).interactive().properties(height=450),
            width='stretch',
        )

        st.caption(
            "Dashed line is where asking price equals the model estimate. "
            "Points below it are listed under. Median absolute residual "
            "is 12 to 14 percent, so anything inside that band is noise. "
            "Hover for details."
        )

        n2 = st.slider("Show top N", 10, 100, 20, key="n_resid")
        rcols = [
            "formattedAddress", "residualPct", "price", "predictedPrice",
            "squareFootage", "yearBuilt", "daysOnMarket", "zipCode",
        ]
        rcols = [c for c in rcols if c in residual.columns]
        st.dataframe(
            residual[rcols].head(n2),
            width='stretch',
            hide_index=True,
        )

with tab3:
    if flagged is None or len(flagged) == 0:
        st.info("No listings flagged in this market.")
    else:
        st.subheader("Excluded from ranking")
        st.caption(
            "Priced below 60 percent of their zip median per square "
            "foot. Extreme relative discount usually reflects something "
            "the data does not capture. Set aside rather than deleted."
        )
        st.dataframe(flagged, width='stretch', hide_index=True)

with tab5:
    st.subheader("How the markets differ")
    st.caption(
        "Same code, six markets. The model fits each one separately, "
        "and what drives price is not the same everywhere."
    )

    rows = []
    for mk in markets:
        d = load(mk, "clean")
        r = load(mk, "residual")
        if d is None or r is None:
            continue
        rows.append({
            "Market": mk.title(),
            "Listings": len(d),
            "Zips": d["zipCode"].nunique(),
            "Median price": f"${d['price'].median():,.0f}",
            "Median sqft": f"{d['squareFootage'].median():,.0f}",
            "Median year": int(d["yearBuilt"].median()),
            "Median abs residual": f"{r['residualPct'].abs().median():.1f}%",
        })

    st.dataframe(
        pd.DataFrame(rows), width='stretch', hide_index=True
    )

    st.markdown("""
**Lot size is the clearest cross-market finding.** Adding it to the
model improves held-out fit by 0.003 in Lubbock, where it is rejected
as noise, and by 0.108 in Prosper. The coefficient tracks how built
out each market is: land carries a premium where there is no more of
it, and almost none where the city can expand in any direction.

**Age behaves differently in new markets.** The discount per year of
age is 0.68% in Lubbock, where the stock spans 1929 to 2023, and
0.05% in Celina, where the median house is five years old. With no
age variation there is nothing for the term to explain.

**New construction does not distort the ranking**, even where it is
half the market. In Anna it is 52% of listings but only 20% of the
top 50, with a median rank in the bottom 40%. Builders price to the
market, so the model correctly reads these as fairly priced.

**Condos do distort it, badly.** Their reported lot size is the
shared parcel rather than the unit's land. Frisco condos report a
median lot of 175,242 square feet, roughly four acres each. Excluding
them cut cross-validation variance three to four fold and revealed
that Frisco's true lot coefficient was 50% higher than measured.
""")

with tab4:
    st.subheader("What this tool cannot do")
    st.markdown("""
**No ground truth.** No listing in this data has sold. The model is fit
to asking prices, so it learns what sellers request, not what buyers
pay. Accuracy is untested.

**Condition is invisible.** Roof, foundation, layout, and renovation
history are not in the data. A house priced far below its peers is
usually cheap for a reason the listing does not state.

**Distressed sales are only partly labeled.** Short sales are flagged
by the data source. Foreclosures and REO are not. A verified
foreclosure in the Frisco sample was labeled Standard and ranked first
by residual.

**Zip is a coarse peer group.** Zip codes are mail routes. Frisco has
four of them citywide, so real submarket variation happens below the
level this tool can see.

**Structure is assumed to be the asset.** Size and age drive the model.
In markets where buyers are purchasing land, scarcity, or access, the
age penalty has the wrong sign entirely.

**Requires inventory volume.** Highland Park returned 32 active
listings across 3 zips. No parameter choice makes that modelable.
""")

st.divider()
st.caption(
    "github.com/kushgupta0/Sentinel  ·  Data from RentCast, captured "
    "2026-08-02 (Lubbock, Frisco) and 2026-08-04 (Anna, Celina, "
    "Prosper, Arlington).  ·  Macro context from FRED."
)
