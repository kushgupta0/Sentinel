"""Residual-based scoring with per-market configuration.

Fits log price against property characteristics, then ranks listings
by how far below the model estimate they are priced.

Caps are market-specific by necessity. Lubbock's median home is about
$220k; Frisco's is $725k. Applying one market's thresholds to another
would exclude ordinary houses rather than genuine outliers. The caps
exist to keep the model inside the range where training data is dense,
so they have to be set from each market's own distribution.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

PROCESSED_DIR = "data/processed"
DEFAULT_MARKET = "lubbock"

MIN_ZIP_COUNT = 5
CURRENT_YEAR = 2026

MARKETS = {
    "lubbock": {"max_sqft": 4000, "max_price": 800_000},
    "frisco": {"max_sqft": 6500, "max_price": 2_000_000},
}

# Lot size is right-skewed: median around 7,500 sqft in both markets
# but with rural parcels up to 20 acres. Entered as a log so a
# quarter-acre lot and a 20-acre parcel sit on a comparable scale.
SPECS = {
    "area_only": ["squareFootage", "age"],
    "with_baths": ["squareFootage", "bathrooms", "age"],
    "with_lot": ["squareFootage", "age", "logLot"],
    "full": ["squareFootage", "bathrooms", "age", "logLot"],
}


def load(market, config):
    path = f"{PROCESSED_DIR}/{market}_clean.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["price", "squareFootage", "yearBuilt", "bathrooms"])
    df = df[(df["yearBuilt"] > 0) & (df["price"] > 0)]
    df["age"] = CURRENT_YEAR - df["yearBuilt"]

    df = df[df["lotSize"].notna() & (df["lotSize"] > 0)]
    df["logLot"] = np.log(df["lotSize"])

    before = len(df)
    df = df[
        (df["squareFootage"] <= config["max_sqft"])
        & (df["price"] <= config["max_price"])
    ]
    capped = before - len(df)

    counts = df["zipCode"].value_counts()
    keep = counts[counts >= MIN_ZIP_COUNT].index
    thin = len(df) - len(df[df["zipCode"].isin(keep)])
    df = df[df["zipCode"].isin(keep)].copy()

    print(f"Market: {market}")
    print(f"Modeling {len(df)} listings across {df['zipCode'].nunique()} zips")
    if capped:
        print(f"Excluded {capped} outside range "
              f"(> {config['max_sqft']:,} sqft or > ${config['max_price']:,})")
    if thin:
        print(f"Dropped {thin} in thin zips")
    return df


def describe(df):
    print("\nSample distribution:")
    for col, fmt in [("price", "${:,.0f}"), ("squareFootage", "{:,.0f}"),
                     ("age", "{:.0f}")]:
        s = df[col]
        print(f"  {col:>14}: median {fmt.format(s.median()):>10}  "
              f"p10 {fmt.format(s.quantile(0.1)):>10}  "
              f"p90 {fmt.format(s.quantile(0.9)):>10}")


def check_collinearity(df):
    cols = ["squareFootage", "bedrooms", "bathrooms", "age"]
    present = [c for c in cols if c in df.columns]
    corr = df[present].corr()

    print("\nFeature correlations:")
    print(corr.round(2).to_string())

    high = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            if abs(corr.loc[a, b]) > 0.7:
                high.append((a, b, corr.loc[a, b]))
    if high:
        print("\n  Strongly correlated pairs (|r| > 0.7):")
        for a, b, r in high:
            print(f"    {a} / {b}: {r:.2f}")


def build_matrix(df, features):
    X = df[features].copy()
    zips = pd.get_dummies(df["zipCode"], prefix="zip", drop_first=True)
    X = pd.concat([X.reset_index(drop=True), zips.reset_index(drop=True)], axis=1)
    return X.astype(float)


def fit(df, features):
    X = build_matrix(df, features)
    y = np.log(df["price"].values)

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    cv = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=1))
    return model, X, r2, cv.mean(), cv.std()


def compare_specs(df):
    """Fit each candidate specification and report held-out performance.

    Selection prefers the simplest specification unless a richer one
    beats it by more than the noise band. Picking the maximum when the
    difference sits inside the cross-validation standard deviation is
    not a real selection.
    """
    print("\nSpecification comparison:")
    results = {}
    for name, features in SPECS.items():
        _, _, r2, cv_mean, cv_std = fit(df, features)
        results[name] = (cv_mean, cv_std, len(features))
        print(f"  {name:>12}: in-sample {r2:.3f}, "
              f"CV {cv_mean:.3f} (+/- {cv_std:.3f})  {features}")

    baseline = "area_only"
    base_cv = results[baseline][0]

    print(f"\n  Change in CV R-squared vs {baseline}:")
    for name, (cv_mean, cv_std, _) in results.items():
        if name == baseline:
            continue
        print(f"    {name:>12}: {cv_mean - base_cv:+.3f}")

    MEANINGFUL = 0.02
    best = baseline
    best_cv = base_cv
    for name, (cv_mean, _, n_feat) in results.items():
        if cv_mean > best_cv + MEANINGFUL:
            best, best_cv = name, cv_mean

    if best == baseline:
        print(f"\n  No specification beats {baseline} by more than "
              f"{MEANINGFUL:.2f}. Keeping the simplest.")
    else:
        print(f"\n  {best} clears the {MEANINGFUL:.2f} threshold.")

    return best


def report_coefficients(model, columns, features):
    print("\nWhat drives price (approx percent change per unit):")
    pairs = list(zip(columns, model.coef_))

    for name, value in pairs:
        if name in features:
            print(f"  {name:>16}: {100 * (np.exp(value) - 1):>+7.3f}% per unit")

    print("\n  Zip effects vs baseline zip:")
    zip_pairs = sorted(
        [(n, v) for n, v in pairs if n.startswith("zip_")],
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, value in zip_pairs:
        print(f"  {name:>16}: {100 * (np.exp(value) - 1):>+7.1f}%")


def main():
    market = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MARKET).lower()
    if market not in MARKETS:
        print(f"No config for '{market}'. Known: {', '.join(MARKETS)}")
        return

    config = MARKETS[market]
    df = load(market, config)
    describe(df)
    check_collinearity(df)

    best = compare_specs(df)
    features = SPECS[best]
    print(f"\nUsing specification: {best} {features}")

    model, X, r2, cv_mean, cv_std = fit(df, features)
    print(f"\nR-squared (in-sample):   {r2:.3f}")
    print(f"R-squared (5-fold CV):   {cv_mean:.3f} (+/- {cv_std:.3f})")

    predicted = np.exp(model.predict(X))
    df = df.reset_index(drop=True)
    df["predictedPrice"] = predicted.round(0)
    df["residual"] = (df["predictedPrice"] - df["price"]).round(0)
    df["residualPct"] = (100 * df["residual"] / df["predictedPrice"]).round(1)

    print(f"\nNegative or zero predictions: {(df['predictedPrice'] <= 0).sum()}")
    report_coefficients(model, X.columns, features)

    df = df.sort_values("residualPct", ascending=False)
    out = f"{PROCESSED_DIR}/{market}_residual.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    show = ["formattedAddress", "zipCode", "price", "predictedPrice",
            "residualPct", "squareFootage", "yearBuilt", "daysOnMarket"]

    print("\nMost underpriced vs model estimate:")
    print(df[show].head(10).to_string(index=False))

    print("\nMost overpriced vs model estimate:")
    print(df[show].tail(5).to_string(index=False))

    print(f"\nResidual spread: {df['residualPct'].std():.1f}% std dev")
    print(f"Median absolute residual: {df['residualPct'].abs().median():.1f}%")


if __name__ == "__main__":
    main()
