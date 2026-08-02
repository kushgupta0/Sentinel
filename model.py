"""Residual-based scoring.

Fits a model of price against property characteristics, then scores
each listing on how far its asking price sits below the estimate.

Specification notes, each fixing a problem found in an earlier version:

1. Price is log-transformed. A plain linear model extrapolated below
   zero for small older homes. Log price makes that structurally
   impossible and matches multiplicative housing effects.

2. Bedrooms is excluded for collinearity with square footage.
   Bathrooms is tested rather than assumed, since it correlates with
   area even more strongly (r = 0.87) and may be acting as a proxy
   for size rather than contributing independently.

3. The sample is capped by area and price. The log specification
   compounds, so it produces absurd estimates outside the range where
   training data exists. With almost no listings above 4,000 sqft,
   the model has no basis for pricing them and should not try.

No sale outcomes have been observed, so R-squared measures how well
listing prices are explained by listed characteristics, not
predictive accuracy against real sales.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

INPUT = "data/processed/listings_clean.csv"
OUTPUT = "data/processed/listings_residual.csv"

MIN_ZIP_COUNT = 5
CURRENT_YEAR = 2026

MAX_SQFT = 4000
MAX_PRICE = 800_000

SPECS = {
    "area_only": ["squareFootage", "age"],
    "with_baths": ["squareFootage", "bathrooms", "age"],
}


def load():
    df = pd.read_csv(INPUT)
    df = df.dropna(subset=["price", "squareFootage", "yearBuilt", "bathrooms"])
    df = df[(df["yearBuilt"] > 0) & (df["price"] > 0)]
    df["age"] = CURRENT_YEAR - df["yearBuilt"]

    before = len(df)
    df = df[(df["squareFootage"] <= MAX_SQFT) & (df["price"] <= MAX_PRICE)]
    capped = before - len(df)

    counts = df["zipCode"].value_counts()
    keep = counts[counts >= MIN_ZIP_COUNT].index
    thin = len(df) - len(df[df["zipCode"].isin(keep)])
    df = df[df["zipCode"].isin(keep)].copy()

    print(f"Modeling {len(df)} listings across {df['zipCode'].nunique()} zips")
    if capped:
        print(f"Excluded {capped} outside range "
              f"(> {MAX_SQFT:,} sqft or > ${MAX_PRICE:,})")
    if thin:
        print(f"Dropped {thin} in thin zips")
    return df


def check_collinearity(df):
    cols = ["squareFootage", "bedrooms", "bathrooms", "age"]
    present = [c for c in cols if c in df.columns]
    corr = df[present].corr()

    print("\nFeature correlations:")
    print(corr.round(2).to_string())

    high = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.7:
                high.append((a, b, r))
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
    """Fit each specification and report which explains more."""
    print("\nSpecification comparison:")
    results = {}
    for name, features in SPECS.items():
        _, _, r2, cv_mean, cv_std = fit(df, features)
        results[name] = cv_mean
        print(f"  {name:>12}: in-sample {r2:.3f}, "
              f"CV {cv_mean:.3f} (+/- {cv_std:.3f})  {features}")

    best = max(results, key=results.get)
    gap = results["with_baths"] - results["area_only"]
    print(f"\n  Bathrooms adds {gap:+.3f} to CV R-squared.")
    if abs(gap) < 0.02:
        print("  Negligible. Area alone explains essentially the same variance,")
        print("  which suggests bathrooms was standing in for size.")
    return best


def report_coefficients(model, columns, features):
    print("\nWhat drives price (approx percent change per unit):")
    pairs = list(zip(columns, model.coef_))

    for name, value in pairs:
        if name in features:
            pct = 100 * (np.exp(value) - 1)
            print(f"  {name:>16}: {pct:>+7.3f}% per unit")

    print("\n  Zip effects vs baseline zip:")
    zip_pairs = sorted(
        [(n, v) for n, v in pairs if n.startswith("zip_")],
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, value in zip_pairs:
        print(f"  {name:>16}: {100 * (np.exp(value) - 1):>+7.1f}%")


def main():
    df = load()
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
    df.to_csv(OUTPUT, index=False)
    print(f"\nWrote {OUTPUT}")

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
