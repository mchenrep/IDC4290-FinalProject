"""Step 4: predict HOURLY CRASH COUNTS (how many crashes happen in a given hour).

Reads output/hourly_crash_weather_flights.csv (made by crash_severity.preprocessing).
If NOAA / BTS columns exist in that file they are used automatically; if not,
the model runs on calendar features only, so you can run it today and re-run it
after you add weather and flight data.

Run from the repository root:  python -m crash_severity.count_model
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("output")
RANDOM_STATE = 42

# Time-ordered split by fraction of the timeline (works for 1 year or 6 years of data).
# Train = first 70%, validation = next 15% (early stopping / sanity), test = last 15%.
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

def load():
    df = pd.read_csv(OUT_DIR / "hourly_crash_weather_flights.csv", parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    # Calendar features. Hour-of-day and day-of-week are cyclic, so use sin/cos pairs
    # (23:00 sits next to 00:00 on a circle, not 23 units away on a line).
    h, d, m = df["hour"].dt.hour, df["hour"].dt.dayofweek, df["hour"].dt.month
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["is_weekend"] = (d >= 5).astype(int)
    df["hour_of_day"], df["dow"] = h, d
    # Lag features: the crash count one week earlier at the same hour. Using lag >= 24h
    # means the feature is known when you forecast a day ahead (no leakage).
    df["lag_168"] = df["crash_count"].shift(168)
    df["lag_24"] = df["crash_count"].shift(24)
    return df.dropna(subset=["lag_168"]).reset_index(drop=True)

def pick_features(df):
    calendar = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "is_weekend", "lag_24", "lag_168"]
    # Anything else numeric that isn't the target/time is a NOAA or BTS column.
    skip = set(calendar) | {"hour", "crash_count", "hour_of_day", "dow", "month_sin", "month_cos"}
    external = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return calendar, external

def split(df):
    n = len(df)
    i, j = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]

def main():
    df = load()
    calendar, external = pick_features(df)
    print(f"calendar features: {calendar}")
    print(f"weather/flight features found: {external or 'NONE (calendar-only run)'}")
    # Fill gaps in weather/flight columns with the TRAIN median only (avoids peeking at test).
    tr, va, te = split(df)
    # Remove external features that contain no usable data.
    empty_features = [c for c in external if tr[c].isna().all()]
    if empty_features:
        print("\nRemoving entirely missing features:")
        print(empty_features)
    external = [c for c in external if c not in empty_features]
    # Fill remaining missing values using training-set medians.
    for c in external:
        med = tr[c].median()
        for part in (tr, va, te):
            part[c] = part[c].fillna(med)
    feats = calendar + external
    y_tr, y_va, y_te = tr["crash_count"], va["crash_count"], te["crash_count"]
    # Overdispersion check. A Poisson variable has variance == mean. If variance is far
    # bigger, a plain Poisson model understates uncertainty (consider negative binomial).
    print(f"\ncrash_count mean {y_tr.mean():.2f}, variance {y_tr.var():.2f}, "
          f"var/mean {y_tr.var() / y_tr.mean():.2f}")
    # Baseline: average crashes for each (day-of-week, hour-of-day) slot in the training data.
    slot_mean = tr.groupby(["dow", "hour_of_day"])["crash_count"].mean()
    base_pred = te.set_index(["dow", "hour_of_day"]).index.map(slot_mean).to_numpy(dtype=float)
    # Model 1: Poisson regression (a GLM). Models log(expected count) = linear combination of features.
    pois = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000))
    # Check missing values in training features
    missing = tr[feats].isna().sum()
    print("\nMissing values by feature:")
    print(missing[missing > 0])
    print("\nFeatures that are entirely missing:")
    print(tr[feats].columns[tr[feats].isna().all()].tolist())
    print("\nTotal missing feature values:", tr[feats].isna().sum().sum())
    pois.fit(tr[feats], y_tr)
    # Model 2: gradient boosting with Poisson loss. Captures interactions
    # (e.g. rush hour AND snow) that the GLM can only get if you add them by hand.
    gb = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=RANDOM_STATE)
    gb.fit(tr[feats], y_tr)
    preds = {"baseline (dow x hour mean)": base_pred,
             "poisson regression": pois.predict(te[feats]),
             "gradient boosting (poisson)": gb.predict(te[feats])}
    rows = []
    for name, p in preds.items():
        p = np.clip(p, 1e-6, None)  # Poisson deviance needs strictly positive predictions
        dev = mean_poisson_deviance(y_te, p)
        rows.append({"model": name, "mean_poisson_deviance": dev,
                     "mae": mean_absolute_error(y_te, p),
                     "corr": np.corrcoef(y_te, p)[0, 1]})
    res = pd.DataFrame(rows)
    base_dev = res.loc[0, "mean_poisson_deviance"]
    res["deviance_skill_vs_baseline"] = 1 - res["mean_poisson_deviance"] / base_dev
    print("\nTest-set results (lower deviance/MAE is better):")
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT_DIR / "crash_count_model_results.csv", index=False)

    # Permutation importance: shuffle one feature, measure how much the test deviance gets worse.
    imp = permutation_importance(gb, te[feats], y_te, scoring="neg_mean_poisson_deviance",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": feats, "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(OUT_DIR / "crash_count_permutation_importance.csv", index=False)
    print("\nTop features:\n", imp_df.head(8).round(4).to_string(index=False))

    # Plots: actual vs predicted over the last 2 weeks of the test set, plus calibration by predicted bin.
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    w = te.iloc[-24 * 14:]
    ax[0].plot(w["hour"], w["crash_count"], label="actual", lw=1)
    ax[0].plot(w["hour"], preds["gradient boosting (poisson)"][-24 * 14:], label="predicted", lw=1)
    ax[0].set(title="Last 14 days of test set", ylabel="crashes per hour"); ax[0].legend()
    ax[0].tick_params(axis="x", rotation=30)
    q = pd.qcut(preds["gradient boosting (poisson)"], 10, duplicates="drop")
    cal = pd.DataFrame({"pred": preds["gradient boosting (poisson)"], "actual": y_te.to_numpy()}).groupby(q, observed=True).mean()
    ax[1].plot(cal["pred"], cal["actual"], marker="o"); mx = cal.max().max()
    ax[1].plot([0, mx], [0, mx], "k--")
    ax[1].set(title="Calibration (deciles of predicted)", xlabel="mean predicted", ylabel="mean actual")
    fig.tight_layout(); fig.savefig(OUT_DIR / "crash_count_model_plots.png", dpi=150)
    print(f"\nWrote results and plots to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()