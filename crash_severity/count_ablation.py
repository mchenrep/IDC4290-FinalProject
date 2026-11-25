"""Do weather or O'Hare flight volume help predict HOURLY CRASH COUNTS beyond calendar and recent-history features?

count_model.py ranks features by permutation importance, but flight departures follow the same daily rhythm
as traffic, so a high importance there does not show that flights add information. This script answers the
direct question: refit the same gradient-boosting count model with and without each external data source and
compare held-out Poisson deviance (lower is better).

Uncertainty: per-hour deviance differences are resampled in whole-DAY blocks, because neighbouring hours are
strongly correlated and resampling single hours would give intervals that are too narrow.

Run from the repository root (after preprocessing):  python -m crash_severity.count_ablation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import count_model as cm

WEATHER = ["temp_f", "precip_in", "wind_mph", "visibility_mi", "wx_rain", "wx_snow", "wx_fog", "wx_freezing"]
FLIGHTS = ["flight_departures", "mean_dep_delay_min", "cancelled_flights"]
N_BOOT, SEED = 2000, 42


def row_deviance(y, mu):
    """Per-observation Poisson deviance (0*log(0) is treated as 0)."""
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu))


def fit_predict(tr, te, feats):
    gb = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        random_state=cm.RANDOM_STATE)
    gb.fit(tr[feats], tr["crash_count"])
    return gb.predict(te[feats])


def day_block_ci(diff, days, n_boot=N_BOOT, seed=SEED):
    """95% interval for mean(diff), resampling whole days."""
    s = pd.Series(diff).groupby(np.asarray(days)).agg(["sum", "count"])
    sums, counts = s["sum"].to_numpy(), s["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.mean(diff)), *np.percentile(means, [2.5, 97.5])


def main():
    df = cm.load()
    calendar, external = cm.pick_features(df)
    weather = [c for c in WEATHER if c in external]
    flights = [c for c in FLIGHTS if c in external]
    tr, va, te = cm.split(df)
    tr, te = tr.copy(), te.copy()
    for c in weather + flights:                    # train medians only, as in count_model.py
        med = tr[c].median()
        tr[c], te[c] = tr[c].fillna(med), te[c].fillna(med)

    sets = {"calendar + lags (base)": calendar}
    if weather:
        sets["base + weather"] = calendar + weather
    if flights:
        sets["base + flights"] = calendar + flights
    if weather and flights:
        sets["base + weather + flights"] = calendar + weather + flights

    dev, rows = {}, []
    for name, feats in sets.items():
        mu = fit_predict(tr, te, feats)
        dev[name] = row_deviance(te["crash_count"], mu)
        rows.append({"feature_set": name, "n_features": len(feats), "test_mean_poisson_deviance": dev[name].mean()})
    res = pd.DataFrame(rows)

    base = dev["calendar + lags (base)"]
    days = te["hour"].dt.date.to_numpy()
    out = []
    for name in sets:
        if name == "calendar + lags (base)":
            continue
        d_mean, lo, hi = day_block_ci(base - dev[name], days)        # positive = the added data REDUCES deviance
        verdict = "improves" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
        out.append({"feature_set": name, "deviance_reduction_vs_base": d_mean, "ci_low": lo, "ci_high": hi,
                    "relative_reduction": d_mean / base.mean(), "verdict": verdict})
    ab = pd.DataFrame(out)

    res.to_csv(cm.OUT_DIR / "count_ablation_results.csv", index=False)
    ab.to_csv(cm.OUT_DIR / "count_ablation_vs_base.csv", index=False)
    print("\nTest-set Poisson deviance (lower is better):")
    print(res.round(4).to_string(index=False))
    print("\nPaired difference vs base (positive = added data helps; 95% day-block bootstrap interval):")
    print(ab.round(4).to_string(index=False))
    print(f"\nWrote count_ablation_results.csv and count_ablation_vs_base.csv to {cm.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
