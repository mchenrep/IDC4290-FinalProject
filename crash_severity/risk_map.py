"""Turn predicted severity into a grid-cell risk map and test it on the held-out period.

1. Out-of-fold predictions for every pre-test crash (blocked by month, so a crash is never scored by a model trained on its own month).
2. Calibrate (isotonic), then add up the predicted probabilities per grid cell = expected number of severe crashes.
3. Rank cells and check how many of the TEST-period severe crashes fall in the top 5% / 10% of cells,
   compared with two naive baselines (past severe count, past crash count).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .models import make_category_frame


def run(pre, test, model_cls, params, cat, num, out_dir, seed=42, k=5):
    pre = pre.reset_index(drop=True).copy()
    y = pre["severe"].to_numpy()
    X = make_category_frame(pre, cat, num)
    groups = pre["crash_date"].dt.to_period("M").astype(str).to_numpy()
    k = max(2, min(k, len(set(groups))))
    oof = np.full(len(pre), np.nan)
    for tr_idx, te_idx in GroupKFold(n_splits=k).split(X, y, groups):
        inner = np.random.default_rng(seed).random(len(tr_idx)) < 0.1      # small holdout for early stopping
        fit_idx, val_idx = tr_idx[~inner], tr_idx[inner]
        m = model_cls(cat, num, seed=seed, **params).fit(X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], y[val_idx])
        oof[te_idx] = m.predict_proba(X.iloc[te_idx])
    pre["p_cal"] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y).predict(oof)

    cells = (pre[pre["grid_id"].notna()].groupby("grid_id")
             .agg(n_crashes=("severe", "size"), severe_hist=("severe", "sum"), expected_severe=("p_cal", "sum"),
                  mean_risk=("p_cal", "mean"), lat=("latitude", "mean"), lon=("longitude", "mean")).reset_index())
    t = (test[test["grid_id"].notna()].groupby("grid_id")["severe"].agg(test_crashes="size", test_severe="sum").reset_index())
    cells = cells.merge(t, on="grid_id", how="left").fillna({"test_crashes": 0, "test_severe": 0})
    total_test_severe = int(test["severe"].sum())
    in_known = int(cells["test_severe"].sum())

    scores = {"model: expected severe crashes": "expected_severe",
              "baseline: past severe count": "severe_hist", "baseline: past crash count": "n_crashes"}
    rows = []
    for name, col in scores.items():
        order = cells.sort_values([col, "n_crashes"], ascending=False)
        rho = spearmanr(cells[col], cells["test_severe"])[0] if cells[col].nunique() > 1 else np.nan
        for frac in (0.05, 0.10):
            top = order.head(max(int(round(len(order) * frac)), 1))
            rows.append({"score": name, "top_fraction_of_cells": frac,
                         "share_of_test_severe_captured": float(top["test_severe"].sum() / max(in_known, 1)),
                         "random_expectation": frac, "spearman_vs_test_severe": float(rho),
                         "n_cells": len(cells), "test_severe_in_known_cells": in_known,
                         "test_severe_total": total_test_severe})
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out_dir / "risk_cells.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "risk_validation.csv", index=False)
    return cells, result
