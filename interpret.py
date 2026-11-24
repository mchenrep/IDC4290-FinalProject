"""Permutation importance, per feature and per feature family (computed on the VALIDATION set)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import FEATURE_GROUPS


def _permute(X, cols, rng):
    Xp = X.copy()
    perm = rng.permutation(len(X))
    for c in cols:
        Xp[c] = X[c].iloc[perm].array          # .array keeps the 'category' dtype and ignores the index
    return Xp


def permutation_importance(model, X, y, groups, n_repeats=3, seed=0):
    """Drop in PR-AUC when the columns of each group are shuffled together (larger = more important)."""
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X))
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = [base - average_precision_score(y, model.predict_proba(_permute(X, cols, rng))) for _ in range(n_repeats)]
        rows.append({"group": name, "n_features": len(cols), "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def feature_and_group_importance(model, X, y, n_repeats=3, seed=0):
    feats = permutation_importance(model, X, y, {c: [c] for c in X.columns}, n_repeats, seed).rename(columns={"group": "feature"})
    groups = permutation_importance(model, X, y, FEATURE_GROUPS, n_repeats, seed)
    return feats.drop(columns=["n_features"]), groups
