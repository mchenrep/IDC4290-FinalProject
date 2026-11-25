"""Evaluation metrics, thresholds, and bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
                             roc_auc_score)


def pick_threshold(y, p, min_recall=0.60):
    """Highest-precision threshold on the VALIDATION set that still catches >= min_recall of severe crashes."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= min_recall)[0]
    return float(thr[ok[np.argmax(prec[:-1][ok])]]) if len(ok) else 0.5


def recall_at_top(y, p, frac=0.05):
    """Share of all severe crashes found among the top `frac` highest-scored crashes (a fixed review budget)."""
    y = np.asarray(y)
    k = max(int(round(len(y) * frac)), 1)
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def evaluate(y, p, thr, p_cal=None):
    y = np.asarray(y)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(y.mean())
    pr = float(average_precision_score(y, p))
    row = {
        "pr_auc": pr, "pr_auc_lift": pr / prevalence if prevalence else np.nan, "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "recall_at_top5pct": recall_at_top(y, p, 0.05),
        "threshold": thr, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "brier_raw": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }
    if p_cal is not None:
        row["brier_calibrated"] = float(brier_score_loss(y, p_cal))
    return row


def bootstrap_ci(y, p, n_boot=300, seed=0, metric=average_precision_score, alpha=0.05):
    """Percentile bootstrap CI for a metric on one set of predictions."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(y, p_a, p_b, n_boot=300, seed=0, metric=average_precision_score, alpha=0.05):
    """metric(b) - metric(a) on the SAME resampled rows, so noise from the sample cancels out.
    Returns (difference on the full data, CI low, CI high, share of resamples where b > a)."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        diffs.append(metric(y[idx], p_b[idx]) - metric(y[idx], p_a[idx]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, p_b) - metric(y, p_a)), float(lo), float(hi), float(np.mean(np.array(diffs) > 0))
