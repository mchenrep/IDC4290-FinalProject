"""Archived severity baseline; use ``python -m crash_severity.experiments`` instead."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT_DIR = Path("output")
RANDOM_STATE = 42

# Time-ordered split
# Train on the earliest data, tune the threshold on the middle, report final numbers on the latest.
# VAL_START = "2023-01-01", TEST_START = "2024-01-01".
VAL_START, TEST_START = "2023-01-01", "2024-01-01"

# Feature sets 
SCENE_CAT = ["weather_condition", "lighting_condition", "roadway_surface_cond", "trafficway_type",
             "alignment", "traffic_control_device", "first_crash_type"]
SCENE_NUM = ["posted_speed_limit", "latitude", "longitude", "hour_sin", "hour_cos", "is_weekend"]
POST_CAT = ["damage", "prim_contributory_cause"]

def load():
    df = pd.read_csv(OUT_DIR / "cleaned_crashes.csv.gz", parse_dates=["crash_date"], low_memory=False)
    df = df.sort_values("crash_date").reset_index(drop=True)
    # Hour of day is cyclic (23:00 is next to 00:00), so encode it on a circle.
    ang = 2 * np.pi * df["hour_of_day"] / 24
    df["hour_sin"], df["hour_cos"] = np.sin(ang), np.cos(ang)
    df["is_weekend"] = df["crash_date"].dt.dayofweek.ge(5).astype(int)
    return df


def split(df):
    tr = df[df["crash_date"] < VAL_START]
    va = df[(df["crash_date"] >= VAL_START) & (df["crash_date"] < TEST_START)]
    te = df[df["crash_date"] >= TEST_START]
    print(f"train {len(tr):,} (severe {tr.severe.mean():.2%}) | val {len(va):,} ({va.severe.mean():.2%}) "
          f"| test {len(te):,} ({te.severe.mean():.2%})")
    return tr, va, te


def make_models(cat, num):
    # Logistic regression: needs one-hot categories and scaled numbers.
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ])
    logreg = Pipeline([("prep", prep),
                       ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))])
    # Gradient boosting: handles categories, missing values and interactions natively.
    gb = HistGradientBoostingClassifier(
        categorical_features="from_dtype", learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=RANDOM_STATE)
    base = DummyClassifier(strategy="prior")
    return {"baseline (prior)": base, "logistic regression": logreg, "gradient boosting": gb}


def prep_frame(df, cat, num):
    X = df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    return X


def align_categories(X_tr, *others):
    """Give every split the same category levels so the models never see a new code."""
    cats = {c: X_tr[c].cat.categories for c in X_tr.select_dtypes("category").columns}
    out = []
    for X in others:
        X = X.copy()
        for c, lv in cats.items():
            X[c] = X[c].astype(object).where(X[c].astype(object).isin(lv)).astype(pd.CategoricalDtype(lv))
        out.append(X)
    return out


def pick_threshold(y, p, min_recall=0.60):
    """Highest-precision threshold on the VALIDATION set that still catches >= min_recall of severe crashes."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= min_recall)[0]
    return float(thr[ok[np.argmax(prec[:-1][ok])]]) if len(ok) else 0.5


def evaluate(name, y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"model": name, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "threshold": thr,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_feature_set(label, df_tr, df_va, df_te, cat, num):
    X_tr, X_va, X_te = (prep_frame(d, cat, num) for d in (df_tr, df_va, df_te))
    X_va, X_te = align_categories(X_tr, X_va, X_te)
    results, fitted, probs = [], {}, {}
    for name, model in make_models(cat, num).items():
        model.fit(X_tr, df_tr["severe"])
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        thr = pick_threshold(df_va["severe"], p_va) if "baseline" not in name else 0.5
        row = evaluate(name, df_te["severe"], p_te, thr)
        row["feature_set"] = label
        results.append(row)
        fitted[name], probs[name] = model, p_te
        print(f"[{label}] {name:20s} ROC-AUC {row['roc_auc']:.3f}  PR-AUC {row['pr_auc']:.3f}  "
              f"P {row['precision']:.3f}  R {row['recall']:.3f}")
    return results, fitted, probs, X_te


def make_plots(label, y_te, probs, fitted, X_te):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for name, p in probs.items():
        if "baseline" in name:
            continue
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y_te, p):.3f})")
        frac, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax[2].plot(mean_pred, frac, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y_te.mean(), color="k", ls="--", label="no-skill (base rate)")
    ax[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    ax[2].plot([0, 1], [0, 1], "k--"); ax[2].set(title="Calibration", xlabel="Mean predicted probability", ylabel="Observed severe rate")
    for a in ax:
        a.legend()
    fig.suptitle(f"Severe-crash model, test set ({label})")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"model_curves_{label}.png", dpi=150); plt.close(fig)

    # Permutation importance: how much does shuffling ONE feature hurt PR-AUC on unseen data?
    imp = permutation_importance(fitted["gradient boosting"], X_te, y_te, scoring="average_precision",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean,
                           "std": imp.importances_std}).sort_values("importance", ascending=False)
    imp_df.to_csv(OUT_DIR / f"model_permutation_importance_{label}.csv", index=False)
    top = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set(title=f"Permutation importance ({label})", xlabel="Drop in PR-AUC when shuffled")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"model_importance_{label}.png", dpi=150); plt.close(fig)
    return imp_df


def main():
    df = load()
    tr, va, te = split(df)
    all_results = []
    for label, cat, num in [("scene_only", SCENE_CAT, SCENE_NUM),
                            ("scene_plus_post_crash", SCENE_CAT + POST_CAT, SCENE_NUM)]:
        res, fitted, probs, X_te = run_feature_set(label, tr, va, te, cat, num)
        all_results += res
        imp = make_plots(label, te["severe"], probs, fitted, X_te)
        print(f"[{label}] top features:\n{imp.head(6).to_string(index=False)}\n")
    pd.DataFrame(all_results).to_csv(OUT_DIR / "model_results.csv", index=False)
    print(f"Wrote results and plots to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()