"""Step 3-4: tune, train and compare models, test whether the extra datasets help, calibrate, interpret, build the risk map.

Typical use
    python run_experiments.py                  # full run on config.py's dates (2019-2024)
    python run_experiments.py --fast           # quick smoke run (small training sample, tiny budgets)

Design rules (so the comparison is honest)
  * Chronological split: train < TRAIN_END <= validation < VAL_END <= test.
  * All choices (hyperparameters, thresholds, calibration, "best model") use TRAIN and VALIDATION data only.
    Test-set metrics are computed after those choices are made and never feed back into them.
  * Hyperparameters are tuned once per model on the `base` feature set and reused for every other feature set,
    so no feature set gets extra tuning.
  * Only 'context' features (known before the crash) are used in the main comparison. 'context+details' is a separate tier.
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

import config
import risk_map
from evaluate import bootstrap_ci, evaluate, paired_bootstrap_diff, pick_threshold
from features import FEATURE_SETS, assert_no_leakage, resolve_feature_set
from interpret import feature_and_group_importance
from models import BASELINES, available_models, make_frames

FAST_OVERRIDES = {"rf": {"n_estimators": 60}}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small training sample and tiny budgets (smoke test)")
    ap.add_argument("--models", nargs="*", help="subset of: dummy logreg rf hist_gb xgb lgbm catboost nn")
    ap.add_argument("--feature-sets", nargs="*", help="subset of feature set names (default: all that have data)")
    ap.add_argument("--n-trials", type=int, default=None, help="random-search trials per model (default 8, fast 2)")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per stochastic model (default 5, fast 2)")
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default 300, fast 100)")
    ap.add_argument("--max-train-rows", type=int, default=None, help="subsample the training set (fast default 30000)")
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--skip-risk-map", action="store_true")
    a = ap.parse_args()
    a.n_trials = a.n_trials if a.n_trials is not None else (2 if a.fast else 8)
    a.seeds = a.seeds if a.seeds is not None else (2 if a.fast else 5)
    a.n_boot = a.n_boot if a.n_boot is not None else (100 if a.fast else 300)
    a.max_train_rows = a.max_train_rows if a.max_train_rows is not None else (30_000 if a.fast else None)
    return a


def load_data():
    df = pd.read_csv(config.OUT_DIR / "cleaned_crashes.csv.gz", parse_dates=["crash_date", "hour"], low_memory=False)
    return df.sort_values("crash_date").reset_index(drop=True)


def split(df):
    tr = df[df["crash_date"] < config.TRAIN_END]
    va = df[(df["crash_date"] >= config.TRAIN_END) & (df["crash_date"] < config.VAL_END)]
    te = df[df["crash_date"] >= config.VAL_END]
    rows = [{"split": n, "rows": len(d), "severe": int(d.severe.sum()), "severe_rate": float(d.severe.mean()),
             "start": d.crash_date.min(), "end": d.crash_date.max()} for n, d in [("train", tr), ("val", va), ("test", te)]]
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit("One of the splits is empty. Check CRASH_TRAIN_END / CRASH_VAL_END against the data's date range.")
    if te["severe"].sum() < 500:
        print(f"WARNING: only {int(te['severe'].sum())} severe crashes in the test period; confidence intervals will be wide.")
    return tr, va, te, summary


def sample_params(space, n_trials, seed):
    rng = np.random.default_rng(seed)
    trials = [{}]                                            # trial 0 = library defaults
    for _ in range(max(n_trials - 1, 0)):
        trials.append({k: v[int(rng.integers(len(v)))] for k, v in space.items()})
    return trials


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def tune(cls, name, frames, y_tr, y_va, n_trials, cat, num):
    """Random search on validation PR-AUC (validation only; the test set is never used)."""
    if not cls.PARAM_SPACE or n_trials <= 1:
        return {}, np.nan
    X_tr, X_va, _ = frames
    best_p, best_s = {}, -1.0
    for i, params in enumerate(sample_params(cls.PARAM_SPACE, n_trials, config.SEED)):
        full = {**FAST_OVERRIDES.get(name, {}), **params} if ARGS.fast else params
        s = average_precision_score(y_va, cls(cat, num, seed=config.SEED, **full).fit(X_tr, y_tr, X_va, y_va).predict_proba(X_va))
        print(f"   [{name}] trial {i + 1}/{n_trials}: val PR-AUC {s:.4f}  {params}")
        if s > best_s:
            best_p, best_s = params, s
    return best_p, best_s


def main():
    global ARGS
    ARGS = args = parse_args()
    t0 = time.time()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "preds").mkdir(exist_ok=True)
    df = load_data()
    tr, va, te, split_summary = split(df)
    split_summary.to_csv(config.OUT_DIR / "split_summary.csv", index=False)
    if args.max_train_rows and len(tr) > args.max_train_rows:
        tr = tr.sample(n=args.max_train_rows, random_state=config.SEED).sort_values("crash_date")
        print(f"fast mode: training sample reduced to {len(tr):,} rows")
    y_tr, y_va, y_te = tr["severe"].to_numpy(), va["severe"].to_numpy(), te["severe"].to_numpy()

    registry = available_models()
    wanted = args.models or list(registry)
    models = {n: registry[n] for n in wanted if n in registry}
    skipped = [n for n in ["xgb", "lgbm", "catboost"] if n not in registry and (not args.models or n in args.models)]
    if skipped:
        print(f"NOT INSTALLED, skipped: {skipped}  (pip install -r requirements.txt)")
    if "nn" in models and models["nn"].__name__ == "SkMLPModel":
        print("PyTorch not installed: 'nn' uses the scikit-learn MLP fallback.")

    sets = {}
    for name in (args.feature_sets or list(FEATURE_SETS)):
        r = resolve_feature_set(name, df)
        if r is None:
            print(f"Feature set '{name}' skipped: its external dataset has no data (add NOAA/BTS files).")
            continue
        assert_no_leakage(*r, allow_details=FEATURE_SETS[name]["tier"] == "post-crash details")
        sets[name] = r
    if "base" not in sets:
        raise SystemExit("The 'base' feature set is required.")

    # ---------------- 1. tune once per model on `base` (validation only)
    base_cat, base_num = sets["base"]
    base_frames = make_frames(tr, va, te, base_cat, base_num)
    best_params = {}
    for name, cls in models.items():
        if name == "dummy":
            best_params[name] = {}
            continue
        print(f"Tuning {name} ...")
        best_params[name], _ = tune(cls, name, base_frames, y_tr, y_va, args.n_trials, base_cat, base_num)
    if ARGS.fast:
        best_params = {n: {**FAST_OVERRIDES.get(n, {}), **p} for n, p in best_params.items()}
    (config.OUT_DIR / "best_params.json").write_text(json.dumps(jsonable(best_params), indent=2))

    # ---------------- 2. fit every model on every feature set (configuration is now frozen)
    rows, preds, best = [], {}, {"val_pr_auc": -1}
    for fs_name, (cat, num) in sets.items():
        frames = make_frames(tr, va, te, cat, num)
        X_tr, X_va, X_te = frames
        tier = FEATURE_SETS[fs_name]["tier"]
        for name, cls in models.items():
            n_seeds = args.seeds if cls.stochastic else 1
            pv, pt, seed_ap, first = [], [], [], None
            for s in range(n_seeds):
                m = cls(cat, num, seed=config.SEED + s, **best_params[name]).fit(X_tr, y_tr, X_va, y_va)
                pv.append(m.predict_proba(X_va))
                pt.append(m.predict_proba(X_te))
                seed_ap.append(average_precision_score(y_te, pt[-1]))
                first = first or m
            p_va, p_te = np.mean(pv, axis=0), np.mean(pt, axis=0)           # average over seeds
            thr = 0.5 if name == "dummy" else pick_threshold(y_va, p_va)
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, y_va)   # calibrate on validation
            p_cal = cal.predict(p_te)
            row = evaluate(y_te, p_te, thr, p_cal)
            lo, hi = bootstrap_ci(y_te, p_te, args.n_boot, seed=config.SEED)
            val_ap = float(average_precision_score(y_va, p_va))
            row.update({"feature_set": fs_name, "tier": tier, "model": name, "n_seeds": n_seeds,
                        "pr_auc_ci_low": lo, "pr_auc_ci_high": hi, "pr_auc_seed_sd": float(np.std(seed_ap)),
                        "val_pr_auc": val_ap, "n_train": len(tr), "n_test": len(te)})
            rows.append(row)
            preds[(fs_name, name)] = {"y": y_te, "p": p_te, "p_cal": p_cal}
            np.savez_compressed(config.OUT_DIR / "preds" / f"{fs_name.replace('+', '_plus_')}__{name}.npz",
                                y=y_te, p=p_te, p_cal=p_cal, thr=thr)
            print(f"[{fs_name:22s}] {name:9s} test PR-AUC {row['pr_auc']:.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"ROC-AUC {row['roc_auc']:.3f}  Brier {row['brier_raw']:.4f}->{row['brier_calibrated']:.4f}")
            eligible = tier == "context" and fs_name != "base_no_location" and name != "dummy"
            if eligible and val_ap > best["val_pr_auc"]:                    # best chosen on VALIDATION only
                best = {"val_pr_auc": val_ap, "feature_set": fs_name, "model": name, "obj": first, "X_va": X_va,
                        "cat": cat, "num": num}
    results = pd.DataFrame(rows)
    results.to_csv(config.OUT_DIR / "model_results.csv", index=False)

    # ---------------- 3. do the extra datasets help? (paired bootstrap vs base, same test rows)
    abl = []
    for name in models:
        if name == "dummy":
            continue
        a = preds[("base", name)]
        for fs_name in sets:
            if fs_name == "base":
                continue
            b = preds[(fs_name, name)]
            diff, lo, hi, share = paired_bootstrap_diff(y_te, a["p"], b["p"], args.n_boot, seed=config.SEED)
            verdict = "improves" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
            abl.append({"model": name, "feature_set": fs_name, "tier": FEATURE_SETS[fs_name]["tier"],
                        "pr_auc_base": average_precision_score(y_te, a["p"]), "pr_auc_set": average_precision_score(y_te, b["p"]),
                        "diff": diff, "ci_low": lo, "ci_high": hi, "share_resamples_better": share, "verdict": verdict})
    ablation = pd.DataFrame(abl)
    ablation.to_csv(config.OUT_DIR / "ablation_vs_base.csv", index=False)

    # ---------------- 4. interpretation and risk map for the best context model (chosen on validation)
    if best["val_pr_auc"] < 0:
        raise SystemExit("No eligible model found for interpretation.")
    info = {k: best[k] for k in ["val_pr_auc", "feature_set", "model"]}
    info["params"] = jsonable(best_params[best["model"]])
    (config.OUT_DIR / "best_model.json").write_text(json.dumps(info, indent=2))
    print(f"Best context model (by validation PR-AUC): {info['model']} on '{info['feature_set']}'")
    if not args.skip_importance:
        feats, groups = feature_and_group_importance(best["obj"], best["X_va"], y_va)
        feats.to_csv(config.OUT_DIR / "perm_importance_features.csv", index=False)
        groups.to_csv(config.OUT_DIR / "perm_importance_groups.csv", index=False)
        print(groups.to_string(index=False))
    if not args.skip_risk_map:
        pre = pd.concat([tr, va]).sort_values("crash_date")
        print("Building risk map (5 blocked out-of-fold fits) ...")
        _, rv = risk_map.run(pre, te, models[info["model"]], best_params[info["model"]], best["cat"], best["num"],
                             config.OUT_DIR, seed=config.SEED)
        print(rv.to_string(index=False))

    run_info = {"python": platform.python_version(), "sklearn": sklearn.__version__, "models_run": list(models),
                "models_skipped_not_installed": skipped, "feature_sets_run": list(sets), "fast_mode": args.fast,
                "n_trials": args.n_trials, "seeds": args.seeds, "n_boot": args.n_boot,
                "split": {"start": config.START, "train_end": config.TRAIN_END, "val_end": config.VAL_END,
                          "end": config.END_EXCLUSIVE}, "runtime_seconds": round(time.time() - t0)}
    (config.OUT_DIR / "run_config.json").write_text(json.dumps(run_info, indent=2))
    print(f"Done in {run_info['runtime_seconds']}s. Next: python make_figures.py")


if __name__ == "__main__":
    main()
