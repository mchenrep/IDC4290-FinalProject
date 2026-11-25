"""Step 5: paper figures and auto-written captions, generated ONLY from files in output/.

Run after the model experiments:   python -m crash_severity.figures
Writes PDF + PNG for every figure to output/figures/ and numbers-from-results captions to output/figures/captions.md.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from . import config

OUT, FIG = config.OUT_DIR, config.FIG_DIR
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]   # colorblind-safe
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 100, "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=PALETTE)})
captions = []


def save(fig, name, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    captions.append(f"**{name}**. {caption}\n")
    print("wrote", name)


def npz(fs, model):
    path = OUT / "preds" / f"{fs.replace('+', '_plus_')}__{model}.npz"
    return np.load(path) if path.exists() else None


def fig_eda(crashes):
    pivot = crashes.pivot_table(index="day_of_week", columns="hour_of_day", values="severe", aggfunc="mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in order if d in pivot.index]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax[0].imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax[0].set(xticks=range(0, 24, 3), xticklabels=range(0, 24, 3), yticks=range(len(pivot)), yticklabels=pivot.index,
              xlabel="Hour of day", title="Severe-crash share by hour and weekday (%)")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], label="% of crashes that were severe")
    yr = crashes.groupby("year")["severe"].agg(["mean", "sum"])
    ax[1].bar(yr.index.astype(str), yr["mean"] * 100, color=PALETTE[0])
    ax[1].set(title="Severe share by year (%)", xlabel="Year", ylabel="% severe")
    peak = pivot.stack().idxmax()
    save(fig, "fig1_severity_patterns",
         f"Share of crashes with a fatal or incapacitating injury ({crashes['severe'].mean():.2%} overall, "
         f"{int(crashes['severe'].sum()):,} of {len(crashes):,} crashes). The highest cell is {peak[0]} at {peak[1]}:00 "
         f"({pivot.stack().max():.1f}%). Left: hour by weekday; right: by year.")


def fig_ablation(results, ablation):
    ctx = results[(results["tier"] == "context") & (results["model"] != "dummy")]
    models = list(dict.fromkeys(ctx["model"]))
    sets = list(dict.fromkeys(ctx["feature_set"]))
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(models) + 3), 4.2))
    w = 0.8 / len(sets)
    for i, fs in enumerate(sets):
        d = ctx[ctx["feature_set"] == fs].set_index("model").reindex(models)
        x = np.arange(len(models)) + i * w - 0.4 + w / 2
        err = np.array([d["pr_auc"] - d["pr_auc_ci_low"], d["pr_auc_ci_high"] - d["pr_auc"]])
        ax.bar(x, d["pr_auc"], w, yerr=err, capsize=2, label=fs)
    prev = ctx["prevalence"].iloc[0]
    ax.axhline(prev, color="k", ls="--", lw=1, label=f"no-skill (prevalence {prev:.3f})")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set(ylabel="Test PR-AUC (severe crashes)", title="Do extra datasets improve prediction? (context features only)")
    ax.legend(fontsize=8, ncol=2)
    c = ablation[ablation["tier"] == "context"]
    verdicts = c.groupby("feature_set")["verdict"].agg(lambda s: "; ".join(f"{k}: {v}" for k, v in s.value_counts().items()))
    txt = " ".join(f"'{k}' vs base -> {v}." for k, v in verdicts.items() if k != "base_no_location")
    save(fig, "fig2_ablation_extra_datasets",
         f"Test-set PR-AUC with 95% bootstrap CIs for each model and feature set; dashed line = prevalence (a model with no skill). "
         f"Paired-bootstrap verdicts across models: {txt}")


def fig_models(results):
    base = results[(results["feature_set"] == "base") & (results["model"] != "dummy")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in base["model"]:
        z = npz("base", m)
        if z is None:
            continue
        pr, rc, _ = precision_recall_curve(z["y"], z["p"])
        ax[0].plot(rc, pr, label=f"{m} (AP {base.set_index('model').loc[m, 'pr_auc']:.3f})")
    ax[0].axhline(base["prevalence"].iloc[0], color="k", ls="--", label="no-skill")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, base features (test)")
    ax[0].legend(fontsize=8)
    b = base.sort_values("pr_auc")
    err = np.array([b["pr_auc"] - b["pr_auc_ci_low"], b["pr_auc_ci_high"] - b["pr_auc"]])
    ax[1].barh(b["model"], b["pr_auc"], xerr=err, color=PALETTE[0], capsize=2)
    ax[1].set(xlabel="Test PR-AUC", title="Model comparison (95% CI)")
    top = b.iloc[-1]
    save(fig, "fig3_model_comparison",
         f"Left: precision-recall curves for each model on the context-only base features. Right: PR-AUC with 95% CIs. "
         f"Best on the test set: {top['model']} (PR-AUC {top['pr_auc']:.3f}, {top['pr_auc_lift']:.2f}x the prevalence of {top['prevalence']:.3f}; "
         f"ROC-AUC {top['roc_auc']:.3f}). Models whose intervals overlap should be treated as tied.")


def fig_details(results):
    d = results[(results["model"] != "dummy") & (results["feature_set"].isin(["base", "context+details"]))]
    if d["feature_set"].nunique() < 2:
        return
    piv = d.pivot(index="model", columns="feature_set", values="pr_auc")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    piv.plot.bar(ax=ax, color=[PALETTE[0], PALETTE[3]], rot=0)
    ax.set(ylabel="Test PR-AUC", title="Adding post-crash details (not usable before a crash happens)")
    ax.axhline(d["prevalence"].iloc[0], color="k", ls="--", lw=1)
    save(fig, "fig4_post_crash_details",
         "PR-AUC using only pre-crash context ('base') versus also using details recorded after the crash (collision type, damage, "
         "contributory cause). The second tier answers a different question (triage after a crash is reported) and is not comparable with the context-only results.")


def fig_calibration(best):
    z = npz(best["feature_set"], best["model"])
    if z is None:
        return
    y, fig, ax = z["y"], *plt.subplots(1, 2, figsize=(10, 4))
    for key, label in [("p", "raw"), ("p_cal", "isotonic-calibrated")]:
        q = pd.qcut(pd.Series(z[key]).rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"p": z[key], "y": y, "q": q}).groupby("q").mean()
        ax[0].plot(g["p"], g["y"], marker="o", label=label)
    hi = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
    ax[0].plot([0, hi], [0, hi], "k--", lw=1)
    ax[0].set(xlabel="Mean predicted probability (decile)", ylabel="Observed severe rate", title="Calibration (test)")
    ax[0].legend()
    ax[1].hist(z["p_cal"], bins=40, color=PALETTE[0])
    ax[1].set(xlabel="Calibrated predicted probability", ylabel="Crashes", yscale="log", title="Distribution of predictions")
    row = pd.read_csv(OUT / "model_results.csv").query("feature_set == @best['feature_set'] and model == @best['model']").iloc[0]
    save(fig, "fig5_calibration",
         f"Reliability of the predicted probabilities for the best context model ({best['model']}, {best['feature_set']}). "
         f"Brier score {row['brier_raw']:.4f} before and {row['brier_calibrated']:.4f} after isotonic calibration (fit on the validation period).")


def fig_importance():
    g, f = OUT / "perm_importance_groups.csv", OUT / "perm_importance_features.csv"
    if not g.exists():
        return
    gd, fd = pd.read_csv(g).sort_values("importance"), pd.read_csv(f).sort_values("importance").tail(12)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].barh(gd["group"], gd["importance"], xerr=gd["std"], color=PALETTE[2], capsize=2)
    ax[0].set(xlabel="Drop in PR-AUC when group is shuffled", title="Importance by feature family (validation)")
    ax[1].barh(fd["feature"], fd["importance"], xerr=fd["std"], color=PALETTE[0], capsize=2)
    ax[1].set(xlabel="Drop in PR-AUC when shuffled", title="Top individual features (validation)")
    top = gd.iloc[-1]
    save(fig, "fig6_feature_importance",
         f"Permutation importance of the best context model on the validation period. The most important family is '{top['group']}' "
         f"(PR-AUC drop {top['importance']:.4f}). Correlated features share credit, so read families rather than single columns.")


def fig_risk():
    c, v = OUT / "risk_cells.csv", OUT / "risk_validation.csv"
    if not c.exists():
        return
    cells, val = pd.read_csv(c), pd.read_csv(v)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.25]}, layout="constrained")
    for a, col, title in [(ax[0], "expected_severe", "Predicted: expected severe crashes (pre-test)"),
                          (ax[1], "test_severe", "Observed: severe crashes in the test period")]:
        s = a.scatter(cells["lon"], cells["lat"], c=np.log1p(cells[col]), s=6, cmap="magma_r")
        a.set(title=title, xlabel="Longitude", ylabel="Latitude", aspect=1 / np.cos(np.radians(41.88)))
        a.grid(False)
        fig.colorbar(s, ax=a, label=f"log(1 + {col.replace('_', ' ')})", fraction=0.05, pad=0.02)
    short = {"model: expected severe crashes": "Model", "baseline: past severe count": "Past severe\ncount",
             "baseline: past crash count": "Past crash\ncount"}
    p = val.pivot(index="score", columns="top_fraction_of_cells", values="share_of_test_severe_captured")
    p = p.rename(index=short).rename(columns=lambda f: f"top {int(round(f * 100))}% of cells")
    p.plot.barh(ax=ax[2], color=[PALETTE[1], PALETTE[0]])
    for frac, c in [(0.05, PALETTE[1]), (0.10, PALETTE[0])]:
        ax[2].axvline(frac, color=c, ls=":", lw=1)                       # dotted = what random ranking would capture
    ax[2].set(xlabel="Share of test-period severe crashes captured (dotted lines = random)", ylabel="",
              title="Held-out validation of the risk ranking")
    ax[2].legend(loc="lower right", fontsize=8)
    m = val[val["score"].str.startswith("model")].set_index("top_fraction_of_cells")
    b = val[val["score"].str.contains("severe count")].set_index("top_fraction_of_cells")
    save(fig, "fig7_risk_map",
         f"Grid-cell risk map (cells of {config.GRID_SIZE_DEG} degrees). Ranked on pre-test data, the model's top 10% of cells contain "
         f"{m.loc[0.10, 'share_of_test_severe_captured']:.1%} of later severe crashes, versus {b.loc[0.10, 'share_of_test_severe_captured']:.1%} "
         f"for ranking by past severe counts alone (random would be 10%). Spearman vs observed counts: "
         f"{m.loc[0.10, 'spearman_vs_test_severe']:.2f} (model) vs {b.loc[0.10, 'spearman_vs_test_severe']:.2f} (past severe count).")


def fig_confusion(best, results):
    z = npz(best["feature_set"], best["model"])
    if z is None:
        return
    pred = (z["p"] >= float(z["thr"])).astype(int)
    cm = confusion_matrix(z["y"], pred, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.1%}\n({cm[i, j]:,})", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not severe", "severe"], yticklabels=["not severe", "severe"],
           xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test)")
    save(fig, "fig8_confusion_matrix",
         f"Row-normalized confusion matrix for {best['model']} ({best['feature_set']}) at the threshold chosen on the validation period "
         f"to catch about 60% of severe crashes. Precision is low because severe crashes are rare.")


def main():
    best = json.loads((OUT / "best_model.json").read_text())
    results = pd.read_csv(OUT / "model_results.csv")
    ablation = pd.read_csv(OUT / "ablation_vs_base.csv")
    crashes = pd.read_csv(OUT / "cleaned_crashes.csv.gz", usecols=["severe", "day_of_week", "hour_of_day", "year"])
    fig_eda(crashes)
    fig_ablation(results, ablation)
    fig_models(results)
    fig_details(results)
    fig_calibration(best)
    fig_importance()
    fig_risk()
    fig_confusion(best, results)
    run = json.loads((OUT / "run_config.json").read_text())
    header = (f"# Figure captions (auto-generated from output/ files)\n\nRun settings: fast_mode={run['fast_mode']}, "
              f"models={run['models_run']}, split={run['split']}.\n\n")
    (FIG / "captions.md").write_text(header + "\n".join(captions))
    print(f"Wrote {len(captions)} figures and captions.md to {FIG}")


if __name__ == "__main__":
    main()
