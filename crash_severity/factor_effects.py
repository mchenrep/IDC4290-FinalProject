"""Which conditions are associated with a crash being severe, and in which direction?

Fits ONE logistic regression of `severe` on the pre-crash conditions and reports adjusted odds ratios
with 95% confidence intervals. An odds ratio above 1 means the condition is associated with a higher
chance that a crash is fatal/incapacitating, compared with the reference level, holding the other listed
conditions fixed. Below 1 means lower. These are ASSOCIATIONS, not causes.

Why this complements the permutation importance: importance says which feature families the model relies
on; this says which way each condition pushes severity, and by how much.

Inference choices
  * Uses crashes BEFORE config.TRAIN_END (the same period as the EDA screening), so the 2023-2024
    validation/test years stay untouched.
  * Standard errors are clustered by grid cell, because crashes in the same place are not independent.
  * Levels with under MIN_LEVEL_SHARE of crashes are merged into OTHER so rare levels cannot blow up.

Run from the repository root (after preprocessing):  python -m crash_severity.factor_effects
Needs:  pip install statsmodels
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from . import config

OUT, FIG = config.OUT_DIR, config.FIG_DIR

CATEGORICAL = {            # column -> reference level (None = use the most common level)
    "lighting_condition": "DAYLIGHT",
    "weather_condition": "CLEAR",
    "roadway_surface_cond": "DRY",
    "trafficway_type": None,
    "alignment": None,
    "traffic_control_device": None,
    "season": None,
}
BINARY = ["is_weekend", "is_rush_hour"]
MIN_LEVEL_SHARE = 0.01     # merge levels rarer than 1% of crashes into OTHER


def load():
    cols = ["crash_date", "severe", "posted_speed_limit", "grid_id", *CATEGORICAL, *BINARY]
    df = pd.read_csv(OUT / "cleaned_crashes.csv.gz", usecols=cols, parse_dates=["crash_date"], low_memory=False)
    df = df[df["crash_date"] < pd.Timestamp(config.TRAIN_END)].copy()
    n0 = len(df)
    df = df.dropna(subset=["posted_speed_limit"])
    print(f"Fitting on {len(df):,} crashes before {config.TRAIN_END} ({n0 - len(df):,} dropped for missing speed limit); "
          f"{int(df['severe'].sum()):,} severe.")
    return df.reset_index(drop=True)


def build_design(df):
    """Dummy-code the predictors. Returns X (with constant), plus a table describing every term."""
    parts, info = [], []
    for col, ref in CATEGORICAL.items():
        s = df[col].fillna("UNKNOWN").astype(str)
        share = s.value_counts(normalize=True)
        s = s.where(s.map(share) >= MIN_LEVEL_SHARE, "OTHER")
        ref_level = ref if ref in set(s) else s.value_counts().idxmax()
        d = pd.get_dummies(s, prefix=col, prefix_sep="=").astype(float)
        d = d.drop(columns=f"{col}={ref_level}")
        parts.append(d)
        for term in d.columns:
            lvl = term.split("=", 1)[1]
            m = s == lvl
            info.append({"term": term, "factor": col, "level": lvl, "reference": ref_level,
                         "n_crashes": int(m.sum()), "n_severe": int(df.loc[m, "severe"].sum())})
    for col in BINARY:
        parts.append(df[[col]].astype(float))
        m = df[col] == 1
        info.append({"term": col, "factor": col, "level": "yes", "reference": "no",
                     "n_crashes": int(m.sum()), "n_severe": int(df.loc[m, "severe"].sum())})
    speed = (df["posted_speed_limit"] / 10.0).rename("speed_limit_per_10mph")
    parts.append(speed.to_frame())
    info.append({"term": "speed_limit_per_10mph", "factor": "posted_speed_limit", "level": "+10 mph",
                 "reference": "per 10 mph", "n_crashes": len(df), "n_severe": int(df["severe"].sum())})
    X = sm.add_constant(pd.concat(parts, axis=1), has_constant="add")
    return X, pd.DataFrame(info)


def fit(df):
    X, info = build_design(df)
    groups = pd.factorize(df["grid_id"].fillna("NO_LOCATION"))[0]
    model = sm.GLM(df["severe"].astype(float), X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = res.conf_int()
    out = pd.DataFrame({"term": res.params.index, "odds_ratio": np.exp(res.params.values),
                        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
                        "p_value": res.pvalues.values})
    out = info.merge(out, on="term", how="left")
    out["clear_effect"] = (out["ci_low"] > 1) | (out["ci_high"] < 1)      # interval excludes 1
    return out.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


def forest_plot(tbl, min_severe=30):
    t = tbl[tbl["n_severe"] >= min_severe].sort_values("odds_ratio")      # skip terms with too few severe crashes
    labels = [f"{r.factor}: {r.level}" if r.factor not in BINARY and r.factor != "posted_speed_limit"
              else (f"{r.factor}" if r.factor in BINARY else "speed limit (+10 mph)") for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(t))))
    y = np.arange(len(t))
    colors = np.where(t["clear_effect"], "#D55E00", "#999999")
    ax.hlines(y, t["ci_low"], t["ci_high"], color=colors, lw=1.5)
    ax.scatter(t["odds_ratio"], y, color=colors, zorder=3, s=18)
    ax.axvline(1, color="black", lw=1, ls="--")
    ax.set(yticks=y, yticklabels=labels, xscale="log", xlabel="Adjusted odds ratio for a severe crash (log scale, 95% CI)",
           title="Conditions associated with severe crashes (orange = interval excludes 1)")
    ax.tick_params(axis="y", labelsize=7)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig9_odds_ratios.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig9_odds_ratios.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    tbl = fit(df)
    tbl.to_csv(OUT / "factor_odds_ratios.csv", index=False)
    forest_plot(tbl)
    show = tbl[tbl["clear_effect"]][["factor", "level", "reference", "odds_ratio", "ci_low", "ci_high", "n_severe"]]
    print("\nConditions whose 95% interval excludes 1 (largest odds ratio first):")
    print(show.round(2).to_string(index=False))
    print(f"\nWrote {OUT / 'factor_odds_ratios.csv'} and {FIG / 'fig9_odds_ratios.png'}")
    print("Reminder: adjusted ASSOCIATIONS, not causal effects; crash data has no exposure (traffic volume) information.")


if __name__ == "__main__":
    main()
