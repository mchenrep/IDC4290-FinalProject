# Chicago Traffic Crash Severity: Preprocessing, EDA, Modeling, Risk Map, Figures

Pipeline for the IDC 4290 final project. It predicts whether a Chicago crash is **severe (fatal or incapacitating injury)**,
tests whether **NOAA weather** and **O'Hare flight** data improve that prediction, and turns the predictions into a
**grid-cell risk map** that is validated on a later, held-out period.

> **Important:** associations and feature importances describe prediction, not causation. Severity is modeled *given that a
> crash was reported*; it is not exposure-adjusted crash risk.

## Run order

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m unittest discover -s tests -v                  # 1. sanity tests (a minute at most)
python crash_preprocess_eda.py                           # 2. download/clean/merge/EDA   -> output/cleaned_crashes.csv.gz etc.
python run_experiments.py --fast                         # 3. quick smoke run (minutes)
python run_experiments.py                                # 4. full run (default: 8 tuning trials, 5 seeds)
python make_figures.py                                   # 5. paper figures + captions  -> output/figures/
```

Run everything from the repository root. The first `crash_preprocess_eda.py` run downloads each month from the City API and
caches it in `data/crash_cache/`; later runs are offline and fast.

## What each file does

| File | Purpose |
|---|---|
| `config.py` | All dates, folders, split points and seeds (every value can be overridden by an environment variable) |
| `features.py` | Time/location features, **feature-set definitions**, the leakage guard |
| `crash_preprocess_eda.py` | Download (cached), clean, build the `severe` label, read NOAA/BTS, join to each crash, EDA screening |
| `models.py` | Baselines + models: logistic regression, random forest, sklearn gradient boosting, XGBoost, LightGBM, CatBoost, neural net |
| `evaluate.py` | PR-AUC, ROC-AUC, Brier, recall@top-5%, thresholds, bootstrap and **paired** bootstrap CIs |
| `run_experiments.py` | Tuning, the models x feature-sets comparison, calibration, ablation vs base, importance, risk map |
| `interpret.py` | Permutation importance per feature and per feature family (validation set) |
| `risk_map.py` | Out-of-fold predictions -> calibrated grid-cell scores -> held-out validation vs naive baselines |
| `make_figures.py` | 8 paper figures (PDF + 300-dpi PNG) and auto-generated captions with numbers taken from the result files |
| `tests/test_pipeline.py` | Label regression test, leakage guards, split order, NOAA/BTS parsing, time zones, shuffled-label test, model smoke test |

## Target, split, and feature sets

- **Target `severe`** = most severe injury is exactly `FATAL` or `INCAPACITATING INJURY` (about 1.8% of crashes).
  Also saved: `any_injury` and a 3-level `severity_class` (0 none, 1 other injury, 2 severe).
- **Chronological split** (edit `config.py`): train 2019-2022, validation 2023, test 2024. Everything chosen (hyperparameters,
  thresholds, calibration, "best model") uses train and validation only.
- **Feature sets** (`features.py`):
  - `base`: time, road/scene, speed limit, location, distance to O'Hare. Known before the crash.
  - `base+weather`, `base+flights`, `base+weather+flights`: add NOAA / BTS features. Skipped automatically if the data is absent.
  - `base_no_location`: robustness check (does the model learn conditions or just memorize places?).
  - `context+details`: **post-crash tier.** Adds collision type, damage and contributory cause, which are recorded *after* the
    crash. Report it separately; never mix it with the context-only results.

## Input data

### Chicago crashes: downloaded automatically
Dataset `85ca-t3if` (Traffic Crashes - Crashes), month by month, for the dates in `config.py`. The City may update past
records, so delete `data/crash_cache/` if you want a fresh pull.

### NOAA hourly weather (optional)
Download NOAA **Local Climatological Data (LCD)** hourly CSVs (start with O'Hare) and put them in `data/noaa/`. Used columns:
`DATE`, `HourlyDryBulbTemperature`, `HourlyPrecipitation` (`T` = trace is handled), `HourlyWindSpeed`, `HourlyVisibility`,
`HourlyPresentWeatherType` (turned into rain / snow / fog / freezing flags).

**Time zone:** LCD timestamps are, to our knowledge, local *standard* time all year, while crash times are clock time. The
script converts NOAA times to Chicago clock time (`NOAA_TIMES_ARE_LOCAL_STANDARD = True` in `config.py`). **Please confirm this
for your export**; set it to `False` if your file is already in clock time.

### BTS flights (optional)
Put BTS **On-Time Performance** monthly CSVs in `data/bts/`. Needed columns: `Origin`, `FlightDate`, `CRSDepTime`;
optional: `DepDelay`, `Cancelled`. Only ORD departures are used, per scheduled hour. Hours with no departures inside the
covered period become real zeros; hours outside the covered period stay missing (`has_flights = 0`).

If a folder is empty the pipeline continues without that dataset. Do not put unrelated CSVs in those folders.

## Outputs (all in `output/`, git-ignored)

- Data: `cleaned_crashes.csv.gz`, `hourly_crash_weather_flights.csv`, `preprocess_log.csv` (row counts per step)
- EDA (training years only): `severity_*_associations.csv`, `severity_by_<field>.csv`, `hourly_crash_count_associations.csv`
- Models: `model_results.csv`, `ablation_vs_base.csv`, `best_params.json`, `best_model.json`, `run_config.json`, `split_summary.csv`, `preds/*.npz`
- Interpretation: `perm_importance_groups.csv`, `perm_importance_features.csv`
- Risk map: `risk_cells.csv`, `risk_validation.csv`
- Figures: `figures/fig1...fig8` (`.pdf` and `.png`) and `figures/captions.md`

## How to read the key result files

- **`ablation_vs_base.csv`** answers "do the extra datasets help?". Each row is a paired-bootstrap comparison of test PR-AUC
  against `base` on the same rows. `verdict` is `improves` only if the 95% CI is entirely above zero.
- **`model_results.csv`** has one row per model x feature set: PR-AUC with CI, lift over prevalence, ROC-AUC, recall in the
  top 5% of scores, Brier score before/after calibration. `pr_auc_seed_sd` is the spread across random seeds.
- **`risk_validation.csv`** shows what share of later severe crashes fall in the top 5% / 10% of cells, for the model and for two
  naive baselines. If the model does not beat "rank by past crash count", say so in the paper.

## Known limitations (put these in the paper)

- Reporting bias: minor crashes are under-reported; severity is conditional on a reported crash.
- Flights cover one airport; weather is one station (O'Hare), so it is a coarse proxy for conditions across the city.
- Grid cells (about 0.5 km) stand in for street segments.
- The `context+details` tier uses information recorded after the crash and answers a triage question, not a prediction one.
- COVID-19 changed traffic in 2020 (inside the training years). Consider a sensitivity run that drops 2020.
- Optional models (XGBoost, LightGBM, CatBoost, PyTorch) were written but could not be run where this code was developed;
  run `python -m unittest tests.test_pipeline.ModelSmoke -v` after installing them.

## Troubleshooting

- **Connection/API error:** retry; finished months are cached, so only missing months are re-downloaded.
- **No weather/flight features:** check the file location and column names (adjust the mappings in `read_noaa()` / `read_bts()`).
- **"One of the splits is empty":** the split dates in `config.py` do not fall inside the downloaded date range.
- **Slow run:** use `--fast`, or lower `--n-trials`, `--seeds`, `--n-boot`; run `--models hist_gb xgb lgbm` to limit models.
