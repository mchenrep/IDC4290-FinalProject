# Chicago Crash Severity Research Pipeline

An end-to-end, reproducible workflow for screening crash, weather, and flight data; comparing severity models; and producing paper-ready tables and figures. The primary outcome is severe injury **among reported crashes**, not whether a crash occurs.

Associations and model importance are predictive, not causal. The pipeline produces analysis tables and figures for the team to interpret; it does not write paper conclusions.

## Quick start

Use Python 3.12 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Optional: install additional boosting/neural-net models
python -m pip install -r requirements-optional.txt
python -m unittest discover -s tests -v
python -m crash_severity
```

The full run may take a while. For a smoke run that skips the spatial analysis:

```powershell
python -m crash_severity --fast --skip-risk-map
```

`python run_pipeline.py` remains available as a short compatibility launcher. `--fast` is for testing the workflow, not for final paper results. The full pipeline reads cached Chicago crash data when available, rebuilds preprocessing and EDA outputs, runs the model comparison, and generates figures. Set `CRASH_DATA_DIR` or `CRASH_OUT_DIR` to use alternate directories. Date and split overrides are in [crash_severity/config.py](./crash_severity/config.py).

To select a smaller set of models or datasets:

```powershell
python -m crash_severity --models logreg hist_gb --feature-sets base base+weather base+flights
```

Optional models are used when installed: XGBoost, LightGBM, CatBoost, and PyTorch. Install them with `python -m pip install -r requirements-optional.txt`. The neural-net entry uses scikit-learn's MLP fallback when PyTorch is unavailable. The run log records models that were skipped.

## Inputs and provenance

The `data/` directory is intentionally git-ignored; it contains large source files and should be shared or downloaded separately.

| Dataset | Expected files | Source and use |
|---|---|---|
| Chicago Traffic Crashes | Monthly CSVs in `data/crash_cache/` | [City of Chicago dataset 85ca-t3if](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if). The pipeline fetches missing months and caches complete months. |
| NOAA hourly weather | LCD CSVs in `data/noaa/` | [NOAA LCD archive](https://www.ncei.noaa.gov/data/local-climatological-data/access/). O'Hare station `USW00094846` is included in the supplied project data. |
| BTS flight operations | Monthly on-time CSVs in `data/bts/` | [Bureau of Transportation Statistics TranStats](https://www.transtats.bts.gov/). The pipeline keeps ORD departures and reads only the needed columns. |

NOAA and BTS are optional if unavailable, but feature sets requiring missing sources are skipped. Expected NOAA fields include `DATE`, `HourlyDryBulbTemperature`, `HourlyPrecipitation`, `HourlyWindSpeed`, `HourlyVisibility`, and `HourlyPresentWeatherType`. Expected BTS fields include flight date, origin, scheduled departure time, and (optionally) departure delay and cancellation status. The readers support common field-name variants; keep only the relevant monthly exports in each folder.

Weather and flight measurements are shifted forward one hour before they are joined to crash records. Thus each crash uses the **previous completed hour**, rather than information accumulated later in the same hour. NOAA timestamps are treated as local standard time and converted to Chicago wall-clock time by default; confirm this matches the export before interpreting results (`NOAA_TIMES_ARE_LOCAL_STANDARD` in [crash_severity/config.py](./crash_severity/config.py)).

Every preprocessing run writes `output/input_manifest.json`, containing file sizes and SHA-256 hashes for the exact source snapshot and analysis code, plus the Chicago dataset ID and analysis dates. `output/run_config.json` records Python, dependency versions, feature sets, model choices, split dates, and run settings. Preserve both files with shared results. Chicago's API can revise historical records, so matching the manifest is necessary to reproduce a particular result exactly.

## Analysis design

- **Target:** `severe = 1` for the exact categories `FATAL` or `INCAPACITATING INJURY`; otherwise 0. A missing injury label is currently counted as 0 (the count is printed during preprocessing); consider a sensitivity analysis excluding unknown labels. Outcome and injury-count fields are excluded from model inputs.
- **Time split:** train through 2022, validation in 2023, held-out test in 2024. Override the dates in [crash_severity/config.py](./crash_severity/config.py) or with `CRASH_TRAIN_END`, `CRASH_VAL_END`, etc.
- **Screening:** point-biserial correlation for numeric predictors and Cramér's V for categorical predictors against crash severity, using training years only. Hourly crash-count associations are reported separately. These are exploratory screens, not causal effects; p-values are unadjusted.
- **Models:** prior baseline, logistic regression, random forest, scikit-learn histogram gradient boosting, plus installed optional boosting and neural-net models.
- **Feature-set comparison:** base crash-time context; base plus NOAA; base plus BTS; base plus both; a no-location robustness set; and a separate post-crash-details set. The post-crash set is not a valid prospective prediction and must not be ranked against context-only sets as if it were.
- **Selection and evaluation:** hyperparameters and the best context model are selected using train/validation data. Average precision (PR-AUC) is primary for the imbalanced target. Paired bootstrap intervals compare each feature set with `base` on the held-out test rows. ROC-AUC, recall at a fixed top-score budget, and Brier score are supplemental.
- **Interpretation:** importance is permutation-based on validation data. It measures predictive contribution in this fitted model, not causation.

## Outputs

All generated files go under `output/` (ignored for new files; legacy tracked outputs may show changes when regenerated):

| Output | Purpose |
|---|---|
| `input_manifest.json`, `preprocess_log.csv` | Input snapshot hashes and preprocessing row counts |
| `cleaned_crashes.csv.gz`, `hourly_crash_weather_flights.csv` | Joined crash table and hourly crash-count panel |
| `severity_numeric_associations.csv`, `severity_categorical_associations.csv`, `hourly_crash_count_associations.csv` | Training-period association screens |
| `model_results.csv`, `ablation_vs_base.csv`, `split_summary.csv`, `best_model.json`, `best_params.json`, `run_config.json` | Model comparison, uncertainty, selected configuration, and run metadata |
| `perm_importance_features.csv`, `perm_importance_groups.csv` | Validation-set model interpretation |
| `risk_cells.csv`, `risk_validation.csv` | Optional spatial ranking and held-out validation |
| `figures/fig1...fig8` (`.pdf`, `.png`), `figures/captions.md` | Publication-ready figures and generated captions |

Use `ablation_vs_base.csv` to assess whether weather or flights improve test PR-AUC beyond the base set; a positive paired-bootstrap interval is evidence of improvement in this held-out period, not proof of a general effect. Use the `tier` column to keep post-crash details separate.

## Project map

| File | Responsibility |
|---|---|
| `crash_severity/pipeline.py` | One-command orchestration |
| `crash_severity/config.py` | Data paths, date splits, geographic settings, seed |
| `crash_severity/preprocessing.py` | Crash retrieval/cleaning, source integration, input manifest, exploratory screening |
| `crash_severity/features.py` | Feature engineering, feature sets, leakage checks |
| `crash_severity/models.py`, `crash_severity/evaluation.py` | Model implementations and metrics/uncertainty |
| `crash_severity/experiments.py`, `crash_severity/interpretation.py`, `crash_severity/risk_map.py` | Model comparison, interpretation, optional spatial evaluation |
| `crash_severity/figures.py` | Paper figures and captions |
| `crash_severity/count_model.py` | Separate hourly crash-count analysis |
| `crash_severity/legacy/` | Superseded severity baseline retained for reference |
| `requirements.txt`, `requirements-optional.txt` | Core and optional model dependencies |
| `tests/test_pipeline.py` | Focused checks for labels, leakage, data readers, time handling, and model behavior |

The main workflow is organized by responsibility inside `crash_severity/`. To run the separate hourly crash-count analysis after preprocessing, use `python -m crash_severity.count_model`.
