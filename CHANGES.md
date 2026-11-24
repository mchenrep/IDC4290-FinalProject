# What changed from the previous version (and why)

1. **Fixed the `severe` label (blocking bug).** The old regex `FATAL|INCAPACITATING` also matched `NONINCAPACITATING INJURY`,
   so 9,210 non-incapacitating-injury crashes in 2019 were wrongly included, inflating the severe share from the true 1.8% to 9.6%. Now an exact category match,
   cross-checked against the injury counts (0 disagreements on 2019), with a regression test.
   **All previous model numbers are invalid** and must be re-run.
2. **`first_crash_type` moved out of "scene_only".** It describes the collision and is only known after the crash. It now lives in
   the separate `context+details` tier together with `damage` and `prim_contributory_cause`; the main comparison is context-only.
3. **Full date range and a real split.** `config.py` defaults to 2019-2024 (train 2019-2022, validation 2023, test 2024); the old
   script was hard-coded to 2019 with a two-month test period. Crash months are now actually cached in `data/crash_cache/`.
4. **Weather and flights are joined to each crash** (not just to the hourly panel), with `has_weather` / `has_flights` flags,
   trace-precipitation handling, rain/snow/fog flags, and a NOAA standard-time to clock-time conversion.
   Models x feature sets are compared with **paired bootstrap** CIs in `ablation_vs_base.csv`.
5. **Hourly panel kept** as a separate crash-*count* analysis; EDA screening now uses training years only.
6. **More models:** random forest, XGBoost, LightGBM, CatBoost, neural net (PyTorch, or scikit-learn fallback) plus dummy and
   logistic baselines; random-search tuning on validation PR-AUC, seed averaging.
7. **Calibration:** isotonic, fit on the validation period; Brier before/after is reported.
8. **Risk map** (`risk_map.py`) validated on the held-out year against naive baselines; **8 figures** with auto-captions.
9. **Housekeeping:** `requirements.txt`, `.gitignore` (data and outputs are no longer committed), tests, README rewritten,
   `model_severity.py` replaced by `run_experiments.py`.
