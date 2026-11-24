"""Central settings for the whole pipeline.

Every value can be overridden with an environment variable, which is handy for quick tests, e.g.
    CRASH_END=2020-01-01 CRASH_TRAIN_END=2019-07-01 CRASH_VAL_END=2019-10-01 python run_experiments.py --fast
"""
import os
from pathlib import Path

# ---------------- Folders ----------------
DATA_DIR = Path(os.environ.get("CRASH_DATA_DIR", "data"))
OUT_DIR = Path(os.environ.get("CRASH_OUT_DIR", "output"))
NOAA_DIR = DATA_DIR / "noaa"
BTS_DIR = DATA_DIR / "bts"
CACHE_DIR = DATA_DIR / "crash_cache"
FIG_DIR = OUT_DIR / "figures"

# ---------------- Date range and chronological split ----------------
# Crashes from START (inclusive) to END_EXCLUSIVE.
# train: START .. TRAIN_END | validation: TRAIN_END .. VAL_END | test: VAL_END .. END_EXCLUSIVE
START = os.environ.get("CRASH_START", "2019-01-01")
END_EXCLUSIVE = os.environ.get("CRASH_END", "2025-01-01")
TRAIN_END = os.environ.get("CRASH_TRAIN_END", "2023-01-01")
VAL_END = os.environ.get("CRASH_VAL_END", "2024-01-01")

# NOAA Local Climatological Data is (to our knowledge) stamped in local STANDARD time all year,
# while crash and BTS times are local clock time (they shift by an hour in summer).
# Keep True unless you confirm your NOAA export is already in clock time.
NOAA_TIMES_ARE_LOCAL_STANDARD = True

# ---------------- Geography ----------------
CHICAGO_LAT = (41.60, 42.05)      # coordinates outside this box are treated as bad
CHICAGO_LON = (-87.95, -87.50)
GRID_SIZE_DEG = 0.005             # risk-map cell size (about 0.5 km)
BLOCK_SIZE_DEG = 0.02             # coarse blocks (about 2 km) for spatial cross-validation
ORD_LAT, ORD_LON = 41.9742, -87.9073

SEED = 42
