"""Feature engineering helpers and the feature-set definitions used by the experiments.

Every column is in exactly one role:
  CONTEXT  - known before or at the moment of the crash (time, road, scene, weather, flights, location)
  DETAIL   - only known after the crash happened (crash type, damage, contributory cause). Can leak the outcome.
  OUTCOME  - injury fields and the target itself. NEVER used as model inputs.
"""
import numpy as np
import pandas as pd

from .config import CHICAGO_LAT, CHICAGO_LON, GRID_SIZE_DEG, BLOCK_SIZE_DEG, ORD_LAT, ORD_LON

# ---------------- Roles ----------------
OUTCOME_COLS = {
    "severe", "severity_class", "any_injury", "most_severe_injury", "injuries_total", "injuries_fatal",
    "injuries_incapacitating", "injuries_non_incapacitating", "injuries_reported_not_evident", "injuries_no_indication",
}
DETAIL_CAT = ["first_crash_type", "damage", "prim_contributory_cause"]

ROAD_CAT = ["weather_condition", "lighting_condition", "roadway_surface_cond", "trafficway_type",
            "alignment", "traffic_control_device"]
TIME_CAT = ["season"]
TIME_NUM = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_rush_hour"]
ROAD_NUM = ["posted_speed_limit"]
LOCATION_NUM = ["latitude", "longitude", "dist_to_ord_km"]
WEATHER_NUM = ["temp_f", "precip_in", "wind_mph", "visibility_mi"]
WEATHER_FLAGS = ["wx_rain", "wx_snow", "wx_fog", "wx_freezing"]
FLIGHT_NUM = ["flight_departures", "mean_dep_delay_min", "cancelled_flights"]

BASE_CAT = ROAD_CAT + TIME_CAT
BASE_NUM = TIME_NUM + ROAD_NUM + LOCATION_NUM

# name -> columns and which external source (if any) must have data for it to run
FEATURE_SETS = {
    "base":                 {"cat": BASE_CAT, "num": BASE_NUM, "needs": None, "tier": "context"},
    "base+weather":         {"cat": BASE_CAT, "num": BASE_NUM + WEATHER_NUM + WEATHER_FLAGS, "needs": "weather", "tier": "context"},
    "base+flights":         {"cat": BASE_CAT, "num": BASE_NUM + FLIGHT_NUM, "needs": "flights", "tier": "context"},
    "base+weather+flights": {"cat": BASE_CAT, "num": BASE_NUM + WEATHER_NUM + WEATHER_FLAGS + FLIGHT_NUM,
                             "needs": "both", "tier": "context"},
    # robustness: does the model learn conditions, or just memorize places?
    "base_no_location":     {"cat": BASE_CAT, "num": [c for c in BASE_NUM if c not in LOCATION_NUM],
                             "needs": None, "tier": "context"},
    # post-crash tier: NOT comparable with the context sets (uses information known only after the crash)
    "context+details":      {"cat": BASE_CAT + DETAIL_CAT, "num": BASE_NUM, "needs": None, "tier": "post-crash details"},
}

FEATURE_GROUPS = {   # for grouped permutation importance
    "time": TIME_CAT + TIME_NUM,
    "road/scene": ROAD_CAT + ROAD_NUM,
    "location": LOCATION_NUM,
    "weather (NOAA)": WEATHER_NUM + WEATHER_FLAGS,
    "flights (ORD)": FLIGHT_NUM,
    "crash details": DETAIL_CAT,
}


def add_time_features(df, col="crash_date"):
    d = df[col]
    df["hour_of_day"] = d.dt.hour
    df["day_of_week"] = d.dt.day_name()
    df["month"] = d.dt.month
    df["year"] = d.dt.year
    ang = 2 * np.pi * df["hour_of_day"] / 24            # hour is cyclic: 23:00 is next to 00:00
    df["hour_sin"], df["hour_cos"] = np.sin(ang), np.cos(ang)
    mang = 2 * np.pi * (df["month"] - 1) / 12
    df["month_sin"], df["month_cos"] = np.sin(mang), np.cos(mang)
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    rush = df["hour_of_day"].between(7, 9) | df["hour_of_day"].between(16, 18)
    df["is_rush_hour"] = (rush & (d.dt.dayofweek < 5)).astype(int)
    df["season"] = df["month"].map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                                    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"})
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_location_features(df):
    """Blank out impossible coordinates, then add grid cell, spatial block and distance to O'Hare."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    ok = lat.between(*CHICAGO_LAT) & lon.between(*CHICAGO_LON)
    df["latitude"], df["longitude"] = lat.where(ok), lon.where(ok)
    row = np.floor((df["latitude"] - CHICAGO_LAT[0]) / GRID_SIZE_DEG)
    col = np.floor((df["longitude"] - CHICAGO_LON[0]) / GRID_SIZE_DEG)
    df["grid_id"] = np.where(ok, "r" + row.astype("Int64").astype(str) + "_c" + col.astype("Int64").astype(str), None)
    brow = np.floor((df["latitude"] - CHICAGO_LAT[0]) / BLOCK_SIZE_DEG)
    bcol = np.floor((df["longitude"] - CHICAGO_LON[0]) / BLOCK_SIZE_DEG)
    df["block_id"] = np.where(ok, "b" + brow.astype("Int64").astype(str) + "_" + bcol.astype("Int64").astype(str), None)
    df["dist_to_ord_km"] = haversine_km(df["latitude"], df["longitude"], ORD_LAT, ORD_LON)
    return df


def resolve_feature_set(name, df):
    """Return (cat, num) restricted to columns that exist and have data, or None if a required source is missing."""
    spec = FEATURE_SETS[name]
    need = spec["needs"]
    has_wx = "has_weather" in df and df["has_weather"].mean() > 0.2
    has_fl = "has_flights" in df and df["has_flights"].mean() > 0.2
    if need == "weather" and not has_wx:
        return None
    if need == "flights" and not has_fl:
        return None
    if need == "both" and not (has_wx and has_fl):
        return None
    cat = [c for c in spec["cat"] if c in df and df[c].notna().any()]
    num = [c for c in spec["num"] if c in df and df[c].notna().any()]
    return cat, num


def assert_no_leakage(cat, num, allow_details=False):
    """Raise if an outcome column (or, unless allowed, a post-crash detail column) is used as an input."""
    used = set(cat) | set(num)
    bad = used & OUTCOME_COLS
    if not allow_details:
        bad |= used & set(DETAIL_CAT)
    if bad:
        raise ValueError(f"Leakage guard: these columns must not be model inputs here: {sorted(bad)}")
