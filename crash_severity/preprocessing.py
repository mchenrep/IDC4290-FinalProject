"""Prepare crash and external datasets, then write exploratory association screens.

Inputs (all optional except the crash download):
  data/noaa/*.csv   NOAA hourly Local Climatological Data (LCD) CSVs
  data/bts/*.csv    BTS On-Time Performance CSVs (monthly exports)
Crash months are downloaded from the City API once and cached in data/crash_cache/.

Outputs go to output/ (see README).
"""
from datetime import timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency, pointbiserialr, spearmanr

from .config import (BTS_DIR, CACHE_DIR, END_EXCLUSIVE, NOAA_DIR, NOAA_TIMES_ARE_LOCAL_STANDARD, OUT_DIR, START,
                     TRAIN_END)
from .features import OUTCOME_COLS, add_location_features, add_time_features

CRASH_API = "https://data.cityofchicago.org/resource/85ca-t3if.json"
PAGE_SIZE = 50_000

CRASH_FIELDS = [
    "crash_record_id", "crash_date", "posted_speed_limit", "weather_condition",
    "lighting_condition", "roadway_surface_cond", "trafficway_type", "alignment",
    "first_crash_type", "most_severe_injury", "injuries_total", "injuries_fatal",
    "injuries_incapacitating", "injuries_non_incapacitating",
    "injuries_reported_not_evident", "injuries_no_indication", "damage",
    "prim_contributory_cause", "traffic_control_device", "latitude", "longitude", "beat_of_occurrence",
]
CATEGORICAL_TO_CLEAN = ["weather_condition", "lighting_condition", "roadway_surface_cond", "trafficway_type",
                        "alignment", "first_crash_type", "damage", "prim_contributory_cause", "traffic_control_device"]
# Exact injury categories. NOTE: do NOT use a substring match on "INCAPACITATING":
# "NONINCAPACITATING INJURY" contains it, which wrongly labeled ~9% of crashes as severe in the first version.
SEVERE_LEVELS = {"FATAL", "INCAPACITATING INJURY"}
ANY_INJURY_LEVELS = SEVERE_LEVELS | {"NONINCAPACITATING INJURY", "REPORTED, NOT EVIDENT"}

pre_log = []


def log(step, df, note=""):
    pre_log.append({"step": step, "rows": len(df), "note": note})
    print(f"[{step}] rows={len(df):,} {note}")


# ---------------------------------------------------------------- crashes: download (with cache)
def _month_cache_path(month_start):
    return CACHE_DIR / f"crashes_{month_start:%Y_%m}.csv"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_input_manifest():
    """Record source-file hashes so a later run can identify the exact input snapshot."""
    sources = []
    for label, folder in (("chicago_crashes", CACHE_DIR), ("noaa_weather", NOAA_DIR), ("bts_flights", BTS_DIR)):
        files = []
        for path in sorted(Path(folder).glob("*.csv")):
            try:
                display_path = str(path.relative_to(Path.cwd()))
            except ValueError:
                display_path = str(path.resolve())
            files.append({"path": display_path, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        sources.append({"dataset": label, "files": files})
    root = Path(__file__).resolve().parents[1]
    code_files = ("crash_severity/__init__.py", "crash_severity/__main__.py", "crash_severity/config.py",
                  "crash_severity/features.py", "crash_severity/preprocessing.py",
                  "crash_severity/models.py", "crash_severity/evaluation.py", "crash_severity/experiments.py",
                  "crash_severity/interpretation.py", "crash_severity/risk_map.py", "crash_severity/figures.py",
                  "crash_severity/pipeline.py", "crash_severity/count_model.py", "run_pipeline.py",
                  "crash_severity/legacy/__init__.py", "crash_severity/legacy/model_severity.py",
                  "requirements.txt", "requirements-optional.txt", "tests/test_pipeline.py")
    code = {name: file_sha256(root / name) for name in code_files if (root / name).is_file()}
    manifest = {
        "chicago_crashes_dataset_id": "85ca-t3if",
        "chicago_crashes_url": "https://data.cityofchicago.org/resource/85ca-t3if.json",
        "date_range": {"start_inclusive": START, "end_exclusive": END_EXCLUSIVE},
        "code_sha256": code,
        "sources": sources,
    }
    (OUT_DIR / "input_manifest.json").write_text(json.dumps(manifest, indent=2))


def _fetch_month(month_start, month_end):
    where = (f"crash_date >= '{month_start:%Y-%m-%dT%H:%M:%S}' "
             f"AND crash_date < '{month_end:%Y-%m-%dT%H:%M:%S}'")
    frames, offset = [], 0
    while True:
        params = {"$select": ",".join(CRASH_FIELDS), "$where": where,
                  "$order": "crash_record_id", "$limit": PAGE_SIZE, "$offset": offset}
        response = requests.get(CRASH_API, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        offset += len(batch)
        if len(batch) < PAGE_SIZE:
            break
    if not frames:
        return pd.DataFrame(columns=CRASH_FIELDS)
    return pd.concat(frames, ignore_index=True).reindex(columns=CRASH_FIELDS)


def download_crashes(start=START, end_exclusive=END_EXCLUSIVE):
    """Fetch crash rows month by month. Finished months are cached on disk, so re-runs are fast and offline."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    frames = []
    for month_start in pd.date_range(start_ts, end_ts, freq="MS"):
        if month_start >= end_ts:
            break
        month_end = min(month_start + pd.offsets.MonthBegin(1), end_ts)
        cache = _month_cache_path(month_start)
        if cache.exists() and set(CRASH_FIELDS) <= set(pd.read_csv(cache, nrows=0).columns):
            frames.append(pd.read_csv(cache, low_memory=False))
            print(f"Cached  {month_start:%Y-%m}")
            continue
        month = _fetch_month(month_start, month_end)
        if month_end <= pd.Timestamp.now().normalize():      # only cache months that are complete
            month.to_csv(cache, index=False)
        frames.append(month)
        print(f"Fetched {month_start:%Y-%m}")
    if not frames:
        raise RuntimeError("No crash records returned. Check the date range/API connection.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- crashes: clean
def clean_cat(s):
    return s.astype("string").str.strip().str.upper().fillna("UNKNOWN").replace({"": "UNKNOWN"})


def build_severity(df):
    """Target = fatal or incapacitating injury (exact category match). Also adds a 3-level class and an any-injury flag."""
    injury = df["most_severe_injury"].astype("string").str.strip().str.upper()
    df["severe"] = injury.isin(SEVERE_LEVELS).astype("int8")
    df["any_injury"] = injury.isin(ANY_INJURY_LEVELS).astype("int8")
    df["severity_class"] = np.select([df["severe"] == 1, df["any_injury"] == 1], [2, 1], default=0).astype("int8")
    # cross-check against the injury counts: both definitions should agree
    fatal = pd.to_numeric(df.get("injuries_fatal"), errors="coerce").fillna(0)
    incap = pd.to_numeric(df.get("injuries_incapacitating"), errors="coerce").fillna(0)
    mismatch = int(((fatal + incap > 0) != (df["severe"] == 1)).sum())
    print(f"Severity check: label vs injury counts disagree on {mismatch:,} rows "
          f"({mismatch / max(len(df), 1):.3%}); missing most_severe_injury on {int(injury.isna().sum()):,} rows (treated as not severe)")
    return df


def clean_crashes(raw):
    df = raw.copy()
    log("1. crashes loaded", df)
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df = df.dropna(subset=["crash_date"])
    n = len(df)
    df = df.drop_duplicates(subset=["crash_record_id"])
    log("2. duplicates removed", df, f"({n - len(df):,} removed)")
    df = df[df["crash_date"].between(START, pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(seconds=1))].copy()
    log("3. date range applied", df, f"({START} to {END_EXCLUSIVE})")
    df["hour"] = df["crash_date"].dt.floor("h")
    df = build_severity(df)
    for col in ["posted_speed_limit", "injuries_total"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_speed = ~df["posted_speed_limit"].between(5, 70)
    print(f"{int(bad_speed.sum()):,} impossible speed limits set to missing")
    df.loc[bad_speed, "posted_speed_limit"] = np.nan
    for col in CATEGORICAL_TO_CLEAN:
        df[col] = clean_cat(df[col])
    df = add_time_features(df)
    df = add_location_features(df)
    log("4. features built", df, f"({int(df['latitude'].isna().sum()):,} rows have missing/invalid coordinates, kept)")
    return df.sort_values("crash_date").reset_index(drop=True)


# ---------------------------------------------------------------- weather and flights
def first_existing(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def noaa_number(series):
    """NOAA numbers can carry letters ('0.05s'); 'T' means trace precipitation."""
    s = series.astype("string").str.strip().replace({"T": "0.005"})
    return pd.to_numeric(s.str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def lst_to_clock(ts):
    """Treat naive timestamps as Central STANDARD time (UTC-6 all year) and return Chicago wall-clock time."""
    aware = ts.dt.tz_localize(timezone(timedelta(hours=-6)))
    return aware.dt.tz_convert("America/Chicago").dt.tz_localize(None)


def read_noaa(folder=NOAA_DIR, to_clock=NOAA_TIMES_ARE_LOCAL_STANDARD):
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        print("No NOAA CSVs found; continuing without weather features.")
        return None
    parts = []
    for file in files:
        d = pd.read_csv(file, low_memory=False)
        date_col = first_existing(d.columns, ["DATE", "date", "observation_time"])
        if date_col is None:
            raise ValueError(f"No DATE column found in NOAA file: {file}")
        ts = pd.to_datetime(d[date_col], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        hour = ts.dt.floor("h")
        out = pd.DataFrame({"hour": lst_to_clock(hour) if to_clock else hour})
        mapping = {"temp_f": ["HourlyDryBulbTemperature"], "precip_in": ["HourlyPrecipitation"],
                   "wind_mph": ["HourlyWindSpeed"], "visibility_mi": ["HourlyVisibility"]}
        for name, candidates in mapping.items():
            col = first_existing(d.columns, candidates)
            if col:
                out[name] = noaa_number(d[col])
        present = first_existing(d.columns, ["HourlyPresentWeatherType"])
        if present:
            t = d[present].astype("string").str.upper().fillna("")
            out["wx_rain"] = t.str.contains(r"RA|DZ").astype(int)
            out["wx_snow"] = t.str.contains(r"SN|SG|PL|IC").astype(int)
            out["wx_fog"] = t.str.contains(r"FG|BR").astype(int)
            out["wx_freezing"] = t.str.contains(r"FZ").astype(int)
        parts.append(out)
    wx = pd.concat(parts, ignore_index=True).dropna(subset=["hour"])
    wx = wx[(wx["hour"] >= pd.Timestamp(START)) & (wx["hour"] < pd.Timestamp(END_EXCLUSIVE))]
    flags = [c for c in wx.columns if c.startswith("wx_")]
    numeric = [c for c in ["temp_f", "precip_in", "wind_mph", "visibility_mi"] if c in wx]
    agg = {**{c: "mean" for c in numeric}, **{c: "max" for c in flags}}   # several readings/stations per hour
    return wx.groupby("hour", as_index=False).agg(agg)


def read_bts(folder=BTS_DIR):
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        print("No BTS CSVs found; continuing without flight features.")
        return None
    parts = []
    for file in files:
        columns = pd.read_csv(file, nrows=0).columns
        mapping = {
            "origin": first_existing(columns, ["Origin", "ORIGIN"]),
            "flight_date": first_existing(columns, ["FlightDate", "FL_DATE", "Flight Date"]),
            "scheduled": first_existing(columns, ["CRSDepTime", "CRS_DEP_TIME", "Scheduled departure time"]),
            "delay": first_existing(columns, ["DepDelay", "DEP_DELAY", "Departure delay"]),
            "cancelled": first_existing(columns, ["Cancelled", "CANCELLED", "Cancelled flag"]),
        }
        if not all(mapping[k] for k in ("origin", "flight_date", "scheduled")):
            raise ValueError(f"BTS file needs origin, flight date, and scheduled departure-time columns: {file}")
        selected = [column for column in mapping.values() if column is not None]
        part = pd.read_csv(file, usecols=selected, low_memory=False).rename(
            columns={column: name for name, column in mapping.items() if column is not None})
        for name in ("delay", "cancelled"):
            if name not in part:
                part[name] = np.nan
        part = part[part["origin"].astype(str).str.upper().eq("ORD")]   # filter per file: files cover every US airport
        parts.append(part[["origin", "flight_date", "scheduled", "delay", "cancelled"]])

    flights = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["origin", "flight_date", "scheduled", "delay", "cancelled"])
    flights = flights[flights["origin"].astype(str).str.upper().eq("ORD")].copy()
    dates = pd.to_datetime(flights["flight_date"], format="mixed", errors="coerce")
    hhmm = pd.to_numeric(flights["scheduled"], errors="coerce").fillna(-1).astype(int)
    valid = (hhmm // 100).between(0, 24) & (hhmm % 100).between(0, 59) & dates.notna()
    flights, dates, hhmm = flights.loc[valid].copy(), dates.loc[valid], hhmm.loc[valid]
    flights["hour"] = (dates.dt.normalize() + pd.to_timedelta(hhmm // 100, unit="h")).dt.floor("h")   # 2400 -> next-day 00:00
    flights = flights[(flights["hour"] >= pd.Timestamp(START)) & (flights["hour"] < pd.Timestamp(END_EXCLUSIVE))]
    if flights.empty:
        print("No ORD departures in the configured date range; continuing without flight features.")
        return None
    flights["dep_delay_min"] = pd.to_numeric(flights["delay"], errors="coerce")
    flights["cancelled_flag"] = pd.to_numeric(flights["cancelled"], errors="coerce").fillna(0)
    hourly = flights.groupby("hour").agg(flight_departures=("origin", "size"),
                                         mean_dep_delay_min=("dep_delay_min", "mean"),
                                         cancelled_flights=("cancelled_flag", "sum"))
    # hours inside the covered period with no departures are real zeros (not missing)
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full)
    hourly[["flight_departures", "cancelled_flights"]] = hourly[["flight_departures", "cancelled_flights"]].fillna(0)
    return hourly.rename_axis("hour").reset_index()


def lag_hourly_features(external, hours=1):
    """Make a completed hourly observation available only in the following hour."""
    if external is None:
        return None
    lagged = external.copy()
    lagged["hour"] += pd.Timedelta(hours=hours)
    return lagged


def attach_external(crashes, weather, flights):
    """Join the previous complete hour of weather and flight data to each crash."""
    n = len(crashes)
    if weather is not None:
        weather = lag_hourly_features(weather)
        crashes = crashes.merge(weather, on="hour", how="left")
        cols = [c for c in ["temp_f", "precip_in", "wind_mph", "visibility_mi"] if c in crashes]
        crashes["has_weather"] = crashes[cols].notna().any(axis=1).astype(int)
        print(f"Weather matched for {crashes['has_weather'].mean():.1%} of crashes")
    if flights is not None:
        flights = lag_hourly_features(flights)
        lo, hi = flights["hour"].min(), flights["hour"].max()
        crashes = crashes.merge(flights, on="hour", how="left")
        crashes["has_flights"] = crashes["hour"].between(lo, hi).astype(int)
        print(f"Flight data covers {lo:%Y-%m-%d} to {hi:%Y-%m-%d}; {crashes['has_flights'].mean():.1%} of crashes fall inside")
    assert len(crashes) == n, "an external join changed the number of crashes"
    return crashes


# ---------------------------------------------------------------- EDA
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denom)) if denom else np.nan


def make_hourly_panel(crashes, weather, flights):
    """One row per hour, including zero-crash hours, so outside data can be tested against crash COUNTS."""
    hours = pd.date_range(START, pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(hours=1), freq="h")
    counts = crashes.groupby("hour").size().rename("crash_count").reindex(hours, fill_value=0)
    panel = counts.rename_axis("hour").reset_index()
    for external in [weather, flights]:
        if external is not None:
            panel = panel.merge(lag_hourly_features(external), on="hour", how="left")
    return panel


EDA_SKIP = OUTCOME_COLS | {"crash_record_id", "crash_date", "hour", "grid_id", "block_id", "year", "beat_of_occurrence",
                           "has_weather", "has_flights"}


def write_eda(crashes, panel):
    """Screening associations use TRAINING years only, so the validation/test years stay untouched."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(TRAIN_END)
    screen = crashes[crashes["crash_date"] < cut] if (crashes["crash_date"] < cut).any() else crashes
    screen_panel = panel[panel["hour"] < cut] if (panel["hour"] < cut).any() else panel
    print(f"EDA screening uses {len(screen):,} crashes before {cut:%Y-%m-%d}")
    numeric_rows, categorical_rows = [], []
    for col in screen.columns:
        if col in EDA_SKIP:
            continue
        if pd.api.types.is_numeric_dtype(screen[col]):
            valid = screen[[col, "severe"]].dropna()
            if valid[col].nunique() > 1 and valid["severe"].nunique() > 1:
                r, p = pointbiserialr(valid["severe"], valid[col])
                numeric_rows.append({"feature": col, "point_biserial_r": r, "p_value": p, "n": len(valid)})
        elif screen[col].nunique(dropna=True) <= 100:
            valid = screen[[col, "severe"]].dropna()
            if len(valid) and valid[col].nunique() > 1:
                categorical_rows.append({"feature": col, "cramers_v": cramers_v(valid[col], valid["severe"]), "n": len(valid)})
                (screen.groupby(col, dropna=False)["severe"].agg(["count", "mean"])
                 .rename(columns={"mean": "severe_rate"}).to_csv(OUT_DIR / f"severity_by_{col}.csv"))
    if numeric_rows:
        (pd.DataFrame(numeric_rows).sort_values("point_biserial_r", key=lambda s: s.abs(), ascending=False)
         .to_csv(OUT_DIR / "severity_numeric_associations.csv", index=False))
    if categorical_rows:
        (pd.DataFrame(categorical_rows).sort_values("cramers_v", ascending=False)
         .to_csv(OUT_DIR / "severity_categorical_associations.csv", index=False))
    count_rows = []
    for col in screen_panel.select_dtypes(include="number").columns:
        if col == "crash_count":
            continue
        valid = screen_panel[[col, "crash_count"]].dropna()
        if len(valid) > 2 and valid[col].nunique() > 1:
            r, p = spearmanr(valid[col], valid["crash_count"])
            count_rows.append({"feature": col, "spearman_r": r, "p_value": p, "n_hours": len(valid)})
    if count_rows:
        (pd.DataFrame(count_rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
         .to_csv(OUT_DIR / "hourly_crash_count_associations.csv", index=False))
    else:
        print("No hourly crash-count associations computed. Add NOAA/BTS CSVs to data/noaa and data/bts to enable this.")
    panel.to_csv(OUT_DIR / "hourly_crash_weather_flights.csv", index=False)
    crashes.groupby(["year", "hour_of_day"]).size().rename("crash_count").to_csv(OUT_DIR / "crashes_by_year_hour.csv")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_crashes = download_crashes()
    write_input_manifest()
    crashes = clean_crashes(raw_crashes)
    weather, flights = read_noaa(), read_bts()
    panel = make_hourly_panel(crashes, weather, flights)
    crashes = attach_external(crashes, weather, flights)
    log("5. external data attached", crashes)
    write_eda(crashes, panel)
    crashes.to_csv(OUT_DIR / "cleaned_crashes.csv.gz", index=False)
    pd.DataFrame(pre_log).to_csv(OUT_DIR / "preprocess_log.csv", index=False)
    print(f"Crashes: {len(crashes):,}; severe (fatal/incapacitating) share: {crashes['severe'].mean():.3%}; "
          f"any injury share: {crashes['any_injury'].mean():.3%}")
    print(f"Wrote {OUT_DIR.resolve()}")
    print("Note: crash-level severity associations and hourly crash-count associations answer different questions.")


if __name__ == "__main__":
    main()
