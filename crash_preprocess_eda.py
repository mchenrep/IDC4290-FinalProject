"""Preprocessing + EDA starter for the Chicago crash project.

Suggested local structure:
  data/noaa/*.csv   NOAA hourly station CSVs (one or more years)
  data/bts/*.csv    BTS On-Time Performance monthly CSV exports

Downloads Chicago Traffic Crashes from the City API. Output files go to output/.
"""
from pathlib import Path
import re
import requests
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, pointbiserialr

# Use complete calendar years and exclude the incomplete current year.
START = "2019-01-01"
END_EXCLUSIVE = "2025-01-01"
DATA_DIR = Path("data")
NOAA_DIR = DATA_DIR / "noaa"
BTS_DIR = DATA_DIR / "bts"
OUT_DIR = Path("output")
CACHE_DIR = DATA_DIR / "crash_cache"
FORCE_REFRESH = False
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
LEAKAGE = {
    "most_severe_injury", "injuries_total", "injuries_fatal", "injuries_incapacitating",
    "injuries_non_incapacitating", "injuries_reported_not_evident",
    "injuries_no_indication", "severe",
}


def fetch_month(month_start, month_end):
    """Download one month from the City API (same paging loop as before)."""
    where = (
        f"crash_date >= '{month_start:%Y-%m-%dT%H:%M:%S}' "
        f"AND crash_date < '{month_end:%Y-%m-%dT%H:%M:%S}'"
    )
    frames, offset = [], 0
    while True:
        params = {"$select": ",".join(CRASH_FIELDS), "$where": where,
                  "$limit": PAGE_SIZE, "$offset": offset}
        response = requests.get(CRASH_API, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        offset += len(batch)
        if len(batch) < PAGE_SIZE:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_crashes(start=START, end_exclusive=END_EXCLUSIVE):
    """Load each month from the local cache if present, otherwise download and cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    frames = []
    for month_start in pd.date_range(start_ts, end_ts, freq="MS"):
        if month_start >= end_ts:
            break
        month_end = min(month_start + pd.offsets.MonthBegin(1), end_ts)
        path = CACHE_DIR / f"crashes_{month_start:%Y_%m}.csv"
        if path.exists() and not FORCE_REFRESH:
            df = pd.read_csv(path, dtype=str)
            print(f"Cached  {month_start:%Y-%m}")
        else:
            df = fetch_month(month_start, month_end)
            if not df.empty:
                tmp = path.with_suffix(".tmp")
                df.to_csv(tmp, index=False)
                tmp.replace(path)
            print(f"Fetched {month_start:%Y-%m}")
        frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No crash records returned. Check the date range/API connection.")
    return pd.concat(frames, ignore_index=True)


def clean_crashes(raw):
    df = raw.copy()
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df = df.dropna(subset=["crash_date"])
    df = df.drop_duplicates(subset=["crash_record_id"])
    df = df[df["crash_date"].between(START, pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(seconds=1))]
    df["hour"] = df["crash_date"].dt.floor("h")
    injury = df["most_severe_injury"].fillna("").str.upper()
    df["severe"] = injury.str.contains("FATAL|INCAPACITATING", regex=True).astype("int8")
    df["hour_of_day"] = df["crash_date"].dt.hour
    df["day_of_week"] = df["crash_date"].dt.day_name()
    df["month"] = df["crash_date"].dt.month
    for col in ["latitude", "longitude", "posted_speed_limit", "injuries_total"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def first_existing(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def parse_noaa_hour(value):
    """Parse NOAA LCD DATE values (typically local date + hour/minute)."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    # LCD hourly rows can encode the hour as HH:MM or as a timestamp.
    ts = pd.to_datetime(text, errors="coerce")
    return ts.floor("h") if not pd.isna(ts) else pd.NaT


def read_noaa(folder=NOAA_DIR):
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
        # Etc/GMT+6 is a fixed UTC-6 zone (Chicago standard time, never shifting for DST)
        ts = ts.dt.tz_localize("Etc/GMT+6").dt.tz_convert("America/Chicago").dt.tz_localize(None)
        out = pd.DataFrame({"hour": ts.dt.floor("h")})
        mapping = {
            "temp_f": ["HourlyDryBulbTemperature"],
            "precip_in": ["HourlyPrecipitation"],
            "wind_mph": ["HourlyWindSpeed"],
            "visibility_mi": ["HourlyVisibility"],
        }
        for name, candidates in mapping.items():
            col = first_existing(d.columns, candidates)
            if col:
                # NOAA may encode missing observations with tokens such as T or blank.
                out[name] = pd.to_numeric(d[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        if "temp_f" in out:        
            out["temp_f"] = out["temp_f"] * 9 / 5 + 32
        if "precip_in" in out:     
            out["precip_in"] = out["precip_in"] / 25.4
        if "wind_mph" in out:      
            out["wind_mph"] = out["wind_mph"] * 2.23694
        if "visibility_mi" in out: 
            out["visibility_mi"] = out["visibility_mi"] * 0.621371
        condition_col = first_existing(d.columns, ["HourlyPresentWeatherType", "HourlySkyConditions"])
        if condition_col:
            out["weather_text"] = d[condition_col].astype("string")
        parts.append(out)
    wx = pd.concat(parts, ignore_index=True).dropna(subset=["hour"])
    wx = wx[(wx["hour"] >= pd.Timestamp(START)) & (wx["hour"] < pd.Timestamp(END_EXCLUSIVE))]
    # If multiple station files are supplied, average numeric readings per hour.
    numeric = [c for c in wx.select_dtypes(include="number").columns if c != "hour"]
    agg = {c: "mean" for c in numeric}
    if "weather_text" in wx:
        agg["weather_text"] = lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA
    return wx.groupby("hour", as_index=False).agg(agg)



def read_bts(folder=BTS_DIR):
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        print("No BTS CSVs found; continuing without flight features.")
        return None
    print(f"\nBTS CSV files found: {len(files)}")
    parts = []
    for file in files:
        d = pd.read_csv(file, low_memory=False)
        parts.append(d)
    flights = pd.concat(parts, ignore_index=True)
    origin = first_existing(flights.columns, ["Origin", "ORIGIN"])
    date_col = first_existing(
        flights.columns, ["FlightDate", "FL_DATE", "Flight Date"])
    scheduled = first_existing(
        flights.columns, ["CRSDepTime", "CRS_DEP_TIME",
                          "Scheduled departure time"])
    delay = first_existing(
        flights.columns, ["DepDelay", "DEP_DELAY", "Departure delay"])
    cancelled = first_existing(
        flights.columns, ["Cancelled", "CANCELLED", "Cancelled flag"])
    if not all([origin, date_col, scheduled]):
        raise ValueError(
            "BTS files need Origin, flight date, and scheduled departure-time columns.")
    print("BTS columns:", flights.columns.tolist())
    print("Date column:", date_col)
    print("Scheduled time column:", scheduled)
    print("Origin column:", origin)
    print("Sample flight dates:")
    print(flights[date_col].head(10).to_list())
    # Keep flights originating at Chicago O'Hare.
    flights = flights[
        flights[origin].astype(str).str.strip().str.upper().eq("ORD")].copy()
    print("Flights originating at ORD:", len(flights))
    # Parse the date without assuming a particular format.
    dates = pd.to_datetime(flights[date_col], errors="coerce")
    print("Successfully parsed dates:", dates.notna().sum())
    print("Unparsed dates:", dates.isna().sum())
    # Convert scheduled departure times to numeric HHMM.
    hhmm = pd.to_numeric(flights[scheduled], errors="coerce")
    # Handle 2400 as midnight on the following day.
    midnight_next_day = hhmm.eq(2400)
    hhmm = hhmm.mask(midnight_next_day, 0)
    hhmm = hhmm.fillna(-1).astype(int)
    hour = hhmm // 100
    minute = hhmm % 100
    valid = (hour.between(0, 23) & minute.between(0, 59) & dates.notna())
    print("Flights with valid date/time:", valid.sum())
    flights = flights.loc[valid].copy()
    dates = dates.loc[valid]
    hhmm = hhmm.loc[valid]
    midnight_next_day = midnight_next_day.loc[valid]
    flights["hour"] = (
        dates.dt.normalize()
        + pd.to_timedelta(hhmm // 100, unit="h")
        + pd.to_timedelta(hhmm % 100, unit="m")
        + pd.to_timedelta(midnight_next_day.astype(int), unit="D")
    ).dt.floor("h")
    flights = flights[
        (flights["hour"] >= pd.Timestamp(START))
        & (flights["hour"] < pd.Timestamp(END_EXCLUSIVE))].copy()
    print("Flights remaining in project date range:", len(flights))
    if flights.empty:
        raise ValueError(
            "No valid ORD flights remain. Check the BTS date and scheduled-time formats.")
    if delay:
        flights["dep_delay_min"] = pd.to_numeric(
            flights[delay], errors="coerce")
    if cancelled:
        flights["cancelled_flag"] = pd.to_numeric(
            flights[cancelled], errors="coerce").fillna(0)
    agg = {"flight_departures": (origin, "size")}
    if "dep_delay_min" in flights:
        agg["mean_dep_delay_min"] = ("dep_delay_min", "mean")
    if "cancelled_flag" in flights:
        agg["cancelled_flights"] = ("cancelled_flag", "sum")
    result = flights.groupby("hour", as_index=False).agg(**agg)
    print("Hourly BTS records generated:", len(result))
    print(result.head())
    return result

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denom)) if denom else np.nan


def make_hourly_panel(crashes, weather, flights):
    # Keep zero-crash hours so outside datasets can be tested against crash occurrence.
    hours = pd.date_range(START, pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(hours=1), freq="h")
    counts = crashes.groupby("hour").size().rename("crash_count").reindex(hours, fill_value=0)
    panel = counts.rename_axis("hour").reset_index()
    for external in [weather, flights]:
        if external is not None:
            panel = panel.merge(external, on="hour", how="left")
    print("\nHourly panel flight-data diagnostics:")

    for col in ["flight_departures", "mean_dep_delay_min",
                "cancelled_flights"]:
        if col in panel.columns:
            print(
                f"{col}: {panel[col].notna().sum():,} valid hours "
                f"out of {len(panel):,}"
            )
    return panel


def write_eda(crashes, panel):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Crash-level severity associations. Excludes injury outcome fields to prevent target leakage.
    numeric_rows, categorical_rows = [], []
    for col in crashes.columns:
        if col in LEAKAGE or col in {"crash_record_id", "crash_date", "hour"}:
            continue
        if pd.api.types.is_numeric_dtype(crashes[col]):
            valid = crashes[[col, "severe"]].dropna()
            if valid[col].nunique() > 1 and valid["severe"].nunique() > 1:
                r, p = pointbiserialr(valid["severe"], valid[col])
                numeric_rows.append({"feature": col, "point_biserial_r": r, "p_value": p, "n": len(valid)})
        elif crashes[col].nunique(dropna=True) <= 100:
            valid = crashes[[col, "severe"]].dropna()
            if len(valid) and valid[col].nunique() > 1:
                categorical_rows.append({"feature": col, "cramers_v": cramers_v(valid[col], valid["severe"]), "n": len(valid)})
                crashes.groupby(col, dropna=False)["severe"].agg(["count", "mean"]).rename(columns={"mean": "severe_rate"}).to_csv(OUT_DIR / f"severity_by_{col}.csv")
    if numeric_rows:
        pd.DataFrame(numeric_rows).sort_values("point_biserial_r", key=lambda s: s.abs(), ascending=False).to_csv(OUT_DIR / "severity_numeric_associations.csv", index=False)
    else:
        print("No numeric severity associations computed (no eligible numeric columns).")
    if categorical_rows:
        pd.DataFrame(categorical_rows).sort_values("cramers_v", ascending=False).to_csv(OUT_DIR / "severity_categorical_associations.csv", index=False)
    else:
        print("No categorical severity associations computed (no eligible categorical columns).")

    # Crash occurrence associations use all hours, including hours with zero crashes.
    count_rows = []
    for col in panel.select_dtypes(include="number").columns:
        if col in {"crash_count"}:
            continue
        valid = panel[[col, "crash_count"]].dropna()
        if len(valid) > 2 and valid[col].nunique() > 1:
            r, p = spearmanr(valid[col], valid["crash_count"])
            count_rows.append({"feature": col, "spearman_r": r, "p_value": p, "n_hours": len(valid)})
    if count_rows:
        pd.DataFrame(count_rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).to_csv(OUT_DIR / "hourly_crash_count_associations.csv", index=False)
    else:
        print("No hourly crash-count associations computed — add NOAA/BTS CSVs to data/noaa and data/bts to enable this.")
    panel.to_csv(OUT_DIR / "hourly_crash_weather_flights.csv", index=False)
    # Keep only the columns needed for crash severity modeling.
    cleaned_columns = ["crash_date", "hour", "hour_of_day", "day_of_week", "month", "severe",
    "posted_speed_limit", "latitude", "longitude", "weather_condition", "lighting_condition",
    "roadway_surface_cond", "trafficway_type", "alignment", "traffic_control_device", 
    "first_crash_type", "damage", "prim_contributory_cause"]

    # Check that all requested columns exist.
    missing_columns = [col for col in cleaned_columns if col not in crashes.columns]

    if missing_columns:
        raise ValueError(
        f"Missing columns when saving cleaned crashes: {missing_columns}")

    # Save only the selected columns.
    cleaned_crashes = crashes[cleaned_columns].copy()

    cleaned_crashes.to_csv(OUT_DIR / "cleaned_crashes.csv.gz",index=False, compression="gzip")

    crashes.groupby([crashes["crash_date"].dt.year, 
        crashes["hour_of_day"]]).size().rename("crash_count").to_csv(OUT_DIR / "crashes_by_year_hour.csv")

    print(f"Cleaned crash file contains {len(cleaned_crashes.columns)} columns "
    f"and {len(cleaned_crashes):,} records.")
    print(f"EDA files written to {OUT_DIR.resolve()}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = download_crashes()
    crashes = clean_crashes(raw)
    weather = read_noaa()
    flights = read_bts()
    panel = make_hourly_panel(crashes, weather, flights)
    write_eda(crashes, panel)
    print(f"Crashes: {len(crashes):,}; severe share: {crashes['severe'].mean():.3%}")
    print("Note: crash-level severity associations and hourly crash-count associations answer different questions.")

if __name__ == "__main__":
    main()
