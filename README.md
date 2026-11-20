# Chicago Traffic Crash Data: Preprocessing, EDA, and Modeling

This repository contains a starter pipeline for **Steps 1–3** of the project:

1. Download and clean Chicago traffic-crash records, then optionally combine them with hourly weather and flight data.
2. Explore associations with crash occurrence and severe crashes to help decide which features or datasets to carry into modeling.
3. Train and evaluate models that predict whether a crash is severe

The pipeline covers **January 1, 2019 through December 31, 2024**. Crash data is downloaded directly from the City of Chicago data API. NOAA weather and BTS flight data are optional local CSV inputs.

> **Important:** Associations in the EDA are screening results, not evidence of causation. The two outcomes are different: crash-level analyses examine the severe-injury indicator, while the hourly panel examines crash counts, including hours with zero crashes.

## Requirements

- Python 3.9 or newer
- Internet access when downloading crash data from the City API
- CSV exports for any optional NOAA or BTS datasets you want to include

## Repository setup

Clone the repository, then place the script and data in this layout:

```text
repository/
├── crash_preprocess_eda.py
|-- model_severity.py
├── data/
│   ├── noaa/       # NOAA hourly LCD CSV files (optional)
│   └── bts/        # BTS monthly On-Time Performance CSV files (optional)
└── output/         # Created by the script
```

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install packages and run the pipeline:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy requests
python crash_preprocess_eda.py
python model_severity.py
```

Run the command from the repository root so the script finds `data/` and writes to `output/` as expected.

## Input data

### Chicago traffic crashes — downloaded automatically

The script queries the City of Chicago **Traffic Crashes – Crashes** dataset (`85ca-t3if`) month by month for 2019–2024. You do **not** need to download separate crash-date files. A working internet connection is required. The City dataset may be updated, so rerunning the script later can produce different results.

### NOAA hourly weather — optional

To include weather features, download NOAA **Local Climatological Data (LCD)** hourly CSV data for the Chicago period **January 2019 through December 2024**. Start with Chicago O’Hare; Midway can also be included. Put each downloaded CSV in `data/noaa/`.

The script looks for a `DATE` column and these LCD fields when available:

- `HourlyDryBulbTemperature`
- `HourlyPrecipitation`
- `HourlyWindSpeed`
- `HourlyVisibility`
- `HourlyPresentWeatherType` or `HourlySkyConditions`

NOAA exports can vary by product/version. If a file uses different headers or timestamp formats, adjust the field mapping or parser in `read_noaa()` before running. If multiple station files are supplied, numeric weather readings are averaged by hour; weather text uses the most common value.

### BTS flight data — optional

To include flight features, download BTS **On-Time Performance** monthly CSV exports for **January 2019 through December 2024** and place them in `data/bts/`. Include these columns in each export:

- Origin airport (`Origin`)
- Flight date (`FlightDate`)
- Scheduled departure time (`CRSDepTime`)
- Departure delay (`DepDelay`, optional)
- Cancellation flag (`Cancelled`, optional)

The script filters to flights with origin **ORD** and aggregates scheduled departures by hour. It does not currently include arrivals or flights from Midway. If BTS exports use different column names, update the candidate names in `read_bts()`.

If the NOAA or BTS folders are empty or missing, the script prints a notice and continues without that dataset. Do not put unrelated CSV files in those folders; the script attempts to read every CSV there.

## Outputs

The script creates `output/` and writes:

- `cleaned_crashes.csv` — cleaned crash-level records, derived time fields, and a `severe` indicator.
- `hourly_crash_weather_flights.csv` — one row per hour, including zero-crash hours, crash counts, and any available NOAA/BTS features.
- `severity_numeric_associations.csv` — point-biserial associations between numeric features and `severe`.
- `severity_categorical_associations.csv` — Cramér’s V associations between categorical features and `severe`.
- `severity_by_<field>.csv` — severity counts and rates by eligible categorical field.
- `hourly_crash_count_associations.csv` — Spearman associations between numeric hourly features and crash counts.
- `crashes_by_year_hour.csv` — crash counts grouped by year and hour of day.

The script excludes injury outcome fields from the severity feature-association tables to reduce target leakage. Review the inputs and features before using the output for model training; association strength alone is not a feature-selection rule.

## Configuration

The date range and folder locations are defined near the top of `crash_preprocess_eda.py`:

```python
START = "2019-01-01"
END_EXCLUSIVE = "2025-01-01"
DATA_DIR = Path("data")
```

To change the date range, update both date constants. `clean_crashes()` and the data readers use these configured bounds.

## Troubleshooting

- **Connection/API error:** Check internet access and retry. The City API is queried in monthly chunks.
- **No NOAA/BTS features in output:** Confirm that the CSV files are in the correct subfolder and that their headers are recognized.
- **Missing-column error:** Check the expected input columns above and update the column mapping in the corresponding reader.
- **Unexpected weather or flight joins:** Confirm that source timestamps represent local Chicago time and inspect the hourly output. The script floors timestamps to the hour; it does not perform timezone conversion.
