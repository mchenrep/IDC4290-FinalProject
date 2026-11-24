# Chicago Traffic Crash Data: Preprocessing, EDA, and Modeling

This repository contains a data preprocessing, exploratory data analysis (EDA), and machine learning pipeline for studying Chicago traffic crashes from **January 1, 2019 through December 31, 2024**.

The project combines Chicago traffic-crash records with optional NOAA hourly weather data and Bureau of Transportation Statistics (BTS) flight data to investigate patterns associated with traffic crashes and develop predictive models.

The project includes three main stages:

1. **Data preprocessing and EDA:** Download and clean Chicago traffic-crash records, process optional weather and flight datasets, create an hourly crash-count dataset, and explore associations between features and crash outcomes.
2. **Crash severity classification:** Train and evaluate machine learning models to predict whether an individual traffic crash is severe.
3. **Hourly crash-count prediction:** Train and evaluate models to predict the number of traffic crashes occurring during a given hour using historical crash counts, calendar information, weather, and flight activity.

> **Important:** Associations identified during EDA are screening results, not evidence of causation. Crash severity and hourly crash counts are separate prediction problems. Crash-level analyses examine the severe-injury indicator, while the hourly panel examines crash counts, including hours with zero crashes.

---

## Requirements

- Python 3.9 or newer
- Internet access when downloading crash data from the City of Chicago API
- NOAA hourly weather CSV exports (optional)
- BTS On-Time Performance monthly CSV exports (optional)

### Python dependencies

Install the required packages using:

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy requests scikit-learn matplotlib
```

The project uses pandas, NumPy, SciPy, and Requests for data processing and analysis, scikit-learn for machine learning, and Matplotlib for visualizations.

---

## Repository setup

Clone the repository and organize the project using the following directory structure:

```text
IDC4290-FinalProject/
│
├── crash_preprocess_eda.py
├── model_severity.py
├── model_crash_count.py
│
├── data/
│   ├── noaa/       # NOAA hourly LCD CSV files
│   └── bts/        # BTS monthly On-Time Performance CSV files
│
└── output/         # Generated datasets, results, and plots
```

The `data/` folders contain optional external datasets. The `output/` folder is created automatically when the preprocessing script runs.

### Creating a virtual environment

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy requests scikit-learn matplotlib
```

### Running the project

Run the scripts from the repository root in the following order:

```powershell
python crash_preprocess_eda.py
python model_severity.py
python model_crash_count.py
```

The preprocessing script should be run before either modeling script because both models rely on datasets generated in the `output/` directory.

---

## Input data

### Chicago traffic crashes — downloaded automatically

The preprocessing script queries the City of Chicago **Traffic Crashes – Crashes** dataset (`85ca-t3if`) month by month for the period 2019–2024.

Separate crash-date CSV downloads are not required. A working internet connection is needed when retrieving the data from the City API.

The dataset is cleaned and filtered to the configured project date range. The preprocessing script also derives time-related features and creates a binary `severe` indicator for crash severity classification.

The City dataset may be updated over time, so rerunning the script at a later date may produce slightly different results.

### NOAA hourly weather — optional

To include weather features, download NOAA **Local Climatological Data (LCD)** hourly CSV data for Chicago covering January 2019 through December 2024.

Start with Chicago O'Hare International Airport. Chicago Midway data may also be included.

Place downloaded CSV files in:

```text
data/noaa/
```

The script looks for a `DATE` column and the following LCD fields when available:

- `HourlyDryBulbTemperature`
- `HourlyPrecipitation`
- `HourlyWindSpeed`
- `HourlyVisibility`
- `HourlyPresentWeatherType` or `HourlySkyConditions`

NOAA exports can vary by product and version. If a file uses different headers or timestamp formats, adjust the field mapping or parser in `read_noaa()` before running the script.

When multiple station files are supplied, numeric weather readings are averaged by hour, while weather text uses the most common value.

### BTS flight data — optional

To include flight activity, download BTS **On-Time Performance** monthly CSV exports for January 2019 through December 2024.

Place all 72 monthly CSV files in:

```text
data/bts/
```

The current preprocessing implementation recognizes the following BTS columns:

- `FL_DATE` — Flight date
- `ORIGIN` — Origin airport
- `CRS_DEP_TIME` — Scheduled departure time
- `DEP_DELAY` — Departure delay
- `CANCELLED` — Cancellation indicator

The script filters flights to those originating from **Chicago O'Hare International Airport (ORD)**.

It parses the flight dates and scheduled departure times, assigns flights to hourly intervals, and calculates:

- `flight_departures` — Number of scheduled departures during the hour.
- `mean_dep_delay_min` — Average departure delay in minutes during the hour.
- `cancelled_flights` — Number of cancelled flights during the hour.

The current implementation processes 72 monthly files covering the full project period. It does not currently include arrivals or flights originating from Chicago Midway.

If BTS exports use different column names, update the candidate column names in `read_bts()`.

### Missing external datasets

NOAA and BTS data are optional. If either input folder is empty or missing, the preprocessing script prints a notice and continues without that dataset.

Do not place unrelated CSV files in these folders because the script attempts to read every CSV file it finds.

---

## Data preprocessing and EDA

The preprocessing script is:

```text
crash_preprocess_eda.py
```

It performs the following operations:

1. Downloads Chicago traffic-crash records from the City API.
2. Cleans crash records and filters them to the project date range.
3. Derives time-related features such as hour of day, day of week, and month.
4. Creates a binary severe-crash indicator.
5. Reads and processes NOAA hourly weather data when available.
6. Reads and processes BTS flight data when available.
7. Aggregates crash records into an hourly crash-count dataset.
8. Combines hourly crash counts with available weather and flight features.
9. Generates EDA association tables and supporting datasets.
10. Saves the processed datasets and analysis outputs.

The hourly dataset contains a continuous hourly index covering the full project period, including hours with zero crashes.

The BTS data is aggregated to hourly intervals and joined to this panel. Missing flight activity is handled separately from missing delay measurements, so hours without scheduled departures are not treated as hours with zero delay.

### Severe crash indicator

The preprocessing script creates a binary target variable:

- `1` — Severe crash.
- `0` — Non-severe crash.

This variable is used by the crash severity classification models.

### EDA methods

The script examines relationships between crash characteristics and severity using:

- Point-biserial correlations for numeric features.
- Cramér's V for categorical features.
- Severity counts and rates grouped by categorical features.
- Spearman correlations between hourly numeric features and hourly crash counts.

These analyses help identify potentially useful features for modeling. They do not establish that any feature causes crashes or increases crash severity.

---

## Machine learning models

The project includes two separate modeling scripts.

### 1. Crash severity classification

Script:

```text
model_severity.py
```

This script predicts whether an individual crash is severe using classification models.

It evaluates three approaches:

- Dummy classifier (baseline)
- Logistic regression
- HistGradientBoostingClassifier (gradient boosting)

The models are evaluated using a chronological data split:

- Training data: Before 2023
- Validation data: 2023
- Testing data: 2024 onward

This approach helps evaluate model performance on later observations rather than randomly mixing records from different years.

#### Feature groups

Two feature configurations are evaluated.

**Scene-only features**

These include information associated with the crash scene, such as:

- Weather condition
- Lighting condition
- Roadway surface condition
- Trafficway type
- Roadway alignment
- Traffic control device
- First crash type
- Posted speed limit
- Latitude and longitude
- Time-related features

**Scene-plus-post-crash features**

This configuration adds:

- `damage`
- `prim_contributory_cause`

These additional variables may provide information recorded after or during crash investigation. They may not be available when making a prediction before or immediately after a crash, so their results should be interpreted separately from the scene-only models.

#### Evaluation metrics

The classification models are evaluated using:

- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- Precision
- Recall
- Confusion matrices
- Brier score and related probability evaluation, where reported
- Feature importance analysis

PR-AUC is particularly important because severe crashes represent a relatively small percentage of the dataset.

### 2. Hourly crash-count prediction

Script:

```text
model_crash_count.py
```

This script predicts the number of traffic crashes occurring during an hour.

Unlike the severity model, which predicts a binary outcome for an individual crash, this model predicts a numerical crash count for each hourly interval.

#### Features

The model uses several categories of predictors.

**Calendar and time features**

- Hour of day
- Day of week
- Month
- Weekend indicator
- Cyclical sine and cosine transformations of time variables

**Historical crash features**

- `lag_24` — Crash count from 24 hours earlier.
- `lag_168` — Crash count from 168 hours earlier.

These features help capture daily and weekly crash patterns.

**Weather features**

- Temperature
- Precipitation
- Wind speed
- Visibility

**Flight activity features**

- Hourly flight departures
- Average departure delay
- Hourly cancelled flights

#### Models

The script compares three approaches:

- Day-of-week and hour-of-day mean baseline
- Poisson regression
- HistGradientBoostingRegressor using Poisson loss

The baseline estimates expected crash counts using historical averages for each day-of-week and hour-of-day combination.

The models use a chronological training, validation, and testing split to preserve the time ordering of observations.

#### Evaluation metrics

The count prediction models are evaluated using:

- Mean Poisson deviance
- Mean Absolute Error (MAE)
- Correlation between predicted and actual crash counts
- Deviance skill relative to the baseline
- Permutation feature importance

Lower Poisson deviance and MAE indicate better predictive performance.

The historical lag features assume that previous hourly crash counts are available when making predictions. Predictions that use actual previous observations should be described as rolling forecasts rather than forecasts made without updated historical information.

The crash-count data also exhibits overdispersion, meaning its variance is substantially larger than its mean. This is an important limitation to discuss when interpreting the Poisson regression results.

---

## Outputs

The preprocessing script creates the `output/` directory and generates the following files.

### Preprocessed datasets

| File | Description |
|---|---|
| `cleaned_crashes.csv.gz` | Gzip-compressed cleaned crash-level records, derived time fields, and severe indicator. |
| `hourly_crash_weather_flights.csv` | Continuous hourly dataset containing crash counts and available weather and flight features. |

The cleaned crash dataset is saved as a **gzip-compressed CSV** to reduce storage requirements while retaining its records and selected columns.

It can be loaded directly with pandas:

```python
df = pd.read_csv("output/cleaned_crashes.csv.gz")
```

Pandas automatically decompresses the file.

### EDA outputs

| File | Description |
|---|---|
| `severity_numeric_associations.csv` | Point-biserial associations between numeric features and crash severity. |
| `severity_categorical_associations.csv` | Cramér's V associations between categorical features and crash severity. |
| `severity_by_<field>.csv` | Severity counts and rates grouped by eligible categorical fields. |
| `hourly_crash_count_associations.csv` | Spearman associations between numeric hourly features and crash counts. |
| `crashes_by_year_hour.csv` | Crash counts grouped by year and hour of day. |

### Modeling outputs

The two modeling scripts generate their evaluation results, feature importance information, and visualizations in the `output/` directory.

These include model comparison results and plots for evaluating predictive performance, such as ROC, precision-recall, calibration, and feature importance visualizations where applicable.

The exact output filenames are defined in the respective modeling scripts.

---

## Current experimental results

The following results were obtained from a successful run using the 2019–2024 project dataset. Results may change if the source datasets or preprocessing procedures are updated.

### Crash severity classification

The best-performing severity model was gradient boosting using both scene and post-crash features.

| Model | ROC-AUC | PR-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline | 0.500 | 0.104 | 0.000 | 0.000 |
| Logistic regression (scene only) | 0.806 | 0.418 | 0.291 | 0.611 |
| Gradient boosting (scene only) | 0.814 | 0.432 | 0.304 | 0.608 |
| Logistic regression (scene + post-crash) | 0.827 | 0.444 | 0.324 | 0.620 |
| Gradient boosting (scene + post-crash) | **0.834** | **0.459** | **0.340** | **0.615** |

The results show that both machine learning approaches substantially outperform the baseline. The addition of post-crash features improves classification performance, although these features have limitations for real-time prediction.

### Hourly crash-count prediction

The best-performing hourly count model was gradient boosting using Poisson loss.

| Model | Mean Poisson Deviance | MAE | Correlation | Deviance Skill |
|---|---:|---:|---:|---:|
| Day/hour baseline | 1.721 | 3.494 | 0.817 | 0.000 |
| Poisson regression | 2.325 | 4.047 | 0.749 | -0.351 |
| Gradient boosting (Poisson) | **1.532** | **3.339** | **0.838** | **0.110** |

Gradient boosting improved Poisson deviance by approximately 11% compared with the baseline. Time-related features were the strongest predictors, followed by historical crash counts. Weather and flight activity also contributed predictive information.

These results indicate that gradient boosting can capture nonlinear relationships and temporal patterns that the standard Poisson regression model does not capture as effectively.

---

## Configuration

The date range and folder locations are defined near the top of `crash_preprocess_eda.py`:

```python
START = "2019-01-01"
END_EXCLUSIVE = "2025-01-01"
DATA_DIR = Path("data")
```

The project uses an exclusive end date, so `2025-01-01` includes all records through December 31, 2024.

To change the date range, update both date constants and ensure the external NOAA and BTS datasets cover the selected period.

The model scripts also contain their own configuration settings for input files, feature selection, model parameters, and output locations.

---

## Troubleshooting

**Connection or API error**

- Check internet access.
- Retry the preprocessing script.
- The City API is queried in monthly chunks.

**No NOAA or BTS features in the output**

- Confirm the CSV files are in the correct subfolder.
- Check that the expected columns are present.
- Inspect the preprocessing diagnostics to confirm records were loaded and parsed.

**Missing-column error**

- Check the expected input columns described above.
- Update the column mapping in `read_noaa()` or `read_bts()` if the downloaded CSV uses different headers.

**Unexpected weather or flight joins**

- Confirm that source timestamps represent local Chicago time.
- Inspect the hourly output.
- The script floors timestamps to the hour and does not perform timezone conversion.

**Compressed crash file cannot be found**

- Confirm that `crash_preprocess_eda.py` completed successfully.
- Verify that `output/cleaned_crashes.csv.gz` exists.
- Use the compressed filename when loading the dataset:

```python
pd.read_csv("output/cleaned_crashes.csv.gz")
```

**Modeling script cannot find its input dataset**

- Run `crash_preprocess_eda.py` first.
- Confirm that the required output files exist.
- Run all commands from the repository root.

**Missing values in modeling features**

- Review the preprocessing output and model diagnostics.
- Confirm that optional weather and flight data were successfully loaded.
- Check how missing external feature values are handled in the modeling scripts.

---

## Project limitations and considerations

- The analysis covers Chicago traffic crashes from 2019 through 2024.
- External weather and flight data availability depends on the supplied CSV exports.
- Flight activity currently represents scheduled departures originating at ORD, not all Chicago-area airport activity.
- Weather observations are aggregated by hour and may not represent conditions at every crash location.
- Associations between features and crash outcomes do not establish causation.
- Severe crashes are a minority class, making classification performance sensitive to decision thresholds.
- Post-crash variables may not be available for real-time severity predictions.
- Hourly crash counts are overdispersed relative to a standard Poisson distribution.
- Historical crash-count features require prior observations to be available when generating predictions.
- Model performance is evaluated on later time periods, but future performance may differ as traffic patterns, weather, and other conditions change.

The project demonstrates how multiple transportation and environmental datasets can be combined to investigate traffic crash patterns and evaluate machine learning approaches for predicting crash severity and hourly crash counts.
