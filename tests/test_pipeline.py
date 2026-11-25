"""Run with:  python -m unittest discover -s tests -v      (no pytest needed)"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crash_severity import config, experiments as rx, preprocessing as pre
from crash_severity.evaluation import paired_bootstrap_diff
from crash_severity.features import DETAIL_CAT, FEATURE_SETS, OUTCOME_COLS, assert_no_leakage
from crash_severity.models import available_models, make_frames


def synthetic(n=6000, prevalence=0.08, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "crash_date": pd.date_range("2019-01-01", periods=n, freq="45min"),
        "lighting_condition": rng.choice(["DAYLIGHT", "DARK", "DUSK"], n),
        "roadway_surface_cond": rng.choice(["DRY", "WET", "SNOW"], n),
        "season": rng.choice(["winter", "summer"], n),
        "posted_speed_limit": rng.choice([25, 30, 35, np.nan], n),
        "hour_sin": rng.normal(size=n), "is_weekend": rng.integers(0, 2, n),
    })
    df["severe"] = (rng.random(n) < prevalence).astype(int)
    return df


class SeverityLabel(unittest.TestCase):
    def test_nonincapacitating_is_not_severe(self):
        """Regression test: a substring match on 'INCAPACITATING' used to flag NONINCAPACITATING as severe."""
        df = pd.DataFrame({
            "most_severe_injury": ["FATAL", "INCAPACITATING INJURY", "NONINCAPACITATING INJURY",
                                   "REPORTED, NOT EVIDENT", "NO INDICATION OF INJURY", None],
            "injuries_fatal": [1, 0, 0, 0, 0, 0], "injuries_incapacitating": [0, 1, 0, 0, 0, 0]})
        out = pre.build_severity(df)
        self.assertEqual(out["severe"].tolist(), [1, 1, 0, 0, 0, 0])
        self.assertEqual(out["any_injury"].tolist(), [1, 1, 1, 1, 0, 0])
        self.assertEqual(out["severity_class"].tolist(), [2, 2, 1, 1, 0, 0])


class LeakageGuards(unittest.TestCase):
    def test_context_sets_have_no_outcome_or_detail_columns(self):
        for name, spec in FEATURE_SETS.items():
            used = set(spec["cat"]) | set(spec["num"])
            self.assertFalse(used & OUTCOME_COLS, name)
            if spec["tier"] == "context":
                self.assertFalse(used & set(DETAIL_CAT), name)

    def test_guard_raises(self):
        with self.assertRaises(ValueError):
            assert_no_leakage(["first_crash_type"], [])
        with self.assertRaises(ValueError):
            assert_no_leakage([], ["injuries_total"], allow_details=True)
        assert_no_leakage(["first_crash_type"], [], allow_details=True)      # allowed in the post-crash tier


class Splits(unittest.TestCase):
    def test_chronological_no_overlap(self):
        old = (config.TRAIN_END, config.VAL_END)
        config.TRAIN_END, config.VAL_END = "2019-03-01", "2019-04-01"
        try:
            df = synthetic(n=4000)
            tr, va, te, _ = rx.split(df)
        finally:
            config.TRAIN_END, config.VAL_END = old
        self.assertLess(tr.crash_date.max(), va.crash_date.min())
        self.assertLess(va.crash_date.max(), te.crash_date.min())
        self.assertEqual(len(tr) + len(va) + len(te), len(df))


class ExternalData(unittest.TestCase):
    def test_standard_time_to_clock_time(self):
        ts = pd.Series(pd.to_datetime(["2019-07-01 12:00", "2019-01-15 12:00"]))
        out = pre.lst_to_clock(ts)
        self.assertEqual(out[0], pd.Timestamp("2019-07-01 13:00"))     # daylight time: +1 hour
        self.assertEqual(out[1], pd.Timestamp("2019-01-15 12:00"))     # standard time: unchanged

    def test_noaa_reader_handles_trace_and_suffixes(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"DATE": ["2019-01-15T00:51:00", "2019-01-15T01:51:00"],
                          "HourlyPrecipitation": ["T", "0.05s"], "HourlyDryBulbTemperature": ["20", "21"],
                          "HourlyPresentWeatherType": ["SN:03", ""]}).to_csv(Path(d) / "x.csv", index=False)
            wx = pre.read_noaa(Path(d), to_clock=False)
        self.assertAlmostEqual(wx.loc[0, "precip_in"], 0.005)
        self.assertAlmostEqual(wx.loc[1, "precip_in"], 0.05)
        self.assertEqual(wx["wx_snow"].tolist(), [1, 0])

    def test_bts_reader_zero_fills_empty_hours(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"FlightDate": ["2019-01-15"] * 3, "Origin": ["ORD", "ORD", "ATL"],
                          "CRSDepTime": [100, 300, 100], "DepDelay": [5, 15, 99], "Cancelled": [0, 1, 0]}
                         ).to_csv(Path(d) / "b.csv", index=False)
            fl = pre.read_bts(Path(d)).set_index("hour")
        self.assertEqual(fl.loc["2019-01-15 01:00", "flight_departures"], 1)
        self.assertEqual(fl.loc["2019-01-15 02:00", "flight_departures"], 0)        # real zero, not missing
        self.assertEqual(fl.loc["2019-01-15 03:00", "cancelled_flights"], 1)        # ATL flight ignored

    def test_external_features_use_the_previous_complete_hour(self):
        crashes = pd.DataFrame({"hour": pd.to_datetime(["2019-01-15 02:00"])})
        weather = pd.DataFrame({"hour": pd.to_datetime(["2019-01-15 01:00"]), "temp_f": [20.0]})
        flights = pd.DataFrame({"hour": pd.to_datetime(["2019-01-15 01:00"]), "flight_departures": [7]})
        joined = pre.attach_external(crashes, weather, flights)
        self.assertEqual(joined.loc[0, "temp_f"], 20.0)
        self.assertEqual(joined.loc[0, "flight_departures"], 7)
        self.assertEqual(joined.loc[0, "has_weather"], 1)
        self.assertEqual(joined.loc[0, "has_flights"], 1)


class Evaluation(unittest.TestCase):
    def test_paired_difference_of_identical_scores_is_zero(self):
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.1).astype(int)
        p = rng.random(2000)
        diff, lo, hi, _ = paired_bootstrap_diff(y, p, p, n_boot=50)
        self.assertEqual((diff, lo, hi), (0.0, 0.0, 0.0))

    def test_shuffled_labels_score_near_prevalence(self):
        """A pipeline with no leakage cannot beat chance on randomly generated labels."""
        from sklearn.metrics import average_precision_score
        df = synthetic(n=8000, prevalence=0.08)
        cat, num = ["lighting_condition", "roadway_surface_cond", "season"], ["posted_speed_limit", "hour_sin", "is_weekend"]
        tr, te = df.iloc[:5000], df.iloc[5000:]
        X_tr, _, X_te = make_frames(tr, te, te, cat, num)
        m = available_models()["hist_gb"](cat, num, seed=0).fit(X_tr, tr["severe"].to_numpy())
        ap = average_precision_score(te["severe"], m.predict_proba(X_te))
        self.assertLess(ap, te["severe"].mean() + 0.03)


class ModelSmoke(unittest.TestCase):
    def test_every_installed_model_fits_and_predicts(self):
        df = synthetic(n=3000)
        df["severe"] = ((df["lighting_condition"] == "DARK") & (np.random.default_rng(1).random(len(df)) < 0.5)).astype(int)
        cat, num = ["lighting_condition", "roadway_surface_cond", "season"], ["posted_speed_limit", "hour_sin", "is_weekend"]
        tr, va, te = df.iloc[:1800], df.iloc[1800:2400], df.iloc[2400:]
        X_tr, X_va, X_te = make_frames(tr, va, te, cat, num)
        for name, cls in available_models().items():
            with self.subTest(model=name):
                params = {"n_estimators": 20} if name == "rf" else {}
                p = cls(cat, num, seed=0, **params).fit(X_tr, tr["severe"].to_numpy(), X_va, va["severe"].to_numpy()).predict_proba(X_te)
                self.assertEqual(len(p), len(te))
                self.assertTrue(np.all((p >= 0) & (p <= 1)))


if __name__ == "__main__":
    unittest.main()
