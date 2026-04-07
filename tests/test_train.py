"""Tests for temporal splitting, baselines, and training pipeline shape."""

import numpy as np
import pandas as pd
import pytest

from mobility_feature_pipeline.baseline import always_negative, low_bikes_rule
from mobility_feature_pipeline.model_config import FEATURE_COLUMNS, TARGET_COLUMN
from mobility_feature_pipeline.train import compute_split_dates, temporal_split


@pytest.fixture
def sample_df():
    """Minimal DataFrame with known timestamps and features for split testing."""
    rng = np.random.RandomState(42)
    n_stations = 2
    # 20 unique timestamps, 15 min apart
    timestamps = pd.date_range("2024-06-15 08:00", periods=20, freq="15min")
    rows = []
    for ts in timestamps:
        for sid in [f"S{i}" for i in range(n_stations)]:
            row = {"station_id": sid, "obs_ts": ts}
            row["feature_cutoff_ts"] = ts
            row["label_window_end"] = ts + pd.Timedelta(hours=1)
            row[TARGET_COLUMN] = rng.randint(0, 2)
            for col in FEATURE_COLUMNS:
                row[col] = rng.rand() * 10
            rows.append(row)
    return pd.DataFrame(rows)


class TestTemporalSplit:
    def test_no_timestamp_overlap(self, sample_df):
        train_df, val_df, test_df, _ = temporal_split(sample_df)
        train_ts = set(train_df["obs_ts"])
        val_ts = set(val_df["obs_ts"])
        test_ts = set(test_df["obs_ts"])
        assert not (train_ts & val_ts)
        assert not (val_ts & test_ts)
        assert not (train_ts & test_ts)

    def test_all_rows_assigned(self, sample_df):
        train_df, val_df, test_df, _ = temporal_split(sample_df)
        assert len(train_df) + len(val_df) + len(test_df) == len(sample_df)

    def test_temporal_ordering(self, sample_df):
        train_df, val_df, test_df, _ = temporal_split(sample_df)
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df["obs_ts"].max() < val_df["obs_ts"].min()
        if len(val_df) > 0 and len(test_df) > 0:
            assert val_df["obs_ts"].max() < test_df["obs_ts"].min()

    def test_split_info_populated(self, sample_df):
        _, _, _, split_info = temporal_split(sample_df)
        for name in ["train", "val", "test"]:
            assert name in split_info
            assert "rows" in split_info[name]
            assert "positive_rate" in split_info[name]
            assert split_info[name]["rows"] > 0

    def test_deterministic(self, sample_df):
        dates1 = compute_split_dates(sample_df)
        dates2 = compute_split_dates(sample_df)
        assert dates1[0] == dates2[0]
        assert dates1[1] == dates2[1]

    def test_all_station_rows_same_split(self, sample_df):
        """All rows for a given obs_ts go to the same split."""
        train_df, val_df, test_df, _ = temporal_split(sample_df)
        for ts in sample_df["obs_ts"].unique():
            in_train = ts in train_df["obs_ts"].values
            in_val = ts in val_df["obs_ts"].values
            in_test = ts in test_df["obs_ts"].values
            assert sum([in_train, in_val, in_test]) == 1


class TestBaselines:
    def test_always_negative(self):
        y_true = np.array([0, 1, 0, 1, 0])
        result = always_negative(y_true)
        assert np.all(result["y_pred"] == 0)
        assert np.all(result["y_score"] == 0.0)
        assert len(result["y_pred"]) == len(y_true)

    def test_low_bikes_rule_k0(self):
        X = pd.DataFrame({"ft_bikes_available": [0.0, 0.5, 1.0, 5.0, 10.0]})
        result = low_bikes_rule(X, k=0)
        expected = np.array([1, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["y_pred"], expected)

    def test_low_bikes_rule_k1(self):
        X = pd.DataFrame({"ft_bikes_available": [0.0, 0.5, 1.0, 5.0, 10.0]})
        result = low_bikes_rule(X, k=1)
        expected = np.array([1, 1, 1, 0, 0])
        np.testing.assert_array_equal(result["y_pred"], expected)

    def test_low_bikes_rule_scores_monotonic(self):
        X = pd.DataFrame({"ft_bikes_available": [0.0, 5.0, 10.0]})
        result = low_bikes_rule(X, k=1)
        # Lower bikes → higher risk score
        assert result["y_score"][0] > result["y_score"][1] > result["y_score"][2]
