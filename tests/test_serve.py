"""Tests for feature consistency between training pipeline and online scoring."""

import math
from datetime import datetime

import pandas as pd
import pytest

from mobility_feature_pipeline.config import Settings
from mobility_feature_pipeline.model_config import FEATURE_COLUMNS
from mobility_feature_pipeline.pipeline import _build_query
from mobility_feature_pipeline.serve import ScoringError, reconstruct_features


def _values_match(a, b):
    """Compare two values, treating NaN/None as equal."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) < 1e-9
    return a == b


@pytest.fixture
def training_df(synthetic_db):
    """Build training dataset from synthetic DB via the batch SQL pipeline."""
    settings = Settings(min_forward_rows=10)
    query = _build_query(settings)
    arrow_table = synthetic_db.execute(query).arrow().read_all()
    return arrow_table.to_pandas()


class TestFeatureConsistency:
    def test_online_matches_training(self, synthetic_db, training_df):
        """score_features.sql must produce identical features to training pipeline."""
        mismatches = []
        for _, row in training_df.iterrows():
            result = reconstruct_features(
                db_path=None,
                station_id=row["station_id"],
                obs_ts=row["obs_ts"].to_pydatetime(),
                _con_override=synthetic_db,
            )
            for col in FEATURE_COLUMNS:
                train_val = None if pd.isna(row[col]) else float(row[col])
                score_val = result["features"][col]
                if not _values_match(train_val, score_val):
                    mismatches.append(
                        f"  {row['station_id']} @ {row['obs_ts']} → {col}: "
                        f"train={train_val}, score={score_val}"
                    )
        assert not mismatches, f"Feature mismatches ({len(mismatches)}):\n" + "\n".join(
            mismatches[:20]
        )

    def test_feature_column_order(self, synthetic_db, training_df):
        """Online features must be returned in exact FEATURE_COLUMNS order."""
        row = training_df.iloc[0]
        result = reconstruct_features(
            db_path=None,
            station_id=row["station_id"],
            obs_ts=row["obs_ts"].to_pydatetime(),
            _con_override=synthetic_db,
        )
        assert list(result["features"].keys()) == FEATURE_COLUMNS

    def test_snapshot_source_ts_present(self, synthetic_db, training_df):
        """Reconstruction must return the snapshot source timestamp."""
        row = training_df.iloc[0]
        result = reconstruct_features(
            db_path=None,
            station_id=row["station_id"],
            obs_ts=row["obs_ts"].to_pydatetime(),
            _con_override=synthetic_db,
        )
        assert result["snapshot_source_ts"] is not None


class TestDomainValidation:
    def test_unknown_station(self, synthetic_db):
        """Unknown station should raise station_not_found."""
        with pytest.raises(ScoringError) as exc_info:
            reconstruct_features(
                db_path=None,
                station_id="NONEXISTENT",
                obs_ts=datetime(2024, 6, 15, 9, 0, 0),
                _con_override=synthetic_db,
            )
        assert exc_info.value.code == "station_not_found"

    def test_small_capacity_rejected(self, synthetic_db):
        """Station with capacity < 5 should raise out_of_domain."""
        with pytest.raises(ScoringError) as exc_info:
            reconstruct_features(
                db_path=None,
                station_id="S3",  # capacity=3 in synthetic data
                obs_ts=datetime(2024, 6, 15, 9, 0, 0),
                _con_override=synthetic_db,
            )
        assert exc_info.value.code == "out_of_domain"

    def test_stale_data_rejected(self, synthetic_db):
        """obs_ts far beyond available data should raise stale_data."""
        # Synthetic data ends around 11:00. Request at 12:00 → >15 min stale.
        with pytest.raises(ScoringError) as exc_info:
            reconstruct_features(
                db_path=None,
                station_id="S1",
                obs_ts=datetime(2024, 6, 15, 12, 0, 0),
                _con_override=synthetic_db,
            )
        assert exc_info.value.code == "stale_data"

    def test_no_snapshot_before_data_start(self, synthetic_db):
        """obs_ts before any data should raise no_snapshot or station_not_found."""
        with pytest.raises(ScoringError) as exc_info:
            reconstruct_features(
                db_path=None,
                station_id="S1",
                obs_ts=datetime(2024, 6, 14, 0, 0, 0),  # day before data
                _con_override=synthetic_db,
            )
        assert exc_info.value.code in ("no_snapshot", "station_not_found")
