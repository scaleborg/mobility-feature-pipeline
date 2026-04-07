"""Tests for the batch triage layer (Slice 4)."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mobility_feature_pipeline.serve import ScoringError
from mobility_feature_pipeline.triage import batch_score


@pytest.fixture
def mock_booster(tmp_path):
    """Train a minimal LightGBM model on synthetic data."""
    # Create data where low bikes → high risk, high bikes → low risk
    rng = np.random.RandomState(42)
    n = 200
    X = rng.rand(n, 22)
    # First feature (ft_bikes_available) drives the label
    y = (X[:, 0] < 0.3).astype(int)
    ds = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 4},
        ds,
        num_boost_round=20,
    )
    return booster


@pytest.fixture
def mock_model_path(mock_booster, tmp_path):
    """Save booster to disk and return path."""
    path = tmp_path / "test_model.lgbm"
    mock_booster.save_model(str(path))
    return path


class TestBatchScoreRanking:
    """Core triage logic tests using the synthetic_db fixture."""

    def test_batch_score_ranking(self, synthetic_db, mock_booster):
        """S1 (stockout) should rank above S2 (healthy). S3 excluded."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)  # 90 min in — S1 at 0 bikes
        result = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            _con_override=synthetic_db,
        )

        # S3 (capacity=3) is excluded by the pre-filter, so only S1 and S2 are candidates
        assert result["candidate_stations"] == 2
        assert result["scored"] == 2  # S1 and S2

        stations = result["stations"]
        assert len(stations) == 2
        # S1 (0 bikes, stockout) should have higher risk than S2 (15 bikes)
        assert stations[0]["station_id"] == "S1"
        assert stations[1]["station_id"] == "S2"
        assert stations[0]["risk_score"] >= stations[1]["risk_score"]

    def test_ranking_uses_score_not_label(self, synthetic_db, mock_booster):
        """Two stations with same label must be ordered by continuous risk_score."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)
        result = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            _con_override=synthetic_db,
        )

        stations = result["stations"]
        # Verify scores are in strict descending order
        for i in range(len(stations) - 1):
            assert stations[i]["risk_score"] >= stations[i + 1]["risk_score"], (
                f"Station at rank {i + 1} (score={stations[i]['risk_score']}) "
                f"should have score >= rank {i + 2} (score={stations[i + 1]['risk_score']})"
            )

    def test_top_n_truncation(self, synthetic_db, mock_booster):
        """top_n=1 returns only the single highest-risk station."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)
        result = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=1,
            _con_override=synthetic_db,
        )

        assert result["scored"] == 2
        assert len(result["stations"]) == 1
        assert result["stations"][0]["rank"] == 1
        assert result["top_n"] == 1

    def test_skipped_stations_reported(self, synthetic_db, mock_booster):
        """Stations that fail reconstruct_features appear in skipped_stations."""
        obs_ts = datetime(2024, 6, 15, 9, 0, 0)

        # S3 is pre-filtered by active_stations.sql (capacity < 5), so it never
        # reaches reconstruct_features. To test the skip-reporting path, mock
        # discovery to include a station that will fail validation.
        original_rf = __import__(
            "mobility_feature_pipeline.serve", fromlist=["reconstruct_features"]
        ).reconstruct_features

        def patched_rf(db_path, station_id, obs_ts, _con_override=None):
            if station_id == "S3":
                raise ScoringError("out_of_domain", "Capacity below threshold")
            return original_rf(db_path, station_id, obs_ts, _con_override=_con_override)

        with patch("mobility_feature_pipeline.triage._discover_candidates") as mock_dc:
            mock_dc.return_value = ["S1", "S2", "S3"]
            with patch(
                "mobility_feature_pipeline.triage.reconstruct_features", side_effect=patched_rf
            ):
                result = batch_score(
                    booster=mock_booster,
                    db_path=Path("/unused"),
                    obs_ts=obs_ts,
                    top_n=10,
                    _con_override=synthetic_db,
                )

        assert result["candidate_stations"] == 3
        assert result["scored"] == 2
        assert result["skipped"] == 1
        skipped_ids = {s["station_id"]: s["reason"] for s in result["skipped_stations"]}
        assert "S3" in skipped_ids
        assert skipped_ids["S3"] == "out_of_domain"

    def test_no_active_stations(self, synthetic_db, mock_booster):
        """Timestamp far in the future → no_active_stations error."""
        obs_ts = datetime(2030, 1, 1, 0, 0, 0)
        with pytest.raises(ScoringError) as exc_info:
            batch_score(
                booster=mock_booster,
                db_path=Path("/unused"),
                obs_ts=obs_ts,
                top_n=10,
                _con_override=synthetic_db,
            )

        assert exc_info.value.code == "no_active_stations"
        assert exc_info.value.status == 404

    def test_all_candidates_fail(self, synthetic_db, mock_booster):
        """If all candidates fail validation, return scored=0 with skipped reasons."""
        # Use a timestamp where candidates exist but data is stale
        # S1/S2 data ends at base + 180min = 11:00. At 11:20 staleness > 15min.
        obs_ts = datetime(2024, 6, 15, 11, 20, 0)

        # Pre-filter might not find candidates if staleness window doesn't overlap.
        # Instead, mock reconstruct_features to always fail.
        with patch("mobility_feature_pipeline.triage.reconstruct_features") as mock_rf:
            mock_rf.side_effect = ScoringError("stale_data", "Data too old")

            # Also mock discovery to return some candidates
            with patch("mobility_feature_pipeline.triage._discover_candidates") as mock_dc:
                mock_dc.return_value = ["S1", "S2"]
                result = batch_score(
                    booster=mock_booster,
                    db_path=Path("/unused"),
                    obs_ts=obs_ts,
                    top_n=10,
                    _con_override=synthetic_db,
                )

        assert result["candidate_stations"] == 2
        assert result["scored"] == 0
        assert result["skipped"] == 2
        assert len(result["stations"]) == 0
        assert len(result["skipped_stations"]) == 2
        assert all(s["reason"] == "stale_data" for s in result["skipped_stations"])

    def test_deterministic_ordering(self, synthetic_db, mock_booster):
        """Same inputs produce identical output ordering."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)
        result1 = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            _con_override=synthetic_db,
        )
        result2 = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            _con_override=synthetic_db,
        )

        ids1 = [s["station_id"] for s in result1["stations"]]
        ids2 = [s["station_id"] for s in result2["stations"]]
        assert ids1 == ids2

        scores1 = [s["risk_score"] for s in result1["stations"]]
        scores2 = [s["risk_score"] for s in result2["stations"]]
        assert scores1 == scores2

    def test_funnel_counts_consistent(self, synthetic_db, mock_booster):
        """candidate_stations == scored + skipped."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)
        result = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            _con_override=synthetic_db,
        )

        assert result["candidate_stations"] == result["scored"] + result["skipped"]

    def test_debug_mode(self, synthetic_db, mock_booster):
        """debug=True includes bounded diagnostic fields per station."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)
        result = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            debug=True,
            _con_override=synthetic_db,
        )

        for station in result["stations"]:
            assert "debug" in station
            dbg = station["debug"]
            assert "availability_ratio" in dbg
            assert "snapshot_age_min" in dbg
            assert "bikes_delta_60m" in dbg
            # Only these three fields
            assert len(dbg) == 3

    def test_no_debug_by_default(self, synthetic_db, mock_booster):
        """Default response does not include debug fields."""
        obs_ts = datetime(2024, 6, 15, 9, 30, 0)
        result = batch_score(
            booster=mock_booster,
            db_path=Path("/unused"),
            obs_ts=obs_ts,
            top_n=10,
            _con_override=synthetic_db,
        )

        for station in result["stations"]:
            assert "debug" not in station


class TestTriageAPIContract:
    """FastAPI endpoint tests for POST /triage."""

    @pytest.fixture
    def client(self, synthetic_db, mock_model_path):
        from mobility_feature_pipeline.server import create_app

        db_path = Path("/fake/path.duckdb")
        app = create_app(model_path=mock_model_path, db_path=db_path)
        return TestClient(app)

    def test_triage_returns_expected_shape(self, client, synthetic_db, mock_booster):
        """POST /triage returns correct response structure."""
        with patch("mobility_feature_pipeline.server.batch_score") as mock_bs:
            mock_bs.return_value = {
                "obs_ts": "2024-06-15T09:30:00",
                "model_name": None,
                "candidate_stations": 3,
                "scored": 2,
                "skipped": 1,
                "top_n": 10,
                "stations": [
                    {
                        "rank": 1,
                        "station_id": "S1",
                        "risk_score": 0.85,
                        "risk_label": 1,
                        "bikes_available": 0.0,
                        "capacity": 30.0,
                        "snapshot_source_ts": "2024-06-15T09:29:00",
                    }
                ],
                "skipped_stations": [{"station_id": "S3", "reason": "out_of_domain"}],
            }

            resp = client.post(
                "/triage",
                json={"obs_ts": "2024-06-15T09:30:00", "top_n": 10},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "candidate_stations" in data
            assert "scored" in data
            assert "skipped" in data
            assert "stations" in data
            assert "skipped_stations" in data
            assert data["candidate_stations"] == 3
            assert data["scored"] == 2

    def test_triage_validation_top_n_zero(self, client):
        """top_n=0 returns 422."""
        resp = client.post(
            "/triage",
            json={"obs_ts": "2024-06-15T09:30:00", "top_n": 0},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"] == "invalid_top_n"

    def test_triage_validation_top_n_too_large(self, client):
        """top_n=51 returns 422."""
        resp = client.post(
            "/triage",
            json={"obs_ts": "2024-06-15T09:30:00", "top_n": 51},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"] == "invalid_top_n"

    def test_triage_no_active_stations(self, client):
        """ScoringError propagates as 404."""
        with patch("mobility_feature_pipeline.server.batch_score") as mock_bs:
            mock_bs.side_effect = ScoringError("no_active_stations", "No stations", status=404)
            resp = client.post(
                "/triage",
                json={"obs_ts": "2030-01-01T00:00:00", "top_n": 10},
            )
            assert resp.status_code == 404
            data = resp.json()
            assert data["error"] == "no_active_stations"

    def test_triage_debug_mode_via_query(self, client):
        """?debug=true is passed through to batch_score."""
        with patch("mobility_feature_pipeline.server.batch_score") as mock_bs:
            mock_bs.return_value = {
                "obs_ts": "2024-06-15T09:30:00",
                "model_name": None,
                "candidate_stations": 1,
                "scored": 1,
                "skipped": 0,
                "top_n": 10,
                "stations": [
                    {
                        "rank": 1,
                        "station_id": "S1",
                        "risk_score": 0.85,
                        "risk_label": 1,
                        "bikes_available": 0.0,
                        "capacity": 30.0,
                        "snapshot_source_ts": "2024-06-15T09:29:00",
                        "debug": {
                            "availability_ratio": 0.0,
                            "snapshot_age_min": 1.0,
                            "bikes_delta_60m": -10.0,
                        },
                    }
                ],
                "skipped_stations": [],
            }

            resp = client.post(
                "/triage?debug=true",
                json={"obs_ts": "2024-06-15T09:30:00"},
            )
            assert resp.status_code == 200
            # Verify debug was passed through
            mock_bs.assert_called_once()
            call_kwargs = mock_bs.call_args
            assert call_kwargs.kwargs.get("debug") is True or (
                len(call_kwargs.args) > 4 and call_kwargs.args[4] is True
            )
