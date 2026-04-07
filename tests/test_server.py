"""API contract tests for the FastAPI scoring server."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mobility_feature_pipeline.server import create_app


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a minimal LightGBM model for testing."""
    import lightgbm as lgb

    X = np.array([[1.0] * 22, [2.0] * 22])
    y = np.array([0, 1])
    ds = lgb.Dataset(X, label=y)
    booster = lgb.train({"objective": "binary", "verbose": -1}, ds, num_boost_round=2)
    model_path = tmp_path / "test_model.lgbm"
    booster.save_model(str(model_path))
    return model_path


@pytest.fixture
def client(synthetic_db, mock_model_path):
    """Create a test client with mocked DB connection."""
    # We need a real db_path for create_app, but we'll mock reconstruct_features
    db_path = Path("/fake/path.duckdb")
    app = create_app(model_path=mock_model_path, db_path=db_path)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client, mock_model_path):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_name"] == mock_model_path.name
        assert data["feature_count"] == 22

    def test_health_includes_db_source(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "db_source" in data


class TestScoreEndpoint:
    def test_score_returns_expected_fields(self, client, synthetic_db):
        """Score endpoint returns lean response by default."""
        with patch("mobility_feature_pipeline.server.score") as mock_score:
            mock_score.return_value = {
                "station_id": "S1",
                "obs_ts": "2024-06-15T09:00:00",
                "risk_score": 0.75,
                "risk_label": 1,
                "threshold": 0.5,
                "snapshot_source_ts": "2024-06-15T08:59:00",
                "model_name": "test_model.lgbm",
            }
            resp = client.post(
                "/score",
                json={"station_id": "S1", "obs_ts": "2024-06-15T09:00:00"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "risk_score" in data
            assert "risk_label" in data
            assert "snapshot_source_ts" in data
            assert "model_name" in data
            assert "features" not in data

    def test_score_debug_includes_features(self, client, synthetic_db):
        """Score with debug=true includes feature dict."""
        with patch("mobility_feature_pipeline.server.score") as mock_score:
            mock_score.return_value = {
                "station_id": "S1",
                "obs_ts": "2024-06-15T09:00:00",
                "risk_score": 0.75,
                "risk_label": 1,
                "threshold": 0.5,
                "snapshot_source_ts": "2024-06-15T08:59:00",
                "model_name": "test_model.lgbm",
                "features": {"ft_bikes_available": 10.0},
            }
            resp = client.post(
                "/score?debug=true",
                json={"station_id": "S1", "obs_ts": "2024-06-15T09:00:00"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "features" in data

    def test_missing_station_id_returns_422(self, client):
        """Missing required field returns validation error."""
        resp = client.post("/score", json={"obs_ts": "2024-06-15T09:00:00"})
        assert resp.status_code == 422

    def test_scoring_error_propagates(self, client):
        """ScoringError from serve layer returns correct HTTP status."""
        from mobility_feature_pipeline.serve import ScoringError

        with patch("mobility_feature_pipeline.server.score") as mock_score:
            mock_score.side_effect = ScoringError("station_not_found", "No data", status=404)
            resp = client.post(
                "/score",
                json={"station_id": "FAKE", "obs_ts": "2024-06-15T09:00:00"},
            )
            assert resp.status_code == 404
            data = resp.json()
            assert data["error"] == "station_not_found"
