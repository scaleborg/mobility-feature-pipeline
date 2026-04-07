"""Feature reconstruction and model scoring for online inference."""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

from mobility_feature_pipeline.config import SQL_DIR
from mobility_feature_pipeline.model_config import (
    FEATURE_COLUMNS,
    MAX_SNAPSHOT_STALENESS_MIN,
    MIN_CAPACITY,
    SCORE_THRESHOLD,
)

_SCORE_SQL = (SQL_DIR / "score_features.sql").read_text()


class ScoringError(Exception):
    def __init__(self, code: str, detail: str, status: int = 422):
        self.code = code
        self.detail = detail
        self.status = status


def load_model(model_path: Path) -> lgb.Booster:
    """Load a saved LightGBM model. Raises FileNotFoundError if missing."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return lgb.Booster(model_file=str(model_path))


def reconstruct_features(
    db_path: Path | None,
    station_id: str,
    obs_ts: datetime,
    _con_override=None,
) -> dict:
    """Reconstruct features for a single (station_id, obs_ts) from raw DuckDB data.

    Returns dict with 'snapshot_source_ts' and all FEATURE_COLUMNS values.
    Raises ScoringError for domain/staleness/missing-data issues.
    """
    if _con_override is not None:
        result = _con_override.execute(
            _SCORE_SQL, {"station_id": station_id, "obs_ts": obs_ts}
        ).fetchdf()
    else:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            result = con.execute(_SCORE_SQL, {"station_id": station_id, "obs_ts": obs_ts}).fetchdf()
        finally:
            con.close()

    # No row returned: ASOF JOIN found nothing → station not in DB
    if result.empty:
        raise ScoringError(
            "station_not_found",
            f"No raw data found for station '{station_id}'",
            status=404,
        )

    row = result.iloc[0]

    # Snapshot source is NULL: station exists but no data at or before obs_ts
    snapshot_source_ts = row["snapshot_source_ts"]
    if pd.isna(snapshot_source_ts):
        raise ScoringError(
            "no_snapshot",
            f"No data at or before {obs_ts} for station '{station_id}'",
            status=404,
        )

    snapshot_source_ts = pd.Timestamp(snapshot_source_ts).to_pydatetime()

    # Domain check: capacity < MIN_CAPACITY
    capacity = row["ft_capacity"]
    if capacity is not None and capacity < MIN_CAPACITY:
        raise ScoringError(
            "out_of_domain",
            f"Station capacity ({capacity:.1f}) is below training threshold "
            f"({MIN_CAPACITY}); prediction unreliable",
        )

    # Staleness check
    obs_ts_pd = pd.Timestamp(obs_ts)
    source_ts_pd = pd.Timestamp(snapshot_source_ts)
    staleness = obs_ts_pd - source_ts_pd
    max_staleness = timedelta(minutes=MAX_SNAPSHOT_STALENESS_MIN)
    if staleness > max_staleness:
        raise ScoringError(
            "stale_data",
            f"Latest data for station is from {snapshot_source_ts}, "
            f"which is >{MAX_SNAPSHOT_STALENESS_MIN} min before requested obs_ts {obs_ts}",
        )

    # Extract features in canonical FEATURE_COLUMNS order
    features = {}
    for col in FEATURE_COLUMNS:
        val = row[col]
        features[col] = None if pd.isna(val) else float(val)

    return {
        "snapshot_source_ts": snapshot_source_ts,
        "features": features,
    }


def score(
    booster: lgb.Booster,
    db_path: Path,
    station_id: str,
    obs_ts: datetime,
    debug: bool = False,
) -> dict:
    """Reconstruct features, validate, score, and return response dict."""
    result = reconstruct_features(db_path, station_id, obs_ts)

    # Build feature vector in FEATURE_COLUMNS order for model input
    # Use float('nan') instead of None so pandas creates numeric columns (LightGBM requirement)
    feature_values = [
        float("nan") if result["features"][col] is None else result["features"][col]
        for col in FEATURE_COLUMNS
    ]
    df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    risk_score = float(booster.predict(df)[0])
    risk_label = int(risk_score >= SCORE_THRESHOLD)

    response = {
        "station_id": station_id,
        "obs_ts": obs_ts.isoformat() if isinstance(obs_ts, datetime) else str(obs_ts),
        "risk_score": round(risk_score, 6),
        "risk_label": risk_label,
        "threshold": SCORE_THRESHOLD,
        "snapshot_source_ts": result["snapshot_source_ts"].isoformat(),
        "model_name": None,  # set by caller
    }

    if debug:
        response["features"] = result["features"]

    return response
