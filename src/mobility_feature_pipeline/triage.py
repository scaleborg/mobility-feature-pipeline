"""Batch scoring and ranking for station rebalancing triage."""

from datetime import datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

from mobility_feature_pipeline.config import SQL_DIR, Settings
from mobility_feature_pipeline.model_config import (
    DEFAULT_TOP_N,
    FEATURE_COLUMNS,
    SCORE_THRESHOLD,
)
from mobility_feature_pipeline.serve import ScoringError, reconstruct_features

_ACTIVE_SQL = (SQL_DIR / "active_stations.sql").read_text()


def _discover_candidates(
    con: duckdb.DuckDBPyConnection,
    obs_ts: datetime,
    min_capacity: float,
) -> list[str]:
    """Run coarse pre-filter to find candidate station IDs."""
    df = con.execute(_ACTIVE_SQL, {"obs_ts": obs_ts, "min_capacity": min_capacity}).fetchdf()
    return df["station_id"].tolist()


def batch_score(
    booster: lgb.Booster,
    db_path: Path,
    obs_ts: datetime,
    top_n: int = DEFAULT_TOP_N,
    debug: bool = False,
    _con_override: duckdb.DuckDBPyConnection | None = None,
) -> dict:
    """Score all candidate stations at obs_ts and return a ranked shortlist.

    Flow: discover candidates → reconstruct via Slice 3 → score → rank → return.
    """
    settings = Settings()

    if _con_override is not None:
        con = _con_override
        should_close = False
    else:
        con = duckdb.connect(str(db_path), read_only=True)
        should_close = True

    try:
        candidates = _discover_candidates(con, obs_ts, settings.min_capacity)

        if not candidates:
            raise ScoringError(
                "no_active_stations",
                f"No stations with recent data at {obs_ts}",
                status=404,
            )

        # Reconstruct features for each candidate; collect successes and failures
        scored_rows = []
        skipped_stations = []

        for sid in candidates:
            try:
                result = reconstruct_features(
                    db_path=None,
                    station_id=sid,
                    obs_ts=obs_ts,
                    _con_override=con,
                )
                scored_rows.append(
                    {
                        "station_id": sid,
                        "features": result["features"],
                        "snapshot_source_ts": result["snapshot_source_ts"],
                    }
                )
            except ScoringError as e:
                skipped_stations.append({"station_id": sid, "reason": e.code})

        n_scored = len(scored_rows)
        n_skipped = len(skipped_stations)

        if n_scored == 0:
            return {
                "obs_ts": obs_ts.isoformat(),
                "model_name": None,
                "candidate_stations": len(candidates),
                "scored": 0,
                "skipped": n_skipped,
                "top_n": top_n,
                "stations": [],
                "skipped_stations": skipped_stations,
            }

        # Build single DataFrame for vectorized prediction
        feature_matrix = []
        for row in scored_rows:
            feature_matrix.append(
                [
                    float("nan") if row["features"][col] is None else row["features"][col]
                    for col in FEATURE_COLUMNS
                ]
            )
        df = pd.DataFrame(feature_matrix, columns=FEATURE_COLUMNS)

        scores = booster.predict(df)

        # Assemble result records
        station_results = []
        for i, row in enumerate(scored_rows):
            risk_score = round(float(scores[i]), 6)
            bikes = row["features"]["ft_bikes_available"]
            entry = {
                "rank": 0,  # set after sorting
                "station_id": row["station_id"],
                "risk_score": risk_score,
                "risk_label": int(risk_score >= SCORE_THRESHOLD),
                "bikes_available": bikes,
                "capacity": row["features"]["ft_capacity"],
                "snapshot_source_ts": row["snapshot_source_ts"].isoformat(),
            }
            if debug:
                snapshot_age = obs_ts - row["snapshot_source_ts"]
                entry["debug"] = {
                    "availability_ratio": row["features"]["ft_availability_ratio"],
                    "snapshot_age_min": round(snapshot_age.total_seconds() / 60, 1),
                    "bikes_delta_60m": row["features"]["ft_bikes_delta_60m"],
                }
            station_results.append(entry)

        # Rank: risk_score desc, bikes_available asc, station_id asc
        station_results.sort(
            key=lambda s: (-s["risk_score"], s["bikes_available"] or 0, s["station_id"])
        )

        # Truncate and assign ranks
        station_results = station_results[:top_n]
        for i, entry in enumerate(station_results):
            entry["rank"] = i + 1

        return {
            "obs_ts": obs_ts.isoformat(),
            "model_name": None,  # set by caller
            "candidate_stations": len(candidates),
            "scored": n_scored,
            "skipped": n_skipped,
            "top_n": top_n,
            "stations": station_results,
            "skipped_stations": skipped_stations,
        }
    finally:
        if should_close:
            con.close()


def print_triage_report(result: dict, model_name: str) -> None:
    """Print a formatted triage report to stdout."""
    print()
    print("── Triage Report ──")
    print(f"Timestamp:   {result['obs_ts']}")
    print(f"Model:       {model_name}")
    print(f"Candidates:  {result['candidate_stations']} stations")
    print(f"Scored:      {result['scored']} stations")

    # Summarise skip reasons
    skipped = result["skipped_stations"]
    if skipped:
        reason_counts: dict[str, int] = {}
        for s in skipped:
            reason_counts[s["reason"]] = reason_counts.get(s["reason"], 0) + 1
        reason_parts = [f"{v} {k}" for k, v in sorted(reason_counts.items())]
        print(f"Skipped:     {result['skipped']} stations ({', '.join(reason_parts)})")
    else:
        print(f"Skipped:     {result['skipped']} stations")

    print()

    stations = result["stations"]
    if not stations:
        print("  No stations scored successfully.")
        return

    print(
        f"{'Rank':>4}  {'Station':<10}  {'Risk Score':>10}  {'Label':>5}  "
        f"{'Bikes':>5}  {'Capacity':>8}  {'Snapshot':>10}"
    )

    for s in stations:
        label = "HIGH" if s["risk_label"] == 1 else "low"
        bikes = f"{s['bikes_available']:.1f}" if s["bikes_available"] is not None else "—"
        cap = f"{s['capacity']:.1f}" if s["capacity"] is not None else "—"
        # Show just the time portion of snapshot_source_ts
        snap_ts = s["snapshot_source_ts"]
        if "T" in snap_ts:
            snap_ts = snap_ts.split("T")[1][:8]
        print(
            f"{s['rank']:>4}  {s['station_id']:<10}  {s['risk_score']:>10.4f}  "
            f"{label:>5}  {bikes:>5}  {cap:>8}  {snap_ts:>10}"
        )

    print()
