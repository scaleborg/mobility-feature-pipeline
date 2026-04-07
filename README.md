# mobility-feature-pipeline

Predicts which Citi Bike stations are likely to run empty in the next hour, then ranks which ones to address first. Built on real-time 1-minute station data from NYC.

A production-style ML pipeline covering dataset generation with point-in-time feature engineering, LightGBM model training, a per-station online scoring API, and a batch triage layer that returns a ranked shortlist for rebalancing.

---

## Pipeline flow

Pipeline

---

## Slice progression

Slice | What it delivers
--- | ---
1 | Supervised dataset pipeline — 22 features, binary label, temporal sampling, validation CLI
2 | Baseline + LightGBM training with temporal split (70/15/15), saved model artifacts
3 | Online scoring API — real-time feature reconstruction, staleness protection, out-of-domain rejection
4 | Rebalancing triage layer — batch-scores all stations, returns ranked shortlist by empty-risk

---

## Upstream dependency

Reads (read-only) from the DuckDB produced by urban-mobility-control-tower:

../urban-mobility-control-tower/analytics/data/mobility.duckdb

Source table: raw_station_metrics_1min — 1-minute Flink tumbling window aggregates of Citi Bike NYC GBFS data.

---

## Target definition

target_empty_next_hour (binary: 0 or 1)

At observation time t, the label is 1 if any 1-minute row for that station in (t, t + 60 min] has avg_bikes_available < 1.0.

This captures whether a stockout happens at any point during the next hour — not just the state at t + 60 min.

---

## Feature list (22 features)

Features capture three layers of signal: the station's current state (snapshot), short-term dynamics (lags and rolling windows), and contextual signals such as time and station capacity.

Grouped notation expands to 22 distinct feature columns.

Group | Features
--- | ---
Snapshot (3) | ft_bikes_available, ft_docks_available, ft_availability_ratio
Lags (4) | ft_bikes_available_lag_15m, ft_bikes_available_lag_30m, ft_bikes_available_lag_60m, ft_bikes_available_lag_24h
Rolling (7) | ft_avg_bikes_60m, ft_min_bikes_60m, ft_max_bikes_60m, ft_avg_bikes_24h, ft_min_bikes_24h, ft_max_bikes_24h, ft_avg_ratio_60m
Trailing event (1) | ft_low_avail_freq_24h
Temporal (3) | ft_hour_of_day, ft_day_of_week, ft_is_weekend
Context (4) | ft_capacity, ft_pct_bikes_of_capacity, ft_pct_docks_of_capacity, ft_bikes_delta_60m

All features are computed strictly from data available at or before the observation timestamp (no leakage).

---

## Quick start

make install          # Install dependencies  
make test             # Run all tests (59 tests across Slices 1–4)

---

## Dataset + training pipeline

make build            # Build dataset from upstream DuckDB  
make validate         # Validate the built dataset  
make train            # Train LightGBM + baselines, save artifacts to models/  
make evaluate         # Re-load saved model, reproduce test metrics  
make slice2           # End-to-end: build → validate → train  

---

## Scoring API

make serve            # Start FastAPI scoring server on :8000

### Single-station scoring

curl -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"station_id": "4025", "obs_ts": "2026-04-04T10:09:00"}'

---

## Triage API

Batch-scores all in-domain stations at a timestamp and returns a ranked shortlist:

# Via API
curl -X POST http://localhost:8000/triage \
  -H 'Content-Type: application/json' \
  -d '{"obs_ts": "2026-04-04T10:09:00", "top_n": 10}'

# Via CLI
make triage OBS_TS="2026-04-04 10:09:00" TOP_N=10

Response includes funnel counts (candidate_stations → scored + skipped → top N returned) and skip reasons for operational transparency.

Add ?debug=true for per-station diagnostics.

---

## CLI commands

mobility-feature-pipeline build       --db-path <path> [--output-dir ./output] [--dry-run]  
mobility-feature-pipeline validate    --parquet-path <path>  
mobility-feature-pipeline train       --parquet-path <path> [--output-dir ./models]  
mobility-feature-pipeline evaluate    --parquet-path <path> --model-path <path>  
mobility-feature-pipeline serve       --model-path <path> --db-path <path> [--port 8000]  
mobility-feature-pipeline triage      --model-path <path> --db-path <path> --obs-ts <ts> [--top-n 10]  
mobility-feature-pipeline attrition   --db-path <path>  
mobility-feature-pipeline sensitivity --db-path <path>  
mobility-feature-pipeline inspect     --db-path <path> --station-id <id> --start <ts> --end <ts>  

---

## Output artifacts

output/ — Parquet training datasets with embedded metadata  
models/ — LightGBM .lgbm models, metrics JSON, test predictions Parquet  

---

## About

P2 — Feature Pipeline for real-time mobility ML system (point-in-time feature engineering)
