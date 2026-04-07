-- obs_grid: 15-minute aligned observation timeline per station.
-- Filters out small-capacity / test stations.
obs_grid AS (
    SELECT DISTINCT
        station_id,
        time_bucket(INTERVAL '{sample_interval} minutes', window_start) AS obs_ts
    FROM raw_station_metrics_1min
    WHERE avg_capacity >= {min_capacity}
)
