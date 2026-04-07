-- active_stations: coarse pre-filter for candidate stations.
-- Returns station IDs that plausibly have recent data near obs_ts
-- and meet the capacity floor.
-- This is a performance optimization only — serve.reconstruct_features()
-- remains the source of truth for domain/staleness validation.
SELECT DISTINCT station_id
FROM raw_station_metrics_1min
WHERE avg_capacity >= $min_capacity
  AND window_start BETWEEN $obs_ts - INTERVAL 15 MINUTE AND $obs_ts
ORDER BY station_id
