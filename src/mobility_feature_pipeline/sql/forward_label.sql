-- forward_label: binary target — did the station experience an empty-risk event
-- at any point in the next 60 minutes?
-- Label = 1 if any 1-min row in (obs_ts, obs_ts + 60 min] has avg_bikes_available < threshold.
-- Observations with insufficient forward coverage are excluded.
forward_label AS (
    SELECT
        g.station_id,
        g.obs_ts,
        g.obs_ts + INTERVAL 60 MINUTE AS label_window_end,
        CASE WHEN MIN(m.avg_bikes_available) < {empty_threshold} THEN 1 ELSE 0 END::INT
            AS target_empty_next_hour,
        COUNT(*) AS fwd_row_count
    FROM obs_grid g
    INNER JOIN raw_station_metrics_1min m
        ON m.station_id = g.station_id
       AND m.window_start > g.obs_ts
       AND m.window_start <= g.obs_ts + INTERVAL 60 MINUTE
    GROUP BY g.station_id, g.obs_ts
    HAVING COUNT(*) >= {min_forward_rows}
)
