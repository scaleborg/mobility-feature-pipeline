-- rolling: window aggregates over prior 60 min and 24 hours.
-- All windows are strictly bounded: [obs_ts - W, obs_ts].
rolling AS (
    SELECT
        g.station_id,
        g.obs_ts,

        -- 60-minute rolling aggregates
        (SELECT AVG(r.avg_bikes_available)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 60 MINUTE AND g.obs_ts
        ) AS ft_avg_bikes_60m,

        (SELECT MIN(r.avg_bikes_available)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 60 MINUTE AND g.obs_ts
        ) AS ft_min_bikes_60m,

        (SELECT MAX(r.avg_bikes_available)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 60 MINUTE AND g.obs_ts
        ) AS ft_max_bikes_60m,

        -- 60-minute rolling availability ratio
        (SELECT AVG(r.avg_availability_ratio)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 60 MINUTE AND g.obs_ts
        ) AS ft_avg_ratio_60m,

        -- 24-hour rolling aggregates
        (SELECT AVG(r.avg_bikes_available)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 24 HOUR AND g.obs_ts
        ) AS ft_avg_bikes_24h,

        (SELECT MIN(r.avg_bikes_available)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 24 HOUR AND g.obs_ts
        ) AS ft_min_bikes_24h,

        (SELECT MAX(r.avg_bikes_available)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 24 HOUR AND g.obs_ts
        ) AS ft_max_bikes_24h,

        -- 24-hour trailing low-availability frequency
        (SELECT SUM(r.low_availability_events)::DOUBLE / NULLIF(SUM(r.event_count), 0)
         FROM raw_station_metrics_1min r
         WHERE r.station_id = g.station_id
           AND r.window_start BETWEEN g.obs_ts - INTERVAL 24 HOUR AND g.obs_ts
        ) AS ft_low_avail_freq_24h

    FROM obs_grid g
)
