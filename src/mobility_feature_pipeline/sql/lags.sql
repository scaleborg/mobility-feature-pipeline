-- lags: explicit point-in-time lag features via ASOF JOIN.
-- Each lag finds the nearest 1-min row at or before (obs_ts - offset).
lag_15m AS (
    SELECT g.station_id, g.obs_ts,
           m.avg_bikes_available AS ft_bikes_available_lag_15m
    FROM obs_grid g
    ASOF JOIN raw_station_metrics_1min m
        ON g.station_id = m.station_id
       AND m.window_start <= g.obs_ts - INTERVAL 15 MINUTE
),

lag_30m AS (
    SELECT g.station_id, g.obs_ts,
           m.avg_bikes_available AS ft_bikes_available_lag_30m
    FROM obs_grid g
    ASOF JOIN raw_station_metrics_1min m
        ON g.station_id = m.station_id
       AND m.window_start <= g.obs_ts - INTERVAL 30 MINUTE
),

lag_60m AS (
    SELECT g.station_id, g.obs_ts,
           m.avg_bikes_available AS ft_bikes_available_lag_60m
    FROM obs_grid g
    ASOF JOIN raw_station_metrics_1min m
        ON g.station_id = m.station_id
       AND m.window_start <= g.obs_ts - INTERVAL 60 MINUTE
),

lag_24h AS (
    SELECT g.station_id, g.obs_ts,
           m.avg_bikes_available AS ft_bikes_available_lag_24h
    FROM obs_grid g
    ASOF JOIN raw_station_metrics_1min m
        ON g.station_id = m.station_id
       AND m.window_start <= g.obs_ts - INTERVAL 24 HOUR
)
