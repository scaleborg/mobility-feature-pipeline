-- snapshot: point-in-time features from the nearest 1-min row at or before obs_ts.
snapshot AS (
    SELECT
        g.station_id,
        g.obs_ts,
        m.avg_bikes_available   AS ft_bikes_available,
        m.avg_docks_available   AS ft_docks_available,
        m.avg_availability_ratio AS ft_availability_ratio,
        m.avg_capacity          AS ft_capacity
    FROM obs_grid g
    ASOF JOIN raw_station_metrics_1min m
        ON g.station_id = m.station_id
       AND m.window_start <= g.obs_ts
)
