-- build_dataset: orchestrates all CTEs into the final supervised training dataset.
-- Each CTE is included from its own SQL file for maintainability.

WITH

{obs_grid},

{snapshot},

{lags},

{rolling},

{temporal},

{forward_label}

SELECT
    -- identity
    s.station_id,
    s.obs_ts,

    -- metadata
    s.obs_ts                          AS feature_cutoff_ts,
    fl.label_window_end,

    -- target
    fl.target_empty_next_hour,

    -- snapshot features
    s.ft_bikes_available,
    s.ft_docks_available,
    s.ft_availability_ratio,

    -- lag features
    l15.ft_bikes_available_lag_15m,
    l30.ft_bikes_available_lag_30m,
    l60.ft_bikes_available_lag_60m,
    l24.ft_bikes_available_lag_24h,

    -- rolling features
    r.ft_avg_bikes_60m,
    r.ft_min_bikes_60m,
    r.ft_max_bikes_60m,
    r.ft_avg_bikes_24h,
    r.ft_min_bikes_24h,
    r.ft_max_bikes_24h,
    r.ft_avg_ratio_60m,

    -- trailing event frequency
    r.ft_low_avail_freq_24h,

    -- temporal features
    t.ft_hour_of_day,
    t.ft_day_of_week,
    t.ft_is_weekend,

    -- capacity context features
    s.ft_capacity,
    s.ft_bikes_available / NULLIF(s.ft_capacity, 0) AS ft_pct_bikes_of_capacity,
    s.ft_docks_available / NULLIF(s.ft_capacity, 0) AS ft_pct_docks_of_capacity,
    s.ft_bikes_available - r.ft_avg_bikes_60m        AS ft_bikes_delta_60m

FROM snapshot s
INNER JOIN forward_label fl
    ON s.station_id = fl.station_id AND s.obs_ts = fl.obs_ts
LEFT JOIN lag_15m l15
    ON s.station_id = l15.station_id AND s.obs_ts = l15.obs_ts
LEFT JOIN lag_30m l30
    ON s.station_id = l30.station_id AND s.obs_ts = l30.obs_ts
LEFT JOIN lag_60m l60
    ON s.station_id = l60.station_id AND s.obs_ts = l60.obs_ts
LEFT JOIN lag_24h l24
    ON s.station_id = l24.station_id AND s.obs_ts = l24.obs_ts
LEFT JOIN rolling r
    ON s.station_id = r.station_id AND s.obs_ts = r.obs_ts
LEFT JOIN temporal t
    ON s.station_id = t.station_id AND s.obs_ts = t.obs_ts
ORDER BY s.obs_ts, s.station_id
