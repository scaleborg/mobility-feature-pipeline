-- temporal: time-of-day and day-of-week features extracted from obs_ts.
temporal AS (
    SELECT
        station_id,
        obs_ts,
        EXTRACT(HOUR FROM obs_ts)::INT   AS ft_hour_of_day,
        EXTRACT(DOW FROM obs_ts)::INT    AS ft_day_of_week,
        CASE WHEN EXTRACT(DOW FROM obs_ts) IN (0, 6) THEN 1 ELSE 0 END::INT AS ft_is_weekend
    FROM obs_grid
)
