"""Diagnostic commands for Slice 1 dataset quality review."""

from datetime import datetime
from pathlib import Path

import duckdb

from mobility_feature_pipeline.config import Settings


def station_attrition(db_path: Path, settings: Settings | None = None) -> None:
    """Report station attrition through each pipeline filter stage."""
    if settings is None:
        settings = Settings()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        _run_attrition(con, settings)
    finally:
        con.close()


def _run_attrition(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    # Stage 1: all stations in source
    total = con.execute(
        "SELECT COUNT(DISTINCT station_id) FROM raw_station_metrics_1min"
    ).fetchone()[0]

    # Stage 2: after capacity filter
    after_capacity = con.execute(
        "SELECT COUNT(DISTINCT station_id) FROM raw_station_metrics_1min WHERE avg_capacity >= ?",
        [settings.min_capacity],
    ).fetchone()[0]

    # Stage 3: stations that appear in the obs_grid AND have sufficient forward coverage
    # First get obs_grid stations (same as after_capacity, but explicit)
    # Then check which survive the forward_label HAVING clause
    after_forward = con.execute(
        f"""
        WITH obs_grid AS (
            SELECT DISTINCT
                station_id,
                time_bucket(INTERVAL '{settings.sample_interval_min} minutes', window_start) AS obs_ts
            FROM raw_station_metrics_1min
            WHERE avg_capacity >= ?
        ),
        fwd AS (
            SELECT
                g.station_id,
                g.obs_ts,
                COUNT(*) AS fwd_row_count
            FROM obs_grid g
            INNER JOIN raw_station_metrics_1min m
                ON m.station_id = g.station_id
               AND m.window_start > g.obs_ts
               AND m.window_start <= g.obs_ts + INTERVAL 60 MINUTE
            GROUP BY g.station_id, g.obs_ts
            HAVING COUNT(*) >= ?
        )
        SELECT COUNT(DISTINCT station_id) FROM fwd
        """,
        [settings.min_capacity, settings.min_forward_rows],
    ).fetchone()[0]

    # Forward row count distribution (for observations that DO have forward rows)
    fwd_dist = con.execute(
        f"""
        WITH obs_grid AS (
            SELECT DISTINCT
                station_id,
                time_bucket(INTERVAL '{settings.sample_interval_min} minutes', window_start) AS obs_ts
            FROM raw_station_metrics_1min
            WHERE avg_capacity >= ?
        ),
        fwd_counts AS (
            SELECT
                g.station_id,
                g.obs_ts,
                COUNT(*) AS fwd_row_count
            FROM obs_grid g
            INNER JOIN raw_station_metrics_1min m
                ON m.station_id = g.station_id
               AND m.window_start > g.obs_ts
               AND m.window_start <= g.obs_ts + INTERVAL 60 MINUTE
            GROUP BY g.station_id, g.obs_ts
        )
        SELECT
            COUNT(*)                          AS total_obs,
            SUM(CASE WHEN fwd_row_count >= ? THEN 1 ELSE 0 END) AS passing_obs,
            MIN(fwd_row_count)                AS min_fwd,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY fwd_row_count) AS p25_fwd,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY fwd_row_count) AS p50_fwd,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY fwd_row_count) AS p75_fwd,
            MAX(fwd_row_count)                AS max_fwd
        FROM fwd_counts
        """,
        [settings.min_capacity, settings.min_forward_rows],
    ).fetchone()

    total_obs, passing_obs, min_fwd, p25, p50, p75, max_fwd = fwd_dist

    # Observations with zero forward rows (tail of dataset)
    zero_fwd = con.execute(
        f"""
        WITH obs_grid AS (
            SELECT DISTINCT
                station_id,
                time_bucket(INTERVAL '{settings.sample_interval_min} minutes', window_start) AS obs_ts
            FROM raw_station_metrics_1min
            WHERE avg_capacity >= ?
        )
        SELECT COUNT(*) FROM obs_grid g
        WHERE NOT EXISTS (
            SELECT 1 FROM raw_station_metrics_1min m
            WHERE m.station_id = g.station_id
              AND m.window_start > g.obs_ts
              AND m.window_start <= g.obs_ts + INTERVAL 60 MINUTE
        )
        """,
        [settings.min_capacity],
    ).fetchone()[0]

    total_grid = total_obs + zero_fwd

    print("── Station Attrition Report ──")
    print(f"Source stations:              {total:,}")
    print(
        f"After capacity >= {settings.min_capacity}:        {after_capacity:,}  "
        f"(dropped {total - after_capacity:,})"
    )
    print(
        f"After forward coverage >= {settings.min_forward_rows}: {after_forward:,}  "
        f"(dropped {after_capacity - after_forward:,})"
    )
    print()
    print("── Observation Attrition ──")
    print(f"Total obs in grid:           {total_grid:,}")
    print(f"  with zero forward rows:    {zero_fwd:,}  (dataset tail, always excluded)")
    print(f"  with some forward rows:    {total_obs:,}")
    print(
        f"  passing min_forward_rows:  {passing_obs:,}  "
        f"({passing_obs / total_grid * 100:.1f}% of grid)"
    )
    print()
    print("── Forward Row Count Distribution ──")
    print(f"  min:  {min_fwd:.0f}")
    print(f"  p25:  {p25:.0f}")
    print(f"  p50:  {p50:.0f}")
    print(f"  p75:  {p75:.0f}")
    print(f"  max:  {max_fwd:.0f}")
    print(f"  threshold: {settings.min_forward_rows}")
    print()
    print("── Exclusion Drivers ──")
    cap_pct = (total - after_capacity) / total * 100 if total > 0 else 0
    fwd_pct = (after_capacity - after_forward) / after_capacity * 100 if after_capacity > 0 else 0
    print(
        f"  Capacity filter removes {cap_pct:.0f}% of stations "
        f"(small/test stations with capacity < {settings.min_capacity})"
    )
    print(
        f"  Forward coverage removes {fwd_pct:.0f}% of remaining stations "
        f"(sparse 1-min data, insufficient for reliable labeling)"
    )
    if zero_fwd > 0:
        print(
            f"  Dataset tail ({zero_fwd:,} obs) excluded: "
            f"no future data exists beyond the last recorded minute"
        )


def forward_coverage_sensitivity(
    db_path: Path,
    thresholds: list[int] | None = None,
    settings: Settings | None = None,
) -> None:
    """Evaluate dataset retention and positive rate across min_forward_rows thresholds."""
    if settings is None:
        settings = Settings()
    if thresholds is None:
        thresholds = [10, 20, 30, 40, 50]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        _run_sensitivity(con, thresholds, settings)
    finally:
        con.close()


def _run_sensitivity(
    con: duckdb.DuckDBPyConnection,
    thresholds: list[int],
    settings: Settings,
) -> None:
    # Materialize the obs_grid + forward row counts + target once
    con.execute(
        f"""
        CREATE TEMP TABLE _sensitivity AS
        WITH obs_grid AS (
            SELECT DISTINCT
                station_id,
                time_bucket(INTERVAL '{settings.sample_interval_min} minutes', window_start) AS obs_ts
            FROM raw_station_metrics_1min
            WHERE avg_capacity >= {settings.min_capacity}
        ),
        fwd AS (
            SELECT
                g.station_id,
                g.obs_ts,
                COUNT(*) AS fwd_row_count,
                CASE WHEN MIN(m.avg_bikes_available) < {settings.empty_threshold}
                     THEN 1 ELSE 0 END AS target
            FROM obs_grid g
            INNER JOIN raw_station_metrics_1min m
                ON m.station_id = g.station_id
               AND m.window_start > g.obs_ts
               AND m.window_start <= g.obs_ts + INTERVAL 60 MINUTE
            GROUP BY g.station_id, g.obs_ts
        )
        SELECT * FROM fwd
        """
    )

    total_grid = con.execute("SELECT COUNT(*) FROM _sensitivity").fetchone()[0]

    print("── Forward Coverage Sensitivity ──")
    print(f"Total observations with any forward data: {total_grid:,}")
    print()

    header = (
        f"{'threshold':>9}  {'stations':>8}  {'obs':>8}  {'% grid':>7}  "
        f"{'pos_rate':>8}  {'min':>5}  {'p25':>5}  {'p50':>5}  {'p75':>5}  {'max':>5}"
    )
    print(header)
    print("─" * len(header))

    for t in sorted(thresholds):
        row = con.execute(
            """
            SELECT
                COUNT(DISTINCT station_id),
                COUNT(*),
                SUM(target),
                MIN(fwd_row_count),
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY fwd_row_count),
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY fwd_row_count),
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY fwd_row_count),
                MAX(fwd_row_count)
            FROM _sensitivity
            WHERE fwd_row_count >= ?
            """,
            [t],
        ).fetchone()

        stations, obs, pos, fwd_min, p25, p50, p75, fwd_max = row
        pct = obs / total_grid * 100 if total_grid > 0 else 0
        pos_rate = pos / obs * 100 if obs > 0 else 0

        print(
            f"{'>=' + str(t):>9}  {stations:>8,}  {obs:>8,}  {pct:>6.1f}%  "
            f"{pos_rate:>7.1f}%  {fwd_min:>5.0f}  {p25:>5.0f}  {p50:>5.0f}  {p75:>5.0f}  {fwd_max:>5.0f}"
        )

    con.execute("DROP TABLE _sensitivity")
    print()


def inspect_station(
    db_path: Path,
    station_id: str,
    start: datetime,
    end: datetime,
    settings: Settings | None = None,
) -> None:
    """Print detailed feature + raw data view for one station over a time window."""
    if settings is None:
        settings = Settings()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        _run_inspect(con, station_id, start, end, settings)
    finally:
        con.close()


def _run_inspect(
    con: duckdb.DuckDBPyConnection,
    station_id: str,
    start: datetime,
    end: datetime,
    settings: Settings,
) -> None:
    # Check station exists
    count = con.execute(
        "SELECT COUNT(*) FROM raw_station_metrics_1min WHERE station_id = ?",
        [station_id],
    ).fetchone()[0]
    if count == 0:
        print(f"Station '{station_id}' not found in raw_station_metrics_1min.")
        return

    # Build dataset query for this station + time window
    from mobility_feature_pipeline.pipeline import _build_query

    query = _build_query(settings)
    filtered = f"""
    SELECT * FROM ({query}) dataset
    WHERE station_id = ?
      AND obs_ts >= ?::TIMESTAMP
      AND obs_ts <= ?::TIMESTAMP
    ORDER BY obs_ts
    """

    rows = con.execute(filtered, [station_id, start, end]).fetchall()
    col_names = [d[0] for d in con.description]

    print(f"── Station Inspection: {station_id} ──")
    print(f"Window: {start} → {end}")
    print(f"Dataset rows in window: {len(rows)}")
    print()

    if not rows:
        print("No dataset rows found for this station/window.")
        print("Check: does this station pass capacity and forward coverage filters?")
        cap = con.execute(
            "SELECT AVG(avg_capacity) FROM raw_station_metrics_1min WHERE station_id = ?",
            [station_id],
        ).fetchone()[0]
        print(f"  Average capacity: {cap}")
        return

    for row in rows:
        row_dict = dict(zip(col_names, row))
        obs_ts = row_dict["obs_ts"]

        print(f"═══ obs_ts: {obs_ts} ═══")
        print(f"  target_empty_next_hour: {row_dict['target_empty_next_hour']}")
        print(f"  feature_cutoff_ts:      {row_dict['feature_cutoff_ts']}")
        print(f"  label_window_end:       {row_dict['label_window_end']}")
        print()

        # Snapshot features
        print("  ── Snapshot ──")
        print(f"    bikes_available:    {row_dict['ft_bikes_available']}")
        print(f"    docks_available:    {row_dict['ft_docks_available']}")
        print(f"    availability_ratio: {row_dict['ft_availability_ratio']}")
        print(f"    capacity:           {row_dict['ft_capacity']}")
        print()

        # Lag features
        print("  ── Lags ──")
        print(f"    lag_15m: {row_dict['ft_bikes_available_lag_15m']}")
        print(f"    lag_30m: {row_dict['ft_bikes_available_lag_30m']}")
        print(f"    lag_60m: {row_dict['ft_bikes_available_lag_60m']}")
        print(f"    lag_24h: {row_dict['ft_bikes_available_lag_24h']}")
        print()

        # Raw rows around obs_ts (5 before, 5 after)
        raw_around = con.execute(
            """
            SELECT window_start, avg_bikes_available, avg_docks_available,
                   avg_capacity, avg_availability_ratio
            FROM raw_station_metrics_1min
            WHERE station_id = ?
              AND window_start BETWEEN ?::TIMESTAMP - INTERVAL 5 MINUTE
                                    AND ?::TIMESTAMP + INTERVAL 5 MINUTE
            ORDER BY window_start
            """,
            [station_id, obs_ts, obs_ts],
        ).fetchall()

        print("  ── Raw rows around obs_ts (±5 min) ──")
        print(f"    {'window_start':<22} {'bikes':>6} {'docks':>6} {'cap':>6} {'ratio':>8}")
        for r in raw_around:
            marker = " ◄" if r[0] == obs_ts else ""
            ratio_str = f"{r[4]:.4f}" if r[4] is not None else "NULL"
            print(
                f"    {str(r[0]):<22} {r[1]:>6.1f} {r[2]:>6.1f} {r[3]:>6.1f} {ratio_str:>8}{marker}"
            )
        print()

        # Raw rows in forward label window
        raw_forward = con.execute(
            """
            SELECT window_start, avg_bikes_available, avg_docks_available,
                   avg_availability_ratio
            FROM raw_station_metrics_1min
            WHERE station_id = ?
              AND window_start > ?::TIMESTAMP
              AND window_start <= ?::TIMESTAMP + INTERVAL 60 MINUTE
            ORDER BY window_start
            """,
            [station_id, obs_ts, obs_ts],
        ).fetchall()

        min_bikes_fwd = min((r[1] for r in raw_forward), default=None)
        print(f"  ── Forward label window ({len(raw_forward)} rows) ──")
        print(f"    min(bikes_available): {min_bikes_fwd}")
        print(
            f"    target justification: "
            f"{'min < 1.0 → label=1' if min_bikes_fwd is not None and min_bikes_fwd < 1.0 else 'min >= 1.0 → label=0'}"
        )

        # Show first/last few rows of forward window
        show_rows = (
            raw_forward[:3] + ([("...",)] if len(raw_forward) > 6 else []) + raw_forward[-3:]
        )
        if len(raw_forward) <= 6:
            show_rows = raw_forward
        print(f"    {'window_start':<22} {'bikes':>6} {'docks':>6} {'ratio':>8}")
        for r in show_rows:
            if r[0] == "...":
                print(f"    {'...':^44}")
                continue
            ratio_str = f"{r[3]:.4f}" if r[3] is not None else "NULL"
            lo = " ◄ LOW" if r[1] < settings.empty_threshold else ""
            print(f"    {str(r[0]):<22} {r[1]:>6.1f} {r[2]:>6.1f} {ratio_str:>8}{lo}")
        print()
