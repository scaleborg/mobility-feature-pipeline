"""Pipeline orchestrator: reads upstream DuckDB, runs feature SQL, writes Parquet."""

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from mobility_feature_pipeline.config import SQL_DIR, Settings


def _read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def _build_query(settings: Settings) -> str:
    """Compose the full dataset query from SQL fragments."""
    params = {
        "sample_interval": settings.sample_interval_min,
        "min_capacity": settings.min_capacity,
        "empty_threshold": settings.empty_threshold,
        "min_forward_rows": settings.min_forward_rows,
    }

    fragments = {
        "obs_grid": _read_sql("obs_grid.sql").format(**params),
        "snapshot": _read_sql("snapshot.sql"),
        "lags": _read_sql("lags.sql"),
        "rolling": _read_sql("rolling.sql"),
        "temporal": _read_sql("temporal.sql"),
        "forward_label": _read_sql("forward_label.sql").format(**params),
    }

    template = _read_sql("build_dataset.sql")
    return template.format(**fragments)


def build_dataset(
    db_path: Path,
    output_dir: Path,
    settings: Settings | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Build the supervised training dataset and write to Parquet.

    Returns the output path, or None if dry_run.
    """
    if settings is None:
        settings = Settings()

    query = _build_query(settings)

    if dry_run:
        print("── SQL Query ──")
        print(query)
        print()
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            count = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
            print(f"Estimated rows: {count:,}")
        finally:
            con.close()
        return None

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        arrow_table = con.execute(query).arrow().read_all()
    finally:
        con.close()

    row_count = arrow_table.num_rows
    if row_count == 0:
        print("WARNING: Query returned 0 rows. No output written.")
        return None

    target_col = arrow_table.column("target_empty_next_hour")
    positive_count = sum(1 for v in target_col if v.as_py() == 1)
    positive_rate = positive_count / row_count

    station_count = len(set(v.as_py() for v in arrow_table.column("station_id")))
    obs_ts_col = arrow_table.column("obs_ts")
    obs_ts_min = min(v.as_py() for v in obs_ts_col)
    obs_ts_max = max(v.as_py() for v in obs_ts_col)

    build_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    metadata = {
        b"upstream_db_path": str(db_path).encode(),
        b"build_timestamp": build_ts.encode(),
        b"row_count": str(row_count).encode(),
        b"positive_rate": f"{positive_rate:.4f}".encode(),
        b"station_count": str(station_count).encode(),
        b"obs_ts_min": str(obs_ts_min).encode(),
        b"obs_ts_max": str(obs_ts_max).encode(),
    }

    existing_meta = arrow_table.schema.metadata or {}
    merged_meta = {**existing_meta, **metadata}
    arrow_table = arrow_table.replace_schema_metadata(merged_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"training_dataset_{build_ts}.parquet"
    pq.write_table(arrow_table, output_path, compression="snappy")

    print("── Dataset Built ──")
    print(f"Output:        {output_path}")
    print(f"Rows:          {row_count:,}")
    print(f"Stations:      {station_count:,}")
    print(f"Time range:    {obs_ts_min} → {obs_ts_max}")
    print(f"Positive rate: {positive_rate:.1%} ({positive_count:,} / {row_count:,})")

    return output_path
