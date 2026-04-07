"""Shared fixtures: in-memory DuckDB with synthetic 1-min station data."""

from datetime import datetime, timedelta

import duckdb
import pytest


def _insert_rows(
    con,
    station_id: str,
    start: datetime,
    minutes: int,
    bikes: float,
    docks: float = 20.0,
    capacity: float = 30.0,
):
    """Insert `minutes` consecutive 1-min rows for a station."""
    for i in range(minutes):
        ws = start + timedelta(minutes=i)
        we = ws + timedelta(minutes=1)
        ratio = bikes / capacity if capacity > 0 else 0.0
        low_events = 1 if ratio < 0.2 else 0
        con.execute(
            """
            INSERT INTO raw_station_metrics_1min VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                station_id,
                ws,
                we,
                bikes,
                docks,
                capacity,
                ratio,
                low_events,
                1,
                we,
            ],
        )


@pytest.fixture
def synthetic_db():
    """Create an in-memory DuckDB with known patterns for testing.

    Station layout:
    - "S1": 3 hours of data. bikes=10 for first 90 min, drops to 0 for next 90 min.
           This creates a clear stockout event at the 90-min mark.
    - "S2": 3 hours of data. bikes=15 throughout (never empty — always label=0
           for observations with sufficient forward coverage).
    - "S3": 3 hours of data. capacity=3 (should be excluded — test station).
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE raw_station_metrics_1min (
            station_id       VARCHAR,
            window_start     TIMESTAMP,
            window_end       TIMESTAMP,
            avg_bikes_available   DOUBLE,
            avg_docks_available   DOUBLE,
            avg_capacity          DOUBLE,
            avg_availability_ratio DOUBLE,
            low_availability_events BIGINT,
            event_count           BIGINT,
            emitted_at            TIMESTAMP
        )
    """)

    base = datetime(2024, 6, 15, 8, 0, 0)  # Saturday 8:00 AM

    # S1: bikes=10 for 90 min, then bikes=0 for 90 min
    _insert_rows(con, "S1", base, 90, bikes=10.0, docks=20.0, capacity=30.0)
    _insert_rows(con, "S1", base + timedelta(minutes=90), 90, bikes=0.0, docks=30.0, capacity=30.0)

    # S2: bikes=15 throughout (3 hours)
    _insert_rows(con, "S2", base, 180, bikes=15.0, docks=15.0, capacity=30.0)

    # S3: tiny capacity station (should be excluded)
    _insert_rows(con, "S3", base, 180, bikes=2.0, docks=1.0, capacity=3.0)

    yield con
    con.close()
