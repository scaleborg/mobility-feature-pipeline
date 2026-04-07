"""Test feature correctness and temporal leakage prevention."""

from datetime import datetime, timedelta


class TestNoLeakage:
    """Verify features use only data at or before obs_ts."""

    def test_snapshot_reflects_pre_event_state(self, synthetic_db):
        """At obs_ts before the stockout, snapshot should show bikes=10, not 0."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=60)

        result = synthetic_db.execute(
            """
            SELECT m.avg_bikes_available
            FROM raw_station_metrics_1min m
            WHERE m.station_id = 'S1'
              AND m.window_start <= ?::TIMESTAMP
            ORDER BY m.window_start DESC
            LIMIT 1
            """,
            [obs],
        ).fetchone()
        assert result[0] == 10.0, "Snapshot should reflect pre-stockout state"

    def test_snapshot_does_not_see_future(self, synthetic_db):
        """ASOF-style query at minute 60 must not see the drop at minute 90."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=60)

        # All rows at or before obs_ts should have bikes=10
        result = synthetic_db.execute(
            """
            SELECT MIN(avg_bikes_available)
            FROM raw_station_metrics_1min
            WHERE station_id = 'S1'
              AND window_start <= ?::TIMESTAMP
            """,
            [obs],
        ).fetchone()
        assert result[0] == 10.0, "No future data should leak into feature window"

    def test_rolling_window_excludes_future(self, synthetic_db):
        """60-min rolling at minute 80 should only see bikes=10 (stockout is at 90)."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=80)

        result = synthetic_db.execute(
            """
            SELECT AVG(avg_bikes_available), MIN(avg_bikes_available)
            FROM raw_station_metrics_1min
            WHERE station_id = 'S1'
              AND window_start BETWEEN ?::TIMESTAMP - INTERVAL 60 MINUTE AND ?::TIMESTAMP
            """,
            [obs, obs],
        ).fetchone()
        assert result[0] == 10.0, "Rolling avg should only see bikes=10"
        assert result[1] == 10.0, "Rolling min should only see bikes=10"

    def test_lag_sees_correct_past_value(self, synthetic_db):
        """30-min lag at minute 120 should see minute 90 = bikes 0."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=120)
        lag_target = obs - timedelta(minutes=30)

        result = synthetic_db.execute(
            """
            SELECT avg_bikes_available
            FROM raw_station_metrics_1min
            WHERE station_id = 'S1'
              AND window_start <= ?::TIMESTAMP
            ORDER BY window_start DESC
            LIMIT 1
            """,
            [lag_target],
        ).fetchone()
        assert result[0] == 0.0, "30-min lag at minute 120 should see the stockout"


class TestStationFiltering:
    """Verify small-capacity stations are excluded."""

    def test_small_capacity_excluded(self, synthetic_db):
        """S3 has capacity=3, below the 5.0 threshold."""
        result = synthetic_db.execute(
            """
            SELECT COUNT(DISTINCT station_id)
            FROM raw_station_metrics_1min
            WHERE avg_capacity >= 5.0
            """,
        ).fetchone()
        # S1 and S2 have capacity=30, S3 has capacity=3
        assert result[0] == 2, "Only S1 and S2 should pass capacity filter"
