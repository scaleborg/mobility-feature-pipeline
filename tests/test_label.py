"""Test target label correctness on known synthetic sequences."""

from datetime import datetime, timedelta


def _run_label_query(con, obs_ts_value, station_id="S1"):
    """Run the forward_label CTE for a specific observation."""
    result = con.execute(
        """
        SELECT
            CASE WHEN MIN(m.avg_bikes_available) < 1.0 THEN 1 ELSE 0 END AS target,
            COUNT(*) AS fwd_rows
        FROM raw_station_metrics_1min m
        WHERE m.station_id = ?
          AND m.window_start > ?::TIMESTAMP
          AND m.window_start <= ?::TIMESTAMP + INTERVAL 60 MINUTE
        """,
        [station_id, obs_ts_value, obs_ts_value],
    ).fetchone()
    return result


class TestLabelCorrectness:
    """Verify target labels against known synthetic data patterns."""

    def test_label_positive_when_stockout_in_window(self, synthetic_db):
        """S1 drops to 0 at minute 90. Observing at minute 60 → stockout within next hour."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=60)
        target, fwd_rows = _run_label_query(synthetic_db, obs)
        assert target == 1, "Should detect stockout at minute 90 (30 min into forward window)"
        assert fwd_rows == 60

    def test_label_negative_when_stockout_outside_window(self, synthetic_db):
        """Observing at minute 15 → stockout at minute 90 is 75 min away, outside 60-min window."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=15)
        target, fwd_rows = _run_label_query(synthetic_db, obs)
        assert target == 0, "Stockout at minute 90 is outside the 60-min forward window"

    def test_label_positive_at_boundary(self, synthetic_db):
        """Observing at minute 30 → stockout at minute 90 is exactly at 60-min boundary."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=30)
        target, fwd_rows = _run_label_query(synthetic_db, obs)
        # minute 90 window_start is at obs+60, which is <= obs+60min, so included
        assert target == 1, "Stockout at exact boundary should be captured"

    def test_label_negative_for_safe_station(self, synthetic_db):
        """S2 never empties. All labels should be 0."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=30)
        target, fwd_rows = _run_label_query(synthetic_db, obs, station_id="S2")
        assert target == 0
        assert fwd_rows == 60

    def test_label_during_active_stockout(self, synthetic_db):
        """S1 at minute 100 is already at 0. Next hour also 0. Label = 1."""
        base = datetime(2024, 6, 15, 8, 0, 0)
        obs = base + timedelta(minutes=100)
        target, fwd_rows = _run_label_query(synthetic_db, obs)
        assert target == 1
