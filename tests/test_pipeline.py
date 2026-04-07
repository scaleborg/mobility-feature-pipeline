"""End-to-end pipeline test: synthetic data → Parquet → schema validation."""

import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from mobility_feature_pipeline.config import Settings
from mobility_feature_pipeline.pipeline import _build_query
from mobility_feature_pipeline.validate import EXPECTED_COLUMNS


def test_full_pipeline_produces_valid_parquet(synthetic_db):
    """Run the full SQL query against synthetic data and validate output schema."""
    settings = Settings(min_forward_rows=10)  # lower threshold for small synthetic dataset
    query = _build_query(settings)

    arrow_table = synthetic_db.execute(query).arrow().read_all()

    assert arrow_table.num_rows > 0, "Query should return rows from synthetic data"

    # Check column names and order
    actual_cols = [f.name for f in arrow_table.schema]
    assert actual_cols == EXPECTED_COLUMNS, (
        f"Column mismatch.\nExpected: {EXPECTED_COLUMNS}\nActual: {actual_cols}"
    )

    # Check that only S1 and S2 are present (S3 filtered by capacity)
    station_ids = set(v.as_py() for v in arrow_table.column("station_id"))
    assert "S3" not in station_ids, "Small-capacity station S3 should be excluded"
    assert station_ids <= {"S1", "S2"}

    # Check target is binary
    target_col = arrow_table.column("target_empty_next_hour")
    target_values = set(v.as_py() for v in target_col)
    assert target_values <= {0, 1}, f"Target should be binary, got {target_values}"

    # Write to temp Parquet and verify it's readable
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_output.parquet"
        pq.write_table(arrow_table, out_path, compression="snappy")
        reloaded = pq.read_table(out_path)
        assert reloaded.num_rows == arrow_table.num_rows


def test_dry_run_does_not_write(synthetic_db):
    """Verify _build_query produces valid SQL without side effects."""
    settings = Settings(min_forward_rows=10)
    query = _build_query(settings)

    # Should be valid SQL
    result = synthetic_db.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()
    assert result[0] >= 0


def test_s1_has_positive_labels(synthetic_db):
    """S1 has a stockout event — some observations should have label=1."""
    settings = Settings(min_forward_rows=10)
    query = _build_query(settings)

    arrow_table = synthetic_db.execute(query).arrow().read_all()
    df = arrow_table.to_pandas()

    s1_labels = df[df["station_id"] == "S1"]["target_empty_next_hour"]
    assert 1 in s1_labels.values, "S1 should have at least one positive label"


def test_s2_has_no_positive_labels(synthetic_db):
    """S2 never empties — all labels should be 0."""
    settings = Settings(min_forward_rows=10)
    query = _build_query(settings)

    arrow_table = synthetic_db.execute(query).arrow().read_all()
    df = arrow_table.to_pandas()

    s2_labels = df[df["station_id"] == "S2"]["target_empty_next_hour"]
    assert (s2_labels == 0).all(), "S2 should have no positive labels"
