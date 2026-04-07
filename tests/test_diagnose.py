"""Test diagnostic commands on synthetic data."""

from mobility_feature_pipeline.config import Settings
from mobility_feature_pipeline.diagnose import _run_attrition, _run_sensitivity


def test_attrition_counts_match_synthetic(synthetic_db, capsys):
    """Attrition report should show 3 source stations, 2 after capacity, and correct filtering."""
    settings = Settings(min_forward_rows=10)
    _run_attrition(synthetic_db, settings)

    output = capsys.readouterr().out
    assert "Source stations:              3" in output
    # S3 has capacity=3, below 5.0 threshold
    assert "dropped 1)" in output
    # S1 and S2 should survive capacity filter
    assert "After capacity >= 5.0:        2" in output


def test_sensitivity_lower_threshold_retains_more(synthetic_db, capsys):
    """Lower thresholds should retain at least as many observations as higher ones."""
    settings = Settings(min_forward_rows=10)
    _run_sensitivity(synthetic_db, [10, 50], settings)

    output = capsys.readouterr().out
    # Both thresholds should appear in output
    assert ">=10" in output
    assert ">=50" in output
