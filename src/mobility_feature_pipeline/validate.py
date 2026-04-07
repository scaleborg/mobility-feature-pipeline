"""Runtime validation and reporting on a built training dataset."""

from pathlib import Path

import pyarrow.parquet as pq

EXPECTED_COLUMNS = [
    "station_id",
    "obs_ts",
    "feature_cutoff_ts",
    "label_window_end",
    "target_empty_next_hour",
    "ft_bikes_available",
    "ft_docks_available",
    "ft_availability_ratio",
    "ft_bikes_available_lag_15m",
    "ft_bikes_available_lag_30m",
    "ft_bikes_available_lag_60m",
    "ft_bikes_available_lag_24h",
    "ft_avg_bikes_60m",
    "ft_min_bikes_60m",
    "ft_max_bikes_60m",
    "ft_avg_bikes_24h",
    "ft_min_bikes_24h",
    "ft_max_bikes_24h",
    "ft_avg_ratio_60m",
    "ft_low_avail_freq_24h",
    "ft_hour_of_day",
    "ft_day_of_week",
    "ft_is_weekend",
    "ft_capacity",
    "ft_pct_bikes_of_capacity",
    "ft_pct_docks_of_capacity",
    "ft_bikes_delta_60m",
]

# Columns that may be null due to sparse history or upstream data quality — warn but do not fail
NULLABLE_COLUMNS = {
    "ft_availability_ratio",
    "ft_bikes_available_lag_15m",
    "ft_bikes_available_lag_30m",
    "ft_bikes_available_lag_60m",
    "ft_bikes_available_lag_24h",
    "ft_avg_bikes_60m",
    "ft_min_bikes_60m",
    "ft_max_bikes_60m",
    "ft_avg_bikes_24h",
    "ft_min_bikes_24h",
    "ft_max_bikes_24h",
    "ft_avg_ratio_60m",
    "ft_low_avail_freq_24h",
    "ft_bikes_delta_60m",
    "ft_pct_bikes_of_capacity",
    "ft_pct_docks_of_capacity",
}


def validate_dataset(parquet_path: Path) -> bool:
    """Validate a built dataset and print a structured report.

    Returns True if all hard checks pass, False otherwise.
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    passed = True
    warnings = []

    print("── Dataset Validation Report ──")
    print(f"File:          {parquet_path}")
    print(f"Row count:     {len(df):,}")

    # Schema check
    actual_cols = list(df.columns)
    if actual_cols != EXPECTED_COLUMNS:
        print("FAIL: Column mismatch")
        print(f"  Expected: {EXPECTED_COLUMNS}")
        print(f"  Actual:   {actual_cols}")
        passed = False
    else:
        print(f"Schema:        ✓ {len(EXPECTED_COLUMNS)} columns in expected order")

    # Station count
    station_count = df["station_id"].nunique()
    print(f"Station count: {station_count:,}")

    # Time range
    obs_min = df["obs_ts"].min()
    obs_max = df["obs_ts"].max()
    print(f"Time range:    {obs_min} → {obs_max}")

    # Positive rate
    target = df["target_empty_next_hour"]
    value_counts = target.value_counts().to_dict()
    pos_count = value_counts.get(1, 0)
    pos_rate = pos_count / len(df) if len(df) > 0 else 0
    print(f"Positive rate: {pos_rate:.1%} ({pos_count:,} / {len(df):,})")
    if pos_rate < 0.01:
        warnings.append(f"Positive rate very low ({pos_rate:.2%}) — check target definition")
    if pos_rate > 0.50:
        warnings.append(f"Positive rate very high ({pos_rate:.2%}) — check target definition")

    # Binary target check
    unique_targets = set(target.dropna().unique())
    if unique_targets <= {0, 1}:
        print("Target:        ✓ binary {0, 1}")
    else:
        print(f"FAIL: Target values not binary: {unique_targets}")
        passed = False

    # Duplicate key check
    dup_count = df.duplicated(subset=["station_id", "obs_ts"]).sum()
    if dup_count == 0:
        print("Duplicates:    ✓ none (station_id, obs_ts)")
    else:
        print(f"FAIL: {dup_count:,} duplicate (station_id, obs_ts) keys")
        passed = False

    # Metadata column consistency
    cutoff_match = (df["feature_cutoff_ts"] == df["obs_ts"]).all()
    if cutoff_match:
        print("Cutoff check:  ✓ feature_cutoff_ts == obs_ts")
    else:
        print("FAIL: feature_cutoff_ts != obs_ts for some rows")
        passed = False

    # Null checks
    print("\n── Null Percentages ──")
    for col in EXPECTED_COLUMNS:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 0:
            marker = "WARN" if col in NULLABLE_COLUMNS else "FAIL"
            print(f"  {col}: {null_pct:.1f}% null [{marker}]")
            if col not in NULLABLE_COLUMNS:
                passed = False
        else:
            print(f"  {col}: 0.0%")

    # Sample rows
    print("\n── Sample Rows (5 random) ──")
    sample = df.sample(n=min(5, len(df)), random_state=42)
    print(sample.to_string(index=False))

    # Warnings
    if warnings:
        print("\n── Warnings ──")
        for w in warnings:
            print(f"  ⚠ {w}")

    print(f"\n── Result: {'PASS' if passed else 'FAIL'} ──")
    return passed
