"""Feature list, split parameters, LightGBM hyperparameters, and serving constants."""

FEATURE_COLUMNS = [
    # Snapshot (3)
    "ft_bikes_available",
    "ft_docks_available",
    "ft_availability_ratio",
    # Lags (4)
    "ft_bikes_available_lag_15m",
    "ft_bikes_available_lag_30m",
    "ft_bikes_available_lag_60m",
    "ft_bikes_available_lag_24h",
    # Rolling 60m (4)
    "ft_avg_bikes_60m",
    "ft_min_bikes_60m",
    "ft_max_bikes_60m",
    "ft_avg_ratio_60m",
    # Rolling 24h (3)
    "ft_avg_bikes_24h",
    "ft_min_bikes_24h",
    "ft_max_bikes_24h",
    # Trailing event frequency (1)
    "ft_low_avail_freq_24h",
    # Temporal (3)
    "ft_hour_of_day",
    "ft_day_of_week",
    "ft_is_weekend",
    # Capacity context (3)
    "ft_capacity",
    "ft_pct_bikes_of_capacity",
    "ft_pct_docks_of_capacity",
    # Derived (1)
    "ft_bikes_delta_60m",
]

TARGET_COLUMN = "target_empty_next_hour"

# Columns excluded from training (kept for artifact output)
METADATA_COLUMNS = ["station_id", "obs_ts", "feature_cutoff_ts", "label_window_end"]

SEED = 42

SPLIT_RATIOS = {"train": 0.70, "val": 0.85}  # cumulative breakpoints

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": SEED,
    "verbose": -1,
}

LGBM_TRAIN_PARAMS = {
    "num_boost_round": 1000,
    "callbacks": None,  # set at runtime (early stopping + logging)
}

LOW_BIKES_THRESHOLDS = [0, 1]
LOW_BIKES_DEFAULT_K = 1

# Serving constants
MIN_CAPACITY = 5.0  # stations below this were excluded from training
MAX_SNAPSHOT_STALENESS_MIN = 15  # reject if snapshot source is older than this
SCORE_THRESHOLD = 0.5
DEFAULT_TOP_N = 10
MAX_TOP_N = 50
