"""Training orchestrator for Slice 2: baselines + LightGBM."""

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from mobility_feature_pipeline.baseline import always_negative, low_bikes_rule
from mobility_feature_pipeline.evaluate import (
    compute_metrics,
    print_comparison_table,
    print_feature_importance,
    print_null_rates,
    print_split_info,
)
from mobility_feature_pipeline.model_config import (
    FEATURE_COLUMNS,
    LGBM_PARAMS,
    LOW_BIKES_THRESHOLDS,
    SEED,
    SPLIT_RATIOS,
    TARGET_COLUMN,
)


def compute_split_dates(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute temporal split boundaries from unique timestamps.

    Returns (split_ts_val, split_ts_test) such that:
    - train: obs_ts < split_ts_val
    - val:   split_ts_val <= obs_ts < split_ts_test
    - test:  obs_ts >= split_ts_test
    """
    unique_ts = df["obs_ts"].sort_values().unique()
    ts_min = pd.Timestamp(unique_ts[0])
    ts_max = pd.Timestamp(unique_ts[-1])
    total_duration = ts_max - ts_min

    split_ts_val = ts_min + total_duration * SPLIT_RATIOS["train"]
    split_ts_test = ts_min + total_duration * SPLIT_RATIOS["val"]

    return split_ts_val, split_ts_test


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split DataFrame temporally. All rows for a given obs_ts go to exactly one split."""
    split_ts_val, split_ts_test = compute_split_dates(df)

    train_df = df[df["obs_ts"] < split_ts_val]
    val_df = df[(df["obs_ts"] >= split_ts_val) & (df["obs_ts"] < split_ts_test)]
    test_df = df[df["obs_ts"] >= split_ts_test]

    # Verify no timestamp overlap
    train_ts = set(train_df["obs_ts"].unique())
    val_ts = set(val_df["obs_ts"].unique())
    test_ts = set(test_df["obs_ts"].unique())
    assert not (train_ts & val_ts), "Train/val timestamp overlap"
    assert not (val_ts & test_ts), "Val/test timestamp overlap"
    assert not (train_ts & test_ts), "Train/test timestamp overlap"

    split_info = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        y = split_df[TARGET_COLUMN]
        split_info[name] = {
            "start": str(split_df["obs_ts"].min()),
            "end": str(split_df["obs_ts"].max()),
            "rows": len(split_df),
            "positive_rate": float(y.mean()) if len(y) > 0 else 0.0,
        }

    return train_df, val_df, test_df, split_info


def compute_null_rates(df: pd.DataFrame) -> dict[str, float]:
    """Compute null rates for model features on a DataFrame."""
    return {col: float(df[col].isna().mean()) for col in FEATURE_COLUMNS}


def train_pipeline(parquet_path: Path, output_dir: Path):
    """Full training pipeline: load, split, baselines, LightGBM, evaluate, save."""
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print(f"Loading dataset: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Total rows: {len(df):,}  |  Positive rate: {df[TARGET_COLUMN].mean():.4f}")

    # --- Split ---
    train_df, val_df, test_df, split_info = temporal_split(df)
    print_split_info(split_info)

    # --- Null rates on training set ---
    null_rates = compute_null_rates(train_df)
    print_null_rates(null_rates)

    # --- Prepare arrays ---
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # --- Baselines ---
    baseline_results = []

    # Always-negative
    neg = always_negative(y_test)
    neg_metrics = compute_metrics(y_test, neg["y_pred"], neg["y_score"])
    baseline_results.append({"name": neg["name"], "metrics": neg_metrics})

    # Low-bikes rule at each threshold
    for k in LOW_BIKES_THRESHOLDS:
        rule = low_bikes_rule(X_test, k=k)
        rule_metrics = compute_metrics(y_test, rule["y_pred"], rule["y_score"])
        baseline_results.append({"name": rule["name"], "metrics": rule_metrics})

    # --- LightGBM ---
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    print(f"\nTraining LightGBM (scale_pos_weight={scale_pos_weight:.2f}) ...")

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=25),
        ],
    )

    best_iteration = booster.best_iteration
    print(f"  Best iteration: {best_iteration}")

    # --- Evaluate LightGBM on test ---
    y_score = booster.predict(X_test, num_iteration=best_iteration)
    y_pred = (y_score >= 0.5).astype(int)
    lgbm_metrics = compute_metrics(y_test, y_pred, y_score)

    # Feature importance
    importance_gain = booster.feature_importance(importance_type="gain")
    feature_names = booster.feature_name()
    importance = sorted(
        [{"feature": f, "gain": float(g)} for f, g in zip(feature_names, importance_gain)],
        key=lambda x: -x["gain"],
    )

    lgbm_result = {"name": "lightgbm", "metrics": lgbm_metrics}

    # --- Print comparison ---
    print_comparison_table(baseline_results + [lgbm_result])
    print_feature_importance(importance)

    # --- Save artifacts ---

    # 1. Model
    model_path = output_dir / f"model_{run_ts}.lgbm"
    booster.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")

    # 2. Test predictions
    preds_df = pd.DataFrame(
        {
            "station_id": test_df["station_id"].values,
            "obs_ts": test_df["obs_ts"].values,
            "y_true": y_test,
            "y_score": y_score,
            "y_pred": y_pred,
        }
    )
    preds_path = output_dir / f"test_predictions_{run_ts}.parquet"
    preds_df.to_parquet(preds_path, compression="snappy")
    print(f"Test predictions saved: {preds_path}")

    # 3. Metrics JSON
    metrics_dict = {
        "run_timestamp": run_ts,
        "seed": SEED,
        "input_parquet": str(parquet_path),
        "data_note": "24h lag/rolling features may have high null rates due to limited history depth",
        "split": split_info,
        "feature_null_rates": null_rates,
        "baselines": {r["name"]: r["metrics"] for r in baseline_results},
        "lightgbm": {
            "hyperparams": params,
            "scale_pos_weight": scale_pos_weight,
            "metrics": lgbm_metrics,
            "feature_importance_gain": importance,
            "best_iteration": best_iteration,
            "early_stopped": best_iteration < 1000,
        },
    }
    metrics_path = output_dir / f"metrics_{run_ts}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"Metrics saved: {metrics_path}")

    return metrics_dict
