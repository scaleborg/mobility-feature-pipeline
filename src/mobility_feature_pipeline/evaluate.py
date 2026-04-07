"""Metric computation and report formatting for Slice 2."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute all classification metrics at threshold=0.5."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Handle edge cases where only one class is present in predictions
    has_both_classes = len(np.unique(y_true)) > 1

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

    if has_both_classes and y_score.sum() > 0:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = float(auc(rec_curve, prec_curve))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    # Log loss needs valid probabilities
    if has_both_classes:
        y_score_clipped = np.clip(y_score, 1e-15, 1 - 1e-15)
        metrics["log_loss"] = float(log_loss(y_true, y_score_clipped))
    else:
        metrics["log_loss"] = None

    return metrics


def print_comparison_table(results: list[dict]):
    """Print a side-by-side comparison table of model metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"]
    labels = {
        "accuracy": "Accuracy (secondary)",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "log_loss": "Log Loss",
    }

    # Header
    name_width = max(len(labels[k]) for k in metric_keys)
    col_width = max(max(len(r["name"]) for r in results), 12)
    header = f"{'Metric':<{name_width}}"
    for r in results:
        header += f"  {r['name']:>{col_width}}"
    print(f"\n{'=' * len(header)}")
    print("TEST SET METRICS")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        row = f"{labels[key]:<{name_width}}"
        for r in results:
            val = r["metrics"].get(key)
            if val is None:
                row += f"  {'N/A':>{col_width}}"
            else:
                row += f"  {val:>{col_width}.4f}"
        print(row)

    # Confusion matrices
    print(f"\n{'CONFUSION MATRICES':}")
    for r in results:
        cm = r["metrics"]["confusion_matrix"]
        print(f"\n  {r['name']}:")
        print(f"    TP={cm['tp']:>6}  FP={cm['fp']:>6}")
        print(f"    FN={cm['fn']:>6}  TN={cm['tn']:>6}")


def print_feature_importance(importance: list[dict], top_n: int = 10):
    """Print top features by gain with caveat about shallow history."""
    print(f"\n{'=' * 50}")
    print(f"FEATURE IMPORTANCE (top {top_n} by gain)")
    print("NOTE: Importance may shift as more data accumulates.")
    print("      24h features have high null rates due to limited history.")
    print(f"{'=' * 50}")
    for i, entry in enumerate(importance[:top_n], 1):
        print(f"  {i:>2}. {entry['feature']:<35} {entry['gain']:>10.1f}")


def print_split_info(split_info: dict):
    """Print temporal split summary."""
    print(f"\n{'=' * 60}")
    print("TEMPORAL SPLIT")
    print(f"{'=' * 60}")
    for name in ["train", "val", "test"]:
        s = split_info[name]
        print(
            f"  {name:<6} {s['start']}  →  {s['end']}  "
            f"rows={s['rows']:>7,}  positive_rate={s['positive_rate']:.4f}"
        )


def print_null_rates(null_rates: dict[str, float]):
    """Print feature null rates, highlighting high-null features."""
    high_null = {k: v for k, v in null_rates.items() if v > 0.5}
    if high_null:
        print(f"\n{'=' * 50}")
        print("HIGH NULL-RATE FEATURES (>50% null on training set)")
        print(f"{'=' * 50}")
        for feat, rate in sorted(high_null.items(), key=lambda x: -x[1]):
            print(f"  {feat:<40} {rate:>6.1%}")


def load_and_evaluate(parquet_path: Path, model_path: Path):
    """Re-load a saved model and reproduce test metrics."""
    import lightgbm as lgb

    from mobility_feature_pipeline.model_config import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
    from mobility_feature_pipeline.train import compute_split_dates

    df = pd.read_parquet(parquet_path)
    split_ts_val, split_ts_test = compute_split_dates(df)

    test_df = df[df["obs_ts"] >= split_ts_test]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    booster = lgb.Booster(model_file=str(model_path))
    y_score = booster.predict(X_test)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_score)

    print(f"\nModel: {model_path.name}")
    print(f"Test rows: {len(y_test):,}  |  Positive rate: {y_test.mean():.4f}")

    results = [{"name": "lightgbm", "metrics": metrics}]
    print_comparison_table(results)

    # Load and print saved metrics for comparison
    metrics_files = sorted(model_path.parent.glob("metrics_*.json"), reverse=True)
    if metrics_files:
        with open(metrics_files[0]) as f:
            saved = json.load(f)
        saved_m = saved.get("lightgbm", {}).get("metrics", {})
        if saved_m:
            print("\nSaved metrics (from training run):")
            for k in ["roc_auc", "pr_auc", "f1"]:
                sv = saved_m.get(k)
                cv = metrics.get(k)
                match_str = (
                    "MATCH" if sv is not None and cv is not None and abs(sv - cv) < 1e-6 else "DIFF"
                )
                print(f"  {k}: saved={sv}  recomputed={cv}  [{match_str}]")

    return metrics
