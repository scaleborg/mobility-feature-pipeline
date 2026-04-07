"""Baseline models for Slice 2: always-negative and low-bikes rule."""

import numpy as np
import pandas as pd


def always_negative(y_true: np.ndarray) -> dict:
    """Predict 0 for every row. Returns y_pred and y_score arrays."""
    n = len(y_true)
    return {
        "name": "always_negative",
        "y_pred": np.zeros(n, dtype=int),
        "y_score": np.zeros(n, dtype=float),
    }


def low_bikes_rule(X: pd.DataFrame, k: int) -> dict:
    """Predict 1 if ft_bikes_available <= k. Operationally interpretable."""
    bikes = X["ft_bikes_available"].values
    y_pred = (bikes <= k).astype(int)
    # Score: inverse of bikes available, clipped to [0, 1] range
    # Stations with fewer bikes get higher risk scores
    max_bikes = max(bikes.max(), 1.0)
    y_score = np.clip(1.0 - bikes / max_bikes, 0.0, 1.0)
    return {
        "name": f"low_bikes_rule_k{k}",
        "y_pred": y_pred,
        "y_score": y_score,
    }
