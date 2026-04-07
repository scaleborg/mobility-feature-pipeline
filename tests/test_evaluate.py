"""Tests for metric computation."""

import numpy as np

from mobility_feature_pipeline.evaluate import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_pred, y_score)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["confusion_matrix"]["tp"] == 2
        assert m["confusion_matrix"]["tn"] == 2
        assert m["confusion_matrix"]["fp"] == 0
        assert m["confusion_matrix"]["fn"] == 0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_metrics(y_true, y_pred, y_score)
        assert m["accuracy"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["confusion_matrix"]["fp"] == 2
        assert m["confusion_matrix"]["fn"] == 2

    def test_all_negative_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.zeros(4, dtype=int)
        y_score = np.zeros(4)
        m = compute_metrics(y_true, y_pred, y_score)
        assert m["accuracy"] == 0.5
        assert m["recall"] == 0.0
        assert m["roc_auc"] is None  # constant score

    def test_confusion_matrix_values(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        y_score = np.array([0.9, 0.3, 0.1, 0.7, 0.8, 0.2])
        m = compute_metrics(y_true, y_pred, y_score)
        cm = m["confusion_matrix"]
        assert cm["tp"] == 2
        assert cm["fn"] == 1
        assert cm["fp"] == 1
        assert cm["tn"] == 2

    def test_pr_auc_exists(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
        m = compute_metrics(y_true, y_pred, y_score)
        assert m["pr_auc"] is not None
        assert 0.0 <= m["pr_auc"] <= 1.0
