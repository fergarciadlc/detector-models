# tests/unit/test_metrics.py
"""Tests for evaluation metrics and plot generation."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, generate_plots, threshold_sweep
from src.evaluation.result import EvalResult


def _binary_result(y_true, y_pred_prob, y_pred) -> EvalResult:
    return EvalResult(
        y_true_binary=y_true,
        y_pred_binary_prob=y_pred_prob,
        y_pred_binary=y_pred,
    )


@pytest.fixture
def perfect_predictions():
    """Predictions where the model is 100% correct."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    y_pred_prob = np.array([0.1, 0.05, 0.2, 0.9, 0.95, 0.85])
    y_pred = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    return y_true, y_pred_prob, y_pred


@pytest.fixture
def mixed_predictions():
    """Realistic predictions with some errors."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    y_pred_prob = np.array([0.1, 0.3, 0.6, 0.2, 0.9, 0.4, 0.8, 0.7])
    y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=np.int32)
    return y_true, y_pred_prob, y_pred


def test_perfect_accuracy(perfect_predictions):
    result = _binary_result(*perfect_predictions)
    metrics = compute_metrics(result, metric_names=["accuracy", "f1"])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_perfect_auc(perfect_predictions):
    result = _binary_result(*perfect_predictions)
    metrics = compute_metrics(result, metric_names=["auc_roc"])
    assert metrics["auc_roc"] == 1.0


def test_eer_perfect_predictions(perfect_predictions):
    result = _binary_result(*perfect_predictions)
    metrics = compute_metrics(result, metric_names=["eer"])
    assert metrics["eer"] == pytest.approx(0.0, abs=0.01)


def test_mixed_predictions_accuracy(mixed_predictions):
    result = _binary_result(*mixed_predictions)
    metrics = compute_metrics(result, metric_names=["accuracy", "f1"])
    assert 0.0 < metrics["accuracy"] < 1.0
    assert 0.0 < metrics["f1"] < 1.0


def test_unknown_metric_is_skipped(perfect_predictions):
    result = _binary_result(*perfect_predictions)
    metrics = compute_metrics(result, metric_names=["accuracy", "nonexistent_metric"])
    assert "accuracy" in metrics
    assert "nonexistent_metric" not in metrics


def test_generate_plots_creates_files(mixed_predictions, tmp_path):
    result = _binary_result(*mixed_predictions)
    paths = generate_plots(
        result,
        plot_names=["confusion_matrix", "roc_curve", "det_curve"],
        output_dir=tmp_path,
    )

    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0


def test_generate_unknown_plot_is_skipped(mixed_predictions, tmp_path):
    result = _binary_result(*mixed_predictions)
    paths = generate_plots(
        result,
        plot_names=["unknown_plot"],
        output_dir=tmp_path,
    )
    assert len(paths) == 0


def test_threshold_sweep_returns_expected_keys(mixed_predictions):
    y_true, y_pred_prob, _ = mixed_predictions
    result = threshold_sweep(
        y_true, y_pred_prob,
        t_range=[0.2, 0.8],
        step=0.1,
    )

    assert "thresholds" in result
    assert "accuracy" in result
    assert "f1" in result
    assert "best_threshold" in result
    assert "best_f1" in result
    assert len(result["thresholds"]) == len(result["accuracy"])


def test_threshold_sweep_best_f1_is_max(mixed_predictions):
    y_true, y_pred_prob, _ = mixed_predictions
    result = threshold_sweep(
        y_true, y_pred_prob,
        t_range=[0.1, 0.9],
        step=0.05,
    )

    assert result["best_f1"] == max(result["f1"])
