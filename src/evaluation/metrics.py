# src/evaluation/metrics.py
"""
Metric computation and plot generation for evaluation.

Supports: accuracy, F1, AUC-ROC, EER, confusion matrix, ROC curve, DET curve.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _compute_eer(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Compute Equal Error Rate from ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute requested evaluation metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_prob: Predicted probabilities (for AUC, EER).
        y_pred: Predicted binary labels (for accuracy, F1).
        metric_names: List of metric names to compute.

    Returns:
        Dict of {metric_name: value}.
    """
    results: dict[str, float] = {}

    dispatch = {
        "accuracy": lambda: float(accuracy_score(y_true, y_pred)),
        "f1": lambda: float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": lambda: float(roc_auc_score(y_true, y_pred_prob)),
        "eer": lambda: _compute_eer(y_true, y_pred_prob),
    }

    for name in metric_names:
        fn = dispatch.get(name)
        if fn is None:
            logger.warning("Unknown metric '%s', skipping", name)
            continue
        try:
            results[name] = fn()
        except Exception:
            logger.warning("Failed to compute metric '%s'", name, exc_info=True)
            results[name] = float("nan")

    return results


def generate_plots(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred: np.ndarray,
    plot_names: list[str],
    output_dir: Path,
) -> list[Path]:
    """Generate evaluation plots and save as PNG.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_prob: Predicted probabilities.
        y_pred: Predicted binary labels.
        plot_names: List of plot names to generate.
        output_dir: Directory to save plot PNGs.

    Returns:
        List of paths to generated plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for name in plot_names:
        try:
            path = _generate_one_plot(
                name, y_true, y_pred_prob, y_pred, output_dir,
            )
            if path:
                saved.append(path)
        except Exception:
            logger.warning("Failed to generate plot '%s'", name, exc_info=True)

    return saved


def _generate_one_plot(
    name: str,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path | None:
    """Generate a single plot."""
    fig, ax = plt.subplots(figsize=(7, 6))

    if name == "confusion_matrix":
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["Real", "Fake"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Confusion Matrix")

    elif name == "roc_curve":
        RocCurveDisplay.from_predictions(
            y_true, y_pred_prob,
            name="Detector",
            ax=ax,
        )
        ax.set_title("ROC Curve")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    elif name == "det_curve":
        DetCurveDisplay.from_predictions(
            y_true, y_pred_prob,
            name="Detector",
            ax=ax,
        )
        ax.set_title("DET Curve")

    else:
        logger.warning("Unknown plot '%s', skipping", name)
        plt.close(fig)
        return None

    fig.tight_layout()
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info("Saved plot: %s", path)
    return path


def threshold_sweep(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    t_range: list[float],
    step: float,
) -> dict[str, list]:
    """Compute metrics across a range of thresholds.

    Args:
        y_true: Ground-truth labels.
        y_pred_prob: Predicted probabilities.
        t_range: [min_threshold, max_threshold].
        step: Step size for threshold values.

    Returns:
        Dict with 'thresholds', 'accuracy', 'f1' lists.
    """
    thresholds = list(np.arange(t_range[0], t_range[1] + step / 2, step))
    accuracies = []
    f1_scores = []

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(np.int32)
        accuracies.append(float(accuracy_score(y_true, y_pred)))
        f1_scores.append(float(f1_score(y_true, y_pred, zero_division=0)))

    best_f1_idx = int(np.argmax(f1_scores))
    logger.info(
        "Threshold sweep: best F1=%.4f at threshold=%.2f",
        f1_scores[best_f1_idx], thresholds[best_f1_idx],
    )

    return {
        "thresholds": [round(t, 4) for t in thresholds],
        "accuracy": accuracies,
        "f1": f1_scores,
        "best_threshold": round(thresholds[best_f1_idx], 4),
        "best_f1": f1_scores[best_f1_idx],
    }
