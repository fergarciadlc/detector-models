# src/evaluation/metrics.py
"""
Metric computation and plot generation for evaluation.

Binary metrics  (require EvalResult.has_binary):
    accuracy, f1, auc_roc, eer

Regression metrics  (require EvalResult.has_regression):
    mae, rmse, r2, pearson, spearman

Plots:
    Binary:     confusion_matrix, roc_curve, det_curve
    Regression: prediction_scatter, error_by_alpha_bucket
"""

from __future__ import annotations

import logging
from math import sqrt
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import pearsonr, spearmanr

from src.evaluation.result import EvalResult

logger = logging.getLogger(__name__)

# ── Metric sets by type ────────────────────────────────────────────────────────

_BINARY_METRICS = {"accuracy", "f1", "auc_roc", "eer"}
_REGRESSION_METRICS = {"mae", "rmse", "r2", "pearson", "spearman"}

_BINARY_PLOTS = {"confusion_matrix", "roc_curve", "det_curve"}
_REGRESSION_PLOTS = {"prediction_scatter", "error_by_alpha_bucket"}


# ── Public API ─────────────────────────────────────────────────────────────────


def compute_metrics(
    result: EvalResult,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute requested metrics from an EvalResult.

    Regression metrics are prefixed ``regression_`` in the returned dict;
    binary metrics are prefixed ``binary_`` when the model also has a
    regression head (to make them distinguishable in MLflow).
    """
    out: dict[str, float] = {}
    multitask = result.has_binary and result.has_regression

    for name in metric_names:
        try:
            if name in _REGRESSION_METRICS:
                if not result.has_regression:
                    logger.warning("Metric '%s' requires regression output, skipping", name)
                    continue
                value = _compute_regression_metric(name, result)
                key = f"regression_{name}" if multitask else name
                out[key] = value

            elif name in _BINARY_METRICS:
                if not result.has_binary:
                    logger.warning("Metric '%s' requires binary output, skipping", name)
                    continue
                value = _compute_binary_metric(name, result)
                key = f"binary_{name}" if multitask else name
                out[key] = value

            else:
                logger.warning("Unknown metric '%s', skipping", name)

        except Exception:
            logger.warning("Failed to compute metric '%s'", name, exc_info=True)
            out[name] = float("nan")

    return out


def generate_plots(
    result: EvalResult,
    plot_names: list[str],
    output_dir: Path,
) -> list[Path]:
    """Generate evaluation plots and save as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for name in plot_names:
        try:
            path = _generate_one_plot(name, result, output_dir)
            if path:
                saved.append(path)
        except Exception:
            logger.warning("Failed to generate plot '%s'", name, exc_info=True)

    return saved


def threshold_sweep(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    t_range: list[float],
    step: float,
) -> dict:
    """Compute accuracy and F1 across a range of binary thresholds."""
    thresholds = list(np.arange(t_range[0], t_range[1] + step / 2, step))
    accuracies = []
    f1_scores_list = []

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(np.int32)
        accuracies.append(float(accuracy_score(y_true, y_pred)))
        f1_scores_list.append(float(f1_score(y_true, y_pred, zero_division=0)))

    best_idx = int(np.argmax(f1_scores_list))
    logger.info(
        "Threshold sweep: best F1=%.4f at threshold=%.2f",
        f1_scores_list[best_idx], thresholds[best_idx],
    )
    return {
        "thresholds": [round(t, 4) for t in thresholds],
        "accuracy": accuracies,
        "f1": f1_scores_list,
        "best_threshold": round(thresholds[best_idx], 4),
        "best_f1": f1_scores_list[best_idx],
    }


# ── Metric implementations ─────────────────────────────────────────────────────


def _compute_binary_metric(name: str, result: EvalResult) -> float:
    y_true = result.y_true_binary
    y_prob = result.y_pred_binary_prob
    y_pred = result.y_pred_binary

    dispatch = {
        "accuracy": lambda: float(accuracy_score(y_true, y_pred)),
        "f1": lambda: float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": lambda: float(roc_auc_score(y_true, y_prob)),
        "eer": lambda: _compute_eer(y_true, y_prob),
    }
    return dispatch[name]()


def _compute_regression_metric(name: str, result: EvalResult) -> float:
    y_true = result.y_true_alpha
    y_pred = result.y_pred_alpha

    dispatch = {
        "mae": lambda: float(mean_absolute_error(y_true, y_pred)),
        "rmse": lambda: float(sqrt(mean_squared_error(y_true, y_pred))),
        "r2": lambda: float(r2_score(y_true, y_pred)),
        "pearson": lambda: float(pearsonr(y_true, y_pred)[0]),
        "spearman": lambda: float(spearmanr(y_true, y_pred)[0]),
    }
    return dispatch[name]()


def _compute_eer(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ── Plot implementations ───────────────────────────────────────────────────────


def _generate_one_plot(
    name: str, result: EvalResult, output_dir: Path
) -> Path | None:
    if name in _BINARY_PLOTS:
        if not result.has_binary:
            logger.warning("Plot '%s' requires binary output, skipping", name)
            return None
        return _plot_binary(name, result, output_dir)

    if name in _REGRESSION_PLOTS:
        if not result.has_regression:
            logger.warning("Plot '%s' requires regression output, skipping", name)
            return None
        return _plot_regression(name, result, output_dir)

    logger.warning("Unknown plot '%s', skipping", name)
    return None


def _plot_binary(name: str, result: EvalResult, output_dir: Path) -> Path | None:
    y_true = result.y_true_binary
    y_prob = result.y_pred_binary_prob
    y_pred = result.y_pred_binary

    fig, ax = plt.subplots(figsize=(7, 6))

    if name == "confusion_matrix":
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["Real", "AI"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Confusion Matrix")

    elif name == "roc_curve":
        RocCurveDisplay.from_predictions(y_true, y_prob, name="Detector", ax=ax)
        ax.set_title("ROC Curve")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    elif name == "det_curve":
        DetCurveDisplay.from_predictions(y_true, y_prob, name="Detector", ax=ax)
        ax.set_title("DET Curve")

    fig.tight_layout()
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path


def _plot_regression(name: str, result: EvalResult, output_dir: Path) -> Path | None:
    y_true = result.y_true_alpha
    y_pred = result.y_pred_alpha

    fig, ax = plt.subplots(figsize=(7, 6))

    if name == "prediction_scatter":
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
        lims = [0.0, 1.0]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
        ax.set_xlabel("True ai_ratio")
        ax.set_ylabel("Predicted ai_ratio")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title("Prediction Scatter (ai_ratio)")
        mae = float(mean_absolute_error(y_true, y_pred))
        ax.text(0.05, 0.92, f"MAE = {mae:.4f}", transform=ax.transAxes, fontsize=10)
        ax.legend()

    elif name == "error_by_alpha_bucket":
        bins = np.arange(0.0, 1.1, 0.1)
        bucket_mae = []
        bucket_labels = []
        bucket_counts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_true >= lo) & (y_true < hi)
            if name == "error_by_alpha_bucket" and hi == 1.1:
                mask = y_true >= lo
            count = int(mask.sum())
            mae_val = float(mean_absolute_error(y_true[mask], y_pred[mask])) if count > 0 else 0.0
            bucket_labels.append(f"{lo:.1f}–{hi:.1f}")
            bucket_mae.append(mae_val)
            bucket_counts.append(count)

        x = np.arange(len(bucket_labels))
        bars = ax.bar(x, bucket_mae, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax.set_xlabel("True ai_ratio bucket")
        ax.set_ylabel("MAE")
        ax.set_title("MAE by ai_ratio Bucket")

        # Annotate bar tops with sample counts
        for bar, count in zip(bars, bucket_counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"n={count}",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.tight_layout()
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path
