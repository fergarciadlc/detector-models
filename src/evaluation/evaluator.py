# src/evaluation/evaluator.py
"""
Inference engine for evaluation.

Loads test spectrograms, extracts deterministic strided windows,
runs model predictions, and aggregates per-file scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.features.windowing import strided_windows

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""

    file_scores: list[dict] = field(default_factory=list)
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_prob: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def evaluate_test_set(
    model,
    entries: list[dict],
    crop_frames: int,
    threshold: float = 0.5,
    hop_frames: int | None = None,
    batch_size: int = 64,
) -> EvalResult:
    """Run inference on test entries with strided windowing and per-file aggregation.

    Args:
        model: Compiled Keras model.
        entries: Feature manifest entries for the test split.
        crop_frames: Window width in frames.
        threshold: Decision threshold for binary classification.
        hop_frames: Stride between windows (defaults to crop_frames).
        batch_size: Batch size for model.predict.

    Returns:
        EvalResult with per-file scores and aggregated arrays.
    """
    file_scores = []
    all_true = []
    all_prob = []

    n_total = len(entries)
    n_errors = 0

    for idx, entry in enumerate(entries):
        feature_path = entry["feature_path"]
        label = entry["label"]

        try:
            spec = np.load(feature_path).astype(np.float32)
        except Exception:
            logger.warning("Failed to load %s, skipping", feature_path)
            n_errors += 1
            continue

        windows = strided_windows(spec, crop_frames, hop_frames)
        n_windows = windows.shape[0]

        preds = model.predict(windows, batch_size=batch_size, verbose=0)
        window_probs = preds.flatten()

        file_prob = float(np.mean(window_probs))

        file_scores.append({
            "feature_path": feature_path,
            "label": int(label),
            "label_name": entry.get("label_name", "unknown"),
            "n_windows": n_windows,
            "window_probs": window_probs.tolist(),
            "predicted_prob": file_prob,
            "predicted_label": int(file_prob >= threshold),
        })

        all_true.append(label)
        all_prob.append(file_prob)

        if (idx + 1) % 50 == 0 or idx == n_total - 1:
            logger.info(
                "  Evaluated %d/%d files (errors: %d)",
                idx + 1, n_total, n_errors,
            )

    y_true = np.array(all_true, dtype=np.int32)
    y_pred_prob = np.array(all_prob, dtype=np.float64)
    y_pred = (y_pred_prob >= threshold).astype(np.int32)

    logger.info(
        "Evaluation complete: %d files, %d errors, threshold=%.2f",
        n_total, n_errors, threshold,
    )

    return EvalResult(
        file_scores=file_scores,
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        y_pred=y_pred,
    )
