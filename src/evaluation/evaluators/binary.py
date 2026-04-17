# src/evaluation/evaluators/binary.py
"""Evaluator for binary classification models."""

from __future__ import annotations

import numpy as np

from src.evaluation.evaluators._base import BaseEvaluator
from src.evaluation.result import EvalResult


class BinaryEvaluator(BaseEvaluator):
    """Evaluates models with a single sigmoid output (real vs AI-generated)."""

    def _init_accumulators(self) -> dict:
        return {"y_true": [], "y_prob": []}

    def _parse_predictions(self, raw_preds) -> dict:
        probs = raw_preds.flatten()
        return {
            "binary_prob": float(np.mean(probs)),
            "window_probs": probs.tolist(),
        }

    def _make_file_score(self, entry, parsed, n_windows, threshold) -> dict:
        prob = parsed["binary_prob"]
        return {
            "feature_path": entry["feature_path"],
            "label": int(entry["label"]),
            "label_name": entry.get("label_name", "unknown"),
            "n_windows": n_windows,
            "window_probs": parsed["window_probs"],
            "predicted_prob": prob,
            "predicted_label": int(prob >= threshold),
        }

    def _accumulate(self, accumulators, entry, parsed) -> None:
        accumulators["y_true"].append(int(entry["label"]))
        accumulators["y_prob"].append(parsed["binary_prob"])

    def _build_result(self, file_scores, accumulators, threshold) -> EvalResult:
        y_true = np.array(accumulators["y_true"], dtype=np.int32)
        y_prob = np.array(accumulators["y_prob"], dtype=np.float64)
        y_pred = (y_prob >= threshold).astype(np.int32)
        return EvalResult(
            file_scores=file_scores,
            y_true_binary=y_true,
            y_pred_binary_prob=y_prob,
            y_pred_binary=y_pred,
        )
