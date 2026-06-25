# src/evaluation/evaluators/multitask.py
"""Evaluator for multitask models (regression head + binary head)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

from src.evaluation.evaluators._base import BaseEvaluator
from src.evaluation.result import EvalResult


class MultitaskEvaluator(BaseEvaluator):
    """Evaluates models with both a regression output (ai_energy_ratio) and a binary
    output (has_ai).

    model.predict() returns a list of arrays ordered by model.output_names.
    _setup() captures the correct indices so _parse_predictions is order-safe.
    """

    def _setup(self, model) -> None:
        names = list(model.output_names)
        self._regression_idx = names.index("regression_output")
        self._binary_idx = names.index("binary_output")
        logger.debug("Multitask output order: %s", names)

    def _init_accumulators(self) -> dict:
        return {
            "y_true_alpha": [],
            "y_pred_alpha": [],
            "y_true_binary": [],
            "y_pred_binary_prob": [],
        }

    def _parse_predictions(self, raw_preds) -> dict:
        alpha_preds = raw_preds[self._regression_idx].flatten()
        binary_probs = raw_preds[self._binary_idx].flatten()
        return {
            "alpha": float(np.mean(alpha_preds)),
            "binary_prob": float(np.mean(binary_probs)),
            "window_alpha_preds": alpha_preds.tolist(),
            "window_binary_probs": binary_probs.tolist(),
        }

    def _make_file_score(self, entry, parsed, n_windows, threshold) -> dict:
        return {
            "feature_path": entry["feature_path"],
            "ai_energy_ratio": entry.get("ai_energy_ratio"),
            "n_windows": n_windows,
            "predicted_alpha": parsed["alpha"],
            "predicted_binary_prob": parsed["binary_prob"],
            "predicted_has_ai": int(parsed["binary_prob"] >= threshold),
            "window_alpha_preds": parsed["window_alpha_preds"],
            "window_binary_probs": parsed["window_binary_probs"],
        }

    def _accumulate(self, accumulators, entry, parsed) -> None:
        ai_energy_ratio = float(entry.get("ai_energy_ratio", 0.0))
        accumulators["y_true_alpha"].append(ai_energy_ratio)
        accumulators["y_pred_alpha"].append(parsed["alpha"])
        accumulators["y_true_binary"].append(int(ai_energy_ratio > 0))
        accumulators["y_pred_binary_prob"].append(parsed["binary_prob"])

    def _build_result(self, file_scores, accumulators, threshold) -> EvalResult:
        y_true_alpha = np.array(accumulators["y_true_alpha"], dtype=np.float64)
        y_pred_alpha = np.array(accumulators["y_pred_alpha"], dtype=np.float64)
        y_true_binary = np.array(accumulators["y_true_binary"], dtype=np.int32)
        y_pred_binary_prob = np.array(accumulators["y_pred_binary_prob"], dtype=np.float64)
        y_pred_binary = (y_pred_binary_prob >= threshold).astype(np.int32)

        return EvalResult(
            file_scores=file_scores,
            y_true_binary=y_true_binary,
            y_pred_binary_prob=y_pred_binary_prob,
            y_pred_binary=y_pred_binary,
            y_true_alpha=y_true_alpha,
            y_pred_alpha=y_pred_alpha,
        )
