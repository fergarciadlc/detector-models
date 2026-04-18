# src/evaluation/evaluators/regression.py
"""Evaluator for single-output regression models (ai_energy_ratio)."""

from __future__ import annotations

import numpy as np

from src.evaluation.evaluators._base import BaseEvaluator
from src.evaluation.result import EvalResult


class RegressionEvaluator(BaseEvaluator):
    """Evaluates models with a single regression output predicting ai_energy_ratio."""

    def _init_accumulators(self) -> dict:
        return {"y_true_alpha": [], "y_pred_alpha": []}

    def _parse_predictions(self, raw_preds) -> dict:
        alpha_preds = raw_preds.flatten()
        return {
            "alpha": float(np.mean(alpha_preds)),
            "window_alpha_preds": alpha_preds.tolist(),
        }

    def _make_file_score(self, entry, parsed, n_windows, threshold) -> dict:
        return {
            "feature_path": entry["feature_path"],
            "ai_energy_ratio": entry.get("ai_energy_ratio"),
            "n_windows": n_windows,
            "predicted_alpha": parsed["alpha"],
            "window_alpha_preds": parsed["window_alpha_preds"],
        }

    def _accumulate(self, accumulators, entry, parsed) -> None:
        ai_energy_ratio = float(entry.get("ai_energy_ratio", 0.0))
        accumulators["y_true_alpha"].append(ai_energy_ratio)
        accumulators["y_pred_alpha"].append(parsed["alpha"])

    def _build_result(self, file_scores, accumulators, threshold) -> EvalResult:
        y_true_alpha = np.array(accumulators["y_true_alpha"], dtype=np.float64)
        y_pred_alpha = np.array(accumulators["y_pred_alpha"], dtype=np.float64)
        return EvalResult(
            file_scores=file_scores,
            y_true_alpha=y_true_alpha,
            y_pred_alpha=y_pred_alpha,
        )
