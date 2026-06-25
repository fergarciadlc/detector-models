# src/evaluation/result.py
"""Unified evaluation result container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalResult:
    """Holds per-file scores and aggregated arrays for all model types.

    Binary models populate only the ``y_*_binary`` fields.
    Multitask models populate both the binary and alpha (regression) fields.
    Callers should check ``has_regression`` / ``has_binary`` before accessing
    type-specific fields.
    """

    file_scores: list[dict] = field(default_factory=list)

    # Binary classification arrays (binary and multitask models)
    y_true_binary: np.ndarray | None = None
    y_pred_binary_prob: np.ndarray | None = None
    y_pred_binary: np.ndarray | None = None

    # Regression arrays (multitask models)
    y_true_alpha: np.ndarray | None = None
    y_pred_alpha: np.ndarray | None = None

    @property
    def has_binary(self) -> bool:
        return self.y_true_binary is not None

    @property
    def has_regression(self) -> bool:
        return self.y_true_alpha is not None
