# src/evaluation/evaluators/_base.py
"""Abstract base evaluator — owns the inference loop, delegates output parsing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from src.evaluation.result import EvalResult
from src.features.windowing import strided_windows

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Runs strided-window inference over test entries and builds an EvalResult.

    Subclasses implement four small hooks that handle model-type-specific
    behaviour: how to unpack raw predictions, how to build a per-file score
    dict, how to accumulate arrays, and how to assemble the final EvalResult.
    """

    def run(
        self,
        model,
        entries: list[dict],
        crop_frames: int,
        threshold: float = 0.5,
        hop_frames: int | None = None,
        batch_size: int = 64,
    ) -> EvalResult:
        """Run inference on all entries and return a populated EvalResult.

        Args:
            model: Compiled Keras model.
            entries: Feature manifest entries (test split).
            crop_frames: Window width in time frames.
            threshold: Decision threshold for binary predictions.
            hop_frames: Stride between windows (defaults to crop_frames).
            batch_size: Batch size for model.predict.

        Returns:
            EvalResult with per-file scores and aggregated arrays.
        """
        self._setup(model)

        file_scores: list[dict] = []
        accumulators = self._init_accumulators()
        n_total = len(entries)
        n_errors = 0

        for idx, entry in enumerate(entries):
            try:
                spec = np.load(entry["feature_path"]).astype(np.float32)
            except Exception:
                logger.warning("Failed to load %s, skipping", entry["feature_path"])
                n_errors += 1
                continue

            windows = strided_windows(spec, crop_frames, hop_frames)
            n_windows = windows.shape[0]
            raw_preds = model.predict(windows, batch_size=batch_size, verbose=0)

            parsed = self._parse_predictions(raw_preds)
            file_score = self._make_file_score(entry, parsed, n_windows, threshold)
            file_scores.append(file_score)
            self._accumulate(accumulators, entry, parsed)

            if (idx + 1) % 50 == 0 or idx == n_total - 1:
                logger.info(
                    "  Evaluated %d/%d files (errors: %d)",
                    idx + 1, n_total, n_errors,
                )

        logger.info(
            "Evaluation complete: %d files, %d errors, threshold=%.2f",
            n_total, n_errors, threshold,
        )
        return self._build_result(file_scores, accumulators, threshold)

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _setup(self, model) -> None:
        """Optional hook called once before the inference loop.

        Subclasses can use this to inspect ``model.output_names`` or other
        model properties that affect how predictions are parsed.
        """

    @abstractmethod
    def _init_accumulators(self) -> dict:
        """Return a fresh dict of empty lists used to accumulate per-file values."""

    @abstractmethod
    def _parse_predictions(self, raw_preds) -> dict:
        """Unpack model.predict output into a flat dict of aggregated scalars.

        Args:
            raw_preds: Direct output of model.predict (array or dict of arrays).

        Returns:
            Dict with keys like ``binary_prob``, ``alpha``, ``window_binary_probs``, etc.
        """

    @abstractmethod
    def _make_file_score(
        self, entry: dict, parsed: dict, n_windows: int, threshold: float
    ) -> dict:
        """Build the per-file score dict stored in EvalResult.file_scores."""

    @abstractmethod
    def _accumulate(self, accumulators: dict, entry: dict, parsed: dict) -> None:
        """Append this file's values to the running accumulators."""

    @abstractmethod
    def _build_result(
        self, file_scores: list[dict], accumulators: dict, threshold: float
    ) -> EvalResult:
        """Convert accumulators into a fully populated EvalResult."""
