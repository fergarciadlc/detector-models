# src/evaluation/evaluators/regression_windowed.py
"""Evaluator for windowed regression on ARIA-w5s.

Unlike the standard `RegressionEvaluator`, which slices each spectrogram
into non-overlapping strided windows and averages predictions to produce
one (true, pred) per file, this evaluator joins each saved spectrogram
with the canonical 5s windows that fall inside its segment and emits one
(true, pred) per (mix, window) — matching how the model is trained.

Each entry in `EvalResult.file_scores` aggregates back to one row per
mix, listing the per-window pairs.  `y_true_alpha` / `y_pred_alpha` are
flattened across all kept windows of all mixes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.aria_w5s_loader import join_windows_to_features, load_aria_w5s_windows
from src.evaluation.result import EvalResult

logger = logging.getLogger(__name__)


class WindowedRegressionEvaluator:
    """Window-aligned regression evaluator for ARIA-w5s."""

    def __init__(
        self,
        windows_jsonl_path: str | Path,
        dataset_base_path: str | Path,
        sample_rate: int,
        hop_length: int,
        split: str = "test",
    ) -> None:
        self._windows_jsonl_path = Path(windows_jsonl_path)
        self._dataset_base_path = Path(dataset_base_path)
        self._sample_rate = int(sample_rate)
        self._hop_length = int(hop_length)
        self._split = split

    def run(
        self,
        model,
        entries: list[dict],
        crop_frames: int,
        threshold: float = 0.5,  # unused — kept for parity with BaseEvaluator
        hop_frames: int | None = None,  # unused
        batch_size: int = 64,
    ) -> EvalResult:
        windows_by_split = load_aria_w5s_windows(self._windows_jsonl_path)
        windows = windows_by_split.get(self._split, [])
        if not windows:
            raise ValueError(
                f"No '{self._split}' windows found in {self._windows_jsonl_path}"
            )

        pairs = join_windows_to_features(
            windows, entries, self._dataset_base_path
        )
        if not pairs:
            raise ValueError(
                "No (feature, window) pairs survived the join — check that "
                "the feature manifest has been backfilled with start_offset_sec."
            )

        # Group pairs by feature_path so each spectrogram is loaded once.
        by_path: dict[str, list[dict]] = defaultdict(list)
        for p in pairs:
            by_path[p["feature_path"]].append(p)

        y_true_all: list[float] = []
        y_pred_all: list[float] = []
        file_scores: list[dict] = []
        n_errors = 0

        for idx, (feature_path, group) in enumerate(by_path.items()):
            try:
                spec = np.load(feature_path).astype(np.float32)
            except Exception:
                logger.warning("Failed to load %s, skipping", feature_path)
                n_errors += 1
                continue

            n_frames = spec.shape[0]
            crops: list[np.ndarray] = []
            valid: list[dict] = []
            for p in group:
                rel_sec = p["window_start_sec"] - p["start_offset_sec"]
                start_frame = int(round(rel_sec * self._sample_rate / self._hop_length))
                start_frame = max(0, min(start_frame, n_frames - crop_frames))
                crop = spec[start_frame : start_frame + crop_frames, :]
                if crop.shape[0] != crop_frames:
                    # Defensive — should not happen given the strict-containment
                    # filter in `join_windows_to_features`.
                    continue
                crops.append(crop[:, :, np.newaxis])
                valid.append(p)

            if not crops:
                continue

            batch = np.stack(crops, axis=0)
            preds = model.predict(batch, batch_size=batch_size, verbose=0)
            preds = preds.flatten()

            window_records = []
            for p, pred in zip(valid, preds):
                true_alpha = float(p["ai_energy_ratio"])
                pred_alpha = float(pred)
                y_true_all.append(true_alpha)
                y_pred_all.append(pred_alpha)
                window_records.append({
                    "window_index": p["window_index"],
                    "window_start_sec": p["window_start_sec"],
                    "window_end_sec": p["window_end_sec"],
                    "true_alpha": true_alpha,
                    "predicted_alpha": pred_alpha,
                })

            file_scores.append({
                "feature_path": feature_path,
                "mix_path": group[0]["mix_path"],
                "n_windows": len(window_records),
                "windows": window_records,
            })

            if (idx + 1) % 50 == 0 or idx == len(by_path) - 1:
                logger.info(
                    "  Evaluated %d/%d files (%d windows, errors: %d)",
                    idx + 1, len(by_path), len(y_true_all), n_errors,
                )

        logger.info(
            "Windowed evaluation complete: %d files, %d windows, %d errors",
            len(by_path), len(y_true_all), n_errors,
        )

        return EvalResult(
            file_scores=file_scores,
            y_true_alpha=np.array(y_true_all, dtype=np.float64),
            y_pred_alpha=np.array(y_pred_all, dtype=np.float64),
        )
