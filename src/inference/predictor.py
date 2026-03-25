# src/inference/predictor.py
"""
Inference engine for unlabeled audio files.

Two input modes:
  1. Raw audio (on-the-fly feature extraction, parallel via ProcessPoolExecutor)
  2. Pre-extracted features (feature manifest .npy spectrograms)

Results are emitted via an optional on_result callback for incremental I/O,
allowing streaming writes to JSONL/CSV without buffering all predictions in memory.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import FeaturesConfig
from src.features.audio_loader import load_audio
from src.features.spectrogram import compute_spectrogram
from src.features.windowing import strided_windows

logger = logging.getLogger(__name__)

OnResultCallback = Callable[[dict], None] | None


# ── Worker (module-level so ProcessPoolExecutor can pickle it) ───────────────


def _extract_one(
    file_path: Path,
    features_cfg: FeaturesConfig,
) -> tuple[str, np.ndarray | None, float | None, str | None]:
    """Load audio and compute spectrogram in a worker process.

    Returns:
        (path_str, spectrogram, duration_sec, error_message)
        On failure: spectrogram and duration_sec are None; error_message is set.
    """
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(k, "1")

    try:
        y, duration_sec = load_audio(file_path, features_cfg.audio)
        spec = compute_spectrogram(y, features_cfg)
        return str(file_path), spec, duration_sec, None
    except Exception as exc:
        return str(file_path), None, None, str(exc)


# ── Inference helpers ────────────────────────────────────────────────────────


def _infer_windows(
    model,
    file_path: str | Path,
    spec: np.ndarray,
    crop_frames: int,
    hop_frames: int | None,
    batch_size: int,
    verbose: bool,
    duration_sec: float | None = None,
) -> dict:
    """Run strided-window inference on a single spectrogram.

    Returns a result dict with summary stats and, when verbose=True,
    the full per-window score list.
    """
    windows = strided_windows(spec, crop_frames, hop_frames)
    preds = model.predict(windows, batch_size=batch_size, verbose=0)
    window_probs = preds.flatten()

    fake_score = float(np.mean(window_probs))

    entry: dict = {
        "file_path": str(file_path),
        "filename": Path(file_path).name,
        "fake_score": round(fake_score, 6),
        "n_windows": int(len(window_probs)),
        "min_score": round(float(np.min(window_probs)), 6),
        "max_score": round(float(np.max(window_probs)), 6),
        "std_score": round(float(np.std(window_probs)), 6),
        "error": None,
    }
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 2)
    if verbose:
        entry["window_scores"] = [round(float(p), 6) for p in window_probs]
    return entry


def _error_entry(file_path: str | Path, error_msg: str) -> dict:
    """Build a result dict for a file that failed extraction or inference."""
    return {
        "file_path": str(file_path),
        "filename": Path(file_path).name,
        "fake_score": None,
        "n_windows": 0,
        "min_score": None,
        "max_score": None,
        "std_score": None,
        "error": error_msg,
    }


# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class PredictResult:
    """Container for a complete prediction run (all files)."""

    predictions: list[dict] = field(default_factory=list)
    n_errors: int = 0

    @property
    def summary(self) -> dict:
        """Aggregate stats over all successfully processed files."""
        valid = [p for p in self.predictions if p.get("error") is None]
        scores = [p["fake_score"] for p in valid]
        out: dict = {
            "total_files": len(self.predictions),
            "n_errors": self.n_errors,
        }
        if scores:
            out["avg_fake_score"] = round(float(np.mean(scores)), 4)
            out["median_fake_score"] = round(float(np.median(scores)), 4)
            out["min_fake_score"] = round(float(np.min(scores)), 4)
            out["max_fake_score"] = round(float(np.max(scores)), 4)
        return out


# ── Public API ───────────────────────────────────────────────────────────────


def predict_from_audio(
    model,
    audio_files: list[Path],
    features_cfg: FeaturesConfig,
    crop_frames: int,
    hop_frames: int | None = None,
    batch_size: int = 64,
    workers: int = 1,
    verbose: bool = False,
    show_progress: bool = True,
    on_result: OnResultCallback = None,
) -> PredictResult:
    """Run inference on raw audio files with on-the-fly feature extraction.

    Feature extraction (CPU-bound) runs in worker processes; model inference
    (GPU-bound when available) runs in the main thread. Results are emitted
    incrementally via on_result so the caller can stream writes to disk.

    Args:
        model: Loaded Keras model.
        audio_files: Audio file paths to process.
        features_cfg: Feature config — must match the config used during training.
        crop_frames: Window width in time frames (experiment config input_shape[0]).
        hop_frames: Stride between windows. Defaults to crop_frames (non-overlapping).
        batch_size: Batch size passed to model.predict.
        workers: Parallel extraction processes (1 = serial, no subprocess overhead).
        verbose: Include per-window scores in result dicts.
        show_progress: Show a tqdm progress bar (default: True).
        on_result: Called with each result dict immediately after processing.

    Returns:
        PredictResult containing all predictions and summary stats.
    """
    result = PredictResult()
    n_total = len(audio_files)

    def _process(path_str: str, spec: np.ndarray | None, duration_sec: float | None, error: str | None) -> dict:
        if error:
            return _error_entry(path_str, error)
        return _infer_windows(model, path_str, spec, crop_frames, hop_frames, batch_size, verbose, duration_sec)

    bar = tqdm(total=n_total, desc="Predicting", unit="file", disable=not show_progress)
    with bar:
        if workers == 1:
            for file_path in audio_files:
                path_str, spec, duration_sec, error = _extract_one(file_path, features_cfg)
                entry = _process(path_str, spec, duration_sec, error)
                if error:
                    result.n_errors += 1
                result.predictions.append(entry)
                if on_result:
                    on_result(entry)
                bar.update(1)
                if result.n_errors:
                    bar.set_postfix(errors=result.n_errors, refresh=False)
        else:
            logger.info("Parallel extraction: workers=%d", workers)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_extract_one, f, features_cfg): f for f in audio_files}
                for fut in as_completed(futures):
                    path_str, spec, duration_sec, error = fut.result()
                    entry = _process(path_str, spec, duration_sec, error)
                    if error:
                        result.n_errors += 1
                    result.predictions.append(entry)
                    if on_result:
                        on_result(entry)
                    bar.update(1)
                    if result.n_errors:
                        bar.set_postfix(errors=result.n_errors, refresh=False)

    logger.info("Prediction complete: %d files, %d errors.", n_total, result.n_errors)
    return result


def predict_from_features(
    model,
    entries: list[dict],
    crop_frames: int,
    hop_frames: int | None = None,
    batch_size: int = 64,
    verbose: bool = False,
    show_progress: bool = True,
    on_result: OnResultCallback = None,
) -> PredictResult:
    """Run inference on pre-extracted feature manifest entries.

    Labels are not required and are ignored if present, making this usable
    for both labelled (cross-dataset evaluation) and unlabelled workflows.

    Args:
        model: Loaded Keras model.
        entries: Feature manifest entry dicts. Each must have 'feature_path'.
        crop_frames: Window width in time frames.
        hop_frames: Stride between windows. Defaults to crop_frames.
        batch_size: Batch size for model.predict.
        verbose: Include per-window scores in result dicts.
        show_progress: Show a tqdm progress bar (default: True).
        on_result: Called with each result dict immediately after processing.

    Returns:
        PredictResult containing all predictions and summary stats.
    """
    result = PredictResult()

    with tqdm(entries, desc="Predicting", unit="file", disable=not show_progress) as bar:
        for entry in bar:
            feature_path = entry["feature_path"]
            duration_sec = entry.get("duration_sec")

            try:
                spec = np.load(feature_path).astype(np.float32)
                pred_entry = _infer_windows(
                    model, feature_path, spec, crop_frames, hop_frames, batch_size, verbose, duration_sec
                )
            except Exception as exc:
                pred_entry = _error_entry(feature_path, str(exc))
                result.n_errors += 1

            result.predictions.append(pred_entry)
            if on_result:
                on_result(pred_entry)
            if result.n_errors:
                bar.set_postfix(errors=result.n_errors, refresh=False)

    logger.info("Prediction complete: %d files, %d errors.", len(entries), result.n_errors)
    return result
