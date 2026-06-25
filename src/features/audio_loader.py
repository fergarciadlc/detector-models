# src/features/audio_loader.py
"""
Load and preprocess audio files using librosa.

Handles: mono conversion, resampling, and duration reporting.
High-frequency cut is applied post-STFT (frequency-bin slicing in spectrogram.py).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np

from src.config import AudioConfig

logger = logging.getLogger(__name__)


def deterministic_window_offset_samples(
    path: str | Path,
    n_samples: int,
    max_samples: int,
) -> int:
    """Compute the deterministic random start offset (in samples) for windowing.

    Mirrors the seeding used inside `load_audio` so external tools (e.g. a
    backfill script for already-extracted features) can reproduce the offset
    without re-decoding the audio if they already know its sample count.
    """
    if n_samples <= max_samples:
        return 0
    seed = int(hashlib.md5(str(path).encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    return int(rng.integers(0, n_samples - max_samples + 1))


def load_audio(
    path: str | Path,
    audio_cfg: AudioConfig,
    max_duration_sec: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Load an audio file, enforce mono, and resample to target SR.

    When `max_duration_sec` is set, a segment of that length is extracted
    instead of the full file.  The start offset is seeded deterministically
    from the file path so the same segment is always selected for a given
    file, making re-extraction reproducible.

    Args:
        path: Path to the audio file.
        audio_cfg: Audio preprocessing config (sample_rate, mono).
        max_duration_sec: If set, cap the returned waveform to this many
            seconds by extracting a deterministic random segment.
            None (default) returns the full file.

    Returns:
        Tuple of (waveform as float32 1-D array, duration in seconds,
        start_offset_sec — the start of the returned segment relative to
        the original file's timeline; 0.0 when no windowing was applied).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If librosa fails to decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, sr = librosa.load(
            path,
            sr=audio_cfg.sample_rate,
            mono=audio_cfg.mono,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio {path}: {exc}") from exc

    y = y.astype(np.float32)
    start_offset_sec = 0.0

    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * sr)
        if len(y) > max_samples:
            start = deterministic_window_offset_samples(path, len(y), max_samples)
            start_offset_sec = start / sr
            y = y[start : start + max_samples]
            logger.debug(
                "Windowed %s: [%.1fs, %.1fs]",
                path.name,
                start_offset_sec,
                start_offset_sec + max_duration_sec,
            )

    duration_sec = len(y) / sr
    return y, duration_sec, start_offset_sec
