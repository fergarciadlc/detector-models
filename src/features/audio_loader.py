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


def load_audio(
    path: str | Path,
    audio_cfg: AudioConfig,
    max_duration_sec: float | None = None,
) -> tuple[np.ndarray, float]:
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
        Tuple of (waveform as float32 1-D array, duration in seconds).

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

    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * sr)
        if len(y) > max_samples:
            # Seed from the full path so the offset is reproducible across runs
            seed = int(hashlib.md5(str(path).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, len(y) - max_samples + 1))
            y = y[start : start + max_samples]
            logger.debug(
                "Windowed %s: [%.1fs, %.1fs] of %.1fs total",
                path.name,
                start / sr,
                (start + max_samples) / sr,
                (start + max_samples + (len(y) - max_samples)) / sr,
            )

    duration_sec = len(y) / sr
    return y, duration_sec
