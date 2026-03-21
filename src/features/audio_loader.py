# src/features/audio_loader.py
"""
Load and preprocess audio files using librosa.

Handles: mono conversion, resampling, and duration reporting.
High-frequency cut is applied post-STFT (frequency-bin slicing in spectrogram.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from src.config import AudioConfig

logger = logging.getLogger(__name__)


def load_audio(
    path: str | Path,
    audio_cfg: AudioConfig,
) -> tuple[np.ndarray, float]:
    """Load an audio file, enforce mono, and resample to target SR.

    Args:
        path: Path to the audio file.
        audio_cfg: Audio preprocessing config (sample_rate, mono).

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
    duration_sec = len(y) / sr

    return y, duration_sec
