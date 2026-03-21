# src/features/spectrogram.py
"""
Compute spectrograms from audio waveforms.

Supports STFT magnitude in dB with optional high-frequency cut
and fixed/per-sample normalization.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

from src.config import FeaturesConfig

logger = logging.getLogger(__name__)


def _hf_cut_bin(n_fft: int, sample_rate: int, hf_cut_hz: int) -> int:
    """Return the frequency-bin index corresponding to hf_cut_hz (exclusive)."""
    freq_per_bin = sample_rate / n_fft
    return int(hf_cut_hz / freq_per_bin) + 1


def compute_spectrogram(
    y: np.ndarray,
    features_cfg: FeaturesConfig,
) -> np.ndarray:
    """Compute a full-length spectrogram from a waveform.

    Pipeline: STFT -> magnitude -> dB -> hf_cut -> normalize.

    Args:
        y: 1-D float32 waveform at features_cfg.audio.sample_rate.
        features_cfg: Full features config (STFT params, normalization, etc.).

    Returns:
        2-D float32 array of shape (n_freq_bins, n_frames).
        n_freq_bins may be reduced if hf_cut_hz is set.
    """
    stft = features_cfg.stft

    S = librosa.stft(
        y,
        n_fft=stft.n_fft,
        hop_length=stft.hop_length,
        win_length=stft.win_length,
    )

    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    if features_cfg.audio.hf_cut_hz is not None:
        max_bin = _hf_cut_bin(
            stft.n_fft,
            features_cfg.audio.sample_rate,
            features_cfg.audio.hf_cut_hz,
        )
        S_db = S_db[:max_bin, :]

    S_db = _normalize(S_db, features_cfg)

    return S_db.astype(np.float32)


def _normalize(S: np.ndarray, features_cfg: FeaturesConfig) -> np.ndarray:
    """Apply normalization strategy to a spectrogram."""
    norm = features_cfg.normalization

    if norm.strategy == "fixed":
        return (S - norm.mean) / norm.std

    if norm.strategy == "per_sample":
        mean = S.mean()
        std = S.std() + 1e-8
        return (S - mean) / std

    raise ValueError(f"Unknown normalization strategy: {norm.strategy}")
