# src/features/spectrogram.py
"""
Compute feature maps from audio waveforms.

Supports:
- STFT magnitude in dB with optional high-frequency cut
- Mel spectrogram in dB
with fixed/per-sample normalization.
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
    """Compute a full-length time-major feature map from a waveform.

    Args:
        y: 1-D float32 waveform at features_cfg.audio.sample_rate.
        features_cfg: Full features config (STFT params, normalization, etc.).

    Returns:
        2-D float32 array of shape (n_frames, n_freq_bins), time-major
        (matches tf.signal.stft frame x bin order for STFT features).
    """
    extractors = {
        "stft_magnitude_db": _compute_stft_magnitude_db,
        "mel_spectrogram_db": _compute_mel_spectrogram_db,
    }
    try:
        extractor = extractors[features_cfg.type]
    except KeyError as exc:
        raise ValueError(f"Unknown feature type: {features_cfg.type}") from exc

    S_db = extractor(y, features_cfg)
    S_db = _normalize(S_db, features_cfg)
    return np.ascontiguousarray(S_db.T.astype(np.float32))


def _compute_stft_magnitude_db(y: np.ndarray, features_cfg: FeaturesConfig) -> np.ndarray:
    """STFT magnitude -> dB, optionally cropped in frequency."""
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
    return S_db


def _compute_mel_spectrogram_db(y: np.ndarray, features_cfg: FeaturesConfig) -> np.ndarray:
    """Power mel spectrogram -> dB."""
    if features_cfg.mel is None:
        raise ValueError("Feature type 'mel_spectrogram_db' requires 'mel' config.")

    stft = features_cfg.stft
    mel = features_cfg.mel
    S_mel = librosa.feature.melspectrogram(
        y=y,
        sr=features_cfg.audio.sample_rate,
        n_fft=stft.n_fft,
        hop_length=stft.hop_length,
        win_length=stft.win_length,
        n_mels=mel.n_mels,
        fmin=mel.fmin,
        fmax=mel.fmax,
        power=mel.power,
    )
    return librosa.power_to_db(S_mel, ref=np.max)


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
