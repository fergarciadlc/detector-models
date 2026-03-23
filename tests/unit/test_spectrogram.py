# tests/unit/test_spectrogram.py
"""Tests for spectrogram computation."""

import numpy as np
import pytest

from src.config import (
    AudioConfig,
    FeaturesConfig,
    MelConfig,
    NormalizationConfig,
    STFTConfig,
)
from src.features.spectrogram import _hf_cut_bin, compute_spectrogram


@pytest.fixture
def features_cfg():
    """Feature config matching stft_db_44k.yaml."""
    return FeaturesConfig(
        name="test_stft",
        type="stft_magnitude_db",
        audio=AudioConfig(sample_rate=44100, mono=True, hf_cut_hz=16000),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=512),
        normalization=NormalizationConfig(strategy="fixed", mean=-4.0, std=3.0),
        storage="full",
        output_format="npy",
    )


@pytest.fixture
def sine_wave():
    """1-second 440 Hz sine wave at 44100 Hz."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_output_shape(features_cfg, sine_wave):
    S = compute_spectrogram(sine_wave, features_cfg)

    assert S.ndim == 2
    assert S.dtype == np.float32

    expected_freq_bins = _hf_cut_bin(2048, 44100, 16000)
    assert S.shape[1] == expected_freq_bins

    # librosa.stft uses center=True by default, padding n_fft//2 on each side
    expected_frames = 1 + len(sine_wave) // 512
    assert S.shape[0] == expected_frames


def test_no_hf_cut(sine_wave):
    cfg = FeaturesConfig(
        name="test",
        type="stft_magnitude_db",
        audio=AudioConfig(sample_rate=44100, mono=True, hf_cut_hz=None),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=512),
        normalization=NormalizationConfig(strategy="fixed", mean=-4.0, std=3.0),
        storage="full",
        output_format="npy",
    )

    S = compute_spectrogram(sine_wave, cfg)

    full_bins = 2048 // 2 + 1  # n_fft/2 + 1
    assert S.shape[1] == full_bins


def test_hf_cut_bin():
    bin_idx = _hf_cut_bin(n_fft=2048, sample_rate=44100, hf_cut_hz=16000)
    freq_per_bin = 44100 / 2048
    assert bin_idx == int(16000 / freq_per_bin) + 1


def test_fixed_normalization(features_cfg, sine_wave):
    S = compute_spectrogram(sine_wave, features_cfg)

    # With fixed normalization (mean=-4, std=3), values should be roughly centered
    # Not all zeros, not all the same
    assert S.std() > 0


def test_per_sample_normalization(sine_wave):
    cfg = FeaturesConfig(
        name="test",
        type="stft_magnitude_db",
        audio=AudioConfig(sample_rate=44100, mono=True),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=512),
        normalization=NormalizationConfig(strategy="per_sample"),
        storage="full",
        output_format="npy",
    )

    S = compute_spectrogram(sine_wave, cfg)

    assert abs(S.mean()) < 0.1  # approximately zero mean
    assert abs(S.std() - 1.0) < 0.1  # approximately unit variance


def test_unknown_normalization(sine_wave):
    cfg = FeaturesConfig(
        name="test",
        type="stft_magnitude_db",
        audio=AudioConfig(sample_rate=44100, mono=True),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=512),
        normalization=NormalizationConfig(strategy="unknown", mean=0, std=1),
        storage="full",
        output_format="npy",
    )

    with pytest.raises(ValueError, match="Unknown normalization"):
        compute_spectrogram(sine_wave, cfg)


def test_mel_spectrogram_output_shape():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    cfg = FeaturesConfig(
        name="test_mel",
        type="mel_spectrogram_db",
        audio=AudioConfig(sample_rate=sr, mono=True),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=1024),
        mel=MelConfig(n_mels=128, fmin=20.0, fmax=8000.0, power=2.0),
        normalization=NormalizationConfig(strategy="fixed", mean=-4.0, std=3.0),
        storage="full",
        output_format="npy",
    )

    S = compute_spectrogram(y, cfg)
    assert S.ndim == 2
    assert S.dtype == np.float32
    assert S.shape[1] == 128


def test_unknown_feature_type_raises(sine_wave):
    cfg = FeaturesConfig(
        name="test_unknown_feature_type",
        type="unknown_feature",
        audio=AudioConfig(sample_rate=44100, mono=True),
        stft=STFTConfig(n_fft=2048, win_length=2048, hop_length=512),
        normalization=NormalizationConfig(strategy="fixed", mean=-4.0, std=3.0),
        storage="full",
        output_format="npy",
    )

    with pytest.raises(ValueError, match="Unknown feature type"):
        compute_spectrogram(sine_wave, cfg)
