# tests/unit/test_audio_loader.py
"""Tests for audio loader."""

import numpy as np
import pytest
import soundfile as sf

from src.config import AudioConfig
from src.features.audio_loader import load_audio


@pytest.fixture
def mono_wav(tmp_path):
    """Create a short mono WAV file (1 second at 44100 Hz)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "test_mono.wav"
    sf.write(path, y, sr)
    return path, sr, duration


@pytest.fixture
def stereo_wav(tmp_path):
    """Create a short stereo WAV file (1 second at 44100 Hz)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    stereo = np.stack([left, right], axis=-1)
    path = tmp_path / "test_stereo.wav"
    sf.write(path, stereo, sr)
    return path, sr, duration


def test_load_mono_audio(mono_wav):
    path, sr, expected_dur = mono_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, duration = load_audio(path, cfg)

    assert y.ndim == 1
    assert y.dtype == np.float32
    assert abs(duration - expected_dur) < 0.05
    assert len(y) == sr  # 1 second


def test_load_stereo_as_mono(stereo_wav):
    path, sr, _ = stereo_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, _ = load_audio(path, cfg)

    assert y.ndim == 1


def test_resample(mono_wav):
    path, original_sr, _ = mono_wav
    target_sr = 22050
    cfg = AudioConfig(sample_rate=target_sr, mono=True)

    y, duration = load_audio(path, cfg)

    assert len(y) == target_sr  # 1 second at new SR
    assert abs(duration - 1.0) < 0.05


def test_file_not_found():
    cfg = AudioConfig(sample_rate=44100, mono=True)
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/audio.wav", cfg)


# ── max_duration_sec windowing ───────────────────────────────────────────────


@pytest.fixture
def long_wav(tmp_path):
    """Create a 10-second mono WAV file."""
    sr = 44100
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "long_audio.wav"
    sf.write(path, y, sr)
    return path, sr, duration


def test_max_duration_truncates(long_wav):
    path, sr, _ = long_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, duration = load_audio(path, cfg, max_duration_sec=3.0)

    assert abs(duration - 3.0) < 0.01
    assert len(y) == int(3.0 * sr)


def test_max_duration_no_truncation_when_file_shorter(long_wav):
    """If max_duration_sec >= file length the full file is returned."""
    path, sr, full_duration = long_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, duration = load_audio(path, cfg, max_duration_sec=60.0)

    assert abs(duration - full_duration) < 0.05


def test_max_duration_deterministic(long_wav):
    """Same file always yields the same segment."""
    path, sr, _ = long_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y1, _ = load_audio(path, cfg, max_duration_sec=3.0)
    y2, _ = load_audio(path, cfg, max_duration_sec=3.0)

    np.testing.assert_array_equal(y1, y2)


def test_max_duration_none_returns_full(long_wav):
    path, sr, full_duration = long_wav
    cfg = AudioConfig(sample_rate=sr, mono=True)

    y, duration = load_audio(path, cfg, max_duration_sec=None)

    assert abs(duration - full_duration) < 0.05
