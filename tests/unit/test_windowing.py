# tests/unit/test_windowing.py
"""Tests for deterministic strided window extraction."""

import numpy as np
import pytest

from src.features.windowing import strided_windows


@pytest.fixture
def spectrogram():
    """A synthetic spectrogram with known dimensions."""
    n_freq, n_frames = 128, 500
    return np.random.randn(n_freq, n_frames).astype(np.float32)


def test_non_overlapping_window_count(spectrogram):
    crop = 100
    windows = strided_windows(spectrogram, crop_frames=crop)

    expected_count = (500 - crop) // crop + 1  # 5
    assert windows.shape[0] == expected_count
    assert windows.shape == (expected_count, 128, crop, 1)


def test_overlapping_window_count(spectrogram):
    crop = 100
    hop = 50
    windows = strided_windows(spectrogram, crop_frames=crop, hop_frames=hop)

    expected_count = len(range(0, 500 - crop + 1, hop))  # 9
    assert windows.shape[0] == expected_count


def test_window_values_are_correct():
    spec = np.arange(20, dtype=np.float32).reshape(2, 10)
    windows = strided_windows(spec, crop_frames=4, hop_frames=4)

    assert windows.shape == (2, 2, 4, 1)
    np.testing.assert_array_equal(windows[0, :, :, 0], spec[:, 0:4])
    np.testing.assert_array_equal(windows[1, :, :, 0], spec[:, 4:8])


def test_short_spectrogram_is_padded():
    spec = np.ones((64, 30), dtype=np.float32)
    crop = 100
    windows = strided_windows(spec, crop_frames=crop)

    assert windows.shape == (1, 64, crop, 1)
    np.testing.assert_array_equal(windows[0, :, :30, 0], 1.0)
    np.testing.assert_array_equal(windows[0, :, 30:, 0], 0.0)


def test_exact_fit():
    """Spectrogram length is an exact multiple of crop_frames."""
    spec = np.ones((32, 200), dtype=np.float32)
    windows = strided_windows(spec, crop_frames=100)

    assert windows.shape == (2, 32, 100, 1)


def test_single_frame_leftover():
    """One extra frame that can't form a full window."""
    spec = np.ones((32, 201), dtype=np.float32)
    windows = strided_windows(spec, crop_frames=100)

    assert windows.shape == (2, 32, 100, 1)


def test_output_dtype_matches_input():
    spec = np.random.randn(16, 50).astype(np.float64)
    windows = strided_windows(spec, crop_frames=20)
    assert windows.dtype == np.float64
