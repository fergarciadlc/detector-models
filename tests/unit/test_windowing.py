# tests/unit/test_windowing.py
"""Tests for deterministic strided window extraction."""

import numpy as np
import pytest

from src.features.windowing import strided_windows


@pytest.fixture
def spectrogram():
    """A synthetic time-major spectrogram with known dimensions."""
    n_frames, n_freq = 500, 128
    return np.random.randn(n_frames, n_freq).astype(np.float32)


def test_non_overlapping_window_count(spectrogram):
    crop = 100
    windows = strided_windows(spectrogram, crop_frames=crop)

    expected_count = (500 - crop) // crop + 1  # 5
    assert windows.shape[0] == expected_count
    assert windows.shape == (expected_count, crop, 128, 1)


def test_overlapping_window_count(spectrogram):
    crop = 100
    hop = 50
    windows = strided_windows(spectrogram, crop_frames=crop, hop_frames=hop)

    expected_count = len(range(0, 500 - crop + 1, hop))  # 9
    assert windows.shape[0] == expected_count


def test_window_values_are_correct():
    spec = np.arange(20, dtype=np.float32).reshape(10, 2)
    windows = strided_windows(spec, crop_frames=4, hop_frames=4)

    assert windows.shape == (2, 4, 2, 1)
    np.testing.assert_array_equal(windows[0, :, :, 0], spec[0:4, :])
    np.testing.assert_array_equal(windows[1, :, :, 0], spec[4:8, :])


def test_short_spectrogram_is_padded():
    spec = np.ones((30, 64), dtype=np.float32)
    crop = 100
    windows = strided_windows(spec, crop_frames=crop)

    assert windows.shape == (1, crop, 64, 1)
    np.testing.assert_array_equal(windows[0, :30, :, 0], 1.0)
    np.testing.assert_array_equal(windows[0, 30:, :, 0], 0.0)


def test_exact_fit():
    """Spectrogram length is an exact multiple of crop_frames."""
    spec = np.ones((200, 32), dtype=np.float32)
    windows = strided_windows(spec, crop_frames=100)

    assert windows.shape == (2, 100, 32, 1)


def test_single_frame_leftover():
    """One extra frame that can't form a full window."""
    spec = np.ones((201, 32), dtype=np.float32)
    windows = strided_windows(spec, crop_frames=100)

    assert windows.shape == (2, 100, 32, 1)


def test_output_dtype_matches_input():
    spec = np.random.randn(50, 16).astype(np.float64)
    windows = strided_windows(spec, crop_frames=20)
    assert windows.dtype == np.float64
