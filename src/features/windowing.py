# src/features/windowing.py
"""
Deterministic strided window extraction for evaluation.

Unlike training (random crops), evaluation uses fixed-stride windows
so results are reproducible across runs.
"""

from __future__ import annotations

import numpy as np


def strided_windows(
    spec: np.ndarray,
    crop_frames: int,
    hop_frames: int | None = None,
) -> np.ndarray:
    """Extract fixed-stride windows from a spectrogram along the time axis.

    Args:
        spec: (n_freq, n_frames) spectrogram array.
        crop_frames: Width of each window in frames.
        hop_frames: Stride between windows. Defaults to crop_frames (non-overlapping).

    Returns:
        (n_windows, n_freq, crop_frames, 1) array of windowed spectrograms.
    """
    if hop_frames is None:
        hop_frames = crop_frames

    n_freq, n_frames = spec.shape

    if n_frames < crop_frames:
        padded = np.zeros((n_freq, crop_frames), dtype=spec.dtype)
        padded[:, :n_frames] = spec
        return padded[np.newaxis, :, :, np.newaxis]

    starts = list(range(0, n_frames - crop_frames + 1, hop_frames))
    if not starts:
        starts = [0]

    windows = np.stack(
        [spec[:, s : s + crop_frames] for s in starts],
        axis=0,
    )
    return windows[:, :, :, np.newaxis]
