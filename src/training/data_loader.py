# src/training/data_loader.py
"""
tf.data pipeline for loading .npy spectrograms with random cropping.

Reads a feature manifest (produced by Stage 2), loads full spectrograms,
generates random time-axis crops, and produces batched datasets for training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_feature_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    """Load a Stage 2 feature manifest.

    Returns:
        Tuple of (manifest metadata dict, list of entry dicts).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {path}")

    with open(path) as f:
        manifest = json.load(f)

    return manifest, manifest["entries"]


def _filter_entries(entries: list[dict], split: str) -> list[dict]:
    """Filter manifest entries by split name."""
    filtered = [e for e in entries if e["split"] == split]
    if not filtered:
        raise ValueError(f"No entries found for split '{split}'")
    return filtered


def _load_npy(path_bytes: bytes) -> np.ndarray:
    """Load a .npy file from a byte-string path (for tf.numpy_function)."""
    return np.load(path_bytes.decode("utf-8")).astype(np.float32)


def build_dataset(
    entries: list[dict],
    crop_frames: int,
    samples_per_file: int,
    batch_size: int,
    shuffle_buffer: int | None = None,
    seed: int = 42,
    is_training: bool = True,
    output_type: str = "binary",  # "binary" | "multitask" | "regression"
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from feature manifest entries.

    For each .npy spectrogram, generates `samples_per_file` random crops
    along the time axis (axis 0).

    For `output_type == "multitask"` or `"regression"`, the continuous
    `ai_ratio` field from the feature manifest is used as the label source
    (falls back to casting `label` to float for binary-labelled datasets).

    Args:
        entries: List of feature manifest entry dicts (feature_path, label, n_frames).
        crop_frames: Number of time frames per crop window.
        samples_per_file: Number of random crops per spectrogram.
        batch_size: Batch size.
        shuffle_buffer: Shuffle buffer size (None = no shuffle).
        seed: Random seed for reproducibility.
        is_training: If True, shuffle and repeat. If False, single pass.

    Returns:
        tf.data.Dataset yielding (spectrogram_crop, label) batches.
        Shapes: ((batch, crop_frames, n_freq, 1), (batch,))
    """
    paths = [e["feature_path"] for e in entries]
    n_frames_list = [e["n_frames"] for e in entries]

    # For continuous-output types use ai_ratio when present; fall back to label.
    if output_type in ("multitask", "regression"):
        labels = [
            float(e["ai_ratio"]) if e.get("ai_ratio") is not None else float(e["label"])
            for e in entries
        ]
    else:
        labels = [e["label"] for e in entries]

    path_ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels, n_frames_list)
    )

    if is_training:
        path_ds = path_ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    rng = tf.random.Generator.from_seed(seed)

    def _process_entry(path_tensor, label, n_frames):
        spec = tf.numpy_function(_load_npy, [path_tensor], tf.float32)
        # spec shape is (n_frames, n_freq) — time-major .npy layout
        spec.set_shape([None, None])

        crops = _random_crop_spectrogram(spec, n_frames, crop_frames, samples_per_file, rng)
        if output_type == "multitask":
            alpha = tf.cast(label, tf.float32)
            alpha_repeated = tf.repeat(alpha, samples_per_file)
            has_ai = tf.cast(alpha > 0.0, tf.float32)
            has_ai_repeated = tf.repeat(has_ai, samples_per_file)
            out_labels = {
                "regression_output": alpha_repeated,
                "binary_output": has_ai_repeated,
            }
            return tf.data.Dataset.from_tensor_slices((crops, out_labels))
        elif output_type == "regression":
            alpha = tf.cast(label, tf.float32)
            alpha_repeated = tf.repeat(alpha, samples_per_file)
            return tf.data.Dataset.from_tensor_slices((crops, alpha_repeated))
        else:
            label_repeated = tf.repeat(label, samples_per_file)
            return tf.data.Dataset.from_tensor_slices((crops, label_repeated))

    ds = path_ds.interleave(
        _process_entry,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )

    if shuffle_buffer and is_training:
        ds = ds.shuffle(shuffle_buffer, seed=seed)

    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _random_crop_spectrogram(
    spec: tf.Tensor,
    n_frames: tf.Tensor,
    crop_frames: int,
    num_crops: int,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """Extract random time-axis crops from a spectrogram.

    Args:
        spec: (n_time_frames, n_freq) spectrogram tensor.
        n_frames: Number of time frames (scalar).
        crop_frames: Time width of each crop.
        num_crops: Number of crops to extract.
        rng: TF random generator.

    Returns:
        (num_crops, crop_frames, n_freq, 1) tensor.
    """
    max_start = tf.maximum(n_frames - crop_frames, 1)
    starts = rng.uniform(
        shape=[num_crops], minval=0, maxval=tf.cast(max_start, tf.int64), dtype=tf.int64
    )

    def _crop_one(start):
        cropped = spec[start : start + crop_frames, :]
        cropped = tf.ensure_shape(cropped, [crop_frames, None])
        return tf.expand_dims(cropped, axis=-1)  # (crop_frames, n_freq, 1)

    crops = tf.map_fn(_crop_one, starts, fn_output_signature=tf.float32)
    return crops


def build_train_val_datasets(
    feature_manifest_path: str | Path,
    crop_frames: int,
    samples_per_file: int,
    batch_size: int,
    shuffle_buffer_multiplier: int = 2,
    seed: int = 42,
    output_type: str = "binary",  # "binary" or "multitask"
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build train and validation datasets from a feature manifest.

    Args:
        feature_manifest_path: Path to the Stage 2 feature manifest.
        crop_frames: Number of time frames per crop.
        samples_per_file: Crops per spectrogram (training only).
        batch_size: Batch size.
        shuffle_buffer_multiplier: Shuffle buffer = multiplier * batch_size.
        seed: Random seed.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    _, entries = load_feature_manifest(feature_manifest_path)

    train_entries = _filter_entries(entries, "train")
    val_entries = _filter_entries(entries, "val")

    shuffle_buffer = shuffle_buffer_multiplier * batch_size

    logger.info(
        "Building datasets: train=%d files (%d crops), val=%d files (%d crops)",
        len(train_entries),
        len(train_entries) * samples_per_file,
        len(val_entries),
        len(val_entries) * samples_per_file,
    )

    train_ds = build_dataset(
        train_entries,
        crop_frames=crop_frames,
        samples_per_file=samples_per_file,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        is_training=True,
        output_type=output_type,
    )

    val_ds = build_dataset(
        val_entries,
        crop_frames=crop_frames,
        samples_per_file=samples_per_file,
        batch_size=batch_size,
        shuffle_buffer=None,
        seed=seed,
        is_training=False,
        output_type=output_type,
    )

    return train_ds, val_ds
