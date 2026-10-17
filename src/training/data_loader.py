# src/training/data_loader.py
"""
tf.data pipeline for loading .npy spectrograms with random cropping.

Reads a feature manifest (produced by Stage 2), loads full spectrograms,
generates random time-axis crops, and produces batched datasets for training.

Two modes are supported:

  * `build_train_val_datasets` — random cropping of the full saved
    spectrogram with the track-level `ai_energy_ratio` as the label
    (legacy regression / multitask / binary).
  * `build_windowed_train_val_datasets` — for ARIA-w5s, crops are aligned
    to the canonical 5s-window boundaries and labelled with the per-window
    `ai_energy_ratio`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.aria_w5s_loader import join_windows_to_features, load_aria_w5s_windows

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
    `ai_energy_ratio` field from the feature manifest is used as the label source
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

    # For continuous-output types use ai_energy_ratio when present; fall back to label.
    if output_type in ("multitask", "regression"):
        labels = [
            float(e["ai_energy_ratio"]) if e.get("ai_energy_ratio") is not None else float(e["label"])
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


# ── Windowed (ARIA-w5s) ──────────────────────────────────────────────────────


def _frame_start_for_window(
    window_start_sec: float,
    start_offset_sec: float,
    sample_rate: int,
    hop_length: int,
    crop_frames: int,
    n_frames: int,
) -> int:
    """Map a window's absolute start time to a starting frame in the saved
    spectrogram, clamped so `[start, start+crop_frames]` fits inside it.
    """
    rel_sec = window_start_sec - start_offset_sec
    start_frame = int(round(rel_sec * sample_rate / hop_length))
    start_frame = max(0, min(start_frame, n_frames - crop_frames))
    return start_frame


def _build_windowed_dataset_from_pairs(
    pairs: list[dict],
    crop_frames: int,
    sample_rate: int,
    hop_length: int,
    batch_size: int,
    shuffle_buffer: int | None,
    seed: int,
    is_training: bool,
    strategy: str = "aligned",
    samples_per_file: int | None = None,
) -> tf.data.Dataset:
    """Build a tf.data dataset of (5s spectrogram crop, ai_energy_ratio) pairs.

    Pairs are first grouped by feature_path so each spectrogram is loaded
    once per visit; from each group we then emit either:

      * `aligned`         — every kept window (K crops per mix).
      * `aligned-random`  — `samples_per_file` random windows (resampled
                            each epoch).  If `samples_per_file >= K` for
                            a mix, all K windows are emitted that epoch.

    Crop start frames are precomputed in Python (deterministic), so the TF
    graph just sees per-mix arrays of int starts and float labels.
    """
    if not pairs:
        raise ValueError("No (feature, window) pairs to build dataset from.")

    if strategy not in ("aligned", "aligned-random"):
        raise ValueError(
            f"Unknown windowed strategy {strategy!r}. "
            "Expected 'aligned' or 'aligned-random'."
        )
    if strategy == "aligned-random" and (samples_per_file is None or samples_per_file < 1):
        raise ValueError(
            "strategy='aligned-random' requires samples_per_file >= 1."
        )

    # ── Group pairs by feature_path ──
    groups: dict[str, list[dict]] = {}
    for p in pairs:
        groups.setdefault(p["feature_path"], []).append(p)

    paths: list[str] = []
    starts_padded: list[list[int]] = []
    labels_padded: list[list[float]] = []
    counts: list[int] = []
    k_max = max(len(g) for g in groups.values())

    for feature_path, group in groups.items():
        sf_list = [
            _frame_start_for_window(
                g["window_start_sec"], g["start_offset_sec"],
                sample_rate, hop_length, crop_frames, g["n_frames"],
            )
            for g in group
        ]
        lb_list = [float(g["ai_energy_ratio"]) for g in group]
        k = len(group)
        # Pad to k_max so the tensor is rectangular; padded slots are
        # masked out by `count` and never read.
        sf_list = sf_list + [0] * (k_max - k)
        lb_list = lb_list + [0.0] * (k_max - k)
        paths.append(feature_path)
        starts_padded.append(sf_list)
        labels_padded.append(lb_list)
        counts.append(k)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(paths, dtype=tf.string),
            tf.constant(starts_padded, dtype=tf.int64),
            tf.constant(labels_padded, dtype=tf.float32),
            tf.constant(counts, dtype=tf.int32),
        )
    )

    if is_training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    rng = tf.random.Generator.from_seed(seed)
    take_cap = (
        tf.constant(samples_per_file, dtype=tf.int32)
        if strategy == "aligned-random" and samples_per_file is not None
        else None
    )

    def _expand_group(path_tensor, starts_pad, labels_pad, count):
        # Decide which of the `count` valid windows to emit this epoch.
        if take_cap is not None:
            n_take = tf.minimum(take_cap, count)
            shuffled = tf.random.experimental.stateless_shuffle(
                tf.range(count, dtype=tf.int32),
                seed=rng.make_seeds(2)[0],
            )
            idx = shuffled[:n_take]
        else:
            idx = tf.range(count, dtype=tf.int32)

        starts_use = tf.gather(starts_pad, tf.cast(idx, tf.int32))
        labels_use = tf.gather(labels_pad, tf.cast(idx, tf.int32))

        spec = tf.numpy_function(_load_npy, [path_tensor], tf.float32)
        spec.set_shape([None, None])

        def _crop_one(start):
            c = spec[start : start + crop_frames, :]
            c = tf.ensure_shape(c, [crop_frames, None])
            return tf.expand_dims(c, axis=-1)

        crops = tf.map_fn(_crop_one, starts_use, fn_output_signature=tf.float32)
        return tf.data.Dataset.from_tensor_slices((crops, labels_use))

    ds = ds.interleave(
        _expand_group,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )

    if shuffle_buffer and is_training:
        ds = ds.shuffle(shuffle_buffer, seed=seed)

    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_windowed_train_val_datasets(
    feature_manifest_path: str | Path,
    windows_jsonl_path: str | Path,
    dataset_base_path: str | Path,
    crop_frames: int,
    sample_rate: int,
    hop_length: int,
    batch_size: int,
    shuffle_buffer_multiplier: int = 2,
    seed: int = 42,
    strategy: str = "aligned",
    samples_per_file: int | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """Build train/val datasets aligned to canonical 5s windows.

    Each saved spectrogram covers `[start_offset_sec, start_offset_sec +
    duration_sec]` of the original track.  Windows from `canonical_v3`
    that fall fully inside that segment are kept; each kept window
    becomes one training example, labelled with its own
    `ai_energy_ratio`.

    Args:
        feature_manifest_path: Stage 2 feature manifest (must include
            `start_offset_sec` per entry; regenerate it with
            `pipelines.extract_features` if missing).
        windows_jsonl_path: canonical_v3.jsonl with one row per 5s window.
        dataset_base_path: Dataset base path (used to resolve absolute
            `source_audio` paths back to relative `mix_path` keys).
        crop_frames: Width of each crop in time frames (must match
            `cfg.input_shape[0]`).
        sample_rate, hop_length: STFT parameters used at extraction time.
        batch_size, shuffle_buffer_multiplier, seed: Standard tf.data knobs.

    Returns:
        (train_ds, val_ds, info) where `info` contains the kept-pair
        counts per split — useful for logging.
    """
    _, entries = load_feature_manifest(feature_manifest_path)
    train_entries = _filter_entries(entries, "train")
    val_entries = _filter_entries(entries, "val")

    windows_by_split = load_aria_w5s_windows(windows_jsonl_path)

    train_pairs = join_windows_to_features(
        windows_by_split["train"], train_entries, dataset_base_path
    )
    val_pairs = join_windows_to_features(
        windows_by_split["val"], val_entries, dataset_base_path
    )

    shuffle_buffer = shuffle_buffer_multiplier * batch_size

    logger.info(
        "Windowed datasets: train=%d windows (from %d feature files), "
        "val=%d windows (from %d feature files)",
        len(train_pairs),
        len(train_entries),
        len(val_pairs),
        len(val_entries),
    )

    train_ds = _build_windowed_dataset_from_pairs(
        train_pairs,
        crop_frames=crop_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        is_training=True,
        strategy=strategy,
        samples_per_file=samples_per_file,
    )
    # Validation always uses 'aligned' so the val score is computed over a
    # stable, complete set of windows (otherwise val_mae would jitter with
    # the per-epoch random subsample).
    val_ds = _build_windowed_dataset_from_pairs(
        val_pairs,
        crop_frames=crop_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        batch_size=batch_size,
        shuffle_buffer=None,
        seed=seed,
        is_training=False,
        strategy="aligned",
        samples_per_file=None,
    )

    info = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_files": len(train_entries),
        "val_files": len(val_entries),
    }
    return train_ds, val_ds, info
