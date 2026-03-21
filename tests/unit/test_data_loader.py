# tests/unit/test_data_loader.py
"""Tests for tf.data pipeline."""

import json

import numpy as np
import pytest

from src.training.data_loader import build_dataset, load_feature_manifest


@pytest.fixture
def feature_dir(tmp_path):
    """Create synthetic .npy spectrograms and a feature manifest."""
    n_freq = 743
    n_files = 6
    entries = []

    for i in range(n_files):
        n_frames = np.random.randint(200, 500)
        spec = np.random.randn(n_freq, n_frames).astype(np.float32)
        npy_path = tmp_path / f"spec_{i}.npy"
        np.save(npy_path, spec)

        entries.append({
            "feature_path": str(npy_path),
            "source_audio": f"/datasets/fake_{i}.mp3",
            "label": i % 2,
            "label_name": "real" if i % 2 == 0 else "fake",
            "split": "train" if i < 4 else "val",
            "n_frames": n_frames,
            "n_freq_bins": n_freq,
            "duration_sec": round(n_frames * 512 / 44100, 2),
        })

    manifest = {
        "feature_config": "test_features",
        "dataset": "test_dataset",
        "entries": entries,
        "summary": {"total": n_files, "errors": 0},
    }

    manifest_path = tmp_path / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path, entries


def test_load_feature_manifest(feature_dir):
    manifest_path, expected_entries = feature_dir
    manifest, entries = load_feature_manifest(manifest_path)

    assert manifest["feature_config"] == "test_features"
    assert len(entries) == len(expected_entries)


def test_load_feature_manifest_not_found():
    with pytest.raises(FileNotFoundError):
        load_feature_manifest("/nonexistent/manifest.json")


def test_build_dataset_shapes(feature_dir):
    manifest_path, entries = feature_dir
    train_entries = [e for e in entries if e["split"] == "train"]

    crop_frames = 67
    samples_per_file = 3
    batch_size = 2

    ds = build_dataset(
        train_entries,
        crop_frames=crop_frames,
        samples_per_file=samples_per_file,
        batch_size=batch_size,
        shuffle_buffer=10,
        seed=42,
        is_training=False,
    )

    batch = next(iter(ds))
    specs, labels = batch

    assert specs.shape[1] == 743       # n_freq
    assert specs.shape[2] == crop_frames  # crop width
    assert specs.shape[3] == 1          # channel
    assert labels.shape[0] == specs.shape[0]


def test_build_dataset_labels(feature_dir):
    manifest_path, entries = feature_dir
    train_entries = [e for e in entries if e["split"] == "train"]

    ds = build_dataset(
        train_entries,
        crop_frames=67,
        samples_per_file=2,
        batch_size=100,
        seed=42,
        is_training=False,
    )

    all_labels = []
    for _, labels in ds:
        all_labels.extend(labels.numpy().tolist())

    assert set(all_labels).issubset({0, 1})
