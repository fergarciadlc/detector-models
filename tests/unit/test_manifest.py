# tests/unit/test_manifest.py
"""Tests for manifest read/write."""

from pathlib import Path

from src.data.manifest import read_manifest, write_manifest
from src.data.scanner import AudioFile
from src.data.splitter import DataSplit


def _make_split() -> DataSplit:
    return DataSplit(
        train=[
            AudioFile(Path("/data/real/s1.mp3"), label=0, label_name="real"),
            AudioFile(Path("/data/fake/s2.mp3"), label=1, label_name="fake"),
        ],
        val=[
            AudioFile(Path("/data/real/s3.mp3"), label=0, label_name="real"),
        ],
        test=[
            AudioFile(Path("/data/fake/s4.mp3"), label=1, label_name="fake"),
        ],
    )


def test_write_and_read_manifest(tmp_path):
    split = _make_split()
    manifest_path = tmp_path / "test_manifest.json"

    write_manifest(
        split=split,
        output_path=manifest_path,
        dataset_name="test_ds",
        metadata={"seed": 42},
    )

    assert manifest_path.exists()

    loaded_split, manifest_data = read_manifest(manifest_path)

    assert manifest_data["dataset"] == "test_ds"
    assert manifest_data["metadata"]["seed"] == 42
    assert len(loaded_split.train) == 2
    assert len(loaded_split.val) == 1
    assert len(loaded_split.test) == 1


def test_roundtrip_preserves_labels(tmp_path):
    split = _make_split()
    manifest_path = tmp_path / "test_manifest.json"

    write_manifest(split, manifest_path, "test_ds")
    loaded_split, _ = read_manifest(manifest_path)

    assert loaded_split.train[0].label == 0
    assert loaded_split.train[0].label_name == "real"
    assert loaded_split.train[1].label == 1
    assert loaded_split.train[1].label_name == "fake"


def test_roundtrip_preserves_paths(tmp_path):
    split = _make_split()
    manifest_path = tmp_path / "test_manifest.json"

    write_manifest(split, manifest_path, "test_ds")
    loaded_split, _ = read_manifest(manifest_path)

    assert loaded_split.train[0].path == Path("/data/real/s1.mp3")


def test_read_missing_manifest():
    import pytest

    with pytest.raises(FileNotFoundError):
        read_manifest("/nonexistent/manifest.json")


def test_summary_in_manifest(tmp_path):
    split = _make_split()
    manifest_path = tmp_path / "test_manifest.json"

    write_manifest(split, manifest_path, "test_ds")
    _, manifest_data = read_manifest(manifest_path)

    assert manifest_data["summary"]["train"]["total"] == 2
    assert manifest_data["summary"]["val"]["total"] == 1
    assert manifest_data["summary"]["test"]["total"] == 1
