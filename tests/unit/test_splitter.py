# tests/unit/test_splitter.py
"""Tests for stratified splitter."""

from src.data.scanner import AudioFile
from src.data.splitter import stratified_split
from pathlib import Path


def _make_files(n_real: int, n_fake: int) -> list[AudioFile]:
    """Helper to create dummy AudioFile lists."""
    files = []
    for i in range(n_real):
        files.append(
            AudioFile(
                path=Path(f"/data/real/song_{i:03d}.mp3"),
                label=0,
                label_name="real",
            )
        )
    for i in range(n_fake):
        files.append(
            AudioFile(
                path=Path(f"/data/fake/song_{i:03d}.mp3"),
                label=1,
                label_name="fake",
            )
        )
    return files


def test_split_ratios():
    files = _make_files(100, 100)
    split = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)

    assert len(split.train) == 140  # 70 + 70
    assert len(split.val) == 20  # 10 + 10
    assert len(split.test) == 40  # 20 + 20


def test_split_preserves_all_files():
    files = _make_files(50, 50)
    split = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)

    total = len(split.train) + len(split.val) + len(split.test)
    assert total == 100


def test_split_stratified():
    """Each split should have roughly the same class ratio."""
    files = _make_files(100, 100)
    split = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)

    for split_files in [split.train, split.val, split.test]:
        n_real = sum(1 for f in split_files if f.label == 0)
        n_fake = sum(1 for f in split_files if f.label == 1)
        assert n_real == n_fake  # perfectly balanced with equal class sizes


def test_split_reproducible():
    files = _make_files(100, 100)
    split_a = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)
    split_b = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)

    assert [f.path for f in split_a.train] == [f.path for f in split_b.train]
    assert [f.path for f in split_a.test] == [f.path for f in split_b.test]


def test_split_different_seeds_differ():
    files = _make_files(100, 100)
    split_a = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)
    split_b = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=99)

    assert [f.path for f in split_a.train] != [f.path for f in split_b.train]


def test_split_summary():
    files = _make_files(100, 100)
    split = stratified_split(files, ratios=[0.7, 0.1, 0.2], seed=42)
    summary = split.summary

    assert set(summary.keys()) == {"train", "val", "test"}
    assert (
        summary["train"]["total"] == summary["train"]["real"] + summary["train"]["fake"]
    )
