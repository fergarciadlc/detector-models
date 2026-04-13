# tests/unit/test_aria_loader.py
"""Tests for the ARIA predefined-split loader."""

import json
from pathlib import Path

import pytest

from src.data.aria_loader import load_aria_split
from src.data.scanner import AudioFile


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def aria_dir(tmp_path):
    """Minimal ARIA dataset directory with a small JSONL split file."""
    # Create a few dummy mix files so paths could theoretically exist
    (tmp_path / "track_a" / "mixes" / "encodec_3" / "U0").mkdir(parents=True)
    (tmp_path / "track_a" / "mixes" / "encodec_3" / "U1").mkdir(parents=True)
    (tmp_path / "track_b" / "mixes" / "encodec_3" / "U0").mkdir(parents=True)
    (tmp_path / "track_c" / "mixes" / "encodec_3" / "U0").mkdir(parents=True)

    rows = [
        # track_a — 2 mixes in train
        {
            "track_id": "track_a", "split": "train", "ai_ratio": 0.0,
            "mix_path": "track_a/mixes/encodec_3/U0/mix.mp3",
            "num_stems": 4, "genre": "rock",
        },
        {
            "track_id": "track_a", "split": "train", "ai_ratio": 0.25,
            "mix_path": "track_a/mixes/encodec_3/U1/mix.mp3",
            "num_stems": 4, "genre": "rock",
        },
        # track_b — 1 mix in val
        {
            "track_id": "track_b", "split": "val", "ai_ratio": 0.5,
            "mix_path": "track_b/mixes/encodec_3/U0/mix.mp3",
            "num_stems": 4, "genre": "pop",
        },
        # track_c — 1 mix in test
        {
            "track_id": "track_c", "split": "test", "ai_ratio": 1.0,
            "mix_path": "track_c/mixes/encodec_3/U0/mix.mp3",
            "num_stems": 4, "genre": "jazz",
        },
    ]

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    jsonl_path = splits_dir / "test_split.jsonl"
    _write_jsonl(jsonl_path, rows)

    return tmp_path, jsonl_path


def test_split_sizes(aria_dir):
    base, jsonl = aria_dir
    split = load_aria_split(base, jsonl)

    assert len(split.train) == 2
    assert len(split.val) == 1
    assert len(split.test) == 1


def test_ai_ratio_preserved(aria_dir):
    base, jsonl = aria_dir
    split = load_aria_split(base, jsonl)

    ratios = {f.ai_ratio for f in split.train}
    assert ratios == {0.0, 0.25}
    assert split.val[0].ai_ratio == 0.5
    assert split.test[0].ai_ratio == 1.0


def test_paths_resolved(aria_dir):
    base, jsonl = aria_dir
    split = load_aria_split(base, jsonl)

    for af in split.train + split.val + split.test:
        assert af.path.is_absolute()
        assert str(base) in str(af.path)


def test_label_sentinels(aria_dir):
    """label / label_name should be set to regression sentinels."""
    base, jsonl = aria_dir
    split = load_aria_split(base, jsonl)

    for af in split.train + split.val + split.test:
        assert af.label == -1
        assert af.label_name == "regression"


def test_limit_restricts_tracks(aria_dir):
    """limit=1 should keep only the first track encountered per split."""
    base, jsonl = aria_dir
    # track_a has 2 mixes; with limit=1 only track_a's mixes should appear in train
    split = load_aria_split(base, jsonl, limit=1)

    assert len(split.train) == 2  # both mixes from track_a (1 track)
    assert len(split.val) == 1    # track_b (1 track)
    assert len(split.test) == 1   # track_c (1 track)


def test_limit_zero_returns_empty(aria_dir):
    base, jsonl = aria_dir
    split = load_aria_split(base, jsonl, limit=0)

    assert len(split.train) == 0
    assert len(split.val) == 0
    assert len(split.test) == 0


def test_missing_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_aria_split(tmp_path, tmp_path / "nonexistent.jsonl")


def test_summary_has_no_real_fake_keys(aria_dir):
    """DataSplit.summary for regression entries should only contain 'total'."""
    base, jsonl = aria_dir
    split = load_aria_split(base, jsonl)
    summary = split.summary

    for stats in summary.values():
        assert "total" in stats
        assert "real" not in stats
        assert "fake" not in stats
