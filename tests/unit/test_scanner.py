# tests/unit/test_scanner.py
"""Tests for audio file scanner."""

import pytest

from src.data.scanner import scan_audio_dir, scan_dataset


@pytest.fixture
def audio_dirs(tmp_path):
    """Create fake audio directories with dummy files."""
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    # Create dummy mp3 files
    for i in range(20):
        (real_dir / f"song_{i:03d}.mp3").touch()
        (fake_dir / f"song_{i:03d}.mp3").touch()

    # Create a non-audio file that should be ignored
    (real_dir / "readme.txt").touch()

    return {"real": str(real_dir), "fake": str(fake_dir)}


def test_scan_audio_dir(audio_dirs):
    files = scan_audio_dir(audio_dirs["real"], format="mp3")
    assert len(files) == 20
    assert all(f.suffix == ".mp3" for f in files)


def test_scan_audio_dir_filters_non_audio(audio_dirs):
    files = scan_audio_dir(audio_dirs["real"], format="mp3")
    assert not any(f.name == "readme.txt" for f in files)


def test_scan_audio_dir_not_found():
    with pytest.raises(FileNotFoundError):
        scan_audio_dir("/nonexistent/path")


def test_scan_dataset(audio_dirs):
    files = scan_dataset(paths=audio_dirs, format="mp3")
    assert len(files) == 40
    assert sum(1 for f in files if f.label == 0) == 20
    assert sum(1 for f in files if f.label == 1) == 20


def test_scan_dataset_with_limit(audio_dirs):
    files = scan_dataset(paths=audio_dirs, format="mp3", limit=5)
    assert len(files) == 10  # 5 per class
    assert sum(1 for f in files if f.label == 0) == 5
    assert sum(1 for f in files if f.label == 1) == 5


def test_scan_dataset_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No audio files"):
        scan_dataset(
            paths={"real": str(empty), "fake": str(empty)},
            format="mp3",
        )
