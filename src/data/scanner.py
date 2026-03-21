# src/data/scanner.py
"""
Scan audio directories and return validated file listings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported audio extensions (lowercase, with dot)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class AudioFile:
    """Represents a discovered audio file with its label."""

    path: Path
    label: int  # 0 = real, 1 = fake
    label_name: str  # "real" or "fake"


def scan_audio_dir(
    directory: str | Path,
    format: str | None = None,
) -> list[Path]:
    """Recursively scan a directory for audio files.

    Args:
        directory: Root directory to scan.
        format: If provided, only match this extension (e.g. "mp3").
            If None, match all known audio extensions.

    Returns:
        Sorted list of audio file paths.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if format:
        target_ext = f".{format.lower().lstrip('.')}"
        allowed = {target_ext}
    else:
        allowed = AUDIO_EXTENSIONS

    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in allowed
    )

    logger.info("Found %d audio files in %s", len(files), directory)
    return files


def scan_dataset(
    paths: dict[str, str],
    format: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[AudioFile]:
    """Scan all class directories and return labeled audio files.

    Args:
        paths: Mapping of class name to directory path.
            Expected: {"real": "/path/to/real", "fake": "/path/to/fake"}
        format: Audio file extension filter.
        limit: Max files per class. None = no limit.
            Files are deterministically selected (sorted, then truncated)
            to ensure reproducibility.
        seed: Not used for truncation (sorted order is deterministic),
            reserved for future randomized sampling.

    Returns:
        List of AudioFile entries with labels assigned.

    Raises:
        ValueError: If no audio files found in any directory.
    """
    label_map = {"real": 0, "fake": 1}
    all_files: list[AudioFile] = []

    for class_name, dir_path in paths.items():
        if class_name not in label_map:
            raise ValueError(
                f"Unknown class name '{class_name}'. "
                f"Expected one of: {list(label_map.keys())}"
            )

        files = scan_audio_dir(dir_path, format=format)

        if len(files) == 0:
            raise ValueError(
                f"No audio files found in {dir_path} (format filter: {format})"
            )

        if limit is not None:
            original_count = len(files)
            files = files[:limit]
            logger.info(
                "Limited %s from %d to %d files",
                class_name,
                original_count,
                len(files),
            )

        label = label_map[class_name]
        for f in files:
            all_files.append(AudioFile(path=f, label=label, label_name=class_name))

    logger.info(
        "Total dataset: %d files (%s)",
        len(all_files),
        ", ".join(
            f"{name}: {sum(1 for f in all_files if f.label_name == name)}"
            for name in paths
        ),
    )

    return all_files
