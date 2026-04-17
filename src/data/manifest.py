# src/data/manifest.py
"""
Read and write dataset manifests (JSON).

Manifest is the contract between Stage 1 (data prep) and Stage 2
(feature extraction). It contains file paths, labels, and split
assignments.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.scanner import AudioFile
from src.data.splitter import DataSplit

logger = logging.getLogger(__name__)


def _audio_file_to_dict(af: AudioFile) -> dict[str, Any]:
    d: dict[str, Any] = {
        "path": str(af.path),
        "label": af.label,
        "label_name": af.label_name,
    }
    if af.ai_ratio is not None:
        d["ai_energy_ratio"] = af.ai_ratio
    return d


def _dict_to_audio_file(d: dict[str, Any]) -> AudioFile:
    return AudioFile(
        path=Path(d["path"]),
        label=d["label"],
        label_name=d["label_name"],
        ai_ratio=d.get("ai_energy_ratio"),
    )


def write_manifest(
    split: DataSplit,
    output_path: str | Path,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a dataset manifest to JSON.

    Args:
        split: The train/val/test split to serialize.
        output_path: Where to write the manifest file.
        dataset_name: Name of the dataset (for identification).
        metadata: Optional extra metadata to include.

    Returns:
        Path to the written manifest file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": split.summary,
        "splits": {
            "train": [_audio_file_to_dict(f) for f in split.train],
            "val": [_audio_file_to_dict(f) for f in split.val],
            "test": [_audio_file_to_dict(f) for f in split.test],
        },
    }

    if metadata:
        manifest["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", output_path)
    return output_path


def read_manifest(manifest_path: str | Path) -> tuple[DataSplit, dict]:
    """Read a dataset manifest from JSON.

    Args:
        manifest_path: Path to the manifest file.

    Returns:
        Tuple of (DataSplit, metadata_dict).

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    split = DataSplit(
        train=[_dict_to_audio_file(d) for d in manifest["splits"]["train"]],
        val=[_dict_to_audio_file(d) for d in manifest["splits"]["val"]],
        test=[_dict_to_audio_file(d) for d in manifest["splits"]["test"]],
    )

    return split, manifest
