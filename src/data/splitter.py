# src/data/splitter.py
"""
Split audio file listings into train/val/test sets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.scanner import AudioFile

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/val/test splits."""

    train: list[AudioFile]
    val: list[AudioFile]
    test: list[AudioFile]

    @property
    def summary(self) -> dict[str, dict]:
        """Return a summary of split sizes and class distribution.

        For binary datasets, each entry includes `real` and `fake` counts.
        For regression datasets (ai_ratio is set), only `total` is reported
        since the label is a continuous value, not a class.
        """
        result = {}
        for split_name in ("train", "val", "test"):
            files = getattr(self, split_name)
            is_regression = bool(files) and files[0].ai_ratio is not None
            if is_regression:
                result[split_name] = {"total": len(files)}
            else:
                result[split_name] = {
                    "total": len(files),
                    "real": sum(1 for f in files if f.label == 0),
                    "fake": sum(1 for f in files if f.label == 1),
                }
        return result


def stratified_split(
    files: list[AudioFile],
    ratios: list[float],
    seed: int,
) -> DataSplit:
    """Split files into train/val/test preserving class proportions.

    Args:
        files: List of labeled audio files.
        ratios: Three floats [train, val, test] summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        DataSplit with train/val/test lists.
    """
    # Separate by class
    by_class: dict[int, list[AudioFile]] = {}
    for f in files:
        by_class.setdefault(f.label, []).append(f)

    # Deterministic shuffle per class
    import random

    rng = random.Random(seed)
    for label in by_class:
        rng.shuffle(by_class[label])

    train_files: list[AudioFile] = []
    val_files: list[AudioFile] = []
    test_files: list[AudioFile] = []

    train_ratio, val_ratio, _ = ratios

    for label, class_files in by_class.items():
        n = len(class_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remainder goes to test to avoid losing files
        train_files.extend(class_files[:n_train])
        val_files.extend(class_files[n_train : n_train + n_val])
        test_files.extend(class_files[n_train + n_val :])

    split = DataSplit(train=train_files, val=val_files, test=test_files)

    logger.info("Split summary: %s", split.summary)

    return split
