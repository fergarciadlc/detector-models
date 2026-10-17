# src/data/aria_loader.py
"""Load the predefined ARIA split from a JSONL file.

ARIA has a continuous label (ai_ratio ∈ [0, 1]) and a pre-stratified split
encoded directly in the JSONL — there is nothing to compute here.  This module
reads the file, resolves absolute paths, and returns a DataSplit whose
AudioFile entries carry `ai_ratio` as the regression target.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.scanner import AudioFile
from src.data.splitter import DataSplit

logger = logging.getLogger(__name__)


def load_aria_split(
    base_path: str | Path,
    jsonl_path: str | Path,
    limit: int | None = None,
) -> DataSplit:
    """Read the ARIA JSONL split and return a DataSplit.

    Each row in the JSONL carries:
        track_id, mix_path (relative to base_path), ai_ratio, split

    The `limit` argument constrains the number of **tracks** per split (not
    mixes).  All mixes that belong to an accepted track are included, so the
    actual number of AudioFile entries per split will be larger than `limit`.

    Args:
        base_path: Root directory of the ARIA dataset.  mix_path values in
            the JSONL are resolved relative to this.
        jsonl_path: Path to the pre-split JSONL file (e.g. canonical_v1.jsonl).
        limit: Max tracks per split.  None = all tracks.

    Returns:
        DataSplit with train / val / test lists of AudioFile.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
    """
    base_path = Path(base_path)
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"ARIA split file not found: {jsonl_path}")

    SPLITS = ("train", "val", "test")

    # Detect window-level vs track-level schema by sniffing the first row.
    # Window-level (canonical_v3+): one row per (mix, 5s window) with
    # window_start_sec/window_end_sec/local ai_energy_ratio.  We dedupe back
    # to one entry per mix here; per-window data is consumed downstream.
    is_windowed = False
    with open(jsonl_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line:
                _first = json.loads(_line)
                is_windowed = "window_start_sec" in _first
                break

    seen_tracks: dict[str, set[str]] = {s: set() for s in SPLITS}
    files: dict[str, list[AudioFile]] = {s: [] for s in SPLITS}

    if is_windowed:
        # Window-level JSONL (e.g. canonical_v3): one row per (mix, 5s window).
        # The manifest still needs one entry per mix — but no track-level
        # ai_energy_ratio is meaningful here, so we leave it None.  The real
        # per-window labels are read straight from the JSONL by the windowed
        # training/eval pipelines, never from the feature manifest, which
        # avoids confusion between mix-aggregate and per-window values.
        seen_mixes: dict[str, set[str]] = {s: set() for s in SPLITS}
        with open(jsonl_path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                row = json.loads(raw_line)
                split_name = row.get("split")
                if split_name not in SPLITS:
                    continue
                mix_path = row["mix_path"]
                if mix_path in seen_mixes[split_name]:
                    continue

                track_id = row["track_id"]
                if limit is not None and track_id not in seen_tracks[split_name]:
                    if len(seen_tracks[split_name]) >= limit:
                        continue

                seen_tracks[split_name].add(track_id)
                seen_mixes[split_name].add(mix_path)
                files[split_name].append(
                    AudioFile(
                        path=base_path / mix_path,
                        label=-1,
                        label_name="regression",
                        ai_ratio=None,
                    )
                )
    else:
        with open(jsonl_path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                row = json.loads(raw_line)

                split_name = row.get("split")
                if split_name not in SPLITS:
                    continue

                track_id = row["track_id"]

                if limit is not None and track_id not in seen_tracks[split_name]:
                    if len(seen_tracks[split_name]) >= limit:
                        continue

                seen_tracks[split_name].add(track_id)
                files[split_name].append(
                    AudioFile(
                        path=base_path / row["mix_path"],
                        label=-1,
                        label_name="regression",
                        ai_ratio=float(row["ai_energy_ratio"] if "ai_energy_ratio" in row else row["ai_ratio"]),
                    )
                )

    split = DataSplit(
        train=files["train"],
        val=files["val"],
        test=files["test"],
    )

    logger.info(
        "ARIA split loaded: "
        "train=%d mixes (%d tracks), val=%d mixes (%d tracks), test=%d mixes (%d tracks)",
        len(split.train), len(seen_tracks["train"]),
        len(split.val), len(seen_tracks["val"]),
        len(split.test), len(seen_tracks["test"]),
    )
    return split
