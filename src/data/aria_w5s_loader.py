# src/data/aria_w5s_loader.py
"""Per-window loader for ARIA-w5s (canonical_v3.jsonl).

Returns the raw 5s-window rows grouped by split.  Each row carries the
mix_path (relative to the dataset base) plus the window's start/end times
on the original track timeline and its local ai_energy_ratio.

This is consumed by the windowed training loader and the windowed
regression evaluator — both join these rows against the saved
spectrograms (which cover only a windowed segment of the original track,
recorded as `start_offset_sec` + `duration_sec` in the feature manifest).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WindowRow:
    """One 5s ground-truth window for a single mix."""

    mix_path: str               # relative to the dataset base directory
    track_id: str
    split: str
    window_index: int
    window_start_sec: float
    window_end_sec: float
    ai_energy_ratio: float
    is_partial: bool


def load_aria_w5s_windows(
    jsonl_path: str | Path,
) -> dict[str, list[WindowRow]]:
    """Load per-split lists of WindowRow from canonical_v3.jsonl.

    Args:
        jsonl_path: Path to a window-level JSONL (must contain the
            `window_start_sec` / `window_end_sec` / `ai_energy_ratio` fields).

    Returns:
        Dict keyed by split name ("train"/"val"/"test") mapping to the
        ordered list of WindowRow entries.

    Raises:
        FileNotFoundError: If the JSONL is missing.
        ValueError: If the JSONL does not look window-level.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"ARIA w5s split file not found: {jsonl_path}")

    SPLITS = ("train", "val", "test")
    rows: dict[str, list[WindowRow]] = {s: [] for s in SPLITS}
    schema_checked = False

    with open(jsonl_path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)

            if not schema_checked:
                if "window_start_sec" not in row:
                    raise ValueError(
                        f"{jsonl_path} does not look window-level "
                        "(missing 'window_start_sec' field)."
                    )
                schema_checked = True

            split_name = row.get("split")
            if split_name not in SPLITS:
                continue

            rows[split_name].append(
                WindowRow(
                    mix_path=row["mix_path"],
                    track_id=row["track_id"],
                    split=split_name,
                    window_index=int(row["window_index"]),
                    window_start_sec=float(row["window_start_sec"]),
                    window_end_sec=float(row["window_end_sec"]),
                    ai_energy_ratio=float(row["ai_energy_ratio"]),
                    is_partial=bool(row.get("is_partial", False)),
                )
            )

    logger.info(
        "ARIA-w5s windows loaded: train=%d, val=%d, test=%d",
        len(rows["train"]),
        len(rows["val"]),
        len(rows["test"]),
    )
    return rows


def join_windows_to_features(
    windows: list[WindowRow],
    feature_entries: list[dict],
    base_path: str | Path,
) -> list[dict]:
    """Pair each WindowRow with the matching feature manifest entry,
    keeping only windows fully contained in the saved spectrogram segment.

    The feature manifest entry must carry `start_offset_sec` and
    `duration_sec` (the saved segment's coverage on the original track
    timeline).  Windows that fall outside `[start_offset_sec,
    start_offset_sec + duration_sec]` or are flagged `is_partial` are
    dropped.

    Args:
        windows: WindowRow list (typically all windows for one split).
        feature_entries: Feature manifest entries for the same split.
        base_path: Dataset base path used to resolve relative mix_paths to
            the absolute paths stored in `entry["source_audio"]`.

    Returns:
        List of dicts, one per usable (entry, window) pair, with the
        information the data pipeline needs:
            feature_path, n_frames, start_offset_sec, duration_sec,
            window_start_sec, window_end_sec, window_index,
            ai_energy_ratio.
    """
    base_path = Path(base_path)
    by_mix: dict[str, dict] = {}
    skipped_no_offset = 0
    for e in feature_entries:
        try:
            rel = str(Path(e["source_audio"]).relative_to(base_path))
        except ValueError:
            # Fall back to the suffix after the base dir name if relative_to
            # fails (e.g. symlinked / different absolute prefix).
            rel = str(e["source_audio"]).split(str(base_path) + "/")[-1]
        if "start_offset_sec" not in e:
            skipped_no_offset += 1
            continue
        by_mix[rel] = e

    if skipped_no_offset > 0:
        logger.warning(
            "%d feature entries missing 'start_offset_sec' — run "
            "pipelines.backfill_feature_offsets first.",
            skipped_no_offset,
        )

    pairs: list[dict] = []
    n_no_match = 0
    n_out_of_range = 0
    n_partial = 0

    for w in windows:
        if w.is_partial:
            n_partial += 1
            continue
        entry = by_mix.get(w.mix_path)
        if entry is None:
            n_no_match += 1
            continue
        offset = float(entry["start_offset_sec"])
        seg_dur = float(entry["duration_sec"])
        if w.window_start_sec < offset or w.window_end_sec > offset + seg_dur:
            n_out_of_range += 1
            continue

        pairs.append(
            {
                "feature_path": entry["feature_path"],
                "n_frames": int(entry["n_frames"]),
                "start_offset_sec": offset,
                "duration_sec": seg_dur,
                "window_start_sec": w.window_start_sec,
                "window_end_sec": w.window_end_sec,
                "window_index": w.window_index,
                "mix_path": w.mix_path,
                "ai_energy_ratio": w.ai_energy_ratio,
            }
        )

    logger.info(
        "Window/feature join: %d kept | %d partial | %d out-of-range | %d no-feature",
        len(pairs),
        n_partial,
        n_out_of_range,
        n_no_match,
    )
    return pairs
