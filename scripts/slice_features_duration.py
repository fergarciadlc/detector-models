#!/usr/bin/env python3
"""
Produce a shorter-duration copy of a feature set by slicing the time axis of
each stored .npy spectrogram, without re-decoding audio or recomputing STFTs.

This is how the paper's 8 s binary features (stft_db_44k_8s) were made: the
first 8 s of each full-length stft_db_44k spectrogram. Re-extracting with
max_duration_sec=8 does not give the same features, for two reasons:
extract_features picks a random (path-seeded) start offset when a file is
longer than max_duration_sec, and the dB reference (ref=np.max) is taken over
the extracted segment. Slicing keeps offset 0 and the full-track reference,
and preserves the train/val/test split of the source manifest.

Spectrograms are time-major (n_frames, n_freq) in C-order, so the kept prefix
[:keep_frames] is contiguous and mmap reads only ~keep/total of each file.

Usage:
    python scripts/slice_features_duration.py \
        --src-manifest data/features/stft_db_44k/fma_medium/feature_manifest.json \
        --out-dir data/features/stft_db_44k_8s/fma_medium \
        --feature-name stft_db_44k_8s \
        --keep-sec 8 \
        --min-crop-frames 427 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def keep_frames_for(keep_sec: float, sample_rate: int, hop_length: int) -> int:
    """Frames kept for a given duration (librosa center=True convention)."""
    return 1 + int(keep_sec * sample_rate) // hop_length


def _slice_one(entry: dict, keep: int, out_dir: Path) -> dict | None:
    src = Path(entry["feature_path"])
    dst = out_dir / entry["split"] / entry["label_name"] / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)

    # mmap so only the kept prefix is read off the (slow) source disk.
    arr = np.load(src, mmap_mode="r")
    sliced = np.ascontiguousarray(arr[:keep])
    np.save(dst, sliced)

    new = dict(entry)
    new["feature_path"] = str(dst)
    new["n_frames"] = int(sliced.shape[0])
    if "duration_sec" in entry:
        if entry.get("n_frames"):
            ratio = sliced.shape[0] / entry["n_frames"]
            new["duration_sec"] = round(entry["duration_sec"] * ratio, 3)
        else:
            new["duration_sec"] = entry["duration_sec"]
    return new


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src-manifest", required=True)
    p.add_argument(
        "--out-dir", required=True, help="Output feature root: data/features/{name}/{dataset}"
    )
    p.add_argument(
        "--feature-name", required=True, help="New feature_config name written into the manifest"
    )
    p.add_argument("--keep-sec", type=float, default=8.0)
    p.add_argument(
        "--min-crop-frames",
        type=int,
        default=427,
        help="Drop entries shorter than this (a spectrogram smaller than the train crop "
        "crashes the loader)",
    )
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    m = json.load(open(args.src_manifest))
    sr = m["audio"]["sample_rate"]
    hop = m["stft"]["hop_length"]
    keep = keep_frames_for(args.keep_sec, sr, hop)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = m["entries"]
    usable = [e for e in entries if e.get("n_frames", 0) >= args.min_crop_frames]
    dropped = len(entries) - len(usable)
    logger.info("keep_frames(%.1fs) = %d | %d entries (%d dropped < %d frames)",
                args.keep_sec, keep, len(usable), dropped, args.min_crop_frames)

    new_entries: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_slice_one, e, keep, out_dir): e for e in usable}
        for fut in as_completed(futs):
            new_entries.append(fut.result())
            done += 1
            if done % 2000 == 0:
                logger.info("  %d / %d", done, len(usable))

    by_split = Counter(e["split"] for e in new_entries)
    m["feature_config"] = args.feature_name
    m["max_duration_sec"] = args.keep_sec
    m["entries"] = new_entries
    m["summary"] = {
        "total": len(new_entries),
        "errors": 0,
        "by_split": dict(by_split),
        "sliced_keep_frames": keep,
        "source_manifest": str(args.src_manifest),
    }
    out_manifest = out_dir / "feature_manifest.json"
    json.dump(m, open(out_manifest, "w"))
    logger.info("Wrote %d entries -> %s", len(new_entries), out_manifest)
    logger.info("by_split: %s", dict(by_split))


if __name__ == "__main__":
    main()
