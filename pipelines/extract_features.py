# pipelines/extract_features.py
"""
Stage 2: Feature Extraction

Reads a Stage 1 manifest, loads each audio file, computes a full-length
spectrogram, and saves one .npy per file.  Produces a feature manifest
for Stage 3.

Usage:
    python -m pipelines.extract_features --config configs/experiments/exp_001.yaml
    python -m pipelines.extract_features --config configs/experiments/exp_001.yaml \
        --manifest data/manifests/full_songs_v1_limit100_split.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import FeaturesConfig, load_experiment
from src.data.manifest import read_manifest
from src.data.scanner import AudioFile
from src.features.audio_loader import load_audio
from src.features.spectrogram import compute_spectrogram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Extract spectrograms from audio files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Path to Stage 1 manifest JSON. "
            "If omitted, auto-resolved from data/manifests/{dataset_name}_split.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features",
        help="Root dir for feature output (default: data/features)",
    )
    return parser.parse_args(argv)


def _resolve_manifest(dataset_name: str, manifest_dir: str = "data/manifests") -> Path:
    """Find the most recent manifest matching the dataset name."""
    manifest_dir = Path(manifest_dir)
    candidates = sorted(manifest_dir.glob(f"{dataset_name}*_split.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No manifest found for dataset '{dataset_name}' in {manifest_dir}"
        )
    return candidates[-1]


def _npy_filename(audio_path: Path) -> str:
    """Derive .npy filename from the source audio path.

    Preserves parent dir name to avoid collisions between files with the
    same basename in different subdirectories (e.g. 001/000142.mp3 vs 002/000142.mp3).
    """
    return f"{audio_path.parent.name}_{audio_path.stem}.npy"


def extract_file(
    audio_file: AudioFile,
    features_cfg: FeaturesConfig,
    output_dir: Path,
) -> dict:
    """Load audio, compute spectrogram, save .npy, return entry metadata."""
    y, duration_sec = load_audio(audio_file.path, features_cfg.audio)
    S = compute_spectrogram(y, features_cfg)

    npy_name = _npy_filename(audio_file.path)
    npy_path = output_dir / npy_name
    np.save(npy_path, S)

    return {
        "feature_path": str(npy_path),
        "source_audio": str(audio_file.path),
        "label": audio_file.label,
        "label_name": audio_file.label_name,
        "n_frames": S.shape[1],
        "n_freq_bins": S.shape[0],
        "duration_sec": round(duration_sec, 2),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = load_experiment(args.config)
    features_cfg = cfg.features
    dataset_name = cfg.dataset.name

    logger.info("=" * 60)
    logger.info("STAGE 2: FEATURE EXTRACTION")
    logger.info("=" * 60)
    logger.info("Feature config: %s (%s)", features_cfg.name, features_cfg.type)
    logger.info("Sample rate: %d Hz", features_cfg.audio.sample_rate)
    logger.info("STFT: n_fft=%d, hop=%d", features_cfg.stft.n_fft, features_cfg.stft.hop_length)
    if features_cfg.audio.hf_cut_hz:
        logger.info("HF cut: %d Hz", features_cfg.audio.hf_cut_hz)
    logger.info("Normalization: %s", features_cfg.normalization.strategy)

    # Resolve manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = _resolve_manifest(dataset_name)
    logger.info("Manifest: %s", manifest_path)

    split, manifest_meta = read_manifest(manifest_path)

    # Prepare output dirs
    output_root = Path(args.output_dir) / features_cfg.name
    split_dirs = {}
    for split_name in ("train", "val", "test"):
        d = output_root / split_name
        d.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = d

    # Process each split
    entries: list[dict] = []
    errors: list[dict] = []

    for split_name, files in [("train", split.train), ("val", split.val), ("test", split.test)]:
        logger.info("Processing %s split (%d files)...", split_name, len(files))
        out_dir = split_dirs[split_name]

        for af in tqdm(files, desc=split_name, unit="file"):
            try:
                entry = extract_file(af, features_cfg, out_dir)
                entry["split"] = split_name
                entries.append(entry)
            except Exception:
                logger.exception("Failed to process %s", af.path)
                errors.append({"path": str(af.path), "split": split_name})

    # Write feature manifest
    feature_manifest = {
        "feature_config": features_cfg.name,
        "dataset": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio": {
            "sample_rate": features_cfg.audio.sample_rate,
            "mono": features_cfg.audio.mono,
            "hf_cut_hz": features_cfg.audio.hf_cut_hz,
        },
        "stft": {
            "n_fft": features_cfg.stft.n_fft,
            "hop_length": features_cfg.stft.hop_length,
            "win_length": features_cfg.stft.win_length,
        },
        "normalization": {
            "strategy": features_cfg.normalization.strategy,
            "mean": features_cfg.normalization.mean,
            "std": features_cfg.normalization.std,
        },
        "summary": {
            "total": len(entries),
            "errors": len(errors),
            "by_split": {
                s: sum(1 for e in entries if e["split"] == s)
                for s in ("train", "val", "test")
            },
        },
        "entries": entries,
    }

    if errors:
        feature_manifest["errors"] = errors

    manifest_out = output_root / "feature_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(feature_manifest, f, indent=2)

    # Summary
    logger.info("-" * 60)
    logger.info("Output: %s", output_root)
    logger.info("Total features: %d", len(entries))
    if errors:
        logger.warning("Errors: %d files failed (see feature_manifest.json)", len(errors))
    for s in ("train", "val", "test"):
        count = sum(1 for e in entries if e["split"] == s)
        logger.info("  %-6s: %d files", s, count)
    logger.info("Feature manifest: %s", manifest_out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
