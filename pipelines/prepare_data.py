# pipelines/prepare_data.py
"""
Stage 1: Data Preparation

Scans audio directories, validates files, generates stratified
train/val/test split, and exports a manifest JSON.

Usage:
    python -m pipelines.prepare_data --config configs/experiments/exp_001.yaml
    python -m pipelines.prepare_data --config configs/experiments/exp_001.yaml --limit 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_experiment
from src.data.manifest import write_manifest
from src.data.scanner import scan_dataset
from src.data.splitter import stratified_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: Scan audio files and generate train/val/test split"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Max files per class (overrides dataset config). Useful for quick testing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="Directory to write manifest (default: data/manifests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load config
    cfg = load_experiment(args.config)
    ds = cfg.dataset

    # CLI --limit overrides config limit
    limit = args.limit if args.limit is not None else ds.limit

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("=" * 60)
    logger.info("Dataset: %s", ds.name)
    logger.info("Real audio: %s", ds.paths["real"])
    logger.info("Fake audio: %s", ds.paths["fake"])
    logger.info("Limit per class: %s", limit or "all")

    # Scan
    files = scan_dataset(
        paths=ds.paths,
        format=ds.format,
        limit=limit,
    )

    # Split
    split = stratified_split(
        files=files,
        ratios=ds.split.ratios,
        seed=ds.split.seed,
    )

    # Write manifest
    suffix = f"_limit{limit}" if limit else ""
    manifest_name = f"{ds.name}{suffix}_split.json"
    output_path = Path(args.output_dir) / manifest_name

    write_manifest(
        split=split,
        output_path=output_path,
        dataset_name=ds.name,
        metadata={
            "experiment": cfg.name,
            "limit": limit,
            "split_seed": ds.split.seed,
            "split_strategy": ds.split.strategy,
            "split_ratios": ds.split.ratios,
        },
    )

    # Summary
    logger.info("-" * 60)
    for split_name, stats in split.summary.items():
        logger.info(
            "  %-6s: %4d files (real: %d, fake: %d)",
            split_name,
            stats["total"],
            stats["real"],
            stats["fake"],
        )
    logger.info("-" * 60)
    logger.info("Manifest: %s", output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
