#!/usr/bin/env python3
"""
Compute regression metrics for a binary model's predictions against ARIA targets.

Binary models output a fake_score ∈ [0,1] per file. This script joins those
predictions with ARIA ground truth (ai_energy_ratio, ai_ratio) and computes
regression metrics (MAE, RMSE, R², Pearson, Spearman) to evaluate how well
the binary score tracks the continuous mixing proportion.

Usage:
    python scripts/binary_regression_metrics.py \
        --predictions predictions/mtg_binary_w5s/predictions.csv \
        --experiment cnn_mtg_stft_db_44k_fma_medium_encodec_5s

    # Override ARIA splits path or split:
    python scripts/binary_regression_metrics.py \
        --predictions predictions/mtg_binary_w5s/predictions.csv \
        --experiment cnn_mtg_stft_db_44k_fma_medium_encodec_5s \
        --aria-splits data/datasets/aria/splits/canonical_v2.jsonl \
        --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Regex to extract the relative mix_path from an absolute file_path.
# Matches: {uuid}/mixes/{codec}/{U_level}/mix_{name}.mp3
_MIX_PATH_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/mixes/.+\.mp3)$"
)

METRICS = ["mae", "rmse", "r2", "pearson", "spearman"]


def extract_mix_path(file_path: str) -> str | None:
    """Extract the relative ARIA mix_path from an absolute file_path."""
    m = _MIX_PATH_RE.search(file_path)
    return m.group(1) if m else None


def load_aria_ground_truth(
    jsonl_path: str | Path, split: str | None = "test"
) -> dict[str, dict]:
    """Load ARIA canonical splits and return a dict keyed by mix_path."""
    ground_truth = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            num_stems = row.get("num_stems", 1)
            num_ai_stems = row.get("num_ai_stems", 0)
            ai_ratio = num_ai_stems / num_stems if num_stems > 0 else 0.0
            ground_truth[row["mix_path"]] = {
                "ai_energy_ratio": row["ai_energy_ratio"],
                "ai_ratio": ai_ratio,
                "track_id": row.get("track_id"),
                "num_stems": num_stems,
                "num_ai_stems": num_ai_stems,
            }
    return ground_truth


def load_predictions(csv_path: str | Path) -> list[dict]:
    """Load binary model predictions CSV."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute regression metrics for binary model predictions vs ARIA targets"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to binary model predictions.csv",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (used in output CSV)",
    )
    parser.add_argument(
        "--aria-splits",
        type=str,
        default="data/datasets/aria/splits/canonical_v2.jsonl",
        help="Path to ARIA canonical splits JSONL",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to use from the ARIA JSONL (default: test, use 'all' for no filter)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {predictions_dir}/analysis)",
    )
    args = parser.parse_args(argv)

    predictions_path = Path(args.predictions)
    split = None if args.split == "all" else args.split
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else predictions_path.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading predictions from %s", predictions_path)
    predictions = load_predictions(predictions_path)
    logger.info("Loaded %d predictions", len(predictions))

    logger.info("Loading ARIA ground truth from %s (split=%s)", args.aria_splits, args.split)
    ground_truth = load_aria_ground_truth(args.aria_splits, split)
    logger.info("Loaded %d ground truth entries", len(ground_truth))

    # Join predictions with ground truth
    joined = []
    n_unmatched = 0
    for pred in predictions:
        mix_path = extract_mix_path(pred["file_path"])
        if mix_path is None:
            logger.warning("Could not extract mix_path from: %s", pred["file_path"])
            n_unmatched += 1
            continue
        gt = ground_truth.get(mix_path)
        if gt is None:
            n_unmatched += 1
            continue
        joined.append({
            "join_key": mix_path,
            "filename": pred["filename"],
            "fake_score": float(pred["fake_score"]),
            "n_windows": int(pred["n_windows"]),
            "std_score": float(pred["std_score"]),
            "ai_energy_ratio": gt["ai_energy_ratio"],
            "ai_ratio": gt["ai_ratio"],
        })

    logger.info("Joined %d rows (%d unmatched)", len(joined), n_unmatched)
    if not joined:
        raise ValueError("No predictions matched ARIA ground truth — check paths")

    # Save joined predictions
    joined_path = output_dir / "joined_predictions.csv"
    with open(joined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
        writer.writeheader()
        writer.writerows(joined)
    logger.info("Saved joined predictions: %s", joined_path)

    # Compute regression metrics for each ground truth target
    y_pred = np.array([r["fake_score"] for r in joined])
    targets = {
        "ai_energy_ratio": np.array([r["ai_energy_ratio"] for r in joined]),
        "ai_ratio": np.array([r["ai_ratio"] for r in joined]),
    }

    metrics_rows = []
    for target_name, y_true in targets.items():
        metrics = compute_regression_metrics(y_true, y_pred)
        for metric_name, value in metrics.items():
            metrics_rows.append({
                "experiment": args.experiment,
                "ground_truth": target_name,
                "metric": metric_name,
                "value": value,
                "n": len(joined),
            })
            logger.info("  %-20s %-10s %.6f", target_name, metric_name, value)

    # Save metrics CSV
    metrics_path = output_dir / "regression_metrics_vs_aria_targets.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "ground_truth", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    logger.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
