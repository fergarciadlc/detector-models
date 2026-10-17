# pipelines/train.py
"""
Stage 3: Training

Builds model from config, trains on extracted features with random
cropping, saves best checkpoint, and optionally logs to MLflow.

Usage:
    python -m pipelines.train --config configs/experiments/regression_sigmoid_mse_windowed_5s.yaml
    python -m pipelines.train --config configs/experiments/regression_sigmoid_mse_windowed_5s.yaml \
        --feature-manifest data/features/stft_db_44k_60s/aria_w5s/feature_manifest.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_experiment
from src.training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: Train model on extracted features"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--feature-manifest",
        type=str,
        default=None,
        help=(
            "Path to Stage 2 feature manifest JSON. "
            "If omitted, auto-resolved from data/features/{feature_name}/{dataset_name}/feature_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/artifacts",
        help="Root dir for training artifacts (default: data/artifacts)",
    )
    return parser.parse_args(argv)


def _resolve_feature_manifest(
    features_name: str, dataset_name: str, features_dir: str = "data/features"
) -> Path:
    """Find the feature manifest for a given feature config + dataset combination."""
    path = Path(features_dir) / features_name / dataset_name / "feature_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {path}. Run Stage 2 (extract_features) first."
        )
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = load_experiment(args.config)

    logger.info("=" * 60)
    logger.info("STAGE 3: TRAINING")
    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.name)
    logger.info("Model: %s (%s)", cfg.model.name, cfg.model.type)
    logger.info(
        "Epochs: %d, Batch: %d, LR: %s",
        cfg.training.epochs, cfg.training.batch_size, cfg.training.learning_rate,
    )

    # Resolve feature manifest
    if args.feature_manifest:
        feature_manifest_path = Path(args.feature_manifest)
    else:
        feature_manifest_path = _resolve_feature_manifest(cfg.features.name, cfg.dataset.name)
    logger.info("Feature manifest: %s", feature_manifest_path)

    output_dir = Path(args.output_dir)

    result = train(
        cfg=cfg,
        feature_manifest_path=feature_manifest_path,
        output_dir=output_dir,
    )

    logger.info("-" * 60)
    logger.info("Training complete!")
    logger.info("  Epochs run: %d", result["epochs_run"])
    logger.info("  Final train loss: %.4f", result["final_train_loss"])
    logger.info("  Final val loss:   %.4f", result["final_val_loss"])
    # logger.info("  Final train acc:  %.4f", result["final_train_accuracy"])
    # logger.info("  Final val acc:    %.4f", result["final_val_accuracy"])
    logger.info("  Checkpoint: %s", result["checkpoint_path"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
