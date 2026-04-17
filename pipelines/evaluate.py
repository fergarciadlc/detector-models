# pipelines/evaluate.py
"""
Stage 4: Evaluation

Load best checkpoint, run deterministic strided-window inference on the test
split, compute metrics, generate plots, and log everything to MLflow.

Usage:
    python -m pipelines.evaluate --config configs/experiments/exp_001.yaml
    python -m pipelines.evaluate --config configs/experiments/exp_001.yaml \
        --model-path data/artifacts/exp_001/best_model.keras
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import tensorflow as tf

from src.config import load_experiment
from src.evaluation.evaluators import create_evaluator
from src.evaluation.metrics import compute_metrics, generate_plots, threshold_sweep
from src.mlflow_logger import MLflowLogger, resolve_mlflow_experiment_name
from src.training.data_loader import load_feature_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: Evaluate trained model on test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to trained model (.keras). "
            "If omitted, auto-resolved from data/artifacts/{name}/best_model.keras"
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        type=str,
        default=None,
        help=(
            "Path to Stage 2 feature manifest JSON. "
            "If omitted, auto-resolved from "
            "data/features/{feature_name}/{dataset_name}/feature_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation artifacts (default: data/artifacts/{name}/eval)",
    )
    return parser.parse_args(argv)


def _save_metrics_csv(metrics: dict[str, float], output_dir: Path) -> Path:
    """Save the evaluation metrics summary as a two-column CSV."""
    path = output_dir / "eval_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, value in metrics.items():
            writer.writerow([name, f"{value:.6f}"])
    logger.info("Saved metrics CSV: %s", path)
    return path


def _resolve_model_path(experiment_name: str) -> Path:
    path = Path("data/artifacts") / experiment_name / "best_model.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run Stage 3 (train) first."
        )
    return path


def _resolve_feature_manifest(features_name: str, dataset_name: str) -> Path:
    path = Path("data/features") / features_name / dataset_name / "feature_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {path}. Run Stage 2 (extract_features) first."
        )
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment(args.config)

    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("=" * 60)
    logger.info("Experiment  : %s", cfg.name)
    logger.info("Output type : %s", cfg.model.output.type)

    # Resolve paths
    model_path = Path(args.model_path) if args.model_path else _resolve_model_path(cfg.name)
    logger.info("Model: %s", model_path)

    if args.feature_manifest:
        manifest_path = Path(args.feature_manifest)
    else:
        manifest_path = _resolve_feature_manifest(cfg.features.name, cfg.dataset.name)
    logger.info("Feature manifest: %s", manifest_path)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("data/artifacts") / cfg.name / "eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # Load model and test entries
    logger.info("Loading model...")
    model = tf.keras.models.load_model(model_path)

    _, entries = load_feature_manifest(manifest_path)
    test_entries = [e for e in entries if e["split"] == "test"]
    if not test_entries:
        raise ValueError("No test entries found in feature manifest")
    logger.info("Test files: %d", len(test_entries))

    # Run evaluation
    crop_frames = cfg.input_shape[0]
    threshold = cfg.model.output.threshold or 0.5
    logger.info("Crop frames: %d, threshold: %.2f", crop_frames, threshold)

    evaluator = create_evaluator(cfg.model.output.type)
    result = evaluator.run(
        model=model,
        entries=test_entries,
        crop_frames=crop_frames,
        threshold=threshold,
        batch_size=cfg.training.batch_size,
    )

    # Compute metrics and generate plots
    metrics = compute_metrics(result, cfg.evaluation.metrics)
    plot_paths = generate_plots(result, cfg.evaluation.plots, output_dir)

    # Threshold sweep (binary models only)
    sweep_result = None
    ts_cfg = cfg.evaluation.threshold_sweep
    if ts_cfg.get("enabled") and result.has_binary:
        logger.info("Running threshold sweep...")
        sweep_result = threshold_sweep(
            y_true=result.y_true_binary,
            y_pred_prob=result.y_pred_binary_prob,
            t_range=ts_cfg["range"],
            step=ts_cfg["step"],
        )

    # Save evaluation report
    report = {
        "experiment": cfg.name,
        "model_path": str(model_path),
        "output_type": cfg.model.output.type,
        "test_files": len(test_entries),
        "threshold": threshold,
        "metrics": metrics,
        "threshold_sweep": sweep_result,
        "file_scores": result.file_scores,
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report: %s", report_path)

    # Save metrics summary as CSV
    metrics_csv = _save_metrics_csv(metrics, output_dir)

    # MLflow — resume training run
    artifact_dir = Path("data/artifacts") / cfg.name
    mlflow_experiment_name = resolve_mlflow_experiment_name(cfg.mlflow_experiment_name)
    mlf = MLflowLogger(experiment_name=mlflow_experiment_name)

    run_meta = MLflowLogger.load_run_meta(artifact_dir)
    if run_meta:
        mlf.resume_run(run_meta["run_id"])
    else:
        logger.warning("No training run found — creating a standalone eval run")
        mlf.start_run(run_name=f"{cfg.name}-eval")

    mlf.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
    if sweep_result:
        mlf.log_metrics({
            "eval_sweep_best_threshold": sweep_result["best_threshold"],
            "eval_sweep_best_f1": sweep_result["best_f1"],
        })
    for plot_path in plot_paths:
        mlf.log_artifact(plot_path, artifact_path="eval")
    mlf.log_artifact(metrics_csv, artifact_path="eval")
    mlf.end_run()

    # Print summary
    logger.info("-" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 60)
    for name, value in metrics.items():
        logger.info("  %-30s %.4f", name, value)
    if sweep_result:
        logger.info(
            "  %-30s %.4f (at threshold %.2f)",
            "best_f1*", sweep_result["best_f1"], sweep_result["best_threshold"],
        )
    logger.info("-" * 60)
    logger.info("Plots: %s", [str(p) for p in plot_paths])
    logger.info("Report: %s", report_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
