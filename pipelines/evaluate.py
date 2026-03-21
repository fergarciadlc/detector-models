# pipelines/evaluate.py
"""
Stage 4: Evaluation

Load best checkpoint, run deterministic strided-window inference on the test
split, compute metrics, generate plots, and log everything to MLflow.

Usage:
    python -m pipelines.evaluate --config configs/experiments/exp_001.yaml
    python -m pipelines.evaluate --config configs/experiments/exp_001.yaml \
        --model-path data/artifacts/specnn_amplitude_baseline/best_model.keras
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import tensorflow as tf

from src.config import load_experiment
from src.evaluation.evaluator import evaluate_test_set
from src.evaluation.metrics import compute_metrics, generate_plots, threshold_sweep
from src.mlflow_logger import MLflowLogger
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
            "If omitted, auto-resolved from data/features/{feature_name}/feature_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation artifacts (default: data/artifacts/{name}/eval)",
    )
    return parser.parse_args(argv)


def _resolve_model_path(experiment_name: str) -> Path:
    path = Path("data/artifacts") / experiment_name / "best_model.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run Stage 3 (train) first."
        )
    return path


def _resolve_feature_manifest(features_name: str) -> Path:
    path = Path("data/features") / features_name / "feature_manifest.json"
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
    logger.info("Experiment: %s", cfg.name)

    # Resolve paths
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = _resolve_model_path(cfg.name)
    logger.info("Model: %s", model_path)

    if args.feature_manifest:
        manifest_path = Path(args.feature_manifest)
    else:
        manifest_path = _resolve_feature_manifest(cfg.features.name)
    logger.info("Feature manifest: %s", manifest_path)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("data/artifacts") / cfg.name / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # Load model
    logger.info("Loading model...")
    model = tf.keras.models.load_model(model_path)

    # Load test entries
    _, entries = load_feature_manifest(manifest_path)
    test_entries = [e for e in entries if e["split"] == "test"]
    if not test_entries:
        raise ValueError("No test entries found in feature manifest")
    logger.info("Test files: %d", len(test_entries))

    # Inference config
    crop_frames = cfg.input_shape[1]
    threshold = cfg.model.output.threshold or 0.5
    logger.info(
        "Crop frames: %d, threshold: %.2f",
        crop_frames, threshold,
    )

    # Run evaluation
    logger.info("Running inference...")
    result = evaluate_test_set(
        model=model,
        entries=test_entries,
        crop_frames=crop_frames,
        threshold=threshold,
        batch_size=cfg.training.batch_size,
    )

    # Compute metrics
    logger.info("Computing metrics: %s", cfg.evaluation.metrics)
    metrics = compute_metrics(
        y_true=result.y_true,
        y_pred_prob=result.y_pred_prob,
        y_pred=result.y_pred,
        metric_names=cfg.evaluation.metrics,
    )

    # Generate plots
    logger.info("Generating plots: %s", cfg.evaluation.plots)
    plot_paths = generate_plots(
        y_true=result.y_true,
        y_pred_prob=result.y_pred_prob,
        y_pred=result.y_pred,
        plot_names=cfg.evaluation.plots,
        output_dir=output_dir,
    )

    # Threshold sweep
    sweep_result = None
    ts_cfg = cfg.evaluation.threshold_sweep
    if ts_cfg.get("enabled"):
        logger.info("Running threshold sweep...")
        sweep_result = threshold_sweep(
            y_true=result.y_true,
            y_pred_prob=result.y_pred_prob,
            t_range=ts_cfg["range"],
            step=ts_cfg["step"],
        )

    # Save evaluation report
    report = {
        "experiment": cfg.name,
        "model_path": str(model_path),
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

    # Log to MLflow
    mlf = MLflowLogger(
        experiment_name="detector-models",
        run_name=f"{cfg.name}-eval",
    )
    mlf.start_run(run_name=f"{cfg.name}-eval")
    mlf.log_params({
        "experiment": cfg.name,
        "threshold": threshold,
        "test_files": len(test_entries),
        "crop_frames": crop_frames,
        "stage": "evaluation",
    })
    mlf.log_metrics(metrics)
    if sweep_result:
        mlf.log_metrics({
            "sweep_best_threshold": sweep_result["best_threshold"],
            "sweep_best_f1": sweep_result["best_f1"],
        })
    mlf.log_artifact(report_path)
    for plot_path in plot_paths:
        mlf.log_artifact(plot_path)
    mlf.end_run()

    # Print summary
    logger.info("-" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 60)
    for name, value in metrics.items():
        logger.info("  %-12s %.4f", name, value)
    if sweep_result:
        logger.info(
            "  %-12s %.4f (at threshold %.2f)",
            "best_f1*", sweep_result["best_f1"], sweep_result["best_threshold"],
        )
    logger.info("-" * 60)
    logger.info("Plots: %s", [str(p) for p in plot_paths])
    logger.info("Report: %s", report_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
