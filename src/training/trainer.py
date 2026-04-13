# src/training/trainer.py
"""
Training orchestrator.

Builds model, compiles, trains with callbacks, and optionally logs to MLflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from src.config import ExperimentConfig
from src.mlflow_logger import MLflowLogger, resolve_mlflow_experiment_name
from src.models.cnn import build_model
from src.training.data_loader import build_train_val_datasets

logger = logging.getLogger(__name__)


def _get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    cls = optimizers.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer: {name}. Supported: {list(optimizers)}")
    return cls(learning_rate=learning_rate)


def _build_callbacks(
    cfg: ExperimentConfig, checkpoint_path: Path
) -> list[tf.keras.callbacks.Callback]:
    callbacks = []
    cb_cfg = cfg.training.callbacks

    if cfg.model.output.type == "multitask":
        monitor_metric = "val_regression_output_mae"
        monitor_mode = "min"
    else:
        monitor_metric = "val_auc_roc"
        monitor_mode = "max"

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "best_model.keras"),
            monitor=monitor_metric,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "last_model.keras"),
            save_best_only=False,
            verbose=0,
        )
    )

    if "early_stopping" in cb_cfg:
        es = cb_cfg["early_stopping"]
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es.get("monitor", "val_loss"),
                patience=es.get("patience", 10),
                restore_best_weights=True,
                verbose=1,
            )
        )

    if "reduce_lr" in cb_cfg:
        rl = cb_cfg["reduce_lr"]
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=rl.get("monitor", "val_loss"),
                factor=rl.get("factor", 0.5),
                patience=rl.get("patience", 5),
                verbose=1,
            )
        )

    return callbacks


class _MLflowKerasCallback(tf.keras.callbacks.Callback):
    """Keras callback that logs metrics to MLflow at the end of each epoch."""

    def __init__(self, mlf_logger: MLflowLogger):
        super().__init__()
        self._mlf = mlf_logger

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self._mlf.log_metrics(logs, step=epoch + 1)


def train(
    cfg: ExperimentConfig,
    feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the full training pipeline.

    Args:
        cfg: Experiment configuration.
        feature_manifest_path: Path to Stage 2 feature manifest.
        output_dir: Root directory for saving artifacts.

    Returns:
        Dict with training results (history, checkpoint_path, epochs_run).
    """
    tf.random.set_seed(cfg.training.seed)

    input_shape = cfg.input_shape
    crop_frames = input_shape[0]

    logger.info("Input shape: %s", input_shape)
    logger.info("Crop frames: %d (%.3fs)", crop_frames, cfg.training.crop.window_sec)

    # Build datasets
    train_ds, val_ds = build_train_val_datasets(
        feature_manifest_path=feature_manifest_path,
        crop_frames=crop_frames,
        samples_per_file=cfg.training.crop.samples_per_file,
        batch_size=cfg.training.batch_size,
        shuffle_buffer_multiplier=cfg.training.shuffle_buffer_multiplier,
        seed=cfg.training.seed,
        output_type=cfg.model.output.type,
    )

    # Build model
    model = build_model(cfg.model, input_shape)
    optimizer = _get_optimizer(cfg.training.optimizer, cfg.training.learning_rate)

    if cfg.model.output.type == "multitask":
        lambda_weight = cfg.model.output.binary_head.get("weight", 0.5)
        model.compile(
            optimizer=optimizer,
            loss={
                "regression_output": cfg.model.output.regression["loss"],
                "binary_output": cfg.model.output.binary_head["loss"],
            },
            loss_weights={
                "regression_output": 1.0,
                "binary_output": lambda_weight,
            },
            metrics={
                "regression_output": ["mae"],
                "binary_output": ["accuracy"],
            },
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=cfg.model.output.loss,
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc_roc")],
        )

    # ── Pretrained weights + backbone freezing ──
    if cfg.training.pretrained_weights:
        weights_path = Path(cfg.training.pretrained_weights)
        logger.info("Loading pretrained weights from: %s", weights_path)

        # Load the pretrained binary model to copy weights from
        pretrained = tf.keras.models.load_model(weights_path, compile=False)

        # Transfer weights for all layers that exist in both models
        transferred = 0
        for layer in model.layers:
            try:
                pretrained_layer = pretrained.get_layer(layer.name)
                layer.set_weights(pretrained_layer.get_weights())
                transferred += 1
            except ValueError:
                # Layer doesn't exist in pretrained model (e.g. new heads)
                pass

        logger.info(
            "Transferred weights for %d/%d layers", transferred, len(model.layers)
        )
        del pretrained  # free memory

    if cfg.training.freeze_backbone_epochs > 0:
        # Freeze all backbone layers (conv_*, bn_*, maxpool_*, global_pool)
        backbone_prefixes = ("conv_", "bn_", "maxpool_", "global_pool")
        frozen_count = 0
        for layer in model.layers:
            if layer.name.startswith(backbone_prefixes):
                layer.trainable = False
                frozen_count += 1
        logger.info("Froze %d backbone layers for %d epochs",
                     frozen_count, cfg.training.freeze_backbone_epochs)

        # Re-compile after changing trainability
        # (reuse the same compile logic from above)
        if cfg.model.output.type == "multitask":
            lambda_weight = cfg.model.output.binary_head.get("weight", 0.5)
            model.compile(
                optimizer=_get_optimizer(
                    cfg.training.optimizer, cfg.training.learning_rate
                ),
                loss={
                    "regression_output": cfg.model.output.regression["loss"],
                    "binary_output": cfg.model.output.binary_head["loss"],
                },
                loss_weights={
                    "regression_output": 1.0,
                    "binary_output": lambda_weight,
                },
                metrics={
                    "regression_output": ["mae"],
                    "binary_output": ["accuracy"],
                },
            )
        else:
            model.compile(
                optimizer=_get_optimizer(
                    cfg.training.optimizer, cfg.training.learning_rate
                ),
                loss=cfg.model.output.loss,
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc_roc")],
            )

    model.summary(print_fn=logger.info)

    # Checkpoint directory
    checkpoint_path = output_dir / cfg.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callbacks = _build_callbacks(cfg, checkpoint_path)

    # MLflow
    mlflow_experiment_name = resolve_mlflow_experiment_name(cfg.mlflow_experiment_name)
    logger.info("MLflow experiment: %s", mlflow_experiment_name)
    mlf = MLflowLogger(experiment_name=mlflow_experiment_name)
    mlf.start_run(run_name=cfg.name)
    mlf.log_params(cfg.to_flat_dict())

    if mlf.active:
        callbacks.append(_MLflowKerasCallback(mlf))

    # Train
    if cfg.training.freeze_backbone_epochs > 0:
        # Phase 1: train with frozen backbone
        logger.info("Phase 1: training heads only for %d epochs",
                     cfg.training.freeze_backbone_epochs)
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.training.freeze_backbone_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Phase 2: unfreeze and fine-tune everything
        for layer in model.layers:
            layer.trainable = True

        # Re-compile with lower learning rate for fine-tuning
        fine_tune_lr = cfg.training.learning_rate * 0.1
        logger.info("Phase 2: fine-tuning all layers (lr=%.6f) for %d epochs",
                     fine_tune_lr,
                     cfg.training.epochs - cfg.training.freeze_backbone_epochs)

        if cfg.model.output.type == "multitask":
            lambda_weight = cfg.model.output.binary_head.get("weight", 0.5)
            model.compile(
                optimizer=_get_optimizer(cfg.training.optimizer, fine_tune_lr),
                loss={
                    "regression_output": cfg.model.output.regression["loss"],
                    "binary_output": cfg.model.output.binary_head["loss"],
                },
                loss_weights={
                    "regression_output": 1.0,
                    "binary_output": lambda_weight,
                },
                metrics={
                    "regression_output": ["mae"],
                    "binary_output": ["accuracy"],
                },
            )
        else:
            model.compile(
                optimizer=_get_optimizer(cfg.training.optimizer, fine_tune_lr),
                loss=cfg.model.output.loss,
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc_roc")],
            )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=cfg.training.freeze_backbone_epochs,
            epochs=cfg.training.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Merge histories for logging
        for key in history_phase1.history:
            history.history[key] = (
                history_phase1.history[key] + history.history[key]
            )
    else:
        # Single-phase training (from scratch or no freezing)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.training.epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # Save training artifacts
    best_model_path = checkpoint_path / "best_model.keras"
    last_model_path = checkpoint_path / "last_model.keras"

    _save_training_history(history, checkpoint_path)
    _save_model_summary(model, checkpoint_path)
    _save_config_snapshot(cfg, checkpoint_path)

    mlf.log_artifact(best_model_path)
    mlf.log_artifact(checkpoint_path / "model_summary.txt")
    mlf.save_run_meta(checkpoint_path)
    mlf.end_run()

    hist = history.history
    if cfg.model.output.type == "multitask":
        result = {
            "checkpoint_path": str(best_model_path),
            "last_checkpoint_path": str(last_model_path),
            "epochs_run": len(hist.get("loss", [])),
            "final_train_loss": hist["loss"][-1],
            "final_val_loss": hist["val_loss"][-1],
            "final_train_mae": hist["regression_output_mae"][-1],
            "final_val_mae": hist["val_regression_output_mae"][-1],
            "final_train_bin_acc": hist["binary_output_accuracy"][-1],
            "final_val_bin_acc": hist["val_binary_output_accuracy"][-1],
        }
    else:
        result = {
            "checkpoint_path": str(best_model_path),
            "last_checkpoint_path": str(last_model_path),
            "epochs_run": len(hist.get("loss", [])),
            "final_train_loss": hist["loss"][-1],
            "final_val_loss": hist["val_loss"][-1],
            "final_train_accuracy": hist["accuracy"][-1],
            "final_val_accuracy": hist["val_accuracy"][-1],
        }

    return result


def _save_training_history(history, checkpoint_path: Path) -> None:
    """Save full epoch-by-epoch metrics for offline analysis."""
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    path = checkpoint_path / "training_history.json"
    with open(path, "w") as f:
        json.dump(hist, f, indent=2)
    logger.info("Saved training history: %s", path)


def _save_model_summary(model: tf.keras.Model, checkpoint_path: Path) -> None:
    """Save model architecture summary to a text file."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path = checkpoint_path / "model_summary.txt"
    path.write_text("\n".join(lines))
    logger.info("Saved model summary: %s", path)


def _save_config_snapshot(cfg: ExperimentConfig, checkpoint_path: Path) -> None:
    """Save a frozen copy of the experiment config for reproducibility."""
    path = checkpoint_path / "config_snapshot.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg.to_flat_dict(), f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config snapshot: %s", path)
