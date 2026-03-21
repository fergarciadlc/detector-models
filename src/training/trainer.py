# src/training/trainer.py
"""
Training orchestrator.

Builds model, compiles, trains with callbacks, and optionally logs to MLflow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.config import ExperimentConfig
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

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
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


class _MLflowLogger:
    """Graceful MLflow wrapper — logs if server is available, warns otherwise."""

    def __init__(self, experiment_name: str, run_name: str):
        self._active = False
        self._run = None
        self._mlflow = None

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            logger.info("MLFLOW_TRACKING_URI not set — skipping MLflow logging")
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._active = True
            logger.info("MLflow connected: %s (experiment: %s)", tracking_uri, experiment_name)
        except Exception:
            logger.warning(
                "MLflow unavailable — training will proceed without logging",
                exc_info=True,
            )

    def start_run(self, run_name: str) -> None:
        if not self._active:
            return
        try:
            self._run = self._mlflow.start_run(run_name=run_name)
        except Exception:
            logger.warning("Failed to start MLflow run", exc_info=True)
            self._active = False

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        try:
            # MLflow has a 500-char limit on param values
            sanitized = {k: str(v)[:500] for k, v in params.items()}
            self._mlflow.log_params(sanitized)
        except Exception:
            logger.warning("Failed to log params to MLflow", exc_info=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        try:
            sanitized = {k: float(v) for k, v in metrics.items()}
            self._mlflow.log_metrics(sanitized, step=step)
        except Exception:
            logger.warning("Failed to log metrics to MLflow: %s", metrics.keys(), exc_info=True)

    def log_artifact(self, path: str) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(path)
        except Exception:
            logger.warning("Failed to log artifact to MLflow", exc_info=True)

    def end_run(self) -> None:
        if not self._active or self._run is None:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            logger.warning("Failed to end MLflow run", exc_info=True)

    def keras_callback(self) -> tf.keras.callbacks.Callback | None:
        """Return a Keras callback for real-time per-epoch logging."""
        if not self._active:
            return None
        return _MLflowKerasCallback(self)


class _MLflowKerasCallback(tf.keras.callbacks.Callback):
    """Keras callback that logs metrics to MLflow at the end of each epoch."""

    def __init__(self, mlf_logger: _MLflowLogger):
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
    crop_frames = input_shape[1]

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
    )

    # Build model
    model = build_model(cfg.model, input_shape)
    optimizer = _get_optimizer(cfg.training.optimizer, cfg.training.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=cfg.model.output.loss,
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)

    # Checkpoint directory
    checkpoint_path = output_dir / cfg.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callbacks = _build_callbacks(cfg, checkpoint_path)

    # MLflow
    mlf = _MLflowLogger(experiment_name="detector-models", run_name=cfg.name)
    mlf.start_run(run_name=cfg.name)
    mlf.log_params(cfg.to_flat_dict())

    mlf_cb = mlf.keras_callback()
    if mlf_cb:
        callbacks.append(mlf_cb)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.training.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_model_path = str(checkpoint_path / "best_model.keras")
    mlf.log_artifact(best_model_path)
    mlf.end_run()

    result = {
        "checkpoint_path": best_model_path,
        "epochs_run": len(history.history.get("loss", [])),
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "final_train_accuracy": history.history["accuracy"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1],
    }

    return result
