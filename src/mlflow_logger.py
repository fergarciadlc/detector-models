# src/mlflow_logger.py
"""
Graceful MLflow wrapper.

Logs parameters, metrics, and artifacts when the MLflow server is reachable.
Falls back to warning-level logs when it isn't, so training/evaluation is
never blocked by a missing tracking server.

Production pattern:
    Training creates a run and saves run metadata (run_id) to disk.
    Evaluation resumes the same run to add eval metrics and artifacts,
    keeping everything in a single run per experiment.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUN_META_FILENAME = "mlflow_run.json"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "detector-models"


def resolve_mlflow_experiment_name(config_value: str | None = None) -> str:
    """Resolve MLflow experiment with env override precedence."""
    env_value = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if env_value:
        return env_value
    if config_value and config_value.strip():
        return config_value.strip()
    return DEFAULT_MLFLOW_EXPERIMENT_NAME


def resolve_mlflow_run_user() -> str | None:
    """Resolve the user shown in MLflow run metadata."""
    # Explicit user for MLflow run metadata.
    run_user = os.environ.get("MLFLOW_RUN_USER", "").strip()
    if run_user:
        return run_user

    # Reuse project-level username if available.
    user_name = os.environ.get("USER_NAME", "").strip()
    if user_name:
        return user_name

    return None


class MLflowLogger:
    """Graceful MLflow wrapper — logs if server is available, warns otherwise."""

    def __init__(self, experiment_name: str):
        self._active = False
        self._run = None
        self._mlflow = None
        self._run_id: str | None = None

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
            logger.info(
                "MLflow connected: %s (experiment: %s)",
                tracking_uri, experiment_name,
            )
        except Exception:
            logger.warning(
                "MLflow unavailable — proceeding without logging",
                exc_info=True,
            )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @staticmethod
    def _timestamped_name(run_name: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{run_name}"

    def start_run(self, run_name: str) -> None:
        """Start a new MLflow run with a timestamped name."""
        if not self._active:
            return
        try:
            stamped = self._timestamped_name(run_name)
            self._run = self._mlflow.start_run(run_name=stamped)
            self._run_id = self._run.info.run_id
            run_user = resolve_mlflow_run_user()
            if run_user:
                self._mlflow.set_tag("mlflow.user", run_user)
            logger.info("MLflow run started: %s (id: %s)", stamped, self._run_id)
        except Exception:
            logger.warning("Failed to start MLflow run", exc_info=True)
            self._active = False

    def resume_run(self, run_id: str) -> None:
        """Resume an existing MLflow run by ID."""
        if not self._active:
            return
        try:
            self._run = self._mlflow.start_run(run_id=run_id)
            self._run_id = run_id
            logger.info("MLflow run resumed: %s", run_id)
        except Exception:
            logger.warning(
                "Failed to resume MLflow run %s — starting a new run",
                run_id, exc_info=True,
            )
            self._active = False

    def log_tags(self, tags: dict[str, Any]) -> None:
        """Set run-level tags. Shown in the Tags column of the MLflow run list."""
        if not self._active:
            return
        try:
            sanitized = {k: str(v)[:500] for k, v in tags.items()}
            self._mlflow.set_tags(sanitized)
        except Exception:
            logger.warning("Failed to log tags to MLflow", exc_info=True)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        try:
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
            logger.warning(
                "Failed to log metrics to MLflow: %s",
                list(metrics.keys()), exc_info=True,
            )

    def log_keras_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a Keras model as an MLflow model artifact.

        Uses mlflow.keras.log_model so the model is stored as a proper MLflow
        model flavor (serveable, registerable) rather than a raw file.

        Args:
            model: Trained Keras model.
            artifact_path: Subdirectory inside the run's artifact store.
            registered_model_name: If set, registers the model in the Model Registry.
            signature: MLflow ModelSignature describing model inputs and outputs.
                       Use mlflow.models.infer_signature() to create one.
            input_example: One or a few sample inputs (numpy array, dict, …).
                           When provided and signature is None, MLflow infers the
                           signature automatically.
            description: Optional free-text description stored in the model's
                         MLmodel metadata and, when a registered_model_name is
                         given, also set on the resulting model version in the
                         MLflow Model Registry.
        """
        if not self._active:
            return
        try:
            import mlflow.keras

            metadata: dict | None = None
            if description or registered_model_name:
                metadata = {}
                if registered_model_name:
                    metadata["model_name"] = registered_model_name
                if description:
                    metadata["description"] = description
            model_info = mlflow.keras.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                metadata=metadata,
            )
            logger.info("Logged Keras model to MLflow (artifact_path=%s)", artifact_path)

            if registered_model_name and model_info.registered_model_version:
                version = str(model_info.registered_model_version)
                try:
                    client = self._mlflow.MlflowClient()
                    if description:
                        client.update_model_version(
                            name=registered_model_name,
                            version=version,
                            description=description,
                        )
                    if tags:
                        for key, value in tags.items():
                            client.set_model_version_tag(
                                name=registered_model_name,
                                version=version,
                                key=key,
                                value=str(value),
                            )
                    logger.info(
                        "Updated registry metadata for %s v%s",
                        registered_model_name, version,
                    )
                except Exception:
                    logger.warning(
                        "Could not update model version metadata in registry",
                        exc_info=True,
                    )
        except Exception:
            logger.warning("Failed to log Keras model to MLflow", exc_info=True)

    def log_dataset_input(
        self,
        dataset: Any,
        context: str = "training",
    ) -> None:
        """Log a dataset as an MLflow run input.

        Datasets logged here appear in the dedicated "Inputs" tab of the run
        in the MLflow UI, with schema, digest, profile, and source lineage.

        Build the dataset object with e.g.:
            mlflow.data.from_tensorflow(features=tf_ds, source=manifest_path)

        Args:
            dataset: An mlflow.data.Dataset instance
                     (TensorFlowDataset, NumpyDataset, PandasDataset, …).
            context: Label that identifies how the dataset is used,
                     e.g. "training" or "validation".
        """
        if not self._active:
            return
        try:
            self._mlflow.log_input(dataset, context=context)
            logger.info("Logged dataset input to MLflow (context=%s)", context)
        except Exception:
            logger.warning(
                "Failed to log dataset input to MLflow (context=%s)",
                context, exc_info=True,
            )

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Log a single file as an artifact.

        Args:
            path: Local path to the file.
            artifact_path: Optional subdirectory in the artifact store.
        """
        if not self._active:
            return
        resolved = Path(path).resolve()
        if not resolved.exists():
            logger.warning("Artifact file not found, skipping: %s", resolved)
            return
        try:
            self._mlflow.log_artifact(str(resolved), artifact_path=artifact_path)
            logger.info("Logged artifact: %s (%d bytes)", resolved.name, resolved.stat().st_size)
        except Exception:
            logger.warning("Failed to log artifact %s to MLflow", resolved, exc_info=True)

    def end_run(self) -> None:
        if not self._active or self._run is None:
            return
        try:
            self._mlflow.end_run()
            logger.info("MLflow run ended: %s", self._run_id)
        except Exception:
            logger.warning("Failed to end MLflow run", exc_info=True)

    def save_run_meta(self, output_dir: Path) -> None:
        """Save run metadata to disk so downstream stages can resume this run."""
        if not self._run_id:
            return
        meta = {
            "run_id": self._run_id,
            "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", ""),
        }
        meta_path = output_dir / RUN_META_FILENAME
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved MLflow run metadata: %s", meta_path)

    @staticmethod
    def load_run_meta(artifact_dir: Path) -> dict | None:
        """Load run metadata saved by a previous stage."""
        meta_path = artifact_dir / RUN_META_FILENAME
        if not meta_path.exists():
            logger.warning("No MLflow run metadata found at %s", meta_path)
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info("Loaded MLflow run metadata: run_id=%s", meta.get("run_id"))
        return meta
