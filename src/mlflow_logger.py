# src/mlflow_logger.py
"""
Graceful MLflow wrapper.

Logs parameters, metrics, and artifacts when the MLflow server is reachable.
Falls back to warning-level logs when it isn't, so training/evaluation is
never blocked by a missing tracking server.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MLflowLogger:
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

    @staticmethod
    def _timestamped_name(run_name: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{run_name}"

    def start_run(self, run_name: str) -> None:
        if not self._active:
            return
        try:
            stamped = self._timestamped_name(run_name)
            self._run = self._mlflow.start_run(run_name=stamped)
            logger.info("MLflow run started: %s", stamped)
        except Exception:
            logger.warning("Failed to start MLflow run", exc_info=True)
            self._active = False

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
                metrics.keys(), exc_info=True,
            )

    def log_artifact(self, path: str | Path) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception:
            logger.warning("Failed to log artifact to MLflow", exc_info=True)

    def end_run(self) -> None:
        if not self._active or self._run is None:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            logger.warning("Failed to end MLflow run", exc_info=True)
