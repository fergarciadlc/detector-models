# pipelines/predict.py
"""
Predict: run a trained model on unlabeled audio and export a predictions report.

Supports three input modes via --input / -i:
  - Single audio file
  - Directory of audio files (add --recursive for subdirectories)
  - Text / list file (.txt or .lst|.list) with one audio path per line

Also accepts --feature-manifest to skip feature extraction entirely and run
inference directly on pre-computed .npy spectrograms from Stage 2.

Outputs are written incrementally (JSONL / CSV) to avoid memory buildup
for large datasets. When --output-dir is omitted, results are streamed to
the terminal instead.

Feature configuration priority:
    1. config_snapshot.yaml found next to the model (auto-discovered, preferred)
       Guarantees feature params match exactly what the model was trained with.
    2. --config experiment YAML (fallback when no snapshot exists)
       A warning is printed; ensure params haven't drifted since training.

Usage:
    # Snapshot auto-discovered — no --config needed
    python -m pipelines.predict \\
        --model-path data/artifacts/specnn_.../best_model.keras \\
        --input /path/to/audio/folder --recursive \\
        --output-dir data/predictions/my_run

    # Single file, terminal output
    python -m pipelines.predict \\
        --model-path data/artifacts/.../best_model.keras \\
        --input /path/to/song.mp3

    # File list
    python -m pipelines.predict \\
        --model-path data/artifacts/.../best_model.keras \\
        --input my_files.txt --output-dir data/predictions/batch_run

    # Pre-extracted features
    python -m pipelines.predict \\
        --model-path data/artifacts/.../best_model.keras \\
        --feature-manifest data/features/stft_db_44k_60s/feature_manifest.json \\
        --output-dir data/predictions/my_run

    # Explicit --config override (e.g. model lives outside the artifact dir)
    python -m pipelines.predict \\
        --model-path /some/path/model.keras \\
        --config configs/experiments/regression_sigmoid_mse_windowed_5s.yaml \\
        --input /path/to/audio/folder
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from src.config import FeaturesConfig, load_experiment, load_features_from_snapshot
from src.inference.predictor import PredictResult, predict_from_audio, predict_from_features
from src.training.data_loader import load_feature_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ── Argument parsing ─────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained model on unlabeled audio and export a predictions report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to experiment config YAML. Optional when config_snapshot.yaml exists "
            "next to the model (auto-discovered and used by default). Required only when "
            "the model was saved without a snapshot."
        ),
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to trained Keras model (.keras)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Input audio: single file, directory, or .txt/.lst/.list file with one path per line"
        ),
    )
    input_group.add_argument(
        "--feature-manifest",
        type=str,
        default=None,
        metavar="PATH",
        help="Pre-extracted feature manifest JSON (skips audio loading and feature extraction)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Scan input directory recursively (only applies when --input is a directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory to write report files. Omit to print results to the terminal.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "both"],
        default="both",
        help="Output file format (default: both)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Include per-window scores in output (JSONL only; omitted from CSV)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes for on-the-fly feature extraction (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Batch size for model.predict (default: 64)",
    )
    return parser.parse_args(argv)


# ── Input resolution ─────────────────────────────────────────────────────────


def _resolve_audio_input(input_str: str, recursive: bool) -> list[Path]:
    """Resolve --input to a flat list of audio file paths."""
    p = Path(input_str)

    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.is_file() and p.suffix.lower() in (".txt", ".lst", ".list"):
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        files = [Path(line.strip()) for line in lines if line.strip()]
        missing = [f for f in files if not f.exists()]
        if missing:
            logger.warning(
                "%d path(s) in list file do not exist and will fail during inference",
                len(missing),
            )
        return files

    if p.is_file():
        if p.suffix.lower() not in _AUDIO_EXTENSIONS:
            logger.warning(
                "File %s has an unrecognised extension; attempting inference anyway", p.name
            )
        return [p]

    if p.is_dir():
        pattern = "**/*" if recursive else "*"
        found = sorted(
            f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS
        )
        if not found:
            raise ValueError(
                f"No audio files found under {p} "
                f"(recognised extensions: {sorted(_AUDIO_EXTENSIONS)}). "
                "Did you mean to pass --recursive?"
            )
        return found

    raise FileNotFoundError(f"Input path not found: {p}")


# ── Incremental writer ────────────────────────────────────────────────────────


class _ReportWriter:
    """Streams prediction results to JSONL / CSV / terminal as they arrive.

    Behaves as an on_result callback: call instance(entry) for each result.
    Call close() when done to flush and close file handles.
    """

    _CSV_FIELDS = [
        "filename",
        "file_path",
        "duration_sec",
        "fake_score",
        "n_windows",
        "min_score",
        "max_score",
        "std_score",
        "error",
    ]

    def __init__(
        self,
        output_dir: Path | None,
        fmt: str,
        verbose: bool,
        total_files: int,
    ) -> None:
        self._output_dir = output_dir
        self._verbose = verbose
        self._total_files = total_files
        self._n_written = 0
        self._count_width = len(str(total_files))

        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer = None

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            if fmt in ("jsonl", "both"):
                jsonl_path = output_dir / "predictions.jsonl"
                self._jsonl_file = open(jsonl_path, "w", encoding="utf-8")  # noqa: SIM115
                logger.info("Writing JSONL → %s", jsonl_path)
            if fmt in ("csv", "both"):
                csv_path = output_dir / "predictions.csv"
                self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._CSV_FIELDS,
                    extrasaction="ignore",
                )
                self._csv_writer.writeheader()
                logger.info("Writing CSV  → %s", csv_path)

    def __call__(self, entry: dict) -> None:
        self._n_written += 1

        if self._jsonl_file is not None:
            self._jsonl_file.write(json.dumps(entry) + "\n")
            self._jsonl_file.flush()

        if self._csv_writer is not None:
            row = {k: entry.get(k) for k in self._CSV_FIELDS}
            self._csv_writer.writerow(row)
            self._csv_file.flush()  # type: ignore[union-attr]

        if self._output_dir is None:
            self._print_terminal(entry)

    def _print_terminal(self, entry: dict) -> None:
        counter = f"[{self._n_written:{self._count_width}}/{self._total_files}]"
        filename = entry["filename"][:45]
        if entry["error"] is not None:
            tqdm.write(f"{counter}  {filename:<45}  ERROR: {entry['error']}")
        else:
            score_str = f"{entry['fake_score']:.4f}"
            std_str = f"{entry.get('std_score', 0.0):.4f}"
            wins_str = str(entry["n_windows"])
            dur = entry.get("duration_sec")
            dur_str = f"  dur={dur:.1f}s" if dur is not None else ""
            tqdm.write(
                f"{counter}  {filename:<45}  fake={score_str}"
                f"  wins={wins_str:>3}  std={std_str}{dur_str}"
            )

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()
        if self._csv_file is not None:
            self._csv_file.close()


# ── Feature config resolution ────────────────────────────────────────────────

_SNAPSHOT_FILENAME = "config_snapshot.yaml"


def _resolve_feature_config(
    model_path: Path,
    config_arg: str | None,
) -> tuple[FeaturesConfig, float, str, str]:
    """Resolve feature config and window size for inference.

    Prefers the config_snapshot.yaml co-located with the model (written by the
    trainer at training time) so feature parameters are guaranteed to match the
    checkpoint.  Falls back to a user-supplied experiment YAML when no snapshot
    is present.

    Args:
        model_path: Path to the .keras model file.
        config_arg: Value of --config, or None if not supplied.

    Returns:
        (features_cfg, window_sec, config_source, config_path_used)
        config_source is "snapshot" or "experiment_yaml".
    """
    snapshot_path = model_path.parent / _SNAPSHOT_FILENAME

    if snapshot_path.exists():
        logger.info("Feature config: snapshot  (%s)", snapshot_path)
        features_cfg, window_sec = load_features_from_snapshot(snapshot_path)
        return features_cfg, window_sec, "snapshot", str(snapshot_path)

    if config_arg is None:
        logger.error(
            "No %s found next to the model and --config was not provided.\n"
            "  Model dir: %s\n"
            "  Pass --config <experiment.yaml> to specify feature parameters.",
            _SNAPSHOT_FILENAME,
            model_path.parent,
        )
        sys.exit(1)

    logger.warning(
        "No %s found next to the model — falling back to --config.\n"
        "  Ensure feature parameters in %s match what the model was trained with.",
        _SNAPSHOT_FILENAME,
        config_arg,
    )
    cfg = load_experiment(config_arg)
    return cfg.features, cfg.training.crop.window_sec, "experiment_yaml", config_arg


# ── Summary helpers ───────────────────────────────────────────────────────────


def _print_summary(result: PredictResult, output_dir: Path | None) -> None:
    summary = result.summary
    logger.info("-" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)
    for k, v in summary.items():
        logger.info("  %-24s %s", k, v)
    if output_dir is not None:
        logger.info("-" * 60)
        logger.info("Output: %s", output_dir)
    logger.info("-" * 60)


def _save_metadata(
    output_dir: Path,
    result: PredictResult,
    model_path: Path,
    config_source: str,
    config_path_used: str,
    feature_config_name: str,
    crop_frames: int,
    window_sec: float,
) -> None:
    metadata = {
        "model_path": str(model_path),
        "config_source": config_source,
        "config_path": config_path_used,
        "feature_config": feature_config_name,
        "crop_frames": crop_frames,
        "window_sec": window_sec,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": result.summary,
    }
    meta_path = output_dir / "predict_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s", meta_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("PREDICT")
    logger.info("=" * 60)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    # Resolve feature config — snapshot next to model takes priority over --config
    features_cfg, window_sec, config_source, config_path_used = _resolve_feature_config(
        model_path, args.config
    )
    crop_frames = features_cfg.frames_for_duration(window_sec)

    logger.info("Feature config: %s  [%s]", features_cfg.name, config_source)
    logger.info("Loading model:  %s", model_path)
    model = tf.keras.models.load_model(model_path)
    logger.info(
        "Window: %d frames (%.2f s at %d Hz)",
        crop_frames,
        window_sec,
        features_cfg.audio.sample_rate,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    # ── Run inference ─────────────────────────────────────────────────────────

    if args.feature_manifest:
        manifest_path = Path(args.feature_manifest)
        if not manifest_path.exists():
            logger.error("Feature manifest not found: %s", manifest_path)
            sys.exit(1)
        _, entries = load_feature_manifest(manifest_path)
        logger.info("Feature manifest: %s (%d entries)", manifest_path, len(entries))

        writer = _ReportWriter(output_dir, args.format, args.verbose, len(entries))
        try:
            result = predict_from_features(
                model=model,
                entries=entries,
                crop_frames=crop_frames,
                batch_size=args.batch_size,
                verbose=args.verbose,
                on_result=writer,
            )
        finally:
            writer.close()

    else:
        try:
            audio_files = _resolve_audio_input(args.input, args.recursive)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        logger.info("Audio files: %d", len(audio_files))

        writer = _ReportWriter(output_dir, args.format, args.verbose, len(audio_files))
        try:
            result = predict_from_audio(
                model=model,
                audio_files=audio_files,
                features_cfg=features_cfg,
                crop_frames=crop_frames,
                batch_size=args.batch_size,
                workers=max(1, args.workers),
                verbose=args.verbose,
                on_result=writer,
            )
        finally:
            writer.close()

    # ── Persist metadata and print summary ───────────────────────────────────

    if output_dir is not None:
        _save_metadata(
            output_dir=output_dir,
            result=result,
            model_path=model_path,
            config_source=config_source,
            config_path_used=config_path_used,
            feature_config_name=features_cfg.name,
            crop_frames=crop_frames,
            window_sec=window_sec,
        )

    _print_summary(result, output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
