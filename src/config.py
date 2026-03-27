# src/config.py
"""
Config loader for the audio classification pipeline.

Loads an experiment YAML, resolves referenced sub-configs
(dataset, features, model), and returns a structured ExperimentConfig.

Usage:
    from src.config import load_experiment

    cfg = load_experiment("configs/experiments/exp_001.yaml")
    print(cfg.features.stft.n_fft)
    print(cfg.model.backbone.filters)
    print(cfg.training.batch_size)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Helpers ─────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data)}")
    return data


def _resolve_path(ref_path: str, project_root: Path) -> Path:
    """Resolve a config reference path relative to project root."""
    path = Path(ref_path)
    if path.is_absolute():
        resolved = path
    else:
        resolved = (project_root / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Referenced config not found: {resolved}")
    return resolved


def _find_project_root(start: Path) -> Path:
    """Walk up from start until we find a directory containing 'configs/'."""
    current = start.resolve()
    for parent in [current, *list(current.parents)]:
        if (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (no 'configs/' dir) starting from {start}"
    )


# ── Dataset ─────────────────────────────────────────────────────


@dataclass
class SplitConfig:
    strategy: str  # stratified | random | predefined
    ratios: list[float]
    seed: int
    predefined_path: str | None = None

    def __post_init__(self):
        if self.strategy != "predefined" and len(self.ratios) != 3:
            raise ValueError(
                f"Split ratios must have 3 values [train, val, test], got {self.ratios}"
            )
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(self.ratios)}")


@dataclass
class DatasetConfig:
    name: str
    description: str
    paths: dict[str, str]  # {real: ..., fake: ...}
    format: str
    split: SplitConfig


# ── Features ────────────────────────────────────────────────────


@dataclass
class AudioConfig:
    sample_rate: int
    mono: bool
    hf_cut_hz: int | None = None


@dataclass
class STFTConfig:
    n_fft: int
    win_length: int
    hop_length: int


@dataclass
class MelConfig:
    n_mels: int
    fmin: float = 0.0
    fmax: float | None = None
    power: float = 2.0


@dataclass
class NormalizationConfig:
    strategy: str  # fixed | per_sample | global_stats
    mean: float | None = None
    std: float | None = None

    def __post_init__(self):
        if self.strategy == "fixed" and (self.mean is None or self.std is None):
            raise ValueError("Normalization strategy 'fixed' requires both mean and std")


@dataclass
class FeaturesConfig:
    name: str
    type: str  # stft_magnitude_db | mel_spectrogram_db
    audio: AudioConfig
    stft: STFTConfig
    normalization: NormalizationConfig
    storage: str  # full
    output_format: str  # npy
    mel: MelConfig | None = None

    @property
    def n_freq_bins(self) -> int:
        """Number of frequency bins in the spectrogram (after hf_cut if set)."""
        if self.type == "mel_spectrogram_db":
            if self.mel is None:
                raise ValueError("Feature type 'mel_spectrogram_db' requires 'mel' config.")
            return self.mel.n_mels

        full_bins = self.stft.n_fft // 2 + 1
        if self.audio.hf_cut_hz is not None:
            freq_per_bin = self.audio.sample_rate / self.stft.n_fft
            return int(self.audio.hf_cut_hz / freq_per_bin) + 1
        return full_bins

    def frames_for_duration(self, duration_sec: float) -> int:
        """Number of time frames for a given audio duration."""
        n_samples = int(duration_sec * self.audio.sample_rate)
        return max(1, (n_samples - self.stft.win_length) // self.stft.hop_length + 1)


# ── Model ───────────────────────────────────────────────────────


@dataclass
class BackboneConfig:
    filters: list[int]
    kernels: list[int]
    strides: list[int]
    use_maxpool: bool
    activation: str
    batch_norm: bool
    pooling: str  # average | max
    padding: str = "valid"

    def __post_init__(self):
        lengths = [len(self.filters), len(self.kernels), len(self.strides)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Backbone filters/kernels/strides must have the same length, got {lengths}"
            )


@dataclass
class HeadConfig:
    dense_units: int
    activation: str
    dropout: float


@dataclass
class OutputConfig:
    type: str  # binary | multiclass | regression
    activation: str
    loss: str
    threshold: float | None = None
    num_classes: int | None = None
    class_names: list[str] | None = None

    def __post_init__(self):
        if self.type == "multiclass" and self.num_classes is None:
            raise ValueError("Output type 'multiclass' requires num_classes")


@dataclass
class ModelConfig:
    name: str
    type: str  # spectrogram_cnn, etc.
    backbone: BackboneConfig
    head: HeadConfig
    output: OutputConfig


# ── Training ────────────────────────────────────────────────────


@dataclass
class CropConfig:
    window_sec: float
    strategy: str  # random | strided
    samples_per_file: int


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    seed: int
    crop: CropConfig
    shuffle_buffer_multiplier: int
    callbacks: dict[str, Any] = field(default_factory=dict)
    augmentation: dict[str, Any] = field(default_factory=lambda: {"enabled": False})


# ── Evaluation ──────────────────────────────────────────────────


@dataclass
class EvaluationConfig:
    metrics: list[str]
    plots: list[str]
    threshold_sweep: dict[str, Any] = field(default_factory=lambda: {"enabled": False})


# ── Experiment (top-level) ──────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Top-level config that ties everything together."""

    name: str
    description: str
    tags: dict[str, str]

    dataset: DatasetConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    limit: int | None = None  # max files per class for this experiment (null = all)
    mlflow_experiment_name: str | None = None
    config_path: Path | None = None

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Compute model input shape from features + training crop config.

        Returns (n_time_frames, n_freq_bins, 1) for 2D CNN — time-major,
        consistent with stored .npy spectrograms and tf.signal.stft layout.
        """
        n_freq = self.features.n_freq_bins
        n_time = self.features.frames_for_duration(self.training.crop.window_sec)
        return (n_time, n_freq, 1)

    def to_flat_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flatten config to a single-level dict for MLflow param logging.

        Example:
            {"dataset.name": "full_songs_v1", "training.epochs": 100, ...}
        """
        result = {}
        raw = asdict(self)
        raw.pop("config_path", None)

        def _flatten(d: dict, parent_key: str = ""):
            for k, v in d.items():
                key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    _flatten(v, key)
                elif isinstance(v, list):
                    result[key] = str(v)
                else:
                    result[key] = v

        _flatten(raw)
        return result


# ── Loader (public API) ────────────────────────────────────────


def load_experiment(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment config and resolve all referenced sub-configs.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        Fully resolved ExperimentConfig with all sub-configs loaded.

    Raises:
        FileNotFoundError: If any referenced config file is missing.
        ValueError: If config validation fails.
    """
    config_path = Path(config_path).resolve()
    project_root = _find_project_root(config_path)

    raw = _load_yaml(config_path)

    # ── Resolve and load referenced configs ──
    dataset_raw = _load_yaml(_resolve_path(raw["dataset"], project_root))["dataset"]
    features_raw = _load_yaml(_resolve_path(raw["features"], project_root))["features"]
    model_raw = _load_yaml(_resolve_path(raw["model"], project_root))["model"]

    exp = raw["experiment"]
    train = raw["training"]
    eval_ = raw["evaluation"]

    # ── Build structured config ──
    return ExperimentConfig(
        name=exp["name"],
        description=exp["description"],
        limit=exp.get("limit"),
        mlflow_experiment_name=exp.get("mlflow_experiment"),
        tags=exp.get("tags", {}),
        dataset=DatasetConfig(
            name=dataset_raw["name"],
            description=dataset_raw.get("description", ""),
            paths=dataset_raw["paths"],
            format=dataset_raw["format"],
            split=SplitConfig(
                strategy=dataset_raw["split"]["strategy"],
                ratios=dataset_raw["split"]["ratios"],
                seed=dataset_raw["split"]["seed"],
                predefined_path=dataset_raw["split"].get("predefined_path"),
            ),
        ),
        features=FeaturesConfig(
            name=features_raw["name"],
            type=features_raw["type"],
            audio=AudioConfig(
                sample_rate=features_raw["audio"]["sample_rate"],
                mono=features_raw["audio"]["mono"],
                hf_cut_hz=features_raw["audio"].get("hf_cut_hz"),
            ),
            stft=STFTConfig(
                n_fft=features_raw["stft"]["n_fft"],
                win_length=features_raw["stft"]["win_length"],
                hop_length=features_raw["stft"]["hop_length"],
            ),
            mel=(
                MelConfig(
                    n_mels=features_raw["mel"]["n_mels"],
                    fmin=features_raw["mel"].get("fmin", 0.0),
                    fmax=features_raw["mel"].get("fmax"),
                    power=features_raw["mel"].get("power", 2.0),
                )
                if "mel" in features_raw
                else None
            ),
            normalization=NormalizationConfig(
                strategy=features_raw["normalization"]["strategy"],
                mean=features_raw["normalization"].get("mean"),
                std=features_raw["normalization"].get("std"),
            ),
            storage=features_raw["storage"],
            output_format=features_raw["output_format"],
        ),
        model=ModelConfig(
            name=model_raw["name"],
            type=model_raw["type"],
            backbone=BackboneConfig(
                filters=model_raw["backbone"]["filters"],
                kernels=model_raw["backbone"]["kernels"],
                strides=model_raw["backbone"]["strides"],
                use_maxpool=model_raw["backbone"]["use_maxpool"],
                activation=model_raw["backbone"]["activation"],
                batch_norm=model_raw["backbone"]["batch_norm"],
                pooling=model_raw["backbone"]["pooling"],
                padding=model_raw["backbone"].get("padding", "valid"),
            ),
            head=HeadConfig(
                dense_units=model_raw["head"]["dense_units"],
                activation=model_raw["head"]["activation"],
                dropout=model_raw["head"]["dropout"],
            ),
            output=OutputConfig(
                type=model_raw["output"]["type"],
                activation=model_raw["output"]["activation"],
                loss=model_raw["output"]["loss"],
                threshold=model_raw["output"].get("threshold"),
                num_classes=model_raw["output"].get("num_classes"),
                class_names=model_raw["output"].get("class_names"),
            ),
        ),
        training=TrainingConfig(
            epochs=train["epochs"],
            batch_size=train["batch_size"],
            learning_rate=train["learning_rate"],
            optimizer=train["optimizer"],
            seed=train["seed"],
            crop=CropConfig(
                window_sec=train["crop"]["window_sec"],
                strategy=train["crop"]["strategy"],
                samples_per_file=train["crop"]["samples_per_file"],
            ),
            shuffle_buffer_multiplier=train["shuffle_buffer_multiplier"],
            callbacks=train.get("callbacks", {}),
            augmentation=train.get("augmentation", {"enabled": False}),
        ),
        evaluation=EvaluationConfig(
            metrics=eval_["metrics"],
            plots=eval_["plots"],
            threshold_sweep=eval_.get("threshold_sweep", {"enabled": False}),
        ),
        config_path=config_path,
    )
