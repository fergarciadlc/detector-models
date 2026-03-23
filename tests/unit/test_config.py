# tests/unit/test_config.py
"""Tests for config loader."""

import textwrap

import pytest
import yaml

from src.config import load_experiment


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project structure with valid configs."""
    # Directory structure
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "features").mkdir(parents=True)
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "experiments").mkdir(parents=True)

    # Dataset config
    (tmp_path / "configs" / "datasets" / "ds.yaml").write_text(
        textwrap.dedent("""\
        dataset:
          name: test_ds
          description: "Test dataset"
          paths:
            real: /data/real
            fake: /data/fake
          format: mp3
          split:
            strategy: stratified
            ratios: [0.7, 0.1, 0.2]
            seed: 42
        """)
    )

    # Features config
    (tmp_path / "configs" / "features" / "feat.yaml").write_text(
        textwrap.dedent("""\
        features:
          name: test_feat
          type: stft_magnitude_db
          audio:
            sample_rate: 44100
            mono: true
            hf_cut_hz: 16000
          stft:
            n_fft: 2048
            win_length: 2048
            hop_length: 512
          normalization:
            strategy: fixed
            mean: -4.0
            std: 3.0
          storage: full
          output_format: npy
        """)
    )

    # Model config
    (tmp_path / "configs" / "models" / "model.yaml").write_text(
        textwrap.dedent("""\
        model:
          name: test_cnn
          type: spectrogram_cnn
          backbone:
            filters:  [16, 32]
            kernels:  [3, 3]
            strides:  [2, 2]
            use_maxpool: true
            activation: relu
            batch_norm: true
            pooling: average
          head:
            dense_units: 64
            activation: swish
            dropout: 0.2
          output:
            type: binary
            activation: sigmoid
            loss: binary_crossentropy
            threshold: 0.5
        """)
    )

    # Experiment config
    (tmp_path / "configs" / "experiments" / "exp.yaml").write_text(
        textwrap.dedent("""\
        experiment:
          name: test_exp
          description: "Test experiment"
          tags:
            task: test

        dataset: configs/datasets/ds.yaml
        features: configs/features/feat.yaml
        model: configs/models/model.yaml

        training:
          epochs: 10
          batch_size: 32
          learning_rate: 0.001
          optimizer: adam
          seed: 42
          crop:
            window_sec: 0.78
            strategy: random
            samples_per_file: 5
          shuffle_buffer_multiplier: 2
          callbacks:
            early_stopping:
              monitor: val_loss
              patience: 5
          augmentation:
            enabled: false

        evaluation:
          metrics: [accuracy, f1]
          plots: [confusion_matrix]
          threshold_sweep:
            enabled: false
        """)
    )

    return tmp_path


def test_load_experiment(project_dir):
    """Test that a valid experiment config loads and resolves correctly."""
    cfg = load_experiment(project_dir / "configs" / "experiments" / "exp.yaml")

    # Top-level
    assert cfg.name == "test_exp"
    assert cfg.tags["task"] == "test"

    # Dataset resolved
    assert cfg.dataset.name == "test_ds"
    assert cfg.dataset.paths["real"] == "/data/real"
    assert cfg.dataset.split.ratios == [0.7, 0.1, 0.2]

    # Features resolved
    assert cfg.features.audio.sample_rate == 44100
    assert cfg.features.stft.n_fft == 2048
    assert cfg.features.normalization.mean == -4.0

    # Model resolved
    assert cfg.model.backbone.filters == [16, 32]
    assert cfg.model.output.type == "binary"

    # Training
    assert cfg.training.batch_size == 32
    assert cfg.training.crop.window_sec == 0.78


def test_input_shape(project_dir):
    """Test computed input_shape property (time-major: n_time, n_freq, channels)."""
    cfg = load_experiment(project_dir / "configs" / "experiments" / "exp.yaml")
    n_time, n_freq, channels = cfg.input_shape

    # feat.yaml applies hf_cut_hz: 16000 → discrete bin count, not full 1025 bins
    assert n_freq == 744
    assert channels == 1
    assert n_time == cfg.features.frames_for_duration(cfg.training.crop.window_sec)


def test_input_shape_with_mel_features(project_dir):
    """Mel configs should report n_mels as the frequency dimension."""
    feat_path = project_dir / "configs" / "features" / "feat.yaml"
    raw = yaml.safe_load(feat_path.read_text())
    raw["features"]["type"] = "mel_spectrogram_db"
    raw["features"]["audio"]["sample_rate"] = 16000
    raw["features"]["audio"]["hf_cut_hz"] = None
    raw["features"]["stft"]["n_fft"] = 2048
    raw["features"]["stft"]["win_length"] = 2048
    raw["features"]["stft"]["hop_length"] = 1024
    raw["features"]["mel"] = {"n_mels": 128, "fmin": 20.0, "fmax": 8000.0, "power": 2.0}
    feat_path.write_text(yaml.dump(raw))

    cfg = load_experiment(project_dir / "configs" / "experiments" / "exp.yaml")
    n_time, n_freq, channels = cfg.input_shape

    assert n_freq == 128
    assert channels == 1
    assert n_time == cfg.features.frames_for_duration(cfg.training.crop.window_sec)


def test_to_flat_dict(project_dir):
    """Test flattening config for MLflow logging."""
    cfg = load_experiment(project_dir / "configs" / "experiments" / "exp.yaml")
    flat = cfg.to_flat_dict()

    assert flat["dataset.name"] == "test_ds"
    assert flat["training.epochs"] == 10
    assert flat["model.backbone.pooling"] == "average"
    assert isinstance(flat["model.backbone.filters"], str)  # lists → str


def test_missing_referenced_config(project_dir):
    """Test that missing referenced config raises FileNotFoundError."""
    exp_path = project_dir / "configs" / "experiments" / "exp.yaml"
    raw = yaml.safe_load(exp_path.read_text())
    raw["dataset"] = "configs/datasets/nonexistent.yaml"
    exp_path.write_text(yaml.dump(raw))

    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_experiment(exp_path)


def test_invalid_split_ratios(project_dir):
    """Test that split ratios not summing to 1.0 raises ValueError."""
    ds_path = project_dir / "configs" / "datasets" / "ds.yaml"
    raw = yaml.safe_load(ds_path.read_text())
    raw["dataset"]["split"]["ratios"] = [0.5, 0.1, 0.1]
    ds_path.write_text(yaml.dump(raw))

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_experiment(project_dir / "configs" / "experiments" / "exp.yaml")


def test_mismatched_backbone_lengths(project_dir):
    """Test that backbone arrays of different lengths raise ValueError."""
    model_path = project_dir / "configs" / "models" / "model.yaml"
    raw = yaml.safe_load(model_path.read_text())
    raw["model"]["backbone"]["filters"] = [16, 32, 64]  # 3 elements
    # kernels and strides still have 2
    model_path.write_text(yaml.dump(raw))

    with pytest.raises(ValueError, match="same length"):
        load_experiment(project_dir / "configs" / "experiments" / "exp.yaml")
