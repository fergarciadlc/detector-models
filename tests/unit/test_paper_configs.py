# tests/unit/test_paper_configs.py
"""Smoke tests: the four experiment configs used in the paper must load.

The paths are listed explicitly (no glob) so that other configs in the repo
cannot make these tests fail.
"""

from pathlib import Path

import pytest
import yaml

from src.config import load_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]

PAPER_EXPERIMENTS = [
    "configs/experiments/exp_music_cnn_mtg.yaml",
    "configs/experiments/exp_music_cnn_mtg_5s.yaml",
    "configs/experiments/regression_sigmoid_mse_windowed_1s.yaml",
    "configs/experiments/regression_sigmoid_mse_windowed_5s.yaml",
]


@pytest.mark.parametrize("rel_path", PAPER_EXPERIMENTS)
def test_paper_experiment_loads(rel_path):
    """Each paper experiment loads and resolves dataset, features and model."""
    exp_path = REPO_ROOT / rel_path
    assert exp_path.is_file()

    raw = yaml.safe_load(exp_path.read_text())
    cfg = load_experiment(exp_path)

    assert cfg.name == raw["experiment"]["name"]

    # Every referenced sub-config exists and was resolved to its own name.
    resolved = {"dataset": cfg.dataset, "features": cfg.features, "model": cfg.model}
    for key, section in resolved.items():
        sub_path = REPO_ROOT / raw[key]
        assert sub_path.is_file(), f"{rel_path}: missing {key} config {raw[key]}"
        assert section.name == yaml.safe_load(sub_path.read_text())[key]["name"]

    # Derived shapes can be computed from the resolved configs.
    n_time, n_freq, channels = cfg.input_shape
    assert n_time > 0
    assert n_freq > 0
    assert channels == 1
    assert cfg.training.crop.window_sec == raw["training"]["crop"]["window_sec"]
