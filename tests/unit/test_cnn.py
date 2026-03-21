# tests/unit/test_cnn.py
"""Tests for CNN model builder."""

import numpy as np
import pytest
import tensorflow as tf

from src.config import BackboneConfig, HeadConfig, ModelConfig, OutputConfig
from src.models.cnn import build_model


@pytest.fixture
def model_cfg():
    """Model config matching spec_cnn_v1.yaml."""
    return ModelConfig(
        name="test_cnn",
        type="spectrogram_cnn",
        backbone=BackboneConfig(
            filters=[16, 32, 64, 128, 256, 512],
            kernels=[3, 3, 3, 3, 3, 3],
            strides=[2, 2, 2, 2, 2, 2],
            use_maxpool=True,
            activation="relu",
            batch_norm=True,
            pooling="average",
        ),
        head=HeadConfig(dense_units=128, activation="swish", dropout=0.2),
        output=OutputConfig(
            type="binary",
            activation="sigmoid",
            loss="binary_crossentropy",
            threshold=0.5,
        ),
    )


@pytest.fixture
def small_model_cfg():
    """Smaller model for fast tests."""
    return ModelConfig(
        name="test_small",
        type="spectrogram_cnn",
        backbone=BackboneConfig(
            filters=[8, 16],
            kernels=[3, 3],
            strides=[2, 2],
            use_maxpool=False,
            activation="relu",
            batch_norm=False,
            pooling="max",
        ),
        head=HeadConfig(dense_units=16, activation="relu", dropout=0.0),
        output=OutputConfig(
            type="binary",
            activation="sigmoid",
            loss="binary_crossentropy",
        ),
    )


INPUT_SHAPE = (743, 67, 1)  # from stft_db_44k + 0.78s crop (hf_cut applied)


def test_build_model_output_shape(model_cfg):
    model = build_model(model_cfg, INPUT_SHAPE)

    assert model.output_shape == (None, 1)
    assert model.input_shape == (None, *INPUT_SHAPE)


def test_forward_pass(model_cfg):
    model = build_model(model_cfg, INPUT_SHAPE)

    dummy = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
    output = model(dummy, training=False)

    assert output.shape == (2, 1)
    assert tf.reduce_all(output >= 0.0) and tf.reduce_all(output <= 1.0)


def test_strided_conv_no_maxpool(small_model_cfg):
    model = build_model(small_model_cfg, INPUT_SHAPE)

    assert model.output_shape == (None, 1)
    layer_names = [layer.name for layer in model.layers]
    assert not any("maxpool" in n for n in layer_names)


def test_model_has_batch_norm(model_cfg):
    model = build_model(model_cfg, INPUT_SHAPE)

    bn_layers = [ly for ly in model.layers if isinstance(ly, tf.keras.layers.BatchNormalization)]
    assert len(bn_layers) == len(model_cfg.backbone.filters)


def test_no_batch_norm(small_model_cfg):
    model = build_model(small_model_cfg, INPUT_SHAPE)

    bn_layers = [ly for ly in model.layers if isinstance(ly, tf.keras.layers.BatchNormalization)]
    assert len(bn_layers) == 0


def test_unknown_pooling():
    cfg = ModelConfig(
        name="test",
        type="spectrogram_cnn",
        backbone=BackboneConfig(
            filters=[8], kernels=[3], strides=[2],
            use_maxpool=False, activation="relu",
            batch_norm=False, pooling="unknown",
        ),
        head=HeadConfig(dense_units=16, activation="relu", dropout=0.0),
        output=OutputConfig(type="binary", activation="sigmoid", loss="bce"),
    )
    with pytest.raises(ValueError, match="Unknown pooling"):
        build_model(cfg, INPUT_SHAPE)
