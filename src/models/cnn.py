# src/models/cnn.py
"""
Spectrogram CNN model builder.

Ports SimpleSpectrogramCNN from the original codebase using the
Keras functional API, driven entirely by ModelConfig.
"""

from __future__ import annotations

import logging

from tensorflow.keras import Model, layers

from src.config import ModelConfig

logger = logging.getLogger(__name__)


def build_model(model_cfg: ModelConfig, input_shape: tuple[int, int, int]) -> Model:
    """Build a spectrogram CNN from config.

    Args:
        model_cfg: Model configuration (backbone, head, output).
        input_shape: (n_time_frames, n_freq_bins, 1).

    Returns:
        Compiled-ready Keras Model (not yet compiled — caller handles that).
    """
    bb = model_cfg.backbone
    head = model_cfg.head
    out = model_cfg.output

    X = layers.Input(shape=input_shape, name="spectrogram")
    Y = X

    # Backbone: Conv2D blocks
    for i, (filters, kernel, stride) in enumerate(
        zip(bb.filters, bb.kernels, bb.strides)
    ):
        if bb.use_maxpool:
            Y = layers.Conv2D(
                filters,
                kernel_size=kernel,
                padding="same",
                activation=bb.activation,
                name=f"conv_{i}",
            )(Y)
            Y = layers.MaxPooling2D(
                pool_size=stride,
                padding=bb.padding,
                name=f"maxpool_{i}",
            )(Y)
        else:
            Y = layers.Conv2D(
                filters,
                kernel_size=kernel,
                strides=stride,
                padding="same",
                activation=bb.activation,
                name=f"conv_{i}",
            )(Y)

        if bb.batch_norm:
            Y = layers.BatchNormalization(name=f"bn_{i}")(Y)

    # Global pooling
    if bb.pooling == "average":
        Y = layers.GlobalAveragePooling2D(name="global_pool")(Y)
    elif bb.pooling == "max":
        Y = layers.GlobalMaxPooling2D(name="global_pool")(Y)
    else:
        raise ValueError(f"Unknown pooling type: {bb.pooling}")

    # Head
    if head.dropout > 0:
        Y = layers.Dropout(head.dropout, name="head_dropout_1")(Y)

    if head.dense_units > 0:
        Y = layers.Dense(
            head.dense_units,
            activation=head.activation,
            name="head_dense",
        )(Y)
        if head.dropout > 0:
            Y = layers.Dropout(head.dropout, name="head_dropout_2")(Y)

    # Output
    if out.type == "binary":
        Y = layers.Dense(1, activation=out.activation, name="output")(Y)
    elif out.type == "multiclass":
        Y = layers.Dense(out.num_classes, activation=out.activation, name="output")(Y)
    else:
        raise ValueError(f"Unknown output type: {out.type}")

    model = Model(X, Y, name=model_cfg.name)

    logger.info(
        "Built model '%s': %s params (input: %s)",
        model.name,
        f"{model.count_params():,}",
        input_shape,
    )

    return model
