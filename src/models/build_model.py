"""
Dual-input focus classification model builder.

Builds the dual-input (image + head-pose angles) Keras model used by the
training notebook and by the pseudo-labeling pipeline. All preprocessing
lives INSIDE the graph so the saved .h5 is self-sufficient: inference code
only feeds raw data.

Input contract:
    - image_input: (224, 224, 1) grayscale image, pixel values in [0, 255]
    - angle_input: (3,) raw [yaw, pitch, roll] in degrees; 999.0 = no face

Before training, the angle Normalization layer must be adapted on the REAL
training angles only (sentinel 999.0 rows excluded) — see
adapt_angle_normalization().
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CHANNELS, IMG_SIZE

# Name of the angle Normalization layer, exposed so callers can retrieve it
# (model.get_layer(ANGLE_NORM_LAYER_NAME)) to adapt it before training.
ANGLE_NORM_LAYER_NAME = "angle_normalization"


def build_dual_model() -> keras.Model:
    """
    Build the dual-input focus model (frozen MobileNetV2 + angle branch).

    Image branch: grayscale input expanded to 3 channels in-graph, rescaled
    from [0, 255] to [-1, 1] (equivalent to MobileNetV2 preprocess_input),
    then passed through a frozen ImageNet MobileNetV2 backbone with global
    average pooling, followed by Dense(128) + Dropout(0.4).

    Angle branch: Normalization (must be adapted by the caller on real
    training angles before fit) followed by Dense(16) -> Dense(16). The
    sentinel 999.0 maps to a large z-score after normalization, which the
    network can exploit as a "no face detected" signal.

    Fusion: Concatenate -> Dense(64) -> Dropout(0.3) -> Dense(1, sigmoid).
    Output: 0 = focused, 1 = distracted.

    Returns:
        Uncompiled keras.Model with inputs [image_input, angle_input] and a
        single sigmoid output.
    """
    # --- Image branch ---
    image_input = keras.Input(shape=(*IMG_SIZE, CHANNELS), name="image_input")

    # Grayscale -> 3 channels (MobileNetV2 expects RGB-shaped input)
    x = layers.Concatenate(name="grayscale_to_rgb")(
        [image_input, image_input, image_input]
    )

    # MobileNetV2 preprocess_input maps [0, 255] -> [-1, 1]: x / 127.5 - 1.
    # Rescaling is used instead of a Lambda so the .h5 stays portable.
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False  # frozen feature extractor
    # training=False keeps BatchNorm in inference mode even if later unfrozen
    x = backbone(x, training=False)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # --- Angle branch ---
    angle_input = keras.Input(shape=(3,), name="angle_input")
    a = layers.Normalization(axis=-1, name=ANGLE_NORM_LAYER_NAME)(angle_input)
    a = layers.Dense(16, activation="relu")(a)
    a = layers.Dense(16, activation="relu")(a)

    # --- Fusion head ---
    merged = layers.Concatenate()([x, a])
    out = layers.Dense(64, activation="relu")(merged)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(1, activation="sigmoid")(out)

    return keras.Model(
        inputs=[image_input, angle_input], outputs=out, name="focus_dual"
    )


def adapt_angle_normalization(model: keras.Model, angles: np.ndarray) -> None:
    """
    Adapt the angle Normalization layer on real training angles.

    Must be called BEFORE model.fit. The caller is responsible for filtering
    out sentinel rows (999.0 = no face) and for passing TRAINING split angles
    only — adapting on validation data would leak statistics.

    Args:
        model: Model returned by build_dual_model().
        angles: Array of shape (n, 3) with raw [yaw, pitch, roll] values from
            the training split, sentinel rows already excluded.
    """
    model.get_layer(ANGLE_NORM_LAYER_NAME).adapt(angles)


if __name__ == "__main__":
    model = build_dual_model()
    model.summary()
