"""Task 2.2 / 2.3 - model builders.

Every model takes raw pixels in [0, 255] and does its own scaling INSIDE the
graph, so the saved .keras file carries its preprocessing with it and serving
cannot get it wrong.
"""

from __future__ import annotations

import keras
from keras import layers

METRICS = lambda: [  # noqa: E731 - fresh metric objects for every compile
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]


def build_baseline_cnn(img_size: int = 224, dropout: float = 0.3) -> keras.Model:
    """Small CNN from scratch: 4 conv blocks -> global average pool -> dropout -> sigmoid.

    - Rescaling(1/255): pixels to [0,1] so gradients are well scaled.
    - Conv2D 3x3: learns local texture/edge filters; channels double per block
      (32->64->128->256) as spatial size halves, keeping compute per block similar.
    - BatchNormalization: stabilises and speeds training, mild regularisation.
    - ReLU: cheap non-linearity without vanishing gradients.
    - MaxPooling2D: halves resolution, adds translation tolerance, grows receptive field.
    - GlobalAveragePooling2D: one number per channel; far fewer parameters than
      Flatten+Dense, which matters with only ~1k training images.
    - Dropout: randomly drops features during training to fight overfitting.
    - Dense(1, sigmoid): probability that the image is AI-generated.
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


BACKBONES = {
    # name: (constructor, preprocessing layer matching what the backbone was trained with,
    #        layer-name prefix of the LAST block)
    "mobilenetv2": (
        keras.applications.MobileNetV2,
        # mobilenet_v2.preprocess_input maps [0,255] -> [-1,1]; identical to this layer.
        lambda: layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess"),
        ("block_16", "Conv_1", "out_relu"),
    ),
    "efficientnetb0": (
        keras.applications.EfficientNetB0,
        # EfficientNet has its own Rescaling+Normalization inside and expects [0,255].
        None,
        ("block7a", "top_"),
    ),
}


def build_transfer(backbone: str = "mobilenetv2", img_size: int = 224, dropout: float = 0.3):
    """Pretrained backbone (ImageNet, no classifier) + our own head. Returns (model, base)."""
    ctor, prep, _ = BACKBONES[backbone]
    base = ctor(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False  # stage 1: frozen feature extractor

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = prep()(inputs) if prep else inputs
    # training=False keeps BatchNorm statistics frozen even when we later unfreeze layers.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


def unfreeze(base: keras.Model, backbone: str, n_layers: int = 0) -> list[str]:
    """Stage 2: unfreeze the last block (n_layers=0) or the last n layers.

    BatchNormalization layers stay frozen: with small batches, updating their
    statistics wrecks the pretrained features.
    """
    _, _, last_block = BACKBONES[backbone]
    base.trainable = True
    if n_layers > 0:
        targets = {layer.name for layer in base.layers[-n_layers:]}
    else:
        targets = {layer.name for layer in base.layers if layer.name.startswith(last_block)}
    for layer in base.layers:
        layer.trainable = layer.name in targets and not isinstance(layer, layers.BatchNormalization)
    return [layer.name for layer in base.layers if layer.trainable and layer.weights]


def trainable_layer_names(model: keras.Model) -> list[str]:
    names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            names += [f"{layer.name}/{sub.name}" for sub in layer.layers if sub.trainable and sub.weights]
        elif layer.trainable and layer.weights:
            names.append(layer.name)
    return names


def layer_table(model: keras.Model) -> list[dict]:
    """Output shape and parameter count per layer (report evidence for Task 2.2)."""
    return [
        {
            "layer": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(layer.output.shape) if hasattr(layer, "output") else "",
            "params": layer.count_params(),
        }
        for layer in model.layers
    ]


def compile_model(model: keras.Model, lr: float) -> None:
    # Binary cross-entropy: the right loss for one sigmoid output (log-likelihood of a Bernoulli label).
    # Adam: adaptive per-parameter steps, robust default for both scratch and fine-tuning.
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=METRICS())
