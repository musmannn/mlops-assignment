"""Bonus - Grad-CAM heat maps for 4 correct and 4 wrong test predictions.

Grad-CAM: take the last convolutional feature map A (H x W x C), compute the
gradient of the output score w.r.t. A, average it over H, W to get one weight
per channel, and sum the channels with those weights -> ReLU -> heat map of the
regions that pushed the prediction up.

    python -m training.gradcam --run mnv2_224
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from training.common import ROOT
from visionguard.preprocess import load_rgb_letterboxed


def gradcam(model: keras.Model, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Model must be a chain ...-> [conv features] -> GlobalAveragePooling2D -> ... -> Dense."""
    layers = model.layers[1:]  # skip InputLayer
    gap = next(i for i, layer in enumerate(layers) if isinstance(layer, keras.layers.GlobalAveragePooling2D))
    with tf.GradientTape() as tape:
        h = tf.convert_to_tensor(x)
        for layer in layers[:gap]:
            h = layer(h, training=False)
        feats = h
        tape.watch(feats)
        for layer in layers[gap:]:
            h = layer(h, training=False)
        score = h[:, 0]
    grads = tape.gradient(score, feats)[0]              # H x W x C
    weights = tf.reduce_mean(grads, axis=(0, 1))       # C
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(score.numpy()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    d = ROOT / "reports" / "figures" / args.run
    model = keras.models.load_model(ROOT / "models" / "runs" / args.run / "model.keras", compile=False)
    size = model.input_shape[1]
    df = pd.read_csv(d / "test_predictions.csv")
    right = df[df.pred == df.label].sample(4, random_state=1)
    wrong = df[df.pred != df.label].head(4)
    picks = pd.concat([right.assign(kind="correct"), wrong.assign(kind="WRONG")])

    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    for j, (_, r) in enumerate(picks.iterrows()):
        x = load_rgb_letterboxed(ROOT / r.path, size)
        cam, p = gradcam(model, x[None])
        heat = tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]
        axes[0, j].imshow(x.astype("uint8"))
        axes[0, j].set_title(f"{r.kind}\ntrue={'ai' if r.label else 'real'} P(AI)={p:.2f}", fontsize=9)
        axes[1, j].imshow(x.astype("uint8"))
        axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM ({args.run}) - top: input, bottom: regions driving P(AI)")
    fig.tight_layout()
    fig.savefig(d / "gradcam.png", dpi=110)
    print("wrote", d / "gradcam.png")


if __name__ == "__main__":
    main()
