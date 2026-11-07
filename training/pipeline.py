"""Task 2.1 - tf.data input pipeline.

Stages (in order):
1. from_tensor_slices   - (path, label) pairs read from a split CSV.
2. map(load)            - Pillow decode + EXIF fix + RGB + letterbox to img_size
                          (shared with serving: visionguard.preprocess). Parallel.
3. cache()              - decoded images kept in RAM after the first epoch, so
                          JPEG decoding is paid once, not every epoch.
4. shuffle()            - TRAIN ONLY, full-buffer shuffle reshuffled each epoch.
5. batch()
6. map(augment)         - TRAIN ONLY, random flip/rotation/zoom/contrast, applied
                          after caching so every epoch sees new variations.
7. prefetch(AUTOTUNE)   - prepares the next batch while the model trains on this one.
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from training.common import ROOT
from visionguard.preprocess import load_rgb_letterboxed

AUTOTUNE = tf.data.AUTOTUNE


def augmenter(seed: int) -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.05, fill_mode="constant", seed=seed),  # +-18 degrees
            keras.layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )


def read_split(csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / csv if not Path(csv).is_absolute() else csv)


def make_dataset(
    csv: str | Path,
    img_size: int,
    batch_size: int,
    training: bool,
    augment: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    df = read_split(csv)
    paths = [str(ROOT / p) for p in df["path"]]
    labels = df["label"].astype("float32").to_numpy()

    def _load(path: bytes) -> np.ndarray:
        return load_rgb_letterboxed(path.decode(), img_size)

    def _map(path, label):
        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((img_size, img_size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def labels_of(csv: str | Path) -> np.ndarray:
    return read_split(csv)["label"].to_numpy()
