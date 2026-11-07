"""Task 2.7 toy part - Keras Tuner (Hyperband) on Fashion-MNIST.

Searches learning rate and number of hidden units, then prints the best settings.

    python labs/tuner_toy.py
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]

import keras
import keras_tuner as kt
import pandas as pd

(x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
x_tr, x_te = x_tr[:20000] / 255.0, x_te / 255.0
y_tr = y_tr[:20000]


def build(hp: kt.HyperParameters) -> keras.Model:
    units = hp.Int("units", 32, 512, step=32)
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model = keras.Sequential([
        keras.Input((28, 28)), keras.layers.Flatten(),
        keras.layers.Dense(units, activation="relu"), keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=9, factor=3, seed=42,
                     directory=str(ROOT / "tuner_dir"), project_name="toy_fashion", overwrite=True)
tuner.search(x_tr, y_tr, validation_split=0.15, batch_size=256, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(patience=2)])

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.get_best_trials(num_trials=100)]
df = pd.DataFrame(rows).drop(columns=[c for c in ["tuner/bracket", "tuner/round", "tuner/trial_id"] if c in rows[0]],
                             errors="ignore")
df = df.sort_values("val_accuracy", ascending=False)
out = ROOT / "reports" / "tuner_toy_results.csv"
df.to_csv(out, index=False)
print(df.head(10).to_string(index=False))

best_hp = tuner.get_best_hyperparameters(1)[0]
best = tuner.get_best_models(1)[0]
_, test_acc = best.evaluate(x_te, y_te, verbose=0)
print(f"\nBEST: units={best_hp.get('units')} lr={best_hp.get('lr'):.5f}  test_accuracy={test_acc:.4f}")
print(f"{len(df)} trials, table saved to {out}")
