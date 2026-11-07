"""Task 3.1 / 3.2 - MLflow tool lab on Fashion-MNIST.

Trains a tiny dense network twice with different learning rates. Each run logs
parameters, per-epoch metrics, one plot and the model.

Local:    python labs/mlflow_toy.py            then  mlflow ui  (http://127.0.0.1:5000)
DagsHub:  put MLFLOW_TRACKING_* in .env, then  python labs/mlflow_toy.py --dagshub
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from training.common import load_dotenv

ap = argparse.ArgumentParser()
ap.add_argument("--dagshub", action="store_true", help="log to the URI in .env instead of local ./mlruns")
ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-2])
ap.add_argument("--epochs", type=int, default=5)
args = ap.parse_args()

if args.dagshub:
    load_dotenv()
    assert os.environ.get("MLFLOW_TRACKING_URI"), "MLFLOW_TRACKING_URI missing - fill in .env"
else:
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
print("tracking URI:", mlflow.get_tracking_uri())

(x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
x_tr, x_te = x_tr[:20000] / 255.0, x_te / 255.0
y_tr = y_tr[:20000]

mlflow.set_experiment("toy-fashion-mnist")
for lr in args.lrs:
    keras.utils.set_random_seed(42)
    with mlflow.start_run(run_name=f"toy_lr_{lr}"):
        mlflow.set_tags({"lab": "3.1", "dataset": "fashion_mnist"})
        mlflow.log_params({"lr": lr, "epochs": args.epochs, "units": 128, "batch_size": 128})
        model = keras.Sequential([
            keras.Input((28, 28)), keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        class Log(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)

        h = model.fit(x_tr, y_tr, validation_split=0.1, epochs=args.epochs, batch_size=128, verbose=2,
                      callbacks=[Log()]).history
        test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(h["loss"], label="train")
        ax.plot(h["val_loss"], label="val")
        ax.set_title(f"loss, lr={lr}")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)
        mlflow.tensorflow.log_model(model, name="model", input_example=np.zeros((1, 28, 28), np.float32))
        print(f"lr={lr}: test_accuracy={test_acc:.4f}")
