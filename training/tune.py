"""Task 2.7 apply part - Keras Tuner on VisionGuard (MobileNetV2 transfer model).

Searches learning rate, dropout and number of unfrozen backbone layers with
RandomSearch (>= 8 trials). Each trial trains a single stage for a few epochs at
160 px to keep the CPU budget reasonable. The result table is logged to MLflow.

    python -m training.tune --trials 10
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import keras
import keras_tuner as kt
import mlflow
import pandas as pd
from keras import layers

from training.common import ROOT, git_commit, load_dotenv, load_params, set_seeds
from training.models import METRICS, build_transfer
from training.pipeline import make_dataset


def build(hp: kt.HyperParameters, img_size: int) -> keras.Model:
    lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4, 3e-5])
    dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
    n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
    model, base = build_transfer("mobilenetv2", img_size, dropout)
    if n_unfrozen:
        base.trainable = True
        for layer in base.layers[:-n_unfrozen]:
            layer.trainable = False
        for layer in base.layers[-n_unfrozen:]:
            layer.trainable = not isinstance(layer, layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=METRICS())
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=160)
    args = ap.parse_args()
    load_dotenv()
    p = load_params()
    set_seeds(p["seed"])
    s = ROOT / p["data"]["splits_dir"]
    train = make_dataset(s / "train.csv", args.img_size, 32, True, True, p["seed"])
    val = make_dataset(s / "val.csv", args.img_size, 32, False)

    tuner = kt.RandomSearch(lambda hp: build(hp, args.img_size), objective=kt.Objective("val_auc", "max"),
                            max_trials=args.trials, seed=p["seed"], directory=str(ROOT / "tuner_dir"),
                            project_name="visionguard", overwrite=True)
    tuner.search(train, validation_data=val, epochs=args.epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True)])

    rows = []
    for t in tuner.oracle.trials.values():
        if t.score is None:
            continue
        rows.append({"trial": t.trial_id, **t.hyperparameters.values, "val_auc": round(t.score, 4),
                     "best_epoch": t.best_step + 1 if t.best_step is not None else None})
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    out = ROOT / "reports" / "tuner_visionguard_results.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    # Which hyperparameter mattered most? Spread of mean val_auc across its values.
    spread = {h: float(df.groupby(h)["val_auc"].mean().max() - df.groupby(h)["val_auc"].mean().min())
              for h in ["lr", "dropout", "unfrozen_layers"]}
    print("\nmean val_auc spread per hyperparameter (bigger = mattered more):", spread)
    best = tuner.get_best_hyperparameters(1)[0].values
    print("BEST:", best)

    mlflow.set_experiment("VisionGuard-tuning")
    with mlflow.start_run(run_name=f"keras_tuner_random_{args.trials}trials"):
        mlflow.set_tags({"git_commit": git_commit(), "tool": "keras-tuner", "project": p["project"]})
        mlflow.log_params({"trials": args.trials, "epochs_per_trial": args.epochs, "img_size": args.img_size,
                           **{f"best_{k}": v for k, v in best.items()}})
        mlflow.log_metrics({"best_val_auc": float(df.val_auc.max()),
                            **{f"spread_{k}": v for k, v in spread.items()}})
        mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
