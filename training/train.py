"""Train one VisionGuard model and log everything to MLflow (local or DagsHub).

Examples
    python -m training.train --model cnn --tracking manual --run-name cnn_baseline
    python -m training.train --model cnn --tracking autolog --run-name cnn_autolog
    python -m training.train --model transfer --backbone mobilenetv2 --run-name mnv2_224

Where the run goes is decided ONLY by environment variables (from .env):
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
If MLFLOW_TRACKING_URI is unset, runs go to the local ./mlruns folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import keras
import mlflow
import numpy as np
import pandas as pd

from training import metrics as M
from training.common import ROOT, dvc_data_version, git_commit, hardware, load_dotenv, load_params, set_seeds
from training.models import (
    build_baseline_cnn,
    build_transfer,
    compile_model,
    layer_table,
    trainable_layer_names,
    unfreeze,
)
from training.pipeline import labels_of, make_dataset


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual tracking: log every Keras metric + learning rate once per epoch.

    `offset` lets stage 2 (fine-tuning) continue the step axis where stage 1 stopped,
    so both stages appear as one continuous curve in MLflow.
    """

    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        named = {(k if k.startswith("val_") or k == "learning_rate" else f"train_{k}"): float(v)
                 for k, v in logs.items()}
        mlflow.log_metrics(named, step=self.offset + epoch + 1)


def callbacks(ckpt: Path, patience: int, manual: bool, offset: int) -> list:
    cbs = [
        # Stop when val_loss has not improved for `patience` epochs and roll back to the best weights.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        # Keep the best-on-validation model on disk in case training crashes or overfits later.
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0),
        # When val_loss plateaus, cut the learning rate by 3x to settle into the minimum.
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(2, patience // 2),
                                          min_lr=1e-7, verbose=1),
    ]
    if manual:
        cbs.append(MlflowEpochLogger(offset))
    return cbs


def merge(h1: dict, h2: dict | None) -> dict:
    if not h2:
        return h1
    return {k: list(h1[k]) + list(h2.get(k, [])) for k in h1 if not k.startswith("learning_rate")}


def parse_args() -> argparse.Namespace:
    p = load_params()
    t = p["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default=t["backbone"])
    ap.add_argument("--img-size", type=int, default=t["img_size"])
    ap.add_argument("--batch-size", type=int, default=t["batch_size"])
    ap.add_argument("--epochs", type=int, default=t["epochs"])
    ap.add_argument("--lr", type=float, default=t["lr"])
    ap.add_argument("--fine-tune-lr", type=float, default=t["fine_tune_lr"])
    ap.add_argument("--fine-tune-epochs", type=int, default=t["fine_tune_epochs"])
    ap.add_argument("--no-fine-tune", action="store_true", help="transfer: skip stage 2")
    ap.add_argument("--unfreeze-layers", type=int, default=t["unfreeze_layers"])
    ap.add_argument("--dropout", type=float, default=t["dropout"])
    ap.add_argument("--patience", type=int, default=t["patience"])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--experiment", default="VisionGuard")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--note", default="", help="what changed vs the previous run (logged as a tag)")
    ap.add_argument("--seed", type=int, default=p["seed"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params()
    load_dotenv()
    set_seeds(args.seed)
    augment = not args.no_augment
    splits = ROOT / params["data"]["splits_dir"]
    arch = args.backbone if args.model == "transfer" else "scratch"
    run_name = args.run_name or f"{args.model}_{arch}_{args.img_size}"
    work = Path(tempfile.mkdtemp(prefix="vg_"))
    fig_dir = ROOT / "reports" / "figures" / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(splits / "train.csv", args.img_size, args.batch_size, True, augment, args.seed)
    val_ds = make_dataset(splits / "val.csv", args.img_size, args.batch_size, False)
    test_ds = make_dataset(splits / "test.csv", args.img_size, args.batch_size, False)
    y_val, y_test = labels_of(splits / "val.csv"), labels_of(splits / "test.csv")

    mlflow.set_experiment(args.experiment)
    manual = args.tracking == "manual"
    if not manual:
        # Autologging: MLflow patches model.fit and records params/metrics itself.
        # log_models=False because we log the final model ourselves (with metadata) below.
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "project": params["project"],
            "git_commit": git_commit(),
            "dvc_data_version": dvc_data_version(),
            "seed": args.seed,
            "model_type": args.model,
            "tracking_mode": args.tracking,
            "note": args.note,
            "hardware": hardware(),
        })
        hp = {
            "model": args.model, "backbone": args.backbone if args.model == "transfer" else "none",
            "img_size": args.img_size, "batch_size": args.batch_size, "augment": augment,
            "dropout": args.dropout, "seed": args.seed, "patience": args.patience,
            "train_size": len(labels_of(splits / "train.csv")),
            "val_size": len(y_val), "test_size": len(y_test),
        }
        if manual:  # autolog records lr/epochs/optimizer itself; manual mode must do it explicitly
            hp |= {"lr": args.lr, "epochs": args.epochs, "optimizer": "adam", "loss": "binary_crossentropy"}
        if args.model == "transfer":
            hp |= {"fine_tune": not args.no_fine_tune, "fine_tune_lr": args.fine_tune_lr,
                   "fine_tune_epochs": args.fine_tune_epochs, "unfreeze_layers": args.unfreeze_layers}
        mlflow.log_params(hp)

        # ---------------- build + stage 1 ----------------
        base = None
        if args.model == "cnn":
            model = build_baseline_cnn(args.img_size, args.dropout)
        else:
            model, base = build_transfer(args.backbone, args.img_size, args.dropout)
        compile_model(model, args.lr)

        summary_lines: list[str] = []
        model.summary(print_fn=lambda s, **_: summary_lines.append(s))
        (work / "model_summary.txt").write_text("\n".join(summary_lines))
        pd.DataFrame(layer_table(model)).to_csv(work / "layer_table.csv", index=False)
        stage_info = {"stage1_trainable": trainable_layer_names(model)}

        t0 = time.time()
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
                       callbacks=callbacks(work / "best_stage1.keras", args.patience, manual, 0)).history
        stage1_epochs = len(h1["loss"])
        history, boundary = h1, None

        # ---------------- stage 2: fine-tune last block ----------------
        if args.model == "transfer" and not args.no_fine_tune:
            if not manual:
                mlflow.tensorflow.autolog(disable=True)  # a 2nd autologged fit would re-log params -> conflict
            stage_info["stage2_unfrozen_base_layers"] = unfreeze(base, args.backbone, args.unfreeze_layers)
            compile_model(model, args.fine_tune_lr)  # recompile so the new trainable flags take effect
            stage_info["stage2_trainable"] = trainable_layer_names(model)
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, verbose=2,
                           callbacks=callbacks(work / "best_stage2.keras", args.patience, True, stage1_epochs)).history
            history, boundary = merge(h1, h2), stage1_epochs
        train_seconds = time.time() - t0
        (work / "trainable_layers.json").write_text(json.dumps(stage_info, indent=2))

        best_epoch = M.plot_curves(history, fig_dir / "curves.png", f"{run_name}: loss & accuracy", boundary)

        # ---------------- threshold on VAL, evaluate on TEST once ----------------
        p_val = model.predict(val_ds, verbose=0).ravel()
        thr = M.choose_threshold(y_val, p_val, params["eval"]["max_fpr_real"])
        p_test = model.predict(test_ds, verbose=0).ravel()
        val_m = M.compute_metrics(y_val, p_val, thr["threshold"])
        test_m = M.compute_metrics(y_test, p_test, thr["threshold"])
        test_m05 = M.compute_metrics(y_test, p_test, 0.5)

        mlflow.log_params({"threshold": round(thr["threshold"], 4), "threshold_rule": thr["rule"]})
        mlflow.log_metrics({f"val_{k}_at_thr": v for k, v in val_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metrics({f"test_{k}_at_0.5": test_m05[k] for k in ("accuracy", "f1", "real_called_fake")})
        mlflow.log_metrics({"best_epoch": best_epoch, "epochs_run": len(history["loss"]),
                            "best_val_loss": float(np.min(history["val_loss"])),
                            "best_val_auc": float(np.max(history["val_auc"])),
                            "train_seconds": round(train_seconds, 1)})

        M.plot_confusion(y_test, p_test, thr["threshold"], fig_dir / "confusion_matrix.png", f"{run_name} - test")
        M.plot_roc({run_name: (y_test, p_test, thr["threshold"])}, fig_dir / "roc_curve.png")
        test_df = pd.read_csv(splits / "test.csv").assign(prob_ai=p_test, pred=(p_test >= thr["threshold"]).astype(int))
        test_df.to_csv(fig_dir / "test_predictions.csv", index=False)

        # ---------------- save model + metadata, log artifacts ----------------
        model_file = work / "model.keras"
        model.save(model_file)
        meta = {"model_name": params["registry"]["model_name"], "model_version": run.info.run_id[:8],
                "run_id": run.info.run_id, "run_name": run_name, "img_size": args.img_size,
                "threshold": thr["threshold"], "threshold_rule": thr["rule"], "labels": {"0": "real", "1": "ai"},
                "test_metrics": {k: round(float(v), 4) for k, v in test_m.items()}}
        model_file.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        for f in ["curves.png", "confusion_matrix.png", "roc_curve.png", "test_predictions.csv"]:
            mlflow.log_artifact(str(fig_dir / f), "evaluation")
        for f in ["train.csv", "val.csv", "test.csv"]:
            mlflow.log_artifact(str(splits / f), "splits")
        mlflow.log_artifact(str(ROOT / "requirements-lock.txt"), "environment")
        for f in ["model_summary.txt", "layer_table.csv", "trainable_layers.json"]:
            mlflow.log_artifact(str(work / f), "model_info")
        mlflow.log_artifact(str(model_file), "keras_file")
        mlflow.log_artifact(str(model_file.with_suffix(".json")), "keras_file")
        # MLflow model flavour -> lets the registry load it later with models:/<name>@champion
        info = mlflow.tensorflow.log_model(model, name="model",
                                           input_example=np.zeros((1, args.img_size, args.img_size, 3), np.float32))
        mlflow.set_tag("model_uri", info.model_uri)  # used by training.register

        # local copy for serving / error analysis
        out = ROOT / "models" / "runs" / run_name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_file, out / "model.keras")
        shutil.copy(model_file.with_suffix(".json"), out / "model.json")
        shutil.copy(work / "layer_table.csv", fig_dir / "layer_table.csv")
        shutil.copy(work / "trainable_layers.json", fig_dir / "trainable_layers.json")
        shutil.copy(work / "model_summary.txt", fig_dir / "model_summary.txt")

        print(f"\nrun {run_name} ({run.info.run_id}) best_epoch={best_epoch} threshold={thr['threshold']:.3f}")
        print("TEST:", {k: round(float(v), 4) for k, v in test_m.items()})
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
