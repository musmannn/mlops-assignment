"""Task 3.5 - compare all VisionGuard runs in one table + chart.

    python -m training.compare_runs
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from training.common import ROOT, load_dotenv

COLS = {
    "tags.mlflow.runName": "run",
    "tags.note": "what changed",
    "params.model": "model",
    "params.backbone": "backbone",
    "params.img_size": "img",
    "params.augment": "aug",
    "params.lr": "lr",
    "params.unfreeze_layers": "unfreeze",
    "params.fine_tune": "fine_tune",
    "metrics.best_val_auc": "val_auc",
    "metrics.val_recall_at_thr": "val_recall@thr",
    "metrics.val_f1_at_thr": "val_f1@thr",
    "metrics.test_accuracy": "test_acc",
    "metrics.test_f1": "test_f1",
    "metrics.test_roc_auc": "test_auc",
    "metrics.test_real_called_fake": "test_FP",
    "metrics.best_epoch": "best_ep",
    "metrics.train_seconds": "train_s",
}


def load_runs(experiment: str = "VisionGuard"):
    df = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'")
    df = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    return df.sort_values("val_recall@thr", ascending=False)


def main() -> None:
    load_dotenv()
    df = load_runs()
    out = ROOT / "reports" / "run_comparison.csv"
    df.round(4).to_csv(out, index=False)
    print(df.round(3).to_string(index=False))

    plot = df.sort_values("val_auc")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(plot) + 1.5))
    y = range(len(plot))
    ax.barh([i - 0.2 for i in y], plot["val_auc"], height=0.38, label="val AUC (selection)")
    ax.barh([i + 0.2 for i in y], plot["test_auc"], height=0.38, label="test AUC (reported)")
    ax.set_yticks(list(y), plot["run"])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("ROC-AUC")
    ax.legend(loc="lower right")
    ax.set_title("VisionGuard runs - validation vs test AUC")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "run_comparison.png", dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
