"""Evaluation helpers: threshold selection, test metrics and plots (Task 2.4 / 2.5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray, max_fpr_real: float = 0.10) -> dict:
    """Pick the decision threshold on VALIDATION data only.

    Product decision: calling a real photo fake (false positive, class 1 = AI)
    is the worse error - it accuses a genuine photographer/journalist of
    faking. So we pick the LOWEST threshold whose false-positive rate on real
    validation photos is <= max_fpr_real; among those it catches the most AI
    images. If no threshold meets the limit we fall back to Youden's J.
    """
    fpr, tpr, thr = roc_curve(y_val, p_val)
    thr = np.clip(thr, 0, 1)
    ok = np.where(fpr <= max_fpr_real)[0]
    if len(ok):
        i = ok[np.argmax(tpr[ok])]
        rule = f"max recall s.t. FPR<={max_fpr_real}"
    else:
        i = int(np.argmax(tpr - fpr))
        rule = "youden_j"
    j = int(np.argmax(tpr - fpr))
    return {
        "threshold": float(thr[i]),
        "rule": rule,
        "val_fpr": float(fpr[i]),
        "val_tpr": float(tpr[i]),
        "youden_threshold": float(thr[j]),
    }


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "real_called_fake": int(fp),
        "fake_called_real": int(fn),
        "tp": int(tp), "tn": int(tn),
    }


def plot_confusion(y, p, threshold, path: Path, title: str) -> None:
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["real", "ai"])
    ax.set_yticks([0, 1], ["real", "ai"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nthreshold={threshold:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple], path: Path, title: str = "ROC curve (test set)") -> None:
    """curves: {name: (y, p, threshold)}"""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p, thr) in curves.items():
        fpr, tpr, t = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        k = int(np.argmin(np.abs(t - thr)))
        ax.scatter(fpr[k], tpr[k], s=40, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(history: dict, path: Path, title: str, stage_boundary: int | None = None) -> int:
    """Loss and accuracy curves with best (min val_loss) epoch marked. Returns best epoch (1-based)."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["loss", "accuracy"], strict=True):
        ax.plot(epochs, history[key], label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], label=f"val {key}")
        ax.axvline(best, color="red", ls="--", lw=1, label=f"best epoch {best}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="grey", ls=":", lw=1, label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_title(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return best
