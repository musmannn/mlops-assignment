"""Report figures that span runs.

    python -m training.figures batch                         # Task 2.1: one augmented training batch
    python -m training.figures compare cnn_baseline mnv2_224 # Task 2.5: CNN vs transfer on the test set
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from training import metrics as M
from training.common import ROOT, load_params

FIG = ROOT / "reports" / "figures"


def augmented_batch() -> None:
    from training.pipeline import make_dataset

    p = load_params()
    ds = make_dataset(ROOT / p["data"]["splits_dir"] / "train.csv", 224, 16, True, True, p["seed"])
    x, y = next(iter(ds))
    fig, axes = plt.subplots(2, 8, figsize=(17, 4.8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x[i].numpy().clip(0, 255).astype("uint8"))
        ax.set_title("ai" if y[i] == 1 else "real", fontsize=10)
        ax.axis("off")
    fig.suptitle("One training batch after letterbox + random flip / rotation / zoom / contrast")
    fig.tight_layout()
    fig.savefig(FIG / "augmented_batch.png", dpi=110)
    print("wrote augmented_batch.png")


def compare(runs: list[str]) -> None:
    out = FIG / "final_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for run in runs:
        df = pd.read_csv(FIG / run / "test_predictions.csv")
        thr = json.loads((ROOT / "models" / "runs" / run / "model.json").read_text())["threshold"]
        m = M.compute_metrics(df.label.to_numpy(), df.prob_ai.to_numpy(), thr)
        rows.append({"model": run, "threshold": round(thr, 3), **{k: round(float(v), 4) for k, v in m.items()}})
        curves[run] = (df.label.to_numpy(), df.prob_ai.to_numpy(), thr)
        M.plot_confusion(df.label.to_numpy(), df.prob_ai.to_numpy(), thr, out / f"cm_{run}.png", f"{run} (test)")
    M.plot_roc(curves, out / "roc_test.png")
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_metrics.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "batch":
        augmented_batch()
    else:
        compare(sys.argv[2:])
