"""Task 2.6 - grid of misclassified test images for a run.

    python -m training.error_analysis --run mnv2_224
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from training.common import ROOT
from visionguard.preprocess import letterbox, open_image, to_rgb

NAMES = {0: "real", 1: "ai"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    d = ROOT / "reports" / "figures" / args.run
    df = pd.read_csv(d / "test_predictions.csv")
    wrong = df[df.pred != df.label].copy()
    wrong["wrongness"] = (wrong.prob_ai - wrong.label).abs()  # how confidently wrong
    wrong = wrong.sort_values("wrongness", ascending=False).head(args.n).reset_index(drop=True)
    audit = pd.read_csv(ROOT / "reports" / "audit.csv").set_index("path")

    cols = 5
    rows = -(-len(wrong) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.9 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for i, r in wrong.iterrows():
        a = audit.loc[r.path]
        ax = axes.ravel()[i]
        ax.imshow(letterbox(to_rgb(open_image(ROOT / r.path)), 256))
        ax.set_title(f"#{i + 1} true={NAMES[r.label]} pred={NAMES[r.pred]}\nP(AI)={r.prob_ai:.2f} "
                     f"{a.width}x{a.height} {a.format}", fontsize=9)
    fig.suptitle(f"{args.run}: {len(df[df.pred != df.label])} test errors, {len(wrong)} most confident shown")
    fig.tight_layout()
    fig.savefig(d / "errors.png", dpi=110)
    wrong[["path", "label", "pred", "prob_ai"]].to_csv(d / "errors.csv", index=False)
    print(wrong[["path", "label", "pred", "prob_ai"]].to_string())
    print("wrote", d / "errors.png")
    Image.MAX_IMAGE_PIXELS = None


if __name__ == "__main__":
    main()
