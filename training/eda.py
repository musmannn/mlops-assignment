"""Task 1.3 - visual EDA: sample grid per class + histograms of width, height, file size.

    python -m training.eda
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import ROOT, load_params
from visionguard.preprocess import letterbox, open_image, to_rgb

FIG = ROOT / "reports" / "figures"


def sample_grid(df: pd.DataFrame, n: int = 8, seed: int = 42) -> None:
    classes = ["ai", "real"]
    fig, axes = plt.subplots(len(classes), n, figsize=(2.2 * n, 2.6 * len(classes)))
    for r, cls in enumerate(classes):
        sub = df[(df["class"] == cls)].sample(n, random_state=seed)
        for c, (_, row) in enumerate(sub.iterrows()):
            ax = axes[r, c]
            ax.imshow(letterbox(to_rgb(open_image(ROOT / row.path)), 224))
            ax.set_title(f"{row.width}x{row.height}", fontsize=8)
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, cls.upper(), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=13, weight="bold")
    fig.suptitle("Random samples per class (letterboxed to 224x224, original size above)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_samples.png", dpi=120)
    plt.close(fig)


def histograms(df: pd.DataFrame) -> None:
    cols = [("width", "Width (px)"), ("height", "Height (px)"), ("file_size_kb", "File size (KB)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (col, title) in zip(axes, cols, strict=True):
        upper = df[col].quantile(0.99)  # clip the long tail so the bulk is visible
        bins = np.linspace(0, upper, 40)
        for cls, color in [("ai", "#d1495b"), ("real", "#2e86ab")]:
            col_vals = df[df["class"] == cls][col]
            ax.hist(col_vals.clip(upper=upper), bins=bins, alpha=0.55, color=color,
                    label=f"{cls} (median {col_vals.median():.0f})")
        ax.set_title(title + " - clipped at 99th percentile")
        ax.set_xlabel(title)
        ax.set_ylabel("Images")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_histograms.png", dpi=120)
    plt.close(fig)


def aspect_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color in [("ai", "#d1495b"), ("real", "#2e86ab")]:
        s = df[df["class"] == cls]
        ax.hist((s.width / s.height).clip(0.3, 3), bins=40, alpha=0.55, label=cls, color=color)
    ax.set_xlabel("Aspect ratio (width / height)")
    ax.set_ylabel("Images")
    ax.legend()
    ax.set_title("Aspect ratio per class")
    fig.tight_layout()
    fig.savefig(FIG / "eda_aspect.png", dpi=120)
    plt.close(fig)


def main() -> None:
    params = load_params()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / params["data"]["audit_csv"])
    df = df[df.readable & ~df.duplicate & (df["class"] != "unknown")]
    sample_grid(df)
    histograms(df)
    aspect_plot(df)
    print("wrote", *(p.name for p in FIG.glob("eda_*.png")))


if __name__ == "__main__":
    main()
