"""Task 1.2 - dataset audit.

Walks data/raw, and for every file records class, file size, width, height,
colour mode, format and SHA-256. Flags unreadable files and exact duplicates
(same hash), then prints the summary tables used in the report.

    python -m training.audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from training.common import ROOT, label_from_folder, load_params, sha256_file

Image.MAX_IMAGE_PIXELS = 200_000_000
SKIP = {".ds_store", "thumbs.db"}


def audit(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        if path.name.lower() in SKIP or path.suffix.lower() == ".dvc":
            continue
        # Label = the nearest ancestor folder whose name identifies a class.
        label = None
        for parent in path.relative_to(raw_dir).parents:
            if parent.name and (label := label_from_folder(parent.name)) is not None:
                break
        row = {
            "path": str(path.relative_to(ROOT)),
            "class": {1: "ai", 0: "real"}.get(label, "unknown"),
            "label": label,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "ext": path.suffix.lower().lstrip("."),
            "sha256": sha256_file(path),
            "width": None, "height": None, "mode": None, "format": None,
            "readable": False, "error": "",
        }
        try:
            with Image.open(path) as im:
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                im.load()  # full decode, catches truncated files
            row["readable"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["duplicate"] = df.duplicated("sha256", keep="first")
    df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    return df


def summarize(df: pd.DataFrame) -> str:
    out = []
    out.append("== Counts per class ==")
    counts = df.groupby("class").agg(
        files=("path", "size"),
        readable=("readable", "sum"),
        corrupt=("readable", lambda s: int((~s).sum())),
        exact_duplicates=("duplicate", "sum"),
    )
    out.append(counts.to_string())
    ok = df[df.readable]
    out.append("\n== Width / height / file size (readable files) ==")
    stats = ok.groupby("class")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
    out.append(stats.to_string())
    out.append("\n== Colour modes ==")
    out.append(pd.crosstab(ok["class"], ok["mode"]).to_string())
    out.append("\n== Formats ==")
    out.append(pd.crosstab(ok["class"], ok["format"]).to_string())
    cross = df[df.duplicate].merge(df[~df.duplicate][["sha256", "class"]], on="sha256", suffixes=("", "_first"))
    out.append(f"\nCorrupt/unreadable files: {int((~df.readable).sum())}")
    out.append(f"Exact duplicate files (extra copies): {int(df.duplicate.sum())}")
    out.append(f"Duplicates whose copies sit in BOTH classes: {int((cross['class'] != cross['class_first']).sum())}")
    if (~df.readable).any():
        out.append("\nUnreadable files:\n" + df[~df.readable][["path", "error"]].to_string(index=False))
    return "\n".join(out)


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=params["data"]["raw_dir"])
    ap.add_argument("--out", default=params["data"]["audit_csv"])
    args = ap.parse_args()

    df = audit(ROOT / args.raw_dir)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    text = summarize(df)
    (out.parent / "audit_summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
