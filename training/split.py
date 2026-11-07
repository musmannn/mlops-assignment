"""Create stratified train/val/test split lists from the audit table.

Corrupt files and exact duplicates are dropped BEFORE splitting, so the same
image can never be in both train and test (that would leak and inflate scores).

    python -m training.split
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from training.common import ROOT, load_params


def main() -> None:
    p = load_params()
    seed, d = p["seed"], p["data"]
    df = pd.read_csv(ROOT / d["audit_csv"])
    before = len(df)
    # Drop every copy of a hash that appears in both classes (label is ambiguous),
    # then keep the first copy of the remaining duplicates.
    conflict = df.groupby("sha256")["label"].transform("nunique") > 1
    df = df[df.readable & (df["class"] != "unknown") & ~conflict].drop_duplicates("sha256")
    df = df[["path", "label"]].astype({"label": int})
    print(f"kept {len(df)} of {before} files after removing corrupt/duplicate/conflicting ones")

    train, temp = train_test_split(df, test_size=d["val_frac"] + d["test_frac"],
                                   stratify=df.label, random_state=seed)
    rel_test = d["test_frac"] / (d["val_frac"] + d["test_frac"])
    val, test = train_test_split(temp, test_size=rel_test, stratify=temp.label, random_state=seed)

    out = ROOT / d["splits_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path").to_csv(out / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):5d} images  ai={int(part.label.sum())}  real={int((part.label == 0).sum())}")


if __name__ == "__main__":
    main()
