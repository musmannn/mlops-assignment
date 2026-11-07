"""Helpers shared by the training scripts: params, seeds, provenance tags."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text())


def set_seeds(seed: int) -> None:
    """Fix every source of randomness we control (Task 2.4)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import keras

    keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TF together


def label_from_folder(folder_name: str) -> int | None:
    """The label of an image is the folder it sits in. 1 = AI-generated, 0 = real."""
    name = folder_name.lower()
    if "ai" in name or "fake" in name or "generated" in name:
        return 1
    if "real" in name:
        return 0
    return None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version() -> str:
    """md5 recorded in data/raw.dvc, i.e. the exact dataset version used."""
    dvc_file = ROOT / "data" / "raw.dvc"
    if not dvc_file.exists():
        return "untracked"
    meta = yaml.safe_load(dvc_file.read_text())
    return meta["outs"][0].get("md5", "unknown")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_dotenv() -> None:
    """Load .env (DagsHub URI/token) into the environment; never hard-code secrets."""
    try:
        from dotenv import load_dotenv as _ld

        _ld(ROOT / ".env")
    except ImportError:
        pass
