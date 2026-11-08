"""Export the champion Keras model to ONNX for serverless serving (Task 7, Vercel).

Why: a Vercel Python function may be at most 500 MB unzipped, and TensorFlow alone is ~1.3 GB.
onnxruntime (~50 MB) runs the same graph. Training stays in Keras; only the serving format changes.

The script proves equivalence on the whole test set: the max |P_keras - P_onnx| and whether the
label at the champion threshold ever differs. The result is written to reports/onnx_equivalence.json.

    python -m training.export_onnx
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import onnxruntime as ort
import pandas as pd

from training.common import ROOT, load_params
from visionguard.preprocess import load_rgb_letterboxed


def main() -> None:
    src = ROOT / "models" / "champion.keras"
    dst = ROOT / "models" / "champion.onnx"
    meta = json.loads((ROOT / "models" / "champion.json").read_text())
    size, thr = meta["img_size"], meta["threshold"]

    model = keras.models.load_model(src, compile=False)
    t0 = time.time()
    model.export(str(dst), format="onnx",
                 input_signature=[keras.InputSpec(shape=(None, size, size, 3), dtype="float32")])
    print(f"exported {dst.name} ({dst.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    test = pd.read_csv(ROOT / load_params()["data"]["splits_dir"] / "test.csv")
    x = np.stack([load_rgb_letterboxed(ROOT / p, size) for p in test.path])
    p_keras = model.predict(x, batch_size=32, verbose=0).ravel()
    p_onnx = np.concatenate([sess.run(None, {in_name: x[i:i + 32]})[0].ravel() for i in range(0, len(x), 32)])

    diff = np.abs(p_keras - p_onnx)
    flips = int(((p_keras >= thr) != (p_onnx >= thr)).sum())
    report = {
        "images": len(test), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "label_flips_at_threshold": flips, "threshold": thr,
        "keras_file_mb": round(src.stat().st_size / 1e6, 1), "onnx_file_mb": round(dst.stat().st_size / 1e6, 1),
        "onnxruntime": ort.__version__,
    }
    (ROOT / "reports" / "onnx_equivalence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert diff.max() < 1e-3 and flips == 0, "ONNX model does not match Keras model"
    print("OK: ONNX and Keras agree on every test image")


if __name__ == "__main__":
    main()
