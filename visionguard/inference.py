"""Inference module (Task 4.1).

`Predictor` loads the champion Keras model ONCE (at app start-up) and keeps it in
memory. Loading a TensorFlow model takes seconds and hundreds of MB of RAM;
doing that per request would make every upload slow and could exhaust memory
under concurrent requests.

The model's metadata (image size, decision threshold, version) is read from a
JSON file saved next to the model by training, so serving uses the same settings
as training/evaluation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visionguard.preprocess import load_rgb_letterboxed

LABELS = {0: "real", 1: "ai"}
DISPLAY = {"real": "Real photo", "ai": "AI-generated"}


@dataclass
class Prediction:
    label: str          # "ai" or "real"
    confidence: float   # probability of the predicted label, 0..1
    prob_ai: float      # raw sigmoid output
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_meta(model_path: str | Path) -> dict:
    meta_path = Path(model_path).with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


class _BasePredictor:
    """Shared logic: metadata, preprocessing, thresholding. Subclasses implement `_prob_ai`."""

    def __init__(self, model_path: str | Path, threshold: float | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        meta = load_meta(self.model_path)
        self.img_size = int(meta.get("img_size", 224))
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.model_name = meta.get("model_name", self.model_path.stem)
        self.model_version = str(meta.get("model_version", "local"))
        self._load()
        # Warm-up: the first call builds/optimises the graph; do it now, not on the first user.
        self._prob_ai(np.zeros((1, self.img_size, self.img_size, 3), np.float32))

    def _load(self) -> None:
        raise NotImplementedError

    def _prob_ai(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def predict_bytes(self, data: bytes) -> Prediction:
        t0 = time.perf_counter()
        x = load_rgb_letterboxed(data, self.img_size)[None, ...]
        prob_ai = self._prob_ai(x)
        label = "ai" if prob_ai >= self.threshold else "real"
        confidence = prob_ai if label == "ai" else 1.0 - prob_ai
        return Prediction(
            label=label,
            confidence=round(confidence, 4),
            prob_ai=round(prob_ai, 4),
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


class Predictor(_BasePredictor):
    """Keras/TensorFlow backend (.keras file) - used by the Docker web app."""

    def _load(self) -> None:
        import keras  # imported lazily so the web app can run without TF in "remote" mode

        self.model = keras.models.load_model(self.model_path, compile=False)

    def _prob_ai(self, x: np.ndarray) -> float:
        return float(self.model.predict(x, verbose=0)[0][0])


class OnnxPredictor(_BasePredictor):
    """onnxruntime backend (.onnx file) - used by the serverless model API on Vercel.

    Same preprocessing and threshold as the Keras model; numerical equivalence is checked by
    training/export_onnx.py (max |diff| ~1e-5, no label flips on the test set).
    """

    def _load(self) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _prob_ai(self, x: np.ndarray) -> float:
        return float(self.session.run(None, {self.input_name: x.astype(np.float32)})[0].ravel()[0])


def predictor_from_env() -> _BasePredictor:
    threshold = os.environ.get("THRESHOLD")
    path = os.environ.get("MODEL_PATH", "models/champion.keras")
    cls = OnnxPredictor if path.endswith(".onnx") else Predictor
    return cls(path, float(threshold) if threshold else None)


if __name__ == "__main__":  # console demo for the report: python -m visionguard.inference img1 img2
    import sys

    p = predictor_from_env()
    print(f"backend: {type(p).__name__}")
    print(f"loaded {p.model_name} v{p.model_version} size={p.img_size} threshold={p.threshold}")
    for path in sys.argv[1:]:
        r = p.predict_bytes(Path(path).read_bytes())
        print(f"{path}: {DISPLAY[r.label]}  confidence={r.confidence:.3f}  p(ai)={r.prob_ai:.3f}  {r.latency_ms} ms")
