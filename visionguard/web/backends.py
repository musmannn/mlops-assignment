"""Inference backends for the web app.

local  - loads the champion .keras model once inside the Flask process (Tasks 4/6).
remote - forwards the image to the model API hosted in the cloud (Tasks 7/8);
         Flask itself can then run on localhost without TensorFlow.
"""

from __future__ import annotations

import time

import requests


class BackendError(RuntimeError):
    """The model could not produce a prediction (service down, timeout, bad response)."""


class LocalBackend:
    name = "local"

    def __init__(self, model_path: str, threshold: float | None = None):
        from visionguard.inference import Predictor  # imports TensorFlow

        self.predictor = Predictor(model_path, threshold)

    def predict(self, data: bytes, filename: str) -> dict:
        return self.predictor.predict_bytes(data).to_dict()

    def info(self) -> dict:
        p = self.predictor
        return {"backend": self.name, "model_loaded": True, "model_name": p.model_name,
                "model_version": p.model_version, "threshold": p.threshold, "img_size": p.img_size}


class RemoteBackend:
    name = "remote"

    def __init__(self, url: str, api_key: str, timeout: float = 60):
        if not url:
            raise ValueError("MODEL_API_URL is required when INFERENCE_BACKEND=remote")
        self.url = url.rstrip("/")
        self.session = requests.Session()  # reuse the TLS connection between requests
        self.session.headers["X-API-Key"] = api_key
        self.timeout = timeout

    # Vercel rejects request bodies over 4.5 MB; stay safely below with multipart overhead.
    MAX_SEND_BYTES = 4_000_000

    @staticmethod
    def shrink(data: bytes, filename: str, long_side: int = 2048) -> tuple[bytes, str]:
        """Downscale + re-encode an oversized upload. The model only sees 224x224, so this
        loses nothing it uses; JPEG quality 92 keeps compression artefacts minimal."""
        import io

        from PIL import Image

        from visionguard.preprocess import to_rgb

        img = to_rgb(Image.open(io.BytesIO(data)))
        img.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return buf.getvalue(), filename.rsplit(".", 1)[0] + ".jpg"

    def predict(self, data: bytes, filename: str) -> dict:
        t0 = time.perf_counter()
        if len(data) > self.MAX_SEND_BYTES:
            data, filename = self.shrink(data, filename)
        try:
            r = self.session.post(f"{self.url}/predict", files={"image": (filename, data)}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise BackendError(f"model API unreachable: {exc}") from exc
        if r.status_code != 200:
            try:
                msg = r.json().get("error", r.text[:200])
            except ValueError:
                msg = r.text[:200]
            raise BackendError(f"model API returned {r.status_code}: {msg}")
        out = r.json()
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)  # includes network time
        return out

    def info(self) -> dict:
        try:
            h = self.session.get(f"{self.url}/health", timeout=10).json()
        except Exception as exc:
            return {"backend": self.name, "model_loaded": False, "api_url": self.url, "error": str(exc)}
        return {"backend": self.name, "api_url": self.url, **h}


class FakeBackend:
    """Deterministic stand-in used by the test-suite and CI (no TensorFlow needed)."""

    name = "fake"

    def __init__(self, prob_ai: float = 0.9, threshold: float = 0.5):
        self.prob_ai, self.threshold = prob_ai, threshold

    def predict(self, data: bytes, filename: str) -> dict:
        label = "ai" if self.prob_ai >= self.threshold else "real"
        conf = self.prob_ai if label == "ai" else 1 - self.prob_ai
        return {"label": label, "confidence": conf, "prob_ai": self.prob_ai, "threshold": self.threshold,
                "model_name": "fake", "model_version": "0", "latency_ms": 1.0}

    def info(self) -> dict:
        return {"backend": self.name, "model_loaded": True, "model_name": "fake", "model_version": "0"}
