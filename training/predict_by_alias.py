"""Task 3.6 - separate script: load the model by registry alias and classify one image.

    python -m training.predict_by_alias path/to/image.jpg [--alias champion]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient

from training.common import load_dotenv, load_params
from visionguard.preprocess import load_rgb_letterboxed


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()
    name = load_params()["registry"]["model_name"]

    mv = MlflowClient().get_model_version_by_alias(name, args.alias)
    uri = f"models:/{name}@{args.alias}"
    print(f"tracking: {mlflow.get_tracking_uri()}")
    print(f"loading {uri} -> version {mv.version} (run {mv.run_id})")
    model = mlflow.tensorflow.load_model(uri)

    size = int(mv.tags.get("img_size") or model.input_shape[1])
    threshold = float(mv.tags.get("threshold") or 0.5)
    x = load_rgb_letterboxed(args.image, size)[None, ...]
    p = float(model.predict(x, verbose=0)[0][0])
    label = "AI-generated" if p >= threshold else "Real"
    print(f"{args.image}: {label}  P(AI)={p:.4f}  threshold={threshold:.3f}  "
          f"confidence={(p if p >= threshold else 1 - p):.3f}")


if __name__ == "__main__":
    main()
