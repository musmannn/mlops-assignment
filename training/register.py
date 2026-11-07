"""Task 3.6 - register the best runs in the MLflow Model Registry and export the champion.

Selection rule (validation data only, test set is never used to choose):
  1. highest recall on AI images at the chosen threshold (FPR on real <= 10%),
  2. tie-break: highest validation AUC.

Registers the top-N runs as successive versions, gives the best the alias
`champion` (and the runner-up `challenger`), then downloads the champion's
.keras file to models/champion.keras for serving. The serving app reads that
local file; it never calls DagsHub per request.

    python -m training.register            # automatic
    python -m training.register --runs <run_id_a> <run_id_b>   # explicit, last = champion
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient

from training.common import ROOT, load_dotenv, load_params


def main() -> None:
    load_dotenv()
    params = load_params()
    name = params["registry"]["model_name"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="VisionGuard")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--runs", nargs="*", help="run ids to register in order; the LAST becomes champion")
    args = ap.parse_args()
    client = MlflowClient()

    if args.runs:
        run_ids = args.runs
    else:
        df = mlflow.search_runs(experiment_names=[args.experiment],
                                filter_string="attributes.status = 'FINISHED'",
                                order_by=["metrics.val_recall_at_thr DESC", "metrics.best_val_auc DESC"])
        df = df[df["tags.model_uri"].notna()]
        # register runner-up first, champion last -> champion gets the newest version number
        run_ids = list(df["run_id"].head(args.top))[::-1]

    versions = []
    for rid in run_ids:
        run = client.get_run(rid)
        uri = run.data.tags.get("model_uri", f"runs:/{rid}/model")
        mv = mlflow.register_model(uri, name)
        for k in ("threshold", "img_size", "backbone", "model"):
            client.set_model_version_tag(name, mv.version, k, run.data.params.get(k, ""))
        client.set_model_version_tag(name, mv.version, "run_name", run.data.tags.get("mlflow.runName", ""))
        client.set_model_version_tag(name, mv.version, "test_roc_auc", f"{run.data.metrics.get('test_roc_auc', 0):.4f}")
        versions.append((mv.version, rid, run.data.tags.get("mlflow.runName")))
        print(f"registered {name} v{mv.version} <- run {rid} ({run.data.tags.get('mlflow.runName')})")

    champ_v, champ_run, champ_name = versions[-1]
    client.set_registered_model_alias(name, "champion", champ_v)
    if len(versions) > 1:
        client.set_registered_model_alias(name, "challenger", versions[-2][0])
    desc = "VisionGuard: P(image is AI-generated). Input 0-255 RGB, letterboxed."
    client.update_registered_model(name, description=desc)
    print(f"alias champion -> v{champ_v} ({champ_name})")

    # Export the champion's .keras file + metadata for the web app / Docker / HF Space.
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(run_id=champ_run, artifact_path="keras_file", dst_path=tmp))
        out = ROOT / "models"
        shutil.copy(local / "model.keras", out / "champion.keras")
        meta = json.loads((local / "model.json").read_text())
        meta.update(model_version=str(champ_v), registry_alias="champion", run_id=champ_run)
        (out / "champion.json").write_text(json.dumps(meta, indent=2))
    print(f"exported models/champion.keras (+ champion.json, threshold={meta['threshold']:.3f})")


if __name__ == "__main__":
    main()
