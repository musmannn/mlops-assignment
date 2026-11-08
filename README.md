# VisionGuard · 24L-0672_A2_Ops

[![CI](https://github.com/musmannn/mlops-assignment/actions/workflows/ci.yml/badge.svg)](https://github.com/musmannn/mlops-assignment/actions/workflows/ci.yml)

A web service that says whether an uploaded image is **AI-generated or real**:
a TensorFlow/Keras model (MobileNetV2 fine-tuned), tracked with MLflow on DagsHub, data versioned with DVC,
served by Flask with a PostgreSQL history, containerised with Docker Compose. The model API (ONNX export of the
champion, served by onnxruntime) is deployed on Vercel and called by the local Flask app.

| | |
|---|---|
| GitHub | https://github.com/musmannn/mlops-assignment |
| DagsHub (code, data, MLflow) | https://dagshub.com/musmannn/mlops-assignment |
| MLflow tracking | https://dagshub.com/musmannn/mlops-assignment.mlflow |
| Model API (Vercel) | https://24l-0672-a2-ops.vercel.app (`GET /health`, `POST /predict` with `X-API-Key`) |
| Dataset | [AI Generated Images vs Real Images (Kaggle)](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) |

## Layout

```
visionguard/            application package (shared by training & serving)
  preprocess.py         ONE image->tensor function used by training AND serving
  inference.py          Predictor: loads champion model once, returns label + confidence
  validation.py         upload checks (type, size, decodability, filename)
  web/                  Flask UI: upload, result, history, health + DB trail (SQLAlchemy)
  api/                  JSON model API (deployed to Vercel)
training/               audit, EDA, split, tf.data pipeline, models, train (MLflow), tune,
                        compare, register, predict_by_alias, error analysis, Grad-CAM
labs/                   tool labs: env check, DVC toy, MLflow toy, Keras Tuner toy, Docker warm-up, CI toy
deploy/vercel/          Vercel function (onnxruntime) + deploy script for the model API
deploy/hf_space/        first attempt (HF now requires PRO for Docker Spaces) - kept for the report
tests/                  pytest suite (runs in CI without TensorFlow)
Dockerfile              optimised multi-stage web image (non-root, healthcheck)
Dockerfile.naive        unoptimised "before" image for the size comparison
docker-compose.yml      web + postgres (named volume, healthcheck, no published DB port)
params.yaml             all experiment defaults
```

## Quick start

```bash
# 1. environment (Python 3.11 - TensorFlow does not support 3.13+ yet)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-lock.txt

# 2. data (DVC pulls the exact dataset version from DagsHub storage)
dvc pull

# 3. train + track (needs .env with MLFLOW_TRACKING_* from .env.example)
python -m training.audit && python -m training.split
python -m training.train --model transfer --run-name mnv2_224
python -m training.register            # registers best runs, alias champion, exports models/champion.keras
python -m training.export_onnx         # ONNX copy for the serverless API + equivalence check

# 4. run the whole stack
cp .env.example .env                   # fill in passwords
docker compose up -d --build
open http://localhost:5000
```

Run the web app without Docker (debug off): `python -m visionguard.web.wsgi`.
Use the cloud model instead of a local one: set `INFERENCE_BACKEND=remote`, `MODEL_API_URL`, `MODEL_API_KEY` in `.env`
(or `INFERENCE_BACKEND=remote docker compose up -d web`). Deploy the API with `bash deploy/vercel/deploy.sh`.

Train on a Colab GPU instead: `notebooks/train_colab.ipynb` (needs only `DAGSHUB_TOKEN` in Colab Secrets).

## Tests & lint

```bash
pytest -q && ruff check .
```
