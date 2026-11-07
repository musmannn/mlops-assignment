# VisionGuard · 24L-0672_A2_Ops

[![CI](https://github.com/<github-user>/24L-0672_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/<github-user>/24L-0672_A2_Ops/actions/workflows/ci.yml)

A web service that says whether an uploaded image is **AI-generated or real**:
a TensorFlow/Keras model (MobileNetV2 fine-tuned), tracked with MLflow on DagsHub, data versioned with DVC,
served by Flask with a PostgreSQL history, containerised with Docker Compose. The model API is deployed on
Hugging Face Spaces.

| | |
|---|---|
| GitHub | `https://github.com/<github-user>/24L-0672_A2_Ops` |
| DagsHub (code, data, MLflow) | `https://dagshub.com/<dagshub-user>/24L-0672_A2_Ops` |
| Model API (Hugging Face Space) | `https://<hf-user>-24l-0672-a2-ops.hf.space` |
| Dataset | [AI Generated Images vs Real Images (Kaggle)](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) |

## Layout

```
visionguard/            application package (shared by training & serving)
  preprocess.py         ONE image->tensor function used by training AND serving
  inference.py          Predictor: loads champion model once, returns label + confidence
  validation.py         upload checks (type, size, decodability, filename)
  web/                  Flask UI: upload, result, history, health + DB trail (SQLAlchemy)
  api/                  JSON model API deployed to Hugging Face Spaces
training/               audit, EDA, split, tf.data pipeline, models, train (MLflow), tune,
                        compare, register, predict_by_alias, error analysis, Grad-CAM
labs/                   tool labs: env check, DVC toy, MLflow toy, Keras Tuner toy, Docker warm-up, CI toy
deploy/hf_space/        Dockerfile + deploy script for the model API
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

# 4. run the whole stack
cp .env.example .env                   # fill in passwords
docker compose up -d --build
open http://localhost:5000
```

Run the web app without Docker (debug off): `python -m visionguard.web.wsgi`.
Use the cloud model instead of a local one: set `INFERENCE_BACKEND=remote`, `MODEL_API_URL`, `MODEL_API_KEY` in `.env`.

## Tests & lint

```bash
pytest -q && ruff check .
```
