# Screenshot checklist - 24L-0672_A2_Ops

Every screenshot must show **your username + date/time + project name** (terminal prompt or browser
address bar). Run commands from the project folder with the venv active:

```bash
cd ~/Downloads/mlops-assign/24L-0672_A2_Ops && source .venv/bin/activate && date
```

Figures marked **(auto)** are already generated in `reports/figures/` - just insert them.

## Task 1 - Data
| ID | Capture | How |
|---|---|---|
| 1.1 | Versions + GPU check | `python labs/env_check.py` |
| 1.2 | Audit tables | `python -m training.audit` (counts, min/median/max, corrupt, duplicates) |
| 1.3 | Sample grid, histograms **(auto)** | `eda_samples.png`, `eda_histograms.png`, `eda_aspect.png` |
| 1.4 toy | Each DVC step | Create empty DagsHub repo `dvc-toy-24L-0672`, then `bash labs/dvc_toy.sh musmannn` - screenshot init, `cat data.csv.dvc`, `git log`, checkout v1, push, delete, pull |
| 1.4 apply | Dataset in DVC | `cat data/raw.dvc`, `dvc push` / `dvc status -c`, DagsHub repo page showing `data/raw` with the DVC badge |

## Task 2 - Models
| ID | Capture | How |
|---|---|---|
| 2.1 | Augmented batch **(auto)** | `reports/figures/augmented_batch.png` |
| 2.2 | Model summary | `cat reports/figures/cnn_baseline/model_summary.txt` (+ `layer_table.csv` as a table) |
| 2.3 | Curves for both stages, trainable layers **(auto)** | `reports/figures/mnv2_224/curves.png`, `trainable_layers.json` |
| 2.4 | Curves with best epoch **(auto)** | `curves.png` of each model (red dashed line) |
| 2.5 | Metric table, confusion matrices, ROC **(auto)** | `reports/figures/final_comparison/` |
| 2.6 | 10 misclassified **(auto)** | `reports/figures/<champion>/errors.png` |
| 2.7 | Tuner results | `cat reports/tuner_toy_results.csv`, `cat reports/tuner_visionguard_results.csv` |
| bonus | Grad-CAM **(auto)** | `reports/figures/<champion>/gradcam.png` |

## Task 3 - MLflow / DagsHub
| ID | Capture | How |
|---|---|---|
| 3.1 | Local MLflow UI, 2 runs + compare view | `mlflow ui --backend-store-uri sqlite:///mlflow.db` -> open http://127.0.0.1:5000, experiment `toy-fashion-mnist`, tick both runs -> Compare |
| 3.2 | DagsHub project, toy run on DagsHub, `.env` ignored | DagsHub repo page; Experiments tab -> `toy-fashion-mnist`; terminal: `git check-ignore -v .env` and `git status` |
| 3.3 | Run page: params, metric curves, tags (manual AND autolog) | DagsHub MLflow UI -> `mnv2_224` and `cnn_autolog` |
| 3.4 | Artifacts tab | run `mnv2_224` -> Artifacts (evaluation/, splits/, environment/, keras_file/, model) |
| 3.5 | Comparison | select all `VisionGuard` runs -> Compare (parallel coordinates / table) + `reports/figures/run_comparison.png` |
| 3.6 | Registry + load by alias | DagsHub MLflow -> Models -> `24L-0672_A2_Ops_visionguard` (2 versions, alias champion); terminal: `python -m training.predict_by_alias <image>` |

## Task 4 - Flask
| ID | Capture | How |
|---|---|---|
| 4.1 | Console output, 2 known images | `MODEL_PATH=models/champion.keras python -m visionguard.inference <ai_img> <real_img>` |
| 4.2 | 3 failure cases | upload a `.txt` (415), a renamed non-image `fake.jpg` (400), a >8 MB image (413) |
| 4.3 | Upload page, result page, history menu | browser at http://localhost:5000 |
| 4.4 | Health check JSON | http://localhost:5000/health (shows `"debug": false`) |

## Task 5 - Database
| | Capture | How |
|---|---|---|
| 5 | History page + raw rows | `/history`; `docker compose exec db psql -U visionguard -c "select id,created_at,label,confidence,model_version from predictions order by id desc limit 10;"` |

## Task 6 - Docker
| ID | Capture | How |
|---|---|---|
| 6.1 | Each warm-up action | `bash labs/docker_warmup.sh` |
| 6.2 | User inside container | `docker compose exec web whoami` |
| 6.3 | Healthy services, DB port closed | `docker compose ps`; `nc -zv localhost 5432` (refused); `docker compose port db 5432` (nothing) |
| 6.4 | Sizes before/after, largest layers | `docker images 24l-0672_a2_ops-web`; `docker history 24l-0672_a2_ops-web:latest` |
| 6.5 | start/stop/restart/logs/shell, failure diagnosis | see `report/troubleshooting.md` |
| 6.6 | Red + green toy runs, green project run, badge | GitHub Actions tab of the toy repo and of `mlops-assignment`; README badge |

## Task 7 / 8 - Cloud API + integration
| | Capture | How |
|---|---|---|
| 7 | Deployment steps | HF Space page, Settings -> secret `MODEL_API_KEY`, build logs, running status |
| 7 | Postman | POST `{API_URL}/predict`, header `X-API-Key`, body form-data `image` = file; also GET `/health`; a request **without** key -> 401 |
| 7 | Other network | call the API from your phone's mobile data (hotspot) - e.g. Postman/curl on a laptop on hotspot |
| 8 | Flask -> cloud API | `.env`: `INFERENCE_BACKEND=remote`; run `python -m visionguard.web.wsgi`; upload several images; History page shows `backend = remote` |
