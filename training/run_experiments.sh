#!/usr/bin/env bash
# All VisionGuard experiments (Tasks 2.2-2.5, 3.3, 3.5). Each run changes ONE thing vs the
# reference run "mnv2_224" (MobileNetV2, 224 px, augmentation on, lr 1e-3, last block fine-tuned).
# Runs are logged to the MLflow server in .env (DagsHub).
#
#   bash training/run_experiments.sh            # all runs
#   bash training/run_experiments.sh mnv2_160   # one run by name
#   RUN_PREFIX=colab_ bash training/run_experiments.sh   # e.g. on Colab: runs named colab_mnv2_224, ...
#   EXTRA_ARGS="--epochs 30" bash training/run_experiments.sh   # extra flags for every run
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
COMMON="--epochs 20 --fine-tune-epochs 12 ${EXTRA_ARGS:-}"
PREFIX="${RUN_PREFIX:-}"
mkdir -p reports/logs

declare -A RUNS=(
  [cnn_baseline]="--model cnn --tracking manual --note baseline_cnn_from_scratch"
  [cnn_autolog]="--model cnn --tracking autolog --note same_as_cnn_baseline_but_mlflow_autolog"
  [mnv2_224]="--model transfer --backbone mobilenetv2 --img-size 224 --note reference_transfer_run"
  [mnv2_224_frozen]="--model transfer --backbone mobilenetv2 --img-size 224 --no-fine-tune --note no_fine_tuning_stage"
  [mnv2_224_noaug]="--model transfer --backbone mobilenetv2 --img-size 224 --no-augment --note augmentation_off"
  [mnv2_224_lr3e-4]="--model transfer --backbone mobilenetv2 --img-size 224 --lr 0.0003 --note stage1_lr_0.0003"
  [mnv2_224_unfreeze40]="--model transfer --backbone mobilenetv2 --img-size 224 --unfreeze-layers 40 --note unfreeze_last_40_layers"
  [mnv2_160]="--model transfer --backbone mobilenetv2 --img-size 160 --note image_size_160"
  [effb0_224]="--model transfer --backbone efficientnetb0 --img-size 224 --note backbone_efficientnetb0"
)
ORDER=(mnv2_224 cnn_baseline cnn_autolog mnv2_224_frozen mnv2_224_noaug mnv2_224_lr3e-4 mnv2_224_unfreeze40 mnv2_160 effb0_224)
[[ $# -gt 0 ]] && ORDER=("$@")

for name in "${ORDER[@]}"; do
  run="${PREFIX}${name}"
  echo "=== $(date +%H:%M:%S) start $run"
  $PY -m training.train ${RUNS[$name]} $COMMON --run-name "$run" > "reports/logs/$run.log" 2>&1
  echo "=== $(date +%H:%M:%S) end $run exit=$? :: $(grep -E '^TEST:' "reports/logs/$run.log")"
done
