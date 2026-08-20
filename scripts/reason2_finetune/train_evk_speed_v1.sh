#!/usr/bin/env bash
set -euo pipefail

# EVK speed-v1: preserve the accepted synchronized v22 training distribution,
# but teach the runtime's compact user-only prompt and exact one-token G/A/R
# completion.  Start at 448x256 (112 visual tokens); 384x224 remains the next
# controlled candidate only if this accuracy gate passes.

repo_root="${REPO_ROOT:-/mnt/c/Users/eivho/repos/qai-physics-reasoning}"
work_root="${FINETUNE_ROOT:-/home/eivho/cosmos-reason2-finetune}"
python_bin="${PYTHON_BIN:-${work_root}/.venv/bin/python}"
dataset_root="${DATASET_ROOT:-/mnt/c/Users/eivho/AppData/Local/QaiConveyor/Saved/Diagnostics/ParcelEvaluation/reason2-ft-parcel-v22-render-synchronized-train-r1/parcel-quarter-cell-occlusion-safe/exact-v1}"
output_dir="${OUTPUT_DIR:-${work_root}/adapters/parcel-generative-evk-speed-v1-448x256}"

exec "${python_bin}" "${repo_root}/scripts/reason2_finetune/train_generative.py" \
  --model "${work_root}/base/Cosmos-Reason2-2B" \
  --dataset "${dataset_root}" \
  --split train \
  --output "${output_dir}" \
  --epochs 1 \
  --matched-boundary-triplets 1000 \
  --evk-speed-classifier \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --learning-rate 1e-4 \
  --bf16-base \
  --no-gradient-checkpointing \
  --seed 20260819 \
  --image-width 448 \
  --image-height 256
