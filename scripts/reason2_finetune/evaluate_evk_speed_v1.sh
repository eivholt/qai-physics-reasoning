#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/c/Users/eivho/repos/qai-physics-reasoning}"
work_root="${FINETUNE_ROOT:-/home/eivho/cosmos-reason2-finetune}"
python_bin="${PYTHON_BIN:-${work_root}/.venv/bin/python}"
adapter="${ADAPTER:-${work_root}/adapters/parcel-generative-evk-speed-v1-448x256}"

for split in validation test; do
  limit=60
  dataset_root="${VALIDATION_DATASET_ROOT:-/mnt/c/Users/eivho/AppData/Local/QaiConveyor/Saved/Diagnostics/ParcelEvaluation/reason2-ft-parcel-v22-render-synchronized-validation-r1/parcel-quarter-cell-occlusion-safe/exact-v1}"
  if [[ "${split}" == "test" ]]; then
    limit=30
    dataset_root="${TEST_DATASET_ROOT:-/mnt/c/Users/eivho/AppData/Local/QaiConveyor/Saved/Diagnostics/ParcelEvaluation/reason2-ft-parcel-v22-render-synchronized-test-r1/parcel-quarter-cell-occlusion-safe/exact-v1}"
  fi
  "${python_bin}" "${repo_root}/scripts/reason2_finetune/evaluate_generative.py" \
    --model "${work_root}/base/Cosmos-Reason2-2B" \
    --adapter "${adapter}" \
    --dataset "${dataset_root}" \
    --split "${split}" \
    --limit-per-class "${limit}" \
    --output "${adapter}/${split}-speed-v1.json" \
    --evk-speed-classifier \
    --image-width 448 \
    --image-height 256 \
    --max-new-tokens 1
done
