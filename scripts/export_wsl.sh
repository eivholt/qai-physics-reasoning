#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 QUANTIZED_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

checkpoint="$(realpath "$1")"
output_dir="$2"
env_name="${QAI_COSMOS_ENV:-qai-cosmos-reason2}"
device="${QAI_DEVICE:-Dragonwing IQ-9075 EVK}"
device_os="${QAI_DEVICE_OS:-1.7}"
full_context_matrix="${FULL_CONTEXT_MATRIX:-0}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${env_name}"

mkdir -p "${output_dir}"

matrix_args=()
case "${full_context_matrix}" in
  0|false|FALSE|no|NO)
    ;;
  1|true|TRUE|yes|YES)
    matrix_args+=(--full-context-matrix)
    ;;
  *)
    echo "FULL_CONTEXT_MATRIX must be 0/1, false/true, or no/yes." >&2
    exit 2
    ;;
esac

python -m qai_hub_models.models.cosmos_reason2_2b.export \
  --checkpoint "${checkpoint}" \
  --target-runtime geniex_qairt \
  --device "${device}" \
  --device-os "${device_os}" \
  --skip-profiling \
  --output-dir "${output_dir}" \
  --zip-assets \
  "${matrix_args[@]}"
