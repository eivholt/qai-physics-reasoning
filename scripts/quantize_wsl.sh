#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR OUTPUT_DIR" >&2
  exit 2
fi

checkpoint="$(realpath "$1")"
output_dir="$2"
env_name="${QAI_COSMOS_ENV:-qai-cosmos-reason2}"
context_length="${CONTEXT_LENGTH:-2048}"
calibration_sequence_length="${CALIBRATION_SEQUENCE_LENGTH:-128}"
num_samples="${NUM_SAMPLES:-20}"
veg_num_samples="${VEG_NUM_SAMPLES:-100}"
image_height="${IMAGE_HEIGHT:-512}"
image_width="${IMAGE_WIDTH:-512}"
paired_calibration_manifest="${VEG_PAIRED_CALIBRATION_MANIFEST:-}"
paired_calibration_args=()
if [[ -n "${paired_calibration_manifest}" ]]; then
  paired_calibration_args=(
    --veg-paired-calibration-manifest
    "$(realpath "${paired_calibration_manifest}")"
  )
fi
fp16_last_block_activations="${VEG_FP16_LAST_BLOCK_ACTIVATIONS:-0}"
fp16_last_block_args=()
case "${fp16_last_block_activations,,}" in
  1|true|yes)
    fp16_last_block_args=(--veg-fp16-last-block-activations)
    ;;
  0|false|no)
    ;;
  *)
    echo "VEG_FP16_LAST_BLOCK_ACTIVATIONS must be 0/1, false/true, or no/yes" >&2
    exit 2
    ;;
esac

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${env_name}"

python "$(dirname "${BASH_SOURCE[0]}")/validate_model_config.py" "${checkpoint}"

mkdir -p "$(dirname "${output_dir}")"

python -m qai_hub_models.models.cosmos_reason2_2b.quantize \
  --checkpoint "${checkpoint}" \
  --context-length "${context_length}" \
  --calibration-sequence-length "${calibration_sequence_length}" \
  --num-samples "${num_samples}" \
  --veg-num-samples "${veg_num_samples}" \
  --image-size "${image_height}" "${image_width}" \
  --precision w4a16 \
  "${paired_calibration_args[@]}" \
  "${fp16_last_block_args[@]}" \
  --output-dir "${output_dir}"

python "$(dirname "${BASH_SOURCE[0]}")/finalize_checkpoint.py" \
  "${checkpoint}" \
  "${output_dir}"
