#!/usr/bin/env bash
set -euo pipefail

# Build the gated 448x256 / CL512 speed-v1 candidate. The output deliberately
# retains the full DeepStack interface now proven to run through stock GenieX;
# no legacy QAIRT-compatibility pruning is applied.

repo_root="${REPO_ROOT:-/mnt/c/Users/eivho/repos/qai-physics-reasoning}"
work_root="${FINETUNE_ROOT:-/home/eivho/cosmos-reason2-finetune}"
train_python="${TRAIN_PYTHON:-${work_root}/.venv/bin/python}"
conda_env="${QAI_COSMOS_ENV:-qai-cosmos-reason2}"
adapter="${ADAPTER:-${work_root}/adapters/parcel-generative-evk-speed-v1-448x256}"
merged="${MERGED:-${work_root}/merged/parcel-generative-evk-speed-v1-448x256}"
quantized="${QUANTIZED:-${work_root}/qairt/parcel-speed-v1-cl512-256x448-w4a16}"
w4_fp16="${W4_FP16:-${work_root}/qairt/parcel-speed-v1-cl512-256x448-w4-fp16-full-deepstack}"
w8_text="${W8_TEXT:-${work_root}/qairt/parcel-speed-v1-cl512-256x448-w8text-full-deepstack}"
export_root="${EXPORT_ROOT:-${work_root}/exports/parcel-speed-v1-cl512-256x448-w8text-full-deepstack}"
bundle="${BUNDLE:-${work_root}/bundles/cosmos-reason2-parcel-speed-v1-cl512-256x448-w8text-geniex-qairt245-os19-r1}"
archive="${ARCHIVE:-${bundle}.tar.gz}"
calibration_manifest="${CALIBRATION_MANIFEST:-${work_root}/calibration/parcel-speed-v1-448x256-single120/paired_calibration_manifest.json}"
device="${QAI_DEVICE:-Dragonwing IQ-9075 EVK}"
device_os="${QAI_DEVICE_OS:-1.9}"

"${train_python}" - "${adapter}" <<'PY'
import json
from pathlib import Path
import sys

adapter = Path(sys.argv[1])
required = (("validation-speed-v1.json", 180), ("test-speed-v1.json", 90))
for name, expected_count in required:
    path = adapter / name
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("count") != expected_count or result.get("accuracy") != 1.0:
        raise SystemExit(f"host gate failed: {path}")
    if any(value != 1.0 for value in result.get("per_class_recall", {}).values()):
        raise SystemExit(f"per-class host gate failed: {path}")
print("host_gate=passed validation=180/180 test=90/90")
PY

if [[ ! -r "${merged}/adapter_merge_manifest.json" ]]; then
  "${train_python}" "${repo_root}/scripts/reason2_finetune/merge_adapter.py" \
    --model "${work_root}/base/Cosmos-Reason2-2B" \
    --adapter "${adapter}" \
    --output "${merged}"
fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${conda_env}"
cd "${repo_root}"

if [[ ! -r "${quantized}/args.json" ]]; then
  python -m qai_hub_models.models.cosmos_reason2_2b.quantize \
    --checkpoint "${merged}" \
    --context-length 512 \
    --calibration-sequence-length 128 \
    --num-samples 8 \
    --veg-num-samples 120 \
    --image-size 256 448 \
    --precision w4a16 \
    --veg-paired-calibration-manifest "${calibration_manifest}" \
    --output-dir "${quantized}"
  python scripts/finalize_checkpoint.py "${merged}" "${quantized}"
fi

if [[ ! -r "${w4_fp16}/w4_fp16.json" ]]; then
  python scripts/make_w4_fp16_checkpoint.py \
    --source "${quantized}" \
    --destination "${w4_fp16}" \
    --keep-vision-w4a16
fi

if [[ ! -r "${w8_text}/text_w8_matrices.json" ]]; then
  python scripts/make_text_w8_matrix_checkpoint.py \
    "${w4_fp16}" "${w8_text}" \
    --parts part1_of_4 part2_of_4 part3_of_4 part4_of_4 \
    --context-length 512 \
    --image-size 256 448
fi

if [[ ! -d "${export_root}" ]] || ! find "${export_root}" -type f -name part4_of_4.bin -print -quit | grep -q .; then
  mkdir -p "${export_root}"
  python -m qai_hub_models.models.cosmos_reason2_2b.export \
    --checkpoint "${w8_text}" \
    --runtime geniex_qairt \
    --device "${device}" \
    --device-os "${device_os}" \
    --context-length 512 \
    --image-size 256 448 \
    --skip-profiling \
    --output-dir "${export_root}"
fi

mapfile -t export_candidates < <(
  find "${export_root}" -type f -name metadata.json -printf '%h\n' \
    | while read -r candidate; do
        if [[ -r "${candidate}/vision_encoder.bin" \
            && -r "${candidate}/part1_of_4.bin" \
            && -r "${candidate}/part2_of_4.bin" \
            && -r "${candidate}/part3_of_4.bin" \
            && -r "${candidate}/part4_of_4.bin" ]]; then
          printf '%s\n' "${candidate}"
        fi
      done
)
if [[ "${#export_candidates[@]}" -ne 1 ]]; then
  printf 'Expected one complete downloaded export, found %d\n' \
    "${#export_candidates[@]}" >&2
  exit 1
fi

if [[ ! -r "${bundle}/geniex_compat.json" ]]; then
  python scripts/prepare_geniex_bundle.py \
    --source "${export_candidates[0]}" \
    --destination "${bundle}" \
    --model-id qwen3_vl_cosmos_reason2_parcel_speed_v1
fi

if [[ ! -r "${archive}" ]]; then
  tar -C "$(dirname -- "${bundle}")" -czf "${archive}" "$(basename -- "${bundle}")"
fi

printf 'speed_v1_bundle=%s\n' "${bundle}"
printf 'speed_v1_archive=%s\n' "${archive}"
sha256sum "${archive}"
