#!/usr/bin/env bash
set -euo pipefail

# Gate BF16 and Q8_0 host candidates through the exact llama.cpp runtime used
# by the demo. The projector is constrained to the trained 448x256 geometry's
# 112 visual tokens instead of padding every request to the old 1024-token
# release contract.

repo_root="${REPO_ROOT:-/mnt/c/Users/eivho/repos/qai-physics-reasoning}"
work_root="${FINETUNE_ROOT:-/home/eivho/cosmos-reason2-finetune}"
python_bin="${PYTHON_BIN:-${work_root}/.venv/bin/python}"
llama_server_bin="${LLAMA_SERVER_BIN_DIR:-/home/eivho/cosmos-reason2-iq9/src/llama.cpp-910196f6/build-cuda-128/bin}"
release_root="${HOST_RELEASE_ROOT:-${work_root}/releases/parcel-speed-v1-448x256-gpu-20260819}"
host_dir="${HOST_RELEASE_DIR:-${release_root}/host}"
projector="mmproj-Cosmos-Reason2-2B-Parcel-Speed-v1-F16.gguf"
port="${HOST_CANDIDATE_PORT:-18081}"
validation_dataset="${VALIDATION_DATASET_ROOT:-/mnt/c/Users/eivho/AppData/Local/QaiConveyor/Saved/Diagnostics/ParcelEvaluation/reason2-ft-parcel-v22-render-synchronized-validation-r1/parcel-quarter-cell-occlusion-safe/exact-v1}"
test_dataset="${TEST_DATASET_ROOT:-/mnt/c/Users/eivho/AppData/Local/QaiConveyor/Saved/Diagnostics/ParcelEvaluation/reason2-ft-parcel-v22-render-synchronized-test-r1/parcel-quarter-cell-occlusion-safe/exact-v1}"
evidence_root="${EVIDENCE_ROOT:-${repo_root}/docs/evidence/results}"
server_pid=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if curl -sf --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
  printf 'Refusing to replace an existing service on port %s.\n' "${port}" >&2
  exit 1
fi
test -r "${host_dir}/${projector}"
mkdir -p "${evidence_root}" "${release_root}/logs"

run_candidate() {
  local label="$1"
  local model_file="$2"
  local lower_label="${label,,}"
  local log="${release_root}/logs/llama-${lower_label}.log"
  local validation_output="${evidence_root}/host_${lower_label}_speed_v1_validation180_20260819.json"
  local test_output="${evidence_root}/host_${lower_label}_speed_v1_test90_20260819.json"
  test -r "${host_dir}/${model_file}"

  env \
    HOST_COSMOS_LLAMA_BIN="${llama_server_bin}" \
    HOST_COSMOS_MODEL_DIR="${host_dir}" \
    HOST_COSMOS_MODEL_FILE="${model_file}" \
    HOST_COSMOS_PROJECTOR_FILE="${projector}" \
    HOST_COSMOS_MODEL_ALIAS=Cosmos-Reason2-2B-Parcel-Speed-v1 \
    HOST_COSMOS_CONTEXT_SIZE=512 \
    HOST_COSMOS_IMAGE_TOKENS=112 \
    HOST_COSMOS_PORT="${port}" \
    "${repo_root}/scripts/start_host_cosmos_reason2_server.sh" \
    >"${log}" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 120); do
    if curl -sf --max-time 2 "http://127.0.0.1:${port}/v1/models" \
        | grep -Fq Cosmos-Reason2-2B-Parcel-Speed-v1; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -120 "${log}" >&2
      return 1
    fi
    sleep 1
  done
  curl -sf --max-time 3 "http://127.0.0.1:${port}/v1/models" \
    | grep -Fq Cosmos-Reason2-2B-Parcel-Speed-v1

  "${python_bin}" "${repo_root}/scripts/reason2_finetune/benchmark_adapter_api.py" \
    --endpoint "http://127.0.0.1:${port}" \
    --model Cosmos-Reason2-2B-Parcel-Speed-v1 \
    --dataset "${validation_dataset}" --split validation --limit-per-class 60 \
    --prompt-profile evk-speed-v1 --image-width 448 --image-height 256 \
    --timeout 60 --output "${validation_output}"
  "${python_bin}" "${repo_root}/scripts/reason2_finetune/benchmark_adapter_api.py" \
    --endpoint "http://127.0.0.1:${port}" \
    --model Cosmos-Reason2-2B-Parcel-Speed-v1 \
    --dataset "${test_dataset}" --split test --limit-per-class 30 \
    --prompt-profile evk-speed-v1 --image-width 448 --image-height 256 \
    --timeout 60 --output "${test_output}"

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  server_pid=""
}

run_candidate BF16 Cosmos-Reason2-2B-Parcel-Speed-v1-BF16.gguf
run_candidate Q8_0 Cosmos-Reason2-2B-Parcel-Speed-v1-Q8_0.gguf

"${python_bin}" - "${evidence_root}" "${release_root}/host_candidate_selection.json" <<'PY'
import json
from pathlib import Path
import sys

evidence_root = Path(sys.argv[1])
output = Path(sys.argv[2])
candidates = {}
for label in ("bf16", "q8_0"):
    reports = [
        json.loads((evidence_root / f"host_{label}_speed_v1_validation180_20260819.json").read_text(encoding="utf-8")),
        json.loads((evidence_root / f"host_{label}_speed_v1_test90_20260819.json").read_text(encoding="utf-8")),
    ]
    candidates[label] = {
        "passed": all(
            report.get("accuracy") == 1.0
            and len(report.get("per_class_recall", {})) == 3
            and all(value == 1.0 for value in report["per_class_recall"].values())
            for report in reports
        ),
        "validation_accuracy": reports[0].get("accuracy"),
        "test_accuracy": reports[1].get("accuracy"),
        "validation_warm_mean_ms": reports[0].get("mean_warm_latency_ms"),
        "test_warm_mean_ms": reports[1].get("mean_warm_latency_ms"),
    }
if not candidates["bf16"]["passed"]:
    raise SystemExit(f"BF16 host gate failed: {candidates['bf16']}")
selected = "q8_0" if candidates["q8_0"]["passed"] else "bf16"
result = {
    "schema_version": 1,
    "prompt_profile": "speed-v1",
    "image_size": "448x256",
    "visual_tokens": 112,
    "selected": selected,
    "candidates": candidates,
}
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
