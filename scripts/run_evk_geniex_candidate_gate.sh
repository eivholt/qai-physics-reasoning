#!/usr/bin/env bash
set -euo pipefail

# Run one imported GenieX candidate as the EVK's only resident Reason2 worker,
# evaluate the frozen synchronized panel, and always restore the production
# GenieX service. This uses a transient systemd unit so an interrupted SSH
# shell cannot orphan a second HTP model.

repo_root="${REPO_ROOT:-/mnt/c/Users/eivho/repos/qai-physics-reasoning}"
python_bin="${PYTHON_BIN:-/home/eivho/cosmos-reason2-finetune/.venv/bin/python}"
evk_target="${EVK_TARGET:-ubuntu@192.168.1.158}"
evk_host="${EVK_HOST:-192.168.1.158}"
runtime_root="${GENIEX_ROOT:-/home/ubuntu/qai-conveyor/runtime/geniex-v0317-qairt245}"
data_dir="${GENIEX_DATADIR:?Set GENIEX_DATADIR to the imported candidate data directory}"
model_id="${GENIEX_MODEL_ID:-local/cosmos-reason2-2b}"
port="${GENIEX_PORT:-18184}"
context_length="${GENIEX_CONTEXT_LENGTH:-512}"
qairt_home="${QAIRT_HOME:-/opt/qairt/2.45.0.260326}"
media_root="${EVK_MEDIA_ROOT:-/home/ubuntu/qai-conveyor/run/media/validation-v22-fresh-v3}"
dataset_seed="${EVK_DATASET_SEED:-68201}"
count_per_class="${EVK_COUNT_PER_CLASS:-30}"
output="${OUTPUT:?Set OUTPUT to the local JSON evidence path}"
require_accuracy="${REQUIRE_ACCURACY:-1}"
stream="${EVK_STREAM:-1}"
soak_repeats="${SOAK_REPEATS:-1}"
local_dataset="${LOCAL_DATASET:-}"
dataset_split="${DATASET_SPLIT:-validation}"
image_width="${IMAGE_WIDTH:-448}"
image_height="${IMAGE_HEIGHT:-256}"
packaged_client="${PACKAGED_CLIENT:-}"
client_smoke_output="${CLIENT_SMOKE_OUTPUT:-}"
client_smoke_trigger="${CLIENT_SMOKE_TRIGGER_FILE:-}"
candidate_unit=qai-conveyor-geniex-candidate.service
production_unit=qai-conveyor-geniex.service
production_was_active=false

if [[ "$(ssh -o BatchMode=yes "${evk_target}" systemctl is-active "${production_unit}" 2>/dev/null || true)" == active ]]; then
  production_was_active=true
fi

restore_production() {
  status=$?
  trap - EXIT INT TERM
  ssh -o BatchMode=yes "${evk_target}" bash -s -- \
    "${candidate_unit}" "${production_unit}" "${production_was_active}" "${port}" <<'REMOTE' || true
set -u
candidate_unit="$1"
production_unit="$2"
production_was_active="$3"
port="$4"
sudo -n systemctl stop "${candidate_unit}" 2>/dev/null || true
sudo -n systemctl reset-failed "${candidate_unit}" 2>/dev/null || true
for _ in $(seq 1 30); do
  if ! pgrep -f "geniex-grammar.*serve.*${port}" >/dev/null; then
    break
  fi
  sleep 1
done
if [ "${production_was_active}" = true ]; then
  sudo -n systemctl reset-failed "${production_unit}" || true
  sudo -n systemctl start "${production_unit}" || true
fi
REMOTE
  exit "${status}"
}
trap restore_production EXIT INT TERM

ssh -o BatchMode=yes "${evk_target}" bash -s -- \
  "${runtime_root}" "${data_dir}" "${port}" "${context_length}" \
  "${qairt_home}" "${candidate_unit}" "${production_unit}" <<'REMOTE'
set -euo pipefail
runtime_root="$1"
data_dir="$2"
port="$3"
context_length="$4"
qairt_home="$5"
candidate_unit="$6"
production_unit="$7"
qairt_target=aarch64-oe-linux-gcc11.2

test -x "${runtime_root}/geniex-grammar"
test -d "${data_dir}"
test -d "${qairt_home}/lib/${qairt_target}"
test -d "${qairt_home}/lib/hexagon-v73/unsigned"
sudo -n systemctl stop "${candidate_unit}" 2>/dev/null || true
sudo -n systemctl reset-failed "${candidate_unit}" 2>/dev/null || true
sudo -n systemctl stop "${production_unit}"
for _ in $(seq 1 30); do
  if ! pgrep -f 'geniex-grammar.*serve.*18183' >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f 'geniex-grammar.*serve.*18183' >/dev/null; then
  printf 'Production GenieX worker did not stop; refusing the candidate.\n' >&2
  exit 1
fi

sudo -n systemd-run \
  --unit="${candidate_unit%.service}" \
  --collect \
  --property=User=ubuntu \
  --property=Group=ubuntu \
  --property=WorkingDirectory="${data_dir}" \
  --setenv="GENIEX_DATADIR=${data_dir}" \
  --setenv="GENIEX_PLUGIN_PATH=${runtime_root}" \
  --setenv="PATH=${qairt_home}/bin/${qairt_target}:${qairt_home}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --setenv="LD_LIBRARY_PATH=${runtime_root}:${runtime_root}/qairt:${runtime_root}/llama_cpp:${qairt_home}/lib/${qairt_target}:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu" \
  --setenv="ADSP_LIBRARY_PATH=${qairt_home}/lib/hexagon-v73/unsigned" \
  "${runtime_root}/geniex-grammar" --skip-update serve \
    --host "0.0.0.0:${port}" --keepalive 3600 --compute npu \
    --nctx "${context_length}" --ngl -1
REMOTE

endpoint="http://${evk_host}:${port}"
for _ in $(seq 1 120); do
  models=$(curl -sf --max-time 3 "${endpoint}/v1/models" || true)
  if printf '%s' "${models}" | grep -Fq "${model_id}"; then
    break
  fi
  sleep 1
done
models=$(curl -sf --max-time 3 "${endpoint}/v1/models")
printf '%s\n' "${models}"
printf '%s' "${models}" | grep -Fq "${model_id}"
IFS=$'\t' read -r htp_profile bundle_archive_sha256 <<<"$(
  ssh -o BatchMode=yes "${evk_target}" python3 - "${data_dir}" <<'PY'
import json
from pathlib import Path
import sys

data_dir = Path(sys.argv[1])
marker = data_dir / "qai_htp_profile_override.json"
if marker.is_file():
    profile = json.loads(marker.read_text(encoding="utf-8")).get("profile", "unknown")
else:
    profile = "bundle_default"
import_marker = data_dir / "qai_conveyor_import.json"
if import_marker.is_file():
    archive_sha256 = json.loads(import_marker.read_text(encoding="utf-8")).get(
        "bundle_archive_sha256"
    ) or ""
else:
    archive_sha256 = ""
print(f"{profile}\t{archive_sha256}")
PY
)"

if [[ -n "${local_dataset}" ]]; then
  stream_args=()
  if [[ "${stream}" == 1 ]]; then
    stream_args=(--stream)
  fi
  "${python_bin}" "${repo_root}/scripts/reason2_finetune/benchmark_adapter_api.py" \
    --endpoint "${endpoint}" \
    --model "${model_id}" \
    --dataset "${local_dataset}" \
    --split "${dataset_split}" \
    --limit-per-class "${count_per_class}" \
    --repeat "${soak_repeats}" \
    --prompt-profile evk-speed-v1 \
    --image-width "${image_width}" \
    --image-height "${image_height}" \
    --timeout 60 \
    "${stream_args[@]}" \
    --output "${output}"
else
  stream_args=()
  if [[ "${stream}" == 1 ]]; then
    stream_args=(--stream)
  fi
  "${python_bin}" "${repo_root}/scripts/reason2_finetune/validate_evk_http_service.py" \
    --endpoint "${endpoint}" \
    --model "${model_id}" \
    --media-root "${media_root}" \
    --media-extension png \
    --seed "${dataset_seed}" \
    --count-per-class "${count_per_class}" \
    --prompt-profile evk-speed-v1 \
    --timeout 60 \
    "${stream_args[@]}" \
    --output "${output}"
fi

"${python_bin}" - \
  "${output}" "${data_dir}" "${runtime_root}" "${qairt_home}" \
  "${htp_profile}" "${bundle_archive_sha256}" "${context_length}" "${port}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8-sig"))
report["runtime_context"] = {
    "geniex_data_dir": sys.argv[2],
    "geniex_runtime": sys.argv[3],
    "qairt_home": sys.argv[4],
    "htp_profile": sys.argv[5],
    "bundle_archive_sha256": sys.argv[6] or None,
    "context_length": int(sys.argv[7]),
    "port": int(sys.argv[8]),
    "direct_embedded_image": bool(report.get("dataset")),
    "stream": bool(report.get("stream")),
}
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

"${python_bin}" - "${output}" "${require_accuracy}" <<'PY'
import json
import sys

path, require_accuracy = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
if require_accuracy == "1":
    if result["accuracy"] != 1.0:
        raise SystemExit(f"accuracy gate failed: {result['accuracy']}")
    failed = {
        key: value
        for key, value in result["per_class_recall"].items()
        if value != 1.0
    }
    if failed:
        raise SystemExit(f"per-class gate failed: {failed}")
print(json.dumps({
    "accuracy": result["accuracy"],
    "per_class_recall": result["per_class_recall"],
    "mean_latency_ms": result.get("mean_latency_ms"),
    "mean_warm_latency_ms": result.get("mean_warm_latency_ms"),
}, separators=(",", ":")))
PY

if [[ -n "${packaged_client}" ]]; then
  if [[ -z "${client_smoke_output}" ]]; then
    printf 'Set CLIENT_SMOKE_OUTPUT when PACKAGED_CLIENT is set.\n' >&2
    exit 2
  fi
  test -f "${packaged_client}"
  if [[ -n "${client_smoke_trigger}" ]]; then
    rm -f -- "${client_smoke_output}" "${client_smoke_trigger}"
    mkdir -p "$(dirname -- "${client_smoke_trigger}")"
    printf '{"endpoint":"%s","model":"%s"}\n' \
      "${endpoint}" "${model_id}" >"${client_smoke_trigger}"
    for _ in $(seq 1 600); do
      if [[ -s "${client_smoke_output}" ]]; then
        break
      fi
      sleep 0.5
    done
    test -s "${client_smoke_output}"
    "${python_bin}" - "${client_smoke_output}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8-sig"))
if result.get("passed") is not True:
    raise SystemExit("packaged-client smoke did not report passed=true")
print("packaged_client_gate=passed")
PY
    rm -f -- "${client_smoke_trigger}"
  else
    smoke_script="${repo_root}/unreal_conveyor_demo/Scripts/smoke_packaged_speed_v1.ps1"
    test -f "${smoke_script}"
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -File "$(wslpath -w "${smoke_script}")" \
      -ClientExecutable "$(wslpath -w "${packaged_client}")" \
      -EvkServer "${endpoint}" \
      -EvkModel "${model_id}" \
      -Output "$(wslpath -w "${client_smoke_output}")"
  fi
fi

ssh -o BatchMode=yes "${evk_target}" bash -s -- "${candidate_unit}" "${port}" <<'REMOTE'
set -euo pipefail
candidate_unit="$1"
port="$2"
test "$(systemctl is-active "${candidate_unit}")" = active
test "$(pgrep -fc "geniex-grammar.*serve.*${port}")" = 1
REMOTE
