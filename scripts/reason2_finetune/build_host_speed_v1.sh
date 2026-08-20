#!/usr/bin/env bash
set -euo pipefail

# Build the static RTX-host artifacts from the same gated speed-v1 adapter used
# for the EVK export.  BF16 is the accuracy reference; Q8_0 remains a candidate
# until it independently passes both frozen image sets.

repo_root="${REPO_ROOT:-/mnt/c/Users/eivho/repos/qai-physics-reasoning}"
work_root="${FINETUNE_ROOT:-/home/eivho/cosmos-reason2-finetune}"
python_bin="${PYTHON_BIN:-${work_root}/.venv/bin/python}"
llama_cpp="${LLAMA_CPP:-/home/eivho/cosmos-reason2-iq9/src/llama.cpp-910196f6}"
adapter="${ADAPTER:-${work_root}/adapters/parcel-generative-evk-speed-v1-448x256}"
merged="${MERGED:-${work_root}/merged/parcel-generative-evk-speed-v1-448x256}"
release_root="${HOST_RELEASE_ROOT:-${work_root}/releases/parcel-speed-v1-448x256-gpu-20260819}"
host_dir="${HOST_RELEASE_DIR:-${release_root}/host}"
bf16="${host_dir}/Cosmos-Reason2-2B-Parcel-Speed-v1-BF16.gguf"
q8="${host_dir}/Cosmos-Reason2-2B-Parcel-Speed-v1-Q8_0.gguf"
projector="${host_dir}/mmproj-Cosmos-Reason2-2B-Parcel-Speed-v1-F16.gguf"

"${python_bin}" - "${adapter}" <<'PY'
import json
from pathlib import Path
import sys

adapter = Path(sys.argv[1])
for name, expected_count in (("validation-speed-v1.json", 180), ("test-speed-v1.json", 90)):
    path = adapter / name
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("count") != expected_count or result.get("accuracy") != 1.0:
        raise SystemExit(f"host gate failed: {path}")
    failed = {
        key: value
        for key, value in result.get("per_class_recall", {}).items()
        if value != 1.0
    }
    if failed:
        raise SystemExit(f"per-class host gate failed: {path}: {failed}")
print("host_adapter_gate=passed validation=180/180 test=90/90")
PY

test -r "${llama_cpp}/convert_hf_to_gguf.py"
test -x "${llama_cpp}/build/bin/llama-quantize"
mkdir -p "${host_dir}"

if [[ ! -r "${merged}/adapter_merge_manifest.json" ]]; then
  "${python_bin}" "${repo_root}/scripts/reason2_finetune/merge_adapter.py" \
    --model "${work_root}/base/Cosmos-Reason2-2B" \
    --adapter "${adapter}" \
    --output "${merged}"
fi

if [[ ! -r "${bf16}" ]]; then
  "${python_bin}" "${llama_cpp}/convert_hf_to_gguf.py" \
    "${merged}" --outfile "${bf16}" --outtype bf16
fi

if [[ ! -r "${projector}" ]]; then
  "${python_bin}" "${llama_cpp}/convert_hf_to_gguf.py" \
    "${merged}" --mmproj --outfile "${projector}" --outtype f16
fi

if [[ ! -r "${q8}" ]]; then
  "${llama_cpp}/build/bin/llama-quantize" "${bf16}" "${q8}" Q8_0
fi

sha256sum "${bf16}" "${q8}" "${projector}" > "${host_dir}/SHA256SUMS"
"${python_bin}" - "${host_dir}" "${merged}" "${adapter}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

host_dir, merged, adapter = map(Path, sys.argv[1:])
artifacts = []
for path in sorted(host_dir.glob("*.gguf")):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    artifacts.append({
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
manifest = {
    "schema_version": 1,
        "model_alias": "Cosmos-Reason2-2B-Parcel-Speed-v1",
    "prompt_profile": "speed-v1",
    "image_width": 448,
    "image_height": 256,
    "max_completion_tokens": 1,
    "llama_cpp_revision": "910196f6b3dfc6aca88fa732e2b02f270ff9b56b",
    "merged_checkpoint": str(merged),
    "source_adapter": str(adapter),
    "artifacts": artifacts,
    "promotion_status": "candidate_pending_gguf_accuracy_gate",
}
(host_dir.parent / "host_candidate_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
PY

printf 'host_speed_v1_release=%s\n' "${release_root}"
