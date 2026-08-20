#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

build_bin="${HOST_COSMOS_LLAMA_BIN:-${repo_root}/.codex_diag/llama_cpp_910196f/build-cuda/bin}"
cuda_env="${HOST_COSMOS_CUDA_ENV:-${HOME}/miniconda3/envs/pytorch-build}"
model_dir="${HOST_COSMOS_MODEL_DIR:-${repo_root}/.codex_diag/cosmos_reason2_gguf_910196f}"
port="${HOST_COSMOS_PORT:-18084}"
model_file="${HOST_COSMOS_MODEL_FILE:-Cosmos-Reason2-2B-Parcel-Speed-v1-Q8_0.gguf}"
projector_file="${HOST_COSMOS_PROJECTOR_FILE:-mmproj-Cosmos-Reason2-2B-Parcel-Speed-v1-F16.gguf}"
context_size="${HOST_COSMOS_CONTEXT_SIZE:-512}"
image_tokens="${HOST_COSMOS_IMAGE_TOKENS:-112}"
model_alias="${HOST_COSMOS_MODEL_ALIAS:-Cosmos-Reason2-2B-Parcel-Speed-v1}"
disable_warmup="${HOST_COSMOS_DISABLE_WARMUP:-0}"

alias_args=()
if [[ -n "${model_alias}" ]]; then
  alias_args=(--alias "${model_alias}")
fi

warmup_args=()
if [[ "${disable_warmup}" == "1" ]]; then
  warmup_args=(--no-warmup)
fi

export LD_LIBRARY_PATH="${build_bin}:${cuda_env}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${build_bin}/llama-server" \
  -m "${model_dir}/${model_file}" \
  --mmproj "${model_dir}/${projector_file}" \
  -ngl all \
  -c "${context_size}" \
  --host 127.0.0.1 \
  --port "${port}" \
  "${alias_args[@]}" \
  --media-path "${repo_root}" \
  --image-min-tokens "${image_tokens}" \
  --image-max-tokens "${image_tokens}" \
  --flash-attn on \
  --no-cache-prompt \
  --slot-prompt-similarity 0 \
  "${warmup_args[@]}"
