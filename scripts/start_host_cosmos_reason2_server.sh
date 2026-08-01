#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

build_bin="${HOST_COSMOS_LLAMA_BIN:-${repo_root}/.codex_diag/llama_cpp_910196f/build-cuda/bin}"
cuda_env="${HOST_COSMOS_CUDA_ENV:-${HOME}/miniconda3/envs/pytorch-build}"
model_dir="${HOST_COSMOS_MODEL_DIR:-${repo_root}/.codex_diag/cosmos_reason2_gguf_910196f}"
port="${HOST_COSMOS_PORT:-18080}"
model_file="${HOST_COSMOS_MODEL_FILE:-Cosmos-Reason2-2B-BF16.gguf}"

export LD_LIBRARY_PATH="${build_bin}:${cuda_env}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${build_bin}/llama-server" \
  -m "${model_dir}/${model_file}" \
  --mmproj "${model_dir}/mmproj-Cosmos-Reason2-2B-F16.gguf" \
  -ngl all \
  -c 16384 \
  --host 127.0.0.1 \
  --port "${port}" \
  --media-path "${repo_root}" \
  --image-min-tokens 1024 \
  --image-max-tokens 1024 \
  --no-cache-prompt \
  --slot-prompt-similarity 0 \
  --no-warmup
