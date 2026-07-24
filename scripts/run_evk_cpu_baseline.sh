#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 MODEL_GGUF MMPROJ_GGUF [IMAGE]" >&2
  exit 2
fi

model="$(realpath "$1")"
mmproj="$(realpath "$2")"
image="${3:-}"
llama_root="${LLAMA_ROOT:-/home/ubuntu/llama.cpp}"
binary="${llama_root}/build/bin/llama-mtmd-cli"

args=(
  -m "${model}"
  --mmproj "${mmproj}"
  --threads "${LLAMA_THREADS:-8}"
  --threads-batch "${LLAMA_THREADS_BATCH:-8}"
  --ctx-size "${LLAMA_CONTEXT_LENGTH:-2048}"
  --predict "${LLAMA_PREDICT:-256}"
  --temp "${LLAMA_TEMPERATURE:-0.2}"
  --top-p "${LLAMA_TOP_P:-0.9}"
  -p "Explain what happens next using physical common sense. Return <think>reasoning</think> followed by a concise answer."
)

if [[ -n "${image}" ]]; then
  args+=(--image "$(realpath "${image}")")
  if [[ -n "${LLAMA_IMAGE_MIN_TOKENS:-}" ]]; then
    args+=(--image-min-tokens "${LLAMA_IMAGE_MIN_TOKENS}")
  fi
fi

exec "${binary}" "${args[@]}"
