#!/usr/bin/env bash
set -euo pipefail

bundle_dir="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
qairt_home="${QAIRT_HOME:-/opt/qairt/2.45.0.260326}"
target="aarch64-oe-linux-gcc11.2"
genie_binary="${qairt_home}/bin/${target}/genie-t2t-run"
qairt_target_lib="${qairt_home}/lib/${target}"
qairt_dsp_lib="${qairt_home}/lib/hexagon-v73/unsigned"
genie_config="${GENIE_CONFIG:-${bundle_dir}/genie_config.json}"
prompt_file="${PROMPT_FILE:-${bundle_dir}/smoke_prompt.txt}"

if [[ ! -x "${genie_binary}" ]]; then
  echo "Genie runner not executable: ${genie_binary}" >&2
  exit 1
fi
if [[ ! -d "${qairt_target_lib}" || ! -d "${qairt_dsp_lib}" ]]; then
  echo "Incomplete QAIRT runtime below ${qairt_home}" >&2
  exit 1
fi

if [[ ! -f "${genie_config}" ]]; then
  mapfile -d '' found_configs < <(
    find "${bundle_dir}" -name genie_config.json -type f -print0
  )
  if [[ "${#found_configs[@]}" -eq 0 ]]; then
    echo "genie_config.json not found below ${bundle_dir}" >&2
    exit 1
  fi
  if [[ "${#found_configs[@]}" -ne 1 ]]; then
    echo "Expected exactly one genie_config.json below ${bundle_dir}; found ${#found_configs[@]}" >&2
    printf '  %s\n' "${found_configs[@]}" >&2
    exit 1
  fi
  genie_config="${found_configs[0]}"
  bundle_dir="$(dirname "${genie_config}")"
fi

if [[ ! -f "${prompt_file}" ]]; then
  prompt_file="${bundle_dir}/smoke_prompt.txt"
  printf '%s\n' \
    '<|im_start|>system' \
    'You are a helpful assistant.<|im_end|>' \
    '<|im_start|>user' \
    'A ball is released from rest. In one short sentence, what happens next and why?<|im_end|>' \
    '<|im_start|>assistant' > "${prompt_file}"
fi
profile_file="${PROFILE_FILE:-${bundle_dir}/text_smoke_profile.txt}"

export PATH="${qairt_home}/bin/${target}:${qairt_home}/bin:${PATH}"
qairt_library_paths=("${qairt_target_lib}")
legacy_qairt_lib="${qairt_home}/lib/aarch64-oe-linux-gcc8.2"
if [[ -d "${legacy_qairt_lib}" ]]; then
  qairt_library_paths+=("${legacy_qairt_lib}")
fi
qairt_library_paths+=("/usr/lib/aarch64-linux-gnu" "/lib/aarch64-linux-gnu")
export LD_LIBRARY_PATH="$(IFS=:; echo "${qairt_library_paths[*]}")"
export ADSP_LIBRARY_PATH="${qairt_dsp_lib}"

cd "${bundle_dir}"

printf 'QAIRT_HOME=%s\nGenie config=%s\nProfile=%s\n' \
  "${qairt_home}" "${genie_config}" "${profile_file}"

genie_args=(
  -c "${genie_config}"
  --prompt_file "${prompt_file}"
  --profile "${profile_file}"
)
if [[ -n "${ABORT_AFTER_MS:-}" ]]; then
  if [[ ! "${ABORT_AFTER_MS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ABORT_AFTER_MS must be a positive integer." >&2
    exit 2
  fi
  genie_args+=(--action ABORT --sleep "${ABORT_AFTER_MS}")
fi

"${genie_binary}" "${genie_args[@]}"
