#!/usr/bin/env bash
set -euo pipefail

env_name="${QAI_COSMOS_ENV:-qai-cosmos-reason2}"
output_dir="${1:-${HOME}/models/Cosmos-Reason2-2B}"
conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"

source "${conda_sh}"
conda activate "${env_name}"

if ! hf auth whoami >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Hugging Face is not authenticated in WSL.

1. Accept the terms at:
   https://huggingface.co/nvidia/Cosmos-Reason2-2B
2. Run: hf auth login
   Or export HF_TOKEN only for the current shell.
3. Re-run this script.
EOF
  exit 2
fi

mkdir -p "${output_dir}"
hf download nvidia/Cosmos-Reason2-2B \
  --local-dir "${output_dir}"

python "$(dirname "${BASH_SOURCE[0]}")/validate_model_config.py" "${output_dir}"
