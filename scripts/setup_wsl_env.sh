#!/usr/bin/env bash
set -euo pipefail

env_name="${QAI_COSMOS_ENV:-qai-cosmos-reason2}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"

if [[ ! -f "${conda_sh}" ]]; then
  echo "Miniconda activation script not found: ${conda_sh}" >&2
  exit 1
fi

source "${conda_sh}"

if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
  conda create -y -n "${env_name}" python=3.10 pip
fi

conda activate "${env_name}"
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  "qai-hub-models[qwen3-vl-4b-instruct]==0.58.0" \
  "onnxruntime-gpu==1.22.0" \
  "https://github.com/qualcomm/aimet/releases/download/2.33.0/aimet_onnx-2.33.0+cu126-cp310-abi3-manylinux_2_34_x86_64.whl" \
  -f https://download.pytorch.org/whl/torch_stable.html

# QAI Hub Models also installs the CPU ``onnxruntime`` distribution. Both
# distributions own the same ``onnxruntime/`` import package, so pip's install
# order can leave the CPU files active even though onnxruntime-gpu is present.
# Keep the CPU distribution metadata that satisfies QAIHM's dependency, but
# make the GPU wheel the final owner of the shared import package.
python -m pip install \
  --force-reinstall \
  --no-deps \
  "onnxruntime-gpu==1.22.0"

# Use a regular install so info.yaml is placed under QAIHM_MODELS_ROOT.
python -m pip install --force-reinstall --no-deps "${repo_root}"

python - <<'PY'
import importlib
import importlib.metadata

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from qai_hub_models._version import __version__ as qai_hub_models_version

module = importlib.import_module(
    "qai_hub_models.models.cosmos_reason2_2b"
)
print("qai_hub_models:", qai_hub_models_version)
print("adapter:", module.MODEL_ID)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

expected_ort_version = importlib.metadata.version("onnxruntime-gpu")
providers = ort.get_available_providers()
print("onnxruntime:", ort.__version__)
print("ORT providers:", providers)
if ort.__version__ != expected_ort_version:
    raise RuntimeError(
        "onnxruntime-gpu metadata/module mismatch: "
        f"distribution={expected_ort_version}, import={ort.__version__}"
    )
if "CUDAExecutionProvider" not in providers:
    raise RuntimeError(
        "onnxruntime-gpu is installed but CUDAExecutionProvider is unavailable"
    )

# Provider enumeration only proves that the wheel was compiled with CUDA.
# Execute one tiny graph to also catch missing CUDA/cuDNN shared libraries.
x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
z_info = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
graph = helper.make_graph(
    [helper.make_node("Add", ["x", "y"], ["z"])],
    "ort_cuda_smoke",
    [x_info, y_info],
    [z_info],
)
smoke_model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 20)],
)
smoke_model.ir_version = 10
session = ort.InferenceSession(
    smoke_model.SerializeToString(),
    providers=["CUDAExecutionProvider"],
)
active_providers = session.get_providers()
if not active_providers or active_providers[0] != "CUDAExecutionProvider":
    raise RuntimeError(f"ORT CUDA session fell back to: {active_providers}")
result = session.run(
    ["z"],
    {
        "x": np.asarray([1.0], dtype=np.float32),
        "y": np.asarray([2.0], dtype=np.float32),
    },
)[0]
np.testing.assert_allclose(result, np.asarray([3.0], dtype=np.float32))
print("ORT CUDA smoke:", result.tolist())
PY
