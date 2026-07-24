# Developer tutorial: Cosmos-Reason2-2B on the IQ-9075 EVK

## What Cosmos-Reason2-2B is—and why run it at the edge

[`Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
is NVIDIA's compact reasoning vision-language model for physical AI. It is a
2.44-billion-parameter post-training of `Qwen3-VL-2B-Instruct` that retains
Qwen3-VL's dense text transformer, vision transformer, and multimodal
architecture. Unlike a video generator, it consumes a text prompt together
with an image or video and produces text: an explanation, answer, prediction,
or proposed plan grounded in what it sees.

The specialization is physical common sense and embodied reasoning. NVIDIA
targets questions involving space, time, motion, object interactions, and
likely consequences in robotics, autonomous-driving, and smart-space scenes.
Useful developer tasks include describing an unfolding scene, locating an
event in time, identifying a safety hazard, explaining why an action failed,
and reasoning about what an embodied agent should do next. The model is
therefore an interesting component for a warehouse robot or vision agent that
must turn camera observations into a semantic decision, rather than merely
detect objects in individual frames. NVIDIA's
[Reason2 reference](https://docs.nvidia.com/cosmos/latest/reason2/reference.html)
shows both temporal video captioning and embodied-reasoning workflows.

The 2B variant is particularly attractive for edge deployment because it is
the smallest Reason2 checkpoint while preserving the same image/video-to-text
interface and physical-AI specialization. Keeping perception and reasoning
near the robot can reduce round-trip latency, avoid depending on a network
connection, keep raw camera data on the device, and make inference cost more
predictable. A Qualcomm NPU also offers a path to much better
performance-per-watt than treating the EVK CPU as a small server.

That compact label should not be mistaken for an easy mobile workload.
NVIDIA documents a 24 GB minimum for its tested BF16 GPU path, recommends
four video frames per second, and validates Reason2 on NVIDIA GPU platforms,
not Qualcomm. Fitting it on IQ-9075 consequently requires weight and
activation quantization, static tensor shapes, split decoder graphs, a
bounded context, and an explicit bridge from decoded video frames to the
vision encoder. Those constraints make this port useful beyond a single
model: it exposes the practical gap between *running* a modern video VLM on
an edge NPU and retaining enough temporal and visual fidelity for correct
physical predictions.

Finally, Reason2 is a reasoning component, not a safety-certified robot
controller. Its output is text, it can misread complex temporal scenes, and
NVIDIA calls for use-case-specific validation before deployment. This
tutorial therefore treats NPU execution, image conditioning, temporal input,
and prediction accuracy as four separate claims and tests each one
explicitly.

This tutorial brings up NVIDIA's `Cosmos-Reason2-2B` on a Dragonwing
IQ-9075 EVK reachable at `ubuntu@<EVK IP>`. The coherent legacy-Genie
baseline combines the original W4A16 vision context with four
W4-weight/FP16-activation text contexts under QAIRT 2.45. Its compatibility
transform removes the unsupported `visual_pos_masks` and
`deepstack_visual_embeds_0..2` inputs. The higher-fidelity paired-frame
experiment restores those inputs by replacing only text part 1 with a
full-interface W4/FP16 context, retaining the coherent parts 2-4, and running
the result through a pinned GenieX lower-level `PixelData` integration. A
globally-W4 vision export remains incompatible with legacy Genie, and native
MP4 ingestion remains unsupported. A CPU GGUF path is included as an
independent correctness baseline.

The NPU integration is experimental. Qualcomm AI Hub Models 0.58.0 publishes
a GenieX llama.cpp recipe for Qwen3-VL-2B, but its public Python package only
contains the full W4A16 QAIRT implementation for Qwen3-VL-4B and 8B. This
repository specializes the parameterized 4B implementation with the 2B
dimensions used by Cosmos-Reason2-2B.

The workflow deliberately follows the proven pattern in
[`eivholt/qai-nemotron`](https://github.com/eivholt/qai-nemotron): pin the
QAI Hub Models/AIMET environment, produce a dynamic-ONNX W4A16 checkpoint,
compile and link through QAI Hub, and run it with the exact matching QAIRT
tree on the EVK. The model architecture and multimodal packaging differ, but
those reproducibility rules carry over unchanged.

## Current status

Status below is current as of 2026-07-24.

| Item | Status |
|---|---|
| Adapter Python syntax, wheel contents, and dependency metadata | Verified locally |
| Adapter imports and helper-call signatures against the exact QAI Hub Models 0.58.0 wheel | Verified statically |
| 2B architecture constants against the public Qwen3-VL-2B config | Verified |
| SSH access to the IQ-9075 EVK | Verified |
| QAIRT `2.45.0.260326` gcc11.2/gcc8.2 host libraries, Genie executables, Hexagon v73 DSP files, and FastRPC access on the EVK | Verified |
| QAIRT `2.47.0.260601` as a separately installed alternative runtime | Verified; Workbench does not currently offer a matching 2.47 compiler |
| A non-Cosmos Genie context binary loading under QAIRT 2.47 on this EVK | Verified; this proves that installation, not the Cosmos port |
| Qwen3-VL support in the EVK's `llama-mtmd-cli` and presence of both Cosmos GGUF files | Verified |
| Cosmos GGUF end-to-end CPU vision inference | Verified with the existing Q4_K_M model and F16 projector |
| Official gated Cosmos checkpoint download and architecture validation | Verified; all 15 files are present |
| Cosmos W4A16 text and vision quantization | Verified for the 512-token smoke checkpoint: 8 text and 20 vision calibration samples |
| QAIRT 2.45 compatibility checkpoint | Verified; ONNX checker passes, split points are unchanged, and the four unsupported auxiliary text inputs are removed |
| Original all-W4A16 text contexts | Historical failure: all jobs succeed and execute on `QnnHtp`, but native p1→p4 drift changes the first token and Genie output is incoherent |
| Official GenieX v0.3.16 against that original all-W4A16 artifact | Evaluated; local import succeeds but reproduces that artifact's incoherent Chinese/repeated `A`, so orchestration alone does not repair it |
| Isolated original-W4A16 AR128 part agreement and forced AR1 prefill | Historical diagnostics only; isolated parts agree closely, while forced AR1 returns immediate EOS |
| All-FP16 activation checkpoint | Verified locally; 9,423 text and 934 vision activations are FP16 while all parameter encodings remain unchanged |
| All-FP16 text split QuantSim | Verified on four CUDA sessions; the bounded `Hello` smoke returns coherent text (`Hello,`) |
| All-FP16 AI Hub vision export | Failed at the QAIRT 2.45 vision link after the vision graph compiled; this is not a deployable bundle |
| W4/FP16 text contexts | Verified; all four cache-controlled links have AR128→AR1 order and legacy Genie produces coherent English |
| Globally-W4 mixed vision context | Compile `jpym3vrlp` and link `jp060e3np` succeed; direct EVK HTP execution succeeds in about 250 ms, but its external I/O is FP32 and legacy Genie rejects it |
| Final hybrid bundle | Verified end to end: original W4A16 vision target `mngx5gv5q` plus the four W4/FP16 text targets |
| Legacy Genie 1.17 text-only result | Verified coherent: 131.152 ms TTFT, 289.7711 prompt tok/s, 17.9982 generation tok/s, 2.489846 s query |
| Cosmos image-to-text through legacy Genie | Verified; default script exits 0, accepts all five image inputs, and describes Qualcomm's `dog.jpg` as a white fluffy dog on green grass |
| Native MP4/video-container input | Unsupported by legacy Genie, stock QAI Hub Models, and stock GenieX v0.3.16; the custom runner instead consumes a prepacked raw `PixelData` tensor |
| Paired video frames through legacy Genie | Functional NPU pass: two distinct frames pack into one `[1024, 1536]` temporal patch with grid `[1, 32, 32]`; both warehouse cases exit 0 |
| Full-DeepStack GenieX bundle | Verified: full-interface W4/FP16 part 1 restores `visual_pos_masks` and `deepstack_visual_embeds_0..2`; coherent W4/FP16 parts 2-4 and the original DeepStack-producing W4A16 vision context are retained |
| Full-DeepStack paired-frame NPU execution | Verified cleanly on `QnnHtp` / Hexagon v73 through the pinned GenieX lower `PixelData` API |
| Paired-frame prediction quality | No robust pass: both DeepStack free-form predictions fail their semantic rubrics; near-miss multiple choice is wrong, and the shelf result fails a shuffled-label control because it remains at option A |
| EVK deployment | Legacy hybrid and full-DeepStack GenieX artifacts use separate new directories; the old active bundle is untouched |

The accumulated-drift diagnosis applies to the original all-W4A16 text
contexts. In that artifact, isolated per-part checks reset the error at every
split, while a native chained run selects token `99590` (`答`) instead of
token `785` (`The`). Genie's handoff bytes match the direct native chain.
GenieX reproduced the same bad all-W4A16 artifact, so changing orchestration
did not repair that artifact. This finding does not apply to the later
full-interface W4/FP16 part-1 replacement.

The W4/FP16 text contexts resolve that text-path problem on the physical NPU.
The final compatibility result is deliberately hybrid: use the original
W4A16 vision context, not the globally-W4 replacement. Genie source
dequantizes its quantized `image_features` into the accumulator used by the
text graph, which is why the original vision context can feed the FP16 text
contexts. The successful image-conditioned answer, exit status, five bound
image inputs, and profile together establish end-to-end execution.

The temporal result has a narrower interpretation. The processor can combine
two distinct frames into one native Qwen3-VL temporal patch, and repeated
legacy-Genie inputs deliver that patch to the NPU contexts. The current CL512
bundle accepts one pair only. Both measured prompts execute, but their answers
fail the benchmark rubrics. The subsequent full-DeepStack GenieX run restores
the visual-mask and intermediate-vision inputs and also executes on the NPU,
but its free-form and controlled multiple-choice results still do not
establish a correct prediction. See
[`docs/video_npu.md`](video_npu.md)
and the
[`paired-frame evidence report`](evidence/iq9075_video_smoke_r1.json).

No earlier independent public result was found for the exact
`nvidia/Cosmos-Reason2-2B` checkpoint on IQ-9075. NVIDIA's published hardware
list does not include Qualcomm, while Qualcomm's IQ-9075 result is for the
related
[Qwen3-VL-2B-Instruct GGUF](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct?runtime=geniex_qairt%2Cgeniex_llamacpp)
model. This tutorial documents project evidence for an experimental port, not
vendor certification.

## Architecture being deployed

The adapter expects:

| Property | Value |
|---|---:|
| Model type | `qwen3_vl` |
| Text layers | 28 |
| Hidden/intermediate size | 2048 / 6144 |
| Attention/KV heads | 16 / 8 |
| Head dimension | 128 |
| Vocabulary size | 151936 |
| Text partitions | 4 × 7 layers |
| Vision depth/width | 24 / 1024 |
| Vision output width | 2048 |
| Vision heads | 16 |
| Patch/merge size | 16 / 2 |
| Deep-stack injection layers | 3 |
| Default 512 × 512 visual tokens | 256 |
| Temporal patch | 2 frames, grid `[1, 32, 32]`, 256 tokens |
| Deployment precision | Original W4A16 vision + W4/FP16 text |

In practical terms, the model has two cooperating towers. The 24-layer vision
transformer repeatedly turns small image regions into increasingly contextual
features; its width of 1024 is the size of the feature vector carried for
each region inside that tower. A 512 × 512 input is divided into 16 × 16
patches and spatially merged in 2 × 2 groups, leaving 256 visual tokens. A
projector maps those features to width 2048 so they can join the text-token
stream. The 28 text layers then refine that combined sequence one block at a
time: attention relates tokens across the prompt and visual scene, while each
block's 6144-wide feed-forward section transforms the information at each
position. Sixteen query heads let attention examine several relationships in
parallel; sharing eight key/value heads reduces cache size and memory traffic.
The three deep-stack paths inject intermediate vision features into selected
text layers instead of relying only on the initial projected tokens. Finally,
splitting the 28 text layers into four groups of seven is a deployment choice,
not four separately trained models: it makes compilation and NPU memory
management tractable, but the output of each partition must remain
numerically sound because it feeds the next one.

The NPU pipeline is:

```text
nvidia/Cosmos-Reason2-2B
  -> Qwen3-VL-2B adapter
  -> AIMET-ONNX W4A16 text + vision calibration
  |-> original W4A16 vision link, including three DeepStack outputs
  |-> W4/FP16 text links
  |    |-> legacy compatibility p1-p4, auxiliary inputs removed
  |    `-> full-interface replacement p1, auxiliary inputs restored
  -> assemble either runtime-specific five-context bundle
  |-> legacy Genie 1.17: compatibility p1-p4
  `-> pinned GenieX PixelData runner: full p1 + coherent p2-p4
      -> matching QAIRT 2.45 / QnnHtp on Hexagon v73
```

## Prerequisites

Host:

- Windows with WSL2 Ubuntu.
- This repository available from WSL, for example at
  `/mnt/c/path/to/qai-physics-reasoning`.
- Miniconda at `~/miniconda3`.
- Python 3.10 in the project environment. QAI Hub Models supports newer
  Python versions, but the selected AIMET wheel is a CPython 3.10 build.
- A CUDA-capable NVIDIA GPU. Quantization is the demanding phase; start with
  the reduced smoke calibration below.
- At least 40 GB of available host memory and ample temporary disk space.
- A Qualcomm AI Hub Workbench account and API token.
- A Hugging Face account that has accepted the gated NVIDIA model terms.
- Docker for the pinned GenieX cross-toolchain build.
- A recursive `qualcomm/GenieX` checkout at commit
  `ab9e904fe21a01673840495309f3471dd6980d10`; the build script verifies its
  `geniex-qairt` submodule revision before compiling.

Target:

- IQ-9075 EVK reachable as `ubuntu@<EVK IP>`.
- An SSH key accepted by the EVK; pass its path to the deployment script.
- QAIRT `2.45.0.260326` at `/opt/qairt/2.45.0.260326`. This is also
  `/opt/qairt/current` and is the primary path because Workbench can compile
  matching 2.45 artifacts.
- QAIRT `2.47.0.260601` at `/home/ubuntu/qairt-2.47.0.260601` as an installed
  alternative that must not be mixed with 2.45 libraries or artifacts.
- Official stable GenieX v0.3.16 at `/home/ubuntu/geniex-cosmos` as an
  independent orchestration comparison. Its bundled runtime is QAIRT
  `2.45.0.260326`. The custom raw runner does not modify or reuse this
  directory; its deploy step copies a separate pinned build.

The EVK had about 34 GiB RAM, no swap, and 42 GiB free disk when inspected.
Recheck before copying a new bundle:

```bash
EVK_IP="<EVK IP>"
EVK_USER=ubuntu
EVK_SSH_KEY=/mnt/c/path/to/evk_ssh_key

ssh -i "$EVK_SSH_KEY" "${EVK_USER}@${EVK_IP}" \
  'free -h; df -h /home/ubuntu'
```

## 1. Create and verify the host environment

In WSL:

```bash
COSMOS_REPO=/mnt/c/path/to/qai-physics-reasoning
cd "$COSMOS_REPO"
bash scripts/setup_wsl_env.sh
```

The setup script creates the `qai-cosmos-reason2` Conda environment and
installs:

- `qai-hub-models==0.58.0`
- the Qwen3-VL dependencies
- `onnxruntime-gpu==1.22.0`
- AIMET-ONNX 2.33.0 for CUDA 12.6
- this repository's overlay package

QAI Hub Models also depends on the CPU `onnxruntime` distribution. The CPU
and GPU wheels install the same Python package, so their install order
matters. The setup script force-reinstalls `onnxruntime-gpu` last while
retaining the CPU distribution metadata needed by QAI Hub Models. It then
runs a one-element ONNX `Add` graph through `CUDAExecutionProvider`; merely
seeing the provider name is not considered sufficient because CUDA or cuDNN
shared-library errors can appear only when a session is created.

Activate it and run lightweight checks:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qai-cosmos-reason2

python - <<'PY'
import importlib.metadata
import onnxruntime as ort
import torch
from qai_hub_models.models import cosmos_reason2_2b

print("qai-hub-models:", importlib.metadata.version("qai-hub-models"))
print("adapter:", cosmos_reason2_2b.MODEL_ID)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("onnxruntime:", ort.__version__)
print("ORT providers:", ort.get_available_providers())
PY

python -m qai_hub_models.models.cosmos_reason2_2b.quantize --help
python -m qai_hub_models.models.cosmos_reason2_2b.export --help
```

Expected:

```text
qai-hub-models: 0.58.0
adapter: cosmos_reason2_2b
CUDA: True
onnxruntime: 1.22.0
ORT providers: [..., 'CUDAExecutionProvider', ...]
```

The setup command itself must also end with `ORT CUDA smoke: [3.0]`. If the
imported ONNX Runtime version is 1.22.0 or the CUDA provider is absent, rerun
`scripts/setup_wsl_env.sh` before quantization; otherwise calibration can
silently become a much slower CPU workload.

The `--help` commands should complete without downloading a model. If the
adapter cannot be imported, confirm that `pip show qai-cosmos-reason2` points
at the active Conda environment rather than the Windows Python installation.

## 2. Configure credentials without committing tokens

### Hugging Face

Open the model page in a browser, sign in, accept the NVIDIA Open Model
License, and agree to share the requested account information:

<https://huggingface.co/nvidia/Cosmos-Reason2-2B>

Then authenticate interactively in WSL:

```bash
hf auth login
hf auth whoami
```

Do not put a Hugging Face token in a script, README, command-line example, or
Git-tracked `.env` file.

The successful bring-up used a process-scoped token and left no persistent
Hugging Face token file. If you prefer that pattern, skip `hf auth login` and
provide `HF_TOKEN` only to the download command as shown in Step 4.

An earlier attempt returned HTTP 401. If that happens, verify that the
currently selected account accepted the gate. If a token may have been
exposed or is no longer trusted, revoke or rotate it in Hugging Face
settings, then replace the local credential:

```bash
hf auth logout
hf auth login
hf auth whoami
```

### Qualcomm AI Hub Workbench

Get a token from `Account -> Settings -> API Token` at
<https://workbench.aihub.qualcomm.com/>. Read it without putting the token
literal in shell history:

```bash
read -rsp "QAI Hub API token: " QAIHUB_TOKEN
printf '\n'
qai-hub configure --api_token "$QAIHUB_TOKEN"
unset QAIHUB_TOKEN
```

Verify that the target is visible:

```bash
python - <<'PY'
import qai_hub as hub

for device in hub.get_devices(name="Dragonwing IQ-9075 EVK"):
    print(device)
PY
```

The export scripts use the device name `Dragonwing IQ-9075 EVK` and device OS
`1.7`.

## 3. Establish the CPU GGUF baseline

The CPU baseline is deliberately separate from the NPU pipeline. It answers
"Can this checkpoint, projector, tokenizer, and prompt produce a plausible
response?" It does not validate the QAI adapter, W4A16 quantization, or NPU
execution.

The board already contains:

```text
/home/ubuntu/models/cosmos_reason2_2b_gguf/Cosmos-Reason2-2B-Q4_K_M.gguf
/home/ubuntu/models/cosmos_reason2_2b_gguf/mmproj-Cosmos-Reason2-2B-F16.gguf
```

The files were 1,282,440,864 and 819,395,424 bytes respectively when
inspected. They are community conversions from
`apolo13x/Cosmos-Reason2-2B-GGUF`, not the official NVIDIA checkpoint and not
inputs to the NPU quantization flow.

For a fresh installation, download only the two required files in WSL:

```bash
COSMOS_GGUF="$HOME/models/cosmos_reason2_2b_gguf"
mkdir -p "$COSMOS_GGUF"

hf download apolo13x/Cosmos-Reason2-2B-GGUF \
  Cosmos-Reason2-2B-Q4_K_M.gguf \
  mmproj-Cosmos-Reason2-2B-F16.gguf \
  --local-dir "$COSMOS_GGUF"

sha256sum \
  "$COSMOS_GGUF/Cosmos-Reason2-2B-Q4_K_M.gguf" \
  "$COSMOS_GGUF/mmproj-Cosmos-Reason2-2B-F16.gguf"
```

The hashes recorded and verified during this bring-up are:

```text
3ed011641891e81fe654328071de251718832e7deafef579033485d33082e4fd  Cosmos-Reason2-2B-Q4_K_M.gguf
8d3284c340d6a9c9237d56f2fc42f2df50d3761f539f3f01be964cefb0b73916  mmproj-Cosmos-Reason2-2B-F16.gguf
```

Both on-board files begin with the `GGUF` magic bytes and matched these hashes
during the 2026-07-24 check. The verified checksum file is stored at:

```text
/home/ubuntu/models/cosmos_reason2_2b_gguf/SHA256SUMS
```

Review the community artifact's provenance and the NVIDIA license before
redistribution. This repository's `NOTICE` file contains the required NVIDIA
attribution and does not redistribute weights.

Copy a fresh baseline to the EVK if needed:

```bash
EVK_IP="<EVK IP>"
EVK_USER=ubuntu
EVK_SSH_KEY=/mnt/c/path/to/evk_ssh_key
EVK_TARGET="${EVK_USER}@${EVK_IP}"

ssh -i "$EVK_SSH_KEY" "$EVK_TARGET" \
  'mkdir -p /home/ubuntu/models/cosmos_reason2_2b_gguf'

scp -i "$EVK_SSH_KEY" \
  "$COSMOS_GGUF/Cosmos-Reason2-2B-Q4_K_M.gguf" \
  "$COSMOS_GGUF/mmproj-Cosmos-Reason2-2B-F16.gguf" \
  "${EVK_TARGET}:/home/ubuntu/models/cosmos_reason2_2b_gguf/"

scp -i "$EVK_SSH_KEY" \
  "$COSMOS_REPO/scripts/run_evk_cpu_baseline.sh" \
  "${EVK_TARGET}:/home/ubuntu/"
```

The EVK's existing llama.cpp checkout is commit
`0dc74e332edee2616e4d8d9ab3b68dfc340fc14a` from 2026-07-17. Its
`llama-mtmd-cli` is executable and the source tree contains both core and
multimodal Qwen3-VL implementations.

A bounded run of this repository's CPU script with context 4096,
`LLAMA_PREDICT=32`, and llama.cpp's `tools/mtmd/test-1.jpeg` exited
successfully in 18.51 seconds. Vision batch encoding took 6.712 seconds, peak
RSS was 2,804,628 KiB, and the model correctly described the image as a New
York Times clipping about the moon landing.

Run a text-plus-image baseline on the EVK:

```bash
ssh -i "$EVK_SSH_KEY" "$EVK_TARGET"

chmod +x /home/ubuntu/run_evk_cpu_baseline.sh

LLAMA_CONTEXT_LENGTH=4096 \
LLAMA_PREDICT=32 \
bash /home/ubuntu/run_evk_cpu_baseline.sh \
  /home/ubuntu/models/cosmos_reason2_2b_gguf/Cosmos-Reason2-2B-Q4_K_M.gguf \
  /home/ubuntu/models/cosmos_reason2_2b_gguf/mmproj-Cosmos-Reason2-2B-F16.gguf \
  /absolute/path/to/test-image.jpg \
  2>&1 | tee /home/ubuntu/cosmos_reason2_cpu_smoke.log
```

Expected output includes recognition of the `qwen3vl` architecture, projector
loading, token generation, and a physically plausible answer about the image.
Save the log and the exact test image. If the binary reports an unknown model
architecture, update llama.cpp to a revision with Qwen3-VL multimodal support
and rebuild `llama-mtmd-cli`. The current CLI also warns that Qwen-VL
grounding tasks may benefit from `--image-min-tokens 1024`; that setting was
not needed for the caption smoke and should be evaluated separately for
localization accuracy by adding `LLAMA_IMAGE_MIN_TOKENS=1024` to the command.

## 4. Download and validate the official checkpoint

The official checkpoint is gated. Do not use the GGUF files as quantization
inputs.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qai-cosmos-reason2
cd "$COSMOS_REPO"

COSMOS_OFFICIAL="$HOME/models/Cosmos-Reason2-2B"
bash scripts/download_checkpoint.sh "$COSMOS_OFFICIAL"
```

For a non-persistent credential, use a hidden prompt instead:

```bash
read -rsp "Hugging Face token: " COSMOS_HF_TOKEN
printf '\n'
HF_TOKEN="$COSMOS_HF_TOKEN" \
  bash scripts/download_checkpoint.sh "$COSMOS_OFFICIAL"
unset COSMOS_HF_TOKEN
```

The script uses:

```text
hf download nvidia/Cosmos-Reason2-2B --local-dir <directory>
```

and then runs the repository validator. The final line should resemble:

```text
Compatible Qwen3-VL-2B checkpoint: .../Cosmos-Reason2-2B/config.json
```

The verified download at `$HOME/models/Cosmos-Reason2-2B` contains 15
files, uses about 4.6 GiB, and has a 4,877,470,304-byte
`model.safetensors`. `scripts/validate_model_config.py` passed against it.

Run the validator explicitly when diagnosing a manually downloaded
checkpoint:

```bash
python scripts/validate_model_config.py "$COSMOS_OFFICIAL"
```

At minimum, confirm these files are present:

```bash
test -f "$COSMOS_OFFICIAL/config.json"
test -f "$COSMOS_OFFICIAL/model.safetensors"
test -f "$COSMOS_OFFICIAL/tokenizer.json"
test -f "$COSMOS_OFFICIAL/preprocessor_config.json"
```

The validator rejects a checkpoint whose model type or text/vision dimensions
do not match the adapter. Do not bypass it by editing `config.json`.

## 5. Quantize a small W4A16 smoke checkpoint

Start with context length 512 and fewer calibration samples. This reduces
bring-up cost but is not a final quality calibration:

```bash
COSMOS_WORK="$HOME/cosmos-reason2-iq9"
COSMOS_W4A16="$COSMOS_WORK/checkpoints/w4a16-cl512"
mkdir -p "$COSMOS_WORK/checkpoints"

CONTEXT_LENGTH=512 \
CALIBRATION_SEQUENCE_LENGTH=128 \
NUM_SAMPLES=8 \
VEG_NUM_SAMPLES=20 \
IMAGE_HEIGHT=512 \
IMAGE_WIDTH=512 \
bash scripts/quantize_wsl.sh \
  "$COSMOS_OFFICIAL" \
  "$COSMOS_W4A16"
```

The operation calibrates the text decoder and vision encoder separately. Keep
`nvidia-smi` open in another terminal and make sure the process is using the
intended GPU.

The measured smoke run used exactly those parameters and is retained at:

```text
$HOME/cosmos-reason2-iq9/checkpoints/w4a16-cl512-smoke-r4
```

Its `args.json` records context length 512, calibration sequence length 128,
8 text samples, 20 vision samples, and a 512 × 512 image size. Use the
generated `args.json` rather than memory as the provenance record for a new
attempt.

On success, the output directory must contain:

```text
model_dynamic.onnx
model.data
model.encodings
embedding_weights.raw
vision_encoder.onnx
vision_encoder.encodings
config.json
tokenizer.json
source_checkpoint.json
NOTICE
```

Depending on the ONNX exporter version, the vision weights are either stored
inline in `vision_encoder.onnx` (the QAI Hub Models 0.58 / Torch 2.10 behavior)
or in an additional external-data file such as `vision_encoder.data`. The
external file is optional when the ONNX model has no external-data references.

Check it:

```bash
for file in \
  model_dynamic.onnx model.data model.encodings embedding_weights.raw \
  vision_encoder.onnx vision_encoder.encodings \
  config.json tokenizer.json source_checkpoint.json NOTICE
do
  test -f "$COSMOS_W4A16/$file" || echo "MISSING: $file"
done
```

`source_checkpoint.json` is a small provenance marker, not another copy of
the weights. It records the absolute path of the BF16 Hugging Face snapshot.
QAI Hub Models still reads that snapshot's configuration, tokenizer,
preprocessor, and safetensors while it constructs the export model, so keep
the source checkpoint accessible until export finishes.

Only after the 512-context pipeline runs end to end should you increase
calibration:

```bash
CONTEXT_LENGTH=2048 \
CALIBRATION_SEQUENCE_LENGTH=128 \
NUM_SAMPLES=20 \
VEG_NUM_SAMPLES=100 \
bash scripts/quantize_wsl.sh \
  "$COSMOS_OFFICIAL" \
  "$COSMOS_WORK/checkpoints/w4a16-cl2048"
```

## 6. Create the QAIRT 2.45 compatibility checkpoint

Legacy Genie 1.17 in QAIRT 2.45 cannot bind the Qwen3-VL auxiliary decoder inputs
`visual_pos_masks` and `deepstack_visual_embeds_0..2`. Create a derived
checkpoint that replaces the three optional deep-stack additions with
shape-preserving zeros, prunes their now-dead branches, and removes those four
runtime inputs:

```bash
COSMOS_W4A16="$COSMOS_WORK/checkpoints/w4a16-cl512-smoke-r4"
COSMOS_QAIRT245="$COSMOS_WORK/checkpoints/w4a16-cl512-smoke-r4-qairt245-r3"

python scripts/make_qairt245_compat_checkpoint.py \
  "$COSMOS_W4A16" \
  "$COSMOS_QAIRT245" \
  --num-visual-tokens 256
```

The destination must not already exist. The script validates the source
shapes and encodings, runs reverse dead-code elimination, checks the resulting
ONNX model, and writes `qairt_245_compat.json`.

For the measured checkpoint:

```text
source model_dynamic.onnx SHA-256:
4e609708b1827e8202e4ede338e271a277e8a5fea3f1e4012187d47e6df83f69

compatibility model_dynamic.onnx SHA-256:
82fa3e6f2dacc7b50826090834a4af57963efed67585ecfac3374dee3a29d429
```

The transform leaves all 28 layer split points unchanged and preserves the
primary `image_features` input. It deliberately disables the three auxiliary
deep-stack image injections, so it is a text-first compatibility path and
does not prove full-fidelity VLM behavior. Use it for the legacy-Genie
baseline only. The full-DeepStack experiment later in this tutorial derives
its replacement part 1 from the unpruned W4A16 checkpoint, where
`visual_pos_masks` and `deepstack_visual_embeds_0..2` are still graph inputs.

### Create the W4/FP16 text export source

The original W4A16 bundle executes on HTP, but numerical drift compounds
through its native decoder chain. The successful text path reuses the
calibrated W4 text weights while replacing all text activation encodings
with AIMET's canonical FP16 form.

The first derivative also converted every vision activation to FP16. Its
bounded text-only host QuantSim was coherent, and the vision graph compiled,
but QAIRT 2.45 link job
[`jgo86ezqp`](https://workbench.aihub.qualcomm.com/jobs/jgo86ezqp/) failed
with exit code 14 at the vision patch-embedding path. That all-FP16
checkpoint is retained as failure history, not as a deployment input.

Create a checkpoint whose text encodings are W4/FP16 while preserving
`vision_encoder.encodings` byte-for-byte:

```bash
COSMOS_W4_TEXT="$COSMOS_WORK/checkpoints/w4-fp16-text-w4a16-vision-cl512-smoke-r4-qairt245-r1"

python scripts/make_w4_fp16_checkpoint.py \
  --source "$COSMOS_QAIRT245" \
  --destination "$COSMOS_W4_TEXT" \
  --keep-vision-w4a16
```

Despite the preserved vision encodings, this directory is the source for
`part1_of_4` through `part4_of_4` only. Do not export its `vision_encoder` for
the recommended legacy-Genie bundle. A later globally-W4 vision experiment
proved that compile/link success and direct QNN execution do not imply legacy
Genie input-type compatibility.

Changing only `args.json` is not sufficient. QAI Hub Models first configures
`Precision.w4` activation quantizers as FP16, but AIMET's subsequent
non-strict encoding load adopts the data types stored in the encoding files.
The conversion script therefore rewrites every text activation entry,
preserves every parameter entry exactly, verifies required bundle metadata,
and writes `w4_fp16.json`. With `--keep-vision-w4a16`, it also preserves the
vision activation file exactly and breaks its destination hard link so later
changes cannot modify the source checkpoint. Large unchanged model files are
hard-linked when possible.

The two measured derivatives are deliberately distinct:

```text
all-FP16 failed experiment:
  text activations:     9,423 FP16
  vision activations:     934 FP16
  w4_fp16.json SHA-256:
2c18783f4312191969722091a0c18b147dbb53e7796f34f05ee0c740b0e9bc26

W4/FP16 text-component export source:
  text activations:     9,423 FP16
  text parameters:      1,738 unchanged (including INT4 weights)
  vision activations:     934 INT16, byte-for-byte preserved
  vision parameters:      205 unchanged (including INT8 weights)
  w4_fp16.json SHA-256:
a0727ab760e1e8d972a2ed58cf1e4438a698b3dacc8713456fed6c1b17883bd9
```

Before spending AI Hub jobs, run a bounded split-QuantSim check:

```bash
python -m qai_hub_models.models.cosmos_reason2_2b.demo \
  --checkpoint "$COSMOS_W4_TEXT" \
  --sequence-length 1 128 \
  --context-length 512 \
  --max-output-tokens 2 \
  --prompt Hello
```

The measured run against the all-FP16 derivative creates four CUDA QuantSim
sessions and emits `Hello,`. The W4/FP16 text checkpoint has the same text
`model.encodings` SHA-256,
`81ffb303031ef0fa9f76198431c853d45d1398ae884ee08241fc6d699cfdbec0`,
so that result validates its text side as well. The physical-NPU proof comes
from the four linked text contexts documented below.

## 7. Export vision and text separately, then assemble

If the BF16 checkpoint has moved since quantization, override the stale path
without editing the generated marker:

```bash
export COSMOS_SOURCE_CHECKPOINT="$HOME/models/Cosmos-Reason2-2B"
```

The adapter validates the resolved directory before submitting cloud work.
This step uploads ONNX graphs, weights, encodings, and bundle assets to
Qualcomm AI Hub Workbench. Confirm that the model license and the
organization's data policy permit that upload.

Export the original W4A16 vision encoder by itself:

```bash
COSMOS_VISION_EXPORT="$COSMOS_WORK/exports/w4a16-vision-only-qairt245"

python -m qai_hub_models.models.cosmos_reason2_2b.export \
  --checkpoint "$COSMOS_QAIRT245" \
  --target-runtime geniex_qairt \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --skip-profiling \
  --skip-downloading \
  --output-dir "$COSMOS_VISION_EXPORT" \
  --components vision_encoder
```

Then export only the four W4/FP16 text parts:

```bash
COSMOS_TEXT_EXPORT="$COSMOS_WORK/exports/w4-fp16-text-only-qairt245"

python -m qai_hub_models.models.cosmos_reason2_2b.export \
  --checkpoint "$COSMOS_W4_TEXT" \
  --target-runtime geniex_qairt \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --skip-profiling \
  --skip-downloading \
  --output-dir "$COSMOS_TEXT_EXPORT" \
  --components part1_of_4 part2_of_4 part3_of_4 part4_of_4
```

The 512-token smoke matrix links prompt/decode graphs with sequence lengths
128 and 1. Use a new output and upload identity for every attempt, and record
the Workbench URLs and returned target-model IDs.

### Proven linked contexts

The final text contexts are:

| Component | Link job | Target model | SHA-256 prefix | Graph order |
|---|---|---|---|---|
| Part 1 | [`jprw31875`](https://workbench.aihub.qualcomm.com/jobs/jprw31875/) | `mno2p6jvq` | `4fa761…` | AR128 → AR1 |
| Part 2 | [`jgjqnkzv5`](https://workbench.aihub.qualcomm.com/jobs/jgjqnkzv5/) | `mm60gk9vm` | `f797…` | AR128 → AR1 |
| Part 3 | [`jpeym4eo5`](https://workbench.aihub.qualcomm.com/jobs/jpeym4eo5/) | `mq3lg676q` | `4c7…` | AR128 → AR1 |
| Part 4 | [`jgzndvoog`](https://workbench.aihub.qualcomm.com/jobs/jgzndvoog/) | `mno2p69vq` | `6d37…` | AR128 → AR1 |

Part 1 is the cache-busted link. Do not reuse a stale part-1 upload or target
that merely has the same logical compile options: require link `jprw31875`,
target `mno2p6jvq`, checksum prefix `4fa761…`, and inspect that its graphs are
AR128 then AR1. If part 1 must be rebuilt, force a new upload/model identity
before linking. A cached context with the wrong order can otherwise be
silently reintroduced.

The final vision context is the original W4A16 artifact:

| Component | Compile job | Link job | Target model | SHA-256 prefix |
|---|---|---|---|---|
| Original W4A16 vision | [`jgo861ddp`](https://workbench.aihub.qualcomm.com/jobs/jgo861ddp/) | [`j579rjnrg`](https://workbench.aihub.qualcomm.com/jobs/j579rjnrg/) | `mngx5gv5q` | `872e4e…` |

Two vision alternatives were evaluated and are not the recommended context:

- The all-FP16 vision compile
  [`jp2ey3xrp`](https://workbench.aihub.qualcomm.com/jobs/jp2ey3xrp/)
  succeeded, but link
  [`jgo86ezqp`](https://workbench.aihub.qualcomm.com/jobs/jgo86ezqp/)
  failed with exit code 14 in the patch-embedding path.
- The globally-W4 mixed vision compile
  [`jpym3vrlp`](https://workbench.aihub.qualcomm.com/jobs/jpym3vrlp/)
  and link
  [`jp060e3np`](https://workbench.aihub.qualcomm.com/jobs/jp060e3np/)
  succeeded. Target `mnzgp1ydq` has SHA-256 prefix `b494c5…`, all external
  inputs and outputs are FP32, and direct `qnn-net-run` on EVK HTP succeeds
  with roughly 250 ms graph execution. Legacy Genie 1.17 nevertheless rejects
  it with `Unsupported input tensor full_attention_mask dtype
  QNN_DATATYPE_FLOAT_32`. Direct-QNN success therefore does not make this
  globally-W4 vision context legacy-Genie compatible.

### Download and assemble the hybrid

Download the five target models above into an empty context directory. With
the QAI Hub Python client configured, one reproducible form is:

```bash
export COSMOS_CONTEXTS="$COSMOS_WORK/exports/proven-hybrid-contexts"
mkdir -p "$COSMOS_CONTEXTS"

python - <<'PY'
import os
from pathlib import Path
import qai_hub as hub

out = Path(os.environ["COSMOS_CONTEXTS"])
targets = {
    "vision_encoder.bin": "mngx5gv5q",
    "part1_of_4.bin": "mno2p6jvq",
    "part2_of_4.bin": "mm60gk9vm",
    "part3_of_4.bin": "mq3lg676q",
    "part4_of_4.bin": "mno2p69vq",
}
for filename, model_id in targets.items():
    hub.get_model(model_id).download(str(out / filename))
PY
```

Use a complete adapter-generated QAI Hub Models release zip as the scaffold
for configs, tokenizer, embeddings, scripts, and sample inputs. Replace all
five context binaries explicitly:

```bash
COSMOS_SCAFFOLD="$COSMOS_WORK/exports/scaffold/cosmos_reason2_2b-geniex_qairt.zip"
COSMOS_FINAL="$COSMOS_WORK/exports/cosmos_reason2_2b-w4a16-vision-w4-fp16-text-qairt245.zip"

python scripts/finalize_qairt245_bundle.py \
  "$COSMOS_SCAFFOLD" \
  "$COSMOS_FINAL" \
  --compat-manifest "$COSMOS_QAIRT245/qairt_245_compat.json" \
  --replacement-context "vision_encoder.bin=$COSMOS_CONTEXTS/vision_encoder.bin" \
  --replacement-context "part1_of_4.bin=$COSMOS_CONTEXTS/part1_of_4.bin" \
  --replacement-context "part2_of_4.bin=$COSMOS_CONTEXTS/part2_of_4.bin" \
  --replacement-context "part3_of_4.bin=$COSMOS_CONTEXTS/part3_of_4.bin" \
  --replacement-context "part4_of_4.bin=$COSMOS_CONTEXTS/part4_of_4.bin"

sha256sum "$COSMOS_CONTEXTS"/*.bin "$COSMOS_FINAL"
wslpath -w "$COSMOS_FINAL"
```

Do not substitute the globally-W4 `mnzgp1ydq` vision target in this command.
The supported legacy-Genie assembly is `mngx5gv5q` vision plus the four text
targets in the table.

A QAI Hub Models 0.58 Qwen3-VL QAIRT bundle should contain all of:

- Four text-part `.bin` context binaries and one vision-encoder `.bin`.
- `embedding_weights.raw`.
- Tokenizer and model configuration files.
- `metadata.json`.
- `htp_backend_ext_config.json`.
- `genie_config.json`.
- `text-generator.json`.
- `img-enc-htp.json`.
- `text-encoder.json`.
- `genie-app-script.txt`.
- `qairt_245_compat.json`.
- `sample_inputs/*.raw`.
- `sample_inputs/prompt_prefix.txt`.
- `sample_inputs/prompt_suffix.txt`.

Do not deploy if any referenced file is missing. The deployment script
extracts and validates the archive again before copying it.

After the 512-token hybrid passes, repeat both component-only export commands
with `--full-context-matrix` for contexts 512, 1024, 2048, and 4096. Continue
to export vision from the compatibility W4A16 checkpoint and text only from
the W4/FP16 checkpoint; do not run a single whole-model export from the
globally-W4 derivative.

QAI Hub Models 0.58 selects QAIRT 2.45 for `geniex_qairt`. Do not select a
newer Workbench compiler unless the exact matching runtime is first installed
on the EVK, and never mix the separately installed QAIRT 2.47 libraries with
these QAIRT 2.45 artifacts.

## 8. Deploy to the EVK

From Windows PowerShell:

```powershell
Set-Location C:\path\to\qai-physics-reasoning

$BundlePath = "\\wsl.localhost\Ubuntu\home\<WSL user>\cosmos-reason2-iq9\exports\cosmos_reason2_2b-w4a16-vision-w4-fp16-text-qairt245.zip"
$EvkIp = "<EVK IP>"
$IdentityFile = "C:\path\to\evk_ssh_key"

.\scripts\deploy_evk.ps1 `
  -BundlePath $BundlePath `
  -EvkIp $EvkIp `
  -EvkUser "ubuntu" `
  -IdentityFile $IdentityFile `
  -RemoteDirectory "/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1"
```

For another attempt, replace this measured path with the archive printed by
`wslpath -w "$COSMOS_FINAL"` in the previous step. The script also accepts
an already-extracted bundle directory or a directory containing exactly one
zip, but passing the exact archive is least ambiguous.

The canonical proven directory is the clean hybrid release
`/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1`. The script extracts and
validates locally, uploads through a unique temporary path, and activates only
that explicit directory. The older active bundle at
`/home/ubuntu/cosmos_reason2_2b_qairt` is not replaced or renamed.

Expected:

```text
Deployed to ubuntu@<EVK IP>:/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1
```

Return to WSL and verify the deployment without running it:

```bash
EVK_IP="<EVK IP>"
EVK_USER=ubuntu
EVK_SSH_KEY=/mnt/c/path/to/evk_ssh_key
EVK_TARGET="${EVK_USER}@${EVK_IP}"

ssh -i "$EVK_SSH_KEY" "$EVK_TARGET" \
  'find /home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1 -maxdepth 3 -type f | sort'
```

Open one interactive target shell and keep it open for Steps 9–11. All
commands in those steps run on the EVK, and the environment exports in Step 9
must remain in that shell:

```bash
ssh -i "$EVK_SSH_KEY" "$EVK_TARGET"
```

## 9. Match the bundle and select QAIRT 2.45 explicitly

First inspect the deployed bundle's recorded tool version:

```bash
python3 - <<'PY'
import json
from pathlib import Path

metadata = json.loads(
    Path("/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1/metadata.json").read_text()
)
print(metadata.get("tool_versions", {}).get("qairt", "not recorded"))
compat = json.loads(
    Path("/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1/qairt_245_compat.json").read_text()
)
print(compat["target_runtime"])
print(compat["removed_runtime_inputs"])
PY
```

If `tool_versions.qairt` is absent, use the Workbench compile-job framework
record as the source of truth. Continue below only for a 2.45 artifact.

`/opt/qairt/current` resolves to `/opt/qairt/2.45.0.260326`, but use the exact
path in captured commands for reproducibility:

```bash
QAIRT_HOME=/opt/qairt/2.45.0.260326
QAIRT_TARGET=aarch64-oe-linux-gcc11.2

export PATH="$QAIRT_HOME/bin/$QAIRT_TARGET:$QAIRT_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

COSMOS_LD_PATH="$QAIRT_HOME/lib/$QAIRT_TARGET"
if [[ -d "$QAIRT_HOME/lib/aarch64-oe-linux-gcc8.2" ]]; then
  COSMOS_LD_PATH="$COSMOS_LD_PATH:$QAIRT_HOME/lib/aarch64-oe-linux-gcc8.2"
fi
COSMOS_LD_PATH="$COSMOS_LD_PATH:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu"
export LD_LIBRARY_PATH="$COSMOS_LD_PATH"

export ADSP_LIBRARY_PATH="$QAIRT_HOME/lib/hexagon-v73/unsigned"
```

This exact 2.45 installation has the primary
`aarch64-oe-linux-gcc11.2` directory and the legacy
`lib/aarch64-oe-linux-gcc8.2` fallback. Keep gcc11.2 first. The separately
installed 2.47 tree has gcc11.2 but no gcc8.2 directory. Do not append an
inherited `LD_LIBRARY_PATH`: it can mix libraries from two QAIRT releases.

Confirm the selected runtime and FastRPC access:

```bash
readlink -f /opt/qairt/current
test -x "$QAIRT_HOME/bin/$QAIRT_TARGET/genie-t2t-run"
test -x "$QAIRT_HOME/bin/$QAIRT_TARGET/genie-app"
test -d "$QAIRT_HOME/lib/hexagon-v73/unsigned"
ldd "$QAIRT_HOME/bin/$QAIRT_TARGET/genie-t2t-run" | grep 'not found' || true
id -nG | tr ' ' '\n' | grep -x fastrpc
test -r /dev/fastrpc-cdsp && test -w /dev/fastrpc-cdsp
```

The `ubuntu` user is in the `fastrpc` group and `/dev/fastrpc-cdsp` was
readable/writable during the verified board check.

Capture exact deployed hashes before running:

```bash
cd /home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1
sha256sum \
  embedding_weights.raw vision_encoder.bin \
  part1_of_4.bin part2_of_4.bin part3_of_4.bin part4_of_4.bin
```

The measured hybrid reports these exact hashes where retained and prefixes
where the provenance record intentionally abbreviates them:

```text
466f2a37c2fb3a83a0792b0d6707202d86359712052a78e79c695869c9b89958  embedding_weights.raw
872e4e951feb448f1f60b694b5ec1113f6ecd4788608a0bd4c40579e0a855d60  vision_encoder.bin
4fa761…  part1_of_4.bin
f797…    part2_of_4.bin
4c7…     part3_of_4.bin
6d37…    part4_of_4.bin
```

Inspect the linked graph contract with the runtime-matched utility:

```bash
QNN_CONTEXT_UTILITY="$QAIRT_HOME/bin/$QAIRT_TARGET/qnn-context-binary-utility"
for context in part1_of_4.bin part2_of_4.bin part3_of_4.bin part4_of_4.bin
do
  printf '%s\n' "$context"
  "$QNN_CONTEXT_UTILITY" \
    --context_binary="$context" \
    --json_file=/dev/stdout 2>/dev/null |
    grep '"graphName"'
done
```

Each final text context contains two graphs in the order AR128, then AR1. Keep
this as a structural invariant. Part 1 is especially important: the proven
cache-busted artifact is link `jprw31875`, target `mno2p6jvq`, and checksum
prefix `4fa761…`. This order warning protects the successful hybrid from a
stale cached part-1 context; it is separate from the historical numerical
drift diagnosis for the original all-W4A16 text contexts.

## 10. Run the text smoke and capture a profile

QAIRT 2.45 distinguishes literal prompts from prompt files:

- `-p` or `--prompt` takes literal prompt text.
- `--prompt_file` takes a file path.
- `--profile` writes the runtime profile.

Deploying the repository runner creates a deterministic prompt file. Run it:

```bash
QAIRT_HOME=/opt/qairt/2.45.0.260326 \
BUNDLE_DIR=/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1 \
bash /home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1/run_text_smoke.sh \
  2>&1 | tee /home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1/text_smoke.log
```

For a manual profiled run:

```bash
GENIE_DIR=/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1
GENIE_CONFIG="$GENIE_DIR/genie_config.json"
PROMPT_FILE="$GENIE_DIR/smoke_prompt.txt"
PROFILE_FILE="$GENIE_DIR/smoke_profile_245.txt"

cd "$GENIE_DIR"
"$QAIRT_HOME/bin/$QAIRT_TARGET/genie-t2t-run" \
  -c "$GENIE_CONFIG" \
  --prompt_file "$PROMPT_FILE" \
  --profile "$PROFILE_FILE" \
  2>&1 | tee smoke_245.log
```

Do not replace `--prompt_file "$PROMPT_FILE"` with `-p "$PROMPT_FILE"`:
the latter sends the filename itself as the prompt.

### Measured hybrid result

Legacy Genie 1.17 with QAIRT 2.45 loads all four W4/FP16 text contexts on
HTP and returns coherent English. The measured text-only profile is:

| Metric | Result |
|---|---:|
| Time to first token | 131.152 ms |
| Prompt processing | 289.7711 tok/s |
| Token generation | 17.9982 tok/s |
| Query time | 2.489846 s |

This meets the text success criteria: the contexts initialize from binary,
the bundle selects `QnnHtp`, deterministic output is coherent, and the
profile contains prompt, generation, and first-token timing.

### Historical all-W4A16 diagnosis and GenieX result

Do not confuse the successful W4/FP16 text contexts with the original
all-W4A16 text artifact. In the original artifact, isolated AR128 parts have
main-output cosine similarities `0.99900`, `0.99968`, `0.99869`, and
`0.99798`, and captured-boundary part 4 selects token `785` (`The`). A true
native p1→p2→p3→p4 chain instead selects token `99590` (`答`) because its
small errors accumulate. Genie handoff buffers are byte-identical to that
native chain.

Qualcomm's official `qualcomm/GenieX` source at commit
`ab9e904fe21a01673840495309f3471dd6980d10`, tested through GenieX v0.3.16
and the same QAIRT `2.45.0.260326`, successfully imports that old bundle but
still emits incoherent Chinese fragments and repeated `A`. This establishes
only that GenieX does not repair the old all-W4A16 text artifact; the later
full-interface W4/FP16 part-1 experiment is a different artifact and uses
GenieX specifically to restore the DeepStack path.

Inspect the evidence:

```bash
grep -RniE '"type"[[:space:]]*:[[:space:]]*"QnnHtp"|QnnHtp' \
  "$GENIE_DIR"/*.json
find "$GENIE_DIR" -maxdepth 1 -type f -name '*.bin' -printf '%f\n' | sort
grep -Ei 'create From Binary|context|error|warning' smoke_245.log
test -s "$PROFILE_FILE"
python3 -m json.tool "$PROFILE_FILE" | head -100
```

If configuration, linked binaries, and context initialization do not jointly
establish the backend, preserve the run but do not count it as NPU proof.

## 11. Run the vision pipeline

Locate the Genie application script supplied by the exporter:

```bash
test -f /home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1/genie-app-script.txt
```

Run it from the directory containing its referenced assets:

```bash
GENIE_APP_SCRIPT=/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1/genie-app-script.txt
GENIE_APP_DIR="$(dirname "$GENIE_APP_SCRIPT")"
cd "$GENIE_APP_DIR"

"$QAIRT_HOME/bin/$QAIRT_TARGET/genie-app" \
  -s "$GENIE_APP_SCRIPT" \
  2>&1 | tee vision_smoke_245.log
```

The default script exits 0, binds and accepts all five image inputs, and
correctly describes Qualcomm's actual `dog.jpg` sample as a white fluffy dog
on green grass. The measured profile is:

| Stage | Result |
|---|---:|
| Image initialization | 915.653 ms |
| Image execution | 221.501 ms |
| Text initialization | 864.636 ms |
| Prompt processing | 414.32 ms / 678.43 tok/s |
| Generation | 12649.36 ms / 15.49 tok/s |
| Default-script wall time | 14.0 s |

This hybrid works because Genie source dequantizes the original W4A16
vision context's quantized `image_features` into the text accumulator. The
globally-W4 replacement's FP32 external interface is therefore unnecessary
and, with legacy Genie 1.17, incompatible.

The QAIRT 2.45 compatibility transform still disables the three auxiliary
deep-stack injections. The successful run validates the primary
`image_features` route and image-conditioned behavior, not a claim that those
disabled auxiliary paths were restored. The bundle remains in the canonical
inactive staging directory; the old active bundle was untouched, so this
validation creates no rollback action for that active path.

## 12. Run paired video frames on the NPU

### Understand the supported input

Neither legacy Genie nor stock GenieX decodes an MP4. Stock QAI Hub Models
also rejects Qwen3-VL's `pixel_values_videos` and `video_grid_thw` inputs, and
GenieX v0.3.16's public Qwen3-VL image frontend hardcodes a temporal grid of
one. The repository's custom runner bypasses that image-path frontend and
uses GenieX's lower-level `PixelData` API, but it still expects an already
decoded and packed raw tensor. It is not an MP4 decoder or a native
multi-clip video frontend.

Decode and sample video outside the runtime, then use the official local
Hugging Face processor to pack two consecutive frames as one Qwen3-VL
temporal patch. The tested 512 × 512 graph contract is:

```text
2 RGB frames
  -> pixel_values shape [1024, 1536]
  -> video_grid_thw [1, 32, 32]
  -> W4A16 vision context on QnnHtp
  -> image_features + deepstack_visual_embeds_0..2
  -> 256 visual tokens with grid [1, 32, 32]
  -> visual_pos_masks + full-interface W4/FP16 text part 1
  -> coherent W4/FP16 text parts 2-4 on QnnHtp
```

This is one *paired-frame temporal input*. With
`temporal_patch_size = 2`, two frames produce one temporal grid plane
(`T = 1`), so motion within the pair enters the vision patch embedding. It is
not native upstream video semantics: the current runner does not ingest
multiple temporal pairs, construct timestamp-aware MRoPE for a sequence of
pairs, or accept an encoded video container.

The earlier legacy-Genie compatibility bundle cannot exercise full
DeepStack. Its transform zeros the three intermediate vision additions,
prunes their branches, and removes `visual_pos_masks` plus
`deepstack_visual_embeds_0..2` from part 1. The GenieX experiment below uses a
different part 1 whose full auxiliary interface is restored.

### Assemble the full-DeepStack GenieX bundle

Create a W4/FP16 checkpoint directly from the unpruned W4A16 checkpoint, not
from the legacy compatibility checkpoint:

```bash
COSMOS_W4A16="$COSMOS_WORK/checkpoints/w4a16-cl512-smoke-r4"
COSMOS_FULL_W4_FP16="$COSMOS_WORK/checkpoints/w4-fp16-full-deepstack-cl512-r1"

python scripts/make_w4_fp16_checkpoint.py \
  --source "$COSMOS_W4A16" \
  --destination "$COSMOS_FULL_W4_FP16" \
  --keep-vision-w4a16
```

Export only `part1_of_4` from that checkpoint through the same
`Dragonwing IQ-9075 EVK` / OS 1.7 / QAIRT 2.45 Workbench flow used in Step 7:

```bash
python -m qai_hub_models.models.cosmos_reason2_2b.export \
  --checkpoint "$COSMOS_FULL_W4_FP16" \
  --target-runtime geniex_qairt \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --skip-profiling \
  --components part1_of_4
```

Download and extract that partial export into a new directory. Assemble the
GenieX bundle from the coherent legacy hybrid, replacing only part 1:

```bash
COSMOS_LEGACY_BUNDLE="$HOME/cosmos-reason2-iq9/exports/w4-fp16-text-w4a16-vision-hybrid-cl512-qairt245-r1"
COSMOS_PART1_EXPORT=/path/to/extracted/full-interface-part1-export
COSMOS_DEEPSTACK_BUNDLE="$HOME/cosmos-reason2-iq9/exports/w4-fp16-full-deepstack-hybrid-cl512-geniex-r2"

python scripts/prepare_geniex_bundle.py \
  --source "$COSMOS_LEGACY_BUNDLE" \
  --destination "$COSMOS_DEEPSTACK_BUNDLE" \
  --model-id qwen3_vl_cosmos_reason2_2b_deepstack_r2 \
  --hardlink \
  --part1-replacement-bundle "$COSMOS_PART1_EXPORT"
```

The assembler verifies that the replacement part 1 has all four auxiliary
inputs in a safe graph order, that its base inputs and outputs match the
coherent source part 1, and that the source vision graph supplies all three
matching DeepStack outputs. It retains the known-good W4/FP16 parts 2-4,
merges the replacement metadata, and removes the misleading legacy
compatibility marker. The measured replacement link was
[`j5m8x4ndp`](https://workbench.aihub.qualcomm.com/jobs/j5m8x4ndp/), target
`mnj67lgdq`; `part1_of_4.bin` has SHA-256
`7a59e6c83456cc93de4c84b9dbeb3acb9ef78b93c3f88eadfd33405eeded0d81`.

### Fetch the pinned Isaac Sim preview

The benchmark uses NVIDIA's official
[PhysicalAI World Model Synthetic Warehouse Operations
Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes).
The source scenarios were generated in Isaac Sim, and their staged events give
known labels. The helper downloads only pinned dataset-card previews and
checks byte counts and SHA-256 hashes; it does not download the multi-terabyte
dataset.

In WSL, with the project Conda environment active:

```bash
COSMOS_REPO=/mnt/c/path/to/qai-physics-reasoning
cd "$COSMOS_REPO"

python scripts/fetch_nvidia_sdg_warehouse.py \
  --asset-set clips \
  --extract-rgb \
  --case predict_near_miss
```

This crops only the leftmost RGB panel and writes:

```text
artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0030.png
artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0045.png
```

The benchmark assumes times 2.0 and 3.0 seconds from the preview duration.
They are documented estimates, not frame-accurate simulator timestamps. See
[`benchmarks/nvidia_sdg_warehouse/README.md`](../benchmarks/nvidia_sdg_warehouse/README.md)
for the ground-truth boundary and scoring rubric.

### Pack the temporal patch

Use the complete local Hugging Face checkpoint as the processor source and
the new full-DeepStack bundle as the shape and context contract:

```bash
COSMOS_PROCESSOR="$HOME/models/Cosmos-Reason2-2B"
COSMOS_VIDEO_CASE="$COSMOS_DEEPSTACK_BUNDLE/video_inputs/predict_near_miss_deepstack_r2"

python scripts/prepare_video_npu_inputs.py \
  --bundle "$COSMOS_DEEPSTACK_BUNDLE" \
  --processor "$COSMOS_PROCESSOR" \
  --output-dir "$COSMOS_VIDEO_CASE" \
  --frames \
    "$COSMOS_REPO/artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0030.png" \
    "$COSMOS_REPO/artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0045.png" \
  --timestamps 2.0 3.0 \
  --question "What is most likely to happen next, and is there an immediate safety risk? Answer in one short sentence."
```

Expected budget:

```text
Prepared paired video inputs: .../video_inputs/predict_near_miss_deepstack_r2
Context: 305/512 prompt tokens; 207 remain
```

The output contains the packed raw tensor, copies of the fixed positional and
attention inputs, timestamped text chunks, `genie-video-app-script.txt`, and
`video_npu_manifest.json`. The manifest records source-frame, processor,
bundle, tensor, text, and script hashes. The command refuses to overwrite an
existing output directory; use a new run suffix when preserving evidence.
The full-DeepStack runner consumes the packed
`sample_inputs/pair_000_pixel_values.raw` file; it does not execute the
generated legacy-Genie script.

CL512 supports only one 512 × 512 pair. Each pair costs 256 visual tokens, so
the tool rejects a second pair before generating a script. It accepts exactly
`2N` frame paths and the same number of strictly increasing timestamps, which
allows multiple pairs once a larger text context is deployed.

### Build, package, deploy, and run GenieX

The standalone integration pins Qualcomm GenieX commit
`ab9e904fe21a01673840495309f3471dd6980d10` (v0.3.16) and its
`geniex-qairt` submodule at
`f00a37cd50645456c8fbf0eaf179fa59640fd5bf`, including `geniex-proc`
`2ec6abcac6db8473fa26ea2f882f99652723cc7b`. Point the build at a clean
recursive checkout at those revisions. The build rejects dirty or divergent
source trees, copies the checkout, applies the repository's small
auxiliary-tensor classification patch to the copy, and cross-compiles in
Qualcomm's GenieX v0.0.1 toolchain image pinned by manifest digest
`sha256:694ca209accd73859ae889a689f5c9df3f8f202074c55c57641f69150b794a8f`:

```bash
GENIEX_SOURCE=/path/to/pinned/GenieX
GENIEX_BUILD="$HOME/cosmos-reason2-iq9/build/geniex-raw-video-v0316-r1"

bash scripts/geniex_raw_video.sh build \
  --geniex-source "$GENIEX_SOURCE" \
  --build-dir "$GENIEX_BUILD"
```

The patch makes `visual_pos_masks` and `deepstack_visual_embeds_*` explicit
auxiliary tensors so a QNN input reorder cannot cause GenieX to mistake one
for the hidden-state handoff. It does not alter model arithmetic.

Create a hash-bound run package for the prepared near-miss tensor:

```bash
GENIEX_PACKAGE="$HOME/cosmos-reason2-iq9/runs/geniex-near-miss-deepstack-r2"

bash scripts/geniex_raw_video.sh package \
  --bundle "$COSMOS_DEEPSTACK_BUNDLE" \
  --video-input-dir "$COSMOS_VIDEO_CASE" \
  --package-dir "$GENIEX_PACKAGE"
```

`package` checks the five model contexts, their shapes and hashes, the restored
part-1 input contract, CL512 capacity, Qwen3-VL MRoPE settings, the raw tensor
size, and the exact 256-image-token layout. It refuses to overwrite an
existing output directory.

The deploy helper deliberately does not copy the multi-gigabyte model. Copy
the new bundle once to a new target directory, then deploy the runner and
small case package. These commands assume the SSH key is already selected by
the default SSH configuration:

```bash
EVK_TARGET="ubuntu@<EVK IP>"
COSMOS_EVK_DEEPSTACK=/home/ubuntu/cosmos_reason2_2b_w4_fp16_full_deepstack_geniex_r2
GENIEX_REMOTE_ROOT=/home/ubuntu/cosmos_geniex_deepstack_near_miss_r2

if ssh "$EVK_TARGET" "test -e '$COSMOS_EVK_DEEPSTACK'"; then
  echo "Refusing to overwrite $COSMOS_EVK_DEEPSTACK" >&2
  exit 1
fi
scp -r "$COSMOS_DEEPSTACK_BUNDLE" \
  "${EVK_TARGET}:$COSMOS_EVK_DEEPSTACK"

bash scripts/geniex_raw_video.sh deploy \
  --build-dir "$GENIEX_BUILD" \
  --package-dir "$GENIEX_PACKAGE" \
  --evk "$EVK_TARGET" \
  --remote-root "$GENIEX_REMOTE_ROOT"
```

Run the target-side preflight and inference:

```bash
bash scripts/geniex_raw_video.sh run \
  --evk "$EVK_TARGET" \
  --remote-root "$GENIEX_REMOTE_ROOT" \
  --bundle "$COSMOS_EVK_DEEPSTACK" \
  --max-tokens 64 \
  --verbose
```

The launch preflight repeats bundle, package, shape, hash, prompt-budget, and
DeepStack-interface validation on the EVK before invoking the lower
`PixelData` runner. The runner inserts exactly 256 `<|image_pad|>` tokens,
passes grid `[1, 32, 32]`, obtains `image_features` and all three intermediate
vision outputs, constructs `visual_pos_masks`, and injects those features
through the full part-1 interface.

For the shelf case, repeat the fetch and preparation steps with
`--case predict_shelf_collision`, frames `0066` and `0088`, timestamps `4.38`
and `5.84`, and a new `video_inputs/predict_shelf_collision_deepstack_r2`
directory. Reuse the build and model bundle, but create a new package and a
new remote run directory; the scripts intentionally refuse to overwrite
either.

### Interpret the measured results

Both full-DeepStack physical-board runs pass their target preflights and
initialize the vision and all four text contexts cleanly through `QnnHtp` on
Hexagon v73. That proves the packed pair, primary vision output, three
DeepStack outputs, visual mask, and W4/FP16 decoder execute on the NPU. It
does not prove that the generated prediction is correct.

The measured free-form results are:

| Case | Prompt / generated tokens | TTFT | Generation | Exact output | Rubric |
|---|---:|---:|---:|---|---|
| Near miss | 305 / 27 | 648.4 ms | 16.60 tok/s | “The forklier will likely continue walking toward the forklift, which is still in motion and could be at risk of collision.” | Fail |
| Shelf collision | 304 / 21 | 648.2 ms | 16.26 tok/s | “The forklier will likely continue to push the forklier, which is at risk of falling.” | Fail |

The near-miss answer mentions collision risk but does not predict the staged
worker dodge and confuses the actor. The shelf answer neither identifies the
shelf collision nor forms a reliable physical prediction.

A multiple-choice control does not rescue the result:

| Case | Correct option | Model output | Interpretation |
|---|---:|---:|---|
| Near miss | B | C | Incorrect |
| Shelf, original option order | A | A | Superficially correct |
| Shelf, shuffled option order | B | A | Reject the earlier result as option-position bias |

The shelf answer remains `A` after the correct event moves to `B`, so the
initial match is not evidence of scene understanding. The explicit verdict
is therefore: **no robust realistic paired-frame NPU prediction case has
passed yet**, even with the full DeepStack inputs restored.

The exact BF16 model on CUDA, with the same 305-token near-miss prompt and
greedy generation, answers:

> The forklift will likely stop suddenly, posing a risk of collision or injury
> to the worker nearby.

The BF16 generation took 4.185 seconds and peaked at 4.615 GiB allocated GPU
memory. It passes hazard recognition but predicts that the forklift stops,
not the staged worker dodge, so it fails the strict future-outcome rubric.
It is nevertheless materially better than the current NPU answer, showing an
additional NPU-export quality loss rather than proving a full BF16 benchmark
pass.
The earlier legacy-compatibility hashes, exact answers, and BF16 status
distinction are in
[`docs/evidence/iq9075_video_smoke_r1.json`](evidence/iq9075_video_smoke_r1.json).

### Scale beyond one pair

The CL512 full-DeepStack interface is now tested, so simply restoring those
four auxiliary inputs is no longer the next step. The next practical profile
is:

1. Export a 384 × 384 vision graph with CL1024 W4/FP16 text contexts and the
   full DeepStack interface. A pair then costs 144 visual tokens; four pairs
   cost 576, leaving useful room in CL1024 for prompt and generation while
   retaining more spatial detail than 256 × 256.
2. Recalibrate the vision path with real, distinct paired-video frames. The
   current calibration duplicates still images to satisfy temporal patching,
   so its activation ranges do not represent motion within a pair.
3. Extend the raw runner and input manifest from one grid `[1, 32, 32]` to
   multiple temporal pairs, with an increasing timestamp and correctly
   interleaved Qwen3-VL multimodal-RoPE record for every pair.
4. Add controlled tests that shuffle labels, compare against the exact BF16
   checkpoint, and require the staged future event—not just a generic hazard
   phrase—to pass.

Keep encoded-video decoding and frame sampling outside GenieX unless a future
runtime exposes a documented video node. The custom lower-API runner bypasses
the stock image frontend, but the current implementation still carries only
one paired-frame temporal patch and does not supply native multi-pair video
semantics. See
[`docs/video_npu.md`](video_npu.md)
for the design boundary and runtime support matrix.

## Failure guide

| Symptom | Likely cause | Action |
|---|---|---|
| `401`, `403`, or gated-repository error | NVIDIA terms not accepted or wrong HF account | Accept the model terms in the browser, run `hf auth login`, then `hf auth whoami` |
| Model config validator fails | Wrong checkpoint or architecture revision | Use `nvidia/Cosmos-Reason2-2B`; do not edit the config to force a match |
| Adapter import fails | Overlay installed outside the active Conda environment | Activate `qai-cosmos-reason2`, rerun `scripts/setup_wsl_env.sh`, inspect `pip show` |
| AIMET import or CUDA provider failure | Wrong Python/CUDA wheel combination | Use the Python 3.10 environment and the pinned AIMET/ONNX Runtime GPU packages |
| Host OOM during quantization | Calibration too large | Start at context 512 with 8 text and 20 vision samples; close other GPU processes |
| Quantization completes but files are missing | One tower failed or output layout changed | Require every file checked by `finalize_checkpoint.py`, including text external data and embedding weights; vision external data is optional when weights are inline |
| Export reports a missing or invalid source checkpoint | The W4A16 directory lacks `source_checkpoint.json`, or its recorded BF16 path moved | Re-run finalization or set `COSMOS_SOURCE_CHECKPOINT` to the complete current BF16 snapshot |
| AI Hub device not found | Wrong device name or OS selection | Use `Dragonwing IQ-9075 EVK` and OS `1.7`; list devices through `qai_hub` |
| Compile job reports unsupported ops or graph rewrite errors | The unvalidated 2B architecture port hit a compiler/AIMET assumption | Preserve the Workbench job URL and logs; isolate the failing text part or vision encoder |
| Context binary fails immediately on the EVK | Compiler/runtime mismatch or incomplete bundle | Confirm the job used 2.45, use `/opt/qairt/2.45.0.260326`, and verify every referenced asset |
| `libQnn*.so` or `libGenie*.so` not found | Incorrect host library path | Put `lib/aarch64-oe-linux-gcc11.2` first; add gcc8.2 only if that directory exists |
| HTP skeleton/device failure | Wrong DSP path or incompatible context | Set `ADSP_LIBRARY_PATH` to `lib/hexagon-v73/unsigned`; verify v73 files and QAIRT compatibility |
| Prompt output refers to a filename | `-p` was given a path | Use `--prompt_file PATH`; reserve `-p` for literal text |
| `llama-mtmd-cli` rejects `--no-display-prompt` | An older baseline script passed a flag unsupported by the current EVK build | Remove that flag; it is not required for inference |
| QAIRT 2.45 and 2.47 libraries appear in one run | An inherited environment mixed installations | Rebuild a clean `PATH`/`LD_LIBRARY_PATH` rooted only at the artifact's matching QAIRT release |
| Process is killed on the EVK | Memory pressure; the board has no swap | Start with context 512, check `free -h`, and inspect kernel/OOM logs |
| Text works but vision fails | Wrong vision target, preprocessing metadata, or image connection problem | First verify that `vision_encoder.bin` is target `mngx5gv5q` with SHA-256 prefix `872e4e…`, then inspect the five image inputs and primary `image_features` connection |
| Process exits zero but no HTP evidence exists | Backend was not proven | Capture verbose runtime evidence and do not label the result an NPU success |
| Genie reports `Unsupported input tensor full_attention_mask dtype QNN_DATATYPE_FLOAT_32` | The globally-W4 vision target `mnzgp1ydq` was assembled into a legacy-Genie bundle | Replace it with original W4A16 vision target `mngx5gv5q`; direct `qnn-net-run` success does not remove the Genie interface mismatch |
| Genie initializes HTP contexts but emits incoherent text | The original all-W4A16 text contexts were used instead of the proven W4/FP16 links | Verify the four text target IDs and checksum prefixes from Step 7; preserve the bad run only as historical drift evidence |
| Legacy QAIRT 2.45 Genie rejects wildcard or auxiliary DeepStack connections | Legacy Genie cannot bind the full Qwen3-VL decoder interface | For the baseline, export the compatibility checkpoint and record that DeepStack is disabled; for the full interface, use the validated replacement part 1 and pinned GenieX runner |
| Linked graph order differs between text parts | Link inputs were supplied in a different order or stale part 1 was returned from cache | Inspect every context; for part 1 require cache-busted link `jprw31875`, target `mno2p6jvq`, checksum prefix `4fa761…`, and AR128→AR1 order |
| `prepare_video_npu_inputs.py` rejects an odd frame count | Temporal patches require exactly two frames | Pass exactly `2N` ordered frames and `2N` increasing timestamps |
| A second 512 × 512 pair exceeds context | Each pair costs 256 visual tokens and CL512 has no room for two pairs | Keep the tested run to one pair; next export the planned 384 × 384 / CL1024 full-DeepStack profile |
| GenieX cannot open an MP4 | Stock GenieX has no encoded-video node; the custom runner accepts packed raw `PixelData`, not a container | Decode and sample frames first, then use `prepare_video_npu_inputs.py` and the raw-runner package step |
| Full-DeepStack paired-frame run exits 0 but prediction is wrong | Functional execution and restored auxiliary inputs do not imply retained temporal accuracy | Recalibrate with distinct frame pairs, add multi-pair MRoPE semantics, and score shuffled-label controls against the exact BF16 reference |

## What to save for a reproducible result

Keep these together for every run:

- Git commit of this repository.
- QAI Hub Models version (`0.58.0`).
- Official checkpoint revision and `config.json`.
- Quantization arguments and calibration sample counts.
- Checksums of ONNX and encoding files.
- `qairt_245_compat.json`, including the removed inputs and source/output ONNX
  checksums.
- AI Hub compile/link job URLs and selected QAIRT version.
- Exported bundle checksum.
- Checksums and graph names for all deployed context binaries.
- EVK QAIRT path and version.
- Full text and vision logs.
- Profile output with prompt rate, decode rate, and time to first token.
- Test prompt, image, context length, and random/temperature settings.
- Captured QuantSim inputs, isolated-part outputs, native chained boundary
  dumps, and both first-token comparisons.

The 512-context hybrid text and vision smoke tests now pass. Preserve their
artifact IDs and profiles before repeating the two-export assembly at context
2048 or 4096.
