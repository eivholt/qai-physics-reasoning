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
independent correctness baseline. The second NPU route uses the official
GenieX `llama_cpp` runtime with Qualcomm's GGML Hexagon backend. This route is
now verified on the physical EVK with GenieX v0.3.17: the exact Cosmos GGUF
loads, image-conditioned CPU/NPU answers agree on a grounded warehouse event,
and the live successful process maps the Hexagon backend while holding the
secure CDSP FastRPC device. Cosmos-Reason2-2B retains Qwen3-VL-2B's model
structure, so this route also keeps DeepStack and visual-mask wiring inside
llama.cpp instead of exposing those tensors at a split QAIRT boundary.

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

Status below is current as of 2026-07-25.

| Item | Status |
|---|---|
| Adapter Python syntax, wheel contents, and dependency metadata | Verified locally |
| Adapter imports and helper-call signatures against the exact QAI Hub Models 0.58.0 wheel | Verified statically |
| 2B architecture constants against the public Qwen3-VL-2B config | Verified |
| SSH access to the IQ-9075 EVK | Verified after the 2026-07-25 reboot |
| QAIRT `2.45.0.260326` gcc11.2/gcc8.2 host libraries, Genie executables, Hexagon v73 DSP files, and FastRPC access on the EVK | Verified |
| QAIRT `2.47.0.260601` as a separately installed alternative runtime | Verified; Workbench does not currently offer a matching 2.47 compiler |
| A non-Cosmos Genie context binary loading under QAIRT 2.47 on this EVK | Verified; this proves that installation, not the Cosmos port |
| Qwen3-VL support in the EVK's `llama-mtmd-cli` and presence of both Cosmos GGUF files | Verified |
| Cosmos GGUF end-to-end CPU vision inference | Verified with the existing Q4_K_M model and F16 projector |
| Official-checkpoint Q4_0 GGUF and F16 projector conversion | Verified locally at pinned llama.cpp commit `910196f6b3dfc6aca88fa732e2b02f270ff9b56b`; sizes and SHA-256 hashes are recorded in Step 4 |
| Host llama.cpp six-frame barrier probe | At 84 visual tokens/image, Q4_0 answers `C` in both option orders and passes only the shuffled case; BF16 passes the unshuffled case. At 1024 visual tokens/image, Q4_0 passes both orders. This is a bounded host diagnostic, not broad quality evidence |
| Host llama.cpp encoded-video smoke | Mechanically successful on a five-second lathe clip, but its generic description/tool-slip framing missed the intended entanglement/unguarded-chuck hazard; this is not a quality pass |
| GenieX v0.3.17 `llama_cpp` Q4_0 on IQ-9075 | Verified end to end on CPU, NPU-only, and hybrid; the successful NPU process maps `libggml-hexagon.so` and `libcdsprpc.so` and holds `/dev/fastrpc-cdsp-secure` |
| GenieX single-image CPU/NPU control | Both answer `forklift`; NPU TTFT is 0.657 s versus CPU 2.458 s |
| GenieX grounded six-frame barrier control | CPU, NPU-only, hybrid, and pure-Q4_0 all correctly state that the forklift knocked the striped marker flat; standard NPU TTFT is 1.754 s versus CPU 7.339 s |
| GenieX exact-wording four-scene panel | Standard Q4_0 NPU scores 5/8 versus recorded BF16 GPU 7/8, with 6/8 exact answer parity; ordered stills are not the GPU runner's native paired-video tensors |
| GenieX shorter deployment prompt | Standard Q4_0 NPU scores 7/8 and returns the same answer pattern as the recorded GPU panel, but the GPU was not rerun with the shorter wording |
| GenieX task-specific edge profile | 4/4 across marker knockdown, safe box pickup, near-miss avoidance, and worker motion using fixed prompts plus declared ROI/final-pair preprocessing for the two small-actor scenes |
| GenieX large-image/context limits | `nctx=8192` aborts during vision-model allocation; one 1344 × 768 image aborts NPU and hybrid vision encoding with `dspqueue_read 0x2e`; `image-max-length` did not downscale it |
| Official gated Cosmos checkpoint download and architecture validation | Verified; all 15 files are present |
| Cosmos text and vision quantization | Verified for CL512; r5 compares the native-aspect 224 × 384 W8/A16 baseline with three compiled mixed-precision vision layouts, all derived from the older paired/explicit-resize calibration checkpoint |
| QAIRT 2.45 compatibility checkpoint | Verified; ONNX checker passes, split points are unchanged, and the four unsupported auxiliary text inputs are removed |
| Original all-W4A16 text contexts | Historical failure: all jobs succeed and execute on `QnnHtp`, but native p1→p4 drift changes the first token and Genie output is incoherent |
| Official GenieX v0.3.16 against that original all-W4A16 artifact | Evaluated; local import succeeds but reproduces that artifact's incoherent Chinese/repeated `A`, so orchestration alone does not repair it |
| Isolated original-W4A16 AR128 part agreement and forced AR1 prefill | Historical diagnostics only; isolated parts agree closely, while forced AR1 returns immediate EOS |
| All-FP16 activation checkpoint | Verified locally; 9,423 text and 934 vision activations are FP16 while all parameter encodings remain unchanged |
| All-FP16 text split QuantSim | Verified on four CUDA sessions; the bounded `Hello` smoke returns coherent text (`Hello,`) |
| All-FP16 AI Hub vision export | Failed at the QAIRT 2.45 vision link after the vision graph compiled; this is not a deployable bundle |
| W4/FP16 text contexts | Verified; all four cache-controlled links have AR128→AR1 order and legacy Genie produces coherent English |
| Globally-W4 mixed vision context | Compile `jpym3vrlp` and link `jp060e3np` succeed; direct EVK HTP execution succeeds in about 250 ms, but its external I/O is FP32 and legacy Genie rejects it |
| Historical legacy hybrid bundle | Verified end to end: original W4A16 vision target `mngx5gv5q` plus the four W4/FP16 text targets |
| Legacy Genie 1.17 text-only result | Verified coherent: 131.152 ms TTFT, 289.7711 prompt tok/s, 17.9982 generation tok/s, 2.489846 s query |
| Cosmos image-to-text through legacy Genie | Verified; default script exits 0, accepts all five image inputs, and describes Qualcomm's `dog.jpg` as a white fluffy dog on green grass |
| Native MP4/video-container input | Unsupported by legacy Genie and stock QAI Hub Models; GenieX documents only images, while an undocumented llama.cpp fallback can reach its ffmpeg helper but leaks the video context and is not native-Qwen3-VL equivalent |
| Learned temporal patch-projection bias | Fixed: the adapter now retains Qwen3-VL's learned 1024-element bias when adapting Conv3d to Conv2d |
| Native video input parity | Verified: NPU preparation uses native Hugging Face processing; per-pair GPU/NPU pixel hashes match, and the runner uses upstream video-pad token `151656` |
| Native-aspect vision profile | Verified on NPU: 224 × 384, `[336, 1536]` pixels, grid `[1, 14, 24]`, and 84 visual tokens per pair |
| Full-DeepStack GenieX bundle | Verified: full-interface W4/FP16 part 1 restores `visual_pos_masks` and `deepstack_visual_embeds_0..2`; coherent W4/FP16 parts 2-4 feed the bias-corrected native-aspect W8/A16 vision context |
| Full-DeepStack multi-pair NPU execution | Verified cleanly for one and three temporal pairs on `QnnHtp` / Hexagon v73 through the pinned GenieX lower `PixelData` API |
| Historical r3 exact one-pair quality | 0/5 strict units that the BF16 GPU reference passes |
| Historical initial three-pair two-scene/order control | r3 records 4/4 GPU/NPU agreement (`A`, `C`, `B`, `A`); valid but too narrow to generalize |
| Historical expanded three-pair four-scene/order control | r4 records BF16 GPU 7/8, NPU 4/8, exact answer parity 5/8; across marker, box, and near miss, GPU 6/6 and NPU 3/6 |
| Historical r4 robust realistic example | Near-miss avoidance passes 2/2 on GPU and NPU as its correct label moves from `C` to `B` |
| Historical r4 compact-prompt diagnostic | Removing 25 prompt tokens moves errors but leaves NPU 4/8 and exact parity 5/8; both box orders remain NPU-only failures |
| Historical r4 focused box/near diagnostic | GPU 4/4, NPU 3/4; NPU recognizes box as option `A` but keeps `A` after box moves to `B`, exposing order sensitivity |
| Historical r3/r4 three-pair free-form quality | Both strict NPU rubrics fail; the exact BF16 GPU diagnostics also fail both |
| Historical r3 four-pair CL512 diagnostic | Unsafe: 408/413-token prompts exceed the AR128 prefill-cache limit of 384 and corrupt NPU decode; guards now reject them |
| R5 frozen suite | Completed for four vision candidates and W8 text part 4; BF16 GPU scores 16/20 |
| Current completed NPU leader | Boundary-FP16 vision scores 13/20, exact GPU parity 15/20, GPU-correct retention 12/16, mean TTFT 857.510 ms |
| Other completed r5 vision candidates | W8/A16 baseline 11/20; W-FP16/A16 11/20; combined FP16 weights/internals with A16 boundaries 12/20 |
| W8 text part 4 | 12/20, exact parity 13/20, GPU-correct retention 11/16; 873.5 ms mean TTFT covers only 3/20 timing-bearing logs |
| W8 text part 1, layers 0–6 | AI Hub link and shared-weight contract verified; four-token HTP smoke passes; frozen 20 remains pending outside the resumed GenieX pilot |
| W8 text part 1, layers 0–3 and 0–2 | Compiled/linked; their stated contract checks are recorded below; EVK smoke and frozen 20 remain pending |
| Balanced P1–P4 extension | BF16 GPU 13/16 overall and 6/8 on new P3/P4; NPU execution remains pending |
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

The historical r3/r4 temporal result has a useful but bounded interpretation.
The current full-DeepStack runner interleaves one or three native Qwen3-VL
temporal pairs with timestamps. Exact native-processor pixels and
`<|video_pad|>` prompt construction match the BF16 reference. One-pair NPU
quality remains 0/5 on strict GPU-correct gates, but the three-pair shared
two-scene/order test matches GPU in all four r3 cases. The expanded r4
four-scene test is more discriminating: GPU scores 7/8, NPU 4/8, and exact
answers agree in 5/8 cases. The near-miss video passes in both option orders
on both devices, but the earlier 4/4 result does not generalize to broad
parity. A 25-token-shorter diagnostic also leaves NPU at 4/8 and preserves
the NPU-only box failures. A focused box/near control reaches NPU 3/4 and
shows that box recognition is order-sensitive, not totally absent. The strict
three-pair free-form cases still fail on both GPU and NPU. See
[`docs/video_npu.md`](video_npu.md)
and the
[`native-aspect parity report`](evidence/iq9075_video_aspect_native_parity_r3.json)
and
[`expanded four-scene report`](evidence/iq9075_video_four_scene_parity_r4.json).

The r5 precision sweep is the current comparison. Its frozen 20 probes
combine the r4 primary set, compact-prompt set, and focused box/near-miss set.
GPU scores 16/20. The original NPU baseline scores 11/20; boundary-FP16
vision improves to 13/20 and agrees exactly with 15/20 GPU letters. It retains
12 of the 16 GPU-correct answers, versus 10/16 for baseline. This is the best
completed NPU result, but it remains three correctness points behind GPU.
The combined FP16-weight/internal candidate has the lowest mean TTFT
(686.080 ms) but scores 12/20, so host numerical fidelity and speed do not
substitute for end-task scoring. See the
[`r5 precision report`](evidence/iq9075_video_precision_parity_r5.json).

The independent GenieX GGUF route is now a real second baseline rather than a
paper design. With the exact tracked four-scene wording, its ordered-still NPU
panel scores 5/8 versus the recorded BF16 GPU's 7/8 and reproduces 6/8 GPU
letters. Changing only the instruction prefix to identify a chronological
frame sequence and requesting one letter raises the NPU result to 7/8. That
answer sequence matches the recorded GPU sequence, but the GPU was not rerun
with the shorter prompt, so this is not an exact same-prompt parity claim. A
declared task-specific profile reaches 4/4 by using fixed ROI preprocessing
for the small worker/box cases and only the decisive first/final pair for
near-miss motion. See the
[`r6 GenieX GGUF report`](evidence/iq9075_geniex_gguf_r6.json).

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
| Current 224 × 384 visual tokens | 84 |
| Temporal patch | 2 frames, grid `[1, 14, 24]`, 84 tokens |
| Current completed accuracy leader | Bias-corrected W8 vision, 925 FP16 internal activations, nine A16 boundaries + W4/FP16 text |

In practical terms, the model has two cooperating towers. The 24-layer vision
transformer repeatedly turns small image regions into increasingly contextual
features; its width of 1024 is the size of the feature vector carried for
each region inside that tower. A 224 × 384 input is divided into 16 × 16
patches, giving a 14 × 24 grid, and spatially merged in 2 × 2 groups, leaving
84 visual tokens. A
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
  |-> preserve learned temporal patch-projection bias
  -> AIMET-ONNX text + 224 x 384 paired-frame vision calibration
  |-> boundary-FP16 vision leader or W8/A16 baseline
  |    `-> primary + three DeepStack outputs
  |-> W4/FP16 text links
  |    |-> legacy compatibility p1-p4, auxiliary inputs removed
  |    `-> full-interface replacement p1, auxiliary inputs restored
  -> assemble either runtime-specific five-context bundle
  |-> legacy Genie 1.17: compatibility p1-p4
  `-> pinned GenieX PixelData runner: native pixels + <|video_pad|>
      -> full p1 + coherent p2-p4
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
- The official GenieX CLI installed separately for the GGUF pilot. Qualcomm
  documents native Linux ARM64 support for Dragonwing QCS9075 and access to
  Hexagon through the board's Qualcomm driver packages. Do not replace the
  pinned v0.3.16 raw-runner tree with this installation.

Qualcomm's
[v0.3.16-to-v0.3.17 comparison](https://github.com/qualcomm/GenieX/compare/v0.3.16...v0.3.17)
was reviewed as well. The relevant point is not that v0.3.17 introduced a new
VLM runtime: `sdk/plugins/llama_cpp/{llm,vlm}` and its tests were already
present in v0.3.16. The route-specific value of the v0.3.17 material is
documentation and discoverability, not a new VLM implementation. Its general
catalog/cache and device-detection changes, plus the subsequent
documentation-only main-branch change, do not alter the multimodal arithmetic.
Neither release repairs the split-QAIRT artifact's `PixelData`, DeepStack,
visual-mask, or quantized arithmetic. Keep the measured raw-runner evidence
pinned to v0.3.16, and evaluate the separately installed `llama_cpp` runtime
as a distinct GGUF experiment.

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

### 4A. Convert and stage a Q4_0 GGUF bundle for GenieX

This is an alternative to the split-QAIRT flow in Steps 5–12, not a
replacement for its measured evidence. NVIDIA states that Cosmos-Reason2-2B
is post-trained from `Qwen3-VL-2B-Instruct` and follows the same architecture.
llama.cpp can therefore convert the text model and multimodal projector using
its Qwen3-VL converter. Qualcomm's
[GenieX model guide](https://geniex.aihub.qualcomm.com/en/models/supported)
documents Qwen3-VL as a `vlm`, a main GGUF next to `mmproj-*.gguf`, and
`Q4_0` as the format with the best Hexagon NPU support. Architecture
compatibility makes the experiment reasonable; it is not evidence that the
converted model is accurate on Hexagon.

Pin the converter before producing either file:

```bash
LLAMA_CPP="$HOME/src/llama.cpp-910196f6"
COSMOS_GGUF_BUILD="$HOME/models/cosmos_reason2_2b_gguf_build"
MODEL_DIR="$HOME/models/cosmos_reason2_2b_geniex_q4_0"

git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
git -C "$LLAMA_CPP" checkout 910196f6b3dfc6aca88fa732e2b02f270ff9b56b

python3 -m pip install -r "$LLAMA_CPP/requirements.txt"
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF
cmake --build "$LLAMA_CPP/build" \
  --config Release \
  --target llama-quantize llama-mtmd-cli \
  -j "$(nproc)"

mkdir -p "$COSMOS_GGUF_BUILD" "$MODEL_DIR"

python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$COSMOS_OFFICIAL" \
  --outfile "$COSMOS_GGUF_BUILD/Cosmos-Reason2-2B-BF16.gguf" \
  --outtype bf16

python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$COSMOS_OFFICIAL" \
  --mmproj \
  --outfile "$MODEL_DIR/mmproj-Cosmos-Reason2-2B-F16.gguf" \
  --outtype f16

"$LLAMA_CPP/build/bin/llama-quantize" \
  "$COSMOS_GGUF_BUILD/Cosmos-Reason2-2B-BF16.gguf" \
  "$MODEL_DIR/Cosmos-Reason2-2B-Q4_0.gguf" \
  Q4_0

sha256sum \
  "$MODEL_DIR/Cosmos-Reason2-2B-Q4_0.gguf" \
  "$MODEL_DIR/mmproj-Cosmos-Reason2-2B-F16.gguf"
```

The pinned conversion completed locally. Its two deployable files are:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Cosmos-Reason2-2B-Q4_0.gguf` | 1,229,455,008 | `90bd38f1a117b2d9738642165dfd8c0c6273a4bdfd214e1354d0502cff61e8a9` |
| `mmproj-Cosmos-Reason2-2B-F16.gguf` | 819,395,424 | `43df50be60fd6c9075548cd6d759e21253a3c62da69055bab7864cc64aa00c2c` |

The default Q4_0 recipe intentionally leaves `output.weight`, the
311,164,928-element vocabulary projection, in Q6_K. GenieX's pinned Hexagon
backend does not advertise Q6_K, so that tensor is expected to fall back from
HTP. Prepare a separate, experimental all-Q4_0 candidate for an on-device
residency/speed A/B:

```bash
"$LLAMA_CPP/build/bin/llama-quantize" \
  --pure \
  "$COSMOS_GGUF_BUILD/Cosmos-Reason2-2B-BF16.gguf" \
  "$COSMOS_GGUF_BUILD/Cosmos-Reason2-2B-Q4_0-pure.gguf" \
  Q4_0
```

That candidate is 1,149,232,800 bytes with SHA-256
`a2eeb9a09bc7e82f903ce243d18e93e0b686b8c8cc0616c7c2b089094b1305fc`.
It preserves both 1024-token barrier answer-order passes (`A`, then `C`) in
the limited host smoke. It is not the default because quantizing the output
head can still change other answers; score the full frozen suite before
trading the standard artifact's Q6_K quality for broader HTP eligibility.

Keep only the Q4_0 main model and F16 projector in `MODEL_DIR` when importing
it. In particular, do not leave the intermediate BF16 main model beside them,
because an inferred local layout with two possible main GGUFs is ambiguous.
The weights remain subject to NVIDIA's license and are not committed to this
repository.

Copy that directory to a separate EVK location:

```bash
EVK_IP="<EVK IP>"
EVK_USER=ubuntu
EVK_SSH_KEY=/mnt/c/path/to/evk_ssh_key
EVK_TARGET="${EVK_USER}@${EVK_IP}"

ssh -i "$EVK_SSH_KEY" "$EVK_TARGET" \
  'mkdir -p /home/ubuntu/models/cosmos_reason2_2b_geniex_q4_0'

scp -i "$EVK_SSH_KEY" \
  "$MODEL_DIR/Cosmos-Reason2-2B-Q4_0.gguf" \
  "$MODEL_DIR/mmproj-Cosmos-Reason2-2B-F16.gguf" \
  "${EVK_TARGET}:/home/ubuntu/models/cosmos_reason2_2b_geniex_q4_0/"
```

On the EVK, install the current official CLI in a separate location, following
Qualcomm's
[Linux ARM64 instructions](https://geniex.aihub.qualcomm.com/en/run/cli/install).
The IQ9075 Ubuntu image normally has the Qualcomm PPA preconfigured:

```bash
ssh -i "$EVK_SSH_KEY" "$EVK_TARGET"

sudo apt update
sudo apt install -y \
  libatomic1 \
  libglib2.0-0 \
  ocl-icd-libopencl1 \
  qcom-fastrpc1

curl -fsSL \
  https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh \
  | sh -s -- \
      --version v0.3.17 \
      --prefix /home/ubuntu/geniex-cosmos-v0317

geniex --version
```

The explicit prefix keeps an existing v0.3.16 or source-built runtime intact.
The installer updates only the small launcher in `$HOME/.local/bin`; its
generated wrapper points at the selected prefix and sets the required library
path. On the measured board, `qcom-fastrpc1` was already installed but
`ocl-icd-libopencl1` was missing. Without the latter, even `--compute cpu`
fails before model loading because the bundled plugin links its OpenCL backend
at load time.

Register the local VLM once, then run the same frozen image and prompt on the
two relevant compute modes:

```bash
MODEL_DIR=/home/ubuntu/models/cosmos_reason2_2b_geniex_q4_0
GENIEX_DATA=/home/ubuntu/geniex-cosmos-v0317-data
PROMPT="Describe the scene and identify imminent hazards. /absolute/path/to/frame.png"

geniex --data-dir "$GENIEX_DATA" pull local/cosmos-reason2-2b:Q4_0 \
  --model-hub localfs \
  --local-path "$MODEL_DIR" \
  --model-type vlm

geniex --data-dir "$GENIEX_DATA" infer local/cosmos-reason2-2b:Q4_0 \
  --compute npu \
  --ngl -1 \
  --nctx 4096 \
  --max-tokens 256 \
  --top-k 1 \
  --seed 42 \
  --prompt "$PROMPT" \
  2>&1 | tee cosmos_geniex_npu.log

geniex --data-dir "$GENIEX_DATA" infer local/cosmos-reason2-2b:Q4_0 \
  --compute hybrid \
  --ngl -1 \
  --nctx 4096 \
  --max-tokens 256 \
  --top-k 1 \
  --seed 42 \
  --prompt "$PROMPT" \
  2>&1 | tee cosmos_geniex_hybrid.log
```

For `llama_cpp`, `--compute npu` pins execution to Hexagon `HTP0`.
`--compute hybrid` leaves the device ID empty and enables llama.cpp's
per-tensor HTP-plus-CPU scheduler with all layers eligible for offload;
Qualcomm's
[platform and runtime guide](https://geniex.aihub.qualcomm.com/en/get-started/platforms)
documents hybrid as the faster Snapdragon path. `--ngl -1` makes the all-layer
intent explicit. The flag is still only a request: preserve logs and
backend/profile output that actually identifies Hexagon before calling a run
NPU-proven.

GenieX v0.3.17 pins llama.cpp commit `ae9291e16`, while the converter and host
smokes above use `910196f6`, 104 commits newer. The older pin has the required
Qwen3-VL, DeepStack, M-RoPE, and frame-pair loaders, so no GGUF schema blocker
was found. It predates upstream commit
[`b4aa7dd477acf065a3b9c6a8cf324c904da1a834`](https://github.com/ggml-org/llama.cpp/commit/b4aa7dd477acf065a3b9c6a8cf324c904da1a834),
which changes Qwen3-VL learned-position interpolation to
`align_corners=True` to match Transformers. This can move grounding
coordinates on non-square inputs. If stock v0.3.17 differs from the current
host result, test a GenieX rebuild with that fix before blaming Hexagon
numerics.

Use `--top-k 1 --seed 42` for deterministic comparisons. Do not rely on
`--temperature 0` as a greedy setting in this GenieX implementation: zero is
treated as "unset" and replaced by the llama.cpp default temperature `0.8` in
[`build_sampling_params`](https://github.com/qualcomm/GenieX/blob/main/sdk/plugins/llama_cpp/src/params.cpp).
Top-k one leaves a single candidate regardless of that fallback.

The CLI can attach multiple ordered image files found in a prompt. This is a
useful static-frame experiment, but it is not native video preprocessing:

```bash
geniex --data-dir "$GENIEX_DATA" infer local/cosmos-reason2-2b:Q4_0 \
  --compute npu \
  --ngl -1 \
  --nctx 4096 \
  --max-tokens 256 \
  --top-k 1 \
  --seed 42 \
  --prompt "The following frames are chronological. Predict the next safety-relevant event. /absolute/path/to/frame_000.png /absolute/path/to/frame_001.png /absolute/path/to/frame_002.png /absolute/path/to/frame_003.png"
```

The exact six-frame barrier probe reveals an important visual-budget
interaction:

| Host model | Visual tokens/image | Choice order | Expected | Returned |
|---|---:|---|---:|---:|
| Q4_0 main + F16 projector | 84 | Primary | `A` | `C` |
| Q4_0 main + F16 projector | 84 | Shuffled | `C` | `C` |
| BF16 main + F16 projector | 84 | Primary | `A` | `A` |
| Q4_0 main + F16 projector | 256 | Primary | `A` | `C` |
| Q4_0 main + F16 projector | 512 | Primary | `A` | `C` |
| Q4_0 main + F16 projector | 1024 | Primary | `A` | `A` |
| Q4_0 main + F16 projector | 1024 | Shuffled | `C` | `C` |

At the natural 384 × 216 input size, Q4_0's two `C` answers expose a
letter-C bias rather than robust grounding. Raising the llama.cpp visual
budget to 256 and then 512 tokens/image does not change the wrong primary
answer; 1024 fixes both orders for this one probe. The BF16 result at 84
tokens shows that quantization sensitivity contributes to the low-budget
failure. Do not generalize from two passing permutations: repeat the same
intervention across the frozen scenes and negative controls.

The direct llama.cpp test can request that budget with
`--image-min-tokens 1024`. Current GenieX `llama_cpp` VLM plumbing appears not
to honor its `image_max_length` field, so do not assume
`--image-max-length 1024` has the same effect. The attempted workaround
confirms the risk: one deterministic 1344 × 768 frame aborts both NPU-only and
hybrid vision encoding with `ggml-hex: dspqueue_read failed: 0x0000002e`,
even when the flag is left at its default. An `nctx=8192` run separately
aborts during vision-model allocation. Keep the verified `nctx=4096`,
384 × 216 grid for this package. If the relevant actor is too small, use a
fixed region-of-interest policy and resize the crop back to 384 × 216; do not
increase the Hexagon grid until a runtime update removes this failure.

Increase `--nctx` only after checking memory headroom; each image adds visual
tokens. The current GenieX CLI documents image paths, not encoded video.
Source inspection shows that an `.mp4` can nevertheless fall through the
generic media loader into pinned llama.cpp's ffmpeg-backed video helper.
That wrapper discards the returned video-context owner while retaining its
lazy bitmap callback, leaking the context and making lifetime behavior
unsuitable as a supported or repeatable interface. Use ordered extracted
frames for GenieX evaluation; reserve direct MP4 for an explicitly labeled
one-shot diagnostic. The pinned helper can fuse consecutive Qwen-VL frames.
The
[pinned helper interface](https://github.com/ggml-org/llama.cpp/blob/910196f6b3dfc6aca88fa732e2b02f270ff9b56b/tools/mtmd/mtmd-helper.h)
defaults to 4 FPS with a timestamp text chunk every five seconds, whereas this
project's native-Qwen3-VL reference uses 2 FPS and a timestamp for every
temporal pair. Direct `llama-mtmd-cli` video output is therefore an
exploratory smoke, not GPU-equivalent preprocessing. The completed
five-second lathe smoke loaded 20 frames and described the scene, but its
hazard response stayed at generic tool-slip/bolt language and missed the
intended entanglement/unguarded-chuck risk, reinforcing that distinction.

The physical-board result is:

| Probe | CPU | NPU-only | Hybrid |
|---|---|---|---|
| One image: identify the vehicle | `forklift`, 2.458 s TTFT | `forklift`, 0.657 s TTFT | Not needed |
| Six frames: marker state and cause | Correct, 7.339 s TTFT | Correct, 1.754 s TTFT | Correct, 1.751 s TTFT |

For the grounded event, CPU says that the blue forklift knocked the striped
marker flat; NPU-only and hybrid produce the same material answer. The live
successful NPU process maps `libggml-hexagon.so` and `libcdsprpc.so`, holds
`/dev/fastrpc-cdsp-secure` and DMA-buffer descriptors, and therefore supplies
direct HTP evidence beyond the `--compute npu` flag.

The exact tracked four-scene wording scores 5/8 on this standard Q4_0 route,
versus the recorded BF16 GPU's 7/8, with 6/8 exact answer parity. The two NPU
misses against GPU are normal-order marker knockdown and shuffled box pickup.
Changing the instruction to:

```text
Which event is shown in this chronological frame sequence?
Give only the letter: ...
```

raises the NPU panel to 7/8 with answers `A/C`, `B/D`, `C/B`, and `C/A`;
this happens to equal the recorded GPU answer sequence. Because the GPU
reference used the longer benchmark wording, report this as a prompt-profile
improvement, not exact same-prompt parity. For free-form edge tasks, a fixed
256 × 144 ROI at `(64, 72)` resized back to 384 × 216 makes the box action
explicit (`safe box pickup`), and the decisive first/final ROI pair makes the
near-miss motion explicit (`The worker is running away from the forklift.`).
The full task-specific profile passes 4/4, but it is intentionally narrower
than a broad zero-shot benchmark.

Use the standard Q4_0 main as the quality default. The pure-Q4_0 output head
improves barrier decode speed from 20.9 to 26.3 tokens/s and total time from
3.19 to 2.85 seconds, but it loses box detail on the full-frame free-form
probe. Full commands, artifact hashes, all eight choice results, the 4/4 edge
profile, and failure boundaries are recorded in
[`iq9075_geniex_gguf_r6.json`](evidence/iq9075_geniex_gguf_r6.json).

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

### Generate the r5 precision checkpoints

The r5 transforms are deterministic, fail closed on the expected tensor
counts and calibration provenance, and write a marker beside each derived
checkpoint. Use new destination directories:

```bash
python scripts/make_vision_a16_boundary_checkpoint.py \
  /path/to/all-fp16-native-aspect \
  /path/to/native-aspect-w8a16 \
  /path/to/new-boundary-fp16

python scripts/make_vision_fp16_weight_checkpoint.py \
  /path/to/native-aspect-w8a16 \
  /path/to/new-wfp16-a16

python scripts/make_vision_fp16_weight_boundary_checkpoint.py \
  /path/to/new-wfp16-a16 \
  /path/to/new-boundary-fp16 \
  /path/to/new-wfp16-boundary
```

The boundary checkpoint keeps all 205 calibrated vision parameter encodings
at W8, keeps nine graph-facing activations at A16, and sets 925 internal
activations to FP16. The weight transform changes the 205 vision parameter
encodings to canonical FP16 while preserving all 934 A16 activations. The
combined transform uses FP16 parameters and internals with the same nine A16
boundaries.

Promote decoder matrices to independently derived W8 ranges from the external
FP32 ONNX initializers:

```bash
python scripts/make_text_w8_matrix_checkpoint.py \
  /path/to/w4-fp16-full-deepstack-cl512 \
  /path/to/new-w8-part1-full \
  --parts part1_of_4 \
  --image-size 224 384

python scripts/make_text_w8_matrix_checkpoint.py \
  /path/to/w4-fp16-full-deepstack-cl512 \
  /path/to/new-w8-part1-layers-0-3 \
  --parts part1_of_4 \
  --layers 0 1 2 3 \
  --image-size 224 384

python scripts/make_text_w8_matrix_checkpoint.py \
  /path/to/w4-fp16-full-deepstack-cl512 \
  /path/to/new-w8-part1-layers-0-2 \
  --parts part1_of_4 \
  --layers 0 1 2 \
  --image-size 224 384

python scripts/make_text_w8_matrix_checkpoint.py \
  /path/to/w4-fp16-full-deepstack-cl512 \
  /path/to/new-w8-part4-full \
  --parts part4_of_4 \
  --image-size 224 384
```

The tool recomputes every selected W8 range; it does not scale a W4 encoding.
Before publishing the destination it requires that the same AIMET calculation
reproduce the source W4 ranges. Part 1 layers 0–6 changes 252 matrices,
layers 0–3 changes 144, layers 0–2 changes 108, and part 4 layers 21–27
changes 252.

Compare the vision candidates against the adapted BF16 encoder on exactly the
manifest-bound pixels before cloud export:

```bash
python scripts/compare_vision_checkpoints.py \
  --source-checkpoint /path/to/Cosmos-Reason2-2B \
  --candidate-checkpoint baseline=/path/to/native-aspect-w8a16 \
  --candidate-checkpoint boundary=/path/to/new-boundary-fp16 \
  --candidate-checkpoint wfp16_a16=/path/to/new-wfp16-a16 \
  --candidate-checkpoint combined=/path/to/new-wfp16-boundary \
  --manifest /path/to/case-1/video_npu_manifest.json \
  --manifest /path/to/case-2/video_npu_manifest.json \
  --output-json /path/to/new-vision-comparison.json \
  --device cuda \
  --image-height 224 \
  --image-width 384
```

The comparison validates the recorded shapes, byte counts, and SHA-256 values
before loading a model. It is a diagnostic, not an NPU accuracy score.

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

### Historical legacy-Genie linked contexts

The proven legacy-compatibility text contexts are:

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

The corresponding legacy vision context is the original W4A16 artifact:

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

### R5 AI Hub artifact sheet

The r5 vision candidates were uploaded, compiled, linked, downloaded, hashed,
and then scored as distinct artifacts:

| Candidate | Source model | Compile job → model | Link job → target | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| W8/A16 baseline | `mqyx8ydvm` | [`jgzndw66g`](https://workbench.aihub.qualcomm.com/jobs/jgzndw66g/) → `mm55gz94m` | [`j5w46xkjg`](https://workbench.aihub.qualcomm.com/jobs/j5w46xkjg/) → `mn0ke303q` | 421,351,424 | `7af8418bb61e2581fc862ce2aabf320ab3f3bfae0a7c5400e7d43b7d386028bd` |
| Boundary-FP16 | `mn7yvrv3n` | [`jgk8x9m2g`](https://workbench.aihub.qualcomm.com/jobs/jgk8x9m2g/) → `mq2g35wjm` | [`j5qvqmo4g`](https://workbench.aihub.qualcomm.com/jobs/j5qvqmo4g/) → `mmr850l6n` | 425,918,464 | `6e217777c18cff6dd60ebff6a1422750b8e93d5c8ab05df1abb00eaaa17e2cac` |
| W-FP16/A16 | `mqpr1p9gm` | [`jp060vqep`](https://workbench.aihub.qualcomm.com/jobs/jp060vqep/) → `mn1xgp7zq` | [`jp81y4985`](https://workbench.aihub.qualcomm.com/jobs/jp81y4985/) → `mmdwpzo3q` | 818,913,280 | `70c2fc1b854a7950116c5a7c680c9bae4c9a01a2c0e66659da591191fa1351a9` |
| Combined FP16 weights/internals + A16 boundaries | `mmxk9242n` | [`jgnkvkzjg`](https://workbench.aihub.qualcomm.com/jobs/jgnkvkzjg/) → `mm60gpodm` | [`jprw3wlk5`](https://workbench.aihub.qualcomm.com/jobs/jprw3wlk5/) → `mq3lgro9q` | 821,522,432 | `7ad306fc69273720c4dd7c86f8b7aa5e9688d7f9534eb8776b99db7cfd7adbb3` |

The text experiments use separate AR128 and AR1 compiles:

| Candidate | Source model | AR128 compile → model | AR1 compile → model | Link → target | Bytes | SHA-256 |
|---|---|---|---|---|---:|---|
| Part 1 layers 0–6 | `mnzgpplzq` | [`jgjqn7lv5`](https://workbench.aihub.qualcomm.com/jobs/jgjqn7lv5/) → `mqe411r5m` | [`j5w46793g`](https://workbench.aihub.qualcomm.com/jobs/j5w46793g/) → `mqyx88zxm` | [`jpv7k21kp`](https://workbench.aihub.qualcomm.com/jobs/jpv7k21kp/) → `mn7yvo34n` | 357,736,448 | `8ea7ba19baaaa90cbb8e1ee34ab07f37ea982860d3718d96a2a3d3405a341125` |
| Part 1 layers 0–3 | `mn7yvok3n` | [`jp43redv5`](https://workbench.aihub.qualcomm.com/jobs/jp43redv5/) → `mm60go0dm` | [`jpxxo061p`](https://workbench.aihub.qualcomm.com/jobs/jpxxo061p/) → `mq3lgol9q` | [`jgnkv1mrg`](https://workbench.aihub.qualcomm.com/jobs/jgnkv1mrg/) → `mm55go5km` | 282,030,080 | `eaeb2564ec11b9aca40aff914edc983f42f6858b89b6f43a305edd89cdec594a` |
| Part 1 layers 0–2 | `mn7yvkl4n` | [`jgzndyd6g`](https://workbench.aihub.qualcomm.com/jobs/jgzndyd6g/) → `mmdwp8yoq` | [`j5w46z6jg`](https://workbench.aihub.qualcomm.com/jobs/j5w46z6jg/) → `mnw4r1erq` | [`jpxxork9p`](https://workbench.aihub.qualcomm.com/jobs/jpxxork9p/) → `mnzgp4vxq` | 256,897,024 | `0a7a0642800aafec446fa6d7e90e1e1dc20959c12b028e75f7a1b214067b376d` |
| Part 2 layers 7–13 | `mm55g5pym` | [`jp16z1rk5`](https://workbench.aihub.qualcomm.com/jobs/jp16z1rk5/) → `mn0kek7zq` | [`jgd214jk5`](https://workbench.aihub.qualcomm.com/jobs/jgd214jk5/) → `mqyx8x99m` | [`j579rnqqg`](https://workbench.aihub.qualcomm.com/jobs/j579rnqqg/) → `mn7yvydon` | 357,339,136 | `d65f751739ea4410920c7c04b06729f4f468e512e665c91577989653bdf0f50a` |
| Part 3 layers 14–20 | `mm55g59ym` | [`j5w46zozg`](https://workbench.aihub.qualcomm.com/jobs/j5w46zozg/) → `mn0kek0zq` | [`jg9dn2vq5`](https://workbench.aihub.qualcomm.com/jobs/jg9dn2vq5/) → `mn7yvypon` | [`jp16z10k5`](https://workbench.aihub.qualcomm.com/jobs/jp16z10k5/) → `mm55g599m` | 357,339,136 | `41b7e3d8576e3904f74714941360cf9dadb140905f287663de5044efb67689f9` |
| Part 4 layers 21–27 | `mqpr1w3lm` | [`jp81ye3k5`](https://workbench.aihub.qualcomm.com/jobs/jp81ye3k5/) → `mq92j4w0m` | [`jgk8x2lwg`](https://workbench.aihub.qualcomm.com/jobs/jgk8x2lwg/) → `mq87gwzgm` | [`j5qvql7ng`](https://workbench.aihub.qualcomm.com/jobs/j5qvql7ng/) → `mn1xg2krq` | 672,043,008 | `f3a460e2310e425bfd8df0c56ee4dea35463bfe060fbd5692bb0a0f97ed3c604` |

Part-1 full initially hit an AI Hub identity-cache problem: ordinary relinks
returned AR1→AR128 even when the requested input order was AR128 then AR1.
Create byte-distinct but semantically identical upload artifacts by changing
only the validated DLC/ZIP end-of-central-directory comment:

```bash
python scripts/cache_bust_dlc.py \
  --marker cosmos-part1-ar128-r5 \
  /path/to/ar128-compiled.dlc \
  /path/to/new-ar128-cache-busted.dlc

python scripts/cache_bust_dlc.py \
  --marker cosmos-part1-ar1-r5 \
  /path/to/ar1-compiled.dlc \
  /path/to/new-ar1-cache-busted.dlc
```

The accepted full part-1 link used new upload model identities `mm55gov6m`
(AR128) and `mn46jeo0q` (AR1), uploaded in that order. The helper validates
the ZIP/ZIP64 structure, member metadata, compressed payloads, CRCs, and
archive comment before atomically publishing the destination. It does not
change a graph payload.

After every link, inspect the context before assembly. Require graph order
AR128→AR1 and an equal, nonzero `sharedWeightsSize` for both graphs. The full
part-1 link reports 353,431,552 shared-weight bytes per graph and passed an
EVK four-token smoke (`The safety marker is`), with 875.0 ms TTFT and
11.34 tokens/s decode. A local no-sharing link also had the requested graph
order, but its 711,176,192-byte binary failed on the EVK while mapping a
490,733,568-byte FastRPC buffer. A one-token prefill from that artifact did
not establish usable decode, so the no-sharing candidate is rejected.

The layers-0–3 link reports 277,934,080 shared-weight bytes for each graph.
The layers-0–2 AI Hub context also has the intended order/shared-weight
configuration. Their EVK smokes and frozen-suite scores remain pending. The
part-4 context passed a four-token HTP smoke and completed the frozen suite;
its score is recorded in Step 12.

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

Neither legacy Genie nor the documented GenieX interface decodes an MP4. The
CLI can attach one or more image paths. Its undocumented generic-media
fallback can reach upstream llama.cpp's ffmpeg helper, but the current wrapper
leaks the returned video context and is excluded from this reproducible path.
Stock QAI Hub Models also rejects Qwen3-VL's
`pixel_values_videos` and `video_grid_thw` inputs, and GenieX v0.3.16's public
Qwen3-VL QAIRT image frontend hardcodes a temporal grid of one. The
repository's custom runner bypasses that image-path frontend and uses
GenieX's lower-level `PixelData` API, but it still expects an already decoded
and packed raw tensor. It is not an MP4 decoder or a native multi-clip video
frontend.

Decode and sample video outside the runtime, then use the official local
Hugging Face processor to pack consecutive frame pairs as Qwen3-VL temporal
patches. The current 224 × 384 graph contract for each pair is:

```text
2 RGB frames
  -> native HF pixel_values shape [336, 1536]
  -> video_grid_thw [1, 14, 24]
  -> selected bias-corrected vision context on QnnHtp
     (r5 leader: W8 weights, FP16 internals, nine A16 boundaries)
  -> image_features + deepstack_visual_embeds_0..2
  -> 84 visual tokens with grid [1, 14, 24]
  -> timestamp + <|video_pad|> span
  -> visual_pos_masks + full-interface W4/FP16 text part 1
  -> coherent W4/FP16 text parts 2-4 on QnnHtp
```

With `temporal_patch_size = 2`, two frames produce one temporal grid plane
(`T = 1`), so motion within each pair enters the vision patch embedding. The
runner can interleave multiple pairs with increasing timestamps and
timestamp-aware MRoPE. It still does not accept an encoded video container or
camera stream.

Three input-parity fixes are essential. The adapter must retain the learned
1024-element bias when converting Qwen3-VL's Conv3d patch projection to
Conv2d; frame preparation must let the native Hugging Face video processor
perform its resize; and the text sequence must use `<|video_pad|>`, token
151656. Exact per-pair hashes now match the BF16 GPU inputs.

The earlier legacy-Genie compatibility bundle cannot exercise full
DeepStack. Its transform zeros the three intermediate vision additions,
prunes their branches, and removes `visual_pos_masks` plus
`deepstack_visual_embeds_0..2` from part 1. The GenieX experiment below uses a
different part 1 whose full auxiliary interface is restored.

### Prepare a native-aligned paired-calibration candidate

Do not calibrate temporal inputs by duplicating one still frame. Extract the
pinned warehouse cases, create 20 distinct motion pairs, and quantize the
224 × 384 vision profile:

```bash
COSMOS_CALIBRATION="$COSMOS_WORK/calibration/warehouse_motion_pairs"
COSMOS_VISION_CHECKPOINT="$COSMOS_WORK/checkpoints/vision-224x384-paired"

python scripts/fetch_nvidia_sdg_warehouse.py \
  --output "$COSMOS_WORK/benchmarks/nvidia_sdg_warehouse" \
  --asset-set clips \
  --extract-rgb

python scripts/prepare_vision_calibration_pairs.py \
  --asset-root "$COSMOS_WORK/benchmarks/nvidia_sdg_warehouse" \
  --output-dir "$COSMOS_CALIBRATION"

CONTEXT_LENGTH=512 \
IMAGE_HEIGHT=224 \
IMAGE_WIDTH=384 \
VEG_NUM_SAMPLES=20 \
VEG_PAIRED_CALIBRATION_MANIFEST="$COSMOS_CALIBRATION/paired_calibration_manifest.json" \
bash scripts/quantize_wsl.sh \
  "$COSMOS_OFFICIAL" \
  "$COSMOS_VISION_CHECKPOINT"
```

The historical r3/r4 integer bundle and all four compiled r5 vision candidates
derive from the same older paired/explicit-resize calibration checkpoint.
Runtime input preparation now uses native Hugging Face processing and exact
GPU/NPU pixel bytes match, but do not describe the r5 sweep as
native-HF-aligned calibration. The code above produces a distinct checkpoint;
it requires its own AI Hub lineage and physical-board comparison before it
can resolve that remaining calibration question.

For the historical host-only block-23 candidate, add
`VEG_FP16_LAST_BLOCK_ACTIVATIONS=1`. Host QuantSim raises the primary
`image_features` cosine against the corrected adapted BF16 reference from
0.950431 to 0.991453 by leaving the 36 activation quantizers in final vision
block 23 as FP16. The r5 sweep instead tests the broader fail-closed precision
layouts from Step 6; its frozen-suite results supersede host cosine as the
deployment selection criterion.

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
COSMOS_VISION_EXPORT=/path/to/extracted/authorized-224x384-vision-export
COSMOS_DEEPSTACK_BUNDLE="$COSMOS_WORK/exports/full-deepstack-224x384"

python scripts/prepare_geniex_bundle.py \
  --source "$COSMOS_LEGACY_BUNDLE" \
  --destination "$COSMOS_DEEPSTACK_BUNDLE" \
  --model-id qwen3_vl_cosmos_reason2_2b_aspect \
  --hardlink \
  --part1-replacement-bundle "$COSMOS_PART1_EXPORT" \
  --vision-replacement-bundle "$COSMOS_VISION_EXPORT"
```

The assembler verifies that the replacement part 1 has all four auxiliary
inputs in a safe graph order, that its base inputs and outputs match the
coherent source part 1, and that the source vision graph supplies all three
matching DeepStack outputs. It also verifies that the replacement vision
metadata, fixed tensors, samples, and context all agree on 224 × 384 and 84
visual tokens. It retains the known-good W4/FP16 parts 2-4, merges the
replacement metadata, and removes the misleading legacy compatibility marker.
The measured full-interface part-1 link was
[`j5m8x4ndp`](https://workbench.aihub.qualcomm.com/jobs/j5m8x4ndp/), target
`mnj67lgdq`; `part1_of_4.bin` has SHA-256
`7a59e6c83456cc93de4c84b9dbeb3acb9ef78b93c3f88eadfd33405eeded0d81`.
The evaluated 224 × 384 integer vision link was
[`j5w46xkjg`](https://workbench.aihub.qualcomm.com/jobs/j5w46xkjg/), target
`mn0ke303q`. Do not reuse the earlier accidental default-512 export for this
profile.

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
COSMOS_VIDEO_CASE="$COSMOS_WORK/video_inputs/predict_near_miss_aspect"

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
Prepared paired video inputs: .../video_inputs/predict_near_miss_aspect
Context: 133/512 prompt tokens; 379 remain
```

The output contains the packed raw tensor, copies of the fixed positional and
attention inputs, timestamped text chunks, `genie-video-app-script.txt`, and
`video_npu_manifest.json`. The manifest records source-frame, processor,
bundle, tensor, text, and script hashes. The command refuses to overwrite an
existing output directory; use a new run suffix when preserving evidence.
The full-DeepStack runner consumes the packed
`sample_inputs/pair_000_pixel_values.raw` file; it does not execute the
generated legacy-Genie script.

The aspect profile accepts exactly `2N` frame paths and timestamps and emits
one 84-token span for every pair. The nominal CL512 context is not the only
limit: AR128 prefill can safely retain only 384 prompt tokens for the AR1
decode cache. Three-pair prompts in the r3 runs fit at 316–348 tokens; the
current three-pair benchmark and diagnostics span 313–360 tokens. Four-pair
prompts at 408 and 413 tokens are rejected.

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
part-1 input contract, AR128/CL512 prefill capacity, Qwen3-VL MRoPE settings,
every raw tensor, and the metadata-derived 84-token-per-pair layout. It refuses
to overwrite an existing output directory.

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
`PixelData` runner. The runner inserts exactly 84 `<|video_pad|>` tokens per
pair, passes grid `[1, 14, 24]`, obtains `image_features` and all three
intermediate vision outputs, constructs `visual_pos_masks`, and injects those
features through the full part-1 interface. It independently stops before
inference if the complete prompt exceeds 384 tokens.

For every additional case, reuse the build and model bundle but create a new
input directory, package, and remote run directory. The scripts intentionally
refuse to overwrite evidence. The benchmark manifest provides the exact
ordered frames and timestamps for the one- and three-pair cases.

### Interpret the measured results

Every recorded run initializes the vision and four text contexts
through `QnnHtp` on Hexagon v73. Exact native-processor pixel hashes match the
BF16 inputs. Functional execution and input parity are therefore established
separately from answer quality.

For the historical r3 one-pair suite, the NPU passes 0/5 strict units that the BF16 GPU reference
passes:

| Gate | BF16 GPU | NPU | Verdict |
|---|---|---|---|
| Shelf collision, free form | Predicts collision with yellow/black marker | Predicts forklift tipping | Fail |
| Near miss, choice | B | C | Fail |
| Shelf normal/shuffled choices | A → B | B → A | Fail |
| Observed near-miss avoidance | Worker steps back/forward | Worker jumps over forklift | Fail |
| Routine box pickup | Carrying box; explicitly no accident | Carrying box only | Fail |

The historical r3 three-pair control uses one shared three-choice answer set for
two videos, then shuffles it identically:

| Video | Order | Correct | BF16 GPU | NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Barrier knockdown | Normal | A | A | A | 730.5 ms |
| Barrier knockdown | Shuffled | C | C | C | 739.8 ms |
| Routine box pickup | Normal | B | B | B | 745.8 ms |
| Routine box pickup | Shuffled | A | A | A | 736.0 ms |

The NPU matches BF16 in all four r3 cases and changes its answer with both the
video and option order. That result is valid for the controlled two-scene
test, but the expanded r4 suite shows it does not generalize.

The historical r4 suite uses the same four choices for marker knockdown, routine box
pickup, near-miss avoidance, and two workers leaving aisles, then moves every
correct label in the shuffled order:

| Video | Order | Correct | BF16 GPU | NPU | Prompt tokens | NPU TTFT |
|---|---|---:|---:|---:|---:|---:|
| Barrier knockdown | Normal | A | A | A | 360 | 743.5 ms |
| Barrier knockdown | Shuffled | C | C | B | 360 | 733.9 ms |
| Routine box pickup | Normal | B | B | C | 357 | 731.3 ms |
| Routine box pickup | Shuffled | D | D | B | 357 | 742.9 ms |
| Near-miss avoidance | Normal | C | C | C | 357 | 750.7 ms |
| Near-miss avoidance | Shuffled | B | B | B | 357 | 740.8 ms |
| Two workers leave aisles | Normal | D | C | C | 357 | 744.7 ms |
| Two workers leave aisles | Shuffled | A | A | A | 357 | 732.0 ms |

BF16 GPU scores 7/8, NPU scores 4/8, and exact answers agree in 5/8
cases. Across the first three unambiguous scenes, GPU scores 6/6 while NPU
scores 3/6. Near-miss avoidance is the strongest positive result: GPU and NPU
both follow its correct label from `C` to `B`, 2/2.

Every row uses byte-identical GPU/NPU packed pixels and has the same prompt
count on both devices. The fire-scenario gate scores only the directly visible
motion of two workers leaving their earlier aisle positions; recognizing the
official fire cause is not required. Both devices choose the wrong `C` in
normal order, so parity there is not accuracy. Broad GPU/NPU video parity
remains unsolved.

A compact-prompt diagnostic removes 25 tokens from each question while
keeping the same four labels, scenes, pixels, and shuffled ordering:

| Video | Order | Correct | BF16 GPU | NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Barrier knockdown | Normal | A | C | C | 733.6 ms |
| Barrier knockdown | Shuffled | C | B | B | 733.8 ms |
| Routine box pickup | Normal | B | B | C | 741.8 ms |
| Routine box pickup | Shuffled | D | D | B | 739.4 ms |
| Near-miss avoidance | Normal | C | C | C | 739.7 ms |
| Near-miss avoidance | Shuffled | B | B | B | 731.1 ms |
| Two workers leave aisles | Normal | D | B | D | 732.6 ms |
| Two workers leave aisles | Shuffled | A | A | A | 734.3 ms |

GPU scores 5/8, NPU remains 4/8, and exact answer parity remains 5/8.
Shorter wording moves the marker and fire errors rather than closing the gap.
The NPU still uniquely fails both box-pickup orders against correct GPU
answers, making prompt verbosity an insufficient explanation.

A focused 327-token box-versus-near-miss diagnostic removes the marker and
worker-motion distractors:

| Video | Choice order | Correct | BF16 GPU | NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Routine box pickup | Box A / near B | A | A | A | 741.9 ms |
| Routine box pickup | Near A / box B | B | B | A | 732.4 ms |
| Near-miss avoidance | Box A / near B | B | B | B | 739.9 ms |
| Near-miss avoidance | Near A / box B | A | A | A | 741.1 ms |

GPU scores 4/4, NPU 3/4, and exact answer parity is 3/4. The NPU can
distinguish the box scene when box is listed first, but retains `A` after the
box label moves to `B`. This is order sensitivity rather than total visual
confusion, and it still falls short of robust box recognition.

A separate tailored barrier choice fails, and both three-pair free-form
rubrics fail on NPU. The BF16 GPU diagnostics also fail those two strict
free-form rubrics.

#### R5 frozen-suite result

The r5 comparison freezes the eight primary prompts, eight compact prompts,
and four focused box/near-miss prompts. The BF16 GPU reference scores 16/20.
Every NPU row below uses the same truth labels and GPU outputs:

| Candidate | NPU correct | Exact GPU parity | GPU-correct retained | Mean TTFT |
|---|---:|---:|---:|---:|
| Native-aspect W8/A16 baseline | 11/20 | 13/20 | 10/16 | 738.070 ms |
| Boundary-FP16 vision | **13/20** | **15/20** | **12/16** | 857.510 ms |
| W-FP16/A16 vision | 11/20 | 13/20 | 10/16 | 772.825 ms |
| Combined W-FP16 + boundary-FP16 | 12/20 | 14/20 | 11/16 | 686.080 ms |
| Boundary-FP16 + W8 text part 4 | 12/20 | 13/20 | 11/16 | 873.5 ms on 3/20 logs |

Boundary-FP16 is the best completed NPU result: 13/20 versus GPU 16/20.
Its 15/20 exact-letter parity is higher than its benchmark accuracy because
GPU and NPU agree on some wrong answers. Conversely, the NPU has one correct
answer where GPU is wrong, so `NPU correct`, `exact parity`, and
`GPU-correct retained` intentionally measure different things.

The part-4 timing mean is incomplete because only three retained logs contain
the required verbose TTFT field. Its 12/20 accuracy is complete. Do not use
that timing mean as a 20-case latency comparison.

Score the retained logs without loading Torch, AIMET, or a model runtime:

```bash
python scripts/score_video_npu_results.py \
  --gpu-results /path/to/gpu-primary \
  --gpu-results /path/to/gpu-compact \
  --gpu-results /path/to/gpu-pairwise \
  --npu-results baseline=/path/to/npu-baseline \
  --npu-results boundary=/path/to/npu-boundary \
  --npu-results wfp16_a16=/path/to/npu-wfp16-a16 \
  --npu-results combined=/path/to/npu-combined \
  --npu-results w8_part4=/path/to/npu-w8-part4 \
  --output /path/to/new-r5-comparison.json \
  --require-complete
```

The part-1 layers 0–6, 0–3, and 0–2 candidates are compiled but have no
frozen-suite result outside the resumed GenieX GGUF pilot. Full layers 0–6
passed a four-token HTP smoke; the two narrower variants still await EVK
smoke. The balanced P1–P4 GPU extension scores 13/16 overall (6/8 on the new
P3/P4 half), but there is no balanced NPU score yet.

Hosted shard screens provide a more precise diagnosis of the text path. Each
screen reuses the exact AI Hub dataset produced by its quantized predecessor,
then compares that one IQ9075 shard with an eager BF16 execution of the same
layers and inputs. Recomputing the selected decoder matrices at W8 instead of
W4 produces:

| Decoder partition | Layers | Metric | W4/FP16 baseline | W8 matrices | Error reduction |
|---|---:|---|---:|---:|---:|
| Part 1 | 0–6 | Hidden relative RMSE | 0.08099 | 0.01296 | 6.25× |
| Part 1 | 0–6 | Mean KV relative RMSE | 0.19134 | 0.01514 | 12.64× |
| Part 2 | 7–13 | Hidden relative RMSE | 0.02559 | 0.00381 | 6.72× |
| Part 2 | 7–13 | Mean KV relative RMSE | 0.21512 | 0.01506 | 14.28× |
| Part 3 | 14–20 | Hidden relative RMSE | 0.10518 | 0.00828 | 12.71× |
| Part 3 | 14–20 | Mean KV relative RMSE | 0.28143 | 0.01948 | 14.44× |

The part-2 and part-3 result is consistent for all three tested predecessor
lanes: baseline part 1, W8 layers 0–6, and W8 layers 0–3. This strongly
implicates W4 decoder-matrix error and justifies promoting those contexts to
the final-choice experiment. It still does **not** establish a better answer:
an intermediate hidden/KV comparison can improve while a small final-logit
margin changes in the wrong direction. The hosted first-token chains
therefore replay all three AR128 chunks through all four decoder partitions
before accepting or rejecting a candidate.

<!-- R5_PENDING_WINNER_UPDATE:
Replace the pending part-1 and balanced-NPU statements only after complete,
hash-bound EVK logs have been scored. Do not infer a winner from smoke output.
Mirror the selected result in README.md, docs/architecture.md,
docs/video_npu.md, benchmarks/nvidia_sdg_warehouse/README.md,
integrations/geniex_raw_video/README.md, and docs/evidence/README.md.
-->

The sanitized answers, hashes, prompt counts, pass rules, and proof boundary
are in the
[`native-aspect parity report`](evidence/iq9075_video_aspect_native_parity_r3.json)
and
[`expanded four-scene report`](evidence/iq9075_video_four_scene_parity_r4.json).
The r5 artifact lineage, completed scores, and pending status are in the
[`precision report`](evidence/iq9075_video_precision_parity_r5.json).

### Scale beyond one pair

The runner and manifest already support multiple timestamped pairs. The next
steps are numerical and contextual:

1. When the EVK returns, run the hash-bound frozen suite for part-1 W8
   layers 0–6, 0–3, and 0–2, then score them against the same GPU artifacts.
2. Run the balanced P1–P4 probes on the selected completed NPU bundle. Do not
   infer its NPU score from the 13/16 GPU baseline.
3. Export a longer-context text runtime whose AR prefill layout safely covers
   four-pair prompts. Four 224 × 384 pairs use 336 visual tokens, but the
   measured total prompts of 408 and 413 exceed the current safe limit of 384.
4. Extend the four-scene order control with additional unambiguous negative
   and motion cases, and retain strict free-form event scoring rather than
   accepting generic hazard words.

Keep encoded-video decoding and native-Qwen3-VL frame/timestamp preparation
outside GenieX unless a future CLI exposes a documented video input with
matching semantics. Multiple image paths are useful for experiments but do
not by themselves reproduce the native 2-FPS, per-temporal-pair timestamp
contract. The custom lower-API runner bypasses the stock image frontend and
supplies bounded multi-pair semantics; it is not an MP4 decoder,
camera-stream pipeline, or unbounded video frontend. See
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
| GenieX imports the local directory but selects the wrong main GGUF | The staging directory also contains the intermediate BF16 main model | Keep only `Cosmos-Reason2-2B-Q4_0.gguf` and `mmproj-Cosmos-Reason2-2B-F16.gguf` in the imported directory |
| `--temperature 0` still samples | GenieX treats zero as an unset sampler field and substitutes temperature `0.8` | Use `--top-k 1 --seed 42` for deterministic greedy-equivalent comparisons |
| Multiple GenieX images disagree with the native video reference | Ordered still-image attachment does not reproduce native video sampling and timestamps | Decode at 2 FPS and use the custom per-pair pipeline for parity tests; treat the GenieX multi-image result as exploratory |
| Q4_0 repeats one option across shuffled image choices | The visual budget may be too small; the host barrier probe repeated `C` at 84 tokens/image | Re-run natural and deterministically upscaled inputs with a larger context; require correctness in both option orders and more than one scene |
| QAIRT 2.45 and 2.47 libraries appear in one run | An inherited environment mixed installations | Rebuild a clean `PATH`/`LD_LIBRARY_PATH` rooted only at the artifact's matching QAIRT release |
| Process is killed on the EVK | Memory pressure; the board has no swap | Start with context 512, check `free -h`, and inspect kernel/OOM logs |
| Text works but current aspect vision fails | Wrong vision target, profile metadata, patch bias, or image connection | Verify the 224 × 384 target, vision SHA-256 prefix `7af841…`, `[84, 2048]` outputs, and learned patch-projection bias before inspecting connections |
| Process exits zero but no HTP evidence exists | Backend was not proven | Capture verbose runtime evidence and do not label the result an NPU success |
| Genie reports `Unsupported input tensor full_attention_mask dtype QNN_DATATYPE_FLOAT_32` | The globally-W4 vision target `mnzgp1ydq` was assembled into a legacy-Genie bundle | Replace it with original W4A16 vision target `mngx5gv5q`; direct `qnn-net-run` success does not remove the Genie interface mismatch |
| Genie initializes HTP contexts but emits incoherent text | The original all-W4A16 text contexts were used instead of the proven W4/FP16 links | Verify the four text target IDs and checksum prefixes from Step 7; preserve the bad run only as historical drift evidence |
| Legacy QAIRT 2.45 Genie rejects wildcard or auxiliary DeepStack connections | Legacy Genie cannot bind the full Qwen3-VL decoder interface | For the baseline, export the compatibility checkpoint and record that DeepStack is disabled; for the full interface, use the validated replacement part 1 and pinned GenieX runner |
| Linked graph order differs between text parts | Link inputs were supplied in a different order or AI Hub reused stale model identities | Inspect every context. For the r5 full-W8 part 1, use cache-busted upload identities `mm55gov6m` then `mn46jeo0q`, link `jpv7k21kp`, target `mn7yvo34n`, SHA-256 `8ea7ba19…`, and require AR128→AR1 |
| Ordered text context fails or exhausts memory on decode | The link did not share weights even though its graph order is correct | Require equal nonzero `sharedWeightsSize` for both graphs; reject the measured 711,176,192-byte no-sharing part 1 that fails its 490,733,568-byte FastRPC mapping |
| `prepare_video_npu_inputs.py` rejects an odd frame count | Temporal patches require exactly two frames | Pass exactly `2N` ordered frames and `2N` increasing timestamps |
| Video prompt exceeds 384 tokens despite fitting CL512 | AR128 prefill can safely transfer only `CL - AR = 384` tokens into AR1 decode | Reduce to three pairs or compile a longer-context text runtime; do not bypass the preparation/package/runner guards |
| Direct MP4 through GenieX is undocumented or unstable | The generic-media fallback can reach mtmd video, but the wrapper drops the video-context owner and its sampling/timestamps differ from native Qwen3-VL | Decode and sample frames first; use ordered images for the GGUF pilot or `prepare_video_npu_inputs.py` plus the custom raw runner for native-pair parity |
| Full-DeepStack run exits 0 but prediction is wrong | Functional execution, DeepStack wiring, and exact inputs do not imply retained quantized accuracy | Compare every output with BF16 and score every precision candidate on the same frozen probes; the completed r5 boundary-FP16 leader is 13/20 versus GPU 16/20 |

## What to save for a reproducible result

Keep these together for every run:

- Git commit of this repository.
- QAI Hub Models version (`0.58.0`).
- Official checkpoint revision and `config.json`.
- Quantization arguments and calibration sample counts.
- Checksums of ONNX and encoding files.
- Precision-transform marker files and source/destination hashes.
- `qairt_245_compat.json`, including the removed inputs and source/output ONNX
  checksums.
- AI Hub compile/link job URLs and selected QAIRT version.
- AR128/AR1 upload identities, graph order, and per-graph shared-weight size.
- Exported bundle checksum.
- Checksums and graph names for all deployed context binaries.
- EVK QAIRT path and version.
- GenieX CLI version, selected `llama_cpp` compute mode, llama.cpp conversion
  commit, GGUF/projector checksums, and backend evidence.
- Full text and vision logs.
- Profile output with prompt rate, decode rate, and time to first token.
- Test prompt, image, context length, and random/temperature settings.
- The exact GPU result roots, NPU result roots, benchmark hash, and generated
  `score_video_npu_results.py` comparison.
- Captured QuantSim inputs, isolated-part outputs, native chained boundary
  dumps, and both first-token comparisons.

The 512-context hybrid text and vision smoke tests now pass. Preserve their
artifact IDs and profiles before repeating the two-export assembly at context
2048 or 4096.
