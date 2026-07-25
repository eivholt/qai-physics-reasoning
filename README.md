# Cosmos-Reason2-2B on Qualcomm IQ-9075

This repository contains an experimental Qualcomm AI Hub Models adapter and
deployment scripts for running NVIDIA's `Cosmos-Reason2-2B` on a Dragonwing
IQ-9075 EVK (`QCS9075`, Hexagon v73).

`Cosmos-Reason2-2B` is a 2.44-billion-parameter vision-language model for
physical AI. Given an image or short video observation plus a prompt, it can
describe a scene, identify hazards, reason about motion and object
interactions, and predict a likely outcome in text. It is an interesting edge
target because a robot can keep camera data local and avoid network latency
while using an NPU for predictable, power-efficient inference. This port is
experimental: successful NPU execution is not the same as a correct or
safety-qualified prediction.

The current full-video path is:

```text
nvidia/Cosmos-Reason2-2B (BF16)
  -> Qwen3-VL-2B-compatible QAI Hub Models wrapper
     `-> preserve Qwen3-VL's learned temporal patch-projection bias
  -> 224 x 384 paired-frame vision calibration
  -> W8 vision with FP16 internals/A16 boundaries
     + W4-weight/FP16-activation text contexts
  -> full DeepStack GenieX PixelData runner
  -> native Hugging Face pixels + <|video_pad|> + timestamped MRoPE
  -> QAIRT 2.45 / QnnHtp on the IQ-9075 NPU
```

The earlier legacy-Genie compatibility path remains useful for text and
single-image bring-up, but it removes the DeepStack and visual-mask inputs and
is no longer the parity target for video.

The adapter is based on the supported Qwen3-VL-4B implementation in Qualcomm
AI Hub Models 0.58.0, with the exact Qwen3-VL-2B dimensions:

| Property | Cosmos-Reason2-2B |
|---|---:|
| Text layers | 28 |
| Hidden size | 2048 |
| Attention / KV heads | 16 / 8 |
| Head dimension | 128 |
| Vision depth / width | 24 / 1024 |
| Text splits | 4 x 7 layers |
| Current completed accuracy leader | Bias-corrected W8 vision, FP16 internals, A16 boundaries + W4/FP16 text |

In plain language, the 24 vision-transformer layers turn image patches into
contextual visual features, then project each feature from width 1024 into the
text model's width 2048. The 28 text-transformer layers combine those visual
tokens with the prompt: 16 attention heads relate positions and objects while
eight shared key/value heads reduce cache traffic, and each layer's
feed-forward network further transforms the result. Three DeepStack
connections can inject intermediate vision features into selected text
layers. Splitting the text tower into four seven-layer NPU graphs is a
deployment technique; the four graphs still form one decoder and must hand
off numerically sound hidden states.

QAI Hub Models 0.58.0 does not ship a complete public QAIRT adapter for
Qwen3-VL-2B, so this specializes Qualcomm's shared Qwen3-VL implementation
using the 2B architecture. It must be validated on the physical board.

## Current result

- The CPU GGUF vision path produces coherent output on the EVK.
- Official GenieX v0.3.17 now runs the exact Q4_0/F16 Cosmos GGUF bundle
  through the IQ-9075 Hexagon NPU. Live-process inspection confirms the
  Hexagon library and secure CDSP FastRPC device are active. A grounded
  six-frame forklift/marker prediction agrees with CPU while reducing TTFT
  from 7.339 to 1.754 seconds. The stock exact long-prompt panel scores 5/8.
  A component ablation localizes its one same-GGUF CPU/NPU regression to
  behavior affected by Hexagon execution of the vision `mmproj` context, not
  the final output layer. Two corrected profiles are now measured. Unmodified
  GenieX with `GGML_HEXAGON_OPFILTER=GELU` scores 6/8, exactly matches the
  all-CPU letters, and averages 2.146 seconds TTFT. The quality-first profile
  keeps the vision encoder/projector on CPU while all decoder layers remain on
  NPU; it scores 7/8, matches all eight recorded BF16 GPU letters, and
  averages 6.402 seconds TTFT versus 7.688 seconds all-CPU. The shorter fixed
  prompt also gives stock NPU and BF16 GPU 7/8 with eight-answer parity, while
  a narrower task-specific profile reaches 4/4.
- The patched v0.3.17 service now accepts H.264 MP4 through ffmpeg/mtmd,
  pairs adjacent frames temporally, and keeps the model and HTP backend loaded
  between independent requests. The new 2 FPS profile requests NPU placement
  for the vision encoder/projector and all 28 decoder layers. It scores 7/8 on the
  frozen encoded-video panel with the same answer sequence as the r9 hybrid,
  while reducing warm inference from about 7.0 seconds to a 1.453-second
  mean. The task-specific two-request warehouse classifier reaches 4/4 at a
  2.90-second warm mean, although a balanced direct four-choice diagnostic is
  only 10/16. A 48-request soak leaves 26 file descriptors and no decoder
  children; an extended run stalls after 54 completed requests, so the worker
  currently needs conservative supervised recycling. This is bounded project
  evidence, not tensor equivalence, broad GPU-equivalent accuracy, or safety
  qualification.
- Historical r1/r2 bring-up: the first all-W4A16 bundle compiles, links, and
  executes on QnnHtp, but its
  four-part text decoder is numerically incorrect. Bundled Genie 1.17 and
  official GenieX v0.3.16 both reproduce that artifact's incoherent output;
  GenieX does not repair that old all-W4A16 artifact.
- The first derived checkpoint changed all 10,357 text and vision activation
  encodings to FP16. Four host CUDA text QuantSim sessions coherently return
  `Hello,`, and the vision graph compiles, but QAIRT 2.45 fails while linking
  that all-FP16 vision graph. This remains useful failure history, not the
  recommended export.
- A globally-W4 mixed vision export also compiled and linked, and direct
  `qnn-net-run` executes it on HTP in about 250 ms. Its external inputs and
  outputs are FP32, however, so legacy Genie 1.17 rejects
  `full_attention_mask` with `QNN_DATATYPE_FLOAT_32`. Do not use that vision
  graph with legacy Genie.
- The historical legacy-Genie bundle combines the original W4A16 vision
  context with the four W4/FP16 text contexts. Text-only legacy Genie produces
  coherent English at 131.152 ms TTFT, 289.7711 prompt tok/s, and 17.9982
  generation tok/s. The default `genie-app` run exits 0, accepts all five
  image inputs, and describes Qualcomm's actual `dog.jpg` sample as a white
  fluffy dog on green grass.
  Genie dequantizes the quantized `image_features` into its accumulator, so
  the original W4A16 vision context interoperates with the FP16 text path.
- A conversion audit found that the Qwen3-VL Conv3d patch projection's learned
  1024-element bias was dropped when it was adapted to Conv2d. The adapter now
  preserves that bias. The NPU input path also uses the native Hugging Face
  video resize and the upstream `<|video_pad|>` token; byte comparisons
  confirm that the GPU and NPU receive identical packed pixels.
- The raw QAIRT/DeepStack profile still predecodes video outside the runtime.
  It packs `2N` frames as `[336, 1536]` temporal patches with grid
  `[1, 14, 24]` and produces 84 visual tokens per pair at 224 × 384. Its 20-sample
  calibration set contains distinct warehouse frame pairs rather than
  duplicated stills. The compiled r5 candidates still derive from the older
  paired/explicit-resize calibration checkpoint. Runtime input preparation is
  native Hugging Face and byte-identical between GPU and NPU, but a
  native-HF-aligned calibration checkpoint was not the compiled source for
  this sweep. The separate patched GenieX GGUF route accepts encoded MP4
  directly, but its mtmd preprocessing is not byte-identical with this path.
- The full-video integration restores `visual_pos_masks` and
  `deepstack_visual_embeds_0..2`: it replaces only part 1 with a full-interface
  W4/FP16 context, retains coherent W4/FP16 parts 2-4, and uses a pinned
  GenieX lower-level `PixelData` runner. Exact-input one-pair inference passes
  0 of 5 strict units that the BF16 GPU reference passes.
- Historical r3/r4 video results: three temporal pairs fit the AR128/CL512
  text runtime safely and produce
  coherent NPU output. An initial two-scene, three-choice test matched BF16
  GPU in all four scene/order combinations, but that narrow 4/4 result does
  not generalize. In the expanded four-scene, four-choice suite, BF16 GPU
  scores 7/8, NPU scores 4/8, and their exact answers agree in 5/8 cases.
  Across the first three unambiguous scenes (marker, box, and near miss), GPU
  scores 6/6 while NPU scores 3/6.
- In the historical r4 expanded suite, the most robust realistic NPU result is
  the near-miss video: both GPU and NPU select the worker-dodge event in normal
  and shuffled order, 2/2. Every GPU/NPU case has identical native-processor
  packed pixels and the same prompt count, so the remaining disagreements are
  downstream of input preparation. Broad video-prediction parity remains
  unsolved.
- The historical r4 compact-prompt ablation removes 25 tokens from every
  four-choice prompt. It moves errors rather than fixing them: GPU scores 5/8,
  NPU remains 4/8, exact parity remains 5/8, and the NPU still uniquely fails
  both box-pickup orders against correct GPU answers.
- The historical r4 focused box-versus-near-miss control narrows that failure:
  GPU scores 4/4 and NPU 3/4. NPU recognizes box pickup when it is option `A`,
  but keeps `A` after the box moves to `B`; this is order sensitivity rather
  than complete visual confusion.
- Four pairs cost only 336 visual tokens but produce total prompts of 408 and
  413 tokens. Those exceed GenieX v0.3.16's safe AR128/CL512 prefill limit of
  384 and caused corrupted NPU text; preparation, packaging, and the runner
  now reject such prompts. The corresponding BF16 GPU free-form diagnostics
  also fail their strict event rubrics, although their text remains coherent.

The completed r5 precision sweep uses a frozen 20-probe set: eight primary
four-choice prompts, eight compact prompts, and four focused box/near-miss
prompts. The BF16 GPU reference scores 16/20. The current NPU accuracy leader
keeps nine graph boundaries at A16 and 925 internal vision activations at
FP16:

| r5 vision/text candidate | NPU correct | Exact GPU parity | GPU-correct retained | Mean TTFT |
|---|---:|---:|---:|---:|
| Native-aspect W8/A16 baseline | 11/20 | 13/20 | 10/16 | 738.070 ms |
| W8 with FP16 internal activations and A16 boundaries | **13/20** | **15/20** | **12/16** | 857.510 ms |
| FP16 vision weights with A16 activations | 11/20 | 13/20 | 10/16 | 772.825 ms |
| FP16 vision weights plus FP16 internals/A16 boundaries | 12/20 | 14/20 | 11/16 | 686.080 ms |
| Boundary-FP16 vision plus W8 text part 4 | 12/20 | 13/20 | 11/16 | 873.5 ms on 3/20 timing-bearing logs |

Thus the best completed NPU run is three correctness points behind GPU
(65% versus 80%), while matching 15 of 20 GPU letters. It is a real
improvement over the 11/20 baseline, but it is not GPU-equivalent. Vision
cosine similarity alone did not predict task accuracy: the combined
FP16-weight candidate is numerically closer on the host yet scores one point
below the boundary-only candidate.

Three early-decoder W8 candidates are compiled but not yet scored. The full
part-1 candidate (layers 0–6, 252 matrices) passed a four-token HTP smoke with
AR128→AR1 shared weights; the layers 0–3 and layers 0–2 variants remain
pending EVK smoke and frozen-suite execution outside the resumed GenieX GGUF
pilot. These are pending results, not inferred improvements. A balanced
P1–P4 GPU extension scores 13/16 overall (the new P3/P4 half is 6/8); its NPU
run is also pending.

For the verified export and deployment workflow, see the
[developer tutorial](docs/developer_tutorial.md). For measured job IDs,
precision experiments, native-chain diagnosis, and the failure guide, see the
[engineering appendix](docs/engineering_appendix.md). See
[video input on the NPU](docs/video_npu.md) for the temporal contract,
measured warehouse results, and scaling plan. The
[sanitized evidence record](docs/evidence/README.md) defines the exact
physical-board proof boundary for the
[legacy paired-frame report](docs/evidence/iq9075_video_smoke_r1.json),
[full-DeepStack GenieX report](docs/evidence/iq9075_video_deepstack_geniex_r2.json),
[native-aspect parity report](docs/evidence/iq9075_video_aspect_native_parity_r3.json),
[expanded four-scene report](docs/evidence/iq9075_video_four_scene_parity_r4.json),
[r5 precision report](docs/evidence/iq9075_video_precision_parity_r5.json),
[r6 GenieX GGUF report](docs/evidence/iq9075_geniex_gguf_r6.json),
[r7 CPU/NPU isolation report](docs/evidence/iq9075_geniex_cpu_npu_parity_r7.json),
and
[r8 vision-placement report](docs/evidence/iq9075_geniex_vision_placement_r8.json).
The patched service is recorded in the
[r9 encoded-video report](docs/evidence/iq9075_geniex_native_video_r9.json);
the preferred 2 FPS full-NPU profile is recorded in the
[r10 full-NPU video report](docs/evidence/iq9075_geniex_full_npu_video_r10.json).

No earlier independent public proof was found for the exact Cosmos-Reason2-2B
checkpoint on IQ-9075. NVIDIA's
[published validation list](https://docs.nvidia.com/cosmos/latest/prerequisites.html)
excludes Qualcomm, while Qualcomm validates the related
[Qwen3-VL-2B-Instruct GGUF](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct?runtime=geniex_qairt%2Cgeniex_llamacpp)
configuration for IQ-9075. This repository is therefore an experimental
project result, not NVIDIA or Qualcomm certification.

## Current prerequisites

- WSL2 with a CUDA-capable NVIDIA GPU and at least 40 GB host RAM.
- Miniconda at `~/miniconda3`.
- A Qualcomm AI Hub Workbench API token configured with `qai-hub configure`.
- Hugging Face access to the gated
  [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
  repository.
- QAIRT 2.45.0.260326 on the target EVK. This matches the current default
  compiler offered by QAI Hub Workbench.

The NVIDIA checkpoint requires accepting the NVIDIA Open Model License and
sharing the Hugging Face account contact information with NVIDIA. The scripts
do not bypass that gate.

## 1. Create the WSL environment

From WSL:

```bash
cd /mnt/c/path/to/qai-physics-reasoning
bash scripts/setup_wsl_env.sh
```

This installs the adapter as an additional
`qai_hub_models.models.cosmos_reason2_2b` package without modifying the
upstream QAI Hub Models source.

## 2. Authenticate and download the checkpoint

Accept the model terms in a browser, then authenticate the WSL CLI. For a
process-scoped token that is not written to disk:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qai-cosmos-reason2
read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN

bash scripts/download_checkpoint.sh
unset HF_TOKEN
```

`hf auth login` is also supported if persistent CLI authentication is
preferred.

The default checkpoint location is:

```text
~/models/Cosmos-Reason2-2B
```

## 3. Quantize to W4A16

Start with a 512-token smoke checkpoint:

```bash
CONTEXT_LENGTH=512 \
NUM_SAMPLES=8 \
VEG_NUM_SAMPLES=20 \
bash scripts/quantize_wsl.sh \
  ~/models/Cosmos-Reason2-2B \
  ~/cosmos-reason2-iq9/checkpoints/w4a16-cl512
```

For a higher-quality 2048-token calibration, use the defaults:

```bash
bash scripts/quantize_wsl.sh \
  ~/models/Cosmos-Reason2-2B \
  ~/cosmos-reason2-iq9/checkpoints/w4a16-cl2048
```

Quantization is a two-pass operation: the text decoder and the vision encoder
are calibrated separately. It can use substantial RAM, VRAM, and temporary
disk space. The finalized W4A16 directory records the original BF16 directory
in `source_checkpoint.json`; QAI Hub Models still needs that local snapshot
while preparing the vision encoder, tokenizer, processor, and bundle assets.

For the current native-aspect video experiment, first fetch the pinned
warehouse previews and build the distinct-pair calibration manifest, then
quantize at 224 × 384:

```bash
COSMOS_REPO=/path/to/qai-physics-reasoning
COSMOS_WORK="${COSMOS_WORK:-$HOME/cosmos-reason2-iq9}"

python "$COSMOS_REPO/scripts/fetch_nvidia_sdg_warehouse.py" \
  --output "$COSMOS_WORK/benchmarks/nvidia_sdg_warehouse" \
  --asset-set clips \
  --extract-rgb

python "$COSMOS_REPO/scripts/prepare_vision_calibration_pairs.py" \
  --asset-root "$COSMOS_WORK/benchmarks/nvidia_sdg_warehouse" \
  --output-dir "$COSMOS_WORK/calibration/warehouse_motion_pairs"

CONTEXT_LENGTH=512 \
IMAGE_HEIGHT=224 \
IMAGE_WIDTH=384 \
VEG_NUM_SAMPLES=20 \
VEG_PAIRED_CALIBRATION_MANIFEST="$COSMOS_WORK/calibration/warehouse_motion_pairs/paired_calibration_manifest.json" \
bash "$COSMOS_REPO/scripts/quantize_wsl.sh" \
  "$HOME/models/Cosmos-Reason2-2B" \
  "$COSMOS_WORK/checkpoints/vision-224x384-paired"
```

The historical block-23 host candidate additionally set
`VEG_FP16_LAST_BLOCK_ACTIVATIONS=1`. The r5 campaign subsequently compiled
and ran broader precision layouts; use the frozen-suite scores above rather
than host cosine alone to choose a deployment.

## 4. Prepare and compile for IQ-9075

Create the QAIRT 2.45 text-first interface checkpoint, then derive a W4/FP16
text checkpoint. The latter is an export source for the four text components
only; do not export its globally-W4 vision encoder for legacy Genie:

```bash
python scripts/make_qairt245_compat_checkpoint.py \
  ~/cosmos-reason2-iq9/checkpoints/w4a16-cl512 \
  ~/cosmos-reason2-iq9/checkpoints/w4a16-cl512-qairt245 \
  --num-visual-tokens 256

python scripts/make_w4_fp16_checkpoint.py \
  --source ~/cosmos-reason2-iq9/checkpoints/w4a16-cl512-qairt245 \
  --destination ~/cosmos-reason2-iq9/checkpoints/w4-fp16-text-w4a16-vision-cl512-qairt245 \
  --keep-vision-w4a16
```

Export the original W4A16 vision component and the W4/FP16 text components as
separate Workbench jobs. Confirm that the model license and your data policy
permit the external upload before running:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qai-cosmos-reason2

python -m qai_hub_models.models.cosmos_reason2_2b.export \
  --checkpoint ~/cosmos-reason2-iq9/checkpoints/w4a16-cl512-qairt245 \
  --target-runtime geniex_qairt \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --skip-profiling \
  --skip-downloading \
  --components vision_encoder

python -m qai_hub_models.models.cosmos_reason2_2b.export \
  --checkpoint ~/cosmos-reason2-iq9/checkpoints/w4-fp16-text-w4a16-vision-cl512-qairt245 \
  --target-runtime geniex_qairt \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --skip-profiling \
  --skip-downloading \
  --components part1_of_4 part2_of_4 part3_of_4 part4_of_4
```

Download the five linked context binaries and assemble them into one complete
release scaffold with `scripts/finalize_qairt245_bundle.py`. Replace all five
contexts explicitly: `vision_encoder.bin` comes from the original W4A16
export, while `part1_of_4.bin` through `part4_of_4.bin` come from the W4/FP16
text export.

Part 1 must be cache-busted and inspected before assembly. The proven link job
is `jprw31875`, target model `mno2p6jvq`, SHA-256 prefix `4fa761…`, with graph
order AR128 then AR1. Reusing a stale part-1 upload can silently restore the
wrong order. The tutorial records the corresponding part 2-4 and vision job
IDs and checksums.

The Workbench target is `Dragonwing IQ-9075 EVK` / Qualcomm Linux 1.7. The
measured smoke matrix uses prompt/decode sequence lengths 128 and 1.

### R5 precision candidates

The fail-closed transforms preserve the raw ONNX graph and record provenance:

```bash
python scripts/make_vision_a16_boundary_checkpoint.py \
  /path/to/all-fp16-vision \
  /path/to/native-w8a16 \
  /path/to/new-boundary-fp16

python scripts/make_vision_fp16_weight_checkpoint.py \
  /path/to/native-w8a16 \
  /path/to/new-wfp16-a16

python scripts/make_vision_fp16_weight_boundary_checkpoint.py \
  /path/to/new-wfp16-a16 \
  /path/to/new-boundary-fp16 \
  /path/to/new-wfp16-boundary

python scripts/make_text_w8_matrix_checkpoint.py \
  /path/to/w4-fp16-full-deepstack-cl512 \
  /path/to/new-w8-layers-0-3 \
  --parts part1_of_4 \
  --layers 0 1 2 3 \
  --image-size 224 384
```

Before compiling, compare vision checkpoints on the exact manifest-bound
pixels with `scripts/compare_vision_checkpoints.py`. After EVK execution,
score all candidates with one common GPU subset:

```bash
python scripts/score_video_npu_results.py \
  --gpu-results /path/to/gpu-primary \
  --gpu-results /path/to/gpu-compact \
  --gpu-results /path/to/gpu-pairwise \
  --npu-results baseline=/path/to/npu-baseline \
  --npu-results boundary=/path/to/npu-boundary \
  --output /path/to/new-comparison.json \
  --require-complete
```

For text parts, inspect every linked context before assembly. It must list
AR128 then AR1, and both graphs must report the same nonzero shared-weight
size. A locally linked no-sharing part 1 reached 711,176,192 bytes and failed
on the EVK while mapping a 490,733,568-byte FastRPC buffer, so it is rejected.
When Workbench reuses an upload identity and silently restores AR1→AR128,
create a semantically identical DLC with a different ZIP comment:

```bash
python scripts/cache_bust_dlc.py \
  --marker cosmos-part1-ar128-r5 \
  /path/to/source.dlc \
  /path/to/new-cache-busted.dlc
```

Upload the AR128 artifact before AR1, link those new model identities, then
reinspect and hash the result. Do not use cache busting to skip graph,
shared-weight, or physical-board validation.

## 5. Deploy and run

Copy the exported bundle to the EVK:

```powershell
.\scripts\deploy_evk.ps1 `
  -BundlePath "\\wsl.localhost\Ubuntu\home\<WSL user>\cosmos-reason2-iq9\exports\w4-fp16-text-w4a16-vision-hybrid-cl512-qairt245-r1" `
  -EvkIp "<EVK IP>" `
  -IdentityFile "C:\path\to\evk_ssh_key" `
  -RemoteDirectory "/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1"
```

This is the canonical inactive staging directory. The older active bundle at
`/home/ubuntu/cosmos_reason2_2b_qairt` remains untouched.

The target-side runner supports a text-only `genie-t2t-run` smoke test and
prints the runtime profile:

```bash
EVK_IP="<EVK IP>"
ssh "ubuntu@$EVK_IP"
bash ~/cosmos_reason2_2b_w4_fp16_hybrid_r1/run_text_smoke.sh
```

The bundle's default `genie-app-script.txt` provides the five image inputs for
the proven full vision run.

## 6. Prepare paired video frames

Encoded video is decoded and sampled outside GenieX. Fetch NVIDIA's
lightweight Isaac Sim warehouse previews, extract a case, and pack its ordered
frames with the local Qwen3-VL processor:

```bash
COSMOS_VIDEO_BUNDLE=/path/to/full-deepstack-aspect-bundle
COSMOS_WORK="${COSMOS_WORK:-$HOME/cosmos-reason2-iq9}"

python scripts/fetch_nvidia_sdg_warehouse.py \
  --asset-set clips \
  --extract-rgb \
  --case predict_near_miss

python scripts/prepare_video_npu_inputs.py \
  --bundle "$COSMOS_VIDEO_BUNDLE" \
  --processor "$HOME/models/Cosmos-Reason2-2B" \
  --output-dir "$COSMOS_WORK/video_inputs/predict_near_miss" \
  --frames \
    artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0030.png \
    artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0045.png \
  --timestamps 2.0 3.0 \
  --question "What is most likely to happen next, and is there an immediate safety risk? Answer in one short sentence."
```

The input preparer derives the image geometry from the bundle. For the current
224 × 384 graph, one pair is `[336, 1536]`, grid `[1, 14, 24]`, and 84 visual
tokens. It uses native Hugging Face preprocessing, and the GenieX runner
inserts `<|video_pad|>`. The AR128/CL512 runtime permits at most 384 prompt
tokens even though the nominal context is 512, so three short-prompt pairs
work while the measured four-pair prompts do not. Exact packaging, `scp`,
QAIRT environment, and execution commands are in the
[engineering appendix](docs/engineering_appendix.md#12-run-paired-video-frames-on-the-npu).

Use `scripts/geniex_raw_video.sh` to build, package, deploy, and run the pinned
lower-level GenieX `PixelData` integration with the full-interface W4/FP16
part-1 replacement. The tutorial records the current positive controlled
result and the remaining free-form failures without conflating either with
the pending mixed-vision candidate.

## CPU baseline

For bring-up independent of the custom exporter, the EVK can run a public
Q4_K_M GGUF plus its F16 multimodal projector with its existing
Qwen3-VL-capable llama.cpp build:

```bash
bash scripts/run_evk_cpu_baseline.sh \
  /home/ubuntu/models/cosmos_reason2_2b_gguf/Cosmos-Reason2-2B-Q4_K_M.gguf \
  /home/ubuntu/models/cosmos_reason2_2b_gguf/mmproj-Cosmos-Reason2-2B-F16.gguf \
  /path/to/image.jpg
```

This is a correctness fallback, not the intended final accelerator path.

## Verification criteria

A successful NPU result must satisfy all of the following:

1. All linked context binaries initialize on the EVK.
2. The runtime reports `QnnHtp`, not a CPU-only backend.
3. Text generation is coherent for a deterministic smoke prompt.
4. The vision encoder consumes an image and its embeddings reach the text
   generator.
5. The generated profile records prompt-processing rate, decode rate, and
   time to first token.

See [docs/architecture.md](docs/architecture.md) for compatibility reasoning
and known risks.
