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

The preferred path is:

```text
nvidia/Cosmos-Reason2-2B (BF16)
  -> Qwen3-VL-2B-compatible QAI Hub Models wrapper
  -> AIMET-ONNX W4A16 calibration
  -> QAIRT 2.45 interface compatibility transform
     |-> export the original W4A16 vision encoder only
     `-> convert text activations to FP16 and export text parts 1-4 only
  -> assemble W4A16 vision + W4/FP16 text linked contexts
  -> legacy Genie 1.17 / QAIRT 2.45 on the IQ-9075 NPU
```

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
| Proven deployment | Original W4A16 vision + W4/FP16 text |

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
- The first all-W4A16 bundle compiles, links, and executes on QnnHtp, but its
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
- The proven bundle combines the original W4A16 vision context with the four
  W4/FP16 text contexts. Text-only legacy Genie produces coherent English at
  131.152 ms TTFT, 289.7711 prompt tok/s, and 17.9982 generation tok/s. The
  default `genie-app` run exits 0, accepts all five image inputs, and describes
  Qualcomm's actual `dog.jpg` sample as a white fluffy dog on green grass.
  Genie dequantizes the quantized `image_features` into its accumulator, so
  the original W4A16 vision context interoperates with the FP16 text path.
- Video containers are not accepted directly. The repository can predecode
  `2N` frames and pack each pair as one native Qwen3-VL temporal patch for
  repeated legacy-Genie NPU execution. The proven 512 × 512 graph consumes
  `[1024, 1536]` pixels with grid `[1, 32, 32]` and produces 256 tokens per
  pair. CL512 fits one pair only; the measured benchmark prompt uses 305/512
  tokens.
- Two NVIDIA Isaac Sim warehouse cases execute end to end on the physical EVK
  but currently fail their semantic benchmark rubrics. An exact BF16 CUDA
  baseline gives a coherent collision-risk answer on the near-miss prompt but
  also misses the staged dodge outcome. It is materially better than the NPU
  answer, confirming a quality gap without claiming a full predictive pass.
- A second experiment restores `visual_pos_masks` and
  `deepstack_visual_embeds_0..2`: it replaces only part 1 with a full-interface
  W4/FP16 context, retains the coherent W4/FP16 parts 2-4, and uses a pinned
  GenieX lower-level `PixelData` runner. Both paired-frame cases execute
  cleanly on QnnHtp/Hexagon v73 at about 16 tokens/s, but both free-form
  predictions still fail their semantic rubrics. A shuffled multiple-choice
  control also exposes option-position bias. No robust realistic NPU video
  prediction has passed yet.

For the complete reproducible workflow, measured job IDs, GenieX import,
native-chain diagnosis, and failure guide, see
[the developer tutorial](docs/developer_tutorial.md). See
[video input on the NPU](docs/video_npu.md) for the temporal contract,
measured warehouse results, and scaling plan. The
[sanitized evidence record](docs/evidence/README.md), including the
[legacy paired-frame report](docs/evidence/iq9075_video_smoke_r1.json) and
[full-DeepStack GenieX report](docs/evidence/iq9075_video_deepstack_geniex_r2.json),
contains the exact physical-board proof boundary.

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

Encoded video is decoded and sampled outside Genie. Fetch NVIDIA's lightweight
Isaac Sim warehouse previews, extract a prediction pair, and pack it with the
local Qwen3-VL processor:

```bash
python scripts/fetch_nvidia_sdg_warehouse.py \
  --asset-set clips \
  --extract-rgb \
  --case predict_near_miss

python scripts/prepare_video_npu_inputs.py \
  --bundle ~/cosmos-reason2-iq9/exports/w4-fp16-text-w4a16-vision-hybrid-cl512-qairt245-r1 \
  --processor ~/models/Cosmos-Reason2-2B \
  --output-dir ~/cosmos-reason2-iq9/exports/w4-fp16-text-w4a16-vision-hybrid-cl512-qairt245-r1/video_inputs/predict_near_miss_r1 \
  --frames \
    artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0030.png \
    artifacts/nvidia_sdg_warehouse/frames/predict_near_miss/frame_0045.png \
  --timestamps 2.0 3.0 \
  --question "What is most likely to happen next, and is there an immediate safety risk? Answer in one short sentence."
```

The command prints `305/512` prompt tokens and `207` remaining for this case.
Copy the generated directory into the same relative location in the clean EVK
bundle, then run its `genie-video-app-script.txt`. Exact `scp`, QAIRT
environment, and execution commands are in
[the developer tutorial](docs/developer_tutorial.md#12-run-paired-video-frames-on-the-npu).

That legacy script uses the compatibility part 1 with DeepStack disabled. For
the tested full-DeepStack path, assemble the full-interface W4/FP16 part-1
replacement and use `scripts/geniex_raw_video.sh` to build, package, deploy,
and run the pinned lower-level GenieX `PixelData` integration. The same
tutorial section records the exact commands and the negative semantic verdict.

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
