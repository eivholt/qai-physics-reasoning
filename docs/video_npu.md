# Video input on the IQ-9075 NPU

## What works today

The project has now executed one pair of pre-extracted video frames through
two IQ-9075 NPU paths: the earlier legacy Genie compatibility graph and a
GenieX runner with the full three-level DeepStack interface restored. Neither
path accepts an MP4, WebM, camera stream, or other encoded video container
directly. Decode and frame selection remain host-side preprocessing.

[`scripts/prepare_video_npu_inputs.py`](../scripts/prepare_video_npu_inputs.py)
uses the local NVIDIA Hugging Face processor to pack exactly `2N` ordered
frames into `N` native Qwen3-VL temporal patches. For the proven 512 × 512
vision graph, each pair has this fixed contract:

| Property | Value |
|---|---:|
| Frames per temporal patch | 2 |
| `pixel_values_videos` / graph `pixel_values` shape | `[1024, 1536]` |
| `video_grid_thw` | `[1, 32, 32]` |
| Vision output shape | `[256, 2048]` |
| Visual tokens per pair | 256 |
| Current text context | 512 tokens |

The tool verifies the bundle metadata, frame order, timestamps, tensor shapes,
ancillary tensor sizes, and context budget. It writes raw tensors, timestamped
text chunks, a complete repeated-input `genie-app` script, and a SHA-256
provenance manifest. It refuses odd frame counts, non-increasing timestamps,
unsupported graph shapes, and prompts that exceed the compiled context.

This is genuine but narrowly scoped temporal preprocessing: the two distinct
frames are packed together by the Qwen3-VL video processor. It is not two
independent still prompts. Legacy Genie appends the encoded temporal patch to
the shared text accumulator. The project GenieX runner instead supplies the
same raw patch through the lower-level `PixelData` API, inserts one 256-token
vision span, constructs its visual-position mask, and injects the three
DeepStack features into the first text partition.

## Runtime support boundary

| Path | Encoded video | Multiple stills | Paired temporal patch | Full DeepStack |
|---|---|---|---|---|
| Legacy Genie compatibility pipeline | No | Repeated inputs | Yes, one pair tested | No |
| Stock QAI Hub Models generator | No; rejects `pixel_values_videos` and `video_grid_thw` | Yes | No stock path | N/A |
| Stock GenieX v0.3.16 Qwen3-VL CLI | No; image paths only | Yes | No raw-video CLI path | Supported for images |
| Project GenieX raw runner | No; pre-extracted raw tensor only | No | Yes, one pair tested | Yes |

The QAIRT 2.45 legacy compatibility graph omits DeepStack vision injections
and auxiliary visual-mask inputs. The r1 runs therefore validate only the
primary `image_features` route. The r2 GenieX run uses a new first text
partition that retains `visual_pos_masks` and
`deepstack_visual_embeds_0..2`; the vision graph produces all three matching
features, and GenieX slices and injects them. This closes that specific
runtime-wiring gap, but does not establish numerical parity with upstream
BF16.

## Context limit

The measured one-pair benchmark prompt consumes 305 of 512 tokens:
49 text tokens plus 256 visual tokens, leaving 207 tokens for generation.
A second 512 × 512 pair would exceed CL512. The preparation tool rejects it
before writing a runnable package.

The preferred next profile is 384 × 384 vision with CL1024 text. Each pair
would produce 144 visual tokens, so four pairs (eight sampled frames) cost
576 tokens. A representative short prompt is estimated at about 649 tokens
including text, leaving roughly 375 for generation. This keeps materially
more shelf and human detail than 256 × 256 while remaining far cheaper than
four 512 × 512 pairs.

Two fallbacks bracket that tradeoff:

1. CL2048 with the current 512 × 512 vision graph consumes 1024 visual tokens
   for four pairs and preserves the most spatial detail.
2. A 256 × 256 vision profile produces only 64 tokens per pair, but its detail
   loss is undesirable for small, distant warehouse hazards.

The full-DeepStack r2 graph demonstrates the required three feature inputs and
visual mask for one pair. Any longer-context graph must retain those inputs
and be calibrated with distinct paired-video frames, not duplicated stills.
It must also add one interleaved MRoPE record per temporal patch and preserve
the timestamps needed by a real multi-pair video path.

## NVIDIA Isaac Sim benchmark

The repository tracks a lightweight benchmark derived from NVIDIA's official
[PhysicalAI World Model Synthetic Warehouse Operations
Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes).
NVIDIA generated the source scenes in Isaac Sim with Isaac Replicator. The
scenario definitions provide known labels for a forklift–human near miss,
forklift–shelf collision, warehouse fire and evacuation, and routine box
pickup.

See
[`benchmarks/nvidia_sdg_warehouse/README.md`](../benchmarks/nvidia_sdg_warehouse/README.md)
and its pinned
[`benchmark.json`](../benchmarks/nvidia_sdg_warehouse/benchmark.json).
The lightweight cases use RGB crops from NVIDIA's dataset-card previews.
Their frame times are documented estimates; use full dataset shards and
simulator annotations when frame-accurate collision timing is required.

### Physical EVK results

#### Full-DeepStack GenieX r2

All five r2 runs loaded the W4A16 vision context and four W4/FP16 text
contexts through `QnnHtp`, detected HTP v73, used the visual-position mask and
three DeepStack feature levels, and exited with status 0. That is a functional
NPU execution pass. It is not a prediction-quality pass.

| Free-form case | Exact answer | Prompt / generated | TTFT | Decode | Strict rubric |
|---|---|---:|---:|---:|---|
| Near miss | “The forklier will likely continue walking toward the forklift, which is still in motion and could be at risk of collision.” | 305 / 27 tokens | 648.4 ms | 16.60 tok/s | Fail: notes possible collision risk but misses the staged worker dodge and narrowly avoided collision |
| Shelf collision | “The forklier will likely continue to push the forklier, which is at risk of falling.” | 304 / 21 tokens | 648.2 ms | 16.26 tok/s | Fail: does not coherently identify the shelf or predict forklift-shelf contact |

A short multiple-choice probe did not rescue the quality result:

| Control | Correct | Output | Prompt / generated | TTFT | Verdict |
|---|---:|---:|---:|---:|---|
| Near miss, fixed options | B | C | 336 / 1 tokens | 656.5 ms | Fail |
| Shelf collision, fixed options | A | A | 330 / 1 tokens | 646.7 ms | Apparent pass only |
| Shelf collision, same frames with choices shuffled | B | A | 330 / 1 tokens | 646.4 ms | Fail; exposes `A`-position bias |

The shelf collision event is option `A` in the first prompt and option `B` in
the shuffled control. Because the output remains `A`, the first match is
rejected as an answer-position artifact, not accepted as grounded visual
prediction. The one-token runs report 0.00 tok/s because there is no
multi-token decode interval; TTFT is the useful latency metric there.

#### Earlier compatibility r1 and BF16 reference

Both earlier paired-frame cases also initialized the legacy compatibility
bundle, executed successfully, and exited with status 0. That graph uses the
primary vision feature but omits the three DeepStack injections.

| Case | Wall time | Observed answer | Benchmark result |
|---|---:|---|---|
| Near miss | 4.25 s | “No, there is a moving forklift and a forkl near the person; the person is likely to step between the two.” | Functional pipeline pass; semantic benchmark fail |
| Shelf collision | 5.02 s | Incoherent and predicts the wrong motion direction | Functional pipeline pass; semantic benchmark fail |

The exact BF16 CUDA model, given the same near-miss frames and prompt, answered:

> The forklift will likely stop suddenly, posing a risk of collision or injury
> to the worker nearby.

That reference took 4.185 s and peaked at 4.615 GiB GPU memory. It identifies
the immediate risk, whereas the NPU answer does not. It still predicts that
the forklift stops rather than the staged worker dodge, so it passes hazard
recognition but fails the strict future-outcome rubric. The comparison shows
a real additional quality loss in the current NPU export without treating the
BF16 response as a complete predictive pass.

Together, r1 and r2 prove that paired temporal input reaches and executes
through both the compatibility and full-DeepStack NPU paths. Neither
establishes a rubric-correct video prediction, usable video-reasoning
accuracy, or warehouse safety performance.

## Evidence and public-support status

The sanitized
[`IQ-9075 NPU evidence report`](evidence/iq9075_npu_smoke_r1.json)
records the exact hybrid context hashes and the verified text and single-image
smokes. The
[`paired-frame video evidence report`](evidence/iq9075_video_smoke_r1.json)
pins the two frame hashes, packed tensors, prompts, answers, timings, and BF16
reference. The
[`full-DeepStack GenieX report`](evidence/iq9075_video_deepstack_geniex_r2.json)
records the five r2 executions, context and input hashes, QAI Hub job IDs,
exact answers and metrics, and shuffled-option control. Their accompanying
[`README`](evidence/README.md)
defines the scope of the proof.

As of 2026-07-24, no earlier independent public result was found for the exact
`nvidia/Cosmos-Reason2-2B` checkpoint on an IQ-9075 NPU. NVIDIA's
[published Cosmos prerequisites](https://docs.nvidia.com/cosmos/latest/prerequisites.html)
list NVIDIA GPU systems and Jetson AGX Thor, not Qualcomm hardware. Qualcomm
publishes an IQ-9075 configuration for the related
[Qwen3-VL-2B-Instruct GGUF](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct?runtime=geniex_qairt%2Cgeniex_llamacpp),
which supports architectural feasibility but is not proof for the Cosmos
checkpoint. Treat this repository as evidence for a new experimental port,
not vendor certification.
