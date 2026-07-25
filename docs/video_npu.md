# Video input on the IQ-9075 NPU

## What works today

The project now executes one or more pre-extracted temporal frame pairs
through a GenieX runner with the full three-level DeepStack interface on the
IQ-9075 NPU. The earlier legacy Genie compatibility path remains as historical
one-pair evidence. Neither path accepts an MP4, WebM, camera stream, or other
encoded video container directly; decoding and frame selection remain
host-side preprocessing.

[`scripts/prepare_video_npu_inputs.py`](../scripts/prepare_video_npu_inputs.py)
uses the local NVIDIA Hugging Face processor to pack exactly `2N` ordered
frames into `N` native Qwen3-VL temporal patches. For the current native-aspect
vision graph, each pair has this contract:

| Property | Value |
|---|---:|
| Frames per temporal patch | 2 |
| Processor target | 224 × 384 |
| `pixel_values_videos` / graph `pixel_values` shape | `[336, 1536]` |
| `video_grid_thw` | `[1, 14, 24]` |
| Primary and each DeepStack output | `[84, 2048]` |
| Visual tokens per pair | 84 |
| Current text context | 512 tokens |
| Safe prefill prompt | At most 384 tokens |

The current completed accuracy leader uses W8 vision weights, 925 FP16
internal activations, and nine A16 graph boundaries. The original W8/A16
context remains the frozen r5 baseline.

The tool verifies the bundle metadata, frame order, timestamps, tensor shapes,
ancillary tensor sizes, and context budget. It writes raw tensors, timestamped
text chunks, and a SHA-256 provenance manifest. It refuses odd frame counts,
non-increasing timestamps, unsupported graph shapes, and prompts above the
runtime's safe prefill limit.

This is genuine but narrowly scoped temporal preprocessing: the two distinct
frames are packed together by the Qwen3-VL video processor. It is not two
independent still prompts. The current input path passes the original RGB
frames to the native Hugging Face processor, and the runner uses the upstream
`<|video_pad|>` token rather than the image-pad token. Exact byte hashes
confirm that each GPU and NPU pair receives the same packed pixels. GenieX
supplies each raw patch through the lower-level `PixelData` API, inserts an
84-token vision span with a timestamp-aware MRoPE record, constructs the
visual-position mask, and injects the three DeepStack features into the first
text partition.

## Runtime support boundary

| Path | Encoded video | Paired temporal patches | Full DeepStack |
|---|---|---|---|
| Legacy Genie compatibility pipeline | No | One pair historically tested | No |
| Stock QAI Hub Models generator | No; rejects `pixel_values_videos` and `video_grid_thw` | No stock path | N/A |
| Stock GenieX v0.3.16 Qwen3-VL CLI | No; image paths only | No raw-video CLI path | Supported for images |
| Official GenieX v0.3.17 GGUF pilot | No supported encoded-video path; six ordered stills tested | No native temporal-patch CLI path | Supported internally by llama.cpp |
| Project GenieX raw runner | No; pre-extracted raw tensors only | Yes; one and three pairs tested | Yes |

The QAIRT 2.45 legacy compatibility graph omits DeepStack vision injections
and auxiliary visual-mask inputs. The r1 runs therefore validate only the
primary `image_features` route. The r2 GenieX run uses a new first text
partition that retains `visual_pos_masks` and
`deepstack_visual_embeds_0..2`; the vision graph produces all three matching
features, and GenieX slices and injects them. This closes that specific
runtime-wiring gap. A later audit also found and fixed a conversion error
before the transformer: the learned 1024-element bias from Qwen3-VL's Conv3d
patch projection had been lost when the layer was adapted to Conv2d. Fixing
the bias, DeepStack wiring, pixels, and pad token removes four concrete parity
problems, but useful answer accuracy still has to be measured separately.

## Context limit

CL512 is not the whole usable prefill budget. The deployed text contexts use
an AR128 prefill graph followed by AR1 decode. GenieX v0.3.16 stores only
`CL - AR = 384` tokens in the prefill KV layout that is transferred into the
decode cache. A prompt from 385 through 512 tokens may fit the nominal context
while still being unsafe at the last prefill-to-decode handoff.

This was observed directly. Four 224 × 384 pairs use 336 visual tokens, but
the two measured prompts total 408 and 413 tokens. Both NPU runs generated
corrupted text. Three-pair variants use 252 visual tokens and total 316–348
prompt tokens in the r3 runs; the current r4/free-form/diagnostic set spans
313–360 tokens, and all measured NPU outputs are coherent. Input preparation,
package verification, and the C++ runner now independently reject prompts
above 384.

The limit is a runtime-layout constraint, not a claim that four temporal pairs
are intrinsically unsupported by Cosmos. A future longer-context export must
retain the full DeepStack interface and compile a text runtime whose safe
prefill capacity covers the complete prompt. Until then, three pairs are the
tested CL512 workaround.

The historical r3/r4 W8/A16 graph used the paired-frame calibration lineage
recorded in those reports. The r5 precision campaign compiled and ran its
native-aspect baseline and three mixed-precision vision variants separately;
the completed scores are reported below. All four r5 compiled artifacts still
derive from the older paired/explicit-resize calibration checkpoint. Their
runtime pixels are native-HF-aligned and byte-identical between GPU and NPU,
but their calibration source is not.

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

#### Official GenieX v0.3.17 GGUF result

The official `llama_cpp` plugin now supplies an independent NPU route for the
exact model. A successful process maps `libggml-hexagon.so` and
`libcdsprpc.so`, holds `/dev/fastrpc-cdsp-secure`, and produces the same
grounded barrier answer as CPU: the blue forklift knocks the striped marker
flat. NPU TTFT is 1.754 seconds versus CPU 7.339 seconds for that six-image
prompt. One-image vehicle recognition is also identical (`forklift`) at
0.657-second NPU versus 2.458-second CPU TTFT.

This path consumes ordered independent images, not native Qwen3-VL temporal
pairs. With the exact r4 four-scene wording it scores 5/8 versus recorded BF16
GPU 7/8. The exact same-GGUF isolation scores CPU 6/8 and NPU 5/8 with 7/8
CPU/NPU answer parity, attributing one additional shuffled-box error to NPU
execution. A controlled placement build shows that moving the final output
layer to CPU does not change that error, while moving the vision `mmproj`
context to CPU restores the correct box answer.

Two corrected profiles result:

| Profile | Long-prompt score | BF16 GPU letter parity | Mean TTFT |
|---|---:|---:|---:|
| Stock full NPU | 5/8 | 6/8 | 1.800 s |
| Stock v0.3.17 with `GGML_HEXAGON_OPFILTER=GELU` | 6/8 | 7/8 | 2.146 s |
| CPU vision context, NPU decoder | **7/8** | **8/8** | 6.402 s |

The GELU fallback exactly matches all eight same-GGUF CPU letters and needs
no custom build. The quality-first CPU-vision/NPU-decoder profile matches the
recorded BF16 GPU answer sequence `A C B D C B C A`; its 6.402-second mean
TTFT remains below the 7.688-second all-CPU mean. Direct process inspection
for both profiles still maps `libggml-hexagon.so` and holds the secure CDSP
plus DMA-heap descriptors. The GELU result localizes the original regression
to behavior affected by that operator's Hexagon placement, but does not alone
prove an isolated kernel defect.

A shorter fixed deployment user prompt reaches stock NPU 7/8; BF16 GPU rerun
with the same user prompt also scores 7/8 and matches all eight NPU answers.
Declared ROI/final-pair preprocessing reaches 4/4 on four targeted tasks.
These improvements are useful deployment evidence but do not replace the
native-pair parity benchmark. See
[`iq9075_geniex_gguf_r6.json`](evidence/iq9075_geniex_gguf_r6.json) and
[`iq9075_geniex_cpu_npu_parity_r7.json`](evidence/iq9075_geniex_cpu_npu_parity_r7.json),
plus
[`iq9075_geniex_vision_placement_r8.json`](evidence/iq9075_geniex_vision_placement_r8.json).

The measured safe boundary is also narrower than the nominal CLI flags:
`nctx=8192` aborts during vision-model allocation, and a single 1344 × 768
image aborts both NPU-only and hybrid with `dspqueue_read 0x2e`.
`--image-max-length` did not prevent that large-grid path. Keep this pilot at
`nctx=4096` and 384 × 216 per image, using fixed ROI crops when a small actor
needs more detail.

Repeated one-shot CLI creation is another measured limit. The r7 matrix
completed, but one CPU and one NPU creation needed a bounded retry; later
creates failed at `GGML_ASSERT(device)` after the CDSP FastRPC device nodes
disappeared. Memory remained available and no inference process was alive.
Prefer a persistent model process, monitor `/dev/fastrpc-cdsp-secure`, and
expect a board reboot or power cycle for recovery on this software image.

#### Historical r3 exact-input one-pair comparison

The corrected aspect graph and runner load through `QnnHtp`, detect HTP v73,
and use the primary feature, all three DeepStack features, and visual mask.
The five strict gate units below include only cases the BF16 GPU reference
passes. The shelf normal/shuffled pair is one atomic control.

| Gate | BF16 GPU | NPU | NPU TTFT | Strict NPU result |
|---|---|---|---:|---|
| Shelf collision, free form | Predicts collision with the yellow/black object | Predicts the forklift may tip over | 389.3 ms | Fail |
| Near miss, choice | `B` | `C` | 389.0 ms | Fail |
| Shelf choice, normal then shuffled | `A` → `B` | `B` → `A` | 382.7 / 383.2 ms | Fail |
| Observed near-miss avoidance | Worker steps back and forward to avoid forklift | Worker “jump[s] over the forklift” | 384.6 ms | Fail |
| Routine box pickup | Carrying box; explicitly no collision or accident | “The worker is walking and carrying a box.” | 387.5 ms | Fail: omits required non-incident state |

The verdict is 0/5 strict GPU-correct gate units on the current one-pair NPU
graph. Native Hugging Face pixel hashes and prompt-token construction match,
so manual resizing and the earlier image-pad token are not plausible
explanations for these remaining differences.

#### Historical r3/r4 three-pair controlled results

Three-pair prompts stay inside the 384-token safe prefill limit. The strongest
initial r3 test used one identical three-choice answer set across the
barrier-knockdown and routine box-pickup videos, then shuffled that set
identically:

| Video | Order | Correct | BF16 GPU | NPU | Prompt tokens | NPU TTFT |
|---|---|---:|---:|---:|---:|---:|
| Barrier knockdown | Normal | A | A | A | 344 | 730.5 ms |
| Barrier knockdown | Shuffled | C | C | C | 344 | 739.8 ms |
| Routine box pickup | Normal | B | B | B | 341 | 745.8 ms |
| Routine box pickup | Shuffled | A | A | A | 341 | 736.0 ms |

All per-pair GPU pixel tensors match the NPU manifests exactly. The NPU's
4/4 agreement with BF16 changes with both scene and option ordering, ruling
out a fixed-position answer and a scene-insensitive policy for this controlled
test. That result remains valid, but the expanded r4 test shows that it was
too narrow to support a general parity claim.

The r4 suite adds grounded near-miss avoidance and visible two-worker motion,
and gives every scene the same four choices in normal and shuffled order:

| Video | Order | Correct | BF16 GPU | NPU | Prompt tokens | GPU time | NPU TTFT |
|---|---|---:|---:|---:|---:|---:|---:|
| Barrier knockdown | Normal | A | A | A | 360 | 0.082812 s | 743.5 ms |
| Barrier knockdown | Shuffled | C | C | B | 360 | 0.059286 s | 733.9 ms |
| Routine box pickup | Normal | B | B | C | 357 | 0.059938 s | 731.3 ms |
| Routine box pickup | Shuffled | D | D | B | 357 | 0.058677 s | 742.9 ms |
| Near-miss avoidance | Normal | C | C | C | 357 | 0.059523 s | 750.7 ms |
| Near-miss avoidance | Shuffled | B | B | B | 357 | 0.058406 s | 740.8 ms |
| Two workers leave aisles | Normal | D | C | C | 357 | 0.058769 s | 744.7 ms |
| Two workers leave aisles | Shuffled | A | A | A | 357 | 0.117618 s | 732.0 ms |

The BF16 GPU scores 7/8, NPU scores 4/8, and exact answers agree in 5/8
rows. Across the first three unambiguous scenes (marker, box, and near miss),
GPU scores 6/6 while NPU scores 3/6. The strongest realistic positive result
is near-miss avoidance: both devices follow its correct label from `C` to `B`
after shuffling, 2/2.

For every row, concatenating the NPU's three ordered pair tensors produces the
exact GPU `pixel_values` hash, and GPU/NPU prompt counts are identical. This
makes an input-preparation mismatch an implausible explanation for the four
wrong NPU answers. The fire-scenario score asks only whether two workers
visibly leave their earlier aisle positions; recognizing the official fire
cause is not required. GPU and NPU both choose the wrong `C` in its normal
order, illustrating that answer parity can also occur on an incorrect answer.

The revised verdict is therefore mixed: the NPU has realistic correct
video-conditioned examples, especially the order-controlled near miss, but
broad GPU parity is unsolved.

A compact-prompt diagnostic tests whether option verbosity is the culprit.
It shortens every prompt by 25 tokens, from 357–360 to 332–335, while keeping
the same pixels, labels, and order control:

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

GPU scores 5/8, NPU remains 4/8, and exact parity remains 5/8. Shorter
wording moves the marker and fire errors instead of closing the gap. Both
box-pickup orders remain persistent NPU-only failures against correct GPU
answers, so prompt verbosity is not a sufficient explanation.

A focused 327-token box-versus-near-miss control removes the other two
distractors:

| Video | Choice order | Correct | BF16 GPU | NPU | NPU TTFT |
|---|---|---:|---:|---:|---:|
| Routine box pickup | Box A / near B | A | A | A | 741.9 ms |
| Routine box pickup | Near A / box B | B | B | A | 732.4 ms |
| Near-miss avoidance | Box A / near B | B | B | B | 739.9 ms |
| Near-miss avoidance | Near A / box B | A | A | A | 741.1 ms |

GPU scores 4/4, NPU 3/4, and exact parity is 3/4. The NPU recognizes
box pickup when it is option `A`, but keeps `A` after box moves to `B`. This
narrows the failure to order sensitivity rather than total visual confusion;
it does not restore robust box recognition.

An earlier, separately worded tailored box-pickup probe follows its shuffled
label (`B` then `A`), which further shows prompt sensitivity rather than a
stable capability. The tailored barrier probe fails (`A` instead of
GPU-correct `B`), and the free-form outputs remain incomplete:

| Three-pair free-form case | NPU answer summary | Strict result |
|---|---|---|
| Barrier knockdown | Marker is attached to and pushed by forklift movement | Fail: never says it is knocked down |
| Routine box pickup | Worker stands and moves away without an accident | Fail: never says the box is picked up or held |

The BF16 GPU reference also fails both strict three-pair free-form rubrics.
Across its full three-pair artifact, GPU passes 7/7 multiple-choice cases and
0/2 free-form cases.

#### Historical r3 four-pair diagnostic

Before the prefill guard was added, four-pair prompts of 408 and 413 tokens
produced corrupted NPU text. They exceed the AR128/CL512 safe limit of 384;
this is not scored as a model-accuracy failure. BF16 GPU text remains coherent
for the same inputs, but both outputs fail the strict event rubrics: it says
the barrier is lifted rather than knocked down and describes standing up
without identifying the box pickup.

#### R5 frozen precision comparison

The r5 suite freezes all 20 existing choice probes: eight primary
four-choice prompts, eight compact prompts, and four focused box/near-miss
prompts. The same BF16 GPU result set is used for every NPU candidate. GPU
scores 16/20.

| Candidate | Vision layout | NPU correct | Exact GPU parity | GPU-correct retained | Mean TTFT |
|---|---|---:|---:|---:|---:|
| Baseline | W8 weights, A16 activations | 11/20 | 13/20 | 10/16 | 738.070 ms |
| Boundary-FP16 | W8 weights, 925 FP16 internal activations, nine A16 boundaries | **13/20** | **15/20** | **12/16** | 857.510 ms |
| W-FP16/A16 | FP16 weights, A16 activations | 11/20 | 13/20 | 10/16 | 772.825 ms |
| Combined | FP16 weights, 925 FP16 internals, nine A16 boundaries | 12/20 | 14/20 | 11/16 | 686.080 ms |
| Boundary-FP16 + W8 text part 4 | Boundary-FP16 vision; decoder layers 21–27 W8 | 12/20 | 13/20 | 11/16 | 873.5 ms on 3/20 logs |

The boundary-FP16 vision graph is the current completed accuracy leader:
13/20 on NPU versus 16/20 on GPU. It closes two baseline errors and raises
exact answer agreement to 15/20, but a three-point correctness gap remains.
The combined candidate is faster in this run but scores one point lower,
showing that better host numerical fidelity or broader FP16 coverage does not
guarantee better task accuracy.

All completed runs use the same byte-checked packed pixels, prompt definitions,
runner contract, and frozen scoring tool. The score includes accuracy against
benchmark truth, exact GPU/NPU answer parity, and retention of the 16 probes
the GPU answers correctly; these are different quantities and should not be
interchanged.

Three early-decoder W8 variants remain pending outside the resumed GenieX
GGUF pilot:

| Pending text candidate | Scope | AI Hub linked target | NPU status |
|---|---|---|---|
| Part 1 full | Layers 0–6, 252 matrices | `mn7yvo34n` | Four-token HTP smoke passed; frozen 20 pending |
| Part 1 layers 0–3 | 144 matrices | `mm55go5km` | EVK smoke and frozen 20 pending |
| Part 1 layers 0–2 | 108 matrices | `mnzgp4vxq` | AI Hub graph order/shared weights proven; EVK inspection, smoke, and frozen 20 pending |

No accuracy is inferred for these pending candidates. The balanced P1–P4 GPU
extension scores 13/16 overall (6/8 for the newly added P3/P4 probes), with
barrier 4/4, box 3/4, near miss 4/4, and fire/worker-motion 2/4. Its NPU result
is likewise pending the EVK's return.

Use `scripts/score_video_npu_results.py` with all GPU result roots and one or
more labeled NPU result roots to reproduce the comparison. `--require-complete`
refuses a candidate with a missing or unparsed answer.

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
exact answers and metrics, and shuffled-option control. The
[`native-aspect parity report`](evidence/iq9075_video_aspect_native_parity_r3.json)
records the corrected one- and three-pair comparisons, the four-pair prefill
failure, and the host-only mixed-vision metric. The
[`expanded four-scene report`](evidence/iq9075_video_four_scene_parity_r4.json)
records all eight primary r4 answers, exact input parity, timings, the compact
and focused-pairwise diagnostics, and the revised verdict. The
[`r5 precision report`](evidence/iq9075_video_precision_parity_r5.json)
records the frozen 20-probe candidate comparison, artifact hashes, AI Hub
jobs, completed runs, and explicitly pending work. The
[`README`](evidence/README.md)
defines the scope of the proof.

As of 2026-07-25, no earlier independent public result was found for the exact
`nvidia/Cosmos-Reason2-2B` checkpoint on an IQ-9075 NPU. NVIDIA's
[published Cosmos prerequisites](https://docs.nvidia.com/cosmos/latest/prerequisites.html)
list NVIDIA GPU systems and Jetson AGX Thor, not Qualcomm hardware. Qualcomm
publishes an IQ-9075 configuration for the related
[Qwen3-VL-2B-Instruct GGUF](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct?runtime=geniex_qairt%2Cgeniex_llamacpp),
which supports architectural feasibility but is not proof for the Cosmos
checkpoint. Treat this repository as evidence for a new experimental port,
not vendor certification.
