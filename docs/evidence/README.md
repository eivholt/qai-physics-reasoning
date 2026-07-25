# IQ-9075 NPU evidence

This directory contains sanitized evidence summaries and artifact identifiers
for the experimental `nvidia/Cosmos-Reason2-2B` port in this repository. It
does not contain model weights, Qualcomm context binaries, the complete raw
logs, credentials, or Hugging Face tokens. The hashes identify the artifacts
used and detect later changes; by themselves they are not an independent
attestation that an omitted binary produced an omitted log.

The result is narrower than vendor certification:

- The historical r1 report proves that the linked text and vision graphs in the
  compatibility bundle initialized and executed through QAIRT's `QnnHtp`
  backend on a physical Dragonwing IQ-9075 EVK. It also records coherent text
  generation and image-conditioned generation for its smoke inputs.
- The historical r2 report proves that one paired-frame temporal patch
  executed through the primary vision output, all three restored DeepStack
  outputs, the visual-position mask, and all four text partitions through
  GenieX on the same class of NPU. It is retained as historical
  square-profile evidence.
- The historical r3 report corrects three parity faults: the adapted patch
  projection now retains Qwen3-VL's learned bias, the NPU uses the native
  Hugging Face video pixels and `<|video_pad|>` prompt token, and the
  224 × 384 vision graph is calibrated with distinct warehouse frame pairs.
  Each pair produces 84 visual tokens.
- The historical r3 exact-input one-pair NPU inference passes 0 of 5 strict
  units that the BF16 GPU reference passes. Its three-pair workaround stays
  below the text runtime's safe prefill limit and produces coherent output.
  The r3 two-scene, three-choice control records 4/4 GPU/NPU agreement in
  normal and shuffled order (`A`, `C`, `B`, `A`). That result remains valid
  for that narrow test.
- The historical r4 report expands the same three-pair method to four scenes
  and four choices. BF16 GPU scores 7/8, NPU scores 4/8, and exact answers
  agree in 5/8 cases. Across the first three unambiguous scenes, GPU scores
  6/6 and NPU scores 3/6. The near-miss scene passes both orders on both
  devices; the earlier 4/4 result therefore does not establish broad parity.
  A 25-token-shorter prompt diagnostic moves errors but leaves NPU at 4/8 and
  exact parity at 5/8; both box-pickup orders remain NPU-only failures. A
  focused box/near-miss control scores GPU 4/4 and NPU 3/4: NPU recognizes box
  pickup when it is listed first but fails after its label moves.
- Four-pair prompts of 408 and 413 tokens exceed the AR128/CL512 safe prefill
  limit of 384 and produce corrupted NPU text. The runner and packaging tools
  now reject that configuration. The same four-pair BF16 GPU diagnostics also
  fail their strict free-form rubrics, but do not corrupt.
- A host QuantSim audit localized the largest late-stage vision error to
  activation quantization in vision block 23. A mixed W8/A16 candidate with
  block-23 activations in FP16 improves `image_features` cosine similarity
  against the corrected adapted BF16 reference from 0.950431 to 0.991453. That
  result is retained as historical diagnosis. A valid full checkpoint later
  compiled to a DLC, but QAIRT 2.45 context linking failed with exit code 14,
  so there is no deployable block-23-only context or NPU accuracy result.
- The r5 report compares four completed vision candidates on the same frozen
  20 probes. BF16 GPU scores 16/20. NPU moves from 11/20 for the native-aspect
  W8/A16 baseline to 13/20 for the boundary-FP16 leader, with exact GPU parity
  improving from 13/20 to 15/20 and GPU-correct retention from 10/16 to
  12/16. FP16 weights/A16 scores 11/20; FP16 weights plus FP16 internals and
  A16 boundaries scores 12/20.
- The frozen r5 artifacts derive from the older paired/explicit-resize
  calibration checkpoint. Separate hosted IQ-9075 runs compare native-aligned
  and older paired calibration on 12 exact temporal pairs. Boundary-FP16
  reaches about 0.9849 `image_features` cosine against BF16 versus about
  0.9592 for W8/A16, but native and paired calibration are effectively tied.
  The native-aligned contexts have not been scored end to end on the frozen
  20 probes.
- A W8 part-4 text experiment scores 12/20 with 13/20 exact parity and 11/16
  GPU-correct retention. Its 873.5 ms TTFT mean covers only three
  timing-bearing logs and is not directly comparable with the complete
  candidate means.
- The r6 report is an independent official-GenieX/GGUF route. GenieX v0.3.17
  loads the exact standard Q4_0 main and F16 projector on the EVK. A live
  successful process maps the GGML Hexagon and CDSP FastRPC libraries while
  holding the secure CDSP and DMA-heap descriptors. The exact-wording
  four-scene panel scores NPU 5/8 versus recorded BF16 GPU 7/8, with 6/8
  answer parity. A shorter declared deployment prompt reaches NPU 7/8; a
  task-specific ROI/final-pair profile reaches 4/4. The latter two are
  explicitly not broad same-prompt GPU-parity claims.
- The r7 report isolates the official GenieX route further. With identical
  standard-Q4_0 model files, frames, long prompts, and deterministic sampler,
  CPU scores 6/8 and NPU 5/8 with 7/8 answer parity. NPU adds one shuffled-box
  error; the other BF16-GPU/NPU difference is already present on GenieX CPU.
  BF16 GPU rerun with the shorter deployment user prompt scores 7/8 and
  matches all eight NPU answers. The report also records loss of the CDSP
  FastRPC device node after repeated one-shot model creation.
- The r8 report separates the output layer from the vision `mmproj` context.
  CPU output placement does not fix the shuffled-box error; CPU vision
  placement does. The quality-first CPU-vision/NPU-decoder panel scores 7/8
  and matches all eight recorded BF16 GPU letters at 6.402 seconds mean TTFT.
  Unmodified stock v0.3.17 with `GGML_HEXAGON_OPFILTER=GELU` scores 6/8,
  matches all eight same-GGUF CPU letters, and averages 2.146 seconds TTFT.
  Live-process evidence confirms that both corrected profiles retain the
  Hexagon backend and secure CDSP access.
- Hosted P1 chunk-0 screens compare five candidates with BF16 vision and three
  candidates with the production boundary-FP16 NPU vision output. W8 layers
  0–6 ranks first in both completed screens, but these hidden-state and
  KV-cache measurements are not first-token or answer-accuracy results. Its
  physical-board four-token smoke passes; the full first-token chain and all
  P1 frozen-suite scores remain pending. The balanced P1–P4 GPU extension
  scores 13/16, while its NPU execution is also pending.
- It does not prove production accuracy, full upstream numerical equivalence,
  support for every prompt, or an official NVIDIA/Qualcomm product
  configuration.
- CPU code still performs orchestration, tokenization, sampling, and file I/O;
  "NPU execution" refers to the compiled model graphs.

As of 2026-07-25, NVIDIA's published Reason2 hardware validation list contains
NVIDIA GPUs and Jetson AGX Thor, not Qualcomm hardware:

<https://docs.nvidia.com/cosmos/latest/prerequisites.html>

Qualcomm separately publishes an IQ-9075 configuration for the related
Qwen3-VL-2B-Instruct architecture:

<https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct>

That related configuration is useful feasibility evidence, but it is not
evidence for the Cosmos-Reason2-2B checkpoint. We found no earlier independent
public IQ-9075 result for the exact Cosmos model, so the report here should be
described as project evidence for a new experimental port.

The nine tracked summaries are:

- [`iq9075_npu_smoke_r1.json`](iq9075_npu_smoke_r1.json), the historical text
  and single-image bundle evidence;
- [`iq9075_video_smoke_r1.json`](iq9075_video_smoke_r1.json), the historical two
  paired-frame warehouse runs through the compatibility graph and the BF16
  comparison;
- [`iq9075_video_deepstack_geniex_r2.json`](iq9075_video_deepstack_geniex_r2.json),
  the historical full-DeepStack paired-frame execution through GenieX, two
  failed free-form rubrics, and the multiple-choice position-control result;
- [`iq9075_video_aspect_native_parity_r3.json`](iq9075_video_aspect_native_parity_r3.json),
  the historical corrected native-aspect GPU/NPU parity, one- and three-pair
  accuracy results, the four-pair prefill limit, and the host-only
  mixed-vision diagnosis;
- [`iq9075_video_four_scene_parity_r4.json`](iq9075_video_four_scene_parity_r4.json),
  the historical expanded four-scene choice suite, exact GPU/NPU input parity,
  all eight primary answers and timings, the compact-prompt and focused
  pairwise diagnostics, and the revised quality verdict; and
- [`iq9075_video_precision_parity_r5.json`](iq9075_video_precision_parity_r5.json),
  the frozen 20-probe precision sweep, exact AI Hub artifact lineage,
  completed physical-board comparisons, balanced GPU extension, and
  explicitly pending EVK work; and
- [`iq9075_geniex_gguf_r6.json`](iq9075_geniex_gguf_r6.json), the official
  GenieX v0.3.17 Q4_0/F16 deployment, direct live-process Hexagon evidence,
  CPU/NPU timing controls, exact and shortened four-scene panels, targeted
  edge profile, and measured image/context failure boundaries; and
- [`iq9075_geniex_cpu_npu_parity_r7.json`](iq9075_geniex_cpu_npu_parity_r7.json),
  the same-GGUF CPU/NPU isolation panel, same-user-prompt BF16 GPU rerun,
  answer-level attribution of the remaining gap, and repeated-process
  FastRPC lifecycle observation; and
- [`iq9075_geniex_vision_placement_r8.json`](iq9075_geniex_vision_placement_r8.json),
  the independent output/vision placement controls, stock GELU fallback,
  full corrected panels, latency tradeoffs, negative operator ablations, and
  direct Hexagon/CDSP process evidence.

## Reproduce the report

Run the smoke tests on the EVK, then copy
`scripts/collect_evk_npu_evidence.py` into the deployed bundle and run:

```bash
python3 collect_evk_npu_evidence.py \
  --bundle "$EVK_BUNDLE" \
  --mode all \
  --text-log packaged_text_smoke_r1.log \
  --text-profile packaged_text_profile_r1.txt \
  --vision-log packaged_vlm_smoke_r1.log \
  --report packaged_npu_evidence_r1.json
```

The collector validates the five-context bundle shape, graph ordering,
HTP runtime libraries, FastRPC access, logs, and profile fields. It deliberately
does not hash the multi-gigabyte contexts on the EVK; the release hashes in the
sanitized summary were verified separately during assembly. Re-running the
command against retained artifacts can reproduce the report checks, but the
tracked summary alone cannot independently reproduce the physical-board run.

The full-DeepStack r2 path uses the standalone lower-level runner documented
in
[`integrations/geniex_raw_video/README.md`](../../integrations/geniex_raw_video/README.md).
Its preparation step binds each raw paired-frame input to the exact bundle
hashes in the r2 report. All five omitted logs are identified by SHA-256 in
that report. The report deliberately does not include private network
addresses, machine-specific paths, model weights, or the logs themselves.

The r3 report is intentionally concise. It records exact generated answers,
prompt sizes, input hashes for the one-pair comparisons, pass/fail decisions,
and the numeric metric needed to distinguish the deployed integer vision
graph from the uncompiled mixed candidate. The raw GPU and EVK logs remain
outside the repository because they include machine-specific paths and
runtime noise.

The r4 report adds the primary, compact, and focused-pairwise result-artifact
and NPU-log hashes, exact prompt counts and GPU timings, and both per-pair and
concatenated pixel hashes. The tracked report remains sanitized;
`summary.json`, the raw run packages, and the verbose NPU logs are retained
outside the repository. The fire-scenario choice is scored only on directly
visible worker motion, not on recognizing the official fire cause.

The r5 report is generated from retained metadata and scored result
directories. Reproduce the lightweight scoring layer with:

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
  --output /path/to/new-r5-score.json \
  --require-complete
```

This reads JSON and text logs only; it does not run the model. Keep source
results immutable and bind any newly completed candidate to its context
SHA-256 before adding it to the tracked report.

The same report also binds the 12-pair hosted vision comparison and both P1
screen manifests to their output hashes and AI Hub job identifiers. The
vision H5 files must be mapped by each group’s `name` attribute; numeric H5
group order is not a stable output-name mapping.

<!-- R5_PENDING_WINNER_UPDATE:
Replace pending part-1 and balanced-NPU status only after complete,
hash-bound EVK logs have been scored. Do not infer a winner from smoke output.
-->

The word *video* in these reports has a precise, limited meaning: ordered
frames are fused in pairs into Qwen3-VL temporal patches and supplied as raw
`PixelData`. The r3–r5 runner can interleave multiple timestamped pairs, but
it still does not decode an MP4, consume a camera stream, or provide an
unbounded video frontend.
