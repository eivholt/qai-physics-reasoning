# IQ-9075 NPU evidence

This directory contains sanitized evidence summaries and artifact identifiers
for the experimental `nvidia/Cosmos-Reason2-2B` port in this repository. It
does not contain model weights, Qualcomm context binaries, the complete raw
logs, credentials, or Hugging Face tokens. The hashes identify the artifacts
used and detect later changes; by themselves they are not an independent
attestation that an omitted binary produced an omitted log.

The result is narrower than vendor certification:

- The r1 report proves that the linked text and vision graphs in the
  compatibility bundle initialized and executed through QAIRT's `QnnHtp`
  backend on a physical Dragonwing IQ-9075 EVK. It also records coherent text
  generation and image-conditioned generation for its smoke inputs.
- The r2 report proves that one paired-frame temporal patch executed through
  the primary vision output, all three restored DeepStack outputs, the
  visual-position mask, and all four text partitions through GenieX on the
  same class of NPU. It is retained as historical square-profile evidence.
- The r3 report corrects three parity faults: the adapted patch projection now
  retains Qwen3-VL's learned bias, the NPU uses the native Hugging Face video
  pixels and `<|video_pad|>` prompt token, and the 224 × 384 vision graph is
  calibrated with distinct warehouse frame pairs. Each pair produces 84
  visual tokens.
- The r3 exact-input one-pair NPU inference passes 0 of 5 strict units that
  the BF16 GPU reference passes. Its three-pair workaround stays below the
  text runtime's safe prefill limit and produces coherent output. The r3
  two-scene, three-choice control records 4/4 GPU/NPU agreement in normal and
  shuffled order (`A`, `C`, `B`, `A`). That result remains valid for that
  narrow test.
- The r4 report expands the same three-pair method to four scenes and four
  choices. BF16 GPU scores 7/8, NPU scores 4/8, and exact answers agree in 5/8
  cases. Across the first three unambiguous scenes, GPU scores 6/6 and NPU
  scores 3/6. The near-miss scene passes both orders on both devices; the
  earlier 4/4 result therefore does not establish broad parity. A
  25-token-shorter prompt diagnostic moves errors but leaves NPU at 4/8 and
  exact parity at 5/8; both box-pickup orders remain NPU-only failures. A
  focused box/near-miss control scores GPU 4/4 and NPU 3/4: NPU recognizes
  box pickup when it is listed first but fails after its label moves.
- Four-pair prompts of 408 and 413 tokens exceed the AR128/CL512 safe prefill
  limit of 384 and produce corrupted NPU text. The runner and packaging tools
  now reject that configuration. The same four-pair BF16 GPU diagnostics also
  fail their strict free-form rubrics, but do not corrupt.
- A host QuantSim audit localizes the largest late-stage vision error to
  activation quantization in vision block 23. A mixed W8/A16 candidate with
  block-23 activations in FP16 improves `image_features` cosine similarity
  against the corrected adapted BF16 reference from 0.950431 to 0.991453.
  This is a host-only numeric result: its QAI Hub upload, compile, and NPU run
  are pending explicit authorization.
- The executed integer graph used 20 distinct paired-frame calibration
  samples, but that calibration used the older explicit-resize preprocessing.
  Native-HF-aligned integer and mixed checkpoints exist locally; neither has
  been compiled or run on the NPU yet.
- It does not prove production accuracy, full upstream numerical equivalence,
  support for every prompt, or an official NVIDIA/Qualcomm product
  configuration.
- CPU code still performs orchestration, tokenization, sampling, and file I/O;
  "NPU execution" refers to the compiled model graphs.

As of 2026-07-24, NVIDIA's published Reason2 hardware validation list contains
NVIDIA GPUs and Jetson AGX Thor, not Qualcomm hardware:

<https://docs.nvidia.com/cosmos/latest/prerequisites.html>

Qualcomm separately publishes an IQ-9075 configuration for the related
Qwen3-VL-2B-Instruct architecture:

<https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct>

That related configuration is useful feasibility evidence, but it is not
evidence for the Cosmos-Reason2-2B checkpoint. We found no earlier independent
public IQ-9075 result for the exact Cosmos model, so the report here should be
described as project evidence for a new experimental port.

The five tracked summaries are:

- [`iq9075_npu_smoke_r1.json`](iq9075_npu_smoke_r1.json), for the text and
  single-image bundle evidence;
- [`iq9075_video_smoke_r1.json`](iq9075_video_smoke_r1.json), for the two
  paired-frame warehouse runs through the compatibility graph and the BF16
  comparison;
- [`iq9075_video_deepstack_geniex_r2.json`](iq9075_video_deepstack_geniex_r2.json),
  for full-DeepStack paired-frame execution through GenieX, two failed
  free-form rubrics, and the multiple-choice position-control result;
- [`iq9075_video_aspect_native_parity_r3.json`](iq9075_video_aspect_native_parity_r3.json),
  for corrected native-aspect GPU/NPU parity, one- and three-pair accuracy
  results, the four-pair prefill limit, and the host-only mixed-vision
  diagnosis; and
- [`iq9075_video_four_scene_parity_r4.json`](iq9075_video_four_scene_parity_r4.json),
  for the expanded four-scene choice suite, exact GPU/NPU input parity, all
  eight primary answers and timings, the compact-prompt and focused pairwise
  diagnostics, and the revised quality verdict.

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

The word *video* in these reports has a precise, limited meaning: ordered
frames are fused in pairs into Qwen3-VL temporal patches and supplied as raw
`PixelData`. The r3/r4 runner can interleave multiple timestamped pairs, but
it still does not decode an MP4, consume a camera stream, or provide an
unbounded video frontend.
