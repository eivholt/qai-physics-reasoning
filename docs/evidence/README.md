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
  same class of NPU.
- Neither r1 nor r2 proves a correct future-event prediction. Both r2
  free-form warehouse rubrics fail. The shelf multiple-choice result that
  initially matches the correct option is rejected because a shuffled-option
  control still returns `A` after the correct answer moves to `B`.
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

The three tracked summaries are:

- [`iq9075_npu_smoke_r1.json`](iq9075_npu_smoke_r1.json), for the text and
  single-image bundle evidence;
- [`iq9075_video_smoke_r1.json`](iq9075_video_smoke_r1.json), for the two
  paired-frame warehouse runs through the compatibility graph and the BF16
  comparison; and
- [`iq9075_video_deepstack_geniex_r2.json`](iq9075_video_deepstack_geniex_r2.json),
  for full-DeepStack paired-frame execution through GenieX, two failed
  free-form rubrics, and the multiple-choice position-control result.

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

The word *video* in the r2 report has a precise, limited meaning: two ordered
frames are fused into one Qwen3-VL temporal patch and supplied as raw
`PixelData`. This is not native MP4 ingestion, a camera stream, or the full
multi-pair, timestamp-aware Qwen3-VL video path.
