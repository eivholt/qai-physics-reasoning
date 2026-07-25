# Architecture and deployment notes

## Why the port is plausible

NVIDIA identifies Cosmos-Reason2-2B as a post-trained
Qwen3-VL-2B-Instruct checkpoint with unchanged network architecture. Its
configuration matches the public Qwen3-VL-2B configuration:

- 28 dense text layers
- hidden size 2048 and intermediate size 6144
- 16 query heads, 8 KV heads, head dimension 128
- tied 151936-token embedding and output weights
- interleaved multimodal RoPE with sections `[24, 20, 20]`
- a 24-layer, width-1024 vision encoder
- three deep-stack vision injection layers

Qualcomm AI Hub Models 0.58.0 has a complete W4A16 QAIRT implementation for
Qwen3-VL-4B and Qwen3-VL-8B. The shared implementation is parameterized by
the dimensions above, so this repository supplies only a concrete 2B class
and the NVIDIA checkpoint name.

The language tower also has the same major dimensions as Qualcomm's
Qwen3-1.7B recipe, which is already validated on IQ-9075. That is useful
evidence for text-decoder operator coverage, but it is not by itself a safe
drop-in because Cosmos uses Qwen3-VL's interleaved multimodal RoPE and
deep-stack vision inputs.

## What physical validation found

The public package does not contain a complete 2B QAIRT model adapter, while
the 4B and 8B packages do. On the IQ-9075, the port successfully compiles,
links, creates QnnHtp contexts, and runs. Those mechanical milestones do not
establish numerical correctness.

The measured W4A16 decoder is incorrect only when its native outputs remain
chained from part 1 through part 4. Each compiled part agrees well with host
QuantSim when it receives a fresh captured QuantSim boundary, but the real
native chain accumulates enough HTP-vs-QuantSim drift to change the first
token. Genie 1.17 and GenieX v0.3.16 hand off byte-identical buffers and
reproduce the bad result, so the evidence does not point to orchestration,
sampling, or split-buffer binding.

The historical legacy-Genie solution replaces only the four text contexts
with W4-weight, FP16-activation contexts and retains the original W4A16
vision context. That hybrid generates coherent text and image-conditioned
text through `QnnHtp`. It remains the legacy image/text bring-up baseline.

The video-parity audit found four additional issues and resolved their
implementation side:

1. Preserve the learned bias when adapting Qwen3-VL's Conv3d temporal patch
   projection to Conv2d. The earlier adapter silently dropped all 1024 bias
   values.
2. Retain `visual_pos_masks` and `deepstack_visual_embeds_0..2` in text part 1
   and bind them through the lower GenieX `PixelData` API.
3. Pass original RGB frames through native Hugging Face video preprocessing
   and use `<|video_pad|>` in the text sequence.
4. Use a 224 × 384 graph that yields 84 tokens per pair and calibrate from
   distinct warehouse frame pairs.

Those fixes eliminate known structural and input mismatches; they do not make
the integer vision graph numerically equivalent to BF16. Strict free-form
accuracy remains weak. The later r5 campaign compiled native-aspect integer
and mixed-precision vision checkpoints and scored them on one frozen
20-probe GPU/NPU suite. Those compiled candidates derive from the older
paired/explicit-resize calibration checkpoint. Native Hugging Face runtime
pixels are byte-identical between GPU and NPU, but this sweep does not prove
native-HF-aligned calibration.

The board has both QAIRT 2.45.0.260326 and 2.47.0.260601. QAI Hub Workbench
currently offers 2.45 as its default compiler and 2.48 as latest, but not
2.47. The primary path therefore compiles with Workbench's 2.45 default and
runs with the exact 2.45 EVK installation. A compiler/runtime mismatch should
not be debugged as if it were a graph issue.

## Video path

The runtime does not accept encoded video directly. Legacy Genie exposes an
image node but no MP4/video node. Stock QAI Hub Models rejects
`pixel_values_videos` and `video_grid_thw`, while GenieX v0.3.16 accepts
multiple still-image paths but has no native video frontend.

The implemented bridge uses Qwen3-VL's native temporal patch representation:

```text
decode and sample outside Genie
  -> group ordered RGB frames into temporal pairs
  -> native Hugging Face Qwen3-VL video processor at 224 × 384
  -> one pixel tensor [336, 1536], grid [1, 14, 24], per pair
  -> boundary-FP16 vision leader on QnnHtp
     (W8 weights, FP16 internals, nine A16 boundaries)
  -> 84 primary + 3 × 84 DeepStack features per pair
  -> <|video_pad|> span + timestamped interleaved MRoPE record per pair
  -> visual_pos_masks + full-interface text part 1
  -> W4/FP16 text contexts on QnnHtp
```

The project runner now interleaves multiple temporal pairs. The limiting
factor is not only CL512: its AR128 prefill graph can safely transfer at most
`CL - AR = 384` prompt tokens into the AR1 decode cache. Measured four-pair
prompts total 408 and 413 tokens and corrupt the NPU decode, while three-pair
prompts span 313–360 in the current benchmark and remain coherent.
Preparation, packaging, and runtime guards now enforce 384.

The historical r3 controlled test used one answer set for a barrier video and a
routine-pickup video, then shuffled the same choices. BF16 GPU and NPU both
answered `A`, `C`, `B`, `A`, but that narrow 4/4 result did not generalize.
The historical r4 control covers four scenes and four choices: BF16 GPU scores
7/8, NPU scores 4/8, and exact answers agree in 5/8 cases. Near-miss
avoidance remains robust in both label orders, while box pickup is
order-sensitive on NPU. Both strict three-pair free-form rubrics still fail,
so broad video-reasoning parity remains unsolved. See
[`video_npu.md`](video_npu.md)
and
[`iq9075_video_aspect_native_parity_r3.json`](evidence/iq9075_video_aspect_native_parity_r3.json)
and
[`iq9075_video_four_scene_parity_r4.json`](evidence/iq9075_video_four_scene_parity_r4.json).

### R5 precision result

The r5 frozen suite joins the eight primary four-choice probes, eight compact
probes, and four focused box/near-miss probes. BF16 GPU scores 16/20. The
native-aspect W8/A16 baseline scores 11/20 on NPU; changing 925 internal
vision activations to FP16 while retaining nine A16 graph boundaries raises
the NPU result to 13/20. Exact GPU/NPU answer parity rises from 13/20 to
15/20, and GPU-correct retention rises from 10/16 to 12/16. Mean TTFT rises
from 738.070 to 857.510 ms.

| Vision precision | NPU correct | Exact parity | GPU-correct retained | Mean TTFT |
|---|---:|---:|---:|---:|
| W8/A16 baseline | 11/20 | 13/20 | 10/16 | 738.070 ms |
| W8, FP16 internals, nine A16 boundaries | **13/20** | **15/20** | **12/16** | 857.510 ms |
| FP16 weights, A16 activations | 11/20 | 13/20 | 10/16 | 772.825 ms |
| FP16 weights and internals, nine A16 boundaries | 12/20 | 14/20 | 11/16 | 686.080 ms |

The boundary-only candidate is the completed accuracy leader. Host numerical
similarity did not rank task accuracy reliably: converting the vision weights
to FP16 as well produced a faster but one-point-worse 12/20 result. The
remaining five boundary/GPU answer disagreements are therefore not explained
by a single global vision-fidelity metric.

The text tower remains W4/FP16 in that completed leader. A W8 part-4
experiment scores 12/20 with 13/20 exact parity and retains 11/16 GPU-correct
answers; its 873.5 ms TTFT mean covers only 3 of 20 timing-bearing logs and is
not directly comparable with the complete means above. Three part-1 variants
are compiled but unscored while the EVK is offline: layers 0–6 (252 W8
matrices), layers 0–3 (144), and layers 0–2 (108). The full variant has passed
a four-token HTP smoke; the narrower variants and all three frozen-suite
scores remain pending.

The benchmark also adds two more balanced option permutations per scene.
BF16 GPU scores 13/16 across P1–P4 (6/8 on the new P3/P4 half): barrier 4/4,
box 3/4, near miss 4/4, and fire/worker-motion 2/4. No balanced NPU result has
been run while the EVK is offline.

See
[`iq9075_video_precision_parity_r5.json`](evidence/iq9075_video_precision_parity_r5.json)
for the artifact-bound result sheet.

### Text context linking contract

Each text part is compiled twice, for AR128 prefill and AR1 decode, then
linked into one context. A usable context must preserve graph order
AR128→AR1 and must report an equal, nonzero shared-weight allocation for both
graphs. AI Hub can reuse model identities aggressively enough that repeating
a link returns the wrong AR1→AR128 order. `scripts/cache_bust_dlc.py` changes
only a validated DLC/ZIP archive comment, producing a new artifact hash while
leaving member payloads unchanged; uploading the cache-busted AR128 model
before AR1 then allows a fresh ordered link.

This identity workaround does not replace inspection. A local no-sharing
part-1 link had the desired graph order but expanded to 711,176,192 bytes and
failed on the EVK while mapping a 490,733,568-byte FastRPC buffer. It is
rejected. The accepted full-W8 part-1 link is 357,736,448 bytes, reports
353,431,552 shared-weight bytes for each graph, and passes a multi-token HTP
smoke.

## Bring-up order

1. Run the public GGUF on the EVK CPU to validate model, tokenizer, and
   projector independently.
2. Quantize the split text tower and establish coherent W4/FP16 text contexts.
3. Preserve the learned temporal patch-projection bias and select the
   224 × 384 vision profile.
4. Prepare distinct, frame-disjoint temporal calibration pairs with the same
   native Hugging Face preprocessing used by inference.
5. Run host BF16 and QuantSim numeric comparisons for the primary and all
   three DeepStack outputs.
6. With explicit authorization for the Qualcomm AI Hub upload, export the
   vision encoder and full-interface text part 1 for IQ-9075.
7. Assemble the pinned GenieX bundle and verify shapes, hashes, graph order,
   pad token, and the AR128/CL512 safe prefill limit.
8. Run exact-input one-pair GPU/NPU comparisons, then three-pair
   cross-scene/order controls.
9. Keep four-pair CL512 prompts blocked when they exceed 384; use a
   longer-context text export before retrying them.
10. Generate each precision candidate with its fail-closed transform, compile
    it as a new AI Hub artifact, and score it on the same frozen suite. Do not
    select a deployment from host cosine alone.
11. For each text link, prove AR128→AR1 order and equal nonzero shared weights.
    Reject no-sharing contexts even if a one-token prefill appears to work.

## NPU proof

A successful process exit is insufficient. The runtime log must show the
`QnnHtp` backend and successful HTP device/context creation. Captured
throughput should include prompt processing, generated-token rate, and time
to first token.

Functional proof and quality proof are separate:

- The context hashes, graph order, `QnnHtp` initialization, FastRPC access,
  bound inputs, and exit status prove that the recorded graphs execute on the
  physical EVK.
- A coherent response to a static smoke input proves only that specific
  image-conditioned path.
- A paired-frame exit 0 proves temporal-patch plumbing, not useful video
  reasoning. Exact pixel/prompt parity narrows the diagnosis but is still not
  an accuracy result.
- A controlled pass must change correctly with both video and answer ordering.
  The historical narrow r3 control does so in 4/4 GPU/NPU cases, but the
  historical expanded r4
  control scores GPU 7/8, NPU 4/8, with 5/8 exact answer parity. This is
  stronger evidence than a single matching label and also shows that broad
  parity has not been reached.
- The completed r5 precision suite raises the best NPU score to 13/20 against
  GPU 16/20, with 15/20 exact answer parity. It improves the deployment but
  still does not establish GPU-equivalent video reasoning.
- Free-form benchmark answers must still be scored against known outcomes and
  an upstream BF16 reference. Current three-pair free-form cases fail on both
  GPU and NPU.

The sanitized evidence is in
[`docs/evidence`](evidence/README.md).
No earlier independent public IQ-9075 result was found for the exact
Cosmos-Reason2-2B checkpoint. NVIDIA's
[published Cosmos validation list](https://docs.nvidia.com/cosmos/latest/prerequisites.html)
excludes Qualcomm; Qualcomm's related IQ-9075 result is for
[Qwen3-VL-2B-Instruct GGUF](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct?runtime=geniex_qairt%2Cgeniex_llamacpp).
This project evidence is not vendor certification.
