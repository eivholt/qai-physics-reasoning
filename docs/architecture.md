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

The deployed solution replaces only the four text contexts with W4-weight,
FP16-activation contexts and retains the original W4A16 vision context. That
hybrid generates coherent text and image-conditioned text through `QnnHtp`.
It remains the legacy image/text bring-up baseline.

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
the current integer vision graph numerically equivalent to BF16. The executed
graph's paired calibration still used the older explicit-resize fallback, and
strict free-form accuracy remains weak. Native-aligned integer and
mixed-precision checkpoints require a new authorized compile and NPU run.

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
  -> bias-corrected W8/A16 vision context on QnnHtp
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

The initial r3 controlled test used one answer set for a barrier video and a
routine-pickup video, then shuffled the same choices. BF16 GPU and NPU both
answered `A`, `C`, `B`, `A`, but that narrow 4/4 result did not generalize.
The stronger r4 control covers four scenes and four choices: BF16 GPU scores
7/8, NPU scores 4/8, and exact answers agree in 5/8 cases. Near-miss
avoidance remains robust in both label orders, while box pickup is
order-sensitive on NPU. Both strict three-pair free-form rubrics still fail,
so broad video-reasoning parity remains unsolved. See
[`video_npu.md`](video_npu.md)
and
[`iq9075_video_aspect_native_parity_r3.json`](evidence/iq9075_video_aspect_native_parity_r3.json)
and
[`iq9075_video_four_scene_parity_r4.json`](evidence/iq9075_video_four_scene_parity_r4.json).

Host QuantSim ablation further identifies block-23 activation quantization as
the dominant measured late-stage vision error. A mixed W8/A16 graph with that
block's activations in FP16 raises primary-output cosine similarity against
the corrected adapted BF16 reference from 0.950431 to 0.991453. This candidate
has not been uploaded, compiled, or run on the NPU; those steps require
explicit Qualcomm AI Hub upload authorization.

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
10. Treat the mixed block-23-FP16 graph as a new candidate requiring its own
    compile, physical-board run, and scored comparison.

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
  The narrow r3 control does so in 4/4 GPU/NPU cases, but the expanded r4
  control scores GPU 7/8, NPU 4/8, with 5/8 exact answer parity. This is
  stronger evidence than a single matching label and also shows that broad
  parity has not been reached.
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
