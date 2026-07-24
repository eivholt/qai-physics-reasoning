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
The clean physical-board bundle is
`/home/ubuntu/cosmos_reason2_2b_w4_fp16_hybrid_r1`.

Remaining failure points include:

1. Accuracy of the quantized/compatibility vision path, especially on
   temporal reasoning.
2. The omitted deep-stack vision features and auxiliary visual-mask inputs.
3. Context pressure from 256 visual tokens per 512 × 512 temporal patch.
4. QAIRT compiler/runtime version skew in future toolchain upgrades.

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
  -> pair two ordered 512 × 512 frames
  -> Hugging Face Qwen3-VL video processor
  -> pixel tensor [1024, 1536], grid [1, 32, 32]
  -> W4A16 vision context on QnnHtp
  -> 256 visual tokens
  -> timestamp + interleaved MRoPE visual record
  -> W4/FP16 text contexts on QnnHtp
```

Legacy Genie can repeat the text/vision append sequence, so the design scales
to multiple temporal patches without changing the orchestration model.
However, the current CL512 graph fits only one pair: the measured near-miss
prompt consumes 305 tokens and leaves 207. CL2048 text contexts can hold four
current pairs, while a future 256 × 256 vision profile would reduce each pair
to 64 tokens.

Two NVIDIA Isaac Sim warehouse prediction cases confirm that one temporal
patch reaches the NPU: both processes exit 0 and initialize the vision and
text contexts through `QnnHtp`. Both answers fail the semantic rubric. The
same near-miss input produces a materially better BF16 CUDA hazard answer,
although BF16 also misses the exact staged dodge outcome. This establishes an
additional quality loss in the current NPU artifact without claiming a full
predictive pass. See
[`video_npu.md`](video_npu.md)
and
[`iq9075_video_smoke_r1.json`](evidence/iq9075_video_smoke_r1.json).

## Bring-up order

1. Run the public GGUF on the EVK CPU to validate model/tokenizer/projector.
2. Quantize a 512-context W4A16 checkpoint with reduced calibration samples.
3. Apply the QAIRT 2.45 interface transform.
4. Convert activation encodings to FP16 while preserving calibrated weights.
5. Run a bounded host QuantSim smoke.
6. With explicit authorization for the Qualcomm AI Hub upload, export four
   text parts plus the vision encoder for IQ-9075.
7. Run a deterministic text-only prompt with `genie-t2t-run`.
8. Run the bundled sample image through `genie-app`.
9. Pack one two-frame temporal patch and run the pinned Isaac Sim prediction
   cases.
10. Repeat quantization/export at 2048 context and test a 256 × 256 vision
    profile after the smoke bundle is stable.

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
  reasoning. Benchmark answers must be scored against known outcomes and an
  upstream BF16 reference.

The sanitized evidence is in
[`docs/evidence`](evidence/README.md).
No earlier independent public IQ-9075 result was found for the exact
Cosmos-Reason2-2B checkpoint. NVIDIA's
[published Cosmos validation list](https://docs.nvidia.com/cosmos/latest/prerequisites.html)
excludes Qualcomm; Qualcomm's related IQ-9075 result is for
[Qwen3-VL-2B-Instruct GGUF](https://aihub.qualcomm.com/models/qwen3_vl_2b_instruct?runtime=geniex_qairt%2Cgeniex_llamacpp).
This project evidence is not vendor certification.
