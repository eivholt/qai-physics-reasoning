# Full-DeepStack raw-video runner for GenieX

This overlay connects already-packed Qwen3-VL temporal tensors to Qualcomm
GenieX's QAIRT VLM runtime. It does not decode a video container. Use
`scripts/prepare_video_npu_inputs.py` first to pack exactly `2N` extracted
frames into `N` temporal-pair tensors.

Shapes are derived from the bundle metadata rather than hardcoded. The
current IQ-9075 profile is 224 × 384: every pair is one float32
`[336, 1536]` tensor with grid `[1, 14, 24]` and produces 84 primary visual
features plus three 84-row DeepStack outputs. The runner inserts exactly 84
`<|video_pad|>` tokens per pair, calls GenieX's Qwen3-VL model directly,
constructs `visual_pos_masks`, and feeds the full first text shard.

Each two-frame pair is one temporal patch. Because
`temporal_patch_size = 2`, its compiled vision grid has temporal extent
`T = 1`; the 84 visual tokens are arranged as `1 × 7 × 12` after spatial
merge. Motion within a pair is encoded by the vision tower. For multiple
pairs, the runner interleaves one increasing timestamp and one
temporal/height/width MRoPE record per pair, using section `[24, 20, 20]`.
Treating the frames as independent MRoPE images is a different contract and
is rejected.

The deployed text graphs use AR128 prefill and AR1 decode at CL512. GenieX
v0.3.16 can safely transfer at most `CL - AR = 384` prompt tokens into the
decode cache. The preparer, package verifier, and runner all enforce this
limit. Three current pairs fit; measured four-pair prompts of 408 and 413 do
not.

## Measured result

The runner has executed the bias-corrected 224 × 384 full-DeepStack CL512
bundle on a physical IQ-9075 through `QnnHtp`, with HTP v73 detected. The
historical r3 two-scene control answered `A`, `C`, `B`, `A` on both BF16 GPU
and NPU, with NPU TTFT of 730.5–745.8 ms. The broader historical r4 suite
showed that 4/4 result did not generalize.

The completed r5 frozen suite contains 20 choice probes. BF16 GPU scores
16/20. The original native-aspect W8/A16 NPU baseline scores 11/20 with
13/20 exact GPU parity. The current NPU leader keeps nine graph boundaries at
A16 and 925 internal vision activations at FP16; it scores 13/20, reaches
15/20 exact GPU parity, retains 12/16 GPU-correct answers, and has mean TTFT
857.510 ms. This is improved video-conditioned inference, not GPU
equivalence. The r5 contexts still derive from the older
paired/explicit-resize calibration checkpoint; only runtime GPU/NPU pixels
use the byte-identical native Hugging Face path.

The part-1 W8 full, layers-0–3, and layers-0–2 candidates and the balanced
P1–P4 NPU extension remain pending only because the EVK is offline. No
accuracy is inferred for them. See the
[historical r3 evidence](../../docs/evidence/iq9075_video_aspect_native_parity_r3.json),
[historical r4 evidence](../../docs/evidence/iq9075_video_four_scene_parity_r4.json),
and
[r5 precision evidence](../../docs/evidence/iq9075_video_precision_parity_r5.json).

### Context contract for precision candidates

Every text context used by this runner contains two graphs in the order
AR128 prefill then AR1 decode. Both graphs must report the same nonzero
shared-weight size. A no-sharing part-1 link expanded to 711,176,192 bytes and
failed on the EVK while mapping a 490,733,568-byte FastRPC buffer, so graph
order alone is insufficient.

If AI Hub reuses an upload identity and restores the wrong graph order,
`scripts/cache_bust_dlc.py` can create a new DLC hash by changing only its
validated ZIP comment. Upload the resulting AR128 artifact before AR1, link
the new identities, and reinspect the output. The accepted full-W8 part-1
context is 357,736,448 bytes, SHA-256
`8ea7ba19baaaa90cbb8e1ee34ab07f37ea982860d3718d96a2a3d3405a341125`,
reports 353,431,552 shared-weight bytes for each graph, and passes a
four-token HTP smoke. Its frozen-suite score remains pending.

## Why the small GenieX patch is required

GenieX v0.3.16 already implements DeepStack extraction and injection, but its
generic hidden-state inference does not classify `visual_pos_masks` or
`deepstack_visual_embeds_*` as auxiliary tensors. The current compiler happens
to place `inputs_embeds` first, so the graph works today, but a harmless QNN
input reorder could silently select a DeepStack tensor as the inter-shard
state.

`patches/0001-deepstack-inputs-are-special.patch` fixes that classification.
The build script copies the pinned vendor checkout to its build directory and
patches only the copy. The runner also tests the linked classifier at startup
and refuses to execute an unpatched library. Package preparation independently
checks that `inputs_embeds` is still the first legacy non-special input in the
compiled part-1 metadata.

The build additionally requires clean GenieX, `geniex-qairt`, and
`geniex-proc` worktrees at the recorded commits, checks the recursive QAIRT
submodules, and uses Qualcomm's v0.0.1 toolchain image by immutable manifest
digest. A custom `--toolchain-image` is accepted only when it is also pinned
with `@sha256:...`.

## Commands

Use explicit, new output directories. The build command is the only
resource-intensive step and is not run automatically.

```bash
bash scripts/geniex_raw_video.sh build \
  --geniex-source /path/to/GenieX \
  --build-dir /path/to/new/geniex-raw-video-build

bash scripts/geniex_raw_video.sh package \
  --bundle /path/to/full-deepstack-cl512-bundle \
  --video-input-dir /path/to/existing/video-input \
  --package-dir /path/to/new/run-package

EVK_TARGET="ubuntu@<EVK IP>"

bash scripts/geniex_raw_video.sh deploy \
  --build-dir /path/to/new/geniex-raw-video-build \
  --package-dir /path/to/new/run-package \
  --evk "$EVK_TARGET" \
  --remote-root /path/to/new/remote-run

bash scripts/geniex_raw_video.sh run \
  --evk "$EVK_TARGET" \
  --remote-root /path/to/new/remote-run \
  --bundle /path/to/remote/full-deepstack-cl512-bundle \
  --max-tokens 64 \
  --verbose
```

`package` binds the selected input to the exact target bundle. `run` repeats
SHA-256 and shape/config checks on the EVK before launching the NPU binary.
It rejects compatibility bundles with removed text-side DeepStack inputs,
changed model-context binaries, path traversal, wrong token layout, and
prompts above the metadata-derived safe prefill limit. This hash-bound manifest
is not a digital signature, and it does not authenticate the runner, shared
libraries, Python launcher, or HTP runtime; retain their build provenance
separately.

The deploy command deliberately does not copy the multi-gigabyte model bundle
and refuses to overwrite an existing remote run directory.
