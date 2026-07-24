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
strongest result uses three temporal pairs and the same answer set for two
videos, then shuffles that set. BF16 GPU and NPU both answer `A`, `C`, `B`,
`A` across the barrier-normal, barrier-shuffled, box-normal, and box-shuffled
cases. NPU vision-plus-prefill TTFT is 730.5–745.8 ms.

This 4/4 controlled result changes with both scene and option position, so it
is evidence of video-conditioned NPU behavior rather than a fixed answer.
It is not a general accuracy claim: a separate tailored barrier choice and
both strict three-pair free-form rubrics fail. See the
[sanitized r3 evidence](../../docs/evidence/iq9075_video_aspect_native_parity_r3.json).

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
