# Full-DeepStack raw-video runner for GenieX

This overlay is the smallest path from an already-packed 512 × 512
Qwen3-VL temporal tensor to Qualcomm GenieX's QAIRT VLM runtime. It does not
decode a video container. Use `scripts/prepare_video_npu_inputs.py` first to
pack exactly two extracted frames into one float32 `[1024, 1536]` tensor.

The standalone runner inserts exactly 256 `<|image_pad|>` tokens between the
prepared prefix and suffix, supplies grid `[1, 32, 32]`, and calls GenieX's
Qwen3-VL model directly. GenieX runs the vision context, reads
`image_features` plus `deepstack_visual_embeds_0..2`, constructs the
`visual_pos_masks` input, and feeds the full first text shard.

The two frames are one temporal patch. Because
`temporal_patch_size = 2`, their compiled vision grid has temporal extent
`T = 1`; the 256 visual tokens are arranged as `1 × 16 × 16` after spatial
merge. Motion within the pair is encoded by the vision tower. The text model's
Qwen3-VL MRoPE receives one temporal/height/width grid and uses stride
interleaving with section `[24, 20, 20]`. Treating the two frames as two
separate MRoPE images would be a different input contract and is rejected.

## Measured result

The runner has executed the full-DeepStack CL512 bundle on a physical
IQ-9075 through `QnnHtp`: HTP v73 was detected, vision plus prefill TTFT was
about 648 ms, and multi-token decoding reached 16.3–16.6 tok/s. Both
free-form NVIDIA Isaac Sim warehouse forecasts still failed their strict
future-event rubrics. A shelf multiple-choice answer that initially matched
the correct option failed after the options were shuffled, exposing a
position bias. The result therefore proves the paired-frame NPU path, not
robust video prediction. See the
[sanitized r2 evidence](../../docs/evidence/iq9075_video_deepstack_geniex_r2.json).

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
  --video-input-dir /path/to/existing/one-pair-video-input \
  --package-dir /path/to/new/run-package

bash scripts/geniex_raw_video.sh deploy \
  --build-dir /path/to/new/geniex-raw-video-build \
  --package-dir /path/to/new/run-package \
  --evk ubuntu@<EVK-IP> \
  --remote-root /home/ubuntu/cosmos-geniex-deepstack-test-r1

bash scripts/geniex_raw_video.sh run \
  --evk ubuntu@<EVK-IP> \
  --remote-root /home/ubuntu/cosmos-geniex-deepstack-test-r1 \
  --bundle /home/ubuntu/cosmos-reason2-full-deepstack-cl512-r1 \
  --max-tokens 64 \
  --verbose
```

`package` binds the selected input to the exact target bundle. `run` repeats
SHA-256 and shape/config checks on the EVK before launching the NPU binary.
It rejects compatibility bundles with removed text-side DeepStack inputs,
changed model-context binaries, path traversal, wrong token layout, and
prompt plus generation lengths over CL512. This hash-bound manifest is not a
digital signature, and it does not authenticate the runner, shared libraries,
Python launcher, or HTP runtime; retain their build provenance separately.

The deploy command deliberately does not copy the multi-gigabyte model bundle
and refuses to overwrite an existing remote run directory.
