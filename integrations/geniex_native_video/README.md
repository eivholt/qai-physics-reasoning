# Native temporal video through GenieX

This overlay repairs the encoded-video path in Qualcomm GenieX v0.3.17's
`llama_cpp` VLM plugin. It is intended for Qwen3-VL-family GGUF models such as
Cosmos-Reason2-2B.

The stock plugin keeps the lazy `mtmd_bitmap` returned for an encoded video but
drops the accompanying `mtmd_helper_video` pointer. The allocation therefore
remains live until process exit. In a persistent `geniex serve` process, every
video request leaks the decoder owner, file descriptors, threads, and reaping
state for the `ffmpeg`/`ffprobe` subprocesses. Patch 0001 retains the owner with
RAII until tokenization has expanded the lazy video, then frees it normally.
Patch 0003 fixes a second cleanup defect: the helper clears its wrapper-level
`alive` flag when it reaches video EOF and then skips `subprocess_join()` and
`subprocess_destroy()`. The patch reaps every created child and closes its
pipes even after normal EOF.

Patch 0002 adds two opt-in environment variables while preserving llama.cpp's
upstream defaults:

- `MTMD_VIDEO_FPS` controls decoded frame sampling. Use `2` for the current
  full-NPU short-clip profile; the earlier r9 profile used `4`.
- `MTMD_VIDEO_TIMESTAMP_INTERVAL_MS` controls generic timestamp text. Use `0`
  for short Qwen3-VL clips so timestamp text does not split adjacent frames
  before the temporal-patch merger can pair them.

Patch 0004 wires the server's existing `grammar_path` and `grammar_string`
request fields into `SamplerConfig`. Stock v0.3.17 declares and accepts both
JSON fields but silently drops them in `parseSamplerConfig`. For bounded
choice probes, a request can then use a grammar such as `root ::= [ABCD]` to
prevent invalid out-of-set answers.

Apply all four patches to a clean official GenieX v0.3.17 checkout:

```bash
git apply --unidiff-zero /path/to/qai-physics-reasoning/integrations/geniex_native_video/patches/0001-retain-mtmd-video-context.patch
git apply --unidiff-zero /path/to/qai-physics-reasoning/integrations/geniex_native_video/patches/0002-configure-mtmd-video-sampling.patch
git apply --unidiff-zero /path/to/qai-physics-reasoning/integrations/geniex_native_video/patches/0003-reap-mtmd-video-subprocesses.patch
git apply --unidiff-zero /path/to/qai-physics-reasoning/integrations/geniex_native_video/patches/0004-wire-server-grammar-fields.patch
```

Build with `MTMD_VIDEO=ON`. The target needs `ffmpeg` and `ffprobe`.

For the fast persistent NPU service:

```bash
export GENIEX_DATADIR=/path/to/geniex-data
export MTMD_VIDEO_FPS=2
export MTMD_VIDEO_TIMESTAMP_INTERVAL_MS=0

geniex --skip-update serve \
  --host 127.0.0.1:18181 \
  --keepalive 3600 \
  --compute npu \
  --nctx 4096 \
  --ngl -1
```

For the measured quality-first service, also export
`GENIEX_EXPERIMENT_MMPROJ_CPU=1` and apply the opt-in patch in
[`../geniex_mmproj_cpu`](../geniex_mmproj_cpu). The vision encoder/projector
then runs on CPU while the 28-layer language decoder remains on NPU.

Submit the video URL in an OpenAI-compatible `image_url` content part. GenieX
copies it to a MIME-typed temporary file, mtmd decodes it, and Qwen3-VL merges
successive frames in pairs before vision encoding. Keep `GenieX-KeepCache`
unset for independent requests; the server resets the KV cache but retains the
loaded model.

For a four-choice benchmark, constrain the response without changing the
question:

```json
{
  "top_k": 1,
  "temperature": 0,
  "seed": 42,
  "grammar_string": "root ::= [ABCD]"
}
```

At 2 FPS, the full-NPU service returns `A C B A C B C A` on the frozen
four-scene/two-order encoded-MP4 panel: 7/8 at a repeated 1.453-second warm
mean. This matches the r9 CPU-vision/NPU-decoder answer sequence while being
about 4.8 times faster. One FPS loses box pickup, and three or four FPS
reproduce the `C`-biased barrier/box failures; each scores 5/8. The recorded
BF16 GPU run returns 6/8 on this panel. This is a bounded benchmark, not
evidence that Q4_0 generally outperforms BF16. The two paths use different
video processors, chat wrappers, and numerical precision.

The grammar is material only when the task has a closed answer set. It fixed
the last malformed `C` in the A/B box-versus-near-miss control, taking the
quality-first service to 4/4, but it cannot repair a wrong in-set prediction:
the remaining fire-onset miss is still `C` instead of `D`.

For the four declared warehouse event families, the tracked two-request
classifier removes four-way option competition:

```bash
python scripts/classify_geniex_warehouse_video.py \
  --video /path/visible/to/server/clip.mp4
```

It first distinguishes forklift-prominent from worker-only activity, then
asks a branch-specific A/B question. The physical EVK run classifies all four
clips correctly with a 2.90-second warm mean. This is a task-specific
classifier, not a zero-shot accuracy claim. A fully balanced four-permutation
diagnostic remains only 10/16, so the simpler 7/8 panel does not establish
general option-order robustness.

Use independent HTTP connections (`Connection: close`) and supervise the
worker. A 48-request 2 FPS soak retains 26 file descriptors, plateaus at 49
threads, and leaves no ffmpeg/ffprobe children. A later extended run stalls
after 54 completed video requests. Until that NPU-service limit is resolved,
recycle the worker conservatively before 40 requests and retain board-reboot
recovery for loss of the CDSP device.

This path supplies paired temporal patches to the GGUF vision tower, but it is
not bit-exact with the Hugging Face processor or the project's QAIRT
Full-DeepStack raw-video runner. In particular, generic mtmd preprocessing does
not expose the project's explicit pair-midpoint timestamp and raw
`visual_pos_masks` contract. Use
[`../geniex_raw_video/README.md`](../geniex_raw_video/README.md) when that
exact native QAIRT contract matters.
