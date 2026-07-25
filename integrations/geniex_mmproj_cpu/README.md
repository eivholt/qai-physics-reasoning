# GenieX multimodal-projector CPU placement

This integration contains the minimal experimental GenieX v0.3.17 patch used
to keep the multimodal projector on CPU while the Cosmos-Reason2-2B decoder
remains on the IQ-9075 NPU.

Apply
[`patches/0001-add-mmproj-cpu-toggle.patch`](patches/0001-add-mmproj-cpu-toggle.patch)
to a clean `qualcomm/GenieX` v0.3.17 checkout, build the Linux ARM64 SDK with
Qualcomm's official toolchain container, and set
`GENIEX_EXPERIMENT_MMPROJ_CPU=1` for the quality-first profile. Without that
environment variable, the patch does not change placement.

The unmodified stock runtime also has a faster compromise:

```bash
export GGML_HEXAGON_OPFILTER=GELU
```

That documented Hexagon filter fixes the observed NPU-only shuffled-box
regression with much less latency, but it does not reach the quality-first
profile's complete answer parity with the recorded BF16 GPU panel.

See the
[developer tutorial](../../docs/developer_tutorial.md#quality-versus-latency-placement-profiles)
and
[sanitized evidence](../../docs/evidence/iq9075_geniex_vision_placement_r8.json)
for commands, measurements, and limitations.
