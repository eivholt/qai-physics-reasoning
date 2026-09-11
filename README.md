# Reason2 Conveyor Safety on Qualcomm IQ-9075

A native Unreal Engine warehouse demo compares a task-specific NVIDIA
Cosmos-Reason2-2B model on an NVIDIA host GPU and a Qualcomm Dragonwing IQ-9075
EVK. The current model, `Cosmos-Reason2-2B-Parcel-Speed-v1` (SpeedV1), classifies
cartons on a conveyor from one image and returns one token: `G`, `A`, or `R`.

The client uses imported NVIDIA Omniverse warehouse, conveyor, forklift,
worker, and parcel assets with native Chaos physics and Lumen rendering.
Isaac Sim was used for the earlier prototypes; it is not required to run the
packaged Unreal client.

[![Watch the Unreal conveyor demo on YouTube](https://img.youtube.com/vi/t5t9_izbAMc/hqdefault.jpg)](https://youtu.be/t5t9_izbAMc)

Start with the [tutorial](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/isaac_sim_conveyor_reason2_tutorial.md)
for the experiment history, prompting choices, fine-tuning, and measurements.
The [Unreal client guide](https://github.com/eivholt/qai-physics-reasoning/blob/main/unreal_conveyor_demo/README.md)
covers developer launch, source assets, packaging, and smoke tests.

## Current demo

| Component | Production configuration |
|---|---|
| Windows client | Unreal Engine 5.8.1; D3D12, Chaos, and Lumen |
| Model input | One lossless 448 × 256 PNG from the fixed conveyor sensor |
| Prompt and output | One user prompt, no system or secondary prompt, one deterministic `G/A/R` token |
| Host GPU | SpeedV1 Q8_0 GGUF + F16 projector, llama.cpp, port 18084 |
| EVK NPU | SpeedV1 full-DeepStack QAIRT bundle with W8 text, GenieX v0.3.17, port 18183 |
| EVK platform | Dragonwing IQ-9075, Hexagon v73, OS 1.9, QAIRT 2.45.0.260326 |

The sensor image and prompt are sent to the selected backend through
`/v1/chat/completions`. The EVK request embeds the PNG directly as a
`data:image/png;base64,...` item. One resident GenieX worker keeps the QAIRT
graphs loaded; no media bridge, filesystem polling, or per-request model
startup is used. The client maps the returned token to the stack light and HUD:

- `G` — GREEN: cartons fully supported by the monitored conveyor.
- `A` — AMBER: a carton touches the rollers but extends at least one-third
  beyond a blue rail, or is tipping.
- `R` — RED: a monitored carton is on the adjacent floor below the rollers.

This is a single-image spatial policy, not an inference about when or how a
carton fell. Both backends use the same camera, prompt, and class mapping.
Press **B** or controller **Start** to switch backends.

The production EVK bundle is:

```text
cosmos-reason2-parcel-speed-v1-cl512-256x448-w8text-geniex-qairt245-os19-r1
```

## Quick start

### Packaged Windows release

Use a complete release archive containing `Reason2-Conveyor-Setup.exe`,
`Game`, and `Payload`. These binaries, model weights, and generated Unreal
assets are not included in a source clone.

The accepted GPU presentation uses Windows 11 x64 and an NVIDIA RTX GPU with
a compatible driver. EVK operation additionally needs the OS 1.9 board,
network access to it, and the SSH access required by the provisioner. Unreal
Editor, WSL, and training/export tools are not needed to run a prepared release.

1. Extract the complete archive to a short path, such as
   `C:\Reason2-Conveyor`; do not run Setup from inside the ZIP.
2. Verify the archive/artifact checksums against the trusted release records.
3. Open PowerShell in the extracted directory and replace `<EVK-IP>` below:

```powershell
.\Reason2-Conveyor-Setup.exe install `
  --app-dir .\Game `
  --payload-root .\Payload `
  --evk-host "<EVK-IP>" `
  --require-evk
```

Setup verifies the payload, starts the host model, provisions the resident EVK
service, writes the client configuration, and launches the demo. For a
host-only installation, replace the two EVK options with `--skip-evk`.
The [release instructions](https://github.com/eivholt/qai-physics-reasoning/blob/main/unreal_conveyor_demo/Provisioner/windows-release/README.txt)
also cover repair and support bundles.

### Prepared developer checkout

After the packaged client, host artifacts, and CUDA runtime have been prepared,
run this from the repository root in Windows PowerShell:

```powershell
.\unreal_conveyor_demo\Scripts\run_windows_demo.ps1
```

The launcher verifies or starts the SpeedV1 GPU service before opening the
newest local package. It does not install artifacts or start the EVK service.
Use the [developer launch instructions](https://github.com/eivholt/qai-physics-reasoning/blob/main/unreal_conveyor_demo/README.md#developer-launch)
to select an EVK address or pin a particular package. Opening the game
executable directly does not start the host model.

## Accepted measurements

These are frozen SpeedV1 task results recorded on 2026-08-19, not a general
warehouse-reasoning benchmark. Timing is warm request latency from the linked
records, excluding the first cold request; it is not rendering frame time or
a guaranteed end-to-end safety response time.

| Gate and evidence | Correct | Warm mean |
|---|---:|---:|
| [Host BF16 validation](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/host_bf16_speed_v1_validation180_20260819.json) | 180/180 | 58.9 ms |
| [Host Q8 independent test](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/host_q8_0_speed_v1_test90_20260819.json) | 90/90 | 64.6 ms |
| [EVK direct-image test](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/geniex_speed_v1_direct_stream_test90_20260819.json) | 90/90 | 685.4 ms |
| [EVK sustained soak](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/geniex_speed_v1_direct_stream_soak360_20260819.json) | 360/360 | 687.3 ms |

All classes passed. The soak repeats the same balanced 90-image panel four
times; it is not 360 unique test images. EVK p95 latency was 694.0 ms for the
90-image test and 691.0 ms for the soak.

The separate [host Q8 deployment check](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/host_q8_speed_v1_deployment_smoke9_20260819.json)
passed 9/9 at 66.7 ms warm mean. Packaged in-scene gates passed
[9/9 on the host](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/packaged_host_speed_v1_deployment_defaults_smoke9_20260819.json)
and [9/9 on the EVK](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/evidence/results/packaged_speed_v1_production_defaults_smoke9_20260819.json).
BF16 is the reference export; Q8 is the distributed host default.

## How SpeedV1 is built

SpeedV1 was trained on 3,000 balanced synchronized-v22 Unreal images, 1,000 per
class. LoRA fine-tunes visual and language projections with a BF16 base and
one-token training completions. The learned adapter is merged into the base
checkpoint before producing the host GGUF and EVK QAIRT exports. Production
inference does not load a separate LoRA adapter.

The [retained fine-tuning workflow](https://github.com/eivholt/qai-physics-reasoning/blob/main/scripts/reason2_finetune/README.md)
covers data generation, training, frozen validation, host conversion, EVK
export, and promotion. Training/export needs its own CUDA/WSL environment,
base-checkpoint access, and QAI Hub tooling; those are build requirements,
not packaged-demo prerequisites. Script defaults include local workspace
paths, so configure them for the target machine before running.

Large datasets, checkpoints, GGUFs, QAIRT graphs, generated `.uasset` files,
and credentials remain outside Git. The
[release payload template](https://github.com/eivholt/qai-physics-reasoning/blob/main/unreal_conveyor_demo/Build/release_payload_template.json)
identifies the accepted deployment artifacts and hashes. Base-model and
third-party asset access and license requirements still apply.

## Earlier experiments

The repository also retains the Isaac Sim supervisor and general-model video
experiments described in the tutorial. Their prompts, media formats, scores,
and latency measurements are historical configurations, not the current
SpeedV1 deployment or evidence of broad GPU/NPU parity.

- [Earlier adapter/export workflow](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/developer_tutorial.md)
- [Engineering and quantization history](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/engineering_appendix.md)
- [Earlier video input and parity experiments](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/video_npu.md)
- [Isaac Sim warehouse supervisor](https://github.com/eivholt/qai-physics-reasoning/blob/main/docs/isaac_sim_edge_supervisor_tutorial.md)

Legacy Genie, paired-frame video, and historical worker-recycling instructions
are not the production parcel-classifier path.

## Limits and safety

SpeedV1 is specialized to the frozen conveyor camera, image geometry, prompt,
and three-class policy. Successful gates do not establish general scene
understanding or safety certification. A production conveyor must retain an
independent conventional safety interlock. This demo is not NVIDIA or
Qualcomm certification.
