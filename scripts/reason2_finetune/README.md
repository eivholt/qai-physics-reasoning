# Reason2 SpeedV1 fine-tuning and release

This directory contains the retained production path for the Unreal parcel
safety classifier. Historical adapter, prompt, camera, and transport
experiments were removed after SpeedV1 passed its release gates.

## Contract

The model receives one 448 x 256 lossless sensor image and this user-only task:

```text
Classify cartons at the central silver conveyor lane. Ignore all other objects and lanes. Answer one letter only: G=fully supported; A=touching rollers but at least one-third beyond a blue rail or tipping; R=on the adjacent floor below the rollers.
```

Training completions contain exactly one token: `G`, `A`, or `R`. Runtime uses
deterministic sampling, disables hidden reasoning, and stops after the first
token. The client derives the longer UI schema without another model call.

## Final workflow

1. Generate synchronized, balanced Unreal images with
   `generate_unreal_dataset.ps1`.
2. Train the 3,000-image SpeedV1 adapter with `train_evk_speed_v1.sh`.
3. Run frozen validation and independent test gates with
   `evaluate_evk_speed_v1.sh`.
4. Build host BF16/Q8 artifacts with `build_host_speed_v1.sh`, then gate both
   conversions with `gate_host_speed_v1_gguf.sh`.
5. Build the CL512/W8 full-DeepStack QAIRT bundle with
   `build_evk_speed_v1.sh`. `prepare_task_vision_calibration.py` prepares the
   matched 448 x 256 calibration manifest when it must be regenerated.
6. Stage and validate the bundle with `scripts/stage_evk_geniex_bundle.sh` and
   `scripts/run_evk_geniex_candidate_gate.sh`.
7. Prove the packaged Unreal path with
   `unreal_conveyor_demo/Scripts/smoke_packaged_speed_v1.ps1`.

The Python modules retained here are direct dependencies of those steps:

- `common.py` and `generative_common.py`: dataset discovery, class mapping,
  prompt, parsing, and metrics;
- `train_generative.py` and `evaluate_generative.py`: adapter training and
  frozen evaluation;
- `merge_adapter.py`: deterministic adapter merge for host and QAIRT export;
- `benchmark_adapter_api.py` and `validate_evk_http_service.py`: host/EVK gate
  clients.

## Promotion gates

- 180/180 host validation and 90/90 independent host test, with perfect
  per-class recall;
- 90/90 EVK direct-image validation, 30 per class;
- 360/360 sustained EVK soak;
- clean nine-case packaged-client gate against host and EVK;
- exactly one resident QAIRT worker and no legacy media bridge or file polling.

Large datasets, checkpoints, adapters, GGUFs, QAIRT graphs, and generated Unreal
assets remain outside Git. The concise published results are linked from
`docs/isaac_sim_conveyor_reason2_tutorial.md`.
