# Repository agent memory

## Current production demo

- The warehouse client is native Unreal and retains the imported NVIDIA
  Omniverse conveyor, warehouse, forklift, worker, and parcel assets.
- The production parcel task uses one 448 x 256 lossless PNG, one user prompt,
  no system or secondary prompt, and one deterministic output token: `G`, `A`,
  or `R`. The client maps that token to the GREEN/AMBER/RED UI fields.
- Host inference uses `Cosmos-Reason2-2B-Parcel-Speed-v1` on port 18084. The
  distributed Q8 model is the release default; BF16 is the speed reference.
- EVK inference uses one resident GenieX v0.3.17 QAIRT worker on port 18183 and
  direct embedded PNG transport. Do not restore the retired media bridge,
  filesystem polling, forced connection close, or active legacy Genie path.
- Production EVK bundle:
  `cosmos-reason2-parcel-speed-v1-cl512-256x448-w8text-geniex-qairt245-os19-r1`.
  One labelled legacy rollback remains outside the active repository payload.

## Accepted measurements

- SpeedV1 training: 3,000 balanced synchronized-v22 images, 1,000 per class.
- Host and EVK gates passed every class: host 180/180 validation and 90/90
  independent test; EVK 90/90; sustained EVK soak 360/360.
- Host Q8 direct inference measured about 66.7 ms warm. EVK direct-image
  inference measured 685.4 ms warm mean and 694.0 ms p95; the 360-request soak
  measured 687.3 ms mean and 691.0 ms p95.
- Canonical evidence is limited to the release/comparison records linked from
  `docs/isaac_sim_conveyor_reason2_tutorial.md` under `docs/evidence/results/`.

## Reproducible paths

- Final fine-tune/export scripts live in `scripts/reason2_finetune/`; start with
  its README. Keep only the SpeedV1 path and its direct dependencies.
- `scripts/run_evk_geniex_candidate_gate.sh` owns exclusive candidate testing
  and restores the production worker afterward. Never run two resident EVK
  workers concurrently.
- `unreal_conveyor_demo/Scripts/package_windows_client.ps1` builds, cooks,
  stages, and optionally smoke-tests the Windows client.
- `unreal_conveyor_demo/Scripts/smoke_packaged_speed_v1.ps1` is the nine-case
  packaged inference gate.
- Omniverse extension generation/import is reproducible through
  `build_omniverse_conveyor_extensions.py` and
  `import_omniverse_conveyor_extensions.py`; final material generation and
  assignment is consolidated in `visual_finish.py`.

## Repository hygiene and safety

- Unreal `.uasset` files are intentionally local and ignored. Do not add them
  to Git. Keep the source/import scripts that recreate required generated
  assets from the Omniverse source.
- Do not reintroduce superseded prompt sweeps, adapter experiments, legacy
  deployment helpers, or bulk experimental evidence into the final demo
  commit.
- Preserve unrelated dirty-worktree changes. A production conveyor must retain
  an independent conventional safety interlock; this visual demo is not a
  machinery safety certification.

## Fork handoff checkpoint (2026-08-20)

- The A08 straight-section roller repair is accepted visually. It replaces
  each defective authored roller at runtime from one known-good Omniverse A08
  roller and must not be normalized back to raw mesh defaults.
- The A05/A08 blue frame material slot contains both the C-frame flanges and
  disconnected tall uprights. A height mask therefore cannot solve one without
  cutting the other. `apply_conveyor_visual_repairs.py` now derives two
  render-only Omniverse overlays by removing exactly two A05 and four A08
  upright geometry islands. The complete opaque, two-sided flanges remain;
  original source meshes retain collision; overlays have no collision.
- Generated overlay/material `.uasset` files remain intentionally ignored.
  Recreate them with `unreal_conveyor_demo/Scripts/apply_conveyor_visual_repairs.py`
  before a clean cook. The audit report is
  `Saved/ImportAudit/conveyor_visual_repairs.json` and must report 2 + 4 removed
  upright islands.
- The accepted local visual package is
  `unreal_conveyor_demo/Saved/Packaged/FlangeGeometryFix/Windows/QaiConveyor.exe`.
  It passed the packaged runtime smoke and the `conveyorseam` close-up showed
  uniform rollers, continuous opaque flanges, and no protruding blue uprights.
- A raw packaged client does not own host-model startup. For developer runs,
  use `unreal_conveyor_demo/Scripts/run_windows_demo.ps1`; it verifies or starts
  the exact Q8 SpeedV1 service on port 18084 before launching the newest client.
  The distribution setup/provisioner remains the production owner of startup.
- Live recovery proof on 2026-08-20: exact model and projector hashes matched
  the release manifest; direct multimodal inference returned `G` for a frozen
  GREEN case; the relaunched packaged client then logged host readiness and a
  first in-scene `G` response in 181 ms, with subsequent warm requests around
  102-118 ms.
