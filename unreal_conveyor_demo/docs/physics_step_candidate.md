# Solver-step force candidate

Experimental, opt-in Unreal 5.8 path for reducing low-frame-rate suspension and conveyor feedback oscillation. Enable with `-QaiPhysicsStepForces`; omit the flag to retain frame-based force submission. The accepted `FlangeGeometryFix` package is unchanged.

The candidate recalculates forklift suspension, tyre forces, lift targets and conveyor forces in `AsyncPhysicsTickActor`, using current Chaos solver transforms and velocities. Input is captured at the original game-frame submission point and held across its physics steps. Visual wheel transforms, telemetry and inference remain on the game thread.

This uses UE 5.8's frozen-game-thread actor callback, not an independently running simulation thread. Existing substepping limits remain: maximum step 8.333 ms, maximum eight substeps. It is not a strict fixed-120-Hz clock, and does not remove rendering bottlenecks or guarantee stability during arbitrarily long stalls.

## Local trial

From the repository root, with the existing EVK service running:

```powershell
& .\unreal_conveyor_demo\Scripts\run_windows_demo.ps1 `
  -ClientExecutable .\unreal_conveyor_demo\Saved\Packaged\PhysicsStepCandidate20260912\Windows\QaiConveyor.exe `
  -ClientArguments @('-Backend=host', '-QaiParcelPrompt=speed-v1',
    '-HostServer=http://127.0.0.1:18084',
    '-HostModel=Cosmos-Reason2-2B-Parcel-Speed-v1',
    '-EvkServer=http://192.168.1.158:18183', '-EvkModel=local/cosmos-reason2-2b',
    '-QaiPhysicsStepForces', '-windowed', '-noborder', '-ForceRes',
    '-ResX=2560', '-ResY=1440', '-WinX=0', '-WinY=0')
```

The dated candidate package is a local build artifact, not a downloadable release. It reuses the accepted cooked assets with a rebuilt Shipping executable; no geometry was recooked.

## Verification

Close the interactive demo before running the rate comparison. Use the nested Shipping executable, not the bootstrap:

```powershell
$Client = (Resolve-Path .\unreal_conveyor_demo\Saved\Packaged\PhysicsStepCandidate20260912\Windows\QaiConveyor\Binaries\Win64\QaiConveyor-Win64-Shipping.exe).Path
& .\unreal_conveyor_demo\Scripts\test_physics_step_rates.ps1 `
  -ClientExecutable $Client -FrameCaps @(30,45,60,120)
```

The harness renders offscreen at 2560x1440, disables inference, discards three seconds of settling, then measures idle chassis vertical range and vertical-velocity RMS. Reports under `Saved/Diagnostics` include actual achieved frame rate: a cap of 60 or 120 does not establish that rate was reached.

Initial local measurements at approximately 45 FPS reduced idle vertical range from 0.77 cm to 0.02 cm and vertical-velocity RMS from 5.38 to 0.16 cm/s. At approximately 30 FPS, the corresponding ranges were 11.05 cm and 0.016 cm. These are short idle tests, not guarantees for every collision or loaded forklift maneuver.

The final candidate also passed the automated lift/reset check and both nine-case SpeedV1 inference smoke tests. The smoke script accepts `-PhysicsStepForces`; use separate `-Output` paths under `Saved/Diagnostics` so trial results do not replace accepted release evidence. Inference cases validate capture and classification, not conveyor motion quality.
