#!/usr/bin/env python3
"""Flatten the active conveyor composition into a small, deterministic USD source.

The full Omniverse collection remains the immutable source cache.  This step
removes only scene branches that have no role in the interactive conveyor demo,
then flattens the remaining composition so the Unreal import does not traverse
hundreds of library layers.  Run with Isaac Sim's python.bat/python.sh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_DIR
    / "Content"
    / "OmniversePayload"
    / "reason2_conveyor_safety_stable_cargo_r38.usda"
)
DEFAULT_OUTPUT = PROJECT_DIR / "Saved" / "LeanStage" / "warehouse_conveyor_runtime.usdc"

REQUIRED_PRIMS = (
    "/World/CodexPoC/ConveyorSafety/Conveyor/Parcels/Parcel1",
    "/World/CodexPoC/ConveyorSafety/Forklifts/Forklift1",
    "/World/CodexPoC/ConveyorSafety/Forklifts/Forklift1/body/DriverMount",
    "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift1Pallet",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorEndline",
    "/World/CodexPoC/ConveyorSafety/Workers/Worker1",
    "/World/CodexPoC/ConveyorSafety/Workers/Worker2",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/WalkForwardLoop",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/WalkForwardLoop_01",
    "/World/CodexPoC/ConveyorSafety/Workers/DriverAnimations/SeatedDriverPose",
)

# These are deliberately narrow, audited removals.  Invisible collision and
# control geometry elsewhere is retained because visibility alone does not make
# a prim unused.
PRUNED_BRANCHES = (
    "/Render",
    "/World/CodexPoC/ConveyorSafety/FloorInset",
    # The client has one interactive forklift. Remove the unused authored
    # vehicle and its separate cargo before flattening/importing so their mesh,
    # material, skeleton, collision, and animation resources are not cooked.
    "/World/CodexPoC/ConveyorSafety/Forklifts/Forklift2",
    "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift2Pallet",
    "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift2Carton",
    # Keep the two walking workers, the one seated forklift driver, and only
    # the clips those three visible people actually bind.
    "/World/CodexPoC/ConveyorSafety/Workers/HumanMotionLibrary",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/WalkForward",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/WalkForward_01",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/Idle",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/IdleTired",
    "/World/CodexPoC/ConveyorSafety/Workers/RetargetedAnimations/SitAndStandChair",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorOblique",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorOverhead",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorPTZ",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorSide",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorIsometric",
    "/World/CodexPoC/ConveyorSafety/Cameras/Presentation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true")
    args, _ = parser.parse_known_args()
    return args


ARGS = parse_args()

from isaacsim import SimulationApp  # noqa: E402


APP = SimulationApp({"headless": True})

from pxr import Kind, Sdf, Usd, UsdGeom, UsdUtils  # noqa: E402


KIND_OVERRIDES = {
    # Prevent the importer from combining the complete belt (and its many
    # materials) into one mesh.  Each referenced module remains a component,
    # which lets Unreal instance the repeated straights/curves and use Nanite.
    "/World/CodexPoC/ConveyorSafety/Conveyor": Kind.Tokens.group,
    "/World/CodexPoC/ConveyorSafety/Conveyor/Modules": Kind.Tokens.assembly,
    "/World/CodexPoC/ConveyorSafety/Conveyor/Parcels": Kind.Tokens.group,
}


def is_under(path: Sdf.Path, roots: tuple[Sdf.Path, ...]) -> bool:
    return any(path == root or path.HasPrefix(root) for root in roots)


def composed_stats(stage: Usd.Stage) -> dict[str, int]:
    prim_count = 0
    visible = 0
    invisible = 0
    for prim in stage.Traverse():
        prim_count += 1
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                invisible += 1
            else:
                visible += 1
    return {
        "prim_count": prim_count,
        "visible_imageables": visible,
        "invisible_imageables": invisible,
    }


def main() -> int:
    source = ARGS.source.resolve()
    output = ARGS.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Collected conveyor source is missing: {source}")
    if output.exists() and not ARGS.refresh:
        raise FileExistsError(f"Refusing to replace {output}; pass --refresh for a generated stage")
    output.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not compose {source}")
    missing = [path for path in REQUIRED_PRIMS if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Source stage is missing required runtime prims: {missing}")
    before = composed_stats(stage)

    prune_paths = tuple(Sdf.Path(path) for path in PRUNED_BRANCHES)
    stage.SetEditTarget(stage.GetSessionLayer())
    for path in PRUNED_BRANCHES:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Audited prune path is missing from source: {path}")
        prim.SetActive(False)

    for path, kind in KIND_OVERRIDES.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Audited kind override prim is missing from source: {path}")
        Usd.ModelAPI(prim).SetKind(kind)

    # Remove live relationship/connection targets into every discarded branch.
    disconnected_targets = 0
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            kept = [target for target in targets if not is_under(target.GetPrimPath(), prune_paths)]
            if kept != targets:
                relationship.SetTargets(kept)
                disconnected_targets += len(targets) - len(kept)
        for attribute in prim.GetAttributes():
            connections = attribute.GetConnections()
            kept = [target for target in connections if not is_under(target.GetPrimPath(), prune_paths)]
            if kept != connections:
                attribute.SetConnections(kept)
                disconnected_targets += len(connections) - len(kept)

    flattened = stage.Flatten()
    if flattened is None or not flattened.Export(str(output)):
        raise RuntimeError(f"USD could not export flattened stage: {output}")

    lean = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
    if lean is None:
        raise RuntimeError(f"Flattened stage could not reopen: {output}")
    missing = [path for path in REQUIRED_PRIMS if not lean.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Flattening removed required runtime prims: {missing}")
    leaked = [
        path
        for path in PRUNED_BRANCHES
        if lean.GetPrimAtPath(path).IsValid() and lean.GetPrimAtPath(path).IsActive()
    ]
    if leaked:
        raise RuntimeError(f"Pruned branches are still active in flattened stage: {leaked}")

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(output)))
    if unresolved:
        raise RuntimeError(f"Lean stage has unresolved dependencies: {list(unresolved)[:20]}")
    after = composed_stats(lean)
    report = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "pruned_branches": list(PRUNED_BRANCHES),
        "kind_overrides": {path: str(kind) for path, kind in KIND_OVERRIDES.items()},
        "disconnected_targets": disconnected_targets,
        "before": before,
        "after": after,
        "dependency_layers": len(layers),
        "referenced_assets": len(assets),
        "output_bytes": output.stat().st_size,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("LEAN_CONVEYOR_STAGE=" + json.dumps(report, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    finally:
        APP.close()
    raise SystemExit(exit_code)
