from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


SOURCE = r"""
import json
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift1"
stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()


def aligned_bounds(path):
    prim = stage.GetPrimAtPath(path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=False,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    return {
        "min": [round(float(value), 5) for value in minimum],
        "max": [round(float(value), 5) for value in maximum],
        "dimensions": [
            round(float(maximum[index] - minimum[index]), 5)
            for index in range(3)
        ],
    }


def body_state(kind):
    path = f"{ROOT}{kind}"
    prim = stage.GetPrimAtPath(path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    position = transform.ExtractTranslation()
    collision = stage.GetPrimAtPath(f"{path}/Collision")
    visual_colliders = []
    for child in Usd.PrimRange(stage.GetPrimAtPath(f"{path}/Asset")):
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled_attr = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr()
        visual_colliders.append({
            "path": str(child.GetPath()),
            "enabled": bool(enabled_attr.Get()),
        })
    return {
        "path": path,
        "position": [round(float(value), 5) for value in position],
        "mass_kg": float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()),
        "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
        "simple_collider_enabled": bool(
            UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
        ),
        "collision_bounds": aligned_bounds(f"{path}/Collision"),
        "runner_bounds": (
            [
                aligned_bounds(f"{path}/{name}")
                for name in ("LeftRunner", "CenterRunner", "RightRunner")
            ]
            if kind == "Pallet"
            else []
        ),
        "visual_bounds": aligned_bounds(f"{path}/Asset"),
        "referenced_colliders": visual_colliders,
    }


was_playing = timeline.is_playing()
if not was_playing:
    timeline.play()
for _ in range(30):
    await omni.kit.app.get_app().next_update_async()
pallet = body_state("Pallet")
carton = body_state("Carton")
pallet_top = pallet["collision_bounds"]["max"][2]
carton_bottom = carton["collision_bounds"]["min"][2]
result = {
    "pallet": pallet,
    "carton": carton,
    "carton_to_pallet_vertical_gap_m": round(carton_bottom - pallet_top, 5),
    "carton_supported_by_pallet": (
        abs(carton_bottom - pallet_top) < 0.01
    ),
    "pallet_supported_above_floor": all(
        bounds["min"][2] > 0.04
        for bounds in pallet["runner_bounds"]
    ),
    "only_simple_colliders_enabled": all(
        not item["enabled"]
        for item in (
            pallet["referenced_colliders"]
            + carton["referenced_colliders"]
        )
    ),
}
print(json.dumps(result))
if not was_playing:
    timeline.stop()
""".strip()


def main() -> None:
    result = IsaacTcpClient(timeout_seconds=180.0).execute(SOURCE)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
