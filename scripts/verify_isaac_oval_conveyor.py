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
import omni.timeline
import omni.usd
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

stage = omni.usd.get_context().get_stage()
root_path = "/World/CodexPoC/ConveyorSafety"
parcel_root = f"{root_path}/Conveyor/Parcels"
surface_root = f"{root_path}/Physics/StaticColliders/OvalConveyor"
root = stage.GetPrimAtPath(root_path)
parcel_count = int(root.GetAttribute("codex:parcelCount").Get() or 0)
parcels = []
for index in range(parcel_count):
    parcel = stage.GetPrimAtPath(f"{parcel_root}/Parcel{index + 1}")
    collider = stage.GetPrimAtPath(f"{parcel.GetPath()}/Collision")
    translate = parcel.GetAttribute("xformOp:translate").Get()
    velocity = parcel.GetAttribute("physics:velocity").Get()
    dimensions = parcel.GetAttribute("codex:dimensionsM").Get()
    force_api = PhysxSchema.PhysxForceAPI(parcel)
    force = (
        force_api.GetForceAttr().Get()
        if parcel.HasAPI(PhysxSchema.PhysxForceAPI)
        else None
    )
    force_enabled = (
        force_api.GetForceEnabledAttr().Get()
        if parcel.HasAPI(PhysxSchema.PhysxForceAPI)
        else None
    )
    parcels.append(
        {
            "path": str(parcel.GetPath()),
            "position": list(translate) if translate is not None else None,
            "velocity": list(velocity) if velocity is not None else None,
            "dimensions_m": (
                list(dimensions) if dimensions is not None else None
            ),
            "force_n": list(force) if force is not None else None,
            "force_enabled": force_enabled,
            "rigid_body": parcel.HasAPI(UsdPhysics.RigidBodyAPI),
            "collider": collider.HasAPI(UsdPhysics.CollisionAPI),
        }
    )
surface = stage.GetPrimAtPath(surface_root)
surface_colliders = [
    str(prim.GetPath())
    for prim in Usd.PrimRange(surface)
    if prim.HasAPI(UsdPhysics.CollisionAPI)
]
surface_material_targets = []
for prim in Usd.PrimRange(surface):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    relationship = UsdShade.MaterialBindingAPI(
        prim
    ).GetDirectBindingRel("physics")
    surface_material_targets.extend(
        str(target) for target in relationship.GetTargets()
    )
floor = stage.GetPrimAtPath(f"{root_path}/Floor")
print(
    json.dumps(
        {
            "timeline_playing": (
                omni.timeline.get_timeline_interface().is_playing()
            ),
            "layout": root.GetAttribute("codex:conveyorLayout").Get(),
            "physics_mode": (
                root.GetAttribute("codex:conveyorPhysicsMode").Get()
            ),
            "parcel_count": parcel_count,
            "parcels": parcels,
            "surface_collider_count": len(surface_colliders),
            "surface_physics_material_targets": sorted(
                set(surface_material_targets)
            ),
            "floor_collider": floor.HasAPI(UsdPhysics.CollisionAPI),
        }
    )
)
""".strip()


def main() -> None:
    response = IsaacTcpClient(timeout_seconds=60.0).execute(SOURCE)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
