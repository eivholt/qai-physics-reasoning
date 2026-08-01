from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


SOURCE = r"""
import json
import math
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
root_path = "/World/CodexPoC/ConveyorSafety"
root = stage.GetPrimAtPath(root_path)
rack_root = stage.GetPrimAtPath(f"{root_path}/Storage/WestRack")
collider_root = stage.GetPrimAtPath(
    f"{root_path}/Physics/StaticColliders/WestRack"
)

planes = []
for index in range(1, 4):
    prim = stage.GetPrimAtPath(f"{collider_root.GetPath()}/ShelfPlane{index}")
    translate = prim.GetAttribute("xformOp:translate").Get()
    scale = prim.GetAttribute("xformOp:scale").Get()
    planes.append(
        {
            "path": str(prim.GetPath()),
            "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
            "top_z_m": float(translate[2] + scale[2]),
            "half_extents_m": list(scale),
        }
    )

rear_guard = stage.GetPrimAtPath(f"{collider_root.GetPath()}/RearGuard")
plane_tops = [plane["top_z_m"] for plane in planes]
cartons = []
for prim in Usd.PrimRange(rack_root):
    marker = prim.GetAttribute("codex:shelfParcel")
    if not marker or not marker.Get():
        continue
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    position = transform.ExtractTranslation()
    dimensions = prim.GetAttribute("codex:dimensionsM").Get()
    velocity = prim.GetAttribute("physics:velocity").Get()
    if velocity is None:
        velocity = (0.0, 0.0, 0.0)
    bottom_z = float(position[2])
    nearest_plane_error = min(
        abs(bottom_z - plane_top) for plane_top in plane_tops
    )
    collider = stage.GetPrimAtPath(f"{prim.GetPath()}/Collision")
    mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
    cartons.append(
        {
            "path": str(prim.GetPath()),
            "position_m": [float(value) for value in position],
            "dimensions_m": [float(value) for value in dimensions],
            "mass_kg": float(mass),
            "speed_mps": math.sqrt(sum(float(value) ** 2 for value in velocity)),
            "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "collider": collider.HasAPI(UsdPhysics.CollisionAPI),
            "bottom_to_nearest_shelf_m": nearest_plane_error,
        }
    )

print(
    json.dumps(
        {
            "timeline_playing": (
                omni.timeline.get_timeline_interface().is_playing()
            ),
            "authored_shelf_carton_count": int(
                root.GetAttribute("codex:shelfCartonCount").Get() or 0
            ),
            "authored_shelf_plane_count": int(
                root.GetAttribute("codex:shelfCollisionPlaneCount").Get() or 0
            ),
            "dynamic_shelf_cartons_enabled": bool(
                root.GetAttribute("codex:dynamicShelfCartonsEnabled").Get()
            ),
            "observed_shelf_carton_count": len(cartons),
            "rigid_body_count": sum(item["rigid_body"] for item in cartons),
            "carton_collider_count": sum(item["collider"] for item in cartons),
            "shelf_planes": planes,
            "rear_guard_collision": rear_guard.HasAPI(
                UsdPhysics.CollisionAPI
            ),
            "cartons_within_3cm_of_shelf": sum(
                item["bottom_to_nearest_shelf_m"] <= 0.03
                for item in cartons
            ),
            "cartons_below_floor": sum(
                item["position_m"][2] < -0.05 for item in cartons
            ),
            "maximum_carton_speed_mps": max(
                (item["speed_mps"] for item in cartons),
                default=0.0,
            ),
            "cartons": cartons,
        }
    )
)
""".strip()


def main() -> None:
    response = IsaacTcpClient(timeout_seconds=60.0).execute(SOURCE)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
