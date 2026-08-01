from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


PUSH_SOURCE = r"""
import json
import math
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

stage = omni.usd.get_context().get_stage()
root_path = "/World/CodexPoC/ConveyorSafety"
root = stage.GetPrimAtPath(root_path)
parcel_count = int(root.GetAttribute("codex:parcelCount").Get())
parcels = [
    stage.GetPrimAtPath(
        f"{root_path}/Conveyor/Parcels/Parcel{index + 1}"
    )
    for index in range(parcel_count)
]
west_candidates = [
    candidate
    for candidate in parcels
    if float(candidate.GetAttribute("xformOp:translate").Get()[0]) < 3.50
]
# Prefer the northwest part of the loop so an outward push has ample floor
# clearance and does not intersect the parked forklift near the southwest.
parcel = max(
    west_candidates or parcels,
    key=lambda candidate: float(
        candidate.GetAttribute("xformOp:translate").Get()[1]
    ),
)
push_path_attr = root.GetAttribute("codex:lastParcelPushTestPath")
if not push_path_attr.IsValid():
    push_path_attr = root.CreateAttribute(
        "codex:lastParcelPushTestPath",
        Sdf.ValueTypeNames.String,
    )
push_path_attr.Set(str(parcel.GetPath()))
position = parcel.GetAttribute("xformOp:translate").Get()
x = float(position[0])
y = float(position[1])
center_x = float(root.GetAttribute("codex:beltCenterXM").Get())
center_y = float(root.GetAttribute("codex:beltCenterYM").Get())
radius = float(root.GetAttribute("codex:conveyorOvalRadiusM").Get())
straight_half = float(
    root.GetAttribute("codex:conveyorOvalStraightHalfLengthM").Get()
)
north_y = center_y + straight_half
south_y = center_y - straight_half
if south_y <= y <= north_y:
    lane_sign = -1.0 if x < center_x else 1.0
    closest_x = center_x + lane_sign * radius
    closest_y = y
    outward_x = lane_sign
    outward_y = 0.0
elif y > north_y:
    delta_x = x - center_x
    delta_y = y - north_y
    delta_length = max(1.0e-6, math.hypot(delta_x, delta_y))
    closest_x = center_x + radius * delta_x / delta_length
    closest_y = north_y + radius * delta_y / delta_length
    outward_x = delta_x / delta_length
    outward_y = delta_y / delta_length
else:
    delta_x = x - center_x
    delta_y = y - south_y
    delta_length = max(1.0e-6, math.hypot(delta_x, delta_y))
    closest_x = center_x + radius * delta_x / delta_length
    closest_y = south_y + radius * delta_y / delta_length
    outward_x = delta_x / delta_length
    outward_y = delta_y / delta_length
velocity_attr = UsdPhysics.RigidBodyAPI(parcel).GetVelocityAttr()
old_velocity = velocity_attr.Get() or Gf.Vec3f(0.0)
push_force_n = 140.0
force_prim = stage.DefinePrim(
    f"{parcel.GetPath()}/ForkPushForce",
    "Xform",
)
force_api = PhysxSchema.PhysxForceAPI.Apply(force_prim)
force_api.CreateForceAttr().Set(
    Gf.Vec3f(
        outward_x * push_force_n,
        outward_y * push_force_n,
        0.0,
    )
)
force_api.CreateTorqueAttr().Set(Gf.Vec3f(0.0))
force_api.CreateModeAttr().Set("force")
force_api.CreateForceEnabledAttr().Set(True)
force_api.CreateWorldFrameEnabledAttr().Set(True)
print(
    json.dumps(
        {
            "parcel": str(parcel.GetPath()),
            "start_position": list(position),
            "closest_centerline": [closest_x, closest_y],
            "outward_direction": [outward_x, outward_y],
            "push_force_n": push_force_n,
        }
    )
)
""".strip()

RELEASE_SOURCE = r"""
import omni.usd
from pxr import Gf, PhysxSchema

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/CodexPoC/ConveyorSafety")
parcel_path = str(
    root.GetAttribute("codex:lastParcelPushTestPath").Get()
)
force_prim = stage.GetPrimAtPath(
    f"{parcel_path}/ForkPushForce"
)
force_api = PhysxSchema.PhysxForceAPI(force_prim)
force_api.GetForceEnabledAttr().Set(False)
force_api.GetForceAttr().Set(Gf.Vec3f(0.0))
print("released")
""".strip()


INSPECT_SOURCE = r"""
import json
import math
import omni.usd
from pxr import PhysxSchema, UsdPhysics

stage = omni.usd.get_context().get_stage()
root_path = "/World/CodexPoC/ConveyorSafety"
root = stage.GetPrimAtPath(root_path)
parcel_path = str(
    root.GetAttribute("codex:lastParcelPushTestPath").Get()
)
parcel = stage.GetPrimAtPath(parcel_path)
position = parcel.GetAttribute("xformOp:translate").Get()
velocity = UsdPhysics.RigidBodyAPI(parcel).GetVelocityAttr().Get()
dimensions = parcel.GetAttribute("codex:dimensionsM").Get()
force_api = PhysxSchema.PhysxForceAPI(parcel)
force_enabled = force_api.GetForceEnabledAttr().Get()
expected_floor_z = 0.5 * float(dimensions[2])
print(
    json.dumps(
        {
            "position": list(position),
            "velocity_mps": list(velocity),
            "force_enabled": bool(force_enabled),
            "expected_floor_center_z_m": expected_floor_z,
            "landed_on_floor": (
                abs(float(position[2]) - expected_floor_z) < 0.04
                and abs(float(velocity[2])) < 0.08
            ),
        }
    )
)
""".strip()


def main() -> None:
    client = IsaacTcpClient(timeout_seconds=60.0)
    push = client.execute(PUSH_SOURCE)
    time.sleep(0.5)
    client.execute(RELEASE_SOURCE)
    time.sleep(5.5)
    result = client.execute(INSPECT_SOURCE)
    print(
        json.dumps(
            {
                "push": json.loads(push["output"]),
                "result": json.loads(result["output"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
