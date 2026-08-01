from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.conveyor_safety import (
    set_conveyor_safety_light_source,
)
from integrations.isaac_sim_mcp.server import IsaacTcpClient


PHYSICS_SOURCE = r"""
import json
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
parcel_root = stage.GetPrimAtPath(
    "/World/CodexPoC/ConveyorSafety/Conveyor/Parcels"
)
worker_root = stage.GetPrimAtPath(
    "/World/CodexPoC/ConveyorSafety/Workers"
)


def positions():
    result = {}
    for parcel in parcel_root.GetChildren():
        matrix = UsdGeom.Xformable(parcel).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        value = matrix.ExtractTranslation()
        result[parcel.GetName()] = [round(float(item), 4) for item in value]
    return result


def worker_positions():
    result = {}
    for worker in worker_root.GetChildren():
        if not worker.GetName().startswith("Worker"):
            continue
        matrix = UsdGeom.Xformable(worker).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        value = matrix.ExtractTranslation()
        result[worker.GetName()] = [
            round(float(item), 4) for item in value
        ]
    return result


start = positions()
worker_start = worker_positions()
start_time = float(timeline.get_current_time())
timeline.play()
for _ in range(30):
    await omni.kit.app.get_app().next_update_async()
end_time = float(timeline.get_current_time())
end = positions()
worker_end = worker_positions()
timeline.stop()
for _ in range(4):
    await omni.kit.app.get_app().next_update_async()
moved = {
    name: round(
        sum((end[name][axis] - value[axis]) ** 2 for axis in range(3)) ** 0.5,
        4,
    )
    for name, value in start.items()
}
workers_moved = {
    name: round(
        sum(
            (worker_end[name][axis] - value[axis]) ** 2
            for axis in range(3)
        ) ** 0.5,
        4,
    )
    for name, value in worker_start.items()
}
table_collider = stage.GetPrimAtPath(
    "/World/CodexPoC/ConveyorSafety/Physics/StaticColliders/PackingTable"
)
table_collision_value = (
    UsdPhysics.CollisionAPI(table_collider).GetCollisionEnabledAttr().Get()
    if table_collider.IsValid()
    and table_collider.HasAPI(UsdPhysics.CollisionAPI)
    else False
)
print(json.dumps({
    "timeline_playing": timeline.is_playing(),
    "simulated_seconds": round(end_time - start_time, 3),
    "parcel_start": start,
    "parcel_end": end,
    "parcel_displacement_m": moved,
    "all_parcels_moved": all(value > 0.05 for value in moved.values()),
    "all_parcels_above_floor": all(value[2] > 0.1 for value in end.values()),
    "worker_start": worker_start,
    "worker_end": worker_end,
    "worker_displacement_m": workers_moved,
    "all_workers_moved": all(value > 0.05 for value in workers_moved.values()),
    "table_collider_valid": table_collider.IsValid(),
    "table_collider_enabled": bool(
        table_collider.IsValid()
        and table_collider.HasAPI(UsdPhysics.CollisionAPI)
        and table_collision_value is not False
    ),
}))
""".strip()


LIGHT_AUDIT_SOURCE = r"""
import json
import omni.usd
from pxr import UsdLux

stage = omni.usd.get_context().get_stage()
roots = (
    "/World/CodexPoC/ConveyorSafety/StackLight",
    "/World/CodexPoC/ConveyorSafety/StackLight_01",
    "/World/CodexPoC/ConveyorSafety/StackLight_02",
)
result = {}
for root in roots:
    result[root] = {}
    for name in ("Green", "Amber", "Red"):
        prim = stage.GetPrimAtPath(f"{root}/{name}Lens/Glow")
        light = UsdLux.SphereLight(prim)
        point_attr = light.GetTreatAsPointAttr()
        result[root][name] = {
            "intensity": float(light.GetIntensityAttr().Get()),
            "radius": float(light.GetRadiusAttr().Get()),
            "treat_as_point": bool(
                point_attr.IsValid() and point_attr.Get()
            ),
        }
    wash_prim = stage.GetPrimAtPath(f"{root}/SceneWash")
    wash = UsdLux.SphereLight(wash_prim)
    wash_point_attr = wash.GetTreatAsPointAttr()
    result[root]["SceneWash"] = {
        "intensity": float(wash.GetIntensityAttr().Get()),
        "radius": float(wash.GetRadiusAttr().Get()),
        "treat_as_point": bool(
            wash_point_attr.IsValid() and wash_point_attr.Get()
        ),
    }
print(json.dumps(result))
""".strip()


def main() -> None:
    client = IsaacTcpClient(timeout_seconds=300.0)
    physics = client.execute(PHYSICS_SOURCE)
    lights = {}
    for signal in ("RED", "AMBER", "GREEN"):
        applied = client.execute(
            set_conveyor_safety_light_source(
                {
                    "signal": signal,
                    "confidence": 1.0,
                    "inference_id": f"r31-{signal.lower()}-audit",
                }
            )
        )
        audited = client.execute(LIGHT_AUDIT_SOURCE)
        lights[signal] = {"applied": applied, "intensities": audited}
    print(json.dumps({"physics": physics, "lights": lights}, indent=2))


if __name__ == "__main__":
    main()
