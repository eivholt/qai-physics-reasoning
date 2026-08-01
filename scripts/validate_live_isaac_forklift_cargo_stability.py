from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


SOURCE = r"""
import gc
import json
import sys

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
module = sys.modules.get("qai.conveyor_safety.extension")
extension_class = getattr(module, "ConveyorSafetyExtension", None)
extension = next(
    (
        item
        for item in gc.get_objects()
        if extension_class is not None and isinstance(item, extension_class)
    ),
    None,
)
if extension is None:
    raise RuntimeError("Conveyor safety extension is not active")

timeline = omni.timeline.get_timeline_interface()
original_auto_inference = extension._auto_inference_enabled
extension._auto_inference_enabled = False
extension._stop_owned_processes()
timeline.stop()
for _ in range(4):
    await omni.kit.app.get_app().next_update_async()

timeline.play()
for _ in range(20):
    await omni.kit.app.get_app().next_update_async()

manual_drive = extension._drive_active_forklift
extension._drive_active_forklift = lambda *args, **kwargs: None
cargo_paths = {
    "pallet": (
        "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift1Pallet"
    ),
    "carton": (
        "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift1Carton"
    ),
}
base_heights = {"pallet": 0.146, "carton": 0.300}
metrics = {
    name: {
        "maximum_upward_velocity_mps": 0.0,
        "maximum_lift_phase_separation_m": -999.0,
        "maximum_world_height_m": -999.0,
    }
    for name in cargo_paths
}

try:
    for frame in range(360):
        throttle = 0.0
        steering = 0.0
        lift = 0.0
        if frame < 60:
            lift = 1.0
        elif frame < 90:
            lift = 0.0
        elif frame < 150:
            lift = -1.0
        elif frame < 210:
            throttle = 1.0
            steering = 0.6
        elif frame < 240:
            throttle = 0.0
        elif frame < 300:
            throttle = -1.0
            steering = -0.6
        manual_drive(stage, throttle, steering, lift, 1.0 / 30.0)
        await omni.kit.app.get_app().next_update_async()

        forklift = stage.GetPrimAtPath(
            "/World/CodexPoC/ConveyorSafety/Forklifts/Forklift1"
        )
        fork_height = float(
            forklift.GetAttribute("codex:forkHeightM").Get() or 0.0
        )
        for name, path in cargo_paths.items():
            cargo = stage.GetPrimAtPath(path)
            matrix = UsdGeom.Xformable(cargo).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world_z = float(matrix.ExtractTranslation()[2])
            velocity = UsdPhysics.RigidBodyAPI(cargo).GetVelocityAttr().Get()
            upward_velocity = float(velocity[2]) if velocity is not None else 0.0
            item = metrics[name]
            item["maximum_upward_velocity_mps"] = max(
                item["maximum_upward_velocity_mps"],
                upward_velocity,
            )
            item["maximum_world_height_m"] = max(
                item["maximum_world_height_m"],
                world_z,
            )
            if frame < 90:
                separation = world_z - (
                    base_heights[name] + fork_height
                )
                item["maximum_lift_phase_separation_m"] = max(
                    item["maximum_lift_phase_separation_m"],
                    separation,
                )
finally:
    extension._drive_active_forklift = manual_drive
    timeline.stop()
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    extension._stop_owned_processes()
    extension._auto_inference_enabled = original_auto_inference

for item in metrics.values():
    for key, value in tuple(item.items()):
        item[key] = round(float(value), 4)

print(json.dumps({
    "frames": 360,
    "drive_acceleration_mps2": 1.0,
    "service_brake_mps2": 1.5,
    "lift_speed_mps": 0.9,
    "lift_acceleration_mps2": 1.35,
    "cargo": metrics,
    "timeline_stopped": not timeline.is_playing(),
}))
"""


def main() -> None:
    response = IsaacTcpClient(timeout_seconds=120.0).execute(SOURCE)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
