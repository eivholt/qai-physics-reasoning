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
import math
import sys
import time

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdSkel

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/CodexPoC/ConveyorSafety")
if not root.IsValid():
    raise RuntimeError("Conveyor safety scene is not open")
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

worker_paths = [
    "/World/CodexPoC/ConveyorSafety/Workers/Worker1",
    "/World/CodexPoC/ConveyorSafety/Workers/Worker2",
]
samples = {path: [] for path in worker_paths}
animation_targets = {path: [] for path in worker_paths}
timeline.play()
wall_start = time.perf_counter()
for _ in range(480):
    await omni.kit.app.get_app().next_update_async()
    sim_time = float(timeline.get_current_time())
    wall_time = time.perf_counter() - wall_start
    for path in worker_paths:
        worker = stage.GetPrimAtPath(path)
        translate = worker.GetAttribute("xformOp:translate").Get()
        rotate = worker.GetAttribute("xformOp:rotateXYZ").Get()
        samples[path].append(
            {
                "time": sim_time,
                "wall_time": wall_time,
                "x": float(translate[0]),
                "y": float(translate[1]),
                "yaw": float(rotate[2]),
                "blocked": bool(
                    worker.GetAttribute("codex:navigationBlocked").Get()
                ),
                "blocked_by": str(
                    worker.GetAttribute("codex:navigationBlockedBy").Get()
                    or ""
                ),
            }
        )
        skel_root_path = str(
            worker.GetAttribute("codex:skelRootPath").Get() or ""
        )
        target = ""
        skel_root = stage.GetPrimAtPath(skel_root_path)
        if skel_root.IsValid():
            targets = skel_root.GetRelationship(
                "skel:animationSource"
            ).GetTargets()
            target = str(targets[0]) if targets else ""
        animation_targets[path].append(target)
timeline.stop()
for _ in range(4):
    await omni.kit.app.get_app().next_update_async()
extension._stop_owned_processes()
extension._auto_inference_enabled = original_auto_inference

walk_root_motion = {}
for animation_name in ("WalkForwardLoop", "WalkForwardLoop_01"):
    animation = UsdSkel.Animation(
        stage.GetPrimAtPath(
            "/World/CodexPoC/ConveyorSafety/Workers/"
            f"RetargetedAnimations/{animation_name}"
        )
    )
    joints = [str(item) for item in animation.GetJointsAttr().Get() or []]
    hip_index = joints.index("RL_BoneRoot/Hip")
    maximum_horizontal = 0.0
    for time_code in animation.GetTranslationsAttr().GetTimeSamples():
        translations = animation.GetTranslationsAttr().Get(
            Usd.TimeCode(time_code)
        )
        hip = translations[hip_index]
        maximum_horizontal = max(
            maximum_horizontal,
            math.hypot(float(hip[0]), float(hip[1])),
        )
    walk_root_motion[animation_name] = round(maximum_horizontal, 6)

# Deterministic planner probe: a box blocks the direct south corridor, so A*
# must select the adjacent row and return to the goal without crossing it.
synthetic_obstacles = [
    ("probe blocker", -1.25, -4.45, 0.45, 0.45),
]
synthetic_floor = (-5.33, 6.50, -5.05, 3.60)
synthetic_path = extension._plan_worker_corridor(
    Gf.Vec3d(-3.10, -4.45, 0.0),
    Gf.Vec3d(1.80, -4.45, 0.0),
    0.38,
    synthetic_obstacles,
    synthetic_floor,
)
synthetic_path_clear = True
synthetic_previous = Gf.Vec3d(-3.10, -4.45, 0.0)
for synthetic_point in synthetic_path:
    synthetic_path_clear = synthetic_path_clear and not bool(
        extension._worker_segment_obstacle(
            synthetic_previous,
            synthetic_point,
            0.38,
            synthetic_obstacles,
            synthetic_floor,
        )
    )
    synthetic_previous = synthetic_point

# Freeze probe: inject an obstacle over Worker1 for one navigation update and
# verify that the committed route position, waypoint, and dwell do not move.
worker_one = stage.GetPrimAtPath(worker_paths[0])
worker_one_position = worker_one.GetAttribute("xformOp:translate").Get()
original_worker_obstacles = extension._worker_obstacles
before_freeze = {
    "base": [
        float(value)
        for value in extension._worker_route_positions[0]
    ],
    "waypoint": extension._worker_route_waypoint_indices[0],
    "dwell": extension._worker_route_dwell_remaining[0],
}


def probe_worker_obstacles(stage_arg, root_arg):
    probe_obstacles, probe_floor = original_worker_obstacles(
        stage_arg,
        root_arg,
    )
    probe_obstacles.append(
        (
            "freeze probe",
            float(worker_one_position[0]),
            float(worker_one_position[1]),
            0.75,
            0.75,
        )
    )
    return probe_obstacles, probe_floor


extension._worker_obstacles = probe_worker_obstacles
try:
    for _ in range(10):
        extension._animate_workers(stage, 0.0, 0.05)
finally:
    extension._worker_obstacles = original_worker_obstacles
after_freeze = {
    "base": [
        float(value)
        for value in extension._worker_route_positions[0]
    ],
    "waypoint": extension._worker_route_waypoint_indices[0],
    "dwell": extension._worker_route_dwell_remaining[0],
    "blocked": bool(
        worker_one.GetAttribute("codex:navigationBlocked").Get()
    ),
    "blocked_by": str(
        worker_one.GetAttribute("codex:navigationBlockedBy").Get() or ""
    ),
}


def wrapped_yaw_delta(first, second):
    return abs((second - first + 180.0) % 360.0 - 180.0)


metrics = {}
for path in worker_paths:
    rows = samples[path]
    moving_steps = []
    yaw_steps = []
    teleport_steps = []
    unique_positions = []
    last_unique = None
    for row in rows:
        position = (row["x"], row["y"])
        if last_unique is None or math.hypot(
            position[0] - last_unique[1][0],
            position[1] - last_unique[1][1],
        ) > 1.0e-7:
            if last_unique is not None:
                elapsed = row["wall_time"] - last_unique[0]
                distance = math.hypot(
                    position[0] - last_unique[1][0],
                    position[1] - last_unique[1][1],
                )
                speed = distance / max(elapsed, 1.0e-6)
                moving_steps.append(speed)
                teleport_steps.append(distance)
            unique_positions.append(position)
            last_unique = (row["wall_time"], position)
    for first, second in zip(rows, rows[1:]):
        yaw_steps.append(wrapped_yaw_delta(first["yaw"], second["yaw"]))
    targets = animation_targets[path]
    animation_switches = sum(
        first != second
        for first, second in zip(targets, targets[1:])
    )
    worker = stage.GetPrimAtPath(path)
    metrics[path.rsplit("/", 1)[-1]] = {
        "start": [round(rows[0]["x"], 4), round(rows[0]["y"], 4)],
        "end": [round(rows[-1]["x"], 4), round(rows[-1]["y"], 4)],
        "unique_positions": len(unique_positions),
        "maximum_speed_mps": round(max(moving_steps or [0.0]), 4),
        "maximum_position_step_m": round(max(teleport_steps or [0.0]), 4),
        "maximum_heading_step_degrees": round(max(yaw_steps or [0.0]), 4),
        "blocked_frames": sum(row["blocked"] for row in rows),
        "blocked_labels": sorted(
            {row["blocked_by"] for row in rows if row["blocked_by"]}
        ),
        "replan_count": int(
            worker.GetAttribute("codex:navigationReplanCount").Get() or 0
        ),
        "animation_source_switches": animation_switches,
    }

print(json.dumps({
    "navigation_mode": root.GetAttribute(
        "codex:workerNavigationMode"
    ).Get(),
    "walk_root_maximum_horizontal_translation_m": walk_root_motion,
    "simulated_seconds": round(
        samples[worker_paths[0]][-1]["time"]
        - samples[worker_paths[0]][0]["time"],
        3,
    ),
    "wall_seconds": round(
        samples[worker_paths[0]][-1]["wall_time"]
        - samples[worker_paths[0]][0]["wall_time"],
        3,
    ),
    "planner_probe": {
        "path": [
            [round(float(point[0]), 3), round(float(point[1]), 3)]
            for point in synthetic_path
        ],
        "path_clear": synthetic_path_clear,
        "used_alternate_row": any(
            float(point[1]) > -4.0 for point in synthetic_path
        ),
    },
    "freeze_probe": {
        "before": before_freeze,
        "after": after_freeze,
        "route_progress_frozen": (
            before_freeze["base"] == after_freeze["base"]
            and before_freeze["waypoint"] == after_freeze["waypoint"]
            and before_freeze["dwell"] == after_freeze["dwell"]
        ),
    },
    "workers": metrics,
    "timeline_stopped": timeline.is_stopped(),
}))
""".strip()


def main() -> None:
    result = IsaacTcpClient(timeout_seconds=180.0).execute(SOURCE)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
