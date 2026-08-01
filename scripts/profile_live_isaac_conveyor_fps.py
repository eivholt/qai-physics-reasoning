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
import statistics
import sys
import time

import carb
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSkel

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is open")

module = sys.modules.get("qai.conveyor_safety.extension")
if module is None:
    module = next(
        (
            value
            for name, value in sys.modules.items()
            if name.endswith("qai.conveyor_safety.extension")
        ),
        None,
    )
extension_class = getattr(module, "ConveyorSafetyExtension", None)
extension = next(
    (
        value
        for value in gc.get_objects()
        if extension_class is not None and isinstance(value, extension_class)
    ),
    None,
)
if extension is None:
    raise RuntimeError("The conveyor safety extension instance is not active")

settings = carb.settings.get_settings()
setting_keys = (
    "/app/runLoops/main/rateLimitEnabled",
    "/app/runLoops/main/rateLimitFrequency",
    "/app/runLoops/present/rateLimitEnabled",
    "/app/runLoops/present/rateLimitFrequency",
    "/app/runLoops/rendering_0/rateLimitEnabled",
    "/app/runLoops/rendering_0/rateLimitFrequency",
    "/app/player/targetRunLoopFrequency",
    "/rtx/post/dlss/execMode",
    "/rtx/pathtracing/fractionalCutoutOpacity",
    "/rtx-transient/resourcemanager/texturestreaming/enabled",
    "/persistent/rtx/modes/rt2/enabled",
    "/rtx/rendermode",
)

subtree_paths = (
    "/World/CodexPoC/ConveyorSafety/Conveyor",
    "/World/CodexPoC/ConveyorSafety/Forklifts",
    "/World/CodexPoC/ConveyorSafety/Workers",
    "/World/CodexPoC/ConveyorSafety/Shelving",
    "/World/CodexPoC/ConveyorSafety/StackLight",
    "/World/CodexPoC/ConveyorSafety/Lighting",
)
subtree_counts = {}
for path in subtree_paths:
    prim = stage.GetPrimAtPath(path)
    subtree_counts[path.rsplit("/", 1)[-1]] = (
        sum(1 for _ in Usd.PrimRange(prim)) if prim.IsValid() else 0
    )

stage_counts = {
    "active_prims": 0,
    "instanceable_prims": 0,
    "instance_proxies": 0,
    "renderable_prims": 0,
    "lights": 0,
    "rigid_bodies": 0,
    "skeletons": 0,
}
renderable_types = (
    UsdGeom.Mesh,
    UsdGeom.Cube,
    UsdGeom.Sphere,
    UsdGeom.Cylinder,
    UsdGeom.Capsule,
)
for prim in stage.Traverse():
    stage_counts["active_prims"] += 1
    stage_counts["instanceable_prims"] += int(prim.IsInstanceable())
    stage_counts["instance_proxies"] += int(prim.IsInstanceProxy())
    stage_counts["renderable_prims"] += int(
        any(prim.IsA(schema_type) for schema_type in renderable_types)
    )
    stage_counts["lights"] += int(prim.HasAPI(UsdLux.LightAPI))
    stage_counts["rigid_bodies"] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
    stage_counts["skeletons"] += int(prim.IsA(UsdSkel.Skeleton))

method_names = (
    "_on_update",
    "_read_inference_trace",
    "_update_presentation_camera",
    "_ensure_inference_running",
    "_drive_active_forklift",
    "_animate_parcels",
    "_animate_workers",
    "_worker_obstacles",
    "_update_clearance",
    "_update_ui",
    "_forklift_pose",
    "_conveyor_obstacle_aabbs",
)
metrics = {}
originals = {}


def install_wrappers():
    for name in method_names:
        original = getattr(extension_class, name, None)
        if original is None:
            continue
        originals[name] = original

        def wrapped(*args, __name=name, __original=original, **kwargs):
            started = time.perf_counter()
            try:
                return __original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                item = metrics.setdefault(
                    __name,
                    {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
                )
                item["calls"] += 1
                item["total_seconds"] += elapsed
                item["max_seconds"] = max(item["max_seconds"], elapsed)

        setattr(extension_class, name, wrapped)


def restore_wrappers():
    for name, original in originals.items():
        setattr(extension_class, name, original)


def reset_metrics():
    metrics.clear()


async def measure(label, frame_count):
    for _ in range(12):
        await omni.kit.app.get_app().next_update_async()
    reset_metrics()
    intervals = []
    started = time.perf_counter()
    previous = started
    for _ in range(frame_count):
        await omni.kit.app.get_app().next_update_async()
        current = time.perf_counter()
        intervals.append(current - previous)
        previous = current
    elapsed = time.perf_counter() - started
    callback_metrics = {}
    for name, item in metrics.items():
        calls = int(item["calls"])
        callback_metrics[name] = {
            "calls": calls,
            "average_ms": round(
                1000.0 * item["total_seconds"] / max(calls, 1),
                4,
            ),
            "maximum_ms": round(1000.0 * item["max_seconds"], 4),
            "total_ms": round(1000.0 * item["total_seconds"], 4),
        }
    sorted_intervals = sorted(intervals)
    percentile_index = min(
        len(sorted_intervals) - 1,
        max(0, math.ceil(0.95 * len(sorted_intervals)) - 1),
    )
    return {
        "label": label,
        "frames": frame_count,
        "elapsed_seconds": round(elapsed, 6),
        "effective_fps": round(frame_count / elapsed, 3),
        "median_frame_ms": round(1000.0 * statistics.median(intervals), 4),
        "p95_frame_ms": round(1000.0 * sorted_intervals[percentile_index], 4),
        "maximum_frame_ms": round(1000.0 * max(intervals), 4),
        "callbacks": callback_metrics,
    }


async def measure_with_noops(label, method_names_to_disable, frame_count=45):
    saved = {
        name: getattr(extension_class, name)
        for name in method_names_to_disable
    }
    try:
        for name in method_names_to_disable:
            setattr(extension_class, name, lambda *args, **kwargs: None)
        return await measure(label, frame_count)
    finally:
        for name, method in saved.items():
            setattr(extension_class, name, method)


timeline = omni.timeline.get_timeline_interface()
original_auto_inference = extension._auto_inference_enabled
physics_scene = stage.GetPrimAtPath(
    "/World/CodexPoC/ConveyorSafety/Physics/Scene"
)
physics_scene_api = PhysxSchema.PhysxSceneAPI(physics_scene)
physics_rate_attr = physics_scene_api.GetTimeStepsPerSecondAttr()
original_physics_rate = int(physics_rate_attr.Get() or 60)
install_wrappers()
try:
    extension._auto_inference_enabled = False
    extension._stop_owned_processes()
    timeline.stop()
    stopped = await measure("stopped", 60)

    timeline.play()
    playing = await measure("playing_without_inference_capture", 60)
    no_parcel_updates = await measure_with_noops(
        "playing_without_parcel_force_updates",
        ("_animate_parcels",),
    )
    no_worker_updates = await measure_with_noops(
        "playing_without_worker_route_updates",
        ("_animate_workers",),
    )
    no_drive_updates = await measure_with_noops(
        "playing_without_articulation_target_updates",
        ("_drive_active_forklift",),
    )
    no_demo_updates = await measure_with_noops(
        "playing_without_demo_dynamic_updates",
        (
            "_animate_parcels",
            "_animate_workers",
            "_drive_active_forklift",
        ),
        frame_count=60,
    )
    timeline.stop()
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    physics_rate_attr.Set(30)
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    timeline.play()
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    playing_30hz_physics = await measure(
        "playing_with_30hz_physics",
        60,
    )
finally:
    timeline.stop()
    physics_rate_attr.Set(original_physics_rate)
    extension._stop_owned_processes()
    extension._auto_inference_enabled = original_auto_inference
    restore_wrappers()

print(json.dumps({
    "settings": {key: settings.get(key) for key in setting_keys},
    "stage_counts": stage_counts,
    "subtree_prim_counts": subtree_counts,
    "measurements": [
        stopped,
        playing,
        no_parcel_updates,
        no_worker_updates,
        no_drive_updates,
        no_demo_updates,
        playing_30hz_physics,
    ],
    "physics_rate_restored": int(physics_rate_attr.Get() or 0),
    "timeline_left_stopped": timeline.is_stopped(),
    "auto_inference_restored": extension._auto_inference_enabled,
}))
""".strip()


def main() -> None:
    response = IsaacTcpClient(timeout_seconds=120.0).execute(SOURCE)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
