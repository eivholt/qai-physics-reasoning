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
import statistics
import time

import carb
import omni.replicator.core as rep
import omni.timeline
import omni.usd

camera_path = "/World/CodexPoC/ConveyorSafety/Cameras/DetectorEndline"
stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath(camera_path).IsValid():
    raise RuntimeError("DetectorEndline camera is missing")
timeline = omni.timeline.get_timeline_interface()
was_playing = timeline.is_playing()
timeline.stop()


async def benchmark(include_bbox, *, wait_for_render=True):
    render_product = rep.create.render_product(camera_path, (512, 288))
    rgb = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb.attach(render_product)
    bbox = None
    if include_bbox:
        bbox = rep.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_tight",
            init_params={"semanticTypes": ["class"]},
        )
        bbox.attach(render_product)
    hydra_texture = getattr(render_product, "hydra_texture", None)
    controlled = bool(
        hydra_texture is not None
        and hasattr(hydra_texture, "set_updates_enabled")
    )
    if controlled:
        hydra_texture.set_updates_enabled(True)
    await rep.orchestrator.step_async(
        rt_subframes=1,
        delta_time=0.0,
        pause_timeline=False,
        wait_for_render=wait_for_render,
    )
    step_seconds = []
    next_update_seconds = []
    rgb_seconds = []
    bbox_seconds = []
    for _ in range(8):
        started = time.perf_counter()
        await rep.orchestrator.step_async(
            rt_subframes=1,
            delta_time=0.0,
            pause_timeline=False,
            wait_for_render=wait_for_render,
        )
        after_step = time.perf_counter()
        if not wait_for_render:
            await omni.kit.app.get_app().next_update_async()
        after_update = time.perf_counter()
        rgb.get_data()
        after_rgb = time.perf_counter()
        if bbox is not None:
            bbox.get_data()
        after_bbox = time.perf_counter()
        step_seconds.append(after_step - started)
        next_update_seconds.append(after_update - after_step)
        rgb_seconds.append(after_rgb - after_update)
        bbox_seconds.append(after_bbox - after_rgb)
    if controlled:
        hydra_texture.set_updates_enabled(False)
    rgb.detach()
    if bbox is not None:
        bbox.detach()
    render_product.destroy()
    return {
        "include_bbox": include_bbox,
        "wait_for_render": wait_for_render,
        "median_step_ms": round(1000.0 * statistics.median(step_seconds), 4),
        "median_next_update_ms": round(
            1000.0 * statistics.median(next_update_seconds), 4
        ),
        "median_rgb_read_ms": round(1000.0 * statistics.median(rgb_seconds), 4),
        "median_bbox_read_ms": round(1000.0 * statistics.median(bbox_seconds), 4),
        "maximum_step_ms": round(1000.0 * max(step_seconds), 4),
    }


results = [
    await benchmark(False),
    await benchmark(True),
    await benchmark(False, wait_for_render=False),
    await benchmark(True, wait_for_render=False),
]
settings = carb.settings.get_settings()
fractional_cutout_path = "/rtx/pathtracing/fractionalCutoutOpacity"
fractional_cutout_original = settings.get(fractional_cutout_path)
settings.set_bool(fractional_cutout_path, False)
results.append(await benchmark(True))
results[-1]["fractional_cutout_opacity"] = False
if fractional_cutout_original is not None:
    settings.set_bool(
        fractional_cutout_path,
        bool(fractional_cutout_original),
    )
if was_playing:
    timeline.play()
print(json.dumps({"results": results}))
""".strip()


def main() -> None:
    print(
        json.dumps(
            IsaacTcpClient(timeout_seconds=120.0).execute(SOURCE),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
