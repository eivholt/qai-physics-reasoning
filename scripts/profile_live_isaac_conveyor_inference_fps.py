from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


def start_source(duration_seconds: float) -> str:
    return f"""
import asyncio
import math
import statistics
import sys
import time
import types

import omni.kit.app
import omni.timeline

module_name = "_qai_conveyor_background_fps_probe"
probe = sys.modules.get(module_name)
if probe is None:
    probe = types.ModuleType(module_name)
    sys.modules[module_name] = probe
existing_task = getattr(probe, "task", None)
if existing_task is not None and not existing_task.done():
    existing_task.cancel()
probe.result = None


async def sample():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(20):
        await omni.kit.app.get_app().next_update_async()
    intervals = []
    started = time.perf_counter()
    previous = started
    while time.perf_counter() - started < {duration_seconds!r}:
        await omni.kit.app.get_app().next_update_async()
        current = time.perf_counter()
        intervals.append(current - previous)
        previous = current
    elapsed = time.perf_counter() - started
    ordered = sorted(intervals)

    def percentile(fraction):
        index = min(
            len(ordered) - 1,
            max(0, math.ceil(fraction * len(ordered)) - 1),
        )
        return ordered[index]

    probe.result = {{
        "frames": len(intervals),
        "elapsed_seconds": round(elapsed, 6),
        "effective_fps": round(len(intervals) / elapsed, 3),
        "median_frame_ms": round(1000.0 * statistics.median(intervals), 4),
        "p95_frame_ms": round(1000.0 * percentile(0.95), 4),
        "p99_frame_ms": round(1000.0 * percentile(0.99), 4),
        "maximum_frame_ms": round(1000.0 * max(intervals), 4),
        "frames_over_50ms": sum(value > 0.050 for value in intervals),
        "frames_over_100ms": sum(value > 0.100 for value in intervals),
        "frames_over_250ms": sum(value > 0.250 for value in intervals),
    }}
    timeline.stop()


probe.task = asyncio.ensure_future(sample())
print("started")
""".strip()


RESULT_SOURCE = r"""
import gc
import json
import sys

probe = sys.modules.get("_qai_conveyor_background_fps_probe")
if probe is None:
    raise RuntimeError("The background FPS probe was not started")
task = getattr(probe, "task", None)
if task is not None and not task.done():
    await task

module = sys.modules.get("qai.conveyor_safety.extension")
extension_class = getattr(module, "ConveyorSafetyExtension", None)
extension = next(
    (
        value
        for value in gc.get_objects()
        if extension_class is not None and isinstance(value, extension_class)
    ),
    None,
)
if extension is not None:
    extension._stop_owned_processes()
print(json.dumps(probe.result))
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    args = parser.parse_args()
    if not 4.0 <= args.duration_seconds <= 60.0:
        raise ValueError("duration must be between 4 and 60 seconds")
    client = IsaacTcpClient(timeout_seconds=120.0)
    start = client.execute(start_source(args.duration_seconds))
    if start.get("status") != "ok":
        print(json.dumps(start, indent=2))
        raise SystemExit(1)
    time.sleep(args.duration_seconds + 3.0)
    print(json.dumps(client.execute(RESULT_SOURCE), indent=2))


if __name__ == "__main__":
    main()
