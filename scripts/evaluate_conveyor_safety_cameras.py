#!/usr/bin/env python3
"""Benchmark camera coverage, actor tracking, policy facts, and Reason2 output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.isaac_sim_mcp.conveyor_safety import (
    CAMERA_PATHS,
    get_conveyor_safety_state_source,
    set_conveyor_safety_forklift_pose_source,
)
from integrations.isaac_sim_mcp.server import IsaacTcpClient
from scripts.run_live_isaac_conveyor_safety import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SERVER_URL,
    DIRECT_HAZARD_PROMPT,
    _read_png_rgb,
    check_model_server,
    capture_conveyor_frame,
    execute_isaac_source,
    extract_forklift_motion_observation,
    recover_forklift_boxes_from_rgb,
    request_hazard_signal,
    resize_png_nearest,
)


# AMBER is a temporal case: both poses are still physically outside the red
# zone, but their chronological displacement projects entry within 3 seconds.
SCENARIOS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "GREEN": ((-0.50, -2.50, 0.0),),
    # A short adjacent-frame displacement proved substantially easier for
    # Reason2-2B to associate than a long jump across the aisle.
    "AMBER": ((-0.90, -1.00, 0.0), (-0.30, -1.00, 0.0)),
    "RED": ((1.80, -1.00, 0.0),),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend-label",
        choices=("host", "evk"),
        default="host",
    )
    parser.add_argument(
        "--perception-mode",
        choices=("direct", "hybrid"),
        default="direct",
        help=(
            "direct sends chronological images and the policy; hybrid is a "
            "legacy single-image ablation. Neither mode sends detector facts."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--model-image-width", type=int)
    parser.add_argument("--model-image-height", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=tuple(name for name in CAMERA_PATHS if name != "presentation"),
        default=[
            "detector_endline",
            "detector_oblique",
            "detector_overhead",
            "detector_side",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT.parent / "camera_evaluation.json",
    )
    return parser


def observation_signal(observation: dict[str, Any]) -> str:
    if observation["present_red_zone_overlap"]:
        return "RED"
    if observation["predicted_red_zone_entry"]:
        return "AMBER"
    return "GREEN"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return round(ordered[index], 6)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if (args.model_image_width is None) != (args.model_image_height is None):
        raise ValueError(
            "--model-image-width and --model-image-height must be provided together"
        )
    if args.model_image_width is None:
        model_image_size = (384, 216) if args.backend_label == "evk" else None
    else:
        model_image_size = (args.model_image_width, args.model_image_height)
        if model_image_size[0] < 32 or model_image_size[1] < 32:
            raise ValueError("model image dimensions must be at least 32 pixels")

    check_model_server(args.server_url, timeout_seconds=args.timeout_seconds)
    client = IsaacTcpClient(timeout_seconds=max(45.0, args.timeout_seconds))
    initial = execute_isaac_source(client, get_conveyor_safety_state_source())
    original_poses = {
        item["index"]: (
            item["position"][0],
            item["position"][1],
            item["yaw_degrees"],
        )
        for item in initial["forklifts"]
    }
    records: list[dict[str, Any]] = []
    try:
        execute_isaac_source(
            client,
            "import omni.timeline\n"
            "omni.timeline.get_timeline_interface().pause()\n"
            "print('{}')",
        )
        for camera in args.cameras:
            for expected, poses in SCENARIOS.items():
                for repeat_index in range(args.repeats):
                    case_started = time.monotonic()
                    previous_observation: dict[str, Any] | None = None
                    source_paths: list[Path] = []
                    model_paths: list[Path] = []
                    observations: list[dict[str, Any]] = []
                    captures: list[dict[str, Any]] = []
                    capture_seconds = 0.0
                    preprocessing_seconds = 0.0
                    state: dict[str, Any] = {}
                    try:
                        for frame_index, (x, y, yaw) in enumerate(poses):
                            execute_isaac_source(
                                client,
                                set_conveyor_safety_forklift_pose_source(
                                    {
                                        "index": 1,
                                        "x": x,
                                        "y": y,
                                        "yaw_degrees": yaw,
                                    }
                                ),
                            )
                            filename = (
                                f"camera_eval_{args.perception_mode}_{camera}_"
                                f"{expected.lower()}_r{repeat_index + 1}_"
                                f"f{frame_index + 1}.png"
                            )
                            capture_started = time.monotonic()
                            capture = capture_conveyor_frame(
                                client,
                                camera=camera,
                                filename=filename,
                                width=(
                                    model_image_size[0]
                                    if model_image_size is not None
                                    else 768
                                ),
                                height=(
                                    model_image_size[1]
                                    if model_image_size is not None
                                    else 432
                                ),
                            )
                            capture_seconds += time.monotonic() - capture_started
                            captures.append(capture)
                            image_path = Path(capture["output_path"]).resolve()
                            source_paths.append(image_path)

                            preprocess_started = time.monotonic()
                            decoded = _read_png_rgb(image_path)
                            expected_indices = tuple(
                                int(index)
                                for index in (
                                    capture.get("enabled_forklift_indices")
                                    or (1,)
                                )
                            )
                            boxes = recover_forklift_boxes_from_rgb(
                                image_path,
                                capture.get("forklift_pixel_boxes"),
                                previous_observation=previous_observation,
                                decoded_image=decoded,
                                expected_forklift_indices=expected_indices,
                            )
                            observation = extract_forklift_motion_observation(
                                image_path,
                                boxes,
                                previous_observation=previous_observation,
                                elapsed_seconds=(
                                    1.0
                                    if previous_observation is not None
                                    else None
                                ),
                                prediction_horizon_seconds=3.0,
                                decoded_image=decoded,
                                expected_forklift_count=len(expected_indices),
                            )
                            observation.pop("_red_zone_mask", None)
                            previous_observation = observation
                            observations.append(observation)
                            model_path = image_path
                            if (
                                model_image_size is not None
                                and decoded[:2] != model_image_size
                            ):
                                model_path = resize_png_nearest(
                                    image_path,
                                    image_path.with_name(
                                        f"{image_path.stem}_model_"
                                        f"{model_image_size[0]}x"
                                        f"{model_image_size[1]}.png"
                                    ),
                                    width=model_image_size[0],
                                    height=model_image_size[1],
                                    decoded_image=decoded,
                                )
                            model_paths.append(model_path)
                            preprocessing_seconds += (
                                time.monotonic() - preprocess_started
                            )

                        state = execute_isaac_source(
                            client,
                            get_conveyor_safety_state_source(),
                        )
                        latest_observation = observations[-1]
                        frontend_signal = observation_signal(latest_observation)
                        request_paths = (
                            model_paths
                            if args.perception_mode == "direct"
                            else [model_paths[-1]]
                        )
                        result = request_hazard_signal(
                            image_path=model_paths[-1],
                            image_paths=request_paths,
                            prompt=DIRECT_HAZARD_PROMPT,
                            server_url=args.server_url,
                            model=args.model,
                            timeout_seconds=args.timeout_seconds,
                            vision_observation=latest_observation,
                            enable_think=False,
                            use_signal_grammar=True,
                        )
                        model_signal = result["signal"]
                        semantic_complete = all(
                            len(capture.get("forklift_pixel_boxes") or [])
                            == len(
                                capture.get("enabled_forklift_indices")
                                or (1,)
                            )
                            for capture in captures
                        )
                        record = {
                            "camera": camera,
                            "expected": expected,
                            "repeat": repeat_index + 1,
                            "perception_mode": args.perception_mode,
                            "model_signal": model_signal,
                            "frontend_signal": frontend_signal,
                            "model_match": model_signal == expected,
                            "frontend_match": frontend_signal == expected,
                            "model_matches_frontend": (
                                model_signal == frontend_signal
                            ),
                            "latest_static_simulator_signal": state[
                                "ground_truth_signal"
                            ],
                            "latest_clearance_m": state[
                                "min_forklift_clearance_m"
                            ],
                            "semantic_actor_complete": semantic_complete,
                            "visible_forklift_count": latest_observation[
                                "visible_forklift_count"
                            ],
                            "vision_observation": latest_observation,
                            "image_paths": [str(path) for path in source_paths],
                            "model_image_paths": result["model_image_paths"],
                            "latency_seconds": result["latency_seconds"],
                            "stage_seconds": {
                                "capture": round(capture_seconds, 6),
                                "preprocessing": round(
                                    preprocessing_seconds, 6
                                ),
                                "model_roundtrip": result[
                                    "server_roundtrip_seconds"
                                ],
                                "case_total": round(
                                    time.monotonic() - case_started, 6
                                ),
                            },
                        }
                    except Exception as exc:
                        model_signal = "ERROR"
                        record = {
                            "camera": camera,
                            "expected": expected,
                            "repeat": repeat_index + 1,
                            "perception_mode": args.perception_mode,
                            "model_signal": model_signal,
                            "model_match": False,
                            "frontend_match": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "image_paths": [
                                capture.get("output_path")
                                for capture in captures
                            ],
                            "stage_seconds": {
                                "capture": round(capture_seconds, 6),
                                "preprocessing": round(
                                    preprocessing_seconds, 6
                                ),
                                "case_total": round(
                                    time.monotonic() - case_started, 6
                                ),
                            },
                        }
                    records.append(record)
                    print(
                        f"{camera:18s} {expected:5s} r{repeat_index + 1} "
                        f"frontend={record.get('frontend_signal', 'ERROR'):5s} "
                        f"model={model_signal:5s}",
                        flush=True,
                    )
    finally:
        for index, pose in original_poses.items():
            execute_isaac_source(
                client,
                set_conveyor_safety_forklift_pose_source(
                    {
                        "index": index,
                        "x": pose[0],
                        "y": pose[1],
                        "yaw_degrees": pose[2],
                    }
                ),
            )
        execute_isaac_source(
            client,
            "import omni.timeline\n"
            "omni.timeline.get_timeline_interface().play()\n"
            "print('{}')",
        )

    scores = {
        camera: {
            "model_matches": sum(
                1
                for record in records
                if record["camera"] == camera and record["model_match"]
            ),
            "frontend_matches": sum(
                1
                for record in records
                if record["camera"] == camera and record["frontend_match"]
            ),
            "cases": sum(1 for record in records if record["camera"] == camera),
        }
        for camera in args.cameras
    }
    best_camera = max(
        args.cameras,
        key=lambda camera: (
            scores[camera]["model_matches"],
            scores[camera]["frontend_matches"],
            camera,
        ),
    )
    successful = [record for record in records if "error" not in record]
    semantic_complete = sum(
        1 for record in successful if record["semantic_actor_complete"]
    )
    model_latencies = [
        float(record["stage_seconds"]["model_roundtrip"])
        for record in successful
    ]
    cycle_latencies = [
        float(record["stage_seconds"]["case_total"])
        for record in successful
    ]
    report = {
        "schema_version": "2.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "backend": args.backend_label,
        "server_url": args.server_url,
        "perception_mode": args.perception_mode,
        "model_image_resolution": (
            list(model_image_size)
            if model_image_size is not None
            else [768, 432]
        ),
        "repeats_per_case": args.repeats,
        "policy": {
            "green": "no current or projected forklift overlap",
            "amber": "chronological movement projects red entry within 3 seconds",
            "red": "any current forklift part overlaps the red zone",
            "workers_ignored": True,
        },
        "oracle_separation": {
            "semantic_annotations_sent_to_model": False,
            "simulator_coordinates_sent_to_model": False,
            "camera_facts_sent_to_model": False,
            "semantic_annotations_used_for_scoring": True,
        },
        "camera_scores": scores,
        "recommended_camera": best_camera,
        "semantic_actor_identification_rate": (
            round(semantic_complete / len(successful), 4)
            if successful else None
        ),
        "timing_summary": {
            "model_roundtrip_seconds": {
                "mean": (
                    round(sum(model_latencies) / len(model_latencies), 6)
                    if model_latencies else None
                ),
                "p50": percentile(model_latencies, 0.50),
                "p95": percentile(model_latencies, 0.95),
            },
            "case_total_seconds": {
                "mean": (
                    round(sum(cycle_latencies) / len(cycle_latencies), 6)
                    if cycle_latencies else None
                ),
                "p50": percentile(cycle_latencies, 0.50),
                "p95": percentile(cycle_latencies, 0.95),
            },
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
