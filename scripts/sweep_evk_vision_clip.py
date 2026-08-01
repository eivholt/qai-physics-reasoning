#!/usr/bin/env python3
"""Sweep EVK video resolution and frame count on one captured Isaac window."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_live_isaac_evk_supervisor import (  # noqa: E402
    DEFAULT_EVK_TARGET,
    DEFAULT_REMOTE_ROOT,
    EvkSshClient,
    _inference_job,
    _run,
    _windows_path_to_wsl,
)


def _integer_list(value: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        parsed = int(item.strip())
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"values must be between {minimum} and {maximum}"
            )
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return tuple(values)


def _float_list(value: str, *, minimum: float, maximum: float) -> tuple[float, ...]:
    values: list[float] = []
    for item in value.split(","):
        parsed = float(item.strip())
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"values must be between {minimum:g} and {maximum:g}"
            )
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing chronological frame_*.png captures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Artifact directory for encoded clips and summary.json.",
    )
    parser.add_argument("--evk-target", default=DEFAULT_EVK_TARGET)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--model-fps", type=float, default=2.0)
    parser.add_argument(
        "--widths",
        type=lambda value: _integer_list(value, minimum=384, maximum=1280),
        default=(384,),
        help="Comma-separated even video widths.",
    )
    parser.add_argument(
        "--frame-counts",
        type=lambda value: _integer_list(value, minimum=4, maximum=24),
        default=(8,),
        help="Comma-separated counts sampled from the end of the source window.",
    )
    parser.add_argument(
        "--crop-scales",
        type=lambda value: _float_list(value, minimum=0.5, maximum=1.0),
        default=(1.0, 0.75, 0.625),
        help=(
            "Comma-separated centered crop fractions. A 0.75 crop keeps the "
            "middle 75%% of each axis before the stable EVK downscale."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def _crop_source_frames(
    *,
    source_frames: list[Path],
    output_dir: Path,
    crop_scale: float,
) -> list[Path]:
    staging_dir = output_dir / "source_frames"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(source_frames):
        target = staging_dir / f"frame_{index:03d}.png"
        if not target.is_file():
            shutil.copy2(source, target)
    if crop_scale == 1.0:
        return sorted(staging_dir.glob("frame_*.png"))

    crop_name = f"crop_{crop_scale:.3f}".replace(".", "_")
    crop_dir = output_dir / "cropped_sources" / crop_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = crop_dir / "frame_%03d.png"
    _run(
        [
            "wsl",
            "-e",
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "2",
            "-i",
            _windows_path_to_wsl(staging_dir / "frame_%03d.png"),
            "-vf",
            (
                f"crop=trunc(iw*{crop_scale}/2)*2:"
                f"trunc(ih*{crop_scale}/2)*2:(iw-ow)/2:(ih-oh)/2"
            ),
            _windows_path_to_wsl(output_pattern),
        ],
        timeout_seconds=60.0,
    )
    cropped = sorted(crop_dir.glob("frame_*.png"))
    if len(cropped) != len(source_frames):
        raise RuntimeError(
            f"Expected {len(source_frames)} cropped frames, found {len(cropped)}"
        )
    return cropped


def main() -> int:
    args = build_parser().parse_args()
    if args.model_fps <= 0:
        raise SystemExit("--model-fps must be positive")
    if any(width % 2 for width in args.widths):
        raise SystemExit("--widths must contain only even integers")

    source_frames = sorted(args.frames_dir.glob("frame_*.png"))
    if not source_frames:
        raise SystemExit(f"No frame_*.png files found in {args.frames_dir}")
    if max(args.frame_counts) > len(source_frames):
        raise SystemExit(
            f"Requested {max(args.frame_counts)} frames, but only "
            f"{len(source_frames)} are available"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = EvkSshClient(
        target=args.evk_target,
        remote_root=args.remote_root,
        timeout_seconds=args.timeout_seconds,
    )
    client.prepare()

    results: list[dict[str, Any]] = []
    request_index = 0
    for crop_scale in args.crop_scales:
        cropped_source_frames = _crop_source_frames(
            source_frames=source_frames,
            output_dir=output_dir,
            crop_scale=crop_scale,
        )
        for frame_count in args.frame_counts:
            selected_frames = cropped_source_frames[-frame_count:]
            for width in args.widths:
                request_index += 1
                crop_token = round(crop_scale * 1000)
                request_id = (
                    f"offline-c{crop_token:04d}-f{frame_count:02d}-"
                    f"w{width:04d}-{request_index:02d}"
                )
                result = _inference_job(
                    client=client,
                    frame_paths=selected_frames,
                    request_dir=output_dir / request_id,
                    model_fps=args.model_fps,
                    video_width=width,
                    request_id=request_id,
                    commandable_actors=("RobotBlue",),
                    supervisor_profile="vision_only",
                    tracked_scene_facts=None,
                    sensor_camera="tactical",
                    vision_input="rolling_video",
                )
                row = {
                    "request_id": request_id,
                    "crop_scale": crop_scale,
                    "frame_count": frame_count,
                    "width": width,
                    "encoded_duration_seconds": result.get(
                        "encoded_duration_seconds"
                    ),
                    "model_responses": result.get("model_responses"),
                    "raw_command": result.get("raw_command"),
                    "command_source": result.get("command_source"),
                    "model_seconds": result.get("total_seconds"),
                    "round_trip_seconds": result.get("host_total_seconds"),
                    "local_clip": result.get("local_clip"),
                }
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "evk_target": args.evk_target,
        "source_frames": [str(path.resolve()) for path in source_frames],
        "model_fps": args.model_fps,
        "crop_scales": args.crop_scales,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
