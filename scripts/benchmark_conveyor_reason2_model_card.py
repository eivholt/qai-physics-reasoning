#!/usr/bin/env python3
"""Benchmark Reason2 with the model-card video and reasoning profile."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.isaac_sim_mcp.conveyor_safety import (
    get_conveyor_safety_state_source,
    set_conveyor_safety_forklift_pose_source,
)
from integrations.isaac_sim_mcp.server import IsaacTcpClient
from scripts.run_live_isaac_conveyor_safety import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SERVER_URL,
    capture_conveyor_frame,
    check_model_server,
    execute_isaac_source,
)
from scripts.run_live_isaac_evk_supervisor import _encode_frames


MODEL_CARD_SYSTEM_PROMPT = """\
You are a helpful physical-AI safety assistant.
Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>.
"""

MODEL_CARD_USER_PROMPT = """\
The video is from one fixed camera overlooking a warehouse conveyor at 4 FPS.
The timestamp at the bottom of each frame gives the temporal order. The bright
red-painted floor beside the blue conveyor is the only forklift exclusion zone.

Consider every physical part of each blue and black forklift, including its
wheels, body, mast, forks, and carried carton. Ignore people, shelf cartons,
conveyor parcels, yellow lane paint, and the stack light.

Apply this policy in order:
1. RED if any physical part of any forklift overlaps the red-painted floor in
   the final frame.
2. Otherwise AMBER if visible motion across the timestamped frames shows any
   forklift moving toward the red floor and likely to enter it within the next
   three seconds.
3. Otherwise GREEN.

Reason from visible position, overlap, direction, and speed. In the <answer>
element return exactly one signal name: GREEN, AMBER, or RED.
"""

VIDEO_FPS = 4.0
FRAMES_PER_CLIP = 8
CAPTURE_WIDTH = 384
CAPTURE_HEIGHT = 216
SIGNAL_PATTERN = re.compile(r"\b(GREEN|AMBER|RED)\b", re.IGNORECASE)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)


def scenario_poses() -> dict[str, list[tuple[float, float, float]]]:
    return {
        "GREEN": [(-0.40, 1.50, 0.0)] * FRAMES_PER_CLIP,
        "AMBER": [
            (-1.40 + 1.40 * index / (FRAMES_PER_CLIP - 1), 1.50, 0.0)
            for index in range(FRAMES_PER_CLIP)
        ],
        "RED": [(2.00, 1.50, 0.0)] * FRAMES_PER_CLIP,
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
        "--camera",
        default="detector_endline",
        choices=("detector_endline", "detector_oblique", "detector_overhead"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--capture-clips",
        action="store_true",
        help="Capture deterministic 8-frame clips; otherwise reuse clip-root.",
    )
    parser.add_argument(
        "--clip-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT.parent
            / "model_card_video_4fps"
        ),
    )
    parser.add_argument(
        "--server-video-root",
        help=(
            "Directory visible to the model server. Host defaults to the "
            "repository-relative clip root; EVK requires an absolute staged "
            "directory such as /home/ubuntu/conveyor-model-card-video."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT.parent
            / "model_card_video_benchmark.json"
        ),
    )
    return parser


def _clip_path(clip_root: Path, signal: str) -> Path:
    return clip_root / signal.lower() / "supervisor_window.mp4"


def capture_clips(
    *,
    clip_root: Path,
    camera: str,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    client = IsaacTcpClient(timeout_seconds=max(45.0, timeout_seconds))
    initial = execute_isaac_source(client, get_conveyor_safety_state_source())
    original_poses = {
        item["index"]: (
            item["position"][0],
            item["position"][1],
            item["yaw_degrees"],
        )
        for item in initial["forklifts"]
    }
    clip_metadata: dict[str, dict[str, Any]] = {}
    try:
        execute_isaac_source(
            client,
            "import omni.timeline\n"
            "omni.timeline.get_timeline_interface().pause()\n"
            "print('{}')",
        )
        for signal, poses in scenario_poses().items():
            frame_paths: list[Path] = []
            semantic_counts: list[int] = []
            for frame_index, (x, y, yaw) in enumerate(poses):
                execute_isaac_source(
                    client,
                    set_conveyor_safety_forklift_pose_source(
                        {
                            "index": 2,
                            "x": x,
                            "y": y,
                            "yaw_degrees": yaw,
                        }
                    ),
                )
                capture = capture_conveyor_frame(
                    client,
                    camera=camera,
                    filename=(
                        f"model_card_4fps_{camera}_{signal.lower()}_"
                        f"f{frame_index:03d}.png"
                    ),
                    width=CAPTURE_WIDTH,
                    height=CAPTURE_HEIGHT,
                )
                frame_paths.append(Path(capture["output_path"]).resolve())
                semantic_counts.append(
                    len(capture.get("forklift_pixel_boxes") or [])
                )
            case_root = clip_root / signal.lower()
            clip = _encode_frames(
                frame_paths=frame_paths,
                output_dir=case_root,
                model_fps=VIDEO_FPS,
                video_width=CAPTURE_WIDTH,
            )
            clip_metadata[signal] = {
                "clip_path": str(clip.resolve()),
                "frame_paths": [str(path) for path in frame_paths],
                "semantic_forklift_counts": semantic_counts,
                "semantic_actor_complete": all(
                    count == 3 for count in semantic_counts
                ),
                "poses": [list(pose) for pose in poses],
            }
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
    return clip_metadata


def video_url_for_server(
    *,
    clip: Path,
    clip_root: Path,
    backend_label: str,
    server_video_root: str | None,
) -> str:
    relative = clip.resolve().relative_to(clip_root.resolve())
    if server_video_root:
        return (
            "file://"
            + server_video_root.rstrip("/\\")
            + "/"
            + relative.as_posix()
        )
    if backend_label == "evk":
        raise ValueError("--server-video-root is required for EVK")
    repository_relative = clip.resolve().relative_to(REPO_ROOT)
    return "file://" + repository_relative.as_posix()


def parse_signal(
    text: str,
    *,
    finish_reason: str | None,
) -> tuple[str | None, str]:
    answer_match = ANSWER_PATTERN.search(text)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    candidates = SIGNAL_PATTERN.findall(answer_text)
    if candidates:
        return candidates[-1].upper(), answer_text
    if finish_reason == "length":
        return None, answer_text
    after_think = text.rsplit("</think>", 1)[-1]
    candidates = SIGNAL_PATTERN.findall(after_think)
    return (
        candidates[-1].upper() if candidates else None,
        answer_text,
    )


def request_model_card_video(
    *,
    server_url: str,
    model: str,
    video_url: str,
    timeout_seconds: float,
    max_completion_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": MODEL_CARD_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": video_url},
                    },
                    {"type": "text", "text": MODEL_CARD_USER_PROMPT},
                ],
            },
        ],
        "max_completion_tokens": max_completion_tokens,
        "enable_think": True,
        "top_k": 1,
        "temperature": 0,
        "seed": 42,
    }
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    choice = result["choices"][0]
    message = choice["message"]
    text = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    finish_reason = choice.get("finish_reason")
    signal, answer_text = parse_signal(
        text,
        finish_reason=finish_reason,
    )
    return {
        "signal": signal,
        "answer_text": answer_text,
        "raw_output": text,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "latency_seconds": round(elapsed, 6),
        "usage": result.get("usage"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.max_completion_tokens < 32:
        raise ValueError("--max-completion-tokens must be at least 32")
    clip_root = args.clip_root.resolve()
    if args.capture_clips:
        clip_metadata = capture_clips(
            clip_root=clip_root,
            camera=args.camera,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        clip_metadata = {
            signal: {
                "clip_path": str(_clip_path(clip_root, signal).resolve()),
            }
            for signal in scenario_poses()
        }
    for signal, metadata in clip_metadata.items():
        if not Path(metadata["clip_path"]).is_file():
            raise FileNotFoundError(
                f"{signal} clip does not exist: {metadata['clip_path']}"
            )

    check_model_server(
        args.server_url,
        timeout_seconds=args.timeout_seconds,
    )
    records: list[dict[str, Any]] = []
    for expected, metadata in clip_metadata.items():
        clip = Path(metadata["clip_path"])
        video_url = video_url_for_server(
            clip=clip,
            clip_root=clip_root,
            backend_label=args.backend_label,
            server_video_root=args.server_video_root,
        )
        for repeat_index in range(args.repeats):
            try:
                result = request_model_card_video(
                    server_url=args.server_url,
                    model=args.model,
                    video_url=video_url,
                    timeout_seconds=args.timeout_seconds,
                    max_completion_tokens=args.max_completion_tokens,
                )
                record = {
                    "expected": expected,
                    "repeat": repeat_index + 1,
                    "model_signal": result["signal"],
                    "match": result["signal"] == expected,
                    "video_url": video_url,
                    **result,
                }
            except Exception as exc:
                record = {
                    "expected": expected,
                    "repeat": repeat_index + 1,
                    "model_signal": None,
                    "match": False,
                    "video_url": video_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
            print(
                f"{args.backend_label:4s} {expected:5s} "
                f"r{repeat_index + 1} -> "
                f"{record.get('model_signal') or 'ERROR'} "
                f"({record.get('latency_seconds', 0):.3f}s)",
                flush=True,
            )

    successful = [record for record in records if "error" not in record]
    latencies = [
        float(record["latency_seconds"])
        for record in successful
    ]
    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": args.backend_label,
        "server_url": args.server_url,
        "model": args.model,
        "prompt_profile": "nvidia_model_card_reasoning",
        "system_prompt": MODEL_CARD_SYSTEM_PROMPT,
        "user_prompt": MODEL_CARD_USER_PROMPT,
        "video_profile": {
            "fps": VIDEO_FPS,
            "frames_per_clip": FRAMES_PER_CLIP,
            "duration_seconds": FRAMES_PER_CLIP / VIDEO_FPS,
            "resolution": [CAPTURE_WIDTH, CAPTURE_HEIGHT],
            "timestamp_overlay": "bottom of every frame",
        },
        "max_completion_tokens": args.max_completion_tokens,
        "enable_think": True,
        "clip_metadata": clip_metadata,
        "score": {
            "matches": sum(1 for record in records if record["match"]),
            "cases": len(records),
        },
        "latency_seconds": {
            "mean": (
                round(sum(latencies) / len(latencies), 6)
                if latencies
                else None
            ),
            "maximum": round(max(latencies), 6) if latencies else None,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["score"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
