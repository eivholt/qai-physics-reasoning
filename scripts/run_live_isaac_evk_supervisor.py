#!/usr/bin/env python3
"""Run a live Isaac Sim aisle-congestion loop with EVK visual advisories.

Isaac Sim keeps playing while a worker encodes the latest supervisor-camera window,
copies it to the EVK, and queries the persistent Cosmos-Reason2-2B GenieX
service. The model detects a conflict; a bounded policy gateway maps that
answer to the original route, a hold, or a modeled bypass. A local
distance guard remains authoritative if inference arrives too late.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.isaac_sim_mcp.server import IsaacMcpServer
from scripts.request_isaac_edge_supervisor_advisory import (
    GENERIC_COMMANDS,
    MODEL,
    VISION_ONLY_TUTORIAL_COMMANDS,
    generic_passage_supervisor_prompt,
    generic_staged_supervisor_prompts,
    request_passage_status_advisory,
    request_staged_generic_supervisor_advisory,
    request_tracked_state_supervisor_advisory,
    tracked_state_supervisor_prompt,
)


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "isaac_sim_live_aisle" / "runs"
DEFAULT_LIVE_STATUS_PATH = (
    REPO_ROOT / "artifacts" / "isaac_sim_live_aisle" / "live_status.json"
)
DEFAULT_CHECKPOINT = (
    "lightweight_warehouse_v21_deep_north_staging.usda"
)
DEFAULT_EVK_TARGET = "ubuntu@192.168.1.158"
DEFAULT_REMOTE_ROOT = "/home/ubuntu/test-data/isaac-live"
DEFAULT_HOST_SERVER_URL = "http://127.0.0.1:18080"
HOST_MODEL = "host/cosmos-reason2-2b:BF16"
REMOTE_CLIENT = "/home/ubuntu/deploy/request_isaac_edge_supervisor_advisory.py"
LOCAL_CLIENT = REPO_ROOT / "scripts" / "request_isaac_edge_supervisor_advisory.py"


def _decode_tool_response(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("isError"):
        text = response.get("content", [{}])[0].get("text", "unknown Isaac error")
        raise RuntimeError(text)
    try:
        raw = json.loads(response["content"][0]["text"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Isaac tool returned an invalid wrapper response") from exc
    if raw.get("status") != "ok":
        raise RuntimeError(str(raw))
    output = raw.get("output")
    if isinstance(output, str) and output.strip():
        try:
            decoded = json.loads(output)
        except json.JSONDecodeError:
            return {"output": output, "result": raw.get("result")}
        if isinstance(decoded, dict):
            return decoded
        return {"output": decoded}
    return {"result": raw.get("result")}


class IsaacLiveController:
    def __init__(self, server: IsaacMcpServer | None = None) -> None:
        self.server = server or IsaacMcpServer()

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return _decode_tool_response(self.server.call_tool(name, arguments or {}))

    def prepare(
        self,
        *,
        checkpoint: str | None,
        robot_scale: float,
        motion_controller: str,
        robot_speed: float,
        scenario: str = "blind_corner",
        configure_blind_corner: bool = True,
    ) -> dict[str, Any]:
        self.call("isaac_set_timeline", {"action": "stop"})
        if checkpoint is not None:
            self.call("isaac_load_stage_checkpoint", {"filename": checkpoint})
        self.call("isaac_set_frame_rate_limit", {"fps": 60})
        if configure_blind_corner:
            self.call("isaac_configure_live_blind_corner", {})
        scene = self.call(
            "isaac_create_live_aisle_navigation",
            {
                "robot_width": 1.42,
                "robot_length": 2.1,
                "robot_scale": robot_scale,
                "start_playing": False,
                "forklift_start_mode": "configured",
                "motion_controller": motion_controller,
                "kinematic_speed_mps": robot_speed,
                "scenario_mode": scenario,
            },
        )
        scene["robot_pov"] = self.call(
            "isaac_configure_live_robot_pov",
            {},
        )
        if motion_controller == "isaac_graph":
            scene["navigation_graph"] = self.call(
                "isaac_configure_live_navigation_graph",
                {},
            )
        else:
            scene["navigation_graph"] = {
                "configured": False,
                "controller": "kinematic_waypoint",
                "reason": (
                    "Tutorial uses a deterministic waypoint actuator while "
                    "retaining the packaged Carter visual and route targets"
                ),
            }
        scene["route_visualizations"] = self.call(
            "isaac_configure_live_route_visualizations",
            {},
        )
        return scene

    def set_congestion(self, released: bool) -> None:
        self.call("isaac_set_live_aisle_congestion", {"released": released})

    def set_forklift_visible(self, visible: bool) -> None:
        self.call(
            "isaac_set_live_forklift_visible",
            {"visible": visible},
        )

    def play(self) -> None:
        self.call("isaac_set_timeline", {"action": "play"})

    def tick(self) -> dict[str, Any]:
        return self.call("isaac_tick_live_aisle")

    def capture(
        self,
        filename: str,
        *,
        sensor_camera: str = "tactical",
        include_route_visualizations: bool = False,
    ) -> Path:
        if sensor_camera == "grid":
            result = self.call(
                "isaac_capture_live_aisle_camera_grid",
                {
                    "filename": filename,
                    "include_route_visualizations": (
                        include_route_visualizations
                    ),
                },
            )
            return self._compose_camera_grid(
                filename=filename,
                output_paths=result["output_paths"],
            )
        tool = {
            "roof": "isaac_capture_live_aisle_roof_frame",
            "tactical": "isaac_capture_live_aisle_tactical_frame",
        }.get(sensor_camera)
        if tool is None:
            raise ValueError("sensor_camera must be roof or tactical")
        result = self.call(
            tool,
            {
                "filename": filename,
                "include_route_visualizations": include_route_visualizations,
            },
        )
        output = Path(result["output_path"])
        if not output.is_file():
            raise RuntimeError(f"Isaac reported a missing capture: {output}")
        return output

    def capture_robot_front(self, filename: str) -> Path:
        result = self.call(
            "isaac_capture_live_robot_front_frame",
            {"filename": filename},
        )
        output = Path(result["output_path"])
        if not output.is_file():
            raise RuntimeError(
                f"Isaac reported a missing robot-front capture: {output}"
            )
        return output

    @staticmethod
    def _compose_camera_grid(
        *,
        filename: str,
        output_paths: dict[str, str],
    ) -> Path:
        layout = (
            "junction",
            "approach",
            "overview",
            "north_bypass",
        )
        sources: list[Path] = []
        for key in layout:
            try:
                source_path = Path(output_paths[key])
            except KeyError as exc:
                raise RuntimeError(
                    f"Isaac camera grid omitted {key!r}"
                ) from exc
            if not source_path.is_file():
                raise RuntimeError(
                    f"Isaac reported a missing grid capture: {source_path}"
                )
            sources.append(source_path)
        first_path = Path(output_paths["overview"])
        output = first_path.parent / filename
        filter_complex = (
            "[0:v]scale=640:360:force_original_aspect_ratio=increase,"
            "crop=640:360[p0];"
            "[1:v]scale=640:360:force_original_aspect_ratio=increase,"
            "crop=640:360[p1];"
            "[2:v]scale=640:360:force_original_aspect_ratio=increase,"
            "crop=640:360[p2];"
            "[3:v]scale=640:360:force_original_aspect_ratio=increase,"
            "crop=640:360[p3];"
            "[p0][p1]hstack=inputs=2[top];"
            "[p2][p3]hstack=inputs=2[bottom];"
            "[top][bottom]vstack=inputs=2[grid]"
        )
        command = ["wsl", "-e", "ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for source in sources:
            command.extend(["-i", _windows_path_to_wsl(source)])
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[grid]",
                "-frames:v",
                "1",
                _windows_path_to_wsl(output),
            ]
        )
        _run(command, timeout_seconds=30.0)
        if not output.is_file():
            raise RuntimeError(f"FFmpeg did not create camera grid {output}")
        return output

    def request_status(
        self,
        *,
        status: str,
        request_id: str,
        latency_seconds: float = 0.0,
    ) -> None:
        self.call(
            "isaac_set_live_aisle_request_status",
            {
                "status": status,
                "request_id": request_id,
                "latency_seconds": latency_seconds,
            },
        )

    def apply(
        self,
        *,
        action: str,
        route: str,
        request_id: str,
        latency_seconds: float,
        decision_source: str,
    ) -> dict[str, Any]:
        return self.call(
            "isaac_apply_live_aisle_advisory",
            {
                "action": action,
                "route": route,
                "request_id": request_id,
                "latency_seconds": latency_seconds,
                "decision_source": decision_source,
            },
        )

    def stop(self) -> None:
        self.call("isaac_set_timeline", {"action": "stop"})


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive
    if len(drive) == 2 and drive[1] == ":":
        remainder = resolved.as_posix()[2:].lstrip("/")
        return f"/mnt/{drive[0].lower()}/{remainder}"
    return resolved.as_posix()


def _run(
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 3000:
            detail = detail[-3000:]
        executable = Path(command[0]).name if command else "subprocess"
        raise RuntimeError(
            f"{executable} exited with status {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete status snapshot without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def _destination_observation(
    *,
    scenario_released: bool,
    last_tick: dict[str, Any],
    expected_route: str,
) -> tuple[bool, bool | None]:
    reached = scenario_released and bool(last_tick.get("destination_reached"))
    if not reached:
        return False, None
    return True, last_tick.get("route") == expected_route


def _emit_json(payload: dict[str, Any], *, pretty: bool = False) -> None:
    """Best-effort console telemetry; persisted event files remain authoritative."""
    try:
        print(
            json.dumps(
                payload,
                indent=2 if pretty else None,
                sort_keys=not pretty,
            ),
            flush=True,
        )
    except (BrokenPipeError, OSError):
        # Background launchers and bounded tool pipes may detach stdout while
        # the simulation should continue recording to its event/status files.
        pass


def _tracked_scene_facts(
    *,
    state_history: Sequence[dict[str, Any]],
    phase: str,
    forklift_released: bool,
    forklift_enabled: bool = True,
) -> dict[str, Any]:
    """Summarize trusted simulator tracks without embedding a model command."""
    if not state_history:
        raise ValueError("state_history must contain at least one capture state")

    first = state_history[0]
    last = state_history[-1]

    def position(state: dict[str, Any], key: str) -> tuple[float, float]:
        values = state.get(key)
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(f"{key} is missing from tracked simulator state")
        return float(values[0]), float(values[1])

    def track(key: str) -> dict[str, Any]:
        start = position(first, key)
        end = position(last, key)
        first_time = float(first.get("timeline_time_seconds", 0.0))
        last_time = float(last.get("timeline_time_seconds", first_time))
        elapsed = max(0.001, last_time - first_time)
        velocity = (
            (end[0] - start[0]) / elapsed,
            (end[1] - start[1]) / elapsed,
        )
        speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        if speed < 0.04:
            direction = "stationary"
        elif abs(velocity[0]) >= abs(velocity[1]):
            direction = "east" if velocity[0] > 0 else "west"
        else:
            direction = "north" if velocity[1] > 0 else "south"
        return {
            "start_xy_m": [round(start[0], 3), round(start[1], 3)],
            "current_xy_m": [round(end[0], 3), round(end[1], 3)],
            "velocity_xy_mps": [
                round(velocity[0], 3),
                round(velocity[1], 3),
            ],
            "motion": direction,
        }

    robot_track = track("blue_position")
    forklift_track = track("forklift_position")
    current_route = str(last.get("route", "direct"))
    forklift_y = forklift_track["current_xy_m"][1]
    forklift_is_entering = (
        forklift_released
        and (
            forklift_track["motion"] == "south"
            or float(forklift_y) <= 4.45
        )
    )
    direct_status = "EMERGING_BLOCK" if forklift_is_entering else "CLEAR"
    evaluated_route = (
        "direct"
        if phase == "clear_baseline" or current_route == "hold"
        else current_route
    )
    route_statuses = {"direct": direct_status}
    actor_tracks = {
        "RobotBlue": {
            "type": "autonomous_mobile_robot",
            **robot_track,
        },
    }
    clearance: dict[str, Any]
    if forklift_enabled:
        route_statuses.update(
            {
                "north_bypass": "OPEN",
                "south_bypass": "BLOCKED_STATIC_PALLETS",
            }
        )
        actor_tracks["ForkliftBlue"] = {
            "type": "forklift",
            "enabled": True,
            "visible": phase != "clear_baseline",
            "released": forklift_released,
            **forklift_track,
        }
        direct_route_shape = (
            "east aisle, then north turn into northeast loading approach"
        )
        clearance = {
            "robot_and_forklift_cannot_safely_pass_in_direct_north_turn": True,
            "south_bypass_blocker": "static pallets",
        }
    else:
        direct_route_shape = (
            "east through the clear east intersection, north to the upper "
            "aisle, then west to the northwest clear-control destination"
        )
        clearance = {
            "other_traffic_enabled": False,
            "control_purpose": "negative control for false interventions",
        }
    evaluated_status = route_statuses.get(evaluated_route, "UNKNOWN")
    return {
        "schema_version": "1.0",
        "source": "isaac_sim_ground_truth_tracker_and_static_route_map",
        "phase": phase,
        "camera_evidence": {
            "attached": True,
            "presentation_routes_hidden": True,
        },
        "commandable_actors": ["RobotBlue"],
        "actor_tracks": actor_tracks,
        "navigation": {
            "active_route": current_route,
            "evaluated_current_route": evaluated_route,
            "evaluated_current_corridor_status": evaluated_status,
            "route_statuses": route_statuses,
            "direct_route_shape": direct_route_shape,
        },
        "clearance": clearance,
    }


@dataclass(frozen=True)
class EvkSshClient:
    target: str
    remote_root: str
    timeout_seconds: float

    @staticmethod
    def _ssh_options() -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=120",
            "-o",
            "ControlPath=/tmp/qai-isaac-evk-%C",
        ]

    def remote(self, arguments: Sequence[str], *, timeout: float | None = None) -> str:
        command = shlex.join(list(arguments))
        result = _run(
            [
                "wsl",
                "-e",
                "ssh",
                *self._ssh_options(),
                self.target,
                command,
            ],
            timeout_seconds=timeout or self.timeout_seconds,
        )
        return result.stdout.strip()

    def copy(self, local_path: Path, remote_path: str) -> None:
        _run(
            [
                "wsl",
                "-e",
                "scp",
                "-q",
                *self._ssh_options(),
                _windows_path_to_wsl(local_path),
                f"{self.target}:{remote_path}",
            ],
            timeout_seconds=self.timeout_seconds,
        )

    def prepare(self) -> None:
        self.remote(["mkdir", "-p", self.remote_root, "/home/ubuntu/deploy"])
        self.copy(LOCAL_CLIENT, REMOTE_CLIENT)
        models = self.remote(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "http://127.0.0.1:18181/v1/models",
            ]
        )
        payload = json.loads(models)
        model_ids = {item.get("id") for item in payload.get("data", [])}
        if not {
            "local/cosmos-reason2-2b",
            "local/cosmos-reason2-2b:Q4_0",
        }.intersection(model_ids):
            raise RuntimeError("The EVK service does not expose Cosmos-Reason2-2B")

    def request(
        self,
        *,
        clip: Path,
        request_id: str,
        commandable_actors: tuple[str, ...],
        supervisor_profile: str,
        tracked_scene_facts: dict[str, Any] | None,
        sensor_camera: str = "tactical",
        vision_input: str = "temporal_pair",
    ) -> dict[str, Any]:
        remote_clip = f"{self.remote_root}/{request_id}{clip.suffix.lower()}"
        self.copy(clip, remote_clip)
        arguments = [
            "python3",
            REMOTE_CLIENT,
            "--scenario",
            "aisle_congestion",
            "--video",
            remote_clip,
            "--commandable-actors",
            ",".join(commandable_actors),
            "--timeout-seconds",
            str(self.timeout_seconds),
        ]
        if supervisor_profile == "tracked":
            if tracked_scene_facts is None:
                raise ValueError(
                    "tracked supervisor profile requires tracked_scene_facts"
                )
            arguments.extend(
                [
                    "--tracked-state-json",
                    json.dumps(
                        tracked_scene_facts,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ]
            )
        else:
            arguments.append("--passage-status-only")
            arguments.extend(
                [
                    "--vision-input",
                    vision_input,
                    "--camera-layout",
                    "grid" if sensor_camera == "grid" else "single",
                ]
            )
        output = self.remote(
            arguments,
            timeout=self.timeout_seconds + 20.0,
        )
        result = json.loads(output)
        result["remote_clip"] = remote_clip
        return result


@dataclass(frozen=True)
class HostCosmosClient:
    server_url: str
    timeout_seconds: float
    model: str = HOST_MODEL

    def prepare(self) -> None:
        request = urllib.request.Request(
            f"{self.server_url.rstrip('/')}/health",
            headers={"Connection": "close"},
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
        ) as response:
            payload = json.load(response)
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"Host Cosmos server is not healthy: {payload!r}"
            )

    def request(
        self,
        *,
        clip: Path,
        request_id: str,
        commandable_actors: tuple[str, ...],
        supervisor_profile: str,
        tracked_scene_facts: dict[str, Any] | None,
        sensor_camera: str = "tactical",
        vision_input: str = "temporal_pair",
    ) -> dict[str, Any]:
        try:
            relative_clip = clip.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "Host inference clips must be inside the repository media root"
            ) from exc
        video_url = "file://" + relative_clip.as_posix()
        if supervisor_profile == "tracked":
            if tracked_scene_facts is None:
                raise ValueError(
                    "tracked supervisor profile requires tracked_scene_facts"
                )
            result = request_tracked_state_supervisor_advisory(
                video_url=video_url,
                tracked_scene_facts=tracked_scene_facts,
                commandable_actors=commandable_actors,
                server_url=self.server_url,
                model=self.model,
                timeout_seconds=self.timeout_seconds,
            )
        else:
            result = request_passage_status_advisory(
                video_url=video_url,
                server_url=self.server_url,
                model=self.model,
                timeout_seconds=self.timeout_seconds,
                vision_input=vision_input,
            )
        result["host_clip"] = video_url
        result["request_id"] = request_id
        return result


def _encode_frames(
    *,
    frame_paths: Sequence[Path],
    output_dir: Path,
    model_fps: float,
    video_width: int,
    temporal_comparison: bool = False,
    comparison_video: bool = False,
    clean_video: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(frame_paths):
        shutil.copy2(source, output_dir / f"frame_{index:03d}.png")
    if temporal_comparison:
        comparison = output_dir / "supervisor_temporal_comparison.png"
        _run(
            [
                "wsl",
                "-e",
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                _windows_path_to_wsl(frame_paths[0]),
                "-i",
                _windows_path_to_wsl(frame_paths[-1]),
                "-filter_complex",
                (
                    f"[0:v]scale={video_width}:-2:flags=lanczos,"
                    "drawtext=text='EARLIER':x=12:y=12:fontsize=30:"
                    "fontcolor=white:box=1:boxcolor=black@0.78:"
                    "boxborderw=7[a];"
                    f"[1:v]scale={video_width}:-2:flags=lanczos,"
                    "drawtext=text='NOW':x=12:y=12:fontsize=30:"
                    "fontcolor=white:box=1:boxcolor=black@0.78:"
                    "boxborderw=7[b];"
                    "[a][b]hstack=inputs=2[comparison]"
                ),
                "-map",
                "[comparison]",
                "-frames:v",
                "1",
                _windows_path_to_wsl(comparison),
            ],
            timeout_seconds=60.0,
        )
        if not comparison.is_file():
            raise RuntimeError(
                f"ffmpeg did not create temporal comparison {comparison}"
            )
        if comparison_video:
            video_height = round(video_width * 9 / 16)
            if video_height % 2:
                video_height += 1
            clip = output_dir / "supervisor_temporal_comparison.mp4"
            _run(
                [
                    "wsl",
                    "-e",
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    _windows_path_to_wsl(comparison),
                    "-vf",
                    (
                        f"scale={video_width}:{video_height}:"
                        "force_original_aspect_ratio=decrease:flags=lanczos,"
                        f"pad={video_width}:{video_height}:"
                        "(ow-iw)/2:(oh-ih)/2:color=black"
                    ),
                    "-t",
                    "1",
                    "-r",
                    f"{model_fps:g}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    _windows_path_to_wsl(clip),
                ],
                timeout_seconds=60.0,
            )
            if not clip.is_file():
                raise RuntimeError(
                    f"ffmpeg did not create EVK comparison video {clip}"
                )
            return clip
        return comparison
    clip = output_dir / "supervisor_window.mp4"
    video_filter = f"scale={video_width}:-2:flags=lanczos"
    if not clean_video:
        video_filter += (
            ",drawtext=text='t=%{pts\\:hms}':"
            "x=12:y=h-th-12:fontsize=22:fontcolor=white:"
            "box=1:boxcolor=black@0.72:boxborderw=6"
        )
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
            f"{model_fps:g}",
            "-i",
            _windows_path_to_wsl(output_dir / "frame_%03d.png"),
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{model_fps:g}",
            _windows_path_to_wsl(clip),
        ],
        timeout_seconds=60.0,
    )
    if not clip.is_file():
        raise RuntimeError(f"ffmpeg did not create {clip}")
    return clip


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_text(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", "")
        .replace("\n", r"\N")
    )


def _compact_actor_inventory(model_responses: Any) -> str:
    if not isinstance(model_responses, dict):
        return "awaiting first inference result"
    traffic_actors = model_responses.get("traffic_actors")
    if isinstance(traffic_actors, list):
        display_names = {
            "mobile_robot": "mobile robot",
            "forklift": "forklift",
        }
        actors = [
            display_names.get(str(actor), str(actor))
            for actor in traffic_actors
        ]
        return ", ".join(actors) if actors else "none"
    inventory = str(model_responses.get("scene_inventory") or "")
    statuses: list[str] = []
    for label, display in (
        ("mobile robot", "RobotBlue"),
        ("forklift", "Forklift"),
    ):
        status = "UNKNOWN"
        for line in inventory.splitlines():
            if line.strip().lower().startswith(f"{label}:"):
                candidate = line.split(":", 1)[1].strip().upper()
                if candidate in {"PRESENT", "ABSENT", "UNCERTAIN"}:
                    status = candidate
                break
        statuses.append(f"{display} {status}")
    return "  |  ".join(statuses)


def _compact_passage_state(model_responses: Any) -> str:
    if not isinstance(model_responses, dict):
        return "awaiting first inference result"
    passage_states = model_responses.get("passage_states")
    if isinstance(passage_states, dict):
        values: list[str] = []
        for passage_name in ("left", "right"):
            state = passage_states.get(f"{passage_name}_passage")
            if not isinstance(state, dict):
                values.append(f"{passage_name}: not reported")
                continue
            status = str(state.get("status") or "UNKNOWN")
            offender = str(state.get("offender") or "UNKNOWN")
            suffix = "" if offender == "NONE" else f" <- {offender.lower()}"
            values.append(f"{passage_name}: {status}{suffix}")
        return "  |  ".join(values)
    blocked = model_responses.get("passages_blocked")
    imminent = model_responses.get("passages_blockage_imminent")
    if not isinstance(blocked, list) or not isinstance(imminent, list):
        return "not reported"
    blocked_text = ", ".join(str(item) for item in blocked) or "none"
    imminent_items: list[str] = []
    for item in imminent:
        if isinstance(item, dict):
            passage = str(item.get("name") or "?")
            offender = str(item.get("offender") or "?")
            imminent_items.append(f"{passage} <- {offender}")
        else:
            imminent_items.append(str(item))
    imminent_text = ", ".join(imminent_items) or "none"
    return f"blocked: {blocked_text}  |  imminent: {imminent_text}"


def _write_presentation_overlay(
    *,
    output_path: Path,
    frame_states: Sequence[dict[str, Any]],
    playback_fps: float,
    response_pulse_seconds: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = len(frame_states) / playback_fps
    header = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Arial,21,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,-1,0,0,0,100,100,0,0,3,1,0,8,20,20,14,1
Style: Status,Arial,17,&H00FFFFFF,&H00FFFFFF,&H00101010,&HB0000000,0,0,0,0,100,100,0,0,3,1,0,8,20,20,50,1
Style: Pulse,Arial,27,&H0000E8FF,&H0000E8FF,&H00101010,&HBC000000,-1,0,0,0,100,100,0,0,3,1,0,8,20,20,158,1
Style: Pip,Arial,17,&H00FFFFFF,&H00FFFFFF,&H00101010,&HB0000000,-1,0,0,0,100,100,0,0,3,1,0,1,28,20,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    events.append(
        "Dialogue: 0,0:00:00.00,"
        f"{_ass_timestamp(duration)},Title,,0,0,0,,"
        "EDGE AI WAREHOUSE SUPERVISOR"
    )
    events.append(
        "Dialogue: 0,0:00:00.00,"
        f"{_ass_timestamp(duration)},Pip,,0,0,0,,"
        "ROBOTBLUE FORWARD CAMERA"
    )

    for index, state in enumerate(frame_states):
        start = index / playback_fps
        end = (index + 1) / playback_fps
        request_id = state.get("request_id") or "-"
        adapter_command = state.get("adapter_command") or "awaiting result"
        route = state.get("accepted_route") or "direct"
        latency = state.get("latency_seconds")
        latency_text = (
            f"{float(latency):.2f} s"
            if isinstance(latency, (int, float))
            else "-"
        )
        status_text = (
            f"MODEL ACTORS  {_ass_text(state.get('actors', 'awaiting result'))}"
            rf"\NMODEL PASSAGES  {_ass_text(state.get('passages', 'awaiting result'))}"
            rf"\NROBOT ADAPTER  {_ass_text(adapter_command)}"
            rf"\NACCEPTED ROUTE  {_ass_text(route)}"
            rf"\NREQUEST  {_ass_text(request_id)}  |  LATENCY  {latency_text}"
        )
        events.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(start)},{_ass_timestamp(end)},"
            f"Status,,0,0,0,,{status_text}"
        )
        if state.get("new_response"):
            response_number = int(state.get("response_number") or 0)
            pulse_end = min(duration, start + response_pulse_seconds)
            events.append(
                "Dialogue: 1,"
                f"{_ass_timestamp(start)},{_ass_timestamp(pulse_end)},"
                "Pulse,,0,0,0,,"
                rf"{{\fad(80,520)}}NEW INFERENCE RESULT #{response_number}"
            )

    output_path.write_text(
        header + "\n".join(events) + "\n",
        encoding="utf-8",
    )
    return output_path


def _encode_presentation(
    *,
    frames_dir: Path,
    output_dir: Path,
    playback_fps: float,
    make_gif: bool,
    robot_frames_dir: Path | None = None,
    frame_states: Sequence[dict[str, Any]] = (),
    response_pulse_seconds: float = 1.25,
    pip_width: int = 360,
) -> dict[str, str]:
    dashboard_enabled = (
        robot_frames_dir is not None
        and bool(frame_states)
        and len(frame_states)
        == len(list(frames_dir.glob("frame_*.png")))
    )
    clean_mp4 = (
        output_dir / "live_aisle_supervisor_clean.mp4"
        if dashboard_enabled
        else output_dir / "live_aisle_supervisor.mp4"
    )
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
            f"{playback_fps:g}",
            "-i",
            _windows_path_to_wsl(frames_dir / "frame_%04d.png"),
            "-vf",
            "scale=1280:-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            _windows_path_to_wsl(clean_mp4),
        ],
        timeout_seconds=120.0,
    )
    if not dashboard_enabled:
        mp4 = clean_mp4
        outputs = {"mp4": str(mp4)}
    else:
        overlay_path = _write_presentation_overlay(
            output_path=output_dir / "presentation_overlay.ass",
            frame_states=frame_states,
            playback_fps=playback_fps,
            response_pulse_seconds=response_pulse_seconds,
        )
        mp4 = output_dir / "live_aisle_supervisor.mp4"
        overlay_filter = (
            "[0:v]scale=1280:720:force_original_aspect_ratio=decrease:"
            "flags=lanczos,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black[base];"
            f"[1:v]scale={pip_width}:-2:flags=lanczos,"
            "pad=iw+8:ih+8:4:4:color=white[pip];"
            "[base][pip]overlay=20:H-h-20[combined];"
            f"[combined]ass='{_windows_path_to_wsl(overlay_path)}'[out]"
        )
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
                f"{playback_fps:g}",
                "-i",
                _windows_path_to_wsl(frames_dir / "frame_%04d.png"),
                "-framerate",
                f"{playback_fps:g}",
                "-i",
                _windows_path_to_wsl(
                    robot_frames_dir / "frame_%04d.png"
                ),
                "-filter_complex",
                overlay_filter,
                "-map",
                "[out]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                _windows_path_to_wsl(mp4),
            ],
            timeout_seconds=180.0,
        )
        if not mp4.is_file():
            raise RuntimeError(
                f"ffmpeg did not create dashboard presentation {mp4}"
            )
        outputs = {
            "mp4": str(mp4),
            "clean_mp4": str(clean_mp4),
            "overlay_ass": str(overlay_path),
            "robot_pov_frames": str(robot_frames_dir),
        }
    if make_gif:
        gif = output_dir / "live_aisle_supervisor.gif"
        _run(
            [
                "wsl",
                "-e",
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                _windows_path_to_wsl(mp4),
                "-vf",
                "fps=5,scale=960:-2:flags=lanczos",
                "-loop",
                "0",
                _windows_path_to_wsl(gif),
            ],
            timeout_seconds=120.0,
        )
        outputs["gif"] = str(gif)
    return outputs


def _inference_job(
    *,
    client: EvkSshClient | HostCosmosClient,
    frame_paths: Sequence[Path],
    request_dir: Path,
    model_fps: float,
    video_width: int,
    request_id: str,
    commandable_actors: tuple[str, ...],
    supervisor_profile: str,
    tracked_scene_facts: dict[str, Any] | None,
    sensor_camera: str,
    vision_input: str,
) -> dict[str, Any]:
    started = time.monotonic()
    use_temporal_pair = (
        supervisor_profile == "vision_only"
        and vision_input == "temporal_pair"
    )
    clip = _encode_frames(
        frame_paths=frame_paths,
        output_dir=request_dir,
        model_fps=model_fps,
        video_width=video_width,
        temporal_comparison=use_temporal_pair,
        comparison_video=(
            use_temporal_pair
            and isinstance(client, EvkSshClient)
        ),
        clean_video=(
            supervisor_profile == "vision_only"
            and vision_input == "rolling_video"
        ),
    )
    result = client.request(
        clip=clip,
        request_id=request_id,
        commandable_actors=commandable_actors,
        supervisor_profile=supervisor_profile,
        tracked_scene_facts=tracked_scene_facts,
        sensor_camera=sensor_camera,
        vision_input=vision_input,
    )
    result["host_total_seconds"] = round(time.monotonic() - started, 6)
    result["local_clip"] = str(clip)
    result["vision_input"] = vision_input
    result["source_frame_count"] = len(frame_paths)
    result["encoded_fps"] = model_fps
    result["encoded_duration_seconds"] = round(
        len(frame_paths) / model_fps,
        6,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-backend",
        choices=("evk", "host"),
        default="evk",
        help=(
            "Run Cosmos-Reason2 on the IQ9075 EVK or a local llama.cpp "
            "server on the host GPU."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("blind_corner", "clear_route_control"),
        default="blind_corner",
        help=(
            "Run the southbound-forklift congestion case or an independent "
            "negative control with the forklift disabled."
        ),
    )
    parser.add_argument("--evk-target", default=DEFAULT_EVK_TARGET)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--host-server-url",
        default=DEFAULT_HOST_SERVER_URL,
        help="OpenAI-compatible Cosmos-Reason2 host server URL.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--preserve-current-stage",
        action="store_true",
        help=(
            "Do not load the checkpoint; replace only /World/CodexPoC/LiveAisle "
            "so manual warehouse and pallet edits remain intact."
        ),
    )
    parser.add_argument(
        "--use-current-forklift-start",
        action="store_true",
        help=(
            "Use and snapshot the forklift's current GUI transform instead of "
            "applying the repeatable central blind-corner start."
        ),
    )
    parser.add_argument("--robot-scale", type=float, default=1.75)
    parser.add_argument("--capture-fps", type=float, default=2.0)
    parser.add_argument(
        "--model-fps",
        type=float,
        default=2.0,
        help=(
            "Playback rate encoded into model clips. The measured EVK live "
            "demo profile uses two FPS."
        ),
    )
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=8,
        help=(
            "Chronological frames per model request. Eight frames at two FPS "
            "is the measured latency/accuracy sweet spot for this demo."
        ),
    )
    parser.add_argument(
        "--sensor-camera",
        choices=("grid", "tactical", "roof"),
        default="tactical",
        help=(
            "Supervisor sensor sent to the model. Grid combines synchronized "
            "context and detail; tactical isolates the blind corner; roof "
            "keeps one full overview."
        ),
    )
    parser.add_argument(
        "--vision-input",
        choices=("temporal_pair", "rolling_video"),
        default="rolling_video",
        help=(
            "Send an EARLIER/NOW composite or every frame from the rolling "
            "window as one real chronological video. Applies to vision_only."
        ),
    )
    parser.add_argument(
        "--presentation-camera",
        choices=("sensor", "grid", "tactical", "roof"),
        default="sensor",
        help=(
            "Camera recorded for the tutorial movie. 'sensor' reuses the "
            "clean model input; another choice records a separate view."
        ),
    )
    parser.add_argument(
        "--presentation-route-visualizations",
        action="store_true",
        help=(
            "Show candidate routes only in the tutorial recording. Model "
            "sensor frames always exclude route visualizations."
        ),
    )
    parser.add_argument(
        "--presentation-dashboard",
        action="store_true",
        help=(
            "Record RobotBlue's forward camera every presentation frame and "
            "compose it as a lower-left picture-in-picture with actor, command, "
            "route, latency, and per-response pulse overlays."
        ),
    )
    parser.add_argument(
        "--presentation-response-pulse-seconds",
        type=float,
        default=1.25,
        help=(
            "Presentation-video duration of the fading NEW INFERENCE RESULT "
            "pulse, including when consecutive results are identical."
        ),
    )
    parser.add_argument(
        "--presentation-pip-width",
        type=int,
        default=360,
        help="RobotBlue picture-in-picture width in the 1280-pixel movie.",
    )
    parser.add_argument(
        "--supervisor-profile",
        choices=("tracked", "vision_only"),
        default="vision_only",
        help=(
            "tracked combines camera evidence with simulator actor tracks and "
            "route occupancy; vision_only retains the harder experimental "
            "camera-only prompt."
        ),
    )
    parser.add_argument(
        "--motion-controller",
        choices=("kinematic_waypoint", "isaac_graph"),
        default="isaac_graph",
        help=(
            "Physics wheel navigation through Isaac DifferentialController or "
            "the deterministic root-motion fallback."
        ),
    )
    parser.add_argument(
        "--robot-speed",
        type=float,
        default=0.3,
        help="Kinematic tutorial robot speed in metres per simulation second.",
    )
    parser.add_argument(
        "--evk-video-width",
        type=int,
        default=384,
        help=(
            "Width of the roof-camera clip sent to the EVK. The measured "
            "stable native-video profile is 384x216."
        ),
    )
    parser.add_argument("--max-runtime-seconds", type=float, default=300.0)
    parser.add_argument(
        "--destination-settle-frames",
        type=int,
        default=4,
        help=(
            "Continue recording this many presentation frames after RobotBlue "
            "first reaches the active route destination."
        ),
    )
    parser.add_argument(
        "--max-responses",
        type=int,
        default=0,
        help="Stop after this many EVK responses; zero runs until the time limit.",
    )
    parser.add_argument(
        "--clear-baseline-responses",
        type=int,
        default=0,
        help=(
            "Keep RobotBlue and the forklift held until this many clear-scene "
            "responses have been recorded, then release the scenario."
        ),
    )
    parser.add_argument(
        "--reroute-confirmations",
        type=int,
        default=3,
        help=(
            "Require this many consecutive camera-only reroute advisories "
            "before changing RobotBlue's route. This suppresses isolated "
            "visual false positives without using simulator actor tracks."
        ),
    )
    parser.add_argument(
        "--forklift-release-x",
        type=float,
        default=1.5,
        help=(
            "Release the blind-corner forklift only after RobotBlue reaches "
            "this world-X coordinate, so both actors visibly converge. Use a "
            "negative value to let the forklift emerge before RobotBlue enters "
            "the tight security-camera view."
        ),
    )
    parser.add_argument(
        "--commandable-actors",
        default="RobotBlue",
        help="Comma-separated actors that the visual supervisor may command.",
    )
    parser.add_argument("--evk-timeout-seconds", type=float, default=40.0)
    parser.add_argument(
        "--request-retry-seconds",
        type=float,
        default=5.0,
        help="Backoff after a failed EVK request.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop the bounded run after this many consecutive EVK failures.",
    )
    parser.add_argument("--safety-hold-distance", type=float, default=3.2)
    parser.add_argument("--presentation-fps", type=float, default=5.0)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--leave-playing", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=DEFAULT_LIVE_STATUS_PATH,
        help="Atomic JSON status consumed by the Isaac supervisor UI.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1.0 <= args.robot_scale <= 2.5:
        raise ValueError("--robot-scale must be between 1 and 2.5")
    if not 0.1 <= args.robot_speed <= 0.8:
        raise ValueError("--robot-speed must be between 0.1 and 0.8")
    if not 0.5 <= args.capture_fps <= 4.0:
        raise ValueError("--capture-fps must be between 0.5 and 4")
    if not 1.0 <= args.model_fps <= 8.0:
        raise ValueError("--model-fps must be between 1 and 8")
    if not 4 <= args.frames_per_window <= 24:
        raise ValueError("--frames-per-window must be between 4 and 24")
    if not 384 <= args.evk_video_width <= 1280 or args.evk_video_width % 2:
        raise ValueError(
            "--evk-video-width must be an even integer between 384 and 1280"
        )
    if args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be positive")
    if not 0 <= args.destination_settle_frames <= 30:
        raise ValueError("--destination-settle-frames must be between 0 and 30")
    if not 0.4 <= args.presentation_response_pulse_seconds <= 5.0:
        raise ValueError(
            "--presentation-response-pulse-seconds must be between 0.4 and 5"
        )
    if not 240 <= args.presentation_pip_width <= 640:
        raise ValueError("--presentation-pip-width must be between 240 and 640")
    if args.max_responses < 0:
        raise ValueError("--max-responses must not be negative")
    if not 1.0 <= args.request_retry_seconds <= 60.0:
        raise ValueError("--request-retry-seconds must be between 1 and 60")
    if not 1 <= args.max_consecutive_failures <= 10:
        raise ValueError("--max-consecutive-failures must be between 1 and 10")
    if not 0 <= args.clear_baseline_responses <= 10:
        raise ValueError("--clear-baseline-responses must be between 0 and 10")
    if not 1 <= args.reroute_confirmations <= 5:
        raise ValueError("--reroute-confirmations must be between 1 and 5")
    if not -5.5 <= args.forklift_release_x <= 4.0:
        raise ValueError("--forklift-release-x must be between -5.5 and 4")
    if not 2.2 <= args.safety_hold_distance <= 6.0:
        raise ValueError("--safety-hold-distance must be between 2.2 and 6")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    session_dir = args.artifact_root.resolve() / run_id
    frames_dir = session_dir / "frames"
    robot_frames_dir = (
        session_dir / "robot_pov_frames"
        if args.presentation_dashboard
        else None
    )
    requests_dir = session_dir / "requests"
    frames_dir.mkdir(parents=True, exist_ok=False)
    if robot_frames_dir is not None:
        robot_frames_dir.mkdir(parents=True, exist_ok=False)
    requests_dir.mkdir(parents=True, exist_ok=True)
    events_path = session_dir / "events.jsonl"
    status_path = args.status_path.expanduser().resolve()
    wall_started = time.monotonic()
    commandable_actors = tuple(
        item.strip() for item in args.commandable_actors.split(",") if item.strip()
    )
    if commandable_actors != ("RobotBlue",):
        raise ValueError(
            "This milestone currently supports --commandable-actors RobotBlue"
        )
    staged_prompts = generic_staged_supervisor_prompts(commandable_actors)
    if args.sensor_camera == "grid":
        grid_calibration = (
            "Each video frame is a synchronized 2-by-2 grid of fixed north-up "
            "cameras: top-left BLIND CORNER, top-right WEST APPROACH, "
            "bottom-left ROOF OVERVIEW, and bottom-right NORTH AISLE. The same "
            "actor may appear in more than one panel; do not count duplicates "
            "as separate traffic. "
        )
        staged_prompts = {
            name: grid_calibration + prompt
            for name, prompt in staged_prompts.items()
        }
    if args.inference_backend == "host":
        inference_client: EvkSshClient | HostCosmosClient = HostCosmosClient(
            server_url=args.host_server_url,
            timeout_seconds=args.evk_timeout_seconds,
        )
        inference_target = args.host_server_url
        inference_model = HOST_MODEL
    else:
        inference_client = EvkSshClient(
            target=args.evk_target,
            remote_root=args.remote_root.rstrip("/"),
            timeout_seconds=args.evk_timeout_seconds,
        )
        inference_target = args.evk_target
        inference_model = MODEL
    status: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "session_dir": str(session_dir),
        "state": "starting",
        "phase": "setup",
        "inference_backend": args.inference_backend,
        "inference_target": inference_target,
        "evk_target": inference_target,
        "model": inference_model,
        "scenario": args.scenario,
        "commandable_actors": list(commandable_actors),
        "supervisor_profile": args.supervisor_profile,
        "vision_input": args.vision_input,
        "motion_controller": args.motion_controller,
        "robot_speed_mps": args.robot_speed,
        "prompt": (
            (
                "HYBRID CAMERA + TRACKED STATE\n"
                "The exact per-window facts are appended when inference starts."
            )
            if args.supervisor_profile == "tracked"
            else generic_passage_supervisor_prompt(args.vision_input)
        ),
        "prompts": (
            staged_prompts
            if args.supervisor_profile == "tracked"
            else {
                "camera_only_passage_supervisor": (
                    generic_passage_supervisor_prompt(args.vision_input)
                )
            }
        ),
        "allowed_commands": list(
            GENERIC_COMMANDS
            if args.supervisor_profile == "tracked"
            else VISION_ONLY_TUTORIAL_COMMANDS
        ),
        "capture_fps": args.capture_fps,
        "model_fps": args.model_fps,
        "frames_per_window": args.frames_per_window,
        "model_window_duration_seconds": round(
            args.frames_per_window / args.model_fps,
            6,
        ),
        "sensor_camera": args.sensor_camera,
        "presentation_camera": args.presentation_camera,
        "presentation_route_visualizations": (
            args.presentation_route_visualizations
        ),
        "presentation_dashboard": args.presentation_dashboard,
        "evk_video_width": args.evk_video_width,
        "forklift_release_x": args.forklift_release_x,
        "reroute_confirmations_required": args.reroute_confirmations,
        "reroute_confirmation_streak": 0,
        "request_id": None,
        "inference_started_epoch": None,
        "last_raw_command": None,
        "last_model_output": None,
        "last_model_responses": None,
        "last_command_source": None,
        "last_model_seconds": None,
        "last_host_total_seconds": None,
        "gateway_action": None,
        "gateway_route": None,
        "active_action": None,
        "active_route": None,
        "window_frames": [],
        "response_history": [],
    }

    def publish(**fields: Any) -> None:
        status.update(fields)
        status["updated_epoch"] = time.time()
        status["wall_seconds"] = round(time.monotonic() - wall_started, 6)
        _atomic_write_json(status_path, status)

    def record(event: str, **fields: Any) -> None:
        entry = {
            "event": event,
            "wall_seconds": round(time.monotonic() - wall_started, 6),
            **fields,
        }
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        _emit_json(entry)

    isaac = IsaacLiveController()
    ring: deque[Path] = deque(maxlen=args.frames_per_window)
    state_ring: deque[dict[str, Any]] = deque(maxlen=args.frames_per_window)
    future: Future[dict[str, Any]] | None = None
    pending_request: dict[str, Any] | None = None
    frame_index = 0
    request_index = 0
    last_submitted_frame = -args.frames_per_window
    local_hold = False
    evk_hold_active = False
    route_before_evk_hold = "direct"
    forklift_enabled = args.scenario == "blind_corner"
    expected_destination_route = (
        "north_bypass" if forklift_enabled else "direct"
    )
    scenario_released = args.clear_baseline_responses == 0
    forklift_released = False
    baseline_responses = 0
    baseline_false_interventions = 0
    reroute_confirmation_streak = 0
    response_counts: dict[str, int] = {}
    responses: list[dict[str, Any]] = []
    request_failures: list[dict[str, Any]] = []
    consecutive_failures = 0
    next_request_allowed = 0.0
    abort_reason: str | None = None
    last_tick: dict[str, Any] = {}
    destination_first_frame: int | None = None
    destination_reached = False
    destination_route_matches_expected: bool | None = None
    completion_reason: str | None = None
    robot_pov_at_reroute: Path | None = None
    presentation_frame_states: list[dict[str, Any]] = []
    last_overlay_response_count = 0

    def request_phase() -> str:
        if not scenario_released:
            return "clear_baseline"
        if not forklift_enabled:
            return "clear_control"
        return "hazard" if forklift_released else "approach"

    record("run_started", run_id=run_id, session_dir=str(session_dir))
    publish(events_path=str(events_path))
    inference_client.prepare()
    record(
        "inference_ready",
        backend=args.inference_backend,
        target=inference_target,
        model=inference_model,
    )
    publish(
        state="ready",
        phase="setup",
        inference_ready=True,
        evk_ready=(args.inference_backend == "evk"),
    )
    checkpoint = None if args.preserve_current_stage else args.checkpoint
    scene = isaac.prepare(
        checkpoint=checkpoint,
        robot_scale=args.robot_scale,
        motion_controller=args.motion_controller,
        robot_speed=args.robot_speed,
        scenario=args.scenario,
        configure_blind_corner=not args.use_current_forklift_start,
    )
    record("scene_ready", scene=scene)
    publish(state="ready", phase="scene_ready", scene=scene)
    isaac.set_congestion(False)
    if scenario_released:
        isaac.set_forklift_visible(forklift_enabled)
        isaac.apply(
            action="CONTINUE",
            route="direct",
            request_id="scenario-start",
            latency_seconds=0.0,
            decision_source="manual",
        )
        record(
            "approach_phase_started",
            trigger="no_clear_baseline",
            forklift_release_x=args.forklift_release_x,
            forklift_enabled=forklift_enabled,
        )
        publish(state="capturing", phase=request_phase())
    else:
        isaac.set_forklift_visible(False)
        isaac.apply(
            action="YIELD",
            route="hold",
            request_id="baseline-hold",
            latency_seconds=0.0,
            decision_source="manual",
        )
        record(
            "clear_baseline_started",
            target_responses=args.clear_baseline_responses,
            commandable_actors=list(commandable_actors),
        )
        publish(state="capturing", phase="clear_baseline")
    isaac.play()
    record(
        "timeline_started",
        obstacle="ForkliftBlue",
        released=forklift_released,
    )

    interval = 1.0 / args.capture_fps
    next_capture = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evk-advisory")
    try:
        while time.monotonic() - wall_started < args.max_runtime_seconds:
            now = time.monotonic()
            if now < next_capture:
                time.sleep(min(0.1, next_capture - now))
                continue

            last_tick = isaac.tick()
            sensor_capture_name = (
                f"{run_id}_sensor_frame_{frame_index:04d}.png"
            )
            sensor_captured = isaac.capture(
                sensor_capture_name,
                sensor_camera=args.sensor_camera,
            )
            presentation_camera = (
                args.sensor_camera
                if args.presentation_camera == "sensor"
                else args.presentation_camera
            )
            if (
                presentation_camera == args.sensor_camera
                and not args.presentation_route_visualizations
            ):
                presentation_captured = sensor_captured
            else:
                presentation_capture_name = (
                    f"{run_id}_presentation_frame_{frame_index:04d}.png"
                )
                presentation_captured = isaac.capture(
                    presentation_capture_name,
                    sensor_camera=presentation_camera,
                    include_route_visualizations=(
                        args.presentation_route_visualizations
                    ),
                )
            permanent = frames_dir / f"frame_{frame_index:04d}.png"
            shutil.copy2(presentation_captured, permanent)
            if robot_frames_dir is not None:
                robot_capture_name = (
                    f"{run_id}_robot_pov_frame_{frame_index:04d}.png"
                )
                robot_captured = isaac.capture_robot_front(
                    robot_capture_name
                )
                shutil.copy2(
                    robot_captured,
                    robot_frames_dir / f"frame_{frame_index:04d}.png",
                )
            ring.append(sensor_captured)
            state_ring.append(dict(last_tick))
            record(
                "frame",
                frame_index=frame_index,
                simulation_seconds=last_tick.get("timeline_time_seconds"),
                separation_m=last_tick.get("separation_m"),
                action=last_tick.get("action"),
                route=last_tick.get("route"),
            )
            publish(
                state="inference" if future is not None else "capturing",
                frame_index=frame_index,
                simulation_seconds=last_tick.get("timeline_time_seconds"),
                robot_position=last_tick.get("blue_position"),
                forklift_position=last_tick.get("forklift_position"),
                separation_m=last_tick.get("separation_m"),
                active_action=last_tick.get("action"),
                active_route=last_tick.get("route"),
                navigation_intention=last_tick.get("navigation_intention"),
                forklift_motion_phase=last_tick.get("forklift_motion_phase"),
                robot_yaw_degrees=last_tick.get("robot_yaw_degrees"),
                distance_to_destination_m=last_tick.get(
                    "distance_to_destination_m"
                ),
                destination_reached=last_tick.get("destination_reached"),
                last_captured_frame=str(permanent),
                last_sensor_frame=str(sensor_captured),
            )

            (
                at_physical_destination,
                observed_route_matches_expected,
            ) = _destination_observation(
                scenario_released=scenario_released,
                last_tick=last_tick,
                expected_route=expected_destination_route,
            )
            if at_physical_destination and destination_first_frame is None:
                destination_first_frame = frame_index
                destination_route_matches_expected = (
                    observed_route_matches_expected
                )
                record(
                    "destination_reached",
                    frame_index=frame_index,
                    route=last_tick.get("route"),
                    expected_route=expected_destination_route,
                    route_matches_expected=(
                        destination_route_matches_expected
                    ),
                    position=last_tick.get("blue_position"),
                    settle_frames=args.destination_settle_frames,
                )
                publish(
                    state="destination_reached",
                    phase="arrival",
                    destination_first_frame=destination_first_frame,
                    destination_route_matches_expected=(
                        destination_route_matches_expected
                    ),
                )

            blue_position = last_tick.get("blue_position") or ()
            blue_x = (
                float(blue_position[0])
                if isinstance(blue_position, (list, tuple))
                and len(blue_position) >= 1
                else -999.0
            )
            if (
                forklift_enabled
                and
                scenario_released
                and not forklift_released
                and blue_x >= args.forklift_release_x
            ):
                isaac.set_congestion(True)
                forklift_released = True
                ring.clear()
                state_ring.clear()
                last_submitted_frame = frame_index
                record(
                    "forklift_emergence_started",
                    robot_x=blue_x,
                    trigger_x=args.forklift_release_x,
                    rolling_window_reset=True,
                )
                publish(
                    state="capturing",
                    phase="hazard",
                    window_frames=[],
                    forklift_released=True,
                )

            separation = float(last_tick.get("separation_m", 999.0))
            if (
                scenario_released
                and forklift_released
                and
                separation <= args.safety_hold_distance
                and last_tick.get("route") == "direct"
                and not local_hold
            ):
                isaac.apply(
                    action="YIELD",
                    route="hold",
                    request_id="local-safety-hold",
                    latency_seconds=0.0,
                    decision_source="local_safety",
                )
                local_hold = True
                record(
                    "local_safety_hold",
                    separation_m=separation,
                    threshold_m=args.safety_hold_distance,
                )
                publish(
                    state="local_safety_hold",
                    gateway_action="YIELD",
                    gateway_route="hold",
                    gateway_source="local_safety",
                )

            if future is not None and future.done():
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - persisted run evidence
                    request_id = str(
                        (pending_request or {}).get(
                            "request_id",
                            f"live-{request_index:03d}",
                        )
                    )
                    isaac.request_status(
                        status="failed",
                        request_id=request_id,
                    )
                    consecutive_failures += 1
                    failure = {
                        "request_id": request_id,
                        "error": str(exc),
                        "consecutive_failures": consecutive_failures,
                    }
                    request_failures.append(failure)
                    next_request_allowed = (
                        time.monotonic() + args.request_retry_seconds
                    )
                    record("evk_request_failed", **failure)
                    publish(
                        state="request_failed",
                        request_id=request_id,
                        inference_started_epoch=None,
                        last_error=str(exc),
                        consecutive_failures=consecutive_failures,
                        request_failures=request_failures[-10:],
                    )
                    future = None
                    pending_request = None
                    if consecutive_failures >= args.max_consecutive_failures:
                        abort_reason = (
                            "EVK request failure limit reached: "
                            f"{consecutive_failures} consecutive failures"
                        )
                        record(
                            "evk_failure_limit_reached",
                            abort_reason=abort_reason,
                        )
                        break
                else:
                    consecutive_failures = 0
                    advisory = result["advisory"]
                    request_id = str(
                        (pending_request or {}).get(
                            "request_id",
                            f"live-{request_index:03d}",
                        )
                    )
                    latency = float(result.get("host_total_seconds", 0.0))
                    raw_command = str(result["raw_command"])
                    phase = str(
                        (pending_request or {}).get("phase", request_phase())
                    )
                    if scenario_released:
                        if advisory["action"] == "REROUTE":
                            reroute_confirmation_streak += 1
                        else:
                            reroute_confirmation_streak = 0
                    response_counts[raw_command] = (
                        response_counts.get(raw_command, 0) + 1
                    )
                    response_record = {
                        "request_id": request_id,
                        "phase": phase,
                        "model_output": result.get("model_output"),
                        "model_responses": result.get("model_responses"),
                        "raw_command": raw_command,
                        "command_source": result.get("command_source"),
                        "action": advisory["action"],
                        "route": advisory["route"],
                        "model_seconds": result.get("total_seconds"),
                        "host_total_seconds": latency,
                        "reroute_confirmation_streak": (
                            reroute_confirmation_streak
                        ),
                    }
                    responses.append(response_record)
                    record(
                        "evk_response",
                        request_id=request_id,
                        phase=phase,
                        result=result,
                    )
                    publish(
                        state="response_received",
                        request_id=request_id,
                        inference_started_epoch=None,
                        last_raw_command=raw_command,
                        last_model_output=result.get("model_output"),
                        last_model_responses=result.get("model_responses"),
                        last_command_source=result.get("command_source"),
                        last_model_seconds=result.get("total_seconds"),
                        last_host_total_seconds=latency,
                        reroute_confirmation_streak=(
                            reroute_confirmation_streak
                        ),
                        response_history=responses[-10:],
                    )

                    if not scenario_released:
                        baseline_responses += 1
                        if advisory["action"] != "CONTINUE":
                            baseline_false_interventions += 1
                            record(
                                "baseline_false_intervention",
                                request_id=request_id,
                                raw_command=raw_command,
                            )
                        else:
                            record(
                                "baseline_clear_response",
                                request_id=request_id,
                                raw_command=raw_command,
                            )
                        if baseline_responses >= args.clear_baseline_responses:
                            isaac.set_forklift_visible(forklift_enabled)
                            isaac.apply(
                                action="CONTINUE",
                                route="direct",
                                request_id="scenario-release",
                                latency_seconds=0.0,
                                decision_source="manual",
                            )
                            isaac.set_congestion(False)
                            ring.clear()
                            state_ring.clear()
                            last_submitted_frame = frame_index
                            scenario_released = True
                            reroute_confirmation_streak = 0
                            local_hold = False
                            evk_hold_active = False
                            record(
                                "approach_phase_started",
                                trigger="clear_baseline_complete",
                                baseline_responses=baseline_responses,
                                baseline_false_interventions=(
                                    baseline_false_interventions
                                ),
                                rolling_window_reset=True,
                                forklift_release_x=args.forklift_release_x,
                                forklift_enabled=forklift_enabled,
                            )
                            publish(
                                state="capturing",
                                phase=request_phase(),
                                request_id=None,
                                window_frames=[],
                                forklift_released=False,
                            )
                    elif advisory["action"] == "CONTINUE":
                        if local_hold:
                            record(
                                "evk_continue_overridden",
                                reason="local safety hold remains authoritative",
                            )
                        elif evk_hold_active:
                            if route_before_evk_hold in (
                                "south_bypass",
                                "north_bypass",
                            ):
                                applied = isaac.apply(
                                    action="REROUTE",
                                    route=route_before_evk_hold,
                                    request_id=request_id,
                                    latency_seconds=latency,
                                    decision_source="evk",
                                )
                            else:
                                applied = isaac.apply(
                                    action="CONTINUE",
                                    route="direct",
                                    request_id=request_id,
                                    latency_seconds=latency,
                                    decision_source="evk",
                                )
                            evk_hold_active = False
                            record("evk_hold_released", applied=applied)
                            publish(
                                state="command_applied",
                                gateway_action=applied.get("action"),
                                gateway_route=applied.get("route"),
                                gateway_source="evk",
                            )
                        else:
                            applied = isaac.apply(
                                action="CONTINUE",
                                route="current",
                                request_id=request_id,
                                latency_seconds=latency,
                                decision_source="evk",
                            )
                            record("evk_continue_applied", applied=applied)
                            publish(
                                state="command_applied",
                                gateway_action=applied.get("action"),
                                gateway_route=applied.get("route"),
                                gateway_source="evk",
                            )
                    elif advisory["action"] == "REROUTE":
                        current_route = str(
                            last_tick.get("route", "direct")
                        )
                        if (
                            reroute_confirmation_streak
                            < args.reroute_confirmations
                        ):
                            record(
                                "evk_reroute_pending_confirmation",
                                request_id=request_id,
                                active_route=current_route,
                                requested_route=advisory["route"],
                                confirmation_streak=(
                                    reroute_confirmation_streak
                                ),
                                confirmations_required=(
                                    args.reroute_confirmations
                                ),
                                latency_seconds=latency,
                            )
                            publish(
                                state="reroute_pending_confirmation",
                                gateway_action="CONTINUE",
                                gateway_route=current_route,
                                gateway_source="temporal_consensus",
                            )
                        elif current_route == advisory["route"]:
                            record(
                                "evk_reroute_reaffirmed",
                                request_id=request_id,
                                active_route=current_route,
                                latency_seconds=latency,
                            )
                            publish(
                                state="route_reaffirmed",
                                gateway_action="REROUTE",
                                gateway_route=current_route,
                                gateway_source="evk",
                            )
                        else:
                            applied = isaac.apply(
                                action="REROUTE",
                                route=advisory["route"],
                                request_id=request_id,
                                latency_seconds=latency,
                                decision_source="evk",
                            )
                            local_hold = False
                            evk_hold_active = False
                            record("evk_advisory_applied", applied=applied)
                            if robot_pov_at_reroute is None:
                                try:
                                    captured_robot_pov = (
                                        isaac.capture_robot_front(
                                            f"{run_id}_robot_pov_at_reroute.png"
                                        )
                                    )
                                    robot_pov_at_reroute = (
                                        session_dir
                                        / "robot_pov_at_reroute.png"
                                    )
                                    shutil.copy2(
                                        captured_robot_pov,
                                        robot_pov_at_reroute,
                                    )
                                    record(
                                        "robot_pov_at_reroute_captured",
                                        output_path=str(
                                            robot_pov_at_reroute
                                        ),
                                        robot_position=last_tick.get(
                                            "blue_position"
                                        ),
                                        forklift_position=last_tick.get(
                                            "forklift_position"
                                        ),
                                        simulation_seconds=last_tick.get(
                                            "timeline_time_seconds"
                                        ),
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    record(
                                        "robot_pov_at_reroute_capture_failed",
                                        error=str(exc),
                                    )
                            publish(
                                state="command_applied",
                                gateway_action=applied.get("action"),
                                gateway_route=applied.get("route"),
                                gateway_source="evk",
                            )
                    else:
                        current_route = str(last_tick.get("route", "direct"))
                        if current_route != "hold":
                            route_before_evk_hold = current_route
                        applied = isaac.apply(
                            action=advisory["action"],
                            route="hold",
                            request_id=request_id,
                            latency_seconds=latency,
                            decision_source="evk",
                        )
                        evk_hold_active = True
                        record("evk_hold_applied", applied=applied)
                        publish(
                            state="command_applied",
                            gateway_action=applied.get("action"),
                            gateway_route=applied.get("route"),
                            gateway_source="evk",
                        )
                    future = None
                    pending_request = None

            if (
                future is None
                and len(ring) == args.frames_per_window
                and frame_index > last_submitted_frame
                and time.monotonic() >= next_request_allowed
                and destination_first_frame is None
                and (
                    args.max_responses == 0
                    or len(responses) < args.max_responses
                )
            ):
                request_index += 1
                request_id = f"live-{request_index:03d}"
                snapshot = tuple(ring)
                state_snapshot = tuple(state_ring)
                last_submitted_frame = frame_index
                current_request_phase = request_phase()
                tracked_facts = (
                    _tracked_scene_facts(
                        state_history=state_snapshot,
                        phase=current_request_phase,
                        forklift_released=forklift_released,
                        forklift_enabled=forklift_enabled,
                    )
                    if args.supervisor_profile == "tracked"
                    else None
                )
                isaac.request_status(
                    status="in_flight",
                    request_id=request_id,
                )
                future = executor.submit(
                    _inference_job,
                    client=inference_client,
                    frame_paths=snapshot,
                    request_dir=requests_dir / request_id,
                    model_fps=args.model_fps,
                    video_width=args.evk_video_width,
                    request_id=request_id,
                    commandable_actors=commandable_actors,
                    supervisor_profile=args.supervisor_profile,
                    tracked_scene_facts=tracked_facts,
                    sensor_camera=args.sensor_camera,
                    vision_input=args.vision_input,
                )
                pending_request = {
                    "request_id": request_id,
                    "phase": current_request_phase,
                    "first_frame": frame_index - args.frames_per_window + 1,
                    "last_frame": frame_index,
                    "vision_input": args.vision_input,
                    "tracked_scene_facts": tracked_facts,
                }
                record(
                    "evk_request_started",
                    request_id=request_id,
                    commandable_actors=list(commandable_actors),
                    phase=current_request_phase,
                    first_frame=frame_index - args.frames_per_window + 1,
                    last_frame=frame_index,
                    supervisor_profile=args.supervisor_profile,
                    vision_input=args.vision_input,
                    encoded_fps=args.model_fps,
                    source_frame_count=len(snapshot),
                    encoded_duration_seconds=round(
                        len(snapshot) / args.model_fps,
                        6,
                    ),
                    tracked_scene_facts=tracked_facts,
                )
                publish(
                    state="inference",
                    request_id=request_id,
                    inference_started_epoch=time.time(),
                    window_frames=[str(path) for path in snapshot],
                    tracked_scene_facts=tracked_facts,
                    prompt=(
                        tracked_state_supervisor_prompt(
                            tracked_facts,
                            commandable_actors,
                        )
                        if tracked_facts is not None
                        else status["prompt"]
                    ),
                )

            if args.presentation_dashboard:
                response_number = len(responses)
                latest_response = responses[-1] if responses else {}
                presentation_frame_states.append(
                    {
                        "frame_index": frame_index,
                        "actors": _compact_actor_inventory(
                            latest_response.get("model_responses")
                        ),
                        "passages": _compact_passage_state(
                            latest_response.get("model_responses")
                        ),
                        "adapter_command": latest_response.get("raw_command"),
                        "accepted_route": (
                            status.get("gateway_route")
                            or last_tick.get("route")
                            or "direct"
                        ),
                        "request_id": latest_response.get("request_id"),
                        "latency_seconds": latest_response.get(
                            "host_total_seconds"
                        ),
                        "response_number": response_number,
                        "new_response": (
                            response_number > last_overlay_response_count
                        ),
                    }
                )
                last_overlay_response_count = response_number

            frame_index += 1
            next_capture += interval
            if next_capture < time.monotonic():
                next_capture = time.monotonic() + interval

            if (
                destination_first_frame is not None
                and (frame_index - 1) - destination_first_frame
                >= args.destination_settle_frames
            ):
                destination_reached = True
                completion_reason = "destination_reached"
                record(
                    "destination_recording_complete",
                    first_frame=destination_first_frame,
                    final_frame=frame_index - 1,
                    route=last_tick.get("route"),
                    expected_route=expected_destination_route,
                    route_matches_expected=(
                        destination_route_matches_expected
                    ),
                )
                break

            if args.max_responses and len(responses) >= args.max_responses:
                completion_reason = "response_limit"
                record("response_limit_reached", responses=len(responses))
                break
    finally:
        if not args.leave_playing:
            isaac.stop()
            record("timeline_stopped")
            publish(state="stopped", timeline_playing=False)
        if future is not None and not future.done():
            record("waiting_for_in_flight_evk_request")
            try:
                result = future.result(timeout=args.evk_timeout_seconds + 20.0)
                record(
                    "late_evk_response",
                    request=pending_request,
                    result=result,
                )
            except Exception as exc:  # noqa: BLE001 - persisted run evidence
                record("late_evk_request_failed", error=str(exc))
        executor.shutdown(wait=True, cancel_futures=False)

    if completion_reason is None:
        completion_reason = "aborted" if abort_reason else "runtime_limit"

    media = _encode_presentation(
        frames_dir=frames_dir,
        output_dir=session_dir,
        playback_fps=args.presentation_fps,
        make_gif=args.make_gif,
        robot_frames_dir=robot_frames_dir,
        frame_states=presentation_frame_states,
        response_pulse_seconds=(
            args.presentation_response_pulse_seconds
        ),
        pip_width=args.presentation_pip_width,
    )
    if robot_pov_at_reroute is not None:
        media["robot_pov_at_reroute"] = str(robot_pov_at_reroute)
    summary = {
        "run_id": run_id,
        "session_dir": str(session_dir),
        "frames": frame_index,
        "capture_fps": args.capture_fps,
        "model_fps": args.model_fps,
        "frames_per_window": args.frames_per_window,
        "model_window_duration_seconds": round(
            args.frames_per_window / args.model_fps,
            6,
        ),
        "sensor_camera": args.sensor_camera,
        "presentation_camera": args.presentation_camera,
        "presentation_route_visualizations": (
            args.presentation_route_visualizations
        ),
        "presentation_dashboard": args.presentation_dashboard,
        "presentation_response_pulse_seconds": (
            args.presentation_response_pulse_seconds
        ),
        "presentation_pip_width": args.presentation_pip_width,
        "supervisor_profile": args.supervisor_profile,
        "vision_input": args.vision_input,
        "motion_controller": args.motion_controller,
        "robot_speed_mps": args.robot_speed,
        "reroute_confirmations_required": args.reroute_confirmations,
        "scenario": args.scenario,
        "inference_backend": args.inference_backend,
        "inference_target": inference_target,
        "model": inference_model,
        "presentation_fps": args.presentation_fps,
        "robot_visual_xy_scale": args.robot_scale,
        "last_state": last_tick,
        "local_safety_hold": local_hold,
        "commandable_actors": list(commandable_actors),
        "responses": responses,
        "response_counts": response_counts,
        "request_failures": request_failures,
        "abort_reason": abort_reason,
        "baseline_responses": baseline_responses,
        "baseline_false_interventions": baseline_false_interventions,
        "scenario_released": scenario_released,
        "forklift_released": forklift_released,
        "forklift_enabled": forklift_enabled,
        "forklift_release_x": args.forklift_release_x,
        "expected_destination_route": expected_destination_route,
        "destination_first_frame": destination_first_frame,
        "destination_reached": destination_reached,
        "destination_route_matches_expected": (
            destination_route_matches_expected
        ),
        "completion_reason": completion_reason,
        "media": media,
        "events": str(events_path),
    }
    summary_path = session_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    publish(
        state="failed" if abort_reason else "complete",
        phase="aborted" if abort_reason else "complete",
        inference_started_epoch=None,
        summary_path=str(summary_path),
        media=media,
        response_history=responses[-10:],
        baseline_responses=baseline_responses,
        baseline_false_interventions=baseline_false_interventions,
        destination_reached=destination_reached,
        completion_reason=completion_reason,
        abort_reason=abort_reason,
    )
    _emit_json(summary, pretty=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Live supervisor failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
