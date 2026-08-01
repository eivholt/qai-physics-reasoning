"""Live controller and deterministic animation for the conveyor safety demo."""

from __future__ import annotations

import asyncio
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse

import carb
import carb.input
import numpy as np
import omni.appwindow
import omni.ext
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.menu.utils import (
    MenuItemDescription,
    add_menu_items,
    remove_menu_items,
)
from isaacsim.core.experimental.prims import Articulation
from isaacsim.robot.experimental.wheeled_robots.controllers import (
    AckermannController,
)
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel


WINDOW_TITLE = "Reason2 Conveyor Safety"
SCENE_ROOT = "/World/CodexPoC/ConveyorSafety"
FORKLIFT_ROOT = f"{SCENE_ROOT}/Forklifts"
WORKER_ROOT = f"{SCENE_ROOT}/Workers"
PARCEL_ROOT = f"{SCENE_ROOT}/Conveyor/Parcels"
CARGO_ROOT = f"{SCENE_ROOT}/DynamicCargo"
PRESENTATION_CAMERA_PATH = f"{SCENE_ROOT}/Cameras/Presentation"
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
INFERENCE_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_live_isaac_conveyor_safety.py"
INFERENCE_LOG_ROOT = (
    REPOSITORY_ROOT / "artifacts" / "isaac_sim_conveyor_safety" / "auto_inference"
)
LIVE_STATUS_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "isaac_sim_conveyor_safety"
    / "live_status.json"
)
BACKEND_OVERRIDE_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "isaac_sim_conveyor_safety"
    / "inference_backend.txt"
)

# 1.0 m/s is a realistic indoor warehouse pace and keeps adjacent rolling
# detector frames close enough for the 2B video model to associate the same
# vehicle instead of treating a large displacement as a new actor.
FORKLIFT_MAX_SPEED_MPS = 1.00
FORKLIFT_MAX_STEER_RADIANS = 0.52
# ForkliftC steers at the rear axle. Invert the operator command so a left
# stick/key request turns the vehicle nose left while the rear wheels visibly
# deflect right, matching a real counterbalanced forklift.
FORKLIFT_OPERATOR_STEERING_SIGN = -1.0
FORKLIFT_WHEEL_BASE_M = 1.65
FORKLIFT_TRACK_WIDTH_M = 1.05
FORKLIFT_FRONT_WHEEL_RADIUS_M = 0.325
FORKLIFT_REAR_WHEEL_RADIUS_M = 0.255
FORKLIFT_MAX_ACCELERATION_MPS2 = 1.00
FORKLIFT_SERVICE_BRAKE_MPS2 = 1.50
FORKLIFT_MAX_STEERING_RATE_RAD_S = 1.20
FORKLIFT_LIFT_MIN_M = 0.0
FORKLIFT_LIFT_MAX_M = 2.0
FORKLIFT_LIFT_SPEED_MPS = 0.90
FORKLIFT_LIFT_ACCELERATION_MPS2 = 1.35
FORK_CARGO_FORWARD_OFFSET_M = 1.44
FORK_PALLET_BASE_Z_M = 0.146
FORK_CARTON_BASE_Z_M = 0.300
FORK_PALLET_YAW_OFFSET_DEGREES = 90.0
# PhysX and the explicit cube proxies handle contact. This short OBB sweep is
# only an anti-tunneling fallback for the next controller step; it must not act
# like a second safety zone in front of the visible obstacle.
COLLISION_CONTACT_LOOKAHEAD_SECONDS = 0.05
# The authored scene already has simple hard PhysX boxes on the conveyor and
# rack. Keep the look-ahead implementation for diagnostics, but do not zero
# throttle before contact; PhysX is the sole obstacle stop in the live demo.
SOFTWARE_COLLISION_THROTTLE_GUARD = False
INPUT_DEADZONE = 0.12
FLOOR_X_LIMIT = 9.55
FLOOR_Y_LIMIT = 7.05
FORKLIFT_MOVING_THRESHOLD_MPS = 0.05
# PhysX drive targets persist until they change. Updating them at the display
# rate needlessly wakes sleeping articulations and forces extra simulation-to-
# USD synchronization. Thirty hertz remains responsive for a 1 m/s forklift.
FORKLIFT_CONTROL_UPDATE_HZ = 30.0
# Parcel forces and worker route transforms are low-frequency control fields;
# PhysX and UsdSkel continue to simulate/render between these updates.
PARCEL_FORCE_UPDATE_HZ = 15.0
WORKER_ROUTE_UPDATE_HZ = 20.0
WORKER_RADIUS_M = 0.38
WORKER_WALK_SPEED_MPS = 0.55
WORKER_PUSH_SPEED_MPS = 1.15
WORKER_MAX_OFFSET_M = 1.25
WORKER_TURN_RATE_DEGREES_PER_SECOND = 180.0
WORKER_LOCOMOTION_ACCELERATION_PER_SECOND = 2.8
WORKER_LOCOMOTION_DECELERATION_PER_SECOND = 4.0
WORKER_REPLAN_BLOCK_SECONDS = 0.20
WORKER_ANIMATION_WALK_THRESHOLD = 0.62
WORKER_ANIMATION_IDLE_THRESHOLD = 0.18
# A sparse rectilinear graph keeps actors in the open west-side warehouse
# corridors.  The two spur nodes are the shelf and conveyor work stations.
# Dynamic obstacles remove graph edges at runtime and A* selects another lane.
WORKER_CORRIDOR_NODES = (
    (-3.10, -4.45),
    (-1.25, -4.45),
    (0.60, -4.45),
    (1.80, -4.45),
    (-3.10, -2.40),
    (-1.25, -2.40),
    (0.60, -2.40),
    (1.80, -2.40),
    (-3.10, -0.10),
    (-1.25, -0.10),
    (0.60, -0.10),
    (1.80, -0.10),
    (-3.10, 2.10),
    (-1.25, 2.10),
    (0.60, 2.10),
    (1.80, 2.10),
    (-3.10, 2.78),
    (1.93, 2.78),
)
WORKER_CORRIDOR_EDGES = tuple(
    (row * 4 + column, row * 4 + column + 1)
    for row in range(4)
    for column in range(3)
) + tuple(
    (row * 4 + column, (row + 1) * 4 + column)
    for row in range(3)
    for column in range(4)
) + ((12, 16), (15, 17))
PRESENTATION_CAMERA_DISTANCES_M = (5.5, 10.0, 15.0)
PRESENTATION_CAMERA_DEFAULT_DISTANCE_INDEX = 1
PRESENTATION_CAMERA_TARGET_HEIGHT_M = 0.80
PRESENTATION_CAMERA_FOCAL_LENGTH_MM = 24.0
PRESENTATION_CAMERA_ORBIT_YAW_RATE_RAD_S = 1.80
PRESENTATION_CAMERA_ORBIT_PITCH_RATE_RAD_S = 1.20
PRESENTATION_CAMERA_MIN_PITCH_RAD = math.radians(10.0)
PRESENTATION_CAMERA_MAX_PITCH_RAD = math.radians(65.0)

_HEADER_STYLE = {"font_size": 18, "color": 0xFFF4F4F4}
_SECTION_STYLE = {"font_size": 14, "color": 0xFF63D7FF}
_MONO_STYLE = {"font_size": 14, "color": 0xFFE6E6E6}


def _environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _hidden_process_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if sys.platform == "win32" else 0


def _gamepad_axis(
    input_interface: carb.input.IInput,
    gamepad: carb.input.Gamepad | None,
    positive: carb.input.GamepadInput,
    negative: carb.input.GamepadInput,
) -> float:
    if gamepad is None:
        return 0.0
    return float(input_interface.get_gamepad_value(gamepad, positive)) - float(
        input_interface.get_gamepad_value(gamepad, negative)
    )


def _is_keyboard_down(
    input_interface: carb.input.IInput,
    keyboard: carb.input.InputDevice,
    key: carb.input.KeyboardInput,
) -> bool:
    return (
        input_interface.get_keyboard_button_flags(keyboard, key)
        == carb.input.BUTTON_FLAG_DOWN
    )


def _read_translate(prim: Any) -> Gf.Vec3d:
    value = prim.GetAttribute("xformOp:translate").Get()
    if value is None:
        return Gf.Vec3d(0.0)
    return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))


def _read_yaw_degrees(prim: Any) -> float:
    value = prim.GetAttribute("xformOp:rotateXYZ").Get()
    return float(value[2]) if value is not None else 0.0


class ConveyorSafetyExtension(omni.ext.IExt):
    """Animate the scene and map Xbox/keyboard input onto either forklift."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._timeline = omni.timeline.get_timeline_interface()
        self._active_index = 0
        self._previous_left_shoulder = False
        self._previous_right_shoulder = False
        self._previous_q = False
        self._previous_e = False
        self._previous_y = False
        self._previous_r = False
        self._previous_view_button = False
        self._previous_backend_button = False
        self._worker_push_offsets: dict[int, Gf.Vec2d] = {}
        self._worker_push_velocities: dict[int, Gf.Vec2d] = {}
        self._worker_route_positions: dict[int, Gf.Vec3d] = {}
        self._worker_route_waypoint_indices: dict[int, int] = {}
        self._worker_route_dwell_remaining: dict[int, float] = {}
        self._worker_motion_states: dict[int, bool] = {}
        self._worker_animation_blends: dict[int, float] = {}
        self._worker_locomotion_weights: dict[int, float] = {}
        self._worker_heading_degrees: dict[int, float] = {}
        self._worker_corridor_paths: dict[int, list[Gf.Vec3d]] = {}
        self._worker_corridor_path_indices: dict[int, int] = {}
        self._worker_corridor_goal_indices: dict[int, int] = {}
        self._worker_blocked_seconds: dict[int, float] = {}
        self._worker_replan_counts: dict[int, int] = {}
        self._worker_last_blocked_labels: dict[int, str] = {}
        self._last_forklift_velocity = Gf.Vec2d(0.0, 0.0)
        self._forklift_articulations: dict[int, Articulation] = {}
        self._ackermann_controllers: dict[int, AckermannController] = {}
        self._steering_dof_indices: dict[int, Any] = {}
        self._wheel_dof_indices: dict[int, Any] = {}
        self._lift_dof_indices: dict[int, Any] = {}
        self._lift_targets_m: dict[int, float] = {}
        self._lift_velocities_mps: dict[int, float] = {}
        self._previous_forklift_positions: dict[int, Gf.Vec3d] = {}
        self._last_steering_targets: dict[int, np.ndarray] = {}
        self._last_wheel_targets: dict[int, np.ndarray] = {}
        self._last_lift_targets_sent: dict[int, float] = {}
        self._forklift_control_accumulator = 0.0
        # Start the 15 Hz parcel field one display tick out of phase with the
        # 30 Hz forklift controller. This prevents all expensive PhysX/USD
        # synchronization points from landing on the same rendered frame.
        self._parcel_force_accumulator = 1.0 / 60.0
        self._worker_route_accumulator = 0.0
        self._presentation_orbit_yaw_rad = math.radians(-45.0)
        self._presentation_orbit_pitch_rad = math.radians(28.0)
        self._presentation_camera_distance_index = (
            PRESENTATION_CAMERA_DEFAULT_DISTANCE_INDEX
        )
        self._last_presentation_camera_signature: (
            tuple[float, float, float, float, float, float] | None
        ) = None
        self._last_blocked_obstacle = ""
        self._last_stage_identifier = ""
        self._last_controller_connected: bool | None = None
        self._last_halo_active_index: int | None = None
        self._last_active_index_written: int | None = None
        self._last_clearance_written: float | None = None
        self._last_ground_truth_written = ""
        self._last_drive_metrics: tuple[float, float, float, float] | None = None
        self._cargo_reset_task: asyncio.Task[None] | None = None
        self._last_ui_update = 0.0
        self._labels: dict[str, ui.Label] = {}
        self._backend_button: ui.Button | None = None
        self._trace_images: list[ui.Image] = []
        self._trace_image_paths = ["", ""]
        self._trace_status: dict[str, Any] = {}
        self._last_trace_poll = 0.0
        self._last_trace_mtime_ns = -1
        self._auto_inference_enabled = _environment_flag(
            "QAI_CONVEYOR_AUTO_INFERENCE",
            True,
        )
        backend_override = (
            BACKEND_OVERRIDE_PATH.read_text(encoding="utf-8").strip().lower()
            if BACKEND_OVERRIDE_PATH.is_file()
            else ""
        )
        self._inference_backend = (
            backend_override
            or os.environ.get(
                "QAI_CONVEYOR_INFERENCE_BACKEND",
                "host",
            ).strip().lower()
        )
        if self._inference_backend not in {"host", "evk"}:
            self._inference_backend = "evk"
        self._backend_server_urls = {
            "host": os.environ.get(
                "QAI_CONVEYOR_HOST_SERVER_URL",
                "http://127.0.0.1:18080",
            ),
            "evk": os.environ.get(
                "QAI_CONVEYOR_EVK_SERVER_URL",
                "http://127.0.0.1:18181",
            ),
        }
        self._backend_models = {
            "host": os.environ.get(
                "QAI_CONVEYOR_HOST_MODEL",
                "Cosmos-Reason2-2B-BF16.gguf",
            ),
            "evk": os.environ.get(
                "QAI_CONVEYOR_EVK_MODEL",
                "local/cosmos-reason2-2b:Q4_0",
            ),
        }
        # Preserve generic overrides for the initially selected target.
        # Backend-specific variables configure both sides of the live toggle.
        generic_server_url = os.environ.get("QAI_CONVEYOR_SERVER_URL")
        generic_model = os.environ.get("QAI_CONVEYOR_MODEL")
        if generic_server_url:
            self._backend_server_urls[self._inference_backend] = (
                generic_server_url
            )
        if generic_model:
            self._backend_models[self._inference_backend] = generic_model
        self._inference_server_url = self._backend_server_urls[
            self._inference_backend
        ]
        self._inference_model = self._backend_models[self._inference_backend]
        self._inference_camera = os.environ.get(
            "QAI_CONVEYOR_CAMERA",
            "detector_endline",
        )
        try:
            self._maximum_capture_fps = max(
                0.0,
                min(
                    30.0,
                    float(
                        os.environ.get(
                            "QAI_CONVEYOR_MAX_CAPTURE_FPS",
                            "1.0",
                        )
                    ),
                ),
            )
        except ValueError:
            self._maximum_capture_fps = 1.0
        self._evk_target = os.environ.get(
            "QAI_CONVEYOR_EVK_TARGET",
            "ubuntu@192.168.1.158",
        )
        self._evk_ssh_key = Path(
            os.environ.get(
                "QAI_CONVEYOR_EVK_SSH_KEY",
                str(Path.home() / ".ssh" / "codex_iq9075_ed25519"),
            )
        )
        self._inference_process: subprocess.Popen[Any] | None = None
        self._tunnel_process: subprocess.Popen[Any] | None = None
        self._evk_recycle_process: subprocess.Popen[Any] | None = None
        self._inference_log: Any | None = None
        self._tunnel_log: Any | None = None
        self._evk_recycle_log: Any | None = None
        self._next_inference_start = 0.0
        self._inference_status = (
            "Waiting for Play"
            if self._auto_inference_enabled
            else "Automatic inference disabled"
        )

        self._window = ui.Window(
            WINDOW_TITLE,
            width=620,
            height=760,
            visible=True,
        )
        self._build_ui()
        self._menu_items = [
            MenuItemDescription(
                name=WINDOW_TITLE,
                onclick_fn=self._show_window,
            )
        ]
        add_menu_items(self._menu_items, "Window")
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="qai.conveyor_safety.update",
            )
        )
        carb.log_info(
            f"{WINDOW_TITLE}: focused ForkliftC Ackermann physics"
        )

    def on_shutdown(self) -> None:
        self._stop_owned_processes(stop_recycle=True)
        if (
            self._cargo_reset_task is not None
            and not self._cargo_reset_task.done()
        ):
            self._cargo_reset_task.cancel()
        self._cargo_reset_task = None
        self._forklift_articulations = {}
        self._ackermann_controllers = {}
        self._steering_dof_indices = {}
        self._wheel_dof_indices = {}
        self._lift_dof_indices = {}
        self._lift_targets_m = {}
        self._lift_velocities_mps = {}
        self._previous_forklift_positions = {}
        self._last_steering_targets = {}
        self._last_wheel_targets = {}
        self._last_lift_targets_sent = {}
        self._worker_route_positions = {}
        self._worker_route_waypoint_indices = {}
        self._worker_route_dwell_remaining = {}
        self._worker_motion_states = {}
        self._worker_animation_blends = {}
        self._worker_locomotion_weights = {}
        self._worker_heading_degrees = {}
        self._worker_corridor_paths = {}
        self._worker_corridor_path_indices = {}
        self._worker_corridor_goal_indices = {}
        self._worker_blocked_seconds = {}
        self._worker_replan_counts = {}
        self._worker_last_blocked_labels = {}
        self._last_presentation_camera_signature = None
        if getattr(self, "_menu_items", None):
            remove_menu_items(self._menu_items, "Window")
        self._update_subscription = None
        self._window = None
        self._labels = {}
        self._backend_button = None
        self._trace_images = []
        self._trace_image_paths = ["", ""]
        self._trace_status = {}

    def _show_window(self) -> None:
        if self._window is not None:
            self._window.visible = True
            self._window.focus()

    def _backend_button_text(self) -> str:
        return (
            f"Inference target: {self._inference_backend.upper()}"
            "  ·  click or Xbox X"
        )

    def _refresh_backend_button(self) -> None:
        if self._backend_button is not None:
            self._backend_button.text = self._backend_button_text()

    def _set_inference_backend(
        self,
        backend: str,
        *,
        persist: bool = True,
    ) -> None:
        target = backend.strip().lower()
        if target not in {"host", "evk"}:
            raise ValueError(f"Unsupported inference backend: {backend}")
        if target == self._inference_backend:
            self._refresh_backend_button()
            return

        # The rolling runner owns capture and HTTP requests, not simulation.
        # Replacing it therefore switches targets without pausing the timeline.
        self._stop_owned_processes()
        self._inference_backend = target
        self._inference_server_url = self._backend_server_urls[target]
        self._inference_model = self._backend_models[target]
        self._next_inference_start = 0.0
        self._inference_status = (
            f"Play · switching to {target.upper()} inference"
            if self._timeline.is_playing()
            else f"Ready for Play · {target.upper()} selected"
        )
        if persist:
            try:
                BACKEND_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
                BACKEND_OVERRIDE_PATH.write_text(
                    target + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                carb.log_warn(
                    f"{WINDOW_TITLE}: could not persist backend target: {exc}"
                )
        self._refresh_backend_button()
        if self._timeline.is_playing():
            self._ensure_inference_running()

    def _toggle_inference_backend(self) -> None:
        self._set_inference_backend(
            "evk" if self._inference_backend == "host" else "host"
        )

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=(
                    ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
                ),
                vertical_scrollbar_policy=(
                    ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
                ),
            ):
                with ui.VStack(spacing=4, style={"margin": 8}):
                    ui.Label(WINDOW_TITLE, height=24, style=_HEADER_STYLE)
                    self._labels["connection"] = ui.Label(
                        "Waiting for scene…",
                        height=20,
                        style={"font_size": 13, "color": 0xFF66C2FF},
                    )
                    ui.Label("Controller", height=20, style=_SECTION_STYLE)
                    self._labels["controller"] = ui.Label(
                        "ForkliftC rear-steer Ackermann physics\n"
                        "Left stick: throttle + steer\n"
                        "RT/LT: forward/reverse\n"
                        "LB/RB: switch forklift\n"
                        "D-pad Up/Down: forks\n"
                        "Right stick: follow-camera orbit\n"
                        "View: cycle camera distance\n"
                        "X: toggle HOST / EVK inference\n"
                        "Y: reset fork carton",
                        height=190,
                        word_wrap=True,
                        style=_MONO_STYLE,
                    )
                    ui.Label("Safety state", height=20, style=_SECTION_STYLE)
                    self._labels["safety"] = ui.Label(
                        "No state yet.",
                        height=96,
                        word_wrap=True,
                        style=_MONO_STYLE,
                    )
                    ui.Label(
                        "Inference lifecycle",
                        height=20,
                        style=_SECTION_STYLE,
                    )
                    self._backend_button = ui.Button(
                        self._backend_button_text(),
                        height=30,
                        clicked_fn=self._toggle_inference_backend,
                        tooltip="Switch live Reason2 inference between host GPU and EVK",
                    )
                    self._labels["inference"] = ui.Label(
                        "Waiting for Play",
                        height=54,
                        word_wrap=True,
                        style=_MONO_STYLE,
                    )

                    ui.Label(
                        "Reason2 request trace",
                        height=20,
                        style=_SECTION_STYLE,
                    )
                    self._labels["trace_summary"] = ui.Label(
                        "No inference request yet.",
                        height=96,
                        word_wrap=True,
                        alignment=ui.Alignment.LEFT_TOP,
                        style=_MONO_STYLE,
                    )
                    ui.Label(
                        "Rolling input images · previous / current",
                        height=18,
                        style={"font_size": 12, "color": 0xFFBBBBBB},
                    )
                    with ui.HStack(height=152, spacing=6):
                        for _ in range(2):
                            image = ui.Image(
                                "",
                                width=ui.Fraction(1),
                                height=150,
                                fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                style={
                                    "background_color": 0xFF151515,
                                    "border_color": 0xFF444444,
                                    "border_width": 1,
                                },
                            )
                            self._trace_images.append(image)
                    self._labels["image_paths"] = ui.Label(
                        "No input frames yet.",
                        height=38,
                        word_wrap=True,
                        style={"font_size": 11, "color": 0xFF888888},
                    )

                    with ui.CollapsableFrame(
                        "Exact prompt",
                        collapsed=True,
                        height=0,
                        style={"font_size": 13},
                    ):
                        with ui.ScrollingFrame(
                            height=150,
                            horizontal_scrollbar_policy=(
                                ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
                            ),
                            vertical_scrollbar_policy=(
                                ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
                            ),
                            style={"background_color": 0xFF181818},
                        ):
                            self._labels["prompt"] = ui.Label(
                                "Waiting for inference…",
                                word_wrap=True,
                                alignment=ui.Alignment.LEFT_TOP,
                                style=_MONO_STYLE,
                            )

                    ui.Label("Raw model output", height=20, style=_SECTION_STYLE)
                    self._labels["raw_output"] = ui.Label(
                        "—",
                        height=74,
                        word_wrap=True,
                        alignment=ui.Alignment.LEFT_TOP,
                        style=_MONO_STYLE,
                    )

                    ui.Label("Keyboard fallback", height=20, style=_SECTION_STYLE)
                    ui.Label(
                        "W/S throttle · A/D steer · I/K forks · Q/E select",
                        height=24,
                        style=_MONO_STYLE,
                    )
                    ui.Label(
                        "The tower is model-driven. Ground truth is logged only "
                        "for validation and is never sent to Reason2.",
                        height=42,
                        word_wrap=True,
                        style={"font_size": 12, "color": 0xFFAAAAAA},
                    )

    def _scene(self) -> tuple[Any, Any]:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None, None
        root = stage.GetPrimAtPath(SCENE_ROOT)
        if not root.IsValid():
            return stage, None
        return stage, root

    @staticmethod
    def _enabled_forklift_indices(root: Any) -> tuple[int, ...]:
        attribute = root.GetAttribute("codex:enabledForkliftIndices")
        values = attribute.Get() if attribute.IsValid() else None
        indices = tuple(
            int(value) - 1
            for value in (values or [1])
            if 1 <= int(value) <= 3
        )
        return indices or (0,)

    def _read_inference_trace(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_trace_poll < 0.2:
            self._update_trace_timer()
            return
        self._last_trace_poll = now
        try:
            stat = LIVE_STATUS_PATH.stat()
        except FileNotFoundError:
            self._labels["trace_summary"].text = (
                "Waiting for live_status.json from the Reason2 runner."
            )
            return
        if stat.st_mtime_ns == self._last_trace_mtime_ns:
            self._update_trace_timer()
            return
        try:
            payload = json.loads(LIVE_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._labels["trace_summary"].text = f"Trace read failed: {exc}"
            return
        if not isinstance(payload, dict):
            return
        self._last_trace_mtime_ns = stat.st_mtime_ns
        self._trace_status = payload
        self._render_inference_trace()

    def _trace_elapsed_text(self) -> str:
        started = self._trace_status.get("inference_started_epoch")
        if isinstance(started, (int, float)):
            return f"{max(0.0, time.time() - float(started)):.2f} s running"
        latency = self._trace_status.get("last_inference_seconds")
        if not isinstance(latency, (int, float)):
            response = self._trace_status.get("last_response")
            if isinstance(response, dict):
                latency = response.get("latency_seconds")
        return (
            f"{float(latency):.3f} s"
            if isinstance(latency, (int, float))
            else "—"
        )

    def _update_trace_timer(self) -> None:
        if isinstance(
            self._trace_status.get("inference_started_epoch"),
            (int, float),
        ):
            self._render_trace_summary()

    def _render_trace_summary(self) -> None:
        status = self._trace_status
        response = status.get("last_response")
        last_response = response if isinstance(response, dict) else {}
        stack_signal = (
            status.get("stack_light_status")
            or status.get("accepted_signal")
            or last_response.get("accepted_signal")
            or "—"
        )
        request_id = (
            status.get("current_inference_id")
            or last_response.get("inference_id")
            or "—"
        )
        output_signal = (
            status.get("last_raw_model_signal")
            or last_response.get("pre_gate_signal")
            or last_response.get("primary_model_signal")
            or status.get("last_model_signal")
            or last_response.get("model_signal")
            or "—"
        )
        gated_signal = (
            status.get("last_model_signal")
            or last_response.get("model_signal")
            or "—"
        )
        decision_trace = (
            status.get("last_decision_trace")
            or last_response.get("gate_reason")
        )
        decision_line = (
            str(decision_trace)
            if decision_trace
            else f"Model {output_signal} accepted without visual override."
        )
        self._labels["trace_summary"].text = (
            f"{str(status.get('phase') or status.get('state') or 'idle').upper()}"
            f"  ·  {str(status.get('backend') or '—').upper()}"
            f"  ·  {status.get('model') or '—'}\n"
            f"REQUEST {request_id}  ·  INFERENCE {self._trace_elapsed_text()}\n"
            f"RAW {output_signal}  ·  GATED {gated_signal}"
            f"  ·  STACK LIGHT {stack_signal}\n"
            f"{decision_line}"
        )

    def _render_inference_trace(self) -> None:
        status = self._trace_status
        self._render_trace_summary()
        prompt = status.get("current_prompt") or status.get("prompt")
        self._labels["prompt"].text = str(prompt or "Waiting for inference…")
        response = status.get("last_response")
        last_response = response if isinstance(response, dict) else {}
        raw_output = (
            status.get("last_raw_output")
            or last_response.get("raw_answer")
            or "—"
        )
        self._labels["raw_output"].text = str(raw_output)

        paths_value = status.get("input_image_paths")
        paths = (
            [str(item) for item in paths_value[-2:]]
            if isinstance(paths_value, list)
            else []
        )
        display_paths = paths if len(paths) == 2 else ["", *paths]
        for index, image in enumerate(self._trace_images):
            next_path = (
                display_paths[index] if index < len(display_paths) else ""
            )
            if next_path != self._trace_image_paths[index]:
                image.source_url = next_path
                self._trace_image_paths[index] = next_path
        path_names = [
            Path(path).name
            for path in paths
        ]
        self._labels["image_paths"].text = (
            "PREVIOUS  "
            + (path_names[-2] if len(path_names) == 2 else "—")
            + "\nCURRENT   "
            + (path_names[-1] if path_names else "—")
        )

    def _on_update(self, event: Any) -> None:
        self._read_inference_trace()
        stage, root = self._scene()
        if stage is None or root is None:
            self._labels["connection"].text = "Waiting for conveyor safety scene…"
            return
        stage_identifier = stage.GetRootLayer().identifier
        if stage_identifier != self._last_stage_identifier:
            self._last_stage_identifier = stage_identifier
            self._last_controller_connected = None
            self._last_halo_active_index = None
            self._last_active_index_written = None
            self._last_clearance_written = None
            self._last_ground_truth_written = ""
            self._last_drive_metrics = None
            self._previous_view_button = False
            self._previous_backend_button = False
            self._presentation_camera_distance_index = (
                PRESENTATION_CAMERA_DEFAULT_DISTANCE_INDEX
            )
            self._forklift_articulations = {}
            self._ackermann_controllers = {}
            self._steering_dof_indices = {}
            self._wheel_dof_indices = {}
            self._lift_dof_indices = {}
            self._lift_targets_m = {}
            self._lift_velocities_mps = {}
            self._previous_forklift_positions = {}
            self._last_steering_targets = {}
            self._last_wheel_targets = {}
            self._last_lift_targets_sent = {}
            self._forklift_control_accumulator = 0.0
            self._parcel_force_accumulator = 1.0 / 60.0
            self._worker_route_accumulator = 0.0
            self._last_presentation_camera_signature = None
            self._worker_push_offsets = {}
            self._worker_push_velocities = {}
            self._worker_route_positions = {}
            self._worker_route_waypoint_indices = {}
            self._worker_route_dwell_remaining = {}
            self._worker_motion_states = {}
            self._worker_animation_blends = {}
            self._worker_locomotion_weights = {}
            self._worker_heading_degrees = {}
            self._worker_corridor_paths = {}
            self._worker_corridor_path_indices = {}
            self._worker_corridor_goal_indices = {}
            self._worker_blocked_seconds = {}
            self._worker_replan_counts = {}
            self._worker_last_blocked_labels = {}

        payload = event.payload if isinstance(event.payload, dict) else {}
        dt = max(0.0, min(float(payload.get("dt", 1.0 / 60.0)), 0.1))
        gamepad = self._app_window.get_gamepad(0)
        controller_connected = gamepad is not None
        if controller_connected != self._last_controller_connected:
            root.GetAttribute("codex:controllerConnected").Set(
                controller_connected
            )
            self._last_controller_connected = controller_connected
        enabled_indices = self._enabled_forklift_indices(root)
        if self._active_index not in enabled_indices:
            self._active_index = enabled_indices[0]

        left_pressed = bool(
            gamepad
            and self._input.get_gamepad_value(
                gamepad,
                carb.input.GamepadInput.LEFT_SHOULDER,
            )
            > 0.5
        )
        right_pressed = bool(
            gamepad
            and self._input.get_gamepad_value(
                gamepad,
                carb.input.GamepadInput.RIGHT_SHOULDER,
            )
            > 0.5
        )
        q_pressed = _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.Q,
        )
        e_pressed = _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.E,
        )
        y_pressed = bool(
            gamepad
            and self._input.get_gamepad_value(
                gamepad,
                carb.input.GamepadInput.Y,
            )
            > 0.5
        )
        r_pressed = _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.R,
        )
        view_pressed = bool(
            gamepad
            and self._input.get_gamepad_value(
                gamepad,
                carb.input.GamepadInput.MENU1,
            )
            > 0.5
        )
        backend_pressed = bool(
            gamepad
            and self._input.get_gamepad_value(
                gamepad,
                carb.input.GamepadInput.X,
            )
            > 0.5
        )
        if (
            left_pressed
            and not self._previous_left_shoulder
            or q_pressed
            and not self._previous_q
        ):
            position = enabled_indices.index(self._active_index)
            self._active_index = enabled_indices[
                (position - 1) % len(enabled_indices)
            ]
        if (
            right_pressed
            and not self._previous_right_shoulder
            or e_pressed
            and not self._previous_e
        ):
            position = enabled_indices.index(self._active_index)
            self._active_index = enabled_indices[
                (position + 1) % len(enabled_indices)
            ]
        self._previous_left_shoulder = left_pressed
        self._previous_right_shoulder = right_pressed
        self._previous_q = q_pressed
        self._previous_e = e_pressed
        if self._active_index != self._last_active_index_written:
            root.GetAttribute("codex:activeForklift").Set(self._active_index)
            self._last_active_index_written = self._active_index
        if self._active_index != self._last_halo_active_index:
            self._update_halos(stage)
            self._last_halo_active_index = self._active_index
        if (
            y_pressed
            and not self._previous_y
            or r_pressed
            and not self._previous_r
        ):
            self._reset_active_cargo(stage)
        self._previous_y = y_pressed
        self._previous_r = r_pressed
        if view_pressed and not self._previous_view_button:
            self._cycle_presentation_camera_distance()
        self._previous_view_button = view_pressed
        if backend_pressed and not self._previous_backend_button:
            self._toggle_inference_backend()
        self._previous_backend_button = backend_pressed

        throttle = _gamepad_axis(
            self._input,
            gamepad,
            carb.input.GamepadInput.LEFT_STICK_UP,
            carb.input.GamepadInput.LEFT_STICK_DOWN,
        )
        # Keep the combined left-stick mapping, but also expose conventional
        # vehicle-style Xbox controls. Some Windows controller drivers report
        # the vertical stick direction inconsistently while triggers remain
        # independent, normalized axes.
        throttle += _gamepad_axis(
            self._input,
            gamepad,
            carb.input.GamepadInput.RIGHT_TRIGGER,
            carb.input.GamepadInput.LEFT_TRIGGER,
        )
        steering = _gamepad_axis(
            self._input,
            gamepad,
            carb.input.GamepadInput.LEFT_STICK_RIGHT,
            carb.input.GamepadInput.LEFT_STICK_LEFT,
        )
        lift_input = _gamepad_axis(
            self._input,
            gamepad,
            carb.input.GamepadInput.DPAD_UP,
            carb.input.GamepadInput.DPAD_DOWN,
        )
        orbit_horizontal = _gamepad_axis(
            self._input,
            gamepad,
            carb.input.GamepadInput.RIGHT_STICK_RIGHT,
            carb.input.GamepadInput.RIGHT_STICK_LEFT,
        )
        orbit_vertical = _gamepad_axis(
            self._input,
            gamepad,
            carb.input.GamepadInput.RIGHT_STICK_UP,
            carb.input.GamepadInput.RIGHT_STICK_DOWN,
        )
        if _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.W,
        ):
            throttle += 1.0
        if _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.S,
        ):
            throttle -= 1.0
        if _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.D,
        ):
            steering += 1.0
        if _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.A,
        ):
            steering -= 1.0
        if _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.I,
        ):
            lift_input += 1.0
        if _is_keyboard_down(
            self._input,
            self._keyboard,
            carb.input.KeyboardInput.K,
        ):
            lift_input -= 1.0
        throttle = max(-1.0, min(1.0, throttle))
        steering = FORKLIFT_OPERATOR_STEERING_SIGN * max(
            -1.0,
            min(1.0, steering),
        )
        lift_input = max(-1.0, min(1.0, lift_input))
        if abs(throttle) < INPUT_DEADZONE:
            throttle = 0.0
        if abs(steering) < INPUT_DEADZONE:
            steering = 0.0
        if abs(lift_input) < INPUT_DEADZONE:
            lift_input = 0.0
        if abs(orbit_horizontal) < INPUT_DEADZONE:
            orbit_horizontal = 0.0
        if abs(orbit_vertical) < INPUT_DEADZONE:
            orbit_vertical = 0.0

        self._update_presentation_camera(
            stage,
            orbit_horizontal,
            orbit_vertical,
            dt,
        )

        capture_state = sys.modules.get("_qai_conveyor_capture_state")
        capture_in_progress = bool(
            capture_state is not None
            and getattr(capture_state, "depth", 0) > 0
        )
        if self._timeline.is_playing():
            self._ensure_inference_running()
            # Replicator requests a zero-delta application update for each
            # detector frame. Do not wake articulations or author parcel and
            # worker USD changes inside that already expensive RTX frame.
            if not capture_in_progress:
                self._forklift_control_accumulator += dt
                forklift_control_interval = 1.0 / FORKLIFT_CONTROL_UPDATE_HZ
                if (
                    self._forklift_control_accumulator
                    >= forklift_control_interval
                ):
                    control_dt = min(
                        self._forklift_control_accumulator,
                        0.1,
                    )
                    self._forklift_control_accumulator = 0.0
                    self._drive_active_forklift(
                        stage,
                        throttle,
                        steering,
                        lift_input,
                        control_dt,
                    )
                sim_time = float(self._timeline.get_current_time())
                self._parcel_force_accumulator += dt
                parcel_force_interval = 1.0 / PARCEL_FORCE_UPDATE_HZ
                if self._parcel_force_accumulator >= parcel_force_interval:
                    parcel_dt = min(self._parcel_force_accumulator, 0.1)
                    self._parcel_force_accumulator = 0.0
                    self._animate_parcels(stage, sim_time, parcel_dt)
                self._worker_route_accumulator += dt
                worker_route_interval = 1.0 / WORKER_ROUTE_UPDATE_HZ
                if self._worker_route_accumulator >= worker_route_interval:
                    worker_dt = min(self._worker_route_accumulator, 0.1)
                    self._worker_route_accumulator = 0.0
                    self._animate_workers(stage, sim_time, worker_dt)
        elif self._auto_inference_enabled:
            if (
                self._inference_process is not None
                and self._inference_process.poll() is None
            ):
                self._inference_status = "Paused · runner waiting for Play"
            else:
                self._inference_status = "Waiting for Play"

        min_clearance = self._update_clearance(
            stage,
            root,
            throttle,
        )
        now = time.monotonic()
        if now - self._last_ui_update >= 0.1:
            self._last_ui_update = now
            self._update_ui(root, gamepad, throttle, steering, min_clearance)

    def _process_kwargs(self, log_handle: Any) -> dict[str, Any]:
        return {
            "cwd": str(REPOSITORY_ROOT),
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "creationflags": _hidden_process_flags(),
        }

    def _open_process_log(self, stem: str) -> Any:
        INFERENCE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        path = INFERENCE_LOG_ROOT / f"{stem}.log"
        return path.open("a", encoding="utf-8", buffering=1)

    def _endpoint_is_listening(self) -> bool:
        parsed = urlparse(self._inference_server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=0.05):
                return True
        except OSError:
            return False

    def _ensure_evk_tunnel(self) -> bool:
        if self._endpoint_is_listening():
            return True
        if self._tunnel_process is not None:
            exit_code = self._tunnel_process.poll()
            if exit_code is None:
                self._inference_status = "Play · waiting for EVK tunnel"
                return False
            self._inference_status = f"EVK tunnel exited ({exit_code})"
            self._tunnel_process = None
            if self._tunnel_log is not None:
                self._tunnel_log.close()
                self._tunnel_log = None
        ssh = shutil.which("ssh")
        if ssh is None:
            self._inference_status = "Cannot start EVK tunnel: ssh not found"
            return False
        parsed = urlparse(self._inference_server_url)
        local_port = parsed.port or 18181
        command = [
            ssh,
            "-i",
            str(self._evk_ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-N",
            "-L",
            f"{local_port}:127.0.0.1:18181",
            self._evk_target,
        ]
        self._tunnel_log = self._open_process_log("evk_tunnel")
        try:
            self._tunnel_process = subprocess.Popen(
                command,
                **self._process_kwargs(self._tunnel_log),
            )
        except Exception as exc:
            self._inference_status = f"Cannot start EVK tunnel: {exc}"
            self._tunnel_log.close()
            self._tunnel_log = None
            return False
        self._inference_status = "Play · starting EVK tunnel"
        return False

    def _python_command(self) -> list[str] | None:
        configured = os.environ.get("QAI_CONVEYOR_PYTHON")
        if configured:
            return [configured]
        python = shutil.which("python")
        if python:
            return [python]
        launcher = shutil.which("py")
        if launcher:
            return [launcher, "-3"]
        return None

    def _ensure_inference_running(self) -> None:
        if not self._auto_inference_enabled:
            self._inference_status = "Automatic inference disabled"
            return
        now = time.monotonic()
        if (
            self._inference_backend == "evk"
            and self._evk_recycle_process is not None
        ):
            recycle_exit = self._evk_recycle_process.poll()
            if recycle_exit is None:
                self._inference_status = "Play · recycling EVK model service"
                return
            self._evk_recycle_process = None
            if self._evk_recycle_log is not None:
                self._evk_recycle_log.close()
                self._evk_recycle_log = None
            if recycle_exit != 0:
                self._inference_status = (
                    f"EVK recycle failed ({recycle_exit}); retrying"
                )
                self._next_inference_start = now + 5.0
                return
            self._inference_status = "Play · EVK model service recycled"
        if self._inference_process is not None:
            exit_code = self._inference_process.poll()
            if exit_code is None:
                self._inference_status = (
                    f"Play · {self._inference_backend.upper()} inference running"
                )
                return
            self._inference_status = f"Inference exited ({exit_code}); retrying"
            self._inference_process = None
            if self._inference_log is not None:
                self._inference_log.close()
                self._inference_log = None
            if self._inference_backend == "evk":
                self._start_evk_service_recycle(exit_code)
                self._next_inference_start = now + 2.0
                return
            self._next_inference_start = now + 2.0
        if now < self._next_inference_start:
            return
        if self._inference_backend == "evk" and not self._ensure_evk_tunnel():
            self._next_inference_start = now + 0.5
            return
        python_command = self._python_command()
        if python_command is None:
            self._inference_status = "Cannot start inference: Python not found"
            self._next_inference_start = now + 5.0
            return
        command = [
            *python_command,
            str(INFERENCE_SCRIPT),
            "--server-url",
            self._inference_server_url,
            "--model",
            self._inference_model,
            "--backend-label",
            self._inference_backend,
            "--perception-mode",
            "direct",
            "--camera",
            self._inference_camera,
            "--max-responses",
            "35" if self._inference_backend == "evk" else "1000000",
            "--max-runtime-seconds",
            "86400",
            "--timeout-seconds",
            "15" if self._inference_backend == "evk" else "120",
            "--play-mode-only",
            "--idle-poll-seconds",
            "0.2",
            "--capture-width",
            "384" if self._inference_backend == "evk" else "512",
            "--capture-height",
            "216" if self._inference_backend == "evk" else "288",
            "--capture-render-steps",
            "1",
            "--capture-rt-subframes",
            "1",
            "--maximum-capture-fps",
            str(self._maximum_capture_fps),
        ]
        self._inference_log = self._open_process_log("reason2_runner")
        try:
            self._inference_process = subprocess.Popen(
                command,
                **self._process_kwargs(self._inference_log),
            )
        except Exception as exc:
            self._inference_status = f"Cannot start inference: {exc}"
            self._inference_log.close()
            self._inference_log = None
            self._next_inference_start = now + 5.0
            return
        self._inference_status = (
            f"Play · starting {self._inference_backend.upper()} inference"
        )

    def _start_evk_service_recycle(self, runner_exit_code: int) -> None:
        if (
            self._evk_recycle_process is not None
            and self._evk_recycle_process.poll() is None
        ):
            return
        ssh = shutil.which("ssh")
        if ssh is None:
            self._inference_status = "Cannot recycle EVK: ssh not found"
            return
        command = [
            ssh,
            "-i",
            str(self._evk_ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            self._evk_target,
            (
                "pkill -TERM -x geniex-grammar 2>/dev/null || true; "
                "sleep 2; "
                "pkill -KILL -x geniex-grammar 2>/dev/null || true; "
                "bash /tmp/start_evk_geniex_isaac_service.sh"
            ),
        ]
        self._evk_recycle_log = self._open_process_log("evk_recycle")
        try:
            self._evk_recycle_process = subprocess.Popen(
                command,
                **self._process_kwargs(self._evk_recycle_log),
            )
        except Exception as exc:
            self._inference_status = f"Cannot recycle EVK: {exc}"
            self._evk_recycle_log.close()
            self._evk_recycle_log = None
            self._evk_recycle_process = None
            return
        self._inference_status = (
            "Play · recycling EVK after runner exit "
            f"({runner_exit_code})"
        )

    def _stop_owned_processes(self, *, stop_recycle: bool = False) -> None:
        processes = [
            getattr(self, "_inference_process", None),
            getattr(self, "_tunnel_process", None),
        ]
        recycle_process = getattr(self, "_evk_recycle_process", None)
        if stop_recycle:
            processes.append(recycle_process)
        for process in processes:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        self._inference_process = None
        self._tunnel_process = None
        if stop_recycle or (
            recycle_process is not None
            and recycle_process.poll() is not None
        ):
            self._evk_recycle_process = None
            if self._evk_recycle_log is not None:
                try:
                    self._evk_recycle_log.close()
                except Exception:
                    pass
                self._evk_recycle_log = None
        for log_name in ("_inference_log", "_tunnel_log"):
            handle = getattr(self, log_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, log_name, None)

    def _drive_active_forklift(
        self,
        stage: Any,
        throttle: float,
        steering: float,
        lift_input: float,
        dt: float,
    ) -> None:
        root = stage.GetPrimAtPath(SCENE_ROOT)
        if not root.IsValid():
            return
        self._ensure_forklift_drivers(stage)
        active = stage.GetPrimAtPath(
            f"{FORKLIFT_ROOT}/Forklift{self._active_index + 1}"
        )
        if not active.IsValid() or self._active_index not in self._forklift_articulations:
            return

        active_position, active_yaw = self._forklift_pose(active)
        commanded_speed = throttle * FORKLIFT_MAX_SPEED_MPS
        steering_angle = steering * FORKLIFT_MAX_STEER_RADIANS
        # Ramp hydraulic lift velocity instead of stopping a 0.9 m/s fork
        # carriage in one controller tick. The pallet and carton remain free
        # rigid bodies, so abrupt lift stops otherwise transfer their upward
        # momentum into an unrealistic launch.
        lift_dt = min(max(dt, 0.0), 0.05)
        for index in self._forklift_articulations:
            desired_lift_velocity = (
                lift_input * FORKLIFT_LIFT_SPEED_MPS
                if index == self._active_index
                else 0.0
            )
            lift_velocity = self._lift_velocities_mps.get(index, 0.0)
            maximum_lift_delta = (
                FORKLIFT_LIFT_ACCELERATION_MPS2 * lift_dt
            )
            lift_velocity += max(
                -maximum_lift_delta,
                min(
                    maximum_lift_delta,
                    desired_lift_velocity - lift_velocity,
                ),
            )
            if (
                abs(desired_lift_velocity) < 1.0e-5
                and abs(lift_velocity) < maximum_lift_delta + 1.0e-5
            ):
                lift_velocity = 0.0
            lift_target = max(
                FORKLIFT_LIFT_MIN_M,
                min(
                    FORKLIFT_LIFT_MAX_M,
                    self._lift_targets_m.get(index, 0.0)
                    + lift_velocity * lift_dt,
                ),
            )
            if (
                lift_target <= FORKLIFT_LIFT_MIN_M
                and lift_velocity < 0.0
                or lift_target >= FORKLIFT_LIFT_MAX_M
                and lift_velocity > 0.0
            ):
                lift_velocity = 0.0
            self._lift_velocities_mps[index] = lift_velocity
            self._lift_targets_m[index] = lift_target
        active_lift_target = self._lift_targets_m.get(
            self._active_index,
            0.0,
        )
        blocked_by = ""
        if (
            SOFTWARE_COLLISION_THROTTLE_GUARD
            and abs(commanded_speed) > 1e-4
        ):
            active_controller = self._ackermann_controllers[
                self._active_index
            ]
            yaw_radians = math.radians(active_yaw)
            lookahead = COLLISION_CONTACT_LOOKAHEAD_SECONDS
            distance = math.copysign(
                max(
                    abs(commanded_speed) * lookahead,
                    0.02,
                ),
                commanded_speed,
            )
            yaw_rate = (
                commanded_speed
                / FORKLIFT_WHEEL_BASE_M
                * math.tan(steering_angle)
            )
            projected_position = Gf.Vec3d(
                active_position[0] + math.cos(yaw_radians) * distance,
                active_position[1] + math.sin(yaw_radians) * distance,
                active_position[2],
            )
            blocked_by = self._pose_collision(
                root,
                projected_position,
                active_yaw + math.degrees(yaw_rate * lookahead),
            )
            if blocked_by:
                commanded_speed = 0.0

        for index, articulation in self._forklift_articulations.items():
            controller = self._ackermann_controllers[index]
            speed = commanded_speed if index == self._active_index else 0.0
            steer = steering_angle if index == self._active_index else 0.0
            acceleration = (
                FORKLIFT_SERVICE_BRAKE_MPS2
                if blocked_by or index != self._active_index
                else FORKLIFT_MAX_ACCELERATION_MPS2
            )
            joint_positions, joint_velocities = controller.forward(
                np.array(
                    [
                        steer,
                        FORKLIFT_MAX_STEERING_RATE_RAD_S,
                        speed,
                        acceleration,
                        min(max(dt, 1e-4), 0.05),
                    ]
                )
            )
            position_target_parts = []
            position_index_parts = []
            if joint_positions is not None and joint_velocities is not None:
                previous_steering = self._last_steering_targets.get(index)
                if (
                    previous_steering is None
                    or not np.allclose(
                        joint_positions,
                        previous_steering,
                        rtol=0.0,
                        atol=1.0e-5,
                    )
                ):
                    position_target_parts.append(
                        np.asarray(joint_positions, dtype=np.float32)
                    )
                    position_index_parts.append(
                        np.asarray(
                            self._steering_dof_indices[index],
                            dtype=np.int32,
                        )
                    )
                    self._last_steering_targets[index] = np.asarray(
                        joint_positions,
                        dtype=np.float32,
                    ).copy()
                previous_wheels = self._last_wheel_targets.get(index)
                if (
                    previous_wheels is None
                    or not np.allclose(
                        joint_velocities,
                        previous_wheels,
                        rtol=0.0,
                        atol=1.0e-4,
                    )
                ):
                    articulation.set_dof_velocity_targets(
                        joint_velocities,
                        dof_indices=self._wheel_dof_indices[index],
                    )
                    self._last_wheel_targets[index] = np.asarray(
                        joint_velocities,
                        dtype=np.float32,
                    ).copy()
            lift_target = self._lift_targets_m.get(index, 0.0)
            previous_lift = self._last_lift_targets_sent.get(index)
            if previous_lift is None or abs(previous_lift - lift_target) > 1.0e-5:
                position_target_parts.append(
                    np.asarray([lift_target], dtype=np.float32)
                )
                position_index_parts.append(
                    np.asarray(
                        self._lift_dof_indices[index],
                        dtype=np.int32,
                    )
                )
                self._last_lift_targets_sent[index] = lift_target
            if position_target_parts:
                articulation.set_dof_position_targets(
                    np.concatenate(position_target_parts),
                    dof_indices=np.concatenate(position_index_parts),
                )

            forklift = stage.GetPrimAtPath(
                f"{FORKLIFT_ROOT}/Forklift{index + 1}"
            )
            fork_height_attr = forklift.GetAttribute("codex:forkHeightM")
            authored_fork_height = float(fork_height_attr.Get() or 0.0)
            if abs(authored_fork_height - lift_target) > 1e-4:
                fork_height_attr.Set(lift_target)
            position, _ = self._forklift_pose(forklift)
            previous = self._previous_forklift_positions.get(index)
            if previous is not None and dt > 1e-6:
                velocity = Gf.Vec2d(
                    (position[0] - previous[0]) / dt,
                    (position[1] - previous[1]) / dt,
                )
                if index == self._active_index:
                    self._last_forklift_velocity = velocity
            self._previous_forklift_positions[index] = Gf.Vec3d(position)

        actual_speed = float(
            self._ackermann_controllers[
                self._active_index
            ].prev_linear_velocity
        )
        drive_metrics = (
            commanded_speed,
            actual_speed,
            math.degrees(steering_angle),
            active_lift_target,
        )
        if (
            self._last_drive_metrics is None
            or any(
                abs(current - previous) > 1e-4
                for current, previous in zip(
                    drive_metrics,
                    self._last_drive_metrics,
                )
            )
        ):
            root.GetAttribute("codex:commandedSpeedMps").Set(commanded_speed)
            root.GetAttribute("codex:actualSpeedMps").Set(actual_speed)
            root.GetAttribute("codex:steeringAngleDegrees").Set(
                math.degrees(steering_angle)
            )
            root.GetAttribute("codex:forkHeightM").Set(active_lift_target)
            self._last_drive_metrics = drive_metrics

        collision_attr = root.GetAttribute("codex:lastCollision")
        block_count_attr = root.GetAttribute("codex:collisionBlockCount")
        if blocked_by:
            collision_message = (
                f"Forklift {self._active_index + 1}: {blocked_by}"
            )
            if collision_attr:
                collision_attr.Set(collision_message)
            if (
                collision_message != self._last_blocked_obstacle
                and block_count_attr
            ):
                block_count_attr.Set(int(block_count_attr.Get() or 0) + 1)
            self._last_blocked_obstacle = collision_message
        else:
            if collision_attr and (
                self._last_blocked_obstacle
                or collision_attr.Get()
            ):
                collision_attr.Set("")
            self._last_blocked_obstacle = ""

    def _ensure_forklift_drivers(self, stage: Any) -> None:
        root = stage.GetPrimAtPath(SCENE_ROOT)
        if not root.IsValid():
            return
        for index in self._enabled_forklift_indices(root):
            if index in self._forklift_articulations:
                continue
            path = f"{FORKLIFT_ROOT}/Forklift{index + 1}"
            prim = stage.GetPrimAtPath(path)
            body = stage.GetPrimAtPath(f"{path}/body")
            if not prim.IsValid() or not body.IsValid():
                continue
            articulation = Articulation(path)
            self._forklift_articulations[index] = articulation
            self._steering_dof_indices[index] = articulation.get_dof_indices(
                ["left_rotator_joint", "right_rotator_joint"]
            )
            self._wheel_dof_indices[index] = articulation.get_dof_indices(
                [
                    "left_front_wheel_joint",
                    "right_front_wheel_joint",
                    "left_back_wheel_joint",
                    "right_back_wheel_joint",
                ]
            )
            self._lift_dof_indices[index] = articulation.get_dof_indices(
                ["lift_joint"]
            )
            self._lift_targets_m[index] = max(
                FORKLIFT_LIFT_MIN_M,
                min(
                    FORKLIFT_LIFT_MAX_M,
                    float(
                        prim.GetAttribute("codex:forkHeightM").Get()
                        or 0.0
                    ),
                ),
            )
            self._lift_velocities_mps[index] = 0.0
            self._ackermann_controllers[index] = AckermannController(
                wheel_base=FORKLIFT_WHEEL_BASE_M,
                track_width=FORKLIFT_TRACK_WIDTH_M,
                front_wheel_radius=FORKLIFT_FRONT_WHEEL_RADIUS_M,
                back_wheel_radius=FORKLIFT_REAR_WHEEL_RADIUS_M,
                invert_steering=True,
                max_wheel_velocity=8.0,
                max_wheel_rotation_angle=FORKLIFT_MAX_STEER_RADIANS,
                max_acceleration=FORKLIFT_SERVICE_BRAKE_MPS2,
                max_steering_angle_velocity=(
                    FORKLIFT_MAX_STEERING_RATE_RAD_S
                ),
            )

    def _forklift_pose(self, prim: Any) -> tuple[Gf.Vec3d, float]:
        body = prim.GetStage().GetPrimAtPath(f"{prim.GetPath()}/body")
        if not body.IsValid():
            return Gf.Vec3d(0.0), 0.0
        matrix = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
            0.0
        )
        position = matrix.ExtractTranslation()
        forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        yaw_degrees = math.degrees(
            math.atan2(float(forward[1]), float(forward[0]))
        )
        return Gf.Vec3d(position), yaw_degrees

    def _update_presentation_camera(
        self,
        stage: Any,
        orbit_horizontal: float,
        orbit_vertical: float,
        dt: float,
    ) -> None:
        viewport = get_active_viewport()
        if (
            viewport is None
            or str(viewport.camera_path) != PRESENTATION_CAMERA_PATH
        ):
            return
        forklift = stage.GetPrimAtPath(
            f"{FORKLIFT_ROOT}/Forklift{self._active_index + 1}"
        )
        camera_prim = stage.GetPrimAtPath(PRESENTATION_CAMERA_PATH)
        if not forklift.IsValid() or not camera_prim.IsValid():
            return

        self._presentation_orbit_yaw_rad -= (
            orbit_horizontal
            * PRESENTATION_CAMERA_ORBIT_YAW_RATE_RAD_S
            * dt
        )
        self._presentation_orbit_pitch_rad = max(
            PRESENTATION_CAMERA_MIN_PITCH_RAD,
            min(
                PRESENTATION_CAMERA_MAX_PITCH_RAD,
                self._presentation_orbit_pitch_rad
                + orbit_vertical
                * PRESENTATION_CAMERA_ORBIT_PITCH_RATE_RAD_S
                * dt,
            ),
        )
        position, _ = self._forklift_pose(forklift)
        target = Gf.Vec3d(
            float(position[0]),
            float(position[1]),
            float(position[2]) + PRESENTATION_CAMERA_TARGET_HEIGHT_M,
        )
        camera_distance = PRESENTATION_CAMERA_DISTANCES_M[
            self._presentation_camera_distance_index
        ]
        signature = (
            round(float(target[0]), 4),
            round(float(target[1]), 4),
            round(float(target[2]), 4),
            round(self._presentation_orbit_yaw_rad, 5),
            round(self._presentation_orbit_pitch_rad, 5),
            round(camera_distance, 4),
        )
        if signature == self._last_presentation_camera_signature:
            return

        horizontal_distance = camera_distance * math.cos(
            self._presentation_orbit_pitch_rad
        )
        eye = Gf.Vec3d(
            float(target[0])
            + horizontal_distance
            * math.cos(self._presentation_orbit_yaw_rad),
            float(target[1])
            + horizontal_distance
            * math.sin(self._presentation_orbit_yaw_rad),
            float(target[2])
            + camera_distance * math.sin(self._presentation_orbit_pitch_rad),
        )
        matrix = Gf.Matrix4d(1.0)
        matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
        camera_xform = UsdGeom.Xformable(camera_prim)
        camera_xform.ClearXformOpOrder()
        camera_xform.AddTransformOp().Set(matrix.GetInverse())
        camera = UsdGeom.Camera(camera_prim)
        if (
            abs(
                float(camera.GetFocalLengthAttr().Get() or 0.0)
                - PRESENTATION_CAMERA_FOCAL_LENGTH_MM
            )
            > 1e-4
        ):
            camera.GetFocalLengthAttr().Set(
                PRESENTATION_CAMERA_FOCAL_LENGTH_MM
            )
        root = stage.GetPrimAtPath(SCENE_ROOT)
        if root.IsValid():
            distance_attribute = root.GetAttribute(
                "codex:presentationCameraDistanceM"
            )
            if distance_attribute.IsValid():
                distance_attribute.Set(float(camera_distance))
            view_attribute = root.GetAttribute(
                "codex:presentationCameraViewIndex"
            )
            if view_attribute.IsValid():
                view_attribute.Set(
                    self._presentation_camera_distance_index + 1
                )
        self._last_presentation_camera_signature = signature

    def _cycle_presentation_camera_distance(self) -> float:
        self._presentation_camera_distance_index = (
            self._presentation_camera_distance_index + 1
        ) % len(PRESENTATION_CAMERA_DISTANCES_M)
        self._last_presentation_camera_signature = None
        return PRESENTATION_CAMERA_DISTANCES_M[
            self._presentation_camera_distance_index
        ]

    def _obb_overlaps_aabb(
        self,
        position: Gf.Vec3d,
        yaw_degrees: float,
        half_length: float,
        half_width: float,
        center_x: float,
        center_y: float,
        half_x: float,
        half_y: float,
    ) -> bool:
        yaw = math.radians(yaw_degrees)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        delta = (float(position[0]) - center_x, float(position[1]) - center_y)
        for axis_x, axis_y in (
            forward,
            right,
            (1.0, 0.0),
            (0.0, 1.0),
        ):
            separation = abs(delta[0] * axis_x + delta[1] * axis_y)
            vehicle_radius = (
                half_length
                * abs(forward[0] * axis_x + forward[1] * axis_y)
                + half_width
                * abs(right[0] * axis_x + right[1] * axis_y)
            )
            obstacle_radius = (
                half_x * abs(axis_x) + half_y * abs(axis_y)
            )
            if separation >= vehicle_radius + obstacle_radius - 1e-5:
                return False
        return True

    def _conveyor_obstacle_aabbs(
        self,
        root: Any,
    ) -> list[tuple[str, float, float, float, float]]:
        def root_float(name: str, fallback: float) -> float:
            attribute = root.GetAttribute(name)
            value = attribute.Get() if attribute.IsValid() else None
            return fallback if value is None else float(value)

        layout_attr = root.GetAttribute("codex:conveyorLayout")
        layout = (
            str(layout_attr.Get() or "")
            if layout_attr.IsValid()
            else ""
        )
        if layout != "oval":
            return [
                (
                    "conveyor",
                    root_float("codex:conveyorColliderCenterXM", 4.7014),
                    root_float("codex:conveyorColliderCenterYM", -1.3594),
                    root_float("codex:conveyorColliderHalfWidthM", 0.5255),
                    root_float("codex:conveyorColliderHalfLengthM", 2.6494),
                )
            ]

        center_x = root_float("codex:beltCenterXM", 4.42)
        center_y = root_float("codex:beltCenterYM", -0.70)
        radius = root_float("codex:conveyorOvalRadiusM", 1.50)
        straight_half = root_float(
            "codex:conveyorOvalStraightHalfLengthM",
            1.00,
        )
        collider_half_width = root_float(
            "codex:conveyorColliderHalfWidthM",
            0.50,
        )
        segment_count_attr = root.GetAttribute(
            "codex:conveyorCurveSegmentCount"
        )
        segment_count = int(
            segment_count_attr.Get() or 12
        ) if segment_count_attr.IsValid() else 12
        segment_count = max(6, segment_count)
        segment_half_length = (
            math.pi * radius / segment_count * 0.59
        )
        obstacles = [
            (
                "conveyor",
                center_x - radius,
                center_y,
                collider_half_width,
                straight_half + 0.08,
            ),
            (
                "conveyor",
                center_x + radius,
                center_y,
                collider_half_width,
                straight_half + 0.08,
            ),
        ]
        for end_sign in (1.0, -1.0):
            for segment_index in range(segment_count):
                theta = (
                    segment_index + 0.5
                ) * math.pi / segment_count
                segment_x = center_x + radius * math.cos(theta)
                segment_y = center_y + end_sign * (
                    straight_half + radius * math.sin(theta)
                )
                yaw = theta if end_sign > 0.0 else math.pi - theta
                half_x = (
                    collider_half_width * abs(math.cos(yaw))
                    + segment_half_length * abs(math.sin(yaw))
                )
                half_y = (
                    collider_half_width * abs(math.sin(yaw))
                    + segment_half_length * abs(math.cos(yaw))
                )
                obstacles.append(
                    (
                        "conveyor",
                        segment_x,
                        segment_y,
                        half_x,
                        half_y,
                    )
                )
        return obstacles

    def _pose_collision(
        self,
        root: Any,
        position: Gf.Vec3d,
        yaw_degrees: float,
    ) -> str:
        half_length = float(
            root.GetAttribute("codex:forkliftHalfLengthM").Get()
        )
        half_width = float(
            root.GetAttribute("codex:forkliftHalfWidthM").Get()
        )
        yaw = math.radians(yaw_degrees)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        world_half_x = (
            half_length * abs(forward[0]) + half_width * abs(right[0])
        )
        world_half_y = (
            half_length * abs(forward[1]) + half_width * abs(right[1])
        )
        if (
            abs(float(position[0])) + world_half_x > FLOOR_X_LIMIT
            or abs(float(position[1])) + world_half_y > FLOOR_Y_LIMIT
        ):
            return "factory wall"

        # Read the same simple proxy-box envelopes authored into the scene.
        # Policy-zone dimensions remain separate and never block motion.
        def root_float(name: str, fallback: float) -> float:
            attribute = root.GetAttribute(name)
            value = attribute.Get() if attribute.IsValid() else None
            return fallback if value is None else float(value)

        obstacles = self._conveyor_obstacle_aabbs(root)
        obstacles.append(
            (
                "west shelving",
                root_float("codex:westRackCenterXM", -4.20),
                0.0,
                root_float("codex:rackHalfDepthM", 0.54),
                root_float("codex:rackHalfLengthM", 5.40),
            ),
        )
        table_center_attr = root.GetAttribute("codex:packingTableCenterM")
        table_half_attr = root.GetAttribute(
            "codex:packingTableHalfExtentsM"
        )
        if table_center_attr.IsValid() and table_half_attr.IsValid():
            table_center = table_center_attr.Get()
            table_half = table_half_attr.Get()
            if table_center is not None and table_half is not None:
                obstacles.append(
                    (
                        "packing table",
                        float(table_center[0]),
                        float(table_center[1]),
                        float(table_half[0]),
                        float(table_half[1]),
                    )
                )
        for name, center_x, center_y, half_x, half_y in obstacles:
            if self._obb_overlaps_aabb(
                position,
                yaw_degrees,
                half_length,
                half_width,
                center_x,
                center_y,
                half_x,
                half_y,
            ):
                return name
        return ""

    def _reset_active_cargo(self, stage: Any) -> None:
        del stage
        if (
            self._cargo_reset_task is not None
            and not self._cargo_reset_task.done()
        ):
            return
        self._cargo_reset_task = asyncio.ensure_future(
            self._reset_active_cargo_async(self._active_index)
        )

    async def _reset_active_cargo_async(self, active_index: int) -> None:
        was_playing = self._timeline.is_playing()
        try:
            if was_playing:
                self._timeline.stop()
                for _ in range(3):
                    await omni.kit.app.get_app().next_update_async()
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            forklift = stage.GetPrimAtPath(
                f"{FORKLIFT_ROOT}/Forklift{active_index + 1}"
            )
            pallet = stage.GetPrimAtPath(
                f"{CARGO_ROOT}/Forklift{active_index + 1}Pallet"
            )
            carton = stage.GetPrimAtPath(
                f"{CARGO_ROOT}/Forklift{active_index + 1}Carton"
            )
            if (
                not forklift.IsValid()
                or not pallet.IsValid()
                or not carton.IsValid()
            ):
                return
            forklift_position, yaw_degrees = self._forklift_pose(forklift)
            yaw = math.radians(yaw_degrees)
            lift_height = float(
                forklift.GetAttribute("codex:forkHeightM").Get() or 0.0
            )
            cargo_x = (
                forklift_position[0]
                + math.cos(yaw) * FORK_CARGO_FORWARD_OFFSET_M
            )
            cargo_y = (
                forklift_position[1]
                + math.sin(yaw) * FORK_CARGO_FORWARD_OFFSET_M
            )
            for cargo, base_height, yaw_offset in (
                (
                    pallet,
                    FORK_PALLET_BASE_Z_M,
                    FORK_PALLET_YAW_OFFSET_DEGREES,
                ),
                (carton, FORK_CARTON_BASE_Z_M, 0.0),
            ):
                cargo_position = Gf.Vec3d(
                    cargo_x,
                    cargo_y,
                    base_height + lift_height,
                )
                api = UsdGeom.XformCommonAPI(cargo)
                api.SetTranslate(cargo_position)
                api.SetRotate(
                    Gf.Vec3f(0.0, 0.0, yaw_degrees + yaw_offset),
                    UsdGeom.XformCommonAPI.RotationOrderXYZ,
                )
                rigid_body = UsdPhysics.RigidBodyAPI(cargo)
                rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0.0))
                rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0))
        finally:
            if was_playing:
                self._timeline.play()

    def _update_halos(self, stage: Any) -> None:
        root = stage.GetPrimAtPath(SCENE_ROOT)
        if not root.IsValid():
            return
        for index in self._enabled_forklift_indices(root):
            halo = stage.GetPrimAtPath(
                f"{FORKLIFT_ROOT}/Forklift{index + 1}/body/ActiveHalo"
            )
            if not halo.IsValid():
                continue
            active = index == self._active_index
            halo.GetAttribute("codex:active").Set(active)
            nominal = (
                Gf.Vec3f(0.12, 0.72, 1.0),
                Gf.Vec3f(0.42, 1.0, 0.36),
                Gf.Vec3f(0.95, 0.28, 0.82),
            )[index]
            halo.GetAttribute("primvars:displayColor").Set(
                [nominal if active else nominal * 0.18]
            )

    def _animate_parcels(
        self,
        stage: Any,
        sim_time: float,
        dt: float,
    ) -> None:
        del sim_time
        root = stage.GetPrimAtPath(SCENE_ROOT)
        if not root.IsValid():
            return

        def root_float(name: str, fallback: float) -> float:
            attribute = root.GetAttribute(name)
            value = attribute.Get() if attribute.IsValid() else None
            return fallback if value is None else float(value)

        enabled_attr = root.GetAttribute("codex:dynamicBeltParcelsEnabled")
        if enabled_attr.IsValid() and not bool(enabled_attr.Get()):
            return
        parcel_count = int(
            root_float("codex:parcelCount", 8.0)
        )
        center_x = root_float("codex:beltCenterXM", 4.42)
        center_y = root_float("codex:beltCenterYM", -0.70)
        radius = root_float("codex:conveyorOvalRadiusM", 1.50)
        straight_half = root_float(
            "codex:conveyorOvalStraightHalfLengthM",
            1.00,
        )
        track_half_width = root_float(
            "codex:conveyorTrackHalfWidthM",
            0.55,
        )
        surface_height = root_float(
            "codex:conveyorSurfaceHeightM",
            0.76,
        )
        belt_speed = root_float("codex:conveyorVelocityMps", 0.55)
        north_y = center_y + straight_half
        south_y = center_y - straight_half

        for index in range(parcel_count):
            parcel = stage.GetPrimAtPath(f"{PARCEL_ROOT}/Parcel{index + 1}")
            if not parcel.IsValid() or not parcel.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                continue
            position = _read_translate(parcel)
            x = float(position[0])
            y = float(position[1])
            if south_y <= y <= north_y:
                lane_sign = -1.0 if x < center_x else 1.0
                closest_x = center_x + lane_sign * radius
                closest_y = y
                tangent_x = 0.0
                tangent_y = 1.0 if lane_sign < 0.0 else -1.0
            elif y > north_y:
                angle = math.atan2(y - north_y, x - center_x)
                angle = max(0.0, min(math.pi, angle))
                closest_x = center_x + radius * math.cos(angle)
                closest_y = north_y + radius * math.sin(angle)
                tangent_x = math.sin(angle)
                tangent_y = -math.cos(angle)
            else:
                angle = math.atan2(y - south_y, x - center_x)
                angle = max(-math.pi, min(0.0, angle))
                closest_x = center_x + radius * math.cos(angle)
                closest_y = south_y + radius * math.sin(angle)
                tangent_x = math.sin(angle)
                tangent_y = -math.cos(angle)

            offset_x = closest_x - x
            offset_y = closest_y - y
            lateral_distance = math.hypot(offset_x, offset_y)
            dimensions_attr = parcel.GetAttribute("codex:dimensionsM")
            dimensions = (
                dimensions_attr.Get()
                if dimensions_attr.IsValid()
                else None
            )
            parcel_height = (
                float(dimensions[2]) if dimensions is not None else 0.30
            )
            parcel_bottom = float(position[2]) - 0.5 * parcel_height
            force_api = (
                PhysxSchema.PhysxForceAPI(parcel)
                if parcel.HasAPI(PhysxSchema.PhysxForceAPI)
                else PhysxSchema.PhysxForceAPI.Apply(parcel)
            )
            force_attr = force_api.GetForceAttr()
            if not force_attr.IsValid():
                force_attr = force_api.CreateForceAttr()
            enabled_force_attr = force_api.GetForceEnabledAttr()
            if not enabled_force_attr.IsValid():
                enabled_force_attr = force_api.CreateForceEnabledAttr()
            if not force_api.GetModeAttr().IsValid():
                force_api.CreateModeAttr().Set("force")
            if not force_api.GetWorldFrameEnabledAttr().IsValid():
                force_api.CreateWorldFrameEnabledAttr().Set(True)
            # Stop applying the belt field at the edge. A fork can therefore
            # overcome belt friction, push the parcel clear, and let gravity
            # carry it onto the colliding warehouse floor.
            if (
                lateral_distance
                > track_half_width + 0.14
                or abs(parcel_bottom - surface_height) > 0.16
            ):
                force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                enabled_force_attr.Set(False)
                continue

            velocity_attr = UsdPhysics.RigidBodyAPI(
                parcel
            ).GetVelocityAttr()
            velocity = velocity_attr.Get()
            if velocity is None:
                velocity = Gf.Vec3f(0.0, 0.0, 0.0)
            correction_gain = 1.35
            correction_x = max(
                -0.30,
                min(0.30, offset_x * correction_gain),
            )
            correction_y = max(
                -0.30,
                min(0.30, offset_y * correction_gain),
            )
            # Do not cancel a meaningful outward impulse near the edge. The
            # belt keeps supplying tangential motion while the forklift's
            # lateral push carries the box beyond the field and onto the floor.
            if lateral_distance > 1.0e-4:
                outward_x = -offset_x / lateral_distance
                outward_y = -offset_y / lateral_distance
                outward_speed = max(
                    0.0,
                    float(velocity[0]) * outward_x
                    + float(velocity[1]) * outward_y,
                )
            else:
                outward_x = 0.0
                outward_y = 0.0
                outward_speed = 0.0
            preserve_outward_speed = (
                outward_speed
                if lateral_distance > track_half_width * 0.55
                else 0.0
            )
            target_x = tangent_x * belt_speed + correction_x
            target_y = tangent_y * belt_speed + correction_y
            target_x += outward_x * preserve_outward_speed
            target_y += outward_y * preserve_outward_speed
            response_hz = 10.0
            acceleration_x = (
                target_x - float(velocity[0])
            ) * response_hz
            acceleration_y = (
                target_y - float(velocity[1])
            ) * response_hz
            acceleration_magnitude = math.hypot(
                acceleration_x,
                acceleration_y,
            )
            maximum_acceleration = 9.0
            if acceleration_magnitude > maximum_acceleration:
                acceleration_scale = (
                    maximum_acceleration / acceleration_magnitude
                )
                acceleration_x *= acceleration_scale
                acceleration_y *= acceleration_scale
            mass_attr = parcel.GetAttribute("codex:massKg")
            mass_kg = (
                float(mass_attr.Get())
                if mass_attr.IsValid() and mass_attr.Get() is not None
                else 3.0
            )
            force_attr.Set(
                Gf.Vec3f(
                    acceleration_x * mass_kg,
                    acceleration_y * mass_kg,
                    0.0,
                )
            )
            enabled_force_attr.Set(True)

    def _worker_contact_normal(
        self,
        root: Any,
        forklift: Any,
        worker_position: Gf.Vec3d,
    ) -> tuple[Gf.Vec2d, float] | None:
        forklift_position, yaw_degrees = self._forklift_pose(forklift)
        yaw = math.radians(yaw_degrees)
        forward = Gf.Vec2d(math.cos(yaw), math.sin(yaw))
        right = Gf.Vec2d(-math.sin(yaw), math.cos(yaw))
        relative = Gf.Vec2d(
            worker_position[0] - forklift_position[0],
            worker_position[1] - forklift_position[1],
        )
        local_forward = Gf.Dot(relative, forward)
        local_right = Gf.Dot(relative, right)
        half_length = float(
            root.GetAttribute("codex:forkliftHalfLengthM").Get()
        )
        half_width = float(
            root.GetAttribute("codex:forkliftHalfWidthM").Get()
        )
        nearest_forward = max(
            -half_length,
            min(half_length, local_forward),
        )
        nearest_right = max(-half_width, min(half_width, local_right))
        delta_forward = local_forward - nearest_forward
        delta_right = local_right - nearest_right
        outside_distance = math.hypot(delta_forward, delta_right)
        if outside_distance >= WORKER_RADIUS_M:
            return None
        if outside_distance > 1e-5:
            normal = (
                forward * (delta_forward / outside_distance)
                + right * (delta_right / outside_distance)
            )
            return normal, WORKER_RADIUS_M - outside_distance

        forward_exit = half_length - abs(local_forward)
        right_exit = half_width - abs(local_right)
        if forward_exit < right_exit:
            sign = 1.0 if local_forward >= 0.0 else -1.0
            return forward * sign, WORKER_RADIUS_M + forward_exit
        sign = 1.0 if local_right >= 0.0 else -1.0
        return right * sign, WORKER_RADIUS_M + right_exit

    def _worker_obstacles(
        self,
        stage: Any,
        root: Any,
    ) -> tuple[
        list[tuple[str, float, float, float, float]],
        tuple[float, float, float, float],
    ]:
        def root_float(name: str, fallback: float) -> float:
            attribute = root.GetAttribute(name)
            value = attribute.Get() if attribute.IsValid() else None
            return fallback if value is None else float(value)

        floor_center_x = root_float("codex:floorCenterXM", 0.585)
        floor_center_y = root_float("codex:floorCenterYM", -0.725)
        floor_half_width = root_float("codex:floorHalfWidthM", 5.915)
        floor_half_depth = root_float("codex:floorHalfDepthM", 4.325)
        obstacles = self._conveyor_obstacle_aabbs(root)
        obstacles.extend([
            (
                "west shelving",
                root_float("codex:westRackCenterXM", -4.20),
                0.0,
                root_float("codex:rackHalfDepthM", 0.54),
                root_float("codex:rackHalfLengthM", 3.40),
            ),
            (
                "north wall",
                floor_center_x,
                root_float("codex:northWallCenterYM", 3.52),
                floor_half_width,
                root_float("codex:northWallHalfDepthM", 0.08),
            ),
            (
                "west wall",
                root_float("codex:westWallCenterXM", -5.25),
                floor_center_y,
                root_float("codex:westWallHalfWidthM", 0.08),
                floor_half_depth,
            ),
        ])
        table_center_attr = root.GetAttribute("codex:packingTableCenterM")
        table_half_attr = root.GetAttribute(
            "codex:packingTableHalfExtentsM"
        )
        if table_center_attr.IsValid() and table_half_attr.IsValid():
            table_center = table_center_attr.Get()
            table_half = table_half_attr.Get()
            if table_center is not None and table_half is not None:
                obstacles.append(
                    (
                        "packing table",
                        float(table_center[0]),
                        float(table_center[1]),
                        float(table_half[0]),
                        float(table_half[1]),
                    )
                )

        dynamic_cargo_attr = root.GetAttribute("codex:dynamicCargoEnabled")
        dynamic_cargo_enabled = bool(
            dynamic_cargo_attr.IsValid() and dynamic_cargo_attr.Get()
        )
        if dynamic_cargo_enabled:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_],
                False,
            )
            for index in self._enabled_forklift_indices(root):
                for cargo_kind in ("Pallet", "Carton"):
                    collision = stage.GetPrimAtPath(
                        f"{CARGO_ROOT}/Forklift{index + 1}"
                        f"{cargo_kind}/Collision"
                    )
                    if not collision.IsValid():
                        continue
                    world_range = bbox_cache.ComputeWorldBound(
                        collision
                    ).ComputeAlignedRange()
                    if world_range.IsEmpty():
                        continue
                    minimum = world_range.GetMin()
                    maximum = world_range.GetMax()
                    if float(maximum[2]) < 0.0 or float(minimum[2]) > 1.9:
                        continue
                    obstacles.append(
                        (
                            f"forklift {index + 1} loose "
                            f"{cargo_kind.lower()}",
                            0.5 * (float(minimum[0]) + float(maximum[0])),
                            0.5 * (float(minimum[1]) + float(maximum[1])),
                            0.5 * (float(maximum[0]) - float(minimum[0])),
                            0.5 * (float(maximum[1]) - float(minimum[1])),
                        )
                    )

        # Treat both articulated vehicles as dynamic navigation obstacles.
        # The existing push response remains active if a forklift reaches a
        # worker faster than the 20 Hz planner can move them out of its path.
        forklift_half_length = root_float(
            "codex:forkliftHalfLengthM",
            1.87,
        )
        forklift_half_width = root_float(
            "codex:forkliftHalfWidthM",
            0.69,
        )
        for index in self._enabled_forklift_indices(root):
            forklift = stage.GetPrimAtPath(
                f"{FORKLIFT_ROOT}/Forklift{index + 1}"
            )
            if not forklift.IsValid():
                continue
            position, yaw_degrees = self._forklift_pose(forklift)
            yaw = math.radians(yaw_degrees)
            half_x = (
                forklift_half_length * abs(math.cos(yaw))
                + forklift_half_width * abs(math.sin(yaw))
            )
            half_y = (
                forklift_half_length * abs(math.sin(yaw))
                + forklift_half_width * abs(math.cos(yaw))
            )
            obstacles.append(
                (
                    f"forklift {index + 1}",
                    float(position[0]),
                    float(position[1]),
                    half_x,
                    half_y,
                )
            )

        floor_bounds = (
            floor_center_x - floor_half_width,
            floor_center_x + floor_half_width,
            floor_center_y - floor_half_depth,
            floor_center_y + floor_half_depth,
        )
        return obstacles, floor_bounds

    def _worker_obstacle_at(
        self,
        position: Gf.Vec3d,
        radius: float,
        obstacles: list[tuple[str, float, float, float, float]],
        floor_bounds: tuple[float, float, float, float],
    ) -> str:
        x = float(position[0])
        y = float(position[1])
        floor_min_x, floor_max_x, floor_min_y, floor_max_y = floor_bounds
        if (
            x - radius < floor_min_x
            or x + radius > floor_max_x
            or y - radius < floor_min_y
            or y + radius > floor_max_y
        ):
            return "floor edge"
        collision_radius = max(0.0, radius - 1e-4)
        for name, center_x, center_y, half_x, half_y in obstacles:
            nearest_x = max(center_x - half_x, min(center_x + half_x, x))
            nearest_y = max(center_y - half_y, min(center_y + half_y, y))
            delta_x = x - nearest_x
            delta_y = y - nearest_y
            if (
                delta_x * delta_x + delta_y * delta_y
                < collision_radius * collision_radius
            ):
                return name
        return ""

    def _resolve_worker_motion(
        self,
        current: Gf.Vec3d,
        candidate: Gf.Vec3d,
        radius: float,
        obstacles: list[tuple[str, float, float, float, float]],
        floor_bounds: tuple[float, float, float, float],
    ) -> tuple[Gf.Vec3d, str]:
        blocked_by = self._worker_obstacle_at(
            candidate,
            radius,
            obstacles,
            floor_bounds,
        )
        if not blocked_by:
            return candidate, ""

        # Resolve axes independently so a pushed worker can slide along a
        # shelf or wall but can never be stepped through it.
        resolved = Gf.Vec3d(current)
        for axis in (0, 1):
            axis_candidate = Gf.Vec3d(resolved)
            axis_candidate[axis] = candidate[axis]
            axis_obstacle = self._worker_obstacle_at(
                axis_candidate,
                radius,
                obstacles,
                floor_bounds,
            )
            if not axis_obstacle:
                resolved = axis_candidate
            else:
                blocked_by = axis_obstacle
        return resolved, blocked_by

    def _worker_segment_obstacle(
        self,
        start: Gf.Vec3d,
        end: Gf.Vec3d,
        radius: float,
        obstacles: list[tuple[str, float, float, float, float]],
        floor_bounds: tuple[float, float, float, float],
    ) -> str:
        """Return the first inflated obstacle intersected by a path edge."""
        floor_min_x, floor_max_x, floor_min_y, floor_max_y = floor_bounds
        for point in (start, end):
            if (
                float(point[0]) - radius < floor_min_x
                or float(point[0]) + radius > floor_max_x
                or float(point[1]) - radius < floor_min_y
                or float(point[1]) + radius > floor_max_y
            ):
                return "floor edge"

        start_x = float(start[0])
        start_y = float(start[1])
        end_x = float(end[0])
        end_y = float(end[1])
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        for name, center_x, center_y, half_x, half_y in obstacles:
            minimum_x = center_x - half_x - radius
            maximum_x = center_x + half_x + radius
            minimum_y = center_y - half_y - radius
            maximum_y = center_y + half_y + radius
            start_inside = (
                minimum_x <= start_x <= maximum_x
                and minimum_y <= start_y <= maximum_y
            )
            end_inside = (
                minimum_x <= end_x <= maximum_x
                and minimum_y <= end_y <= maximum_y
            )
            # A forklift may physically push a worker into its inflated
            # navigation envelope. Permit only an edge that exits that same
            # envelope; otherwise A* could never recover a route.
            if start_inside and not end_inside:
                continue
            entry = 0.0
            exit = 1.0
            intersects = True
            for origin, delta, minimum, maximum in (
                (start_x, delta_x, minimum_x, maximum_x),
                (start_y, delta_y, minimum_y, maximum_y),
            ):
                if abs(delta) <= 1.0e-8:
                    if origin < minimum or origin > maximum:
                        intersects = False
                        break
                    continue
                axis_entry = (minimum - origin) / delta
                axis_exit = (maximum - origin) / delta
                if axis_entry > axis_exit:
                    axis_entry, axis_exit = axis_exit, axis_entry
                entry = max(entry, axis_entry)
                exit = min(exit, axis_exit)
                if entry > exit:
                    intersects = False
                    break
            if intersects:
                return name
        return ""

    def _plan_worker_corridor(
        self,
        start: Gf.Vec3d,
        goal: Gf.Vec3d,
        radius: float,
        obstacles: list[tuple[str, float, float, float, float]],
        floor_bounds: tuple[float, float, float, float],
    ) -> list[Gf.Vec3d]:
        """Run A* over the sparse warehouse corridor graph."""
        points = [Gf.Vec3d(x, y, 0.0) for x, y in WORKER_CORRIDOR_NODES]
        start_index = len(points)
        goal_index = start_index + 1
        points.extend((Gf.Vec3d(start), Gf.Vec3d(goal)))
        adjacency: dict[int, list[tuple[int, float]]] = {
            index: [] for index in range(len(points))
        }

        def add_edge(first: int, second: int) -> bool:
            if self._worker_segment_obstacle(
                points[first],
                points[second],
                radius,
                obstacles,
                floor_bounds,
            ):
                return False
            distance = Gf.Vec2d(
                float(points[first][0] - points[second][0]),
                float(points[first][1] - points[second][1]),
            ).GetLength()
            adjacency[first].append((second, distance))
            adjacency[second].append((first, distance))
            return True

        for first, second in WORKER_CORRIDOR_EDGES:
            add_edge(first, second)

        add_edge(start_index, goal_index)
        corridor_count = len(WORKER_CORRIDOR_NODES)
        for endpoint_index in (start_index, goal_index):
            nearest = sorted(
                range(corridor_count),
                key=lambda node_index: Gf.Vec2d(
                    float(
                        points[endpoint_index][0]
                        - points[node_index][0]
                    ),
                    float(
                        points[endpoint_index][1]
                        - points[node_index][1]
                    ),
                ).GetLength(),
            )
            connected = 0
            for node_index in nearest:
                distance = Gf.Vec2d(
                    float(
                        points[endpoint_index][0]
                        - points[node_index][0]
                    ),
                    float(
                        points[endpoint_index][1]
                        - points[node_index][1]
                    ),
                ).GetLength()
                if distance > 3.0 and connected:
                    break
                if add_edge(endpoint_index, node_index):
                    connected += 1
                    if connected >= 4:
                        break

        def heuristic(node_index: int) -> float:
            return Gf.Vec2d(
                float(points[node_index][0] - goal[0]),
                float(points[node_index][1] - goal[1]),
            ).GetLength()

        queue: list[tuple[float, float, int]] = [
            (heuristic(start_index), 0.0, start_index)
        ]
        costs = {start_index: 0.0}
        previous: dict[int, int] = {}
        while queue:
            _, current_cost, current_index = heapq.heappop(queue)
            if current_cost > costs.get(current_index, math.inf) + 1.0e-8:
                continue
            if current_index == goal_index:
                break
            for neighbor_index, edge_cost in adjacency[current_index]:
                candidate_cost = current_cost + edge_cost
                if candidate_cost >= costs.get(neighbor_index, math.inf):
                    continue
                costs[neighbor_index] = candidate_cost
                previous[neighbor_index] = current_index
                heapq.heappush(
                    queue,
                    (
                        candidate_cost + heuristic(neighbor_index),
                        candidate_cost,
                        neighbor_index,
                    ),
                )
        if goal_index not in costs:
            return []
        path_indices = [goal_index]
        while path_indices[-1] != start_index:
            path_indices.append(previous[path_indices[-1]])
        path_indices.reverse()
        return [Gf.Vec3d(points[index]) for index in path_indices[1:]]

    @staticmethod
    def _smooth_worker_heading(
        current_degrees: float,
        target_degrees: float,
        dt: float,
    ) -> float:
        delta = (
            (target_degrees - current_degrees + 180.0) % 360.0
        ) - 180.0
        maximum_step = WORKER_TURN_RATE_DEGREES_PER_SECOND * max(dt, 0.0)
        delta = max(-maximum_step, min(maximum_step, delta))
        return (current_degrees + delta + 180.0) % 360.0 - 180.0

    def _write_worker_navigation_state(
        self,
        worker: Any,
        index: int,
        blocked_by: str,
    ) -> None:
        if self._worker_last_blocked_labels.get(index) != blocked_by:
            blocked_attr = worker.GetAttribute("codex:navigationBlocked")
            blocked_by_attr = worker.GetAttribute("codex:navigationBlockedBy")
            if blocked_attr.IsValid():
                blocked_attr.Set(bool(blocked_by))
            if blocked_by_attr.IsValid():
                blocked_by_attr.Set(blocked_by)
            self._worker_last_blocked_labels[index] = blocked_by
        replan_attr = worker.GetAttribute("codex:navigationReplanCount")
        replan_count = self._worker_replan_counts.get(index, 0)
        if (
            replan_attr.IsValid()
            and int(replan_attr.Get() or 0) != replan_count
        ):
            replan_attr.Set(replan_count)

    def _set_worker_motion_animation(
        self,
        stage: Any,
        worker: Any,
        index: int,
        moving: bool,
        dt: float,
    ) -> None:
        blend = self._worker_animation_blends.get(
            index,
            1.0 if moving else 0.0,
        )
        target_blend = 1.0 if moving else 0.0
        blend_rate = 4.0 if moving else 3.0
        blend += max(
            -blend_rate * dt,
            min(blend_rate * dt, target_blend - blend),
        )
        blend = max(0.0, min(1.0, blend))
        self._worker_animation_blends[index] = blend
        current_state = self._worker_motion_states.get(index)
        next_state = current_state
        if current_state is None:
            next_state = moving
        elif not current_state and blend >= WORKER_ANIMATION_WALK_THRESHOLD:
            next_state = True
        elif current_state and blend <= WORKER_ANIMATION_IDLE_THRESHOLD:
            next_state = False
        if next_state is None or current_state == next_state:
            return
        animation_attribute = (
            "codex:walkAnimation" if next_state else "codex:idleAnimation"
        )
        animation_path = worker.GetAttribute(animation_attribute).Get()
        skel_root_path = worker.GetAttribute("codex:skelRootPath").Get()
        if not animation_path or not skel_root_path:
            return
        skel_root = stage.GetPrimAtPath(str(skel_root_path))
        if not skel_root.IsValid():
            return
        UsdSkel.BindingAPI.Apply(
            skel_root
        ).CreateAnimationSourceRel().SetTargets(
            [Sdf.Path(str(animation_path))]
        )
        for skeleton_prim in Usd.PrimRange(skel_root):
            if not skeleton_prim.IsA(UsdSkel.Skeleton):
                continue
            UsdSkel.BindingAPI.Apply(
                skeleton_prim
            ).CreateAnimationSourceRel().SetTargets(
                [Sdf.Path(str(animation_path))]
            )
        self._worker_motion_states[index] = next_state

    def _animate_workers(
        self,
        stage: Any,
        sim_time: float,
        dt: float,
    ) -> None:
        del sim_time
        root = stage.GetPrimAtPath(SCENE_ROOT)
        active_forklift = stage.GetPrimAtPath(
            f"{FORKLIFT_ROOT}/Forklift{self._active_index + 1}"
        )
        push_enabled_attr = (
            root.GetAttribute("codex:workerPushEnabled")
            if root.IsValid()
            else None
        )
        push_enabled = bool(
            push_enabled_attr and push_enabled_attr.Get()
        )
        worker_count_attr = root.GetAttribute("codex:workerCount")
        worker_count = int(
            worker_count_attr.Get() or 2
        ) if worker_count_attr.IsValid() else 2
        radius_attr = root.GetAttribute("codex:workerRadiusM")
        worker_radius = float(
            radius_attr.Get() or WORKER_RADIUS_M
        ) if radius_attr.IsValid() else WORKER_RADIUS_M
        speed_attr = root.GetAttribute("codex:workerWalkSpeedMps")
        walk_speed = float(
            speed_attr.Get() or WORKER_WALK_SPEED_MPS
        ) if speed_attr.IsValid() else WORKER_WALK_SPEED_MPS
        shared_obstacles, floor_bounds = self._worker_obstacles(stage, root)
        for index in range(worker_count):
            worker = stage.GetPrimAtPath(f"{WORKER_ROOT}/Worker{index + 1}")
            if not worker.IsValid():
                continue
            obstacles = list(shared_obstacles)
            for other_index in range(worker_count):
                if other_index == index:
                    continue
                other_worker = stage.GetPrimAtPath(
                    f"{WORKER_ROOT}/Worker{other_index + 1}"
                )
                if not other_worker.IsValid():
                    continue
                other_position = _read_translate(other_worker)
                obstacles.append(
                    (
                        f"worker {other_index + 1}",
                        float(other_position[0]),
                        float(other_position[1]),
                        worker_radius * 0.55,
                        worker_radius * 0.55,
                    )
                )
            route_values = worker.GetAttribute(
                "codex:routeWaypoints"
            ).Get()
            if not route_values or len(route_values) < 2:
                continue
            route_points = [
                Gf.Vec3d(float(value[0]), float(value[1]), 0.0)
                for value in route_values
            ]
            dwell_values = list(
                worker.GetAttribute("codex:routeDwellSeconds").Get() or []
            )
            facing_values = list(
                worker.GetAttribute(
                    "codex:routeFacingYawDegrees"
                ).Get()
                or []
            )
            while len(dwell_values) < len(route_points):
                dwell_values.append(0.0)
            while len(facing_values) < len(route_points):
                facing_values.append(1000.0)

            current_position = _read_translate(worker)
            if index not in self._worker_route_positions:
                # Extension reloads can happen mid-route. Adopt the visible
                # position instead of snapping the character back to waypoint
                # zero, then continue toward the nearest authored route goal.
                initial_base = Gf.Vec3d(
                    float(current_position[0]),
                    float(current_position[1]),
                    0.0,
                )
                nearest_route_index = min(
                    range(len(route_points)),
                    key=lambda route_index: Gf.Vec2d(
                        float(
                            route_points[route_index][0]
                            - initial_base[0]
                        ),
                        float(
                            route_points[route_index][1]
                            - initial_base[1]
                        ),
                    ).GetLength(),
                )
                self._worker_route_positions[index] = initial_base
                self._worker_route_waypoint_indices[index] = (
                    nearest_route_index
                )
                self._worker_route_dwell_remaining[index] = 0.0
                self._worker_push_offsets[index] = Gf.Vec2d(0.0)
                self._worker_heading_degrees[index] = _read_yaw_degrees(
                    worker
                )
                self._worker_locomotion_weights[index] = 0.0
                self._worker_animation_blends[index] = 0.0
                self._worker_corridor_paths[index] = []
                self._worker_corridor_path_indices[index] = 0
                self._worker_corridor_goal_indices[index] = -1
                self._worker_blocked_seconds[index] = 0.0
                self._worker_replan_counts[index] = 0

            base_position = Gf.Vec3d(
                self._worker_route_positions[index]
            )
            target_index = self._worker_route_waypoint_indices[index]
            dwell_remaining = self._worker_route_dwell_remaining[index]
            original_target_index = target_index
            original_dwell_remaining = dwell_remaining
            original_path_index = self._worker_corridor_path_indices.get(
                index,
                0,
            )
            proposed_base_position = Gf.Vec3d(base_position)
            proposed_target_index = target_index
            proposed_dwell_remaining = dwell_remaining
            proposed_path_index = original_path_index
            desired_heading: float | None = None
            route_blocked_by = ""
            route_motion_requested = False

            if dwell_remaining > 0.0:
                facing_yaw = float(facing_values[target_index])
                if -360.0 <= facing_yaw <= 360.0:
                    desired_heading = facing_yaw
                proposed_dwell_remaining = max(
                    0.0,
                    dwell_remaining - dt,
                )
                if proposed_dwell_remaining <= 1.0e-5:
                    proposed_target_index = (
                        target_index + 1
                    ) % len(route_points)
                    self._worker_corridor_goal_indices[index] = -1
            else:
                goal_position = route_points[target_index]
                corridor_path = self._worker_corridor_paths.get(index, [])
                corridor_goal_index = self._worker_corridor_goal_indices.get(
                    index,
                    -1,
                )
                if (
                    corridor_goal_index != target_index
                    or proposed_path_index >= len(corridor_path)
                ):
                    corridor_path = self._plan_worker_corridor(
                        base_position,
                        goal_position,
                        worker_radius,
                        obstacles,
                        floor_bounds,
                    )
                    self._worker_corridor_paths[index] = corridor_path
                    self._worker_corridor_path_indices[index] = 0
                    self._worker_corridor_goal_indices[index] = target_index
                    proposed_path_index = 0
                    self._worker_replan_counts[index] = (
                        self._worker_replan_counts.get(index, 0) + 1
                    )
                if not corridor_path:
                    route_blocked_by = "no collision-free corridor"
                else:
                    while proposed_path_index < len(corridor_path):
                        path_point = corridor_path[proposed_path_index]
                        distance_to_point = Gf.Vec2d(
                            float(path_point[0] - base_position[0]),
                            float(path_point[1] - base_position[1]),
                        ).GetLength()
                        if distance_to_point > 0.045:
                            break
                        proposed_base_position = Gf.Vec3d(path_point)
                        base_position = Gf.Vec3d(path_point)
                        proposed_path_index += 1
                    if proposed_path_index >= len(corridor_path):
                        arrival_index = target_index
                        arrival_dwell = max(
                            0.0,
                            float(dwell_values[arrival_index]),
                        )
                        facing_yaw = float(facing_values[arrival_index])
                        if -360.0 <= facing_yaw <= 360.0:
                            desired_heading = facing_yaw
                        if arrival_dwell > 0.0:
                            proposed_target_index = arrival_index
                            proposed_dwell_remaining = arrival_dwell
                        else:
                            proposed_target_index = (
                                arrival_index + 1
                            ) % len(route_points)
                        self._worker_corridor_goal_indices[index] = -1
                    else:
                        path_point = corridor_path[proposed_path_index]
                        blocked_edge = self._worker_segment_obstacle(
                            base_position,
                            path_point,
                            worker_radius,
                            obstacles,
                            floor_bounds,
                        )
                        if blocked_edge:
                            corridor_path = self._plan_worker_corridor(
                                base_position,
                                goal_position,
                                worker_radius,
                                obstacles,
                                floor_bounds,
                            )
                            self._worker_corridor_paths[index] = corridor_path
                            self._worker_corridor_path_indices[index] = 0
                            proposed_path_index = 0
                            self._worker_replan_counts[index] = (
                                self._worker_replan_counts.get(index, 0) + 1
                            )
                            if not corridor_path:
                                route_blocked_by = blocked_edge
                            else:
                                path_point = corridor_path[0]
                                blocked_edge = self._worker_segment_obstacle(
                                    base_position,
                                    path_point,
                                    worker_radius,
                                    obstacles,
                                    floor_bounds,
                                )
                                if blocked_edge:
                                    route_blocked_by = blocked_edge
                        if not route_blocked_by and corridor_path:
                            path_point = corridor_path[proposed_path_index]
                            to_path_point = Gf.Vec2d(
                                float(path_point[0] - base_position[0]),
                                float(path_point[1] - base_position[1]),
                            )
                            distance_to_point = to_path_point.GetLength()
                            if distance_to_point > 1.0e-5:
                                direction = to_path_point / distance_to_point
                                desired_heading = math.degrees(
                                    math.atan2(
                                        -float(direction[0]),
                                        float(direction[1]),
                                    )
                                )
                                route_motion_requested = True

            current_heading = self._worker_heading_degrees.get(
                index,
                _read_yaw_degrees(worker),
            )
            if desired_heading is not None:
                current_heading = self._smooth_worker_heading(
                    current_heading,
                    desired_heading,
                    dt,
                )
            self._worker_heading_degrees[index] = current_heading

            locomotion_weight = self._worker_locomotion_weights.get(
                index,
                0.0,
            )
            heading_alignment = 0.0
            if route_motion_requested and desired_heading is not None:
                remaining_turn = (
                    (desired_heading - current_heading + 180.0) % 360.0
                ) - 180.0
                heading_alignment = max(
                    0.0,
                    min(
                        1.0,
                        (math.cos(math.radians(remaining_turn)) - 0.15)
                        / 0.85,
                    ),
                )
            target_locomotion = (
                heading_alignment if not route_blocked_by else 0.0
            )
            locomotion_rate = (
                WORKER_LOCOMOTION_ACCELERATION_PER_SECOND
                if target_locomotion > locomotion_weight
                else WORKER_LOCOMOTION_DECELERATION_PER_SECOND
            )
            locomotion_weight += max(
                -locomotion_rate * dt,
                min(
                    locomotion_rate * dt,
                    target_locomotion - locomotion_weight,
                ),
            )
            locomotion_weight = max(0.0, min(1.0, locomotion_weight))
            self._worker_locomotion_weights[index] = locomotion_weight

            if (
                route_motion_requested
                and not route_blocked_by
                and proposed_path_index
                < len(self._worker_corridor_paths.get(index, []))
            ):
                path_point = self._worker_corridor_paths[index][
                    proposed_path_index
                ]
                to_path_point = Gf.Vec2d(
                    float(path_point[0] - base_position[0]),
                    float(path_point[1] - base_position[1]),
                )
                distance_to_point = to_path_point.GetLength()
                if distance_to_point > 1.0e-5:
                    direction = to_path_point / distance_to_point
                    step = min(
                        walk_speed
                        * locomotion_weight
                        * heading_alignment
                        * dt,
                        distance_to_point,
                    )
                    proposed_base_position = Gf.Vec3d(
                        float(base_position[0])
                        + float(direction[0]) * step,
                        float(base_position[1])
                        + float(direction[1]) * step,
                        0.0,
                    )
                    if step >= distance_to_point - 1.0e-5:
                        proposed_path_index += 1

            offset = self._worker_push_offsets.get(index, Gf.Vec2d(0.0))
            velocity = self._worker_push_velocities.get(
                index,
                Gf.Vec2d(0.0),
            )
            # A damped spring lets an upright worker stagger aside, then
            # naturally return to the authored walking path.
            velocity += offset * (-4.0 * dt)
            velocity *= math.exp(-3.2 * dt)
            offset += velocity * dt
            offset_length = offset.GetLength()
            if offset_length > WORKER_MAX_OFFSET_M:
                offset *= WORKER_MAX_OFFSET_M / offset_length

            position = Gf.Vec3d(
                proposed_base_position[0] + offset[0],
                proposed_base_position[1] + offset[1],
                0.0,
            )
            if push_enabled and active_forklift.IsValid():
                contact = self._worker_contact_normal(
                    root,
                    active_forklift,
                    position,
                )
                if contact is not None:
                    normal, penetration = contact
                    vehicle_push_speed = max(
                        0.0,
                        Gf.Dot(self._last_forklift_velocity, normal),
                    )
                    target_push_speed = (
                        WORKER_PUSH_SPEED_MPS + vehicle_push_speed * 0.85
                    )
                    current_push_speed = Gf.Dot(velocity, normal)
                    if current_push_speed < target_push_speed:
                        velocity += normal * (
                            target_push_speed - current_push_speed
                        )
                    offset += normal * min(0.035, penetration)
                    offset_length = offset.GetLength()
                    if offset_length > WORKER_MAX_OFFSET_M:
                        offset *= WORKER_MAX_OFFSET_M / offset_length
                    position = Gf.Vec3d(
                        proposed_base_position[0] + offset[0],
                        proposed_base_position[1] + offset[1],
                        0.0,
                    )
            resolved_position, final_blocked_by = self._resolve_worker_motion(
                current_position,
                position,
                worker_radius,
                obstacles,
                floor_bounds,
            )
            visible_move_blocked = bool(
                final_blocked_by
                and Gf.Vec2d(
                    float(resolved_position[0] - position[0]),
                    float(resolved_position[1] - position[1]),
                ).GetLength()
                > 1.0e-4
            )
            blocked_by = route_blocked_by or (
                final_blocked_by if visible_move_blocked else ""
            )
            if blocked_by:
                # Freeze every form of authored route progress—including dwell
                # countdown—until the visible character can complete a move.
                proposed_base_position = Gf.Vec3d(
                    self._worker_route_positions[index]
                )
                proposed_target_index = original_target_index
                proposed_dwell_remaining = original_dwell_remaining
                proposed_path_index = original_path_index
                blocked_seconds = (
                    self._worker_blocked_seconds.get(index, 0.0) + dt
                )
                self._worker_blocked_seconds[index] = blocked_seconds
                if blocked_seconds >= WORKER_REPLAN_BLOCK_SECONDS:
                    self._worker_corridor_paths[index] = []
                    self._worker_corridor_path_indices[index] = 0
                    self._worker_corridor_goal_indices[index] = -1
            else:
                self._worker_blocked_seconds[index] = 0.0
                self._worker_route_positions[index] = Gf.Vec3d(
                    proposed_base_position
                )
                self._worker_route_waypoint_indices[index] = (
                    proposed_target_index
                )
                self._worker_route_dwell_remaining[index] = (
                    proposed_dwell_remaining
                )
                self._worker_corridor_path_indices[index] = (
                    proposed_path_index
                )
            if abs(float(resolved_position[0] - position[0])) > 1e-5:
                velocity[0] = 0.0
            if abs(float(resolved_position[1] - position[1])) > 1e-5:
                velocity[1] = 0.0
            offset = Gf.Vec2d(
                float(resolved_position[0])
                - float(proposed_base_position[0]),
                float(resolved_position[1])
                - float(proposed_base_position[1]),
            )
            self._worker_push_offsets[index] = offset
            self._worker_push_velocities[index] = velocity
            worker_api = UsdGeom.XformCommonAPI(worker)
            worker_api.SetTranslate(resolved_position)
            worker_api.SetRotate(
                Gf.Vec3f(0.0, 0.0, current_heading),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
            displacement = Gf.Vec2d(
                float(resolved_position[0] - current_position[0]),
                float(resolved_position[1] - current_position[1]),
            ).GetLength()
            moving = bool(
                not blocked_by
                and displacement > 1.0e-4
                and locomotion_weight > 0.08
            )
            self._set_worker_motion_animation(
                stage,
                worker,
                index,
                moving,
                dt,
            )
            self._write_worker_navigation_state(
                worker,
                index,
                blocked_by,
            )

    def _forklift_clearance(self, root: Any, prim: Any) -> float:
        position, yaw_degrees = self._forklift_pose(prim)
        return self._clearance_for_pose(
            root,
            position,
            yaw_degrees,
        )

    def _clearance_for_pose(
        self,
        root: Any,
        position: Gf.Vec3d,
        yaw_degrees: float,
    ) -> float:
        forklift_half_length = float(
            root.GetAttribute("codex:forkliftHalfLengthM").Get()
        )
        forklift_half_width = float(
            root.GetAttribute("codex:forkliftHalfWidthM").Get()
        )
        yaw = math.radians(yaw_degrees)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -math.sin(yaw)
        right_y = math.cos(yaw)
        clearances: list[float] = []
        for _, center_x, center_y, half_x, half_y in (
            self._conveyor_obstacle_aabbs(root)
        ):
            closest_x = max(
                center_x - half_x,
                min(center_x + half_x, float(position[0])),
            )
            closest_y = max(
                center_y - half_y,
                min(center_y + half_y, float(position[1])),
            )
            vx = float(position[0]) - closest_x
            vy = float(position[1]) - closest_y
            center_distance = math.hypot(vx, vy)
            if center_distance <= 1e-6:
                return 0.0
            nx = vx / center_distance
            ny = vy / center_distance
            # The articulated ForkliftC asset faces local +X.
            support = (
                forklift_half_length
                * abs(nx * forward_x + ny * forward_y)
                + forklift_half_width
                * abs(nx * right_x + ny * right_y)
            )
            clearances.append(max(0.0, center_distance - support))
        return min(clearances) if clearances else 99.0

    def _update_clearance(
        self,
        stage: Any,
        root: Any,
        _throttle: float,
    ) -> float:
        clearances: list[float] = []
        for index in self._enabled_forklift_indices(root):
            prim = stage.GetPrimAtPath(f"{FORKLIFT_ROOT}/Forklift{index + 1}")
            if prim.IsValid():
                clearance = self._forklift_clearance(root, prim)
                clearances.append(clearance)
        min_clearance = min(clearances) if clearances else 99.0
        forklift_moving = (
            math.hypot(
                float(self._last_forklift_velocity[0]),
                float(self._last_forklift_velocity[1]),
            )
            >= FORKLIFT_MOVING_THRESHOLD_MPS
        )
        if min_clearance < 1.0:
            ground_truth = "RED"
        elif forklift_moving:
            ground_truth = "AMBER"
        else:
            ground_truth = "GREEN"
        if (
            self._last_clearance_written is None
            or abs(min_clearance - self._last_clearance_written) > 0.005
        ):
            root.GetAttribute("codex:minForkliftClearanceM").Set(min_clearance)
            self._last_clearance_written = min_clearance
        if ground_truth != self._last_ground_truth_written:
            root.GetAttribute("codex:groundTruthSignal").Set(ground_truth)
            self._last_ground_truth_written = ground_truth
        return min_clearance

    def _update_ui(
        self,
        root: Any,
        gamepad: carb.input.Gamepad | None,
        throttle: float,
        steering: float,
        min_clearance: float,
    ) -> None:
        self._labels["connection"].text = (
            "Xbox Controller connected"
            if gamepad is not None
            else "No gamepad · keyboard fallback active"
        )
        self._labels["controller"].text = (
            "PHYSICS ForkliftC rear-steer Ackermann\n"
            f"ACTIVE  Forklift {self._active_index + 1}\n"
            f"TARGET {float(root.GetAttribute('codex:commandedSpeedMps').Get()):+.2f} m/s"
            f"   ACTUAL {float(root.GetAttribute('codex:actualSpeedMps').Get()):+.2f} m/s\n"
            f"   STEER {float(root.GetAttribute('codex:steeringAngleDegrees').Get()):+.1f}°\n"
            f"FORK HEIGHT {float(root.GetAttribute('codex:forkHeightM').Get()):.2f} m"
            "   D-pad Up/Down / I K\n"
            "CAMERA  Right stick orbit · View cycles zoom\n"
            f"VIEW {self._presentation_camera_distance_index + 1}/3"
            f"  {PRESENTATION_CAMERA_DISTANCES_M[self._presentation_camera_distance_index]:.1f} m"
            " · follows active forklift\n"
            f"X toggles inference · {self._inference_backend.upper()} selected\n"
            "Y / R resets loose cargo"
        )
        model_signal = str(root.GetAttribute("codex:modelSignal").Get())
        ground_truth = str(root.GetAttribute("codex:groundTruthSignal").Get())
        self._labels["safety"].text = (
            f"MODEL TOWER    {model_signal}\n"
            f"VALIDATION     {ground_truth}\n"
            f"MIN CLEARANCE  {min_clearance:.2f} m\n"
            "HARD COLLIDERS  oval conveyor + rack + floor"
        )
        self._labels["inference"].text = (
            f"TARGET {self._inference_backend.upper()}"
            f"  ·  {self._inference_server_url}\n"
            f"{self._inference_status}"
        )
