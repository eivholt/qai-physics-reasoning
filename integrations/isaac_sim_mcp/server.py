from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from integrations.isaac_sim_mcp.conveyor_safety import (
        CAMERA_PATHS as CONVEYOR_SAFETY_CAMERA_PATHS,
        capture_conveyor_safety_camera_source,
        create_conveyor_safety_scene_source,
        enable_conveyor_safety_extension_source,
        get_conveyor_safety_state_source,
        set_conveyor_safety_camera_source,
        set_conveyor_safety_forklift_pose_source,
        set_conveyor_safety_light_source,
    )
    from integrations.isaac_sim_mcp.edge_supervisor import (
        SCENARIOS,
        asset_status_source,
        capture_camera_views_source,
        capture_realistic_camera_views_source,
        capture_realistic_sequence_source,
        capture_sequence_source,
        create_lightweight_live_scene_source,
        create_realistic_scene_source,
        create_scene_source as create_edge_supervisor_scene_source,
        inspect_realistic_warehouse_source,
        set_progress_source as set_edge_supervisor_progress_source,
    )
    from integrations.isaac_sim_mcp.live_aisle_supervisor import (
        apply_advisory_source as apply_live_aisle_advisory_source,
        capture_live_camera_grid_source,
        capture_live_robot_front_frame_source,
        capture_live_roof_frame_source,
        capture_live_tactical_frame_source,
        configure_blind_corner_source,
        configure_navigation_graph_source,
        configure_robot_pov_source,
        configure_route_visualizations_source,
        create_live_aisle_source,
        stock_blind_corner_shelves_source,
        set_congestion_source as set_live_aisle_congestion_source,
        set_forklift_visible_source,
        set_request_status_source as set_live_aisle_request_status_source,
        tick_live_aisle_source,
    )
except ModuleNotFoundError:
    # Preserve `python integrations\isaac_sim_mcp\server.py ...` from the README.
    from conveyor_safety import (  # type: ignore[no-redef]
        CAMERA_PATHS as CONVEYOR_SAFETY_CAMERA_PATHS,
        capture_conveyor_safety_camera_source,
        create_conveyor_safety_scene_source,
        enable_conveyor_safety_extension_source,
        get_conveyor_safety_state_source,
        set_conveyor_safety_camera_source,
        set_conveyor_safety_forklift_pose_source,
        set_conveyor_safety_light_source,
    )
    from edge_supervisor import (  # type: ignore[no-redef]
        SCENARIOS,
        asset_status_source,
        capture_camera_views_source,
        capture_realistic_camera_views_source,
        capture_realistic_sequence_source,
        capture_sequence_source,
        create_lightweight_live_scene_source,
        create_realistic_scene_source,
        create_scene_source as create_edge_supervisor_scene_source,
        inspect_realistic_warehouse_source,
        set_progress_source as set_edge_supervisor_progress_source,
    )
    from live_aisle_supervisor import (  # type: ignore[no-redef]
        apply_advisory_source as apply_live_aisle_advisory_source,
        capture_live_camera_grid_source,
        capture_live_robot_front_frame_source,
        capture_live_roof_frame_source,
        capture_live_tactical_frame_source,
        configure_blind_corner_source,
        configure_navigation_graph_source,
        configure_robot_pov_source,
        configure_route_visualizations_source,
        create_live_aisle_source,
        stock_blind_corner_shelves_source,
        set_congestion_source as set_live_aisle_congestion_source,
        set_forklift_visible_source,
        set_request_status_source as set_live_aisle_request_status_source,
        tick_live_aisle_source,
    )


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8226
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
SERVER_NAME = "qai-isaac-sim-control"
SERVER_VERSION = "0.2.0"
DEFAULT_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "isaac_sim_poc"
)
DEFAULT_CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "isaac_sim_checkpoints"
)
DEFAULT_TUTORIAL_EXTENSION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "isaac_sim_supervisor_omniverse"
    / "exts"
)


class IsaacConnectionError(RuntimeError):
    """Raised when the Isaac Sim Python server cannot be reached or decoded."""


@dataclass(frozen=True)
class IsaacTcpClient:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls) -> IsaacTcpClient:
        return cls(
            host=os.environ.get("ISAAC_SIM_HOST", DEFAULT_HOST),
            port=int(os.environ.get("ISAAC_SIM_PORT", str(DEFAULT_PORT))),
            timeout_seconds=float(
                os.environ.get(
                    "ISAAC_SIM_TIMEOUT_SECONDS",
                    str(DEFAULT_TIMEOUT_SECONDS),
                )
            ),
        )

    def execute(self, source: str) -> dict[str, Any]:
        if not source.strip():
            raise ValueError("Isaac Sim source must not be empty")

        try:
            with socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout_seconds,
            ) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.sendall(source.encode("utf-8"))
                connection.shutdown(socket.SHUT_WR)

                chunks: list[bytes] = []
                byte_count = 0
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > MAX_RESPONSE_BYTES:
                        raise IsaacConnectionError(
                            "Isaac Sim response exceeded the 16 MiB safety limit"
                        )
                    chunks.append(chunk)
        except (ConnectionError, OSError, TimeoutError) as error:
            raise IsaacConnectionError(
                f"Could not reach Isaac Sim Python server at "
                f"{self.host}:{self.port}. Start Isaac Sim with "
                f"'--enable isaacsim.code_editor.python_server'. "
                f"Underlying error: {error}"
            ) from error

        payload = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if not payload:
            raise IsaacConnectionError(
                "Isaac Sim closed the connection without returning JSON"
            )

        try:
            response = json.loads(payload)
        except json.JSONDecodeError as error:
            raise IsaacConnectionError(
                f"Isaac Sim returned invalid JSON: {payload[:500]}"
            ) from error

        if not isinstance(response, dict):
            raise IsaacConnectionError(
                f"Isaac Sim returned {type(response).__name__}, expected an object"
            )
        return response


TOOLS: list[dict[str, Any]] = [
    {
        "name": "isaac_ping",
        "description": (
            "Check that the running Isaac Sim Python server is reachable. "
            "This is read-only and should be called before other Isaac tools."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_get_status",
        "description": (
            "Return the current USD stage identifier, timeline state, and prim count."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_frame_rate_limit",
        "description": (
            "Cap the running Isaac Sim main, present, and rendering loops to a "
            "bounded frequency using precision sleep, reducing idle GPU load."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 240,
                    "default": 60,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_markings_enabled",
        "description": (
            "Enable or disable only /World/CodexPoC/Markings without changing "
            "the supervisor routes, conflict zone, actors, or warehouse."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
            },
            "required": ["enabled"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_presentation_overlays_enabled",
        "description": (
            "Enable or disable all tutorial-only Markings, Supervisor route and "
            "conflict visuals, actor footprint pads, and actor beacons while "
            "leaving physical actors and warehouse assets unchanged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
            },
            "required": ["enabled"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_save_stage_checkpoint",
        "description": (
            "Export the current USD root layer to a bounded repository "
            "checkpoint path before risky navigation experiments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "default": "warehouse_checkpoint.usda",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_load_stage_checkpoint",
        "description": (
            "Open a previously exported bounded repository USD checkpoint and "
            "wait for its referenced assets to finish loading."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_lightweight_warehouse_wall_height",
        "description": (
            "Set the absolute height of the four lightweight warehouse walls, "
            "keeping their bottom edges on the floor and leaving all other "
            "scene prims unchanged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "height_m": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 12.0,
                    "default": 3.9,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_enable_edge_ai_supervisor_ui",
        "description": (
            "Register and enable the repository's dockable Edge AI Warehouse "
            "Supervisor panel in the running Isaac Sim application."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_enable_conveyor_safety",
        "description": (
            "Enable or hot-reload the Xbox controller and live status panel "
            "for the Reason2 conveyor safety scene."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_create_conveyor_safety_scene",
        "description": (
            "Create a new logistics floor with a central animated conveyor, "
            "nine looping parcels, four moving workers, three controllable "
            "forklifts, detector cameras, and a model-driven stack light."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "new_stage": {"type": "boolean", "default": True},
                "start_playing": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_conveyor_safety_camera",
        "description": (
            "Capture one conveyor safety detector or presentation camera. "
            "The physical stack light remains visible in detector captures; "
            "the runner logs its input color to monitor visual feedback bias."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": list(CONVEYOR_SAFETY_CAMERA_PATHS),
                    "default": "detector_ptz",
                },
                "filename": {
                    "type": "string",
                    "default": "conveyor_safety.png",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_conveyor_safety_light",
        "description": (
            "Apply a validated Cosmos Reason2 GREEN, AMBER, or RED result to "
            "the three-lens tower without using simulator ground truth."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": ["GREEN", "AMBER", "RED"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                },
                "inference_id": {"type": "string", "maxLength": 128},
            },
            "required": ["signal"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_get_conveyor_safety_state",
        "description": (
            "Return forklift poses, exact clearance validation, controller "
            "status, and current model-driven tower state. These facts are "
            "for testing and are not sent to Reason2."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_conveyor_safety_forklift_pose",
        "description": (
            "Place one forklift at a bounded floor pose for deterministic "
            "camera evaluation without requiring gamepad input."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                },
                "x": {"type": "number", "minimum": -9.0, "maximum": 9.0},
                "y": {"type": "number", "minimum": -6.7, "maximum": 6.7},
                "yaw_degrees": {
                    "type": "number",
                    "minimum": -360.0,
                    "maximum": 360.0,
                },
            },
            "required": ["index", "x", "y", "yaw_degrees"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_conveyor_safety_camera",
        "description": (
            "Show a detector candidate or the wide presentation camera in the "
            "interactive Isaac Sim viewport."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": list(CONVEYOR_SAFETY_CAMERA_PATHS),
                    "default": "presentation",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_create_live_aisle_navigation",
        "description": (
            "Add one packaged target-driven Nova Carter plus the existing "
            "Forklift B obstacle to the realistic warehouse for a live "
            "aisle-congestion experiment. Scripted actors are preserved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "robot_width": {
                    "type": "number",
                    "minimum": 1.1,
                    "maximum": 1.8,
                    "default": 1.42,
                },
                "robot_length": {
                    "type": "number",
                    "minimum": 1.5,
                    "maximum": 2.8,
                    "default": 2.1,
                },
                "robot_scale": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 2.5,
                    "default": 1.75,
                },
                "start_playing": {"type": "boolean", "default": True},
                "forklift_start_mode": {
                    "type": "string",
                    "enum": ["current", "configured"],
                    "default": "current",
                },
                "motion_controller": {
                    "type": "string",
                    "enum": ["isaac_graph", "kinematic_waypoint"],
                    "default": "isaac_graph",
                },
                "scenario_mode": {
                    "type": "string",
                    "enum": ["blind_corner", "clear_route_control"],
                    "default": "blind_corner",
                },
                "kinematic_speed_mps": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.8,
                    "default": 0.18,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_configure_live_blind_corner",
        "description": (
            "Author a repeatable forklift approach from behind the central "
            "rack, clockwise turn, and final right-passage blockage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_x": {
                    "type": "number",
                    "minimum": 1.5,
                    "maximum": 4.0,
                    "default": 2.2,
                },
                "start_y": {
                    "type": "number",
                    "minimum": 4.0,
                    "maximum": 6.0,
                    "default": 4.835,
                },
                "turn_x": {
                    "type": "number",
                    "minimum": 4.5,
                    "maximum": 5.7,
                    "default": 5.2,
                },
                "turn_y": {
                    "type": "number",
                    "minimum": 4.0,
                    "maximum": 6.0,
                    "default": 4.835,
                },
                "end_x": {
                    "type": "number",
                    "minimum": 5.6,
                    "maximum": 6.3,
                    "default": 5.94,
                },
                "end_y": {
                    "type": "number",
                    "minimum": 3.0,
                    "maximum": 4.4,
                    "default": 3.6,
                },
                "speed_mps": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.8,
                    "default": 0.8,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_stock_live_blind_corner_shelves",
        "description": (
            "Fill the northeast blind-corner rack with official Isaac warehouse "
            "cartons to make roof-versus-robot occlusion unambiguous."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 5,
                    "default": 4,
                },
                "levels": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 3,
                    "default": 3,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_configure_live_robot_pov",
        "description": (
            "Create a RobotBlue camera just ahead of the enlarged shell while "
            "preserving NVIDIA's packaged front Owl camera."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "forward_m": {
                    "type": "number",
                    "minimum": 0.8,
                    "maximum": 1.5,
                    "default": 1.15,
                },
                "height_m": {
                    "type": "number",
                    "minimum": 0.45,
                    "maximum": 1.1,
                    "default": 0.72,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_configure_live_navigation_graph",
        "description": (
            "Wire the packaged Carter Quintic/Stanley/differential/articulation "
            "nodes into a complete target-driven navigation graph."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_configure_live_route_visualizations",
        "description": (
            "Show the white nominal route, cyan north bypass, and orange south "
            "bypass without re-enabling the disabled warehouse Markings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_live_aisle_congestion",
        "description": (
            "Release or hold Forklift B in the live aisle. Releasing it sends "
            "the obstacle west into RobotBlue's intended path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "released": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_live_forklift_visible",
        "description": (
            "Show or hide ForkliftOrange without changing its authored blind-"
            "corner transform; used for a genuine no-hazard baseline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "visible": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_apply_live_aisle_advisory",
        "description": (
            "Apply a validated EVK-style CONTINUE, YIELD, STOP, or REROUTE "
            "advisory to RobotBlue's target-driven navigation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["CONTINUE", "YIELD", "STOP", "REROUTE"],
                },
                "route": {
                    "type": "string",
                    "enum": [
                        "current",
                        "direct",
                        "direct_to_goal",
                        "hold",
                        "hold_west_gate",
                        "south_bypass",
                        "north_bypass",
                    ],
                },
                "request_id": {"type": "string", "maxLength": 128},
                "decision_source": {
                    "type": "string",
                    "enum": ["evk", "local_safety", "manual"],
                    "default": "evk",
                },
                "latency_seconds": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 120.0,
                },
            },
            "required": ["action", "route"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_tick_live_aisle",
        "description": (
            "Advance the bounded south-bypass waypoint state machine when a "
            "waypoint is reached and return live robot positions and EVK state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_live_aisle_request_status",
        "description": (
            "Update the live scene's bounded EVK request status for inspection "
            "and later presentation capture."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "idle",
                        "buffering",
                        "in_flight",
                        "applied",
                        "failed",
                    ],
                },
                "request_id": {"type": "string", "maxLength": 128},
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_live_aisle_camera_grid",
        "description": (
            "Capture synchronized north-up overview, approach, blind-corner, "
            "and north-bypass frames for a four-panel supervisor mosaic. "
            "Presentation route overlays are excluded by default."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "default": "live_camera_grid.png",
                },
                "include_route_visualizations": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_live_aisle_roof_frame",
        "description": (
            "Capture the current RoofOverview frame for the live aisle EVK "
            "sliding window under artifacts/isaac_sim_live_aisle/frames. "
            "Presentation route overlays are excluded by default."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "default": "live_frame.png",
                },
                "include_route_visualizations": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_live_aisle_tactical_frame",
        "description": (
            "Capture the fixed blind-corner tactical supervisor camera at a "
            "useful actor scale. Presentation route overlays are excluded by "
            "default."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "default": "live_tactical_frame.png",
                },
                "include_route_visualizations": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_live_robot_front_frame",
        "description": (
            "Capture RobotBlue's packaged front Owl camera so blind-corner "
            "occlusion can be compared with the roof supervisor view."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "default": "robotblue_front.png",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_create_poc_scene",
        "description": (
            "Idempotently create a small floor, colored marker cube, target cube, "
            "and dome light under /World/CodexPoC. Existing scene content outside "
            "that namespace is left untouched."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_marker_pose",
        "description": (
            "Move /World/CodexPoC/Marker to an XYZ position after the PoC scene "
            "has been created."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["x", "y", "z"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_timeline",
        "description": "Play, pause, or stop the Isaac Sim timeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "stop"],
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_list_prims",
        "description": (
            "List prim paths and types below a USD root path, with bounded depth "
            "and result count."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {
                    "type": "string",
                    "description": "Absolute USD prim path, such as /World.",
                    "default": "/World",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                    "default": 3,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 100,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_viewport",
        "description": (
            "Capture the active Isaac Sim viewport to a PNG under the repository's "
            "artifacts/isaac_sim_poc directory. The filename must be a simple "
            "basename and cannot select an arbitrary output directory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "PNG basename, default codex_poc.png.",
                    "default": "codex_poc.png",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_get_edge_supervisor_assets",
        "description": (
            "Resolve the configured Isaac Sim asset root and check the NVIDIA "
            "warehouse, Nova Carter, forklift, and construction-worker USDs "
            "used by the edge-supervisor tutorial."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_create_edge_supervisor_scene",
        "description": (
            "Replace /World/CodexPoC with a tutorial warehouse containing four "
            "roof cameras, two AMRs, a forklift, a worker, shelves, traffic "
            "markings, and a visual supervisor conflict zone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_create_realistic_edge_supervisor_scene",
        "description": (
            "Replace /World/CodexPoC with the edge-supervisor actors, routes, and "
            "roof cameras composed inside NVIDIA's pre-modeled full warehouse. "
            "Existing compact-scene recording files are not modified."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_create_lightweight_live_warehouse",
        "description": (
            "Replace /World/CodexPoC with a lightweight live-navigation "
            "warehouse built only from a floor, low walls, standalone NVIDIA "
            "rack modules, actors, props, lights, and roof cameras."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_inspect_realistic_warehouse",
        "description": (
            "Return the composed NVIDIA warehouse bounds and bounded transforms "
            "for top-level floor, ceiling, light, rack, and shelf prims."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_set_edge_supervisor_progress",
        "description": (
            "Move the tutorial actors to a normalized scenario time and return "
            "the deterministic CONTINUE, YIELD, STOP, or REROUTE notification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "progress": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIOS),
                    "default": "mixed_traffic",
                },
            },
            "required": ["progress"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_edge_supervisor_cameras",
        "description": (
            "Capture synchronized overview, west-aisle, intersection, and "
            "east-aisle roof camera PNGs for one tutorial progress value."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "progress": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIOS),
                    "default": "mixed_traffic",
                },
            },
            "required": ["progress"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_edge_supervisor_sequence",
            "description": (
                "Capture 8 to 48 overview-camera PNG frames spanning the complete "
                "selected tutorial scenario for GIF or MP4 encoding."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_count": {
                    "type": "integer",
                    "minimum": 8,
                    "maximum": 48,
                    "default": 25,
                },
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIOS),
                    "default": "mixed_traffic",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_realistic_edge_supervisor_cameras",
        "description": (
            "Capture four roof-camera PNGs from the NVIDIA full-warehouse scene "
            "under a separate realistic artifact directory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "progress": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIOS),
                    "default": "mixed_traffic",
                },
                "variant": {
                    "type": "string",
                    "enum": ["default", "v2", "v3", "v4"],
                    "default": "default",
                },
            },
            "required": ["progress"],
            "additionalProperties": False,
        },
    },
    {
        "name": "isaac_capture_realistic_edge_supervisor_sequence",
        "description": (
            "Capture 8 to 48 overview frames from the NVIDIA full-warehouse "
            "scene under a separate realistic artifact directory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_count": {
                    "type": "integer",
                    "minimum": 8,
                    "maximum": 48,
                    "default": 25,
                },
                "scenario": {
                    "type": "string",
                    "enum": list(SCENARIOS),
                    "default": "mixed_traffic",
                },
                "variant": {
                    "type": "string",
                    "enum": ["default", "v2", "v3", "v4"],
                    "default": "default",
                },
            },
            "additionalProperties": False,
        },
    },
]


def _pretty_isaac_response(response: dict[str, Any]) -> dict[str, Any]:
    is_error = response.get("status") != "ok"
    result: dict[str, Any] = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(response, indent=2, ensure_ascii=False),
            }
        ]
    }
    if is_error:
        result["isError"] = True
    return result


def _timeline_state_source() -> str:
    return """
import json
import omni.timeline
import omni.usd

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
if stage is None:
    raise RuntimeError("No USD stage is currently open")

state = "playing" if timeline.is_playing() else ("stopped" if timeline.is_stopped() else "paused")
print(json.dumps({
    "stage_identifier": stage.GetRootLayer().identifier,
    "timeline_state": state,
    "prim_count": sum(1 for _ in stage.Traverse()),
}))
""".strip()


def _frame_rate_limit_source(arguments: dict[str, Any]) -> str:
    fps_value = arguments.get("fps", 60)
    if (
        isinstance(fps_value, bool)
        or not isinstance(fps_value, int)
        or not 1 <= fps_value <= 240
    ):
        raise ValueError("fps must be an integer between 1 and 240")

    return f"""
import json
import carb

fps = {fps_value}
settings = carb.settings.get_settings()
loop_names = ("main", "present", "rendering_0", "rendering_1")
for loop_name in loop_names:
    prefix = f"/app/runLoops/{{loop_name}}"
    settings.set_bool(f"{{prefix}}/rateLimitEnabled", True)
    settings.set_int(f"{{prefix}}/rateLimitFrequency", fps)
    settings.set_bool(f"{{prefix}}/rateLimitUsePrecisionSleep", True)
settings.set_bool("/app/runLoops/main/rateLimitUseBusyLoop", False)

print(json.dumps({{
    "requested_fps": fps,
    "loops": {{
        loop_name: {{
            "enabled": settings.get(
                f"/app/runLoops/{{loop_name}}/rateLimitEnabled"
            ),
            "frequency": settings.get(
                f"/app/runLoops/{{loop_name}}/rateLimitFrequency"
            ),
            "precision_sleep": settings.get(
                f"/app/runLoops/{{loop_name}}/rateLimitUsePrecisionSleep"
            ),
        }}
        for loop_name in loop_names
    }},
    "main_busy_loop": settings.get(
        "/app/runLoops/main/rateLimitUseBusyLoop"
    ),
}}))
""".strip()


def _set_markings_enabled_source(arguments: dict[str, Any]) -> str:
    enabled = arguments.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    return f"""
import json
import omni.usd

path = "/World/CodexPoC/Markings"
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
prim = stage.GetPrimAtPath(path)
if not prim.IsValid():
    raise RuntimeError("The edge-supervisor Markings group does not exist")
prim.SetActive({enabled!r})
print(json.dumps({{
    "path": path,
    "active": prim.IsActive(),
}}))
""".strip()


def _set_presentation_overlays_enabled_source(
    arguments: dict[str, Any],
) -> str:
    enabled = arguments.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    return f"""
import json
import omni.usd

root = "/World/CodexPoC"
paths = (
    f"{{root}}/Markings",
    f"{{root}}/Supervisor",
    f"{{root}}/Actors/RobotBlue/IdentityPad",
    f"{{root}}/Actors/RobotBlue/Beacon",
    f"{{root}}/Actors/RobotGreen/IdentityPad",
    f"{{root}}/Actors/RobotGreen/Beacon",
    f"{{root}}/Actors/ForkliftOrange/IdentityPad",
    f"{{root}}/Actors/ForkliftOrange/Beacon",
    f"{{root}}/Actors/WorkerYellow/IdentityPad",
)
context = omni.usd.get_context()
stage = context.get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
results = {{}}
for path in paths:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        prim.SetActive({enabled!r})
        results[path] = prim.IsActive()
    else:
        results[path] = None
context.get_selection().clear_selected_prim_paths()
print(json.dumps({{
    "enabled": {enabled!r},
    "paths": results,
    "selection_cleared": True,
}}))
""".strip()


def _checkpoint_path(arguments: dict[str, Any]) -> Path:
    filename = arguments.get("filename", "warehouse_checkpoint.usda")
    if (
        not isinstance(filename, str)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:usd|usda|usdc)",
            filename,
            flags=re.IGNORECASE,
        )
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("filename must be a simple .usd, .usda, or .usdc basename")
    DEFAULT_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    return (DEFAULT_CHECKPOINT_ROOT / filename).resolve()


def _save_stage_checkpoint_source(arguments: dict[str, Any]) -> str:
    output_path = _checkpoint_path(arguments)
    return f"""
import json
import os
import omni.usd

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
output_path = {str(output_path)!r}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
saved = stage.GetRootLayer().Export(output_path)
if not saved:
    raise RuntimeError(f"Failed to export stage checkpoint: {{output_path}}")
print(json.dumps({{
    "saved": True,
    "output_path": output_path,
    "bytes": os.path.getsize(output_path),
}}))
""".strip()


def _load_stage_checkpoint_source(arguments: dict[str, Any]) -> str:
    checkpoint_path = _checkpoint_path(arguments)
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path.name}")
    return f"""
import json
import omni.usd

checkpoint_path = {str(checkpoint_path)!r}
context = omni.usd.get_context()
opened, error = await context.open_stage_async(checkpoint_path)
if not opened:
    raise RuntimeError(f"Failed to open checkpoint: {{error}}")
stable_frames = 0
for _ in range(1200):
    loading_status = context.get_stage_loading_status()
    stable_frames = stable_frames + 1 if not any(loading_status) else 0
    if stable_frames >= 30:
        break
    await omni.kit.app.get_app().next_update_async()
stage = context.get_stage()
print(json.dumps({{
    "loaded": True,
    "checkpoint_path": checkpoint_path,
    "root_layer": stage.GetRootLayer().identifier if stage else None,
    "loading_status": context.get_stage_loading_status(),
}}))
""".strip()


def _set_lightweight_warehouse_wall_height_source(
    arguments: dict[str, Any],
) -> str:
    height_m = arguments.get("height_m", 3.9)
    if (
        isinstance(height_m, bool)
        or not isinstance(height_m, (int, float))
        or not 0.5 <= float(height_m) <= 12.0
    ):
        raise ValueError("height_m must be a number between 0.5 and 12.0")
    height_m = float(height_m)
    return f"""
import json
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")

wall_paths = (
    "/World/CodexPoC/Environment/Walls/Y_m4_75",
    "/World/CodexPoC/Environment/Walls/Y_6_35",
    "/World/CodexPoC/Environment/Walls/X_m7_32",
    "/World/CodexPoC/Environment/Walls/X_7_32",
)
height_m = {height_m!r}
updated = []
for path in wall_paths:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Cube):
        raise RuntimeError(f"Expected lightweight warehouse wall is missing: {{path}}")
    translate_attr = prim.GetAttribute("xformOp:translate")
    scale_attr = prim.GetAttribute("xformOp:scale")
    translate = translate_attr.Get() if translate_attr.IsValid() else None
    scale = scale_attr.Get() if scale_attr.IsValid() else None
    if translate is None or scale is None:
        raise RuntimeError(f"Wall has no authored translate/scale transform: {{path}}")

    before_translate = [float(component) for component in translate]
    before_scale = [float(component) for component in scale]
    floor_z = float(translate[2]) - float(scale[2]) * 0.5
    after_translate = Gf.Vec3d(
        float(translate[0]),
        float(translate[1]),
        floor_z + height_m * 0.5,
    )
    after_scale = Gf.Vec3f(
        float(scale[0]),
        float(scale[1]),
        height_m,
    )
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(after_translate)
    xform_api.SetScale(after_scale)
    updated.append({{
        "path": path,
        "before_translate": before_translate,
        "before_scale": before_scale,
        "after_translate": [float(component) for component in after_translate],
        "after_scale": [float(component) for component in after_scale],
        "floor_z": floor_z,
    }})

print(json.dumps({{
    "updated": True,
    "height_m": height_m,
    "wall_count": len(updated),
    "walls": updated,
}}))
""".strip()


def _enable_edge_ai_supervisor_ui_source() -> str:
    extension_root = DEFAULT_TUTORIAL_EXTENSION_ROOT.resolve()
    if not extension_root.is_dir():
        raise ValueError(
            f"tutorial extension folder does not exist: {extension_root}"
        )
    return f"""
import json
import omni.kit.app
import omni.ui as ui

extension_root = {str(extension_root)!r}
extension_name = "qai.edge_ai_supervisor"
manager = omni.kit.app.get_app().get_extension_manager()
known_paths = [
    item.get("path")
    for item in manager.get_folders()
    if isinstance(item, dict)
]
if extension_root not in known_paths:
    manager.add_path(extension_root)
was_enabled = manager.is_extension_enabled(extension_name)
if was_enabled:
    manager.set_extension_enabled_immediate(extension_name, False)
    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()
enabled = manager.set_extension_enabled_immediate(extension_name, True)
for _ in range(4):
    await omni.kit.app.get_app().next_update_async()
window = ui.Workspace.get_window("Edge AI Warehouse Supervisor")
print(json.dumps({{
    "extension_root": extension_root,
    "extension_name": extension_name,
    "reloaded": bool(was_enabled),
    "enable_call_succeeded": bool(enabled),
    "enabled": manager.is_extension_enabled(extension_name),
    "enabled_extension_id": manager.get_enabled_extension_id(extension_name),
    "window_created": window is not None,
    "window_visible": bool(window.visible) if window is not None else False,
}}))
""".strip()


def _create_poc_scene_source() -> str:
    return """
import json
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")

UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/CodexPoC")

floor = UsdGeom.Cube.Define(stage, "/World/CodexPoC/Floor")
floor.CreateSizeAttr(1.0)
floor.CreateDisplayColorAttr().Set([Gf.Vec3f(0.18, 0.20, 0.23)])
floor_xform = UsdGeom.XformCommonAPI(floor.GetPrim())
floor_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.05))
floor_xform.SetScale(Gf.Vec3f(8.0, 8.0, 0.1))

marker = UsdGeom.Cube.Define(stage, "/World/CodexPoC/Marker")
marker.CreateSizeAttr(1.0)
marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.45, 1.0)])
marker_xform = UsdGeom.XformCommonAPI(marker.GetPrim())
marker_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.5))
marker_xform.SetScale(Gf.Vec3f(0.5, 0.5, 0.5))

target = UsdGeom.Cube.Define(stage, "/World/CodexPoC/Target")
target.CreateSizeAttr(1.0)
target.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.25, 0.12)])
target_xform = UsdGeom.XformCommonAPI(target.GetPrim())
target_xform.SetTranslate(Gf.Vec3d(2.0, 0.0, 0.25))
target_xform.SetScale(Gf.Vec3f(0.5, 0.5, 0.5))

light = UsdLux.DomeLight.Define(stage, "/World/CodexPoC/Light")
light.CreateIntensityAttr(750.0)

from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
viewport = get_active_viewport()
if viewport is not None:
    frame_viewport_prims(viewport, prims=["/World/CodexPoC"])

print(json.dumps({
    "created": True,
    "root": "/World/CodexPoC",
    "marker": "/World/CodexPoC/Marker",
    "target": "/World/CodexPoC/Target",
}))
""".strip()


def _set_marker_source(arguments: dict[str, Any]) -> str:
    coordinates = []
    for name in ("x", "y", "z"):
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        coordinates.append(float(value))
    x, y, z = coordinates
    return f"""
import json
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
prim = stage.GetPrimAtPath("/World/CodexPoC/Marker")
if not prim.IsValid():
    raise RuntimeError("Run isaac_create_poc_scene before moving the marker")
UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d({x!r}, {y!r}, {z!r}))
print(json.dumps({{"path": str(prim.GetPath()), "position": [{x!r}, {y!r}, {z!r}]}}))
""".strip()


def _set_timeline_source(arguments: dict[str, Any]) -> str:
    action = arguments.get("action")
    if action not in {"play", "pause", "stop"}:
        raise ValueError("action must be one of: play, pause, stop")
    return f"""
import json
import omni.timeline

timeline = omni.timeline.get_timeline_interface()
getattr(timeline, {action!r})()
state = "playing" if timeline.is_playing() else ("stopped" if timeline.is_stopped() else "paused")
print(json.dumps({{"requested_action": {action!r}, "timeline_state": state}}))
""".strip()


def _list_prims_source(arguments: dict[str, Any]) -> str:
    root_path = arguments.get("root_path", "/World")
    max_depth = arguments.get("max_depth", 3)
    limit = arguments.get("limit", 100)
    if not isinstance(root_path, str) or not root_path.startswith("/"):
        raise ValueError("root_path must be an absolute USD path")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ValueError("max_depth must be an integer")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if not 0 <= max_depth <= 8:
        raise ValueError("max_depth must be between 0 and 8")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    root_json = json.dumps(root_path)
    return f"""
import json
import omni.usd

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
root_path = {root_json}
prefix = root_path.rstrip("/") + "/"
root_depth = root_path.rstrip("/").count("/")
items = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path != root_path and not path.startswith(prefix):
        continue
    relative_depth = path.count("/") - root_depth
    if relative_depth > {max_depth!r}:
        continue
    items.append({{"path": path, "type": prim.GetTypeName()}})
    if len(items) >= {limit!r}:
        break
print(json.dumps({{"root_path": root_path, "items": items, "truncated": len(items) >= {limit!r}}}))
""".strip()


def _capture_viewport_source(arguments: dict[str, Any]) -> str:
    filename = arguments.get("filename", "codex_poc.png")
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    if (
        not filename
        or filename != Path(filename).name
        or not filename.lower().endswith(".png")
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in filename)
    ):
        raise ValueError(
            "filename must be a simple .png basename containing only letters, "
            "numbers, dot, underscore, or hyphen"
        )

    capture_root = Path(
        os.environ.get("ISAAC_SIM_CAPTURE_ROOT", str(DEFAULT_CAPTURE_ROOT))
    ).resolve()
    capture_root.mkdir(parents=True, exist_ok=True)
    output_path = (capture_root / filename).resolve()
    if output_path.parent != capture_root:
        raise ValueError("capture output escaped the configured capture root")
    output_json = json.dumps(str(output_path))
    return f"""
import json
import os
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

output_path = {output_json}
viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active Isaac Sim viewport is available")
capture = capture_viewport_to_file(viewport, file_path=output_path)
captured_aovs = await capture.wait_for_result(completion_frames=30)
print(json.dumps({{
    "output_path": output_path,
    "exists": os.path.isfile(output_path),
    "captured_aovs": [str(item) for item in (captured_aovs or [])],
}}))
""".strip()


class IsaacMcpServer:
    def __init__(self, client: IsaacTcpClient | None = None) -> None:
        self.client = client or IsaacTcpClient.from_environment()

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        arguments = arguments or {}
        try:
            if name == "isaac_ping":
                source = "1 + 1"
            elif name == "isaac_get_status":
                source = _timeline_state_source()
            elif name == "isaac_set_frame_rate_limit":
                source = _frame_rate_limit_source(arguments)
            elif name == "isaac_set_markings_enabled":
                source = _set_markings_enabled_source(arguments)
            elif name == "isaac_set_presentation_overlays_enabled":
                source = _set_presentation_overlays_enabled_source(arguments)
            elif name == "isaac_save_stage_checkpoint":
                source = _save_stage_checkpoint_source(arguments)
            elif name == "isaac_load_stage_checkpoint":
                source = _load_stage_checkpoint_source(arguments)
            elif name == "isaac_set_lightweight_warehouse_wall_height":
                source = _set_lightweight_warehouse_wall_height_source(arguments)
            elif name == "isaac_enable_edge_ai_supervisor_ui":
                source = _enable_edge_ai_supervisor_ui_source()
            elif name == "isaac_enable_conveyor_safety":
                source = enable_conveyor_safety_extension_source()
            elif name == "isaac_create_conveyor_safety_scene":
                source = create_conveyor_safety_scene_source(arguments)
            elif name == "isaac_capture_conveyor_safety_camera":
                source = capture_conveyor_safety_camera_source(arguments)
            elif name == "isaac_set_conveyor_safety_light":
                source = set_conveyor_safety_light_source(arguments)
            elif name == "isaac_get_conveyor_safety_state":
                source = get_conveyor_safety_state_source()
            elif name == "isaac_set_conveyor_safety_forklift_pose":
                source = set_conveyor_safety_forklift_pose_source(arguments)
            elif name == "isaac_set_conveyor_safety_camera":
                source = set_conveyor_safety_camera_source(arguments)
            elif name == "isaac_create_live_aisle_navigation":
                source = create_live_aisle_source(arguments)
            elif name == "isaac_configure_live_blind_corner":
                source = configure_blind_corner_source(arguments)
            elif name == "isaac_stock_live_blind_corner_shelves":
                source = stock_blind_corner_shelves_source(arguments)
            elif name == "isaac_configure_live_robot_pov":
                source = configure_robot_pov_source(arguments)
            elif name == "isaac_configure_live_navigation_graph":
                source = configure_navigation_graph_source()
            elif name == "isaac_configure_live_route_visualizations":
                source = configure_route_visualizations_source()
            elif name == "isaac_set_live_aisle_congestion":
                source = set_live_aisle_congestion_source(arguments)
            elif name == "isaac_set_live_forklift_visible":
                source = set_forklift_visible_source(arguments)
            elif name == "isaac_apply_live_aisle_advisory":
                source = apply_live_aisle_advisory_source(arguments)
            elif name == "isaac_tick_live_aisle":
                source = tick_live_aisle_source()
            elif name == "isaac_set_live_aisle_request_status":
                source = set_live_aisle_request_status_source(arguments)
            elif name == "isaac_capture_live_aisle_camera_grid":
                source = capture_live_camera_grid_source(arguments)
            elif name == "isaac_capture_live_aisle_roof_frame":
                source = capture_live_roof_frame_source(arguments)
            elif name == "isaac_capture_live_aisle_tactical_frame":
                source = capture_live_tactical_frame_source(arguments)
            elif name == "isaac_capture_live_robot_front_frame":
                source = capture_live_robot_front_frame_source(arguments)
            elif name == "isaac_create_poc_scene":
                source = _create_poc_scene_source()
            elif name == "isaac_set_marker_pose":
                source = _set_marker_source(arguments)
            elif name == "isaac_set_timeline":
                source = _set_timeline_source(arguments)
            elif name == "isaac_list_prims":
                source = _list_prims_source(arguments)
            elif name == "isaac_capture_viewport":
                source = _capture_viewport_source(arguments)
            elif name == "isaac_get_edge_supervisor_assets":
                source = asset_status_source()
            elif name == "isaac_create_edge_supervisor_scene":
                source = create_edge_supervisor_scene_source()
            elif name == "isaac_create_realistic_edge_supervisor_scene":
                source = create_realistic_scene_source()
            elif name == "isaac_create_lightweight_live_warehouse":
                source = create_lightweight_live_scene_source()
            elif name == "isaac_inspect_realistic_warehouse":
                source = inspect_realistic_warehouse_source()
            elif name == "isaac_set_edge_supervisor_progress":
                source = set_edge_supervisor_progress_source(arguments)
            elif name == "isaac_capture_edge_supervisor_cameras":
                source = capture_camera_views_source(arguments)
            elif name == "isaac_capture_edge_supervisor_sequence":
                source = capture_sequence_source(arguments)
            elif name == "isaac_capture_realistic_edge_supervisor_cameras":
                source = capture_realistic_camera_views_source(arguments)
            elif name == "isaac_capture_realistic_edge_supervisor_sequence":
                source = capture_realistic_sequence_source(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
            return _pretty_isaac_response(self.client.execute(source))
        except (IsaacConnectionError, ValueError) as error:
            return {
                "content": [{"type": "text", "text": str(error)}],
                "isError": True,
            }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            return None

        try:
            if method == "initialize":
                params = request.get("params") or {}
                protocol_version = params.get("protocolVersion", "2024-11-05")
                result = {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                    "instructions": (
                        "Use isaac_ping first. Tools control only the running local "
                        "Isaac Sim instance and keep PoC scene edits under "
                        "/World/CodexPoC."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                name = params.get("name")
                arguments = params.get("arguments")
                if not isinstance(name, str):
                    raise ValueError("tools/call requires a string tool name")
                if arguments is not None and not isinstance(arguments, dict):
                    raise ValueError("tools/call arguments must be an object")
                result = self.call_tool(name, arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
        except (TypeError, ValueError) as error:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": str(error)},
            }

        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def serve_stdio(self) -> None:
        for raw_line in sys.stdin.buffer:
            if not raw_line.strip():
                continue
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise ValueError("JSON-RPC message must be an object")
                response = self.handle_request(request)
            except (json.JSONDecodeError, ValueError) as error:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": str(error)},
                }
            if response is not None:
                encoded = json.dumps(
                    response,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                sys.stdout.write(encoded + "\n")
                sys.stdout.flush()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose a running Isaac Sim instance as local MCP tools."
    )
    parser.add_argument(
        "--check-isaac",
        action="store_true",
        help="Check port 8226 directly and exit instead of starting MCP stdio.",
    )
    parser.add_argument(
        "--tool",
        choices=[tool["name"] for tool in TOOLS],
        help="Call one bounded tool directly instead of starting MCP stdio.",
    )
    parser.add_argument(
        "--arguments",
        default="{}",
        help="JSON object passed to --tool, default {}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    server = IsaacMcpServer()
    if args.check_isaac:
        try:
            print(
                json.dumps(
                    server.client.execute("1 + 1"),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except IsaacConnectionError as error:
            print(str(error), file=sys.stderr)
            return 1
        return 0

    if args.tool:
        try:
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("--arguments must decode to a JSON object")
        except (json.JSONDecodeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        result = server.call_tool(args.tool, arguments)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result.get("isError") else 0

    server.serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
