from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


SCENE_ROOT = "/World/CodexPoC"
LIVE_ROOT = f"{SCENE_ROOT}/LiveAisle"
LIVE_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "isaac_sim_live_aisle"
    / "frames"
)
NAV_ASSET_PATH = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
TRADITIONAL_FORKLIFT_ASSET_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Props/Forklift/forklift.usd"
)
CARDBOX_ASSET_PATH = (
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd"
)
ROOF_CAMERA_PATH = f"{SCENE_ROOT}/Sensors/RoofOverview"
TACTICAL_CAMERA_PATH = f"{SCENE_ROOT}/Sensors/BlindCornerTactical"
APPROACH_CAMERA_PATH = f"{SCENE_ROOT}/Sensors/WestApproachTactical"
NORTH_CAMERA_PATH = f"{SCENE_ROOT}/Sensors/NorthBypassTactical"
SUPERVISOR_CLEARANCE_ROW_PATHS = (
    f"{SCENE_ROOT}/AisleLayout/Row0",
    f"{SCENE_ROOT}/AisleLayout/Row1",
)
VISUAL_MARKER_ROOT = f"{SCENE_ROOT}/NvidiaProps"
VISUAL_MARKER_PREFIXES = ("TrafficCone_",)
VISUAL_OVERLAY_PATHS = (
    f"{SCENE_ROOT}/Markings",
    f"{SCENE_ROOT}/Coverage",
    f"{SCENE_ROOT}/Supervisor",
    f"{SCENE_ROOT}/Actors/ForkliftOrange/IdentityPad",
    f"{SCENE_ROOT}/Actors/ForkliftOrange/CodexAmberBeacon",
)
SECURITY_CAMERA_MATRIX = (
    (0.9056236985, 0.4239171719, 0.0145309128, 0.0),
    (-0.0681735826, 0.1116587737, 0.9913720253, 0.0),
    (0.4186415532, -0.8988101396, 0.1300221604, 0.0),
    (7.0738054009, -4.1019809722, 1.7340605748, 1.0),
)
SECURITY_CAMERA_FOCAL_LENGTH = 24.0
SECURITY_CAMERA_HORIZONTAL_APERTURE = 30.0
SECURITY_CAMERA_VERTICAL_APERTURE = 16.875
SECURITY_CAMERA_POLICY = "user_authored_two_passage_full_view"
ROBOT_FRONT_CAMERA_PATH = (
    f"{LIVE_ROOT}/RobotBlue/chassis_link/sensors/CodexFrontPOV"
)
ROBOT_FRONT_CAPTURE_PROXY_PATH = (
    f"{SCENE_ROOT}/Sensors/RobotFrontCaptureProxy"
)
UNUSED_SUPERVISOR_CAMERA_PATHS = (
    ROOF_CAMERA_PATH,
    f"{SCENE_ROOT}/Sensors/WestAisle",
    f"{SCENE_ROOT}/Sensors/Intersection",
    f"{SCENE_ROOT}/Sensors/EastAisle",
    APPROACH_CAMERA_PATH,
    NORTH_CAMERA_PATH,
    ROBOT_FRONT_CAPTURE_PROXY_PATH,
)
ROBOT_SOURCE_CAMERA_PATH = (
    f"{LIVE_ROOT}/RobotBlue/chassis_link/sensors/front_owl/camera"
)
CODEX_NAV_GRAPH_PATH = f"{LIVE_ROOT}/RobotBlue/CodexNavigationGraph"
ROBOT_START = (-3.5, 1.6, 0.0)

# Both policies terminate at the same loading destination on the far side of
# the stocked central rack. The baseline reaches it through the right passage;
# the bypass goes through the left passage and converges behind the forklift's
# final blocking pose. This makes the destination visually stable before and
# after intervention.
LEFT_PASSAGE_CENTER_X = 0.0
RIGHT_PASSAGE_CENTER_X = 5.94
COMMON_LOADING_DESTINATION = (RIGHT_PASSAGE_CENTER_X, 6.2, 0.0)
ORIGINAL_ROUTE_WAYPOINTS = (
    (RIGHT_PASSAGE_CENTER_X, 1.6, 0.0),
    COMMON_LOADING_DESTINATION,
)
CLEAR_CONTROL_ROUTE_WAYPOINTS = ORIGINAL_ROUTE_WAYPOINTS
DIRECT_GOAL = ORIGINAL_ROUTE_WAYPOINTS[-1]
NORTH_LOADING_GOAL = COMMON_LOADING_DESTINATION

# The traditional forklift begins behind the central stocked rack, drives east
# toward the right passage, then turns clockwise and stops across it.
BLIND_CORNER_START = (2.2, 4.835, 0.0)
BLIND_CORNER_TURN = (5.20, 4.835, 0.0)
BLIND_CORNER_END = (RIGHT_PASSAGE_CENTER_X, 3.60, 0.0)
BLIND_CORNER_SPEED_MPS = 0.8
FORKLIFT_START_YAW_DEGREES = 90.0
FORKLIFT_SOUTH_YAW_DEGREES = 0.0
SOUTH_BYPASS_WAYPOINTS = (
    (0.0, 1.6, 0.0),
    (0.0, -1.6, 0.0),
    (5.6, -1.6, 0.0),
    (5.6, 1.6, 0.0),
    DIRECT_GOAL,
)
NORTH_BYPASS_WAYPOINTS = (
    (LEFT_PASSAGE_CENTER_X, 1.6, 0.0),
    (LEFT_PASSAGE_CENTER_X, 5.8, 0.0),
    COMMON_LOADING_DESTINATION,
)


def _waypoint_headings(
    start: tuple[float, float, float],
    waypoints: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    headings: list[float] = []
    previous = start
    for waypoint in waypoints:
        headings.append(
            math.atan2(
                waypoint[1] - previous[1],
                waypoint[0] - previous[0],
            )
        )
        previous = waypoint
    return tuple(headings)


ROUTE_HEADINGS = {
    "direct": _waypoint_headings(ROBOT_START, ORIGINAL_ROUTE_WAYPOINTS),
    "south_bypass": _waypoint_headings(ROBOT_START, SOUTH_BYPASS_WAYPOINTS),
    "north_bypass": _waypoint_headings(ROBOT_START, NORTH_BYPASS_WAYPOINTS),
}


def _float_argument(
    arguments: dict[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}")
    return number


def _bool_argument(
    arguments: dict[str, Any],
    name: str,
    default: bool,
) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def create_live_aisle_source(arguments: dict[str, Any]) -> str:
    """Create one navigated Carter and one forklift obstacle in the warehouse."""
    robot_width = _float_argument(arguments, "robot_width", 1.42, 1.1, 1.8)
    robot_length = _float_argument(arguments, "robot_length", 2.1, 1.5, 2.8)
    robot_scale = _float_argument(arguments, "robot_scale", 1.75, 1.0, 2.5)
    robot_height_scale = 1.0 + (robot_scale - 1.0) * 0.25
    start_playing = _bool_argument(arguments, "start_playing", True)
    forklift_start_mode = arguments.get("forklift_start_mode", "current")
    if forklift_start_mode not in ("current", "configured"):
        raise ValueError(
            "forklift_start_mode must be current or configured"
        )
    motion_controller = arguments.get("motion_controller", "isaac_graph")
    if motion_controller not in ("isaac_graph", "kinematic_waypoint"):
        raise ValueError(
            "motion_controller must be isaac_graph or kinematic_waypoint"
        )
    scenario_mode = arguments.get("scenario_mode", "blind_corner")
    if scenario_mode not in ("blind_corner", "clear_route_control"):
        raise ValueError(
            "scenario_mode must be blind_corner or clear_route_control"
        )
    direct_waypoints = (
        CLEAR_CONTROL_ROUTE_WAYPOINTS
        if scenario_mode == "clear_route_control"
        else ORIGINAL_ROUTE_WAYPOINTS
    )
    direct_headings = _waypoint_headings(ROBOT_START, direct_waypoints)
    navigation_intention = (
        "RobotBlue follows the baseline route through the right passage to "
        "the far-side loading destination"
        if scenario_mode == "clear_route_control"
        else
        "RobotBlue follows the baseline route through the right passage to "
        "the far-side loading destination"
    )
    kinematic_speed_mps = _float_argument(
        arguments,
        "kinematic_speed_mps",
        0.18,
        0.1,
        0.8,
    )
    live_root = LIVE_ROOT
    nav_asset = NAV_ASSET_PATH
    direct_goal = direct_waypoints[-1]
    return f'''
import json
import math
import carb
import omni.graph.core as og
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_full_asset_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ROOT = {SCENE_ROOT!r}
LIVE_ROOT = {live_root!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
if not stage.GetPrimAtPath(ROOT).IsValid():
    raise RuntimeError("Create the realistic edge-supervisor warehouse first")

timeline = omni.timeline.get_timeline_interface()
timeline.stop()
carb.settings.get_settings().set_bool(
    "/app/omni.graph.scriptnode/opt_in",
    True,
)
if stage.GetPrimAtPath(LIVE_ROOT).IsValid():
    stage.RemovePrim(LIVE_ROOT)
UsdGeom.Xform.Define(stage, LIVE_ROOT)

# Preserve the scripted actors in the stage but hide them while the live
# navigation actors are active.
for actor_name in ("RobotBlue", "RobotGreen", "WorkerYellow"):
    actor = stage.GetPrimAtPath(f"{{ROOT}}/Actors/{{actor_name}}")
    if actor.IsValid():
        actor.SetActive(False)

# Remove the unrelated middle rack row from the southwest supervisor's
# sightline. The carton-filled Row2/Segment1 blind-corner rack remains
# complete, so RobotBlue's low forward camera still cannot see the parked
# southbound forklift before it emerges around the corner.
inactive_supervisor_rows = []
for row_path in {SUPERVISOR_CLEARANCE_ROW_PATHS!r}:
    supervisor_clearance_row = stage.GetPrimAtPath(row_path)
    if not supervisor_clearance_row.IsValid():
        # The clean tutorial checkpoint may already have this irrelevant row
        # removed rather than merely deactivated.
        continue
    supervisor_clearance_row.SetActive(False)
    inactive_supervisor_rows.append(row_path)

# The cone props were useful while laying out the scenario, but read as
# additional traffic actors in both the presentation and machine-vision
# frames. Keep them in the source warehouse and suppress them only for the
# live demonstration.
inactive_visual_markers = []
visual_marker_root = stage.GetPrimAtPath({VISUAL_MARKER_ROOT!r})
if visual_marker_root.IsValid():
    for marker in visual_marker_root.GetChildren():
        if marker.GetName().startswith({VISUAL_MARKER_PREFIXES!r}):
            marker.SetActive(False)
            inactive_visual_markers.append(str(marker.GetPath()))

# Suppress the tutorial-only floor lanes, camera coverage brackets, conflict
# zone, and legacy colored route overlays. These remain authored in the source
# checkpoint for reference but are not actors or physical warehouse fixtures.
inactive_visual_overlays = []
for overlay_path in {VISUAL_OVERLAY_PATHS!r}:
    overlay = stage.GetPrimAtPath(overlay_path)
    if overlay.IsValid():
        overlay.SetActive(False)
        inactive_visual_overlays.append(overlay_path)

# Remove two orphaned cartons left on the floor during layout experiments.
removed_orphan_props = []
for orphan_name in ("Cardbox_0", "Cardbox_2"):
    orphan_path = f"{{ROOT}}/NvidiaProps/{{orphan_name}}"
    if stage.GetPrimAtPath(orphan_path).IsValid():
        stage.RemovePrim(orphan_path)
        removed_orphan_props.append(orphan_path)

# Mirror the user's densely stocked Segment1 contents into adjacent Segment0.
# This includes every manually stacked carton plus the vertical pallet wall at
# floor level, closing RobotBlue's residual view of a parked forklift through
# the otherwise empty rack. Internal references preserve the exact authored
# assets and transforms without duplicating mesh data.
adjacent_stock_root_path = f"{{ROOT}}/AdjacentShelfStock"
if stage.GetPrimAtPath(adjacent_stock_root_path).IsValid():
    stage.RemovePrim(adjacent_stock_root_path)
adjacent_stock_root = UsdGeom.Xform.Define(
    stage,
    adjacent_stock_root_path,
).GetPrim()
UsdGeom.XformCommonAPI(adjacent_stock_root).SetTranslate(
    Gf.Vec3d(-6.0, 0.0, 0.0)
)
adjacent_stock_items = []
source_carton_root = stage.GetPrimAtPath(
    f"{{ROOT}}/BlindCornerShelfStock"
)
if not source_carton_root.IsValid():
    raise RuntimeError("The densely stocked blind-corner rack is missing")
for source_carton in source_carton_root.GetChildren():
    destination = stage.DefinePrim(
        f"{{adjacent_stock_root_path}}/Cartons/"
        f"{{source_carton.GetName()}}",
        source_carton.GetTypeName() or "Xform",
    )
    destination.GetReferences().AddInternalReference(
        source_carton.GetPath()
    )
    adjacent_stock_items.append(str(destination.GetPath()))
for source_pallet_name in (
    "Pallet_1",
    "Pallet_02",
    "Pallet_03",
    "Pallet_04",
    "Pallet_05",
):
    source_pallet = stage.GetPrimAtPath(
        f"{{ROOT}}/NvidiaProps/{{source_pallet_name}}"
    )
    if not source_pallet.IsValid():
        raise RuntimeError(
            f"The populated rack pallet is missing: {{source_pallet_name}}"
        )
    destination = stage.DefinePrim(
        f"{{adjacent_stock_root_path}}/BottomPallets/"
        f"{{source_pallet_name}}",
        source_pallet.GetTypeName() or "Xform",
    )
    destination.GetReferences().AddInternalReference(
        source_pallet.GetPath()
    )
    adjacent_stock_items.append(str(destination.GetPath()))

forklift_path = f"{{ROOT}}/Actors/ForkliftOrange"
source_forklift = stage.GetPrimAtPath(forklift_path)
if not source_forklift.IsValid():
    raise RuntimeError("The realistic warehouse forklift actor is missing")
source_translate = source_forklift.GetAttribute("xformOp:translate")
current_forklift_start = (
    source_translate.Get() if source_translate else None
) or Gf.Vec3d(
    *{BLIND_CORNER_START!r}
)
configured_start_attr = source_forklift.GetAttribute(
    "codex:blindCornerStartWorld"
)
if (
    {forklift_start_mode!r} == "configured"
    and configured_start_attr
    and configured_start_attr.HasAuthoredValueOpinion()
):
    forklift_start = configured_start_attr.Get()
else:
    # The default mode deliberately treats the current stage transform as the
    # author's start pose. This preserves GUI positioning instead of silently
    # replacing it with an old hard-coded aisle coordinate.
    forklift_start = current_forklift_start
configured_end_attr = source_forklift.GetAttribute(
    "codex:blindCornerEndWorld"
)
forklift_end = (
    configured_end_attr.Get()
    if configured_end_attr
    and configured_end_attr.HasAuthoredValueOpinion()
    else Gf.Vec3d(*{BLIND_CORNER_END!r})
)
configured_speed_attr = source_forklift.GetAttribute(
    "codex:blindCornerSpeedMps"
)
forklift_speed = (
    configured_speed_attr.Get()
    if configured_speed_attr
    and configured_speed_attr.HasAuthoredValueOpinion()
    else {BLIND_CORNER_SPEED_MPS!r}
)

# Replace the autonomous Forklift B reference with NVIDIA's traditional
# counterbalance forklift prop while retaining the actor path and authored
# blind-corner trajectory.
stage.RemovePrim(forklift_path)
forklift = add_reference_to_stage(
    usd_path={TRADITIONAL_FORKLIFT_ASSET_URL!r},
    prim_path=forklift_path,
)
forklift.SetActive(True)
forklift_translate = forklift.GetAttribute("xformOp:translate")
if not forklift_translate:
    forklift_translate = UsdGeom.Xformable(forklift).AddTranslateOp().GetAttr()
forklift_translate.Set(forklift_start)
forklift_rotate = forklift.GetAttribute("xformOp:rotateXYZ")
if not forklift_rotate:
    forklift_rotate = UsdGeom.Xformable(forklift).AddRotateXYZOp().GetAttr()
# The prop's forks point toward -Y at zero yaw. At the hidden start, rotate it
# east so it can approach the right passage before turning clockwise.
forklift_rotate.Set(
    Gf.Vec3f(0.0, 0.0, {FORKLIFT_START_YAW_DEGREES!r})
)
for name, type_name, value in (
    (
        "codex:blindCornerStartWorld",
        Sdf.ValueTypeNames.Double3,
        forklift_start,
    ),
    (
        "codex:blindCornerEndWorld",
        Sdf.ValueTypeNames.Double3,
        forklift_end,
    ),
    (
        "codex:blindCornerSpeedMps",
        Sdf.ValueTypeNames.Double,
        forklift_speed,
    ),
):
    attr = forklift.GetAttribute(name)
    if not attr:
        attr = forklift.CreateAttribute(name, type_name)
    attr.Set(value)

asset_url = get_full_asset_path({nav_asset!r})
if not asset_url:
    raise RuntimeError("The packaged Nova Carter navigation asset did not resolve")

robot_specs = {{
    "RobotBlue": {ROBOT_START!r},
}}
for robot_name, start in robot_specs.items():
    robot_path = f"{{LIVE_ROOT}}/{{robot_name}}"
    robot_prim = add_reference_to_stage(usd_path=asset_url, prim_path=robot_path)
    # Move the complete referenced articulation at its common root. Offsetting
    # rigid links independently distorts the packaged joint frames and makes
    # the robot drift or collapse as soon as PhysX starts.
    translate_attr = robot_prim.GetAttribute("xformOp:translate")
    if not translate_attr:
        translate_attr = UsdGeom.Xformable(
            robot_prim
        ).AddTranslateOp().GetAttr()
    translate_attr.Set(Gf.Vec3d(*start))

stable_frames = 0
for _ in range(900):
    loading_status = omni.usd.get_context().get_stage_loading_status()
    stable_frames = stable_frames + 1 if not any(loading_status) else 0
    if stable_frames >= 30:
        break
    await omni.kit.app.get_app().next_update_async()

# Re-assert the traditional forklift root transform after reference
# composition settles, then disable any optional physics bodies. The current
# official prop is static, but this keeps the wrapper robust to asset updates.
forklift = stage.GetPrimAtPath(forklift_path)
forklift_translate = forklift.GetAttribute("xformOp:translate")
forklift_translate.Set(forklift_start)
forklift.GetAttribute("xformOp:rotateXYZ").Set(
    Gf.Vec3f(0.0, 0.0, {FORKLIFT_START_YAW_DEGREES!r})
)
disabled_forklift_bodies = 0
for descendant in Usd.PrimRange(forklift):
    if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body = UsdPhysics.RigidBodyAPI(descendant)
        enabled_attr = rigid_body.GetRigidBodyEnabledAttr()
        if not enabled_attr:
            enabled_attr = rigid_body.CreateRigidBodyEnabledAttr()
        enabled_attr.Set(False)
        disabled_forklift_bodies += 1

# Re-author the articulation-root placement after reference composition has
# settled. Some packaged assets author their root xformOpOrder late while
# loading, which can otherwise leave the Carter at the asset origin.
for robot_name, start in robot_specs.items():
    robot_prim = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/{{robot_name}}")
    if not robot_prim.IsValid():
        raise RuntimeError(f"Navigation prim did not load for {{robot_name}}")
    translate_attr = robot_prim.GetAttribute("xformOp:translate")
    if not translate_attr:
        translate_attr = UsdGeom.Xformable(
            robot_prim
        ).AddTranslateOp().GetAttr()
    translate_attr.Set(Gf.Vec3d(*start))

def set_target_world(robot_name, world_xyz, target_yaw=0.0):
    robot_path = f"{{LIVE_ROOT}}/{{robot_name}}"
    robot_prim = stage.GetPrimAtPath(robot_path)
    target = stage.GetPrimAtPath(f"{{robot_path}}/targetXform")
    if not robot_prim.IsValid() or not target.IsValid():
        raise RuntimeError(f"Navigation prims did not load for {{robot_name}}")
    attr = target.GetAttribute("xformOp:translate")
    if not attr:
        attr = UsdGeom.Xformable(target).AddTranslateOp().GetAttr()
    # The packaged navigation graph deliberately treats targetXform's authored
    # translation as a stage/world target even though the prim is nested under
    # the robot reference.
    attr.Set(Gf.Vec3d(*world_xyz))
    for planner_path in (
        f"{{robot_path}}/CodexNavigationGraph/Planner",
        f"{{robot_path}}/ActionGraph/quintic_path_planner_01",
    ):
        planner = stage.GetPrimAtPath(planner_path)
        if planner.IsValid():
            planner.GetAttribute("inputs:targetPosition").Set(
                Gf.Vec3d(*world_xyz)
            )
            planner.GetAttribute("inputs:targetOrientation").Set(
                Gf.Quatd(
                    math.cos(target_yaw * 0.5),
                    Gf.Vec3d(0.0, 0.0, math.sin(target_yaw * 0.5)),
                )
            )

for robot_name, start in robot_specs.items():
    chassis_path = f"{{LIVE_ROOT}}/{{robot_name}}/chassis_link"
    chassis = stage.GetPrimAtPath(chassis_path)
    if not chassis.IsValid():
        raise RuntimeError(f"Nova Carter chassis did not load for {{robot_name}}")
    # Keep every packaged rigid-link transform unchanged relative to the
    # articulation root. This preserves wheel/caster placement and joint
    # geometry while the root transform places the complete robot at the gate.
    link_names = (
        "chassis_link",
        "wheel_left",
        "wheel_right",
        "caster_frame_base",
        "caster_swivel_left",
        "caster_swivel_right",
        "caster_wheel_left",
        "caster_wheel_right",
    )
    for link_name in link_names:
        link = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/{{robot_name}}/{{link_name}}")
        if not link.IsValid():
            raise RuntimeError(f"Nova Carter link did not load: {{link_name}}")
    # Enlarge the official Carter visuals without scaling the articulation
    # links or joint frames. Uniformly scaling chassis_link itself changes the
    # wheel physics and prevents the packaged navigation graph from moving.
    # Keep wheels and casters at their authored scale so their visual contact
    # patches remain aligned with the native articulation and the floor.
    visual_paths = [f"{{chassis_path}}/visual"]
    scaled_visuals = 0
    for visual_path in visual_paths:
        visual = stage.GetPrimAtPath(visual_path)
        if not visual.IsValid():
            continue
        scale_attr = visual.GetAttribute("xformOp:scale")
        if not scale_attr:
            scale_attr = UsdGeom.Xformable(visual).AddScaleOp().GetAttr()
        scale_value = (
            Gf.Vec3f(
                {robot_scale!r},
                {robot_scale!r},
                {robot_height_scale!r},
            )
            if scale_attr.GetTypeName() == Sdf.ValueTypeNames.Float3
            else Gf.Vec3d(
                {robot_scale!r},
                {robot_scale!r},
                {robot_height_scale!r},
            )
        )
        if scale_attr.Set(scale_value):
            scaled_visuals += 1

kinematic_robot_bodies = 0
if {motion_controller!r} == "kinematic_waypoint":
    for descendant in Usd.PrimRange(
        stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue")
    ):
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI(descendant)
            enabled_attr = rigid_body.GetRigidBodyEnabledAttr()
            if not enabled_attr:
                enabled_attr = rigid_body.CreateRigidBodyEnabledAttr()
            enabled_attr.Set(False)
            kinematic_robot_bodies += 1
    packaged_graph = stage.GetPrimAtPath(
        f"{{LIVE_ROOT}}/RobotBlue/ActionGraph"
    )
    if packaged_graph.IsValid():
        packaged_runtime_graph = og.Controller.graph(
            f"{{LIVE_ROOT}}/RobotBlue/ActionGraph"
        )
        if packaged_runtime_graph:
            packaged_runtime_graph.set_disabled(True)
        packaged_graph.SetActive(False)

set_target_world(
    "RobotBlue",
    {direct_waypoints[0]!r},
    {direct_headings[0]!r},
)

state = UsdGeom.Xform.Define(stage, f"{{LIVE_ROOT}}/State").GetPrim()
def set_state(name, type_name, value):
    attr = state.GetAttribute(name)
    if not attr:
        attr = state.CreateAttribute(name, type_name)
    attr.Set(value)

set_state("codex:blueAction", Sdf.ValueTypeNames.String, "CONTINUE")
set_state("codex:blueRoute", Sdf.ValueTypeNames.String, "direct")
set_state(
    "codex:navigationIntention",
    Sdf.ValueTypeNames.String,
    {navigation_intention!r},
)
set_state("codex:waypointIndex", Sdf.ValueTypeNames.Int, 0)
set_state(
    "codex:scenarioMode",
    Sdf.ValueTypeNames.String,
    {scenario_mode!r},
)
set_state(
    "codex:directRouteWaypoints",
    Sdf.ValueTypeNames.Double3Array,
    [Gf.Vec3d(*value) for value in {direct_waypoints!r}],
)
set_state(
    "codex:directRouteHeadings",
    Sdf.ValueTypeNames.DoubleArray,
    list({direct_headings!r}),
)
set_state("codex:obstacleReleased", Sdf.ValueTypeNames.Bool, False)
set_state(
    "codex:motionController",
    Sdf.ValueTypeNames.String,
    {motion_controller!r},
)
set_state(
    "codex:kinematicSpeedMps",
    Sdf.ValueTypeNames.Double,
    {kinematic_speed_mps!r},
)
set_state(
    "codex:forkliftStartWorld",
    Sdf.ValueTypeNames.Double3,
    forklift_start,
)
set_state(
    "codex:forkliftEndWorld",
    Sdf.ValueTypeNames.Double3,
    forklift_end,
)
set_state(
    "codex:forkliftTurnWorld",
    Sdf.ValueTypeNames.Double3,
    Gf.Vec3d(*{BLIND_CORNER_TURN!r}),
)
set_state(
    "codex:forkliftSpeedMps",
    Sdf.ValueTypeNames.Double,
    forklift_speed,
)
set_state(
    "codex:forkliftYawDegrees",
    Sdf.ValueTypeNames.Double,
    {FORKLIFT_START_YAW_DEGREES!r},
)
set_state(
    "codex:forkliftEndYawDegrees",
    Sdf.ValueTypeNames.Double,
    {FORKLIFT_SOUTH_YAW_DEGREES!r},
)
set_state(
    "codex:forkliftMotionPhase",
    Sdf.ValueTypeNames.String,
    "holding_blind_corner",
)
set_state(
    "codex:lastTickTimelineSeconds",
    Sdf.ValueTypeNames.Double,
    timeline.get_current_time(),
)
set_state("codex:evkRequestStatus", Sdf.ValueTypeNames.String, "idle")
set_state("codex:lastDecisionSource", Sdf.ValueTypeNames.String, "none")
set_state("codex:lastRequestId", Sdf.ValueTypeNames.String, "")
set_state("codex:lastLatencySeconds", Sdf.ValueTypeNames.Double, 0.0)
set_state(
    "codex:robotClearanceWidthMeters",
    Sdf.ValueTypeNames.Double,
    {robot_width!r},
)
set_state(
    "codex:robotClearanceLengthMeters",
    Sdf.ValueTypeNames.Double,
    {robot_length!r},
)

# A southwest wall-corner camera looks diagonally across RobotBlue's approach
# and the northeast blind corner. Removing only the unrelated middle rack row
# gives this external supervisor a clear view of RobotBlue, while the stocked
# northeast rack still hides the parked forklift. The forklift becomes visible
# only after moving south around the corner.
sensors_path = {f"{SCENE_ROOT}/Sensors"!r}
if not stage.GetPrimAtPath(sensors_path).IsValid():
    UsdGeom.Xform.Define(stage, sensors_path)
removed_unused_cameras = []
for unused_camera_path in {UNUSED_SUPERVISOR_CAMERA_PATHS!r}:
    unused_camera = stage.GetPrimAtPath(unused_camera_path)
    if unused_camera.IsValid() and unused_camera.GetTypeName() == "Camera":
        stage.RemovePrim(unused_camera_path)
        removed_unused_cameras.append(unused_camera_path)
tactical_camera = UsdGeom.Camera.Define(
    stage,
    {TACTICAL_CAMERA_PATH!r},
)
tactical_xform = UsdGeom.Xformable(tactical_camera.GetPrim())
tactical_xform.ClearXformOpOrder()
tactical_xform.AddTransformOp().Set(Gf.Matrix4d({SECURITY_CAMERA_MATRIX!r}))
tactical_camera.CreateFocalLengthAttr({SECURITY_CAMERA_FOCAL_LENGTH!r})
tactical_camera.CreateHorizontalApertureAttr(
    {SECURITY_CAMERA_HORIZONTAL_APERTURE!r}
)
tactical_camera.CreateVerticalApertureAttr(
    {SECURITY_CAMERA_VERTICAL_APERTURE!r}
)
tactical_camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))

# The original right corner is strongly shadowed. Add plausible overhead
# area lights so the official forklift remains legible without visual aids.
perception_root = f"{{LIVE_ROOT}}/PerceptionAids"
UsdGeom.Xform.Define(stage, perception_root)

# Favor consistent machine vision over dramatic contrast. Broaden the existing
# key, lift the ambient dome, and add two large ceiling fills so rack shadows
# retain texture instead of becoming featureless black regions.
dome = stage.GetPrimAtPath(f"{{ROOT}}/Lighting/Dome")
if dome.IsValid():
    dome.GetAttribute("inputs:intensity").Set(1800.0)
key = stage.GetPrimAtPath(f"{{ROOT}}/Lighting/Key")
if key.IsValid():
    key.GetAttribute("inputs:intensity").Set(800.0)
    key.GetAttribute("inputs:angle").Set(15.0)
ceiling_fills = []
for name, x_value in (("West", -3.5), ("East", 3.5)):
    ceiling_fill = UsdLux.RectLight.Define(
        stage,
        f"{{perception_root}}/CeilingFill{{name}}",
    )
    ceiling_fill.CreateIntensityAttr(800.0)
    ceiling_fill.CreateWidthAttr(7.0)
    ceiling_fill.CreateHeightAttr(5.0)
    ceiling_fill.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
    UsdGeom.XformCommonAPI(ceiling_fill.GetPrim()).SetTranslate(
        Gf.Vec3d(x_value, 0.8, 8.5)
    )
    ceiling_fills.append(str(ceiling_fill.GetPrim().GetPath()))
fill = UsdLux.SphereLight.Define(
    stage,
    f"{{perception_root}}/BlindCornerFill",
)
fill.CreateIntensityAttr(2600.0)
fill.CreateRadiusAttr(3.0)
fill.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.62))
UsdGeom.XformCommonAPI(fill.GetPrim()).SetTranslate(
    Gf.Vec3d(5.3, 3.3, 6.8)
)

# Remove the old perception-aid beacon and light. The official forklift
# silhouette must be the only forklift cue available to the visual model.
for beacon_artifact_path in (
    f"{{ROOT}}/Actors/ForkliftOrange/CodexAmberBeacon",
    f"{{ROOT}}/Actors/ForkliftOrange/CodexAmberBeaconLight",
):
    if stage.GetPrimAtPath(beacon_artifact_path).IsValid():
        stage.RemovePrim(beacon_artifact_path)
beacon_path = None

if {start_playing!r}:
    timeline.play()
    await omni.kit.app.get_app().next_update_async()

placement_cache = UsdGeom.XformCache()
robot_root = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue")
robot_chassis = stage.GetPrimAtPath(
    f"{{LIVE_ROOT}}/RobotBlue/chassis_link"
)
root_translate_attr = robot_root.GetAttribute("xformOp:translate")
root_translate_value = root_translate_attr.Get()
chassis_world = placement_cache.GetLocalToWorldTransform(
    robot_chassis
).ExtractTranslation()

print(json.dumps({{
    "created": True,
    "live_root": LIVE_ROOT,
    "navigation_asset": asset_url,
    "navigation": "packaged_nova_carter_target_omnigraph",
    "motion_controller": {motion_controller!r},
    "scenario_mode": {scenario_mode!r},
    "kinematic_robot_bodies_disabled": kinematic_robot_bodies,
    "kinematic_speed_mps": {kinematic_speed_mps!r},
    "robot_width_m": {robot_width!r},
    "robot_length_m": {robot_length!r},
    "robot_visual_xy_scale": {robot_scale!r},
    "robot_visual_z_scale": {robot_height_scale!r},
    "robot_root_translate": (
        [float(v) for v in root_translate_value]
        if root_translate_value is not None
        else None
    ),
    "robot_chassis_start_world": [float(v) for v in chassis_world],
    "scaled_robot_visual_branches": scaled_visuals,
    "articulation_scale": 1.0,
    "footprint_visual": False,
    "clearance_policy": "metadata_only",
    "obstacle": "official_traditional_forklift",
    "obstacle_asset": {TRADITIONAL_FORKLIFT_ASSET_URL!r},
    "obstacle_dimensions_m": [1.214, 3.495, 2.155],
    "obstacle_motion": "kinematic_wrapper",
    "forklift_start_mode": {forklift_start_mode!r},
    "forklift_start_world": [float(v) for v in forklift_start],
    "forklift_end_world": [float(v) for v in forklift_end],
    "forklift_speed_mps": forklift_speed,
    "forklift_start_yaw_degrees": {FORKLIFT_START_YAW_DEGREES!r},
    "forklift_end_yaw_degrees": {FORKLIFT_SOUTH_YAW_DEGREES!r},
    "disabled_forklift_rigid_bodies": disabled_forklift_bodies,
    "old_actors_preserved_but_inactive": True,
    "blue_goal_world": {direct_goal!r},
    "blue_original_route_waypoints": {direct_waypoints!r},
    "supervisor_cameras": {{
        "blind_corner_tactical": {TACTICAL_CAMERA_PATH!r},
    }},
    "removed_unused_supervisor_cameras": removed_unused_cameras,
    "tactical_camera_policy": {SECURITY_CAMERA_POLICY!r},
    "tactical_camera_focal_length": {SECURITY_CAMERA_FOCAL_LENGTH!r},
    "supervisor_camera_clearance": {{
        "inactive_rows": inactive_supervisor_rows,
        "reason": "remove_unrelated_foreground_occlusion",
    }},
    "inactive_visual_markers": inactive_visual_markers,
    "visual_marker_policy": "source_preserved_demo_suppressed",
    "inactive_visual_overlays": inactive_visual_overlays,
    "visual_overlay_policy": "routes_zones_and_markings_suppressed",
    "removed_orphan_props": removed_orphan_props,
    "blind_corner_stocked_rack_preserved": (
        {f"{SCENE_ROOT}/AisleLayout/Row2/Segment1"!r}
    ),
    "adjacent_stocked_rack": (
        {f"{SCENE_ROOT}/AisleLayout/Row2/Segment0"!r}
    ),
    "adjacent_stock_root": adjacent_stock_root_path,
    "adjacent_stock_item_count": len(adjacent_stock_items),
    "adjacent_stock_offset_m": [-6.0, 0.0, 0.0],
    "blind_corner_fill_light": f"{{perception_root}}/BlindCornerFill",
    "global_machine_vision_fills": ceiling_fills,
    "lighting_policy": "soft_global_fill_with_broadened_key",
    "forklift_safety_beacon": beacon_path,
    "obstacle_released": False,
    "timeline_playing": timeline.is_playing(),
}}))
'''.strip()


def configure_blind_corner_source(arguments: dict[str, Any]) -> str:
    """Author the repeatable approach-turn-block forklift trajectory."""
    start_x = _float_argument(
        arguments,
        "start_x",
        BLIND_CORNER_START[0],
        1.5,
        4.0,
    )
    start_y = _float_argument(
        arguments,
        "start_y",
        BLIND_CORNER_START[1],
        4.0,
        6.0,
    )
    turn_x = _float_argument(
        arguments,
        "turn_x",
        BLIND_CORNER_TURN[0],
        4.5,
        5.7,
    )
    turn_y = _float_argument(
        arguments,
        "turn_y",
        BLIND_CORNER_TURN[1],
        4.0,
        6.0,
    )
    end_x = _float_argument(
        arguments,
        "end_x",
        BLIND_CORNER_END[0],
        5.6,
        6.3,
    )
    end_y = _float_argument(
        arguments,
        "end_y",
        BLIND_CORNER_END[1],
        3.0,
        4.4,
    )
    speed = _float_argument(
        arguments,
        "speed_mps",
        BLIND_CORNER_SPEED_MPS,
        0.1,
        0.8,
    )
    return f'''
import json
import math
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
timeline = omni.timeline.get_timeline_interface()
timeline.stop()
forklift = stage.GetPrimAtPath({f"{SCENE_ROOT}/Actors/ForkliftOrange"!r})
if not forklift.IsValid():
    raise RuntimeError("The realistic warehouse forklift actor is missing")
forklift.SetActive(True)

start = Gf.Vec3d({start_x!r}, {start_y!r}, 0.0)
turn = Gf.Vec3d({turn_x!r}, {turn_y!r}, 0.0)
end = Gf.Vec3d({end_x!r}, {end_y!r}, 0.0)
translate = forklift.GetAttribute("xformOp:translate")
if not translate:
    translate = UsdGeom.Xformable(forklift).AddTranslateOp().GetAttr()
translate.Set(start)
rotate = forklift.GetAttribute("xformOp:rotateXYZ")
if not rotate:
    rotate = UsdGeom.Xformable(forklift).AddRotateXYZOp().GetAttr()
rotate.Set(Gf.Vec3f(0.0, 0.0, {FORKLIFT_START_YAW_DEGREES!r}))

def set_attr(prim, name, type_name, value):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)

set_attr(
    forklift,
    "codex:blindCornerStartWorld",
    Sdf.ValueTypeNames.Double3,
    start,
)
set_attr(
    forklift,
    "codex:blindCornerEndWorld",
    Sdf.ValueTypeNames.Double3,
    end,
)
set_attr(
    forklift,
    "codex:blindCornerTurnWorld",
    Sdf.ValueTypeNames.Double3,
    turn,
)
set_attr(
    forklift,
    "codex:blindCornerSpeedMps",
    Sdf.ValueTypeNames.Double,
    {speed!r},
)
set_attr(
    forklift,
    "codex:blindCornerYawDegrees",
    Sdf.ValueTypeNames.Double,
    {FORKLIFT_START_YAW_DEGREES!r},
)

state = stage.GetPrimAtPath({f"{LIVE_ROOT}/State"!r})
if state.IsValid():
    set_attr(
        state,
        "codex:forkliftStartWorld",
        Sdf.ValueTypeNames.Double3,
        start,
    )
    set_attr(
        state,
        "codex:forkliftEndWorld",
        Sdf.ValueTypeNames.Double3,
        end,
    )
    set_attr(
        state,
        "codex:forkliftTurnWorld",
        Sdf.ValueTypeNames.Double3,
        turn,
    )
    set_attr(
        state,
        "codex:forkliftSpeedMps",
        Sdf.ValueTypeNames.Double,
        {speed!r},
    )
    set_attr(
        state,
        "codex:forkliftYawDegrees",
        Sdf.ValueTypeNames.Double,
        {FORKLIFT_START_YAW_DEGREES!r},
    )
    set_attr(
        state,
        "codex:forkliftEndYawDegrees",
        Sdf.ValueTypeNames.Double,
        {FORKLIFT_SOUTH_YAW_DEGREES!r},
    )
    set_attr(
        state,
        "codex:forkliftMotionPhase",
        Sdf.ValueTypeNames.String,
        "holding_blind_corner",
    )
    obstacle = state.GetAttribute("codex:obstacleReleased")
    if obstacle:
        obstacle.Set(False)
    last_tick = state.GetAttribute("codex:lastTickTimelineSeconds")
    if last_tick:
        last_tick.Set(timeline.get_current_time())
    set_attr(
        state,
        "codex:directRouteWaypoints",
        Sdf.ValueTypeNames.Double3Array,
        [Gf.Vec3d(*value) for value in {ORIGINAL_ROUTE_WAYPOINTS!r}],
    )
    set_attr(
        state,
        "codex:directRouteHeadings",
        Sdf.ValueTypeNames.DoubleArray,
        list({_waypoint_headings(ROBOT_START, ORIGINAL_ROUTE_WAYPOINTS)!r}),
    )
    set_attr(
        state,
        "codex:blueAction",
        Sdf.ValueTypeNames.String,
        "CONTINUE",
    )
    set_attr(
        state,
        "codex:blueRoute",
        Sdf.ValueTypeNames.String,
        "direct",
    )
    set_attr(
        state,
        "codex:waypointIndex",
        Sdf.ValueTypeNames.Int,
        0,
    )
    set_attr(
        state,
        "codex:navigationIntention",
        Sdf.ValueTypeNames.String,
        "RobotBlue follows the baseline route through the right passage to "
        "the far-side loading destination",
    )

robot = stage.GetPrimAtPath({f"{LIVE_ROOT}/RobotBlue"!r})
target = stage.GetPrimAtPath({f"{LIVE_ROOT}/RobotBlue/targetXform"!r})
if robot.IsValid():
    robot_translate = robot.GetAttribute("xformOp:translate")
    if not robot_translate:
        robot_translate = UsdGeom.Xformable(robot).AddTranslateOp().GetAttr()
    robot_translate.Set(Gf.Vec3d(*{ROBOT_START!r}))
    robot_orient = robot.GetAttribute("xformOp:orient")
    if robot_orient:
        robot_orient.Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
if target.IsValid():
    target_translate = target.GetAttribute("xformOp:translate")
    if not target_translate:
        target_translate = UsdGeom.Xformable(target).AddTranslateOp().GetAttr()
    target_translate.Set(Gf.Vec3d(*{ORIGINAL_ROUTE_WAYPOINTS[0]!r}))
    for planner_path in (
        {f"{CODEX_NAV_GRAPH_PATH}/Planner"!r},
        {f"{LIVE_ROOT}/RobotBlue/ActionGraph/quintic_path_planner_01"!r},
    ):
        planner = stage.GetPrimAtPath(planner_path)
        if planner.IsValid():
            planner.GetAttribute("inputs:targetPosition").Set(
                Gf.Vec3d(*{ORIGINAL_ROUTE_WAYPOINTS[0]!r})
            )
            planner.GetAttribute("inputs:targetOrientation").Set(
                Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
            )

print(json.dumps({{
    "configured": True,
    "scenario": "two_passage_approach_turn_block",
    "forklift_start_world": [float(v) for v in start],
    "forklift_turn_world": [float(v) for v in turn],
    "forklift_end_world": [float(v) for v in end],
    "forklift_speed_mps": {speed!r},
    "forklift_start_yaw_degrees": {FORKLIFT_START_YAW_DEGREES!r},
    "forklift_end_yaw_degrees": {FORKLIFT_SOUTH_YAW_DEGREES!r},
    "forklift_orientation": "starts_eastbound_then_turns_clockwise_south",
    "robot_start_world": {ROBOT_START!r},
    "direct_route_waypoints": {ORIGINAL_ROUTE_WAYPOINTS!r},
    "left_bypass_waypoints": {NORTH_BYPASS_WAYPOINTS!r},
    "common_destination_world": {COMMON_LOADING_DESTINATION!r},
    "pallets_untouched": True,
    "timeline_playing": timeline.is_playing(),
}}))
'''.strip()


def stock_blind_corner_shelves_source(arguments: dict[str, Any]) -> str:
    """Fill the northeast rack with official cartons to strengthen occlusion."""
    columns = arguments.get("columns", 4)
    levels = arguments.get("levels", 3)
    if isinstance(columns, bool) or not isinstance(columns, int):
        raise ValueError("columns must be an integer between 3 and 5")
    if not 3 <= columns <= 5:
        raise ValueError("columns must be an integer between 3 and 5")
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise ValueError("levels must be an integer between 2 and 3")
    if not 2 <= levels <= 3:
        raise ValueError("levels must be an integer between 2 and 3")
    return f'''
import json
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_full_asset_path
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
root = {f"{SCENE_ROOT}/BlindCornerShelfStock"!r}
if stage.GetPrimAtPath(root).IsValid():
    stage.RemovePrim(root)
UsdGeom.Xform.Define(stage, root)
asset_url = get_full_asset_path({CARDBOX_ASSET_PATH!r})
if not asset_url:
    raise RuntimeError("The official Isaac warehouse carton asset did not resolve")

columns = {columns!r}
levels = {levels!r}
x_values = [
    1.55 + index * (2.9 / max(1, columns - 1))
    for index in range(columns)
]
y_values = (2.94, 3.46)
# These are the measured shelf-top Z coordinates plus a 1 cm seating gap.
z_values = (1.335, 2.835, 4.135)[:levels]
created = []
for level_index, z_value in enumerate(z_values):
    for depth_index, y_value in enumerate(y_values):
        for column_index, x_value in enumerate(x_values):
            path = (
                f"{{root}}/L{{level_index}}_D{{depth_index}}_"
                f"C{{column_index}}"
            )
            prim = add_reference_to_stage(usd_path=asset_url, prim_path=path)
            api = UsdGeom.XformCommonAPI(prim)
            api.SetTranslate(Gf.Vec3d(x_value, y_value, z_value))
            api.SetRotate(
                Gf.Vec3f(
                    0.0,
                    0.0,
                    180.0 if (column_index + depth_index) % 2 else 0.0,
                ),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
            api.SetScale(Gf.Vec3f(2.05, 2.05, 2.05))
            created.append(path)

stable_frames = 0
for _ in range(300):
    loading = omni.usd.get_context().get_stage_loading_status()
    stable_frames = stable_frames + 1 if not any(loading) else 0
    if stable_frames >= 8:
        break
    await omni.kit.app.get_app().next_update_async()

print(json.dumps({{
    "stocked": True,
    "root": root,
    "official_asset": asset_url,
    "rack": "Row2 Segment1 northeast blind-corner rack",
    "carton_count": len(created),
    "columns": columns,
    "depth_rows": len(y_values),
    "levels": levels,
    "placement": "measured shelf tops plus 0.01 m",
    "pallets_untouched": True,
}}))
'''.strip()


def configure_robot_pov_source(arguments: dict[str, Any]) -> str:
    """Create a chassis-mounted PoV camera just ahead of the enlarged shell."""
    forward_m = _float_argument(arguments, "forward_m", 1.15, 0.8, 1.5)
    height_m = _float_argument(arguments, "height_m", 0.72, 0.45, 1.1)
    return f'''
import json
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
source = stage.GetPrimAtPath({ROBOT_SOURCE_CAMERA_PATH!r})
if not source.IsValid():
    raise RuntimeError("The packaged RobotBlue front Owl camera is missing")
camera_path = {ROBOT_FRONT_CAMERA_PATH!r}
if stage.GetPrimAtPath(camera_path).IsValid():
    stage.RemovePrim(camera_path)
camera = UsdGeom.Camera.Define(stage, camera_path).GetPrim()

# Preserve NVIDIA's composed camera model/calibration attributes, but author
# an independent transform outside the enlarged presentation shell.
for source_attr in source.GetAttributes():
    name = source_attr.GetName()
    if name.startswith("xformOp:") or name == "xformOpOrder":
        continue
    value = source_attr.Get()
    if value is None:
        continue
    target_attr = camera.GetAttribute(name)
    if not target_attr:
        target_attr = camera.CreateAttribute(
            name,
            source_attr.GetTypeName(),
            source_attr.IsCustom(),
        )
    target_attr.Set(value)

xformable = UsdGeom.Xformable(camera)
xformable.AddTranslateOp().Set(
    Gf.Vec3d({forward_m!r}, 0.0, {height_m!r})
)
source_orient = source.GetAttribute("xformOp:orient").Get()
xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(source_orient)
xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

for _ in range(4):
    await omni.kit.app.get_app().next_update_async()
print(json.dumps({{
    "configured": True,
    "camera_path": camera_path,
    "source_camera_path": {ROBOT_SOURCE_CAMERA_PATH!r},
    "forward_m": {forward_m!r},
    "height_m": {height_m!r},
    "parent": {f"{LIVE_ROOT}/RobotBlue/chassis_link/sensors"!r},
    "moves_with_robot": True,
    "packaged_camera_unchanged": True,
}}))
'''.strip()


def configure_navigation_graph_source() -> str:
    """Build an owned graph from Isaac's packaged navigation node types."""
    return f'''
import json
import omni.graph.core as og
import omni.usd
from pxr import Gf, Sdf, Vt

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
graph = {CODEX_NAV_GRAPH_PATH!r}
if stage.GetPrimAtPath(graph).IsValid():
    stage.RemovePrim(graph)

keys = og.Controller.Keys
og.Controller.edit(
    {{
        "graph_path": graph,
        "evaluator_name": "execution",
        "pipeline_stage": (
            og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION
        ),
    }},
    {{
        keys.CREATE_NODES: [
            ("Tick", "omni.graph.action.OnPlaybackTick"),
            (
                "GetTransform",
                "omni.graph.nodes.GetPrimLocalToWorldTransform",
            ),
            ("GetRotation", "omni.graph.nodes.GetMatrix4Quaternion"),
            ("GetTranslation", "omni.graph.nodes.GetMatrix4Translation"),
            ("Odometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
            (
                "Planner",
                "isaacsim.robot.wheeled_robots.QuinticPathPlanner",
            ),
            ("Goal", "isaacsim.robot.wheeled_robots.CheckGoal2D"),
            (
                "Stanley",
                "isaacsim.robot.wheeled_robots.StanleyControlPID",
            ),
            (
                "Differential",
                "isaacsim.robot.wheeled_robots.DifferentialController",
            ),
            (
                "Articulation",
                "isaacsim.core.nodes.IsaacArticulationController",
            ),
        ],
        keys.CONNECT: [
            (
                "GetTransform.outputs:localToWorldTransform",
                "GetTranslation.inputs:matrix",
            ),
            (
                "GetTransform.outputs:localToWorldTransform",
                "GetRotation.inputs:matrix",
            ),
            ("Tick.outputs:tick", "Odometry.inputs:execIn"),
            ("Tick.outputs:tick", "Planner.inputs:execIn"),
            (
                "GetTranslation.outputs:translation",
                "Planner.inputs:currentPosition",
            ),
            (
                "GetRotation.outputs:quaternion",
                "Planner.inputs:currentOrientation",
            ),
            (
                "GetTranslation.outputs:translation",
                "Goal.inputs:currentPosition",
            ),
            (
                "GetRotation.outputs:quaternion",
                "Goal.inputs:currentOrientation",
            ),
            ("Planner.outputs:execOut", "Goal.inputs:execIn"),
            ("Planner.outputs:target", "Goal.inputs:target"),
            (
                "Planner.outputs:targetChanged",
                "Goal.inputs:targetChanged",
            ),
            (
                "GetTranslation.outputs:translation",
                "Stanley.inputs:currentPosition",
            ),
            (
                "GetRotation.outputs:quaternion",
                "Stanley.inputs:currentOrientation",
            ),
            (
                "Odometry.outputs:linearVelocity",
                "Stanley.inputs:currentSpeed",
            ),
            ("Planner.outputs:pathArrays", "Stanley.inputs:pathArrays"),
            ("Planner.outputs:target", "Stanley.inputs:target"),
            (
                "Planner.outputs:targetChanged",
                "Stanley.inputs:targetChanged",
            ),
            ("Goal.outputs:execOut", "Stanley.inputs:execIn"),
            ("Goal.outputs:reachedGoal", "Stanley.inputs:reachedGoal"),
            ("Tick.outputs:tick", "Differential.inputs:execIn"),
            ("Tick.outputs:tick", "Articulation.inputs:execIn"),
            (
                "Differential.outputs:velocityCommand",
                "Articulation.inputs:velocityCommand",
            ),
        ],
        keys.SET_VALUES: [
            ("GetTransform.inputs:usePath", True),
            ("Planner.inputs:initialVelocity", 0.25),
            ("Planner.inputs:initialAccel", 0.01),
            ("Planner.inputs:goalVelocity", 0.25),
            ("Planner.inputs:goalAccel", 0.01),
            ("Planner.inputs:maxAccel", 1.5),
            ("Planner.inputs:maxJerk", 0.3),
            ("Planner.inputs:step", 0.16666666667),
            ("Goal.inputs:thresholds", Gf.Vec2d(0.5, 0.5)),
            ("Stanley.inputs:drawPath", False),
            ("Stanley.inputs:gains", Gf.Vec3d(1.0, 1.0, 1.0)),
            ("Stanley.inputs:maxVelocity", 0.5),
            ("Stanley.inputs:step", 0.16666666667),
            ("Stanley.inputs:thresholds", Gf.Vec2d(0.1, 0.1)),
            ("Stanley.inputs:wheelBase", 0.4132),
            ("Differential.inputs:wheelDistance", 0.4132),
            ("Differential.inputs:wheelRadius", 0.14),
            ("Differential.inputs:maxLinearSpeed", 0.5),
            ("Differential.inputs:maxAngularSpeed", 1.0),
            ("Differential.inputs:maxWheelSpeed", 100.0),
        ],
    }},
)

chassis_path = {f"{LIVE_ROOT}/RobotBlue/chassis_link"!r}
get_transform = stage.GetPrimAtPath(f"{{graph}}/GetTransform")
get_transform.GetAttribute("inputs:primPath").Set(chassis_path)
get_transform.GetRelationship("inputs:prim").SetTargets(
    [Sdf.Path(chassis_path)]
)
stage.GetPrimAtPath(f"{{graph}}/Odometry").GetRelationship(
    "inputs:chassisPrim"
).SetTargets([Sdf.Path(chassis_path)])
articulation = stage.GetPrimAtPath(f"{{graph}}/Articulation")
articulation.GetRelationship("inputs:targetPrim").SetTargets(
    [Sdf.Path(chassis_path)]
)
articulation.GetAttribute(
    "inputs:jointNames"
).Set(Vt.TokenArray(("joint_wheel_right", "joint_wheel_left")))

planner = stage.GetPrimAtPath(f"{{graph}}/Planner")
planner.GetAttribute("inputs:targetOrientation").Set(
    Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
)
target = stage.GetPrimAtPath(
    {f"{LIVE_ROOT}/RobotBlue/targetXform"!r}
)
target_value = target.GetAttribute("xformOp:translate").Get()
planner.GetAttribute("inputs:targetPosition").Set(
    Gf.Vec3d(*target_value)
)
packaged_graph = stage.GetPrimAtPath(
    {f"{LIVE_ROOT}/RobotBlue/ActionGraph"!r}
)
if packaged_graph.IsValid():
    packaged_runtime_graph = og.Controller.graph(
        {f"{LIVE_ROOT}/RobotBlue/ActionGraph"!r}
    )
    if packaged_runtime_graph:
        packaged_runtime_graph.set_disabled(True)
    packaged_graph.SetActive(False)

for _ in range(8):
    await omni.kit.app.get_app().next_update_async()
print(json.dumps({{
    "configured": True,
    "graph": graph,
    "navigation": (
        "bounded waypoint steering -> Isaac DifferentialController -> "
        "IsaacArticulationController"
    ),
    "planner_target_world": [float(v) for v in target_value],
    "planner_target_heading_radians": 0.0,
    "planner_heading_policy": "live_waypoint_bearing",
    "built_in_node_types": True,
    "incomplete_packaged_graph_disabled": True,
    "physics_wheel_actuation": True,
    "scripted_robot_motion": False,
}}))
'''.strip()


def configure_route_visualizations_source() -> str:
    """Preserve candidate path geometry while keeping it hidden in the demo."""
    routes = {
        "OriginalRoute": {
            "points": tuple(
                (x, y, 0.04)
                for x, y, _ in (
                    ROBOT_START,
                    *ORIGINAL_ROUTE_WAYPOINTS,
                )
            ),
            "color": (0.95, 0.95, 0.95),
        },
        "NorthBypass": {
            "points": tuple(
                (x, y, 0.05)
                for x, y, _ in (
                    ROBOT_START,
                    *NORTH_BYPASS_WAYPOINTS,
                )
            ),
            "color": (0.05, 0.55, 1.0),
        },
        "SouthBypass": {
            "points": tuple(
                (x, y, 0.06)
                for x, y, _ in (
                    ROBOT_START,
                    *SOUTH_BYPASS_WAYPOINTS,
                )
            ),
            "color": (1.0, 0.42, 0.03),
        },
    }
    return f'''
import json
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
root = {f"{LIVE_ROOT}/RouteOptions"!r}
if stage.GetPrimAtPath(root).IsValid():
    stage.RemovePrim(root)
UsdGeom.Xform.Define(stage, root)
route_specs = {routes!r}
state = stage.GetPrimAtPath({f"{LIVE_ROOT}/State"!r})
if state.IsValid():
    direct_waypoints_attr = state.GetAttribute("codex:directRouteWaypoints")
    authored_direct = (
        direct_waypoints_attr.Get() if direct_waypoints_attr else None
    )
    if authored_direct:
        route_specs["OriginalRoute"]["points"] = tuple(
            [(float({ROBOT_START[0]}), float({ROBOT_START[1]}), 0.04)]
            + [
                (float(point[0]), float(point[1]), 0.04)
                for point in authored_direct
            ]
        )
    scenario_attr = state.GetAttribute("codex:scenarioMode")
    if scenario_attr and scenario_attr.Get() == "clear_route_control":
        route_specs = {{"OriginalRoute": route_specs["OriginalRoute"]}}
created = {{}}
for name, spec in route_specs.items():
    curve = UsdGeom.BasisCurves.Define(stage, f"{{root}}/{{name}}")
    points = [Gf.Vec3f(*point) for point in spec["points"]]
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([0.075] * len(points))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
    created[name] = {{
        "points": [list(point) for point in spec["points"]],
        "color": list(spec["color"]),
    }}
route_root = stage.GetPrimAtPath(root)
route_root.SetActive(False)
for _ in range(3):
    await omni.kit.app.get_app().next_update_async()
print(json.dumps({{
    "configured": True,
    "root": root,
    "active": False,
    "visualization_policy": "authored_but_disabled",
    "routes": created,
    "legend": {{
        "NorthBypass": "cyan upper route",
        "SouthBypass": "orange lower route",
        "OriginalRoute": "white nominal L-route",
    }},
    "warehouse_markings_unchanged": False,
}}))
'''.strip()


def set_forklift_visible_source(arguments: dict[str, Any]) -> str:
    visible = _bool_argument(arguments, "visible", True)
    return f'''
import json
import omni.usd

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
forklift = stage.GetPrimAtPath(
    {f"{SCENE_ROOT}/Actors/ForkliftOrange"!r}
)
if not forklift.IsValid():
    raise RuntimeError("The realistic warehouse forklift actor is missing")
forklift.SetActive({visible!r})
print(json.dumps({{
    "visible": {visible!r},
    "forklift": {f"{SCENE_ROOT}/Actors/ForkliftOrange"!r},
    "transform_preserved": True,
}}))
'''.strip()


def set_congestion_source(arguments: dict[str, Any]) -> str:
    released = _bool_argument(arguments, "released", True)
    return f'''
import json
import omni.usd

LIVE_ROOT = {LIVE_ROOT!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
state = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/State")
if not state.IsValid():
    raise RuntimeError("Create the live aisle navigation scene first")

state.GetAttribute("codex:obstacleReleased").Set({released!r})
print(json.dumps({{
    "obstacle_released": {released!r},
    "intention": (
        "The forklift is approaching the right passage from behind the "
        "central stocked rack, then turning clockwise across it"
        if {released!r}
        else "The forklift is holding behind the central stocked rack"
    ),
}}))
'''.strip()


def apply_advisory_source(arguments: dict[str, Any]) -> str:
    action = arguments.get("action")
    route = arguments.get("route")
    request_id = arguments.get("request_id", "")
    decision_source = arguments.get("decision_source", "evk")
    latency_seconds = _float_argument(
        arguments,
        "latency_seconds",
        0.0,
        0.0,
        120.0,
    )
    if action not in ("CONTINUE", "YIELD", "STOP", "REROUTE"):
        raise ValueError("action must be CONTINUE, YIELD, STOP, or REROUTE")
    allowed_routes = {
        "CONTINUE": ("current", "direct", "direct_to_goal"),
        "YIELD": ("hold", "hold_west_gate"),
        "STOP": ("hold", "hold_west_gate"),
        "REROUTE": ("south_bypass", "north_bypass"),
    }
    if route not in allowed_routes[action]:
        raise ValueError(f"route is not valid for action {action}")
    if not isinstance(request_id, str) or len(request_id) > 128:
        raise ValueError("request_id must be a string of at most 128 characters")
    if decision_source not in ("evk", "local_safety", "manual"):
        raise ValueError("decision_source must be evk, local_safety, or manual")
    route_waypoints_json = json.dumps(
        {
            "direct": ORIGINAL_ROUTE_WAYPOINTS,
            "south_bypass": SOUTH_BYPASS_WAYPOINTS,
            "north_bypass": NORTH_BYPASS_WAYPOINTS,
        }
    )
    route_headings_json = json.dumps(ROUTE_HEADINGS)
    direct_goal = DIRECT_GOAL
    return f'''
import json
import math
import omni.usd
from pxr import Gf, UsdGeom

LIVE_ROOT = {LIVE_ROOT!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
state = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/State")
robot = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue")
chassis = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue/chassis_link")
target = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue/targetXform")
if not all(prim.IsValid() for prim in (state, robot, chassis, target)):
    raise RuntimeError("Create the live aisle navigation scene first")

def set_target_world(world_xyz, target_yaw=None):
    attr = target.GetAttribute("xformOp:translate")
    if not attr:
        attr = UsdGeom.Xformable(target).AddTranslateOp().GetAttr()
    attr.Set(Gf.Vec3d(*world_xyz))
    for planner_path in (
        {f"{CODEX_NAV_GRAPH_PATH}/Planner"!r},
        f"{{LIVE_ROOT}}/RobotBlue/ActionGraph/quintic_path_planner_01",
    ):
        planner = stage.GetPrimAtPath(planner_path)
        if planner.IsValid():
            planner.GetAttribute("inputs:targetPosition").Set(
                Gf.Vec3d(*world_xyz)
            )
            if target_yaw is not None:
                planner.GetAttribute("inputs:targetOrientation").Set(
                    Gf.Quatd(
                        math.cos(target_yaw * 0.5),
                        Gf.Vec3d(
                            0.0,
                            0.0,
                            math.sin(target_yaw * 0.5),
                        ),
                    )
                )

action = {action!r}
route = {route!r}
requested_route = route
position = UsdGeom.XformCache().GetLocalToWorldTransform(
    chassis
).ExtractTranslation()
route_waypoints = {route_waypoints_json}
route_headings = {route_headings_json}
direct_waypoints_attr = state.GetAttribute("codex:directRouteWaypoints")
direct_headings_attr = state.GetAttribute("codex:directRouteHeadings")
if direct_waypoints_attr:
    authored_direct = direct_waypoints_attr.Get()
    if authored_direct:
        route_waypoints["direct"] = [
            tuple(float(component) for component in waypoint)
            for waypoint in authored_direct
        ]
if direct_headings_attr:
    authored_headings = direct_headings_attr.Get()
    if authored_headings:
        route_headings["direct"] = [
            float(heading) for heading in authored_headings
        ]
waypoint_index = -1
if action == "CONTINUE" and route == "current":
    route = state.GetAttribute("codex:blueRoute").Get() or "direct"
    waypoint_index = state.GetAttribute("codex:waypointIndex").Get()
    waypoints = route_waypoints.get(route, route_waypoints["direct"])
    headings = route_headings.get(route, route_headings["direct"])
    if not isinstance(waypoint_index, int) or not (
        0 <= waypoint_index < len(waypoints)
    ):
        route = "direct"
        waypoints = route_waypoints[route]
        headings = route_headings[route]
        waypoint_index = 0 if position[0] < 5.8 else 1
    set_target_world(
        tuple(waypoints[waypoint_index]),
        headings[waypoint_index],
    )
    intention = (
        state.GetAttribute("codex:navigationIntention").Get()
        or "RobotBlue continues its current navigation route"
    )
elif action in ("YIELD", "STOP"):
    set_target_world(tuple(position))
    intention = "RobotBlue is holding for a supervisor safety advisory"
elif action == "REROUTE":
    waypoints = route_waypoints[route]
    existing_route = state.GetAttribute("codex:blueRoute").Get()
    existing_index = state.GetAttribute("codex:waypointIndex").Get()
    if (
        existing_route == route
        and isinstance(existing_index, int)
        and 0 <= existing_index < len(waypoints)
    ):
        waypoint_index = existing_index
    elif route == "north_bypass":
        if position[1] < 4.3:
            # A late wall-camera intervention can arrive after RobotBlue has
            # passed the center junction. Return west along the traversed aisle
            # before turning north; a diagonal jump to waypoint 1 would cut
            # through the stocked rack.
            waypoint_index = 0
        else:
            waypoint_index = 2
    else:
        if position[0] < -0.35:
            waypoint_index = 0
        elif position[1] > -1.1:
            waypoint_index = 1
        elif position[0] < 5.1:
            waypoint_index = 2
        elif position[1] < 1.1:
            waypoint_index = 3
        else:
            waypoint_index = 4
    set_target_world(
        tuple(waypoints[waypoint_index]),
        route_headings[route][waypoint_index],
    )
    intention = (
        "RobotBlue is taking the left passage around the occupied right "
        "passage, then converging on the unchanged loading destination"
        if route == "north_bypass"
        else
        f"RobotBlue is rerouting through the {{route.replace('_', ' ')}} "
        "to avoid conflicting warehouse traffic"
    )
else:
    waypoints = route_waypoints["direct"]
    headings = route_headings["direct"]
    existing_route = state.GetAttribute("codex:blueRoute").Get()
    existing_index = state.GetAttribute("codex:waypointIndex").Get()
    waypoint_index = (
        existing_index
        if (
            existing_route == "direct"
            and isinstance(existing_index, int)
            and 0 <= existing_index < len(waypoints)
        )
        else 0
    )
    set_target_world(
        tuple(waypoints[waypoint_index]),
        headings[waypoint_index],
    )
    scenario_mode = state.GetAttribute("codex:scenarioMode").Get()
    intention = (
        "RobotBlue is following the baseline route through the right passage "
        "to the far-side loading destination"
    )

stanley = stage.GetPrimAtPath({f"{CODEX_NAV_GRAPH_PATH}/Stanley"!r})
planner_path_visible = action not in ("YIELD", "STOP")
if stanley.IsValid():
    stanley.GetAttribute("inputs:drawPath").Set(planner_path_visible)
try:
    from isaacsim.util.debug_draw import _debug_draw
    _debug_draw.acquire_debug_draw_interface().clear_lines()
except ImportError:
    pass

state.GetAttribute("codex:blueAction").Set(action)
state.GetAttribute("codex:blueRoute").Set(route)
state.GetAttribute("codex:navigationIntention").Set(intention)
state.GetAttribute("codex:waypointIndex").Set(waypoint_index)
state.GetAttribute("codex:evkRequestStatus").Set("applied")
state.GetAttribute("codex:lastDecisionSource").Set({decision_source!r})
state.GetAttribute("codex:lastRequestId").Set({request_id!r})
state.GetAttribute("codex:lastLatencySeconds").Set({latency_seconds!r})
print(json.dumps({{
    "applied": True,
    "action": action,
    "route": route,
    "requested_route": requested_route,
    "waypoint_index": waypoint_index,
    "navigation_intention": intention,
    "request_id": {request_id!r},
    "decision_source": {decision_source!r},
    "latency_seconds": {latency_seconds!r},
    "planner_path_visible": planner_path_visible,
}}))
'''.strip()


def tick_live_aisle_source() -> str:
    route_waypoints_json = json.dumps(
        {
            "direct": ORIGINAL_ROUTE_WAYPOINTS,
            "south_bypass": SOUTH_BYPASS_WAYPOINTS,
            "north_bypass": NORTH_BYPASS_WAYPOINTS,
        }
    )
    route_headings_json = json.dumps(ROUTE_HEADINGS)
    return f'''
import json
import math
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom

LIVE_ROOT = {LIVE_ROOT!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
state = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/State")
blue_root = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue")
blue = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue/chassis_link")
forklift = stage.GetPrimAtPath({f"{SCENE_ROOT}/Actors/ForkliftOrange"!r})
target = stage.GetPrimAtPath(f"{{LIVE_ROOT}}/RobotBlue/targetXform")
if not all(prim.IsValid() for prim in (state, blue_root, blue, forklift, target)):
    raise RuntimeError("Create the live aisle navigation scene first")

cache = UsdGeom.XformCache()
timeline = omni.timeline.get_timeline_interface()
timeline_seconds = timeline.get_current_time()
last_tick_attr = state.GetAttribute("codex:lastTickTimelineSeconds")
last_tick = last_tick_attr.Get() or timeline_seconds
delta_seconds = max(0.0, min(2.0, timeline_seconds - last_tick))
last_tick_attr.Set(timeline_seconds)
route_waypoints = {route_waypoints_json}
route_headings = {route_headings_json}
direct_waypoints_attr = state.GetAttribute("codex:directRouteWaypoints")
direct_headings_attr = state.GetAttribute("codex:directRouteHeadings")
if direct_waypoints_attr:
    authored_direct = direct_waypoints_attr.Get()
    if authored_direct:
        route_waypoints["direct"] = [
            tuple(float(component) for component in waypoint)
            for waypoint in authored_direct
        ]
if direct_headings_attr:
    authored_headings = direct_headings_attr.Get()
    if authored_headings:
        route_headings["direct"] = [
            float(heading) for heading in authored_headings
        ]
route = state.GetAttribute("codex:blueRoute").Get() or "direct"
index = state.GetAttribute("codex:waypointIndex").Get()
motion_controller_attr = state.GetAttribute("codex:motionController")
motion_controller = (
    motion_controller_attr.Get()
    if motion_controller_attr
    else "isaac_graph"
)
motion_delta_seconds = (
    max(1.0, delta_seconds)
    if motion_controller == "kinematic_waypoint"
    else delta_seconds
)
robot_yaw_degrees = None
navigation_linear_mps = 0.0
navigation_angular_radps = 0.0
if (
    motion_controller == "kinematic_waypoint"
    and state.GetAttribute("codex:blueAction").Get()
    not in ("YIELD", "STOP")
    and route in route_waypoints
    and isinstance(index, int)
    and 0 <= index < len(route_waypoints[route])
):
    robot_translate = blue_root.GetAttribute("xformOp:translate")
    if not robot_translate:
        robot_translate = UsdGeom.Xformable(
            blue_root
        ).AddTranslateOp().GetAttr()
    current = robot_translate.Get() or Gf.Vec3d(*{ROBOT_START!r})
    waypoint = route_waypoints[route][index]
    offset_x = waypoint[0] - current[0]
    offset_y = waypoint[1] - current[1]
    remaining = math.hypot(offset_x, offset_y)
    if remaining > 1e-6:
        speed_attr = state.GetAttribute("codex:kinematicSpeedMps")
        speed = speed_attr.Get() if speed_attr else 0.55
        travel = min(remaining, speed * motion_delta_seconds)
        robot_translate.Set(
            Gf.Vec3d(
                current[0] + offset_x * (travel / remaining),
                current[1] + offset_y * (travel / remaining),
                current[2],
            )
        )
        heading_radians = math.atan2(offset_y, offset_x)
        robot_yaw_degrees = math.degrees(heading_radians)
        # The referenced Carter root already owns xformOp:orient. XformCommonAPI
        # cannot replace that op with rotateXYZ, so SetRotate leaves the visible
        # robot at its original heading. Author the existing orient op directly
        # so the complete articulation and all visuals turn together.
        robot_orient = blue_root.GetAttribute("xformOp:orient")
        if not robot_orient:
            robot_orient = UsdGeom.Xformable(blue_root).AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble
            ).GetAttr()
        robot_orient.Set(
            Gf.Quatd(
                math.cos(heading_radians * 0.5),
                Gf.Vec3d(
                    0.0,
                    0.0,
                    math.sin(heading_radians * 0.5),
                ),
            )
        )
        cache = UsdGeom.XformCache()
elif (
    motion_controller == "isaac_graph"
    and route in route_waypoints
    and isinstance(index, int)
    and 0 <= index < len(route_waypoints[route])
):
    current_transform = cache.GetLocalToWorldTransform(blue)
    current = current_transform.ExtractTranslation()
    current_quat = current_transform.ExtractRotationQuat()
    current_imaginary = current_quat.GetImaginary()
    current_yaw = math.atan2(
        2.0
        * (
            current_quat.GetReal() * current_imaginary[2]
            + current_imaginary[0] * current_imaginary[1]
        ),
        1.0
        - 2.0
        * (
            current_imaginary[1] * current_imaginary[1]
            + current_imaginary[2] * current_imaginary[2]
        ),
    )
    robot_yaw_degrees = math.degrees(current_yaw)
    waypoint = route_waypoints[route][index]
    offset_x = waypoint[0] - current[0]
    offset_y = waypoint[1] - current[1]
    remaining = math.hypot(offset_x, offset_y)
    desired_yaw = math.atan2(offset_y, offset_x)
    heading_error = (
        (desired_yaw - current_yaw + math.pi) % (2.0 * math.pi)
    ) - math.pi
    action = state.GetAttribute("codex:blueAction").Get()
    if action not in ("YIELD", "STOP") and remaining > 0.18:
        speed_attr = state.GetAttribute("codex:kinematicSpeedMps")
        requested_speed = speed_attr.Get() if speed_attr else 0.18
        navigation_linear_mps = min(
            requested_speed,
            max(0.06, remaining * 0.6),
        )
        if abs(heading_error) > 1.0:
            navigation_linear_mps = 0.0
        elif abs(heading_error) > 0.45:
            navigation_linear_mps *= 0.35
        navigation_angular_radps = max(
            -0.8,
            min(0.8, -heading_error * 1.8),
        )
    differential = stage.GetPrimAtPath(
        f"{{LIVE_ROOT}}/RobotBlue/CodexNavigationGraph/Differential"
    )
    if not differential.IsValid():
        raise RuntimeError("Configure the Isaac navigation graph first")
    differential.GetAttribute("inputs:linearVelocity").Set(
        navigation_linear_mps
    )
    differential.GetAttribute("inputs:angularVelocity").Set(
        navigation_angular_radps
    )
blue_pos = cache.GetLocalToWorldTransform(blue).ExtractTranslation()
if state.GetAttribute("codex:obstacleReleased").Get():
    forklift_translate = forklift.GetAttribute("xformOp:translate")
    forklift_rotate = forklift.GetAttribute("xformOp:rotateXYZ")
    if not forklift_rotate:
        forklift_rotate = UsdGeom.Xformable(
            forklift
        ).AddRotateXYZOp().GetAttr()
    current = forklift_translate.Get()
    if current is None:
        current = state.GetAttribute("codex:forkliftStartWorld").Get()
    turn = state.GetAttribute("codex:forkliftTurnWorld").Get()
    end = state.GetAttribute("codex:forkliftEndWorld").Get()
    speed = state.GetAttribute("codex:forkliftSpeedMps").Get() or {
        BLIND_CORNER_SPEED_MPS!r
    }
    start_yaw = state.GetAttribute("codex:forkliftYawDegrees").Get()
    end_yaw = state.GetAttribute("codex:forkliftEndYawDegrees").Get()
    phase_attr = state.GetAttribute("codex:forkliftMotionPhase")
    phase = phase_attr.Get() or "holding_blind_corner"
    if phase == "holding_blind_corner":
        phase = "approaching_right_passage"
    forklift_target = (
        turn
        if phase == "approaching_right_passage"
        else end
    )
    offset = forklift_target - current
    remaining = math.sqrt(
        offset[0] * offset[0]
        + offset[1] * offset[1]
        + offset[2] * offset[2]
    )
    if remaining > 1e-6:
        travel = min(remaining, speed * motion_delta_seconds)
        updated = current + offset * (travel / remaining)
        forklift_translate.Set(updated)
        if phase == "approaching_right_passage":
            forklift_rotate.Set(
                Gf.Vec3f(0.0, 0.0, float(start_yaw))
            )
            phase_attr.Set(
                "approaching_right_passage"
                if travel < remaining
                else "turning_into_right_passage"
            )
        else:
            total_turn_distance = math.sqrt(
                (end[0] - turn[0]) * (end[0] - turn[0])
                + (end[1] - turn[1]) * (end[1] - turn[1])
                + (end[2] - turn[2]) * (end[2] - turn[2])
            )
            remaining_after = max(0.0, remaining - travel)
            turn_fraction = (
                1.0
                if total_turn_distance <= 1e-6
                else 1.0 - min(1.0, remaining_after / total_turn_distance)
            )
            yaw = start_yaw + (end_yaw - start_yaw) * turn_fraction
            forklift_rotate.Set(Gf.Vec3f(0.0, 0.0, float(yaw)))
            phase_attr.Set(
                "turning_into_right_passage"
                if travel < remaining
                else "blocking_right_passage"
            )
    else:
        if phase == "approaching_right_passage":
            phase_attr.Set("turning_into_right_passage")
        else:
            forklift_rotate.Set(
                Gf.Vec3f(0.0, 0.0, float(end_yaw))
            )
            phase_attr.Set("blocking_right_passage")
    cache = UsdGeom.XformCache()
forklift_pos = cache.GetLocalToWorldTransform(forklift).ExtractTranslation()
advanced = False
waypoints = route_waypoints.get(route, [])
if route in route_waypoints and isinstance(index, int) and 0 <= index < len(waypoints):
    waypoint = waypoints[index]
    distance = math.hypot(blue_pos[0] - waypoint[0], blue_pos[1] - waypoint[1])
    # The enlarged tutorial AMR needs to reach the center of the narrow east
    # cross-aisle before turning. A loose threshold made its nose cut the rack
    # end-post while the wheel controller was still 0.55 m west of the corner.
    if distance < 0.20 and index < len(waypoints) - 1:
        index += 1
        attr = target.GetAttribute("xformOp:translate")
        if not attr:
            attr = UsdGeom.Xformable(target).AddTranslateOp().GetAttr()
        attr.Set(Gf.Vec3d(*waypoints[index]))
        for planner_path in (
            {f"{CODEX_NAV_GRAPH_PATH}/Planner"!r},
            f"{{LIVE_ROOT}}/RobotBlue/ActionGraph/"
            "quintic_path_planner_01",
        ):
            planner = stage.GetPrimAtPath(planner_path)
            if planner.IsValid():
                planner.GetAttribute("inputs:targetPosition").Set(
                    Gf.Vec3d(*waypoints[index])
                )
                target_yaw = route_headings[route][index]
                planner.GetAttribute("inputs:targetOrientation").Set(
                    Gf.Quatd(
                        math.cos(target_yaw * 0.5),
                        Gf.Vec3d(
                            0.0,
                            0.0,
                            math.sin(target_yaw * 0.5),
                        ),
                    )
                )
        state.GetAttribute("codex:waypointIndex").Set(index)
        advanced = True

separation = math.hypot(
    blue_pos[0] - forklift_pos[0],
    blue_pos[1] - forklift_pos[1],
)
route_destination = route_waypoints.get(route, [tuple(blue_pos)])[-1]
distance_to_destination = math.hypot(
    blue_pos[0] - route_destination[0],
    blue_pos[1] - route_destination[1],
)
destination_reached = (
    route in route_waypoints
    and isinstance(index, int)
    and index == len(route_waypoints[route]) - 1
    and distance_to_destination <= 0.20
)
print(json.dumps({{
    "timeline_time_seconds": timeline_seconds,
    "timeline_playing": timeline.is_playing(),
    "blue_position": [float(v) for v in blue_pos],
    "forklift_position": [float(v) for v in forklift_pos],
    "separation_m": separation,
    "action": state.GetAttribute("codex:blueAction").Get(),
    "route": route,
    "waypoint_index": index,
    "waypoint_advanced": advanced,
    "robot_yaw_degrees": robot_yaw_degrees,
    "navigation_linear_mps": navigation_linear_mps,
    "navigation_angular_radps": navigation_angular_radps,
    "route_destination": [float(v) for v in route_destination],
    "distance_to_destination_m": distance_to_destination,
    "destination_reached": destination_reached,
    "navigation_intention": state.GetAttribute(
        "codex:navigationIntention"
    ).Get(),
    "obstacle_released": state.GetAttribute("codex:obstacleReleased").Get(),
    "forklift_motion_phase": state.GetAttribute(
        "codex:forkliftMotionPhase"
    ).Get(),
    "evk_request_status": state.GetAttribute("codex:evkRequestStatus").Get(),
    "last_request_id": state.GetAttribute("codex:lastRequestId").Get(),
    "last_latency_seconds": state.GetAttribute(
        "codex:lastLatencySeconds"
    ).Get(),
    "motion_controller": motion_controller,
}}))
'''.strip()


def set_request_status_source(arguments: dict[str, Any]) -> str:
    status = arguments.get("status")
    request_id = arguments.get("request_id", "")
    if status not in ("idle", "buffering", "in_flight", "applied", "failed"):
        raise ValueError(
            "status must be idle, buffering, in_flight, applied, or failed"
        )
    if not isinstance(request_id, str) or len(request_id) > 128:
        raise ValueError("request_id must be a string of at most 128 characters")
    return f'''
import json
import omni.usd

state = omni.usd.get_context().get_stage().GetPrimAtPath(
    {f"{LIVE_ROOT}/State"!r}
)
if not state.IsValid():
    raise RuntimeError("Create the live aisle navigation scene first")
state.GetAttribute("codex:evkRequestStatus").Set({status!r})
state.GetAttribute("codex:lastRequestId").Set({request_id!r})
print(json.dumps({{
    "status": {status!r},
    "request_id": {request_id!r},
}}))
'''.strip()


def _capture_live_camera_frame_source(
    arguments: dict[str, Any],
    *,
    camera_path: str,
    camera_label: str,
    hide_robot_visual: bool = False,
    hide_route_visualizations: bool = False,
) -> str:
    filename = arguments.get("filename", "live_frame.png")
    if (
        not isinstance(filename, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png", filename)
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("filename must be a simple .png basename")
    LIVE_CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = str((LIVE_CAPTURE_ROOT / filename).resolve())
    return f'''
import json
import os
import omni.timeline
import omni.usd
import omni.replicator.core as rep
from omni.replicator.core.functional import write_image
from pxr import Gf, Sdf, UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
selection = omni.usd.get_context().get_selection()
saved_selection = selection.get_selected_prim_paths()
selection.set_selected_prim_paths([], False)
timeline = omni.timeline.get_timeline_interface()
timeline_was_playing = timeline.is_playing()
if timeline_was_playing:
    timeline.pause()
camera = stage.GetPrimAtPath({camera_path!r})
if not camera.IsValid():
    raise RuntimeError({f"The {camera_label} camera is missing"!r})
render_camera_path = {camera_path!r}
if {camera_label!r} == "robotblue_codex_front_pov":
    # Hydra render products do not reliably initialize from a camera nested
    # inside the articulated Carter hierarchy. Mirror its current world pose
    # and optics onto a world-space camera for off-screen comparison captures.
    source_camera = UsdGeom.Camera(camera)
    proxy_camera = UsdGeom.Camera.Define(
        stage,
        {ROBOT_FRONT_CAPTURE_PROXY_PATH!r},
    )
    proxy_xform = UsdGeom.Xformable(proxy_camera.GetPrim())
    proxy_xform.ClearXformOpOrder()
    proxy_xform.AddTransformOp().Set(
        UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    )
    for source_attr, proxy_attr in (
        (source_camera.GetFocalLengthAttr(), proxy_camera.GetFocalLengthAttr()),
        (
            source_camera.GetHorizontalApertureAttr(),
            proxy_camera.GetHorizontalApertureAttr(),
        ),
        (
            source_camera.GetVerticalApertureAttr(),
            proxy_camera.GetVerticalApertureAttr(),
        ),
        (
            source_camera.GetClippingRangeAttr(),
            proxy_camera.GetClippingRangeAttr(),
        ),
        (source_camera.GetProjectionAttr(), proxy_camera.GetProjectionAttr()),
    ):
        value = source_attr.Get()
        if value is not None:
            proxy_attr.Set(value)
    camera = proxy_camera.GetPrim()
    render_camera_path = {ROBOT_FRONT_CAPTURE_PROXY_PATH!r}
if {camera_label!r} == "roof_overview":
    camera_api = UsdGeom.XformCommonAPI(camera)
    camera_api.SetTranslate(Gf.Vec3d(0.0, 0.8, 21.0))
    camera_api.SetRotate(
        Gf.Vec3f(0.0, 0.0, 0.0),
        UsdGeom.XformCommonAPI.RotationOrderYXZ,
    )
    camera_geom = UsdGeom.Camera(camera)
    camera_geom.GetFocalLengthAttr().Set(17.5)
    camera_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))
# BlindCornerTactical is deliberately read-only during capture. Its current
# stage transform is the user's authored security-camera framing and must not
# be silently replaced by a stored layout matrix.
robot_visual = stage.GetPrimAtPath(
    {f"{LIVE_ROOT}/RobotBlue/chassis_link/visual"!r}
)
route_options = stage.GetPrimAtPath({f"{LIVE_ROOT}/RouteOptions"!r})
legacy_supervisor = stage.GetPrimAtPath({f"{SCENE_ROOT}/Supervisor"!r})
stanley = stage.GetPrimAtPath({f"{CODEX_NAV_GRAPH_PATH}/Stanley"!r})
saved_visibility = None
routes_were_active = None
legacy_supervisor_was_active = None
planner_draw_was_enabled = None
if {hide_robot_visual!r} and robot_visual.IsValid():
    visibility = UsdGeom.Imageable(robot_visual).GetVisibilityAttr()
    saved_visibility = visibility.Get()
    visibility.Set(UsdGeom.Tokens.invisible)
if {hide_route_visualizations!r} and route_options.IsValid():
    routes_were_active = route_options.IsActive()
    route_options.SetActive(False)
if {hide_route_visualizations!r} and legacy_supervisor.IsValid():
    legacy_supervisor_was_active = legacy_supervisor.IsActive()
    legacy_supervisor.SetActive(False)
if {hide_route_visualizations!r} and stanley.IsValid():
    draw_path_attr = stanley.GetAttribute("inputs:drawPath")
    planner_draw_was_enabled = draw_path_attr.Get()
    draw_path_attr.Set(False)
# Planner debug lines are renderer state, not USD children. Clear them for
# every sensor and presentation capture so only the intentional RouteOptions
# curves can appear in tutorial media.
try:
    from isaacsim.util.debug_draw import _debug_draw
    _debug_draw.acquire_debug_draw_interface().clear_lines()
except ImportError:
    pass
output_path = {output_path!r}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# Render directly from the USD camera. This is independent of the interactive
# viewport, so users can keep Perspective and supervisor views open side by
# side without capture focus changes or flicker.
render_product = rep.create.render_product(
    render_camera_path,
    (1280, 720),
    name="CodexSupervisorCapture",
    force_new=True,
)
rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
rgb_annotator.attach(render_product)
rgb_data = None
for _ in range(3):
    await rep.orchestrator.step_async(rt_subframes=8)
    if {hide_route_visualizations!r}:
        # Some packaged navigation components regenerate their debug path on
        # the Replicator render step even while the timeline is paused. Clear
        # it after stepping, then let the render product refresh once without
        # advancing simulation time.
        try:
            from isaacsim.util.debug_draw import _debug_draw
            _debug_draw.acquire_debug_draw_interface().clear_lines()
            await omni.kit.app.get_app().next_update_async()
        except ImportError:
            pass
    rgb_data = rgb_annotator.get_data()
    if rgb_data is not None and getattr(rgb_data, "size", 0) > 0:
        break
if rgb_data is None or getattr(rgb_data, "size", 0) == 0:
    rgb_annotator.detach()
    render_product.destroy()
    raise RuntimeError("Off-screen supervisor render returned no RGB data")
write_image(path=output_path, data=rgb_data)
for _ in range(60):
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        break
    await omni.kit.app.get_app().next_update_async()
rgb_annotator.detach()
render_product.destroy()
if {hide_robot_visual!r} and robot_visual.IsValid():
    UsdGeom.Imageable(robot_visual).GetVisibilityAttr().Set(
        saved_visibility or UsdGeom.Tokens.inherited
    )
if (
    {hide_route_visualizations!r}
    and route_options.IsValid()
    and routes_were_active is not None
):
    route_options.SetActive(routes_were_active)
if (
    {hide_route_visualizations!r}
    and legacy_supervisor.IsValid()
    and legacy_supervisor_was_active is not None
):
    legacy_supervisor.SetActive(legacy_supervisor_was_active)
if (
    {hide_route_visualizations!r}
    and stanley.IsValid()
    and planner_draw_was_enabled is not None
):
    stanley.GetAttribute("inputs:drawPath").Set(planner_draw_was_enabled)
selection.set_selected_prim_paths(saved_selection, False)
if timeline_was_playing:
    timeline.play()
print(json.dumps({{
    "output_path": output_path,
    "camera": {camera_label!r},
    "camera_path": {camera_path!r},
    "render_camera_path": render_camera_path,
    "capture_backend": "offscreen_render_product",
    "resolution": [1280, 720],
    "exists": os.path.isfile(output_path),
    "bytes": os.path.getsize(output_path) if os.path.isfile(output_path) else 0,
    "route_visualizations_in_capture": {not hide_route_visualizations!r},
}}))
'''.strip()


def capture_live_camera_grid_source(arguments: dict[str, Any]) -> str:
    """Capture four synchronized, north-up supervisor views for one mosaic."""
    filename = arguments.get("filename", "live_camera_grid.png")
    if (
        not isinstance(filename, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png", filename)
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("filename must be a simple .png basename")
    include_route_visualizations = _bool_argument(
        arguments,
        "include_route_visualizations",
        False,
    )
    LIVE_CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    output_paths = {
        label: str(
            (LIVE_CAPTURE_ROOT / f"{stem}_{label}.png").resolve()
        )
        for label in ("overview", "approach", "junction", "north_bypass")
    }
    camera_paths = {
        "overview": ROOF_CAMERA_PATH,
        "approach": APPROACH_CAMERA_PATH,
        "junction": TACTICAL_CAMERA_PATH,
        "north_bypass": NORTH_CAMERA_PATH,
    }
    return f'''
import json
import os
import omni.timeline
import omni.usd
from pxr import Sdf
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
camera_paths = {camera_paths!r}
output_paths = {output_paths!r}
for label, camera_path in camera_paths.items():
    if not stage.GetPrimAtPath(camera_path).IsValid():
        raise RuntimeError(f"The {{label}} grid camera is missing")

route_options = stage.GetPrimAtPath({f"{LIVE_ROOT}/RouteOptions"!r})
legacy_supervisor = stage.GetPrimAtPath({f"{SCENE_ROOT}/Supervisor"!r})
stanley = stage.GetPrimAtPath({f"{CODEX_NAV_GRAPH_PATH}/Stanley"!r})
routes_were_active = None
legacy_supervisor_was_active = None
planner_draw_was_enabled = None
viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active Isaac Sim viewport is available")
previous_camera_path = viewport.camera_path
timeline = omni.timeline.get_timeline_interface()
was_playing = timeline.is_playing()
if was_playing:
    timeline.pause()
for _ in range(2):
    await omni.kit.app.get_app().next_update_async()

try:
    if {not include_route_visualizations!r} and route_options.IsValid():
        routes_were_active = route_options.IsActive()
        route_options.SetActive(False)
    if {not include_route_visualizations!r} and legacy_supervisor.IsValid():
        legacy_supervisor_was_active = legacy_supervisor.IsActive()
        legacy_supervisor.SetActive(False)
    if {not include_route_visualizations!r} and stanley.IsValid():
        draw_path_attr = stanley.GetAttribute("inputs:drawPath")
        planner_draw_was_enabled = draw_path_attr.Get()
        draw_path_attr.Set(False)
        try:
            from isaacsim.util.debug_draw import _debug_draw
            _debug_draw.acquire_debug_draw_interface().clear_lines()
        except ImportError:
            pass
    for label, camera_path in camera_paths.items():
        viewport.camera_path = Sdf.Path(camera_path)
        for _ in range(12):
            await omni.kit.app.get_app().next_update_async()
        output_path = output_paths[label]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.isfile(output_path):
            os.remove(output_path)
        capture = capture_viewport_to_file(viewport, file_path=output_path)
        await capture.wait_for_result(completion_frames=2)
        for _ in range(60):
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                break
            await omni.kit.app.get_app().next_update_async()
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Grid camera {{label}} did not produce a frame")
finally:
    if (
        {not include_route_visualizations!r}
        and route_options.IsValid()
        and routes_were_active is not None
    ):
        route_options.SetActive(routes_were_active)
    if (
        {not include_route_visualizations!r}
        and legacy_supervisor.IsValid()
        and legacy_supervisor_was_active is not None
    ):
        legacy_supervisor.SetActive(legacy_supervisor_was_active)
    if (
        {not include_route_visualizations!r}
        and stanley.IsValid()
        and planner_draw_was_enabled is not None
    ):
        stanley.GetAttribute("inputs:drawPath").Set(planner_draw_was_enabled)
    restore_path = str(previous_camera_path)
    if not restore_path:
        restore_path = "/OmniverseKit_Persp"
    viewport.camera_path = Sdf.Path(restore_path)
    if was_playing:
        timeline.play()

print(json.dumps({{
    "camera": "supervisor_grid",
    "camera_paths": camera_paths,
    "output_paths": output_paths,
    "synchronized_by_timeline_pause": True,
    "north_up": True,
    "route_visualizations_in_capture": {include_route_visualizations!r},
}}))
'''.strip()


def capture_live_roof_frame_source(arguments: dict[str, Any]) -> str:
    include_route_visualizations = _bool_argument(
        arguments,
        "include_route_visualizations",
        False,
    )
    return _capture_live_camera_frame_source(
        arguments,
        camera_path=ROOF_CAMERA_PATH,
        camera_label="roof_overview",
        hide_route_visualizations=not include_route_visualizations,
    )


def capture_live_tactical_frame_source(arguments: dict[str, Any]) -> str:
    include_route_visualizations = _bool_argument(
        arguments,
        "include_route_visualizations",
        False,
    )
    return _capture_live_camera_frame_source(
        arguments,
        camera_path=TACTICAL_CAMERA_PATH,
        camera_label="blind_corner_tactical",
        hide_route_visualizations=not include_route_visualizations,
    )


def capture_live_robot_front_frame_source(arguments: dict[str, Any]) -> str:
    return _capture_live_camera_frame_source(
        arguments,
        camera_path=ROBOT_FRONT_CAMERA_PATH,
        camera_label="robotblue_codex_front_pov",
        hide_route_visualizations=True,
    )
