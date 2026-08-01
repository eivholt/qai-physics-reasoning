from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SCENE_ROOT = "/World/CodexPoC/ConveyorSafety"
FORKLIFT_ROOT = f"{SCENE_ROOT}/Forklifts"
WORKER_ROOT = f"{SCENE_ROOT}/Workers"
CONVEYOR_ROOT = f"{SCENE_ROOT}/Conveyor"
STACK_LIGHT_ROOT = f"{SCENE_ROOT}/StackLight"
STACK_LIGHT_ROOTS = (
    STACK_LIGHT_ROOT,
    f"{SCENE_ROOT}/StackLight_01",
    f"{SCENE_ROOT}/StackLight_02",
)
STACK_LIGHT_BOUNCE_PATHS = tuple(
    f"{SCENE_ROOT}/Lighting/StackLightBounce_{index:02d}"
    for index in range(1, 4)
)
CAMERA_ROOT = f"{SCENE_ROOT}/Cameras"
CARGO_ROOT = f"{SCENE_ROOT}/DynamicCargo"
PHYSICS_ROOT = f"{SCENE_ROOT}/Physics"
PHYSICS_STEPS_PER_SECOND = 30

ISAAC_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0"
)
FORKLIFT_ASSET = (
    f"{ISAAC_ASSET_ROOT}/Isaac/Robots/IsaacSim/ForkliftC/forklift_c.usd"
)
SIMPLE_WAREHOUSE_PROP_ROOT = (
    f"{ISAAC_ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/Props"
)
SIMPLE_WAREHOUSE_FLOOR_ASSET = (
    f"{SIMPLE_WAREHOUSE_PROP_ROOT}/SM_floor02.usd"
)
RACK_SHELF_ASSET = f"{SIMPLE_WAREHOUSE_PROP_ROOT}/SM_RackShelf_01.usd"
RACK_FRAME_ASSET = f"{SIMPLE_WAREHOUSE_PROP_ROOT}/SM_RackFrame_03.usd"
PARCEL_CARTON_ASSET = f"{SIMPLE_WAREHOUSE_PROP_ROOT}/SM_CardBoxD_04.usd"
FORK_CARTON_ASSET = f"{SIMPLE_WAREHOUSE_PROP_ROOT}/SM_CardBoxA_01.usd"
FORK_PALLET_ASSET = f"{ISAAC_ASSET_ROOT}/Isaac/Props/Pallet/pallet.usd"
FORK_CARGO_FORWARD_OFFSET_M = 1.44
FORK_PALLET_BASE_Z_M = 0.146
FORK_CARTON_BASE_Z_M = 0.300
FORK_PALLET_YAW_OFFSET_DEGREES = 90.0
CONVEYOR_STRAIGHT_ASSET = (
    f"{ISAAC_ASSET_ROOT}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
    "Equipment/Conveyors/ConveyorBelt_A/"
    "ConveyorBelt_A05_PR_NVD_01.usd"
)
CONVEYOR_CURVE_ASSET = (
    f"{ISAAC_ASSET_ROOT}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
    "Equipment/Conveyors/ConveyorBelt_A/"
    "ConveyorBelt_A11_PR_NVD_01.usd"
)
CONVEYOR_EXTENSION_ASSET = (
    f"{ISAAC_ASSET_ROOT}/Isaac/Props/Conveyors/ConveyorBelt_A08.usd"
)
PACKING_TABLE_ASSET = (
    f"{ISAAC_ASSET_ROOT}/Isaac/Props/PackingTable/packing_table.usd"
)
PEOPLE_ASSET_ROOT = (
    f"{ISAAC_ASSET_ROOT}/Isaac/People/Characters"
)
WORKER_ASSETS = (
    f"{PEOPLE_ASSET_ROOT}/male_adult_construction_01_new/"
    "male_adult_construction_01_new.usd",
    f"{PEOPLE_ASSET_ROOT}/male_adult_construction_03/"
    "male_adult_construction_03.usd",
    f"{PEOPLE_ASSET_ROOT}/male_adult_construction_05_new/"
    "male_adult_construction_05_new.usd",
    f"{PEOPLE_ASSET_ROOT}/male_adult_construction_01_new/"
    "male_adult_construction_01_new.usd",
)
MOTION_LIBRARY_ASSET = (
    f"{ISAAC_ASSET_ROOT}/Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
)
CAMERA_PATHS = {
    "detector_oblique": f"{CAMERA_ROOT}/DetectorOblique",
    "detector_overhead": f"{CAMERA_ROOT}/DetectorOverhead",
    "detector_ptz": f"{CAMERA_ROOT}/DetectorPTZ",
    "detector_endline": f"{CAMERA_ROOT}/DetectorEndline",
    "detector_side": f"{CAMERA_ROOT}/DetectorSide",
    "detector_isometric": f"{CAMERA_ROOT}/DetectorIsometric",
    "presentation": f"{CAMERA_ROOT}/Presentation",
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = REPOSITORY_ROOT / "artifacts" / "isaac_sim_conveyor_safety" / "frames"
EXTENSION_ROOT = REPOSITORY_ROOT / "isaac_sim_supervisor_omniverse" / "exts"


def _validate_png_basename(value: Any, *, default: str) -> str:
    filename = default if value is None else value
    if (
        not isinstance(filename, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png", filename)
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("filename must be a simple .png basename")
    return filename


def enable_conveyor_safety_extension_source() -> str:
    extension_root = EXTENSION_ROOT.resolve()
    if not extension_root.is_dir():
        raise ValueError(f"extension folder does not exist: {extension_root}")
    return f"""
import json
import omni.kit.app
import omni.ui as ui

extension_root = {str(extension_root)!r}
extension_name = "qai.conveyor_safety"
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
window = ui.Workspace.get_window("Reason2 Conveyor Safety")
print(json.dumps({{
    "extension": extension_name,
    "was_enabled": was_enabled,
    "enable_result": bool(enabled),
    "is_enabled": manager.is_extension_enabled(extension_name),
    "window_visible": bool(window and window.visible),
}}))
""".strip()


def create_conveyor_safety_scene_source(arguments: dict[str, Any]) -> str:
    new_stage = arguments.get("new_stage", True)
    start_playing = arguments.get("start_playing", True)
    if not isinstance(new_stage, bool):
        raise ValueError("new_stage must be a boolean")
    if not isinstance(start_playing, bool):
        raise ValueError("start_playing must be a boolean")

    return f"""
import json
import math
import numpy as np
import sys
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.cloner import Cloner
from pxr import (
    Gf,
    Sdf,
    Semantics,
    PhysxSchema,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdShade,
    UsdSkel,
)

context = omni.usd.get_context()
timeline = omni.timeline.get_timeline_interface()
timeline.stop()
capture_cache = sys.modules.pop("_qai_conveyor_capture_cache", None)
if capture_cache is not None:
    cached_resource = getattr(capture_cache, "resource", None)
    if cached_resource:
        try:
            cached_resource["rgb"].detach()
            cached_resource["bbox"].detach()
            cached_resource["render_product"].destroy()
        except Exception:
            pass
# Never let a previously enabled controller keep stale articulation handles
# while the scene subtree is being replaced.  The UI/controller is explicitly
# hot-reloaded after construction by isaac_enable_conveyor_safety.
extension_manager = omni.kit.app.get_app().get_extension_manager()
if extension_manager.is_extension_enabled("qai.conveyor_safety"):
    extension_manager.set_extension_enabled_immediate(
        "qai.conveyor_safety",
        False,
    )
    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()
if {new_stage!r}:
    await context.new_stage_async()
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()
stage = context.get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")

world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
stage.SetDefaultPrim(world)
if stage.GetPrimAtPath("/World/CodexPoC").IsValid():
    stage.RemovePrim("/World/CodexPoC")
UsdGeom.Xform.Define(stage, "/World/CodexPoC")
root = UsdGeom.Xform.Define(stage, {SCENE_ROOT!r}).GetPrim()

def set_xform(prim, translate, scale=None, rotate=(0.0, 0.0, 0.0)):
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*translate))
    api.SetRotate(
        Gf.Vec3f(*rotate),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    if scale is not None:
        api.SetScale(Gf.Vec3f(*scale))
    return prim

def set_color(prim, color, opacity=1.0):
    imageable = UsdGeom.Gprim(prim)
    imageable.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    imageable.CreateDisplayOpacityAttr([float(opacity)])
    return prim

def cube(path, translate, scale, color, opacity=1.0, rotate=(0.0, 0.0, 0.0)):
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    set_xform(prim, translate, scale, rotate)
    set_color(prim, color, opacity)
    return prim

def sphere(path, translate, radius, color):
    prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
    UsdGeom.Sphere(prim).CreateRadiusAttr(float(radius))
    set_xform(prim, translate)
    set_color(prim, color)
    return prim

def cylinder(path, translate, radius, height, color, rotate=(0.0, 0.0, 0.0)):
    prim = UsdGeom.Cylinder.Define(stage, path).GetPrim()
    geom = UsdGeom.Cylinder(prim)
    geom.CreateRadiusAttr(float(radius))
    geom.CreateHeightAttr(float(height))
    set_xform(prim, translate, rotate=rotate)
    set_color(prim, color)
    return prim

def preview_surface_material(
    path,
    color,
    roughness,
    metallic=0.0,
    opacity=1.0,
):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{{path}}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(Gf.Vec3f(*color))
    shader.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(float(roughness))
    shader.CreateInput(
        "metallic",
        Sdf.ValueTypeNames.Float,
    ).Set(float(metallic))
    shader.CreateInput(
        "clearcoat",
        Sdf.ValueTypeNames.Float,
    ).Set(0.0)
    shader.CreateInput(
        "opacity",
        Sdf.ValueTypeNames.Float,
    ).Set(float(opacity))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    return material

def bind_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return prim

def bind_physics_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )
    return prim

def collision_cube(
    path,
    translate,
    scale,
    rotate=(0.0, 0.0, 0.0),
    kinematic=False,
):
    prim = cube(
        path,
        translate,
        scale,
        (0.15, 0.65, 1.0),
        opacity=0.0,
        rotate=rotate,
    )
    UsdGeom.Imageable(prim).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(prim)
    if kinematic:
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr().Set(True)
        body.CreateKinematicEnabledAttr().Set(True)
    return prim

def disable_composed_collisions(root_prim):
    disabled_paths = []
    for composed_prim in Usd.PrimRange(root_prim):
        if not composed_prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(
            composed_prim
        ).CreateCollisionEnabledAttr().Set(False)
        # PhysX validates a referenced mesh's approximation before honoring
        # collisionEnabled=false. A stock triangle-mesh approximation nested
        # under our dynamic parcel body therefore emits a fallback warning on
        # every load even though the visual collider is disabled. Author a
        # legal dynamic-body approximation as well; the separate simple Cube
        # remains the only enabled collider.
        if composed_prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(
                composed_prim
            ).CreateApproximationAttr().Set("convexHull")
        disabled_paths.append(str(composed_prim.GetPath()))
    return disabled_paths

def attr(prim, name, type_name, value):
    target = prim.GetAttribute(name)
    if not target:
        target = prim.CreateAttribute(name, type_name)
    target.Set(value)
    return target

def reference_asset(
    path,
    asset_url,
    translate=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    rotate=(0.0, 0.0, 0.0),
):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(asset_url)
    set_xform(prim, translate, scale, rotate)
    return prim

def look_at(camera_path, eye, target, focal_length, up=(0.0, 0.0, 1.0)):
    camera = UsdGeom.Camera.Define(stage, camera_path)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetLookAt(
        Gf.Vec3d(*eye),
        Gf.Vec3d(*target),
        Gf.Vec3d(*up),
    )
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix.GetInverse())
    camera.CreateFocalLengthAttr(float(focal_length))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(11.784)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    return camera.GetPrim()

attr(root, "codex:sceneKind", Sdf.ValueTypeNames.String, "reason2_conveyor_safety")
belt_center_x_m = 4.42
belt_center_y_m = 1.268030
# The edited oval keeps the 3.0 m centre-line diameter but extends both lanes
# through the north-wall cutaway. An A05 and A08 module form each straight;
# A11 remains the official 180-degree turn at each end.
conveyor_oval_radius_m = 1.50
conveyor_oval_straight_half_length_m = 2.351374
conveyor_track_half_width_m = 0.55
conveyor_surface_height_m = 0.76
conveyor_surface_half_height_m = 0.38
conveyor_curve_segment_count = 12
belt_half_width_m = conveyor_track_half_width_m
belt_half_length_m = (
    conveyor_oval_straight_half_length_m
    + conveyor_oval_radius_m
    + conveyor_track_half_width_m
)
conveyor_visual_center_x_m = belt_center_x_m
conveyor_visual_center_y_m = belt_center_y_m
conveyor_visual_half_width_m = (
    conveyor_oval_radius_m + conveyor_track_half_width_m
)
conveyor_visual_half_length_m = belt_half_length_m
rack_half_length_m = 3.40
parcel_count = 8
conveyor_module_count = 6
scene_half_depth_m = 3.60
south_floor_extension_m = 1.45
extended_floor_center_y_m = -south_floor_extension_m * 0.5
extended_floor_half_depth_m = (
    scene_half_depth_m + south_floor_extension_m * 0.5
)
floor_west_edge_m = -5.33
floor_east_edge_m = 6.50
floor_center_x_m = 0.5 * (floor_west_edge_m + floor_east_edge_m)
floor_half_width_m = 0.5 * (floor_east_edge_m - floor_west_edge_m)
floor_inset_margin_x_m = 0.35
floor_inset_margin_y_m = 0.15
north_wall_half_depth_m = 0.08
north_wall_center_y_m = scene_half_depth_m - north_wall_half_depth_m
west_wall_center_x_m = -5.25
wall_height_m = 6.0
wall_center_z_m = 0.5 * wall_height_m
marked_zone_west_edge_m = (
    belt_center_x_m
    - conveyor_oval_radius_m
    - conveyor_track_half_width_m
    - 1.0
)
marked_zone_east_edge_m = floor_east_edge_m
# Keep the north edge just inside the wall. At the near-camera edge, use the
# same one-metre design margin as the west side of the oval. This removes the
# extra red foreground created when the conveyor was shifted north.
marked_zone_north_edge_m = scene_half_depth_m - 0.25
marked_zone_south_edge_m = (
    belt_center_y_m
    - conveyor_oval_straight_half_length_m
    - conveyor_oval_radius_m
    - conveyor_track_half_width_m
    - 1.0
)
marked_zone_center_y_m = 0.5 * (
    marked_zone_north_edge_m + marked_zone_south_edge_m
)
marked_zone_half_length_m = 0.5 * (
    marked_zone_north_edge_m - marked_zone_south_edge_m
)
marked_zone_center_x_m = 0.5 * (
    marked_zone_west_edge_m + marked_zone_east_edge_m
)
marked_zone_half_width_m = 0.5 * (
    marked_zone_east_edge_m - marked_zone_west_edge_m
)
conveyor_additional_approach_m = 0.50
conveyor_collider_surface_inset_m = 0.05
conveyor_collider_center_x_m = conveyor_visual_center_x_m
conveyor_collider_center_y_m = conveyor_visual_center_y_m
conveyor_collider_half_width_m = (
    conveyor_track_half_width_m - conveyor_collider_surface_inset_m
)
conveyor_collider_half_length_m = (
    conveyor_oval_straight_half_length_m
    + conveyor_oval_radius_m
    + conveyor_track_half_width_m
    - conveyor_collider_surface_inset_m
)
conveyor_collider_inset_m = conveyor_collider_surface_inset_m
behind_wall_floor_west_edge_m = 0.585
behind_wall_floor_east_edge_m = floor_east_edge_m
behind_wall_floor_south_edge_m = 3.45
behind_wall_floor_north_edge_m = 7.584183
behind_wall_floor_center_x_m = 0.5 * (
    behind_wall_floor_west_edge_m + behind_wall_floor_east_edge_m
)
behind_wall_floor_center_y_m = 0.5 * (
    behind_wall_floor_south_edge_m + behind_wall_floor_north_edge_m
)
behind_wall_floor_half_width_m = 0.5 * (
    behind_wall_floor_east_edge_m - behind_wall_floor_west_edge_m
)
behind_wall_floor_half_depth_m = 0.5 * (
    behind_wall_floor_north_edge_m - behind_wall_floor_south_edge_m
)
packing_table_center_m = (-1.105764, 3.030923, 0.497026)
packing_table_half_extents_m = (1.236823, 0.390813, 0.497026)
attr(root, "codex:beltCenterXM", Sdf.ValueTypeNames.Double, belt_center_x_m)
attr(root, "codex:beltCenterYM", Sdf.ValueTypeNames.Double, belt_center_y_m)
attr(root, "codex:beltHalfWidthM", Sdf.ValueTypeNames.Double, belt_half_width_m)
attr(root, "codex:beltHalfLengthM", Sdf.ValueTypeNames.Double, belt_half_length_m)
attr(root, "codex:conveyorLayout", Sdf.ValueTypeNames.String, "oval")
attr(
    root,
    "codex:conveyorOvalRadiusM",
    Sdf.ValueTypeNames.Double,
    conveyor_oval_radius_m,
)
attr(
    root,
    "codex:conveyorOvalStraightHalfLengthM",
    Sdf.ValueTypeNames.Double,
    conveyor_oval_straight_half_length_m,
)
attr(
    root,
    "codex:conveyorTrackHalfWidthM",
    Sdf.ValueTypeNames.Double,
    conveyor_track_half_width_m,
)
attr(
    root,
    "codex:conveyorSurfaceHeightM",
    Sdf.ValueTypeNames.Double,
    conveyor_surface_height_m,
)
attr(
    root,
    "codex:conveyorCurveSegmentCount",
    Sdf.ValueTypeNames.Int,
    conveyor_curve_segment_count,
)
# Worker and forklift avoidance use the oval's segment envelope rather than a
# single bounding rectangle, so the open island inside the loop remains usable.
attr(
    root,
    "codex:conveyorVisualCenterXM",
    Sdf.ValueTypeNames.Double,
    conveyor_visual_center_x_m,
)
attr(
    root,
    "codex:conveyorVisualCenterYM",
    Sdf.ValueTypeNames.Double,
    conveyor_visual_center_y_m,
)
attr(
    root,
    "codex:conveyorVisualHalfWidthM",
    Sdf.ValueTypeNames.Double,
    conveyor_visual_half_width_m,
)
attr(
    root,
    "codex:conveyorVisualHalfLengthM",
    Sdf.ValueTypeNames.Double,
    conveyor_visual_half_length_m,
)
attr(root, "codex:parcelCount", Sdf.ValueTypeNames.Int, parcel_count)
attr(
    root,
    "codex:markedZoneCenterYM",
    Sdf.ValueTypeNames.Double,
    marked_zone_center_y_m,
)
attr(
    root,
    "codex:markedZoneHalfLengthM",
    Sdf.ValueTypeNames.Double,
    marked_zone_half_length_m,
)
attr(
    root,
    "codex:conveyorColliderCenterXM",
    Sdf.ValueTypeNames.Double,
    conveyor_collider_center_x_m,
)
attr(
    root,
    "codex:conveyorColliderCenterYM",
    Sdf.ValueTypeNames.Double,
    conveyor_collider_center_y_m,
)
attr(
    root,
    "codex:conveyorColliderHalfWidthM",
    Sdf.ValueTypeNames.Double,
    conveyor_collider_half_width_m,
)
attr(
    root,
    "codex:conveyorColliderInsetM",
    Sdf.ValueTypeNames.Double,
    conveyor_collider_inset_m,
)
attr(
    root,
    "codex:conveyorAdditionalApproachM",
    Sdf.ValueTypeNames.Double,
    conveyor_additional_approach_m,
)
attr(
    root,
    "codex:conveyorColliderHalfLengthM",
    Sdf.ValueTypeNames.Double,
    conveyor_collider_half_length_m,
)
attr(root, "codex:westRackCenterXM", Sdf.ValueTypeNames.Double, -4.20)
attr(root, "codex:rackHalfDepthM", Sdf.ValueTypeNames.Double, 0.54)
attr(root, "codex:rackHalfLengthM", Sdf.ValueTypeNames.Double, rack_half_length_m)
attr(
    root,
    "codex:packingTableCenterM",
    Sdf.ValueTypeNames.Double3,
    Gf.Vec3d(*packing_table_center_m),
)
attr(
    root,
    "codex:packingTableHalfExtentsM",
    Sdf.ValueTypeNames.Double3,
    Gf.Vec3d(*packing_table_half_extents_m),
)
attr(
    root,
    "codex:floorCenterXM",
    Sdf.ValueTypeNames.Double,
    floor_center_x_m,
)
attr(
    root,
    "codex:floorCenterYM",
    Sdf.ValueTypeNames.Double,
    extended_floor_center_y_m,
)
attr(
    root,
    "codex:floorHalfWidthM",
    Sdf.ValueTypeNames.Double,
    floor_half_width_m,
)
attr(
    root,
    "codex:floorHalfDepthM",
    Sdf.ValueTypeNames.Double,
    extended_floor_half_depth_m,
)
attr(
    root,
    "codex:northWallCenterYM",
    Sdf.ValueTypeNames.Double,
    north_wall_center_y_m,
)
attr(
    root,
    "codex:northWallHalfDepthM",
    Sdf.ValueTypeNames.Double,
    north_wall_half_depth_m,
)
attr(
    root,
    "codex:westWallCenterXM",
    Sdf.ValueTypeNames.Double,
    west_wall_center_x_m,
)
attr(root, "codex:westWallHalfWidthM", Sdf.ValueTypeNames.Double, 0.08)
attr(root, "codex:workerRadiusM", Sdf.ValueTypeNames.Double, 0.38)
attr(root, "codex:workerWalkSpeedMps", Sdf.ValueTypeNames.Double, 0.55)
attr(
    root,
    "codex:workerNavigationMode",
    Sdf.ValueTypeNames.String,
    "collision_free_corridor_astar_dynamic_replan",
)
# Measured from the articulated ForkliftC composed bounds: local X is the
# forward axis and spans [-1.86670, 1.86670], while local Y is 1.36331 m wide.
attr(root, "codex:forkliftHalfLengthM", Sdf.ValueTypeNames.Double, 1.87)
attr(root, "codex:forkliftHalfWidthM", Sdf.ValueTypeNames.Double, 0.69)
attr(root, "codex:redThresholdM", Sdf.ValueTypeNames.Double, 1.0)
attr(root, "codex:amberOuterThresholdM", Sdf.ValueTypeNames.Double, 1.25)
attr(root, "codex:amberWarningBandM", Sdf.ValueTypeNames.Double, 0.25)
attr(root, "codex:predictionHorizonSeconds", Sdf.ValueTypeNames.Double, 3.0)
attr(root, "codex:activeForklift", Sdf.ValueTypeNames.Int, 0)
attr(root, "codex:enabledForkliftCount", Sdf.ValueTypeNames.Int, 2)
attr(root, "codex:enabledForkliftIndices", Sdf.ValueTypeNames.IntArray, [1, 2])
attr(root, "codex:workerCount", Sdf.ValueTypeNames.Int, 2)
attr(root, "codex:modelSignal", Sdf.ValueTypeNames.String, "GREEN")
attr(root, "codex:modelConfidence", Sdf.ValueTypeNames.Double, 0.0)
attr(root, "codex:lastInferenceId", Sdf.ValueTypeNames.String, "")
attr(root, "codex:lastLightMutated", Sdf.ValueTypeNames.Bool, True)
attr(root, "codex:lightMutationCount", Sdf.ValueTypeNames.Int, 1)
attr(root, "codex:controllerConnected", Sdf.ValueTypeNames.Bool, False)
attr(root, "codex:minForkliftClearanceM", Sdf.ValueTypeNames.Double, 99.0)
attr(root, "codex:groundTruthSignal", Sdf.ValueTypeNames.String, "GREEN")
attr(
    root,
    "codex:selectedDetectorCamera",
    Sdf.ValueTypeNames.String,
    "detector_endline",
)
attr(root, "codex:collisionBlockingEnabled", Sdf.ValueTypeNames.Bool, True)
attr(root, "codex:workerPushEnabled", Sdf.ValueTypeNames.Bool, True)
attr(root, "codex:dynamicCargoEnabled", Sdf.ValueTypeNames.Bool, True)
attr(root, "codex:dynamicBeltParcelsEnabled", Sdf.ValueTypeNames.Bool, True)
attr(root, "codex:conveyorVelocityMps", Sdf.ValueTypeNames.Double, 0.55)
attr(root, "codex:conveyorFrictionCoefficient", Sdf.ValueTypeNames.Double, 0.0)
attr(
    root,
    "codex:conveyorPhysicsMode",
    Sdf.ValueTypeNames.String,
    "rigid_parcels_with_oval_surface_velocity_field",
)
attr(
    root,
    "codex:obstacleColliderShape",
    Sdf.ValueTypeNames.String,
    "continuous_low_poly_oval_mesh_plus_simple_boxes",
)
attr(root, "codex:lastCollision", Sdf.ValueTypeNames.String, "")
attr(root, "codex:collisionBlockCount", Sdf.ValueTypeNames.Int, 0)
attr(
    root,
    "codex:drivingModel",
    Sdf.ValueTypeNames.String,
    "ForkliftC rear-steer Ackermann articulation",
)
attr(root, "codex:commandedSpeedMps", Sdf.ValueTypeNames.Double, 0.0)
attr(root, "codex:actualSpeedMps", Sdf.ValueTypeNames.Double, 0.0)
attr(root, "codex:steeringAngleDegrees", Sdf.ValueTypeNames.Double, 0.0)
attr(root, "codex:forkHeightM", Sdf.ValueTypeNames.Double, 0.0)
attr(
    root,
    "codex:presentationCameraDistanceM",
    Sdf.ValueTypeNames.Double,
    8.0,
)
attr(
    root,
    "codex:presentationCameraViewIndex",
    Sdf.ValueTypeNames.Int,
    2,
)

# Factory shell and traffic floor.
physics_scene = UsdPhysics.Scene.Define(stage, f"{{{PHYSICS_ROOT!r}}}/Scene")
physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(
    physics_scene.GetPrim()
)
physx_scene_api.CreateEnableGPUDynamicsAttr().Set(False)
# Slow indoor vehicles and box proxies remain stable at 30 Hz (3.3 cm per
# step at the 1 m/s speed cap), while the viewport and camera can render at
# 60 Hz. The previous default 60 Hz scene forced multiple expensive physics
# steps into rendered frames and made the interactive demo visibly stutter.
physx_scene_api.CreateTimeStepsPerSecondAttr().Set(
    {PHYSICS_STEPS_PER_SECOND}
)
floor = cube(
    f"{{{SCENE_ROOT!r}}}/Floor",
    (floor_center_x_m, extended_floor_center_y_m, -0.15),
    (floor_half_width_m, extended_floor_half_depth_m, 0.15),
    (0.16, 0.18, 0.20),
)
UsdPhysics.CollisionAPI.Apply(floor)
floor_inset = cube(
    f"{{{SCENE_ROOT!r}}}/FloorInset",
    (floor_center_x_m, extended_floor_center_y_m, 0.005),
    (
        floor_half_width_m - floor_inset_margin_x_m,
        extended_floor_half_depth_m - floor_inset_margin_y_m,
        0.02,
    ),
    (0.28, 0.30, 0.32),
)
# Use four official Simple Warehouse floor meshes as the visible epoxy finish.
# Their authored UVs preserve the MI_Floor_02b albedo, normal, mask, and
# roughness wear maps that a procedural Cube cannot display correctly. The
# simple Cube below remains the sole collider.
UsdGeom.Imageable(floor_inset).MakeInvisible()
floor_finish_root = f"{{{SCENE_ROOT!r}}}/FloorFinish"
UsdGeom.Xform.Define(stage, floor_finish_root)
floor_finish_tile_half_width_m = (
    floor_half_width_m - floor_inset_margin_x_m
) * 0.5
floor_finish_tile_half_depth_m = (
    extended_floor_half_depth_m - floor_inset_margin_y_m
) * 0.5
floor_finish_tile_scale = (
    floor_finish_tile_half_width_m / 3.0,
    floor_finish_tile_half_depth_m / 3.0,
    1.0,
)
floor_finish_tile_positions = tuple(
    (x, y)
    for x in (
        floor_center_x_m - floor_finish_tile_half_width_m,
        floor_center_x_m + floor_finish_tile_half_width_m,
    )
    for y in (
        extended_floor_center_y_m - floor_finish_tile_half_depth_m,
        extended_floor_center_y_m + floor_finish_tile_half_depth_m,
    )
)
for tile_index, (tile_x, tile_y) in enumerate(
    floor_finish_tile_positions,
    start=1,
):
    floor_tile = reference_asset(
        f"{{floor_finish_root}}/Tile{{tile_index}}",
        {SIMPLE_WAREHOUSE_FLOOR_ASSET!r},
        translate=(tile_x, tile_y, 0.008),
        scale=floor_finish_tile_scale,
    )
    disable_composed_collisions(floor_tile)
# The user-added tile continues the east half of the worn epoxy floor behind
# the wall, where the extended conveyor disappears from the room. A simple
# hidden floor box supports any parcel that is pushed through the cutaway.
behind_wall_floor_tile = reference_asset(
    f"{{floor_finish_root}}/Tile4_01",
    {SIMPLE_WAREHOUSE_FLOOR_ASSET!r},
    translate=(3.3675, 5.346683, 0.008),
    scale=floor_finish_tile_scale,
)
disable_composed_collisions(behind_wall_floor_tile)
behind_wall_floor = cube(
    f"{{{SCENE_ROOT!r}}}/BehindWallFloor",
    (
        behind_wall_floor_center_x_m,
        behind_wall_floor_center_y_m,
        -0.15,
    ),
    (
        behind_wall_floor_half_width_m,
        behind_wall_floor_half_depth_m,
        0.15,
    ),
    (0.16, 0.18, 0.20),
)
UsdPhysics.CollisionAPI.Apply(behind_wall_floor)

# Two precisely sized openings surround the parallel conveyor lanes. The
# upper lintel and centre plinth overlap the module bounds by no more than the
# small clearance below, eliminating the gaps/overhangs in the manual edit.
north_wall_opening_clearance_m = 0.02
west_opening_min_x_m = 2.344025 - north_wall_opening_clearance_m
west_opening_max_x_m = 3.494978 + north_wall_opening_clearance_m
east_opening_min_x_m = 5.348651 - north_wall_opening_clearance_m
east_opening_max_x_m = floor_east_edge_m
north_wall_opening_height_m = 1.30
north_wall_segments = (
    (
        "NorthWall",
        0.5 * (floor_west_edge_m + west_opening_min_x_m),
        0.5 * (west_opening_min_x_m - floor_west_edge_m),
        wall_center_z_m,
        wall_center_z_m,
    ),
    (
        "NorthWall_01",
        0.5 * (west_opening_min_x_m + east_opening_max_x_m),
        0.5 * (east_opening_max_x_m - west_opening_min_x_m),
        0.5 * (north_wall_opening_height_m + wall_height_m),
        0.5 * (wall_height_m - north_wall_opening_height_m),
    ),
    (
        "NorthWall_02",
        0.5 * (west_opening_max_x_m + east_opening_min_x_m),
        0.5 * (east_opening_min_x_m - west_opening_max_x_m),
        0.5 * north_wall_opening_height_m,
        0.5 * north_wall_opening_height_m,
    ),
)
for path, center_x, half_x, center_z, half_z in north_wall_segments:
    wall = cube(
        f"{{{SCENE_ROOT!r}}}/Shell/{{path}}",
        (center_x, north_wall_center_y_m, center_z),
        (half_x, north_wall_half_depth_m, half_z),
        (0.42, 0.46, 0.49),
    )
    UsdPhysics.CollisionAPI.Apply(wall)

# The detector sits beyond the south end of the aisle, so omitting that wall
# avoids clipping the low camera. The remaining west wall is one cheap box.
west_wall = cube(
    f"{{{SCENE_ROOT!r}}}/Shell/WestWall",
    (
        west_wall_center_x_m,
        extended_floor_center_y_m,
        wall_center_z_m,
    ),
    (0.08, extended_floor_half_depth_m, wall_center_z_m),
    (0.42, 0.46, 0.49),
)
UsdPhysics.CollisionAPI.Apply(west_wall)

# The translucent red region is the forbidden area less than one metre from
# the conveyor. AMBER means any forklift motion, so there is deliberately no
# yellow/orange floor band that could be mistaken for a second occupancy zone.
# Workers are intentionally allowed inside the red marking and must not affect
# the signal.
UsdGeom.Scope.Define(stage, f"{{{SCENE_ROOT!r}}}/Materials")
clearance_zone = cube(
    f"{{{SCENE_ROOT!r}}}/ClearanceZone",
    (marked_zone_center_x_m, marked_zone_center_y_m, 0.035),
    (marked_zone_half_width_m, marked_zone_half_length_m, 0.018),
    (0.70, 0.015, 0.01),
    1.0,
)
matte_red_material = preview_surface_material(
    f"{{{SCENE_ROOT!r}}}/Materials/MatteRedSafetyZone",
    (0.70, 0.015, 0.01),
    roughness=0.92,
)
bind_material(clearance_zone, matte_red_material)
# Keep the policy floor visually binary. Yellow boundaries, amber strips, and
# dashed traffic markings created extra line/color hypotheses for the vision
# model without encoding a distinct physical region.

# Extended oval conveyor built entirely from Isaac Sim assets: an A05 plus an
# A08 on each lane, and two A11 180-degree turns. The production meshes stay
# visual-only; one low-poly ring provides the stable, cheap collision surface.
UsdGeom.Xform.Define(stage, {CONVEYOR_ROOT!r})
conveyor_straight_asset = {CONVEYOR_STRAIGHT_ASSET!r}
conveyor_extension_asset = {CONVEYOR_EXTENSION_ASSET!r}
conveyor_curve_asset = {CONVEYOR_CURVE_ASSET!r}
oval_north_y_m = belt_center_y_m + conveyor_oval_straight_half_length_m
oval_south_y_m = belt_center_y_m - conveyor_oval_straight_half_length_m
oval_west_lane_x_m = belt_center_x_m - conveyor_oval_radius_m
oval_east_lane_x_m = belt_center_x_m + conveyor_oval_radius_m
# A05's roller centre is authored at local X=0.5014 and runs from local Y=0
# toward -Y for exactly 2.0 m. It forms the south half of each straight.
straight_origin_center_x_m = 0.5014
for name, lane_x in (
    ("StraightWest", oval_west_lane_x_m),
    ("StraightEast", oval_east_lane_x_m),
):
    module = reference_asset(
        f"{{{CONVEYOR_ROOT!r}}}/Modules/{{name}}",
        conveyor_straight_asset,
        translate=(
            lane_x - straight_origin_center_x_m,
            oval_south_y_m + 2.0,
            0.0,
        ),
        scale=(0.01, 0.01, 0.01),
    )
    disable_composed_collisions(module)
# The user's A08 additions form the north half. Their measured transforms make
# the A05/A08 and A08/A11 seams overlap by only millimetres.
for name, translate in (
    ("ExtensionWest", (2.919516, 0.915113, 0.0)),
    ("ExtensionEast", (5.924142, 0.915113, 0.0)),
):
    module = reference_asset(
        f"{{{CONVEYOR_ROOT!r}}}/Modules/{{name}}",
        conveyor_extension_asset,
        translate=translate,
        rotate=(0.0, 0.0, -90.0),
    )
    # A08 carries an authored transform-op stack that is incompatible with
    # XformCommonAPI, so its common translate/rotate request is ignored. One
    # explicit matrix op reliably authors the measured -90-degree placement.
    module_xform = UsdGeom.Xformable(module)
    module_xform.ClearXformOpOrder()
    module_placement = Gf.Matrix4d(
        0.0, -1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        translate[0], translate[1], translate[2], 1.0,
    )
    module_xform.AddTransformOp(
        opSuffix="placement"
    ).Set(module_placement)
    disable_composed_collisions(module)
# A11 spans local X=0..4 m and bends toward +Y. A 180-degree rotation makes
# the matching south turn while preserving the 3.0 m roller centre spacing.
for name, translate, yaw in (
    (
        "CurveNorth",
        (belt_center_x_m - 2.0, oval_north_y_m, 0.0),
        0.0,
    ),
    (
        "CurveSouth",
        (belt_center_x_m + 2.0, oval_south_y_m, 0.0),
        180.0,
    ),
):
    module = reference_asset(
        f"{{{CONVEYOR_ROOT!r}}}/Modules/{{name}}",
        conveyor_curve_asset,
        translate=translate,
        scale=(0.01, 0.01, 0.01),
        rotate=(0.0, 0.0, yaw),
    )
    disable_composed_collisions(module)

def oval_point_tangent(phase):
    straight_length = 2.0 * conveyor_oval_straight_half_length_m
    arc_length = math.pi * conveyor_oval_radius_m
    perimeter = 2.0 * straight_length + 2.0 * arc_length
    distance = (float(phase) % 1.0) * perimeter
    if distance < straight_length:
        return (
            oval_west_lane_x_m,
            oval_south_y_m + distance,
            0.0,
            1.0,
        )
    distance -= straight_length
    if distance < arc_length:
        angle = math.pi - distance / conveyor_oval_radius_m
        return (
            belt_center_x_m + conveyor_oval_radius_m * math.cos(angle),
            oval_north_y_m + conveyor_oval_radius_m * math.sin(angle),
            math.sin(angle),
            -math.cos(angle),
        )
    distance -= arc_length
    if distance < straight_length:
        return (
            oval_east_lane_x_m,
            oval_north_y_m - distance,
            0.0,
            -1.0,
        )
    distance -= straight_length
    angle = -distance / conveyor_oval_radius_m
    return (
        belt_center_x_m + conveyor_oval_radius_m * math.cos(angle),
        oval_south_y_m + conveyor_oval_radius_m * math.sin(angle),
        math.sin(angle),
        -math.cos(angle),
    )

surface_collider_root = (
    f"{{{PHYSICS_ROOT!r}}}/StaticColliders/OvalConveyor"
)
UsdGeom.Scope.Define(stage, surface_collider_root)
# Match Isaac Sim's custom conveyor sample: built-in contact friction is zero
# on the belt geometry because the velocity field supplies the tangential
# transport force. Floor and parcel friction remain at their normal defaults.
conveyor_physics_material = UsdShade.Material.Define(
    stage,
    f"{{{SCENE_ROOT!r}}}/Materials/ConveyorPhysics",
)
conveyor_material_api = UsdPhysics.MaterialAPI.Apply(
    conveyor_physics_material.GetPrim()
)
conveyor_material_api.CreateStaticFrictionAttr().Set(0.0)
conveyor_material_api.CreateDynamicFrictionAttr().Set(0.0)
conveyor_material_api.CreateRestitutionAttr().Set(0.0)
PhysxSchema.PhysxMaterialAPI.Apply(
    conveyor_physics_material.GetPrim()
).CreateFrictionCombineModeAttr("min")
# A single closed, low-poly ring removes the internal vertical faces produced
# by overlapping boxes. Those faces could snag a rigid carton at a curve
# junction. This mesh has only four vertices per sample and is still vastly
# simpler than the referenced production meshes.
conveyor_collider_sample_count = (
    2 * conveyor_curve_segment_count + 8
)
conveyor_collider_points = []
for sample_index in range(conveyor_collider_sample_count):
    x, y, tangent_x, tangent_y = oval_point_tangent(
        sample_index / float(conveyor_collider_sample_count)
    )
    outward_x = -tangent_y
    outward_y = tangent_x
    for z, side_sign in (
        (conveyor_surface_height_m, 1.0),
        (conveyor_surface_height_m, -1.0),
        (0.0, 1.0),
        (0.0, -1.0),
    ):
        conveyor_collider_points.append(
            Gf.Vec3f(
                x
                + outward_x
                * conveyor_collider_half_width_m
                * side_sign,
                y
                + outward_y
                * conveyor_collider_half_width_m
                * side_sign,
                z,
            )
        )
conveyor_collider_face_counts = []
conveyor_collider_face_indices = []
for sample_index in range(conveyor_collider_sample_count):
    next_index = (
        sample_index + 1
    ) % conveyor_collider_sample_count
    outer_top = sample_index * 4
    inner_top = outer_top + 1
    outer_bottom = outer_top + 2
    inner_bottom = outer_top + 3
    next_outer_top = next_index * 4
    next_inner_top = next_outer_top + 1
    next_outer_bottom = next_outer_top + 2
    next_inner_bottom = next_outer_top + 3
    conveyor_collider_face_counts.extend((4, 4, 4, 4))
    conveyor_collider_face_indices.extend(
        (
            outer_top,
            inner_top,
            next_inner_top,
            next_outer_top,
            outer_bottom,
            next_outer_bottom,
            next_inner_bottom,
            inner_bottom,
            outer_top,
            next_outer_top,
            next_outer_bottom,
            outer_bottom,
            inner_top,
            inner_bottom,
            next_inner_bottom,
            next_inner_top,
        )
    )
conveyor_surface_collider = UsdGeom.Mesh.Define(
    stage,
    f"{{surface_collider_root}}/Surface",
)
conveyor_surface_collider.CreatePointsAttr().Set(
    conveyor_collider_points
)
conveyor_surface_collider.CreateFaceVertexCountsAttr().Set(
    conveyor_collider_face_counts
)
conveyor_surface_collider.CreateFaceVertexIndicesAttr().Set(
    conveyor_collider_face_indices
)
conveyor_surface_collider.CreateSubdivisionSchemeAttr().Set(
    UsdGeom.Tokens.none
)
conveyor_surface_collider.CreateDoubleSidedAttr().Set(True)
UsdGeom.Imageable(conveyor_surface_collider.GetPrim()).MakeInvisible()
UsdPhysics.CollisionAPI.Apply(conveyor_surface_collider.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(
    conveyor_surface_collider.GetPrim()
).CreateApproximationAttr().Set("none")
bind_physics_material(
    conveyor_surface_collider.GetPrim(),
    conveyor_physics_material,
)

# Each transported carton is a real rigid body with a simple box collider.
# Dimensions and aspect ratios are randomized with a fixed seed so screenshots
# are varied while benchmark runs remain reproducible.
parcel_carton_asset = {PARCEL_CARTON_ASSET!r}
parcel_rng = np.random.default_rng(20260731)
for index in range(parcel_count):
    parcel_length_m = float(parcel_rng.uniform(0.50, 0.82))
    parcel_width_m = float(parcel_rng.uniform(0.32, 0.54))
    parcel_height_m = float(parcel_rng.uniform(0.22, 0.42))
    phase = (index + 0.12) / float(parcel_count)
    x, y, tangent_x, tangent_y = oval_point_tangent(phase)
    yaw = math.degrees(math.atan2(tangent_y, tangent_x))
    parcel_path = f"{{{CONVEYOR_ROOT!r}}}/Parcels/Parcel{{index + 1}}"
    parcel = UsdGeom.Xform.Define(stage, parcel_path).GetPrim()
    set_xform(
        parcel,
        (
            x,
            y,
            conveyor_surface_height_m + 0.5 * parcel_height_m + 0.015,
        ),
        rotate=(
            0.0,
            0.0,
            yaw + float(parcel_rng.uniform(-3.0, 3.0)),
        ),
    )
    attr(parcel, "codex:phase", Sdf.ValueTypeNames.Double, phase)
    attr(
        parcel,
        "codex:dimensionsM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(parcel_length_m, parcel_width_m, parcel_height_m),
    )
    parcel_mass_kg = max(
        2.5,
        parcel_length_m * parcel_width_m * parcel_height_m * 42.0,
    )
    attr(parcel, "codex:massKg", Sdf.ValueTypeNames.Double, parcel_mass_kg)
    attr(
        parcel,
        "codex:initialPosition",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(
            x,
            y,
            conveyor_surface_height_m + 0.5 * parcel_height_m + 0.015,
        ),
    )
    attr(
        parcel,
        "codex:initialYawDegrees",
        Sdf.ValueTypeNames.Double,
        yaw,
    )
    body = UsdPhysics.RigidBodyAPI.Apply(parcel)
    body.CreateRigidBodyEnabledAttr().Set(True)
    body.CreateVelocityAttr().Set(
        Gf.Vec3f(
            tangent_x * 0.55,
            tangent_y * 0.55,
            0.0,
        )
    )
    UsdPhysics.MassAPI.Apply(parcel).CreateMassAttr().Set(parcel_mass_kg)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(parcel)
    physx_body.CreateEnableCCDAttr().Set(True)
    physx_body.CreateLinearDampingAttr().Set(0.12)
    physx_body.CreateAngularDampingAttr().Set(0.35)
    # Drive the rigid carton with a world-space tangential force instead of
    # editing its transform. The extension updates this force only while the
    # box is in contact with the oval surface, so a forklift can push it free.
    belt_force = PhysxSchema.PhysxForceAPI.Apply(parcel)
    belt_force.CreateForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    belt_force.CreateTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    belt_force.CreateModeAttr().Set("force")
    belt_force.CreateForceEnabledAttr().Set(True)
    belt_force.CreateWorldFrameEnabledAttr().Set(True)
    collision_cube(
        f"{{parcel_path}}/Collision",
        (0.0, 0.0, 0.0),
        (
            0.5 * parcel_length_m,
            0.5 * parcel_width_m,
            0.5 * parcel_height_m,
        ),
    )
    parcel_asset = stage.DefinePrim(f"{{parcel_path}}/Asset", "Xform")
    parcel_asset.GetReferences().AddReference(parcel_carton_asset)
    disable_composed_collisions(parcel_asset)
    set_xform(
        parcel_asset,
        (0.0, 0.0, -0.5 * parcel_height_m),
        scale=(
            parcel_length_m / 0.40,
            parcel_width_m / 0.26,
            parcel_height_m / 0.16,
        ),
    )
    semantic = Semantics.SemanticsAPI.Apply(parcel, "Semantics")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set("parcel")

# Outer storage is assembled exclusively from official Simple Warehouse rack
# shelves, frames, and carton assets. Cartons sit near the aisle-facing shelf
# edge and use deterministic size/rotation profiles so the racks look stocked
# rather than populated by evenly spaced miniature copies.
rack_shelf_asset = {RACK_SHELF_ASSET!r}
rack_frame_asset = {RACK_FRAME_ASSET!r}
shelf_carton_profiles = (
    ((1.46, 1.64, 1.24), -4.0),
    ((2.08, 1.54, 1.92), 88.0),
    ((1.68, 2.18, 1.48), 2.0),
    ((2.22, 1.76, 1.72), 92.0),
    ((1.54, 2.04, 1.34), -2.0),
    ((2.04, 1.50, 2.02), 89.0),
)
shelf_carton_depth_jitter_m = (-0.06, 0.04, -0.02, 0.07)
shelf_carton_front_offset_m = 0.30
shelf_carton_gap_m = 0.04
rack_upright_x_offsets_m = (-0.47, 0.47)
rack_upright_half_extents_m = (0.065, 0.075, 1.50)
# SM_CardBoxD_04's authored pivot and local bound minimum are at the bottom of
# the carton, not at its centre. SM_RackShelf_01's top surface is 0.0254 m
# above its authored origin. Seat every carton from those two measured facts;
# adding a scaled half-height here makes the boxes visibly float.
shelf_surface_offset_m = 0.026
shelf_carton_seating_gap_m = 0.004
shelf_level_z_values = (0.48, 1.55, 2.62)
shelf_collision_half_thickness_m = 0.015
shelf_collision_plane_count = 0
rack_upright_collider_count = 0
shelf_carton_count = 0
for side, x in (("West", -4.20),):
    rack_root = f"{{{SCENE_ROOT!r}}}/Storage/{{side}}Rack"
    UsdGeom.Xform.Define(stage, rack_root)
    aisle_direction = 1.0 if side == "West" else -1.0
    carton_base_x = x + aisle_direction * shelf_carton_front_offset_m
    rack_collider_root = (
        f"{{{PHYSICS_ROOT!r}}}/StaticColliders/{{side}}Rack"
    )
    UsdGeom.Scope.Define(stage, rack_collider_root)
    # One continuous thin slab per tier avoids collision seams where the two
    # visual shelf assets overlap. A narrow rear guard represents the rack
    # frame without filling the entire stocked volume with one solid box.
    for level_index, level_z in enumerate(shelf_level_z_values):
        collision_cube(
            f"{{rack_collider_root}}/ShelfPlane{{level_index + 1}}",
            (
                x,
                0.0,
                level_z
                + shelf_surface_offset_m
                - shelf_collision_half_thickness_m,
            ),
            (
                0.535,
                rack_half_length_m + 0.26,
                shelf_collision_half_thickness_m,
            ),
        )
        shelf_collision_plane_count += 1
    collision_cube(
        f"{{rack_collider_root}}/RearGuard",
        (x - aisle_direction * 0.50, 0.0, 1.50),
        (0.04, rack_half_length_m + 0.26, 1.50),
    )
    for frame_index, frame_y in enumerate((-3.33, 0.0, 3.33)):
        frame = reference_asset(
            f"{{rack_root}}/Frame{{frame_index + 1}}",
            rack_frame_asset,
            translate=(x, frame_y, 0.0),
            rotate=(0.0, 0.0, 90.0),
        )
        disable_composed_collisions(frame)
        # The official frame combines both 3 m uprights and diagonal braces
        # into visual meshes. Two narrow boxes per frame protect only the
        # vertical beams, preserving open shelf access and cheap contacts.
        for upright_index, upright_x_offset in enumerate(
            rack_upright_x_offsets_m
        ):
            upright = collision_cube(
                f"{{rack_collider_root}}/Frame{{frame_index + 1}}"
                f"Upright{{upright_index + 1}}",
                (x + upright_x_offset, frame_y, 1.50),
                rack_upright_half_extents_m,
            )
            attr(
                upright,
                "codex:colliderPurpose",
                Sdf.ValueTypeNames.String,
                "rack_vertical_upright",
            )
            rack_upright_collider_count += 1
    for segment_index, segment_y in enumerate((-1.67, 1.67)):
        for level_index, level_z in enumerate(shelf_level_z_values):
            shelf = reference_asset(
                f"{{rack_root}}/Segment{{segment_index + 1}}/"
                f"Shelf{{level_index + 1}}",
                rack_shelf_asset,
                translate=(x, segment_y, level_z),
                rotate=(0.0, 0.0, 90.0),
            )
            disable_composed_collisions(shelf)
        for level_index, level_z in enumerate(shelf_level_z_values):
            # Pack each row from its actual rotated carton widths. This keeps
            # a real 4 cm gap between neighboring rigid bodies and centers the
            # complete row between the two frame uprights, regardless of the
            # deterministic size/yaw profile used on that tier.
            row_specs = []
            for carton_index in range(4):
                profile_index = (
                    segment_index * 5
                    + level_index * 3
                    + carton_index
                    + (1 if side == "East" else 0)
                ) % len(shelf_carton_profiles)
                scale, yaw = shelf_carton_profiles[profile_index]
                carton_dimensions_m = (
                    0.40 * float(scale[0]),
                    0.26 * float(scale[1]),
                    0.16 * float(scale[2]),
                )
                yaw_radians = math.radians(yaw)
                projected_half_y_m = 0.5 * (
                    abs(math.sin(yaw_radians)) * carton_dimensions_m[0]
                    + abs(math.cos(yaw_radians)) * carton_dimensions_m[1]
                )
                row_specs.append(
                    (scale, yaw, carton_dimensions_m, projected_half_y_m)
                )
            row_span_m = (
                sum(2.0 * spec[3] for spec in row_specs)
                + shelf_carton_gap_m * (len(row_specs) - 1)
            )
            row_cursor_y_m = segment_y - 0.5 * row_span_m
            for carton_index, (
                scale,
                yaw,
                carton_dimensions_m,
                projected_half_y_m,
            ) in enumerate(row_specs):
                carton_x = (
                    carton_base_x
                    + aisle_direction
                    * shelf_carton_depth_jitter_m[
                        (segment_index * 2 + level_index + carton_index)
                        % len(shelf_carton_depth_jitter_m)
                    ]
                )
                carton_y = row_cursor_y_m + projected_half_y_m
                row_cursor_y_m = (
                    carton_y
                    + projected_half_y_m
                    + shelf_carton_gap_m
                )
                carton_z = (
                    level_z
                    + shelf_surface_offset_m
                    + shelf_carton_seating_gap_m
                )
                shelf_carton_path = (
                    f"{{rack_root}}/Segment{{segment_index + 1}}/"
                    f"Level{{level_index + 1}}"
                    f"Carton{{carton_index + 1}}"
                )
                carton = UsdGeom.Xform.Define(
                    stage,
                    shelf_carton_path,
                ).GetPrim()
                set_xform(
                    carton,
                    (carton_x, carton_y, carton_z),
                    rotate=(0.0, 0.0, yaw),
                )
                carton_asset = stage.DefinePrim(
                    f"{{shelf_carton_path}}/Asset",
                    "Xform",
                )
                carton_asset.GetReferences().AddReference(
                    parcel_carton_asset
                )
                disable_composed_collisions(carton_asset)
                set_xform(
                    carton_asset,
                    (0.0, 0.0, 0.0),
                    scale=scale,
                )
                shelf_carton_mass_kg = max(
                    2.0,
                    carton_dimensions_m[0]
                    * carton_dimensions_m[1]
                    * carton_dimensions_m[2]
                    * 42.0,
                )
                shelf_body = UsdPhysics.RigidBodyAPI.Apply(carton)
                shelf_body.CreateRigidBodyEnabledAttr().Set(True)
                UsdPhysics.MassAPI.Apply(
                    carton
                ).CreateMassAttr().Set(shelf_carton_mass_kg)
                shelf_physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(
                    carton
                )
                shelf_physx_body.CreateEnableCCDAttr().Set(True)
                shelf_physx_body.CreateLinearDampingAttr().Set(0.12)
                shelf_physx_body.CreateAngularDampingAttr().Set(0.35)
                collision_cube(
                    f"{{shelf_carton_path}}/Collision",
                    (
                        0.0,
                        0.0,
                        0.5 * carton_dimensions_m[2],
                    ),
                    (
                        0.5 * carton_dimensions_m[0],
                        0.5 * carton_dimensions_m[1],
                        0.5 * carton_dimensions_m[2],
                    ),
                )
                attr(
                    carton,
                    "codex:dimensionsM",
                    Sdf.ValueTypeNames.Double3,
                    Gf.Vec3d(*carton_dimensions_m),
                )
                attr(
                    carton,
                    "codex:massKg",
                    Sdf.ValueTypeNames.Double,
                    shelf_carton_mass_kg,
                )
                attr(
                    carton,
                    "codex:shelfParcel",
                    Sdf.ValueTypeNames.Bool,
                    True,
                )
                shelf_semantic = Semantics.SemanticsAPI.Apply(
                    carton,
                    "Semantics",
                )
                shelf_semantic.CreateSemanticTypeAttr().Set("class")
                shelf_semantic.CreateSemanticDataAttr().Set("parcel")
                shelf_carton_count += 1
attr(
    root,
    "codex:shelfCartonCount",
    Sdf.ValueTypeNames.Int,
    shelf_carton_count,
)
attr(
    root,
    "codex:shelfCollisionPlaneCount",
    Sdf.ValueTypeNames.Int,
    shelf_collision_plane_count,
)
attr(
    root,
    "codex:rackUprightColliderCount",
    Sdf.ValueTypeNames.Int,
    rack_upright_collider_count,
)
attr(
    root,
    "codex:dynamicShelfCartonsEnabled",
    Sdf.ValueTypeNames.Bool,
    True,
)

# Preserve the user's official Isaac packing table placement. Its detailed
# composed colliders are disabled in favor of one invisible box: cheaper for
# PhysX, sufficient for forklift contact, and a reliable tabletop for props.
packing_table = reference_asset(
    f"{{{SCENE_ROOT!r}}}/packing_table",
    {PACKING_TABLE_ASSET!r},
    translate=(packing_table_center_m[0], packing_table_center_m[1], 0.0),
)
disable_composed_collisions(packing_table)
packing_table_collider = collision_cube(
    f"{{{PHYSICS_ROOT!r}}}/StaticColliders/PackingTable",
    packing_table_center_m,
    packing_table_half_extents_m,
)
attr(
    packing_table_collider,
    "codex:colliderPurpose",
    Sdf.ValueTypeNames.String,
    "simple_packing_table_box",
)

# Two independently controllable ForkliftC articulations face opposite
# directions in the west aisle. This Isaac Sim robot has four driven wheel
# joints, two rear steering joints, a lift joint, rigid chassis/fork links,
# and authored collision shapes.
UsdGeom.Xform.Define(stage, {FORKLIFT_ROOT!r})
forklift_asset = {FORKLIFT_ASSET!r}
forklift_starts = (
    (-1.00, -1.85, 0.10, 0.0),
    (-0.50, 0.85, 0.10, 180.0),
)
id_colors = (
    (0.12, 0.72, 1.0),
    (0.42, 1.0, 0.36),
    (0.95, 0.28, 0.82),
)
forklift_paths = [
    f"{{{FORKLIFT_ROOT!r}}}/Forklift{{index + 1}}"
    for index in range(len(forklift_starts))
]
source_path = forklift_paths[0]
source = stage.DefinePrim(source_path, "Xform")
source.GetReferences().AddReference(forklift_asset)
source.SetInstanceable(False)
source_xform = UsdGeom.Xformable(source)
source_xform.ClearXformOpOrder()
source_xform.AddTranslateOp(
    UsdGeom.XformOp.PrecisionDouble
).Set(Gf.Vec3d(*forklift_starts[0][:3]))
source_xform.AddRotateXYZOp(
    UsdGeom.XformOp.PrecisionFloat
).Set(Gf.Vec3f(0.0, 0.0, forklift_starts[0][3]))
# ForkliftC is authored in centimetres and supplies a composed 0.01 scale
# attribute. Clearing the wrapper's xform order above would otherwise leave
# that attribute unordered, making the robot 100x too large and invalidating
# both its rendered bounds and PhysX contacts.
source_xform.AddScaleOp(
    UsdGeom.XformOp.PrecisionDouble
).Set(Gf.Vec3d(0.01, 0.01, 0.01))

# Use Isaac Sim's Cloner to give each repeated robot an independent USD and
# PhysX articulation. Plain repeated references can alias articulation state
# in a live stage; copy_from_source avoids shared joint/collision state.
if len(forklift_paths) > 1:
    clone_positions = np.array(
        [start[:3] for start in forklift_starts[1:]],
        dtype=np.float64,
    )
    clone_orientations = np.array(
        [
            (
                math.cos(math.radians(start[3]) * 0.5),
                0.0,
                0.0,
                math.sin(math.radians(start[3]) * 0.5),
            )
            for start in forklift_starts[1:]
        ],
        dtype=np.float64,
    )
    Cloner(stage=stage).clone(
        source_prim_path=source_path,
        prim_paths=forklift_paths[1:],
        positions=clone_positions,
        orientations=clone_orientations,
        copy_from_source=True,
    )

for index, ((x, y, z, yaw), wrapper_path) in enumerate(
    zip(forklift_starts, forklift_paths)
):
    wrapper = stage.GetPrimAtPath(wrapper_path)
    attr(wrapper, "codex:forkliftIndex", Sdf.ValueTypeNames.Int, index)
    attr(wrapper, "codex:label", Sdf.ValueTypeNames.String, f"Forklift {{index + 1}}")
    attr(wrapper, "codex:forkHeightM", Sdf.ValueTypeNames.Double, 0.0)
    body = stage.GetPrimAtPath(f"{{wrapper_path}}/body")
    if body.IsValid():
        PhysxSchema.PhysxRigidBodyAPI(body).GetEnableCCDAttr().Set(True)
    halo = cylinder(
        f"{{wrapper_path}}/body/ActiveHalo",
        (0.0, 0.0, 2.52),
        0.34,
        0.10,
        id_colors[index],
    )
    attr(halo, "codex:active", Sdf.ValueTypeNames.Bool, index == 0)
    # The official pallet and carton are separate rigid bodies. The pallet
    # rests on the articulation's colliding fork links, and the larger carton
    # rests on the pallet through contact physics. Nothing is parented to the
    # moving mast, so abrupt steering or fork motion can slide, tip, and drop
    # either body.
    yaw_radians = math.radians(yaw)
    forward_x = math.cos(yaw_radians)
    forward_y = math.sin(yaw_radians)
    cargo_center_x = x + forward_x * {FORK_CARGO_FORWARD_OFFSET_M!r}
    cargo_center_y = y + forward_y * {FORK_CARGO_FORWARD_OFFSET_M!r}
    pallet_path = f"{{{CARGO_ROOT!r}}}/Forklift{{index + 1}}Pallet"
    fork_pallet = UsdGeom.Xform.Define(stage, pallet_path).GetPrim()
    set_xform(
        fork_pallet,
        (cargo_center_x, cargo_center_y, {FORK_PALLET_BASE_Z_M!r}),
        rotate=(0.0, 0.0, yaw + {FORK_PALLET_YAW_OFFSET_DEGREES!r}),
    )
    pallet_body = UsdPhysics.RigidBodyAPI.Apply(fork_pallet)
    pallet_body.CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(fork_pallet).CreateMassAttr().Set(22.0)
    pallet_physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(fork_pallet)
    pallet_physx_body.CreateEnableCCDAttr().Set(True)
    pallet_physx_body.CreateLinearDampingAttr().Set(0.35)
    pallet_physx_body.CreateAngularDampingAttr().Set(0.80)
    pallet_physx_body.CreateMaxDepenetrationVelocityAttr().Set(0.75)
    pallet_physx_body.CreateMaxLinearVelocityAttr().Set(4.0)
    pallet_physx_body.CreateMaxAngularVelocityAttr().Set(180.0)
    pallet_physx_body.CreateSolverPositionIterationCountAttr().Set(8)
    pallet_physx_body.CreateSolverVelocityIterationCountAttr().Set(2)
    # Measured composed pallet bounds are 1.2132 x 0.8023 x 0.1425 m. Four
    # inexpensive boxes approximate its load-bearing structure while leaving
    # two open channels for the real forklift tines: a thin upper deck and
    # three lower runners. This admits the forks instead of treating the
    # pallet as an unrealistic solid slab.
    collision_cube(
        f"{{pallet_path}}/Collision",
        (0.00154, 0.00115, 0.1195),
        (0.615, 0.407, 0.024),
    )
    for runner_name, runner_x, runner_half_x in (
        ("LeftRunner", -0.565, 0.040),
        ("CenterRunner", 0.0, 0.045),
        ("RightRunner", 0.565, 0.040),
    ):
        collision_cube(
            f"{{pallet_path}}/{{runner_name}}",
            (runner_x, 0.00115, 0.015),
            (runner_half_x, 0.407, 0.015),
        )
    attr(
        fork_pallet,
        "codex:colliderCenterM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.00154, 0.00115, 0.1195),
    )
    attr(
        fork_pallet,
        "codex:colliderHalfExtentsM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.615, 0.407, 0.024),
    )
    attr(
        fork_pallet,
        "codex:visualDimensionsM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(1.21323, 0.80230, 0.14251),
    )
    attr(fork_pallet, "codex:massKg", Sdf.ValueTypeNames.Double, 22.0)
    attr(fork_pallet, "codex:colliderShapeCount", Sdf.ValueTypeNames.Int, 4)
    attr(
        fork_pallet,
        "codex:colliderModel",
        Sdf.ValueTypeNames.String,
        "simple_top_deck_plus_three_runners",
    )
    attr(fork_pallet, "codex:forkliftPallet", Sdf.ValueTypeNames.Bool, True)
    fork_pallet_asset = stage.DefinePrim(f"{{pallet_path}}/Asset", "Xform")
    fork_pallet_asset.GetReferences().AddReference({FORK_PALLET_ASSET!r})
    disable_composed_collisions(fork_pallet_asset)
    set_xform(fork_pallet_asset, (0.0, 0.0, 0.0))

    cargo_path = f"{{{CARGO_ROOT!r}}}/Forklift{{index + 1}}Carton"
    fork_carton = UsdGeom.Xform.Define(stage, cargo_path).GetPrim()
    set_xform(
        fork_carton,
        (cargo_center_x, cargo_center_y, {FORK_CARTON_BASE_Z_M!r}),
        rotate=(0.0, 0.0, yaw),
    )
    cargo_body = UsdPhysics.RigidBodyAPI.Apply(fork_carton)
    cargo_body.CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(fork_carton).CreateMassAttr().Set(10.5)
    carton_physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(fork_carton)
    carton_physx_body.CreateEnableCCDAttr().Set(True)
    carton_physx_body.CreateLinearDampingAttr().Set(0.30)
    carton_physx_body.CreateAngularDampingAttr().Set(0.80)
    carton_physx_body.CreateMaxDepenetrationVelocityAttr().Set(0.75)
    carton_physx_body.CreateMaxLinearVelocityAttr().Set(4.0)
    carton_physx_body.CreateMaxAngularVelocityAttr().Set(180.0)
    carton_physx_body.CreateSolverPositionIterationCountAttr().Set(8)
    carton_physx_body.CreateSolverVelocityIterationCountAttr().Set(2)
    # Enlarge SM_CardBoxA_01 to 0.72 x 0.68 x 0.50 m. The collider adds a
    # 1 cm face margin and starts 1 cm below the visual so it settles cleanly
    # onto the pallet's upper deck.
    collision_cube(
        f"{{cargo_path}}/Collision",
        (0.0, 0.0, 0.25),
        (0.37, 0.35, 0.26),
    )
    attr(
        fork_carton,
        "codex:colliderCenterM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.0, 0.0, 0.25),
    )
    attr(
        fork_carton,
        "codex:colliderHalfExtentsM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.37, 0.35, 0.26),
    )
    attr(
        fork_carton,
        "codex:visualDimensionsM",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.72, 0.68, 0.50),
    )
    attr(fork_carton, "codex:massKg", Sdf.ValueTypeNames.Double, 10.5)
    fork_carton_asset = stage.DefinePrim(f"{{cargo_path}}/Asset", "Xform")
    fork_carton_asset.GetReferences().AddReference({FORK_CARTON_ASSET!r})
    disable_composed_collisions(fork_carton_asset)
    set_xform(
        fork_carton_asset,
        (0.0, 0.0, 0.0),
        scale=(1.03, 1.36, 1.00),
    )
    attr(
        fork_carton,
        "codex:forkliftPerceptionCargo",
        Sdf.ValueTypeNames.Bool,
        True,
    )
    semantic = Semantics.SemanticsAPI.Apply(fork_carton, "Semantics")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set(
        f"forklift_proximity_carton_{{index + 1}}"
    )

# Real Isaac Sim construction-worker characters deliberately operate inside
# the red policy zone. Their routes skim the west side of the oval without
# crossing its physical track. Worker 2 pauses close enough to handle a parcel
# while the segment-based avoidance geometry prevents stepping into the belt.
UsdGeom.Xform.Define(stage, {WORKER_ROOT!r})
worker_routes = (
    (
        "oval_conveyor_rack_rectangle",
        (
            (1.93, -4.45, 0.0),
            (-3.10, -4.45, 0.0),
            (-3.10, 2.10, 0.0),
            (0.60, 2.10, 0.0),
            (1.93, 2.78, 0.0),
            (1.93, -4.45, 0.0),
        ),
        (0.0, 0.0, 0.0, 0.0, 0.8, 0.0),
        (1000.0, 1000.0, 1000.0, 1000.0, 90.0, 1000.0),
        True,
        180.0,
    ),
    (
        "north_shelf_to_oval_shuttle",
        (
            (-3.10, 2.78, 0.0),
            (-3.10, 2.10, 0.0),
            (0.60, 2.10, 0.0),
            (1.93, 2.78, 0.0),
            (0.60, 2.10, 0.0),
            (-3.10, 2.10, 0.0),
            (-3.10, 2.78, 0.0),
        ),
        (0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (1000.0, 1000.0, 1000.0, 90.0, 1000.0, 1000.0, 1000.0),
        True,
        90.0,
    ),
)
worker_assets = {WORKER_ASSETS!r}
for index, (
    route_name,
    waypoints,
    dwell_seconds,
    facing_yaw_degrees,
    touches_belt,
    yaw,
) in enumerate(worker_routes):
    worker_path = f"{{{WORKER_ROOT!r}}}/Worker{{index + 1}}"
    worker = UsdGeom.Xform.Define(stage, worker_path).GetPrim()
    set_xform(worker, waypoints[0], rotate=(0.0, 0.0, yaw))
    attr(
        worker,
        "codex:pathStart",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(*waypoints[0]),
    )
    attr(
        worker,
        "codex:pathEnd",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(*waypoints[1]),
    )
    attr(
        worker,
        "codex:routeName",
        Sdf.ValueTypeNames.String,
        route_name,
    )
    attr(
        worker,
        "codex:routeWaypoints",
        Sdf.ValueTypeNames.Double3Array,
        [Gf.Vec3d(*point) for point in waypoints],
    )
    attr(
        worker,
        "codex:routeDwellSeconds",
        Sdf.ValueTypeNames.DoubleArray,
        list(dwell_seconds),
    )
    attr(
        worker,
        "codex:routeFacingYawDegrees",
        Sdf.ValueTypeNames.DoubleArray,
        list(facing_yaw_degrees),
    )
    attr(worker, "codex:navigationBlocked", Sdf.ValueTypeNames.Bool, False)
    attr(worker, "codex:navigationBlockedBy", Sdf.ValueTypeNames.String, "")
    attr(worker, "codex:navigationReplanCount", Sdf.ValueTypeNames.Int, 0)
    attr(worker, "codex:phase", Sdf.ValueTypeNames.Double, 0.0)
    attr(worker, "codex:touchesParcels", Sdf.ValueTypeNames.Bool, touches_belt)
    attr(worker, "codex:characterAsset", Sdf.ValueTypeNames.Asset, worker_assets[index])
    character = stage.DefinePrim(f"{{worker_path}}/Character", "Xform")
    character.GetReferences().AddReference(worker_assets[index])
    # Worker response is implemented by the extension's damped scripted push.
    # Disable any composed rigid collision shapes so a worker cannot become a
    # physics wedge between the forklift and the conveyor's hard proxy box.
    disable_composed_collisions(character)

# ForkliftC is authored in centimetres below a 0.01-scale wrapper. Mount one
# official construction worker under each available body so the driver follows
# the articulation exactly. The static seated animation is bound after the
# Human Motion Library payload finishes loading.
driver_asset = worker_assets[2]
driver_mount_paths = []
for index, wrapper_path in enumerate(forklift_paths):
    body = stage.GetPrimAtPath(f"{{wrapper_path}}/body")
    if not body.IsValid():
        continue
    driver_mount_path = f"{{wrapper_path}}/body/DriverMount"
    driver_mount = UsdGeom.Xform.Define(stage, driver_mount_path).GetPrim()
    set_xform(
        driver_mount,
        (-62.0, 4.0, 77.0),
        scale=(100.0, 100.0, 100.0),
        # The ForkliftC mast/controls face local +X. The construction worker's
        # authored forward axis requires the opposite quarter-turn so the
        # seated driver looks toward the steering wheel rather than the tank.
        rotate=(0.0, 0.0, -90.0),
    )
    attr(
        driver_mount,
        "codex:characterAsset",
        Sdf.ValueTypeNames.Asset,
        driver_asset,
    )
    attr(
        driver_mount,
        "codex:driverForForklift",
        Sdf.ValueTypeNames.Int,
        index + 1,
    )
    driver_character = stage.DefinePrim(
        f"{{driver_mount_path}}/Character",
        "Xform",
    )
    driver_character.GetReferences().AddReference(driver_asset)
    disable_composed_collisions(driver_character)
    driver_mount_paths.append(driver_mount_path)

# Use the official Isaac Sim Human Motion Library directly through USD
# skeleton bindings. This keeps the realistic characters naturally posed and
# animated without coupling their bounded tutorial paths to navmesh behavior.
motion_library_path = f"{{{WORKER_ROOT!r}}}/HumanMotionLibrary"
motion_library_asset = {MOTION_LIBRARY_ASSET!r}
motion_library = stage.DefinePrim(motion_library_path, "Xform")
motion_library.GetPayloads().AddPayload(motion_library_asset)

# Preserve the three realistic-scale, base-free stack lights from the manual
# edit. All child coordinates are local to the uniformly scaled tower root so
# a future placement tweak can move a whole tower without separating its wash.
stack_light_roots = {STACK_LIGHT_ROOTS!r}
stack_light_positions = (
    (-3.650637, -3.510892, 2.832299),
    (4.351811, 3.370365, 1.790201),
    (4.420411, -2.041461, 1.446061),
)
stack_light_bounce_positions = (
    (-3.50, -3.25, 3.45),
    (4.05, 3.05, 2.35),
    (4.42, -2.04, 2.45),
)
stack_light_scale = 0.216
# Treat the emitters as point lights and keep their fallback sphere geometry
# tiny.  Otherwise RTX renders a visible gray emitter disk on nearby walls.
stack_lens_glow_radius = 0.01
stack_scene_wash_radius = 0.01
stack_scene_bounce_radius = 0.30
stack_scene_bounce_intensity = 220000.0
stack_lens_specs = (
    ("Green", 0.245638, (0.04, 1.0, 0.14)),
    ("Amber", 0.578879, (1.0, 0.54, 0.02)),
    ("Red", 0.920017, (1.0, 0.03, 0.02)),
)
for stack_root_path, stack_position in zip(
    stack_light_roots,
    stack_light_positions,
):
    stack_root = UsdGeom.Xform.Define(stage, stack_root_path).GetPrim()
    set_xform(
        stack_root,
        stack_position,
        scale=(stack_light_scale,) * 3,
    )
    cylinder(
        f"{{stack_root_path}}/Pole",
        (0.0, 0.0, 0.0),
        0.279,
        2.875,
        (0.25, 0.28, 0.30),
    )
    for name, local_z, color in stack_lens_specs:
        display_color = color if name == "Green" else tuple(
            channel * 0.10 for channel in color
        )
        lens = cylinder(
            f"{{stack_root_path}}/{{name}}Lens",
            (0.0, 0.0, local_z),
            0.28,
            0.34,
            display_color,
        )
        attr(
            lens,
            "codex:nominalColor",
            Sdf.ValueTypeNames.Color3f,
            Gf.Vec3f(*color),
        )
        glow = UsdLux.SphereLight.Define(
            stage,
            f"{{stack_root_path}}/{{name}}Lens/Glow",
        )
        glow.CreateRadiusAttr(stack_lens_glow_radius)
        glow.CreateTreatAsPointAttr(True)
        glow.CreateColorAttr(Gf.Vec3f(*color))
        glow.CreateIntensityAttr(60000.0 if name == "Green" else 0.0)
    wash = UsdLux.SphereLight.Define(
        stage,
        f"{{stack_root_path}}/SceneWash",
    )
    wash.CreateRadiusAttr(stack_scene_wash_radius)
    wash.CreateTreatAsPointAttr(True)
    wash.CreateColorAttr(Gf.Vec3f(0.04, 1.0, 0.14))
    wash.CreateIntensityAttr(50000.0)
    set_xform(wash.GetPrim(), (0.0, 0.0, 0.465762))

# RTX real-time renders the tiny point-treated lens lights cleanly, but their
# local wash is largely occluded by the pole and mounting geometry. Three
# world-space downward DiskLights provide the dramatic colored bounce without
# putting a visible emitter disk in any of the normal overhead camera views.
UsdGeom.Scope.Define(stage, f"{{{SCENE_ROOT!r}}}/Lighting")
for bounce_path, bounce_position in zip(
    {STACK_LIGHT_BOUNCE_PATHS!r},
    stack_light_bounce_positions,
):
    bounce = UsdLux.DiskLight.Define(stage, bounce_path)
    bounce.CreateRadiusAttr(stack_scene_bounce_radius)
    bounce.CreateColorAttr(Gf.Vec3f(0.04, 1.0, 0.14))
    bounce.CreateIntensityAttr(stack_scene_bounce_intensity)
    set_xform(bounce.GetPrim(), bounce_position)

# Camera experiment set. Detector captures still hide the active-forklift
# selection halos because those are operator UI, not physical scene lighting.
UsdGeom.Xform.Define(stage, {CAMERA_ROOT!r})
look_at({CAMERA_PATHS["detector_oblique"]!r}, (8.0, -7.0, 8.0), (0.4, 0.0, 0.70), 19.0)
look_at(
    {CAMERA_PATHS["detector_overhead"]!r},
    (0.2, 0.0, 12.5),
    (0.2, 0.0, 0.0),
    20.0,
    # Rotate the long conveyor into the 16:9 image's wide dimension. This
    # makes the forklift, its forks, and both one-metre boundaries larger.
    up=(1.0, 0.0, 0.0),
)
look_at(
    {CAMERA_PATHS["detector_ptz"]!r},
    (0.0, 0.0, 9.0),
    (0.0, 0.0, 0.25),
    22.0,
    up=(-1.0, 0.0, 0.0),
)
detector_endline_camera = UsdGeom.Camera(look_at(
    {CAMERA_PATHS["detector_endline"]!r},
    (marked_zone_west_edge_m, -6.10, 2.45),
    (marked_zone_west_edge_m, 0.10, 0.80),
    9.5,
))
# Put the camera and optical axis directly on the red-zone west edge so the
# boundary projects as a straight sightline into the horizon. Shift the sensor
# window toward the aisle to retain both forklifts and the west rack.
detector_endline_camera.CreateHorizontalApertureOffsetAttr(-3.0)
look_at({CAMERA_PATHS["detector_side"]!r}, (-9.0, 0.0, 8.0), (0.2, 0.0, 0.70), 20.0)
isometric_camera = UsdGeom.Camera(
    look_at(
        {CAMERA_PATHS["detector_isometric"]!r},
        (-7.0, -7.0, 7.8),
        (0.55, 0.0, 0.55),
        20.0,
    )
)
isometric_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
isometric_camera.GetHorizontalApertureAttr().Set(140.0)
isometric_camera.GetVerticalApertureAttr().Set(78.75)
look_at(
    {CAMERA_PATHS["presentation"]!r},
    (9.0, -9.0, 9.0),
    (0.80, -0.30, 0.80),
    18.0,
)

dome = UsdLux.DomeLight.Define(stage, f"{{{SCENE_ROOT!r}}}/Lighting/Dome")
dome.CreateIntensityAttr(700.0)
dome.CreateColorAttr(Gf.Vec3f(0.90, 0.93, 1.0))
for index, (x, y) in enumerate(((-3.2, -2.5), (3.2, -2.5), (-3.2, 3.5), (3.2, 3.5))):
    light = UsdLux.DiskLight.Define(stage, f"{{{SCENE_ROOT!r}}}/Lighting/Area{{index + 1}}")
    light.CreateIntensityAttr(18000.0)
    light.CreateRadiusAttr(2.2)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.78))
    set_xform(light.GetPrim(), (x, y, 7.0), rotate=(0.0, 0.0, 0.0))

from omni.kit.viewport.utility import get_active_viewport
viewport = get_active_viewport()
if viewport is not None:
    viewport.camera_path = Sdf.Path({CAMERA_PATHS["presentation"]!r})
context.get_selection().set_selected_prim_paths([{SCENE_ROOT!r}], False)
for _ in range(60):
    await omni.kit.app.get_app().next_update_async()

# Isaac's current construction workers use the 101-joint RL_BoneRoot rig,
# while the Human Motion Library uses an 81-joint Root/Pelvis rig. A direct USD
# animation relationship is accepted but leaves the character in bind pose.
# Bake the required clips through Isaac's official retargeter before binding.
retarget_extension_name = "omni.anim.retarget.core"
if not extension_manager.is_extension_enabled(retarget_extension_name):
    extension_manager.set_extension_enabled_immediate(
        retarget_extension_name,
        True,
    )
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()

source_skeleton_path = f"{{motion_library_path}}/Skeleton"
target_skeleton_path = None
for driver_mount_path in driver_mount_paths:
    driver_mount = stage.GetPrimAtPath(driver_mount_path)
    for prim in Usd.PrimRange(driver_mount):
        if prim.IsA(UsdSkel.Skeleton):
            target_skeleton_path = str(prim.GetPath())
            break
    if target_skeleton_path is not None:
        break
if target_skeleton_path is None:
    raise RuntimeError("A driver target skeleton was not found")

worker_walk_source_names = ("WalkForward", "WalkForward_01")
worker_animation_names = ("WalkForwardLoop", "WalkForwardLoop_01")
worker_idle_animation_names = ("Idle", "IdleTired")
driver_pose_source = (
    f"{{motion_library_path}}/BuiltinActions/Sit/SitAndStandChair"
)
source_animation_paths = [
    f"{{motion_library_path}}/BuiltinActions/MoveWalk/{{name}}"
    for name in worker_walk_source_names
]
source_animation_paths.extend(
    f"{{motion_library_path}}/BuiltinActions/Idle/{{name}}"
    for name in worker_idle_animation_names
)
source_animation_paths.append(driver_pose_source)
retargeted_animation_root = (
    f"{{{WORKER_ROOT!r}}}/RetargetedAnimations"
)
UsdGeom.Scope.Define(stage, retargeted_animation_root)
omni.kit.commands.execute(
    "CreateRetargetAnimationsCommand",
    source_skeleton_path=source_skeleton_path,
    target_skeleton_path=target_skeleton_path,
    source_animation_paths=source_animation_paths,
    target_animation_parent_path=retargeted_animation_root,
    set_root_identity=True,
    allow_default_transforms=False,
)
for _ in range(3):
    await omni.kit.app.get_app().next_update_async()

# The stock WalkForward clips begin with almost one second of standing and
# weight shifting. Extract a matched gait cycle from the locomotion section,
# repeat it twice, and bind that visibly active loop to the patrol workers.
walking_loop_source_start_time_code = 154
walking_loop_source_end_time_code = 228
walking_loop_cycle_time_codes = (
    walking_loop_source_end_time_code
    - walking_loop_source_start_time_code
)
walking_loop_duration_time_codes = walking_loop_cycle_time_codes * 2
for source_name, loop_name in zip(
    worker_walk_source_names,
    worker_animation_names,
):
    source_walk_path = (
        f"{{retargeted_animation_root}}/{{source_name}}"
    )
    source_walk_prim = stage.GetPrimAtPath(source_walk_path)
    if not source_walk_prim.IsValid() or not source_walk_prim.IsA(
        UsdSkel.Animation
    ):
        raise RuntimeError(
            f"Retargeted worker walk clip was not created: {{source_walk_path}}"
        )
    source_walk = UsdSkel.Animation(source_walk_prim)
    source_walk_joints = list(source_walk.GetJointsAttr().Get() or [])
    root_motion_joint_indices = [
        joint_index
        for joint_index, joint_name in enumerate(source_walk_joints)
        if str(joint_name) == "RL_BoneRoot/Hip"
    ]
    if not root_motion_joint_indices:
        raise RuntimeError(
            f"Retargeted walk clip has no root hip: {{source_walk_path}}"
        )
    loop_walk_path = f"{{retargeted_animation_root}}/{{loop_name}}"
    loop_walk = UsdSkel.Animation.Define(stage, loop_walk_path)
    loop_walk.CreateJointsAttr().Set(source_walk_joints)
    loop_translations = loop_walk.CreateTranslationsAttr()
    loop_rotations = loop_walk.CreateRotationsAttr()
    loop_scales = loop_walk.CreateScalesAttr()
    for target_time_code in range(
        0,
        walking_loop_duration_time_codes + 1,
        2,
    ):
        if target_time_code == walking_loop_duration_time_codes:
            source_time_code = walking_loop_source_end_time_code
        else:
            source_time_code = (
                walking_loop_source_start_time_code
                + target_time_code % walking_loop_cycle_time_codes
            )
        source_time = Usd.TimeCode(float(source_time_code))
        target_time = Usd.TimeCode(float(target_time_code))
        translations = source_walk.GetTranslationsAttr().Get(source_time)
        rotations = source_walk.GetRotationsAttr().Get(source_time)
        scales = source_walk.GetScalesAttr().Get(source_time)
        if translations is not None:
            in_place_translations = list(translations)
            for root_motion_index in root_motion_joint_indices:
                root_translation = in_place_translations[root_motion_index]
                # Navigation owns horizontal motion. Retargeted WalkForward
                # otherwise displaces the visible hips by roughly 0.4--0.5 m
                # beyond the collision-controlled Worker Xform, making an
                # actor appear to cross a wall or floor edge.
                in_place_translations[root_motion_index] = Gf.Vec3f(
                    0.0,
                    0.0,
                    float(root_translation[2]),
                )
            loop_translations.Set(in_place_translations, target_time)
        if rotations is not None:
            loop_rotations.Set(list(rotations), target_time)
        if scales is not None:
            loop_scales.Set(list(scales), target_time)

worker_animation_bindings = []
worker_index = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if (
        path.startswith(f"{{{WORKER_ROOT!r}}}/Worker")
        and prim.IsA(UsdSkel.Root)
    ):
        animation_name = worker_animation_names[
            worker_index % len(worker_animation_names)
        ]
        animation_path = (
            f"{{retargeted_animation_root}}/{{animation_name}}"
        )
        idle_animation_name = worker_idle_animation_names[
            worker_index % len(worker_idle_animation_names)
        ]
        idle_animation_path = (
            f"{{retargeted_animation_root}}/{{idle_animation_name}}"
        )
        UsdSkel.BindingAPI.Apply(prim).CreateAnimationSourceRel().SetTargets(
            [Sdf.Path(animation_path)]
        )
        skeleton_paths = []
        for skeleton_prim in Usd.PrimRange(prim):
            if not skeleton_prim.IsA(UsdSkel.Skeleton):
                continue
            # Hydra does not reliably consume an inherited animation
            # relationship through these referenced construction-worker
            # assets. Bind the concrete skeleton as well as its SkelRoot.
            UsdSkel.BindingAPI.Apply(
                skeleton_prim
            ).CreateAnimationSourceRel().SetTargets(
                [Sdf.Path(animation_path)]
            )
            skeleton_paths.append(str(skeleton_prim.GetPath()))
        worker_wrapper = stage.GetPrimAtPath(
            f"{{{WORKER_ROOT!r}}}/Worker{{worker_index + 1}}"
        )
        attr(
            worker_wrapper,
            "codex:walkAnimation",
            Sdf.ValueTypeNames.String,
            animation_path,
        )
        attr(
            worker_wrapper,
            "codex:idleAnimation",
            Sdf.ValueTypeNames.String,
            idle_animation_path,
        )
        attr(
            worker_wrapper,
            "codex:skelRootPath",
            Sdf.ValueTypeNames.String,
            path,
        )
        worker_animation_bindings.append(
            {{
                "skelroot": path,
                "skeletons": skeleton_paths,
                "walk_animation": animation_path,
                "idle_animation": idle_animation_path,
            }}
        )
        worker_index += 1

# The library's annotation values are 30 fps motion-frame indices while its
# USD joint transforms are sampled every two 60 fps time codes. Time code 422
# is the stable middle of SitLoop. Strip the retargeted actor's horizontal
# walk-to-chair displacement but preserve its seated height and all joint
# rotations, then hold that compatible 101-joint pose for the whole timeline.
driver_pose_sample = 422.0
retargeted_driver_pose_path = (
    f"{{retargeted_animation_root}}/SitAndStandChair"
)
driver_pose_path = (
    f"{{{WORKER_ROOT!r}}}/DriverAnimations/SeatedDriverPose"
)
source_pose_prim = stage.GetPrimAtPath(retargeted_driver_pose_path)
if not source_pose_prim.IsValid() or not source_pose_prim.IsA(UsdSkel.Animation):
    raise RuntimeError(
        "The retargeted Human Motion Library seated pose was not created"
    )
source_pose = UsdSkel.Animation(source_pose_prim)
pose_time = Usd.TimeCode(driver_pose_sample)
pose_joints = list(source_pose.GetJointsAttr().Get() or [])
pose_translations = list(
    source_pose.GetTranslationsAttr().Get(pose_time) or []
)
pose_rotations = list(source_pose.GetRotationsAttr().Get(pose_time) or [])
pose_scales = list(source_pose.GetScalesAttr().Get(pose_time) or [])
for index, joint in enumerate(pose_joints):
    if str(joint) != "RL_BoneRoot/Hip":
        continue
    hip_translation = pose_translations[index]
    pose_translations[index] = Gf.Vec3f(
        0.0,
        0.0,
        float(hip_translation[2]),
    )
    break
driver_whole_body_pitch_degrees = -20.0
driver_whole_body_pitch_radians = math.radians(
    driver_whole_body_pitch_degrees
)
driver_whole_body_pitch = Gf.Quatf(
    math.cos(driver_whole_body_pitch_radians * 0.5),
    Gf.Vec3f(
        math.sin(driver_whole_body_pitch_radians * 0.5),
        0.0,
        0.0,
    ),
)
for index, joint in enumerate(pose_joints):
    if str(joint) != "RL_BoneRoot/Hip":
        continue
    # Rotate the whole skeleton at its root hip pivot. This preserves the
    # mount position while pitching torso, pelvis, arms, and legs together.
    pose_rotations[index] = driver_whole_body_pitch * pose_rotations[index]
    break
UsdGeom.Scope.Define(stage, f"{{{WORKER_ROOT!r}}}/DriverAnimations")
seated_pose = UsdSkel.Animation.Define(stage, driver_pose_path)
seated_pose.CreateJointsAttr().Set(pose_joints)
seated_pose.CreateTranslationsAttr().Set(
    pose_translations,
    Usd.TimeCode(0.0),
)
seated_pose.CreateRotationsAttr().Set(
    pose_rotations,
    Usd.TimeCode(0.0),
)
if pose_scales:
    seated_pose.CreateScalesAttr().Set(
        pose_scales,
        Usd.TimeCode(0.0),
    )

driver_animation_bindings = []
for driver_mount_path in driver_mount_paths:
    driver_mount = stage.GetPrimAtPath(driver_mount_path)
    for prim in Usd.PrimRange(driver_mount):
        if not prim.IsA(UsdSkel.Root):
            continue
        UsdSkel.BindingAPI.Apply(
            prim
        ).CreateAnimationSourceRel().SetTargets([Sdf.Path(driver_pose_path)])
        driver_animation_bindings.append(
            {{"skelroot": str(prim.GetPath()), "animation": driver_pose_path}}
        )

# The extracted gait repeats exactly twice over this interval. Blocked workers
# switch to an idle animation; moving workers never return to the stock clip's
# initial standing frames.
stage.SetStartTimeCode(0.0)
stage.SetEndTimeCode(float(walking_loop_duration_time_codes))
stage.SetTimeCodesPerSecond(60.0)
timeline.set_looping(True)
timeline.set_current_time(0.0)
if {start_playing!r}:
    timeline.play()
    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

print(json.dumps({{
    "scene_root": {SCENE_ROOT!r},
    "new_stage": {new_stage!r},
    "forklift_asset": forklift_asset,
    "forklift_carton_asset": {FORK_CARTON_ASSET!r},
    "forklift_pallet_asset": {FORK_PALLET_ASSET!r},
    "forklift_count": len(forklift_starts),
    "forklift_driving_model": "ForkliftC rear-steer Ackermann articulation",
    "forklift_steering_joints": [
        "left_rotator_joint",
        "right_rotator_joint",
    ],
    "forklift_wheel_joints": [
        "left_front_wheel_joint",
        "right_front_wheel_joint",
        "left_back_wheel_joint",
        "right_back_wheel_joint",
    ],
    "forklift_lift_joint": "lift_joint",
    "forklift_lift_range_m": [0.0, 2.0],
    "dynamic_forklift_cargo_count": len(forklift_starts),
    "dynamic_forklift_pallet_count": len(forklift_starts),
    "forklift_pallet_dimensions_m": [1.21323, 0.80230, 0.14251],
    "forklift_carton_dimensions_m": [0.72, 0.68, 0.50],
    "forklift_pallet_collider": "four_simple_boxes_with_tine_channels",
    "forklift_cargo_stabilization": (
        "ramped_lift_and_chassis_acceleration_plus_capped_depenetration"
    ),
    "forklift_pallet_yaw_offset_degrees": 90.0,
    "collision_blocking": True,
    "obstacle_collision_shape": (
        "continuous_low_poly_oval_mesh_plus_simple_boxes"
    ),
    "conveyor_collider_center_m": [
        conveyor_collider_center_x_m,
        conveyor_collider_center_y_m,
    ],
    "conveyor_collider_half_extents_m": [
        conveyor_collider_half_width_m,
        conveyor_collider_half_length_m,
        0.38,
    ],
    "conveyor_collider_inset_m": conveyor_collider_inset_m,
    "conveyor_additional_approach_m": conveyor_additional_approach_m,
    "rack_collider_count": (
        shelf_collision_plane_count + 1 + rack_upright_collider_count
    ),
    "shelf_collision_plane_count": shelf_collision_plane_count,
    "rack_upright_collider_count": rack_upright_collider_count,
    "dynamic_shelf_carton_count": shelf_carton_count,
    "worker_push_response": "upright_damped_stagger",
    "worker_patrol_controller": "collision_free_corridor_astar_dynamic_replan",
    "worker_routes": [route[0] for route in worker_routes],
    "worker_obstacles": [
        "visible_conveyor",
        "west_rack",
        "packing_table",
        "north_and_west_walls",
        "floor_edges",
        "dynamic_forklift_pallet_and_carton",
    ],
    "worker_walk_animations": list(worker_animation_names),
    "worker_walk_source_animations": list(worker_walk_source_names),
    "worker_walk_loop_source_time_codes": [
        walking_loop_source_start_time_code,
        walking_loop_source_end_time_code,
    ],
    "worker_idle_animations": list(worker_idle_animation_names),
    "rack_assets": [{RACK_SHELF_ASSET!r}, {RACK_FRAME_ASSET!r}],
    "rack_sides": ["West"],
    "parcel_carton_asset": parcel_carton_asset,
    "conveyor_layout": "oval",
    "conveyor_module_assets": [
        conveyor_straight_asset,
        conveyor_extension_asset,
        conveyor_curve_asset,
    ],
    "conveyor_module_count": conveyor_module_count,
    "belt_center_m": [belt_center_x_m, belt_center_y_m],
    "conveyor_oval_radius_m": conveyor_oval_radius_m,
    "conveyor_oval_straight_half_length_m":
        conveyor_oval_straight_half_length_m,
    "conveyor_surface_height_m": conveyor_surface_height_m,
    "conveyor_curve_segment_count": conveyor_curve_segment_count,
    "conveyor_surface_collider": "closed_low_poly_oval_mesh",
    "conveyor_physics_mode":
        "rigid_parcels_with_oval_surface_velocity_field",
    "parcel_dimensions_randomized": True,
    "parcel_rigid_bodies": parcel_count,
    "marked_zone_center_y_m": marked_zone_center_y_m,
    "marked_zone_half_length_m": marked_zone_half_length_m,
    "marked_zone_south_edge_m": marked_zone_south_edge_m,
    "marked_zone_north_edge_m": marked_zone_north_edge_m,
    "warehouse_floor_asset": {SIMPLE_WAREHOUSE_FLOOR_ASSET!r},
    "warehouse_floor_tile_count": len(floor_finish_tile_positions) + 1,
    "behind_wall_floor_supported": True,
    "packing_table_asset": {PACKING_TABLE_ASSET!r},
    "packing_table_collider": "single_static_box",
    "warehouse_floor_finish": "SM_floor02 with MI_Floor_02b wear maps",
    "clearance_zone_material": "matte UsdPreviewSurface roughness 0.92",
    "worker_count": len(worker_routes),
    "worker_assets": list(worker_assets[:len(worker_routes)]),
    "worker_motion_library": motion_library_asset,
    "worker_animation_bindings": worker_animation_bindings,
    "driver_count": len(driver_animation_bindings),
    "driver_asset": driver_asset,
    "driver_pose_source": driver_pose_source,
    "driver_retargeted_pose": retargeted_driver_pose_path,
    "driver_pose_sample": driver_pose_sample,
    "driver_whole_body_pitch_degrees": driver_whole_body_pitch_degrees,
    "driver_pose_retargeter": retarget_extension_name,
    "driver_animation_bindings": driver_animation_bindings,
    "parcel_count": parcel_count,
    "detector_cameras": {list(CAMERA_PATHS)[:-1]!r},
    "presentation_camera": {CAMERA_PATHS["presentation"]!r},
    "presentation_camera_distances_m": [5.5, 10.0, 15.0],
    "presentation_camera_default_view": 2,
    "presentation_camera_toggle_input": "Xbox View / GamepadInput.MENU1",
    "timeline_playing": timeline.is_playing(),
    "stack_light_visible_to_detector": True,
    "stack_light_active_intensity": 60000.0,
    "stack_light_wash_intensity": 50000.0,
    "stack_light_bounce_intensity": stack_scene_bounce_intensity,
    "stack_light_bounce_radius": stack_scene_bounce_radius,
    "stack_light_bounce_count": len({STACK_LIGHT_BOUNCE_PATHS!r}),
    "stack_light_lens_glow_radius_local": stack_lens_glow_radius,
    "stack_light_scene_wash_radius_local": stack_scene_wash_radius,
    "stack_light_count": len(stack_light_roots),
    "stack_light_bases_removed": True,
    "stack_light_color_bias_monitoring": True,
}}))
""".strip()


def capture_conveyor_safety_camera_source(arguments: dict[str, Any]) -> str:
    camera_name = arguments.get("camera", "detector_oblique")
    if camera_name not in CAMERA_PATHS:
        raise ValueError(
            f"camera must be one of {', '.join(sorted(CAMERA_PATHS))}"
        )
    filename = _validate_png_basename(
        arguments.get("filename"),
        default=f"{camera_name}.png",
    )
    CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = str((CAPTURE_ROOT / filename).resolve())
    camera_path = CAMERA_PATHS[camera_name]
    hide_active_halos = camera_name != "presentation"
    render_steps = arguments.get("render_steps", 3)
    rt_subframes = arguments.get("rt_subframes", 8)
    capture_width = arguments.get("width", 1024)
    capture_height = arguments.get("height", 576)
    if (
        not isinstance(render_steps, int)
        or isinstance(render_steps, bool)
        or not 1 <= render_steps <= 4
    ):
        raise ValueError("render_steps must be an integer from 1 through 4")
    if (
        not isinstance(rt_subframes, int)
        or isinstance(rt_subframes, bool)
        or not 1 <= rt_subframes <= 16
    ):
        raise ValueError("rt_subframes must be an integer from 1 through 16")
    for name, value in (
        ("width", capture_width),
        ("height", capture_height),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 128 <= value <= 2048
        ):
            raise ValueError(f"{name} must be an integer from 128 through 2048")
    return f"""
import json
import os
import sys
import time
import types
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core.functional import write_image
from pxr import Gf, Semantics, Usd, UsdGeom

source_started = time.monotonic()
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
camera = stage.GetPrimAtPath({camera_path!r})
if not camera.IsValid():
    raise RuntimeError("Create the conveyor safety scene before capture")
ptz_tracked_forklift = None
if {camera_name!r} == "detector_ptz":
    belt_center_x = 4.42
    belt_center_y = 1.268030
    belt_half_width = 2.05
    belt_half_length = 4.401374
    candidates = []
    for index in range(1, 4):
        forklift = stage.GetPrimAtPath(
            f"{{{FORKLIFT_ROOT!r}}}/Forklift{{index}}"
        )
        if not forklift.IsValid():
            continue
        body = stage.GetPrimAtPath(f"{{forklift.GetPath()}}/body")
        if not body.IsValid():
            continue
        value = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        x = float(value[0])
        y = float(value[1])
        closest_x = max(
            belt_center_x - belt_half_width,
            min(belt_center_x + belt_half_width, x),
        )
        closest_y = max(
            belt_center_y - belt_half_length,
            min(belt_center_y + belt_half_length, y),
        )
        squared_distance = (x - closest_x) ** 2 + (y - closest_y) ** 2
        candidates.append(
            (squared_distance, index, x, y, closest_x, closest_y)
        )
    if not candidates:
        raise RuntimeError("No forklifts are available for PTZ tracking")
    _, ptz_tracked_forklift, x, y, closest_x, closest_y = min(candidates)
    direction_x = x - closest_x
    direction_y = y - closest_y
    direction_length = max(
        1e-6,
        (direction_x * direction_x + direction_y * direction_y) ** 0.5,
    )
    up = Gf.Vec3d(
        direction_x / direction_length,
        direction_y / direction_length,
        0.0,
    )
    target = Gf.Vec3d(
        (x + closest_x) * 0.5,
        (y + closest_y) * 0.5,
        0.25,
    )
    eye = Gf.Vec3d(target[0], target[1], 9.0)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetLookAt(eye, target, up)
    xformable = UsdGeom.Xformable(camera)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix.GetInverse())
root = stage.GetPrimAtPath({SCENE_ROOT!r})
active_index = 1
if root.IsValid():
    active_value = root.GetAttribute("codex:activeForklift").Get()
    if active_value is not None:
        active_index = int(active_value) + 1
target_index = ptz_tracked_forklift or active_index
target_label = f"forklift_proximity_carton_{{target_index}}"
target_carton = stage.GetPrimAtPath(
    f"{{{CARGO_ROOT!r}}}/Forklift{{target_index}}Carton"
)
if target_carton.IsValid():
    # Refresh the label on the composed asset meshes before every synthetic
    # data capture. Replicator can otherwise retain the referenced asset's
    # pre-existing class after repeated render-product detach/attach cycles.
    for semantic_prim in Usd.PrimRange(target_carton):
        semantic = Semantics.SemanticsAPI.Apply(
            semantic_prim,
            "CodexForkCarton",
        )
        semantic.CreateSemanticTypeAttr().Set("class")
        semantic.CreateSemanticDataAttr().Set(target_label)
forklift_labels = []
for index in range(1, 4):
    forklift_label = f"safety_forklift_{{index}}"
    forklift_labels.append(forklift_label)
    forklift_asset = stage.GetPrimAtPath(
        f"{{{FORKLIFT_ROOT!r}}}/Forklift{{index}}"
    )
    if not forklift_asset.IsValid():
        continue
    # Label only visible geometry, not articulation/Xform containers.
    # Replicator can cache a container-level 2D box at the last teleport pose,
    # whereas mesh-level boxes continue to follow the moving links. The loose
    # carton lives under DynamicCargo and remains deliberately excluded.
    for semantic_prim in Usd.PrimRange(forklift_asset):
        semantic_path = str(semantic_prim.GetPath())
        if (
            semantic_prim.GetName() == "ActiveHalo"
            or "/DriverMount/" in semantic_path
        ):
            semantic_prim.RemoveAPI(
                Semantics.SemanticsAPI,
                "CodexSafetyForklift",
            )
            continue
        if not semantic_prim.IsA(UsdGeom.Gprim):
            semantic_prim.RemoveAPI(
                Semantics.SemanticsAPI,
                "CodexSafetyForklift",
            )
            continue
        semantic = Semantics.SemanticsAPI.Apply(
            semantic_prim,
            "CodexSafetyForklift",
        )
        semantic.CreateSemanticTypeAttr().Set("class")
        semantic.CreateSemanticDataAttr().Set(forklift_label)
semantic_labeling_finished = time.monotonic()
saved_halo_visibility = []
if {hide_active_halos!r}:
    for index in range(1, 4):
        halo = stage.GetPrimAtPath(
            f"{{{FORKLIFT_ROOT!r}}}/Forklift{{index}}/body/ActiveHalo"
        )
        if halo.IsValid():
            halo_visibility = UsdGeom.Imageable(halo).GetVisibilityAttr()
            saved_halo_visibility.append((halo, halo_visibility.Get()))
            halo_visibility.Set(UsdGeom.Tokens.invisible)

output_path = {output_path!r}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cache_module_name = "_qai_conveyor_capture_cache"
capture_cache = sys.modules.get(cache_module_name)
if capture_cache is None:
    capture_cache = types.ModuleType(cache_module_name)
    capture_cache.resource = None
    sys.modules[cache_module_name] = capture_cache
stage_identifier = stage.GetRootLayer().identifier
resource_key = (
    stage_identifier,
    {camera_path!r},
    int({capture_width}),
    int({capture_height}),
)
resource = getattr(capture_cache, "resource", None)
render_resource_reused = bool(
    resource is not None and resource.get("key") == resource_key
)
if resource is not None and not render_resource_reused:
    try:
        resource["rgb"].detach()
        resource["bbox"].detach()
        resource["render_product"].destroy()
    except Exception:
        pass
    resource = None
if resource is None:
    render_product = rep.create.render_product(
        {camera_path!r},
        ({capture_width}, {capture_height}),
        name="Reason2ConveyorCapture",
        force_new=True,
    )
    annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
    annotator.attach(render_product)
    bbox_annotator = rep.AnnotatorRegistry.get_annotator(
        "bounding_box_2d_tight",
        init_params={{"semanticTypes": ["class"]}},
    )
    bbox_annotator.attach(render_product)
    resource = {{
        "key": resource_key,
        "render_product": render_product,
        "rgb": annotator,
        "bbox": bbox_annotator,
    }}
    capture_cache.resource = resource
else:
    render_product = resource["render_product"]
    annotator = resource["rgb"]
    bbox_annotator = resource["bbox"]
render_setup_finished = time.monotonic()
rgb_data = None
bbox_data = None
hydra_texture = getattr(render_product, "hydra_texture", None)
render_product_updates_controlled = bool(
    hydra_texture is not None
    and hasattr(hydra_texture, "set_updates_enabled")
)
capture_state_module_name = "_qai_conveyor_capture_state"
capture_state = sys.modules.get(capture_state_module_name)
if capture_state is None:
    capture_state = types.ModuleType(capture_state_module_name)
    capture_state.depth = 0
    sys.modules[capture_state_module_name] = capture_state
try:
    capture_state.depth = int(getattr(capture_state, "depth", 0)) + 1
    if render_product_updates_controlled:
        hydra_texture.set_updates_enabled(True)
    for _ in range({render_steps}):
        await rep.orchestrator.step_async(
            rt_subframes={rt_subframes},
            pause_timeline=False,
            delta_time=0.0,
        )
        rgb_data = annotator.get_data()
        bbox_data = bbox_annotator.get_data()
finally:
    # Keep the render product and annotators attached for reuse, but stop its
    # Hydra texture from consuming RTX work during simulation frames between
    # model captures.
    if render_product_updates_controlled:
        hydra_texture.set_updates_enabled(False)
    capture_state.depth = max(
        0,
        int(getattr(capture_state, "depth", 1)) - 1,
    )
render_finished = time.monotonic()
if rgb_data is None or getattr(rgb_data, "size", 0) == 0:
    try:
        annotator.detach()
        bbox_annotator.detach()
        render_product.destroy()
    finally:
        capture_cache.resource = None
    raise RuntimeError("Off-screen detector render returned no RGB data")
fork_carton_pixel_boxes = []
forklift_boxes_by_label = {{}}
if isinstance(bbox_data, dict):
    label_map = bbox_data.get("info", {{}}).get("idToLabels", {{}})
    for item in bbox_data.get("data", []):
        semantic_id = str(int(item["semanticId"]))
        labels = label_map.get(semantic_id, {{}})
        class_values = str(labels.get("class", "")).split(",")
        if target_label in class_values:
            candidate = {{
                "x_min": int(item["x_min"]),
                "y_min": int(item["y_min"]),
                "x_max": int(item["x_max"]),
                "y_max": int(item["y_max"]),
                "semantic_label": target_label,
            }}
            if candidate not in fork_carton_pixel_boxes:
                fork_carton_pixel_boxes.append(candidate)
        for forklift_index, forklift_label in enumerate(forklift_labels, start=1):
            if forklift_label not in class_values:
                continue
            component_box = {{
                "x_min": int(item["x_min"]),
                "y_min": int(item["y_min"]),
                "x_max": int(item["x_max"]),
                "y_max": int(item["y_max"]),
            }}
            candidate = {{
                **component_box,
                "semantic_label": forklift_label,
                "forklift_index": forklift_index,
                "component_boxes": [component_box],
            }}
            existing = forklift_boxes_by_label.get(forklift_label)
            if existing is None:
                forklift_boxes_by_label[forklift_label] = candidate
            else:
                existing["x_min"] = min(existing["x_min"], candidate["x_min"])
                existing["y_min"] = min(existing["y_min"], candidate["y_min"])
                existing["x_max"] = max(existing["x_max"], candidate["x_max"])
                existing["y_max"] = max(existing["y_max"], candidate["y_max"])
                if component_box not in existing["component_boxes"]:
                    existing["component_boxes"].append(component_box)
forklift_pixel_boxes = [
    forklift_boxes_by_label[label]
    for label in forklift_labels
    if label in forklift_boxes_by_label
]
bbox_parse_finished = time.monotonic()
write_image(path=output_path, data=rgb_data)
for _ in range(60):
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        break
    await omni.kit.app.get_app().next_update_async()
write_finished = time.monotonic()
for halo, visibility in saved_halo_visibility:
    UsdGeom.Imageable(halo).GetVisibilityAttr().Set(
        visibility or UsdGeom.Tokens.inherited
    )
enabled_forklift_indices = [
    index
    for index in range(1, 4)
    if stage.GetPrimAtPath(
        f"{{{FORKLIFT_ROOT!r}}}/Forklift{{index}}"
    ).IsValid()
]
capture_ground_truth_signal = str(
    root.GetAttribute("codex:groundTruthSignal").Get() or "GREEN"
)
capture_min_forklift_clearance_m = float(
    root.GetAttribute("codex:minForkliftClearanceM").Get() or 99.0
)
capture_active_forklift = int(
    root.GetAttribute("codex:activeForklift").Get() or 0
) + 1
source_finished = time.monotonic()
print(json.dumps({{
    "camera": {camera_name!r},
    "camera_path": {camera_path!r},
    "output_path": output_path,
    "resolution": [{capture_width}, {capture_height}],
    "ptz_tracked_forklift": ptz_tracked_forklift,
    "perception_forklift": target_index,
    "fork_carton_pixel_boxes": fork_carton_pixel_boxes,
    "forklift_pixel_boxes": forklift_pixel_boxes,
    "enabled_forklift_indices": enabled_forklift_indices,
    "capture_ground_truth_signal": capture_ground_truth_signal,
    "capture_min_forklift_clearance_m": capture_min_forklift_clearance_m,
    "capture_active_forklift": capture_active_forklift,
    "workers_included_in_detection": False,
    "exists": os.path.isfile(output_path),
    "bytes": os.path.getsize(output_path) if os.path.isfile(output_path) else 0,
    "stack_light_visible_to_model": True,
    "active_forklift_halos_visible_to_model": {not hide_active_halos!r},
    "render_resource_reused": render_resource_reused,
    "render_product_updates_disabled_between_captures":
        render_product_updates_controlled,
    "render_resource_key": list(resource_key),
    "capture_render_steps": {render_steps},
    "capture_rt_subframes": {rt_subframes},
    "dynamic_updates_suppressed_during_capture": True,
    "capture_stage_seconds": {{
        "semantic_labeling": round(
            semantic_labeling_finished - source_started, 6
        ),
        "render_setup": round(
            render_setup_finished - semantic_labeling_finished, 6
        ),
        "render": round(render_finished - render_setup_finished, 6),
        "bbox_parse": round(bbox_parse_finished - render_finished, 6),
        "write": round(write_finished - bbox_parse_finished, 6),
        "teardown": round(source_finished - write_finished, 6),
        "total": round(source_finished - source_started, 6),
    }},
}}))
""".strip()


def set_conveyor_safety_light_source(arguments: dict[str, Any]) -> str:
    signal = arguments.get("signal")
    if signal not in ("GREEN", "AMBER", "RED"):
        raise ValueError("signal must be GREEN, AMBER, or RED")
    confidence = arguments.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    inference_id = arguments.get("inference_id", "")
    if not isinstance(inference_id, str) or len(inference_id) > 128:
        raise ValueError("inference_id must be a string of at most 128 characters")
    return f"""
import json
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath({SCENE_ROOT!r})
if not root.IsValid():
    raise RuntimeError("Create the conveyor safety scene first")
previous_signal = str(root.GetAttribute("codex:modelSignal").Get() or "")
stack_light_roots = {STACK_LIGHT_ROOTS!r}
stack_light_point_radius = 0.01
stack_light_bounce_paths = {STACK_LIGHT_BOUNCE_PATHS!r}
stack_light_bounce_radius = 0.30
stack_light_bounce_intensity = 220000.0
wash_paths = [f"{{path}}/SceneWash" for path in stack_light_roots]
required_light_paths = list(wash_paths) + list(stack_light_bounce_paths)
for stack_root_path in stack_light_roots:
    for lens_name in ("Green", "Amber", "Red"):
        required_light_paths.extend((
            f"{{stack_root_path}}/{{lens_name}}Lens",
            f"{{stack_root_path}}/{{lens_name}}Lens/Glow",
        ))

def light_geometry_is_current():
    for stack_root_path in stack_light_roots:
        light_paths = [f"{{stack_root_path}}/SceneWash"]
        light_paths.extend(
            f"{{stack_root_path}}/{{name}}Lens/Glow"
            for name in ("Green", "Amber", "Red")
        )
        for path in light_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return False
            sphere_light = UsdLux.SphereLight(prim)
            radius = sphere_light.GetRadiusAttr().Get()
            treat_as_point = sphere_light.GetTreatAsPointAttr()
            if (
                radius is None
                or abs(float(radius) - stack_light_point_radius) > 1.0e-6
                or not treat_as_point.IsValid()
                or not bool(treat_as_point.Get())
            ):
                return False
    return True

def bounce_geometry_is_current():
    for path in stack_light_bounce_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return False
        light = UsdLux.DiskLight(prim)
        radius = light.GetRadiusAttr().Get()
        intensity = light.GetIntensityAttr().Get()
        if (
            radius is None
            or abs(float(radius) - stack_light_bounce_radius) > 1.0e-6
            or intensity is None
            or abs(float(intensity) - stack_light_bounce_intensity) > 1.0e-3
        ):
            return False
    return True

light_mutated = (
    previous_signal != {signal!r}
    or not all(
        stage.GetPrimAtPath(path).IsValid()
        for path in required_light_paths
    )
    or not light_geometry_is_current()
    or not bounce_geometry_is_current()
)
last_light_mutated_attr = root.GetAttribute("codex:lastLightMutated")
if not last_light_mutated_attr.IsValid():
    last_light_mutated_attr = root.CreateAttribute(
        "codex:lastLightMutated",
        Sdf.ValueTypeNames.Bool,
    )
light_mutation_count_attr = root.GetAttribute("codex:lightMutationCount")
if not light_mutation_count_attr.IsValid():
    light_mutation_count_attr = root.CreateAttribute(
        "codex:lightMutationCount",
        Sdf.ValueTypeNames.Int,
    )
    light_mutation_count_attr.Set(0)
root.GetAttribute("codex:modelSignal").Set({signal!r})
root.GetAttribute("codex:modelConfidence").Set(float({float(confidence)!r}))
root.GetAttribute("codex:lastInferenceId").Set({inference_id!r})
last_light_mutated_attr.Set(light_mutated)
if light_mutated:
    light_mutation_count_attr.Set(
        int(light_mutation_count_attr.Get() or 0) + 1
    )
    colors = {{
        "GREEN": Gf.Vec3f(0.04, 1.0, 0.14),
        "AMBER": Gf.Vec3f(1.0, 0.54, 0.02),
        "RED": Gf.Vec3f(1.0, 0.03, 0.02),
    }}
    for stack_root_path in stack_light_roots:
        for name in ("Green", "Amber", "Red"):
            upper = name.upper()
            active = upper == {signal!r}
            lens = stage.GetPrimAtPath(
                f"{{stack_root_path}}/{{name}}Lens"
            )
            if lens.IsValid():
                lens.GetAttribute("primvars:displayColor").Set([
                    colors[upper] if active else colors[upper] * 0.10
                ])
            glow = stage.GetPrimAtPath(
                f"{{stack_root_path}}/{{name}}Lens/Glow"
            )
            if glow.IsValid():
                glow_light = UsdLux.SphereLight(glow)
                glow_light.GetRadiusAttr().Set(stack_light_point_radius)
                glow_light.CreateTreatAsPointAttr(True)
                glow_light.GetIntensityAttr().Set(
                    60000.0 if active else 0.0
                )
        wash_prim = stage.GetPrimAtPath(f"{{stack_root_path}}/SceneWash")
        if wash_prim.IsValid():
            wash = UsdLux.SphereLight(wash_prim)
            wash.GetRadiusAttr().Set(stack_light_point_radius)
            wash.CreateTreatAsPointAttr(True)
            wash.GetColorAttr().Set(colors[{signal!r}])
            wash.GetIntensityAttr().Set(50000.0)
    for bounce_path in stack_light_bounce_paths:
        bounce_prim = stage.GetPrimAtPath(bounce_path)
        if bounce_prim.IsValid():
            bounce = UsdLux.DiskLight(bounce_prim)
            bounce.GetRadiusAttr().Set(stack_light_bounce_radius)
            bounce.GetColorAttr().Set(colors[{signal!r}])
            bounce.GetIntensityAttr().Set(stack_light_bounce_intensity)
print(json.dumps({{
    "signal": {signal!r},
    "confidence": float({float(confidence)!r}),
    "inference_id": {inference_id!r},
    "model_driven": True,
    "light_mutated": light_mutated,
    "light_mutation_count": int(
        light_mutation_count_attr.Get() or 0
    ),
    "active_lens_intensity": 60000.0,
    "scene_wash_intensity": 50000.0,
    "scene_bounce_intensity": stack_light_bounce_intensity,
    "scene_bounce_count": len(stack_light_bounce_paths),
    "sphere_lights_treated_as_points": True,
    "sphere_light_radius": stack_light_point_radius,
    "stack_light_count": len(stack_light_roots),
}}))
""".strip()


def set_conveyor_safety_light_and_get_state_source(
    arguments: dict[str, Any],
) -> str:
    """Apply the model signal and return post-application state in one call."""

    return (
        set_conveyor_safety_light_source(arguments)
        + "\n"
        + get_conveyor_safety_state_source()
    )


def set_conveyor_safety_forklift_pose_source(arguments: dict[str, Any]) -> str:
    index = arguments.get("index")
    if not isinstance(index, int) or not 1 <= index <= 3:
        raise ValueError("index must be 1, 2, or 3")
    values: dict[str, float] = {}
    for name in ("x", "y", "yaw_degrees"):
        value = arguments.get(name)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        values[name] = float(value)
    if not -9.0 <= values["x"] <= 9.0 or not -6.7 <= values["y"] <= 6.7:
        raise ValueError("forklift pose must remain inside the factory floor")
    return f"""
import json
import math
import sys
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation
from pxr import Gf, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
path = {f"{FORKLIFT_ROOT}/Forklift{index}"!r}
forklift = stage.GetPrimAtPath(path)
if not forklift.IsValid():
    raise RuntimeError("Create the conveyor safety scene first")
timeline = omni.timeline.get_timeline_interface()
was_playing = timeline.is_playing()
timeline.stop()
for _ in range(3):
    await omni.kit.app.get_app().next_update_async()
yaw_radians = math.radians({values["yaw_degrees"]!r})
robot = Articulation(path)
robot.set_world_poses(
    positions=[[{values["x"]!r}, {values["y"]!r}, 0.0]],
    orientations=[[
        math.cos(yaw_radians * 0.5),
        0.0,
        0.0,
        math.sin(yaw_radians * 0.5),
    ]],
)
for _ in range(3):
    await omni.kit.app.get_app().next_update_async()
# Articulation.set_world_poses may rebuild the wrapper xform order. Retain the
# official asset's centimetre-to-metre scale after deterministic pose changes.
forklift_xform = UsdGeom.Xformable(forklift)
ordered_ops = list(forklift_xform.GetOrderedXformOps())
if not any(str(op.GetOpName()) == "xformOp:scale" for op in ordered_ops):
    try:
        scale_op = forklift_xform.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble
        )
    except Exception:
        scale_op = UsdGeom.XformOp(
            forklift.GetAttribute("xformOp:scale")
        )
        forklift_xform.SetXformOpOrder([*ordered_ops, scale_op])
    scale_op.Set(Gf.Vec3d(0.01, 0.01, 0.01))
lift_joint = stage.GetPrimAtPath(f"{{path}}/lift_joint")
if lift_joint.IsValid():
    lift_drive = UsdPhysics.DriveAPI.Get(lift_joint, "linear")
    lift_drive.GetTargetPositionAttr().Set(0.0)
forklift.GetAttribute("codex:forkHeightM").Set(0.0)
root = stage.GetPrimAtPath({SCENE_ROOT!r})
if (
    root.IsValid()
    and int(root.GetAttribute("codex:activeForklift").Get()) == {index - 1}
):
    root.GetAttribute("codex:forkHeightM").Set(0.0)
for cargo_kind, cargo_height, cargo_yaw_offset in (
    (
        "Pallet",
        {FORK_PALLET_BASE_Z_M!r},
        {FORK_PALLET_YAW_OFFSET_DEGREES!r},
    ),
    ("Carton", {FORK_CARTON_BASE_Z_M!r}, 0.0),
):
    cargo = stage.GetPrimAtPath(
        f"{{{CARGO_ROOT!r}}}/Forklift{{{index!r}}}{{cargo_kind}}"
    )
    if cargo.IsValid():
        cargo_api = UsdGeom.XformCommonAPI(cargo)
        cargo_api.SetTranslate(Gf.Vec3d(
            {values["x"]!r}
            + math.cos(yaw_radians) * {FORK_CARGO_FORWARD_OFFSET_M!r},
            {values["y"]!r}
            + math.sin(yaw_radians) * {FORK_CARGO_FORWARD_OFFSET_M!r},
            cargo_height,
        ))
        cargo_api.SetRotate(
            Gf.Vec3f(
                0.0,
                0.0,
                {values["yaw_degrees"]!r} + cargo_yaw_offset,
            ),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        body = UsdPhysics.RigidBodyAPI(cargo)
        body.GetVelocityAttr().Set(Gf.Vec3f(0.0))
        body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0))
# Deterministic teleports happen outside normal physics playback. Invalidate
# only that benchmark capture resource so its semantic annotator cannot return
# the pre-teleport box. Normal live driving never takes this path and continues
# to reuse the persistent render product and annotators.
capture_cache = sys.modules.get("_qai_conveyor_capture_cache")
resource = getattr(capture_cache, "resource", None)
if resource is not None:
    try:
        resource["rgb"].detach()
        resource["bbox"].detach()
        resource["render_product"].destroy()
    except Exception:
        pass
    capture_cache.resource = None
if was_playing:
    timeline.play()
print(json.dumps({{
    "forklift": path,
    "position": [{values["x"]!r}, {values["y"]!r}, 0.0],
    "yaw_degrees": {values["yaw_degrees"]!r},
}}))
""".strip()


def set_conveyor_safety_camera_source(arguments: dict[str, Any]) -> str:
    camera_name = arguments.get("camera", "presentation")
    if camera_name not in CAMERA_PATHS:
        raise ValueError(
            f"camera must be one of {', '.join(sorted(CAMERA_PATHS))}"
        )
    return f"""
import json
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath({SCENE_ROOT!r})
camera = stage.GetPrimAtPath({CAMERA_PATHS[camera_name]!r})
if not root.IsValid() or not camera.IsValid():
    raise RuntimeError("Create the conveyor safety scene first")
if {camera_name!r} != "presentation":
    root.GetAttribute("codex:selectedDetectorCamera").Set({camera_name!r})
viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active Isaac Sim viewport is available")
viewport.camera_path = Sdf.Path({CAMERA_PATHS[camera_name]!r})
print(json.dumps({{
    "camera": {camera_name!r},
    "camera_path": {CAMERA_PATHS[camera_name]!r},
}}))
""".strip()


def get_conveyor_safety_state_source() -> str:
    return f"""
import json
import math
import omni.appwindow
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath({SCENE_ROOT!r})
if not root.IsValid():
    raise RuntimeError("Create the conveyor safety scene first")
belt_half_width = float(root.GetAttribute("codex:beltHalfWidthM").Get())
belt_half_length = float(root.GetAttribute("codex:beltHalfLengthM").Get())
belt_center_x = float(root.GetAttribute("codex:beltCenterXM").Get())
belt_center_y = float(root.GetAttribute("codex:beltCenterYM").Get())
forklift_half_length = float(root.GetAttribute("codex:forkliftHalfLengthM").Get())
forklift_half_width = float(root.GetAttribute("codex:forkliftHalfWidthM").Get())

def clearance_for(prim):
    body = stage.GetPrimAtPath(f"{{prim.GetPath()}}/body")
    if not body.IsValid():
        raise RuntimeError(f"Articulated forklift body missing: {{prim.GetPath()}}")
    matrix = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translate = matrix.ExtractTranslation()
    forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    x = float(translate[0])
    y = float(translate[1])
    yaw = math.atan2(float(forward[1]), float(forward[0]))
    closest_x = max(
        belt_center_x - belt_half_width,
        min(belt_center_x + belt_half_width, x),
    )
    closest_y = max(
        belt_center_y - belt_half_length,
        min(belt_center_y + belt_half_length, y),
    )
    vx = x - closest_x
    vy = y - closest_y
    center_distance = math.hypot(vx, vy)
    if center_distance <= 1e-6:
        clearance = 0.0
    else:
        nx = vx / center_distance
        ny = vy / center_distance
        # ForkliftC's authored forward axis is local +X.
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -math.sin(yaw)
        right_y = math.cos(yaw)
        support = (
            forklift_half_length * abs(nx * forward_x + ny * forward_y)
            + forklift_half_width * abs(nx * right_x + ny * right_y)
        )
        clearance = max(0.0, center_distance - support)
    return {{
        "path": str(prim.GetPath()),
        "position": [x, y, float(translate[2])],
        "yaw_degrees": math.degrees(yaw),
        "clearance_m": round(clearance, 4),
    }}

forklifts = []
for index in range(1, 4):
    prim = stage.GetPrimAtPath(f"{{{FORKLIFT_ROOT!r}}}/Forklift{{index}}")
    if prim.IsValid():
        item = clearance_for(prim)
        item["index"] = index
        item["fork_height_m"] = float(
            prim.GetAttribute("codex:forkHeightM").Get() or 0.0
        )
        forklifts.append(item)
cargo = []
for index in range(1, 4):
    for cargo_kind in ("Pallet", "Carton"):
        prim = stage.GetPrimAtPath(
            f"{{{CARGO_ROOT!r}}}/Forklift{{index}}{{cargo_kind}}"
        )
        if not prim.IsValid():
            continue
        translate = prim.GetAttribute("xformOp:translate").Get()
        velocity_attr = prim.GetAttribute("physics:velocity")
        velocity = velocity_attr.Get() if velocity_attr else None
        cargo.append({{
            "index": index,
            "kind": cargo_kind.lower(),
            "path": str(prim.GetPath()),
            "position": [float(value) for value in translate],
            "velocity_mps": (
                [float(value) for value in velocity]
                if velocity is not None
                else [0.0, 0.0, 0.0]
            ),
            "dynamic": True,
        }})
min_clearance = min(item["clearance_m"] for item in forklifts)
actual_speed_mps = float(root.GetAttribute("codex:actualSpeedMps").Get())
if min_clearance < 1.0:
    ground_truth = "RED"
else:
    live_signal = str(root.GetAttribute("codex:groundTruthSignal").Get() or "")
    ground_truth = (
        "AMBER"
        if (
            omni.timeline.get_timeline_interface().is_playing()
            and live_signal == "AMBER"
        )
        else "GREEN"
    )
root.GetAttribute("codex:minForkliftClearanceM").Set(float(min_clearance))
root.GetAttribute("codex:groundTruthSignal").Set(ground_truth)
gamepad = omni.appwindow.get_default_app_window().get_gamepad(0)
last_light_mutated_attr = root.GetAttribute("codex:lastLightMutated")
light_mutation_count_attr = root.GetAttribute("codex:lightMutationCount")
print(json.dumps({{
    "scene_root": {SCENE_ROOT!r},
    "timeline_playing": omni.timeline.get_timeline_interface().is_playing(),
    "active_forklift": int(root.GetAttribute("codex:activeForklift").Get()) + 1,
    "controller_connected": gamepad is not None,
    "driving_model": root.GetAttribute("codex:drivingModel").Get(),
    "commanded_speed_mps": float(
        root.GetAttribute("codex:commandedSpeedMps").Get()
    ),
    "actual_speed_mps": actual_speed_mps,
    "steering_angle_degrees": float(
        root.GetAttribute("codex:steeringAngleDegrees").Get()
    ),
    "fork_height_m": float(
        root.GetAttribute("codex:forkHeightM").Get()
    ),
    "forklifts": forklifts,
    "dynamic_cargo": cargo,
    "collision_blocking_enabled": bool(
        root.GetAttribute("codex:collisionBlockingEnabled").Get()
    ),
    "worker_push_enabled": bool(
        root.GetAttribute("codex:workerPushEnabled").Get()
    ),
    "last_collision": root.GetAttribute("codex:lastCollision").Get(),
    "collision_block_count": int(
        root.GetAttribute("codex:collisionBlockCount").Get()
    ),
    "min_forklift_clearance_m": min_clearance,
    "ground_truth_signal": ground_truth,
    "model_signal": root.GetAttribute("codex:modelSignal").Get(),
    "model_confidence": root.GetAttribute("codex:modelConfidence").Get(),
    "last_inference_id": root.GetAttribute("codex:lastInferenceId").Get(),
    "last_light_mutated": bool(
        last_light_mutated_attr.Get()
        if last_light_mutated_attr.IsValid()
        else False
    ),
    "light_mutation_count": int(
        (
            light_mutation_count_attr.Get()
            if light_mutation_count_attr.IsValid()
            else 0
        )
        or 0
    ),
    "selected_detector_camera": root.GetAttribute("codex:selectedDetectorCamera").Get(),
    "worker_count": len([
        child
        for child in stage.GetPrimAtPath({WORKER_ROOT!r}).GetChildren()
        if child.GetName().startswith("Worker")
    ]),
    "parcel_count": len(list(stage.GetPrimAtPath({f"{CONVEYOR_ROOT}/Parcels"!r}).GetChildren())),
    "stack_light_visible_to_detector": True,
    "stack_light_color_bias_monitoring": True,
}}))
""".strip()
