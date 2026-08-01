from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCENE_ROOT = "/World/CodexPoC"
CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "isaac_sim_edge_supervisor"
)
REALISTIC_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "isaac_sim_edge_supervisor_realistic"
)
REALISTIC_V2_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "isaac_sim_edge_supervisor_realistic_v2"
)
REALISTIC_V3_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "isaac_sim_edge_supervisor_realistic_v3"
)
REALISTIC_V4_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "isaac_sim_edge_supervisor_realistic_v4"
)
CAMERA_PATHS = {
    "overview": f"{SCENE_ROOT}/Sensors/RoofOverview",
    "west_aisle": f"{SCENE_ROOT}/Sensors/WestAisle",
    "intersection": f"{SCENE_ROOT}/Sensors/Intersection",
    "east_aisle": f"{SCENE_ROOT}/Sensors/EastAisle",
}
SCENARIOS = ("mixed_traffic", "aisle_congestion", "blind_corner")
SCENARIO_LABELS = {
    "mixed_traffic": "Mixed-traffic intersection",
    "aisle_congestion": "Opposing-AMR aisle congestion",
    "blind_corner": "Blind-corner forklift emergence",
}
NVIDIA_ASSET_PATHS = {
    "warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "nova_carter": "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
    "forklift": "/Isaac/Robots/IsaacSim/ForkliftB/forklift_b.usd",
    "worker": (
        "/Isaac/People/Characters/original_male_adult_construction_03/"
        "male_adult_construction_03.usd"
    ),
    "pallet": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
    "cardbox": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
    "traffic_cone": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "rack_shelf": "/Isaac/Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd",
    "rack_frame": "/Isaac/Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd",
    "rack_pile": "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
    "rack_shield": "/Isaac/Environments/Simple_Warehouse/Props/SM_Rackshield_02.usd",
    "aisle_sign": "/Isaac/Environments/Simple_Warehouse/Props/S_AisleSign.usd",
}


def supervisor_decision(
    progress: float,
    scenario: str = "mixed_traffic",
) -> dict[str, Any]:
    """Return the deterministic notification used to stage the tutorial."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")

    if scenario == "mixed_traffic":
        if progress < 0.22:
            action, hazard, severity, route = (
                "CONTINUE",
                "none",
                "info",
                "direct",
            )
            reason = "No predicted conflict in the three-second horizon"
        elif progress < 0.52:
            action, hazard, severity, route = (
                "YIELD",
                "forklift_crossing",
                "warning",
                "hold_west_gate",
            )
            reason = "Forklift predicted to enter the shared intersection"
        elif progress < 0.68:
            action, hazard, severity, route = (
                "STOP",
                "worker_proximity",
                "critical",
                "hold_west_gate",
            )
            reason = "Worker detected inside the robot exclusion zone"
        elif progress < 0.88:
            action, hazard, severity, route = (
                "REROUTE",
                "robot_congestion",
                "warning",
                "south_bypass",
            )
            reason = "Second AMR is blocking the east aisle"
        else:
            action, hazard, severity, route = (
                "CONTINUE",
                "none",
                "info",
                "direct_to_goal",
            )
            reason = "Conflicts cleared and the alternate lane is open"
    elif scenario == "aisle_congestion":
        if progress < 0.20:
            action, hazard, severity, route = (
                "CONTINUE",
                "none",
                "info",
                "direct",
            )
            reason = "Direct aisle has the lowest travel cost"
        elif progress < 0.42:
            action, hazard, severity, route = (
                "YIELD",
                "opposing_robot",
                "warning",
                "hold_west_gate",
            )
            reason = "Opposing AMR has entered the single-width aisle"
        elif progress < 0.88:
            action, hazard, severity, route = (
                "REROUTE",
                "aisle_congestion",
                "warning",
                "south_bypass",
            )
            reason = "South loop is clear and avoids the blocked aisle"
        else:
            action, hazard, severity, route = (
                "CONTINUE",
                "none",
                "info",
                "direct_to_goal",
            )
            reason = "RobotBlue has cleared the congested shelf row"
    else:
        if progress < 0.25:
            action, hazard, severity, route = (
                "CONTINUE",
                "none",
                "info",
                "direct",
            )
            reason = "Blind corner is clear in the prediction horizon"
        elif progress < 0.50:
            action, hazard, severity, route = (
                "YIELD",
                "forklift_blind_corner",
                "warning",
                "hold_west_gate",
            )
            reason = "Roof camera sees a forklift hidden from RobotBlue"
        elif progress < 0.82:
            action, hazard, severity, route = (
                "REROUTE",
                "forklift_blind_corner",
                "warning",
                "north_bypass",
            )
            reason = "North loop clears the forklift's swept path"
        else:
            action, hazard, severity, route = (
                "CONTINUE",
                "none",
                "info",
                "direct_to_goal",
            )
            reason = "Forklift has cleared the cross-aisle"
    return {
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "recipient": "RobotBlue",
        "action": action,
        "reason": reason,
        "hazard": hazard,
        "severity": severity,
        "route": route,
        "horizon_seconds": 3.0,
    }


def _capture_root() -> Path:
    root = Path(
        os.environ.get("ISAAC_SIM_TUTORIAL_CAPTURE_ROOT", str(CAPTURE_ROOT))
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scenario_value(arguments: dict[str, Any]) -> str:
    scenario = arguments.get("scenario", "mixed_traffic")
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}")
    return scenario


def _realistic_capture_root(arguments: dict[str, Any]) -> Path:
    variant = arguments.get("variant", "default")
    if variant == "default":
        return REALISTIC_CAPTURE_ROOT
    if variant == "v2":
        return REALISTIC_V2_CAPTURE_ROOT
    if variant == "v3":
        return REALISTIC_V3_CAPTURE_ROOT
    if variant == "v4":
        return REALISTIC_V4_CAPTURE_ROOT
    raise ValueError("variant must be one of: default, v2, v3, v4")


def asset_status_source() -> str:
    candidates_json = json.dumps(NVIDIA_ASSET_PATHS)
    return f'''
import json
from isaacsim.storage.native import get_assets_root_path, get_full_asset_path

root = get_assets_root_path()
candidates = {candidates_json}
resolved = {{}}
for name, path in candidates.items():
    try:
        resolved[name] = get_full_asset_path(path)
    except Exception as error:
        resolved[name] = None
print(json.dumps({{
    "asset_root": root,
    "assets": resolved,
    "all_available": all(bool(value) for value in resolved.values()),
}}))
'''.strip()


def create_scene_source() -> str:
    return r'''
import json
import carb
import omni.usd
from isaacsim.storage.native import get_full_asset_path
from pxr import Gf, Sdf, UsdGeom, UsdLux

ROOT = "/World/CodexPoC"
render_settings = carb.settings.get_settings()
render_settings.set("/rtx/post/tonemap/exposure", 0.0)
render_settings.set("/rtx/post/tonemap/cameraShutter", 50.0)
render_settings.set("/rtx/post/tonemap/filmIso", 100.0)
render_settings.set("/rtx/post/tonemap/fNumber", 5.0)
render_settings.set("/rtx/post/histogram/enabled", True)
render_settings.set("/rtx/post/aa/autoExposureMode", 1)
render_settings.set("/rtx/post/aa/exposureMultiplier", 0.1)
render_settings.set("/rtx/post/aa/exposure", 1.0)
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")

# This namespace is intentionally disposable and owned by the tutorial.
if stage.GetPrimAtPath(ROOT).IsValid():
    stage.RemovePrim(ROOT)

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, ROOT)

asset_paths = {
    "warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "nova_carter": "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
    "forklift": "/Isaac/Robots/IsaacSim/ForkliftB/forklift_b.usd",
    "worker": (
        "/Isaac/People/Characters/original_male_adult_construction_03/"
        "male_adult_construction_03.usd"
    ),
    "pallet": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
    "cardbox": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
    "traffic_cone": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "rack_shelf": "/Isaac/Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd",
    "rack_frame": "/Isaac/Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd",
    "rack_pile": "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
    "rack_shield": "/Isaac/Environments/Simple_Warehouse/Props/SM_Rackshield_02.usd",
    "aisle_sign": "/Isaac/Environments/Simple_Warehouse/Props/S_AisleSign.usd",
}
resolved_assets = {}
for asset_name, asset_path in asset_paths.items():
    try:
        resolved_assets[asset_name] = get_full_asset_path(asset_path)
    except Exception:
        resolved_assets[asset_name] = None


def xform(path, translate=(0.0, 0.0, 0.0)):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*translate))
    return prim


def cube(path, translate, scale, color, opacity=1.0):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    shape.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    shape.CreateDisplayOpacityAttr().Set([opacity])
    api = UsdGeom.XformCommonAPI(shape.GetPrim())
    api.SetTranslate(Gf.Vec3d(*translate))
    api.SetScale(Gf.Vec3f(*scale))
    return shape.GetPrim()


def cylinder(path, translate, radius, height, color):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    shape.CreateRadiusAttr().Set(radius)
    shape.CreateHeightAttr().Set(height)
    shape.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(shape.GetPrim()).SetTranslate(Gf.Vec3d(*translate))
    return shape.GetPrim()


def sphere(path, translate, radius, color):
    shape = UsdGeom.Sphere.Define(stage, path)
    shape.CreateRadiusAttr().Set(radius)
    shape.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(shape.GetPrim()).SetTranslate(Gf.Vec3d(*translate))
    return shape.GetPrim()


def reference_asset(path, asset_url, translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), rotate=(0.0, 0.0, 0.0)):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(asset_url)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*translate))
    if tuple(scale) != (1.0, 1.0, 1.0):
        api.SetScale(Gf.Vec3f(*scale))
    if tuple(rotate) != (0.0, 0.0, 0.0):
        api.SetRotate(Gf.Vec3f(*rotate))
    return prim


def rack(path, center_x, center_y):
    xform(path)
    steel = (0.18, 0.23, 0.30)
    beam = (0.95, 0.55, 0.08)
    for index, x_offset in enumerate((-1.55, 1.55)):
        for side, y_offset in enumerate((-0.42, 0.42)):
            cube(
                f"{path}/Post_{index}_{side}",
                (center_x + x_offset, center_y + y_offset, 1.55),
                (0.09, 0.09, 1.55),
                steel,
            )
    for level, z_value in enumerate((0.55, 1.55, 2.55)):
        cube(
            f"{path}/Shelf_{level}",
            (center_x, center_y, z_value),
            (1.65, 0.52, 0.08),
            steel,
        )
        cube(
            f"{path}/Beam_{level}",
            (center_x, center_y - 0.48, z_value + 0.10),
            (1.65, 0.06, 0.12),
            beam,
        )
    box_colors = (
        (0.55, 0.30, 0.12),
        (0.16, 0.46, 0.62),
        (0.62, 0.46, 0.16),
    )
    for level, z_value in enumerate((0.83, 1.83, 2.83)):
        for slot, x_offset in enumerate((-0.95, 0.0, 0.95)):
            cube(
                f"{path}/Box_{level}_{slot}",
                (center_x + x_offset, center_y, z_value),
                (0.36, 0.34, 0.25),
                box_colors[(level + slot) % len(box_colors)],
            )


# Building shell, marked traffic lanes, and camera coverage boundaries.
cube(f"{ROOT}/Floor", (0.0, 0.0, -0.08), (7.0, 5.0, 0.08), (0.19, 0.22, 0.25))
for wall_y in (-4.92, 4.92):
    cube(
        f"{ROOT}/Walls/WallY_{str(wall_y).replace('-', 'm').replace('.', '_')}",
        (0.0, wall_y, 1.0),
        (7.0, 0.08, 1.0),
        (0.28, 0.31, 0.34),
    )
for wall_x in (-6.92, 6.92):
    cube(
        f"{ROOT}/Walls/WallX_{str(wall_x).replace('-', 'm').replace('.', '_')}",
        (wall_x, 0.0, 1.0),
        (0.08, 5.0, 1.0),
        (0.28, 0.31, 0.34),
    )

lane_yellow = (0.96, 0.72, 0.08)
for y_value in (-2.35, 0.35):
    cube(
        f"{ROOT}/Markings/Lane_{str(y_value).replace('-', 'm').replace('.', '_')}",
        (0.0, y_value, 0.015),
        (6.25, 0.035, 0.015),
        lane_yellow,
    )
for x_value in (-1.35, 1.35):
    cube(
        f"{ROOT}/Markings/Crossing_{str(x_value).replace('-', 'm').replace('.', '_')}",
        (x_value, -1.0, 0.018),
        (0.04, 1.35, 0.018),
        (0.92, 0.92, 0.92),
    )
for index in range(7):
    cube(
        f"{ROOT}/Markings/Zebra_{index}",
        (-1.08 + index * 0.36, -1.0, 0.022),
        (0.10, 1.25, 0.022),
        (0.92, 0.92, 0.92),
    )

# Camera coverage is represented by unobtrusive floor corner brackets.
coverage_colors = (
    (0.08, 0.68, 1.0),
    (0.70, 0.20, 0.95),
)
for zone_index, (center_x, color) in enumerate(((-3.3, coverage_colors[0]), (3.3, coverage_colors[1]))):
    for side_x_index, side_x in enumerate((-1.0, 1.0)):
        for side_y_index, side_y in enumerate((-1.0, 1.0)):
            base_x = center_x + side_x * 2.7
            base_y = side_y * 4.2
            cube(
                f"{ROOT}/Coverage/Zone{zone_index}_X_{side_x_index}_{side_y_index}",
                (base_x, base_y, 0.026),
                (0.42, 0.035, 0.026),
                color,
            )
            cube(
                f"{ROOT}/Coverage/Zone{zone_index}_Y_{side_x_index}_{side_y_index}",
                (base_x, base_y, 0.026),
                (0.035, 0.42, 0.026),
                color,
            )

for row, y_value in enumerate((-3.55, 3.55)):
    for column, x_value in enumerate((-4.7, 0.0, 4.7)):
        rack(f"{ROOT}/Warehouse/Rack_{row}_{column}", x_value, y_value)

# The complete full_warehouse.usd reference is deliberately not composed here:
# its first cloud load can take several minutes. The tutorial keeps a compact
# procedural shell and uses official NVIDIA actors and props, producing a fast,
# deterministic scene while retaining recognizable production assets.
if resolved_assets["pallet"]:
    for index, (x_value, y_value, rotation) in enumerate(
        ((-5.2, -2.85, 0.0), (3.8, 2.75, 90.0), (5.25, -2.85, 0.0))
    ):
        reference_asset(
            f"{ROOT}/NvidiaProps/Pallet_{index}",
            resolved_assets["pallet"],
            translate=(x_value, y_value, 0.0),
            rotate=(0.0, 0.0, rotation),
        )
if resolved_assets["cardbox"]:
    for index, (x_value, y_value, rotation) in enumerate(
        ((-5.2, -2.85, 0.0), (3.8, 2.75, 25.0), (5.25, -2.85, -12.0))
    ):
        reference_asset(
            f"{ROOT}/NvidiaProps/Cardbox_{index}",
            resolved_assets["cardbox"],
            translate=(x_value, y_value, 0.35),
            rotate=(0.0, 0.0, rotation),
        )
if resolved_assets["traffic_cone"]:
    for index, (x_value, y_value) in enumerate(
        ((-1.55, -2.55), (1.55, -2.55), (-1.55, 0.55), (1.55, 0.55))
    ):
        reference_asset(
            f"{ROOT}/NvidiaProps/TrafficCone_{index}",
            resolved_assets["traffic_cone"],
            translate=(x_value, y_value, 0.0),
        )

# RobotBlue: the supervised pallet-class autonomous mobile robot. The
# procedural fallback and official Nova Carter reference are intentionally
# enlarged from the stock mobile-base dimensions so the tutorial models a
# 1.4 m-class logistics AMR rather than a small inspection robot.
xform(f"{ROOT}/Actors/RobotBlue", (-6.4, 1.6, 0.0))
cube(f"{ROOT}/Actors/RobotBlue/Base", (0.0, 0.0, 0.40), (0.78, 0.58, 0.34), (0.05, 0.34, 0.92))
cube(f"{ROOT}/Actors/RobotBlue/Top", (0.0, 0.0, 0.82), (0.58, 0.45, 0.11), (0.15, 0.60, 1.0))
cube(f"{ROOT}/Actors/RobotBlue/Bumper", (0.70, 0.0, 0.32), (0.12, 0.52, 0.16), (0.04, 0.06, 0.08))
cylinder(f"{ROOT}/Actors/RobotBlue/Beacon", (0.0, 0.0, 1.08), 0.16, 0.12, (0.15, 0.95, 0.30))
# The 2.3 x 1.64 m frame visualizes the local planner's reserved footprint,
# including clearance, without covering the physical AMR. It is drawn in the
# supervisor overlay layer above the rack tops so a roof-view tutorial can
# still show the tracked footprint while the physical AMR is shelf-occluded.
xform(f"{ROOT}/Actors/RobotBlue/IdentityPad", (0.0, 0.0, 4.40))
for edge_name, edge_translate, edge_scale in (
    ("North", (0.0, 0.82, 0.0), (1.15, 0.045, 0.025)),
    ("South", (0.0, -0.82, 0.0), (1.15, 0.045, 0.025)),
    ("West", (-1.15, 0.0, 0.0), (0.045, 0.82, 0.025)),
    ("East", (1.15, 0.0, 0.0), (0.045, 0.82, 0.025)),
):
    cube(
        f"{ROOT}/Actors/RobotBlue/IdentityPad/{edge_name}",
        edge_translate,
        edge_scale,
        (0.05, 0.34, 0.92),
        0.82,
    )

# RobotGreen: a second AMR that eventually blocks the east aisle.
xform(f"{ROOT}/Actors/RobotGreen", (6.4, 1.6, 0.0))
cube(f"{ROOT}/Actors/RobotGreen/Base", (0.0, 0.0, 0.40), (0.78, 0.58, 0.34), (0.08, 0.72, 0.34))
cube(f"{ROOT}/Actors/RobotGreen/Top", (0.0, 0.0, 0.82), (0.58, 0.45, 0.11), (0.32, 0.94, 0.46))
cylinder(f"{ROOT}/Actors/RobotGreen/Beacon", (0.0, 0.0, 1.08), 0.16, 0.12, (0.15, 0.95, 0.30))
xform(f"{ROOT}/Actors/RobotGreen/IdentityPad", (0.0, 0.0, 4.40))
for edge_name, edge_translate, edge_scale in (
    ("North", (0.0, 0.82, 0.0), (1.15, 0.045, 0.025)),
    ("South", (0.0, -0.82, 0.0), (1.15, 0.045, 0.025)),
    ("West", (-1.15, 0.0, 0.0), (0.045, 0.82, 0.025)),
    ("East", (1.15, 0.0, 0.0), (0.045, 0.82, 0.025)),
):
    cube(
        f"{ROOT}/Actors/RobotGreen/IdentityPad/{edge_name}",
        edge_translate,
        edge_scale,
        (0.08, 0.72, 0.34),
        0.82,
    )

# ForkliftOrange: a generic moving forklift silhouette.
xform(f"{ROOT}/Actors/ForkliftOrange", (0.0, 4.8, 0.0))
cube(f"{ROOT}/Actors/ForkliftOrange/Body", (0.0, 0.0, 0.48), (0.55, 0.78, 0.42), (1.0, 0.38, 0.04))
cube(f"{ROOT}/Actors/ForkliftOrange/Cab", (0.0, 0.18, 1.05), (0.48, 0.42, 0.52), (0.12, 0.14, 0.16))
for side, x_value in enumerate((-0.32, 0.32)):
    cube(f"{ROOT}/Actors/ForkliftOrange/Fork_{side}", (x_value, -1.02, 0.14), (0.09, 0.72, 0.07), (0.16, 0.18, 0.20))
cylinder(f"{ROOT}/Actors/ForkliftOrange/Beacon", (0.0, 0.18, 1.66), 0.13, 0.12, (1.0, 0.82, 0.05))
cube(
    f"{ROOT}/Actors/ForkliftOrange/IdentityPad",
    (0.0, 0.0, 0.030),
    (1.55, 2.30, 0.018),
    (1.0, 0.38, 0.04),
    0.42,
)

# WorkerYellow: capsule body and head, easy to distinguish from vehicles.
xform(f"{ROOT}/Actors/WorkerYellow", (0.0, 4.6, 0.0))
cylinder(f"{ROOT}/Actors/WorkerYellow/Body", (0.0, 0.0, 0.78), 0.25, 1.05, (0.98, 0.74, 0.08))
sphere(f"{ROOT}/Actors/WorkerYellow/Head", (0.0, 0.0, 1.55), 0.23, (0.78, 0.48, 0.27))
cube(f"{ROOT}/Actors/WorkerYellow/Vest", (0.0, -0.22, 0.92), (0.22, 0.05, 0.30), (0.98, 0.98, 0.38))
cube(
    f"{ROOT}/Actors/WorkerYellow/IdentityPad",
    (0.0, 0.0, 0.030),
    (0.85, 0.85, 0.018),
    (0.98, 0.74, 0.08),
    0.50,
)

# Replace the fallback silhouettes with official Isaac Sim assets whenever
# they resolve. Wrappers remain stable so the same scenario controller moves
# either representation.
if resolved_assets["nova_carter"]:
    for actor_name in ("RobotBlue", "RobotGreen"):
        actor_path = f"{ROOT}/Actors/{actor_name}"
        for child_name in ("Base", "Top", "Bumper"):
            stage.RemovePrim(f"{actor_path}/{child_name}")
        reference_asset(
            f"{actor_path}/Asset",
            resolved_assets["nova_carter"],
            scale=(1.45, 1.45, 1.45),
            rotate=(0.0, 0.0, -90.0),
        )

if resolved_assets["forklift"]:
    forklift_path = f"{ROOT}/Actors/ForkliftOrange"
    for child_name in ("Body", "Cab", "Fork_0", "Fork_1"):
        stage.RemovePrim(f"{forklift_path}/{child_name}")
    reference_asset(
        f"{forklift_path}/Asset",
        resolved_assets["forklift"],
        rotate=(0.0, 0.0, 180.0),
    )

if resolved_assets["worker"]:
    worker_path = f"{ROOT}/Actors/WorkerYellow"
    for child_name in ("Body", "Head", "Vest"):
        stage.RemovePrim(f"{worker_path}/{child_name}")
    reference_asset(
        f"{worker_path}/Asset",
        resolved_assets["worker"],
        rotate=(0.0, 0.0, 180.0),
    )

# The direct corridor is bounded by the upper and centre rack rows.
direct_y = 1.6

# The conflict zone doubles as an easily visible status indicator.
cube(f"{ROOT}/Supervisor/ConflictZone", (0.0, direct_y, 0.035), (0.92, 0.92, 0.025), (0.12, 0.68, 0.22), 0.30)

# Candidate aisle routes. The controller highlights the selected path while
# leaving alternatives dimly visible for the roof-camera explanation.
route_dim = (0.12, 0.23, 0.31)
cube(f"{ROOT}/Supervisor/Routes/Direct", (0.0, direct_y, 0.072), (6.35, 0.10, 0.018), route_dim, 0.26)

# South bypass: turn through the rack's deliberate central opening, use the
# aisle between the centre and lower rows, then return at the east shelf end.
cube(f"{ROOT}/Supervisor/Routes/SouthEntry", (0.0, 0.0, 0.074), (0.10, 1.60, 0.018), route_dim, 0.26)
cube(f"{ROOT}/Supervisor/Routes/SouthHorizontal", (2.8, -1.6, 0.074), (2.80, 0.10, 0.018), route_dim, 0.26)
cube(f"{ROOT}/Supervisor/Routes/SouthExit", (5.6, 0.0, 0.074), (0.10, 1.60, 0.018), route_dim, 0.26)

# North bypass: reverse to the west shelf end, travel around the upper row,
# and return through the east end cross-aisle.
cube(f"{ROOT}/Supervisor/Routes/NorthEntry", (-5.6, 3.2, 0.074), (0.10, 1.60, 0.018), route_dim, 0.26)
cube(f"{ROOT}/Supervisor/Routes/NorthHorizontal", (0.0, 4.8, 0.074), (5.60, 0.10, 0.018), route_dim, 0.26)
cube(f"{ROOT}/Supervisor/Routes/NorthExit", (5.6, 3.2, 0.074), (0.10, 1.60, 0.018), route_dim, 0.26)

for index, x_value in enumerate((5.95, 6.25, 6.55)):
    cube(
        f"{ROOT}/Supervisor/Goal_{index}",
        (x_value, direct_y, 0.04),
        (0.12, 0.48, 0.03),
        (0.10, 0.75, 0.95),
    )

# Keep the physical actors and warehouse visible while disabling tutorial-only
# floor, route, status, footprint, and beacon overlays.
for overlay_path in (
    f"{ROOT}/Markings",
    f"{ROOT}/Supervisor",
    f"{ROOT}/Actors/RobotBlue/IdentityPad",
    f"{ROOT}/Actors/RobotBlue/Beacon",
    f"{ROOT}/Actors/RobotGreen/IdentityPad",
    f"{ROOT}/Actors/RobotGreen/Beacon",
    f"{ROOT}/Actors/ForkliftOrange/IdentityPad",
    f"{ROOT}/Actors/ForkliftOrange/Beacon",
    f"{ROOT}/Actors/WorkerYellow/IdentityPad",
):
    overlay_prim = stage.GetPrimAtPath(overlay_path)
    if overlay_prim.IsValid():
        overlay_prim.SetActive(False)

# Four roof cameras: one overview and three overlapping zones.
xform(f"{ROOT}/Sensors")
camera_specs = {
    "RoofOverview": ((0.0, 0.8, 21.0), 15.0),
    "WestAisle": ((-3.6, 1.0, 11.0), 22.0),
    "Intersection": ((0.0, 1.6, 10.0), 24.0),
    "EastAisle": ((3.6, 1.0, 11.0), 22.0),
}
for name, (position, focal_length) in camera_specs.items():
    camera = UsdGeom.Camera.Define(stage, f"{ROOT}/Sensors/{name}")
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr().Set(focal_length)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))
    UsdGeom.XformCommonAPI(camera.GetPrim()).SetTranslate(Gf.Vec3d(*position))

dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/Lighting/Dome")
dome.CreateIntensityAttr(850.0)
distant = UsdLux.DistantLight.Define(stage, f"{ROOT}/Lighting/Key")
distant.CreateIntensityAttr(1500.0)
distant.CreateAngleAttr(0.8)
UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(35.0, -25.0, 20.0))

from omni.kit.viewport.utility import get_active_viewport
viewport = get_active_viewport()
if viewport is not None:
    viewport.camera_path = Sdf.Path(f"{ROOT}/Sensors/RoofOverview")

stable_frames = 0
for _ in range(900):
    loading_status = omni.usd.get_context().get_stage_loading_status()
    stable_frames = stable_frames + 1 if not any(loading_status) else 0
    if stable_frames >= 60:
        break
    await omni.kit.app.get_app().next_update_async()

print(json.dumps({
    "created": True,
    "root": ROOT,
    "actors": [
        f"{ROOT}/Actors/RobotBlue",
        f"{ROOT}/Actors/RobotGreen",
        f"{ROOT}/Actors/ForkliftOrange",
        f"{ROOT}/Actors/WorkerYellow",
    ],
    "cameras": [
        f"{ROOT}/Sensors/RoofOverview",
        f"{ROOT}/Sensors/WestAisle",
        f"{ROOT}/Sensors/Intersection",
        f"{ROOT}/Sensors/EastAisle",
    ],
    "supervisor_mode": "deterministic_tutorial_oracle",
    "asset_mode": "nvidia_actors_and_props",
    "environment_strategy": "compact_procedural_shell",
    "resolved_assets": resolved_assets,
}))
'''.strip()


def create_realistic_scene_source() -> str:
    """Build the same scenario inside NVIDIA's pre-modeled warehouse.

    The compact builder remains the default for fast iteration. This variant
    swaps only its procedural shell/racks for ``full_warehouse.usd`` and keeps
    the stable actor, route, camera, and notification paths.
    """
    source = create_scene_source()
    source = source.replace(
        'render_settings.set("/rtx/post/tonemap/fNumber", 5.0)',
        'render_settings.set("/rtx/post/tonemap/fNumber", 8.0)',
    )
    source = source.replace(
        'render_settings.set("/rtx/post/histogram/enabled", True)',
        (
            'render_settings.set("/rtx/post/histogram/enabled", False)\n'
            'render_settings.set("/rtx/raytracing/showLights", 2)\n'
            'render_settings.set("/rtx-transient/post/aa/limitedOps", False)\n'
            'render_settings.set("/rtx/post/aa/op", 4)'
        ),
    )
    source = source.replace(
        'render_settings.set("/rtx/post/aa/autoExposureMode", 1)',
        'render_settings.set("/rtx/post/aa/autoExposureMode", 2)',
    )
    source = source.replace(
        'render_settings.set("/rtx/post/aa/exposure", 1.0)',
        'render_settings.set("/rtx/post/aa/exposure", 0.2)',
    )
    start_marker = "# Building shell, marked traffic lanes, and camera coverage boundaries."
    end_marker = 'if resolved_assets["pallet"]:'
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    environment_source = r'''
# Compose NVIDIA's complete pre-modeled Simple Warehouse as the visual base.
# It supplies the building envelope, production materials, lights, shelves,
# racks, stocked props, ducts, and other details missing from the compact shell.
if not resolved_assets["warehouse"]:
    raise RuntimeError("The NVIDIA full_warehouse.usd asset did not resolve")
warehouse_placement = xform(
    f"{ROOT}/Environment/WarehousePlacement",
    translate=(16.9, 18.95, 0.0),
)
UsdGeom.XformCommonAPI(warehouse_placement).SetRotate(
    Gf.Vec3f(0.0, 0.0, 90.0)
)
warehouse_asset = reference_asset(
    f"{ROOT}/Environment/WarehousePlacement/NvidiaWarehouse",
    resolved_assets["warehouse"],
)
storage_prefixes = (
    "sm_rack",
    "rack_",
    "sm_palette",
    "palette_",
    "pallet_",
    "sm_cardbox",
    "cardbox",
    "box_",
    "sm_crate",
    "crate_",
    "sm_barel",
    "sm_barrel",
    "barel_",
    "barrel_",
    "sm_bottle",
    "bottle_",
)
for warehouse_child in warehouse_asset.GetChildren():
    child_name = warehouse_child.GetName().lower()
    # Roof-camera presentation cutaway: ceiling panels, overhead beams, and
    # their tall support pillars would otherwise occlude actors and routes.
    if (
        "ceiling" in child_name
        or "beam" in child_name
        or "pillar" in child_name
        or child_name.startswith(storage_prefixes)
    ):
        UsdGeom.Imageable(warehouse_child).MakeInvisible()

# Rebuild the working area from standalone Isaac Sim rack components. Three
# parallel rows form two physical aisles. Every row has a two-metre central
# opening for cross traffic, while x=+/-5.6 remain clear end cross-aisles.
required_rack_assets = ("rack_shelf", "rack_frame")
if not all(resolved_assets[name] for name in required_rack_assets):
    raise RuntimeError("The standalone NVIDIA rack assets did not resolve")

row_centers = (-3.2, 0.0, 3.2)
segment_centers = (-3.0, 3.0)
frame_positions = (-5.0, -1.0, 1.0, 5.0)
for row_index, row_y in enumerate(row_centers):
    for segment_index, segment_x in enumerate(segment_centers):
        for level_index, shelf_z in enumerate((1.3, 2.8, 4.1)):
            reference_asset(
                f"{ROOT}/AisleLayout/Row{row_index}/"
                f"Segment{segment_index}/Shelf{level_index}",
                resolved_assets["rack_shelf"],
                translate=(segment_x, row_y, shelf_z),
            )
        if (
            (row_index + segment_index) % 2 == 0
            and resolved_assets["pallet"]
            and resolved_assets["cardbox"]
        ):
            reference_asset(
                f"{ROOT}/AisleLayout/Row{row_index}/"
                f"Segment{segment_index}/StockPallet",
                resolved_assets["pallet"],
                translate=(
                    segment_x,
                    row_y,
                    1.34,
                ),
            )
            reference_asset(
                f"{ROOT}/AisleLayout/Row{row_index}/"
                f"Segment{segment_index}/StockCarton",
                resolved_assets["cardbox"],
                translate=(
                    segment_x,
                    row_y,
                    1.58,
                ),
            )
    for frame_index, frame_x in enumerate(frame_positions):
        for frame_level, frame_z in enumerate((0.0, 3.0)):
            reference_asset(
                f"{ROOT}/AisleLayout/Row{row_index}/"
                f"Frame{frame_index}_{frame_level}",
                resolved_assets["rack_frame"],
                translate=(frame_x, row_y, frame_z),
            )
    if resolved_assets["rack_shield"]:
        for shield_index, shield_x in enumerate((-5.0, 5.0)):
            reference_asset(
                f"{ROOT}/AisleLayout/Row{row_index}/Shield{shield_index}",
                resolved_assets["rack_shield"],
                translate=(shield_x, row_y, 0.0),
                rotate=(0.0, 0.0, 180.0 if shield_index else 0.0),
            )

# Optional floor markings are authored for future experiments but disabled in
# the presentation scene. Selected supervisor routes and the conflict zone live
# under /Supervisor and remain independently controllable.
aisle_cyan = (0.08, 0.70, 0.88)
for aisle_name, aisle_y in (
    ("South", -1.6),
    ("Direct", 1.6),
    ("NorthPerimeter", 4.8),
):
    for dash_index, dash_x in enumerate(
        (-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0)
    ):
        cube(
            f"{ROOT}/Markings/{aisle_name}/Dash{dash_index}",
            (dash_x, aisle_y, 0.028),
            (0.42, 0.035, 0.014),
            aisle_cyan,
            0.72,
        )

cube(
    f"{ROOT}/Markings/HoldLine",
    (-1.2, 1.6, 0.034),
    (0.055, 0.72, 0.018),
    (0.94, 0.94, 0.94),
    0.94,
)
markings_group = stage.GetPrimAtPath(f"{ROOT}/Markings")
if markings_group.IsValid():
    markings_group.SetActive(False)

'''
    source = source[:start] + environment_source + source[end:]
    source = source.replace(
        '"asset_mode": "nvidia_actors_and_props",',
        '"asset_mode": "nvidia_full_warehouse_actors_and_props",',
    )
    source = source.replace(
        '"environment_strategy": "compact_procedural_shell",',
        '"environment_strategy": "nvidia_full_warehouse_custom_aisle_layout",',
    )
    source = source.replace(
        "dome.CreateIntensityAttr(850.0)",
        "dome.CreateIntensityAttr(150.0)",
    )
    source = source.replace(
        "distant.CreateIntensityAttr(1500.0)",
        "distant.CreateIntensityAttr(250.0)",
    )
    return source


def create_lightweight_live_scene_source() -> str:
    """Build the aisle layout from standalone assets without the heavy shell."""
    source = create_realistic_scene_source()
    source = source.replace(
        "from pxr import Gf, Sdf, UsdGeom, UsdLux",
        (
            "from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics\n"
            "from isaacsim.core.experimental.objects import GroundPlane"
        ),
    )
    start_marker = "# Compose NVIDIA's complete pre-modeled Simple Warehouse as the visual base."
    rebuild_marker = "# Rebuild the working area from standalone Isaac Sim rack components."
    start = source.index(start_marker)
    rebuild = source.index(rebuild_marker, start)
    lightweight_shell = r'''
# Lightweight live-navigation shell. The standalone rack and actor references
# below preserve the recognizable Isaac Sim warehouse look without composing
# full_warehouse.usd and its irrelevant ceiling, ducts, storage, and fixtures.
physics_scene = UsdPhysics.Scene.Define(stage, f"{ROOT}/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr(9.81)
GroundPlane(
    f"{ROOT}/Environment/PhysicsGround",
    sizes=40.0,
    positions=[0.0, 0.0, -0.01],
    templates=None,
)
floor_prim = cube(
    f"{ROOT}/Environment/Floor",
    (0.0, 0.8, -0.08),
    (24.0, 18.0, 0.16),
    (0.12, 0.15, 0.18),
)
for wall_y in (-4.75, 6.35):
    wall_prim = cube(
        f"{ROOT}/Environment/Walls/Y_{str(wall_y).replace('-', 'm').replace('.', '_')}",
        (0.0, wall_y, 1.95),
        (14.8, 0.16, 3.90),
        (0.22, 0.25, 0.29),
    )
    UsdPhysics.CollisionAPI.Apply(wall_prim)
for wall_x in (-7.32, 7.32):
    wall_prim = cube(
        f"{ROOT}/Environment/Walls/X_{str(wall_x).replace('-', 'm').replace('.', '_')}",
        (wall_x, 0.8, 1.95),
        (0.16, 11.1, 3.90),
        (0.22, 0.25, 0.29),
    )
    UsdPhysics.CollisionAPI.Apply(wall_prim)

'''
    source = source[:start] + lightweight_shell + source[rebuild:]
    source = source.replace(
        '"asset_mode": "nvidia_full_warehouse_actors_and_props",',
        '"asset_mode": "nvidia_standalone_racks_actors_and_props",',
    )
    source = source.replace(
        '"environment_strategy": "nvidia_full_warehouse_custom_aisle_layout",',
        '"environment_strategy": "lightweight_standalone_asset_aisles",',
    )
    source = source.replace(
        "dome.CreateIntensityAttr(150.0)",
        "dome.CreateIntensityAttr(550.0)",
    )
    source = source.replace(
        "distant.CreateIntensityAttr(250.0)",
        "distant.CreateIntensityAttr(900.0)",
    )
    source = source.replace(
        'render_settings.set("/rtx/raytracing/showLights", 2)',
        'render_settings.set("/rtx/raytracing/showLights", 0)',
    )
    source = source.replace(
        "scale=(1.45, 1.45, 1.45)",
        "scale=(2.0, 2.0, 2.0)",
    )
    source = source.replace(
        '\nprint(json.dumps({\n    "created": True,',
        (
            "\nomni.usd.get_context().get_selection()"
            ".clear_selected_prim_paths()\n"
            'print(json.dumps({\n    "created": True,'
        ),
    )
    return source


def inspect_realistic_warehouse_source() -> str:
    """Return bounded layout diagnostics for the composed warehouse asset."""
    return r'''
import json
import carb
import omni.usd
from pxr import Usd, UsdGeom

ROOT = "/World/CodexPoC/Environment/WarehousePlacement/NvidiaWarehouse"
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
root = stage.GetPrimAtPath(ROOT)
if not root.IsValid():
    raise RuntimeError("Run isaac_create_realistic_edge_supervisor_scene first")

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
)


def rounded_vec(value):
    return [round(float(value[index]), 3) for index in range(3)]


def world_info(prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    info = {
        "path": str(prim.GetPath()),
        "translate": rounded_vec(matrix.ExtractTranslation()),
        "bbox_min": rounded_vec(world_range.GetMin()),
        "bbox_max": rounded_vec(world_range.GetMax()),
    }
    intensity = prim.GetAttribute("inputs:intensity")
    exposure = prim.GetAttribute("inputs:exposure")
    if intensity.IsValid():
        info["intensity"] = intensity.Get()
    if exposure.IsValid():
        info["exposure"] = exposure.Get()
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        info["visibility"] = str(imageable.ComputeVisibility())
    return info


groups = {
    "floor": [],
    "ceiling": [],
    "structural": [],
    "storage": [],
    "light": [],
}
for child in root.GetChildren():
    name = child.GetName().lower()
    if "floor" in name and len(groups["floor"]) < 12:
        groups["floor"].append(world_info(child))
    if "ceiling" in name and len(groups["ceiling"]) < 12:
        groups["ceiling"].append(world_info(child))
    if (
        any(token in name for token in ("beam", "column", "pillar"))
        and len(groups["structural"]) < 80
    ):
        groups["structural"].append(world_info(child))
    if (
        any(token in name for token in ("rack", "shelf", "pallet", "palette", "box"))
        and len(groups["storage"]) < 80
    ):
        groups["storage"].append(world_info(child))
    if "light" in name and len(groups["light"]) < 40:
        groups["light"].append(world_info(child))

root_range = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
print(json.dumps({
    "root": ROOT,
    "bbox_min": rounded_vec(root_range.GetMin()),
    "bbox_max": rounded_vec(root_range.GetMax()),
    "groups": groups,
    "top_level_prim_count": len(root.GetChildren()),
    "render_settings": {
        key: carb.settings.get_settings().get(key)
        for key in (
            "/rtx/rendermode",
            "/rtx/post/tonemap/exposure",
            "/rtx/post/tonemap/cameraShutter",
            "/rtx/post/tonemap/filmIso",
            "/rtx/post/tonemap/fNumber",
            "/rtx/post/tonemap/op",
            "/rtx/post/histogram/enabled",
            "/rtx/post/histogram/whiteScale",
            "/rtx/post/histogram/useExposureClamping",
            "/rtx/post/histogram/minEV",
            "/rtx/post/histogram/maxEV",
            "/rtx-transient/post/aa/limitedOps",
            "/rtx/post/aa/op",
            "/rtx/post/aa/autoExposureMode",
            "/rtx/post/aa/exposureMultiplier",
            "/rtx/post/aa/exposure",
            "/rtx/post/dlss/execMode",
            "/rtx/raytracing/showLights",
        )
    },
}))
'''.strip()


def _progress_value(arguments: dict[str, Any]) -> float:
    value = arguments.get("progress")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("progress must be a number between 0 and 1")
    progress = float(value)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be a number between 0 and 1")
    return progress


def _isaac_update_function_source() -> str:
    return r'''
def set_actor_translate(path, value):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError("Run isaac_create_edge_supervisor_scene first")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*value))


def set_gprim_visual(path, color, opacity):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    gprim = UsdGeom.Gprim(prim)
    gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    gprim.GetDisplayOpacityAttr().Set([float(opacity)])


def set_route_visual(route, color):
    dim = (0.12, 0.23, 0.31)
    direct = [f"{ROOT}/Supervisor/Routes/Direct"]
    south = [
        f"{ROOT}/Supervisor/Routes/SouthEntry",
        f"{ROOT}/Supervisor/Routes/SouthHorizontal",
        f"{ROOT}/Supervisor/Routes/SouthExit",
    ]
    north = [
        f"{ROOT}/Supervisor/Routes/NorthEntry",
        f"{ROOT}/Supervisor/Routes/NorthHorizontal",
        f"{ROOT}/Supervisor/Routes/NorthExit",
    ]
    for path in direct + south + north:
        set_gprim_visual(path, dim, 0.25)
    if route == "south_bypass":
        active = south
    elif route == "north_bypass":
        active = north
    else:
        active = direct
    for path in active:
        set_gprim_visual(path, color, 0.92)


def update_scene(progress, scenario):
    progress = max(0.0, min(1.0, float(progress)))
    if scenario not in ("mixed_traffic", "aisle_congestion", "blind_corner"):
        raise RuntimeError("Unknown edge-supervisor scenario")

    def south_bypass_position(alpha):
        # Central rack opening -> south aisle -> east shelf end.
        if alpha < 0.15:
            return (-1.8 + 1.8 * (alpha / 0.15), 1.6)
        if alpha < 0.35:
            return (0.0, 1.6 - 3.2 * ((alpha - 0.15) / 0.20))
        if alpha < 0.80:
            return (5.6 * ((alpha - 0.35) / 0.45), -1.6)
        return (5.6, -1.6 + 3.2 * ((alpha - 0.80) / 0.20))

    def north_bypass_position(alpha):
        # Reverse to west shelf end -> north perimeter -> east shelf end.
        if alpha < 0.18:
            return (-1.8 - 3.8 * (alpha / 0.18), 1.6)
        if alpha < 0.38:
            return (-5.6, 1.6 + 3.2 * ((alpha - 0.18) / 0.20))
        if alpha < 0.80:
            return (-5.6 + 11.2 * ((alpha - 0.38) / 0.42), 4.8)
        return (5.6, 4.8 - 3.2 * ((alpha - 0.80) / 0.20))

    if scenario == "mixed_traffic":
        if progress < 0.22:
            robot_x = -6.4 + 4.6 * (progress / 0.22)
            robot_y = 1.6
        elif progress < 0.68:
            robot_x = -1.8
            robot_y = 1.6
        elif progress < 0.88:
            alpha = (progress - 0.68) / 0.20
            robot_x, robot_y = south_bypass_position(alpha)
        else:
            alpha = (progress - 0.88) / 0.12
            robot_x = 5.6 + 0.9 * alpha
            robot_y = 1.6

        forklift_alpha = max(0.0, min(1.0, (progress - 0.08) / 0.50))
        forklift_x = 0.0
        forklift_y = 4.8 - 9.6 * forklift_alpha
        worker_alpha = max(0.0, min(1.0, (progress - 0.36) / 0.38))
        worker_x = 0.0
        worker_y = 4.6 - 9.2 * worker_alpha
        green_alpha = max(0.0, min(1.0, (progress - 0.48) / 0.27))
        green_x = 6.4 - 3.2 * green_alpha
        green_y = 1.6

        if progress < 0.22:
            action, hazard, severity, route = "CONTINUE", "none", "info", "direct"
            reason = "No predicted conflict in the three-second horizon"
        elif progress < 0.52:
            action, hazard, severity, route = "YIELD", "forklift_crossing", "warning", "hold_west_gate"
            reason = "Forklift predicted to enter the shared intersection"
        elif progress < 0.68:
            action, hazard, severity, route = "STOP", "worker_proximity", "critical", "hold_west_gate"
            reason = "Worker detected inside the robot exclusion zone"
        elif progress < 0.88:
            action, hazard, severity, route = "REROUTE", "robot_congestion", "warning", "south_bypass"
            reason = "Second AMR is blocking the east aisle"
        else:
            action, hazard, severity, route = "CONTINUE", "none", "info", "direct_to_goal"
            reason = "Conflicts cleared and the alternate lane is open"

    elif scenario == "aisle_congestion":
        if progress < 0.20:
            robot_x = -6.4 + 4.6 * (progress / 0.20)
            robot_y = 1.6
        elif progress < 0.42:
            robot_x, robot_y = -1.8, 1.6
        elif progress < 0.88:
            alpha = (progress - 0.42) / 0.46
            robot_x, robot_y = south_bypass_position(alpha)
        else:
            robot_x, robot_y = 6.50, 1.6

        green_alpha = max(0.0, min(1.0, (progress - 0.05) / 0.35))
        green_x = 6.4 - 4.2 * green_alpha
        green_y = 1.6
        forklift_x, forklift_y = -6.2, -3.9
        worker_x, worker_y = 5.7, 4.6

        if progress < 0.20:
            action, hazard, severity, route = "CONTINUE", "none", "info", "direct"
            reason = "Direct aisle has the lowest travel cost"
        elif progress < 0.42:
            action, hazard, severity, route = "YIELD", "opposing_robot", "warning", "hold_west_gate"
            reason = "Opposing AMR has entered the single-width aisle"
        elif progress < 0.88:
            action, hazard, severity, route = "REROUTE", "aisle_congestion", "warning", "south_bypass"
            reason = "South loop is clear and avoids the blocked aisle"
        else:
            action, hazard, severity, route = "CONTINUE", "none", "info", "direct_to_goal"
            reason = "RobotBlue has cleared the congested shelf row"

    else:
        if progress < 0.25:
            robot_x = -6.4 + 4.6 * (progress / 0.25)
            robot_y = 1.6
        elif progress < 0.50:
            robot_x, robot_y = -1.8, 1.6
        elif progress < 0.82:
            alpha = (progress - 0.50) / 0.32
            robot_x, robot_y = north_bypass_position(alpha)
        else:
            robot_x, robot_y = 6.50, 1.6

        forklift_alpha = max(0.0, min(1.0, (progress - 0.15) / 0.40))
        forklift_x = 0.0
        forklift_y = -4.6 + 7.4 * forklift_alpha
        worker_x, worker_y = -6.4, -3.8
        green_x, green_y = 4.2, -1.6

        if progress < 0.25:
            action, hazard, severity, route = "CONTINUE", "none", "info", "direct"
            reason = "Blind corner is clear in the prediction horizon"
        elif progress < 0.50:
            action, hazard, severity, route = "YIELD", "forklift_blind_corner", "warning", "hold_west_gate"
            reason = "Roof camera sees a forklift hidden from RobotBlue"
        elif progress < 0.82:
            action, hazard, severity, route = "REROUTE", "forklift_blind_corner", "warning", "north_bypass"
            reason = "North loop clears the forklift's swept path"
        else:
            action, hazard, severity, route = "CONTINUE", "none", "info", "direct_to_goal"
            reason = "Forklift has cleared the cross-aisle"

    set_actor_translate(f"{ROOT}/Actors/RobotBlue", (robot_x, robot_y, 0.0))
    set_actor_translate(f"{ROOT}/Actors/ForkliftOrange", (forklift_x, forklift_y, 0.0))
    set_actor_translate(f"{ROOT}/Actors/WorkerYellow", (worker_x, worker_y, 0.0))
    set_actor_translate(f"{ROOT}/Actors/RobotGreen", (green_x, green_y, 0.0))

    color = {
        "CONTINUE": (0.15, 0.95, 0.30),
        "YIELD": (1.0, 0.78, 0.05),
        "STOP": (1.0, 0.10, 0.08),
        "REROUTE": (0.78, 0.16, 0.96),
    }[action]

    beacon_prim = stage.GetPrimAtPath(f"{ROOT}/Actors/RobotBlue/Beacon")
    if beacon_prim.IsValid():
        beacon = UsdGeom.Gprim(beacon_prim)
        beacon.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    conflict_prim = stage.GetPrimAtPath(f"{ROOT}/Supervisor/ConflictZone")
    if conflict_prim.IsValid():
        conflict = UsdGeom.Gprim(conflict_prim)
        conflict.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    set_route_visual(route, color)
    return {
        "scenario": scenario,
        "recipient": "RobotBlue",
        "action": action,
        "reason": reason,
        "hazard": hazard,
        "severity": severity,
        "route": route,
        "horizon_seconds": 3.0,
        "robot_position": [robot_x, robot_y, 0.0],
        "forklift_position": [forklift_x, forklift_y, 0.0],
        "worker_position": [worker_x, worker_y, 0.0],
        "other_robot_position": [green_x, green_y, 0.0],
    }
'''.strip()


def set_progress_source(arguments: dict[str, Any]) -> str:
    progress = _progress_value(arguments)
    scenario = _scenario_value(arguments)
    update_function = _isaac_update_function_source()
    return f'''
import json
import math
import omni.usd
from pxr import Gf, UsdGeom

ROOT = {SCENE_ROOT!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")

{update_function}

notification = update_scene({progress!r}, {scenario!r})
print(json.dumps({{
    "scenario": {scenario!r},
    "progress": {progress!r},
    "notification": notification,
}}))
'''.strip()


def capture_camera_views_source(
    arguments: dict[str, Any],
    *,
    capture_root_base: Path | None = None,
) -> str:
    progress = _progress_value(arguments)
    scenario = _scenario_value(arguments)
    base = capture_root_base or _capture_root()
    capture_root = (base / scenario).resolve()
    capture_root.mkdir(parents=True, exist_ok=True)
    camera_items = list(CAMERA_PATHS.items())
    root_json = json.dumps(str(capture_root))
    cameras_json = json.dumps(camera_items)
    update_function = _isaac_update_function_source()
    return f'''
import json
import math
import os
import omni.usd
from pxr import Gf, Sdf, UsdGeom
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

ROOT = {SCENE_ROOT!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active Isaac Sim viewport is available")

{update_function}

stable_frames = 0
for _ in range(900):
    loading_status = omni.usd.get_context().get_stage_loading_status()
    stable_frames = stable_frames + 1 if not any(loading_status) else 0
    if stable_frames >= 60:
        break
    await omni.kit.app.get_app().next_update_async()

notification = update_scene({progress!r}, {scenario!r})
output_root = {root_json}
os.makedirs(output_root, exist_ok=True)
camera_items = {cameras_json}
outputs = {{}}
for camera_name, camera_path in camera_items:
    viewport.camera_path = Sdf.Path(camera_path)
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    output_path = os.path.join(output_root, f"{{camera_name}}.png")
    capture = capture_viewport_to_file(viewport, file_path=output_path)
    await capture.wait_for_result(completion_frames=5)
    await omni.kit.app.get_app().next_update_async()
    outputs[camera_name] = output_path

viewport.camera_path = Sdf.Path({CAMERA_PATHS["overview"]!r})
for _ in range(2):
    await omni.kit.app.get_app().next_update_async()
for _ in range(120):
    if all(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in outputs.values()
    ):
        break
    await omni.kit.app.get_app().next_update_async()
print(json.dumps({{
    "scenario": {scenario!r},
    "progress": {progress!r},
    "notification": notification,
    "camera_outputs": outputs,
    "all_exist": all(os.path.isfile(path) for path in outputs.values()),
}}))
'''.strip()


def _frame_count_value(arguments: dict[str, Any]) -> int:
    value = arguments.get("frame_count", 25)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("frame_count must be an integer between 8 and 48")
    if not 8 <= value <= 48:
        raise ValueError("frame_count must be an integer between 8 and 48")
    return value


def capture_sequence_source(
    arguments: dict[str, Any],
    *,
    capture_root_base: Path | None = None,
) -> str:
    frame_count = _frame_count_value(arguments)
    scenario = _scenario_value(arguments)
    base = capture_root_base or _capture_root()
    settle_frames = (
        8
        if base in (
            REALISTIC_V2_CAPTURE_ROOT,
            REALISTIC_V3_CAPTURE_ROOT,
            REALISTIC_V4_CAPTURE_ROOT,
        )
        else 3
    )
    frame_root = (base / scenario / "frames").resolve()
    frame_root.mkdir(parents=True, exist_ok=True)
    root_json = json.dumps(str(frame_root))
    update_function = _isaac_update_function_source()
    return f'''
import json
import math
import os
import omni.usd
from pxr import Gf, Sdf, UsdGeom
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

ROOT = {SCENE_ROOT!r}
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is currently open")
viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active Isaac Sim viewport is available")

{update_function}

stable_frames = 0
for _ in range(900):
    loading_status = omni.usd.get_context().get_stage_loading_status()
    stable_frames = stable_frames + 1 if not any(loading_status) else 0
    if stable_frames >= 60:
        break
    await omni.kit.app.get_app().next_update_async()

output_root = {root_json}
os.makedirs(output_root, exist_ok=True)
viewport.camera_path = Sdf.Path({CAMERA_PATHS["overview"]!r})
outputs = []
notifications = []
for index in range({frame_count!r}):
    progress = index / max(1, {frame_count!r} - 1)
    notification = update_scene(progress, {scenario!r})
    for _ in range({settle_frames!r}):
        await omni.kit.app.get_app().next_update_async()
    output_path = os.path.join(output_root, f"frame_{{index:03d}}.png")
    capture = capture_viewport_to_file(viewport, file_path=output_path)
    await capture.wait_for_result(completion_frames=3)
    await omni.kit.app.get_app().next_update_async()
    outputs.append(output_path)
    notifications.append({{
        "frame": index,
        "progress": progress,
        "action": notification["action"],
        "hazard": notification["hazard"],
    }})

update_scene(0.52, {scenario!r})
for _ in range(120):
    if all(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in outputs
    ):
        break
    await omni.kit.app.get_app().next_update_async()
print(json.dumps({{
    "scenario": {scenario!r},
    "frame_count": len(outputs),
    "frame_root": output_root,
    "first_frame": outputs[0],
    "last_frame": outputs[-1],
    "all_exist": all(os.path.isfile(path) for path in outputs),
    "notifications": notifications,
}}))
'''.strip()


def capture_realistic_camera_views_source(arguments: dict[str, Any]) -> str:
    capture_root = _realistic_capture_root(arguments)
    capture_root.mkdir(parents=True, exist_ok=True)
    return capture_camera_views_source(
        arguments,
        capture_root_base=capture_root,
    )


def capture_realistic_sequence_source(arguments: dict[str, Any]) -> str:
    capture_root = _realistic_capture_root(arguments)
    capture_root.mkdir(parents=True, exist_ok=True)
    return capture_sequence_source(
        arguments,
        capture_root_base=capture_root,
    )
