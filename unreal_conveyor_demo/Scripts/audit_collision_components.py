"""Audit the authored USD collision proxies retained by the Unreal import."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "collision_components.json"
COLLIDER_NAMES = {
    "Surface",
    "ShelfPlane1",
    "ShelfPlane2",
    "ShelfPlane3",
    "RearGuard",
    "Frame1Upright1",
    "Frame1Upright2",
    "Frame2Upright1",
    "Frame2Upright2",
    "Frame3Upright1",
    "Frame3Upright2",
    "PackingTable",
    "NorthWall",
    "NorthWall_01",
    "NorthWall_02",
    "WestWall",
}


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
rows: list[dict[str, object]] = []
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    for component in actor.get_components_by_class(unreal.PrimitiveComponent):
        logical_name = component.get_name().rsplit("_", 1)[0]
        if logical_name not in COLLIDER_NAMES:
            continue
        origin, extent, radius = unreal.SystemLibrary.get_component_bounds(component)
        rows.append(
            {
                "actor": actor.get_actor_label(),
                "component": component.get_name(),
                "logical_name": logical_name,
                "class": component.get_class().get_name(),
                "origin": [round(value, 3) for value in origin.to_tuple()],
                "extent": [round(value, 3) for value in extent.to_tuple()],
                "radius": round(radius, 3),
                "visible": bool(component.get_editor_property("visible")),
                "collision_enabled": str(component.get_collision_enabled()),
            }
        )

rows.sort(key=lambda row: str(row["logical_name"]))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

found = {str(row["logical_name"]) for row in rows}
required = {"Surface", "ShelfPlane1", "PackingTable", "NorthWall", "WestWall"}
missing = sorted(required - found)
if missing:
    raise RuntimeError(f"Required authored collision proxies are missing: {missing}")
if any(float(value) <= 0.0 for row in rows for value in row["extent"]):
    raise RuntimeError("One or more authored collision proxies has an empty bound")

unreal.log(f"QAI_CONVEYOR_COLLISION_COMPONENTS={json.dumps(rows, separators=(',', ':'))}")
