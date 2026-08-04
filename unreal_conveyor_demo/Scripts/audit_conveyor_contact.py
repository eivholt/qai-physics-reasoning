"""Measure the rendered roller contact plane used by conveyor parcel physics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "conveyor_contact.json"


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
world = unreal.EditorLevelLibrary.get_editor_world()
records: list[dict[str, object]] = []

for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
    for component in actor.get_components_by_class(unreal.StaticMeshComponent):
        name = component.get_name()
        mesh = component.get_editor_property("static_mesh")
        mesh_name = mesh.get_name() if mesh else ""
        if "roller" not in name.lower() and "roller" not in mesh_name.lower():
            continue
        # SystemLibrary exposes the already transformed world AABB, including
        # the deep USD parent hierarchy and centimetre conversion.
        center, extent, _sphere_radius = unreal.SystemLibrary.get_component_bounds(component)
        top = center.z + extent.z
        bottom = center.z - extent.z
        records.append(
            {
                "actor": actor.get_name(),
                "component": name,
                "mesh": mesh_name,
                "center": [center.x, center.y, center.z],
                "bottom_z_cm": bottom,
                "top_z_cm": top,
            }
        )

tops = [float(record["top_z_cm"]) for record in records]
summary = {
    "roller_count": len(records),
    "top_z_min_cm": min(tops) if tops else None,
    "top_z_median_cm": median(tops) if tops else None,
    "top_z_max_cm": max(tops) if tops else None,
    "records": sorted(records, key=lambda record: (str(record["actor"]), str(record["component"]))),
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
unreal.log(f"QAI_CONVEYOR_CONTACT_AUDIT {json.dumps({key: value for key, value in summary.items() if key != 'records'})}")
