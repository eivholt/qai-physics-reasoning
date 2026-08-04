"""Report the exact level components using the corrected floor/rack materials."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "visual_components.json"


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
rows: list[dict[str, object]] = []
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    for component in actor.get_components_by_class(unreal.StaticMeshComponent):
        mesh = component.get_editor_property("static_mesh")
        mesh_path = mesh.get_path_name() if mesh is not None else ""
        if not (mesh_path.endswith("SM_ClearanceZone.SM_ClearanceZone") or mesh_path.endswith("SM_WestRack.SM_WestRack")):
            continue
        origin, extent, radius = unreal.SystemLibrary.get_component_bounds(component)
        rows.append(
            {
                "actor": actor.get_actor_label(),
                "component": component.get_name(),
                "mesh": mesh_path,
                "origin": [round(value, 3) for value in origin.to_tuple()],
                "extent": [round(value, 3) for value in extent.to_tuple()],
                "radius": round(radius, 3),
                "location": [round(value, 3) for value in component.get_world_location().to_tuple()],
                "materials": [
                    component.get_material(index).get_path_name() if component.get_material(index) else ""
                    for index in range(component.get_num_materials())
                ],
            }
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
if len(rows) != 2:
    raise RuntimeError(f"Expected clearance and rack components, found {len(rows)}")
unreal.log(f"QAI_CONVEYOR_VISUAL_COMPONENTS={json.dumps(rows, separators=(',', ':'))}")
