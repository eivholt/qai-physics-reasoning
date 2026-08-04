"""Record the native conveyor module transforms and local mesh bounds."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "conveyor_path.json"
NAMES = {"CurveNorth_0", "CurveSouth_0", "StraightEast_0", "StraightWest_0"}


def values(vector: object) -> list[float]:
    return [round(float(value), 4) for value in vector.to_tuple()]


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
rows: list[dict[str, object]] = []
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    for component in actor.get_components_by_class(unreal.StaticMeshComponent):
        if component.get_name() not in NAMES:
            continue
        mesh = component.get_editor_property("static_mesh")
        if not isinstance(mesh, unreal.StaticMesh):
            continue
        bounds = mesh.get_bounding_box()
        rows.append(
            {
                "component": component.get_name(),
                "location": values(component.get_world_location()),
                "rotation": values(component.get_world_rotation()),
                "scale": values(component.get_world_scale()),
                "local_min": values(bounds.min),
                "local_max": values(bounds.max),
                "mesh": mesh.get_path_name(),
            }
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
unreal.log(f"QAI_CONVEYOR_PATH_AUDIT={OUTPUT}")
