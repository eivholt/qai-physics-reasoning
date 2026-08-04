"""Import the private IQ9 STEP temporarily and inventory removable subassemblies."""

from __future__ import annotations

import json
import os
from pathlib import Path

import unreal


PROJECT_DIR = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())
SOURCE_STEP = os.path.normpath(
    os.path.join(
        PROJECT_DIR,
        "..",
        "resources",
        "DP25-73418-2_RevB",
        "10-75699-RB8-PVT.stp",
    )
)
DESTINATION = "/Game/IQ9EVK/CADInspect"


def main() -> None:
    if not os.path.isfile(SOURCE_STEP):
        raise RuntimeError(f"IQ9 EVK STEP source is missing: {SOURCE_STEP}")
    unreal.EditorAssetLibrary.delete_directory(DESTINATION)
    scene = unreal.DatasmithSceneElement.construct_datasmith_scene_from_cad_files([SOURCE_STEP])
    if not scene:
        raise RuntimeError("Datasmith CAD translator did not open the IQ9 STEP source")
    tessellation = scene.get_options(unreal.DatasmithCommonTessellationOptions)
    if tessellation:
        options = tessellation.get_editor_property("options")
        options.set_editor_property("chord_tolerance", 1.0)
        options.set_editor_property("max_edge_length", 20.0)
        options.set_editor_property("normal_tolerance", 30.0)
        tessellation.set_editor_property("options", options)
    result = scene.import_scene(DESTINATION)
    if not result or not result.import_succeed:
        raise RuntimeError("Datasmith CAD inspection import failed")

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    records: list[dict[str, object]] = []
    for actor in actor_subsystem.get_all_level_actors():
        if not isinstance(actor, unreal.StaticMeshActor):
            continue
        component = actor.static_mesh_component
        mesh = component.static_mesh if component else None
        mesh_path = mesh.get_path_name() if mesh else ""
        if "/IQ9EVK/CADInspect/" not in mesh_path:
            continue
        origin, extent = actor.get_actor_bounds(False)
        records.append(
            {
                "actor": actor.get_actor_label(),
                "actor_name": actor.get_name(),
                "mesh": mesh_path,
                "origin_cm": list(origin.to_tuple()),
                "extent_cm": list(extent.to_tuple()),
                "size_cm": [extent.x * 2.0, extent.y * 2.0, extent.z * 2.0],
                "materials": [
                    component.get_material(index).get_path_name()
                    if component.get_material(index)
                    else ""
                    for index in range(component.get_num_materials())
                ],
            }
        )
    records.sort(key=lambda item: (str(item["actor"]), str(item["mesh"])))
    output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "iq9_components.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    unreal.EditorAssetLibrary.delete_directory(DESTINATION)
    unreal.log(f"QAI_IQ9_COMPONENT_AUDIT actors={len(records)} output={output}")


if __name__ == "__main__":
    main()
