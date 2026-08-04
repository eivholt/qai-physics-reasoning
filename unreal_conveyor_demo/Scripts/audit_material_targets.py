"""Record the exact material slots used by hero warehouse components."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


CONTENT_ROOT = "/Game/ConveyorRuntime"
TOKENS = (
    "forklift",
    "wheel",
    "tire",
    "roller",
    "cylinder",
    "lpg",
    "parcel",
    "cardbox",
    "rack",
    "shelf",
    "floor",
    "worker",
    "driver",
)


def main() -> None:
    unreal.EditorLoadingAndSavingUtils.load_map(f"{CONTENT_ROOT}/WarehouseConveyor")
    rows: list[dict[str, object]] = []
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    for actor in actors:
        for component in actor.get_components_by_class(unreal.MeshComponent):
            mesh = None
            if isinstance(component, unreal.StaticMeshComponent):
                mesh = component.get_editor_property("static_mesh")
            elif isinstance(component, unreal.SkeletalMeshComponent):
                mesh = component.get_editor_property("skeletal_mesh_asset")
            mesh_path = mesh.get_path_name() if mesh is not None else ""
            tags = [str(tag) for tag in component.get_editor_property("component_tags")]
            material_paths = []
            for slot in range(component.get_num_materials()):
                material = component.get_material(slot)
                material_paths.append(material.get_path_name() if material is not None else "")
            searchable = " ".join(
                [actor.get_actor_label(), component.get_name(), mesh_path, *tags, *material_paths]
            ).lower()
            if any(token in searchable for token in TOKENS):
                rows.append(
                    {
                        "actor": actor.get_actor_label(),
                        "component": component.get_name(),
                        "class": component.get_class().get_name(),
                        "mesh": mesh_path,
                        "tags": tags,
                        "materials": material_paths,
                    }
                )
    output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "material_targets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_MATERIAL_TARGETS={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
