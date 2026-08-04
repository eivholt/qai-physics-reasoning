"""Inventory movable cargo/rack candidates in the native runtime level."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "dynamic_physics_candidates.json"
TOKENS = (
    "rack", "shelf", "box", "carton", "parcel", "pallet", "packing", "crate", "forklift"
)


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
rows: list[dict[str, object]] = []
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    for component in actor.get_components_by_class(unreal.PrimitiveComponent):
        mesh = None
        try:
            mesh = component.get_editor_property("static_mesh")
        except Exception:
            pass
        mesh_path = mesh.get_path_name() if mesh is not None else ""
        searchable = f"{actor.get_actor_label()} {component.get_name()} {component.get_path_name()} {mesh_path}".lower()
        if not any(token in searchable for token in TOKENS):
            continue
        origin, extent, _ = unreal.SystemLibrary.get_component_bounds(component)
        parent = component.get_attach_parent()
        rows.append(
            {
                "actor": actor.get_actor_label(),
                "component": component.get_name(),
                "path": component.get_path_name(),
                "parent": parent.get_name() if parent else "",
                "mesh": mesh_path,
                "origin": [round(value, 3) for value in origin.to_tuple()],
                "extent": [round(value, 3) for value in extent.to_tuple()],
                "tags": [str(tag) for tag in component.get_editor_property("component_tags")],
                "visible": bool(component.get_editor_property("visible")),
            }
        )

rows.sort(key=lambda row: (str(row["actor"]), str(row["path"])))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
unreal.log(f"QAI_DYNAMIC_PHYSICS_CANDIDATES={len(rows)} output={OUTPUT}")
