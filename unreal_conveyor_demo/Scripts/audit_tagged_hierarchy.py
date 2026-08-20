"""Report tagged scene roots and their immediate children for native binding."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


unreal.EditorLoadingAndSavingUtils.load_map("/Game/ConveyorRuntime/WarehouseConveyor")
rows = []
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    components = actor.get_components_by_class(unreal.SceneComponent)
    for component in components:
        tags = [str(tag) for tag in component.get_editor_property("component_tags")]
        name = component.get_name()
        if not tags and name.lower().rsplit("_", 1)[0] not in {
            "forklift1", "body", "lift"
        }:
            continue
        parent = component.get_attach_parent()
        children = [child.get_name() for child in component.get_children_components(False)]
        rows.append(
            {
                "actor": actor.get_actor_label(),
                "name": name,
                "parent": parent.get_name() if parent else "",
                "tags": tags,
                "children": children,
                "path": component.get_path_name(),
            }
        )

output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "tagged_hierarchy.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
unreal.log(f"QAI_TAGGED_HIERARCHY={len(rows)} output={output}")
