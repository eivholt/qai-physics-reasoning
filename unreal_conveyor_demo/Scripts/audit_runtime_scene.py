"""Audit runtime wheel, worker, and animation bindings in the imported level."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "runtime_scene.json"


def names(values: object) -> list[str]:
    return [str(value) for value in list(values)]


def component_record(actor: unreal.Actor, component: unreal.SceneComponent) -> dict[str, object]:
    parent = component.get_attach_parent()
    record: dict[str, object] = {
        "actor": actor.get_actor_label(),
        "name": component.get_name(),
        "class": component.get_class().get_name(),
        "tags": names(component.get_editor_property("component_tags")),
        "mobility": str(component.get_editor_property("mobility")),
        "world_location": [round(value, 3) for value in component.get_world_location().to_tuple()],
        "parent": parent.get_name() if parent is not None else "",
    }
    if isinstance(component, unreal.SkeletalMeshComponent):
        skeletal_mesh = component.get_editor_property("skeletal_mesh_asset")
        record["skeletal_mesh"] = skeletal_mesh.get_path_name() if skeletal_mesh is not None else ""
        for property_name in ("animation_mode", "animation_data"):
            try:
                record[property_name] = str(component.get_editor_property(property_name))
            except Exception as error:
                record[property_name] = f"<unavailable: {error}>"
    if isinstance(component, unreal.PrimitiveComponent):
        origin, extent, _ = unreal.SystemLibrary.get_component_bounds(component)
        record["bounds_origin"] = [round(value, 3) for value in origin.to_tuple()]
        record["bounds_extent"] = [round(value, 3) for value in extent.to_tuple()]
    return record


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
components: list[dict[str, object]] = []
actor_records: list[dict[str, object]] = []
for actor in actors:
    scene_components = actor.get_components_by_class(unreal.SceneComponent)
    actor_tags = names(actor.get_editor_property("tags"))
    interesting = bool(actor_tags) or any(
        token in actor.get_actor_label().lower()
        for token in ("worker", "male_adult", "forklift", "walk")
    )
    if interesting:
        actor_records.append(
            {
                "label": actor.get_actor_label(),
                "class": actor.get_class().get_name(),
                "tags": actor_tags,
                "root": actor.get_editor_property("root_component").get_name(),
                "location": [round(value, 3) for value in actor.get_actor_location().to_tuple()],
                "components": len(scene_components),
            }
        )
    for component in scene_components:
        component_tags = names(component.get_editor_property("component_tags"))
        if component_tags or isinstance(component, unreal.SkeletalMeshComponent):
            components.append(component_record(actor, component))

wheel_tags = sorted(
    tag
    for component in components
    for tag in component["tags"]
    if tag.startswith("Qai.Forklift") and ".Wheel." in tag
)
worker_components = [
    component for component in components if any(tag.startswith("Qai.Worker") for tag in component["tags"])
]
skeletal_components = [component for component in components if component["class"] == "SkeletalMeshComponent"]
forklift_components = [
    component_record(actor, component)
    for actor in actors
    for component in actor.get_components_by_class(unreal.SceneComponent)
    if any(
        token in component.get_name().lower()
        for token in ("forklift", "lift", "fork", "mast", "forcer", "carriage", "stage")
    )
    or any(
        str(tag).startswith("Qai.Forklift")
        for tag in list(component.get_editor_property("component_tags"))
    )
]
report = {
    "wheel_tags": wheel_tags,
    "wheel_tag_count": len(wheel_tags),
    "worker_roots": worker_components,
    "worker_root_count": len(worker_components),
    "skeletal_components": skeletal_components,
    "forklift_components": forklift_components,
    "actors": actor_records,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

errors: list[str] = []
if len(wheel_tags) != 8 or len(set(wheel_tags)) != 8:
    errors.append(f"expected eight unique wheel bindings, found {wheel_tags}")
if len(worker_components) != 2:
    errors.append(f"expected two worker roots, found {len(worker_components)}")
if len(skeletal_components) != 2:
    errors.append(f"expected exactly two animated workers, found {len(skeletal_components)} skeletal components")
worker_animations = [
    component["animation_data"]
    for component in skeletal_components
    if "Workers/Worker" in component.get("skeletal_mesh", "")
]
if len(worker_animations) != 2 or any("saved_looping: True" not in value or "saved_playing: True" not in value for value in worker_animations):
    errors.append(f"worker walk loops are not persisted as playing/looping: {worker_animations}")
if errors:
    raise RuntimeError("; ".join(errors))

unreal.log(f"QAI_CONVEYOR_RUNTIME_SCENE={json.dumps(report, separators=(',', ':'))}")
