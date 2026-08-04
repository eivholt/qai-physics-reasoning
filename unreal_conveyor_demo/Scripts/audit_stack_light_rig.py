"""Audit stack-light geometry, imported helpers, and exposure controls."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "stack_light_rig.json"


def path_name(value: object | None) -> str:
    return value.get_path_name() if isinstance(value, unreal.Object) else ""


def vector(value: object) -> list[float]:
    return [round(float(item), 3) for item in value.to_tuple()]


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
rows: list[dict[str, object]] = []
post_process: list[dict[str, object]] = []
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    label = actor.get_actor_label()
    label_lower = label.lower()
    components = actor.get_components_by_class(unreal.SceneComponent)
    interesting = any(
        token in label_lower
        for token in ("stack", "glow", "green", "amber", "red")
    ) or any(
        any(token in component.get_name().lower() for token in ("stack", "glow", "green", "amber", "red"))
        for component in components
    )
    if interesting:
        component_rows: list[dict[str, object]] = []
        for component in components:
            row: dict[str, object] = {
                "name": component.get_name(),
                "class": component.get_class().get_name(),
                "location": vector(component.get_world_location()),
                "rotation": vector(component.get_world_rotation()),
                "tags": [str(tag) for tag in component.get_editor_property("component_tags")],
                "parent": component.get_attach_parent().get_name() if component.get_attach_parent() else "",
            }
            if isinstance(component, unreal.LightComponent):
                row.update(
                    {
                        "intensity": float(component.get_editor_property("intensity")),
                        "light_color": str(component.get_editor_property("light_color")),
                        "cast_shadows": bool(component.get_editor_property("cast_shadows")),
                        "visible": bool(component.get_editor_property("visible")),
                    }
                )
                if isinstance(component, unreal.LocalLightComponent):
                    row["attenuation_radius"] = float(component.get_editor_property("attenuation_radius"))
                if isinstance(component, unreal.PointLightComponent):
                    row["source_radius"] = float(component.get_editor_property("source_radius"))
                    row["soft_source_radius"] = float(component.get_editor_property("soft_source_radius"))
            if isinstance(component, unreal.StaticMeshComponent):
                row["mesh"] = path_name(component.get_editor_property("static_mesh"))
                row["materials"] = [path_name(component.get_material(index)) for index in range(component.get_num_materials())]
            component_rows.append(row)
        rows.append(
            {
                "actor": label,
                "class": actor.get_class().get_name(),
                "location": vector(actor.get_actor_location()),
                "rotation": vector(actor.get_actor_rotation()),
                "tags": [str(tag) for tag in actor.get_editor_property("tags")],
                "components": component_rows,
            }
        )

    component = actor.get_component_by_class(unreal.PostProcessComponent)
    if component is not None:
        settings = component.get_editor_property("settings")
        fields: dict[str, object] = {"actor": label}
        for name in (
            "auto_exposure_method",
            "auto_exposure_bias",
            "auto_exposure_min_brightness",
            "auto_exposure_max_brightness",
            "b_override_auto_exposure_method",
            "b_override_auto_exposure_bias",
            "b_override_auto_exposure_min_brightness",
            "b_override_auto_exposure_max_brightness",
            "b_override_bloom_intensity",
            "bloom_intensity",
            "b_override_bloom_threshold",
            "bloom_threshold",
        ):
            try:
                fields[name] = str(settings.get_editor_property(name))
            except Exception as error:  # Unreal changes some names across minor releases.
                fields[name] = f"unavailable: {error}"
        post_process.append(fields)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps({"actors": rows, "post_process": post_process}, indent=2) + "\n", encoding="utf-8")
unreal.log(f"QAI_STACK_LIGHT_AUDIT={OUTPUT}")
