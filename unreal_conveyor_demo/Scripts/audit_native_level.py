"""Print compact transforms and bounds for the optimized native runtime level."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"


def vec(value: unreal.Vector) -> list[float]:
    return [round(value.x, 3), round(value.y, 3), round(value.z, 3)]


def rot(value: unreal.Rotator) -> list[float]:
    return [round(value.pitch, 3), round(value.yaw, 3), round(value.roll, 3)]


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
rows: list[dict[str, object]] = []
for actor in actors:
    label = actor.get_actor_label()
    tags = [str(tag) for tag in actor.get_editor_property("tags")]
    light = actor.get_component_by_class(unreal.LightComponent)
    sky = actor.get_component_by_class(unreal.SkyLightComponent)
    if tags or label in {"ConveyorSafety", "World", "DetectorEndline"} or light is not None or sky is not None:
        origin, extent = actor.get_actor_bounds(False)
        row: dict[str, object] = {
            "label": label,
            "class": actor.get_class().get_name(),
            "tags": tags,
            "location": vec(actor.get_actor_location()),
            "rotation": rot(actor.get_actor_rotation()),
            "bounds_origin": vec(origin),
            "bounds_extent": vec(extent),
        }
        if light is not None:
            row["light_intensity"] = light.get_editor_property("intensity")
            row["cast_shadows"] = light.get_editor_property("cast_shadows")
        if sky is not None:
            row["sky_intensity"] = sky.get_editor_property("intensity")
            row["real_time_capture"] = sky.get_editor_property("real_time_capture")
        rows.append(row)

output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "native_level_bounds.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
unreal.log(f"Wrote native level bounds: {output}")
