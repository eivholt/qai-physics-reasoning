#!/usr/bin/env python3
"""Apply the shipping light budget to an already imported native level.

This is deliberately separate from the USD importer so lighting iterations do
not rebuild hundreds of meshes and textures. The importer applies the same
values for clean builds.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_finish import apply_visual_finish


LEVEL_PATH = "/Game/ConveyorRuntime/WarehouseConveyor"
PROJECT = Path(unreal.Paths.project_dir())
REPORT = PROJECT / "Saved" / "ImportAudit" / "native_light_tuning.json"


def set_property(obj: object, name: str, value: object) -> None:
    obj.set_editor_property(name, value)  # type: ignore[attr-defined]


def main() -> None:
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.load_level(LEVEL_PATH):
        raise RuntimeError(f"Could not load {LEVEL_PATH}")

    report: list[dict[str, object]] = []
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        label = actor.get_actor_label()
        light = actor.get_component_by_class(unreal.LightComponent)
        sky = actor.get_component_by_class(unreal.SkyLightComponent)
        if label.startswith("Area") and light is not None:
            set_property(light, "intensity", 2400.0)
            set_property(light, "cast_shadows", label in {"Area1", "Area4"})
            report.append({"actor": label, "intensity": 2400.0, "cast_shadows": label in {"Area1", "Area4"}})
        elif label == "Glow" and light is not None:
            set_property(light, "intensity", 300.0)
            set_property(light, "cast_shadows", False)
            try:
                set_property(light, "attenuation_radius", 90.0)
            except Exception:
                pass
            report.append({"actor": label, "intensity": 300.0, "cast_shadows": False})
        elif (label.startswith("StackLightBounce") or label == "SceneWash") and light is not None:
            intensity = 0.0 if label.startswith("StackLightBounce") else 350.0
            set_property(light, "intensity", intensity)
            set_property(light, "cast_shadows", False)
            report.append({"actor": label, "intensity": intensity, "cast_shadows": False})
        if sky is not None:
            set_property(sky, "intensity", 1.15)
            set_property(sky, "real_time_capture", False)
            report.append({"actor": label, "intensity": 1.15, "real_time_capture": False})

    visual_report = apply_visual_finish()

    if not level_subsystem.save_current_level():
        raise RuntimeError(f"Could not save {LEVEL_PATH}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"level": LEVEL_PATH, "lights": report, **visual_report}, indent=2) + "\n",
        encoding="utf-8",
    )
    unreal.log(f"QAI_CONVEYOR_LIGHT_TUNING={json.dumps(report, separators=(',', ':'))}")


main()
