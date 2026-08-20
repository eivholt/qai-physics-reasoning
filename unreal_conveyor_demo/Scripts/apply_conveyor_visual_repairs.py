"""Apply only the straight-conveyor frame/roller visual repairs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visual_finish import (  # noqa: E402
    assign_low_profile_conveyor_frame_material,
    create_a11_two_sided_conveyor_frame_material,
    create_bridge_roller_material,
    create_hidden_straight_frame_material,
    create_low_profile_conveyor_frame_material,
    create_low_profile_frame_overlays,
    repair_low_profile_conveyor_materials,
)


def main() -> None:
    unreal.EditorLoadingAndSavingUtils.load_map(
        "/Game/ConveyorRuntime/WarehouseConveyor"
    )
    clipped_material = create_low_profile_conveyor_frame_material()
    opaque_frame_material = create_a11_two_sided_conveyor_frame_material()
    create_hidden_straight_frame_material()
    create_bridge_roller_material()
    report = {
        "low_profile_straight_frame_geometry": create_low_profile_frame_overlays(
            opaque_frame_material
        ),
        "two_sided_low_profile_materials": repair_low_profile_conveyor_materials(),
        "low_profile_straight_frame": assign_low_profile_conveyor_frame_material(
            clipped_material
        ),
    }
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise RuntimeError("Could not save WarehouseConveyor after visual repair")
    unreal.EditorAssetLibrary.save_directory(
        "/Game/ConveyorRuntime", only_if_is_dirty=False, recursive=True
    )
    output = (
        Path(unreal.Paths.project_saved_dir())
        / "ImportAudit"
        / "conveyor_visual_repairs.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_CONVEYOR_VISUAL_REPAIRS={output}")


main()
