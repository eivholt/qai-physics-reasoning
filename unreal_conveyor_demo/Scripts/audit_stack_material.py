"""Inspect the imported stack-lens material interface and editable parameters."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


ASSET = "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/MI_DisplayColor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "stack_material.json"


material = unreal.EditorAssetLibrary.load_asset(ASSET)
if not isinstance(material, unreal.MaterialInstanceConstant):
    raise RuntimeError(f"Missing imported display-color material: {ASSET}")

library = unreal.MaterialEditingLibrary
parent = material.get_editor_property("parent")
report: dict[str, object] = {
    "asset": material.get_path_name(),
    "parent": parent.get_path_name() if isinstance(parent, unreal.Object) else "",
    "instance_methods": sorted(name for name in dir(material) if "parameter" in name.lower()),
    "library_methods": sorted(name for name in dir(library) if "parameter" in name.lower()),
}
for key, method_name in (
    ("scalar_parameter_names", "get_scalar_parameter_names"),
    ("vector_parameter_names", "get_vector_parameter_names"),
    ("texture_parameter_names", "get_texture_parameter_names"),
    ("static_switch_parameter_names", "get_static_switch_parameter_names"),
):
    method = getattr(library, method_name, None)
    if method is None:
        report[key] = []
        continue
    try:
        report[key] = [str(value) for value in method(material)]
    except Exception as error:
        report[key] = f"unavailable: {error}"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
unreal.log(f"QAI_STACK_MATERIAL_AUDIT={OUTPUT}")
