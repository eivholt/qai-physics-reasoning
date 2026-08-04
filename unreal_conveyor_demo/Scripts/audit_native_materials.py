"""Audit every native mesh slot and imported PreviewSurface parameter."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


CONTENT_ROOT = "/Game/ConveyorRuntime"


def object_path(value: object | None) -> str:
    return value.get_path_name() if value is not None else ""  # type: ignore[attr-defined]


assets = unreal.EditorAssetLibrary.list_assets(CONTENT_ROOT, recursive=True, include_folder=False)
rows: list[dict[str, object]] = []
for asset_path in assets:
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    if isinstance(asset, unreal.StaticMesh):
        materials = []
        for index, slot in enumerate(list(asset.get_editor_property("static_materials"))):
            materials.append(
                {
                    "slot": index,
                    "name": str(slot.get_editor_property("material_slot_name")),
                    "material": object_path(slot.get_editor_property("material_interface")),
                }
            )
        rows.append({"asset": asset_path, "class": "StaticMesh", "materials": materials})
    elif isinstance(asset, unreal.SkeletalMesh):
        materials = []
        for index, slot in enumerate(list(asset.get_editor_property("materials"))):
            materials.append(
                {
                    "slot": index,
                    "name": str(slot.get_editor_property("material_slot_name")),
                    "material": object_path(slot.get_editor_property("material_interface")),
                }
            )
        rows.append({"asset": asset_path, "class": "SkeletalMesh", "materials": materials})
    elif isinstance(asset, unreal.MaterialInstanceConstant):
        row: dict[str, object] = {
            "asset": asset_path,
            "class": "MaterialInstanceConstant",
            "parent": object_path(asset.get_editor_property("parent")),
            "textures": {},
            "scalars": {},
            "vectors": {},
        }
        library = unreal.MaterialEditingLibrary
        try:
            for name in library.get_texture_parameter_names(asset):
                value = library.get_material_instance_texture_parameter_value(asset, name)
                row["textures"][str(name)] = object_path(value)  # type: ignore[index]
        except Exception as error:
            row["texture_error"] = str(error)
        try:
            for name in library.get_scalar_parameter_names(asset):
                value = library.get_material_instance_scalar_parameter_value(asset, name)
                row["scalars"][str(name)] = value  # type: ignore[index]
        except Exception as error:
            row["scalar_error"] = str(error)
        try:
            for name in library.get_vector_parameter_names(asset):
                value = library.get_material_instance_vector_parameter_value(asset, name)
                row["vectors"][str(name)] = str(value)  # type: ignore[index]
        except Exception as error:
            row["vector_error"] = str(error)
        rows.append(row)

output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "native_materials.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
unreal.log(f"Wrote native material audit: {output}")
