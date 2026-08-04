#!/usr/bin/env python3
"""Upgrade vertex attributes on an already imported native warehouse.

The clean importer applies the same settings. This focused pass avoids a full
USD rebuild when rendering quality is being iterated on an existing project.
"""

from __future__ import annotations

import json
from pathlib import Path

import unreal


CONTENT_ROOT = "/Game/ConveyorRuntime"
PROJECT = Path(unreal.Paths.project_dir())
REPORT = PROJECT / "Saved" / "ImportAudit" / "render_precision.json"


def is_hero_mesh(asset_path: str) -> bool:
    normalized = asset_path.lower().replace("\\", "/")
    return "/forklifts/" in normalized or "/conveyor/modules/" in normalized


def main() -> None:
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    upgraded: list[str] = []
    already_precise: list[str] = []

    for asset_path in unreal.EditorAssetLibrary.list_assets(
        CONTENT_ROOT, recursive=True, include_folder=False
    ):
        if not is_hero_mesh(asset_path):
            continue
        mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
        if not isinstance(mesh, unreal.StaticMesh):
            continue
        settings = subsystem.get_lod_build_settings(mesh, 0)
        full_uvs = bool(settings.get_editor_property("use_full_precision_u_vs"))
        high_tangents = bool(settings.get_editor_property("use_high_precision_tangent_basis"))
        if full_uvs and high_tangents:
            already_precise.append(asset_path)
            continue
        settings.set_editor_property("use_full_precision_u_vs", True)
        settings.set_editor_property("use_high_precision_tangent_basis", True)
        subsystem.set_lod_build_settings(mesh, 0, settings)
        if not unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False):
            raise RuntimeError(f"Could not save upgraded static mesh: {asset_path}")
        upgraded.append(asset_path)

    report = {
        "content_root": CONTENT_ROOT,
        "hero_meshes": len(upgraded) + len(already_precise),
        "upgraded": upgraded,
        "already_precise": already_precise,
        "full_precision_uvs": True,
        "high_precision_tangent_basis": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_CONVEYOR_RENDER_PRECISION={json.dumps(report, separators=(',', ':'))}")


main()
