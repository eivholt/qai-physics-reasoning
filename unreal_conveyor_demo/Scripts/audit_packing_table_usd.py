"""Audit the composed packing-table prim in the baked runtime stage."""

from __future__ import annotations

import json
from pathlib import Path

import unreal
from pxr import Usd, UsdGeom, UsdShade


PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())).resolve()
STAGE_PATH = PROJECT_DIR / "Saved" / "BakedStageGenerated" / "warehouse_conveyor_baked.usdc"
PRIM_PATH = "/World/CodexPoC/ConveyorSafety/packing_table"
OUTPUT_PATH = PROJECT_DIR / "Saved" / "ImportAudit" / "packing_table_usd.json"


stage = Usd.Stage.Open(str(STAGE_PATH), Usd.Stage.LoadAll)
if stage is None:
    raise RuntimeError(f"Could not open {STAGE_PATH}")
root = stage.GetPrimAtPath(PRIM_PATH)
if not root.IsValid():
    raise RuntimeError(f"Missing packing-table prim {PRIM_PATH}")

rows: list[dict[str, object]] = []
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Imageable):
        continue
    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    rows.append(
        {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "kind": str(prim.GetMetadata("kind") or ""),
            "instance": prim.IsInstance(),
            "visible": str(UsdGeom.Imageable(prim).ComputeVisibility()),
            "material": str(material.GetPath()) if material else "",
            "binding": str(relationship.GetPath()) if relationship else "",
        }
    )

report = {
    "stage": str(STAGE_PATH),
    "root": PRIM_PATH,
    "descendants": rows,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
unreal.log(f"QAI_PACKING_TABLE_USD={json.dumps(report, separators=(',', ':'))}")
