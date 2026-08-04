#!/usr/bin/env python3
"""Audit the authored UV/UDIM tile used by every forklift mesh.

Run with Isaac Sim's Python so the bundled USD build can read the source asset.
The report is intentionally independent of Unreal and is used by the native
importer to verify that the eight-tile forklift atlas is not collapsed to 1001.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_forklift_udims.py SOURCE_USD REPORT_JSON")
    source = Path(sys.argv[1]).resolve()
    report_path = Path(sys.argv[2]).resolve()
    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open {source}")

    rows: list[dict[str, object]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        values = primvar.ComputeFlattened() if primvar else None
        if not values:
            continue
        u_values = [float(value[0]) for value in values]
        v_values = [float(value[1]) for value in values]
        center_u = sum(u_values) / len(u_values)
        center_v = sum(v_values) / len(v_values)
        tile_u = math.floor(center_u + 1.0e-4)
        tile_v = math.floor(center_v + 1.0e-4)
        udim = 1001 + tile_u + 10 * tile_v
        material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        rows.append(
            {
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "uv_count": len(values),
                "min_u": min(u_values),
                "max_u": max(u_values),
                "min_v": min(v_values),
                "max_v": max(v_values),
                "center_u": center_u,
                "center_v": center_v,
                "udim": udim,
                "material": str(material.GetPath()) if material else "",
            }
        )

    if not rows:
        raise RuntimeError("No textured forklift meshes were found")
    report = {
        "schema_version": 1,
        "source": str(source),
        "mesh_count": len(rows),
        "udims": sorted({int(row["udim"]) for row in rows}),
        "meshes": sorted(rows, key=lambda row: str(row["path"])),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("QAI_FORKLIFT_UDIMS=" + json.dumps(report, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
