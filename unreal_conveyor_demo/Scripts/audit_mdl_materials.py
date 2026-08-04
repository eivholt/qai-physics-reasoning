"""Audit composed MDL shader inputs in the lean conveyor USD stage."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from isaacsim import SimulationApp


kit = SimulationApp({"headless": True})
import omni.kit.app  # noqa: E402
from pxr import Usd, UsdShade  # noqa: E402


manager = omni.kit.app.get_app().get_extension_manager()
distill_extensions = [
    extension.get("id", "")
    for extension in manager.get_extensions()
    if "distill" in extension.get("id", "").lower()
]
print(f"distill_extensions={distill_extensions}")
try:
    enabled = manager.set_extension_enabled_immediate("omni.mdl.distill_and_bake", True)
    import omni.mdl.distill_and_bake  # noqa: E402,F401

    print(f"distill_and_bake_available={enabled}")
except ImportError as error:
    print(f"distill_and_bake_available=False error={error}")


ROOT = Path(__file__).resolve().parents[1]
stage_path = ROOT / "Saved" / "LeanStage" / "warehouse_conveyor_runtime.usdc"
output_path = ROOT / "Saved" / "LeanStage" / "material_audit.json"
stage = Usd.Stage.Open(str(stage_path))
if stage is None:
    raise RuntimeError(f"Could not open {stage_path}")

rows: list[dict[str, object]] = []
bound_paths = set()
for scene_prim in stage.Traverse():
    bound_material, _ = UsdShade.MaterialBindingAPI(scene_prim).ComputeBoundMaterial()
    if bound_material:
        bound_paths.add(bound_material.GetPath())

for prim in stage.Traverse():
    material = UsdShade.Material(prim)
    if not material:
        continue
    shaders: list[dict[str, object]] = []
    for child in Usd.PrimRange(prim):
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        source_asset = shader.GetSourceAsset("mdl")
        source_subidentifier = shader.GetSourceAssetSubIdentifier("mdl")
        shader_inputs: dict[str, object] = {}
        for shader_input in shader.GetInputs():
            value = shader_input.Get()
            shader_inputs[shader_input.GetBaseName()] = None if value is None else str(value)
        shaders.append(
            {
                "path": str(child.GetPath()),
                "id": str(shader.GetIdAttr().Get() or ""),
                "source_asset": str(source_asset or ""),
                "source_subidentifier": str(source_subidentifier or ""),
                "inputs": shader_inputs,
            }
        )
    signature_payload = json.dumps(
        [{key: value for key, value in shader.items() if key != "path"} for shader in shaders],
        sort_keys=True,
        separators=(",", ":"),
    )
    rows.append(
        {
            "path": str(prim.GetPath()),
            "bound": prim.GetPath() in bound_paths,
            "signature": hashlib.sha256(signature_payload.encode("utf-8")).hexdigest(),
            "shaders": shaders,
        }
    )

output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
bound_signatures = {row["signature"] for row in rows if row["bound"]}
print(
    f"materials={len(rows)} bound={len(bound_paths)} "
    f"unique_bound_signatures={len(bound_signatures)} output={output_path}"
)
kit.close()
