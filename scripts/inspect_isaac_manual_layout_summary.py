from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


SOURCE = r"""
import json
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
root_path = "/World/CodexPoC/ConveyorSafety"
cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    useExtentsHint=True,
)


def vec(value):
    return [round(float(item), 6) for item in value] if value is not None else None


def info(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return {"path": path, "valid": False}
    matrix = (
        UsdGeom.Xformable(prim).GetLocalTransformation()
        if prim.IsA(UsdGeom.Xformable)
        else None
    )
    translate = list(matrix.ExtractTranslation()) if matrix is not None else None
    scale = list(matrix.ExtractRotationMatrix().GetOrthonormalized().GetRow(0)) if False else None
    bounds = None
    try:
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not aligned.IsEmpty():
            minimum = aligned.GetMin()
            maximum = aligned.GetMax()
            bounds = {
                "min": vec(minimum),
                "max": vec(maximum),
                "size": vec(maximum - minimum),
            }
    except Exception:
        pass
    refs = prim.GetMetadata("references")
    return {
        "path": path,
        "valid": True,
        "type": prim.GetTypeName(),
        "translate": vec(translate),
        "matrix": (
            [round(float(value), 6) for row in matrix for value in row]
            if matrix is not None
            else None
        ),
        "bounds": bounds,
        "references": str(refs) if refs else "",
        "asset_info": str(prim.GetAssetInfo()) if prim.HasAssetInfo() else "",
        "prim_stack_layers": [
            spec.layer.identifier for spec in prim.GetPrimStack()
        ],
        "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
        "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
    }


def direct_children(path):
    prim = stage.GetPrimAtPath(path)
    return [info(str(child.GetPath())) for child in prim.GetChildren()] if prim.IsValid() else []


root = stage.GetPrimAtPath(root_path)
root_attrs = {}
for name in (
    "codex:beltCenterXM",
    "codex:beltCenterYM",
    "codex:conveyorOvalRadiusM",
    "codex:conveyorOvalStraightHalfLengthM",
    "codex:conveyorTrackHalfWidthM",
    "codex:markedZoneHalfLengthM",
):
    attr = root.GetAttribute(name)
    root_attrs[name] = attr.Get() if attr.IsValid() else None

stack_roots = [
    info(str(child.GetPath()))
    for child in root.GetChildren()
    if child.GetName().startswith("StackLight")
]
parcel_positions = []
parcel_root = stage.GetPrimAtPath(f"{root_path}/Conveyor/Parcels")
for parcel in parcel_root.GetChildren():
    parcel_positions.append(info(str(parcel.GetPath())))

print(
    json.dumps(
        {
            "root_layer": stage.GetRootLayer().identifier,
            "root_attrs": root_attrs,
            "shell": direct_children(f"{root_path}/Shell"),
            "floor": [
                info(f"{root_path}/Floor"),
                info(f"{root_path}/FloorInset"),
                info(f"{root_path}/ClearanceZone"),
            ],
            "floor_finish": direct_children(f"{root_path}/FloorFinish"),
            "conveyor": info(f"{root_path}/Conveyor"),
            "conveyor_modules": direct_children(f"{root_path}/Conveyor/Modules"),
            "oval_colliders": direct_children(
                f"{root_path}/Physics/StaticColliders/OvalConveyor"
            ),
            "parcels": parcel_positions,
            "stack_lights": stack_roots,
            "stack_light_children": {
                name: direct_children(f"{root_path}/{name}")
                for name in ("StackLight", "StackLight_01", "StackLight_02")
            },
            "table": info(f"{root_path}/packing_table"),
            "table_children": direct_children(f"{root_path}/packing_table"),
        }
    )
)
""".strip()


def main() -> None:
    response = IsaacTcpClient(timeout_seconds=120.0).execute(SOURCE)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
