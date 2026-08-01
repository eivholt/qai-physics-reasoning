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
from pxr import UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
allowed = {
    "convexHull",
    "convexDecomposition",
    "boundingCube",
    "boundingSphere",
    "sdf",
}
dynamic_meshes = []
invalid = []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    ancestor = prim.GetParent()
    dynamic_parent = None
    while ancestor and ancestor.IsValid():
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            dynamic_parent = str(ancestor.GetPath())
            break
        ancestor = ancestor.GetParent()
    if dynamic_parent is None:
        continue
    collision_enabled = UsdPhysics.CollisionAPI(
        prim
    ).GetCollisionEnabledAttr().Get()
    approximation = UsdPhysics.MeshCollisionAPI(
        prim
    ).GetApproximationAttr().Get()
    item = {
        "path": str(prim.GetPath()),
        "dynamic_parent": dynamic_parent,
        "collision_enabled": collision_enabled is not False,
        "approximation": str(approximation or "none"),
    }
    dynamic_meshes.append(item)
    if item["approximation"] not in allowed:
        invalid.append(item)
print(json.dumps({
    "dynamic_mesh_count": len(dynamic_meshes),
    "invalid_dynamic_mesh_count": len(invalid),
    "invalid_dynamic_meshes": invalid,
    "disabled_carton_visual_meshes": [
        item for item in dynamic_meshes
        if not item["collision_enabled"] and "CardBox" in item["path"]
    ],
}))
""".strip()


def main() -> None:
    print(
        json.dumps(
            IsaacTcpClient(timeout_seconds=120.0).execute(SOURCE),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
