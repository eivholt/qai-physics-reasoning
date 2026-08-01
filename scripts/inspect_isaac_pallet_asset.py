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
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
path = "/World/CodexPoC/PalletProbe"
stage.RemovePrim(path)
probe = stage.DefinePrim(path, "Xform")
probe.GetReferences().AddReference(
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Props/Pallet/pallet.usd"
)
for _ in range(20):
    await omni.kit.app.get_app().next_update_async()
cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    useExtentsHint=True,
)
world_range = cache.ComputeWorldBound(probe).ComputeAlignedRange()
minimum = world_range.GetMin()
maximum = world_range.GetMax()
colliders = []
for prim in Usd.PrimRange(probe):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    approximation = None
    if prim.IsA(UsdGeom.Mesh):
        approximation = UsdPhysics.MeshCollisionAPI(
            prim
        ).GetApproximationAttr().Get()
    colliders.append({
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "approximation": str(approximation or ""),
    })
result = {
    "min": [float(item) for item in minimum],
    "max": [float(item) for item in maximum],
    "size": [float(item) for item in maximum - minimum],
    "colliders": colliders,
    "layers": sorted({
        spec.layer.identifier
        for prim in Usd.PrimRange(probe)
        for spec in prim.GetPrimStack()
    }),
}
stage.RemovePrim(path)
print(json.dumps(result))
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
