from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


BASELINE_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "isaac_sim_checkpoints"
    / "reason2_conveyor_safety_oval_physics_r25.usda"
)


SOURCE_TEMPLATE = r"""
import json
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

baseline_path = __BASELINE_PATH__
root_path = "/World/CodexPoC/ConveyorSafety"
current = omni.usd.get_context().get_stage()
baseline = Usd.Stage.Open(baseline_path)
cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    useExtentsHint=True,
)


def matrix_values(prim):
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
        return None
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    return [round(float(value), 6) for row in matrix for value in row]


def references(prim):
    value = prim.GetMetadata("references") if prim and prim.IsValid() else None
    return str(value) if value else ""


def world_bounds(prim):
    try:
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            return None
        minimum = box.GetMin()
        maximum = box.GetMax()
        return {
            "min": [round(float(value), 4) for value in minimum],
            "max": [round(float(value), 4) for value in maximum],
            "size": [
                round(float(maximum[i] - minimum[i]), 4)
                for i in range(3)
            ],
        }
    except Exception:
        return None


def record(prim, change):
    path = str(prim.GetPath())
    return {
        "change": change,
        "path": path,
        "type": prim.GetTypeName(),
        "local_matrix": matrix_values(prim),
        "world_bounds": world_bounds(prim),
        "references": references(prim),
        "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
        "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
    }


current_paths = {
    str(prim.GetPath()): prim
    for prim in current.Traverse()
    if str(prim.GetPath()).startswith(root_path)
}
baseline_paths = {
    str(prim.GetPath()): prim
    for prim in baseline.Traverse()
    if str(prim.GetPath()).startswith(root_path)
}
changes = []
for path, prim in current_paths.items():
    if path.count("/") > root_path.count("/") + 3:
        continue
    old = baseline_paths.get(path)
    if old is None:
        changes.append(record(prim, "added"))
        continue
    changed = (
        prim.GetTypeName() != old.GetTypeName()
        or matrix_values(prim) != matrix_values(old)
        or references(prim) != references(old)
    )
    if changed:
        item = record(prim, "changed")
        item["baseline_local_matrix"] = matrix_values(old)
        item["baseline_references"] = references(old)
        changes.append(item)
for path, prim in baseline_paths.items():
    if (
        path.count("/") <= root_path.count("/") + 3
        and path not in current_paths
    ):
        changes.append(record(prim, "removed"))

interesting_roots = []
for prim in current.Traverse():
    path = str(prim.GetPath())
    if not path.startswith(root_path):
        continue
    lower = path.lower()
    if (
        path.count("/") <= root_path.count("/") + 3
        and any(word in lower for word in ("stacklight", "wall", "table"))
    ):
        interesting_roots.append(record(prim, "current"))

print(
    json.dumps(
        {
            "current_root_layer": current.GetRootLayer().identifier,
            "baseline_root_layer": baseline.GetRootLayer().identifier,
            "change_count": len(changes),
            "changes": changes,
            "interesting_prims": interesting_roots,
        }
    )
)
""".strip()


def main() -> None:
    source = SOURCE_TEMPLATE.replace(
        "__BASELINE_PATH__",
        json.dumps(str(BASELINE_PATH.resolve())),
    )
    response = IsaacTcpClient(timeout_seconds=120.0).execute(source)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
