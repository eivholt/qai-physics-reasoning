#!/usr/bin/env python3
"""Audit composed conveyor prims before creating the lean installer stage."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument(
        "--expand",
        action="append",
        default=[],
        help="Also report the direct children beneath this prim path (repeatable)",
    )
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp  # noqa: E402


APP = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402


def subtree_stats(prim: Usd.Prim) -> dict[str, object]:
    types: collections.Counter[str] = collections.Counter()
    visible = 0
    invisible = 0
    layers: set[str] = set()
    prim_count = 0
    for descendant in Usd.PrimRange(prim):
        prim_count += 1
        types[descendant.GetTypeName() or "(scope)"] += 1
        imageable = UsdGeom.Imageable(descendant)
        if imageable:
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                invisible += 1
            else:
                visible += 1
        for spec in descendant.GetPrimStack():
            layers.add(spec.layer.identifier)
    return {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName() or "(scope)",
        "active": prim.IsActive(),
        "instance": prim.IsInstance(),
        "prim_count": prim_count,
        "visible_imageables": visible,
        "invisible_imageables": invisible,
        "types": types.most_common(12),
        "contributing_layers": len(layers),
    }


def main() -> int:
    stage = Usd.Stage.Open(str(ARGS.stage.resolve()), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open {ARGS.stage}")
    pseudo = stage.GetPseudoRoot()
    report = {
        "root_children": [subtree_stats(child) for child in pseudo.GetChildren()],
        "world_children": [subtree_stats(child) for child in stage.GetPrimAtPath("/World").GetChildren()],
        "conveyor_children": [
            subtree_stats(child)
            for child in stage.GetPrimAtPath("/World/CodexPoC/ConveyorSafety").GetChildren()
        ],
        "expanded": {},
    }
    for path in ARGS.expand:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Requested --expand prim does not exist: {path}")
        report["expanded"][path] = [subtree_stats(child) for child in prim.GetChildren()]
    print("CONVEYOR_AUDIT=" + json.dumps(report, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    finally:
        APP.close()
    raise SystemExit(exit_code)
