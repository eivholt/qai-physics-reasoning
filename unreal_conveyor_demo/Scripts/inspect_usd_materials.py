#!/usr/bin/env python3
"""Report shader source types in the collected stage; run with Isaac Sim Python."""

from __future__ import annotations

import collections
import sys
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Usd, UsdShade  # noqa: E402


def main() -> int:
    stage_path = Path(sys.argv[1]).resolve()
    stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open {stage_path}")
    identifiers: collections.Counter[str] = collections.Counter()
    implementations: collections.Counter[str] = collections.Counter()
    source_assets: collections.Counter[str] = collections.Counter()
    materials = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            materials += 1
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        identifier = shader.GetIdAttr().Get()
        identifiers[str(identifier or "(none)")] += 1
        implementation = shader.GetImplementationSourceAttr().Get()
        implementations[str(implementation or "(none)")] += 1
        for source_type in shader.GetSourceAssetSubIdentifierAttr().GetMetadata("allowedTokens") or []:
            asset = shader.GetSourceAsset(str(source_type))
            if asset:
                source_assets[str(asset)] += 1
    print(f"materials={materials}", flush=True)
    print("shader_ids=" + repr(identifiers.most_common()), flush=True)
    print("implementations=" + repr(implementations.most_common()), flush=True)
    print("source_assets=" + repr(source_assets.most_common(30)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
