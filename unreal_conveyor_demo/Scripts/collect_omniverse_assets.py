#!/usr/bin/env python3
"""Collect the exact conveyor stage and its complete Omniverse dependency closure.

Run this script with Isaac Sim's ``python.bat``/``python.sh``. It deliberately
fails on any missing USD, MDL, texture, or other external reference. The
resulting payload is the only visual source used by the Unreal demo and is
staged into the personal-use installers as NonUFS content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
DEFAULT_SOURCE = (
    REPO_DIR
    / "artifacts"
    / "isaac_sim_checkpoints"
    / "reason2_conveyor_safety_stable_cargo_r38.usda"
)
DEFAULT_OUTPUT = PROJECT_DIR / "Content" / "OmniversePayload"
REQUIRED_PRIMS = (
    "/World/CodexPoC/ConveyorSafety/Forklifts/Forklift1",
    "/World/CodexPoC/ConveyorSafety/Conveyor/Parcels/Parcel1",
    "/World/CodexPoC/ConveyorSafety/DynamicCargo/Forklift1Pallet",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorEndline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Allow updating a previously collected payload")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify and manifest an already-collected payload without contacting source servers",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    args, _ = parser.parse_known_args()
    return args


ARGS = parse_args()

# Omniverse extensions are unavailable until SimulationApp has initialized.
from isaacsim import SimulationApp  # noqa: E402


APP = SimulationApp({"headless": ARGS.headless})

import omni.client  # noqa: E402
from omni.kit.usd.collect import Collector, CollectorFailureOptions  # noqa: E402
from pxr import Sdf, Usd, UsdShade, UsdUtils  # noqa: E402


def local_path(value: str) -> Path:
    # urllib treats a Windows drive letter as a URI scheme (``C:``). Resolve
    # native absolute paths before URI parsing so self-containment checks do
    # not misclassify every collected layer as external on Windows.
    if len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}:
        return Path(value)
    parsed = urlparse(value)
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return Path(path)
    if parsed.scheme:
        raise ValueError(f"Collector returned a non-local URL: {value}")
    return Path(unquote(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def progress(current: int, total: int) -> None:
    if total and (current == total or current % max(1, total // 20) == 0):
        print(f"asset collection: {current}/{total}", flush=True)


def ensure_output_policy(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [item for item in output.iterdir() if item.name != ".gitkeep"]
    if existing and not ARGS.refresh:
        raise RuntimeError(
            f"{output} already contains a payload. Re-run with --refresh to verify and update it."
        )


def verify_collected_stage(root: Path, output: Path) -> dict[str, object]:
    output = output.resolve()
    root = root.resolve()
    if output not in root.parents:
        raise RuntimeError(f"Collected root is outside the payload: {root}")

    stage = Usd.Stage.Open(str(root), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"USD could not reopen the collected root: {root}")
    print(f"opened collected stage: {root}", flush=True)
    missing_prims = [path for path in REQUIRED_PRIMS if not stage.GetPrimAtPath(path).IsValid()]
    if missing_prims:
        raise RuntimeError(f"Collected stage is missing required conveyor prims: {missing_prims}")

    dependencies = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root)))
    if len(dependencies) != 3:
        raise RuntimeError(f"Unexpected ComputeAllDependencies result: {dependencies!r}")
    layers, assets, unresolved = dependencies
    print(
        f"dependency scan: {len(layers)} layers, {len(assets)} assets, {len(unresolved)} unresolved",
        flush=True,
    )
    if unresolved:
        missing_path = output / "missing_dependency_paths.txt"
        missing_path.write_text(
            "\n".join(sorted(str(item) for item in unresolved)) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Collected stage still has {len(unresolved)} unresolved dependencies; "
            f"full list: {missing_path}. First entries: {list(unresolved)[:20]}"
        )
    (output / "missing_dependency_paths.txt").unlink(missing_ok=True)
    (output / "asset_collection_error.txt").unlink(missing_ok=True)

    external = []
    for layer in layers:
        real_path = getattr(layer, "realPath", "") or getattr(layer, "identifier", "")
        if not real_path:
            continue
        try:
            resolved = local_path(str(real_path)).resolve()
        except ValueError:
            external.append(str(real_path))
            continue
        if resolved != root and output not in resolved.parents:
            external.append(str(resolved))
    for asset in assets:
        asset_value = getattr(asset, "path", str(asset))
        parsed = urlparse(str(asset_value))
        if parsed.scheme in {"http", "https", "omniverse"}:
            external.append(str(asset_value))
    if external:
        raise RuntimeError(
            "The collected stage is not self-contained; external references remain: "
            + ", ".join(external[:20])
        )

    shader_ids: dict[str, int] = {}
    material_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_count += 1
        if prim.IsA(UsdShade.Shader):
            identifier = str(UsdShade.Shader(prim).GetIdAttr().Get() or "(source-asset)")
            shader_ids[identifier] = shader_ids.get(identifier, 0) + 1
    print(f"material scan: {material_count} materials, shader ids={shader_ids}", flush=True)

    files = []
    total_bytes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {".gitkeep", "asset_manifest.json"}):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": size,
                "sha256": sha256(path),
            }
        )
    if not files:
        raise RuntimeError("Collector produced an empty payload")
    return {
        "dependency_layer_count": len(layers),
        "referenced_asset_count": len(assets),
        "material_count": material_count,
        "shader_ids": shader_ids,
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


async def collect(source: Path, output: Path) -> str:
    options = (
        CollectorFailureOptions.EXTERNAL_USD_REFERENCES
        | CollectorFailureOptions.OTHER_EXTERNAL_REFERENCES
    )
    collector = Collector(
        usd_path=str(source),
        collect_dir=str(output),
        usd_only=False,
        flat_collection=False,
        material_only=False,
        failure_options=options,
        skip_existing=ARGS.refresh,
        max_concurrent_tasks=32,
        force_non_read_only=True,
    )
    success, root_url = await collector.collect(progress_callback=progress)
    if not success or not root_url:
        raise RuntimeError("Omniverse Collector did not produce a complete root stage")
    return root_url


def main() -> int:
    source = ARGS.source.resolve()
    output = ARGS.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Conveyor source stage is missing: {source}")
    ensure_output_policy(output)
    print(f"collecting exact stage: {source}", flush=True)
    print(f"payload destination:   {output}", flush=True)

    if ARGS.verify_existing:
        direct_root = output / source.name
        candidates = [direct_root] if direct_root.is_file() else list(output.rglob(source.name))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one existing collected root named {source.name}; found {len(candidates)}"
            )
        root = candidates[0]
    else:
        root_url = APP.run_coroutine(collect(source, output))
        root = local_path(root_url)
    details = verify_collected_stage(root, output)
    manifest = {
        "schema_version": 1,
        "purpose": "Private Unreal conveyor demo Omniverse asset payload",
        "source_stage": str(source),
        "source_stage_sha256": sha256(source),
        "root_stage": root.resolve().relative_to(output).as_posix(),
        "required_prims": list(REQUIRED_PRIMS),
        "collector": "omni.kit.usd.collect",
        **details,
    }
    manifest_path = output / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"verified {manifest['file_count']} files / {manifest['total_bytes'] / (1024**3):.2f} GiB; "
        f"root={manifest['root_stage']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except BaseException:
        exit_code = 1
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        try:
            ARGS.output.mkdir(parents=True, exist_ok=True)
            (ARGS.output / "asset_collection_error.txt").write_text(failure, encoding="utf-8")
        except OSError:
            pass
    finally:
        APP.close()
    raise SystemExit(exit_code)
