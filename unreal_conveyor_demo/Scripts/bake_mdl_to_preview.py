"""Distill the lean conveyor's NVIDIA MDL materials to portable PBR USD.

Isaac Sim's MDL SDK is used only at build time.  The output stage contains
UsdPreviewSurface materials and baked textures that Unreal can import as
native cooked assets without Omniverse or MDL at runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim import SimulationApp
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0, help="Bake only N materials for pipeline diagnostics")
parser.add_argument("--start", type=int, default=0, help="Skip N sorted distinct bound materials")
parser.add_argument("--output-dir", type=Path, default=None, help="Generated bake directory under Saved")
args, _ = parser.parse_known_args()

# Hero assets use 2K bakes so labels, cloth weave, paint grain, cardboard
# fibre, and floor detail survive close views. Distant/background materials
# remain at 512 px. Forklift C keeps its eight authored 1K UDIM tiles packed
# into an 8192x1024 atlas, avoiding an unreliable 16K texture on portable Macs.
HERO_BAKE_RESOLUTION = 2048
BACKGROUND_BAKE_RESOLUTION = 512
FORKLIFT_TILE_RESOLUTION = 1024
HERO_PATH_MARKERS = (
    "/forklifts/",
    "/workers/",
    "/storage/",
    "/parcels/",
    "/dynamiccargo/",
    "/floorfinish/",
)


def bake_resolution_for_material(prim: Usd.Prim) -> int:
    path = str(prim.GetPath()).lower()
    return (
        HERO_BAKE_RESOLUTION
        if any(marker in path for marker in HERO_PATH_MARKERS)
        else BACKGROUND_BAKE_RESOLUTION
    )

project_dir = Path(__file__).resolve().parents[1]
source_stage = (project_dir / "Saved" / "LeanStage" / "warehouse_conveyor_runtime.usdc").resolve()
output_dir = (args.output_dir or (project_dir / "Saved" / "BakedStageGenerated")).resolve()
saved_dir = (project_dir / "Saved").resolve()
if saved_dir not in output_dir.parents or not output_dir.name.startswith("BakedStage"):
    raise RuntimeError(f"Refusing unsafe output directory: {output_dir}")
if not source_stage.is_file():
    raise FileNotFoundError(source_stage)
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)
output_stage = output_dir / "warehouse_conveyor_baked.usdc"
shutil.copy2(source_stage, output_stage)
texture_dir = output_dir / "BakedTextures"
texture_dir.mkdir()

kit = SimulationApp({"headless": True})
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402


manager = omni.kit.app.get_app().get_extension_manager()
if not manager.set_extension_enabled_immediate("omni.mdl.distill_and_bake", True):
    raise RuntimeError("Isaac Sim could not enable omni.mdl.distill_and_bake")
import omni.mdl.distill_and_bake  # noqa: E402


sanitized_root = output_dir / "SanitizedAssets"
sanitized_root.mkdir()
rewritten_assets = 0
supplemental_assets: dict[str, dict[str, object]] = {}
hydrated_udim_patterns: set[Path] = set()
hydrated_mdl_paths: set[Path] = set()
payload_root = project_dir / "Content" / "OmniversePayload"
asset_manifest_path = payload_root / "asset_manifest.json"
asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
declared_asset_paths = {entry["path"] for entry in asset_manifest["files"]}


def retrieve_exact_omniverse_asset(destination: Path) -> str:
    relative = destination.relative_to(payload_root).as_posix()
    host, remote_path = relative.split("/", 1)
    if host != "omniverse-content-production.s3-us-west-2.amazonaws.com":
        raise RuntimeError(f"Refusing unexpected Omniverse dependency host: {host}")
    url = f"https://{host}/{urllib.parse.quote(remote_path, safe='/')}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "QaiConveyorBuild/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    temporary.replace(destination)
    return url


def record_supplemental_asset(path: Path, url: str) -> None:
    payload_relative = path.relative_to(payload_root).as_posix()
    if payload_relative not in declared_asset_paths:
        supplemental_assets[payload_relative] = {
            "path": payload_relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "retrieved_url": url,
        }


def hydrate_udim_set(pattern_path: Path) -> None:
    if pattern_path in hydrated_udim_patterns:
        return
    relative = pattern_path.relative_to(payload_root).as_posix()
    host, remote_pattern = relative.split("/", 1)
    if host != "omniverse-content-production.s3-us-west-2.amazonaws.com":
        raise RuntimeError(f"Refusing unexpected UDIM host: {host}")
    directory, filename_pattern = remote_pattern.rsplit("/", 1)
    prefix = f"{directory}/{filename_pattern.split('<UDIM>', 1)[0]}"
    expression = re.compile("^" + re.escape(filename_pattern).replace(re.escape("<UDIM>"), r"\d{4}") + "$")

    # A successful collection/bake has already verified and retained every
    # exact tile in the private Omniverse payload. Prefer that local set so
    # repeat builds and installer assembly do not depend on NVIDIA S3.
    existing_tiles = sorted(
        path for path in pattern_path.parent.iterdir()
        if path.is_file() and expression.fullmatch(path.name)
    )
    if existing_tiles:
        for destination in existing_tiles:
            payload_relative = destination.relative_to(payload_root).as_posix()
            record_supplemental_asset(destination, f"https://{payload_relative}")
        hydrated_udim_patterns.add(pattern_path)
        return

    list_url = f"https://{host}/?{urllib.parse.urlencode({'list-type': '2', 'prefix': prefix})}"
    request = urllib.request.Request(list_url, headers={"User-Agent": "QaiConveyorBuild/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        listing = ET.fromstring(response.read())
    keys = []
    for element in listing.iter():
        if element.tag.endswith("}Key") or element.tag == "Key":
            key = element.text or ""
            if expression.fullmatch(key.rsplit("/", 1)[-1]):
                keys.append(key)
    if not keys:
        raise FileNotFoundError(f"No official UDIM tiles matched {pattern_path}")
    for key in keys:
        destination = payload_root / host / Path(key)
        asset_url = f"https://{host}/{urllib.parse.quote(key, safe='/')}"
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".download")
            request = urllib.request.Request(asset_url, headers={"User-Agent": "QaiConveyorBuild/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            temporary.replace(destination)
        record_supplemental_asset(destination, asset_url)
    hydrated_udim_patterns.add(pattern_path)


def hydrate_mdl_dependencies(source: Path) -> None:
    if source in hydrated_mdl_paths:
        return
    text = source.read_text(encoding="utf-8", errors="replace")
    for relative_name in re.findall(r'texture_2d\("([^"?]+)"', text):
        dependency = (source.parent / relative_name).resolve()
        if "<UDIM>" in dependency.name:
            hydrate_udim_set(dependency)
            continue
        retrieved_url = ""
        if not dependency.is_file():
            retrieved_url = retrieve_exact_omniverse_asset(dependency)
        record_supplemental_asset(
            dependency,
            retrieved_url or f"https://{dependency.relative_to(payload_root).as_posix()}",
        )
    hydrated_mdl_paths.add(source)


def sanitize_asset_path(asset_path: Sdf.AssetPath) -> Sdf.AssetPath:
    """Copy space-bearing MDL assets into a URI-safe build cache."""
    global rewritten_assets
    source = Path(asset_path.path)
    if "Material Library" not in source.parts or not source.is_file():
        return asset_path
    marker_index = source.parts.index("Material Library")
    destination = sanitized_root.joinpath(*source.parts[marker_index + 1 :])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        shutil.copy2(source, destination)
    if source.suffix.lower() == ".mdl":
        hydrate_mdl_dependencies(source)
        text = source.read_text(encoding="utf-8", errors="replace")
        library_root = Path(*source.parts[: marker_index + 1])
        relative_names = re.findall(r'texture_2d\("([^"?]+)"', text)
        for relative_name in relative_names:
            dependency = (source.parent / relative_name).resolve()
            if "<UDIM>" in dependency.name:
                continue
            dependency_relative = dependency.relative_to(library_root)
            dependency_destination = sanitized_root / dependency_relative
            dependency_destination.parent.mkdir(parents=True, exist_ok=True)
            if not dependency_destination.is_file() or dependency_destination.stat().st_size != dependency.stat().st_size:
                shutil.copy2(dependency, dependency_destination)
    rewritten_assets += 1
    return Sdf.AssetPath(destination.as_posix())


# Author URI-safe overrides before the renderer registers MDL entities. This
# avoids an MDL SDK access violation on paths containing "Material Library".
rewrite_stage = Usd.Stage.Open(str(output_stage))
if rewrite_stage is None:
    raise RuntimeError(f"Could not open stage for URI normalization: {output_stage}")
rewrite_material_paths = set()
for scene_prim in rewrite_stage.Traverse():
    if not omni.usd.is_prim_material_supported(scene_prim):
        continue
    bound_material, _ = UsdShade.MaterialBindingAPI(scene_prim).ComputeBoundMaterial()
    if bound_material:
        rewrite_material_paths.add(bound_material.GetPath())
for material_path in rewrite_material_paths:
    for rewrite_prim in Usd.PrimRange(rewrite_stage.GetPrimAtPath(material_path)):
        for attribute in rewrite_prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                source_path = Path(value.path)
                if source_path.suffix.lower() == ".mdl" and source_path.is_file():
                    hydrate_mdl_dependencies(source_path)
                if "<UDIM>" in source_path.name:
                    hydrate_udim_set(source_path)
                normalized = sanitize_asset_path(value)
                if normalized.path != value.path:
                    attribute.Set(normalized)
rewrite_stage.GetRootLayer().Save()
rewrite_stage = None
print(f"sanitized_asset_references={rewritten_assets}", flush=True)
print(f"supplemental_mdl_assets={len(supplemental_assets)}", flush=True)


usd_context = omni.usd.get_context()
if not usd_context.open_stage(str(output_stage)):
    raise RuntimeError(f"Omniverse could not open copied stage: {output_stage}")
kit.update()
stage = usd_context.get_stage()
if stage is None:
    raise RuntimeError(f"Could not open copied stage: {output_stage}")


def has_preview_surface(prim: Usd.Prim) -> bool:
    return any(
        UsdShade.Shader(child) and UsdShade.Shader(child).GetIdAttr().Get() == "UsdPreviewSurface"
        for child in Usd.PrimRange(prim)
    )


def get_bound_material_paths() -> set:
    paths = set()
    for scene_prim in stage.Traverse():
        if not omni.usd.is_prim_material_supported(scene_prim):
            continue
        material, _relationship = UsdShade.MaterialBindingAPI(scene_prim).ComputeBoundMaterial()
        if material:
            paths.add(material.GetPath())
    return paths


def material_signature(prim: Usd.Prim) -> str:
    shaders = []
    for child in Usd.PrimRange(prim):
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        shaders.append(
            {
                "id": str(shader.GetIdAttr().Get() or ""),
                "source_asset": str(shader.GetSourceAsset("mdl") or ""),
                "source_subidentifier": str(shader.GetSourceAssetSubIdentifier("mdl") or ""),
                "inputs": {
                    shader_input.GetBaseName(): str(shader_input.Get())
                    for shader_input in shader.GetInputs()
                },
            }
        )
    payload = json.dumps(shaders, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


original_bound_material_paths = get_bound_material_paths()
canonical_by_signature = {}
duplicate_to_canonical = {}
for material_path in sorted(original_bound_material_paths):
    signature = material_signature(stage.GetPrimAtPath(material_path))
    canonical = canonical_by_signature.setdefault(signature, material_path)
    if canonical != material_path:
        duplicate_to_canonical[material_path] = canonical

# The flattened source contains identical copies of library materials under
# many asset scopes. Retarget only exact material-binding relationship targets;
# geometry, UVs, and all distinct material parameter sets remain untouched.
for scene_prim in stage.Traverse():
    for relationship in scene_prim.GetRelationships():
        targets = relationship.GetTargets()
        remapped = [duplicate_to_canonical.get(target, target) for target in targets]
        if remapped != targets:
            relationship.SetTargets(remapped)

bound_material_paths = get_bound_material_paths()
unexpected_paths = bound_material_paths - set(canonical_by_signature.values())
if unexpected_paths:
    raise RuntimeError(f"Material binding deduplication failed: {sorted(unexpected_paths)}")

for duplicate_path in sorted(duplicate_to_canonical, reverse=True):
    stage.RemovePrim(duplicate_path)

materials = [
    stage.GetPrimAtPath(path)
    for path in sorted(bound_material_paths)
    if not has_preview_surface(stage.GetPrimAtPath(path))
]
materials = materials[args.start :]
if args.limit > 0:
    materials = materials[: args.limit]

failures: list[dict[str, str]] = []
for index, prim in enumerate(materials, start=1):
    bake_resolution = bake_resolution_for_material(prim)
    print(
        f"bake_start={args.start + index - 1} resolution={bake_resolution} "
        f"material={prim.GetPath()}",
        flush=True,
    )
    try:
        distiller = omni.mdl.distill_and_bake.MdlDistillAndBake(
            prim,
            ouput_folder=str(texture_dir),
            ouput_resolution=bake_resolution,
            ouput_samples=1,
            baking_to_new_material=False,
        )
        # distill() is a deprecated fire-and-forget wrapper. Waiting for the
        # coroutine is essential: otherwise the stage can be saved while the
        # MDL SDK is still authoring its PreviewSurface and texture assets.
        result = asyncio.get_event_loop().run_until_complete(distiller.distill_async())
        if result is False:
            failures.append({"material": str(prim.GetPath()), "error": "distill returned False"})
        elif not has_preview_surface(prim):
            failures.append({"material": str(prim.GetPath()), "error": "no UsdPreviewSurface was authored"})
    except Exception as error:  # preserve all material-level diagnostics
        failures.append({"material": str(prim.GetPath()), "error": str(error)})
    if index % 20 == 0 or index == len(materials):
        print(f"baked={index}/{len(materials)} failures={len(failures)}", flush=True)


def make_horizontal_atlas(
    texture_root: Path,
    source_stem: str,
    destination: Path,
    channel: str | None = None,
    linear_tint: tuple[float, float, float] | None = None,
) -> None:
    """Convert the ForkliftC eight-tile UDIM set to one portable 8K atlas.

    Unreal's USD PreviewSurface importer does not preserve MDL UDIM sampling.
    The prior single-tile bake therefore wrapped every tile onto 1001. Packing
    all eight exact source tiles horizontally keeps the same 1024px-per-tile
    build budget and works with a regular cooked Texture2D on PC and Mac.
    """
    tiles: list[Image.Image] = []
    for udim in range(1001, 1009):
        source = texture_root / f"{source_stem}.{udim}.png"
        if not source.is_file():
            raise FileNotFoundError(f"Required ForkliftC UDIM tile is missing: {source}")
        with Image.open(source) as opened:
            tile = opened.convert("RGB").resize(
                (FORKLIFT_TILE_RESOLUTION, FORKLIFT_TILE_RESOLUTION),
                Image.Resampling.LANCZOS,
            )
        if channel:
            tile = tile.getchannel(channel)
        elif linear_tint:
            channels = list(tile.split())
            for index, multiplier in enumerate(linear_tint):
                lookup = []
                for value in range(256):
                    encoded = value / 255.0
                    linear = encoded / 12.92 if encoded <= 0.04045 else ((encoded + 0.055) / 1.055) ** 2.4
                    tinted = max(0.0, min(1.0, linear * multiplier))
                    encoded_tinted = 12.92 * tinted if tinted <= 0.0031308 else 1.055 * (tinted ** (1.0 / 2.4)) - 0.055
                    lookup.append(round(encoded_tinted * 255.0))
                channels[index] = channels[index].point(lookup)
            tile = Image.merge("RGB", channels)
        tiles.append(tile)
    mode = tiles[0].mode
    atlas = Image.new(
        mode,
        (FORKLIFT_TILE_RESOLUTION * len(tiles), FORKLIFT_TILE_RESOLUTION),
    )
    for index, tile in enumerate(tiles):
        atlas.paste(tile, (FORKLIFT_TILE_RESOLUTION * index, 0))
    atlas.save(destination, format="PNG", optimize=True)


def set_preview_texture(material_prim: Usd.Prim, input_name: str, texture_path: Path) -> None:
    for shader_prim in Usd.PrimRange(material_prim):
        shader = UsdShade.Shader(shader_prim)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            raise RuntimeError(f"{material_prim.GetPath()} has no PreviewSurface {input_name} input")
        connected_sources, _invalid_sources = shader_input.GetConnectedSources()
        if len(connected_sources) != 1:
            raise RuntimeError(f"{material_prim.GetPath()} {input_name} has no single texture source")
        texture_shader = UsdShade.Shader(connected_sources[0].source.GetPrim())
        file_input = texture_shader.GetInput("file") if texture_shader else None
        if not file_input:
            raise RuntimeError(f"{material_prim.GetPath()} {input_name} texture has no file input")
        file_input.Set(Sdf.AssetPath(texture_path.as_posix()))
        return
    raise RuntimeError(f"{material_prim.GetPath()} has no UsdPreviewSurface shader")


def make_glass_portable(material_prim: Usd.Prim) -> None:
    for shader_prim in Usd.PrimRange(material_prim):
        shader = UsdShade.Shader(shader_prim)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        values = {
            "diffuseColor": Gf.Vec3f(0.055, 0.075, 0.095),
            "roughness": 0.16,
            "metallic": 0.0,
            "opacity": 0.28,
            "ior": 1.45,
        }
        for input_name, value in values.items():
            shader_input = shader.GetInput(input_name)
            if shader_input:
                shader_input.DisconnectSource()
                shader_input.Set(value)
        return


def repair_forklift_udims() -> dict[str, object]:
    texture_root = (
        payload_root
        / "omniverse-content-production.s3-us-west-2.amazonaws.com"
        / "Assets/Isaac/6.0/Isaac/Robots/IsaacSim/ForkliftC/Props/Source/Forklift_C/Textures"
    )
    atlas_root = output_dir / "ForkliftAtlases"
    atlas_root.mkdir()
    atlases = {
        "body": atlas_root / "T_Forklift_C01_Albedo_Atlas.png",
        "blue": atlas_root / "T_Forklift_C01_Albedo_Blue_Atlas.png",
        "normal": atlas_root / "T_Forklift_C01_Normal_Atlas.png",
        "roughness": atlas_root / "T_Forklift_C01_Roughness_Atlas.png",
        "metallic": atlas_root / "T_Forklift_C01_Metallic_Atlas.png",
    }
    make_horizontal_atlas(texture_root, "T_Forklift_C01_Albedo", atlases["body"])
    make_horizontal_atlas(
        texture_root,
        "T_Forklift_C01_Albedo",
        atlases["blue"],
        linear_tint=(0.039215688, 0.25490198, 0.91764706),
    )
    make_horizontal_atlas(texture_root, "T_Forklift_C01_Normal", atlases["normal"])
    make_horizontal_atlas(texture_root, "T_Forklift_C01_ORM", atlases["roughness"], channel="G")
    make_horizontal_atlas(texture_root, "T_Forklift_C01_ORM", atlases["metallic"], channel="B")

    material_prims: dict[str, Usd.Prim] = {}
    scaled_uv_meshes = 0
    for scene_prim in stage.Traverse():
        if "/Forklifts/Forklift" not in str(scene_prim.GetPath()):
            continue
        material, _relationship = UsdShade.MaterialBindingAPI(scene_prim).ComputeBoundMaterial()
        if not material:
            continue
        material_name = material.GetPrim().GetName()
        material_prims[material_name] = material.GetPrim()
        if material_name not in {"M_Forklift_C01", "M_Forklift_C01_Blue"}:
            continue
        primvar = UsdGeom.PrimvarsAPI(scene_prim).GetPrimvar("st")
        values = primvar.Get() if primvar else None
        if not values:
            continue
        primvar.Set([Gf.Vec2f(float(value[0]) / 8.0, float(value[1])) for value in values])
        scaled_uv_meshes += 1

    for material_name, base_atlas in (
        ("M_Forklift_C01", atlases["body"]),
        ("M_Forklift_C01_Blue", atlases["blue"]),
    ):
        material_prim = material_prims.get(material_name)
        if material_prim is None:
            raise RuntimeError(f"Required ForkliftC material was not bound: {material_name}")
        set_preview_texture(material_prim, "diffuseColor", base_atlas)
        set_preview_texture(material_prim, "normal", atlases["normal"])
        set_preview_texture(material_prim, "roughness", atlases["roughness"])
        set_preview_texture(material_prim, "metallic", atlases["metallic"])

    glass = material_prims.get("M_Forklift_C01_Glass")
    if glass is not None:
        make_glass_portable(glass)
    if scaled_uv_meshes < 20:
        raise RuntimeError(f"ForkliftC UDIM repair touched too few meshes: {scaled_uv_meshes}")
    return {
        "scaled_uv_meshes": scaled_uv_meshes,
        "atlas_files": [path.relative_to(output_dir).as_posix() for path in atlases.values()],
    }


forklift_udim_report = repair_forklift_udims()


def is_effectively_opaque(texture_path: Path) -> bool:
    """Reject compression noise without erasing authored decal cutouts.

    The MDL distiller sometimes bakes a nominally opaque material to an
    opacity JPEG.  JPEG ringing makes that map range below 255, so a simple
    min/max test is too strict and Unreal selects its translucent master
    material.  Requiring both a near-white mean and fewer than 0.1% visibly
    non-white pixels distinguishes that artifact from the real conveyor and
    forklift decal masks in this scene.
    """
    with Image.open(texture_path) as source:
        histogram = source.convert("L").histogram()
    pixel_count = sum(histogram)
    if pixel_count == 0:
        return False
    mean = sum(value * count for value, count in enumerate(histogram)) / pixel_count
    non_white_fraction = sum(histogram[:250]) / pixel_count
    return mean >= 254.5 and non_white_fraction <= 0.001


# USD Preview Surface considers any connected opacity input translucent.
# Disconnect only opacity textures proven to be effectively solid white.
# This preserves the genuinely varying conveyor/forklift decal masks while
# keeping painted forklift bodywork on Unreal's faster opaque render path.
removed_opaque_texture_inputs: list[dict[str, str]] = []
for material_path in sorted(bound_material_paths):
    material_prim = stage.GetPrimAtPath(material_path)
    for shader_prim in Usd.PrimRange(material_prim):
        shader = UsdShade.Shader(shader_prim)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        opacity_input = shader.GetInput("opacity")
        if not opacity_input:
            continue
        connected_sources, _invalid_sources = opacity_input.GetConnectedSources()
        if len(connected_sources) != 1:
            continue
        texture_shader = UsdShade.Shader(connected_sources[0].source.GetPrim())
        file_input = texture_shader.GetInput("file") if texture_shader else None
        file_asset = file_input.Get() if file_input else None
        if not isinstance(file_asset, Sdf.AssetPath) or not file_asset.path:
            continue
        resolved = Path(file_asset.resolvedPath or file_asset.path.removeprefix("file:"))
        if not resolved.is_file() or not is_effectively_opaque(resolved):
            continue
        opacity_input.DisconnectSource()
        opacity_input.Set(1.0)
        removed_opaque_texture_inputs.append(
            {"material": str(material_path), "texture": resolved.as_posix()}
        )

stage.GetRootLayer().Save()
kit.update()

# Distillation commonly emits identical constant maps into each material's
# private folder. Share byte-identical maps only within the same semantic role
# (e.g. roughness with roughness, never roughness with base color) so Unreal
# imports one texture without changing color-space or compression semantics.
canonical_texture_by_key: dict[tuple[str, str], Path] = {}
duplicate_texture_to_canonical: dict[Path, Path] = {}
for texture_path in sorted(texture_dir.rglob("*.jpg")):
    key = (texture_path.name, file_sha256(texture_path))
    canonical = canonical_texture_by_key.setdefault(key, texture_path)
    if canonical != texture_path:
        duplicate_texture_to_canonical[texture_path.resolve()] = canonical.resolve()

for scene_prim in stage.Traverse():
    for attribute in scene_prim.GetAttributes():
        value = attribute.Get()
        if not isinstance(value, Sdf.AssetPath) or not value.path:
            continue
        resolved_value = value.resolvedPath or value.path.removeprefix("file:")
        resolved_path = Path(resolved_value).resolve()
        canonical = duplicate_texture_to_canonical.get(resolved_path)
        if canonical:
            attribute.Set(Sdf.AssetPath(canonical.as_posix()))

stage.GetRootLayer().Save()
deduplicated_texture_bytes = sum(path.stat().st_size for path in duplicate_texture_to_canonical)
for duplicate_texture in duplicate_texture_to_canonical:
    duplicate_texture.unlink()

preview_materials = 0
remaining_mdl_materials = 0
for material_path in bound_material_paths:
    prim = stage.GetPrimAtPath(material_path)
    has_preview = False
    has_mdl = False
    for child in Usd.PrimRange(prim):
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        has_preview |= shader.GetIdAttr().Get() == "UsdPreviewSurface"
        has_mdl |= bool(shader.GetSourceAsset("mdl"))
    preview_materials += int(has_preview)
    remaining_mdl_materials += int(has_mdl and not has_preview)

files = sorted(path for path in output_dir.rglob("*") if path.is_file())
manifest = {
    "schema_version": 1,
    "source_stage": str(source_stage),
    "output_stage": str(output_stage),
    "source_bound_materials": len(original_bound_material_paths),
    "bound_materials": len(bound_material_paths),
    "deduplicated_materials": len(duplicate_to_canonical),
    "deduplicated_texture_files": len(duplicate_texture_to_canonical),
    "deduplicated_texture_bytes": deduplicated_texture_bytes,
    "removed_effectively_opaque_texture_inputs": removed_opaque_texture_inputs,
    "forklift_udim_atlas": forklift_udim_report,
    "supplemental_mdl_assets": sorted(supplemental_assets.values(), key=lambda entry: str(entry["path"])),
    "attempted_materials": len(materials),
    "preview_materials": preview_materials,
    "remaining_mdl_materials": remaining_mdl_materials,
    "failures": failures,
    "files": [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ],
}
manifest_path = output_dir / "baked_stage_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"QAI_CONVEYOR_BAKE={json.dumps(manifest, separators=(',', ':'))}", flush=True)
kit.close()

if failures or (args.limit == 0 and remaining_mdl_materials):
    raise RuntimeError(
        f"MDL bake incomplete: failures={len(failures)} remaining_mdl={remaining_mdl_materials}"
    )
