"""Replace the collapsed packing table with its exact authored sub-meshes.

The original optimized import collapsed the complete Omniverse packing-table
assembly into two Unreal sections. That made the metal table, colored plastic
crates, cardboard bodies, and label decals share material overrides. This
targeted import keeps the rest of the warehouse optimized while preserving the
24 authored packing-table mesh/material bindings.
"""

from __future__ import annotations

import json
from pathlib import Path

import unreal


PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())).resolve()
STAGE_PATH = PROJECT_DIR / "Saved" / "BakedStageGenerated" / "warehouse_conveyor_baked.usdc"
LEVEL_PATH = "/Game/ConveyorRuntime/WarehouseConveyor"
DESTINATION = "/Game/ConveyorRuntime/PackingTableDetailed"
PRIM_PATH = "/World/CodexPoC/ConveyorSafety/packing_table"
COLLAPSED_MESH = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/SM_packing_table"
)
AUDIT_PATH = PROJECT_DIR / "Saved" / "ImportAudit" / "detailed_packing_table.json"
MATERIAL_FOLDER = f"{DESTINATION}/Materials"
SCENE_LOOKS = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Conveyor/Modules/ExtensionEast/Looks"
)
TABLE_MATERIAL = "/Game/ConveyorRuntime/VisualFinish/MI_BrushedSortingTable"
YELLOW_MATERIAL = f"{SCENE_LOOKS}/MI_Plastic_Yellow_A"
BLUE_MATERIAL = f"{SCENE_LOOKS}/MI_Plastic_Blue_A"
BROWN_CARDBOARD_MATERIAL = "/Game/ConveyorRuntime/VisualFinish/MI_CardboardFiber_A"
WHITE_CARDBOARD_MATERIAL = "/Game/ConveyorRuntime/VisualFinish/MI_CardboardFiber_D"
PAYLOAD_ROOT = (
    PROJECT_DIR
    / "Content"
    / "OmniversePayload"
    / "omniverse-content-production.s3-us-west-2.amazonaws.com"
    / "Assets"
    / "Isaac"
    / "6.0"
    / "Isaac"
    / "Props"
    / "PackingTable"
)
LABEL_TEXTURES = PAYLOAD_ROOT / "materials" / "textures"


def set_property(obj: object, name: str, value: object) -> None:
    obj.set_editor_property(name, value)  # type: ignore[attr-defined]


def load_material(path: str) -> unreal.MaterialInterface:
    material = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(material, unreal.MaterialInterface):
        raise RuntimeError(f"Required packing-table material is missing: {path}")
    return material


def import_texture(asset_name: str, filename: Path) -> unreal.Texture:
    if not filename.is_file():
        raise RuntimeError(f"Required Omniverse packing-table texture is missing: {filename}")
    asset_path = f"{MATERIAL_FOLDER}/{asset_name}"
    existing = unreal.EditorAssetLibrary.load_asset(asset_path)
    if isinstance(existing, unreal.Texture):
        return existing
    task = unreal.AssetImportTask()
    set_property(task, "automated", True)
    set_property(task, "save", False)
    set_property(task, "replace_existing", True)
    set_property(task, "filename", str(filename))
    set_property(task, "destination_path", MATERIAL_FOLDER)
    set_property(task, "destination_name", asset_name)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(imported, unreal.Texture):
        raise RuntimeError(f"Could not import {filename} as {asset_path}")
    return imported


def expression(material: unreal.Material, cls: type, x: int, y: int) -> object:
    value = unreal.MaterialEditingLibrary.create_material_expression(
        material, cls.static_class(), x, y
    )
    if value is None:
        raise RuntimeError(f"Could not create {cls.__name__} for {material.get_path_name()}")
    return value


def connect_output(source: object, output: str, prop: unreal.MaterialProperty) -> None:
    if not unreal.MaterialEditingLibrary.connect_material_property(source, output, prop):
        raise RuntimeError(f"Could not connect packing-table material output {prop}")


def create_masked_artwork_material(
    name: str, albedo_file: str, opacity_file: str
) -> unreal.Material:
    asset_path = f"{MATERIAL_FOLDER}/{name}"
    material = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"Could not create {asset_path}")
    albedo = import_texture(f"T_{name}_Albedo", LABEL_TEXTURES / albedo_file)
    opacity = import_texture(f"T_{name}_Opacity", LABEL_TEXTURES / opacity_file)
    set_property(material, "blend_mode", unreal.BlendMode.BLEND_MASKED)
    set_property(material, "two_sided", True)
    set_property(material, "opacity_mask_clip_value", 0.16)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
    base = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -480, -120)
    set_property(base, "parameter_name", "ArtworkColor")
    set_property(base, "texture", albedo)
    set_property(base, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    connect_output(base, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    mask = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -480, 80)
    set_property(mask, "parameter_name", "ArtworkMask")
    set_property(mask, "texture", opacity)
    # PNG import initially classifies these authored masks as color textures.
    # Sampling the red channel as color is lossless for the grayscale artwork
    # and avoids a platform shader mismatch before the texture is recooked.
    set_property(mask, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    connect_output(mask, "R", unreal.MaterialProperty.MP_OPACITY_MASK)
    roughness = expression(material, unreal.MaterialExpressionConstant, -220, 260)
    set_property(roughness, "r", 0.62)
    connect_output(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    unreal.MaterialEditingLibrary.layout_material_expressions(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    return material


level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if not level_subsystem.load_level(LEVEL_PATH):
    raise RuntimeError(f"Could not load {LEVEL_PATH}")

# Re-running the script must not accumulate actors or assets.
for actor in actor_subsystem.get_all_level_actors():
    if "Qai.DetailedPackingTable" in [str(tag) for tag in actor.get_editor_property("tags")]:
        actor_subsystem.destroy_actor(actor)
if unreal.EditorAssetLibrary.does_directory_exist(DESTINATION):
    if not unreal.EditorAssetLibrary.delete_directory(DESTINATION):
        raise RuntimeError(f"Could not replace {DESTINATION}")

collapsed_mesh = unreal.EditorAssetLibrary.load_asset(COLLAPSED_MESH)
if not isinstance(collapsed_mesh, unreal.StaticMesh):
    raise RuntimeError(f"Missing collapsed packing table {COLLAPSED_MESH}")
collapsed_components: list[unreal.StaticMeshComponent] = []
for actor in actor_subsystem.get_all_level_actors():
    for component in actor.get_components_by_class(unreal.StaticMeshComponent):
        if component.get_editor_property("static_mesh") == collapsed_mesh:
            collapsed_components.append(component)
if len(collapsed_components) != 1:
    raise RuntimeError(f"Expected one collapsed packing table, found {len(collapsed_components)}")

before_actor_paths = {actor.get_path_name() for actor in actor_subsystem.get_all_level_actors()}
options = unreal.UsdStageImportOptions()
settings = {
    "import_actors": True,
    "import_geometry": True,
    "import_skeletal_animations": False,
    "import_level_sequences": False,
    "import_materials": True,
    "import_groom_assets": False,
    "import_sparse_volume_textures": False,
    "import_sounds": False,
    "import_only_used_materials": True,
    "prims_to_import": [PRIM_PATH],
    "purposes_to_import": 7,
    "nanite_triangle_threshold": 50000,
    "render_context_to_import": "universal",
    "material_purpose": "allPurpose",
    "fallback_collision_type": unreal.UsdCollisionType.NONE,
    "subdivision_level": 0,
    "import_at_specific_time_code": False,
    "share_assets_for_identical_prims": True,
    "existing_actor_policy": unreal.ReplaceActorPolicy.APPEND,
    "existing_asset_policy": unreal.ReplaceAssetPolicy.REPLACE,
    "prim_path_folder_structure": True,
    "use_prim_kinds_for_collapsing": False,
    "merge_identical_material_slots": False,
    "interpret_lods": True,
}
for name, value in settings.items():
    set_property(options, name, value)

task = unreal.AssetImportTask()
set_property(task, "automated", True)
set_property(task, "save", False)
set_property(task, "replace_existing", True)
set_property(task, "replace_existing_settings", True)
set_property(task, "filename", str(STAGE_PATH))
set_property(task, "destination_path", DESTINATION)
set_property(task, "options", options)
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
imported_paths = list(task.get_editor_property("imported_object_paths"))
if not imported_paths:
    raise RuntimeError("USD importer returned no detailed packing-table assets")

new_actors = [
    actor
    for actor in actor_subsystem.get_all_level_actors()
    if actor.get_path_name() not in before_actor_paths
]
if not new_actors:
    raise RuntimeError("Detailed packing-table import created no level actor")
for actor in new_actors:
    tags = list(actor.get_editor_property("tags"))
    tags.append("Qai.DetailedPackingTable")
    set_property(actor, "tags", tags)

# Retain the authored low-cost PackingTable collision proxy elsewhere in the
# scene, but hide and disable the visually collapsed table itself.
for component in collapsed_components:
    component.set_visibility(False, True)
    component.set_hidden_in_game(True)
    component.set_collision_profile_name("NoCollision")
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)

table_material = load_material(TABLE_MATERIAL)
yellow_material = load_material(YELLOW_MATERIAL)
blue_material = load_material(BLUE_MATERIAL)
brown_cardboard = load_material(BROWN_CARDBOARD_MATERIAL)
white_cardboard = load_material(WHITE_CARDBOARD_MATERIAL)
label_material = create_masked_artwork_material(
    "M_CorrugatedBoxLabels",
    "t_corrugatedboxes_b01_decal_albedo.png",
    "t_corrugatedboxes_b01_decal_opacity.png",
)
trim_material = create_masked_artwork_material(
    "M_CorrugatedBoxTrim",
    "t_corrugatedboxes_b01_trim_Albedo.png",
    "t_corrugatedboxes_b01_trim_opacity.png",
)

components: list[dict[str, object]] = []
category_counts = {"metal_table": 0, "yellow_plastic": 0, "blue_plastic": 0, "cardboard": 0, "label_decal": 0, "other": 0}
for actor in new_actors:
    for component in actor.get_components_by_class(unreal.StaticMeshComponent):
        component.set_collision_profile_name("NoCollision")
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        actor_name = actor.get_actor_label().lower()
        component_name = component.get_name().lower()
        role_name = f"{actor_name} {component_name}"
        category = "other"
        assigned_material: unreal.MaterialInterface | None = None
        if actor_name == "packing_table" and component_name.startswith("cube"):
            # These are the authored collision proxies, not visible containers.
            component.set_visibility(False, True)
            component.set_hidden_in_game(True)
        elif "heavydutypackingtable" in role_name:
            category = "metal_table"
            assigned_material = table_material
        elif "crate_a07" in role_name:
            category = "yellow_plastic"
            assigned_material = yellow_material
        elif "crate_a08" in role_name:
            category = "blue_plastic"
            assigned_material = blue_material
        elif "trans__decal__trim" in component_name:
            category = "label_decal"
            assigned_material = trim_material
        elif "trans__decal__" in component_name:
            category = "label_decal"
            assigned_material = label_material
        elif "corrugatedbox" in role_name and "body" in component_name:
            category = "cardboard"
            assigned_material = brown_cardboard if "brown" in actor_name else white_cardboard
        if assigned_material is not None:
            for index in range(max(1, component.get_num_materials())):
                component.set_material(index, assigned_material)
        materials = [
            component.get_material(index).get_path_name() if component.get_material(index) else ""
            for index in range(component.get_num_materials())
        ]
        category_counts[category] += 1
        components.append(
            {
                "actor": actor.get_actor_label(),
                "component": component.get_name(),
                "mesh": component.get_editor_property("static_mesh").get_path_name(),
                "materials": materials,
                "category": category,
            }
        )

required = ("metal_table", "yellow_plastic", "blue_plastic", "cardboard", "label_decal")
missing = [name for name in required if category_counts[name] == 0]
report = {
    "stage": str(STAGE_PATH),
    "prim": PRIM_PATH,
    "imported_assets": len(imported_paths),
    "imported_object_paths": imported_paths,
    "new_actors": [actor.get_actor_label() for actor in new_actors],
    "collapsed_components_hidden": [component.get_name() for component in collapsed_components],
    "category_counts": category_counts,
    "components": components,
}
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
AUDIT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
unreal.log(f"QAI_DETAILED_PACKING_TABLE={json.dumps(report, separators=(',', ':'))}")
if missing:
    raise RuntimeError(f"Detailed packing table is missing authored categories: {missing}; {category_counts}")

unreal.EditorAssetLibrary.save_directory(DESTINATION, only_if_is_dirty=False, recursive=True)
if not level_subsystem.save_current_level():
    raise RuntimeError(f"Could not save {LEVEL_PATH}")
