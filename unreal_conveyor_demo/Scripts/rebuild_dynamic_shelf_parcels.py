"""Split baked rack cartons into tagged movable actors using authored USD poses."""

from __future__ import annotations

import re
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
SOURCE = (
    Path(unreal.Paths.project_content_dir())
    / "OmniversePayload"
    / "reason2_conveyor_safety_stable_cargo_r38.usda"
)
RACK_MESH = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Storage/SM_WestRack"
)
BOX_MESH = (
    "/Game/ConveyorRuntime/Parcels/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Conveyor/Parcels/Parcel1/Asset/SM_CardBoxD_04"
)
BOX_MATERIAL = (
    "/Game/ConveyorRuntime/VisualFinish/MI_CardboardFiber_D"
)
HIDDEN_MATERIAL = "/Game/ConveyorRuntime/VisualFinish/M_HiddenBakedRackCartons"
ACTOR_PREFIX = "QaiShelfParcel_"


def load_required(path: str, expected: type) -> object:
    value = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(value, expected):
        raise RuntimeError(f"Missing required {expected.__name__}: {path}")
    return value


def balanced_block(text: str, opening_brace: int) -> str:
    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace : index + 1]
    raise RuntimeError("Unbalanced USD block")


def parse_shelf_parcels() -> list[dict[str, object]]:
    text = SOURCE.read_text(encoding="utf-8")
    storage_start = text.index('def "Storage"')
    storage = balanced_block(text, text.index("{", storage_start))
    parcels: list[dict[str, object]] = []
    pattern = re.compile(r'def Xform "(Level[123]Carton[1234])"[^\{]*\{')
    for match in pattern.finditer(storage):
        block = balanced_block(storage, match.end() - 1)
        header = block.split('def Xform "Asset"', 1)[0]
        dimensions_match = re.search(r"codex:dimensionsM = \(([^)]+)\)", header)
        translation_match = re.search(r"xformOp:translate = \(([^)]+)\)", header)
        rotation_match = re.search(r"xformOp:rotateXYZ = \(([^)]+)\)", header)
        if not dimensions_match or not translation_match or not rotation_match:
            raise RuntimeError(f"Incomplete authored shelf parcel: {match.group(1)}")
        dimensions = tuple(float(value.strip()) for value in dimensions_match.group(1).split(","))
        translation = tuple(float(value.strip()) for value in translation_match.group(1).split(","))
        rotation = tuple(float(value.strip()) for value in rotation_match.group(1).split(","))
        parcels.append(
            {
                "name": match.group(1),
                "dimensions": dimensions,
                "translation": translation,
                "yaw": rotation[2],
            }
        )
    if len(parcels) != 24:
        raise RuntimeError(f"Expected 24 authored shelf parcels, found {len(parcels)}")
    return parcels


def hidden_material() -> unreal.Material:
    existing = unreal.EditorAssetLibrary.load_asset(HIDDEN_MATERIAL)
    if isinstance(existing, unreal.Material):
        return existing
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        HIDDEN_MATERIAL.rsplit("/", 1)[-1],
        HIDDEN_MATERIAL.rsplit("/", 1)[0],
        unreal.Material,
        unreal.MaterialFactoryNew(),
    )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("Could not create the baked-carton suppression material")
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    zero = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -200, 0
    )
    zero.set_editor_property("r", 0.0)
    unreal.MaterialEditingLibrary.connect_material_property(
        zero, "", unreal.MaterialProperty.MP_OPACITY_MASK
    )
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def main() -> None:
    parcels = parse_shelf_parcels()
    unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in list(subsystem.get_all_level_actors()):
        if actor.get_actor_label().startswith(ACTOR_PREFIX):
            subsystem.destroy_actor(actor)

    rack = load_required(RACK_MESH, unreal.StaticMesh)
    box = load_required(BOX_MESH, unreal.StaticMesh)
    cardboard = load_required(BOX_MATERIAL, unreal.MaterialInterface)
    hidden = hidden_material()
    rack.set_material(4, hidden)
    unreal.EditorAssetLibrary.save_loaded_asset(rack, only_if_is_dirty=False)
    for actor in subsystem.get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            if component.get_editor_property("static_mesh") == rack:
                component.set_material(4, hidden)

    for index, row in enumerate(parcels, start=1):
        x, y, z = row["translation"]
        dx, dy, dz = row["dimensions"]
        actor = subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor,
            unreal.Vector(x * 100.0, y * 100.0, z * 100.0),
            unreal.Rotator(roll=0.0, pitch=0.0, yaw=float(row["yaw"])),
        )
        actor.set_actor_label(f"{ACTOR_PREFIX}{index:02d}_{row['name']}")
        actor.set_editor_property("tags", [unreal.Name("Qai.ShelfParcel")])
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_static_mesh(box)
        component.set_material(0, cardboard)
        # The reused USD static mesh retains its centimetre conversion at the
        # original scene-component level. A directly spawned actor therefore
        # needs the same 0.01 assembly scale in addition to the authored box
        # variation scale.
        component.set_world_scale3d(
            unreal.Vector(0.01 * dx / 0.4, 0.01 * dy / 0.26, 0.01 * dz / 0.16)
        )
        component.set_editor_property(
            "component_tags",
            [unreal.Name("Qai.ShelfParcel"), unreal.Name(f"Qai.ShelfParcel.{index}")],
        )
        component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)

    unreal.EditorLoadingAndSavingUtils.save_current_level()
    unreal.log(f"QAI_DYNAMIC_SHELF_PARCELS=24 source={SOURCE}")


main()
