"""Place the optimized IQ9 EVK proxy on the sorting-station desktop."""

from __future__ import annotations

import json
import math
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
PROXY_MESH = "/Game/IQ9EVK/Runtime/SM_SM_IQ9_EVK_Core_Proxy"
LED_MATERIAL = "/Game/IQ9EVK/Runtime/M_IQ9_LED"
PACKING_TABLE_MESH = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/SM_packing_table"
)
ACTOR_LABEL = "IQ9 EVK tabletop"

# Bounds measured from the locally tessellated STEP source, in centimetres.
PROXY_LOCAL_CENTER = unreal.Vector(-4.94, 5.10, -1.25)
PROXY_LOCAL_EXTENT = unreal.Vector(5.36, 5.40, 2.85)


def load_required(path: str, expected_type: type):
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(asset, expected_type):
        raise RuntimeError(f"Missing {expected_type.__name__}: {path}")
    return asset


def create_led_material() -> unreal.Material:
    library = unreal.MaterialEditingLibrary
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    material = unreal.EditorAssetLibrary.load_asset(LED_MATERIAL)
    if not isinstance(material, unreal.Material):
        material = asset_tools.create_asset(
            "M_IQ9_LED",
            "/Game/IQ9EVK/Runtime",
            unreal.Material,
            unreal.MaterialFactoryNew(),
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("Could not create IQ9 LED material")

    library.delete_all_material_expressions(material)
    color = library.create_material_expression(
        material, unreal.MaterialExpressionVectorParameter, -460, -70)
    color.set_editor_property("parameter_name", "LedColor")
    color.set_editor_property("default_value", unreal.LinearColor(0.0, 1.0, 0.08, 1.0))
    strength = library.create_material_expression(
        material, unreal.MaterialExpressionScalarParameter, -460, 90)
    strength.set_editor_property("parameter_name", "LedStrength")
    strength.set_editor_property("default_value", 0.0)
    multiply = library.create_material_expression(
        material, unreal.MaterialExpressionMultiply, -120, -20)
    library.connect_material_expressions(color, "RGB", multiply, "A")
    library.connect_material_expressions(strength, "", multiply, "B")
    library.connect_material_property(multiply, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    library.layout_material_expressions(material)
    library.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def rotate_xy(vector: unreal.Vector, yaw_degrees: float) -> unreal.Vector:
    angle = math.radians(yaw_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return unreal.Vector(
        vector.x * cosine - vector.y * sine,
        vector.x * sine + vector.y * cosine,
        vector.z,
    )


def main() -> None:
    unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
    proxy_mesh = load_required(PROXY_MESH, unreal.StaticMesh)
    table_mesh = load_required(PACKING_TABLE_MESH, unreal.StaticMesh)
    create_led_material()

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = actor_subsystem.get_all_level_actors()
    for actor in actors:
        if actor.get_actor_label() == ACTOR_LABEL:
            actor_subsystem.destroy_actor(actor)

    table_component = None
    for actor in actors:
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            if component.static_mesh == table_mesh:
                table_component = component
                break
        if table_component:
            break
    if table_component is None:
        raise RuntimeError("Could not find the sorting-station desktop component")

    table_origin, table_extent, _ = unreal.SystemLibrary.get_component_bounds(table_component)
    table_rotation = table_component.get_world_rotation()
    # Put the EVK slightly toward the front-right of the worktop, clear of
    # the authored sorting trays.  It remains reachable by the forklifts.
    tabletop_offset = rotate_xy(unreal.Vector(15.0, -10.0, 0.0), table_rotation.yaw)
    desired_center = unreal.Vector(
        table_origin.x + tabletop_offset.x,
        table_origin.y + tabletop_offset.y,
        table_origin.z + table_extent.z + PROXY_LOCAL_EXTENT.z + 0.25,
    )
    prop_yaw = table_rotation.yaw
    rotated_local_center = rotate_xy(PROXY_LOCAL_CENTER, prop_yaw)
    actor_location = unreal.Vector(
        desired_center.x - rotated_local_center.x,
        desired_center.y - rotated_local_center.y,
        desired_center.z - PROXY_LOCAL_CENTER.z,
    )

    actor = actor_subsystem.spawn_actor_from_object(
        proxy_mesh, actor_location, unreal.Rotator(0.0, prop_yaw, 0.0))
    if actor is None:
        raise RuntimeError("Could not spawn optimized IQ9 EVK")
    actor.set_actor_label(ACTOR_LABEL)
    actor.set_editor_property("tags", [unreal.Name("Qai.IQ9EVK.Actor")])
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if component is None:
        raise RuntimeError("IQ9 EVK proxy actor has no static mesh component")
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_collision_profile_name("NoCollision")
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_editor_property("component_tags", [
        unreal.Name("Qai.IQ9EVK"),
        unreal.Name("Qai.DynamicProp"),
    ])
    component.set_editor_property("cast_shadow", True)

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.save_current_level():
        raise RuntimeError("Could not save WarehouseConveyor after IQ9 placement")
    unreal.EditorAssetLibrary.save_directory("/Game/IQ9EVK/Runtime", only_if_is_dirty=False, recursive=True)

    static_mesh_subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    report = {
        "source": "resources/DP25-73418-2_RevB/10-75699-RB8-PVT.stp",
        "confidential_local_derivative": True,
        "proxy_mesh": proxy_mesh.get_path_name(),
        "vertices_lod0": static_mesh_subsystem.get_number_verts(proxy_mesh, 0),
        "source_instance_count": 5794,
        "source_size_cm": [10.72, 10.80, 5.70],
        "configuration": "open_top_core_only",
        "omitted": ["top_lid", "external_cables", "external_peripherals"],
        "texture_bakes": ["base_color_2k", "normal_2k", "roughness_2k"],
        "actor": ACTOR_LABEL,
        "location": list(actor_location.to_tuple()),
        "rotation": [0.0, prop_yaw, 0.0],
        "table_bounds": {
            "origin": list(table_origin.to_tuple()),
            "extent": list(table_extent.to_tuple()),
            "top_z": table_origin.z + table_extent.z,
        },
        "physics_tag": "Qai.DynamicProp",
        "led_material": LED_MATERIAL,
    }
    output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "iq9_evk.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_IQ9_EVK={output}")


if __name__ == "__main__":
    main()
