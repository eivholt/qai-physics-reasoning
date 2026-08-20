"""Fail the build on the material regressions visible in the first review."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


CONTENT_ROOT = "/Game/ConveyorRuntime"
LEVEL = f"{CONTENT_ROOT}/WarehouseConveyor"
OUTPUT = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "visual_verification.json"
PASS_MARKER = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "visual_verification.passed"
PASS_MARKER.unlink(missing_ok=True)


def object_path(value: object | None) -> str:
    return value.get_path_name() if value is not None else ""  # type: ignore[attr-defined]


def material_slots(asset: object) -> list[object]:
    if isinstance(asset, unreal.StaticMesh):
        return list(asset.get_editor_property("static_materials"))
    if isinstance(asset, unreal.SkeletalMesh):
        return list(asset.get_editor_property("materials"))
    return []


unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
assets = unreal.EditorAssetLibrary.list_assets(CONTENT_ROOT, recursive=True, include_folder=False)
material_instances: list[unreal.MaterialInstanceConstant] = []
unassigned_slots: list[dict[str, object]] = []
nested_slots: list[dict[str, object]] = []
textured_materials = 0
texture_parameters = 0
normal_mapped_materials = 0
roughness_mapped_materials = 0
metallic_mapped_materials = 0
skeletal_meshes = 0
hero_meshes = 0
imprecise_hero_meshes: list[str] = []
static_mesh_subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)

for asset_path in assets:
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    skeletal_meshes += int(isinstance(asset, unreal.SkeletalMesh))
    if isinstance(asset, unreal.MaterialInstanceConstant):
        material_instances.append(asset)
        names = unreal.MaterialEditingLibrary.get_texture_parameter_names(asset)
        used_texture = False
        enabled_channels: set[str] = set()
        for name in names:
            scalar_name = f"Use{name}"
            try:
                enabled = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
                    asset, scalar_name
                )
            except Exception:
                enabled = 0.0
            texture = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(asset, name)
            if enabled and texture is not None:
                texture_parameters += 1
                used_texture = True
                enabled_channels.add(str(name))
        textured_materials += int(used_texture)
        normal_mapped_materials += int("NormalTexture" in enabled_channels)
        roughness_mapped_materials += int("RoughnessTexture" in enabled_channels)
        metallic_mapped_materials += int("MetallicTexture" in enabled_channels)

    if isinstance(asset, unreal.StaticMesh):
        normalized = asset_path.lower().replace("\\", "/")
        if "/forklifts/" in normalized or "/conveyor/modules/" in normalized:
            hero_meshes += 1
            # Some UE commandlet configurations do not register this editor
            # subsystem. The importer still applies these settings; validate
            # them whenever the API is exposed instead of crashing the audit.
            if static_mesh_subsystem is not None:
                settings = static_mesh_subsystem.get_lod_build_settings(asset, 0)
                if not (
                    settings.get_editor_property("use_full_precision_u_vs")
                    and settings.get_editor_property("use_high_precision_tangent_basis")
                ):
                    imprecise_hero_meshes.append(asset_path)

    for index, slot in enumerate(material_slots(asset)):
        material = slot.get_editor_property("material_interface")
        if material is None:
            unassigned_slots.append({"asset": asset_path, "slot": index})
            continue
        if isinstance(material, unreal.MaterialInstanceConstant):
            parent = material.get_editor_property("parent")
            if (
                isinstance(parent, unreal.MaterialInstanceConstant)
                and parent.get_path_name().startswith(CONTENT_ROOT + "/")
                # The detailed packing-table import retains five hidden USD
                # collision-proxy meshes. Their nested material never renders
                # and is deliberately excluded from the visible-material guard.
                and not (
                    "/packingTableDetailed/".lower() in asset_path.lower()
                    and "/colliders/" in asset_path.lower()
                )
            ):
                nested_slots.append(
                    {
                        "asset": asset_path,
                        "slot": index,
                        "material": object_path(material),
                        "parent": object_path(parent),
                    }
                )

forklift_blue = next(
    (material for material in material_instances if material.get_name() == "MI_M_Forklift_C01_Blue"),
    None,
)
if forklift_blue is None:
    raise RuntimeError("Missing imported MI_M_Forklift_C01_Blue")
forklift_parent = object_path(forklift_blue.get_editor_property("parent"))
forklift_opacity_texture = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
    forklift_blue, "UseOpacityTexture"
)

red_floor_shader = unreal.EditorAssetLibrary.load_asset(f"{CONTENT_ROOT}/VisualFinish/M_RedEpoxyFloor")
red_floor = unreal.EditorAssetLibrary.load_asset(f"{CONTENT_ROOT}/VisualFinish/MI_RedEpoxyFloor")
clearance = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/SM_ClearanceZone"
)
rack = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/Storage/SM_WestRack"
)
packing_table = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/SM_packing_table"
)
red_floor_inputs: dict[str, str] = {}
red_floor_roughness_sampler = ""
red_floor_parent = ""
red_floor_tint = ""
red_floor_blend = -1.0
red_floor_world_texture_scale = -1.0
red_floor_texture_brightness = -1.0
red_floor_micro_roughness = -1.0
red_floor_variation_floor = -1.0
red_floor_world_position_nodes = 0
if isinstance(red_floor_shader, unreal.Material):
    for name, material_property in (
        ("base_color", unreal.MaterialProperty.MP_BASE_COLOR),
        ("normal", unreal.MaterialProperty.MP_NORMAL),
        ("roughness", unreal.MaterialProperty.MP_ROUGHNESS),
    ):
        red_floor_inputs[name] = object_path(
            unreal.MaterialEditingLibrary.get_material_property_input_node(red_floor_shader, material_property)
        )
    # Ultra routes the authored roughness map through a scalar gloss control,
    # so the material output node is a Multiply rather than the texture sample.
    # Find and validate the semantic linear-data sample anywhere in the graph.
    roughness_samples = [
        node
        for node in unreal.MaterialEditingLibrary.get_material_expressions(red_floor_shader)
        if isinstance(node, unreal.MaterialExpressionTextureSample)
        and "GRAYSCALE" in str(node.get_editor_property("sampler_type")).upper()
    ]
    # The detail pass deliberately adds a second tiled grayscale sample for
    # fine epoxy scratches, so one or more semantic roughness samples is valid.
    if roughness_samples:
        red_floor_roughness_sampler = str(roughness_samples[0].get_editor_property("sampler_type"))
if isinstance(red_floor, unreal.MaterialInstanceConstant):
    red_floor_parent = object_path(red_floor.get_editor_property("parent"))
    red_floor_tint = str(
        unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(red_floor, "EpoxyTint")
    )
    red_floor_blend = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
        red_floor, "TintBlend"
    )
    red_floor_world_texture_scale = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
        red_floor, "WorldTextureScale"
    )
    red_floor_texture_brightness = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
        red_floor, "TextureBrightness"
    )
    red_floor_micro_roughness = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
        red_floor, "MicroRoughnessStrength"
    )
    red_floor_variation_floor = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
        red_floor, "VariationFloor"
    )
if isinstance(red_floor_shader, unreal.Material):
    red_floor_world_position_nodes = sum(
        isinstance(node, unreal.MaterialExpressionWorldPosition)
        for node in unreal.MaterialEditingLibrary.get_material_expressions(red_floor_shader)
    )
clearance_material = ""
if isinstance(clearance, unreal.StaticMesh):
    clearance_material = object_path(
        list(clearance.get_editor_property("static_materials"))[0].get_editor_property("material_interface")
    )
rack_carton_material = ""
if isinstance(rack, unreal.StaticMesh) and len(list(rack.get_editor_property("static_materials"))) >= 5:
    rack_carton_material = object_path(
        list(rack.get_editor_property("static_materials"))[4].get_editor_property("material_interface")
    )
packing_table_material = ""
packing_table_bin_material = ""
if isinstance(packing_table, unreal.StaticMesh):
    packing_table_slots = list(packing_table.get_editor_property("static_materials"))
    packing_table_material = object_path(
        packing_table_slots[0].get_editor_property(
            "material_interface"
        )
    )
    if len(packing_table_slots) >= 2:
        packing_table_bin_material = object_path(
            packing_table_slots[1].get_editor_property("material_interface")
        )
fill_lights = []
ceiling_lights = []
stack_lens_shader = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/M_StackLens"
)
stack_halo_shader = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/M_StackHaloTint"
)
presentation_backdrop = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/M_PresentationBackdrop"
)
presentation_backdrop_gradient: dict[str, object] = {
    "world_position_nodes": 0,
    "lerp_nodes": 0,
    "emissive_input": "",
    "luminance_values": [],
    "height_values": [],
}
if isinstance(presentation_backdrop, unreal.Material):
    backdrop_nodes = unreal.MaterialEditingLibrary.get_material_expressions(presentation_backdrop)
    presentation_backdrop_gradient["world_position_nodes"] = sum(
        isinstance(node, unreal.MaterialExpressionWorldPosition) for node in backdrop_nodes
    )
    presentation_backdrop_gradient["lerp_nodes"] = sum(
        isinstance(node, unreal.MaterialExpressionLinearInterpolate) for node in backdrop_nodes
    )
    presentation_backdrop_gradient["emissive_input"] = object_path(
        unreal.MaterialEditingLibrary.get_material_property_input_node(
            presentation_backdrop, unreal.MaterialProperty.MP_EMISSIVE_COLOR
        )
    )
    presentation_backdrop_gradient["luminance_values"] = sorted(
        float(node.get_editor_property("constant").r)
        for node in backdrop_nodes
        if isinstance(node, unreal.MaterialExpressionConstant3Vector)
    )
    presentation_backdrop_gradient["height_values"] = sorted(
        float(node.get_editor_property("r"))
        for node in backdrop_nodes
        if isinstance(node, unreal.MaterialExpressionConstant)
    )
stack_lens_inputs: dict[str, str] = {}
stack_lens_parameters: list[str] = []
stack_lens_assignments: dict[str, list[str]] = {"Green": [], "Amber": [], "Red": []}
stack_halo_inputs: dict[str, str] = {}
stack_halo_parameters: list[str] = []
static_driver_components = 0
driver_tshirt_components: list[dict[str, str]] = []
unified_wall: dict[str, object] = {}
hidden_north_wall_components: list[dict[str, object]] = []
sorting_portal_parts: list[dict[str, object]] = []
qualcomm_billboard_parts: list[dict[str, object]] = []
qualcomm_billboard_texture_size = [0, 0]
qualcomm_billboard_logo_textures: list[str] = []
detailed_table_counts = {
    "metal_table": 0,
    "yellow_plastic": 0,
    "blue_plastic": 0,
    "cardboard": 0,
    "label_decal": 0,
}
collapsed_table_components: list[dict[str, object]] = []
dynamic_shelf_components = 0
iq9_evk: dict[str, object] = {
    "actor": "",
    "mesh": "",
    "material": "",
    "component_tags": [],
    "collision": "",
    "vertices_lod0": 0,
    "texture_sizes": {},
    "face_parts": [],
    "old_proxy_present": any(
        unreal.EditorAssetLibrary.does_asset_exist(path)
        for path in (
            "/Game/IQ9EVK/Runtime/SM_SM_IQ9_EVK_Core_Proxy",
            "/Game/IQ9EVK/Runtime/MI_SM_IQ9_EVK_Core_Proxy",
        )
    ),
    "led_parameters": [],
    "temporary_cad_source_present": any(
        unreal.EditorAssetLibrary.does_directory_exist(path)
        for path in ("/Game/IQ9EVK/CADSource", "/Game/IQ9EVK/CADInspect")
    ),
}
iq9_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
if isinstance(iq9_mesh, unreal.StaticMesh) and static_mesh_subsystem is not None:
    iq9_evk["vertices_lod0"] = static_mesh_subsystem.get_number_verts(iq9_mesh, 0)
for texture_name in (
    "T_IQ9_Impostor_Top",
    "T_IQ9_Impostor_Front",
    "T_IQ9_Impostor_Back",
    "T_IQ9_Impostor_Right",
    "T_IQ9_Impostor_Left",
):
    texture = unreal.EditorAssetLibrary.load_asset(
        f"/Game/IQ9EVK/Runtime/Impostor/{texture_name}")
    if isinstance(texture, unreal.Texture2D):
        iq9_evk["texture_sizes"][texture_name] = [
            int(texture.blueprint_get_size_x()),
            int(texture.blueprint_get_size_y()),
        ]
iq9_led_material = unreal.EditorAssetLibrary.load_asset("/Game/IQ9EVK/Runtime/M_IQ9_LED")
if isinstance(iq9_led_material, unreal.Material):
    iq9_evk["led_parameters"] = sorted(
        [str(name) for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(iq9_led_material)]
        + [str(name) for name in unreal.MaterialEditingLibrary.get_vector_parameter_names(iq9_led_material)]
    )
qualcomm_billboard_texture = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/T_QualcommLogo_4K"
)
if isinstance(qualcomm_billboard_texture, unreal.Texture2D):
    qualcomm_billboard_texture_size = [
        int(qualcomm_billboard_texture.blueprint_get_size_x()),
        int(qualcomm_billboard_texture.blueprint_get_size_y()),
    ]
qualcomm_billboard_logo = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/M_QualcommBillboardLogo"
)
if isinstance(qualcomm_billboard_logo, unreal.Material):
    qualcomm_billboard_logo_textures = sorted(
        object_path(node.get_editor_property("texture"))
        for node in unreal.MaterialEditingLibrary.get_material_expressions(qualcomm_billboard_logo)
        if isinstance(node, unreal.MaterialExpressionTextureSample)
        and node.get_editor_property("texture") is not None
    )
hero_material_names = {
    "MI_ForkliftClearCoat": "forklift_clear_coat",
    "MI_ForkliftClearCoat_IsaacYellow": "forklift_clear_coat_isaac_yellow",
    "MI_ForkliftTireDetailed": "tire",
    "MI_AnisotropicRoller": "roller",
    "MI_BrushedSortingTable": "sorting_table",
    "MI_CardboardFiber_A": "cardboard_a",
    "MI_CardboardFiber_D": "cardboard_d",
    "MI_ConcreteScratched": "concrete",
}
hero_material_assignments = {key: 0 for key in hero_material_names.values()}
hero_texture_sizes: dict[str, list[int]] = {}
hero_material_assets: dict[str, str] = {}
for material_name, report_name in hero_material_names.items():
    material = unreal.EditorAssetLibrary.load_asset(f"{CONTENT_ROOT}/VisualFinish/{material_name}")
    if isinstance(material, unreal.MaterialInstanceConstant):
        hero_material_assets[report_name] = object_path(material)
        texture = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(
            material, "BaseColorTexture"
        )
        if isinstance(texture, unreal.Texture2D):
            hero_texture_sizes[report_name] = [
                int(texture.blueprint_get_size_x()),
                int(texture.blueprint_get_size_y()),
            ]
dragonwing_material_names = {
    "MI_DragonwingWorker1Pants": "worker1_pants",
    "MI_DragonwingWorker1ReflectiveVest": "worker1_reflective_vest",
    "MI_DragonwingWorker1ReflectivePants": "worker1_reflective_pants",
    "MI_DragonwingWorker2Shirt": "worker2_shirt",
    "MI_DragonwingDriverTShirt": "driver_tshirt",
}
dragonwing_material_assignments = {key: 0 for key in dragonwing_material_names.values()}
dragonwing_material_assets: dict[str, str] = {}
dragonwing_brand_colors: dict[str, list[float]] = {}
for material_name, report_name in dragonwing_material_names.items():
    material = unreal.EditorAssetLibrary.load_asset(f"{CONTENT_ROOT}/VisualFinish/{material_name}")
    if isinstance(material, unreal.MaterialInstanceConstant):
        dragonwing_material_assets[report_name] = object_path(material)
        color = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(
            material, "DragonwingPurple"
        )
        dragonwing_brand_colors[report_name] = [
            float(color.r), float(color.g), float(color.b), float(color.a)
        ]
dragonwing_fabric = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/M_DragonwingFabric"
)
dragonwing_fabric_used_with_skeletal_mesh = bool(
    isinstance(dragonwing_fabric, unreal.Material)
    and dragonwing_fabric.get_editor_property("used_with_skeletal_mesh")
)
forklift_brand_parameter_names: list[str] = []
isaac_yellow_forklift_color: list[float] = []
forklift_clear_coat = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/M_ForkliftClearCoat"
)
if isinstance(forklift_clear_coat, unreal.Material):
    forklift_brand_parameter_names.extend(
        str(name)
        for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(forklift_clear_coat)
    )
    forklift_brand_parameter_names.extend(
        str(name)
        for name in unreal.MaterialEditingLibrary.get_vector_parameter_names(forklift_clear_coat)
    )
isaac_yellow_forklift = unreal.EditorAssetLibrary.load_asset(
    f"{CONTENT_ROOT}/VisualFinish/MI_ForkliftClearCoat_IsaacYellow"
)
if isinstance(isaac_yellow_forklift, unreal.MaterialInstanceConstant):
    color = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(
        isaac_yellow_forklift, "DragonwingPurple"
    )
    isaac_yellow_forklift_color = [
        float(color.r), float(color.g), float(color.b), float(color.a)
    ]
if isinstance(stack_lens_shader, unreal.Material):
    for name, material_property in (
        ("base_color", unreal.MaterialProperty.MP_BASE_COLOR),
        ("emissive", unreal.MaterialProperty.MP_EMISSIVE_COLOR),
        ("roughness", unreal.MaterialProperty.MP_ROUGHNESS),
    ):
        stack_lens_inputs[name] = object_path(
            unreal.MaterialEditingLibrary.get_material_property_input_node(stack_lens_shader, material_property)
        )
    stack_lens_parameters.extend(
        str(name) for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(stack_lens_shader)
    )
    stack_lens_parameters.extend(
        str(name) for name in unreal.MaterialEditingLibrary.get_vector_parameter_names(stack_lens_shader)
    )
if isinstance(stack_halo_shader, unreal.Material):
    for name, material_property in (
        ("emissive", unreal.MaterialProperty.MP_EMISSIVE_COLOR),
    ):
        stack_halo_inputs[name] = object_path(
            unreal.MaterialEditingLibrary.get_material_property_input_node(stack_halo_shader, material_property)
        )
    stack_halo_parameters.extend(
        str(name) for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(stack_halo_shader)
    )
    stack_halo_parameters.extend(
        str(name) for name in unreal.MaterialEditingLibrary.get_vector_parameter_names(stack_halo_shader)
    )
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    actor_label = actor.get_actor_label()
    actor_tags = {str(tag) for tag in actor.get_editor_property("tags")}
    if actor_label == "IQ9 EVK tabletop":
        iq9_component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if iq9_component is not None:
            iq9_evk.update(
                {
                    "actor": actor_label,
                    "mesh": object_path(iq9_component.get_editor_property("static_mesh")),
                    "material": object_path(iq9_component.get_material(0)),
                    "component_tags": sorted(
                        str(tag) for tag in iq9_component.get_editor_property("component_tags")
                    ),
                    "collision": str(iq9_component.get_collision_enabled()),
                    "location": [round(value, 3) for value in actor.get_actor_location().to_tuple()],
                }
            )
    if "Qai.IQ9EVK.ImpostorFace" in actor_tags:
        iq9_face_component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if iq9_face_component is not None:
            iq9_evk["face_parts"].append(
                {
                    "actor": actor_label,
                    "mesh": object_path(iq9_face_component.get_editor_property("static_mesh")),
                    "material": object_path(iq9_face_component.get_material(0)),
                    "collision": str(iq9_face_component.get_collision_enabled()),
                }
            )
    if actor_label == "UnifiedNorthWall":
        wall_component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if wall_component is not None:
            wall_mesh = wall_component.get_editor_property("static_mesh")
            unified_wall = {
                "mesh": object_path(wall_mesh),
                "location": [
                    round(actor.get_actor_location().x, 3),
                    round(actor.get_actor_location().y, 3),
                    round(actor.get_actor_location().z, 3),
                ],
                "scale": [
                    round(actor.get_actor_scale3d().x, 3),
                    round(actor.get_actor_scale3d().y, 3),
                    round(actor.get_actor_scale3d().z, 3),
                ],
                "materials": [
                    object_path(wall_component.get_material(index))
                    for index in range(wall_component.get_num_materials())
                ],
                "collision": str(wall_component.get_collision_enabled()),
            }
    if actor_label.startswith("SortingPortal_"):
        portal_component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if portal_component is not None:
            sorting_portal_parts.append(
                {
                    "actor": actor_label,
                    "material": object_path(portal_component.get_material(0)),
                    "collision": str(portal_component.get_collision_enabled()),
                }
            )
    if actor_label.startswith("QualcommWallBillboard_"):
        billboard_component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if billboard_component is not None:
            rotation = actor.get_actor_rotation()
            qualcomm_billboard_parts.append(
                {
                    "actor": actor_label,
                    "location": [
                        round(actor.get_actor_location().x, 3),
                        round(actor.get_actor_location().y, 3),
                        round(actor.get_actor_location().z, 3),
                    ],
                    "rotation": [
                        round(rotation.roll, 3),
                        round(rotation.pitch, 3),
                        round(rotation.yaw, 3),
                    ],
                    "scale": [
                        round(actor.get_actor_scale3d().x, 3),
                        round(actor.get_actor_scale3d().y, 3),
                        round(actor.get_actor_scale3d().z, 3),
                    ],
                    "mesh": object_path(billboard_component.get_editor_property("static_mesh")),
                    "material": object_path(billboard_component.get_material(0)),
                    "collision": str(billboard_component.get_collision_enabled()),
                }
            )
    if actor.get_actor_label().startswith("WarehouseFill_"):
        light = actor.get_component_by_class(unreal.PointLightComponent)
        if light is not None:
            fill_lights.append(
                {
                    "actor": actor.get_actor_label(),
                    "intensity": light.get_editor_property("intensity"),
                    "cast_shadows": light.get_editor_property("cast_shadows"),
                }
            )
    if actor.get_actor_label() in {"Area1", "Area2", "Area3", "Area4"}:
        light = actor.get_component_by_class(unreal.RectLightComponent)
        if light is not None:
            ceiling_lights.append(
                {
                    "actor": actor.get_actor_label(),
                    "intensity": light.get_editor_property("intensity"),
                    "cast_shadows": light.get_editor_property("cast_shadows"),
                    "source_width": light.get_editor_property("source_width"),
                    "source_height": light.get_editor_property("source_height"),
                }
            )
    for component in actor.get_components_by_class(unreal.PrimitiveComponent):
        component_tags = {str(tag) for tag in component.get_editor_property("component_tags")}
        for index in range(component.get_num_materials()):
            component_material = component.get_material(index)
            if (
                component_material is not None
                and component_material.get_name() in dragonwing_material_names
            ):
                dragonwing_material_assignments[
                    dragonwing_material_names[component_material.get_name()]
                ] += 1
        if "Qai.ShelfParcel" in component_tags:
            dynamic_shelf_components += 1
        if component.get_name() in {"NorthWall_0", "NorthWall_01_0", "NorthWall_02_0"}:
            hidden_north_wall_components.append(
                {
                    "component": component.get_name(),
                    "visible": bool(component.get_editor_property("visible")),
                    "hidden_in_game": bool(component.get_editor_property("hidden_in_game")),
                    "collision": str(component.get_collision_enabled()),
                }
            )
        if isinstance(component, unreal.StaticMeshComponent):
            static_mesh = component.get_editor_property("static_mesh")
            if component.get_name() == "packing_table_0":
                collapsed_table_components.append(
                    {
                        "visible": bool(component.get_editor_property("visible")),
                        "hidden_in_game": bool(component.get_editor_property("hidden_in_game")),
                        "collision": str(component.get_collision_enabled()),
                    }
                )
            if "Qai.DetailedPackingTable" in actor_tags and bool(
                component.get_editor_property("visible")
            ):
                material_paths = " ".join(
                    object_path(component.get_material(index)).lower()
                    for index in range(component.get_num_materials())
                )
                if "mi_brushedsortingtable" in material_paths:
                    detailed_table_counts["metal_table"] += 1
                elif "plastic_yellow" in material_paths:
                    detailed_table_counts["yellow_plastic"] += 1
                elif "plastic_blue" in material_paths:
                    detailed_table_counts["blue_plastic"] += 1
                elif "mi_cardboardfiber" in material_paths:
                    detailed_table_counts["cardboard"] += 1
                elif "m_corrugatedbox" in material_paths:
                    detailed_table_counts["label_decal"] += 1
            if static_mesh is not None and "male_adult_construction_05" in static_mesh.get_path_name():
                static_driver_components += 1
                if component.get_name().startswith("opaque__fabric__tshirt_"):
                    driver_tshirt_components.append(
                        {
                            "component": component.get_name(),
                            "material": object_path(component.get_material(0)),
                        }
                    )
            for index in range(component.get_num_materials()):
                material = component.get_material(index)
                if material is not None and material.get_name() in hero_material_names:
                    hero_material_assignments[hero_material_names[material.get_name()]] += 1
        tags = component_tags
        for color_name in stack_lens_assignments:
            if any(tag.startswith(f"Qai.{color_name}.") for tag in tags):
                stack_lens_assignments[color_name].append(object_path(component.get_material(0)))

report = {
    "material_instances": len(material_instances),
    "textured_materials": textured_materials,
    "enabled_texture_parameters": texture_parameters,
    "normal_mapped_materials": normal_mapped_materials,
    "roughness_mapped_materials": roughness_mapped_materials,
    "metallic_mapped_materials": metallic_mapped_materials,
    "skeletal_meshes": skeletal_meshes,
    "static_driver_components": static_driver_components,
    "driver_tshirt_components": driver_tshirt_components,
    "unassigned_slots": unassigned_slots,
    "nested_material_slots": nested_slots,
    "hero_meshes": hero_meshes,
    "imprecise_hero_meshes": imprecise_hero_meshes,
    "forklift_blue_parent": forklift_parent,
    "forklift_blue_uses_opacity_texture": forklift_opacity_texture,
    "red_floor_inputs": red_floor_inputs,
    "red_floor_roughness_sampler": red_floor_roughness_sampler,
    "red_floor_parent": red_floor_parent,
    "red_floor_tint": red_floor_tint,
    "red_floor_blend": red_floor_blend,
    "red_floor_world_texture_scale": red_floor_world_texture_scale,
    "red_floor_texture_brightness": red_floor_texture_brightness,
    "red_floor_micro_roughness": red_floor_micro_roughness,
    "red_floor_variation_floor": red_floor_variation_floor,
    "red_floor_world_position_nodes": red_floor_world_position_nodes,
    "clearance_material": clearance_material,
    "rack_carton_material": rack_carton_material,
    "dynamic_shelf_components": dynamic_shelf_components,
    "packing_table_material": packing_table_material,
    "packing_table_bin_material": packing_table_bin_material,
    "fill_lights": sorted(fill_lights, key=lambda row: row["actor"]),
    "ceiling_lights": sorted(ceiling_lights, key=lambda row: row["actor"]),
    "hero_material_assets": hero_material_assets,
    "hero_material_assignments": hero_material_assignments,
    "hero_texture_sizes": hero_texture_sizes,
    "dragonwing_material_assets": dragonwing_material_assets,
    "dragonwing_material_assignments": dragonwing_material_assignments,
    "dragonwing_brand_colors": dragonwing_brand_colors,
    "dragonwing_fabric_used_with_skeletal_mesh": dragonwing_fabric_used_with_skeletal_mesh,
    "forklift_brand_parameter_names": sorted(set(forklift_brand_parameter_names)),
    "isaac_yellow_forklift_color": isaac_yellow_forklift_color,
    "stack_lens_inputs": stack_lens_inputs,
    "stack_lens_parameters": sorted(set(stack_lens_parameters)),
    "stack_lens_assignments": stack_lens_assignments,
    "stack_halo_inputs": stack_halo_inputs,
    "stack_halo_parameters": sorted(set(stack_halo_parameters)),
    "presentation_backdrop_gradient": presentation_backdrop_gradient,
    "unified_wall": unified_wall,
    "hidden_north_wall_components": sorted(
        hidden_north_wall_components, key=lambda row: str(row["component"])
    ),
    "sorting_portal_parts": sorted(sorting_portal_parts, key=lambda row: str(row["actor"])),
    "qualcomm_billboard": {
        "texture_size": qualcomm_billboard_texture_size,
        "logo_textures": qualcomm_billboard_logo_textures,
        "parts": sorted(qualcomm_billboard_parts, key=lambda row: str(row["actor"])),
    },
    "detailed_table_counts": detailed_table_counts,
    "collapsed_table_components": collapsed_table_components,
    "iq9_evk": iq9_evk,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

errors = []
if unassigned_slots:
    errors.append(f"{len(unassigned_slots)} mesh material slots are unassigned")
if nested_slots:
    errors.append(f"{len(nested_slots)} mesh slots still use broken nested USD instances")
if hero_meshes < 20 or imprecise_hero_meshes:
    errors.append(
        f"hero machinery vertex precision incomplete: {hero_meshes} meshes, "
        f"{len(imprecise_hero_meshes)} imprecise"
    )
if "UsdPreviewSurfaceTranslucent" in forklift_parent:
    errors.append("forklift blue body still uses Unreal's translucent USD material")
if forklift_opacity_texture != 0.0:
    errors.append("forklift blue body still samples the near-white opacity texture")
if set(red_floor_inputs) != {"base_color", "normal", "roughness"} or not all(red_floor_inputs.values()):
    errors.append(f"red epoxy shader is missing PBR inputs: {red_floor_inputs}")
if "LINEAR_GRAYSCALE" not in red_floor_roughness_sampler.upper():
    errors.append(
        "red epoxy roughness sampler would fail the runtime shader compile: "
        f"{red_floor_roughness_sampler}"
    )
if (
    not red_floor_parent.endswith("/VisualFinish/M_RedEpoxyFloor.M_RedEpoxyFloor")
    or not 0.15 <= red_floor_blend <= 0.40
    or not 0.0015 <= red_floor_world_texture_scale <= 0.0030
    or not 2.0 <= red_floor_texture_brightness <= 4.0
    or not 0.15 <= red_floor_micro_roughness <= 0.35
    or not 0.50 <= red_floor_variation_floor <= 0.75
    or red_floor_world_position_nodes != 1
):
    errors.append(
        f"red epoxy instance overrides are invalid: parent={red_floor_parent}, "
        f"tint={red_floor_tint}, blend={red_floor_blend}, "
        f"world_texture_scale={red_floor_world_texture_scale}, "
        f"texture_brightness={red_floor_texture_brightness}, "
        f"micro_roughness={red_floor_micro_roughness}, "
        f"variation_floor={red_floor_variation_floor}, "
        f"world_position_nodes={red_floor_world_position_nodes}"
    )
if not clearance_material.endswith("/VisualFinish/MI_RedEpoxyFloor.MI_RedEpoxyFloor"):
    errors.append(f"clearance mesh still has a flat material: {clearance_material}")
if not rack_carton_material.endswith(
    "/VisualFinish/M_HiddenBakedRackCartons.M_HiddenBakedRackCartons"
):
    errors.append(f"baked rack cartons are still rendered: {rack_carton_material}")
if dynamic_shelf_components != 24:
    errors.append(f"expected 24 independent shelf cartons, found {dynamic_shelf_components}")
if (
    iq9_evk.get("actor") != "IQ9 EVK tabletop"
    or not str(iq9_evk.get("mesh", "")).endswith(
        "/Engine/BasicShapes/Cube.Cube"
    )
    or not str(iq9_evk.get("material", "")).endswith(
        "/IQ9EVK/Runtime/Impostor/M_IQ9_ImpostorBody.M_IQ9_ImpostorBody"
    )
    or set(iq9_evk.get("component_tags", [])) != {"Qai.DynamicProp", "Qai.IQ9EVK"}
    or iq9_evk.get("collision") != "<CollisionEnabled.NO_COLLISION: 0>"
    # Engine basic-shape render data may report zero vertices in a commandlet
    # even though the exact Cube asset path above is valid and fully rendered.
    or int(iq9_evk.get("vertices_lod0", 0)) != 0
    and not 8 <= int(iq9_evk.get("vertices_lod0", 0)) <= 60
    or len(iq9_evk.get("face_parts", [])) != 5
    or any(
        not str(face.get("mesh", "")).endswith("/Engine/BasicShapes/Plane.Plane")
        or "/IQ9EVK/Runtime/Impostor/M_IQ9_Impostor_" not in str(face.get("material", ""))
        or face.get("collision") != "<CollisionEnabled.NO_COLLISION: 0>"
        for face in iq9_evk.get("face_parts", [])
    )
    or set(iq9_evk.get("led_parameters", [])) != {"LedColor", "LedStrength"}
    or iq9_evk.get("temporary_cad_source_present")
    or iq9_evk.get("old_proxy_present")
    or set(iq9_evk.get("texture_sizes", {}))
    != {
        "T_IQ9_Impostor_Top",
        "T_IQ9_Impostor_Front",
        "T_IQ9_Impostor_Back",
        "T_IQ9_Impostor_Right",
        "T_IQ9_Impostor_Left",
    }
    or iq9_evk.get("texture_sizes", {}).get("T_IQ9_Impostor_Top") != [1024, 1024]
    or any(
        iq9_evk.get("texture_sizes", {}).get(f"T_IQ9_Impostor_{face}") != [512, 1024]
        for face in ("Front", "Back")
    )
    or any(
        iq9_evk.get("texture_sizes", {}).get(f"T_IQ9_Impostor_{face}") != [1024, 512]
        for face in ("Right", "Left")
    )
):
    errors.append(f"optimized movable IQ9 EVK asset is incomplete: {iq9_evk}")
if not packing_table_material.endswith(
    "/VisualFinish/MI_BrushedSortingTable.MI_BrushedSortingTable"
):
    errors.append(f"sorting table still uses its yellow display material: {packing_table_material}")
if "/packing_table/container_h20/" not in packing_table_bin_material or not packing_table_bin_material.endswith(
    "/MI_opaque__plastic__container.MI_opaque__plastic__container"
):
    errors.append(f"sorting table lost its authored colored bins: {packing_table_bin_material}")
if len(fill_lights) != 4 or any(row["cast_shadows"] for row in fill_lights):
    errors.append(f"warehouse fill-light budget is invalid: {fill_lights}")
if (
    len(ceiling_lights) != 4
    or sum(int(row["cast_shadows"]) for row in ceiling_lights) != 2
    or any(row["source_width"] < 470.0 or row["source_height"] < 250.0 for row in ceiling_lights)
):
    errors.append(f"ceiling soft-shadow budget is invalid: {ceiling_lights}")
expected_assignments = {
    "forklift_clear_coat": 2,
    # Available for runtime A/B selection, intentionally not level-assigned.
    "forklift_clear_coat_isaac_yellow": 0,
    "tire": 8,
    "roller": 78,
    "sorting_table": 5,
    "cardboard_a": 4,
    "cardboard_d": 34,
    "concrete": 2,
}
if hero_material_assignments != expected_assignments:
    errors.append(f"hero material assignments regressed: {hero_material_assignments}")
expected_detailed_table = {
    "metal_table": 4,
    "yellow_plastic": 4,
    "blue_plastic": 3,
    "cardboard": 4,
    "label_decal": 8,
}
if detailed_table_counts != expected_detailed_table:
    errors.append(f"detailed sorting-table material split regressed: {detailed_table_counts}")
if (
    len(collapsed_table_components) != 1
    or collapsed_table_components[0]["visible"]
    or not collapsed_table_components[0]["hidden_in_game"]
    or collapsed_table_components[0]["collision"] != "<CollisionEnabled.NO_COLLISION: 0>"
):
    errors.append(
        f"collapsed metallic sorting-table mesh is still visible: {collapsed_table_components}"
    )
if set(hero_material_assets) != set(expected_assignments):
    errors.append(f"hero material assets are incomplete: {hero_material_assets}")
expected_dragonwing_assignments = {
    "worker1_pants": 1,
    "worker1_reflective_vest": 1,
    "worker1_reflective_pants": 1,
    "worker2_shirt": 1,
    "driver_tshirt": 2,
}
if dragonwing_material_assignments != expected_dragonwing_assignments:
    errors.append(f"Dragonwing clothing assignments regressed: {dragonwing_material_assignments}")
if set(dragonwing_material_assets) != set(expected_dragonwing_assignments):
    errors.append(f"Dragonwing clothing material assets are incomplete: {dragonwing_material_assets}")
if not dragonwing_fabric_used_with_skeletal_mesh:
    errors.append("Dragonwing fabric shader is missing its skeletal-mesh usage permutation")
expected_dragonwing_linear = [0.0318960331, 0.0003035270, 0.2086368701, 1.0]
if any(
    len(color) != 4
    or any(abs(actual - expected) > 1.0e-5 for actual, expected in zip(color, expected_dragonwing_linear))
    for color in dragonwing_brand_colors.values()
):
    errors.append(
        f"Dragonwing clothing purple is not sRGB #32017E in linear space: {dragonwing_brand_colors}"
    )
if (
    {row["component"] for row in driver_tshirt_components}
    != {"opaque__fabric__tshirt_0", "opaque__fabric__tshirt_1"}
    or any(
        not row["material"].endswith(
            "/VisualFinish/MI_DragonwingDriverTShirt.MI_DragonwingDriverTShirt"
        )
        for row in driver_tshirt_components
    )
):
    errors.append(f"Dragonwing seated-driver shirt assignments regressed: {driver_tshirt_components}")
required_forklift_brand_parameters = {
    "DragonwingBrandStrength",
    "DragonwingPaintMaskSharpness",
    "DragonwingPaintDetailBrightness",
    "DragonwingPaintDetailFloor",
    "DragonwingPurple",
}
if not required_forklift_brand_parameters.issubset(forklift_brand_parameter_names):
    errors.append(
        f"Dragonwing forklift paint shader parameters are incomplete: {forklift_brand_parameter_names}"
    )
expected_isaac_yellow_linear = [0.9189189, 0.48398682, 0.03193154, 1.0]
if (
    len(isaac_yellow_forklift_color) != 4
    or any(
        abs(actual - expected) > 1.0e-5
        for actual, expected in zip(isaac_yellow_forklift_color, expected_isaac_yellow_linear)
    )
):
    errors.append(
        "original Isaac yellow forklift material is missing or has the wrong authored tint: "
        f"{isaac_yellow_forklift_color}"
    )
for name in (
    "forklift_clear_coat",
    "forklift_clear_coat_isaac_yellow",
    "tire",
    "cardboard_a",
    "cardboard_d",
):
    size = hero_texture_sizes.get(name, [0, 0])
    if max(size) < 1024:
        errors.append(f"{name} base-color bake is below hero resolution: {size}")
if set(stack_lens_inputs) != {"base_color", "emissive", "roughness"} or not all(stack_lens_inputs.values()):
    errors.append(f"stack-lens shader is missing required outputs: {stack_lens_inputs}")
if not {"LensColor", "BaseBrightness", "EmissiveStrength"}.issubset(stack_lens_parameters):
    errors.append(f"stack-lens runtime parameters are incomplete: {stack_lens_parameters}")
if set(stack_halo_inputs) != {"emissive"} or not all(stack_halo_inputs.values()):
    errors.append(f"stack-halo shader is missing required outputs: {stack_halo_inputs}")
if not {"HaloColor", "HaloStrength"}.issubset(stack_halo_parameters):
    errors.append(f"stack-halo runtime parameters are incomplete: {stack_halo_parameters}")
gradient_luminance = presentation_backdrop_gradient["luminance_values"]
gradient_heights = presentation_backdrop_gradient["height_values"]
if (
    presentation_backdrop_gradient["world_position_nodes"] != 1
    or presentation_backdrop_gradient["lerp_nodes"] < 1
    or not presentation_backdrop_gradient["emissive_input"]
    or not isinstance(gradient_luminance, list)
    or len(gradient_luminance) != 2
    or gradient_luminance[0] < 30.0
    or gradient_luminance[1] < 60.0
    or gradient_luminance[1] > 120.0
    or not isinstance(gradient_heights, list)
    or len(gradient_heights) != 2
    or gradient_heights[0] > -10000.0
    or gradient_heights[1] < 20000.0
):
    errors.append(f"presentation background is not a restrained grey gradient: {presentation_backdrop_gradient}")
if (
    not str(unified_wall.get("mesh", "")).endswith("/Engine/BasicShapes/Cube.Cube")
    or unified_wall.get("location") != [58.5, -352.0, 365.0]
    or unified_wall.get("scale") != [11.83, 0.16, 4.7]
    or unified_wall.get("collision") != "<CollisionEnabled.NO_COLLISION: 0>"
    or not unified_wall.get("materials")
    or any(
        not str(material).endswith("/VisualFinish/M_UnifiedWall.M_UnifiedWall")
        for material in unified_wall.get("materials", [])
    )
):
    errors.append(f"north wall is not using the continuous masked render surface: {unified_wall}")
if (
    {row["component"] for row in hidden_north_wall_components}
    != {"NorthWall_0", "NorthWall_01_0", "NorthWall_02_0"}
    or any(row["visible"] or not row["hidden_in_game"] for row in hidden_north_wall_components)
    or any(row["collision"] == "<CollisionEnabled.NO_COLLISION: 0>" for row in hidden_north_wall_components)
):
    errors.append(
        f"authored north-wall collision/render split is invalid: {hidden_north_wall_components}"
    )
portal_material_suffixes = {
    "/VisualFinish/M_SortingPortalFrame.M_SortingPortalFrame",
    "/VisualFinish/M_SortingPortalInfeed.M_SortingPortalInfeed",
    "/VisualFinish/M_SortingPortalOutfeed.M_SortingPortalOutfeed",
}
if (
    len(sorting_portal_parts) != 12
    or any(row["collision"] != "<CollisionEnabled.NO_COLLISION: 0>" for row in sorting_portal_parts)
    or any(
        not any(str(row["material"]).endswith(suffix) for suffix in portal_material_suffixes)
        for row in sorting_portal_parts
    )
):
    errors.append(f"sorting portal assembly is incomplete or obstructive: {sorting_portal_parts}")
expected_billboard_materials = {
    "QualcommWallBillboard_Back": "/VisualFinish/M_SortingPortalFrame.M_SortingPortalFrame",
    "QualcommWallBillboard_White": "/VisualFinish/M_QualcommBillboardWhite.M_QualcommBillboardWhite",
    "QualcommWallBillboard_Logo": "/VisualFinish/M_QualcommBillboardLogo.M_QualcommBillboardLogo",
}
billboard_by_actor = {str(row["actor"]): row for row in qualcomm_billboard_parts}
billboard_logo = billboard_by_actor.get("QualcommWallBillboard_Logo", {})
if (
    qualcomm_billboard_texture_size != [1019, 256]
    or set(qualcomm_billboard_logo_textures)
    != {f"{CONTENT_ROOT}/VisualFinish/T_QualcommLogo_4K.T_QualcommLogo_4K"}
    or set(billboard_by_actor) != set(expected_billboard_materials)
    or any(
        not str(billboard_by_actor[label]["material"]).endswith(material_suffix)
        for label, material_suffix in expected_billboard_materials.items()
    )
    or any(
        row["collision"] != "<CollisionEnabled.NO_COLLISION: 0>"
        for row in qualcomm_billboard_parts
    )
    or billboard_logo.get("location") != [-260.0, -336.8, 320.0]
    or billboard_logo.get("rotation") != [-90.0, 0.0, 0.0]
    or billboard_logo.get("scale") != [5.2, -1.306, 1.0]
):
    errors.append(
        "Qualcomm Dragonwing wall billboard lost its upright logo, physical backing, or placement: "
        f"texture={qualcomm_billboard_texture_size}, samples={qualcomm_billboard_logo_textures}, "
        f"parts={qualcomm_billboard_parts}"
    )
for color_name, assignments in stack_lens_assignments.items():
    suffix = f"/VisualFinish/MI_StackLens_{color_name}.MI_StackLens_{color_name}"
    if len(assignments) != 3 or any(not material.endswith(suffix) for material in assignments):
        errors.append(f"{color_name} stack-lens assignments are invalid: {assignments}")
# The port deliberately keeps exactly the two walking workers as skeletal
# characters. The remaining seated driver is retained as seven pre-skinned
# static material-part meshes, avoiding an idle runtime skeleton.
if skeletal_meshes != 2:
    errors.append(f"expected 2 required worker skeletal meshes, found {skeletal_meshes}")
if static_driver_components != 7:
    errors.append(
        f"expected 7 static seated-driver components, found {static_driver_components}"
    )
# Guard the exact PBR set so a future import cannot silently drop textures.
if textured_materials < 30 or texture_parameters < 110:
    errors.append(
        f"too few exact textures survived import: {textured_materials} materials, "
        f"{texture_parameters} enabled parameters"
    )
if normal_mapped_materials < 45 or roughness_mapped_materials < 25:
    errors.append(
        "PBR channel coverage regressed: "
        f"normal={normal_mapped_materials}, roughness={roughness_mapped_materials}, "
        f"metallic={metallic_mapped_materials}"
    )
if errors:
    raise RuntimeError("; ".join(errors))

PASS_MARKER.write_text("ok\n", encoding="utf-8")
unreal.log(f"QAI_CONVEYOR_VISUAL_VERIFICATION={json.dumps(report, separators=(',', ':'))}")
