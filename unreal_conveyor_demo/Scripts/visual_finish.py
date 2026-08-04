"""Apply native Unreal material shaders and balanced warehouse fill lighting."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


CONTENT_ROOT = "/Game/ConveyorRuntime"
MATERIAL_FOLDER = f"{CONTENT_ROOT}/VisualFinish"
RED_FLOOR_MATERIAL = f"{MATERIAL_FOLDER}/M_RedEpoxyFloor"
RED_FLOOR_INSTANCE = f"{MATERIAL_FOLDER}/MI_RedEpoxyFloor"
STACK_LENS_MATERIAL = f"{MATERIAL_FOLDER}/M_StackLens"
STACK_HALO_MATERIAL = f"{MATERIAL_FOLDER}/M_StackHaloTint"
BACKDROP_MATERIAL = f"{MATERIAL_FOLDER}/M_PresentationBackdrop"
DETAIL_SURFACE_MATERIAL = f"{MATERIAL_FOLDER}/M_DetailSurface"
CLEAR_COAT_MATERIAL = f"{MATERIAL_FOLDER}/M_ForkliftClearCoat"
ANISOTROPIC_MATERIAL = f"{MATERIAL_FOLDER}/M_AnisotropicMetal"
DRAGONWING_FABRIC_MATERIAL = f"{MATERIAL_FOLDER}/M_DragonwingFabric"
HIDDEN_RACK_CARTON_MATERIAL = f"{MATERIAL_FOLDER}/M_HiddenBakedRackCartons"
UNIFIED_WALL_MESH = f"{MATERIAL_FOLDER}/SM_UnifiedNorthWall"
UNIFIED_WALL_MATERIAL = f"{MATERIAL_FOLDER}/M_UnifiedWall"
PORTAL_FRAME_MATERIAL = f"{MATERIAL_FOLDER}/M_SortingPortalFrame"
PORTAL_INFEED_MATERIAL = f"{MATERIAL_FOLDER}/M_SortingPortalInfeed"
PORTAL_OUTFEED_MATERIAL = f"{MATERIAL_FOLDER}/M_SortingPortalOutfeed"
QUALCOMM_BILLBOARD_TEXTURE = f"{MATERIAL_FOLDER}/T_QualcommLogo_4K"
QUALCOMM_BILLBOARD_WHITE = f"{MATERIAL_FOLDER}/M_QualcommBillboardWhite"
QUALCOMM_BILLBOARD_LOGO = f"{MATERIAL_FOLDER}/M_QualcommBillboardLogo"
QUALCOMM_BILLBOARD_SOURCE = "SourceAssets/Branding/Dragonwing-Logo-Cropped.png"
STACK_LENS_INSTANCES = {
    "Green": f"{MATERIAL_FOLDER}/MI_StackLens_Green",
    "Amber": f"{MATERIAL_FOLDER}/MI_StackLens_Amber",
    "Red": f"{MATERIAL_FOLDER}/MI_StackLens_Red",
}
FLOOR_MI = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "FloorFinish/Tile1/Looks/MI_Floor_01"
)
FORKLIFT_MI = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Forklifts/Forklift1/Looks/MI_M_Forklift_C01"
)
FORKLIFT_BLUE_MI = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Forklifts/Forklift1/Looks/MI_M_Forklift_C01_Blue"
)
ROLLER_MI = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Conveyor/Modules/ExtensionEast/Looks/MI_Aluminium_Brushed_A"
)
CARD_BOX_A_MI = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "DynamicCargo/Forklift1Carton/Asset/Looks/MI_CardBoxA"
)
CARD_BOX_MI = (
    f"{CONTENT_ROOT}/Parcels/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Conveyor/Parcels/Parcel1/Asset/Looks/MI_CardBoxD"
)
CLEARANCE_MESH = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/SM_ClearanceZone"
)
RACK_MESH = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/Storage/SM_WestRack"
)
PACKING_TABLE_MESH = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/SM_packing_table"
)
WORKER1_LOOKS = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Workers/Worker1/Character/male_adult_construction_01/Looks"
)
WORKER2_LOOKS = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Workers/Worker2/Character/male_adult_construction_03/Looks"
)
DRIVER_LOOKS = (
    f"{CONTENT_ROOT}/Scene/warehouse_conveyor_baked/World/CodexPoC/ConveyorSafety/"
    "Forklifts/Forklift1/body/DriverMount/Character/male_adult_construction_05/Looks"
)


def srgb_channel(value: int) -> float:
    encoded = value / 255.0
    return encoded / 12.92 if encoded <= 0.04045 else ((encoded + 0.055) / 1.055) ** 2.4


# Dragonwing purple, authored as sRGB #32017E and converted to the scene-linear
# value expected by Unreal material vector parameters.
DRAGONWING_PURPLE = unreal.LinearColor(
    srgb_channel(0x32), srgb_channel(0x01), srgb_channel(0x7E), 1.0
)


def set_property(obj: object, name: str, value: object) -> None:
    obj.set_editor_property(name, value)  # type: ignore[attr-defined]


def load_required(path: str, expected_type: type) -> object:
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(asset, expected_type):
        raise RuntimeError(f"Required {expected_type.__name__} is missing: {path}")
    return asset


def expression(material: unreal.Material, cls: type, x: int, y: int) -> object:
    value = unreal.MaterialEditingLibrary.create_material_expression(material, cls.static_class(), x, y)
    if value is None:
        raise RuntimeError(f"Could not create material expression {cls.__name__}")
    return value


def connect(source: object, output: str, target: object, input_name: str) -> None:
    if not unreal.MaterialEditingLibrary.connect_material_expressions(source, output, target, input_name):
        raise RuntimeError(f"Could not connect {source}.{output} to {target}.{input_name}")


def connect_output(source: object, output: str, material_property: unreal.MaterialProperty) -> None:
    if not unreal.MaterialEditingLibrary.connect_material_property(source, output, material_property):
        raise RuntimeError(f"Could not connect {source}.{output} to {material_property}")


def source_texture(source: unreal.MaterialInstanceConstant, parameter: str) -> unreal.Texture:
    texture = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(
        source, parameter
    )
    if not isinstance(texture, unreal.Texture):
        raise RuntimeError(f"{source.get_path_name()} has no {parameter}")
    return texture


def create_textured_master(
    asset_path: str,
    source: unreal.MaterialInstanceConstant,
    *,
    clear_coat: bool = False,
    anisotropic: bool = False,
    micro_roughness: bool = False,
) -> unreal.Material:
    """Create a compact native shader around the exact imported PBR maps."""
    material = unreal.EditorAssetLibrary.load_asset(asset_path)
    if isinstance(material, unreal.Material):
        # Upgrade pre-existing masters in place. The forklift LPG cylinder is
        # part of the shared body atlas rather than a separable mesh slot, so
        # anisotropy must be masked per pixel by the authored metallic map.
        scalar_names = {
            str(name) for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(material)
        }
        if anisotropic and "QaiAnisotropySchema" not in scalar_names:
            nodes = unreal.MaterialEditingLibrary.get_material_expressions(material)
            metallic = next(
                (
                    node
                    for node in nodes
                    if isinstance(node, unreal.MaterialExpressionTextureSampleParameter2D)
                    and str(node.get_editor_property("parameter_name")) == "MetallicTexture"
                ),
                None,
            )
            if metallic is None:
                raise RuntimeError(f"{asset_path} is missing its metallic texture parameter")
            anisotropy_node = next(
                (
                    node
                    for node in nodes
                    if isinstance(node, unreal.MaterialExpressionScalarParameter)
                    and str(node.get_editor_property("parameter_name")) == "Anisotropy"
                ),
                None,
            )
            if anisotropy_node is None:
                anisotropy_node = expression(
                    material, unreal.MaterialExpressionScalarParameter, -380, 930
                )
                set_property(anisotropy_node, "parameter_name", "Anisotropy")
                set_property(anisotropy_node, "default_value", 0.55)
            mask = expression(material, unreal.MaterialExpressionMultiply, -130, 900)
            connect(metallic, "R", mask, "A")
            connect(anisotropy_node, "", mask, "B")
            attributes = next(
                (node for node in nodes if isinstance(node, unreal.MaterialExpressionMakeMaterialAttributes)),
                None,
            )
            if attributes is not None:
                connect(mask, "", attributes, "Anisotropy")
            else:
                connect_output(mask, "", unreal.MaterialProperty.MP_ANISOTROPY)
            schema = expression(material, unreal.MaterialExpressionScalarParameter, -130, 1010)
            set_property(schema, "parameter_name", "QaiAnisotropySchema")
            set_property(schema, "default_value", 2.0)
            unreal.MaterialEditingLibrary.layout_material_expressions(material)
            unreal.MaterialEditingLibrary.recompile_material(material)
            unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
        return material

    name = asset_path.rsplit("/", 1)[-1]
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
    )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"Could not create {asset_path}")
    if clear_coat:
        set_property(material, "shading_model", unreal.MaterialShadingModel.MSM_CLEAR_COAT)
        set_property(material, "use_material_attributes", True)
        attributes = expression(
            material, unreal.MaterialExpressionMakeMaterialAttributes, 250, 80
        )
    else:
        attributes = None

    def surface_output(
        source_expression: object,
        output_name: str,
        material_property: unreal.MaterialProperty,
        attribute_input: str,
    ) -> None:
        if attributes is not None:
            connect(source_expression, output_name, attributes, attribute_input)
        else:
            connect_output(source_expression, output_name, material_property)

    base = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -950, -360)
    set_property(base, "parameter_name", "BaseColorTexture")
    set_property(base, "texture", source_texture(source, "BaseColorTexture"))
    set_property(base, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    surface_output(base, "RGB", unreal.MaterialProperty.MP_BASE_COLOR, "BaseColor")

    normal = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -950, -80)
    set_property(normal, "parameter_name", "NormalTexture")
    set_property(normal, "texture", source_texture(source, "NormalTexture"))
    set_property(normal, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    surface_output(normal, "RGB", unreal.MaterialProperty.MP_NORMAL, "Normal")

    roughness = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -950, 190)
    set_property(roughness, "parameter_name", "RoughnessTexture")
    set_property(roughness, "texture", source_texture(source, "RoughnessTexture"))
    set_property(roughness, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
    roughness_scale = expression(material, unreal.MaterialExpressionScalarParameter, -680, 330)
    set_property(roughness_scale, "parameter_name", "RoughnessScale")
    set_property(roughness_scale, "default_value", 1.0)
    roughness_product = expression(material, unreal.MaterialExpressionMultiply, -430, 230)
    connect(roughness, "R", roughness_product, "A")
    connect(roughness_scale, "", roughness_product, "B")
    roughness_output: object = roughness_product

    if micro_roughness:
        micro_uv = expression(material, unreal.MaterialExpressionTextureCoordinate, -960, 470)
        set_property(micro_uv, "u_tiling", 9.0)
        set_property(micro_uv, "v_tiling", 9.0)
        micro = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -680, 500)
        set_property(micro, "parameter_name", "MicroRoughnessTexture")
        set_property(micro, "texture", source_texture(source, "RoughnessTexture"))
        set_property(micro, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
        connect(micro_uv, "", micro, "UVs")
        micro_strength = expression(material, unreal.MaterialExpressionScalarParameter, -420, 500)
        set_property(micro_strength, "parameter_name", "MicroRoughnessStrength")
        set_property(micro_strength, "default_value", 0.10)
        micro_blend = expression(material, unreal.MaterialExpressionLinearInterpolate, -160, 310)
        connect(roughness_product, "", micro_blend, "A")
        connect(micro, "R", micro_blend, "B")
        connect(micro_strength, "", micro_blend, "Alpha")
        roughness_output = micro_blend
    surface_output(roughness_output, "", unreal.MaterialProperty.MP_ROUGHNESS, "Roughness")

    metallic = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -950, 720)
    set_property(metallic, "parameter_name", "MetallicTexture")
    set_property(metallic, "texture", source_texture(source, "MetallicTexture"))
    set_property(metallic, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
    surface_output(metallic, "R", unreal.MaterialProperty.MP_METALLIC, "Metallic")
    specular = expression(material, unreal.MaterialExpressionScalarParameter, -670, 800)
    set_property(specular, "parameter_name", "Specular")
    set_property(specular, "default_value", 0.5)
    surface_output(specular, "", unreal.MaterialProperty.MP_SPECULAR, "Specular")

    if clear_coat:
        coat = expression(material, unreal.MaterialExpressionScalarParameter, -380, 740)
        set_property(coat, "parameter_name", "ClearCoat")
        set_property(coat, "default_value", 0.72)
        connect(coat, "", attributes, "ClearCoat")
        coat_roughness = expression(material, unreal.MaterialExpressionScalarParameter, -380, 840)
        set_property(coat_roughness, "parameter_name", "ClearCoatRoughness")
        set_property(coat_roughness, "default_value", 0.14)
        connect(coat_roughness, "", attributes, "ClearCoatRoughness")
    if anisotropic:
        anisotropy = expression(material, unreal.MaterialExpressionScalarParameter, -380, 930)
        set_property(anisotropy, "parameter_name", "Anisotropy")
        set_property(anisotropy, "default_value", 0.68)
        anisotropy_mask = expression(material, unreal.MaterialExpressionMultiply, -130, 900)
        connect(metallic, "R", anisotropy_mask, "A")
        connect(anisotropy, "", anisotropy_mask, "B")
        surface_output(
            anisotropy_mask, "", unreal.MaterialProperty.MP_ANISOTROPY, "Anisotropy"
        )
        schema = expression(material, unreal.MaterialExpressionScalarParameter, -130, 1010)
        set_property(schema, "parameter_name", "QaiAnisotropySchema")
        set_property(schema, "default_value", 2.0)

    if attributes is not None:
        connect_output(attributes, "", unreal.MaterialProperty.MP_MATERIAL_ATTRIBUTES)

    unreal.MaterialEditingLibrary.layout_material_expressions(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def configure_dragonwing_forklift_paint(material: unreal.Material) -> None:
    """Recolor only authored blue paint while preserving the shared PBR atlas."""
    library = unreal.MaterialEditingLibrary
    parameter_names = {str(name) for name in library.get_scalar_parameter_names(material)}
    if "DragonwingBrandStrength" in parameter_names:
        return

    nodes = library.get_material_expressions(material)
    base = next(
        (
            node
            for node in nodes
            if isinstance(node, unreal.MaterialExpressionTextureSampleParameter2D)
            and str(node.get_editor_property("parameter_name")) == "BaseColorTexture"
        ),
        None,
    )
    attributes = next(
        (node for node in nodes if isinstance(node, unreal.MaterialExpressionMakeMaterialAttributes)),
        None,
    )
    if base is None or attributes is None:
        raise RuntimeError("Forklift clear-coat master is missing its base-color graph")

    # The Omniverse body atlas also contains the LPG cylinder and dark mechanical
    # pieces. Blue dominance is therefore the safest authored-pixel paint mask:
    # max(R,G) is subtracted from B, sharpened, and clamped before blending.
    max_rg = expression(material, unreal.MaterialExpressionMax, -760, -630)
    connect(base, "R", max_rg, "A")
    connect(base, "G", max_rg, "B")
    blue_delta = expression(material, unreal.MaterialExpressionSubtract, -540, -640)
    connect(base, "B", blue_delta, "A")
    connect(max_rg, "", blue_delta, "B")
    mask_sharpness = expression(material, unreal.MaterialExpressionScalarParameter, -540, -540)
    set_property(mask_sharpness, "parameter_name", "DragonwingPaintMaskSharpness")
    set_property(mask_sharpness, "default_value", 6.0)
    sharpened_mask = expression(material, unreal.MaterialExpressionMultiply, -310, -610)
    connect(blue_delta, "", sharpened_mask, "A")
    connect(mask_sharpness, "", sharpened_mask, "B")
    paint_mask = expression(material, unreal.MaterialExpressionSaturate, -80, -610)
    connect(sharpened_mask, "", paint_mask, "None")
    brand_strength = expression(material, unreal.MaterialExpressionScalarParameter, -80, -510)
    set_property(brand_strength, "parameter_name", "DragonwingBrandStrength")
    set_property(brand_strength, "default_value", 1.0)
    final_mask = expression(material, unreal.MaterialExpressionMultiply, 150, -590)
    connect(paint_mask, "", final_mask, "A")
    connect(brand_strength, "", final_mask, "B")

    desaturate = expression(material, unreal.MaterialExpressionDesaturation, -760, -850)
    connect(base, "RGB", desaturate, "None")
    full_desaturation = expression(material, unreal.MaterialExpressionConstant, -760, -760)
    set_property(full_desaturation, "r", 1.0)
    connect(full_desaturation, "", desaturate, "Fraction")
    detail_brightness = expression(material, unreal.MaterialExpressionScalarParameter, -540, -830)
    set_property(detail_brightness, "parameter_name", "DragonwingPaintDetailBrightness")
    set_property(detail_brightness, "default_value", 1.35)
    detail_scaled = expression(material, unreal.MaterialExpressionMultiply, -310, -820)
    connect(desaturate, "", detail_scaled, "A")
    connect(detail_brightness, "", detail_scaled, "B")
    detail_floor = expression(material, unreal.MaterialExpressionScalarParameter, -310, -730)
    set_property(detail_floor, "parameter_name", "DragonwingPaintDetailFloor")
    set_property(detail_floor, "default_value", 0.05)
    detail_lifted = expression(material, unreal.MaterialExpressionAdd, -80, -800)
    connect(detail_scaled, "", detail_lifted, "A")
    connect(detail_floor, "", detail_lifted, "B")
    detail = expression(material, unreal.MaterialExpressionSaturate, 150, -800)
    connect(detail_lifted, "", detail, "None")
    brand_color = expression(material, unreal.MaterialExpressionVectorParameter, 150, -720)
    set_property(brand_color, "parameter_name", "DragonwingPurple")
    set_property(brand_color, "default_value", DRAGONWING_PURPLE)
    branded_paint = expression(material, unreal.MaterialExpressionMultiply, 380, -790)
    connect(detail, "", branded_paint, "A")
    connect(brand_color, "", branded_paint, "B")
    paint_blend = expression(material, unreal.MaterialExpressionLinearInterpolate, 620, -660)
    connect(base, "RGB", paint_blend, "A")
    connect(branded_paint, "", paint_blend, "B")
    connect(final_mask, "", paint_blend, "Alpha")
    connect(paint_blend, "", attributes, "BaseColor")

    library.layout_material_expressions(material)
    library.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)


def create_dragonwing_fabric_master(
    source: unreal.MaterialInstanceConstant,
) -> unreal.Material:
    """Create a texture-preserving brand shader for isolated clothing slots."""
    material = unreal.EditorAssetLibrary.load_asset(DRAGONWING_FABRIC_MATERIAL)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_DragonwingFabric", MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("Could not create Dragonwing fabric material")

    library = unreal.MaterialEditingLibrary
    set_property(material, "used_with_skeletal_mesh", True)
    library.delete_all_material_expressions(material)
    base = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -900, -310)
    set_property(base, "parameter_name", "BaseColorTexture")
    set_property(base, "texture", source_texture(source, "BaseColorTexture"))
    set_property(base, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    desaturate = expression(material, unreal.MaterialExpressionDesaturation, -660, -310)
    connect(base, "RGB", desaturate, "None")
    full_desaturation = expression(material, unreal.MaterialExpressionConstant, -660, -210)
    set_property(full_desaturation, "r", 1.0)
    connect(full_desaturation, "", desaturate, "Fraction")
    brightness = expression(material, unreal.MaterialExpressionScalarParameter, -430, -260)
    set_property(brightness, "parameter_name", "DetailBrightness")
    set_property(brightness, "default_value", 1.22)
    scaled = expression(material, unreal.MaterialExpressionMultiply, -200, -300)
    connect(desaturate, "", scaled, "A")
    connect(brightness, "", scaled, "B")
    detail_floor = expression(material, unreal.MaterialExpressionScalarParameter, -200, -200)
    set_property(detail_floor, "parameter_name", "DetailFloor")
    set_property(detail_floor, "default_value", 0.08)
    lifted = expression(material, unreal.MaterialExpressionAdd, 20, -280)
    connect(scaled, "", lifted, "A")
    connect(detail_floor, "", lifted, "B")
    detail = expression(material, unreal.MaterialExpressionSaturate, 240, -280)
    connect(lifted, "", detail, "None")
    color = expression(material, unreal.MaterialExpressionVectorParameter, 20, -130)
    set_property(color, "parameter_name", "DragonwingPurple")
    set_property(color, "default_value", DRAGONWING_PURPLE)
    tinted = expression(material, unreal.MaterialExpressionMultiply, 480, -230)
    connect(detail, "", tinted, "A")
    connect(color, "", tinted, "B")
    connect_output(tinted, "", unreal.MaterialProperty.MP_BASE_COLOR)

    normal = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -650, 80)
    set_property(normal, "parameter_name", "NormalTexture")
    set_property(normal, "texture", source_texture(source, "NormalTexture"))
    set_property(normal, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    connect_output(normal, "RGB", unreal.MaterialProperty.MP_NORMAL)
    roughness = expression(material, unreal.MaterialExpressionScalarParameter, -380, 210)
    set_property(roughness, "parameter_name", "Roughness")
    set_property(roughness, "default_value", 0.70)
    connect_output(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    specular = expression(material, unreal.MaterialExpressionScalarParameter, -140, 210)
    set_property(specular, "parameter_name", "Specular")
    set_property(specular, "default_value", 0.34)
    connect_output(specular, "", unreal.MaterialProperty.MP_SPECULAR)

    library.layout_material_expressions(material)
    library.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def create_dragonwing_fabric_instance(
    name: str,
    parent: unreal.Material,
    source: unreal.MaterialInstanceConstant,
    *,
    roughness: float,
    specular: float,
    detail_brightness: float = 1.22,
) -> unreal.MaterialInstanceConstant:
    path = f"{MATERIAL_FOLDER}/{name}"
    instance = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, MATERIAL_FOLDER, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
        )
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError(f"Could not create {path}")
    library = unreal.MaterialEditingLibrary
    library.set_material_instance_parent(instance, parent)
    library.set_material_instance_texture_parameter_value(
        instance, "BaseColorTexture", source_texture(source, "BaseColorTexture")
    )
    library.set_material_instance_texture_parameter_value(
        instance, "NormalTexture", source_texture(source, "NormalTexture")
    )
    library.set_material_instance_vector_parameter_value(
        instance, "DragonwingPurple", DRAGONWING_PURPLE
    )
    library.set_material_instance_scalar_parameter_value(instance, "Roughness", roughness)
    library.set_material_instance_scalar_parameter_value(instance, "Specular", specular)
    library.set_material_instance_scalar_parameter_value(
        instance, "DetailBrightness", detail_brightness
    )
    unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False)
    return instance


def create_textured_instance(
    name: str,
    parent: unreal.Material,
    source: unreal.MaterialInstanceConstant,
    *,
    roughness_scale: float,
    specular: float,
    micro_strength: float | None = None,
    clear_coat: float | None = None,
    clear_coat_roughness: float | None = None,
    anisotropy: float | None = None,
) -> unreal.MaterialInstanceConstant:
    path = f"{MATERIAL_FOLDER}/{name}"
    instance = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name,
            MATERIAL_FOLDER,
            unreal.MaterialInstanceConstant,
            unreal.MaterialInstanceConstantFactoryNew(),
        )
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError(f"Could not create {path}")
    library = unreal.MaterialEditingLibrary
    library.set_material_instance_parent(instance, parent)
    for parameter in ("BaseColorTexture", "NormalTexture", "RoughnessTexture", "MetallicTexture"):
        library.set_material_instance_texture_parameter_value(
            instance, parameter, source_texture(source, parameter)
        )
    if micro_strength is not None:
        library.set_material_instance_texture_parameter_value(
            instance, "MicroRoughnessTexture", source_texture(source, "RoughnessTexture")
        )
        library.set_material_instance_scalar_parameter_value(
            instance, "MicroRoughnessStrength", micro_strength
        )
    library.set_material_instance_scalar_parameter_value(instance, "RoughnessScale", roughness_scale)
    library.set_material_instance_scalar_parameter_value(instance, "Specular", specular)
    if clear_coat is not None:
        library.set_material_instance_scalar_parameter_value(instance, "ClearCoat", clear_coat)
    if clear_coat_roughness is not None:
        library.set_material_instance_scalar_parameter_value(
            instance, "ClearCoatRoughness", clear_coat_roughness
        )
    if anisotropy is not None:
        library.set_material_instance_scalar_parameter_value(instance, "Anisotropy", anisotropy)
    unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False)
    return instance


def create_hero_materials() -> dict[str, unreal.MaterialInstanceConstant]:
    forklift = load_required(FORKLIFT_MI, unreal.MaterialInstanceConstant)
    forklift_blue = load_required(FORKLIFT_BLUE_MI, unreal.MaterialInstanceConstant)
    roller = load_required(ROLLER_MI, unreal.MaterialInstanceConstant)
    cardboard_a = load_required(CARD_BOX_A_MI, unreal.MaterialInstanceConstant)
    cardboard_d = load_required(CARD_BOX_MI, unreal.MaterialInstanceConstant)
    floor = load_required(FLOOR_MI, unreal.MaterialInstanceConstant)

    detail_master = create_textured_master(
        DETAIL_SURFACE_MATERIAL, floor, micro_roughness=True
    )
    clear_coat_master = create_textured_master(
        CLEAR_COAT_MATERIAL, forklift_blue, clear_coat=True, anisotropic=True
    )
    configure_dragonwing_forklift_paint(clear_coat_master)
    anisotropic_master = create_textured_master(
        ANISOTROPIC_MATERIAL, roller, anisotropic=True
    )
    return {
        "forklift_clear_coat": create_textured_instance(
            "MI_ForkliftClearCoat",
            clear_coat_master,
            forklift_blue,
            roughness_scale=0.78,
            specular=0.56,
            clear_coat=0.76,
            clear_coat_roughness=0.13,
            anisotropy=0.52,
        ),
        "tire": create_textured_instance(
            "MI_ForkliftTireDetailed",
            detail_master,
            forklift,
            roughness_scale=1.12,
            specular=0.24,
            micro_strength=0.04,
        ),
        "roller": create_textured_instance(
            "MI_AnisotropicRoller",
            anisotropic_master,
            roller,
            roughness_scale=0.78,
            specular=0.62,
            anisotropy=0.68,
        ),
        # The source worktop/frame arrived as a yellow USD display-color slot.
        # Reuse the authored brushed-aluminium PBR set at a calmer gloss level;
        # slot 1 remains the exact colored blue/yellow/white bin material.
        "sorting_table": create_textured_instance(
            "MI_BrushedSortingTable",
            anisotropic_master,
            roller,
            roughness_scale=0.90,
            specular=0.58,
            anisotropy=0.58,
        ),
        "cardboard_a": create_textured_instance(
            "MI_CardboardFiber_A",
            detail_master,
            cardboard_a,
            roughness_scale=0.96,
            specular=0.28,
            micro_strength=0.08,
        ),
        "cardboard_d": create_textured_instance(
            "MI_CardboardFiber_D",
            detail_master,
            cardboard_d,
            roughness_scale=0.96,
            specular=0.28,
            micro_strength=0.08,
        ),
        "concrete": create_textured_instance(
            "MI_ConcreteScratched",
            detail_master,
            floor,
            roughness_scale=0.94,
            specular=0.42,
            micro_strength=0.12,
        ),
    }


def assign_hero_materials(
    materials: dict[str, unreal.MaterialInstanceConstant],
) -> dict[str, int]:
    counts = {name: 0 for name in materials}
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    packing_table = load_required(PACKING_TABLE_MESH, unreal.StaticMesh)
    road_wheel_prefixes = (
        "left_back_wheel_",
        "left_front_wheel_",
        "right_back_wheel_",
        "right_front_wheel_",
    )
    for actor in actors:
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            name = component.get_name()
            for slot in range(component.get_num_materials()):
                current = component.get_material(slot)
                current_name = current.get_name() if current is not None else ""
                selected: str | None = None
                if slot == 0 and name.startswith("SM_Forklift_C01_Body02_01_"):
                    selected = "forklift_clear_coat"
                elif slot == 0 and name.startswith(road_wheel_prefixes):
                    selected = "tire"
                elif slot == 0 and "Roller" in name:
                    selected = "roller"
                elif slot == 0 and component.get_editor_property("static_mesh") == packing_table:
                    selected = "sorting_table"
                elif current_name in {"MI_CardBoxA", "MI_CardboardFiber_A"}:
                    selected = "cardboard_a"
                elif current_name in {"MI_CardBoxD", "MI_CardboardFiber_D"}:
                    selected = "cardboard_d"
                elif slot == 0 and name.startswith(("FloorFinish", "BehindWallFloor")):
                    selected = "concrete"
                if selected is not None:
                    component.set_material(slot, materials[selected])
                    counts[selected] += 1

    rack = load_required(RACK_MESH, unreal.StaticMesh)
    hidden_rack_cartons = load_required(HIDDEN_RACK_CARTON_MATERIAL, unreal.MaterialInterface)
    rack.set_material(4, hidden_rack_cartons)
    packing_table.set_material(0, materials["sorting_table"])
    unreal.EditorAssetLibrary.save_loaded_asset(rack, only_if_is_dirty=False)
    unreal.EditorAssetLibrary.save_loaded_asset(packing_table, only_if_is_dirty=False)
    if counts["forklift_clear_coat"] != 2 or counts["tire"] != 8:
        raise RuntimeError(f"Hero forklift material binding is incomplete: {counts}")
    if counts["roller"] < 70 or counts["cardboard_d"] < 8:
        raise RuntimeError(f"Hero warehouse material binding is incomplete: {counts}")
    if counts["sorting_table"] != 1:
        raise RuntimeError(f"Sorting-table metallic binding is incomplete: {counts}")
    return counts


def create_dragonwing_brand_materials() -> dict[str, unreal.MaterialInstanceConstant]:
    worker1_pants = load_required(
        f"{WORKER1_LOOKS}/MI_opaque__fabric__workpants", unreal.MaterialInstanceConstant
    )
    worker1_reflective_vest = load_required(
        f"{WORKER1_LOOKS}/MI_retro__reflective__vest", unreal.MaterialInstanceConstant
    )
    worker1_reflective_pants = load_required(
        f"{WORKER1_LOOKS}/MI_retro__reflective__pants", unreal.MaterialInstanceConstant
    )
    worker2_shirt = load_required(
        f"{WORKER2_LOOKS}/MI_opaque__fabric__shirt", unreal.MaterialInstanceConstant
    )
    driver_tshirt = load_required(
        f"{DRIVER_LOOKS}/MI_opaque__fabric__tshirt", unreal.MaterialInstanceConstant
    )
    master = create_dragonwing_fabric_master(worker1_pants)
    return {
        "worker1_pants": create_dragonwing_fabric_instance(
            "MI_DragonwingWorker1Pants",
            master,
            worker1_pants,
            roughness=0.73,
            specular=0.32,
        ),
        "worker1_reflective_vest": create_dragonwing_fabric_instance(
            "MI_DragonwingWorker1ReflectiveVest",
            master,
            worker1_reflective_vest,
            roughness=0.25,
            specular=0.82,
            detail_brightness=1.34,
        ),
        "worker1_reflective_pants": create_dragonwing_fabric_instance(
            "MI_DragonwingWorker1ReflectivePants",
            master,
            worker1_reflective_pants,
            roughness=0.25,
            specular=0.82,
            detail_brightness=1.34,
        ),
        "worker2_shirt": create_dragonwing_fabric_instance(
            "MI_DragonwingWorker2Shirt",
            master,
            worker2_shirt,
            roughness=0.70,
            specular=0.34,
        ),
        "driver_tshirt": create_dragonwing_fabric_instance(
            "MI_DragonwingDriverTShirt",
            master,
            driver_tshirt,
            roughness=0.72,
            specular=0.33,
            detail_brightness=1.25,
        ),
    }


def assign_dragonwing_brand_materials(
    materials: dict[str, unreal.MaterialInstanceConstant],
) -> dict[str, int]:
    counts = {name: 0 for name in materials}
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    for actor in actors:
        actor_name = actor.get_name()
        for component in actor.get_components_by_class(unreal.SkeletalMeshComponent):
            mesh = component.get_editor_property("skeletal_mesh_asset")
            mesh_path = mesh.get_path_name() if mesh is not None else ""
            for slot in range(component.get_num_materials()):
                current = component.get_material(slot)
                current_name = current.get_name() if current is not None else ""
                selected: str | None = None
                if "Workers/Worker1/" in mesh_path or actor_name == "male_adult_construction_01":
                    if current_name.startswith("MI_opaque__fabric__workpants") or current_name in {
                        "MI_QualcommWorker1Pants", "MI_DragonwingWorker1Pants"
                    }:
                        selected = "worker1_pants"
                    elif current_name.startswith("MI_retro__reflective__vest") or current_name in {
                        "MI_QualcommWorker1ReflectiveVest", "MI_DragonwingWorker1ReflectiveVest"
                    }:
                        selected = "worker1_reflective_vest"
                    elif current_name.startswith("MI_retro__reflective__pants") or current_name in {
                        "MI_QualcommWorker1ReflectivePants", "MI_DragonwingWorker1ReflectivePants"
                    }:
                        selected = "worker1_reflective_pants"
                elif "Workers/Worker2/" in mesh_path or actor_name == "male_adult_construction_03":
                    if current_name.startswith("MI_opaque__fabric__shirt") or current_name in {
                        "MI_QualcommWorker2Shirt", "MI_DragonwingWorker2Shirt"
                    }:
                        selected = "worker2_shirt"
                if selected is not None:
                    component.set_material(slot, materials[selected])
                    counts[selected] += 1

        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            static_mesh = component.get_editor_property("static_mesh")
            mesh_path = static_mesh.get_path_name() if static_mesh is not None else ""
            component_name = component.get_name()
            if not (
                component_name.startswith("opaque__fabric__tshirt_")
                and "male_adult_construction_05" in mesh_path
            ):
                continue
            for slot in range(component.get_num_materials()):
                current = component.get_material(slot)
                current_name = current.get_name() if current is not None else ""
                if current_name.startswith("MI_opaque__fabric__tshirt") or current_name in {
                    "MI_QualcommDriverTShirt", "MI_DragonwingDriverTShirt"
                }:
                    component.set_material(slot, materials["driver_tshirt"])
                    counts["driver_tshirt"] += 1

    expected = {
        "worker1_pants": 1,
        "worker1_reflective_vest": 1,
        "worker1_reflective_pants": 1,
        "worker2_shirt": 1,
        "driver_tshirt": 2,
    }
    if counts != expected:
        raise RuntimeError(f"Dragonwing clothing material binding is incomplete: {counts}")
    return counts


def create_red_epoxy_material() -> unreal.MaterialInstanceConstant:
    """Create a world-mapped red safety surface from the Omniverse floor PBR set."""
    floor = load_required(FLOOR_MI, unreal.MaterialInstanceConstant)
    library = unreal.MaterialEditingLibrary
    base_texture = library.get_material_instance_texture_parameter_value(floor, "BaseColorTexture")
    normal_texture = library.get_material_instance_texture_parameter_value(floor, "NormalTexture")
    roughness_texture = library.get_material_instance_texture_parameter_value(floor, "RoughnessTexture")
    if base_texture is None or normal_texture is None or roughness_texture is None:
        raise RuntimeError("The Omniverse floor PBR texture set is incomplete")

    material = unreal.EditorAssetLibrary.load_asset(RED_FLOOR_MATERIAL)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_RedEpoxyFloor", MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("Could not create the red safety-floor material")

    # Rebuild this small owned graph deterministically. World-space projection
    # avoids the clearance helper mesh's unsuitable UVs and keeps macro details
    # aligned with the neighboring authored warehouse floor.
    library.delete_all_material_expressions(material)
    world_position = expression(material, unreal.MaterialExpressionWorldPosition, -1260, -140)
    world_scale = expression(material, unreal.MaterialExpressionScalarParameter, -1260, -40)
    set_property(world_scale, "parameter_name", "WorldTextureScale")
    set_property(world_scale, "default_value", 0.0022)
    world_uv = expression(material, unreal.MaterialExpressionMultiply, -1020, -120)
    connect(world_position, "XY", world_uv, "A")
    connect(world_scale, "", world_uv, "B")

    base = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -780, -380)
    set_property(base, "parameter_name", "SurfaceColorTexture")
    set_property(base, "texture", base_texture)
    set_property(base, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    connect(world_uv, "", base, "UVs")
    desaturate = expression(material, unreal.MaterialExpressionDesaturation, -540, -380)
    connect(base, "RGB", desaturate, "None")
    desaturation = expression(material, unreal.MaterialExpressionConstant, -760, -260)
    set_property(desaturation, "r", 1.0)
    connect(desaturation, "", desaturate, "Fraction")
    brightness = expression(material, unreal.MaterialExpressionScalarParameter, -540, -250)
    set_property(brightness, "parameter_name", "TextureBrightness")
    set_property(brightness, "default_value", 2.8)
    boosted = expression(material, unreal.MaterialExpressionMultiply, -300, -380)
    connect(desaturate, "", boosted, "A")
    connect(brightness, "", boosted, "B")
    variation_floor = expression(material, unreal.MaterialExpressionScalarParameter, -300, -250)
    set_property(variation_floor, "parameter_name", "VariationFloor")
    set_property(variation_floor, "default_value", 0.62)
    lifted = expression(material, unreal.MaterialExpressionAdd, -70, -350)
    connect(boosted, "", lifted, "A")
    connect(variation_floor, "", lifted, "B")
    bounded = expression(material, unreal.MaterialExpressionSaturate, 150, -350)
    connect(lifted, "", bounded, "None")
    tint = expression(material, unreal.MaterialExpressionVectorParameter, -80, -210)
    set_property(tint, "parameter_name", "EpoxyTint")
    set_property(tint, "default_value", unreal.LinearColor(0.42, 0.003, 0.001, 1.0))
    textured_tint = expression(material, unreal.MaterialExpressionMultiply, 380, -320)
    connect(bounded, "", textured_tint, "A")
    connect(tint, "RGB", textured_tint, "B")
    flat_blend = expression(material, unreal.MaterialExpressionScalarParameter, 160, -190)
    set_property(flat_blend, "parameter_name", "TintBlend")
    set_property(flat_blend, "default_value", 0.25)
    color = expression(material, unreal.MaterialExpressionLinearInterpolate, 610, -260)
    connect(textured_tint, "", color, "A")
    connect(tint, "RGB", color, "B")
    connect(flat_blend, "", color, "Alpha")
    connect_output(color, "", unreal.MaterialProperty.MP_BASE_COLOR)

    normal = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -540, 20)
    set_property(normal, "parameter_name", "SurfaceNormalTexture")
    set_property(normal, "texture", normal_texture)
    set_property(normal, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    connect(world_uv, "", normal, "UVs")
    connect_output(normal, "RGB", unreal.MaterialProperty.MP_NORMAL)

    roughness = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -540, 230)
    set_property(roughness, "parameter_name", "SurfaceRoughnessTexture")
    set_property(roughness, "texture", roughness_texture)
    set_property(roughness, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
    connect(world_uv, "", roughness, "UVs")
    roughness_scale = expression(material, unreal.MaterialExpressionScalarParameter, -300, 270)
    set_property(roughness_scale, "parameter_name", "RoughnessScale")
    set_property(roughness_scale, "default_value", 0.74)
    macro_roughness = expression(material, unreal.MaterialExpressionMultiply, -70, 230)
    connect(roughness, "R", macro_roughness, "A")
    connect(roughness_scale, "", macro_roughness, "B")

    micro_scale = expression(material, unreal.MaterialExpressionScalarParameter, -1020, 380)
    set_property(micro_scale, "parameter_name", "MicroWorldScale")
    set_property(micro_scale, "default_value", 0.018)
    micro_uv = expression(material, unreal.MaterialExpressionMultiply, -780, 400)
    connect(world_position, "XY", micro_uv, "A")
    connect(micro_scale, "", micro_uv, "B")
    micro = expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -540, 440)
    set_property(micro, "parameter_name", "MicroRoughnessTexture")
    set_property(micro, "texture", roughness_texture)
    set_property(micro, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
    connect(micro_uv, "", micro, "UVs")
    micro_strength = expression(material, unreal.MaterialExpressionScalarParameter, -300, 500)
    set_property(micro_strength, "parameter_name", "MicroRoughnessStrength")
    set_property(micro_strength, "default_value", 0.18)
    final_roughness = expression(material, unreal.MaterialExpressionLinearInterpolate, 180, 300)
    connect(macro_roughness, "", final_roughness, "A")
    connect(micro, "R", final_roughness, "B")
    connect(micro_strength, "", final_roughness, "Alpha")
    connect_output(final_roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)

    metallic = expression(material, unreal.MaterialExpressionConstant, 180, 470)
    set_property(metallic, "r", 0.0)
    connect_output(metallic, "", unreal.MaterialProperty.MP_METALLIC)
    specular = expression(material, unreal.MaterialExpressionConstant, 180, 550)
    set_property(specular, "r", 0.50)
    connect_output(specular, "", unreal.MaterialProperty.MP_SPECULAR)
    library.delete_unused_expressions(material)
    library.layout_material_expressions(material)
    library.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)

    instance = unreal.EditorAssetLibrary.load_asset(RED_FLOOR_INSTANCE)
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "MI_RedEpoxyFloor",
            MATERIAL_FOLDER,
            unreal.MaterialInstanceConstant,
            unreal.MaterialInstanceConstantFactoryNew(),
        )
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError("Could not create the red epoxy material instance")
    library.set_material_instance_parent(instance, material)
    library.set_material_instance_vector_parameter_value(
        instance, "EpoxyTint", unreal.LinearColor(0.42, 0.003, 0.001, 1.0)
    )
    for name, value in (
        ("TintBlend", 0.25),
        ("WorldTextureScale", 0.0022),
        ("TextureBrightness", 2.8),
        ("VariationFloor", 0.62),
        ("RoughnessScale", 0.74),
        ("MicroWorldScale", 0.018),
        ("MicroRoughnessStrength", 0.18),
    ):
        library.set_material_instance_scalar_parameter_value(instance, name, value)
    for name, texture in (
        ("SurfaceColorTexture", base_texture),
        ("SurfaceNormalTexture", normal_texture),
        ("SurfaceRoughnessTexture", roughness_texture),
        ("MicroRoughnessTexture", roughness_texture),
    ):
        library.set_material_instance_texture_parameter_value(instance, name, texture)
    unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False)
    return instance


def create_stack_lens_materials() -> dict[str, unreal.MaterialInstanceConstant]:
    """Create a compact HDR lens shader with a runtime emissive parameter."""
    material = unreal.EditorAssetLibrary.load_asset(STACK_LENS_MATERIAL)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_StackLens", MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
        if not isinstance(material, unreal.Material):
            raise RuntimeError("Could not create the stack-lens material")
        set_property(material, "two_sided", True)

        lens_color = expression(material, unreal.MaterialExpressionVectorParameter, -800, -160)
        set_property(lens_color, "parameter_name", "LensColor")
        set_property(lens_color, "default_value", unreal.LinearColor(0.015, 1.0, 0.08, 1.0))
        base_brightness = expression(material, unreal.MaterialExpressionScalarParameter, -800, 20)
        set_property(base_brightness, "parameter_name", "BaseBrightness")
        set_property(base_brightness, "default_value", 0.08)
        base_multiply = expression(material, unreal.MaterialExpressionMultiply, -470, -100)
        connect(lens_color, "RGB", base_multiply, "A")
        connect(base_brightness, "", base_multiply, "B")
        connect_output(base_multiply, "", unreal.MaterialProperty.MP_BASE_COLOR)

        emissive_strength = expression(material, unreal.MaterialExpressionScalarParameter, -800, 220)
        set_property(emissive_strength, "parameter_name", "EmissiveStrength")
        set_property(emissive_strength, "default_value", 0.0)
        emissive_multiply = expression(material, unreal.MaterialExpressionMultiply, -470, 160)
        connect(lens_color, "RGB", emissive_multiply, "A")
        connect(emissive_strength, "", emissive_multiply, "B")
        connect_output(emissive_multiply, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

        roughness = expression(material, unreal.MaterialExpressionConstant, -350, 350)
        set_property(roughness, "r", 0.18)
        connect_output(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
        specular = expression(material, unreal.MaterialExpressionConstant, -350, 440)
        set_property(specular, "r", 0.62)
        connect_output(specular, "", unreal.MaterialProperty.MP_SPECULAR)
        unreal.MaterialEditingLibrary.layout_material_expressions(material)
        unreal.MaterialEditingLibrary.recompile_material(material)
        unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)

    expected_parameters = {
        str(name) for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(material)
    }
    expected_parameters.update(
        str(name) for name in unreal.MaterialEditingLibrary.get_vector_parameter_names(material)
    )
    if not {"LensColor", "BaseBrightness", "EmissiveStrength"}.issubset(expected_parameters):
        raise RuntimeError(f"Stack-lens shader is incomplete: {sorted(expected_parameters)}")

    colors = {
        "Green": unreal.LinearColor(0.015, 1.0, 0.08, 1.0),
        "Amber": unreal.LinearColor(1.0, 0.24, 0.008, 1.0),
        "Red": unreal.LinearColor(1.0, 0.025, 0.008, 1.0),
    }
    instances: dict[str, unreal.MaterialInstanceConstant] = {}
    for color_name, path in STACK_LENS_INSTANCES.items():
        instance = unreal.EditorAssetLibrary.load_asset(path)
        if not isinstance(instance, unreal.MaterialInstanceConstant):
            instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                path.rsplit("/", 1)[-1],
                MATERIAL_FOLDER,
                unreal.MaterialInstanceConstant,
                unreal.MaterialInstanceConstantFactoryNew(),
            )
        if not isinstance(instance, unreal.MaterialInstanceConstant):
            raise RuntimeError(f"Could not create {path}")
        unreal.MaterialEditingLibrary.set_material_instance_parent(instance, material)
        unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
            instance, "LensColor", colors[color_name]
        )
        unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
            instance, "BaseBrightness", 0.08
        )
        unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
            instance, "EmissiveStrength", 0.0
        )
        unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False)
        instances[color_name] = instance
    return instances


def create_stack_halo_material() -> unreal.Material:
    """Create a soft colored-transmittance halo used around active lenses."""
    material = unreal.EditorAssetLibrary.load_asset(STACK_HALO_MATERIAL)
    material_existed = isinstance(material, unreal.Material)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_StackHaloTint", MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("Could not create the stack halo material")

    # Modulation is intentional here: it represents colored transmittance
    # through the local signal volume and retains hue against a white wall.
    set_property(material, "blend_mode", unreal.BlendMode.BLEND_MODULATE)
    set_property(material, "shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    set_property(material, "two_sided", False)
    if material_existed:
        unreal.MaterialEditingLibrary.recompile_material(material)
        unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
        return material
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    halo_color = expression(material, unreal.MaterialExpressionVectorParameter, -760, -100)
    set_property(halo_color, "parameter_name", "HaloColor")
    set_property(halo_color, "default_value", unreal.LinearColor(0.01, 0.72, 0.025, 1.0))
    halo_strength = expression(material, unreal.MaterialExpressionScalarParameter, -760, 70)
    set_property(halo_strength, "parameter_name", "HaloStrength")
    set_property(halo_strength, "default_value", 0.0)
    texcoord = expression(material, unreal.MaterialExpressionTextureCoordinate, -760, 260)
    center = expression(material, unreal.MaterialExpressionConstant2Vector, -760, 390)
    set_property(center, "r", 0.5)
    set_property(center, "g", 0.5)
    soft_radial = expression(material, unreal.MaterialExpressionSphereMask, -430, 300)
    connect(texcoord, "", soft_radial, "A")
    connect(center, "", soft_radial, "B")
    set_property(soft_radial, "attenuation_radius", 0.5)
    set_property(soft_radial, "hardness_percent", 12.0)

    tint_weight = expression(material, unreal.MaterialExpressionMultiply, -140, 250)
    connect(soft_radial, "", tint_weight, "A")
    connect(halo_strength, "", tint_weight, "B")
    neutral = expression(material, unreal.MaterialExpressionConstant3Vector, -140, 390)
    set_property(neutral, "constant", unreal.LinearColor(1.0, 1.0, 1.0, 1.0))
    transmittance = expression(material, unreal.MaterialExpressionLinearInterpolate, 160, 180)
    connect(neutral, "", transmittance, "A")
    connect(halo_color, "RGB", transmittance, "B")
    connect(tint_weight, "", transmittance, "Alpha")
    connect_output(transmittance, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    unreal.MaterialEditingLibrary.layout_material_expressions(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def assign_stack_lens_materials(
    instances: dict[str, unreal.MaterialInstanceConstant],
) -> dict[str, object]:
    assigned: dict[str, int] = {name: 0 for name in instances}
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        for component in actor.get_components_by_class(unreal.PrimitiveComponent):
            tags = {str(tag) for tag in component.get_editor_property("component_tags")}
            for color_name, instance in instances.items():
                prefix = f"Qai.{color_name}."
                if any(tag.startswith(prefix) and not tag.endswith(".Glow") for tag in tags):
                    overrides = list(component.get_editor_property("override_materials"))
                    if overrides:
                        overrides[0] = instance
                    else:
                        overrides.append(instance)
                    component.set_editor_property("override_materials", overrides)
                    assigned[color_name] += 1
                    break
    if any(count != 3 for count in assigned.values()):
        raise RuntimeError(f"Expected three imported lenses per signal color, assigned {assigned}")
    return {
        "material": STACK_LENS_MATERIAL,
        "instances": {name: instance.get_path_name() for name, instance in instances.items()},
        "assigned_lenses": assigned,
    }


def assign_materials(red_floor: unreal.MaterialInstanceConstant) -> dict[str, object]:
    clearance = load_required(CLEARANCE_MESH, unreal.StaticMesh)
    rack = load_required(RACK_MESH, unreal.StaticMesh)
    card_box = load_required(CARD_BOX_MI, unreal.MaterialInstanceConstant)
    hidden_rack_cartons = load_required(HIDDEN_RACK_CARTON_MATERIAL, unreal.MaterialInterface)
    concrete_floor = load_required(FLOOR_MI, unreal.MaterialInstanceConstant)

    clearance.set_material(0, red_floor)
    if len(list(rack.get_editor_property("static_materials"))) < 5:
        raise RuntimeError("The warehouse rack no longer has its expected shelf-carton material slot")
    rack.set_material(4, hidden_rack_cartons)
    unreal.EditorAssetLibrary.save_loaded_asset(clearance, only_if_is_dirty=False)
    unreal.EditorAssetLibrary.save_loaded_asset(rack, only_if_is_dirty=False)

    # Clear imported component overrides on these two assets so the corrected
    # static slots are the authoritative cooked bindings.
    corrected_components = 0
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            if mesh == clearance:
                component.set_material(0, red_floor)
                corrected_components += 1
            elif mesh == rack:
                component.set_material(4, hidden_rack_cartons)
                corrected_components += 1
            elif component.get_name().startswith("BehindWallFloor"):
                # The original stage uses the same concrete PBR surface behind
                # the dividing wall. USD imported this helper plane with a
                # display-black override, which looked like an open void.
                component.set_material(0, concrete_floor)
                corrected_components += 1
    return {
        "red_floor_material": red_floor.get_path_name(),
        "rack_baked_carton_material": hidden_rack_cartons.get_path_name(),
        "dynamic_shelf_carton_material": card_box.get_path_name(),
        "corrected_components": corrected_components,
    }


def configure_fill_lights() -> list[dict[str, object]]:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = actor_subsystem.get_all_level_actors()
    by_label = {actor.get_actor_label(): actor for actor in actors}
    for panel_label in ("Area1", "Area2", "Area3", "Area4"):
        panel = by_label.get(panel_label)
        component = panel.get_component_by_class(unreal.RectLightComponent) if panel is not None else None
        if component is None:
            raise RuntimeError(f"Missing authored warehouse panel light {panel_label}")
        set_property(component, "intensity", 4200.0)
        casts_hero_shadows = panel_label in {"Area1", "Area4"}
        set_property(component, "cast_shadows", casts_hero_shadows)
        set_property(component, "source_width", 480.0)
        set_property(component, "source_height", 260.0)
        set_property(component, "indirect_lighting_intensity", 1.25)
        set_property(component, "volumetric_scattering_intensity", 0.12)
        try:
            set_property(component, "contact_shadow_length", 0.055 if casts_hero_shadows else 0.0)
            set_property(
                component,
                "contact_shadow_casting_intensity",
                0.72 if casts_hero_shadows else 0.0,
            )
        except RuntimeError:
            pass
    fills = (
        ("WarehouseFill_Rack", unreal.Vector(-120.0, 325.0, 245.0), 180.0, 620.0),
        ("WarehouseFill_Loading", unreal.Vector(-330.0, -120.0, 190.0), 120.0, 520.0),
        ("WarehouseFill_Conveyor", unreal.Vector(420.0, 110.0, 215.0), 160.0, 650.0),
        ("WarehouseFill_FarBay", unreal.Vector(410.0, -330.0, 185.0), 120.0, 520.0),
    )
    report: list[dict[str, object]] = []
    for label, location, intensity, radius in fills:
        actor = by_label.get(label)
        if actor is None:
            actor = actor_subsystem.spawn_actor_from_class(unreal.PointLight, location, unreal.Rotator())
            if actor is None:
                raise RuntimeError(f"Could not create {label}")
            actor.set_actor_label(label)
        else:
            actor.set_actor_location(location, False, False)
        component = actor.get_component_by_class(unreal.PointLightComponent)
        if component is None:
            raise RuntimeError(f"{label} has no point-light component")
        set_property(component, "intensity", intensity)
        set_property(component, "attenuation_radius", radius)
        set_property(component, "cast_shadows", False)
        set_property(component, "use_temperature", True)
        set_property(component, "temperature", 4800.0)
        set_property(component, "source_radius", 55.0)
        set_property(component, "soft_source_radius", 140.0)
        report.append({"actor": label, "intensity": intensity, "attenuation_radius": radius})
    return report


def create_machine_portal_materials() -> dict[str, unreal.Material]:
    """Build the restrained wall and machine-portal finish materials."""
    definitions = {
        "wall": (UNIFIED_WALL_MATERIAL, unreal.LinearColor(0.72, 0.735, 0.75, 1.0), 0.0, 0.78),
        "frame": (PORTAL_FRAME_MATERIAL, unreal.LinearColor(0.035, 0.045, 0.06, 1.0), 0.82, 0.26),
    }
    result: dict[str, unreal.Material] = {}
    for key, (asset_path, color, metallic_value, roughness_value) in definitions.items():
        material = unreal.EditorAssetLibrary.load_asset(asset_path)
        if not isinstance(material, unreal.Material):
            material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                asset_path.rsplit("/", 1)[-1], MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
            )
        if not isinstance(material, unreal.Material):
            raise RuntimeError(f"Could not create {asset_path}")
        unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
        base = expression(material, unreal.MaterialExpressionConstant3Vector, -420, -120)
        set_property(base, "constant", color)
        connect_output(base, "", unreal.MaterialProperty.MP_BASE_COLOR)
        metallic = expression(material, unreal.MaterialExpressionConstant, -420, 20)
        set_property(metallic, "r", metallic_value)
        connect_output(metallic, "", unreal.MaterialProperty.MP_METALLIC)
        roughness = expression(material, unreal.MaterialExpressionConstant, -420, 150)
        set_property(roughness, "r", roughness_value)
        connect_output(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
        unreal.MaterialEditingLibrary.layout_material_expressions(material)
        unreal.MaterialEditingLibrary.recompile_material(material)
        unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
        result[key] = material

    for key, asset_path, color in (
        ("infeed", PORTAL_INFEED_MATERIAL, unreal.LinearColor(0.20, 3.6, 9.0, 1.0)),
        ("outfeed", PORTAL_OUTFEED_MATERIAL, unreal.LinearColor(9.0, 2.1, 0.10, 1.0)),
    ):
        material = unreal.EditorAssetLibrary.load_asset(asset_path)
        if not isinstance(material, unreal.Material):
            material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                asset_path.rsplit("/", 1)[-1], MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
            )
        if not isinstance(material, unreal.Material):
            raise RuntimeError(f"Could not create {asset_path}")
        unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
        set_property(material, "shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        glow = expression(material, unreal.MaterialExpressionConstant3Vector, -300, 0)
        set_property(glow, "constant", color)
        connect_output(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        unreal.MaterialEditingLibrary.layout_material_expressions(material)
        unreal.MaterialEditingLibrary.recompile_material(material)
        unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
        result[key] = material
    return result


def create_unified_portal_wall(materials: dict[str, unreal.Material]) -> dict[str, object]:
    """Replace the authored seams with a continuous top slab and portal infill."""
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    source_names = {"NorthWall_0", "NorthWall_01_0", "NorthWall_02_0"}
    source_components = [
        component
        for actor in actors
        for component in actor.get_components_by_class(unreal.StaticMeshComponent)
        if component.get_name() in source_names
    ]
    if {component.get_name() for component in source_components} != source_names:
        raise RuntimeError(
            f"Could not find the three authored north-wall pieces: "
            f"{sorted(component.get_name() for component in source_components)}"
        )

    unified_actor = next(
        (actor for actor in actors if actor.get_actor_label() == "UnifiedNorthWall"), None
    )
    unified_mesh = load_required("/Engine/BasicShapes/Cube", unreal.StaticMesh)
    if not isinstance(unified_actor, unreal.StaticMeshActor):
        unified_actor = actor_subsystem.spawn_actor_from_object(
            unified_mesh, unreal.Vector(58.5, -352.0, 365.0), unreal.Rotator()
        )
        if not isinstance(unified_actor, unreal.StaticMeshActor):
            raise RuntimeError("Could not spawn the unified north wall")
        unified_actor.set_actor_label("UnifiedNorthWall")
    unified_component = unified_actor.get_component_by_class(unreal.StaticMeshComponent)
    if unified_component is None:
        raise RuntimeError("Unified north wall has no static-mesh component")
    unified_component.set_static_mesh(unified_mesh)
    unified_actor.set_actor_location(unreal.Vector(58.5, -352.0, 365.0), False, False)
    unified_actor.set_actor_rotation(unreal.Rotator(), False)
    unified_actor.set_actor_scale3d(unreal.Vector(11.83, 0.16, 4.7))
    for slot in range(max(1, unified_component.get_num_materials())):
        unified_component.set_material(slot, materials["wall"])
    unified_component.set_collision_profile_name("NoCollision")
    unified_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    set_property(unified_component, "cast_shadow", True)

    # Keep the exact authored wall collision, but remove its three render
    # surfaces. This preserves both conveyor apertures and forklift blocking.
    for component in source_components:
        component.set_visibility(False, True)
        component.set_hidden_in_game(True)

    cube = load_required("/Engine/BasicShapes/Cube", unreal.StaticMesh)

    def ensure_block(
        label: str,
        location: unreal.Vector,
        size: unreal.Vector,
        material: unreal.Material,
    ) -> unreal.StaticMeshActor:
        actor = next(
            (candidate for candidate in actor_subsystem.get_all_level_actors() if candidate.get_actor_label() == label),
            None,
        )
        if not isinstance(actor, unreal.StaticMeshActor):
            actor = actor_subsystem.spawn_actor_from_object(cube, location, unreal.Rotator())
            if not isinstance(actor, unreal.StaticMeshActor):
                raise RuntimeError(f"Could not create portal part {label}")
            actor.set_actor_label(label)
        actor.set_actor_location(location, False, False)
        actor.set_actor_rotation(unreal.Rotator(), False)
        actor.set_actor_scale3d(unreal.Vector(size.x / 100.0, size.y / 100.0, size.z / 100.0))
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if component is None:
            raise RuntimeError(f"Portal part {label} has no static-mesh component")
        component.set_material(0, material)
        component.set_collision_profile_name("NoCollision")
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        set_property(component, "cast_shadow", material in (materials["wall"], materials["frame"]))
        return actor

    wall_parts: list[str] = []
    for label, left, right in (
        ("UnifiedNorthWall_LowerLeft", -533.0, 232.402),
        ("UnifiedNorthWall_LowerCenter", 351.497, 532.865),
    ):
        wall_parts.append(
            ensure_block(
                label,
                unreal.Vector((left + right) * 0.5, -352.0, 65.0),
                unreal.Vector(right - left, 16.0, 130.0),
                materials["wall"],
            ).get_actor_label()
        )

    portal_specs = (
        ("Infeed", 232.402, 351.497, materials["infeed"]),
        ("Outfeed", 532.865, 650.0, materials["outfeed"]),
    )
    portal_parts: list[str] = []
    for portal_name, left, right, glow_material in portal_specs:
        center = (left + right) * 0.5
        width = right - left
        parts = (
            ("LeftPost", unreal.Vector(left, -338.0, 65.0), unreal.Vector(12.0, 28.0, 130.0), materials["frame"]),
            ("RightPost", unreal.Vector(right, -338.0, 65.0), unreal.Vector(12.0, 28.0, 130.0), materials["frame"]),
            ("Header", unreal.Vector(center, -338.0, 139.0), unreal.Vector(width + 24.0, 28.0, 18.0), materials["frame"]),
            ("Scanner", unreal.Vector(center, -322.5, 127.0), unreal.Vector(width - 18.0, 3.0, 5.0), glow_material),
            ("LeftGuide", unreal.Vector(left + 7.0, -322.5, 65.0), unreal.Vector(3.0, 3.0, 104.0), glow_material),
            ("RightGuide", unreal.Vector(right - 7.0, -322.5, 65.0), unreal.Vector(3.0, 3.0, 104.0), glow_material),
        )
        for suffix, location, size, material in parts:
            label = f"SortingPortal_{portal_name}_{suffix}"
            portal_parts.append(ensure_block(label, location, size, material).get_actor_label())

    return {
        "actor": unified_actor.get_actor_label(),
        "mesh": unified_mesh.get_path_name(),
        "source_render_components_hidden": sorted(component.get_name() for component in source_components),
        "continuous_wall_parts": [unified_actor.get_actor_label(), *sorted(wall_parts)],
        "portal_parts": sorted(portal_parts),
    }


def create_qualcomm_wall_billboard(
    portal_materials: dict[str, unreal.Material],
) -> dict[str, object]:
    """Mount a sharp, color-stable Qualcomm Dragonwing sign on the north wall."""
    source_path = Path(unreal.Paths.project_dir()) / QUALCOMM_BILLBOARD_SOURCE
    if not source_path.is_file():
        raise RuntimeError(f"Missing Qualcomm billboard source: {source_path}")

    # Reimport on every visual-finish pass so changing the project-owned brand
    # source cannot leave stale pixels in an already-created Unreal texture.
    import_task = unreal.AssetImportTask()
    set_property(import_task, "filename", str(source_path))
    set_property(import_task, "destination_path", MATERIAL_FOLDER)
    set_property(import_task, "destination_name", "T_QualcommLogo_4K")
    set_property(import_task, "replace_existing", True)
    set_property(import_task, "automated", True)
    set_property(import_task, "save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([import_task])
    texture = unreal.EditorAssetLibrary.load_asset(QUALCOMM_BILLBOARD_TEXTURE)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError("Could not import the cropped Dragonwing billboard texture")
    set_property(texture, "srgb", True)
    set_property(texture, "compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)
    set_property(texture, "lod_group", unreal.TextureGroup.TEXTUREGROUP_WORLD)
    set_property(texture, "never_stream", False)
    set_property(texture, "max_texture_size", 4096)
    unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False)

    def ensure_material(asset_path: str) -> unreal.Material:
        material = unreal.EditorAssetLibrary.load_asset(asset_path)
        if not isinstance(material, unreal.Material):
            material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                asset_path.rsplit("/", 1)[-1], MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
            )
        if not isinstance(material, unreal.Material):
            raise RuntimeError(f"Could not create {asset_path}")
        unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
        return material

    white_material = ensure_material(QUALCOMM_BILLBOARD_WHITE)
    white = expression(white_material, unreal.MaterialExpressionConstant3Vector, -360, -80)
    set_property(white, "constant", unreal.LinearColor(0.94, 0.94, 0.94, 1.0))
    connect_output(white, "", unreal.MaterialProperty.MP_BASE_COLOR)
    # A neutral emissive fill prevents red/green signal washes from tinting
    # the requested white sign background. Keep the lit base/specular response
    # for edge definition; this is not a fullbright replacement shader.
    white_emissive_strength = expression(white_material, unreal.MaterialExpressionConstant, -120, -40)
    set_property(white_emissive_strength, "r", 24.0)
    white_emissive = expression(white_material, unreal.MaterialExpressionMultiply, 120, -70)
    connect(white, "", white_emissive, "A")
    connect(white_emissive_strength, "", white_emissive, "B")
    connect_output(white_emissive, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    white_roughness = expression(white_material, unreal.MaterialExpressionConstant, -360, 70)
    set_property(white_roughness, "r", 0.30)
    connect_output(white_roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    white_specular = expression(white_material, unreal.MaterialExpressionConstant, -360, 180)
    set_property(white_specular, "r", 0.32)
    connect_output(white_specular, "", unreal.MaterialProperty.MP_SPECULAR)
    unreal.MaterialEditingLibrary.layout_material_expressions(white_material)
    unreal.MaterialEditingLibrary.recompile_material(white_material)
    unreal.EditorAssetLibrary.save_loaded_asset(white_material, only_if_is_dirty=False)

    logo_material = ensure_material(QUALCOMM_BILLBOARD_LOGO)
    set_property(logo_material, "two_sided", True)
    logo_sample = expression(logo_material, unreal.MaterialExpressionTextureSample, -620, -40)
    set_property(logo_sample, "texture", texture)
    logo_white = expression(logo_material, unreal.MaterialExpressionConstant3Vector, -620, -230)
    set_property(logo_white, "constant", unreal.LinearColor(0.94, 0.94, 0.94, 1.0))
    logo_color = expression(logo_material, unreal.MaterialExpressionLinearInterpolate, -260, -70)
    connect(logo_white, "", logo_color, "A")
    connect(logo_sample, "RGB", logo_color, "B")
    connect(logo_sample, "A", logo_color, "Alpha")
    connect_output(logo_color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    logo_emissive_strength = expression(logo_material, unreal.MaterialExpressionConstant, -20, 10)
    set_property(logo_emissive_strength, "r", 24.0)
    logo_emissive = expression(logo_material, unreal.MaterialExpressionMultiply, 220, -70)
    connect(logo_color, "", logo_emissive, "A")
    connect(logo_emissive_strength, "", logo_emissive, "B")
    connect_output(logo_emissive, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    logo_roughness = expression(logo_material, unreal.MaterialExpressionConstant, -260, 100)
    set_property(logo_roughness, "r", 0.30)
    connect_output(logo_roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    logo_specular = expression(logo_material, unreal.MaterialExpressionConstant, -260, 210)
    set_property(logo_specular, "r", 0.32)
    connect_output(logo_specular, "", unreal.MaterialProperty.MP_SPECULAR)
    unreal.MaterialEditingLibrary.layout_material_expressions(logo_material)
    unreal.MaterialEditingLibrary.recompile_material(logo_material)
    unreal.EditorAssetLibrary.save_loaded_asset(logo_material, only_if_is_dirty=False)

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cube = load_required("/Engine/BasicShapes/Cube", unreal.StaticMesh)
    plane = load_required("/Engine/BasicShapes/Plane", unreal.StaticMesh)

    def ensure_actor(
        label: str,
        mesh: unreal.StaticMesh,
        location: unreal.Vector,
        rotation: unreal.Rotator,
        scale: unreal.Vector,
        material: unreal.Material,
        cast_shadow: bool,
    ) -> unreal.StaticMeshActor:
        actor = next(
            (
                candidate
                for candidate in actor_subsystem.get_all_level_actors()
                if candidate.get_actor_label() == label
            ),
            None,
        )
        if not isinstance(actor, unreal.StaticMeshActor):
            actor = actor_subsystem.spawn_actor_from_object(mesh, location, rotation)
            if not isinstance(actor, unreal.StaticMeshActor):
                raise RuntimeError(f"Could not create billboard part {label}")
            actor.set_actor_label(label)
        actor.set_actor_location(location, False, False)
        actor.set_actor_rotation(rotation, False)
        actor.set_actor_scale3d(scale)
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if component is None:
            raise RuntimeError(f"Billboard part {label} has no static-mesh component")
        component.set_static_mesh(mesh)
        component.set_material(0, material)
        component.set_collision_profile_name("NoCollision")
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        set_property(component, "cast_shadow", cast_shadow)
        return actor

    # The wall face is Y=-344. The backing slightly intersects it, while the
    # white board and logo sit progressively forward to avoid z-fighting.
    backing = ensure_actor(
        "QualcommWallBillboard_Back",
        cube,
        unreal.Vector(80.0, -342.5, 320.0),
        unreal.Rotator(),
        unreal.Vector(5.70, 0.05, 1.45),
        portal_materials["frame"],
        True,
    )
    board = ensure_actor(
        "QualcommWallBillboard_White",
        cube,
        unreal.Vector(80.0, -339.0, 320.0),
        unreal.Rotator(),
        unreal.Vector(5.58, 0.03, 1.33),
        white_material,
        True,
    )
    logo = ensure_actor(
        "QualcommWallBillboard_Logo",
        plane,
        unreal.Vector(80.0, -336.8, 320.0),
        unreal.Rotator(roll=-90.0, pitch=0.0, yaw=0.0),
        # BasicShapes/Plane maps local +Y downward after the -90 degree wall
        # rotation, so invert that scale to keep the logo text upright.
        unreal.Vector(5.20, -1.306, 1.0),
        logo_material,
        False,
    )
    return {
        "source": str(source_path),
        "source_dimensions": [1019, 256],
        "texture": texture.get_path_name(),
        "materials": [white_material.get_path_name(), logo_material.get_path_name()],
        "actors": [backing.get_actor_label(), board.get_actor_label(), logo.get_actor_label()],
        "board_size_cm": [570.0, 5.0, 145.0],
        "logo_size_cm": [520.0, 130.6],
        "location": [80.0, -336.8, 320.0],
    }


def create_presentation_backdrop() -> dict[str, object]:
    """Lift only the empty presentation surround from black to neutral grey."""
    material = unreal.EditorAssetLibrary.load_asset(BACKDROP_MATERIAL)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_PresentationBackdrop", MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
        if not isinstance(material, unreal.Material):
            raise RuntimeError("Could not create the presentation backdrop material")

    # This shader is used only by the large presentation plane and sky shell;
    # it does not relight or recolor any warehouse geometry. A shallow
    # world-space gradient prevents the empty surround from reading as black.
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
    set_property(material, "shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    set_property(material, "two_sided", True)

    world_position = expression(material, unreal.MaterialExpressionWorldPosition, -900, -60)
    bottom_z = expression(material, unreal.MaterialExpressionConstant, -680, 90)
    set_property(bottom_z, "r", -15000.0)
    relative_height = expression(material, unreal.MaterialExpressionSubtract, -440, -40)
    connect(world_position, "Z", relative_height, "A")
    connect(bottom_z, "", relative_height, "B")
    gradient_height = expression(material, unreal.MaterialExpressionConstant, -440, 110)
    set_property(gradient_height, "r", 30000.0)
    normalized_height = expression(material, unreal.MaterialExpressionDivide, -210, -40)
    connect(relative_height, "", normalized_height, "A")
    connect(gradient_height, "", normalized_height, "B")
    gradient = expression(material, unreal.MaterialExpressionSaturate, 20, -40)
    connect(normalized_height, "", gradient, "None")

    # Values are scene-referred luminance because the demo uses locked manual
    # exposure. Keep the range deliberately restrained: charcoal grey below,
    # softly lifting toward light grey above, with no white hotspot.
    bottom_color = expression(material, unreal.MaterialExpressionConstant3Vector, -210, -250)
    set_property(bottom_color, "constant", unreal.LinearColor(45.0, 45.0, 45.0, 1.0))
    top_color = expression(material, unreal.MaterialExpressionConstant3Vector, -210, 190)
    set_property(top_color, "constant", unreal.LinearColor(90.0, 90.0, 90.0, 1.0))
    color = expression(material, unreal.MaterialExpressionLinearInterpolate, 270, -40)
    connect(bottom_color, "", color, "A")
    connect(top_color, "", color, "B")
    connect(gradient, "", color, "Alpha")
    connect_output(color, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    unreal.MaterialEditingLibrary.delete_unused_expressions(material)
    unreal.MaterialEditingLibrary.layout_material_expressions(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)

    plane = load_required("/Engine/BasicShapes/Plane", unreal.StaticMesh)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = next(
        (value for value in actor_subsystem.get_all_level_actors() if value.get_actor_label() == "PresentationBackdrop"),
        None,
    )
    location = unreal.Vector(0.0, 0.0, -12.0)
    if actor is None:
        actor = actor_subsystem.spawn_actor_from_object(plane, location, unreal.Rotator())
        if actor is None:
            raise RuntimeError("Could not create the presentation backdrop actor")
        actor.set_actor_label("PresentationBackdrop")
    actor.set_actor_location(location, False, False)
    actor.set_actor_scale3d(unreal.Vector(300.0, 300.0, 1.0))
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if component is None:
        raise RuntimeError("PresentationBackdrop has no static-mesh component")
    component.set_material(0, material)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    set_property(component, "cast_shadow", False)
    sphere = load_required("/Engine/BasicShapes/Sphere", unreal.StaticMesh)
    sky_actor = next(
        (value for value in actor_subsystem.get_all_level_actors() if value.get_actor_label() == "PresentationSkyShell"),
        None,
    )
    if sky_actor is None:
        sky_actor = actor_subsystem.spawn_actor_from_object(sphere, unreal.Vector(), unreal.Rotator())
        if sky_actor is None:
            raise RuntimeError("Could not create the presentation sky shell")
        sky_actor.set_actor_label("PresentationSkyShell")
    sky_actor.set_actor_location(unreal.Vector(), False, False)
    sky_actor.set_actor_scale3d(unreal.Vector(300.0, 300.0, 300.0))
    sky_component = sky_actor.get_component_by_class(unreal.StaticMeshComponent)
    if sky_component is None:
        raise RuntimeError("PresentationSkyShell has no static-mesh component")
    sky_component.set_material(0, material)
    sky_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    set_property(sky_component, "cast_shadow", False)
    return {
        "actor": actor.get_actor_label(),
        "sky_actor": sky_actor.get_actor_label(),
        "material": material.get_path_name(),
        "gradient": {
            "bottom_z": -15000.0,
            "top_z": 15000.0,
            "bottom_luminance": 45.0,
            "top_luminance": 90.0,
        },
        "location": [location.x, location.y, location.z],
        "scale": [300.0, 300.0, 1.0],
    }


def apply_visual_finish() -> dict[str, object]:
    material = create_red_epoxy_material()
    material_report = assign_materials(material)
    hero_materials = create_hero_materials()
    hero_material_report = assign_hero_materials(hero_materials)
    dragonwing_materials = create_dragonwing_brand_materials()
    dragonwing_material_report = assign_dragonwing_brand_materials(dragonwing_materials)
    stack_lens_instances = create_stack_lens_materials()
    stack_lens_report = assign_stack_lens_materials(stack_lens_instances)
    stack_halo_material = create_stack_halo_material()
    light_report = configure_fill_lights()
    portal_materials = create_machine_portal_materials()
    portal_wall_report = create_unified_portal_wall(portal_materials)
    billboard_report = create_qualcomm_wall_billboard(portal_materials)
    backdrop_report = create_presentation_backdrop()
    return {
        "materials": material_report,
        "hero_materials": {
            "assets": {name: value.get_path_name() for name, value in hero_materials.items()},
            "assigned_components": hero_material_report,
        },
        "dragonwing_branding": {
            "srgb_hex": "#32017E",
            "assets": {name: value.get_path_name() for name, value in dragonwing_materials.items()},
            "assigned_components": dragonwing_material_report,
        },
        "stack_lenses": stack_lens_report,
        "stack_halo": stack_halo_material.get_path_name(),
        "fill_lights": light_report,
        "portal_wall": portal_wall_report,
        "qualcomm_billboard": billboard_report,
        "presentation_backdrop": backdrop_report,
    }


def main() -> None:
    unreal.EditorLoadingAndSavingUtils.load_map(f"{CONTENT_ROOT}/WarehouseConveyor")
    report = apply_visual_finish()
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.save_current_level():
        raise RuntimeError("Could not save the WarehouseConveyor level")
    unreal.EditorAssetLibrary.save_directory(CONTENT_ROOT, only_if_is_dirty=False, recursive=True)
    output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "visual_finish.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_VISUAL_FINISH={output}")


if __name__ == "__main__":
    main()
