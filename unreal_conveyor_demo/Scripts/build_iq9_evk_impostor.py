"""Replace the detailed IQ9 EVK actor with a five-view textured box impostor."""

from __future__ import annotations

import json
import math
from pathlib import Path

import unreal


LEVEL = "/Game/ConveyorRuntime/WarehouseConveyor"
RUNTIME_FOLDER = "/Game/IQ9EVK/Runtime"
IMPOSTOR_FOLDER = f"{RUNTIME_FOLDER}/Impostor"
CONTENT_SOURCE = Path(unreal.Paths.project_dir()) / "ContentSource" / "IQ9EVK" / "Impostor"
ACTOR_LABEL = "IQ9 EVK tabletop"
FACE_ACTOR_TAG = "Qai.IQ9EVK.ImpostorFace"
PACKING_TABLE_MESH = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/SM_packing_table"
)
LED_MATERIAL = f"{RUNTIME_FOLDER}/M_IQ9_LED"
ENGINE_CUBE = "/Engine/BasicShapes/Cube.Cube"
ENGINE_PLANE = "/Engine/BasicShapes/Plane.Plane"

# Qualcomm's public specification gives the PCB as 100 x 100 x 2 mm. The
# supplied 10-75699 STEP assembly measures the bottom enclosure at roughly
# 106 x 106 x 43 mm; the previous 57 mm value was the full assembly height and
# incorrectly turned the exposed-electronics volume into solid enclosure.
SIZE_X = 10.60
SIZE_Y = 10.60
SIZE_Z = 4.30
HALF_X = SIZE_X * 0.5
HALF_Y = SIZE_Y * 0.5
HALF_Z = SIZE_Z * 0.5
FACE_GAP = 0.025

TEXTURE_FILES = {
    "Top": "iq9_top.png",
    "Front": "iq9_front.png",
    "Back": "iq9_back.png",
    "Right": "iq9_right.png",
    "Left": "iq9_left.png",
}

OLD_RUNTIME_ASSETS = (
    f"{RUNTIME_FOLDER}/SM_SM_IQ9_EVK_Core_Proxy",
    f"{RUNTIME_FOLDER}/MI_SM_IQ9_EVK_Core_Proxy",
    f"{RUNTIME_FOLDER}/T_SM_IQ9_EVK_Core_Proxy_BaseColor",
    f"{RUNTIME_FOLDER}/T_SM_IQ9_EVK_Core_Proxy_Normal",
    f"{RUNTIME_FOLDER}/T_SM_IQ9_EVK_Core_Proxy_Roughness",
)


def set_property(obj, name: str, value) -> None:
    try:
        obj.set_editor_property(name, value)
    except Exception as exc:
        unreal.log_warning(f"IQ9_IMPOSTOR property={name} unavailable: {exc}")


def load_required(path: str, expected_type: type):
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(asset, expected_type):
        raise RuntimeError(f"Missing {expected_type.__name__}: {path}")
    return asset


def rotate_yaw(vector: unreal.Vector, yaw_degrees: float) -> unreal.Vector:
    angle = math.radians(yaw_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return unreal.Vector(
        vector.x * cosine - vector.y * sine,
        vector.x * sine + vector.y * cosine,
        vector.z,
    )


def import_texture(face: str, filename: str) -> unreal.Texture2D:
    source = CONTENT_SOURCE / filename
    if not source.is_file():
        raise RuntimeError(f"Missing processed IQ9 face image: {source}")
    asset_name = f"T_IQ9_Impostor_{face}"
    asset_path = f"{IMPOSTOR_FOLDER}/{asset_name}"
    task = unreal.AssetImportTask()
    set_property(task, "filename", str(source))
    set_property(task, "destination_path", IMPOSTOR_FOLDER)
    set_property(task, "destination_name", asset_name)
    set_property(task, "replace_existing", True)
    set_property(task, "automated", True)
    set_property(task, "save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"Could not import IQ9 face texture: {asset_path}")
    set_property(texture, "srgb", True)
    set_property(texture, "compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)
    set_property(texture, "lod_group", unreal.TextureGroup.TEXTUREGROUP_WORLD)
    set_property(texture, "never_stream", False)
    # These source plates are already tightly cropped.  Retaining 1K on every
    # face keeps the small connector labels legible without making the prop a
    # meaningful part of the scene's texture budget.
    set_property(texture, "max_texture_size", 1024)
    unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False)
    return texture


def material_expression(material: unreal.Material, cls: type, x: int, y: int):
    value = unreal.MaterialEditingLibrary.create_material_expression(
        material, cls.static_class(), x, y)
    if value is None:
        raise RuntimeError(f"Could not create {cls.__name__} in {material.get_path_name()}")
    return value


def create_face_material(face: str, texture: unreal.Texture2D) -> unreal.Material:
    name = f"M_IQ9_Impostor_{face}"
    path = f"{IMPOSTOR_FOLDER}/{name}"
    material = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, IMPOSTOR_FOLDER, unreal.Material, unreal.MaterialFactoryNew())
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"Could not create {path}")
    set_property(material, "two_sided", True)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
    image = material_expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -460, -100)
    set_property(image, "parameter_name", "FaceTexture")
    set_property(image, "texture", texture)
    set_property(image, "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    unreal.MaterialEditingLibrary.connect_material_property(
        image, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    roughness = material_expression(material, unreal.MaterialExpressionConstant, -160, 90)
    set_property(roughness, "r", 0.60 if face == "Top" else 0.48)
    unreal.MaterialEditingLibrary.connect_material_property(
        roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    specular = material_expression(material, unreal.MaterialExpressionConstant, -160, 190)
    set_property(specular, "r", 0.38)
    unreal.MaterialEditingLibrary.connect_material_property(
        specular, "", unreal.MaterialProperty.MP_SPECULAR)
    unreal.MaterialEditingLibrary.layout_material_expressions(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def create_body_material() -> unreal.Material:
    name = "M_IQ9_ImpostorBody"
    path = f"{IMPOSTOR_FOLDER}/{name}"
    material = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(material, unreal.Material):
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, IMPOSTOR_FOLDER, unreal.Material, unreal.MaterialFactoryNew())
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"Could not create {path}")
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
    color = material_expression(material, unreal.MaterialExpressionConstant3Vector, -350, -70)
    # #2A2AEA converted approximately from sRGB to linear material space.
    set_property(color, "constant", unreal.LinearColor(0.023, 0.023, 0.823, 1.0))
    unreal.MaterialEditingLibrary.connect_material_property(
        color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    roughness = material_expression(material, unreal.MaterialExpressionConstant, -350, 80)
    set_property(roughness, "r", 0.52)
    unreal.MaterialEditingLibrary.connect_material_property(
        roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    unreal.MaterialEditingLibrary.layout_material_expressions(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    return material


def configure_mesh_component(
    component: unreal.StaticMeshComponent,
    material: unreal.MaterialInterface,
    cast_shadow: bool,
) -> None:
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_collision_profile_name("NoCollision")
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_material(0, material)
    set_property(component, "cast_shadow", cast_shadow)


def main() -> None:
    unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cube_mesh = load_required(ENGINE_CUBE, unreal.StaticMesh)
    plane_mesh = load_required(ENGINE_PLANE, unreal.StaticMesh)
    table_mesh = load_required(PACKING_TABLE_MESH, unreal.StaticMesh)
    load_required(LED_MATERIAL, unreal.Material)

    face_materials = {
        face: create_face_material(face, import_texture(face, filename))
        for face, filename in TEXTURE_FILES.items()
    }
    body_material = create_body_material()

    for actor in actor_subsystem.get_all_level_actors():
        if actor.get_actor_label() == ACTOR_LABEL or actor.actor_has_tag(unreal.Name(FACE_ACTOR_TAG)):
            actor_subsystem.destroy_actor(actor)
    table_component = None
    for actor in actor_subsystem.get_all_level_actors():
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            if component.static_mesh == table_mesh:
                table_component = component
                break
        if table_component:
            break
    if table_component is None:
        raise RuntimeError("Could not find sorting-station desktop")

    table_origin, table_extent, _ = unreal.SystemLibrary.get_component_bounds(table_component)
    table_rotation = table_component.get_world_rotation()
    tabletop_offset = rotate_yaw(unreal.Vector(15.0, -10.0, 0.0), table_rotation.yaw)
    center = unreal.Vector(
        table_origin.x + tabletop_offset.x,
        table_origin.y + tabletop_offset.y,
        table_origin.z + table_extent.z + HALF_Z + 0.25,
    )
    yaw = table_rotation.yaw

    root_actor = actor_subsystem.spawn_actor_from_object(
        cube_mesh,
        center,
        unreal.Rotator(pitch=0.0, yaw=yaw, roll=0.0),
    )
    if root_actor is None:
        raise RuntimeError("Could not spawn IQ9 impostor body")
    root_actor.set_actor_label(ACTOR_LABEL)
    root_actor.set_actor_scale3d(unreal.Vector(SIZE_X / 100.0, SIZE_Y / 100.0, SIZE_Z / 100.0))
    set_property(root_actor, "tags", [unreal.Name("Qai.IQ9EVK.Actor")])
    root_component = root_actor.get_component_by_class(unreal.StaticMeshComponent)
    if not isinstance(root_component, unreal.StaticMeshComponent):
        raise RuntimeError("IQ9 impostor body has no mesh component")
    configure_mesh_component(root_component, body_material, True)
    set_property(root_component, "component_tags", [
        unreal.Name("Qai.IQ9EVK"),
        unreal.Name("Qai.DynamicProp"),
    ])

    faces = (
        ("Top", unreal.Vector(0.0, 0.0, HALF_Z + FACE_GAP), unreal.Rotator(pitch=0.0, yaw=yaw, roll=0.0), unreal.Vector(SIZE_X / 100.0, SIZE_Y / 100.0, 1.0)),
        ("Front", unreal.Vector(HALF_X + FACE_GAP, 0.0, 0.0), unreal.Rotator(pitch=90.0, yaw=yaw, roll=0.0), unreal.Vector(SIZE_Z / 100.0, SIZE_Y / 100.0, 1.0)),
        ("Back", unreal.Vector(-HALF_X - FACE_GAP, 0.0, 0.0), unreal.Rotator(pitch=-90.0, yaw=yaw, roll=0.0), unreal.Vector(SIZE_Z / 100.0, SIZE_Y / 100.0, 1.0)),
        ("Right", unreal.Vector(0.0, HALF_Y + FACE_GAP, 0.0), unreal.Rotator(pitch=0.0, yaw=yaw, roll=-90.0), unreal.Vector(SIZE_X / 100.0, SIZE_Z / 100.0, 1.0)),
        ("Left", unreal.Vector(0.0, -HALF_Y - FACE_GAP, 0.0), unreal.Rotator(pitch=0.0, yaw=yaw, roll=90.0), unreal.Vector(SIZE_X / 100.0, SIZE_Z / 100.0, 1.0)),
    )
    face_actors = []
    for face, local_offset, world_rotation, scale in faces:
        world_offset = rotate_yaw(local_offset, yaw)
        actor = actor_subsystem.spawn_actor_from_object(
            plane_mesh,
            center + world_offset,
            world_rotation,
        )
        if actor is None:
            raise RuntimeError(f"Could not spawn IQ9 {face} face")
        actor.set_actor_label(f"IQ9 impostor {face.lower()} face")
        actor.set_actor_scale3d(scale)
        set_property(actor, "tags", [unreal.Name(FACE_ACTOR_TAG)])
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if not isinstance(component, unreal.StaticMeshComponent):
            raise RuntimeError(f"IQ9 {face} face has no mesh component")
        configure_mesh_component(component, face_materials[face], False)
        actor.attach_to_actor(
            root_actor,
            unreal.Name("None"),
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            False,
        )
        face_actors.append(actor)

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.save_current_level():
        raise RuntimeError("Could not save WarehouseConveyor after IQ9 impostor placement")
    unreal.EditorAssetLibrary.save_directory(
        RUNTIME_FOLDER, only_if_is_dirty=False, recursive=True)

    # The CAD-derived 36k-vertex mesh is now only an offline bake source. Do
    # not keep it referenced or cooked after the five appearance maps exist.
    deleted = []
    for asset_path in OLD_RUNTIME_ASSETS:
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
            if not unreal.EditorAssetLibrary.delete_asset(asset_path):
                raise RuntimeError(f"Could not remove old IQ9 runtime asset: {asset_path}")
            deleted.append(asset_path)

    report = {
        "source": "resources/DP25-73418-2_RevB/10-75699-RB8-PVT.stp",
        "configuration": "top_photo_procedural_enclosure_impostor",
        "source_views": [str(CONTENT_SOURCE / filename) for filename in TEXTURE_FILES.values()],
        "dimensions_cm": [SIZE_X, SIZE_Y, SIZE_Z],
        "geometry": {"cube_triangles": 12, "face_plane_triangles": 10, "total_triangles": 22},
        "textures": {
            "top": "1024x1024 perspective-corrected",
            "front_back": "512x1024 UV-corrected",
            "left_right": "1024x512",
        },
        "appearance_source": (
            "user-photographed orthographic PCB top plus simplified "
            "procedural enclosure sides"
        ),
        "palette": {
            "pcb": "satin black",
            "memory_slots": "beige",
            "enclosure": "Qualcomm blue #2A2AEA",
        },
        "leds": ["steady green", "random blinking amber"],
        "actor": ACTOR_LABEL,
        "location": list(center.to_tuple()),
        "rotation_yaw": yaw,
        "physics_tag": "Qai.DynamicProp",
        "face_actor_count": len(face_actors),
        "removed_runtime_assets": deleted,
    }
    output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "iq9_evk.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_IQ9_IMPOSTOR={output}")


if __name__ == "__main__":
    main()
