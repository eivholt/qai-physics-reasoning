"""Import the audited lean USD stage as native, cooked Unreal assets.

Run with UnrealEditor-Cmd.exe and ``-ExecutePythonScript``.  The script is
intentionally deterministic: it creates a fresh generated content subtree,
imports only the conveyor prim, consumes portable baked PBR materials, saves
the generated level, applies conservative mesh/texture cook settings, and writes an audit
manifest under Saved/ImportAudit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_finish import apply_visual_finish


PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())).resolve()
STAGE = Path(
    os.environ.get(
        "QAI_CONVEYOR_LEAN_STAGE",
        str(PROJECT_DIR / "Saved" / "BakedStageGenerated" / "warehouse_conveyor_baked.usdc"),
    )
).resolve()
CONTENT_ROOT = "/Game/ConveyorRuntime"
LEVEL_PATH = f"{CONTENT_ROOT}/WarehouseConveyor"
STAGING_LEVEL_PATH = "/Game/QaiConveyorImportStaging"
AUDIT_PATH = PROJECT_DIR / "Saved" / "ImportAudit" / "optimized_asset_manifest.json"

REQUIRED_RUNTIME_NAMES = {
    "Forklift1",
    "Forklift1Pallet",
    "Parcel1",
    "DetectorEndline",
    "Worker1",
    "Worker2",
    "DriverMount",
    "male_adult_construction_05",
}

# Import the non-moving scene in deliberately sized chunks.  In particular the
# six conveyor modules stay separate, so each can use Nanite without exceeding
# its 64-material limit.  Parcels are imported in a second, non-collapsing pass
# so all eight retain independently movable roots.
COLLAPSED_PRIMS = [
    "/World/CodexPoC/ConveyorSafety/Physics",
    "/World/CodexPoC/ConveyorSafety/Floor",
    "/World/CodexPoC/ConveyorSafety/FloorFinish",
    "/World/CodexPoC/ConveyorSafety/BehindWallFloor",
    "/World/CodexPoC/ConveyorSafety/Shell",
    "/World/CodexPoC/ConveyorSafety/ClearanceZone",
    "/World/CodexPoC/ConveyorSafety/Conveyor/Modules",
    "/World/CodexPoC/ConveyorSafety/Storage",
    "/World/CodexPoC/ConveyorSafety/packing_table",
    "/World/CodexPoC/ConveyorSafety/Forklifts",
    "/World/CodexPoC/ConveyorSafety/DynamicCargo",
    "/World/CodexPoC/ConveyorSafety/StackLight",
    "/World/CodexPoC/ConveyorSafety/StackLight_01",
    "/World/CodexPoC/ConveyorSafety/StackLight_02",
    "/World/CodexPoC/ConveyorSafety/Lighting",
    "/World/CodexPoC/ConveyorSafety/Workers",
    "/World/CodexPoC/ConveyorSafety/Cameras/DetectorEndline",
]
PARCEL_PRIMS = ["/World/CodexPoC/ConveyorSafety/Conveyor/Parcels"]


def set_property(obj: object, name: str, value: object) -> None:
    try:
        obj.set_editor_property(name, value)  # type: ignore[attr-defined]
    except Exception as error:
        raise RuntimeError(f"Could not set {type(obj).__name__}.{name}={value!r}: {error}") from error


def prepare_level() -> None:
    asset_library = unreal.EditorAssetLibrary
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    # The generated warehouse is also the editor startup map.  Move to a tiny
    # level outside the generated subtree so repeat imports can delete every
    # prior asset instead of leaving the loaded map behind.
    if asset_library.does_asset_exist(STAGING_LEVEL_PATH):
        if not level_subsystem.load_level(STAGING_LEVEL_PATH):
            raise RuntimeError(f"Could not load import staging level {STAGING_LEVEL_PATH}")
    elif not level_subsystem.new_level(STAGING_LEVEL_PATH):
        raise RuntimeError(f"Could not create import staging level {STAGING_LEVEL_PATH}")
    if asset_library.does_directory_exist(CONTENT_ROOT):
        if not asset_library.delete_directory(CONTENT_ROOT):
            raise RuntimeError(f"Could not replace generated content directory {CONTENT_ROOT}")
    if not level_subsystem.new_level(LEVEL_PATH):
        raise RuntimeError(f"Could not create generated level {LEVEL_PATH}")


def make_options(prims: list[str], collapse: bool) -> unreal.UsdStageImportOptions:
    options = unreal.UsdStageImportOptions()
    settings = {
        "import_actors": True,
        "import_geometry": True,
        "import_skeletal_animations": True,
        "import_level_sequences": False,
        "import_materials": True,
        "import_groom_assets": False,
        "import_sparse_volume_textures": False,
        "import_sounds": False,
        "import_only_used_materials": True,
        "prims_to_import": prims,
        "purposes_to_import": 7,  # default, render and proxy
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
        "use_prim_kinds_for_collapsing": collapse,
        "merge_identical_material_slots": True,
        "interpret_lods": True,
    }
    for name, value in settings.items():
        set_property(options, name, value)
    return options


def import_stage() -> list[str]:
    tasks = []
    for destination, prims, collapse in (
        (f"{CONTENT_ROOT}/Scene", COLLAPSED_PRIMS, True),
        (f"{CONTENT_ROOT}/Parcels", PARCEL_PRIMS, False),
    ):
        task = unreal.AssetImportTask()
        set_property(task, "automated", True)
        set_property(task, "save", False)
        set_property(task, "replace_existing", True)
        set_property(task, "replace_existing_settings", True)
        set_property(task, "filename", str(STAGE))
        set_property(task, "destination_path", destination)
        set_property(task, "options", make_options(prims, collapse))
        tasks.append(task)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    imported = [
        path
        for task in tasks
        for path in list(task.get_editor_property("imported_object_paths"))
    ]
    if not imported:
        raise RuntimeError("USD importer returned no native Unreal assets")
    return imported


def optimize_generated_assets() -> dict[str, object]:
    assets = unreal.EditorAssetLibrary.list_assets(CONTENT_ROOT, recursive=True, include_folder=False)
    static_mesh_subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    asset_counts: dict[str, int] = {}
    texture_source_bytes = 0
    normal_textures = 0
    scalar_textures = 0
    nanite_meshes = 0
    high_precision_hero_meshes = 0
    static_mesh_triangles = 0
    for path in assets:
        asset = unreal.EditorAssetLibrary.load_asset(path)
        if asset is None:
            continue
        class_name = asset.get_class().get_name()
        asset_counts[class_name] = asset_counts.get(class_name, 0) + 1
        dirty = False
        if isinstance(asset, unreal.Texture2D):
            # Ultra bakes are 1024px. Keep enough resolution for labels,
            # corrugation and material grain without letting upstream source
            # outliers dominate portable builds. The forklift atlas is eight
            # 1024px UDIM tiles laid out horizontally.
            texture_name = asset.get_name().lower()
            max_texture_size = 8192 if "forklift" in texture_name and "atlas" in texture_name else 2048
            current_max = int(asset.get_editor_property("max_texture_size"))
            if current_max != max_texture_size:
                set_property(asset, "max_texture_size", max_texture_size)
                dirty = True
            # The MDL distiller emits semantically named maps. Give Unreal
            # enough information to choose compact platform-native formats:
            # BC5 for normals and a single linear channel for scalar PBR data.
            # Base color and emissive maps remain sRGB RGBA.
            if "normal" in texture_name:
                set_property(asset, "compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
                set_property(asset, "srgb", False)
                normal_textures += 1
                dirty = True
            elif any(
                token in texture_name
                for token in (
                    "roughness",
                    "metallic",
                    "opacity",
                    "anisotropy",
                    "clearcoat_weight",
                    "clearcoat_roughness",
                )
            ):
                set_property(asset, "compression_settings", unreal.TextureCompressionSettings.TC_GRAYSCALE)
                set_property(asset, "srgb", False)
                scalar_textures += 1
                dirty = True
            try:
                texture_source_bytes += int(asset.blueprint_get_size_x()) * int(asset.blueprint_get_size_y()) * 4
            except Exception:
                pass
        elif isinstance(asset, unreal.StaticMesh):
            set_property(asset, "allow_cpu_access", False)
            dirty = True
            # Machinery carries dense UDIM atlas UVs and glossy hard-surface
            # normals. Keep those vertex attributes at full precision so the
            # base-pass vertex shaders cannot introduce texture swimming or
            # faceted specular highlights when TSR resolves sub-pixel detail.
            normalized_path = path.lower().replace("\\", "/")
            if "/forklifts/" in normalized_path or "/conveyor/modules/" in normalized_path:
                build_settings = static_mesh_subsystem.get_lod_build_settings(asset, 0)
                set_property(build_settings, "use_full_precision_u_vs", True)
                set_property(build_settings, "use_high_precision_tangent_basis", True)
                static_mesh_subsystem.set_lod_build_settings(asset, 0, build_settings)
                high_precision_hero_meshes += 1
            try:
                triangles = int(asset.get_num_triangles(0))
                static_mesh_triangles += triangles
                nanite_settings = asset.get_editor_property("nanite_settings")
                nanite_settings.enabled = triangles >= 50000
                set_property(asset, "nanite_settings", nanite_settings)
                nanite_meshes += int(nanite_settings.enabled)
            except Exception as error:
                unreal.log_warning(f"Could not tune Nanite for {path}: {error}")
        if dirty:
            unreal.EditorAssetLibrary.save_loaded_asset(asset, only_if_is_dirty=False)
    return {
        "asset_counts": asset_counts,
        "asset_count": len(assets),
        "texture_uncompressed_estimate_bytes": texture_source_bytes,
        "normal_textures": normal_textures,
        "scalar_textures": scalar_textures,
        "static_mesh_triangles": static_mesh_triangles,
        "nanite_meshes": nanite_meshes,
        "high_precision_hero_meshes": high_precision_hero_meshes,
    }


def authored_parent_material(material: object | None) -> object | None:
    """Return the authored USD material behind a redundant import instance.

    Unreal's USD skeletal importer creates suffixed child instances for
    GeomSubset bindings. Those children override the parent's texture switches
    back to zero, rendering people and the forklift driver gray/black. The
    original material variations already exist as distinct, deduplicated USD
    materials, so a child named Parent_N has no authored visual meaning.
    """
    current = material
    while isinstance(current, unreal.MaterialInstanceConstant):
        parent = current.get_editor_property("parent")
        if not isinstance(parent, unreal.MaterialInstanceConstant):
            break
        # Authored imported materials are parented directly to an Engine
        # USDCore master. Any additional instance whose parent is inside the
        # generated content tree is an Unreal binding artifact, sometimes in
        # chains such as Material_2 -> Material_0 -> Material.
        if not parent.get_path_name().startswith(CONTENT_ROOT + "/"):
            break
        current = parent
    return current


def repair_imported_material_bindings() -> dict[str, int]:
    assets = unreal.EditorAssetLibrary.list_assets(CONTENT_ROOT, recursive=True, include_folder=False)
    repaired_slots = 0
    repaired_overrides = 0
    for asset_path in assets:
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        if isinstance(asset, unreal.StaticMesh):
            materials = list(asset.get_editor_property("static_materials"))
            dirty = False
            for slot in materials:
                current = slot.get_editor_property("material_interface")
                repaired = authored_parent_material(current)
                if repaired is not current:
                    slot.set_editor_property("material_interface", repaired)
                    repaired_slots += 1
                    dirty = True
            if dirty:
                asset.set_editor_property("static_materials", materials)
                unreal.EditorAssetLibrary.save_loaded_asset(asset, only_if_is_dirty=False)
        elif isinstance(asset, unreal.SkeletalMesh):
            materials = list(asset.get_editor_property("materials"))
            dirty = False
            for slot in materials:
                current = slot.get_editor_property("material_interface")
                repaired = authored_parent_material(current)
                if repaired is not current:
                    slot.set_editor_property("material_interface", repaired)
                    repaired_slots += 1
                    dirty = True
            if dirty:
                asset.set_editor_property("materials", materials)
                unreal.EditorAssetLibrary.save_loaded_asset(asset, only_if_is_dirty=False)

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    for actor in actors:
        for component in actor.get_components_by_class(unreal.MeshComponent):
            overrides = list(component.get_editor_property("override_materials"))
            dirty = False
            for index, current in enumerate(overrides):
                repaired = authored_parent_material(current)
                if repaired is not current:
                    overrides[index] = repaired
                    repaired_overrides += 1
                    dirty = True
            if dirty:
                component.set_editor_property("override_materials", overrides)
    return {
        "repaired_material_slots": repaired_slots,
        "repaired_component_overrides": repaired_overrides,
    }


def add_actor_tag(actor: unreal.Actor, tag: str) -> None:
    tags = list(actor.get_editor_property("tags"))
    if tag not in [str(value) for value in tags]:
        tags.append(tag)
        set_property(actor, "tags", tags)


def add_component_tag(component: unreal.SceneComponent, tag: str) -> None:
    tags = list(component.get_editor_property("component_tags"))
    if tag not in [str(value) for value in tags]:
        tags.append(tag)
        set_property(component, "component_tags", tags)


def actor_root_component(actor: unreal.Actor) -> unreal.SceneComponent | None:
    return actor.get_editor_property("root_component")


def make_subtree_movable(root: unreal.SceneComponent, components: list[unreal.SceneComponent]) -> None:
    for component in components:
        cursor = component
        while cursor is not None:
            if cursor == root:
                component.set_mobility(unreal.ComponentMobility.MOVABLE)
                break
            cursor = cursor.get_attach_parent()


def configure_runtime_bindings() -> dict[str, int]:
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    tagged_actors = 0
    tagged_components = 0
    animated_workers = 0
    seated_drivers = 0
    static_driver_component_count = 0
    dynamic_actor_labels = {"Forklift1", "Worker1", "Worker2"}
    component_tags = {
        "Forklift1": "Qai.Forklift1",
        **{f"Parcel{index}": f"Qai.Parcel{index}" for index in range(1, 9)},
        "Forklift1Pallet": "Qai.Forklift1Pallet",
        "Forklift1Carton": "Qai.Forklift1Carton",
        "Worker1": "Qai.Worker1",
        "Worker2": "Qai.Worker2",
        **{f"GreenLens_{index}": f"Qai.Green.{index}" for index in range(3)},
        **{f"AmberLens_{index}": f"Qai.Amber.{index}" for index in range(3)},
        **{f"RedLens_{index}": f"Qai.Red.{index}" for index in range(3)},
    }
    dynamic_component_prefixes = ("Parcel", "Forklift1", "Worker1", "Worker2")

    wheel_names = {
        "left_front_wheel": "LeftFront",
        "right_front_wheel": "RightFront",
        "left_back_wheel": "LeftRear",
        "right_back_wheel": "RightRear",
    }

    # UE 5.8 builds the shared USD seated pose in /Engine/Transient but emits a
    # warning instead of assigning it to either referenced driver component.
    # Promote that evaluated clip into the project and bind it explicitly. The
    # The retained driver uses the authored skeleton, so one promoted pose is
    # smaller and more deterministic than a transient duplicate.
    driver_animation = None
    for suffix in range(4):
        transient_path = f"/Engine/Transient.AS_SeatedDriverPose_{suffix}"
        transient_animation = unreal.find_object(None, transient_path, unreal.AnimSequence.static_class())
        if transient_animation is None:
            continue
        driver_animation = unreal.AssetToolsHelpers.get_asset_tools().duplicate_asset(
            "AS_SeatedDriverPose",
            f"{CONTENT_ROOT}/Animations",
            transient_animation,
        )
        if driver_animation is not None:
            unreal.log(
                f"QAI_DRIVER_POSE promoted={transient_path} "
                f"asset={driver_animation.get_path_name()}"
            )
            break
    if driver_animation is None:
        unreal.log(
            "QAI_DRIVER_POSE no transient clip found; expecting the USD stage "
            "to contain baked seated reference poses"
        )

    for actor in actors:
        label = actor.get_actor_label()
        components = actor.get_components_by_class(unreal.SceneComponent)
        if label in dynamic_actor_labels or label == "DetectorEndline":
            add_actor_tag(actor, f"Qai.{label}")
            tagged_actors += 1
        if label in dynamic_actor_labels:
            root = actor_root_component(actor)
            if root is not None:
                make_subtree_movable(root, components)

        # USD may put the skeletal component on a generated parent actor whose
        # label is unrelated to the referenced character prim.  Classify the
        # three people by their authored animation asset instead of actor label.
        # This also makes the audit prove that the seated-driver clip remains
        # bound after Unreal's USD hierarchy flattening.
        for skeletal_component in actor.get_components_by_class(unreal.SkeletalMeshComponent):
            animation_data = skeletal_component.get_editor_property("animation_data")
            animation_asset = animation_data.get_editor_property("anim_to_play")
            skeletal_mesh = skeletal_component.get_editor_property("skeletal_mesh_asset")
            mesh_path = skeletal_mesh.get_path_name().lower() if skeletal_mesh else ""
            is_driver_mesh = "male_adult_construction_05" in mesh_path
            if animation_asset is None and is_driver_mesh and driver_animation is not None:
                animation_asset = driver_animation
                animation_data.set_editor_property("anim_to_play", animation_asset)
            unreal.log(
                "QAI_CHARACTER_COMPONENT "
                f"actor={label} component={skeletal_component.get_name()} "
                f"mesh={skeletal_mesh.get_path_name() if skeletal_mesh else '<none>'} "
                f"animation={animation_asset.get_path_name() if animation_asset else '<none>'}"
            )
            if animation_asset is None and not is_driver_mesh:
                continue
            animation_name = animation_asset.get_name().lower() if animation_asset else ""
            is_walker = "walkforwardloop" in animation_name
            is_driver = is_driver_mesh or "seateddriverpose" in animation_name
            if not is_walker and not is_driver:
                continue
            if animation_asset is None:
                seated_drivers += 1
                continue
            skeletal_component.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
            animation_data.set_editor_property("saved_looping", is_walker)
            animation_data.set_editor_property("saved_playing", True)
            animation_data.set_editor_property("saved_play_rate", 1.0)
            animation_data.set_editor_property("saved_position", 0.0)
            set_property(skeletal_component, "animation_data", animation_data)
            # Initializing the single-node instance is required for components
            # that had no animation relationship when UE instantiated them.
            # Persisted AnimationData alone otherwise leaves drivers in bind
            # pose until another runtime call happens to initialize the proxy.
            skeletal_component.play_animation(animation_asset, is_walker)
            skeletal_component.set_component_tick_enabled(True)
            if is_driver:
                seated_drivers += 1
            else:
                animated_workers += 1

        # The driver is intentionally pre-skinned to his seated frame in USD.
        # He does not animate, so importing him as static meshes saves a
        # skeleton and a per-frame skinning evaluation.
        static_driver_components = []
        for static_component in actor.get_components_by_class(unreal.StaticMeshComponent):
            static_mesh = static_component.get_editor_property("static_mesh")
            mesh_path = static_mesh.get_path_name().lower() if static_mesh else ""
            if "male_adult_construction_05" in mesh_path:
                static_driver_components.append(static_component)
        if static_driver_components:
            static_driver_component_count += len(static_driver_components)
            for static_component in static_driver_components:
                static_component.set_mobility(unreal.ComponentMobility.MOVABLE)
            unreal.log(
                f"QAI_STATIC_DRIVER actor={label} components={len(static_driver_components)}"
            )

        for component in components:
            name = component.get_name()
            logical = name.rsplit("_", 1)[0]
            tag = component_tags.get(name) or component_tags.get(logical)
            if tag:
                add_component_tag(component, tag)
                tagged_components += 1
                if logical.startswith(dynamic_component_prefixes):
                    make_subtree_movable(component, components)

            wheel_role = next((role for prefix, role in wheel_names.items() if name.startswith(prefix)), None)
            if wheel_role:
                cursor = component
                forklift_index = None
                while cursor is not None:
                    ancestor_name = cursor.get_name()
                    if ancestor_name.startswith("Forklift1"):
                        forklift_index = 1
                        break
                    cursor = cursor.get_attach_parent()
                if forklift_index is not None:
                    add_component_tag(component, f"Qai.Forklift{forklift_index}.Wheel.{wheel_role}")
                    component.set_mobility(unreal.ComponentMobility.MOVABLE)
                    tagged_components += 1

            if name.startswith("lift_"):
                cursor = component.get_attach_parent()
                while cursor is not None:
                    ancestor_name = cursor.get_name()
                    if ancestor_name.startswith("Forklift1"):
                        add_component_tag(component, "Qai.Forklift1.Lift")
                        component.set_mobility(unreal.ComponentMobility.MOVABLE)
                        tagged_components += 1
                        break
                    cursor = cursor.get_attach_parent()

        # USD imports light actors as children of their lens components.  Tag
        # the light roots from that attachment so runtime signal changes toggle
        # both the exact lens geometry and the corresponding illumination.
        root = actor_root_component(actor)
        parent = root.get_attach_parent() if root is not None else None
        if label == "Glow" and parent is not None:
            parent_raw_name = parent.get_name()
            parent_name = parent_raw_name.rsplit("_", 1)[0]
            light_tag = component_tags.get(parent_raw_name) or component_tags.get(parent_name)
            if light_tag:
                add_actor_tag(actor, light_tag + ".Glow")
                tagged_actors += 1
        if label.startswith("StackLightBounce"):
            suffix = label.rsplit("_", 1)[-1]
            index = {"01": 0, "02": 1, "03": 2}.get(suffix)
            if index is not None:
                add_actor_tag(actor, f"Qai.StackBounce.{index}")
                tagged_actors += 1
        if label == "Glow" or label.startswith("StackLightBounce") or label == "SceneWash":
            light_component = actor.get_component_by_class(unreal.LightComponent)
            if light_component is not None:
                set_property(light_component, "cast_shadows", False)
                # USD light units do not map one-to-one to Unreal's lumen-based
                # components.  These authored bounce/wash helpers otherwise
                # overpower the exact emissive lenses after import.
                if label.startswith("StackLightBounce"):
                    set_property(light_component, "intensity", 0.0)
                    try:
                        set_property(light_component, "attenuation_radius", 260.0)
                    except RuntimeError:
                        pass
                elif label == "SceneWash":
                    set_property(light_component, "intensity", 350.0)
                elif label == "Glow":
                    # USD authored exposure values import as extreme Unreal
                    # lumen values. Keep the operational signal readable but
                    # prevent it from tinting the whole warehouse.
                    set_property(light_component, "intensity", 300.0)
                    try:
                        set_property(light_component, "attenuation_radius", 90.0)
                    except RuntimeError:
                        pass
        if label.startswith("Area"):
            light_component = actor.get_component_by_class(unreal.LightComponent)
            if light_component is not None:
                # Treat the authored fixtures as large warehouse luminaires.
                # Only two keys pay for VSM/RT shadows; the other two remain
                # shadowless fill so Ultra keeps headroom for hit lighting.
                casts_hero_shadows = label in {"Area1", "Area4"}
                set_property(light_component, "intensity", 4200.0)
                set_property(light_component, "cast_shadows", casts_hero_shadows)
                set_property(light_component, "indirect_lighting_intensity", 1.25)
                set_property(light_component, "volumetric_scattering_intensity", 0.12)
                if isinstance(light_component, unreal.RectLightComponent):
                    set_property(light_component, "source_width", 480.0)
                    set_property(light_component, "source_height", 260.0)
                try:
                    set_property(
                        light_component,
                        "contact_shadow_length",
                        0.055 if casts_hero_shadows else 0.0,
                    )
                    set_property(
                        light_component,
                        "contact_shadow_casting_intensity",
                        0.72 if casts_hero_shadows else 0.0,
                    )
                except RuntimeError:
                    pass
        sky_component = actor.get_component_by_class(unreal.SkyLightComponent)
        if sky_component is not None:
            # Omniverse's authored dome value (700) is an exposure-domain value,
            # while Unreal treats it as a direct intensity multiplier.
            set_property(sky_component, "intensity", 0.35)
            set_property(sky_component, "real_time_capture", False)
    if animated_workers != 2:
        raise RuntimeError(f"Expected two looping worker animations, configured {animated_workers}")
    if static_driver_component_count:
        if seated_drivers:
            raise RuntimeError("Driver import mixed static and skeletal representations")
        # Unreal collapses the driver into the scene hierarchy. The exact
        # NVIDIA character reference consists of seven material-part meshes;
        # validate that complete unit instead of relying on discarded Xform
        # actor labels.
        driver_part_count = 7
        if static_driver_component_count % driver_part_count:
            raise RuntimeError(
                "Static driver geometry is incomplete: "
                f"components={static_driver_component_count} parts_per_driver={driver_part_count}"
            )
        seated_drivers = static_driver_component_count // driver_part_count
    if seated_drivers != 1:
        raise RuntimeError(f"Expected one authored seated-driver pose, configured {seated_drivers}")
    return {
        "tagged_actors": tagged_actors,
        "tagged_components": tagged_components,
        "animated_workers": animated_workers,
        "seated_drivers": seated_drivers,
    }


def audit_level() -> dict[str, object]:
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    actor_rows: list[dict[str, object]] = []
    discovered_names: set[str] = set()
    component_count = 0
    for actor in actors:
        components = actor.get_components_by_class(unreal.SceneComponent)
        names = []
        for component in components:
            name = component.get_name()
            names.append(name)
            discovered_names.add(name)
        component_count += len(components)
        label = actor.get_actor_label()
        discovered_names.add(label)
        actor_rows.append(
            {
                "label": label,
                "class": actor.get_class().get_name(),
                "component_count": len(components),
                "components": sorted(names),
            }
        )
    missing = sorted(name for name in REQUIRED_RUNTIME_NAMES if not any(name in value for value in discovered_names))
    if missing:
        raise RuntimeError(f"Native level is missing required runtime object names: {missing}")
    return {
        "actor_count": len(actors),
        "scene_component_count": component_count,
        "actors": sorted(actor_rows, key=lambda row: str(row["label"])),
    }


def main() -> None:
    if not STAGE.is_file():
        raise FileNotFoundError(f"Lean conveyor stage is missing: {STAGE}")
    prepare_level()
    imported = import_stage()
    asset_report = optimize_generated_assets()
    material_report = repair_imported_material_bindings()
    binding_report = configure_runtime_bindings()
    visual_report = apply_visual_finish()
    level_report = audit_level()
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.save_current_level():
        raise RuntimeError(f"Could not save generated native level {LEVEL_PATH}")
    unreal.EditorAssetLibrary.save_directory(CONTENT_ROOT, only_if_is_dirty=False, recursive=True)
    unreal.EditorAssetLibrary.delete_asset(STAGING_LEVEL_PATH)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "source_stage": str(STAGE),
        "content_root": CONTENT_ROOT,
        "level": LEVEL_PATH,
        "imported_object_paths": sorted(imported),
        **asset_report,
        **material_report,
        **binding_report,
        **visual_report,
        **level_report,
    }
    AUDIT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_CONVEYOR_IMPORT={json.dumps(report, separators=(',', ':'))}")


main()
