"""Build a compact, open-top IQ9 EVK runtime proxy from the local STEP.

The local CAD assembly is used only as an offline source. External cables,
adapters, carrier boards and the removable top lid are omitted before the
remaining enclosure/PCB assembly is baked to a single portable runtime mesh.
No source or rendered view leaves the private workspace.
"""

import os
import unreal


PROJECT_DIR = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())
SOURCE_STEP = os.path.normpath(
    os.path.join(PROJECT_DIR, "..", "resources", "DP25-73418-2_RevB", "10-75699-RB8-PVT.stp")
)
CAD_DESTINATION = "/Game/IQ9EVK/CADSource"
PROXY_PACKAGE = "/Game/IQ9EVK/Runtime/SM_IQ9_EVK_Core_Proxy"

# Stable centimetre coordinates in the Qualcomm STEP assembly. The actual EVK
# enclosure occupies this square; connected cables and evaluation peripherals
# extend to x=-20/+10 and y=-4/+12 and are deliberately not redistributed into
# the runtime proxy.
CORE_MIN_X = -10.35
CORE_MAX_X = 0.35
CORE_MIN_Y = -0.35
CORE_MAX_Y = 10.35
OMIT_NAME_TOKENS = (
    "TOP_COVER",
    "WINDOW",
    "SCREW_FLAT_HEAD",
    "CABLE",
    "PLUG",
    "OVERMOLD",
)


def set_prop(obj, name, value):
    try:
        obj.set_editor_property(name, value)
        return True
    except Exception as exc:
        unreal.log_warning(f"IQ9EVK_PROXY option_unavailable name={name} error={exc}")
        return False


def main():
    if not os.path.isfile(SOURCE_STEP):
        raise RuntimeError(f"IQ9 EVK source STEP not found: {SOURCE_STEP}")

    scene = unreal.DatasmithSceneElement.construct_datasmith_scene_from_cad_files([SOURCE_STEP])
    if not scene:
        raise RuntimeError("Datasmith CAD translator did not open the STEP source")
    tessellation = scene.get_options(unreal.DatasmithCommonTessellationOptions)
    if tessellation:
        options = tessellation.get_editor_property("options")
        set_prop(options, "chord_tolerance", 1.0)
        set_prop(options, "max_edge_length", 20.0)
        set_prop(options, "normal_tolerance", 30.0)
        set_prop(tessellation, "options", options)

    result = scene.import_scene(CAD_DESTINATION)
    if not result or not result.import_succeed:
        raise RuntimeError("Datasmith CAD import failed while building the proxy")

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = []
    component_bounds = []
    for actor in actor_subsystem.get_all_level_actors():
        if not isinstance(actor, unreal.StaticMeshActor):
            continue
        component = actor.static_mesh_component
        if not component or not component.static_mesh:
            continue
        origin, box_extent = actor.get_actor_bounds(False)
        if max(box_extent.x, box_extent.y, box_extent.z) > 100.0:
            continue
        component_bounds.append((actor, origin, box_extent))
    if component_bounds:
        xs = sorted(item[1].x for item in component_bounds)
        ys = sorted(item[1].y for item in component_bounds)
        zs = sorted(item[1].z for item in component_bounds)
        middle = len(component_bounds) // 2
        median = unreal.Vector(xs[middle], ys[middle], zs[middle])
        for actor, origin, box_extent in component_bounds:
            offset = origin - median
            # Datasmith also emits large helper/default bounds that are not
            # physical EVK geometry.  Keep a generous one-metre envelope.
            label = actor.get_actor_label().upper()
            is_core = (
                CORE_MIN_X <= origin.x <= CORE_MAX_X
                and CORE_MIN_Y <= origin.y <= CORE_MAX_Y
            )
            is_omitted_part = any(token in label for token in OMIT_NAME_TOKENS)
            if (max(abs(offset.x), abs(offset.y), abs(offset.z)) <= 100.0
                    and max(box_extent.x, box_extent.y, box_extent.z) <= 100.0
                    and is_core
                    and not is_omitted_part):
                actors.append(actor)
    if not actors:
        raise RuntimeError("The imported CAD scene created no StaticMeshActor sources")

    bounds_min = None
    bounds_max = None
    component_count = 0
    for actor in actors:
        component = actor.static_mesh_component
        component_count += 1
        origin, actor_extent = actor.get_actor_bounds(False)
        actor_min = origin - actor_extent
        actor_max = origin + actor_extent
        if bounds_min is None:
            bounds_min = actor_min
            bounds_max = actor_max
        else:
            bounds_min = unreal.Vector(
                min(bounds_min.x, actor_min.x),
                min(bounds_min.y, actor_min.y),
                min(bounds_min.z, actor_min.z))
            bounds_max = unreal.Vector(
                max(bounds_max.x, actor_max.x),
                max(bounds_max.y, actor_max.y),
                max(bounds_max.z, actor_max.z))
    if bounds_min is None:
        raise RuntimeError("The imported CAD actors have no render bounds")
    extent = (bounds_max - bounds_min) * 0.5
    center = (bounds_max + bounds_min) * 0.5
    unreal.log(
        "IQ9EVK_PROXY source_actors={} filtered_from={} omitted={} components={} center_cm=({:.2f},{:.2f},{:.2f}) "
        "size_cm=({:.2f},{:.2f},{:.2f})".format(
            len(actors), len(component_bounds), len(component_bounds) - len(actors),
            component_count, center.x, center.y, center.z,
            extent.x * 2.0, extent.y * 2.0, extent.z * 2.0))
    if max(extent.x, extent.y, extent.z) > 60.0:
        raise RuntimeError(
            "Refusing IQ9 proxy bake because filtered CAD bounds still exceed 120 cm")

    proxy_options = unreal.CreateProxyMeshActorOptions()
    set_prop(proxy_options, "base_package_name", PROXY_PACKAGE)
    set_prop(proxy_options, "new_actor_label", "IQ9 EVK bake proxy")
    set_prop(proxy_options, "destroy_source_actors", False)
    set_prop(proxy_options, "spawn_merged_actor", True)

    proxy = proxy_options.get_editor_property("mesh_proxy_settings")
    set_prop(proxy, "screen_size", 760)
    set_prop(proxy, "override_voxel_size", True)
    set_prop(proxy, "voxel_size", 0.16)
    set_prop(proxy, "merge_distance", 0.06)
    set_prop(proxy, "recalculate_normals", True)
    set_prop(proxy, "use_hard_angle_threshold", True)
    set_prop(proxy, "hard_angle_threshold", 55.0)
    set_prop(proxy, "create_collision", True)
    set_prop(proxy, "generate_lightmap_uvs", True)
    set_prop(proxy, "support_ray_tracing", True)
    set_prop(proxy, "allow_distance_field", True)
    set_prop(proxy, "group_identical_meshes_for_baking", True)

    material = proxy.get_editor_property("material_settings")
    set_prop(material, "texture_size", unreal.IntPoint(2048, 2048))
    set_prop(material, "normal_map", True)
    set_prop(material, "roughness_map", True)
    set_prop(material, "metallic_map", True)
    set_prop(material, "ambient_occlusion_map", True)
    set_prop(proxy, "material_settings", material)
    set_prop(proxy_options, "mesh_proxy_settings", proxy)

    static_mesh_subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    create_result = static_mesh_subsystem.create_proxy_mesh_actor(actors, proxy_options)
    if isinstance(create_result, tuple):
        success = bool(create_result[0])
        proxy_actor = create_result[1] if len(create_result) > 1 else None
    else:
        success = bool(create_result)
        proxy_actor = None
    if not success:
        raise RuntimeError("Unreal proxy generation failed")

    if proxy_actor:
        mesh_component = proxy_actor.static_mesh_component
        mesh = mesh_component.static_mesh if mesh_component else None
        if mesh:
            mesh.set_editor_property("allow_cpu_access", False)
            mesh.set_editor_property("support_ray_tracing", True)
            unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False)
            unreal.log(f"IQ9EVK_PROXY mesh={mesh.get_path_name()}")

    unreal.EditorAssetLibrary.save_directory("/Game/IQ9EVK/Runtime", only_if_is_dirty=False, recursive=True)
    # CAD packages are offline intermediates and must not enter a cook or an
    # installer. The STEP remains in the private resources directory.
    unreal.EditorAssetLibrary.delete_directory(CAD_DESTINATION)
    unreal.log("IQ9EVK_PROXY complete")


if __name__ == "__main__":
    main()
