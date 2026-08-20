"""Import the widened NVIDIA A11 quarter modules as cooked Unreal assets."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


PROJECT_DIR = Path(
    unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())
).resolve()
SOURCE_DIR = PROJECT_DIR / "Saved" / "OmniverseConveyorGenerated"
DESTINATION = "/Game/ConveyorRuntime/OmniverseExtensions"
ORIGINAL_A11 = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Conveyor/Modules/SM_CurveNorth"
)
PRESENTATION_ROLLER_MATERIAL = (
    "/Game/ConveyorRuntime/VisualFinish/MI_AnisotropicRoller"
)
A11_TWO_SIDED_FRAME_MATERIAL = (
    "/Game/ConveyorRuntime/VisualFinish/M_A11TwoSidedConveyorFrame"
)


def set_property(obj: object, name: str, value: object) -> None:
    obj.set_editor_property(name, value)  # type: ignore[attr-defined]


def make_options() -> unreal.UsdStageImportOptions:
    options = unreal.UsdStageImportOptions()
    settings = {
        "import_actors": False,
        "import_geometry": True,
        "import_skeletal_animations": False,
        "import_level_sequences": False,
        "import_materials": False,
        "import_groom_assets": False,
        "import_sparse_volume_textures": False,
        "import_sounds": False,
        "prims_to_import": ["/World/Geometry"],
        "purposes_to_import": 7,
        "nanite_triangle_threshold": 50000,
        "render_context_to_import": "universal",
        "material_purpose": "allPurpose",
        "fallback_collision_type": unreal.UsdCollisionType.NONE,
        "subdivision_level": 0,
        "share_assets_for_identical_prims": True,
        "existing_asset_policy": unreal.ReplaceAssetPolicy.REPLACE,
        "prim_path_folder_structure": False,
        "use_prim_kinds_for_collapsing": True,
        # Keep the A11 GeomSubset slots distinct even though materials are not
        # duplicated. They are rebound to the already cooked source A11
        # materials below.
        "merge_identical_material_slots": False,
        "interpret_lods": True,
    }
    for name, value in settings.items():
        set_property(options, name, value)
    return options


def main() -> None:
    sources = {
        "SM_A11_WestQuarter_Omniverse": SOURCE_DIR
        / "ConveyorBelt_A11_WestQuarter.usda",
        "SM_A11_EastQuarter_Omniverse": SOURCE_DIR
        / "ConveyorBelt_A11_EastQuarter.usda",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generated A11 quarter stages are missing: {missing}")

    library = unreal.EditorAssetLibrary
    if library.does_directory_exist(DESTINATION):
        if not library.delete_directory(DESTINATION):
            raise RuntimeError(f"Could not replace {DESTINATION}")

    tasks: list[unreal.AssetImportTask] = []
    for target_name, source in sources.items():
        task = unreal.AssetImportTask()
        set_property(task, "automated", True)
        set_property(task, "save", False)
        set_property(task, "replace_existing", True)
        set_property(task, "replace_existing_settings", True)
        set_property(task, "filename", str(source))
        set_property(task, "destination_path", DESTINATION)
        set_property(task, "destination_name", target_name)
        set_property(task, "options", make_options())
        tasks.append(task)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)

    imported_paths = sorted(
        {
            path
            for task in tasks
            for path in list(task.get_editor_property("imported_object_paths"))
        }
    )
    meshes = []
    for path in library.list_assets(DESTINATION, recursive=True, include_folder=False):
        asset = library.load_asset(path)
        if isinstance(asset, unreal.StaticMesh):
            meshes.append(asset)
    if len(meshes) != 2:
        raise RuntimeError(
            f"Expected exactly two collapsed A11 quarter meshes, found "
            f"{len(meshes)}: {[mesh.get_path_name() for mesh in meshes]}"
        )

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    rename_data = []
    for target_name in sources:
        matching = [
            mesh
            for mesh in meshes
            if f"/{target_name}/" in mesh.get_path_name()
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"Could not identify imported mesh for {target_name}: "
                f"{[mesh.get_path_name() for mesh in meshes]}"
            )
        rename_data.append(
            unreal.AssetRenameData(matching[0], DESTINATION, target_name)
        )
    if not asset_tools.rename_assets(rename_data):
        raise RuntimeError("Could not assign stable names to imported A11 quarters")
    meshes = []
    for target_name in sources:
        mesh = library.load_asset(f"{DESTINATION}/{target_name}")
        if not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError(f"Renamed A11 mesh is missing: {target_name}")
        meshes.append(mesh)

    original = library.load_asset(ORIGINAL_A11)
    if not isinstance(original, unreal.StaticMesh):
        raise RuntimeError(f"Missing original imported NVIDIA A11 mesh: {ORIGINAL_A11}")
    original_materials = list(original.get_editor_property("static_materials"))
    if len(original_materials) != 8:
        raise RuntimeError(
            f"Expected eight source A11 materials, found {len(original_materials)}"
        )
    presentation_roller_material = library.load_asset(PRESENTATION_ROLLER_MATERIAL)
    if not isinstance(presentation_roller_material, unreal.MaterialInterface):
        raise RuntimeError(
            f"Missing cooked conveyor roller finish: {PRESENTATION_ROLLER_MATERIAL}"
        )
    a11_two_sided_frame_material = library.load_asset(A11_TWO_SIDED_FRAME_MATERIAL)
    if not isinstance(a11_two_sided_frame_material, unreal.MaterialInterface):
        raise RuntimeError(
            f"Missing cooked A11 two-sided frame finish: {A11_TWO_SIDED_FRAME_MATERIAL}"
        )
    report_meshes = []
    for mesh in meshes:
        materials = list(mesh.get_editor_property("static_materials"))
        if len(materials) not in (61, 62):
            raise RuntimeError(
                f"Unexpected split A11 material layout on {mesh.get_path_name()}: "
                f"{len(materials)} slots"
            )
        repaired = 0
        for index, slot in enumerate(materials):
            # The importer merges duplicate body subset materials, leaving six
            # body/decal slots, then appends the 27 authored A11 rollers and
            # the surviving clear-blue acrylic rubber bands.
            # In the presentation build those two coincident layers otherwise read as
            # alternating matte, transparent and gold-edged cylinders under
            # Lumen/RT. Keep the exact Omniverse geometry, but normalize both
            # visible roller layers to the same cooked steel finish used by
            # the adjacent A08 modules.
            if index == 0:
                material = a11_two_sided_frame_material
            elif index >= 6:
                material = presentation_roller_material
            else:
                source_slot = original_materials[index]
                material = source_slot.get_editor_property("material_interface")
            slot.set_editor_property("material_interface", material)
            repaired += 1
        mesh.set_editor_property("static_materials", materials)
        # The commandlet has no StaticMeshEditorSubsystem. The USD importer
        # already generated full Nanite resources for both >200k-triangle
        # modules, so no second mesh build is necessary here.
        nanite = mesh.get_editor_property("nanite_settings")
        nanite.enabled = True
        set_property(mesh, "nanite_settings", nanite)
        set_property(mesh, "allow_cpu_access", False)
        library.save_loaded_asset(mesh, only_if_is_dirty=False)
        report_meshes.append(
            {
                "path": mesh.get_path_name(),
                "triangles": int(mesh.get_num_triangles(0)),
                "material_slots": len(materials),
                "repaired_material_slots": repaired,
                "bounds": str(mesh.get_bounds()),
            }
        )

    # The temporary imported material/texture packages exist only to preserve
    # every repeated roller and rubber-band binding. Once all slots point at
    # the original cooked A11 materials, remove those duplicates.
    kept_paths = {mesh.get_path_name().split(".", 1)[0] for mesh in meshes}
    deleted_temporary_assets = 0
    for path in library.list_assets(DESTINATION, recursive=True, include_folder=False):
        package_path = path.split(".", 1)[0]
        if package_path in kept_paths:
            continue
        if library.delete_asset(path):
            deleted_temporary_assets += 1

    library.save_directory(DESTINATION, only_if_is_dirty=False, recursive=True)
    report = {
        "imported_object_paths": imported_paths,
        "meshes": report_meshes,
        "deleted_temporary_assets": deleted_temporary_assets,
    }
    unreal.log(f"QAI_OMNIVERSE_CONVEYOR_EXTENSIONS={json.dumps(report)}")


main()
