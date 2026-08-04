"""Import the confidential IQ9 EVK STEP assembly locally for optimization.

The source and all derived assets remain inside the private project.  This
script intentionally performs no network access.
"""

import os
import unreal


PROJECT_DIR = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())
SOURCE_STEP = os.path.normpath(
    os.path.join(PROJECT_DIR, "..", "resources", "DP25-73418-2_RevB", "10-75699-RB8-PVT.stp")
)
DESTINATION = "/Game/IQ9EVK/CADSource"


def main():
    if not os.path.isfile(SOURCE_STEP):
        raise RuntimeError(f"IQ9 EVK source STEP not found: {SOURCE_STEP}")

    unreal.log(f"IQ9EVK_IMPORT source={SOURCE_STEP} destination={DESTINATION}")
    scene = unreal.DatasmithSceneElement.construct_datasmith_scene_from_cad_files([SOURCE_STEP])
    if not scene:
        raise RuntimeError("Datasmith CAD translator did not open the STEP source")

    tessellation = scene.get_options(unreal.DatasmithCommonTessellationOptions)
    if tessellation:
        options = tessellation.get_editor_property("options")
        options.set_editor_property("chord_tolerance", 1.0)
        options.set_editor_property("max_edge_length", 20.0)
        options.set_editor_property("normal_tolerance", 30.0)
        tessellation.set_editor_property("options", options)
        unreal.log("IQ9EVK_IMPORT tessellation chord_mm=1.0 max_edge_mm=20 normal_deg=30")

    result = scene.import_scene(DESTINATION)
    if not result or not result.import_succeed:
        raise RuntimeError("Datasmith CAD import failed")
    imported = [mesh.get_path_name() for mesh in list(result.imported_meshes or []) if mesh]
    unreal.log(f"IQ9EVK_IMPORT imported_count={len(imported)}")
    for object_path in imported[:100]:
        unreal.log(f"IQ9EVK_IMPORT asset={object_path}")

    unreal.EditorAssetLibrary.save_directory(DESTINATION, only_if_is_dirty=False, recursive=True)
    unreal.log("IQ9EVK_IMPORT complete")


if __name__ == "__main__":
    main()
