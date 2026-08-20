"""Build the west-rack render mesh with its six uprights depth-separated.

The imported rack is intentionally a single optimized static mesh.  Its upright
faces, however, were authored coplanar with the orange shelf members, which can
shimmer under temporal anti-aliasing.  This editor utility moves only the six
visible upright triangle islands 7.5 mm toward the rack's depth centre.  The
runtime collision proxies are moved by the same amount in QaiConveyorWorld.
"""

from __future__ import annotations

import unreal


SOURCE_ASSET = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Storage/SM_WestRack"
)
OUTPUT_ASSET = "/Game/ConveyorRuntime/VisualFinish/SM_WestRack_NoFlicker"
RACK_CENTER_X_CM = -420.0
UPRIGHT_INSET_CM = 0.75
UPRIGHT_CENTERS_CM = (
    (-467.0, -333.0),
    (-467.0, 0.0),
    (-467.0, 333.0),
    (-373.0, -333.0),
    (-373.0, 0.0),
    (-373.0, 333.0),
)


def selection_from_upright_box(mesh: unreal.DynamicMesh, x: float, y: float):
    # The 16 x 16 cm selection box encloses one 8.5 x 12.7 cm visible post.
    # Requiring all three triangle vertices to lie in the box excludes the long
    # orange beams that cross the post at each shelf level.
    box = unreal.GeometryScript_Box.make_box_from_center_extents(
        unreal.Vector(x, y, 150.0), unreal.Vector(8.0, 8.0, 151.0)
    )
    result = unreal.GeometryScript_MeshSelection.select_mesh_elements_in_box(
        mesh,
        box,
        unreal.GeometryScriptMeshSelectionType.TRIANGLES,
        False,
        3,
    )
    return result[1] if isinstance(result, tuple) else result


def main() -> None:
    source = unreal.load_asset(SOURCE_ASSET)
    if not source:
        raise RuntimeError(f"Could not load source west-rack mesh: {SOURCE_ASSET}")

    if unreal.EditorAssetLibrary.does_asset_exist(OUTPUT_ASSET):
        if not unreal.EditorAssetLibrary.delete_asset(OUTPUT_ASSET):
            raise RuntimeError(f"Could not replace generated asset: {OUTPUT_ASSET}")
    output = unreal.EditorAssetLibrary.duplicate_asset(SOURCE_ASSET, OUTPUT_ASSET)
    if not output:
        raise RuntimeError(f"Could not duplicate {SOURCE_ASSET} to {OUTPUT_ASSET}")

    dynamic_mesh = unreal.DynamicMesh()
    copy_result = unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh_v2(
        source,
        dynamic_mesh,
        unreal.GeometryScriptCopyMeshFromAssetOptions(),
        unreal.GeometryScriptMeshReadLOD(),
        True,
    )
    if isinstance(copy_result, tuple):
        dynamic_mesh = copy_result[0]

    moved_triangles = 0
    for x, y in UPRIGHT_CENTERS_CM:
        selection = selection_from_upright_box(dynamic_mesh, x, y)
        _, triangle_count = unreal.GeometryScript_MeshSelection.get_mesh_selection_info(selection)
        if triangle_count != 70:
            raise RuntimeError(
                f"Unexpected west-rack upright selection at ({x}, {y}): "
                f"selected {triangle_count} triangles, expected 70"
            )
        direction = 1.0 if x < RACK_CENTER_X_CM else -1.0
        unreal.GeometryScript_MeshTransforms.translate_mesh_selection(
            dynamic_mesh,
            selection,
            unreal.Vector(direction * UPRIGHT_INSET_CM, 0.0, 0.0),
        )
        moved_triangles += triangle_count

    write_result = unreal.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(
        dynamic_mesh,
        output,
        unreal.GeometryScriptCopyMeshToAssetOptions(),
        unreal.GeometryScriptMeshWriteLOD(),
        True,
    )
    if isinstance(write_result, tuple):
        outcome = write_result[1]
        if outcome != unreal.GeometryScriptOutcomePins.SUCCESS:
            raise RuntimeError(f"Could not update generated west-rack mesh: {outcome}")

    unreal.EditorAssetLibrary.save_loaded_asset(output, only_if_is_dirty=False)
    unreal.log(
        f"Built {OUTPUT_ASSET}: moved {moved_triangles} upright triangles "
        f"inward by {UPRIGHT_INSET_CM:.2f} cm"
    )


main()
