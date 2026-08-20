"""Build widened turn modules from NVIDIA's authored Omniverse A11 asset.

This script runs with Isaac Sim's Python environment.  It preserves the
authored meshes/material bindings and produces cropped quarter-turn USD files;
Unreal then imports those files as native cooked meshes.  It deliberately does
not synthesize rollers, rails, or supports from generic primitives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback
from typing import Any

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    print(
        "QAI_A11_BUILD_ARGS",
        {"source": str(source_path), "inspect": args.inspect, "output": str(output_dir)},
    )

    app = SimulationApp({"headless": True})
    try:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(source_path))
        if stage is None:
            raise RuntimeError(f"Could not open Omniverse source: {args.source}")

        included_purposes = [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ]
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            included_purposes,
        )
        if args.inspect:
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                name = prim.GetName()
                if not (
                    name == "SM_ConveyorBelt_A11_01"
                    or "Roller01" in name
                    or "Roller27" in name
                    or "Roller54" in name
                    or "Rubberband01" in name
                ):
                    continue
                mesh = UsdGeom.Mesh(prim)
                world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                points = mesh.GetPointsAttr().Get() or []
                counts = mesh.GetFaceVertexCountsAttr().Get() or []
                indices = mesh.GetFaceVertexIndicesAttr().Get() or []
                primvars = []
                for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
                    values = primvar.Get()
                    primvars.append(
                        (
                            primvar.GetPrimvarName(),
                            primvar.GetInterpolation(),
                            len(values) if values is not None else 0,
                            primvar.IsIndexed(),
                            len(primvar.GetIndices() or []),
                        )
                    )
                subsets = []
                for child in prim.GetChildren():
                    if child.IsA(UsdGeom.Subset):
                        subset_indices = UsdGeom.Subset(child).GetIndicesAttr().Get() or []
                        subsets.append((child.GetName(), len(subset_indices)))
                print(
                    "QAI_A11_MESH",
                    {
                        "path": str(prim.GetPath()),
                        "bounds_min": tuple(world_range.GetMin()),
                        "bounds_max": tuple(world_range.GetMax()),
                        "points": len(points),
                        "faces": len(counts),
                        "indices": len(indices),
                        "primvars": primvars,
                        "subsets": subsets,
                        "transform": tuple(
                            tuple(row)
                            for row in UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                                Usd.TimeCode.Default()
                            )
                        ),
                    },
                )
            return

        if args.output_dir is None:
            parser.error("--output-dir is required unless --inspect is used")
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        def filter_face_values(
            values: Any,
            kept_faces: list[int],
            kept_corners: list[int],
            old_face_count: int,
            old_corner_count: int,
        ) -> Any:
            if values is None:
                return values
            if len(values) == old_face_count:
                return type(values)([values[index] for index in kept_faces])
            if len(values) == old_corner_count:
                return type(values)([values[index] for index in kept_corners])
            return values

        def crop_mesh(
            output_stage: Usd.Stage,
            prim: Usd.Prim,
            keep_west: bool,
            split_x: float,
        ) -> tuple[int, int]:
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
            indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            kept_faces: list[int] = []
            kept_corners: list[int] = []
            new_counts: list[int] = []
            new_indices: list[int] = []
            old_to_new: dict[int, int] = {}
            cursor = 0
            for face_index, count in enumerate(counts):
                face_indices = indices[cursor : cursor + count]
                centroid_x = sum(
                    world.Transform(Gf.Vec3d(points[index]))[0]
                    for index in face_indices
                ) / max(1, count)
                keep = centroid_x <= split_x if keep_west else centroid_x > split_x
                if keep:
                    old_to_new[face_index] = len(kept_faces)
                    kept_faces.append(face_index)
                    new_counts.append(count)
                    new_indices.extend(face_indices)
                    kept_corners.extend(range(cursor, cursor + count))
                cursor += count

            mesh.GetFaceVertexCountsAttr().Set(new_counts)
            mesh.GetFaceVertexIndicesAttr().Set(new_indices)

            old_face_count = len(counts)
            old_corner_count = len(indices)
            normals_attr = mesh.GetNormalsAttr()
            normals = normals_attr.Get()
            filtered_normals = filter_face_values(
                normals,
                kept_faces,
                kept_corners,
                old_face_count,
                old_corner_count,
            )
            if filtered_normals is not normals:
                normals_attr.Set(filtered_normals)

            holes_attr = mesh.GetHoleIndicesAttr()
            holes = holes_attr.Get()
            if holes:
                holes_attr.Set(
                    [old_to_new[index] for index in holes if index in old_to_new]
                )

            for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
                interpolation = primvar.GetInterpolation()
                if interpolation not in (
                    UsdGeom.Tokens.uniform,
                    UsdGeom.Tokens.faceVarying,
                ):
                    continue
                if primvar.IsIndexed():
                    old_indices = primvar.GetIndices()
                    filtered = filter_face_values(
                        old_indices,
                        kept_faces,
                        kept_corners,
                        old_face_count,
                        old_corner_count,
                    )
                    if filtered is not old_indices:
                        primvar.SetIndices(filtered)
                else:
                    values = primvar.Get()
                    filtered = filter_face_values(
                        values,
                        kept_faces,
                        kept_corners,
                        old_face_count,
                        old_corner_count,
                    )
                    if filtered is not values:
                        primvar.Set(filtered)

            for child in prim.GetChildren():
                if not child.IsA(UsdGeom.Subset):
                    continue
                subset = UsdGeom.Subset(child)
                old_subset = subset.GetIndicesAttr().Get() or []
                subset.GetIndicesAttr().Set(
                    [old_to_new[index] for index in old_subset if index in old_to_new]
                )
            return old_face_count, len(kept_faces)

        source_bounds = cache.ComputeWorldBound(
            stage.GetPrimAtPath("/World/Geometry/SM_ConveyorBelt_A11_01")
        ).ComputeAlignedRange()
        split_x = 0.5 * (source_bounds.GetMin()[0] + source_bounds.GetMax()[0])
        for label, keep_west in (("West", True), ("East", False)):
            output_path = output_dir / f"ConveyorBelt_A11_{label}Quarter.usda"
            stage.Flatten().Export(str(output_path))
            output_stage = Usd.Stage.Open(str(output_path))
            output_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), included_purposes
            )
            removed_meshes = 0
            cropped_meshes = 0
            cropped_faces = 0
            source_faces = 0
            for prim in list(output_stage.Traverse()):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                name = prim.GetName()
                if name in (
                    "SM_ConveyorBelt_A11_01",
                    "SM_ConveyorBelt_A11_Decal_01",
                ):
                    old_count, new_count = crop_mesh(
                        output_stage, prim, keep_west, split_x
                    )
                    source_faces += old_count
                    cropped_faces += new_count
                    cropped_meshes += 1
                    continue
                if "Roller" in name or "Rubberband" in name:
                    bounds = output_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    center_x = bounds.GetMidpoint()[0]
                    keep = center_x <= split_x if keep_west else center_x > split_x
                    if not keep:
                        output_stage.RemovePrim(prim.GetPath())
                        removed_meshes += 1
            output_stage.GetRootLayer().Save()
            print(
                "QAI_A11_QUARTER",
                {
                    "side": label,
                    "output": str(output_path),
                    "split_x": split_x,
                    "cropped_meshes": cropped_meshes,
                    "source_faces": source_faces,
                    "kept_faces": cropped_faces,
                    "removed_meshes": removed_meshes,
                },
            )
    except BaseException:
        print("QAI_A11_BUILD_ERROR")
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
