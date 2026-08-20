#!/usr/bin/env python3
"""Print the authored USD skeleton and animation bindings used by the demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "Saved" / "BakedStageGenerated" / "warehouse_conveyor_baked.usdc",
    )
    parser.add_argument("--repair-driver-bindings", action="store_true")
    parser.add_argument("--bake-driver-meshes", action="store_true")
    parser.add_argument("--remove-drivers", action="store_true")
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.stage.resolve()), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open {args.stage}")

    if args.remove_drivers:
        removed = []
        for path in (
            "/World/CodexPoC/ConveyorSafety/Forklifts/Forklift1/body/DriverMount",
            "/World/CodexPoC/ConveyorSafety/Workers/DriverAnimations",
        ):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Expected removable driver branch is missing: {path}")
            prim.SetActive(False)
            removed.append(path)
        stage.GetRootLayer().Save()
        stage_path = args.stage.resolve()
        manifest_path = stage_path.parent / "baked_stage_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(item for item in manifest["files"] if item["path"] == stage_path.name)
            entry["bytes"] = stage_path.stat().st_size
            entry["sha256"] = hashlib.sha256(stage_path.read_bytes()).hexdigest()
            manifest["removed_decorative_driver_branches"] = removed
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("QAI_DRIVER_REMOVAL=" + json.dumps({"removed": removed}), flush=True)

    if args.repair_driver_bindings:
        pose_path = Sdf.Path(
            "/World/CodexPoC/ConveyorSafety/Workers/DriverAnimations/SeatedDriverPose"
        )
        pose = stage.GetPrimAtPath(pose_path)
        pose_time = pose.GetAttribute("translations").GetTimeSamples()[0]
        seated_transforms = UsdSkel.MakeTransforms(
            pose.GetAttribute("translations").Get(pose_time),
            pose.GetAttribute("rotations").Get(pose_time),
            pose.GetAttribute("scales").Get(pose_time),
        )
        for forklift_index in (1, 2):
            skel_root_path = (
                f"/World/CodexPoC/ConveyorSafety/Forklifts/Forklift{forklift_index}/body/DriverMount/"
                "Character/male_adult_construction_05/ManRoot/male_adult_construction_05"
            )
            skeleton_path = (
                skel_root_path + "/male_adult_construction_05/male_adult_construction_05"
            )
            skel_root = stage.GetPrimAtPath(skel_root_path)
            skeleton = stage.GetPrimAtPath(skeleton_path)
            if not skel_root.IsValid() or skel_root.GetTypeName() != "SkelRoot":
                raise RuntimeError(f"Required driver SkelRoot is missing: {skel_root_path}")
            if not skeleton.IsValid() or skeleton.GetTypeName() != "Skeleton":
                raise RuntimeError(f"Required driver Skeleton is missing: {skeleton_path}")
            skeleton.GetAttribute("restTransforms").Set(seated_transforms)
            skel_root.CreateRelationship("skel:animationSource").ClearTargets(True)
            skeleton.CreateRelationship("skel:animationSource").ClearTargets(True)
            local_pose_path = Sdf.Path(skel_root_path + f"/SeatedDriverPose_{forklift_index}")
            if stage.GetPrimAtPath(local_pose_path).IsValid():
                stage.RemovePrim(local_pose_path)
        pose.SetActive(False)
        stage.GetRootLayer().Save()

        stage_path = args.stage.resolve()
        manifest_path = stage_path.parent / "baked_stage_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(item for item in manifest["files"] if item["path"] == stage_path.name)
            entry["bytes"] = stage_path.stat().st_size
            digest = hashlib.sha256(stage_path.read_bytes()).hexdigest()
            entry["sha256"] = digest
            manifest["driver_skeleton_bindings"] = 0
            manifest["localized_driver_animations"] = 0
            manifest["baked_driver_reference_poses"] = 2
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("QAI_DRIVER_REFERENCE_POSE=" + json.dumps({"drivers": 2}), flush=True)

    if args.bake_driver_meshes:
        baked_meshes = []
        for forklift_index in (1, 2):
            skel_root_path = (
                f"/World/CodexPoC/ConveyorSafety/Forklifts/Forklift{forklift_index}/body/DriverMount/"
                "Character/male_adult_construction_05/ManRoot/male_adult_construction_05"
            )
            skel_root_prim = stage.GetPrimAtPath(skel_root_path)
            if not skel_root_prim.IsValid() or skel_root_prim.GetTypeName() != "SkelRoot":
                raise RuntimeError(f"Required driver SkelRoot is missing: {skel_root_path}")

            # The reference-pose repair above leaves the skeleton in the exact
            # authored seated state. Bake that deformation into point data at
            # time zero, promote the sample to a default value, and strip the
            # now-unused skin binding so Unreal imports inexpensive static mesh
            # geometry instead of a permanently ticking character.
            UsdSkel.BakeSkinning(UsdSkel.Root(skel_root_prim), Gf.Interval(0.0))
            for prim in Usd.PrimRange(skel_root_prim):
                if prim.GetTypeName() != "Mesh":
                    continue
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                points = points_attr.Get(Usd.TimeCode(0.0))
                if points is None:
                    points = points_attr.Get()
                if points is None:
                    raise RuntimeError(f"Baked driver mesh has no points: {prim.GetPath()}")
                points_attr.Clear()
                points_attr.Set(points)
                normals_attr = mesh.GetNormalsAttr()
                if normals_attr and normals_attr.HasValue():
                    normals = normals_attr.Get(Usd.TimeCode(0.0))
                    if normals is None:
                        normals = normals_attr.Get()
                    if normals is not None:
                        normals_attr.Clear()
                        normals_attr.Set(normals)
                for property_name in (
                    "primvars:skel:geomBindTransform",
                    "primvars:skel:jointIndices",
                    "primvars:skel:jointWeights",
                    "skel:animationSource",
                    "skel:blendShapes",
                    "skel:blendShapeTargets",
                    "skel:joints",
                    "skel:skeleton",
                ):
                    prim.RemoveProperty(property_name)
                baked_meshes.append(str(prim.GetPath()))

            for prim in Usd.PrimRange(skel_root_prim):
                if prim.GetTypeName() == "Skeleton":
                    prim.SetActive(False)
            skel_root_prim.SetTypeName("Xform")

        stage.GetRootLayer().Save()
        stage_path = args.stage.resolve()
        manifest_path = stage_path.parent / "baked_stage_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(item for item in manifest["files"] if item["path"] == stage_path.name)
            entry["bytes"] = stage_path.stat().st_size
            entry["sha256"] = hashlib.sha256(stage_path.read_bytes()).hexdigest()
            manifest["baked_static_driver_meshes"] = len(baked_meshes)
            manifest["driver_runtime_skeletons"] = 0
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            "QAI_DRIVER_STATIC_BAKE="
            + json.dumps({"drivers": 2, "meshes": len(baked_meshes)}),
            flush=True,
        )

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        if "DriverMount" not in path and "DriverAnimations" not in path:
            continue
        relationships = []
        for relationship in prim.GetRelationships():
            targets = [str(target) for target in relationship.GetTargets()]
            if any(token in relationship.GetName().lower() for token in ("skel", "anim")):
                relationships.append((relationship.GetName(), targets))
        attributes = []
        for attribute in prim.GetAttributes():
            samples = attribute.GetTimeSamples()
            if samples or any(token in attribute.GetName().lower() for token in ("joint", "transform", "blend")):
                value = attribute.Get() if not samples else None
                try:
                    value = f"<{len(value)} values>" if value is not None else None
                except TypeError:
                    value = str(value)
                attributes.append((attribute.GetName(), len(samples), samples[:2], value))
        has_animation_binding = any(name == "skel:animationSource" and targets for name, targets in relationships)
        if type_name in {"Skeleton", "SkelAnimation", "SkelRoot"} or has_animation_binding:
            print(f"PRIM {path} type={type_name} active={prim.IsActive()}")
            for name, targets in relationships:
                print(f"  REL {name} -> {targets}")
            for name, sample_count, first_samples, value in attributes:
                print(f"  ATTR {name} sample_count={sample_count} first={first_samples} value={value}")


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
