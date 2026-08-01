from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from integrations.isaac_sim_mcp.conveyor_safety import (
    CAMERA_PATHS,
    capture_conveyor_safety_camera_source,
    create_conveyor_safety_scene_source,
    get_conveyor_safety_state_source,
    set_conveyor_safety_forklift_pose_source,
    set_conveyor_safety_light_and_get_state_source,
    set_conveyor_safety_light_source,
)
from integrations.isaac_sim_mcp.server import TOOLS
from scripts.run_live_isaac_conveyor_safety import (
    AMBER_CONFIRMATION_PROMPT,
    HAZARD_PROMPT,
    RED_CONFIRMATION_PROMPT,
    _atomic_write_json,
    apply_visual_consistency_gates,
    build_parser,
    extract_forklift_motion_observation,
    extract_visual_overlap_observation,
    recover_forklift_boxes_from_rgb,
    request_hazard_signal,
    wait_for_timeline_playing,
)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ConveyorSafetySourceTests(unittest.TestCase):
    def test_status_writer_retries_windows_sharing_violation(self) -> None:
        original_replace = Path.replace
        replace_calls = 0

        def flaky_replace(source: Path, target: Path) -> Path:
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls < 3:
                raise PermissionError("destination is briefly open")
            return original_replace(source, target)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live_status.json"
            with (
                mock.patch.object(Path, "replace", new=flaky_replace),
                mock.patch(
                    "scripts.run_live_isaac_conveyor_safety.time.sleep"
                ) as sleep,
            ):
                _atomic_write_json(path, {"state": "running"})
            self.assertEqual(replace_calls, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"state": "running"},
            )
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_server_exposes_bounded_conveyor_tools(self) -> None:
        names = {tool["name"] for tool in TOOLS}
        self.assertIn("isaac_create_conveyor_safety_scene", names)
        self.assertIn("isaac_enable_conveyor_safety", names)
        self.assertIn("isaac_capture_conveyor_safety_camera", names)
        self.assertIn("isaac_set_conveyor_safety_light", names)
        self.assertIn("isaac_get_conveyor_safety_state", names)
        self.assertIn("isaac_set_conveyor_safety_forklift_pose", names)

    def test_scene_has_required_actors_animation_and_cameras(self) -> None:
        source = create_conveyor_safety_scene_source(
            {"new_stage": True, "start_playing": True}
        )
        self.assertIn("await context.new_stage_async()", source)
        self.assertIn('"forklift_count": len(forklift_starts)', source)
        self.assertIn('"worker_count": len(worker_routes)', source)
        self.assertIn('"oval_conveyor_rack_rectangle"', source)
        self.assertIn('"north_shelf_to_oval_shuttle"', source)
        self.assertIn('"codex:routeWaypoints"', source)
        self.assertIn('"codex:routeDwellSeconds"', source)
        self.assertIn('"codex:routeFacingYawDegrees"', source)
        self.assertIn('"codex:workerNavigationMode"', source)
        self.assertIn('"codex:navigationBlocked"', source)
        self.assertIn('"codex:navigationBlockedBy"', source)
        self.assertIn('"codex:navigationReplanCount"', source)
        self.assertIn("codex:conveyorVisualCenterYM", source)
        self.assertIn('"codex:enabledForkliftCount"', source)
        self.assertIn('"codex:enabledForkliftIndices"', source)
        self.assertIn("Sdf.ValueTypeNames.IntArray, [1, 2]", source)
        self.assertIn("(-1.00, -1.85, 0.10, 0.0)", source)
        self.assertIn("(-0.50, 0.85, 0.10, 180.0)", source)
        self.assertIn('"parcel_count": parcel_count', source)
        self.assertIn("parcel_count = 8", source)
        self.assertIn("codex:touchesParcels", source)
        self.assertIn("codex:phase", source)
        self.assertIn("HumanMotionLibrary.usd", source)
        self.assertIn("UsdSkel.BindingAPI.Apply", source)
        self.assertIn("for skeleton_prim in Usd.PrimRange(prim):", source)
        self.assertIn("skeleton_prim.IsA(UsdSkel.Skeleton)", source)
        self.assertIn('"WalkForward"', source)
        self.assertIn('"WalkForward_01"', source)
        self.assertIn('"WalkForwardLoop"', source)
        self.assertIn('"WalkForwardLoop_01"', source)
        self.assertIn("walking_loop_source_start_time_code = 154", source)
        self.assertIn("walking_loop_source_end_time_code = 228", source)
        self.assertIn("root_motion_joint_indices", source)
        self.assertIn('str(joint_name) == "RL_BoneRoot/Hip"', source)
        self.assertIn("in_place_translations", source)
        self.assertIn("float(root_translation[2])", source)
        self.assertIn(
            "stage.SetEndTimeCode(float(walking_loop_duration_time_codes))",
            source,
        )
        self.assertIn("/BuiltinActions/MoveWalk/", source)
        self.assertIn('"codex:walkAnimation"', source)
        self.assertIn('"codex:idleAnimation"', source)
        self.assertIn('"codex:skelRootPath"', source)
        self.assertIn("SitAndStandChair", source)
        self.assertIn("driver_pose_sample = 422.0", source)
        self.assertIn("driver_whole_body_pitch_degrees = -20.0", source)
        self.assertIn(
            "pose_rotations[index] = "
            "driver_whole_body_pitch * pose_rotations[index]",
            source,
        )
        self.assertIn('"CreateRetargetAnimationsCommand"', source)
        self.assertIn('retarget_extension_name = "omni.anim.retarget.core"', source)
        self.assertIn('"RL_BoneRoot/Hip"', source)
        self.assertIn("float(hip_translation[2])", source)
        self.assertIn('"driver_count": len(driver_animation_bindings)', source)
        self.assertIn(
            'driver_mount_path = f"{wrapper_path}/body/DriverMount"',
            source,
        )
        self.assertIn("(-62.0, 4.0, 77.0)", source)
        self.assertIn("rotate=(0.0, 0.0, -90.0)", source)
        self.assertIn("scale=(100.0, 100.0, 100.0)", source)
        self.assertIn(
            "motion_library.GetPayloads().AddPayload(motion_library_asset)",
            source,
        )
        self.assertNotIn('"CreatePayloadCommand"', source)
        self.assertIn("SM_RackShelf_01.usd", source)
        self.assertIn("SM_RackFrame_03.usd", source)
        self.assertIn("SM_CardBoxD_04.usd", source)
        self.assertIn("SM_CardBoxA_01.usd", source)
        self.assertIn("shelf_carton_profiles = (", source)
        self.assertIn("((1.46, 1.64, 1.24), -4.0)", source)
        self.assertIn("((2.04, 1.50, 2.02), 89.0)", source)
        self.assertIn(
            "shelf_carton_depth_jitter_m = (-0.06, 0.04, -0.02, 0.07)",
            source,
        )
        self.assertIn("shelf_carton_front_offset_m = 0.30", source)
        self.assertIn("shelf_carton_gap_m = 0.04", source)
        self.assertIn("rack_upright_x_offsets_m = (-0.47, 0.47)", source)
        self.assertIn(
            "rack_upright_half_extents_m = (0.065, 0.075, 1.50)",
            source,
        )
        self.assertIn("projected_half_y_m = 0.5 * (", source)
        self.assertIn("row_span_m = (", source)
        self.assertIn(
            "row_cursor_y_m = segment_y - 0.5 * row_span_m", source
        )
        self.assertIn(
            "carton_y = row_cursor_y_m + projected_half_y_m", source
        )
        self.assertIn("shelf_surface_offset_m = 0.026", source)
        self.assertIn("shelf_carton_seating_gap_m = 0.004", source)
        self.assertIn("shelf_level_z_values = (0.48, 1.55, 2.62)", source)
        self.assertIn("shelf_collision_half_thickness_m = 0.015", source)
        self.assertIn("ShelfPlane", source)
        self.assertIn("shelf_collision_plane_count += 1", source)
        self.assertIn("rack_upright_collider_count += 1", source)
        self.assertIn('"rack_vertical_upright"', source)
        self.assertIn('"codex:rackUprightColliderCount"', source)
        self.assertIn('"codex:shelfCollisionPlaneCount"', source)
        self.assertIn('"codex:dynamicShelfCartonsEnabled"', source)
        self.assertIn(
            "level_z\n                    + shelf_surface_offset_m",
            source,
        )
        self.assertNotIn("shelf_carton_authored_half_height_m", source)
        self.assertIn("scale, yaw = shelf_carton_profiles[profile_index]", source)
        self.assertIn("(carton_x, carton_y, carton_z)", source)
        self.assertIn("scale=scale", source)
        self.assertIn("carton_dimensions_m = (", source)
        self.assertIn("UsdPhysics.RigidBodyAPI.Apply(carton)", source)
        self.assertIn(
            "shelf_physx_body.CreateEnableCCDAttr().Set(True)",
            source,
        )
        self.assertIn('"codex:shelfParcel"', source)
        self.assertIn('"codex:shelfCartonCount"', source)
        self.assertIn('"dynamic_shelf_carton_count": shelf_carton_count', source)
        self.assertNotIn("scale=(1.25, 1.25, 1.25)", source)
        self.assertIn("ConveyorBelt_A05_PR_NVD_01.usd", source)
        self.assertIn("ConveyorBelt_A08.usd", source)
        self.assertIn("ConveyorBelt_A11_PR_NVD_01.usd", source)
        self.assertIn("conveyor_oval_radius_m = 1.50", source)
        self.assertIn(
            "conveyor_oval_straight_half_length_m = 2.351374",
            source,
        )
        self.assertIn("conveyor_curve_segment_count = 12", source)
        self.assertIn("belt_center_x_m = 4.42", source)
        self.assertIn("belt_center_y_m = 1.268030", source)
        self.assertIn('"codex:beltCenterXM"', source)
        self.assertIn('"codex:conveyorLayout"', source)
        self.assertIn('"codex:conveyorOvalRadiusM"', source)
        self.assertIn('"codex:conveyorOvalStraightHalfLengthM"', source)
        self.assertIn('"codex:conveyorTrackHalfWidthM"', source)
        self.assertIn('"codex:conveyorSurfaceHeightM"', source)
        self.assertIn('"codex:conveyorColliderCenterXM"', source)
        self.assertIn('"codex:conveyorColliderHalfWidthM"', source)
        self.assertIn('"codex:conveyorColliderInsetM"', source)
        self.assertIn('"codex:conveyorAdditionalApproachM"', source)
        self.assertIn("conveyor_additional_approach_m = 0.50", source)
        self.assertIn("conveyor_collider_surface_inset_m = 0.05", source)
        self.assertIn(
            "conveyor_track_half_width_m"
            " - conveyor_collider_surface_inset_m",
            source,
        )
        self.assertIn('"codex:westRackCenterXM"', source)
        self.assertIn('"codex:rackHalfDepthM"', source)
        self.assertIn('"codex:rackHalfLengthM"', source)
        self.assertIn('"codex:floorHalfWidthM"', source)
        self.assertIn('"codex:floorHalfDepthM"', source)
        self.assertIn('"codex:northWallCenterYM"', source)
        self.assertIn('"codex:westWallCenterXM"', source)
        self.assertIn("floor_west_edge_m = -5.33", source)
        self.assertIn("floor_east_edge_m = 6.50", source)
        self.assertIn("north_wall_center_y_m", source)
        self.assertIn("wall_height_m = 6.0", source)
        self.assertIn("floor_finish_tile_half_width_m", source)
        self.assertIn('"codex:workerRadiusM"', source)
        self.assertIn('"codex:workerWalkSpeedMps"', source)
        self.assertIn("straight_origin_center_x_m = 0.5014", source)
        self.assertIn('"StraightWest"', source)
        self.assertIn('"StraightEast"', source)
        self.assertIn('"ExtensionWest"', source)
        self.assertIn('"ExtensionEast"', source)
        self.assertIn('"CurveNorth"', source)
        self.assertIn('"CurveSouth"', source)
        self.assertIn('for side, x in (("West", -4.20),):', source)
        self.assertIn("south_floor_extension_m = 1.45", source)
        self.assertIn("extended_floor_center_y_m", source)
        self.assertIn("extended_floor_half_depth_m", source)
        self.assertIn("marked_zone_south_edge_m = (", source)
        self.assertIn("marked_zone_north_edge_m = scene_half_depth_m - 0.25", source)
        self.assertIn("marked_zone_center_y_m = 0.5 * (", source)
        self.assertIn(
            "(marked_zone_center_x_m, marked_zone_center_y_m, 0.035)",
            source,
        )
        self.assertIn("SM_floor02.usd", source)
        self.assertIn("FloorFinish", source)
        self.assertIn("floor_finish_tile_positions", source)
        self.assertIn(
            '"warehouse_floor_tile_count": len(floor_finish_tile_positions) + 1',
            source,
        )
        self.assertIn("BehindWallFloor", source)
        self.assertIn("Tile4_01", source)
        self.assertIn("north_wall_segments = (", source)
        self.assertIn("PackingTable/packing_table.usd", source)
        self.assertIn("StaticColliders/PackingTable", source)
        self.assertIn('roughness=0.92', source)
        self.assertIn("UsdShade.MaterialBindingAPI.Apply", source)
        self.assertIn("/Shell/WestWall", source)
        self.assertNotIn('"EastWall"', source)
        self.assertNotIn('"SouthWall"', source)
        self.assertNotIn('("East", 8.8)', source)
        self.assertIn('"rack_sides": ["West"]', source)
        self.assertIn('"conveyor_module_count": conveyor_module_count', source)
        self.assertIn('"conveyor_layout": "oval"', source)
        self.assertIn("DynamicCargo", source)
        self.assertIn("UsdPhysics.Scene.Define", source)
        self.assertIn("CreateTimeStepsPerSecondAttr().Set(", source)
        self.assertIn("30\n)", source)
        self.assertIn("UsdPhysics.CollisionAPI.Apply", source)
        self.assertIn("disable_composed_collisions", source)
        self.assertIn('CreateApproximationAttr().Set("convexHull")', source)
        self.assertIn(
            '"continuous_low_poly_oval_mesh_plus_simple_boxes"',
            source,
        )
        self.assertIn("StaticColliders/OvalConveyor", source)
        self.assertIn("conveyor_collider_sample_count", source)
        self.assertIn("UsdGeom.Mesh.Define", source)
        self.assertIn("UsdPhysics.MeshCollisionAPI.Apply", source)
        self.assertIn('"closed_low_poly_oval_mesh"', source)
        self.assertIn("CreateKinematicEnabledAttr", source)
        self.assertIn("UsdPhysics.MassAPI.Apply", source)
        self.assertIn("Isaac/Props/Pallet/pallet.usd", source)
        self.assertIn("CreateMassAttr().Set(22.0)", source)
        self.assertIn('(0.00154, 0.00115, 0.1195),', source)
        self.assertIn('(0.615, 0.407, 0.024),', source)
        self.assertIn('(runner_x, 0.00115, 0.015),', source)
        self.assertIn('(runner_half_x, 0.407, 0.015),', source)
        self.assertIn("rotate=(0.0, 0.0, yaw + 90.0)", source)
        curve_block = source[
            source.index('"CurveNorth"'):
            source.index("def oval_point_tangent")
        ]
        pallet_block = source[
            source.index("pallet_path ="):
            source.index("cargo_path =", source.index("pallet_path ="))
        ]
        self.assertIn("rotate=(0.0, 0.0, yaw)", curve_block)
        self.assertNotIn("yaw + 90.0", curve_block)
        self.assertIn("rotate=(0.0, 0.0, yaw + 90.0)", pallet_block)
        self.assertIn('"codex:colliderShapeCount"', source)
        self.assertIn('"simple_top_deck_plus_three_runners"', source)
        self.assertIn("CreateMassAttr().Set(10.5)", source)
        self.assertIn('(0.0, 0.0, 0.25),', source)
        self.assertIn('(0.37, 0.35, 0.26),', source)
        self.assertIn('scale=(1.03, 1.36, 1.00)', source)
        self.assertIn(
            "pallet_physx_body.CreateEnableCCDAttr().Set(True)", source
        )
        self.assertIn(
            "carton_physx_body.CreateEnableCCDAttr().Set(True)", source
        )
        self.assertIn(
            "pallet_physx_body.CreateMaxDepenetrationVelocityAttr().Set(0.75)",
            source,
        )
        self.assertIn(
            "carton_physx_body.CreateMaxDepenetrationVelocityAttr().Set(0.75)",
            source,
        )
        self.assertIn("CreateSolverPositionIterationCountAttr().Set(8)", source)
        self.assertIn('"codex:colliderCenterM"', source)
        self.assertIn('"codex:colliderHalfExtentsM"', source)
        self.assertIn('"codex:visualDimensionsM"', source)
        self.assertIn('"codex:dynamicBeltParcelsEnabled"', source)
        self.assertIn('"codex:conveyorVelocityMps"', source)
        self.assertIn('"codex:conveyorFrictionCoefficient"', source)
        self.assertIn("UsdPhysics.MaterialAPI.Apply", source)
        self.assertIn("CreateStaticFrictionAttr().Set(0.0)", source)
        self.assertIn('CreateFrictionCombineModeAttr("min")', source)
        self.assertIn(
            "bind_physics_material(\n"
            "    conveyor_surface_collider.GetPrim()",
            source,
        )
        self.assertIn("np.random.default_rng(20260731)", source)
        self.assertIn("UsdPhysics.RigidBodyAPI.Apply(parcel)", source)
        self.assertIn("PhysxSchema.PhysxForceAPI.Apply(parcel)", source)
        self.assertIn("physx_body.CreateEnableCCDAttr().Set(True)", source)
        self.assertIn('"parcel_dimensions_randomized": True', source)
        self.assertIn("ForkliftC/forklift_c.usd", source)
        self.assertIn("source_xform.AddScaleOp", source)
        self.assertIn("Gf.Vec3d(0.01, 0.01, 0.01)", source)
        self.assertIn("from isaacsim.core.cloner import Cloner", source)
        self.assertIn("copy_from_source=True", source)
        self.assertIn("left_rotator_joint", source)
        self.assertIn("right_rotator_joint", source)
        self.assertIn("left_front_wheel_joint", source)
        self.assertIn("right_back_wheel_joint", source)
        self.assertIn('"forklift_lift_joint": "lift_joint"', source)
        self.assertIn('"forklift_lift_range_m": [0.0, 2.0]', source)
        self.assertIn("codex:forkHeightM", source)
        self.assertIn("ForkliftC rear-steer Ackermann articulation", source)
        self.assertNotIn("ForkSupports/Support", source)
        self.assertIn("codex:collisionBlockingEnabled", source)
        self.assertIn("codex:workerPushEnabled", source)
        self.assertIn("forklift_proximity_carton_", source)
        self.assertNotIn("ReflectiveForkCaps", source)
        self.assertIn(CAMERA_PATHS["detector_oblique"], source)
        self.assertIn(CAMERA_PATHS["detector_overhead"], source)
        self.assertIn(CAMERA_PATHS["detector_ptz"], source)
        self.assertIn(CAMERA_PATHS["detector_isometric"], source)
        self.assertIn(CAMERA_PATHS["presentation"], source)
        self.assertIn("UsdGeom.Tokens.orthographic", source)
        self.assertIn("stack_light_roots = (", source)
        self.assertIn("StackLight_01", source)
        self.assertIn("StackLight_02", source)
        self.assertIn("stack_light_scale = 0.216", source)
        self.assertIn("2.875", source)
        self.assertIn("set_xform(wash.GetPrim(), (0.0, 0.0, 0.465762))", source)
        self.assertNotIn("/Base", source)
        self.assertNotIn("/ClearanceBoundary/", source)
        self.assertNotIn("/TrafficLanes/", source)
        self.assertNotIn("/AmberWarningBand/", source)
        self.assertIn('"stack_light_visible_to_detector": True', source)
        self.assertIn('"stack_light_active_intensity": 60000.0', source)
        self.assertIn('"stack_light_wash_intensity": 50000.0', source)
        self.assertIn("stack_scene_bounce_intensity = 220000.0", source)
        self.assertIn("stack_scene_bounce_radius = 0.30", source)
        self.assertIn("UsdLux.DiskLight.Define(stage, bounce_path)", source)
        self.assertIn('"stack_light_bounce_count"', source)
        self.assertIn("stack_lens_glow_radius = 0.01", source)
        self.assertIn("stack_scene_wash_radius = 0.01", source)
        self.assertIn(
            "glow.CreateRadiusAttr(stack_lens_glow_radius)", source
        )
        self.assertIn(
            "wash.CreateRadiusAttr(stack_scene_wash_radius)", source
        )
        self.assertEqual(source.count("CreateTreatAsPointAttr(True)"), 2)
        self.assertIn("stack_light_color_bias_monitoring", source)

    def test_detector_and_presentation_capture_keep_tower_visible(self) -> None:
        detector = capture_conveyor_safety_camera_source(
            {"camera": "detector_oblique", "filename": "detector.png"}
        )
        presentation = capture_conveyor_safety_camera_source(
            {"camera": "presentation", "filename": "presentation.png"}
        )
        self.assertNotIn("saved_visibility", detector)
        self.assertIn('"stack_light_visible_to_model": True', detector)
        self.assertIn('cache_module_name = "_qai_conveyor_capture_cache"', detector)
        self.assertIn('"render_resource_reused"', detector)
        self.assertIn("capture_cache.resource = resource", detector)
        self.assertIn("hydra_texture.set_updates_enabled(True)", detector)
        self.assertIn("hydra_texture.set_updates_enabled(False)", detector)
        self.assertIn('"_qai_conveyor_capture_state"', detector)
        self.assertIn("capture_state.depth", detector)
        self.assertIn(
            '"dynamic_updates_suppressed_during_capture": True',
            detector,
        )
        self.assertIn(
            '"render_product_updates_disabled_between_captures"',
            detector,
        )
        self.assertIn(
            '"active_forklift_halos_visible_to_model": False',
            detector,
        )
        self.assertIn("bounding_box_2d_tight", detector)
        self.assertIn('"fork_carton_pixel_boxes"', detector)
        self.assertIn('"forklift_pixel_boxes"', detector)
        self.assertIn('"enabled_forklift_indices"', detector)
        self.assertIn('"capture_ground_truth_signal"', detector)
        self.assertIn('"capture_min_forklift_clearance_m"', detector)
        self.assertIn('"capture_active_forklift"', detector)
        self.assertIn("safety_forklift_", detector)
        self.assertIn('or "/DriverMount/" in semantic_path', detector)
        self.assertIn('"workers_included_in_detection": False', detector)
        self.assertNotIn("was_playing", detector)
        self.assertIn("pause_timeline=False", detector)
        self.assertIn("delta_time=0.0", detector)
        self.assertIn('"stack_light_visible_to_model": True', presentation)
        self.assertIn(
            '"active_forklift_halos_visible_to_model": True',
            presentation,
        )

    def test_light_accepts_only_three_signals(self) -> None:
        for signal in ("GREEN", "AMBER", "RED"):
            source = set_conveyor_safety_light_source({"signal": signal})
            self.assertIn('"codex:modelSignal"', source)
            self.assertIn(repr(signal), source)
            self.assertIn("if light_mutated:", source)
            self.assertIn('"light_mutated": light_mutated', source)
            self.assertIn("for stack_root_path in stack_light_roots:", source)
            self.assertIn("required_light_paths", source)
            self.assertIn("light_geometry_is_current", source)
            self.assertIn("bounce_geometry_is_current", source)
            self.assertIn("GetTreatAsPointAttr()", source)
            self.assertIn("CreateTreatAsPointAttr(True)", source)
            self.assertIn("UsdLux.DiskLight(bounce_prim)", source)
            self.assertIn('"scene_bounce_intensity"', source)
            self.assertIn("stack_light_point_radius = 0.01", source)
            self.assertNotIn("GetRadiusAttr().Set(1.30)", source)
            self.assertIn('"stack_light_count": len(stack_light_roots)', source)
        with self.assertRaisesRegex(ValueError, "GREEN, AMBER, or RED"):
            set_conveyor_safety_light_source({"signal": "BLUE"})

    def test_light_and_post_apply_state_share_one_source(self) -> None:
        source = set_conveyor_safety_light_and_get_state_source(
            {
                "signal": "AMBER",
                "confidence": 1.0,
                "inference_id": "combined-call",
            }
        )
        self.assertIn('"light_mutated": light_mutated', source)
        self.assertIn('"ground_truth_signal": ground_truth', source)
        self.assertIn('"last_light_mutated"', source)

    def test_pose_is_bounded(self) -> None:
        source = set_conveyor_safety_forklift_pose_source(
            {"index": 2, "x": 3.0, "y": -2.0, "yaw_degrees": 90.0}
        )
        self.assertIn("Forklift2", source)
        self.assertIn("forklift_xform.GetOrderedXformOps()", source)
        self.assertIn('sys.modules.get("_qai_conveyor_capture_cache")', source)
        self.assertIn('resource["bbox"].detach()', source)
        with self.assertRaisesRegex(ValueError, "inside the factory"):
            set_conveyor_safety_forklift_pose_source(
                {"index": 1, "x": 20.0, "y": 0.0, "yaw_degrees": 0.0}
            )

    def test_state_keeps_validation_separate_from_model(self) -> None:
        source = get_conveyor_safety_state_source()
        self.assertIn("ground_truth_signal", source)
        self.assertIn("model_signal", source)
        self.assertIn('"dynamic_cargo"', source)
        self.assertIn('"collision_block_count"', source)
        self.assertIn('"worker_push_enabled"', source)
        self.assertIn('"actual_speed_mps"', source)
        self.assertIn('"fork_height_m"', source)
        self.assertIn('"stack_light_visible_to_detector": True', source)
        self.assertIn('"light_mutation_count"', source)

    def test_capture_performance_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual((args.capture_width, args.capture_height), (512, 288))
        self.assertEqual(args.capture_render_steps, 1)
        self.assertEqual(args.capture_rt_subframes, 1)
        self.assertEqual(args.maximum_capture_fps, 0.0)
        self.assertEqual(args.perception_mode, "direct")

    def test_scene_uses_lower_wide_endline_detector(self) -> None:
        source = create_conveyor_safety_scene_source(
            {"new_stage": True, "start_playing": True}
        )
        self.assertIn("(marked_zone_west_edge_m, -6.10, 2.45)", source)
        self.assertIn("(marked_zone_west_edge_m, 0.10, 0.80)", source)
        self.assertIn("    9.5,", source)
        self.assertIn(
            "CreateHorizontalApertureOffsetAttr(-3.0)",
            source,
        )
        self.assertNotIn("(0.2, -6.2, 13.5)", source)


class ConveyorSafetyPromptTests(unittest.TestCase):
    @staticmethod
    def _write_test_png(
        path: Path,
        carton_rows: range,
        *,
        off_center_distractor: bool = False,
    ) -> None:
        width = 400 if off_center_distractor else 100
        height = 100
        carton_x_min = width // 2 - 7
        carton_x_max = width // 2 + 6
        rows = []
        for y in range(height):
            row = bytearray()
            for x in range(width):
                color = (220, 220, 220)
                if y >= 70:
                    color = (230, 35, 30)
                elif y >= 65:
                    color = (240, 160, 25)
                if carton_x_min <= x <= carton_x_max and y in carton_rows:
                    color = (220, 205, 170)
                if (
                    off_center_distractor
                    and 170 <= x <= 183
                    and 20 <= y <= 39
                ):
                    color = (220, 205, 170)
                row.extend(color)
            rows.append(b"\x00" + bytes(row))

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)

    def test_prompt_is_simple_visual_only_policy(self) -> None:
        self.assertIn(
            "Return R if any part of any forklift is inside the marked red zone.",
            HAZARD_PROMPT,
        )
        self.assertIn(
            "return A if any forklift is moving at all",
            HAZARD_PROMPT,
        )
        self.assertNotIn("moving toward", HAZARD_PROMPT)
        self.assertIn("Otherwise, return G.", HAZARD_PROMPT)
        self.assertIn(
            "Ignore human workers, parcels, and the stack light.",
            HAZARD_PROMPT,
        )
        self.assertIn("newest image last", HAZARD_PROMPT)
        for leaked_fact in (
            "present_red_zone_overlap",
            "predicted_red_zone_entry",
            "frontend_proposed_signal",
            "REQUIRED_OUTPUT",
            "camera-derived facts",
        ):
            self.assertNotIn(leaked_fact, HAZARD_PROMPT)

    def test_red_confirmation_is_visual_only_and_binary(self) -> None:
        self.assertIn(
            "tire, body, mast, or fork",
            RED_CONFIRMATION_PROMPT,
        )
        self.assertIn("red-painted floor", RED_CONFIRMATION_PROMPT)
        self.assertIn("Return one letter only: R or G.", RED_CONFIRMATION_PROMPT)
        for leaked_fact in (
            "present_red_zone_overlap",
            "clearance",
            "REQUIRED_OUTPUT",
            "camera-derived facts",
        ):
            self.assertNotIn(leaked_fact, RED_CONFIRMATION_PROMPT)

    def test_amber_confirmation_is_visual_only_and_temporal(self) -> None:
        self.assertIn("images are chronological", AMBER_CONFIRMATION_PROMPT)
        self.assertIn(
            "any forklift moved at all, in any direction",
            AMBER_CONFIRMATION_PROMPT,
        )
        self.assertNotIn("moved closer", AMBER_CONFIRMATION_PROMPT)
        self.assertIn(
            "Return one letter only: A or G.",
            AMBER_CONFIRMATION_PROMPT,
        )
        for leaked_fact in (
            "predicted_red_zone_entry",
            "velocity",
            "time_to_red_zone",
            "camera-derived facts",
        ):
            self.assertNotIn(leaked_fact, AMBER_CONFIRMATION_PROMPT)

    def test_rgb_carton_fallback_resolves_all_three_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            expected = {
                "GREEN": (range(30, 42), False, False),
                "AMBER": (range(60, 69), False, True),
                "RED": (range(66, 78), True, True),
            }
            for signal, (carton_rows, red_overlap, amber_overlap) in expected.items():
                with self.subTest(signal=signal):
                    self._write_test_png(path, carton_rows)
                    observation = extract_visual_overlap_observation(path, [])
                    self.assertEqual(
                        observation["source"],
                        "camera_rgb_carton_component_plus_floor_masks",
                    )
                    self.assertEqual(observation["red_overlap"], red_overlap)
                    self.assertEqual(observation["amber_overlap"], amber_overlap)

    def test_rgb_carton_fallback_ignores_off_center_tan_distractor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distractor-frame.png"
            self._write_test_png(
                path,
                range(66, 78),
                off_center_distractor=True,
            )

            observation = extract_visual_overlap_observation(path, [])

        self.assertTrue(observation["red_overlap"])
        self.assertEqual(
            observation["fork_carton_pixel_boxes"][0]["x_min"],
            193,
        )

    def test_motion_observation_predicts_entry_and_ignores_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion-frame.png"
            self._write_test_png(path, range(10, 20))
            current_boxes = [
                {
                    "x_min": 45,
                    "y_min": 40,
                    "x_max": 55,
                    "y_max": 50,
                    "semantic_label": "safety_forklift_1",
                    "forklift_index": 1,
                },
                {
                    "x_min": 15,
                    "y_min": 10,
                    "x_max": 25,
                    "y_max": 20,
                    "semantic_label": "safety_forklift_2",
                    "forklift_index": 2,
                },
                {
                    "x_min": 75,
                    "y_min": 10,
                    "x_max": 85,
                    "y_max": 20,
                    "semantic_label": "safety_forklift_3",
                    "forklift_index": 3,
                },
            ]
            previous = {
                "forklifts": [
                    {
                        "semantic_label": item["semantic_label"],
                        "center": [
                            (item["x_min"] + item["x_max"]) * 0.5,
                            (item["y_min"] + item["y_max"]) * 0.5
                            - (10 if item["forklift_index"] == 1 else 0),
                        ],
                    }
                    for item in current_boxes
                ]
            }

            observation = extract_forklift_motion_observation(
                path,
                current_boxes,
                previous_observation=previous,
                elapsed_seconds=1.0,
                prediction_horizon_seconds=3.0,
            )

        self.assertFalse(observation["present_red_zone_overlap"])
        self.assertTrue(observation["forklift_motion_detected"])
        self.assertTrue(observation["predicted_red_zone_entry"])
        self.assertTrue(observation["workers_ignored"])
        self.assertEqual(
            observation["forklifts"][0]["time_to_red_zone_seconds"],
            2.1,
        )
        self.assertEqual(observation["red_overlap_inset_pixels"], 1)

    def test_component_boxes_remove_empty_union_corner_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "components.png"
            self._write_test_png(path, range(10, 20))
            observation = extract_forklift_motion_observation(
                path,
                [
                    {
                        "x_min": 10,
                        "y_min": 10,
                        "x_max": 90,
                        "y_max": 90,
                        "semantic_label": "safety_forklift_1",
                        "forklift_index": 1,
                        "component_boxes": [
                            {
                                "x_min": 10,
                                "y_min": 10,
                                "x_max": 45,
                                "y_max": 60,
                            },
                            {
                                "x_min": 46,
                                "y_min": 10,
                                "x_max": 65,
                                "y_max": 69,
                            },
                        ],
                    }
                ],
                expected_forklift_count=1,
            )

        self.assertFalse(observation["present_red_zone_overlap"])
        self.assertEqual(
            len(observation["forklifts"][0]["component_boxes"]),
            2,
        )

    def test_visual_gates_delay_early_red_and_reject_static_amber(self) -> None:
        early_red = apply_visual_consistency_gates(
            {
                "signal": "RED",
                "normalized_answer": '{"signal":"RED"}',
                "raw_answer": "R",
            },
            {
                "present_red_zone_overlap": False,
                "forklift_motion_detected": True,
                "predicted_red_zone_entry": True,
            },
        )
        static_amber = apply_visual_consistency_gates(
            {
                "signal": "AMBER",
                "normalized_answer": '{"signal":"AMBER"}',
                "raw_answer": "A",
            },
            {
                "present_red_zone_overlap": False,
                "forklift_motion_detected": False,
                "predicted_red_zone_entry": False,
            },
        )

        self.assertEqual(early_red["signal"], "AMBER")
        self.assertEqual(early_red["raw_answer"], "R")
        self.assertEqual(early_red["pre_gate_signal"], "RED")
        self.assertTrue(early_red["gate_applied"])
        self.assertIn("no visible red-zone contact", early_red["gate_reason"])
        self.assertEqual(static_amber["signal"], "GREEN")
        self.assertEqual(static_amber["raw_answer"], "A")
        self.assertEqual(static_amber["pre_gate_signal"], "AMBER")
        self.assertTrue(static_amber["gate_applied"])
        self.assertIn("no visible forklift motion", static_amber["gate_reason"])

    def test_visual_gate_accepts_motion_away_from_red_as_amber(self) -> None:
        moving_amber = apply_visual_consistency_gates(
            {
                "signal": "AMBER",
                "normalized_answer": '{"signal":"AMBER"}',
                "raw_answer": "A",
            },
            {
                "present_red_zone_overlap": False,
                "forklift_motion_detected": True,
                "predicted_red_zone_entry": False,
            },
        )

        self.assertEqual(moving_amber["signal"], "AMBER")
        self.assertFalse(moving_amber["gate_applied"])
        self.assertIsNone(moving_amber["gate_reason"])

    def test_rgb_vehicle_fallback_fills_one_transient_semantic_miss(self) -> None:
        width = 200
        height = 120
        regions = (
            (28, 72, 57, 91),
            (85, 24, 114, 43),
            (142, 58, 171, 77),
        )
        rows = []
        for y in range(height):
            row = bytearray()
            for x in range(width):
                color = (220, 220, 220)
                if any(
                    x_min <= x <= x_max and y_min <= y <= y_max
                    for x_min, y_min, x_max, y_max in regions
                ):
                    color = (230, 180, 20)
                row.extend(color)
            rows.append(b"\x00" + bytes(row))

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            )
            + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b"")
        )
        semantic_boxes = [
            {
                "x_min": 20,
                "y_min": 65,
                "x_max": 70,
                "y_max": 100,
                "semantic_label": "safety_forklift_1",
                "forklift_index": 1,
            },
            {
                "x_min": 128,
                "y_min": 48,
                "x_max": 185,
                "y_max": 90,
                "semantic_label": "safety_forklift_3",
                "forklift_index": 3,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yellow-vehicles.png"
            path.write_bytes(png)
            recovered = recover_forklift_boxes_from_rgb(
                path,
                semantic_boxes,
            )

        self.assertEqual(
            [item["forklift_index"] for item in recovered],
            [1, 2, 3],
        )
        self.assertEqual(
            recovered[1]["box_source"],
            "camera_rgb_yellow_vehicle_fallback",
        )

    def test_single_forklift_focus_accepts_one_semantic_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.png"
            self._write_test_png(path, range(10, 20))
            boxes = [
                {
                    "x_min": 35,
                    "y_min": 30,
                    "x_max": 65,
                    "y_max": 60,
                    "semantic_label": "safety_forklift_1",
                    "forklift_index": 1,
                }
            ]
            recovered = recover_forklift_boxes_from_rgb(
                path,
                boxes,
                expected_forklift_indices=(1,),
            )
            observation = extract_forklift_motion_observation(
                path,
                recovered,
                expected_forklift_count=1,
            )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(observation["visible_forklift_count"], 1)

    def test_request_sends_image_and_visual_only_prompt(self) -> None:
        captured: dict[str, object] = {}

        def urlopen(request: object, timeout: float) -> _Response:
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
            return _Response(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": '{"signal":"AMBER"}'}}
                        ],
                        "usage": {"completion_tokens": 5},
                    }
                ).encode("utf-8")
            )

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            result = request_hazard_signal(
                image_path=image,
                server_url="http://127.0.0.1:18080",
                model="test-model",
                timeout_seconds=7.0,
                urlopen=urlopen,
                vision_observation={
                    "present_red_zone_overlap": False,
                    "predicted_red_zone_entry": True,
                    "prediction_horizon_seconds": 3.0,
                    "forklifts": [
                        {
                            "forklift_index": 1,
                            "present_red_zone_overlap": False,
                            "predicted_red_zone_entry": True,
                            "time_to_red_zone_seconds": 1.5,
                        }
                    ],
                },
            )

        self.assertEqual(result["signal"], "AMBER")
        payload = captured["payload"]
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload["enable_think"])  # type: ignore[index]
        self.assertNotIn("grammar_string", payload)
        content = payload["messages"][0]["content"]  # type: ignore[index]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(
            content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], HAZARD_PROMPT)
        self.assertNotIn("present_red_zone_overlap", content[1]["text"])
        self.assertNotIn("REQUIRED_OUTPUT", content[1]["text"])

    def test_request_supports_evk_constrained_transport(self) -> None:
        captured: dict[str, object] = {}

        def urlopen(request: object, timeout: float) -> _Response:
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
            return _Response(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": '{"signal":"RED"}'}}
                        ]
                    }
                ).encode("utf-8")
            )

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "evk-frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            result = request_hazard_signal(
                image_path=image,
                server_url="http://127.0.0.1:18181",
                model="local/cosmos-reason2-2b:Q4_0",
                timeout_seconds=120.0,
                urlopen=urlopen,
                vision_observation={
                    "present_red_zone_overlap": True,
                    "predicted_red_zone_entry": False,
                    "prediction_horizon_seconds": 3.0,
                    "forklifts": [
                        {
                            "forklift_index": 1,
                            "present_red_zone_overlap": True,
                            "predicted_red_zone_entry": False,
                            "time_to_red_zone_seconds": None,
                        }
                    ],
                },
                enable_think=False,
                use_signal_grammar=True,
            )

        self.assertEqual(result["signal"], "RED")
        payload = captured["payload"]
        self.assertIsInstance(payload, dict)
        self.assertFalse(payload["enable_think"])  # type: ignore[index]
        self.assertIn("grammar_string", payload)
        self.assertEqual(payload["grammar_string"], "root ::= [GAR]")  # type: ignore[index]
        self.assertEqual(payload["grammar"], "root ::= [GAR]")  # type: ignore[index]

    def test_play_mode_gate_waits_without_inference_until_timeline_plays(
        self,
    ) -> None:
        states = iter(
            [
                {"timeline_playing": False},
                {"timeline_playing": False},
                {"timeline_playing": True, "ground_truth_signal": "RED"},
            ]
        )
        sleeps: list[float] = []
        waited_states: list[dict[str, object]] = []

        result = wait_for_timeline_playing(
            object(),  # type: ignore[arg-type]
            poll_seconds=0.25,
            state_reader=lambda _client: next(states),
            on_wait=waited_states.append,
            sleep_fn=sleeps.append,
        )

        self.assertTrue(result["timeline_playing"])
        self.assertEqual(result["ground_truth_signal"], "RED")
        self.assertEqual(sleeps, [0.25, 0.25])
        self.assertEqual(len(waited_states), 2)


class ConveyorSafetyExtensionTests(unittest.TestCase):
    def test_extension_maps_requested_xbox_controls(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "isaac_sim_supervisor_omniverse"
            / "exts"
            / "qai.conveyor_safety"
            / "qai"
            / "conveyor_safety"
            / "extension.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("GamepadInput.LEFT_STICK_UP", source)
        self.assertIn("GamepadInput.LEFT_STICK_DOWN", source)
        self.assertIn("GamepadInput.LEFT_STICK_LEFT", source)
        self.assertIn("GamepadInput.LEFT_STICK_RIGHT", source)
        self.assertIn("GamepadInput.LEFT_TRIGGER", source)
        self.assertIn("GamepadInput.RIGHT_TRIGGER", source)
        self.assertIn("GamepadInput.DPAD_UP", source)
        self.assertIn("GamepadInput.DPAD_DOWN", source)
        self.assertIn("GamepadInput.RIGHT_STICK_LEFT", source)
        self.assertIn("GamepadInput.RIGHT_STICK_RIGHT", source)
        self.assertIn("GamepadInput.RIGHT_STICK_UP", source)
        self.assertIn("GamepadInput.RIGHT_STICK_DOWN", source)
        self.assertIn("GamepadInput.LEFT_SHOULDER", source)
        self.assertIn("GamepadInput.RIGHT_SHOULDER", source)
        self.assertIn("GamepadInput.X", source)
        self.assertIn("GamepadInput.Y", source)
        self.assertIn("GamepadInput.MENU1", source)
        self.assertIn("KeyboardInput.R", source)
        self.assertIn("KeyboardInput.I", source)
        self.assertIn("KeyboardInput.K", source)
        self.assertIn("self._animate_parcels", source)
        self.assertIn("self._animate_workers", source)
        self.assertIn("self._pose_collision", source)
        self.assertIn(
            'root_float("codex:conveyorColliderCenterXM", 4.7014)',
            source,
        )
        self.assertIn(
            'root_float("codex:conveyorColliderCenterYM", -1.3594)',
            source,
        )
        self.assertIn(
            'root_float("codex:conveyorColliderHalfWidthM", 0.5255)',
            source,
        )
        self.assertIn("self._conveyor_obstacle_aabbs", source)
        self.assertIn(
            'root_float("codex:conveyorOvalRadiusM", 1.50)',
            source,
        )
        self.assertIn(
            '"codex:conveyorOvalStraightHalfLengthM"',
            source,
        )
        self.assertIn('root_float("codex:westRackCenterXM", -4.20)', source)
        self.assertIn("FORKLIFT_MAX_SPEED_MPS = 1.00", source)
        self.assertIn("FORKLIFT_MAX_ACCELERATION_MPS2 = 1.00", source)
        self.assertIn("FORKLIFT_MAX_STEERING_RATE_RAD_S = 1.20", source)
        self.assertIn("FORKLIFT_CONTROL_UPDATE_HZ = 30.0", source)
        self.assertIn("PARCEL_FORCE_UPDATE_HZ = 15.0", source)
        self.assertIn("WORKER_ROUTE_UPDATE_HZ = 20.0", source)
        self.assertIn("self._forklift_control_accumulator", source)
        self.assertIn("self._parcel_force_accumulator", source)
        self.assertIn("self._worker_route_accumulator", source)
        self.assertIn('sys.modules.get("_qai_conveyor_capture_state")', source)
        self.assertIn("if not capture_in_progress:", source)
        self.assertIn("self._last_steering_targets", source)
        self.assertIn("self._last_wheel_targets", source)
        self.assertIn("self._last_lift_targets_sent", source)
        self.assertIn("not np.allclose(", source)
        self.assertIn("position_target_parts", source)
        self.assertIn("_enabled_forklift_indices", source)
        self.assertNotIn('("conveyor", -0.50', source)
        self.assertNotIn('("east shelving", 8.80', source)
        self.assertIn("self._obb_overlaps_aabb", source)
        self.assertIn("self._worker_contact_normal", source)
        self.assertIn("self._worker_obstacles", source)
        self.assertIn("self._worker_obstacle_at", source)
        self.assertIn("self._resolve_worker_motion", source)
        self.assertIn("self._worker_segment_obstacle", source)
        self.assertIn("self._plan_worker_corridor", source)
        self.assertIn("heapq.heappush", source)
        self.assertIn("WORKER_CORRIDOR_NODES", source)
        self.assertIn("WORKER_CORRIDOR_EDGES", source)
        self.assertIn("self._smooth_worker_heading", source)
        self.assertIn("self._worker_locomotion_weights", source)
        self.assertIn("self._worker_animation_blends", source)
        self.assertIn("WORKER_REPLAN_BLOCK_SECONDS", source)
        self.assertIn("self._set_worker_motion_animation", source)
        self.assertIn("self._worker_route_positions", source)
        self.assertIn("self._worker_route_waypoint_indices", source)
        self.assertIn("self._worker_route_dwell_remaining", source)
        self.assertIn('"codex:routeWaypoints"', source)
        self.assertIn('"codex:routeDwellSeconds"', source)
        self.assertIn('"codex:routeFacingYawDegrees"', source)
        self.assertIn('"codex:dynamicBeltParcelsEnabled"', source)
        self.assertIn('"codex:conveyorVelocityMps"', source)
        self.assertIn("lateral_distance", source)
        self.assertIn("PhysxSchema.PhysxForceAPI", source)
        self.assertIn("force_attr.Set", source)
        self.assertIn("preserve_outward_speed", source)
        self.assertNotIn("self._worker_patrol_progress", source)
        self.assertIn("for skeleton_prim in Usd.PrimRange(skel_root):", source)
        self.assertIn('for cargo_kind in ("Pallet", "Carton"):', source)
        self.assertIn("loose ", source)
        self.assertIn("UsdGeom.BBoxCache", source)
        self.assertIn('"codex:walkAnimation"', source)
        self.assertIn('"codex:idleAnimation"', source)
        self.assertIn("WORKER_WALK_SPEED_MPS = 0.55", source)
        self.assertIn("WORKER_RADIUS_M = 0.38", source)
        self.assertIn("proposed_target_index = original_target_index", source)
        self.assertIn("proposed_dwell_remaining = original_dwell_remaining", source)
        self.assertIn("visible_move_blocked", source)
        self.assertIn("WORKER_ANIMATION_WALK_THRESHOLD", source)
        self.assertIn("WORKER_ANIMATION_IDLE_THRESHOLD", source)
        self.assertIn("self._reset_active_cargo", source)
        self.assertIn("self._reset_active_cargo_async", source)
        self.assertIn("from isaacsim.core.experimental.prims import Articulation", source)
        self.assertIn("AckermannController", source)
        self.assertIn("invert_steering=True", source)
        self.assertIn("FORKLIFT_OPERATOR_STEERING_SIGN = -1.0", source)
        self.assertIn(
            "steering = FORKLIFT_OPERATOR_STEERING_SIGN * max(",
            source,
        )
        self.assertIn("FORKLIFT_SERVICE_BRAKE_MPS2 = 1.50", source)
        self.assertIn("COLLISION_CONTACT_LOOKAHEAD_SECONDS = 0.05", source)
        self.assertIn("SOFTWARE_COLLISION_THROTTLE_GUARD = False", source)
        self.assertIn(
            "SOFTWARE_COLLISION_THROTTLE_GUARD\n"
            "            and abs(commanded_speed)",
            source,
        )
        self.assertNotIn("COLLISION_BRAKE_LOOKAHEAD_SECONDS", source)
        self.assertNotIn("stopping_distance", source)
        self.assertIn("FORKLIFT_MAX_STEERING_RATE_RAD_S", source)
        self.assertIn("FORKLIFT_LIFT_MAX_M = 2.0", source)
        self.assertIn("FORKLIFT_LIFT_SPEED_MPS = 0.90", source)
        self.assertIn("FORKLIFT_LIFT_ACCELERATION_MPS2 = 1.35", source)
        self.assertIn("self._lift_velocities_mps", source)
        self.assertIn("desired_lift_velocity", source)
        self.assertIn('["lift_joint"]', source)
        self.assertIn("self._lift_targets_m", source)
        self.assertIn("self._update_presentation_camera(", source)
        self.assertIn("PRESENTATION_CAMERA_PATH", source)
        self.assertIn(
            "PRESENTATION_CAMERA_DISTANCES_M = (5.5, 10.0, 15.0)",
            source,
        )
        self.assertIn("self._previous_view_button", source)
        self.assertIn(
            "view_pressed and not self._previous_view_button",
            source,
        )
        self.assertIn(
            "self._cycle_presentation_camera_distance()",
            source,
        )
        self.assertIn(
            "def _cycle_presentation_camera_distance(self) -> float:",
            source,
        )
        self.assertIn("codex:presentationCameraDistanceM", source)
        self.assertIn("codex:presentationCameraViewIndex", source)
        self.assertIn("get_active_viewport()", source)
        self.assertIn("matrix.SetLookAt(eye, target", source)
        self.assertIn(
            '"CAMERA  Right stick orbit · View cycles zoom',
            source,
        )
        self.assertIn("prev_linear_velocity", source)
        self.assertNotIn("_update_fork_supports", source)
        self.assertIn("self._timeline.stop()", source)
        self.assertIn("codex:lastCollision", source)
        self.assertIn("QAI_CONVEYOR_AUTO_INFERENCE", source)
        self.assertIn("QAI_CONVEYOR_MAX_CAPTURE_FPS", source)
        self.assertIn("self._maximum_capture_fps = 1.0", source)
        self.assertIn("str(self._maximum_capture_fps)", source)
        self.assertIn("BACKEND_OVERRIDE_PATH", source)
        self.assertIn("def _toggle_inference_backend", source)
        self.assertIn("def _set_inference_backend", source)
        self.assertIn("self._previous_backend_button", source)
        self.assertIn("Inference target:", source)
        self.assertIn("clicked_fn=self._toggle_inference_backend", source)
        self.assertIn("QAI_CONVEYOR_HOST_SERVER_URL", source)
        self.assertIn("QAI_CONVEYOR_EVK_SERVER_URL", source)
        self.assertIn("BACKEND_OVERRIDE_PATH.write_text", source)
        self.assertIn("self._ensure_inference_running()", source)
        self.assertIn("def _start_evk_service_recycle", source)
        self.assertIn("self._evk_recycle_process", source)
        self.assertIn(
            '"35" if self._inference_backend == "evk" else "1000000"',
            source,
        )
        self.assertIn(
            '"15" if self._inference_backend == "evk" else "120"',
            source,
        )
        self.assertIn("pkill -TERM -x geniex-grammar", source)
        self.assertIn('"--play-mode-only"', source)
        self.assertIn("subprocess.Popen", source)
        self.assertIn('"Reason2 request trace"', source)
        self.assertIn('"Rolling input images · previous / current"', source)
        self.assertIn('"Exact prompt"', source)
        self.assertIn('"Raw model output"', source)
        self.assertIn("last_inference_seconds", source)
        self.assertIn("stack_light_status", source)
        self.assertIn(
            '"HARD COLLIDERS  oval conveyor + rack + floor"',
            source,
        )
        self.assertIn('self._trace_image_paths = ["", ""]', source)
        self.assertIn(
            "if next_path != self._trace_image_paths[index]:",
            source,
        )
        self.assertIn('"--capture-width"', source)
        self.assertIn('"--capture-rt-subframes"', source)
        self.assertIn('"--maximum-capture-fps"', source)
        self.assertIn("str(self._maximum_capture_fps)", source)
        self.assertIn(
            '"384" if self._inference_backend == "evk" else "512"',
            source,
        )
        self.assertNotIn("set_conveyor_safety_light", source)


if __name__ == "__main__":
    unittest.main()
