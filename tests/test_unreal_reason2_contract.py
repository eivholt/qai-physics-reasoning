from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLD_SOURCE = (
    PROJECT_ROOT
    / "unreal_conveyor_demo"
    / "Source"
    / "QaiConveyor"
    / "QaiConveyorWorld.cpp"
)
WORLD_HEADER = WORLD_SOURCE.with_suffix(".h")
GAME_MODE_SOURCE = WORLD_SOURCE.with_name("QaiConveyorGameMode.cpp")
PAWN_SOURCE = WORLD_SOURCE.with_name("QaiConveyorPawn.cpp")
WINDOWS_ENGINE_CONFIG = (
    PROJECT_ROOT
    / "unreal_conveyor_demo"
    / "Config"
    / "Windows"
    / "WindowsEngine.ini"
)
DATASET_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "reason2_finetune"
    / "generate_unreal_dataset.ps1"
)
PACKAGE_SCRIPT = (
    PROJECT_ROOT
    / "unreal_conveyor_demo"
    / "Scripts"
    / "package_windows_client.ps1"
)
RUN_WINDOWS_DEMO_SCRIPT = PACKAGE_SCRIPT.with_name("run_windows_demo.ps1")
DEFAULT_INPUT_CONFIG = (
    PROJECT_ROOT / "unreal_conveyor_demo" / "Config" / "DefaultInput.ini"
)
DEFAULT_GAME_USER_SETTINGS = (
    PROJECT_ROOT
    / "unreal_conveyor_demo"
    / "Config"
    / "DefaultGameUserSettings.ini"
)
EVK_SERVICE = (
    PROJECT_ROOT
    / "unreal_conveyor_demo"
    / "Provisioner"
    / "evk-services"
    / "qai-conveyor-geniex.service"
)
VISUAL_FINISH_SCRIPT = (
    PROJECT_ROOT / "unreal_conveyor_demo" / "Scripts" / "visual_finish.py"
)
OMNIVERSE_EXTENSIONS_SCRIPT = (
    PROJECT_ROOT
    / "unreal_conveyor_demo"
    / "Scripts"
    / "import_omniverse_conveyor_extensions.py"
)


class UnrealReason2ContractTests(unittest.TestCase):
    def test_host_and_evk_default_to_promoted_speed_resolution(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")

        def constant(name: str) -> int:
            match = re.search(rf"constexpr int32 {name} = (\d+);", source)
            self.assertIsNotNone(match, name)
            return int(match.group(1))

        self.assertEqual(constant("DefaultHostCaptureWidth"), 448)
        self.assertEqual(constant("DefaultHostCaptureHeight"), 256)
        self.assertEqual(constant("DefaultEvkCaptureWidth"), 448)
        self.assertEqual(constant("DefaultEvkCaptureHeight"), 256)
        self.assertIn('TEXT("QaiHostCaptureWidth=")', source)
        self.assertIn('TEXT("QaiHostCaptureHeight=")', source)
        self.assertIn('TEXT("QaiEvkCaptureWidth=")', source)
        self.assertIn('TEXT("QaiEvkCaptureHeight=")', source)
        self.assertIn('HostServerUrl = TEXT("http://127.0.0.1:18084")', header)
        self.assertIn(
            'HostModel = TEXT("Cosmos-Reason2-2B-Parcel-Speed-v1")',
            header,
        )
        self.assertNotIn('HostServerUrl = TEXT("http://127.0.0.1:18080")', header)

    def test_runtime_contract_remains_one_lossless_png(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        self.assertIn("host_frames=1 host_transport=single_lossless_png", source)
        self.assertIn("evk_frames=1", source)
        self.assertIn("evk_transport=direct_lossless_png", source)
        self.assertIn("no EVK upload, file polling, or filesystem churn", source)
        dispatch = source[
            source.index("void AQaiConveyorWorld::DispatchPreparedInference(") :
            source.index("void AQaiConveyorWorld::HandleModelResponse(")
        ]
        self.assertNotIn("UploadEvkMediaAndDispatch(", dispatch)
        self.assertNotIn('StartsWith(TEXT("file://"))', dispatch)
        self.assertNotIn('SetHeader(TEXT("Connection"), TEXT("close"))', dispatch)
        self.assertNotIn("UploadEvkMediaAndDispatch", source)

    def test_live_prompt_is_lane_scoped_for_both_backends_without_a_new_switch(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")

        self.assertIn(
            'if (ParcelPromptProfile == TEXT("exact-v1"))',
            source,
        )
        self.assertNotIn(
            'ActiveBackend == TEXT("evk") && ParcelPromptProfile == TEXT("exact-v1")',
            source,
        )
        self.assertIn("central silver straight roller-conveyor lane", source)
        self.assertIn("a floor carton is fallen, never merely unstable", source)
        self.assertIn('TEXT("evk-single-lane-v1")', source)
        self.assertIn("SubmittedPromptProfile", header)
        self.assertNotIn("QaiEvkPrompt", source + header)

    def test_speed_v1_is_a_shared_one_token_contract(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn('ParcelPromptProfile == TEXT("speed-v1")', source)
        self.assertIn(
            "Classify cartons at the central silver conveyor lane.",
            source,
        )
        self.assertIn('if (ParcelPromptProfile == TEXT("speed-v1"))', source)
        self.assertIn("CompletionTokenBudget = 1;", source)
        self.assertIn('ClosedSetAnswer == TEXT("G")', source)
        self.assertIn('ClosedSetAnswer == TEXT("A")', source)
        self.assertIn('ClosedSetAnswer == TEXT("R")', source)

    def test_production_sensor_matches_fine_tuning_camera_and_cell(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")

        self.assertIn(
            'FString InferenceCameraVariant = TEXT("parcel-quarter-cell-occlusion-safe")',
            header,
        )
        self.assertIn('TEXT("parcel-quarter-cell-occlusion-safe")', source)
        self.assertIn("? 62.0f", source)
        self.assertNotIn('TEXT("QuarterCellWall")', source)
        self.assertIn("quarter_cell_divider status=removed", source)
        self.assertIn("ConveyorReturnLaneShiftCm = 260.0f", source)
        self.assertIn("ConveyorRightStraightX", source)
        self.assertIn("SortingPortal_Outfeed", source)
        self.assertIn("RuntimeConveyorTurnRollers", header)
        self.assertIn("RuntimeConveyorTurnQuarters", header)
        self.assertIn("RuntimeConveyorTurnBridges", header)
        self.assertIn("SM_A11_WestQuarter_Omniverse", source)
        self.assertIn("SM_A11_EastQuarter_Omniverse", source)
        self.assertIn("visual_source=omniverse_nvidia_a11+a08", source)
        self.assertIn("engine_basic_shapes=false", source)
        self.assertIn("legacy_usd_scale_ignored=true", source)
        self.assertIn(
            "a11_roller_layer_material=visual_finish_anisotropic_steel",
            source,
        )
        self.assertIn("a11_frame_material=omniverse_blue_opaque_two_sided", source)
        self.assertIn("M_A11TwoSidedConveyorFrame", source)
        self.assertIn("Quarter->SetMaterial(0, A11TwoSidedFrameMaterial)", source)
        self.assertIn("for (int32 MaterialIndex = 6;", source)
        self.assertIn(
            "Quarter->SetMaterial(\n                        MaterialIndex,\n                        A08PresentationRollerMaterial)",
            source,
        )
        import_script = OMNIVERSE_EXTENSIONS_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PRESENTATION_ROLLER_MATERIAL", import_script)
        self.assertIn("if index == 0:", import_script)
        self.assertIn("material = a11_two_sided_frame_material", import_script)
        self.assertIn("elif index >= 6:", import_script)
        self.assertIn("material = presentation_roller_material", import_script)
        visual_finish = VISUAL_FINISH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("M_A11TwoSidedConveyorFrame", visual_finish)
        self.assertIn('set_property(material, "two_sided", True)', visual_finish)
        self.assertIn('set_property(material, "blend_mode", unreal.BlendMode.BLEND_OPAQUE)', visual_finish)
        self.assertIn("M_BridgeRollerSteelUniform", visual_finish)
        self.assertIn('"used_with_instanced_static_meshes", True', visual_finish)
        self.assertIn('set_property(roughness, "r", 0.24)', visual_finish)
        self.assertIn("repair_low_profile_conveyor_materials", visual_finish)
        self.assertIn('set_property(material, "blend_mode", unreal.BlendMode.BLEND_MASKED)', visual_finish)
        layout_source = source[
            source.index("bool AQaiConveyorWorld::ConfigureConveyorLayout()") :
            source.index("void AQaiConveyorWorld::ConfigureForkliftMastMaterials()")
        ]
        self.assertIn("FVector::OneVector", layout_source)
        self.assertNotIn("/Engine/BasicShapes/Cylinder.Cylinder", layout_source)
        self.assertNotIn("/Engine/BasicShapes/Cube.Cube", layout_source)

    def test_default_forklift_pose_faces_the_sensor_observation_slice(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("ConveyorTuning::ConveyorLeftStraightX - 342.0f", source)
        self.assertIn("ConveyorTuning::BeltCenterY - 75.0f", source)
        self.assertIn("const FQuat DesiredStartRotation = FQuat::Identity", source)
        self.assertIn("facing=monitored_lane", source)
        self.assertNotIn("source=approved_overview", source)

    def test_left_portal_has_a_local_physical_centering_arm(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")

        self.assertIn("RuntimeConveyorCenteringArm", header)
        self.assertIn("RuntimeConveyorCenteringArmCurveExtension", header)
        self.assertIn("QaiOmniverseA08PortalCenteringArm", source)
        self.assertIn(
            "QaiOmniverseA08PortalCenteringArmCurveExtension",
            source,
        )
        self.assertIn(
            "centering_arm_visual_source=omniverse_nvidia_a08_visual_finish_steel",
            source,
        )
        self.assertIn("MI_AnisotropicRoller.MI_AnisotropicRoller", source)
        self.assertIn("A08PresentationRollerMaterial", source)
        self.assertNotIn("A08AuthoredRollerMaterial", source)
        self.assertIn("OmniverseA08RollerComponents.AddUnique", source)
        self.assertIn("RollerComponent->SetVisibility(false, true)", source)
        self.assertIn("QaiOmniverseA08NormalizedExtensionRollers", source)
        self.assertIn("M_BridgeRollerSteelUniform.M_BridgeRollerSteelUniform", source)
        self.assertIn("bridge_frame_material=omniverse_blue_opaque_two_sided", source)
        self.assertIn(
            "bridge_roller_material=uniform_opaque_pbr_steel_instancing_safe",
            source,
        )
        self.assertIn("bridge_roller_material_slots=%d", source)
        self.assertIn("RuntimeConveyorTurnRollers->GetNumMaterials()", source)
        self.assertIn("straight_a08_authored_rollers_hidden=%d", source)
        self.assertIn("centering_arm_placement=portal_backside", source)
        self.assertIn("centering_arm_sensor_visibility=hidden", source)
        self.assertIn('CenteringArmObstacle.Name = TEXT("portal centering arm")', source)
        self.assertIn("ConveyorCenteringArmStartYCm = -346.0f", source)
        self.assertIn("ConveyorCenteringArmEndYCm = -286.0f", source)
        self.assertIn("ConveyorCenteringArmCurveLeadCm = 75.0f", source)
        self.assertIn("ConveyorCenteringArmTailExtensionCm = 50.0f", source)
        self.assertIn(
            "ExistingGuideDirection * ConveyorCenteringArmTailExtensionCm",
            source,
        )
        self.assertIn("tail_extension_cm=%.1f", source)
        self.assertIn("EvaluateCenteringArmCurveExtensionStart", source)
        self.assertIn("curve_extension=true", source)
        self.assertIn("ConveyorCenteringAssistEndYCm = -210.0f", source)
        self.assertIn("ConveyorLeftStraightX - 42.0f", source)
        self.assertIn("ConveyorCenteringAssistMaximumAccelerationCm = 32.0f", source)
        self.assertIn("CenteringGuideEnvelope", source)
        self.assertIn("forward_preserving_slide", source)
        self.assertIn("anti_congestion=true", source)
        self.assertIn("Body.LinearVelocity -= GuideNormal * IntoGuideSpeed", source)
        self.assertIn("ConveyorCenteringArmBottomClearanceCm = 0.2f", source)
        self.assertIn("+ ConveyorCenteringArmBottomClearanceCm", source)
        self.assertIn("CenteringArmRollerTopZCm", source)
        self.assertIn("ConveyorTuning::ConveyorRollerTopFallbackCm", source)
        self.assertIn("RegisteredCenteringArmBounds", source)
        self.assertIn("CenteringArmBoundsCorrection", source)
        self.assertIn("measured_bottom_z_cm=%.2f", source)
        self.assertIn("collision_deck_overlap_cm=%.1f", source)
        self.assertIn("Capture->HiddenComponents.Add(RuntimeConveyorCenteringArm)", source)
        self.assertIn("RuntimeConveyorCenteringArmCurveExtension", source)
        self.assertIn("global_lateral_position_centering=none", source)
        self.assertIn("portal_guide_centering=localized", source)

    def test_straight_conveyor_uprights_are_removed_without_cutting_frame_flanges(self) -> None:
        visual_finish = VISUAL_FINISH_SCRIPT.read_text(encoding="utf-8")
        world = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("M_LowProfileConveyorFrame", visual_finish)
        self.assertIn("FrameClipHeightCm", visual_finish)
        self.assertIn("MaterialExpressionWorldPosition", visual_finish)
        self.assertIn("MP_OPACITY_MASK", visual_finish)
        self.assertIn('"blend_mode", unreal.BlendMode.BLEND_MASKED', visual_finish)
        self.assertIn("a05_roller_layers_uniform_steel", visual_finish)
        self.assertIn("M_HiddenStraightFrameSlot", visual_finish)
        self.assertIn("SM_StraightWest_FrameLowProfile", visual_finish)
        self.assertIn("SM_A08_FrameLowProfile", visual_finish)
        self.assertIn("a05_blue_frame_replaced_by_opaque_geometry_overlay", visual_finish)
        self.assertIn("a08_blue_frame_replaced_by_opaque_geometry_overlay", visual_finish)
        self.assertIn("upright_roots", visual_finish)
        self.assertIn("maximum > 8500.0", visual_finish)
        self.assertIn("for material_index in range(2, a05_slot_count)", visual_finish)
        self.assertIn("for material_index in range(1, a08_slot_count)", visual_finish)
        self.assertIn("create_low_profile_material_copy", visual_finish)
        self.assertIn('mesh.get_name() == "SM_ConveyorBelt_A08_02"', visual_finish)
        self.assertIn("component.set_material(material_index, clipped)", visual_finish)
        self.assertIn('"collision_changed": False', visual_finish)
        self.assertIn("OmniverseA08FrameComponents", world)
        self.assertIn("OmniverseA05StraightComponents", world)
        self.assertIn("M_LowProfileConveyorFrame.M_LowProfileConveyorFrame", world)
        self.assertIn("Bridge->SetMaterial(0, A11TwoSidedFrameMaterial)", world)
        self.assertIn("StraightComponent->SetMaterial(0, BridgeRollerMaterial)", world)
        self.assertIn("StraightComponent->SetMaterial(8, BridgeRollerMaterial)", world)
        self.assertIn("AddStraightFrameOverlay", world)
        self.assertIn("Source->SetMaterial(BlueSlot, HiddenStraightFrameMaterial)", world)
        self.assertIn("Overlay->SetMaterial(BlueSlot, A11TwoSidedFrameMaterial)", world)
        self.assertIn("Overlay->SetCollisionEnabled(ECollisionEnabled::NoCollision)", world)
        self.assertIn("source_collision_unchanged=true", world)
        header = WORLD_HEADER.read_text(encoding="utf-8")
        self.assertIn("RuntimeConveyorStraightRollers", header)
        self.assertIn("RuntimeConveyorStraightFrameOverlays", header)
        self.assertIn("QaiOmniverseA08NormalizedExtensionRollers", world)
        self.assertIn("for (UStaticMeshComponent* RollerComponent : OmniverseA08RollerComponents)", world)
        self.assertIn("const FBoxSphereBounds TargetBounds", world)
        self.assertIn("RollerComponent->SetVisibility(false, true)", world)
        self.assertIn("straight_a08_replacement_source=single_known_good_omniverse_a08", world)
        self.assertIn("straight_a08_replacement_rollers=%d", world)
        self.assertIn("CameraPreset.Equals(TEXT(\"conveyorseam\")", PAWN_SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("Bridge->SetMaterial(0, LowProfileFrameMaterial)", world)

    def test_worker_routes_bypass_pallets_and_join_shelf_to_conveyor(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("closed_pallet_bypass_shelf_sorting_belt", source)
        self.assertIn("worker1_cross_aisle_min_y_cm=-185", source)
        self.assertNotIn("FVector(-315.0f, -260.0f, 0.0f)", source)
        self.assertIn("upper_shelf_to_sensor_edge_shuttle", source)
        self.assertIn("FVector(-555.0f, 260.0f, 0.0f)", source)
        self.assertIn("FVector(190.0f, 190.0f, 0.0f)", source)
        self.assertIn("FVector(150.0f, 0.0f, 0.0f)", source)
        self.assertIn("worker2_sensor_dwell_seconds=4.0", source)
        self.assertIn("worker2_inference_role=visible_context_not_parcel_target", source)
        self.assertIn("Worker.Waypoints.Num() - 2", source)
        self.assertIn("GetWorkerLoosePalletClearanceCm", source)
        self.assertIn("minimum_loose_pallet_clearance_cm", source)
        self.assertIn("ConfigureCollision(Capsule, ECC_Pawn)", source)
        self.assertIn("worker_traversal=pawn_ignores_film_lip", source)
        self.assertIn(
            "MatCollider->SetCollisionResponseToChannel(ECC_Pawn, ECR_Ignore)",
            source,
        )

    def test_both_portals_have_structural_world_bounding_boxes(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("AddPortalBoundingBoxes", source)
        self.assertIn('TEXT("sorting portal frame")', source)
        self.assertIn("PortalBoundingBoxCount == 6", source)
        self.assertIn("shape=world_aabb", source)
        self.assertIn("collision_targets=parcels+props+workers+forklifts", source)

    def test_adapter_bookkeeping_is_not_presented_as_model_output(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        evk_service = EVK_SERVICE.read_text(encoding="utf-8")

        self.assertIn("HiddenAdapterProperties", source)
        self.assertIn('TEXT("edge_completion")', source)
        self.assertNotIn('"edge_completion":', evk_service)

    def test_single_lane_sensor_excludes_unmonitored_dynamic_cargo(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("HiddenDynamicCargoPrimitives", source)
        self.assertIn("for (const FDynamicBoxRuntime& Body : DynamicBoxes)", source)
        self.assertIn("if (Body.bPallet)", source)
        self.assertIn("dynamic_cargo_primitives_hidden=%d", source)
        self.assertIn("for (const FForkliftRuntime& Forklift : Forklifts)", source)
        self.assertIn("HideForkliftComponentTree(Forklift.LiftAssembly.Get())", source)
        self.assertIn("Forklift.LiftDrivenComponents", source)
        self.assertIn('Identity.Contains(TEXT("qai.forklift"))', source)
        self.assertIn('Identity.Contains(TEXT("/forklifts/forklift"))', source)
        self.assertIn("forklift_primitives_hidden=%d", source)
        self.assertIn("bSampleInsideProjection", source)
        self.assertIn("SampleCameraSpace.X > GNearClippingPlane", source)

    def test_runtime_generated_assets_are_included_in_packaged_clients(self) -> None:
        default_game = (
            PROJECT_ROOT / "unreal_conveyor_demo" / "Config" / "DefaultGame.ini"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '+DirectoriesToAlwaysCook=(Path="/Game/ConveyorRuntime/OmniverseExtensions")',
            default_game,
        )
        self.assertIn(
            '+DirectoriesToAlwaysCook=(Path="/Game/IQ9EVK/Runtime")',
            default_game,
        )

    def test_safety_mat_is_a_film_thin_wheel_surface(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("SafetyMatThicknessCm = 0.2f", source)
        self.assertIn("MatColliderExtent.Z = FMath::Max(MatColliderExtent.Z, 0.05f)", source)
        self.assertNotIn(
            "InferenceRedZoneBounds.GetExtent().ComponentMax(FVector(1.0f))",
            source,
        )

    def test_forklift_vertical_launches_are_arrested_as_a_body_pair(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")

        self.assertIn("ForkliftMaximumUpwardVelocityCm = 40.0f", source)
        self.assertIn("ForkliftSupportedVerticalDampingPerSecond = 9.0f", source)
        self.assertIn("ForkliftStartupLinearDamping = 48.0f", source)
        self.assertIn("StabilizedChassisVelocity.Z = FMath::Min", source)
        self.assertIn("CarriageVelocity.Z += VerticalVelocityCorrection", source)
        self.assertIn("forklift_vertical_stabilizer", source)

    def test_operator_signal_requires_two_matching_observations(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        hud = GAME_MODE_SOURCE.with_name("QaiConveyorHUD.cpp").read_text(
            encoding="utf-8"
        )

        self.assertIn("ModelSignalCandidateCount >= 2", source)
        self.assertIn("model_signal_confirmed", source)
        self.assertIn("reason2_confirmed", source)
        self.assertIn("GetSubmittedGroundTruthSignal()", hud)
        self.assertIn("GetModelSignal()", hud)

    def test_redundant_runtime_switches_are_removed(self) -> None:
        runtime_text = "\n".join(
            (
                WORLD_SOURCE.read_text(encoding="utf-8"),
                WORLD_HEADER.read_text(encoding="utf-8"),
                DATASET_SCRIPT.read_text(encoding="utf-8"),
            )
        )
        for removed_switch in (
            "QaiCaptureInterval",
            "QaiHudScreenshotTest",
            "QaiQuarterConveyorCell",
            "QaiParcelEvaluationPromptNoise",
            "QaiInferenceResolution",
        ):
            with self.subTest(removed_switch=removed_switch):
                self.assertNotIn(removed_switch, runtime_text)

    def test_rtx_render_path_remains_enabled(self) -> None:
        game_mode = GAME_MODE_SOURCE.read_text(encoding="utf-8")
        windows_config = WINDOWS_ENGINE_CONFIG.read_text(encoding="utf-8")
        dataset_script = DATASET_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('TEXT("QaiRayTracing=")', game_mode)
        self.assertIn("r.RayTracing=True", windows_config)
        self.assertIn("r.Lumen.HardwareRayTracing=True", windows_config)
        self.assertIn("hardware_lumen_hit_lighting", game_mode)
        self.assertIn("[string]$CaptureRHI = 'D3D12'", dataset_script)

    def test_six_runtime_resource_optimizations_are_enabled(self) -> None:
        world = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")
        game_mode = GAME_MODE_SOURCE.read_text(encoding="utf-8")

        # 1. Live capture pauses while a request is active and uses backend-aware
        # pacing, while the explicit/offline capture cadence remains available.
        self.assertIn("(bInferenceEnabled && bInferenceBusy)", world)
        self.assertIn('ActiveBackend == TEXT("evk") ? (1.0f / 0.75f) : (1.0f / 1.5f)', world)
        self.assertIn("bOfflineDatasetCapture || bCaptureFpsOverridden", world)

        # 2. Settled lamps do no per-frame render-state work and the eighteen
        # permanently dark runtime spotlights no longer exist.
        self.assertIn("if (!bForceUpdate && !bNeedsFade)", world)
        self.assertIn("legacy_zero_spots_removed=18", world)
        self.assertNotIn("SpawnActor<ASpotLight>", world)

        # 3/4. Tiny RT geometry is excluded and repeated authored meshes are
        # consolidated with clustered frustum/distance culling.
        self.assertIn("ConfigureStaticRenderOptimizations", world)
        self.assertIn("UHierarchicalInstancedStaticMeshComponent", world)
        self.assertIn("SetCullDistances(3000, 4500)", world)
        self.assertIn("SetVisibleInRayTracing(false)", world)
        self.assertIn("RuntimeStaticMeshInstances", header)

        # 5. Navigation/safety work is decoupled from the 120 Hz solver, and
        # the route motor integrates its impulse over the decision interval.
        self.assertIn("WorkerDecisionStep = 1.0f / 30.0f", world)
        self.assertIn("SafetyEvaluationStep = 1.0f / 15.0f", world)
        self.assertIn("RequestedAcceleration\n                * ConveyorTuning::WorkerMassKg", world)
        self.assertIn("OnlyTickPoseWhenRendered", world)

        # 6. Ultra remains untouched; non-Ultra tiers use coarser fog grids,
        # HZB local-fog culling and lower local shadow budgets.
        self.assertIn("bUltra ? 4 : (bHighOrBetter ? 8 : 12)", game_mode)
        self.assertIn("bUltra ? 128 : (bHighOrBetter ? 72 : 48)", game_mode)
        self.assertIn('TEXT("r.LocalFogVolume.UseHZB"), bUltra ? 0 : 1', game_mode)
        self.assertIn("bUltra ? 4096 : (bHighOrBetter ? 2048 : 1024)", game_mode)

    def test_unused_second_forklift_is_removed_from_client(self) -> None:
        source_directory = GAME_MODE_SOURCE.parent
        world = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")
        pawn = (source_directory / "QaiConveyorPawn.cpp").read_text(encoding="utf-8")
        pawn_header = (source_directory / "QaiConveyorPawn.h").read_text(encoding="utf-8")
        hud = (source_directory / "QaiConveyorHUD.cpp").read_text(encoding="utf-8")
        input_config = (
            PROJECT_ROOT / "unreal_conveyor_demo" / "Config" / "DefaultInput.ini"
        ).read_text(encoding="utf-8")
        lean_stage = (
            PROJECT_ROOT
            / "unreal_conveyor_demo"
            / "Scripts"
            / "build_lean_conveyor_stage.py"
        ).read_text(encoding="utf-8")

        self.assertIn("FForkliftRuntime Forklifts[1]", header)
        self.assertNotIn("FForkliftRuntime Forklifts[2]", header)
        self.assertIn("RemoveUnusedAuthoredForklift", world)
        self.assertIn("forklift_removed forklift=2", world)
        self.assertIn("BoundWheels == 4", world)
        self.assertIn("forklifts_authored=1", world)
        self.assertNotIn("CycleForklift", pawn)
        self.assertNotIn("SelectForklift2", pawn_header)
        self.assertNotIn('ActionName="Forklift2"', input_config)
        self.assertNotIn("previous/next forklift", hud)
        self.assertIn(
            '"/World/CodexPoC/ConveyorSafety/Forklifts/Forklift2"',
            lean_stage,
        )

    def test_runtime_lumen_and_ray_tracing_toggles_are_explicit(self) -> None:
        source_directory = GAME_MODE_SOURCE.parent
        game_mode_header = GAME_MODE_SOURCE.with_suffix(".h").read_text(encoding="utf-8")
        game_mode = GAME_MODE_SOURCE.read_text(encoding="utf-8")
        world = WORLD_SOURCE.read_text(encoding="utf-8")
        world_header = WORLD_HEADER.read_text(encoding="utf-8")
        pawn = (source_directory / "QaiConveyorPawn.cpp").read_text(encoding="utf-8")
        hud = (source_directory / "QaiConveyorHUD.cpp").read_text(encoding="utf-8")
        input_config = (
            PROJECT_ROOT / "unreal_conveyor_demo" / "Config" / "DefaultInput.ini"
        ).read_text(encoding="utf-8")
        windows_config = WINDOWS_ENGINE_CONFIG.read_text(encoding="utf-8")

        self.assertIn("ToggleRuntimeLumen", game_mode_header)
        self.assertIn("ToggleRuntimeRayTracing", game_mode_header)
        self.assertIn('TEXT("r.RayTracing.Enable")', game_mode)
        self.assertIn('TEXT("r.Lumen.DiffuseIndirect.Allow")', game_mode)
        self.assertIn("ApplyRuntimeRenderFeatureStateToCapture", world)
        self.assertIn('BindKey(EKeys::F6', pawn)
        self.assertIn('BindKey(EKeys::F7', pawn)
        self.assertIn("EKeys::F9", pawn)
        self.assertNotIn('ActionName="ToggleLumen"', input_config)
        self.assertNotIn('ActionName="ToggleRayTracing"', input_config)
        self.assertIn('TEXT("F6: Lumen %s', hud)
        self.assertIn("F9: sensor view %s", hud)
        self.assertNotIn("DrawSensorViewOverlay(Canvas)", hud)
        self.assertIn("UpdateSensorViewSurfaceProjection", world)
        self.assertIn("QaiSensorViewSurfaceProjection", world)
        self.assertIn("LineTraceMultiByChannel", world)
        self.assertIn("constexpr uint8 DepthPriority = 0", world)
        self.assertIn("style=surface_red_laser_corner_brackets_with_reticle", world)
        self.assertIn("LaserHaloColor", world)
        self.assertIn("LaserBeamColor", world)
        self.assertIn("LaserCoreColor", world)
        self.assertIn("CornerBracketFraction = 0.18f", world)
        self.assertIn("const FSensorSurfaceHit CenterHit", world)
        self.assertNotIn("DashDutyCycle", world)
        self.assertIn("Capture->HiddenComponents.Add(SensorViewProjectionLines)", world)
        self.assertIn("inference_capture=excluded", world)
        self.assertIn("bDrawSensorViewOverlay = true", world_header)
        self.assertIn("r.RayTracing.EnableOnDemand=True", windows_config)

    def test_shipping_workaround_is_limited_to_the_cook(self) -> None:
        package_script = PACKAGE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"-run=Cook"', package_script)
        self.assertIn('"-nothreading"', package_script)
        self.assertIn('"-clientconfig=Shipping"', package_script)
        self.assertIn('"-QaiRuntimeMotionTest"', package_script)
        self.assertIn('"Windows\\QaiConveyor.exe"', package_script)
        self.assertIn("$ClientCandidates", package_script)
        self.assertIn('"-NoSaveConfig"', package_script)
        self.assertIn("[System.IO.File]::ReadAllBytes", package_script)
        self.assertIn("[System.IO.File]::WriteAllBytes", package_script)
        self.assertNotIn("-noraytracing", package_script.lower())
        self.assertNotIn("-d3d11", package_script.lower())

    def test_developer_launcher_starts_the_exact_gpu_release_before_the_client(self) -> None:
        launcher = RUN_WINDOWS_DEMO_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"Cosmos-Reason2-2B-Parcel-Speed-v1"', launcher)
        self.assertIn('"http://127.0.0.1:18084"', launcher)
        self.assertIn('"--image-min-tokens", "112"', launcher)
        self.assertIn('"--image-max-tokens", "112"', launcher)
        self.assertIn('"--flash-attn", "on"', launcher)
        self.assertIn("Get-FileHash", launcher)
        self.assertIn("Test-HostReady", launcher)
        self.assertLess(launcher.index("if (-not (Test-HostReady))"), launcher.index("$Client = Start-Process"))

    def test_interactive_window_has_large_default_and_explicit_fullscreen_shortcuts(self) -> None:
        game_mode = GAME_MODE_SOURCE.read_text(encoding="utf-8")
        pawn = GAME_MODE_SOURCE.with_name("QaiConveyorPawn.cpp").read_text(
            encoding="utf-8"
        )
        hud = GAME_MODE_SOURCE.with_name("QaiConveyorHUD.cpp").read_text(
            encoding="utf-8"
        )
        input_config = DEFAULT_INPUT_CONFIG.read_text(encoding="utf-8")
        settings = DEFAULT_GAME_USER_SETTINGS.read_text(encoding="utf-8")

        self.assertIn("PreferredWindowWidth = 1600", game_mode)
        self.assertIn("PreferredWindowHeight = 900", game_mode)
        self.assertIn("CurrentResolution.X < PreferredWindowWidth", game_mode)
        self.assertIn("ApplyInteractiveWindowDefaults();", game_mode)
        self.assertIn("ResolutionSizeX=1600", settings)
        self.assertIn("ResolutionSizeY=900", settings)
        self.assertIn("FullscreenMode=2", settings)
        self.assertIn("EKeys::F11", pawn)
        self.assertIn("FInputChord(EKeys::Enter, false, false, true, false)", pawn)
        self.assertIn("EWindowMode::WindowedFullscreen", pawn)
        self.assertIn("F11 or Alt+Enter: fullscreen", hud)
        self.assertIn("bAltEnterTogglesFullscreen=False", input_config)
        self.assertIn("bF11TogglesFullscreen=False", input_config)

    def test_evk_intro_ignores_desktop_window_switch_keys(self) -> None:
        pawn = GAME_MODE_SOURCE.with_name("QaiConveyorPawn.cpp").read_text(
            encoding="utf-8"
        )
        header = GAME_MODE_SOURCE.with_name("QaiConveyorPawn.h").read_text(
            encoding="utf-8"
        )

        self.assertIn("void AnyInputPressed(FKey PressedKey);", header)
        self.assertIn("IsDesktopWindowSwitchKey", pawn)
        self.assertIn("Key == EKeys::LeftAlt", pawn)
        self.assertIn("Key == EKeys::RightAlt", pawn)
        self.assertIn("Key == EKeys::Tab", pawn)
        self.assertIn("bIntroCameraActive", pawn)
        self.assertIn("dismiss=any_button_except_alt_tab", pawn)
        self.assertIn("reason=desktop_window_switch", pawn)

    def test_parcel_evaluation_bilateral_contract(self) -> None:
        source = WORLD_SOURCE.read_text(encoding="utf-8")
        header = WORLD_HEADER.read_text(encoding="utf-8")
        generator = DATASET_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("QaiParcelEvaluationBilateral", source)
        self.assertIn("bParcelEvaluationBilateral", header)
        self.assertIn('TEXT("camera-far")', source)
        self.assertIn('TEXT("camera-near")', source)
        self.assertIn('\\"support_side\\"', source)
        self.assertIn('TEXT("horizontal-overhang")', source)
        self.assertIn('TEXT("mild-tilt")', source)
        self.assertIn('TEXT("active-tipping")', source)
        self.assertIn("AmberPoseIndex < 8", source)
        self.assertIn('\\"amber_pose_bucket\\"', source)
        self.assertIn('TEXT("moderate-skew")', source)
        self.assertIn('TEXT("strong-skew")', source)
        self.assertIn("YawPoseIndex < 5", source)
        self.assertIn("YawPoseIndex < 14", source)
        self.assertIn("ProjectedLateralHalfExtent", source)
        self.assertIn('\\"yaw_pose_bucket\\"', source)
        self.assertIn("CaseRandom.FRandRange(145.0f, 245.0f)", source)
        self.assertIn("RuntimeChaosConveyorParcelIndices", header)
        self.assertIn("RuntimeChaosConveyorVisualRoots", header)
        self.assertIn("conveyor_parcel_visual_attach_failed", source)
        self.assertIn("visual_proxy_delta_cm", source)
        self.assertIn("MaximumVisualProxyDeltaCm", source)
        self.assertIn("previous case's target contents", source)
        self.assertIn("FlushRenderingCommands();", source)
        self.assertIn("[switch]$Bilateral", generator)


if __name__ == "__main__":
    unittest.main()
