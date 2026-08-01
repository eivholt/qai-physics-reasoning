from __future__ import annotations

import json
import socketserver
import threading
import unittest

from integrations.isaac_sim_mcp.edge_supervisor import supervisor_decision
from integrations.isaac_sim_mcp.server import (
    IsaacMcpServer,
    IsaacTcpClient,
)


class _FakeIsaacHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        chunks: list[bytes] = []
        while True:
            chunk = self.request.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        source = b"".join(chunks).decode("utf-8")
        response = {
            "status": "ok",
            "output": "",
            "result": {"source": source},
        }
        self.request.sendall(json.dumps(response).encode("utf-8"))


class _FakeIsaacServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class IsaacSimMcpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tcp_server = _FakeIsaacServer(("127.0.0.1", 0), _FakeIsaacHandler)
        self.thread = threading.Thread(
            target=self.tcp_server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        host, port = self.tcp_server.server_address
        self.client = IsaacTcpClient(host=host, port=port, timeout_seconds=2.0)
        self.mcp = IsaacMcpServer(self.client)

    def tearDown(self) -> None:
        self.tcp_server.shutdown()
        self.tcp_server.server_close()
        self.thread.join(timeout=2.0)

    def test_tcp_client_half_closes_and_decodes_json(self) -> None:
        response = self.client.execute("1 + 1")
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["result"]["source"], "1 + 1")

    def test_initialize_and_tool_listing(self) -> None:
        initialized = self.mcp.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-03-26"},
            }
        )
        self.assertEqual(
            initialized["result"]["protocolVersion"],
            "2025-03-26",
        )

        listed = self.mcp.handle_request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
        names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertIn("isaac_ping", names)
        self.assertIn("isaac_set_frame_rate_limit", names)
        self.assertIn("isaac_set_markings_enabled", names)
        self.assertIn("isaac_set_presentation_overlays_enabled", names)
        self.assertIn("isaac_save_stage_checkpoint", names)
        self.assertIn("isaac_load_stage_checkpoint", names)
        self.assertIn("isaac_set_lightweight_warehouse_wall_height", names)
        self.assertIn("isaac_enable_edge_ai_supervisor_ui", names)
        self.assertIn("isaac_create_live_aisle_navigation", names)
        self.assertIn("isaac_apply_live_aisle_advisory", names)
        self.assertIn("isaac_capture_live_aisle_roof_frame", names)
        self.assertIn("isaac_capture_live_aisle_tactical_frame", names)
        self.assertIn("isaac_capture_live_aisle_camera_grid", names)
        self.assertIn("isaac_configure_live_blind_corner", names)
        self.assertIn("isaac_stock_live_blind_corner_shelves", names)
        self.assertIn("isaac_configure_live_robot_pov", names)
        self.assertIn("isaac_configure_live_navigation_graph", names)
        self.assertIn("isaac_set_live_forklift_visible", names)
        self.assertIn("isaac_configure_live_route_visualizations", names)
        self.assertIn("isaac_capture_live_robot_front_frame", names)
        self.assertIn("isaac_create_poc_scene", names)
        self.assertIn("isaac_set_marker_pose", names)
        self.assertIn("isaac_capture_viewport", names)
        self.assertIn("isaac_create_edge_supervisor_scene", names)
        self.assertIn("isaac_set_edge_supervisor_progress", names)
        self.assertIn("isaac_capture_edge_supervisor_sequence", names)
        self.assertIn("isaac_create_realistic_edge_supervisor_scene", names)
        self.assertIn("isaac_create_lightweight_live_warehouse", names)
        self.assertIn("isaac_capture_realistic_edge_supervisor_sequence", names)

    def test_marker_arguments_are_encoded_as_numeric_source(self) -> None:
        result = self.mcp.call_tool(
            "isaac_set_marker_pose",
            {"x": 1, "y": -2.5, "z": 0.5},
        )
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("Gf.Vec3d(1.0, -2.5, 0.5)", source)

    def test_frame_rate_limit_is_bounded_and_uses_precision_sleep(self) -> None:
        result = self.mcp.call_tool(
            "isaac_set_frame_rate_limit",
            {"fps": 60},
        )
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("fps = 60", source)
        self.assertIn('loop_names = ("main", "present", "rendering_0", "rendering_1")', source)
        self.assertIn('settings.set_bool(f"{prefix}/rateLimitEnabled", True)', source)
        self.assertIn('settings.set_bool(f"{prefix}/rateLimitUsePrecisionSleep", True)', source)
        self.assertIn(
            'settings.set_bool("/app/runLoops/main/rateLimitUseBusyLoop", False)',
            source,
        )

        invalid = self.mcp.call_tool(
            "isaac_set_frame_rate_limit",
            {"fps": 0},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("between 1 and 240", invalid["content"][0]["text"])

    def test_markings_toggle_is_limited_to_owned_group(self) -> None:
        result = self.mcp.call_tool(
            "isaac_set_markings_enabled",
            {"enabled": False},
        )
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn('path = "/World/CodexPoC/Markings"', source)
        self.assertIn("prim.SetActive(False)", source)
        self.assertNotIn("/Supervisor", source)

        invalid = self.mcp.call_tool(
            "isaac_set_markings_enabled",
            {"enabled": "false"},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("must be a boolean", invalid["content"][0]["text"])

    def test_presentation_overlay_toggle_preserves_physical_assets(self) -> None:
        result = self.mcp.call_tool(
            "isaac_set_presentation_overlays_enabled",
            {"enabled": False},
        )
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn('f"{root}/Markings"', source)
        self.assertIn('f"{root}/Supervisor"', source)
        self.assertIn('f"{root}/Actors/RobotBlue/IdentityPad"', source)
        self.assertIn('f"{root}/Actors/RobotBlue/Beacon"', source)
        self.assertIn('f"{root}/Actors/WorkerYellow/IdentityPad"', source)
        self.assertIn("prim.SetActive(False)", source)
        self.assertIn("clear_selected_prim_paths()", source)
        self.assertNotIn("/Asset", source)

        invalid = self.mcp.call_tool(
            "isaac_set_presentation_overlays_enabled",
            {"enabled": "false"},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("must be a boolean", invalid["content"][0]["text"])

    def test_wall_height_keeps_wall_bottoms_on_the_floor(self) -> None:
        result = self.mcp.call_tool(
            "isaac_set_lightweight_warehouse_wall_height",
            {"height_m": 3.9},
        )
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("/World/CodexPoC/Environment/Walls/Y_m4_75", source)
        self.assertIn("/World/CodexPoC/Environment/Walls/X_7_32", source)
        self.assertIn("floor_z = float(translate[2])", source)
        self.assertIn("floor_z + height_m * 0.5", source)
        self.assertIn("height_m = 3.9", source)

        invalid = self.mcp.call_tool(
            "isaac_set_lightweight_warehouse_wall_height",
            {"height_m": 0.1},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("between 0.5 and 12.0", invalid["content"][0]["text"])

    def test_edge_ai_supervisor_ui_is_loaded_from_bounded_repo_path(self) -> None:
        result = self.mcp.call_tool(
            "isaac_enable_edge_ai_supervisor_ui",
            {},
        )
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("qai.edge_ai_supervisor", source)
        self.assertIn("manager.add_path(extension_root)", source)
        self.assertIn("Edge AI Warehouse Supervisor", source)

    def test_live_aisle_uses_packaged_navigation_and_bounded_overrides(self) -> None:
        created = self.mcp.call_tool(
            "isaac_create_live_aisle_navigation",
            {
                "robot_width": 1.42,
                "robot_length": 2.1,
                "robot_scale": 1.75,
                "start_playing": True,
            },
        )
        self.assertNotIn("isError", created)
        payload = json.loads(created["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("nova_carter_nav_only.usd", source)
        self.assertIn("/app/omni.graph.scriptnode/opt_in", source)
        self.assertIn("LIVE_ROOT = '/World/CodexPoC/LiveAisle'", source)
        self.assertNotIn("/CodexPalletDeck", source)
        self.assertIn('f"{chassis_path}/visual"', source)
        self.assertNotIn("/wheel_left/visual", source)
        self.assertIn('"caster_wheel_right"', source)
        self.assertIn(
            'robot_prim.GetAttribute("xformOp:translate")',
            source,
        )
        self.assertIn("translate_attr.Set(Gf.Vec3d(*start))", source)
        self.assertIn('"robot_chassis_start_world"', source)
        self.assertNotIn("base_translate[0] + start[0]", source)
        self.assertIn("Gf.Vec3f(", source)
        self.assertIn("1.75", source)
        self.assertIn("1.1875", source)
        self.assertNotIn('chassis.GetAttribute("xformOp:scale")', source)
        self.assertIn("codex:robotClearanceWidthMeters", source)
        self.assertIn("codex:robotClearanceLengthMeters", source)
        self.assertIn("attr.Set(Gf.Vec3d(*world_xyz))", source)
        self.assertIn('"RobotBlue",\n    (5.94, 1.6, 0.0),\n    0.0,', source)
        self.assertIn('planner.GetAttribute("inputs:targetOrientation")', source)
        self.assertIn(
            '"blue_original_route_waypoints": '
            "((5.94, 1.6, 0.0), (5.94, 6.2, 0.0))",
            source,
        )
        self.assertIn("actor.SetActive(False)", source)
        self.assertIn("Usd.PrimRange(forklift)", source)
        self.assertIn("enabled_attr.Set(False)", source)
        self.assertIn('"forklift_start_mode": \'current\'', source)
        self.assertIn("codex:blindCornerStartWorld", source)
        self.assertIn('forklift.GetAttribute("xformOp:rotateXYZ")', source)
        self.assertIn("Gf.Vec3f(0.0, 0.0, 90.0)", source)
        self.assertIn('"forklift_start_yaw_degrees": 90.0', source)
        self.assertIn('"forklift_end_yaw_degrees": 0.0', source)
        self.assertIn(
            '"tactical_camera_policy": '
            "'user_authored_two_passage_full_view'",
            source,
        )
        self.assertIn('"tactical_camera_focal_length": 24.0', source)
        self.assertIn(
            "/Actors/ForkliftOrange/IdentityPad",
            source,
        )
        self.assertIn(
            "/Actors/ForkliftOrange/CodexAmberBeacon",
            source,
        )
        self.assertIn("beacon_path = None", source)
        self.assertIn('("Cardbox_0", "Cardbox_2")', source)

        tactical_capture = self.mcp.call_tool(
            "isaac_capture_live_aisle_tactical_frame",
            {"filename": "offscreen-test.png"},
        )
        self.assertNotIn("isError", tactical_capture)
        capture_source = json.loads(
            tactical_capture["content"][0]["text"]
        )["result"]["source"]
        self.assertIn("rep.create.render_product(", capture_source)
        self.assertIn(
            'rep.AnnotatorRegistry.get_annotator("LdrColor")',
            capture_source,
        )
        self.assertIn(
            '"capture_backend": "offscreen_render_product"',
            capture_source,
        )
        self.assertNotIn("get_active_viewport", capture_source)

        kinematic = self.mcp.call_tool(
            "isaac_create_live_aisle_navigation",
            {
                "motion_controller": "kinematic_waypoint",
                "start_playing": False,
            },
        )
        self.assertNotIn("isError", kinematic)
        kinematic_source = json.loads(
            kinematic["content"][0]["text"]
        )["result"]["source"]
        self.assertIn(
            "'kinematic_waypoint' == \"kinematic_waypoint\"",
            kinematic_source,
        )
        self.assertIn("codex:kinematicSpeedMps", kinematic_source)
        tick = self.mcp.call_tool("isaac_tick_live_aisle", {})
        self.assertNotIn("isError", tick)
        tick_source = json.loads(tick["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn("motion_delta_seconds", tick_source)
        self.assertIn("max(1.0, delta_seconds)", tick_source)
        self.assertIn(
            'blue_root.GetAttribute("xformOp:orient")',
            tick_source,
        )
        self.assertIn(
            '"robot_yaw_degrees": robot_yaw_degrees',
            tick_source,
        )
        self.assertIn(
            '"destination_reached": destination_reached',
            tick_source,
        )
        self.assertIn(
            'differential.GetAttribute("inputs:linearVelocity")',
            tick_source,
        )
        self.assertIn(
            '"navigation_angular_radps": navigation_angular_radps',
            tick_source,
        )
        self.assertNotIn(
            "UsdGeom.XformCommonAPI(forklift).SetTranslate("
            "Gf.Vec3d(5.6, 1.6, 0.0))",
            source,
        )
        self.assertNotIn("stage.RemovePrim(ROOT)", source)

        blind_corner = self.mcp.call_tool(
            "isaac_configure_live_blind_corner",
            {},
        )
        self.assertNotIn("isError", blind_corner)
        source = json.loads(blind_corner["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn(
            "scenario\": \"two_passage_approach_turn_block",
            source,
        )
        self.assertIn("Gf.Vec3d(2.2, 4.835, 0.0)", source)
        self.assertIn("Gf.Vec3d(5.2, 4.835, 0.0)", source)
        self.assertIn('forklift.GetAttribute("xformOp:rotateXYZ")', source)
        self.assertIn("Gf.Vec3f(0.0, 0.0, 90.0)", source)
        self.assertIn('"forklift_start_yaw_degrees": 90.0', source)
        self.assertIn('"forklift_end_yaw_degrees": 0.0', source)
        self.assertIn("Gf.Vec3d(5.94, 3.6, 0.0)", source)
        self.assertIn("pallets_untouched", source)

        stocked = self.mcp.call_tool(
            "isaac_stock_live_blind_corner_shelves",
            {"columns": 4, "levels": 3},
        )
        self.assertNotIn("isError", stocked)
        source = json.loads(stocked["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn("SM_CardBoxD_04.usd", source)
        self.assertIn("/BlindCornerShelfStock", source)
        self.assertIn("Row2 Segment1", source)
        self.assertIn("Gf.Vec3f(2.05, 2.05, 2.05)", source)
        self.assertIn("(1.335, 2.835, 4.135)", source)

        pov = self.mcp.call_tool(
            "isaac_configure_live_robot_pov",
            {"forward_m": 1.15, "height_m": 0.72},
        )
        self.assertNotIn("isError", pov)
        source = json.loads(pov["content"][0]["text"])["result"]["source"]
        self.assertIn("/sensors/CodexFrontPOV", source)
        self.assertIn("/sensors/front_owl/camera", source)
        self.assertIn("Gf.Vec3d(1.15, 0.0, 0.72)", source)

        navigation = self.mcp.call_tool(
            "isaac_configure_live_navigation_graph",
            {},
        )
        self.assertNotIn("isError", navigation)
        source = json.loads(navigation["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn("QuinticPathPlanner", source)
        self.assertIn("StanleyControlPID", source)
        self.assertIn("DifferentialController", source)
        self.assertIn("IsaacArticulationController", source)
        self.assertIn("scripted_robot_motion", source)
        self.assertIn('"physics_wheel_actuation": True', source)
        self.assertIn("live_waypoint_bearing", source)
        self.assertNotIn("packaged_planner.GetAttribute", source)

        routes = self.mcp.call_tool(
            "isaac_configure_live_route_visualizations",
            {},
        )
        self.assertNotIn("isError", routes)
        source = json.loads(routes["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn("cyan upper route", source)
        self.assertIn("orange lower route", source)
        self.assertIn("warehouse_markings_unchanged", source)
        self.assertNotIn("UsdGeom.Tokens.bezier", source)

        reroute = self.mcp.call_tool(
            "isaac_apply_live_aisle_advisory",
            {
                "action": "REROUTE",
                "route": "south_bypass",
                "request_id": "request-1",
                "decision_source": "evk",
                "latency_seconds": 8.6,
            },
        )
        self.assertNotIn("isError", reroute)
        payload = json.loads(reroute["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("route = 'south_bypass'", source)
        self.assertIn("set_target_world", source)
        self.assertIn("codex:lastLatencySeconds", source)
        self.assertIn("codex:lastDecisionSource", source)

        north_reroute = self.mcp.call_tool(
            "isaac_apply_live_aisle_advisory",
            {
                "action": "REROUTE",
                "route": "north_bypass",
                "request_id": "request-north",
                "decision_source": "evk",
                "latency_seconds": 3.2,
            },
        )
        self.assertNotIn("isError", north_reroute)
        source = json.loads(north_reroute["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn("route = 'north_bypass'", source)
        self.assertIn("[0.0, 5.8, 0.0]", source)
        self.assertIn("[5.94, 6.2, 0.0]", source)

        continue_current = self.mcp.call_tool(
            "isaac_apply_live_aisle_advisory",
            {
                "action": "CONTINUE",
                "route": "current",
                "decision_source": "evk",
            },
        )
        self.assertNotIn("isError", continue_current)
        source = json.loads(continue_current["content"][0]["text"])["result"][
            "source"
        ]
        self.assertIn('route == "current"', source)
        self.assertIn("codex:blueRoute", source)
        self.assertIn("set_target_world(", source)
        self.assertIn('"inputs:drawPath"', source)
        self.assertIn("planner_path_visible", source)
        self.assertIn("clear_lines()", source)

        invalid = self.mcp.call_tool(
            "isaac_apply_live_aisle_advisory",
            {"action": "CONTINUE", "route": "south_bypass"},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("route is not valid", invalid["content"][0]["text"])

    def test_checkpoint_export_is_bounded_to_repository_artifacts(self) -> None:
        saved = self.mcp.call_tool(
            "isaac_save_stage_checkpoint",
            {"filename": "warehouse_before_navigation.usda"},
        )
        self.assertNotIn("isError", saved)
        payload = json.loads(saved["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("isaac_sim_checkpoints", source)
        self.assertIn("warehouse_before_navigation.usda", source)
        self.assertIn("GetRootLayer().Export", source)

        invalid = self.mcp.call_tool(
            "isaac_save_stage_checkpoint",
            {"filename": "..\\outside.usda"},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("simple .usd", invalid["content"][0]["text"])

    def test_invalid_usd_path_is_rejected_before_network_call(self) -> None:
        result = self.mcp.call_tool(
            "isaac_list_prims",
            {"root_path": "World", "max_depth": 1, "limit": 10},
        )
        self.assertTrue(result["isError"])
        self.assertIn("absolute USD path", result["content"][0]["text"])

    def test_capture_filename_cannot_escape_artifact_directory(self) -> None:
        result = self.mcp.call_tool(
            "isaac_capture_viewport",
            {"filename": "..\\outside.png"},
        )
        self.assertTrue(result["isError"])
        self.assertIn("simple .png basename", result["content"][0]["text"])

    def test_edge_supervisor_scene_stays_in_owned_namespace(self) -> None:
        result = self.mcp.call_tool("isaac_create_edge_supervisor_scene", {})
        self.assertNotIn("isError", result)
        payload = json.loads(result["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn('ROOT = "/World/CodexPoC"', source)
        self.assertIn('stage.RemovePrim(ROOT)', source)
        self.assertIn('/Sensors/RoofOverview', source)

    def test_realistic_scene_uses_nvidia_warehouse_and_separate_captures(self) -> None:
        created = self.mcp.call_tool(
            "isaac_create_realistic_edge_supervisor_scene",
            {},
        )
        self.assertNotIn("isError", created)
        payload = json.loads(created["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn('/Environment/WarehousePlacement/NvidiaWarehouse', source)
        self.assertIn("full_warehouse.usd", source)
        self.assertIn("nvidia_full_warehouse_custom_aisle_layout", source)
        self.assertIn("SM_RackShelf_01.usd", source)
        self.assertIn("SM_RackFrame_03.usd", source)
        self.assertIn("row_centers = (-3.2, 0.0, 3.2)", source)
        self.assertIn("segment_centers = (-3.0, 3.0)", source)
        self.assertIn("/AisleLayout/Row", source)
        self.assertIn("storage_prefixes = (", source)
        self.assertIn('"box_",', source)
        self.assertIn('or "beam" in child_name', source)
        self.assertIn('or "pillar" in child_name', source)
        self.assertIn("/Actors/RobotBlue/IdentityPad", source)
        self.assertIn("/Actors/RobotGreen/IdentityPad", source)
        self.assertIn("/Actors/ForkliftOrange/IdentityPad", source)
        self.assertIn("/Actors/WorkerYellow/IdentityPad", source)
        self.assertIn("scale=(1.45, 1.45, 1.45)", source)
        self.assertIn("(0.0, 0.0, 4.40)", source)
        self.assertIn("(1.15, 0.045, 0.025)", source)
        self.assertIn("direct_y = 1.6", source)
        self.assertIn('("South", -1.6)', source)
        self.assertIn('("NorthPerimeter", 4.8)', source)
        self.assertIn("/Markings/HoldLine", source)
        self.assertNotIn("/Markings/Crosswalk", source)
        self.assertIn("markings_group.SetActive(False)", source)
        self.assertIn("for overlay_path in (", source)
        self.assertIn('f"{ROOT}/Supervisor"', source)
        self.assertIn('f"{ROOT}/Actors/RobotBlue/IdentityPad"', source)
        self.assertIn('f"{ROOT}/Actors/ForkliftOrange/Beacon"', source)
        self.assertIn("overlay_prim.SetActive(False)", source)
        self.assertIn("/Supervisor/Routes/SouthEntry", source)
        self.assertIn("/Supervisor/Routes/NorthHorizontal", source)
        self.assertIn('render_settings.set("/rtx/raytracing/showLights", 2)', source)
        self.assertIn(
            'render_settings.set("/rtx-transient/post/aa/limitedOps", False)',
            source,
        )
        self.assertIn('render_settings.set("/rtx/post/aa/op", 4)', source)
        self.assertNotIn("intensity.Set(0.0)", source)
        self.assertIn("dome.CreateIntensityAttr(150.0)", source)
        self.assertIn("distant.CreateIntensityAttr(250.0)", source)
        self.assertIn(
            '"RoofOverview": ((0.0, 0.8, 21.0), 15.0)',
            source,
        )

        captured = self.mcp.call_tool(
            "isaac_capture_realistic_edge_supervisor_sequence",
            {"scenario": "aisle_congestion", "frame_count": 8},
        )
        self.assertNotIn("isError", captured)
        payload = json.loads(captured["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("isaac_sim_edge_supervisor_realistic", source)

        captured_v2 = self.mcp.call_tool(
            "isaac_capture_realistic_edge_supervisor_sequence",
            {
                "scenario": "aisle_congestion",
                "frame_count": 8,
                "variant": "v2",
            },
        )
        self.assertNotIn("isError", captured_v2)
        payload = json.loads(captured_v2["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("isaac_sim_edge_supervisor_realistic_v2", source)
        self.assertIn("for _ in range(8):", source)

        captured_v3 = self.mcp.call_tool(
            "isaac_capture_realistic_edge_supervisor_sequence",
            {
                "scenario": "aisle_congestion",
                "frame_count": 8,
                "variant": "v3",
            },
        )
        self.assertNotIn("isError", captured_v3)
        payload = json.loads(captured_v3["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("isaac_sim_edge_supervisor_realistic_v3", source)
        self.assertIn("for _ in range(8):", source)

        captured_v4 = self.mcp.call_tool(
            "isaac_capture_realistic_edge_supervisor_sequence",
            {
                "scenario": "aisle_congestion",
                "frame_count": 8,
                "variant": "v4",
            },
        )
        self.assertNotIn("isError", captured_v4)
        payload = json.loads(captured_v4["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("isaac_sim_edge_supervisor_realistic_v4", source)
        self.assertIn("for _ in range(8):", source)

    def test_lightweight_live_warehouse_omits_full_warehouse_reference(self) -> None:
        created = self.mcp.call_tool(
            "isaac_create_lightweight_live_warehouse",
            {},
        )
        self.assertNotIn("isError", created)
        payload = json.loads(created["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn("lightweight_standalone_asset_aisles", source)
        self.assertIn("SM_RackShelf_01.usd", source)
        self.assertIn("SM_RackFrame_03.usd", source)
        self.assertIn('f"{ROOT}/Environment/Floor"', source)
        self.assertIn("UsdPhysics.Scene.Define", source)
        self.assertIn("GroundPlane(", source)
        self.assertIn("(24.0, 18.0, 0.16)", source)
        self.assertIn("(0.0, wall_y, 1.95)", source)
        self.assertIn("(14.8, 0.16, 3.90)", source)
        self.assertIn("(wall_x, 0.8, 1.95)", source)
        self.assertIn("(0.16, 11.1, 3.90)", source)
        self.assertNotIn(
            'f"{ROOT}/Environment/WarehousePlacement/NvidiaWarehouse"',
            source,
        )
        self.assertIn("scale=(2.0, 2.0, 2.0)", source)

    def test_edge_supervisor_progress_is_bounded(self) -> None:
        valid = self.mcp.call_tool(
            "isaac_set_edge_supervisor_progress",
            {"progress": 0.6},
        )
        self.assertNotIn("isError", valid)
        payload = json.loads(valid["content"][0]["text"])
        source = payload["result"]["source"]
        self.assertIn(
            "update_scene(0.6, 'mixed_traffic')",
            source,
        )
        self.assertIn("if beacon_prim.IsValid():", source)
        self.assertIn("if conflict_prim.IsValid():", source)

        invalid = self.mcp.call_tool(
            "isaac_set_edge_supervisor_progress",
            {"progress": 1.1},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("between 0 and 1", invalid["content"][0]["text"])

    def test_edge_supervisor_frame_count_is_bounded(self) -> None:
        result = self.mcp.call_tool(
            "isaac_capture_edge_supervisor_sequence",
            {"frame_count": 7},
        )
        self.assertTrue(result["isError"])
        self.assertIn("between 8 and 48", result["content"][0]["text"])

    def test_edge_supervisor_scenarios_choose_distinct_routes(self) -> None:
        mixed = supervisor_decision(0.75, "mixed_traffic")
        congested = supervisor_decision(0.75, "aisle_congestion")
        blind = supervisor_decision(0.65, "blind_corner")
        self.assertEqual(mixed["route"], "south_bypass")
        self.assertEqual(congested["hazard"], "aisle_congestion")
        self.assertEqual(blind["route"], "north_bypass")

        invalid = self.mcp.call_tool(
            "isaac_set_edge_supervisor_progress",
            {"progress": 0.5, "scenario": "unknown"},
        )
        self.assertTrue(invalid["isError"])
        self.assertIn("scenario must be one of", invalid["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
