import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_live_isaac_evk_supervisor as live


class LiveIsaacEvkSupervisorTests(unittest.TestCase):
    def test_encode_clean_twenty_frame_rolling_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [root / f"source_{index:03d}.png" for index in range(20)]
            for frame in frames:
                frame.touch()
            output = root / "request"
            commands = []

            def fake_run(command, timeout_seconds):
                commands.append(command)
                (output / "supervisor_window.mp4").touch()

            with mock.patch.object(live, "_run", side_effect=fake_run):
                clip = live._encode_frames(
                    frame_paths=frames,
                    output_dir=output,
                    model_fps=2.0,
                    video_width=384,
                    clean_video=True,
                )

            self.assertEqual(clip.name, "supervisor_window.mp4")
            self.assertEqual(
                len(list(output.glob("frame_*.png"))),
                20,
            )

        command = commands[0]
        self.assertEqual(command[command.index("-framerate") + 1], "2")
        self.assertEqual(command[command.index("-r") + 1], "2")
        video_filter = command[command.index("-vf") + 1]
        self.assertEqual(video_filter, "scale=384:-2:flags=lanczos")
        self.assertNotIn("drawtext", video_filter)
        self.assertNotIn("EARLIER", " ".join(command))
        self.assertNotIn("NOW", " ".join(command))

    def test_encode_temporal_comparison_for_evk_uses_stable_video_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [root / "source_0.png", root / "source_1.png"]
            for frame in frames:
                frame.touch()
            output = root / "request"
            commands = []

            def fake_run(command, timeout_seconds):
                commands.append(command)
                if len(commands) == 1:
                    (output / "supervisor_temporal_comparison.png").touch()
                else:
                    (output / "supervisor_temporal_comparison.mp4").touch()

            with mock.patch.object(live, "_run", side_effect=fake_run):
                clip = live._encode_frames(
                    frame_paths=frames,
                    output_dir=output,
                    model_fps=4.0,
                    video_width=384,
                    temporal_comparison=True,
                    comparison_video=True,
                )

        self.assertEqual(clip.name, "supervisor_temporal_comparison.mp4")
        self.assertEqual(len(commands), 2)
        video_filter = commands[1][commands[1].index("-vf") + 1]
        self.assertIn("scale=384:216", video_filter)
        self.assertIn("pad=384:216", video_filter)
        self.assertIn("force_original_aspect_ratio=decrease", video_filter)

    def test_decode_tool_response_unwraps_json_output(self):
        response = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "status": "ok",
                            "output": json.dumps({"action": "REROUTE"}),
                            "result": None,
                        }
                    ),
                }
            ]
        }
        self.assertEqual(
            live._decode_tool_response(response),
            {"action": "REROUTE"},
        )

    def test_decode_tool_response_rejects_errors(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            live._decode_tool_response(
                {
                    "isError": True,
                    "content": [{"type": "text", "text": "boom"}],
                }
            )

    def test_windows_path_maps_to_wsl_mount(self):
        mapped = live._windows_path_to_wsl(
            Path(r"C:\Users\eivho\clip.mp4")
        )
        self.assertEqual(mapped, "/mnt/c/Users/eivho/clip.mp4")

    def test_parser_defaults_to_measured_two_fps_profile(self):
        args = live.build_parser().parse_args([])
        self.assertEqual(args.capture_fps, 2.0)
        self.assertEqual(args.model_fps, 2.0)
        self.assertEqual(args.frames_per_window, 8)
        self.assertEqual(args.sensor_camera, "tactical")
        self.assertEqual(args.vision_input, "rolling_video")
        self.assertEqual(args.presentation_camera, "sensor")
        self.assertFalse(args.presentation_route_visualizations)
        self.assertEqual(args.supervisor_profile, "vision_only")
        self.assertEqual(args.motion_controller, "isaac_graph")
        self.assertEqual(args.robot_speed, 0.3)
        self.assertEqual(args.inference_backend, "evk")
        self.assertEqual(args.scenario, "blind_corner")
        self.assertEqual(args.destination_settle_frames, 4)
        self.assertEqual(args.max_runtime_seconds, 300.0)
        self.assertEqual(
            args.host_server_url,
            "http://127.0.0.1:18080",
        )
        self.assertEqual(args.evk_video_width, 384)
        self.assertEqual(args.robot_scale, 1.75)
        self.assertEqual(args.safety_hold_distance, 3.2)
        self.assertEqual(args.clear_baseline_responses, 0)
        self.assertEqual(args.forklift_release_x, 1.5)
        self.assertEqual(args.max_responses, 0)
        self.assertEqual(args.commandable_actors, "RobotBlue")
        self.assertEqual(args.request_retry_seconds, 5.0)
        self.assertEqual(args.max_consecutive_failures, 3)

        rolling = live.build_parser().parse_args(
            [
                "--supervisor-profile",
                "vision_only",
                "--vision-input",
                "rolling_video",
                "--frames-per-window",
                "20",
                "--capture-fps",
                "2",
                "--model-fps",
                "2",
            ]
        )
        self.assertEqual(rolling.vision_input, "rolling_video")
        self.assertEqual(rolling.frames_per_window, 20)
        self.assertEqual(rolling.capture_fps, 2.0)
        self.assertEqual(rolling.model_fps, 2.0)

    def test_evk_client_forwards_rolling_video_layout(self):
        client = live.EvkSshClient(
            target="ubuntu@example",
            remote_root="/tmp/live",
            timeout_seconds=30.0,
        )
        result_json = json.dumps(
            {
                "raw_command": "RobotBlue CONTINUE_CURRENT_ROUTE",
                "advisory": {"action": "CONTINUE", "route": "current"},
            }
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(live.EvkSshClient, "copy"),
            mock.patch.object(
                live.EvkSshClient,
                "remote",
                return_value=result_json,
            ) as remote,
        ):
            clip = Path(directory) / "window.mp4"
            clip.touch()
            client.request(
                clip=clip,
                request_id="live-001",
                commandable_actors=("RobotBlue",),
                supervisor_profile="vision_only",
                tracked_scene_facts=None,
                vision_input="rolling_video",
            )

        arguments = remote.call_args.args[0]
        self.assertIn("--passage-status-only", arguments)
        self.assertNotIn("--visual-gap-only", arguments)
        index = arguments.index("--vision-input")
        self.assertEqual(arguments[index + 1], "rolling_video")

    def test_atomic_status_writer_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live_status.json"
            live._atomic_write_json(path, {"state": "inference", "request": 3})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"state": "inference", "request": 3},
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_physical_destination_stops_even_when_route_is_unexpected(self):
        reached, route_matches = live._destination_observation(
            scenario_released=True,
            last_tick={
                "destination_reached": True,
                "route": "north_bypass",
            },
            expected_route="direct",
        )
        self.assertTrue(reached)
        self.assertFalse(route_matches)

        held, held_route_matches = live._destination_observation(
            scenario_released=False,
            last_tick={
                "destination_reached": True,
                "route": "direct",
            },
            expected_route="direct",
        )
        self.assertFalse(held)
        self.assertIsNone(held_route_matches)

    def test_host_client_uses_repo_relative_media_url_and_bf16_model(self):
        client = live.HostCosmosClient(
            server_url="http://127.0.0.1:18080",
            timeout_seconds=10.0,
        )
        with tempfile.TemporaryDirectory(dir=live.REPO_ROOT) as directory:
            clip = Path(directory) / "window.mp4"
            clip.touch()
            expected = {
                "advisory": {"action": "CONTINUE", "route": "current"},
                "raw_command": "RobotBlue CONTINUE_CURRENT_ROUTE",
            }
            with mock.patch.object(
                live,
                "request_passage_status_advisory",
                return_value=expected.copy(),
            ) as request:
                result = client.request(
                    clip=clip,
                    request_id="live-001",
                    commandable_actors=("RobotBlue",),
                    supervisor_profile="vision_only",
                    tracked_scene_facts=None,
                )
        kwargs = request.call_args.kwargs
        self.assertTrue(kwargs["video_url"].startswith("file://"))
        self.assertTrue(kwargs["video_url"].endswith("/window.mp4"))
        self.assertEqual(kwargs["model"], live.HOST_MODEL)
        self.assertEqual(result["request_id"], "live-001")

    def test_presentation_summary_separates_model_state_from_adapter(self):
        responses = {
            "traffic_actors": ["mobile_robot", "forklift"],
            "passage_states": {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
                "right_passage": {
                    "status": "BLOCKAGE_IMMINENT",
                    "offender": "FORKLIFT",
                },
            },
            "passages_blocked": [],
            "passages_blockage_imminent": [
                {"name": "right", "offender": "forklift"}
            ],
        }

        self.assertEqual(
            live._compact_actor_inventory(responses),
            "mobile robot, forklift",
        )
        self.assertEqual(
            live._compact_passage_state(responses),
            "left: CLEAR  |  right: BLOCKAGE_IMMINENT <- forklift",
        )

    def test_tracked_scene_facts_mark_blind_corner_and_open_north(self):
        facts = live._tracked_scene_facts(
            state_history=(
                {
                    "timeline_time_seconds": 10.0,
                    "blue_position": [1.0, 1.6, 0.0],
                    "forklift_position": [6.1, 4.8, 0.0],
                    "route": "direct",
                },
                {
                    "timeline_time_seconds": 12.0,
                    "blue_position": [2.2, 1.6, 0.0],
                    "forklift_position": [6.1, 4.0, 0.0],
                    "route": "direct",
                },
            ),
            phase="hazard",
            forklift_released=True,
        )
        navigation = facts["navigation"]
        self.assertEqual(
            navigation["evaluated_current_corridor_status"],
            "EMERGING_BLOCK",
        )
        self.assertEqual(navigation["route_statuses"]["north_bypass"], "OPEN")
        self.assertEqual(
            navigation["route_statuses"]["south_bypass"],
            "BLOCKED_STATIC_PALLETS",
        )
        self.assertEqual(
            facts["actor_tracks"]["ForkliftBlue"]["motion"],
            "south",
        )

    def test_tracked_scene_facts_clear_baseline_evaluates_direct(self):
        state = {
            "timeline_time_seconds": 2.0,
            "blue_position": [-5.6, 1.6, 0.0],
            "forklift_position": [6.1, 4.8, 0.0],
            "route": "hold",
        }
        facts = live._tracked_scene_facts(
            state_history=(state,),
            phase="clear_baseline",
            forklift_released=False,
        )
        self.assertEqual(
            facts["navigation"]["evaluated_current_route"],
            "direct",
        )
        self.assertEqual(
            facts["navigation"]["evaluated_current_corridor_status"],
            "CLEAR",
        )
        self.assertFalse(
            facts["actor_tracks"]["ForkliftBlue"]["visible"],
        )

    def test_clear_control_disables_forklift_in_tracked_facts(self):
        state = {
            "timeline_time_seconds": 4.0,
            "blue_position": [-4.0, 1.6, 0.0],
            "forklift_position": [6.1, 4.8, 0.0],
            "route": "direct",
        }
        facts = live._tracked_scene_facts(
            state_history=(state,),
            phase="clear_control",
            forklift_released=False,
            forklift_enabled=False,
        )
        self.assertNotIn("ForkliftBlue", facts["actor_tracks"])
        self.assertEqual(
            set(facts["navigation"]["route_statuses"]),
            {"direct"},
        )
        self.assertEqual(
            facts["navigation"]["evaluated_current_corridor_status"],
            "CLEAR",
        )


if __name__ == "__main__":
    unittest.main()
