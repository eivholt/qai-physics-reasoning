import io
import json
import unittest
from unittest import mock

from scripts import request_isaac_edge_supervisor_advisory as advisory


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class SequenceUrlOpen:
    def __init__(self, letters: list[str]):
        self.letters = iter(letters)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        answer = next(self.letters)
        body = {"choices": [{"message": {"content": answer}}]}
        return _Response(json.dumps(body).encode("utf-8"))


class IsaacEdgeSupervisorAdvisoryTests(unittest.TestCase):
    def test_visual_inventory_requests_one_bounded_video_summary(self):
        prompt = advisory.visual_inventory_prompt()

        self.assertIn("attached warehouse image or video", prompt)
        self.assertIn("every frame", prompt)
        self.assertIn("operator cage", prompt)
        self.assertIn("directly visible in this", prompt)
        self.assertIn("Do not infer an actor behind a shelf", prompt)
        self.assertIn("mobile robot: STATUS", prompt)
        self.assertIn("forklift: STATUS", prompt)
        self.assertIn("Do not list individual frames", prompt)

    def test_visual_gap_classification_a_reroutes_without_scene_metadata(self):
        opener = SequenceUrlOpen(
            [
                "mobile robot: PRESENT\nforklift: PRESENT",
                "RobotBlue REROUTE_NORTH_BYPASS",
            ]
        )
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[10.0, 10.5, 11.0, 12.25],
        ):
            result = advisory.request_visual_gap_advisory(
                video_url="file:///tmp/comparison.mp4",
                urlopen=opener,
            )

        self.assertEqual(
            result["raw_command"],
            "RobotBlue REROUTE_NORTH_BYPASS",
        )
        self.assertEqual(result["advisory"]["action"], "REROUTE")
        self.assertEqual(result["advisory"]["route"], "north_bypass")
        self.assertEqual(
            result["model_responses"]["visual_gap_classification"],
            "RobotBlue REROUTE_NORTH_BYPASS",
        )
        presence_payload = json.loads(opener.requests[0][0].data)
        self.assertNotIn("grammar_string", presence_payload)
        self.assertIn(
            "forklift",
            presence_payload["messages"][0]["content"][1]["text"],
        )
        payload = json.loads(opener.requests[1][0].data)
        self.assertEqual(
            payload["grammar_string"],
            advisory.VISION_ONLY_TUTORIAL_GRAMMAR,
        )
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertIn("EARLIER", prompt)
        self.assertIn("NOW", prompt)
        self.assertIn("white low-profile robot", prompt)
        self.assertNotIn("TRACKED_SCENE_FACTS", prompt)
        self.assertNotIn("actor_tracks", prompt)
        self.assertNotIn("coordinates", prompt.lower())

    def test_visual_gap_uncertain_classification_continues(self):
        opener = SequenceUrlOpen(
            ["mobile robot: PRESENT\nforklift: ABSENT"]
        )
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[2.0, 2.75]
        ):
            result = advisory.request_visual_gap_advisory(
                video_url="file:///tmp/uncertain.mp4",
                urlopen=opener,
            )

        self.assertEqual(
            result["raw_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )
        self.assertEqual(result["advisory"]["action"], "CONTINUE")
        self.assertEqual(result["advisory"]["route"], "current")
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(
            result["model_responses"]["visual_gap_classification"],
            "not_requested",
        )

    def test_visual_gap_requires_both_actor_categories_present(self):
        opener = SequenceUrlOpen(
            ["mobile robot: ABSENT\nforklift: PRESENT"]
        )
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[2.0, 2.75]
        ):
            result = advisory.request_visual_gap_advisory(
                video_url="file:///tmp/forklift-only.mp4",
                urlopen=opener,
            )

        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(
            result["model_responses"]["actor_presence"],
            "ACTORS_NOT_BOTH_VISIBLE",
        )
        self.assertEqual(
            result["model_responses"]["visual_gap_classification"],
            "not_requested",
        )
        self.assertEqual(
            result["raw_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )

    def test_rolling_video_uses_chronological_motion_prompt(self):
        opener = SequenceUrlOpen(
            [
                "mobile robot: PRESENT\nforklift: PRESENT",
                "RobotBlue REROUTE_NORTH_BYPASS",
            ]
        )
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[1.0, 2.0, 3.0, 4.5],
        ):
            result = advisory.request_visual_gap_advisory(
                video_url="file:///tmp/rolling-window.mp4",
                vision_input="rolling_video",
                urlopen=opener,
            )

        decision_payload = json.loads(opener.requests[1][0].data)
        prompt = decision_payload["messages"][0]["content"][1]["text"]
        self.assertIn("chronological order", prompt)
        self.assertIn("first frame through its final frame", prompt)
        self.assertIn("earlier frames to later frames", prompt)
        self.assertNotIn("on the left", prompt)
        self.assertNotIn("on the right", prompt)
        self.assertEqual(result["vision_input"], "rolling_video")
        self.assertIn("rolling_video", result["scope"])

    def test_visual_gap_rejects_unknown_input_packaging(self):
        with self.assertRaisesRegex(ValueError, "vision_input"):
            advisory.request_visual_gap_advisory(
                video_url="file:///tmp/window.mp4",
                vision_input="unknown",
            )

    def test_cli_forwards_rolling_video_to_visual_gap_mode(self):
        with (
            mock.patch.object(
                advisory,
                "request_visual_gap_advisory",
                return_value={"ok": True},
            ) as request,
            mock.patch("builtins.print"),
        ):
            status = advisory.main(
                [
                    "--scenario",
                    "blind_corner",
                    "--video-url",
                    "file:///tmp/window.mp4",
                    "--visual-gap-only",
                    "--vision-input",
                    "rolling_video",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(request.call_args.kwargs["vision_input"], "rolling_video")

    def test_passage_status_prompt_is_generic_and_command_free(self):
        prompt = advisory.generic_passage_supervisor_prompt("rolling_video")

        self.assertIn("warehouse-wide traffic observer", prompt)
        self.assertIn("mobile robot: STATUS", prompt)
        self.assertIn('"left_passage"', prompt)
        self.assertIn('"right_passage"', prompt)
        self.assertIn("NOT_VISIBLE", prompt)
        self.assertIn("BLOCKAGE_IMMINENT", prompt)
        self.assertIn("chronological order", prompt)
        self.assertIn("latest frame", prompt)
        self.assertIn("field placeholders, not suggested answers", prompt)
        self.assertIn("yellow-guarded image-right end", prompt)
        self.assertNotIn("RobotBlue", prompt)
        self.assertNotIn("REROUTE", prompt)
        self.assertNotIn("CONTINUE_CURRENT_ROUTE", prompt)

    def test_passage_status_imminent_maps_through_local_adapter(self):
        report = json.dumps(
            {
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
            },
            separators=(",", ":"),
        )
        condition = json.dumps(
            {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
                "right_passage": {
                    "status": "BLOCKAGE_IMMINENT",
                    "offender": "FORKLIFT",
                },
            },
            separators=(",", ":"),
        )
        opener = SequenceUrlOpen(
            [
                "mobile robot: PRESENT\nforklift: PRESENT",
                condition,
            ]
        )
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[10.0, 10.5, 11.0, 12.25],
        ):
            result = advisory.request_passage_status_advisory(
                video_url="file:///tmp/window.mp4",
                vision_input="rolling_video",
                urlopen=opener,
            )

        self.assertEqual(result["model_output"], report)
        self.assertEqual(result["advisory"]["action"], "REROUTE")
        self.assertEqual(result["advisory"]["route"], "north_bypass")
        self.assertEqual(
            result["raw_command"],
            "RobotBlue REROUTE_NORTH_BYPASS",
        )
        self.assertEqual(
            result["command_source"],
            "deterministic_passage_status_adapter",
        )
        self.assertEqual(
            result["model_responses"]["traffic_actors"],
            ["mobile_robot", "forklift"],
        )
        inventory_payload = json.loads(opener.requests[0][0].data)
        self.assertNotIn("grammar_string", inventory_payload)
        payload = json.loads(opener.requests[1][0].data)
        self.assertEqual(
            payload["grammar_string"],
            advisory.PASSAGE_CONDITION_GRAMMAR,
        )
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertNotIn("RobotBlue", prompt)
        self.assertNotIn("REROUTE", prompt)

    def test_passage_status_clear_maps_to_continue(self):
        report = json.dumps(
            {
                "traffic_actors": ["mobile_robot"],
                "passage_states": {
                    "left_passage": {
                        "status": "CLEAR",
                        "offender": "NONE",
                    },
                    "right_passage": {
                        "status": "CLEAR",
                        "offender": "NONE",
                    },
                },
                "passages_blocked": [],
                "passages_blockage_imminent": [],
            },
            separators=(",", ":"),
        )
        condition = json.dumps(
            {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
                "right_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
            },
            separators=(",", ":"),
        )
        opener = SequenceUrlOpen(
            [
                "mobile robot: PRESENT\nforklift: UNCERTAIN",
                condition,
            ]
        )
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[2.0, 2.25, 3.0, 3.5],
        ):
            result = advisory.request_passage_status_advisory(
                video_url="file:///tmp/window.mp4",
                urlopen=opener,
            )

        self.assertEqual(result["advisory"]["action"], "CONTINUE")
        self.assertEqual(result["advisory"]["route"], "current")
        self.assertEqual(
            result["raw_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )

    def test_passage_status_accepts_uncertain_inventory_when_passage_names_forklift(
        self,
    ):
        condition = json.dumps(
            {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
                "right_passage": {
                    "status": "BLOCKED",
                    "offender": "FORKLIFT",
                },
            },
            separators=(",", ":"),
        )
        opener = SequenceUrlOpen(
            [
                "mobile robot: PRESENT\nforklift: UNCERTAIN",
                condition,
            ]
        )

        result = advisory.request_passage_status_advisory(
            video_url="file:///tmp/window.mp4",
            urlopen=opener,
        )

        self.assertEqual(
            result["raw_command"],
            "RobotBlue REROUTE_NORTH_BYPASS",
        )
        self.assertEqual(
            result["gateway_evidence"]["forklift_inventory_status"],
            "UNCERTAIN",
        )

    def test_passage_status_rejects_blocker_when_inventory_says_absent(self):
        condition = json.dumps(
            {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
                "right_passage": {
                    "status": "BLOCKED",
                    "offender": "FORKLIFT",
                },
            },
            separators=(",", ":"),
        )
        opener = SequenceUrlOpen(
            [
                "mobile robot: PRESENT\nforklift: ABSENT",
                condition,
            ]
        )

        result = advisory.request_passage_status_advisory(
            video_url="file:///tmp/window.mp4",
            urlopen=opener,
        )

        self.assertEqual(
            result["raw_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )

    def test_passage_status_normalizes_fenced_pretty_json(self):
        report = """\
```json
{
  "traffic_actors": ["mobile_robot", "forklift"],
  "passage_states": {
    "left_passage": {"status": "CLEAR", "offender": "NONE"},
    "right_passage": {"status": "BLOCKED", "offender": "FORKLIFT"}
  },
  "passages_blocked": ["right"],
  "passages_blockage_imminent": []
}
```"""

        normalized = advisory.normalize_passage_status_report(report)
        parsed = advisory.parse_passage_status_report(normalized)

        self.assertEqual(parsed["traffic_actors"], ["mobile_robot", "forklift"])
        self.assertEqual(parsed["passages_blocked"], ["right"])
        self.assertEqual(
            parsed["passage_states"]["right_passage"],
            {"status": "BLOCKED", "offender": "FORKLIFT"},
        )
        self.assertEqual(
            parsed["passages_blockage_imminent"],
            [],
        )

    def test_passage_condition_repairs_offender_from_clear_status(self):
        report = json.dumps(
            {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "MOBILE_ROBOT",
                },
                "right_passage": {
                    "status": "CLEAR",
                    "offender": "UNKNOWN",
                },
            }
        )

        normalized = json.loads(
            advisory.normalize_passage_condition_report(report)
        )

        self.assertEqual(
            normalized,
            {
                "left_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
                "right_passage": {
                    "status": "CLEAR",
                    "offender": "NONE",
                },
            },
        )

    def test_passage_condition_repairs_unspecified_blocker_to_unknown(self):
        report = json.dumps(
            {
                "left_passage": {
                    "status": "UNCERTAIN",
                    "offender": "FORKLIFT",
                },
                "right_passage": {
                    "status": "BLOCKED",
                    "offender": "NONE",
                },
            }
        )

        normalized = json.loads(
            advisory.normalize_passage_condition_report(report)
        )

        self.assertEqual(
            normalized,
            {
                "left_passage": {
                    "status": "UNCERTAIN",
                    "offender": "UNKNOWN",
                },
                "right_passage": {
                    "status": "BLOCKED",
                    "offender": "UNKNOWN",
                },
            },
        )

    def test_cli_forwards_rolling_video_to_passage_status_mode(self):
        with (
            mock.patch.object(
                advisory,
                "request_passage_status_advisory",
                return_value={"ok": True},
            ) as request,
            mock.patch("builtins.print"),
        ):
            status = advisory.main(
                [
                    "--scenario",
                    "blind_corner",
                    "--video-url",
                    "file:///tmp/window.mp4",
                    "--passage-status-only",
                    "--vision-input",
                    "rolling_video",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(request.call_args.kwargs["vision_input"], "rolling_video")

    def test_oracle_route_for_each_scenario(self):
        cases = {
            "mixed_traffic": (["B", "A"], "south_bypass"),
            "aisle_congestion": (["B", "A"], "south_bypass"),
            "blind_corner": (["A", "A"], "north_bypass"),
        }
        for scenario, (answers, expected_route) in cases.items():
            with self.subTest(scenario=scenario):
                opener = SequenceUrlOpen(answers)
                with mock.patch.object(
                    advisory.time,
                    "monotonic",
                    side_effect=[1.0, 2.0, 3.0, 5.0],
                ):
                    result = advisory.request_advisory(
                        scenario=scenario,
                        video_url="file:///tmp/clip.mp4",
                        timeout_seconds=12.0,
                        urlopen=opener,
                    )

                self.assertEqual(result["advisory"]["action"], "REROUTE")
                self.assertEqual(result["advisory"]["route"], expected_route)
                self.assertTrue(result["scenario_oracle"]["match"])
                self.assertEqual(result["request_seconds"], [1.0, 2.0])
                self.assertEqual(result["total_seconds"], 3.0)
                self.assertEqual(len(opener.requests), 2)

                first_request, timeout = opener.requests[0]
                self.assertEqual(
                    first_request.full_url,
                    "http://127.0.0.1:18181/v1/chat/completions",
                )
                self.assertEqual(timeout, 12.0)
                self.assertEqual(first_request.get_header("Connection"), "close")
                first_payload = json.loads(first_request.data)
                self.assertEqual(first_payload["model"], advisory.MODEL)
                self.assertEqual(first_payload["grammar_string"], "root ::= [AB]")
                self.assertEqual(
                    first_payload["messages"][0]["content"][0]["image_url"]["url"],
                    "file:///tmp/clip.mp4",
                )

                second_payload = json.loads(opener.requests[1][0].data)
                self.assertEqual(
                    second_payload["messages"][0]["content"][1]["text"],
                    advisory.SCENARIOS[scenario]["route_prompt"],
                )

    def test_clear_answer_returns_direct_without_route_request(self):
        opener = SequenceUrlOpen(["A"])
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[10.0, 11.5]
        ):
            result = advisory.request_advisory(
                scenario="aisle_congestion",
                video_url="file:///tmp/clear.mp4",
                urlopen=opener,
            )

        self.assertEqual(result["advisory"]["action"], "CONTINUE")
        self.assertEqual(result["advisory"]["route"], "direct")
        self.assertFalse(result["scenario_oracle"]["match"])
        self.assertEqual(result["request_seconds"], [1.5])
        self.assertEqual(len(opener.requests), 1)

    def test_rejects_out_of_grammar_response(self):
        opener = SequenceUrlOpen(["C"])
        with self.assertRaisesRegex(ValueError, "out-of-grammar"):
            advisory.request_advisory(
                scenario="blind_corner",
                video_url="file:///tmp/clip.mp4",
                urlopen=opener,
            )

    def test_live_aisle_conflict_uses_one_request_and_policy_route(self):
        opener = SequenceUrlOpen(["B"])
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[20.0, 25.25]
        ):
            result = advisory.request_live_aisle_advisory(
                video_url="file:///tmp/live.mp4",
                navigation_intention="RobotBlue is driving east to Dock 2",
                urlopen=opener,
            )

        self.assertEqual(result["advisory"]["action"], "REROUTE")
        self.assertEqual(result["advisory"]["route"], "south_bypass")
        self.assertEqual(result["request_seconds"], [5.25])
        self.assertEqual(len(opener.requests), 1)
        request_payload = json.loads(opener.requests[0][0].data)
        prompt = request_payload["messages"][0]["content"][1]["text"]
        self.assertIn("RobotBlue is driving east to Dock 2", prompt)
        self.assertIn("Do you see a problem ahead", prompt)

    def test_live_aisle_clear_result_continues_direct(self):
        opener = SequenceUrlOpen(["A"])
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[2.0, 3.0]
        ):
            result = advisory.request_live_aisle_advisory(
                video_url="file:///tmp/live-clear.mp4",
                urlopen=opener,
            )

        self.assertEqual(result["advisory"]["action"], "CONTINUE")
        self.assertEqual(result["advisory"]["route"], "direct")
        self.assertFalse(result["scenario_oracle"]["match"])

    def test_generic_supervisor_returns_complete_bounded_command(self):
        opener = SequenceUrlOpen(["RobotBlue REROUTE NORTH_BYPASS"])
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[4.0, 7.25]
        ):
            result = advisory.request_generic_supervisor_advisory(
                video_url="file:///tmp/generic.mp4",
                commandable_actors=("RobotBlue",),
                urlopen=opener,
            )

        self.assertEqual(
            result["raw_command"],
            "RobotBlue REROUTE NORTH_BYPASS",
        )
        self.assertEqual(result["advisory"]["action"], "REROUTE")
        self.assertEqual(result["advisory"]["route"], "north_bypass")
        self.assertEqual(result["commandable_actors"], ["RobotBlue"])
        payload = json.loads(opener.requests[0][0].data)
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertIn("safety supervisor for all visible warehouse", prompt)
        self.assertIn("Commandable actors currently visible: RobotBlue", prompt)
        self.assertIn("NORTH_BYPASS means trafficable space", prompt)
        self.assertIn("No route illustrations are present", prompt)
        self.assertIn("do not intervene solely", prompt)
        self.assertNotIn("navigation intention", prompt.lower())
        self.assertEqual(
            payload["grammar_string"],
            advisory.GENERIC_COMMAND_GRAMMAR,
        )

    def test_generic_supervisor_rejects_uncommandable_actor(self):
        with self.assertRaisesRegex(ValueError, "exactly one commandable actor"):
            advisory.request_generic_supervisor_advisory(
                video_url="file:///tmp/generic.mp4",
                commandable_actors=("RobotBlue", "ForkliftOrange"),
            )

    def test_vision_only_tutorial_prompt_has_no_dynamic_scene_facts(self):
        visual_report = (
            "ACTORS: one white mobile robot\n"
            "LOCATIONS: robot in an open aisle\n"
            "MOTION: moving east\n"
            "DISTANCE: indeterminate\n"
            "PATH_OVERLAP: no, no other mobile actor\n"
            "UNCERTAINTY: none"
        )
        opener = SequenceUrlOpen(
            [visual_report, "RobotBlue CONTINUE_CURRENT_ROUTE"]
        )
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[1.0, 2.25, 3.0, 4.5],
        ):
            result = advisory.request_vision_only_tutorial_advisory(
                video_url="file:///tmp/camera-only.mp4",
                urlopen=opener,
            )

        self.assertEqual(
            result["raw_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )
        self.assertEqual(
            result["command_source"],
            "gateway_confirmed_model_policy_from_visual_report",
        )
        self.assertEqual(result["request_seconds"], [1.25, 1.5])
        self.assertEqual(result["model_responses"]["visual_report"], visual_report)
        self.assertEqual(
            result["model_responses"]["policy_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )
        self.assertEqual(
            result["gateway_command"],
            "RobotBlue CONTINUE_CURRENT_ROUTE",
        )
        self.assertEqual(len(opener.requests), 2)

        perception_payload = json.loads(opener.requests[0][0].data)
        perception_prompt = perception_payload["messages"][0]["content"][1]["text"]
        self.assertNotIn("grammar_string", perception_payload)
        self.assertEqual(
            perception_prompt,
            advisory.vision_only_perception_prompt(),
        )
        self.assertNotIn("TRACKED_SCENE_FACTS", perception_prompt)
        self.assertNotIn("EMERGING_BLOCK", perception_prompt)
        self.assertNotIn("ForkliftOrange", perception_prompt)
        self.assertNotIn('"actor_tracks"', perception_prompt)
        self.assertNotIn('"route_statuses"', perception_prompt)

        command_payload = json.loads(opener.requests[1][0].data)
        self.assertEqual(
            command_payload["messages"][0]["content"],
            [{"type": "text", "text": result["prompts"]["command"]}],
        )
        command_prompt = command_payload["messages"][0]["content"][0]["text"]
        self.assertEqual(
            command_payload["grammar_string"],
            advisory.VISION_ONLY_TUTORIAL_GRAMMAR,
        )
        self.assertIn(visual_report, command_prompt)
        self.assertNotIn(
            "RobotBlue REROUTE_SOUTH_BYPASS",
            command_payload["grammar_string"],
        )
        self.assertEqual(
            set(advisory.VISION_ONLY_TUTORIAL_COMMANDS),
            {
                "RobotBlue CONTINUE_CURRENT_ROUTE",
                "RobotBlue REROUTE_NORTH_BYPASS",
            },
        )

    def test_visual_report_gateway_rejects_static_object_false_positive(self):
        report = (
            "ACTORS: A white vehicle moves near a shelf. A forklift is parked.\n"
            "MOTION: The forklift remains stationary. The white vehicle "
            "approaches a yellow pallet.\n"
            "DISTANCE: The vehicle gets closer to the pallet.\n"
            "PATH_OVERLAP: The vehicle overlaps the shelf area."
        )
        self.assertFalse(advisory.visual_report_supports_reroute(report))

    def test_visual_report_gateway_accepts_linked_two_actor_conflict(self):
        report = (
            "ACTORS: A white robot and a forklift are visible.\n"
            "MOTION: The robot moves toward the forklift.\n"
            "DISTANCE: The distance between the robot and forklift decreases.\n"
            "PATH_OVERLAP: Their traffic paths overlap."
        )
        self.assertTrue(advisory.visual_report_supports_reroute(report))

    def test_visual_report_gateway_accepts_now_as_closest_gap(self):
        report = (
            "TRAFFIC_ACTORS: A compact white robot and a dark forklift.\n"
            "CLOSEST_GAP: white robot to forklift = NOW.\n"
            "CONFLICT_GEOMETRY: YES"
        )
        self.assertTrue(advisory.visual_report_supports_reroute(report))

    def test_shared_passage_risk_alone_is_not_enough(self):
        report = (
            "TRAFFIC_ACTORS: A compact white robot and a dark forklift.\n"
            "SHARED_PASSAGE_RISK: yes"
        )
        self.assertFalse(advisory.visual_report_supports_reroute(report))

    def test_gap_cannot_introduce_actor_missing_from_inventory(self):
        report = (
            "TRAFFIC_ACTORS: commandable RobotBlue = EARLIER; "
            "commandable RobotBlue = NOW\n"
            "CLOSEST_GAP: commandable RobotBlue to warehouse forklift = NOW\n"
            "CONFLICT_GEOMETRY: NO"
        )
        self.assertFalse(advisory.visual_report_supports_reroute(report))

    def test_tracked_state_supervisor_selects_only_open_bypass(self):
        opener = SequenceUrlOpen(["RobotBlue REROUTE NORTH_BYPASS"])
        facts = {
            "navigation": {
                "evaluated_current_corridor_status": "EMERGING_BLOCK",
                "route_statuses": {
                    "direct": "EMERGING_BLOCK",
                    "north_bypass": "OPEN",
                    "south_bypass": "BLOCKED_STATIC_PALLETS",
                },
            },
            "actor_tracks": {
                "RobotBlue": {"motion": "east"},
                "ForkliftOrange": {"motion": "south"},
            },
        }
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[2.0, 3.75],
        ):
            result = advisory.request_tracked_state_supervisor_advisory(
                video_url="file:///tmp/tracked.mp4",
                tracked_scene_facts=facts,
                urlopen=opener,
            )

        self.assertEqual(result["advisory"]["action"], "REROUTE")
        self.assertEqual(result["advisory"]["route"], "north_bypass")
        self.assertEqual(
            result["command_source"],
            "model_from_camera_and_tracked_scene_facts",
        )
        payload = json.loads(opener.requests[0][0].data)
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertIn("TRACKED_SCENE_FACTS=", prompt)
        self.assertIn("EMERGING_BLOCK", prompt)
        self.assertIn("BLOCKED_STATIC_PALLETS", prompt)
        self.assertNotIn(
            "RobotBlue REROUTE SOUTH_BYPASS",
            payload["grammar_string"],
        )
        self.assertIn(
            "RobotBlue REROUTE NORTH_BYPASS",
            payload["grammar_string"],
        )

    def test_tracked_state_supervisor_clear_route_continues(self):
        opener = SequenceUrlOpen(["RobotBlue CONTINUE CURRENT_ROUTE"])
        result = advisory.request_tracked_state_supervisor_advisory(
            video_url="file:///tmp/clear.mp4",
            tracked_scene_facts={
                "navigation": {
                    "evaluated_current_corridor_status": "CLEAR",
                    "route_statuses": {
                        "direct": "CLEAR",
                        "north_bypass": "OPEN",
                        "south_bypass": "BLOCKED_STATIC_PALLETS",
                    },
                }
            },
            urlopen=opener,
        )
        self.assertEqual(result["advisory"]["action"], "CONTINUE")
        self.assertEqual(result["advisory"]["route"], "current")
        self.assertEqual(result["advisory"]["hazard"], "none")

    def test_staged_generic_clear_scene_uses_one_request(self):
        opener = SequenceUrlOpen(["NO_CONFLICT"])
        with mock.patch.object(
            advisory.time, "monotonic", side_effect=[10.0, 11.25]
        ):
            result = advisory.request_staged_generic_supervisor_advisory(
                video_url="file:///tmp/generic-clear.mp4",
                urlopen=opener,
            )

        self.assertEqual(
            result["model_responses"]["conflict"],
            "NO_CONFLICT",
        )
        self.assertIsNone(result["model_responses"]["intervention"])
        self.assertEqual(
            result["raw_command"],
            "RobotBlue CONTINUE CURRENT_ROUTE",
        )
        self.assertEqual(
            result["command_source"],
            "gateway_from_model_clear_classification",
        )
        self.assertEqual(len(opener.requests), 1)
        payload = json.loads(opener.requests[0][0].data)
        self.assertEqual(
            payload["grammar_string"],
            advisory.GENERIC_CONFLICT_GRAMMAR,
        )
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertIn("all visible warehouse traffic", prompt)
        self.assertIn("Static shelves", prompt)
        self.assertIn("compact white autonomous mobile robot", prompt)
        self.assertIn("large dark industrial vehicle", prompt)
        self.assertNotIn("navigation intention", prompt.lower())

    def test_staged_generic_conflict_uses_model_intervention_command(self):
        opener = SequenceUrlOpen(
            ["CONFLICT", "RobotBlue REROUTE NORTH_BYPASS"]
        )
        with mock.patch.object(
            advisory.time,
            "monotonic",
            side_effect=[1.0, 2.0, 3.0, 5.5],
        ):
            result = advisory.request_staged_generic_supervisor_advisory(
                video_url="file:///tmp/generic-conflict.mp4",
                urlopen=opener,
            )

        self.assertEqual(
            result["model_responses"],
            {
                "conflict": "CONFLICT",
                "intervention": "RobotBlue REROUTE NORTH_BYPASS",
            },
        )
        self.assertEqual(result["advisory"]["action"], "REROUTE")
        self.assertEqual(result["advisory"]["route"], "north_bypass")
        self.assertEqual(result["command_source"], "model_intervention_command")
        self.assertEqual(result["request_seconds"], [1.0, 2.5])
        self.assertEqual(len(opener.requests), 2)
        second_payload = json.loads(opener.requests[1][0].data)
        self.assertEqual(
            second_payload["grammar_string"],
            advisory.GENERIC_INTERVENTION_GRAMMAR,
        )
        intervention_prompt = second_payload["messages"][0]["content"][1][
            "text"
        ]
        self.assertIn("No route illustrations", intervention_prompt)

    def test_parser_requires_scenario_and_one_video_source(self):
        parser = advisory.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--scenario",
                    "mixed_traffic",
                    "--video",
                    "a.mp4",
                    "--video-url",
                    "file:///b.mp4",
                ]
            )


if __name__ == "__main__":
    unittest.main()
