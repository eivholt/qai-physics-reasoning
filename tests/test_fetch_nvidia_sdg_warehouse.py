from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.fetch_nvidia_sdg_warehouse import (
    AssetValidationError,
    DEFAULT_MANIFEST,
    _case_frame_indices,
    _paired_frame_cases,
    extract_rgb_frames,
    fetch_asset,
    load_manifest,
)
from scripts.prepare_vision_calibration_pairs import calibration_pairs


class FetchNvidiaSdgWarehouseTests(unittest.TestCase):
    def test_tracked_manifest_pins_source_and_ground_truth_cases(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)

        self.assertEqual(
            manifest["source"]["repo_id"],
            "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes",
        )
        self.assertEqual(manifest["source"]["license"], "OpenMDW-1.1")
        self.assertEqual(len(manifest["source"]["revision"]), 40)
        self.assertEqual(
            manifest["preview_layout"]["panels_left_to_right"][0], "rgb"
        )
        self.assertEqual(manifest["preview_layout"]["rgb_crop_xyxy"], [0, 0, 384, 216])

        labels = {
            case["expected_label"] for case in manifest["classification_cases"]
        }
        self.assertEqual(
            labels,
            {
                "forklift_human_near_miss",
                "forklift_shelf_collision",
                "warehouse_fire_and_evacuation",
                "routine_box_pickup",
            },
        )
        negative = next(
            case
            for case in manifest["classification_cases"]
            if case["expected_label"] == "routine_box_pickup"
        )
        self.assertFalse(negative["is_safety_incident"])
        self.assertTrue(negative["negative_control"])

        predictions = {
            case["id"]: case for case in manifest["prediction_cases"]
        }
        self.assertEqual(
            predictions["predict_near_miss"]["rgb_frame_indices"], [30, 45]
        )
        self.assertEqual(
            predictions["predict_shelf_collision"]["rgb_frame_indices"], [66, 88]
        )
        observations = {
            case["id"]: case for case in manifest["observation_cases"]
        }
        self.assertEqual(
            observations["observe_near_miss_avoidance"]["rgb_frame_indices"],
            [45, 75],
        )
        self.assertEqual(
            observations["observe_barrier_collision"]["rgb_frame_indices"],
            [198, 225],
        )
        box_pickup = observations["observe_routine_box_pickup"]
        self.assertEqual(box_pickup["rgb_frame_indices"], [60, 120])
        self.assertFalse(box_pickup["expected"]["is_safety_incident"])
        self.assertTrue(box_pickup["expected"]["negative_control"])
        profile = manifest["video_case_profile"]
        self.assertEqual(profile["pair_count"], 3)
        self.assertEqual(profile["visual_tokens_per_pair"], 84)
        self.assertEqual(profile["total_visual_tokens"], 252)
        self.assertEqual(profile["context_tokens"], 512)
        self.assertEqual(profile["prefill_ar_tokens"], 128)
        self.assertEqual(profile["decode_ar_tokens"], 1)
        self.assertEqual(profile["prefill_kv_capacity_tokens"], 384)

        videos = {case["id"]: case for case in manifest["video_cases"]}
        barrier = videos["video_barrier_knockdown_4fps"]
        routine = videos["video_routine_box_pickup_4fps"]
        near_miss = videos["video_near_miss_avoidance_4fps"]
        fire_motion = videos["video_fire_evacuation_4fps"]
        self.assertEqual(
            _case_frame_indices(barrier),
            [198, 202, 213, 217, 221, 225],
        )
        self.assertEqual(
            _case_frame_indices(routine),
            [81, 85, 95, 99, 102, 106],
        )
        self.assertEqual(
            _case_frame_indices(near_miss),
            [50, 54, 65, 69, 73, 77],
        )
        self.assertEqual(
            _case_frame_indices(fire_motion),
            [107, 111, 121, 125, 128, 132],
        )
        self.assertTrue(barrier["expected"]["is_safety_incident"])
        self.assertTrue(routine["expected"]["negative_control"])
        self.assertEqual(
            barrier["freeform_evaluation_role"], "diagnostic"
        )
        self.assertEqual(
            routine["freeform_evaluation_role"], "diagnostic"
        )
        self.assertEqual(
            barrier["retained_candidate_pair_indices"], [0, 2, 3]
        )
        self.assertEqual(
            routine["retained_candidate_pair_indices"], [0, 2, 3]
        )
        for case in videos.values():
            self.assertEqual(len(case["temporal_pairs"]), 3)
            budget = case["context_budget"]
            self.assertEqual(
                budget["prompt_tokens"],
                budget["text_tokens"] + budget["visual_tokens"],
            )
            self.assertEqual(
                budget["total_tokens"],
                budget["prompt_tokens"] + budget["max_new_tokens"],
            )
            self.assertEqual(
                budget["headroom_tokens"],
                profile["context_tokens"] - budget["total_tokens"],
            )
            self.assertLessEqual(
                budget["total_tokens"],
                profile["context_tokens"],
            )
            self.assertEqual(
                budget["prefill_headroom_tokens"],
                profile["prefill_kv_capacity_tokens"]
                - budget["prompt_tokens"],
            )
            self.assertLessEqual(
                budget["prompt_tokens"],
                profile["prefill_kv_capacity_tokens"],
            )

        box_probes = {
            probe["id"]: probe for probe in routine["choice_probes"]
        }
        self.assertEqual(
            box_probes["cross_scene_event_choice"]["expected_letter"], "B"
        )
        self.assertEqual(
            box_probes["cross_scene_event_choice_shuffled"][
                "expected_letter"
            ],
            "A",
        )
        self.assertEqual(
            {
                probe["evaluation_role"] for probe in box_probes.values()
            },
            {
                "primary_grounded_evaluation_gate",
                "expanded_grounded_evaluation_gate",
                "prompt_compression_diagnostic",
                "box_near_discrimination_diagnostic",
            },
        )
        barrier_probes = {
            probe["id"]: probe for probe in barrier["choice_probes"]
        }
        self.assertEqual(
            {
                probe["evaluation_role"]
                for probe in barrier_probes.values()
            },
            {
                "primary_grounded_evaluation_gate",
                "expanded_grounded_evaluation_gate",
                "prompt_compression_diagnostic",
            },
        )
        self.assertEqual(
            barrier_probes["cross_scene_event_choice"]["expected_letter"], "A"
        )
        self.assertEqual(
            barrier_probes["cross_scene_event_choice_shuffled"][
                "expected_letter"
            ],
            "C",
        )
        shared_probe_ids = barrier_probes.keys() & box_probes.keys()
        self.assertEqual(
            {
                barrier_probes[probe_id]["prompt"]
                for probe_id in shared_probe_ids
            },
            {
                box_probes[probe_id]["prompt"]
                for probe_id in shared_probe_ids
            },
        )
        for case in videos.values():
            for probe in case["choice_probes"]:
                budget = probe["context_budget"]
                self.assertLessEqual(
                    budget["prompt_tokens"],
                    profile["prefill_kv_capacity_tokens"],
                )
                self.assertEqual(
                    budget["prefill_headroom_tokens"],
                    profile["prefill_kv_capacity_tokens"]
                    - budget["prompt_tokens"],
                )

        planned_calibration = calibration_pairs(manifest)
        self.assertEqual(len(planned_calibration), 20)
        self.assertEqual(
            {
                pair["clip_asset_id"]
                for pair in planned_calibration
            },
            {
                "near_miss_clip",
                "collision_clip",
                "fire_clip",
                "box_pickup_clip",
            },
        )
        calibration_frames: dict[str, set[int]] = {}
        for pair in planned_calibration:
            calibration_frames.setdefault(pair["clip_asset_id"], set()).update(
                pair["frame_indices"]
            )
        for case in videos.values():
            self.assertTrue(
                set(_case_frame_indices(case)).isdisjoint(
                    calibration_frames[case["clip_asset_id"]]
                )
            )

    def test_video_pair_timestamps_match_preview_cadence(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        assets = {asset["id"]: asset for asset in manifest["assets"]}
        for case in manifest["video_cases"]:
            fps = float(assets[case["clip_asset_id"]]["assumed_preview_fps"])
            for pair in case["temporal_pairs"]:
                expected_times = [
                    round(index / fps, 6) for index in pair["frame_indices"]
                ]
                self.assertEqual(
                    pair["assumed_times_seconds"],
                    expected_times,
                )
                expected_midpoint = round(
                    sum(pair["frame_indices"]) / (2 * fps),
                    6,
                )
                self.assertEqual(
                    pair["pair_midpoint_seconds"],
                    expected_midpoint,
                )
                self.assertEqual(
                    pair["prompt_timestamp_text"],
                    f"{pair['pair_midpoint_seconds']:.1f}",
                )

    def test_calibration_rejects_overlap_with_video_case(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        collision_sequence = next(
            sequence
            for sequence in manifest["vision_calibration"]["sequences"]
            if sequence["clip_asset_id"] == "collision_clip"
        )
        collision_sequence["pair_frame_indices"][0] = [198, 202]

        with self.assertRaisesRegex(ValueError, "overlaps evaluation"):
            calibration_pairs(manifest)

    def test_selects_prediction_observation_and_video_pairs(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        selected = _paired_frame_cases(
            manifest,
            [
                "observe_routine_box_pickup",
                "predict_near_miss",
                "video_barrier_knockdown_4fps",
            ],
        )
        self.assertEqual(
            [case["id"] for case in selected],
            [
                "observe_routine_box_pickup",
                "predict_near_miss",
                "video_barrier_knockdown_4fps",
            ],
        )
        all_cases = _paired_frame_cases(manifest, [])
        self.assertEqual(len(all_cases), 9)

    def test_flattens_four_temporal_pairs_for_extraction(self) -> None:
        case = {
            "id": "video_fixture",
            "temporal_pairs": [
                {"frame_indices": [0, 1]},
                {"frame_indices": [3, 4]},
                {"frame_indices": [6, 7]},
                {"frame_indices": [9, 10]},
            ],
        }
        self.assertEqual(
            _case_frame_indices(case),
            [0, 1, 3, 4, 6, 7, 9, 10],
        )

        case["temporal_pairs"][2]["frame_indices"] = [4, 7]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _case_frame_indices(case)

    def test_fetch_is_anonymous_and_validates_hash(self) -> None:
        payload = b"official preview fixture"
        asset = {
            "id": "fixture",
            "relative_path": "assets/fixture.bin",
            "source_url": "https://example.invalid/fixture.bin",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        seen_requests = []

        def fake_urlopen(request):
            seen_requests.append(request)
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as temporary:
            path, downloaded = fetch_asset(
                asset, Path(temporary), urlopen=fake_urlopen
            )
            self.assertTrue(downloaded)
            self.assertEqual(path.read_bytes(), payload)

            _, downloaded = fetch_asset(
                asset, Path(temporary), urlopen=fake_urlopen
            )
            self.assertFalse(downloaded)

        self.assertEqual(len(seen_requests), 1)
        request = seen_requests[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.get_method(), "GET")

    def test_bad_forced_download_does_not_replace_existing_asset(self) -> None:
        expected = b"expected bytes"
        asset = {
            "id": "fixture",
            "relative_path": "assets/fixture.bin",
            "source_url": "https://example.invalid/fixture.bin",
            "size_bytes": len(expected),
            "sha256": hashlib.sha256(expected).hexdigest(),
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "assets" / "fixture.bin"
            destination.parent.mkdir()
            destination.write_bytes(expected)

            with self.assertRaises(AssetValidationError):
                fetch_asset(
                    asset,
                    root,
                    force=True,
                    urlopen=lambda _: io.BytesIO(b"corrupt"),
                )
            self.assertEqual(destination.read_bytes(), expected)
            self.assertFalse(destination.with_name("fixture.bin.part").exists())

    def test_extracts_only_leftmost_rgb_panel(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "assets" / "fixture.gif"
            source.parent.mkdir()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            frames = []
            for color in colors:
                frame = Image.new("RGB", (20, 3), (0, 0, 0))
                frame.paste(color, (0, 0, 4, 3))
                frames.append(frame)
            frames[0].save(
                source,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            asset = {
                "id": "fixture",
                "kind": "preview_clip",
                "relative_path": "assets/fixture.gif",
                "size_bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "frame_count": 3,
            }
            manifest = {
                "preview_layout": {
                    "width": 20,
                    "height": 3,
                    "rgb_crop_xyxy": [0, 0, 4, 3],
                },
                "assets": [asset],
            }
            case = {
                "id": "predict_fixture",
                "clip_asset_id": "fixture",
                "temporal_pairs": [
                    {
                        "frame_indices": [0, 2],
                        "assumed_times_seconds": [0.0, 0.2],
                        "pair_midpoint_seconds": 0.1,
                        "prompt_timestamp_text": "0.1",
                    }
                ],
            }

            written = extract_rgb_frames(manifest, root, case)
            self.assertEqual([path.name for path in written], [
                "frame_0000.png",
                "frame_0002.png",
            ])
            with Image.open(written[0]) as extracted:
                self.assertEqual(extracted.size, (4, 3))
                self.assertEqual(extracted.convert("RGB").getpixel((1, 1)), colors[0])
            with Image.open(written[1]) as extracted:
                self.assertEqual(extracted.convert("RGB").getpixel((1, 1)), colors[2])

    def test_rejects_case_id_path_traversal(self) -> None:
        manifest = {
            "preview_layout": {
                "width": 20,
                "height": 3,
                "rgb_crop_xyxy": [0, 0, 4, 3],
            },
            "assets": [
                {
                    "id": "fixture",
                    "kind": "preview_clip",
                    "relative_path": "assets/fixture.gif",
                    "size_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "frame_count": 1,
                }
            ],
        }
        case = {
            "id": "../escape",
            "clip_asset_id": "fixture",
            "rgb_frame_indices": [0],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "assets" / "fixture.gif"
            source.parent.mkdir()
            source.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Unsafe paired-frame case id"):
                extract_rgb_frames(manifest, root, case)


if __name__ == "__main__":
    unittest.main()
