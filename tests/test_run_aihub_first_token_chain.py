from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.run_aihub_first_token_chain import (
    P1_GRAPH_NAME,
    cache_inputs,
    initialize_state,
    load_ground_truth,
    parse_candidates,
    submit_job,
    score_logits_tensor,
    validate_output_h5,
    validate_chunk0_static_binding,
    validate_frozen_prompt_ids,
    validate_seed_dataset,
    validate_vision_scene_binding,
)

HAS_H5PY = importlib.util.find_spec("h5py") is not None


class CandidateAndResumeTests(unittest.TestCase):
    def test_candidate_order_is_explicit_and_duplicates_rejected(self) -> None:
        screen = {
            "targets": {
                "baseline": {"model_id": "m1", "job_id": "j1"},
                "w8": {"model_id": "m2", "job_id": "j2"},
                "duplicate_job": {"model_id": "m3", "job_id": "j2"},
            }
        }
        candidates = parse_candidates(["w8", "baseline"], screen)
        self.assertEqual(list(candidates), ["w8", "baseline"])
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            parse_candidates(["w8", "w8"], screen)
        with self.assertRaisesRegex(ValueError, "job IDs must be unique"):
            parse_candidates(["w8", "duplicate_job"], screen)

    def test_seed_dataset_requires_one_exact_sample_and_fingerprint(
        self,
    ) -> None:
        expected = OrderedDict(
            [
                ("a", np.asarray([1.0, 2.0], dtype=np.float32)),
                ("b", np.asarray([True], dtype=np.bool_)),
            ]
        )
        entries = OrderedDict(
            (name, [value.copy()]) for name, value in expected.items()
        )
        # validate_seed_dataset uses the single-sample builder fingerprint.
        from scripts.build_p1_video_prefill import dataset_fingerprint

        validate_seed_dataset(entries, expected, dataset_fingerprint(expected))
        duplicated = OrderedDict(
            (name, [value.copy(), value.copy()])
            for name, value in expected.items()
        )
        with self.assertRaisesRegex(ValueError, "expected 1"):
            validate_seed_dataset(
                duplicated,
                expected,
                dataset_fingerprint(expected),
            )
        entries["a"][0][0] = 9.0
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            validate_seed_dataset(
                entries,
                expected,
                dataset_fingerprint(expected),
            )

    def test_chunk0_static_archive_is_byte_bound_to_seed_inputs(self) -> None:
        static = OrderedDict(
            [("inputs_embeds", np.asarray([1.0, 2.0], dtype=np.float32))]
        )
        inputs = OrderedDict(
            [
                ("past_key_0_in", np.zeros((1,), dtype=np.float32)),
                ("inputs_embeds", static["inputs_embeds"].copy()),
            ]
        )
        validate_chunk0_static_binding(inputs, static)
        inputs["inputs_embeds"][0] = 9.0
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_chunk0_static_binding(inputs, static)

    def test_state_resume_binds_manifest_screen_candidate_and_ground_truth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            artifact.mkdir()
            input_manifest_path = artifact / "input_manifest.json"
            input_manifest_path.write_text("{}\n", encoding="utf-8")
            screen_path = root / "screen.json"
            screen = {
                "input": {"dataset_fingerprint_sha256": "fingerprint"},
                "targets": {
                    "baseline": {
                        "job_id": "j1",
                        "job_name": "name",
                        "job_url": "url",
                        "model_id": "m1",
                        "dataset_id": "d1",
                        "status": "SUCCESS",
                    }
                },
            }
            screen_path.write_text(json.dumps(screen), encoding="utf-8")
            artifact_manifest = {
                "dataset_fingerprint_sha256": "fingerprint",
                "source": {
                    "vision": {
                        "vision_output_kind": "qai_hub_inference_h5",
                        "vision_source_label": "npu_scene",
                    }
                },
                "all_chunks": [
                    {"valid_tokens": 128},
                    {"valid_tokens": 128},
                    {"valid_tokens": 104},
                ],
            }
            candidates = OrderedDict(
                [
                    (
                        "baseline",
                        {"model_id": "m1", "chunk0_job_id": "j1"},
                    )
                ]
            )
            ground_truth = {"expected_choice": "A"}
            state_path = root / "chain.json"
            state = initialize_state(
                state_path,
                artifact,
                artifact_manifest,
                screen_path,
                screen,
                candidates,
                ground_truth,
            )
            self.assertEqual(state["candidate_order"], ["baseline"])
            initialize_state(
                state_path,
                artifact,
                artifact_manifest,
                screen_path,
                screen,
                candidates,
                ground_truth,
            )
            input_manifest_path.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input manifest SHA"):
                initialize_state(
                    state_path,
                    artifact,
                    artifact_manifest,
                    screen_path,
                    screen,
                    candidates,
                    ground_truth,
                )

    def test_paid_job_id_is_persisted_before_any_status_poll(self) -> None:
        class Job:
            job_id = "jpaid"
            name = "job"
            url = "url"
            model = SimpleNamespace(model_id="m1")

            def get_status(self):
                raise AssertionError("submit_job must not poll status")

        class Hub:
            @staticmethod
            def upload_dataset(entries):
                return SimpleNamespace(dataset_id="d1")

            @staticmethod
            def submit_inference_job(*args, **kwargs):
                return Job()

            @staticmethod
            def Device(*args, **kwargs):
                return object()

        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "chain.json"
            state = {"jobs": {}}
            entries = OrderedDict(
                [("x", [np.asarray([1.0], dtype=np.float32)])]
            )
            submit_job(
                Hub,
                state,
                state_path,
                key="chunk1:p1:baseline",
                model=SimpleNamespace(model_id="m1"),
                graph=P1_GRAPH_NAME,
                entries=entries,
                batch_order=["baseline"],
                name="job",
            )
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["jobs"]["chunk1:p1:baseline"]["job_id"],
                "jpaid",
            )
            self.assertEqual(
                persisted["jobs"]["chunk1:p1:baseline"]["status"],
                "SUBMITTED_UNPOLLED",
            )


@unittest.skipUnless(HAS_H5PY, "h5py is required")
class CacheAndH5ContractTests(unittest.TestCase):
    def _write_cache_output(self, path: Path, fill: float) -> None:
        import h5py

        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            for order, (name, shape) in enumerate(
                [
                    ("past_key_0_out", (8, 1, 128, 128)),
                    ("past_value_0_out", (8, 1, 128, 128)),
                ]
            ):
                group = data.create_group(str(order))
                group.attrs.update(
                    order=order,
                    name=name,
                    batch_count=1,
                )
                group.create_dataset(
                    "batch_0",
                    data=np.full(shape, fill, dtype=np.float32),
                )

    def test_cache_is_left_aligned_in_chronological_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.h5"
            second = root / "second.h5"
            self._write_cache_output(first, 1.0)
            self._write_cache_output(second, 2.0)
            state = {
                "jobs": {
                    "chunk0:p1:c": {
                        "status": "SUCCESS",
                        "output_h5": first.name,
                        "batch_order": ["c"],
                    },
                    "chunk1:p1:c": {
                        "status": "SUCCESS",
                        "output_h5": second.name,
                        "batch_order": ["c"],
                    },
                }
            }
            cache = cache_inputs(
                state,
                root,
                shard="p1",
                candidate="c",
                chunk_index=2,
                first_layer=0,
                last_layer=0,
                valid_tokens=[128, 128, 104],
            )
            key = cache["past_key_0_in"]
            value = cache["past_value_0_in"]
            np.testing.assert_array_equal(key[..., :128], 1.0)
            np.testing.assert_array_equal(key[..., 128:256], 2.0)
            np.testing.assert_array_equal(key[..., 256:], 0.0)
            np.testing.assert_array_equal(value[:, :, :128, :], 1.0)
            np.testing.assert_array_equal(value[:, :, 128:256, :], 2.0)
            np.testing.assert_array_equal(value[:, :, 256:, :], 0.0)

    def test_whole_h5_contract_enforces_batch_count_dtype_shape(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "output.h5"
            spec = SimpleNamespace(
                name="hidden",
                dtype="float32",
                shape=(1, 2),
            )
            model = SimpleNamespace(
                model_id="m",
                output_spec={"graph": [spec]},
            )
            with h5py.File(path, "w") as handle:
                data = handle.create_group("data")
                group = data.create_group("0")
                group.attrs.update(order=0, name="hidden", batch_count=1)
                group.create_dataset(
                    "batch_0",
                    data=np.zeros((1, 2), dtype=np.float32),
                )
            validate_output_h5(path, model, "graph", 1)
            with self.assertRaisesRegex(ValueError, "batch_count"):
                validate_output_h5(path, model, "graph", 2)


class GroundTruthBindingTests(unittest.TestCase):
    def test_vision_selection_is_bound_to_video_pairs(self) -> None:
        manifest_sha = "a" * 64
        pixel_hashes = ["b" * 64, "c" * 64]
        video_manifest = {
            "pairs": [
                {
                    "pair_index": pair_index,
                    "pixel_values": {"sha256": pixel_hash},
                }
                for pair_index, pixel_hash in enumerate(pixel_hashes)
            ]
        }
        source = {
            "vision": {
                "video_manifest_sha256": manifest_sha,
                "selected_indices": [4, 7],
                "selected_cases": [
                    {
                        "index": selected_index,
                        "pair_index": pair_index,
                        "manifest_sha256": manifest_sha,
                        "pixel_sha256": pixel_hash,
                    }
                    for pair_index, (selected_index, pixel_hash) in enumerate(
                        zip([4, 7], pixel_hashes, strict=True)
                    )
                ],
            }
        }
        self.assertEqual(
            validate_vision_scene_binding(
                source,
                video_manifest,
                manifest_sha,
            ),
            pixel_hashes,
        )
        source["vision"]["selected_cases"][1]["pixel_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "pixel SHA differs"):
            validate_vision_scene_binding(
                source,
                video_manifest,
                manifest_sha,
            )

    def test_same_length_wrong_prompt_ids_are_rejected(self) -> None:
        wrong = np.asarray([101, 999, 103], dtype=np.int64)
        from scripts.build_p1_video_prefill import sha256_array

        record = {
            "shape": [3],
            "dtype": "int64",
            "sha256": sha256_array(wrong),
        }
        regenerated = np.asarray([101, 102, 103], dtype=np.int64)
        with patch(
            "scripts.run_aihub_first_token_chain.tokenize_video_prompt",
            return_value=regenerated,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "exact authenticated video prompt",
            ):
                validate_frozen_prompt_ids(
                    wrong,
                    record,
                    tokenizer_path=Path("/not/read/by/mock/tokenizer.json"),
                    video_manifest={"text_chunks": [], "pairs": []},
                )

    def test_wrong_scene_parent_is_rejected_before_scoring(self) -> None:
        benchmark = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "nvidia_sdg_warehouse"
            / "benchmark.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.json"
            frames = [
                {
                    "path": f"/tmp/wrong_scene/frame_{index:04d}.png",
                    "timestamp_seconds": timestamp,
                    "sha256": "0" * 64,
                }
                for index, timestamp in zip(
                    [198, 202, 213, 217, 221, 225],
                    [
                        13.141593,
                        13.40708,
                        14.137168,
                        14.402655,
                        14.668142,
                        14.933628,
                    ],
                    strict=True,
                )
            ]
            pixel_hashes = [f"{value:x}" * 64 for value in (1, 2, 3)]
            video.write_text(
                json.dumps(
                    {
                        "frames": frames,
                        "text_chunks": [],
                        "pairs": [
                            {
                                "pair_index": pair_index,
                                "pixel_values": {"sha256": pixel_hash},
                            }
                            for pair_index, pixel_hash in enumerate(
                                pixel_hashes
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )
            from scripts.build_p1_video_prefill import sha256_file

            video_sha = sha256_file(video)
            manifest = {
                "source": {
                    "video_manifest_path": str(video),
                    "video_manifest_sha256": video_sha,
                    "vision": {
                        "video_manifest_sha256": video_sha,
                        "selected_indices": [0, 1, 2],
                        "selected_cases": [
                            {
                                "index": pair_index,
                                "pair_index": pair_index,
                                "manifest_sha256": video_sha,
                                "pixel_sha256": pixel_hash,
                            }
                            for pair_index, pixel_hash in enumerate(
                                pixel_hashes
                            )
                        ],
                    },
                }
            }
            with self.assertRaisesRegex(ValueError, "frame parent"):
                load_ground_truth(
                    benchmark,
                    case_id="video_barrier_knockdown_4fps",
                    probe_id="four_scene_event_choice",
                    artifact_dir=root,
                    artifact_manifest=manifest,
                )

    def test_final_logits_uses_authenticated_row_and_choice_tokens(self) -> None:
        logits = np.zeros((1, 128, 40), dtype=np.float32)
        logits[0, 102, 34] = 99.0
        logits[0, 100, 33] = 5.0
        scored = score_logits_tensor(
            logits,
            final_row=100,
            expected_choice="B",
            choice_token_ids={"A": 32, "B": 33, "C": 34, "D": 35},
        )
        self.assertEqual(scored["predicted_token_id"], 33)
        self.assertEqual(scored["predicted_choice"], "B")
        self.assertTrue(scored["first_token_correct"])


if __name__ == "__main__":
    unittest.main()
