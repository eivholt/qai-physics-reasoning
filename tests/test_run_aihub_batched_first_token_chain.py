from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.build_p1_video_prefill import P1_TARGETS
from scripts.run_aihub_batched_first_token_chain import (
    ProbeRuntime,
    batch_index_for_stream,
    build_plan,
    chain_job_key,
    dependencies_for_job,
    execute,
    initialize_or_validate_state,
    isolated_cache_inputs,
    probe_key,
    stream_key,
    validate_submission_intents,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime(ordinal: int, root: Path) -> ProbeRuntime:
    case_id = f"case_{ordinal}"
    probe_id = f"probe_{ordinal}"
    key = probe_key(case_id, probe_id)
    source = {
        "genie_config_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "embedding_table_sha256": "3" * 64,
        "video_manifest_sha256": f"{ordinal + 4:x}" * 64,
        "contract": {"context_length": 512},
        "vision": {
            "vision_output_kind": "qai_hub_inference_h5",
            "vision_source_label": "paired_boundary_npu_vision",
            "vision_outputs_sha256": "8" * 64,
            "vision_index_manifest_sha256": "9" * 64,
            "qai_hub_provenance": {
                "inference_job_id": "job",
                "model_id": "model",
                "dataset_id": "dataset",
            },
        },
    }
    manifest = {
        "dataset_fingerprint_sha256": f"{ordinal:x}" * 64,
        "static_chunk_archives": [
            {"sha256": f"{index + 10:x}" * 64} for index in range(3)
        ],
        "all_chunks": [
            {"valid_tokens": 128},
            {"valid_tokens": 128},
            {"valid_tokens": 80 + ordinal},
        ],
        "source": source,
    }
    ground_truth = {
        "case_id": case_id,
        "probe_id": probe_id,
        "expected_choice": "A",
    }
    artifact_dir = root / f"artifact_{ordinal}"
    artifact_dir.mkdir()
    _write_json(artifact_dir / "input_manifest.json", manifest)
    return ProbeRuntime(
        key=key,
        case_id=case_id,
        probe_id=probe_id,
        artifact_dir=artifact_dir,
        static_chunks=tuple({} for _ in range(3)),
        artifact_manifest=manifest,
        ground_truth=ground_truth,
    )


class BatchPlanTests(unittest.TestCase):
    def test_four_probes_two_candidates_plan_exactly_fifteen_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel_path = root / "panel.json"
            benchmark_path = root / "benchmark.json"
            _write_json(panel_path, {"schema_version": 1, "panel_id": "panel"})
            _write_json(benchmark_path, {"video_cases": []})
            runtimes = tuple(_runtime(index, root) for index in range(4))
            candidates = OrderedDict(
                [
                    ("baseline_w4_fp16", P1_TARGETS["baseline_w4_fp16"]),
                    ("w8_layers_0_6", P1_TARGETS["w8_layers_0_6"]),
                ]
            )
            plan = build_plan(
                panel_path,
                benchmark_path,
                {"panel_id": "exact_disagreement_r1"},
                runtimes,
                candidates,
                "cosmos_test",
            )

            self.assertEqual(plan["planned_paid_inference_jobs"], 15)
            self.assertEqual(plan["batching"]["p1_samples_per_job"], 4)
            self.assertEqual(plan["batching"]["fixed_samples_per_job"], 8)
            expected_streams = [
                stream_key(runtime.key, candidate)
                for runtime in runtimes
                for candidate in candidates
            ]
            self.assertEqual(plan["stream_order"], expected_streams)
            self.assertEqual(len(plan["job_order"]), len(set(plan["job_order"])))
            for chunk_index in range(3):
                for candidate in candidates:
                    contract = plan["job_contracts"][
                        chain_job_key(chunk_index, "p1", candidate)
                    ]
                    self.assertEqual(
                        contract["batch_order"],
                        [
                            stream_key(runtime.key, candidate)
                            for runtime in runtimes
                        ],
                    )
                for shard in ("p2", "p3", "p4"):
                    contract = plan["job_contracts"][
                        chain_job_key(chunk_index, shard)
                    ]
                    self.assertEqual(contract["batch_order"], expected_streams)

    def test_dependencies_include_current_predecessor_and_prior_cache(self) -> None:
        plan = {"candidate_order": ["baseline", "w8"]}
        self.assertEqual(
            dependencies_for_job("chunk0:p2", plan),
            ["chunk0:p1:baseline", "chunk0:p1:w8"],
        )
        self.assertEqual(
            dependencies_for_job("chunk1:p2", plan),
            [
                "chunk1:p1:baseline",
                "chunk1:p1:w8",
                "chunk0:p2",
            ],
        )
        self.assertEqual(
            dependencies_for_job("chunk2:p4", plan),
            ["chunk2:p3", "chunk1:p4"],
        )

    def test_state_preflight_is_no_write_and_resume_is_plan_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "new-work" / "batched_chain.json"
            plan = {
                "schema_version": 1,
                "job_contracts": {},
                "job_order": [],
            }
            state = initialize_or_validate_state(
                state_path,
                plan,
                write_if_missing=False,
            )
            self.assertFalse(state_path.exists())
            self.assertFalse(state_path.parent.exists())
            self.assertEqual(state["jobs"], {})

            initialize_or_validate_state(
                state_path,
                plan,
                write_if_missing=True,
            )
            self.assertTrue(state_path.is_file())
            changed = {**plan, "job_order": ["chunk0:p2"]}
            with self.assertRaisesRegex(ValueError, "immutable plan"):
                initialize_or_validate_state(
                    state_path,
                    changed,
                    write_if_missing=False,
                )


class CacheIsolationTests(unittest.TestCase):
    def test_cache_rows_are_keyed_by_probe_candidate_and_shard(self) -> None:
        first_probe = probe_key("case_a", "probe")
        second_probe = probe_key("case_b", "probe")
        candidate = "baseline"
        order = [
            stream_key(first_probe, candidate),
            stream_key(second_probe, candidate),
        ]
        state = {
            "plan": {
                "probes": {
                    first_probe: {"valid_tokens": [2, 1, 1]},
                    second_probe: {"valid_tokens": [2, 1, 1]},
                }
            },
            "jobs": {
                "chunk0:p1:baseline": {
                    "status": "SUCCESS",
                    "output_h5": "chunk0.h5",
                    "batch_order": order,
                },
                "chunk1:p1:baseline": {
                    "status": "SUCCESS",
                    "output_h5": "chunk1.h5",
                    "batch_order": order,
                },
            },
        }

        def fake_load(
            path: Path,
            names: list[str],
            batch_index: int,
        ) -> dict[str, np.ndarray]:
            chunk = 0 if path.name == "chunk0.h5" else 1
            value = np.float32(10 * (chunk + 1) + batch_index)
            result: dict[str, np.ndarray] = {}
            for name in names:
                if "past_key" in name:
                    result[name] = np.full((1, 1, 2, 2), value, np.float32)
                else:
                    result[name] = np.full((1, 1, 2, 2), value, np.float32)
            return result

        with (
            patch(
                "scripts.run_aihub_batched_first_token_chain."
                "load_h5_tensors",
                side_effect=fake_load,
            ),
            patch.multiple(
                "scripts.run_aihub_batched_first_token_chain",
                NUM_KV_HEADS=1,
                HEAD_DIM=2,
                PREFILL_KV_LENGTH=4,
                PREFILL_AR=2,
            ),
        ):
            first = isolated_cache_inputs(
                state,
                Path("/unused"),
                probe=first_probe,
                candidate=candidate,
                shard="p1",
                chunk_index=2,
                first_layer=0,
                last_layer=0,
            )
            second = isolated_cache_inputs(
                state,
                Path("/unused"),
                probe=second_probe,
                candidate=candidate,
                shard="p1",
                chunk_index=2,
                first_layer=0,
                last_layer=0,
            )
        first_key = first["past_key_0_in"][0, 0, 0]
        second_key = second["past_key_0_in"][0, 0, 0]
        np.testing.assert_array_equal(first_key, [10, 10, 20, 0])
        np.testing.assert_array_equal(second_key, [11, 11, 21, 0])
        self.assertEqual(batch_index_for_stream({"batch_order": order}, order[1]), 1)


class PreflightAndSubmissionSafetyTests(unittest.TestCase):
    def test_execute_preflight_does_not_create_workdir_or_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel_path = root / "panel.json"
            benchmark_path = root / "benchmark.json"
            _write_json(
                panel_path,
                {"schema_version": 1, "panel_id": "panel", "probes": []},
            )
            _write_json(benchmark_path, {"video_cases": []})
            runtimes = tuple(_runtime(index, root) for index in range(4))
            work_dir = root / "must_not_exist"
            args = argparse.Namespace(
                panel_manifest=panel_path,
                work_dir=work_dir,
                benchmark_manifest=benchmark_path,
                candidate=["baseline_w4_fp16", "w8_layers_0_6"],
                name_prefix="cosmos_test",
                submit_missing=True,
                preflight_only=True,
            )
            hub = SimpleNamespace(
                upload_dataset=lambda *_args, **_kwargs: self.fail(
                    "preflight uploaded a dataset"
                ),
                submit_inference_job=lambda *_args, **_kwargs: self.fail(
                    "preflight submitted a job"
                ),
            )
            with (
                patch(
                    "scripts.run_aihub_batched_first_token_chain."
                    "load_panel_runtime",
                    return_value=(
                        {
                            "schema_version": 1,
                            "panel_id": "exact_disagreement_r1",
                        },
                        runtimes,
                    ),
                ),
                patch(
                    "scripts.run_aihub_batched_first_token_chain."
                    "validate_model_contracts",
                    return_value={},
                ),
            ):
                result = execute(args, hub=hub)
            self.assertEqual(result["preflight"], "passed")
            self.assertEqual(result["planned_paid_inference_jobs"], 15)
            self.assertFalse(work_dir.exists())

    def test_ambiguous_paid_submission_boundary_fails_closed(self) -> None:
        state = {
            "plan": {
                "job_contracts": {
                    "chunk0:p2": {
                        "model_id": "model",
                        "graph": "graph",
                        "options": "options",
                        "batch_order": ["stream"],
                    }
                }
            },
            "submission_intents": {
                "chunk0:p2": {"phase": "submitting_job"}
            },
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate paid job"):
            validate_submission_intents(state)


if __name__ == "__main__":
    unittest.main()
