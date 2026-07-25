from __future__ import annotations

import unittest
from collections import OrderedDict
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_aihub_p1_screen import (
    compare_outputs,
    parse_assignments,
    tensor_metrics,
    validate_existing_screen,
    validate_model,
    validate_reference_manifest,
    validate_resumed_jobs,
    load_h5_outputs,
    unpolled_job_record,
    validate_p1_output_h5,
    validate_uploaded_dataset,
)

HAS_H5PY = importlib.util.find_spec("h5py") is not None


class ScreenMetricTests(unittest.TestCase):
    def test_perfect_match(self) -> None:
        reference = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32)
        metrics = tensor_metrics(reference.copy(), reference)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["relative_rmse"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)

    def test_single_sample_axis_is_aligned(self) -> None:
        reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
        candidate = reference[np.newaxis, ...]
        metrics = tensor_metrics(candidate, reference)
        self.assertEqual(metrics["shape"], [1, 2])
        self.assertEqual(metrics["rmse"], 0.0)

    def test_comparison_preserves_explicit_reference_provenance(self) -> None:
        reference = OrderedDict(
            [
                ("add_15805", np.ones((1, 2, 3), dtype=np.float32)),
                ("past_key_0_out", np.ones((1, 1), dtype=np.float32)),
                ("past_value_0_out", np.ones((1, 1), dtype=np.float32)),
            ]
        )
        comparison = compare_outputs(
            OrderedDict(
                (name, value.copy()) for name, value in reference.items()
            ),
            reference,
            reference_label="host_fp32_same_npu_vision_inputs",
            visual_mask=np.asarray([[True, False]], dtype=np.bool_),
            valid_tokens=2,
        )
        self.assertEqual(
            comparison["reference_type"],
            "host_fp32_same_npu_vision_inputs",
        )
        self.assertEqual(
            comparison["aggregate"]["kv_mean_relative_rmse"],
            0.0,
        )
        self.assertEqual(
            comparison["aggregate"]["hidden_regions"]["visual"][
                "token_count"
            ],
            1,
        )
        self.assertEqual(
            comparison["aggregate"]["hidden_regions"]["text"]["token_count"],
            1,
        )


class ScreenContractTests(unittest.TestCase):
    def test_assignment_parser_rejects_unknown_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown target"):
            parse_assignments(["not_a_target=j123"])

    def test_model_contract_requires_order_shape_and_dtype(self) -> None:
        inputs = OrderedDict(
            [
                ("a", np.zeros((1, 2), dtype=np.float32)),
                ("b", np.zeros((3,), dtype=np.bool_)),
            ]
        )
        model = SimpleNamespace(
            model_id="mtest",
            input_spec={
                "ar128_cl512_1_of_4": [
                    SimpleNamespace(name="a", dtype="float32", shape=(1, 2)),
                    SimpleNamespace(name="b", dtype="bool", shape=(3,)),
                ]
            },
        )
        validate_model(model, inputs)
        wrong_order = OrderedDict(reversed(list(inputs.items())))
        with self.assertRaisesRegex(ValueError, "input contract differs"):
            validate_model(model, wrong_order)

    def test_existing_screen_reuse_rejects_fingerprint_and_condition(self) -> None:
        screen = {
            "graph_name": "ar128_cl512_1_of_4",
            "graph_options": (
                "--qnn_options context_enable_graphs=ar128_cl512_1_of_4"
            ),
            "device": {"name": "Dragonwing IQ-9075 EVK", "os": "1.7"},
            "condition_label": "wrong",
            "input": {"dataset_fingerprint_sha256": "old"},
        }
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_existing_screen(
                screen,
                {"dataset_fingerprint_sha256": "new"},
                "expected",
            )

    def test_resumed_jobs_require_exact_model_graph_device_and_dataset(
        self,
    ) -> None:
        def job(model_id: str, dataset_id: str):
            return SimpleNamespace(
                model=SimpleNamespace(model_id=model_id),
                options=(
                    "--qnn_options "
                    "context_enable_graphs=ar128_cl512_1_of_4"
                ),
                device=SimpleNamespace(
                    name="Dragonwing IQ-9075 EVK",
                    os="1.7",
                ),
                inputs=SimpleNamespace(dataset_id=dataset_id),
            )

        self.assertEqual(
            validate_resumed_jobs(
                {"a": job("m1", "d1"), "b": job("m2", "d1")},
                expected_targets={"a": "m1", "b": "m2"},
            ),
            "d1",
        )
        with self.assertRaisesRegex(ValueError, "different datasets"):
            validate_resumed_jobs(
                {"a": job("m1", "d1"), "b": job("m2", "d2")},
                expected_targets={"a": "m1", "b": "m2"},
            )

    def test_uploaded_dataset_is_byte_exact_single_sample(self) -> None:
        from scripts.build_p1_video_prefill import dataset_fingerprint

        expected = OrderedDict(
            [("x", np.asarray([1.0, 2.0], dtype=np.float32))]
        )
        validate_uploaded_dataset(
            {"x": [expected["x"].copy()]},
            expected,
            dataset_fingerprint(expected),
        )
        with self.assertRaisesRegex(ValueError, "expected 1"):
            validate_uploaded_dataset(
                {"x": [expected["x"].copy(), expected["x"].copy()]},
                expected,
                dataset_fingerprint(expected),
            )

    def test_unpolled_record_does_not_call_status(self) -> None:
        job = SimpleNamespace(
            job_id="j1",
            name="name",
            url="url",
            model=SimpleNamespace(model_id="m1"),
            inputs=SimpleNamespace(dataset_id="d1"),
            device="device",
            options="options",
            get_status=lambda: (_ for _ in ()).throw(
                AssertionError("must not poll")
            ),
        )
        self.assertEqual(
            unpolled_job_record(job, "label")["status"],
            "SUBMITTED_UNPOLLED",
        )

    def test_reference_manifest_must_match_graph_dataset_and_archive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "host.npz"
            np.savez(reference, add_15805=np.zeros((1,), dtype=np.float32))
            from scripts.build_p1_video_prefill import sha256_file

            manifest = {
                "graph_name": "ar128_cl512_1_of_4",
                "input": {"dataset_fingerprint_sha256": "fingerprint"},
                "output": {"archive_sha256": sha256_file(reference)},
            }
            reference.with_suffix(".manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            record = validate_reference_manifest(
                reference,
                {"dataset_fingerprint_sha256": "fingerprint"},
            )
            self.assertEqual(record["sha256"], sha256_file(reference))
            manifest["input"]["dataset_fingerprint_sha256"] = "wrong"
            reference.with_suffix(".manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_reference_manifest(
                    reference,
                    {"dataset_fingerprint_sha256": "fingerprint"},
                )

    @unittest.skipUnless(HAS_H5PY, "h5py is required")
    def test_h5_loader_rejects_duplicate_attr_names(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.h5"
            with h5py.File(path, "w") as handle:
                data = handle.create_group("data")
                for index in range(2):
                    group = data.create_group(str(index))
                    group.attrs["order"] = index
                    group.attrs["name"] = "same_name"
                    group.attrs["batch_count"] = 1
                    group.create_dataset(
                        "batch_0",
                        data=np.zeros((1,), dtype=np.float32),
                    )
            with self.assertRaisesRegex(ValueError, "duplicate tensor names"):
                load_h5_outputs(path)

    @unittest.skipUnless(HAS_H5PY, "h5py is required")
    def test_p1_output_h5_requires_one_exact_batch(self) -> None:
        import h5py

        specs = [
            SimpleNamespace(
                name="add_15805",
                dtype="float32",
                shape=(1, 128, 2048),
            ),
            *[
                spec
                for layer in range(7)
                for spec in (
                    SimpleNamespace(
                        name=f"past_key_{layer}_out",
                        dtype="float32",
                        shape=(8, 1, 128, 128),
                    ),
                    SimpleNamespace(
                        name=f"past_value_{layer}_out",
                        dtype="float32",
                        shape=(8, 1, 128, 128),
                    ),
                )
            ],
        ]
        model = SimpleNamespace(
            model_id="m",
            output_spec={"ar128_cl512_1_of_4": specs},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.h5"
            with h5py.File(path, "w") as handle:
                data = handle.create_group("data")
                for order, spec in enumerate(specs):
                    group = data.create_group(str(order))
                    group.attrs.update(
                        order=order,
                        name=spec.name,
                        batch_count=1,
                    )
                    group.create_dataset(
                        "batch_0",
                        data=np.zeros(spec.shape, dtype=np.float32),
                    )
            validate_p1_output_h5(path, model)
            with h5py.File(path, "r+") as handle:
                next(iter(handle["data"].values())).attrs["batch_count"] = 2
            with self.assertRaisesRegex(ValueError, "batch_count"):
                validate_p1_output_h5(path, model)


if __name__ == "__main__":
    unittest.main()
