from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.make_vision_a16_boundary_checkpoint import (
    AIMET_ENCODING_VERSION,
    ALL_FP16_MARKER_FILENAME,
    ARGS_FILENAME,
    ENCODED_GRAPH_INPUTS,
    ENCODINGS_FILENAME,
    EXPECTED_ACTIVATION_COUNT,
    EXPECTED_GRAPH_INPUTS,
    EXPECTED_GRAPH_OUTPUTS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_RESTORED_COUNT,
    FLOAT16_ENCODING,
    MARKER_FILENAME,
    MODEL_FILENAME,
    OUTPUT_PROJECTION_BY_GRAPH_OUTPUT,
    PATCH_PROJECTION_OUTPUT,
    PROVENANCE_FIELD,
    RESTORED_ACTIVATION_NAMES,
    _canonical_sha256,
    _validate_graph_contract,
    create_vision_a16_boundary_checkpoint,
)


def _a16_encoding(name: str) -> dict[str, object]:
    return {
        "name": name,
        "bw": 16,
        "dtype": "INT",
        "enc_type": "PER_TENSOR",
        "is_sym": False,
        "scale": [0.001],
        "offset": [-123.0],
    }


def _w8_encoding(index: int) -> dict[str, object]:
    return {
        "name": f"weight_{index}",
        "bw": 8,
        "dtype": "INT",
        "enc_type": "PER_CHANNEL" if index % 2 else "PER_TENSOR",
        "is_sym": True,
        "scale": [0.01],
        "offset": [-127.0],
    }


ACTIVATION_NAMES = [
    *RESTORED_ACTIVATION_NAMES,
    *[
        f"internal_activation_{index}"
        for index in range(
            EXPECTED_ACTIVATION_COUNT - EXPECTED_RESTORED_COUNT
        )
    ],
]

GRAPH_SUMMARY = {
    "graph_inputs": list(EXPECTED_GRAPH_INPUTS),
    "encoded_graph_inputs": list(ENCODED_GRAPH_INPUTS),
    "patch_projection_output": PATCH_PROJECTION_OUTPUT,
    "graph_outputs": list(EXPECTED_GRAPH_OUTPUTS),
    "output_projection_by_graph_output": (
        OUTPUT_PROJECTION_BY_GRAPH_OUTPUT
    ),
    "restored_activation_names": list(RESTORED_ACTIVATION_NAMES),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_args(precision: str) -> dict[str, object]:
    return {
        "checkpoint": "/models/Cosmos-Reason2-2B",
        "context_length": 512,
        "calibration_sequence_length": 128,
        "image_size": [224, 384],
        "num_samples": 20,
        "output_dir": "/calibration/output",
        "precision": precision,
        "raw_args": [
            "--checkpoint",
            "/models/Cosmos-Reason2-2B",
            "--precision",
            precision,
        ],
        "skip_llm": True,
        "skip_veg": False,
        "veg_num_samples": 20,
        "veg_paired_calibration_manifest": (
            "/calibration/paired_calibration_manifest.json"
        ),
        "veg_paired_calibration_manifest_sha256": "a" * 64,
        "vision_calibration_source": "paired_frame_manifest",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_checkpoint_pair(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    a16 = root / "w4a16"
    all_fp16 = root / "all-fp16"
    a16.mkdir()
    all_fp16.mkdir()

    parameters = [
        _w8_encoding(index) for index in range(EXPECTED_PARAMETER_COUNT)
    ]
    metadata = {
        "version": AIMET_ENCODING_VERSION,
        "quantizer_args": {
            "activation_bitwidth": 16,
            "param_bitwidth": 8,
        },
    }
    a16_encodings = {
        **metadata,
        "activation_encodings": [
            _a16_encoding(name) for name in ACTIVATION_NAMES
        ],
        "param_encodings": parameters,
    }
    all_fp16_encodings = {
        **metadata,
        "activation_encodings": [
            {"name": name, **FLOAT16_ENCODING}
            for name in ACTIVATION_NAMES
        ],
        "param_encodings": copy.deepcopy(parameters),
    }

    _write_json(a16 / ARGS_FILENAME, _checkpoint_args("w4a16"))
    _write_json(a16 / ENCODINGS_FILENAME, a16_encodings)
    (a16 / MODEL_FILENAME).write_bytes(b"identical vision ONNX")

    _write_json(all_fp16 / ARGS_FILENAME, _checkpoint_args("w4"))
    _write_json(all_fp16 / ENCODINGS_FILENAME, all_fp16_encodings)
    (all_fp16 / MODEL_FILENAME).write_bytes(b"identical vision ONNX")
    required_payloads = {
        "model.encodings": b"text encodings",
        "model_dynamic.onnx": b"text graph",
        "model.data": b"text weights",
        "embedding_weights.raw": b"embedding weights",
        "source_checkpoint.json": b'{"source": "fixture"}\n',
        "config.json": b"{}\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
    }
    for name, payload in required_payloads.items():
        (all_fp16 / name).write_bytes(payload)

    marker = {
        "schema_version": 1,
        "source_checkpoint": str(a16.resolve()),
        "source_file_sha256": {
            ARGS_FILENAME: _sha256(a16 / ARGS_FILENAME),
            ENCODINGS_FILENAME: _sha256(a16 / ENCODINGS_FILENAME),
        },
        "precision": {
            "source_precision": "w4a16",
            "destination_precision": "w4",
        },
        "vision_activation_precision": "fp16",
        "encoding_files": {
            ENCODINGS_FILENAME: {
                "activation_count": EXPECTED_ACTIVATION_COUNT,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "activations_converted_to_float16": True,
                "destination_activation_types": {
                    "FLOAT16": EXPECTED_ACTIVATION_COUNT
                },
                "parameter_encodings_sha256": _canonical_sha256(
                    parameters
                ),
            }
        },
    }
    _write_json(all_fp16 / ALL_FP16_MARKER_FILENAME, marker)
    return all_fp16, a16


class VisionA16BoundaryCheckpointTests(unittest.TestCase):
    @mock.patch(
        "scripts.make_vision_a16_boundary_checkpoint."
        "_validate_graph_contract",
        return_value=GRAPH_SUMMARY,
    )
    def test_restores_only_nine_boundaries_and_preserves_w4_w8(
        self,
        _graph: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            destination = root / "hybrid"
            all_before = {
                path.name: path.read_bytes()
                for path in all_fp16.iterdir()
                if path.is_file()
            }
            a16_before = {
                path.name: path.read_bytes()
                for path in a16.iterdir()
                if path.is_file()
            }

            create_vision_a16_boundary_checkpoint(
                all_fp16,
                a16,
                destination,
            )

            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in all_fp16.iterdir()
                    if path.is_file()
                },
                all_before,
            )
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in a16.iterdir()
                    if path.is_file()
                },
                a16_before,
            )

            all_encodings = json.loads(
                (all_fp16 / ENCODINGS_FILENAME).read_text()
            )
            a16_encodings = json.loads(
                (a16 / ENCODINGS_FILENAME).read_text()
            )
            output = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            output_by_name = {
                entry["name"]: entry
                for entry in output["activation_encodings"]
            }
            source_a16_by_name = {
                entry["name"]: entry
                for entry in a16_encodings["activation_encodings"]
            }
            source_float_by_name = {
                entry["name"]: entry
                for entry in all_encodings["activation_encodings"]
            }
            self.assertEqual(
                [
                    name
                    for name, entry in output_by_name.items()
                    if entry.get("dtype") == "INT"
                ],
                list(RESTORED_ACTIVATION_NAMES),
            )
            for name in RESTORED_ACTIVATION_NAMES:
                self.assertEqual(
                    output_by_name[name],
                    source_a16_by_name[name],
                )
            internal_names = (
                set(output_by_name) - set(RESTORED_ACTIVATION_NAMES)
            )
            self.assertEqual(len(internal_names), 925)
            for name in internal_names:
                self.assertEqual(
                    output_by_name[name],
                    source_float_by_name[name],
                )
                self.assertEqual(
                    output_by_name[name],
                    {"name": name, **FLOAT16_ENCODING},
                )
            self.assertEqual(
                output["param_encodings"],
                all_encodings["param_encodings"],
            )

            args = json.loads((destination / ARGS_FILENAME).read_text())
            self.assertEqual(args["precision"], "w4")
            self.assertEqual(
                args["raw_args"][-1],
                "w4",
            )
            self.assertEqual(
                args[PROVENANCE_FIELD]["restored_a16_count"],
                EXPECTED_RESTORED_COUNT,
            )
            marker = json.loads((destination / MARKER_FILENAME).read_text())
            self.assertEqual(
                marker["counts"]["remaining_float16_count"],
                925,
            )
            self.assertEqual(
                marker["precision_contract"]["vision_parameters"],
                "W8",
            )
            self.assertEqual(
                marker["hashes"]["all_fp16_onnx_sha256"],
                marker["hashes"]["a16_source_onnx_sha256"],
            )
            self.assertEqual(
                marker["hashes"][
                    "parameter_encodings_canonical_sha256"
                ],
                _canonical_sha256(all_encodings["param_encodings"]),
            )
            self.assertEqual(
                (
                    destination / ALL_FP16_MARKER_FILENAME
                ).read_bytes(),
                all_before[ALL_FP16_MARKER_FILENAME],
            )
            self.assertEqual(
                (destination / "model.data").read_bytes(),
                all_before["model.data"],
            )

    def test_rejects_mismatched_onnx_bytes_before_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            destination = root / "hybrid"
            (a16 / MODEL_FILENAME).write_bytes(b"different ONNX")

            with self.assertRaisesRegex(ValueError, "byte hashes"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    destination,
                )
            self.assertFalse(destination.exists())

    def test_rejects_parameter_difference_before_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            destination = root / "hybrid"
            encodings_path = all_fp16 / ENCODINGS_FILENAME
            encodings = json.loads(encodings_path.read_text())
            encodings["param_encodings"][0]["scale"] = [0.02]
            _write_json(encodings_path, encodings)

            with self.assertRaisesRegex(ValueError, "not exactly equal"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    destination,
                )
            self.assertFalse(destination.exists())

    def test_rejects_noncanonical_internal_float_or_wrong_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            path = all_fp16 / ENCODINGS_FILENAME
            encodings = json.loads(path.read_text())
            encodings["activation_encodings"][-1]["dtype"] = "INT"
            _write_json(path, encodings)
            with self.assertRaisesRegex(ValueError, "canonical FLOAT16"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    root / "bad-float",
                )

            all_fp16, a16 = _write_checkpoint_pair(root / "second")
            path = all_fp16 / ENCODINGS_FILENAME
            encodings = json.loads(path.read_text())
            encodings["activation_encodings"].pop()
            _write_json(path, encodings)
            with self.assertRaisesRegex(
                ValueError,
                "expected exactly 934 activation",
            ):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    root / "bad-count",
                )

    def test_rejects_wrong_calibration_or_unrelated_fp16_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            args_path = all_fp16 / ARGS_FILENAME
            args = json.loads(args_path.read_text())
            args["image_size"] = [384, 384]
            _write_json(args_path, args)
            with self.assertRaisesRegex(ValueError, "image_size"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    root / "wrong-size",
                )

            all_fp16, a16 = _write_checkpoint_pair(root / "second")
            marker_path = all_fp16 / ALL_FP16_MARKER_FILENAME
            marker = json.loads(marker_path.read_text())
            marker["source_checkpoint"] = str(root / "unrelated")
            _write_json(marker_path, marker)
            with self.assertRaisesRegex(ValueError, "does not identify"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    root / "wrong-source",
                )

    @mock.patch(
        "scripts.make_vision_a16_boundary_checkpoint."
        "_validate_graph_contract",
        return_value=GRAPH_SUMMARY,
    )
    def test_transactional_copy_failure_removes_partial_directory(
        self,
        _graph: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            destination = root / "hybrid"
            calls = 0

            def fail_copy(source: str, output: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated copy failure")
                return shutil.copy2(source, output)

            with self.assertRaisesRegex(OSError, "simulated copy failure"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    destination,
                    copy_function=fail_copy,
                )
            self.assertFalse(destination.exists())
            self.assertFalse(
                destination.with_name(f".{destination.name}.partial").exists()
            )

    def test_refuses_existing_destination_and_stale_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            all_fp16, a16 = _write_checkpoint_pair(root)
            destination = root / "hybrid"
            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing"):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    destination,
                )

            destination.rmdir()
            partial = destination.with_name(f".{destination.name}.partial")
            partial.mkdir()
            with self.assertRaisesRegex(
                FileExistsError,
                "Temporary destination",
            ):
                create_vision_a16_boundary_checkpoint(
                    all_fp16,
                    a16,
                    destination,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None,
        "onnx is not installed",
    )
    def test_graph_inspection_proves_exact_boundary_set(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_graph(path: Path, *, bad_output: bool = False) -> None:
                inputs = [
                    helper.make_tensor_value_info(
                        name,
                        TensorProto.FLOAT,
                        [1, 1, 1, 1],
                    )
                    for name in EXPECTED_GRAPH_INPUTS
                ]
                output_infos = [
                    helper.make_tensor_value_info(
                        name,
                        TensorProto.FLOAT,
                        [1, 1, 1, 1],
                    )
                    for name in EXPECTED_GRAPH_OUTPUTS
                ]
                nodes = [
                    helper.make_node(
                        "Conv",
                        ["pixel_values", "patch_weight", "patch_bias"],
                        [PATCH_PROJECTION_OUTPUT],
                        name="node_conv2d",
                    )
                ]
                initializers = [
                    helper.make_tensor(
                        "patch_weight",
                        TensorProto.FLOAT,
                        [1, 1, 1, 1],
                        [1.0],
                    ),
                    helper.make_tensor(
                        "patch_bias",
                        TensorProto.FLOAT,
                        [1],
                        [0.0],
                    ),
                ]
                for index, output_name in enumerate(EXPECTED_GRAPH_OUTPUTS):
                    projection = OUTPUT_PROJECTION_BY_GRAPH_OUTPUT[
                        output_name
                    ]
                    if bad_output and output_name == "image_features":
                        projection = "unexpected_output_projection"
                    weight = f"output_weight_{index}"
                    bias = f"output_bias_{index}"
                    nodes.extend(
                        [
                            helper.make_node(
                                "Conv",
                                [PATCH_PROJECTION_OUTPUT, weight, bias],
                                [projection],
                                name=f"node_{projection}",
                            ),
                            helper.make_node(
                                "Identity",
                                [projection],
                                [output_name],
                                name=f"node_{output_name}",
                            ),
                        ]
                    )
                    initializers.extend(
                        [
                            helper.make_tensor(
                                weight,
                                TensorProto.FLOAT,
                                [1, 1, 1, 1],
                                [1.0],
                            ),
                            helper.make_tensor(
                                bias,
                                TensorProto.FLOAT,
                                [1],
                                [0.0],
                            ),
                        ]
                    )
                graph = helper.make_graph(
                    nodes,
                    "vision_boundary_fixture",
                    inputs,
                    output_infos,
                    initializer=initializers,
                )
                model = helper.make_model(graph)
                onnx.save(model, path)

            good = root / "good.onnx"
            write_graph(good)
            summary = _validate_graph_contract(
                good,
                set(ACTIVATION_NAMES),
            )
            self.assertEqual(
                summary["restored_activation_names"],
                list(RESTORED_ACTIVATION_NAMES),
            )

            bad = root / "bad.onnx"
            write_graph(bad, bad_output=True)
            with self.assertRaisesRegex(
                ValueError,
                "graph output projections",
            ):
                _validate_graph_contract(
                    bad,
                    set(ACTIVATION_NAMES)
                    | {"unexpected_output_projection"},
                )


if __name__ == "__main__":
    unittest.main()
