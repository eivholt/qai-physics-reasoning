from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.make_vision_a16_boundary_checkpoint import (
    ALL_FP16_MARKER_FILENAME,
    MARKER_FILENAME as BOUNDARY_MARKER_FILENAME,
    PROVENANCE_FIELD as BOUNDARY_PROVENANCE_FIELD,
    RESTORED_ACTIVATION_NAMES,
)
from scripts.make_vision_fp16_weight_boundary_checkpoint import (
    ARGS_FILENAME,
    DESTINATION_EXPORT_PRECISION,
    ENCODINGS_FILENAME,
    EXPECTED_ACTIVATION_COUNT,
    EXPECTED_A16_BOUNDARY_COUNT,
    EXPECTED_FP16_INTERNAL_COUNT,
    EXPECTED_PARAMETER_COUNT,
    FLOAT16_ENCODING,
    MARKER_FILENAME,
    MODEL_FILENAME,
    PROVENANCE_FIELD,
    SOURCE_BOUNDARY_EXPORT_PRECISION,
    SOURCE_WEIGHT_EXPORT_PRECISION,
    WEIGHT_MARKER_FILENAME,
    WEIGHT_PROVENANCE_FIELD,
    _canonical_sha256,
    create_vision_fp16_weight_boundary_checkpoint,
)


GRAPH_SUMMARY = {
    "graph_inputs": [
        "pixel_values",
        "position_ids_cos",
        "position_ids_sin",
        "window_attention_mask",
        "full_attention_mask",
    ],
    "encoded_graph_inputs": [
        "pixel_values",
        "position_ids_cos",
        "position_ids_sin",
        "full_attention_mask",
    ],
    "patch_projection_output": "conv2d",
    "graph_outputs": [
        "image_features",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2",
    ],
    "output_projection_by_graph_output": {
        "image_features": "conv2d_152",
        "deepstack_visual_embeds_0": "conv2d_38",
        "deepstack_visual_embeds_1": "conv2d_76",
        "deepstack_visual_embeds_2": "conv2d_114",
    },
    "restored_activation_names": list(RESTORED_ACTIVATION_NAMES),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _a16_encoding(name: str) -> dict[str, object]:
    return {
        "name": name,
        "bw": 16,
        "dtype": "INT",
        "enc_type": "PER_TENSOR",
        "is_sym": False,
        "scale": [0.01],
        "offset": [-32768.0],
    }


def _w8_encoding(index: int) -> dict[str, object]:
    return {
        "name": f"parameter_{index:04d}",
        "bw": 8,
        "dtype": "INT",
        "enc_type": "PER_CHANNEL",
        "is_sym": True,
        "scale": [0.01, 0.02],
        "offset": [-128.0, -128.0],
    }


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


def _write_required_artifacts(
    checkpoint: Path,
    *,
    prefix: bytes,
) -> None:
    payloads = {
        MODEL_FILENAME: b"identical raw vision graph and weights",
        "model.encodings": prefix + b" text encodings",
        "model_dynamic.onnx": prefix + b" text graph",
        "model.data": prefix + b" text weights",
        "embedding_weights.raw": prefix + b" embedding weights",
        "source_checkpoint.json": prefix + b" source checkpoint",
        "config.json": prefix + b" config",
        "preprocessor_config.json": prefix + b" preprocessor",
        "tokenizer.json": prefix + b" tokenizer",
    }
    for name, payload in payloads.items():
        (checkpoint / name).write_bytes(payload)


def _write_source_pair(
    root: Path,
) -> tuple[Path, Path, dict[str, bytes], dict[str, bytes]]:
    weight = root / "fp16-weight"
    boundary = root / "a16-boundary"
    weight.mkdir()
    boundary.mkdir()

    internal_names = [
        f"internal_activation_{index:04d}"
        for index in range(EXPECTED_FP16_INTERNAL_COUNT)
    ]
    activation_names = [
        *RESTORED_ACTIVATION_NAMES,
        *internal_names,
    ]
    weight_activations = [
        _a16_encoding(name) for name in activation_names
    ]
    boundary_activations = [
        *(_a16_encoding(name) for name in RESTORED_ACTIVATION_NAMES),
        *(
            {"name": name, **FLOAT16_ENCODING}
            for name in internal_names
        ),
    ]
    w8_parameters = [
        _w8_encoding(index) for index in range(EXPECTED_PARAMETER_COUNT)
    ]
    fp16_parameters = [
        {
            "name": entry["name"],
            **FLOAT16_ENCODING,
        }
        for entry in w8_parameters
    ]
    metadata = {
        "version": "1.0.0",
        "producer": {"name": "AIMET", "version": "2.20.0"},
        "quantizer_args": {
            "activation_bitwidth": 16,
            "param_bitwidth": 8,
        },
    }
    weight_encodings = {
        **metadata,
        "activation_encodings": weight_activations,
        "param_encodings": fp16_parameters,
    }
    boundary_encodings = {
        **metadata,
        "activation_encodings": boundary_activations,
        "param_encodings": w8_parameters,
    }
    _write_json(weight / ENCODINGS_FILENAME, weight_encodings)
    _write_json(boundary / ENCODINGS_FILENAME, boundary_encodings)
    (weight / MODEL_FILENAME).write_bytes(
        b"identical raw vision graph and weights"
    )
    _write_required_artifacts(boundary, prefix=b"boundary")
    model_hash = _sha256(weight / MODEL_FILENAME)
    weight_encodings_hash = _sha256(weight / ENCODINGS_FILENAME)
    boundary_encodings_hash = _sha256(boundary / ENCODINGS_FILENAME)

    weight_args = _checkpoint_args(SOURCE_WEIGHT_EXPORT_PRECISION)
    weight_args[WEIGHT_PROVENANCE_FIELD] = {
        "activation_encoding_count": EXPECTED_ACTIVATION_COUNT,
        "activation_quantization": "A16 integer",
        "destination_parameter_quantization": "FLOAT16",
        "image_size": [224, 384],
        "marker": WEIGHT_MARKER_FILENAME,
        "output_encodings_sha256": weight_encodings_hash,
        "parameter_encoding_count": EXPECTED_PARAMETER_COUNT,
        "scheme": "FLOAT16 vision parameters with A16 integer activations",
        "source_checkpoint_name": "w4a16",
        "source_encodings_sha256": "b" * 64,
        "source_model_sha256": model_hash,
        "source_parameter_quantization": "W8 integer",
    }
    _write_json(weight / ARGS_FILENAME, weight_args)
    weight_args_hash = _sha256(weight / ARGS_FILENAME)
    weight_marker = {
        "schema_version": 1,
        "purpose": "weight fixture",
        "source_checkpoint": "/checkpoints/w4a16",
        "image_size": [224, 384],
        "counts": {
            "activation_encodings": EXPECTED_ACTIVATION_COUNT,
            "parameters_converted_int8_to_float16": (
                EXPECTED_PARAMETER_COUNT
            ),
        },
        "hashes": {
            "source_args_sha256": "c" * 64,
            "output_args_sha256": weight_args_hash,
            "source_encodings_sha256": "d" * 64,
            "output_encodings_sha256": weight_encodings_hash,
            "source_model_sha256": model_hash,
            "output_model_sha256": model_hash,
            "activation_encodings_canonical_sha256": _canonical_sha256(
                weight_activations
            ),
            "parameter_names_canonical_sha256": _canonical_sha256(
                [entry["name"] for entry in fp16_parameters]
            ),
            "top_level_metadata_canonical_sha256": _canonical_sha256(
                metadata
            ),
        },
        "encoding_summary": {},
        "replacement_parameter_encoding": FLOAT16_ENCODING,
        "exact_unchanged_fields": ["fixture"],
    }
    _write_json(weight / WEIGHT_MARKER_FILENAME, weight_marker)

    (boundary / ALL_FP16_MARKER_FILENAME).write_bytes(
        b'{"source": "all-fp16"}\n'
    )
    boundary_args = _checkpoint_args(SOURCE_BOUNDARY_EXPORT_PRECISION)
    boundary_args[BOUNDARY_PROVENANCE_FIELD] = {
        "schema_version": 1,
        "scheme": "boundary fixture",
        "marker": BOUNDARY_MARKER_FILENAME,
        "a16_source_checkpoint_name": "w4a16",
        "all_fp16_source_checkpoint_name": "all-fp16",
        "image_size": [224, 384],
        "restored_a16_count": EXPECTED_A16_BOUNDARY_COUNT,
        "remaining_float16_count": EXPECTED_FP16_INTERNAL_COUNT,
        "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
        "a16_source_encodings_sha256": "e" * 64,
        "all_fp16_source_encodings_sha256": "f" * 64,
        "output_encodings_sha256": boundary_encodings_hash,
        "vision_onnx_sha256": model_hash,
    }
    _write_json(boundary / ARGS_FILENAME, boundary_args)
    boundary_args_hash = _sha256(boundary / ARGS_FILENAME)
    calibration = {
        name: boundary_args[name]
        for name in (
            "image_size",
            "vision_calibration_source",
            "veg_paired_calibration_manifest",
            "veg_paired_calibration_manifest_sha256",
            "veg_num_samples",
            "num_samples",
        )
    }
    boundary_marker = {
        "schema_version": 1,
        "purpose": "boundary fixture",
        "sources": {
            "all_fp16_checkpoint": "/checkpoints/all-fp16",
            "w4a16_checkpoint": "/checkpoints/w4a16",
        },
        "precision_contract": {
            "checkpoint_export_precision": (
                SOURCE_BOUNDARY_EXPORT_PRECISION
            ),
            "text_weights": "W4",
            "vision_parameters": "W8",
            "restored_boundaries": "A16",
            "remaining_vision_activations": "FLOAT16",
        },
        "calibration": calibration,
        "counts": {
            "activation_count": EXPECTED_ACTIVATION_COUNT,
            "restored_a16_count": EXPECTED_A16_BOUNDARY_COUNT,
            "remaining_float16_count": EXPECTED_FP16_INTERNAL_COUNT,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
        },
        "graph_contract": GRAPH_SUMMARY,
        "hashes": {
            "all_fp16_args_sha256": "1" * 64,
            "output_args_sha256": boundary_args_hash,
            "a16_source_args_sha256": "2" * 64,
            "all_fp16_encodings_sha256": "3" * 64,
            "a16_source_encodings_sha256": "4" * 64,
            "output_encodings_sha256": boundary_encodings_hash,
            "all_fp16_marker_sha256": _sha256(
                boundary / ALL_FP16_MARKER_FILENAME
            ),
            "all_fp16_onnx_sha256": model_hash,
            "a16_source_onnx_sha256": model_hash,
            "output_onnx_sha256": model_hash,
            "parameter_encodings_canonical_sha256": _canonical_sha256(
                w8_parameters
            ),
        },
        "exact_unchanged_fields": ["fixture"],
        "restored_from_w4a16_byte_for_byte": list(
            RESTORED_ACTIVATION_NAMES
        ),
    }
    _write_json(boundary / BOUNDARY_MARKER_FILENAME, boundary_marker)

    weight_payloads = {
        path.name: path.read_bytes()
        for path in weight.iterdir()
        if path.is_file()
    }
    boundary_payloads = {
        path.name: path.read_bytes()
        for path in boundary.iterdir()
        if path.is_file()
    }
    return weight, boundary, weight_payloads, boundary_payloads


class VisionFP16WeightBoundaryCheckpointTests(unittest.TestCase):
    @mock.patch(
        "scripts.make_vision_fp16_weight_boundary_checkpoint."
        "_validate_graph_contract",
        return_value=GRAPH_SUMMARY,
    )
    def test_composes_sources_and_preserves_linkable_exporter_contract(
        self,
        _graph: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight, boundary, weight_before, boundary_before = (
                _write_source_pair(root)
            )
            destination = root / "combined"

            create_vision_fp16_weight_boundary_checkpoint(
                weight,
                boundary,
                destination,
            )

            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in weight.iterdir()
                    if path.is_file()
                },
                weight_before,
            )
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in boundary.iterdir()
                    if path.is_file()
                },
                boundary_before,
            )
            weight_encodings = json.loads(
                weight_before[ENCODINGS_FILENAME]
            )
            boundary_encodings = json.loads(
                boundary_before[ENCODINGS_FILENAME]
            )
            output_encodings = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            self.assertEqual(
                output_encodings["activation_encodings"],
                boundary_encodings["activation_encodings"],
            )
            self.assertEqual(
                output_encodings["param_encodings"],
                weight_encodings["param_encodings"],
            )
            self.assertEqual(
                [
                    entry
                    for entry in output_encodings["param_encodings"]
                    if entry["dtype"] == "FLOAT" and entry["bw"] == 16
                ],
                output_encodings["param_encodings"],
            )

            output_args = json.loads(
                (destination / ARGS_FILENAME).read_text()
            )
            weight_args = json.loads(weight_before[ARGS_FILENAME])
            self.assertEqual(
                output_args["precision"],
                DESTINATION_EXPORT_PRECISION,
            )
            self.assertEqual(
                output_args["raw_args"],
                weight_args["raw_args"],
            )
            self.assertEqual(
                output_args[WEIGHT_PROVENANCE_FIELD],
                weight_args[WEIGHT_PROVENANCE_FIELD],
            )
            provenance = output_args[PROVENANCE_FIELD]
            self.assertEqual(
                provenance["checkpoint_export_precision"],
                SOURCE_WEIGHT_EXPORT_PRECISION,
            )
            self.assertEqual(
                provenance["restored_a16_count"],
                EXPECTED_A16_BOUNDARY_COUNT,
            )

            self.assertEqual(
                (destination / MODEL_FILENAME).read_bytes(),
                weight_before[MODEL_FILENAME],
            )
            self.assertTrue(
                os.path.samefile(
                    boundary / MODEL_FILENAME,
                    destination / MODEL_FILENAME,
                )
            )
            for name in (
                "model.encodings",
                "model_dynamic.onnx",
                "model.data",
                WEIGHT_MARKER_FILENAME,
            ):
                self.assertEqual(
                    (destination / name).read_bytes(),
                    (
                        weight_before[name]
                        if name == WEIGHT_MARKER_FILENAME
                        else boundary_before[name]
                    ),
                )
            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            self.assertEqual(
                marker["precision_contract"][
                    "checkpoint_export_precision"
                ],
                SOURCE_WEIGHT_EXPORT_PRECISION,
            )
            self.assertEqual(
                marker["counts"],
                {
                    "activation_encodings": EXPECTED_ACTIVATION_COUNT,
                    "a16_boundaries": EXPECTED_A16_BOUNDARY_COUNT,
                    "float16_internal_activations": (
                        EXPECTED_FP16_INTERNAL_COUNT
                    ),
                    "float16_parameters": EXPECTED_PARAMETER_COUNT,
                },
            )

    @mock.patch(
        "scripts.make_vision_fp16_weight_boundary_checkpoint."
        "_validate_graph_contract",
        return_value=GRAPH_SUMMARY,
    )
    def test_rejects_layout_marker_or_onnx_drift_before_output(
        self,
        _graph: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight, boundary, _, _ = _write_source_pair(root)
            encodings_path = boundary / ENCODINGS_FILENAME
            encodings = json.loads(encodings_path.read_text())
            encodings["activation_encodings"][
                EXPECTED_A16_BOUNDARY_COUNT
            ] = _a16_encoding("internal_activation_0000")
            _write_json(encodings_path, encodings)
            with self.assertRaisesRegex(ValueError, "proven nine boundaries"):
                create_vision_fp16_weight_boundary_checkpoint(
                    weight,
                    boundary,
                    root / "bad-layout",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight, boundary, _, _ = _write_source_pair(root)
            marker_path = boundary / BOUNDARY_MARKER_FILENAME
            marker = json.loads(marker_path.read_text())
            marker["hashes"]["output_onnx_sha256"] = "0" * 64
            _write_json(marker_path, marker)
            with self.assertRaisesRegex(
                ValueError,
                "boundary source hashes differ",
            ):
                create_vision_fp16_weight_boundary_checkpoint(
                    weight,
                    boundary,
                    root / "bad-marker",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight, boundary, _, _ = _write_source_pair(root)
            (boundary / MODEL_FILENAME).write_bytes(b"different model")
            with self.assertRaisesRegex(ValueError, "ONNX byte hashes"):
                create_vision_fp16_weight_boundary_checkpoint(
                    weight,
                    boundary,
                    root / "bad-onnx",
                )

    def test_rejects_exporter_or_calibration_mismatch(self) -> None:
        mutations = {
            "precision is": lambda args: args.update({"precision": "w4"}),
            "raw_args precision": lambda args: args["raw_args"].__setitem__(
                -1, "w4"
            ),
            "veg_num_samples": lambda args: args.update(
                {"veg_num_samples": 19}
            ),
        }
        for expected_error, mutate in mutations.items():
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    weight, boundary, _, _ = _write_source_pair(root)
                    args_path = weight / ARGS_FILENAME
                    args = json.loads(args_path.read_text())
                    mutate(args)
                    _write_json(args_path, args)
                    with self.assertRaisesRegex(ValueError, expected_error):
                        create_vision_fp16_weight_boundary_checkpoint(
                            weight,
                            boundary,
                            root / "destination",
                        )
                    self.assertFalse((root / "destination").exists())

    def test_refuses_overwrite_and_cleans_failed_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight, boundary, _, _ = _write_source_pair(root)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing"):
                create_vision_fp16_weight_boundary_checkpoint(
                    weight,
                    boundary,
                    existing,
                )

            destination = root / "destination"
            calls = 0

            def fail_copy(source_path: str, output_path: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated copy failure")
                return shutil.copy2(source_path, output_path)

            with mock.patch(
                "scripts.make_vision_fp16_weight_boundary_checkpoint."
                "_validate_graph_contract",
                return_value=GRAPH_SUMMARY,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "simulated copy failure",
                ):
                    create_vision_fp16_weight_boundary_checkpoint(
                        weight,
                        boundary,
                        destination,
                        copy_function=fail_copy,
                    )
            self.assertFalse(destination.exists())
            self.assertFalse(
                destination.with_name(
                    f".{destination.name}.partial"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
