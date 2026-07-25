from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.make_vision_fp16_weight_checkpoint import (
    ARGS_FILENAME,
    ENCODINGS_FILENAME,
    EXPECTED_ACTIVATION_COUNT,
    EXPECTED_PARAMETER_COUNT,
    FLOAT16_ENCODING,
    MARKER_FILENAME,
    MODEL_FILENAME,
    PROVENANCE_FIELD,
    create_vision_fp16_weight_checkpoint,
)


def _int_encoding(
    name: str,
    *,
    bitwidth: int,
    per_channel: bool = False,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "bw": bitwidth,
        "dtype": "INT",
        "enc_type": "PER_CHANNEL" if per_channel else "PER_TENSOR",
        "is_sym": per_channel,
        "scale": [0.01, 0.02] if per_channel else [0.01],
        "offset": [-128.0, -128.0] if per_channel else [-32768.0],
    }
    return entry


def _write_checkpoint(root: Path) -> dict[str, bytes]:
    root.mkdir()
    activations = [
        _int_encoding(f"activation_{index:04d}", bitwidth=16)
        for index in range(EXPECTED_ACTIVATION_COUNT)
    ]
    parameters = [
        _int_encoding(
            f"parameter_{index:04d}",
            bitwidth=8,
            per_channel=index < 153,
        )
        for index in range(EXPECTED_PARAMETER_COUNT)
    ]
    encodings = {
        "version": "1.0.0",
        "activation_encodings": activations,
        "param_encodings": parameters,
        "producer": {"name": "AIMET", "version": "2.20.0"},
        "quantizer_args": {
            "activation_bitwidth": 16,
            "param_bitwidth": 8,
        },
    }
    args = {
        "precision": "w4a16",
        "image_size": [224, 384],
        "vision_calibration_source": "paired_frame_manifest",
        "veg_paired_calibration_manifest": "/calibration/pairs.json",
        "veg_paired_calibration_manifest_sha256": "a" * 64,
        "output_dir": "/original/output",
        "raw_args": ["--image-size", "224", "384"],
    }
    payloads = {
        ENCODINGS_FILENAME: json.dumps(encodings).encode(),
        ARGS_FILENAME: json.dumps(args).encode(),
        MODEL_FILENAME: b"raw vision graph and weights",
        "model.encodings": b"preserved text encodings",
        "model_dynamic.onnx": b"preserved text graph",
        "config.json": b'{"preserved": true}\n',
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return payloads


class VisionFP16WeightCheckpointTests(unittest.TestCase):
    def test_converts_only_parameters_and_records_exact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            source_payloads = _write_checkpoint(source)
            source_encodings = json.loads(
                source_payloads[ENCODINGS_FILENAME]
            )
            source_args = json.loads(source_payloads[ARGS_FILENAME])

            create_vision_fp16_weight_checkpoint(source, destination)

            for name, payload in source_payloads.items():
                self.assertEqual((source / name).read_bytes(), payload)

            converted = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            self.assertEqual(
                converted["activation_encodings"],
                source_encodings["activation_encodings"],
            )
            self.assertEqual(
                [
                    entry["name"]
                    for entry in converted["param_encodings"]
                ],
                [
                    entry["name"]
                    for entry in source_encodings["param_encodings"]
                ],
            )
            for entry in converted["param_encodings"]:
                self.assertEqual(
                    entry,
                    {"name": entry["name"], **FLOAT16_ENCODING},
                )
            for field in ("version", "producer", "quantizer_args"):
                self.assertEqual(converted[field], source_encodings[field])

            converted_args = json.loads(
                (destination / ARGS_FILENAME).read_text()
            )
            provenance = converted_args.pop(PROVENANCE_FIELD)
            self.assertEqual(converted_args, source_args)
            self.assertEqual(
                provenance["activation_encoding_count"],
                EXPECTED_ACTIVATION_COUNT,
            )
            self.assertEqual(
                provenance["parameter_encoding_count"],
                EXPECTED_PARAMETER_COUNT,
            )
            self.assertEqual(
                provenance["destination_parameter_quantization"],
                "FLOAT16",
            )

            self.assertEqual(
                (destination / MODEL_FILENAME).read_bytes(),
                source_payloads[MODEL_FILENAME],
            )
            self.assertTrue(
                os.path.samefile(
                    source / MODEL_FILENAME,
                    destination / MODEL_FILENAME,
                )
            )
            self.assertEqual(
                (destination / "model.encodings").read_bytes(),
                source_payloads["model.encodings"],
            )

            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            model_hash = hashlib.sha256(
                source_payloads[MODEL_FILENAME]
            ).hexdigest()
            self.assertEqual(
                marker["hashes"]["source_model_sha256"],
                model_hash,
            )
            self.assertEqual(
                marker["hashes"]["output_model_sha256"],
                model_hash,
            )
            self.assertEqual(
                marker["counts"][
                    "parameters_converted_int8_to_float16"
                ],
                EXPECTED_PARAMETER_COUNT,
            )
            self.assertIn(
                "activation_encodings values and ordering",
                marker["exact_unchanged_fields"],
            )

    def test_output_is_deterministic_for_same_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            _write_checkpoint(source)

            first = create_vision_fp16_weight_checkpoint(
                source, root / "first"
            )
            second = create_vision_fp16_weight_checkpoint(
                source, root / "second"
            )

            for name in (
                ENCODINGS_FILENAME,
                ARGS_FILENAME,
                MARKER_FILENAME,
            ):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )

    def test_rejects_preexisting_float_and_wrong_encoding_contract(self) -> None:
        mutations = {
            "pre-existing FLOAT parameter": lambda payload: payload[
                "param_encodings"
            ][0].update(FLOAT16_ENCODING),
            "must be W8 INT": lambda payload: payload[
                "param_encodings"
            ][0].update({"bw": 4}),
            "must be A16 INT": lambda payload: payload[
                "activation_encodings"
            ][0].update({"bw": 8}),
            "expected exactly 205 parameter": lambda payload: payload[
                "param_encodings"
            ].pop(),
            "expected exactly 934 activation": lambda payload: payload[
                "activation_encodings"
            ].pop(),
        }
        for expected_error, mutate in mutations.items():
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source"
                    destination = root / "destination"
                    _write_checkpoint(source)
                    path = source / ENCODINGS_FILENAME
                    payload = json.loads(path.read_text())
                    mutate(payload)
                    path.write_text(json.dumps(payload))

                    with self.assertRaisesRegex(
                        ValueError, expected_error
                    ):
                        create_vision_fp16_weight_checkpoint(
                            source, destination
                        )
                    self.assertFalse(destination.exists())

    def test_rejects_wrong_geometry_or_nonpaired_provenance(self) -> None:
        mutations = {
            "image_size is": lambda payload: payload.update(
                {"image_size": [384, 224]}
            ),
            "vision_calibration_source": lambda payload: payload.update(
                {"vision_calibration_source": "imagenette_class_balanced"}
            ),
            "calibration SHA-256 is invalid": lambda payload: payload.update(
                {"veg_paired_calibration_manifest_sha256": "invalid"}
            ),
        }
        for expected_error, mutate in mutations.items():
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source"
                    destination = root / "destination"
                    _write_checkpoint(source)
                    path = source / ARGS_FILENAME
                    payload = json.loads(path.read_text())
                    mutate(payload)
                    path.write_text(json.dumps(payload))

                    with self.assertRaisesRegex(
                        ValueError, expected_error
                    ):
                        create_vision_fp16_weight_checkpoint(
                            source, destination
                        )
                    self.assertFalse(destination.exists())

    def test_refuses_existing_or_nested_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            _write_checkpoint(source)
            existing = root / "existing"
            existing.mkdir()

            with self.assertRaisesRegex(FileExistsError, "refusing"):
                create_vision_fp16_weight_checkpoint(source, existing)
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                create_vision_fp16_weight_checkpoint(
                    source, source / "nested"
                )

    def test_copy_failure_removes_transactional_temporary_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)
            calls = 0

            def failing_copy(source_path: str, destination_path: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected copy failure")
                return str(
                    Path(
                        shutil.copy2(source_path, destination_path)
                    )
                )

            with self.assertRaisesRegex(OSError, "injected copy failure"):
                create_vision_fp16_weight_checkpoint(
                    source,
                    destination,
                    copy_function=failing_copy,
                )
            self.assertFalse(destination.exists())
            self.assertFalse((root / ".destination.partial").exists())


if __name__ == "__main__":
    unittest.main()
