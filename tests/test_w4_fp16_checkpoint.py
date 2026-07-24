from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qai_hub_models import Precision
from qai_hub_models.utils.checkpoint import CheckpointType

from scripts.make_w4_fp16_checkpoint import (
    FLOAT16_ENCODING,
    MARKER_FILENAME,
    create_w4_fp16_checkpoint,
)


def _encoding(
    name: str,
    *,
    bitwidth: int,
    parameter: bool = False,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "bw": bitwidth,
        "dtype": "INT",
        "enc_type": "PER_CHANNEL" if parameter else "PER_TENSOR",
        "scale": [0.01],
        "offset": [-1.0],
    }
    if parameter:
        entry["axis"] = 0
    return entry


def _write_checkpoint(root: Path) -> dict[str, bytes]:
    text = {
        "version": "1.0.0",
        "activation_encodings": [
            _encoding("text_hidden", bitwidth=16),
            _encoding("past_key", bitwidth=8),
        ],
        "param_encodings": [
            _encoding("proj.weight", bitwidth=4, parameter=True),
            _encoding("proj.bias", bitwidth=16),
        ],
        "quantizer_args": {"activation_bitwidth": 16},
    }
    vision = {
        "version": "1.0.0",
        "activation_encodings": [
            _encoding("image_embeddings", bitwidth=16),
        ],
        "param_encodings": [
            _encoding("patch.weight", bitwidth=8, parameter=True),
        ],
    }
    args = {
        "precision": "w4a16",
        "output_dir": "/calibration/output",
        "raw_args": [
            "--checkpoint",
            "/model",
            "--precision",
            "w4a16",
        ],
    }
    payloads = {
        "model.encodings": json.dumps(text).encode(),
        "vision_encoder.encodings": json.dumps(vision).encode(),
        "args.json": json.dumps(args).encode(),
        "model_dynamic.onnx": b"model graph",
        "model.data": b"model weights",
        "vision_encoder.onnx": b"vision graph and weights",
        "embedding_weights.raw": b"embedding table",
        "source_checkpoint.json": b'{"source_checkpoint": "/model"}\n',
        "config.json": b"{}\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "qairt_245_compat.json": b'{"schema_version": 1}\n',
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return payloads


class W4FP16CheckpointTests(unittest.TestCase):
    def test_converts_all_activations_and_preserves_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            source_payloads = _write_checkpoint(source)
            source_text = json.loads(source_payloads["model.encodings"])
            source_vision = json.loads(
                source_payloads["vision_encoder.encodings"]
            )

            create_w4_fp16_checkpoint(source, destination)

            for name, payload in source_payloads.items():
                self.assertEqual((source / name).read_bytes(), payload)

            args = json.loads((destination / "args.json").read_text())
            self.assertEqual(args["precision"], "w4")
            self.assertEqual(
                args["raw_args"],
                ["--checkpoint", "/model", "--precision", "w4"],
            )
            self.assertEqual(args["output_dir"], "/calibration/output")
            checkpoint_type = CheckpointType.from_checkpoint(destination)
            self.assertEqual(
                checkpoint_type.precision(
                    Precision.w4a16,
                    checkpoint=destination,
                ),
                Precision.w4,
            )

            for filename, original in (
                ("model.encodings", source_text),
                ("vision_encoder.encodings", source_vision),
            ):
                converted = json.loads((destination / filename).read_text())
                self.assertEqual(
                    converted["param_encodings"],
                    original["param_encodings"],
                )
                self.assertEqual(
                    [entry["name"] for entry in converted["activation_encodings"]],
                    [
                        entry["name"]
                        for entry in original["activation_encodings"]
                    ],
                )
                for entry in converted["activation_encodings"]:
                    self.assertEqual(
                        entry,
                        {"name": entry["name"], **FLOAT16_ENCODING},
                    )

            self.assertEqual(
                (destination / "model.data").read_bytes(),
                source_payloads["model.data"],
            )
            self.assertTrue(
                (destination / "qairt_245_compat.json").is_file()
            )

            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            self.assertEqual(
                marker["encoding_files"]["model.encodings"][
                    "source_activation_types"
                ],
                {"INT16": 1, "INT8": 1},
            )
            self.assertEqual(
                marker["source_file_sha256"]["model.encodings"],
                hashlib.sha256(
                    source_payloads["model.encodings"]
                ).hexdigest(),
            )
            self.assertIn("INT8", marker["weight_note"])
            self.assertEqual(marker["vision_activation_precision"], "fp16")

    def test_can_preserve_w4a16_vision_activations_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            source_payloads = _write_checkpoint(source)

            create_w4_fp16_checkpoint(
                source,
                destination,
                keep_vision_w4a16=True,
            )

            text = json.loads(
                (destination / "model.encodings").read_text()
            )
            for entry in text["activation_encodings"]:
                self.assertEqual(
                    entry,
                    {"name": entry["name"], **FLOAT16_ENCODING},
                )

            self.assertEqual(
                (destination / "vision_encoder.encodings").read_bytes(),
                source_payloads["vision_encoder.encodings"],
            )
            (source / "vision_encoder.encodings").write_bytes(b"changed")
            self.assertEqual(
                (destination / "vision_encoder.encodings").read_bytes(),
                source_payloads["vision_encoder.encodings"],
            )
            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            vision_summary = marker["encoding_files"][
                "vision_encoder.encodings"
            ]
            self.assertFalse(
                vision_summary["activations_converted_to_float16"]
            )
            self.assertEqual(
                vision_summary["destination_activation_types"],
                {"INT16": 1},
            )
            self.assertEqual(
                marker["vision_activation_precision"],
                "w4a16",
            )
            self.assertIn(
                "vision activation encodings",
                marker["unchanged"],
            )
            self.assertEqual(
                CheckpointType.from_checkpoint(destination).precision(
                    Precision.w4a16,
                    checkpoint=destination,
                ),
                Precision.w4,
            )

    def test_rejects_non_w4a16_args_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            _write_checkpoint(source)
            args_path = source / "args.json"
            args = json.loads(args_path.read_text())
            args["precision"] = "w4"
            args_path.write_text(json.dumps(args))

            with self.assertRaisesRegex(ValueError, "expected 'w4a16'"):
                create_w4_fp16_checkpoint(source, destination)
            self.assertFalse(destination.exists())

    def test_rejects_float_or_duplicate_source_activations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _write_checkpoint(source)
            path = source / "model.encodings"
            encodings = json.loads(path.read_text())
            encodings["activation_encodings"][0] = {
                "name": "text_hidden",
                **FLOAT16_ENCODING,
            }
            path.write_text(json.dumps(encodings))

            with self.assertRaisesRegex(ValueError, "not an INT8/INT16"):
                create_w4_fp16_checkpoint(source, root / "float-output")

            _write_checkpoint(source)
            encodings = json.loads(path.read_text())
            encodings["activation_encodings"].append(
                dict(encodings["activation_encodings"][0])
            )
            path.write_text(json.dumps(encodings))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                create_w4_fp16_checkpoint(source, root / "duplicate-output")

    def test_rejects_destination_nested_in_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            _write_checkpoint(source)
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                create_w4_fp16_checkpoint(source, source / "converted")


if __name__ == "__main__":
    unittest.main()
