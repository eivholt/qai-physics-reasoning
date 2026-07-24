from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_fp16_split_io_checkpoint import (
    DEFAULT_BOUNDARIES,
    FLOAT16_ENCODING,
    MARKER_FILENAME,
    create_fp16_split_io_checkpoint,
)


def _write_checkpoint(root: Path) -> bytes:
    activations = [
        {
            "name": name,
            "bw": 16,
            "dtype": "INT",
            "enc_type": "PER_TENSOR",
            "is_sym": False,
            "scale": [0.2 + index / 100],
            "offset": [-200.0 - index],
        }
        for index, name in enumerate(DEFAULT_BOUNDARIES)
    ]
    activations.append(
        {
            "name": "unrelated",
            "bw": 16,
            "dtype": "INT",
            "enc_type": "PER_TENSOR",
            "scale": [0.01],
            "offset": [-100.0],
        }
    )
    payload = json.dumps(
        {
            "version": "1.0.0",
            "activation_encodings": activations,
            "param_encodings": [],
        }
    ).encode()
    (root / "model.encodings").write_bytes(payload)
    (root / "model_dynamic.onnx").write_bytes(b"onnx")
    (root / "model.data").write_bytes(b"weights")
    return payload


class FP16SplitIOCheckpointTests(unittest.TestCase):
    def test_replaces_only_declared_boundaries_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            source_bytes = _write_checkpoint(source)

            create_fp16_split_io_checkpoint(source, destination)

            self.assertEqual((source / "model.encodings").read_bytes(), source_bytes)
            patched = json.loads((destination / "model.encodings").read_text())
            by_name = {
                entry["name"]: entry
                for entry in patched["activation_encodings"]
            }
            for name in DEFAULT_BOUNDARIES:
                self.assertEqual(by_name[name], {"name": name, **FLOAT16_ENCODING})
            self.assertEqual(by_name["unrelated"]["scale"], [0.01])

            marker = json.loads((destination / MARKER_FILENAME).read_text())
            self.assertEqual(marker["boundary_tensors"], list(DEFAULT_BOUNDARIES))
            self.assertEqual(
                marker["source_encodings_sha256"],
                hashlib.sha256(source_bytes).hexdigest(),
            )
            self.assertEqual(
                (source / "model_dynamic.onnx").stat().st_ino,
                (destination / "model_dynamic.onnx").stat().st_ino,
            )

    def test_rejects_missing_boundary_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            _write_checkpoint(source)
            with self.assertRaisesRegex(ValueError, "found 0"):
                create_fp16_split_io_checkpoint(
                    source,
                    destination,
                    boundaries=("missing",),
                )
            self.assertFalse(destination.exists())

    def test_rejects_non_int16_source_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _write_checkpoint(source)
            path = source / "model.encodings"
            encodings = json.loads(path.read_text())
            encodings["activation_encodings"][0]["dtype"] = "FLOAT"
            path.write_text(json.dumps(encodings))
            with self.assertRaisesRegex(ValueError, "not an INT16"):
                create_fp16_split_io_checkpoint(source, root / "destination")


if __name__ == "__main__":
    unittest.main()
