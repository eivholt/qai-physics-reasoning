from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compare_vision_checkpoints import (
    MetricAccumulator,
    compute_metrics,
    load_input_manifests,
    resolve_geometry,
    validate_candidate_checkpoints,
)


def _write_manifest(root: Path, *, corrupt_sha256: bool = False) -> Path:
    sample_inputs = root / "sample_inputs"
    sample_inputs.mkdir(parents=True)
    pixel_path = sample_inputs / "pair_000_pixel_values.raw"
    shape = (4, 1536)
    values = [float(index % 17) / 16.0 for index in range(math.prod(shape))]
    pixel_path.write_bytes(struct.pack(f"<{len(values)}f", *values))
    digest = hashlib.sha256(pixel_path.read_bytes()).hexdigest()
    if corrupt_sha256:
        digest = "0" * 64

    manifest = {
        "schema_version": 1,
        "graph_contract": {
            "image_size": [32, 32],
            "pixel_values_shape": list(shape),
            "grid_thw": [1, 2, 2],
            "image_features_shape": [1, 2048],
        },
        "pairs": [
            {
                "pair_index": 0,
                "pixel_values": {
                    "file": "sample_inputs/pair_000_pixel_values.raw",
                    "sha256": digest,
                    "bytes": pixel_path.stat().st_size,
                    "shape": list(shape),
                    "grid_thw": [1, 2, 2],
                    "serialization": "float32 little-endian raw",
                },
            }
        ],
    }
    manifest_path = root / "video_npu_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_candidate(root: Path, image_size: list[int]) -> Path:
    root.mkdir(parents=True)
    (root / "vision_encoder.onnx").write_bytes(b"small-test-onnx")
    (root / "vision_encoder.encodings").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (root / "args.json").write_text(
        json.dumps({"image_size": image_size}) + "\n",
        encoding="utf-8",
    )
    return root


class MetricTests(unittest.TestCase):
    def test_metrics_have_expected_values(self) -> None:
        reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        candidate = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        metrics = compute_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["cosine"], 0.5)
        self.assertAlmostEqual(metrics["relative_l2"], 1.0)
        self.assertAlmostEqual(metrics["mean_abs_error"], 0.5)
        self.assertAlmostEqual(metrics["max_abs_error"], 1.0)
        self.assertAlmostEqual(metrics["norm_ratio"], 1.0)

    def test_streaming_aggregate_matches_concatenated_metrics(self) -> None:
        references = (
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([-1.0, 3.0, 4.0], dtype=np.float32),
        )
        candidates = (
            np.asarray([1.5, 1.0], dtype=np.float32),
            np.asarray([-2.0, 3.5, 4.0], dtype=np.float32),
        )
        accumulator = MetricAccumulator()
        for reference, candidate in zip(references, candidates, strict=True):
            accumulator.update(reference, candidate)

        expected = compute_metrics(
            np.concatenate(references),
            np.concatenate(candidates),
        )
        actual = accumulator.metrics()

        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            self.assertAlmostEqual(actual[name], expected[name])

    def test_metrics_reject_shape_mismatch_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "different shapes"):
            compute_metrics(np.ones((2,)), np.ones((3,)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_metrics(np.ones((2,)), np.asarray([1.0, np.nan]))


class ManifestValidationTests(unittest.TestCase):
    def test_loads_exact_float32_tensor_after_hash_and_shape_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_manifest(Path(directory))

            manifests = load_input_manifests([manifest_path])

            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            self.assertEqual(
                (manifest.geometry.image_height, manifest.geometry.image_width),
                (32, 32),
            )
            self.assertEqual(manifest.geometry.pixel_shape, (4, 1536))
            self.assertEqual(manifest.geometry.grid_thw, (1, 2, 2))
            self.assertEqual(manifest.geometry.output_shape, (1, 2048))
            self.assertEqual(len(manifest.cases), 1)
            case = manifest.cases[0]
            self.assertEqual(case.pixel_values.shape, (4, 1536))
            self.assertEqual(case.pixel_values.dtype, np.dtype("<f4"))
            self.assertFalse(case.pixel_values.flags.writeable)
            self.assertEqual(
                case.pixel_sha256,
                hashlib.sha256(case.pixel_path.read_bytes()).hexdigest(),
            )

    def test_rejects_pixel_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_manifest(
                Path(directory),
                corrupt_sha256=True,
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_input_manifests([manifest_path])

    def test_rejects_path_escape_before_reading_pixel_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_manifest(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pairs"][0]["pixel_values"]["file"] = "../outside.raw"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "escapes"):
                load_input_manifests([manifest_path])

    def test_rejects_manifest_geometry_inconsistent_with_cosmos_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_manifest(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["graph_contract"]["grid_thw"] = [1, 1, 4]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected"):
                load_input_manifests([manifest_path])

    def test_candidate_args_json_is_an_independent_geometry_cross_check(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = load_input_manifests([_write_manifest(root / "input")])
            matching = validate_candidate_checkpoints(
                [f"matching={_write_candidate(root / 'matching', [32, 32])}"]
            )
            self.assertEqual(
                resolve_geometry(manifests, matching),
                manifests[0].geometry,
            )

            conflicting = validate_candidate_checkpoints(
                [
                    "conflicting="
                    f"{_write_candidate(root / 'conflicting', [64, 32])}"
                ]
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_geometry(manifests, conflicting)


if __name__ == "__main__":
    unittest.main()
