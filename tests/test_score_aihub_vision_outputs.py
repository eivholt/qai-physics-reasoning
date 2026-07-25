from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.score_aihub_vision_outputs import (
    OUTPUT_NAMES,
    build_report,
    canonical_float32_sha256,
    load_h5_candidate,
    load_reference_outputs,
    parse_candidate_specs,
    score_outputs,
    tensor_metrics,
    write_report,
)


try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _reference_tensors() -> dict[str, np.ndarray]:
    base = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4) / 10
    return {
        name: np.ascontiguousarray(base + index)
        for index, name in enumerate(OUTPUT_NAMES)
    }


def _write_reference(
    root: Path,
    *,
    corrupt_hash: str | None = None,
    duplicate_mapping: bool = False,
) -> tuple[Path, Path, dict[str, np.ndarray]]:
    tensors = _reference_tensors()
    npz_path = root / "bf16_reference_outputs.npz"
    np.savez(npz_path, **tensors)
    cases = [
        {
            "case_id": f"000:video_npu_manifest:{'a' * 12}:pair_{index:03d}",
            "index": index,
            "manifest_sha256": "a" * 64,
            "pair_index": 0 if duplicate_mapping else index,
            "pixel_sha256": hashlib.sha256(
                f"pixel-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(3)
    ]
    outputs = {
        name: {
            "canonical_float32_sha256": canonical_float32_sha256(value),
            "shape": list(value.shape),
        }
        for name, value in tensors.items()
    }
    if corrupt_hash is not None:
        outputs[corrupt_hash]["canonical_float32_sha256"] = "f" * 64
    index = {
        "cases": cases,
        "geometry": {
            "grid_thw": [1, 2, 4],
            "image_height": 32,
            "image_width": 64,
            "output_shape": [2, 4],
        },
        "outputs": outputs,
        "reference": "adapted Cosmos-Reason2-2B BF16 vision encoder",
        "schema_version": 1,
    }
    index_path = root / "bf16_reference_outputs.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return npz_path, index_path, tensors


def _write_h5(
    path: Path,
    tensors: dict[str, np.ndarray],
    *,
    delta: float = 0.0,
    group_names: tuple[str, ...] = (
        "deepstack_visual_embeds_2",
        "image_features",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
    ),
    batch_count: int | None = None,
    nonfinite: tuple[str, int] | None = None,
    shape_mismatch: tuple[str, int] | None = None,
) -> None:
    if not HAS_H5PY:
        raise unittest.SkipTest("h5py is required")
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for numeric_key, name in enumerate(group_names):
            group = data.create_group(str(numeric_key))
            group.attrs["name"] = name
            # Deliberately make order disagree with both numeric key and name.
            group.attrs["order"] = len(group_names) - numeric_key
            count = (
                tensors[name].shape[0]
                if batch_count is None
                else batch_count
            )
            group.attrs["batch_count"] = count
            for index in range(min(count, tensors[name].shape[0])):
                value = np.asarray(tensors[name][index] + delta).copy()
                if nonfinite == (name, index):
                    value.reshape(-1)[0] = np.nan
                if shape_mismatch == (name, index):
                    value = value[:-1]
                group.create_dataset(f"batch_{index}", data=value)


class MetricAndReferenceTests(unittest.TestCase):
    def test_tensor_metrics_match_hand_computed_values(self) -> None:
        reference = np.asarray([1.0, 0.0], dtype=np.float32)
        candidate = np.asarray([0.0, 1.0], dtype=np.float32)

        metrics = tensor_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["cosine"], 0.0)
        self.assertAlmostEqual(metrics["relative_l2"], 2**0.5)
        self.assertAlmostEqual(metrics["mean_absolute_error"], 1.0)
        self.assertAlmostEqual(metrics["max_absolute_error"], 1.0)

    def test_reference_validates_hashes_shapes_and_case_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path, index_path, tensors = _write_reference(root)

            reference = load_reference_outputs(npz_path, index_path)

            self.assertEqual(reference.sample_count, 3)
            self.assertEqual(tuple(reference.tensors), OUTPUT_NAMES)
            self.assertEqual(
                reference.tensors["image_features"].shape,
                (3, 2, 4),
            )
            self.assertEqual(reference.cases[2]["pair_index"], 2)
            self.assertEqual(
                canonical_float32_sha256(
                    reference.tensors["deepstack_visual_embeds_1"]
                ),
                canonical_float32_sha256(
                    tensors["deepstack_visual_embeds_1"]
                ),
            )

    def test_reference_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            npz_path, index_path, _ = _write_reference(
                Path(temporary),
                corrupt_hash="image_features",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_reference_outputs(npz_path, index_path)

    def test_reference_rejects_duplicate_case_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            npz_path, index_path, _ = _write_reference(
                Path(temporary),
                duplicate_mapping=True,
            )
            with self.assertRaisesRegex(ValueError, "Duplicate.*mapping"):
                load_reference_outputs(npz_path, index_path)


@unittest.skipUnless(HAS_H5PY, "h5py is required")
class H5ScoringTests(unittest.TestCase):
    def test_maps_by_name_not_numeric_or_order_and_compares_candidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path, index_path, tensors = _write_reference(root)
            first_path = root / "first output.h5"
            second_path = root / "second output.h5"
            _write_h5(first_path, tensors, delta=0.0)
            _write_h5(second_path, tensors, delta=0.25)

            report = score_outputs(
                reference_npz=npz_path,
                reference_index=index_path,
                candidate_specs=[
                    f"second={second_path}",
                    f"first={first_path}",
                ],
            )

            self.assertEqual(list(report["candidates"]), ["first", "second"])
            first = report["candidates"]["first"]["tensors"]
            for name in OUTPUT_NAMES:
                self.assertAlmostEqual(
                    first[name]["metrics"]["cosine"],
                    1.0,
                )
                self.assertEqual(
                    first[name]["metrics"]["relative_l2"],
                    0.0,
                )
            pairwise = report["pairwise_candidate_comparisons"]
            self.assertEqual(len(pairwise), 1)
            self.assertEqual(pairwise[0]["reference_label"], "first")
            self.assertEqual(pairwise[0]["candidate_label"], "second")
            self.assertGreater(
                pairwise[0]["tensors"]["image_features"]["metrics"][
                    "relative_l2"
                ],
                0.0,
            )
            self.assertNotIn(str(root), json.dumps(report))

    def test_report_serialization_is_deterministic_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path, index_path, tensors = _write_reference(root)
            h5_path = root / "secret-local-folder-name.h5"
            _write_h5(h5_path, tensors)
            reference = load_reference_outputs(npz_path, index_path)
            spec = parse_candidate_specs([f"candidate={h5_path}"])[0]
            candidate = load_h5_candidate(spec, reference)
            report = build_report(reference, [candidate])
            first = root / "first.json"
            second = root / "second.json"

            write_report(report, first)
            write_report(report, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            serialized = first.read_text(encoding="utf-8")
            self.assertNotIn(str(root), serialized)
            self.assertNotIn(h5_path.name, serialized)
            json.loads(serialized)

    def test_rejects_duplicate_or_unknown_h5_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path, index_path, tensors = _write_reference(root)
            reference = load_reference_outputs(npz_path, index_path)
            duplicate_path = root / "duplicate.h5"
            names = (
                "image_features",
                "image_features",
                "deepstack_visual_embeds_0",
                "deepstack_visual_embeds_1",
            )
            _write_h5(duplicate_path, tensors, group_names=names)
            spec = parse_candidate_specs([f"duplicate={duplicate_path}"])[0]

            with self.assertRaisesRegex(ValueError, "duplicate H5 tensor name"):
                load_h5_candidate(spec, reference)

    def test_rejects_batch_count_shape_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path, index_path, tensors = _write_reference(root)
            reference = load_reference_outputs(npz_path, index_path)

            short_path = root / "short.h5"
            _write_h5(short_path, tensors, batch_count=2)
            short_spec = parse_candidate_specs([f"short={short_path}"])[0]
            with self.assertRaisesRegex(ValueError, "batch_count.*expected"):
                load_h5_candidate(short_spec, reference)

            wrong_shape_path = root / "wrong-shape.h5"
            _write_h5(
                wrong_shape_path,
                tensors,
                shape_mismatch=("image_features", 1),
            )
            wrong_shape_spec = parse_candidate_specs(
                [f"shape={wrong_shape_path}"]
            )[0]
            with self.assertRaisesRegex(ValueError, "shape.*expected"):
                load_h5_candidate(wrong_shape_spec, reference)

            nonfinite_path = root / "nonfinite.h5"
            _write_h5(
                nonfinite_path,
                tensors,
                nonfinite=("deepstack_visual_embeds_2", 0),
            )
            nonfinite_spec = parse_candidate_specs(
                [f"nonfinite={nonfinite_path}"]
            )[0]
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_h5_candidate(nonfinite_spec, reference)

    def test_rejects_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, tensors = _write_reference(root)
            first = root / "first.h5"
            second = root / "second.h5"
            _write_h5(first, tensors)
            _write_h5(second, tensors)

            with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
                parse_candidate_specs(
                    [f"same={first}", f"same={second}"]
                )


if __name__ == "__main__":
    unittest.main()
