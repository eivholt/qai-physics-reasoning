from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from scripts.build_p1_video_prefill import (
    GOLDEN_PROFILES,
    P1_FIRST_LAYER,
    P1_LAST_LAYER,
    PreparedPrompt,
    VIDEO_PAD_TOKEN_ID,
    VISION_START_TOKEN_ID,
    build_attention_mask,
    compute_interleaved_mrope,
    compute_mrope_positions,
    dataset_fingerprint,
    load_selected_vision_outputs,
    make_p1_chunk_inputs,
    sha256_array,
    sha256_file,
    zero_p1_kv_inputs,
)

HAS_H5PY = importlib.util.find_spec("h5py") is not None


def _barrier_shape_ids() -> np.ndarray:
    ids = np.full(360, 42, dtype=np.int64)
    for start in (21, 114, 207):
        ids[start] = VISION_START_TOKEN_ID
        ids[start + 1 : start + 85] = VIDEO_PAD_TOKEN_ID
    return ids


class MRoPEContractTests(unittest.TestCase):
    def test_barrier_geometry_matches_pinned_geniex_positions(self) -> None:
        ids = _barrier_shape_ids()
        positions, deltas = compute_mrope_positions(
            ids,
            [(1, 14, 24)] * 3,
            spatial_merge_size=2,
        )

        self.assertEqual(deltas, (-216, -216, -216))
        np.testing.assert_array_equal(positions.max(axis=1), [143, 143, 143])
        np.testing.assert_array_equal(
            positions[:, [21, 22, 105, 114, 115, 198, 207, 208, 291, 359]],
            [
                [21, 22, 22, 42, 43, 43, 63, 64, 64, 143],
                [21, 22, 28, 42, 43, 49, 63, 64, 70, 143],
                [21, 22, 33, 42, 43, 54, 63, 64, 75, 143],
            ],
        )
        self.assertEqual(
            sha256_array(positions),
            "0bd57a8b56370cff4fc1d7ea821828fe9e6c35b07a13ddcc295defdbec221e84",
        )

    def test_interleaved_mrope_and_padding_match_golden_profile(self) -> None:
        positions, _ = compute_mrope_positions(
            _barrier_shape_ids(),
            [(1, 14, 24)] * 3,
            spatial_merge_size=2,
        )
        cosine, sine = compute_interleaved_mrope(
            positions,
            theta=5_000_000.0,
            mrope_section=(24, 20, 20),
        )
        expected = GOLDEN_PROFILES["barrier_normal_bf16_r1"]["chunks"]
        for chunk_index, record in enumerate(expected):
            start = chunk_index * 128
            valid = int(record["valid_tokens"])
            chunk_cosine = np.ones((128, 64), dtype=np.float32)
            chunk_sine = np.zeros((128, 64), dtype=np.float32)
            chunk_cosine[:valid] = cosine[start : start + valid]
            chunk_sine[:valid] = sine[start : start + valid]
            self.assertEqual(
                sha256_array(chunk_cosine.reshape(1, 1, 128, 64)),
                record["position_ids_cos"],
            )
            self.assertEqual(
                sha256_array(chunk_sine.reshape(1, 1, 128, 64)),
                record["position_ids_sin"],
            )
        np.testing.assert_array_equal(chunk_cosine[104:], 1.0)
        np.testing.assert_array_equal(chunk_sine[104:], 0.0)

    def test_attention_and_visual_masks_match_all_three_golden_chunks(
        self,
    ) -> None:
        ids = _barrier_shape_ids()
        visual = ids == VIDEO_PAD_TOKEN_ID
        expected = GOLDEN_PROFILES["barrier_normal_bf16_r1"]["chunks"]
        for chunk_index, record in enumerate(expected):
            start = chunk_index * 128
            valid = int(record["valid_tokens"])
            attention = build_attention_mask(start, valid).reshape(
                1, 1, 128, 512
            )
            mask = np.zeros(128, dtype=np.bool_)
            mask[:valid] = visual[start : start + valid]
            mask = mask.reshape(1, 128)
            self.assertEqual(
                sha256_array(attention),
                record["attention_mask"],
            )
            self.assertEqual(
                sha256_array(mask),
                record["visual_pos_masks"],
            )
            self.assertEqual(int(mask.sum()), record["visual_tokens"])


class P1InputContractTests(unittest.TestCase):
    def test_zero_kv_layout_and_order(self) -> None:
        kv = zero_p1_kv_inputs()
        self.assertEqual(
            list(kv),
            [
                name
                for layer in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1)
                for name in (
                    f"past_key_{layer}_in",
                    f"past_value_{layer}_in",
                )
            ],
        )
        for layer in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1):
            self.assertEqual(
                kv[f"past_key_{layer}_in"].shape,
                (8, 1, 128, 384),
            )
            self.assertEqual(
                kv[f"past_value_{layer}_in"].shape,
                (8, 1, 384, 128),
            )

    def test_only_chunk_zero_may_infer_synthetic_kv(self) -> None:
        chunks = [
            OrderedDict([("inputs_embeds", np.zeros((1,), np.float32))]),
            OrderedDict([("inputs_embeds", np.ones((1,), np.float32))]),
        ]
        prepared = PreparedPrompt(
            input_ids=np.zeros(2, dtype=np.int64),
            positions=np.zeros((3, 2), dtype=np.int32),
            chunks=chunks,
            chunk_valid_tokens=[1, 1],
            chunk_visual_tokens=[0, 0],
            source_record={},
        )
        chunk_zero = make_p1_chunk_inputs(prepared, 0)
        self.assertEqual(list(chunk_zero)[-1], "inputs_embeds")
        with self.assertRaisesRegex(ValueError, "Only chunk 0"):
            make_p1_chunk_inputs(prepared, 1)

        nonzero = zero_p1_kv_inputs()
        nonzero["past_key_0_in"][0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "all-zero KV"):
            make_p1_chunk_inputs(prepared, 0, kv_inputs=nonzero)

        wrong_shape = zero_p1_kv_inputs()
        wrong_shape["past_value_0_in"] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            make_p1_chunk_inputs(prepared, 0, kv_inputs=wrong_shape)

        nonfinite = zero_p1_kv_inputs()
        nonfinite["past_key_0_in"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            make_p1_chunk_inputs(prepared, 0, kv_inputs=nonfinite)

    def test_dataset_fingerprint_includes_order_dtype_shape_and_bytes(
        self,
    ) -> None:
        first = OrderedDict(
            [
                ("a", np.asarray([1, 2], dtype=np.float32)),
                ("b", np.asarray([3], dtype=np.int32)),
            ]
        )
        same = OrderedDict((name, value.copy()) for name, value in first.items())
        reversed_order = OrderedDict(reversed(list(same.items())))
        changed_dtype = OrderedDict(
            [
                ("a", np.asarray([1, 2], dtype=np.float64)),
                ("b", np.asarray([3], dtype=np.int32)),
            ]
        )
        self.assertEqual(dataset_fingerprint(first), dataset_fingerprint(same))
        self.assertNotEqual(
            dataset_fingerprint(first),
            dataset_fingerprint(reversed_order),
        )
        self.assertNotEqual(
            dataset_fingerprint(first),
            dataset_fingerprint(changed_dtype),
        )


@unittest.skipUnless(HAS_H5PY, "h5py is required")
class VisionH5ProvenanceTests(unittest.TestCase):
    def test_hub_h5_does_not_inherit_bf16_reference_label(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_manifest = root / "video_npu_manifest.json"
            video_manifest.write_text('{"case": "test"}\n', encoding="utf-8")
            index_manifest = root / "bf16_reference_outputs.json"
            index_manifest.write_text(
                json.dumps(
                    {
                        "reference": "host BF16 reference",
                        "cases": [
                            {
                                "manifest_sha256": sha256_file(video_manifest),
                                "pair_index": 0,
                                "index": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "paired_boundary_jexample.h5"
            names = [
                "image_features",
                "deepstack_visual_embeds_0",
                "deepstack_visual_embeds_1",
                "deepstack_visual_embeds_2",
            ]
            with h5py.File(output_path, "w") as handle:
                data = handle.create_group("data")
                for order, name in enumerate(names):
                    group = data.create_group(str(order))
                    group.attrs["order"] = order
                    group.attrs["name"] = name
                    group.attrs["batch_count"] = 1
                    group.create_dataset(
                        "batch_0",
                        data=np.full((2, 4), order, dtype=np.float32),
                    )

            selected, record = load_selected_vision_outputs(
                output_path,
                index_manifest,
                video_manifest,
                pair_count=1,
                expected_tokens_per_pair=2,
                expected_hidden_size=4,
                vision_source_label="paired_boundary_iq9075_jexample",
                vision_job_id="jexample",
                vision_model_id="mexample",
                vision_dataset_id="dexample",
            )

            self.assertEqual(selected["image_features"].shape, (1, 2, 4))
            self.assertEqual(
                record["vision_output_kind"],
                "qai_hub_inference_h5",
            )
            self.assertEqual(
                record["vision_source_label"],
                "paired_boundary_iq9075_jexample",
            )
            self.assertEqual(
                record["vision_index_reference_label"],
                "host BF16 reference",
            )
            self.assertEqual(
                record["qai_hub_provenance"],
                {
                    "inference_job_id": "jexample",
                    "model_id": "mexample",
                    "dataset_id": "dexample",
                },
            )

    def test_rejects_duplicate_negative_and_wrong_shape_indices(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_manifest = root / "video.json"
            video_manifest.write_text('{"case": "test"}\n', encoding="utf-8")
            output_path = root / "vision.h5"
            names = [
                "image_features",
                "deepstack_visual_embeds_0",
                "deepstack_visual_embeds_1",
                "deepstack_visual_embeds_2",
            ]
            with h5py.File(output_path, "w") as handle:
                data = handle.create_group("data")
                for order, name in enumerate(names):
                    group = data.create_group(str(order))
                    group.attrs.update(
                        order=order,
                        name=name,
                        batch_count=2,
                    )
                    group.create_dataset(
                        "batch_0",
                        data=np.zeros((2, 4), dtype=np.float32),
                    )
                    group.create_dataset(
                        "batch_1",
                        data=np.zeros((2, 4), dtype=np.float32),
                    )

            def write_index(indices: list[int]) -> Path:
                path = root / "index.json"
                path.write_text(
                    json.dumps(
                        {
                            "cases": [
                                {
                                    "manifest_sha256": sha256_file(
                                        video_manifest
                                    ),
                                    "pair_index": pair_index,
                                    "index": index,
                                }
                                for pair_index, index in enumerate(indices)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return path

            with self.assertRaisesRegex(ValueError, "unique"):
                load_selected_vision_outputs(
                    output_path,
                    write_index([0, 0]),
                    video_manifest,
                    pair_count=2,
                    expected_tokens_per_pair=2,
                    expected_hidden_size=4,
                )
            with self.assertRaisesRegex(ValueError, "non-negative"):
                load_selected_vision_outputs(
                    output_path,
                    write_index([-1, 1]),
                    video_manifest,
                    pair_count=2,
                    expected_tokens_per_pair=2,
                    expected_hidden_size=4,
                )
            with self.assertRaisesRegex(ValueError, "exact contract"):
                load_selected_vision_outputs(
                    output_path,
                    write_index([0, 1]),
                    video_manifest,
                    pair_count=2,
                    expected_tokens_per_pair=3,
                    expected_hidden_size=4,
                )


if __name__ == "__main__":
    unittest.main()
