from __future__ import annotations

import unittest
from collections import OrderedDict

import numpy as np

from scripts.capture_host_text_shard_reference import (
    expected_input_shapes,
    validate_dataset_entries,
)


def valid_entries(shard: str, batch_count: int = 2):
    return OrderedDict(
        (
            name,
            [np.zeros(shape, dtype=np.float32) for _ in range(batch_count)],
        )
        for name, shape in expected_input_shapes(shard).items()
    )


class CaptureHostTextShardReferenceTests(unittest.TestCase):
    def test_expected_partition_contracts(self) -> None:
        p2 = expected_input_shapes("p2")
        p3 = expected_input_shapes("p3")
        self.assertEqual(p2["add_15805"], (1, 128, 2048))
        self.assertEqual(p3["add_30134"], (1, 128, 2048))
        self.assertIn("past_key_7_in", p2)
        self.assertNotIn("past_key_14_in", p2)
        self.assertIn("past_key_14_in", p3)
        self.assertNotIn("past_key_7_in", p3)

    def test_validates_batches_and_labels(self) -> None:
        entries, labels = validate_dataset_entries(
            valid_entries("p2"),
            shard="p2",
            batch_labels=["baseline", "w8_p1"],
        )
        self.assertEqual(labels, ["baseline", "w8_p1"])
        self.assertEqual(len(entries["add_15805"]), 2)

    def test_rejects_input_order_changes(self) -> None:
        entries = valid_entries("p3")
        entries.move_to_end("attention_mask", last=False)
        with self.assertRaisesRegex(ValueError, "input order differs"):
            validate_dataset_entries(
                entries,
                shard="p3",
                batch_labels=None,
            )

    def test_rejects_wrong_shape_dtype_and_nonfinite(self) -> None:
        entries = valid_entries("p2")
        entries["add_15805"][0] = np.zeros(
            (1, 127, 2048),
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, "shape is"):
            validate_dataset_entries(
                entries,
                shard="p2",
                batch_labels=None,
            )

        entries = valid_entries("p2")
        entries["attention_mask"][0] = entries["attention_mask"][0].astype(
            np.float16
        )
        with self.assertRaisesRegex(ValueError, "dtype is"):
            validate_dataset_entries(
                entries,
                shard="p2",
                batch_labels=None,
            )

        entries = valid_entries("p2")
        entries["position_ids_cos"][0].flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_dataset_entries(
                entries,
                shard="p2",
                batch_labels=None,
            )

    def test_rejects_bad_batch_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2 batch labels"):
            validate_dataset_entries(
                valid_entries("p3"),
                shard="p3",
                batch_labels=["only_one"],
            )
        with self.assertRaisesRegex(ValueError, "unique and non-empty"):
            validate_dataset_entries(
                valid_entries("p3"),
                shard="p3",
                batch_labels=["same", "same"],
            )


if __name__ == "__main__":
    unittest.main()
