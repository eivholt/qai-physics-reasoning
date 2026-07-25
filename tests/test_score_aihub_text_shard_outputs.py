from __future__ import annotations

import unittest
from collections import OrderedDict

import numpy as np

from scripts.score_aihub_text_shard_outputs import (
    aggregate_metrics,
    parse_candidates,
)


class ScoreAIHubTextShardOutputsTests(unittest.TestCase):
    def test_parse_candidates_preserves_order_and_rejects_duplicates(self):
        self.assertEqual(
            parse_candidates(["baseline=j1", "w8=j2"]),
            OrderedDict([("baseline", "j1"), ("w8", "j2")]),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate candidate label"):
            parse_candidates(["same=j1", "same=j2"])
        with self.assertRaisesRegex(ValueError, "Duplicate candidate job ID"):
            parse_candidates(["a=j1", "b=j1"])
        with self.assertRaisesRegex(ValueError, "At least two"):
            parse_candidates(["only=j1"])

    def test_aggregate_metrics_reports_each_batch(self):
        reference = OrderedDict(
            [
                ("hidden", np.ones((2, 1, 2, 2), dtype=np.float32)),
                ("past_key_7_out", np.ones((2, 1, 1), dtype=np.float32)),
                ("past_value_7_out", np.ones((2, 1, 1), dtype=np.float32)),
            ]
        )
        outputs = OrderedDict(
            (name, value.copy()) for name, value in reference.items()
        )
        outputs["hidden"][1] += 1.0
        metrics = aggregate_metrics(
            outputs,
            reference,
            hidden_name="hidden",
            batch_labels=["a", "b"],
        )
        self.assertEqual(
            [item["batch_label"] for item in metrics["per_batch"]],
            ["a", "b"],
        )
        self.assertEqual(
            metrics["per_batch"][0]["hidden"]["relative_rmse"],
            0.0,
        )
        self.assertGreater(
            metrics["per_batch"][1]["hidden"]["relative_rmse"],
            0.0,
        )
        self.assertEqual(metrics["kv_tensor_count"], 2)

    def test_aggregate_metrics_rejects_order_or_shape_mismatch(self):
        reference = OrderedDict(
            [
                ("hidden", np.zeros((2, 1), dtype=np.float32)),
                ("past_key_7_out", np.zeros((2, 1), dtype=np.float32)),
            ]
        )
        reordered = OrderedDict(reversed(list(reference.items())))
        with self.assertRaisesRegex(ValueError, "tensor order differs"):
            aggregate_metrics(
                reordered,
                reference,
                hidden_name="hidden",
                batch_labels=["a", "b"],
            )
        wrong_shape = OrderedDict(
            (name, value.copy()) for name, value in reference.items()
        )
        wrong_shape["hidden"] = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape differs"):
            aggregate_metrics(
                wrong_shape,
                reference,
                hidden_name="hidden",
                batch_labels=["a", "b"],
            )


if __name__ == "__main__":
    unittest.main()
