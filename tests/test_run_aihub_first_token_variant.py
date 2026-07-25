from __future__ import annotations

import argparse
import unittest
from collections import OrderedDict
from pathlib import Path

from scripts.run_aihub_first_token_variant import (
    base_argv,
    parse_replacements,
)


FIXED = OrderedDict(
    [
        ("p2", {"model_id": "mbaseline"}),
        ("p3", {"model_id": "motherone"}),
        ("p4", {"model_id": "mlastpart"}),
    ]
)


class RunAIHubFirstTokenVariantTests(unittest.TestCase):
    def test_parse_replacements(self) -> None:
        self.assertEqual(
            parse_replacements(
                ["p3=mm55g599m", "p2=mn7yvydon"],
                FIXED,
            ),
            {"p3": "mm55g599m", "p2": "mn7yvydon"},
        )

    def test_rejects_invalid_or_noop_replacements(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_replacements([], FIXED)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_replacements(["p1=mn7yvydon"], FIXED)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_replacements(
                ["p2=mn7yvydon", "p2=mm55g599m"],
                FIXED,
            )
        with self.assertRaisesRegex(ValueError, "Invalid AI Hub"):
            parse_replacements(["p2=not-a-model"], FIXED)
        with self.assertRaisesRegex(ValueError, "equals the baseline"):
            parse_replacements(["p2=mbaseline"], FIXED)

    def test_builds_base_runner_arguments(self) -> None:
        args = argparse.Namespace(
            input_artifact_dir=Path("/tmp/input"),
            work_dir=Path("/tmp/output"),
            seed_p1_screen=Path("/tmp/screen.json"),
            candidate=["baseline", "w8"],
            benchmark_manifest=Path("/tmp/benchmark.json"),
            case_id="case",
            probe_id="probe",
            preflight_only=True,
        )
        argv = base_argv(args)
        self.assertEqual(argv[1:3], ["/tmp/input", "/tmp/output"])
        self.assertEqual(argv.count("--candidate"), 2)
        self.assertIn("--preflight-only", argv)
        self.assertNotIn("--fixed-shard-model", argv)


if __name__ == "__main__":
    unittest.main()
