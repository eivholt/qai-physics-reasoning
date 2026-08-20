from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINETUNE = ROOT / "scripts" / "reason2_finetune"
sys.path.insert(0, str(FINETUNE))

from generative_common import (  # noqa: E402
    EVK_SPEED_CLASSIFIER_USER_PROMPT,
    evk_speed_classifier_response_for_code,
    prediction_from_response,
    task_record,
    use_evk_speed_classifier_contract,
)


class Reason2SpeedContractTests(unittest.TestCase):
    def test_contract_is_user_only_and_one_character(self) -> None:
        record = task_record({
            "image": "sample.png",
            "image_key": "sample",
            "code": "A",
            "label": "OVER_EDGE",
        })
        use_evk_speed_classifier_contract(record)

        self.assertEqual(
            record["prompt"],
            [{"role": "user", "content": EVK_SPEED_CLASSIFIER_USER_PROMPT}],
        )
        self.assertEqual(record["completion"][0]["content"], "A")

    def test_every_direct_code_maps_to_derived_ui_fields(self) -> None:
        expected = {
            "G": ("SUPPORTED", "NO", "NO", "GREEN"),
            "A": ("OVER_EDGE", "NO", "YES", "AMBER"),
            "R": ("FALLEN", "YES", "NO", "RED"),
        }
        for code, (label, fallen, unstable, answer) in expected.items():
            with self.subTest(code=code):
                prediction, parsed = prediction_from_response(code)
                self.assertEqual(prediction, label)
                self.assertEqual(parsed["fallen_test"], fallen)
                self.assertEqual(parsed["unstable_test"], unstable)
                self.assertEqual(parsed["prediction_answer"], answer)
                self.assertEqual(evk_speed_classifier_response_for_code(code), code)

    def test_direct_parser_does_not_accept_prose_prefixes(self) -> None:
        prediction, parsed = prediction_from_response("GREEN is safe")
        self.assertIsNone(prediction)
        self.assertIsNone(parsed)


class Reason2SpeedReleaseWorkflowTests(unittest.TestCase):
    def test_local_archive_staging_is_hash_verified_and_atomic(self) -> None:
        source = (ROOT / "scripts" / "stage_evk_geniex_bundle.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('local_archive="${LOCAL_ARCHIVE:-}"', source)
        self.assertIn('sha256sum "${local_archive}"', source)
        self.assertIn("--strip-components=1", source)
        self.assertIn('mv "${importing}" "${destination_dir}"', source)

    def test_candidate_gate_restores_production_geniex_and_records_runtime(self) -> None:
        source = (ROOT / "scripts" / "run_evk_geniex_candidate_gate.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("qai-conveyor-geniex.service", source)
        self.assertIn("restore_production", source)
        self.assertNotIn("qai-conveyor-media-bridge.service", source)
        self.assertNotIn("genie-app", source)
        self.assertIn('"bundle_archive_sha256": sys.argv[6] or None', source)
        self.assertIn('"direct_embedded_image": bool(report.get("dataset"))', source)

if __name__ == "__main__":
    unittest.main()
