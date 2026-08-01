from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_isaac_edge_supervisor_media import (
    _load_evk_timing_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "iq9075_isaac_realistic_aisle_advisory_r1.json"
)


class BuildIsaacEdgeSupervisorMediaTest(unittest.TestCase):
    def test_loads_measured_evk_task_split(self) -> None:
        timing = _load_evk_timing_evidence(EVIDENCE_PATH)

        self.assertEqual(timing["reference_scenario"], "aisle_congestion")
        self.assertEqual(timing["request_seconds"], [5.579779, 3.035378])
        self.assertAlmostEqual(timing["total_seconds"], 8.615157)


if __name__ == "__main__":
    unittest.main()
