from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_video_npu_results import (
    NpuResultSpec,
    ProbeKey,
    build_report,
    group_npu_candidate_roots,
    load_npu_results,
    parse_genie_log,
    permutation_index,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    REPO_ROOT / "benchmarks" / "nvidia_sdg_warehouse" / "benchmark.json"
)


def _write_gpu(
    root: Path,
    case_id: str,
    prompt_id: str,
    expected: str,
    answer: str,
    gpu_seconds: float = 0.08,
) -> None:
    payload = {
        "case_id": case_id,
        "prompt_id": prompt_id,
        "expected": {"letter": expected},
        "answer": answer,
        "rubric_assessment": {
            "selected_letter": answer,
            "pass": answer == expected,
        },
        "timing": {"gpu_seconds": gpu_seconds},
    }
    path = root / f"{case_id}.{prompt_id}.gpu_bf16.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_npu(
    root: Path,
    case_id: str,
    prompt_id: str,
    answer: str,
    ttft_ms: float,
    decode: float = 0.0,
) -> None:
    path = root / "nested" / f"{case_id}.{prompt_id}.log"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Loading full-DeepStack QAIRT contexts on QnnHtp...",
                "COSMOS_GENIEX_OUTPUT_BEGIN",
                answer,
                "COSMOS_GENIEX_OUTPUT_END",
                "Prompt tokens: 357",
                "Generated tokens: 1",
                f"TTFT (vision + prefill): {ttft_ms} ms",
                f"Decode: {decode:.2f} tokens/s",
            ]
        ),
        encoding="utf-8",
    )


class ScoreVideoNpuResultsTests(unittest.TestCase):
    def test_parses_current_log_and_strips_ansi(self) -> None:
        parsed = parse_genie_log(
            "\x1b[32m[INFO]\x1b[0m ready\n"
            "COSMOS_GENIEX_OUTPUT_BEGIN\n"
            "C) the worker dodges\n"
            "COSMOS_GENIEX_OUTPUT_END\n"
            "Prompt tokens: 357\n"
            "Generated tokens: 5\n"
            "TTFT (vision + prefill): 868.3 ms\n"
            "Decode: 11.40 tokens/s\n"
        )
        self.assertEqual(parsed["parse_status"], "ok")
        self.assertEqual(parsed["answer"], "C) the worker dodges")
        self.assertEqual(parsed["selected_letter"], "C")
        self.assertEqual(parsed["marker_format"], "cosmos_geniex")
        self.assertEqual(parsed["prompt_tokens"], 357)
        self.assertEqual(parsed["generated_tokens"], 5)
        self.assertEqual(parsed["ttft_ms"], 868.3)
        self.assertEqual(parsed["decode_tokens_per_second"], 11.4)

    def test_parses_last_legacy_answer_block(self) -> None:
        parsed = parse_genie_log(
            "[BEGIN]: A [END]\n"
            "[BEGIN]: Option B is shown [END]\n"
            "TTFT: 700 ms\n"
        )
        self.assertEqual(parsed["answer_block_count"], 2)
        self.assertEqual(parsed["answer"], "Option B is shown")
        self.assertEqual(parsed["selected_letter"], "B")
        self.assertEqual(parsed["marker_format"], "legacy_begin_end")
        self.assertEqual(parsed["ttft_ms"], 700.0)

    def test_reports_missing_marker_without_inventing_answer(self) -> None:
        parsed = parse_genie_log(
            "Prompt tokens: 360\nGenerated tokens: 0\n"
        )
        self.assertEqual(parsed["parse_status"], "missing_output_marker")
        self.assertIsNone(parsed["answer"])
        self.assertIsNone(parsed["selected_letter"])

    def test_maps_normal_shuffled_and_numbered_permutations(self) -> None:
        self.assertEqual(permutation_index("four_scene_event_choice"), 1)
        self.assertEqual(
            permutation_index("four_scene_event_choice_shuffled"), 2
        )
        self.assertEqual(
            permutation_index("four_scene_event_choice_permutation_4"), 4
        )

    def test_builds_accuracy_parity_retention_groups_and_changes(self) -> None:
        cases = [
            (
                "video_barrier_knockdown_4fps",
                "four_scene_event_choice",
                "A",
                "A",
            ),
            (
                "video_barrier_knockdown_4fps",
                "four_scene_event_choice_shuffled",
                "C",
                "B",
            ),
            (
                "video_routine_box_pickup_4fps",
                "four_scene_event_choice",
                "B",
                "B",
            ),
            (
                "video_routine_box_pickup_4fps",
                "four_scene_event_choice_shuffled",
                "D",
                "D",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu = root / "gpu"
            baseline = root / "baseline"
            candidate = root / "candidate"
            gpu.mkdir()
            baseline.mkdir()
            candidate.mkdir()
            for case_id, prompt_id, expected, gpu_answer in cases:
                _write_gpu(
                    gpu, case_id, prompt_id, expected, gpu_answer
                )

            baseline_answers = ["A", "C", "A", "D"]
            candidate_answers = ["B", "C", "B", "D"]
            for index, ((case_id, prompt_id, _, _), answer) in enumerate(
                zip(cases, baseline_answers)
            ):
                _write_npu(
                    baseline,
                    case_id,
                    prompt_id,
                    answer,
                    700.0 + index * 100,
                )
            for index, ((case_id, prompt_id, _, _), answer) in enumerate(
                zip(cases, candidate_answers)
            ):
                _write_npu(
                    candidate,
                    case_id,
                    prompt_id,
                    answer,
                    600.0 + index * 100,
                )

            report = build_report(
                BENCHMARK,
                [gpu],
                [("baseline", baseline), ("candidate", candidate)],
            )

        self.assertEqual(report["gpu_reference"]["correct"], 3)
        summary = report["candidates"]["baseline"]["summary"]
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["accuracy"], 0.75)
        self.assertEqual(summary["exact_choice_parity"], 2)
        self.assertEqual(summary["exact_choice_parity_rate"], 0.5)
        self.assertEqual(summary["gpu_correct_retained"], 2)
        self.assertEqual(summary["gpu_correct_total"], 3)
        self.assertEqual(
            summary["gpu_correct_retention_rate"], round(2 / 3, 6)
        )
        self.assertEqual(summary["npu_correct_on_gpu_incorrect"], 1)
        self.assertEqual(summary["timing"]["ttft_ms"]["mean"], 850.0)

        by_scene = report["candidates"]["baseline"]["by_scene"]
        self.assertEqual(
            by_scene["video_barrier_knockdown_4fps"]["correct"], 2
        )
        self.assertEqual(
            by_scene["video_routine_box_pickup_4fps"]["correct"], 1
        )
        by_permutation = report["candidates"]["baseline"][
            "by_permutation"
        ]
        self.assertEqual(
            by_permutation["four_scene_event_choice_order/p1"]["correct"],
            1,
        )
        self.assertEqual(
            by_permutation["four_scene_event_choice_order/p2"]["correct"],
            2,
        )

        comparison = report["answer_changes"][0]
        self.assertEqual(comparison["changed"], 2)
        self.assertEqual(comparison["improved"], 1)
        self.assertEqual(comparison["regressed"], 1)
        self.assertEqual(comparison["correct_delta"], 0)
        self.assertEqual(len(comparison["changes"]), 2)

        serialized = json.dumps(report)
        self.assertNotIn(str(root), serialized)

    def test_missing_log_counts_against_total_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu = root / "gpu"
            npu = root / "npu"
            gpu.mkdir()
            npu.mkdir()
            _write_gpu(
                gpu,
                "video_barrier_knockdown_4fps",
                "four_scene_event_choice",
                "A",
                "A",
            )
            report = build_report(
                BENCHMARK, [gpu], [("partial", npu)]
            )
        summary = report["candidates"]["partial"]["summary"]
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["parsed"], 0)
        self.assertEqual(summary["missing_or_unparsed"], 1)
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(summary["exact_choice_parity_rate"], 0.0)
        self.assertEqual(summary["gpu_correct_retention_rate"], 0.0)

    def test_merges_repeated_explicit_candidate_labels(self) -> None:
        first_key = (
            "video_barrier_knockdown_4fps",
            "four_scene_event_choice",
            "A",
        )
        second_key = (
            "video_routine_box_pickup_4fps",
            "four_scene_event_choice",
            "B",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu = root / "gpu"
            first = root / "primary results"
            second = root / "compact"
            gpu.mkdir()
            first.mkdir()
            second.mkdir()
            for case_id, prompt_id, expected in (first_key, second_key):
                _write_gpu(
                    gpu, case_id, prompt_id, expected, expected
                )
            _write_npu(first, *first_key, 700.0)
            _write_npu(second, *second_key, 800.0)

            report = build_report(
                BENCHMARK,
                [gpu],
                [
                    ("merged", first),
                    ("merged", second),
                ],
            )

        self.assertEqual(list(report["candidates"]), ["merged"])
        merged = report["candidates"]["merged"]
        self.assertEqual(merged["summary"]["correct"], 2)
        self.assertEqual(merged["summary"]["parsed"], 2)
        self.assertEqual(
            merged["source_root_names"], ["primary_results", "compact"]
        )
        sources = [
            entry["npu"]["source_file"] for entry in merged["cases"]
        ]
        self.assertTrue(
            any(source.startswith("primary_results/") for source in sources)
        )
        self.assertTrue(
            any(source.startswith("compact/") for source in sources)
        )
        self.assertNotIn(str(root), json.dumps(report))

    def test_keeps_unlabelled_candidates_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one" / "results"
            second = root / "two" / "results"
            grouped = group_npu_candidate_roots(
                [
                    NpuResultSpec("results", first, False),
                    NpuResultSpec("results", second, False),
                ]
            )
        self.assertEqual(
            grouped,
            [
                ("results", [first]),
                ("results-2", [second]),
            ],
        )

    def test_rejects_duplicate_logs_across_merged_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a"
            second = root / "b"
            first.mkdir()
            second.mkdir()
            filename = (
                "video_barrier_knockdown_4fps."
                "four_scene_event_choice.log"
            )
            (first / filename).write_text(
                "COSMOS_GENIEX_OUTPUT_BEGIN\nA\n"
                "COSMOS_GENIEX_OUTPUT_END\n",
                encoding="utf-8",
            )
            (second / filename).write_text(
                "COSMOS_GENIEX_OUTPUT_BEGIN\nA\n"
                "COSMOS_GENIEX_OUTPUT_END\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate NPU result"):
                load_npu_results(
                    [first, second],
                    [
                        ProbeKey(
                            "video_barrier_knockdown_4fps",
                            "four_scene_event_choice",
                        )
                    ],
                )


if __name__ == "__main__":
    unittest.main()
