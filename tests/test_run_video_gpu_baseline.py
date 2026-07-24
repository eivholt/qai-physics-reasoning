from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_video_gpu_baseline import (
    FREEFORM_PROMPT_ID,
    PromptSpec,
    apply_upstream_processor,
    build_messages,
    build_video_payload,
    checkpoint_provenance,
    extract_choice,
    load_benchmark,
    prompt_specs_for_case,
    resolve_cuda_device,
    score_answer,
    safe_component,
    select_video_cases,
    validate_observed_contract,
    validate_processor_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    REPO_ROOT / "benchmarks" / "nvidia_sdg_warehouse" / "benchmark.json"
)


class _FakeImage:
    size = (384, 216)
    mode = "RGB"


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return {"input_ids": "fake"} if kwargs.get("tokenize") else "rendered"


class RunVideoGpuBaselineTests(unittest.TestCase):
    def test_resolves_bare_cuda_to_current_index(self) -> None:
        class FakeDevice:
            def __init__(self, kind: str, index: int | None = None) -> None:
                self.type = kind
                self.index = index

        class FakeCuda:
            def __init__(self) -> None:
                self.selected = None

            @staticmethod
            def current_device() -> int:
                return 2

            def set_device(self, device: FakeDevice) -> None:
                self.selected = device

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def device(kind: str, index: int | None = None) -> FakeDevice:
                return FakeDevice(kind.split(":", 1)[0], index)

        device = resolve_cuda_device(FakeTorch, "cuda")
        self.assertEqual((device.type, device.index), ("cuda", 2))
        self.assertIs(FakeTorch.cuda.selected, device)

    def test_selects_freeform_and_shared_normal_shuffled_probes(self) -> None:
        manifest, _ = load_benchmark(BENCHMARK)
        cases = select_video_cases(
            manifest,
            [
                "video_barrier_knockdown_4fps",
                "video_routine_box_pickup_4fps",
            ],
        )
        specs = [
            spec
            for case in cases
            for spec in prompt_specs_for_case(case, ["all"])
        ]
        self.assertEqual(len(specs), 16)
        expected = {
            (spec.case_id, spec.prompt_id): spec.expected.get("letter")
            for spec in specs
            if spec.kind == "choice"
        }
        self.assertEqual(
            expected,
            {
                (
                    "video_barrier_knockdown_4fps",
                    "cross_scene_event_choice",
                ): "A",
                (
                    "video_barrier_knockdown_4fps",
                    "cross_scene_event_choice_shuffled",
                ): "C",
                (
                    "video_barrier_knockdown_4fps",
                    "four_scene_event_choice",
                ): "A",
                (
                    "video_barrier_knockdown_4fps",
                    "four_scene_event_choice_shuffled",
                ): "C",
                (
                    "video_barrier_knockdown_4fps",
                    "four_scene_compact_choice",
                ): "A",
                (
                    "video_barrier_knockdown_4fps",
                    "four_scene_compact_choice_shuffled",
                ): "C",
                (
                    "video_routine_box_pickup_4fps",
                    "cross_scene_event_choice",
                ): "B",
                (
                    "video_routine_box_pickup_4fps",
                    "cross_scene_event_choice_shuffled",
                ): "A",
                (
                    "video_routine_box_pickup_4fps",
                    "four_scene_event_choice",
                ): "B",
                (
                    "video_routine_box_pickup_4fps",
                    "four_scene_event_choice_shuffled",
                ): "D",
                (
                    "video_routine_box_pickup_4fps",
                    "four_scene_compact_choice",
                ): "B",
                (
                    "video_routine_box_pickup_4fps",
                    "four_scene_compact_choice_shuffled",
                ): "D",
                (
                    "video_routine_box_pickup_4fps",
                    "box_near_pairwise_choice",
                ): "A",
                (
                    "video_routine_box_pickup_4fps",
                    "box_near_pairwise_choice_shuffled",
                ): "B",
            },
        )
        self.assertEqual(
            sum(spec.prompt_id == FREEFORM_PROMPT_ID for spec in specs), 2
        )

    def test_rejects_unknown_case_and_prompt(self) -> None:
        manifest, _ = load_benchmark(BENCHMARK)
        with self.assertRaisesRegex(ValueError, "Unknown video case"):
            select_video_cases(manifest, ["missing"])
        case = manifest["video_cases"][0]
        with self.assertRaisesRegex(ValueError, "unknown prompt"):
            prompt_specs_for_case(case, ["missing"])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            prompt_specs_for_case(case, ["all", "freeform"])
        with self.assertRaisesRegex(ValueError, "Duplicate video case"):
            select_video_cases(
                manifest,
                [
                    "video_barrier_knockdown_4fps",
                    "video_barrier_knockdown_4fps",
                ],
            )
        with self.assertRaisesRegex(ValueError, "duplicate prompt"):
            prompt_specs_for_case(
                case,
                [
                    "cross_scene_event_choice",
                    "cross_scene_event_choice",
                ],
            )

    def test_rejects_dot_segments_in_manifest_components(self) -> None:
        for value in (".", "..", "../escape", "nested/path"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must contain only"):
                    safe_component(value, "case id")

    def test_builds_native_pair_metadata_and_sanitized_frame_records(self) -> None:
        case = {
            "id": "video_case",
            "temporal_pairs": [
                {
                    "frame_indices": [2, 6],
                    "assumed_times_seconds": [0.2, 0.6],
                    "prompt_timestamp_text": "0.4",
                }
            ],
        }
        asset = {
            "assumed_preview_fps": 10,
            "frame_count": 100,
            "documented_scenario_duration_seconds": 10,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_dir = root / "video_case"
            frame_dir.mkdir()
            (frame_dir / "frame_0002.png").write_bytes(b"frame-two")
            (frame_dir / "frame_0006.png").write_bytes(b"frame-six")

            videos, metadata, frames, pairs = build_video_payload(
                case,
                asset,
                root,
                image_loader=lambda _: _FakeImage(),
                metadata_factory=lambda **kwargs: kwargs,
            )

            self.assertEqual(len(videos), 1)
            self.assertEqual(
                metadata,
                [
                    {
                        "total_num_frames": 100,
                        "fps": 10.0,
                        "width": 384,
                        "height": 216,
                        "duration": 10.0,
                        "frames_indices": [2, 6],
                    }
                ],
            )
            self.assertEqual(
                [frame["file"] for frame in frames],
                [
                    "video_case/frame_0002.png",
                    "video_case/frame_0006.png",
                ],
            )
            self.assertTrue(all("local_path" not in frame for frame in frames))
            self.assertEqual(pairs[0]["prompt_timestamp_text"], "0.4")
            serialized = json.dumps({"frames": frames, "pairs": pairs})
            self.assertNotIn(str(root), serialized)

    def test_local_paths_require_explicit_opt_in(self) -> None:
        case = {
            "id": "video_case",
            "temporal_pairs": [
                {
                    "frame_indices": [0, 1],
                    "assumed_times_seconds": [0.0, 0.2],
                    "prompt_timestamp_text": "0.1",
                }
            ],
        }
        asset = {
            "assumed_preview_fps": 5,
            "frame_count": 50,
            "documented_scenario_duration_seconds": 10,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_dir = root / "video_case"
            frame_dir.mkdir()
            for index in (0, 1):
                (frame_dir / f"frame_{index:04d}.png").write_bytes(b"x")
            _, _, frames, _ = build_video_payload(
                case,
                asset,
                root,
                image_loader=lambda _: _FakeImage(),
                metadata_factory=lambda **kwargs: kwargs,
                include_local_paths=True,
            )
            self.assertTrue(
                all(Path(frame["local_path"]).is_absolute() for frame in frames)
            )

    def test_calls_upstream_processor_without_sampling_or_manual_resize(self) -> None:
        processor = _FakeProcessor()
        videos = [["frame-a", "frame-b"], ["frame-c", "frame-d"]]
        messages = build_messages("system", "question", videos)
        rendered, inputs = apply_upstream_processor(
            processor,
            messages,
            [{"frames_indices": [0, 1]}, {"frames_indices": [2, 3]}],
        )
        self.assertEqual(rendered, "rendered")
        self.assertEqual(inputs, {"input_ids": "fake"})
        self.assertEqual(len(processor.calls), 2)
        tokenized = processor.calls[1]
        self.assertTrue(tokenized["tokenize"])
        self.assertEqual(
            tokenized["processor_kwargs"],
            {
                "do_sample_frames": False,
                "video_metadata": [
                    {"frames_indices": [0, 1]},
                    {"frames_indices": [2, 3]},
                ],
                "return_mm_token_type_ids": True,
            },
        )
        self.assertNotIn("do_resize", tokenized["processor_kwargs"])

    def test_processor_contract_requires_multimodal_token_ids(self) -> None:
        complete = {
            "input_ids": object(),
            "attention_mask": object(),
            "mm_token_type_ids": object(),
            "pixel_values_videos": object(),
            "video_grid_thw": object(),
        }
        validate_processor_outputs(complete)
        complete.pop("mm_token_type_ids")
        with self.assertRaisesRegex(ValueError, "mm_token_type_ids"):
            validate_processor_outputs(complete)

    def test_validates_pinned_prefill_and_context_contract(self) -> None:
        spec = PromptSpec(
            case_id="case",
            prompt_id="choice",
            kind="choice",
            prompt="prompt",
            expected={"letter": "B"},
            context_budget={"prompt_tokens": 344, "visual_tokens": 252},
            evaluation_role="gate",
        )
        contract = validate_observed_contract(
            spec,
            {
                "context_tokens": 512,
                "prefill_kv_capacity_tokens": 384,
            },
            prompt_tokens=344,
            visual_tokens=252,
            max_new_tokens=64,
        )
        self.assertEqual(contract["prefill_headroom"], 40)
        self.assertEqual(contract["generation_headroom"], 104)
        with self.assertRaisesRegex(ValueError, "processor produced 345"):
            validate_observed_contract(
                spec,
                {
                    "context_tokens": 512,
                    "prefill_kv_capacity_tokens": 384,
                },
                prompt_tokens=345,
                visual_tokens=252,
                max_new_tokens=64,
            )

    def test_scores_choice_and_leaves_freeform_for_semantic_review(self) -> None:
        self.assertEqual(extract_choice("B"), "B")
        self.assertEqual(extract_choice("B) because the worker has the box"), "B")
        self.assertEqual(extract_choice("The answer is C."), "C")
        self.assertEqual(extract_choice("D — workers are evacuating"), "D")
        self.assertIsNone(extract_choice("The model cannot decide."))
        choice = PromptSpec(
            case_id="case",
            prompt_id="choice",
            kind="choice",
            prompt="prompt",
            expected={"letter": "C"},
            context_budget={},
            evaluation_role="gate",
        )
        self.assertTrue(score_answer(choice, "C")["pass"])
        self.assertFalse(score_answer(choice, "A")["pass"])
        freeform = PromptSpec(
            case_id="case",
            prompt_id="freeform",
            kind="freeform",
            prompt="prompt",
            expected={},
            context_budget={},
            evaluation_role="diagnostic",
        )
        self.assertIsNone(score_answer(freeform, "anything")["pass"])

    def test_checkpoint_provenance_uses_names_not_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            for filename in (
                "config.json",
                "chat_template.json",
                "tokenizer_config.json",
                "model.safetensors",
            ):
                (checkpoint / filename).write_bytes(filename.encode())
            record = checkpoint_provenance(checkpoint)
            serialized = json.dumps(record)
            self.assertEqual(record["identifier"], "checkpoint")
            self.assertIn("model.safetensors", record["files"])
            self.assertNotIn(str(checkpoint), serialized)
            self.assertNotIn("local_path", serialized)


if __name__ == "__main__":
    unittest.main()
