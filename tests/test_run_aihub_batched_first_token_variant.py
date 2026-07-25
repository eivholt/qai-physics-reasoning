from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.run_aihub_batched_first_token_chain as base
from scripts.run_aihub_batched_first_token_variant import (
    decorate_plan,
    execute_variant,
    replaceable_shards,
    validate_derived_index_binding,
)
from scripts.run_aihub_first_token_variant import parse_replacements


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _vision_source(
    index_path: Path,
    selected_case: dict[str, object],
) -> dict[str, object]:
    return {
        "vision": {
            "vision_output_kind": "qai_hub_inference_h5",
            "vision_outputs_sha256": "9" * 64,
            "vision_index_manifest_sha256": hashlib.sha256(
                index_path.read_bytes()
            ).hexdigest(),
            "vision_index_manifest_path": str(index_path),
            "video_manifest_sha256": selected_case["manifest_sha256"],
            "selected_cases": [selected_case],
            "selected_indices": [selected_case["index"]],
            "qai_hub_provenance": {
                "inference_job_id": "job",
                "model_id": "model",
                "dataset_id": "dataset",
            },
        }
    }


class DerivedIndexBindingTests(unittest.TestCase):
    def test_accepts_exact_alias_and_original_source_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_sha = "a" * 64
            pixel_sha = "b" * 64
            original = {
                "case_id": "source:0",
                "index": 0,
                "manifest_sha256": "c" * 64,
                "pair_index": 0,
                "pixel_sha256": pixel_sha,
            }
            alias = {
                "case_id": "alias:variant:0",
                "index": 0,
                "manifest_sha256": "d" * 64,
                "pair_index": 0,
                "pixel_sha256": pixel_sha,
                "alias_provenance": {
                    "match_kind": "exact_pair_index_and_pixel_sha256",
                    "source_index_sha256": source_sha,
                    "source_batch_index": 0,
                    "target_manifest_sha256": "d" * 64,
                    "pair_index": 0,
                    "pixel_sha256": pixel_sha,
                },
            }
            index_path = root / "derived.json"
            _write_json(
                index_path,
                {
                    "schema_version": 2,
                    "index_kind": "pixel_verified_vision_batch_aliases",
                    "cases": [original, alias],
                    "derivation": {
                        "source_index": {
                            "sha256": source_sha,
                            "case_count": 1,
                        }
                    },
                },
            )
            validate_derived_index_binding(
                _vision_source(index_path, original)
            )
            validate_derived_index_binding(_vision_source(index_path, alias))

            incomplete = copy.deepcopy(alias)
            del incomplete["alias_provenance"]["source_index_sha256"]
            with self.assertRaisesRegex(
                ValueError,
                "not uniquely present|incomplete",
            ):
                validate_derived_index_binding(
                    _vision_source(index_path, incomplete)
                )

    def test_rejects_unaliased_case_outside_source_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = {
                "case_id": "source:0",
                "index": 0,
                "manifest_sha256": "c" * 64,
                "pair_index": 0,
                "pixel_sha256": "b" * 64,
            }
            uncovered = {
                "case_id": "uncovered",
                "index": 1,
                "manifest_sha256": "d" * 64,
                "pair_index": 0,
                "pixel_sha256": "e" * 64,
            }
            index_path = root / "derived.json"
            _write_json(
                index_path,
                {
                    "schema_version": 2,
                    "index_kind": "pixel_verified_vision_batch_aliases",
                    "cases": [original, uncovered],
                    "derivation": {
                        "source_index": {
                            "sha256": "a" * 64,
                            "case_count": 1,
                        }
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_derived_index_binding(
                    _vision_source(index_path, uncovered)
                )


class VariantPlanTests(unittest.TestCase):
    def test_only_p2_p3_are_replaceable_and_plan_records_variant(self) -> None:
        subset = replaceable_shards(base.FIXED_SHARDS)
        self.assertEqual(list(subset), ["p2", "p3"])
        replacements = parse_replacements(
            ["p2=mn7yvydon", "p3=mm55g599m"],
            subset,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_replacements(["p4=mn7yvydon"], subset)
        plan = decorate_plan(
            {
                "planned_paid_inference_jobs": 15,
                "job_order": [f"job_{index}" for index in range(15)],
            },
            variant_label="w8_p2_p3",
            replacements=replacements,
            baseline_shards=base.FIXED_SHARDS,
        )
        self.assertEqual(plan["planned_paid_inference_jobs"], 15)
        self.assertEqual(plan["variant"]["label"], "w8_p2_p3")
        self.assertEqual(
            plan["variant"]["fixed_shard_replacements"]["p2"],
            {
                "baseline_model_id": base.FIXED_SHARDS["p2"]["model_id"],
                "replacement_model_id": "mn7yvydon",
            },
        )

    def test_execute_patches_only_its_process_and_restores_globals(self) -> None:
        original_shards = base.FIXED_SHARDS
        original_builder = base.build_plan
        original_condition = base.derive_condition_label
        original_validator = base.validate_alias_provenance
        observed: dict[str, object] = {}

        def fake_plan(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "planned_paid_inference_jobs": 15,
                "job_order": [f"job_{index}" for index in range(15)],
            }

        def fake_execute(
            args: argparse.Namespace,
            *,
            hub: object,
        ) -> dict[str, object]:
            del hub
            observed["p2"] = base.FIXED_SHARDS["p2"]["model_id"]
            observed["p3"] = base.FIXED_SHARDS["p3"]["model_id"]
            observed["name_prefix"] = args.name_prefix
            observed["plan"] = base.build_plan()
            observed["condition"] = base.derive_condition_label(
                {"source": {"vision": {}}}
            )
            observed["validator"] = base.validate_alias_provenance
            return {"planned_paid_inference_jobs": 15}

        args = argparse.Namespace(
            panel_manifest=Path("/panel.json"),
            work_dir=Path("/work"),
            benchmark_manifest=Path("/benchmark.json"),
            candidate=["baseline_w4_fp16", "w8_layers_0_6"],
            fixed_shard_model=["p2=mn7yvydon", "p3=mm55g599m"],
            variant_label="w8_p2_p3",
            name_prefix="cosmos_panel",
            submit_missing=False,
            preflight_only=True,
        )
        with (
            patch.object(base, "build_plan", fake_plan),
            patch.object(base, "execute", side_effect=fake_execute),
        ):
            result = execute_variant(args, hub=SimpleNamespace())

        self.assertEqual(observed["p2"], "mn7yvydon")
        self.assertEqual(observed["p3"], "mm55g599m")
        self.assertEqual(
            observed["name_prefix"],
            "cosmos_panel_w8_p2_p3",
        )
        self.assertEqual(
            observed["plan"]["variant"]["label"],
            "w8_p2_p3",
        )
        self.assertTrue(str(observed["condition"]).endswith("_w8_p2_p3"))
        self.assertIs(observed["validator"], validate_derived_index_binding)
        self.assertEqual(result["variant_label"], "w8_p2_p3")
        self.assertIs(base.FIXED_SHARDS, original_shards)
        self.assertIs(base.build_plan, original_builder)
        self.assertIs(base.derive_condition_label, original_condition)
        self.assertIs(base.validate_alias_provenance, original_validator)

    def test_unsafe_variant_label_is_rejected_before_execution(self) -> None:
        args = argparse.Namespace(
            fixed_shard_model=["p2=mn7yvydon"],
            variant_label="../unsafe",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe variant"):
            execute_variant(args, hub=SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
