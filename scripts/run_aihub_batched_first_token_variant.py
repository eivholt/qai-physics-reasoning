#!/usr/bin/env python3
"""Run a labelled batched panel with explicit fixed P2/P3 replacements.

This is an additive wrapper around
``run_aihub_batched_first_token_chain.py``.  It does not change the base
runner or its active state schema.  A variant receives its own immutable plan,
condition label, job-name prefix, and work directory.  Only the fixed P2/P3
target-model IDs may change; graph names, layer ranges, tensor contracts, P1
candidates, P4, batching order, and cache isolation remain frozen.

The wrapper also permits normal-order control artifacts to join prompt-variant
artifacts when all were rebuilt against the same derived alias index.  Every
selected row must be either an authenticated alias or an original source-index
row covered by that derived index's source-case inventory.
"""

from __future__ import annotations

import argparse
import copy
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import scripts.run_aihub_batched_first_token_chain as base
    from scripts.run_aihub_first_token_variant import (
        SAFE_LABEL_RE,
        parse_replacements,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_aihub_batched_first_token_chain as base  # type: ignore[no-redef]
    from run_aihub_first_token_variant import (  # type: ignore[no-redef]
        SAFE_LABEL_RE,
        parse_replacements,
    )


REPLACEABLE_FIXED_SHARDS = ("p2", "p3")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def replaceable_shards(
    fixed_shards: Mapping[str, Mapping[str, Any]],
) -> "OrderedDict[str, Mapping[str, Any]]":
    return OrderedDict(
        (shard, fixed_shards[shard])
        for shard in REPLACEABLE_FIXED_SHARDS
    )


def validate_derived_index_binding(source: Mapping[str, Any]) -> None:
    """Accept only exact alias rows or authenticated original source rows."""

    vision = source.get("vision")
    if not isinstance(vision, dict):
        raise ValueError("Input artifact has no vision provenance")
    if vision.get("vision_output_kind") != "qai_hub_inference_h5":
        raise ValueError("Variant panel requires an AI Hub vision H5")
    for field in (
        "vision_outputs_sha256",
        "vision_index_manifest_sha256",
        "video_manifest_sha256",
    ):
        value = vision.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"Vision provenance has invalid {field}")
    hub_provenance = vision.get("qai_hub_provenance")
    if not isinstance(hub_provenance, dict) or any(
        not isinstance(hub_provenance.get(field), str)
        or not hub_provenance[field]
        for field in ("inference_job_id", "model_id", "dataset_id")
    ):
        raise ValueError("Vision H5 has incomplete AI Hub provenance")

    index_path = Path(str(vision.get("vision_index_manifest_path", "")))
    if not index_path.is_file():
        raise ValueError("Derived vision-index manifest is missing")
    if base.sha256_file(index_path) != vision["vision_index_manifest_sha256"]:
        raise ValueError("Derived vision-index SHA no longer matches artifact")
    index = base.read_json(index_path)
    if (
        index.get("schema_version") != 2
        or index.get("index_kind")
        != "pixel_verified_vision_batch_aliases"
    ):
        raise ValueError(
            "All variant/control artifacts must use one derived alias index"
        )
    derivation = index.get("derivation")
    source_index = (
        derivation.get("source_index")
        if isinstance(derivation, dict)
        else None
    )
    if not isinstance(source_index, dict):
        raise ValueError("Derived vision index has no source-index provenance")
    source_index_sha = source_index.get("sha256")
    source_case_count = source_index.get("case_count")
    if (
        not isinstance(source_index_sha, str)
        or SHA256_RE.fullmatch(source_index_sha) is None
        or isinstance(source_case_count, bool)
        or not isinstance(source_case_count, int)
        or source_case_count <= 0
    ):
        raise ValueError("Derived vision index source inventory is invalid")
    index_cases = index.get("cases")
    if (
        not isinstance(index_cases, list)
        or source_case_count > len(index_cases)
    ):
        raise ValueError("Derived vision index cases inventory is incomplete")

    selected_cases = vision.get("selected_cases")
    selected_indices = vision.get("selected_indices")
    if (
        not isinstance(selected_cases, list)
        or not isinstance(selected_indices, list)
        or len(selected_cases) != len(selected_indices)
        or not selected_cases
    ):
        raise ValueError("Vision provenance has no selected case inventory")
    target_manifest_sha = str(vision["video_manifest_sha256"])
    for pair_index, (case, selected_index) in enumerate(
        zip(selected_cases, selected_indices, strict=True)
    ):
        if not isinstance(case, dict):
            raise ValueError(f"Selected vision case {pair_index} is invalid")
        pixel_sha = case.get("pixel_sha256")
        if (
            case.get("manifest_sha256") != target_manifest_sha
            or case.get("pair_index") != pair_index
            or case.get("index") != selected_index
            or not isinstance(pixel_sha, str)
            or SHA256_RE.fullmatch(pixel_sha) is None
        ):
            raise ValueError(
                f"Selected vision case {pair_index} differs from target "
                "manifest/pair/index provenance"
            )
        exact_matches = [
            (ordinal, indexed)
            for ordinal, indexed in enumerate(index_cases)
            if indexed == case
        ]
        if len(exact_matches) != 1:
            raise ValueError(
                f"Selected vision case {pair_index} is not uniquely present "
                "in the derived index"
            )
        ordinal, _ = exact_matches[0]
        alias = case.get("alias_provenance")
        if alias is None:
            if ordinal >= source_case_count or selected_index != ordinal:
                raise ValueError(
                    f"Selected source case {pair_index} is outside the "
                    "authenticated original source inventory"
                )
            continue
        if not isinstance(alias, dict):
            raise ValueError(f"Selected alias case {pair_index} is invalid")
        expected_alias = {
            "match_kind": "exact_pair_index_and_pixel_sha256",
            "source_index_sha256": source_index_sha,
            "source_batch_index": selected_index,
            "target_manifest_sha256": target_manifest_sha,
            "pair_index": pair_index,
            "pixel_sha256": pixel_sha,
        }
        for field, expected in expected_alias.items():
            if alias.get(field) != expected:
                raise ValueError(
                    f"Selected alias case {pair_index} has incomplete "
                    f"{field} provenance"
                )


def decorate_plan(
    plan: Mapping[str, Any],
    *,
    variant_label: str,
    replacements: Mapping[str, str],
    baseline_shards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decorated = copy.deepcopy(dict(plan))
    decorated["variant"] = {
        "label": variant_label,
        "fixed_shard_replacements": {
            shard: {
                "baseline_model_id": baseline_shards[shard]["model_id"],
                "replacement_model_id": model_id,
            }
            for shard, model_id in replacements.items()
        },
        "unchanged_contract": (
            "Only target model IDs change; graphs, layer ranges, IO contracts, "
            "batch order, and cache streams remain frozen."
        ),
    }
    return decorated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel_manifest", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument(
        "--fixed-shard-model",
        action="append",
        required=True,
        metavar="SHARD=MODEL_ID",
        help="Replace p2 and/or p3 by an exact AI Hub target-model ID",
    )
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--name-prefix", default="cosmos_exact_panel")
    parser.add_argument("--submit-missing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def execute_variant(
    args: argparse.Namespace,
    *,
    hub: Any,
) -> dict[str, Any]:
    if SAFE_LABEL_RE.fullmatch(args.variant_label) is None:
        raise ValueError(f"Unsafe variant label: {args.variant_label!r}")
    original_shards = base.FIXED_SHARDS
    baseline_shards = copy.deepcopy(original_shards)
    replacements = parse_replacements(
        args.fixed_shard_model,
        replaceable_shards(baseline_shards),
    )
    variant_shards = copy.deepcopy(baseline_shards)
    for shard, model_id in replacements.items():
        variant_shards[shard]["model_id"] = model_id

    original_build_plan = base.build_plan
    original_condition = base.derive_condition_label
    original_binding_validator = base.validate_alias_provenance
    base.FIXED_SHARDS = variant_shards
    try:
        def variant_condition(manifest: Mapping[str, Any]) -> str:
            return (
                f"{original_condition(manifest)}_{args.variant_label}"
            )

        def variant_build_plan(*positional: Any, **keyword: Any) -> dict[str, Any]:
            plan = original_build_plan(*positional, **keyword)
            return decorate_plan(
                plan,
                variant_label=args.variant_label,
                replacements=replacements,
                baseline_shards=baseline_shards,
            )

        base.derive_condition_label = variant_condition
        base.build_plan = variant_build_plan
        base.validate_alias_provenance = validate_derived_index_binding
        base_args = argparse.Namespace(
            panel_manifest=args.panel_manifest,
            work_dir=args.work_dir,
            benchmark_manifest=args.benchmark_manifest,
            candidate=args.candidate,
            name_prefix=(
                f"{args.name_prefix}_{args.variant_label}"
            ),
            submit_missing=bool(args.submit_missing),
            preflight_only=bool(args.preflight_only),
        )
        result = base.execute(base_args, hub=hub)
        result["variant_label"] = args.variant_label
        result["fixed_shard_replacements"] = replacements
        return result
    finally:
        base.FIXED_SHARDS = original_shards
        base.build_plan = original_build_plan
        base.derive_condition_label = original_condition
        base.validate_alias_provenance = original_binding_validator


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import json
    import qai_hub as hub

    result = execute_variant(args, hub=hub)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
