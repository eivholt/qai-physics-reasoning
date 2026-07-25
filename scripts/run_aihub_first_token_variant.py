#!/usr/bin/env python3
"""Run the audited first-token chain with explicit fixed-shard replacements.

This thin entry point preserves the validation, persistence, and scheduling
semantics of ``run_aihub_first_token_chain.py`` while allowing an experiment
to replace P2, P3, and/or P4 by exact AI Hub target-model IDs.  Each variant
must use its own work directory and a safe label, which is appended to the
condition name and therefore to submitted job names.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import scripts.run_aihub_first_token_chain as base
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_aihub_first_token_chain as base  # type: ignore[no-redef]

SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MODEL_ID_RE = re.compile(r"^m[a-z0-9]{8}$")
REPLACEABLE_SHARDS = ("p2", "p3", "p4")


def parse_replacements(
    values: Sequence[str],
    fixed_shards: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Fixed-shard replacement must be SHARD=MODEL_ID: {value!r}"
            )
        shard, model_id = value.split("=", 1)
        if shard not in REPLACEABLE_SHARDS or shard not in fixed_shards:
            raise ValueError(
                f"Unsupported replacement shard {shard!r}; choose from "
                f"{list(REPLACEABLE_SHARDS)!r}"
            )
        if shard in replacements:
            raise ValueError(f"Duplicate replacement shard: {shard}")
        if MODEL_ID_RE.fullmatch(model_id) is None:
            raise ValueError(f"Invalid AI Hub target-model ID: {model_id!r}")
        if model_id == fixed_shards[shard]["model_id"]:
            raise ValueError(
                f"Replacement for {shard} equals the baseline target model"
            )
        replacements[shard] = model_id
    if not replacements:
        raise ValueError("At least one fixed-shard replacement is required")
    return replacements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_artifact_dir", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--seed-p1-screen", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument(
        "--fixed-shard-model",
        action="append",
        required=True,
        metavar="SHARD=MODEL_ID",
    )
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def base_argv(args: argparse.Namespace) -> list[str]:
    result = [
        "run_aihub_first_token_chain.py",
        str(args.input_artifact_dir),
        str(args.work_dir),
        "--seed-p1-screen",
        str(args.seed_p1_screen),
        "--benchmark-manifest",
        str(args.benchmark_manifest),
        "--case-id",
        args.case_id,
        "--probe-id",
        args.probe_id,
    ]
    for candidate in args.candidate:
        result.extend(["--candidate", candidate])
    if args.preflight_only:
        result.append("--preflight-only")
    return result


def main() -> None:
    args = build_parser().parse_args()
    if SAFE_LABEL_RE.fullmatch(args.variant_label) is None:
        raise ValueError(f"Unsafe variant label: {args.variant_label!r}")
    replacements = parse_replacements(
        args.fixed_shard_model,
        base.FIXED_SHARDS,
    )
    original_shards = copy.deepcopy(base.FIXED_SHARDS)
    original_condition = base.derive_condition_label
    original_argv = sys.argv
    try:
        for shard, model_id in replacements.items():
            base.FIXED_SHARDS[shard]["model_id"] = model_id

        def variant_condition(manifest: Mapping[str, Any]) -> str:
            return (
                f"{original_condition(manifest)}_"
                f"{args.variant_label}"
            )

        base.derive_condition_label = variant_condition
        sys.argv = base_argv(args)
        base.main()
    finally:
        base.FIXED_SHARDS.clear()
        base.FIXED_SHARDS.update(original_shards)
        base.derive_condition_label = original_condition
        sys.argv = original_argv


if __name__ == "__main__":
    main()
