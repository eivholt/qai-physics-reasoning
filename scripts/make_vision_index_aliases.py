#!/usr/bin/env python3
"""Create pixel-verified aliases for a frozen vision-output batch index.

Prompt variants can contain byte-identical video pixels while having different
video-manifest hashes.  The P1 input builder deliberately binds vision rows to
the exact video-manifest hash, so a prompt variant needs an explicit alias
record before an existing AI Hub vision result can be reused.

This tool fails closed.  Every target pair must match exactly one source case
by ``(pair_index, pixel_sha256)``; the target raw pixel file must still exist
and match its recorded size and SHA-256; and the source index must have a
strict one-row-per-batch inventory.  The derived index records both sides of
every match and never relies on scene names or list position alone.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def lowercase_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def nonnegative_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def positive_int(value: Any, description: str) -> int:
    parsed = nonnegative_int(value, description)
    if parsed == 0:
        raise ValueError(f"{description} must be a positive integer")
    return parsed


def safe_relative_file(
    manifest_path: Path,
    value: Any,
    description: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} must remain inside the manifest")
    root = manifest_path.parent.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} escapes the manifest directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def validate_source_cases(
    source_index: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    raw_cases = source_index.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Source vision index must contain a non-empty cases list")

    normalized: list[dict[str, Any]] = []
    by_match_key: dict[tuple[int, str], dict[str, Any]] = {}
    seen_case_ids: set[str] = set()
    seen_manifest_pairs: set[tuple[str, int]] = set()
    for expected_index, raw_case in enumerate(raw_cases):
        description = f"source cases[{expected_index}]"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{description} must be an object")
        index = nonnegative_int(raw_case.get("index"), f"{description}.index")
        if index != expected_index:
            raise ValueError(
                "Source cases must be a one-to-one batch inventory: "
                f"{description}.index is {index}, expected {expected_index}"
            )
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{description}.case_id must be a non-empty string")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate source case_id: {case_id}")
        seen_case_ids.add(case_id)
        manifest_sha = lowercase_sha256(
            raw_case.get("manifest_sha256"),
            f"{description}.manifest_sha256",
        )
        pair_index = nonnegative_int(
            raw_case.get("pair_index"),
            f"{description}.pair_index",
        )
        manifest_pair = (manifest_sha, pair_index)
        if manifest_pair in seen_manifest_pairs:
            raise ValueError(
                f"Duplicate source manifest/pair binding: {manifest_pair}"
            )
        seen_manifest_pairs.add(manifest_pair)
        pixel_sha = lowercase_sha256(
            raw_case.get("pixel_sha256"),
            f"{description}.pixel_sha256",
        )
        match_key = (pair_index, pixel_sha)
        if match_key in by_match_key:
            other = by_match_key[match_key]
            raise ValueError(
                "Ambiguous source pixel binding for "
                f"pair {pair_index}/{pixel_sha}: "
                f"{other['case_id']!r} and {case_id!r}"
            )
        record = {
            **copy.deepcopy(raw_case),
            "case_id": case_id,
            "index": index,
            "manifest_sha256": manifest_sha,
            "pair_index": pair_index,
            "pixel_sha256": pixel_sha,
        }
        normalized.append(record)
        by_match_key[match_key] = record
    return normalized, by_match_key


def validate_target_manifest(
    label: str,
    manifest_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    if LABEL_RE.fullmatch(label) is None:
        raise ValueError(f"Unsafe target label: {label!r}")
    resolved = manifest_path.expanduser().resolve()
    manifest = read_json_object(resolved, f"target manifest {label!r}")
    manifest_sha = sha256_file(resolved)
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Target manifest {label!r} has no visual pairs")

    normalized: list[dict[str, Any]] = []
    seen_pixel_files: set[Path] = set()
    for expected_pair_index, pair in enumerate(pairs):
        description = f"target {label!r} pairs[{expected_pair_index}]"
        if not isinstance(pair, dict):
            raise ValueError(f"{description} must be an object")
        pair_index = nonnegative_int(
            pair.get("pair_index"),
            f"{description}.pair_index",
        )
        if pair_index != expected_pair_index:
            raise ValueError(
                f"{description}.pair_index is {pair_index}; "
                f"expected {expected_pair_index}"
            )
        pixel = pair.get("pixel_values")
        if not isinstance(pixel, dict):
            raise ValueError(f"{description}.pixel_values must be an object")
        pixel_sha = lowercase_sha256(
            pixel.get("sha256"),
            f"{description}.pixel_values.sha256",
        )
        pixel_bytes = positive_int(
            pixel.get("bytes"),
            f"{description}.pixel_values.bytes",
        )
        pixel_path = safe_relative_file(
            resolved,
            pixel.get("file"),
            f"{description}.pixel_values.file",
        )
        if pixel_path in seen_pixel_files:
            raise ValueError(
                f"Target {label!r} reuses pixel file {pixel_path.name!r}"
            )
        seen_pixel_files.add(pixel_path)
        actual_bytes = pixel_path.stat().st_size
        if actual_bytes != pixel_bytes:
            raise ValueError(
                f"{description} byte count changed: {actual_bytes} != "
                f"{pixel_bytes}"
            )
        actual_sha = sha256_file(pixel_path)
        if actual_sha != pixel_sha:
            raise ValueError(
                f"{description} pixel SHA changed: {actual_sha} != {pixel_sha}"
            )
        normalized.append(
            {
                "pair_index": pair_index,
                "pixel_sha256": pixel_sha,
                "pixel_bytes": pixel_bytes,
            }
        )
    return manifest_sha, normalized


def build_alias_index(
    source_index_path: Path,
    targets: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    source_path = source_index_path.expanduser().resolve()
    source_index = read_json_object(source_path, "source vision index")
    source_sha = sha256_file(source_path)
    source_cases, by_match_key = validate_source_cases(source_index)
    if not targets:
        raise ValueError("At least one target manifest is required")

    seen_labels: set[str] = set()
    seen_target_shas: set[str] = set()
    existing_manifest_shas = {
        str(case["manifest_sha256"]) for case in source_cases
    }
    alias_cases: list[dict[str, Any]] = []
    alias_sets: list[dict[str, Any]] = []
    for label, target_path in targets:
        if label in seen_labels:
            raise ValueError(f"Duplicate target label: {label!r}")
        seen_labels.add(label)
        target_sha, target_pairs = validate_target_manifest(label, target_path)
        if target_sha in seen_target_shas:
            raise ValueError(
                f"Duplicate target manifest SHA for target {label!r}"
            )
        seen_target_shas.add(target_sha)

        if target_sha in existing_manifest_shas:
            raise ValueError(
                f"Target {label!r} is already bound by the source index; "
                "an alias is unnecessary"
            )

        mappings: list[dict[str, Any]] = []
        for target_pair in target_pairs:
            pair_index = int(target_pair["pair_index"])
            pixel_sha = str(target_pair["pixel_sha256"])
            source_case = by_match_key.get((pair_index, pixel_sha))
            if source_case is None:
                raise ValueError(
                    f"Target {label!r} pair {pair_index}/{pixel_sha} has no "
                    "exact source-index match"
                )
            alias_case_id = f"alias:{label}:{pair_index}"
            alias_provenance = {
                "match_kind": "exact_pair_index_and_pixel_sha256",
                "source_index_sha256": source_sha,
                "source_case_id": source_case["case_id"],
                "source_manifest_sha256": source_case["manifest_sha256"],
                "source_batch_index": source_case["index"],
                "target_label": label,
                "target_manifest_sha256": target_sha,
                "pair_index": pair_index,
                "pixel_sha256": pixel_sha,
            }
            alias_cases.append(
                {
                    "case_id": alias_case_id,
                    "index": source_case["index"],
                    "manifest_sha256": target_sha,
                    "pair_index": pair_index,
                    "pixel_sha256": pixel_sha,
                    "alias_provenance": alias_provenance,
                }
            )
            mappings.append(alias_provenance)
        alias_sets.append(
            {
                "target_label": label,
                "target_manifest_sha256": target_sha,
                "pair_count": len(target_pairs),
                "pixel_files_verified": True,
                "mappings": mappings,
            }
        )

    return {
        "schema_version": 2,
        "index_kind": "pixel_verified_vision_batch_aliases",
        "reference": source_index.get("reference"),
        "geometry": copy.deepcopy(source_index.get("geometry")),
        "cases": [*source_cases, *alias_cases],
        "derivation": {
            "generator": "scripts/make_vision_index_aliases.py",
            "policy": (
                "Each alias is an exact unique match on "
                "(pair_index, pixel_sha256); target raw files were rehashed."
            ),
            "source_index": {
                "sha256": source_sha,
                "schema_version": source_index.get("schema_version"),
                "case_count": len(source_cases),
                "reference": source_index.get("reference"),
            },
            "alias_case_count": len(alias_cases),
            "targets": alias_sets,
        },
        "security": {
            "contains_credentials": False,
            "contains_local_paths": False,
        },
    }


def parse_target(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError(
            "Target must use LABEL=/path/to/video_npu_manifest.json"
        )
    if LABEL_RE.fullmatch(label) is None:
        raise argparse.ArgumentTypeError(f"Unsafe target label: {label!r}")
    return label, Path(raw_path)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        existing = read_json_object(resolved, "existing derived vision index")
        if existing == value:
            return
        raise FileExistsError(
            f"Refusing to overwrite a different derived index: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite an unresolved temporary file: {temporary}"
        )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_index", type=Path)
    parser.add_argument("output_index", type=Path)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        type=parse_target,
        metavar="LABEL=MANIFEST",
        help="Prompt-variant manifest to alias (repeatable)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the derivation summary without writing output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    derived = build_alias_index(args.source_index, args.target)
    summary = {
        "status": "passed" if args.check_only else "written",
        "source_index_sha256": derived["derivation"]["source_index"]["sha256"],
        "alias_case_count": derived["derivation"]["alias_case_count"],
        "targets": [
            {
                "target_label": record["target_label"],
                "target_manifest_sha256": record["target_manifest_sha256"],
                "pair_count": record["pair_count"],
            }
            for record in derived["derivation"]["targets"]
        ],
    }
    if not args.check_only:
        write_json_atomic(args.output_index, derived)
        summary["output_index"] = str(args.output_index.expanduser().resolve())
        summary["output_index_sha256"] = sha256_file(
            args.output_index.expanduser().resolve()
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
