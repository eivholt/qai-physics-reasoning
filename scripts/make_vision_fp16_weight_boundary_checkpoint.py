#!/usr/bin/env python3
"""Compose FP16 vision weights with the proven A16 boundary layout.

Two independently validated checkpoints are required:

* a W-FP16/A16 checkpoint from ``make_vision_fp16_weight_checkpoint.py``;
* a W8/A16-boundary checkpoint from
  ``make_vision_a16_boundary_checkpoint.py``.

The FP16-weight checkpoint may be a vision-only donor. The complete boundary
checkpoint is the output base so its text-model artifacts remain byte-for-byte
unchanged. The donor supplies its ``w4a16`` exporter arguments, 205 canonical
FLOAT16 parameter encodings, and weight-transform provenance. The boundary
checkpoint supplies nine A16 boundaries and 925 canonical FLOAT16 internal
activations.

The transform is deterministic and fail-closed. It validates paired
224x384/20-sample calibration, both source provenance chains, identical raw
vision ONNX bytes, identical tensor names/order and encoding metadata, and the
boundary topology before creating output. The destination is assembled in a
sibling temporary directory and atomically renamed into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

if __package__:
    from .make_vision_a16_boundary_checkpoint import (
        ALL_FP16_MARKER_FILENAME,
        MARKER_FILENAME as BOUNDARY_MARKER_FILENAME,
        PROVENANCE_FIELD as BOUNDARY_PROVENANCE_FIELD,
        RESTORED_ACTIVATION_NAMES,
        _validate_a16_entry,
        _validate_graph_contract,
    )
    from .make_vision_fp16_weight_checkpoint import (
        AIMET_ENCODING_VERSION,
        ARGS_FILENAME,
        ENCODINGS_FILENAME,
        EXPECTED_ACTIVATION_COUNT,
        EXPECTED_IMAGE_SIZE,
        EXPECTED_PARAMETER_COUNT,
        FLOAT16_ENCODING,
        MARKER_FILENAME as WEIGHT_MARKER_FILENAME,
        MODEL_FILENAME,
        PROVENANCE_FIELD as WEIGHT_PROVENANCE_FIELD,
        _canonical_sha256,
        _entries_by_name,
        _hardlink_or_copy,
        _json_bytes,
        _load_json_object,
        sha256_file,
    )
else:
    from make_vision_a16_boundary_checkpoint import (  # type: ignore
        ALL_FP16_MARKER_FILENAME,
        MARKER_FILENAME as BOUNDARY_MARKER_FILENAME,
        PROVENANCE_FIELD as BOUNDARY_PROVENANCE_FIELD,
        RESTORED_ACTIVATION_NAMES,
        _validate_a16_entry,
        _validate_graph_contract,
    )
    from make_vision_fp16_weight_checkpoint import (  # type: ignore
        AIMET_ENCODING_VERSION,
        ARGS_FILENAME,
        ENCODINGS_FILENAME,
        EXPECTED_ACTIVATION_COUNT,
        EXPECTED_IMAGE_SIZE,
        EXPECTED_PARAMETER_COUNT,
        FLOAT16_ENCODING,
        MARKER_FILENAME as WEIGHT_MARKER_FILENAME,
        MODEL_FILENAME,
        PROVENANCE_FIELD as WEIGHT_PROVENANCE_FIELD,
        _canonical_sha256,
        _entries_by_name,
        _hardlink_or_copy,
        _json_bytes,
        _load_json_object,
        sha256_file,
    )

SOURCE_WEIGHT_EXPORT_PRECISION = "w4a16"
SOURCE_BOUNDARY_EXPORT_PRECISION = "w4"
DESTINATION_EXPORT_PRECISION = SOURCE_WEIGHT_EXPORT_PRECISION
MARKER_FILENAME = "vision_fp16_weight_a16_boundaries.json"
PROVENANCE_FIELD = "vision_fp16_weight_a16_boundary_postprocess"

EXPECTED_A16_BOUNDARY_COUNT = len(RESTORED_ACTIVATION_NAMES)
EXPECTED_FP16_INTERNAL_COUNT = (
    EXPECTED_ACTIVATION_COUNT - EXPECTED_A16_BOUNDARY_COUNT
)

REQUIRED_THIN_WEIGHT_FILES = (
    ARGS_FILENAME,
    ENCODINGS_FILENAME,
    MODEL_FILENAME,
    WEIGHT_MARKER_FILENAME,
)

REQUIRED_FULL_BOUNDARY_FILES = (
    ARGS_FILENAME,
    ENCODINGS_FILENAME,
    MODEL_FILENAME,
    "model.encodings",
    "model_dynamic.onnx",
    "model.data",
    "embedding_weights.raw",
    "source_checkpoint.json",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    BOUNDARY_MARKER_FILENAME,
    ALL_FP16_MARKER_FILENAME,
)

PRESERVED_BOUNDARY_FILES = (
    MODEL_FILENAME,
    "model.encodings",
    "model_dynamic.onnx",
    "model.data",
    "embedding_weights.raw",
    "source_checkpoint.json",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    BOUNDARY_MARKER_FILENAME,
    ALL_FP16_MARKER_FILENAME,
)


def _validate_raw_precision(
    args: dict[str, Any],
    *,
    expected: str,
    path: Path,
) -> None:
    raw_args = args.get("raw_args")
    if not isinstance(raw_args, list) or not all(
        isinstance(value, str) for value in raw_args
    ):
        raise ValueError(f"{path}: raw_args must be a list of strings")
    indices = [
        index
        for index, value in enumerate(raw_args)
        if value == "--precision"
    ]
    if len(indices) != 1:
        raise ValueError(
            f"{path}: raw_args must contain exactly one --precision flag"
        )
    index = indices[0]
    if index + 1 >= len(raw_args) or raw_args[index + 1] != expected:
        found = None if index + 1 >= len(raw_args) else raw_args[index + 1]
        raise ValueError(
            f"{path}: raw_args precision is {found!r}; expected {expected!r}"
        )


def _validate_args(
    path: Path,
    *,
    expected_precision: str,
    required_provenance_field: str,
) -> dict[str, Any]:
    args = _load_json_object(path)
    if args.get("precision") != expected_precision:
        raise ValueError(
            f"{path}: precision is {args.get('precision')!r}; expected "
            f"{expected_precision!r}"
        )
    _validate_raw_precision(args, expected=expected_precision, path=path)
    if args.get("image_size") != list(EXPECTED_IMAGE_SIZE):
        raise ValueError(
            f"{path}: image_size is {args.get('image_size')!r}; expected "
            f"{list(EXPECTED_IMAGE_SIZE)!r}"
        )
    if args.get("vision_calibration_source") != "paired_frame_manifest":
        raise ValueError(
            f"{path}: expected paired-frame calibration provenance"
        )
    manifest_path = args.get("veg_paired_calibration_manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError(
            f"{path}: paired-frame calibration manifest path is missing"
        )
    manifest_hash = args.get("veg_paired_calibration_manifest_sha256")
    if not isinstance(manifest_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", manifest_hash
    ) is None:
        raise ValueError(
            f"{path}: paired-frame calibration manifest SHA-256 is invalid"
        )
    for field in ("num_samples", "veg_num_samples"):
        if args.get(field) != 20:
            raise ValueError(
                f"{path}: {field} is {args.get(field)!r}; expected 20"
            )
    if PROVENANCE_FIELD in args:
        raise ValueError(
            f"{path}: {PROVENANCE_FIELD} already exists; source appears "
            "already combined"
        )
    if not isinstance(args.get(required_provenance_field), dict):
        raise ValueError(
            f"{path}: missing {required_provenance_field} provenance"
        )
    return args


def _validate_hash(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"[0-9a-f]{64}", value
    ) is None:
        raise ValueError(f"{path}: provenance hash {field!r} is invalid")
    return value


def _parse_encodings(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    encodings = _load_json_object(path)
    if encodings.get("version") != AIMET_ENCODING_VERSION:
        raise ValueError(
            f"{path}: encoding version is {encodings.get('version')!r}; "
            f"expected {AIMET_ENCODING_VERSION!r}"
        )
    activations = _entries_by_name(
        encodings.get("activation_encodings"),
        field="activation_encodings",
        path=path,
    )
    parameters = _entries_by_name(
        encodings.get("param_encodings"),
        field="param_encodings",
        path=path,
    )
    overlap = sorted(set(activations) & set(parameters))
    if overlap:
        raise ValueError(
            f"{path}: tensors occur in both activation and parameter "
            f"encodings: {overlap[:5]!r}"
        )
    if len(activations) != EXPECTED_ACTIVATION_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_ACTIVATION_COUNT} activation "
            f"encodings, found {len(activations)}"
        )
    if len(parameters) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_PARAMETER_COUNT} parameter "
            f"encodings, found {len(parameters)}"
        )
    metadata = {
        key: value
        for key, value in encodings.items()
        if key not in {"activation_encodings", "param_encodings"}
    }
    return encodings, activations, parameters, metadata


def _validate_weight_source_encodings(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encodings, activations, parameters, metadata = _parse_encodings(path)
    invalid_activations = [
        name
        for name, entry in activations.items()
        if entry.get("dtype") != "INT" or entry.get("bw") != 16
    ]
    if invalid_activations:
        raise ValueError(
            f"{path}: FP16-weight source must retain all "
            f"{EXPECTED_ACTIVATION_COUNT} A16 activations; first mismatch is "
            f"{invalid_activations[0]!r}"
        )
    invalid_parameters = [
        name
        for name, entry in parameters.items()
        if entry != {"name": name, **FLOAT16_ENCODING}
    ]
    if invalid_parameters:
        raise ValueError(
            f"{path}: all {EXPECTED_PARAMETER_COUNT} weight-source "
            "parameters must use canonical FLOAT16 encoding; first mismatch "
            f"is {invalid_parameters[0]!r}"
        )
    summary = {
        "activation_names": [
            entry["name"] for entry in encodings["activation_encodings"]
        ],
        "parameter_names": [
            entry["name"] for entry in encodings["param_encodings"]
        ],
        "parameter_encodings_canonical_sha256": _canonical_sha256(
            encodings["param_encodings"]
        ),
        "top_level_metadata_canonical_sha256": _canonical_sha256(metadata),
    }
    return encodings, summary


def _validate_boundary_source_encodings(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encodings, activations, parameters, metadata = _parse_encodings(path)
    boundary_names = set(RESTORED_ACTIVATION_NAMES)
    actual_a16_names = {
        name
        for name, entry in activations.items()
        if entry.get("dtype") == "INT" and entry.get("bw") == 16
    }
    if actual_a16_names != boundary_names:
        missing = sorted(boundary_names - actual_a16_names)
        extra = sorted(actual_a16_names - boundary_names)
        raise ValueError(
            f"{path}: A16 activation set does not match the proven nine "
            f"boundaries; missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    for name in RESTORED_ACTIVATION_NAMES:
        _validate_a16_entry(activations[name], path=path)
    invalid_internal = [
        name
        for name, entry in activations.items()
        if name not in boundary_names
        and entry != {"name": name, **FLOAT16_ENCODING}
    ]
    if invalid_internal:
        raise ValueError(
            f"{path}: all {EXPECTED_FP16_INTERNAL_COUNT} internal "
            "activations must use canonical FLOAT16 encoding; first "
            f"mismatch is {invalid_internal[0]!r}"
        )
    invalid_parameters = [
        name
        for name, entry in parameters.items()
        if entry.get("dtype") != "INT" or entry.get("bw") != 8
    ]
    if invalid_parameters:
        raise ValueError(
            f"{path}: all {EXPECTED_PARAMETER_COUNT} boundary-source "
            f"parameters must be W8 INT; first mismatch is "
            f"{invalid_parameters[0]!r}"
        )
    summary = {
        "activation_names": [
            entry["name"] for entry in encodings["activation_encodings"]
        ],
        "parameter_names": [
            entry["name"] for entry in encodings["param_encodings"]
        ],
        "activation_encodings_canonical_sha256": _canonical_sha256(
            encodings["activation_encodings"]
        ),
        "source_parameter_encodings_canonical_sha256": _canonical_sha256(
            encodings["param_encodings"]
        ),
        "a16_boundary_encodings_canonical_sha256": _canonical_sha256(
            [activations[name] for name in RESTORED_ACTIVATION_NAMES]
        ),
        "top_level_metadata_canonical_sha256": _canonical_sha256(metadata),
    }
    return encodings, summary


def _calibration_contract(args: dict[str, Any]) -> dict[str, Any]:
    return {
        name: args.get(name)
        for name in (
            "image_size",
            "vision_calibration_source",
            "veg_paired_calibration_manifest",
            "veg_paired_calibration_manifest_sha256",
            "veg_num_samples",
            "num_samples",
        )
    }


def _validate_matching_sources(
    weight_args: dict[str, Any],
    boundary_args: dict[str, Any],
    weight_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
) -> dict[str, Any]:
    calibration = _calibration_contract(weight_args)
    if calibration != _calibration_contract(boundary_args):
        raise ValueError(
            "FP16-weight and boundary calibration provenance does not match"
        )
    if weight_summary["activation_names"] != (
        boundary_summary["activation_names"]
    ):
        raise ValueError(
            "FP16-weight and boundary activation tensor names/order do not "
            "match"
        )
    if weight_summary["parameter_names"] != (
        boundary_summary["parameter_names"]
    ):
        raise ValueError(
            "FP16-weight and boundary parameter tensor names/order do not "
            "match"
        )
    if weight_summary["top_level_metadata_canonical_sha256"] != (
        boundary_summary["top_level_metadata_canonical_sha256"]
    ):
        raise ValueError(
            "FP16-weight and boundary top-level encoding metadata differs"
        )
    return calibration


def _validate_weight_provenance(
    args_path: Path,
    marker_path: Path,
    *,
    args: dict[str, Any],
    args_hash: str,
    encodings_hash: str,
    model_hash: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    provenance = args[WEIGHT_PROVENANCE_FIELD]
    expected_fields = {
        "activation_encoding_count": EXPECTED_ACTIVATION_COUNT,
        "activation_quantization": "A16 integer",
        "destination_parameter_quantization": "FLOAT16",
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "marker": WEIGHT_MARKER_FILENAME,
        "parameter_encoding_count": EXPECTED_PARAMETER_COUNT,
        "source_parameter_quantization": "W8 integer",
    }
    mismatches = [
        name
        for name, expected in expected_fields.items()
        if provenance.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{args_path}: FP16-weight provenance differs in: "
            + ", ".join(mismatches)
        )
    if (
        _validate_hash(
            provenance.get("output_encodings_sha256"),
            field="output_encodings_sha256",
            path=args_path,
        )
        != encodings_hash
    ):
        raise ValueError(
            f"{args_path}: FP16-weight provenance does not identify encodings"
        )
    if (
        _validate_hash(
            provenance.get("source_model_sha256"),
            field="source_model_sha256",
            path=args_path,
        )
        != model_hash
    ):
        raise ValueError(
            f"{args_path}: FP16-weight provenance does not identify ONNX"
        )

    marker = _load_json_object(marker_path)
    if marker.get("schema_version") != 1:
        raise ValueError(f"{marker_path}: unsupported weight marker schema")
    counts = marker.get("counts")
    if not isinstance(counts, dict) or any(
        counts.get(name) != expected
        for name, expected in {
            "activation_encodings": EXPECTED_ACTIVATION_COUNT,
            "parameters_converted_int8_to_float16": (
                EXPECTED_PARAMETER_COUNT
            ),
        }.items()
    ):
        raise ValueError(f"{marker_path}: weight count contract does not match")
    hashes = marker.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError(f"{marker_path}: weight hash provenance is missing")
    expected_hashes = {
        "output_args_sha256": args_hash,
        "output_encodings_sha256": encodings_hash,
        "source_model_sha256": model_hash,
        "output_model_sha256": model_hash,
        "parameter_names_canonical_sha256": _canonical_sha256(
            summary["parameter_names"]
        ),
        "top_level_metadata_canonical_sha256": summary[
            "top_level_metadata_canonical_sha256"
        ],
    }
    marker_mismatches = [
        name
        for name, expected in expected_hashes.items()
        if hashes.get(name) != expected
    ]
    if marker_mismatches:
        raise ValueError(
            f"{marker_path}: weight source hashes differ in: "
            + ", ".join(marker_mismatches)
        )
    return marker


def _validate_boundary_provenance(
    args_path: Path,
    marker_path: Path,
    *,
    args: dict[str, Any],
    args_hash: str,
    encodings_hash: str,
    model_hash: str,
    all_fp16_marker_hash: str,
    summary: dict[str, Any],
    graph_summary: dict[str, Any],
) -> dict[str, Any]:
    provenance = args[BOUNDARY_PROVENANCE_FIELD]
    expected_fields = {
        "schema_version": 1,
        "marker": BOUNDARY_MARKER_FILENAME,
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "restored_a16_count": EXPECTED_A16_BOUNDARY_COUNT,
        "remaining_float16_count": EXPECTED_FP16_INTERNAL_COUNT,
        "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
    }
    mismatches = [
        name
        for name, expected in expected_fields.items()
        if provenance.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{args_path}: boundary provenance differs in: "
            + ", ".join(mismatches)
        )
    if (
        _validate_hash(
            provenance.get("output_encodings_sha256"),
            field="output_encodings_sha256",
            path=args_path,
        )
        != encodings_hash
        or _validate_hash(
            provenance.get("vision_onnx_sha256"),
            field="vision_onnx_sha256",
            path=args_path,
        )
        != model_hash
    ):
        raise ValueError(
            f"{args_path}: boundary provenance does not identify checkpoint"
        )

    marker = _load_json_object(marker_path)
    if marker.get("schema_version") != 1:
        raise ValueError(f"{marker_path}: unsupported boundary marker schema")
    expected_precision = {
        "checkpoint_export_precision": SOURCE_BOUNDARY_EXPORT_PRECISION,
        "text_weights": "W4",
        "vision_parameters": "W8",
        "restored_boundaries": "A16",
        "remaining_vision_activations": "FLOAT16",
    }
    if marker.get("precision_contract") != expected_precision:
        raise ValueError(f"{marker_path}: boundary precision contract differs")
    counts = marker.get("counts")
    if not isinstance(counts, dict) or any(
        counts.get(name) != expected
        for name, expected in {
            "activation_count": EXPECTED_ACTIVATION_COUNT,
            "restored_a16_count": EXPECTED_A16_BOUNDARY_COUNT,
            "remaining_float16_count": EXPECTED_FP16_INTERNAL_COUNT,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
        }.items()
    ):
        raise ValueError(
            f"{marker_path}: boundary count contract does not match"
        )
    if marker.get("calibration") != _calibration_contract(args):
        raise ValueError(
            f"{marker_path}: boundary calibration does not match args"
        )
    if marker.get("graph_contract") != graph_summary:
        raise ValueError(
            f"{marker_path}: boundary graph contract does not match ONNX"
        )
    if marker.get("restored_from_w4a16_byte_for_byte") != list(
        RESTORED_ACTIVATION_NAMES
    ):
        raise ValueError(
            f"{marker_path}: restored boundary tensor list does not match"
        )
    hashes = marker.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError(f"{marker_path}: boundary hash provenance is missing")
    expected_hashes = {
        "output_args_sha256": args_hash,
        "output_encodings_sha256": encodings_hash,
        "output_onnx_sha256": model_hash,
        "all_fp16_onnx_sha256": model_hash,
        "a16_source_onnx_sha256": model_hash,
        "all_fp16_marker_sha256": all_fp16_marker_hash,
        "parameter_encodings_canonical_sha256": summary[
            "source_parameter_encodings_canonical_sha256"
        ],
    }
    marker_mismatches = [
        name
        for name, expected in expected_hashes.items()
        if hashes.get(name) != expected
    ]
    if marker_mismatches:
        raise ValueError(
            f"{marker_path}: boundary source hashes differ in: "
            + ", ".join(marker_mismatches)
        )
    return marker


def _compose_encodings(
    weight_encodings: dict[str, Any],
    boundary_encodings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = dict(weight_encodings)
    output["activation_encodings"] = copy_activations = [
        dict(entry) for entry in boundary_encodings["activation_encodings"]
    ]
    if output["param_encodings"] != weight_encodings["param_encodings"]:
        raise AssertionError("FLOAT16 parameter encodings changed")
    if copy_activations != boundary_encodings["activation_encodings"]:
        raise AssertionError("Boundary activation encodings changed")
    summary = {
        "activation_count": EXPECTED_ACTIVATION_COUNT,
        "a16_boundary_count": EXPECTED_A16_BOUNDARY_COUNT,
        "float16_internal_activation_count": EXPECTED_FP16_INTERNAL_COUNT,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "destination_activation_types": {
            "INT16": EXPECTED_A16_BOUNDARY_COUNT,
            "FLOAT16": EXPECTED_FP16_INTERNAL_COUNT,
        },
        "destination_parameter_types": {
            "FLOAT16": EXPECTED_PARAMETER_COUNT
        },
        "activation_encodings_canonical_sha256": _canonical_sha256(
            output["activation_encodings"]
        ),
        "parameter_encodings_canonical_sha256": _canonical_sha256(
            output["param_encodings"]
        ),
    }
    return output, summary


def create_vision_fp16_weight_boundary_checkpoint(
    fp16_weight_source: Path,
    boundary_source: Path,
    destination: Path,
    *,
    copy_function: Callable[[str, str], str] = _hardlink_or_copy,
) -> Path:
    """Create the combined vision checkpoint transactionally."""
    fp16_weight_source = fp16_weight_source.expanduser().resolve()
    boundary_source = boundary_source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    for label, source in (
        ("FP16-weight", fp16_weight_source),
        ("boundary", boundary_source),
    ):
        if not source.is_dir():
            raise FileNotFoundError(
                f"{label} checkpoint is not a directory: {source}"
            )
    if fp16_weight_source == boundary_source:
        raise ValueError("FP16-weight and boundary sources must be different")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite: {destination}"
        )
    if any(
        destination == source or source in destination.parents
        for source in (fp16_weight_source, boundary_source)
    ):
        raise ValueError("Destination must not be inside either source")
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary destination already exists: {temporary}"
        )

    weight_required = [
        fp16_weight_source / name for name in REQUIRED_THIN_WEIGHT_FILES
    ]
    boundary_required = [
        boundary_source / name for name in REQUIRED_FULL_BOUNDARY_FILES
    ]
    missing = sorted(
        str(path)
        for path in (*weight_required, *boundary_required)
        if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Source checkpoint is incomplete. Missing: " + ", ".join(missing)
        )
    if (fp16_weight_source / MARKER_FILENAME).exists() or (
        boundary_source / MARKER_FILENAME
    ).exists():
        raise ValueError(
            f"A source already contains {MARKER_FILENAME}; refusing to "
            "recombine"
        )

    weight_args_path = fp16_weight_source / ARGS_FILENAME
    boundary_args_path = boundary_source / ARGS_FILENAME
    weight_encodings_path = fp16_weight_source / ENCODINGS_FILENAME
    boundary_encodings_path = boundary_source / ENCODINGS_FILENAME
    weight_model_path = fp16_weight_source / MODEL_FILENAME
    boundary_model_path = boundary_source / MODEL_FILENAME
    weight_marker_path = fp16_weight_source / WEIGHT_MARKER_FILENAME
    boundary_marker_path = boundary_source / BOUNDARY_MARKER_FILENAME

    weight_args = _validate_args(
        weight_args_path,
        expected_precision=SOURCE_WEIGHT_EXPORT_PRECISION,
        required_provenance_field=WEIGHT_PROVENANCE_FIELD,
    )
    boundary_args = _validate_args(
        boundary_args_path,
        expected_precision=SOURCE_BOUNDARY_EXPORT_PRECISION,
        required_provenance_field=BOUNDARY_PROVENANCE_FIELD,
    )
    weight_encodings, weight_summary = (
        _validate_weight_source_encodings(weight_encodings_path)
    )
    boundary_encodings, boundary_summary = (
        _validate_boundary_source_encodings(boundary_encodings_path)
    )
    calibration = _validate_matching_sources(
        weight_args,
        boundary_args,
        weight_summary,
        boundary_summary,
    )

    weight_model_hash = sha256_file(weight_model_path)
    boundary_model_hash = sha256_file(boundary_model_path)
    if weight_model_hash != boundary_model_hash:
        raise ValueError(
            "FP16-weight and boundary vision ONNX byte hashes do not match"
        )
    graph_summary = _validate_graph_contract(
        weight_model_path,
        set(weight_summary["activation_names"]),
    )

    weight_args_hash = sha256_file(weight_args_path)
    boundary_args_hash = sha256_file(boundary_args_path)
    weight_encodings_hash = sha256_file(weight_encodings_path)
    boundary_encodings_hash = sha256_file(boundary_encodings_path)
    weight_marker_hash = sha256_file(weight_marker_path)
    boundary_marker_hash = sha256_file(boundary_marker_path)
    all_fp16_marker_hash = sha256_file(
        boundary_source / ALL_FP16_MARKER_FILENAME
    )
    preserved_boundary_hashes = {
        name: sha256_file(boundary_source / name)
        for name in PRESERVED_BOUNDARY_FILES
    }
    _validate_weight_provenance(
        weight_args_path,
        weight_marker_path,
        args=weight_args,
        args_hash=weight_args_hash,
        encodings_hash=weight_encodings_hash,
        model_hash=weight_model_hash,
        summary=weight_summary,
    )
    _validate_boundary_provenance(
        boundary_args_path,
        boundary_marker_path,
        args=boundary_args,
        args_hash=boundary_args_hash,
        encodings_hash=boundary_encodings_hash,
        model_hash=boundary_model_hash,
        all_fp16_marker_hash=all_fp16_marker_hash,
        summary=boundary_summary,
        graph_summary=graph_summary,
    )

    output_encodings, output_summary = _compose_encodings(
        weight_encodings,
        boundary_encodings,
    )
    output_encodings_bytes = _json_bytes(output_encodings)
    output_encodings_hash = hashlib.sha256(
        output_encodings_bytes
    ).hexdigest()

    output_args = dict(weight_args)
    output_args[PROVENANCE_FIELD] = {
        "schema_version": 1,
        "scheme": (
            "FLOAT16 vision parameters, nine A16 compiler/Genie boundaries, "
            "925 FLOAT16 internal vision activations"
        ),
        "marker": MARKER_FILENAME,
        "fp16_weight_source_checkpoint_name": fp16_weight_source.name,
        "boundary_source_checkpoint_name": boundary_source.name,
        "checkpoint_export_precision": DESTINATION_EXPORT_PRECISION,
        "exporter_selection_reason": (
            "Use the already linkable w4a16 FP16-weight vision exporter "
            "while preserving complete text artifacts from the boundary "
            "checkpoint"
        ),
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "parameter_encoding_count": EXPECTED_PARAMETER_COUNT,
        "restored_a16_count": EXPECTED_A16_BOUNDARY_COUNT,
        "remaining_float16_count": EXPECTED_FP16_INTERNAL_COUNT,
        "weight_source_encodings_sha256": weight_encodings_hash,
        "boundary_source_encodings_sha256": boundary_encodings_hash,
        "output_encodings_sha256": output_encodings_hash,
        "vision_onnx_sha256": weight_model_hash,
        "weight_marker_sha256": weight_marker_hash,
        "boundary_marker_sha256": boundary_marker_hash,
    }
    if {
        key: value
        for key, value in output_args.items()
        if key != PROVENANCE_FIELD
    } != weight_args:
        raise AssertionError("Unexpected args.json field changed")
    _validate_raw_precision(
        output_args,
        expected=DESTINATION_EXPORT_PRECISION,
        path=weight_args_path,
    )
    output_args_bytes = _json_bytes(output_args)
    output_args_hash = hashlib.sha256(output_args_bytes).hexdigest()

    marker = {
        "schema_version": 1,
        "purpose": (
            "Combine validated FLOAT16 vision parameters with the validated "
            "A16-boundary/FLOAT16-internal activation layout"
        ),
        "sources": {
            "fp16_weight_checkpoint": str(fp16_weight_source),
            "boundary_checkpoint": str(boundary_source),
        },
        "precision_contract": {
            "checkpoint_export_precision": DESTINATION_EXPORT_PRECISION,
            "export_args_source": "fp16_weight_checkpoint",
            "text_artifacts_source": "boundary_checkpoint",
            "vision_parameters": "FLOAT16",
            "restored_boundaries": "A16",
            "remaining_vision_activations": "FLOAT16",
        },
        "calibration": calibration,
        "counts": {
            "activation_encodings": EXPECTED_ACTIVATION_COUNT,
            "a16_boundaries": EXPECTED_A16_BOUNDARY_COUNT,
            "float16_internal_activations": EXPECTED_FP16_INTERNAL_COUNT,
            "float16_parameters": EXPECTED_PARAMETER_COUNT,
        },
        "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
        "graph_contract": graph_summary,
        "hashes": {
            "weight_source_args_sha256": weight_args_hash,
            "boundary_source_args_sha256": boundary_args_hash,
            "output_args_sha256": output_args_hash,
            "weight_source_encodings_sha256": weight_encodings_hash,
            "boundary_source_encodings_sha256": boundary_encodings_hash,
            "output_encodings_sha256": output_encodings_hash,
            "weight_source_model_sha256": weight_model_hash,
            "boundary_source_model_sha256": boundary_model_hash,
            "output_model_sha256": weight_model_hash,
            "weight_marker_sha256": weight_marker_hash,
            "boundary_marker_sha256": boundary_marker_hash,
            "boundary_all_fp16_marker_sha256": all_fp16_marker_hash,
            "preserved_boundary_files_canonical_sha256": _canonical_sha256(
                preserved_boundary_hashes
            ),
            **{
                name: value
                for name, value in output_summary.items()
                if name.endswith("_sha256")
            },
        },
        "encoding_summary": output_summary,
        "exact_unchanged_fields": [
            "all complete boundary checkpoint files except args.json and "
            "vision_encoder.encodings",
            "vision_encoder.onnx bytes",
            "all 205 canonical FLOAT16 parameter encodings",
            "all top-level vision encoding metadata",
            "w4a16 exporter selection and raw_args",
            "all boundary text-model artifacts and encodings",
            "both source provenance marker byte streams",
            f"all args.json fields except {PROVENANCE_FIELD}",
        ],
        "copied_from_boundary_checkpoint_byte_for_byte": [
            "all complete text-model artifacts and encodings",
            "all nine calibrated A16 boundary encodings",
            "all 925 canonical FLOAT16 internal activation encodings",
        ],
    }
    marker_bytes = _json_bytes(marker)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            boundary_source,
            temporary,
            copy_function=copy_function,
            ignore=shutil.ignore_patterns(
                ARGS_FILENAME,
                ENCODINGS_FILENAME,
                MARKER_FILENAME,
                WEIGHT_MARKER_FILENAME,
            ),
        )
        copy_function(
            str(weight_marker_path),
            str(temporary / WEIGHT_MARKER_FILENAME),
        )
        (temporary / ENCODINGS_FILENAME).write_bytes(
            output_encodings_bytes
        )
        (temporary / ARGS_FILENAME).write_bytes(output_args_bytes)
        (temporary / MARKER_FILENAME).write_bytes(marker_bytes)

        for name, expected_hash in preserved_boundary_hashes.items():
            output_path = temporary / name
            if not output_path.is_file() or sha256_file(
                output_path
            ) != expected_hash:
                raise RuntimeError(
                    f"Boundary source artifact changed during assembly: {name}"
                )
        if sha256_file(temporary / ENCODINGS_FILENAME) != (
            output_encodings_hash
        ):
            raise RuntimeError("Written vision encodings hash mismatch")
        if sha256_file(temporary / ARGS_FILENAME) != output_args_hash:
            raise RuntimeError("Written args.json hash mismatch")
        if sha256_file(
            temporary / WEIGHT_MARKER_FILENAME
        ) != weight_marker_hash:
            raise RuntimeError("FP16-weight provenance marker changed")

        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fp16_weight_source",
        type=Path,
        help=(
            "W-FP16/A16 checkpoint from "
            "make_vision_fp16_weight_checkpoint.py"
        ),
    )
    parser.add_argument(
        "boundary_source",
        type=Path,
        help=(
            "W8/A16-boundary checkpoint from "
            "make_vision_a16_boundary_checkpoint.py"
        ),
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="New combined checkpoint (must not already exist)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_vision_fp16_weight_boundary_checkpoint(
        args.fp16_weight_source,
        args.boundary_source,
        args.destination,
    )
    print(
        f"Created combined vision precision checkpoint: {output} "
        f"({EXPECTED_PARAMETER_COUNT} FLOAT16 parameters; "
        f"{EXPECTED_A16_BOUNDARY_COUNT} A16 boundaries; "
        f"{EXPECTED_FP16_INTERNAL_COUNT} FLOAT16 internal activations; "
        f"{DESTINATION_EXPORT_PRECISION} exporter path)"
    )


if __name__ == "__main__":
    main()
