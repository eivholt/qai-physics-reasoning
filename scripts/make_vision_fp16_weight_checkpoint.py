#!/usr/bin/env python3
"""Convert Cosmos vision W8 parameter encodings to FLOAT16.

This is a deterministic, fail-closed transform for the calibrated 224x384
paired-frame vision checkpoint. It changes all 205 vision parameter encodings
from INT8 to AIMET's canonical FLOAT16 encoding while preserving all 934
INT16 activation encodings and the raw ONNX graph/weights byte-for-byte.

Only ``vision_encoder.encodings`` and ``args.json`` are replaced. A provenance
marker is added, while every other checkpoint artifact is hard-linked when
possible and copied otherwise. Output is assembled in a sibling temporary
directory and atomically renamed into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

AIMET_ENCODING_VERSION = "1.0.0"
ARGS_FILENAME = "args.json"
ENCODINGS_FILENAME = "vision_encoder.encodings"
MODEL_FILENAME = "vision_encoder.onnx"
MARKER_FILENAME = "vision_fp16_weights.json"
PROVENANCE_FIELD = "vision_fp16_weight_postprocess"

EXPECTED_IMAGE_SIZE = (224, 384)
EXPECTED_ACTIVATION_COUNT = 934
EXPECTED_PARAMETER_COUNT = 205

FLOAT16_ENCODING = {
    "bw": 16,
    "dtype": "FLOAT",
    "enc_type": "PER_TENSOR",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _entries_by_name(
    entries: Any,
    *,
    field: str,
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError(f"{path}: {field} must use AIMET 1.0 list format")
    by_name: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {field}[{index}] is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{path}: {field}[{index}] has no non-empty tensor name"
            )
        if name in by_name:
            raise ValueError(f"{path}: duplicate {field} tensor {name!r}")
        by_name[name] = entry
    return by_name


def _validate_args(path: Path) -> dict[str, Any]:
    args = _load_json_object(path)
    if args.get("precision") != "w4a16":
        raise ValueError(
            f"{path}: precision is {args.get('precision')!r}; expected "
            "'w4a16'"
        )

    image_size = args.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in image_size
        )
    ):
        raise ValueError(
            f"{path}: image_size must be [HEIGHT, WIDTH], found {image_size!r}"
        )
    if tuple(image_size) != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"{path}: image_size is {image_size!r}; expected "
            f"{list(EXPECTED_IMAGE_SIZE)!r}"
        )

    calibration_source = args.get("vision_calibration_source")
    if calibration_source != "paired_frame_manifest":
        raise ValueError(
            f"{path}: vision_calibration_source is {calibration_source!r}; "
            "expected 'paired_frame_manifest'"
        )
    manifest_path = args.get("veg_paired_calibration_manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError(
            f"{path}: missing paired-frame calibration manifest provenance"
        )
    manifest_hash = args.get("veg_paired_calibration_manifest_sha256")
    if not isinstance(manifest_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", manifest_hash
    ) is None:
        raise ValueError(
            f"{path}: paired-frame calibration SHA-256 is invalid"
        )
    if PROVENANCE_FIELD in args:
        raise ValueError(
            f"{path}: {PROVENANCE_FIELD} already exists; source appears "
            "already transformed"
        )
    return args


def _convert_encodings(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encodings = _load_json_object(path)
    if encodings.get("version") != AIMET_ENCODING_VERSION:
        raise ValueError(
            f"{path}: encoding version is {encodings.get('version')!r}; "
            f"expected {AIMET_ENCODING_VERSION!r}"
        )

    activation_by_name = _entries_by_name(
        encodings.get("activation_encodings"),
        field="activation_encodings",
        path=path,
    )
    parameter_by_name = _entries_by_name(
        encodings.get("param_encodings"),
        field="param_encodings",
        path=path,
    )
    overlap = sorted(set(activation_by_name) & set(parameter_by_name))
    if overlap:
        raise ValueError(
            f"{path}: tensors occur in both activation and parameter "
            f"encodings: {overlap[:5]!r}"
        )

    if len(activation_by_name) != EXPECTED_ACTIVATION_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_ACTIVATION_COUNT} activation "
            f"encodings, found {len(activation_by_name)}"
        )
    invalid_activations = [
        name
        for name, entry in activation_by_name.items()
        if entry.get("dtype") != "INT" or entry.get("bw") != 16
    ]
    if invalid_activations:
        name = invalid_activations[0]
        entry = activation_by_name[name]
        raise ValueError(
            f"{path}: all {EXPECTED_ACTIVATION_COUNT} activation encodings "
            "must be A16 INT; "
            f"{name!r} has dtype={entry.get('dtype')!r}, "
            f"bw={entry.get('bw')!r}"
        )

    if len(parameter_by_name) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_PARAMETER_COUNT} parameter "
            f"encodings, found {len(parameter_by_name)}"
        )
    float_parameters = [
        name
        for name, entry in parameter_by_name.items()
        if entry.get("dtype") == "FLOAT"
    ]
    if float_parameters:
        raise ValueError(
            f"{path}: source contains pre-existing FLOAT parameter "
            f"encodings: {float_parameters[:5]!r}"
        )
    invalid_parameters = [
        name
        for name, entry in parameter_by_name.items()
        if entry.get("dtype") != "INT" or entry.get("bw") != 8
    ]
    if invalid_parameters:
        name = invalid_parameters[0]
        entry = parameter_by_name[name]
        raise ValueError(
            f"{path}: all {EXPECTED_PARAMETER_COUNT} parameter encodings "
            "must be W8 INT; "
            f"{name!r} has dtype={entry.get('dtype')!r}, "
            f"bw={entry.get('bw')!r}"
        )

    source_activations = encodings["activation_encodings"]
    source_parameters = encodings["param_encodings"]
    source_metadata = {
        key: value
        for key, value in encodings.items()
        if key not in {"activation_encodings", "param_encodings"}
    }
    parameter_names = [entry["name"] for entry in source_parameters]

    converted = dict(encodings)
    converted["param_encodings"] = [
        {"name": name, **FLOAT16_ENCODING}
        for name in parameter_names
    ]

    if converted["activation_encodings"] != source_activations:
        raise AssertionError("Activation encodings changed during conversion")
    converted_metadata = {
        key: value
        for key, value in converted.items()
        if key not in {"activation_encodings", "param_encodings"}
    }
    if converted_metadata != source_metadata:
        raise AssertionError("Top-level encoding metadata changed")
    if [
        entry["name"] for entry in converted["param_encodings"]
    ] != parameter_names:
        raise AssertionError("Parameter tensor names or ordering changed")

    summary = {
        "activation_count": len(source_activations),
        "parameter_count": len(source_parameters),
        "source_activation_types": {"INT16": len(source_activations)},
        "source_parameter_types": {"INT8": len(source_parameters)},
        "destination_activation_types": {"INT16": len(source_activations)},
        "destination_parameter_types": {"FLOAT16": len(source_parameters)},
        "activation_encodings_canonical_sha256": _canonical_sha256(
            source_activations
        ),
        "parameter_names_canonical_sha256": _canonical_sha256(
            parameter_names
        ),
        "top_level_metadata_canonical_sha256": _canonical_sha256(
            source_metadata
        ),
    }
    return converted, summary


def _write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def create_vision_fp16_weight_checkpoint(
    source: Path,
    destination: Path,
    *,
    copy_function: Callable[[str, str], str] = _hardlink_or_copy,
) -> Path:
    """Create a W-FP16/A16 vision checkpoint without modifying ONNX bytes."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"Source checkpoint is not a directory: {source}"
        )
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite: {destination}"
        )
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be inside the source checkpoint")

    source_args = source / ARGS_FILENAME
    source_encodings = source / ENCODINGS_FILENAME
    source_model = source / MODEL_FILENAME
    missing = [
        str(path)
        for path in (source_args, source_encodings, source_model)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Source checkpoint is incomplete. Missing: " + ", ".join(missing)
        )

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary destination already exists: {temporary}"
        )

    source_args_payload = _validate_args(source_args)
    converted_encodings, encoding_summary = _convert_encodings(
        source_encodings
    )
    source_args_hash = sha256_file(source_args)
    source_encodings_hash = sha256_file(source_encodings)
    source_model_hash = sha256_file(source_model)
    converted_encodings_bytes = _json_bytes(converted_encodings)
    converted_encodings_hash = hashlib.sha256(
        converted_encodings_bytes
    ).hexdigest()

    converted_args = dict(source_args_payload)
    converted_args[PROVENANCE_FIELD] = {
        "activation_encoding_count": EXPECTED_ACTIVATION_COUNT,
        "activation_quantization": "A16 integer",
        "destination_parameter_quantization": "FLOAT16",
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "marker": MARKER_FILENAME,
        "output_encodings_sha256": converted_encodings_hash,
        "parameter_encoding_count": EXPECTED_PARAMETER_COUNT,
        "scheme": "FLOAT16 vision parameters with A16 integer activations",
        "source_checkpoint_name": source.name,
        "source_encodings_sha256": source_encodings_hash,
        "source_model_sha256": source_model_hash,
        "source_parameter_quantization": "W8 integer",
    }
    if {
        key: value
        for key, value in converted_args.items()
        if key != PROVENANCE_FIELD
    } != source_args_payload:
        raise AssertionError("Unexpected args.json field changed")
    converted_args_bytes = _json_bytes(converted_args)
    converted_args_hash = hashlib.sha256(converted_args_bytes).hexdigest()

    marker = {
        "schema_version": 1,
        "purpose": (
            "Use FLOAT16 vision parameters with calibrated A16 integer "
            "activations while leaving the raw vision ONNX unchanged"
        ),
        "source_checkpoint": str(source),
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "counts": {
            "activation_encodings": EXPECTED_ACTIVATION_COUNT,
            "parameters_converted_int8_to_float16": (
                EXPECTED_PARAMETER_COUNT
            ),
        },
        "hashes": {
            "source_args_sha256": source_args_hash,
            "output_args_sha256": converted_args_hash,
            "source_encodings_sha256": source_encodings_hash,
            "output_encodings_sha256": converted_encodings_hash,
            "source_model_sha256": source_model_hash,
            "output_model_sha256": source_model_hash,
            "activation_encodings_canonical_sha256": encoding_summary[
                "activation_encodings_canonical_sha256"
            ],
            "parameter_names_canonical_sha256": encoding_summary[
                "parameter_names_canonical_sha256"
            ],
            "top_level_metadata_canonical_sha256": encoding_summary[
                "top_level_metadata_canonical_sha256"
            ],
        },
        "encoding_summary": encoding_summary,
        "replacement_parameter_encoding": FLOAT16_ENCODING,
        "exact_unchanged_fields": [
            "vision_encoder.onnx bytes",
            "activation_encodings values and ordering",
            "parameter tensor names and ordering",
            "encoding version, producer, and quantizer_args",
            f"all args.json fields except {PROVENANCE_FIELD}",
            "all other checkpoint artifacts",
        ],
    }
    marker_bytes = _json_bytes(marker)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            source,
            temporary,
            copy_function=copy_function,
            ignore=shutil.ignore_patterns(
                ARGS_FILENAME,
                ENCODINGS_FILENAME,
                MARKER_FILENAME,
            ),
        )
        _write_bytes(temporary / ENCODINGS_FILENAME, converted_encodings_bytes)
        _write_bytes(temporary / ARGS_FILENAME, converted_args_bytes)
        _write_bytes(temporary / MARKER_FILENAME, marker_bytes)

        output_model = temporary / MODEL_FILENAME
        if not output_model.is_file():
            raise RuntimeError(
                f"Copied checkpoint is missing raw ONNX model: {output_model}"
            )
        if os.path.samefile(source_model, output_model):
            output_model_hash = source_model_hash
        else:
            output_model_hash = sha256_file(output_model)
        if output_model_hash != source_model_hash:
            raise RuntimeError(
                "Raw vision ONNX changed while assembling checkpoint"
            )
        if sha256_file(temporary / ENCODINGS_FILENAME) != (
            converted_encodings_hash
        ):
            raise RuntimeError("Written vision encodings hash mismatch")
        if sha256_file(temporary / ARGS_FILENAME) != converted_args_hash:
            raise RuntimeError("Written args.json hash mismatch")

        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Integer W8A16 224x384 paired-calibration checkpoint",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="New W-FP16/A16 checkpoint (must not already exist)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_vision_fp16_weight_checkpoint(
        args.source,
        args.destination,
    )
    print(
        f"Created vision W-FP16/A16 checkpoint: {output} "
        f"({EXPECTED_PARAMETER_COUNT} parameters converted; "
        f"{EXPECTED_ACTIVATION_COUNT} A16 activations preserved)"
    )


if __name__ == "__main__":
    main()
