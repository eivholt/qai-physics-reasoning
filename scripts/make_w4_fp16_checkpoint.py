#!/usr/bin/env python3
"""Convert a calibrated Cosmos W4A16 checkpoint to W4 + FP16 activations.

The W4 and W4A16 QAIHM paths use the same INT4 text-weight encodings.  Their
activation contracts differ: W4 uses FLOAT16 quantizers, while W4A16 loads
calibrated integer activation encodings.  No activation calibration range is
needed for FLOAT16, so an existing checkpoint can be converted without
recalibrating its weights.

This utility is deliberately fail-closed.  It accepts AIMET 1.0 checkpoints
whose activation entries are INT8/INT16, preserves every parameter encoding,
and replaces every text activation entry with AIMET's canonical FLOAT16
encoding.  Vision activations are converted by default; ``--keep-vision-w4a16``
retains them byte-for-byte for targets where the FP16 vision graph does not
link.  The large ONNX/data/tokenizer files are hard-linked when the source and
destination are on the same filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

AIMET_ENCODING_VERSION = "1.0.0"
ENCODING_FILES = ("model.encodings", "vision_encoder.encodings")
ARGS_FILENAME = "args.json"
MARKER_FILENAME = "w4_fp16.json"
SOURCE_PRECISION = "w4a16"
DESTINATION_PRECISION = "w4"
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
        raise ValueError(
            f"{path}: {field} must use AIMET 1.0 list format"
        )
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


def _convert_encoding_file(
    path: Path,
    *,
    require_int4_parameter: bool,
    convert_activations: bool = True,
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
    if not activation_by_name:
        raise ValueError(f"{path}: activation_encodings is empty")

    overlap = sorted(set(activation_by_name) & set(parameter_by_name))
    if overlap:
        raise ValueError(
            f"{path}: tensors occur in both activation and parameter "
            f"encodings: {overlap[:5]!r}"
        )

    source_types: dict[str, int] = {}
    for name, entry in activation_by_name.items():
        dtype = entry.get("dtype")
        bitwidth = entry.get("bw")
        if dtype != "INT" or bitwidth not in {8, 16}:
            raise ValueError(
                f"{path}: activation {name!r} is not an INT8/INT16 "
                f"W4A16 encoding: dtype={dtype!r}, bw={bitwidth!r}"
            )
        key = f"{dtype}{bitwidth}"
        source_types[key] = source_types.get(key, 0) + 1

    if require_int4_parameter and not any(
        entry.get("dtype") == "INT" and entry.get("bw") == 4
        for entry in parameter_by_name.values()
    ):
        raise ValueError(f"{path}: no INT4 parameter encoding was found")

    parameters_before = _canonical_sha256(
        encodings["param_encodings"]
    )
    if convert_activations:
        destination_activations = [
            {"name": entry["name"], **FLOAT16_ENCODING}
            for entry in encodings["activation_encodings"]
        ]
        destination_types = {"FLOAT16": len(destination_activations)}
        encodings["activation_encodings"] = destination_activations
    else:
        destination_activations = encodings["activation_encodings"]
        destination_types = dict(source_types)
    if _canonical_sha256(encodings["param_encodings"]) != parameters_before:
        raise AssertionError("Parameter encodings changed during conversion")

    summary = {
        "activation_count": len(destination_activations),
        "activations_converted_to_float16": convert_activations,
        "destination_activation_types": destination_types,
        "parameter_count": len(encodings["param_encodings"]),
        "source_activation_types": source_types,
        "parameter_encodings_sha256": parameters_before,
    }
    return encodings, summary


def _convert_args(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    args = _load_json_object(path)
    precision = args.get("precision")
    if precision != SOURCE_PRECISION:
        raise ValueError(
            f"{path}: precision is {precision!r}; expected "
            f"{SOURCE_PRECISION!r}"
        )

    original_raw_args = args.get("raw_args")
    raw_precision_updated = False
    if original_raw_args is not None:
        if not isinstance(original_raw_args, list) or not all(
            isinstance(value, str) for value in original_raw_args
        ):
            raise ValueError(f"{path}: raw_args must be a list of strings")
        raw_args = list(original_raw_args)
        precision_flags = [
            index
            for index, value in enumerate(raw_args)
            if value == "--precision"
        ]
        if len(precision_flags) > 1:
            raise ValueError(f"{path}: raw_args has multiple --precision flags")
        if precision_flags:
            index = precision_flags[0]
            if index + 1 >= len(raw_args):
                raise ValueError(
                    f"{path}: raw_args --precision has no value"
                )
            if raw_args[index + 1] != SOURCE_PRECISION:
                raise ValueError(
                    f"{path}: raw_args precision is "
                    f"{raw_args[index + 1]!r}; expected "
                    f"{SOURCE_PRECISION!r}"
                )
            raw_args[index + 1] = DESTINATION_PRECISION
            args["raw_args"] = raw_args
            raw_precision_updated = True

    args["precision"] = DESTINATION_PRECISION
    return args, {
        "source_precision": SOURCE_PRECISION,
        "destination_precision": DESTINATION_PRECISION,
        "raw_args_precision_updated": raw_precision_updated,
    }


def _write_json_replacing_hardlink(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_replacing_hardlink(source: Path, destination: Path) -> None:
    """Copy a small preserved file without retaining a mutable hardlink."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def create_w4_fp16_checkpoint(
    source: Path,
    destination: Path,
    *,
    keep_vision_w4a16: bool = False,
) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source checkpoint is not a directory: {source}")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if destination.is_relative_to(source):
        raise ValueError("Destination must not be inside the source checkpoint")

    required = [
        source / ARGS_FILENAME,
        *(source / filename for filename in ENCODING_FILES),
        source / "model_dynamic.onnx",
        source / "model.data",
        source / "embedding_weights.raw",
        source / "vision_encoder.onnx",
        source / "source_checkpoint.json",
        source / "config.json",
        source / "preprocessor_config.json",
        source / "tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Source checkpoint is incomplete. Missing: " + ", ".join(missing)
        )

    converted_args, args_summary = _convert_args(source / ARGS_FILENAME)
    converted_encodings: dict[str, dict[str, Any]] = {}
    encoding_summaries: dict[str, dict[str, Any]] = {}
    source_hashes = {ARGS_FILENAME: sha256_file(source / ARGS_FILENAME)}
    for filename in ENCODING_FILES:
        convert_activations = (
            filename != "vision_encoder.encodings"
            or not keep_vision_w4a16
        )
        converted, summary = _convert_encoding_file(
            source / filename,
            require_int4_parameter=filename == "model.encodings",
            convert_activations=convert_activations,
        )
        if convert_activations:
            converted_encodings[filename] = converted
        encoding_summaries[filename] = summary
        source_hashes[filename] = sha256_file(source / filename)

    try:
        shutil.copytree(source, destination, copy_function=_hardlink_or_copy)
        _write_json_replacing_hardlink(
            destination / ARGS_FILENAME, converted_args
        )
        for filename, encodings in converted_encodings.items():
            _write_json_replacing_hardlink(destination / filename, encodings)
        if keep_vision_w4a16:
            _copy_replacing_hardlink(
                source / "vision_encoder.encodings",
                destination / "vision_encoder.encodings",
            )

        marker = {
            "schema_version": 1,
            "purpose": (
                "Reuse calibrated text weights with FLOAT16 text "
                "activations for QAIHM Precision.w4 export"
            ),
            "source_checkpoint": str(source),
            "source_file_sha256": source_hashes,
            "precision": args_summary,
            "encoding_files": encoding_summaries,
            "replacement_activation_encoding": FLOAT16_ENCODING,
            "unchanged": [
                "ONNX graph topology",
                "external model weights",
                "all parameter encodings",
                "embedding table",
                "tokenizer and model configuration",
                *(
                    ["vision activation encodings"]
                    if keep_vision_w4a16
                    else []
                ),
            ],
            "weight_note": (
                "Text parameter encodings retain calibrated INT4 weights. "
                "The vision encoder retains its original INT8 parameter "
                "encodings. "
                + (
                    "Its W4A16 activation encodings are preserved for "
                    "QAIRT vision-link compatibility."
                    if keep_vision_w4a16
                    else "Its activations become FLOAT16."
                )
            ),
            "vision_activation_precision": (
                "w4a16" if keep_vision_w4a16 else "fp16"
            ),
        }
        _write_json_replacing_hardlink(
            destination / MARKER_FILENAME, marker
        )
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--keep-vision-w4a16",
        action="store_true",
        help=(
            "Convert text activations to FP16 but preserve "
            "vision_encoder.encodings byte-for-byte."
        ),
    )
    args = parser.parse_args()
    output = create_w4_fp16_checkpoint(
        args.source,
        args.destination,
        keep_vision_w4a16=args.keep_vision_w4a16,
    )
    marker = _load_json_object(output / MARKER_FILENAME)
    text_count = marker["encoding_files"]["model.encodings"][
        "activation_count"
    ]
    vision_count = marker["encoding_files"]["vision_encoder.encodings"][
        "activation_count"
    ]
    vision_action = (
        "preserved as W4A16"
        if args.keep_vision_w4a16
        else "converted to FP16"
    )
    print(
        f"Created W4 + FP16-activation checkpoint: {output} "
        f"({text_count} text activations converted to FP16; "
        f"{vision_count} vision activations {vision_action})"
    )


if __name__ == "__main__":
    main()
