#!/usr/bin/env python3
"""Restore selected A16 boundaries in an otherwise FP16 vision graph.

The all-FP16 Cosmos vision experiment improves host numerical agreement, but
QAIRT 2.45 cannot link its first floating-point patch-projection convolution.
This utility creates a compiler-oriented hybrid checkpoint:

* the checkpoint remains on the standard ``w4`` export path;
* all 205 calibrated vision parameter encodings remain W8;
* 925 internal vision activations remain FLOAT16;
* the four encoded graph inputs, patch-projection output, and four encoded
  graph-output projections are restored byte-for-byte from the matching
  W4A16 checkpoint.

The transform is deliberately fail-closed. It validates the paired 224x384
calibration provenance, the original W4-to-FP16 marker, identical ONNX bytes,
identical W8 parameter encodings, the exact 934-activation contract, and the
ONNX boundary topology before creating anything. Output is assembled in a
sibling temporary directory and atomically renamed into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Callable

AIMET_ENCODING_VERSION = "1.0.0"
ARGS_FILENAME = "args.json"
ENCODINGS_FILENAME = "vision_encoder.encodings"
MODEL_FILENAME = "vision_encoder.onnx"
ALL_FP16_MARKER_FILENAME = "w4_fp16.json"
MARKER_FILENAME = "vision_a16_boundaries.json"
PROVENANCE_FIELD = "vision_a16_boundary_postprocess"

EXPECTED_IMAGE_SIZE = (224, 384)
EXPECTED_ACTIVATION_COUNT = 934
EXPECTED_PARAMETER_COUNT = 205
EXPECTED_RESTORED_COUNT = 9

EXPECTED_GRAPH_INPUTS = (
    "pixel_values",
    "position_ids_cos",
    "position_ids_sin",
    "window_attention_mask",
    "full_attention_mask",
)
ENCODED_GRAPH_INPUTS = (
    "pixel_values",
    "position_ids_cos",
    "position_ids_sin",
    "full_attention_mask",
)
PATCH_PROJECTION_OUTPUT = "conv2d"
EXPECTED_GRAPH_OUTPUTS = (
    "image_features",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
)
OUTPUT_PROJECTION_BY_GRAPH_OUTPUT = {
    "image_features": "conv2d_152",
    "deepstack_visual_embeds_0": "conv2d_38",
    "deepstack_visual_embeds_1": "conv2d_76",
    "deepstack_visual_embeds_2": "conv2d_114",
}
RESTORED_ACTIVATION_NAMES = (
    *ENCODED_GRAPH_INPUTS,
    PATCH_PROJECTION_OUTPUT,
    "conv2d_38",
    "conv2d_76",
    "conv2d_114",
    "conv2d_152",
)

FLOAT16_ENCODING = {
    "bw": 16,
    "dtype": "FLOAT",
    "enc_type": "PER_TENSOR",
}

REQUIRED_ALL_FP16_FILES = (
    ARGS_FILENAME,
    ENCODINGS_FILENAME,
    MODEL_FILENAME,
    ALL_FP16_MARKER_FILENAME,
    "model.encodings",
    "model_dynamic.onnx",
    "model.data",
    "embedding_weights.raw",
    "source_checkpoint.json",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
)


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


def _validate_checkpoint_args(
    path: Path,
    *,
    expected_precision: str,
) -> dict[str, Any]:
    args = _load_json_object(path)
    if args.get("precision") != expected_precision:
        raise ValueError(
            f"{path}: precision is {args.get('precision')!r}; expected "
            f"{expected_precision!r}"
        )
    _validate_raw_precision(args, expected=expected_precision, path=path)

    image_size = args.get("image_size")
    if image_size != list(EXPECTED_IMAGE_SIZE):
        raise ValueError(
            f"{path}: image_size is {image_size!r}; expected "
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
    if args.get("veg_num_samples") != 20:
        raise ValueError(
            f"{path}: veg_num_samples is {args.get('veg_num_samples')!r}; "
            "expected 20"
        )
    if PROVENANCE_FIELD in args:
        raise ValueError(
            f"{path}: {PROVENANCE_FIELD} already exists; source appears "
            "already transformed"
        )
    return args


def _validate_matching_calibration(
    all_fp16_args: dict[str, Any],
    a16_args: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "image_size",
        "vision_calibration_source",
        "veg_paired_calibration_manifest",
        "veg_paired_calibration_manifest_sha256",
        "veg_num_samples",
        "num_samples",
    )
    mismatches = [
        name
        for name in fields
        if all_fp16_args.get(name) != a16_args.get(name)
    ]
    if mismatches:
        raise ValueError(
            "All-FP16 and W4A16 calibration provenance differs in: "
            + ", ".join(mismatches)
        )
    return {name: all_fp16_args.get(name) for name in fields}


def _validate_a16_entry(
    entry: dict[str, Any],
    *,
    path: Path,
) -> None:
    if entry.get("dtype") != "INT" or entry.get("bw") != 16:
        raise ValueError(
            f"{path}: activation {entry.get('name')!r} is not A16 INT"
        )
    if entry.get("enc_type") != "PER_TENSOR":
        raise ValueError(
            f"{path}: activation {entry.get('name')!r} does not use "
            "PER_TENSOR encoding"
        )
    if not isinstance(entry.get("is_sym"), bool):
        raise ValueError(
            f"{path}: activation {entry.get('name')!r} has no boolean is_sym"
        )
    scale = entry.get("scale")
    offset = entry.get("offset")
    if (
        not isinstance(scale, list)
        or not scale
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in scale
        )
        or not isinstance(offset, list)
        or len(offset) != len(scale)
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in offset
        )
    ):
        raise ValueError(
            f"{path}: activation {entry.get('name')!r} has invalid "
            "scale/offset values"
        )


def _validate_encoding_pair(
    all_fp16_path: Path,
    a16_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    all_fp16 = _load_json_object(all_fp16_path)
    a16 = _load_json_object(a16_path)
    for path, value in ((all_fp16_path, all_fp16), (a16_path, a16)):
        if value.get("version") != AIMET_ENCODING_VERSION:
            raise ValueError(
                f"{path}: encoding version is {value.get('version')!r}; "
                f"expected {AIMET_ENCODING_VERSION!r}"
            )

    all_activations = _entries_by_name(
        all_fp16.get("activation_encodings"),
        field="activation_encodings",
        path=all_fp16_path,
    )
    a16_activations = _entries_by_name(
        a16.get("activation_encodings"),
        field="activation_encodings",
        path=a16_path,
    )
    all_parameters = _entries_by_name(
        all_fp16.get("param_encodings"),
        field="param_encodings",
        path=all_fp16_path,
    )
    a16_parameters = _entries_by_name(
        a16.get("param_encodings"),
        field="param_encodings",
        path=a16_path,
    )

    if len(all_activations) != EXPECTED_ACTIVATION_COUNT:
        raise ValueError(
            f"{all_fp16_path}: expected exactly {EXPECTED_ACTIVATION_COUNT} "
            f"activation encodings, found {len(all_activations)}"
        )
    if len(a16_activations) != EXPECTED_ACTIVATION_COUNT:
        raise ValueError(
            f"{a16_path}: expected exactly {EXPECTED_ACTIVATION_COUNT} "
            f"activation encodings, found {len(a16_activations)}"
        )
    all_activation_names = [
        entry["name"] for entry in all_fp16["activation_encodings"]
    ]
    a16_activation_names = [
        entry["name"] for entry in a16["activation_encodings"]
    ]
    if all_activation_names != a16_activation_names:
        raise ValueError(
            "All-FP16 and W4A16 activation tensor names/order do not match"
        )

    invalid_float = [
        name
        for name, entry in all_activations.items()
        if entry != {"name": name, **FLOAT16_ENCODING}
    ]
    if invalid_float:
        raise ValueError(
            f"{all_fp16_path}: all {EXPECTED_ACTIVATION_COUNT} source "
            "activations must use canonical FLOAT16 encoding; first "
            f"mismatch is {invalid_float[0]!r}"
        )
    for entry in a16["activation_encodings"]:
        _validate_a16_entry(entry, path=a16_path)

    if len(all_parameters) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"{all_fp16_path}: expected exactly {EXPECTED_PARAMETER_COUNT} "
            f"parameter encodings, found {len(all_parameters)}"
        )
    if len(a16_parameters) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"{a16_path}: expected exactly {EXPECTED_PARAMETER_COUNT} "
            f"parameter encodings, found {len(a16_parameters)}"
        )
    invalid_parameters = [
        name
        for name, entry in all_parameters.items()
        if entry.get("dtype") != "INT" or entry.get("bw") != 8
    ]
    if invalid_parameters:
        raise ValueError(
            f"{all_fp16_path}: all {EXPECTED_PARAMETER_COUNT} vision "
            "parameters must remain W8 INT; first mismatch is "
            f"{invalid_parameters[0]!r}"
        )
    if all_fp16["param_encodings"] != a16["param_encodings"]:
        raise ValueError(
            "All-FP16 and W4A16 parameter encodings are not exactly equal"
        )

    metadata = lambda value: {
        key: item
        for key, item in value.items()
        if key not in {"activation_encodings", "param_encodings"}
    }
    if metadata(all_fp16) != metadata(a16):
        raise ValueError(
            "All-FP16 and W4A16 top-level encoding metadata differs"
        )

    missing = sorted(set(RESTORED_ACTIVATION_NAMES) - set(all_activations))
    if missing:
        raise ValueError(
            "Required boundary activation encodings are missing: "
            + ", ".join(missing)
        )

    parameter_hash = _canonical_sha256(all_fp16["param_encodings"])
    summary = {
        "activation_count": EXPECTED_ACTIVATION_COUNT,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "activation_names_canonical_sha256": _canonical_sha256(
            all_activation_names
        ),
        "all_fp16_activations_canonical_sha256": _canonical_sha256(
            all_fp16["activation_encodings"]
        ),
        "a16_activations_canonical_sha256": _canonical_sha256(
            a16["activation_encodings"]
        ),
        "parameter_encodings_canonical_sha256": parameter_hash,
        "top_level_metadata_canonical_sha256": _canonical_sha256(
            metadata(all_fp16)
        ),
    }
    return all_fp16, a16, a16_activations, summary


def _validate_all_fp16_marker(
    path: Path,
    *,
    a16_source: Path,
    a16_args_hash: str,
    a16_encodings_hash: str,
    parameter_encodings_hash: str,
) -> dict[str, Any]:
    marker = _load_json_object(path)
    if marker.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported marker schema")
    if marker.get("vision_activation_precision") != "fp16":
        raise ValueError(f"{path}: source is not marked as all-FP16 vision")
    precision = marker.get("precision")
    if (
        not isinstance(precision, dict)
        or precision.get("source_precision") != "w4a16"
        or precision.get("destination_precision") != "w4"
    ):
        raise ValueError(f"{path}: invalid W4-to-FP16 precision provenance")

    marker_source = marker.get("source_checkpoint")
    if not isinstance(marker_source, str) or (
        Path(marker_source).expanduser().resolve() != a16_source
    ):
        raise ValueError(
            f"{path}: marker does not identify the supplied W4A16 source"
        )
    source_hashes = marker.get("source_file_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError(f"{path}: source hash provenance is missing")
    expected_hashes = {
        ARGS_FILENAME: a16_args_hash,
        ENCODINGS_FILENAME: a16_encodings_hash,
    }
    for filename, expected in expected_hashes.items():
        if source_hashes.get(filename) != expected:
            raise ValueError(
                f"{path}: recorded {filename} source hash does not match"
            )

    files = marker.get("encoding_files")
    vision = (
        files.get(ENCODINGS_FILENAME)
        if isinstance(files, dict)
        else None
    )
    if not isinstance(vision, dict):
        raise ValueError(f"{path}: vision encoding provenance is missing")
    if (
        vision.get("activation_count") != EXPECTED_ACTIVATION_COUNT
        or vision.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or vision.get("activations_converted_to_float16") is not True
        or vision.get("destination_activation_types")
        != {"FLOAT16": EXPECTED_ACTIVATION_COUNT}
        or vision.get("parameter_encodings_sha256")
        != parameter_encodings_hash
    ):
        raise ValueError(f"{path}: vision conversion summary does not match")
    return marker


def _nearest_conv_output(
    output_name: str,
    producer_by_tensor: dict[str, Any],
) -> str:
    queue: deque[tuple[str, int]] = deque([(output_name, 0)])
    seen: set[str] = set()
    nearest_depth: int | None = None
    nearest: set[str] = set()
    while queue:
        tensor, depth = queue.popleft()
        if tensor in seen or (
            nearest_depth is not None and depth > nearest_depth
        ):
            continue
        seen.add(tensor)
        producer = producer_by_tensor.get(tensor)
        if producer is None:
            continue
        if producer.op_type == "Conv":
            nearest_depth = depth
            nearest.update(name for name in producer.output if name)
            continue
        queue.extend(
            (name, depth + 1) for name in producer.input if name
        )
    if len(nearest) != 1:
        raise ValueError(
            f"Graph output {output_name!r} does not have one unambiguous "
            f"nearest Conv output; found {sorted(nearest)!r}"
        )
    return next(iter(nearest))


def _has_input_ancestor(
    tensor_name: str,
    input_name: str,
    producer_by_tensor: dict[str, Any],
) -> bool:
    queue = deque([tensor_name])
    seen: set[str] = set()
    while queue:
        tensor = queue.popleft()
        if tensor == input_name:
            return True
        if tensor in seen:
            continue
        seen.add(tensor)
        producer = producer_by_tensor.get(tensor)
        if producer is not None:
            queue.extend(name for name in producer.input if name)
    return False


def _validate_graph_contract(
    path: Path,
    activation_names: set[str],
) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The onnx package is required to validate the vision graph"
        ) from exc

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Cannot load ONNX graph: {path}") from exc
    graph = model.graph
    graph_inputs = tuple(value.name for value in graph.input)
    graph_outputs = tuple(value.name for value in graph.output)
    if graph_inputs != EXPECTED_GRAPH_INPUTS:
        raise ValueError(
            f"{path}: graph inputs are {graph_inputs!r}; expected "
            f"{EXPECTED_GRAPH_INPUTS!r}"
        )
    if graph_outputs != EXPECTED_GRAPH_OUTPUTS:
        raise ValueError(
            f"{path}: graph outputs are {graph_outputs!r}; expected "
            f"{EXPECTED_GRAPH_OUTPUTS!r}"
        )

    producer_by_tensor: dict[str, Any] = {}
    for node in graph.node:
        for output in node.output:
            if not output:
                continue
            if output in producer_by_tensor:
                raise ValueError(
                    f"{path}: multiple nodes produce tensor {output!r}"
                )
            producer_by_tensor[output] = node

    encoded_inputs = tuple(
        name for name in graph_inputs if name in activation_names
    )
    if encoded_inputs != ENCODED_GRAPH_INPUTS:
        raise ValueError(
            f"{path}: encoded graph inputs are {encoded_inputs!r}; expected "
            f"{ENCODED_GRAPH_INPUTS!r}"
        )

    patch_node = producer_by_tensor.get(PATCH_PROJECTION_OUTPUT)
    if patch_node is None or patch_node.op_type != "Conv":
        raise ValueError(
            f"{path}: {PATCH_PROJECTION_OUTPUT!r} is not produced by Conv"
        )
    if not _has_input_ancestor(
        PATCH_PROJECTION_OUTPUT,
        "pixel_values",
        producer_by_tensor,
    ):
        raise ValueError(
            f"{path}: patch projection is not downstream of pixel_values"
        )

    output_projections = {
        output: _nearest_conv_output(output, producer_by_tensor)
        for output in graph_outputs
    }
    if output_projections != OUTPUT_PROJECTION_BY_GRAPH_OUTPUT:
        raise ValueError(
            f"{path}: graph output projections are {output_projections!r}; "
            f"expected {OUTPUT_PROJECTION_BY_GRAPH_OUTPUT!r}"
        )
    if not set(output_projections.values()).issubset(activation_names):
        raise ValueError(
            f"{path}: graph output projection encodings are incomplete"
        )

    graph_restored = (
        *encoded_inputs,
        PATCH_PROJECTION_OUTPUT,
        *(
            OUTPUT_PROJECTION_BY_GRAPH_OUTPUT[output]
            for output in (
                "deepstack_visual_embeds_0",
                "deepstack_visual_embeds_1",
                "deepstack_visual_embeds_2",
                "image_features",
            )
        ),
    )
    if graph_restored != RESTORED_ACTIVATION_NAMES:
        raise ValueError(
            f"{path}: graph-derived boundary tensors are "
            f"{graph_restored!r}; expected {RESTORED_ACTIVATION_NAMES!r}"
        )
    if len(set(graph_restored)) != EXPECTED_RESTORED_COUNT:
        raise ValueError(
            f"{path}: graph does not prove exactly "
            f"{EXPECTED_RESTORED_COUNT} unique restored activations"
        )
    return {
        "graph_inputs": list(graph_inputs),
        "encoded_graph_inputs": list(encoded_inputs),
        "patch_projection_output": PATCH_PROJECTION_OUTPUT,
        "graph_outputs": list(graph_outputs),
        "output_projection_by_graph_output": output_projections,
        "restored_activation_names": list(graph_restored),
    }


def _restore_boundary_encodings(
    all_fp16: dict[str, Any],
    a16_by_name: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    restored = set(RESTORED_ACTIVATION_NAMES)
    output = dict(all_fp16)
    output["activation_encodings"] = [
        (
            dict(a16_by_name[entry["name"]])
            if entry["name"] in restored
            else dict(entry)
        )
        for entry in all_fp16["activation_encodings"]
    ]

    output_by_name = {
        entry["name"]: entry for entry in output["activation_encodings"]
    }
    restored_entries = [
        output_by_name[name] for name in RESTORED_ACTIVATION_NAMES
    ]
    if any(
        entry != a16_by_name[entry["name"]]
        for entry in restored_entries
    ):
        raise AssertionError("A16 boundary encodings were not restored exactly")
    remaining = [
        entry
        for entry in output["activation_encodings"]
        if entry["name"] not in restored
    ]
    if len(restored_entries) != EXPECTED_RESTORED_COUNT:
        raise AssertionError("Unexpected restored activation count")
    expected_remaining = EXPECTED_ACTIVATION_COUNT - EXPECTED_RESTORED_COUNT
    if len(remaining) != expected_remaining or any(
        entry != {"name": entry["name"], **FLOAT16_ENCODING}
        for entry in remaining
    ):
        raise AssertionError("Non-boundary activations did not remain FLOAT16")
    if output["param_encodings"] != all_fp16["param_encodings"]:
        raise AssertionError("Parameter encodings changed")

    summary = {
        "activation_count": EXPECTED_ACTIVATION_COUNT,
        "restored_a16_count": EXPECTED_RESTORED_COUNT,
        "remaining_float16_count": expected_remaining,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
        "restored_a16_canonical_sha256": _canonical_sha256(
            restored_entries
        ),
        "remaining_float16_canonical_sha256": _canonical_sha256(remaining),
    }
    return output, summary


def _write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def create_vision_a16_boundary_checkpoint(
    all_fp16_source: Path,
    a16_source: Path,
    destination: Path,
    *,
    copy_function: Callable[[str, str], str] = _hardlink_or_copy,
) -> Path:
    """Create the W4/W8 hybrid vision checkpoint transactionally."""
    all_fp16_source = all_fp16_source.expanduser().resolve()
    a16_source = a16_source.expanduser().resolve()
    destination = destination.expanduser().resolve()

    if not all_fp16_source.is_dir():
        raise FileNotFoundError(
            f"All-FP16 source is not a directory: {all_fp16_source}"
        )
    if not a16_source.is_dir():
        raise FileNotFoundError(
            f"W4A16 source is not a directory: {a16_source}"
        )
    if all_fp16_source == a16_source:
        raise ValueError("All-FP16 and W4A16 sources must be different")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite: {destination}"
        )
    if (
        destination == all_fp16_source
        or all_fp16_source in destination.parents
        or destination == a16_source
        or a16_source in destination.parents
    ):
        raise ValueError("Destination must not be inside either source")

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary destination already exists: {temporary}"
        )

    all_required = [
        all_fp16_source / filename for filename in REQUIRED_ALL_FP16_FILES
    ]
    a16_required = [
        a16_source / filename
        for filename in (ARGS_FILENAME, ENCODINGS_FILENAME, MODEL_FILENAME)
    ]
    missing = [
        str(path)
        for path in (*all_required, *a16_required)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Source checkpoint is incomplete. Missing: " + ", ".join(missing)
        )
    if (all_fp16_source / MARKER_FILENAME).exists():
        raise ValueError(
            f"All-FP16 source already contains {MARKER_FILENAME}"
        )

    all_args = _validate_checkpoint_args(
        all_fp16_source / ARGS_FILENAME,
        expected_precision="w4",
    )
    a16_args = _validate_checkpoint_args(
        a16_source / ARGS_FILENAME,
        expected_precision="w4a16",
    )
    calibration = _validate_matching_calibration(all_args, a16_args)

    all_encodings, _, a16_by_name, encoding_pair_summary = (
        _validate_encoding_pair(
            all_fp16_source / ENCODINGS_FILENAME,
            a16_source / ENCODINGS_FILENAME,
        )
    )
    all_onnx_hash = sha256_file(all_fp16_source / MODEL_FILENAME)
    a16_onnx_hash = sha256_file(a16_source / MODEL_FILENAME)
    if all_onnx_hash != a16_onnx_hash:
        raise ValueError(
            "All-FP16 and W4A16 vision ONNX byte hashes do not match"
        )

    a16_args_hash = sha256_file(a16_source / ARGS_FILENAME)
    a16_encodings_hash = sha256_file(a16_source / ENCODINGS_FILENAME)
    all_args_hash = sha256_file(all_fp16_source / ARGS_FILENAME)
    all_encodings_hash = sha256_file(
        all_fp16_source / ENCODINGS_FILENAME
    )
    all_marker_hash = sha256_file(
        all_fp16_source / ALL_FP16_MARKER_FILENAME
    )
    _validate_all_fp16_marker(
        all_fp16_source / ALL_FP16_MARKER_FILENAME,
        a16_source=a16_source,
        a16_args_hash=a16_args_hash,
        a16_encodings_hash=a16_encodings_hash,
        parameter_encodings_hash=encoding_pair_summary[
            "parameter_encodings_canonical_sha256"
        ],
    )
    graph_summary = _validate_graph_contract(
        all_fp16_source / MODEL_FILENAME,
        {
            entry["name"]
            for entry in all_encodings["activation_encodings"]
        },
    )

    output_encodings, output_summary = _restore_boundary_encodings(
        all_encodings,
        a16_by_name,
    )
    output_encodings_bytes = _json_bytes(output_encodings)
    output_encodings_hash = hashlib.sha256(
        output_encodings_bytes
    ).hexdigest()

    output_args = dict(all_args)
    output_args[PROVENANCE_FIELD] = {
        "schema_version": 1,
        "scheme": (
            "W4 text path, W8 vision parameters, A16 graph boundaries, "
            "FLOAT16 internal vision activations"
        ),
        "marker": MARKER_FILENAME,
        "a16_source_checkpoint_name": a16_source.name,
        "all_fp16_source_checkpoint_name": all_fp16_source.name,
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "restored_a16_count": EXPECTED_RESTORED_COUNT,
        "remaining_float16_count": (
            EXPECTED_ACTIVATION_COUNT - EXPECTED_RESTORED_COUNT
        ),
        "restored_a16_names": list(RESTORED_ACTIVATION_NAMES),
        "a16_source_encodings_sha256": a16_encodings_hash,
        "all_fp16_source_encodings_sha256": all_encodings_hash,
        "output_encodings_sha256": output_encodings_hash,
        "vision_onnx_sha256": all_onnx_hash,
    }
    if {
        key: value
        for key, value in output_args.items()
        if key != PROVENANCE_FIELD
    } != all_args:
        raise AssertionError("Unexpected args.json field changed")
    if output_args.get("precision") != "w4":
        raise AssertionError("Output left the standard W4 precision path")
    _validate_raw_precision(
        output_args,
        expected="w4",
        path=all_fp16_source / ARGS_FILENAME,
    )
    output_args_bytes = _json_bytes(output_args)
    output_args_hash = hashlib.sha256(output_args_bytes).hexdigest()

    marker = {
        "schema_version": 1,
        "purpose": (
            "Restore calibrated A16 compiler/Genie boundaries while "
            "retaining FLOAT16 internal vision activations"
        ),
        "sources": {
            "all_fp16_checkpoint": str(all_fp16_source),
            "w4a16_checkpoint": str(a16_source),
        },
        "precision_contract": {
            "checkpoint_export_precision": "w4",
            "text_weights": "W4",
            "vision_parameters": "W8",
            "restored_boundaries": "A16",
            "remaining_vision_activations": "FLOAT16",
        },
        "calibration": calibration,
        "counts": output_summary,
        "graph_contract": graph_summary,
        "hashes": {
            "all_fp16_args_sha256": all_args_hash,
            "output_args_sha256": output_args_hash,
            "a16_source_args_sha256": a16_args_hash,
            "all_fp16_encodings_sha256": all_encodings_hash,
            "a16_source_encodings_sha256": a16_encodings_hash,
            "output_encodings_sha256": output_encodings_hash,
            "all_fp16_marker_sha256": all_marker_hash,
            "all_fp16_onnx_sha256": all_onnx_hash,
            "a16_source_onnx_sha256": a16_onnx_hash,
            "output_onnx_sha256": all_onnx_hash,
            **encoding_pair_summary,
        },
        "exact_unchanged_fields": [
            "vision_encoder.onnx bytes",
            "all 205 W8 vision parameter encodings",
            "925 non-boundary FLOAT16 activation encodings",
            "all top-level encoding metadata",
            f"all args.json fields except {PROVENANCE_FIELD}",
            "the original W4-to-FP16 marker",
            "all other all-FP16 checkpoint artifacts",
        ],
        "restored_from_w4a16_byte_for_byte": list(
            RESTORED_ACTIVATION_NAMES
        ),
    }
    marker_bytes = _json_bytes(marker)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            all_fp16_source,
            temporary,
            copy_function=copy_function,
            ignore=shutil.ignore_patterns(
                ARGS_FILENAME,
                ENCODINGS_FILENAME,
                MARKER_FILENAME,
            ),
        )
        _write_bytes(
            temporary / ENCODINGS_FILENAME,
            output_encodings_bytes,
        )
        _write_bytes(temporary / ARGS_FILENAME, output_args_bytes)
        _write_bytes(temporary / MARKER_FILENAME, marker_bytes)

        if sha256_file(temporary / MODEL_FILENAME) != all_onnx_hash:
            raise RuntimeError("Vision ONNX changed during checkpoint assembly")
        if sha256_file(temporary / ENCODINGS_FILENAME) != (
            output_encodings_hash
        ):
            raise RuntimeError("Written vision encodings hash mismatch")
        if sha256_file(temporary / ARGS_FILENAME) != output_args_hash:
            raise RuntimeError("Written args.json hash mismatch")
        if sha256_file(
            temporary / ALL_FP16_MARKER_FILENAME
        ) != all_marker_hash:
            raise RuntimeError("Original W4-to-FP16 marker changed")

        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "all_fp16_source",
        type=Path,
        help="W4 checkpoint whose 934 vision activations are FLOAT16",
    )
    parser.add_argument(
        "a16_source",
        type=Path,
        help="Matching calibrated W4A16 checkpoint",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="New hybrid checkpoint (must not already exist)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_vision_a16_boundary_checkpoint(
        args.all_fp16_source,
        args.a16_source,
        args.destination,
    )
    print(
        f"Created compiler-boundary vision checkpoint: {output} "
        f"({EXPECTED_RESTORED_COUNT} A16 boundaries; "
        f"{EXPECTED_ACTIVATION_COUNT - EXPECTED_RESTORED_COUNT} "
        "FLOAT16 internal activations; "
        f"{EXPECTED_PARAMETER_COUNT} W8 parameters)"
    )


if __name__ == "__main__":
    main()
