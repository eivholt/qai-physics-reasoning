#!/usr/bin/env python3
"""Promote selected Cosmos text matrices from W4 to independently derived W8.

This utility accepts the known CL512, AR128-calibrated, W4/FP16
full-DeepStack Cosmos-Reason2-2B checkpoint and can promote every matrix owned
by any of the four decoder partitions to W8.  An experiment may further
restrict the conversion to explicitly selected layers inside those
partitions.

The W8 ranges are recomputed from the external FP32 ONNX initializers with
AIMET's min/max, signed-symmetric, non-strict ``BlockTensorQuantizer``.  W4
scales are never multiplied by a constant.  Before producing any output, the
same AIMET path must reproduce every selected source W4 encoding exactly.

Only ``model.encodings`` changes by default.  With explicit
``--image-size HEIGHT WIDTH``, ``args.json`` may also backfill a missing
top-level image size, but the requested pair must exactly equal the existing
``raw_args --image-size`` pair and raw arguments are never rewritten.  The
utility validates that the text graph has no pixel/grid input and has dynamic
visual-token boundaries first.  The dynamic ONNX graph, external weights, all
FP16 activation encodings, FP32 graph boundaries (apart from the existing
boolean visual mask), CL512 metadata, AR128/AR1-capable dynamic axes, and all
four DeepStack inputs are preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

AIMET_ENCODING_VERSION = "1.0.0"
ARGS_FILENAME = "args.json"
ENCODINGS_FILENAME = "model.encodings"
MODEL_FILENAME = "model_dynamic.onnx"
EXTERNAL_WEIGHTS_FILENAME = "model.data"
SOURCE_FP16_MARKER = "w4_fp16.json"
MARKER_FILENAME = "text_w8_matrices.json"

DEFAULT_CONTEXT_LENGTH = 512
SUPPORTED_CONTEXT_LENGTHS = (512, 1024)
EXPECTED_CALIBRATION_SEQUENCE_LENGTH = 128
EXPECTED_LAYER_COUNT = 28
EXPECTED_LAYERS_PER_PART = 7
EXPECTED_ACTIVATION_COUNT = 9423
EXPECTED_PARAMETER_COUNT = 1738
EXPECTED_MATRICES_PER_LAYER = 36
EXPECTED_NAMED_MATRICES_PER_LAYER = 34
EXPECTED_ANONYMOUS_MATRICES_PER_LAYER = 2
EXPECTED_MATRIX_COUNT = (
    EXPECTED_LAYER_COUNT * EXPECTED_MATRICES_PER_LAYER
)
EXPECTED_INT16_PARAMETER_COUNT = 729
EXPECTED_PREEXISTING_W8_PARAMETER_COUNT = 1

SUPPORTED_PART_LAYERS = {
    "part1_of_4": tuple(range(0, 7)),
    "part2_of_4": tuple(range(7, 14)),
    "part3_of_4": tuple(range(14, 21)),
    "part4_of_4": tuple(range(21, 28)),
}
SUPPORTED_PARTS = tuple(SUPPORTED_PART_LAYERS)

_NAMED_MATRIX_RE = re.compile(
    r"model\.model\.layers\.(?P<layer>\d+)\."
    r"(?:"
    r"self_attn\."
    r"(?:"
    r"q_proj_sha\.(?P<q_shard>\d+)"
    r"|k_proj_sha\.(?P<k_shard>\d+)"
    r"|v_proj_sha\.(?P<v_shard>\d+)"
    r"|o_proj_conv"
    r")"
    r"|mlp\.down_proj"
    r")\.weight"
)
_ANONYMOUS_MATRIX_RE = re.compile(r"val_\d+")

_REQUIRED_PRESERVED_FILES = (
    "source_checkpoint.json",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "embedding_weights.raw",
    "vision_encoder.onnx",
    "vision_encoder.encodings",
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


def _raw_flag_value(raw_args: list[str], flag: str) -> str:
    indices = [
        index for index, value in enumerate(raw_args) if value == flag
    ]
    if len(indices) != 1:
        raise ValueError(
            f"args.json raw_args must contain exactly one {flag}"
        )
    index = indices[0]
    if index + 1 >= len(raw_args):
        raise ValueError(f"args.json raw_args {flag} has no value")
    return raw_args[index + 1]


def _raw_image_size(
    raw_args: list[str],
) -> tuple[tuple[int, int], int]:
    flag = "--image-size"
    indices = [
        index for index, value in enumerate(raw_args) if value == flag
    ]
    if len(indices) != 1:
        raise ValueError(
            f"args.json raw_args must contain exactly one {flag}"
        )
    index = indices[0]
    if index + 2 >= len(raw_args):
        raise ValueError(
            "args.json raw_args --image-size must have HEIGHT WIDTH"
        )
    try:
        values = (int(raw_args[index + 1]), int(raw_args[index + 2]))
    except ValueError as exc:
        raise ValueError(
            "args.json raw_args --image-size values must be integers"
        ) from exc
    if any(value <= 0 for value in values):
        raise ValueError(
            "args.json raw_args --image-size values must be positive"
        )
    return values, index


def _normalize_requested_image_size(
    image_size: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    if image_size is None:
        return None
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in image_size
        )
    ):
        raise ValueError("image_size must be [HEIGHT, WIDTH]")
    normalized = (int(image_size[0]), int(image_size[1]))
    if any(value <= 0 for value in normalized):
        raise ValueError("image_size values must be positive")
    return normalized


def _prepare_args(
    path: Path,
    *,
    requested_image_size: tuple[int, int] | list[int] | None,
    expected_context_length: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    args = _load_json_object(path)
    expected = {
        "precision": "w4",
        "context_length": expected_context_length,
        "calibration_sequence_length": (
            EXPECTED_CALIBRATION_SEQUENCE_LENGTH
        ),
        "skip_llm": False,
    }
    for field, wanted in expected.items():
        if args.get(field) != wanted:
            raise ValueError(
                f"{path}: {field} is {args.get(field)!r}; expected "
                f"{wanted!r}"
            )

    raw_args = args.get("raw_args")
    if not isinstance(raw_args, list) or not all(
        isinstance(value, str) for value in raw_args
    ):
        raise ValueError(f"{path}: raw_args must be a list of strings")
    raw_expected = {
        "--precision": "w4",
        "--context-length": str(expected_context_length),
        "--calibration-sequence-length": str(
            EXPECTED_CALIBRATION_SEQUENCE_LENGTH
        ),
    }
    for flag, wanted in raw_expected.items():
        actual = _raw_flag_value(raw_args, flag)
        if actual != wanted:
            raise ValueError(
                f"{path}: raw_args {flag} is {actual!r}; expected "
                f"{wanted!r}"
            )
    raw_image_size, _ = _raw_image_size(raw_args)
    requested = _normalize_requested_image_size(requested_image_size)

    missing = object()
    recorded = args.get("image_size", missing)
    if recorded is not missing and (
        not isinstance(recorded, list)
        or len(recorded) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in recorded
        )
    ):
        raise ValueError(
            f"{path}: image_size must be [HEIGHT, WIDTH] when present"
        )

    if requested is None:
        if recorded is missing:
            raise ValueError(
                f"{path}: top-level image_size is missing; pass explicit "
                "'--image-size HEIGHT WIDTH' matching the existing "
                "raw_args pair to backfill it"
            )
        recorded_tuple = (int(recorded[0]), int(recorded[1]))
        if recorded_tuple != raw_image_size:
            raise ValueError(
                f"{path}: top-level image_size {recorded!r} disagrees with "
                f"raw_args --image-size {list(raw_image_size)!r}"
            )
        summary = {
            "applied": False,
            "source_top_level_image_size": list(recorded_tuple),
            "source_raw_args_image_size": list(raw_image_size),
            "destination_top_level_image_size": list(recorded_tuple),
            "destination_raw_args_image_size": list(raw_image_size),
            "raw_args_changed": False,
            "changed_fields": [],
        }
        return args, args, summary

    if requested != raw_image_size:
        raise ValueError(
            f"{path}: explicit image_size {list(requested)!r} must exactly "
            f"equal raw_args --image-size {list(raw_image_size)!r}"
        )
    if recorded is not missing and tuple(recorded) != raw_image_size:
        raise ValueError(
            f"{path}: top-level image_size {recorded!r} disagrees with "
            f"raw_args --image-size {list(raw_image_size)!r}"
        )

    converted = copy.deepcopy(args)
    converted["image_size"] = list(raw_image_size)

    source_without_migrated_fields = dict(args)
    source_without_migrated_fields.pop("image_size", None)
    converted_without_migrated_fields = dict(converted)
    converted_without_migrated_fields.pop("image_size", None)
    if source_without_migrated_fields != converted_without_migrated_fields:
        raise AssertionError(
            "Unexpected args.json field changed during image-size migration"
        )

    applied = recorded is missing
    summary = {
        "applied": applied,
        "source_top_level_image_size": (
            None if recorded is missing else list(recorded)
        ),
        "source_raw_args_image_size": list(raw_image_size),
        "destination_top_level_image_size": list(raw_image_size),
        "destination_raw_args_image_size": list(raw_image_size),
        "raw_args_changed": False,
        "changed_fields": ["args.image_size"] if applied else [],
    }
    return args, converted, summary


def _validate_fp16_source_marker(path: Path) -> dict[str, Any]:
    marker = _load_json_object(path)
    try:
        precision = marker["precision"]["destination_precision"]
        summary = marker["encoding_files"][ENCODINGS_FILENAME]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: incomplete W4/FP16 provenance") from exc
    if precision != "w4":
        raise ValueError(
            f"{path}: destination precision is {precision!r}; expected 'w4'"
        )
    if (
        summary.get("activation_count") != EXPECTED_ACTIVATION_COUNT
        or summary.get("activations_converted_to_float16") is not True
        or summary.get("destination_activation_types")
        != {"FLOAT16": EXPECTED_ACTIVATION_COUNT}
    ):
        raise ValueError(
            f"{path}: source does not prove the exact "
            f"{EXPECTED_ACTIVATION_COUNT}-tensor FP16 activation contract"
        )
    replacement = marker.get("replacement_activation_encoding")
    if replacement != {
        "bw": 16,
        "dtype": "FLOAT",
        "enc_type": "PER_TENSOR",
    }:
        raise ValueError(
            f"{path}: source has an unexpected FP16 activation encoding"
        )
    return marker


def _shape_of_value_info(value_info: Any) -> list[int | str | None]:
    shape: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append(None)
    return shape


def _is_dynamic(value: int | str | None) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_fixed_shape_positions(
    *,
    name: str,
    shape: list[int | str | None],
    rank: int,
    fixed: dict[int, int],
    dynamic: tuple[int, ...],
) -> None:
    if len(shape) != rank:
        raise ValueError(
            f"{MODEL_FILENAME}: {name!r} has rank {len(shape)}, "
            f"expected {rank}"
        )
    for axis, wanted in fixed.items():
        if shape[axis] != wanted:
            raise ValueError(
                f"{MODEL_FILENAME}: {name!r} shape axis {axis} is "
                f"{shape[axis]!r}; expected {wanted}"
            )
    for axis in dynamic:
        if not _is_dynamic(shape[axis]):
            raise ValueError(
                f"{MODEL_FILENAME}: {name!r} shape axis {axis} must "
                "remain dynamic for AR128/AR1 export"
            )


def _validate_graph_io(model: Any, onnx: Any) -> list[dict[str, Any]]:
    expected_inputs = {
        "inputs_embeds",
        "attention_mask",
        "position_ids_cos",
        "position_ids_sin",
        "visual_pos_masks",
        *(f"past_key_{layer}_in" for layer in range(EXPECTED_LAYER_COUNT)),
        *(
            f"past_value_{layer}_in"
            for layer in range(EXPECTED_LAYER_COUNT)
        ),
        *(f"deepstack_visual_embeds_{index}" for index in range(3)),
    }
    expected_outputs = {
        "logits",
        *(f"past_key_{layer}_out" for layer in range(EXPECTED_LAYER_COUNT)),
        *(
            f"past_value_{layer}_out"
            for layer in range(EXPECTED_LAYER_COUNT)
        ),
    }
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != expected_inputs:
        raise ValueError(
            f"{MODEL_FILENAME}: input contract mismatch; missing "
            f"{sorted(expected_inputs - set(inputs))[:5]!r}, unexpected "
            f"{sorted(set(inputs) - expected_inputs)[:5]!r}"
        )
    if set(outputs) != expected_outputs:
        raise ValueError(
            f"{MODEL_FILENAME}: output contract mismatch; missing "
            f"{sorted(expected_outputs - set(outputs))[:5]!r}, unexpected "
            f"{sorted(set(outputs) - expected_outputs)[:5]!r}"
        )

    float_type = int(onnx.TensorProto.FLOAT)
    bool_type = int(onnx.TensorProto.BOOL)
    for name, value in inputs.items():
        expected_type = bool_type if name == "visual_pos_masks" else float_type
        actual_type = int(value.type.tensor_type.elem_type)
        if actual_type != expected_type:
            raise ValueError(
                f"{MODEL_FILENAME}: input {name!r} dtype is {actual_type}; "
                f"expected {expected_type}"
            )
    for name, value in outputs.items():
        actual_type = int(value.type.tensor_type.elem_type)
        if actual_type != float_type:
            raise ValueError(
                f"{MODEL_FILENAME}: output {name!r} is not FP32"
            )

    _validate_fixed_shape_positions(
        name="inputs_embeds",
        shape=_shape_of_value_info(inputs["inputs_embeds"]),
        rank=3,
        fixed={0: 1, 2: 2048},
        dynamic=(1,),
    )
    _validate_fixed_shape_positions(
        name="attention_mask",
        shape=_shape_of_value_info(inputs["attention_mask"]),
        rank=4,
        fixed={0: 1, 1: 1},
        dynamic=(2, 3),
    )
    for name in ("position_ids_cos", "position_ids_sin"):
        _validate_fixed_shape_positions(
            name=name,
            shape=_shape_of_value_info(inputs[name]),
            rank=4,
            fixed={0: 1, 1: 1, 3: 64},
            dynamic=(2,),
        )
    _validate_fixed_shape_positions(
        name="visual_pos_masks",
        shape=_shape_of_value_info(inputs["visual_pos_masks"]),
        rank=2,
        fixed={0: 1},
        dynamic=(1,),
    )
    for index in range(3):
        name = f"deepstack_visual_embeds_{index}"
        _validate_fixed_shape_positions(
            name=name,
            shape=_shape_of_value_info(inputs[name]),
            rank=2,
            fixed={1: 2048},
            dynamic=(0,),
        )
    for layer in range(EXPECTED_LAYER_COUNT):
        _validate_fixed_shape_positions(
            name=f"past_key_{layer}_in",
            shape=_shape_of_value_info(inputs[f"past_key_{layer}_in"]),
            rank=4,
            fixed={0: 8, 1: 1, 2: 128},
            dynamic=(3,),
        )
        _validate_fixed_shape_positions(
            name=f"past_value_{layer}_in",
            shape=_shape_of_value_info(inputs[f"past_value_{layer}_in"]),
            rank=4,
            fixed={0: 8, 1: 1, 3: 128},
            dynamic=(2,),
        )
        _validate_fixed_shape_positions(
            name=f"past_key_{layer}_out",
            shape=_shape_of_value_info(outputs[f"past_key_{layer}_out"]),
            rank=4,
            fixed={0: 8, 1: 1, 2: 128},
            dynamic=(3,),
        )
        _validate_fixed_shape_positions(
            name=f"past_value_{layer}_out",
            shape=_shape_of_value_info(outputs[f"past_value_{layer}_out"]),
            rank=4,
            fixed={0: 8, 1: 1, 3: 128},
            dynamic=(2,),
        )
    _validate_fixed_shape_positions(
        name="logits",
        shape=_shape_of_value_info(outputs["logits"]),
        rank=3,
        fixed={0: 1, 2: 151936},
        dynamic=(1,),
    )

    signature = []
    for direction, values in (
        ("input", model.graph.input),
        ("output", model.graph.output),
    ):
        for value in values:
            signature.append(
                {
                    "direction": direction,
                    "name": value.name,
                    "dtype": int(value.type.tensor_type.elem_type),
                    "shape": _shape_of_value_info(value),
                }
            )
    return signature


def _validate_image_size_independent_text_interface(
    model: Any,
) -> dict[str, Any]:
    """Prove the text graph has no pixel/grid geometry boundary.

    This does not claim that arbitrary vision encoders are interchangeable.
    It proves the metadata migration cannot alter this ONNX model's tensor
    contract: the text graph accepts only already-merged visual embeddings,
    and every sequence/visual-token dimension at that boundary is dynamic.
    """
    inputs = {value.name: value for value in model.graph.input}
    forbidden = {
        "pixel_values",
        "image_grid_thw",
        "video_grid_thw",
        "window_index",
        "full_attention_mask",
    }
    present_forbidden = sorted(forbidden & set(inputs))
    if present_forbidden:
        raise ValueError(
            f"{MODEL_FILENAME}: text graph unexpectedly has vision-geometry "
            f"inputs {present_forbidden!r}"
        )
    visual_inputs = {
        "visual_pos_masks": (1,),
        "deepstack_visual_embeds_0": (0,),
        "deepstack_visual_embeds_1": (0,),
        "deepstack_visual_embeds_2": (0,),
    }
    for name, dynamic_axes in visual_inputs.items():
        if name not in inputs:
            raise ValueError(
                f"{MODEL_FILENAME}: full DeepStack input {name!r} is missing"
            )
        shape = _shape_of_value_info(inputs[name])
        for axis in dynamic_axes:
            if not _is_dynamic(shape[axis]):
                raise ValueError(
                    f"{MODEL_FILENAME}: visual boundary {name!r} axis "
                    f"{axis} is static; image-size metadata migration is "
                    "unsafe"
                )
    return {
        "pixel_or_grid_inputs": [],
        "visual_boundary_inputs": sorted(visual_inputs),
        "visual_token_axes": "dynamic",
        "scope": (
            "text ONNX interface only; vision context must independently "
            "match the selected image geometry"
        ),
    }


def _validate_activation_encodings(
    entries: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    if len(entries) != EXPECTED_ACTIVATION_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_ACTIVATION_COUNT} "
            f"activation encodings, found {len(entries)}"
        )
    for entry in entries:
        wanted = {
            "name": entry.get("name"),
            "bw": 16,
            "dtype": "FLOAT",
            "enc_type": "PER_TENSOR",
        }
        if entry != wanted:
            raise ValueError(
                f"{path}: activation {entry.get('name')!r} is not the "
                "canonical FLOAT16 encoding"
            )


def _validate_parameter_histogram(
    entries: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    if len(entries) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_PARAMETER_COUNT} "
            f"parameter encodings, found {len(entries)}"
        )
    int16 = [
        entry
        for entry in entries
        if entry.get("dtype") == "INT" and entry.get("bw") == 16
    ]
    int8 = [
        entry
        for entry in entries
        if entry.get("dtype") == "INT" and entry.get("bw") == 8
    ]
    int4 = [
        entry
        for entry in entries
        if entry.get("dtype") == "INT" and entry.get("bw") == 4
    ]
    if len(int16) != EXPECTED_INT16_PARAMETER_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_INT16_PARAMETER_COUNT} INT16 "
            f"parameters, found {len(int16)}"
        )
    if len(int8) != EXPECTED_PREEXISTING_W8_PARAMETER_COUNT:
        raise ValueError(
            f"{path}: expected one pre-existing W8 parameter, found "
            f"{len(int8)}"
        )
    if (
        len(int8) != 1
        or int8[0].get("name") != "model.lm_head.weight"
        or int8[0].get("enc_type") != "PER_CHANNEL"
        or int8[0].get("is_sym") is not True
    ):
        raise ValueError(
            f"{path}: the sole pre-existing W8 parameter must be the "
            "symmetric per-channel model.lm_head.weight"
        )
    if len(int4) != EXPECTED_MATRIX_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_MATRIX_COUNT} W4 matrices, "
            f"found {len(int4)}"
        )
    if len(int16) + len(int8) + len(int4) != len(entries):
        raise ValueError(
            f"{path}: found an unsupported parameter encoding type"
        )


def _classify_layer_matrices(
    model: Any,
    parameters: list[dict[str, Any]],
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    w4_entries = [
        entry
        for entry in parameters
        if entry.get("dtype") == "INT" and entry.get("bw") == 4
    ]
    named_by_layer: dict[int, list[str]] = {
        layer: [] for layer in range(EXPECTED_LAYER_COUNT)
    }
    anonymous: list[str] = []
    for entry in w4_entries:
        name = entry["name"]
        match = _NAMED_MATRIX_RE.fullmatch(name)
        if match is not None:
            layer = int(match.group("layer"))
            if layer not in named_by_layer:
                raise ValueError(
                    f"{ENCODINGS_FILENAME}: matrix {name!r} has layer "
                    f"{layer}, outside 0..{EXPECTED_LAYER_COUNT - 1}"
                )
            named_by_layer[layer].append(name)
        elif _ANONYMOUS_MATRIX_RE.fullmatch(name) is not None:
            anonymous.append(name)
        else:
            raise ValueError(
                f"{ENCODINGS_FILENAME}: unexpected W4 matrix name {name!r}"
            )

    for layer, names in named_by_layer.items():
        if len(names) != EXPECTED_NAMED_MATRICES_PER_LAYER:
            raise ValueError(
                f"{ENCODINGS_FILENAME}: layer {layer} has {len(names)} "
                f"named W4 matrices; expected "
                f"{EXPECTED_NAMED_MATRICES_PER_LAYER}"
            )
        q_shards: set[int] = set()
        k_shards: set[int] = set()
        v_shards: set[int] = set()
        suffixes: list[str] = []
        for name in names:
            match = _NAMED_MATRIX_RE.fullmatch(name)
            assert match is not None
            if match.group("q_shard") is not None:
                q_shards.add(int(match.group("q_shard")))
            elif match.group("k_shard") is not None:
                k_shards.add(int(match.group("k_shard")))
            elif match.group("v_shard") is not None:
                v_shards.add(int(match.group("v_shard")))
            else:
                suffixes.append(name)
        if q_shards != set(range(16)):
            raise ValueError(
                f"{ENCODINGS_FILENAME}: layer {layer} q shards are "
                f"{sorted(q_shards)!r}; expected 0..15"
            )
        if k_shards != set(range(8)) or v_shards != set(range(8)):
            raise ValueError(
                f"{ENCODINGS_FILENAME}: layer {layer} must have k/v "
                "shards 0..7"
            )
        if (
            sum(name.endswith("self_attn.o_proj_conv.weight") for name in names)
            != 1
            or sum(name.endswith("mlp.down_proj.weight") for name in names)
            != 1
        ):
            raise ValueError(
                f"{ENCODINGS_FILENAME}: layer {layer} is missing its "
                "o_proj or down_proj matrix"
            )

    matrix_names = {entry["name"] for entry in w4_entries}
    consumers: dict[str, list[tuple[int, str]]] = {
        name: [] for name in matrix_names
    }
    for node_index, node in enumerate(model.graph.node):
        for value in node.input:
            if value in consumers:
                consumers[value].append((node_index, node.op_type))
    invalid_consumers = {
        name: found
        for name, found in consumers.items()
        if len(found) != 1
    }
    if invalid_consumers:
        name = sorted(invalid_consumers)[0]
        raise ValueError(
            f"{MODEL_FILENAME}: matrix {name!r} must have exactly one "
            f"consumer, found {invalid_consumers[name]!r}"
        )

    for names in named_by_layer.values():
        for name in names:
            if consumers[name][0][1] != "Conv":
                raise ValueError(
                    f"{MODEL_FILENAME}: named matrix {name!r} is not "
                    "consumed by Conv"
                )
    for name in anonymous:
        if consumers[name][0][1] != "MatMul":
            raise ValueError(
                f"{MODEL_FILENAME}: anonymous matrix {name!r} is not "
                "consumed by MatMul"
            )

    layer_start = {
        layer: min(consumers[name][0][0] for name in names)
        for layer, names in named_by_layer.items()
    }
    starts = [layer_start[layer] for layer in range(EXPECTED_LAYER_COUNT)]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError(
            f"{MODEL_FILENAME}: text layers are not in strict graph order"
        )

    anonymous_by_layer: dict[int, list[str]] = {}
    for layer in range(EXPECTED_LAYER_COUNT):
        lower = layer_start[layer]
        upper = (
            layer_start[layer + 1]
            if layer + 1 < EXPECTED_LAYER_COUNT
            else math.inf
        )
        names = sorted(
            (
                name
                for name in anonymous
                if lower <= consumers[name][0][0] < upper
            ),
            key=lambda name: consumers[name][0][0],
        )
        if len(names) != EXPECTED_ANONYMOUS_MATRICES_PER_LAYER:
            raise ValueError(
                f"{MODEL_FILENAME}: layer {layer} has {len(names)} "
                "anonymous W4 MatMul matrices; expected 2"
            )
        anonymous_by_layer[layer] = names

    by_layer = {
        layer: [
            *sorted(named_by_layer[layer]),
            *anonymous_by_layer[layer],
        ]
        for layer in range(EXPECTED_LAYER_COUNT)
    }
    for layer, names in by_layer.items():
        if len(names) != EXPECTED_MATRICES_PER_LAYER:
            raise AssertionError(
                f"Layer {layer} matrix classification is incomplete"
            )
    if {
        name for names in by_layer.values() for name in names
    } != matrix_names:
        raise AssertionError("Layer matrix classification lost a W4 tensor")
    return by_layer, anonymous_by_layer


def _external_tensor_array(
    tensor: Any,
    *,
    checkpoint: Path,
    onnx: Any,
    np: Any,
) -> tuple[Any, dict[str, Any]]:
    if int(tensor.data_type) != int(onnx.TensorProto.FLOAT):
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} is not FP32"
        )
    if tensor.raw_data:
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} is embedded; "
            "expected one external FP32 source"
        )
    external = {item.key: item.value for item in tensor.external_data}
    if not {"location", "offset", "length"} <= set(external):
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} has incomplete "
            "external-data metadata"
        )
    location = external["location"]
    if not location or Path(location).is_absolute():
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} has unsafe "
            f"external-data location {location!r}"
        )
    data_path = (checkpoint / location).resolve()
    if checkpoint not in data_path.parents:
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} external data "
            "escapes the checkpoint"
        )
    if not data_path.is_file():
        raise ValueError(
            f"{MODEL_FILENAME}: external data does not exist: {data_path}"
        )
    try:
        offset = int(external["offset"])
        length = int(external["length"])
    except ValueError as exc:
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} has invalid "
            "external offset/length"
        ) from exc
    dims = tuple(int(value) for value in tensor.dims)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} has invalid "
            f"shape {dims!r}"
        )
    expected_length = math.prod(dims) * 4
    if offset < 0 or length != expected_length:
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} external "
            f"length is {length}; expected {expected_length}"
        )
    if offset + length > data_path.stat().st_size:
        raise ValueError(
            f"{MODEL_FILENAME}: initializer {tensor.name!r} external "
            "range exceeds its data file"
        )
    array = np.memmap(
        data_path,
        dtype=np.float32,
        mode="r",
        offset=offset,
        shape=dims,
        order="C",
    )
    return array, {
        "location": location,
        "offset": offset,
        "length": length,
        "shape": list(dims),
    }


def _channel_axis(
    entry: dict[str, Any],
    shape: tuple[int, ...],
) -> int:
    name = entry["name"]
    scales = entry.get("scale")
    offsets = entry.get("offset")
    if not isinstance(scales, list) or not isinstance(offsets, list):
        raise ValueError(
            f"{ENCODINGS_FILENAME}: {entry['name']!r} has no scale/offset "
            "lists"
        )
    if not scales or len(offsets) != len(scales):
        raise ValueError(
            f"{ENCODINGS_FILENAME}: {name!r} scale/offset lengths "
            "do not match"
        )

    # Do not infer this from dimensions alone: o_proj is square
    # [out_channels, in_channels, 1, 1], so both axes 0 and 1 equal the
    # per-channel encoding count.  The graph classifier has already required
    # every named matrix to be a Conv and every anonymous matrix to be a
    # MatMul.  Match the corresponding QAIHM layout explicitly.
    if _NAMED_MATRIX_RE.fullmatch(name) is not None:
        if len(shape) != 4 or shape[2:] != (1, 1):
            raise ValueError(
                f"{MODEL_FILENAME}: named Conv matrix {name!r} has shape "
                f"{shape!r}; expected [out_channels, in_channels, 1, 1]"
            )
        axis = 0
    elif _ANONYMOUS_MATRIX_RE.fullmatch(name) is not None:
        if len(shape) != 2:
            raise ValueError(
                f"{MODEL_FILENAME}: anonymous MatMul matrix {name!r} has "
                f"shape {shape!r}; expected [in_channels, out_channels]"
            )
        axis = 1
    else:
        raise ValueError(
            f"{ENCODINGS_FILENAME}: selected matrix {name!r} has no "
            "validated Conv/MatMul taxonomy"
        )
    if shape[axis] <= 1 or shape[axis] != len(scales):
        raise ValueError(
            f"{ENCODINGS_FILENAME}: {name!r} channel axis {axis} has "
            f"dimension {shape[axis]}, but there are {len(scales)} "
            "per-channel encodings"
        )
    return axis


def _aimet_minmax_encodings(
    array: Any,
    *,
    channel_axis: int,
    bitwidth: int,
    libpymo: Any,
) -> tuple[list[float], list[float]]:
    quantizer_shape = [
        int(dimension) if axis == channel_axis else 1
        for axis, dimension in enumerate(array.shape)
    ]
    quantizer = libpymo.BlockTensorQuantizer(
        quantizer_shape,
        bitwidth,
        libpymo.QuantizationMode.QUANTIZATION_TF,
    )
    quantizer.setStrictSymmetric(False)
    quantizer.setUnsignedSymmetric(False)
    quantizer.updateStats(array)
    computed = quantizer.computeEncodings(True)
    if len(computed) != int(array.shape[channel_axis]):
        raise RuntimeError(
            "AIMET returned an unexpected per-channel encoding count"
        )
    scales = [float(encoding.delta) for encoding in computed]
    offsets = [float(encoding.offset) for encoding in computed]
    if any(not math.isfinite(value) or value <= 0 for value in scales):
        raise RuntimeError("AIMET returned a non-positive W8 scale")
    expected_offset = float(-(2 ** (bitwidth - 1)))
    if any(value != expected_offset for value in offsets):
        raise RuntimeError(
            "AIMET did not produce signed, non-strict symmetric offsets"
        )
    return scales, offsets


def _raw_tensor_sha256(array: Any) -> str:
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    chunk = 1024 * 1024
    for offset in range(0, len(view), chunk):
        digest.update(view[offset : offset + chunk])
    return digest.hexdigest()


def _recompute_target_encodings(
    *,
    checkpoint: Path,
    model: Any,
    encodings: dict[str, Any],
    target_names: set[str],
    onnx: Any,
    np: Any,
    libpymo: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initializers: dict[str, Any] = {}
    for tensor in model.graph.initializer:
        if tensor.name in initializers:
            raise ValueError(
                f"{MODEL_FILENAME}: duplicate initializer {tensor.name!r}"
            )
        initializers[tensor.name] = tensor
    missing = sorted(target_names - set(initializers))
    if missing:
        raise ValueError(
            f"{MODEL_FILENAME}: selected W4 matrices have no initializer: "
            f"{missing[:5]!r}"
        )

    converted = dict(encodings)
    converted_parameters: list[dict[str, Any]] = []
    raw_descriptors: list[dict[str, Any]] = []
    for entry in encodings["param_encodings"]:
        name = entry["name"]
        if name not in target_names:
            converted_parameters.append(entry)
            continue
        if set(entry) != {
            "name",
            "bw",
            "dtype",
            "enc_type",
            "is_sym",
            "scale",
            "offset",
        }:
            raise ValueError(
                f"{ENCODINGS_FILENAME}: selected matrix {name!r} has an "
                "unexpected AIMET encoding schema"
            )
        if (
            entry.get("dtype") != "INT"
            or entry.get("bw") != 4
            or entry.get("enc_type") != "PER_CHANNEL"
            or entry.get("is_sym") is not True
        ):
            raise ValueError(
                f"{ENCODINGS_FILENAME}: selected matrix {name!r} is not "
                "signed symmetric per-channel W4"
            )

        array, descriptor = _external_tensor_array(
            initializers[name],
            checkpoint=checkpoint,
            onnx=onnx,
            np=np,
        )
        try:
            axis = _channel_axis(entry, tuple(int(x) for x in array.shape))
            source_scales, source_offsets = _aimet_minmax_encodings(
                array,
                channel_axis=axis,
                bitwidth=4,
                libpymo=libpymo,
            )
            if (
                source_scales != entry["scale"]
                or source_offsets != entry["offset"]
            ):
                raise ValueError(
                    f"{ENCODINGS_FILENAME}: AIMET cannot reproduce source "
                    f"W4 encoding for {name!r} from the raw FP32 weights"
                )
            w8_scales, w8_offsets = _aimet_minmax_encodings(
                array,
                channel_axis=axis,
                bitwidth=8,
                libpymo=libpymo,
            )
            descriptor.update(
                {
                    "name": name,
                    "channel_axis": axis,
                    "raw_sha256": _raw_tensor_sha256(array),
                }
            )
        finally:
            del array

        replacement = dict(entry)
        replacement["bw"] = 8
        replacement["scale"] = w8_scales
        replacement["offset"] = w8_offsets
        converted_parameters.append(replacement)
        raw_descriptors.append(descriptor)

    converted["param_encodings"] = converted_parameters
    if len(raw_descriptors) != len(target_names):
        raise AssertionError("Not every selected matrix was converted")
    return converted, raw_descriptors


def _validate_top_level_quantizer_contract(encodings: dict[str, Any]) -> None:
    if encodings.get("version") != AIMET_ENCODING_VERSION:
        raise ValueError(
            f"{ENCODINGS_FILENAME}: encoding version is "
            f"{encodings.get('version')!r}; expected "
            f"{AIMET_ENCODING_VERSION!r}"
        )
    quantizer_args = encodings.get("quantizer_args")
    if not isinstance(quantizer_args, dict):
        raise ValueError(
            f"{ENCODINGS_FILENAME}: quantizer_args is missing"
        )
    expected = {
        "activation_bitwidth": 16,
        "dtype": "int",
        "is_symmetric": True,
        "param_bitwidth": 4,
        "per_channel_quantization": True,
        "quant_scheme": "min_max",
    }
    for field, wanted in expected.items():
        if quantizer_args.get(field) != wanted:
            raise ValueError(
                f"{ENCODINGS_FILENAME}: quantizer_args.{field} is "
                f"{quantizer_args.get(field)!r}; expected {wanted!r}"
            )


def _load_onnx_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        from aimet_onnx.common import libpymo
    except ImportError as exc:
        raise RuntimeError(
            "This transform requires numpy, onnx, and aimet_onnx in the "
            "QAI quantization environment"
        ) from exc
    return onnx, np, libpymo


def _normalize_parts(parts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not parts:
        raise ValueError("At least one text part must be selected")
    unknown = sorted(set(parts) - set(SUPPORTED_PARTS))
    if unknown:
        raise ValueError(
            f"Unsupported text parts {unknown!r}; choose from "
            f"{list(SUPPORTED_PARTS)!r}"
        )
    if len(parts) != len(set(parts)):
        raise ValueError("Text parts must not be repeated")
    return tuple(part for part in SUPPORTED_PARTS if part in parts)


def _normalize_layers(
    selected_parts: tuple[str, ...],
    layers: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    available = tuple(
        layer
        for part in selected_parts
        for layer in SUPPORTED_PART_LAYERS[part]
    )
    if layers is None:
        return available
    if not layers:
        raise ValueError("Explicit text layers must not be empty")
    if any(
        not isinstance(layer, int) or isinstance(layer, bool)
        for layer in layers
    ):
        raise ValueError("Explicit text layers must be integers")
    if len(layers) != len(set(layers)):
        raise ValueError("Explicit text layers must not be repeated")
    unsupported = sorted(set(layers) - set(available))
    if unsupported:
        raise ValueError(
            f"Explicit text layers {unsupported!r} are outside selected "
            f"parts {list(selected_parts)!r}"
        )
    empty_parts = [
        part
        for part in selected_parts
        if not set(layers).intersection(SUPPORTED_PART_LAYERS[part])
    ]
    if empty_parts:
        raise ValueError(
            "Every selected text part must contain at least one explicit "
            f"layer; empty selections: {empty_parts!r}"
        )
    return tuple(layer for layer in available if layer in layers)


def _verify_preserved_file(source: Path, destination: Path) -> None:
    if not destination.is_file():
        raise RuntimeError(f"Copied checkpoint is missing {destination}")
    try:
        if os.path.samefile(source, destination):
            return
    except OSError:
        pass
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Preserved file changed during copy: {source.name}")


def create_text_w8_matrix_checkpoint(
    source: Path,
    destination: Path,
    *,
    parts: tuple[str, ...] | list[str],
    layers: tuple[int, ...] | list[int] | None = None,
    image_size: tuple[int, int] | list[int] | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    copy_function: Callable[[str, str], str] = _hardlink_or_copy,
) -> Path:
    """Create one mixed W8/W4 text checkpoint transactionally."""
    selected_parts = _normalize_parts(parts)
    selected_layers = _normalize_layers(selected_parts, layers)
    if context_length not in SUPPORTED_CONTEXT_LENGTHS:
        raise ValueError(
            f"context_length must be one of {SUPPORTED_CONTEXT_LENGTHS}; "
            f"got {context_length!r}"
        )
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
    if (source / MARKER_FILENAME).exists():
        raise ValueError(
            f"Source already contains {MARKER_FILENAME}; always derive "
            "a precision candidate from the pristine W4/FP16 checkpoint"
        )

    required = [
        source / ARGS_FILENAME,
        source / ENCODINGS_FILENAME,
        source / MODEL_FILENAME,
        source / EXTERNAL_WEIGHTS_FILENAME,
        source / SOURCE_FP16_MARKER,
        *(source / name for name in _REQUIRED_PRESERVED_FILES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Source checkpoint is incomplete. Missing: " + ", ".join(missing)
        )

    source_args, converted_args, image_size_migration = _prepare_args(
        source / ARGS_FILENAME,
        requested_image_size=image_size,
        expected_context_length=context_length,
    )
    _validate_fp16_source_marker(source / SOURCE_FP16_MARKER)
    encodings = _load_json_object(source / ENCODINGS_FILENAME)
    _validate_top_level_quantizer_contract(encodings)
    activation_by_name = _entries_by_name(
        encodings.get("activation_encodings"),
        field="activation_encodings",
        path=source / ENCODINGS_FILENAME,
    )
    parameter_by_name = _entries_by_name(
        encodings.get("param_encodings"),
        field="param_encodings",
        path=source / ENCODINGS_FILENAME,
    )
    overlap = sorted(set(activation_by_name) & set(parameter_by_name))
    if overlap:
        raise ValueError(
            f"{ENCODINGS_FILENAME}: activation/parameter names overlap: "
            f"{overlap[:5]!r}"
        )
    _validate_activation_encodings(
        encodings["activation_encodings"],
        path=source / ENCODINGS_FILENAME,
    )
    _validate_parameter_histogram(
        encodings["param_encodings"],
        path=source / ENCODINGS_FILENAME,
    )

    onnx, np, libpymo = _load_onnx_dependencies()
    try:
        model = onnx.load_model(
            source / MODEL_FILENAME,
            load_external_data=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot load ONNX graph without external data: "
            f"{source / MODEL_FILENAME}"
        ) from exc
    io_signature = _validate_graph_io(model, onnx)
    image_size_independence = (
        _validate_image_size_independent_text_interface(model)
    )
    by_layer, anonymous_by_layer = _classify_layer_matrices(
        model,
        encodings["param_encodings"],
    )
    target_names = {
        name for layer in selected_layers for name in by_layer[layer]
    }
    expected_target_count = (
        len(selected_layers) * EXPECTED_MATRICES_PER_LAYER
    )
    if len(target_names) != expected_target_count:
        raise AssertionError("Selected text matrix count is inconsistent")

    source_activation_hash = _canonical_sha256(
        encodings["activation_encodings"]
    )
    source_untargeted_parameters = [
        entry
        for entry in encodings["param_encodings"]
        if entry["name"] not in target_names
    ]
    converted, raw_descriptors = _recompute_target_encodings(
        checkpoint=source,
        model=model,
        encodings=encodings,
        target_names=target_names,
        onnx=onnx,
        np=np,
        libpymo=libpymo,
    )
    if converted["activation_encodings"] != encodings["activation_encodings"]:
        raise AssertionError("Activation encodings changed")
    converted_untargeted_parameters = [
        entry
        for entry in converted["param_encodings"]
        if entry["name"] not in target_names
    ]
    if converted_untargeted_parameters != source_untargeted_parameters:
        raise AssertionError("An untargeted parameter encoding changed")
    if {
        entry["name"]
        for entry in converted["param_encodings"]
        if entry.get("dtype") == "INT" and entry.get("bw") == 8
    } != {"model.lm_head.weight", *target_names}:
        raise AssertionError("Output W8 parameter set is inconsistent")
    converted_bytes = _json_bytes(converted)
    source_args_bytes = (source / ARGS_FILENAME).read_bytes()
    converted_args_bytes = (
        _json_bytes(converted_args)
        if image_size_migration["applied"]
        else source_args_bytes
    )
    if not image_size_migration["applied"] and converted_args != source_args:
        raise AssertionError("Unrequested args.json migration was detected")

    model_hash = sha256_file(source / MODEL_FILENAME)
    marker = {
        "schema_version": 1,
        "purpose": (
            "Evaluate independently derived W8 text matrices in selected "
            "Cosmos decoder partitions while preserving FP16 activations"
        ),
        "source_checkpoint": str(source),
        "selected_parts": list(selected_parts),
        "selected_layers": list(selected_layers),
        "selection_mode": (
            "full_parts" if layers is None else "explicit_layers"
        ),
        "image_size_metadata_migration": image_size_migration,
        "text_graph_image_size_independence": image_size_independence,
        "counts": {
            "activations_preserved_float16": EXPECTED_ACTIVATION_COUNT,
            "matrices_promoted_w4_to_w8": len(target_names),
            "matrices_per_layer": EXPECTED_MATRICES_PER_LAYER,
            "preexisting_w8_parameters_preserved": (
                EXPECTED_PREEXISTING_W8_PARAMETER_COUNT
            ),
            "untargeted_parameters_preserved": len(
                source_untargeted_parameters
            ),
        },
        "derivation": {
            "backend": "AIMET BlockTensorQuantizer",
            "quant_scheme": "min_max / QUANTIZATION_TF",
            "source_bitwidth_reproduced": 4,
            "destination_bitwidth": 8,
            "per_channel": True,
            "signed_symmetric": True,
            "strict_symmetric": False,
            "raw_weight_dtype": "FP32",
            "source_w4_reproduction_required": "exact",
        },
        "export_contract": {
            "context_length": context_length,
            "sequence_lengths": [
                EXPECTED_CALIBRATION_SEQUENCE_LENGTH,
                1,
            ],
            "activation_precision": "FP16",
            "external_tensor_precision": (
                "FP32, except visual_pos_masks BOOL"
            ),
            "deepstack_inputs": [
                "visual_pos_masks",
                "deepstack_visual_embeds_0",
                "deepstack_visual_embeds_1",
                "deepstack_visual_embeds_2",
            ],
            "image_size": image_size_migration[
                "destination_top_level_image_size"
            ],
        },
        "anonymous_matrices_by_selected_layer": {
            str(layer): anonymous_by_layer[layer]
            for layer in selected_layers
        },
        "hashes": {
            "source_args_sha256": hashlib.sha256(
                source_args_bytes
            ).hexdigest(),
            "output_args_sha256": hashlib.sha256(
                converted_args_bytes
            ).hexdigest(),
            "source_model_encodings_sha256": sha256_file(
                source / ENCODINGS_FILENAME
            ),
            "output_model_encodings_sha256": hashlib.sha256(
                converted_bytes
            ).hexdigest(),
            "activation_encodings_canonical_sha256": (
                source_activation_hash
            ),
            "untargeted_parameter_encodings_canonical_sha256": (
                _canonical_sha256(source_untargeted_parameters)
            ),
            "target_source_w4_encodings_canonical_sha256": (
                _canonical_sha256(
                    [
                        entry
                        for entry in encodings["param_encodings"]
                        if entry["name"] in target_names
                    ]
                )
            ),
            "target_output_w8_encodings_canonical_sha256": (
                _canonical_sha256(
                    [
                        entry
                        for entry in converted["param_encodings"]
                        if entry["name"] in target_names
                    ]
                )
            ),
            "target_raw_initializer_descriptors_canonical_sha256": (
                _canonical_sha256(raw_descriptors)
            ),
            "model_dynamic_onnx_sha256": model_hash,
            "graph_io_signature_canonical_sha256": _canonical_sha256(
                io_signature
            ),
            "text_graph_image_size_independence_canonical_sha256": (
                _canonical_sha256(image_size_independence)
            ),
        },
        "exact_unchanged": [
            "model_dynamic.onnx bytes and graph topology",
            "model.data external FP32 weights",
            "all 9423 FLOAT16 activation encodings",
            "all untargeted parameter encodings",
            "all graph input and output names, dtypes, shapes, and order",
            "CL512 checkpoint metadata",
            "dynamic sequence axes used for AR128 and AR1 export",
            "visual_pos_masks and all three DeepStack inputs",
            "embedding table, vision encoder, tokenizer, and configuration",
            *(
                []
                if image_size_migration["applied"]
                else ["args.json bytes"]
            ),
        ],
        "args_changes": (
            image_size_migration["changed_fields"]
            if image_size_migration["applied"]
            else []
        ),
    }
    marker_bytes = _json_bytes(marker)

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary destination already exists: {temporary}"
        )
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
        (temporary / ARGS_FILENAME).write_bytes(converted_args_bytes)
        (temporary / ENCODINGS_FILENAME).write_bytes(converted_bytes)
        (temporary / MARKER_FILENAME).write_bytes(marker_bytes)

        if sha256_file(temporary / ENCODINGS_FILENAME) != marker["hashes"][
            "output_model_encodings_sha256"
        ]:
            raise RuntimeError("Written model encodings hash mismatch")
        if sha256_file(temporary / ARGS_FILENAME) != marker["hashes"][
            "output_args_sha256"
        ]:
            raise RuntimeError("Written args.json hash mismatch")
        _verify_preserved_file(
            source / MODEL_FILENAME,
            temporary / MODEL_FILENAME,
        )
        _verify_preserved_file(
            source / EXTERNAL_WEIGHTS_FILENAME,
            temporary / EXTERNAL_WEIGHTS_FILENAME,
        )
        if _canonical_sha256(
            _load_json_object(temporary / ENCODINGS_FILENAME)[
                "activation_encodings"
            ]
        ) != marker["hashes"]["activation_encodings_canonical_sha256"]:
            raise RuntimeError("Written activation encodings changed")
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
        help="Pristine w4-fp16-full-deepstack-cl512-r1 checkpoint",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="New mixed W8/W4 checkpoint (must not already exist)",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=SUPPORTED_PARTS,
        required=True,
        help=(
            "Decoder partitions whose 36 matrices/layer are recomputed as "
            "W8. All four seven-layer decoder partitions are supported."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optionally promote only these decoder layers inside --parts. "
            "Every selected part must contribute at least one layer."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        choices=SUPPORTED_CONTEXT_LENGTHS,
        default=DEFAULT_CONTEXT_LENGTH,
        help=(
            "Expected text context length. The source args.json and raw "
            "export arguments must both match this value."
        ),
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help=(
            "Backfill missing top-level text-export image_size metadata. "
            "HEIGHT WIDTH must exactly match the existing raw_args "
            "--image-size pair; raw_args are preserved. This does not "
            "convert or authorize exporting the checkpoint's vision encoder."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_text_w8_matrix_checkpoint(
        args.source,
        args.destination,
        parts=args.parts,
        layers=args.layers,
        image_size=args.image_size,
        context_length=args.context_length,
    )
    marker = _load_json_object(output / MARKER_FILENAME)
    print(
        f"Created targeted text W8 checkpoint: {output} "
        f"({marker['counts']['matrices_promoted_w4_to_w8']} matrices; "
        f"parts={','.join(marker['selected_parts'])}; "
        f"layers={','.join(str(x) for x in marker['selected_layers'])})"
    )


if __name__ == "__main__":
    main()
