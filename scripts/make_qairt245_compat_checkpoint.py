#!/usr/bin/env python3
"""Create a QAIRT 2.45-compatible Cosmos quantized checkpoint.

QAIRT 2.45's legacy Genie 1.17 runtime cannot bind the Qwen3-VL deepstack
inputs, and its vision pipeline does not support the corresponding wildcard
connector. This script preserves the calibrated model and encodings while
replacing each optional deepstack injection with an exact-zero residual.

This transform addresses that binding contract only. It does not fix unrelated
token, MRoPE, sampling, or orchestration behavior in a runtime.

The primary vision embeddings remain enabled. Only the three auxiliary
deepstack injections are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import onnx

MODEL_FILENAME = "model_dynamic.onnx"
MANIFEST_FILENAME = "qairt_245_compat.json"
MAIN_EMBEDDING_INPUT = "inputs_embeds"
VISUAL_MASK_INPUT = "visual_pos_masks"
DEEPSTACK_INPUT_PREFIX = "deepstack_visual_embeds_"


@dataclass(frozen=True)
class DeepstackInjection:
    """Names whose calibrated encodings must survive one disabled injection."""

    source_input: str
    zero_tensor: str
    add_output: str
    zero_node: str
    add_node: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link checkpoint payloads when possible, then fall back to copying."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        else:
            result.append(dimension.dim_param)
    return result


def _node_label(node: onnx.NodeProto) -> str:
    return node.name or (node.output[0] if node.output else f"<{node.op_type}>")


def _graph_dependencies(
    model: onnx.ModelProto,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for output_name in node.output:
            if not output_name:
                continue
            if output_name in producers:
                raise ValueError(
                    f"Multiple ONNX nodes produce tensor {output_name!r}"
                )
            producers[output_name] = index
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(index)
    return producers, consumers


def _first_add_consumers(
    model: onnx.ModelProto,
    consumers: dict[str, list[int]],
    source_tensor: str,
) -> list[tuple[int, str]]:
    """Return first Add nodes reachable from ``source_tensor`` and their inputs."""
    queue = deque([source_tensor])
    visited_tensors = {source_tensor}
    visited_nodes: set[int] = set()
    results: list[tuple[int, str]] = []

    while queue:
        tensor_name = queue.popleft()
        for node_index in consumers.get(tensor_name, []):
            if node_index in visited_nodes:
                continue
            visited_nodes.add(node_index)
            node = model.graph.node[node_index]
            if node.op_type == "Add":
                results.append((node_index, tensor_name))
                continue
            for output_name in node.output:
                if output_name and output_name not in visited_tensors:
                    visited_tensors.add(output_name)
                    queue.append(output_name)
    return results


def _prune_unreachable_graph(model: onnx.ModelProto) -> set[str]:
    """Remove pure ONNX nodes and inputs that cannot affect a graph output."""
    if model.graph.sparse_initializer:
        raise ValueError("Sparse initializers are not supported by safe graph pruning")
    for node in model.graph.node:
        if any(
            attribute.type
            in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for attribute in node.attribute
        ):
            raise ValueError("Control-flow subgraphs are not supported by safe pruning")

    producers, _ = _graph_dependencies(model)
    required_tensors = {output.name for output in model.graph.output}
    pending = deque(required_tensors)
    required_nodes: set[int] = set()

    while pending:
        tensor_name = pending.popleft()
        producer_index = producers.get(tensor_name)
        if producer_index is None or producer_index in required_nodes:
            continue
        required_nodes.add(producer_index)
        for input_name in model.graph.node[producer_index].input:
            if input_name and input_name not in required_tensors:
                required_tensors.add(input_name)
                pending.append(input_name)

    retained_nodes = [
        node
        for index, node in enumerate(model.graph.node)
        if index in required_nodes
    ]
    used_tensors = {
        tensor_name
        for node in retained_nodes
        for tensor_name in (*node.input, *node.output)
        if tensor_name
    }
    used_tensors.update(output.name for output in model.graph.output)

    original_inputs = {value.name for value in model.graph.input}
    retained_inputs = [
        value for value in model.graph.input if value.name in used_tensors
    ]
    retained_initializers = [
        value for value in model.graph.initializer if value.name in used_tensors
    ]
    retained_value_info = [
        value for value in model.graph.value_info if value.name in used_tensors
    ]
    retained_annotations = [
        annotation
        for annotation in model.graph.quantization_annotation
        if annotation.tensor_name in used_tensors
    ]

    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    del model.graph.quantization_annotation[:]
    model.graph.quantization_annotation.extend(retained_annotations)

    return original_inputs - {value.name for value in retained_inputs}


def _disable_deepstack_injections(
    model: onnx.ModelProto,
    *,
    num_visual_tokens: int,
) -> tuple[list[str], list[DeepstackInjection]]:
    """Rewrite deepstack residuals to exact zeros and prune their input branches."""
    if num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive")

    graph_inputs = {value.name: value for value in model.graph.input}
    if MAIN_EMBEDDING_INPUT not in graph_inputs:
        raise ValueError(f"ONNX graph has no {MAIN_EMBEDDING_INPUT!r} input")
    if VISUAL_MASK_INPUT not in graph_inputs:
        raise ValueError(f"ONNX graph has no {VISUAL_MASK_INPUT!r} input")

    deepstack_names = sorted(
        name for name in graph_inputs if name.startswith(DEEPSTACK_INPUT_PREFIX)
    )
    if not deepstack_names:
        raise ValueError("ONNX graph has no deepstack visual embedding inputs")

    mask_shape = _shape(graph_inputs[VISUAL_MASK_INPUT])
    if len(mask_shape) != 2:
        raise ValueError(
            f"{VISUAL_MASK_INPUT} must have rank 2, found shape {mask_shape}"
        )

    embedding_shape = _shape(graph_inputs[MAIN_EMBEDDING_INPUT])
    if len(embedding_shape) != 3:
        raise ValueError(
            f"{MAIN_EMBEDDING_INPUT} must have rank 3, found shape "
            f"{embedding_shape}"
        )
    hidden_size = embedding_shape[2]
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"Invalid embedding hidden size: {hidden_size!r}")

    for name in deepstack_names:
        deepstack_shape = _shape(graph_inputs[name])
        if len(deepstack_shape) != 2:
            raise ValueError(
                f"{name} must have rank 2, found shape {deepstack_shape}"
            )
        visual_tokens = deepstack_shape[0]
        if isinstance(visual_tokens, int) and visual_tokens != num_visual_tokens:
            raise ValueError(
                f"{name} has {visual_tokens!r} visual tokens; expected "
                f"{num_visual_tokens}"
            )
        if deepstack_shape[1] != hidden_size:
            raise ValueError(
                f"{name} hidden size {deepstack_shape[1]!r} does not match "
                f"{MAIN_EMBEDDING_INPUT} hidden size {hidden_size}"
            )

    producers, consumers = _graph_dependencies(model)
    value_info = {
        value.name: value
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    discovered: list[tuple[int, int, str, str]] = []
    for source_name in deepstack_names:
        first_adds = _first_add_consumers(model, consumers, source_name)
        if len(first_adds) != 1:
            labels = [
                _node_label(model.graph.node[node_index])
                for node_index, _ in first_adds
            ]
            raise ValueError(
                f"{source_name} must reach exactly one first Add injection; "
                f"found {labels}"
            )
        add_index, auxiliary_tensor = first_adds[0]
        add_node = model.graph.node[add_index]
        if len(add_node.input) != 2 or auxiliary_tensor not in add_node.input:
            raise ValueError(
                f"Injection {_node_label(add_node)!r} is not a binary Add "
                f"of {auxiliary_tensor!r}"
            )

        zero_index = producers.get(auxiliary_tensor)
        if zero_index is None:
            raise ValueError(
                f"Injection tensor {auxiliary_tensor!r} has no producer"
            )
        zero_node = model.graph.node[zero_index]
        if zero_node.op_type != "Mul" or list(zero_node.output) != [
            auxiliary_tensor
        ]:
            raise ValueError(
                f"Injection tensor {auxiliary_tensor!r} must come from one "
                f"Mul output, found {_node_label(zero_node)!r} "
                f"({zero_node.op_type})"
            )
        if consumers.get(auxiliary_tensor) != [add_index]:
            raise ValueError(
                f"Injection tensor {auxiliary_tensor!r} must be consumed only "
                f"by {_node_label(add_node)!r}"
            )

        auxiliary_info = value_info.get(auxiliary_tensor)
        if auxiliary_info is None:
            raise ValueError(
                f"Missing ONNX value_info for injection tensor "
                f"{auxiliary_tensor!r}"
            )
        auxiliary_shape = _shape(auxiliary_info)
        if auxiliary_shape != embedding_shape:
            raise ValueError(
                f"Injection tensor {auxiliary_tensor!r} has shape "
                f"{auxiliary_shape}; expected {embedding_shape}"
            )
        if len(add_node.output) != 1:
            raise ValueError(
                f"Injection {_node_label(add_node)!r} must have one output"
            )
        discovered.append(
            (zero_index, add_index, auxiliary_tensor, source_name)
        )

    zero_indices = {item[0] for item in discovered}
    add_indices = {item[1] for item in discovered}
    if len(zero_indices) != len(deepstack_names) or len(add_indices) != len(
        deepstack_names
    ):
        raise ValueError("Deepstack inputs do not map one-to-one to injections")

    mask_add_indices = {
        node_index
        for node_index, _ in _first_add_consumers(
            model, consumers, VISUAL_MASK_INPUT
        )
    }
    if mask_add_indices != add_indices:
        raise ValueError(
            f"{VISUAL_MASK_INPUT} reaches Add nodes "
            f"{sorted(mask_add_indices)}, expected {sorted(add_indices)}"
        )

    injections: list[DeepstackInjection] = []
    for zero_index, add_index, auxiliary_tensor, source_name in discovered:
        zero_node = model.graph.node[zero_index]
        add_node = model.graph.node[add_index]
        zero_node.op_type = "Sub"
        del zero_node.input[:]
        zero_node.input.extend([MAIN_EMBEDDING_INPUT, MAIN_EMBEDDING_INPUT])
        zero_node.ClearField("attribute")
        zero_node.ClearField("domain")
        injections.append(
            DeepstackInjection(
                source_input=source_name,
                zero_tensor=auxiliary_tensor,
                add_output=add_node.output[0],
                zero_node=_node_label(zero_node),
                add_node=_node_label(add_node),
            )
        )

    expected_removed = {VISUAL_MASK_INPUT, *deepstack_names}
    removed_inputs = _prune_unreachable_graph(model)
    if removed_inputs != expected_removed:
        raise ValueError(
            "Dead-code pruning removed unexpected graph inputs: "
            f"removed {sorted(removed_inputs)}; expected "
            f"{sorted(expected_removed)}"
        )

    return [VISUAL_MASK_INPUT, *deepstack_names], injections


def neutralize_deepstack_inputs(
    model: onnx.ModelProto,
    *,
    num_visual_tokens: int,
) -> list[str]:
    """Replace optional deepstack graph inputs with calibrated zero residuals."""
    removed, _ = _disable_deepstack_injections(
        model,
        num_visual_tokens=num_visual_tokens,
    )
    return removed


def _validate_preserved_activation_encodings(
    encodings_path: Path,
    injections: list[DeepstackInjection],
) -> None:
    try:
        encodings = json.loads(encodings_path.read_text(encoding="utf-8"))
        activation_encodings = encodings["activation_encodings"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid AIMET encodings file: {encodings_path}") from exc

    if isinstance(activation_encodings, list):
        names = {
            entry.get("name")
            for entry in activation_encodings
            if isinstance(entry, dict)
        }
    elif isinstance(activation_encodings, dict):
        names = set(activation_encodings)
    else:
        raise ValueError(
            f"Invalid activation_encodings in {encodings_path}: expected list "
            "or object"
        )

    required = {
        tensor_name
        for injection in injections
        for tensor_name in (injection.zero_tensor, injection.add_output)
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "AIMET encodings are missing rewritten injection tensors: "
            + ", ".join(missing)
        )


def create_compat_checkpoint(
    source: Path,
    destination: Path,
    *,
    num_visual_tokens: int,
    copy_function: Callable[[str, str], str] = _link_or_copy,
) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    source_model = source / MODEL_FILENAME
    if not source_model.is_file():
        raise FileNotFoundError(f"Missing source ONNX model: {source_model}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite: {destination}"
        )

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary destination already exists: {temporary}"
        )

    try:
        shutil.copytree(
            source,
            temporary,
            copy_function=copy_function,
            ignore=shutil.ignore_patterns(MODEL_FILENAME, MANIFEST_FILENAME),
        )

        model = onnx.load(source_model, load_external_data=False)
        removed_inputs, injections = _disable_deepstack_injections(
            model,
            num_visual_tokens=num_visual_tokens,
        )
        encodings_path = temporary / "model.encodings"
        if encodings_path.is_file():
            _validate_preserved_activation_encodings(
                encodings_path,
                injections,
            )

        output_model = temporary / MODEL_FILENAME
        onnx.save(model, output_model)
        onnx.checker.check_model(output_model)

        manifest = {
            "format_version": 1,
            "source_checkpoint": str(source),
            "source_model_sha256": sha256_file(source_model),
            "output_model_sha256": sha256_file(output_model),
            "removed_runtime_inputs": removed_inputs,
            "num_visual_tokens": num_visual_tokens,
            "behavior": {
                "text": "unchanged; optional deepstack inputs are zero in text mode",
                "vision": (
                    "primary image embeddings retained; auxiliary deepstack "
                    "visual injections disabled"
                ),
            },
            "target_runtime": "QAIRT 2.45 Genie 1.17 compatibility",
        }
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original W4A16 checkpoint")
    parser.add_argument(
        "destination",
        type=Path,
        help="New compatibility checkpoint (must not already exist)",
    )
    parser.add_argument(
        "--num-visual-tokens",
        type=int,
        default=256,
        help="Post-merge visual-token count (default: 256 for 512x512)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_compat_checkpoint(
        args.source,
        args.destination,
        num_visual_tokens=args.num_visual_tokens,
    )
    manifest = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    print(f"Created QAIRT 2.45 compatibility checkpoint: {output}")
    print(f"Removed runtime inputs: {manifest['removed_runtime_inputs']}")
    print(f"ONNX SHA-256: {manifest['output_model_sha256']}")


if __name__ == "__main__":
    main()
