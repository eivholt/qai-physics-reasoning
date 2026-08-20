#!/usr/bin/env python3
"""Prepare a self-converted Qwen3-VL QAIRT bundle for Qualcomm GenieX.

GenieX dispatches local QAIRT bundles by the ``model_id`` prefix in
``metadata.json``.  A custom Cosmos export therefore needs a ``qwen3_vl_``
prefix so GenieX selects its Qwen3-VL pipeline instead of rejecting the
otherwise valid bundle.

The source bundle is never modified.  The destination can use hard links for
large context binaries when both directories are on the same filesystem; the
metadata and provenance marker are always independent files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.vision_profile import VisionProfile
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from vision_profile import VisionProfile

METADATA_FILENAME = "metadata.json"
GENIE_CONFIG_FILENAME = "genie_config.json"
MARKER_FILENAME = "geniex_compat.json"
QAIRT245_COMPAT_MARKER = "qairt_245_compat.json"
W4_FP16_MARKER = "w4_fp16.json"
TEXT_W8_MARKER = "text_w8_matrices.json"
DEFAULT_MODEL_ID = "qwen3_vl_cosmos_reason2_2b"
PART1_CONTEXT = "part1_of_4.bin"
PART1_CONTEXT_REPORT = "part1_of_4.context.json"
VISION_CONTEXT = "vision_encoder.bin"
VISION_PROFILE_FILES = (
    "img-enc-htp.json",
)
VISION_SAMPLE_FILES = (
    "pixel_values.raw",
    "position_ids_cos.raw",
    "position_ids_sin.raw",
    "window_attention_mask.raw",
    "full_attention_mask.raw",
)
REQUIRED_CONTEXTS = {
    PART1_CONTEXT,
    "part2_of_4.bin",
    "part3_of_4.bin",
    "part4_of_4.bin",
    VISION_CONTEXT,
}
DEEPSTACK_INPUTS = {
    "visual_pos_masks",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
}
DEEPSTACK_INPUT_ORDER = (
    "visual_pos_masks",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
)
PART1_PRECISION_MARKERS = (
    W4_FP16_MARKER,
    TEXT_W8_MARKER,
)
EXPECTED_PART1_GRAPH_ORDER = (
    "ar128_cl512_1_of_4",
    "ar1_cl512_1_of_4",
)
QNN_EXTERNAL_DTYPES = {
    "float32": "QNN_DATATYPE_FLOAT_32",
    "bool": "QNN_DATATYPE_BOOL_8",
    "bool8": "QNN_DATATYPE_BOOL_8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _validate_bundle(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = sorted(name for name in REQUIRED_CONTEXTS if not (source / name).is_file())
    if missing:
        raise ValueError(f"Bundle is missing required contexts: {missing}")

    metadata = _load_json(source / METADATA_FILENAME, "metadata.json")
    config = _load_json(source / GENIE_CONFIG_FILENAME, "genie_config.json")

    try:
        rope_scaling = config["dialog"]["engine"]["model"]["positional-encoding"][
            "rope-scaling"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("genie_config.json has no positional-encoding rope-scaling") from exc

    if rope_scaling.get("rope-type") != "qwen3vl-mrope":
        raise ValueError(
            "GenieX Qwen3-VL dispatch requires rope-type 'qwen3vl-mrope'"
        )
    if rope_scaling.get("mrope-section") != [24, 20, 20]:
        raise ValueError(
            "Unexpected Qwen3-VL mrope-section; expected [24, 20, 20]"
        )

    vision = metadata.get("genie", {}).get("vision_preprocessing")
    if not isinstance(vision, dict):
        raise ValueError("metadata.json has no genie.vision_preprocessing object")

    return metadata, config


def _validate_part1_replacement(
    source: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    context = source / PART1_CONTEXT
    if not source.is_dir() or not context.is_file():
        raise ValueError(
            "Part-1 replacement bundle must contain "
            f"{PART1_CONTEXT}: {source}"
        )
    metadata = _load_json(source / METADATA_FILENAME, "replacement metadata.json")
    try:
        model_file = metadata["model_files"][PART1_CONTEXT]
        inputs = model_file["inputs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Replacement metadata has no part1_of_4.bin input contract"
        ) from exc
    if not isinstance(inputs, dict):
        raise ValueError("Replacement part-1 inputs must be an ordered object")

    missing = sorted(DEEPSTACK_INPUTS - set(inputs))
    if missing:
        raise ValueError(
            "Replacement part 1 is not a full DeepStack interface; missing "
            + ", ".join(missing)
        )

    input_names = list(inputs)
    try:
        embedding_index = input_names.index("inputs_embeds")
        auxiliary_indices = [input_names.index(name) for name in DEEPSTACK_INPUTS]
    except ValueError as exc:
        raise ValueError(
            "Replacement part-1 input order is missing inputs_embeds or "
            "a DeepStack auxiliary"
        ) from exc
    if any(index < embedding_index for index in auxiliary_indices):
        raise ValueError(
            "Replacement part-1 input order puts a DeepStack auxiliary before "
            "inputs_embeds; this is unsafe for GenieX graph-spec inference"
        )
    actual_deepstack_order = tuple(
        name for name in inputs if name in DEEPSTACK_INPUTS
    )
    if actual_deepstack_order != DEEPSTACK_INPUT_ORDER:
        raise ValueError(
            "Replacement part-1 DeepStack inputs are not in the required "
            f"order {list(DEEPSTACK_INPUT_ORDER)}: "
            f"{list(actual_deepstack_order)}"
        )

    mask = inputs["visual_pos_masks"]
    if (
        not isinstance(mask, dict)
        or mask.get("shape") not in ([1, 1], [1, 128])
    ):
        raise ValueError(
            "Replacement visual_pos_masks shape must be [1, 1] or [1, 128]"
        )
    if mask.get("dtype") not in {"bool", "bool8"}:
        raise ValueError(
            "Replacement visual_pos_masks external dtype must be bool/bool8"
        )
    deepstack_shape: list[int] | None = None
    for name in sorted(DEEPSTACK_INPUTS - {"visual_pos_masks"}):
        tensor = inputs[name]
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
            or shape[1] != 2048
        ):
            raise ValueError(
                f"Replacement {name} shape is {shape!r}; expected "
                "[positive visual capacity, 2048]"
            )
        if tensor.get("dtype") != "float32":
            raise ValueError(
                f"Replacement {name} external dtype must be float32"
            )
        if deepstack_shape is None:
            deepstack_shape = shape
        elif shape != deepstack_shape:
            raise ValueError(
                "Replacement DeepStack input shapes must match; "
                f"{name} is {shape}, expected {deepstack_shape}"
            )

    unexpected_deepstack = sorted(
        name
        for name in inputs
        if name.startswith("deepstack_visual_embeds_")
        and name not in DEEPSTACK_INPUTS
    )
    if unexpected_deepstack:
        raise ValueError(
            "Replacement part 1 has unsupported DeepStack inputs: "
            + ", ".join(unexpected_deepstack)
        )

    replacement_marker = source / W4_FP16_MARKER
    if not replacement_marker.is_file():
        raise ValueError(
            f"Part-1 replacement bundle must contain {W4_FP16_MARKER}"
        )
    supplementary = metadata.get("supplementary_files")
    if (
        not isinstance(supplementary, dict)
        or W4_FP16_MARKER not in supplementary
    ):
        raise ValueError(
            f"Replacement {W4_FP16_MARKER} has no supplementary_files "
            "metadata entry"
        )
    precision_markers = [W4_FP16_MARKER]
    text_w8_path = source / TEXT_W8_MARKER
    text_w8_declared = (
        isinstance(supplementary, dict)
        and TEXT_W8_MARKER in supplementary
    )
    if text_w8_path.is_file() != text_w8_declared:
        raise ValueError(
            f"Replacement {TEXT_W8_MARKER} file and supplementary_files "
            "metadata entry must either both be present or both be absent"
        )
    if text_w8_path.is_file():
        text_w8 = _load_json(text_w8_path, TEXT_W8_MARKER)
        selected_parts = text_w8.get("selected_parts")
        if (
            not isinstance(selected_parts, list)
            or PART1_CONTEXT.removesuffix(".bin") not in selected_parts
        ):
            raise ValueError(
                f"Replacement {TEXT_W8_MARKER} does not prove that "
                "part1_of_4 was promoted to W8"
            )
        export_contract = text_w8.get("export_contract")
        if not isinstance(export_contract, dict):
            raise ValueError(
                f"Replacement {TEXT_W8_MARKER} has no export_contract"
            )
        context_length = export_contract.get("context_length")
        if not isinstance(context_length, int) or context_length <= 0:
            raise ValueError(
                f"Replacement {TEXT_W8_MARKER} export_contract.context_length "
                f"must be a positive integer; got {context_length!r}"
            )
        expected_contract = {
            "activation_precision": "FP16",
            "sequence_lengths": [128, 1],
            "deepstack_inputs": list(DEEPSTACK_INPUT_ORDER),
            "external_tensor_precision": (
                "FP32, except visual_pos_masks BOOL"
            ),
        }
        for field, expected in expected_contract.items():
            if export_contract.get(field) != expected:
                raise ValueError(
                    f"Replacement {TEXT_W8_MARKER} export_contract.{field} "
                    f"is {export_contract.get(field)!r}; expected "
                    f"{expected!r}"
                )
        precision_markers.append(TEXT_W8_MARKER)
    return metadata, tuple(precision_markers)


def _validate_vision_replacement(
    source: Path,
) -> tuple[dict[str, Any], VisionProfile]:
    context = source / VISION_CONTEXT
    if not source.is_dir() or not context.is_file():
        raise ValueError(
            "Vision replacement bundle must contain "
            f"{VISION_CONTEXT}: {source}"
        )
    metadata = _load_json(
        source / METADATA_FILENAME, "vision replacement metadata.json"
    )
    profile = VisionProfile.from_metadata(metadata)
    profile.validate_img_encoder_config(source / "img-enc-htp.json")
    profile.validate_ancillary_files(source / "sample_inputs")
    pixel_values = source / "sample_inputs" / "pixel_values.raw"
    if not pixel_values.is_file():
        raise ValueError(
            f"Vision replacement is missing sample tensor: {pixel_values}"
        )
    if pixel_values.stat().st_size != profile.pixel_bytes:
        raise ValueError(
            f"Vision replacement pixel_values.raw has "
            f"{pixel_values.stat().st_size} bytes; expected "
            f"{profile.pixel_bytes}"
        )
    try:
        outputs = metadata["model_files"][VISION_CONTEXT]["outputs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Vision replacement metadata has no vision output contract"
        ) from exc
    for name in (
        "image_features",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2",
    ):
        tensor = outputs.get(name) if isinstance(outputs, dict) else None
        if not isinstance(tensor, dict) or tensor.get("shape") != list(
            profile.visual_shape
        ):
            raise ValueError(
                f"Vision replacement output {name} must have shape "
                f"{list(profile.visual_shape)}"
            )
    return metadata, profile


def _validate_vision_text_capacity(
    text_metadata: dict[str, Any],
    profile: VisionProfile,
) -> None:
    """Ensure the retained first shard can consume one visual prefill slice."""

    try:
        part1_inputs = text_metadata["model_files"][PART1_CONTEXT][
            "inputs"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Text bundle metadata has no part-1 input contract"
        ) from exc
    for name in sorted(DEEPSTACK_INPUTS - {"visual_pos_masks"}):
        tensor = part1_inputs.get(name) if isinstance(part1_inputs, dict) else None
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[1] != profile.hidden_size
            or shape[0] <= 0
        ):
            raise ValueError(
                f"Retained part 1 cannot accept {name} from the replacement "
                f"vision graph: shape {shape!r}"
            )
    mask = (
        part1_inputs.get("visual_pos_masks")
        if isinstance(part1_inputs, dict)
        else None
    )
    if not isinstance(mask, dict):
        raise ValueError(
            "Retained part 1 has no visual_pos_masks DeepStack input"
        )


def _part1_sequence_length(model_file: dict[str, Any], label: str) -> int:
    try:
        tensor = model_file["inputs"]["inputs_embeds"]
        shape = tensor["shape"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{label} has no inputs_embeds shape") from exc
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != 1
        or shape[2] != 2048
        or shape[1] not in (1, 128)
    ):
        raise ValueError(
            f"{label} inputs_embeds shape is {shape!r}; expected "
            "[1, 1|128, 2048]"
        )
    return int(shape[1])


def _metadata_tensor_contract(
    tensors: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(tensors, dict):
        raise ValueError(f"{label} must be an ordered object")
    contract: list[dict[str, Any]] = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise ValueError(f"{label}.{name} must be an object")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if (
            not isinstance(shape, list)
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in shape
            )
        ):
            raise ValueError(f"{label}.{name} has invalid shape {shape!r}")
        qnn_dtype = QNN_EXTERNAL_DTYPES.get(dtype)
        if qnn_dtype is None:
            raise ValueError(
                f"{label}.{name} has unsupported external dtype {dtype!r}"
            )
        contract.append(
            {
                "name": name,
                "dataType": qnn_dtype,
                "dimensions": shape,
            }
        )
    return contract


def _qnn_graph_tensor_contract(
    graph_info: dict[str, Any],
    *,
    field: str,
    label: str,
) -> list[dict[str, Any]]:
    tensors = graph_info.get(field)
    if not isinstance(tensors, list):
        raise ValueError(f"{label}.{field} must be a list")
    contract: list[dict[str, Any]] = []
    for index, tensor_wrapper in enumerate(tensors):
        tensor = (
            tensor_wrapper.get("info")
            if isinstance(tensor_wrapper, dict)
            else None
        )
        if not isinstance(tensor, dict):
            raise ValueError(f"{label}.{field}[{index}] has no info object")
        contract.append(
            {
                "name": tensor.get("name"),
                "dataType": tensor.get("dataType"),
                "dimensions": tensor.get("dimensions"),
            }
        )
    return contract


def _validate_qnn_graph_against_metadata(
    graph_info: dict[str, Any],
    model_file: dict[str, Any],
    *,
    label: str,
) -> None:
    for qnn_field, metadata_field in (
        ("graphInputs", "inputs"),
        ("graphOutputs", "outputs"),
    ):
        actual = _qnn_graph_tensor_contract(
            graph_info,
            field=qnn_field,
            label=label,
        )
        expected = _metadata_tensor_contract(
            model_file.get(metadata_field),
            label=f"{label}.{metadata_field}",
        )
        if actual != expected:
            raise ValueError(
                f"{label} {metadata_field} do not exactly match the "
                "metadata name/order/dtype/shape contract"
            )


def _load_part1_context_report(
    replacement_source: Path,
    report_path: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    selected = (
        report_path.expanduser().resolve()
        if report_path is not None
        else replacement_source / PART1_CONTEXT_REPORT
    )
    if not selected.is_file():
        return None, None
    envelope = _load_json(selected, "part-1 QNN context report")
    if envelope.get("schema_version") != 1:
        raise ValueError("Part-1 QNN context report schema_version must be 1")
    if envelope.get("context_filename") != PART1_CONTEXT:
        raise ValueError(
            f"Part-1 QNN context report must name {PART1_CONTEXT}"
        )
    actual_context_hash = sha256_file(replacement_source / PART1_CONTEXT)
    if envelope.get("context_sha256") != actual_context_hash:
        raise ValueError(
            "Part-1 QNN context report is not bound to the replacement "
            "context SHA-256"
        )
    qnn_info = envelope.get("qnn_context_binary_info")
    if not isinstance(qnn_info, dict):
        raise ValueError(
            "Part-1 QNN context report has no qnn_context_binary_info"
        )
    try:
        artifact_type = qnn_info["header"]["artifact_type"]
        info = qnn_info["info"]
        build_id = info["buildId"]
        graphs = info["graphs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Part-1 QNN context report is incomplete"
        ) from exc
    if artifact_type != "CONTEXT_BINARY_INFO":
        raise ValueError(
            "Part-1 QNN context report is not context-binary information"
        )
    if not isinstance(build_id, str) or not build_id.startswith("v2.45.0."):
        raise ValueError(
            f"Part-1 QNN context was not inspected with QAIRT 2.45: "
            f"{build_id!r}"
        )
    if not isinstance(graphs, list) or len(graphs) != 2:
        raise ValueError(
            "Replacement part 1 must contain exactly two linked graphs"
        )
    graph_infos: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        graph_info = graph.get("info") if isinstance(graph, dict) else None
        if not isinstance(graph_info, dict):
            raise ValueError(
                f"Part-1 QNN context graph {index} has no info object"
            )
        graph_infos.append(graph_info)
    graph_order = tuple(graph.get("graphName") for graph in graph_infos)
    if graph_order != EXPECTED_PART1_GRAPH_ORDER:
        raise ValueError(
            "Replacement part-1 linked graph order is "
            f"{list(graph_order)}; expected "
            f"{list(EXPECTED_PART1_GRAPH_ORDER)}"
        )
    return {
        "envelope": envelope,
        "build_id": build_id,
        "graph_infos": graph_infos,
        "graph_order": list(graph_order),
        "report_sha256": sha256_file(selected),
        "context_sha256": actual_context_hash,
    }, selected


def _validate_part1_compatibility(
    source_metadata: dict[str, Any],
    replacement_metadata: dict[str, Any],
    vision_metadata: dict[str, Any] | None = None,
    context_report: dict[str, Any] | None = None,
) -> bool:
    """Fail before copying if the partial export cannot join the base bundle."""

    try:
        source_part1 = source_metadata["model_files"][PART1_CONTEXT]
        replacement_part1 = replacement_metadata["model_files"][PART1_CONTEXT]
        source_inputs = source_part1["inputs"]
        source_outputs = source_part1["outputs"]
        replacement_inputs = replacement_part1["inputs"]
        replacement_outputs = replacement_part1["outputs"]
        selected_vision_metadata = (
            vision_metadata
            if vision_metadata is not None
            else source_metadata
        )
        vision_outputs = selected_vision_metadata["model_files"][
            VISION_CONTEXT
        ]["outputs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Source/replacement metadata lacks a complete part-1 or vision contract"
        ) from exc
    if not all(
        isinstance(contract, dict)
        for contract in (
            source_inputs,
            source_outputs,
            replacement_inputs,
            replacement_outputs,
            vision_outputs,
        )
    ):
        raise ValueError(
            "Source/replacement part-1 and vision contracts must be objects"
        )

    source_sequence = _part1_sequence_length(
        source_part1, "Source part 1"
    )
    replacement_sequence = _part1_sequence_length(
        replacement_part1, "Replacement part 1"
    )
    retain_source_metadata = False
    if replacement_sequence == source_sequence:
        source_base_inputs = [
            (name, tensor)
            for name, tensor in source_inputs.items()
            if name not in DEEPSTACK_INPUTS
        ]
        replacement_base_inputs = [
            (name, tensor)
            for name, tensor in replacement_inputs.items()
            if name not in DEEPSTACK_INPUTS
        ]
        if source_base_inputs != replacement_base_inputs:
            raise ValueError(
                "Replacement part-1 base input contract does not exactly "
                "match the source bundle"
            )
        if list(source_outputs.items()) != list(replacement_outputs.items()):
            raise ValueError(
                "Replacement part-1 output contract does not exactly match "
                "the source bundle"
            )
    elif source_sequence == 1 and replacement_sequence == 128:
        if context_report is None:
            raise ValueError(
                "Replacement metadata describes AR128 while the source "
                "bundle describes AR1; provide a SHA-bound QAIRT 2.45 "
                f"{PART1_CONTEXT_REPORT} proving both linked graph contracts "
                "and AR128-before-AR1 order"
            )
        graph_infos = context_report["graph_infos"]
        _validate_qnn_graph_against_metadata(
            graph_infos[0],
            replacement_part1,
            label=EXPECTED_PART1_GRAPH_ORDER[0],
        )
        _validate_qnn_graph_against_metadata(
            graph_infos[1],
            source_part1,
            label=EXPECTED_PART1_GRAPH_ORDER[1],
        )
        source_input_names = list(source_inputs)
        replacement_input_names = list(replacement_inputs)
        if source_input_names != replacement_input_names:
            raise ValueError(
                "AR128 and AR1 part-1 input names/order differ"
            )
        if list(source_outputs) != list(replacement_outputs):
            raise ValueError(
                "AR128 and AR1 part-1 output names/order differ"
            )
        for name in source_input_names:
            if source_inputs[name].get("dtype") != replacement_inputs[
                name
            ].get("dtype"):
                raise ValueError(
                    f"AR128 and AR1 external input dtype differs for {name}"
                )
        for name in source_outputs:
            if source_outputs[name].get("dtype") != replacement_outputs[
                name
            ].get("dtype"):
                raise ValueError(
                    f"AR128 and AR1 external output dtype differs for {name}"
                )
        retain_source_metadata = True
    else:
        raise ValueError(
            "Unsupported source/replacement part-1 metadata graph views: "
            f"AR{source_sequence} and AR{replacement_sequence}"
        )

    source_deepstack_names = [
        name for name in source_inputs if name in DEEPSTACK_INPUTS
    ]
    if source_deepstack_names and set(source_deepstack_names) != DEEPSTACK_INPUTS:
        raise ValueError(
            "Source part 1 has an incomplete DeepStack input contract: "
            f"{source_deepstack_names}"
        )
    if source_deepstack_names:
        if source_deepstack_names != list(DEEPSTACK_INPUT_ORDER):
            raise ValueError(
                "Source part-1 DeepStack inputs are not in the required "
                f"order {list(DEEPSTACK_INPUT_ORDER)}: "
                f"{source_deepstack_names}"
            )
        for name in DEEPSTACK_INPUT_ORDER:
            source_tensor = source_inputs[name]
            replacement_tensor = replacement_inputs[name]
            if name == "visual_pos_masks" and retain_source_metadata:
                compatible = (
                    source_tensor.get("dtype")
                    == replacement_tensor.get("dtype")
                    and source_tensor.get("shape") == [1, 1]
                    and replacement_tensor.get("shape") == [1, 128]
                )
            else:
                compatible = source_tensor == replacement_tensor
            if not compatible:
                raise ValueError(
                    "Replacement part-1 DeepStack capacity/external contract "
                    f"for {name} does not exactly match the source bundle"
                )

    part1_visual_shape = replacement_inputs[
        "deepstack_visual_embeds_0"
    ].get("shape")
    for name in (
        "image_features",
        *sorted(DEEPSTACK_INPUTS - {"visual_pos_masks"}),
    ):
        tensor = vision_outputs.get(name)
        if not isinstance(tensor, dict):
            raise ValueError(
                f"Source vision context cannot supply replacement part 1: "
                f"missing output {name}"
            )
        vision_shape = tensor.get("shape")
        if (
            not isinstance(vision_shape, list)
            or len(vision_shape) != 2
            or not isinstance(part1_visual_shape, list)
            or len(part1_visual_shape) != 2
            or vision_shape[1] != part1_visual_shape[1]
            or vision_shape[0] <= 0
            or part1_visual_shape[0] <= 0
        ):
            raise ValueError(
                f"Source vision output {name} shape {tensor.get('shape')!r} "
                "is incompatible with replacement DeepStack capacity "
                f"{part1_visual_shape!r}"
            )
        if vision_shape[0] > part1_visual_shape[0]:
            raise ValueError(
                f"Source vision output {name} has {vision_shape[0]} rows, "
                "which exceeds replacement part-1 DeepStack capacity "
                f"{part1_visual_shape[0]}"
            )
    return retain_source_metadata


def _replace_file_atomically(
    source: Path,
    destination: Path,
    *,
    copy_function: Callable[[str, str], str],
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ValueError(f"Temporary replacement already exists: {temporary}")
    copy_function(str(source), str(temporary))
    temporary.replace(destination)


def prepare_bundle(
    source: Path,
    destination: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    hardlink: bool = False,
    part1_replacement_bundle: Path | None = None,
    part1_context_report: Path | None = None,
    vision_replacement_bundle: Path | None = None,
) -> Path:
    """Copy a bundle and select GenieX's Qwen3-VL runtime dispatcher."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source bundle is not a directory: {source}")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if not model_id.startswith("qwen3_vl_"):
        raise ValueError("GenieX model_id must start with 'qwen3_vl_'")

    metadata, _ = _validate_bundle(source)
    replacement_source: Path | None = None
    replacement_metadata: dict[str, Any] | None = None
    replacement_precision_markers: tuple[str, ...] = ()
    replacement_context_report: dict[str, Any] | None = None
    replacement_context_report_path: Path | None = None
    retain_source_part1_metadata = False
    vision_source: Path | None = None
    vision_metadata: dict[str, Any] | None = None
    vision_profile: VisionProfile | None = None
    if vision_replacement_bundle is not None:
        vision_source = vision_replacement_bundle.expanduser().resolve()
        vision_metadata, vision_profile = _validate_vision_replacement(
            vision_source
        )
    if part1_replacement_bundle is not None:
        replacement_source = part1_replacement_bundle.expanduser().resolve()
        (
            replacement_metadata,
            replacement_precision_markers,
        ) = _validate_part1_replacement(replacement_source)
        (
            replacement_context_report,
            replacement_context_report_path,
        ) = _load_part1_context_report(
            replacement_source,
            part1_context_report,
        )
        retain_source_part1_metadata = _validate_part1_compatibility(
            metadata,
            replacement_metadata,
            vision_metadata=vision_metadata,
            context_report=replacement_context_report,
        )
    elif part1_context_report is not None:
        raise ValueError(
            "--part1-context-report requires --part1-replacement-bundle"
        )
    if vision_profile is not None:
        _validate_vision_text_capacity(
            replacement_metadata
            if replacement_metadata is not None
            else metadata,
            vision_profile,
        )
    original_model_id = metadata.get("model_id")
    if not isinstance(original_model_id, str) or not original_model_id:
        raise ValueError("metadata.json has no non-empty model_id")

    copy_function: Callable[[str, str], str] = (
        _hardlink_or_copy if hardlink else shutil.copy2
    )
    shutil.copytree(source, destination, copy_function=copy_function)

    try:
        destination_metadata = destination / METADATA_FILENAME
        # copytree may have hard-linked metadata. Replace it atomically so the
        # source file remains byte-for-byte untouched.
        patched_metadata = dict(metadata)
        patched_metadata["model_id"] = model_id
        if replacement_source is not None or vision_source is not None:
            source_model_files = metadata.get("model_files")
            if not isinstance(source_model_files, dict):
                raise ValueError(
                    "Source metadata has no model_files object for replacement"
                )
            patched_metadata["model_files"] = dict(source_model_files)

        if (
            vision_source is not None
            and vision_metadata is not None
            and vision_profile is not None
        ):
            patched_metadata["model_files"][VISION_CONTEXT] = (
                vision_metadata["model_files"][VISION_CONTEXT]
            )
            _replace_file_atomically(
                vision_source / VISION_CONTEXT,
                destination / VISION_CONTEXT,
                copy_function=copy_function,
            )
            for filename in VISION_PROFILE_FILES:
                _replace_file_atomically(
                    vision_source / filename,
                    destination / filename,
                    copy_function=shutil.copy2,
                )
            destination_samples = destination / "sample_inputs"
            destination_samples.mkdir(exist_ok=True)
            for filename in VISION_SAMPLE_FILES:
                _replace_file_atomically(
                    vision_source / "sample_inputs" / filename,
                    destination_samples / filename,
                    copy_function=shutil.copy2,
                )
            # Prepared video inputs are bound to the old graph shapes/hashes.
            # Never carry them into a bundle with a different vision context.
            stale_video_inputs = destination / "video_inputs"
            if stale_video_inputs.is_dir():
                shutil.rmtree(stale_video_inputs)

            source_genie = metadata.get("genie")
            replacement_genie = vision_metadata.get("genie")
            if not isinstance(source_genie, dict) or not isinstance(
                replacement_genie, dict
            ):
                raise ValueError(
                    "Source/replacement metadata has no genie object"
                )
            replacement_preprocessing = replacement_genie.get(
                "vision_preprocessing"
            )
            if not isinstance(replacement_preprocessing, dict):
                raise ValueError(
                    "Vision replacement has no preprocessing metadata"
                )
            patched_metadata["genie"] = dict(source_genie)
            patched_metadata["genie"]["vision_preprocessing"] = (
                replacement_preprocessing
            )
            replacement_supplementary = vision_metadata.get(
                "supplementary_files"
            )
            if isinstance(replacement_supplementary, dict):
                supplementary = patched_metadata.get("supplementary_files")
                patched_metadata["supplementary_files"] = (
                    dict(supplementary)
                    if isinstance(supplementary, dict)
                    else {}
                )
                for filename in VISION_PROFILE_FILES:
                    if filename in replacement_supplementary:
                        patched_metadata["supplementary_files"][filename] = (
                            replacement_supplementary[filename]
                        )

        if replacement_source is not None and replacement_metadata is not None:
            replacement_model_file = (
                metadata["model_files"][PART1_CONTEXT]
                if retain_source_part1_metadata
                else replacement_metadata["model_files"][PART1_CONTEXT]
            )
            patched_metadata["model_files"][PART1_CONTEXT] = replacement_model_file

            _replace_file_atomically(
                replacement_source / PART1_CONTEXT,
                destination / PART1_CONTEXT,
                copy_function=copy_function,
            )
            replacement_supplementary = replacement_metadata[
                "supplementary_files"
            ]
            supplementary = patched_metadata.get("supplementary_files")
            patched_metadata["supplementary_files"] = (
                dict(supplementary)
                if isinstance(supplementary, dict)
                else {}
            )
            for filename in PART1_PRECISION_MARKERS:
                (destination / filename).unlink(missing_ok=True)
                patched_metadata["supplementary_files"].pop(filename, None)
            for filename in replacement_precision_markers:
                _replace_file_atomically(
                    replacement_source / filename,
                    destination / filename,
                    copy_function=shutil.copy2,
                )
                patched_metadata["supplementary_files"][filename] = (
                    replacement_supplementary[filename]
                )
            (destination / PART1_CONTEXT_REPORT).unlink(missing_ok=True)
            patched_metadata["supplementary_files"].pop(
                PART1_CONTEXT_REPORT, None
            )
            if replacement_context_report_path is not None:
                _replace_file_atomically(
                    replacement_context_report_path,
                    destination / PART1_CONTEXT_REPORT,
                    copy_function=shutil.copy2,
                )
                patched_metadata["supplementary_files"][
                    PART1_CONTEXT_REPORT
                ] = (
                    "SHA-bound QAIRT 2.45 linked graph-order and external "
                    "tensor contract report for part1_of_4.bin."
                )
            compatibility_marker = destination / QAIRT245_COMPAT_MARKER
            compatibility_marker.unlink(missing_ok=True)
            supplementary = patched_metadata.get("supplementary_files")
            if isinstance(supplementary, dict):
                patched_metadata["supplementary_files"] = dict(supplementary)
                patched_metadata["supplementary_files"].pop(
                    QAIRT245_COMPAT_MARKER, None
                )

        temporary_metadata = destination / f".{METADATA_FILENAME}.tmp"
        temporary_metadata.write_text(
            json.dumps(patched_metadata, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        temporary_metadata.replace(destination_metadata)

        inherited_marker_path = source / MARKER_FILENAME
        inherited_marker = (
            _load_json(inherited_marker_path, "source geniex_compat.json")
            if inherited_marker_path.is_file()
            else None
        )
        marker = {
            "schema_version": 3,
            "purpose": "Select Qualcomm GenieX's Qwen3-VL QAIRT pipeline",
            "source_bundle_name": source.name,
            "source_metadata_sha256": sha256_file(source / METADATA_FILENAME),
            "source_compatibility_marker": (
                {
                    "sha256": sha256_file(inherited_marker_path),
                    "schema_version": inherited_marker.get("schema_version"),
                    "part1_replacement": inherited_marker.get(
                        "part1_replacement"
                    ),
                    "vision_replacement": inherited_marker.get(
                        "vision_replacement"
                    ),
                }
                if inherited_marker is not None
                else None
            ),
            "original_model_id": original_model_id,
            "geniex_model_id": model_id,
            "final_metadata_sha256": sha256_file(destination_metadata),
            "final_context_sha256": {
                filename: sha256_file(destination / filename)
                for filename in sorted(REQUIRED_CONTEXTS)
            },
            "qwen3_vl_contract": {
                "rope_type": "qwen3vl-mrope",
                "mrope_section": [24, 20, 20],
                "mrope_interleaving": "stride",
            },
            "hardlink_requested": hardlink,
            "part1_replacement": (
                {
                    "source_bundle_name": replacement_source.name,
                    "context_sha256": sha256_file(
                        replacement_source / PART1_CONTEXT
                    ),
                    "metadata_sha256": sha256_file(
                        replacement_source / METADATA_FILENAME
                    ),
                    "deepstack_inputs": sorted(DEEPSTACK_INPUTS),
                    "deepstack_visual_shape": replacement_metadata[
                        "model_files"
                    ][PART1_CONTEXT]["inputs"][
                        "deepstack_visual_embeds_0"
                    ]["shape"],
                    "visual_pos_masks_shape": replacement_metadata[
                        "model_files"
                    ][PART1_CONTEXT]["inputs"]["visual_pos_masks"]["shape"],
                    "precision_provenance": {
                        filename: {
                            "sha256": sha256_file(
                                replacement_source / filename
                            ),
                            "metadata_description": replacement_metadata[
                                "supplementary_files"
                            ][filename],
                        }
                        for filename in replacement_precision_markers
                    },
                    "context_report": (
                        {
                            "sha256": replacement_context_report[
                                "report_sha256"
                            ],
                            "context_sha256": replacement_context_report[
                                "context_sha256"
                            ],
                            "qairt_build_id": replacement_context_report[
                                "build_id"
                            ],
                            "graph_order": replacement_context_report[
                                "graph_order"
                            ],
                            "final_metadata_graph_view": (
                                "source_ar1"
                                if retain_source_part1_metadata
                                else "replacement"
                            ),
                        }
                        if replacement_context_report is not None
                        else None
                    ),
                    "removed_compatibility_marker": QAIRT245_COMPAT_MARKER,
                }
                if replacement_source is not None
                else None
            ),
            "vision_replacement": (
                {
                    "source_bundle_name": vision_source.name,
                    "context_sha256": sha256_file(
                        vision_source / VISION_CONTEXT
                    ),
                    "metadata_sha256": sha256_file(
                        vision_source / METADATA_FILENAME
                    ),
                    "image_height": vision_profile.image_height,
                    "image_width": vision_profile.image_width,
                    "grid_thw": list(vision_profile.grid_thw),
                    "pixel_values_shape": list(vision_profile.pixel_shape),
                    "visual_tokens": vision_profile.visual_tokens,
                    "reused_text_contexts": [
                        PART1_CONTEXT,
                        "part2_of_4.bin",
                        "part3_of_4.bin",
                        "part4_of_4.bin",
                    ],
                }
                if vision_source is not None and vision_profile is not None
                else None
            ),
        }
        # ``copytree(..., copy_function=_hardlink_or_copy)`` may have linked an
        # existing compatibility marker from an already prepared source
        # bundle.  Replace through a new inode so rewriting provenance never
        # mutates the source marker through that hard link.
        temporary_marker = destination / f".{MARKER_FILENAME}.tmp"
        temporary_marker.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(destination / MARKER_FILENAME)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Hard-link large files when possible (same-filesystem diagnostics)",
    )
    parser.add_argument(
        "--part1-replacement-bundle",
        type=Path,
        default=None,
        help=(
            "Partial full-interface W4/FP export containing a replacement "
            "part1_of_4.bin and metadata.json"
        ),
    )
    parser.add_argument(
        "--part1-context-report",
        type=Path,
        default=None,
        help=(
            "SHA-bound QAIRT 2.45 qnn-context-binary-utility report. "
            "Required when replacement metadata presents the AR128 graph "
            "while the source bundle presents AR1."
        ),
    )
    parser.add_argument(
        "--vision-replacement-bundle",
        type=Path,
        default=None,
        help=(
            "Partial export containing a replacement vision_encoder.bin, "
            "profile metadata/config, and sample vision tensors. Text "
            "contexts are retained from --source."
        ),
    )
    args = parser.parse_args()

    output = prepare_bundle(
        args.source,
        args.destination,
        model_id=args.model_id,
        hardlink=args.hardlink,
        part1_replacement_bundle=args.part1_replacement_bundle,
        part1_context_report=args.part1_context_report,
        vision_replacement_bundle=args.vision_replacement_bundle,
    )
    print(f"Prepared GenieX bundle: {output}")


if __name__ == "__main__":
    main()
