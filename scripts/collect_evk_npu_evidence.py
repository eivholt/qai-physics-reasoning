#!/usr/bin/env python3
"""Validate and summarize evidence from a Cosmos IQ9075 QAIRT run.

This script is intentionally read-mostly and uses only the Python standard
library. Run it on the EVK after the text and/or vision smoke commands. It
does not launch inference or hash the multi-gigabyte context binaries.

Examples:

    python3 collect_evk_npu_evidence.py \
      --bundle /home/ubuntu/cosmos_reason2_2b_qairt \
      --mode preflight

    python3 collect_evk_npu_evidence.py \
      --bundle /home/ubuntu/cosmos_reason2_2b_qairt \
      --mode all \
      --text-log text_smoke.log \
      --text-profile text_smoke_profile.txt \
      --vision-log vision_smoke_245.log

The default vision-output expression is specific to the QAI Hub Models
``qwen3_vl_4b_instruct/2/dog.jpg`` sample used by this adapter. Override it
when running a different image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TEXT_BINS = [
    "part1_of_4.bin",
    "part2_of_4.bin",
    "part3_of_4.bin",
    "part4_of_4.bin",
]
EXPECTED_VISION_BINS = ["vision_encoder.bin"]
EXPECTED_BINS = EXPECTED_TEXT_BINS + EXPECTED_VISION_BINS
MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
MAX_ANSWER_CHARS = 4000
QAIRT245_COMPAT_MARKER = "qairt_245_compat.json"
DISABLED_DEEPSTACK_INPUTS = {
    "visual_pos_masks",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
}

REQUIRED_SAMPLE_INPUTS = [
    "pixel_values.raw",
    "position_ids_cos.raw",
    "position_ids_sin.raw",
    "window_attention_mask.raw",
    "full_attention_mask.raw",
    "prompt_prefix.txt",
    "prompt_suffix.txt",
]

REQUIRED_PIPELINE_LINES = [
    "node config create imageEncoderConfig img-enc-htp.json",
    "node config create lutEncoderConfig text-encoder.json",
    "node config create textGeneratorConfig text-generator.json",
    (
        "pipeline connect GeniePipeline imageEncoder "
        "GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT textGenerator "
        "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT"
    ),
    (
        "pipeline connect GeniePipeline lutEncoder "
        "GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT textGenerator "
        "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT"
    ),
    (
        "node set textFile lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "
        "sample_inputs/prompt_prefix.txt"
    ),
    (
        "node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT "
        "sample_inputs/pixel_values.raw"
    ),
    (
        "node set embedding imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS "
        "sample_inputs/position_ids_cos.raw"
    ),
    (
        "node set embedding imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN "
        "sample_inputs/position_ids_sin.raw"
    ),
    (
        "node set embedding imageEncoder "
        "GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK "
        "sample_inputs/window_attention_mask.raw"
    ),
    (
        "node set embedding imageEncoder "
        "GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK "
        "sample_inputs/full_attention_mask.raw"
    ),
    (
        "node set textFile lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "
        "sample_inputs/prompt_suffix.txt"
    ),
    "pipeline execute GeniePipeline",
]
DEEPSTACK_WILDCARD_PIPELINE_LINE = (
    "pipeline connect GeniePipeline imageEncoder GENIE_NODE_WILDCARD "
    "textGenerator GENIE_NODE_WILDCARD"
)

FATAL_LOG_PATTERNS = [
    re.compile(r"^\s*(?:\[[^\]]+\]\s*)?(?:fatal|error)(?:\W|$)", re.IGNORECASE),
    re.compile(r"segmentation fault|core dumped|^\s*traceback", re.IGNORECASE),
    re.compile(
        r"failed to (?:create|load|initialize|open)|"
        r"(?:device|context|graph) creation failed",
        re.IGNORECASE,
    ),
    re.compile(r"\bQNN_[A-Z0-9_]*ERROR\b"),
]

QAIRT_VERSION_PATTERN = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)"
    r"(?:\.(?P<patch>\d+))?"
    r"(?:\.(?P<ident>\d+_?\d+))?"
    r"(?:-(?P<flavor>.+))?$"
)
CONTEXT_GRAPH_PATTERN = re.compile(
    r"^ar(?P<sequence>\d+)_cl(?P<context>\d+)_"
    r"(?P<part>\d+)_of_4$"
)


def new_section() -> dict[str, Any]:
    return {
        "status": "pass",
        "checks": [],
        "warnings": [],
        "errors": [],
        "evidence": {},
    }


def add_check(section: dict[str, Any], name: str, detail: str) -> None:
    section["checks"].append({"name": name, "detail": detail})


def add_warning(section: dict[str, Any], message: str) -> None:
    section["warnings"].append(message)


def add_error(section: dict[str, Any], message: str) -> None:
    section["errors"].append(message)
    section["status"] = "fail"


def finalize_section(section: dict[str, Any]) -> None:
    if section["errors"]:
        section["status"] = "fail"
    elif section["status"] not in {"needs_review", "not_assessed"}:
        section["status"] = "pass"


def extract_context_graph_order(
    utility_output: str, expected_part: int
) -> tuple[list[str], list[tuple[int, int]]]:
    """Extract ``(sequence_length, context_length)`` in binary graph order."""
    graph_names = re.findall(
        r'"graphName"\s*:\s*"([^"]+)"', utility_output
    )
    if not graph_names:
        raise ValueError("context utility output contains no graphName entries")

    normalized: list[tuple[int, int]] = []
    for graph_name in graph_names:
        match = CONTEXT_GRAPH_PATTERN.fullmatch(graph_name)
        if match is None:
            raise ValueError(
                f"unexpected context graph name {graph_name!r}"
            )
        part = int(match.group("part"))
        if part != expected_part:
            raise ValueError(
                f"context graph {graph_name!r} belongs to part {part}, "
                f"not part {expected_part}"
            )
        normalized.append(
            (
                int(match.group("sequence")),
                int(match.group("context")),
            )
        )
    return graph_names, normalized


def context_graph_order_errors(
    normalized_orders: dict[str, list[tuple[int, int]]],
) -> list[str]:
    """Return graph-order contract failures across linked text partitions."""
    errors: list[str] = []
    if not normalized_orders:
        return ["No text context graph orders were collected."]

    first_name = next(iter(normalized_orders))
    reference = normalized_orders[first_name]
    for filename, order in normalized_orders.items():
        if order != reference:
            errors.append(
                "Linked graph order mismatch: "
                f"{filename} has {order}, while {first_name} has {reference}."
            )

    for filename, order in normalized_orders.items():
        by_context: dict[int, list[int]] = {}
        for sequence_length, context_length in order:
            by_context.setdefault(context_length, []).append(sequence_length)
        for context_length, sequence_lengths in by_context.items():
            if 1 not in sequence_lengths:
                errors.append(
                    f"{filename} context {context_length} has no AR-1 graph."
                )
                continue
            prompt_positions = [
                index
                for index, value in enumerate(sequence_lengths)
                if value > 1
            ]
            token_position = sequence_lengths.index(1)
            if not prompt_positions:
                errors.append(
                    f"{filename} context {context_length} has no prefill graph."
                )
            elif token_position < max(prompt_positions):
                errors.append(
                    f"{filename} context {context_length} links AR-1 before "
                    "its prefill graph."
                )
    return errors


def resolve_input_path(bundle: Path, value: str | None, default: str) -> Path:
    path = Path(value) if value else Path(default)
    if not path.is_absolute():
        path = bundle / path
    return path.resolve()


def sha256_small_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(
    path: Path, section: dict[str, Any], description: str
) -> dict[str, Any] | None:
    if not path.is_file():
        add_error(section, f"Missing {description}: {path}")
        return None
    if path.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
        add_error(
            section,
            f"{description} exceeds the {MAX_EVIDENCE_FILE_BYTES}-byte "
            f"lightweight evidence limit: {path}",
        )
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        add_error(section, f"Cannot parse {description} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        add_error(section, f"{description} must contain a JSON object: {path}")
        return None
    add_check(section, description, str(path))
    return value


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def parse_qairt_version(
    value: Any,
) -> tuple[int, int, int | None, str | None, str | None] | None:
    """Parse the QAIRT version forms emitted by QAI Hub Models metadata."""
    if not isinstance(value, str):
        return None
    match = QAIRT_VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")) if match.group("patch") else None,
        match.group("ident"),
        match.group("flavor"),
    )


def qairt_versions_compatible(compiled: Any, runtime: Any) -> bool:
    """Mirror QAIHM's version matching when either side is a family tag.

    Workbench may serialize its ``2.45`` API-family tag rather than a full
    SDK build. Major/minor must always match. Patch, build identifier, and
    flavor must match only when both sides record that level of detail.
    """
    compiled_version = parse_qairt_version(compiled)
    runtime_version = parse_qairt_version(runtime)
    if compiled_version is None or runtime_version is None:
        return False

    compiled_major, compiled_minor, compiled_patch, compiled_ident, compiled_flavor = (
        compiled_version
    )
    runtime_major, runtime_minor, runtime_patch, runtime_ident, runtime_flavor = (
        runtime_version
    )
    if (compiled_major, compiled_minor) != (runtime_major, runtime_minor):
        return False
    if (
        compiled_patch is not None
        and runtime_patch is not None
        and compiled_patch != runtime_patch
    ):
        return False
    if (
        compiled_ident is not None
        and runtime_ident is not None
        and not (
            compiled_ident.startswith(runtime_ident)
            or runtime_ident.startswith(compiled_ident)
        )
    ):
        return False
    return not (
        compiled_flavor is not None
        and runtime_flavor is not None
        and compiled_flavor != runtime_flavor
    )


def validate_engine_config(
    *,
    section: dict[str, Any],
    config: dict[str, Any] | None,
    filename: str,
    top_key: str,
    expected_bins: list[str],
) -> None:
    if config is None:
        return

    backend_type = nested(config, top_key, "engine", "backend", "type")
    if backend_type != "QnnHtp":
        add_error(
            section,
            f"{filename} backend is {backend_type!r}; expected exactly 'QnnHtp'.",
        )
    else:
        add_check(section, f"{filename} backend", "QnnHtp")

    extension = nested(config, top_key, "engine", "backend", "extensions")
    if extension != "htp_backend_ext_config.json":
        add_error(
            section,
            f"{filename} backend extension is {extension!r}; expected "
            "'htp_backend_ext_config.json'.",
        )
    else:
        add_check(section, f"{filename} HTP extension", extension)

    context_bins = nested(
        config, top_key, "engine", "model", "binary", "ctx-bins"
    )
    if context_bins != expected_bins:
        add_error(
            section,
            f"{filename} ctx-bins are {context_bins!r}; expected {expected_bins!r}.",
        )
    else:
        add_check(
            section,
            f"{filename} context binaries",
            ", ".join(expected_bins),
        )


def validate_bundle(
    bundle: Path, expected_qairt_version: str
) -> dict[str, Any]:
    section = new_section()
    if not bundle.is_dir():
        add_error(section, f"Bundle directory does not exist: {bundle}")
        return section

    add_check(section, "bundle directory", str(bundle))

    compat_manifest_path = bundle / QAIRT245_COMPAT_MARKER
    compatibility_mode = compat_manifest_path.is_file()
    compat_manifest: dict[str, Any] | None = None
    if compatibility_mode:
        compat_manifest = read_json(
            compat_manifest_path,
            section,
            "QAIRT 2.45 compatibility manifest",
        )
        if compat_manifest is not None:
            removed_inputs = compat_manifest.get("removed_runtime_inputs")
            if not isinstance(removed_inputs, list) or set(
                removed_inputs
            ) != DISABLED_DEEPSTACK_INPUTS:
                add_error(
                    section,
                    f"{QAIRT245_COMPAT_MARKER} removed_runtime_inputs is "
                    f"{removed_inputs!r}; expected exactly "
                    f"{sorted(DISABLED_DEEPSTACK_INPUTS)!r}.",
                )
            else:
                add_check(
                    section,
                    "QAIRT 2.45 compatibility mode",
                    "deep-stack graph inputs disabled with provenance marker",
                )
    section["evidence"]["qairt245_compatibility_mode"] = compatibility_mode

    actual_bins = sorted(
        path.name for path in bundle.glob("*.bin") if path.is_file()
    )
    if actual_bins != sorted(EXPECTED_BINS):
        add_error(
            section,
            "Top-level context binaries are "
            f"{actual_bins!r}; expected exactly {sorted(EXPECTED_BINS)!r}.",
        )
    else:
        add_check(section, "exact context-binary set", ", ".join(actual_bins))

    binary_sizes: dict[str, int] = {}
    for filename in EXPECTED_BINS:
        path = bundle / filename
        if not path.is_file():
            add_error(section, f"Missing context binary: {path}")
            continue
        if path.is_symlink():
            add_error(section, f"Context binary must not be a symlink: {path}")
        size = path.stat().st_size
        binary_sizes[filename] = size
        if size <= 0:
            add_error(section, f"Empty context binary: {path}")
    section["evidence"]["context_binary_sizes"] = binary_sizes

    required_files = [
        "embedding_weights.raw",
        "genie_config.json",
        "genie-app-script.txt",
        "htp_backend_ext_config.json",
        "img-enc-htp.json",
        "metadata.json",
        "text-encoder.json",
        "text-generator.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for filename in required_files:
        path = bundle / filename
        if not path.is_file() or path.stat().st_size <= 0:
            add_error(section, f"Missing or empty required bundle file: {path}")

    sample_dir = bundle / "sample_inputs"
    for filename in REQUIRED_SAMPLE_INPUTS:
        path = sample_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            add_error(section, f"Missing or empty sample input: {path}")

    dialog_config = read_json(
        bundle / "genie_config.json", section, "text smoke config"
    )
    text_config = read_json(
        bundle / "text-generator.json", section, "VLM text config"
    )
    vision_config = read_json(
        bundle / "img-enc-htp.json", section, "VLM vision config"
    )
    metadata = read_json(bundle / "metadata.json", section, "bundle metadata")

    validate_engine_config(
        section=section,
        config=dialog_config,
        filename="genie_config.json",
        top_key="dialog",
        expected_bins=EXPECTED_TEXT_BINS,
    )
    validate_engine_config(
        section=section,
        config=text_config,
        filename="text-generator.json",
        top_key="text-generator",
        expected_bins=EXPECTED_TEXT_BINS,
    )
    validate_engine_config(
        section=section,
        config=vision_config,
        filename="img-enc-htp.json",
        top_key="image-encoder",
        expected_bins=EXPECTED_VISION_BINS,
    )

    script_path = bundle / "genie-app-script.txt"
    if script_path.is_file():
        pipeline_script_valid = True
        script_lines = {
            line.strip()
            for line in script_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for required_line in REQUIRED_PIPELINE_LINES:
            if required_line not in script_lines:
                pipeline_script_valid = False
                add_error(
                    section,
                    "genie-app-script.txt is missing the required line: "
                    f"{required_line}",
                )
        wildcard_present = DEEPSTACK_WILDCARD_PIPELINE_LINE in script_lines
        if compatibility_mode and wildcard_present:
            pipeline_script_valid = False
            add_error(
                section,
                "QAIRT 2.45 compatibility bundle must not contain Genie's "
                "unsupported deep-stack wildcard connection.",
            )
        elif not compatibility_mode and not wildcard_present:
            pipeline_script_valid = False
            add_error(
                section,
                "Standard deep-stack bundle is missing the required line: "
                f"{DEEPSTACK_WILDCARD_PIPELINE_LINE}",
            )
        if pipeline_script_valid:
            deepstack_detail = (
                "QAIRT 2.45 compatibility mode omits the unsupported wildcard"
                if compatibility_mode
                else "deep-stack wildcard is wired"
            )
            add_check(
                section,
                "vision pipeline",
                "image encoder, LUT encoder, text generator, sample image, "
                f"and execute command are wired; {deepstack_detail}",
            )

    if metadata is not None:
        supports_vision = nested(metadata, "genie", "supports_vision")
        if supports_vision is not True:
            add_error(
                section,
                "metadata.json does not declare genie.supports_vision=true.",
            )
        else:
            add_check(section, "metadata vision support", "true")

        pipeline_nodes = nested(metadata, "genie", "pipeline", "nodes")
        expected_nodes = {
            "imageEncoder": "img-enc-htp.json",
            "lutEncoder": "text-encoder.json",
            "textGenerator": "text-generator.json",
        }
        if pipeline_nodes != expected_nodes:
            add_error(
                section,
                "metadata.json Genie pipeline nodes are "
                f"{pipeline_nodes!r}; expected {expected_nodes!r}.",
            )
        else:
            add_check(
                section,
                "metadata pipeline nodes",
                "image encoder, LUT encoder, and text generator",
            )

        pipeline_connections = nested(
            metadata, "genie", "pipeline", "connections"
        )
        if not isinstance(pipeline_connections, list):
            add_error(
                section,
                "metadata.json Genie pipeline connections are missing or "
                "not a list.",
            )
        else:
            wildcard_connections = [
                connection
                for connection in pipeline_connections
                if isinstance(connection, dict)
                and "GENIE_NODE_WILDCARD"
                in {
                    connection.get("producer_node_io"),
                    connection.get("consumer_node_io"),
                }
            ]
            expected_wildcards = 0 if compatibility_mode else 1
            if len(wildcard_connections) != expected_wildcards:
                add_error(
                    section,
                    "metadata.json contains "
                    f"{len(wildcard_connections)} wildcard pipeline "
                    f"connections; expected {expected_wildcards}.",
                )
            else:
                add_check(
                    section,
                    "metadata deep-stack connector",
                    (
                        "omitted for QAIRT 2.45 compatibility"
                        if compatibility_mode
                        else "one wildcard connector"
                    ),
                )

        if compatibility_mode:
            model_files = metadata.get("model_files")
            if not isinstance(model_files, dict):
                add_error(
                    section,
                    "Compatibility bundle metadata has no model_files map.",
                )
            else:
                exposed_inputs: dict[str, list[str]] = {}
                for filename in EXPECTED_TEXT_BINS:
                    file_metadata = model_files.get(filename)
                    inputs = (
                        file_metadata.get("inputs")
                        if isinstance(file_metadata, dict)
                        else None
                    )
                    if not isinstance(inputs, dict):
                        add_error(
                            section,
                            f"metadata.json has no input map for {filename}.",
                        )
                        continue
                    unexpected = sorted(
                        set(inputs) & DISABLED_DEEPSTACK_INPUTS
                    )
                    if unexpected:
                        exposed_inputs[filename] = unexpected
                if exposed_inputs:
                    add_error(
                        section,
                        "Compatibility text contexts still expose disabled "
                        f"deep-stack inputs: {exposed_inputs!r}.",
                    )
                else:
                    add_check(
                        section,
                        "compatibility text inputs",
                        "all four parts omit visual_pos_masks and auxiliary "
                        "deep-stack embeddings",
                    )

        metadata_inputs = nested(metadata, "genie", "sample_inputs")
        metadata_input_files: list[str] = []
        if isinstance(metadata_inputs, list):
            metadata_input_files = sorted(
                item["file"]
                for item in metadata_inputs
                if isinstance(item, dict) and isinstance(item.get("file"), str)
            )
        expected_input_files = sorted(
            f"sample_inputs/{name}" for name in REQUIRED_SAMPLE_INPUTS
        )
        if metadata_input_files != expected_input_files:
            add_error(
                section,
                "metadata.json sample input files are "
                f"{metadata_input_files!r}; expected {expected_input_files!r}.",
            )
        else:
            add_check(
                section,
                "metadata sample inputs",
                "all seven text/image/position/attention inputs",
            )

        runtime = metadata.get("runtime")
        if runtime not in {"geniex_qairt", "genie"}:
            add_error(
                section,
                f"metadata.json runtime is {runtime!r}; expected Genie QAIRT.",
            )
        else:
            add_check(section, "metadata runtime", str(runtime))

        precision = metadata.get("precision")
        if precision not in {"w4a16", "w4"}:
            add_error(
                section,
                f"metadata.json precision is {precision!r}; expected "
                "'w4a16' or 'w4'.",
            )
        else:
            add_check(section, "metadata precision", precision)

        tool_versions = metadata.get("tool_versions", {})
        section["evidence"]["tool_versions"] = tool_versions
        if isinstance(tool_versions, dict):
            compiled_qairt = tool_versions.get("qairt")
            if compiled_qairt is None:
                add_warning(
                    section,
                    "metadata.json does not record the compile-time QAIRT "
                    "version; retain the Workbench compile/link job record.",
                )
            elif not qairt_versions_compatible(
                compiled_qairt, expected_qairt_version
            ):
                add_error(
                    section,
                    "Bundle QAIRT version "
                    f"{compiled_qairt!r} is not compatible with the selected "
                    "runtime "
                    f"{expected_qairt_version!r}.",
                )
            else:
                add_check(
                    section,
                    "compile/runtime QAIRT compatibility",
                    f"bundle {compiled_qairt}; runtime {expected_qairt_version}",
                )
                if compiled_qairt != expected_qairt_version:
                    add_warning(
                        section,
                        "Bundle metadata records a compatible QAIRT family or "
                        "partial version rather than the exact EVK build "
                        f"({compiled_qairt!r} versus "
                        f"{expected_qairt_version!r}); retain the Workbench "
                        "compile/link job record.",
                    )

            qaihm_version = tool_versions.get("ai_hub_models")
            if qaihm_version not in {None, "0.58.0"}:
                add_error(
                    section,
                    "Bundle reports QAI Hub Models "
                    f"{qaihm_version!r}; this checker targets 0.58.0.",
                )
            elif qaihm_version == "0.58.0":
                add_check(section, "QAI Hub Models schema", "0.58.0")

    small_artifacts = [
        "genie_config.json",
        "text-generator.json",
        "img-enc-htp.json",
        "genie-app-script.txt",
        "metadata.json",
    ]
    if compatibility_mode:
        small_artifacts.append(QAIRT245_COMPAT_MARKER)
    section["evidence"]["small_artifact_sha256"] = {
        filename: sha256_small_file(bundle / filename)
        for filename in small_artifacts
        if (bundle / filename).is_file()
    }

    finalize_section(section)
    return section


def validate_runtime(
    qairt_home_arg: str, expected_version: str, target: str
) -> dict[str, Any]:
    section = new_section()
    declared_home = Path(qairt_home_arg).expanduser().absolute()
    resolved_home = declared_home.resolve()
    section["evidence"].update(
        {
            "declared_qairt_home": str(declared_home),
            "resolved_qairt_home": str(resolved_home),
            "expected_qairt_version": expected_version,
            "target": target,
        }
    )

    if str(declared_home) != str(resolved_home):
        add_error(
            section,
            f"QAIRT_HOME must be an exact versioned path, not a symlink: "
            f"{declared_home} -> {resolved_home}",
        )
    elif not resolved_home.is_dir():
        add_error(section, f"QAIRT_HOME does not exist: {resolved_home}")
        return section
    else:
        add_check(section, "exact QAIRT path", str(resolved_home))

    if resolved_home.name != expected_version:
        add_error(
            section,
            "QAIRT path basename does not exactly match expected version "
            f"{expected_version!r}: {resolved_home}",
        )

    bin_dir = resolved_home / "bin" / target
    lib_dir = resolved_home / "lib" / target
    dsp_dir = resolved_home / "lib" / "hexagon-v73" / "unsigned"
    expected_paths = {
        "genie-t2t-run": bin_dir / "genie-t2t-run",
        "genie-app": bin_dir / "genie-app",
        "libGenie.so": lib_dir / "libGenie.so",
        "libQnnHtp.so": lib_dir / "libQnnHtp.so",
        "libQnnHtpV73Stub.so": lib_dir / "libQnnHtpV73Stub.so",
        "libQnnHtpV73Skel.so": dsp_dir / "libQnnHtpV73Skel.so",
    }
    for name, path in expected_paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            add_error(section, f"Missing or empty QAIRT artifact {name}: {path}")
        elif name in {"genie-t2t-run", "genie-app"} and not os.access(
            path, os.X_OK
        ):
            add_error(section, f"QAIRT runner is not executable: {path}")
        else:
            add_check(section, name, str(path))

    fastrpc = Path("/dev/fastrpc-cdsp")
    if not fastrpc.exists():
        add_error(section, f"FastRPC device is absent: {fastrpc}")
    elif not os.access(fastrpc, os.R_OK | os.W_OK):
        add_error(
            section,
            f"FastRPC device is not readable and writable by this user: {fastrpc}",
        )
    else:
        add_check(section, "FastRPC access", f"read/write {fastrpc}")

    runner = expected_paths["genie-t2t-run"]
    if runner.is_file():
        library_paths = [str(lib_dir)]
        legacy = resolved_home / "lib" / "aarch64-oe-linux-gcc8.2"
        if legacy.is_dir():
            library_paths.append(str(legacy))
        library_paths.extend(
            ["/usr/lib/aarch64-linux-gnu", "/lib/aarch64-linux-gnu"]
        )
        check_env = dict(os.environ)
        check_env["LD_LIBRARY_PATH"] = ":".join(library_paths)
        try:
            result = subprocess.run(
                ["ldd", str(runner)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=check_env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            add_error(section, f"Could not inspect runner dependencies: {exc}")
        else:
            ldd_output = (result.stdout + result.stderr).strip()
            section["evidence"]["ldd"] = ldd_output
            if result.returncode != 0 or "not found" in ldd_output:
                add_error(
                    section,
                    "genie-t2t-run has unresolved dependencies under the "
                    "constructed QAIRT library path.",
                )
            else:
                add_check(
                    section,
                    "runner dependencies",
                    "all dependencies resolve with the exact QAIRT library path",
                )

    section["evidence"]["required_environment"] = {
        "PATH_prefix": str(bin_dir),
        "LD_LIBRARY_PATH_prefix": str(lib_dir),
        "ADSP_LIBRARY_PATH": str(dsp_dir),
    }
    finalize_section(section)
    return section


def validate_context_graph_order(
    bundle: Path,
    qairt_home_arg: str,
    target: str,
    section: dict[str, Any],
) -> None:
    """Require every linked text part to expose graphs in identical order."""
    utility = (
        Path(qairt_home_arg).expanduser().resolve()
        / "bin"
        / target
        / "qnn-context-binary-utility"
    )
    if not utility.is_file() or not os.access(utility, os.X_OK):
        add_error(
            section,
            "Cannot inspect linked graph order because "
            f"qnn-context-binary-utility is unavailable: {utility}",
        )
        return

    graph_names_by_file: dict[str, list[str]] = {}
    normalized_orders: dict[str, list[tuple[int, int]]] = {}
    for part in range(1, 5):
        filename = f"part{part}_of_4.bin"
        context_binary = bundle / filename
        if not context_binary.is_file():
            continue
        try:
            result = subprocess.run(
                [
                    str(utility),
                    f"--context_binary={context_binary}",
                    "--json_file=/dev/stdout",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            add_error(
                section,
                f"Could not inspect linked graphs in {filename}: {exc}",
            )
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:1000]
            add_error(
                section,
                f"Context utility failed for {filename}: {detail}",
            )
            continue
        try:
            graph_names, normalized = extract_context_graph_order(
                result.stdout, expected_part=part
            )
        except ValueError as exc:
            add_error(section, f"{filename}: {exc}")
            continue
        graph_names_by_file[filename] = graph_names
        normalized_orders[filename] = normalized

    section["evidence"]["context_graph_order"] = graph_names_by_file
    if len(normalized_orders) != 4:
        add_error(
            section,
            "Could not establish graph order for all four text context "
            f"binaries; inspected {sorted(normalized_orders)}.",
        )
    graph_order_errors = context_graph_order_errors(normalized_orders)
    for error in graph_order_errors:
        add_error(section, error)
    if len(normalized_orders) == 4 and not graph_order_errors:
        add_check(
            section,
            "linked text graph order",
            "all four partitions use the same prefill-before-AR-1 order",
        )
    finalize_section(section)


def read_log(path: Path, section: dict[str, Any], description: str) -> str | None:
    if not path.is_file() or path.stat().st_size <= 0:
        add_error(section, f"Missing or empty {description}: {path}")
        return None
    if path.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
        add_error(
            section,
            f"{description} exceeds the {MAX_EVIDENCE_FILE_BYTES}-byte "
            f"lightweight evidence limit: {path}",
        )
        return None
    try:
        value = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add_error(section, f"Cannot read {description} {path}: {exc}")
        return None
    section["evidence"].update(
        {
            "log_path": str(path),
            "log_size_bytes": path.stat().st_size,
            "log_sha256": sha256_small_file(path),
        }
    )
    return value


def fatal_log_matches(log_text: str) -> list[str]:
    matches: list[str] = []
    for line in log_text.splitlines():
        if any(pattern.search(line) for pattern in FATAL_LOG_PATTERNS):
            matches.append(line[:500])
            if len(matches) == 20:
                break
    return matches


def extract_standard_answer(log_text: str) -> str | None:
    match = re.search(r"\[BEGIN\]:(.*?)\[END\]", log_text, re.DOTALL)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer if re.search(r"[A-Za-z]{2}", answer) else None


def metric_value(event: dict[str, Any], name: str) -> float | None:
    metric = event.get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def validate_text(
    text_log: Path,
    profile_path: Path,
    expected_version: str,
    output_expression: str,
) -> dict[str, Any]:
    section = new_section()
    log_text = read_log(text_log, section, "text log")
    if log_text is not None:
        fatal_matches = fatal_log_matches(log_text)
        section["evidence"]["fatal_log_matches"] = fatal_matches
        if fatal_matches:
            add_error(section, "Text log contains fatal/error markers.")

        context_count = len(
            re.findall(r"Using create From Binary", log_text, re.IGNORECASE)
        )
        section["evidence"]["create_from_binary_count"] = context_count
        if context_count < 1:
            add_error(
                section,
                "Text log does not contain a create-from-binary context marker.",
            )
        else:
            add_check(
                section,
                "text context initialization",
                f"{context_count} create-from-binary marker(s)",
            )

        if f"qairt-{expected_version.rsplit('.', 1)[0]}" in log_text:
            add_check(
                section,
                "runtime build marker",
                f"log references QAIRT {expected_version.rsplit('.', 1)[0]}",
            )
        else:
            add_warning(
                section,
                "Text log does not expose a QAIRT build path; rely on the "
                "recorded exact runtime path and ldd evidence.",
            )

        answer = extract_standard_answer(log_text)
        if answer is None:
            add_error(
                section,
                "Text log has no non-empty [BEGIN]: ... [END] answer.",
            )
        else:
            add_check(
                section,
                "text answer",
                f"{len(answer)} characters between [BEGIN] and [END]",
            )
            section["evidence"]["answer"] = answer[:MAX_ANSWER_CHARS]
            section["evidence"]["answer_truncated"] = (
                len(answer) > MAX_ANSWER_CHARS
            )
            try:
                output_pattern = re.compile(output_expression, re.IGNORECASE)
            except re.error as exc:
                add_error(section, f"Invalid --text-output-regex: {exc}")
            else:
                output_match = output_pattern.search(answer)
                section["evidence"]["text_output_regex"] = output_expression
                if output_match is None:
                    add_error(
                        section,
                        "Text answer does not match the expected deterministic "
                        "physical-common-sense result. Override "
                        "--text-output-regex if using another prompt.",
                    )
                else:
                    add_check(
                        section,
                        "text answer content",
                        f"matched {output_match.group(0)!r}",
                    )

    profile = read_json(profile_path, section, "text Genie profile")
    if profile is not None:
        section["evidence"].update(
            {
                "profile_path": str(profile_path),
                "profile_size_bytes": profile_path.stat().st_size,
                "profile_sha256": sha256_small_file(profile_path),
            }
        )
        artifact_type = nested(profile, "header", "artifact_type")
        if artifact_type != "GENIE_PROFILE":
            add_error(
                section,
                f"Profile artifact type is {artifact_type!r}, not 'GENIE_PROFILE'.",
            )

        events: list[dict[str, Any]] = []
        components = profile.get("components")
        if isinstance(components, list):
            for component in components:
                if isinstance(component, dict) and isinstance(
                    component.get("events"), list
                ):
                    events.extend(
                        event
                        for event in component["events"]
                        if isinstance(event, dict)
                    )

        create_event = next(
            (
                event
                for event in events
                if event.get("type") == "GenieDialog_create"
            ),
            None,
        )
        query_event = next(
            (
                event
                for event in events
                if event.get("type") == "GenieDialog_query"
            ),
            None,
        )
        if create_event is None:
            add_error(section, "Profile has no GenieDialog_create event.")
        if query_event is None:
            add_error(section, "Profile has no GenieDialog_query event.")
        else:
            required_metrics = [
                "num-prompt-tokens",
                "prompt-processing-rate",
                "time-to-first-token",
                "num-generated-tokens",
                "token-generation-rate",
            ]
            metrics: dict[str, float] = {}
            for name in required_metrics:
                value = metric_value(query_event, name)
                if value is None or value <= 0:
                    add_error(
                        section,
                        f"Profile metric {name!r} is missing or not positive.",
                    )
                else:
                    metrics[name] = value
            section["evidence"]["profile_metrics"] = metrics
            if len(metrics) == len(required_metrics):
                add_check(
                    section,
                    "text profile metrics",
                    "prompt tokens/rate, TTFT, generated tokens, and decode "
                    "rate are all positive",
                )

    finalize_section(section)
    return section


def validate_vision(
    vision_log: Path, output_expression: str
) -> dict[str, Any]:
    section = new_section()
    log_text = read_log(vision_log, section, "vision log")
    if log_text is None:
        return section

    fatal_matches = fatal_log_matches(log_text)
    section["evidence"]["fatal_log_matches"] = fatal_matches
    if fatal_matches:
        add_error(section, "Vision log contains fatal/error markers.")

    context_count = len(
        re.findall(r"Using create From Binary", log_text, re.IGNORECASE)
    )
    section["evidence"]["create_from_binary_count"] = context_count
    if context_count < 2:
        add_error(
            section,
            "Vision log has fewer than two create-from-binary markers; it "
            "does not prove that both image and text contexts initialized.",
        )
    else:
        add_check(
            section,
            "vision and text context initialization",
            f"{context_count} create-from-binary markers",
        )

    try:
        output_pattern = re.compile(output_expression, re.IGNORECASE)
    except re.error as exc:
        add_error(section, f"Invalid --vision-output-regex: {exc}")
    else:
        output_match = output_pattern.search(log_text)
        section["evidence"]["vision_output_regex"] = output_expression
        if output_match is None:
            add_error(
                section,
                "Vision log does not contain the expected image-conditioned "
                "answer marker. Override --vision-output-regex for a different "
                "sample image.",
            )
        else:
            add_check(
                section,
                "image-conditioned answer",
                f"matched {output_match.group(0)!r}",
            )
            section["evidence"]["vision_output_match"] = output_match.group(0)

    answer = extract_standard_answer(log_text)
    if answer is not None:
        section["evidence"]["standard_answer"] = answer[:MAX_ANSWER_CHARS]
        section["evidence"]["answer_truncated"] = (
            len(answer) > MAX_ANSWER_CHARS
        )

    finalize_section(section)
    return section


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a five-context Cosmos QAIRT bundle and collect separate "
            "text/vision NPU evidence without launching inference."
        )
    )
    parser.add_argument("--bundle", required=True, help="Extracted Genie bundle")
    parser.add_argument(
        "--qairt-home",
        default="/opt/qairt/2.45.0.260326",
        help="Exact, versioned QAIRT runtime path (symlinks are rejected)",
    )
    parser.add_argument(
        "--expected-qairt-version",
        default="2.45.0.260326",
        help="QAIRT version expected in the exact runtime path",
    )
    parser.add_argument(
        "--target",
        default="aarch64-oe-linux-gcc11.2",
        help="QAIRT target directory",
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "text", "vision", "all"),
        default="all",
        help="Evidence phase to require",
    )
    parser.add_argument(
        "--text-log",
        help="Text log path, absolute or relative to the bundle",
    )
    parser.add_argument(
        "--text-profile",
        help="Text profile path, absolute or relative to the bundle",
    )
    parser.add_argument(
        "--text-output-regex",
        default=(
            r"\b(?:fall(?:s|ing)?|drop(?:s|ping)?|gravit(?:y|ational)|"
            r"ground|downward|accelerat(?:e|es|ing))\b"
        ),
        help=(
            "Regex expected in the deterministic text answer; the default "
            "matches the bundled released-ball prompt"
        ),
    )
    parser.add_argument(
        "--vision-log",
        help="Vision log path, absolute or relative to the bundle",
    )
    parser.add_argument(
        "--vision-output-regex",
        default=r"\b(?:dog|puppy|canine)\b",
        help=(
            "Regex expected in the image-conditioned answer; the default "
            "matches the adapter's dog sample image"
        ),
    )
    parser.add_argument(
        "--report",
        help=(
            "JSON report path, absolute or relative to the bundle "
            "(default: npu_evidence.json)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bundle = Path(args.bundle).expanduser().resolve()
    report_path = resolve_input_path(
        bundle, args.report, "npu_evidence.json"
    )

    bundle_section = validate_bundle(
        bundle, args.expected_qairt_version
    )
    validate_context_graph_order(
        bundle,
        args.qairt_home,
        args.target,
        bundle_section,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "mode": args.mode,
        "bundle": str(bundle),
        "sections": {
            "bundle": bundle_section,
            "runtime": validate_runtime(
                args.qairt_home,
                args.expected_qairt_version,
                args.target,
            ),
            "text": {**new_section(), "status": "not_assessed"},
            "vision": {**new_section(), "status": "not_assessed"},
        },
    }

    required_sections = ["bundle", "runtime"]
    if args.mode in {"text", "all"}:
        text_log = resolve_input_path(bundle, args.text_log, "text_smoke.log")
        text_profile = resolve_input_path(
            bundle, args.text_profile, "text_smoke_profile.txt"
        )
        report["sections"]["text"] = validate_text(
            text_log,
            text_profile,
            args.expected_qairt_version,
            args.text_output_regex,
        )
        required_sections.append("text")

    if args.mode in {"vision", "all"}:
        vision_log = resolve_input_path(
            bundle, args.vision_log, "vision_smoke_245.log"
        )
        report["sections"]["vision"] = validate_vision(
            vision_log, args.vision_output_regex
        )
        required_sections.append("vision")

    overall_pass = all(
        report["sections"][name]["status"] == "pass"
        for name in required_sections
    )
    report["required_sections"] = required_sections
    report["overall_status"] = "pass" if overall_pass else "fail"

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_name(f".{report_path.name}.tmp")
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(report_path)
    except OSError as exc:
        print(f"ERROR: cannot write report {report_path}: {exc}", file=sys.stderr)
        return 2

    print(f"Bundle:  {report['sections']['bundle']['status']}")
    print(f"Runtime: {report['sections']['runtime']['status']}")
    print(f"Text:    {report['sections']['text']['status']}")
    print(f"Vision:  {report['sections']['vision']['status']}")
    print(f"Overall: {report['overall_status']}")
    print(f"Report:  {report_path}")

    for name in required_sections:
        for error in report["sections"][name]["errors"]:
            print(f"ERROR [{name}]: {error}", file=sys.stderr)
        for warning in report["sections"][name]["warnings"]:
            print(f"WARN  [{name}]: {warning}", file=sys.stderr)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
