#!/usr/bin/env python3
"""Finalize a compiled QAIRT 2.45 compatibility bundle.

The text graphs in a compatibility checkpoint no longer consume Qwen3-VL
deepstack inputs. If an export was packaged by an older installed copy of the
adapter, however, its metadata may still contain Genie's unsupported wildcard
connector. This script removes exactly that stale connector, embeds the
checkpoint transform manifest, and leaves every compiled context unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

COMPAT_MARKER = "qairt_245_compat.json"
GENIE_SCRIPT = "genie-app-script.txt"
METADATA = "metadata.json"
WILDCARD = "GENIE_NODE_WILDCARD"
REQUIRED_CONTEXTS = {
    "part1_of_4.bin",
    "part2_of_4.bin",
    "part3_of_4.bin",
    "part4_of_4.bin",
    "vision_encoder.bin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_root(names: list[str]) -> str:
    roots = {
        PurePosixPath(name).parts[0]
        for name in names
        if PurePosixPath(name).parts
    }
    if len(roots) != 1:
        raise ValueError(
            f"Bundle must have exactly one top-level directory, found {sorted(roots)}"
        )
    root = next(iter(roots)) + "/"
    if not all(name == root or name.startswith(root) for name in names):
        raise ValueError("Bundle contains entries outside its top-level directory")
    return root


def _load_compat_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        removed_inputs = set(manifest["removed_runtime_inputs"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid compatibility manifest: {path}") from exc

    expected = {
        "visual_pos_masks",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2",
    }
    if removed_inputs != expected:
        raise ValueError(
            "Compatibility manifest removed-input set does not match the "
            f"QAIRT 2.45 contract: {sorted(removed_inputs)}"
        )
    return manifest


def _patch_genie_script(payload: bytes) -> bytes:
    text = payload.decode("utf-8")
    wildcard_lines = [line for line in text.splitlines() if WILDCARD in line]
    if len(wildcard_lines) != 1:
        raise ValueError(
            f"Expected exactly one wildcard pipeline line, found {len(wildcard_lines)}"
        )
    patched_lines = [
        line for line in text.splitlines() if WILDCARD not in line
    ]
    patched = "\n".join(patched_lines)
    if text.endswith("\n"):
        patched += "\n"
    return patched.encode("utf-8")


def _patch_metadata(
    payload: bytes,
    *,
    removed_inputs: set[str],
) -> bytes:
    try:
        metadata = json.loads(payload)
        connections = metadata["genie"]["pipeline"]["connections"]
        model_files = metadata["model_files"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Bundle metadata has an unexpected schema") from exc

    wildcard_connections = [
        connection
        for connection in connections
        if WILDCARD
        in {
            connection.get("producer_node_io"),
            connection.get("consumer_node_io"),
        }
    ]
    if len(wildcard_connections) != 1:
        raise ValueError(
            "Expected exactly one wildcard metadata connection, found "
            f"{len(wildcard_connections)}"
        )
    wildcard_connection = wildcard_connections[0]
    if {
        wildcard_connection.get("producer_node_io"),
        wildcard_connection.get("consumer_node_io"),
    } != {WILDCARD}:
        raise ValueError("Wildcard metadata connection is only partially wildcard")

    metadata["genie"]["pipeline"]["connections"] = [
        connection
        for connection in connections
        if connection is not wildcard_connection
    ]

    primary_image_connections = [
        connection
        for connection in metadata["genie"]["pipeline"]["connections"]
        if connection.get("producer_node_io")
        == "GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT"
        and connection.get("consumer_node_io")
        == "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT"
    ]
    if len(primary_image_connections) != 1:
        raise ValueError("Primary image-embedding pipeline connection is missing")

    exposed_removed_inputs: dict[str, list[str]] = {}
    for filename, file_metadata in model_files.items():
        inputs = set(file_metadata.get("inputs", {}))
        unexpected = sorted(inputs & removed_inputs)
        if unexpected:
            exposed_removed_inputs[filename] = unexpected
    if exposed_removed_inputs:
        raise ValueError(
            "Compiled contexts still expose disabled deepstack inputs: "
            f"{exposed_removed_inputs}"
        )

    metadata.setdefault("supplementary_files", {})[COMPAT_MARKER] = (
        "QAIRT 2.45 compatibility transform and bundle-finalization provenance."
    )
    return (json.dumps(metadata, indent=2) + "\n").encode("utf-8")


def _replacement_zip_info(source_info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Clone archive attributes without retaining the old payload size/CRC."""
    replacement = zipfile.ZipInfo(source_info.filename, source_info.date_time)
    replacement.compress_type = source_info.compress_type
    replacement.comment = source_info.comment
    replacement.extra = source_info.extra
    replacement.create_system = source_info.create_system
    replacement.create_version = source_info.create_version
    replacement.extract_version = source_info.extract_version
    replacement.internal_attr = source_info.internal_attr
    replacement.external_attr = source_info.external_attr
    return replacement


def finalize_bundle(
    source: Path,
    destination: Path,
    compat_manifest_path: Path,
    *,
    context_replacements: dict[str, Path] | None = None,
) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    compat_manifest_path = compat_manifest_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source bundle does not exist: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite: {destination}"
        )

    replacements: dict[str, Path] = {}
    for name, replacement_path in (context_replacements or {}).items():
        if name not in REQUIRED_CONTEXTS:
            raise ValueError(
                f"Replacement context must be one of {sorted(REQUIRED_CONTEXTS)}: "
                f"{name!r}"
            )
        resolved_replacement = replacement_path.expanduser().resolve()
        if not resolved_replacement.is_file():
            raise FileNotFoundError(
                f"Replacement context does not exist: {resolved_replacement}"
            )
        replacements[name] = resolved_replacement

    manifest = _load_compat_manifest(compat_manifest_path)
    removed_inputs = set(manifest["removed_runtime_inputs"])
    manifest["bundle_finalization"] = {
        "source_archive": str(source),
        "source_archive_sha256": sha256_file(source),
        "compiled_contexts_modified": False,
        "removed_pipeline_connection": f"{WILDCARD} -> {WILDCARD}",
        "replacement_contexts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(replacements.items())
        },
    }
    if replacements:
        manifest["bundle_finalization"]["compiled_contexts_modified"] = True
    marker_payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    try:
        with (
            zipfile.ZipFile(source, "r") as source_zip,
            zipfile.ZipFile(temporary, "w", allowZip64=True) as output_zip,
        ):
            infos = source_zip.infolist()
            names = [info.filename for info in infos]
            root = _archive_root(names)
            required = {
                root + GENIE_SCRIPT,
                root + METADATA,
                *(root + filename for filename in REQUIRED_CONTEXTS),
            }
            missing = sorted(required - set(names))
            if missing:
                raise ValueError(f"Bundle is missing required entries: {missing}")
            if root + COMPAT_MARKER in names:
                raise ValueError(
                    f"Source bundle already contains {COMPAT_MARKER}; "
                    "refusing an ambiguous second finalization"
                )

            for info in infos:
                relative_name = info.filename.removeprefix(root)
                if relative_name in replacements:
                    replacement_info = _replacement_zip_info(info)
                    with (
                        replacements[relative_name].open("rb") as source_entry,
                        output_zip.open(
                            replacement_info,
                            "w",
                            force_zip64=True,
                        ) as output_entry,
                    ):
                        shutil.copyfileobj(
                            source_entry,
                            output_entry,
                            length=8 * 1024 * 1024,
                        )
                elif info.filename == root + GENIE_SCRIPT:
                    output_zip.writestr(
                        info,
                        _patch_genie_script(source_zip.read(info)),
                    )
                elif info.filename == root + METADATA:
                    output_zip.writestr(
                        info,
                        _patch_metadata(
                            source_zip.read(info),
                            removed_inputs=removed_inputs,
                        ),
                    )
                elif info.is_dir():
                    output_zip.writestr(info, b"")
                else:
                    with (
                        source_zip.open(info, "r") as source_entry,
                        output_zip.open(info, "w", force_zip64=True) as output_entry,
                    ):
                        shutil.copyfileobj(
                            source_entry,
                            output_entry,
                            length=8 * 1024 * 1024,
                        )

            marker_info = zipfile.ZipInfo(root + COMPAT_MARKER)
            marker_info.compress_type = zipfile.ZIP_DEFLATED
            marker_info.external_attr = 0o100644 << 16
            output_zip.writestr(marker_info, marker_payload)

        with zipfile.ZipFile(temporary, "r") as result_zip:
            root = _archive_root(result_zip.namelist())
            corrupt_entry = result_zip.testzip()
            if corrupt_entry is not None:
                raise ValueError(
                    f"Finalized bundle has a corrupt entry: {corrupt_entry}"
                )
            if WILDCARD.encode() in result_zip.read(root + GENIE_SCRIPT):
                raise ValueError("Finalized Genie script still contains wildcard")
            if WILDCARD in result_zip.read(root + METADATA).decode("utf-8"):
                raise ValueError("Finalized metadata still contains wildcard")
            if root + COMPAT_MARKER not in result_zip.namelist():
                raise ValueError("Finalized bundle is missing compatibility manifest")

        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original QAIHM release zip")
    parser.add_argument(
        "destination",
        type=Path,
        help="Finalized release zip (must not already exist)",
    )
    parser.add_argument(
        "--compat-manifest",
        type=Path,
        required=True,
        help=f"Path to the checkpoint's {COMPAT_MARKER}",
    )
    parser.add_argument(
        "--replacement-context",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Replace one linked context while finalizing, for example "
            "part3_of_4.bin=/path/to/locally-linked.bin. Repeat as needed."
        ),
    )
    return parser


def _parse_context_replacements(values: list[str]) -> dict[str, Path]:
    replacements: dict[str, Path] = {}
    for value in values:
        name, separator, path_text = value.partition("=")
        if not separator or not name or not path_text:
            raise ValueError(
                f"Invalid --replacement-context {value!r}; expected NAME=PATH"
            )
        if name in replacements:
            raise ValueError(f"Duplicate replacement context: {name}")
        replacements[name] = Path(path_text)
    return replacements


def main() -> None:
    args = build_parser().parse_args()
    result = finalize_bundle(
        args.source,
        args.destination,
        args.compat_manifest,
        context_replacements=_parse_context_replacements(
            args.replacement_context
        ),
    )
    print(f"Finalized QAIRT 2.45 bundle: {result}")
    print(f"SHA-256: {sha256_file(result)}")


if __name__ == "__main__":
    main()
