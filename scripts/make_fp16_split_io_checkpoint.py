#!/usr/bin/env python3
"""Create an experimental W4A16 checkpoint with FP16 split-boundary tensors.

This is a controlled diagnostic, not the preferred deployment transform.
The measured W4A16 native chain accumulates HTP-vs-QuantSim drift, and its
three inter-part hidden tensors have coarse INT16 grids because two BOS
features are large outliers. Later traces also showed that adjacent encodings
match exactly and that captured QuantSim boundaries already lie on those
grids, so split-I/O precision alone does not explain all observed error.

The transform changes only those three hidden-state encodings to FLOAT16.
Weights, internal W4A16 activations, KV tensors, graph topology, and the
vision encoder remain unchanged. Prefer ``make_w4_fp16_checkpoint.py`` when
testing FP16 activations throughout the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ENCODINGS_FILENAME = "model.encodings"
MARKER_FILENAME = "fp16_split_io.json"
DEFAULT_BOUNDARIES = ("add_15805", "add_30134", "add_44463")
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


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _load_encodings(path: Path) -> dict[str, Any]:
    try:
        encodings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read AIMET encodings: {path}") from exc
    if not isinstance(encodings, dict):
        raise ValueError("AIMET encodings must contain a JSON object")
    activations = encodings.get("activation_encodings")
    if not isinstance(activations, list):
        raise ValueError("activation_encodings must use AIMET 1.0 list format")
    return encodings


def _patch_boundaries(
    encodings: dict[str, Any], boundaries: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    activations = encodings["activation_encodings"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for entry in activations:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            by_name.setdefault(entry["name"], []).append(entry)

    original: dict[str, dict[str, Any]] = {}
    for name in boundaries:
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one activation encoding for {name!r}, "
                f"found {len(matches)}"
            )
        entry = matches[0]
        if entry.get("dtype") != "INT" or entry.get("bw") != 16:
            raise ValueError(
                f"Boundary {name!r} is not an INT16 encoding: {entry}"
            )
        original[name] = dict(entry)
        entry.clear()
        entry.update({"name": name, **FLOAT16_ENCODING})
    return original


def create_fp16_split_io_checkpoint(
    source: Path,
    destination: Path,
    *,
    boundaries: tuple[str, ...] = DEFAULT_BOUNDARIES,
) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source checkpoint is not a directory: {source}")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if not boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError("Boundary tensor names must be non-empty and unique")

    source_encodings = source / ENCODINGS_FILENAME
    encodings = _load_encodings(source_encodings)
    original = _patch_boundaries(encodings, boundaries)

    shutil.copytree(source, destination, copy_function=_hardlink_or_copy)
    try:
        # Break the hard link before replacing the encoding file.
        temporary = destination / f".{ENCODINGS_FILENAME}.tmp"
        temporary.write_text(
            json.dumps(encodings, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination / ENCODINGS_FILENAME)

        marker = {
            "schema_version": 1,
            "purpose": "Use FP16 I/O at independently compiled text partitions",
            "source_checkpoint": str(source),
            "source_encodings_sha256": sha256_file(source_encodings),
            "boundary_tensors": list(boundaries),
            "original_encodings": original,
            "replacement_encoding": FLOAT16_ENCODING,
            "unchanged": [
                "model_dynamic.onnx",
                "model.data",
                "weight encodings",
                "internal activation encodings",
                "KV-cache encodings",
                "vision encoder",
            ],
        }
        (destination / MARKER_FILENAME).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--boundary",
        action="append",
        dest="boundaries",
        help="Override a split tensor name (repeat for each boundary)",
    )
    args = parser.parse_args()
    boundaries = tuple(args.boundaries or DEFAULT_BOUNDARIES)
    output = create_fp16_split_io_checkpoint(
        args.source,
        args.destination,
        boundaries=boundaries,
    )
    print(f"Created FP16 split-I/O checkpoint: {output}")


if __name__ == "__main__":
    main()
