#!/usr/bin/env python3
"""Copy lightweight Hugging Face metadata into a quantized checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SOURCE_CHECKPOINT_MARKER = "source_checkpoint.json"

METADATA_FILES = [
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def write_source_checkpoint_marker(source: Path, destination: Path) -> Path:
    """Record the local BF16 snapshot required by QAIHM during later export."""
    marker = destination / SOURCE_CHECKPOINT_MARKER
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "source_checkpoint": str(source),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--notice",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "NOTICE",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        print(f"Source checkpoint directory does not exist: {source}")
        return 1
    destination.mkdir(parents=True, exist_ok=True)

    for name in METADATA_FILES:
        src = source / name
        dst = destination / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {name}")

    notice_dst = destination / "NOTICE"
    if args.notice.is_file() and not notice_dst.exists():
        shutil.copy2(args.notice, notice_dst)
        print("Copied NOTICE")

    marker = write_source_checkpoint_marker(source, destination)
    print(f"Wrote {marker.name}")

    required = [
        destination / "config.json",
        destination / "tokenizer.json",
        destination / "model_dynamic.onnx",
        destination / "model.data",
        destination / "model.encodings",
        destination / "embedding_weights.raw",
        destination / "vision_encoder.onnx",
        destination / "vision_encoder.encodings",
        marker,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing expected checkpoint files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print(f"Checkpoint ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
