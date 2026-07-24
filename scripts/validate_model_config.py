#!/usr/bin/env python3
"""Validate that a checkpoint is compatible with the local QAIHM adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TEXT = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

EXPECTED_VISION = {
    "depth": 24,
    "hidden_size": 1024,
    "num_heads": 16,
    "out_hidden_size": 2048,
    "patch_size": 16,
    "spatial_merge_size": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    config_path = args.checkpoint.expanduser().resolve() / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    errors: list[str] = []
    if config.get("model_type") != "qwen3_vl":
        errors.append(f"model_type={config.get('model_type')!r}, expected 'qwen3_vl'")

    for section_name, expected in (
        ("text_config", EXPECTED_TEXT),
        ("vision_config", EXPECTED_VISION),
    ):
        actual = config.get(section_name, {})
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                errors.append(
                    f"{section_name}.{key}={actual.get(key)!r}, "
                    f"expected {expected_value!r}"
                )

    if errors:
        print("Checkpoint is not compatible:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Compatible Qwen3-VL-2B checkpoint: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
