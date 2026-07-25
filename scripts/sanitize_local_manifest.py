#!/usr/bin/env python3
"""Create a publishable, path-sanitized derivative of a local JSON manifest.

Raw execution manifests intentionally retain absolute artifact paths for local
reproducibility and are marked ``local_only_unsanitized``.  This tool replaces
local paths/private IPv4 addresses, redacts credential-shaped fields, records
the source-manifest hash, and refuses to publish host-model evidence whose
weight shards lack SHA-256 identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

LOCAL_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|tmp|mnt|Users|var|opt)(?:/|$))"
    r"[^\s\"'<>]*"
)
PRIVATE_IPV4 = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
SECRET_KEYS = {
    "api_key",
    "api_token",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "hf_token",
    "huggingface_token",
    "hf_api_token",
    "token",
}
TOKEN_VALUE = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,})\b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitized_path(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    basename = normalized.rsplit("/", 1)[-1]
    return f"<LOCAL_PATH>/{basename}" if basename else "<LOCAL_PATH>"


def sanitize_value(value: Any, key: str | None = None) -> Any:
    if key is not None and key.lower() in SECRET_KEYS:
        return "<REDACTED>"
    if isinstance(value, dict):
        return {
            child_key: sanitize_value(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        value = LOCAL_PATH.sub(
            lambda match: sanitized_path(match.group(0)),
            value,
        )
        value = PRIVATE_IPV4.sub("<EVK IP>", value)
        return TOKEN_VALUE.sub("<REDACTED>", value)
    return value


def validate_publishable_host_weights(manifest: dict[str, Any]) -> None:
    model = manifest.get("model")
    if not isinstance(model, dict) or "weight_shards" not in model:
        return
    shards = model.get("weight_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Host model evidence has no weight-shard inventory")
    missing = [
        record.get("name", f"index {index}")
        for index, record in enumerate(shards)
        if not isinstance(record, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
    ]
    if missing:
        raise ValueError(
            "Publishable host evidence requires SHA-256 for every weight "
            f"shard; missing: {missing}"
        )


def sanitize_manifest(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")
    validate_publishable_host_weights(manifest)
    sanitized = sanitize_value(manifest)
    sanitized["publication"] = {
        "classification": "publishable_sanitized",
        "contains_local_paths": False,
        "source_manifest_sha256": sha256_file(input_path),
        "sanitizer": "scripts/sanitize_local_manifest.py",
    }
    serialized = json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
    if LOCAL_PATH.search(serialized):
        raise ValueError("Sanitized manifest still contains a local path")
    if PRIVATE_IPV4.search(serialized):
        raise ValueError("Sanitized manifest still contains a private IPv4")
    if TOKEN_VALUE.search(serialized):
        raise ValueError("Sanitized manifest still contains a credential token")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()
    result = sanitize_manifest(args.input_manifest, args.output_manifest)
    print(
        json.dumps(
            {
                "output": str(args.output_manifest.resolve()),
                "classification": result["publication"]["classification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
