#!/usr/bin/env python3
"""Build a hash-pinned, secret-redacted provisioner payload for one installer platform."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT / "Build" / "release_payload_template.json"
DEFAULT_SOURCE = PROJECT / "Build" / "ReleasePayloadSource"
DEFAULT_OUTPUT = PROJECT / "Provisioner" / "Payload"
DEFAULT_SECRETS = PROJECT / "Build" / "Secrets" / "credentials.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "macos"), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_SECRETS)
    parser.add_argument("--allow-download", action="store_true", help="Permit hash-pinned artifacts with URLs to remain external")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_unique(source: Path, filename: str) -> Path | None:
    matches = [path for path in source.rglob(filename) if path.is_file()]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} below {source}; found {len(matches)}")
    return matches[0]


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".copying")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    released: list[dict[str, object]] = []
    for original in template["artifacts"]:
        entry = dict(original)
        platform = entry.get("platform")
        if platform and platform != args.platform:
            continue
        filename = str(entry["filename"])
        found = find_unique(source, filename) if source.is_dir() else None
        if found:
            destination = output / str(entry["path"])
            copy_atomic(found, destination)
            entry["bytes"] = destination.stat().st_size
            entry["sha256"] = file_hash(destination)
            expected_size = original.get("bytes")
            expected_hash = str(original.get("sha256", ""))
            if expected_size and int(expected_size) != entry["bytes"]:
                raise RuntimeError(f"Pinned size mismatch for {filename}")
            if expected_hash and expected_hash.lower() != entry["sha256"]:
                raise RuntimeError(f"Pinned SHA-256 mismatch for {filename}")
            print(f"packaged {entry['id']}: {entry['bytes']} bytes sha256={entry['sha256']}")
        elif entry.get("download_url") and args.allow_download and entry.get("sha256") and entry.get("bytes"):
            print(f"external hash-pinned download {entry['id']}: {entry['download_url']}")
        elif entry.get("required", True):
            raise FileNotFoundError(f"Required release artifact is missing: {filename} (searched {source})")
        else:
            raise FileNotFoundError(
                f"Optional artifact {filename} is not packaged; pass --allow-download to release its pinned URL"
            )
        released.append(entry)

    if args.credentials.is_file():
        values = json.loads(args.credentials.read_text(encoding="utf-8"))
        allowed = {"huggingface_token", "qualcomm_ai_hub_api_token", "evk_username", "evk_password"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported credential keys: {sorted(unknown)}")
        (output / "credentials.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        print(f"embedded {len(values)} credential fields (values intentionally not printed)")

    manifest = {
        "schema_version": 1,
        "platform": args.platform,
        "private_personal_use": True,
        "artifacts": released,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "payload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output / 'payload_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
