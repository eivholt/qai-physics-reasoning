#!/usr/bin/env python3
"""Fetch and prepare NVIDIA SDG-Warehouse dataset-card preview media.

Downloads use only Python's standard HTTP client and never add an
Authorization header. Pillow is imported only when RGB frame extraction is
requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "benchmarks" / "nvidia_sdg_warehouse" / "benchmark.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "nvidia_sdg_warehouse"
USER_AGENT = "qai-physics-reasoning-sdg-preview-fetch/1"


class AssetValidationError(RuntimeError):
    """Raised when a fetched or cached asset does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark schema: {manifest.get('schema_version')!r}")
    if not manifest.get("assets") or not manifest.get("prediction_cases"):
        raise ValueError("Manifest must define assets and prediction_cases")
    return manifest


def _destination(output_dir: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe asset relative_path: {relative_path!r}")
    destination = output_dir.joinpath(*relative.parts)
    resolved_output = output_dir.resolve()
    if not destination.resolve().is_relative_to(resolved_output):
        raise ValueError(f"Asset escapes output directory: {relative_path!r}")
    return destination


def validate_asset(path: Path, asset: dict[str, Any]) -> None:
    actual_size = path.stat().st_size
    expected_size = int(asset["size_bytes"])
    if actual_size != expected_size:
        raise AssetValidationError(
            f"{asset['id']}: expected {expected_size} bytes, found {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset["sha256"]:
        raise AssetValidationError(
            f"{asset['id']}: SHA-256 mismatch; expected {asset['sha256']}, "
            f"found {actual_sha256}"
        )


def fetch_asset(
    asset: dict[str, Any],
    output_dir: Path,
    *,
    force: bool = False,
    urlopen: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> tuple[Path, bool]:
    """Fetch one asset and return ``(path, downloaded)``.

    A verified existing asset is reused. With ``force=True``, a bad existing
    asset is left untouched until a fully downloaded replacement passes both
    validation checks.
    """

    destination = _destination(output_dir, asset["relative_path"])
    if destination.exists() and not force:
        validate_asset(destination, asset)
        return destination, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(
        asset["source_url"],
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    try:
        with urlopen(request) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        validate_asset(partial, asset)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination, True


def select_assets(
    manifest: dict[str, Any],
    asset_set: str,
    *,
    required_asset_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    kinds = {
        "all": {"preview_clip", "scenario_still"},
        "clips": {"preview_clip"},
        "stills": {"scenario_still"},
    }
    selected_kinds = kinds[asset_set]
    required = set(required_asset_ids)
    return [
        asset
        for asset in manifest["assets"]
        if asset["kind"] in selected_kinds or asset["id"] in required
    ]


def _prediction_cases(
    manifest: dict[str, Any], case_ids: Iterable[str]
) -> list[dict[str, Any]]:
    available = {case["id"]: case for case in manifest["prediction_cases"]}
    requested = list(case_ids)
    if not requested:
        return list(available.values())
    unknown = sorted(set(requested) - available.keys())
    if unknown:
        raise ValueError(
            f"Unknown prediction case(s): {', '.join(unknown)}; "
            f"choose from {', '.join(available)}"
        )
    return [available[case_id] for case_id in requested]


def _safe_case_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Prediction case id must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(
            f"Unsafe prediction case id (must be one path component): {value!r}"
        )
    return value


def extract_rgb_frames(
    manifest: dict[str, Any],
    output_dir: Path,
    case: dict[str, Any],
) -> list[Path]:
    """Crop selected frames from the leftmost RGB panel of an animated preview."""

    case_id = _safe_case_id(case.get("id"))
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for --extract-rgb; install it with "
            "`python -m pip install Pillow`"
        ) from error

    assets = {asset["id"]: asset for asset in manifest["assets"]}
    asset = assets[case["clip_asset_id"]]
    source = _destination(output_dir, asset["relative_path"])
    validate_asset(source, asset)

    indices = [int(index) for index in case["rgb_frame_indices"]]
    if not indices or min(indices) < 0:
        raise ValueError(f"{case['id']}: frame indices must be non-negative")

    layout = manifest["preview_layout"]
    width = int(layout["width"])
    height = int(layout["height"])
    crop = tuple(int(value) for value in layout["rgb_crop_xyxy"])
    destination_dir = _destination(output_dir, f"frames/{case_id}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with Image.open(source) as preview:
        if preview.size != (width, height):
            raise AssetValidationError(
                f"{asset['id']}: expected preview dimensions {(width, height)}, "
                f"found {preview.size}"
            )
        frame_count = int(getattr(preview, "n_frames", 1))
        if max(indices) >= frame_count:
            raise AssetValidationError(
                f"{asset['id']}: requested frame {max(indices)}, "
                f"but preview has {frame_count} frame(s)"
            )
        if frame_count != int(asset["frame_count"]):
            raise AssetValidationError(
                f"{asset['id']}: manifest records {asset['frame_count']} frames, "
                f"decoder found {frame_count}"
            )

        for index in indices:
            preview.seek(index)
            rgb = preview.convert("RGB").crop(crop)
            output = destination_dir / f"frame_{index:04d}.png"
            rgb.save(output, format="PNG")
            written.append(output)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--asset-set",
        choices=("all", "clips", "stills"),
        default="all",
        help="asset group to fetch (default: all)",
    )
    parser.add_argument(
        "--extract-rgb",
        action="store_true",
        help="extract selected two-frame cases from the leftmost RGB panel",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="prediction case to extract; repeatable (default: all cases)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace cached inputs after a new download passes validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    cases = _prediction_cases(manifest, args.case) if args.extract_rgb else []
    required = [case["clip_asset_id"] for case in cases]

    for asset in select_assets(
        manifest, args.asset_set, required_asset_ids=required
    ):
        path, downloaded = fetch_asset(asset, args.output, force=args.force)
        action = "downloaded" if downloaded else "verified"
        print(f"{action}: {path}")

    for case in cases:
        for path in extract_rgb_frames(manifest, args.output, case):
            print(f"extracted: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssetValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
