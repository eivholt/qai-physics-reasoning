#!/usr/bin/env python3
"""Extract frame-disjoint warehouse pairs for vision activation calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.fetch_nvidia_sdg_warehouse import (
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        _case_frame_indices,
        _destination,
        load_manifest,
        sha256_file,
        validate_asset,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from fetch_nvidia_sdg_warehouse import (
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        _case_frame_indices,
        _destination,
        load_manifest,
        sha256_file,
        validate_asset,
    )


def _evaluation_frames(manifest: dict[str, Any]) -> dict[str, set[int]]:
    frames: dict[str, set[int]] = {}
    for group in ("prediction_cases", "observation_cases", "video_cases"):
        for case in manifest.get(group, []):
            asset_id = str(case["clip_asset_id"])
            frames.setdefault(asset_id, set()).update(
                _case_frame_indices(case)
            )
    return frames


def calibration_pairs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand and validate the tracked, frame-disjoint calibration plan."""
    calibration = manifest.get("vision_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Benchmark manifest has no vision_calibration object")
    sequences = calibration.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("vision_calibration.sequences must be a non-empty list")

    assets = {asset["id"]: asset for asset in manifest["assets"]}
    evaluation = _evaluation_frames(manifest)
    expanded: list[dict[str, Any]] = []
    counts: list[int] = []
    seen_ids: set[str] = set()
    for sequence_index, sequence in enumerate(sequences):
        if not isinstance(sequence, dict):
            raise ValueError(
                f"vision_calibration.sequences[{sequence_index}] must be an object"
            )
        asset_id = sequence.get("clip_asset_id")
        if asset_id not in assets:
            raise ValueError(f"Unknown calibration clip asset: {asset_id!r}")
        asset = assets[asset_id]
        if asset.get("kind") != "preview_clip":
            raise ValueError(f"Calibration asset is not a preview clip: {asset_id}")
        raw_pairs = sequence.get("pair_frame_indices")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError(
                f"Calibration sequence {asset_id} has no pair_frame_indices"
            )
        counts.append(len(raw_pairs))
        source_fps = float(asset["assumed_preview_fps"])
        target_fps = float(calibration["target_sampling_fps"])
        frame_count = int(asset["frame_count"])

        for pair_index, indices_value in enumerate(raw_pairs):
            if (
                not isinstance(indices_value, list)
                or len(indices_value) != 2
                or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in indices_value
                )
            ):
                raise ValueError(
                    f"Calibration pair {asset_id}[{pair_index}] must contain "
                    "exactly two integer frame indices"
                )
            first, second = indices_value
            if first < 0 or second <= first or second >= frame_count:
                raise ValueError(
                    f"Invalid calibration pair {asset_id}[{pair_index}]: "
                    f"{indices_value}"
                )
            overlap = evaluation.get(asset_id, set()).intersection(indices_value)
            if overlap:
                raise ValueError(
                    f"Calibration pair {asset_id}[{pair_index}] overlaps "
                    f"evaluation frame(s): {sorted(overlap)}"
                )
            effective_fps = source_fps / (second - first)
            if not (target_fps * 0.75 <= effective_fps <= target_fps * 1.25):
                raise ValueError(
                    f"Calibration pair {asset_id}[{pair_index}] samples at "
                    f"{effective_fps:.3f} fps, outside the 25% tolerance around "
                    f"{target_fps:.3f} fps"
                )
            pair_id = (
                f"{asset_id}_f{first:04d}_f{second:04d}"
            )
            if pair_id in seen_ids:
                raise ValueError(f"Duplicate calibration pair id: {pair_id}")
            seen_ids.add(pair_id)
            expanded.append(
                {
                    "id": pair_id,
                    "clip_asset_id": asset_id,
                    "frame_indices": [first, second],
                    "timestamps_seconds": [
                        round(first / source_fps, 6),
                        round(second / source_fps, 6),
                    ],
                }
            )

    if len(set(counts)) != 1:
        raise ValueError(
            "Calibration plan must contain the same number of pairs per clip"
        )
    return expanded


def prepare_calibration_pairs(
    benchmark_manifest: Path,
    asset_root: Path,
    output_dir: Path,
) -> Path:
    """Extract planned RGB frames and write a strict local-pair manifest."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required; install it with `python -m pip install Pillow`"
        ) from exc

    benchmark_manifest = benchmark_manifest.expanduser().resolve()
    asset_root = asset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output_dir}")

    manifest = load_manifest(benchmark_manifest)
    pairs = calibration_pairs(manifest)
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    layout = manifest["preview_layout"]
    expected_size = (int(layout["width"]), int(layout["height"]))
    crop = tuple(int(value) for value in layout["rgb_crop_xyxy"])
    temporary = output_dir.with_name(f".{output_dir.name}.partial")
    if temporary.exists():
        raise ValueError(f"Temporary output already exists: {temporary}")

    entries: list[dict[str, Any]] = []
    try:
        for pair in pairs:
            asset = assets[pair["clip_asset_id"]]
            source = _destination(asset_root, asset["relative_path"])
            validate_asset(source, asset)
            pair_dir = temporary / "frames" / pair["id"]
            pair_dir.mkdir(parents=True, exist_ok=True)
            relative_frames: list[str] = []
            hashes: list[str] = []
            with Image.open(source) as preview:
                if preview.size != expected_size:
                    raise ValueError(
                        f"{asset['id']} preview size {preview.size}, "
                        f"expected {expected_size}"
                    )
                if int(getattr(preview, "n_frames", 1)) != int(
                    asset["frame_count"]
                ):
                    raise ValueError(
                        f"{asset['id']} decoded frame count differs from manifest"
                    )
                for index in pair["frame_indices"]:
                    preview.seek(index)
                    rgb = preview.convert("RGB").crop(crop)
                    filename = f"frame_{index:04d}.png"
                    destination = pair_dir / filename
                    rgb.save(destination, format="PNG")
                    relative = destination.relative_to(temporary).as_posix()
                    relative_frames.append(relative)
                    hashes.append(sha256_file(destination))
            entries.append(
                {
                    "id": pair["id"],
                    "frames": relative_frames,
                    "timestamps_seconds": pair["timestamps_seconds"],
                    "sha256": hashes,
                    "source": {
                        "clip_asset_id": pair["clip_asset_id"],
                        "frame_indices": pair["frame_indices"],
                    },
                }
            )

        output_manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Cosmos-Reason2 paired-frame vision activation calibration",
            "source": {
                "benchmark_manifest": str(benchmark_manifest),
                "benchmark_manifest_sha256": hashlib.sha256(
                    benchmark_manifest.read_bytes()
                ).hexdigest(),
                "calibration_id": manifest["vision_calibration"]["id"],
                "relationship_to_evaluation": manifest["vision_calibration"][
                    "relationship_to_evaluation"
                ],
            },
            "pairs": entries,
        }
        temporary.mkdir(parents=True, exist_ok=True)
        (temporary / "paired_calibration_manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "paired_calibration_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_calibration_pairs(
        args.benchmark_manifest,
        args.asset_root,
        args.output_dir,
    )
    print(f"Prepared paired calibration manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
