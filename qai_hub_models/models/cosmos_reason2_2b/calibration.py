"""Deterministic vision-calibration inputs for Cosmos-Reason2-2B."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".webp")
PAIRED_MANIFEST_SCHEMA_VERSION = 1
SINGLE_FRAME_DUPLICATE_MODE = "duplicate_single_frame"
VISION_PATCH_SIZE = 16
VISION_TEMPORAL_PATCH_SIZE = 2
VISION_INPUT_CHANNELS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_class_balanced_paths(
    train_dir: Path,
    num_samples: int,
) -> list[Path]:
    """Select sorted images round-robin across sorted class directories."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    class_images: list[list[Path]] = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if images:
            class_images.append(images)

    available = sum(len(images) for images in class_images)
    if available < num_samples:
        raise RuntimeError(
            f"Calibration dataset has {available} images but "
            f"{num_samples} samples were requested."
        )

    selected: list[Path] = []
    sample_index = 0
    while len(selected) < num_samples:
        for images in class_images:
            if sample_index < len(images):
                selected.append(images[sample_index])
                if len(selected) == num_samples:
                    break
        sample_index += 1
    return selected


def load_imagenette_calibration_data(
    *,
    hf_repo: str,
    num_samples: int,
    image_height: int,
    image_width: int,
) -> list[Any]:
    """Load a deterministic, class-balanced Imagenette calibration set."""
    from PIL import Image
    from transformers import AutoProcessor

    from qai_hub_models.datasets.imagenet import IMAGENETTE_ASSET

    IMAGENETTE_ASSET.fetch(extract=True)
    train_dir = IMAGENETTE_ASSET.extracted_path / "train"
    image_paths = select_class_balanced_paths(train_dir, num_samples)
    processor = AutoProcessor.from_pretrained(hf_repo)

    calibration_data = []
    for image_path in image_paths:
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((image_width, image_height))
        inputs = processor.image_processor(
            images=[image],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].squeeze(0)
        calibration_data.append(
            pixel_values.detach().cpu().to(torch.float32).numpy()
        )
    return calibration_data


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read paired calibration manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Paired calibration manifest must be a JSON object")
    if value.get("schema_version") != PAIRED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Paired calibration manifest schema_version must be "
            f"{PAIRED_MANIFEST_SCHEMA_VERSION}"
        )
    return value


def resolve_paired_frame_paths(
    manifest_path: Path,
    num_samples: int,
) -> list[tuple[Path, Path]]:
    """Resolve and validate the first ``num_samples`` local frame pairs."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    temporal_mode = manifest.get("temporal_mode", "distinct_frames")
    if temporal_mode not in ("distinct_frames", SINGLE_FRAME_DUPLICATE_MODE):
        raise ValueError(
            "Paired calibration manifest temporal_mode must be "
            "distinct_frames or duplicate_single_frame"
        )
    allow_duplicate_frame = temporal_mode == SINGLE_FRAME_DUPLICATE_MODE
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Paired calibration manifest must contain a pairs list")
    if len(pairs) < num_samples:
        raise ValueError(
            f"Paired calibration manifest has {len(pairs)} pairs but "
            f"{num_samples} samples were requested"
        )

    resolved_pairs: list[tuple[Path, Path]] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(pairs[:num_samples]):
        if not isinstance(entry, dict):
            raise ValueError(f"pairs[{index}] must be an object")
        pair_id = entry.get("id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"pairs[{index}].id must be a non-empty string")
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate paired calibration id: {pair_id}")
        seen_ids.add(pair_id)

        frames = entry.get("frames")
        if (
            not isinstance(frames, list)
            or len(frames) != VISION_TEMPORAL_PATCH_SIZE
            or not all(isinstance(value, str) and value for value in frames)
        ):
            raise ValueError(
                f"pairs[{index}].frames must contain exactly two paths"
            )
        timestamps = entry.get("timestamps_seconds")
        if (
            not isinstance(timestamps, list)
            or len(timestamps) != VISION_TEMPORAL_PATCH_SIZE
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
                for value in timestamps
            )
            or timestamps[1] <= timestamps[0]
        ):
            raise ValueError(
                f"pairs[{index}].timestamps_seconds must contain two "
                "finite, non-negative, increasing values"
            )
        if any("://" in value for value in frames):
            raise ValueError(
                f"pairs[{index}] contains a URL; only local files are allowed"
            )

        frame_paths = tuple(
            (
                Path(value).expanduser()
                if Path(value).is_absolute()
                else manifest_path.parent / value
            ).resolve()
            for value in frames
        )
        if frame_paths[0] == frame_paths[1] and not allow_duplicate_frame:
            raise ValueError(f"pairs[{index}] repeats the same frame path")
        for frame_path in frame_paths:
            if (
                not frame_path.is_file()
                or frame_path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                raise ValueError(
                    f"pairs[{index}] frame is not a supported local image: "
                    f"{frame_path}"
                )

        expected_hashes = entry.get("sha256")
        if (
            not isinstance(expected_hashes, list)
            or len(expected_hashes) != VISION_TEMPORAL_PATCH_SIZE
            or not all(
                isinstance(value, str)
                and len(value) == 64
                and all(
                    character in "0123456789abcdefABCDEF"
                    for character in value
                )
                for value in expected_hashes
            )
        ):
            raise ValueError(
                f"pairs[{index}].sha256 must contain two SHA-256 strings"
            )
        actual_hashes = []
        for frame_path, expected in zip(
            frame_paths, expected_hashes, strict=True
        ):
            actual = sha256_file(frame_path)
            actual_hashes.append(actual)
            if actual.lower() != expected.lower():
                raise ValueError(
                    f"pairs[{index}] SHA-256 mismatch for {frame_path}"
                )

        if actual_hashes[0] == actual_hashes[1] and not allow_duplicate_frame:
            raise ValueError(f"pairs[{index}] contains duplicate frame content")
        if allow_duplicate_frame and actual_hashes[0] != actual_hashes[1]:
            raise ValueError(
                f"pairs[{index}] must duplicate one frame in "
                f"{SINGLE_FRAME_DUPLICATE_MODE} mode"
            )
        resolved_pairs.append(frame_paths)
    return resolved_pairs


def _default_processor_loader(hf_repo: str) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(hf_repo, local_files_only=True)


def _default_frame_loader(
    path: Path,
    size: tuple[int, int] | None,
) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        if size is None:
            return image.copy()
        resampling = getattr(Image, "Resampling", Image)
        return image.resize(
            size,
            resample=resampling.BICUBIC,
        ).copy()


def load_paired_frame_calibration_data(
    *,
    hf_repo: str,
    manifest_path: Path,
    num_samples: int,
    image_height: int,
    image_width: int,
    processor_loader: Callable[[str], Any] = _default_processor_loader,
    frame_loader: Callable[
        [Path, tuple[int, int] | None], Any
    ] = _default_frame_loader,
) -> list[Any]:
    """Pack local temporal inputs with inference-identical resizing."""
    if image_height % VISION_PATCH_SIZE or image_width % VISION_PATCH_SIZE:
        raise ValueError(
            f"Image dimensions must be divisible by {VISION_PATCH_SIZE}"
        )
    manifest = _load_manifest(manifest_path.expanduser().resolve())
    allow_static_temporal_patch = (
        manifest.get("temporal_mode") == SINGLE_FRAME_DUPLICATE_MODE
    )
    frame_pairs = resolve_paired_frame_paths(manifest_path, num_samples)
    processor = processor_loader(hf_repo)
    video_processor = getattr(processor, "video_processor", None)
    if not callable(video_processor):
        raise ValueError("Local processor has no callable video_processor")

    grid_h = image_height // VISION_PATCH_SIZE
    grid_w = image_width // VISION_PATCH_SIZE
    expected_grid = [[1, grid_h, grid_w]]
    expected_shape = (
        grid_h * grid_w,
        VISION_INPUT_CHANNELS
        * VISION_TEMPORAL_PATCH_SIZE
        * VISION_PATCH_SIZE
        * VISION_PATCH_SIZE,
    )
    calibration_data = []
    for pair_index, frame_paths in enumerate(frame_pairs):
        original_frames = [frame_loader(path, None) for path in frame_paths]
        outputs = video_processor(
            videos=[original_frames],
            return_tensors="pt",
            do_sample_frames=False,
        )

        def unpack(
            value: Any,
        ) -> tuple[torch.Tensor, list[list[int]]]:
            try:
                pixel_values_value = value["pixel_values_videos"]
                grid_value = value["video_grid_thw"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "Video processor did not return Qwen3-VL video tensors"
                ) from exc
            pixels = (
                torch.as_tensor(pixel_values_value)
                .detach()
                .cpu()
                .to(torch.float32)
            )
            rows = torch.as_tensor(grid_value).detach().cpu().tolist()
            return pixels, rows

        pixel_values, grid_rows = unpack(outputs)
        if (
            tuple(pixel_values.shape) != expected_shape
            or grid_rows != expected_grid
        ):
            resized_frames = [
                frame_loader(path, (image_width, image_height))
                for path in frame_paths
            ]
            outputs = video_processor(
                videos=[resized_frames],
                return_tensors="pt",
                do_resize=False,
                do_sample_frames=False,
            )
            pixel_values, grid_rows = unpack(outputs)

        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                f"Pair {pair_index} has pixel shape "
                f"{tuple(pixel_values.shape)}, expected {expected_shape}"
            )
        if grid_rows != expected_grid:
            raise ValueError(
                f"Pair {pair_index} has grid {grid_rows}, "
                f"expected {expected_grid}"
            )
        temporal = pixel_values.reshape(
            expected_shape[0],
            VISION_INPUT_CHANNELS,
            VISION_TEMPORAL_PATCH_SIZE,
            VISION_PATCH_SIZE,
            VISION_PATCH_SIZE,
        )
        if (
            torch.equal(temporal[:, :, 0], temporal[:, :, 1])
            and not allow_static_temporal_patch
        ):
            raise ValueError(
                f"Pair {pair_index} has no temporal difference after packing"
            )
        calibration_data.append(pixel_values.numpy())
    return calibration_data


def load_vision_calibration_data(
    *,
    hf_repo: str,
    num_samples: int,
    image_height: int,
    image_width: int,
    paired_manifest: Path | None,
) -> list[Any]:
    if paired_manifest is not None:
        return load_paired_frame_calibration_data(
            hf_repo=hf_repo,
            manifest_path=paired_manifest,
            num_samples=num_samples,
            image_height=image_height,
            image_width=image_width,
        )
    return load_imagenette_calibration_data(
        hf_repo=hf_repo,
        num_samples=num_samples,
        image_height=image_height,
        image_width=image_width,
    )
