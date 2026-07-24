#!/usr/bin/env python3
"""Derive and validate a static Qwen3-VL vision profile from bundle metadata.

QAI Hub exports the vision tensor shapes and the preprocessing dimensions in
``metadata.json``.  Treating those declarations as the source of truth keeps
the input preparers and GenieX packaging code independent of one hard-coded
resolution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FLOAT32_BYTES = 4
RGB_CHANNELS = 3
VISION_CONTEXT = "vision_encoder.bin"


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _shape(
    tensors: dict[str, Any], name: str, label: str
) -> tuple[int, ...]:
    try:
        values = tensors[name]["shape"]
        shape = tuple(int(value) for value in values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has no valid shape for {name}") from exc
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"{label} shape for {name} is invalid: {shape}")
    return shape


@dataclass(frozen=True)
class VisionProfile:
    """Static tensor contract for one two-frame temporal patch."""

    image_width: int
    image_height: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    hidden_size: int
    rope_dim: int

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        return (
            1,
            self.image_height // self.patch_size,
            self.image_width // self.patch_size,
        )

    @property
    def patch_count(self) -> int:
        return math.prod(self.grid_thw)

    @property
    def pixel_columns(self) -> int:
        return (
            RGB_CHANNELS
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

    @property
    def pixel_shape(self) -> tuple[int, int]:
        return (self.patch_count, self.pixel_columns)

    @property
    def pixel_bytes(self) -> int:
        return math.prod(self.pixel_shape) * FLOAT32_BYTES

    @property
    def visual_tokens(self) -> int:
        return self.patch_count // (self.spatial_merge_size**2)

    @property
    def visual_shape(self) -> tuple[int, int]:
        return (self.visual_tokens, self.hidden_size)

    @property
    def image_size(self) -> tuple[int, int]:
        """Return the processor size in Pillow's ``(width, height)`` order."""

        return (self.image_width, self.image_height)

    @property
    def ancillary_shapes(self) -> dict[str, tuple[int, ...]]:
        patches = self.patch_count
        return {
            "position_ids_cos.raw": (patches, self.rope_dim),
            "position_ids_sin.raw": (patches, self.rope_dim),
            "window_attention_mask.raw": (1, patches, patches),
            "full_attention_mask.raw": (1, patches, patches),
        }

    def as_manifest_contract(self) -> dict[str, Any]:
        return {
            "frames_per_pair": self.temporal_patch_size,
            "image_size": list(self.image_size),
            "pixel_values_shape": list(self.pixel_shape),
            "grid_thw": list(self.grid_thw),
            "image_features_shape": list(self.visual_shape),
            "visual_tokens_per_pair": self.visual_tokens,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "VisionProfile":
        """Validate metadata and return its derived vision contract."""

        try:
            preprocessing = metadata["genie"]["vision_preprocessing"]
            vision = metadata["model_files"][VISION_CONTEXT]
            inputs = vision["inputs"]
            outputs = vision["outputs"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "metadata.json has no complete vision preprocessing/context"
            ) from exc
        if not all(
            isinstance(value, dict)
            for value in (preprocessing, inputs, outputs)
        ):
            raise ValueError("Vision preprocessing and tensor maps must be objects")

        width = _positive_int(
            preprocessing.get("image_width"), "vision image_width"
        )
        height = _positive_int(
            preprocessing.get("image_height"), "vision image_height"
        )
        patch = _positive_int(
            preprocessing.get("patch_size"), "vision patch_size"
        )
        temporal = _positive_int(
            preprocessing.get("temporal_patch_size"),
            "vision temporal_patch_size",
        )
        merge = _positive_int(
            preprocessing.get("spatial_merge_size"),
            "vision spatial_merge_size",
        )
        if temporal != 2:
            raise ValueError(
                "Paired-frame input requires temporal_patch_size=2, got "
                f"{temporal}"
            )
        if width % patch or height % patch:
            raise ValueError(
                f"Vision size {height}x{width} is not divisible by patch "
                f"size {patch}"
            )
        grid_height = height // patch
        grid_width = width // patch
        if grid_height % merge or grid_width % merge:
            raise ValueError(
                f"Vision grid {grid_height}x{grid_width} is not divisible by "
                f"spatial merge size {merge}"
            )

        position_cos = _shape(inputs, "position_ids_cos", "vision inputs")
        position_sin = _shape(inputs, "position_ids_sin", "vision inputs")
        if len(position_cos) != 2 or position_sin != position_cos:
            raise ValueError(
                "Vision position_ids_cos/sin must have one matching "
                f"[patches, rope_dim] shape, got {position_cos}/{position_sin}"
            )
        image_features = _shape(
            outputs, "image_features", "vision outputs"
        )
        if len(image_features) != 2:
            raise ValueError(
                "Vision image_features must have [tokens, hidden] shape"
            )
        profile = cls(
            image_width=width,
            image_height=height,
            patch_size=patch,
            temporal_patch_size=temporal,
            spatial_merge_size=merge,
            hidden_size=image_features[1],
            rope_dim=position_cos[1],
        )

        expected_inputs = {
            "pixel_values": profile.pixel_shape,
            "position_ids_cos": profile.ancillary_shapes[
                "position_ids_cos.raw"
            ],
            "position_ids_sin": profile.ancillary_shapes[
                "position_ids_sin.raw"
            ],
            "window_attention_mask": profile.ancillary_shapes[
                "window_attention_mask.raw"
            ],
            "full_attention_mask": profile.ancillary_shapes[
                "full_attention_mask.raw"
            ],
        }
        for name, expected in expected_inputs.items():
            actual = _shape(inputs, name, "vision inputs")
            if actual != expected:
                raise ValueError(
                    f"Vision input {name} has shape {actual}; expected "
                    f"{expected} for {height}x{width}"
                )

        if image_features != profile.visual_shape:
            raise ValueError(
                f"Vision image_features has shape {image_features}; expected "
                f"{profile.visual_shape} for {height}x{width}"
            )
        for name, tensor in outputs.items():
            if not name.startswith("deepstack_visual_embeds_"):
                continue
            actual = _shape(outputs, name, "vision outputs")
            if actual != profile.visual_shape:
                raise ValueError(
                    f"Vision output {name} has shape {actual}; expected "
                    f"{profile.visual_shape}"
                )
        return profile

    def validate_img_encoder_config(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            params = payload["image-encoder"]["engine"]["model"][
                "vision-param"
            ]
            actual = (
                int(params["height"]),
                int(params["width"]),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot read the vision grid from {path}"
            ) from exc
        expected = self.grid_thw[1:]
        if actual != expected:
            raise ValueError(
                f"{path.name} declares vision grid {actual}; expected "
                f"{expected}"
            )

    def validate_ancillary_files(self, sample_inputs: Path) -> None:
        for filename, shape in self.ancillary_shapes.items():
            path = sample_inputs / filename
            if not path.is_file():
                raise ValueError(f"Bundle is missing ancillary tensor: {path}")
            expected_bytes = math.prod(shape) * FLOAT32_BYTES
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"Ancillary tensor {filename} has {actual_bytes} bytes; "
                    f"expected {expected_bytes}"
                )


def load_vision_profile(bundle: Path) -> tuple[dict[str, Any], VisionProfile]:
    """Read ``metadata.json`` and validate its profile-specific assets."""

    metadata_path = bundle / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read metadata.json: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json must contain an object: {metadata_path}")
    profile = VisionProfile.from_metadata(metadata)
    profile.validate_img_encoder_config(bundle / "img-enc-htp.json")
    profile.validate_ancillary_files(bundle / "sample_inputs")
    return metadata, profile
