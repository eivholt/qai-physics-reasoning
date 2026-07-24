"""Static vision-profile configuration for Cosmos-Reason2 QAI exports."""

from __future__ import annotations

from dataclasses import dataclass

PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2


@dataclass(frozen=True)
class CosmosVisionProfile:
    """One fixed-resolution, two-frame Qwen3-VL vision graph."""

    image_height: int
    image_width: int
    patch_size: int = PATCH_SIZE
    spatial_merge_size: int = SPATIAL_MERGE_SIZE
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE

    def __post_init__(self) -> None:
        for name in (
            "image_height",
            "image_width",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.temporal_patch_size != 2:
            raise ValueError(
                "Cosmos paired-frame deployment requires "
                "temporal_patch_size=2"
            )
        divisor = self.patch_size * self.spatial_merge_size
        if self.image_height % divisor or self.image_width % divisor:
            raise ValueError(
                f"Image size {self.image_height}x{self.image_width} must be "
                f"divisible by patch_size * spatial_merge_size ({divisor})"
            )

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        return (
            1,
            self.image_height // self.patch_size,
            self.image_width // self.patch_size,
        )

    @property
    def pixel_shape(self) -> tuple[int, int]:
        patches = self.grid_thw[1] * self.grid_thw[2]
        patch_values = (
            3
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        return (patches, patch_values)

    @property
    def visual_tokens(self) -> int:
        return (
            self.grid_thw[1]
            * self.grid_thw[2]
            // (self.spatial_merge_size**2)
        )


DEFAULT_VISION_PROFILE = CosmosVisionProfile(512, 512)
WAREHOUSE_ASPECT_PROFILE = CosmosVisionProfile(224, 384)


def configure_cosmos_vision_profile(
    image_height: int,
    image_width: int,
) -> CosmosVisionProfile:
    """Apply one profile to vision compilation and release metadata.

    Decoder partition capacities intentionally remain unchanged. GenieX
    slices the actual vision features into text-prefill chunks and zero-fills
    the unused DeepStack rows, so the proven text contexts can be reused when
    a smaller vision graph is selected.
    """

    profile = CosmosVisionProfile(image_height, image_width)
    from .model import (
        Cosmos_Reason2_2B_Collection,
        Cosmos_Reason2_2B_VisionEncoder,
    )

    Cosmos_Reason2_2B_VisionEncoder.DEFAULT_IMAGE_SIZE = (
        profile.image_height,
        profile.image_width,
    )
    Cosmos_Reason2_2B_VisionEncoder.default_image_height = (
        profile.image_height
    )
    Cosmos_Reason2_2B_VisionEncoder.default_image_width = (
        profile.image_width
    )
    Cosmos_Reason2_2B_Collection.default_image_height = (
        profile.image_height
    )
    Cosmos_Reason2_2B_Collection.default_image_width = profile.image_width
    return profile
