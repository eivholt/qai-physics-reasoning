from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from qai_hub_models.models.cosmos_reason2_2b.vision_profile import (
    CosmosVisionProfile,
    WAREHOUSE_ASPECT_PROFILE,
)
from scripts.vision_profile import VisionProfile


def _metadata(height: int, width: int) -> dict[str, object]:
    patch_size = 16
    merge_size = 2
    patches = (height // patch_size) * (width // patch_size)
    visual_tokens = patches // (merge_size**2)
    return {
        "model_files": {
            "vision_encoder.bin": {
                "inputs": {
                    "pixel_values": {"shape": [patches, 1536]},
                    "position_ids_cos": {"shape": [patches, 32]},
                    "position_ids_sin": {"shape": [patches, 32]},
                    "window_attention_mask": {
                        "shape": [1, patches, patches]
                    },
                    "full_attention_mask": {
                        "shape": [1, patches, patches]
                    },
                },
                "outputs": {
                    name: {"shape": [visual_tokens, 2048]}
                    for name in (
                        "image_features",
                        "deepstack_visual_embeds_0",
                        "deepstack_visual_embeds_1",
                        "deepstack_visual_embeds_2",
                    )
                },
            }
        },
        "genie": {
            "vision_preprocessing": {
                "image_width": width,
                "image_height": height,
                "patch_size": patch_size,
                "temporal_patch_size": 2,
                "spatial_merge_size": merge_size,
            }
        },
    }


class VisionProfileTests(unittest.TestCase):
    def test_derives_aspect_preserving_graph_contract(self) -> None:
        profile = VisionProfile.from_metadata(_metadata(224, 384))

        self.assertEqual(profile.image_size, (384, 224))
        self.assertEqual(profile.grid_thw, (1, 14, 24))
        self.assertEqual(profile.pixel_shape, (336, 1536))
        self.assertEqual(profile.visual_tokens, 84)
        self.assertEqual(profile.visual_shape, (84, 2048))
        self.assertEqual(profile.pixel_bytes, 336 * 1536 * 4)

    def test_validates_profile_specific_config_and_ancillaries(self) -> None:
        profile = VisionProfile.from_metadata(_metadata(224, 384))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "img-enc-htp.json"
            config.write_text(
                json.dumps(
                    {
                        "image-encoder": {
                            "engine": {
                                "model": {
                                    "vision-param": {
                                        "height": 14,
                                        "width": 24,
                                    }
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            samples = root / "sample_inputs"
            samples.mkdir()
            for filename, shape in profile.ancillary_shapes.items():
                with (samples / filename).open("wb") as handle:
                    handle.truncate(math.prod(shape) * 4)

            profile.validate_img_encoder_config(config)
            profile.validate_ancillary_files(samples)

    def test_rejects_graph_shape_that_disagrees_with_preprocessing(self) -> None:
        metadata = _metadata(224, 384)
        metadata["model_files"]["vision_encoder.bin"]["inputs"][
            "pixel_values"
        ]["shape"] = [1024, 1536]

        with self.assertRaisesRegex(ValueError, "expected \\(336, 1536\\)"):
            VisionProfile.from_metadata(metadata)

    def test_cosmos_export_profile_has_expected_dimensions(self) -> None:
        self.assertEqual(WAREHOUSE_ASPECT_PROFILE.grid_thw, (1, 14, 24))
        self.assertEqual(WAREHOUSE_ASPECT_PROFILE.pixel_shape, (336, 1536))
        self.assertEqual(WAREHOUSE_ASPECT_PROFILE.visual_tokens, 84)
        with self.assertRaisesRegex(ValueError, "divisible"):
            CosmosVisionProfile(225, 384)


if __name__ == "__main__":
    unittest.main()
