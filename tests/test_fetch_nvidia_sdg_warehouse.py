from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.fetch_nvidia_sdg_warehouse import (
    AssetValidationError,
    DEFAULT_MANIFEST,
    extract_rgb_frames,
    fetch_asset,
    load_manifest,
)


class FetchNvidiaSdgWarehouseTests(unittest.TestCase):
    def test_tracked_manifest_pins_source_and_ground_truth_cases(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)

        self.assertEqual(
            manifest["source"]["repo_id"],
            "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes",
        )
        self.assertEqual(manifest["source"]["license"], "OpenMDW-1.1")
        self.assertEqual(len(manifest["source"]["revision"]), 40)
        self.assertEqual(
            manifest["preview_layout"]["panels_left_to_right"][0], "rgb"
        )
        self.assertEqual(manifest["preview_layout"]["rgb_crop_xyxy"], [0, 0, 384, 216])

        labels = {
            case["expected_label"] for case in manifest["classification_cases"]
        }
        self.assertEqual(
            labels,
            {
                "forklift_human_near_miss",
                "forklift_shelf_collision",
                "warehouse_fire_and_evacuation",
                "routine_box_pickup",
            },
        )
        negative = next(
            case
            for case in manifest["classification_cases"]
            if case["expected_label"] == "routine_box_pickup"
        )
        self.assertFalse(negative["is_safety_incident"])
        self.assertTrue(negative["negative_control"])

        predictions = {
            case["id"]: case for case in manifest["prediction_cases"]
        }
        self.assertEqual(
            predictions["predict_near_miss"]["rgb_frame_indices"], [30, 45]
        )
        self.assertEqual(
            predictions["predict_shelf_collision"]["rgb_frame_indices"], [66, 88]
        )

    def test_fetch_is_anonymous_and_validates_hash(self) -> None:
        payload = b"official preview fixture"
        asset = {
            "id": "fixture",
            "relative_path": "assets/fixture.bin",
            "source_url": "https://example.invalid/fixture.bin",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        seen_requests = []

        def fake_urlopen(request):
            seen_requests.append(request)
            return io.BytesIO(payload)

        with tempfile.TemporaryDirectory() as temporary:
            path, downloaded = fetch_asset(
                asset, Path(temporary), urlopen=fake_urlopen
            )
            self.assertTrue(downloaded)
            self.assertEqual(path.read_bytes(), payload)

            _, downloaded = fetch_asset(
                asset, Path(temporary), urlopen=fake_urlopen
            )
            self.assertFalse(downloaded)

        self.assertEqual(len(seen_requests), 1)
        request = seen_requests[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.get_method(), "GET")

    def test_bad_forced_download_does_not_replace_existing_asset(self) -> None:
        expected = b"expected bytes"
        asset = {
            "id": "fixture",
            "relative_path": "assets/fixture.bin",
            "source_url": "https://example.invalid/fixture.bin",
            "size_bytes": len(expected),
            "sha256": hashlib.sha256(expected).hexdigest(),
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "assets" / "fixture.bin"
            destination.parent.mkdir()
            destination.write_bytes(expected)

            with self.assertRaises(AssetValidationError):
                fetch_asset(
                    asset,
                    root,
                    force=True,
                    urlopen=lambda _: io.BytesIO(b"corrupt"),
                )
            self.assertEqual(destination.read_bytes(), expected)
            self.assertFalse(destination.with_name("fixture.bin.part").exists())

    def test_extracts_only_leftmost_rgb_panel(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "assets" / "fixture.gif"
            source.parent.mkdir()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            frames = []
            for color in colors:
                frame = Image.new("RGB", (20, 3), (0, 0, 0))
                frame.paste(color, (0, 0, 4, 3))
                frames.append(frame)
            frames[0].save(
                source,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            asset = {
                "id": "fixture",
                "kind": "preview_clip",
                "relative_path": "assets/fixture.gif",
                "size_bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "frame_count": 3,
            }
            manifest = {
                "preview_layout": {
                    "width": 20,
                    "height": 3,
                    "rgb_crop_xyxy": [0, 0, 4, 3],
                },
                "assets": [asset],
            }
            case = {
                "id": "predict_fixture",
                "clip_asset_id": "fixture",
                "rgb_frame_indices": [0, 2],
            }

            written = extract_rgb_frames(manifest, root, case)
            self.assertEqual([path.name for path in written], [
                "frame_0000.png",
                "frame_0002.png",
            ])
            with Image.open(written[0]) as extracted:
                self.assertEqual(extracted.size, (4, 3))
                self.assertEqual(extracted.convert("RGB").getpixel((1, 1)), colors[0])
            with Image.open(written[1]) as extracted:
                self.assertEqual(extracted.convert("RGB").getpixel((1, 1)), colors[2])

    def test_rejects_case_id_path_traversal(self) -> None:
        manifest = {
            "preview_layout": {
                "width": 20,
                "height": 3,
                "rgb_crop_xyxy": [0, 0, 4, 3],
            },
            "assets": [
                {
                    "id": "fixture",
                    "kind": "preview_clip",
                    "relative_path": "assets/fixture.gif",
                    "size_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "frame_count": 1,
                }
            ],
        }
        case = {
            "id": "../escape",
            "clip_asset_id": "fixture",
            "rgb_frame_indices": [0],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "assets" / "fixture.gif"
            source.parent.mkdir()
            source.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Unsafe prediction case id"):
                extract_rgb_frames(manifest, root, case)


if __name__ == "__main__":
    unittest.main()
