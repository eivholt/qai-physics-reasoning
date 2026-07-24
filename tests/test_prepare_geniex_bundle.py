from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_geniex_bundle import (
    DEEPSTACK_INPUTS,
    DEFAULT_MODEL_ID,
    MARKER_FILENAME,
    PART1_CONTEXT,
    QAIRT245_COMPAT_MARKER,
    REQUIRED_CONTEXTS,
    W4_FP16_MARKER,
    prepare_bundle,
)


def _write_fixture(root: Path, *, model_id: str = "cosmos_reason2_2b") -> None:
    model_files = {
        name: {
            "inputs": {
                "inputs_embeds": {
                    "shape": [1, 1, 2048],
                    "dtype": "float32",
                }
            },
            "outputs": {"hidden": {"shape": [1, 1, 2048]}},
        }
        for name in REQUIRED_CONTEXTS
    }
    model_files["vision_encoder.bin"]["outputs"] = {
        "image_features": {"shape": [256, 2048], "dtype": "uint16"},
        "deepstack_visual_embeds_0": {
            "shape": [256, 2048],
            "dtype": "uint16",
        },
        "deepstack_visual_embeds_1": {
            "shape": [256, 2048],
            "dtype": "uint16",
        },
        "deepstack_visual_embeds_2": {
            "shape": [256, 2048],
            "dtype": "uint16",
        },
    }
    metadata = {
        "model_id": model_id,
        "model_files": model_files,
        "supplementary_files": {
            QAIRT245_COMPAT_MARKER: "legacy compatibility provenance",
            W4_FP16_MARKER: "W4/FP16 provenance",
        },
        "genie": {
            "vision_preprocessing": {
                "image_width": 512,
                "image_height": 512,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
            }
        },
    }
    config = {
        "dialog": {
            "engine": {
                "model": {
                    "positional-encoding": {
                        "rope-scaling": {
                            "rope-type": "qwen3vl-mrope",
                            "mrope-section": [24, 20, 20],
                        }
                    }
                }
            }
        }
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "genie_config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in REQUIRED_CONTEXTS:
        (root / name).write_bytes(name.encode())
    (root / QAIRT245_COMPAT_MARKER).write_text("compat", encoding="utf-8")
    (root / W4_FP16_MARKER).write_text("source marker", encoding="utf-8")


def _write_part1_replacement(
    root: Path,
    *,
    omit_input: str | None = None,
    auxiliary_first: bool = False,
    include_marker: bool = True,
) -> None:
    inputs = {
        "inputs_embeds": {"shape": [1, 1, 2048], "dtype": "float32"},
        "visual_pos_masks": {"shape": [1, 1], "dtype": "bool"},
        "deepstack_visual_embeds_0": {
            "shape": [256, 2048],
            "dtype": "float32",
        },
        "deepstack_visual_embeds_1": {
            "shape": [256, 2048],
            "dtype": "float32",
        },
        "deepstack_visual_embeds_2": {
            "shape": [256, 2048],
            "dtype": "float32",
        },
    }
    if omit_input is not None:
        inputs.pop(omit_input)
    if auxiliary_first and "visual_pos_masks" in inputs:
        inputs = {
            "visual_pos_masks": inputs["visual_pos_masks"],
            **{
                name: value
                for name, value in inputs.items()
                if name != "visual_pos_masks"
            },
        }
    metadata = {
        "model_files": {
            PART1_CONTEXT: {
                "inputs": inputs,
                "outputs": {"hidden": {"shape": [1, 1, 2048]}},
            }
        },
        "supplementary_files": {
            W4_FP16_MARKER: "replacement W4/FP16 provenance",
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / PART1_CONTEXT).write_bytes(b"full-deepstack-w4-fp16-part1")
    if include_marker:
        (root / W4_FP16_MARKER).write_text(
            "replacement marker", encoding="utf-8"
        )


class PrepareGenieXBundleTests(unittest.TestCase):
    def test_prepares_independent_metadata_with_dispatch_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            _write_fixture(source)
            source_bytes = (source / "metadata.json").read_bytes()

            prepare_bundle(source, destination, hardlink=True)

            self.assertEqual((source / "metadata.json").read_bytes(), source_bytes)
            patched = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(patched["model_id"], DEFAULT_MODEL_ID)
            marker = json.loads((destination / MARKER_FILENAME).read_text())
            self.assertEqual(marker["original_model_id"], "cosmos_reason2_2b")
            self.assertEqual(
                marker["source_metadata_sha256"],
                hashlib.sha256(source_bytes).hexdigest(),
            )

    def test_rejects_non_qwen3_vl_dispatch_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _write_fixture(source)
            with self.assertRaisesRegex(ValueError, "must start"):
                prepare_bundle(
                    source,
                    root / "destination",
                    model_id="cosmos_reason2_2b",
                )

    def test_rejects_incompatible_mrope_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            _write_fixture(source)
            config_path = source / "genie_config.json"
            config = json.loads(config_path.read_text())
            config["dialog"]["engine"]["model"]["positional-encoding"][
                "rope-scaling"
            ]["mrope-section"] = [16, 24, 24]
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mrope-section"):
                prepare_bundle(source, destination)
            self.assertFalse(destination.exists())

    def test_replaces_only_part1_and_removes_legacy_compat_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source)
            _write_part1_replacement(replacement)

            prepare_bundle(
                source,
                destination,
                hardlink=True,
                part1_replacement_bundle=replacement,
            )

            self.assertEqual(
                (destination / PART1_CONTEXT).read_bytes(),
                b"full-deepstack-w4-fp16-part1",
            )
            for name in REQUIRED_CONTEXTS - {PART1_CONTEXT}:
                self.assertEqual(
                    (destination / name).read_bytes(),
                    (source / name).read_bytes(),
                )
            self.assertFalse((destination / QAIRT245_COMPAT_MARKER).exists())
            self.assertEqual(
                (destination / W4_FP16_MARKER).read_text(),
                "replacement marker",
            )

            metadata = json.loads((destination / "metadata.json").read_text())
            part1_inputs = metadata["model_files"][PART1_CONTEXT]["inputs"]
            self.assertEqual(
                DEEPSTACK_INPUTS & set(part1_inputs),
                DEEPSTACK_INPUTS,
            )
            self.assertNotIn(
                QAIRT245_COMPAT_MARKER,
                metadata["supplementary_files"],
            )
            self.assertEqual(
                metadata["supplementary_files"][W4_FP16_MARKER],
                "replacement W4/FP16 provenance",
            )
            marker = json.loads((destination / MARKER_FILENAME).read_text())
            self.assertEqual(
                marker["part1_replacement"]["context_sha256"],
                hashlib.sha256(b"full-deepstack-w4-fp16-part1").hexdigest(),
            )
            self.assertEqual(marker["source_bundle_name"], source.name)
            self.assertEqual(
                marker["part1_replacement"]["source_bundle_name"],
                replacement.name,
            )
            self.assertNotIn(str(root), (destination / MARKER_FILENAME).read_text())
            self.assertEqual(
                (source / PART1_CONTEXT).read_bytes(),
                PART1_CONTEXT.encode(),
            )
            self.assertTrue((source / QAIRT245_COMPAT_MARKER).is_file())

    def test_rejects_incomplete_or_unsafe_part1_before_copying(self) -> None:
        for label, kwargs, message in (
            (
                "missing",
                {"omit_input": "deepstack_visual_embeds_2"},
                "missing",
            ),
            (
                "reordered",
                {"auxiliary_first": True},
                "before inputs_embeds",
            ),
        ):
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = root / "source"
                replacement = root / "replacement"
                destination = root / "destination"
                source.mkdir()
                replacement.mkdir()
                _write_fixture(source)
                _write_part1_replacement(replacement, **kwargs)

                with self.assertRaisesRegex(ValueError, message):
                    prepare_bundle(
                        source,
                        destination,
                        part1_replacement_bundle=replacement,
                    )
                self.assertFalse(destination.exists())

    def test_rejects_replacement_without_w4_fp16_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source)
            _write_part1_replacement(replacement, include_marker=False)

            with self.assertRaisesRegex(ValueError, W4_FP16_MARKER):
                prepare_bundle(
                    source,
                    destination,
                    part1_replacement_bundle=replacement,
                )
            self.assertFalse(destination.exists())

    def test_rejects_incompatible_base_part1_contract_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source)
            _write_part1_replacement(replacement)
            metadata_path = replacement / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["model_files"][PART1_CONTEXT]["inputs"]["inputs_embeds"][
                "shape"
            ] = [1, 1, 4096]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "base input contract"):
                prepare_bundle(
                    source,
                    destination,
                    part1_replacement_bundle=replacement,
                )
            self.assertFalse(destination.exists())

    def test_rejects_source_vision_without_matching_deepstack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source)
            _write_part1_replacement(replacement)
            metadata_path = source / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["model_files"]["vision_encoder.bin"]["outputs"].pop(
                "deepstack_visual_embeds_2"
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "missing output deepstack_visual_embeds_2"
            ):
                prepare_bundle(
                    source,
                    destination,
                    part1_replacement_bundle=replacement,
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
