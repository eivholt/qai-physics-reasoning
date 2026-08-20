from __future__ import annotations

import hashlib
import json
import math
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
    TEXT_W8_MARKER,
    W4_FP16_MARKER,
    prepare_bundle,
)


def _write_sparse(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size)


def _write_fixture(
    root: Path,
    *,
    model_id: str = "cosmos_reason2_2b",
    full_deepstack: bool = False,
) -> None:
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
    if full_deepstack:
        model_files[PART1_CONTEXT]["inputs"].update(
            {
                "visual_pos_masks": {
                    "shape": [1, 128],
                    "dtype": "bool",
                },
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
        )
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
    (root / MARKER_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "part1_replacement": None,
                "vision_replacement": {
                    "context_sha256": "fixture-vision",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_part1_replacement(
    root: Path,
    *,
    omit_input: str | None = None,
    auxiliary_first: bool = False,
    include_marker: bool = True,
    include_text_w8_marker: bool = False,
    context_length: int = 512,
    visual_mask_shape: list[int] | None = None,
    visual_mask_dtype: str = "bool",
) -> None:
    if visual_mask_shape is None:
        visual_mask_shape = [1, 1]
    inputs = {
        "inputs_embeds": {"shape": [1, 1, 2048], "dtype": "float32"},
        "visual_pos_masks": {
            "shape": visual_mask_shape,
            "dtype": visual_mask_dtype,
        },
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
    if include_text_w8_marker:
        metadata["supplementary_files"][TEXT_W8_MARKER] = (
            "replacement part-1 W8 provenance"
        )
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / PART1_CONTEXT).write_bytes(b"full-deepstack-w4-fp16-part1")
    if include_marker:
        (root / W4_FP16_MARKER).write_text(
            "replacement marker", encoding="utf-8"
        )
    if include_text_w8_marker:
        (root / TEXT_W8_MARKER).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selected_parts": ["part1_of_4"],
                    "export_contract": {
                        "activation_precision": "FP16",
                        "context_length": context_length,
                        "sequence_lengths": [128, 1],
                        "deepstack_inputs": [
                            "visual_pos_masks",
                            "deepstack_visual_embeds_0",
                            "deepstack_visual_embeds_1",
                            "deepstack_visual_embeds_2",
                        ],
                        "external_tensor_precision": (
                            "FP32, except visual_pos_masks BOOL"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )


def _write_vision_replacement(
    root: Path,
    *,
    image_height: int = 224,
    image_width: int = 384,
) -> None:
    patch_size = 16
    merge_size = 2
    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    patches = grid_height * grid_width
    visual_tokens = patches // (merge_size**2)
    inputs = {
        "pixel_values": {
            "shape": [patches, 1536],
            "dtype": "float32",
        },
        "position_ids_cos": {
            "shape": [patches, 32],
            "dtype": "float32",
        },
        "position_ids_sin": {
            "shape": [patches, 32],
            "dtype": "float32",
        },
        "window_attention_mask": {
            "shape": [1, patches, patches],
            "dtype": "float32",
        },
        "full_attention_mask": {
            "shape": [1, patches, patches],
            "dtype": "float32",
        },
    }
    outputs = {
        name: {
            "shape": [visual_tokens, 2048],
            "dtype": "uint16",
        }
        for name in (
            "image_features",
            "deepstack_visual_embeds_0",
            "deepstack_visual_embeds_1",
            "deepstack_visual_embeds_2",
        )
    }
    metadata = {
        "model_id": "cosmos_reason2_2b_vision_profile",
        "model_files": {
            "vision_encoder.bin": {
                "inputs": inputs,
                "outputs": outputs,
            }
        },
        "supplementary_files": {
            "img-enc-htp.json": "aspect-profile image encoder config",
        },
        "genie": {
            "vision_preprocessing": {
                "image_width": image_width,
                "image_height": image_height,
                "patch_size": patch_size,
                "temporal_patch_size": 2,
                "spatial_merge_size": merge_size,
            }
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "vision_encoder.bin").write_bytes(b"aspect-vision-context")
    (root / "img-enc-htp.json").write_text(
        json.dumps(
            {
                "image-encoder": {
                    "engine": {
                        "model": {
                            "vision-param": {
                                "height": grid_height,
                                "width": grid_width,
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
    sample_shapes = {
        "pixel_values.raw": (patches, 1536),
        "position_ids_cos.raw": (patches, 32),
        "position_ids_sin.raw": (patches, 32),
        "window_attention_mask.raw": (1, patches, patches),
        "full_attention_mask.raw": (1, patches, patches),
    }
    for filename, shape in sample_shapes.items():
        _write_sparse(samples / filename, math.prod(shape) * 4)


class PrepareGenieXBundleTests(unittest.TestCase):
    def test_prepares_independent_metadata_with_dispatch_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            _write_fixture(source)
            source_bytes = (source / "metadata.json").read_bytes()
            source_marker_bytes = (source / MARKER_FILENAME).read_bytes()

            prepare_bundle(source, destination, hardlink=True)

            self.assertEqual((source / "metadata.json").read_bytes(), source_bytes)
            self.assertEqual(
                (source / MARKER_FILENAME).read_bytes(),
                source_marker_bytes,
            )
            patched = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(patched["model_id"], DEFAULT_MODEL_ID)
            marker = json.loads((destination / MARKER_FILENAME).read_text())
            self.assertEqual(marker["original_model_id"], "cosmos_reason2_2b")
            self.assertEqual(
                marker["source_metadata_sha256"],
                hashlib.sha256(source_bytes).hexdigest(),
            )
            self.assertEqual(marker["schema_version"], 3)
            self.assertEqual(
                marker["source_compatibility_marker"]["sha256"],
                hashlib.sha256(source_marker_bytes).hexdigest(),
            )
            self.assertEqual(
                set(marker["final_context_sha256"]),
                REQUIRED_CONTEXTS,
            )
            for filename in REQUIRED_CONTEXTS:
                self.assertEqual(
                    marker["final_context_sha256"][filename],
                    hashlib.sha256(
                        (destination / filename).read_bytes()
                    ).hexdigest(),
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

    def test_preserves_w8_part1_provenance_and_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source, full_deepstack=True)
            _write_part1_replacement(
                replacement,
                include_text_w8_marker=True,
                visual_mask_shape=[1, 128],
                visual_mask_dtype="bool",
            )
            marker_bytes = (replacement / TEXT_W8_MARKER).read_bytes()

            prepare_bundle(
                source,
                destination,
                part1_replacement_bundle=replacement,
            )

            self.assertEqual(
                (destination / TEXT_W8_MARKER).read_bytes(),
                marker_bytes,
            )
            metadata = json.loads(
                (destination / "metadata.json").read_text()
            )
            self.assertEqual(
                metadata["supplementary_files"][TEXT_W8_MARKER],
                "replacement part-1 W8 provenance",
            )
            provenance = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )["part1_replacement"]
            self.assertEqual(
                provenance["deepstack_visual_shape"],
                [256, 2048],
            )
            self.assertEqual(
                provenance["precision_provenance"][TEXT_W8_MARKER][
                    "sha256"
                ],
                hashlib.sha256(marker_bytes).hexdigest(),
            )

    def test_accepts_w8_part1_with_larger_context_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source, full_deepstack=True)
            _write_part1_replacement(
                replacement,
                include_text_w8_marker=True,
                context_length=1024,
                visual_mask_shape=[1, 128],
                visual_mask_dtype="bool",
            )

            prepare_bundle(
                source,
                destination,
                part1_replacement_bundle=replacement,
            )

            marker = json.loads(
                (destination / TEXT_W8_MARKER).read_text()
            )
            self.assertEqual(
                marker["export_contract"]["context_length"], 1024
            )

    def test_rejects_w8_marker_without_metadata_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source)
            _write_part1_replacement(
                replacement,
                include_text_w8_marker=True,
            )
            metadata_path = replacement / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["supplementary_files"].pop(TEXT_W8_MARKER)
            metadata_path.write_text(json.dumps(metadata))

            with self.assertRaisesRegex(
                ValueError, "both be present or both be absent"
            ):
                prepare_bundle(
                    source,
                    destination,
                    part1_replacement_bundle=replacement,
                )
            self.assertFalse(destination.exists())

    def test_rejects_part1_capacity_or_external_dtype_drift(self) -> None:
        mutations = (
            (
                "capacity",
                "deepstack_visual_embeds_0",
                "shape",
                [84, 2048],
                "capacity/external contract",
            ),
            (
                "dtype",
                "deepstack_visual_embeds_0",
                "dtype",
                "uint16",
                "external dtype must be float32",
            ),
        )
        for label, tensor_name, field, value, message in mutations:
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
                _write_fixture(source, full_deepstack=True)
                _write_part1_replacement(
                    replacement,
                    visual_mask_shape=[1, 128],
                    visual_mask_dtype="bool",
                )
                metadata_path = replacement / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                part1_inputs = metadata["model_files"][PART1_CONTEXT][
                    "inputs"
                ]
                if field == "shape":
                    for deepstack_name in (
                        "deepstack_visual_embeds_0",
                        "deepstack_visual_embeds_1",
                        "deepstack_visual_embeds_2",
                    ):
                        part1_inputs[deepstack_name][field] = value
                else:
                    part1_inputs[tensor_name][field] = value
                metadata_path.write_text(json.dumps(metadata))

                with self.assertRaisesRegex(ValueError, message):
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

            with self.assertRaisesRegex(ValueError, "inputs_embeds shape"):
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

    def test_replaces_only_vision_profile_and_reuses_all_text_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source, full_deepstack=True)
            _write_vision_replacement(replacement)
            stale = source / "video_inputs" / "old_square"
            stale.mkdir(parents=True)
            (stale / "bound-to-square.txt").write_text(
                "stale", encoding="utf-8"
            )
            source_metadata = (source / "metadata.json").read_bytes()
            text_contexts = {
                name: (source / name).read_bytes()
                for name in REQUIRED_CONTEXTS - {"vision_encoder.bin"}
            }

            prepare_bundle(
                source,
                destination,
                hardlink=True,
                vision_replacement_bundle=replacement,
            )

            self.assertEqual(
                (destination / "vision_encoder.bin").read_bytes(),
                b"aspect-vision-context",
            )
            for name, expected in text_contexts.items():
                self.assertEqual((destination / name).read_bytes(), expected)
            self.assertFalse((destination / "video_inputs").exists())
            self.assertTrue((source / "video_inputs" / "old_square").is_dir())
            self.assertEqual((source / "metadata.json").read_bytes(), source_metadata)

            metadata = json.loads(
                (destination / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["genie"]["vision_preprocessing"]["image_height"],
                224,
            )
            self.assertEqual(
                metadata["genie"]["vision_preprocessing"]["image_width"],
                384,
            )
            self.assertEqual(
                metadata["model_files"]["vision_encoder.bin"]["outputs"][
                    "image_features"
                ]["shape"],
                [84, 2048],
            )
            self.assertEqual(
                json.loads(
                    (destination / "img-enc-htp.json").read_text(encoding="utf-8")
                )["image-encoder"]["engine"]["model"]["vision-param"],
                {"height": 14, "width": 24},
            )
            self.assertEqual(
                (destination / "sample_inputs" / "pixel_values.raw").stat().st_size,
                336 * 1536 * 4,
            )
            marker = json.loads(
                (destination / MARKER_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["vision_replacement"]["pixel_values_shape"],
                [336, 1536],
            )
            self.assertEqual(marker["vision_replacement"]["visual_tokens"], 84)
            self.assertEqual(
                marker["vision_replacement"]["reused_text_contexts"],
                [
                    "part1_of_4.bin",
                    "part2_of_4.bin",
                    "part3_of_4.bin",
                    "part4_of_4.bin",
                ],
            )

    def test_rejects_invalid_vision_profile_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            replacement = root / "replacement"
            destination = root / "destination"
            source.mkdir()
            replacement.mkdir()
            _write_fixture(source, full_deepstack=True)
            _write_vision_replacement(replacement)
            bad_tensor = (
                replacement / "sample_inputs" / "position_ids_cos.raw"
            )
            _write_sparse(bad_tensor, bad_tensor.stat().st_size - 4)

            with self.assertRaisesRegex(ValueError, "Ancillary tensor"):
                prepare_bundle(
                    source,
                    destination,
                    vision_replacement_bundle=replacement,
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
