from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from qai_hub_models.models.cosmos_reason2_2b.calibration import (
    load_paired_frame_calibration_data,
    resolve_paired_frame_paths,
    select_class_balanced_paths,
)
from qai_hub_models.models.cosmos_reason2_2b.quantize import (
    _extract_vision_calibration_args,
    _extract_vision_mixed_precision_args,
    _record_calibration_provenance,
)
from qai_hub_models.models.cosmos_reason2_2b.model import (
    _last_vision_block_activation_quantizer_names,
    _set_last_vision_block_activations_float16,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeProcessor:
    def __init__(self, *, native_matches: bool = True) -> None:
        self.native_matches = native_matches
        self.calls: list[dict[str, object]] = []

    def video_processor(self, **kwargs):
        self.calls.append(kwargs)
        static_fallback = kwargs.get("do_resize") is False
        if not self.native_matches and not static_fallback:
            pixels = torch.zeros((1, 3, 2, 16, 16), dtype=torch.float32)
            pixels[:, :, 1] = 0.5
            return {
                "pixel_values_videos": pixels.reshape(1, 1536),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        pixels = torch.zeros((4, 3, 2, 16, 16), dtype=torch.float32)
        pixels[:, :, 1] = 0.5
        return {
            "pixel_values_videos": pixels.reshape(4, 1536),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class VisionCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_imagenette_selection_is_deterministic_and_class_balanced(self) -> None:
        train = self.root / "train"
        for class_name in ("class_c", "class_a", "class_b"):
            class_dir = train / class_name
            class_dir.mkdir(parents=True)
            for image_index in range(3):
                (class_dir / f"{image_index}.jpg").write_bytes(
                    f"{class_name}-{image_index}".encode()
                )

        selected = select_class_balanced_paths(train, 7)

        self.assertEqual(
            [(path.parent.name, path.name) for path in selected],
            [
                ("class_a", "0.jpg"),
                ("class_b", "0.jpg"),
                ("class_c", "0.jpg"),
                ("class_a", "1.jpg"),
                ("class_b", "1.jpg"),
                ("class_c", "1.jpg"),
                ("class_a", "2.jpg"),
            ],
        )

    def _write_manifest(
        self,
        duplicate: bool = False,
        temporal_mode: str | None = None,
    ) -> Path:
        from PIL import Image

        first = self.root / "first.png"
        second = self.root / "second.png"
        Image.new("RGB", (20, 18), (20, 40, 60)).save(first)
        Image.new(
            "RGB",
            (20, 18),
            (20, 40, 60) if duplicate else (80, 100, 120),
        ).save(second)
        manifest = self.root / "pairs.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    **(
                        {"temporal_mode": temporal_mode}
                        if temporal_mode is not None
                        else {}
                    ),
                    "pairs": [
                        {
                            "id": "pair-0",
                            "frames": [first.name, second.name],
                            "timestamps_seconds": [0.0, 0.5],
                            "sha256": [_sha256(first), _sha256(second)],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_local_pair_uses_native_video_processor_resize(self) -> None:
        manifest = self._write_manifest()
        processor = _FakeProcessor()

        calibration = load_paired_frame_calibration_data(
            hf_repo="/local/checkpoint",
            manifest_path=manifest,
            num_samples=1,
            image_height=32,
            image_width=32,
            processor_loader=lambda _: processor,
        )

        self.assertEqual(len(calibration), 1)
        self.assertEqual(calibration[0].shape, (4, 1536))
        self.assertEqual(len(processor.calls), 1)
        call = processor.calls[0]
        self.assertFalse(call["do_sample_frames"])
        self.assertNotIn("do_resize", call)
        frames = call["videos"][0]
        self.assertEqual([frame.size for frame in frames], [(20, 18), (20, 18)])

    def test_local_pair_falls_back_for_static_graph_geometry(self) -> None:
        manifest = self._write_manifest()
        processor = _FakeProcessor(native_matches=False)

        calibration = load_paired_frame_calibration_data(
            hf_repo="/local/checkpoint",
            manifest_path=manifest,
            num_samples=1,
            image_height=32,
            image_width=32,
            processor_loader=lambda _: processor,
        )

        self.assertEqual(len(calibration), 1)
        self.assertEqual(calibration[0].shape, (4, 1536))
        self.assertEqual(len(processor.calls), 2)
        native_frames = processor.calls[0]["videos"][0]
        fallback_frames = processor.calls[1]["videos"][0]
        self.assertEqual(
            [frame.size for frame in native_frames],
            [(20, 18), (20, 18)],
        )
        self.assertEqual(
            [frame.size for frame in fallback_frames],
            [(32, 32), (32, 32)],
        )
        self.assertNotIn("do_resize", processor.calls[0])
        self.assertFalse(processor.calls[1]["do_resize"])

    def test_pair_manifest_rejects_duplicate_frame_content(self) -> None:
        manifest = self._write_manifest(duplicate=True)

        with self.assertRaisesRegex(ValueError, "duplicate frame content"):
            resolve_paired_frame_paths(manifest, 1)

    def test_single_frame_mode_accepts_duplicate_frame_content(self) -> None:
        manifest = self._write_manifest(
            duplicate=True,
            temporal_mode="duplicate_single_frame",
        )

        pairs = resolve_paired_frame_paths(manifest, 1)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(_sha256(pairs[0][0]), _sha256(pairs[0][1]))

    def test_single_frame_mode_rejects_distinct_frame_content(self) -> None:
        manifest = self._write_manifest(
            temporal_mode="duplicate_single_frame",
        )

        with self.assertRaisesRegex(ValueError, "must duplicate one frame"):
            resolve_paired_frame_paths(manifest, 1)

    def test_cli_extracts_local_paired_manifest_option(self) -> None:
        manifest = self._write_manifest()

        remaining, selected = _extract_vision_calibration_args(
            [
                "--checkpoint",
                "/local/checkpoint",
                "--veg-paired-calibration-manifest",
                str(manifest),
                "--skip-llm",
            ]
        )

        self.assertEqual(selected, manifest.resolve())
        self.assertEqual(
            remaining,
            ["--checkpoint", "/local/checkpoint", "--skip-llm"],
        )

    def test_cli_extracts_final_block_fp16_option(self) -> None:
        remaining, enabled = _extract_vision_mixed_precision_args(
            [
                "--checkpoint",
                "/local/checkpoint",
                "--veg-fp16-last-block-activations",
                "--skip-llm",
            ]
        )

        self.assertTrue(enabled)
        self.assertEqual(
            remaining,
            ["--checkpoint", "/local/checkpoint", "--skip-llm"],
        )

    def test_final_block_activation_selector_uses_parameter_boundaries(
        self,
    ) -> None:
        class FakeQuantizer:
            def __init__(
                self,
                *,
                enabled: bool,
                data_type: str,
                bitwidth: int,
            ) -> None:
                self.enabled = enabled
                self.data_type = data_type
                self.bitwidth = bitwidth
                self.reset_count = 0

            def reset_encoding_stats(self) -> None:
                self.reset_count += 1

        def node(op_type, inputs, outputs):
            return SimpleNamespace(
                op_type=op_type,
                input=inputs,
                output=outputs,
            )

        quantizers = {
            "block23_norm": FakeQuantizer(
                enabled=True,
                data_type="INT",
                bitwidth=16,
            ),
            "block23_hidden": FakeQuantizer(
                enabled=True,
                data_type="INT",
                bitwidth=16,
            ),
            "block23_mlp": FakeQuantizer(
                enabled=True,
                data_type="INT",
                bitwidth=16,
            ),
            "block23_residual": FakeQuantizer(
                enabled=True,
                data_type="INT",
                bitwidth=16,
            ),
            "block23_disabled": FakeQuantizer(
                enabled=False,
                data_type="INT",
                bitwidth=16,
            ),
            "merger_norm": FakeQuantizer(
                enabled=True,
                data_type="INT",
                bitwidth=16,
            ),
            "blocks.23.mlp.linear_fc2.weight": FakeQuantizer(
                enabled=True,
                data_type="INT",
                bitwidth=8,
            ),
        }
        quant_sim = SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(
                    graph=SimpleNamespace(
                        node=[
                            node(
                                "LayerNormalization",
                                [
                                    "block22_output_updated",
                                    "blocks.23.norm1.weight_qdq",
                                ],
                                ["block23_norm"],
                            ),
                            node(
                                "Add",
                                ["block23_norm_updated", "block22_output_updated"],
                                ["block23_hidden"],
                            ),
                            node(
                                "Identity",
                                ["block23_hidden_updated"],
                                ["block23_disabled"],
                            ),
                            node(
                                "Conv",
                                [
                                    "block23_disabled_updated",
                                    "blocks.23.mlp.linear_fc2.weight_qdq",
                                ],
                                ["block23_mlp"],
                            ),
                            node(
                                "Add",
                                ["block23_mlp_updated", "block22_output_updated"],
                                ["block23_residual"],
                            ),
                            node(
                                "QcQuantizeOp",
                                ["block23_residual"],
                                ["block23_residual_updated"],
                            ),
                            node(
                                "LayerNormalization",
                                [
                                    "block23_residual_updated",
                                    "merger.norm.weight_qdq",
                                ],
                                ["merger_norm"],
                            ),
                        ]
                    )
                )
            ),
            activation_names=set(quantizers),
            qc_quantize_op_dict=quantizers,
        )

        self.assertEqual(
            _last_vision_block_activation_quantizer_names(quant_sim),
            ("block23_hidden", "block23_mlp", "block23_norm"),
        )
        selected = _set_last_vision_block_activations_float16(
            quant_sim,
            float_data_type="FLOAT",
        )
        self.assertEqual(
            selected,
            ("block23_hidden", "block23_mlp", "block23_norm"),
        )
        for name in selected:
            quantizer = quantizers[name]
            self.assertTrue(quantizer.enabled)
            self.assertEqual(quantizer.data_type, "FLOAT")
            self.assertEqual(quantizer.bitwidth, 16)
            self.assertEqual(quantizer.reset_count, 1)
        for name in (
            "block23_residual",
            "block23_disabled",
            "merger_norm",
        ):
            quantizer = quantizers[name]
            self.assertEqual(quantizer.data_type, "INT")
            self.assertEqual(quantizer.bitwidth, 16)
            self.assertEqual(quantizer.reset_count, 0)
        parameter = quantizers["blocks.23.mlp.linear_fc2.weight"]
        self.assertEqual(parameter.data_type, "INT")
        self.assertEqual(parameter.bitwidth, 8)
        self.assertEqual(parameter.reset_count, 0)

    def test_provenance_records_non_default_image_size(self) -> None:
        manifest = self._write_manifest()
        output = self.root / "output"
        output.mkdir()
        (output / "args.json").write_text("{}", encoding="utf-8")

        _record_calibration_provenance(
            output,
            ["--image-size", "224", "384"],
            manifest,
        )

        recorded = json.loads(
            (output / "args.json").read_text(encoding="utf-8")
        )
        self.assertEqual(recorded["image_size"], [224, 384])
        self.assertEqual(
            recorded["vision_calibration_source"],
            "paired_frame_manifest",
        )

    def test_provenance_records_final_block_fp16_quantizers(self) -> None:
        output = self.root / "output"
        output.mkdir()
        (output / "args.json").write_text("{}", encoding="utf-8")

        _record_calibration_provenance(
            output,
            ["--veg-fp16-last-block-activations"],
            None,
            fp16_last_block_activations=True,
            fp16_activation_quantizer_names=("add_120", "layer_norm_50"),
        )

        recorded = json.loads(
            (output / "args.json").read_text(encoding="utf-8")
        )
        mixed = recorded["vision_mixed_precision"]
        self.assertEqual(mixed["fp16_activation_blocks"], [23])
        self.assertEqual(mixed["fp16_activation_quantizer_count"], 2)
        self.assertEqual(
            mixed["fp16_activation_quantizer_names"],
            ["add_120", "layer_norm_50"],
        )
        self.assertEqual(mixed["parameter_quantization"], "W8 integer")


if __name__ == "__main__":
    unittest.main()
