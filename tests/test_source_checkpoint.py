from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import qai_hub_models.models as qaihm_models

# The adapter is an overlay package. Prefer the workspace copy while keeping
# the installed QAI Hub Models 0.58 package for its shared implementation.
WORKSPACE_MODELS = Path(__file__).resolve().parents[1] / "qai_hub_models" / "models"
qaihm_models.__path__.insert(0, str(WORKSPACE_MODELS))

from qai_hub_models import Precision  # noqa: E402
from qai_hub_models.models._shared.qwen3_vl.model import (  # noqa: E402
    Qwen3VLCollectionBase,
    Qwen3VLVisionEncoderBase,
)
from qai_hub_models.models._shared.qwen2_vl.vision_encoder_adaptations import (  # noqa: E402
    Conv2dInplaceConv3d,
)
from qai_hub_models.models.cosmos_reason2_2b.demo import (  # noqa: E402
    _configure_demo_source_checkpoint,
)
from qai_hub_models.models.cosmos_reason2_2b.export import (  # noqa: E402
    _ordered_compiled_models_by_component,
)
from qai_hub_models.models.cosmos_reason2_2b.model import (  # noqa: E402
    HF_REPO_NAME,
    QAIRT245_COMPAT_MARKER,
    SOURCE_CHECKPOINT_ENV,
    SOURCE_CHECKPOINT_MARKER,
    W4_FP16_MARKER,
    Cosmos_Reason2_2B_Collection,
    Cosmos_Reason2_2B_PreSplit,
    Cosmos_Reason2_2B_QuantizablePreSplit,
    Cosmos_Reason2_2B_VisionEncoder,
    _pad_prefill_kwargs_to_sequence_multiple,
    _restore_temporal_patch_projection_bias,
    resolve_source_checkpoint,
)
from qai_hub_models.models.cosmos_reason2_2b.quantize import (  # noqa: E402
    _configure_local_checkpoint,
)
from scripts.finalize_checkpoint import (  # noqa: E402
    main as finalize_checkpoint,
)
from scripts.finalize_checkpoint import write_source_checkpoint_marker  # noqa: E402
from scripts.collect_evk_npu_evidence import (  # noqa: E402
    context_graph_order_errors,
    extract_context_graph_order,
)


def _make_source(root: Path) -> Path:
    source = root / "bf16"
    source.mkdir()
    config = {
        "model_type": "qwen3_vl",
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
        },
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"stub")
    return source


class SourceCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.original_hf_names = {
            Cosmos_Reason2_2B_Collection: Cosmos_Reason2_2B_Collection._hf_repo_name,
            Cosmos_Reason2_2B_PreSplit: Cosmos_Reason2_2B_PreSplit._hf_repo_name,
            Cosmos_Reason2_2B_QuantizablePreSplit: (
                Cosmos_Reason2_2B_QuantizablePreSplit._hf_repo_name
            ),
            Cosmos_Reason2_2B_VisionEncoder: (
                Cosmos_Reason2_2B_VisionEncoder._hf_repo_name
            ),
        }

    def tearDown(self) -> None:
        for model_cls, hf_name in self.original_hf_names.items():
            model_cls._hf_repo_name = hf_name
        self.temporary_directory.cleanup()

    def test_marker_and_environment_override_are_movable(self) -> None:
        source = _make_source(self.root)
        quantized = self.root / "w4a16"
        quantized.mkdir()
        marker = write_source_checkpoint_marker(source, quantized)

        self.assertEqual(
            json.loads(marker.read_text(encoding="utf-8"))["source_checkpoint"],
            str(source),
        )
        with patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: ""}):
            self.assertEqual(resolve_source_checkpoint(quantized), source)

        marker.write_text(
            json.dumps({"schema_version": 1, "source_checkpoint": "/moved/away"}),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: str(source)}):
            self.assertEqual(resolve_source_checkpoint(quantized), source)

    def test_finalizer_records_source_checkpoint(self) -> None:
        source = _make_source(self.root)
        quantized = self.root / "w4a16"
        quantized.mkdir()
        for name in [
            "model_dynamic.onnx",
            "model.data",
            "model.encodings",
            "embedding_weights.raw",
            "vision_encoder.onnx",
            "vision_encoder.encodings",
        ]:
            (quantized / name).write_bytes(b"stub")

        with patch.object(
            sys,
            "argv",
            ["finalize_checkpoint.py", str(source), str(quantized)],
        ):
            self.assertEqual(finalize_checkpoint(), 0)

        marker = quantized / SOURCE_CHECKPOINT_MARKER
        self.assertEqual(
            json.loads(marker.read_text(encoding="utf-8"))["source_checkpoint"],
            str(source),
        )

    def test_missing_marker_has_actionable_error(self) -> None:
        quantized = self.root / "w4a16"
        quantized.mkdir()

        with (
            patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: ""}),
            self.assertRaisesRegex(ValueError, SOURCE_CHECKPOINT_MARKER),
        ):
            resolve_source_checkpoint(quantized)

    def test_collection_configures_source_before_shared_construction(self) -> None:
        source = _make_source(self.root)
        quantized = self.root / "w4a16"
        quantized.mkdir()
        write_source_checkpoint_marker(source, quantized)
        captured: dict[str, object] = {}

        def fake_from_pretrained(
            cls, checkpoint="DEFAULT", host_device=None, **kwargs
        ):
            captured["checkpoint"] = checkpoint
            captured["host_device"] = host_device
            captured["kwargs"] = kwargs
            return "stub-collection"

        with (
            patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: ""}),
            patch.object(
                Qwen3VLCollectionBase,
                "from_pretrained",
                new=classmethod(fake_from_pretrained),
            ),
        ):
            result = Cosmos_Reason2_2B_Collection.from_pretrained(
                checkpoint=quantized,
                precision=Precision.w4a16,
            )

        self.assertEqual(result, "stub-collection")
        self.assertEqual(captured["checkpoint"], quantized)
        self.assertEqual(Cosmos_Reason2_2B_Collection._hf_repo_name, str(source))
        self.assertEqual(Cosmos_Reason2_2B_PreSplit._hf_repo_name, str(source))
        self.assertEqual(
            Cosmos_Reason2_2B_QuantizablePreSplit._hf_repo_name, str(source)
        )
        self.assertEqual(
            Cosmos_Reason2_2B_VisionEncoder._hf_repo_name, str(source)
        )

    def test_compat_checkpoint_disables_only_pipeline_deepstack(self) -> None:
        source = _make_source(self.root)
        quantized = self.root / "w4a16"
        quantized.mkdir()
        write_source_checkpoint_marker(source, quantized)
        (quantized / QAIRT245_COMPAT_MARKER).write_text(
            "{}\n", encoding="utf-8"
        )

        class StubCollection:
            num_deepstack_layers = 3

        stub = StubCollection()

        def fake_from_pretrained(cls, **kwargs):
            return stub

        with (
            patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: ""}),
            patch.object(
                Qwen3VLCollectionBase,
                "from_pretrained",
                new=classmethod(fake_from_pretrained),
            ),
        ):
            result = Cosmos_Reason2_2B_Collection.from_pretrained(
                checkpoint=quantized
            )

        self.assertIs(result, stub)
        self.assertEqual(result.num_deepstack_layers, 0)
        self.assertEqual(Cosmos_Reason2_2B_PreSplit.num_deepstack_layers, 3)

    def test_quantizer_routes_every_loader_to_local_checkpoint(self) -> None:
        source = _make_source(self.root)

        _configure_local_checkpoint(["--checkpoint", str(source)])

        for model_cls in self.original_hf_names:
            self.assertEqual(model_cls._hf_repo_name, str(source))

    def test_calibration_prompts_are_padded_before_reverse_slicing(self) -> None:
        input_ids = torch.arange(193, dtype=torch.int64).reshape(1, 193)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.ones((4, 8), dtype=torch.float32)

        padded = _pad_prefill_kwargs_to_sequence_multiple(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            },
            sequence_length=128,
            pad_token_id=151643,
        )

        self.assertEqual(tuple(padded["input_ids"].shape), (1, 256))
        self.assertEqual(tuple(padded["attention_mask"].shape), (1, 256))
        self.assertTrue(torch.equal(padded["input_ids"][:, -193:], input_ids))
        self.assertTrue(
            torch.equal(padded["attention_mask"][:, -193:], attention_mask)
        )
        self.assertTrue(
            torch.all(padded["input_ids"][:, :63] == 151643).item()
        )
        self.assertTrue(
            torch.all(padded["attention_mask"][:, :63] == 0).item()
        )
        self.assertIs(padded["pixel_values"], pixel_values)

    def test_temporal_patch_adaptation_preserves_qwen3_bias(self) -> None:
        torch.manual_seed(7)
        projection = torch.nn.Conv3d(
            in_channels=3,
            out_channels=5,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            bias=True,
        )
        assert projection.bias is not None
        projection.bias.data.copy_(
            torch.tensor([-0.75, -0.25, 0.125, 0.5, 1.25])
        )
        pixel_values = torch.randn(6, 3 * 2 * 4 * 4)
        expected = projection(
            pixel_values.reshape(-1, 3, 2, 4, 4)
        ).reshape(6, 5)

        adapted = Conv2dInplaceConv3d(
            projection,
            in_channels=3,
            temporal_patch_size=2,
            patch_size=4,
        )
        self.assertIsNone(adapted.conv2d.bias)
        _restore_temporal_patch_projection_bias(
            adapted, projection.bias.detach().clone()
        )
        actual = adapted(pixel_values).reshape(6, 5)

        self.assertIsNotNone(adapted.conv2d.bias)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_vision_input_spec_uses_constructed_non_square_geometry(self) -> None:
        encoder = Cosmos_Reason2_2B_VisionEncoder.__new__(
            Cosmos_Reason2_2B_VisionEncoder
        )
        torch.nn.Module.__init__(encoder)
        encoder._image_height = 224
        encoder._image_width = 384
        encoder._patch_size = 16
        encoder._pos_emb_cos = torch.zeros((336, 32), dtype=torch.float32)

        spec = encoder.get_input_spec()

        self.assertEqual(spec["pixel_values"].shape, (336, 1536))
        self.assertEqual(spec["position_ids_cos"].shape, (336, 32))
        self.assertEqual(spec["full_attention_mask"].shape, (1, 336, 336))

    def test_mixed_precision_vision_compile_forces_quantized_io(self) -> None:
        checkpoint = self.root / "mixed"
        checkpoint.mkdir()
        (checkpoint / W4_FP16_MARKER).write_text("{}", encoding="utf-8")
        encoder = Cosmos_Reason2_2B_VisionEncoder.__new__(
            Cosmos_Reason2_2B_VisionEncoder
        )
        torch.nn.Module.__init__(encoder)
        encoder._checkpoint = str(checkpoint)

        with patch.object(
            Qwen3VLVisionEncoderBase,
            "get_hub_compile_options",
            return_value="--target_runtime qnn_context_binary",
        ):
            options = encoder.get_hub_compile_options(None, None)

        self.assertEqual(options.split().count("--quantize_io"), 1)

    def test_vision_checkpoint_recovers_recorded_image_size(self) -> None:
        checkpoint = self.root / "vision"
        checkpoint.mkdir()
        (checkpoint / "args.json").write_text(
            json.dumps({"image_size": [224, 384]}),
            encoding="utf-8",
        )
        captured: dict[str, object] = {}

        def fake_from_pretrained(cls, **kwargs):
            captured.update(kwargs)
            return object()

        with patch.object(
            Qwen3VLVisionEncoderBase,
            "from_pretrained",
            new=classmethod(fake_from_pretrained),
        ):
            Cosmos_Reason2_2B_VisionEncoder.from_pretrained(
                checkpoint=checkpoint,
                precision=Precision.w4a16,
            )

        self.assertEqual(captured["image_height"], 224)
        self.assertEqual(captured["image_width"], 384)

    def test_vision_checkpoint_rejects_explicit_geometry_mismatch(self) -> None:
        checkpoint = self.root / "vision"
        checkpoint.mkdir()
        (checkpoint / "args.json").write_text(
            json.dumps({"image_size": [224, 384]}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "image_height.*512 != 224"):
            Cosmos_Reason2_2B_VisionEncoder.from_pretrained(
                checkpoint=checkpoint,
                image_height=512,
                image_width=384,
                precision=Precision.w4a16,
            )
        with self.assertRaisesRegex(ValueError, "image_width.*512 != 384"):
            Cosmos_Reason2_2B_VisionEncoder.from_pretrained(
                checkpoint=checkpoint,
                image_width=512,
                precision=Precision.w4a16,
            )

    def test_standard_default_call_fails_before_shared_construction(self) -> None:
        called = False

        def fake_from_pretrained(cls, **kwargs):
            nonlocal called
            called = True
            return "unexpected"

        with patch.object(
            Qwen3VLCollectionBase,
            "from_pretrained",
            new=classmethod(fake_from_pretrained),
        ):
            with self.assertRaisesRegex(
                ValueError, "no packaged default checkpoint"
            ):
                Cosmos_Reason2_2B_Collection.from_pretrained()

        self.assertFalse(called)
        with self.assertRaisesRegex(ValueError, "no packaged default checkpoint"):
            Cosmos_Reason2_2B_QuantizablePreSplit.fetch_default_checkpoint(
                Precision.w4a16
            )
        self.assertNotEqual(Cosmos_Reason2_2B_PreSplit._hf_repo_name, "")
        self.assertTrue(HF_REPO_NAME)

    def test_demo_uses_marker_and_rejects_implicit_default(self) -> None:
        source = _make_source(self.root)
        quantized = self.root / "w4a16"
        quantized.mkdir()
        write_source_checkpoint_marker(source, quantized)

        with (
            patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: ""}),
            patch.object(sys, "argv", ["demo", "--checkpoint", str(quantized)]),
        ):
            self.assertEqual(_configure_demo_source_checkpoint(None), str(source))

        with (
            patch.dict(os.environ, {SOURCE_CHECKPOINT_ENV: ""}),
            patch.object(sys, "argv", ["demo"]),
            self.assertRaisesRegex(ValueError, "no packaged default checkpoint"),
        ):
            _configure_demo_source_checkpoint(None)

    def test_linker_uses_declared_graph_order_not_mapping_order(self) -> None:
        class FakeCollection:
            component_names = ["part1_of_4", "part4_of_4"]

            @staticmethod
            def get_component_graph_names(component_name: str) -> list[str]:
                part = 1 if component_name == "part1_of_4" else 4
                return [
                    f"ar128_cl512_{part}_of_4",
                    f"ar1_cl512_{part}_of_4",
                ]

        compiled = {
            ("part4_of_4", "ar1_cl512_4_of_4"): "part4-ar1",
            ("part1_of_4", "ar128_cl512_1_of_4"): "part1-ar128",
            ("part4_of_4", "ar128_cl512_4_of_4"): "part4-ar128",
            ("part1_of_4", "ar1_cl512_1_of_4"): "part1-ar1",
        }

        ordered = _ordered_compiled_models_by_component(
            compiled, FakeCollection()
        )

        self.assertEqual(
            ordered["part1_of_4"], ["part1-ar128", "part1-ar1"]
        )
        self.assertEqual(
            ordered["part4_of_4"], ["part4-ar128", "part4-ar1"]
        )

    def test_context_utility_graph_order_is_normalized(self) -> None:
        graph_names, normalized = extract_context_graph_order(
            """
            {"graphName": "ar128_cl512_4_of_4"}
            {"graphName": "ar1_cl512_4_of_4"}
            """,
            expected_part=4,
        )

        self.assertEqual(
            graph_names,
            ["ar128_cl512_4_of_4", "ar1_cl512_4_of_4"],
        )
        self.assertEqual(normalized, [(128, 512), (1, 512)])

    def test_context_graph_order_rejects_reversed_partition(self) -> None:
        errors = context_graph_order_errors(
            {
                "part1_of_4.bin": [(128, 512), (1, 512)],
                "part2_of_4.bin": [(128, 512), (1, 512)],
                "part3_of_4.bin": [(128, 512), (1, 512)],
                "part4_of_4.bin": [(1, 512), (128, 512)],
            }
        )

        self.assertTrue(
            any("Linked graph order mismatch" in error for error in errors)
        )
        self.assertTrue(
            any("links AR-1 before" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
