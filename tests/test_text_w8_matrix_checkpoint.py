from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.make_text_w8_matrix_checkpoint import (
    ENCODINGS_FILENAME,
    EXPECTED_ACTIVATION_COUNT,
    EXPECTED_INT16_PARAMETER_COUNT,
    EXPECTED_LAYER_COUNT,
    EXPECTED_MATRICES_PER_LAYER,
    MARKER_FILENAME,
    MODEL_FILENAME,
    create_text_w8_matrix_checkpoint,
)

HAS_REQUIRED_QUANTIZATION_PACKAGES = (
    importlib.util.find_spec("aimet_onnx") is not None
    and importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("numpy") is not None
)


def _w4_encoding(name: str) -> dict[str, object]:
    return {
        "name": name,
        "bw": 4,
        "dtype": "INT",
        "enc_type": "PER_CHANNEL",
        "is_sym": True,
        "scale": [1.0, 6.0 / 7.0],
        "offset": [-8.0, -8.0],
    }


def _matrix_names(layer: int) -> tuple[list[str], list[str]]:
    prefix = f"model.model.layers.{layer}"
    named = [
        *(
            f"{prefix}.self_attn.q_proj_sha.{index}.weight"
            for index in range(16)
        ),
        *(
            f"{prefix}.self_attn.k_proj_sha.{index}.weight"
            for index in range(8)
        ),
        *(
            f"{prefix}.self_attn.v_proj_sha.{index}.weight"
            for index in range(8)
        ),
        f"{prefix}.self_attn.o_proj_conv.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]
    anonymous = [f"val_{5000 + layer * 10 + index}" for index in range(2)]
    return named, anonymous


def _make_value_info(onnx, name: str, dtype: int, shape):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _graph_inputs_and_outputs(onnx):
    float_type = onnx.TensorProto.FLOAT
    bool_type = onnx.TensorProto.BOOL
    inputs = [
        _make_value_info(
            onnx, "inputs_embeds", float_type, [1, "seq", 2048]
        ),
        _make_value_info(
            onnx,
            "attention_mask",
            float_type,
            [1, 1, "seq", "context"],
        ),
        _make_value_info(
            onnx,
            "position_ids_cos",
            float_type,
            [1, 1, "seq", 64],
        ),
        _make_value_info(
            onnx,
            "position_ids_sin",
            float_type,
            [1, 1, "seq", 64],
        ),
    ]
    for layer in range(EXPECTED_LAYER_COUNT):
        inputs.extend(
            [
                _make_value_info(
                    onnx,
                    f"past_key_{layer}_in",
                    float_type,
                    [8, 1, 128, f"past_key_{layer}"],
                ),
                _make_value_info(
                    onnx,
                    f"past_value_{layer}_in",
                    float_type,
                    [8, 1, f"past_value_{layer}", 128],
                ),
            ]
        )
    inputs.append(
        _make_value_info(
            onnx, "visual_pos_masks", bool_type, [1, "seq"]
        )
    )
    inputs.extend(
        _make_value_info(
            onnx,
            f"deepstack_visual_embeds_{index}",
            float_type,
            [f"deepstack_{index}", 2048],
        )
        for index in range(3)
    )

    outputs = [
        _make_value_info(
            onnx, "logits", float_type, [1, "seq", 151936]
        )
    ]
    for layer in range(EXPECTED_LAYER_COUNT):
        outputs.extend(
            [
                _make_value_info(
                    onnx,
                    f"past_key_{layer}_out",
                    float_type,
                    [8, 1, 128, "seq"],
                ),
                _make_value_info(
                    onnx,
                    f"past_value_{layer}_out",
                    float_type,
                    [8, 1, "seq", 128],
                ),
            ]
        )
    return inputs, outputs


def _write_checkpoint(root: Path) -> dict[str, bytes]:
    import numpy as np
    import onnx

    root.mkdir()
    data = bytearray()
    tensors = []
    nodes = []
    parameter_encodings: list[dict[str, object]] = []
    raw_by_name: dict[str, bytes] = {}

    def add_initializer(name: str, array, op_type: str) -> None:
        raw = np.asarray(array, dtype=np.float32).tobytes(order="C")
        offset = len(data)
        data.extend(raw)
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.extend(array.shape)
        # ONNX's helper deliberately requires raw_data to exist before it
        # rewrites the tensor as an external-data reference.
        tensor.raw_data = raw
        onnx.external_data_helper.set_external_data(
            tensor,
            location="model.data",
            offset=offset,
            length=len(raw),
        )
        tensor.ClearField("raw_data")
        tensors.append(tensor)
        nodes.append(
            onnx.helper.make_node(
                op_type,
                ["inputs_embeds", name],
                [f"unused_{len(nodes)}"],
                name=f"consumer_{len(nodes)}",
            )
        )
        raw_by_name[name] = raw

    named_array = np.asarray(
        [
            [[[-8.0]], [[7.0]], [[-2.0]]],
            [[[-4.0]], [[6.0]], [[1.0]]],
        ],
        dtype=np.float32,
    )
    square_conv_array = np.asarray(
        [
            [[[-8.0]], [[7.0]]],
            [[[-4.0]], [[6.0]]],
        ],
        dtype=np.float32,
    )
    anonymous_array = np.asarray(
        [
            [-8.0, -4.0],
            [7.0, 6.0],
            [-2.0, 1.0],
        ],
        dtype=np.float32,
    )
    for layer in range(EXPECTED_LAYER_COUNT):
        named, anonymous = _matrix_names(layer)
        for name in named:
            array = (
                square_conv_array
                if name.endswith("self_attn.o_proj_conv.weight")
                else named_array
            )
            add_initializer(name, array, "Conv")
            parameter_encodings.append(_w4_encoding(name))
        for name in anonymous:
            add_initializer(name, anonymous_array, "MatMul")
            parameter_encodings.append(_w4_encoding(name))

    parameter_encodings.extend(
        {
            "name": f"norm_parameter_{index}",
            "bw": 16,
            "dtype": "INT",
            "enc_type": "PER_TENSOR",
            "is_sym": False,
            "scale": [0.001],
            "offset": [-100.0],
        }
        for index in range(EXPECTED_INT16_PARAMETER_COUNT)
    )
    parameter_encodings.append(
        {
            "name": "model.lm_head.weight",
            "bw": 8,
            "dtype": "INT",
            "enc_type": "PER_CHANNEL",
            "is_sym": True,
            "scale": [0.01],
            "offset": [-128.0],
        }
    )
    activations = [
        {
            "name": f"activation_{index}",
            "bw": 16,
            "dtype": "FLOAT",
            "enc_type": "PER_TENSOR",
        }
        for index in range(EXPECTED_ACTIVATION_COUNT)
    ]
    encodings = {
        "version": "1.0.0",
        "activation_encodings": activations,
        "param_encodings": parameter_encodings,
        "producer": {"name": "AIMET", "version": "2.20.0"},
        "quantizer_args": {
            "activation_bitwidth": 16,
            "dtype": "int",
            "is_symmetric": True,
            "param_bitwidth": 4,
            "per_channel_quantization": True,
            "quant_scheme": "min_max",
        },
    }
    inputs, outputs = _graph_inputs_and_outputs(onnx)
    graph = onnx.helper.make_graph(
        nodes,
        "synthetic_cosmos_text",
        inputs,
        outputs,
        initializer=tensors,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx.save_model(model, root / MODEL_FILENAME)

    args = {
        "precision": "w4",
        "context_length": 512,
        "calibration_sequence_length": 128,
        "image_size": [512, 512],
        "skip_llm": False,
        "raw_args": [
            "--checkpoint",
            "/source",
            "--context-length",
            "512",
            "--calibration-sequence-length",
            "128",
            "--image-size",
            "512",
            "512",
            "--precision",
            "w4",
        ],
    }
    fp16_marker = {
        "precision": {"destination_precision": "w4"},
        "encoding_files": {
            ENCODINGS_FILENAME: {
                "activation_count": EXPECTED_ACTIVATION_COUNT,
                "activations_converted_to_float16": True,
                "destination_activation_types": {
                    "FLOAT16": EXPECTED_ACTIVATION_COUNT
                },
            }
        },
        "replacement_activation_encoding": {
            "bw": 16,
            "dtype": "FLOAT",
            "enc_type": "PER_TENSOR",
        },
    }
    payloads = {
        ENCODINGS_FILENAME: json.dumps(encodings).encode(),
        "args.json": json.dumps(args).encode(),
        "w4_fp16.json": json.dumps(fp16_marker).encode(),
        "model.data": bytes(data),
        "source_checkpoint.json": b'{"source_checkpoint": "/source"}\n',
        "config.json": b"{}\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "embedding_weights.raw": b"embedding",
        "vision_encoder.onnx": b"vision",
        "vision_encoder.encodings": b"{}\n",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    payloads[MODEL_FILENAME] = (root / MODEL_FILENAME).read_bytes()
    return payloads


@unittest.skipUnless(
    HAS_REQUIRED_QUANTIZATION_PACKAGES,
    "AIMET/ONNX quantization packages are unavailable",
)
class TextW8MatrixCheckpointTests(unittest.TestCase):
    def test_part1_recomputes_exact_w8_and_preserves_all_contracts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            source_payloads = _write_checkpoint(source)
            original = json.loads(source_payloads[ENCODINGS_FILENAME])

            create_text_w8_matrix_checkpoint(
                source,
                destination,
                parts=["part1_of_4"],
            )

            for name, payload in source_payloads.items():
                self.assertEqual((source / name).read_bytes(), payload)
            converted = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            self.assertEqual(
                converted["activation_encodings"],
                original["activation_encodings"],
            )
            by_name = {
                entry["name"]: entry
                for entry in converted["param_encodings"]
            }
            part1_names = {
                name
                for layer in range(7)
                for group in _matrix_names(layer)
                for name in group
            }
            self.assertEqual(
                len(part1_names),
                7 * EXPECTED_MATRICES_PER_LAYER,
            )
            for name in part1_names:
                self.assertEqual(by_name[name]["bw"], 8)
                self.assertEqual(by_name[name]["offset"], [-128.0, -128.0])
                self.assertEqual(
                    by_name[name]["scale"],
                    [0.0625, 6.0 / 127.0],
                )

            untouched_name = _matrix_names(7)[0][0]
            original_by_name = {
                entry["name"]: entry
                for entry in original["param_encodings"]
            }
            self.assertEqual(
                by_name[untouched_name],
                original_by_name[untouched_name],
            )
            self.assertEqual(
                (destination / "args.json").read_bytes(),
                source_payloads["args.json"],
            )
            self.assertTrue(
                os.path.samefile(
                    source / MODEL_FILENAME,
                    destination / MODEL_FILENAME,
                )
            )
            self.assertTrue(
                os.path.samefile(
                    source / "model.data",
                    destination / "model.data",
                )
            )

            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            self.assertEqual(marker["selected_parts"], ["part1_of_4"])
            self.assertEqual(marker["selected_layers"], list(range(7)))
            self.assertEqual(
                marker["counts"]["matrices_promoted_w4_to_w8"],
                252,
            )
            self.assertEqual(
                marker["export_contract"]["sequence_lengths"],
                [128, 1],
            )
            self.assertEqual(
                marker["export_contract"]["deepstack_inputs"],
                [
                    "visual_pos_masks",
                    "deepstack_visual_embeds_0",
                    "deepstack_visual_embeds_1",
                    "deepstack_visual_embeds_2",
                ],
            )
            self.assertEqual(
                marker["derivation"]["source_w4_reproduction_required"],
                "exact",
            )

    def test_explicit_raw_image_size_backfill_is_narrow_and_provenanced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)
            args_path = source / "args.json"
            source_args = json.loads(args_path.read_text())
            del source_args["image_size"]
            args_path.write_text(json.dumps(source_args))
            source_args_bytes = args_path.read_bytes()

            create_text_w8_matrix_checkpoint(
                source,
                destination,
                parts=["part1_of_4"],
                image_size=[512, 512],
            )

            self.assertEqual(args_path.read_bytes(), source_args_bytes)
            migrated = json.loads(
                (destination / "args.json").read_text()
            )
            self.assertEqual(migrated["image_size"], [512, 512])
            raw_index = migrated["raw_args"].index("--image-size")
            self.assertEqual(
                migrated["raw_args"][raw_index + 1 : raw_index + 3],
                ["512", "512"],
            )
            source_comparable = dict(source_args)
            migrated_comparable = dict(migrated)
            migrated_comparable.pop("image_size")
            self.assertEqual(migrated_comparable, source_comparable)
            self.assertFalse(migrated["skip_llm"])
            self.assertEqual(migrated["context_length"], 512)
            self.assertEqual(
                migrated["calibration_sequence_length"],
                128,
            )

            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            migration = marker["image_size_metadata_migration"]
            self.assertTrue(migration["applied"])
            self.assertIsNone(
                migration["source_top_level_image_size"]
            )
            self.assertEqual(
                migration["source_raw_args_image_size"],
                [512, 512],
            )
            self.assertEqual(
                migration["destination_top_level_image_size"],
                [512, 512],
            )
            self.assertFalse(migration["raw_args_changed"])
            self.assertEqual(
                migration["changed_fields"],
                ["args.image_size"],
            )
            self.assertEqual(
                marker["export_contract"]["image_size"],
                [512, 512],
            )
            independence = marker[
                "text_graph_image_size_independence"
            ]
            self.assertEqual(independence["pixel_or_grid_inputs"], [])
            self.assertEqual(
                independence["visual_token_axes"],
                "dynamic",
            )
            self.assertIn(
                "vision context must independently match",
                independence["scope"],
            )
            self.assertEqual(
                marker["hashes"]["source_args_sha256"],
                hashlib.sha256(source_args_bytes).hexdigest(),
            )
            self.assertEqual(
                marker["hashes"]["output_args_sha256"],
                hashlib.sha256(
                    (destination / "args.json").read_bytes()
                ).hexdigest(),
            )
            self.assertTrue(
                os.path.samefile(
                    source / "vision_encoder.onnx",
                    destination / "vision_encoder.onnx",
                )
            )

    def test_missing_image_metadata_requires_matching_explicit_backfill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)
            args_path = source / "args.json"
            args = json.loads(args_path.read_text())
            del args["image_size"]
            args_path.write_text(json.dumps(args))

            with self.assertRaisesRegex(
                ValueError,
                "top-level image_size is missing",
            ):
                create_text_w8_matrix_checkpoint(
                    source,
                    destination,
                    parts=["part1_of_4"],
                )
            with self.assertRaisesRegex(
                ValueError,
                "must exactly equal raw_args",
            ):
                create_text_w8_matrix_checkpoint(
                    source,
                    destination,
                    parts=["part1_of_4"],
                    image_size=[224, 384],
                )
            self.assertFalse(destination.exists())

    def test_can_target_all_decoder_parts_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)

            create_text_w8_matrix_checkpoint(
                source,
                destination,
                parts=[
                    "part4_of_4",
                    "part2_of_4",
                    "part1_of_4",
                    "part3_of_4",
                ],
            )

            converted = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            w8_names = {
                entry["name"]
                for entry in converted["param_encodings"]
                if entry["dtype"] == "INT" and entry["bw"] == 8
            }
            expected = {
                "model.lm_head.weight",
                *(
                    name
                    for layer in range(28)
                    for group in _matrix_names(layer)
                    for name in group
                ),
            }
            self.assertEqual(w8_names, expected)
            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            self.assertEqual(
                marker["selected_parts"],
                [
                    "part1_of_4",
                    "part2_of_4",
                    "part3_of_4",
                    "part4_of_4",
                ],
            )
            self.assertEqual(
                marker["counts"]["matrices_promoted_w4_to_w8"],
                1008,
            )

    def test_can_target_middle_decoder_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)

            create_text_w8_matrix_checkpoint(
                source,
                destination,
                parts=["part3_of_4", "part2_of_4"],
            )

            converted = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            w8_names = {
                entry["name"]
                for entry in converted["param_encodings"]
                if entry["dtype"] == "INT" and entry["bw"] == 8
            }
            expected = {
                "model.lm_head.weight",
                *(
                    name
                    for layer in range(7, 21)
                    for group in _matrix_names(layer)
                    for name in group
                ),
            }
            self.assertEqual(w8_names, expected)
            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            self.assertEqual(
                marker["selected_parts"],
                ["part2_of_4", "part3_of_4"],
            )
            self.assertEqual(
                marker["selected_layers"],
                list(range(7, 21)),
            )
            self.assertEqual(
                marker["counts"]["matrices_promoted_w4_to_w8"],
                504,
            )

    def test_can_target_only_early_fusion_layers_in_part1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)

            create_text_w8_matrix_checkpoint(
                source,
                destination,
                parts=["part1_of_4"],
                layers=[0, 1, 2, 3],
            )

            converted = json.loads(
                (destination / ENCODINGS_FILENAME).read_text()
            )
            w8_names = {
                entry["name"]
                for entry in converted["param_encodings"]
                if entry["dtype"] == "INT" and entry["bw"] == 8
            }
            expected = {
                "model.lm_head.weight",
                *(
                    name
                    for layer in range(4)
                    for group in _matrix_names(layer)
                    for name in group
                ),
            }
            self.assertEqual(w8_names, expected)
            original = json.loads(
                (source / ENCODINGS_FILENAME).read_text()
            )
            original_by_name = {
                entry["name"]: entry
                for entry in original["param_encodings"]
            }
            converted_by_name = {
                entry["name"]: entry
                for entry in converted["param_encodings"]
            }
            layer4_name = _matrix_names(4)[0][0]
            self.assertEqual(
                converted_by_name[layer4_name],
                original_by_name[layer4_name],
            )
            marker = json.loads(
                (destination / MARKER_FILENAME).read_text()
            )
            self.assertEqual(marker["selected_parts"], ["part1_of_4"])
            self.assertEqual(marker["selected_layers"], [0, 1, 2, 3])
            self.assertEqual(marker["selection_mode"], "explicit_layers")
            self.assertEqual(
                marker["counts"]["matrices_promoted_w4_to_w8"],
                4 * EXPECTED_MATRICES_PER_LAYER,
            )

    def test_explicit_layers_must_match_every_selected_part(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)

            with self.assertRaisesRegex(
                ValueError,
                "outside selected parts",
            ):
                create_text_w8_matrix_checkpoint(
                    source,
                    destination,
                    parts=["part1_of_4"],
                    layers=[0, 21],
                )
            with self.assertRaisesRegex(
                ValueError,
                "empty selections",
            ):
                create_text_w8_matrix_checkpoint(
                    source,
                    destination,
                    parts=["part1_of_4", "part4_of_4"],
                    layers=[0, 1, 2, 3],
                )
            self.assertFalse(destination.exists())

    def test_rejects_w4_scale_not_reproducible_from_raw_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)
            path = source / ENCODINGS_FILENAME
            encodings = json.loads(path.read_text())
            target = _matrix_names(0)[0][0]
            for entry in encodings["param_encodings"]:
                if entry["name"] == target:
                    entry["scale"][0] = 0.5
                    break
            path.write_text(json.dumps(encodings))

            with self.assertRaisesRegex(
                ValueError,
                "cannot reproduce source W4 encoding",
            ):
                create_text_w8_matrix_checkpoint(
                    source,
                    destination,
                    parts=["part1_of_4"],
                )
            self.assertFalse(destination.exists())
            self.assertFalse((root / ".destination.partial").exists())

    def test_rejects_non_fp16_activation_or_missing_deepstack(self) -> None:
        mutations = {
            "canonical FLOAT16": self._mutate_activation,
            "input contract mismatch": self._remove_deepstack_input,
        }
        for expected_error, mutation in mutations.items():
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source"
                    destination = root / "destination"
                    _write_checkpoint(source)
                    mutation(source)

                    with self.assertRaisesRegex(
                        ValueError,
                        expected_error,
                    ):
                        create_text_w8_matrix_checkpoint(
                            source,
                            destination,
                            parts=["part1_of_4"],
                        )
                    self.assertFalse(destination.exists())

    @staticmethod
    def _mutate_activation(source: Path) -> None:
        path = source / ENCODINGS_FILENAME
        encodings = json.loads(path.read_text())
        encodings["activation_encodings"][0]["dtype"] = "INT"
        path.write_text(json.dumps(encodings))

    @staticmethod
    def _remove_deepstack_input(source: Path) -> None:
        import onnx

        path = source / MODEL_FILENAME
        model = onnx.load_model(path, load_external_data=False)
        keep = [
            value
            for value in model.graph.input
            if value.name != "deepstack_visual_embeds_2"
        ]
        del model.graph.input[:]
        model.graph.input.extend(keep)
        onnx.save_model(model, path)

    def test_rejects_repeat_transform_and_cleans_copy_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            first = root / "first"
            second = root / "second"
            _write_checkpoint(source)
            create_text_w8_matrix_checkpoint(
                source,
                first,
                parts=["part1_of_4"],
            )
            with self.assertRaisesRegex(ValueError, "always derive"):
                create_text_w8_matrix_checkpoint(
                    first,
                    second,
                    parts=["part4_of_4"],
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_checkpoint(source)
            calls = 0

            def failing_copy(source_path: str, destination_path: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected copy failure")
                return str(shutil.copy2(source_path, destination_path))

            with self.assertRaisesRegex(OSError, "injected copy failure"):
                create_text_w8_matrix_checkpoint(
                    source,
                    destination,
                    parts=["part1_of_4"],
                    copy_function=failing_copy,
                )
            self.assertFalse(destination.exists())
            self.assertFalse((root / ".destination.partial").exists())


if __name__ == "__main__":
    unittest.main()
