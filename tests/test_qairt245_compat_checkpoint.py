from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from scripts.make_qairt245_compat_checkpoint import (
    DEEPSTACK_INPUT_PREFIX,
    MAIN_EMBEDDING_INPUT,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    VISUAL_MASK_INPUT,
    create_compat_checkpoint,
    neutralize_deepstack_inputs,
)

NUM_VISUAL_TOKENS = 4
HIDDEN_SIZE = 8
NUM_DEEPSTACK_LAYERS = 3


def _tensor(
    name: str,
    dtype: int,
    shape: list[int | str],
) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _make_deepstack_model() -> onnx.ModelProto:
    inputs = [
        _tensor(MAIN_EMBEDDING_INPUT, TensorProto.FLOAT, [1, "seq", HIDDEN_SIZE]),
        _tensor("attention_passthrough", TensorProto.FLOAT, [1, "seq"]),
        _tensor(VISUAL_MASK_INPUT, TensorProto.BOOL, [1, "seq"]),
        *[
            _tensor(
                f"{DEEPSTACK_INPUT_PREFIX}{index}",
                TensorProto.FLOAT,
                [NUM_VISUAL_TOKENS, HIDDEN_SIZE],
            )
            for index in range(NUM_DEEPSTACK_LAYERS)
        ],
    ]
    outputs = [
        _tensor("model_output", TensorProto.FLOAT, [1, "seq", HIDDEN_SIZE]),
        _tensor("attention_output", TensorProto.FLOAT, [1, "seq"]),
    ]
    value_info = [
        _tensor("mask_float", TensorProto.FLOAT, [1, "seq"]),
        _tensor("base_0", TensorProto.FLOAT, [1, "seq", HIDDEN_SIZE]),
    ]
    nodes = [
        helper.make_node(
            "Cast",
            [VISUAL_MASK_INPUT],
            ["mask_float"],
            name="mask_cast",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Add",
            [MAIN_EMBEDDING_INPUT, MAIN_EMBEDDING_INPUT],
            ["base_0"],
            name="base_add_0",
        ),
    ]

    base_tensor = "base_0"
    for index in range(NUM_DEEPSTACK_LAYERS):
        auxiliary_tensor = f"deepstack_mul_{index}"
        injected_tensor = f"injected_{index}"
        next_base = (
            "model_output"
            if index == NUM_DEEPSTACK_LAYERS - 1
            else f"base_{index + 1}"
        )
        nodes.extend(
            [
                helper.make_node(
                    "Mul",
                    [f"{DEEPSTACK_INPUT_PREFIX}{index}", "mask_float"],
                    [auxiliary_tensor],
                    name=f"deepstack_zero_{index}",
                ),
                helper.make_node(
                    "Add",
                    [base_tensor, auxiliary_tensor],
                    [injected_tensor],
                    name=f"inject_add_{index}",
                ),
                helper.make_node(
                    "Add",
                    [injected_tensor, MAIN_EMBEDDING_INPUT],
                    [next_base],
                    name=f"base_add_{index + 1}",
                ),
            ]
        )
        value_info.extend(
            [
                _tensor(
                    auxiliary_tensor,
                    TensorProto.FLOAT,
                    [1, "seq", HIDDEN_SIZE],
                ),
                _tensor(
                    injected_tensor,
                    TensorProto.FLOAT,
                    [1, "seq", HIDDEN_SIZE],
                ),
            ]
        )
        if next_base != "model_output":
            value_info.append(
                _tensor(
                    next_base,
                    TensorProto.FLOAT,
                    [1, "seq", HIDDEN_SIZE],
                )
            )
        base_tensor = next_base

    nodes.append(
        helper.make_node(
            "Identity",
            ["attention_passthrough"],
            ["attention_output"],
            name="attention_identity",
        )
    )
    graph = helper.make_graph(
        nodes,
        "synthetic_deepstack",
        inputs,
        outputs,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    return model


def _activation_encodings() -> dict[str, object]:
    names = [
        tensor_name
        for index in range(NUM_DEEPSTACK_LAYERS)
        for tensor_name in (f"deepstack_mul_{index}", f"injected_{index}")
    ]
    return {
        "version": "1.0.0",
        "activation_encodings": [
            {
                "name": name,
                "dtype": "INT",
                "bw": 16,
                "enc_type": "PER_TENSOR",
                "is_sym": False,
                "scale": [0.01],
                "offset": [0.0],
            }
            for name in names
        ],
        "param_encodings": [],
    }


class Qairt245CompatCheckpointTest(unittest.TestCase):
    def test_rewrite_preserves_calibrated_tensor_names_and_prunes_inputs(self) -> None:
        model = _make_deepstack_model()
        original_node_names = {node.name for node in model.graph.node}

        removed = neutralize_deepstack_inputs(
            model,
            num_visual_tokens=NUM_VISUAL_TOKENS,
        )

        self.assertEqual(
            removed,
            [
                VISUAL_MASK_INPUT,
                *[
                    f"{DEEPSTACK_INPUT_PREFIX}{index}"
                    for index in range(NUM_DEEPSTACK_LAYERS)
                ],
            ],
        )
        self.assertEqual(
            [value.name for value in model.graph.input],
            [MAIN_EMBEDDING_INPUT, "attention_passthrough"],
        )
        rewritten = {
            node.name: node
            for node in model.graph.node
            if node.name.startswith("deepstack_zero_")
        }
        self.assertEqual(len(rewritten), NUM_DEEPSTACK_LAYERS)
        for index in range(NUM_DEEPSTACK_LAYERS):
            node = rewritten[f"deepstack_zero_{index}"]
            self.assertEqual(node.op_type, "Sub")
            self.assertEqual(
                list(node.input),
                [MAIN_EMBEDDING_INPUT, MAIN_EMBEDDING_INPUT],
            )
            self.assertEqual(list(node.output), [f"deepstack_mul_{index}"])

            injection = next(
                item
                for item in model.graph.node
                if item.name == f"inject_add_{index}"
            )
            self.assertEqual(injection.op_type, "Add")
            self.assertEqual(list(injection.output), [f"injected_{index}"])
            self.assertIn(f"deepstack_mul_{index}", injection.input)

        retained_node_names = {node.name for node in model.graph.node}
        self.assertLess(retained_node_names, original_node_names)
        self.assertNotIn("mask_cast", retained_node_names)
        self.assertFalse(retained_node_names - original_node_names)
        onnx.checker.check_model(model)

    def test_rewrite_rejects_ambiguous_deepstack_fanout(self) -> None:
        model = _make_deepstack_model()
        model.graph.node.extend(
            [
                helper.make_node(
                    "Mul",
                    [f"{DEEPSTACK_INPUT_PREFIX}0", "mask_float"],
                    ["ambiguous_aux"],
                    name="ambiguous_mul",
                ),
                helper.make_node(
                    "Add",
                    [MAIN_EMBEDDING_INPUT, "ambiguous_aux"],
                    ["ambiguous_output"],
                    name="ambiguous_add",
                ),
            ]
        )
        model.graph.value_info.extend(
            [
                _tensor(
                    "ambiguous_aux",
                    TensorProto.FLOAT,
                    [1, "seq", HIDDEN_SIZE],
                )
            ]
        )
        model.graph.output.extend(
            [
                _tensor(
                    "ambiguous_output",
                    TensorProto.FLOAT,
                    [1, "seq", HIDDEN_SIZE],
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "exactly one first Add"):
            neutralize_deepstack_inputs(
                model,
                num_visual_tokens=NUM_VISUAL_TOKENS,
            )

    def test_rewrite_rejects_injection_shape_mismatch(self) -> None:
        model = _make_deepstack_model()
        auxiliary = next(
            value
            for value in model.graph.value_info
            if value.name == "deepstack_mul_1"
        )
        auxiliary.type.tensor_type.shape.dim[2].dim_value = HIDDEN_SIZE - 1

        with self.assertRaisesRegex(ValueError, "has shape"):
            neutralize_deepstack_inputs(
                model,
                num_visual_tokens=NUM_VISUAL_TOKENS,
            )

    def test_rewrite_rejects_pruning_unrelated_input(self) -> None:
        model = _make_deepstack_model()
        model.graph.input.extend(
            [_tensor("unexpected_unused", TensorProto.FLOAT, [1])]
        )

        with self.assertRaisesRegex(ValueError, "unexpected graph inputs"):
            neutralize_deepstack_inputs(
                model,
                num_visual_tokens=NUM_VISUAL_TOKENS,
            )

    def test_checkpoint_manifest_and_encodings_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "compat"
            source.mkdir()
            onnx.save(_make_deepstack_model(), source / MODEL_FILENAME)
            encodings_path = source / "model.encodings"
            encodings_text = json.dumps(_activation_encodings(), sort_keys=True)
            encodings_path.write_text(encodings_text, encoding="utf-8")

            output = create_compat_checkpoint(
                source,
                destination,
                num_visual_tokens=NUM_VISUAL_TOKENS,
            )

            manifest = json.loads(
                (output / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["removed_runtime_inputs"],
                [
                    VISUAL_MASK_INPUT,
                    *[
                        f"{DEEPSTACK_INPUT_PREFIX}{index}"
                        for index in range(NUM_DEEPSTACK_LAYERS)
                    ],
                ],
            )
            self.assertEqual(
                (output / "model.encodings").read_text(encoding="utf-8"),
                encodings_text,
            )
            output_model = onnx.load(output / MODEL_FILENAME)
            self.assertEqual(
                [value.name for value in output_model.graph.input],
                [MAIN_EMBEDDING_INPUT, "attention_passthrough"],
            )
            onnx.checker.check_model(output_model)

    def test_checkpoint_rejects_missing_preserved_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "compat"
            source.mkdir()
            onnx.save(_make_deepstack_model(), source / MODEL_FILENAME)
            encodings = _activation_encodings()
            assert isinstance(encodings["activation_encodings"], list)
            encodings["activation_encodings"].pop()
            (source / "model.encodings").write_text(
                json.dumps(encodings),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing rewritten injection tensors",
            ):
                create_compat_checkpoint(
                    source,
                    destination,
                    num_visual_tokens=NUM_VISUAL_TOKENS,
                )
            self.assertFalse(destination.exists())
            self.assertFalse((root / ".compat.partial").exists())


if __name__ == "__main__":
    unittest.main()
