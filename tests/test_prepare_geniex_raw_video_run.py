from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import prepare_geniex_raw_video_run as raw_run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sparse(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size)


def _tensor(shape: list[int], dtype: str = "uint16") -> dict[str, object]:
    return {"shape": shape, "dtype": dtype}


def _write_bundle(
    root: Path,
    *,
    include_deepstack: bool = True,
    unsafe_input_order: bool = False,
) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    part1_inputs: dict[str, object] = {
        "past_key_0_in": _tensor([8, 1, 128, 511], "uint8"),
        "past_value_0_in": _tensor([8, 1, 511, 128], "uint8"),
    }
    deepstack_inputs = {
        "visual_pos_masks": _tensor([1, 128], "bool8"),
        "deepstack_visual_embeds_0": _tensor([128, 2048]),
        "deepstack_visual_embeds_1": _tensor([128, 2048]),
        "deepstack_visual_embeds_2": _tensor([128, 2048]),
    }
    if include_deepstack and unsafe_input_order:
        part1_inputs.update(deepstack_inputs)
    part1_inputs.update(
        {
            "inputs_embeds": _tensor([1, 128, 2048]),
            "position_ids_cos": _tensor([1, 1, 128, 64]),
            "position_ids_sin": _tensor([1, 1, 128, 64]),
            "attention_mask": _tensor([1, 1, 128, 512]),
        }
    )
    if include_deepstack and not unsafe_input_order:
        part1_inputs.update(deepstack_inputs)

    vision_outputs = {
        "image_features": _tensor([256, 2048]),
        "deepstack_visual_embeds_0": _tensor([256, 2048]),
        "deepstack_visual_embeds_1": _tensor([256, 2048]),
        "deepstack_visual_embeds_2": _tensor([256, 2048]),
    }
    metadata = {
        "model_files": {
            "vision_encoder.bin": {
                "inputs": {
                    "pixel_values": _tensor([1024, 1536]),
                    "position_ids_cos": _tensor([1024, 32]),
                    "position_ids_sin": _tensor([1024, 32]),
                    "window_attention_mask": _tensor([1, 1024, 1024]),
                    "full_attention_mask": _tensor([1, 1024, 1024]),
                },
                "outputs": vision_outputs,
            },
            "part1_of_4.bin": {
                "inputs": part1_inputs,
                "outputs": {
                    "hidden_1": _tensor([1, 128, 2048]),
                    "past_key_0_out": _tensor([8, 1, 128, 128]),
                },
            },
            **{
                f"part{index}_of_4.bin": {
                    "inputs": {
                        f"past_key_{index}_in": _tensor(
                            [8, 1, 128, 511], "uint8"
                        ),
                        f"past_value_{index}_in": _tensor(
                            [8, 1, 511, 128], "uint8"
                        ),
                        f"hidden_{index - 1}": _tensor([1, 128, 2048]),
                        "position_ids_cos": _tensor([1, 1, 128, 64]),
                        "position_ids_sin": _tensor([1, 1, 128, 64]),
                        "attention_mask": _tensor([1, 1, 128, 512]),
                    },
                    "outputs": {
                        (
                            "logits" if index == 4 else f"hidden_{index}"
                        ): _tensor(
                            [1, 128, 151936 if index == 4 else 2048]
                        )
                    },
                }
                for index in range(2, 5)
            },
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
            "context": {"size": 512, "n-vocab": 151936},
            "embedding": {
                "size": 2048,
                "datatype": "float32",
                "lut-path": "embedding_weights.raw",
            },
            "engine": {
                "model": {
                    "binary": {
                        "ctx-bins": [
                            "part1_of_4.bin",
                            "part2_of_4.bin",
                            "part3_of_4.bin",
                            "part4_of_4.bin",
                        ]
                    },
                    "positional-encoding": {
                        "rope-scaling": {
                            "rope-type": "qwen3vl-mrope",
                            "mrope-section": [24, 20, 20],
                            "spatial-merge-size": 2,
                        }
                    },
                }
            },
        }
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "genie_config.json").write_text(json.dumps(config), encoding="utf-8")
    (bundle / "htp_backend_ext_config.json").write_text("{}", encoding="utf-8")
    (bundle / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_sparse(
        bundle / "embedding_weights.raw",
        raw_run.EXPECTED_VOCAB_SIZE * raw_run.EXPECTED_HIDDEN_SIZE * 4,
    )
    for filename in raw_run.CONTEXT_FILES:
        (bundle / filename).write_bytes(filename.encode())
    return bundle


def _write_video_inputs(root: Path) -> Path:
    video = root / "video"
    inputs = video / "sample_inputs"
    inputs.mkdir(parents=True)
    pixel = inputs / "pair_000_pixel_values.raw"
    _write_sparse(pixel, raw_run.EXPECTED_PIXEL_BYTES)
    prefix = inputs / "video_prefix_pair_000.txt"
    prefix.write_text(
        "<|im_start|>system\nsafe<|im_end|>\n"
        "<|im_start|>user\n<2.5 seconds><|vision_start|>",
        encoding="utf-8",
    )
    suffix = inputs / "video_suffix.txt"
    suffix.write_text(
        "<|vision_end|>What happens next?<|im_end|>\n"
        "<|im_start|>assistant\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "graph_contract": {
            "frames_per_pair": 2,
            "image_size": [512, 512],
            "pixel_values_shape": [1024, 1536],
            "grid_thw": [1, 32, 32],
            "image_features_shape": [256, 2048],
            "visual_tokens_per_pair": 256,
        },
        "context_budget": {
            "text_tokens": 20,
            "visual_tokens": 256,
            "prompt_tokens": 276,
        },
        "pairs": [
            {
                "pixel_values": {
                    "file": "sample_inputs/pair_000_pixel_values.raw",
                    "sha256": _sha256(pixel),
                    "bytes": raw_run.EXPECTED_PIXEL_BYTES,
                    "shape": [1024, 1536],
                    "grid_thw": [1, 32, 32],
                }
            }
        ],
        "text_chunks": [
            {
                "file": "sample_inputs/video_prefix_pair_000.txt",
                "sha256": _sha256(prefix),
                "tokens": 11,
                "text": prefix.read_text(encoding="utf-8"),
            },
            {
                "file": "sample_inputs/video_suffix.txt",
                "sha256": _sha256(suffix),
                "tokens": 9,
                "text": suffix.read_text(encoding="utf-8"),
            },
        ],
    }
    (video / "video_npu_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return video


class PrepareGenieXRawVideoRunTests(unittest.TestCase):
    def _hash_without_reading_sparse_embedding(self, path: Path) -> str:
        if path.name == "embedding_weights.raw":
            return "e" * 64
        return _sha256(path)

    def test_prepares_and_verifies_bound_full_deepstack_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root)
            video = _write_video_inputs(root)
            package = root / "package"
            with mock.patch.object(
                raw_run,
                "sha256_file",
                side_effect=self._hash_without_reading_sparse_embedding,
            ):
                raw_run.prepare_run_package(bundle, video, package)
                manifest, paths = raw_run.verify_run_package(
                    package, bundle, max_tokens=64
                )

            self.assertEqual(
                manifest["target_bundle"]["contract"]["deepstack_levels"], 3
            )
            self.assertEqual(
                manifest["target_bundle"]["contract"]["mrope_interleaving"],
                "stride",
            )
            self.assertEqual(manifest["input"]["expected_prompt_tokens"], 276)
            self.assertEqual(
                paths["pixel_values"].stat().st_size,
                raw_run.EXPECTED_PIXEL_BYTES,
            )

    def test_rejects_compatibility_bundle_without_text_deepstack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root, include_deepstack=False)
            video = _write_video_inputs(root)
            with self.assertRaisesRegex(ValueError, "not a full-DeepStack"):
                raw_run.prepare_run_package(
                    bundle, video, root / "should-not-exist"
                )

    def test_rejects_unsafe_part1_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root, unsafe_input_order=True)
            video = _write_video_inputs(root)
            with self.assertRaisesRegex(ValueError, "Unsafe part1 QNN input order"):
                raw_run.prepare_run_package(
                    bundle, video, root / "should-not-exist"
                )

    def test_rejects_tampered_package_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root)
            video = _write_video_inputs(root)
            package = root / "package"
            with mock.patch.object(
                raw_run,
                "sha256_file",
                side_effect=self._hash_without_reading_sparse_embedding,
            ):
                raw_run.prepare_run_package(bundle, video, package)
                pixel = package / "inputs" / "pair_000_pixel_values.raw"
                with pixel.open("r+b") as handle:
                    handle.write(b"x")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    raw_run.verify_run_package(
                        package, bundle, max_tokens=64
                    )

    def test_rejects_prompt_plus_generation_over_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root)
            video = _write_video_inputs(root)
            package = root / "package"
            with mock.patch.object(
                raw_run,
                "sha256_file",
                side_effect=self._hash_without_reading_sparse_embedding,
            ):
                raw_run.prepare_run_package(bundle, video, package)
                with self.assertRaisesRegex(ValueError, "exceeds CL512"):
                    raw_run.verify_run_package(
                        package, bundle, max_tokens=237
                    )

    def test_rejects_source_manifest_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root)
            video = _write_video_inputs(root)
            manifest_path = video / "video_npu_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pairs"][0]["pixel_values"]["file"] = "../outside.raw"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                raw_run.prepare_run_package(
                    bundle, video, root / "should-not-exist"
                )


if __name__ == "__main__":
    unittest.main()
