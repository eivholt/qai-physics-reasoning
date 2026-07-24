from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_video_npu_inputs import (
    ANCILLARY_SPECS,
    EXPECTED_GRID_THW,
    EXPECTED_PIXEL_SHAPE,
    FLOAT32_BYTES,
    NATIVE_RESIZE_MODE,
    STATIC_RESIZE_FALLBACK_MODE,
    TEXT_DECODE_AR,
    TEXT_PREFILL_AR,
    prepare_video_npu_inputs,
)


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        payload: bytes | None = None,
        values: list[list[int]] | None = None,
    ) -> None:
        self.shape = shape
        self._payload = payload
        self._values = values

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return self

    def contiguous(self) -> _FakeTensor:
        return self

    def numpy(self) -> _FakeTensor:
        return self

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C" or self._payload is None:
            raise ValueError("unexpected serialization request")
        return self._payload

    def tolist(self) -> list[list[int]]:
        if self._values is None:
            raise ValueError("no list values")
        return self._values


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("video text must not add implicit special tokens")
        return [1, 2, 3, 4, 5]


class _FakeVideoProcessor:
    def __init__(
        self,
        *,
        pixel_shape: tuple[int, ...] = EXPECTED_PIXEL_SHAPE,
        grid: tuple[int, ...] = EXPECTED_GRID_THW,
        native_pixel_shape: tuple[int, ...] | None = None,
        native_grid: tuple[int, ...] | None = None,
    ) -> None:
        self.pixel_shape = pixel_shape
        self.grid = grid
        self.native_pixel_shape = native_pixel_shape
        self.native_grid = native_grid
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, _FakeTensor]:
        self.calls.append(kwargs)
        call_number = len(self.calls)
        uses_static_fallback = kwargs.get("do_resize") is False
        pixel_shape = (
            self.pixel_shape
            if uses_static_fallback or self.native_pixel_shape is None
            else self.native_pixel_shape
        )
        grid = (
            self.grid
            if uses_static_fallback or self.native_grid is None
            else self.native_grid
        )
        byte_count = math.prod(pixel_shape) * FLOAT32_BYTES
        payload = bytes([call_number]) * byte_count
        return {
            "pixel_values_videos": _FakeTensor(
                pixel_shape,
                payload=payload,
            ),
            "video_grid_thw": _FakeTensor(
                (1, 3),
                values=[list(grid)],
            ),
        }


class _FakeProcessor:
    def __init__(self, video_processor: _FakeVideoProcessor) -> None:
        self.video_processor = video_processor
        self.tokenizer = _FakeTokenizer()


def _write_sparse(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size)


def _write_bundle(
    bundle: Path,
    *,
    context_size: int,
    image_height: int = 512,
    image_width: int = 512,
) -> None:
    bundle.mkdir()
    sample_inputs = bundle / "sample_inputs"
    sample_inputs.mkdir()
    patch_size = 16
    merge_size = 2
    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    patches = grid_height * grid_width
    visual_tokens = patches // (merge_size**2)
    pixel_shape = [patches, 3 * 2 * patch_size * patch_size]
    ancillary_specs = {
        "position_ids_cos.raw": (patches, 32),
        "position_ids_sin.raw": (patches, 32),
        "window_attention_mask.raw": (1, patches, patches),
        "full_attention_mask.raw": (1, patches, patches),
    }
    metadata = {
        "model_files": {
            "vision_encoder.bin": {
                "inputs": {
                    "pixel_values": {"shape": pixel_shape},
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
                    "image_features": {
                        "shape": [visual_tokens, 2048]
                    },
                },
            },
            **{
                f"part{index}_of_4.bin": {
                    "inputs": {
                        "past_key_0_in": {
                            "shape": [8, 1, 128, context_size - 1]
                        }
                    }
                }
                for index in range(1, 5)
            },
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
    config = {"dialog": {"context": {"size": context_size}}}
    text_generator = {
        "text-generator": {"context": {"size": context_size}}
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "genie_config.json").write_text(json.dumps(config), encoding="utf-8")
    (bundle / "text-generator.json").write_text(
        json.dumps(text_generator), encoding="utf-8"
    )
    (bundle / "img-enc-htp.json").write_text(
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
    (bundle / "text-encoder.json").write_text("{}", encoding="utf-8")
    (bundle / "htp_backend_ext_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (bundle / "vision_encoder.bin").write_bytes(b"vision")
    for index in range(1, 5):
        (bundle / f"part{index}_of_4.bin").write_bytes(
            f"text-part-{index}".encode()
        )
    for filename, shape in ancillary_specs.items():
        _write_sparse(
            sample_inputs / filename,
            math.prod(shape) * FLOAT32_BYTES,
        )


def _write_frames(root: Path, count: int) -> list[Path]:
    frames: list[Path] = []
    for index in range(count):
        path = root / f"frame_{index:03d}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(path)
    return frames


def _write_processor_dir(root: Path) -> Path:
    processor = root / "processor"
    processor.mkdir()
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
    ):
        (processor / filename).write_text(
            json.dumps({"fixture": filename}),
            encoding="utf-8",
        )
    return processor


class PrepareVideoNpuInputsTests(unittest.TestCase):
    def test_prepares_one_pair_with_native_processor_resize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 2)
            video_processor = _FakeVideoProcessor()
            processor = _FakeProcessor(video_processor)
            loaded_sizes: list[tuple[int, int] | None] = []

            def frame_loader(
                path: Path,
                size: tuple[int, int] | None,
            ) -> str:
                loaded_sizes.append(size)
                return path.name

            output = bundle / "video_inputs" / "one_pair"
            prepare_video_npu_inputs(
                bundle,
                processor_path,
                output,
                frames,
                [0.0, 0.25],
                processor_loader=lambda _: processor,
                frame_loader=frame_loader,
            )

            self.assertEqual(loaded_sizes, [None, None])
            self.assertEqual(len(video_processor.calls), 1)
            call = video_processor.calls[0]
            self.assertNotIn("do_resize", call)
            self.assertFalse(call["do_sample_frames"])
            manifest = json.loads(
                (output / "video_npu_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["context_budget"]["text_tokens"], 10)
            self.assertEqual(manifest["context_budget"]["visual_tokens"], 256)
            self.assertEqual(manifest["context_budget"]["prompt_tokens"], 266)
            self.assertEqual(
                manifest["bundle"]["text_runtime_contract"],
                {
                    "prefill_ar": TEXT_PREFILL_AR,
                    "decode_ar": TEXT_DECODE_AR,
                    "prefill_kv_capacity": 384,
                    "max_safe_prompt_tokens": 384,
                },
            )
            self.assertEqual(
                manifest["context_budget"]["prefill_safe_limit"], 384
            )
            self.assertEqual(
                manifest["context_budget"]["remaining_prefill_capacity"], 118
            )
            self.assertEqual(manifest["pairs"][0]["pair_timestamp_text"], "0.1")
            self.assertEqual(
                manifest["frames"][0]["sha256"],
                hashlib.sha256(frames[0].read_bytes()).hexdigest(),
            )
            self.assertFalse(
                manifest["processor"]["trust_remote_code"]
            )
            self.assertEqual(
                manifest["processor"]["resize_mode"],
                NATIVE_RESIZE_MODE,
            )
            self.assertEqual(
                manifest["pairs"][0]["resize_mode"],
                NATIVE_RESIZE_MODE,
            )
            self.assertIn(
                "vision_encoder.bin", manifest["bundle"]["files"]
            )
            self.assertIn(
                "part4_of_4.bin", manifest["bundle"]["files"]
            )

            script = (output / "genie-video-app-script.txt").read_text()
            self.assertEqual(
                script.count(
                    "GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT"
                ),
                1,
            )
            self.assertNotIn("<|video_pad|>", script)
            self.assertNotIn("<|image_pad|>", script)
            prefix = (
                output / "sample_inputs" / "video_prefix_pair_000.txt"
            ).read_text()
            self.assertIn("<0.1 seconds><|vision_start|>", prefix)
            for filename in ANCILLARY_SPECS:
                self.assertEqual(
                    (output / "sample_inputs" / filename).read_bytes(),
                    (bundle / "sample_inputs" / filename).read_bytes(),
                )

    def test_falls_back_to_explicit_resize_for_static_square_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 2)
            video_processor = _FakeVideoProcessor(
                native_pixel_shape=(336, 1536),
                native_grid=(1, 14, 24),
            )
            processor = _FakeProcessor(video_processor)
            loaded_sizes: list[tuple[int, int] | None] = []

            def frame_loader(
                path: Path,
                size: tuple[int, int] | None,
            ) -> str:
                loaded_sizes.append(size)
                return path.name

            output = bundle / "video_inputs" / "fallback"
            prepare_video_npu_inputs(
                bundle,
                processor_path,
                output,
                frames,
                [0.0, 0.25],
                processor_loader=lambda _: processor,
                frame_loader=frame_loader,
            )

            self.assertEqual(
                loaded_sizes,
                [None, None, (512, 512), (512, 512)],
            )
            self.assertEqual(len(video_processor.calls), 2)
            self.assertNotIn("do_resize", video_processor.calls[0])
            self.assertFalse(video_processor.calls[1]["do_resize"])
            manifest = json.loads(
                (output / "video_npu_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["processor"]["resize_mode"],
                STATIC_RESIZE_FALLBACK_MODE,
            )
            self.assertEqual(
                manifest["processor"]["pair_resize_modes"],
                [STATIC_RESIZE_FALLBACK_MODE],
            )
            self.assertEqual(
                manifest["pairs"][0]["resize_mode"],
                STATIC_RESIZE_FALLBACK_MODE,
            )

    def test_prepares_two_pairs_for_future_cl1024(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=1024)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 4)
            video_processor = _FakeVideoProcessor()
            processor = _FakeProcessor(video_processor)
            output = bundle / "video_inputs" / "two_pairs"

            prepare_video_npu_inputs(
                bundle,
                processor_path,
                output,
                frames,
                [0.0, 0.25, 0.5, 0.75],
                processor_loader=lambda _: processor,
                frame_loader=lambda path, size: (path.name, size),
            )

            manifest = json.loads(
                (output / "video_npu_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["context_budget"]["text_tokens"], 15)
            self.assertEqual(manifest["context_budget"]["visual_tokens"], 512)
            self.assertEqual(manifest["context_budget"]["prompt_tokens"], 527)
            self.assertEqual(len(manifest["pairs"]), 2)
            self.assertNotEqual(
                manifest["pairs"][0]["pixel_values"]["sha256"],
                manifest["pairs"][1]["pixel_values"]["sha256"],
            )
            bridge = (
                output / "sample_inputs" / "video_bridge_pair_001.txt"
            ).read_text()
            self.assertEqual(
                bridge,
                "<|vision_end|><0.6 seconds><|vision_start|>",
            )
            script = (output / "genie-video-app-script.txt").read_text()
            self.assertEqual(
                script.count("GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT"),
                2,
            )

    def test_prepares_four_aspect_preserving_pairs_for_cl512(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(
                bundle,
                context_size=512,
                image_height=224,
                image_width=384,
            )
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 8)
            video_processor = _FakeVideoProcessor(
                pixel_shape=(336, 1536),
                grid=(1, 14, 24),
            )
            processor = _FakeProcessor(video_processor)
            loaded_sizes: list[tuple[int, int] | None] = []

            def frame_loader(
                path: Path,
                size: tuple[int, int] | None,
            ) -> str:
                loaded_sizes.append(size)
                return path.name

            output = bundle / "video_inputs" / "four_aspect_pairs"
            prepare_video_npu_inputs(
                bundle,
                processor_path,
                output,
                frames,
                [index * 0.25 for index in range(8)],
                processor_loader=lambda _: processor,
                frame_loader=frame_loader,
            )

            self.assertEqual(loaded_sizes, [None] * 8)
            self.assertEqual(len(video_processor.calls), 4)
            manifest = json.loads(
                (output / "video_npu_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["graph_contract"],
                {
                    "frames_per_pair": 2,
                    "image_size": [384, 224],
                    "pixel_values_shape": [336, 1536],
                    "grid_thw": [1, 14, 24],
                    "image_features_shape": [84, 2048],
                    "visual_tokens_per_pair": 84,
                },
            )
            self.assertEqual(manifest["context_budget"]["text_tokens"], 25)
            self.assertEqual(manifest["context_budget"]["visual_tokens"], 336)
            self.assertEqual(manifest["context_budget"]["prompt_tokens"], 361)
            self.assertEqual(
                manifest["context_budget"]["remaining_prefill_capacity"], 23
            )
            self.assertEqual(len(manifest["pairs"]), 4)
            self.assertEqual(
                manifest["processor"]["resize_mode"],
                NATIVE_RESIZE_MODE,
            )
            self.assertEqual(
                (output / "sample_inputs" / "pair_003_pixel_values.raw").stat().st_size,
                336 * 1536 * FLOAT32_BYTES,
            )

    def test_rejects_two_512_pairs_before_packing_for_cl512(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 4)
            video_processor = _FakeVideoProcessor()
            output = bundle / "video_inputs" / "overflow"

            with self.assertRaisesRegex(
                ValueError, "exceeds the safe GenieX prefill cache"
            ):
                prepare_video_npu_inputs(
                    bundle,
                    processor_path,
                    output,
                    frames,
                    [0.0, 0.25, 0.5, 0.75],
                    processor_loader=lambda _: _FakeProcessor(video_processor),
                    frame_loader=lambda path, size: path,
                )

            self.assertEqual(video_processor.calls, [])
            self.assertFalse(output.exists())

    def test_rejects_prompt_above_cl_minus_ar_before_packing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(
                bundle,
                context_size=512,
                image_height=224,
                image_width=384,
            )
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 8)
            video_processor = _FakeVideoProcessor(
                pixel_shape=(336, 1536),
                grid=(1, 14, 24),
            )
            processor = _FakeProcessor(video_processor)

            class _WideTokenizer:
                def encode(
                    self, text: str, *, add_special_tokens: bool
                ) -> list[int]:
                    if add_special_tokens:
                        raise AssertionError(
                            "video text must not add implicit special tokens"
                        )
                    return list(range(11))

            processor.tokenizer = _WideTokenizer()
            output = bundle / "video_inputs" / "unsafe_prefill"
            with self.assertRaisesRegex(
                ValueError,
                r"AR128/CL512 permits at most 384 prompt tokens",
            ):
                prepare_video_npu_inputs(
                    bundle,
                    processor_path,
                    output,
                    frames,
                    [index * 0.25 for index in range(8)],
                    processor_loader=lambda _: processor,
                    frame_loader=lambda path, size: path,
                )

            self.assertEqual(video_processor.calls, [])
            self.assertFalse(output.exists())

    def test_rejects_processor_shape_mismatch_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 2)
            video_processor = _FakeVideoProcessor(pixel_shape=(336, 1536))
            output = bundle / "video_inputs" / "bad_shape"

            with self.assertRaisesRegex(ValueError, "processor shape"):
                prepare_video_npu_inputs(
                    bundle,
                    processor_path,
                    output,
                    frames,
                    [0.0, 0.25],
                    processor_loader=lambda _: _FakeProcessor(video_processor),
                    frame_loader=lambda path, size: path,
                )

            self.assertFalse(output.exists())
            self.assertFalse(
                output.with_name(f".{output.name}.partial").exists()
            )

    def test_rejects_odd_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 3)

            with self.assertRaisesRegex(ValueError, "2N"):
                prepare_video_npu_inputs(
                    bundle,
                    processor_path,
                    bundle / "video_inputs" / "odd",
                    frames,
                    [0.0, 0.25, 0.5],
                    processor_loader=lambda _: _FakeProcessor(
                        _FakeVideoProcessor()
                    ),
                )

    def test_rejects_context_mismatch_between_runtime_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            (bundle / "text-generator.json").write_text(
                json.dumps(
                    {"text-generator": {"context": {"size": 1024}}}
                ),
                encoding="utf-8",
            )
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 2)

            with self.assertRaisesRegex(ValueError, "context mismatch"):
                prepare_video_npu_inputs(
                    bundle,
                    processor_path,
                    bundle / "video_inputs" / "mismatch",
                    frames,
                    [0.0, 0.25],
                    processor_loader=lambda _: _FakeProcessor(
                        _FakeVideoProcessor()
                    ),
                )

    def test_rejects_output_outside_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            _write_bundle(bundle, context_size=512)
            processor_path = _write_processor_dir(root)
            frames = _write_frames(root, 2)

            with self.assertRaisesRegex(ValueError, "inside the bundle"):
                prepare_video_npu_inputs(
                    bundle,
                    processor_path,
                    root / "standalone-output",
                    frames,
                    [0.0, 0.25],
                    processor_loader=lambda _: _FakeProcessor(
                        _FakeVideoProcessor()
                    ),
                )


if __name__ == "__main__":
    unittest.main()
