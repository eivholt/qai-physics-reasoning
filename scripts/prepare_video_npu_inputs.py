#!/usr/bin/env python3
"""Prepare paired video-frame inputs for the legacy Cosmos Genie NPU pipeline.

The deployed Qwen3-VL vision graph consumes one temporal patch at a time:
exactly two frames, represented by a static ``[1024, 1536]`` tensor.  This
tool packs any even number of pre-extracted frames into such pairs, surrounds
each pair with the timestamped Qwen3-VL text layout, and emits a complete
``genie-app`` script that repeatedly invokes the same NPU vision context.

Video decoding is intentionally out of scope.  Supply already extracted frame
files and their source-video timestamps.  Run the generated script with the
bundle directory as the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

EXPECTED_PIXEL_SHAPE = (1024, 1536)
EXPECTED_GRID_THW = (1, 32, 32)
EXPECTED_IMAGE_SIZE = (512, 512)
EXPECTED_VISUAL_OUTPUT_SHAPE = (256, 2048)
VISUAL_TOKENS_PER_PAIR = 256
FLOAT32_BYTES = 4

ANCILLARY_SPECS: dict[str, tuple[int, ...]] = {
    "position_ids_cos.raw": (1024, 32),
    "position_ids_sin.raw": (1024, 32),
    "window_attention_mask.raw": (1, 1024, 1024),
    "full_attention_mask.raw": (1, 1024, 1024),
}

NODE_CONFIG_FILES = (
    "img-enc-htp.json",
    "text-encoder.json",
    "text-generator.json",
)

CONTEXT_BINARY_FILES = (
    "vision_encoder.bin",
    "part1_of_4.bin",
    "part2_of_4.bin",
    "part3_of_4.bin",
    "part4_of_4.bin",
)

REQUIRED_BUNDLE_FILES = (
    "metadata.json",
    "genie_config.json",
    "htp_backend_ext_config.json",
    *NODE_CONFIG_FILES,
    *CONTEXT_BINARY_FILES,
)

PROCESSOR_PROVENANCE_FILES = (
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
)


@dataclass(frozen=True)
class PackedPair:
    """Serialized processor output for one two-frame temporal patch."""

    pixel_values: bytes
    shape: tuple[int, ...]
    grid_thw: tuple[int, ...]


@dataclass(frozen=True)
class BundleContract:
    context_size: int
    image_width: int
    image_height: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _shape_from_metadata(
    tensors: dict[str, Any], name: str, label: str
) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in tensors[name]["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"metadata.json has no valid {label} shape for {name}") from exc
    return shape


def _validate_bundle(bundle: Path) -> BundleContract:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle / name).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing required files: {missing}")

    metadata = _load_json(bundle / "metadata.json", "metadata.json")
    config = _load_json(bundle / "genie_config.json", "genie_config.json")
    text_generator = _load_json(
        bundle / "text-generator.json", "text-generator.json"
    )
    try:
        vision_file = metadata["model_files"]["vision_encoder.bin"]
        vision_inputs = vision_file["inputs"]
        vision_outputs = vision_file["outputs"]
        preprocessing = metadata["genie"]["vision_preprocessing"]
        dialog_context_size = int(config["dialog"]["context"]["size"])
        context_size = int(
            text_generator["text-generator"]["context"]["size"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Bundle metadata/config does not match the Cosmos Genie schema") from exc
    if dialog_context_size != context_size:
        raise ValueError(
            "Bundle context mismatch: genie_config.json declares "
            f"{dialog_context_size}, while text-generator.json declares "
            f"{context_size}"
        )

    expected_inputs = {
        "pixel_values": EXPECTED_PIXEL_SHAPE,
        "position_ids_cos": ANCILLARY_SPECS["position_ids_cos.raw"],
        "position_ids_sin": ANCILLARY_SPECS["position_ids_sin.raw"],
        "window_attention_mask": ANCILLARY_SPECS[
            "window_attention_mask.raw"
        ],
        "full_attention_mask": ANCILLARY_SPECS["full_attention_mask.raw"],
    }
    for name, expected in expected_inputs.items():
        actual = _shape_from_metadata(vision_inputs, name, "vision input")
        if actual != expected:
            raise ValueError(
                f"Unsupported vision input shape for {name}: {actual}; "
                f"expected {expected}"
            )

    output_shape = _shape_from_metadata(
        vision_outputs, "image_features", "vision output"
    )
    if output_shape != EXPECTED_VISUAL_OUTPUT_SHAPE:
        raise ValueError(
            f"Unsupported image_features shape: {output_shape}; "
            f"expected {EXPECTED_VISUAL_OUTPUT_SHAPE}"
        )

    expected_preprocessing = {
        "image_width": EXPECTED_IMAGE_SIZE[0],
        "image_height": EXPECTED_IMAGE_SIZE[1],
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }
    mismatches = {
        key: (preprocessing.get(key), expected)
        for key, expected in expected_preprocessing.items()
        if preprocessing.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Unsupported bundle vision preprocessing values: {mismatches}"
        )
    if context_size <= 0:
        raise ValueError("text-generator.json context size must be positive")

    expected_cache_length = context_size - 1
    for filename in CONTEXT_BINARY_FILES[1:]:
        try:
            part_inputs = metadata["model_files"][filename]["inputs"]
            cache_name = next(
                name
                for name in part_inputs
                if name.startswith("past_key_") and name.endswith("_in")
            )
            cache_shape = tuple(
                int(value)
                for value in part_inputs[cache_name]["shape"]
            )
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata.json has no valid cache shape for {filename}"
            ) from exc
        if not cache_shape or cache_shape[-1] != expected_cache_length:
            raise ValueError(
                f"Compiled cache length for {filename} is "
                f"{cache_shape[-1] if cache_shape else None}; expected "
                f"{expected_cache_length} for context {context_size}"
            )

    sample_inputs = bundle / "sample_inputs"
    for filename, shape in ANCILLARY_SPECS.items():
        path = sample_inputs / filename
        if not path.is_file():
            raise ValueError(f"Bundle is missing ancillary tensor: {path}")
        expected_size = math.prod(shape) * FLOAT32_BYTES
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"Ancillary tensor {filename} has {path.stat().st_size} bytes; "
                f"expected {expected_size}"
            )

    return BundleContract(
        context_size=context_size,
        image_width=int(preprocessing["image_width"]),
        image_height=int(preprocessing["image_height"]),
    )


def _validate_frames_and_timestamps(
    frame_paths: Sequence[Path], timestamps: Sequence[float]
) -> tuple[list[Path], list[float]]:
    if not frame_paths or len(frame_paths) % 2:
        raise ValueError("Pass exactly 2N frame paths (an even, non-zero count)")
    if len(frame_paths) != len(timestamps):
        raise ValueError(
            "The number of timestamps must exactly match the number of frames"
        )

    resolved_frames: list[Path] = []
    normalized_timestamps: list[float] = []
    previous: float | None = None
    for index, (path, timestamp) in enumerate(zip(frame_paths, timestamps, strict=True)):
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"Frame {index} does not exist: {resolved}")
        value = float(timestamp)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Timestamp {index} must be finite and non-negative")
        if previous is not None and value <= previous:
            raise ValueError("Frame timestamps must be strictly increasing")
        resolved_frames.append(resolved)
        normalized_timestamps.append(value)
        previous = value
    return resolved_frames, normalized_timestamps


def _load_local_processor(processor_path: Path) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; run this script in the Cosmos QAIHM environment"
        ) from exc
    return AutoProcessor.from_pretrained(
        str(processor_path),
        local_files_only=True,
        trust_remote_code=False,
    )


def _load_frame(path: Path, image_size: tuple[int, int]) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load extracted video frames") from exc

    with Image.open(path) as source:
        rgb = source.convert("RGB")
        resampling = getattr(Image, "Resampling", Image)
        return rgb.resize(image_size, resample=resampling.BICUBIC).copy()


def _mapping_value(mapping: Any, key: str) -> Any:
    try:
        return mapping[key]
    except (KeyError, TypeError):
        try:
            return getattr(mapping, key)
        except AttributeError as exc:
            raise ValueError(f"Processor output is missing {key}") from exc


def _tensor_shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Processor pixel output has no valid shape") from exc


def _tensor_to_float32_bytes(value: Any) -> bytes:
    try:
        array = value.detach().cpu().float().contiguous().numpy()
        return array.tobytes(order="C")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Cannot serialize processor pixel output as float32") from exc


def _grid_tuple(value: Any) -> tuple[int, ...]:
    try:
        rows = value.detach().cpu().tolist()
    except AttributeError:
        try:
            rows = value.tolist()
        except AttributeError as exc:
            raise ValueError("Processor video_grid_thw is not tensor-like") from exc
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ValueError(f"Expected one video grid row, got {rows!r}")
    try:
        return tuple(int(item) for item in rows[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid video grid values: {rows!r}") from exc


def _pack_pair(
    processor: Any,
    frame_paths: Sequence[Path],
    image_size: tuple[int, int],
    frame_loader: Callable[[Path, tuple[int, int]], Any],
) -> PackedPair:
    frames = [frame_loader(path, image_size) for path in frame_paths]
    try:
        output = processor.video_processor(
            videos=[frames],
            do_resize=False,
            do_sample_frames=False,
            return_tensors="pt",
        )
    except AttributeError as exc:
        raise ValueError("Local processor has no Qwen3-VL video_processor") from exc

    pixel_values = _mapping_value(output, "pixel_values_videos")
    grid = _mapping_value(output, "video_grid_thw")
    return PackedPair(
        pixel_values=_tensor_to_float32_bytes(pixel_values),
        shape=_tensor_shape(pixel_values),
        grid_thw=_grid_tuple(grid),
    )


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except AttributeError as exc:
        raise ValueError("Local Qwen3-VL processor has no tokenizer.encode") from exc
    try:
        return len(token_ids)
    except TypeError as exc:
        raise ValueError("Tokenizer encode result has no length") from exc


def _build_text_chunks(
    pair_timestamps: Sequence[float],
    *,
    system_prompt: str,
    question: str,
) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = [
        (
            "video_prefix_pair_000.txt",
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<{pair_timestamps[0]:.1f} seconds><|vision_start|>",
        )
    ]
    for pair_index, timestamp in enumerate(pair_timestamps[1:], start=1):
        chunks.append(
            (
                f"video_bridge_pair_{pair_index:03d}.txt",
                f"<|vision_end|><{timestamp:.1f} seconds><|vision_start|>",
            )
        )
    chunks.append(
        (
            "video_suffix.txt",
            f"<|vision_end|>{question}<|im_end|>\n"
            "<|im_start|>assistant\n",
        )
    )
    return chunks


def _script_path(path: Path, bundle: Path) -> str:
    try:
        relative = Path(os.path.relpath(path, bundle)).as_posix()
    except ValueError as exc:
        raise ValueError("Output directory and bundle must be on the same filesystem") from exc
    if any(character.isspace() for character in relative):
        raise ValueError(
            "Generated Genie paths cannot contain whitespace; choose another output path"
        )
    return relative


def _generate_genie_script(
    *,
    bundle: Path,
    output_dir: Path,
    pair_count: int,
    chunk_names: Sequence[str],
) -> str:
    input_dir = output_dir / "sample_inputs"
    lines = [
        "version",
        "pipeline config create pipelineConfig",
        "pipeline create GeniePipeline pipelineConfig",
        "",
        "node config create imageEncoderConfig img-enc-htp.json",
        "node create imageEncoder imageEncoderConfig",
        "",
        "node config create lutEncoderConfig text-encoder.json",
        "node create lutEncoder lutEncoderConfig",
        "",
        "node config create textGeneratorConfig text-generator.json",
        "node create textGenerator textGeneratorConfig",
        "node set textCallback textGenerator GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT",
        "",
        "pipeline add GeniePipeline imageEncoder",
        "pipeline add GeniePipeline lutEncoder",
        "pipeline add GeniePipeline textGenerator",
        "",
        "pipeline connect GeniePipeline imageEncoder "
        "GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT textGenerator "
        "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
        "pipeline connect GeniePipeline lutEncoder "
        "GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT textGenerator "
        "GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
        "",
    ]

    for pair_index in range(pair_count):
        chunk_name = chunk_names[pair_index]
        lines.append(
            "node set textFile lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "
            + _script_path(input_dir / chunk_name, bundle)
        )
        lines.append(
            "node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT "
            + _script_path(
                input_dir / f"pair_{pair_index:03d}_pixel_values.raw", bundle
            )
        )
        for filename, node_io in (
            (
                "position_ids_cos.raw",
                "GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS",
            ),
            (
                "position_ids_sin.raw",
                "GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN",
            ),
            (
                "window_attention_mask.raw",
                "GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK",
            ),
            (
                "full_attention_mask.raw",
                "GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK",
            ),
        ):
            lines.append(
                f"node set embedding imageEncoder {node_io} "
                + _script_path(input_dir / filename, bundle)
            )

    lines.extend(
        [
            "node set textFile lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "
            + _script_path(input_dir / chunk_names[-1], bundle),
            "",
            "pipeline execute GeniePipeline",
            "",
            "node free imageEncoder",
            "node free lutEncoder",
            "node free textGenerator",
            "pipeline free GeniePipeline",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_video_npu_inputs(
    bundle: Path,
    processor_path: Path,
    output_dir: Path,
    frame_paths: Sequence[Path],
    timestamps: Sequence[float],
    *,
    question: str = "What happens in this video?",
    system_prompt: str = "You are a helpful assistant.",
    reserve_generation_tokens: int = 1,
    processor_loader: Callable[[Path], Any] = _load_local_processor,
    frame_loader: Callable[[Path, tuple[int, int]], Any] = _load_frame,
) -> Path:
    """Prepare paired raw inputs, text chunks, script, and provenance manifest."""

    bundle = bundle.expanduser().resolve()
    processor_path = processor_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"Bundle directory does not exist: {bundle}")
    if not processor_path.is_dir():
        raise ValueError(f"Local processor directory does not exist: {processor_path}")
    if not output_dir.is_relative_to(bundle):
        raise ValueError(
            "Output directory must be inside the bundle so generated Genie "
            "paths remain portable when that bundle is copied to the EVK"
        )
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    if reserve_generation_tokens < 0:
        raise ValueError("reserve_generation_tokens must be non-negative")
    if not question:
        raise ValueError("question must not be empty")

    contract = _validate_bundle(bundle)
    frames, normalized_timestamps = _validate_frames_and_timestamps(
        frame_paths, timestamps
    )
    processor = processor_loader(processor_path)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Local processor has no tokenizer")

    pair_timestamps = [
        (normalized_timestamps[index] + normalized_timestamps[index + 1]) / 2
        for index in range(0, len(normalized_timestamps), 2)
    ]
    text_chunks = _build_text_chunks(
        pair_timestamps,
        system_prompt=system_prompt,
        question=question,
    )
    chunk_token_counts = {
        filename: _token_count(tokenizer, text)
        for filename, text in text_chunks
    }
    text_tokens = sum(chunk_token_counts.values())
    pair_count = len(pair_timestamps)
    visual_tokens = pair_count * VISUAL_TOKENS_PER_PAIR
    prompt_tokens = text_tokens + visual_tokens
    required_tokens = prompt_tokens + reserve_generation_tokens
    if required_tokens > contract.context_size:
        raise ValueError(
            "Video prompt exceeds bundle context: "
            f"{text_tokens} text + {visual_tokens} visual + "
            f"{reserve_generation_tokens} reserved generation = {required_tokens}, "
            f"context is {contract.context_size}. Current CL512 supports one "
            "512x512 frame pair; use a larger text context for multiple pairs."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.partial")
    if temporary.exists():
        raise ValueError(f"Temporary output already exists: {temporary}")

    try:
        input_dir = temporary / "sample_inputs"
        input_dir.mkdir(parents=True)

        ancillary_manifest: dict[str, Any] = {}
        for filename in ANCILLARY_SPECS:
            source = bundle / "sample_inputs" / filename
            destination = input_dir / filename
            shutil.copy2(source, destination)
            ancillary_manifest[filename] = {
                "source": str(source),
                "source_sha256": sha256_file(source),
                "output": f"sample_inputs/{filename}",
                "output_sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "shape": list(ANCILLARY_SPECS[filename]),
                "serialization": "float32 little-endian raw",
            }

        pairs_manifest: list[dict[str, Any]] = []
        expected_pixel_bytes = math.prod(EXPECTED_PIXEL_SHAPE) * FLOAT32_BYTES
        for pair_index in range(pair_count):
            first = pair_index * 2
            pair_frames = frames[first : first + 2]
            packed = _pack_pair(
                processor,
                pair_frames,
                (contract.image_width, contract.image_height),
                frame_loader,
            )
            if packed.shape != EXPECTED_PIXEL_SHAPE:
                raise ValueError(
                    f"Pair {pair_index} processor shape {packed.shape}; "
                    f"expected {EXPECTED_PIXEL_SHAPE}"
                )
            if packed.grid_thw != EXPECTED_GRID_THW:
                raise ValueError(
                    f"Pair {pair_index} grid {packed.grid_thw}; "
                    f"expected {EXPECTED_GRID_THW}"
                )
            if len(packed.pixel_values) != expected_pixel_bytes:
                raise ValueError(
                    f"Pair {pair_index} serialized to {len(packed.pixel_values)} "
                    f"bytes; expected {expected_pixel_bytes} float32 bytes"
                )

            filename = f"pair_{pair_index:03d}_pixel_values.raw"
            pixel_path = input_dir / filename
            pixel_path.write_bytes(packed.pixel_values)
            pairs_manifest.append(
                {
                    "pair_index": pair_index,
                    "frame_indices": [first, first + 1],
                    "frame_timestamps_seconds": normalized_timestamps[
                        first : first + 2
                    ],
                    "pair_timestamp_seconds": pair_timestamps[pair_index],
                    "pair_timestamp_text": f"{pair_timestamps[pair_index]:.1f}",
                    "pixel_values": {
                        "file": f"sample_inputs/{filename}",
                        "sha256": sha256_file(pixel_path),
                        "bytes": pixel_path.stat().st_size,
                        "shape": list(packed.shape),
                        "grid_thw": list(packed.grid_thw),
                        "serialization": "float32 little-endian raw",
                    },
                }
            )

        chunks_manifest: list[dict[str, Any]] = []
        for filename, text in text_chunks:
            path = input_dir / filename
            path.write_text(text, encoding="utf-8", newline="\n")
            chunks_manifest.append(
                {
                    "file": f"sample_inputs/{filename}",
                    "sha256": sha256_file(path),
                    "tokens": chunk_token_counts[filename],
                    "text": text,
                }
            )

        chunk_names = [filename for filename, _ in text_chunks]
        script_text = _generate_genie_script(
            bundle=bundle,
            output_dir=output_dir,
            pair_count=pair_count,
            chunk_names=chunk_names,
        )
        script_path = temporary / "genie-video-app-script.txt"
        script_path.write_text(script_text, encoding="utf-8", newline="\n")

        frame_manifest = [
            {
                "index": index,
                "path": str(path),
                "sha256": sha256_file(path),
                "timestamp_seconds": normalized_timestamps[index],
            }
            for index, path in enumerate(frames)
        ]
        processor_files = {
            filename: {
                "path": str(processor_path / filename),
                "sha256": sha256_file(processor_path / filename),
            }
            for filename in PROCESSOR_PROVENANCE_FILES
            if (processor_path / filename).is_file()
        }
        bundle_files = {
            filename: {
                "path": str(bundle / filename),
                "sha256": sha256_file(bundle / filename),
                "bytes": (bundle / filename).stat().st_size,
            }
            for filename in (
                "metadata.json",
                "genie_config.json",
                "htp_backend_ext_config.json",
                *NODE_CONFIG_FILES,
                *CONTEXT_BINARY_FILES,
            )
        }
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Qwen3-VL paired-frame input for legacy Genie NPU pipeline",
            "bundle": {
                "path": str(bundle),
                "metadata_sha256": sha256_file(bundle / "metadata.json"),
                "genie_config_sha256": sha256_file(bundle / "genie_config.json"),
                "context_size": contract.context_size,
                "files": bundle_files,
            },
            "processor": {
                "path": str(processor_path),
                "local_files_only": True,
                "trust_remote_code": False,
                "files": processor_files,
                "library_versions": {
                    "transformers": _distribution_version("transformers"),
                    "torch": _distribution_version("torch"),
                    "Pillow": _distribution_version("Pillow"),
                },
            },
            "graph_contract": {
                "frames_per_pair": 2,
                "image_size": [contract.image_width, contract.image_height],
                "pixel_values_shape": list(EXPECTED_PIXEL_SHAPE),
                "grid_thw": list(EXPECTED_GRID_THW),
                "image_features_shape": list(EXPECTED_VISUAL_OUTPUT_SHAPE),
                "visual_tokens_per_pair": VISUAL_TOKENS_PER_PAIR,
            },
            "context_budget": {
                "context_size": contract.context_size,
                "text_tokens": text_tokens,
                "visual_tokens": visual_tokens,
                "visual_tokens_per_pair": VISUAL_TOKENS_PER_PAIR,
                "prompt_tokens": prompt_tokens,
                "reserved_generation_tokens": reserve_generation_tokens,
                "remaining_after_prompt": contract.context_size - prompt_tokens,
            },
            "frames": frame_manifest,
            "pairs": pairs_manifest,
            "ancillary_tensors": ancillary_manifest,
            "text_chunks": chunks_manifest,
            "script": {
                "file": "genie-video-app-script.txt",
                "sha256": sha256_file(script_path),
                "working_directory": str(bundle),
                "run_argument": _script_path(
                    output_dir / "genie-video-app-script.txt", bundle
                ),
            },
        }
        (temporary / "video_npu_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--processor",
        type=Path,
        required=True,
        help="Local NVIDIA Cosmos/Qwen3-VL Hugging Face checkpoint directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--timestamps", nargs="+", type=float, required=True)
    parser.add_argument(
        "--question",
        default="What happens in this video?",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
    )
    parser.add_argument(
        "--reserve-generation-tokens",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    output = prepare_video_npu_inputs(
        args.bundle,
        args.processor,
        args.output_dir,
        args.frames,
        args.timestamps,
        question=args.question,
        system_prompt=args.system_prompt,
        reserve_generation_tokens=args.reserve_generation_tokens,
    )
    manifest = _load_json(output / "video_npu_manifest.json", "video manifest")
    budget = manifest["context_budget"]
    script = manifest["script"]
    print(f"Prepared paired video inputs: {output}")
    print(
        "Context: "
        f"{budget['prompt_tokens']}/{budget['context_size']} prompt tokens; "
        f"{budget['remaining_after_prompt']} remain"
    )
    print(f"Run from {script['working_directory']}:")
    print(f"  genie-app -s {script['run_argument']}")


if __name__ == "__main__":
    main()
