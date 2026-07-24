#!/usr/bin/env python3
"""Prepare and verify a full-DeepStack GenieX raw-video run package.

The regular video preparer emits a legacy-Genie script plus a provenance
manifest.  Qualcomm GenieX's Qwen3-VL runtime can consume the same packed
``pixel_values`` tensor directly, compute vision RoPE/masks itself, read all
three DeepStack outputs, and inject them into the first decoder shard.

This tool copies only the one-pair inputs needed by the standalone GenieX
runner and binds them to an exact full-DeepStack QAIRT bundle with SHA-256
hashes.  Verification is intentionally repeated on the target before launch.
It fails closed on path traversal, hashes, tensor shapes, model dimensions,
DeepStack wiring, Qwen3-VL MRoPE settings, and context capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.vision_profile import VisionProfile
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from vision_profile import VisionProfile

SCHEMA_VERSION = 2
MANIFEST_FILENAME = "geniex_raw_video_manifest.json"
# Backwards-compatible constants for the original proven profile. Runtime
# contracts are derived from metadata.json.
EXPECTED_CONTEXT_SIZE = 512
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_VOCAB_SIZE = 151936
EXPECTED_PIXEL_SHAPE = (1024, 1536)
EXPECTED_PIXEL_BYTES = 1024 * 1536 * 4
EXPECTED_GRID_THW = (1, 32, 32)
EXPECTED_VISUAL_SHAPE = (256, 2048)
EXPECTED_VISUAL_TOKENS = 256
EXPECTED_DEEPSTACK_LEVELS = 3
EXPECTED_MROPE_SECTION = [24, 20, 20]
# These QAIRT text artifacts are compiled as AR128 prefill plus AR1 decode
# graphs. The pinned GenieX runtime cannot safely transition more than CL-AR
# prompt tokens from the prefill KV layout into the decode KV layout.
TEXT_PREFILL_AR = 128
TEXT_DECODE_AR = 1
EXPECTED_VISION_PREPROCESSING = {
    "image_width": 512,
    "image_height": 512,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
}

VISION_INPUTS = {
    "pixel_values": EXPECTED_PIXEL_SHAPE,
    "position_ids_cos": (1024, 32),
    "position_ids_sin": (1024, 32),
    "window_attention_mask": (1, 1024, 1024),
    "full_attention_mask": (1, 1024, 1024),
}
VISION_OUTPUTS = {
    "image_features": EXPECTED_VISUAL_SHAPE,
    **{
        f"deepstack_visual_embeds_{index}": EXPECTED_VISUAL_SHAPE
        for index in range(EXPECTED_DEEPSTACK_LEVELS)
    },
}
DEEPSTACK_TEXT_INPUTS = {
    "visual_pos_masks",
    *{
        f"deepstack_visual_embeds_{index}"
        for index in range(EXPECTED_DEEPSTACK_LEVELS)
    },
}
LEGACY_GENIEX_SPECIAL_INPUTS = {
    "attention_mask",
    "position_ids",
    "position_ids_cos",
    "position_ids_sin",
}
CONTEXT_FILES = (
    "vision_encoder.bin",
    "part1_of_4.bin",
    "part2_of_4.bin",
    "part3_of_4.bin",
    "part4_of_4.bin",
)
BUNDLE_FILES = (
    "metadata.json",
    "genie_config.json",
    "img-enc-htp.json",
    "htp_backend_ext_config.json",
    "tokenizer.json",
    "embedding_weights.raw",
    *CONTEXT_FILES,
)


@dataclass(frozen=True)
class TargetContract:
    context_size: int
    profile: VisionProfile

    @property
    def prefill_kv_capacity(self) -> int:
        return self.context_size - TEXT_PREFILL_AR

    def text_runtime_contract(self) -> dict[str, int]:
        return {
            "prefill_ar": TEXT_PREFILL_AR,
            "decode_ar": TEXT_DECODE_AR,
            "prefill_kv_capacity": self.prefill_kv_capacity,
            "max_safe_prompt_tokens": self.prefill_kv_capacity,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _shape(tensors: dict[str, Any], name: str, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in tensors[name]["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has no valid shape for {name}") from exc


def _safe_file(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a non-empty string")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} escapes its package directory: {relative!r}")
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    return candidate


def _validate_hash(path: Path, expected: str, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} has no valid SHA-256")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}"
        )


def _validate_target_bundle(
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any], TargetContract]:
    missing = [name for name in BUNDLE_FILES if not (bundle / name).is_file()]
    if missing:
        raise ValueError(f"Target bundle is missing required files: {missing}")

    metadata = _load_json(bundle / "metadata.json", "target metadata.json")
    config = _load_json(bundle / "genie_config.json", "target genie_config.json")
    try:
        model_files = metadata["model_files"]
        vision = model_files["vision_encoder.bin"]
        vision_inputs = vision["inputs"]
        vision_outputs = vision["outputs"]
        preprocessing = metadata["genie"]["vision_preprocessing"]
        dialog = config["dialog"]
        context_size = int(dialog["context"]["size"])
        vocab_size = int(dialog["context"]["n-vocab"])
        embedding_size = int(dialog["embedding"]["size"])
        embedding_type = str(dialog["embedding"]["datatype"])
        rope = dialog["engine"]["model"]["positional-encoding"]["rope-scaling"]
        context_names = tuple(dialog["engine"]["model"]["binary"]["ctx-bins"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Target bundle does not match the expected QAIRT/Genie schema"
        ) from exc

    if context_size <= TEXT_PREFILL_AR:
        raise ValueError(
            "Target context size must exceed the fixed "
            f"AR{TEXT_PREFILL_AR} prefill width"
        )
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Target vocabulary is {vocab_size}; expected {EXPECTED_VOCAB_SIZE}"
        )
    if embedding_size != EXPECTED_HIDDEN_SIZE or embedding_type != "float32":
        raise ValueError(
            "Target embedding LUT must be float32 with hidden size "
            f"{EXPECTED_HIDDEN_SIZE}"
        )
    expected_embedding_bytes = (
        EXPECTED_VOCAB_SIZE * EXPECTED_HIDDEN_SIZE * 4
    )
    actual_embedding_bytes = (bundle / "embedding_weights.raw").stat().st_size
    if actual_embedding_bytes != expected_embedding_bytes:
        raise ValueError(
            "embedding_weights.raw has "
            f"{actual_embedding_bytes} bytes; expected {expected_embedding_bytes}"
        )
    if context_names != CONTEXT_FILES[1:]:
        raise ValueError(
            f"Unexpected text context order: {context_names!r}; "
            f"expected {CONTEXT_FILES[1:]!r}"
        )

    profile = VisionProfile.from_metadata(metadata)
    profile.validate_img_encoder_config(bundle / "img-enc-htp.json")
    for name in VISION_OUTPUTS:
        actual = _shape(vision_outputs, name, "vision outputs")
        if actual != profile.visual_shape:
            raise ValueError(
                f"Vision output {name} has shape {actual}; expected "
                f"{profile.visual_shape}"
            )
    if rope.get("rope-type") != "qwen3vl-mrope":
        raise ValueError("Target must use qwen3vl-mrope")
    if rope.get("mrope-section") != EXPECTED_MROPE_SECTION:
        raise ValueError(
            "Target mrope-section must be "
            f"{EXPECTED_MROPE_SECTION}, got {rope.get('mrope-section')!r}"
        )
    if int(rope.get("spatial-merge-size", -1)) != 2:
        raise ValueError("Target MRoPE spatial-merge-size must be 2")

    part1_inputs = model_files.get("part1_of_4.bin", {}).get("inputs", {})
    if not isinstance(part1_inputs, dict):
        raise ValueError("Target metadata has no part1 input map")
    missing_deepstack = sorted(DEEPSTACK_TEXT_INPUTS - set(part1_inputs))
    if missing_deepstack:
        raise ValueError(
            "Target is not a full-DeepStack bundle; part1 is missing "
            f"{missing_deepstack}"
        )
    # GenieX v0.3.16 originally omitted the DeepStack inputs from
    # isSpecialTensor().  The overlay build patches that classifier, but keep
    # the original order constraint as a second fail-closed guard: should an
    # unpatched binary ever be used, its first inferred state input must still
    # be inputs_embeds rather than an auxiliary visual tensor.
    first_legacy_state_input: str | None = None
    for name in part1_inputs:
        is_kv = (
            name.endswith(("_in", "_out"))
            and ("key" in name or "value" in name)
        )
        if name not in LEGACY_GENIEX_SPECIAL_INPUTS and not is_kv:
            first_legacy_state_input = name
            break
    if first_legacy_state_input != "inputs_embeds":
        raise ValueError(
            "Unsafe part1 QNN input order: legacy GenieX would infer "
            f"{first_legacy_state_input!r} instead of 'inputs_embeds'. "
            "Re-export/relink with inputs_embeds first."
        )
    for index in range(EXPECTED_DEEPSTACK_LEVELS):
        name = f"deepstack_visual_embeds_{index}"
        shape = _shape(part1_inputs, name, "part1 inputs")
        if len(shape) != 2 or shape[-1] != EXPECTED_HIDDEN_SIZE or shape[0] <= 0:
            raise ValueError(
                f"Part1 input {name} has shape {shape}; expected "
                f"[positive rows, {EXPECTED_HIDDEN_SIZE}]"
            )
    mask_shape = _shape(part1_inputs, "visual_pos_masks", "part1 inputs")
    if not mask_shape or mask_shape[-1] <= 0:
        raise ValueError(
            f"Part1 visual_pos_masks has invalid shape {mask_shape}"
        )

    for filename in CONTEXT_FILES[2:]:
        inputs = model_files.get(filename, {}).get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"Target metadata has no input map for {filename}")
        unexpected = sorted(DEEPSTACK_TEXT_INPUTS & set(inputs))
        if unexpected:
            raise ValueError(
                f"{filename} unexpectedly exposes first-shard DeepStack "
                f"inputs: {unexpected}"
            )

    cache_length = context_size - 1
    for filename in CONTEXT_FILES[1:]:
        inputs = model_files.get(filename, {}).get("inputs", {})
        try:
            cache_name = next(
                name
                for name in inputs
                if name.startswith("past_key_") and name.endswith("_in")
            )
        except StopIteration as exc:
            raise ValueError(f"{filename} has no past-key input") from exc
        cache_shape = _shape(inputs, cache_name, f"{filename} inputs")
        if not cache_shape or cache_shape[-1] != cache_length:
            raise ValueError(
                f"{filename} cache length is "
                f"{cache_shape[-1] if cache_shape else None}; "
                f"expected {cache_length}"
            )

    return metadata, config, TargetContract(
        context_size=context_size,
        profile=profile,
    )


def _validate_source_video(
    video_dir: Path,
    target: TargetContract,
) -> tuple[dict[str, Any], Path, list[Path], list[Path], int]:
    source_manifest_path = video_dir / "video_npu_manifest.json"
    source = _load_json(source_manifest_path, "video_npu_manifest.json")
    try:
        contract = source["graph_contract"]
        budget = source["context_budget"]
        pairs = source["pairs"]
        chunks = source["text_chunks"]
    except KeyError as exc:
        raise ValueError("Video manifest is missing its graph/input contract") from exc

    profile = target.profile
    expected_contract = profile.as_manifest_contract()
    mismatches = {
        name: (contract.get(name), expected)
        for name, expected in expected_contract.items()
        if contract.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Source video tensor contract mismatch: {mismatches}")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Video manifest must contain at least one frame pair")
    if not isinstance(chunks, list) or len(chunks) != len(pairs) + 1:
        raise ValueError(
            "Video manifest must contain prefix, one bridge per additional "
            "pair, and suffix text"
        )

    pixel_paths: list[Path] = []
    for expected_index, pair in enumerate(pairs):
        try:
            pair_index = int(pair["pair_index"])
            pixel_info = pair["pixel_values"]
            pixel_relative = pixel_info["file"]
            pixel_sha = pixel_info["sha256"]
            pixel_bytes = int(pixel_info["bytes"])
            pixel_shape = tuple(
                int(value) for value in pixel_info["shape"]
            )
            pixel_grid = tuple(
                int(value) for value in pixel_info["grid_thw"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Video manifest has no valid pair {expected_index} tensor entry"
            ) from exc
        if pair_index != expected_index:
            raise ValueError(
                f"Video pair indices must be contiguous from zero; got "
                f"{pair_index} at position {expected_index}"
            )
        pixel_path = _safe_file(
            video_dir, pixel_relative, f"packed pixel tensor {expected_index}"
        )
        if (
            pixel_bytes != profile.pixel_bytes
            or pixel_path.stat().st_size != pixel_bytes
        ):
            raise ValueError(
                f"Packed tensor {expected_index} must contain "
                f"{profile.pixel_bytes} bytes"
            )
        if (
            pixel_shape != profile.pixel_shape
            or pixel_grid != profile.grid_thw
        ):
            raise ValueError(
                f"Packed tensor {expected_index} shape/grid does not match "
                "the target vision graph"
            )
        _validate_hash(
            pixel_path, pixel_sha, f"packed pixel tensor {expected_index}"
        )
        pixel_paths.append(pixel_path)

    chunk_paths: list[Path] = []
    for entry in chunks:
        try:
            relative = entry["file"]
            expected_hash = entry["sha256"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Video manifest has an invalid text chunk") from exc
        path = _safe_file(video_dir, relative, "text chunk")
        _validate_hash(path, expected_hash, f"text chunk {relative}")
        if entry.get("text") != path.read_text(encoding="utf-8"):
            raise ValueError(f"Text chunk differs from its manifest: {relative}")
        chunk_paths.append(path)

    expected_names = [
        "video_prefix_pair_000.txt",
        *[
            f"video_bridge_pair_{index:03d}.txt"
            for index in range(1, len(pairs))
        ],
        "video_suffix.txt",
    ]
    actual_names = [path.name for path in chunk_paths]
    if actual_names != expected_names:
        raise ValueError(
            f"Unexpected ordered text chunks: {actual_names}; expected "
            f"{expected_names}"
        )
    prefix_text = chunk_paths[0].read_text(encoding="utf-8")
    suffix_text = chunk_paths[-1].read_text(encoding="utf-8")
    if (
        not prefix_text.endswith("<|vision_start|>")
        or prefix_text.count("<|vision_start|>") != 1
        or "<|image_pad|>" in prefix_text
        or "<|video_pad|>" in prefix_text
    ):
        raise ValueError("Prefix has an unsafe Qwen3-VL vision-token layout")
    if (
        not suffix_text.startswith("<|vision_end|>")
        or suffix_text.count("<|vision_end|>") != 1
        or "<|image_pad|>" in suffix_text
        or "<|video_pad|>" in suffix_text
    ):
        raise ValueError("Suffix has an unsafe Qwen3-VL vision-token layout")
    for index, bridge_path in enumerate(chunk_paths[1:-1], start=1):
        bridge = bridge_path.read_text(encoding="utf-8")
        if (
            not bridge.startswith("<|vision_end|>")
            or not bridge.endswith("<|vision_start|>")
            or bridge.count("<|vision_end|>") != 1
            or bridge.count("<|vision_start|>") != 1
            or "<|image_pad|>" in bridge
            or "<|video_pad|>" in bridge
        ):
            raise ValueError(
                f"Bridge {index} has an unsafe Qwen3-VL vision-token layout"
            )

    try:
        chunk_text_tokens = sum(int(entry["tokens"]) for entry in chunks)
        text_tokens = int(budget["text_tokens"])
        visual_tokens = int(budget["visual_tokens"])
        prompt_tokens = int(budget["prompt_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Video manifest has an invalid context budget") from exc
    if (
        text_tokens != chunk_text_tokens
        or visual_tokens != len(pairs) * profile.visual_tokens
        or prompt_tokens != text_tokens + visual_tokens
    ):
        raise ValueError("Video manifest context budget is internally inconsistent")
    if prompt_tokens > target.prefill_kv_capacity:
        raise ValueError(
            "Video prompt exceeds the safe GenieX prefill cache: "
            f"{prompt_tokens} prompt tokens, but AR{TEXT_PREFILL_AR}/"
            f"CL{target.context_size} permits at most "
            f"{target.prefill_kv_capacity}. Reduce the pair count or use a "
            "larger compiled text context."
        )

    source_text_contract = source.get("bundle", {}).get(
        "text_runtime_contract"
    )
    if source_text_contract is not None:
        expected_text_contract = target.text_runtime_contract()
        if source_text_contract != expected_text_contract:
            raise ValueError(
                "Source video text-runtime contract does not match the "
                f"target bundle: {source_text_contract!r} != "
                f"{expected_text_contract!r}"
            )
    if "prefill_safe_limit" in budget:
        try:
            source_prefill_limit = int(budget["prefill_safe_limit"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Video manifest has an invalid prefill-safe limit"
            ) from exc
        if source_prefill_limit != target.prefill_kv_capacity:
            raise ValueError(
                "Video manifest prefill-safe limit does not match the target "
                f"bundle: {source_prefill_limit} != "
                f"{target.prefill_kv_capacity}"
            )

    return (
        source,
        source_manifest_path,
        pixel_paths,
        chunk_paths,
        prompt_tokens,
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_run_package(
    bundle: Path,
    video_dir: Path,
    output_dir: Path,
) -> Path:
    """Create a portable raw-tensor run package bound to a target bundle."""

    bundle = bundle.expanduser().resolve()
    video_dir = video_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"Target bundle does not exist: {bundle}")
    if not video_dir.is_dir():
        raise ValueError(f"Video input directory does not exist: {video_dir}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    _, _, target = _validate_target_bundle(bundle)
    (
        source,
        source_manifest_path,
        pixel_paths,
        chunk_paths,
        prompt_tokens,
    ) = _validate_source_video(video_dir, target)

    temporary = output_dir.with_name(f".{output_dir.name}.partial")
    if temporary.exists():
        raise ValueError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        inputs = temporary / "inputs"
        inputs.mkdir()
        copied_pixels: list[Path] = []
        for index, source_path in enumerate(pixel_paths):
            destination = inputs / f"pair_{index:03d}_pixel_values.raw"
            shutil.copy2(source_path, destination)
            copied_pixels.append(destination)
        copied_chunks: list[Path] = []
        for source_path in chunk_paths:
            destination = inputs / source_path.name
            shutil.copy2(source_path, destination)
            copied_chunks.append(destination)

        bundle_records = {
            name: {
                "file": name,
                **_file_record(bundle / name),
            }
            for name in BUNDLE_FILES
        }
        pixel_records = [
            {
                "file": str(path.relative_to(temporary).as_posix()),
                **_file_record(path),
            }
            for path in copied_pixels
        ]
        chunk_records = [
            {
                "file": str(path.relative_to(temporary).as_posix()),
                **_file_record(path),
            }
            for path in copied_chunks
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "Qwen3-VL temporal frame pairs through full-DeepStack "
                "Cosmos-Reason2-2B GenieX QAIRT"
            ),
            "source_video_manifest": {
                "sha256": sha256_file(source_manifest_path),
                "source_schema_version": source.get("schema_version"),
            },
            "target_bundle": {
                "files": bundle_records,
                "contract": {
                    "context_size": target.context_size,
                    "hidden_size": EXPECTED_HIDDEN_SIZE,
                    "vocab_size": EXPECTED_VOCAB_SIZE,
                    "deepstack_levels": EXPECTED_DEEPSTACK_LEVELS,
                    "mrope_section": EXPECTED_MROPE_SECTION,
                    "mrope_interleaving": "stride",
                    "text_runtime": target.text_runtime_contract(),
                    "vision": target.profile.as_manifest_contract(),
                },
            },
            "input": {
                "pixel_values": pixel_records,
                "text_chunks": chunk_records,
                "pair_count": len(copied_pixels),
                "pixel_values_shape": list(target.profile.pixel_shape),
                "grid_thw": list(target.profile.grid_thw),
                "visual_tokens_per_pair": target.profile.visual_tokens,
                "visual_tokens": (
                    len(copied_pixels) * target.profile.visual_tokens
                ),
                "expected_prompt_tokens": prompt_tokens,
                "prefill_safe_limit": target.prefill_kv_capacity,
            },
            "runtime_assumptions": {
                "frames_per_temporal_patch": 2,
                "temporal_grid_extent": 1,
                "text_prefill": {
                    **target.text_runtime_contract(),
                    "failure_mode": (
                        "Prompts above CL-AR are rejected because pinned "
                        "GenieX v0.3.16 cannot safely merge the final AR128 "
                        "prefill KV block into the AR1 decode cache."
                    ),
                },
                "explanation": (
                    "The two source frames are fused by temporal_patch_size=2 "
                    "into one T=1 vision grid per pair. GenieX emits one "
                    "Qwen3-VL MRoPE grid for each timestamped pair; motion "
                    "within a pair is represented by the vision encoder."
                ),
            },
        }
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def verify_run_package(
    package_dir: Path,
    bundle: Path,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    """Verify a prepared package and exact target bundle before execution."""

    package_dir = package_dir.expanduser().resolve()
    bundle = bundle.expanduser().resolve()
    if not package_dir.is_dir():
        raise ValueError(f"Run package does not exist: {package_dir}")
    if not bundle.is_dir():
        raise ValueError(f"Target bundle does not exist: {bundle}")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    manifest = _load_json(
        package_dir / MANIFEST_FILENAME, MANIFEST_FILENAME
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema: {manifest.get('schema_version')!r}"
        )
    _, _, target = _validate_target_bundle(bundle)

    try:
        bundle_records = manifest["target_bundle"]["files"]
        manifest_target_contract = manifest["target_bundle"]["contract"]
        input_section = manifest["input"]
        pixel_records = input_section["pixel_values"]
        chunk_records = input_section["text_chunks"]
        pair_count = int(input_section["pair_count"])
        expected_prompt_tokens = int(input_section["expected_prompt_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Run manifest is missing required records") from exc
    if set(bundle_records) != set(BUNDLE_FILES):
        raise ValueError("Run manifest bundle file set is incomplete or unexpected")
    for name in BUNDLE_FILES:
        record = bundle_records[name]
        if record.get("file") != name:
            raise ValueError(f"Run manifest remaps bundle file {name}")
        path = bundle / name
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"Target bundle file size changed: {name}")
        _validate_hash(path, record.get("sha256"), f"target bundle {name}")

    expected_text_runtime = target.text_runtime_contract()
    if manifest_target_contract.get("text_runtime") != expected_text_runtime:
        raise ValueError(
            "Run manifest text-runtime contract does not match the target "
            "bundle"
        )

    if (
        not isinstance(pixel_records, list)
        or not isinstance(chunk_records, list)
        or pair_count <= 0
        or len(pixel_records) != pair_count
        or len(chunk_records) != pair_count + 1
    ):
        raise ValueError("Run manifest has an invalid pair/chunk file set")

    pixel_paths: list[Path] = []
    for index, record in enumerate(pixel_records):
        if not isinstance(record, dict):
            raise ValueError(f"Pixel record {index} must be an object")
        path = _safe_file(
            package_dir,
            record.get("file"),
            f"package pixel input {index}",
        )
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"Package pixel input size changed: {index}")
        _validate_hash(
            path, record.get("sha256"), f"package pixel input {index}"
        )
        if path.stat().st_size != target.profile.pixel_bytes:
            raise ValueError(
                f"Package pixel tensor {index} has the wrong byte count"
            )
        pixel_paths.append(path)

    chunk_paths: list[Path] = []
    for index, record in enumerate(chunk_records):
        if not isinstance(record, dict):
            raise ValueError(f"Text chunk record {index} must be an object")
        path = _safe_file(
            package_dir,
            record.get("file"),
            f"package text chunk {index}",
        )
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"Package text chunk size changed: {index}")
        _validate_hash(
            path, record.get("sha256"), f"package text chunk {index}"
        )
        chunk_paths.append(path)

    expected_input_contract = {
        "pixel_values_shape": list(target.profile.pixel_shape),
        "grid_thw": list(target.profile.grid_thw),
        "visual_tokens_per_pair": target.profile.visual_tokens,
        "visual_tokens": pair_count * target.profile.visual_tokens,
    }
    mismatches = {
        name: (input_section.get(name), expected)
        for name, expected in expected_input_contract.items()
        if input_section.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Run manifest vision contract mismatch: {mismatches}")

    if expected_prompt_tokens <= pair_count * target.profile.visual_tokens:
        raise ValueError("Run manifest has an invalid prompt token count")
    if input_section.get("prefill_safe_limit") != target.prefill_kv_capacity:
        raise ValueError(
            "Run manifest prefill-safe limit does not match the target bundle"
        )
    if expected_prompt_tokens > target.prefill_kv_capacity:
        raise ValueError(
            f"Prompt ({expected_prompt_tokens}) exceeds the safe "
            f"AR{TEXT_PREFILL_AR}/CL{target.context_size} prefill cache "
            f"({target.prefill_kv_capacity})"
        )
    if expected_prompt_tokens + max_tokens > target.context_size:
        raise ValueError(
            f"Prompt ({expected_prompt_tokens}) + generation ({max_tokens}) "
            f"exceeds CL{target.context_size}"
        )
    return manifest, {
        "pixel_values": pixel_paths,
        "text_chunks": chunk_paths,
    }


def launch_runner(
    runner: Path,
    package_dir: Path,
    bundle: Path,
    *,
    max_tokens: int,
    verbose: bool = False,
) -> int:
    manifest, paths = verify_run_package(
        package_dir, bundle, max_tokens=max_tokens
    )
    runner = runner.expanduser().resolve()
    if not runner.is_file():
        raise ValueError(f"GenieX runner does not exist: {runner}")
    expected_prompt_tokens = manifest["input"]["expected_prompt_tokens"]
    command = [
        str(runner),
        "--model-dir",
        str(bundle.expanduser().resolve()),
    ]
    for path in paths["pixel_values"]:
        command.extend(("--pixel-values", str(path)))
    for path in paths["text_chunks"]:
        command.extend(("--text-chunk", str(path)))
    command.extend(
        (
            "--expected-prompt-tokens",
            str(expected_prompt_tokens),
            "--max-tokens",
            str(max_tokens),
        )
    )
    if verbose:
        command.append("--verbose")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a bound run package")
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--video-input-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify without executing")
    verify.add_argument("--package-dir", type=Path, required=True)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--max-tokens", type=int, default=64)

    launch = subparsers.add_parser(
        "launch", help="Verify hashes/contracts, then execute the NPU runner"
    )
    launch.add_argument("--package-dir", type=Path, required=True)
    launch.add_argument("--bundle", type=Path, required=True)
    launch.add_argument("--runner", type=Path, required=True)
    launch.add_argument("--max-tokens", type=int, default=64)
    launch.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        output = prepare_run_package(
            args.bundle, args.video_input_dir, args.output_dir
        )
        print(f"Prepared full-DeepStack GenieX run package: {output}")
        return 0
    if args.command == "verify":
        manifest, _ = verify_run_package(
            args.package_dir, args.bundle, max_tokens=args.max_tokens
        )
        print(
            "Verified full-DeepStack run package: "
            f"{manifest['input']['expected_prompt_tokens']} prompt tokens, "
            f"{args.max_tokens} generation tokens"
        )
        return 0
    return launch_runner(
        args.runner,
        args.package_dir,
        args.bundle,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
