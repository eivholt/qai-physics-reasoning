#!/usr/bin/env python3
"""Run deterministic BF16 GPU references for paired-video benchmark cases.

The runner deliberately uses the upstream Hugging Face Qwen3-VL video path:
each temporal pair is supplied as one video content block, source frame
indices/fps are supplied as ``VideoMetadata``, and frame sampling is disabled.
This makes the prompt tokens and packed pixels directly comparable with the
NPU input preparer.

Heavy dependencies are imported only by :func:`run`.  Manifest selection,
path sanitization, prompt scoring, and contract validation can therefore be
unit-tested without Torch, Transformers, Pillow, a model, or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    REPO_ROOT / "benchmarks" / "nvidia_sdg_warehouse" / "benchmark.json"
)
DEFAULT_FRAME_ROOT = (
    REPO_ROOT / "artifacts" / "nvidia_sdg_warehouse" / "frames"
)
FREEFORM_PROMPT_ID = "freeform"
ALL_PROMPTS_ID = "all"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
CHOICE_PREFIX = re.compile(r"^\s*([A-Z])(?:\b|\))", re.IGNORECASE)
CHOICE_NAMED = re.compile(
    r"\b(?:answer|choice|option)\s*(?:is|:)?\s*([A-Z])\b",
    re.IGNORECASE,
)
CHECKPOINT_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
REQUIRED_PROCESSOR_OUTPUTS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values_videos",
    "video_grid_thw",
}


@dataclass(frozen=True)
class PromptSpec:
    """One freeform or choice prompt attached to a video case."""

    case_id: str
    prompt_id: str
    kind: str
    prompt: str
    expected: Mapping[str, Any]
    context_budget: Mapping[str, Any]
    evaluation_role: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not SAFE_COMPONENT.fullmatch(value)
    ):
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_', or '-': "
            f"{value!r}"
        )
    return value


def load_benchmark(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported benchmark schema: {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("video_cases"), list):
        raise ValueError("Benchmark must define video_cases")
    if not isinstance(manifest.get("video_case_profile"), dict):
        raise ValueError("Benchmark must define video_case_profile")
    return manifest, raw


def select_video_cases(
    manifest: Mapping[str, Any],
    requested_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    cases = manifest["video_cases"]
    indexed = {case.get("id"): case for case in cases}
    if len(indexed) != len(cases):
        raise ValueError("Video case ids must be unique")
    if not requested_case_ids:
        selected = list(cases)
    else:
        if len(set(requested_case_ids)) != len(requested_case_ids):
            raise ValueError("Duplicate video case selector")
        unknown = sorted(set(requested_case_ids) - indexed.keys())
        if unknown:
            raise ValueError(
                "Unknown video case(s): "
                + ", ".join(unknown)
                + "; available: "
                + ", ".join(str(value) for value in indexed)
            )
        selected = [indexed[case_id] for case_id in requested_case_ids]
    for case in selected:
        safe_component(case.get("id"), "video case id")
    return selected


def prompt_specs_for_case(
    case: Mapping[str, Any],
    requested_prompt_ids: Sequence[str],
) -> list[PromptSpec]:
    """Select a case's freeform prompt and/or named ``choice_probes``."""

    selectors = list(requested_prompt_ids) or [ALL_PROMPTS_ID]
    if len(set(selectors)) != len(selectors):
        raise ValueError(f"{case['id']}: duplicate prompt selector")
    if ALL_PROMPTS_ID in selectors and len(selectors) != 1:
        raise ValueError("'all' cannot be combined with other prompt selectors")
    probes = case.get("choice_probes", [])
    if not isinstance(probes, list):
        raise ValueError(f"{case['id']}: choice_probes must be a list")
    probe_index = {probe.get("id"): probe for probe in probes}
    if len(probe_index) != len(probes):
        raise ValueError(f"{case['id']}: choice probe ids must be unique")

    selected_ids = (
        [FREEFORM_PROMPT_ID, *probe_index]
        if selectors == [ALL_PROMPTS_ID]
        else selectors
    )
    known = {FREEFORM_PROMPT_ID, *probe_index}
    unknown = sorted(set(selected_ids) - known)
    if unknown:
        raise ValueError(
            f"{case['id']}: unknown prompt(s) {', '.join(unknown)}; "
            f"available: {', '.join(sorted(known))}"
        )

    result: list[PromptSpec] = []
    for prompt_id in selected_ids:
        safe_component(prompt_id, "prompt id")
        if prompt_id == FREEFORM_PROMPT_ID:
            result.append(
                PromptSpec(
                    case_id=case["id"],
                    prompt_id=FREEFORM_PROMPT_ID,
                    kind="freeform",
                    prompt=case["prompt"],
                    expected=case["expected"],
                    context_budget=case["context_budget"],
                    evaluation_role=case.get(
                        "freeform_evaluation_role", "diagnostic"
                    ),
                )
            )
            continue
        probe = probe_index[prompt_id]
        expected_letter = probe.get("expected_letter")
        if (
            not isinstance(expected_letter, str)
            or len(expected_letter) != 1
            or not expected_letter.isalpha()
        ):
            raise ValueError(
                f"{case['id']}/{prompt_id}: expected_letter must be one letter"
            )
        result.append(
            PromptSpec(
                case_id=case["id"],
                prompt_id=prompt_id,
                kind="choice",
                prompt=probe["prompt"],
                expected={
                    "letter": expected_letter.upper(),
                    "event_label": probe.get("expected_event_label"),
                },
                context_budget=probe["context_budget"],
                evaluation_role=probe.get(
                    "evaluation_role", "choice_probe"
                ),
            )
        )
    return result


def select_prompt_specs(
    cases: Sequence[Mapping[str, Any]],
    requested_prompt_ids: Sequence[str],
) -> list[PromptSpec]:
    return [
        spec
        for case in cases
        for spec in prompt_specs_for_case(case, requested_prompt_ids)
    ]


def _relative_frame_name(case_id: str, frame_index: int) -> Path:
    safe_component(case_id, "video case id")
    if (
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index < 0
    ):
        raise ValueError(f"Invalid frame index: {frame_index!r}")
    return Path(case_id) / f"frame_{frame_index:04d}.png"


def build_video_payload(
    case: Mapping[str, Any],
    asset: Mapping[str, Any],
    frame_root: Path,
    *,
    image_loader: Callable[[Path], Any],
    metadata_factory: Callable[..., Any],
    include_local_paths: bool = False,
) -> tuple[list[list[Any]], list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load original RGB frames and construct one metadata object per pair."""

    pairs = case.get("temporal_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{case['id']}: temporal_pairs must be non-empty")
    try:
        fps = float(asset["assumed_preview_fps"])
        total_frames = int(asset["frame_count"])
        duration = float(asset["documented_scenario_duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{case['id']}: clip asset lacks valid timing metadata"
        ) from exc
    if fps <= 0 or total_frames <= 0 or duration <= 0:
        raise ValueError(f"{case['id']}: clip timing metadata must be positive")

    videos: list[list[Any]] = []
    metadata: list[Any] = []
    frame_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    decoded_size: tuple[int, int] | None = None
    resolved_frame_root = frame_root.resolve()
    for pair_index, pair in enumerate(pairs):
        indices = pair.get("frame_indices")
        timestamps = pair.get("assumed_times_seconds")
        if (
            not isinstance(indices, list)
            or len(indices) != 2
            or not isinstance(timestamps, list)
            or len(timestamps) != 2
        ):
            raise ValueError(
                f"{case['id']}: temporal pair {pair_index} must contain "
                "two frame indices and timestamps"
            )
        images: list[Any] = []
        pair_frame_records: list[dict[str, Any]] = []
        for frame_index, timestamp in zip(indices, timestamps):
            relative = _relative_frame_name(case["id"], frame_index)
            frame_path = frame_root / relative
            if not frame_path.resolve().is_relative_to(resolved_frame_root):
                raise ValueError(
                    f"Extracted RGB frame escapes --frame-root: "
                    f"{relative.as_posix()}"
                )
            if not frame_path.is_file():
                raise ValueError(f"Missing extracted RGB frame: {relative.as_posix()}")
            image = image_loader(frame_path)
            size = tuple(int(value) for value in image.size)
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(f"Invalid decoded frame size for {relative}")
            if decoded_size is None:
                decoded_size = size
            elif size != decoded_size:
                raise ValueError(
                    f"Frame sizes differ: {size} versus {decoded_size}"
                )
            record: dict[str, Any] = {
                "file": relative.as_posix(),
                "frame_index": frame_index,
                "timestamp_seconds": float(timestamp),
                "sha256": sha256_file(frame_path),
                "size_bytes": frame_path.stat().st_size,
                "decoded_size": list(size),
                "mode": str(image.mode),
            }
            if include_local_paths:
                record["local_path"] = str(frame_path.resolve())
            images.append(image)
            frame_records.append(record)
            pair_frame_records.append(record)

        midpoint = sum(float(value) for value in timestamps) / 2
        expected_text = f"{midpoint:.1f}"
        if pair.get("prompt_timestamp_text") != expected_text:
            raise ValueError(
                f"{case['id']}: pair {pair_index} timestamp text "
                f"{pair.get('prompt_timestamp_text')!r} != {expected_text!r}"
            )
        videos.append(images)
        metadata.append(
            metadata_factory(
                total_num_frames=total_frames,
                fps=fps,
                width=decoded_size[0],
                height=decoded_size[1],
                duration=duration,
                frames_indices=list(indices),
            )
        )
        pair_records.append(
            {
                "pair_index": pair_index,
                "frame_indices": list(indices),
                "assumed_times_seconds": [
                    float(value) for value in timestamps
                ],
                "pair_midpoint_seconds": midpoint,
                "prompt_timestamp_text": expected_text,
                "frames": pair_frame_records,
            }
        )
    return videos, metadata, frame_records, pair_records


def build_messages(
    system_prompt: str,
    prompt: str,
    videos: Sequence[Sequence[Any]],
) -> list[dict[str, Any]]:
    if not system_prompt or not prompt:
        raise ValueError("System and user prompts must be non-empty")
    content: list[dict[str, Any]] = [
        {"type": "video", "video": list(video)} for video in videos
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def apply_upstream_processor(
    processor: Any,
    messages: list[dict[str, Any]],
    video_metadata: Sequence[Any],
) -> tuple[str, Any]:
    """Apply native chat/video processing without manual resize or sampling."""

    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        processor_kwargs={
            "do_sample_frames": False,
            "video_metadata": list(video_metadata),
            "return_mm_token_type_ids": True,
        },
    )
    return rendered, inputs


def validate_processor_outputs(inputs: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_PROCESSOR_OUTPUTS - inputs.keys())
    if missing:
        raise ValueError(
            "Upstream processor did not return required Qwen3-VL fields: "
            + ", ".join(missing)
        )


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes())


def tensor_record(tensor: Any) -> dict[str, Any]:
    shape = [int(value) for value in tensor.shape]
    return {
        "shape": shape,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": tensor_sha256(tensor),
    }


def split_pixel_records(
    pixel_values: Any,
    grid_values: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Hash each video's packed patch rows without changing the tensor."""

    records: list[dict[str, Any]] = []
    offset = 0
    for grid in grid_values:
        if len(grid) != 3:
            raise ValueError(f"Invalid video grid: {grid!r}")
        rows = int(grid[0]) * int(grid[1]) * int(grid[2])
        part = pixel_values[offset : offset + rows]
        if int(part.shape[0]) != rows:
            raise ValueError("Packed video tensor has fewer rows than its grids")
        records.append(tensor_record(part))
        offset += rows
    if offset != int(pixel_values.shape[0]):
        raise ValueError(
            "Packed video rows do not equal the sum of video_grid_thw"
        )
    return records


def visual_token_count(
    grid_values: Sequence[Sequence[int]],
    merge_size: int,
) -> int:
    if merge_size <= 0:
        raise ValueError("Vision merge size must be positive")
    divisor = merge_size**2
    total = 0
    for grid in grid_values:
        product = int(grid[0]) * int(grid[1]) * int(grid[2])
        if product % divisor:
            raise ValueError(f"Video grid is not divisible by merge size: {grid}")
        total += product // divisor
    return total


def validate_observed_contract(
    spec: PromptSpec,
    profile: Mapping[str, Any],
    *,
    prompt_tokens: int,
    visual_tokens: int,
    max_new_tokens: int,
) -> dict[str, int]:
    """Fail closed when local tokenization differs from the pinned manifest."""

    budget = spec.context_budget
    expected_prompt = int(budget["prompt_tokens"])
    expected_visual = int(budget["visual_tokens"])
    if prompt_tokens != expected_prompt:
        raise ValueError(
            f"{spec.case_id}/{spec.prompt_id}: processor produced "
            f"{prompt_tokens} prompt tokens; benchmark pins {expected_prompt}"
        )
    if visual_tokens != expected_visual:
        raise ValueError(
            f"{spec.case_id}/{spec.prompt_id}: processor produced "
            f"{visual_tokens} visual tokens; benchmark pins {expected_visual}"
        )
    context_size = int(profile["context_tokens"])
    prefill_limit = int(profile["prefill_kv_capacity_tokens"])
    if prompt_tokens > prefill_limit:
        raise ValueError(
            f"{spec.case_id}/{spec.prompt_id}: {prompt_tokens} prompt tokens "
            f"exceed the safe prefill limit {prefill_limit}"
        )
    if prompt_tokens + max_new_tokens > context_size:
        raise ValueError(
            f"{spec.case_id}/{spec.prompt_id}: prompt plus generation exceeds "
            f"context {context_size}"
        )
    return {
        "context_size": context_size,
        "prefill_safe_limit": prefill_limit,
        "prefill_headroom": prefill_limit - prompt_tokens,
        "generation_headroom": context_size
        - prompt_tokens
        - max_new_tokens,
    }


def extract_choice(
    answer: str,
    valid_letters: Iterable[str] = ("A", "B", "C", "D"),
) -> str | None:
    valid = {value.upper() for value in valid_letters}
    for pattern in (CHOICE_PREFIX, CHOICE_NAMED):
        match = pattern.search(answer)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()
    return None


def score_answer(spec: PromptSpec, answer: str) -> dict[str, Any]:
    if spec.kind == "freeform":
        return {
            "method": "manual_semantic_review_required",
            "pass": None,
            "evaluation_role": spec.evaluation_role,
            "note": "Compare the answer with the expected semantic rubric.",
        }
    expected = str(spec.expected["letter"]).upper()
    selected = extract_choice(answer)
    return {
        "method": "exact_choice",
        "pass": selected == expected,
        "expected_letter": expected,
        "selected_letter": selected,
        "evaluation_role": spec.evaluation_role,
    }


def checkpoint_provenance(
    checkpoint: Path,
    *,
    hash_weights: bool = True,
    include_local_paths: bool = False,
) -> dict[str, Any]:
    """Hash local checkpoint inputs while emitting only relative names."""

    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    files: set[Path] = {
        checkpoint / name
        for name in CHECKPOINT_METADATA_FILES
        if (checkpoint / name).is_file()
    }
    files.update(checkpoint.glob("*.safetensors.index.json"))
    if hash_weights:
        files.update(checkpoint.glob("*.safetensors"))
    records: dict[str, Any] = {}
    for path in sorted(files, key=lambda value: value.name):
        record = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if include_local_paths:
            record["local_path"] = str(path.resolve())
        records[path.name] = record
    required = {"config.json", "chat_template.json", "tokenizer_config.json"}
    missing = sorted(required - records.keys())
    if missing:
        raise ValueError(
            "Checkpoint is missing required provenance files: "
            + ", ".join(missing)
        )
    result: dict[str, Any] = {
        "identifier": checkpoint.name,
        "local_files_only": True,
        "weight_files_hashed": hash_weights,
        "files": records,
    }
    if include_local_paths:
        result["local_path"] = str(checkpoint.resolve())
    return result


def _load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        return source.convert("RGB")


def _assets_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    assets = {
        asset.get("id"): asset
        for asset in manifest.get("assets", [])
        if asset.get("kind") == "preview_clip"
    }
    if None in assets:
        raise ValueError("Preview clip assets must have ids")
    return assets


def _move_inputs(inputs: Mapping[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def resolve_cuda_device(torch: Any, requested: str) -> Any:
    """Resolve bare ``cuda`` to the active indexed device.

    Some Torch versions accept ``torch.device("cuda")`` for ``Tensor.to`` but
    reject that unindexed device in metadata calls such as
    ``torch.cuda.get_device_name``.
    """

    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("This BF16 reference runner requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    return device


def _timed_generate(
    model: Any,
    device_inputs: Mapping[str, Any],
    *,
    max_new_tokens: int,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() / 1024**3
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    with torch.inference_mode():
        generated = model.generate(
            **device_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    gpu_seconds = start_event.elapsed_time(end_event) / 1000
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return generated, {
        "wall_seconds": wall_seconds,
        "gpu_seconds": gpu_seconds,
        "baseline_allocated_gpu_gib": baseline,
        "peak_allocated_gpu_gib": peak,
        "peak_incremental_gpu_gib": peak - baseline,
    }


def run(args: argparse.Namespace) -> Path:
    """Execute the selected reference cases and return ``summary.json``."""

    if not args.device.startswith("cuda"):
        raise ValueError("This BF16 reference runner requires a CUDA device")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    safe_component(args.run_id, "run id")
    if args.output.exists():
        raise ValueError(f"Output directory already exists: {args.output}")

    # Heavy imports are intentionally confined to the execution path.
    import PIL
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.video_utils import VideoMetadata

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = resolve_cuda_device(torch, args.device)

    manifest, benchmark_raw = load_benchmark(args.benchmark)
    cases = select_video_cases(manifest, args.cases)
    specs = select_prompt_specs(cases, args.prompts)
    if not specs:
        raise ValueError("No prompts selected")
    case_index = {case["id"]: case for case in cases}
    assets = _assets_by_id(manifest)
    profile = manifest["video_case_profile"]
    profile_id = profile.get("id")
    expected_pair_count = int(profile["pair_count"])
    for case in cases:
        if case.get("profile_id") != profile_id:
            raise ValueError(
                f"{case['id']}: profile {case.get('profile_id')!r} does not "
                f"match video_case_profile {profile_id!r}"
            )
        if len(case.get("temporal_pairs", [])) != expected_pair_count:
            raise ValueError(
                f"{case['id']}: expected {expected_pair_count} temporal pairs"
            )
    system_prompt = args.system_prompt or profile.get("default_system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("No non-empty system prompt was supplied or configured")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )

    prepared: list[dict[str, Any]] = []
    for spec in specs:
        case = case_index[spec.case_id]
        try:
            asset = assets[case["clip_asset_id"]]
        except KeyError as exc:
            raise ValueError(
                f"{case['id']}: unknown clip asset {case.get('clip_asset_id')!r}"
            ) from exc
        videos, metadata, frames, pairs = build_video_payload(
            case,
            asset,
            args.frame_root,
            image_loader=_load_rgb_image,
            metadata_factory=VideoMetadata,
            include_local_paths=args.include_local_paths,
        )
        messages = build_messages(system_prompt, spec.prompt, videos)
        rendered, inputs = apply_upstream_processor(
            processor, messages, metadata
        )
        validate_processor_outputs(inputs)
        grids = [
            [int(value) for value in row]
            for row in inputs["video_grid_thw"].tolist()
        ]
        visual_tokens = visual_token_count(
            grids, int(processor.video_processor.merge_size)
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        contract = validate_observed_contract(
            spec,
            profile,
            prompt_tokens=prompt_tokens,
            visual_tokens=visual_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        pixel_parts = split_pixel_records(
            inputs["pixel_values_videos"], grids
        )
        if len(pixel_parts) != len(pairs):
            raise ValueError(
                f"{case['id']}: processor returned {len(pixel_parts)} videos "
                f"for {len(pairs)} temporal pairs"
            )
        for pair, pixels in zip(pairs, pixel_parts):
            pair["pixel_values"] = pixels
            pair["grid_thw"] = grids[pair["pair_index"]]
        expanded_prompt = processor.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        prepared.append(
            {
                "spec": spec,
                "case": case,
                "frames": frames,
                "pairs": pairs,
                "rendered": rendered,
                "expanded_prompt_sha256": sha256_bytes(
                    expanded_prompt.encode("utf-8")
                ),
                "inputs": inputs,
                "grids": grids,
                "visual_tokens": visual_tokens,
                "prompt_tokens": prompt_tokens,
                "contract": contract,
            }
        )

    checkpoint = checkpoint_provenance(
        args.checkpoint,
        hash_weights=not args.skip_weight_hash,
        include_local_paths=args.include_local_paths,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    allocated_after_load = torch.cuda.memory_allocated() / 1024**3
    peak_after_load = torch.cuda.max_memory_allocated() / 1024**3

    if not args.no_warmup:
        warm_inputs = _move_inputs(
            prepared[0]["inputs"], device, torch
        )
        with torch.inference_mode():
            model.generate(
                **warm_inputs,
                max_new_tokens=1,
                do_sample=False,
            )
        torch.cuda.synchronize()

    created = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    for item in prepared:
        spec: PromptSpec = item["spec"]
        device_inputs = _move_inputs(item["inputs"], device, torch)
        generated, timing = _timed_generate(
            model,
            device_inputs,
            max_new_tokens=args.max_new_tokens,
            torch=torch,
        )
        output_ids = generated[0, item["prompt_tokens"] :].detach().cpu()
        answer = processor.tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip()
        tokens_per_second = (
            int(output_ids.numel()) / timing["gpu_seconds"]
            if timing["gpu_seconds"]
            else None
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "created_utc": created,
            "case_id": spec.case_id,
            "prompt_id": spec.prompt_id,
            "prompt_kind": spec.kind,
            "evaluation_role": spec.evaluation_role,
            "profile_id": item["case"]["profile_id"],
            "clip_asset_id": item["case"]["clip_asset_id"],
            "benchmark": {
                "id": manifest.get("benchmark_id"),
                "file": args.benchmark.name,
                "sha256": sha256_bytes(benchmark_raw),
                "source_revision": manifest.get("source", {}).get("revision"),
            },
            "question": spec.prompt,
            "expected": dict(spec.expected),
            "ground_truth_basis": item["case"].get("ground_truth_basis"),
            "system_prompt": system_prompt,
            "frames": item["frames"],
            "temporal_pairs": item["pairs"],
            "processor": {
                "method": (
                    "AutoProcessor.apply_chat_template with one native video "
                    "block per temporal pair"
                ),
                "do_sample_frames": False,
                "manual_resize": False,
                "rendered_chat_template": item["rendered"],
                "rendered_chat_template_sha256": sha256_bytes(
                    item["rendered"].encode("utf-8")
                ),
                "expanded_prompt_sha256": item[
                    "expanded_prompt_sha256"
                ],
            },
            "prompt_tokens": item["prompt_tokens"],
            "text_tokens": item["prompt_tokens"] - item["visual_tokens"],
            "visual_tokens": item["visual_tokens"],
            "context_contract": item["contract"],
            "input_ids": tensor_record(item["inputs"]["input_ids"]),
            "attention_mask": tensor_record(
                item["inputs"]["attention_mask"]
            ),
            "mm_token_type_ids": tensor_record(
                item["inputs"]["mm_token_type_ids"]
            ),
            "pixel_values_videos": tensor_record(
                item["inputs"]["pixel_values_videos"]
            ),
            "video_grid_thw": tensor_record(
                item["inputs"]["video_grid_thw"]
            ),
            "video_grid_thw_values": item["grids"],
            "answer": answer,
            "answer_sha256": sha256_bytes(answer.encode("utf-8")),
            "output_token_ids": output_ids.tolist(),
            "generated_tokens": int(output_ids.numel()),
            "max_new_tokens": args.max_new_tokens,
            "sampling": "greedy",
            "seed": args.seed,
            "rubric_assessment": score_answer(spec, answer),
            "timing": {
                **timing,
                "generated_tokens_per_second": tokens_per_second,
            },
        }
        if args.include_local_paths:
            result["local_paths"] = {
                "benchmark": str(args.benchmark.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "frame_root": str(args.frame_root.resolve()),
            }
        results.append(result)

    args.output.mkdir(parents=True)
    result_entries: list[dict[str, Any]] = []
    for result in results:
        filename = (
            safe_component(result["case_id"], "case id")
            + "."
            + safe_component(result["prompt_id"], "prompt id")
            + ".gpu_bf16.json"
        )
        path = args.output / filename
        write_json(path, result)
        result_entries.append(
            {
                "case_id": result["case_id"],
                "prompt_id": result["prompt_id"],
                "file": filename,
                "sha256": sha256_file(path),
                "answer": result["answer"],
                "pass": result["rubric_assessment"]["pass"],
                "prompt_tokens": result["prompt_tokens"],
                "input_ids_sha256": result["input_ids"]["sha256"],
                "pixel_values_sha256": result[
                    "pixel_values_videos"
                ]["sha256"],
                "gpu_seconds": result["timing"]["gpu_seconds"],
            }
        )

    choice_results = [
        entry for entry, result in zip(result_entries, results)
        if result["prompt_kind"] == "choice"
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": created,
        "run_id": args.run_id,
        "purpose": (
            "Deterministic host BF16 reference using upstream native "
            "Qwen3-VL paired-video preprocessing"
        ),
        "benchmark": {
            "id": manifest.get("benchmark_id"),
            "file": args.benchmark.name,
            "sha256": sha256_bytes(benchmark_raw),
            "source_revision": manifest.get("source", {}).get("revision"),
        },
        "checkpoint": checkpoint,
        "execution": {
            "precision": "bfloat16",
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(device),
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "video_processor_class": type(
                processor.video_processor
            ).__name__,
            "max_new_tokens": args.max_new_tokens,
            "sampling": "greedy",
            "seed": args.seed,
            "warmup": not args.no_warmup,
            "load_wall_seconds": load_seconds,
            "allocated_gpu_gib_after_load": allocated_after_load,
            "peak_gpu_gib_after_load": peak_after_load,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pillow": PIL.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "selection": {
            "case_ids": [case["id"] for case in cases],
            "prompt_ids": list(args.prompts) or [ALL_PROMPTS_ID],
        },
        "results": result_entries,
        "assessment": {
            "choice_passed": sum(
                entry["pass"] is True for entry in choice_results
            ),
            "choice_total": len(choice_results),
            "freeform_requires_manual_review": sum(
                result["prompt_kind"] == "freeform" for result in results
            ),
        },
        "privacy": {
            "local_absolute_paths_recorded": args.include_local_paths,
            "note": (
                "Local paths are omitted by default; frame names are relative "
                "to --frame-root."
            ),
        },
    }
    if args.include_local_paths:
        summary["local_paths"] = {
            "benchmark": str(args.benchmark.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "frame_root": str(args.frame_root.resolve()),
            "output": str(args.output.resolve()),
        }
    summary_path = args.output / "summary.json"
    write_json(summary_path, summary)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=[],
        help="video case id; repeatable (default: every video case)",
    )
    parser.add_argument(
        "--prompt",
        "--probe",
        dest="prompts",
        action="append",
        default=[],
        help=(
            "freeform, all, or a choice_probes id; repeatable "
            "(default: all)"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help=(
            "override video_case_profile.default_system_prompt "
            "(default: use the benchmark value)"
        ),
    )
    parser.add_argument("--run-id", default="cosmos-video-gpu-bf16")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="do not run the default one-token deterministic warmup",
    )
    parser.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="skip hashing large safetensor weights (metadata is still hashed)",
    )
    parser.add_argument(
        "--include-local-paths",
        action="store_true",
        help="opt in to recording local absolute paths in JSON artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
