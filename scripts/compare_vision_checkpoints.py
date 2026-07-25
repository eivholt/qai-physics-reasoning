#!/usr/bin/env python3
"""Compare quantized Cosmos vision checkpoints with the adapted BF16 model.

The inputs are the exact packed FP32 pixel tensors recorded by one or more
``video_npu_manifest.json`` files.  The script validates every recorded shape,
byte count, and SHA-256 before loading a model.  It then runs the corrected
adapted floating-point vision encoder in BF16 once, followed by each quantized
checkpoint in isolation, and writes deterministic per-case and aggregate
numeric-fidelity results.

Heavy dependencies are imported only after argument and input validation, so
``--help`` and unit tests for the manifest/metric helpers do not load Torch,
Transformers, AIMET, ONNX Runtime, or the model.

Example:

    python scripts/compare_vision_checkpoints.py \
      --source-checkpoint /models/Cosmos-Reason2-2B \
      --candidate-checkpoint baseline=/checkpoints/vision-w8a16 \
      --candidate-checkpoint block23=/checkpoints/vision-b23fp16 \
      --manifest /runs/box/video_npu_manifest.json \
      --manifest /runs/near-miss/video_npu_manifest.json \
      --output-json /results/vision-checkpoint-comparison.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

OUTPUT_NAMES = (
    "image_features",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
)
FLOAT32_BYTES = 4
VISION_PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
SPATIAL_MERGE_SIZE = 2
RGB_CHANNELS = 3
VISION_OUTPUT_HIDDEN_SIZE = 2048
PIXEL_COLUMNS = (
    RGB_CHANNELS
    * TEMPORAL_PATCH_SIZE
    * VISION_PATCH_SIZE
    * VISION_PATCH_SIZE
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class VisionGeometry:
    """Static Cosmos/Qwen3-VL tensor geometry for one temporal pair."""

    image_height: int
    image_width: int
    pixel_shape: tuple[int, int]
    grid_thw: tuple[int, int, int]
    output_shape: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_size_height_width": [
                self.image_height,
                self.image_width,
            ],
            "pixel_values_shape": list(self.pixel_shape),
            "grid_thw": list(self.grid_thw),
            "vision_output_shape": list(self.output_shape),
        }


@dataclass(frozen=True)
class InputCase:
    """One validated packed temporal-pair input."""

    case_id: str
    manifest_path: Path
    manifest_sha256: str
    pair_index: int
    pixel_path: Path
    pixel_recorded_path: str
    pixel_sha256: str
    pixel_values: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ValidatedManifest:
    """A validated input manifest and its cases."""

    path: Path
    sha256: str
    geometry: VisionGeometry
    cases: tuple[InputCase, ...]


@dataclass(frozen=True)
class CandidateCheckpoint:
    """Validated quantized checkpoint metadata."""

    label: str
    path: Path
    geometry_height_width: tuple[int, int] | None
    files: dict[str, dict[str, Any]]


@dataclass
class MetricAccumulator:
    """Streaming sufficient statistics for the five reported metrics."""

    count: int = 0
    dot: float = 0.0
    reference_squared_norm: float = 0.0
    candidate_squared_norm: float = 0.0
    difference_squared_norm: float = 0.0
    absolute_error_sum: float = 0.0
    max_absolute_error: float = 0.0

    def update(self, reference: Any, candidate: Any) -> None:
        """Accumulate one equal-shaped finite array pair in float64."""

        np = _numpy()
        reference_array = np.asarray(reference, dtype=np.float64)
        candidate_array = np.asarray(candidate, dtype=np.float64)
        _validate_metric_arrays(reference_array, candidate_array)

        reference_flat = reference_array.reshape(-1)
        candidate_flat = candidate_array.reshape(-1)
        difference = candidate_flat - reference_flat
        absolute_error = np.abs(difference)

        self.count += int(reference_flat.size)
        self.dot += float(np.dot(reference_flat, candidate_flat))
        self.reference_squared_norm += float(
            np.dot(reference_flat, reference_flat)
        )
        self.candidate_squared_norm += float(
            np.dot(candidate_flat, candidate_flat)
        )
        self.difference_squared_norm += float(
            np.dot(difference, difference)
        )
        self.absolute_error_sum += float(absolute_error.sum(dtype=np.float64))
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(absolute_error.max()),
        )

    def metrics(self) -> dict[str, float | None]:
        """Return metrics over every element accumulated so far."""

        if self.count <= 0:
            raise ValueError("Cannot compute aggregate metrics with no values")
        return _metrics_from_sums(
            count=self.count,
            dot=self.dot,
            reference_squared_norm=self.reference_squared_norm,
            candidate_squared_norm=self.candidate_squared_norm,
            difference_squared_norm=self.difference_squared_norm,
            absolute_error_sum=self.absolute_error_sum,
            max_absolute_error=self.max_absolute_error,
        )


def _numpy() -> Any:
    """Import NumPy only when validation or metrics are actually requested."""

    import numpy

    return numpy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_float32(array: Any) -> str:
    """Hash an array in canonical C-order little-endian float32 form."""

    np = _numpy()
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_metric_arrays(reference: Any, candidate: Any) -> None:
    np = _numpy()
    if reference.shape != candidate.shape:
        raise ValueError(
            "Metric arrays have different shapes: "
            f"{reference.shape} != {candidate.shape}"
        )
    if reference.size == 0:
        raise ValueError("Metric arrays must not be empty")
    if not bool(np.isfinite(reference).all()):
        raise ValueError("Reference array contains a non-finite value")
    if not bool(np.isfinite(candidate).all()):
        raise ValueError("Candidate array contains a non-finite value")


def _metrics_from_sums(
    *,
    count: int,
    dot: float,
    reference_squared_norm: float,
    candidate_squared_norm: float,
    difference_squared_norm: float,
    absolute_error_sum: float,
    max_absolute_error: float,
) -> dict[str, float | None]:
    reference_norm = math.sqrt(max(reference_squared_norm, 0.0))
    candidate_norm = math.sqrt(max(candidate_squared_norm, 0.0))
    difference_norm = math.sqrt(max(difference_squared_norm, 0.0))

    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine: float | None = 1.0
        relative_l2: float | None = 0.0
        norm_ratio: float | None = 1.0
    elif reference_norm == 0.0:
        cosine = None
        relative_l2 = None
        norm_ratio = None
    elif candidate_norm == 0.0:
        cosine = None
        relative_l2 = difference_norm / reference_norm
        norm_ratio = 0.0
    else:
        cosine = dot / (reference_norm * candidate_norm)
        # Bound only roundoff excursions; meaningful values remain untouched.
        cosine = min(1.0, max(-1.0, cosine))
        relative_l2 = difference_norm / reference_norm
        norm_ratio = candidate_norm / reference_norm

    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "mean_abs_error": absolute_error_sum / count,
        "max_abs_error": max_absolute_error,
        "norm_ratio": norm_ratio,
    }


def compute_metrics(
    reference: Any,
    candidate: Any,
) -> dict[str, float | None]:
    """Compute fidelity metrics after finite/shape validation."""

    accumulator = MetricAccumulator()
    accumulator.update(reference, candidate)
    return accumulator.metrics()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if parsed < 0 or parsed != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _shape(value: Any, length: int, label: str) -> tuple[int, ...]:
    values = _list(value, label)
    if len(values) != length:
        raise ValueError(f"{label} must contain {length} dimensions")
    return tuple(
        _positive_int(dimension, f"{label}[{index}]")
        for index, dimension in enumerate(values)
    )


def _geometry_from_contract(contract: dict[str, Any]) -> VisionGeometry:
    image_size = _shape(contract.get("image_size"), 2, "graph_contract.image_size")
    image_width, image_height = image_size
    if (
        image_height % VISION_PATCH_SIZE
        or image_width % VISION_PATCH_SIZE
    ):
        raise ValueError(
            "graph_contract.image_size must contain multiples of "
            f"{VISION_PATCH_SIZE}"
        )

    grid_thw = _shape(contract.get("grid_thw"), 3, "graph_contract.grid_thw")
    expected_grid = (
        1,
        image_height // VISION_PATCH_SIZE,
        image_width // VISION_PATCH_SIZE,
    )
    if grid_thw != expected_grid:
        raise ValueError(
            f"graph_contract.grid_thw is {grid_thw}; expected {expected_grid}"
        )
    if (
        grid_thw[1] % SPATIAL_MERGE_SIZE
        or grid_thw[2] % SPATIAL_MERGE_SIZE
    ):
        raise ValueError(
            "Vision patch grid must be divisible by spatial merge size "
            f"{SPATIAL_MERGE_SIZE}"
        )

    pixel_shape = _shape(
        contract.get("pixel_values_shape"),
        2,
        "graph_contract.pixel_values_shape",
    )
    expected_pixel_shape = (math.prod(grid_thw), PIXEL_COLUMNS)
    if pixel_shape != expected_pixel_shape:
        raise ValueError(
            "graph_contract.pixel_values_shape is "
            f"{pixel_shape}; expected {expected_pixel_shape}"
        )

    output_shape = _shape(
        contract.get("image_features_shape"),
        2,
        "graph_contract.image_features_shape",
    )
    expected_output_shape = (
        math.prod(grid_thw) // (SPATIAL_MERGE_SIZE**2),
        VISION_OUTPUT_HIDDEN_SIZE,
    )
    if output_shape != expected_output_shape:
        raise ValueError(
            "graph_contract.image_features_shape is "
            f"{output_shape}; expected {expected_output_shape}"
        )

    return VisionGeometry(
        image_height=image_height,
        image_width=image_width,
        pixel_shape=(pixel_shape[0], pixel_shape[1]),
        grid_thw=(grid_thw[0], grid_thw[1], grid_thw[2]),
        output_shape=(output_shape[0], output_shape[1]),
    )


def _resolve_recorded_file(manifest_path: Path, recorded: str) -> Path:
    relative = Path(recorded)
    if relative.is_absolute():
        raise ValueError(
            "Recorded pixel file must be relative to its manifest: "
            f"{recorded}"
        )
    manifest_root = manifest_path.parent.resolve()
    resolved = (manifest_root / relative).resolve()
    if resolved != manifest_root and manifest_root not in resolved.parents:
        raise ValueError(
            f"Recorded pixel file escapes the manifest directory: {recorded}"
        )
    return resolved


def _load_manifest(path: Path, manifest_ordinal: int) -> ValidatedManifest:
    np = _numpy()
    manifest_path = path.expanduser().resolve()
    payload = _load_json_object(manifest_path, "video NPU manifest")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported video manifest schema_version in {manifest_path}"
        )
    manifest_sha256 = sha256_file(manifest_path)
    contract = _object(payload.get("graph_contract"), "graph_contract")
    geometry = _geometry_from_contract(contract)

    pairs = _list(payload.get("pairs"), "pairs")
    if not pairs:
        raise ValueError(f"Manifest contains no temporal pairs: {manifest_path}")

    cases: list[InputCase] = []
    seen_pair_indices: set[int] = set()
    for pair_ordinal, raw_pair in enumerate(pairs):
        pair = _object(raw_pair, f"pairs[{pair_ordinal}]")
        pair_index = _nonnegative_int(
            pair.get("pair_index"),
            f"pairs[{pair_ordinal}].pair_index",
        )
        if pair_index in seen_pair_indices:
            raise ValueError(
                f"Duplicate pair_index {pair_index} in {manifest_path}"
            )
        seen_pair_indices.add(pair_index)

        pixel = _object(
            pair.get("pixel_values"),
            f"pairs[{pair_ordinal}].pixel_values",
        )
        recorded_path = pixel.get("file")
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ValueError(
                f"pairs[{pair_ordinal}].pixel_values.file must be a string"
            )
        if pixel.get("serialization") != "float32 little-endian raw":
            raise ValueError(
                f"pairs[{pair_ordinal}] has unsupported pixel serialization"
            )
        pair_shape = _shape(
            pixel.get("shape"),
            2,
            f"pairs[{pair_ordinal}].pixel_values.shape",
        )
        if pair_shape != geometry.pixel_shape:
            raise ValueError(
                f"Pair {pair_index} pixel shape is {pair_shape}; expected "
                f"{geometry.pixel_shape}"
            )
        pair_grid = _shape(
            pixel.get("grid_thw"),
            3,
            f"pairs[{pair_ordinal}].pixel_values.grid_thw",
        )
        if pair_grid != geometry.grid_thw:
            raise ValueError(
                f"Pair {pair_index} grid is {pair_grid}; expected "
                f"{geometry.grid_thw}"
            )

        declared_bytes = _positive_int(
            pixel.get("bytes"),
            f"pairs[{pair_ordinal}].pixel_values.bytes",
        )
        expected_bytes = math.prod(geometry.pixel_shape) * FLOAT32_BYTES
        if declared_bytes != expected_bytes:
            raise ValueError(
                f"Pair {pair_index} declares {declared_bytes} pixel bytes; "
                f"expected {expected_bytes}"
            )

        declared_sha256 = pixel.get("sha256")
        if (
            not isinstance(declared_sha256, str)
            or SHA256_RE.fullmatch(declared_sha256) is None
        ):
            raise ValueError(
                f"Pair {pair_index} has no valid lowercase SHA-256"
            )

        pixel_path = _resolve_recorded_file(manifest_path, recorded_path)
        if not pixel_path.is_file():
            raise ValueError(f"Pair {pair_index} pixel file is missing: {pixel_path}")
        actual_bytes = pixel_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Pair {pair_index} pixel file has {actual_bytes} bytes; "
                f"expected {expected_bytes}"
            )
        actual_sha256 = sha256_file(pixel_path)
        if actual_sha256 != declared_sha256:
            raise ValueError(
                f"Pair {pair_index} pixel SHA-256 mismatch: "
                f"{actual_sha256} != {declared_sha256}"
            )

        values = np.fromfile(pixel_path, dtype=np.dtype("<f4"))
        values = values.reshape(geometry.pixel_shape)
        if not bool(np.isfinite(values).all()):
            raise ValueError(
                f"Pair {pair_index} pixel tensor contains a non-finite value"
            )
        values.setflags(write=False)
        case_id = (
            f"{manifest_ordinal:03d}:{manifest_path.stem}:"
            f"{manifest_sha256[:12]}:pair_{pair_index:03d}"
        )
        cases.append(
            InputCase(
                case_id=case_id,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                pair_index=pair_index,
                pixel_path=pixel_path,
                pixel_recorded_path=recorded_path,
                pixel_sha256=actual_sha256,
                pixel_values=values,
            )
        )

    cases.sort(key=lambda case: case.pair_index)
    return ValidatedManifest(
        path=manifest_path,
        sha256=manifest_sha256,
        geometry=geometry,
        cases=tuple(cases),
    )


def load_input_manifests(paths: Sequence[Path]) -> tuple[ValidatedManifest, ...]:
    """Load and validate manifests in stable path order."""

    if not paths:
        raise ValueError("At least one video NPU manifest is required")
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same video NPU manifest was supplied more than once")
    return tuple(
        _load_manifest(path, ordinal)
        for ordinal, path in enumerate(sorted(resolved, key=str))
    )


def _candidate_geometry(args_path: Path) -> tuple[int, int] | None:
    if not args_path.is_file():
        return None
    payload = _load_json_object(args_path, "candidate args.json")
    raw_image_size = payload.get("image_size")
    if raw_image_size is None:
        return None
    image_size = _shape(raw_image_size, 2, f"{args_path}: image_size")
    image_height, image_width = image_size
    if (
        image_height % VISION_PATCH_SIZE
        or image_width % VISION_PATCH_SIZE
    ):
        raise ValueError(
            f"{args_path}: image_size must contain multiples of "
            f"{VISION_PATCH_SIZE}"
        )
    return image_height, image_width


def _parse_candidate_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        label = Path(raw_path).name
    if not label or CANDIDATE_LABEL_RE.fullmatch(label) is None:
        raise ValueError(
            "Candidate label must contain only letters, numbers, '.', '_', "
            f"or '-': {label!r}"
        )
    if not raw_path:
        raise ValueError(f"Candidate checkpoint path is empty: {spec!r}")
    return label, Path(raw_path).expanduser().resolve()


def validate_candidate_checkpoints(
    specs: Sequence[str],
) -> tuple[CandidateCheckpoint, ...]:
    """Validate quantized checkpoint files and collect artifact hashes."""

    if not specs:
        raise ValueError("At least one candidate checkpoint is required")
    candidates: list[CandidateCheckpoint] = []
    labels: set[str] = set()
    paths: set[Path] = set()
    for spec in specs:
        label, path = _parse_candidate_spec(spec)
        if label in labels:
            raise ValueError(f"Duplicate candidate label: {label}")
        if path in paths:
            raise ValueError(f"Duplicate candidate checkpoint path: {path}")
        labels.add(label)
        paths.add(path)
        if not path.is_dir():
            raise ValueError(f"Candidate checkpoint is not a directory: {path}")

        files: dict[str, dict[str, Any]] = {}
        for filename in ("vision_encoder.onnx", "vision_encoder.encodings"):
            artifact = path / filename
            if not artifact.is_file():
                raise ValueError(
                    f"Candidate {label} is missing required file: {artifact}"
                )
            files[filename] = {
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        args_path = path / "args.json"
        if args_path.is_file():
            files["args.json"] = {
                "bytes": args_path.stat().st_size,
                "sha256": sha256_file(args_path),
            }
        candidates.append(
            CandidateCheckpoint(
                label=label,
                path=path,
                geometry_height_width=_candidate_geometry(args_path),
                files=files,
            )
        )
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.label, str(candidate.path)))
    )


def resolve_geometry(
    manifests: Sequence[ValidatedManifest],
    candidates: Sequence[CandidateCheckpoint],
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> VisionGeometry:
    """Resolve one shared geometry and reject every conflicting declaration."""

    if not manifests:
        raise ValueError("No validated manifests were supplied")
    if (image_height is None) != (image_width is None):
        raise ValueError(
            "--image-height and --image-width must be supplied together"
        )

    geometry = manifests[0].geometry
    for manifest in manifests[1:]:
        if manifest.geometry != geometry:
            raise ValueError(
                "All manifests must use one vision geometry: "
                f"{manifest.path} differs from {manifests[0].path}"
            )

    expected_height_width = (geometry.image_height, geometry.image_width)
    if image_height is not None and image_width is not None:
        override = (
            _positive_int(image_height, "--image-height"),
            _positive_int(image_width, "--image-width"),
        )
        if override != expected_height_width:
            raise ValueError(
                "Explicit image geometry does not match the manifests: "
                f"{override} != {expected_height_width}"
            )

    for candidate in candidates:
        if (
            candidate.geometry_height_width is not None
            and candidate.geometry_height_width != expected_height_width
        ):
            raise ValueError(
                f"Candidate {candidate.label} args.json image_size "
                f"{candidate.geometry_height_width} does not match manifest "
                f"geometry {expected_height_width}"
            )
    return geometry


def _source_identity(source: Path) -> dict[str, Any]:
    source_path = source.expanduser().resolve()
    if not source_path.is_dir():
        raise ValueError(
            f"Source checkpoint is not a local directory: {source_path}"
        )
    config = source_path / "config.json"
    if not config.is_file():
        raise ValueError(
            f"Source checkpoint is missing required config.json: {source_path}"
        )

    files: dict[str, dict[str, Any]] = {}
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
    ):
        path = source_path / filename
        if path.is_file():
            files[filename] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "path": str(source_path),
        "metadata_files": files,
    }


def _canonical_output(array: Any, expected_shape: tuple[int, int], label: str) -> Any:
    np = _numpy()
    canonical = np.ascontiguousarray(array, dtype=np.float32)
    if tuple(canonical.shape) != expected_shape:
        raise ValueError(
            f"{label} has shape {tuple(canonical.shape)}; expected "
            f"{expected_shape}"
        )
    if not bool(np.isfinite(canonical).all()):
        raise ValueError(f"{label} contains a non-finite value")
    canonical.setflags(write=False)
    return canonical


def _normalize_outputs(
    outputs: Any,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> dict[str, Any]:
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    if len(outputs) != len(OUTPUT_NAMES):
        raise ValueError(
            f"{label} returned {len(outputs)} outputs; expected "
            f"{len(OUTPUT_NAMES)}"
        )

    normalized: dict[str, Any] = {}
    for name, output in zip(OUTPUT_NAMES, outputs, strict=True):
        if hasattr(output, "detach"):
            output = output.detach()
        if hasattr(output, "float"):
            output = output.float()
        if hasattr(output, "cpu"):
            output = output.cpu()
        if hasattr(output, "numpy"):
            output = output.numpy()
        normalized[name] = _canonical_output(
            output,
            expected_shape,
            f"{label} output {name}",
        )
    return normalized


def _run_reference(
    *,
    model_class: Any,
    precision: Any,
    torch: Any,
    source: Path,
    cases: Sequence[InputCase],
    geometry: VisionGeometry,
    device: Any,
) -> dict[str, dict[str, Any]]:
    model = model_class.from_pretrained(
        checkpoint=source,
        device=device,
        image_height=geometry.image_height,
        image_width=geometry.image_width,
        precision=precision.float,
    )
    projection = getattr(getattr(model, "patch_embed", None), "proj", None)
    conv2d = getattr(projection, "conv2d", None)
    if conv2d is None or getattr(conv2d, "bias", None) is None:
        raise RuntimeError(
            "Adapted BF16 reference has no restored temporal patch bias"
        )
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    results: dict[str, dict[str, Any]] = {}
    output_cache: dict[str, dict[str, Any]] = {}
    try:
        with torch.inference_mode():
            for case in cases:
                cached = output_cache.get(case.pixel_sha256)
                if cached is not None:
                    results[case.case_id] = cached
                    continue
                values = case.pixel_values.copy()
                pixel_values = torch.from_numpy(values).to(
                    device=device,
                    dtype=torch.bfloat16,
                )
                outputs = model(pixel_values)
                results[case.case_id] = _normalize_outputs(
                    outputs,
                    expected_shape=geometry.output_shape,
                    label=f"BF16 reference case {case.case_id}",
                )
                output_cache[case.pixel_sha256] = results[case.case_id]
                del pixel_values, outputs
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _run_candidate(
    *,
    model_class: Any,
    precision: Any,
    torch: Any,
    candidate: CandidateCheckpoint,
    cases: Sequence[InputCase],
    geometry: VisionGeometry,
    device: Any,
) -> dict[str, dict[str, Any]]:
    model = model_class.from_pretrained(
        checkpoint=candidate.path,
        device=device,
        image_height=geometry.image_height,
        image_width=geometry.image_width,
        precision=precision.w4a16,
    )
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    output_cache: dict[str, dict[str, Any]] = {}
    try:
        with torch.inference_mode():
            for case in cases:
                cached = output_cache.get(case.pixel_sha256)
                if cached is not None:
                    results[case.case_id] = cached
                    continue
                # AIMET/ORT consumes the recorded FP32 bytes.  Keep this tensor
                # on CPU; the configured execution provider performs any
                # required transfer.
                pixel_values = torch.from_numpy(case.pixel_values.copy())
                outputs = model(pixel_values)
                results[case.case_id] = _normalize_outputs(
                    outputs,
                    expected_shape=geometry.output_shape,
                    label=f"Candidate {candidate.label} case {case.case_id}",
                )
                output_cache[case.pixel_sha256] = results[case.case_id]
                del pixel_values, outputs
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _all_cases(
    manifests: Sequence[ValidatedManifest],
) -> tuple[InputCase, ...]:
    return tuple(case for manifest in manifests for case in manifest.cases)


def _manifest_report(manifest: ValidatedManifest) -> dict[str, Any]:
    return {
        "path": str(manifest.path),
        "sha256": manifest.sha256,
        "pair_count": len(manifest.cases),
    }


def _reference_case_report(
    case: InputCase,
    outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "manifest_path": str(case.manifest_path),
        "manifest_sha256": case.manifest_sha256,
        "pair_index": case.pair_index,
        "pixel_values": {
            "recorded_file": case.pixel_recorded_path,
            "resolved_path": str(case.pixel_path),
            "sha256": case.pixel_sha256,
            "shape": list(case.pixel_values.shape),
            "serialization": "float32 little-endian raw",
        },
        "reference_outputs": {
            name: {
                "shape": list(output.shape),
                "canonical_float32_sha256": sha256_float32(output),
            }
            for name, output in outputs.items()
        },
    }


def _candidate_report(
    *,
    candidate: CandidateCheckpoint,
    cases: Sequence[InputCase],
    reference_outputs: dict[str, dict[str, Any]],
    candidate_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    accumulators = {name: MetricAccumulator() for name in OUTPUT_NAMES}
    reference_hashes = {name: hashlib.sha256() for name in OUTPUT_NAMES}
    candidate_hashes = {name: hashlib.sha256() for name in OUTPUT_NAMES}
    case_reports: list[dict[str, Any]] = []
    np = _numpy()

    for case in cases:
        outputs_report: dict[str, Any] = {}
        for name in OUTPUT_NAMES:
            reference = reference_outputs[case.case_id][name]
            result = candidate_outputs[case.case_id][name]
            accumulators[name].update(reference, result)
            reference_bytes = np.ascontiguousarray(
                reference, dtype=np.dtype("<f4")
            ).tobytes(order="C")
            candidate_bytes = np.ascontiguousarray(
                result, dtype=np.dtype("<f4")
            ).tobytes(order="C")
            reference_hashes[name].update(reference_bytes)
            candidate_hashes[name].update(candidate_bytes)
            outputs_report[name] = {
                "shape": list(result.shape),
                "reference_canonical_float32_sha256": hashlib.sha256(
                    reference_bytes
                ).hexdigest(),
                "candidate_canonical_float32_sha256": hashlib.sha256(
                    candidate_bytes
                ).hexdigest(),
                "metrics": compute_metrics(reference, result),
            }
        case_reports.append(
            {
                "case_id": case.case_id,
                "pixel_values_sha256": case.pixel_sha256,
                "outputs": outputs_report,
            }
        )

    aggregate = {
        name: {
            "element_count": accumulators[name].count,
            "reference_concatenated_float32_sha256": (
                reference_hashes[name].hexdigest()
            ),
            "candidate_concatenated_float32_sha256": (
                candidate_hashes[name].hexdigest()
            ),
            "metrics": accumulators[name].metrics(),
        }
        for name in OUTPUT_NAMES
    }
    return {
        "label": candidate.label,
        "checkpoint": {
            "path": str(candidate.path),
            "precision_api": "Precision.w4a16",
            "args_image_size_height_width": (
                list(candidate.geometry_height_width)
                if candidate.geometry_height_width is not None
                else None
            ),
            "files": candidate.files,
        },
        "cases": case_reports,
        "aggregate": aggregate,
    }


def compare_vision_checkpoints(
    *,
    source_checkpoint: Path,
    candidate_specs: Sequence[str],
    manifest_paths: Sequence[Path],
    output_json: Path,
    device_name: str = "auto",
    image_height: int | None = None,
    image_width: int | None = None,
) -> dict[str, Any]:
    """Run the manifest-driven BF16-versus-quantized comparison."""

    manifests = load_input_manifests(manifest_paths)
    candidates = validate_candidate_checkpoints(candidate_specs)
    geometry = resolve_geometry(
        manifests,
        candidates,
        image_height=image_height,
        image_width=image_width,
    )
    source_identity = _source_identity(source_checkpoint)
    source = Path(source_identity["path"])
    cases = _all_cases(manifests)

    # Deliberately lazy: reaching this point means every local artifact has
    # already passed lightweight validation.
    import torch
    from qai_hub_models import Precision
    from qai_hub_models.models.cosmos_reason2_2b.model import (
        Cosmos_Reason2_2B_VisionEncoder,
        configure_source_checkpoint,
    )

    if device_name == "auto":
        resolved_device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device_name = device_name
    if resolved_device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(resolved_device_name)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    configure_source_checkpoint(source)
    reference_outputs = _run_reference(
        model_class=Cosmos_Reason2_2B_VisionEncoder,
        precision=Precision,
        torch=torch,
        source=source,
        cases=cases,
        geometry=geometry,
        device=device,
    )

    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_outputs = _run_candidate(
            model_class=Cosmos_Reason2_2B_VisionEncoder,
            precision=Precision,
            torch=torch,
            candidate=candidate,
            cases=cases,
            geometry=geometry,
            device=device,
        )
        candidate_reports.append(
            _candidate_report(
                candidate=candidate,
                cases=cases,
                reference_outputs=reference_outputs,
                candidate_outputs=candidate_outputs,
            )
        )
        del candidate_outputs
        gc.collect()

    report = {
        "schema_version": 1,
        "tool": "scripts/compare_vision_checkpoints.py",
        "model": "nvidia/Cosmos-Reason2-2B",
        "execution": {
            "device": resolved_device_name,
            "torch_version": str(torch.__version__),
            "qai_hub_models_version": _distribution_version("qai-hub-models"),
            "reference_precision": (
                "Precision.float adapted encoder cast to torch.bfloat16"
            ),
            "candidate_precision_api": "Precision.w4a16",
            "tf32_enabled": False,
        },
        "geometry": geometry.as_dict(),
        "inputs": {
            "manifest_count": len(manifests),
            "case_count": len(cases),
            "unique_pixel_tensor_count": len(
                {case.pixel_sha256 for case in cases}
            ),
            "manifests": [_manifest_report(manifest) for manifest in manifests],
        },
        "reference": {
            "source_checkpoint": source_identity,
            "adapter": (
                "Cosmos_Reason2_2B_VisionEncoder with restored temporal "
                "patch-projection bias"
            ),
            "cases": [
                _reference_case_report(case, reference_outputs[case.case_id])
                for case in cases
            ],
        },
        "candidates": candidate_reports,
    }
    output_path = output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(output_path)
    return report


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help="Local Hugging Face Cosmos-Reason2-2B source snapshot",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help=(
            "Quantized vision checkpoint; repeat for multiple candidates. "
            "LABEL defaults to the directory name."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help=(
            "video_npu_manifest.json containing exact packed pixel tensors; "
            "repeat for multiple inputs"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the BF16 reference (default: auto)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        help="Optional manifest/checkpoint geometry cross-check",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        help="Optional manifest/checkpoint geometry cross-check",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = compare_vision_checkpoints(
            source_checkpoint=args.source_checkpoint,
            candidate_specs=args.candidate_checkpoint,
            manifest_paths=args.manifest,
            output_json=args.output_json,
            device_name=args.device,
            image_height=args.image_height,
            image_width=args.image_width,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        "Wrote "
        f"{args.output_json}: {report['inputs']['case_count']} cases, "
        f"{len(report['candidates'])} candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
