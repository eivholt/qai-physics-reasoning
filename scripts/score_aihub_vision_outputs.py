#!/usr/bin/env python3
"""Score AI Hub vision-output H5 files against a saved BF16 reference.

The reference consists of an NPZ archive and its JSON index manifest.  The
index records the exact tensor shapes, canonical float32 hashes, and the case
mapped to every batch index.  Each AI Hub H5 output is supplied as
``LABEL=PATH``.  Labels are the only source identifiers included in the
report; local paths are deliberately omitted.

AI Hub H5 groups are matched exclusively through their ``name`` attribute.
Numeric group keys and the optional ``order`` attribute are never used to
identify a tensor.

Example:

    python scripts/score_aihub_vision_outputs.py \
      --reference-npz bf16_reference_outputs.npz \
      --reference-index bf16_reference_outputs.json \
      --candidate-h5 native=output_native.h5 \
      --candidate-h5 paired=output_paired.h5 \
      --output-json vision_fidelity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OUTPUT_NAMES = (
    "image_features",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
)
OUTPUT_NAME_SET = frozenset(OUTPUT_NAMES)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


@dataclass(frozen=True)
class ReferenceOutputs:
    """Validated BF16 reference arrays and their case mapping."""

    tensors: Mapping[str, np.ndarray]
    cases: tuple[dict[str, Any], ...]
    geometry: dict[str, Any]
    npz_sha256: str
    index_sha256: str

    @property
    def sample_count(self) -> int:
        return len(self.cases)


@dataclass(frozen=True)
class CandidateSpec:
    """One user-labelled AI Hub H5 output."""

    label: str
    path: Path


@dataclass(frozen=True)
class CandidateOutputs:
    """Validated candidate arrays without host-path provenance."""

    label: str
    tensors: Mapping[str, np.ndarray]
    h5_sha256: str


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_float32_sha256(value: Any) -> str:
    """Hash an array after canonical little-endian C-order float32 conversion."""

    array = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file")
    return resolved


def _strict_nonnegative_int(value: Any, description: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{description} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return parsed


def _strict_positive_int(value: Any, description: str) -> int:
    parsed = _strict_nonnegative_int(value, description)
    if parsed == 0:
        raise ValueError(f"{description} must be a positive integer")
    return parsed


def _shape(
    value: Any,
    description: str,
    *,
    dimensions: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an array")
    if dimensions is not None and len(value) != dimensions:
        raise ValueError(
            f"{description} must contain exactly {dimensions} dimensions"
        )
    if not value:
        raise ValueError(f"{description} must not be empty")
    return tuple(
        _strict_positive_int(item, f"{description}[{index}]")
        for index, item in enumerate(value)
    )


def _lowercase_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _validate_geometry(
    raw_geometry: Any,
    output_shape: tuple[int, ...],
) -> dict[str, Any]:
    if not isinstance(raw_geometry, dict):
        raise ValueError("Reference index geometry must be an object")
    image_height = _strict_positive_int(
        raw_geometry.get("image_height"),
        "geometry.image_height",
    )
    image_width = _strict_positive_int(
        raw_geometry.get("image_width"),
        "geometry.image_width",
    )
    grid_thw = _shape(
        raw_geometry.get("grid_thw"),
        "geometry.grid_thw",
        dimensions=3,
    )
    recorded_output_shape = _shape(
        raw_geometry.get("output_shape"),
        "geometry.output_shape",
        dimensions=2,
    )
    if recorded_output_shape != output_shape[1:]:
        raise ValueError(
            "Reference geometry output_shape does not match tensor shapes: "
            f"{recorded_output_shape} != {output_shape[1:]}"
        )
    return {
        "grid_thw": list(grid_thw),
        "image_height": image_height,
        "image_width": image_width,
        "output_shape": list(recorded_output_shape),
    }


def _validate_cases(raw_cases: Any, sample_count: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_cases, list):
        raise ValueError("Reference index cases must be an array")
    if len(raw_cases) != sample_count:
        raise ValueError(
            "Reference case count does not match tensor sample count: "
            f"{len(raw_cases)} != {sample_count}"
        )

    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_source_pairs: set[tuple[str, int]] = set()
    for expected_index, raw_case in enumerate(raw_cases):
        description = f"cases[{expected_index}]"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{description} must be an object")
        index = _strict_nonnegative_int(
            raw_case.get("index"),
            f"{description}.index",
        )
        if index != expected_index:
            raise ValueError(
                "Reference cases must be ordered one-to-one with NPZ rows: "
                f"{description}.index is {index}, expected {expected_index}"
            )
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None:
            raise ValueError(f"{description}.case_id is not a safe case ID")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate reference case_id: {case_id}")
        seen_case_ids.add(case_id)

        manifest_sha256 = _lowercase_sha256(
            raw_case.get("manifest_sha256"),
            f"{description}.manifest_sha256",
        )
        pair_index = _strict_nonnegative_int(
            raw_case.get("pair_index"),
            f"{description}.pair_index",
        )
        source_pair = (manifest_sha256, pair_index)
        if source_pair in seen_source_pairs:
            raise ValueError(
                "Duplicate reference manifest/pair mapping at "
                f"{description}"
            )
        seen_source_pairs.add(source_pair)
        pixel_sha256 = _lowercase_sha256(
            raw_case.get("pixel_sha256"),
            f"{description}.pixel_sha256",
        )
        cases.append(
            {
                "case_id": case_id,
                "index": index,
                "manifest_sha256": manifest_sha256,
                "pair_index": pair_index,
                "pixel_sha256": pixel_sha256,
            }
        )
    return tuple(cases)


def _validate_finite_float32(
    value: Any,
    description: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{description} must contain real numeric values")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{description} contains a non-finite value")
    canonical = np.ascontiguousarray(array, dtype=np.float32)
    if not bool(np.isfinite(canonical).all()):
        raise ValueError(
            f"{description} becomes non-finite when converted to float32"
        )
    canonical.setflags(write=False)
    return canonical


def load_reference_outputs(
    npz_path: Path,
    index_path: Path,
) -> ReferenceOutputs:
    """Load and cross-check a BF16 NPZ and its JSON index manifest."""

    resolved_npz = _require_file(npz_path, "Reference NPZ")
    resolved_index = _require_file(index_path, "Reference index JSON")
    index = _read_json_object(resolved_index, "reference index JSON")
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported reference index schema_version")

    raw_outputs = index.get("outputs")
    if not isinstance(raw_outputs, dict):
        raise ValueError("Reference index outputs must be an object")
    recorded_names = set(raw_outputs)
    if recorded_names != OUTPUT_NAME_SET:
        raise ValueError(
            "Reference index must describe exactly these outputs: "
            + ", ".join(OUTPUT_NAMES)
        )

    try:
        archive_context = np.load(resolved_npz, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read reference NPZ") from exc
    tensors: dict[str, np.ndarray] = {}
    with archive_context as archive:
        archive_names = set(archive.files)
        if archive_names != OUTPUT_NAME_SET:
            raise ValueError(
                "Reference NPZ must contain exactly these outputs: "
                + ", ".join(OUTPUT_NAMES)
            )
        expected_shape: tuple[int, ...] | None = None
        for name in OUTPUT_NAMES:
            record = raw_outputs[name]
            if not isinstance(record, dict):
                raise ValueError(f"Reference output record {name} must be an object")
            recorded_shape = _shape(
                record.get("shape"),
                f"outputs.{name}.shape",
                dimensions=3,
            )
            if expected_shape is None:
                expected_shape = recorded_shape
            elif recorded_shape != expected_shape:
                raise ValueError(
                    "All reference outputs must have the same shape: "
                    f"{recorded_shape} != {expected_shape}"
                )
            expected_hash = _lowercase_sha256(
                record.get("canonical_float32_sha256"),
                f"outputs.{name}.canonical_float32_sha256",
            )
            tensor = _validate_finite_float32(
                archive[name],
                f"Reference tensor {name}",
            )
            if tensor.shape != recorded_shape:
                raise ValueError(
                    f"Reference tensor {name} has shape {tensor.shape}; "
                    f"index records {recorded_shape}"
                )
            actual_hash = canonical_float32_sha256(tensor)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Reference tensor {name} canonical float32 SHA-256 "
                    "does not match the index"
                )
            tensors[name] = tensor

    if expected_shape is None:  # pragma: no cover - OUTPUT_NAMES is nonempty
        raise AssertionError("No reference output shape was validated")
    cases = _validate_cases(index.get("cases"), expected_shape[0])
    geometry = _validate_geometry(index.get("geometry"), expected_shape)
    return ReferenceOutputs(
        tensors=tensors,
        cases=cases,
        geometry=geometry,
        npz_sha256=sha256_file(resolved_npz),
        index_sha256=sha256_file(resolved_index),
    )


def parse_candidate_specs(specs: Sequence[str]) -> tuple[CandidateSpec, ...]:
    """Parse repeated ``LABEL=H5`` arguments and require unique safe labels."""

    if not specs:
        raise ValueError("At least one --candidate-h5 LABEL=H5 is required")
    candidates: list[CandidateSpec] = []
    seen_labels: set[str] = set()
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not raw_path:
            raise ValueError(
                "Each --candidate-h5 must use the form LABEL=H5"
            )
        if LABEL_RE.fullmatch(label) is None:
            raise ValueError(
                f"Invalid candidate label {label!r}; use 1-64 safe characters"
            )
        if label in seen_labels:
            raise ValueError(f"Duplicate candidate label: {label}")
        seen_labels.add(label)
        path = _require_file(Path(raw_path), f"Candidate H5 for {label}")
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Candidate {label} is not an H5 file")
        candidates.append(CandidateSpec(label=label, path=path))
    return tuple(sorted(candidates, key=lambda candidate: candidate.label))


def _decode_h5_name(value: Any, description: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{description} is not valid UTF-8") from exc
    if isinstance(value, str):
        return value
    raise ValueError(f"{description} must be a string")


def load_h5_candidate(
    spec: CandidateSpec,
    reference: ReferenceOutputs,
) -> CandidateOutputs:
    """Load one AI Hub H5, mapping groups only by their ``name`` attribute."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to read AI Hub H5 outputs") from exc

    tensors: dict[str, np.ndarray] = {}
    with h5py.File(spec.path, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ValueError(f"Candidate {spec.label} has no /data group")
        data_group = handle["data"]
        if len(data_group) != len(OUTPUT_NAMES):
            raise ValueError(
                f"Candidate {spec.label} must have exactly "
                f"{len(OUTPUT_NAMES)} groups in /data"
            )
        named_groups: dict[str, Any] = {}
        for group_key in data_group:
            group = data_group[group_key]
            if not isinstance(group, h5py.Group):
                raise ValueError(
                    f"Candidate {spec.label} /data contains a non-group item"
                )
            if "name" not in group.attrs:
                raise ValueError(
                    f"Candidate {spec.label} H5 group is missing attrs['name']"
                )
            name = _decode_h5_name(
                group.attrs["name"],
                f"Candidate {spec.label} H5 group name",
            )
            if name in named_groups:
                raise ValueError(
                    f"Candidate {spec.label} has duplicate H5 tensor name {name}"
                )
            named_groups[name] = group

        actual_names = set(named_groups)
        if actual_names != OUTPUT_NAME_SET:
            missing = sorted(OUTPUT_NAME_SET - actual_names)
            unexpected = sorted(actual_names - OUTPUT_NAME_SET)
            raise ValueError(
                f"Candidate {spec.label} H5 tensor names are invalid; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for name in OUTPUT_NAMES:
            group = named_groups[name]
            if "batch_count" not in group.attrs:
                raise ValueError(
                    f"Candidate {spec.label} {name} is missing batch_count"
                )
            batch_count = _strict_positive_int(
                group.attrs["batch_count"],
                f"Candidate {spec.label} {name} batch_count",
            )
            if batch_count != reference.sample_count:
                raise ValueError(
                    f"Candidate {spec.label} {name} batch_count is "
                    f"{batch_count}; expected {reference.sample_count}"
                )
            expected_batch_names = {
                f"batch_{index}" for index in range(batch_count)
            }
            actual_batch_names = set(group)
            if actual_batch_names != expected_batch_names:
                raise ValueError(
                    f"Candidate {spec.label} {name} batch datasets do not "
                    "match batch_count"
                )
            expected_sample_shape = reference.tensors[name].shape[1:]
            batches: list[np.ndarray] = []
            for index in range(batch_count):
                dataset = group[f"batch_{index}"]
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(
                        f"Candidate {spec.label} {name}/batch_{index} "
                        "is not a dataset"
                    )
                batch = _validate_finite_float32(
                    dataset[...],
                    f"Candidate {spec.label} {name}/batch_{index}",
                )
                if batch.shape != expected_sample_shape:
                    raise ValueError(
                        f"Candidate {spec.label} {name}/batch_{index} has "
                        f"shape {batch.shape}; expected {expected_sample_shape}"
                    )
                batches.append(batch)
            tensor = np.ascontiguousarray(np.stack(batches), dtype=np.float32)
            tensor.setflags(write=False)
            tensors[name] = tensor

    return CandidateOutputs(
        label=spec.label,
        tensors=tensors,
        h5_sha256=sha256_file(spec.path),
    )


def tensor_metrics(
    reference: Any,
    candidate: Any,
) -> dict[str, float | None]:
    """Compute the four requested fidelity metrics over equal-shaped arrays."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Metric tensor shapes differ: "
            f"{reference_array.shape} != {candidate_array.shape}"
        )
    if reference_array.size == 0:
        raise ValueError("Metric tensors must not be empty")
    if not bool(np.isfinite(reference_array).all()):
        raise ValueError("Metric reference contains a non-finite value")
    if not bool(np.isfinite(candidate_array).all()):
        raise ValueError("Metric candidate contains a non-finite value")

    reference_flat = reference_array.astype(np.float64, copy=False).reshape(-1)
    candidate_flat = candidate_array.astype(np.float64, copy=False).reshape(-1)
    difference = candidate_flat - reference_flat
    reference_squared_norm = float(
        np.sum(reference_flat * reference_flat, dtype=np.float64)
    )
    candidate_squared_norm = float(
        np.sum(candidate_flat * candidate_flat, dtype=np.float64)
    )
    difference_squared_norm = float(
        np.sum(difference * difference, dtype=np.float64)
    )
    reference_norm = math.sqrt(max(reference_squared_norm, 0.0))
    candidate_norm = math.sqrt(max(candidate_squared_norm, 0.0))

    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine: float | None = 1.0
        relative_l2: float | None = 0.0
    elif reference_norm == 0.0:
        cosine = None
        relative_l2 = None
    elif candidate_norm == 0.0:
        cosine = None
        relative_l2 = math.sqrt(difference_squared_norm) / reference_norm
    else:
        dot = float(
            np.sum(reference_flat * candidate_flat, dtype=np.float64)
        )
        cosine = dot / (reference_norm * candidate_norm)
        cosine = min(1.0, max(-1.0, cosine))
        relative_l2 = math.sqrt(difference_squared_norm) / reference_norm

    metrics: dict[str, float | None] = {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "mean_absolute_error": float(
            np.mean(np.abs(difference), dtype=np.float64)
        ),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }
    if any(
        value is not None and not math.isfinite(value)
        for value in metrics.values()
    ):
        raise ValueError("Metric calculation produced a non-finite value")
    return metrics


def _tensor_record(
    value: np.ndarray,
    metrics: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "canonical_float32_sha256": canonical_float32_sha256(value),
        "shape": list(value.shape),
    }
    if metrics is not None:
        record["metrics"] = metrics
    return record


def build_report(
    reference: ReferenceOutputs,
    candidates: Sequence[CandidateOutputs],
) -> dict[str, Any]:
    """Build a deterministic report containing no host filesystem paths."""

    if not candidates:
        raise ValueError("At least one candidate output is required")
    by_label: dict[str, CandidateOutputs] = {}
    for candidate in candidates:
        if candidate.label in by_label:
            raise ValueError(f"Duplicate candidate label: {candidate.label}")
        by_label[candidate.label] = candidate

    candidate_reports: dict[str, Any] = {}
    for label in sorted(by_label):
        candidate = by_label[label]
        candidate_reports[label] = {
            "h5_sha256": candidate.h5_sha256,
            "sample_count": reference.sample_count,
            "tensors": {
                name: _tensor_record(
                    candidate.tensors[name],
                    tensor_metrics(
                        reference.tensors[name],
                        candidate.tensors[name],
                    ),
                )
                for name in OUTPUT_NAMES
            },
        }

    pairwise: list[dict[str, Any]] = []
    labels = sorted(by_label)
    for reference_index, reference_label in enumerate(labels):
        for candidate_label in labels[reference_index + 1 :]:
            first = by_label[reference_label]
            second = by_label[candidate_label]
            pairwise.append(
                {
                    "candidate_label": candidate_label,
                    "reference_label": reference_label,
                    "tensors": {
                        name: {
                            "metrics": tensor_metrics(
                                first.tensors[name],
                                second.tensors[name],
                            )
                        }
                        for name in OUTPUT_NAMES
                    },
                }
            )

    return {
        "candidates": candidate_reports,
        "pairwise_candidate_comparisons": pairwise,
        "reference": {
            "cases": list(reference.cases),
            "geometry": reference.geometry,
            "index_sha256": reference.index_sha256,
            "kind": "adapted_cosmos_reason2_2b_bf16_vision_npz",
            "npz_sha256": reference.npz_sha256,
            "sample_count": reference.sample_count,
            "tensors": {
                name: _tensor_record(reference.tensors[name])
                for name in OUTPUT_NAMES
            },
        },
        "schema_version": 1,
        "tool": "scripts/score_aihub_vision_outputs.py",
    }


def score_outputs(
    *,
    reference_npz: Path,
    reference_index: Path,
    candidate_specs: Sequence[str],
) -> dict[str, Any]:
    """Load, validate, and compare all requested output files."""

    reference = load_reference_outputs(reference_npz, reference_index)
    specs = parse_candidate_specs(candidate_specs)
    candidates = tuple(
        load_h5_candidate(spec, reference)
        for spec in specs
    )
    return build_report(reference, candidates)


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write a deterministic strict-JSON report."""

    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-npz",
        type=Path,
        required=True,
        help="Saved BF16 vision-output NPZ",
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        required=True,
        help="JSON index and hash manifest for the BF16 NPZ",
    )
    parser.add_argument(
        "--candidate-h5",
        action="append",
        required=True,
        metavar="LABEL=H5",
        help=(
            "Labelled AI Hub inference output; repeat for every candidate"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Destination for the sanitized deterministic report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = score_outputs(
            reference_npz=args.reference_npz,
            reference_index=args.reference_index,
            candidate_specs=args.candidate_h5,
        )
        write_report(report, args.output_json)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Wrote {args.output_json}: "
        f"{report['reference']['sample_count']} samples, "
        f"{len(report['candidates'])} candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
