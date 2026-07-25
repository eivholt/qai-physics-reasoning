#!/usr/bin/env python3
"""Submit and score an exact Cosmos video-prefill P1 screen on AI Hub.

The script deliberately treats a P1-only run as an intermediate numerical
screen, not as proof of end-to-end answer quality.  It can resume an existing
baseline job, reuse that job's uploaded dataset, submit selected model
variants, download their outputs, and compare them with a host reference.

Example:

    python scripts/run_aihub_p1_screen.py \
        /tmp/cosmos-p1-bf16-chunk0 \
        /tmp/cosmos-p1-screen \
        --resume baseline_w4_fp16=jp36071lp \
        --submit-missing --wait \
        --host-reference /tmp/cosmos-p1-host-bf16.npz
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.build_p1_video_prefill import (
        P1_GRAPH_NAME,
        P1_TARGETS,
        dataset_fingerprint,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_p1_video_prefill import (  # type: ignore[no-redef]
        P1_GRAPH_NAME,
        P1_TARGETS,
        dataset_fingerprint,
        sha256_file,
    )

DEVICE_NAME = "Dragonwing IQ-9075 EVK"
DEVICE_OS = "1.7"
GRAPH_OPTION = f"--qnn_options context_enable_graphs={P1_GRAPH_NAME}"
TERMINAL_STATES = {"SUCCESS", "FAILED", "CANCELED", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def derive_condition_label(input_manifest: Mapping[str, Any]) -> str:
    vision = input_manifest.get("source", {}).get("vision", {})
    if not isinstance(vision, dict):
        return "unknown_vision_source"
    output_kind = vision.get("vision_output_kind")
    source_label = str(vision.get("vision_source_label") or "").strip()
    normalized_source = re.sub(r"[^a-z0-9]+", "_", source_label.lower()).strip(
        "_"
    )
    if output_kind == "qai_hub_inference_h5":
        return normalized_source or "npu_vision_h5"
    if output_kind == "host_reference_npz":
        return normalized_source or "host_vision_reference"
    # Backward-compatible artifacts created before explicit source typing.
    outputs_path = str(vision.get("vision_outputs_path", "")).lower()
    if outputs_path.endswith((".h5", ".hdf5")):
        return normalized_source or "npu_vision_h5"
    if outputs_path.endswith(".npz"):
        return "host_vision_reference"
    return "unknown_vision_source"


def load_input_artifact(
    artifact_dir: Path,
) -> tuple["OrderedDict[str, np.ndarray]", dict[str, Any]]:
    artifact_dir = artifact_dir.expanduser().resolve()
    manifest_path = artifact_dir / "input_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("graph_name") != P1_GRAPH_NAME:
        raise ValueError(
            f"Expected graph {P1_GRAPH_NAME}, got {manifest.get('graph_name')}"
        )
    if int(manifest.get("chunk_index", -1)) != 0:
        raise ValueError("This screen is intentionally limited to P1 chunk 0")
    archive_record = manifest.get("tensor_archive")
    if not isinstance(archive_record, dict):
        raise ValueError("Input manifest has no tensor_archive object")
    archive_path = artifact_dir / str(archive_record["path"])
    expected_archive_sha = str(archive_record["sha256"])
    actual_archive_sha = sha256_file(archive_path)
    if actual_archive_sha != expected_archive_sha:
        raise ValueError(
            f"Input archive SHA mismatch: {actual_archive_sha} != "
            f"{expected_archive_sha}"
        )

    order = manifest.get("input_order")
    if not isinstance(order, list) or not all(
        isinstance(name, str) for name in order
    ):
        raise ValueError("Input manifest has no valid input_order")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != set(order):
            raise ValueError("Input archive names do not match input_order")
        inputs = OrderedDict(
            (name, np.ascontiguousarray(archive[name])) for name in order
        )
    actual_fingerprint = dataset_fingerprint(inputs)
    expected_fingerprint = manifest.get("dataset_fingerprint_sha256")
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Dataset fingerprint mismatch: {actual_fingerprint} != "
            f"{expected_fingerprint}"
        )
    for name, value in inputs.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(
            array
        ).all():
            raise ValueError(f"Input tensor {name} contains non-finite values")
        if name.startswith("past_") and np.any(array != 0):
            raise ValueError(
                f"Chunk-0 input {name} must be an all-zero KV cache"
            )
    return inputs, manifest


def input_contract(
    inputs: Mapping[str, np.ndarray],
) -> list[tuple[str, str, tuple[int, ...]]]:
    return [
        (name, str(np.asarray(value).dtype), tuple(np.asarray(value).shape))
        for name, value in inputs.items()
    ]


def model_contract(
    model: Any,
    graph_name: str = P1_GRAPH_NAME,
) -> list[tuple[str, str, tuple[int, ...]]]:
    if graph_name not in model.input_spec:
        raise ValueError(
            f"Model {model.model_id} has no graph named {graph_name}"
        )
    return [
        (spec.name, str(spec.dtype), tuple(spec.shape))
        for spec in model.input_spec[graph_name]
    ]


def validate_model(model: Any, inputs: Mapping[str, np.ndarray]) -> None:
    actual = input_contract(inputs)
    expected = model_contract(model)
    if actual != expected:
        lines = [
            f"Model {model.model_id} input contract differs from artifact:"
        ]
        for index in range(max(len(actual), len(expected))):
            got = actual[index] if index < len(actual) else None
            wanted = expected[index] if index < len(expected) else None
            if got != wanted:
                lines.append(f"  [{index}] artifact={got!r} model={wanted!r}")
        raise ValueError("\n".join(lines))


def validate_existing_screen(
    screen: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    condition_label: str,
) -> None:
    mismatches: list[str] = []
    for key, expected in (
        ("graph_name", P1_GRAPH_NAME),
        ("graph_options", GRAPH_OPTION),
    ):
        actual = screen.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")
    device = screen.get("device")
    if not isinstance(device, dict):
        mismatches.append("device record is missing")
    else:
        if device.get("name") != DEVICE_NAME:
            mismatches.append(
                f"device.name={device.get('name')!r}, expected {DEVICE_NAME!r}"
            )
        if str(device.get("os")) != DEVICE_OS:
            mismatches.append(
                f"device.os={device.get('os')!r}, expected {DEVICE_OS!r}"
            )
    screen_input = screen.get("input")
    actual_fingerprint = (
        screen_input.get("dataset_fingerprint_sha256")
        if isinstance(screen_input, dict)
        else None
    )
    expected_fingerprint = input_manifest.get("dataset_fingerprint_sha256")
    if actual_fingerprint != expected_fingerprint:
        mismatches.append(
            "dataset fingerprint "
            f"{actual_fingerprint!r}, expected {expected_fingerprint!r}"
        )
    existing_condition = screen.get("condition_label")
    if existing_condition is not None and existing_condition != condition_label:
        mismatches.append(
            f"condition_label={existing_condition!r}, "
            f"expected {condition_label!r}"
        )
    if mismatches:
        raise ValueError(
            "Refusing to reuse incompatible screen output directory:\n- "
            + "\n- ".join(mismatches)
        )


def validate_resumed_jobs(
    jobs: Mapping[str, Any],
    *,
    expected_targets: Mapping[str, str],
) -> str | None:
    dataset_ids: set[str] = set()
    errors: list[str] = []
    for label, job in jobs.items():
        expected_model_id = expected_targets[label]
        if job.model.model_id != expected_model_id:
            errors.append(
                f"{label}: model {job.model.model_id} != {expected_model_id}"
            )
        if job.options != GRAPH_OPTION:
            errors.append(
                f"{label}: options {job.options!r} != {GRAPH_OPTION!r}"
            )
        if job.device.name != DEVICE_NAME:
            errors.append(
                f"{label}: device {job.device.name!r} != {DEVICE_NAME!r}"
            )
        if str(job.device.os) != DEVICE_OS:
            errors.append(
                f"{label}: device OS {job.device.os!r} != {DEVICE_OS!r}"
            )
        dataset_id = getattr(getattr(job, "inputs", None), "dataset_id", None)
        if not dataset_id:
            errors.append(f"{label}: job has no input dataset")
        else:
            dataset_ids.add(str(dataset_id))
    if len(dataset_ids) > 1:
        errors.append(f"resumed jobs use different datasets: {sorted(dataset_ids)}")
    if errors:
        raise ValueError(
            "Resumed AI Hub job validation failed:\n- "
            + "\n- ".join(errors)
        )
    return next(iter(dataset_ids), None)


def validate_uploaded_dataset(
    entries: Mapping[str, Sequence[np.ndarray]],
    expected_inputs: Mapping[str, np.ndarray],
    expected_fingerprint: str,
) -> None:
    if list(entries) != list(expected_inputs):
        raise ValueError("Uploaded dataset tensor names/order differ from artifact")
    downloaded: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for name, expected in expected_inputs.items():
        batches = entries[name]
        if len(batches) != 1:
            raise ValueError(
                f"Uploaded dataset {name} has {len(batches)} samples; expected 1"
            )
        actual = np.ascontiguousarray(batches[0])
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(
                f"Uploaded dataset {name} contract "
                f"{actual.dtype}/{actual.shape} != "
                f"{expected.dtype}/{expected.shape}"
            )
        if not np.array_equal(actual, expected, equal_nan=True):
            raise ValueError(f"Uploaded dataset {name} bytes differ from artifact")
        downloaded[name] = actual
    if dataset_fingerprint(downloaded) != expected_fingerprint:
        raise ValueError("Uploaded dataset fingerprint differs from artifact")


def parse_assignments(values: Sequence[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=ID, got {value!r}")
        label, identifier = value.split("=", 1)
        if label not in P1_TARGETS:
            raise ValueError(
                f"Unknown target {label!r}; choose from {list(P1_TARGETS)}"
            )
        if not identifier:
            raise ValueError(f"Missing ID in {value!r}")
        assignments[label] = identifier
    return assignments


def status_code(job: Any) -> str:
    status = job.get_status()
    state = getattr(status, "state", None)
    name = getattr(state, "name", None)
    if name:
        return str(name)
    return str(state or status).upper()


def job_record(job: Any, label: str) -> dict[str, Any]:
    dataset = getattr(job, "inputs", None)
    return {
        "label": label,
        "job_id": job.job_id,
        "job_name": job.name,
        "job_url": job.url,
        "model_id": job.model.model_id,
        "dataset_id": (
            getattr(dataset, "dataset_id", None)
            if dataset is not None
            else None
        ),
        "device": str(job.device),
        "options": job.options,
        "status": status_code(job),
        "observed_at": utc_now(),
    }


def unpolled_job_record(job: Any, label: str) -> dict[str, Any]:
    dataset = getattr(job, "inputs", None)
    return {
        "label": label,
        "job_id": job.job_id,
        "job_name": job.name,
        "job_url": job.url,
        "model_id": job.model.model_id,
        "dataset_id": getattr(dataset, "dataset_id", None),
        "device": str(job.device),
        "options": job.options,
        "status": "SUBMITTED_UNPOLLED",
        "observed_at": utc_now(),
    }


def update_job_record(
    screen: dict[str, Any],
    job: Any,
    label: str,
    expected_name_prefix: str,
) -> dict[str, Any]:
    targets = screen.setdefault("targets", {})
    record = dict(targets.get(label, {}))
    record.update(job_record(job, label))
    expected_name = f"{expected_name_prefix}_{label}_r1"
    if job.name != expected_name:
        record["job_name_condition_mismatch"] = {
            "actual": job.name,
            "expected": expected_name,
            "note": (
                "The submitted job ID is preserved. Rely on condition_label "
                "and input vision provenance, not this legacy job name, when "
                "interpreting results."
            ),
        }
    else:
        record.pop("job_name_condition_mismatch", None)
    targets[label] = record
    return record


def load_h5_outputs(path: Path) -> "OrderedDict[str, np.ndarray]":
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to read AI Hub output H5") from exc

    records: list[tuple[int, str, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"AI Hub H5 has no /data group: {path}")
        for group in handle["data"].values():
            order = int(group.attrs["order"])
            raw_name = group.attrs["name"]
            name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            batch_count = int(group.attrs["batch_count"])
            if batch_count <= 0:
                raise ValueError(f"{path}:{name} has invalid batch_count")
            batches = [
                np.asarray(group[f"batch_{index}"])
                for index in range(batch_count)
            ]
            if any(
                np.issubdtype(value.dtype, np.floating)
                and not np.isfinite(value).all()
                for value in batches
            ):
                raise ValueError(f"{path}:{name} contains non-finite values")
            value = batches[0] if batch_count == 1 else np.stack(batches)
            records.append((order, name, value))
    orders = [order for order, _, _ in records]
    names = [name for _, name, _ in records]
    if len(set(orders)) != len(orders):
        raise ValueError(f"AI Hub H5 contains duplicate tensor order: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"AI Hub H5 contains duplicate tensor names: {path}")
    return OrderedDict(
        (name, np.ascontiguousarray(value))
        for _, name, value in sorted(records)
    )


def validate_p1_output_spec(model: Any) -> None:
    expected = [
        ("add_15805", "float32", (1, 128, 2048)),
        *[
            item
            for layer in range(7)
            for item in (
                (f"past_key_{layer}_out", "float32", (8, 1, 128, 128)),
                (f"past_value_{layer}_out", "float32", (8, 1, 128, 128)),
            )
        ],
    ]
    if P1_GRAPH_NAME not in model.output_spec:
        raise ValueError(f"Model {model.model_id} has no P1 output graph")
    actual = [
        (spec.name, str(spec.dtype), tuple(spec.shape))
        for spec in model.output_spec[P1_GRAPH_NAME]
    ]
    if actual != expected:
        raise ValueError(
            f"Model {model.model_id} P1 output contract differs:\n"
            f"actual={actual!r}\nexpected={expected!r}"
        )


def validate_p1_output_h5(path: Path, model: Any) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to validate AI Hub output") from exc
    validate_p1_output_spec(model)
    specs = model.output_spec[P1_GRAPH_NAME]
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"AI Hub H5 has no /data group: {path}")
        records: list[tuple[int, str, Any]] = []
        for group in handle["data"].values():
            raw_name = group.attrs["name"]
            name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            records.append((int(group.attrs["order"]), name, group))
        names = [name for _, name, _ in records]
        orders = [order for order, _, _ in records]
        if len(set(names)) != len(names) or len(set(orders)) != len(orders):
            raise ValueError(f"AI Hub H5 has duplicate names/order: {path}")
        records.sort(key=lambda item: item[0])
        if [name for _, name, _ in records] != [spec.name for spec in specs]:
            raise ValueError(f"AI Hub H5 output names/order differ: {path}")
        for (_, name, group), spec in zip(records, specs, strict=True):
            if int(group.attrs["batch_count"]) != 1:
                raise ValueError(f"{path}:{name} batch_count must be 1")
            if set(group.keys()) != {"batch_0"}:
                raise ValueError(f"{path}:{name} has unexpected batch datasets")
            dataset = group["batch_0"]
            if str(dataset.dtype) != str(spec.dtype):
                raise ValueError(f"{path}:{name} dtype differs from model spec")
            if tuple(dataset.shape) != tuple(spec.shape):
                raise ValueError(f"{path}:{name} shape differs from model spec")
            if np.issubdtype(dataset.dtype, np.floating):
                if dataset.chunks is None:
                    finite = np.isfinite(np.asarray(dataset)).all()
                else:
                    finite = all(
                        np.isfinite(np.asarray(dataset[selection])).all()
                        for selection in dataset.iter_chunks()
                    )
                if not finite:
                    raise ValueError(f"{path}:{name} contains non-finite values")


def load_reference(path: Path) -> "OrderedDict[str, np.ndarray]":
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return OrderedDict(
                (name, np.asarray(archive[name])) for name in archive.files
            )
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_h5_outputs(path)
    raise ValueError(f"Unsupported output archive: {path}")


def validate_reference_manifest(
    reference_path: Path,
    input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    reference_path = reference_path.expanduser().resolve()
    manifest_path = reference_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Host reference requires an adjacent provenance manifest: "
            f"{manifest_path}"
        )
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if manifest.get("graph_name") != P1_GRAPH_NAME:
        errors.append(
            f"graph {manifest.get('graph_name')!r} != {P1_GRAPH_NAME!r}"
        )
    reference_fingerprint = (
        manifest.get("input", {}).get("dataset_fingerprint_sha256")
        if isinstance(manifest.get("input"), dict)
        else None
    )
    input_fingerprint = input_manifest.get("dataset_fingerprint_sha256")
    if reference_fingerprint != input_fingerprint:
        errors.append(
            f"dataset fingerprint {reference_fingerprint!r} != "
            f"{input_fingerprint!r}"
        )
    reference_sha = sha256_file(reference_path)
    recorded_sha = (
        manifest.get("output", {}).get("archive_sha256")
        if isinstance(manifest.get("output"), dict)
        else None
    )
    if recorded_sha != reference_sha:
        errors.append(
            f"archive SHA {recorded_sha!r} != actual {reference_sha!r}"
        )
    if errors:
        raise ValueError(
            "Host-reference provenance validation failed:\n- "
            + "\n- ".join(errors)
        )
    return {
        "path": str(reference_path),
        "sha256": reference_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def align_shapes(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    while (
        candidate.ndim > reference.ndim
        and candidate.shape[0] == 1
        and candidate.shape[1:] == reference.shape
    ):
        candidate = candidate[0]
    while (
        reference.ndim > candidate.ndim
        and reference.shape[0] == 1
        and reference.shape[1:] == candidate.shape
    ):
        reference = reference[0]
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Output shape mismatch: {candidate.shape} != {reference.shape}"
        )
    return candidate, reference


def tensor_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    candidate, reference = align_shapes(candidate, reference)
    candidate64 = candidate.astype(np.float64, copy=False).reshape(-1)
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    finite = bool(
        np.isfinite(candidate64).all() and np.isfinite(reference64).all()
    )
    if not finite:
        return {
            "shape": list(candidate.shape),
            "finite": False,
        }
    difference = candidate64 - reference64
    mse = float(np.mean(np.square(difference)))
    rmse = math.sqrt(mse)
    reference_rms = math.sqrt(float(np.mean(np.square(reference64))))
    denominator = float(
        np.linalg.norm(candidate64) * np.linalg.norm(reference64)
    )
    cosine = (
        float(np.dot(candidate64, reference64) / denominator)
        if denominator
        else float("nan")
    )
    snr_db = (
        20.0 * math.log10(reference_rms / rmse)
        if reference_rms > 0.0 and rmse > 0.0
        else (float("inf") if rmse == 0.0 else float("-inf"))
    )
    return {
        "shape": list(candidate.shape),
        "finite": True,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "max_abs": float(np.max(np.abs(difference))),
        "reference_rms": reference_rms,
        "relative_rmse": (
            rmse / reference_rms if reference_rms else float("nan")
        ),
        "cosine": cosine,
        "snr_db": snr_db,
    }


def compare_outputs(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    reference_label: str,
    visual_mask: np.ndarray | None = None,
    valid_tokens: int | None = None,
) -> dict[str, Any]:
    missing = [name for name in reference if name not in candidate]
    extra = [name for name in candidate if name not in reference]
    if missing or extra:
        raise ValueError(
            f"Output-name mismatch: missing={missing}, extra={extra}"
        )
    tensors = {
        name: tensor_metrics(candidate[name], reference[name])
        for name in reference
    }
    hidden = tensors.get("add_15805")
    kv = [
        metrics
        for name, metrics in tensors.items()
        if name.startswith("past_") and metrics.get("finite")
    ]
    aggregate: dict[str, Any] = {
        "hidden": hidden,
        "kv_tensor_count": len(kv),
    }
    if (
        hidden is not None
        and visual_mask is not None
        and "add_15805" in candidate
    ):
        hidden_candidate, hidden_reference = align_shapes(
            candidate["add_15805"],
            reference["add_15805"],
        )
        mask = np.asarray(visual_mask, dtype=np.bool_)
        if mask.shape != hidden_candidate.shape[:2]:
            raise ValueError(
                f"Visual-mask shape {mask.shape} does not match hidden "
                f"tokens {hidden_candidate.shape[:2]}"
            )
        if valid_tokens is None:
            valid_tokens = mask.shape[-1]
        if not 0 <= valid_tokens <= mask.shape[-1]:
            raise ValueError("valid_tokens is outside hidden sequence")
        valid_mask = np.zeros_like(mask)
        valid_mask[:, :valid_tokens] = True
        regions = {
            "valid": valid_mask,
            "visual": valid_mask & mask,
            "text": valid_mask & ~mask,
            "padding": ~valid_mask,
        }
        aggregate["hidden_regions"] = {
            name: {
                "token_count": int(region.sum()),
                **tensor_metrics(
                    hidden_candidate[region],
                    hidden_reference[region],
                ),
            }
            for name, region in regions.items()
            if region.any()
        }
    if kv:
        aggregate["kv_mean_relative_rmse"] = float(
            np.mean([metrics["relative_rmse"] for metrics in kv])
        )
        aggregate["kv_mean_cosine"] = float(
            np.mean([metrics["cosine"] for metrics in kv])
        )
    return {
        "reference_type": reference_label,
        "aggregate": aggregate,
        "tensors": tensors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_artifact_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--resume",
        action="append",
        default=[],
        metavar="LABEL=JOB_ID",
        help="Existing job to include and use as a dataset source",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=list(P1_TARGETS),
        help="Limit processing to this target (repeatable)",
    )
    parser.add_argument(
        "--submit-missing",
        action="store_true",
        help="Submit selected targets that have no --resume job",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll all jobs to terminal state and download successful outputs",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--host-reference",
        type=Path,
        help="High-precision host P1 output NPZ used for numeric ranking",
    )
    parser.add_argument(
        "--reference-label",
        default="host_high_precision",
        help="Provenance label written beside all reference comparisons",
    )
    parser.add_argument(
        "--name-prefix",
        help=(
            "AI Hub job-name prefix. By default it is derived from the "
            "manifest's actual vision source, never from an assumed dtype."
        ),
    )
    parser.add_argument(
        "--condition-label",
        help="Explicit source-condition label stored in the screen manifest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    inputs, input_manifest = load_input_artifact(args.input_artifact_dir)
    condition_label = (
        args.condition_label or derive_condition_label(input_manifest)
    )
    name_prefix = (
        args.name_prefix
        or f"cosmos_p1_screen_barrier_{condition_label}_chunk0"
    )
    selected = args.target or list(P1_TARGETS)
    resumed_ids = parse_assignments(args.resume)
    unexpected = set(resumed_ids) - set(selected)
    if unexpected:
        raise ValueError(f"Resumed targets not selected: {sorted(unexpected)}")

    import qai_hub as hub

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "screen_manifest.json"
    screen: dict[str, Any] = (
        read_json(record_path)
        if record_path.is_file()
        else {
            "schema_version": 1,
            "purpose": (
                "P1 chunk-0 numerical screen; not end-to-end answer parity"
            ),
            "created_at": utc_now(),
            "graph_name": P1_GRAPH_NAME,
            "graph_options": GRAPH_OPTION,
            "condition_label": condition_label,
            "expected_job_name_prefix": name_prefix,
            "device": {"name": DEVICE_NAME, "os": DEVICE_OS},
            "input": {
                "artifact_dir": str(args.input_artifact_dir.resolve()),
                "dataset_fingerprint_sha256": input_manifest[
                    "dataset_fingerprint_sha256"
                ],
                "vision": input_manifest.get("source", {}).get("vision"),
            },
            "targets": {},
            "publication": {
                "classification": "local_only_unsanitized",
                "contains_local_paths": True,
                "sanitizer": "scripts/sanitize_local_manifest.py",
            },
            "limitations": [
                (
                    "P1 outputs are intermediate hidden/KV tensors and do not "
                    "establish generated-answer correctness."
                ),
                (
                    "Only candidates compared with the host reference may be "
                    "ranked for numerical fidelity."
                ),
            ],
        }
    )
    if record_path.is_file():
        validate_existing_screen(screen, input_manifest, condition_label)
    screen["condition_label"] = condition_label
    screen["expected_job_name_prefix"] = name_prefix
    screen["input"]["vision"] = input_manifest.get("source", {}).get("vision")
    screen.setdefault(
        "publication",
        {
            "classification": "local_only_unsanitized",
            "contains_local_paths": True,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
    )

    jobs: dict[str, Any] = {}
    for label, job_id in resumed_ids.items():
        existing_id = (
            screen.get("targets", {}).get(label, {}).get("job_id")
        )
        if existing_id and existing_id != job_id:
            raise ValueError(
                f"Refusing to replace {label} job {existing_id} with {job_id}"
            )
        jobs[label] = hub.get_job(job_id)
    for label in selected:
        existing_id = (
            screen.get("targets", {}).get(label, {}).get("job_id")
        )
        if label not in jobs and existing_id:
            jobs[label] = hub.get_job(existing_id)

    dataset = None
    for job in jobs.values():
        if getattr(job, "inputs", None) is not None:
            dataset = job.inputs
            break
    if jobs and dataset is None:
        raise RuntimeError("Existing AI Hub job has no reusable input dataset")

    models: dict[str, Any] = {}
    for label in selected:
        model = hub.get_model(P1_TARGETS[label])
        validate_model(model, inputs)
        validate_p1_output_spec(model)
        models[label] = model
    validate_resumed_jobs(jobs, expected_targets=P1_TARGETS)
    if dataset is not None:
        downloaded_entries = dataset.download()
        if not isinstance(downloaded_entries, dict):
            raise ValueError("AI Hub dataset did not download as tensor entries")
        validate_uploaded_dataset(
            downloaded_entries,
            inputs,
            str(input_manifest["dataset_fingerprint_sha256"]),
        )

    if args.submit_missing:
        if dataset is None:
            dataset_entries = {
                name: [value] for name, value in inputs.items()
            }
            dataset = hub.upload_dataset(dataset_entries)
        device = hub.Device(DEVICE_NAME, os=DEVICE_OS)
        for label in selected:
            if label in jobs:
                continue
            job = hub.submit_inference_job(
                models[label],
                device=device,
                inputs=dataset,
                options=GRAPH_OPTION,
                # Keep source condition visible in Workbench job lists.
                name=f"{name_prefix}_{label}_r1",
            )
            jobs[label] = job
            screen["targets"][label] = unpolled_job_record(job, label)
            screen["updated_at"] = utc_now()
            write_json(record_path, screen)
            print(
                json.dumps(
                    {
                        "event": "submitted",
                        "label": label,
                        "job_id": job.job_id,
                        "dataset_id": dataset.dataset_id,
                    }
                ),
                flush=True,
            )

    for label, job in jobs.items():
        update_job_record(screen, job, label, name_prefix)
    screen["updated_at"] = utc_now()
    write_json(record_path, screen)

    if args.wait and jobs:
        last_status: dict[str, str] = {}
        while True:
            terminal = True
            for label, job in jobs.items():
                code = status_code(job)
                if code != last_status.get(label):
                    print(
                        json.dumps(
                            {
                                "event": "status",
                                "label": label,
                                "job_id": job.job_id,
                                "status": code,
                            }
                        ),
                        flush=True,
                    )
                    last_status[label] = code
                update_job_record(screen, job, label, name_prefix)
                if code not in TERMINAL_STATES:
                    terminal = False
            screen["updated_at"] = utc_now()
            write_json(record_path, screen)
            if terminal:
                break
            time.sleep(args.poll_seconds)

    reference_provenance = (
        validate_reference_manifest(args.host_reference, input_manifest)
        if args.host_reference
        else None
    )
    reference = load_reference(args.host_reference) if args.host_reference else None
    if args.host_reference:
        reference_sha = reference_provenance["sha256"]
        existing_reference = screen.get("reference")
        if (
            isinstance(existing_reference, dict)
            and existing_reference.get("sha256") != reference_sha
        ):
            for target_record in screen["targets"].values():
                if isinstance(target_record, dict):
                    target_record.pop("host_comparison", None)
            screen.pop("ranking", None)
        screen["reference"] = {
            **reference_provenance,
            "label": args.reference_label,
        }
    for label, job in jobs.items():
        code = status_code(job)
        update_job_record(screen, job, label, name_prefix)
        if code != "SUCCESS":
            continue
        output_path = output_dir / f"{label}_{job.job_id}.h5"
        existing_output = screen["targets"][label].get("output")
        if output_path.is_file():
            if not isinstance(existing_output, dict) or not existing_output.get(
                "sha256"
            ):
                raise ValueError(
                    f"Refusing to adopt unrecorded output file: {output_path}"
                )
            if sha256_file(output_path) != existing_output["sha256"]:
                raise ValueError(
                    f"Refusing to reuse modified output file: {output_path}"
                )
            validate_p1_output_h5(output_path, models[label])
        if not output_path.is_file():
            partial_path = output_path.with_suffix(".partial.h5")
            if partial_path.exists():
                if not partial_path.is_file():
                    raise ValueError(
                        f"Partial output path is not a file: {partial_path}"
                    )
                partial_path.unlink()
            job.download_output_data(str(partial_path))
            validate_p1_output_h5(partial_path, models[label])
            partial_path.replace(output_path)
        validate_p1_output_h5(output_path, models[label])
        target_record = screen["targets"][label]
        target_record["output"] = {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
        }
        if reference is not None:
            comparison = compare_outputs(
                load_h5_outputs(output_path),
                reference,
                reference_label=args.reference_label,
                visual_mask=inputs["visual_pos_masks"],
                valid_tokens=int(input_manifest["valid_tokens"]),
            )
            comparison_path = (
                output_dir / f"{label}_{job.job_id}_vs_host.json"
            )
            write_json(comparison_path, comparison)
            target_record["host_comparison"] = {
                "path": comparison_path.name,
                "reference_path": str(args.host_reference.resolve()),
                "reference_sha256": reference_provenance["sha256"],
                "aggregate": comparison["aggregate"],
            }

    rankable: list[dict[str, Any]] = []
    for label, target_record in screen["targets"].items():
        host_comparison = target_record.get("host_comparison")
        if not isinstance(host_comparison, dict):
            continue
        current_reference = screen.get("reference")
        if (
            not isinstance(current_reference, dict)
            or host_comparison.get("reference_sha256")
            != current_reference.get("sha256")
        ):
            continue
        aggregate = host_comparison.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
        hidden = aggregate.get("hidden")
        if not isinstance(hidden, dict) or not hidden.get("finite"):
            continue
        rankable.append(
            {
                "label": label,
                "job_id": target_record.get("job_id"),
                "model_id": target_record.get("model_id"),
                "hidden_relative_rmse": hidden.get("relative_rmse"),
                "hidden_cosine": hidden.get("cosine"),
                "kv_mean_relative_rmse": aggregate.get(
                    "kv_mean_relative_rmse"
                ),
                "kv_mean_cosine": aggregate.get("kv_mean_cosine"),
            }
        )
    if rankable:
        screen["ranking"] = {
            "basis": (
                "Independent rankings against the same high-precision P1 "
                "reference; lower relative RMSE is better. No composite score "
                "or end-to-end answer-quality claim is implied."
            ),
            "by_hidden_relative_rmse": sorted(
                rankable,
                key=lambda record: record["hidden_relative_rmse"],
            ),
            "by_kv_mean_relative_rmse": sorted(
                rankable,
                key=lambda record: record["kv_mean_relative_rmse"],
            ),
        }

    screen["updated_at"] = utc_now()
    write_json(record_path, screen)
    print(
        json.dumps(
            {
                "screen_manifest": str(record_path),
                "dataset_id": (
                    dataset.dataset_id if dataset is not None else None
                ),
                "jobs": {
                    label: {
                        "job_id": job.job_id,
                        "status": status_code(job),
                    }
                    for label, job in jobs.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
