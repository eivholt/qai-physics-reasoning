#!/usr/bin/env python3
"""Score hosted P2/P3 outputs against an exact BF16 shard reference."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from scripts.run_aihub_first_token_chain import (
        TERMINAL_FAILURE_STATES,
        graph_option,
        sha256_array,
        sha256_file,
        validate_output_h5,
    )
    from scripts.run_aihub_p1_screen import (
        DEVICE_NAME,
        DEVICE_OS,
        load_h5_outputs,
        status_code,
        tensor_metrics,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_aihub_first_token_chain import (  # type: ignore[no-redef]
        TERMINAL_FAILURE_STATES,
        graph_option,
        sha256_array,
        sha256_file,
        validate_output_h5,
    )
    from run_aihub_p1_screen import (  # type: ignore[no-redef]
        DEVICE_NAME,
        DEVICE_OS,
        load_h5_outputs,
        status_code,
        tensor_metrics,
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def parse_candidates(values: list[str]) -> "OrderedDict[str, str]":
    candidates: "OrderedDict[str, str]" = OrderedDict()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be LABEL=JOB_ID: {value!r}")
        label, job_id = value.split("=", 1)
        if not label or not job_id:
            raise ValueError(f"Candidate must be LABEL=JOB_ID: {value!r}")
        if label in candidates:
            raise ValueError(f"Duplicate candidate label: {label}")
        if job_id in candidates.values():
            raise ValueError(f"Duplicate candidate job ID: {job_id}")
        candidates[label] = job_id
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required")
    return candidates


def load_reference(
    reference_path: Path,
    manifest_path: Path,
) -> tuple["OrderedDict[str, np.ndarray]", dict[str, Any]]:
    manifest = read_json(manifest_path)
    output = manifest.get("output")
    shard = manifest.get("shard")
    source = manifest.get("input")
    if not all(isinstance(value, dict) for value in (output, shard, source)):
        raise ValueError("Reference manifest is missing output/shard/input")
    if sha256_file(reference_path) != output.get("archive_sha256"):
        raise ValueError("Reference NPZ SHA-256 differs from its manifest")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = OrderedDict(
            (name, np.ascontiguousarray(archive[name]))
            for name in archive.files
        )
    tensor_records = output.get("tensors")
    if not isinstance(tensor_records, dict):
        raise ValueError("Reference manifest has no output tensor records")
    # The JSON manifest is serialized with sorted keys, while NPZ preserves
    # the graph-output insertion order.  Bind the same names and per-tensor
    # hashes here; candidate graph order is checked separately against NPZ.
    if set(reference) != set(tensor_records):
        raise ValueError("Reference NPZ tensor names differ from manifest")
    for name, value in reference.items():
        record = tensor_records[name]
        if (
            list(value.shape) != record.get("shape")
            or str(value.dtype) != record.get("dtype")
            or sha256_array(value) != record.get("sha256")
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Reference tensor contract differs: {name}")
    batch_labels = source.get("batch_labels")
    if not isinstance(batch_labels, list) or not batch_labels:
        raise ValueError("Reference manifest has no batch labels")
    if any(value.shape[0] != len(batch_labels) for value in reference.values()):
        raise ValueError("Reference tensors do not match batch label count")
    return reference, manifest


def aggregate_metrics(
    outputs: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    hidden_name: str,
    batch_labels: list[str],
) -> dict[str, Any]:
    if list(outputs) != list(reference):
        raise ValueError(
            "Candidate output tensor order differs from BF16 reference:\n"
            f"candidate={list(outputs)!r}\nreference={list(reference)!r}"
        )
    for name in reference:
        if outputs[name].shape != reference[name].shape:
            raise ValueError(
                f"Candidate/reference shape differs for {name}: "
                f"{outputs[name].shape} != {reference[name].shape}"
            )

    per_batch: list[dict[str, Any]] = []
    for batch_index, batch_label in enumerate(batch_labels):
        metrics = {
            name: tensor_metrics(
                outputs[name][batch_index],
                reference[name][batch_index],
            )
            for name in reference
        }
        kv = [
            value
            for name, value in metrics.items()
            if name.startswith("past_") and value.get("finite")
        ]
        hidden = metrics[hidden_name]
        per_batch.append(
            {
                "batch_index": batch_index,
                "batch_label": batch_label,
                "hidden": hidden,
                "kv_tensor_count": len(kv),
                "kv_mean_relative_rmse": float(
                    np.mean([value["relative_rmse"] for value in kv])
                ),
                "kv_mean_cosine": float(
                    np.mean([value["cosine"] for value in kv])
                ),
            }
        )

    overall = {
        name: tensor_metrics(outputs[name], reference[name])
        for name in reference
    }
    overall_kv = [
        value
        for name, value in overall.items()
        if name.startswith("past_") and value.get("finite")
    ]
    return {
        "hidden": overall[hidden_name],
        "kv_tensor_count": len(overall_kv),
        "kv_mean_relative_rmse": float(
            np.mean([value["relative_rmse"] for value in overall_kv])
        ),
        "kv_mean_cosine": float(
            np.mean([value["cosine"] for value in overall_kv])
        ),
        "per_batch": per_batch,
        "tensors": overall,
    }


def download_output_atomic(
    job: Any,
    model: Any,
    *,
    graph: str,
    batch_count: int,
    output_path: Path,
) -> None:
    if output_path.exists():
        validate_output_h5(output_path, model, graph, batch_count)
        return
    partial = output_path.with_suffix(".partial.h5")
    partial.unlink(missing_ok=True)
    try:
        job.download_output_data(str(partial))
        validate_output_h5(partial, model, graph, batch_count)
        os.replace(partial, output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference_path = args.reference_npz.expanduser().resolve()
    manifest_path = args.reference_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, manifest = load_reference(reference_path, manifest_path)
    candidates = parse_candidates(args.candidate)
    shard = manifest["shard"]
    source = manifest["input"]
    graph = str(shard["graph"])
    expected_options = graph_option(graph)
    batch_labels = list(source["batch_labels"])
    batch_count = len(batch_labels)

    import qai_hub as hub

    results: list[dict[str, Any]] = []
    for label, job_id in candidates.items():
        job = hub.get_job(job_id)
        code = status_code(job)
        if code in TERMINAL_FAILURE_STATES:
            raise RuntimeError(f"AI Hub job {job_id} ended as {code}")
        if code != "SUCCESS":
            raise RuntimeError(f"AI Hub job {job_id} is {code}, not SUCCESS")
        if (
            job.device.name != DEVICE_NAME
            or str(job.device.os) != DEVICE_OS
        ):
            raise ValueError(
                f"Job {job_id} device is {job.device.name}/{job.device.os}"
            )
        if job.options != expected_options:
            raise ValueError(f"Job {job_id} graph options differ")
        if job.inputs.dataset_id != source["dataset_id"]:
            raise ValueError(f"Job {job_id} uses a different dataset")
        output_path = output_dir / f"{label}_{job_id}.h5"
        download_output_atomic(
            job,
            job.model,
            graph=graph,
            batch_count=batch_count,
            output_path=output_path,
        )
        outputs = load_h5_outputs(output_path)
        metrics = aggregate_metrics(
            outputs,
            reference,
            hidden_name=str(shard["hidden_output"]),
            batch_labels=batch_labels,
        )
        results.append(
            {
                "candidate": label,
                "job_id": job.job_id,
                "model_id": job.model.model_id,
                "dataset_id": job.inputs.dataset_id,
                "graph": graph,
                "output_h5": output_path.name,
                "output_sha256": sha256_file(output_path),
                "metrics": metrics,
            }
        )

    ranking = sorted(
        (
            {
                "candidate": record["candidate"],
                "hidden_relative_rmse": record["metrics"]["hidden"][
                    "relative_rmse"
                ],
                "hidden_cosine": record["metrics"]["hidden"]["cosine"],
                "kv_mean_relative_rmse": record["metrics"][
                    "kv_mean_relative_rmse"
                ],
                "kv_mean_cosine": record["metrics"]["kv_mean_cosine"],
            }
            for record in results
        ),
        key=lambda value: value["hidden_relative_rmse"],
    )
    report = {
        "schema_version": 1,
        "purpose": (
            f"Hosted IQ9075 {str(shard['id']).upper()} output fidelity "
            "against an exact BF16 host reference"
        ),
        "reference": {
            "npz_sha256": sha256_file(reference_path),
            "manifest_sha256": sha256_file(manifest_path),
            "source_job_id": source["source_job_id"],
            "source_model_id": source["source_model_id"],
            "dataset_id": source["dataset_id"],
            "batch_fingerprint_sha256": source[
                "batch_fingerprint_sha256"
            ],
            "batch_labels": batch_labels,
        },
        "shard": shard,
        "results": results,
        "ranking": ranking,
        "limitations": [
            (
                "This is an intermediate-tensor comparison, not a generated "
                "answer or accuracy result."
            ),
            (
                "The BF16 reference consumes the exact quantized predecessor "
                "outputs in the reused hosted dataset; it isolates this shard "
                "rather than measuring the full upstream chain."
            ),
        ],
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": False,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
    }
    report_path = output_dir / f"{str(shard['id'])}_fidelity.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    print(json.dumps({"report": str(report_path), "ranking": ranking}, indent=2))


if __name__ == "__main__":
    main()
