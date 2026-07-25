#!/usr/bin/env python3
"""Advance a resumable three-chunk Cosmos P1->P4 AI Hub chain.

P1 is run separately for each candidate model.  P2, P3, and P4 are fixed and
batch the candidate states as separate samples.  The script preserves each
shard's layer-local 384-token KV cache across all three AR128 chunks and scores
the final P4 logits at the last valid row recorded by the frozen manifest.

Each invocation collects completed outputs, submits every newly-ready job, and
returns without polling.  Re-run the same command until ``complete`` is true.
All paid job IDs and dataset IDs are persisted immediately in ``chain.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.build_p1_video_prefill import (
        CONTEXT_LENGTH,
        HEAD_DIM,
        NUM_KV_HEADS,
        P1_GRAPH_NAME,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        dataset_fingerprint,
        load_bundle_contract,
        sha256_array,
        sha256_file,
        tokenize_video_prompt,
    )
    from scripts.run_aihub_p1_screen import (
        DEVICE_NAME,
        DEVICE_OS,
        derive_condition_label,
        load_input_artifact,
        read_json,
        status_code,
        validate_model,
        validate_resumed_jobs,
        write_json,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_p1_video_prefill import (  # type: ignore[no-redef]
        CONTEXT_LENGTH,
        HEAD_DIM,
        NUM_KV_HEADS,
        P1_GRAPH_NAME,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        dataset_fingerprint,
        load_bundle_contract,
        sha256_array,
        sha256_file,
        tokenize_video_prompt,
    )
    from run_aihub_p1_screen import (  # type: ignore[no-redef]
        DEVICE_NAME,
        DEVICE_OS,
        derive_condition_label,
        load_input_artifact,
        read_json,
        status_code,
        validate_model,
        validate_resumed_jobs,
        write_json,
    )

FIXED_SHARDS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "p2",
            {
                "model_id": "mm60gk9vm",
                "graph": "ar128_cl512_2_of_4",
                "first_layer": 7,
                "last_layer": 13,
                "hidden_input": "add_15805",
                "hidden_output": "add_30134",
            },
        ),
        (
            "p3",
            {
                "model_id": "mq3lg676q",
                "graph": "ar128_cl512_3_of_4",
                "first_layer": 14,
                "last_layer": 20,
                "hidden_input": "add_30134",
                "hidden_output": "add_44463",
            },
        ),
        (
            "p4",
            {
                "model_id": "mno2p69vq",
                "graph": "ar128_cl512_4_of_4",
                "first_layer": 21,
                "last_layer": 27,
                "hidden_input": "add_44463",
                "hidden_output": "logits",
            },
        ),
    ]
)

P1_SHARD = {
    "graph": P1_GRAPH_NAME,
    "first_layer": 0,
    "last_layer": 6,
    "hidden_output": "add_15805",
}
SHARD_PREDECESSOR = {"p2": "p1", "p3": "p2", "p4": "p3"}
CHOICES = ("A", "B", "C", "D")
TERMINAL_FAILURE_STATES = {"FAILED", "CANCELED", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def graph_option(graph: str) -> str:
    return f"--qnn_options context_enable_graphs={graph}"


def load_static_chunks(
    artifact_dir: Path,
) -> tuple[list["OrderedDict[str, np.ndarray]"], dict[str, Any]]:
    artifact_dir = artifact_dir.expanduser().resolve()
    manifest = read_json(artifact_dir / "input_manifest.json")
    records = manifest.get("static_chunk_archives")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError(
            "Input artifact must contain three static_chunk_archives; rebuild "
            "it with the current build_p1_video_prefill.py"
        )
    chunks: list["OrderedDict[str, np.ndarray]"] = []
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Invalid static chunk archive record")
        if int(record["chunk_index"]) != expected_index:
            raise ValueError("Static chunk archive order is not consecutive")
        path = artifact_dir / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Static chunk archive SHA mismatch: {path}")
        order = record.get("input_order")
        if not isinstance(order, list):
            raise ValueError(f"Static chunk archive has no input order: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(order):
                raise ValueError(f"Static chunk tensor names mismatch: {path}")
            chunk = OrderedDict(
                (name, np.ascontiguousarray(archive[name])) for name in order
            )
        if dataset_fingerprint(chunk) != record.get(
            "tensor_fingerprint_sha256"
        ):
            raise ValueError(f"Static chunk fingerprint mismatch: {path}")
        for name, value in chunk.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(
                value
            ).all():
                raise ValueError(f"Static tensor {name} contains non-finite values")
        chunks.append(chunk)

    all_chunks = manifest.get("all_chunks")
    if not isinstance(all_chunks, list) or len(all_chunks) != len(chunks):
        raise ValueError("Input manifest has no matching all_chunks records")
    valid_tokens = [int(record["valid_tokens"]) for record in all_chunks]
    if (
        valid_tokens[:-1] != [PREFILL_AR] * (len(valid_tokens) - 1)
        or not 1 <= valid_tokens[-1] <= PREFILL_AR
    ):
        raise ValueError(
            f"Invalid frozen AR128 prompt chunks: {valid_tokens}"
        )
    prompt_record = manifest.get("prompt_input_ids")
    prompt_tokens = (
        int(prompt_record.get("shape", [-1])[0])
        if isinstance(prompt_record, dict)
        else -1
    )
    if (
        sum(valid_tokens) != prompt_tokens
        or prompt_tokens > PREFILL_KV_LENGTH
        or CONTEXT_LENGTH != 512
    ):
        raise ValueError("Unexpected frozen prompt/context contract")
    return chunks, manifest


def validate_vision_scene_binding(
    source: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    video_manifest_sha256: str,
) -> list[str]:
    """Bind selected vision batches to every authenticated video pair."""

    vision = source.get("vision")
    if not isinstance(vision, dict):
        raise ValueError("Input artifact has no vision source provenance")
    if vision.get("video_manifest_sha256") != video_manifest_sha256:
        raise ValueError(
            "Vision source video-manifest SHA differs from input artifact"
        )

    pairs = video_manifest.get("pairs")
    selected_cases = vision.get("selected_cases")
    selected_indices = vision.get("selected_indices")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Video manifest has no visual pairs")
    if not isinstance(selected_cases, list) or not isinstance(
        selected_indices, list
    ):
        raise ValueError(
            "Vision source has no selected_cases/selected_indices inventory"
        )
    if len(selected_cases) != len(pairs) or len(selected_indices) != len(pairs):
        raise ValueError(
            "Vision source selected batch count differs from video pair count"
        )
    try:
        normalized_indices = [int(value) for value in selected_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError("Vision selected indices are not integers") from exc
    if any(value < 0 for value in normalized_indices):
        raise ValueError("Vision selected indices must be non-negative")
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("Vision selected indices must be unique")

    pixel_hashes: list[str] = []
    for pair_index, (pair, selected_case, selected_index) in enumerate(
        zip(pairs, selected_cases, normalized_indices, strict=True)
    ):
        if not isinstance(pair, dict) or not isinstance(selected_case, dict):
            raise ValueError("Video pair/selected vision case is not an object")
        if int(pair.get("pair_index", -1)) != pair_index:
            raise ValueError("Video pair indices are not consecutive")
        pixel_values = pair.get("pixel_values")
        if not isinstance(pixel_values, dict):
            raise ValueError(f"Video pair {pair_index} has no pixel values")
        pixel_hash = str(pixel_values.get("sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", pixel_hash) is None:
            raise ValueError(f"Video pair {pair_index} has an invalid pixel SHA")
        if selected_case.get("manifest_sha256") != video_manifest_sha256:
            raise ValueError(
                f"Vision case {pair_index} has a different video-manifest SHA"
            )
        if int(selected_case.get("pair_index", -1)) != pair_index:
            raise ValueError("Selected vision pair indices are not consecutive")
        if int(selected_case.get("index", -1)) != selected_index:
            raise ValueError(
                f"Vision case {pair_index} index differs from selected_indices"
            )
        if selected_case.get("pixel_sha256") != pixel_hash:
            raise ValueError(
                f"Vision case {pair_index} pixel SHA differs from video pair"
            )
        pixel_hashes.append(pixel_hash)
    return pixel_hashes


def validate_frozen_prompt_ids(
    prompt_ids: np.ndarray,
    prompt_record: Mapping[str, Any],
    *,
    tokenizer_path: Path,
    video_manifest: Mapping[str, Any],
) -> np.ndarray:
    """Authenticate stored prompt IDs against the exact builder tokenizer."""

    actual = np.asarray(prompt_ids)
    expected_shape = tuple(int(value) for value in prompt_record.get("shape", []))
    expected_dtype = str(prompt_record.get("dtype", ""))
    if actual.dtype != np.dtype(expected_dtype) or actual.shape != expected_shape:
        raise ValueError(
            "Prompt input-ID dtype/shape differs from input artifact"
        )
    if sha256_array(actual) != prompt_record.get("sha256"):
        raise ValueError("Prompt input-ID SHA no longer matches input artifact")

    regenerated = tokenize_video_prompt(tokenizer_path, video_manifest)
    if (
        regenerated.dtype != actual.dtype
        or regenerated.shape != actual.shape
        or not np.array_equal(regenerated, actual)
    ):
        raise ValueError(
            "Prompt input IDs differ from exact authenticated video prompt"
        )
    return actual


def load_ground_truth(
    benchmark_path: Path,
    *,
    case_id: str,
    probe_id: str,
    artifact_dir: Path,
    artifact_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    benchmark_path = benchmark_path.expanduser().resolve()
    benchmark = read_json(benchmark_path)
    cases = benchmark.get("video_cases")
    if not isinstance(cases, list):
        raise ValueError("Benchmark manifest has no video_cases list")
    matching_cases = [
        record
        for record in cases
        if isinstance(record, dict) and record.get("id") == case_id
    ]
    if len(matching_cases) != 1:
        raise ValueError(
            f"Expected exactly one benchmark case {case_id!r}, "
            f"found {len(matching_cases)}"
        )
    probes = matching_cases[0].get("choice_probes")
    if not isinstance(probes, list):
        raise ValueError(f"Benchmark case {case_id!r} has no choice probes")
    matching_probes = [
        record
        for record in probes
        if isinstance(record, dict) and record.get("id") == probe_id
    ]
    if len(matching_probes) != 1:
        raise ValueError(
            f"Expected exactly one probe {probe_id!r}, "
            f"found {len(matching_probes)}"
        )
    probe = matching_probes[0]
    expected_choice = probe.get("expected_letter")
    if expected_choice not in CHOICES:
        raise ValueError(f"Invalid expected choice: {expected_choice!r}")

    source = artifact_manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Input artifact has no source provenance")
    video_manifest_path = Path(str(source["video_manifest_path"])).resolve()
    if sha256_file(video_manifest_path) != source.get("video_manifest_sha256"):
        raise ValueError("Video manifest SHA no longer matches input artifact")
    video_manifest = read_json(video_manifest_path)
    pixel_pair_hashes = validate_vision_scene_binding(
        source,
        video_manifest,
        str(source["video_manifest_sha256"]),
    )
    temporal_pairs = matching_cases[0].get("temporal_pairs")
    frames = video_manifest.get("frames")
    if not isinstance(temporal_pairs, list) or not isinstance(frames, list):
        raise ValueError("Benchmark/video manifest has no temporal frame records")
    expected_frame_indices = [
        int(frame_index)
        for pair in temporal_pairs
        for frame_index in pair["frame_indices"]
    ]
    expected_times = [
        float(timestamp)
        for pair in temporal_pairs
        for timestamp in pair["assumed_times_seconds"]
    ]
    actual_frame_indices: list[int] = []
    actual_times: list[float] = []
    frame_hashes: list[str] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Video frame record is not an object")
        frame_path = Path(str(frame["path"]))
        if frame_path.parent.name != case_id:
            raise ValueError(
                f"Video frame parent {frame_path.parent.name!r} != "
                f"benchmark case {case_id!r}"
            )
        match = re.fullmatch(r"frame_(\d+)\.png", frame_path.name)
        if match is None:
            raise ValueError(f"Unexpected frame basename: {frame_path.name}")
        actual_frame_indices.append(int(match.group(1)))
        actual_times.append(float(frame["timestamp_seconds"]))
        frame_hashes.append(str(frame["sha256"]))
    if actual_frame_indices != expected_frame_indices:
        raise ValueError(
            f"Frozen frame indices {actual_frame_indices} != benchmark "
            f"{expected_frame_indices}"
        )
    if not np.allclose(actual_times, expected_times, rtol=0.0, atol=1e-6):
        raise ValueError("Frozen frame timestamps differ from benchmark")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in frame_hashes):
        raise ValueError("Frozen frame SHA inventory is invalid")
    text_chunks = video_manifest.get("text_chunks")
    if not isinstance(text_chunks, list):
        raise ValueError("Video manifest has no text_chunks")
    frozen_prompt = "".join(
        str(record["text"])
        for record in text_chunks
        if isinstance(record, dict)
    )
    probe_prompt = str(probe.get("prompt", ""))
    if not probe_prompt or frozen_prompt.count(probe_prompt) != 1:
        raise ValueError(
            "Frozen video prompt does not uniquely contain benchmark probe text"
        )
    expected_prompt_tokens = int(
        probe.get("context_budget", {}).get("prompt_tokens", -1)
    )
    prompt_record = artifact_manifest.get("prompt_input_ids")
    if not isinstance(prompt_record, dict):
        raise ValueError("Input artifact has no prompt_input_ids record")
    if expected_prompt_tokens != int(prompt_record.get("shape", [-1])[0]):
        raise ValueError(
            "Benchmark and frozen prompt token counts do not match"
        )
    loaded_prompt_ids = np.load(
        artifact_dir / str(prompt_record["path"]),
        allow_pickle=False,
    )

    bundle = load_bundle_contract(Path(str(source["bundle_dir"])))
    if sha256_file(bundle.tokenizer_path) != source.get("tokenizer_sha256"):
        raise ValueError("Tokenizer SHA no longer matches input artifact")
    validate_frozen_prompt_ids(
        loaded_prompt_ids,
        prompt_record,
        tokenizer_path=bundle.tokenizer_path,
        video_manifest=video_manifest,
    )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("tokenizers is required for choice-token binding") from exc
    tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))
    choice_token_ids: "OrderedDict[str, int]" = OrderedDict()
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False).ids
        if len(ids) != 1:
            raise ValueError(f"Choice {choice!r} does not map to one token: {ids}")
        choice_token_ids[choice] = int(ids[0])

    return {
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "case_id": case_id,
        "probe_id": probe_id,
        "expected_choice": expected_choice,
        "expected_event_label": probe.get("expected_event_label"),
        "probe_prompt": probe_prompt,
        "video_manifest_path": str(video_manifest_path),
        "video_manifest_sha256": source["video_manifest_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "prompt_input_ids_sha256": prompt_record["sha256"],
        "prompt_tokens": expected_prompt_tokens,
        "frame_indices": actual_frame_indices,
        "frame_timestamps_seconds": actual_times,
        "frame_sha256": frame_hashes,
        "pixel_pair_sha256": pixel_pair_hashes,
        "choice_token_ids": choice_token_ids,
        "final_chunk_index": len(artifact_manifest["all_chunks"]) - 1,
        "final_valid_row": (
            int(artifact_manifest["all_chunks"][-1]["valid_tokens"]) - 1
        ),
    }


def parse_candidates(
    values: Sequence[str],
    seed_screen: Mapping[str, Any],
) -> "OrderedDict[str, dict[str, str]]":
    if not values:
        raise ValueError("At least one --candidate label is required")
    targets = seed_screen.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("Seed P1 screen has no targets")
    candidates: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    for label in values:
        if label in candidates:
            raise ValueError(f"Duplicate candidate label: {label}")
        record = targets.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"Seed P1 screen has no target {label!r}")
        model_id = record.get("model_id")
        job_id = record.get("job_id")
        if not isinstance(model_id, str) or not isinstance(job_id, str):
            raise ValueError(f"Seed target {label!r} has no model/job ID")
        candidates[label] = {
            "model_id": model_id,
            "chunk0_job_id": job_id,
        }
    job_ids = [record["chunk0_job_id"] for record in candidates.values()]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Candidate seed job IDs must be unique")
    return candidates


def validate_seed_dataset(
    dataset_entries: Mapping[str, Sequence[np.ndarray]],
    expected_inputs: Mapping[str, np.ndarray],
    expected_fingerprint: str,
) -> None:
    if list(dataset_entries) != list(expected_inputs):
        raise ValueError("Seed dataset tensor names/order differ from artifact")
    downloaded: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for name, expected in expected_inputs.items():
        batches = dataset_entries[name]
        if len(batches) != 1:
            raise ValueError(
                f"Seed dataset {name} has {len(batches)} samples; expected 1"
            )
        actual = np.ascontiguousarray(batches[0])
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(
                f"Seed dataset {name} contract "
                f"{actual.dtype}/{actual.shape} != "
                f"{expected.dtype}/{expected.shape}"
            )
        if not np.array_equal(actual, expected, equal_nan=True):
            raise ValueError(f"Seed dataset {name} bytes differ from artifact")
        downloaded[name] = actual
    if dataset_fingerprint(downloaded) != expected_fingerprint:
        raise ValueError("Seed dataset fingerprint differs from artifact")


def validate_chunk0_static_binding(
    chunk0_inputs: Mapping[str, np.ndarray],
    chunk0_static: Mapping[str, np.ndarray],
) -> None:
    missing = [name for name in chunk0_static if name not in chunk0_inputs]
    if missing:
        raise ValueError(f"Chunk-0 inputs are missing static tensors: {missing}")
    for name, static_value in chunk0_static.items():
        input_value = np.asarray(chunk0_inputs[name])
        if (
            input_value.dtype != static_value.dtype
            or input_value.shape != static_value.shape
            or not np.array_equal(input_value, static_value, equal_nan=True)
        ):
            raise ValueError(
                f"Chunk-0 static tensor {name} differs from seed input archive"
            )


def job_key(chunk_index: int, shard: str, candidate: str | None = None) -> str:
    suffix = f":{candidate}" if candidate is not None else ""
    return f"chunk{chunk_index}:{shard}{suffix}"


def batch_fingerprint(entries: Mapping[str, Sequence[np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, batches in entries.items():
        digest.update(name.encode("utf-8"))
        digest.update(len(batches).to_bytes(4, "little"))
        for sample_index, value in enumerate(batches):
            sample = OrderedDict([(f"{sample_index}:{name}", value)])
            digest.update(dataset_fingerprint(sample).encode("ascii"))
    return digest.hexdigest()


def h5_group_by_name(handle: Any, name: str) -> Any:
    if "data" not in handle:
        raise ValueError("AI Hub output H5 has no /data group")
    for group in handle["data"].values():
        raw_name = group.attrs["name"]
        group_name = (
            raw_name.decode("utf-8")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        if group_name == name:
            return group
    raise KeyError(f"AI Hub output is missing tensor {name!r}")


def validate_h5_layout(handle: Any) -> None:
    if "data" not in handle:
        raise ValueError("AI Hub output H5 has no /data group")
    names: list[str] = []
    orders: list[int] = []
    for group in handle["data"].values():
        raw_name = group.attrs["name"]
        names.append(
            raw_name.decode("utf-8")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        orders.append(int(group.attrs["order"]))
        if int(group.attrs["batch_count"]) <= 0:
            raise ValueError("AI Hub H5 has a non-positive batch_count")
    if len(set(names)) != len(names):
        raise ValueError("AI Hub H5 has duplicate attrs-name tensors")
    if len(set(orders)) != len(orders):
        raise ValueError("AI Hub H5 has duplicate tensor order")


def h5_dataset_is_finite(dataset: Any) -> bool:
    if not np.issubdtype(dataset.dtype, np.floating):
        return True
    if dataset.chunks is not None:
        return all(
            np.isfinite(np.asarray(dataset[selection])).all()
            for selection in dataset.iter_chunks()
        )
    return bool(np.isfinite(np.asarray(dataset)).all())


def validate_output_h5(
    path: Path,
    model: Any,
    graph: str,
    expected_batch_count: int,
) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to validate AI Hub output") from exc
    with h5py.File(path, "r") as handle:
        validate_h5_layout(handle)
        records: list[tuple[int, str, Any]] = []
        for group in handle["data"].values():
            raw_name = group.attrs["name"]
            name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            records.append((int(group.attrs["order"]), name, group))
        records.sort(key=lambda item: item[0])
        expected_specs = model.output_spec[graph]
        if [name for _, name, _ in records] != [
            spec.name for spec in expected_specs
        ]:
            raise ValueError(f"{path} output names/order differ from model spec")
        for (_, name, group), spec in zip(
            records, expected_specs, strict=True
        ):
            batch_count = int(group.attrs["batch_count"])
            if batch_count != expected_batch_count:
                raise ValueError(
                    f"{path}:{name} batch_count {batch_count} != "
                    f"{expected_batch_count}"
                )
            expected_batch_keys = {
                f"batch_{index}" for index in range(expected_batch_count)
            }
            if set(group.keys()) != expected_batch_keys:
                raise ValueError(f"{path}:{name} batch datasets are incomplete")
            for batch_key in sorted(expected_batch_keys):
                dataset = group[batch_key]
                if str(dataset.dtype) != str(spec.dtype):
                    raise ValueError(
                        f"{path}:{name}/{batch_key} dtype {dataset.dtype} "
                        f"!= {spec.dtype}"
                    )
                if tuple(dataset.shape) != tuple(spec.shape):
                    raise ValueError(
                        f"{path}:{name}/{batch_key} shape {dataset.shape} "
                        f"!= {tuple(spec.shape)}"
                    )
                if not h5_dataset_is_finite(dataset):
                    raise ValueError(
                        f"{path}:{name}/{batch_key} contains non-finite values"
                    )


def load_h5_tensors(
    path: Path,
    names: Sequence[str],
    batch_index: int,
) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to read AI Hub output") from exc
    values: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        validate_h5_layout(handle)
        for name in names:
            group = h5_group_by_name(handle, name)
            batch_count = int(group.attrs["batch_count"])
            if not 0 <= batch_index < batch_count:
                raise IndexError(
                    f"{path}:{name} has {batch_count} samples, requested "
                    f"{batch_index}"
                )
            values[name] = np.asarray(group[f"batch_{batch_index}"])
            if np.issubdtype(
                values[name].dtype, np.floating
            ) and not np.isfinite(values[name]).all():
                raise ValueError(f"{path}:{name} contains non-finite values")
    return values


def output_batch_index(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    candidate: str,
) -> int:
    order = record.get("batch_order")
    if not isinstance(order, list):
        # P1 candidate jobs contain exactly one sample.
        return 0
    try:
        return order.index(candidate)
    except ValueError as exc:
        raise ValueError(
            f"Candidate {candidate!r} missing from job batch order {order}"
        ) from exc


def output_record(
    state: Mapping[str, Any],
    chunk_index: int,
    shard: str,
    candidate: str,
) -> Mapping[str, Any]:
    key = job_key(
        chunk_index,
        shard,
        candidate if shard == "p1" else None,
    )
    record = state["jobs"].get(key)
    if not isinstance(record, dict) or record.get("status") != "SUCCESS":
        raise RuntimeError(f"Required output is not ready: {key}")
    if not record.get("output_h5"):
        raise RuntimeError(f"Required output was not downloaded: {key}")
    return record


def cache_inputs(
    state: Mapping[str, Any],
    work_dir: Path,
    *,
    shard: str,
    candidate: str,
    chunk_index: int,
    first_layer: int,
    last_layer: int,
    valid_tokens: Sequence[int],
) -> "OrderedDict[str, np.ndarray]":
    keys = {
        layer: np.zeros(
            (NUM_KV_HEADS, 1, HEAD_DIM, PREFILL_KV_LENGTH),
            dtype=np.float32,
        )
        for layer in range(first_layer, last_layer + 1)
    }
    values = {
        layer: np.zeros(
            (NUM_KV_HEADS, 1, PREFILL_KV_LENGTH, HEAD_DIM),
            dtype=np.float32,
        )
        for layer in range(first_layer, last_layer + 1)
    }
    offset = 0
    for prior_chunk in range(chunk_index):
        record = output_record(
            state,
            prior_chunk,
            shard,
            candidate,
        )
        output_path = work_dir / str(record["output_h5"])
        batch_index = output_batch_index(state, record, candidate)
        names = [
            name
            for layer in range(first_layer, last_layer + 1)
            for name in (
                f"past_key_{layer}_out",
                f"past_value_{layer}_out",
            )
        ]
        tensors = load_h5_tensors(output_path, names, batch_index)
        count = int(valid_tokens[prior_chunk])
        if offset + count > PREFILL_KV_LENGTH:
            raise ValueError("Prompt history exceeds fixed P1 KV capacity")
        for layer in range(first_layer, last_layer + 1):
            new_key = tensors[f"past_key_{layer}_out"]
            new_value = tensors[f"past_value_{layer}_out"]
            if new_key.shape != (NUM_KV_HEADS, 1, HEAD_DIM, PREFILL_AR):
                raise ValueError(f"Unexpected key output shape: {new_key.shape}")
            if new_value.shape != (NUM_KV_HEADS, 1, PREFILL_AR, HEAD_DIM):
                raise ValueError(
                    f"Unexpected value output shape: {new_value.shape}"
                )
            if not np.isfinite(new_key).all() or not np.isfinite(
                new_value
            ).all():
                raise ValueError(
                    f"Non-finite KV output in {shard} layer {layer}"
                )
            keys[layer][..., offset : offset + count] = new_key[..., :count]
            values[layer][:, :, offset : offset + count, :] = new_value[
                :, :, :count, :
            ]
        offset += count

    result: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for layer in range(first_layer, last_layer + 1):
        result[f"past_key_{layer}_in"] = keys[layer]
        result[f"past_value_{layer}_in"] = values[layer]
    return result


def model_ordered_inputs(
    model: Any,
    graph: str,
    payload: Mapping[str, np.ndarray],
) -> "OrderedDict[str, np.ndarray]":
    if graph not in model.input_spec:
        raise ValueError(f"Model {model.model_id} has no graph {graph}")
    ordered: "OrderedDict[str, np.ndarray]" = OrderedDict()
    expected_names = [spec.name for spec in model.input_spec[graph]]
    if set(payload) != set(expected_names):
        raise ValueError(
            f"{model.model_id}/{graph} names differ: "
            f"missing={sorted(set(expected_names) - set(payload))}, "
            f"extra={sorted(set(payload) - set(expected_names))}"
        )
    for spec in model.input_spec[graph]:
        array = np.ascontiguousarray(payload[spec.name])
        actual = (str(array.dtype), tuple(array.shape))
        expected = (str(spec.dtype), tuple(spec.shape))
        if actual != expected:
            raise ValueError(
                f"{model.model_id}/{graph}/{spec.name}: "
                f"{actual} != {expected}"
            )
        ordered[spec.name] = array
    return ordered


def validate_model_output_contract(
    model: Any,
    *,
    graph: str,
    first_layer: int,
    last_layer: int,
    hidden_output: str,
) -> None:
    hidden_shape = (
        (1, PREFILL_AR, 151936)
        if hidden_output == "logits"
        else (1, PREFILL_AR, 2048)
    )
    expected = [
        (hidden_output, "float32", hidden_shape),
        *[
            item
            for layer in range(first_layer, last_layer + 1)
            for item in (
                (
                    f"past_key_{layer}_out",
                    "float32",
                    (NUM_KV_HEADS, 1, HEAD_DIM, PREFILL_AR),
                ),
                (
                    f"past_value_{layer}_out",
                    "float32",
                    (NUM_KV_HEADS, 1, PREFILL_AR, HEAD_DIM),
                ),
            )
        ],
    ]
    if graph not in model.output_spec:
        raise ValueError(f"Model {model.model_id} has no output graph {graph}")
    actual = [
        (spec.name, str(spec.dtype), tuple(spec.shape))
        for spec in model.output_spec[graph]
    ]
    if actual != expected:
        raise ValueError(
            f"Model {model.model_id}/{graph} output contract differs:\n"
            f"actual={actual!r}\nexpected={expected!r}"
        )


def validate_fixed_model_input_contract(
    model: Any,
    *,
    graph: str,
    first_layer: int,
    last_layer: int,
    hidden_input: str,
) -> None:
    expected = [
        *[
            item
            for layer in range(first_layer, last_layer + 1)
            for item in (
                (
                    f"past_key_{layer}_in",
                    "float32",
                    (NUM_KV_HEADS, 1, HEAD_DIM, PREFILL_KV_LENGTH),
                ),
                (
                    f"past_value_{layer}_in",
                    "float32",
                    (NUM_KV_HEADS, 1, PREFILL_KV_LENGTH, HEAD_DIM),
                ),
            )
        ],
        (hidden_input, "float32", (1, PREFILL_AR, 2048)),
        ("position_ids_cos", "float32", (1, 1, PREFILL_AR, HEAD_DIM // 2)),
        ("position_ids_sin", "float32", (1, 1, PREFILL_AR, HEAD_DIM // 2)),
        (
            "attention_mask",
            "float32",
            (1, 1, PREFILL_AR, CONTEXT_LENGTH),
        ),
    ]
    if graph not in model.input_spec:
        raise ValueError(f"Model {model.model_id} has no input graph {graph}")
    actual = [
        (spec.name, str(spec.dtype), tuple(spec.shape))
        for spec in model.input_spec[graph]
    ]
    if actual != expected:
        raise ValueError(
            f"Model {model.model_id}/{graph} input contract differs:\n"
            f"actual={actual!r}\nexpected={expected!r}"
        )


def predecessor_hidden(
    state: Mapping[str, Any],
    work_dir: Path,
    *,
    chunk_index: int,
    predecessor: str,
    candidate: str,
) -> np.ndarray:
    record = output_record(state, chunk_index, predecessor, candidate)
    output_name = (
        P1_SHARD["hidden_output"]
        if predecessor == "p1"
        else FIXED_SHARDS[predecessor]["hidden_output"]
    )
    batch_index = output_batch_index(state, record, candidate)
    return load_h5_tensors(
        work_dir / str(record["output_h5"]),
        [output_name],
        batch_index,
    )[output_name]


def build_p1_entries(
    model: Any,
    state: Mapping[str, Any],
    work_dir: Path,
    static_chunks: Sequence[Mapping[str, np.ndarray]],
    valid_tokens: Sequence[int],
    candidate: str,
    chunk_index: int,
) -> "OrderedDict[str, list[np.ndarray]]":
    payload: "OrderedDict[str, np.ndarray]" = cache_inputs(
        state,
        work_dir,
        shard="p1",
        candidate=candidate,
        chunk_index=chunk_index,
        first_layer=0,
        last_layer=6,
        valid_tokens=valid_tokens,
    )
    payload.update(static_chunks[chunk_index])
    ordered = model_ordered_inputs(model, P1_GRAPH_NAME, payload)
    return OrderedDict((name, [value]) for name, value in ordered.items())


def build_fixed_entries(
    model: Any,
    state: Mapping[str, Any],
    work_dir: Path,
    static_chunks: Sequence[Mapping[str, np.ndarray]],
    valid_tokens: Sequence[int],
    candidates: Sequence[str],
    shard: str,
    chunk_index: int,
) -> "OrderedDict[str, list[np.ndarray]]":
    config = FIXED_SHARDS[shard]
    predecessor = SHARD_PREDECESSOR[shard]
    samples: list["OrderedDict[str, np.ndarray]"] = []
    for candidate in candidates:
        payload: "OrderedDict[str, np.ndarray]" = cache_inputs(
            state,
            work_dir,
            shard=shard,
            candidate=candidate,
            chunk_index=chunk_index,
            first_layer=int(config["first_layer"]),
            last_layer=int(config["last_layer"]),
            valid_tokens=valid_tokens,
        )
        payload[str(config["hidden_input"])] = predecessor_hidden(
            state,
            work_dir,
            chunk_index=chunk_index,
            predecessor=predecessor,
            candidate=candidate,
        )
        static = static_chunks[chunk_index]
        for name in (
            "position_ids_cos",
            "position_ids_sin",
            "attention_mask",
        ):
            payload[name] = static[name]
        samples.append(
            model_ordered_inputs(model, str(config["graph"]), payload)
        )
    return OrderedDict(
        (name, [sample[name] for sample in samples])
        for name in samples[0]
    )


def job_is_ready(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("status") == "SUCCESS"
        and bool(record.get("output_h5"))
    )


def expected_job_contract(
    state: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    parts = key.split(":")
    if len(parts) not in {2, 3} or not parts[0].startswith("chunk"):
        raise ValueError(f"Invalid chain job key: {key}")
    chunk_index = int(parts[0][5:])
    if not 0 <= chunk_index < 3:
        raise ValueError(f"Invalid chunk index in job key: {key}")
    shard = parts[1]
    if shard == "p1":
        if len(parts) != 3 or parts[2] not in state["candidates"]:
            raise ValueError(f"Invalid P1 candidate job key: {key}")
        label = parts[2]
        return {
            "model_id": state["candidates"][label]["model_id"],
            "graph": P1_GRAPH_NAME,
            "batch_order": [label],
        }
    if len(parts) != 2 or shard not in FIXED_SHARDS:
        raise ValueError(f"Invalid fixed-shard job key: {key}")
    return {
        "model_id": FIXED_SHARDS[shard]["model_id"],
        "graph": FIXED_SHARDS[shard]["graph"],
        "batch_order": list(state["candidate_order"]),
    }


def submit_job(
    hub: Any,
    state: dict[str, Any],
    state_path: Path,
    *,
    key: str,
    model: Any,
    graph: str,
    entries: Mapping[str, Sequence[np.ndarray]],
    batch_order: Sequence[str],
    name: str,
) -> None:
    dataset = hub.upload_dataset(entries)
    job = hub.submit_inference_job(
        model,
        device=hub.Device(DEVICE_NAME, os=DEVICE_OS),
        inputs=dataset,
        options=graph_option(graph),
        name=name,
    )
    state["jobs"][key] = {
        "job_id": job.job_id,
        "job_name": job.name,
        "job_url": job.url,
        "model_id": model.model_id,
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint_sha256": batch_fingerprint(entries),
        "graph": graph,
        "options": graph_option(graph),
        "batch_order": list(batch_order),
        "status": "SUBMITTED_UNPOLLED",
        "submitted_at": utc_now(),
    }
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "event": "submitted",
                "key": key,
                "job_id": job.job_id,
                "dataset_id": dataset.dataset_id,
                "batch_order": list(batch_order),
            }
        ),
        flush=True,
    )


def collect_jobs(
    hub: Any,
    state: dict[str, Any],
    state_path: Path,
    work_dir: Path,
) -> None:
    job_ids = [record.get("job_id") for record in state["jobs"].values()]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Existing chain contains duplicate paid job IDs")
    for key, record in state["jobs"].items():
        expected = expected_job_contract(state, key)
        for field in ("model_id", "graph", "batch_order"):
            if record.get(field) != expected[field]:
                raise ValueError(
                    f"Existing chain {key} {field}={record.get(field)!r} "
                    f"!= {expected[field]!r}"
                )
        expected_options = graph_option(str(expected["graph"]))
        if record.get("options") != expected_options:
            raise ValueError(
                f"Existing chain {key} options differ from graph contract"
            )
        if not record.get("dataset_id"):
            raise ValueError(f"Existing chain {key} has no dataset ID")
        if key.startswith("chunk0:p1:"):
            label = key.rsplit(":", 1)[-1]
            if record.get("job_id") != state["candidates"][label].get(
                "chunk0_job_id"
            ):
                raise ValueError(f"Existing seed job ID differs for {label}")
        job = hub.get_job(record["job_id"])
        validation_errors: list[str] = []
        if job.model.model_id != record["model_id"]:
            validation_errors.append(
                f"model {job.model.model_id} != {record['model_id']}"
            )
        if job.options != record["options"]:
            validation_errors.append(
                f"options {job.options!r} != {record['options']!r}"
            )
        if job.device.name != DEVICE_NAME or str(job.device.os) != DEVICE_OS:
            validation_errors.append(
                f"device {job.device.name}/{job.device.os} != "
                f"{DEVICE_NAME}/{DEVICE_OS}"
            )
        actual_dataset_id = getattr(
            getattr(job, "inputs", None), "dataset_id", None
        )
        if actual_dataset_id != record.get("dataset_id"):
            validation_errors.append(
                f"dataset {actual_dataset_id} != {record.get('dataset_id')}"
            )
        if validation_errors:
            raise ValueError(
                f"AI Hub resume validation failed for {key}:\n- "
                + "\n- ".join(validation_errors)
            )
        if record.get("output_h5"):
            existing_output = work_dir / str(record["output_h5"])
            if (
                not existing_output.is_file()
                or sha256_file(existing_output) != record.get("output_sha256")
            ):
                raise ValueError(
                    f"Downloaded output is missing or modified: {existing_output}"
                )
            validate_output_h5(
                existing_output,
                job.model,
                str(record["graph"]),
                len(record["batch_order"]),
            )
        code = status_code(job)
        record["status"] = code
        record["observed_at"] = utc_now()
        if code in TERMINAL_FAILURE_STATES:
            write_json(state_path, state)
            raise RuntimeError(f"AI Hub job {record['job_id']} ended as {code}")
        if code == "SUCCESS" and not record.get("output_h5"):
            safe_key = key.replace(":", "_")
            output_path = work_dir / f"{safe_key}_{job.job_id}.h5"
            if output_path.is_file():
                expected_sha = record.get("output_sha256")
                if not expected_sha or sha256_file(output_path) != expected_sha:
                    raise ValueError(
                        f"Refusing to reuse unverified output: {output_path}"
                    )
            else:
                partial_path = output_path.with_suffix(".partial.h5")
                if partial_path.exists():
                    if not partial_path.is_file():
                        raise ValueError(
                            f"Partial output path is not a file: {partial_path}"
                        )
                    partial_path.unlink()
                job.download_output_data(str(partial_path))
                validate_output_h5(
                    partial_path,
                    job.model,
                    str(record["graph"]),
                    len(record["batch_order"]),
                )
                partial_path.replace(output_path)
            validate_output_h5(
                output_path,
                job.model,
                str(record["graph"]),
                len(record["batch_order"]),
            )
            record["output_h5"] = output_path.name
            record["output_sha256"] = sha256_file(output_path)
            print(
                json.dumps(
                    {
                        "event": "downloaded",
                        "key": key,
                        "job_id": job.job_id,
                        "path": output_path.name,
                    }
                ),
                flush=True,
            )
        state["updated_at"] = utc_now()
        write_json(state_path, state)


def initialize_state(
    state_path: Path,
    artifact_dir: Path,
    artifact_manifest: Mapping[str, Any],
    seed_screen_path: Path,
    seed_screen: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, str]],
    ground_truth: Mapping[str, Any],
) -> dict[str, Any]:
    if state_path.is_file():
        state = read_json(state_path)
        mismatches: list[str] = []
        if state.get("candidate_order") != list(candidates):
            mismatches.append("candidate order differs")
        if state.get("candidates") != candidates:
            mismatches.append("candidate model/seed-job mapping differs")
        expected_condition = derive_condition_label(artifact_manifest)
        if state.get("condition_label") != expected_condition:
            mismatches.append("condition label differs")
        if state.get("device") != {"name": DEVICE_NAME, "os": DEVICE_OS}:
            mismatches.append("device/OS differs")
        state_input = state.get("input", {})
        if state_input.get("chunk0_dataset_fingerprint_sha256") != (
            artifact_manifest.get("dataset_fingerprint_sha256")
        ):
            mismatches.append("input dataset fingerprint differs")
        if state_input.get("input_manifest_sha256") != sha256_file(
            artifact_dir / "input_manifest.json"
        ):
            mismatches.append("input manifest SHA differs")
        if state.get("seed_p1_screen", {}).get(
            "sha256_at_initialization"
        ) != sha256_file(seed_screen_path):
            mismatches.append("seed P1 screen SHA differs")
        if state.get("expected") != ground_truth:
            mismatches.append("ground-truth/prompt binding differs")
        if state.get("fixed_shards") != FIXED_SHARDS:
            mismatches.append("fixed shard contract differs")
        if mismatches:
            raise ValueError(
                "Refusing to reuse incompatible chain output directory:\n- "
                + "\n- ".join(mismatches)
            )
        return state

    input_fingerprint = artifact_manifest.get("dataset_fingerprint_sha256")
    if (
        seed_screen.get("input", {}).get("dataset_fingerprint_sha256")
        != input_fingerprint
    ):
        raise ValueError(
            "Seed P1 screen and static artifact dataset fingerprints differ"
        )
    condition = derive_condition_label(artifact_manifest)
    state: dict[str, Any] = {
        "schema_version": 1,
        "purpose": (
            "Three-chunk P1->P4 first-token chain with per-shard KV history"
        ),
        "created_at": utc_now(),
        "condition_label": condition,
        "device": {"name": DEVICE_NAME, "os": DEVICE_OS},
        "input": {
            "artifact_dir": str(artifact_dir),
            "input_manifest_sha256": sha256_file(
                artifact_dir / "input_manifest.json"
            ),
            "chunk0_dataset_fingerprint_sha256": input_fingerprint,
            "vision": artifact_manifest.get("source", {}).get("vision"),
            "valid_tokens": [
                int(record["valid_tokens"])
                for record in artifact_manifest["all_chunks"]
            ],
        },
        "seed_p1_screen": {
            "path": str(seed_screen_path),
            "sha256_at_initialization": sha256_file(seed_screen_path),
        },
        "candidates": candidates,
        "candidate_order": list(candidates),
        "fixed_shards": FIXED_SHARDS,
        "jobs": {},
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": True,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
        "expected": ground_truth,
        "limitations": [
            (
                "This validates the first generated token for one frozen "
                "multiple-choice video prompt, not free-form answer parity."
            ),
            (
                "P2/P3/P4 batch candidates as independent samples; no state "
                "is shared between candidate batch elements."
            ),
        ],
    }
    for label, candidate in candidates.items():
        screen_record = seed_screen["targets"][label]
        state["jobs"][job_key(0, "p1", label)] = {
            "job_id": candidate["chunk0_job_id"],
            "job_name": screen_record.get("job_name"),
            "job_url": screen_record.get("job_url"),
            "model_id": candidate["model_id"],
            "dataset_id": screen_record.get("dataset_id"),
            "dataset_fingerprint_sha256": input_fingerprint,
            "graph": P1_GRAPH_NAME,
            "options": graph_option(P1_GRAPH_NAME),
            "batch_order": [label],
            "status": screen_record.get("status", "UNKNOWN"),
            "seeded": True,
        }
    write_json(state_path, state)
    return state


def schedule_ready_jobs(
    hub: Any,
    state: dict[str, Any],
    state_path: Path,
    work_dir: Path,
    static_chunks: Sequence[Mapping[str, np.ndarray]],
    models: Mapping[str, Any],
) -> None:
    labels = list(state["candidate_order"])
    valid_tokens = state["input"]["valid_tokens"]
    condition = state["condition_label"]

    for chunk_index in range(3):
        if chunk_index > 0:
            for label in labels:
                key = job_key(chunk_index, "p1", label)
                previous = state["jobs"].get(
                    job_key(chunk_index - 1, "p1", label)
                )
                if key in state["jobs"] or not job_is_ready(previous):
                    continue
                model = models[f"p1:{label}"]
                entries = build_p1_entries(
                    model,
                    state,
                    work_dir,
                    static_chunks,
                    valid_tokens,
                    label,
                    chunk_index,
                )
                submit_job(
                    hub,
                    state,
                    state_path,
                    key=key,
                    model=model,
                    graph=P1_GRAPH_NAME,
                    entries=entries,
                    batch_order=[label],
                    name=(
                        f"cosmos_first_token_{condition}_chunk{chunk_index}_"
                        f"p1_{label}_r1"
                    ),
                )

        for shard in FIXED_SHARDS:
            key = job_key(chunk_index, shard)
            if key in state["jobs"]:
                continue
            predecessor = SHARD_PREDECESSOR[shard]
            dependencies = [
                state["jobs"].get(
                    job_key(
                        chunk_index,
                        predecessor,
                        label if predecessor == "p1" else None,
                    )
                )
                for label in labels
            ]
            # For fixed predecessor shards, every list entry intentionally
            # refers to the same batched job and merely checks it once.
            if not dependencies or not all(
                job_is_ready(record) for record in dependencies
            ):
                continue
            model = models[shard]
            entries = build_fixed_entries(
                model,
                state,
                work_dir,
                static_chunks,
                valid_tokens,
                labels,
                shard,
                chunk_index,
            )
            config = FIXED_SHARDS[shard]
            submit_job(
                hub,
                state,
                state_path,
                key=key,
                model=model,
                graph=str(config["graph"]),
                entries=entries,
                batch_order=labels,
                name=(
                    f"cosmos_first_token_{condition}_chunk{chunk_index}_"
                    f"{shard}_batched_r1"
                ),
            )


def score_logits_tensor(
    logits: np.ndarray,
    *,
    final_row: int,
    expected_choice: str,
    choice_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    logits = np.asarray(logits)
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or not 0 <= final_row < logits.shape[1]
        or max(choice_token_ids.values()) >= logits.shape[2]
    ):
        raise ValueError(f"Unexpected logits contract: {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain non-finite values")
    row = logits[0, final_row].astype(np.float64)
    choice_logits = OrderedDict(
        (choice, float(row[token_id]))
        for choice, token_id in choice_token_ids.items()
    )
    predicted_token_id = int(np.argmax(row))
    choice_prediction = max(choice_logits, key=choice_logits.get)
    competing = max(
        value
        for choice, value in choice_logits.items()
        if choice != expected_choice
    )
    top_count = min(10, row.size)
    top_indices = np.argpartition(row, -top_count)[-top_count:]
    top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
    return {
        "logits_row": final_row,
        "predicted_token_id": predicted_token_id,
        "predicted_choice": next(
            (
                choice
                for choice, token_id in choice_token_ids.items()
                if token_id == predicted_token_id
            ),
            None,
        ),
        "expected_choice": expected_choice,
        "first_token_correct": (
            predicted_token_id == choice_token_ids[expected_choice]
        ),
        "choice_constrained_prediction": choice_prediction,
        "choice_constrained_correct": choice_prediction == expected_choice,
        "expected_choice_margin": choice_logits[expected_choice] - competing,
        "choice_logits": choice_logits,
        "top_token_ids": [
            {"token_id": int(index), "logit": float(row[index])}
            for index in top_indices
        ],
    }


def score_final_logits(
    state: dict[str, Any],
    state_path: Path,
    work_dir: Path,
) -> bool:
    expected = state["expected"]
    final_chunk_index = int(expected["final_chunk_index"])
    final_record = state["jobs"].get(job_key(final_chunk_index, "p4"))
    if not job_is_ready(final_record):
        state["complete"] = False
        write_json(state_path, state)
        return False
    labels = list(state["candidate_order"])
    expected_choice = str(expected["expected_choice"])
    choice_token_ids = OrderedDict(
        (choice, int(token_id))
        for choice, token_id in expected["choice_token_ids"].items()
    )
    final_row = int(expected["final_valid_row"])
    results: list[dict[str, Any]] = []
    output_path = work_dir / str(final_record["output_h5"])
    for label in labels:
        batch_index = output_batch_index(state, final_record, label)
        logits = load_h5_tensors(
            output_path,
            ["logits"],
            batch_index,
        )["logits"]
        if logits.shape != (1, PREFILL_AR, 151936):
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        scored = score_logits_tensor(
            logits,
            final_row=final_row,
            expected_choice=expected_choice,
            choice_token_ids=choice_token_ids,
        )
        results.append(
            {
                "candidate": label,
                "model_id": state["candidates"][label]["model_id"],
                "batch_index": batch_index,
                **scored,
            }
        )
    result = {
        "schema_version": 1,
        "purpose": (
            "First-token result for frozen "
            f"{expected['case_id']}/{expected['probe_id']} video prompt"
        ),
        "condition_label": state["condition_label"],
        "source_chain_manifest": state_path.name,
        "p4_job_id": final_record["job_id"],
        "p4_output_sha256": final_record["output_sha256"],
        "ground_truth_binding": {
            key: expected[key]
            for key in (
                "benchmark_sha256",
                "case_id",
                "probe_id",
                "expected_choice",
                "expected_event_label",
                "video_manifest_sha256",
                "tokenizer_sha256",
                "prompt_input_ids_sha256",
                "prompt_tokens",
                "final_chunk_index",
                "final_valid_row",
            )
        },
        "results": results,
        "all_candidates_first_token_correct": all(
            record["first_token_correct"] for record in results
        ),
        "limitations": state["limitations"],
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": False,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
    }
    result_path = work_dir / "first_token_results.json"
    write_json(result_path, result)
    state["complete"] = True
    state["first_token_results"] = {
        "path": result_path.name,
        "sha256": sha256_file(result_path),
        "all_candidates_first_token_correct": result[
            "all_candidates_first_token_correct"
        ],
    }
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_artifact_dir", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument(
        "--seed-p1-screen",
        required=True,
        type=Path,
        help="Completed/active P1 screen containing chunk-0 job IDs",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate label from the seed P1 screen (repeatable)",
    )
    parser.add_argument(
        "--benchmark-manifest",
        required=True,
        type=Path,
        help="Tracked benchmark JSON containing the expected choice",
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all local/live contracts without writing state or jobs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = args.input_artifact_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    seed_screen_path = args.seed_p1_screen.expanduser().resolve()
    state_path = work_dir / "chain.json"

    static_chunks, artifact_manifest = load_static_chunks(artifact_dir)
    chunk0_inputs, chunk0_manifest = load_input_artifact(artifact_dir)
    if chunk0_manifest.get("dataset_fingerprint_sha256") != (
        artifact_manifest.get("dataset_fingerprint_sha256")
    ):
        raise ValueError("Static and chunk-0 artifact manifests disagree")
    validate_chunk0_static_binding(chunk0_inputs, static_chunks[0])
    ground_truth = load_ground_truth(
        args.benchmark_manifest,
        case_id=args.case_id,
        probe_id=args.probe_id,
        artifact_dir=artifact_dir,
        artifact_manifest=artifact_manifest,
    )
    seed_screen = read_json(seed_screen_path)
    candidates = parse_candidates(args.candidate, seed_screen)

    import qai_hub as hub

    seed_jobs = {
        label: hub.get_job(record["chunk0_job_id"])
        for label, record in candidates.items()
    }
    common_dataset_id = validate_resumed_jobs(
        seed_jobs,
        expected_targets={
            label: record["model_id"] for label, record in candidates.items()
        },
    )
    seed_statuses = {
        label: status_code(job) for label, job in seed_jobs.items()
    }
    incomplete_seed_jobs = {
        label: status
        for label, status in seed_statuses.items()
        if status != "SUCCESS"
    }
    if incomplete_seed_jobs:
        raise ValueError(
            f"Seed P1 jobs are not successful: {incomplete_seed_jobs}"
        )
    for label, job in seed_jobs.items():
        recorded_dataset = seed_screen["targets"][label].get("dataset_id")
        if recorded_dataset != common_dataset_id:
            raise ValueError(
                f"Seed screen {label} dataset {recorded_dataset!r} does not "
                f"match actual common dataset {common_dataset_id!r}"
            )
    seed_dataset_entries = next(iter(seed_jobs.values())).inputs.download()
    if not isinstance(seed_dataset_entries, dict):
        raise ValueError("AI Hub seed dataset did not download as tensor entries")
    validate_seed_dataset(
        seed_dataset_entries,
        chunk0_inputs,
        str(artifact_manifest["dataset_fingerprint_sha256"]),
    )

    models: dict[str, Any] = {
        f"p1:{label}": hub.get_model(candidates[label]["model_id"])
        for label in candidates
    }
    models.update(
        {
            shard: hub.get_model(config["model_id"])
            for shard, config in FIXED_SHARDS.items()
        }
    )
    for label in candidates:
        validate_model(models[f"p1:{label}"], chunk0_inputs)
        validate_model_output_contract(
            models[f"p1:{label}"],
            graph=P1_GRAPH_NAME,
            first_layer=0,
            last_layer=6,
            hidden_output="add_15805",
        )
    for shard, config in FIXED_SHARDS.items():
        validate_fixed_model_input_contract(
            models[shard],
            graph=str(config["graph"]),
            first_layer=int(config["first_layer"]),
            last_layer=int(config["last_layer"]),
            hidden_input=str(config["hidden_input"]),
        )
        validate_model_output_contract(
            models[shard],
            graph=str(config["graph"]),
            first_layer=int(config["first_layer"]),
            last_layer=int(config["last_layer"]),
            hidden_output=str(config["hidden_output"]),
        )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "candidate_order": list(candidates),
                    "seed_job_statuses": seed_statuses,
                    "common_seed_dataset_id": common_dataset_id,
                    "dataset_fingerprint_sha256": artifact_manifest[
                        "dataset_fingerprint_sha256"
                    ],
                    "ground_truth": ground_truth,
                },
                indent=2,
            )
        )
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    state = initialize_state(
        state_path,
        artifact_dir,
        artifact_manifest,
        seed_screen_path,
        seed_screen,
        candidates,
        ground_truth,
    )

    collect_jobs(hub, state, state_path, work_dir)
    schedule_ready_jobs(
        hub,
        state,
        state_path,
        work_dir,
        static_chunks,
        models,
    )
    complete = score_final_logits(state, state_path, work_dir)
    print(
        json.dumps(
            {
                "chain_manifest": str(state_path),
                "complete": complete,
                "jobs": {
                    key: {
                        "job_id": record["job_id"],
                        "status": record["status"],
                        "dataset_id": record.get("dataset_id"),
                    }
                    for key, record in state["jobs"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
