#!/usr/bin/env python3
"""Run a resumable multi-probe Cosmos P1->P4 first-token chain on AI Hub.

The runner batches prompt variants without sharing model state between them:

* each P1 candidate job contains one sample per probe;
* each fixed P2/P3/P4 job contains one sample per
  ``(probe, candidate)`` stream; and
* every stream reconstructs its own layer-local KV history from its own H5
  batch row.

With four probes, two P1 candidates, three prompt chunks, and three fixed
shards, the complete experiment uses exactly 15 hosted inference jobs:
``3 * (2 P1 candidates + P2 + P3 + P4)``.

All artifacts are rebuilt from their authenticated video manifest and hosted
vision H5 before any state is written.  ``--preflight-only`` performs local and
live model validation without creating the work directory, uploading a
dataset, submitting a job, or modifying an existing state file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.build_p1_video_prefill import (
        HEAD_DIM,
        NUM_KV_HEADS,
        P1_GRAPH_NAME,
        P1_TARGETS,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        dataset_fingerprint,
        make_p1_chunk_inputs,
        prepare_prompt,
        sha256_array,
        sha256_file,
        zero_p1_kv_inputs,
    )
    from scripts.run_aihub_first_token_chain import (
        CHOICES,
        FIXED_SHARDS,
        P1_SHARD,
        SHARD_PREDECESSOR,
        TERMINAL_FAILURE_STATES,
        batch_fingerprint,
        graph_option,
        h5_group_by_name,
        job_is_ready,
        load_ground_truth,
        load_h5_tensors,
        load_static_chunks,
        model_ordered_inputs,
        score_logits_tensor,
        validate_chunk0_static_binding,
        validate_fixed_model_input_contract,
        validate_h5_layout,
        validate_model_output_contract,
        validate_output_h5,
    )
    from scripts.run_aihub_p1_screen import (
        DEVICE_NAME,
        DEVICE_OS,
        derive_condition_label,
        load_input_artifact,
        read_json,
        status_code,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_p1_video_prefill import (  # type: ignore[no-redef]
        HEAD_DIM,
        NUM_KV_HEADS,
        P1_GRAPH_NAME,
        P1_TARGETS,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        dataset_fingerprint,
        make_p1_chunk_inputs,
        prepare_prompt,
        sha256_array,
        sha256_file,
        zero_p1_kv_inputs,
    )
    from run_aihub_first_token_chain import (  # type: ignore[no-redef]
        CHOICES,
        FIXED_SHARDS,
        P1_SHARD,
        SHARD_PREDECESSOR,
        TERMINAL_FAILURE_STATES,
        batch_fingerprint,
        graph_option,
        h5_group_by_name,
        job_is_ready,
        load_ground_truth,
        load_h5_tensors,
        load_static_chunks,
        model_ordered_inputs,
        score_logits_tensor,
        validate_chunk0_static_binding,
        validate_fixed_model_input_contract,
        validate_h5_layout,
        validate_model_output_contract,
        validate_output_h5,
    )
    from run_aihub_p1_screen import (  # type: ignore[no-redef]
        DEVICE_NAME,
        DEVICE_OS,
        derive_condition_label,
        load_input_artifact,
        read_json,
        status_code,
        utc_now,
        write_json,
    )


SCHEMA_VERSION = 1
PANEL_SCHEMA_VERSION = 1
CHUNK_COUNT = 3
STATE_FILENAME = "batched_chain.json"
RESULT_FILENAME = "batched_first_token_results.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
JOB_KEY_RE = re.compile(r"^chunk([0-2]):(p1|p2|p3|p4)(?::(.+))?$")


@dataclass(frozen=True)
class ProbeRuntime:
    key: str
    case_id: str
    probe_id: str
    artifact_dir: Path
    static_chunks: tuple[Mapping[str, np.ndarray], ...]
    artifact_manifest: Mapping[str, Any]
    ground_truth: Mapping[str, Any]


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probe_key(case_id: str, probe_id: str) -> str:
    if not case_id or not probe_id:
        raise ValueError("Case and probe IDs must be non-empty")
    if "::" in case_id or "::" in probe_id:
        raise ValueError("Case/probe IDs may not contain the '::' separator")
    return f"{case_id}::{probe_id}"


def stream_key(key: str, candidate: str) -> str:
    if not key or not candidate or "::" in candidate:
        raise ValueError("Invalid probe/candidate stream key")
    return f"{key}::{candidate}"


def split_stream_key(value: str) -> tuple[str, str]:
    parts = value.rsplit("::", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid stream key: {value!r}")
    return parts[0], parts[1]


def chain_job_key(
    chunk_index: int,
    shard: str,
    candidate: str | None = None,
) -> str:
    if not 0 <= chunk_index < CHUNK_COUNT:
        raise ValueError(f"Invalid chunk index: {chunk_index}")
    if shard == "p1":
        if not candidate:
            raise ValueError("P1 job keys require a candidate")
        return f"chunk{chunk_index}:p1:{candidate}"
    if shard not in FIXED_SHARDS or candidate is not None:
        raise ValueError(f"Invalid fixed-shard job key: {shard}/{candidate}")
    return f"chunk{chunk_index}:{shard}"


def parse_job_key(key: str) -> tuple[int, str, str | None]:
    match = JOB_KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid batched-chain job key: {key}")
    chunk_index = int(match.group(1))
    shard = match.group(2)
    candidate = match.group(3)
    if (shard == "p1") != (candidate is not None):
        raise ValueError(f"Invalid candidate suffix in job key: {key}")
    return chunk_index, shard, candidate


def arrays_equal(
    actual: np.ndarray,
    expected: np.ndarray,
) -> bool:
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and np.array_equal(actual, expected, equal_nan=True)
    )


def validate_alias_provenance(source: Mapping[str, Any]) -> None:
    vision = source.get("vision")
    if not isinstance(vision, dict):
        raise ValueError("Input artifact has no vision provenance")
    if vision.get("vision_output_kind") != "qai_hub_inference_h5":
        raise ValueError(
            "Batched NPU panel requires an authenticated AI Hub vision H5"
        )
    for field in (
        "vision_outputs_sha256",
        "vision_index_manifest_sha256",
        "video_manifest_sha256",
    ):
        value = vision.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"Vision provenance has invalid {field}")
    hub_provenance = vision.get("qai_hub_provenance")
    if not isinstance(hub_provenance, dict) or any(
        not isinstance(hub_provenance.get(field), str)
        or not hub_provenance[field]
        for field in ("inference_job_id", "model_id", "dataset_id")
    ):
        raise ValueError("Vision H5 has incomplete AI Hub provenance")

    index_path = Path(str(vision.get("vision_index_manifest_path", "")))
    if not index_path.is_file():
        raise ValueError("Derived vision-index manifest is missing")
    index = read_json(index_path)
    if (
        index.get("schema_version") != 2
        or index.get("index_kind")
        != "pixel_verified_vision_batch_aliases"
    ):
        raise ValueError(
            "Batched prompt variants require the pixel-verified alias index"
        )
    derivation = index.get("derivation")
    if not isinstance(derivation, dict):
        raise ValueError("Derived vision index has no derivation record")
    source_index = derivation.get("source_index")
    if not isinstance(source_index, dict):
        raise ValueError("Derived vision index has no source-index record")
    source_index_sha = source_index.get("sha256")
    if (
        not isinstance(source_index_sha, str)
        or SHA256_RE.fullmatch(source_index_sha) is None
    ):
        raise ValueError("Derived vision index has an invalid source SHA")

    video_manifest_sha = str(vision["video_manifest_sha256"])
    selected_cases = vision.get("selected_cases")
    selected_indices = vision.get("selected_indices")
    if not isinstance(selected_cases, list) or not isinstance(
        selected_indices, list
    ):
        raise ValueError("Vision provenance has no selected alias inventory")
    for pair_index, (case, selected_index) in enumerate(
        zip(selected_cases, selected_indices, strict=True)
    ):
        if not isinstance(case, dict):
            raise ValueError("Selected alias case must be an object")
        alias = case.get("alias_provenance")
        if not isinstance(alias, dict):
            raise ValueError(
                f"Selected vision case {pair_index} is not an explicit alias"
            )
        expected = {
            "match_kind": "exact_pair_index_and_pixel_sha256",
            "source_index_sha256": source_index_sha,
            "source_batch_index": int(selected_index),
            "target_manifest_sha256": video_manifest_sha,
            "pair_index": pair_index,
            "pixel_sha256": case.get("pixel_sha256"),
        }
        for field, value in expected.items():
            if alias.get(field) != value:
                raise ValueError(
                    f"Selected alias {pair_index} {field} differs from "
                    "its authenticated binding"
                )


def validate_rebuilt_artifact(
    artifact_dir: Path,
    static_chunks: Sequence[Mapping[str, np.ndarray]],
    artifact_manifest: Mapping[str, Any],
    chunk0_inputs: Mapping[str, np.ndarray],
) -> None:
    source = artifact_manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Input artifact has no source record")
    validate_alias_provenance(source)
    vision = source["vision"]
    rebuilt = prepare_prompt(
        bundle_dir=Path(str(source["bundle_dir"])),
        video_manifest_path=Path(str(source["video_manifest_path"])),
        vision_outputs_path=Path(str(vision["vision_outputs_path"])),
        vision_index_manifest_path=Path(
            str(vision["vision_index_manifest_path"])
        ),
        vision_source_label=str(vision["vision_source_label"]),
        vision_job_id=str(vision["qai_hub_provenance"]["inference_job_id"]),
        vision_model_id=str(vision["qai_hub_provenance"]["model_id"]),
        vision_dataset_id=str(vision["qai_hub_provenance"]["dataset_id"]),
    )
    if rebuilt.source_record != source:
        differing = sorted(
            key
            for key in set(rebuilt.source_record) | set(source)
            if rebuilt.source_record.get(key) != source.get(key)
        )
        raise ValueError(
            "Rebuilt artifact source provenance differs in fields: "
            f"{differing}"
        )
    if len(rebuilt.chunks) != CHUNK_COUNT:
        raise ValueError(
            f"Rebuilt prompt has {len(rebuilt.chunks)} chunks; "
            f"expected {CHUNK_COUNT}"
        )
    if len(static_chunks) != CHUNK_COUNT:
        raise ValueError("Input artifact does not contain exactly three chunks")
    for chunk_index, (actual, expected) in enumerate(
        zip(static_chunks, rebuilt.chunks, strict=True)
    ):
        if list(actual) != list(expected):
            raise ValueError(
                f"Chunk {chunk_index} rebuilt tensor order differs"
            )
        for name in actual:
            if not arrays_equal(
                np.asarray(actual[name]),
                np.asarray(expected[name]),
            ):
                raise ValueError(
                    f"Chunk {chunk_index} tensor {name} differs from "
                    "the rebuilt manifest/H5 input"
                )
    rebuilt_chunk0 = make_p1_chunk_inputs(rebuilt, 0)
    if list(chunk0_inputs) != list(rebuilt_chunk0):
        raise ValueError("Chunk-0 input tensor order differs from rebuild")
    for name in chunk0_inputs:
        if not arrays_equal(
            np.asarray(chunk0_inputs[name]),
            np.asarray(rebuilt_chunk0[name]),
        ):
            raise ValueError(
                f"Chunk-0 tensor {name} differs from exact rebuild"
            )


def resolve_artifact_dir(
    panel_path: Path,
    value: Any,
    description: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = panel_path.parent / path
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{description} is not a directory: {resolved}")
    return resolved


def load_panel_runtime(
    panel_manifest_path: Path,
    benchmark_path: Path,
) -> tuple[dict[str, Any], tuple[ProbeRuntime, ...]]:
    panel_path = panel_manifest_path.expanduser().resolve()
    benchmark_path = benchmark_path.expanduser().resolve()
    panel = read_json(panel_path)
    if panel.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValueError("Unsupported panel-manifest schema_version")
    panel_id = panel.get("panel_id")
    if not isinstance(panel_id, str) or SAFE_ID_RE.fullmatch(panel_id) is None:
        raise ValueError("Panel manifest has an unsafe panel_id")
    raw_probes = panel.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("Panel manifest must contain a non-empty probes list")

    runtimes: list[ProbeRuntime] = []
    seen_keys: set[str] = set()
    for ordinal, record in enumerate(raw_probes):
        description = f"panel probes[{ordinal}]"
        if not isinstance(record, dict):
            raise ValueError(f"{description} must be an object")
        case_id = record.get("case_id")
        probe_id = record.get("probe_id")
        if not isinstance(case_id, str) or not isinstance(probe_id, str):
            raise ValueError(f"{description} has invalid case/probe IDs")
        key = probe_key(case_id, probe_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate panel probe: {key}")
        seen_keys.add(key)
        artifact_dir = resolve_artifact_dir(
            panel_path,
            record.get("input_artifact_dir"),
            f"{description}.input_artifact_dir",
        )
        input_manifest_path = artifact_dir / "input_manifest.json"
        actual_manifest_sha = sha256_file(input_manifest_path)
        pinned_manifest_sha = record.get("input_manifest_sha256")
        if pinned_manifest_sha is not None:
            if (
                not isinstance(pinned_manifest_sha, str)
                or SHA256_RE.fullmatch(pinned_manifest_sha) is None
            ):
                raise ValueError(
                    f"{description}.input_manifest_sha256 is invalid"
                )
            if actual_manifest_sha != pinned_manifest_sha:
                raise ValueError(
                    f"{description} input manifest SHA changed: "
                    f"{actual_manifest_sha} != {pinned_manifest_sha}"
                )

        static_chunks, artifact_manifest = load_static_chunks(artifact_dir)
        chunk0_inputs, chunk0_manifest = load_input_artifact(artifact_dir)
        if chunk0_manifest != artifact_manifest:
            raise ValueError(
                f"{description} chunk-0 and static manifests differ"
            )
        validate_chunk0_static_binding(chunk0_inputs, static_chunks[0])
        validate_rebuilt_artifact(
            artifact_dir,
            static_chunks,
            artifact_manifest,
            chunk0_inputs,
        )
        ground_truth = load_ground_truth(
            benchmark_path,
            case_id=case_id,
            probe_id=probe_id,
            artifact_dir=artifact_dir,
            artifact_manifest=artifact_manifest,
        )
        runtimes.append(
            ProbeRuntime(
                key=key,
                case_id=case_id,
                probe_id=probe_id,
                artifact_dir=artifact_dir,
                static_chunks=tuple(static_chunks),
                artifact_manifest=artifact_manifest,
                ground_truth=ground_truth,
            )
        )

    first_source = runtimes[0].artifact_manifest["source"]
    first_vision = first_source["vision"]
    shared_contract = {
        "genie_config_sha256": first_source["genie_config_sha256"],
        "tokenizer_sha256": first_source["tokenizer_sha256"],
        "embedding_table_sha256": first_source["embedding_table_sha256"],
        "contract": first_source["contract"],
        "vision_outputs_sha256": first_vision["vision_outputs_sha256"],
        "vision_index_manifest_sha256": first_vision[
            "vision_index_manifest_sha256"
        ],
        "vision_output_kind": first_vision["vision_output_kind"],
        "vision_source_label": first_vision["vision_source_label"],
        "qai_hub_provenance": first_vision["qai_hub_provenance"],
    }
    for runtime in runtimes[1:]:
        source = runtime.artifact_manifest["source"]
        vision = source["vision"]
        candidate_contract = {
            "genie_config_sha256": source["genie_config_sha256"],
            "tokenizer_sha256": source["tokenizer_sha256"],
            "embedding_table_sha256": source["embedding_table_sha256"],
            "contract": source["contract"],
            "vision_outputs_sha256": vision["vision_outputs_sha256"],
            "vision_index_manifest_sha256": vision[
                "vision_index_manifest_sha256"
            ],
            "vision_output_kind": vision["vision_output_kind"],
            "vision_source_label": vision["vision_source_label"],
            "qai_hub_provenance": vision["qai_hub_provenance"],
        }
        if candidate_contract != shared_contract:
            raise ValueError(
                f"Probe {runtime.key} does not share the exact bundle/vision "
                "contract used by the panel"
            )

    return panel, tuple(runtimes)


def parse_candidates(values: Sequence[str]) -> "OrderedDict[str, str]":
    if not values:
        raise ValueError("At least one P1 candidate is required")
    result: "OrderedDict[str, str]" = OrderedDict()
    for label in values:
        if label in result:
            raise ValueError(f"Duplicate candidate label: {label}")
        if label not in P1_TARGETS:
            raise ValueError(
                f"Unknown P1 candidate {label!r}; choose from "
                f"{list(P1_TARGETS)}"
            )
        result[label] = P1_TARGETS[label]
    return result


def safe_job_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-")
    if not normalized:
        raise ValueError("AI Hub job name is empty after normalization")
    return normalized[:120]


def build_plan(
    panel_path: Path,
    benchmark_path: Path,
    panel: Mapping[str, Any],
    runtimes: Sequence[ProbeRuntime],
    candidates: Mapping[str, str],
    name_prefix: str,
) -> dict[str, Any]:
    if not runtimes:
        raise ValueError("Cannot build an empty panel plan")
    panel_id = str(panel["panel_id"])
    condition_labels = {
        derive_condition_label(runtime.artifact_manifest)
        for runtime in runtimes
    }
    if len(condition_labels) != 1:
        raise ValueError(
            f"Panel artifacts have different condition labels: "
            f"{sorted(condition_labels)}"
        )
    condition_label = next(iter(condition_labels))
    probe_order = [runtime.key for runtime in runtimes]
    candidate_order = list(candidates)
    streams = [
        stream_key(key, candidate)
        for key in probe_order
        for candidate in candidate_order
    ]

    probe_records: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for runtime in runtimes:
        manifest = runtime.artifact_manifest
        source = manifest["source"]
        probe_records[runtime.key] = {
            "case_id": runtime.case_id,
            "probe_id": runtime.probe_id,
            "artifact_dir": str(runtime.artifact_dir),
            "input_manifest_sha256": sha256_file(
                runtime.artifact_dir / "input_manifest.json"
            ),
            "chunk0_dataset_fingerprint_sha256": manifest[
                "dataset_fingerprint_sha256"
            ],
            "static_chunk_archive_sha256": [
                record["sha256"]
                for record in manifest["static_chunk_archives"]
            ],
            "valid_tokens": [
                int(record["valid_tokens"])
                for record in manifest["all_chunks"]
            ],
            "video_manifest_sha256": source["video_manifest_sha256"],
            "vision_outputs_sha256": source["vision"][
                "vision_outputs_sha256"
            ],
            "vision_index_manifest_sha256": source["vision"][
                "vision_index_manifest_sha256"
            ],
            "ground_truth": copy.deepcopy(runtime.ground_truth),
        }

    prefix = safe_job_name(name_prefix)
    job_order: list[str] = []
    contracts: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for chunk_index in range(CHUNK_COUNT):
        for candidate, model_id in candidates.items():
            key = chain_job_key(chunk_index, "p1", candidate)
            batch_order = [
                stream_key(probe, candidate) for probe in probe_order
            ]
            job_order.append(key)
            contracts[key] = {
                "model_id": model_id,
                "graph": P1_GRAPH_NAME,
                "options": graph_option(P1_GRAPH_NAME),
                "batch_order": batch_order,
                "job_name": safe_job_name(
                    f"{prefix}_{panel_id}_chunk{chunk_index}_p1_"
                    f"{candidate}_r1"
                ),
            }
        for shard, config in FIXED_SHARDS.items():
            key = chain_job_key(chunk_index, shard)
            job_order.append(key)
            contracts[key] = {
                "model_id": config["model_id"],
                "graph": config["graph"],
                "options": graph_option(str(config["graph"])),
                "batch_order": streams,
                "job_name": safe_job_name(
                    f"{prefix}_{panel_id}_chunk{chunk_index}_{shard}_r1"
                ),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": panel_id,
        "panel_manifest_path": str(panel_path.expanduser().resolve()),
        "panel_manifest_sha256": sha256_file(
            panel_path.expanduser().resolve()
        ),
        "benchmark_path": str(benchmark_path.expanduser().resolve()),
        "benchmark_sha256": sha256_file(
            benchmark_path.expanduser().resolve()
        ),
        "condition_label": condition_label,
        "device": {"name": DEVICE_NAME, "os": DEVICE_OS},
        "candidates": dict(candidates),
        "candidate_order": candidate_order,
        "probe_order": probe_order,
        "stream_order": streams,
        "probes": probe_records,
        "fixed_shards": copy.deepcopy(FIXED_SHARDS),
        "job_order": job_order,
        "job_contracts": contracts,
        "planned_paid_inference_jobs": len(job_order),
        "batching": {
            "p1_samples_per_job": len(probe_order),
            "fixed_samples_per_job": len(streams),
            "cache_isolation_key": "(probe_key,candidate,shard)",
            "order": (
                "probe manifest order outermost, candidate CLI order innermost"
            ),
        },
    }


def validate_model_contracts(
    hub: Any,
    plan: Mapping[str, Any],
    runtimes: Sequence[ProbeRuntime],
) -> dict[str, Any]:
    candidates = plan["candidates"]
    models: dict[str, Any] = {
        f"p1:{candidate}": hub.get_model(model_id)
        for candidate, model_id in candidates.items()
    }
    models.update(
        {
            shard: hub.get_model(config["model_id"])
            for shard, config in FIXED_SHARDS.items()
        }
    )

    for candidate in plan["candidate_order"]:
        model = models[f"p1:{candidate}"]
        if model.model_id != candidates[candidate]:
            raise ValueError(
                f"AI Hub returned model {model.model_id} for "
                f"{candidates[candidate]}"
            )
        validate_model_output_contract(
            model,
            graph=P1_GRAPH_NAME,
            first_layer=0,
            last_layer=6,
            hidden_output=str(P1_SHARD["hidden_output"]),
        )
        for runtime in runtimes:
            for static in runtime.static_chunks:
                payload = zero_p1_kv_inputs()
                payload.update(static)
                model_ordered_inputs(model, P1_GRAPH_NAME, payload)

    for shard, config in FIXED_SHARDS.items():
        model = models[shard]
        if model.model_id != config["model_id"]:
            raise ValueError(
                f"AI Hub returned model {model.model_id} for "
                f"{config['model_id']}"
            )
        validate_fixed_model_input_contract(
            model,
            graph=str(config["graph"]),
            first_layer=int(config["first_layer"]),
            last_layer=int(config["last_layer"]),
            hidden_input=str(config["hidden_input"]),
        )
        validate_model_output_contract(
            model,
            graph=str(config["graph"]),
            first_layer=int(config["first_layer"]),
            last_layer=int(config["last_layer"]),
            hidden_output=str(config["hidden_output"]),
        )
    return models


def initialize_or_validate_state(
    state_path: Path,
    plan: Mapping[str, Any],
    *,
    write_if_missing: bool,
) -> dict[str, Any]:
    plan_copy = copy.deepcopy(dict(plan))
    plan_sha = canonical_json_sha256(plan_copy)
    if state_path.is_file():
        state = read_json(state_path)
        mismatches: list[str] = []
        if state.get("schema_version") != SCHEMA_VERSION:
            mismatches.append("state schema version differs")
        if state.get("plan_sha256") != plan_sha:
            mismatches.append("immutable plan SHA differs")
        if state.get("plan") != plan_copy:
            mismatches.append("immutable plan content differs")
        if mismatches:
            raise ValueError(
                "Refusing to reuse incompatible batched-chain state:\n- "
                + "\n- ".join(mismatches)
            )
        jobs = state.get("jobs")
        intents = state.get("submission_intents")
        if not isinstance(jobs, dict) or not isinstance(intents, dict):
            raise ValueError("Batched-chain state has invalid job inventories")
        return state

    state = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Multi-probe three-chunk P1->P4 first-token chain with isolated "
            "per-probe/candidate/shard KV history"
        ),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "plan_sha256": plan_sha,
        "plan": plan_copy,
        "jobs": {},
        "submission_intents": {},
        "complete": False,
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": True,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
        "security": {
            "contains_credentials": False,
            "note": "No environment variables or API tokens are serialized.",
        },
    }
    if write_if_missing:
        write_json(state_path, state)
    return state


def runtime_by_key(
    runtimes: Sequence[ProbeRuntime],
) -> dict[str, ProbeRuntime]:
    return {runtime.key: runtime for runtime in runtimes}


def batch_index_for_stream(
    record: Mapping[str, Any],
    stream: str,
) -> int:
    order = record.get("batch_order")
    if not isinstance(order, list):
        raise ValueError("Job record has no deterministic batch order")
    if len(order) != len(set(order)):
        raise ValueError("Job record has duplicate batch-order entries")
    try:
        return order.index(stream)
    except ValueError as exc:
        raise ValueError(
            f"Stream {stream!r} is absent from job batch order"
        ) from exc


def physical_output_record(
    state: Mapping[str, Any],
    *,
    chunk_index: int,
    shard: str,
    candidate: str,
) -> Mapping[str, Any]:
    key = chain_job_key(
        chunk_index,
        shard,
        candidate if shard == "p1" else None,
    )
    record = state["jobs"].get(key)
    if not job_is_ready(record):
        raise RuntimeError(f"Required batched output is not ready: {key}")
    return record


def isolated_cache_inputs(
    state: Mapping[str, Any],
    work_dir: Path,
    *,
    probe: str,
    candidate: str,
    shard: str,
    chunk_index: int,
    first_layer: int,
    last_layer: int,
) -> "OrderedDict[str, np.ndarray]":
    probe_record = state["plan"]["probes"].get(probe)
    if not isinstance(probe_record, dict):
        raise ValueError(f"Unknown cache probe: {probe}")
    valid_tokens = probe_record.get("valid_tokens")
    if (
        not isinstance(valid_tokens, list)
        or len(valid_tokens) != CHUNK_COUNT
    ):
        raise ValueError(f"Probe {probe} has invalid valid-token history")
    stream = stream_key(probe, candidate)
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
        record = physical_output_record(
            state,
            chunk_index=prior_chunk,
            shard=shard,
            candidate=candidate,
        )
        batch_index = batch_index_for_stream(record, stream)
        output_path = work_dir / str(record["output_h5"])
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
        if count <= 0 or offset + count > PREFILL_KV_LENGTH:
            raise ValueError(
                f"Probe {probe} history exceeds fixed KV capacity"
            )
        for layer in range(first_layer, last_layer + 1):
            new_key = tensors[f"past_key_{layer}_out"]
            new_value = tensors[f"past_value_{layer}_out"]
            if new_key.shape != (NUM_KV_HEADS, 1, HEAD_DIM, PREFILL_AR):
                raise ValueError(f"Unexpected key output shape: {new_key.shape}")
            if new_value.shape != (
                NUM_KV_HEADS,
                1,
                PREFILL_AR,
                HEAD_DIM,
            ):
                raise ValueError(
                    f"Unexpected value output shape: {new_value.shape}"
                )
            if not np.isfinite(new_key).all() or not np.isfinite(
                new_value
            ).all():
                raise ValueError(
                    f"Non-finite KV output in {probe}/{candidate}/{shard}"
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


def predecessor_hidden(
    state: Mapping[str, Any],
    work_dir: Path,
    *,
    probe: str,
    candidate: str,
    chunk_index: int,
    predecessor: str,
) -> np.ndarray:
    record = physical_output_record(
        state,
        chunk_index=chunk_index,
        shard=predecessor,
        candidate=candidate,
    )
    output_name = (
        str(P1_SHARD["hidden_output"])
        if predecessor == "p1"
        else str(FIXED_SHARDS[predecessor]["hidden_output"])
    )
    batch_index = batch_index_for_stream(
        record,
        stream_key(probe, candidate),
    )
    return load_h5_tensors(
        work_dir / str(record["output_h5"]),
        [output_name],
        batch_index,
    )[output_name]


def build_batched_p1_entries(
    model: Any,
    state: Mapping[str, Any],
    work_dir: Path,
    runtimes: Mapping[str, ProbeRuntime],
    *,
    candidate: str,
    chunk_index: int,
) -> "OrderedDict[str, list[np.ndarray]]":
    samples: list["OrderedDict[str, np.ndarray]"] = []
    for key in state["plan"]["probe_order"]:
        runtime = runtimes[key]
        payload = isolated_cache_inputs(
            state,
            work_dir,
            probe=key,
            candidate=candidate,
            shard="p1",
            chunk_index=chunk_index,
            first_layer=0,
            last_layer=6,
        )
        payload.update(runtime.static_chunks[chunk_index])
        samples.append(model_ordered_inputs(model, P1_GRAPH_NAME, payload))
    return OrderedDict(
        (name, [sample[name] for sample in samples])
        for name in samples[0]
    )


def build_batched_fixed_entries(
    model: Any,
    state: Mapping[str, Any],
    work_dir: Path,
    runtimes: Mapping[str, ProbeRuntime],
    *,
    shard: str,
    chunk_index: int,
) -> "OrderedDict[str, list[np.ndarray]]":
    config = FIXED_SHARDS[shard]
    predecessor = str(SHARD_PREDECESSOR[shard])
    samples: list["OrderedDict[str, np.ndarray]"] = []
    for stream in state["plan"]["stream_order"]:
        key, candidate = split_stream_key(stream)
        runtime = runtimes[key]
        payload = isolated_cache_inputs(
            state,
            work_dir,
            probe=key,
            candidate=candidate,
            shard=shard,
            chunk_index=chunk_index,
            first_layer=int(config["first_layer"]),
            last_layer=int(config["last_layer"]),
        )
        payload[str(config["hidden_input"])] = predecessor_hidden(
            state,
            work_dir,
            probe=key,
            candidate=candidate,
            chunk_index=chunk_index,
            predecessor=predecessor,
        )
        for name in (
            "position_ids_cos",
            "position_ids_sin",
            "attention_mask",
        ):
            payload[name] = runtime.static_chunks[chunk_index][name]
        samples.append(
            model_ordered_inputs(model, str(config["graph"]), payload)
        )
    return OrderedDict(
        (name, [sample[name] for sample in samples])
        for name in samples[0]
    )


def build_entries_for_job(
    key: str,
    state: Mapping[str, Any],
    work_dir: Path,
    runtimes: Mapping[str, ProbeRuntime],
    models: Mapping[str, Any],
) -> "OrderedDict[str, list[np.ndarray]]":
    chunk_index, shard, candidate = parse_job_key(key)
    if shard == "p1":
        assert candidate is not None
        return build_batched_p1_entries(
            models[f"p1:{candidate}"],
            state,
            work_dir,
            runtimes,
            candidate=candidate,
            chunk_index=chunk_index,
        )
    return build_batched_fixed_entries(
        models[shard],
        state,
        work_dir,
        runtimes,
        shard=shard,
        chunk_index=chunk_index,
    )


def dependencies_for_job(
    key: str,
    plan: Mapping[str, Any],
) -> list[str]:
    chunk_index, shard, candidate = parse_job_key(key)
    if shard == "p1":
        if chunk_index == 0:
            return []
        assert candidate is not None
        return [chain_job_key(chunk_index - 1, "p1", candidate)]
    if shard == "p2":
        dependencies = [
            chain_job_key(chunk_index, "p1", label)
            for label in plan["candidate_order"]
        ]
    else:
        predecessor = str(SHARD_PREDECESSOR[shard])
        dependencies = [chain_job_key(chunk_index, predecessor)]
    if chunk_index > 0:
        dependencies.append(chain_job_key(chunk_index - 1, shard))
    return dependencies


def validate_submission_intents(state: Mapping[str, Any]) -> None:
    intents = state.get("submission_intents")
    if not isinstance(intents, dict):
        raise ValueError("State has no submission-intent inventory")
    for key, intent in intents.items():
        if key not in state["plan"]["job_contracts"]:
            raise ValueError(f"Unknown submission intent: {key}")
        if not isinstance(intent, dict):
            raise ValueError(f"Submission intent {key} is not an object")
        phase = intent.get("phase")
        if phase not in {"before_dataset_upload", "dataset_uploaded", "submitting_job"}:
            raise ValueError(f"Submission intent {key} has invalid phase")
        if phase == "submitting_job":
            raise RuntimeError(
                "Submission state is ambiguous for "
                f"{key}: the process stopped after recording the paid-job "
                "submission boundary. Reconcile the recorded dataset/job name "
                "in AI Hub and add the job ID before resuming; the runner will "
                "not risk a duplicate paid job."
            )


def submit_job_atomic(
    hub: Any,
    state: dict[str, Any],
    state_path: Path,
    *,
    key: str,
    model: Any,
    entries: Mapping[str, Sequence[np.ndarray]],
) -> None:
    if key in state["jobs"]:
        raise ValueError(f"Refusing to resubmit existing job: {key}")
    contract = state["plan"]["job_contracts"][key]
    fingerprint = batch_fingerprint(entries)
    immutable_intent = {
        "model_id": contract["model_id"],
        "graph": contract["graph"],
        "options": contract["options"],
        "batch_order": contract["batch_order"],
        "job_name": contract["job_name"],
        "dataset_fingerprint_sha256": fingerprint,
        "plan_sha256": state["plan_sha256"],
    }
    intent = state["submission_intents"].get(key)
    if intent is None:
        intent = {
            **copy.deepcopy(immutable_intent),
            "phase": "before_dataset_upload",
            "created_at": utc_now(),
        }
        state["submission_intents"][key] = intent
        state["updated_at"] = utc_now()
        write_json(state_path, state)
    else:
        for field, expected in immutable_intent.items():
            if intent.get(field) != expected:
                raise ValueError(
                    f"Submission intent {key} {field} differs from exact inputs"
                )

    phase = intent.get("phase")
    if phase == "submitting_job":
        validate_submission_intents(state)
        raise AssertionError("unreachable")
    if phase == "before_dataset_upload":
        dataset = hub.upload_dataset(entries)
        intent["dataset_id"] = dataset.dataset_id
        intent["phase"] = "dataset_uploaded"
        intent["dataset_uploaded_at"] = utc_now()
        state["updated_at"] = utc_now()
        write_json(state_path, state)
    elif phase == "dataset_uploaded":
        dataset_id = intent.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"Submission intent {key} has no dataset ID")
        dataset = hub.get_dataset(dataset_id)
        if dataset.dataset_id != dataset_id:
            raise ValueError(f"AI Hub returned the wrong dataset for {key}")
    else:
        raise ValueError(f"Submission intent {key} has invalid phase {phase!r}")

    intent["phase"] = "submitting_job"
    intent["submission_boundary_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    job = hub.submit_inference_job(
        model,
        device=hub.Device(DEVICE_NAME, os=DEVICE_OS),
        inputs=dataset,
        options=str(contract["options"]),
        name=str(contract["job_name"]),
    )
    state["jobs"][key] = {
        "job_id": job.job_id,
        "job_name": job.name,
        "job_url": job.url,
        "model_id": model.model_id,
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint_sha256": fingerprint,
        "input_plan_sha256": state["plan_sha256"],
        "graph": contract["graph"],
        "options": contract["options"],
        "batch_order": copy.deepcopy(contract["batch_order"]),
        "status": "SUBMITTED_UNPOLLED",
        "submitted_at": utc_now(),
    }
    del state["submission_intents"][key]
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "event": "submitted",
                "key": key,
                "job_id": job.job_id,
                "dataset_id": dataset.dataset_id,
                "batch_order": contract["batch_order"],
            }
        ),
        flush=True,
    )


def validate_job_record(
    state: Mapping[str, Any],
    key: str,
    record: Mapping[str, Any],
) -> None:
    contract = state["plan"]["job_contracts"].get(key)
    if not isinstance(contract, dict):
        raise ValueError(f"State contains an unplanned paid job: {key}")
    for field in ("model_id", "graph", "options", "batch_order"):
        if record.get(field) != contract[field]:
            raise ValueError(
                f"Existing job {key} {field} differs from immutable plan"
            )
    if record.get("input_plan_sha256") != state["plan_sha256"]:
        raise ValueError(f"Existing job {key} input-plan SHA differs")
    fingerprint = record.get("dataset_fingerprint_sha256")
    if not isinstance(fingerprint, str) or SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"Existing job {key} has an invalid dataset fingerprint")
    for field in ("job_id", "dataset_id"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"Existing job {key} has no {field}")


def collect_jobs(
    hub: Any,
    state: dict[str, Any],
    state_path: Path,
    work_dir: Path,
    models: Mapping[str, Any],
) -> None:
    validate_submission_intents(state)
    job_ids = [
        record.get("job_id")
        for record in state["jobs"].values()
        if isinstance(record, dict)
    ]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Batched chain contains duplicate paid job IDs")

    for key in state["plan"]["job_order"]:
        record = state["jobs"].get(key)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"Existing job {key} is not an object")
        validate_job_record(state, key, record)
        job = hub.get_job(record["job_id"])
        errors: list[str] = []
        if job.model.model_id != record["model_id"]:
            errors.append(
                f"model {job.model.model_id} != {record['model_id']}"
            )
        if job.options != record["options"]:
            errors.append(
                f"options {job.options!r} != {record['options']!r}"
            )
        if job.device.name != DEVICE_NAME or str(job.device.os) != DEVICE_OS:
            errors.append(
                f"device {job.device.name}/{job.device.os} != "
                f"{DEVICE_NAME}/{DEVICE_OS}"
            )
        actual_dataset_id = getattr(
            getattr(job, "inputs", None),
            "dataset_id",
            None,
        )
        if actual_dataset_id != record["dataset_id"]:
            errors.append(
                f"dataset {actual_dataset_id} != {record['dataset_id']}"
            )
        if errors:
            raise ValueError(
                f"AI Hub resume validation failed for {key}:\n- "
                + "\n- ".join(errors)
            )

        expected_batch_count = len(record["batch_order"])
        if record.get("output_h5"):
            output_path = work_dir / str(record["output_h5"])
            if (
                not output_path.is_file()
                or sha256_file(output_path) != record.get("output_sha256")
            ):
                raise ValueError(
                    f"Downloaded output is missing or changed: {output_path}"
                )
            validate_output_h5(
                output_path,
                models[
                    f"p1:{parse_job_key(key)[2]}"
                    if parse_job_key(key)[1] == "p1"
                    else parse_job_key(key)[1]
                ],
                str(record["graph"]),
                expected_batch_count,
            )

        code = status_code(job)
        record["status"] = code
        record["observed_at"] = utc_now()
        if code in TERMINAL_FAILURE_STATES:
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            raise RuntimeError(f"AI Hub job {record['job_id']} ended as {code}")
        if code == "SUCCESS" and not record.get("output_h5"):
            safe_key = key.replace(":", "_")
            output_path = work_dir / f"{safe_key}_{job.job_id}.h5"
            partial_path = output_path.with_suffix(".partial.h5")
            if output_path.is_file():
                expected_sha = record.get("output_sha256")
                if not expected_sha or sha256_file(output_path) != expected_sha:
                    raise ValueError(
                        f"Refusing to reuse unverified output: {output_path}"
                    )
            else:
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
                    expected_batch_count,
                )
                partial_path.replace(output_path)
            validate_output_h5(
                output_path,
                job.model,
                str(record["graph"]),
                expected_batch_count,
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


def schedule_ready_jobs(
    hub: Any,
    state: dict[str, Any],
    state_path: Path,
    work_dir: Path,
    runtimes: Mapping[str, ProbeRuntime],
    models: Mapping[str, Any],
    *,
    submit_missing: bool,
) -> None:
    if not submit_missing:
        return
    validate_submission_intents(state)
    for key in state["plan"]["job_order"]:
        if key in state["jobs"]:
            continue
        dependencies = dependencies_for_job(key, state["plan"])
        if not all(job_is_ready(state["jobs"].get(item)) for item in dependencies):
            continue
        entries = build_entries_for_job(
            key,
            state,
            work_dir,
            runtimes,
            models,
        )
        contract = state["plan"]["job_contracts"][key]
        if list(entries) != [
            spec.name
            for spec in models[
                f"p1:{parse_job_key(key)[2]}"
                if parse_job_key(key)[1] == "p1"
                else parse_job_key(key)[1]
            ].input_spec[contract["graph"]]
        ]:
            raise ValueError(f"Job {key} input order differs from model contract")
        if any(
            len(samples) != len(contract["batch_order"])
            for samples in entries.values()
        ):
            raise ValueError(f"Job {key} input batch count differs from plan")
        model_key = (
            f"p1:{parse_job_key(key)[2]}"
            if parse_job_key(key)[1] == "p1"
            else parse_job_key(key)[1]
        )
        submit_job_atomic(
            hub,
            state,
            state_path,
            key=key,
            model=models[model_key],
            entries=entries,
        )


def score_results(
    state: dict[str, Any],
    state_path: Path,
    work_dir: Path,
) -> bool:
    final_key = chain_job_key(CHUNK_COUNT - 1, "p4")
    final_record = state["jobs"].get(final_key)
    if not job_is_ready(final_record):
        state["complete"] = False
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        return False

    output_path = work_dir / str(final_record["output_h5"])
    probe_results: list[dict[str, Any]] = []
    for key in state["plan"]["probe_order"]:
        probe = state["plan"]["probes"][key]
        expected = probe["ground_truth"]
        final_chunk_index = int(expected["final_chunk_index"])
        if final_chunk_index != CHUNK_COUNT - 1:
            raise ValueError(
                f"Probe {key} final chunk {final_chunk_index} is not "
                f"{CHUNK_COUNT - 1}"
            )
        expected_choice = str(expected["expected_choice"])
        if expected_choice not in CHOICES:
            raise ValueError(f"Probe {key} has an invalid expected choice")
        choice_token_ids = OrderedDict(
            (choice, int(token_id))
            for choice, token_id in expected["choice_token_ids"].items()
        )
        candidate_results: list[dict[str, Any]] = []
        for candidate in state["plan"]["candidate_order"]:
            stream = stream_key(key, candidate)
            batch_index = batch_index_for_stream(final_record, stream)
            logits = load_h5_tensors(
                output_path,
                ["logits"],
                batch_index,
            )["logits"]
            if logits.shape != (1, PREFILL_AR, 151936):
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            scored = score_logits_tensor(
                logits,
                final_row=int(expected["final_valid_row"]),
                expected_choice=expected_choice,
                choice_token_ids=choice_token_ids,
            )
            candidate_results.append(
                {
                    "candidate": candidate,
                    "model_id": state["plan"]["candidates"][candidate],
                    "stream_key": stream,
                    "batch_index": batch_index,
                    **scored,
                }
            )
        probe_results.append(
            {
                "probe_key": key,
                "ground_truth_binding": {
                    field: expected[field]
                    for field in (
                        "benchmark_sha256",
                        "case_id",
                        "probe_id",
                        "expected_choice",
                        "expected_event_label",
                        "video_manifest_sha256",
                        "tokenizer_sha256",
                        "prompt_input_ids_sha256",
                        "prompt_tokens",
                        "frame_indices",
                        "frame_timestamps_seconds",
                        "frame_sha256",
                        "pixel_pair_sha256",
                        "final_chunk_index",
                        "final_valid_row",
                    )
                },
                "results": candidate_results,
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Batched exact first-token NPU precision panel",
        "panel_id": state["plan"]["panel_id"],
        "condition_label": state["plan"]["condition_label"],
        "source_chain_manifest": state_path.name,
        "source_plan_sha256": state["plan_sha256"],
        "p4_job_id": final_record["job_id"],
        "p4_output_sha256": final_record["output_sha256"],
        "probe_order": state["plan"]["probe_order"],
        "candidate_order": state["plan"]["candidate_order"],
        "probes": probe_results,
        "all_first_tokens_correct": all(
            candidate["first_token_correct"]
            for probe in probe_results
            for candidate in probe["results"]
        ),
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": False,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
        "limitations": [
            (
                "This scores the first generated token for frozen "
                "multiple-choice prompts, not free-form answer parity."
            ),
            (
                "Batch elements share a physical inference job but never "
                "share KV-cache rows or hidden-state rows."
            ),
        ],
    }
    result_path = work_dir / RESULT_FILENAME
    write_json(result_path, result)
    state["complete"] = True
    state["results"] = {
        "path": result_path.name,
        "sha256": sha256_file(result_path),
        "all_first_tokens_correct": result["all_first_tokens_correct"],
    }
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel_manifest", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument(
        "--benchmark-manifest",
        required=True,
        type=Path,
        help="Tracked benchmark JSON containing each expected choice",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="P1 candidate label from build_p1_video_prefill.P1_TARGETS",
    )
    parser.add_argument(
        "--name-prefix",
        default="cosmos_exact_panel",
        help="Safe prefix for all AI Hub inference job names",
    )
    parser.add_argument(
        "--submit-missing",
        action="store_true",
        help="Submit newly ready jobs; otherwise only collect existing jobs",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate artifacts, H5 provenance, batching, state compatibility, "
            "and live model contracts without any write or paid submission"
        ),
    )
    return parser


def execute(
    args: argparse.Namespace,
    *,
    hub: Any,
) -> dict[str, Any]:
    panel_path = args.panel_manifest.expanduser().resolve()
    benchmark_path = args.benchmark_manifest.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    state_path = work_dir / STATE_FILENAME
    panel, runtime_sequence = load_panel_runtime(
        panel_path,
        benchmark_path,
    )
    candidates = parse_candidates(args.candidate)
    plan = build_plan(
        panel_path,
        benchmark_path,
        panel,
        runtime_sequence,
        candidates,
        args.name_prefix,
    )
    models = validate_model_contracts(hub, plan, runtime_sequence)
    preflight_state = initialize_or_validate_state(
        state_path,
        plan,
        write_if_missing=False,
    )
    validate_submission_intents(preflight_state)

    summary: dict[str, Any] = {
        "panel_id": plan["panel_id"],
        "plan_sha256": canonical_json_sha256(plan),
        "probe_order": plan["probe_order"],
        "candidate_order": plan["candidate_order"],
        "stream_order": plan["stream_order"],
        "planned_paid_inference_jobs": plan[
            "planned_paid_inference_jobs"
        ],
        "p1_samples_per_job": plan["batching"]["p1_samples_per_job"],
        "fixed_samples_per_job": plan["batching"][
            "fixed_samples_per_job"
        ],
    }
    if args.preflight_only:
        return {"preflight": "passed", **summary}

    work_dir.mkdir(parents=True, exist_ok=True)
    state = initialize_or_validate_state(
        state_path,
        plan,
        write_if_missing=True,
    )
    runtime_map = runtime_by_key(runtime_sequence)
    collect_jobs(hub, state, state_path, work_dir, models)
    schedule_ready_jobs(
        hub,
        state,
        state_path,
        work_dir,
        runtime_map,
        models,
        submit_missing=bool(args.submit_missing),
    )
    complete = score_results(state, state_path, work_dir)
    return {
        **summary,
        "chain_manifest": str(state_path),
        "complete": complete,
        "submitted_job_count": len(state["jobs"]),
        "jobs": {
            key: {
                "job_id": record["job_id"],
                "status": record["status"],
                "dataset_id": record.get("dataset_id"),
            }
            for key, record in state["jobs"].items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import qai_hub as hub

    result = execute(args, hub=hub)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
