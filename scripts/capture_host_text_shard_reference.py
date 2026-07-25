#!/usr/bin/env python3
"""Capture a BF16 host reference for an exact hosted P2 or P3 dataset.

The source dataset is taken from a completed AI Hub inference job.  Every
input tensor, batch element, model identity, graph option, shape, dtype, and
dataset ID is validated before the local decoder layers execute.  The output
is an intermediate hidden/KV reference for numerical screening; it is not an
answer-accuracy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.run_aihub_first_token_chain import (
        CONTEXT_LENGTH,
        FIXED_SHARDS,
        HEAD_DIM,
        NUM_KV_HEADS,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        batch_fingerprint,
        graph_option,
        sha256_array,
        sha256_file,
    )
    from scripts.run_aihub_p1_screen import status_code
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_aihub_first_token_chain import (  # type: ignore[no-redef]
        CONTEXT_LENGTH,
        FIXED_SHARDS,
        HEAD_DIM,
        NUM_KV_HEADS,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        batch_fingerprint,
        graph_option,
        sha256_array,
        sha256_file,
    )
    from run_aihub_p1_screen import status_code  # type: ignore[no-redef]

SUPPORTED_SHARDS = ("p2", "p3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_input_shapes(shard: str) -> "OrderedDict[str, tuple[int, ...]]":
    config = FIXED_SHARDS[shard]
    expected: "OrderedDict[str, tuple[int, ...]]" = OrderedDict()
    for layer_index in range(
        int(config["first_layer"]),
        int(config["last_layer"]) + 1,
    ):
        expected[f"past_key_{layer_index}_in"] = (
            NUM_KV_HEADS,
            1,
            HEAD_DIM,
            PREFILL_KV_LENGTH,
        )
        expected[f"past_value_{layer_index}_in"] = (
            NUM_KV_HEADS,
            1,
            PREFILL_KV_LENGTH,
            HEAD_DIM,
        )
    expected[str(config["hidden_input"])] = (1, PREFILL_AR, 2048)
    expected["position_ids_cos"] = (1, 1, PREFILL_AR, HEAD_DIM // 2)
    expected["position_ids_sin"] = (1, 1, PREFILL_AR, HEAD_DIM // 2)
    expected["attention_mask"] = (
        1,
        1,
        PREFILL_AR,
        CONTEXT_LENGTH,
    )
    return expected


def validate_dataset_entries(
    entries: Mapping[str, Sequence[np.ndarray]],
    *,
    shard: str,
    batch_labels: Sequence[str] | None,
) -> tuple["OrderedDict[str, list[np.ndarray]]", list[str]]:
    expected = expected_input_shapes(shard)
    if list(entries) != list(expected):
        raise ValueError(
            "AI Hub dataset input order differs from the frozen shard "
            f"contract:\nactual={list(entries)!r}\n"
            f"expected={list(expected)!r}"
        )
    batch_counts = {len(values) for values in entries.values()}
    if len(batch_counts) != 1:
        raise ValueError("AI Hub dataset tensors have inconsistent batch counts")
    batch_count = next(iter(batch_counts), 0)
    if batch_count <= 0:
        raise ValueError("AI Hub dataset has no batch samples")
    if batch_labels is None:
        labels = [f"batch_{index}" for index in range(batch_count)]
    else:
        labels = list(batch_labels)
        if len(labels) != batch_count:
            raise ValueError(
                f"Expected {batch_count} batch labels, got {len(labels)}"
            )
        if len(set(labels)) != len(labels) or any(not label for label in labels):
            raise ValueError("Batch labels must be unique and non-empty")

    normalized: "OrderedDict[str, list[np.ndarray]]" = OrderedDict()
    for name, shape in expected.items():
        values: list[np.ndarray] = []
        for sample_index, raw_value in enumerate(entries[name]):
            value = np.ascontiguousarray(np.asarray(raw_value))
            if value.dtype != np.float32:
                raise ValueError(
                    f"{name}[{sample_index}] dtype is {value.dtype}, "
                    "expected float32"
                )
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name}[{sample_index}] shape is {value.shape}, "
                    f"expected {shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{name}[{sample_index}] is not finite")
            values.append(value)
        normalized[name] = values
    return normalized, labels


def cache_tensors(cache: Any, layer_index: int) -> tuple[Any, Any]:
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_index], cache.value_cache[layer_index]
    layer = cache.layers[layer_index]
    if hasattr(layer, "keys"):
        return layer.keys, layer.values
    return layer[0], layer[1]


def run_reference(
    entries: Mapping[str, Sequence[np.ndarray]],
    *,
    shard: str,
    model_dir: Path,
    device_name: str,
) -> tuple["OrderedDict[str, np.ndarray]", dict[str, Any]]:
    import torch
    import transformers
    from transformers import DynamicCache, Qwen3VLForConditionalGeneration

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    cuda_index = device.index if device.index is not None else 0
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.set_device(cuda_index)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    torch.set_grad_enabled(False)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    language_model = model.model.language_model
    config = language_model.config
    shard_config = FIXED_SHARDS[shard]
    layer_indices = list(
        range(
            int(shard_config["first_layer"]),
            int(shard_config["last_layer"]) + 1,
        )
    )
    layers = torch.nn.ModuleList(
        [language_model.layers[index] for index in layer_indices]
    )
    model.model.language_model = None
    del language_model, model
    layers.to(device)
    layers.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hidden_name = str(shard_config["hidden_input"])
    hidden_output = str(shard_config["hidden_output"])
    batch_count = len(entries[hidden_name])
    outputs_by_name: "OrderedDict[str, list[np.ndarray]]" = OrderedDict(
        [(hidden_output, [])]
    )
    for layer_index in layer_indices:
        outputs_by_name[f"past_key_{layer_index}_out"] = []
        outputs_by_name[f"past_value_{layer_index}_out"] = []

    with torch.inference_mode():
        for sample_index in range(batch_count):
            hidden = torch.from_numpy(entries[hidden_name][sample_index]).to(
                device=device,
                dtype=torch.bfloat16,
            )
            half_cos = torch.from_numpy(
                entries["position_ids_cos"][sample_index]
            ).to(device=device, dtype=torch.bfloat16)
            half_sin = torch.from_numpy(
                entries["position_ids_sin"][sample_index]
            ).to(device=device, dtype=torch.bfloat16)
            cos = torch.cat((half_cos[:, 0], half_cos[:, 0]), dim=-1)
            sin = torch.cat((half_sin[:, 0], half_sin[:, 0]), dim=-1)
            attention_mask = torch.from_numpy(
                entries["attention_mask"][sample_index]
            ).to(device=device, dtype=torch.bfloat16)

            cache = DynamicCache(config=config)
            for layer_index in layer_indices:
                key_qnn = torch.from_numpy(
                    entries[f"past_key_{layer_index}_in"][sample_index]
                ).to(device=device, dtype=torch.bfloat16)
                value_qnn = torch.from_numpy(
                    entries[f"past_value_{layer_index}_in"][sample_index]
                ).to(device=device, dtype=torch.bfloat16)
                cache.update(
                    key_qnn.permute(1, 0, 3, 2).contiguous(),
                    value_qnn.permute(1, 0, 2, 3).contiguous(),
                    layer_index,
                )

            for layer_index, layer in zip(layer_indices, layers, strict=True):
                hidden = layer(
                    hidden,
                    attention_mask=attention_mask,
                    position_embeddings=(cos, sin),
                    past_key_values=cache,
                    use_cache=True,
                )

            outputs_by_name[hidden_output].append(
                hidden.float().cpu().numpy().astype(np.float32, copy=False)
            )
            for layer_index in layer_indices:
                keys, values = cache_tensors(cache, layer_index)
                keys = keys[:, :, -PREFILL_AR:, :]
                values = values[:, :, -PREFILL_AR:, :]
                outputs_by_name[f"past_key_{layer_index}_out"].append(
                    keys.permute(1, 0, 3, 2)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                outputs_by_name[f"past_value_{layer_index}_out"].append(
                    values.permute(1, 0, 2, 3)
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            del cache, hidden

    outputs: "OrderedDict[str, np.ndarray]" = OrderedDict(
        (
            name,
            np.ascontiguousarray(np.stack(values, axis=0)),
        )
        for name, values in outputs_by_name.items()
    )
    runtime = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "max_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if device.type == "cuda"
            else None
        ),
    }
    del layers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, runtime


def weight_inventory(model_dir: Path, hash_weights: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        record: dict[str, Any] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
        }
        if hash_weights:
            record["sha256"] = sha256_file(path)
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
    return records


def write_npz_atomic(
    path: Path,
    outputs: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **outputs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--expected-source-model-id", required=True)
    parser.add_argument("--shard", choices=SUPPORTED_SHARDS, required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--batch-label", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="SHA-256 every model shard (extra local I/O)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_dir}")

    import qai_hub as hub

    source_job = hub.get_job(args.source_job_id)
    if status_code(source_job) != "SUCCESS":
        raise ValueError(f"Source job {args.source_job_id} is not successful")
    if source_job.model.model_id != args.expected_source_model_id:
        raise ValueError(
            f"Source job model is {source_job.model.model_id}, expected "
            f"{args.expected_source_model_id}"
        )
    expected_options = graph_option(
        str(FIXED_SHARDS[args.shard]["graph"])
    )
    if source_job.options != expected_options:
        raise ValueError(
            f"Source job options are {source_job.options!r}, expected "
            f"{expected_options!r}"
        )
    raw_entries = source_job.inputs.download()
    if not isinstance(raw_entries, dict):
        raise ValueError("AI Hub source dataset is not a tensor-entry mapping")
    entries, batch_labels = validate_dataset_entries(
        raw_entries,
        shard=args.shard,
        batch_labels=args.batch_label,
    )
    input_fingerprint = batch_fingerprint(entries)
    outputs, runtime = run_reference(
        entries,
        shard=args.shard,
        model_dir=model_dir,
        device_name=args.device,
    )

    output_path = args.output_npz.expanduser().resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {manifest_path}")
    write_npz_atomic(output_path, outputs)
    weights = weight_inventory(model_dir, args.hash_weights)
    manifest = {
        "schema_version": 1,
        "purpose": (
            f"Host BF16 reference for exact {args.shard.upper()} "
            "intermediate numerical screening"
        ),
        "created_at": utc_now(),
        "shard": {
            **FIXED_SHARDS[args.shard],
            "id": args.shard,
        },
        "input": {
            "source_job_id": source_job.job_id,
            "source_model_id": source_job.model.model_id,
            "dataset_id": source_job.inputs.dataset_id,
            "graph_options": source_job.options,
            "batch_labels": batch_labels,
            "batch_fingerprint_sha256": input_fingerprint,
            "tensors": {
                name: [
                    {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": sha256_array(value),
                    }
                    for value in values
                ]
                for name, values in entries.items()
            },
        },
        "model": {
            "path": str(model_dir),
            "config_sha256": sha256_file(model_dir / "config.json"),
            "weight_shards": weights,
        },
        "runtime": runtime,
        "output": {
            "path": output_path.name,
            "archive_sha256": sha256_file(output_path),
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": sha256_array(value),
                    "finite": bool(np.isfinite(value).all()),
                }
                for name, value in outputs.items()
            },
        },
        "limitations": [
            (
                "This is an eager BF16 decoder reference, not a bit-exact "
                "simulation of Qualcomm HTP kernels."
            ),
            (
                "The result contains intermediate hidden/KV tensors and "
                "does not establish generated-answer correctness."
            ),
            (
                "Fidelity comparisons are valid only for runs using the "
                "same source dataset ID and batch fingerprint."
            ),
        ],
        "security": {"contains_credentials": False},
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": True,
            "weight_hashes_complete": all(
                "sha256" in record for record in weights
            ),
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest": str(manifest_path),
                "output_sha256": manifest["output"]["archive_sha256"],
                "dataset_id": source_job.inputs.dataset_id,
                "batch_fingerprint_sha256": input_fingerprint,
                "runtime": runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
