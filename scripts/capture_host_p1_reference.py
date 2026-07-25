#!/usr/bin/env python3
"""Capture a host BF16 reference for one exact Cosmos P1 input artifact.

This executes text decoder layers 0--6 with the same precomputed video
embeddings, MRoPE, causal mask, DeepStack inputs, and synthetic initial KV
state supplied to the AI Hub P1 graph.  The output is an intermediate
hidden/KV reference for numerical screening; it is not a generated answer.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from scripts.build_p1_video_prefill import (
        HEAD_DIM,
        NUM_KV_HEADS,
        P1_FIRST_LAYER,
        P1_GRAPH_NAME,
        P1_LAST_LAYER,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        dataset_fingerprint,
        sha256_array,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_p1_video_prefill import (  # type: ignore[no-redef]
        HEAD_DIM,
        NUM_KV_HEADS,
        P1_FIRST_LAYER,
        P1_GRAPH_NAME,
        P1_LAST_LAYER,
        PREFILL_AR,
        PREFILL_KV_LENGTH,
        dataset_fingerprint,
        sha256_array,
        sha256_file,
    )

HIDDEN_OUTPUT_NAME = "add_15805"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_input_artifact(
    artifact_dir: Path,
) -> tuple["OrderedDict[str, np.ndarray]", dict[str, Any], Path]:
    artifact_dir = artifact_dir.expanduser().resolve()
    manifest_path = artifact_dir / "input_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("graph_name") != P1_GRAPH_NAME:
        raise ValueError(
            f"Expected graph {P1_GRAPH_NAME}, got {manifest.get('graph_name')}"
        )
    if int(manifest.get("chunk_index", -1)) != 0:
        raise ValueError("Host P1 reference currently supports chunk 0 only")
    archive_record = manifest.get("tensor_archive")
    if not isinstance(archive_record, dict):
        raise ValueError("Input manifest has no tensor_archive object")
    archive_path = artifact_dir / str(archive_record["path"])
    if sha256_file(archive_path) != str(archive_record["sha256"]):
        raise ValueError("Input tensor archive SHA does not match its manifest")
    order = manifest.get("input_order")
    if not isinstance(order, list):
        raise ValueError("Input manifest has no input_order")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(order) != set(archive.files):
            raise ValueError("Input tensor names do not match input_order")
        inputs = OrderedDict(
            (name, np.ascontiguousarray(archive[name])) for name in order
        )
    if dataset_fingerprint(inputs) != manifest.get(
        "dataset_fingerprint_sha256"
    ):
        raise ValueError("Input tensor dataset fingerprint does not match")
    return inputs, manifest, archive_path


def cache_tensors(cache: Any, layer_index: int) -> tuple[Any, Any]:
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_index], cache.value_cache[layer_index]
    layer = cache.layers[layer_index]
    if hasattr(layer, "keys"):
        return layer.keys, layer.values
    return layer[0], layer[1]


def validate_inputs(inputs: Mapping[str, np.ndarray]) -> None:
    expected_static = {
        "inputs_embeds": (1, PREFILL_AR, 2048),
        "position_ids_cos": (1, 1, PREFILL_AR, HEAD_DIM // 2),
        "position_ids_sin": (1, 1, PREFILL_AR, HEAD_DIM // 2),
        "attention_mask": (
            1,
            1,
            PREFILL_AR,
            PREFILL_KV_LENGTH + PREFILL_AR,
        ),
        "visual_pos_masks": (1, PREFILL_AR),
        "deepstack_visual_embeds_0": (256, 2048),
        "deepstack_visual_embeds_1": (256, 2048),
        "deepstack_visual_embeds_2": (256, 2048),
    }
    for layer_index in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1):
        expected_static[f"past_key_{layer_index}_in"] = (
            NUM_KV_HEADS,
            1,
            HEAD_DIM,
            PREFILL_KV_LENGTH,
        )
        expected_static[f"past_value_{layer_index}_in"] = (
            NUM_KV_HEADS,
            1,
            PREFILL_KV_LENGTH,
            HEAD_DIM,
        )
    missing = [name for name in expected_static if name not in inputs]
    if missing:
        raise ValueError(f"Missing P1 input tensors: {missing}")
    mismatched = [
        f"{name}: {np.asarray(inputs[name]).shape} != {shape}"
        for name, shape in expected_static.items()
        if tuple(np.asarray(inputs[name]).shape) != shape
    ]
    if mismatched:
        raise ValueError("P1 input shape mismatch:\n- " + "\n- ".join(mismatched))
    if np.any(inputs["visual_pos_masks"][:, :].sum(axis=1) > 256):
        raise ValueError("DeepStack input buffers are too short for visual mask")


def run_reference(
    inputs: Mapping[str, np.ndarray],
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
    model.to(device)
    model.eval()
    language_model = model.model.language_model
    language_model.eval()

    # Release vision and LM-head modules before the decoder pass.  The reference
    # uses already-frozen vision tensors and needs only layers 0--6.
    model.model.language_model = None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.inference_mode():
        hidden = torch.from_numpy(inputs["inputs_embeds"]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        half_cos = torch.from_numpy(inputs["position_ids_cos"]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        half_sin = torch.from_numpy(inputs["position_ids_sin"]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        # Qualcomm's _apply_rope_single consumes a half-width complex pair.
        # HF rotate_half is algebraically identical when each half is repeated.
        cos = torch.cat((half_cos[:, 0], half_cos[:, 0]), dim=-1)
        sin = torch.cat((half_sin[:, 0], half_sin[:, 0]), dim=-1)
        attention_mask = torch.from_numpy(inputs["attention_mask"]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        visual_mask = torch.from_numpy(inputs["visual_pos_masks"]).to(
            device=device,
            dtype=torch.bool,
        )
        deepstack = [
            torch.from_numpy(inputs[f"deepstack_visual_embeds_{index}"]).to(
                device=device,
                dtype=torch.bfloat16,
            )
            for index in range(3)
        ]

        cache = DynamicCache(config=language_model.config)
        for layer_index in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1):
            key_qnn = torch.from_numpy(
                inputs[f"past_key_{layer_index}_in"]
            ).to(device=device, dtype=torch.bfloat16)
            value_qnn = torch.from_numpy(
                inputs[f"past_value_{layer_index}_in"]
            ).to(device=device, dtype=torch.bfloat16)
            key_hf = key_qnn.permute(1, 0, 3, 2).contiguous()
            value_hf = value_qnn.permute(1, 0, 2, 3).contiguous()
            cache.update(key_hf, value_hf, layer_index)

        visual_count = int(visual_mask.sum().item())
        for layer_index in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1):
            hidden = language_model.layers[layer_index](
                hidden,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                past_key_values=cache,
                use_cache=True,
            )
            if layer_index < len(deepstack):
                hidden[visual_mask] += deepstack[layer_index][:visual_count]

        outputs: "OrderedDict[str, np.ndarray]" = OrderedDict()
        outputs[HIDDEN_OUTPUT_NAME] = (
            hidden.float().cpu().numpy().astype(np.float32, copy=False)
        )
        for layer_index in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1):
            keys, values = cache_tensors(cache, layer_index)
            keys = keys[:, :, -PREFILL_AR:, :]
            values = values[:, :, -PREFILL_AR:, :]
            outputs[f"past_key_{layer_index}_out"] = (
                keys.permute(1, 0, 3, 2)
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            outputs[f"past_value_{layer_index}_out"] = (
                values.permute(1, 0, 2, 3)
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
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
    del language_model, cache, hidden
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_artifact_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="SHA-256 all model shards (extra local I/O)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_dir}")
    inputs, input_manifest, input_archive = load_input_artifact(
        args.input_artifact_dir
    )
    validate_inputs(inputs)
    outputs, runtime = run_reference(inputs, model_dir, args.device)
    weights = weight_inventory(model_dir, args.hash_weights)

    output_path = args.output_npz.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    np.savez_compressed(output_path, **outputs)
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Host BF16 reference for exact P1 layers 0--6 numerical screening"
        ),
        "created_at": utc_now(),
        "graph_name": P1_GRAPH_NAME,
        "input": {
            "artifact_dir": str(args.input_artifact_dir.resolve()),
            "archive_path": str(input_archive),
            "archive_sha256": sha256_file(input_archive),
            "dataset_fingerprint_sha256": input_manifest[
                "dataset_fingerprint_sha256"
            ],
            "vision": input_manifest.get("source", {}).get("vision"),
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
                "P1 emits intermediate hidden/KV tensors and does not by "
                "itself establish generated-answer correctness."
            ),
            (
                "Fidelity comparisons are valid only for NPU runs using this "
                "same dataset fingerprint and vision-tensor source."
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
            "note": (
                "Publishable host evidence requires every weight shard SHA-256 "
                "and a path-sanitized derivative manifest."
            ),
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest": str(manifest_path),
                "output_sha256": manifest["output"]["archive_sha256"],
                "dataset_fingerprint_sha256": manifest["input"][
                    "dataset_fingerprint_sha256"
                ],
                "runtime": runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
