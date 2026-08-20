from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Output is not empty: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a Cosmos Reason2 PEFT/LoRA adapter into its BF16 base "
            "checkpoint for deterministic serving and Qualcomm quantization."
        )
    )
    parser.add_argument("--model", required=True, help="Base HF checkpoint")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory")
    parser.add_argument("--output", required=True, help="Merged HF output directory")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map used while merging (default: auto)",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    adapter_path = Path(args.adapter).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for label, path in (("model", model_path), ("adapter", adapter_path)):
        if not path.is_dir():
            raise SystemExit(f"{label} directory does not exist: {path}")
    prepare_output(output_path, args.overwrite)

    print(f"loading_base={model_path}", flush=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    print(f"loading_adapter={adapter_path}", flush=True)
    adapted = PeftModel.from_pretrained(base, adapter_path)
    print("merging_adapter=safe", flush=True)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.eval()

    # The EVK path consumes a normal HF checkpoint. No PEFT runtime dispatch or
    # adapter sidecar is permitted after this point: quantization must observe
    # the actual merged weights that were accepted on the host.
    print(f"saving_merged={output_path}", flush=True)
    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.save_pretrained(output_path)

    adapter_files = sorted(
        path
        for path in adapter_path.iterdir()
        if path.is_file() and path.name in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "generative_training_metrics.json",
        }
    )
    weight_files = sorted(output_path.glob("*.safetensors"))
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": str(model_path),
        "adapter_checkpoint": str(adapter_path),
        "output_checkpoint": str(output_path),
        "dtype": "bfloat16",
        "safe_merge": True,
        "peft_runtime_required": False,
        "adapter_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in adapter_files
        },
        "merged_weight_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in weight_files
        },
        "merged_parameter_count": sum(
            parameter.numel() for parameter in merged.parameters()
        ),
    }
    (output_path / "adapter_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
