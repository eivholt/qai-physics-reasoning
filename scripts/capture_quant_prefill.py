#!/usr/bin/env python3
"""Capture the first AR-128 QuantSim call for direct QNN comparison.

This diagnostic runs one host token through the normal split-model demo and
writes each prefill part's float inputs/outputs as raw tensors. The input
manifest can then drive ``qnn-net-run`` against the compiled EVK contexts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qai_hub_models.models._shared.qwen3_vl.model import Qwen3VLPartBase

from qai_hub_models.models.cosmos_reason2_2b.demo import (
    cosmos_reason2_2b_chat_demo,
)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _tensor_record(tensor: torch.Tensor, path: Path) -> dict[str, Any]:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    array.tofile(path)
    finite = np.isfinite(array)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "source_dtype": str(tensor.dtype),
        "stored_dtype": "float32",
        "finite": bool(finite.all()),
        "min": float(array[finite].min()) if finite.any() else None,
        "max": float(array[finite].max()) if finite.any() else None,
        "mean": float(array[finite].mean()) if finite.any() else None,
    }


def capture_prefill(
    checkpoint: Path,
    output_dir: Path,
    *,
    prompt: str,
) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    original_forward = Qwen3VLPartBase.forward
    captures: dict[str, Any] = {}

    def traced_forward(
        self: Qwen3VLPartBase,
        *args: torch.Tensor,
        **kwargs: Any,
    ):
        input_names = self._get_onnx_input_names()
        if kwargs:
            raise ValueError(
                f"Unexpected keyword inputs while tracing part {self.part_id}: "
                f"{sorted(kwargs)}"
            )
        if len(input_names) != len(args):
            raise ValueError(
                f"Part {self.part_id} has {len(input_names)} ONNX inputs but "
                f"received {len(args)} tensors"
            )

        attention_index = input_names.index("attention_mask")
        sequence_length = int(args[attention_index].shape[2])
        result = original_forward(self, *args, **kwargs)

        key = f"part{self.part_id}_of_{self.num_splits}"
        if sequence_length != 128 or key in captures:
            return result

        part_dir = output_dir / key
        input_dir = part_dir / "inputs"
        output_tensor_dir = part_dir / "outputs"
        input_dir.mkdir(parents=True)
        output_tensor_dir.mkdir()

        input_records: dict[str, Any] = {}
        input_list_items: list[str] = []
        for name, tensor in zip(input_names, args, strict=True):
            filename = _safe_filename(name) + ".raw"
            input_records[name] = _tensor_record(tensor, input_dir / filename)
            input_list_items.append(f"{name}:=inputs/{filename}")

        output_names = self._get_onnx_output_names()
        output_tensors = list(result)
        if len(output_names) != len(output_tensors):
            raise ValueError(
                f"Part {self.part_id} has {len(output_names)} ONNX outputs but "
                f"returned {len(output_tensors)} tensors"
            )
        output_records: dict[str, Any] = {}
        for name, tensor in zip(output_names, output_tensors, strict=True):
            filename = _safe_filename(name) + ".raw"
            output_records[name] = _tensor_record(
                tensor,
                output_tensor_dir / filename,
            )

        (part_dir / "input_list.txt").write_text(
            " ".join(input_list_items) + "\n",
            encoding="utf-8",
        )
        captures[key] = {
            "part_id": self.part_id,
            "sequence_length": sequence_length,
            "input_order": input_names,
            "output_order": output_names,
            "inputs": input_records,
            "outputs": output_records,
        }
        return result

    Qwen3VLPartBase.forward = traced_forward
    old_argv = sys.argv
    try:
        sys.argv = [
            "capture_quant_prefill.py",
            "--checkpoint",
            str(checkpoint),
            "--sequence-length",
            "1",
            "128",
            "--context-length",
            "512",
            "--max-output-tokens",
            "1",
            "--prompt",
            prompt,
        ]
        cosmos_reason2_2b_chat_demo()
    finally:
        sys.argv = old_argv
        Qwen3VLPartBase.forward = original_forward

    expected = {f"part{part}_of_4" for part in range(1, 5)}
    if set(captures) != expected:
        raise RuntimeError(
            f"Captured prefill parts {sorted(captures)}; expected {sorted(expected)}"
        )
    (output_dir / "capture.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "prompt": prompt,
                "parts": captures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Captured AR-128 QuantSim tensors: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--prompt",
        default=(
            "A ball is released from rest. In one short sentence, what "
            "happens next and why?"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    capture_prefill(
        args.checkpoint,
        args.output_dir,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
