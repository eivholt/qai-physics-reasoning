"""Quantize Cosmos-Reason2-2B text and vision towers to W4A16."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qai_hub_models.models._shared.vlm.quantize import quantize_vlm

from .calibration import resolve_paired_frame_paths
from .model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    MODEL_ID,
    SAMPLE_IMAGE,
    SUPPORTED_PRECISIONS,
    VISION_DEPTH,
    Cosmos_Reason2_2B_PreSplit,
    Cosmos_Reason2_2B_QuantizablePreSplit,
    Cosmos_Reason2_2B_VisionEncoder,
    configure_source_checkpoint,
)

PAIRED_CALIBRATION_FLAG = "--veg-paired-calibration-manifest"
FP16_LAST_BLOCK_ACTIVATIONS_FLAG = "--veg-fp16-last-block-activations"


def _configure_local_checkpoint(argv: list[str]) -> None:
    """Route every shared QAIHM loader to the supplied local checkpoint."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint")
    parsed, _ = parser.parse_known_args(argv)

    if parsed.checkpoint is None:
        if any(arg in ("-h", "--help") for arg in argv):
            return
        parser.error(
            "--checkpoint LOCAL_DIR is required; this adapter has no packaged "
            "default checkpoint"
        )

    checkpoint = Path(parsed.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error(f"--checkpoint must be a local directory: {checkpoint}")

    # QAI Hub Models 0.58 resolves model configuration through the concrete
    # class's _hf_repo_name during both text QuantSim construction and VEG
    # loading, even when ``--checkpoint`` was supplied to the text quantizer.
    configure_source_checkpoint(checkpoint)


def _extract_vision_calibration_args(
    argv: list[str],
) -> tuple[list[str], Path | None]:
    """Remove this adapter's local-only calibration option for shared QAIHM."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(PAIRED_CALIBRATION_FLAG, type=Path)
    parsed, remaining = parser.parse_known_args(argv)
    manifest = parsed.veg_paired_calibration_manifest
    if manifest is None:
        return remaining, None
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        parser.error(f"paired calibration manifest does not exist: {manifest}")
    return remaining, manifest


def _extract_vision_mixed_precision_args(
    argv: list[str],
) -> tuple[list[str], bool]:
    """Remove the local final-block mixed-precision option for shared QAIHM."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        FP16_LAST_BLOCK_ACTIVATIONS_FLAG,
        action="store_true",
    )
    parsed, remaining = parser.parse_known_args(argv)
    return remaining, bool(parsed.veg_fp16_last_block_activations)


def _output_dir_from_args(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parsed, _ = parser.parse_known_args(argv)
    return parsed.output_dir.expanduser().resolve()


def _record_calibration_provenance(
    output_dir: Path,
    original_argv: list[str],
    paired_manifest: Path | None,
    fp16_last_block_activations: bool = False,
    fp16_activation_quantizer_names: tuple[str, ...] = (),
) -> None:
    args_path = output_dir / "args.json"
    try:
        payload = json.loads(args_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Quantization did not produce a readable args.json: {args_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Quantization args are not a JSON object: {args_path}")
    geometry_parser = argparse.ArgumentParser(add_help=False)
    geometry_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH),
    )
    geometry, _ = geometry_parser.parse_known_args(original_argv)
    payload["image_size"] = list(geometry.image_size)
    payload["raw_args"] = original_argv
    payload["vision_calibration_source"] = (
        "paired_frame_manifest"
        if paired_manifest is not None
        else "imagenette_class_balanced"
    )
    if paired_manifest is not None:
        payload["veg_paired_calibration_manifest"] = str(paired_manifest)
        payload["veg_paired_calibration_manifest_sha256"] = hashlib.sha256(
            paired_manifest.read_bytes()
        ).hexdigest()
        manifest_payload = json.loads(paired_manifest.read_text(encoding="utf-8"))
        payload["vision_calibration_temporal_mode"] = manifest_payload.get(
            "temporal_mode", "distinct_frames"
        )
    if fp16_last_block_activations:
        if not fp16_activation_quantizer_names:
            raise RuntimeError(
                "Final-block FP16 was requested but no activation quantizers "
                "were recorded"
            )
        payload["vision_mixed_precision"] = {
            "activation_quantization": "A16 integer",
            "fp16_activation_blocks": [VISION_DEPTH - 1],
            "fp16_activation_quantizer_count": len(
                fp16_activation_quantizer_names
            ),
            "fp16_activation_quantizer_names": list(
                fp16_activation_quantizer_names
            ),
            "parameter_quantization": "W8 integer",
            "scheme": "W8A16 with final vision block FLOAT16 activations",
        }
    temporary = args_path.with_name(f".{args_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args_path)


def main() -> None:
    original_sys_argv = list(sys.argv)
    original_argv = original_sys_argv[1:]
    _configure_local_checkpoint(original_argv)
    shared_argv, paired_manifest = _extract_vision_calibration_args(
        original_argv
    )
    shared_argv, fp16_last_block_activations = (
        _extract_vision_mixed_precision_args(shared_argv)
    )
    showing_help = any(value in ("-h", "--help") for value in shared_argv)
    output_dir = (
        None if showing_help else _output_dir_from_args(shared_argv)
    )
    if showing_help:
        print(
            "\nCosmos-specific option:\n"
            f"  {PAIRED_CALIBRATION_FLAG} PATH\n"
            "                        Calibrate the vision encoder from an "
            "explicit local JSON manifest of distinct frame pairs.\n"
            f"  {FP16_LAST_BLOCK_ACTIVATIONS_FLAG}\n"
            "                        Keep final vision transformer block "
            f"{VISION_DEPTH - 1} activations in FLOAT16 while retaining "
            "W8 weights and A16 elsewhere.\n"
        )
    Cosmos_Reason2_2B_VisionEncoder.paired_calibration_manifest = (
        paired_manifest
    )
    Cosmos_Reason2_2B_VisionEncoder.fp16_last_block_activations = (
        fp16_last_block_activations
    )
    demo_sample_image = (
        str(resolve_paired_frame_paths(paired_manifest, 1)[0][0])
        if paired_manifest is not None
        else SAMPLE_IMAGE
    )
    sys.argv = [original_sys_argv[0], *shared_argv]
    try:
        quantize_vlm(
            quantized_model_cls=Cosmos_Reason2_2B_QuantizablePreSplit,
            fp_model_cls=Cosmos_Reason2_2B_PreSplit,
            vision_encoder_cls=Cosmos_Reason2_2B_VisionEncoder,
            supported_precisions=SUPPORTED_PRECISIONS,
            description="Quantize Cosmos-Reason2-2B",
            model_id=MODEL_ID,
            sample_image=demo_sample_image,
            default_image_height=DEFAULT_IMAGE_HEIGHT,
            default_image_width=DEFAULT_IMAGE_WIDTH,
        )
        if output_dir is not None:
            _record_calibration_provenance(
                output_dir,
                original_argv,
                paired_manifest,
                fp16_last_block_activations,
                Cosmos_Reason2_2B_VisionEncoder
                .last_fp16_activation_quantizer_names,
            )
    finally:
        Cosmos_Reason2_2B_VisionEncoder.paired_calibration_manifest = None
        Cosmos_Reason2_2B_VisionEncoder.fp16_last_block_activations = False
        Cosmos_Reason2_2B_VisionEncoder.last_fp16_activation_quantizer_names = (
            ()
        )
        sys.argv = original_sys_argv


if __name__ == "__main__":
    main()
