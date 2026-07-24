"""Quantize Cosmos-Reason2-2B text and vision towers to W4A16."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qai_hub_models.models._shared.vlm.quantize import quantize_vlm

from .model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    MODEL_ID,
    SAMPLE_IMAGE,
    SUPPORTED_PRECISIONS,
    Cosmos_Reason2_2B_PreSplit,
    Cosmos_Reason2_2B_QuantizablePreSplit,
    Cosmos_Reason2_2B_VisionEncoder,
    configure_source_checkpoint,
)


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


def main() -> None:
    _configure_local_checkpoint(sys.argv[1:])
    quantize_vlm(
        quantized_model_cls=Cosmos_Reason2_2B_QuantizablePreSplit,
        fp_model_cls=Cosmos_Reason2_2B_PreSplit,
        vision_encoder_cls=Cosmos_Reason2_2B_VisionEncoder,
        supported_precisions=SUPPORTED_PRECISIONS,
        description="Quantize Cosmos-Reason2-2B",
        model_id=MODEL_ID,
        sample_image=SAMPLE_IMAGE,
        default_image_height=DEFAULT_IMAGE_HEIGHT,
        default_image_width=DEFAULT_IMAGE_WIDTH,
    )


if __name__ == "__main__":
    main()
