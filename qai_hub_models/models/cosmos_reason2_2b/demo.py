"""Run the standard QAI Hub Models VLM chat demo."""

from __future__ import annotations

import argparse
import sys

from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models._shared.qwen3_vl.model import DEFAULT_USER_PROMPT, END_TOKENS
from qai_hub_models.utils.checkpoint import CheckpointSpec

from . import MODEL_ID
from .model import (
    HF_REPO_NAME,
    HF_REPO_URL,
    Cosmos_Reason2_2B_PreSplit,
    Cosmos_Reason2_2B_QuantizablePreSplit,
    Cosmos_Reason2_2B_VisionEncoder,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    configure_source_checkpoint,
    resolve_source_checkpoint,
)


def _configure_demo_source_checkpoint(
    test_checkpoint: CheckpointSpec | None,
) -> str:
    checkpoint = test_checkpoint
    if checkpoint is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--checkpoint", default="DEFAULT")
        parsed, _ = parser.parse_known_args(sys.argv[1:])
        checkpoint = parsed.checkpoint
        if checkpoint == "DEFAULT" and any(
            arg in ("-h", "--help") for arg in sys.argv[1:]
        ):
            return HF_REPO_NAME

    assert checkpoint is not None
    source = resolve_source_checkpoint(checkpoint)
    return configure_source_checkpoint(source)


def cosmos_reason2_2b_chat_demo(
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    use_presplit = "--use-presplit" in sys.argv
    if use_presplit:
        sys.argv.remove("--use-presplit")

    source_checkpoint = _configure_demo_source_checkpoint(test_checkpoint)
    quantized_cls = (
        Cosmos_Reason2_2B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper
    )
    fp_cls = Cosmos_Reason2_2B_PreSplit if use_presplit else FPSplitModelWrapper

    llm_chat_demo(
        model_cls=quantized_cls,
        fp_model_cls=fp_cls,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        model_id=MODEL_ID,
        end_tokens=END_TOKENS,
        hf_repo_name=source_checkpoint,
        hf_repo_url=HF_REPO_URL,
        default_prompt=DEFAULT_USER_PROMPT,
        test_checkpoint=test_checkpoint,
        vision_encoder_cls=Cosmos_Reason2_2B_VisionEncoder,
    )


def main() -> None:
    cosmos_reason2_2b_chat_demo()


if __name__ == "__main__":
    main()
