"""Cosmos-Reason2-2B model adapter for Qualcomm AI Hub Models."""

from qai_hub_models.models._shared.qwen3_vl.model import (
    Qwen3VLPositionProcessor as PositionProcessor,
)

from .model import (
    DEFAULT_PRECISION,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MIN_MEMORY_RECOMMENDED,
    MODEL_ID,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    Cosmos_Reason2_2B_Collection,
    Cosmos_Reason2_2B_Part1_Of_4,
    Cosmos_Reason2_2B_Part2_Of_4,
    Cosmos_Reason2_2B_Part3_Of_4,
    Cosmos_Reason2_2B_Part4_Of_4,
    Cosmos_Reason2_2B_PartBase,
    Cosmos_Reason2_2B_PreSplit,
    Cosmos_Reason2_2B_QuantizablePreSplit,
    Cosmos_Reason2_2B_VisionEncoder,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
)

VisionEncoder = Cosmos_Reason2_2B_VisionEncoder
Model = Cosmos_Reason2_2B_Collection

__all__ = [
    "DEFAULT_PRECISION",
    "HF_REPO_NAME",
    "HIDDEN_SIZE",
    "MIN_MEMORY_RECOMMENDED",
    "MODEL_ID",
    "NUM_ATTN_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "NUM_LAYERS",
    "NUM_LAYERS_PER_SPLIT",
    "NUM_SPLITS",
    "Cosmos_Reason2_2B_Collection",
    "Cosmos_Reason2_2B_Part1_Of_4",
    "Cosmos_Reason2_2B_Part2_Of_4",
    "Cosmos_Reason2_2B_Part3_Of_4",
    "Cosmos_Reason2_2B_Part4_Of_4",
    "Cosmos_Reason2_2B_PartBase",
    "Cosmos_Reason2_2B_PreSplit",
    "Cosmos_Reason2_2B_QuantizablePreSplit",
    "Cosmos_Reason2_2B_VisionEncoder",
    "FPSplitModelWrapper",
    "Model",
    "PositionProcessor",
    "QuantizedSplitModelWrapper",
    "VisionEncoder",
]
