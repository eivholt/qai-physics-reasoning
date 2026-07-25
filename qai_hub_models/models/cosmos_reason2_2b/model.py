"""QAI Hub Models adapter for NVIDIA Cosmos-Reason2-2B.

Cosmos-Reason2-2B is post-trained from Qwen3-VL-2B-Instruct and retains the
same network architecture. This module specializes Qualcomm's shared Qwen3-VL
implementation with the 2B dimensions and NVIDIA checkpoint location.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import SplitForwardMixin  # noqa: F401
from qai_hub_models.models._shared.qwen3_vl.model import (
    Qwen3VLCollectionBase,
    Qwen3VLPartBase,
    Qwen3VLPreSplitBase,
    Qwen3VLQuantizablePreSplitBase,
    Qwen3VLSplitForwardMixin,
    Qwen3VLVisionEncoderBase,
)
from qai_hub_models.models._shared.qwen3_vl.vision_encoder import (
    Qwen3VLVisionEncoder as SharedQwen3VLVisionEncoder,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

from .calibration import load_vision_calibration_data

logger = logging.getLogger(__name__)

# Keep the default export intentionally small for first-board bring-up. The
# full supported matrix is available through export.py's
# ``--full-context-matrix`` flag.
DEFAULT_EXPORT_CONTEXT_LENGTHS = [512]
FULL_EXPORT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Reuse Qualcomm's public calibration/demo image. It is not model-specific.
SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    "qwen3_vl_4b_instruct", 2, "dog.jpg"
)

# Qwen3-VL-2B text architecture.
NUM_LAYERS = 28
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 7
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 16
HEAD_DIM = 128
NUM_DEEPSTACK_LAYERS = 3

# Qwen3-VL-2B vision architecture.
VISION_HIDDEN_SIZE = 1024
VISION_OUT_HIDDEN_SIZE = 2048
VISION_DEPTH = 24
VISION_NUM_HEADS = 16
VISION_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2

HF_REPO_NAME = "nvidia/Cosmos-Reason2-2B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"
SOURCE_CHECKPOINT_ENV = "COSMOS_SOURCE_CHECKPOINT"
SOURCE_CHECKPOINT_MARKER = "source_checkpoint.json"
QAIRT245_COMPAT_MARKER = "qairt_245_compat.json"
W4_FP16_MARKER = "w4_fp16.json"
TEXT_W8_MARKER = "text_w8_matrices.json"

MIN_MEMORY_RECOMMENDED = 40

DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16, Precision.w4]
# This out-of-tree model has no checkpoint in Qualcomm's public asset store.
# Callers must supply a locally quantized checkpoint.
DEFAULT_CHECKPOINT: dict[Precision, str] = {}

DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_WIDTH = 512


def num_visual_tokens_for_image_size(image_size: tuple[int, int]) -> int:
    """Return the number of post-merge visual tokens for ``(width, height)``."""
    width, height = image_size
    return (
        (height // VISION_PATCH_SIZE)
        * (width // VISION_PATCH_SIZE)
        // (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE)
    )


DEFAULT_NUM_VISUAL_TOKENS = num_visual_tokens_for_image_size(
    (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
)

SPLIT_MODEL_NAME = "Cosmos_Reason2_2B"


def _last_vision_block_activation_quantizer_names(
    quant_sim: Any,
) -> tuple[str, ...]:
    """Find enabled activation quantizers produced by vision block 23.

    AIMET appends ``_qdq``/``_updated`` to graph inputs after inserting its
    custom quantizer nodes, while its quantizer dictionary retains the original
    tensor names. The learned parameters therefore provide stable structural
    boundaries: select from block 23's first through last parameterized
    operation. This deliberately leaves the block's final residual output and
    the downstream vision merger on A16.
    """

    def strip_quantizer_suffix(name: str) -> str:
        for suffix in ("_qdq", "_updated"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    try:
        nodes = [
            node
            for node in quant_sim.model.model.graph.node
            if node.op_type != "QcQuantizeOp"
        ]
        activation_names = set(quant_sim.activation_names)
        quantizers = quant_sim.qc_quantize_op_dict
    except AttributeError as exc:
        raise TypeError("Unexpected AIMET QuantSim graph structure") from exc

    block_prefix = f"blocks.{VISION_DEPTH - 1}."
    parameterized_node_indices = [
        index
        for index, node in enumerate(nodes)
        if any(
            strip_quantizer_suffix(value).startswith(block_prefix)
            for value in node.input
        )
    ]
    if not parameterized_node_indices:
        raise RuntimeError(
            f"Could not locate final vision transformer {block_prefix!r}"
        )
    block_start = min(parameterized_node_indices)
    block_end = max(parameterized_node_indices) + 1

    selected = tuple(
        sorted(
            {
                output
                for node in nodes[block_start:block_end]
                for output in node.output
                if output in activation_names
                and output in quantizers
                and bool(quantizers[output].enabled)
            }
        )
    )
    if not selected:
        raise RuntimeError(
            "No enabled activation quantizers were found in vision block 23"
        )
    return selected


def _set_last_vision_block_activations_float16(
    quant_sim: Any,
    *,
    float_data_type: Any | None = None,
) -> tuple[str, ...]:
    """Replace integer A16 inside the final vision block with FLOAT16."""
    if float_data_type is None:
        from aimet_onnx.common.defs import QuantizationDataType

        float_data_type = QuantizationDataType.float

    selected = _last_vision_block_activation_quantizer_names(quant_sim)
    for name in selected:
        quantizer = quant_sim.qc_quantize_op_dict[name]
        quantizer.reset_encoding_stats()
        quantizer.data_type = float_data_type
        quantizer.bitwidth = 16
    return selected


def _pad_prefill_kwargs_to_sequence_multiple(
    kwargs: dict[str, Any],
    sequence_length: int,
    pad_token_id: int,
) -> dict[str, Any]:
    """Left-pad a calibration prompt so every prefill slice has one shape.

    QAIHM 0.58 slices prompts in reverse order before applying its per-slice
    static padding. For a prompt whose length is not a multiple of the prefill
    sequence length, the first slice is short. Qwen3-VL's text inputs are then
    padded to the static sequence length, but its deepstack inputs retain the
    short slice length. The calibration mmap consequently locks the wrong
    visual-token shape and rejects the next full slice.

    Padding the complete prompt first preserves QAIHM's existing left-padding
    semantics while making every generated slice exactly ``sequence_length``.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    input_ids = kwargs.get("input_ids")
    attention_mask = kwargs.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        attention_mask, torch.Tensor
    ):
        return kwargs
    if input_ids.ndim == 0 or attention_mask.shape != input_ids.shape:
        return kwargs

    remainder = input_ids.shape[-1] % sequence_length
    if remainder == 0:
        return kwargs

    padding = sequence_length - remainder
    pad_shape = (*input_ids.shape[:-1], padding)
    padded = dict(kwargs)
    padded["input_ids"] = torch.cat(
        (
            torch.full(
                pad_shape,
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            input_ids,
        ),
        dim=-1,
    )
    padded["attention_mask"] = torch.cat(
        (
            torch.zeros(
                pad_shape,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
            attention_mask,
        ),
        dim=-1,
    )
    return padded


def _validate_source_checkpoint(source: Path) -> Path:
    """Validate a local, full-precision Hugging Face snapshot."""
    required = [
        source / "config.json",
        source / "preprocessor_config.json",
        source / "tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not any(path.is_file() for path in source.glob("*.safetensors")):
        missing.append(str(source / "*.safetensors"))
    if missing:
        raise ValueError(
            "Cosmos source checkpoint is incomplete. Missing: "
            + ", ".join(missing)
        )

    try:
        with (source / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Cosmos source config: {source / 'config.json'}"
        ) from exc

    text_config = config.get("text_config", {})
    vision_config = config.get("vision_config", {})
    expected = {
        "model_type": (config.get("model_type"), "qwen3_vl"),
        "text_config.hidden_size": (text_config.get("hidden_size"), HIDDEN_SIZE),
        "text_config.num_hidden_layers": (
            text_config.get("num_hidden_layers"),
            NUM_LAYERS,
        ),
        "text_config.num_attention_heads": (
            text_config.get("num_attention_heads"),
            NUM_ATTN_HEADS,
        ),
        "text_config.num_key_value_heads": (
            text_config.get("num_key_value_heads"),
            NUM_KEY_VALUE_HEADS,
        ),
        "vision_config.depth": (vision_config.get("depth"), VISION_DEPTH),
        "vision_config.hidden_size": (
            vision_config.get("hidden_size"),
            VISION_HIDDEN_SIZE,
        ),
    }
    mismatches = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "Cosmos source checkpoint has an incompatible architecture: "
            + "; ".join(mismatches)
        )
    return source


def resolve_source_checkpoint(checkpoint: str | os.PathLike[str]) -> Path:
    """Resolve the local BF16 snapshot needed while exporting a W4A16 checkpoint."""
    checkpoint_text = os.fspath(checkpoint)
    if checkpoint_text.startswith("DEFAULT"):
        raise ValueError(
            "Cosmos-Reason2-2B has no packaged default checkpoint. Pass a local "
            "BF16 or W4A16 directory with --checkpoint."
        )

    checkpoint_path = Path(checkpoint_text).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_path}")

    override = os.environ.get(SOURCE_CHECKPOINT_ENV, "").strip()
    if override:
        source = Path(override).expanduser().resolve()
    elif any(path.is_file() for path in checkpoint_path.glob("*.safetensors")):
        # A direct full-precision checkpoint is its own source.
        source = checkpoint_path
    else:
        marker = checkpoint_path / SOURCE_CHECKPOINT_MARKER
        if not marker.is_file():
            raise ValueError(
                f"Quantized checkpoint is missing {SOURCE_CHECKPOINT_MARKER}: "
                f"{checkpoint_path}. Re-run checkpoint finalization or set "
                f"{SOURCE_CHECKPOINT_ENV} to the local BF16 snapshot."
            )
        try:
            with marker.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            marker_source = payload["source_checkpoint"]
            if not isinstance(marker_source, str) or not marker_source.strip():
                raise TypeError("source_checkpoint must be a non-empty string")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid source checkpoint marker: {marker}") from exc
        source = Path(marker_source).expanduser().resolve()

    if not source.is_dir():
        raise ValueError(
            f"Cosmos source checkpoint does not exist: {source}. If the "
            f"quantized checkpoint was moved, set {SOURCE_CHECKPOINT_ENV} to "
            "the current local BF16 snapshot."
        )
    return _validate_source_checkpoint(source)


class Cosmos_Reason2_2B_PreSplit(Qwen3VLPreSplitBase):
    """Floating-point pre-split model."""

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    vision_patch_size = VISION_PATCH_SIZE
    spatial_merge_size = SPATIAL_MERGE_SIZE
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS
    _hf_repo_name = HF_REPO_NAME


class Cosmos_Reason2_2B_QuantizablePreSplit(
    Qwen3VLQuantizablePreSplitBase[Cosmos_Reason2_2B_PreSplit]
):
    """AIMET-ONNX quantizable pre-split model."""

    FPModel = Cosmos_Reason2_2B_PreSplit
    _hf_repo_name = HF_REPO_NAME

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    num_layers = NUM_LAYERS

    # q_norm per Q head + k_norm per KV head + post-attention layer norm.
    ada_scale_num_rmsnorm_per_blk: int | None = (
        NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1
    )

    @classmethod
    def fetch_default_checkpoint(cls, precision: Precision) -> str:
        raise ValueError(
            "Cosmos-Reason2-2B has no packaged default checkpoint. "
            "Quantize the official NVIDIA checkpoint locally, then pass its "
            "output directory with --checkpoint."
        )

    def _prefill_dataset(
        self,
        generator: Any,
        dataloader: Any,
        num_inputs: int,
        sample_to_kwargs: Callable[[Any, torch.device], dict[str, Any]],
        desc: str = "Pre-filling data",
    ) -> list[list[torch.Tensor | Any]]:
        """Normalize full prompts before QAIHM's reverse calibration slicing."""
        sequence_length = int(generator.sequence_length)
        # Match Generator.prepare_inputs(), which uses EOS for its static
        # left-padding tokens.
        pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        def fixed_shape_sample_to_kwargs(
            sample: Any, device: torch.device
        ) -> dict[str, Any]:
            kwargs = sample_to_kwargs(sample, device)
            return _pad_prefill_kwargs_to_sequence_multiple(
                kwargs,
                sequence_length=sequence_length,
                pad_token_id=int(pad_token_id),
            )

        return super()._prefill_dataset(
            generator,
            dataloader,
            num_inputs,
            fixed_shape_sample_to_kwargs,
            desc,
        )


class Cosmos_Reason2_2B_VisionEncoder(Qwen3VLVisionEncoderBase):
    """Qwen3-VL-2B vision embedding generator."""

    paired_calibration_manifest: Path | None = None
    fp16_last_block_activations = False
    last_fp16_activation_quantizer_names: tuple[str, ...] = ()
    DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    _hf_repo_name = HF_REPO_NAME
    vision_patch_size = VISION_PATCH_SIZE
    vision_hidden_size = VISION_HIDDEN_SIZE
    vision_num_heads = VISION_NUM_HEADS
    default_image_height = DEFAULT_IMAGE_HEIGHT
    default_image_width = DEFAULT_IMAGE_WIDTH
    quant_presplit_cls = Cosmos_Reason2_2B_QuantizablePreSplit

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike[str] | Path = "DEFAULT",
        device: torch.device | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        precision: Precision = Precision.float,
        **kwargs: Any,
    ) -> Any:
        """Recover a non-default vision geometry from checkpoint provenance."""
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if (
            checkpoint_path.is_dir()
            and (checkpoint_path / "args.json").is_file()
        ):
            try:
                quantize_args = json.loads(
                    (checkpoint_path / "args.json").read_text(encoding="utf-8")
                )
                recorded_size = quantize_args["image_size"]
                recorded_height, recorded_width = (
                    int(recorded_size[0]),
                    int(recorded_size[1]),
                )
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                IndexError,
            ) as exc:
                raise ValueError(
                    "Quantized vision checkpoint has no valid image_size in "
                    f"{checkpoint_path / 'args.json'}"
                ) from exc
            if (
                recorded_height <= 0
                or recorded_width <= 0
                or recorded_height % VISION_PATCH_SIZE
                or recorded_width % VISION_PATCH_SIZE
            ):
                raise ValueError(
                    "Quantized vision checkpoint image_size must contain "
                    f"positive multiples of {VISION_PATCH_SIZE}"
                )
            if (
                image_height is not None
                and int(image_height) != recorded_height
            ):
                raise ValueError(
                    "Explicit image_height does not match the quantized "
                    f"vision checkpoint: {image_height} != {recorded_height}"
                )
            if image_width is not None and int(image_width) != recorded_width:
                raise ValueError(
                    "Explicit image_width does not match the quantized "
                    f"vision checkpoint: {image_width} != {recorded_width}"
                )
            image_height = (
                recorded_height if image_height is None else image_height
            )
            image_width = recorded_width if image_width is None else image_width
        return super().from_pretrained(
            checkpoint=checkpoint,
            device=device,
            image_height=image_height,
            image_width=image_width,
            precision=precision,
            **kwargs,
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Preserve Qwen3-VL's learned temporal-patch projection bias.

        QAIHM 0.58 reuses Qwen2.5-VL's ``Conv2dInplaceConv3d`` adaptation,
        which constructs its replacement convolution with ``bias=False``.
        Qwen3-VL's patch projection has a learned bias, so retain it across
        that otherwise equivalent weight-layout conversion.
        """
        visual = kwargs.get("visual", args[0] if args else None)
        projection = getattr(getattr(visual, "patch_embed", None), "proj", None)
        source_bias = getattr(projection, "bias", None)
        preserved_bias = (
            source_bias.detach().clone()
            if isinstance(source_bias, torch.Tensor)
            else None
        )

        super().__init__(*args, **kwargs)
        _restore_temporal_patch_projection_bias(
            self.patch_embed.proj, preserved_bias
        )

    def get_input_spec(
        self,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> Any:
        """Use the constructed graph geometry instead of class defaults."""
        if image_height is None:
            image_height = int(self._image_height)
        if image_width is None:
            image_width = int(self._image_width)
        rope_dim = int(self._pos_emb_cos.shape[-1])
        return SharedQwen3VLVisionEncoder.get_static_input_spec(
            image_height,
            image_width,
            patch_size=int(self._patch_size),
            rope_dim=rope_dim,
        )

    @classmethod
    def get_calibration_data(
        cls,
        num_samples: int,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
    ) -> list[Any]:
        """Load balanced stills or an opt-in local paired-frame manifest."""
        return load_vision_calibration_data(
            hf_repo=cls._hf_repo_name,
            num_samples=num_samples,
            image_height=image_height,
            image_width=image_width,
            paired_manifest=cls.paired_calibration_manifest,
        )

    @classmethod
    def create_quantsim(
        cls,
        veg_model: Any,
        host_device: torch.device,
    ) -> tuple[Any, dict[str, Any]]:
        """Optionally keep the final vision block's activations in FP16."""
        quant_sim, fixed_inputs = super().create_quantsim(
            veg_model,
            host_device,
        )
        cls.last_fp16_activation_quantizer_names = ()
        if not cls.fp16_last_block_activations:
            return quant_sim, fixed_inputs

        # This is the same AIMET representation used by QAIHM's W4/FP16 path:
        # the quantizer remains present in the exported encoding contract, but
        # integer A16 quantize/dequantize is replaced by native FLOAT16.
        selected = _set_last_vision_block_activations_float16(quant_sim)
        cls.last_fp16_activation_quantizer_names = selected
        print(
            "  Using FLOAT16 activation quantizers for final vision block "
            f"{VISION_DEPTH - 1}: {len(selected)} tensors"
        )
        return quant_sim, fixed_inputs


def _restore_temporal_patch_projection_bias(
    adapted_projection: torch.nn.Module,
    source_bias: torch.Tensor | None,
) -> None:
    """Attach the source Conv3d bias to its Conv2d patch adaptation."""
    if source_bias is None:
        return
    conv2d = getattr(adapted_projection, "conv2d", None)
    if not isinstance(conv2d, torch.nn.Conv2d):
        raise TypeError(
            "Temporal patch projection adaptation has no Conv2d module"
        )
    expected_shape = (conv2d.out_channels,)
    if tuple(source_bias.shape) != expected_shape:
        raise ValueError(
            "Temporal patch projection bias has shape "
            f"{tuple(source_bias.shape)}, expected {expected_shape}"
        )
    conv2d.bias = torch.nn.Parameter(
        source_bias.to(
            device=conv2d.weight.device,
            dtype=conv2d.weight.dtype,
        ),
        requires_grad=source_bias.requires_grad,
    )


class Cosmos_Reason2_2B_PartBase(Qwen3VLPartBase):
    """Base for the four text-decoder partitions."""

    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    num_splits = NUM_SPLITS
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    default_precision = DEFAULT_PRECISION
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS
    fp_presplit_cls = Cosmos_Reason2_2B_PreSplit
    quant_presplit_cls = Cosmos_Reason2_2B_QuantizablePreSplit
    export_sequence_lengths = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    export_context_lengths = DEFAULT_EXPORT_CONTEXT_LENGTHS


class Cosmos_Reason2_2B_Part1_Of_4(Cosmos_Reason2_2B_PartBase):
    part_id = 1


class Cosmos_Reason2_2B_Part2_Of_4(Cosmos_Reason2_2B_PartBase):
    part_id = 2


class Cosmos_Reason2_2B_Part3_Of_4(Cosmos_Reason2_2B_PartBase):
    part_id = 3


class Cosmos_Reason2_2B_Part4_Of_4(Cosmos_Reason2_2B_PartBase):
    part_id = 4


_SPLIT_PART_CLASSES: list[type] = [
    Cosmos_Reason2_2B_Part1_Of_4,
    Cosmos_Reason2_2B_Part2_Of_4,
    Cosmos_Reason2_2B_Part3_Of_4,
    Cosmos_Reason2_2B_Part4_Of_4,
]


class FPSplitModelWrapper(Qwen3VLSplitForwardMixin, Cosmos_Reason2_2B_PreSplit):
    """Floating-point evaluation through the split ONNX parts."""

    split_part_classes = _SPLIT_PART_CLASSES
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS


class QuantizedSplitModelWrapper(
    Qwen3VLSplitForwardMixin, Cosmos_Reason2_2B_QuantizablePreSplit
):
    """Quantized evaluation through the split ONNX parts."""

    split_part_classes = _SPLIT_PART_CLASSES
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS


def configure_source_checkpoint(source: Path) -> str:
    """Route every QAIHM Hugging Face lookup to one validated local snapshot."""
    source_name = str(source)
    Cosmos_Reason2_2B_PreSplit._hf_repo_name = source_name
    Cosmos_Reason2_2B_QuantizablePreSplit._hf_repo_name = source_name
    Cosmos_Reason2_2B_VisionEncoder._hf_repo_name = source_name
    Cosmos_Reason2_2B_Collection._hf_repo_name = source_name
    return source_name


class Cosmos_Reason2_2B_Collection(Qwen3VLCollectionBase):
    """Four text partitions plus one vision encoder."""

    _hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Cosmos_Reason2_2B_PreSplit
    quant_presplit_cls = Cosmos_Reason2_2B_QuantizablePreSplit
    part_base_cls = Cosmos_Reason2_2B_PartBase
    vision_encoder_cls = Cosmos_Reason2_2B_VisionEncoder
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    vision_patch_size = VISION_PATCH_SIZE
    default_image_height = DEFAULT_IMAGE_HEIGHT
    default_image_width = DEFAULT_IMAGE_WIDTH
    default_precision = DEFAULT_PRECISION
    sample_image = SAMPLE_IMAGE
    parts = {
        "part1_of_4": Cosmos_Reason2_2B_Part1_Of_4,
        "part2_of_4": Cosmos_Reason2_2B_Part2_Of_4,
        "part3_of_4": Cosmos_Reason2_2B_Part3_Of_4,
        "part4_of_4": Cosmos_Reason2_2B_Part4_Of_4,
    }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device=None,
        **kwargs,
    ):
        source = resolve_source_checkpoint(checkpoint)
        configure_source_checkpoint(source)
        instance = super().from_pretrained(
            checkpoint=checkpoint,
            host_device=host_device,
            **kwargs,
        )
        # The compatibility checkpoint replaces optional deepstack graph
        # inputs with internal zeros. Suppress the unsupported Genie wildcard
        # connector in its generated vision pipeline assets as well.
        if (Path(checkpoint).expanduser() / QAIRT245_COMPAT_MARKER).is_file():
            instance.num_deepstack_layers = 0
        return instance

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike[str],
        metadata: Any,
    ) -> None:
        super().write_supplementary_files(output_dir, metadata)
        markers = {
            QAIRT245_COMPAT_MARKER: (
                "QAIRT 2.45 compatibility transform provenance."
            ),
            W4_FP16_MARKER: (
                "W4 text weights and FLOAT16 activation transform provenance."
            ),
            TEXT_W8_MARKER: (
                "Selected text matrices independently promoted from W4 to "
                "W8 provenance."
            ),
        }
        for filename, description in markers.items():
            marker = Path(self._checkpoint) / filename
            if marker.is_file():
                shutil.copy2(marker, Path(output_dir) / filename)
                metadata.supplementary_files[filename] = description
