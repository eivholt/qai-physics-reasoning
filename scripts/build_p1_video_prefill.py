#!/usr/bin/env python3
"""Build exact GenieX AR128/P1 inputs for a frozen Cosmos video prompt.

This intentionally mirrors the pinned GenieX Qwen3-VL input providers rather
than the generic QAI Hub Models generator.  The important differences are:

* video features replace ``<|video_pad|>`` embeddings;
* Qwen3-VL uses interleaved three-axis MRoPE;
* the final short prefill chunk is right-padded with the EOS embedding;
* unused RoPE rows are identity rotations;
* visual masks are right-padded and DeepStack rows advance by the number of
  visual tokens already consumed; and
* only the initial chunk may use an all-zero KV cache.

The generated NumPy archive is suitable for one Qualcomm AI Hub inference
sample.  It contains no credentials and records a byte-level provenance
manifest alongside the tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_PAD_TOKEN_ID = 151655
VIDEO_PAD_TOKEN_ID = 151656

PREFILL_AR = 128
CONTEXT_LENGTH = 512
PREFILL_KV_LENGTH = CONTEXT_LENGTH - PREFILL_AR
NUM_KV_HEADS = 8
HEAD_DIM = 128
P1_FIRST_LAYER = 0
P1_LAST_LAYER = 6

P1_GRAPH_NAME = "ar128_cl512_1_of_4"

P1_TARGETS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("baseline_w4_fp16", "mnj67lgdq"),
        ("w8_layers_0_6", "mn7yvo34n"),
        ("w8_layers_0_3", "mm55go5km"),
        ("w8_layers_0_2", "mnzgp4vxq"),
        ("w8_layers_1_3", "mqe41p2vm"),
    ]
)


GOLDEN_PROFILES: dict[str, dict[str, Any]] = {
    "barrier_normal_bf16_r1": {
        "description": (
            "Barrier-knockdown four-scene normal-order prompt with the saved "
            "Cosmos BF16 vision/DeepStack reference."
        ),
        "video_manifest_sha256": (
            "c3c52519734f3b9e04312c201c348deace0399972c03931e3c427ceeeb6bf7dc"
        ),
        "input_ids_int64_sha256": (
            "950d7ac13d4b2717ec391dadcef90043ee1611af0b978eb7a09641d03bf7ff63"
        ),
        "prompt_tokens": 360,
        "chunks": [
            {
                "valid_tokens": 128,
                "visual_tokens": 97,
                "inputs_embeds": (
                    "8ee57cffdc619ca59766d4035a66ad70ca44f7d8ddaed22bc7a2cc9ae4968f5a"
                ),
                "position_ids_cos": (
                    "e5ca8560fa2d35641136cce28a439909c6e303dda5172c3850f287a778d02731"
                ),
                "position_ids_sin": (
                    "dc92dd36f53065441fa2b4c15033d9fa9c983e812459d7351428455057485ac2"
                ),
                "attention_mask": (
                    "57a18d760153cf23d075420408d0c8fef3ec41eb69e23fee47d9418a10a5d929"
                ),
                "visual_pos_masks": (
                    "f405f00611f2f4549a950a37ea99a4b084344d343f9ba822cd0adc75fde0aacb"
                ),
                "deepstack_visual_embeds_0": (
                    "399b7a42be51cd4ff45b3cbbd8c2846021d28670e0ccc471e4bda72744150ac0"
                ),
                "deepstack_visual_embeds_1": (
                    "a00c09fa66045c3dcd79348a2695831edd1a5562174205a895d9dbe9b82d4669"
                ),
                "deepstack_visual_embeds_2": (
                    "f922439bba55d7a580fffc796a35a99948a28a81cbdb1e08d617ce5284348c02"
                ),
            },
            {
                "valid_tokens": 128,
                "visual_tokens": 119,
                "inputs_embeds": (
                    "623f388a3e3e94020c1f51be6c74de0baf69e4f4bdc79bac5ed774926a029a29"
                ),
                "position_ids_cos": (
                    "7747c57df15bb5415ad7a86be9e10ae65a0a1fe73f029474e4ea280b38e672d1"
                ),
                "position_ids_sin": (
                    "b6736ba62783e5e95b8d2f42358714d3d04570d813f12e21b272a70947148c66"
                ),
                "attention_mask": (
                    "9250747b3c0d04b85dfaa3a00bcb7a0f4f5888805d5ae9882b44a5f2523c779f"
                ),
                "visual_pos_masks": (
                    "ee44b920b136b84943883586b055d3bff541460516fc4c2db1786f3c91ee4c8b"
                ),
                "deepstack_visual_embeds_0": (
                    "745c65448be05e7ccf04a8764ff2ef2b8814a07107f7e30e2b2cb3210625b25c"
                ),
                "deepstack_visual_embeds_1": (
                    "aa287740ac1cf38ddd6aab159b932a3a1835344c05b086ba2b947a813b5f2fef"
                ),
                "deepstack_visual_embeds_2": (
                    "f9e925ff258f25c895ccfd482919ae34856266e7979da3924838e1581d834a8f"
                ),
            },
            {
                "valid_tokens": 104,
                "visual_tokens": 36,
                "inputs_embeds": (
                    "0d61159ce2929c325d8bd8dbec065e8ea214a95abc4e162d363dc32e250988da"
                ),
                "position_ids_cos": (
                    "24fa54a6ddc5b02cae6f2af9510174e666eeacf977ca5bfc8470f298a8d78584"
                ),
                "position_ids_sin": (
                    "a74702f03f1314a92333788f5ddf718301e732857974e243033546a3951de971"
                ),
                "attention_mask": (
                    "523d4845a78845e5209d062c524ed69f52de5a8f6b2ccf212c09c2c679ea8b56"
                ),
                "visual_pos_masks": (
                    "5a675750f9a5d3672ae0717a3bec378b3cca62596bd56f4d3e767af703b12b2a"
                ),
                "deepstack_visual_embeds_0": (
                    "22f4e1cf8ebc7b6839549984a231ec22253eee32da92cf704e219554d5dc7391"
                ),
                "deepstack_visual_embeds_1": (
                    "6fd4b7a7dcf2994809f3a70550802923105f70f2cc3e3523a8e04cdcc6102c8f"
                ),
                "deepstack_visual_embeds_2": (
                    "fdcc62b20441b540c0f499c625e091bcb61f1cc2976f482cea7b30f629225fa6"
                ),
            },
        ],
    }
}


@dataclass(frozen=True)
class BundleContract:
    vocab_size: int
    hidden_size: int
    eos_token_id: int
    rope_theta: float
    mrope_section: tuple[int, ...]
    spatial_merge_size: int
    embedding_path: Path
    tokenizer_path: Path


@dataclass
class PreparedPrompt:
    input_ids: np.ndarray
    positions: np.ndarray
    chunks: list["OrderedDict[str, np.ndarray]"]
    chunk_valid_tokens: list[int]
    chunk_visual_tokens: list[int]
    source_record: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_path(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def load_bundle_contract(bundle_dir: Path) -> BundleContract:
    bundle_dir = bundle_dir.expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    config_path = _require_path(bundle_dir / "genie_config.json", "Genie config")
    config = _read_json(config_path)
    dialog = config.get("dialog") or config.get("text-generator")
    if not isinstance(dialog, dict):
        raise ValueError("genie_config.json has no dialog/text-generator object")

    context = dialog.get("context")
    embedding = dialog.get("embedding")
    engine = dialog.get("engine")
    if not all(isinstance(value, dict) for value in (context, embedding, engine)):
        raise ValueError("Incomplete context/embedding/engine config")

    model = engine.get("model")
    if not isinstance(model, dict):
        raise ValueError("Missing engine.model config")
    positional = model.get("positional-encoding")
    if not isinstance(positional, dict):
        raise ValueError("Missing positional-encoding config")
    scaling = positional.get("rope-scaling")
    if not isinstance(scaling, dict):
        raise ValueError("Missing rope-scaling config")

    section = tuple(int(value) for value in scaling.get("mrope-section", []))
    if not section or len(section) != 3:
        raise ValueError(f"Expected three MRoPE sections, got {section!r}")
    rope_dim = int(positional.get("rope-dim", sum(section)))
    if sum(section) != rope_dim:
        raise ValueError(
            f"MRoPE section sum {sum(section)} does not equal rope-dim {rope_dim}"
        )

    vocab_size = int(context["n-vocab"])
    hidden_size = int(embedding["size"])
    eos_token_id = int(context["eos-token"])
    embedding_path = _require_path(
        bundle_dir / str(embedding["lut-path"]),
        "Embedding table",
    )
    expected_bytes = vocab_size * hidden_size * np.dtype("<f4").itemsize
    actual_bytes = embedding_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Embedding table is {actual_bytes} bytes; expected {expected_bytes}"
        )

    tokenizer_value = dialog.get("tokenizer")
    if not isinstance(tokenizer_value, dict):
        raise ValueError("Missing tokenizer config")
    tokenizer_path = _require_path(
        bundle_dir / str(tokenizer_value["path"]),
        "Tokenizer",
    )

    return BundleContract(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        eos_token_id=eos_token_id,
        rope_theta=float(positional["rope-theta"]),
        mrope_section=section,
        spatial_merge_size=int(scaling["spatial-merge-size"]),
        embedding_path=embedding_path,
        tokenizer_path=tokenizer_path,
    )


def tokenize_video_prompt(
    tokenizer_path: Path,
    video_manifest: Mapping[str, Any],
) -> np.ndarray:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "The 'tokenizers' package is required to build prompt IDs"
        ) from exc

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_probe = tokenizer.encode(
        "<|video_pad|>",
        add_special_tokens=False,
    ).ids
    if pad_probe != [VIDEO_PAD_TOKEN_ID]:
        raise ValueError(
            "Tokenizer does not map <|video_pad|> to token 151656"
        )

    text_chunks = video_manifest.get("text_chunks")
    pairs = video_manifest.get("pairs")
    if not isinstance(text_chunks, list) or not isinstance(pairs, list):
        raise ValueError("Video manifest must contain text_chunks and pairs lists")
    if len(text_chunks) != len(pairs) + 1 or not pairs:
        raise ValueError(
            "Expected one prefix/bridge per visual pair plus one suffix"
        )

    graph_contract = video_manifest.get("graph_contract")
    if not isinstance(graph_contract, dict):
        raise ValueError("Video manifest has no graph_contract object")
    visual_tokens_per_pair = int(graph_contract["visual_tokens_per_pair"])

    prompt: list[int] = []
    for pair_index, chunk_record in enumerate(text_chunks[:-1]):
        if not isinstance(chunk_record, dict):
            raise ValueError(f"Text chunk {pair_index} is not an object")
        tokens = tokenizer.encode(
            str(chunk_record["text"]),
            add_special_tokens=False,
        ).ids
        expected_count = int(chunk_record.get("tokens", len(tokens)))
        if len(tokens) != expected_count:
            raise ValueError(
                f"Text chunk {pair_index} produced {len(tokens)} tokens; "
                f"manifest records {expected_count}"
            )
        if (
            not tokens
            or tokens[-1] != VISION_START_TOKEN_ID
            or tokens.count(VISION_START_TOKEN_ID) != 1
            or IMAGE_PAD_TOKEN_ID in tokens
            or VIDEO_PAD_TOKEN_ID in tokens
        ):
            raise ValueError(
                f"Text chunk {pair_index} violates the vision-start contract"
            )
        if pair_index == 0:
            if VISION_END_TOKEN_ID in tokens:
                raise ValueError("Prefix unexpectedly contains vision-end")
        elif (
            tokens[0] != VISION_END_TOKEN_ID
            or tokens.count(VISION_END_TOKEN_ID) != 1
        ):
            raise ValueError(
                f"Bridge {pair_index} does not begin with one vision-end token"
            )
        prompt.extend(tokens)
        prompt.extend([VIDEO_PAD_TOKEN_ID] * visual_tokens_per_pair)

    suffix_record = text_chunks[-1]
    if not isinstance(suffix_record, dict):
        raise ValueError("Suffix text chunk is not an object")
    suffix = tokenizer.encode(
        str(suffix_record["text"]),
        add_special_tokens=False,
    ).ids
    expected_suffix_count = int(suffix_record.get("tokens", len(suffix)))
    if len(suffix) != expected_suffix_count:
        raise ValueError(
            f"Suffix produced {len(suffix)} tokens; "
            f"manifest records {expected_suffix_count}"
        )
    if not suffix or suffix[0] != VISION_END_TOKEN_ID:
        raise ValueError("Suffix does not begin with vision-end")
    prompt.extend(suffix)

    prompt_ids = np.asarray(prompt, dtype=np.int64)
    context_budget = video_manifest.get("context_budget")
    if isinstance(context_budget, dict):
        expected_prompt_tokens = int(context_budget["prompt_tokens"])
        expected_visual_tokens = int(context_budget["visual_tokens"])
        if prompt_ids.size != expected_prompt_tokens:
            raise ValueError(
                f"Prompt has {prompt_ids.size} tokens; expected "
                f"{expected_prompt_tokens}"
            )
        if int(np.count_nonzero(prompt_ids == VIDEO_PAD_TOKEN_ID)) != (
            expected_visual_tokens
        ):
            raise ValueError("Visual-token count does not match manifest")

    return prompt_ids


def image_grids_from_manifest(
    video_manifest: Mapping[str, Any],
) -> list[tuple[int, int, int]]:
    pairs = video_manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Video manifest has no visual pairs")
    grids: list[tuple[int, int, int]] = []
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Pair {pair_index} is not an object")
        pixel_values = pair.get("pixel_values")
        if not isinstance(pixel_values, dict):
            raise ValueError(f"Pair {pair_index} has no pixel_values object")
        grid = tuple(int(value) for value in pixel_values["grid_thw"])
        if len(grid) != 3:
            raise ValueError(f"Pair {pair_index} has invalid grid {grid!r}")
        grids.append(grid)
    return grids


def compute_mrope_positions(
    input_ids: np.ndarray,
    image_grids: Sequence[tuple[int, int, int]],
    spatial_merge_size: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Port ``computeMRoPEPositions`` from the pinned GenieX runtime."""

    ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    positions = np.zeros((3, ids.size), dtype=np.int32)
    running_position = 0
    image_index = 0
    token_index = 0

    while token_index < ids.size:
        token = int(ids[token_index])
        if token == VISION_START_TOKEN_ID and image_index < len(image_grids):
            positions[:, token_index] = running_position
            running_position += 1
            token_index += 1

            temporal, height, width = image_grids[image_index]
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError(
                    f"Grid {image_grids[image_index]} is not divisible by "
                    f"merge size {spatial_merge_size}"
                )
            llm_height = height // spatial_merge_size
            llm_width = width // spatial_merge_size
            image_start = running_position

            for temporal_index in range(temporal):
                for height_index in range(llm_height):
                    for width_index in range(llm_width):
                        if token_index >= ids.size:
                            raise ValueError("Visual token run ended early")
                        if int(ids[token_index]) != VIDEO_PAD_TOKEN_ID:
                            raise ValueError(
                                f"Expected video-pad token at {token_index}"
                            )
                        positions[:, token_index] = (
                            image_start + temporal_index,
                            image_start + height_index,
                            image_start + width_index,
                        )
                        token_index += 1

            running_position = image_start + max(
                temporal,
                llm_height,
                llm_width,
            )
            image_index += 1
        else:
            positions[:, token_index] = running_position
            running_position += 1
            token_index += 1

    if image_index != len(image_grids):
        raise ValueError(
            f"Consumed {image_index} image grids; expected {len(image_grids)}"
        )
    maximum = int(positions[0].max()) if ids.size else -1
    delta = maximum + 1 - int(ids.size)
    return positions, (delta, delta, delta)


def compute_interleaved_mrope(
    positions: np.ndarray,
    *,
    theta: float,
    mrope_section: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Port GenieX ``MRoPEInputProvider::fillCosSin`` in STRIDE mode."""

    positions = np.asarray(positions, dtype=np.int32)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError(
            f"Expected positions [3, sequence], got {positions.shape}"
        )
    section = tuple(int(value) for value in mrope_section)
    if len(section) != 3:
        raise ValueError(f"Expected three MRoPE sections, got {section!r}")
    half_dim = sum(section)
    inverse_frequency = 1.0 / np.power(
        np.float64(theta),
        np.arange(half_dim, dtype=np.float64) / np.float64(half_dim),
    )
    frequencies = (
        positions[:, :, np.newaxis].astype(np.float64)
        * inverse_frequency[np.newaxis, np.newaxis, :]
    )

    cosine = np.cos(frequencies[0])
    sine = np.sin(frequencies[0])
    for dimension in (1, 2):
        offset = dimension
        length = section[dimension] * 3
        for frequency_index in range(offset, min(length, half_dim), 3):
            cosine[:, frequency_index] = np.cos(
                frequencies[dimension, :, frequency_index]
            )
            sine[:, frequency_index] = np.sin(
                frequencies[dimension, :, frequency_index]
            )

    return (
        np.ascontiguousarray(cosine, dtype=np.float32),
        np.ascontiguousarray(sine, dtype=np.float32),
    )


def build_attention_mask(
    n_past: int,
    valid_tokens: int,
    *,
    sequence_length: int = PREFILL_AR,
    kv_length: int = PREFILL_KV_LENGTH,
) -> np.ndarray:
    if not 0 <= valid_tokens <= sequence_length:
        raise ValueError(
            f"valid_tokens must be in [0, {sequence_length}], got {valid_tokens}"
        )
    mask = np.full(
        (sequence_length, kv_length + sequence_length),
        -1e9,
        dtype=np.float32,
    )
    visible_past = min(n_past, kv_length)
    for row in range(valid_tokens):
        mask[row, :visible_past] = 0.0
        mask[row, kv_length : kv_length + row + 1] = 0.0
    return mask


def _load_qai_hub_h5(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to read AI Hub output H5 files") from exc

    tensors: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"AI Hub H5 has no /data group: {path}")
        records: list[tuple[int, str, np.ndarray]] = []
        for group in handle["data"].values():
            order = int(group.attrs["order"])
            raw_name = group.attrs["name"]
            name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            batch_count = int(group.attrs["batch_count"])
            if batch_count <= 0:
                raise ValueError(f"{path}:{name} has invalid batch_count")
            arrays = [
                np.asarray(group[f"batch_{index}"])
                for index in range(batch_count)
            ]
            records.append((order, name, np.stack(arrays, axis=0)))
        orders = [order for order, _, _ in records]
        names = [name for _, name, _ in records]
        if len(set(orders)) != len(orders):
            raise ValueError(f"AI Hub H5 has duplicate tensor order: {path}")
        if len(set(names)) != len(names):
            raise ValueError(f"AI Hub H5 has duplicate tensor names: {path}")
        for _, name, array in sorted(records):
            tensors[name] = array
    return tensors


def _load_vision_output_file(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {
                name: np.asarray(archive[name])
                for name in archive.files
            }
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_qai_hub_h5(path)
    raise ValueError(f"Unsupported vision output file: {path}")


def load_selected_vision_outputs(
    vision_outputs_path: Path,
    vision_index_manifest_path: Path,
    video_manifest_path: Path,
    *,
    pair_count: int,
    expected_tokens_per_pair: int,
    expected_hidden_size: int,
    vision_source_label: str | None = None,
    vision_job_id: str | None = None,
    vision_model_id: str | None = None,
    vision_dataset_id: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    vision_outputs_path = _require_path(
        vision_outputs_path,
        "Vision output file",
    )
    vision_index_manifest_path = _require_path(
        vision_index_manifest_path,
        "Vision index manifest",
    )
    video_manifest_path = _require_path(video_manifest_path, "Video manifest")

    index_manifest = _read_json(vision_index_manifest_path)
    cases = index_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Vision index manifest has no cases list")
    video_manifest_sha = sha256_file(video_manifest_path)
    matching = [
        case
        for case in cases
        if isinstance(case, dict)
        and str(case.get("manifest_sha256")) == video_manifest_sha
    ]
    matching.sort(key=lambda case: int(case["pair_index"]))
    if len(matching) != pair_count:
        raise ValueError(
            f"Found {len(matching)} vision batches for manifest "
            f"{video_manifest_sha}; expected {pair_count}"
        )
    if [int(case["pair_index"]) for case in matching] != list(range(pair_count)):
        raise ValueError("Vision batches do not cover consecutive pair indices")
    selected_indices = [int(case["index"]) for case in matching]
    if any(index < 0 for index in selected_indices):
        raise ValueError("Vision batch indices must be non-negative")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Vision batch indices must be unique")

    all_outputs = _load_vision_output_file(vision_outputs_path)
    required_names = [
        "image_features",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2",
    ]
    selected: dict[str, np.ndarray] = {}
    for name in required_names:
        if name not in all_outputs:
            raise ValueError(f"Vision output file is missing {name}")
        array = np.asarray(all_outputs[name])
        if array.ndim != 3:
            raise ValueError(f"{name} must have [batch, tokens, hidden] shape")
        if max(selected_indices) >= array.shape[0]:
            raise ValueError(
                f"{name} has {array.shape[0]} batches, cannot select "
                f"{selected_indices}"
            )
        selected[name] = np.ascontiguousarray(
            array[selected_indices],
            dtype=np.float32,
        )
        expected_shape = (
            pair_count,
            expected_tokens_per_pair,
            expected_hidden_size,
        )
        if selected[name].shape != expected_shape:
            raise ValueError(
                f"{name} selected shape {selected[name].shape} does not "
                f"match exact contract {expected_shape}"
            )
        if not np.isfinite(selected[name]).all():
            raise ValueError(f"{name} contains non-finite values")

    is_hub_output = vision_outputs_path.suffix.lower() in {".h5", ".hdf5"}
    output_kind = (
        "qai_hub_inference_h5" if is_hub_output else "host_reference_npz"
    )
    if vision_source_label is None:
        if is_hub_output:
            vision_source_label = (
                f"qai_hub_inference:{vision_outputs_path.stem}"
            )
        else:
            vision_source_label = str(
                index_manifest.get("reference") or vision_outputs_path.stem
            )
    if not is_hub_output and any(
        value is not None
        for value in (vision_job_id, vision_model_id, vision_dataset_id)
    ):
        raise ValueError(
            "AI Hub job/model/dataset IDs may only describe an H5 output"
        )

    return selected, {
        "vision_outputs_path": str(vision_outputs_path),
        "vision_outputs_sha256": sha256_file(vision_outputs_path),
        "vision_output_kind": output_kind,
        "vision_source_label": vision_source_label,
        "vision_index_manifest_path": str(vision_index_manifest_path),
        "vision_index_manifest_sha256": sha256_file(
            vision_index_manifest_path
        ),
        "video_manifest_sha256": video_manifest_sha,
        "selected_indices": selected_indices,
        "selected_cases": matching,
        "vision_index_reference_label": index_manifest.get("reference"),
        "qai_hub_provenance": (
            {
                "inference_job_id": vision_job_id,
                "model_id": vision_model_id,
                "dataset_id": vision_dataset_id,
            }
            if is_hub_output
            else None
        ),
    }


def prepare_prompt(
    *,
    bundle_dir: Path,
    video_manifest_path: Path,
    vision_outputs_path: Path,
    vision_index_manifest_path: Path,
    vision_source_label: str | None = None,
    vision_job_id: str | None = None,
    vision_model_id: str | None = None,
    vision_dataset_id: str | None = None,
) -> PreparedPrompt:
    bundle_dir = bundle_dir.expanduser().resolve()
    video_manifest_path = _require_path(video_manifest_path, "Video manifest")
    vision_outputs_path = _require_path(
        vision_outputs_path,
        "Vision output file",
    )
    vision_index_manifest_path = _require_path(
        vision_index_manifest_path,
        "Vision index manifest",
    )

    contract = load_bundle_contract(bundle_dir)
    video_manifest = _read_json(video_manifest_path)
    input_ids = tokenize_video_prompt(contract.tokenizer_path, video_manifest)
    grids = image_grids_from_manifest(video_manifest)
    visual_token_counts = [
        temporal
        * (height // contract.spatial_merge_size)
        * (width // contract.spatial_merge_size)
        for temporal, height, width in grids
    ]
    if len(set(visual_token_counts)) != 1:
        raise ValueError(
            "All frozen visual pairs must have one static token count"
        )
    positions, deltas = compute_mrope_positions(
        input_ids,
        grids,
        contract.spatial_merge_size,
    )
    cosine, sine = compute_interleaved_mrope(
        positions,
        theta=contract.rope_theta,
        mrope_section=contract.mrope_section,
    )

    vision, vision_record = load_selected_vision_outputs(
        vision_outputs_path,
        vision_index_manifest_path,
        video_manifest_path,
        pair_count=len(grids),
        expected_tokens_per_pair=visual_token_counts[0],
        expected_hidden_size=contract.hidden_size,
        vision_source_label=vision_source_label,
        vision_job_id=vision_job_id,
        vision_model_id=vision_model_id,
        vision_dataset_id=vision_dataset_id,
    )
    visual_rows = vision["image_features"].reshape(-1, contract.hidden_size)
    visual_mask_full = input_ids == VIDEO_PAD_TOKEN_ID
    if int(np.count_nonzero(visual_mask_full)) != visual_rows.shape[0]:
        raise ValueError(
            f"Prompt has {np.count_nonzero(visual_mask_full)} visual tokens "
            f"but vision output has {visual_rows.shape[0]} rows"
        )

    table = np.memmap(
        contract.embedding_path,
        mode="r",
        dtype="<f4",
        shape=(contract.vocab_size, contract.hidden_size),
    )
    if input_ids.min(initial=0) < 0 or input_ids.max(initial=0) >= (
        contract.vocab_size
    ):
        raise ValueError("Prompt token ID is outside the embedding vocabulary")
    prompt_embeddings = np.asarray(table[input_ids], dtype=np.float32)
    prompt_embeddings[visual_mask_full] = visual_rows
    eos_embedding = np.asarray(
        table[contract.eos_token_id],
        dtype=np.float32,
    ).copy()

    deepstack_full = {
        name: array.reshape(-1, contract.hidden_size)
        for name, array in vision.items()
        if name.startswith("deepstack_visual_embeds_")
    }
    chunks: list["OrderedDict[str, np.ndarray]"] = []
    chunk_valid_tokens: list[int] = []
    chunk_visual_tokens: list[int] = []
    total_tokens = int(input_ids.size)
    for start in range(0, total_tokens, PREFILL_AR):
        valid_tokens = min(PREFILL_AR, total_tokens - start)
        end = start + valid_tokens
        prior_visual_tokens = int(
            np.count_nonzero(visual_mask_full[:start])
        )

        chunk_embeddings = np.empty(
            (PREFILL_AR, contract.hidden_size),
            dtype=np.float32,
        )
        chunk_embeddings[:valid_tokens] = prompt_embeddings[start:end]
        if valid_tokens < PREFILL_AR:
            chunk_embeddings[valid_tokens:] = eos_embedding

        chunk_cosine = np.ones(
            (PREFILL_AR, cosine.shape[1]),
            dtype=np.float32,
        )
        chunk_sine = np.zeros(
            (PREFILL_AR, sine.shape[1]),
            dtype=np.float32,
        )
        chunk_cosine[:valid_tokens] = cosine[start:end]
        chunk_sine[:valid_tokens] = sine[start:end]

        chunk_visual_mask = np.zeros(PREFILL_AR, dtype=np.bool_)
        chunk_visual_mask[:valid_tokens] = visual_mask_full[start:end]

        static_inputs: "OrderedDict[str, np.ndarray]" = OrderedDict()
        static_inputs["inputs_embeds"] = chunk_embeddings.reshape(
            1,
            PREFILL_AR,
            contract.hidden_size,
        )
        static_inputs["position_ids_cos"] = chunk_cosine.reshape(
            1,
            1,
            PREFILL_AR,
            cosine.shape[1],
        )
        static_inputs["position_ids_sin"] = chunk_sine.reshape(
            1,
            1,
            PREFILL_AR,
            sine.shape[1],
        )
        static_inputs["attention_mask"] = build_attention_mask(
            start,
            valid_tokens,
        ).reshape(1, 1, PREFILL_AR, CONTEXT_LENGTH)
        static_inputs["visual_pos_masks"] = chunk_visual_mask.reshape(
            1,
            PREFILL_AR,
        )

        for level in range(3):
            name = f"deepstack_visual_embeds_{level}"
            source = deepstack_full[name]
            output = np.zeros(
                (256, contract.hidden_size),
                dtype=np.float32,
            )
            available = max(0, source.shape[0] - prior_visual_tokens)
            copied_rows = min(output.shape[0], available)
            if copied_rows:
                output[:copied_rows] = source[
                    prior_visual_tokens : prior_visual_tokens + copied_rows
                ]
            static_inputs[name] = output

        chunks.append(static_inputs)
        chunk_valid_tokens.append(valid_tokens)
        chunk_visual_tokens.append(
            int(np.count_nonzero(chunk_visual_mask))
        )

    return PreparedPrompt(
        input_ids=input_ids,
        positions=positions,
        chunks=chunks,
        chunk_valid_tokens=chunk_valid_tokens,
        chunk_visual_tokens=chunk_visual_tokens,
        source_record={
            "bundle_dir": str(bundle_dir),
            "genie_config_sha256": sha256_file(bundle_dir / "genie_config.json"),
            "tokenizer_sha256": sha256_file(contract.tokenizer_path),
            "embedding_table_sha256": sha256_file(contract.embedding_path),
            "video_manifest_path": str(video_manifest_path),
            "video_manifest_sha256": sha256_file(video_manifest_path),
            "vision": vision_record,
            "contract": {
                "vocab_size": contract.vocab_size,
                "hidden_size": contract.hidden_size,
                "eos_token_id": contract.eos_token_id,
                "rope_theta": contract.rope_theta,
                "mrope_section": list(contract.mrope_section),
                "spatial_merge_size": contract.spatial_merge_size,
                "prefill_ar": PREFILL_AR,
                "context_length": CONTEXT_LENGTH,
                "prefill_kv_length": PREFILL_KV_LENGTH,
            },
            "mrope_deltas": list(deltas),
        },
    )


def zero_p1_kv_inputs() -> "OrderedDict[str, np.ndarray]":
    inputs: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for layer in range(P1_FIRST_LAYER, P1_LAST_LAYER + 1):
        inputs[f"past_key_{layer}_in"] = np.zeros(
            (NUM_KV_HEADS, 1, HEAD_DIM, PREFILL_KV_LENGTH),
            dtype=np.float32,
        )
        inputs[f"past_value_{layer}_in"] = np.zeros(
            (NUM_KV_HEADS, 1, PREFILL_KV_LENGTH, HEAD_DIM),
            dtype=np.float32,
        )
    return inputs


def make_p1_chunk_inputs(
    prepared: PreparedPrompt,
    chunk_index: int,
    *,
    kv_inputs: Mapping[str, np.ndarray] | None = None,
) -> "OrderedDict[str, np.ndarray]":
    if not 0 <= chunk_index < len(prepared.chunks):
        raise IndexError(
            f"Chunk index {chunk_index} is outside [0, {len(prepared.chunks)})"
        )
    if kv_inputs is None:
        if chunk_index != 0:
            raise ValueError(
                "Only chunk 0 may infer an all-zero KV cache. Later chunks "
                "require real outputs from the previous inference."
            )
        kv_inputs = zero_p1_kv_inputs()

    expected_kv_names = list(zero_p1_kv_inputs())
    if list(kv_inputs) != expected_kv_names:
        raise ValueError(
            "KV input names/order do not match the P1 AR128 contract"
        )
    expected_kv = zero_p1_kv_inputs()
    for name in expected_kv_names:
        value = np.asarray(kv_inputs[name])
        if value.shape != expected_kv[name].shape:
            raise ValueError(
                f"{name} shape {value.shape} != {expected_kv[name].shape}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must be floating point")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        if chunk_index == 0 and np.any(value != 0):
            raise ValueError(f"Chunk 0 requires an all-zero KV cache: {name}")

    inputs: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for name in expected_kv_names:
        inputs[name] = np.ascontiguousarray(
            kv_inputs[name],
            dtype=np.float32,
        )
    inputs.update(prepared.chunks[chunk_index])
    return inputs


def tensor_record(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    finite = (
        np.isfinite(contiguous).all()
        if np.issubdtype(contiguous.dtype, np.floating)
        else True
    )
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "nbytes": int(contiguous.nbytes),
        "sha256": sha256_array(contiguous),
        "finite": bool(finite),
    }


def dataset_fingerprint(inputs: Mapping[str, np.ndarray]) -> str:
    """Hash ordered names plus dtype, shape, and C-order tensor bytes."""

    digest = hashlib.sha256()
    for name, value in inputs.items():
        array = np.ascontiguousarray(value)
        name_bytes = name.encode("utf-8")
        dtype_bytes = array.dtype.str.encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "little"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "little"))
        digest.update(dtype_bytes)
        digest.update(len(array.shape).to_bytes(4, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_golden_profile(
    prepared: PreparedPrompt,
    profile_name: str,
) -> dict[str, Any]:
    if profile_name not in GOLDEN_PROFILES:
        raise KeyError(
            f"Unknown golden profile {profile_name!r}; choose from "
            f"{sorted(GOLDEN_PROFILES)}"
        )
    profile = GOLDEN_PROFILES[profile_name]
    errors: list[str] = []
    actual_manifest_sha = prepared.source_record["video_manifest_sha256"]
    if actual_manifest_sha != profile["video_manifest_sha256"]:
        errors.append(
            "video manifest SHA mismatch: "
            f"{actual_manifest_sha} != {profile['video_manifest_sha256']}"
        )
    actual_ids_sha = sha256_array(
        np.asarray(prepared.input_ids, dtype=np.int64)
    )
    if actual_ids_sha != profile["input_ids_int64_sha256"]:
        errors.append(
            "input ID SHA mismatch: "
            f"{actual_ids_sha} != {profile['input_ids_int64_sha256']}"
        )
    if prepared.input_ids.size != profile["prompt_tokens"]:
        errors.append(
            f"prompt tokens {prepared.input_ids.size} != "
            f"{profile['prompt_tokens']}"
        )
    if len(prepared.chunks) != len(profile["chunks"]):
        errors.append(
            f"chunk count {len(prepared.chunks)} != "
            f"{len(profile['chunks'])}"
        )

    checked_chunks: list[dict[str, Any]] = []
    for chunk_index, expected in enumerate(profile["chunks"]):
        if chunk_index >= len(prepared.chunks):
            break
        chunk = prepared.chunks[chunk_index]
        if prepared.chunk_valid_tokens[chunk_index] != expected["valid_tokens"]:
            errors.append(f"chunk {chunk_index} valid-token mismatch")
        if (
            prepared.chunk_visual_tokens[chunk_index]
            != expected["visual_tokens"]
        ):
            errors.append(f"chunk {chunk_index} visual-token mismatch")
        actual_hashes = {
            name: sha256_array(value)
            for name, value in chunk.items()
        }
        for name, expected_hash in expected.items():
            if name in {"valid_tokens", "visual_tokens"}:
                continue
            actual_hash = actual_hashes.get(name)
            if actual_hash != expected_hash:
                errors.append(
                    f"chunk {chunk_index} {name} SHA mismatch: "
                    f"{actual_hash} != {expected_hash}"
                )
        checked_chunks.append(
            {
                "chunk_index": chunk_index,
                "hashes": actual_hashes,
            }
        )

    if errors:
        raise ValueError(
            f"Golden profile {profile_name} validation failed:\n- "
            + "\n- ".join(errors)
        )
    return {
        "profile": profile_name,
        "status": "passed",
        "chunks": checked_chunks,
    }


def write_input_artifact(
    output_dir: Path,
    prepared: PreparedPrompt,
    inputs: Mapping[str, np.ndarray],
    *,
    chunk_index: int,
    golden_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = output_dir / f"p1_chunk{chunk_index}_inputs.npz"
    np.savez_compressed(tensor_path, **inputs)
    input_ids_path = output_dir / "prompt_input_ids.npy"
    np.save(input_ids_path, np.asarray(prepared.input_ids, dtype=np.int64))
    static_chunk_archives: list[dict[str, Any]] = []
    for index, chunk in enumerate(prepared.chunks):
        static_path = output_dir / f"p1_chunk{index}_static_inputs.npz"
        np.savez_compressed(static_path, **chunk)
        static_chunk_archives.append(
            {
                "chunk_index": index,
                "path": static_path.name,
                "sha256": sha256_file(static_path),
                "tensor_fingerprint_sha256": dataset_fingerprint(chunk),
                "input_order": list(chunk),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "purpose": (
            "Exact pinned-GenieX AR128/P1 input for a real frozen video prompt"
        ),
        "graph_name": P1_GRAPH_NAME,
        "chunk_index": chunk_index,
        "valid_tokens": prepared.chunk_valid_tokens[chunk_index],
        "visual_tokens": prepared.chunk_visual_tokens[chunk_index],
        "input_order": list(inputs),
        "dataset_fingerprint_sha256": dataset_fingerprint(inputs),
        "tensors": {
            name: tensor_record(array)
            for name, array in inputs.items()
        },
        "tensor_archive": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
        },
        "static_chunk_archives": static_chunk_archives,
        "prompt_input_ids": {
            "path": input_ids_path.name,
            "shape": list(prepared.input_ids.shape),
            "dtype": "int64",
            "sha256": sha256_array(
                np.asarray(prepared.input_ids, dtype=np.int64)
            ),
        },
        "all_chunks": [
            {
                "chunk_index": index,
                "valid_tokens": prepared.chunk_valid_tokens[index],
                "visual_tokens": prepared.chunk_visual_tokens[index],
                "static_tensor_hashes": {
                    name: sha256_array(array)
                    for name, array in chunk.items()
                },
            }
            for index, chunk in enumerate(prepared.chunks)
        ],
        "source": prepared.source_record,
        "golden_validation": golden_validation,
        "security": {
            "contains_credentials": False,
            "note": (
                "No environment variables, API tokens, or network credentials "
                "are serialized."
            ),
        },
        "publication": {
            "classification": "local_only_unsanitized",
            "contains_local_paths": True,
            "sanitizer": "scripts/sanitize_local_manifest.py",
        },
        "limitations": [
            "Chunk 0 uses the only valid synthetic state: an all-zero initial KV cache.",
            (
                "This artifact exercises P1 only. It does not by itself produce "
                "language-model logits or a benchmark answer."
            ),
        ],
    }
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("video_manifest", type=Path)
    parser.add_argument(
        "vision_outputs",
        type=Path,
        help="BF16 .npz reference or Qualcomm AI Hub output .h5",
    )
    parser.add_argument(
        "vision_index_manifest",
        type=Path,
        help="Manifest mapping video-manifest hashes/pair indices to batches",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument(
        "--golden-profile",
        choices=sorted(GOLDEN_PROFILES),
        help="Fail unless all recorded golden hashes match",
    )
    parser.add_argument(
        "--vision-source-label",
        help=(
            "Explicit provenance label, especially for an AI Hub H5 whose "
            "batch-index manifest originated with a host reference"
        ),
    )
    parser.add_argument("--vision-job-id", help="AI Hub vision inference job ID")
    parser.add_argument("--vision-model-id", help="AI Hub vision target model ID")
    parser.add_argument(
        "--vision-dataset-id",
        help="AI Hub vision input dataset ID",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepared = prepare_prompt(
        bundle_dir=args.bundle_dir,
        video_manifest_path=args.video_manifest,
        vision_outputs_path=args.vision_outputs,
        vision_index_manifest_path=args.vision_index_manifest,
        vision_source_label=args.vision_source_label,
        vision_job_id=args.vision_job_id,
        vision_model_id=args.vision_model_id,
        vision_dataset_id=args.vision_dataset_id,
    )
    inputs = make_p1_chunk_inputs(prepared, args.chunk_index)
    golden_validation = (
        validate_golden_profile(prepared, args.golden_profile)
        if args.golden_profile
        else None
    )
    manifest = write_input_artifact(
        args.output_dir,
        prepared,
        inputs,
        chunk_index=args.chunk_index,
        golden_validation=golden_validation,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "dataset_fingerprint_sha256": manifest[
                    "dataset_fingerprint_sha256"
                ],
                "input_manifest": "input_manifest.json",
                "golden_validation": golden_validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
