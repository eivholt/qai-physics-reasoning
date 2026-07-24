#!/usr/bin/env python3
"""Prepare a self-converted Qwen3-VL QAIRT bundle for Qualcomm GenieX.

GenieX dispatches local QAIRT bundles by the ``model_id`` prefix in
``metadata.json``.  A custom Cosmos export therefore needs a ``qwen3_vl_``
prefix so GenieX selects its Qwen3-VL pipeline instead of rejecting the
otherwise valid bundle.

The source bundle is never modified.  The destination can use hard links for
large context binaries when both directories are on the same filesystem; the
metadata and provenance marker are always independent files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.vision_profile import VisionProfile
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from vision_profile import VisionProfile

METADATA_FILENAME = "metadata.json"
GENIE_CONFIG_FILENAME = "genie_config.json"
MARKER_FILENAME = "geniex_compat.json"
QAIRT245_COMPAT_MARKER = "qairt_245_compat.json"
W4_FP16_MARKER = "w4_fp16.json"
DEFAULT_MODEL_ID = "qwen3_vl_cosmos_reason2_2b"
PART1_CONTEXT = "part1_of_4.bin"
VISION_CONTEXT = "vision_encoder.bin"
VISION_PROFILE_FILES = (
    "img-enc-htp.json",
)
VISION_SAMPLE_FILES = (
    "pixel_values.raw",
    "position_ids_cos.raw",
    "position_ids_sin.raw",
    "window_attention_mask.raw",
    "full_attention_mask.raw",
)
REQUIRED_CONTEXTS = {
    PART1_CONTEXT,
    "part2_of_4.bin",
    "part3_of_4.bin",
    "part4_of_4.bin",
    VISION_CONTEXT,
}
DEEPSTACK_INPUTS = {
    "visual_pos_masks",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _validate_bundle(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = sorted(name for name in REQUIRED_CONTEXTS if not (source / name).is_file())
    if missing:
        raise ValueError(f"Bundle is missing required contexts: {missing}")

    metadata = _load_json(source / METADATA_FILENAME, "metadata.json")
    config = _load_json(source / GENIE_CONFIG_FILENAME, "genie_config.json")

    try:
        rope_scaling = config["dialog"]["engine"]["model"]["positional-encoding"][
            "rope-scaling"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("genie_config.json has no positional-encoding rope-scaling") from exc

    if rope_scaling.get("rope-type") != "qwen3vl-mrope":
        raise ValueError(
            "GenieX Qwen3-VL dispatch requires rope-type 'qwen3vl-mrope'"
        )
    if rope_scaling.get("mrope-section") != [24, 20, 20]:
        raise ValueError(
            "Unexpected Qwen3-VL mrope-section; expected [24, 20, 20]"
        )

    vision = metadata.get("genie", {}).get("vision_preprocessing")
    if not isinstance(vision, dict):
        raise ValueError("metadata.json has no genie.vision_preprocessing object")

    return metadata, config


def _validate_part1_replacement(source: Path) -> dict[str, Any]:
    context = source / PART1_CONTEXT
    if not source.is_dir() or not context.is_file():
        raise ValueError(
            "Part-1 replacement bundle must contain "
            f"{PART1_CONTEXT}: {source}"
        )
    metadata = _load_json(source / METADATA_FILENAME, "replacement metadata.json")
    try:
        model_file = metadata["model_files"][PART1_CONTEXT]
        inputs = model_file["inputs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Replacement metadata has no part1_of_4.bin input contract"
        ) from exc
    if not isinstance(inputs, dict):
        raise ValueError("Replacement part-1 inputs must be an ordered object")

    missing = sorted(DEEPSTACK_INPUTS - set(inputs))
    if missing:
        raise ValueError(
            "Replacement part 1 is not a full DeepStack interface; missing "
            + ", ".join(missing)
        )

    input_names = list(inputs)
    try:
        embedding_index = input_names.index("inputs_embeds")
        auxiliary_indices = [input_names.index(name) for name in DEEPSTACK_INPUTS]
    except ValueError as exc:
        raise ValueError(
            "Replacement part-1 input order is missing inputs_embeds or "
            "a DeepStack auxiliary"
        ) from exc
    if any(index < embedding_index for index in auxiliary_indices):
        raise ValueError(
            "Replacement part-1 input order puts a DeepStack auxiliary before "
            "inputs_embeds; this is unsafe for GenieX graph-spec inference"
        )

    mask = inputs["visual_pos_masks"]
    if (
        not isinstance(mask, dict)
        or mask.get("shape") not in ([1, 1], [1, 128])
    ):
        raise ValueError(
            "Replacement visual_pos_masks shape must be [1, 1] or [1, 128]"
        )
    deepstack_shape: list[int] | None = None
    for name in sorted(DEEPSTACK_INPUTS - {"visual_pos_masks"}):
        tensor = inputs[name]
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
            or shape[1] != 2048
        ):
            raise ValueError(
                f"Replacement {name} shape is {shape!r}; expected "
                "[positive visual capacity, 2048]"
            )
        if deepstack_shape is None:
            deepstack_shape = shape
        elif shape != deepstack_shape:
            raise ValueError(
                "Replacement DeepStack input shapes must match; "
                f"{name} is {shape}, expected {deepstack_shape}"
            )

    unexpected_deepstack = sorted(
        name
        for name in inputs
        if name.startswith("deepstack_visual_embeds_")
        and name not in DEEPSTACK_INPUTS
    )
    if unexpected_deepstack:
        raise ValueError(
            "Replacement part 1 has unsupported DeepStack inputs: "
            + ", ".join(unexpected_deepstack)
        )

    replacement_marker = source / W4_FP16_MARKER
    if not replacement_marker.is_file():
        raise ValueError(
            f"Part-1 replacement bundle must contain {W4_FP16_MARKER}"
        )
    supplementary = metadata.get("supplementary_files")
    if (
        not isinstance(supplementary, dict)
        or W4_FP16_MARKER not in supplementary
    ):
        raise ValueError(
            f"Replacement {W4_FP16_MARKER} has no supplementary_files "
            "metadata entry"
        )
    return metadata


def _validate_vision_replacement(
    source: Path,
) -> tuple[dict[str, Any], VisionProfile]:
    context = source / VISION_CONTEXT
    if not source.is_dir() or not context.is_file():
        raise ValueError(
            "Vision replacement bundle must contain "
            f"{VISION_CONTEXT}: {source}"
        )
    metadata = _load_json(
        source / METADATA_FILENAME, "vision replacement metadata.json"
    )
    profile = VisionProfile.from_metadata(metadata)
    profile.validate_img_encoder_config(source / "img-enc-htp.json")
    profile.validate_ancillary_files(source / "sample_inputs")
    pixel_values = source / "sample_inputs" / "pixel_values.raw"
    if not pixel_values.is_file():
        raise ValueError(
            f"Vision replacement is missing sample tensor: {pixel_values}"
        )
    if pixel_values.stat().st_size != profile.pixel_bytes:
        raise ValueError(
            f"Vision replacement pixel_values.raw has "
            f"{pixel_values.stat().st_size} bytes; expected "
            f"{profile.pixel_bytes}"
        )
    try:
        outputs = metadata["model_files"][VISION_CONTEXT]["outputs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Vision replacement metadata has no vision output contract"
        ) from exc
    for name in (
        "image_features",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2",
    ):
        tensor = outputs.get(name) if isinstance(outputs, dict) else None
        if not isinstance(tensor, dict) or tensor.get("shape") != list(
            profile.visual_shape
        ):
            raise ValueError(
                f"Vision replacement output {name} must have shape "
                f"{list(profile.visual_shape)}"
            )
    return metadata, profile


def _validate_vision_text_capacity(
    text_metadata: dict[str, Any],
    profile: VisionProfile,
) -> None:
    """Ensure the retained first shard can consume one visual prefill slice."""

    try:
        part1_inputs = text_metadata["model_files"][PART1_CONTEXT][
            "inputs"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Text bundle metadata has no part-1 input contract"
        ) from exc
    for name in sorted(DEEPSTACK_INPUTS - {"visual_pos_masks"}):
        tensor = part1_inputs.get(name) if isinstance(part1_inputs, dict) else None
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[1] != profile.hidden_size
            or shape[0] <= 0
        ):
            raise ValueError(
                f"Retained part 1 cannot accept {name} from the replacement "
                f"vision graph: shape {shape!r}"
            )
    mask = (
        part1_inputs.get("visual_pos_masks")
        if isinstance(part1_inputs, dict)
        else None
    )
    if not isinstance(mask, dict):
        raise ValueError(
            "Retained part 1 has no visual_pos_masks DeepStack input"
        )


def _validate_part1_compatibility(
    source_metadata: dict[str, Any],
    replacement_metadata: dict[str, Any],
    vision_metadata: dict[str, Any] | None = None,
) -> None:
    """Fail before copying if the partial export cannot join the base bundle."""

    try:
        source_part1 = source_metadata["model_files"][PART1_CONTEXT]
        replacement_part1 = replacement_metadata["model_files"][PART1_CONTEXT]
        source_inputs = source_part1["inputs"]
        source_outputs = source_part1["outputs"]
        replacement_inputs = replacement_part1["inputs"]
        replacement_outputs = replacement_part1["outputs"]
        selected_vision_metadata = (
            vision_metadata
            if vision_metadata is not None
            else source_metadata
        )
        vision_outputs = selected_vision_metadata["model_files"][
            VISION_CONTEXT
        ]["outputs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Source/replacement metadata lacks a complete part-1 or vision contract"
        ) from exc
    if not all(
        isinstance(contract, dict)
        for contract in (
            source_inputs,
            source_outputs,
            replacement_inputs,
            replacement_outputs,
            vision_outputs,
        )
    ):
        raise ValueError(
            "Source/replacement part-1 and vision contracts must be objects"
        )

    replacement_base_inputs = [
        (name, tensor)
        for name, tensor in replacement_inputs.items()
        if name not in DEEPSTACK_INPUTS
    ]
    if list(source_inputs.items()) != replacement_base_inputs:
        raise ValueError(
            "Replacement part-1 base input contract does not exactly match "
            "the source bundle"
        )
    if list(source_outputs.items()) != list(replacement_outputs.items()):
        raise ValueError(
            "Replacement part-1 output contract does not exactly match "
            "the source bundle"
        )

    part1_visual_shape = replacement_inputs[
        "deepstack_visual_embeds_0"
    ].get("shape")
    for name in (
        "image_features",
        *sorted(DEEPSTACK_INPUTS - {"visual_pos_masks"}),
    ):
        tensor = vision_outputs.get(name)
        if not isinstance(tensor, dict):
            raise ValueError(
                f"Source vision context cannot supply replacement part 1: "
                f"missing output {name}"
            )
        vision_shape = tensor.get("shape")
        if (
            not isinstance(vision_shape, list)
            or len(vision_shape) != 2
            or not isinstance(part1_visual_shape, list)
            or len(part1_visual_shape) != 2
            or vision_shape[1] != part1_visual_shape[1]
            or vision_shape[0] <= 0
            or part1_visual_shape[0] <= 0
        ):
            raise ValueError(
                f"Source vision output {name} shape {tensor.get('shape')!r} "
                "is incompatible with replacement DeepStack capacity "
                f"{part1_visual_shape!r}"
            )


def _replace_file_atomically(
    source: Path,
    destination: Path,
    *,
    copy_function: Callable[[str, str], str],
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ValueError(f"Temporary replacement already exists: {temporary}")
    copy_function(str(source), str(temporary))
    temporary.replace(destination)


def prepare_bundle(
    source: Path,
    destination: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    hardlink: bool = False,
    part1_replacement_bundle: Path | None = None,
    vision_replacement_bundle: Path | None = None,
) -> Path:
    """Copy a bundle and select GenieX's Qwen3-VL runtime dispatcher."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source bundle is not a directory: {source}")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if not model_id.startswith("qwen3_vl_"):
        raise ValueError("GenieX model_id must start with 'qwen3_vl_'")

    metadata, _ = _validate_bundle(source)
    replacement_source: Path | None = None
    replacement_metadata: dict[str, Any] | None = None
    vision_source: Path | None = None
    vision_metadata: dict[str, Any] | None = None
    vision_profile: VisionProfile | None = None
    if vision_replacement_bundle is not None:
        vision_source = vision_replacement_bundle.expanduser().resolve()
        vision_metadata, vision_profile = _validate_vision_replacement(
            vision_source
        )
    if part1_replacement_bundle is not None:
        replacement_source = part1_replacement_bundle.expanduser().resolve()
        replacement_metadata = _validate_part1_replacement(replacement_source)
        _validate_part1_compatibility(
            metadata,
            replacement_metadata,
            vision_metadata=vision_metadata,
        )
    if vision_profile is not None:
        _validate_vision_text_capacity(
            replacement_metadata
            if replacement_metadata is not None
            else metadata,
            vision_profile,
        )
    original_model_id = metadata.get("model_id")
    if not isinstance(original_model_id, str) or not original_model_id:
        raise ValueError("metadata.json has no non-empty model_id")

    copy_function: Callable[[str, str], str] = (
        _hardlink_or_copy if hardlink else shutil.copy2
    )
    shutil.copytree(source, destination, copy_function=copy_function)

    try:
        destination_metadata = destination / METADATA_FILENAME
        # copytree may have hard-linked metadata. Replace it atomically so the
        # source file remains byte-for-byte untouched.
        patched_metadata = dict(metadata)
        patched_metadata["model_id"] = model_id
        if replacement_source is not None or vision_source is not None:
            source_model_files = metadata.get("model_files")
            if not isinstance(source_model_files, dict):
                raise ValueError(
                    "Source metadata has no model_files object for replacement"
                )
            patched_metadata["model_files"] = dict(source_model_files)

        if (
            vision_source is not None
            and vision_metadata is not None
            and vision_profile is not None
        ):
            patched_metadata["model_files"][VISION_CONTEXT] = (
                vision_metadata["model_files"][VISION_CONTEXT]
            )
            _replace_file_atomically(
                vision_source / VISION_CONTEXT,
                destination / VISION_CONTEXT,
                copy_function=copy_function,
            )
            for filename in VISION_PROFILE_FILES:
                _replace_file_atomically(
                    vision_source / filename,
                    destination / filename,
                    copy_function=shutil.copy2,
                )
            destination_samples = destination / "sample_inputs"
            destination_samples.mkdir(exist_ok=True)
            for filename in VISION_SAMPLE_FILES:
                _replace_file_atomically(
                    vision_source / "sample_inputs" / filename,
                    destination_samples / filename,
                    copy_function=shutil.copy2,
                )
            # Prepared video inputs are bound to the old graph shapes/hashes.
            # Never carry them into a bundle with a different vision context.
            stale_video_inputs = destination / "video_inputs"
            if stale_video_inputs.is_dir():
                shutil.rmtree(stale_video_inputs)

            source_genie = metadata.get("genie")
            replacement_genie = vision_metadata.get("genie")
            if not isinstance(source_genie, dict) or not isinstance(
                replacement_genie, dict
            ):
                raise ValueError(
                    "Source/replacement metadata has no genie object"
                )
            replacement_preprocessing = replacement_genie.get(
                "vision_preprocessing"
            )
            if not isinstance(replacement_preprocessing, dict):
                raise ValueError(
                    "Vision replacement has no preprocessing metadata"
                )
            patched_metadata["genie"] = dict(source_genie)
            patched_metadata["genie"]["vision_preprocessing"] = (
                replacement_preprocessing
            )
            replacement_supplementary = vision_metadata.get(
                "supplementary_files"
            )
            if isinstance(replacement_supplementary, dict):
                supplementary = patched_metadata.get("supplementary_files")
                patched_metadata["supplementary_files"] = (
                    dict(supplementary)
                    if isinstance(supplementary, dict)
                    else {}
                )
                for filename in VISION_PROFILE_FILES:
                    if filename in replacement_supplementary:
                        patched_metadata["supplementary_files"][filename] = (
                            replacement_supplementary[filename]
                        )

        if replacement_source is not None and replacement_metadata is not None:
            replacement_model_file = replacement_metadata["model_files"][
                PART1_CONTEXT
            ]
            patched_metadata["model_files"][PART1_CONTEXT] = replacement_model_file

            _replace_file_atomically(
                replacement_source / PART1_CONTEXT,
                destination / PART1_CONTEXT,
                copy_function=copy_function,
            )
            replacement_marker = replacement_source / W4_FP16_MARKER
            _replace_file_atomically(
                replacement_marker,
                destination / W4_FP16_MARKER,
                copy_function=shutil.copy2,
            )
            replacement_supplementary = replacement_metadata[
                "supplementary_files"
            ]
            supplementary = patched_metadata.get("supplementary_files")
            patched_metadata["supplementary_files"] = (
                dict(supplementary)
                if isinstance(supplementary, dict)
                else {}
            )
            patched_metadata["supplementary_files"][W4_FP16_MARKER] = (
                replacement_supplementary[W4_FP16_MARKER]
            )
            compatibility_marker = destination / QAIRT245_COMPAT_MARKER
            compatibility_marker.unlink(missing_ok=True)
            supplementary = patched_metadata.get("supplementary_files")
            if isinstance(supplementary, dict):
                patched_metadata["supplementary_files"] = dict(supplementary)
                patched_metadata["supplementary_files"].pop(
                    QAIRT245_COMPAT_MARKER, None
                )

        temporary_metadata = destination / f".{METADATA_FILENAME}.tmp"
        temporary_metadata.write_text(
            json.dumps(patched_metadata, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        temporary_metadata.replace(destination_metadata)

        marker = {
            "schema_version": 2,
            "purpose": "Select Qualcomm GenieX's Qwen3-VL QAIRT pipeline",
            "source_bundle_name": source.name,
            "source_metadata_sha256": sha256_file(source / METADATA_FILENAME),
            "original_model_id": original_model_id,
            "geniex_model_id": model_id,
            "qwen3_vl_contract": {
                "rope_type": "qwen3vl-mrope",
                "mrope_section": [24, 20, 20],
                "mrope_interleaving": "stride",
            },
            "hardlink_requested": hardlink,
            "part1_replacement": (
                {
                    "source_bundle_name": replacement_source.name,
                    "context_sha256": sha256_file(
                        replacement_source / PART1_CONTEXT
                    ),
                    "metadata_sha256": sha256_file(
                        replacement_source / METADATA_FILENAME
                    ),
                    "deepstack_inputs": sorted(DEEPSTACK_INPUTS),
                    "removed_compatibility_marker": QAIRT245_COMPAT_MARKER,
                }
                if replacement_source is not None
                else None
            ),
            "vision_replacement": (
                {
                    "source_bundle_name": vision_source.name,
                    "context_sha256": sha256_file(
                        vision_source / VISION_CONTEXT
                    ),
                    "metadata_sha256": sha256_file(
                        vision_source / METADATA_FILENAME
                    ),
                    "image_height": vision_profile.image_height,
                    "image_width": vision_profile.image_width,
                    "grid_thw": list(vision_profile.grid_thw),
                    "pixel_values_shape": list(vision_profile.pixel_shape),
                    "visual_tokens": vision_profile.visual_tokens,
                    "reused_text_contexts": [
                        PART1_CONTEXT,
                        "part2_of_4.bin",
                        "part3_of_4.bin",
                        "part4_of_4.bin",
                    ],
                }
                if vision_source is not None and vision_profile is not None
                else None
            ),
        }
        # ``copytree(..., copy_function=_hardlink_or_copy)`` may have linked an
        # existing compatibility marker from an already prepared source
        # bundle.  Replace through a new inode so rewriting provenance never
        # mutates the source marker through that hard link.
        temporary_marker = destination / f".{MARKER_FILENAME}.tmp"
        temporary_marker.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(destination / MARKER_FILENAME)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Hard-link large files when possible (same-filesystem diagnostics)",
    )
    parser.add_argument(
        "--part1-replacement-bundle",
        type=Path,
        default=None,
        help=(
            "Partial full-interface W4/FP export containing a replacement "
            "part1_of_4.bin and metadata.json"
        ),
    )
    parser.add_argument(
        "--vision-replacement-bundle",
        type=Path,
        default=None,
        help=(
            "Partial export containing a replacement vision_encoder.bin, "
            "profile metadata/config, and sample vision tensors. Text "
            "contexts are retained from --source."
        ),
    )
    args = parser.parse_args()

    output = prepare_bundle(
        args.source,
        args.destination,
        model_id=args.model_id,
        hardlink=args.hardlink,
        part1_replacement_bundle=args.part1_replacement_bundle,
        vision_replacement_bundle=args.vision_replacement_bundle,
    )
    print(f"Prepared GenieX bundle: {output}")


if __name__ == "__main__":
    main()
