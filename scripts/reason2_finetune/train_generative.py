from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

from generative_common import (
    ORIGINAL_USER_PROMPT,
    use_edge_first_contract,
    use_evk_single_lane_contract,
    use_evk_speed_classifier_contract,
    auxiliary_task_record,
    stability_boundary_task_record,
    discover_task_records,
    support_boundary_task_record,
)


CLASS_VALUE_TEXT = (
    "YES",
    "NO",
    "0",
    "1",
    "2",
    "SAFE",
    "UNSTABLE_PARCEL",
    "FALLEN_PARCEL",
    "GREEN",
    "AMBER",
    "RED",
)


def class_value_token_mask(
    labels: torch.Tensor,
    token_sequences: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Mark exact class-value spans in completion-only causal-LM labels.

    The production answers contain a long, mostly shared JSON schema. Ordinary
    completion loss can therefore become tiny while the handful of tokens that
    distinguish GREEN, AMBER, and RED remain wrong. Match complete token
    sequences rather than individual token IDs so fragments such as ``AL`` or
    the digits cannot accidentally weight unrelated evidence text.
    """
    mask = torch.zeros_like(labels, dtype=torch.bool)
    sequence_length = labels.shape[-1]
    for sequence in token_sequences:
        width = len(sequence)
        if not width or width > sequence_length:
            continue
        needle = labels.new_tensor(sequence)
        windows = labels.unfold(-1, width, 1)
        matches = windows.eq(needle).all(dim=-1)
        for offset in range(width):
            mask[..., offset : offset + matches.shape[-1]] |= matches
    return mask


class ClassTokenWeightedSFTTrainer(SFTTrainer):
    """SFTTrainer that emphasizes class-defining values, not JSON boilerplate."""

    def __init__(
        self,
        *args,
        class_token_sequences: tuple[tuple[int, ...], ...],
        class_token_loss_weight: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_token_sequences = class_token_sequences
        self.class_token_loss_weight = class_token_loss_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        if labels is None:
            raise RuntimeError("Class-token weighted loss requires causal-LM labels")

        model_inputs = dict(inputs)
        model_inputs["use_cache"] = False
        outputs = model(**model_inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        class_mask = class_value_token_mask(
            labels, self.class_token_sequences
        )[..., 1:]
        valid_mask = shift_labels.ne(-100)
        weights = torch.ones_like(token_loss)
        weights = torch.where(
            class_mask,
            weights.new_full((), self.class_token_loss_weight),
            weights,
        )
        weighted_valid = weights * valid_mask.to(weights.dtype)
        loss = (token_loss * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


def load_replay(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for row in rows:
        result.append({
            "images": [row["image"]],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "source": row.get("source", "general_replay"),
            "code": "REPLAY",
            "label": "REPLAY",
        })
    return result


def materialize_images(records: list[dict], size: tuple[int, int] | None) -> list[dict]:
    result = []
    for record in records:
        item = dict(record)
        images = []
        for path in record["images"]:
            with Image.open(path) as source:
                image = source.convert("RGB")
                if size:
                    image = image.resize(size, Image.Resampling.LANCZOS)
                images.append(image.copy())
        item["images"] = images
        result.append(item)
    return result


def balanced_limit(records: list[dict], maximum: int) -> list[dict]:
    """Select an equal number of task examples per safety class for pilots.

    Full training uses every record. A small randomly truncated pilot can
    otherwise appear to learn while merely seeing an easier class more often.
    """
    if maximum <= 0 or maximum >= len(records):
        return records
    per_class = maximum // 3
    remainder = maximum % 3
    quotas = {
        code: per_class + (1 if index < remainder else 0)
        for index, code in enumerate(("G", "A", "R"))
    }
    selected = []
    counts = {code: 0 for code in quotas}
    for record in records:
        code = record["code"]
        if code in quotas and counts[code] < quotas[code]:
            selected.append(record)
            counts[code] += 1
    if len(selected) != maximum:
        raise RuntimeError(
            f"Could not create balanced {maximum}-sample pilot; selected {counts}"
        )
    return selected


def load_split_metadata(root: Path, split: str) -> dict[str, dict]:
    direct_path = root / split / "metadata.jsonl"
    paths = [direct_path] if direct_path.exists() else [
        path
        for path in root.rglob("metadata.jsonl")
        if path.parent.name == split
    ]
    if not paths:
        raise FileNotFoundError(
            f"Hard-example selection requires a {split}/metadata.jsonl below {root}"
        )
    result = {}
    for path in sorted(paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                stem = Path(row["image"]).stem
                if stem in result and result[stem] != row:
                    raise RuntimeError(
                        f"Ambiguous metadata for {stem}: {path} overlaps another camera bucket"
                    )
                result[stem] = row
    return result


def load_evaluation_failure_seeds(evaluation_path: Path | None) -> dict[str, list[dict]]:
    failure_seeds: dict[str, list[dict]] = {"A": [], "R": []}
    if not evaluation_path:
        return failure_seeds

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validation_metadata: dict[str, dict] = {}
    metadata_paths = {
        Path(row["image"]).parent.parent / "metadata.jsonl"
        for row in evaluation["rows"]
    }
    for metadata_path in sorted(metadata_paths):
        if not metadata_path.exists():
            continue
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metadata_row = json.loads(line)
            stem = Path(metadata_row["image"]).stem
            if stem in validation_metadata and validation_metadata[stem] != metadata_row:
                raise RuntimeError(
                    f"Ambiguous evaluation metadata for {stem}: {metadata_path}"
                )
            validation_metadata[stem] = metadata_row

    label_to_code = {"OVER_EDGE": "A", "FALLEN": "R"}
    for row in evaluation["rows"]:
        if row["truth"] == row["prediction"] or row["truth"] not in label_to_code:
            continue
        stem = Path(row["image"]).stem
        if stem in validation_metadata:
            failure_seeds[label_to_code[row["truth"]]].append(validation_metadata[stem])
    return failure_seeds


BOUNDARY_FEATURE_NAMES = (
    "belt_distance_cm",
    "lateral_offset_cm",
    "along_offset_cm",
    "supported_fraction",
    "roll_deg",
    "yaw_jitter_deg",
)
BOUNDARY_FEATURE_SCALES = (350.0, 160.0, 30.0, 1.0, 30.0, 30.0)


def boundary_distance(candidate: dict, seeds: list[dict]) -> float:
    if not seeds:
        return float("inf")
    return min(
        sum(
            ((float(candidate[name]) - float(seed[name])) / scale) ** 2
            for name, scale in zip(BOUNDARY_FEATURE_NAMES, BOUNDARY_FEATURE_SCALES)
        )
        for seed in seeds
    )


def matched_boundary_triplet_limit(
    records: list[dict],
    root: Path,
    split: str,
    group_count: int,
    hard_example_evaluation: Path | None,
) -> list[dict]:
    """Select complete G/A/R sweeps instead of unrelated class examples.

    Each generated sweep holds parcel identity, camera, lighting, and belt
    position fixed while moving the judged parcel through supported, over-edge,
    and fallen states. Ranking groups near both AMBER and RED validation misses
    preserves the physical transition that independent hard mining discarded.
    """
    metadata = load_split_metadata(root, split)
    records_by_stem = {Path(row["images"][0]).stem: row for row in records}
    groups: dict[int, dict[str, tuple[dict, dict]]] = {}
    for stem, row in records_by_stem.items():
        item = metadata[stem]
        if not item.get("matched_boundary_sweep"):
            continue
        groups.setdefault(int(item["variant_index"]), {})[row["code"]] = (row, item)
    complete = {key: value for key, value in groups.items() if set(value) == set("GAR")}
    if group_count > len(complete):
        raise ValueError(
            f"Requested {group_count} matched triplets; only {len(complete)} complete groups exist"
        )

    failures = load_evaluation_failure_seeds(hard_example_evaluation)
    if not failures["A"] and not failures["R"]:
        # A seed-free matched curriculum is a coverage sample, not a hard-
        # mining run. Select complete variant cycles deterministically so the
        # simulator's bilateral, yaw, and AMBER-pose schedules remain exactly
        # balanced instead of depending on the shuffled record insertion
        # order. Hard-example runs below retain their distance ranking.
        selected_keys = sorted(complete)[:group_count]
        selected = [complete[key][code][0] for key in selected_keys for code in "GAR"]
        if len({row["images"][0] for row in selected}) != len(selected):
            raise RuntimeError("Matched boundary curriculum unexpectedly contains duplicate images")
        return selected
    amber_rank = sorted(
        complete,
        key=lambda key: boundary_distance(complete[key]["A"][1], failures["A"]),
    )
    red_rank = sorted(
        complete,
        key=lambda key: boundary_distance(complete[key]["R"][1], failures["R"]),
    )
    selected_keys: list[int] = []
    if failures["A"] and not failures["R"]:
        # Do not spend half of an AMBER-only hard-mining budget on the stable,
        # arbitrary ordering produced by an empty RED seed set.
        selected_keys.extend(amber_rank[:group_count])
    elif failures["R"] and not failures["A"]:
        selected_keys.extend(red_rank[:group_count])
    else:
        # When both boundaries have failures, alternate their nearest matched
        # sweeps so neither class can consume the whole curriculum.
        for amber_key, red_key in zip(amber_rank, red_rank):
            for key in (amber_key, red_key):
                if key not in selected_keys:
                    selected_keys.append(key)
                    if len(selected_keys) == group_count:
                        break
            if len(selected_keys) == group_count:
                break
    if len(selected_keys) < group_count:
        for key in amber_rank:
            if key not in selected_keys:
                selected_keys.append(key)
                if len(selected_keys) == group_count:
                    break

    selected = [complete[key][code][0] for key in selected_keys for code in "GAR"]
    if len({row["images"][0] for row in selected}) != len(selected):
        raise RuntimeError("Matched boundary curriculum unexpectedly contains duplicate images")
    return selected


def scaled_hard_amber_limit(
    records: list[dict],
    root: Path,
    split: str,
    green: int,
    amber: int,
    red: int,
    hard_fraction: float,
    hard_red_fraction: float,
    hard_example_evaluation: Path | None = None,
) -> list[dict]:
    """Build a unique curriculum biased toward AMBER cases mistaken for RED.

    Low `supported_fraction` AMBER parcels retain only a small roller-contact
    patch and are the observed failure boundary. Keep most AMBER records from
    that end, but span the remainder of the range to avoid learning a narrow
    pose shortcut.
    """
    metadata = load_split_metadata(root, split)
    by_code = {code: [row for row in records if row["code"] == code] for code in "GAR"}
    quotas = {"G": green, "A": amber, "R": red}
    for code, quota in quotas.items():
        if quota > len(by_code[code]):
            raise ValueError(f"Requested {quota} {code} records; only {len(by_code[code])} exist")

    failure_seeds = load_evaluation_failure_seeds(hard_example_evaluation)

    def distance_to_failures(record: dict, code: str) -> float:
        seeds = failure_seeds[code]
        if not seeds:
            return float("inf")
        candidate = metadata[Path(record["images"][0]).stem]
        return boundary_distance(candidate, seeds)

    if failure_seeds["A"]:
        amber_rows = sorted(by_code["A"], key=lambda row: distance_to_failures(row, "A"))
    else:
        amber_rows = sorted(
            by_code["A"],
            key=lambda row: float(metadata[Path(row["images"][0]).stem]["supported_fraction"]),
        )
    hard_count = min(amber, max(0, round(amber * hard_fraction)))
    hard = amber_rows[:hard_count]
    remaining_count = amber - hard_count
    remainder_pool = amber_rows[hard_count:]
    if remaining_count:
        # Evenly cover the easier support range rather than taking one cluster.
        indices = [
            min(len(remainder_pool) - 1, int((i + 0.5) * len(remainder_pool) / remaining_count))
            for i in range(remaining_count)
        ]
        easier = [remainder_pool[index] for index in indices]
    else:
        easier = []
    if failure_seeds["R"]:
        red_rows = sorted(by_code["R"], key=lambda row: distance_to_failures(row, "R"))
    else:
        red_rows = sorted(
            by_code["R"],
            # Fallen parcels nearest the belt are visually closest to an overhang
            # and are the observed RED->AMBER failures after hard-AMBER training.
            key=lambda row: abs(float(metadata[Path(row["images"][0]).stem]["lateral_offset_cm"])),
        )
    hard_red_count = min(red, max(0, round(red * hard_red_fraction)))
    hard_red = red_rows[:hard_red_count]
    red_remaining_count = red - hard_red_count
    red_remainder_pool = red_rows[hard_red_count:]
    if red_remaining_count:
        red_indices = [
            min(
                len(red_remainder_pool) - 1,
                int((i + 0.5) * len(red_remainder_pool) / red_remaining_count),
            )
            for i in range(red_remaining_count)
        ]
        other_red = [red_remainder_pool[index] for index in red_indices]
    else:
        other_red = []
    selected = by_code["G"][:green] + hard + easier + hard_red + other_red
    if len({row["images"][0] for row in selected}) != len(selected):
        raise RuntimeError("Scaled curriculum unexpectedly contains duplicate images")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVIDIA-aligned generative multimodal QLoRA SFT for Reason2."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--adapter",
        help="Continue an existing multimodal adapter instead of creating a new one.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--exact-production-prompt",
        action="store_true",
        help="Use the original runtime prompt for every task record.",
    )
    parser.add_argument(
        "--evk-single-lane-prompt",
        action="store_true",
        help=(
            "Train every production record with the exact short single-lane "
            "EVK prompt and its two-test JSON response. The client derives "
            "the redundant class fields from those tests."
        ),
    )
    parser.add_argument(
        "--evk-speed-classifier",
        action="store_true",
        help=(
            "Train every production record with the compact, user-only "
            "single-lane prompt and an exact one-token G/A/R completion."
        ),
    )
    parser.add_argument(
        "--edge-first-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of production records trained with the otherwise "
            "equivalent answer-first streaming contract. The accepted "
            "adapter remains unchanged; use 1.0 for a separate EVK candidate."
        ),
    )
    parser.add_argument(
        "--auxiliary-reasoning-copy",
        action="store_true",
        help=(
            "Add one rich, class-distinct generated-answer companion for each "
            "selected production JSON example."
        ),
    )
    parser.add_argument(
        "--extra-amber-auxiliary-copies",
        type=int,
        default=0,
        help=(
            "Add extra rich AMBER companions to strengthen the difficult "
            "still-touching-rollers versus fallen boundary in pilot curricula."
        ),
    )
    parser.add_argument(
        "--extra-red-auxiliary-copies",
        type=int,
        default=0,
        help=(
            "Add extra rich RED companions, primarily to preserve an exactly "
            "balanced three-class prior when adding G/A contrast records."
        ),
    )
    parser.add_argument(
        "--support-boundary-copies",
        type=int,
        default=0,
        help=(
            "Add this many balanced A/R contrast companions per selected A and "
            "R production record. This targets still-touching versus lost "
            "roller support without changing the AMBER/RED class prior."
        ),
    )
    parser.add_argument(
        "--stability-boundary-copies",
        type=int,
        default=0,
        help=(
            "Add this many balanced G/A contrast companions per selected G "
            "and A record. This teaches rotation-safe full support versus a "
            "level substantial overhang without making tilt a shortcut."
        ),
    )
    parser.add_argument("--replay")
    parser.add_argument("--replay-samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-task-samples", type=int, default=0)
    parser.add_argument(
        "--scaled-class-samples",
        help="Unique G,A,R production-image counts, for example 30,60,30.",
    )
    parser.add_argument(
        "--matched-boundary-triplets",
        type=int,
        default=0,
        help=(
            "Select this many complete G/A/R matched sweep groups, ranked "
            "against hard AMBER and RED evaluation failures. This is mutually "
            "exclusive with --scaled-class-samples."
        ),
    )
    parser.add_argument(
        "--hard-amber-fraction",
        type=float,
        default=2.0 / 3.0,
        help="Fraction of selected AMBER images drawn from lowest roller support.",
    )
    parser.add_argument(
        "--hard-red-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of selected RED images drawn from fallen parcels nearest "
            "the belt, which are visually closest to difficult overhangs."
        ),
    )
    parser.add_argument(
        "--hard-example-evaluation",
        help=(
            "Evaluation JSON whose misclassified A/R rows seed nearest-neighbor "
            "hard-example selection from the training split."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--class-token-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Multiply loss on exact YES/NO, class-id, label, and "
            "GREEN/AMBER/RED value tokens. A value above 1 preserves full "
            "generative SFT while preventing shared JSON tokens from "
            "dominating the objective."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--language-only-lora",
        action="store_true",
        help=(
            "Adapt only decoder projections. By default this task also adapts "
            "Qwen3-VL visual attention/MLP projections; the conveyor support "
            "boundary is not separable with language-only pilot tuning."
        ),
    )
    parser.add_argument(
        "--freeze-vision-lora",
        action="store_true",
        help=(
            "When continuing an existing multimodal adapter, preserve its "
            "learned visual LoRA and update only language-side LoRA."
        ),
    )
    parser.add_argument(
        "--vision-only-lora",
        action="store_true",
        help=(
            "When continuing an existing multimodal adapter, preserve its "
            "prompt/JSON language LoRA and update only visual LoRA. This is "
            "the conservative choice for matched physical-boundary images."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--bf16-base", action="store_true", help="Use BF16 LoRA instead of 4-bit QLoRA.")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help=(
            "Keep activations in GPU memory instead of recomputing them. "
            "This is faster for the 2B model when the device has sufficient VRAM."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.edge_first_fraction <= 1.0:
        raise ValueError("--edge-first-fraction must be between 0 and 1")
    if args.freeze_vision_lora and args.vision_only_lora:
        raise ValueError("--freeze-vision-lora and --vision-only-lora are mutually exclusive")
    if args.vision_only_lora and not args.adapter:
        raise ValueError("--vision-only-lora requires --adapter")
    if args.class_token_loss_weight < 1.0:
        raise ValueError("--class-token-loss-weight must be at least 1")
    if args.matched_boundary_triplets and args.scaled_class_samples:
        raise ValueError(
            "--matched-boundary-triplets and --scaled-class-samples are mutually exclusive"
        )
    prompt_contracts = sum((
        bool(args.exact_production_prompt),
        bool(args.evk_single_lane_prompt),
        bool(args.evk_speed_classifier),
    ))
    if prompt_contracts > 1:
        raise ValueError(
            "--exact-production-prompt, --evk-single-lane-prompt, and "
            "--evk-speed-classifier are mutually exclusive"
        )

    random.seed(args.seed)
    dataset_root = Path(args.dataset)
    task = discover_task_records(dataset_root, args.split)
    if args.exact_production_prompt:
        for record in task:
            record["prompt"][-1]["content"] = ORIGINAL_USER_PROMPT
    if args.evk_single_lane_prompt:
        for record in task:
            use_evk_single_lane_contract(record)
    if args.evk_speed_classifier:
        for record in task:
            use_evk_speed_classifier_contract(record)
    random.shuffle(task)
    if args.matched_boundary_triplets:
        task = matched_boundary_triplet_limit(
            task,
            dataset_root,
            args.split,
            args.matched_boundary_triplets,
            Path(args.hard_example_evaluation) if args.hard_example_evaluation else None,
        )
        random.shuffle(task)
    elif args.scaled_class_samples:
        counts = [int(value) for value in args.scaled_class_samples.split(",")]
        if len(counts) != 3 or any(value < 0 for value in counts):
            raise ValueError("--scaled-class-samples must be three non-negative G,A,R counts")
        task = scaled_hard_amber_limit(
            task,
            dataset_root,
            args.split,
            counts[0],
            counts[1],
            counts[2],
            args.hard_amber_fraction,
            args.hard_red_fraction,
            Path(args.hard_example_evaluation) if args.hard_example_evaluation else None,
        )
        random.shuffle(task)
    elif args.max_task_samples:
        task = balanced_limit(task, args.max_task_samples)
    edge_rng = random.Random(args.seed ^ 0xE6F1)
    edge_first_count = 0
    for record in task:
        if edge_rng.random() < args.edge_first_fraction:
            use_edge_first_contract(record)
            edge_first_count += 1
    production_task_count = len(task)
    production_records = list(task)
    if args.auxiliary_reasoning_copy:
        auxiliary = [auxiliary_task_record(record) for record in task]
        amber = [record for record in auxiliary if record["code"] == "A"]
        red = [record for record in auxiliary if record["code"] == "R"]
        task = (
            task
            + auxiliary
            + amber * args.extra_amber_auxiliary_copies
            + red * args.extra_red_auxiliary_copies
        )
        random.shuffle(task)
    if args.support_boundary_copies:
        boundary = [
            support_boundary_task_record(record)
            for record in production_records
            if record["code"] in ("A", "R")
        ]
        task = task + boundary * args.support_boundary_copies
        random.shuffle(task)
    if args.stability_boundary_copies:
        stability = [
            stability_boundary_task_record(record)
            for record in production_records
            if record["code"] in ("G", "A")
        ]
        task = task + stability * args.stability_boundary_copies
        random.shuffle(task)
    replay = load_replay(Path(args.replay) if args.replay else None)
    random.shuffle(replay)
    if args.replay_samples:
        replay = replay[:args.replay_samples]
    else:
        replay = []
    records = task + replay
    random.shuffle(records)
    if not records:
        raise SystemExit("No training records")

    # Preserve the exact curriculum before PIL images replace the source
    # paths.  This makes continuation runs reproducible and lets later
    # evaluations distinguish a useful hard-example curriculum from an
    # accidental class-prior shift.
    curriculum_manifest = [
        {
            "images": [str(path) for path in record["images"]],
            "source": record.get("source"),
            "code": record.get("code"),
            "label": record.get("label"),
            "prompt": record.get("prompt"),
            "completion": record.get("completion"),
        }
        for record in records
    ]

    image_size = None
    if args.image_width > 0 and args.image_height > 0:
        image_size = (args.image_width, args.image_height)
    records = materialize_images(records, image_size)
    dataset = Dataset.from_list(records)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"
    speed_class_token_ids = None
    if args.evk_speed_classifier:
        speed_class_token_ids = {
            code: processor.tokenizer.encode(code, add_special_tokens=False)
            for code in ("G", "A", "R")
        }
        if any(len(token_ids) != 1 for token_ids in speed_class_token_ids.values()):
            raise RuntimeError(
                "The speed-v1 G/A/R completions must each encode as exactly one token: "
                f"{speed_class_token_ids}"
            )
    quantization = None if args.bf16_base else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if not args.language_only_lora:
        # Qwen3-VL uses fused qkv and linear_fc* names in the vision tower.
        # These are the same visual branches used by the proven diagnostic
        # adapter; omitting them left generated answers at the base SAFE prior
        # even on training images.
        target_modules += ["qkv", "linear_fc1", "linear_fc2"]
    peft_config = None
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        if args.freeze_vision_lora:
            for name, parameter in model.named_parameters():
                if "visual" in name:
                    parameter.requires_grad = False
        elif args.vision_only_lora:
            for name, parameter in model.named_parameters():
                if "lora_" in name and "visual" not in name:
                    parameter.requires_grad = False
            if not any(
                parameter.requires_grad and "visual" in name and "lora_" in name
                for name, parameter in model.named_parameters()
            ):
                raise RuntimeError("No trainable visual LoRA parameters remain")
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    training = SFTConfig(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if not args.bf16_base else "adamw_torch_fused",
        bf16=True,
        tf32=True,
        max_length=None,
        completion_only_loss=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if not args.no_gradient_checkpointing
            else None
        ),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer_kwargs = {
        "model": model,
        "args": training,
        "train_dataset": dataset,
        "processing_class": processor,
        "peft_config": peft_config,
    }
    if args.class_token_loss_weight > 1.0:
        class_token_sequences = tuple(
            tuple(processor.tokenizer.encode(value, add_special_tokens=False))
            for value in CLASS_VALUE_TEXT
        )
        if any(not sequence for sequence in class_token_sequences):
            raise RuntimeError("A class-value token sequence is unexpectedly empty")
        trainer = ClassTokenWeightedSFTTrainer(
            **trainer_kwargs,
            class_token_sequences=class_token_sequences,
            class_token_loss_weight=args.class_token_loss_weight,
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    Path(args.output, "curriculum_manifest.json").write_text(
        json.dumps(curriculum_manifest, indent=2), encoding="utf-8"
    )
    metrics = dict(result.metrics)
    metrics.update({
        "task_samples": len(task),
        "production_task_samples": production_task_count,
        "auxiliary_task_samples": len(task) - production_task_count,
        "extra_amber_auxiliary_copies": args.extra_amber_auxiliary_copies,
        "extra_red_auxiliary_copies": args.extra_red_auxiliary_copies,
        "support_boundary_copies": args.support_boundary_copies,
        "stability_boundary_copies": args.stability_boundary_copies,
        "scaled_class_samples": args.scaled_class_samples,
        "matched_boundary_triplets": args.matched_boundary_triplets,
        "hard_amber_fraction": args.hard_amber_fraction,
        "hard_red_fraction": args.hard_red_fraction,
        "replay_samples": len(replay),
        "image_size": list(image_size) if image_size else None,
        "epochs_requested": args.epochs,
        "max_steps_requested": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "class_token_loss_weight": args.class_token_loss_weight,
        "class_value_text": list(CLASS_VALUE_TEXT),
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "quantization": "bf16" if args.bf16_base else "nf4",
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "completion_only_loss": True,
        "vision_lora": not args.language_only_lora,
        "target_modules": target_modules,
        "source_adapter": args.adapter,
        "vision_lora_frozen": args.freeze_vision_lora,
        "vision_only_lora": args.vision_only_lora,
        "exact_production_prompt": args.exact_production_prompt,
        "evk_single_lane_prompt": args.evk_single_lane_prompt,
        "evk_speed_classifier": args.evk_speed_classifier,
        "speed_class_token_ids": speed_class_token_ids,
        "edge_first_fraction": args.edge_first_fraction,
        "edge_first_records": edge_first_count,
        "task_class_counts": {
            code: sum(row["code"] == code for row in task) for code in ("G", "A", "R")
        },
    })
    Path(args.output, "generative_training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
