#!/usr/bin/env python3
"""Build a class-balanced Reason2 vision-calibration manifest.

The parcel demo submits one still observation. Qwen's vision preprocessor
duplicates that image across the temporal patch, so calibration must use the
same layout instead of unrelated motion pairs. The output manifest records
that choice explicitly and is accepted only by the matching local quantizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


CLASSES = ("G", "A", "R")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    dataset: Path,
    output: Path,
    *,
    split: str,
    samples_per_class: int,
    seed: int,
) -> dict[str, object]:
    split_root = dataset.expanduser().resolve() / split
    if samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")

    rng = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    for label in CLASSES:
        class_root = split_root / label
        images = sorted(
            path
            for path in class_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(images) < samples_per_class:
            raise ValueError(
                f"{class_root} has {len(images)} images; "
                f"{samples_per_class} requested"
            )
        rng.shuffle(images)
        selected.extend((label, path) for path in images[:samples_per_class])

    # Interleave classes so a reduced --veg-num-samples remains balanced.
    by_class = {
        label: [path for item_label, path in selected if item_label == label]
        for label in CLASSES
    }
    pairs: list[dict[str, object]] = []
    for sample_index in range(samples_per_class):
        for label in CLASSES:
            image = by_class[label][sample_index]
            image_hash = sha256_file(image)
            pairs.append(
                {
                    "id": f"parcel-{label.lower()}-{sample_index:04d}",
                    "frames": [str(image), str(image)],
                    "timestamps_seconds": [0.0, 1.0],
                    "sha256": [image_hash, image_hash],
                    "source": {
                        "class": label,
                        "dataset": str(dataset.expanduser().resolve()),
                        "split": split,
                    },
                }
            )

    payload: dict[str, object] = {
        "schema_version": 1,
        "temporal_mode": "duplicate_single_frame",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "classes": list(CLASSES),
        "samples_per_class": samples_per_class,
        "pairs": pairs,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    payload = build_manifest(
        args.dataset,
        args.output,
        split=args.split,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(
        f"Prepared {len(payload['pairs'])} balanced duplicate-frame pairs: "
        f"{args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
