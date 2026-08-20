from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path

LABELS = {"G": "SUPPORTED", "A": "OVER_EDGE", "R": "FALLEN"}

SYSTEM_PROMPT = (
    "You are a visual warehouse inspector. Follow the user's monitored-object "
    "and ignore instructions. Judge only visible physical support. Return "
    "exactly G, A, or R."
)
USER_PROMPT = (
    "Inspect only cardboard shipping cartons in the monitored silver roller-"
    "conveyor cell. Ignore forklifts and anything carried by them, wooden "
    "pallets, workers, stack lights, shelves, tables, signs, and other "
    "non-target objects. G means every monitored carton has its complete "
    "bottom footprint on the metal rollers. A means a monitored carton still "
    "touches rollers but part of its bottom footprint hangs beyond a blue side "
    "rail. R means a monitored carton is completely off the rollers and rests "
    "on the floor. Classify the most unsafe monitored carton. Answer only G, "
    "A, or R."
)

# These are equivalent task instructions, not separate visual-label rules.
# A stable mix teaches the adapter to follow a caller-defined target/exclusion
# boundary instead of memorizing one exact paragraph or every possible piece
# of warehouse clutter.
USER_PROMPT_VARIANTS = (
    USER_PROMPT,
    (
        "Monitor cardboard parcels supported by the silver roller conveyor. "
        "Treat all other scene contents as context: do not classify people, "
        "vehicles or their loads, pallets, lights, furniture, storage, or "
        "signage. Return G when all monitored parcels are fully supported by "
        "rollers, A when one still touches rollers but overhangs a blue rail, "
        "and R when one is completely off the rollers on the floor. Report the "
        "worst visible state as exactly G, A, or R."
    ),
    (
        "The target objects are corrugated-cardboard shipping cartons in the "
        "roller-conveyor workcell. Ignore every non-target object, including "
        "forklifts, cargo on forks, workers, pallets and signal lamps. For the "
        "least-supported target carton answer G=fully on rollers, A=partly "
        "over a side rail while still touching rollers, or R=off the rollers "
        "and on the floor. Output one letter only."
    ),
    (
        "Classify visible physical support of conveyor cartons, not nearby "
        "warehouse activity. Objects that are not cardboard cartons belonging "
        "to the monitored rollers must not affect the answer. G: complete "
        "roller support. A: partial roller contact with bottom footprint beyond "
        "a blue rail. R: no roller support and resting on the floor. Use the "
        "most hazardous carton and answer only G, A, or R."
    ),
)


def prompt_for_sample(sample: dict[str, str]) -> str:
    """Choose an equivalent prompt reproducibly for a training sample."""
    image_key = sample.get("image_key", sample.get("image", ""))
    key = f"{image_key}|{sample.get('code', '')}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return USER_PROMPT_VARIANTS[index % len(USER_PROMPT_VARIANTS)]


def discover_samples(root: Path, split: str | None = None) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for code, label in LABELS.items():
        pattern = f"**/{split}/{code}/*.png" if split else f"**/{code}/*.png"
        for image in sorted(root.glob(pattern)):
            samples.append({"image": str(image), "code": code, "label": label})
    return samples


def parse_label(text: str) -> str | None:
    upper = text.upper().strip()
    if upper in LABELS:
        return LABELS[upper]
    matches = re.findall(r"SUPPORTED|OVER_EDGE|FALLEN", upper)
    return matches[-1] if matches else None


def per_class_metrics(rows: list[dict[str, str]]) -> dict:
    labels = list(LABELS.values())
    matrix = {truth: {pred: 0 for pred in labels + ["INVALID"]} for truth in labels}
    for row in rows:
        truth = row["truth"]
        pred = row.get("prediction") or "INVALID"
        matrix[truth][pred if pred in matrix[truth] else "INVALID"] += 1
    recalls = {}
    for truth in labels:
        total = sum(matrix[truth].values())
        recalls[truth] = matrix[truth][truth] / total if total else 0.0
    correct = sum(matrix[label][label] for label in labels)
    total = sum(sum(values.values()) for values in matrix.values())
    return {
        "count": total,
        "accuracy": correct / total if total else 0.0,
        "per_class_recall": recalls,
        "confusion_matrix": matrix,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
