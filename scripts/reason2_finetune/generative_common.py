from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from common import LABELS, discover_samples


SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. Follow the user's requested "
    "task, monitored-object boundary, exclusions, and output format. Base every "
    "answer on visible evidence rather than assumptions."
)

# This is the original full runtime contract. Keeping it verbatim in the
# training distribution makes the primary acceptance test honest: generation,
# parsing, and visual reasoning must all work without a classifier shortcut.
ORIGINAL_USER_PROMPT = (
    "You are a strict warehouse conveyor parcel-safety inspector. Inspect this "
    "single current image from one fixed camera. The monitored area is the "
    "complete silver U-shaped roller conveyor plus the floor immediately inside "
    "its U-shaped loop and within approximately one parcel length outside its "
    "rails. Inspect cardboard shipping parcels in that monitored area. Ignore "
    "forklift cargo, people, shelves, pallets, sorting tables, billboard, and "
    "boxes well outside this immediate conveyor area.\n\n"
    "Perform BOTH tests below on every image. Never skip either test and never "
    "return NOT_RUN.\n"
    "TEST 1 - FALLEN PARCEL. Set fallen_test to YES if any cardboard parcel in "
    "the monitored area is visibly (a) airborne away from the rollers, (b) below "
    "roller-top height after losing roller support, or (c) resting on the floor "
    "inside the U-loop or beside the conveyor's outer rail. A floor parcel in "
    "this immediate monitored area is fallen even if it is stationary and no "
    "earlier image is available. Do not reinterpret a parcel on the floor as "
    "safely supported by the conveyor. Otherwise set fallen_test to NO.\n"
    "TEST 2 - UNSTABLE PARCEL. Independently inspect every parcel that still "
    "touches the rollers. Set unstable_test to YES only if at least one such "
    "parcel is visibly in imminent danger of falling: roughly one third or more "
    "of its footprint extends beyond the supported roller surface, or it is "
    "visibly tipping over an outer edge. Do not count an already fallen parcel "
    "again in this test. Normal rotation through the U-curve, diagonal alignment, "
    "ordinary conveyor motion, crowding, and being near an edge while still "
    "fully supported are safe. Otherwise set unstable_test to NO.\n\n"
    "After recording both test answers, apply this strict priority:\n"
    "1. If fallen_test=YES: prediction_class_id=0, "
    "prediction_label=FALLEN_PARCEL, prediction_answer=RED. This rule wins "
    "regardless of unstable_test.\n"
    "2. Else if unstable_test=YES: prediction_class_id=1, "
    "prediction_label=UNSTABLE_PARCEL, prediction_answer=AMBER.\n"
    "3. Else: prediction_class_id=2, prediction_label=SAFE, "
    "prediction_answer=GREEN.\n"
    "Base both tests only on visible spatial evidence in this image. Do not "
    "require motion or an earlier image.\n\n"
    "Return exactly one JSON object and no markdown or extra text. The object "
    "must contain all six keys in this order: fallen_test, unstable_test, "
    "prediction_class_id, prediction_label, prediction_answer, visible_evidence. "
    "fallen_test and unstable_test must each be exactly YES or NO. "
    "visible_evidence must be one short sentence naming the parcel and concrete "
    "visible spatial evidence; use none when both tests are NO. A missing key, "
    "NOT_RUN, or an alternative-list value is invalid."
)

# Edge deployment retains the same physical reasoning contract and complete
# six-field generated answer, but places the actionable state first.  A
# streaming EVK caller may act after GREEN/AMBER/RED is complete while the
# model continues producing tests and evidence.  This is still autoregressive
# instruction following; it is not a classifier-logit shortcut.
EDGE_FIRST_USER_PROMPT = ORIGINAL_USER_PROMPT.replace(
    "The object must contain all six keys in this order: fallen_test, "
    "unstable_test, prediction_class_id, prediction_label, prediction_answer, "
    "visible_evidence. ",
    "The object must contain all six keys in this order: prediction_answer, "
    "fallen_test, unstable_test, prediction_class_id, prediction_label, "
    "visible_evidence. Put prediction_answer first so an edge controller can "
    "react as soon as GREEN, AMBER, or RED is complete, then finish the full "
    "object. ",
)

# Frozen prompt used by the static EVK graph for the exact-v1 production
# profile.  Keep this byte-for-byte aligned with BuildParcelInferencePrompt in
# QaiConveyorWorld.cpp so offline EVK acceptance replays exercise the same
# instruction contract as the packaged client.
EVK_SINGLE_LANE_USER_PROMPT = (
    "Inspect cardboard shipping cartons around the central silver straight "
    "roller-conveyor lane in this single image. Ignore the forklift and its "
    "cargo, wooden pallets, people, tables, shelves, signs, and the separate "
    "conveyor at the right edge.\n"
    "First inspect the gray floor immediately beside the central lane. "
    "fallen_test is YES if one or more cardboard cartons are visibly resting "
    "on that floor below roller-top height; a floor carton is fallen, never "
    "merely unstable. Otherwise fallen_test is NO.\n"
    "Then inspect cartons still touching the central lane's rollers. "
    "unstable_test is YES if any such carton has roughly one third or more of "
    "its footprint beyond a blue rail or is visibly tipping off; otherwise it "
    "is NO. Do not count a fallen floor carton in unstable_test.\n"
    "Return exactly one JSON object beginning with fallen_test and "
    "unstable_test. Both values must be YES or NO. No prose or markdown."
)

# Speed-v1 deployment contract.  The three answers are deliberately bare
# single tokenizer tokens in the Cosmos Reason2/Qwen3-VL vocabulary.  Keep the
# physical G/A/R boundary explicit, but omit JSON fields that the client can
# derive deterministically.  Like the accepted single-lane contract, this is a
# user-only prompt with no hidden system-message prefix.
EVK_SPEED_CLASSIFIER_USER_PROMPT = (
    "Classify cartons at the central silver conveyor lane. Ignore all other "
    "objects and lanes. Answer one letter only: G=fully supported; A=touching "
    "rollers but at least one-third beyond a blue rail or tipping; R=on the "
    "adjacent floor below the rollers."
)

# Candidate edge wording for the physically common case where a parcel has
# crossed a side rail but remains flat.  Rotation by itself remains safe when
# the complete footprint is still contained by the rails.
EVK_FLAT_OVERHANG_USER_PROMPT = (
    "Inspect cardboard shipping cartons around the central silver straight "
    "roller-conveyor lane in this single image. Ignore the forklift and its "
    "cargo, wooden pallets, people, tables, shelves, signs, and the separate "
    "conveyor at the right edge.\n"
    "First inspect the gray floor immediately beside the central lane. "
    "fallen_test is YES if one or more cardboard cartons are visibly resting "
    "on that floor below roller-top height; otherwise fallen_test is NO.\n"
    "Then inspect cartons still touching the central lane's rollers. "
    "unstable_test is YES if any part of a carton's bottom footprint extends "
    "outside either blue side rail. A carton can be unstable while still "
    "flat and horizontal; do not wait for it to tilt. A rotated carton is "
    "safe only when its complete footprint remains between the blue rails. "
    "Do not count a fallen floor carton in unstable_test.\n"
    "Return exactly one JSON object beginning with fallen_test and "
    "unstable_test. Both values must be YES or NO. No prose or markdown."
)

# Equivalent low-prefill contract for deployment experiments.  Keep the full
# six-field generative answer and all physical distinctions; remove repetition
# only.  This is not the production default until its held-out accuracy is
# measured and, if needed, it is included as a companion prompt during SFT.
COMPACT_USER_PROMPT = (
    "Inspect this current warehouse image as a strict parcel-safety inspector. "
    "Monitor cardboard parcels on the complete silver U-shaped roller conveyor "
    "and on the floor within one parcel length of its rails. Ignore forklifts "
    "and their cargo, people, pallets, shelves, tables, signs, and lights. Run "
    "both tests for every monitored parcel. fallen_test=YES if a parcel has "
    "lost roller support and is airborne, below roller-top height, or on the "
    "monitored floor. unstable_test=YES only if a parcel still touches rollers "
    "but is tipping over a rail or at least one third of its footprint is "
    "unsupported. Otherwise each test is NO. Fallen has priority. Map fallen "
    "to 0/FALLEN_PARCEL/RED, unstable to 1/UNSTABLE_PARCEL/AMBER, and otherwise "
    "2/SAFE/GREEN. Return only one JSON object with these keys in order: "
    "fallen_test, unstable_test, prediction_class_id, prediction_label, "
    "prediction_answer, visible_evidence. Use YES or NO for tests and one short "
    "visible-evidence sentence, or none when both tests are NO."
)

USER_PROMPT_VARIANTS = (
    ORIGINAL_USER_PROMPT,
    (
        "Inspect the current warehouse image for cardboard shipping parcels in "
        "the silver U-shaped roller-conveyor workcell and on the immediately "
        "adjacent floor. Ignore forklifts and their cargo, people, pallets, "
        "shelves, furniture, lights, and signs. Run both tests. fallen_test is "
        "YES when a monitored parcel has lost roller support and is airborne "
        "below roller height or on the nearby floor. unstable_test is YES when "
        "a parcel still touching rollers visibly has at least about one third of "
        "its footprint beyond a rail or is tipping off. Fallen wins over "
        "unstable. Return only JSON with fallen_test, unstable_test, "
        "prediction_class_id, prediction_label, prediction_answer, and "
        "visible_evidence. Map fallen to 0/FALLEN_PARCEL/RED, unstable to "
        "1/UNSTABLE_PARCEL/AMBER, and otherwise 2/SAFE/GREEN."
    ),
    (
        "Evaluate visible physical support of cardboard cartons belonging to "
        "the monitored roller conveyor. Treat every other object as irrelevant "
        "context, including vehicles, carried loads, workers, pallets, storage, "
        "lights, tables, and signage. A carton completely supported by rollers "
        "is safe. A carton still touching rollers but with roughly one third or "
        "more hanging beyond a rail is unstable. A carton off the rollers, below "
        "roller height, or on the nearby floor is fallen. Judge the worst carton "
        "and return the same six-field JSON safety record requested by the "
        "warehouse application."
    ),
)

STATE = {
    "G": {
        "fallen_test": "NO",
        "unstable_test": "NO",
        "prediction_class_id": 2,
        "prediction_label": "SAFE",
        "prediction_answer": "GREEN",
        # The production prompt explicitly requires this literal value when
        # both tests are NO. Preserve the contract instead of teaching a
        # superficially richer but invalid SAFE response.
        "evidence": "none",
    },
    "A": {
        "fallen_test": "NO",
        "unstable_test": "YES",
        "prediction_class_id": 1,
        "prediction_label": "UNSTABLE_PARCEL",
        "prediction_answer": "AMBER",
        "evidence": (
            "A monitored cardboard parcel still touches the rollers while a "
            "substantial part of its bottom footprint extends beyond a rail."
        ),
    },
    "R": {
        "fallen_test": "YES",
        "unstable_test": "NO",
        "prediction_class_id": 0,
        "prediction_label": "FALLEN_PARCEL",
        "prediction_answer": "RED",
        "evidence": (
            "A monitored cardboard parcel has lost roller support and is below "
            "roller-top height on the immediately adjacent floor."
        ),
    },
}

AUXILIARY_USER_PROMPT = (
    "Inspect only cardboard shipping parcels belonging to the silver roller "
    "conveyor and the immediately adjacent floor. Ignore forklifts and cargo "
    "on their forks, people, pallets, shelves, tables, signs, and lights. "
    "Explain the visible physical support of the least-safe monitored parcel, "
    "then conclude GREEN if fully roller-supported, AMBER if still touching "
    "rollers but tipping or substantially over a rail, or RED if it has lost "
    "roller support and is airborne below roller height or on the nearby floor."
)

AUXILIARY_RESPONSES = {
    "G": (
        "GREEN — SAFE. Every monitored cardboard parcel has its complete "
        "bottom footprint resting on the roller surface. No parcel is tipping "
        "over a rail, below roller-top height, or on the adjacent floor."
    ),
    "A": (
        "AMBER — UNSTABLE_PARCEL. A monitored cardboard parcel still contacts "
        "the rollers, but a substantial part of its bottom footprint extends "
        "past the supporting rail and it is in imminent danger of falling."
    ),
    "R": (
        "RED — FALLEN_PARCEL. A monitored cardboard parcel has lost roller "
        "support and is visibly below roller-top height on the immediately "
        "adjacent floor. The fallen condition takes priority over instability."
    ),
}

SUPPORT_BOUNDARY_USER_PROMPT = (
    "Inspect the least-safe cardboard parcel in the silver roller-conveyor "
    "workcell. Ignore forklifts, fork cargo, people, pallets, furniture, signs, "
    "and lights. Decide the physical support boundary: is the parcel STILL "
    "TOUCHING AND SUPPORTED BY ROLLERS while overhanging or tipping, or has it "
    "LOST ROLLER SUPPORT and dropped below roller-top height or onto the nearby "
    "floor? Do not call an overhanging parcel fallen while any meaningful part "
    "of its bottom still rests on rollers. Explain the visible contact evidence "
    "and conclude exactly AMBER for STILL TOUCHING or RED for LOST SUPPORT."
)

SUPPORT_BOUNDARY_RESPONSES = {
    "A": (
        "AMBER — STILL TOUCHING. The unstable parcel retains visible bottom "
        "contact with the roller surface. It overhangs substantially and may "
        "tip, but it has not dropped below roller-top height and is not on the "
        "floor, so it is not a fallen parcel."
    ),
    "R": (
        "RED — LOST SUPPORT. The fallen parcel no longer has meaningful bottom "
        "contact with the rollers. It is visibly below roller-top height or on "
        "the nearby floor, so it is fallen rather than merely overhanging."
    ),
}

STABILITY_BOUNDARY_USER_PROMPT = (
    "Inspect the least-safe cardboard parcel still at roller-top height in the "
    "central silver conveyor lane. Ignore people, forklifts and fork cargo, "
    "pallets, furniture, signs, lights, and cartons on other lanes. Decide the "
    "roller-support boundary: is the complete bottom footprint FULLY SUPPORTED "
    "by rollers, or does roughly one third or more extend beyond a blue rail "
    "while the parcel still touches rollers? A parcel may be strongly rotated "
    "or diagonal and still be GREEN when its complete footprint remains on the "
    "rollers. A parcel may remain level and horizontal and still be AMBER when "
    "a substantial part of its footprint overhangs the rail. Explain the visible "
    "support relation and conclude exactly GREEN for FULLY SUPPORTED or AMBER "
    "for SUBSTANTIAL OVERHANG."
)

STABILITY_BOUNDARY_RESPONSES = {
    "G": (
        "GREEN — FULLY SUPPORTED. The parcel may be rotated, but its complete "
        "bottom footprint remains inside the blue rails on the roller surface; "
        "no substantial part extends unsupported beyond an edge."
    ),
    "A": (
        "AMBER — SUBSTANTIAL OVERHANG. The parcel remains level and still "
        "touches the rollers, but roughly one third or more of its bottom "
        "footprint extends unsupported beyond a blue rail."
    ),
}


def prompt_for_sample(sample: dict[str, str]) -> str:
    """Use the exact production prompt for half the corpus and held-out-safe
    equivalent wording for the remainder.
    """
    key = f"{sample.get('image_key', sample['image'])}|prompt".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    if value % 2 == 0:
        return ORIGINAL_USER_PROMPT
    return USER_PROMPT_VARIANTS[1 + value % (len(USER_PROMPT_VARIANTS) - 1)]


def response_for_code(code: str) -> str:
    state = STATE[code]
    result = {
        "fallen_test": state["fallen_test"],
        "unstable_test": state["unstable_test"],
        "prediction_class_id": state["prediction_class_id"],
        "prediction_label": state["prediction_label"],
        "prediction_answer": state["prediction_answer"],
        "visible_evidence": state["evidence"],
    }
    return json.dumps(result, separators=(",", ":"))


def edge_first_response_for_code(code: str) -> str:
    state = STATE[code]
    result = {
        "prediction_answer": state["prediction_answer"],
        "fallen_test": state["fallen_test"],
        "unstable_test": state["unstable_test"],
        "prediction_class_id": state["prediction_class_id"],
        "prediction_label": state["prediction_label"],
        "visible_evidence": state["evidence"],
    }
    return json.dumps(result, separators=(",", ":"))


def evk_single_lane_response_for_code(code: str) -> str:
    """Return only the two observations consumed by the EVK client.

    The packaged client deterministically derives GREEN/AMBER/RED and the
    display label from these tests.  Training the static edge decoder to emit
    extra redundant fields wastes its short token budget and creates an
    avoidable prompt/response mismatch with the deployed contract.
    """
    state = STATE[code]
    result = {
        "fallen_test": state["fallen_test"],
        "unstable_test": state["unstable_test"],
    }
    return json.dumps(result, separators=(",", ":"))


def evk_speed_classifier_response_for_code(code: str) -> str:
    """Return the exact one-token completion used by speed-v1."""
    if code not in STATE:
        raise ValueError(f"Unknown parcel state code: {code!r}")
    return code


def use_edge_first_contract(record: dict) -> None:
    """Mutate one production task record to the streaming-safe contract."""
    record["prompt"][-1]["content"] = EDGE_FIRST_USER_PROMPT
    record["completion"][0]["content"] = edge_first_response_for_code(
        record["code"]
    )


def use_evk_single_lane_contract(record: dict) -> None:
    """Mutate a task record to the exact short production EVK contract."""
    # Both QaiConveyorWorld.cpp and validate_evk_http_service.py submit a
    # single user message. Do not retain the generic SFT system prefix here:
    # a static decoder is sensitive to that otherwise invisible token shift.
    record["prompt"] = [
        {"role": "user", "content": EVK_SINGLE_LANE_USER_PROMPT}
    ]
    record["completion"][0]["content"] = evk_single_lane_response_for_code(
        record["code"]
    )


def use_evk_speed_classifier_contract(record: dict) -> None:
    """Mutate a task record to the user-only, one-token speed-v1 contract."""
    record["prompt"] = [
        {"role": "user", "content": EVK_SPEED_CLASSIFIER_USER_PROMPT}
    ]
    record["completion"][0]["content"] = (
        evk_speed_classifier_response_for_code(record["code"])
    )


def task_record(sample: dict[str, str]) -> dict:
    return {
        "images": [sample["image"]],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for_sample(sample)},
        ],
        "completion": [
            {"role": "assistant", "content": response_for_code(sample["code"])}
        ],
        "source": "unreal_conveyor",
        "code": sample["code"],
        "label": sample["label"],
    }


def auxiliary_task_record(record: dict) -> dict:
    """Create a class-distinct generated-answer companion for a JSON record.

    The six-field JSON answers share most tokens across classes. This companion
    prevents completion loss from being minimized by learning only the common
    schema and a SAFE prior, without introducing a direct-logit classifier.
    """
    return {
        "images": list(record["images"]),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": AUXILIARY_USER_PROMPT},
        ],
        "completion": [
            {"role": "assistant", "content": AUXILIARY_RESPONSES[record["code"]]}
        ],
        "source": "unreal_conveyor_auxiliary_reasoning",
        "code": record["code"],
        "label": record["label"],
    }


def support_boundary_task_record(record: dict) -> dict:
    """Teach the difficult AMBER-versus-RED contact distinction symmetrically.

    Pairing equal A and R records avoids changing the class prior while making
    the decisive visual relation occupy a much larger share of completion loss.
    """
    if record["code"] not in SUPPORT_BOUNDARY_RESPONSES:
        raise ValueError("Support-boundary records require A or R examples")
    return {
        "images": list(record["images"]),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SUPPORT_BOUNDARY_USER_PROMPT},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": SUPPORT_BOUNDARY_RESPONSES[record["code"]],
            }
        ],
        "source": "unreal_conveyor_support_boundary",
        "code": record["code"],
        "label": record["label"],
    }


def stability_boundary_task_record(record: dict) -> dict:
    """Teach GREEN versus AMBER without using tilt as the class cue.

    Equal G and A companions keep the binary contrast prior balanced while
    spelling out that yaw is safe and a level partial overhang is not.
    """
    if record["code"] not in STABILITY_BOUNDARY_RESPONSES:
        raise ValueError("Stability-boundary records require G or A examples")
    return {
        "images": list(record["images"]),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": STABILITY_BOUNDARY_USER_PROMPT},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": STABILITY_BOUNDARY_RESPONSES[record["code"]],
            }
        ],
        "source": "unreal_conveyor_stability_boundary",
        "code": record["code"],
        "label": record["label"],
    }


def discover_task_records(root: Path, split: str) -> list[dict]:
    return [task_record(sample) for sample in discover_samples(root, split)]


def extract_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from plain or reasoning-tag output."""
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
    return None


def prediction_from_response(text: str) -> tuple[str | None, dict | None]:
    direct_code = text.strip().upper()
    if direct_code in STATE:
        state = STATE[direct_code]
        parsed = {
            "fallen_test": state["fallen_test"],
            "unstable_test": state["unstable_test"],
            "prediction_class_id": state["prediction_class_id"],
            "prediction_label": state["prediction_label"],
            "prediction_answer": state["prediction_answer"],
        }
        return LABELS[direct_code], parsed

    parsed = extract_json_object(text)
    if not parsed:
        return None, None
    answer = str(parsed.get("prediction_answer", "")).upper()
    code = {"GREEN": "G", "AMBER": "A", "RED": "R"}.get(answer)
    if not code:
        # The EVK production contract intentionally emits only the two tests;
        # the client derives the redundant class fields with fallen priority.
        fallen = str(parsed.get("fallen_test", "")).strip().upper()
        unstable = str(parsed.get("unstable_test", "")).strip().upper()
        if fallen == "YES":
            code = "R"
        elif fallen == "NO" and unstable == "YES":
            code = "A"
        elif fallen == "NO" and unstable == "NO":
            code = "G"
    return (LABELS.get(code) if code else None), parsed
