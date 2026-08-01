#!/usr/bin/env python3
"""Request a bounded warehouse advisory for an Isaac Sim roof-camera clip.

The current live mode gives Cosmos a generic all-actor safety-supervisor role
and only identifies actors that the supervisor is allowed to command. Output
is constrained to a complete allow-listed actor/action/route command. The
older scenario-specific closed-choice modes remain available for reproducing
the earlier evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


MODEL = "local/cosmos-reason2-2b:Q4_0"
AB_GRAMMAR = "root ::= [AB]"
ALLOWED_ACTIONS = {"CONTINUE", "YIELD", "STOP", "REROUTE"}
ALLOWED_ROUTES = {
    "current",
    "direct",
    "hold",
    "hold_west_gate",
    "south_bypass",
    "north_bypass",
}
GENERIC_COMMANDS = {
    "RobotBlue CONTINUE CURRENT_ROUTE": ("CONTINUE", "current"),
    "RobotBlue YIELD HOLD": ("YIELD", "hold"),
    "RobotBlue STOP HOLD": ("STOP", "hold"),
    "RobotBlue REROUTE NORTH_BYPASS": ("REROUTE", "north_bypass"),
    "RobotBlue REROUTE SOUTH_BYPASS": ("REROUTE", "south_bypass"),
}
GENERIC_COMMAND_GRAMMAR = (
    'root ::= "RobotBlue CONTINUE CURRENT_ROUTE"'
    ' | "RobotBlue YIELD HOLD"'
    ' | "RobotBlue STOP HOLD"'
    ' | "RobotBlue REROUTE NORTH_BYPASS"'
    ' | "RobotBlue REROUTE SOUTH_BYPASS"'
)
VISION_ONLY_TUTORIAL_COMMANDS = {
    "RobotBlue CONTINUE_CURRENT_ROUTE": ("CONTINUE", "current"),
    "RobotBlue REROUTE_NORTH_BYPASS": ("REROUTE", "north_bypass"),
}
VISION_ONLY_TUTORIAL_GRAMMAR = (
    'root ::= "RobotBlue CONTINUE_CURRENT_ROUTE"'
    ' | "RobotBlue REROUTE_NORTH_BYPASS"'
)
VISUAL_GAP_GRAMMAR = VISION_ONLY_TUTORIAL_GRAMMAR
VISUAL_GAP_ANSWERS = set(VISION_ONLY_TUTORIAL_COMMANDS)
PASSAGE_NAMES = ("left", "right")
PASSAGE_STATE_OPTIONS = (
    {"status": "CLEAR", "offender": "NONE"},
    {"status": "NOT_VISIBLE", "offender": "NONE"},
    {"status": "UNCERTAIN", "offender": "UNKNOWN"},
    {"status": "BLOCKAGE_IMMINENT", "offender": "FORKLIFT"},
    {"status": "BLOCKAGE_IMMINENT", "offender": "MOBILE_ROBOT"},
    {"status": "BLOCKAGE_IMMINENT", "offender": "UNKNOWN"},
    {"status": "BLOCKED", "offender": "FORKLIFT"},
    {"status": "BLOCKED", "offender": "MOBILE_ROBOT"},
    {"status": "BLOCKED", "offender": "UNKNOWN"},
)
PASSAGE_CONDITION_OBJECTS = tuple(
    {
        "left_passage": left_state,
        "right_passage": right_state,
    }
    for left_state in PASSAGE_STATE_OPTIONS
    for right_state in PASSAGE_STATE_OPTIONS
)
PASSAGE_CONDITION_REPORTS = tuple(
    json.dumps(report, separators=(",", ":"))
    for report in PASSAGE_CONDITION_OBJECTS
)
PASSAGE_CONDITION_GRAMMAR = "root ::= " + " | ".join(
    json.dumps(report) for report in PASSAGE_CONDITION_REPORTS
)
PASSAGE_CONDITION_ANSWERS = set(PASSAGE_CONDITION_REPORTS)


def _derived_passage_collections(
    condition: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    blocked: list[str] = []
    imminent: list[dict[str, str]] = []
    for passage_name in PASSAGE_NAMES:
        state = condition[f"{passage_name}_passage"]
        status = state["status"]
        offender = state["offender"].lower()
        if status == "BLOCKED":
            blocked.append(passage_name)
        elif status == "BLOCKAGE_IMMINENT":
            imminent.append(
                {"name": passage_name, "offender": offender}
            )
    return blocked, imminent


PASSAGE_STATUS_OBJECTS = tuple(
    {
        "traffic_actors": list(traffic_actors),
        "passage_states": condition,
        "passages_blocked": _derived_passage_collections(condition)[0],
        "passages_blockage_imminent": (
            _derived_passage_collections(condition)[1]
        ),
    }
    for traffic_actors in (
        (),
        ("mobile_robot",),
        ("forklift",),
        ("mobile_robot", "forklift"),
    )
    for condition in PASSAGE_CONDITION_OBJECTS
)
PASSAGE_STATUS_REPORTS = tuple(
    json.dumps(report, separators=(",", ":"))
    for report in PASSAGE_STATUS_OBJECTS
)
PASSAGE_STATUS_GRAMMAR = "root ::= " + " | ".join(
    json.dumps(report) for report in PASSAGE_STATUS_REPORTS
)
PASSAGE_STATUS_ANSWERS = set(PASSAGE_STATUS_REPORTS)
VISION_INPUT_CHOICES = {"temporal_pair", "rolling_video"}
GENERIC_CONFLICT_GRAMMAR = 'root ::= "NO_CONFLICT" | "CONFLICT"'
GENERIC_INTERVENTION_COMMANDS = {
    "RobotBlue YIELD HOLD": ("YIELD", "hold"),
    "RobotBlue STOP HOLD": ("STOP", "hold"),
    "RobotBlue REROUTE NORTH_BYPASS": ("REROUTE", "north_bypass"),
    "RobotBlue REROUTE SOUTH_BYPASS": ("REROUTE", "south_bypass"),
}
GENERIC_INTERVENTION_GRAMMAR = (
    'root ::= "RobotBlue YIELD HOLD"'
    ' | "RobotBlue STOP HOLD"'
    ' | "RobotBlue REROUTE NORTH_BYPASS"'
    ' | "RobotBlue REROUTE SOUTH_BYPASS"'
)
DEFAULT_NAVIGATION_INTENTION = (
    "RobotBlue is navigating east through the direct aisle to the loading goal"
)

SCENARIOS = {
    "mixed_traffic": {
        "hazard_prompt": (
            "Watch the complete fixed roof-camera warehouse clip. Which "
            "description is directly visible? Answer one letter only: "
            "A) RobotBlue's direct route stays clear for the complete clip; "
            "B) a forklift, worker, or opposing robot repeatedly conflicts "
            "with RobotBlue's direct route."
        ),
        "hazard_answer": "B",
        "hazard": "mixed_traffic_conflict",
        "route_prompt": (
            "After waiting for the immediate conflicts, which route does "
            "RobotBlue visibly use around the central shelf row? Answer one "
            "letter only: A) the lower/south bypass; B) the upper/north bypass."
        ),
        "routes": {"A": "south_bypass", "B": "north_bypass"},
        "oracle_route": "south_bypass",
    },
    "aisle_congestion": {
        "hazard_prompt": (
            "Watch the complete fixed roof-camera warehouse clip. Which "
            "description is directly visible? Answer one letter only: "
            "A) RobotBlue's direct aisle remains clear; B) another autonomous "
            "robot enters or occupies RobotBlue's direct aisle."
        ),
        "hazard_answer": "B",
        "hazard": "aisle_congestion",
        "route_prompt": (
            "Which route does RobotBlue visibly use to pass the central shelf "
            "row? Answer one letter only: A) the lower/south bypass; "
            "B) the upper/north bypass."
        ),
        "routes": {"A": "south_bypass", "B": "north_bypass"},
        "oracle_route": "south_bypass",
    },
    "blind_corner": {
        "hazard_prompt": (
            "Watch the complete fixed roof-camera warehouse clip. Which "
            "description is directly visible? Answer one letter only: "
            "A) a forklift emerges from behind the shelf row into RobotBlue's "
            "direct route; B) no forklift enters RobotBlue's direct route."
        ),
        "hazard_answer": "A",
        "hazard": "blind_corner_forklift",
        "route_prompt": (
            "Which route does RobotBlue visibly use to pass the central shelf "
            "row? Answer one letter only: A) the upper/north bypass; "
            "B) the lower/south bypass."
        ),
        "routes": {"A": "north_bypass", "B": "south_bypass"},
        "oracle_route": "north_bypass",
    },
}

UrlOpen = Callable[..., Any]


def generic_supervisor_prompt(
    commandable_actors: tuple[str, ...] = ("RobotBlue",),
) -> str:
    """Build the exact bounded prompt displayed by the Isaac UI."""
    if commandable_actors != ("RobotBlue",):
        raise ValueError(
            "This milestone currently supports exactly one commandable actor: "
            "RobotBlue"
        )
    actor_text = ", ".join(commandable_actors)
    commands = "\n".join(GENERIC_COMMANDS)
    return (
        "Act as the fixed overhead safety supervisor for all visible warehouse "
        "traffic. Observe every visible autonomous robot, forklift, vehicle, "
        "worker, carried load, shelf, pallet, and obstruction. Assess the "
        "complete clip in chronological order and infer motion only from visible "
        "change across frames. Static shelves, cartons, pallets, painted routes, "
        "and stationary actors are normal map context; their presence alone is "
        "never a reason to intervene; do not intervene solely because stationary "
        "objects or actors are visible. "
        "Use this general decision order: (1) if no moving actor is approaching "
        "a conflict with a commandable actor, CONTINUE CURRENT_ROUTE; (2) if an "
        "immediate conflict exists but no complete bypass is visibly verified "
        "open, YIELD or STOP; (3) REROUTE only when motion shows the actor's current "
        "direction becoming blocked, and then select a complete physical route that "
        "is visibly open. Look for emerging congestion, blocked passages, collisions, "
        "near misses, unsafe following, and interactions that may become unsafe. "
        "Do not assume which actor causes a problem. In this fixed map-aligned "
        "roof view, image-top is north and image-bottom is south. NORTH_BYPASS "
        "means trafficable space around the upper/north side of the shelf rows; "
        "SOUTH_BYPASS means trafficable space around the lower/south side. No "
        "route illustrations are present in the sensor input; judge physical "
        "clearance from the warehouse image itself. "
        f"Commandable actors currently visible: {actor_text}. "
        "You may issue a command only to an actor in that list. If its motion "
        "cannot reduce a risk involving non-commandable actors, do not invent an "
        "intervention. Select the lowest-intervention safe command. Reply with "
        "exactly one of these complete lines and no explanation:\n"
        f"{commands}"
    )


def vision_only_perception_prompt(camera_layout: str = "single") -> str:
    """Return the camera-only visual report prompt shared by both scenes."""
    if camera_layout not in ("single", "grid"):
        raise ValueError("camera_layout must be single or grid")
    grid_context = (
        """\
If a frame is a synchronized 2-by-2 camera grid, top-left is BLIND CORNER,
top-right is WEST APPROACH, bottom-left is ROOF OVERVIEW, and bottom-right is
NORTH AISLE. Count the same actor shown in multiple panels only once. Use the
bottom-left overview for mutual positions and path overlap; use top panels for
actor identity and motion detail.
"""
        if camera_layout == "grid"
        else ""
    )
    return """\
You are inspecting warehouse traffic using only fixed safety-camera pixels.
The input may be a chronological video, or a temporal comparison with EARLIER
on the left and NOW on the right. In a comparison, match the same physical
actor across panels, count it once, and compare distances within each panel,
never across the panel boundary. Describe only visible evidence. Do not assume
a task, destination, cooperation, or intent.
""" + grid_context + """
Known visual identity binding: commandable RobotBlue is the compact,
low-profile white autonomous mobile robot. A forklift is a larger industrial
counterbalance vehicle with long front forks, an upright mast, and an operator
cage. These bindings identify appearance only; other actors may also be
present. Do not assume that a non-commandable actor is visible.

Return exactly four short labeled lines with no panel-by-panel prose:
TRAFFIC_ACTORS: each distinct robot, forklift, vehicle, or person, counted once
CLOSEST_GAP: for every distinct actor pair, write
"actor A to actor B = NOW" when the right panel has the shorter nearest-body
gap, "= EARLIER" when the left panel has the shorter gap, "= SAME" when equal,
or "= UNCERTAIN" when unclear. A pair must be individually visible in both
panels before selecting NOW, EARLIER, or SAME; otherwise select UNCERTAIN.
Write NONE if fewer than two actors are visible
CONFLICT_GEOMETRY: YES only when CLOSEST_GAP is NOW and the pair visibly shares
a passage; otherwise NO or UNCERTAIN
SCENE_ELEMENTS: important visible shelves/racks, cartons, pallets,
cones/barriers, walls, and aisles
"""


def vision_only_command_prompt(visual_report: str) -> str:
    """Build the bounded command prompt from Reason2's own camera report."""
    return (
        "Convert a camera report into one warehouse-traffic command. Use only "
        "the report. First count distinct traffic actors in ACTORS. Shelves, "
        "pallets, cartons, cones, floors, and walls are never traffic actors.\n"
        "RULE 1: zero or one traffic actor -> RobotBlue "
        "CONTINUE_CURRENT_ROUTE, regardless of distance or overlap with static "
        "objects.\n"
        "RULE 2: two or more traffic actors -> RobotBlue "
        "REROUTE_NORTH_BYPASS only if CLOSEST_GAP selects NOW/final for those "
        "actors, those actors approach, their mutual distance "
        "decreases, their traffic paths overlap, or one occupies the white "
        "robot/AGV's passage. Otherwise CONTINUE.\n"
        "EXAMPLE CLEAR: ACTORS: one white robot. DISTANCE: robot approaches a "
        "shelf. PATH_OVERLAP: robot overlaps the shelf area.\n"
        "COMMAND: RobotBlue CONTINUE_CURRENT_ROUTE\n"
        "EXAMPLE HAZARD: TRAFFIC_ACTORS: one white robot and one forklift. "
        "CLOSEST_GAP: white robot to forklift = NOW. "
        "CONFLICT_GEOMETRY: YES.\n"
        "COMMAND: RobotBlue REROUTE_NORTH_BYPASS\n"
        "Evaluate the report below. Reply with exactly one allowed line.\n"
        "VISUAL_REPORT:\n"
        + visual_report
        + "\nALLOWED:\n"
        "RobotBlue CONTINUE_CURRENT_ROUTE\n"
        "RobotBlue REROUTE_NORTH_BYPASS"
    )


def vision_only_tutorial_prompt(camera_layout: str = "single") -> str:
    """Return the identical two-stage camera-only template shown in the UI."""
    return (
        "STAGE 1 — CAMERA REPORT\n"
        + vision_only_perception_prompt(camera_layout)
        + "\n\nSTAGE 2 — BOUNDED COMMAND\n"
        + vision_only_command_prompt("{{VISUAL_REPORT_FROM_STAGE_1}}")
    )


def visual_report_supports_reroute(visual_report: str) -> bool:
    """Conservatively validate a two-actor conflict from model-produced text."""
    robot = r"(?:robot(?:blue)?|agv|amr|white (?:robot|vehicle|object|unit))"
    forklift = r"(?:forklift)"
    raw_report = visual_report.lower()
    actors_match = re.search(
        r"(?:traffic_actors|actors)\s*:\s*([^\r\n]+)",
        raw_report,
    )
    if actors_match is None:
        return False
    actor_inventory = actors_match.group(1)
    if (
        re.search(robot, actor_inventory) is None
        or re.search(forklift, actor_inventory) is None
    ):
        return False
    report = " ".join(raw_report.split())

    linked_patterns = (
        rf"{robot}.{{0,100}}(?:approach\w*|towards?|converg\w*).{{0,60}}{forklift}",
        rf"{forklift}.{{0,100}}(?:approach\w*|towards?|converg\w*).{{0,60}}{robot}",
        rf"distance between.{{0,60}}{robot}.{{0,60}}{forklift}.{{0,40}}decreas\w*",
        rf"distance between.{{0,60}}{forklift}.{{0,60}}{robot}.{{0,40}}decreas\w*",
        rf"{robot}.{{0,100}}path\w*.{{0,40}}overlap\w*.{{0,80}}{forklift}",
        rf"{forklift}.{{0,100}}path\w*.{{0,40}}overlap\w*.{{0,80}}{robot}",
        rf"{robot}.{{0,100}}(?:passage|aisle).{{0,50}}(?:occup\w*|block\w*).{{0,80}}{forklift}",
        rf"{forklift}.{{0,100}}(?:occup\w*|block\w*).{{0,50}}{robot}.{{0,40}}(?:passage|aisle)",
        rf"(?:gap|distance).{{0,80}}{robot}.{{0,80}}{forklift}.{{0,80}}(?:shorter|smaller|closer).{{0,40}}(?:now|final|later|new)",
        rf"(?:gap|distance).{{0,80}}{forklift}.{{0,80}}{robot}.{{0,80}}(?:shorter|smaller|closer).{{0,40}}(?:now|final|later|new)",
        rf"{robot}.{{0,80}}(?:and|to|-).{{0,30}}{forklift}.{{0,80}}(?:gap|distance).{{0,40}}(?:shorter|smaller).{{0,30}}(?:now|final|later)",
        rf"{forklift}.{{0,80}}(?:and|to|-).{{0,30}}{robot}.{{0,80}}(?:gap|distance).{{0,40}}(?:shorter|smaller).{{0,30}}(?:now|final|later)",
        rf"closest_gap\s*:.{{0,80}}{robot}.{{0,40}}(?:to|-).{{0,20}}{forklift}\s*=\s*now",
        rf"closest_gap\s*:.{{0,80}}{forklift}.{{0,40}}(?:to|-).{{0,20}}{robot}\s*=\s*now",
        rf"(?:^|\s)gap\s*:.{{0,60}}{robot}.{{0,30}}(?:to|-).{{0,20}}{forklift}\s*=\s*now",
        rf"(?:^|\s)gap\s*:.{{0,60}}{forklift}.{{0,30}}(?:to|-).{{0,20}}{robot}\s*=\s*now",
    )
    if any(re.search(pattern, report) for pattern in linked_patterns):
        return True

    relational_patterns = (
        r"(?:their|both actors['’]?)\s+(?:traffic\s+)?paths?\s+overlap",
        r"distance between (?:them|the two actors).{0,30}decreas",
        r"(?:they|both actors).{0,30}(?:approach each other|converg)",
        r"(?:pair|mutual|their).{0,30}(?:gap|distance).{0,30}(?:shorter|smaller).{0,20}(?:now|final|later)",
    )
    return any(re.search(pattern, report) for pattern in relational_patterns)


def generic_staged_supervisor_prompts(
    commandable_actors: tuple[str, ...] = ("RobotBlue",),
) -> dict[str, str]:
    """Build the short conflict and intervention prompts used by live mode."""
    if commandable_actors != ("RobotBlue",):
        raise ValueError(
            "This milestone currently supports exactly one commandable actor: "
            "RobotBlue"
        )
    intervention_commands = "\n".join(GENERIC_INTERVENTION_COMMANDS)
    return {
        "conflict": (
            "Act as the fixed overhead safety supervisor for all visible "
            "warehouse traffic. Watch the complete clip in chronological order "
            "and infer motion only from visible change. The only commandable "
            "actor is RobotBlue, the compact white autonomous mobile robot; "
            "\"Blue\" is its control ID, not its rendered paint color. The "
            "large dark industrial vehicle at the right-side blind corner is "
            "a forklift, although it may blend into shadow. Decide "
            "whether a moving robot, forklift, vehicle, worker, or carried load "
            "is approaching, crossing, or blocking RobotBlue's path closely "
            "enough to create emerging congestion, a "
            "near miss, or a collision risk. Static shelves, cartons, pallets, "
            "and walls are map context and never establish a traffic conflict by "
            "themselves. A stopped vehicle or worker is a conflict only when "
            "RobotBlue is visibly approaching its occupied passage. Reply with "
            "exactly one label: NO_CONFLICT means no "
            "moving actor is approaching a conflict with RobotBlue, so it may "
            "continue its current route; CONFLICT means visible motion is "
            "creating or about to create a conflict with RobotBlue."
        ),
        "intervention": (
            "The preceding review of this same roof-camera clip established a "
            "moving conflict involving RobotBlue. Select the lowest-intervention "
            "safe command from physical clearance visible in the images. "
            "Image-top is north and image-bottom is south. NORTH_BYPASS means "
            "trafficable space around the upper side of the shelf rows; "
            "SOUTH_BYPASS means trafficable space around the lower side. No route "
            "illustrations are present in the sensor input. Choose YIELD or STOP "
            "if no complete bypass is visibly open. Reply with exactly one line "
            "and no explanation:\n"
            f"{intervention_commands}"
        ),
    }


def _completion_url(server_url: str) -> str:
    return f"{server_url.rstrip('/')}/v1/chat/completions"


def _request_constrained_text(
    *,
    server_url: str,
    model: str,
    video_url: str | None,
    prompt: str,
    grammar_string: str,
    allowed_answers: set[str],
    max_completion_tokens: int,
    timeout_seconds: float,
    urlopen: UrlOpen = urllib.request.urlopen,
    answer_normalizer: Callable[[str], str] | None = None,
) -> tuple[str, float]:
    content: list[dict[str, Any]] = []
    if video_url is not None:
        content.append({"type": "image_url", "image_url": {"url": video_url}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_completion_tokens": max_completion_tokens,
        "enable_think": False,
        "top_k": 1,
        "temperature": 0,
        "seed": 42,
        "grammar_string": grammar_string,
    }
    request = urllib.request.Request(
        _completion_url(server_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
        if len(response_body) > 3000:
            response_body = response_body[-3000:]
        raise RuntimeError(
            f"GenieX HTTP {exc.code}"
            + (f": {response_body}" if response_body else "")
        ) from exc
    elapsed = time.monotonic() - started

    try:
        answer = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("GenieX response has no assistant content") from exc
    if answer_normalizer is not None:
        answer = answer_normalizer(answer)
    if answer not in allowed_answers:
        raise ValueError(f"GenieX returned an out-of-grammar answer: {answer!r}")
    return answer, elapsed


def _request_free_text(
    *,
    server_url: str,
    model: str,
    video_url: str,
    prompt: str,
    max_completion_tokens: int,
    timeout_seconds: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> tuple[str, float]:
    """Request a deterministic visual report without an output grammar."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_completion_tokens": max_completion_tokens,
        "enable_think": False,
        "top_k": 1,
        "temperature": 0,
        "repeat_penalty": 1.15,
        "seed": 42,
    }
    request = urllib.request.Request(
        _completion_url(server_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
        if len(response_body) > 3000:
            response_body = response_body[-3000:]
        raise RuntimeError(
            f"GenieX HTTP {exc.code}"
            + (f": {response_body}" if response_body else "")
        ) from exc
    elapsed = time.monotonic() - started
    try:
        answer = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("GenieX response has no assistant content") from exc
    if not answer:
        raise ValueError("GenieX returned an empty visual report")
    return answer, elapsed


def _request_letter(
    *,
    server_url: str,
    model: str,
    video_url: str,
    prompt: str,
    timeout_seconds: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> tuple[str, float]:
    return _request_constrained_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=prompt,
        grammar_string=AB_GRAMMAR,
        allowed_answers={"A", "B"},
        max_completion_tokens=4,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )


def _validated_advisory(
    *,
    scenario: str,
    action: str,
    hazard: str,
    route: str,
) -> dict[str, Any]:
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Policy gateway rejected action: {action!r}")
    if route not in ALLOWED_ROUTES:
        raise ValueError(f"Policy gateway rejected route: {route!r}")
    reasons = {
        "CONTINUE": "No intervention is required; continue the current route",
        "YIELD": "Yield while the visible traffic interaction develops",
        "STOP": "Stop because an immediate unsafe interaction is visible",
        "REROUTE": "Use the selected bypass to reduce congestion or collision risk",
    }
    return {
        "schema_version": "1.0",
        "source": "roof_overview",
        "recipient": "RobotBlue",
        "action": action,
        "hazard": hazard,
        "route": route,
        "reason": reasons[action],
        "ttl_ms": 1500,
        "scenario": scenario,
    }


def request_advisory(
    *,
    scenario: str,
    video_url: str,
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    try:
        spec = SCENARIOS[scenario]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario: {scenario!r}") from exc

    hazard_answer, hazard_seconds = _request_letter(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=spec["hazard_prompt"],
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    answers = {"hazard": hazard_answer, "route": None}
    timings = [round(hazard_seconds, 6)]

    if hazard_answer == spec["hazard_answer"]:
        route_answer, route_seconds = _request_letter(
            server_url=server_url,
            model=model,
            video_url=video_url,
            prompt=spec["route_prompt"],
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
        answers["route"] = route_answer
        timings.append(round(route_seconds, 6))
        route = spec["routes"][route_answer]
        advisory = _validated_advisory(
            scenario=scenario,
            action="REROUTE",
            hazard=spec["hazard"],
            route=route,
        )
    else:
        advisory = _validated_advisory(
            scenario=scenario,
            action="CONTINUE",
            hazard="none",
            route="direct",
        )

    oracle_action = "REROUTE"
    oracle_route = spec["oracle_route"]
    return {
        "model": model,
        "video_url": video_url,
        "closed_choice_answers": answers,
        "request_seconds": timings,
        "total_seconds": round(sum(timings), 6),
        "advisory": advisory,
        "scenario_oracle": {
            "action": oracle_action,
            "route": oracle_route,
            "match": (
                advisory["action"] == oracle_action
                and advisory["route"] == oracle_route
            ),
        },
        "scope": "task-specific closed-choice visual advisory",
    }


def request_live_aisle_advisory(
    *,
    video_url: str,
    navigation_intention: str = DEFAULT_NAVIGATION_INTENTION,
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Detect one live aisle conflict and map it to a bounded route command.

    The first milestone has one modeled alternative, ``south_bypass``. Keeping
    route selection deterministic removes a second VLM round trip: Cosmos
    detects the emerging conflict, while the policy gateway owns the route.
    """
    intention = navigation_intention.strip()
    if not intention or len(intention) > 500:
        raise ValueError(
            "navigation_intention must contain between 1 and 500 characters"
        )
    prompt = (
        "You are the roof-camera supervisor for RobotBlue. "
        f"Current navigation intention: {intention}. "
        "Watch the complete clip in chronological order. Do you see a problem "
        "ahead in RobotBlue's current aisle? Answer one letter only: "
        "A) the direct aisle ahead remains clear; "
        "B) a forklift is entering or occupying the same narrow aisle, so the "
        "vehicles cannot safely pass."
    )
    answer, request_seconds = _request_letter(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    if answer == "B":
        advisory = _validated_advisory(
            scenario="aisle_congestion",
            action="REROUTE",
            hazard="aisle_congestion",
            route="south_bypass",
        )
    else:
        advisory = _validated_advisory(
            scenario="aisle_congestion",
            action="CONTINUE",
            hazard="none",
            route="direct",
        )
    return {
        "model": model,
        "video_url": video_url,
        "navigation_intention": intention,
        "closed_choice_answers": {"hazard": answer, "route": None},
        "request_seconds": [round(request_seconds, 6)],
        "total_seconds": round(request_seconds, 6),
        "advisory": advisory,
        "scenario_oracle": {
            "action": "REROUTE",
            "route": "south_bypass",
            "match": (
                advisory["action"] == "REROUTE"
                and advisory["route"] == "south_bypass"
            ),
        },
        "scope": "live single-conflict closed-choice visual advisory",
    }


def request_generic_supervisor_advisory(
    *,
    video_url: str,
    commandable_actors: tuple[str, ...] = ("RobotBlue",),
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Request one generic multi-actor safety command for the allowed actor."""
    prompt = generic_supervisor_prompt(commandable_actors)
    command, request_seconds = _request_constrained_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=prompt,
        grammar_string=GENERIC_COMMAND_GRAMMAR,
        allowed_answers=set(GENERIC_COMMANDS),
        max_completion_tokens=16,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    action, route = GENERIC_COMMANDS[command]
    advisory = _validated_advisory(
        scenario="generic_warehouse_supervision",
        action=action,
        hazard="visual_safety_review",
        route=route,
    )
    return {
        "model": model,
        "video_url": video_url,
        "commandable_actors": list(commandable_actors),
        "raw_command": command,
        "request_seconds": [round(request_seconds, 6)],
        "total_seconds": round(request_seconds, 6),
        "advisory": advisory,
        "scope": "generic all-actor visual supervision with bounded actuation",
    }


def request_vision_only_tutorial_advisory(
    *,
    video_url: str,
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    camera_layout: str = "single",
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Describe the video, then derive a command without simulator facts."""
    perception_prompt = vision_only_perception_prompt(camera_layout)
    visual_report, perception_seconds = _request_free_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=perception_prompt,
        max_completion_tokens=256,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    command_prompt = vision_only_command_prompt(visual_report)
    proposed_command, command_seconds = _request_constrained_text(
        server_url=server_url,
        model=model,
        video_url=None,
        prompt=command_prompt,
        grammar_string=VISION_ONLY_TUTORIAL_GRAMMAR,
        allowed_answers=set(VISION_ONLY_TUTORIAL_COMMANDS),
        max_completion_tokens=16,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    report_supports_reroute = visual_report_supports_reroute(visual_report)
    command = (
        "RobotBlue REROUTE_NORTH_BYPASS"
        if report_supports_reroute
        else "RobotBlue CONTINUE_CURRENT_ROUTE"
    )
    command_source = (
        "gateway_confirmed_model_policy_from_visual_report"
        if command == proposed_command
        else "gateway_corrected_model_policy_from_visual_report"
    )
    action, route = VISION_ONLY_TUTORIAL_COMMANDS[command]
    advisory = _validated_advisory(
        scenario="vision_only_tutorial",
        action=action,
        hazard="visual_safety_review",
        route=route,
    )
    return {
        "model": model,
        "video_url": video_url,
        "prompt": vision_only_tutorial_prompt(camera_layout),
        "prompts": {
            "perception": perception_prompt,
            "command": command_prompt,
        },
        "model_responses": {
            "visual_report": visual_report,
            "policy_command": proposed_command,
        },
        "proposed_command": proposed_command,
        "gateway_command": command,
        "gateway_evidence": {
            "two_actor_conflict_supported": report_supports_reroute,
            "source": "model_visual_report_only",
        },
        "raw_command": command,
        "request_seconds": [
            round(perception_seconds, 6),
            round(command_seconds, 6),
        ],
        "total_seconds": round(perception_seconds + command_seconds, 6),
        "advisory": advisory,
        "command_source": command_source,
        "scope": (
            "two-stage identical-prompt camera-only tutorial decision; "
            "no simulator tracks or coordinates"
        ),
    }


def visual_gap_prompt() -> str:
    """Return the EVK-friendly grammar-constrained visual gap prompt."""
    return """\
The fixed warehouse-camera frame contains EARLIER on the left and NOW on the
right. Compare gaps only within each labeled snapshot. RobotBlue is the compact
white low-profile robot. A forklift is a larger industrial counterbalance
vehicle with long front forks, an upright mast, and an operator cage. Shelves,
cartons, pallets, cones, barriers, floors, and walls are static objects, not
traffic actors.

Reply RobotBlue REROUTE_NORTH_BYPASS only when the visible forklift and
RobotBlue have a clearly smaller nearest-body gap in NOW than in EARLIER and
are converging on the same narrow passage. If no forklift is visible, either
actor is unclear, the gap is not clearly smaller, or the evidence is uncertain,
reply RobotBlue CONTINUE_CURRENT_ROUTE.
Reply with exactly one of those two commands and no explanation.
"""


def rolling_video_gap_prompt() -> str:
    """Return the bounded prompt for a real chronological camera window."""
    return """\
Watch the complete fixed warehouse-camera video in chronological order, from
its first frame through its final frame. RobotBlue is the compact white
low-profile robot. A forklift is a larger industrial counterbalance vehicle
with long front forks, an upright mast, and an operator cage. Shelves, cartons,
pallets, cones, barriers, floors, and walls are static objects, not traffic
actors.

Reply RobotBlue REROUTE_NORTH_BYPASS only when visible motion across the video
shows the forklift and RobotBlue converging on the same narrow passage and
their nearest-body gap clearly decreases from earlier frames to later frames.
If no forklift is visible, either actor is unclear, their gap does not clearly
decrease, or the evidence is uncertain, reply
RobotBlue CONTINUE_CURRENT_ROUTE.
Reply with exactly one of those two commands and no explanation.
"""


def visual_inventory_prompt() -> str:
    """Return the compact actor inventory used before route reasoning."""
    return """\
Inspect every frame of the attached warehouse image or video using only visible
pixels, then report each traffic-actor category once. A compact white
low-profile autonomous unit is category mobile robot. A forklift is a larger industrial
counterbalance vehicle with long front forks, an upright mast, and an
operator cage. Shelving, cartons, pallets, cones, barriers, floors, and walls
are not traffic actors.
Use PRESENT only when category-identifying pixels are directly visible in this
clip. Do not infer an actor behind a shelf, outside the frame, or from where it
appeared in an earlier request. If a candidate could instead be part of a rack
or pallet, use UNCERTAIN. Use ABSENT when no candidate for that category is
directly visible.
Reply with exactly these two lines, replacing STATUS with PRESENT, ABSENT, or
UNCERTAIN:
mobile robot: STATUS
forklift: STATUS
Do not list individual frames or add other categories, descriptions, motion,
or intent.
"""


def passage_condition_prompt(vision_input: str = "rolling_video") -> str:
    """Return the explicit two-passage condition prompt."""
    if vision_input not in VISION_INPUT_CHOICES:
        raise ValueError(
            "vision_input must be temporal_pair or rolling_video"
        )
    media_description = (
        "Watch the complete fixed-camera warehouse video in chronological "
        "order."
        if vision_input == "rolling_video"
        else (
            "Compare the fixed-camera EARLIER and NOW images using only visible "
            "change."
        )
    )
    return f"""\
Act as a warehouse-wide traffic observer, not as a robot controller.
{media_description} Report both configured passage conditions using only
visible pixels.

The compact white low-profile autonomous unit is category mobile_robot. A
larger industrial counterbalance vehicle with long front forks, an upright
mast, and an operator cage is category forklift. Shelves, cartons, pallets,
walls, floors, and barriers are not traffic actors.

The large stocked rack dominating the middle of the camera view separates two
configured traversable floor passages:
- left_passage is the floor opening around the image-left end of that central
  rack, between it and the stocked rack visible at image-left.
- right_passage is the floor opening around the yellow-guarded image-right end
  of that central rack, between it and the stocked rack visible at image-right.

Report each passage independently on every request:
- CLEAR: the passage is visible and has enough empty floor for traffic.
- BLOCKAGE_IMMINENT: chronological visible motion shows a traffic actor
  entering or turning toward the passage but not yet substantially occupying
  the traversable opening.
- BLOCKED: a traffic actor's body, mast, forks, or load visibly occupies the
  traversable opening in the latest frame. Use BLOCKED once it occupies the
  opening even if that actor is still moving.
- NOT_VISIBLE: the passage itself cannot be seen well enough to assess.
- UNCERTAIN: the passage is visible but its condition cannot be distinguished.

Use offender NONE only with CLEAR or NOT_VISIBLE, UNKNOWN with UNCERTAIN, and
FORKLIFT, MOBILE_ROBOT, or UNKNOWN with BLOCKAGE_IMMINENT or BLOCKED. A
forklift merely visible behind a rack is not yet an imminent blockage. Do not
copy the right-passage condition onto the left passage. A passage is not CLEAR
when any part of a traffic actor visibly lies in the traversable opening
between the two rack ends, even if narrow strips of floor remain visible.

Return exactly one JSON object and no Markdown, robot command, or explanation:
{{"left_passage":{{"status":"STATUS","offender":"OFFENDER"}},"right_passage":{{"status":"STATUS","offender":"OFFENDER"}}}}
Replace every STATUS and OFFENDER independently from visible evidence. The
capitalized words above are field placeholders, not suggested answers.
"""


def generic_passage_supervisor_prompt(
    vision_input: str = "rolling_video",
) -> str:
    """Return the two camera-only prompts shown in live status and evidence."""
    return (
        "STAGE 1 — GENERIC TRAFFIC ACTOR INVENTORY\n"
        + visual_inventory_prompt()
        + "\nSTAGE 2 — EXPLICIT TWO-PASSAGE CONDITION\n"
        + passage_condition_prompt(vision_input)
        + "\nLOCAL ADAPTER (not a model prompt): preserve both passage states "
        "and map a visible forklift blockage of right_passage affecting a "
        "visible mobile robot to the configured bypass policy."
    )


def normalize_passage_condition_report(report: str) -> str:
    """Normalize only allow-listed explicit two-passage conditions."""
    candidate = report.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence is not None:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid passage-condition JSON: {report!r}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "left_passage",
        "right_passage",
    }:
        raise ValueError(f"Invalid passage-condition keys: {report!r}")
    normalized: dict[str, dict[str, str]] = {}
    for passage_name in PASSAGE_NAMES:
        key = f"{passage_name}_passage"
        state = payload[key]
        if not isinstance(state, dict) or set(state) != {
            "status",
            "offender",
        }:
            raise ValueError(
                f"Invalid {key} condition: {report!r}"
            )
        status = str(state["status"]).upper()
        offender = str(state["offender"]).upper()
        # Treat the status as authoritative when the model emits an
        # internally inconsistent offender. This is a bounded schema repair,
        # not a hazard inference: CLEAR/NOT_VISIBLE cannot name an offender,
        # while UNCERTAIN cannot claim a specific actor. The untouched raw
        # response remains available to the supervisor UI and evidence log.
        if status in {"CLEAR", "NOT_VISIBLE"}:
            offender = "NONE"
        elif status == "UNCERTAIN":
            offender = "UNKNOWN"
        elif (
            status in {"BLOCKAGE_IMMINENT", "BLOCKED"}
            and offender == "NONE"
        ):
            offender = "UNKNOWN"
        normalized[key] = {
            "status": status,
            "offender": offender,
        }
    canonical = json.dumps(normalized, separators=(",", ":"))
    if canonical not in PASSAGE_CONDITION_ANSWERS:
        raise ValueError(
            f"Passage condition is outside the allow-list: {report!r}"
        )
    return canonical


def normalize_passage_status_report(report: str) -> str:
    """Normalize only allow-listed JSON passage states to canonical JSON."""
    candidate = report.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence is not None:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid passage-status JSON: {report!r}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "traffic_actors",
        "passage_states",
        "passages_blocked",
        "passages_blockage_imminent",
    }:
        raise ValueError(f"Invalid passage-status keys: {report!r}")

    traffic_actors = payload["traffic_actors"]
    passage_states = json.loads(
        normalize_passage_condition_report(
            json.dumps(payload["passage_states"])
        )
    )
    passages_blocked = payload["passages_blocked"]
    imminent = payload["passages_blockage_imminent"]
    if isinstance(imminent, dict):
        imminent = [
            {"name": name, "offender": offender}
            for name, offender in imminent.items()
        ]
    normalized = {
        "traffic_actors": traffic_actors,
        "passage_states": passage_states,
        "passages_blocked": passages_blocked,
        "passages_blockage_imminent": imminent,
    }
    canonical = json.dumps(normalized, separators=(",", ":"))
    if canonical not in PASSAGE_STATUS_ANSWERS:
        raise ValueError(f"Passage status is outside the allow-list: {report!r}")
    return canonical


def parse_passage_status_report(report: str) -> dict[str, Any]:
    """Parse an allow-listed shared warehouse state."""
    return json.loads(normalize_passage_status_report(report))


def request_passage_status_advisory(
    *,
    video_url: str,
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    vision_input: str = "rolling_video",
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Request generic passage state, then map it to RobotBlue deterministically."""
    inventory_prompt = visual_inventory_prompt()
    inventory, inventory_seconds = _request_free_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=inventory_prompt,
        max_completion_tokens=36,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    robot_status = _inventory_status(inventory, "mobile robot")
    forklift_status = _inventory_status(inventory, "forklift")
    if robot_status is None or forklift_status is None:
        raise ValueError(
            f"Invalid generic traffic-actor inventory: {inventory!r}"
        )
    traffic_actors: list[str] = []
    if robot_status == "PRESENT":
        traffic_actors.append("mobile_robot")
    if forklift_status == "PRESENT":
        traffic_actors.append("forklift")

    condition_prompt = passage_condition_prompt(vision_input)
    raw_conditions: list[str] = []

    def normalize_and_remember(answer: str) -> str:
        raw_conditions.append(answer)
        return normalize_passage_condition_report(answer)

    condition_report, condition_seconds = _request_constrained_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=condition_prompt,
        grammar_string=PASSAGE_CONDITION_GRAMMAR,
        allowed_answers=PASSAGE_CONDITION_ANSWERS,
        max_completion_tokens=64,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
        answer_normalizer=normalize_and_remember,
    )
    passage_condition = json.loads(condition_report)
    blocked, imminent_items = _derived_passage_collections(
        passage_condition
    )
    scene_state = {
        "traffic_actors": traffic_actors,
        "passage_states": passage_condition,
        "passages_blocked": blocked,
        "passages_blockage_imminent": imminent_items,
    }
    report = json.dumps(scene_state, separators=(",", ":"))
    if report not in PASSAGE_STATUS_ANSWERS:
        raise ValueError(f"Fused passage status is outside the allow-list: {report}")
    mobile_robot_visible = "mobile_robot" in scene_state["traffic_actors"]
    forklift_visible = "forklift" in scene_state["traffic_actors"]
    forklift_inventory_status = _inventory_status(inventory, "forklift")
    right_state = passage_condition["right_passage"]
    right_blocked = (
        right_state["status"] == "BLOCKED"
        and right_state["offender"] == "FORKLIFT"
    )
    right_imminent = (
        right_state["status"] == "BLOCKAGE_IMMINENT"
        and right_state["offender"] == "FORKLIFT"
    )
    route_affected = (
        mobile_robot_visible
        # The passage stage is itself a constrained visual classification and
        # names the offender. Accept an UNCERTAIN coarse actor-inventory label
        # when the spatial passage report identifies a forklift, but never
        # override a contradictory ABSENT inventory result.
        and forklift_inventory_status != "ABSENT"
        and (right_blocked or right_imminent)
    )
    command = (
        "RobotBlue REROUTE_NORTH_BYPASS"
        if route_affected
        else "RobotBlue CONTINUE_CURRENT_ROUTE"
    )
    action, route = VISION_ONLY_TUTORIAL_COMMANDS[command]
    advisory = _validated_advisory(
        scenario="generic_passage_status",
        action=action,
        hazard=(
            "north_east_passage_blockage"
            if route_affected
            else "none"
        ),
        route=route,
    )
    return {
        "model": model,
        "video_url": video_url,
        "vision_input": vision_input,
        "prompt": generic_passage_supervisor_prompt(vision_input),
        "prompts": {
            "traffic_actor_inventory": inventory_prompt,
            "passage_condition": condition_prompt,
        },
        "model_output": report,
        "raw_model_outputs": {
            "traffic_actor_inventory": inventory,
            "passage_condition": (
                raw_conditions[0] if raw_conditions else condition_report
            ),
        },
        "normalized_model_output": report,
        "model_responses": {
            "scene_inventory": inventory,
            "passage_condition": condition_report,
            "passage_status": report,
            **scene_state,
        },
        "proposed_command": None,
        "gateway_command": command,
        "gateway_evidence": {
            "source": "deterministic_passage_status_adapter",
            "mobile_robot_visible": mobile_robot_visible,
            "forklift_visible": forklift_visible,
            "forklift_inventory_status": forklift_inventory_status,
            "left_passage": passage_condition["left_passage"],
            "right_passage": right_state,
            "right_passage_blocked_by_forklift": right_blocked,
            "right_passage_blockage_imminent_by_forklift": (
                right_imminent
            ),
        },
        "raw_command": command,
        "request_seconds": [
            round(inventory_seconds, 6),
            round(condition_seconds, 6),
        ],
        "total_seconds": round(inventory_seconds + condition_seconds, 6),
        "advisory": advisory,
        "command_source": "deterministic_passage_status_adapter",
        "scope": (
            f"two-stage camera-only {vision_input} generic passage-state "
            "report; model emits no robot command and receives no simulator "
            "tracks"
        ),
    }


def _inventory_status(inventory: str, category: str) -> str | None:
    segment = re.search(
        rf"\b{re.escape(category)}\s*:\s*([^\n;]{{0,40}})",
        inventory,
        flags=re.IGNORECASE,
    )
    if not segment:
        return None
    status = re.search(
        r"\b(PRESENT|ABSENT|UNCERTAIN)\b",
        segment.group(1),
        flags=re.IGNORECASE,
    )
    return status.group(1).upper() if status else None


def request_visual_gap_advisory(
    *,
    video_url: str,
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    vision_input: str = "temporal_pair",
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Return an EVK-friendly grammar-constrained camera-only gap decision."""
    if vision_input not in VISION_INPUT_CHOICES:
        raise ValueError(
            "vision_input must be temporal_pair or rolling_video"
        )
    inventory_prompt = visual_inventory_prompt()
    inventory, inventory_seconds = _request_free_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=inventory_prompt,
        max_completion_tokens=40,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    mobile_robot_visible = (
        _inventory_status(inventory, "mobile robot") == "PRESENT"
    )
    forklift_visible = _inventory_status(inventory, "forklift") == "PRESENT"
    both_actors_visible = mobile_robot_visible and forklift_visible
    actor_presence = (
        "BOTH_ACTORS_VISIBLE"
        if both_actors_visible
        else "ACTORS_NOT_BOTH_VISIBLE"
    )
    prompt = (
        rolling_video_gap_prompt()
        if vision_input == "rolling_video"
        else visual_gap_prompt()
    )
    request_seconds = [inventory_seconds]
    if both_actors_visible:
        command, gap_seconds = _request_constrained_text(
            server_url=server_url,
            model=model,
            video_url=video_url,
            prompt=prompt,
            grammar_string=VISUAL_GAP_GRAMMAR,
            allowed_answers=VISUAL_GAP_ANSWERS,
            max_completion_tokens=24,
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
        request_seconds.append(gap_seconds)
    else:
        command = "RobotBlue CONTINUE_CURRENT_ROUTE"
    action, route = VISION_ONLY_TUTORIAL_COMMANDS[command]
    advisory = _validated_advisory(
        scenario="vision_gap_tutorial",
        action=action,
        hazard="visual_gap_closing" if action == "REROUTE" else "none",
        route=route,
    )
    return {
        "model": model,
        "video_url": video_url,
        "vision_input": vision_input,
        "prompt": inventory_prompt + "\n\n" + prompt,
        "prompts": {
            "scene_inventory": inventory_prompt,
            "visual_gap": prompt,
        },
        "model_responses": {
            "actor_presence": actor_presence,
            "scene_inventory": inventory,
            "visual_gap_classification": (
                command
                if both_actors_visible
                else "not_requested"
            ),
        },
        "proposed_command": command,
        "gateway_command": command,
        "gateway_evidence": {
            "source": "model_visual_gap_classification_only",
            "actor_presence": actor_presence,
            "mobile_robot_visible": mobile_robot_visible,
            "forklift_visible": forklift_visible,
            "classification": command,
        },
        "raw_command": command,
        "request_seconds": [
            round(item, 6) for item in request_seconds
        ],
        "total_seconds": round(sum(request_seconds), 6),
        "advisory": advisory,
        "command_source": "inventory_gated_model_visual_gap",
        "scope": (
            f"camera-only {vision_input} temporal gap classification; "
            "no simulator tracks or coordinates"
        ),
    }


def request_staged_generic_supervisor_advisory(
    *,
    video_url: str,
    commandable_actors: tuple[str, ...] = ("RobotBlue",),
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Review motion first, then ask for an intervention only when warranted."""
    prompts = generic_staged_supervisor_prompts(commandable_actors)
    conflict, conflict_seconds = _request_constrained_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=prompts["conflict"],
        grammar_string=GENERIC_CONFLICT_GRAMMAR,
        allowed_answers={"NO_CONFLICT", "CONFLICT"},
        max_completion_tokens=8,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    model_responses: dict[str, str | None] = {
        "conflict": conflict,
        "intervention": None,
    }
    timings = [round(conflict_seconds, 6)]
    if conflict == "NO_CONFLICT":
        command = "RobotBlue CONTINUE CURRENT_ROUTE"
        command_source = "gateway_from_model_clear_classification"
    else:
        command, intervention_seconds = _request_constrained_text(
            server_url=server_url,
            model=model,
            video_url=video_url,
            prompt=prompts["intervention"],
            grammar_string=GENERIC_INTERVENTION_GRAMMAR,
            allowed_answers=set(GENERIC_INTERVENTION_COMMANDS),
            max_completion_tokens=16,
            timeout_seconds=timeout_seconds,
            urlopen=urlopen,
        )
        model_responses["intervention"] = command
        timings.append(round(intervention_seconds, 6))
        command_source = "model_intervention_command"
    action, route = GENERIC_COMMANDS[command]
    advisory = _validated_advisory(
        scenario="generic_warehouse_supervision",
        action=action,
        hazard=(
            "none"
            if conflict == "NO_CONFLICT"
            else "visual_safety_review"
        ),
        route=route,
    )
    return {
        "model": model,
        "video_url": video_url,
        "commandable_actors": list(commandable_actors),
        "model_responses": model_responses,
        "raw_command": command,
        "command_source": command_source,
        "request_seconds": timings,
        "total_seconds": round(sum(timings), 6),
        "advisory": advisory,
        "scope": (
            "staged generic all-actor visual supervision with bounded actuation"
        ),
    }


def _tracked_allowed_commands(
    tracked_scene_facts: dict[str, Any],
) -> tuple[str, ...]:
    navigation = tracked_scene_facts.get("navigation", {})
    current_status = str(
        navigation.get("evaluated_current_corridor_status", "UNKNOWN")
    )
    route_statuses = navigation.get("route_statuses", {})
    if current_status in {"CLEAR", "OPEN"}:
        return (
            "RobotBlue CONTINUE CURRENT_ROUTE",
            "RobotBlue YIELD HOLD",
            "RobotBlue STOP HOLD",
        )
    commands: list[str] = []
    if route_statuses.get("north_bypass") == "OPEN":
        commands.append("RobotBlue REROUTE NORTH_BYPASS")
    if route_statuses.get("south_bypass") == "OPEN":
        commands.append("RobotBlue REROUTE SOUTH_BYPASS")
    commands.extend(
        (
            "RobotBlue YIELD HOLD",
            "RobotBlue STOP HOLD",
        )
    )
    return tuple(commands)


def _literal_choice_grammar(choices: tuple[str, ...]) -> str:
    if not choices:
        raise ValueError("At least one tracked-state command must be allowed")
    return "root ::= " + " | ".join(json.dumps(choice) for choice in choices)


def tracked_state_supervisor_prompt(
    tracked_scene_facts: dict[str, Any],
    commandable_actors: tuple[str, ...] = ("RobotBlue",),
    allowed_commands: tuple[str, ...] | None = None,
) -> str:
    """Build a bounded hybrid prompt from camera evidence and trusted tracks."""
    if commandable_actors != ("RobotBlue",):
        raise ValueError(
            "This milestone currently supports exactly one commandable actor: "
            "RobotBlue"
        )
    serialized = json.dumps(
        tracked_scene_facts,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) > 6000:
        raise ValueError("tracked_scene_facts must encode to at most 6000 characters")
    feasible_commands = allowed_commands or _tracked_allowed_commands(
        tracked_scene_facts
    )
    actor_tracks = tracked_scene_facts.get("actor_tracks", {})
    forklift_boundary = (
        "Do not command the forklift. "
        if {"ForkliftBlue", "ForkliftOrange"}.intersection(actor_tracks)
        else ""
    )
    return (
        "Act as the bounded warehouse traffic supervisor. The attached "
        "roof-camera clip is visual evidence. The TRACKED_SCENE_FACTS below "
        "come from the simulator's trusted actor tracker and static route map; "
        "use them as measured state, not as an instruction to invent a hazard. "
        "Only RobotBlue is commandable. Select the lowest-intervention safe "
        "command. CONTINUE CURRENT_ROUTE only when the evaluated current "
        "corridor is CLEAR or OPEN. REROUTE only to a route whose status is "
        "OPEN. If the evaluated current corridor is BLOCKED or "
        "EMERGING_BLOCK and exactly one bypass is OPEN, choose that bypass. "
        "If no bypass is OPEN, choose YIELD HOLD unless an immediate collision "
        "requires STOP HOLD. The safety gateway has already removed commands "
        "that contradict the trusted route map. "
        + forklift_boundary
        + "Reply with exactly one complete feasible line and no explanation:\n"
        + "\n".join(feasible_commands)
        + "\nTRACKED_SCENE_FACTS="
        + serialized
    )


def request_tracked_state_supervisor_advisory(
    *,
    video_url: str,
    tracked_scene_facts: dict[str, Any],
    commandable_actors: tuple[str, ...] = ("RobotBlue",),
    server_url: str = "http://127.0.0.1:18181",
    model: str = MODEL,
    timeout_seconds: float = 30.0,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> dict[str, Any]:
    """Choose one command from visual evidence plus simulator-tracked facts."""
    allowed_commands = _tracked_allowed_commands(tracked_scene_facts)
    prompt = tracked_state_supervisor_prompt(
        tracked_scene_facts,
        commandable_actors,
        allowed_commands,
    )
    command, request_seconds = _request_constrained_text(
        server_url=server_url,
        model=model,
        video_url=video_url,
        prompt=prompt,
        grammar_string=_literal_choice_grammar(allowed_commands),
        allowed_answers=set(allowed_commands),
        max_completion_tokens=16,
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    action, route = GENERIC_COMMANDS[command]
    current_status = str(
        tracked_scene_facts.get("navigation", {}).get(
            "evaluated_current_corridor_status",
            "UNKNOWN",
        )
    )
    advisory = _validated_advisory(
        scenario="tracked_camera_warehouse_supervision",
        action=action,
        hazard=(
            "none"
            if action == "CONTINUE"
            else "tracked_corridor_conflict"
        ),
        route=route,
    )
    return {
        "model": model,
        "video_url": video_url,
        "commandable_actors": list(commandable_actors),
        "tracked_scene_facts": tracked_scene_facts,
        "model_responses": {
            "tracked_policy_command": command,
            "evaluated_current_corridor_status": current_status,
        },
        "raw_command": command,
        "command_source": "model_from_camera_and_tracked_scene_facts",
        "request_seconds": [round(request_seconds, 6)],
        "total_seconds": round(request_seconds, 6),
        "advisory": advisory,
        "scope": (
            "hybrid camera plus simulator-tracked-state supervision with "
            "bounded actuation"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video",
        type=Path,
        help="Video file visible to GenieX; intended for same-EVK use.",
    )
    source.add_argument(
        "--video-url",
        help="Video URL passed verbatim to GenieX, such as file:///path/clip.mp4.",
    )
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--server-url", default="http://127.0.0.1:18181")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--live-single-route",
        action="store_true",
        help=(
            "Use one live aisle-conflict request; the policy gateway maps a "
            "detected conflict to the modeled south bypass."
        ),
    )
    parser.add_argument(
        "--navigation-intention",
        default=DEFAULT_NAVIGATION_INTENTION,
        help="Current RobotBlue navigation intention included in the live prompt.",
    )
    parser.add_argument(
        "--generic-supervisor",
        action="store_true",
        help=(
            "Use the generic all-actor supervisor prompt and return one complete "
            "allow-listed actor/action/route command."
        ),
    )
    parser.add_argument(
        "--vision-only-tutorial",
        action="store_true",
        help=(
            "Use the identical camera-only tutorial prompt and four-command "
            "grammar without simulator positions, velocities, route occupancy, "
            "a hazard label, or an expected answer."
        ),
    )
    parser.add_argument(
        "--visual-gap-only",
        action="store_true",
        help=(
            "Use the EVK-friendly grammar-constrained camera gap "
            "classification without simulator state."
        ),
    )
    parser.add_argument(
        "--passage-status-only",
        action="store_true",
        help=(
            "Report generic visible traffic actors and shared passage state; "
            "a deterministic local adapter maps that report to RobotBlue."
        ),
    )
    parser.add_argument(
        "--vision-input",
        choices=sorted(VISION_INPUT_CHOICES),
        default="temporal_pair",
        help=(
            "Interpret the attached media as an EARLIER/NOW composite or a "
            "real chronological rolling-window video."
        ),
    )
    parser.add_argument(
        "--camera-layout",
        choices=("single", "grid"),
        default="single",
        help="Describe whether each model frame is one camera or a 2-by-2 grid.",
    )
    parser.add_argument(
        "--tracked-state-json",
        help=(
            "Use the hybrid camera-plus-tracker supervisor with this JSON "
            "object of trusted actor and route facts."
        ),
    )
    parser.add_argument(
        "--commandable-actors",
        default="RobotBlue",
        help="Comma-separated actors the generic supervisor may command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    if args.video is not None:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise SystemExit(f"Video does not exist: {video_path}")
        video_url = video_path.as_uri()
    else:
        video_url = args.video_url

    try:
        if args.tracked_state_json:
            tracked_scene_facts = json.loads(args.tracked_state_json)
            if not isinstance(tracked_scene_facts, dict):
                raise ValueError("--tracked-state-json must decode to an object")
            commandable_actors = tuple(
                item.strip()
                for item in args.commandable_actors.split(",")
                if item.strip()
            )
            result = request_tracked_state_supervisor_advisory(
                video_url=video_url,
                tracked_scene_facts=tracked_scene_facts,
                commandable_actors=commandable_actors,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.passage_status_only:
            result = request_passage_status_advisory(
                video_url=video_url,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                vision_input=args.vision_input,
            )
        elif args.visual_gap_only:
            result = request_visual_gap_advisory(
                video_url=video_url,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                vision_input=args.vision_input,
            )
        elif args.vision_only_tutorial:
            result = request_vision_only_tutorial_advisory(
                video_url=video_url,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                camera_layout=args.camera_layout,
            )
        elif args.generic_supervisor:
            commandable_actors = tuple(
                item.strip()
                for item in args.commandable_actors.split(",")
                if item.strip()
            )
            result = request_staged_generic_supervisor_advisory(
                video_url=video_url,
                commandable_actors=commandable_actors,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.live_single_route:
            result = request_live_aisle_advisory(
                video_url=video_url,
                navigation_intention=args.navigation_intention,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            result = request_advisory(
                scenario=args.scenario,
                video_url=video_url,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Advisory request failed: {exc}") from exc

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
