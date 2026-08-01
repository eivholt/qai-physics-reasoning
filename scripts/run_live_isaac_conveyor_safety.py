#!/usr/bin/env python3
"""Run the image-space Cosmos Reason2 conveyor stack-light demonstration."""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.isaac_sim_mcp.conveyor_safety import (
    CAMERA_PATHS,
    CAPTURE_ROOT,
    capture_conveyor_safety_camera_source,
    create_conveyor_safety_scene_source,
    enable_conveyor_safety_extension_source,
    get_conveyor_safety_state_source,
    set_conveyor_safety_light_and_get_state_source,
)
from integrations.isaac_sim_mcp.server import IsaacTcpClient

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "isaac_sim_conveyor_safety" / "runs"
DEFAULT_STATUS_PATH = (
    REPO_ROOT / "artifacts" / "isaac_sim_conveyor_safety" / "live_status.json"
)
DEFAULT_SERVER_URL = "http://127.0.0.1:18080"
DEFAULT_MODEL = "Cosmos-Reason2-2B-BF16.gguf"
DEFAULT_PREDICTION_HORIZON_SECONDS = 3.0
SIGNALS = ("GREEN", "AMBER", "RED")
SIGNAL_ANSWERS = tuple(
    json.dumps({"signal": signal}, separators=(",", ":")) for signal in SIGNALS
)
SIGNAL_LETTERS = {"G": "GREEN", "A": "AMBER", "R": "RED"}
SIGNAL_GRAMMAR = "root ::= [GAR]"
_IMAGE_DATA_URL_CACHE: dict[tuple[str, int, int], str] = {}
_IMAGE_DATA_URL_CACHE_LIMIT = 4

HAZARD_PROMPT = """\
Review the images in chronological order, with the newest image last.
Return R if any part of any forklift is inside the marked red zone.
Otherwise, return A if any forklift is moving at all.
Otherwise, return G.
Ignore human workers, parcels, and the stack light.
Return one letter only: R, A, or G.
"""

RED_CONFIRMATION_PROMPT = """\
Return R if any tire, body, mast, or fork of the forklift touches or covers the
red-painted floor beside the conveyor. Otherwise return G.
Ignore people, parcels, and the stack light.
Return one letter only: R or G.
"""

AMBER_CONFIRMATION_PROMPT = """\
The images are chronological. Decide whether any forklift moved between them.
Return A if any forklift moved at all, in any direction. Otherwise return G.
Ignore people, parcels, and the stack light.
Return one letter only: A or G.
"""

# Backward-compatible import used by the camera evaluator.
DIRECT_HAZARD_PROMPT = HAZARD_PROMPT


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # On Windows the debug widget can briefly hold the destination open
        # while reading it. Retry that bounded sharing violation instead of
        # killing and relaunching the entire rolling inference runner.
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _parse_isaac_response(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("status") != "ok":
        raise RuntimeError(f"Isaac Sim command failed: {response!r}")
    output = str(response.get("output") or "").strip()
    if not output:
        return {}
    try:
        payload = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Isaac Sim returned invalid output: {output[-2000:]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Isaac Sim result must be a JSON object")
    return payload


def execute_isaac_source(
    client: IsaacTcpClient,
    source: str,
) -> dict[str, Any]:
    return _parse_isaac_response(client.execute(source))


def capture_conveyor_frame(
    client: IsaacTcpClient,
    *,
    camera: str,
    filename: str,
    attempts: int = 3,
    render_steps: int = 1,
    rt_subframes: int = 2,
    width: int = 1024,
    height: int = 576,
) -> dict[str, Any]:
    """Retry transient Replicator annotator detachments serially."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            capture = execute_isaac_source(
                client,
                capture_conveyor_safety_camera_source(
                    {
                        "camera": camera,
                        "filename": filename,
                        "render_steps": render_steps,
                        "rt_subframes": rt_subframes,
                        "width": width,
                        "height": height,
                    }
                ),
            )
            if capture.get("output_path"):
                return capture
            last_error = RuntimeError(
                "Detector capture returned no image path"
            )
        except Exception as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(0.35)
    raise RuntimeError(
        f"Detector capture failed after {attempts} attempts: {last_error}"
    ) from last_error


def image_data_url(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _IMAGE_DATA_URL_CACHE.get(key)
    if cached is not None:
        return cached
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    _IMAGE_DATA_URL_CACHE[key] = data_url
    while len(_IMAGE_DATA_URL_CACHE) > _IMAGE_DATA_URL_CACHE_LIMIT:
        _IMAGE_DATA_URL_CACHE.pop(next(iter(_IMAGE_DATA_URL_CACHE)))
    return data_url


def _read_png_rgb(path: Path) -> tuple[int, int, list[bytearray], int]:
    """Decode the bounded Isaac RGB/RGBA PNG without an external dependency."""

    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG file: {path}")
    position = 8
    idat_chunks: list[bytes] = []
    width = height = bit_depth = color_type = interlace = None
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError(f"Truncated PNG chunk in {path}")
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        chunk_data = data[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            (
                width,
                height,
                bit_depth,
                color_type,
                _compression,
                _filter,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if (
        width is None
        or height is None
        or bit_depth != 8
        or color_type not in (2, 6)
        or interlace != 0
    ):
        raise ValueError(
            "Detector PNG must be non-interlaced 8-bit RGB or RGBA"
        )
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(b"".join(idat_chunks))
    rows: list[bytearray] = []
    offset = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        scanline = bytearray(raw[offset : offset + stride])
        offset += stride
        for index, value in enumerate(scanline):
            left = scanline[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                scanline[index] = (value + left) & 0xFF
            elif filter_type == 2:
                scanline[index] = (value + above) & 0xFF
            elif filter_type == 3:
                scanline[index] = (value + (left + above) // 2) & 0xFF
            elif filter_type == 4:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                predictor = (left, above, upper_left)[distances.index(min(distances))]
                scanline[index] = (value + predictor) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter {filter_type}")
        rows.append(scanline)
        previous = scanline
    return width, height, rows, channels


def resize_png_nearest(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
    decoded_image: tuple[int, int, list[bytearray], int] | None = None,
) -> Path:
    """Write a bounded RGB PNG for the EVK vision transport."""

    if width < 32 or height < 32:
        raise ValueError("model image dimensions must be at least 32 pixels")
    source_width, source_height, rows, channels = (
        decoded_image if decoded_image is not None else _read_png_rgb(source)
    )
    if source_width == width and source_height == height and channels == 3:
        return source
    raw_rows = bytearray()
    for output_y in range(height):
        source_y = min(source_height - 1, output_y * source_height // height)
        source_row = rows[source_y]
        raw_rows.append(0)
        for output_x in range(width):
            source_x = min(source_width - 1, output_x * source_width // width)
            offset = source_x * channels
            raw_rows.extend(source_row[offset : offset + 3])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=6))
        + chunk(b"IEND", b"")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(png)
    temporary.replace(destination)
    return destination


def crop_png(
    source: Path,
    destination: Path,
    *,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    decoded_image: tuple[int, int, list[bytearray], int] | None = None,
) -> Path:
    """Crop an Isaac RGB frame without adding a heavyweight image dependency."""

    width, height, rows, channels = (
        decoded_image if decoded_image is not None else _read_png_rgb(source)
    )
    x_min = max(0, min(width - 1, int(x_min)))
    y_min = max(0, min(height - 1, int(y_min)))
    x_max = max(x_min, min(width - 1, int(x_max)))
    y_max = max(y_min, min(height - 1, int(y_max)))
    output_width = x_max - x_min + 1
    output_height = y_max - y_min + 1
    raw_rows = bytearray()
    for y in range(y_min, y_max + 1):
        raw_rows.append(0)
        row = rows[y]
        for x in range(x_min, x_max + 1):
            offset = x * channels
            raw_rows.extend(row[offset : offset + 3])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                output_width,
                output_height,
                8,
                2,
                0,
                0,
                0,
            ),
        )
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=6))
        + chunk(b"IEND", b"")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(png)
    temporary.replace(destination)
    return destination


def recover_forklift_boxes_from_rgb(
    image_path: Path,
    semantic_boxes: list[dict[str, Any]] | None,
    *,
    previous_observation: dict[str, Any] | None = None,
    decoded_image: tuple[int, int, list[bytearray], int] | None = None,
    expected_forklift_indices: tuple[int, ...] = (1, 2, 3),
) -> list[dict[str, Any]]:
    """Fill transient semantic misses from large yellow vehicle regions."""

    if (
        not expected_forklift_indices
        or len(set(expected_forklift_indices)) != len(expected_forklift_indices)
        or any(index not in (1, 2, 3) for index in expected_forklift_indices)
    ):
        raise ValueError(
            "expected_forklift_indices must contain unique indices 1 through 3"
        )
    expected_indices = tuple(expected_forklift_indices)
    boxes = [
        dict(item)
        for item in (semantic_boxes or [])
        if int(item.get("forklift_index", -1)) in expected_indices
    ]
    present_indices = {
        int(item["forklift_index"])
        for item in boxes
        if "forklift_index" in item
    }
    if present_indices == set(expected_indices):
        return boxes
    missing_indices = [
        index for index in expected_indices if index not in present_indices
    ]

    width, height, rows, channels = (
        decoded_image if decoded_image is not None else _read_png_rgb(image_path)
    )
    yellow_pixels: set[tuple[int, int]] = set()
    for y, row in enumerate(rows):
        for x in range(width):
            offset = x * channels
            red, green, blue = row[offset : offset + 3]
            if (
                red > 150
                and green > 105
                and blue < 105
                and red - green < 150
                and green - blue > 45
            ):
                yellow_pixels.add((x, y))

    components: list[dict[str, int]] = []
    while yellow_pixels:
        seed = yellow_pixels.pop()
        stack = [seed]
        x_values: list[int] = []
        y_values: list[int] = []
        while stack:
            x, y = stack.pop()
            x_values.append(x)
            y_values.append(y)
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor = (x + delta_x, y + delta_y)
                    if neighbor in yellow_pixels:
                        yellow_pixels.remove(neighbor)
                        stack.append(neighbor)
        pixel_count = len(x_values)
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        component_width = x_max - x_min + 1
        component_height = y_max - y_min + 1
        fill_ratio = pixel_count / float(component_width * component_height)
        if (
            pixel_count >= max(180, int(width * height * 0.00085))
            and component_width >= max(24, int(width * 0.045))
            and component_height >= max(14, int(height * 0.030))
            # A vehicle touching the yellow boundary can merge with a short
            # boundary segment. Keep that recoverable while still rejecting
            # the full-height rack rails.
            and component_height <= int(height * 0.34)
            and x_min > int(width * 0.10)
            and x_max < int(width * 0.90)
            # Forklift body panels form dense yellow regions. Conveyor safety
            # borders and lane dashes are thin, sparse components and must not
            # be expanded into vehicle-sized fallback boxes.
            and fill_ratio >= 0.12
        ):
            components.append(
                {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "pixel_count": pixel_count,
                }
            )

    def component_matches_existing(component: dict[str, int]) -> bool:
        center_x = (component["x_min"] + component["x_max"]) * 0.5
        center_y = (component["y_min"] + component["y_max"]) * 0.5
        return any(
            int(box["x_min"]) <= center_x <= int(box["x_max"])
            and int(box["y_min"]) <= center_y <= int(box["y_max"])
            for box in boxes
        )

    unmatched_components = [
        component
        for component in components
        if not component_matches_existing(component)
    ]
    if len(unmatched_components) < len(missing_indices):
        return boxes

    candidates: list[dict[str, Any]] = []
    for component in unmatched_components:
        center_x = (component["x_min"] + component["x_max"]) * 0.5
        center_y = (component["y_min"] + component["y_max"]) * 0.5
        if center_x < width * 0.5:
            x_min = component["x_min"] - int(width * 0.015)
            x_max = component["x_max"] + int(width * 0.075)
        else:
            x_min = component["x_min"] - int(width * 0.075)
            x_max = component["x_max"] + int(width * 0.015)
        candidates.append(
            {
                "center": [center_x, center_y],
                "x_min": max(0, x_min),
                "y_min": max(0, component["y_min"] - int(height * 0.085)),
                "x_max": min(width - 1, x_max),
                "y_max": min(
                    height - 1,
                    component["y_max"] + int(height * 0.025),
                ),
            }
        )

    previous_by_index = {
        int(item["forklift_index"]): item.get("center")
        for item in (
            previous_observation.get("forklifts", [])
            if previous_observation is not None
            else []
        )
        if "forklift_index" in item
    }
    default_normalized_centers = {
        1: (0.31, 0.76),
        2: (0.69, 0.43),
        3: (0.34, 0.30),
    }

    def assignment_cost(
        forklift_index: int,
        candidate: dict[str, Any],
    ) -> float:
        center = candidate["center"]
        previous_center = previous_by_index.get(forklift_index)
        if (
            isinstance(previous_center, list)
            and len(previous_center) >= 2
        ):
            target_x = float(previous_center[0])
            target_y = float(previous_center[1])
        else:
            normalized = default_normalized_centers[forklift_index]
            target_x = normalized[0] * width
            target_y = normalized[1] * height
        delta_x = (float(center[0]) - target_x) / width
        delta_y = (float(center[1]) - target_y) / height
        return delta_x * delta_x + delta_y * delta_y

    best_assignment: tuple[dict[str, Any], ...] | None = None
    best_cost = float("inf")
    for assignment in itertools.permutations(
        candidates,
        len(missing_indices),
    ):
        cost = sum(
            assignment_cost(forklift_index, candidate)
            for forklift_index, candidate in zip(
                missing_indices,
                assignment,
            )
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    if best_assignment is None:
        return boxes
    for missing_index, candidate in zip(
        missing_indices,
        best_assignment,
    ):
        boxes.append(
            {
                key: candidate[key]
                for key in ("x_min", "y_min", "x_max", "y_max")
            }
            | {
                "semantic_label": f"safety_forklift_{missing_index}",
                "forklift_index": missing_index,
                "box_source": "camera_rgb_yellow_vehicle_fallback",
            }
        )
    return sorted(boxes, key=lambda item: int(item["forklift_index"]))


def extract_visual_overlap_observation(
    image_path: Path,
    fork_carton_pixel_boxes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Intersect the fork carton's rendered 2D box with colored floor bands."""

    width, height, rows, channels = _read_png_rgb(image_path)
    red_counts = [0] * height
    amber_counts = [0] * height
    for y, row in enumerate(rows):
        for x in range(width):
            offset = x * channels
            red, green, blue = row[offset : offset + 3]
            if red > 190 and green < 100 and blue < 100:
                red_counts[y] += 1
            if red > 190 and 110 < green < 240 and blue < 100:
                amber_counts[y] += 1
    minimum_band_pixels = max(32, int(width * 0.20))
    red_rows = [
        row for row, count in enumerate(red_counts) if count >= minimum_band_pixels
    ]
    amber_rows = [
        row
        for row, count in enumerate(amber_counts)
        if count >= minimum_band_pixels
    ]
    if not red_rows or not amber_rows:
        raise RuntimeError(
            "PTZ color segmentation could not resolve both safety floor bands"
        )
    boxes = fork_carton_pixel_boxes or []
    source = "camera_rgb_plus_semantic_2d_bbox"
    if not boxes:
        # The tracked forklift is normalized above the conveyor and centered
        # horizontally. Find connected tan regions in that narrow corridor.
        # This operates on the captured RGB frame and survives Replicator
        # occasionally retaining an asset's old semantic cache.
        x_start = int(width * 0.42)
        x_end = int(width * 0.58)
        unvisited: set[tuple[int, int]] = set()
        for y, row in enumerate(rows):
            for x in range(x_start, x_end):
                offset = x * channels
                red, green, blue = row[offset : offset + 3]
                if (
                    155 <= red <= 240
                    and 125 <= green <= 225
                    and 90 <= blue <= 205
                    and red - green >= 3
                    and green - blue >= 12
                    and red - blue >= 25
                ):
                    unvisited.add((x, y))
        components: list[tuple[int, dict[str, int | str]]] = []
        while unvisited:
            seed = unvisited.pop()
            stack = [seed]
            points = [seed]
            while stack:
                x, y = stack.pop()
                for neighbor in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
                        points.append(neighbor)
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)
            component_width = x_max - x_min + 1
            component_height = y_max - y_min + 1
            if (
                len(points) >= 64
                and component_width >= 12
                and component_height >= 8
                and component_width <= component_height * 4
                and y_min >= int(height * 0.18)
            ):
                components.append(
                    (
                        y_min,
                        {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "semantic_label": "rgb_fork_carton_component",
                        },
                    )
                )
        if components:
            # The dynamic PTZ places the tracked fork-tip carton on the image
            # centerline. Forklift trim and workers can contain smaller tan
            # regions higher in the image, so first reject off-centre
            # distractors and then choose the first centred carton.
            image_center_x = (width - 1) * 0.5
            center_tolerance = width * 0.055
            centered_components = [
                item
                for item in components
                if abs(
                    (
                        int(item[1]["x_min"])
                        + int(item[1]["x_max"])
                    )
                    * 0.5
                    - image_center_x
                )
                <= center_tolerance
            ]
            candidate_components = centered_components or components
            boxes = [min(candidate_components, key=lambda item: item[0])[1]]
            source = "camera_rgb_carton_component_plus_floor_masks"
    carton_rows: list[int] = []
    normalized_boxes: list[dict[str, int | str]] = []
    for box in boxes:
        try:
            x_min = max(0, min(width - 1, int(box["x_min"])))
            x_max = max(0, min(width - 1, int(box["x_max"])))
            y_min = max(0, min(height - 1, int(box["y_min"])))
            y_max = max(0, min(height - 1, int(box["y_max"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fork-carton pixel box: {box!r}") from exc
        if x_max < x_min or y_max < y_min:
            raise ValueError(f"Inverted fork-carton pixel box: {box!r}")
        carton_rows.extend(range(y_min, y_max + 1))
        normalized_boxes.append(
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "semantic_label": str(box.get("semantic_label", "")),
            }
        )
    if not carton_rows:
        raise RuntimeError(
            "No semantic 2D box was detected for the fork-mounted carton"
        )
    carton_range = [min(carton_rows), max(carton_rows)]
    red_range = [min(red_rows), max(red_rows)]
    amber_range = [min(amber_rows), max(amber_rows)]

    def pixels_intersect_range(rows_to_check: list[int], band: list[int]) -> bool:
        return any(band[0] <= row <= band[1] for row in rows_to_check)

    return {
        "source": source,
        "resolution": [width, height],
        "fork_carton_pixel_boxes": normalized_boxes,
        "fork_carton_row_range": carton_range,
        "red_row_range": red_range,
        "amber_row_range": amber_range,
        "red_overlap": pixels_intersect_range(carton_rows, red_range),
        "amber_overlap": pixels_intersect_range(carton_rows, amber_range),
        "simulator_coordinates_used": False,
        "metric_clearance_used": False,
    }


def extract_forklift_motion_observation(
    image_path: Path,
    forklift_pixel_boxes: list[dict[str, Any]] | None,
    *,
    previous_observation: dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
    prediction_horizon_seconds: float = DEFAULT_PREDICTION_HORIZON_SECONDS,
    minimum_motion_pixels_per_second: float = 5.0,
    decoded_image: tuple[int, int, list[bytearray], int] | None = None,
    red_zone_mask: dict[str, Any] | None = None,
    expected_forklift_count: int = 3,
) -> dict[str, Any]:
    """Detect current overlap and forklift motion using camera pixels only."""

    if prediction_horizon_seconds <= 0:
        raise ValueError("prediction_horizon_seconds must be positive")
    if not 1 <= expected_forklift_count <= 3:
        raise ValueError("expected_forklift_count must be from 1 through 3")
    if red_zone_mask is None:
        width, height, rows, channels = (
            decoded_image
            if decoded_image is not None
            else _read_png_rgb(image_path)
        )
        red_pixels: set[tuple[int, int]] = set()
        for y, row in enumerate(rows):
            for x in range(width):
                offset = x * channels
                red, green, blue = row[offset : offset + 3]
                if red > 190 and green < 100 and blue < 100:
                    red_pixels.add((x, y))
        red_components: list[list[tuple[int, int]]] = []
        while red_pixels:
            seed = red_pixels.pop()
            stack = [seed]
            component = [seed]
            while stack:
                x, y = stack.pop()
                for neighbor in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if neighbor in red_pixels:
                        red_pixels.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
            component_y_values = [point[1] for point in component]
            component_x_values = [point[0] for point in component]
            if (
                len(component) >= max(500, int(width * height * 0.0015))
                and (
                    max(component_y_values) - min(component_y_values)
                    >= int(height * 0.20)
                    or max(component_x_values) - min(component_x_values)
                    >= int(width * 0.20)
                )
            ):
                red_components.append(component)
        minimum_red_pixels_per_row = max(10, int(width * 0.012))
        red_zone_row_spans: dict[int, list[list[int]]] = {}
        for component in red_components:
            red_x_values_by_row: dict[int, list[int]] = {}
            for x, y in component:
                red_x_values_by_row.setdefault(y, []).append(x)
            for y, x_values in red_x_values_by_row.items():
                if len(x_values) < minimum_red_pixels_per_row:
                    continue
                red_zone_row_spans.setdefault(y, []).append(
                    [min(x_values), max(x_values)]
                )
        if not red_zone_row_spans:
            raise RuntimeError(
                "Room-wide color segmentation could not resolve the red-zone mask"
            )
        red_zone_bounds: dict[str, int] = {
            "x_min": min(
                span[0]
                for spans in red_zone_row_spans.values()
                for span in spans
            ),
            "y_min": min(red_zone_row_spans),
            "x_max": max(
                span[1]
                for spans in red_zone_row_spans.values()
                for span in spans
            ),
            "y_max": max(red_zone_row_spans),
        }
        resolved_red_zone_mask = {
            "resolution": [width, height],
            "bounds": red_zone_bounds,
            "component_count": len(red_components),
            "row_spans": red_zone_row_spans,
        }
    else:
        width, height = [
            int(value) for value in red_zone_mask["resolution"]
        ]
        red_zone_bounds = {
            key: int(value)
            for key, value in red_zone_mask["bounds"].items()
        }
        red_zone_row_spans = {
            int(y): [[int(span[0]), int(span[1])] for span in spans]
            for y, spans in red_zone_mask["row_spans"].items()
        }
        resolved_red_zone_mask = red_zone_mask

    # Tight semantic boxes are still axis-aligned rectangles. At an oblique
    # camera angle their empty corner pixels can cross the diagonal edge of
    # the painted zone before any rendered forklift pixel does. A small,
    # resolution-scaled inset removes that geometric false positive while
    # retaining the mast, forks, body, and tires in the overlap test.
    red_overlap_inset_pixels = max(
        1,
        int(round(max(width, height) * 0.008)),
    )

    def primitive_box_overlaps_red_zone(box: dict[str, Any]) -> bool:
        box_width = int(box["x_max"]) - int(box["x_min"]) + 1
        box_height = int(box["y_max"]) - int(box["y_min"]) + 1
        inset_x = min(
            red_overlap_inset_pixels,
            max(0, (box_width - 1) // 3),
        )
        inset_y = min(
            red_overlap_inset_pixels,
            max(0, (box_height - 1) // 3),
        )
        y_start = max(0, int(box["y_min"]) + inset_y)
        y_end = min(height - 1, int(box["y_max"]) - inset_y)
        box_x_min = int(box["x_min"]) + inset_x
        box_x_max = int(box["x_max"]) - inset_x
        for y in range(y_start, y_end + 1):
            for span in red_zone_row_spans.get(y, []):
                if box_x_max >= span[0] and box_x_min <= span[1]:
                    return True
        return False

    def box_overlaps_red_zone(box: dict[str, Any]) -> bool:
        component_boxes = box.get("component_boxes") or []
        if component_boxes:
            return any(
                primitive_box_overlaps_red_zone(component)
                for component in component_boxes
            )
        return primitive_box_overlaps_red_zone(box)

    def projected_entry_seconds(
        box: dict[str, Any],
        *,
        velocity_x: float,
        velocity_y: float,
    ) -> float | None:
        sample_seconds = 0.1
        sample_count = int(
            prediction_horizon_seconds / sample_seconds
        )
        for sample_index in range(1, sample_count + 1):
            future_seconds = min(
                prediction_horizon_seconds,
                sample_index * sample_seconds,
            )
            translated = {
                "x_min": int(round(int(box["x_min"]) + velocity_x * future_seconds)),
                "y_min": int(round(int(box["y_min"]) + velocity_y * future_seconds)),
                "x_max": int(round(int(box["x_max"]) + velocity_x * future_seconds)),
                "y_max": int(round(int(box["y_max"]) + velocity_y * future_seconds)),
            }
            if box.get("component_boxes"):
                translated["component_boxes"] = [
                    {
                        "x_min": int(
                            round(
                                int(component["x_min"])
                                + velocity_x * future_seconds
                            )
                        ),
                        "y_min": int(
                            round(
                                int(component["y_min"])
                                + velocity_y * future_seconds
                            )
                        ),
                        "x_max": int(
                            round(
                                int(component["x_max"])
                                + velocity_x * future_seconds
                            )
                        ),
                        "y_max": int(
                            round(
                                int(component["y_max"])
                                + velocity_y * future_seconds
                            )
                        ),
                    }
                    for component in box["component_boxes"]
                ]
            if box_overlaps_red_zone(translated):
                return future_seconds
        return None

    normalized_boxes: list[dict[str, Any]] = []
    for box in forklift_pixel_boxes or []:
        try:
            normalized = {
                "x_min": max(0, min(width - 1, int(box["x_min"]))),
                "y_min": max(0, min(height - 1, int(box["y_min"]))),
                "x_max": max(0, min(width - 1, int(box["x_max"]))),
                "y_max": max(0, min(height - 1, int(box["y_max"]))),
                "semantic_label": str(box["semantic_label"]),
                "forklift_index": int(box["forklift_index"]),
            }
            if box.get("box_source"):
                normalized["box_source"] = str(box["box_source"])
            component_boxes = []
            for component in box.get("component_boxes") or []:
                normalized_component = {
                    "x_min": max(
                        0,
                        min(width - 1, int(component["x_min"])),
                    ),
                    "y_min": max(
                        0,
                        min(height - 1, int(component["y_min"])),
                    ),
                    "x_max": max(
                        0,
                        min(width - 1, int(component["x_max"])),
                    ),
                    "y_max": max(
                        0,
                        min(height - 1, int(component["y_max"])),
                    ),
                }
                if (
                    normalized_component["x_max"]
                    >= normalized_component["x_min"]
                    and normalized_component["y_max"]
                    >= normalized_component["y_min"]
                    and normalized_component not in component_boxes
                ):
                    component_boxes.append(normalized_component)
            if component_boxes:
                normalized["component_boxes"] = component_boxes
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid forklift pixel box: {box!r}") from exc
        if (
            normalized["x_max"] < normalized["x_min"]
            or normalized["y_max"] < normalized["y_min"]
        ):
            raise ValueError(f"Inverted forklift pixel box: {box!r}")
        normalized_boxes.append(normalized)
    if len(normalized_boxes) != expected_forklift_count:
        raise RuntimeError(
            "The detector must resolve "
            f"{expected_forklift_count} forklift box"
            f"{'es' if expected_forklift_count != 1 else ''}"
        )

    previous_by_label: dict[str, dict[str, Any]] = {}
    if previous_observation is not None:
        previous_by_label = {
            str(item["semantic_label"]): item
            for item in previous_observation.get("forklifts", [])
        }
    valid_elapsed = (
        elapsed_seconds is not None
        and elapsed_seconds > 1e-6
        and previous_by_label
    )
    present_red_zone_overlap = False
    forklift_motion_detected = False
    predicted_red_zone_entry = False
    forklift_observations: list[dict[str, Any]] = []
    for box in normalized_boxes:
        label = str(box["semantic_label"])
        center_x = (int(box["x_min"]) + int(box["x_max"])) * 0.5
        center_y = (int(box["y_min"]) + int(box["y_max"])) * 0.5
        current_overlap = box_overlaps_red_zone(box)
        present_red_zone_overlap = (
            present_red_zone_overlap or current_overlap
        )
        velocity_x = 0.0
        velocity_y = 0.0
        speed = 0.0
        motion_detected = False
        time_to_entry = None
        previous = previous_by_label.get(label)
        if valid_elapsed and previous is not None:
            previous_center = previous.get("center", [center_x, center_y])
            velocity_x = (center_x - float(previous_center[0])) / float(
                elapsed_seconds
            )
            velocity_y = (center_y - float(previous_center[1])) / float(
                elapsed_seconds
            )
            speed = (velocity_x * velocity_x + velocity_y * velocity_y) ** 0.5
            motion_detected = speed >= minimum_motion_pixels_per_second
            forklift_motion_detected = (
                forklift_motion_detected or motion_detected
            )
            if not current_overlap and motion_detected:
                time_to_entry = projected_entry_seconds(
                    box,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                )
                if time_to_entry is not None:
                    predicted_red_zone_entry = True
        forklift_observations.append(
            {
                **box,
                "box": {
                    key: box[key]
                    for key in ("x_min", "y_min", "x_max", "y_max")
                },
                "center": [round(center_x, 3), round(center_y, 3)],
                "velocity_pixels_per_second": [
                    round(velocity_x, 3),
                    round(velocity_y, 3),
                ],
                "speed_pixels_per_second": round(speed, 3),
                "motion_detected": motion_detected,
                "present_red_zone_overlap": current_overlap,
                "predicted_red_zone_entry": time_to_entry is not None,
                "time_to_red_zone_seconds": (
                    round(float(time_to_entry), 3)
                    if time_to_entry is not None
                    else None
                ),
            }
        )
    return {
        "source": (
            "roomwide_rgb_red_mask_plus_rgb_vehicle_boxes"
            if any(item.get("box_source") for item in normalized_boxes)
            else "roomwide_rgb_red_mask_plus_semantic_forklift_boxes"
        ),
        "resolution": [width, height],
        "red_zone_pixel_bounds": red_zone_bounds,
        "red_zone_component_count": int(
            resolved_red_zone_mask["component_count"]
        ),
        "red_zone_row_span_count": sum(
            len(spans) for spans in red_zone_row_spans.values()
        ),
        "red_overlap_inset_pixels": red_overlap_inset_pixels,
        "forklifts": forklift_observations,
        "visible_forklift_count": len(forklift_observations),
        "present_red_zone_overlap": present_red_zone_overlap,
        "forklift_motion_detected": forklift_motion_detected,
        "predicted_red_zone_entry": predicted_red_zone_entry,
        "prediction_horizon_seconds": prediction_horizon_seconds,
        "elapsed_since_previous_capture_seconds": (
            round(float(elapsed_seconds), 6)
            if elapsed_seconds is not None
            else None
        ),
        "workers_ignored": True,
        "parcels_ignored": True,
        "fork_cartons_ignored": True,
        "simulator_coordinates_used": False,
        "metric_clearance_used": False,
        "controller_input_used": False,
        "_red_zone_mask": resolved_red_zone_mask,
    }


def build_hazard_prompt(_observation: dict[str, Any]) -> str:
    """Return the visual-only policy without leaking frontend conclusions."""
    return HAZARD_PROMPT


def apply_visual_consistency_gates(
    result: dict[str, Any],
    vision_observation: dict[str, Any],
) -> dict[str, Any]:
    """Reject model states contradicted by the current visual geometry."""

    gated = dict(result)
    model_signal = str(gated["signal"])
    gated["pre_gate_signal"] = model_signal
    gated["gate_applied"] = False
    gated["gate_reason"] = None
    red_overlap_confirmed = bool(
        vision_observation["present_red_zone_overlap"]
    )
    amber_motion_confirmed = bool(
        vision_observation["forklift_motion_detected"]
    )
    if gated["signal"] == "RED" and not red_overlap_confirmed:
        fallback = "AMBER" if amber_motion_confirmed else "GREEN"
        gated["signal"] = fallback
        gated["normalized_answer"] = json.dumps(
            {"signal": fallback},
            separators=(",", ":"),
        )
        gated["gate_applied"] = True
        gated["gate_reason"] = (
            f"Model RED -> {fallback}: no visible red-zone contact"
            + (
                "; forklift motion is visible."
                if amber_motion_confirmed
                else " and no forklift motion is visible."
            )
        )
    elif gated["signal"] == "AMBER" and not amber_motion_confirmed:
        gated["signal"] = "GREEN"
        gated["normalized_answer"] = json.dumps(
            {"signal": "GREEN"},
            separators=(",", ":"),
        )
        gated["gate_applied"] = True
        gated["gate_reason"] = (
            "Model AMBER -> GREEN: no visible forklift motion."
        )
    return gated


def completion_url(server_url: str) -> str:
    return f"{server_url.rstrip('/')}/v1/chat/completions"


def check_model_server(
    server_url: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    health_request = urllib.request.Request(
        f"{server_url.rstrip('/')}/health",
        headers={"Connection": "close"},
    )
    try:
        with urllib.request.urlopen(
            health_request,
            timeout=timeout_seconds,
        ) as response:
            payload = json.load(response)
        if payload.get("status") == "ok":
            return payload
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass

    # GenieX exposes the OpenAI model inventory but not llama.cpp's /health.
    models_request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/models",
        headers={"Connection": "close"},
    )
    with urllib.request.urlopen(
        models_request,
        timeout=timeout_seconds,
    ) as response:
        models = json.load(response)
    if not isinstance(models, dict) or not models.get("data"):
        raise RuntimeError(f"Cosmos model server is not ready: {models!r}")
    return {"status": "ok", "source": "v1/models", "models": models["data"]}


def request_hazard_signal(
    *,
    image_path: Path,
    server_url: str,
    model: str,
    timeout_seconds: float,
    urlopen: Any = urllib.request.urlopen,
    vision_observation: dict[str, Any] | None = None,
    enable_think: bool = True,
    use_signal_grammar: bool = False,
    image_paths: list[Path] | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    request_started = time.monotonic()
    observation = (
        extract_visual_overlap_observation(image_path)
        if vision_observation is None
        else vision_observation
    )
    request_image_paths = image_paths or [image_path]
    request_prompt = prompt or build_hazard_prompt(observation)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    *[
                    {
                        "type": "image_url",
                            "image_url": {"url": image_data_url(path)},
                        }
                        for path in request_image_paths
                    ],
                    {"type": "text", "text": request_prompt},
                ],
            }
        ],
        "max_completion_tokens": 32,
        "enable_think": enable_think,
        "top_k": 1,
        "temperature": 0,
        "seed": 42,
    }
    if use_signal_grammar:
        payload["grammar_string"] = SIGNAL_GRAMMAR
        # llama.cpp uses `grammar`; the EVK GenieX wrapper uses
        # `grammar_string`. Supplying both keeps the request closed-set across
        # the host experimentation and target verification backends.
        payload["grammar"] = SIGNAL_GRAMMAR
    request = urllib.request.Request(
        completion_url(server_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    request_ready = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Model server HTTP {exc.code}: {body[-3000:]}") from exc
    response_received = time.monotonic()
    try:
        raw_answer = str(result["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Model response contains no assistant content") from exc
    signal = SIGNAL_LETTERS.get(raw_answer.upper())
    structured_text = raw_answer
    if structured_text.startswith("```"):
        lines = structured_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        structured_text = "\n".join(lines).strip()
    try:
        structured_answer = json.loads(structured_text)
    except json.JSONDecodeError:
        structured_answer = None

    def find_structured_signal(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ("signal", "prediction", "classification"):
                if key not in value:
                    continue
                candidate = value[key]
                if isinstance(candidate, str) and candidate.upper() in SIGNALS:
                    return candidate.upper()
                nested = find_structured_signal(candidate)
                if nested is not None:
                    return nested
            for candidate in value.values():
                nested = find_structured_signal(candidate)
                if nested is not None:
                    return nested
        elif isinstance(value, list):
            for candidate in value:
                nested = find_structured_signal(candidate)
                if nested is not None:
                    return nested
        return None

    if signal is None:
        signal = find_structured_signal(structured_answer)
    if signal is None:
        named_signal = re.search(
            r'"(?:signal|prediction)"\s*:\s*"(GREEN|AMBER|RED)"',
            raw_answer,
            flags=re.IGNORECASE,
        )
        if named_signal:
            signal = named_signal.group(1).upper()
    if signal is None:
        final_match = re.search(
            r"\bFINAL\s*:\s*(GREEN|AMBER|RED)\b",
            raw_answer,
            flags=re.IGNORECASE,
        )
        if final_match:
            signal = final_match.group(1).upper()
    if signal is None:
        # Some OpenAI-compatible runtimes omit FINAL for an otherwise complete
        # answer. Accept a signal only at the end; never mistake a color in a
        # truncated chain of thought for the classification.
        tail_match = re.search(
            r"\b(GREEN|AMBER|RED)\b[\s.!]*$",
            raw_answer,
            flags=re.IGNORECASE,
        )
        if not tail_match:
            raise RuntimeError(
                f"Model answer has no final signal: {raw_answer!r}"
            )
        signal = tail_match.group(1).upper()
    canonical_answer = json.dumps({"signal": signal}, separators=(",", ":"))
    response_parsed = time.monotonic()
    return {
        "signal": signal,
        "raw_answer": raw_answer,
        "normalized_answer": canonical_answer,
        "vision_observation": observation,
        "model_image_path": str(image_path),
        "model_image_paths": [str(path) for path in request_image_paths],
        "prompt_sent": request_prompt,
        "latency_seconds": round(response_received - request_ready, 6),
        "request_prepare_seconds": round(request_ready - request_started, 6),
        "server_roundtrip_seconds": round(
            response_received - request_ready, 6
        ),
        "response_parse_seconds": round(
            response_parsed - response_received, 6
        ),
        "total_request_seconds": round(
            response_parsed - request_started, 6
        ),
        "usage": result.get("usage"),
    }


def _new_run_directory(output_root: Path) -> tuple[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return run_id, output_dir


def wait_for_timeline_playing(
    client: IsaacTcpClient,
    *,
    poll_seconds: float,
    state_reader: Any | None = None,
    on_wait: Any | None = None,
    sleep_fn: Any = time.sleep,
) -> dict[str, Any]:
    """Wait without capturing or requesting inference until Isaac is playing."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if state_reader is None:
        state_reader = lambda active_client: execute_isaac_source(
            active_client,
            get_conveyor_safety_state_source(),
        )
    while True:
        state = state_reader(client)
        if bool(state.get("timeline_playing")):
            return state
        if on_wait is not None:
            on_wait(state)
        sleep_fn(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend-label",
        choices=("host", "evk"),
        default="host",
        help="Audit label only; any OpenAI-compatible URL can be used.",
    )
    parser.add_argument(
        "--perception-mode",
        choices=("direct", "hybrid"),
        default="direct",
        help=(
            "direct sends rolling images and the policy to Reason2; hybrid is "
            "a legacy single-image ablation. Neither mode sends detector facts."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=tuple(name for name in CAMERA_PATHS if name != "presentation"),
        default="detector_endline",
    )
    parser.add_argument("--create-scene", action="store_true")
    parser.add_argument("--max-responses", type=int, default=30)
    parser.add_argument("--max-runtime-seconds", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--consensus-votes", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--model-image-width", type=int)
    parser.add_argument("--model-image-height", type=int)
    parser.add_argument(
        "--play-mode-only",
        action="store_true",
        help=(
            "Keep the runner alive while Isaac is paused, but only capture and "
            "request inference while the timeline is playing."
        ),
    )
    parser.add_argument(
        "--idle-poll-seconds",
        type=float,
        default=0.2,
        help="Timeline polling interval while --play-mode-only is waiting.",
    )
    parser.add_argument(
        "--prediction-horizon-seconds",
        type=float,
        default=DEFAULT_PREDICTION_HORIZON_SECONDS,
        help=(
            "Constant-velocity image-space look-ahead retained for diagnostic "
            "projected-entry metrics; AMBER now means any forklift motion."
        ),
    )
    parser.add_argument(
        "--capture-render-steps",
        type=int,
        default=1,
        choices=(1, 2, 3, 4),
        help="Replicator render iterations per detector frame.",
    )
    parser.add_argument(
        "--capture-rt-subframes",
        type=int,
        default=1,
        choices=tuple(range(1, 17)),
        help="RTX subframes per detector render iteration.",
    )
    parser.add_argument("--capture-width", type=int, default=512)
    parser.add_argument("--capture-height", type=int, default=288)
    parser.add_argument(
        "--maximum-capture-fps",
        type=float,
        default=0.0,
        help=(
            "Upper bound for detector captures; zero keeps the existing "
            "inference-limited cadence."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_responses < 1:
        raise ValueError("--max-responses must be at least 1")
    if args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be positive")
    if args.idle_poll_seconds <= 0:
        raise ValueError("--idle-poll-seconds must be positive")
    if args.prediction_horizon_seconds <= 0:
        raise ValueError("--prediction-horizon-seconds must be positive")
    if not 0.0 <= args.maximum_capture_fps <= 30.0:
        raise ValueError("--maximum-capture-fps must be between 0 and 30")
    if (args.model_image_width is None) != (args.model_image_height is None):
        raise ValueError(
            "--model-image-width and --model-image-height must be provided together"
        )
    if args.model_image_width is None:
        model_image_size = (384, 216) if args.backend_label == "evk" else None
    else:
        model_image_size = (args.model_image_width, args.model_image_height)
        if model_image_size[0] < 32 or model_image_size[1] < 32:
            raise ValueError("model image dimensions must be at least 32 pixels")
    capture_resolution = (args.capture_width, args.capture_height)
    if any(not 128 <= value <= 2048 for value in capture_resolution):
        raise ValueError("capture dimensions must be between 128 and 2048 pixels")
    resize_required = (
        model_image_size is not None
        and model_image_size != capture_resolution
    )
    minimum_capture_interval_seconds = (
        1.0 / args.maximum_capture_fps
        if args.maximum_capture_fps > 0.0
        else 0.0
    )

    check_model_server(args.server_url, timeout_seconds=args.timeout_seconds)
    client = IsaacTcpClient(timeout_seconds=max(30.0, args.timeout_seconds))
    if args.create_scene:
        execute_isaac_source(client, enable_conveyor_safety_extension_source())
        execute_isaac_source(
            client,
            create_conveyor_safety_scene_source(
                {"new_stage": True, "start_playing": True}
            ),
        )
    initial_state = execute_isaac_source(client, get_conveyor_safety_state_source())

    run_id, output_dir = _new_run_directory(args.output_root.resolve())
    history: list[dict[str, Any]] = []
    votes: deque[str] = deque(maxlen=args.consensus_votes)
    accepted_signal = str(initial_state.get("model_signal") or "GREEN")
    started = time.monotonic()
    previous_observation: dict[str, Any] | None = None
    previous_capture_finished_monotonic: float | None = None
    previous_capture_started_monotonic: float | None = None
    rolling_frame_paths: deque[str] = deque(maxlen=2)
    rolling_model_frame_paths: deque[Path] = deque(maxlen=2)
    cached_red_zone_mask: dict[str, Any] | None = None

    status: dict[str, Any] = {
        "schema_version": "1.1",
        "run_id": run_id,
        "state": "running",
        "phase": "ready",
        "updated_epoch": time.time(),
        "backend": args.backend_label,
        "server_url": args.server_url,
        "model": args.model,
        "camera": args.camera,
        "prompt": HAZARD_PROMPT,
        "perception_mode": args.perception_mode,
        "reason2_role": (
            "end_to_end_visual_physical_reasoning"
            if args.perception_mode == "direct"
            else "single_frame_visual_reasoning_ablation"
        ),
        "vision_frontend": (
            "hidden semantic scoring oracle; no annotations or facts sent to model"
        ),
        "stack_light_visible_to_model": True,
        "stack_light_color_bias_monitoring": True,
        "ground_truth_sent_to_model": False,
        "simulator_coordinates_sent_to_model": False,
        "model_image_resolution": (
            list(model_image_size)
            if model_image_size is not None
            else list(capture_resolution)
        ),
        "capture_resolution": list(capture_resolution),
        "enable_think": False,
        "signal_grammar_enabled": True,
        "consensus_votes": args.consensus_votes,
        "prediction_horizon_seconds": args.prediction_horizon_seconds,
        "workers_ignored": True,
        "play_mode_only": args.play_mode_only,
        "capture_render_steps": args.capture_render_steps,
        "capture_rt_subframes": args.capture_rt_subframes,
        "maximum_capture_fps": args.maximum_capture_fps,
        "minimum_capture_interval_seconds": minimum_capture_interval_seconds,
        "current_inference_id": None,
        "current_prompt": HAZARD_PROMPT,
        "input_image_paths": [],
        "model_input_image_path": None,
        "inference_started_epoch": None,
        "last_inference_seconds": None,
        "last_raw_output": None,
        "last_raw_model_signal": None,
        "last_model_signal": None,
        "last_decision_trace": None,
        "stack_light_status": accepted_signal,
        "responses": history,
    }
    _atomic_write_json(args.status_path.resolve(), status)

    for request_index in range(args.max_responses):
        if time.monotonic() - started >= args.max_runtime_seconds:
            break
        if args.play_mode_only:
            def publish_paused(_state: dict[str, Any]) -> None:
                status.update(
                    {
                        "phase": "paused",
                        "updated_epoch": time.time(),
                        "inference_started_epoch": None,
                    }
                )
                _atomic_write_json(args.status_path.resolve(), status)

            timeline_state = execute_isaac_source(
                client,
                get_conveyor_safety_state_source(),
            )
            if not bool(timeline_state.get("timeline_playing")):
                publish_paused(timeline_state)
            wait_for_timeline_playing(
                client,
                poll_seconds=args.idle_poll_seconds,
                on_wait=publish_paused,
            )
            status.update(
                {
                    "phase": "ready",
                    "updated_epoch": time.time(),
                }
            )
            _atomic_write_json(args.status_path.resolve(), status)
            if time.monotonic() - started >= args.max_runtime_seconds:
                break
        inference_id = f"{run_id}-{request_index + 1:03d}"
        filename = f"{inference_id}_{args.camera}.png"
        stack_light_signal_at_capture = accepted_signal
        status.update(
            {
                "phase": "capturing",
                "updated_epoch": time.time(),
                "current_inference_id": inference_id,
                "inference_started_epoch": None,
                "stack_light_signal_at_capture": stack_light_signal_at_capture,
            }
        )
        _atomic_write_json(args.status_path.resolve(), status)
        capture_cadence_wait_seconds = 0.0
        if previous_capture_started_monotonic is not None:
            capture_deadline = (
                previous_capture_started_monotonic
                + minimum_capture_interval_seconds
            )
            capture_cadence_wait_seconds = max(
                0.0,
                capture_deadline - time.monotonic(),
            )
            if capture_cadence_wait_seconds > 0.0:
                time.sleep(capture_cadence_wait_seconds)
        capture_started_monotonic = time.monotonic()
        previous_capture_started_monotonic = capture_started_monotonic
        capture = capture_conveyor_frame(
            client,
            camera=args.camera,
            filename=filename,
            render_steps=args.capture_render_steps,
            rt_subframes=args.capture_rt_subframes,
            width=capture_resolution[0],
            height=capture_resolution[1],
        )
        capture_finished_monotonic = time.monotonic()
        image_path = Path(capture["output_path"]).resolve()
        expected_forklift_indices = tuple(
            int(index)
            for index in (
                capture.get("enabled_forklift_indices") or (1, 2, 3)
            )
        )
        expected_forklift_count = len(expected_forklift_indices)
        decode_started_monotonic = time.monotonic()
        semantic_box_count = len(capture.get("forklift_pixel_boxes") or [])
        decoded_image = (
            _read_png_rgb(image_path)
            if (
                cached_red_zone_mask is None
                or resize_required
                or semantic_box_count != expected_forklift_count
            )
            else None
        )
        decode_finished_monotonic = time.monotonic()
        model_image_path = image_path
        resize_started_monotonic = time.monotonic()
        if resize_required and model_image_size is not None:
            model_image_path = resize_png_nearest(
                image_path,
                image_path.with_name(
                    f"{image_path.stem}_model_"
                    f"{model_image_size[0]}x{model_image_size[1]}.png"
                ),
                width=model_image_size[0],
                height=model_image_size[1],
                decoded_image=decoded_image,
            )
        resize_finished_monotonic = time.monotonic()
        state_read_started_monotonic = time.monotonic()
        if all(
            key in capture
            for key in (
                "capture_ground_truth_signal",
                "capture_min_forklift_clearance_m",
                "capture_active_forklift",
            )
        ):
            capture_state = {
                "ground_truth_signal": capture[
                    "capture_ground_truth_signal"
                ],
                "min_forklift_clearance_m": float(
                    capture["capture_min_forklift_clearance_m"]
                ),
                "active_forklift": int(
                    capture["capture_active_forklift"]
                ),
            }
        else:
            # Compatibility path for an already-running older Isaac tool
            # source. New captures fold these values into the render call.
            capture_state = execute_isaac_source(
                client,
                get_conveyor_safety_state_source(),
            )
        state_read_finished_monotonic = time.monotonic()
        elapsed_since_previous = (
            capture_started_monotonic - previous_capture_finished_monotonic
            if previous_capture_finished_monotonic is not None
            else None
        )
        frontend_started_monotonic = time.monotonic()
        forklift_boxes = recover_forklift_boxes_from_rgb(
            image_path,
            capture.get("forklift_pixel_boxes"),
            previous_observation=previous_observation,
            decoded_image=decoded_image,
            expected_forklift_indices=expected_forklift_indices,
        )
        vision_observation = extract_forklift_motion_observation(
            image_path,
            forklift_boxes,
            previous_observation=previous_observation,
            elapsed_seconds=elapsed_since_previous,
            prediction_horizon_seconds=args.prediction_horizon_seconds,
            decoded_image=decoded_image,
            red_zone_mask=cached_red_zone_mask,
            expected_forklift_count=expected_forklift_count,
        )
        cached_red_zone_mask = vision_observation.pop("_red_zone_mask")
        frontend_finished_monotonic = time.monotonic()
        previous_observation = vision_observation
        previous_capture_finished_monotonic = capture_finished_monotonic
        rolling_frame_paths.append(str(image_path))
        rolling_model_frame_paths.append(model_image_path)
        current_prompt = HAZARD_PROMPT
        status.update(
            {
                "phase": "inference",
                "updated_epoch": time.time(),
                "current_inference_id": inference_id,
                "current_prompt": current_prompt,
                "input_image_paths": list(rolling_frame_paths),
                "model_input_image_path": str(model_image_path),
                "inference_started_epoch": time.time(),
            }
        )
        _atomic_write_json(args.status_path.resolve(), status)
        try:
            primary_result = request_hazard_signal(
                image_path=model_image_path,
                server_url=args.server_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                vision_observation=vision_observation,
                enable_think=False,
                use_signal_grammar=True,
                image_paths=(
                    list(rolling_model_frame_paths)
                    if args.perception_mode == "direct"
                    else [model_image_path]
                ),
                prompt=current_prompt,
            )
            primary_model_signal = primary_result["signal"]
            red_confirmation_result: dict[str, Any] | None = None
            if (
                vision_observation["present_red_zone_overlap"]
                and primary_model_signal != "RED"
            ):
                current_prompt = (
                    HAZARD_PROMPT
                    + "\nConditional red confirmation:\n"
                    + RED_CONFIRMATION_PROMPT
                )
                status.update(
                    {
                        "current_prompt": current_prompt,
                        "updated_epoch": time.time(),
                    }
                )
                _atomic_write_json(args.status_path.resolve(), status)
                red_confirmation_result = request_hazard_signal(
                    image_path=model_image_path,
                    server_url=args.server_url,
                    model=args.model,
                    timeout_seconds=args.timeout_seconds,
                    vision_observation=vision_observation,
                    enable_think=False,
                    use_signal_grammar=True,
                    image_paths=[model_image_path],
                    prompt=RED_CONFIRMATION_PROMPT,
                )
            result = dict(primary_result)
            if red_confirmation_result is not None:
                confirmation_signal = red_confirmation_result["signal"]
                if confirmation_signal == "RED":
                    result["signal"] = "RED"
                    result["normalized_answer"] = json.dumps(
                        {"signal": "RED"},
                        separators=(",", ":"),
                    )
                result["raw_answer"] = (
                    f"primary={primary_result['raw_answer']}; "
                    "red_confirmation="
                    f"{red_confirmation_result['raw_answer']}"
                )
                result["prompt_sent"] = current_prompt
                for timing_key in (
                    "latency_seconds",
                    "request_prepare_seconds",
                    "server_roundtrip_seconds",
                    "response_parse_seconds",
                    "total_request_seconds",
                ):
                    result[timing_key] = round(
                        float(primary_result[timing_key])
                        + float(red_confirmation_result[timing_key]),
                        6,
                    )
            amber_confirmation_result: dict[str, Any] | None = None
            if (
                vision_observation["forklift_motion_detected"]
                and not vision_observation["present_red_zone_overlap"]
                and result["signal"] != "AMBER"
            ):
                current_prompt = (
                    HAZARD_PROMPT
                    + "\nConditional amber confirmation:\n"
                    + AMBER_CONFIRMATION_PROMPT
                )
                status.update(
                    {
                        "current_prompt": current_prompt,
                        "updated_epoch": time.time(),
                    }
                )
                _atomic_write_json(args.status_path.resolve(), status)
                amber_confirmation_result = request_hazard_signal(
                    image_path=model_image_path,
                    server_url=args.server_url,
                    model=args.model,
                    timeout_seconds=args.timeout_seconds,
                    vision_observation=vision_observation,
                    enable_think=False,
                    use_signal_grammar=True,
                    image_paths=list(rolling_model_frame_paths),
                    prompt=AMBER_CONFIRMATION_PROMPT,
                )
                if amber_confirmation_result["signal"] == "AMBER":
                    result["signal"] = "AMBER"
                    result["normalized_answer"] = json.dumps(
                        {"signal": "AMBER"},
                        separators=(",", ":"),
                    )
                result["raw_answer"] = (
                    f"{result['raw_answer']}; "
                    "amber_confirmation="
                    f"{amber_confirmation_result['raw_answer']}"
                )
                result["prompt_sent"] = current_prompt
                for timing_key in (
                    "latency_seconds",
                    "request_prepare_seconds",
                    "server_roundtrip_seconds",
                    "response_parse_seconds",
                    "total_request_seconds",
                ):
                    result[timing_key] = round(
                        float(result[timing_key])
                        + float(amber_confirmation_result[timing_key]),
                        6,
                    )
            amber_motion_confirmed = bool(
                vision_observation["forklift_motion_detected"]
            )
            red_overlap_confirmed = bool(
                vision_observation["present_red_zone_overlap"]
            )
            result = apply_visual_consistency_gates(
                result,
                vision_observation,
            )
        except Exception as exc:
            status.update(
                {
                    "state": "failed",
                    "phase": "failed",
                    "updated_epoch": time.time(),
                    "inference_started_epoch": None,
                    "last_raw_output": f"{type(exc).__name__}: {exc}",
                }
            )
            _atomic_write_json(args.status_path.resolve(), status)
            raise
        inference_finished_monotonic = time.monotonic()
        votes.append(result["signal"])
        consensus_reached = (
            len(votes) == args.consensus_votes
            and len(set(votes)) == 1
        )
        apply_started_monotonic = time.monotonic()
        if consensus_reached:
            accepted_signal = votes[-1]
            apply_state = execute_isaac_source(
                client,
                set_conveyor_safety_light_and_get_state_source(
                    {
                        "signal": accepted_signal,
                        "confidence": 1.0,
                        "inference_id": inference_id,
                    }
                ),
            )
        else:
            apply_state = execute_isaac_source(
                client,
                get_conveyor_safety_state_source(),
            )
        apply_finished_monotonic = time.monotonic()
        frontend_signal = (
            "RED"
            if vision_observation["present_red_zone_overlap"]
            else (
                "AMBER"
                if vision_observation["forklift_motion_detected"]
                else "GREEN"
            )
        )
        record = {
            "request_index": request_index + 1,
            "inference_id": inference_id,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "image_path": str(image_path),
            "model_image_path": result["model_image_path"],
            "model_image_paths": result["model_image_paths"],
            "camera": args.camera,
            "capture_render_resource_reused": capture.get(
                "render_resource_reused"
            ),
            "capture_render_product_updates_disabled_between_captures":
                capture.get(
                    "render_product_updates_disabled_between_captures"
                ),
            "model_signal": result["signal"],
            "pre_gate_signal": result["pre_gate_signal"],
            "primary_model_signal": primary_model_signal,
            "gate_applied": result["gate_applied"],
            "gate_reason": result["gate_reason"],
            "red_confirmation_triggered": (
                red_confirmation_result is not None
            ),
            "red_confirmation_signal": (
                red_confirmation_result["signal"]
                if red_confirmation_result is not None
                else None
            ),
            "red_overlap_confirmed": red_overlap_confirmed,
            "amber_confirmation_triggered": (
                amber_confirmation_result is not None
            ),
            "amber_confirmation_signal": (
                amber_confirmation_result["signal"]
                if amber_confirmation_result is not None
                else None
            ),
            "amber_motion_confirmed": amber_motion_confirmed,
            "forklift_motion_detected": vision_observation[
                "forklift_motion_detected"
            ],
            "frontend_signal": frontend_signal,
            "frontend_match": (
                frontend_signal == capture_state["ground_truth_signal"]
            ),
            "semantic_forklift_box_count": sum(
                1
                for item in vision_observation["forklifts"]
                if not item.get("box_source")
            ),
            "expected_forklift_count": expected_forklift_count,
            "fallback_forklift_box_count": sum(
                1
                for item in vision_observation["forklifts"]
                if item.get("box_source")
            ),
            "accepted_signal": accepted_signal,
            "light_mutated": apply_state.get("last_light_mutated"),
            "light_mutation_count": apply_state.get("light_mutation_count"),
            "stack_light_signal_at_capture": stack_light_signal_at_capture,
            "model_matched_visible_stack_light": (
                result["signal"] == stack_light_signal_at_capture
            ),
            "possible_stack_light_color_bias": (
                result["signal"] == stack_light_signal_at_capture
                and result["signal"] != frontend_signal
            ),
            "consensus_reached": consensus_reached,
            "vote_window": list(votes),
            "latency_seconds": result["latency_seconds"],
            "stage_seconds": {
                "capture_cadence_wait": round(
                    capture_cadence_wait_seconds,
                    6,
                ),
                "capture_wall": round(
                    capture_finished_monotonic - capture_started_monotonic, 6
                ),
                "capture_internal": capture.get("capture_stage_seconds"),
                "decode": round(
                    decode_finished_monotonic - decode_started_monotonic, 6
                ),
                "resize": round(
                    resize_finished_monotonic - resize_started_monotonic, 6
                ),
                "state_read": round(
                    state_read_finished_monotonic
                    - state_read_started_monotonic,
                    6,
                ),
                "vision_frontend": round(
                    frontend_finished_monotonic - frontend_started_monotonic,
                    6,
                ),
                "request_prepare": result["request_prepare_seconds"],
                "model_roundtrip": result["server_roundtrip_seconds"],
                "response_parse": result["response_parse_seconds"],
                "apply": round(
                    apply_finished_monotonic - apply_started_monotonic, 6
                ),
                "capture_to_light": round(
                    apply_finished_monotonic - capture_finished_monotonic, 6
                ),
                "cycle": round(
                    apply_finished_monotonic - capture_started_monotonic, 6
                ),
            },
            "capture_ground_truth_signal": capture_state["ground_truth_signal"],
            "capture_min_clearance_m": capture_state[
                "min_forklift_clearance_m"
            ],
            "capture_active_forklift": capture_state["active_forklift"],
            "apply_ground_truth_signal": apply_state["ground_truth_signal"],
            "apply_min_clearance_m": apply_state["min_forklift_clearance_m"],
            "capture_match": (
                result["signal"] == capture_state["ground_truth_signal"]
            ),
            "raw_answer": result["raw_answer"],
            "vision_observation": result["vision_observation"],
            "usage": result["usage"],
        }
        history.append(record)
        stack_light_bias_suspect_count = sum(
            1
            for item in history
            if item["possible_stack_light_color_bias"]
        )
        status.update(
            {
                "phase": "applied",
                "updated_epoch": time.time(),
                "last_response": record,
                "last_inference_seconds": result["latency_seconds"],
                "last_raw_output": result["raw_answer"],
                "last_raw_model_signal": result["pre_gate_signal"],
                "last_model_signal": result["signal"],
                "last_decision_trace": result["gate_reason"],
                "stack_light_status": accepted_signal,
                "stack_light_bias_suspect_count": (
                    stack_light_bias_suspect_count
                ),
                "inference_started_epoch": None,
                "accepted_signal": accepted_signal,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "responses": history[-20:],
            }
        )
        _atomic_write_json(args.status_path.resolve(), status)
        print(
            f"{inference_id} model={result['signal']} "
            f"capture_truth={capture_state['ground_truth_signal']} "
            f"clearance={capture_state['min_forklift_clearance_m']:.2f}m "
            f"latency={result['latency_seconds']:.2f}s "
            f"cycle={record['stage_seconds']['cycle']:.2f}s",
            flush=True,
        )

    presentation_filename = f"{run_id}_presentation.png"
    presentation = execute_isaac_source(
        client,
        capture_conveyor_safety_camera_source(
            {"camera": "presentation", "filename": presentation_filename}
        ),
    )
    matches = sum(1 for item in history if item["capture_match"])
    frontend_matches = sum(1 for item in history if item["frontend_match"])
    complete_semantic_frames = sum(
        1
        for item in history
        if item["semantic_forklift_box_count"]
        == item["expected_forklift_count"]
    )

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(
            (len(ordered) - 1) * fraction
        ))))
        return round(ordered[index], 6)

    timing_summary: dict[str, dict[str, float | None]] = {}
    for stage_name in (
        "capture_wall",
        "decode",
        "resize",
        "state_read",
        "vision_frontend",
        "request_prepare",
        "model_roundtrip",
        "apply",
        "capture_to_light",
        "cycle",
    ):
        values = [
            float(item["stage_seconds"][stage_name])
            for item in history
        ]
        timing_summary[stage_name] = {
            "mean": round(sum(values) / len(values), 6) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
    summary = {
        **status,
        "state": "complete",
        "phase": "complete",
        "updated_epoch": time.time(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "response_count": len(history),
        "capture_matches": matches,
        "capture_accuracy": round(matches / len(history), 4) if history else None,
        "frontend_matches": frontend_matches,
        "frontend_accuracy": (
            round(frontend_matches / len(history), 4) if history else None
        ),
        "complete_semantic_frames": complete_semantic_frames,
        "semantic_actor_identification_rate": (
            round(complete_semantic_frames / len(history), 4)
            if history else None
        ),
        "timing_summary": timing_summary,
        "presentation_path": presentation["output_path"],
        "responses": history,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(args.status_path.resolve(), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
