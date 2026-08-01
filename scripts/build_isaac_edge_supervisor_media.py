#!/usr/bin/env python3
"""Build tutorial GIF and image panels from Isaac Sim viewport captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.isaac_sim_mcp.edge_supervisor import (
    SCENARIO_LABELS,
    SCENARIOS,
    supervisor_decision,
)


CAMERA_LABELS = {
    "overview": "Roof overview",
    "west_aisle": "West aisle camera",
    "intersection": "Intersection camera",
    "east_aisle": "East aisle camera",
}

TACTICAL_CROPS = {
    "mixed_traffic": (0.18, 0.12, 0.82, 0.90),
    "aisle_congestion": (0.20, 0.06, 0.74, 0.60),
    "blind_corner": (0.14, 0.10, 0.76, 0.88),
}

ACTOR_LEGEND = (
    ("Robot Blue", (52, 150, 255)),
    ("Robot Green", (75, 220, 118)),
    ("Forklift", (255, 150, 48)),
    ("Worker", (255, 225, 74)),
)


def _pillow() -> tuple[Any, Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required. Install it with `python -m pip install Pillow`."
        ) from error
    return Image, ImageDraw, ImageFont, ImageOps


def _font(ImageFont: Any, size: int, *, bold: bool = False) -> Any:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _banner_colors(action: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return {
        "CONTINUE": ((19, 92, 48, 225), (118, 255, 167, 255)),
        "YIELD": ((107, 73, 7, 230), (255, 220, 92, 255)),
        "STOP": ((112, 18, 18, 235), (255, 156, 145, 255)),
        "REROUTE": ((77, 25, 110, 230), (225, 169, 255, 255)),
    }[action]


def _load_evk_timing_evidence(path: Path) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read EVK timing evidence: {path}") from error

    result = evidence.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"EVK timing evidence has no result object: {path}")
    request_seconds = result.get("request_seconds")
    total_seconds = result.get("total_seconds")
    if (
        not isinstance(request_seconds, list)
        or not request_seconds
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in request_seconds
        )
        or isinstance(total_seconds, bool)
        or not isinstance(total_seconds, (int, float))
    ):
        raise RuntimeError(f"EVK timing evidence has invalid timings: {path}")

    simulation = evidence.get("simulation")
    reference_scenario = (
        simulation.get("scenario", "unknown")
        if isinstance(simulation, dict)
        else "unknown"
    )
    return {
        "request_seconds": [float(value) for value in request_seconds],
        "total_seconds": float(total_seconds),
        "reference_scenario": str(reference_scenario),
        "report_id": str(evidence.get("report_id", path.stem)),
    }


def _notification_frame(
    image: Any,
    progress: float,
    *,
    scenario: str,
    camera_label: str,
    output_size: tuple[int, int],
) -> Any:
    Image, ImageDraw, ImageFont, ImageOps = _pillow()
    canvas = ImageOps.fit(
        image.convert("RGB"),
        output_size,
        method=Image.Resampling.LANCZOS,
    ).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    decision = supervisor_decision(progress, scenario)
    background, accent = _banner_colors(decision["action"])
    banner_height = max(78, output_size[1] // 7)
    draw.rectangle((0, 0, output_size[0], banner_height), fill=background)
    draw.rectangle((0, banner_height - 6, output_size[0], banner_height), fill=accent)

    title_font = _font(ImageFont, max(18, output_size[0] // 35), bold=True)
    body_font = _font(ImageFont, max(13, output_size[0] // 55))
    draw.text(
        (18, 10),
        (
            f"ROBOT BLUE: {decision['action']} | "
            f"ROUTE: {decision['route'].replace('_', ' ').upper()}"
        ),
        fill=accent,
        font=title_font,
    )
    draw.text(
        (18, 45),
        decision["reason"],
        fill=(245, 248, 250, 255),
        font=body_font,
    )
    scenario_font = _font(ImageFont, max(11, output_size[0] // 72), bold=True)
    draw.text(
        (18, output_size[1] - 31),
        SCENARIO_LABELS[scenario],
        fill=(230, 238, 244, 255),
        font=scenario_font,
        stroke_width=2,
        stroke_fill=(10, 15, 20, 230),
    )
    camera_font = _font(ImageFont, max(12, output_size[0] // 68), bold=True)
    label_box = draw.textbbox((0, 0), camera_label, font=camera_font)
    label_width = label_box[2] - label_box[0]
    draw.rounded_rectangle(
        (
            output_size[0] - label_width - 30,
            output_size[1] - 39,
            output_size[0] - 10,
            output_size[1] - 10,
        ),
        radius=6,
        fill=(12, 18, 24, 205),
    )
    draw.text(
        (output_size[0] - label_width - 20, output_size[1] - 34),
        camera_label,
        fill=(225, 235, 242, 255),
        font=camera_font,
    )
    return canvas.convert("RGB")


def _tactical_frame(
    image: Any,
    progress: float,
    *,
    scenario: str,
    output_size: tuple[int, int],
    evk_timing: dict[str, Any] | None,
) -> Any:
    Image, ImageDraw, ImageFont, ImageOps = _pillow()
    width, height = output_size
    source = image.convert("RGB")
    decision = supervisor_decision(progress, scenario)
    background, accent = _banner_colors(decision["action"])

    canvas = Image.new("RGBA", output_size, (8, 13, 19, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    header_height = max(92, height // 9)
    draw.rectangle((0, 0, width, header_height), fill=background)
    draw.rectangle(
        (0, header_height - 6, width, header_height),
        fill=accent,
    )

    title_font = _font(ImageFont, max(26, width // 46), bold=True)
    body_font = _font(ImageFont, max(17, width // 78))
    label_font = _font(ImageFont, max(15, width // 90), bold=True)
    detail_font = _font(ImageFont, max(15, width // 96))
    draw.text(
        (24, 12),
        (
            f"{decision['action']}  |  "
            f"{decision['route'].replace('_', ' ').upper()}"
        ),
        fill=accent,
        font=title_font,
    )
    draw.text(
        (24, 55),
        decision["reason"],
        fill=(245, 248, 250, 255),
        font=body_font,
    )
    scenario_label = SCENARIO_LABELS[scenario].upper()
    scenario_box = draw.textbbox((0, 0), scenario_label, font=label_font)
    draw.text(
        (width - (scenario_box[2] - scenario_box[0]) - 24, 22),
        scenario_label,
        fill=(230, 238, 244, 255),
        font=label_font,
    )

    margin = 20
    gutter = 20
    main_width = round(width * 2 / 3)
    main_height = round(main_width * 9 / 16)
    main_x = margin
    main_y = header_height + 16
    available_bottom = height - 126
    if main_y + main_height > available_bottom:
        main_height = available_bottom - main_y
        main_width = round(main_height * 16 / 9)
    main = ImageOps.fit(
        source,
        (main_width, main_height),
        method=Image.Resampling.LANCZOS,
    ).convert("RGBA")
    canvas.alpha_composite(main, (main_x, main_y))
    draw.rectangle(
        (main_x, main_y, main_x + main_width, main_y + main_height),
        outline=(205, 220, 230, 230),
        width=3,
    )
    draw.rounded_rectangle(
        (main_x + 12, main_y + 12, main_x + 177, main_y + 47),
        radius=6,
        fill=(10, 15, 20, 220),
    )
    draw.text(
        (main_x + 24, main_y + 20),
        "WIDE ROOF VIEW",
        fill=(235, 242, 246, 255),
        font=label_font,
    )
    side_x = main_x + main_width + gutter
    side_width = width - side_x - margin
    crop_fraction = TACTICAL_CROPS[scenario]
    crop_box = (
        round(source.width * crop_fraction[0]),
        round(source.height * crop_fraction[1]),
        round(source.width * crop_fraction[2]),
        round(source.height * crop_fraction[3]),
    )
    focus_height = round(side_width * 9 / 16)
    focus = ImageOps.fit(
        source.crop(crop_box),
        (side_width, focus_height),
        method=Image.Resampling.LANCZOS,
    ).convert("RGBA")
    canvas.alpha_composite(focus, (side_x, main_y))
    draw.rectangle(
        (side_x, main_y, side_x + side_width, main_y + focus_height),
        outline=accent,
        width=4,
    )
    draw.rounded_rectangle(
        (side_x + 12, main_y + 12, side_x + 198, main_y + 47),
        radius=6,
        fill=(10, 15, 20, 220),
    )
    draw.text(
        (side_x + 24, main_y + 20),
        "CONFLICT-AREA ZOOM",
        fill=accent,
        font=label_font,
    )

    card_y = main_y + focus_height + 18
    card_bottom = main_y + main_height
    draw.rounded_rectangle(
        (side_x, card_y, side_x + side_width, card_bottom),
        radius=10,
        fill=(13, 20, 28, 245),
        outline=(68, 86, 100, 230),
        width=2,
    )
    detail_x = side_x + 22
    text_y = card_y + 18
    for label, value in (
        ("RECIPIENT", decision["recipient"]),
        ("HAZARD", decision["hazard"].replace("_", " ")),
        ("ACTION", decision["action"]),
        ("ROUTE", decision["route"].replace("_", " ")),
    ):
        draw.text(
            (detail_x, text_y),
            label,
            fill=(148, 170, 184, 255),
            font=label_font,
        )
        draw.text(
            (detail_x + side_width // 3, text_y),
            value.upper(),
            fill=accent if label in {"ACTION", "ROUTE"} else (240, 245, 248, 255),
            font=label_font,
        )
        text_y += max(38, height // 20)

    timing_font = _font(ImageFont, max(12, width // 112), bold=True)
    timing_detail_font = _font(ImageFont, max(11, width // 128))
    timing_y = card_bottom - 106
    draw.line(
        (detail_x, timing_y, side_x + side_width - 22, timing_y),
        fill=(68, 86, 100, 230),
        width=2,
    )
    draw.text(
        (detail_x, timing_y + 9),
        "EVK NPU REFERENCE",
        fill=(148, 170, 184, 255),
        font=timing_font,
    )
    if evk_timing:
        total_seconds = evk_timing["total_seconds"]
        request_seconds = evk_timing["request_seconds"]
        total_label = f"{total_seconds:.2f} s TOTAL"
        total_box = draw.textbbox((0, 0), total_label, font=timing_font)
        draw.text(
            (
                side_x + side_width - 22 - (total_box[2] - total_box[0]),
                timing_y + 9,
            ),
            total_label,
            fill=(116, 211, 255, 255),
            font=timing_font,
        )

        timing_bar_x0 = detail_x
        timing_bar_x1 = side_x + side_width - 22
        timing_bar_y0 = timing_y + 35
        timing_bar_y1 = timing_bar_y0 + 10
        draw.rounded_rectangle(
            (timing_bar_x0, timing_bar_y0, timing_bar_x1, timing_bar_y1),
            radius=4,
            fill=(42, 54, 64, 255),
        )
        first_seconds = request_seconds[0]
        first_ratio = min(1.0, max(0.0, first_seconds / total_seconds))
        split_x = round(
            timing_bar_x0 + first_ratio * (timing_bar_x1 - timing_bar_x0)
        )
        draw.rounded_rectangle(
            (timing_bar_x0, timing_bar_y0, split_x, timing_bar_y1),
            radius=4,
            fill=(77, 184, 236, 255),
        )
        draw.rectangle(
            (split_x, timing_bar_y0, timing_bar_x1, timing_bar_y1),
            fill=(170, 111, 235, 255),
        )
        stage_labels = [f"HAZARD {request_seconds[0]:.2f} s"]
        if len(request_seconds) > 1:
            stage_labels.append(f"ROUTE {request_seconds[1]:.2f} s")
        draw.text(
            (detail_x, timing_bar_y1 + 5),
            f"EARLIER CLIP: {' + '.join(stage_labels)}",
            fill=(225, 235, 242, 255),
            font=timing_detail_font,
        )
        draw.text(
            (detail_x, timing_bar_y1 + 24),
            "CURRENT: SCRIPTED · EVK NOT CONNECTED",
            fill=(255, 205, 104, 255),
            font=timing_detail_font,
        )
    else:
        draw.text(
            (detail_x, timing_y + 36),
            "NO TIMING SUPPLIED · CURRENT: SCRIPTED, EVK NOT CONNECTED",
            fill=(255, 205, 104, 255),
            font=timing_detail_font,
        )

    legend_y = height - 102
    draw.rounded_rectangle(
        (margin, legend_y, width - margin, height - 16),
        radius=10,
        fill=(10, 15, 20, 238),
    )
    bar_x0 = margin + 22
    bar_x1 = width - margin - 22
    bar_y = legend_y + 26
    draw.line((bar_x0, bar_y, bar_x1, bar_y), fill=(66, 82, 94, 255), width=8)
    current_x = round(bar_x0 + progress * (bar_x1 - bar_x0))
    draw.line((bar_x0, bar_y, current_x, bar_y), fill=accent, width=8)
    draw.ellipse(
        (current_x - 9, bar_y - 9, current_x + 9, bar_y + 9),
        fill=accent,
    )
    draw.text(
        (bar_x0, bar_y + 18),
        f"SCENARIO PROGRESS {progress:>5.0%}",
        fill=(226, 235, 240, 255),
        font=label_font,
    )
    legend_x = width // 3
    for actor_name, actor_color in ACTOR_LEGEND:
        draw.ellipse(
            (legend_x, bar_y + 17, legend_x + 18, bar_y + 35),
            fill=actor_color,
        )
        draw.text(
            (legend_x + 27, bar_y + 15),
            actor_name,
            fill=(230, 238, 244, 255),
            font=detail_font,
        )
        legend_x += max(175, width // 7)

    return canvas.convert("RGB")


def _frame_paths(capture_root: Path) -> list[Path]:
    paths = sorted((capture_root / "frames").glob("frame_*.png"))
    if len(paths) < 2:
        raise RuntimeError(
            f"Expected at least two frame_*.png files under "
            f"{capture_root / 'frames'}"
        )
    return paths


def build_gif(
    capture_root: Path,
    output_dir: Path,
    *,
    scenario: str,
    width: int,
    fps: float,
    layout: str,
    evk_timing: dict[str, Any] | None,
) -> Path:
    Image, _, _, _ = _pillow()
    paths = _frame_paths(capture_root)
    height = round(width * 9 / 16)
    frames = []
    for index, path in enumerate(paths):
        progress = index / max(1, len(paths) - 1)
        with Image.open(path) as source:
            if layout == "tactical":
                annotated = _tactical_frame(
                    source,
                    progress,
                    scenario=scenario,
                    output_size=(width, height),
                    evk_timing=evk_timing,
                )
            else:
                annotated = _notification_frame(
                    source,
                    progress,
                    scenario=scenario,
                    camera_label=CAMERA_LABELS["overview"],
                    output_size=(width, height),
                )
            frames.append(
                annotated.quantize(
                    colors=160,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.FLOYDSTEINBERG,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{scenario}.gif"
    duration_ms = max(20, round(1000 / fps))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output


def build_camera_grid(
    capture_root: Path,
    output_dir: Path,
    *,
    scenario: str,
    progress: float,
) -> tuple[Path, Path]:
    Image, ImageDraw, ImageFont, _ = _pillow()
    panel_size = (640, 360)
    panels: list[Any] = []
    for camera_name in CAMERA_LABELS:
        path = capture_root / f"{camera_name}.png"
        if not path.is_file():
            raise RuntimeError(f"Missing camera capture: {path}")
        with Image.open(path) as source:
            panels.append(
                _notification_frame(
                    source,
                    progress,
                    scenario=scenario,
                    camera_label=CAMERA_LABELS[camera_name],
                    output_size=panel_size,
                )
            )

    grid = Image.new("RGB", (1280, 720), (10, 15, 20))
    for panel, position in zip(
        panels,
        ((0, 0), (640, 0), (0, 360), (640, 360)),
        strict=True,
    ):
        grid.paste(panel, position)

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"{scenario}_camera_grid.png"
    grid.save(grid_path, optimize=True)

    # The overview still is useful as a compact tutorial hero image.
    hero = panels[0].resize((1280, 720), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(hero, "RGBA")
    legend_font = _font(ImageFont, 18, bold=True)
    legend = "Blue: supervised AMR | Green: second AMR | Orange: forklift | Yellow: worker"
    draw.rounded_rectangle((18, 658, 765, 705), radius=8, fill=(10, 15, 20, 218))
    draw.text((32, 672), legend, fill=(235, 242, 246, 255), font=legend_font)
    hero_path = output_dir / f"{scenario}_overview.png"
    hero.save(hero_path, optimize=True)
    return grid_path, hero_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "isaac_sim_edge_supervisor",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "media" / "isaac_sim_edge_supervisor",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="mixed_traffic",
    )
    parser.add_argument("--camera-progress", type=float, default=0.60)
    parser.add_argument("--gif-width", type=int, default=768)
    parser.add_argument("--gif-fps", type=float, default=5.0)
    parser.add_argument(
        "--layout",
        choices=("standard", "tactical"),
        default="standard",
    )
    parser.add_argument(
        "--evk-timing-evidence",
        type=Path,
        help=(
            "Measured EVK evidence JSON to display as a clearly labeled "
            "reference latency in tactical GIFs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.camera_progress <= 1.0:
        raise SystemExit("--camera-progress must be between 0 and 1")
    if not 320 <= args.gif_width <= 1600:
        raise SystemExit("--gif-width must be between 320 and 1600")
    if args.layout == "tactical" and args.gif_width < 960:
        raise SystemExit("--layout tactical requires --gif-width of at least 960")
    if not 0.5 <= args.gif_fps <= 20.0:
        raise SystemExit("--gif-fps must be between 0.5 and 20")

    capture_root = (args.capture_root / args.scenario).resolve()
    output_dir = args.output_dir.resolve()
    evk_timing = (
        _load_evk_timing_evidence(args.evk_timing_evidence.resolve())
        if args.evk_timing_evidence
        else None
    )
    gif_path = build_gif(
        capture_root,
        output_dir,
        scenario=args.scenario,
        width=args.gif_width,
        fps=args.gif_fps,
        layout=args.layout,
        evk_timing=evk_timing,
    )
    grid_path, hero_path = build_camera_grid(
        capture_root,
        output_dir,
        scenario=args.scenario,
        progress=args.camera_progress,
    )
    print(
        json.dumps(
            {
                "gif": str(gif_path),
                "camera_grid": str(grid_path),
                "hero": str(hero_path),
                "scenario": args.scenario,
                "layout": args.layout,
                "frame_count": len(_frame_paths(capture_root)),
                "evk_reference_total_seconds": (
                    evk_timing["total_seconds"] if evk_timing else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
