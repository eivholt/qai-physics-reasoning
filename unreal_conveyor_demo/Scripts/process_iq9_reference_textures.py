"""Build the lightweight IQ9 EVK texture set from supplied product views.

The runtime prop remains a five-plane box impostor.  The front/back planes in
the level use Unreal's local U axis vertically, so those two source plates are
stored rotated.  Right/left planes use conventional landscape UVs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


QUALCOMM_BLUE = (42, 42, 234, 255)
TOP_SIZE = (1024, 1024)
SIDE_SIZE = (1024, 512)
ENCLOSURE_TOP = (78, 72, 184)
ENCLOSURE_BOTTOM = (48, 44, 128)
ENCLOSURE_TEXT = (246, 247, 255)
DEBOSSED_DARK = (28, 25, 86)
DEBOSSED_HIGHLIGHT = (100, 94, 198)

# Source quadrilateral just inside the enclosure lip in the real top-down
# photograph.  This excludes the tabletop while retaining the complete PCB.
# Pillow QUAD order is north-west, south-west, south-east, north-east.
TOP_PHOTO_QUAD = (
    80, 278,
    90, 1085,
    880, 1088,
    879, 280,
)


def alpha_bounds(image: Image.Image, padding: int = 0) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        raise RuntimeError("Reference view has no visible pixels")
    left, top, right, bottom = bounds
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )


def flatten(image: Image.Image, background: tuple[int, int, int, int]) -> Image.Image:
    output = Image.new("RGBA", image.size, background)
    output.alpha_composite(image)
    return output.convert("RGB")


def finish(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.resize(size, Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.035)
    return image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=115, threshold=3))


def make_top(source: Image.Image) -> Image.Image:
    board = source.transform(
        TOP_SIZE,
        Image.Transform.QUAD,
        TOP_PHOTO_QUAD,
        resample=Image.Resampling.BICUBIC,
    )
    # Place the photographed USB/Display/logo edge on the actor's +X "Front"
    # face.  The image itself remains an exact photograph; this is only a
    # physical orientation correction.
    board = board.transpose(Image.Transpose.ROTATE_270)
    return finish(board.convert("RGB"), TOP_SIZE)


def centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    draw.text(((SIDE_SIZE[0] - width) / 2, y), text, font=text_font, fill=fill)


def centered_debossed_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    text_font: ImageFont.ImageFont,
) -> None:
    # A faint lower edge plus a darker face reads like shallow molded-in text
    # without pretending the lightweight texture has displaced geometry.
    centered_text(draw, text, y + 2, text_font, DEBOSSED_HIGHLIGHT)
    centered_text(draw, text, y, text_font, DEBOSSED_DARK)


def draw_vent_grid(
    draw: ImageDraw.ImageDraw,
    center_y: int,
    columns: int = 15,
    rows: int = 3,
) -> None:
    radius = 13
    step_x = 43
    step_y = 38
    width = (columns - 1) * step_x
    start_x = (SIDE_SIZE[0] - width) / 2
    start_y = center_y - ((rows - 1) * step_y) / 2
    for row in range(rows):
        row_offset = step_x / 2 if row % 2 else 0
        for column in range(columns - (1 if row % 2 else 0)):
            cx = start_x + column * step_x + row_offset
            cy = start_y + row * step_y
            points = [
                (
                    cx + math.cos(math.radians(60 * index)) * radius,
                    cy + math.sin(math.radians(60 * index)) * radius,
                )
                for index in range(6)
            ]
            # A small lower-right highlight makes the printed holes read as
            # recessed vents at normal presentation distance.
            draw.polygon([(x + 2, y + 3) for x, y in points], fill=(105, 99, 204))
            draw.polygon(points, fill=(13, 14, 34))


def make_enclosure_face(face: str) -> Image.Image:
    panel = Image.new("RGB", SIDE_SIZE, ENCLOSURE_BOTTOM)
    draw = ImageDraw.Draw(panel)
    for y in range(SIDE_SIZE[1]):
        t = y / max(1, SIDE_SIZE[1] - 1)
        # Satin enclosure: broad, quiet vertical tonal rolloff with a subtle
        # highlight across the upper third rather than photographed electronics.
        highlight = 7.0 * math.exp(-((t - 0.27) / 0.22) ** 2)
        colour = tuple(
            int(ENCLOSURE_TOP[channel] * (1.0 - t) + ENCLOSURE_BOTTOM[channel] * t + highlight)
            for channel in range(3)
        )
        draw.line((0, y, SIDE_SIZE[0], y), fill=colour)

    draw.line((0, 4, SIDE_SIZE[0], 4), fill=(127, 121, 228), width=7)
    draw.line((0, SIDE_SIZE[1] - 5, SIDE_SIZE[0], SIDE_SIZE[1] - 5), fill=(30, 28, 86), width=8)
    draw.rectangle((5, 5, SIDE_SIZE[0] - 6, SIDE_SIZE[1] - 6), outline=(57, 53, 145), width=5)

    if face == "front":
        centered_text(draw, "Qualcomm", 90, font(54), ENCLOSURE_TEXT)
        centered_text(draw, "Dragonwing", 155, font(84), ENCLOSURE_TEXT)
        centered_text(draw, "IQ-9075 EVK", 286, font(38), ENCLOSURE_TEXT)
    elif face == "back":
        centered_debossed_text(draw, "SPEAKERS      MODE      LOW SPEED HEADER", 68, font(29))
        draw_vent_grid(draw, center_y=270)
        centered_debossed_text(draw, "IQ-9075 EVK", 405, font(25))
    elif face == "right":
        centered_debossed_text(draw, "PCIe x4        CSI3   CSI2   CSI1   CSI0", 68, font(30))
        draw_vent_grid(draw, center_y=270)
        centered_debossed_text(draw, "IQ-9075 EVK", 405, font(25))
    elif face == "left":
        centered_debossed_text(draw, "12–36VDC      JTAG DEBUG      DS      OFF/ON", 68, font(29))
        draw_vent_grid(draw, center_y=270)
        centered_debossed_text(draw, "IQ-9075 EVK", 405, font(25))
    else:
        raise ValueError(f"Unknown enclosure face: {face}")
    return finish(panel, SIDE_SIZE)


def rotate_for_unreal(image: Image.Image, direction: str) -> Image.Image:
    if direction == "cw":
        return image.transpose(Image.Transpose.ROTATE_270)
    if direction == "ccw":
        return image.transpose(Image.Transpose.ROTATE_90)
    return image


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def checkerboard(size: tuple[int, int], cell: int = 24) -> Image.Image:
    canvas = Image.new("RGB", size, (42, 45, 50))
    draw = ImageDraw.Draw(canvas)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if ((x // cell) + (y // cell)) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(58, 62, 68))
    return canvas


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    preview = image.copy()
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    return preview


def make_contact_sheet(top: Image.Image, world_faces: dict[str, Image.Image]) -> Image.Image:
    canvas = checkerboard((1600, 1180))
    draw = ImageDraw.Draw(canvas)
    title_font = font(34)
    label_font = font(24)
    note_font = font(18)
    draw.text((42, 26), "IQ-9075 EVK — lightweight presentation prop", fill=(238, 241, 246), font=title_font)
    draw.text(
        (43, 70),
        "Top uses the real photo; clean procedural enclosure faces avoid duplicating the exposed electronics.",
        fill=(174, 186, 202),
        font=note_font,
    )

    top_preview = contain(top, (620, 620))
    canvas.paste(top_preview, (52, 132))
    draw.text((52, 770), "TOP · exposed board", fill=(238, 241, 246), font=label_font)

    positions = {
        "Front / logo": (740, 132),
        "Rear / mode header": (740, 388),
        "Right / PCIe + CSI": (740, 644),
        "Left / power": (740, 900),
    }
    for label, position in positions.items():
        image = contain(world_faces[label], (800, 205))
        canvas.paste(image, position)
        draw.text((position[0], position[1] + 210), label.upper(), fill=(238, 241, 246), font=label_font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    source_root = project_root / "ContentSource" / "IQ9EVK" / "Reference"
    output_root = project_root / "ContentSource" / "IQ9EVK" / "Impostor"
    audit_root = project_root / "Saved" / "ImportAudit"
    output_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "top": source_root / "iq9_top_photo_primary.jpg",
        "top_reference": source_root / "iq9_top_photo_reference.jpg",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing IQ9 reference views: " + ", ".join(missing))

    with Image.open(paths["top"]) as source:
        top = make_top(source.convert("RGBA"))
    world_faces: dict[str, Image.Image] = {
        "Front / logo": make_enclosure_face("front"),
        "Rear / mode header": make_enclosure_face("back"),
        "Right / PCIe + CSI": make_enclosure_face("right"),
        "Left / power": make_enclosure_face("left"),
    }

    # Front and back require opposite quarter-turns because their actor planes
    # use +90/-90 pitch.  Right and left are already landscape in world space.
    outputs = {
        "iq9_top.png": top,
        # The +X appearance plane is seen with reversed handedness. Mirror the
        # direct-view logo before its required portrait-source quarter-turn.
        "iq9_front.png": rotate_for_unreal(
            ImageOps.mirror(world_faces["Front / logo"]), "cw"
        ),
        # The -X plane is viewed with its world horizontal axis reversed.  Fix
        # that handedness before applying the plane's quarter-turn UV layout.
        "iq9_back.png": rotate_for_unreal(
            ImageOps.mirror(world_faces["Rear / mode header"]), "ccw"
        ),
        # The PCIe/CSI plane preserves horizontal handedness but inverts Y.
        "iq9_right.png": ImageOps.flip(world_faces["Right / PCIe + CSI"]),
        # The power plane already inverts Y. Mirroring the source compensates
        # its remaining horizontal reversal and produces an upright view.
        "iq9_left.png": ImageOps.mirror(world_faces["Left / power"]),
    }
    for name, image in outputs.items():
        path = output_root / name
        image.save(path, "PNG", optimize=True)
        print(f"IQ9_REFERENCE_TEXTURE {path} {image.width}x{image.height}")

    contact_sheet = make_contact_sheet(top, world_faces)
    audit_path = audit_root / "iq9_reference_texture_contact_sheet.png"
    contact_sheet.save(audit_path, "PNG", optimize=True)
    print(f"IQ9_REFERENCE_CONTACT_SHEET {audit_path}")


if __name__ == "__main__":
    main()
