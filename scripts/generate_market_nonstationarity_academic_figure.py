#!/usr/bin/env python3
"""Generate a publication-style schematic for market non-stationarity.

The output is a clean English PNG for academic talks and paper motivation
figures. It intentionally avoids decorative effects and uses simple geometric
marks that remain readable after being inserted into slides.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "market_nonstationarity_academic_schematic.png"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"

Color = Tuple[int, int, int]
Point = Tuple[float, float]


BLUE = (44, 127, 184)
ORANGE = (230, 126, 34)
RED = (198, 74, 52)
GREEN = (59, 145, 92)
GRAY = (90, 98, 108)
LIGHT_GRAY = (232, 236, 241)
TEXT = (24, 29, 35)
MUTED = (96, 105, 116)
PANEL_BG = (248, 250, 252)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def line_points(points: Sequence[Point]) -> list[Tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points]


def gaussian_points(
    left: int,
    right: int,
    baseline: int,
    height: int,
    *,
    center: float,
    sigma: float,
    n: int = 240,
) -> list[Point]:
    pts = []
    width = right - left
    for i in range(n):
        t = i / (n - 1)
        x = left + width * t
        y = baseline - height * math.exp(-0.5 * ((t - center) / sigma) ** 2)
        pts.append((x, y))
    return pts


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: Color = TEXT,
    *,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, fnt)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: Color,
    max_width: int,
    line_gap: int = 10,
) -> int:
    x, y = xy
    h = text_size(draw, "Ag", fnt)[1] + line_gap
    for line in wrap_text(draw, text, fnt, max_width):
        draw_text(draw, (x, y), line, fnt, fill)
        y += h
    return y


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    *,
    width: int = 4,
    head: int = 18,
) -> None:
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    p1 = (end[0] + head * math.cos(a1), end[1] + head * math.sin(a1))
    p2 = (end[0] + head * math.cos(a2), end[1] + head * math.sin(a2))
    draw.polygon([end, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))], fill=color)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    *,
    width: int = 3,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        seg_end = min(length, pos + dash)
        p0 = (int(x0 + ux * pos), int(y0 + uy * pos))
        p1 = (int(x0 + ux * seg_end), int(y0 + uy * seg_end))
        draw.line([p0, p1], fill=color, width=width)
        pos += dash + gap


def draw_panel_header(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    letter: str,
    title: str,
    subtitle: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x0, y0, x1, _ = box
    badge = (x0 + 36, y0 + 34, x0 + 96, y0 + 94)
    draw.ellipse(badge, fill=(237, 243, 250), outline=(194, 205, 219), width=2)
    draw_text(draw, ((badge[0] + badge[2]) // 2, (badge[1] + badge[3]) // 2 + 1), letter, fonts["panel_letter"], BLUE, anchor="mm")
    draw_text(draw, (x0 + 122, y0 + 36), title, fonts["panel_title"], TEXT)
    draw_wrapped(draw, (x0 + 122, y0 + 90), subtitle, fonts["body"], MUTED, x1 - x0 - 160, line_gap=8)


def draw_panel_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL_BG, outline=(205, 213, 224), width=2)


def draw_panel_a(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    draw_panel_header(
        draw,
        box,
        "A",
        "Return distribution shift",
        "The reward / return distribution changes across market regimes.",
        fonts,
    )

    left, right = x0 + 96, x1 - 72
    top, baseline = y0 + 245, y1 - 142
    draw.line([(left, baseline), (right, baseline)], fill=(120, 130, 145), width=3)
    draw.line([(left, baseline), (left, top)], fill=(120, 130, 145), width=3)
    for i in range(1, 4):
        y = baseline - i * ((baseline - top) // 4)
        draw.line([(left, y), (right, y)], fill=(224, 229, 236), width=2)

    old = gaussian_points(left, right, baseline, baseline - top - 18, center=0.37, sigma=0.12)
    new = gaussian_points(left, right, baseline, baseline - top - 58, center=0.62, sigma=0.18)

    old_poly = line_points(old + [(right, baseline), (left, baseline)])
    new_poly = line_points(new + [(right, baseline), (left, baseline)])
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.polygon(old_poly, fill=(*BLUE, 42))
    odraw.polygon(new_poly, fill=(*ORANGE, 42))
    img.alpha_composite(overlay)
    draw = ImageDraw.Draw(img)
    draw.line(line_points(old), fill=BLUE, width=6, joint="curve")
    draw.line(line_points(new), fill=ORANGE, width=6, joint="curve")
    draw_arrow(draw, (left + 330, baseline - 168), (left + 514, baseline - 168), ORANGE, width=4, head=18)

    draw_text(draw, (left, baseline + 42), "return r", fonts["axis"], MUTED)
    draw_text(draw, (left - 42, top - 12), "density", fonts["axis"], MUTED)
    draw_text(draw, (left + 102, baseline - 320), "Regime t", fonts["legend"], BLUE)
    draw_text(draw, (left + 480, baseline - 258), "Regime t+1", fonts["legend"], ORANGE)
    draw_text(draw, ((left + right) // 2, y1 - 58), "return law changes across regimes", fonts["mono"], TEXT, anchor="mm")


def corr_to_color(v: float) -> Color:
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        return (
            int(248 - 58 * v),
            int(248 - 126 * v),
            int(248 - 166 * v),
        )
    v = -v
    return (
        int(248 - 173 * v),
        int(248 - 118 * v),
        int(248 - 42 * v),
    )


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    matrix: Sequence[Sequence[float]],
    x: int,
    y: int,
    cell: int,
    label: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    n = len(matrix)
    draw_text(draw, (x + n * cell // 2, y - 42), label, fonts["legend"], TEXT, anchor="mm")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            x0, y0 = x + j * cell, y + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=corr_to_color(val), outline=(255, 255, 255), width=2)
    draw.rectangle((x, y, x + n * cell, y + n * cell), outline=(120, 130, 145), width=2)
    assets = ["A", "B", "C", "D", "E"]
    for i, a in enumerate(assets):
        draw_text(draw, (x - 18, y + i * cell + cell // 2), a, fonts["small"], MUTED, anchor="rm")
        draw_text(draw, (x + i * cell + cell // 2, y + n * cell + 28), a, fonts["small"], MUTED, anchor="mm")


def draw_panel_b(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    draw_panel_header(
        draw,
        box,
        "B",
        "Time-varying asset dependence",
        "Cross-asset correlations and dependence structure evolve over time.",
        fonts,
    )
    m0 = [
        [1.00, 0.62, 0.22, -0.08, 0.15],
        [0.62, 1.00, 0.35, -0.18, 0.04],
        [0.22, 0.35, 1.00, 0.28, -0.10],
        [-0.08, -0.18, 0.28, 1.00, 0.55],
        [0.15, 0.04, -0.10, 0.55, 1.00],
    ]
    m1 = [
        [1.00, 0.12, -0.32, 0.54, 0.41],
        [0.12, 1.00, 0.58, 0.08, -0.24],
        [-0.32, 0.58, 1.00, 0.16, -0.38],
        [0.54, 0.08, 0.16, 1.00, 0.27],
        [0.41, -0.24, -0.38, 0.27, 1.00],
    ]
    cell = 72
    y = y0 + 290
    x_left = x0 + 86
    x_right = x0 + 678
    draw_heatmap(draw, m0, x_left, y, cell, "Corr(t)", fonts)
    draw_heatmap(draw, m1, x_right, y, cell, "Corr(t+1)", fonts)
    draw_arrow(draw, (x_left + cell * 5 + 44, y + cell * 2 + cell // 2), (x_right - 44, y + cell * 2 + cell // 2), GRAY, width=4, head=17)

    bar_x, bar_y = x0 + 278, y1 - 186
    draw_text(draw, (bar_x, bar_y - 48), "correlation", fonts["small"], MUTED)
    for k in range(90):
        t = k / 89
        col = corr_to_color(-1 + 2 * t)
        draw.rectangle((bar_x + k * 4, bar_y, bar_x + k * 4 + 4, bar_y + 22), fill=col)
    draw.rectangle((bar_x, bar_y, bar_x + 360, bar_y + 22), outline=(120, 130, 145), width=1)
    draw_text(draw, (bar_x, bar_y + 52), "-1", fonts["small"], MUTED)
    draw_text(draw, (bar_x + 177, bar_y + 52), "0", fonts["small"], MUTED, anchor="mm")
    draw_text(draw, (bar_x + 360, bar_y + 52), "+1", fonts["small"], MUTED, anchor="ra")
    draw_text(draw, ((x0 + x1) // 2, y1 - 58), "asset dependence changes over time", fonts["mono"], TEXT, anchor="mm")


def draw_panel_c(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    draw_panel_header(
        draw,
        box,
        "C",
        "Static portfolio validity decays",
        "A fixed allocation can become suboptimal as regimes and dependencies drift.",
        fonts,
    )
    left, right = x0 + 104, x1 - 84
    top, bottom = y0 + 265, y1 - 150
    draw.line([(left, bottom), (right, bottom)], fill=(120, 130, 145), width=3)
    draw.line([(left, bottom), (left, top)], fill=(120, 130, 145), width=3)
    for i in range(1, 5):
        y = bottom - i * ((bottom - top) // 5)
        draw.line([(left, y), (right, y)], fill=(224, 229, 236), width=2)
    for frac, lab in [(0.32, "shift"), (0.68, "shift")]:
        x = int(left + frac * (right - left))
        draw_dashed_line(draw, (x, top + 6), (x, bottom), (156, 163, 175), width=3, dash=18, gap=13)
        draw_text(draw, (x, top - 16), lab, fonts["small"], MUTED, anchor="mm")

    old: list[Point] = []
    adapt: list[Point] = []
    mismatch: list[Point] = []
    for i in range(160):
        t = i / 159
        x = left + t * (right - left)
        static = 0.80 - 0.34 * t - 0.08 * max(0, t - 0.34) - 0.12 * max(0, t - 0.68)
        static += 0.018 * math.sin(7 * math.pi * t)
        dynamic = 0.62 + 0.10 * t + 0.018 * math.sin(6 * math.pi * t + 0.5)
        mis = 0.18 + 0.52 * t
        old.append((x, bottom - static * (bottom - top)))
        adapt.append((x, bottom - dynamic * (bottom - top)))
        mismatch.append((x, bottom - mis * (bottom - top)))

    draw.line(line_points(old), fill=RED, width=6, joint="curve")
    draw.line(line_points(adapt), fill=GREEN, width=5, joint="curve")
    draw_dashed_line(draw, line_points(mismatch)[0], line_points(mismatch)[-1], ORANGE, width=5, dash=22, gap=12)
    draw_text(draw, (left + 32, bottom + 42), "time", fonts["axis"], MUTED)
    draw_text(draw, (left - 52, top - 8), "validity", fonts["axis"], MUTED)
    draw_text(draw, (right - 304, top + 92), "model mismatch", fonts["legend"], ORANGE)
    draw_text(draw, (right - 318, top + 178), "adaptive policy", fonts["legend"], GREEN)
    draw_text(draw, (right - 326, bottom - 95), "static portfolio", fonts["legend"], RED)
    draw_text(draw, ((x0 + x1) // 2, y1 - 58), "static portfolio validity decreases under drift", fonts["mono"], TEXT, anchor="mm")


def build(width: int, height: int) -> Image.Image:
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    fonts = {
        "title": font(FONT_BOLD, 66),
        "subtitle": font(FONT_REGULAR, 35),
        "panel_letter": font(FONT_BOLD, 34),
        "panel_title": font(FONT_BOLD, 37),
        "body": font(FONT_REGULAR, 27),
        "axis": font(FONT_REGULAR, 25),
        "legend": font(FONT_BOLD, 25),
        "small": font(FONT_REGULAR, 22),
        "mono": font(FONT_REGULAR, 25),
        "footer": font(FONT_REGULAR, 30),
    }

    draw_text(draw, (128, 88), "Non-stationary Markets Degrade Static Portfolio Decisions", fonts["title"])
    draw_text(
        draw,
        (128, 172),
        "Motivation: return distributions and asset dependence structures evolve, so a portfolio optimized on past states may lose out-of-sample validity.",
        fonts["subtitle"],
        MUTED,
    )
    draw.line([(128, 236), (width - 128, 236)], fill=(210, 216, 225), width=3)

    panels = [
        (96, 300, 1228, 1550),
        (1354, 300, 2486, 1550),
        (2612, 300, 3744, 1550),
    ]
    for panel in panels:
        draw_panel_box(draw, panel)
    draw_panel_a(img, panels[0], fonts)
    draw_panel_b(img, panels[1], fonts)
    draw_panel_c(img, panels[2], fonts)

    footer_y = 1660
    draw.rounded_rectangle((448, footer_y, width - 448, footer_y + 145), radius=14, fill=(246, 248, 251), outline=(210, 216, 225), width=2)
    draw_text(
        draw,
        (width // 2, footer_y + 56),
        "Key implication",
        font(FONT_BOLD, 31),
        TEXT,
        anchor="mm",
    )
    draw_text(
        draw,
        (width // 2, footer_y + 104),
        "Portfolio policies should adapt to both regime-level distribution shifts and time-varying cross-asset dependence.",
        fonts["footer"],
        MUTED,
        anchor="mm",
    )
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = build(args.width, args.height)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
