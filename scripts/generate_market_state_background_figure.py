#!/usr/bin/env python3
"""Generate a PPT-ready background figure about market non-stationarity.

The figure is intentionally synthetic and explanatory: it shows return
distribution shift, evolving asset relationships, and the decay of a static
portfolio edge over time.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "market_state_asset_relation_evolution.png"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"

Color = Tuple[int, int, int]
Point = Tuple[float, float]


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def rgba(color: Color, alpha: int) -> Tuple[int, int, int, int]:
    return color[0], color[1], color[2], alpha


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix(c1: Color, c2: Color, t: float) -> Color:
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
    )


def add_glow(
    base: Image.Image,
    draw_fn,
    *,
    blur: int,
    alpha: int,
) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw_fn(draw, alpha)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer)
    sharp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(sharp)
    draw_fn(draw, 255)
    base.alpha_composite(sharp)


def draw_background(width: int, height: int) -> Image.Image:
    small_w, small_h = max(width // 4, 1), max(height // 4, 1)
    img = Image.new("RGB", (small_w, small_h))
    pixels = img.load()
    base_a = (5, 8, 17)
    base_b = (13, 18, 33)
    glows = [
        (0.10, 0.12, 0.48, 0.42, (34, 211, 238), 0.85),
        (0.90, 0.14, 0.42, 0.36, (251, 113, 133), 0.70),
        (0.53, 0.90, 0.60, 0.26, (245, 158, 11), 0.35),
        (0.52, 0.42, 0.48, 0.36, (167, 139, 250), 0.55),
    ]
    for y in range(small_h):
        for x in range(small_w):
            nx = x / max(small_w - 1, 1)
            ny = y / max(small_h - 1, 1)
            c = list(mix(base_a, base_b, 0.38 * nx + 0.62 * ny))
            for cx, cy, rx, ry, gc, strength in glows:
                dx = (nx - cx) / rx
                dy = (ny - cy) / ry
                d = math.sqrt(dx * dx + dy * dy)
                if d < 1.0:
                    s = (1.0 - d) ** 2 * strength
                    c[0] = min(255, int(c[0] + gc[0] * s * 0.55))
                    c[1] = min(255, int(c[1] + gc[1] * s * 0.55))
                    c[2] = min(255, int(c[2] + gc[2] * s * 0.55))
            pixels[x, y] = tuple(c)
    img = img.resize((width, height), Image.Resampling.BICUBIC).convert("RGBA")

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid_color = (255, 255, 255, 18)
    for x in range(0, width, 120):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 120):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    random.seed(19)
    for _ in range(260):
        x = random.randint(40, width - 40)
        y = random.randint(40, height - 40)
        a = random.randint(22, 74)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255, a))
    img.alpha_composite(overlay)
    return img


def draw_rounded_panel(
    base: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
    radius: int = 34,
) -> None:
    x0, y0, x1, y1 = box
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle((x0 + 22, y0 + 30, x1 + 22, y1 + 30), radius=radius, fill=(0, 0, 0, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(34))
    base.alpha_composite(shadow)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)
    draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=radius - 2, outline=(255, 255, 255, 20), width=1)
    base.alpha_composite(layer)


def text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont):
    return draw.textbbox(xy, text, font=font)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int, int],
    *,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_bbox(draw, (0, 0), candidate, font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int, int],
    max_width: int,
    line_gap: int = 10,
) -> int:
    x, y = xy
    line_height = text_bbox(draw, (0, 0), "Ag", font)[3] + line_gap
    for line in wrap_text(draw, text, font, max_width):
        draw_text(draw, (x, y), line, font, fill)
        y += line_height
    return y


def draw_pill(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
    text_fill: Tuple[int, int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=outline, width=2)
    draw_text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text, font, text_fill, anchor="mm")


def path_points(points: Sequence[Point]) -> list[Tuple[int, int]]:
    return [(int(x), int(y)) for x, y in points]


def draw_glow_line(
    base: Image.Image,
    points: Sequence[Point],
    color: Color,
    *,
    width: int,
    glow_width: int = 18,
    alpha: int = 255,
    joint: str = "curve",
) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.line(path_points(points), fill=rgba(color, 70), width=glow_width, joint=joint)
    layer = layer.filter(ImageFilter.GaussianBlur(glow_width // 2))
    base.alpha_composite(layer)
    sharp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(sharp)
    draw.line(path_points(points), fill=rgba(color, alpha), width=width, joint=joint)
    base.alpha_composite(sharp)


def gaussian_curve(
    left: int,
    baseline: int,
    width: int,
    height: int,
    *,
    center: float,
    sigma: float,
    points: int = 220,
) -> list[Point]:
    result = []
    for i in range(points):
        t = i / (points - 1)
        x = left + t * width
        yv = math.exp(-0.5 * ((t - center) / sigma) ** 2)
        y = baseline - yv * height
        result.append((x, y))
    return result


def draw_distribution_panel(
    base: Image.Image,
    box: Tuple[int, int, int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(base)
    cyan = (34, 211, 238)
    pink = (251, 113, 133)
    amber = (245, 158, 11)
    draw_text(draw, (x0 + 62, y0 + 70), "1", fonts["kicker"], rgba(cyan, 255))
    draw_text(draw, (x0 + 125, y0 + 70), "Return Distribution Shift", fonts["panel_title"], (248, 250, 252, 255))
    draw_wrapped_text(
        draw,
        (x0 + 125, y0 + 138),
        "Expected returns, volatility, and tail risk move as the market regime changes.",
        fonts["body"],
        (174, 187, 205, 255),
        x1 - x0 - 185,
    )

    chart_left, chart_right = x0 + 95, x1 - 85
    baseline, top = y0 + 760, y0 + 310
    draw.line([(chart_left, baseline), (chart_right, baseline)], fill=(148, 163, 184, 110), width=3)
    draw.line([(chart_left, baseline), (chart_left, top)], fill=(148, 163, 184, 70), width=2)
    for i in range(5):
        y = baseline - i * 90
        draw.line([(chart_left, y), (chart_right, y)], fill=(255, 255, 255, 22), width=1)

    old = gaussian_curve(chart_left, baseline, chart_right - chart_left, 330, center=0.36, sigma=0.12)
    new = gaussian_curve(chart_left, baseline, chart_right - chart_left, 285, center=0.62, sigma=0.20)

    fill_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(fill_layer)
    old_poly = path_points(old + [(chart_right, baseline), (chart_left, baseline)])
    new_poly = path_points(new + [(chart_right, baseline), (chart_left, baseline)])
    fdraw.polygon(old_poly, fill=rgba(cyan, 52))
    fdraw.polygon(new_poly, fill=rgba(pink, 55))
    base.alpha_composite(fill_layer)
    draw_glow_line(base, old, cyan, width=8, glow_width=24)
    draw_glow_line(base, new, pink, width=8, glow_width=24)
    draw = ImageDraw.Draw(base)
    draw.line([(x0 + 460, y0 + 555), (x0 + 630, y0 + 555)], fill=rgba(amber, 220), width=5)
    draw.polygon([(x0 + 630, y0 + 555), (x0 + 604, y0 + 539), (x0 + 604, y0 + 571)], fill=rgba(amber, 220))

    draw_pill(
        draw,
        (x0 + 130, y0 + 830, x0 + 350, y0 + 900),
        "Regime t0",
        fonts["small_bold"],
        fill=rgba(cyan, 30),
        outline=rgba(cyan, 150),
        text_fill=(217, 249, 255, 255),
    )
    draw_pill(
        draw,
        (x0 + 505, y0 + 830, x0 + 725, y0 + 900),
        "Regime t1",
        fonts["small_bold"],
        fill=rgba(pink, 30),
        outline=rgba(pink, 150),
        text_fill=(255, 228, 232, 255),
    )
    draw_text(draw, (chart_left, baseline + 58), "low return", fonts["tiny"], (148, 163, 184, 210))
    draw_text(draw, (chart_right - 175, baseline + 58), "high return", fonts["tiny"], (148, 163, 184, 210))
    draw_text(draw, (x0 + 140, y1 - 112), "Shifted center", fonts["caption_bold"], rgba(amber, 255))
    draw_text(draw, (x0 + 140, y1 - 67), "Wider tails and changing risk premia", fonts["caption"], (226, 232, 240, 235))


def draw_network_panel(
    base: Image.Image,
    box: Tuple[int, int, int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(base)
    cyan = (34, 211, 238)
    pink = (251, 113, 133)
    violet = (167, 139, 250)
    amber = (245, 158, 11)
    green = (52, 211, 153)
    draw_text(draw, (x0 + 62, y0 + 70), "2", fonts["kicker"], rgba(violet, 255))
    draw_text(draw, (x0 + 125, y0 + 70), "Asset Relationships Rewire", fonts["panel_title"], (248, 250, 252, 255))
    draw_wrapped_text(
        draw,
        (x0 + 125, y0 + 138),
        "Correlations and lead-lag links evolve, changing which assets diversify or amplify risk.",
        fonts["body"],
        (174, 187, 205, 255),
        x1 - x0 - 185,
    )

    cx, cy = (x0 + x1) // 2, y0 + 575
    radius = 300
    nodes: list[Tuple[int, int, str, Color]] = []
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    colors = [cyan, cyan, green, violet, pink, amber, pink]
    for i, label in enumerate(labels):
        angle = -math.pi / 2 + i * 2 * math.pi / len(labels)
        r = radius * (0.86 + 0.16 * math.sin(i * 1.7))
        nodes.append((int(cx + math.cos(angle) * r), int(cy + math.sin(angle) * r), label, colors[i]))

    old_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)]
    new_edges = [(0, 4), (4, 2), (2, 6), (6, 1), (1, 5), (5, 3), (3, 4)]
    for a, b in old_edges:
        p1, p2 = nodes[a], nodes[b]
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(148, 163, 184, 75), width=5)
    for idx, (a, b) in enumerate(new_edges):
        p1, p2 = nodes[a], nodes[b]
        color = pink if idx % 2 else cyan
        draw_glow_line(base, [(p1[0], p1[1]), (p2[0], p2[1])], color, width=7, glow_width=22, alpha=215)
    draw = ImageDraw.Draw(base)

    for x, y, label, color in nodes:
        glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse((x - 39, y - 39, x + 39, y + 39), fill=rgba(color, 95))
        glow = glow.filter(ImageFilter.GaussianBlur(18))
        base.alpha_composite(glow)
        draw = ImageDraw.Draw(base)
        draw.ellipse((x - 33, y - 33, x + 33, y + 33), fill=(12, 18, 32, 255), outline=rgba(color, 235), width=5)
        draw_text(draw, (x, y + 1), label, fonts["node"], (248, 250, 252, 255), anchor="mm")

    draw_pill(
        draw,
        (x0 + 95, y0 + 830, x0 + 383, y0 + 900),
        "old links fade",
        fonts["small_bold"],
        fill=(148, 163, 184, 24),
        outline=(148, 163, 184, 120),
        text_fill=(203, 213, 225, 255),
    )
    draw_pill(
        draw,
        (x0 + 435, y0 + 830, x0 + 760, y0 + 900),
        "new links dominate",
        fonts["small_bold"],
        fill=rgba(pink, 28),
        outline=rgba(pink, 145),
        text_fill=(255, 228, 232, 255),
    )
    draw_text(draw, (x0 + 140, y1 - 112), "Diversification map changes", fonts["caption_bold"], rgba(amber, 255))
    draw_text(draw, (x0 + 140, y1 - 67), "Static assumptions become stale", fonts["caption"], (226, 232, 240, 235))


def draw_decay_panel(
    base: Image.Image,
    box: Tuple[int, int, int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(base)
    cyan = (34, 211, 238)
    pink = (251, 113, 133)
    amber = (245, 158, 11)
    green = (52, 211, 153)
    draw_text(draw, (x0 + 62, y0 + 70), "3", fonts["kicker"], rgba(pink, 255))
    draw_text(draw, (x0 + 125, y0 + 70), "Static Portfolio Edge Decays", fonts["panel_title"], (248, 250, 252, 255))
    draw_wrapped_text(
        draw,
        (x0 + 125, y0 + 138),
        "A portfolio optimized for yesterday's state can become less efficient as conditions drift.",
        fonts["body"],
        (174, 187, 205, 255),
        x1 - x0 - 185,
    )

    left, right = x0 + 95, x1 - 80
    top, bottom = y0 + 320, y0 + 780
    draw.line([(left, bottom), (right, bottom)], fill=(148, 163, 184, 110), width=3)
    draw.line([(left, bottom), (left, top)], fill=(148, 163, 184, 70), width=2)
    for i in range(5):
        y = bottom - i * 92
        draw.line([(left, y), (right, y)], fill=(255, 255, 255, 22), width=1)

    static_points: list[Point] = []
    adaptive_points: list[Point] = []
    for i in range(120):
        t = i / 119
        x = left + t * (right - left)
        static_val = 0.74 - 0.42 * t + 0.045 * math.sin(7 * math.pi * t)
        adaptive_val = 0.57 + 0.22 * t + 0.035 * math.sin(5 * math.pi * t + 0.8)
        static_points.append((x, bottom - static_val * (bottom - top)))
        adaptive_points.append((x, bottom - adaptive_val * (bottom - top)))

    gap_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(gap_layer)
    gap_poly = path_points(adaptive_points + list(reversed(static_points)))
    gdraw.polygon(gap_poly, fill=rgba(amber, 36))
    base.alpha_composite(gap_layer)
    draw_glow_line(base, adaptive_points, green, width=8, glow_width=24)
    draw_glow_line(base, static_points, pink, width=8, glow_width=24)
    draw = ImageDraw.Draw(base)

    draw_pill(
        draw,
        (x0 + 118, y0 + 904, x0 + 400, y0 + 974),
        "adaptive policy",
        fonts["small_bold"],
        fill=rgba(green, 28),
        outline=rgba(green, 145),
        text_fill=(220, 252, 231, 255),
    )
    draw_pill(
        draw,
        (x0 + 445, y0 + 904, x0 + 733, y0 + 974),
        "old portfolio",
        fonts["small_bold"],
        fill=rgba(pink, 28),
        outline=rgba(pink, 145),
        text_fill=(255, 228, 232, 255),
    )
    draw.line([(right - 135, static_points[-1][1]), (right - 135, adaptive_points[-1][1])], fill=rgba(amber, 210), width=5)
    draw.polygon(
        [
            (right - 135, adaptive_points[-1][1] - 2),
            (right - 151, adaptive_points[-1][1] + 26),
            (right - 119, adaptive_points[-1][1] + 26),
        ],
        fill=rgba(amber, 210),
    )
    draw.polygon(
        [
            (right - 135, static_points[-1][1] + 2),
            (right - 151, static_points[-1][1] - 26),
            (right - 119, static_points[-1][1] - 26),
        ],
        fill=rgba(amber, 210),
    )
    draw_text(draw, (right - 365, top + 58), "efficacy gap", fonts["caption_bold"], rgba(amber, 255))
    draw_text(draw, (left, bottom + 58), "time", fonts["tiny"], (148, 163, 184, 210))
    draw_text(draw, (x0 + 140, y1 - 112), "Once efficient, later fragile", fonts["caption_bold"], rgba(amber, 255))
    draw_text(draw, (x0 + 140, y1 - 67), "Portfolio validity must be monitored over time", fonts["caption"], (226, 232, 240, 235))


def draw_connector_flow(base: Image.Image) -> None:
    cyan = (34, 211, 238)
    pink = (251, 113, 133)
    amber = (245, 158, 11)
    y = 1145
    arrows = [
        ((1160, y), (1360, y), cyan),
        ((2400, y), (2600, y), pink),
    ]
    for start, end, color in arrows:
        x0, y0 = start
        x1, y1 = end
        draw_glow_line(base, [(x0, y0), (x1, y1)], color, width=8, glow_width=28, alpha=230)
        draw = ImageDraw.Draw(base)
        draw.polygon([(x1, y1), (x1 - 34, y1 - 22), (x1 - 34, y1 + 22)], fill=rgba(color, 235))
    draw = ImageDraw.Draw(base)
    draw.line([(310, 1768), (3530, 1768)], fill=rgba(amber, 150), width=3)
    for x, label in [(310, "t0"), (1390, "t1"), (2470, "t2"), (3530, "future")]:
        draw.ellipse((x - 10, 1758, x + 10, 1778), fill=rgba(amber, 235))
        draw_text(draw, (x, 1810), label, load_font(FONT_MONO, 34), (252, 211, 77, 235), anchor="mm")


def build_figure(width: int, height: int) -> Image.Image:
    img = draw_background(width, height)
    draw = ImageDraw.Draw(img)
    fonts = {
        "title": load_font(FONT_BOLD, 92),
        "subtitle": load_font(FONT_REGULAR, 42),
        "kicker": load_font(FONT_BOLD, 82),
        "panel_title": load_font(FONT_BOLD, 48),
        "body": load_font(FONT_REGULAR, 33),
        "small_bold": load_font(FONT_BOLD, 31),
        "caption_bold": load_font(FONT_BOLD, 36),
        "caption": load_font(FONT_REGULAR, 31),
        "tiny": load_font(FONT_REGULAR, 28),
        "node": load_font(FONT_BOLD, 30),
    }
    draw_text(draw, (180, 152), "Market States Evolve Over Time", fonts["title"], (248, 250, 252, 255))
    draw_text(
        draw,
        (184, 268),
        "Return distributions shift, asset relationships rewire, and yesterday's efficient portfolio can lose its edge.",
        fonts["subtitle"],
        (203, 213, 225, 245),
    )
    draw_pill(
        draw,
        (3040, 132, 3650, 218),
        "NON-STATIONARY MARKET",
        load_font(FONT_BOLD, 31),
        fill=(15, 23, 42, 128),
        outline=(255, 255, 255, 58),
        text_fill=(226, 232, 240, 245),
    )

    panels = [
        (150, 430, 1120, 1605),
        (1435, 430, 2405, 1605),
        (2720, 430, 3690, 1605),
    ]
    outlines = [(34, 211, 238, 110), (167, 139, 250, 110), (251, 113, 133, 110)]
    for panel, outline in zip(panels, outlines):
        draw_rounded_panel(img, panel, fill=(11, 18, 32, 198), outline=outline, radius=42)

    draw_distribution_panel(img, panels[0], fonts)
    draw_network_panel(img, panels[1], fonts)
    draw_decay_panel(img, panels[2], fonts)
    draw_connector_flow(img)

    draw = ImageDraw.Draw(img)
    footer = "Implication: portfolio policies must adapt to evolving regimes and cross-asset dependencies."
    footer_font = load_font(FONT_BOLD, 43)
    footer_box = (485, 1904, 3355, 2028)
    draw.rounded_rectangle(footer_box, radius=38, fill=(2, 6, 23, 142), outline=(245, 158, 11, 105), width=2)
    draw_text(
        draw,
        ((footer_box[0] + footer_box[2]) // 2, (footer_box[1] + footer_box[3]) // 2),
        footer,
        footer_font,
        (255, 247, 237, 248),
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
    image = build_figure(args.width, args.height)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
