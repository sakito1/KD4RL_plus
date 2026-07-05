#!/usr/bin/env python3
"""Draw a detailed controller model diagram for the paper/PPT assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_experiments_outputs" / "paper_experiments_final" / "00_paper_assets" / "editable"
PAPER_FIG_DIR = ROOT / "paper" / "figures" / "editable"
PAPER_MAIN_FIG_DIR = ROOT / "paper" / "figures"


COLORS = {
    "ink": "#172033",
    "muted": "#5D687A",
    "grid": "#D9E1EC",
    "blue": "#315F9E",
    "blue_light": "#F3F7FD",
    "teal": "#138A7E",
    "teal_light": "#EFFAF8",
    "purple": "#6F56A5",
    "purple_light": "#F6F1FB",
    "orange": "#B66A1D",
    "orange_light": "#FFF6E8",
    "red": "#D94B55",
    "red_light": "#FFF4F5",
    "green": "#4E8B3D",
    "green_light": "#F4FAF1",
    "gray_light": "#F8FAFD",
    "white": "#FFFFFF",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
        }
    )


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    *,
    fc: str,
    ec: str,
    lw: float = 1.2,
    radius: float = 0.18,
    z: int = 2,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    lw: float = 1.45,
    rad: float = 0.0,
    z: int = 5,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        zorder=z,
    )
    ax.add_patch(patch)


def text(
    ax: plt.Axes,
    x: float,
    y: float,
    s: str,
    *,
    size: float = 9.0,
    weight: str = "normal",
    color: str = COLORS["ink"],
    ha: str = "center",
    va: str = "center",
    style: str = "normal",
    z: int = 10,
) -> None:
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        style=style,
        zorder=z,
    )


def draw_tensor_icon(ax: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    w = 0.56 * scale
    h = 0.40 * scale
    for k in range(3):
        dx = 0.06 * k * scale
        dy = 0.045 * k * scale
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor="#EAF1FF", edgecolor=COLORS["blue"], linewidth=0.75, zorder=4))
        for i in range(1, 4):
            ax.plot([x + dx + w * i / 4, x + dx + w * i / 4], [y + dy, y + dy + h], color="#9AB2D6", linewidth=0.45, zorder=5)
        for j in range(1, 3):
            ax.plot([x + dx, x + dx + w], [y + dy + h * j / 3, y + dy + h * j / 3], color="#9AB2D6", linewidth=0.45, zorder=5)


def draw_bar_icon(ax: plt.Axes, x: float, y: float, bars: list[float], color: str, scale: float = 1.0) -> None:
    bw = 0.08 * scale
    gap = 0.035 * scale
    max_h = 0.46 * scale
    ax.plot([x - 0.03 * scale, x + len(bars) * (bw + gap)], [y, y], color=COLORS["muted"], linewidth=0.7, zorder=4)
    for i, val in enumerate(bars):
        h = max_h * val
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, h, facecolor=color, edgecolor=color, linewidth=0.4, zorder=5))


def draw_feature_ticks(ax: plt.Axes, x: float, y: float, colors: list[str], scale: float = 1.0) -> None:
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((x + i * 0.105 * scale, y), 0.075 * scale, 0.12 * scale, facecolor=c, edgecolor="#FFFFFF", linewidth=0.45, zorder=5))


def input_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    subtitle: str,
    edge: str,
    fill: str,
    icon: str,
) -> None:
    box(ax, (x, y), (w, h), fc=fill, ec=edge, lw=1.0, radius=0.12)
    text(ax, x + 0.22, y + h - 0.23, title, size=9.2, weight="bold", color=edge, ha="left")
    text(ax, x + 1.05, y + h - 0.50, subtitle, size=7.4, color=COLORS["muted"], ha="left")

    ix, iy = x + 0.25, y + 0.14
    if icon == "tensor":
        draw_tensor_icon(ax, ix, iy, 0.76)
    elif icon == "bars_current":
        draw_bar_icon(ax, ix, iy + 0.02, [0.62, 0.43, 0.78, 0.35, 0.53, 0.25], COLORS["blue"], 1.0)
    elif icon == "bars_cand":
        draw_bar_icon(ax, ix, iy + 0.02, [0.18, 0.75, 0.38, 0.66, 0.28, 0.55], COLORS["green"], 1.0)
    elif icon == "holding":
        draw_feature_ticks(ax, ix, iy + 0.20, ["#DDE7F6", "#9CC5E8", "#5C9FD6", "#F6D083"], 1.25)
        text(ax, ix + 0.45, iy + 0.07, "age  ret  dd  conc.", size=6.2, color=COLORS["muted"], ha="center")
    elif icon == "action":
        draw_feature_ticks(ax, ix, iy + 0.20, ["#EC6D6D", "#F3B45A", "#7AC7B7"], 1.25)
        text(ax, ix + 0.45, iy + 0.07, "turnover  conc.  overlap", size=6.0, color=COLORS["muted"], ha="center")


def process_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str, fill: str) -> None:
    box(ax, (x, y), (w, h), fc=fill, ec=color, lw=1.0, radius=0.12)
    text(ax, x + w / 2, y + h * 0.62, title, size=8.6, weight="bold", color=color)
    text(ax, x + w / 2, y + h * 0.30, subtitle, size=6.8, color=COLORS["muted"])


def head_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, formula: str, color: str, fill: str) -> None:
    box(ax, (x, y), (w, h), fc=fill, ec=color, lw=1.05, radius=0.12)
    text(ax, x + w / 2, y + h * 0.68, title, size=8.5, weight="bold", color=color)
    text(ax, x + w / 2, y + h * 0.30, formula, size=8.3, color=COLORS["ink"])


def build_figure() -> plt.Figure:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(15.8, 8.35))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title and state definition.
    text(ax, 8.0, 8.58, "Controller Model: State-Conditioned Base Revision Policy", size=18, weight="bold")
    box(ax, (2.45, 8.02), (12.30, 0.38), fc=COLORS["gray_light"], ec=COLORS["grid"], lw=0.9, radius=0.10)
    text(
        ax,
        8.60,
        8.21,
        r"$s_t^{ctrl}=\{X_t^{ctrl},\bar{w}_t,\bar{w}_t^{cand},u_t,a_t^{ctrl}\}$",
        size=13.5,
        weight="bold",
    )

    # Input group.
    box(ax, (0.35, 1.06), (3.85, 6.75), fc=COLORS["blue_light"], ec=COLORS["blue"], lw=1.25, radius=0.18)
    text(ax, 2.275, 7.55, "Controller State", size=13, weight="bold", color=COLORS["blue"])
    text(ax, 2.275, 7.27, "current holding vs. candidate replacement", size=7.7, color=COLORS["muted"])

    input_card(
        ax,
        0.62,
        6.30,
        3.32,
        0.82,
        title=r"$X_t^{ctrl}$",
        subtitle="recent market tensor",
        edge=COLORS["blue"],
        fill="#FFFFFF",
        icon="tensor",
    )
    input_card(
        ax,
        0.62,
        5.18,
        3.32,
        0.82,
        title=r"$\bar{w}_t$",
        subtitle="drifted current holding",
        edge=COLORS["blue"],
        fill="#FFFFFF",
        icon="bars_current",
    )
    input_card(
        ax,
        0.62,
        4.06,
        3.32,
        0.82,
        title=r"$\bar{w}_t^{cand}$",
        subtitle="candidate base from Outer Actor",
        edge=COLORS["green"],
        fill="#FFFFFF",
        icon="bars_cand",
    )
    input_card(
        ax,
        0.62,
        2.94,
        3.32,
        0.82,
        title=r"$u_t$",
        subtitle="holding-state vector",
        edge=COLORS["purple"],
        fill="#FFFFFF",
        icon="holding",
    )
    input_card(
        ax,
        0.62,
        1.82,
        3.32,
        0.82,
        title=r"$a_t^{ctrl}$",
        subtitle="action-comparison features",
        edge=COLORS["orange"],
        fill="#FFFFFF",
        icon="action",
    )

    # Encoder group.
    box(ax, (4.75, 1.06), (3.55, 6.75), fc=COLORS["teal_light"], ec=COLORS["teal"], lw=1.25, radius=0.18)
    text(ax, 6.525, 7.55, "State / Action Encoding", size=13, weight="bold", color=COLORS["teal"])
    process_box(ax, 5.05, 6.27, 2.95, 0.72, "Market Encoder", "LSTM-Attn over recent market window", COLORS["teal"], "#FFFFFF")
    process_box(ax, 5.05, 5.15, 2.95, 0.72, "Current Holding Encoder", r"embed $\bar{w}_t$ and drift state", COLORS["teal"], "#FFFFFF")
    process_box(ax, 5.05, 4.03, 2.95, 0.72, "Candidate Encoder", r"embed $\bar{w}_t^{cand}$ from outer actor", COLORS["teal"], "#FFFFFF")
    process_box(ax, 5.05, 2.91, 2.95, 0.72, "Holding-State MLP", "age, segment return, drawdown, concentration", COLORS["teal"], "#FFFFFF")
    process_box(ax, 5.05, 1.79, 2.95, 0.72, "Action-Comparison MLP", "turnover, candidate concentration, support overlap", COLORS["teal"], "#FFFFFF")

    # Fusion and shared trunk.
    box(ax, (8.86, 3.18), (2.32, 2.50), fc=COLORS["purple_light"], ec=COLORS["purple"], lw=1.25, radius=0.18)
    text(ax, 10.02, 5.32, "Fusion Representation", size=12.5, weight="bold", color=COLORS["purple"])
    text(ax, 10.02, 4.82, r"$z_t=[e^m_t,e^h_t,e^c_t,e^u_t,e^a_t]$", size=9.5)
    process_box(ax, 9.18, 3.90, 1.70, 0.62, "Shared trunk", "nonlinear policy features", COLORS["purple"], "#FFFFFF")
    text(ax, 10.02, 3.49, r"base exit logit $e_t^{base}$", size=8.3, color=COLORS["muted"])

    # Auxiliary heads.
    box(ax, (11.85, 4.38), (3.52, 2.84), fc=COLORS["orange_light"], ec=COLORS["orange"], lw=1.25, radius=0.18)
    text(ax, 13.61, 6.95, "Auxiliary Estimates", size=13, weight="bold", color=COLORS["orange"])
    head_box(ax, 12.15, 6.00, 2.92, 0.50, "Holding return", r"$\hat{R}^{hold}_t$", COLORS["orange"], "#FFFFFF")
    head_box(ax, 12.15, 5.35, 2.92, 0.50, "Holding risk", r"$\hat{D}^{hold}_t$", COLORS["orange"], "#FFFFFF")
    head_box(ax, 12.15, 4.70, 2.92, 0.50, "Switch advantage", r"$\hat{A}^{sw}_t$", COLORS["orange"], "#FFFFFF")

    # Exit probability module.
    box(ax, (11.85, 1.38), (3.52, 2.50), fc=COLORS["red_light"], ec=COLORS["red"], lw=1.25, radius=0.18)
    text(ax, 13.61, 3.55, "Exit Probability Head", size=13, weight="bold", color=COLORS["red"])
    text(ax, 13.61, 3.06, r"$\pi_t^{exit}=\sigma(e_t^{base}+\eta\tanh(\hat{A}_t^{sw}/c_A))$", size=10.6, weight="bold")
    box(ax, (12.32, 1.84), (2.58, 0.48), fc="#FFFFFF", ec=COLORS["red"], lw=0.9, radius=0.10)
    text(ax, 13.61, 2.08, r"bounded advantage modulation", size=7.9, color=COLORS["muted"])
    text(ax, 13.61, 1.62, r"threshold gate: hold / switch", size=8.5, weight="bold", color=COLORS["red"])

    # Decision output.
    diamond = mpl.patches.RegularPolygon((16.10, 2.62), numVertices=4, radius=0.48, orientation=0.785398, facecolor="#FFFFFF", edgecolor=COLORS["red"], linewidth=1.15, zorder=3)
    ax.add_patch(diamond)
    text(ax, 16.10, 2.62, "Gate", size=9.0, weight="bold", color=COLORS["red"])
    box(ax, (15.52, 3.55), (1.15, 0.45), fc=COLORS["green_light"], ec=COLORS["green"], lw=1.0, radius=0.10)
    text(ax, 16.095, 3.78, "Hold", size=8.6, weight="bold", color=COLORS["green"])
    box(ax, (15.52, 1.25), (1.15, 0.45), fc=COLORS["red_light"], ec=COLORS["red"], lw=1.0, radius=0.10)
    text(ax, 16.095, 1.48, "Switch", size=8.6, weight="bold", color=COLORS["red"])

    # Interpretation note.
    box(ax, (4.80, 0.28), (11.65, 0.52), fc="#FFFFFF", ec=COLORS["grid"], lw=1.0, radius=0.12)
    text(
        ax,
        10.62,
        0.54,
        r"Interpretation: $\pi_t^{exit}$ is a nonlinear policy signal, not a one-dimensional future-return predictor.",
        size=9.3,
        weight="bold",
        color=COLORS["ink"],
    )

    # Arrows from inputs to encoders.
    ys = [6.71, 5.59, 4.47, 3.35, 2.23]
    for y in ys:
        arrow(ax, (3.95, y), (5.05, y), color=COLORS["blue"], lw=1.25)

    # Arrows from encoders to fusion.
    for y in [6.63, 5.51, 4.39, 3.27, 2.15]:
        arrow(ax, (8.00, y), (8.86, 4.43), color=COLORS["teal"], lw=1.25)

    # Fusion to heads.
    arrow(ax, (11.18, 4.80), (11.85, 5.70), color=COLORS["purple"], lw=1.35)
    arrow(ax, (11.18, 4.00), (11.85, 2.78), color=COLORS["purple"], lw=1.35)
    arrow(ax, (13.61, 4.70), (13.61, 3.88), color=COLORS["orange"], lw=1.2)
    arrow(ax, (15.37, 2.62), (15.62, 3.55), color=COLORS["green"], lw=1.1)
    arrow(ax, (15.37, 2.62), (15.62, 1.70), color=COLORS["red"], lw=1.1)

    # Small formula callouts.
    arrow(ax, (15.37, 2.62), (15.64, 2.62), color=COLORS["red"], lw=1.1)
    text(ax, 15.92, 2.03, r"$\pi_t^{exit}\geq\tau$", size=7.8, color=COLORS["red"])
    text(ax, 15.91, 3.19, r"$\pi_t^{exit}<\tau$", size=7.8, color=COLORS["green"])

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    outputs = [
        OUT_DIR / "controller_detailed_model_design.svg",
        OUT_DIR / "controller_detailed_model_design.pdf",
        OUT_DIR / "controller_detailed_model_design.png",
        PAPER_FIG_DIR / "controller_detailed_model_design.svg",
        PAPER_MAIN_FIG_DIR / "controller_detailed_model_design.pdf",
        PAPER_MAIN_FIG_DIR / "controller_detailed_model_design.png",
    ]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
