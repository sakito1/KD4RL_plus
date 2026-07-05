from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
    }
)


C = {
    "ink": "#1F2937",
    "muted": "#667085",
    "line": "#AAB4C3",
    "panel": "#F8FAFC",
    "input": "#2F5597",
    "input_l": "#F3F6FC",
    "ctrl": "#315F9E",
    "ctrl_l": "#F4F7FD",
    "outer": "#138A7E",
    "outer_l": "#F0FAF8",
    "inner": "#D94B55",
    "inner_l": "#FFF5F6",
    "base": "#4E8B3D",
    "base_l": "#F4FAF1",
    "exec": "#B66A1D",
    "exec_l": "#FFF7EC",
    "dark": "#111827",
}


def rounded(ax, x, y, w, h, fc, ec, lw=1.25, radius=0.018, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, s, size=8.5, color=None, weight=None, ha="center", va="center", style=None, z=10):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color or C["ink"],
        weight=weight,
        ha=ha,
        va=va,
        style=style,
        zorder=z,
    )


def arrow(ax, p0, p1, color=None, lw=1.6, rad=0.0, ms=10, ls="-", alpha=1.0, z=8):
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["dark"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        linestyle=ls,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def bars(ax, x, y, w, h, color, n=7, alpha=0.90, z=6):
    vals = [0.46, 0.72, 0.35, 0.60, 0.50, 0.82, 0.58, 0.68]
    gap = w * 0.055
    bw = (w - gap * (n - 1)) / n
    for i in range(n):
        bh = h * vals[i % len(vals)]
        ax.add_patch(
            Rectangle(
                (x + i * (bw + gap), y),
                bw,
                bh,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=z,
            )
        )


def spark(ax, x, y, w, h, color, alpha=0.10):
    pts = [0.42, 0.58, 0.52, 0.70, 0.55, 0.80, 0.70, 0.84]
    xs = [x + i * w / (len(pts) - 1) for i in range(len(pts))]
    ys = [y + p * h for p in pts]
    ax.plot(xs, ys, color=color, lw=1.35, solid_capstyle="round", zorder=7)
    ax.fill_between(xs, [y] * len(xs), ys, color=color, alpha=alpha, zorder=5)


def tiny_matrix(ax, x, y, w, h, color):
    rows, cols = 4, 5
    gap = 0.003
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    for r in range(rows):
        for c in range(cols):
            a = 0.22 + 0.11 * ((r + 2 * c) % 5)
            ax.add_patch(
                Rectangle(
                    (x + c * (cw + gap), y + r * (ch + gap)),
                    cw,
                    ch,
                    facecolor=color,
                    edgecolor="none",
                    alpha=a,
                    zorder=6,
                )
            )


def module(ax, x, y, w, h, title, subtitle, color, fc):
    rounded(ax, x, y, w, h, fc, color, lw=1.45, radius=0.020, z=3)
    label(ax, x + 0.024, y + h - 0.040, title, size=12.0, color=color, weight="bold", ha="left")
    label(ax, x + 0.024, y + h - 0.070, subtitle, size=7.6, color=color, style="italic", ha="left")


def step(ax, x, y, w, h, title, note, color, glyph):
    rounded(ax, x, y, w, h, "white", color, lw=0.75, radius=0.010, z=4)
    label(ax, x + 0.015, y + h * 0.62, title, size=7.4, color=C["ink"], weight="bold", ha="left")
    label(ax, x + 0.015, y + h * 0.33, note, size=6.2, color=C["muted"], ha="left")
    gx, gy = x + w * 0.64, y + h * 0.22
    if glyph == "bars":
        bars(ax, gx, gy, w * 0.27, h * 0.45, color, n=6)
    elif glyph == "line":
        spark(ax, gx, gy, w * 0.28, h * 0.48, color)
    elif glyph == "matrix":
        tiny_matrix(ax, gx, gy, w * 0.25, h * 0.52, color)
    elif glyph == "gate":
        for i, txt in enumerate(["H", "S"]):
            cx = gx + i * w * 0.105
            ax.add_patch(Circle((cx, y + h * 0.48), h * 0.16, facecolor=color, edgecolor="none", alpha=0.90, zorder=7))
            label(ax, cx, y + h * 0.48, txt, size=5.9, color="white", weight="bold")
    elif glyph == "mask":
        n = 7
        gap = w * 0.012
        bw = w * 0.030
        for i in range(n):
            fc = color if i in (0, 1, 4, 6) else "#E5E7EB"
            ax.add_patch(Rectangle((gx + i * (bw + gap), y + h * 0.39), bw, h * 0.16, facecolor=fc, edgecolor="none", zorder=6))


def input_card(ax, x, y, w, h, title, note, color, glyph):
    rounded(ax, x, y, w, h, "white", color, lw=0.85, radius=0.010, z=4)
    label(ax, x + 0.014, y + h * 0.62, title, size=7.0, color=C["ink"], weight="bold", ha="left")
    label(ax, x + 0.014, y + h * 0.32, note, size=5.9, color=C["muted"], ha="left")
    if glyph == "matrix":
        tiny_matrix(ax, x + w * 0.70, y + h * 0.22, w * 0.20, h * 0.50, color)
    elif glyph == "line":
        spark(ax, x + w * 0.64, y + h * 0.23, w * 0.27, h * 0.48, color)
    else:
        bars(ax, x + w * 0.67, y + h * 0.25, w * 0.24, h * 0.42, color, n=6)


def draw_decision_diamond(ax, x, y, w, h):
    pts = [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=C["ctrl"], linewidth=1.0, zorder=5))
    label(ax, x + w / 2, y + h / 2, "Hold\nor\nSwitch", size=6.6, color=C["ctrl"], weight="bold")


def draw():
    fig, ax = plt.subplots(figsize=(12.2, 4.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Shared state and data inputs.
    rounded(ax, 0.060, 0.835, 0.880, 0.118, C["input_l"], C["input"], lw=1.25, radius=0.018)
    label(ax, 0.090, 0.928, "Input State", size=10.2, color=C["input"], weight="bold", ha="left")
    label(ax, 0.185, 0.928, "market data, current portfolio, and candidate relation features", size=7.0, color=C["muted"], ha="left")
    input_card(ax, 0.085, 0.855, 0.240, 0.055, "Market / asset tensor", r"$X_t$ with prices, volume, indicators", C["input"], "matrix")
    input_card(ax, 0.380, 0.855, 0.240, 0.055, "Current holding state", "weights, age, return, drawdown, drift", C["base"], "bars")
    input_card(ax, 0.675, 0.855, 0.240, 0.055, "Candidate and pairwise state", "turnover, overlap, concentration", C["inner"], "line")

    # Main modules.
    module(ax, 0.055, 0.392, 0.278, 0.355, "Controller", "daily revision gate", C["ctrl"], C["ctrl_l"])
    module(ax, 0.363, 0.392, 0.278, 0.355, "Outer Actor", "proposes a candidate base", C["outer"], C["outer_l"])
    module(ax, 0.671, 0.392, 0.278, 0.355, "Inner Actor", "refines positions within the active base", C["inner"], C["inner_l"])

    step(ax, 0.080, 0.588, 0.228, 0.060, "State encoder", "base, candidate, pairwise features", C["ctrl"], "matrix")
    step(ax, 0.080, 0.505, 0.228, 0.060, "Switch head", r"outputs $p_t^{sw}$", C["ctrl"], "gate")
    draw_decision_diamond(ax, 0.137, 0.412, 0.116, 0.080)

    step(ax, 0.388, 0.594, 0.228, 0.052, "Market window encoder", "LSTM-HA for temporal context", C["outer"], "line")
    step(ax, 0.388, 0.525, 0.228, 0.052, "Asset scoring", "CAAN and MLP score head", C["outer"], "bars")
    step(ax, 0.388, 0.456, 0.228, 0.052, "Sparse allocation", r"Top-K support and softmax", C["outer"], "mask")
    step(ax, 0.388, 0.398, 0.228, 0.045, r"Candidate base $w_t^{cand}$", "proposed on switch", C["outer"], "bars")

    step(ax, 0.696, 0.594, 0.228, 0.052, "Short-term encoder", "LSTM-Attn on recent market window", C["inner"], "line")
    step(ax, 0.696, 0.525, 0.228, 0.052, "Support mask", "trade only inside active base", C["inner"], "mask")
    step(ax, 0.696, 0.456, 0.228, 0.052, "Refinement head", r"target weights $w_t^{in}$", C["inner"], "bars")
    rounded(ax, 0.716, 0.408, 0.188, 0.048, "white", C["inner"], lw=0.80, radius=0.010, z=4)
    label(ax, 0.810, 0.432, r"$w_t=(1-\alpha)b_t+\alpha w_t^{in}$", size=7.4, color=C["ink"])

    # Portfolio state layer.
    rounded(ax, 0.285, 0.160, 0.310, 0.100, C["base_l"], C["base"], lw=1.25, radius=0.018)
    label(ax, 0.440, 0.228, r"Active Base Portfolio  $b_t$", size=10.0, color=C["base"], weight="bold")
    label(ax, 0.440, 0.195, "kept by hold, replaced by switch", size=6.8, color=C["muted"])
    bars(ax, 0.333, 0.170, 0.210, 0.025, C["base"], n=9)

    rounded(ax, 0.682, 0.160, 0.250, 0.100, C["exec_l"], C["exec"], lw=1.25, radius=0.018)
    label(ax, 0.807, 0.228, "Executed Portfolio", size=10.0, color=C["exec"], weight="bold")
    label(ax, 0.807, 0.195, "return, cost, risk and next state", size=6.8, color=C["muted"])
    bars(ax, 0.732, 0.170, 0.150, 0.025, C["exec"], n=7)

    rounded(ax, 0.070, 0.160, 0.140, 0.100, "#FFFFFF", C["input"], lw=1.00, radius=0.015)
    label(ax, 0.140, 0.225, "Daily feedback", size=8.0, color=C["input"], weight="bold")
    spark(ax, 0.095, 0.170, 0.090, 0.026, C["base"])

    # Flow arrows.
    arrow(ax, (0.205, 0.835), (0.190, 0.747), C["ctrl"], lw=1.35)
    arrow(ax, (0.500, 0.835), (0.502, 0.747), C["outer"], lw=1.35)
    arrow(ax, (0.795, 0.835), (0.810, 0.747), C["inner"], lw=1.35)

    arrow(ax, (0.388, 0.429), (0.333, 0.545), C["ctrl"], lw=1.25, rad=0.10, ls=(0, (4, 3)))
    label(ax, 0.352, 0.512, "candidate\nsummary", size=5.8, color=C["muted"])
    arrow(ax, (0.195, 0.412), (0.330, 0.260), C["ctrl"], lw=1.45, rad=0.12)
    label(ax, 0.255, 0.314, "decision", size=6.6, color=C["ctrl"], weight="bold")
    arrow(ax, (0.503, 0.392), (0.470, 0.260), C["outer"], lw=1.55)
    label(ax, 0.505, 0.330, r"switch selects $b_t$", size=6.6, color=C["outer"], weight="bold")
    arrow(ax, (0.595, 0.210), (0.696, 0.545), C["inner"], lw=1.35)
    label(ax, 0.640, 0.370, "active support", size=6.3, color=C["muted"])
    arrow(ax, (0.810, 0.392), (0.810, 0.260), C["inner"], lw=1.55)
    arrow(ax, (0.595, 0.210), (0.682, 0.210), C["exec"], lw=1.55)
    arrow(ax, (0.932, 0.210), (0.955, 0.885), C["input"], lw=1.10, rad=0.13, ls=(0, (4, 3)), alpha=0.70)
    arrow(ax, (0.070, 0.210), (0.055, 0.885), C["input"], lw=1.10, rad=-0.10, ls=(0, (4, 3)), alpha=0.70)

    # Compact cadence legend.
    label(ax, 0.065, 0.075, "Cadence:", size=7.6, color=C["ink"], weight="bold", ha="left")
    label(ax, 0.125, 0.075, "controller checks daily", size=7.0, color=C["ctrl"], ha="left")
    label(ax, 0.300, 0.075, r"outer actor proposes $w_t^{cand}$; controller selects $b_t$", size=7.0, color=C["outer"], ha="left")
    label(ax, 0.575, 0.075, "inner actor adjusts weights every trading day", size=7.0, color=C["inner"], ha="left")

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.035)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "editable").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cmtflow_architecture_vector.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "cmtflow_architecture_vector.png", dpi=280, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "editable" / "cmtflow_figure1_modules_editable.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
