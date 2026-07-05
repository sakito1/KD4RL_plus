from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.linewidth": 0.8,
    }
)


C = {
    "ink": "#172033",
    "muted": "#5f6b7a",
    "line": "#8a96a8",
    "input": "#1f3f99",
    "input_l": "#f8fbff",
    "ctrl": "#1f4fb2",
    "ctrl_l": "#f7fbff",
    "outer": "#08786c",
    "outer_l": "#effbf8",
    "inner": "#e3262e",
    "inner_l": "#fff6f6",
    "base": "#3b7d2a",
    "base_l": "#f4fbf2",
    "exec": "#c16313",
    "exec_l": "#fffaf3",
    "gold": "#c7821d",
    "gold_l": "#fff7df",
    "purple": "#6b3bbf",
}


def rounded(ax, x, y, w, h, fc, ec, lw=1.1, r=0.012, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def txt(ax, x, y, s, size=7, color=None, weight=None, ha="left", va="center", rotation=0, z=6, style=None):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color or C["ink"],
        weight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=z,
        style=style,
    )


def arrow(ax, a, b, color=None, lw=1.1, ms=8, style="-|>", rad=0.0, z=8, alpha=1.0, ls="-"):
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["ink"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        zorder=z,
        alpha=alpha,
        linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def box(ax, x, y, w, h, label, ec, fc="white", color=None, size=6.5, weight="bold", r=0.008):
    rounded(ax, x, y, w, h, fc, ec, lw=0.85, r=r, z=3)
    txt(ax, x + w / 2, y + h / 2, label, size=size, color=color or ec, weight=weight, ha="center")


def bars(ax, x, y, w, h, color, vals=None, n=8, alpha=0.9):
    vals = vals or [0.55, 0.78, 0.32, 0.88, 0.45, 0.70, 0.28, 0.62]
    vals = (vals * 3)[:n]
    gap = w * 0.04
    bw = (w - gap * (n - 1)) / n
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, h * v, fc=color, ec="none", alpha=alpha, zorder=5))


def signed_bars(ax, x, y, w, h, pos="#d62728", neg="#2457c5"):
    vals = [0.45, -0.35, 0.20, -0.55, 0.65, -0.25, 0.35]
    gap = w * 0.04
    bw = (w - gap * (len(vals) - 1)) / len(vals)
    mid = y + h * 0.50
    ax.plot([x, x + w], [mid, mid], color="#b6bdc8", lw=0.5, zorder=5)
    for i, v in enumerate(vals):
        hh = abs(v) * h * 0.45
        yy = mid if v >= 0 else mid - hh
        ax.add_patch(Rectangle((x + i * (bw + gap), yy), bw, hh, fc=pos if v >= 0 else neg, ec="none", zorder=5))


def heatmap(ax, x, y, w, h, rows=3, cols=8, colors=("#f5b7b1", "#f7dc6f", "#a9dfbf", "#b3c7ff")):
    for r in range(rows):
        for c in range(cols):
            col = colors[(r * 2 + c) % len(colors)]
            ax.add_patch(
                Rectangle(
                    (x + c * w / cols, y + r * h / rows),
                    w / cols - 0.001,
                    h / rows - 0.001,
                    fc=col,
                    ec="white",
                    lw=0.18,
                    zorder=5,
                    alpha=0.88,
                )
            )


def tensor(ax, x, y, w, h, color="#9ec5ff"):
    for layer in range(4):
        dx, dy = layer * w * 0.10, layer * h * 0.08
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, fc="#eff6ff", ec="#3b5fc9", lw=0.65, zorder=3 + layer * 0.01))
        for i in range(5):
            ax.plot([x + dx + (i + 1) * w / 6] * 2, [y + dy, y + dy + h], color="#3b5fc9", lw=0.3, alpha=0.6, zorder=4)
        for j in range(4):
            ax.plot([x + dx, x + dx + w], [y + dy + (j + 1) * h / 5] * 2, color="#3b5fc9", lw=0.3, alpha=0.6, zorder=4)


def sparkline(ax, x, y, w, h, color="#2f63c7"):
    pts = [0.42, 0.58, 0.48, 0.62, 0.44, 0.72, 0.60, 0.68, 0.52, 0.74, 0.66, 0.82]
    xs = [x + i * w / (len(pts) - 1) for i in range(len(pts))]
    ys = [y + p * h for p in pts]
    ax.plot(xs, ys, color=color, lw=1.0, zorder=5)
    ax.fill_between(xs, [y] * len(xs), ys, color=color, alpha=0.07, zorder=4)


def network(ax, x, y, w, h, color):
    cols = [[0.15, 0.40, 0.65], [0.22, 0.50, 0.78], [0.34, 0.66]]
    xs = [x + w * 0.16, x + w * 0.50, x + w * 0.84]
    for i in range(2):
        for yy1 in cols[i]:
            for yy2 in cols[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y + h * yy1, y + h * yy2], color=color, lw=0.45, alpha=0.40, zorder=5)
    for i, xx in enumerate(xs):
        for yy in cols[i]:
            ax.scatter([xx], [y + h * yy], s=9, fc="white", ec=color, lw=0.7, zorder=6)


def draw():
    fig, ax = plt.subplots(figsize=(12.6, 6.1))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top input band.
    rounded(ax, 0.18, 0.870, 0.67, 0.105, "white", C["input"], lw=1.05, r=0.010)
    txt(ax, 0.515, 0.957, "Inputs (Market & Asset Data)", size=9.0, color=C["input"], weight="bold", ha="center")
    tensor(ax, 0.205, 0.895, 0.045, 0.052)
    txt(ax, 0.270, 0.925, "Market / Asset Tensor\n$X_{t-L+1:t}$\n(assets × features × time)", size=5.1, va="center")
    sparkline(ax, 0.405, 0.897, 0.130, 0.050, C["input"])
    txt(ax, 0.555, 0.924, "Market Features\n(prices, volume,\nindicators, etc.)", size=5.3, va="center")
    heatmap(ax, 0.690, 0.895, 0.050, 0.050, rows=5, cols=5, colors=("#dbeafe", "#c7d2fe", "#eff6ff"))
    txt(ax, 0.755, 0.924, "Additional Features\n(e.g., macro, sector)", size=5.3, va="center")
    txt(ax, 0.355, 0.920, "...", size=10, color=C["muted"], ha="center")
    txt(ax, 0.818, 0.920, "...", size=10, color=C["muted"], ha="center")

    # Main modules.
    rounded(ax, 0.045, 0.270, 0.305, 0.555, C["ctrl_l"], C["ctrl"], lw=1.20, r=0.013)
    rounded(ax, 0.385, 0.300, 0.255, 0.520, C["outer_l"], C["outer"], lw=1.20, r=0.013)
    rounded(ax, 0.670, 0.270, 0.250, 0.555, C["inner_l"], C["inner"], lw=1.20, r=0.013)

    txt(ax, 0.198, 0.805, "Controller", size=10.0, color=C["ctrl"], weight="bold", ha="center")
    txt(ax, 0.512, 0.802, "Outer Actor", size=10.0, color=C["outer"], weight="bold", ha="center")
    txt(ax, 0.795, 0.802, "Inner Actor", size=10.0, color=C["inner"], weight="bold", ha="center")
    txt(ax, 0.512, 0.780, "Segment-level base construction", size=5.9, color=C["outer"], style="italic", ha="center")
    txt(ax, 0.795, 0.780, "Daily refinement", size=5.9, color=C["inner"], style="italic", ha="center")

    # Controller details.
    for y, title, small in [
        (0.665, "Current Holding State", "(drifted weights, age,\nsegment return,\ndrawdown)"),
        (0.535, "Candidate State", "(outer candidate summary)"),
        (0.405, "Pairwise Features", "(turnover, concentration,\noverlap)"),
    ]:
        rounded(ax, 0.065, y, 0.135, 0.095, "white", "#9aaed6", lw=0.75, r=0.006)
        txt(ax, 0.074, y + 0.070, title, size=5.6, color=C["ink"], weight="bold")
        txt(ax, 0.074, y + 0.045, small, size=4.4, color=C["ink"], va="top")
        if "Pairwise" in title:
            heatmap(ax, 0.078, y + 0.013, 0.080, 0.022, rows=2, cols=8, colors=("#f4a5a5", "#b9d1ff", "#fff2f2"))
        else:
            bars(ax, 0.078, y + 0.012, 0.078, 0.028, C["ctrl"], n=7)

    rounded(ax, 0.230, 0.515, 0.080, 0.210, "white", "#9aaed6", lw=0.75, r=0.006)
    txt(ax, 0.270, 0.695, "State / Action\nEncoder", size=5.8, weight="bold", ha="center")
    txt(ax, 0.270, 0.648, "Asset-wise\nLSTM +\nTemporal\nAttention", size=5.0, ha="center", color=C["ink"])
    network(ax, 0.242, 0.535, 0.055, 0.075, C["ctrl"])
    arrow(ax, (0.200, 0.710), (0.230, 0.650), C["ink"], lw=0.9)
    arrow(ax, (0.200, 0.575), (0.230, 0.620), C["ink"], lw=0.9)
    arrow(ax, (0.200, 0.440), (0.230, 0.585), C["ink"], lw=0.9)

    rounded(ax, 0.222, 0.395, 0.095, 0.065, "white", "#9aaed6", lw=0.75, r=0.006)
    txt(ax, 0.270, 0.437, "Switch Head", size=5.8, weight="bold", ha="center")
    txt(ax, 0.270, 0.416, "MLP", size=5.3, ha="center")
    bars(ax, 0.238, 0.403, 0.056, 0.015, C["purple"], n=5)
    rounded(ax, 0.222, 0.300, 0.095, 0.070, "white", C["inner"], lw=0.75, r=0.006)
    txt(ax, 0.270, 0.352, "$p$(switch)", size=5.9, color=C["inner"], weight="bold", ha="center")
    ax.plot([0.238, 0.270, 0.302], [0.320, 0.345, 0.320], color=C["inner"], lw=0.7, zorder=5)
    ax.plot([0.270, 0.291], [0.320, 0.338], color=C["inner"], lw=1.0, zorder=5)
    rounded(ax, 0.222, 0.255, 0.095, 0.030, C["gold_l"], C["gold"], lw=0.75, r=0.006)
    txt(ax, 0.270, 0.270, "Hold / Switch", size=5.8, color=C["ink"], weight="bold", ha="center")
    arrow(ax, (0.270, 0.515), (0.270, 0.460), C["ink"], lw=0.9)
    arrow(ax, (0.270, 0.395), (0.270, 0.370), C["ink"], lw=0.9)

    # Outer details.
    steps = [
        (0.405, 0.705, 0.210, 0.055, "Market Window", "spark"),
        (0.405, 0.642, 0.210, 0.052, "LSTM-HA", "tokens"),
        (0.405, 0.579, 0.210, 0.052, "CAAN", "tokens"),
        (0.405, 0.516, 0.210, 0.052, "Scoring Head (MLP)", "bars"),
        (0.405, 0.453, 0.210, 0.052, "Top-K Sparse Selection", "mask"),
        (0.405, 0.390, 0.210, 0.052, "Allocation Head (Softmax)", "bars"),
        (0.405, 0.327, 0.210, 0.052, "Candidate Base Portfolio  $w_t^{cand}$", "green"),
    ]
    for x, y, w, h, label, kind in steps:
        rounded(ax, x, y, w, h, "white", C["outer"], lw=0.72, r=0.006)
        label_size = 5.1 if kind == "mask" else 5.6
        txt(ax, x + w / 2, y + h - 0.012, label, size=label_size, color=C["ink"], weight="bold", ha="center", va="top")
        if kind == "spark":
            sparkline(ax, x + 0.018, y + 0.008, w - 0.040, h * 0.45, C["outer"])
            txt(ax, x + w - 0.018, y + 0.022, "...", size=7, color=C["muted"], ha="right")
        elif kind == "tokens":
            for i in range(9):
                fc = "#c6efe8" if i % 3 else "#6cc5b8"
                ax.add_patch(Rectangle((x + 0.030 + i * 0.017, y + 0.012), 0.009, 0.010, fc=fc, ec=C["outer"], lw=0.2, zorder=5))
            txt(ax, x + w - 0.018, y + 0.020, "...", size=7, color=C["muted"], ha="right")
        elif kind == "bars":
            bars(ax, x + 0.030, y + 0.010, w - 0.065, h * 0.35, C["outer"], n=8)
            txt(ax, x + w - 0.018, y + 0.020, "...", size=7, color=C["muted"], ha="right")
        elif kind == "mask":
            for i in range(10):
                fc = C["base"] if i in {1, 4, 8} else "white"
                ax.add_patch(Rectangle((x + 0.028 + i * 0.016, y + 0.013), 0.010, 0.013, fc=fc, ec="#94a3b8", lw=0.35, zorder=5))
            txt(ax, x + w - 0.018, y + 0.020, "top-k", size=4.8, color=C["muted"], ha="right")
        else:
            bars(ax, x + 0.030, y + 0.010, w - 0.060, h * 0.35, C["base"], n=8)
            txt(ax, x + w - 0.018, y + 0.020, "...", size=7, color=C["muted"], ha="right")

    # Inner details.
    inner_steps = [
        (0.690, 0.705, 0.205, 0.055, "Short-Term Window", "spark"),
        (0.690, 0.642, 0.205, 0.052, "LSTM-Attn (2-layer)", "redtokens"),
        (0.690, 0.579, 0.205, 0.052, "Support Mask\nfrom active base", "mask"),
        (0.690, 0.486, 0.205, 0.075, "Refinement Head (MLP)", "net"),
    ]
    for x, y, w, h, label, kind in inner_steps:
        rounded(ax, x, y, w, h, "white", C["inner"], lw=0.72, r=0.006)
        label_size = 5.0 if kind == "mask" else 5.6
        txt(ax, x + w / 2, y + h - 0.012, label, size=label_size, color=C["ink"], weight="bold", ha="center", va="top")
        if kind == "spark":
            sparkline(ax, x + 0.018, y + 0.008, w - 0.040, h * 0.45, C["inner"])
            txt(ax, x + w - 0.018, y + 0.022, "...", size=7, color=C["muted"], ha="right")
        elif kind == "redtokens":
            for i in range(9):
                ax.add_patch(Rectangle((x + 0.030 + i * 0.016, y + 0.012), 0.009, 0.010, fc="#ffb4b4", ec=C["inner"], lw=0.2, zorder=5))
            txt(ax, x + w - 0.018, y + 0.020, "...", size=7, color=C["muted"], ha="right")
        elif kind == "mask":
            for i in range(10):
                fc = "#69b64b" if i in {0, 2, 7} else "white"
                ax.add_patch(Rectangle((x + 0.028 + i * 0.016, y + 0.007), 0.010, 0.011, fc=fc, ec="#94a3b8", lw=0.35, zorder=5))
            txt(ax, x + w - 0.018, y + 0.020, "...", size=7, color=C["muted"], ha="right")
        else:
            network(ax, x + 0.080, y + 0.010, 0.050, 0.040, C["inner"])
            txt(ax, x + w - 0.018, y + 0.020, "...", size=7, color=C["muted"], ha="right")

    rounded(ax, 0.690, 0.360, 0.090, 0.070, "white", C["inner"], lw=0.72, r=0.006)
    txt(ax, 0.735, 0.413, r"Target Weights $w_t^{in}$", size=5.3, weight="bold", ha="center")
    signed_bars(ax, 0.707, 0.370, 0.055, 0.028)
    rounded(ax, 0.802, 0.360, 0.092, 0.070, "white", C["inner"], lw=0.72, r=0.006)
    txt(ax, 0.848, 0.413, r"Executed Weights  $w_t$", size=5.3, weight="bold", ha="center")
    bars(ax, 0.818, 0.371, 0.054, 0.032, C["inner"], n=6)
    rounded(ax, 0.720, 0.304, 0.145, 0.036, "white", C["inner"], lw=0.72, r=0.006)
    txt(ax, 0.792, 0.322, r"$w_t=(1-\alpha)b_t+\alpha w_t^{in}$", size=6.3, color=C["ink"], weight="bold", ha="center")

    # Active base and executor.
    rounded(ax, 0.510, 0.205, 0.230, 0.075, C["base_l"], C["base"], lw=1.05, r=0.008)
    txt(ax, 0.625, 0.263, r"Active Base Portfolio  $b_t$", size=7.4, color=C["base"], weight="bold", ha="center")
    bars(ax, 0.565, 0.218, 0.105, 0.028, C["base"], n=8)

    rounded(ax, 0.930, 0.350, 0.125, 0.165, C["exec_l"], C["exec"], lw=1.05, r=0.008)
    txt(ax, 0.992, 0.492, "Executed Portfolio &\nMarket Feedback", size=6.2, color=C["exec"], weight="bold", ha="center")
    txt(ax, 0.992, 0.452, "(returns, costs, new data)", size=5.0, color=C["ink"], ha="center")
    bars(ax, 0.947, 0.372, 0.050, 0.038, C["inner"], n=6)
    sparkline(ax, 1.000, 0.373, 0.038, 0.038, C["input"])
    txt(ax, 1.043, 0.423, "...", size=7, color=C["muted"], ha="right")

    # Replace decision diamond.
    cx, cy = 0.455, 0.205
    ax.add_patch(
        Polygon(
            [(cx, cy + 0.035), (cx + 0.045, cy), (cx, cy - 0.035), (cx - 0.045, cy)],
            closed=True,
            fc=C["gold_l"],
            ec=C["gold"],
            lw=0.9,
            zorder=4,
        )
    )
    txt(ax, cx, cy, "Replace?", size=5.8, color=C["ink"], weight="bold", ha="center")

    # Timeline.
    rounded(ax, 0.100, 0.045, 0.790, 0.060, "white", C["line"], lw=0.85, r=0.006)
    arrow(ax, (0.260, 0.075), (0.845, 0.075), C["ink"], lw=0.75, style="-|>", ms=6)
    txt(ax, 0.122, 0.075, "daily evaluation", size=7.0, weight="bold")
    for i, (x, lab) in enumerate([(0.290, "Day $t$"), (0.475, "Day $t+1$"), (0.650, "Day $t+2$"), (0.825, "Day $T$")]):
        ax.scatter([x], [0.075], s=28, fc="white", ec=C["line"], lw=0.8, zorder=6)
        txt(ax, x, 0.052, lab, size=5.3, ha="center", va="top")
    txt(ax, 0.740, 0.062, "……", size=8, color=C["muted"], ha="center")

    # Cross-module arrows.
    for x in [0.270, 0.512, 0.795]:
        arrow(ax, (x, 0.870), (x, 0.825), C["input"], lw=0.9)
    arrow(ax, (0.640, 0.560), (0.670, 0.560), C["inner"], lw=1.0)
    arrow(ax, (0.385, 0.355), (0.350, 0.300), C["outer"], lw=1.0)
    arrow(ax, (0.350, 0.300), (0.455, 0.240), C["outer"], lw=1.0)
    arrow(ax, (0.315, 0.255), (0.455, 0.205), C["ctrl"], lw=1.0)
    arrow(ax, (0.500, 0.205), (0.510, 0.235), C["base"], lw=1.0)
    txt(ax, 0.392, 0.226, "No (Hold)", size=5.3, color=C["ink"], ha="center")
    txt(ax, 0.530, 0.290, "Yes (Switch)", size=5.0, color=C["ink"], ha="center")
    arrow(ax, (0.740, 0.242), (0.795, 0.360), C["base"], lw=1.0)
    arrow(ax, (0.865, 0.395), (0.930, 0.430), C["inner"], lw=1.0)
    arrow(ax, (1.055, 0.430), (1.060, 0.105), C["exec"], lw=0.9, style="-", ms=1)
    arrow(ax, (1.060, 0.105), (0.890, 0.075), C["exec"], lw=0.9, style="-|>", ms=7)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cmtflow_architecture_vector.{ext}", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
