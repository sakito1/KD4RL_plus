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
        "font.size": 7.6,
        "axes.linewidth": 0.8,
    }
)


COL = {
    "ink": "#182230",
    "muted": "#667085",
    "line": "#9aa6b2",
    "grid": "#e5e7eb",
    "panel": "#f8fafc",
    "outer": "#0f766e",
    "outer_l": "#d9f7ef",
    "controller": "#b42318",
    "controller_l": "#ffe4e2",
    "inner": "#1d4ed8",
    "inner_l": "#dbeafe",
    "exec": "#b45309",
    "exec_l": "#fef3c7",
    "violet": "#6d28d9",
    "violet_l": "#ede9fe",
    "gray_l": "#f3f6fa",
}


def make_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rounded(ax, x, y, w, h, fc, ec, lw=1.15, r=0.026, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=7.5, color=None, weight=None, ha="left", va="center", rotation=0, z=6):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color or COL["ink"],
        weight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=z,
    )


def module_title(ax, x, y, title, subtitle=None, color=None):
    label(ax, x, y, title, size=8.6, color=color or COL["ink"], weight="bold", va="top")
    if subtitle:
        label(ax, x, y - 0.028, subtitle, size=6.6, color=COL["muted"], va="top")


def arrow(ax, a, b, color=None, lw=1.25, style="-|>", ms=8.5, rad=0.0, alpha=1.0, z=7):
    arr = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or COL["ink"],
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def pill(ax, x, y, text, fc, ec, w=None, h=0.034, size=6.3):
    w = w or max(0.060, 0.0065 * len(text) + 0.028)
    rounded(ax, x, y - h / 2, w, h, fc, ec, lw=0.75, r=0.012, z=5)
    label(ax, x + w / 2, y, text, size=size, color=ec, weight="bold", ha="center")
    return w


def subbox(ax, x, y, w, h, text, fc="white", ec=None, color=None, size=6.5, weight="bold"):
    ec = ec or COL["line"]
    rounded(ax, x, y, w, h, fc, ec, lw=0.75, r=0.010, z=4)
    label(ax, x + w / 2, y + h / 2, text, size=size, color=color or COL["ink"], weight=weight, ha="center")


def tensor_stack(ax, x, y, w, h, layers=4, color="#93c5fd"):
    dx, dy = w * 0.065, h * 0.065
    for layer in range(layers - 1, -1, -1):
        xo = x + layer * dx
        yo = y + layer * dy
        ax.add_patch(Rectangle((xo, yo), w, h, fc="#eaf4ff", ec="#3b82f6", lw=0.75, zorder=3 + layer * 0.01))
        rows, cols = 4, 5
        for r in range(rows):
            for c in range(cols):
                alpha = 0.22 + 0.45 * ((r * 3 + c * 5 + layer) % 9) / 8
                ax.add_patch(
                    Rectangle(
                        (xo + c * w / cols, yo + r * h / rows),
                        w / cols - 0.001,
                        h / rows - 0.001,
                        fc=mpl.colors.to_rgba(color, alpha),
                        ec="white",
                        lw=0.18,
                        zorder=4,
                    )
                )
    label(ax, x + w * 0.50, y - 0.022, "assets", size=6.2, color=COL["muted"], ha="center")
    label(ax, x - 0.006, y + h * 0.50, "features", size=6.2, color=COL["muted"], ha="right", rotation=55)
    label(ax, x + w + layers * dx + 0.014, y + h * 0.48, "time", size=6.2, color=COL["muted"], rotation=90)


def mini_heatmap(ax, x, y, w, h, rows=4, cols=7, color="#2563eb"):
    for r in range(rows):
        for c in range(cols):
            val = ((r + 2) * (c + 3)) % 13 / 12
            ax.add_patch(
                Rectangle(
                    (x + c * w / cols, y + r * h / rows),
                    w / cols - 0.001,
                    h / rows - 0.001,
                    fc=mpl.colors.to_rgba(color, 0.12 + 0.50 * val),
                    ec="white",
                    lw=0.18,
                    zorder=5,
                )
            )


def mini_bars(ax, x, y, w, h, color, values=None, n=8):
    values = values or [0.26, 0.55, 0.38, 0.76, 0.62, 0.92, 0.48, 0.84]
    values = values[:n]
    gap = w * 0.035
    bw = (w - gap * (n - 1)) / n
    for i, v in enumerate(values):
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, h * v, fc=color, ec="none", alpha=0.86, zorder=5))


def network_icon(ax, x, y, w, h, color):
    xs = [x + w * 0.18, x + w * 0.46, x + w * 0.75]
    ys = [
        [y + h * 0.25, y + h * 0.55, y + h * 0.82],
        [y + h * 0.20, y + h * 0.50, y + h * 0.78],
        [y + h * 0.34, y + h * 0.68],
    ]
    for i in range(len(xs) - 1):
        for yy in ys[i]:
            for yy2 in ys[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [yy, yy2], color=color, alpha=0.40, lw=0.55, zorder=5)
    for i, xx in enumerate(xs):
        for yy in ys[i]:
            ax.scatter([xx], [yy], s=9, color="white", edgecolor=color, linewidth=0.7, zorder=6)


def save_both(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = make_ax((11.2, 5.45))

    label(ax, 0.018, 0.965, "CMTFlow inference at day $t$", size=10.8, weight="bold")
    label(
        ax,
        0.018,
        0.925,
        "The controller gates segment-level base revision; the inner actor makes support-constrained daily tilts.",
        size=7.1,
        color=COL["muted"],
    )

    # Inputs.
    rounded(ax, 0.020, 0.510, 0.155, 0.330, COL["panel"], "#4f83cc", lw=1.15, r=0.026)
    module_title(ax, 0.034, 0.814, "Market tensor", "$X_{t-L:t}$", "#1d4ed8")
    tensor_stack(ax, 0.045, 0.595, 0.065, 0.087, layers=4, color="#60a5fa")
    label(ax, 0.041, 0.548, "prices, volume,\ncausal features", size=6.3, color=COL["ink"], va="top")

    rounded(ax, 0.020, 0.145, 0.155, 0.235, "white", COL["line"], lw=1.05, r=0.026)
    module_title(ax, 0.034, 0.352, "Portfolio state", "$s_t^{ctrl}$", COL["ink"])
    mini_bars(ax, 0.045, 0.158, 0.092, 0.038, "#6b7280")
    label(ax, 0.043, 0.294, "drifted $\\tilde{w}_t$\nactive base $b_t$\nage / return / drawdown", size=6.2, va="top")

    # Outer actor module.
    rounded(ax, 0.225, 0.560, 0.300, 0.315, COL["outer_l"], COL["outer"], lw=1.35, r=0.030)
    module_title(ax, 0.241, 0.846, "Outer actor", "LSTM-HA + CAAN -> Top-K base", COL["outer"])
    subbox(ax, 0.245, 0.665, 0.052, 0.092, "LSTM-\nHA", fc="#ecfeff", ec=COL["outer"], color=COL["outer"])
    subbox(ax, 0.319, 0.665, 0.060, 0.092, "CAAN", fc="#ecfeff", ec=COL["outer"], color=COL["outer"])
    network_icon(ax, 0.327, 0.672, 0.044, 0.076, COL["outer"])
    subbox(ax, 0.403, 0.665, 0.050, 0.092, "MLP", fc="#ecfeff", ec=COL["outer"], color=COL["outer"])
    subbox(ax, 0.463, 0.665, 0.045, 0.092, "Top-$K$", fc="#ecfeff", ec=COL["outer"], color=COL["outer"], size=6.0)
    arrow(ax, (0.297, 0.711), (0.319, 0.711), COL["outer"], lw=1.05)
    arrow(ax, (0.379, 0.711), (0.403, 0.711), COL["outer"], lw=1.05)
    arrow(ax, (0.453, 0.711), (0.463, 0.711), COL["outer"], lw=1.05)
    mini_heatmap(ax, 0.246, 0.608, 0.085, 0.042, color="#0f766e")
    mini_bars(ax, 0.356, 0.608, 0.098, 0.052, COL["outer"])
    label(ax, 0.244, 0.590, "asset features", size=5.9, color=COL["muted"])
    label(ax, 0.356, 0.590, "scores $q_t^{out}$", size=5.9, color=COL["muted"])
    label(ax, 0.455, 0.614, "$w_t^{cand}$", size=8.0, color=COL["outer"], weight="bold")

    # Controller.
    rounded(ax, 0.565, 0.498, 0.205, 0.370, COL["controller_l"], COL["controller"], lw=1.35, r=0.030)
    module_title(ax, 0.581, 0.839, "Controller", "market encoder + state/action gate", COL["controller"])
    subbox(ax, 0.584, 0.705, 0.062, 0.070, "current\nbase", fc="#fff7f7", ec=COL["controller"], color=COL["controller"], size=5.8)
    subbox(ax, 0.674, 0.705, 0.062, 0.070, "candidate\nbase", fc="#fff7f7", ec=COL["controller"], color=COL["controller"], size=5.8)
    label(ax, 0.660, 0.740, "$\\oplus$", size=10.0, color=COL["controller"], weight="bold", ha="center")
    subbox(ax, 0.604, 0.602, 0.110, 0.060, "state/action\nfeatures", fc="#fff7f7", ec=COL["controller"], color=COL["controller"])
    arrow(ax, (0.637, 0.705), (0.637, 0.663), COL["controller"], lw=0.95)
    arrow(ax, (0.704, 0.705), (0.704, 0.663), COL["controller"], lw=0.95)
    subbox(ax, 0.612, 0.525, 0.090, 0.060, "exit logit\n+ switch adv", fc="#ffffff", ec=COL["controller"], color=COL["controller"], size=5.8)
    arrow(ax, (0.658, 0.610), (0.658, 0.585), COL["controller"], lw=1.05)
    label(ax, 0.712, 0.555, "$p_t^{sw}$", size=8.0, color=COL["controller"], weight="bold")
    label(ax, 0.607, 0.509, "$g_t\\in\\{0,1\\}$", size=7.4, color=COL["controller"], weight="bold")

    # Base memory and branch.
    rounded(ax, 0.815, 0.555, 0.155, 0.285, "#fffdf7", COL["exec"], lw=1.25, r=0.028)
    module_title(ax, 0.830, 0.810, "Base memory", "adaptive segment boundary", COL["exec"])
    subbox(ax, 0.835, 0.700, 0.050, 0.060, "hold", fc="#fff7ed", ec=COL["exec"], color=COL["exec"])
    subbox(ax, 0.905, 0.700, 0.050, 0.060, "switch", fc="#fff7ed", ec=COL["exec"], color=COL["exec"])
    label(ax, 0.833, 0.653, "$g_t=0:$ use $\\tilde{b}_t$", size=6.3)
    label(ax, 0.833, 0.622, "$g_t=1:$ use $w_t^{cand}$", size=6.3)
    label(ax, 0.835, 0.585, "$\\tau_{k+1}\\leftarrow t$ when switched", size=6.1, color=COL["muted"])

    # Inner actor.
    rounded(ax, 0.225, 0.105, 0.300, 0.310, COL["inner_l"], COL["inner"], lw=1.35, r=0.030)
    module_title(ax, 0.241, 0.386, "Inner actor", "LSTM + temporal attention", COL["inner"])
    subbox(ax, 0.247, 0.250, 0.065, 0.083, "LSTM\nAttn", fc="#eff6ff", ec=COL["inner"], color=COL["inner"])
    subbox(ax, 0.338, 0.250, 0.074, 0.083, "support\nmask", fc="#eff6ff", ec=COL["inner"], color=COL["inner"])
    subbox(ax, 0.439, 0.250, 0.056, 0.083, "tilt\nhead", fc="#eff6ff", ec=COL["inner"], color=COL["inner"])
    arrow(ax, (0.312, 0.292), (0.338, 0.292), COL["inner"], lw=1.05)
    arrow(ax, (0.412, 0.292), (0.439, 0.292), COL["inner"], lw=1.05)
    mini_bars(ax, 0.250, 0.170, 0.092, 0.048, COL["inner"])
    mini_bars(ax, 0.373, 0.170, 0.092, 0.048, "#6b7280")
    label(ax, 0.249, 0.148, "$w_t^{in}$", size=7.2, color=COL["inner"], weight="bold")
    label(ax, 0.367, 0.148, "$\\Delta w_t^{inner}=w_t-b_t$", size=6.6, color=COL["muted"])
    label(ax, 0.250, 0.120, "$w_t=(1-\\alpha)b_t+\\alpha w_t^{in}$", size=6.3)

    # Executor.
    rounded(ax, 0.815, 0.150, 0.155, 0.245, COL["exec_l"], COL["exec"], lw=1.25, r=0.028)
    module_title(ax, 0.830, 0.365, "Executor", "realized portfolio path", COL["exec"])
    mini_bars(ax, 0.836, 0.257, 0.074, 0.050, COL["exec"])
    label(ax, 0.832, 0.226, "transaction cost $\\mu_t$", size=6.3)
    label(ax, 0.832, 0.195, "$V_{t+1}=V_t\\mu_t y_t^{\\top}w_t$", size=6.3)

    # Main arrows and labels.
    arrow(ax, (0.176, 0.675), (0.225, 0.720), COL["outer"], lw=1.35)
    arrow(ax, (0.176, 0.250), (0.225, 0.255), COL["inner"], lw=1.30)
    arrow(ax, (0.176, 0.335), (0.200, 0.485), COL["controller"], lw=1.00, style="-", ms=1)
    arrow(ax, (0.200, 0.485), (0.545, 0.485), COL["controller"], lw=1.00, style="-", ms=1)
    arrow(ax, (0.545, 0.485), (0.565, 0.610), COL["controller"], lw=1.05)
    arrow(ax, (0.525, 0.715), (0.565, 0.760), COL["controller"], lw=1.10)
    arrow(ax, (0.525, 0.715), (0.815, 0.725), COL["outer"], lw=1.35)
    arrow(ax, (0.770, 0.590), (0.815, 0.690), COL["controller"], lw=1.35)
    arrow(ax, (0.890, 0.555), (0.520, 0.340), COL["exec"], lw=1.10, rad=0.10)
    arrow(ax, (0.525, 0.260), (0.815, 0.270), COL["inner"], lw=1.35)
    arrow(ax, (0.893, 0.555), (0.893, 0.395), COL["exec"], lw=1.35)
    arrow(ax, (0.970, 0.245), (0.988, 0.245), COL["exec"], lw=1.25)
    arrow(ax, (0.988, 0.245), (0.988, 0.060), COL["exec"], style="-", lw=1.05, ms=1)
    arrow(ax, (0.988, 0.060), (0.100, 0.060), COL["exec"], style="-", lw=1.05, ms=1)
    arrow(ax, (0.100, 0.060), (0.100, 0.145), COL["exec"], lw=1.05)
    label(ax, 0.490, 0.038, "next-day feedback: realized return, drifted weights, and updated holding state", size=6.3, color=COL["exec"], ha="center")

    label(ax, 0.533, 0.742, "$w_t^{cand}$", size=7.5, color=COL["outer"], weight="bold")
    label(ax, 0.777, 0.648, "$g_t$", size=7.5, color=COL["controller"], weight="bold")
    label(ax, 0.660, 0.287, "$w_t$", size=7.5, color=COL["inner"], weight="bold")
    label(ax, 0.735, 0.427, "$b_t$", size=7.3, color=COL["exec"], weight="bold")

    save_both(fig, "cmtflow_architecture_vector")


def training_flow():
    fig, ax = make_ax((11.2, 4.65))

    label(ax, 0.018, 0.960, "Training and evaluation protocol", size=10.8, weight="bold")
    label(
        ax,
        0.018,
        0.920,
        "Fixed-HRL actor warmup/joint training is followed by controller PG and controller-active joint finetuning.",
        size=7.1,
        color=COL["muted"],
    )

    # Chronological data strip.
    rounded(ax, 0.035, 0.735, 0.930, 0.105, "#ffffff", COL["line"], lw=0.9, r=0.018)
    label(ax, 0.050, 0.803, "Chronological data", size=7.8, weight="bold")
    splits = [
        (0.175, 0.445, "Train", "#d9f7ef", COL["outer"]),
        (0.445, 0.610, "Validation", "#ede9fe", COL["violet"]),
        (0.610, 0.925, "Test", "#fef3c7", COL["exec"]),
    ]
    for x0, x1, name, fc, ec in splits:
        rounded(ax, x0, 0.765, x1 - x0, 0.042, fc, ec, lw=0.75, r=0.008, z=4)
        label(ax, (x0 + x1) / 2, 0.786, name, size=6.6, color=ec, weight="bold", ha="center")
    label(ax, 0.050, 0.762, "causal normalization + transaction cost", size=6.2, color=COL["muted"])

    # Three training stages.
    stages = [
        (
            0.045,
            "Stage I",
            "Outer warmup",
            COL["outer"],
            COL["outer_l"],
            ["fixed $H=30$ reference", "LSTM-HA + CAAN scores", "Top-$K$ candidate base"],
            "$R^{out}$: segment log return",
        ),
        (
            0.360,
            "Stage II",
            "Inner warmup",
            COL["inner"],
            COL["inner_l"],
            ["base-relative reward", "LSTM-attention encoder", "support-masked daily tilt"],
            "$R^{in}$: daily excess log return",
        ),
        (
            0.675,
            "Stages III-V",
            "Controller + joint",
            COL["controller"],
            COL["controller_l"],
            ["fixed-HRL joint backbone", "controller PG: no labels", "final E2E joint finetune"],
            "$R^{ctrl}$: return uplift - overflow",
        ),
    ]
    for x, tag, title, color, fc, bullets, reward in stages:
        rounded(ax, x, 0.405, 0.275, 0.260, fc, color, lw=1.25, r=0.028)
        pill(ax, x + 0.018, 0.632, tag, "#ffffff", color, w=0.070)
        module_title(ax, x + 0.018, 0.595, title, reward, color)
        yy = 0.520
        for item in bullets:
            ax.scatter([x + 0.030], [yy], s=10, color=color, zorder=6)
            label(ax, x + 0.045, yy, item, size=6.5)
            yy -= 0.043
        mini_heatmap(ax, x + 0.185, 0.438, 0.055, 0.045, color=color)
        mini_bars(ax, x + 0.188, 0.500, 0.052, 0.045, color, n=6)
    arrow(ax, (0.320, 0.535), (0.360, 0.535), COL["ink"], lw=1.25)
    arrow(ax, (0.635, 0.535), (0.675, 0.535), COL["ink"], lw=1.25)
    label(ax, 0.333, 0.565, "best outer", size=6.0, color=COL["muted"], ha="center")
    label(ax, 0.648, 0.565, "fixed HRL", size=6.0, color=COL["muted"], ha="center")

    # Evaluation loop.
    label(ax, 0.045, 0.320, "Daily evaluation loop", size=8.6, weight="bold")
    eval_boxes = [
        (0.050, "market state\n$X_{t-L:t},s_t$"),
        (0.235, "candidate base\n$w_t^{cand}$"),
        (0.420, "exit probability\n$p_t^{sw}$"),
        (0.605, "base update\n$g_t$"),
        (0.790, "execute $w_t$\nTR / Sharpe / MDD / CR"),
    ]
    colors = [COL["line"], COL["outer"], COL["controller"], COL["exec"], COL["inner"]]
    for (x, text), c in zip(eval_boxes, colors):
        rounded(ax, x, 0.150, 0.145, 0.115, "#ffffff", c, lw=1.0, r=0.018)
        label(ax, x + 0.0725, 0.207, text, size=6.6, ha="center")
    for i in range(len(eval_boxes) - 1):
        x = eval_boxes[i][0] + 0.145
        arrow(ax, (x, 0.207), (eval_boxes[i + 1][0], 0.207), COL["ink"], lw=1.10)
    arrow(ax, (0.862, 0.150), (0.862, 0.095), COL["exec"], style="-", lw=1.0, ms=1)
    arrow(ax, (0.862, 0.095), (0.080, 0.095), COL["exec"], style="-", lw=1.0, ms=1)
    arrow(ax, (0.080, 0.095), (0.080, 0.150), COL["exec"], lw=1.0)
    label(ax, 0.455, 0.072, "realized return and drifted weights feed the next daily state", size=6.5, color=COL["exec"], ha="center")

    # Ablation strip.
    rounded(ax, 0.045, 0.015, 0.895, 0.040, COL["gray_l"], COL["line"], lw=0.65, r=0.012)
    label(
        ax,
        0.060,
        0.035,
        "Ablations: outer-only | outer+inner | outer+controller | fixed-window controllers | full CMTFlow",
        size=6.6,
        color=COL["muted"],
    )

    save_both(fig, "cmtflow_training_flow_vector")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    training_flow()
