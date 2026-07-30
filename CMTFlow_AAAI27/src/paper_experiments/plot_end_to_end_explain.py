import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#17212B",
        "axes.edgecolor": "#2C333A",
        "axes.linewidth": 0.9,
        "xtick.color": "#17212B",
        "ytick.color": "#17212B",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFE",
        "grid.color": "#E5E9EF",
        "grid.linewidth": 0.72,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 1.0,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
    }
)
import matplotlib.pyplot as plt
from PIL import Image

from paper_experiments.metrics import summarize_all


COLORS = {
    "full_controller": "#D55E00",
    "controller_outer": "#E69F00",
    "fixed_hrl": "#2F3A44",
    "fixed_hrl_no_inner": "#0072B2",
    "Controller-PG checkpoint": "#E69F00",
    "Fixed HRL checkpoint": "#2F3A44",
    "random": "#B9C2CC",
    "fixed_window": "#8B93C7",
    "fixed_window_best": "#4F46A5",
    "exit": "#009E73",
    "switch": "#CC79A7",
    "hold": "#7A7F86",
    "drawdown_fill": "#56B4E9",
    "positive_fill": "#F0E442",
    "negative_fill": "#A6CEE3",
    "paper_ink": "#17212B",
    "muted_ink": "#5E6975",
    "panel": "#F7F9FC",
    "panel_edge": "#CBD3DD",
}

ABLATION_SERIES = [
    ("Fixed HRL w/o Inner", "fixed_hrl_no_inner"),
    ("Fixed HRL", "fixed_hrl"),
    ("Controller+Outer", "controller_outer"),
    ("Full controller", "full_controller"),
]
ABLATION_ORDER = [scenario for _, scenario in ABLATION_SERIES]
MAIN_COMPARISON_SERIES = [
    ("Fixed HRL", "fixed_hrl"),
    ("Full controller", "full_controller"),
]
MAIN_COMPARISON_ORDER = [scenario for _, scenario in MAIN_COMPARISON_SERIES]
MAIN_COMPARISON_METRICS = ["total_return", "sharpe"]

FIGURE_GROUPS = {
    "fig01": "01_stage_progression",
    "fig02": "01_stage_progression",
    "fig02b": "01_stage_progression",
    "fig03": "02_inference_ablation",
    "fig03b": "02_inference_ablation",
    "fig03c": "02_inference_ablation",
    "fig04": "03_inner_alpha",
    "fig04b": "03_inner_alpha",
    "fig05": "04_switch_alignment",
    "fig05b": "04_switch_alignment",
    "fig06": "05_switch_events",
    "fig06b": "05_switch_events",
    "fig07": "06_random_switch",
    "fig08": "07_case_windows",
    "fig09": "07_case_windows",
    "fig10": "07_case_windows",
}


STYLE_BY_SCENARIO = {
    "fixed_hrl": {"linestyle": (0, (4, 2)), "linewidth": 2.15, "zorder": 3},
    "fixed_hrl_no_inner": {"linestyle": (0, (1, 2)), "linewidth": 2.0, "zorder": 2},
    "controller_outer": {"linestyle": "-", "linewidth": 2.35, "zorder": 4},
    "full_controller": {"linestyle": "-", "linewidth": 2.75, "zorder": 5},
    "stage_hrl_fixed_best_fixed_hrl": {"linestyle": (0, (4, 2)), "linewidth": 2.2, "zorder": 3},
    "stage_controller_best_full_controller": {"linestyle": "-", "linewidth": 2.45, "zorder": 4},
}

DISPLAY_LABELS = {
    "Fixed HRL w/o Inner": "No Inner Fixed",
    "Fixed HRL": "Fixed HRL",
    "Controller+Outer": "Controller + Outer",
    "Full controller": "Full Controller",
    "Fixed HRL checkpoint": "Fixed HRL",
    "Controller-PG checkpoint": "Controller-PG",
}

DROP_STAGE_LABELS = {"Final E2E checkpoint"}


CASE_WINDOWS = {
    ("nas", 49): [
        {
            "stem": "fig08_case_window_nas_2020_07_switch_cluster",
            "title": "NASDAQ Market: Switch Cluster Suppresses Drawdown",
            "start_step": 52,
            "length": 30,
            "key_step": 52,
        },
        {
            "stem": "fig09_case_window_nas_2021_04_negative_hold",
            "title": "NASDAQ Market: Switch Avoids a Negative Hold Path",
            "start_step": 232,
            "length": 30,
            "key_step": 248,
        },
    ],
    ("sh", 90): [
        {
            "stem": "fig10_case_window_sh_2021_07_large_avoidance",
            "title": "SH Market: Switch Avoids a Large Hold Loss",
            "start_step": 348,
            "length": 30,
            "key_step": 365,
        },
    ],
}


def _market_label(market: str) -> str:
    labels = {"sh": "CSI-300", "nas": "Nasdaq-100"}
    return labels.get(str(market).lower(), str(market).upper())


def _paper_title(title: str, market: str) -> str:
    return f"{title}: {_market_label(market)}"


def _style_axis(ax, *, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2C333A")
    ax.spines["bottom"].set_color("#2C333A")
    ax.tick_params(length=3.5, width=0.8)


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label), str(label))


def _line_style_for(scenario: str, label: str) -> dict:
    style = STYLE_BY_SCENARIO.get(str(scenario), {}).copy()
    if not style and ("Fixed HRL" in str(label) or str(scenario) == "fixed_hrl"):
        style = {"linestyle": (0, (4, 2)), "linewidth": 2.1, "zorder": 3}
    style.setdefault("linestyle", "-")
    style.setdefault("linewidth", 2.35)
    style.setdefault("zorder", 3)
    return style


def _series_color(scenario: str, label: str):
    return COLORS.get(str(scenario), COLORS.get(str(label), None))


def _fmt_pct(value, digits=1, signed=False):
    if not np.isfinite(value):
        return "n/a"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.{digits}f}%"


def _portfolio_summary(df: pd.DataFrame, *, normalized=False) -> dict:
    if df.empty or "portfolio_value" not in df:
        return {}
    values = pd.to_numeric(df["portfolio_value"], errors="coerce").dropna()
    if values.empty:
        return {}
    if normalized:
        total_return = (float(values.iloc[-1]) / max(float(values.iloc[0]), 1e-12) - 1.0) * 100.0
    else:
        total_return = float(pd.to_numeric(df.get("cumulative_return"), errors="coerce").dropna().iloc[-1] * 100.0) if "cumulative_return" in df and pd.to_numeric(df.get("cumulative_return"), errors="coerce").notna().any() else (float(values.iloc[-1]) / max(float(values.iloc[0]), 1e-12) - 1.0) * 100.0
    if "drawdown" in df:
        dd = pd.to_numeric(df["drawdown"], errors="coerce").max() * 100.0
    else:
        dd = _max_drawdown(values) * 100.0
    switch_count = int(pd.to_numeric(df.get("is_switch", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "is_switch" in df else None
    free_switch_count = int(pd.to_numeric(df.get("is_free_switch", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "is_free_switch" in df else None
    return {
        "return_pct": total_return,
        "mdd_pct": float(dd),
        "switch_count": switch_count,
        "free_switch_count": free_switch_count,
    }


def _add_metric_box(ax, lines, *, loc="upper left"):
    if not lines:
        return None
    anchors = {
        "upper left": (0.018, 0.965, "left", "top"),
        "upper right": (0.982, 0.965, "right", "top"),
        "lower left": (0.018, 0.035, "left", "bottom"),
        "lower right": (0.982, 0.035, "right", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper left"])
    return ax.text(
        x,
        y,
        "\n".join(str(line) for line in lines if str(line)),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.4,
        linespacing=1.26,
        color=COLORS["paper_ink"],
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.08",
            "facecolor": "white",
            "alpha": 0.94,
            "edgecolor": COLORS["panel_edge"],
            "linewidth": 0.75,
        },
        zorder=10,
    )


def _annotate_endpoint(ax, line, label: str):
    xdata = np.asarray(line.get_xdata(), dtype=object)
    ydata = np.asarray(line.get_ydata(), dtype=float)
    finite = np.where(np.isfinite(ydata))[0]
    if len(finite) == 0:
        return None
    idx = int(finite[-1])
    return ax.annotate(
        _display_label(label),
        (xdata[idx], ydata[idx]),
        xytext=(7, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.2,
        color=line.get_color(),
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def _annotate_bar(ax, bar, text, *, color=None):
    y = bar.get_height()
    va = "bottom" if y >= 0 else "top"
    offset = 3 if y >= 0 else -3
    ax.annotate(
        text,
        (bar.get_x() + bar.get_width() / 2, y),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=8.1,
        color=color or COLORS["paper_ink"],
        fontweight="bold" if color else "normal",
    )


def _figure_group_for_stem(stem: str) -> str:
    first = str(stem).split("_", 1)[0]
    if "case_window" in str(stem):
        return "07_case_windows"
    return FIGURE_GROUPS.get(first, "99_misc")


def _clear_generated_figures(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()
    for path in output_dir.glob("*.pdf"):
        path.unlink()
    for group in sorted(set(FIGURE_GROUPS.values()) | {"99_misc"}):
        group_dir = output_dir / group
        if not group_dir.exists():
            continue
        for path in group_dir.glob("*.png"):
            path.unlink()
        for path in group_dir.glob("*.pdf"):
            path.unlink()


EXPORT_DPI = 450


def _flatten_png_to_pdf(png_path: Path, pdf_path: Path, *, dpi: int = EXPORT_DPI):
    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, "white")
            background.paste(img, mask=img.getchannel("A"))
            rgb = background
        else:
            rgb = img.convert("RGB")
        rgb.save(pdf_path, "PDF", resolution=float(dpi))


def _save(fig, output_dir: Path, stem: str, tight=True):
    target_dir = Path(output_dir) / _figure_group_for_stem(stem)
    target_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    png_path = target_dir / f"{stem}.png"
    pdf_path = target_dir / f"{stem}.pdf"
    editable_pdf_path = target_dir / f"{stem}_editable.pdf"
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(editable_pdf_path, bbox_inches="tight")
    _flatten_png_to_pdf(png_path, pdf_path, dpi=EXPORT_DPI)
    plt.close(fig)


def _drawdown_series(values):
    values = pd.to_numeric(values, errors="coerce").astype(float)
    peak = np.maximum.accumulate(np.maximum(values, 1e-12))
    return (peak - values) / np.maximum(peak, 1e-12)


def _max_drawdown(values):
    dd = _drawdown_series(values)
    return float(np.nanmax(dd)) if len(dd) else np.nan


def _parse_curve(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([])
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return np.array([])
    if not isinstance(parsed, (list, tuple)):
        return np.array([])
    return np.asarray(parsed, dtype=float)


def _load_trace(input_dir: Path, market: str, seed: int, scenario: str):
    path = input_dir / "traces" / f"{market}_seed{seed}_{scenario}_portfolio.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _wealth_multiple(df: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(df["portfolio_value"], errors="coerce")
    if values.dropna().empty:
        return pd.Series(dtype="float64")
    base = float(values.dropna().iloc[0])
    return values / max(base, 1e-12)


def _fixed_window_search_dirs(input_dir: Path):
    input_dir = Path(input_dir)
    candidates = [
        input_dir / "_cache" / "fixed_windows",
        input_dir / "fixed_windows",
        input_dir.parent / "paper_experiments_final" / "_cache" / "fixed_windows",
        Path("paper_experiments_outputs") / "paper_experiments_final" / "_cache" / "fixed_windows",
    ]
    seen = set()
    for path in candidates:
        resolved = Path(path)
        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            yield resolved


def _fixed_window_paths(input_dir: Path, market: str, seed: int):
    paths = []
    pattern = f"{market}_seed{seed}_fixed_window_*_portfolio.csv"
    for directory in _fixed_window_search_dirs(input_dir):
        for path in directory.glob(pattern):
            match = re.search(r"_fixed_window_(\d+)_portfolio\.csv$", path.name)
            if not match:
                continue
            paths.append((int(match.group(1)), path))
    unique = {}
    for window, path in paths:
        unique[window] = path
    return sorted(unique.items(), key=lambda item: item[0])


def _portfolio_summary_row(label: str, df: pd.DataFrame, *, window: int = None) -> dict:
    row = {"label": label, "fixed_window_days": window}
    row.update(summarize_all(df))
    return row


def _format_pct(value: float) -> str:
    return "" if pd.isna(value) else f"{float(value) * 100.0:.2f}%"


def _format_ratio(value: float) -> str:
    return "" if pd.isna(value) else f"{float(value):.2f}"


def _write_fixed_window_stats(output_dir: Path, market: str, seed: int, stats: pd.DataFrame, full_row: pd.Series) -> None:
    group_dir = Path(output_dir) / "06_random_switch"
    group_dir.mkdir(parents=True, exist_ok=True)
    csv_path = group_dir / f"fig07_fixed_window_timing_stats_{market}_seed{seed}.csv"
    stats.to_csv(csv_path, index=False)

    fixed = stats[stats["label"].eq("Fixed window")].copy()
    n = len(fixed)
    if n == 0:
        return
    best_tr = fixed.loc[pd.to_numeric(fixed["total_return"], errors="coerce").idxmax()]
    best_sharpe = fixed.loc[pd.to_numeric(fixed["sharpe"], errors="coerce").idxmax()]
    best_cr = fixed.loc[pd.to_numeric(fixed["calmar"], errors="coerce").idxmax()]
    best_mdd = fixed.loc[pd.to_numeric(fixed["max_drawdown"], errors="coerce").idxmin()]
    full_tr = float(full_row["total_return"])
    full_sharpe = float(full_row["sharpe"])
    full_mdd = float(full_row["max_drawdown"])
    full_cr = float(full_row["calmar"])
    beat_tr = int((pd.to_numeric(fixed["total_return"], errors="coerce") < full_tr).sum())
    beat_sharpe = int((pd.to_numeric(fixed["sharpe"], errors="coerce") < full_sharpe).sum())
    beat_mdd = int((pd.to_numeric(fixed["max_drawdown"], errors="coerce") > full_mdd).sum())
    beat_cr = int((pd.to_numeric(fixed["calmar"], errors="coerce") < full_cr).sum())
    market_name = "Nasdaq-100" if market == "nas" else "CSI-300"
    md = f"""# Dense Fixed Holding-Window Timing Baseline ({market_name}, seed {seed})

这张图展示 Dense Fixed Holding-Window Timing Baseline。比较对象是大量不同固定持仓期窗口：{int(fixed['fixed_window_days'].min())}d 到 {int(fixed['fixed_window_days'].max())}d，共 {n} 个 fixed holding-window baselines。

## 怎么看图

- 灰紫色细线：不同固定持仓期窗口的累计财富曲线。
- 红色线：learned controller 的实际累计财富曲线。
- 左图统计框：controller 相比固定窗口集合的胜出次数。
- 右图柱形面板：controller 在 TR、Sharpe、MDD 和 CR 上相对 60 个固定窗口的胜出比例。

## 统计结论

- Learned controller 的 TR 为 {_format_pct(full_tr)}，Sharpe 为 {_format_ratio(full_sharpe)}，MDD 为 {_format_pct(full_mdd)}，CR 为 {_format_ratio(full_cr)}。
- 固定窗口中最高 TR 来自 {int(best_tr['fixed_window_days'])}d，TR 为 {_format_pct(best_tr['total_return'])}；controller 在 TR 上优于 {beat_tr}/{n} 个固定窗口。
- 固定窗口中最高 Sharpe 来自 {int(best_sharpe['fixed_window_days'])}d，Sharpe 为 {_format_ratio(best_sharpe['sharpe'])}；controller 在 Sharpe 上优于 {beat_sharpe}/{n} 个固定窗口。
- 固定窗口中最低 MDD 来自 {int(best_mdd['fixed_window_days'])}d，MDD 为 {_format_pct(best_mdd['max_drawdown'])}；controller 在 MDD 上优于 {beat_mdd}/{n} 个固定窗口。
- 固定窗口中最高 CR 来自 {int(best_cr['fixed_window_days'])}d，CR 为 {_format_ratio(best_cr['calmar'])}；controller 在 CR 上优于 {beat_cr}/{n} 个固定窗口。

注意：{market_name} 上可能存在少数事后挑选的固定窗口（例如 {int(best_tr['fixed_window_days'])}d）在部分指标上高于 controller。因此论文中不应写成“controller 在所有固定窗口和所有指标上都是第一”。更合理的结论是：controller 不需要事后选择固定窗口，却在固定窗口集合中取得高分位表现，并在关键风险指标上体现出更稳定的控制能力。

## 可写入论文的表述

Compared with a dense set of fixed holding-window baselines, the learned controller achieves high-percentile risk-return performance without ex-post selection of a constant holding period. This result indicates that the controller learns state-dependent timing for revising the active base portfolio rather than relying on a manually tuned fixed window.
"""
    (group_dir / f"fig07_fixed_window_timing_stats_{market}_seed{seed}.md").write_text(md, encoding="utf-8")


def _prepare_ablation_rows(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty or "scenario" not in group:
        return pd.DataFrame()
    rows = group.copy()
    if "stage" in rows:
        rows = rows[rows["stage"].isna()]
    rows = rows[rows["scenario"].isin(ABLATION_ORDER)].copy()
    rows["scenario"] = pd.Categorical(rows["scenario"], categories=ABLATION_ORDER, ordered=True)
    return rows.sort_values("scenario").reset_index(drop=True)


def _prepare_main_comparison_rows(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty or "scenario" not in group:
        return pd.DataFrame()
    rows = group.copy()
    if "stage" in rows:
        rows = rows[rows["stage"].isna()]
    rows = rows[rows["scenario"].isin(MAIN_COMPARISON_ORDER)].copy()
    rows["scenario"] = pd.Categorical(rows["scenario"], categories=MAIN_COMPARISON_ORDER, ordered=True)
    return rows.sort_values("scenario").reset_index(drop=True)


def _plot_curves(input_dir: Path, output_dir: Path, market: str, seed: int, scenarios, stem: str, title: str, drawdown=False):
    fig, ax = plt.subplots(figsize=(7.75, 4.55))
    plotted = []
    for label, scenario in scenarios:
        df = _load_trace(input_dir, market, seed, scenario)
        if df.empty:
            continue
        x = pd.to_datetime(df["date"])
        if drawdown:
            y = pd.to_numeric(df["drawdown"], errors="coerce") * 100.0
        else:
            values = pd.to_numeric(df["portfolio_value"], errors="coerce")
            y = values / max(float(values.iloc[0]), 1e-12)
        style = _line_style_for(scenario, label)
        line = ax.plot(
            x,
            y,
            label=_display_label(label),
            color=_series_color(scenario, label),
            **style,
        )[0]
        plotted.append((label, scenario, df, x, y, line))

    if not drawdown and len(plotted) >= 2:
        fixed = next((item for item in plotted if "Fixed HRL" in str(item[0]) or item[1] == "fixed_hrl"), None)
        full = next((item for item in plotted if "Final" in str(item[0]) or item[1] == "full_controller"), None)
        if fixed is not None and full is not None and len(fixed[3]) == len(full[3]):
            ax.fill_between(
                full[3],
                full[4],
                fixed[4],
                where=np.asarray(full[4]) >= np.asarray(fixed[4]),
                color=COLORS["positive_fill"],
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
            ax.fill_between(
                full[3],
                full[4],
                fixed[4],
                where=np.asarray(full[4]) < np.asarray(fixed[4]),
                color=COLORS["negative_fill"],
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )

    for label, scenario, df, _x, _y, line in plotted:
        if scenario in {"full_controller", "stage_best_model_full_controller"} or "Final" in str(label):
            _annotate_endpoint(ax, line, label)

    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)" if drawdown else "Normalized portfolio value")
    if drawdown:
        ax.invert_yaxis()
    _style_axis(ax)
    ax.legend(frameon=False, loc="best", ncol=1)
    _save(fig, output_dir, stem)


def _plot_bar(df: pd.DataFrame, output_dir: Path, market: str, seed: int, stem: str, title: str, metrics=None):
    if df.empty:
        return
    metrics = list(metrics or ["total_return", "sharpe", "max_drawdown", "switch_count"])
    metric_titles = {
        "total_return": "Total Return (%)",
        "sharpe": "Sharpe",
        "max_drawdown": "Max Drawdown (%)",
        "switch_count": "Switch Count",
    }
    fig_width = 6.8 if len(metrics) <= 2 else 12.6
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 4.15))
    axes = np.atleast_1d(axes)
    scenario_names = df["scenario"].astype(str).tolist()
    labels = {
        "fixed_hrl_no_inner": "No Inner\nFixed",
        "fixed_hrl": "Fixed\nHRL",
        "controller_outer": "Controller\n+ Outer",
        "full_controller": "Full\nController",
    }
    methods = [labels.get(name, name) for name in scenario_names]
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(df.get(metric), errors="coerce")
        scale = 100.0 if metric in {"total_return", "max_drawdown"} else 1.0
        colors = [COLORS.get(name, "#808080") for name in scenario_names]
        plot_vals = vals * scale
        bars = ax.bar(np.arange(len(vals)), plot_vals, color=colors, width=0.68, edgecolor="white", linewidth=1.0)
        ax.set_title(metric_titles.get(metric, metric.replace("_", " ").title()), pad=8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(methods, rotation=0, ha="center")
        _style_axis(ax)
        finite = plot_vals[np.isfinite(plot_vals)]
        if len(finite):
            low = min(0.0, float(finite.min()))
            high = float(finite.max())
            pad = max((high - low) * 0.14, high * 0.08 if high > 0 else 0.5, 0.5)
            ax.set_ylim(low - pad * 0.15, high + pad)
        finite_vals = [float(v) for v in plot_vals if np.isfinite(v)]
        best_idx = None
        if finite_vals and metric != "switch_count":
            if metric == "max_drawdown":
                best_val = min(finite_vals)
                best_idx = int(np.nanargmin(plot_vals))
            else:
                best_val = max(finite_vals)
                best_idx = int(np.nanargmax(plot_vals))
        for idx, (bar, val) in enumerate(zip(bars, plot_vals)):
            if np.isfinite(val):
                if idx == best_idx:
                    bar.set_edgecolor(COLORS["paper_ink"])
                    bar.set_linewidth(1.3)
                    _annotate_bar(ax, bar, f"{val:.2f}", color=COLORS["paper_ink"])
                else:
                    _annotate_bar(ax, bar, f"{val:.2f}", color=COLORS["muted_ink"])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    _save(fig, output_dir, stem)


def _plot_stage_bar(group: pd.DataFrame, output_dir: Path, market: str, seed: int):
    stage = group[group.get("stage").notna()].copy() if "stage" in group else pd.DataFrame()
    if "stage" in stage:
        stage = stage[~stage["stage"].isin(DROP_STAGE_LABELS)].copy()
    if stage.empty:
        return
    stage["method"] = stage["stage"].astype(str).map(_display_label)
    metrics = ["total_return", "sharpe", "max_drawdown", "switch_count"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.6, 4.1))
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(stage.get(metric), errors="coerce")
        scale = 100.0 if metric in {"total_return", "max_drawdown"} else 1.0
        colors = [COLORS.get(x, "#808080") for x in stage["stage"]]
        plot_vals = vals * scale
        bars = ax.bar(np.arange(len(vals)), plot_vals, color=colors, edgecolor="white", linewidth=1.0)
        ax.set_title(metric.replace("_", " ").title(), pad=8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(stage["method"], rotation=18, ha="right", fontsize=8)
        _style_axis(ax)
        finite_vals = [float(v) for v in plot_vals if np.isfinite(v)]
        best_idx = None
        if finite_vals and metric != "switch_count":
            best_idx = int(np.nanargmin(plot_vals)) if metric == "max_drawdown" else int(np.nanargmax(plot_vals))
        for idx, (bar, val) in enumerate(zip(bars, plot_vals)):
            if np.isfinite(val):
                if idx == best_idx:
                    bar.set_edgecolor(COLORS["paper_ink"])
                    bar.set_linewidth(1.3)
                _annotate_bar(ax, bar, f"{val:.2f}", color=COLORS["paper_ink"] if idx == best_idx else COLORS["muted_ink"])
    fig.suptitle(_paper_title("Stage Progression Metrics", market), fontsize=15, fontweight="bold", y=0.995)
    _save(fig, output_dir, f"fig02b_stage_progression_bar_{market}_seed{seed}")


def plot_inner_alpha(input_dir: Path, output_dir: Path, market: str, seed: int):
    fig, ax = plt.subplots(figsize=(7.55, 4.6))
    plotted = []
    for scenario in ["fixed_hrl", "full_controller"]:
        df = _load_trace(input_dir, market, seed, scenario)
        if df.empty or "inner_alpha" not in df:
            continue
        label = "Full controller" if scenario == "full_controller" else "Fixed HRL"
        y = pd.to_numeric(df["inner_alpha"], errors="coerce").fillna(0).cumsum()
        style = _line_style_for(scenario, label)
        line = ax.plot(pd.to_datetime(df["date"]), y, label=_display_label(label), color=COLORS.get(scenario), **style)[0]
        plotted.append((label, y, line))
    ax.axhline(0, color=COLORS["paper_ink"], linewidth=0.8)
    ax.set_title(_paper_title("Cumulative Inner Alpha", market), pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log alpha")
    for label, _y, line in plotted:
        if "Full" in label:
            _annotate_endpoint(ax, line, label)
    _style_axis(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, f"fig04_cumulative_inner_alpha_{market}_seed{seed}")

    df = _load_trace(input_dir, market, seed, "full_controller")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7.45, 4.55))
        alpha = pd.to_numeric(df.get("inner_alpha"), errors="coerce").dropna()
        ax.hist(alpha, bins=42, color=COLORS["fixed_hrl_no_inner"], alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=COLORS["paper_ink"], linewidth=1.0)
        if len(alpha):
            ax.axvline(alpha.mean(), color=COLORS["full_controller"], linewidth=2.0, label=f"mean={alpha.mean():.4f}")
        ax.set_title(_paper_title("Daily Inner Alpha Distribution", market), pad=10)
        ax.set_xlabel("Daily inner alpha")
        ax.set_ylabel("Frequency")
        _style_axis(ax)
        ax.legend(frameon=False)
        _save(fig, output_dir, f"fig04b_inner_alpha_distribution_{market}_seed{seed}")


def plot_alignment(input_dir: Path, output_dir: Path, market: str, seed: int):
    path = input_dir / "traces" / f"{market}_seed{seed}_full_controller_actions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    free = df[df.get("decision_type") == "free_decision"].copy()
    if free.empty:
        return
    free["exit_prob"] = pd.to_numeric(free["exit_prob"], errors="coerce")
    free["controller_switch_advantage"] = pd.to_numeric(free["controller_switch_advantage"], errors="coerce")
    bin_edges = np.linspace(0, 1, 11)
    bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    free["bin"] = pd.cut(free["exit_prob"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    grouped = free.groupby("bin", observed=False)["controller_switch_advantage"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] > 0].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.55, 4.65))
    means = pd.to_numeric(grouped["mean"], errors="coerce")
    bar_colors = [COLORS["full_controller"] if val >= 0 else COLORS["fixed_hrl_no_inner"] for val in means]
    bars = ax.bar(np.arange(len(grouped)), means, color=bar_colors, alpha=0.9, edgecolor="white", linewidth=0.7)
    ax.axhline(0, color=COLORS["paper_ink"], linewidth=0.8)
    for bar, count in zip(bars, grouped["count"]):
        if int(count) > 0:
            _annotate_bar(ax, bar, f"n={int(count)}", color=COLORS["muted_ink"])
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels([str(x) for x in grouped["bin"]], rotation=0, ha="center", fontsize=8.4)
    ax.set_title(_paper_title("Exit Probability Calibration", market), pad=10)
    ax.set_xlabel("Exit probability bin")
    ax.set_ylabel("Average switch advantage")
    _style_axis(ax)
    _save(fig, output_dir, f"fig05_exit_prob_calibration_{market}_seed{seed}")

    fig, ax = plt.subplots(figsize=(7.05, 4.55))
    data = [
        free.loc[free["is_switch"] == 0, "controller_switch_advantage"].dropna(),
        free.loc[free["is_switch"] == 1, "controller_switch_advantage"].dropna(),
    ]
    box = ax.boxplot(
        data,
        tick_labels=["Held", "Switched"],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": COLORS["paper_ink"], "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "color": COLORS["panel_edge"]},
        whiskerprops={"color": COLORS["panel_edge"]},
        capprops={"color": COLORS["panel_edge"]},
    )
    for patch, color in zip(box["boxes"], [COLORS["hold"], COLORS["switch"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.axhline(0, color=COLORS["paper_ink"], linewidth=0.8)
    ax.set_title(_paper_title("Switch Advantage: Held vs. Switched", market), pad=10)
    ax.set_ylabel("Switch advantage")
    _style_axis(ax)
    _save(fig, output_dir, f"fig05b_switch_advantage_switched_vs_held_{market}_seed{seed}")


def plot_switch_events(input_dir: Path, output_dir: Path, market: str, seed: int):
    path = input_dir / "traces" / f"{market}_seed{seed}_full_controller_switch_events.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    metrics = [
        ("pre_return_20", "Pre-switch realized"),
        ("post_actual_return_20", "Actual post-switch"),
        ("post_hold_return_20", "Continue-hold counterfactual"),
        ("post_switch_return_20", "Switch counterfactual"),
    ]
    fig, ax = plt.subplots(figsize=(7.65, 4.65))
    means = [pd.to_numeric(df[col], errors="coerce").mean() * 100.0 for col, _ in metrics]
    bars = ax.bar(
        np.arange(len(metrics)),
        means,
        color=[COLORS["exit"], COLORS["full_controller"], COLORS["hold"], COLORS["controller_outer"]],
        edgecolor="white",
        linewidth=1.0,
    )
    ax.axhline(0, color=COLORS["paper_ink"], linewidth=0.8)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean 20-day log return (%)")
    ax.set_title(_paper_title("Switch Event Study", market), pad=10)
    best_idx = int(np.nanargmax(means)) if any(np.isfinite(means)) else None
    for idx, (bar, val) in enumerate(zip(bars, means)):
        if np.isfinite(val):
            if idx == best_idx:
                bar.set_edgecolor(COLORS["paper_ink"])
                bar.set_linewidth(1.3)
            _annotate_bar(ax, bar, f"{val:.2f}", color=COLORS["paper_ink"] if idx == best_idx else COLORS["muted_ink"])
    avoided = pd.to_numeric(df.get("avoided_loss_20"), errors="coerce").dropna()
    _style_axis(ax)
    _save(fig, output_dir, f"fig06_switch_event_study_{market}_seed{seed}")

    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    ax.hist(
        avoided * 100.0,
        bins=min(34, max(6, len(avoided))),
        color=COLORS["full_controller"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.axvline(0, color=COLORS["paper_ink"], linewidth=0.9)
    if len(avoided):
        ax.axvline(avoided.mean() * 100.0, color=COLORS["exit"], linewidth=2.0, label=f"mean={avoided.mean()*100:.2f}%")
    ax.set_title(_paper_title("Avoided-Loss Distribution", market), pad=10)
    ax.set_xlabel("Switch advantage / avoided loss (%)")
    ax.set_ylabel("Event count")
    _style_axis(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, f"fig06b_switch_avoided_loss_distribution_{market}_seed{seed}")


def plot_random(input_dir: Path, output_dir: Path, market: str, seed: int):
    full = _load_trace(input_dir, market, seed, "full_controller")
    fixed = _load_trace(input_dir, market, seed, "fixed_hrl")
    fixed_paths = _fixed_window_paths(input_dir, market, seed)
    if full.empty or not fixed_paths:
        return

    rows = []
    loaded = []
    for window, path in fixed_paths:
        df = pd.read_csv(path)
        if df.empty or "portfolio_value" not in df:
            continue
        y = _wealth_multiple(df)
        if y.empty:
            continue
        loaded.append((window, df, y))
        rows.append(_portfolio_summary_row("Fixed window", df, window=window))
    if not loaded:
        return

    full_row = _portfolio_summary_row("Learned controller", full)
    stats = pd.DataFrame([full_row, *rows])
    _write_fixed_window_stats(output_dir, market, seed, stats, pd.Series(full_row))

    fixed_stats = stats[stats["label"].eq("Fixed window")].copy()
    best_windows_by_metric = {
        "TR": int(fixed_stats.loc[pd.to_numeric(fixed_stats["total_return"], errors="coerce").idxmax(), "fixed_window_days"]),
        "Sharpe": int(fixed_stats.loc[pd.to_numeric(fixed_stats["sharpe"], errors="coerce").idxmax(), "fixed_window_days"]),
        "MDD": int(fixed_stats.loc[pd.to_numeric(fixed_stats["max_drawdown"], errors="coerce").idxmin(), "fixed_window_days"]),
        "CR": int(fixed_stats.loc[pd.to_numeric(fixed_stats["calmar"], errors="coerce").idxmax(), "fixed_window_days"]),
    }

    fig, (ax, ax_rank) = plt.subplots(
        1,
        2,
        figsize=(10.4, 5.0),
        gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.18},
    )
    windows = np.array([item[0] for item in loaded], dtype=float)
    cmap = plt.cm.Purples
    denom = max(float(windows.max() - windows.min()), 1.0)
    for window, df, y in loaded:
        shade = 0.25 + 0.45 * ((float(window) - float(windows.min())) / denom)
        color = cmap(shade)
        ax.plot(pd.to_datetime(df["date"]), y, color=color, alpha=0.30, linewidth=0.76, zorder=1)

    if not fixed.empty:
        fixed_y = _wealth_multiple(fixed)
        if not fixed_y.empty:
            ax.plot(
                pd.to_datetime(fixed["date"]),
                fixed_y,
                label="Reference fixed HRL (30d)",
                color=COLORS["fixed_hrl"],
                linewidth=1.55,
                linestyle="--",
                alpha=0.85,
                zorder=2,
            )
    full_y = _wealth_multiple(full)
    full_line = ax.plot(
        pd.to_datetime(full["date"]),
        full_y,
        label="Learned controller",
        color=COLORS["full_controller"],
        linewidth=2.85,
        linestyle="-",
        zorder=4,
    )[0]
    _annotate_endpoint(ax, full_line, "Learned controller")

    n = len(fixed_stats)
    full_tr = float(full_row["total_return"])
    full_sharpe = float(full_row["sharpe"])
    full_mdd = float(full_row["max_drawdown"])
    full_cr = float(full_row["calmar"])
    beat_tr = int((pd.to_numeric(fixed_stats["total_return"], errors="coerce") < full_tr).sum())
    beat_sharpe = int((pd.to_numeric(fixed_stats["sharpe"], errors="coerce") < full_sharpe).sum())
    beat_mdd = int((pd.to_numeric(fixed_stats["max_drawdown"], errors="coerce") > full_mdd).sum())
    beat_cr = int((pd.to_numeric(fixed_stats["calmar"], errors="coerce") < full_cr).sum())
    _add_metric_box(
        ax,
        [
            f"Fixed windows: {int(windows.min())}-{int(windows.max())}d, n={n}",
            f"TR wins: {beat_tr}/{n}; Sharpe wins: {beat_sharpe}/{n}",
            f"MDD wins: {beat_mdd}/{n}; CR wins: {beat_cr}/{n}",
        ],
        loc="lower right",
    )

    percentile_rows = [
        ("TR", beat_tr / n * 100.0),
        ("Sharpe", beat_sharpe / n * 100.0),
        ("MDD", beat_mdd / n * 100.0),
        ("CR", beat_cr / n * 100.0),
    ]
    y_pos = np.arange(len(percentile_rows))
    vals = [v for _, v in percentile_rows]
    rank_colors = [COLORS["full_controller"], COLORS["full_controller"], COLORS["exit"], COLORS["full_controller"]]
    ax_rank.barh(y_pos, vals, color=rank_colors, alpha=0.88, height=0.58)
    ax_rank.axvline(50, color=COLORS["muted_ink"], linestyle=(0, (3, 2)), linewidth=1.0)
    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels([name for name, _ in percentile_rows])
    ax_rank.set_xlim(0, 100)
    ax_rank.set_xlabel("Controller win rate vs fixed windows (%)")
    ax_rank.set_title("Controller percentile", pad=8)
    for i, value in enumerate(vals):
        ax_rank.text(min(value + 2.0, 98.0), i, f"{value:.0f}%", va="center", ha="left" if value < 96 else "right", fontsize=8.7, color=COLORS["paper_ink"])
    ax_rank.invert_yaxis()
    _style_axis(ax_rank, grid_axis="x")

    fig.suptitle(_paper_title("Dense Fixed Holding-Window Timing Baseline", market), y=0.985, fontsize=13.8, fontweight="bold")
    ax.set_title("Wealth paths across fixed holding windows", pad=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.145, top=0.840, wspace=0.22)
    _save(fig, output_dir, f"fig07_random_switch_comparison_{market}_seed{seed}", tight=False)


def plot_case_windows(input_dir: Path, output_dir: Path, market: str, seed: int):
    cases = CASE_WINDOWS.get((market, seed), [])
    if not cases:
        return
    full = _load_trace(input_dir, market, seed, "full_controller")
    fixed = _load_trace(input_dir, market, seed, "fixed_hrl")
    actions_path = input_dir / "traces" / f"{market}_seed{seed}_full_controller_actions.csv"
    events_path = input_dir / "traces" / f"{market}_seed{seed}_full_controller_switch_events.csv"
    if full.empty or fixed.empty or not actions_path.exists() or not events_path.exists():
        return
    actions = pd.read_csv(actions_path)
    events = pd.read_csv(events_path)
    for df in (full, fixed, actions, events):
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"])
        if "step" in df:
            df["step"] = pd.to_numeric(df["step"], errors="coerce")

    for case in cases:
        start_step = int(case["start_step"])
        length = int(case.get("length", 30))
        end_step = start_step + length - 1
        full_window = full[(full["step"] >= start_step) & (full["step"] <= end_step)].copy()
        fixed_window = fixed[(fixed["step"] >= start_step) & (fixed["step"] <= end_step)].copy()
        if len(full_window) < 8 or len(fixed_window) < 8:
            continue
        merged = full_window[["date", "step", "portfolio_value"]].rename(columns={"portfolio_value": "full_value"})
        merged = merged.merge(
            fixed_window[["date", "portfolio_value"]].rename(columns={"portfolio_value": "fixed_value"}),
            on="date",
            how="inner",
        )
        if merged.empty:
            continue
        merged["full_norm"] = pd.to_numeric(merged["full_value"], errors="coerce") / max(float(merged["full_value"].iloc[0]), 1e-12)
        merged["fixed_norm"] = pd.to_numeric(merged["fixed_value"], errors="coerce") / max(float(merged["fixed_value"].iloc[0]), 1e-12)
        merged["full_return_pct"] = (merged["full_norm"] - 1.0) * 100.0
        merged["fixed_return_pct"] = (merged["fixed_norm"] - 1.0) * 100.0
        merged["full_dd_pct"] = _drawdown_series(merged["full_norm"]) * 100.0
        merged["fixed_dd_pct"] = _drawdown_series(merged["fixed_norm"]) * 100.0

        start_date = merged["date"].iloc[0]
        end_date = merged["date"].iloc[-1]
        switch_rows = actions[
            (actions["date"] >= start_date)
            & (actions["date"] <= end_date)
            & (pd.to_numeric(actions.get("is_free_switch"), errors="coerce").fillna(0) > 0)
        ].copy()
        switch_rows = switch_rows.merge(merged[["date", "full_return_pct"]], on="date", how="left")
        key_step = int(case["key_step"])
        key_action = actions[(actions["step"] == key_step) & (pd.to_numeric(actions.get("is_switch"), errors="coerce").fillna(0) > 0)].head(1)
        key_event = events[events["step"] == key_step].head(1)

        full_ret = float(merged["full_norm"].iloc[-1] - 1.0)
        fixed_ret = float(merged["fixed_norm"].iloc[-1] - 1.0)
        full_mdd = _max_drawdown(merged["full_norm"])
        fixed_mdd = _max_drawdown(merged["fixed_norm"])

        fig = plt.figure(figsize=(11.0, 8.15))
        grid = fig.add_gridspec(4, 1, height_ratios=[2.15, 1.2, 1.2, 1.65], hspace=0.18)
        ax_ret = fig.add_subplot(grid[0])
        ax_dd = fig.add_subplot(grid[1], sharex=ax_ret)
        ax_exit = fig.add_subplot(grid[2], sharex=ax_ret)
        ax_cf = fig.add_subplot(grid[3])

        ax_ret.plot(merged["date"], merged["full_return_pct"], label="Full controller", color=COLORS["full_controller"], linewidth=2.85, zorder=4)
        ax_ret.plot(merged["date"], merged["fixed_return_pct"], label="Fixed HRL", color=COLORS["fixed_hrl"], linewidth=2.05, linestyle=(0, (4, 2)), zorder=3)
        ax_ret.fill_between(
            merged["date"],
            merged["full_return_pct"],
            merged["fixed_return_pct"],
            where=merged["full_return_pct"] >= merged["fixed_return_pct"],
            color=COLORS["positive_fill"],
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax_ret.fill_between(
            merged["date"],
            merged["full_return_pct"],
            merged["fixed_return_pct"],
            where=merged["full_return_pct"] < merged["fixed_return_pct"],
            color=COLORS["negative_fill"],
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        if not switch_rows.empty:
            ax_ret.scatter(
                switch_rows["date"],
                switch_rows["full_return_pct"],
                marker="v",
                s=42,
                color=COLORS["switch"],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
                label="Controller switch",
            )
            for idx, date in enumerate(switch_rows["date"], start=1):
                ax_ret.axvline(date, color=COLORS["switch"], alpha=0.14, linewidth=0.9, zorder=0)
                if idx <= 5:
                    y_val = float(switch_rows.loc[switch_rows["date"] == date, "full_return_pct"].iloc[0])
                    ax_ret.annotate(f"S{idx}", (date, y_val), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.2, color=COLORS["switch"], fontweight="bold")
        ax_ret.axhline(0, color=COLORS["paper_ink"], linewidth=0.8, alpha=0.7)
        ax_ret.set_ylabel("Window return (%)")
        _style_axis(ax_ret)
        ax_ret.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")

        ax_dd.plot(merged["date"], merged["full_dd_pct"], color=COLORS["full_controller"], linewidth=2.2, label="Full controller")
        ax_dd.plot(merged["date"], merged["fixed_dd_pct"], color=COLORS["fixed_hrl"], linewidth=1.8, linestyle=(0, (4, 2)), label="Fixed HRL")
        ax_dd.fill_between(merged["date"], merged["full_dd_pct"], merged["fixed_dd_pct"], where=merged["fixed_dd_pct"] >= merged["full_dd_pct"], color=COLORS["drawdown_fill"], alpha=0.28)
        ax_dd.set_ylabel("Drawdown (%)")
        _style_axis(ax_dd)

        free = actions[(actions["date"] >= start_date) & (actions["date"] <= end_date) & (actions["decision_type"] == "free_decision")].copy()
        free["exit_prob"] = pd.to_numeric(free.get("exit_prob"), errors="coerce")
        ax_exit.plot(free["date"], free["exit_prob"], color=COLORS["exit"], linewidth=2.15, label="Exit probability")
        ax_exit.axhline(0.5, color=COLORS["paper_ink"], linewidth=1.0, linestyle=":", alpha=0.85)
        ax_exit.text(
            0.99,
            0.51,
            "threshold=0.5",
            transform=ax_exit.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=COLORS["muted_ink"],
        )
        if not switch_rows.empty:
            switch_probs = switch_rows[["date", "exit_prob"]].copy()
            switch_probs["exit_prob"] = pd.to_numeric(switch_probs["exit_prob"], errors="coerce")
            ax_exit.scatter(switch_probs["date"], switch_probs["exit_prob"], color=COLORS["switch"], marker="v", s=46, zorder=4)
            for idx, row in enumerate(switch_probs.itertuples(index=False), start=1):
                if np.isfinite(row.exit_prob):
                    ax_exit.annotate(f"S{idx}", (row.date, row.exit_prob), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.7, color=COLORS["switch"], fontweight="bold")
        ax_exit.set_ylim(-0.02, 1.02)
        ax_exit.set_ylabel("Exit prob.")
        _style_axis(ax_exit)
        ax_exit.set_xlabel("Date")

        plotted_cf = False
        if not key_action.empty:
            row = key_action.iloc[0]
            hold_curve = _parse_curve(row.get("hold_curve_20"))
            switch_curve = _parse_curve(row.get("switch_curve_20"))
            horizon = np.arange(max(len(hold_curve), len(switch_curve)))
            if len(hold_curve):
                ax_cf.plot(np.arange(len(hold_curve)), (hold_curve - 1.0) * 100.0, label="Continue-hold counterfactual", color=COLORS["hold"], linewidth=2.05, linestyle=(0, (4, 2)))
                plotted_cf = True
            if len(switch_curve):
                ax_cf.plot(np.arange(len(switch_curve)), (switch_curve - 1.0) * 100.0, label="Switch counterfactual", color=COLORS["controller_outer"], linewidth=2.35)
                plotted_cf = True
            if len(horizon):
                actual = full[(full["step"] >= key_step) & (full["step"] < key_step + len(horizon))].copy()
                if len(actual) >= 2:
                    actual_norm = pd.to_numeric(actual["portfolio_value"], errors="coerce") / max(float(actual["portfolio_value"].iloc[0]), 1e-12)
                    ax_cf.plot(np.arange(len(actual_norm)), (actual_norm - 1.0) * 100.0, label="Actual controller path", color=COLORS["full_controller"], linewidth=2.15)
                    plotted_cf = True
        if not key_event.empty:
            ev = key_event.iloc[0]
            info = (
                f"Key switch: hold {ev['post_hold_return_20'] * 100:.1f}% vs "
                f"switch {ev['post_switch_return_20'] * 100:.1f}%"
            )
            ax_cf.text(
                0.01,
                0.05,
                info,
                transform=ax_cf.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                color=COLORS["muted_ink"],
            )
        if plotted_cf:
            ax_cf.axhline(0, color=COLORS["paper_ink"], linewidth=0.8, alpha=0.7)
            ax_cf.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
        ax_cf.set_xlabel("Trading days after key switch")
        ax_cf.set_ylabel("20-day path (%)")
        _style_axis(ax_cf)

        fig.suptitle(
            f"{case['title']} ({start_date.date()} to {end_date.date()})",
            fontsize=15,
            fontweight="bold",
            y=0.97,
        )
        for ax in [ax_ret, ax_dd, ax_exit, ax_cf]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for label in ax_ret.get_xticklabels() + ax_dd.get_xticklabels():
            label.set_visible(False)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.08, hspace=0.28)
        _save(fig, output_dir, case["stem"], tight=False)


def main_from_paths(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metrics = input_dir / "metrics" / "all_metrics.csv"
    if not metrics.exists():
        return
    _clear_generated_figures(output_dir)
    all_metrics = pd.read_csv(metrics)
    for (market, seed), group in all_metrics.groupby(["market", "seed"]):
        market = str(market)
        seed = int(seed)
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            [
                ("Fixed HRL checkpoint", "stage_hrl_fixed_best_fixed_hrl"),
                ("Controller-PG checkpoint", "stage_controller_best_full_controller"),
            ],
            f"fig01_stage_progression_cumulative_{market}_seed{seed}",
            _paper_title("Stage Progression: Cumulative Return", market),
            drawdown=False,
        )
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            [
                ("Fixed HRL checkpoint", "stage_hrl_fixed_best_fixed_hrl"),
                ("Controller-PG checkpoint", "stage_controller_best_full_controller"),
            ],
            f"fig02_stage_progression_drawdown_{market}_seed{seed}",
            _paper_title("Stage Progression: Drawdown", market),
            drawdown=True,
        )
        _plot_stage_bar(group, output_dir, market, seed)
        main_comparison = _prepare_main_comparison_rows(group)
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            MAIN_COMPARISON_SERIES,
            f"fig03_main_comparison_cumulative_{market}_seed{seed}",
            _paper_title("Main Comparison: Cumulative Return", market),
            drawdown=False,
        )
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            MAIN_COMPARISON_SERIES,
            f"fig03b_main_comparison_drawdown_{market}_seed{seed}",
            _paper_title("Main Comparison: Drawdown", market),
            drawdown=True,
        )
        _plot_bar(
            main_comparison,
            output_dir,
            market,
            seed,
            f"fig03c_main_comparison_bar_{market}_seed{seed}",
            _paper_title("Main Comparison Metrics", market),
            metrics=MAIN_COMPARISON_METRICS,
        )
        plot_inner_alpha(input_dir, output_dir, market, seed)
        plot_alignment(input_dir, output_dir, market, seed)
        plot_switch_events(input_dir, output_dir, market, seed)
        plot_random(input_dir, output_dir, market, seed)
        plot_case_windows(input_dir, output_dir, market, seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate paper explanation figures.")
    parser.add_argument("--input_dir", default="paper_experiments_outputs/end_to_end_explain")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    main_from_paths(input_dir, output_dir)


if __name__ == "__main__":
    main()
