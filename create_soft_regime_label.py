# segment_label_fullseries_k2.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # 引入高斯滤波用于最终平滑
# ----------------- Plot Style Utils -----------------
def set_plot_style():
    """
    统一绘图风格：更适合论文/报告展示
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "#f5f6f8",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d0d4da",
        "axes.linewidth": 0.8,

        "grid.color": "#dfe3e8",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,

        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d0d4da",

        "savefig.facecolor": "#f5f6f8",
        "savefig.bbox": "tight",

        "font.family": "DejaVu Sans"
    })


def _style_axis(ax, hide_top_right=False):
    ax.grid(True, alpha=0.65)
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c7ccd4")
    ax.spines["bottom"].set_color("#c7ccd4")


def _plot_price_pretty(ax, dates, price, label="Price"):
    price = np.asarray(price, dtype=np.float32)
    ymin = float(np.nanmin(price))
    ax.plot(
        dates, price,
        color="#2b2b2b",
        lw=2.2,
        alpha=0.97,
        label=label,
        zorder=3
    )
    ax.fill_between(
        dates, price, ymin,
        color="#9aa4b2",
        alpha=0.10,
        zorder=1
    )
    _style_axis(ax, hide_top_right=True)
# ----------------- 1) 去噪：EMA on log price -----------------
def ema_smooth(x, span=10):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float32)


# ----------------- 2) 初分段：极值点（差分变号） -----------------
def find_extrema_points(x):
    """
    x: (T,) 平滑后的序列
    返回：分段边界点 indices（包含0和T-1）
    """
    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    if T <= 2:
        return np.array([0, max(T - 1, 0)], dtype=np.int64)

    dx = np.diff(x)  # (T-1,)
    signs = dx[:-1] * dx[1:]
    extrema = np.where(signs < 0)[0] + 1  # 极值点位置

    pts = np.unique(np.concatenate([np.array([0, T - 1]), extrema.astype(np.int64)]))
    pts.sort()
    return pts


# ----------------- 3) 段特征：斜率 + 形状向量 -----------------
def segment_slope(logp, i0, i1):
    """
    简单斜率：(logp[i1]-logp[i0])/(i1-i0)
    """
    if i1 <= i0:
        return 0.0
    return float((logp[i1] - logp[i0]) / (i1 - i0))


def segment_shape(logp, i0, i1, m=32):
    """
    段形状向量：对 [i0,i1] 的 logp 做等距采样 m 点，然后 z-score
    """
    if i1 <= i0:
        return np.zeros((m,), dtype=np.float32)
    idx = np.linspace(i0, i1, m).astype(np.int64)
    v = logp[idx].astype(np.float32)
    v = v - v.mean()
    v = v / (v.std() + 1e-6)
    return v


def shape_dist(v1, v2):
    return float(np.mean((v1 - v2) ** 2))


# ----------------- 4) 合并相邻段（核心） -----------------
def merge_segments(logp, pts,
                   slope_tol=0.0005,
                   shape_tol=0.5,
                   min_len=5,
                   shape_m=32,
                   max_passes=50):
    """
    迭代合并：
      - 短段（<min_len）优先合并
      - 若相邻两段 |r1-r2| <= slope_tol 且 shape_dist <= shape_tol 则合并
    """
    pts = pts.astype(np.int64).copy()
    if len(pts) <= 2:
        return pts

    def seg_feat(k):
        i0, i1 = int(pts[k]), int(pts[k + 1])
        L = i1 - i0
        r = segment_slope(logp, i0, i1)
        sh = segment_shape(logp, i0, i1, m=shape_m)
        return i0, i1, L, r, sh

    for _ in range(max_passes):
        changed = False
        if len(pts) <= 2:
            break

        k = 0
        while k < len(pts) - 2:
            i0, i1, L1, r1, sh1 = seg_feat(k)
            j0, j1, L2, r2, sh2 = seg_feat(k + 1)

            # 规则1：短段强制合并（避免碎片）
            if L1 < min_len or L2 < min_len:
                pts = np.delete(pts, k + 1)
                changed = True
                continue

            # 规则2：斜率相近 + 形状相近 -> 合并
            if abs(r1 - r2) <= slope_tol:
                d = shape_dist(sh1, sh2)
                if d <= shape_tol:
                    pts = np.delete(pts, k + 1)
                    changed = True
                    continue

            k += 1

        if not changed:
            break

    return pts


# ----------------- 5) K=2 段标签：用斜率阈值二分类 -----------------
def label_segments_k2(logp, pts, threshold="median"):
    segs = []
    slopes = []
    for k in range(len(pts) - 1):
        i0, i1 = int(pts[k]), int(pts[k + 1])
        r = segment_slope(logp, i0, i1)
        segs.append((i0, i1, r))
        slopes.append(r)

    slopes = np.array(slopes, dtype=np.float32)
    if len(slopes) == 0:
        T = len(logp)
        return np.zeros((T,), dtype=np.int64), [], 0.0

    if threshold == "median":
        thr = float(np.median(slopes))
    elif threshold == "zero":
        thr = 0.0
    elif isinstance(threshold, (float, int)):
        thr = float(threshold)
    else:
        raise ValueError("threshold must be 'median', 'zero', or a float.")

    labels = np.zeros((len(logp),), dtype=np.int64)
    seg_info = []
    for (i0, i1, r) in segs:
        lab = 1 if r >= thr else 0
        labels[i0:i1 + 1] = lab
        seg_info.append((i0, i1, float(r), int(lab)))

    return labels, seg_info, thr


# ----------------- 6) K=2 标签生成器 (Base) -----------------
def build_fullseries_segment_labels_k2(
        df,
        price_col="adjclose",
        smooth_span=10,
        slope_tol=0.0005,
        shape_tol=0.5,
        min_len=10,
        threshold="median"
):
    price = df[price_col].to_numpy(dtype=np.float32)
    idx = df.index

    logp = np.log(price + 1e-8)
    logp_s = ema_smooth(logp, span=smooth_span)

    pts0 = find_extrema_points(logp_s)
    pts = merge_segments(
        logp_s, pts0,
        slope_tol=slope_tol,
        shape_tol=shape_tol,
        min_len=min_len,
        shape_m=32,
        max_passes=50
    )

    labels, seg_info, thr = label_segments_k2(logp_s, pts, threshold=threshold)
    return idx, price, labels, pts, seg_info, thr


# ==============================================================================
#  新增模块：Ensemble Soft Label (集成软标签)
# ==============================================================================

def build_ensemble_soft_label(
        df,
        price_col="adjclose",
        # 参数网格：你可以根据需要增减
        spans=[5, 10, 15, 20, 25, 30],  # 覆盖从周线到季线的趋势
        slope_tols=[0.0001, 0.0003, 0.0005, 0.0010],  # 覆盖对“平盘”的不同定义
        min_lens=[5, 10, 15, 20, 25]  # 覆盖不同的抗噪级别
):
    """
    通过网格遍历多组参数，生成多个二分类标签，最后取平均得到 0~1 的软标签。
    """
    print(f"[Ensemble] Generating labels with {len(spans)}x{len(slope_tols)}x{len(min_lens)} combinations...")
    label_matrix = []

    # 遍历所有参数组合
    for span in spans:
        for tol in slope_tols:
            for mlen in min_lens:
                # 调用基础生成函数
                _, _, labels, _, _, _ = build_fullseries_segment_labels_k2(
                    df,
                    price_col=price_col,
                    smooth_span=span,
                    slope_tol=tol,
                    min_len=mlen,
                    shape_tol=0.5,
                    threshold="median"  # 使用中位数切分，保证相对强弱
                )
                label_matrix.append(labels.astype(np.float32))

    # 堆叠所有结果: (N_models, T)
    label_matrix = np.stack(label_matrix, axis=0)

    # 取平均得到概率: (T,)
    ensemble_mean = np.mean(label_matrix, axis=0)

    # (可选) 计算分歧度，如果你想知道哪些时候市场分歧很大
    ensemble_std = np.std(label_matrix, axis=0)

    return ensemble_mean, ensemble_std


# ==============================================================================
#  新增模块：双轴可视化 (价格 vs 概率)
# ==============================================================================

# def plot_price_vs_prob(dates, price, prob, title="Ensemble Soft Label", outpath=None):
#     fig, ax1 = plt.subplots(figsize=(15, 6))
#
#     # 左轴画价格
#     color = 'tab:gray'
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Price', color=color)
#     ax1.plot(dates, price, color=color, lw=1.0, alpha=0.8, label="Price")
#     ax1.tick_params(axis='y', labelcolor=color)
#
#     # 右轴画概率
#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('Bull Probability', color=color)
#
#     # 绘制概率曲线
#     ax2.plot(dates, prob, color=color, lw=1.5, alpha=0.9, label="Bull Prob")
#
#     # 填充颜色增强视觉效果：>0.5 红，<0.5 绿
#     ax2.fill_between(dates, prob, 0.5, where=(prob >= 0.5), color='red', alpha=0.15)
#     ax2.fill_between(dates, prob, 0.5, where=(prob < 0.5), color='green', alpha=0.15)
#
#     ax2.set_ylim(-0.05, 1.05)
#     ax2.tick_params(axis='y', labelcolor=color)
#
#     # 添加参考线
#     ax2.axhline(0.5, color='black', linestyle='--', alpha=0.3)
#     ax2.axhline(0.8, color='red', linestyle=':', alpha=0.3)
#     ax2.axhline(0.2, color='green', linestyle=':', alpha=0.3)
#
#     plt.title(title)
#     plt.tight_layout()
#
#     if outpath:
#         plt.savefig(outpath, dpi=150)
#         print(f"Plot saved to {outpath}")
#     else:
#         plt.show()
#     plt.close()
def plot_price_vs_prob(dates, price, prob, title="Ensemble Soft Label", outpath=None):
    fig, ax1 = plt.subplots(figsize=(15, 6.5))

    price = np.asarray(price, dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32)

    # 左轴：价格
    _plot_price_pretty(ax1, dates, price, label="Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="#2b2b2b")
    ax1.tick_params(axis="y", labelcolor="#2b2b2b")

    # 右轴：概率
    ax2 = ax1.twinx()
    ax2.set_ylabel("Bull Probability", color="#355c9a")

    ax2.plot(
        dates, prob,
        color="#4c72b0",
        lw=2.2,
        alpha=0.96,
        label="Bull Prob",
        zorder=4
    )

    ax2.fill_between(
        dates, prob, 0.5,
        where=(prob >= 0.5),
        color="#9ecae1",
        alpha=0.35,
        interpolate=True,
        zorder=2
    )
    ax2.fill_between(
        dates, prob, 0.5,
        where=(prob < 0.5),
        color="#fdd0a2",
        alpha=0.35,
        interpolate=True,
        zorder=2
    )

    ax2.axhline(0.5, color="#7f7f7f", linestyle="--", lw=1.1, alpha=0.75)
    ax2.axhline(0.8, color="#c44e52", linestyle=":", lw=1.0, alpha=0.55)
    ax2.axhline(0.2, color="#55a868", linestyle=":", lw=1.0, alpha=0.55)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#355c9a")
    _style_axis(ax2, hide_top_right=False)

    plt.title(title, pad=12)
    plt.tight_layout(pad=1.8)

    if outpath:
        plt.savefig(outpath, dpi=220)
        print(f"Plot saved to {outpath}")
    else:
        plt.show()
    plt.close()

# ----------------- [改造点 1] 基于斜率强度的概率映射 -----------------
def label_segments_probabilistic(logp, pts, slope_stats=None, sensitivity=3.0):
    """
    不再返回 0/1，而是返回基于斜率相对强弱的概率 0~1。

    :param slope_stats: (median, std) 用于标准化斜率。如果为 None，则使用当前序列自适应计算。
    :param sensitivity: Sigmoid 的陡峭程度。值越大，越接近 0/1 硬截断；值越小越平滑。
    """
    segs = []
    slopes = []

    # 1. 提取所有段的斜率
    for k in range(len(pts) - 1):
        i0, i1 = int(pts[k]), int(pts[k + 1])
        r = segment_slope(logp, i0, i1)
        segs.append((i0, i1, r))
        slopes.append(r)

    slopes = np.array(slopes, dtype=np.float32)
    if len(slopes) == 0:
        return np.zeros((len(logp),), dtype=np.float32)

    # 2. 确定分布统计量 (自适应或全局)
    if slope_stats is None:
        # 使用中位数和 MAD (Median Absolute Deviation) 比 std 更鲁棒
        med = np.median(slopes)
        mad = np.median(np.abs(slopes - med))
        scale = mad * 1.4826 if mad > 1e-9 else np.std(slopes)
    else:
        med, scale = slope_stats

    if scale < 1e-9: scale = 1.0  # 避免除零

    # 3. Sigmoid 映射
    # z = (r - median) / scale
    # prob = 1 / (1 + exp(-sensitivity * z))
    labels = np.zeros((len(logp),), dtype=np.float32)

    for (i0, i1, r) in segs:
        z_score = (r - med) / scale
        # Sigmoid function centered at median
        prob = 1.0 / (1.0 + np.exp(-sensitivity * z_score))
        prob = 0.0 if prob < 1e-6 else prob
        labels[i0:i1 + 1] = prob  # 整段赋予相同的概率值

    return labels


# ----------------- [改造点 2] 单次生成流程更新 -----------------
def build_fullseries_prob_labels(
        df,
        price_col="adjclose",
        smooth_span=10,
        slope_tol=0.0005,
        min_len=10,
        sensitivity=2.0  # 控制 S 曲线的平缓度
):
    price = df[price_col].to_numpy(dtype=np.float32)
    logp = np.log(price + 1e-8)
    logp_s = ema_smooth(logp, span=smooth_span)

    pts0 = find_extrema_points(logp_s)
    pts = merge_segments(
        logp_s, pts0,
        slope_tol=slope_tol,
        shape_tol=0.5,
        min_len=min_len,
        shape_m=32
    )

    # 使用概率映射代替硬切分
    labels = label_segments_probabilistic(logp_s, pts, sensitivity=sensitivity)

    return labels


# ----------------- [改造点 3] 集成与最终平滑 -----------------
def build_smooth_ensemble_label(
        df,
        price_col="adjclose",
        spans=[10, 20, 30],
        slope_tols=[0.0001, 0.0005],
        min_lens=[10, 20],
        sensitivity=2.0,  # 这里的 sensitivity 控制每一层基模型的“软度”
        final_smooth_sigma=2.0  # 最后高斯滤波的强度
):
    print(f"[Ensemble] Generating Soft Labels with Probabilistic Mapping...")
    label_matrix = []

    for span in spans:
        for tol in slope_tols:
            for mlen in min_lens:
                # 调用新的概率生成函数
                lb = build_fullseries_prob_labels(
                    df,
                    price_col=price_col,
                    smooth_span=span,
                    slope_tol=tol,
                    min_len=mlen,
                    sensitivity=sensitivity
                )
                label_matrix.append(lb)

    # 堆叠 (N, T)
    label_matrix = np.stack(label_matrix, axis=0)

    # 1. 均值聚合
    raw_ensemble = np.mean(label_matrix, axis=0)

    # 2. [关键] 最终高斯平滑
    # 这一步是为了消除由于不同参数分段点不一致导致的微小跳变
    if final_smooth_sigma > 0:
        smooth_ensemble = gaussian_filter1d(raw_ensemble, sigma=final_smooth_sigma)
        # 截断回 [0, 1] 防止溢出
        smooth_ensemble = np.clip(smooth_ensemble, 0.0, 1.0)
    else:
        smooth_ensemble = raw_ensemble

    return smooth_ensemble, np.std(label_matrix, axis=0)
def find_micro_macro_divergence_regions(
        macro_label,
        micro_label,
        dates=None,
        diff_threshold=0.25,
        min_region_len=3
):
    """
    找出 micro label 相对 macro label 差异显著的区间。

    参数
    ----
    macro_label : array-like, (T,)
        宏观软标签，通常是 build_smooth_ensemble_label 的输出
    micro_label : array-like, (T,)
        微观标签，通常是 build_fullseries_prob_labels 的输出
    dates : array-like or None
        日期索引，仅用于结果展示
    diff_threshold : float
        判定“显著差异”的阈值，默认 |micro - macro| >= 0.25
    min_region_len : int
        最短连续区间长度，过短的尖峰不单独视为显著区间

    返回
    ----
    diff : np.ndarray
        micro - macro
    abs_diff : np.ndarray
        |micro - macro|
    regions : list[dict]
        每个元素包含:
        {
            'start': 起点idx,
            'end': 终点idx,
            'length': 长度,
            'mean_diff': 区间均值差,
            'mean_abs_diff': 区间平均绝对差,
            'type': 'micro_gt_macro' or 'micro_lt_macro',
            'start_date': ...,
            'end_date': ...
        }
    """
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    if len(macro_label) != len(micro_label):
        raise ValueError("macro_label 和 micro_label 长度必须一致")

    diff = micro_label - macro_label
    abs_diff = np.abs(diff)

    flag = abs_diff >= diff_threshold
    regions = []

    start = None
    for i, v in enumerate(flag):
        if v and start is None:
            start = i
        elif (not v) and (start is not None):
            end = i - 1
            if end - start + 1 >= min_region_len:
                seg_diff = diff[start:end + 1]
                mean_diff = float(np.mean(seg_diff))
                mean_abs_diff = float(np.mean(np.abs(seg_diff)))
                regions.append({
                    "start": int(start),
                    "end": int(end),
                    "length": int(end - start + 1),
                    "mean_diff": mean_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "type": "micro_gt_macro" if mean_diff >= 0 else "micro_lt_macro",
                    "start_date": str(dates[start]) if dates is not None else int(start),
                    "end_date": str(dates[end]) if dates is not None else int(end),
                })
            start = None

    if start is not None:
        end = len(flag) - 1
        if end - start + 1 >= min_region_len:
            seg_diff = diff[start:end + 1]
            mean_diff = float(np.mean(seg_diff))
            mean_abs_diff = float(np.mean(np.abs(seg_diff)))
            regions.append({
                "start": int(start),
                "end": int(end),
                "length": int(end - start + 1),
                "mean_diff": mean_diff,
                "mean_abs_diff": mean_abs_diff,
                "type": "micro_gt_macro" if mean_diff >= 0 else "micro_lt_macro",
                "start_date": str(dates[start]) if dates is not None else int(start),
                "end_date": str(dates[end]) if dates is not None else int(end),
            })

    return diff, abs_diff, regions


def plot_macro_micro_comparison(
        dates,
        price,
        macro_label,
        micro_label,
        title="Macro vs Micro Label Comparison",
        outpath=None,
        diff_threshold=0.25,
        min_region_len=3,
        top_k_regions=8
):
    """
    绘制 macro soft label 与 micro label 的对比图，并高亮显著差异区间。

    图像结构：
    1) Price
    2) Macro vs Micro
    3) Diff = micro - macro, 并高亮显著区间
    """
    price = np.asarray(price, dtype=np.float32)
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    diff, abs_diff, regions = find_micro_macro_divergence_regions(
        macro_label=macro_label,
        micro_label=micro_label,
        dates=dates,
        diff_threshold=diff_threshold,
        min_region_len=min_region_len
    )

    # 只保留最显著的若干区间，避免图过花
    regions_sorted = sorted(regions, key=lambda x: x["mean_abs_diff"], reverse=True)
    regions_plot = regions_sorted[:top_k_regions]

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.6, 1.2]}
    )

    # ------------------------------------------------------------------
    # 1) 价格图
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(dates, price, color='tab:gray', lw=1.2, alpha=0.9, label="Price")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    # ------------------------------------------------------------------
    # 2) Macro / Micro 对比
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(dates, macro_label, color='tab:blue', lw=2.0, alpha=0.95, label="Macro Soft Label")
    ax.plot(dates, micro_label, color='tab:orange', lw=1.4, alpha=0.95, label="Micro Label")

    ax.fill_between(dates, macro_label, micro_label,
                    where=(micro_label >= macro_label),
                    alpha=0.18, label="Micro > Macro")
    ax.fill_between(dates, macro_label, micro_label,
                    where=(micro_label < macro_label),
                    alpha=0.18, label="Micro < Macro")

    ax.axhline(0.5, color='black', linestyle='--', alpha=0.35)
    ax.axhline(0.8, color='red', linestyle=':', alpha=0.2)
    ax.axhline(0.2, color='green', linestyle=':', alpha=0.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Label Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2)

    # ------------------------------------------------------------------
    # 3) 差值图
    # ------------------------------------------------------------------
    ax = axes[2]
    ax.plot(dates, diff, color='tab:purple', lw=1.3, label="Diff (Micro - Macro)")
    ax.axhline(0.0, color='black', linestyle='--', alpha=0.35)
    ax.axhline(diff_threshold, color='red', linestyle=':', alpha=0.35, label=f"+th={diff_threshold:.2f}")
    ax.axhline(-diff_threshold, color='green', linestyle=':', alpha=0.35, label=f"-th={diff_threshold:.2f}")

    # 高亮显著区间
    for reg in regions_plot:
        s, e = reg["start"], reg["end"]
        if reg["type"] == "micro_gt_macro":
            color = "red"
        else:
            color = "green"

        for a in axes:
            a.axvspan(dates[s], dates[e], color=color, alpha=0.10)

        # 在 diff 图上标注区间中心
        mid = (s + e) // 2
        ax.text(
            dates[mid],
            diff[mid],
            f"{reg['length']}d",
            fontsize=8,
            alpha=0.8
        )

    ax.set_ylabel("Micro - Macro")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")
        print(f"Plot saved to {outpath}")
    else:
        plt.show()
    plt.close()

    return {
        "diff": diff,
        "abs_diff": abs_diff,
        "regions": regions_sorted
    }
def select_top_micro_macro_divergence_regions(
        macro_label,
        micro_label,
        top_k=6,
        smooth_span=7,
        min_region_len=5,
        merge_gap=5,
        score_power=1.0
):
    """
    只选出 micro 与 macro 差异最大的若干区间，用于论文展示。
    不把所有 micro 波动都画出来，只保留最有解释价值的区间。

    返回:
        diff_raw: 原始差值 micro - macro
        diff_smooth: 平滑后的差值
        regions: list[dict]
    """
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    if len(macro_label) != len(micro_label):
        raise ValueError("macro_label 和 micro_label 长度必须一致")

    # 原始差值
    diff_raw = micro_label - macro_label

    # 平滑差值，仅用于找区间
    diff_smooth = pd.Series(diff_raw).ewm(span=smooth_span, adjust=False).mean().to_numpy()

    # 用符号变化来分段：正偏离区间 / 负偏离区间
    sign = np.sign(diff_smooth)
    sign[np.abs(diff_smooth) < 1e-6] = 0

    raw_regions = []
    start = 0
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1]:
            raw_regions.append([start, i - 1])
            start = i
    raw_regions.append([start, len(sign) - 1])

    # 过滤短区间 + 太弱区间
    filtered = []
    for s, e in raw_regions:
        if e - s + 1 < min_region_len:
            continue
        seg = diff_smooth[s:e + 1]
        mean_abs = float(np.mean(np.abs(seg)))
        if mean_abs < 0.05:
            continue
        filtered.append([s, e])

    # 合并相近区间
    merged = []
    for seg in filtered:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] - prev[1] <= merge_gap:
                prev[1] = seg[1]
            else:
                merged.append(seg)

    # 评分
    regions = []
    for s, e in merged:
        seg_raw = diff_raw[s:e + 1]
        seg_smooth = diff_smooth[s:e + 1]

        mean_diff = float(np.mean(seg_raw))
        mean_abs_diff = float(np.mean(np.abs(seg_raw)))
        max_abs_diff = float(np.max(np.abs(seg_raw)))
        mean_abs_smooth = float(np.mean(np.abs(seg_smooth)))
        length = int(e - s + 1)

        # 分数：既看幅度，也看持续时间
        score = mean_abs_smooth * (length ** score_power)

        regions.append({
            "start": int(s),
            "end": int(e),
            "length": length,
            "mean_diff": mean_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff_smooth": mean_abs_smooth,
            "score": float(score),
            "type": "micro_gt_macro" if mean_diff >= 0 else "micro_lt_macro",
        })

    regions = sorted(regions, key=lambda x: x["score"], reverse=True)[:top_k]
    return diff_raw, diff_smooth, regions
def plot_macro_with_micro_trigger_regions(
        dates,
        price,
        macro_label,
        micro_label,
        title="Macro Label with Top Micro Divergence Regions",
        outpath=None,
        top_k=6,
        smooth_span=7,
        min_region_len=5,
        merge_gap=5
):
    """
    主图不叠加完整 micro 曲线，只高亮 micro 与 macro 差距最大的若干区间。
    适合论文主图。
    """
    price = np.asarray(price, dtype=np.float32)
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    diff_raw, diff_smooth, regions = select_top_micro_macro_divergence_regions(
        macro_label=macro_label,
        micro_label=micro_label,
        top_k=top_k,
        smooth_span=smooth_span,
        min_region_len=min_region_len,
        merge_gap=merge_gap
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.4, 1.0]}
    )

    # 1) Price
    ax = axes[0]
    ax.plot(dates, price, color="tab:gray", lw=1.2, alpha=0.95, label="Price")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    # 2) Macro label only
    ax = axes[1]
    ax.plot(dates, macro_label, color="tab:blue", lw=2.0, label="Macro Soft Label")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.35)
    ax.axhline(0.8, color="red", linestyle=":", alpha=0.20)
    ax.axhline(0.2, color="green", linestyle=":", alpha=0.20)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Macro Label")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    # 3) 只画差值，不画 micro 全曲线
    ax = axes[2]
    ax.plot(dates, diff_smooth, color="tab:purple", lw=1.5, label="Smoothed Diff (Micro - Macro)")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.set_ylabel("Micro-Macro")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    # 高亮 top-k 区间
    for i, reg in enumerate(regions, 1):
        s, e = reg["start"], reg["end"]
        color = "red" if reg["type"] == "micro_gt_macro" else "green"

        for a in axes:
            a.axvspan(dates[s], dates[e], color=color, alpha=0.12)

        mid = (s + e) // 2
        axes[2].text(
            dates[mid],
            diff_smooth[mid],
            f"Top{i}",
            fontsize=8,
            alpha=0.85
        )

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")
        print(f"Plot saved to {outpath}")
    else:
        plt.show()
    plt.close()

    return {
        "diff_raw": diff_raw,
        "diff_smooth": diff_smooth,
        "regions": regions
    }
def plot_top_micro_trigger_windows(
        dates,
        price,
        macro_label,
        micro_label,
        regions,
        out_dir,
        prefix="case",
        window_pad=20,
        micro_vis_span=5
):
    """
    对 top-k 显著区间画局部放大图。
    这里可以画 micro，因为局部窗口里可读性是足够的。
    """
    os.makedirs(out_dir, exist_ok=True)

    price = np.asarray(price, dtype=np.float32)
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)
    micro_vis = pd.Series(micro_label).ewm(span=micro_vis_span, adjust=False).mean().to_numpy()

    for i, reg in enumerate(regions, 1):
        s0, e0 = reg["start"], reg["end"]
        s = max(0, s0 - window_pad)
        e = min(len(price) - 1, e0 + window_pad)

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True,
            gridspec_kw={"height_ratios": [1.8, 1.2]}
        )

        axes[0].plot(dates[s:e + 1], price[s:e + 1], color="tab:gray", lw=1.2, label="Price")
        axes[0].axvspan(dates[s0], dates[e0], color="gold", alpha=0.18)
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="upper left")
        axes[0].set_title(
            f"{prefix} | Top-{i} | {reg['type']} | len={reg['length']} | score={reg['score']:.3f}"
        )

        axes[1].plot(dates[s:e + 1], macro_label[s:e + 1], lw=2.0, label="Macro Soft Label")
        axes[1].plot(dates[s:e + 1], micro_vis[s:e + 1], lw=1.8, label=f"Micro Label (vis span={micro_vis_span})")
        axes[1].axvspan(dates[s0], dates[e0], color="gold", alpha=0.18)
        axes[1].axhline(0.5, color="black", linestyle="--", alpha=0.35)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="upper left")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{prefix}_top{i}_trigger_window.png")
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved local trigger plot to {save_path}")

def select_top_micro_macro_points(
        macro_label,
        micro_label,
        dates=None,
        top_k=8,
        min_gap=15,
        min_abs_diff=0.10,
        start_idx=None,
        end_idx=None
):
    """
    只根据 |micro - macro| 挑选差异最大的若干点，不做任何平滑。

    参数
    ----
    macro_label : array-like
    micro_label : array-like
    dates       : 可选，仅用于结果输出
    top_k       : 选几个点
    min_gap     : 两个触发点之间至少间隔多少个时间步，避免挤在一起
    min_abs_diff: 至少达到这个差值才保留
    start_idx   : 可选，只在 [start_idx, end_idx] 区间内选点
    end_idx     : 可选，只在 [start_idx, end_idx] 区间内选点

    返回
    ----
    diff : np.ndarray
        micro - macro
    selected : list[dict]
        每个点：
        {
            "idx": int,
            "date": ...,
            "macro": float,
            "micro": float,
            "diff": float,
            "abs_diff": float,
            "direction": "up" / "down"
        }
    """
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    if len(macro_label) != len(micro_label):
        raise ValueError("macro_label 和 micro_label 长度必须一致")

    diff = micro_label - macro_label
    abs_diff = np.abs(diff)
    n = len(diff)

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1

    start_idx = max(0, int(start_idx))
    end_idx = min(n - 1, int(end_idx))

    candidates = []
    for i in range(start_idx, end_idx + 1):
        if abs_diff[i] < min_abs_diff:
            continue
        candidates.append({
            "idx": i,
            "date": str(dates[i]) if dates is not None else int(i),
            "macro": float(macro_label[i]),
            "micro": float(micro_label[i]),
            "diff": float(diff[i]),
            "abs_diff": float(abs_diff[i]),
            "direction": "up" if diff[i] > 0 else "down"
        })

    # 按差异绝对值排序
    candidates = sorted(candidates, key=lambda x: x["abs_diff"], reverse=True)

    # 稀疏化，避免点过近
    selected = []
    used_idx = []
    for cand in candidates:
        idx = cand["idx"]
        if all(abs(idx - j) >= min_gap for j in used_idx):
            selected.append(cand)
            used_idx.append(idx)
        if len(selected) >= top_k:
            break

    # 最终按时间排序，便于画图
    selected = sorted(selected, key=lambda x: x["idx"])
    return diff, selected
def plot_price_with_micro_macro_points(
        dates,
        price,
        macro_label,
        micro_label,
        title="Price with Top |Micro-Macro| Points",
        outpath=None,
        top_k=8,
        min_gap=15,
        min_abs_diff=0.10,
        start_idx=None,
        end_idx=None,
        plot_macro_panel=True
):
    """
    在价格图上用箭头标记 micro 与 macro 差异最大的点。
    不在箭头旁标数字。
    """
    price = np.asarray(price, dtype=np.float32)
    macro_label = np.asarray(macro_label, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    n = len(price)
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1

    start_idx = max(0, int(start_idx))
    end_idx = min(n - 1, int(end_idx))

    diff, selected = select_top_micro_macro_points(
        macro_label=macro_label,
        micro_label=micro_label,
        dates=dates,
        top_k=top_k,
        min_gap=min_gap,
        min_abs_diff=min_abs_diff,
        start_idx=start_idx,
        end_idx=end_idx
    )

    x = dates[start_idx:end_idx + 1]
    p = price[start_idx:end_idx + 1]
    m = macro_label[start_idx:end_idx + 1]

    if plot_macro_panel:
        fig, axes = plt.subplots(
            2, 1, figsize=(16, 9.5), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08}
        )
        ax_price = axes[0]
        ax_macro = axes[1]
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=(16, 6.5))
        ax_macro = None

    # ---------------------------
    # 上图：价格 + 箭头
    # ---------------------------
    _plot_price_pretty(ax_price, x, p, label="Price")
    ax_price.set_title(title, pad=12)
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")

    prange = float(np.nanmax(p) - np.nanmin(p) + 1e-8)
    arrow_offset = 0.07 * prange

    arrow_style_up = dict(
        arrowstyle="-|>",
        lw=2.2,
        color="#d62728",
        mutation_scale=17,
        shrinkA=0,
        shrinkB=0
    )
    arrow_style_down = dict(
        arrowstyle="-|>",
        lw=2.2,
        color="#1f77b4",
        mutation_scale=17,
        shrinkA=0,
        shrinkB=0
    )

    for pt in selected:
        i = pt["idx"]
        if not (start_idx <= i <= end_idx):
            continue

        x0 = dates[i]
        y0 = price[i]

        ax_price.axvline(x=x0, color="#6c757d", linestyle="--", alpha=0.16, lw=0.9)
        if ax_macro is not None:
            ax_macro.axvline(x=x0, color="#6c757d", linestyle="--", alpha=0.16, lw=0.9)

        if pt["direction"] == "up":
            ax_price.annotate(
                "",
                xy=(x0, y0),
                xytext=(x0, y0 - arrow_offset),
                arrowprops=arrow_style_up
            )
            ax_price.scatter([x0], [y0], s=24, color="#d62728", zorder=5)
        else:
            ax_price.annotate(
                "",
                xy=(x0, y0),
                xytext=(x0, y0 + arrow_offset),
                arrowprops=arrow_style_down
            )
            ax_price.scatter([x0], [y0], s=24, color="#1f77b4", zorder=5)

    # ---------------------------
    # 下图：macro soft label
    # ---------------------------
    if ax_macro is not None:
        ax_macro.plot(
            x, m,
            color="#4c72b0",
            lw=2.3,
            label="Macro Soft Label",
            zorder=3
        )
        ax_macro.fill_between(
            x, 0.5, m,
            where=(m >= 0.5),
            color="#c6dbef",
            alpha=0.42,
            interpolate=True,
            zorder=1
        )
        ax_macro.fill_between(
            x, 0.5, m,
            where=(m < 0.5),
            color="#fdd0a2",
            alpha=0.42,
            interpolate=True,
            zorder=1
        )
        ax_macro.axhline(0.5, color="#7f7f7f", linestyle="--", lw=1.1, alpha=0.75)
        ax_macro.set_ylim(-0.05, 1.05)
        ax_macro.set_ylabel("Macro")
        ax_macro.set_xlabel("Date")
        _style_axis(ax_macro, hide_top_right=True)
        ax_macro.legend(loc="upper left")

        for pt in selected:
            i = pt["idx"]
            if not (start_idx <= i <= end_idx):
                continue

            x0 = dates[i]
            y0 = macro_label[i]

            if pt["direction"] == "up":
                ax_macro.scatter([x0], [y0], marker="^", s=68, color="#d62728", zorder=5)
            else:
                ax_macro.scatter([x0], [y0], marker="v", s=68, color="#1f77b4", zorder=5)

    plt.tight_layout(pad=1.8)
    if outpath:
        plt.savefig(outpath, dpi=220)
        print(f"Plot saved to {outpath}")
    else:
        plt.show()
    plt.close()

    return {
        "diff": diff,
        "selected_points": selected
    }
def resolve_plot_window(df, mode="tail", value=300, start_date=None, end_date=None):
    """
    返回一个绘图区间 [start_idx, end_idx]

    mode:
    - "tail" : 取最后 value 个点
    - "range": 按日期范围截取
    - "full" : 全部
    """
    n = len(df)

    if mode == "full":
        return 0, n - 1

    if mode == "tail":
        end_idx = n - 1
        start_idx = max(0, n - int(value))
        return start_idx, end_idx

    if mode == "range":
        if start_date is None or end_date is None:
            raise ValueError("mode='range' 时必须提供 start_date 和 end_date")
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError("给定日期区间内没有数据")
        return int(idx[0]), int(idx[-1])

    raise ValueError("mode must be one of ['full', 'tail', 'range']")
def plot_micro_label_with_points(
        dates,
        price,
        micro_label,
        selected_points,
        title="Micro Label with Selected Trigger Points",
        outpath=None,
        start_idx=None,
        end_idx=None
):
    """
    画 micro label 图，并把选中的 top 差异点标出来。
    """
    price = np.asarray(price, dtype=np.float32)
    micro_label = np.asarray(micro_label, dtype=np.float32)

    n = len(price)
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1

    start_idx = max(0, int(start_idx))
    end_idx = min(n - 1, int(end_idx))

    x = dates[start_idx:end_idx + 1]
    p = price[start_idx:end_idx + 1]
    micro = micro_label[start_idx:end_idx + 1]

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08}
    )

    # 上图：价格
    ax1 = axes[0]
    _plot_price_pretty(ax1, x, p, label="Price")
    ax1.set_title(title, pad=12)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")

    # 下图：micro label
    ax2 = axes[1]
    ax2.plot(
        x, micro,
        color="#dd8452",
        lw=2.0,
        alpha=0.97,
        label="Micro Label",
        zorder=3
    )
    ax2.fill_between(
        x, 0.5, micro,
        where=(micro >= 0.5),
        color="#fee6ce",
        alpha=0.48,
        interpolate=True,
        zorder=1
    )
    ax2.fill_between(
        x, 0.5, micro,
        where=(micro < 0.5),
        color="#d9f0d3",
        alpha=0.35,
        interpolate=True,
        zorder=1
    )
    ax2.axhline(0.5, color="#7f7f7f", linestyle="--", lw=1.1, alpha=0.75)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Micro")
    ax2.set_xlabel("Date")
    _style_axis(ax2, hide_top_right=True)
    ax2.legend(loc="upper left")

    for pt in selected_points:
        i = pt["idx"]
        if not (start_idx <= i <= end_idx):
            continue

        x0 = dates[i]
        y_micro = micro_label[i]

        ax1.axvline(x=x0, color="#6c757d", linestyle="--", alpha=0.16, lw=0.9)
        ax2.axvline(x=x0, color="#6c757d", linestyle="--", alpha=0.16, lw=0.9)

        if pt["direction"] == "up":
            ax2.scatter([x0], [y_micro], marker="^", s=68, color="#d62728", zorder=5)
        else:
            ax2.scatter([x0], [y_micro], marker="v", s=68, color="#1f77b4", zorder=5)

    plt.tight_layout(pad=1.8)
    if outpath:
        plt.savefig(outpath, dpi=220)
        print(f"Plot saved to {outpath}")
    else:
        plt.show()
    plt.close()
# ----------------- Main Demo -----------------
if __name__ == "__main__":
    set_plot_style()
    # 1. 尝试读取你的数据
    # 这里为了演示方便，如果文件不存在，我生成一个合成数据
    # csv_path = "Dataset/Nas100数据/feature/AAPL.O.csv"
    csv_path = "Dataset/沪深数据/feature/000001.SZ.csv"

    if os.path.exists(csv_path):
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
        # name = "AAPL.O"
        name = "000001.SZ"

    # 2. 生成更平滑的软标签
    soft_label, std_label = build_smooth_ensemble_label(
        df,
        spans=[i for i in range(10, 30, 5)],  # 覆盖从周线到季线的趋势
        slope_tols=[0.0002, 0.0005, 0.001],  # 覆盖对“平盘”的不同定义
        min_lens=[5, 10, 15, 21],  # 覆盖不同的抗噪级别
        sensitivity=3.0,  # Sigmoid 较平缓
        final_smooth_sigma=3.0  # 最后加一个 sigma=3 的高斯模糊
    )

    # 3. 保存结果
    # out_dir = "nas_ensemble_labels"
    out_dir = "wind_ensemble_labels"
    os.makedirs(out_dir, exist_ok=True)

    df["soft_label"] = soft_label
    df["label_std"] = std_label
    out_csv = os.path.join(out_dir, f"{name}_soft_label.csv")
    df.to_csv(out_csv)
    print(f"Saved soft labels to {out_csv}")

    # 4. 画图查看效果
    out_png = os.path.join(out_dir, f"{name}_ensemble_plot.png")
    plot_price_vs_prob(
        df.index,
        df["adjclose"],
        soft_label,
        title=f"{name} - Ensemble Soft Label (Prob)",
        outpath=out_png
    )
    labels_micro = build_fullseries_prob_labels(
        df,
        price_col="adjclose",
        smooth_span=3,  # 短周期
        slope_tol=0.0002,  # 高敏感
        min_len=3,
        sensitivity=1.5
    )
    # 5. 保存 micro label
    df["label_micro"] = labels_micro

    # 6. 选择一个更清楚的时间窗口
    # 方案A：最后300个交易日
    start_idx, end_idx = resolve_plot_window(df, mode="tail", value=600)

    # 方案B：按日期手动指定
    # start_idx, end_idx = resolve_plot_window(
    #     df, mode="range",
    #     start_date="2018-01-01",
    #     end_date="2020-12-31"
    # )

    # 7. 只在该窗口里找 top-k 差异点，并在价格图上标箭头
    out_png_arrow = os.path.join(out_dir, f"{name}_price_macro_top_points.png")
    point_res = plot_price_with_micro_macro_points(
        dates=df.index,
        price=df["adjclose"].to_numpy(),
        macro_label=soft_label,
        micro_label=labels_micro,
        title=f"{name} - Top |Micro - Macro| Points",
        outpath=out_png_arrow,
        top_k=20,
        min_gap=15,
        min_abs_diff=0.12,
        start_idx=start_idx,
        end_idx=end_idx,
        plot_macro_panel=True
    )

    # 8. 保存这些点
    point_df = pd.DataFrame(point_res["selected_points"])
    if len(point_df) > 0:
        out_point_csv = os.path.join(out_dir, f"{name}_top_micro_macro_points.csv")
        point_df.to_csv(out_point_csv, index=False, encoding="utf-8-sig")
        print(f"Saved top point table to {out_point_csv}")

    # 8. 画 micro label 图
    out_png_micro = os.path.join(out_dir, f"{name}_micro_label_top_points.png")
    plot_micro_label_with_points(
        dates=df.index,
        price=df["adjclose"].to_numpy(),
        micro_label=labels_micro,
        selected_points=point_res["selected_points"],
        title=f"{name} - Micro Label with Top |Micro - Macro| Points",
        outpath=out_png_micro,
        start_idx=start_idx,
        end_idx=end_idx
    )

    # 9. 更新总输出
    out_csv = os.path.join(out_dir, f"{name}_soft_macro_micro_label.csv")
    df.to_csv(out_csv)
    print(f"Saved macro/micro labels to {out_csv}")