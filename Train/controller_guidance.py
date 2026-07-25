from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GuidanceLabels:
    labels: torch.Tensor
    risk_percentile: torch.Tensor
    advantage_percentile: torch.Tensor
    priority: torch.Tensor
    advantage_only_labels: torch.Tensor


@dataclass(frozen=True)
class EconomicGuidanceLabels:
    labels: torch.Tensor
    mask: torch.Tensor
    trigger: torch.Tensor


def build_economic_guidance_labels(
        risk: torch.Tensor,
        advantage: torch.Tensor,
        *,
        risk_threshold: float = 0.05,
        advantage_threshold: float = 0.05,
) -> EconomicGuidanceLabels:
    risk = risk.detach().view(-1).float()
    advantage = advantage.detach().view(-1).float()
    if risk.shape != advantage.shape:
        raise ValueError("risk and advantage must have identical shapes")

    valid = torch.isfinite(risk) & torch.isfinite(advantage)
    trigger = valid & (
        ((risk >= float(risk_threshold)) & (advantage > 0.0))
        | (advantage >= float(advantage_threshold))
    )
    labels = trigger.to(dtype=risk.dtype)
    mask = valid.to(dtype=risk.dtype)
    return EconomicGuidanceLabels(labels=labels, mask=mask, trigger=trigger)


def balanced_guidance_weights(
        labels: torch.Tensor,
        mask: torch.Tensor,
) -> torch.Tensor:
    labels = labels.detach().view(-1).float()
    mask = mask.detach().view(-1).float()
    if labels.shape != mask.shape:
        raise ValueError("labels and mask must have identical shapes")

    valid = torch.isfinite(labels) & torch.isfinite(mask) & (mask > 0.0)
    positive = valid & (labels > 0.5)
    negative = valid & ~positive
    weights = torch.zeros_like(labels)
    valid_count = int(torch.count_nonzero(valid).item())
    positive_count = int(torch.count_nonzero(positive).item())
    negative_count = int(torch.count_nonzero(negative).item())
    if positive_count and negative_count:
        weights[positive] = float(valid_count) / (2.0 * float(positive_count))
        weights[negative] = float(valid_count) / (2.0 * float(negative_count))
    elif positive_count:
        weights[positive] = 1.0
    elif negative_count:
        weights[negative] = 1.0
    return weights


def _stable_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().view(-1)
    result = torch.full_like(values, float("nan"), dtype=torch.float32)
    valid_indices = torch.nonzero(torch.isfinite(values), as_tuple=False).view(-1)
    if valid_indices.numel() == 0:
        return result

    valid_values = values[valid_indices]
    order = torch.argsort(valid_values, stable=True)
    if valid_indices.numel() == 1:
        percentiles = torch.ones(1, device=values.device, dtype=torch.float32)
    else:
        percentiles = torch.linspace(
            0.0,
            1.0,
            steps=valid_indices.numel(),
            device=values.device,
            dtype=torch.float32,
        )
    result[valid_indices[order]] = percentiles
    return result


def _stable_topk_mask(
        scores: torch.Tensor,
        eligible: torch.Tensor,
        *,
        topk: int,
) -> torch.Tensor:
    mask = torch.zeros(scores.numel(), device=scores.device, dtype=torch.bool)
    if topk <= 0:
        return mask

    eligible_indices = torch.nonzero(eligible, as_tuple=False).view(-1)
    if eligible_indices.numel() == 0:
        return mask

    eligible_scores = scores[eligible_indices]
    order = torch.argsort(eligible_scores, descending=True, stable=True)
    selected = eligible_indices[order[:min(int(topk), int(order.numel()))]]
    mask[selected] = True
    return mask


def build_topk_guidance_labels(
        risk: torch.Tensor,
        advantage: torch.Tensor,
        *,
        topk: int = 20,
) -> GuidanceLabels:
    risk = risk.detach().view(-1).float()
    advantage = advantage.detach().view(-1).float()
    if risk.shape != advantage.shape:
        raise ValueError("risk and advantage must have identical shapes")

    risk_percentile = _stable_percentile_rank(risk)
    advantage_percentile = _stable_percentile_rank(advantage)
    priority = torch.maximum(risk_percentile, advantage_percentile)
    eligible = torch.isfinite(risk) & torch.isfinite(advantage) & (advantage > 0.0)
    labels = _stable_topk_mask(priority, eligible, topk=topk).float()
    advantage_only_labels = _stable_topk_mask(
        advantage_percentile,
        eligible,
        topk=topk,
    ).float()
    return GuidanceLabels(
        labels=labels,
        risk_percentile=risk_percentile,
        advantage_percentile=advantage_percentile,
        priority=priority,
        advantage_only_labels=advantage_only_labels,
    )


def balanced_guidance_bce(
        logits: torch.Tensor,
        labels: torch.Tensor,
) -> torch.Tensor:
    logits = logits.view(-1)
    labels = labels.to(device=logits.device, dtype=logits.dtype).view(-1)
    if logits.shape != labels.shape or logits.numel() == 0:
        raise ValueError("logits and labels must be non-empty with identical shapes")

    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    class_means = []
    positive = labels > 0.5
    negative = ~positive
    if torch.any(positive):
        class_means.append(raw[positive].mean())
    if torch.any(negative):
        class_means.append(raw[negative].mean())
    return torch.stack(class_means).mean()


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def analyze_guidance_windows(
        windows: Iterable[Dict],
        *,
        topk: int = 20,
) -> Tuple[List[Dict], Dict]:
    detail_rows = []
    per_window_label_counts = []
    zero_positive_windows = 0
    selected_advantages = []
    hold_advantages = []
    selected_risk_percentiles = []
    selected_advantage_percentiles = []
    selected_gaps = []
    overlap_count = 0
    selected_count = 0
    risk_dominant = 0
    advantage_dominant = 0
    tied_dominant = 0

    window_list = list(windows)
    for fallback_id, window in enumerate(window_list):
        risk = torch.as_tensor(window["risk"], dtype=torch.float32).view(-1)
        advantage = torch.as_tensor(window["advantage"], dtype=torch.float32).view(-1)
        labels = build_topk_guidance_labels(risk, advantage, topk=topk)
        window_id = int(window.get("window_id", fallback_id))
        start_index = int(window.get("start_index", 0))

        positive_advantage_count = int(torch.count_nonzero(
            torch.isfinite(advantage) & (advantage > 0.0)
        ).item())
        if positive_advantage_count == 0:
            zero_positive_windows += 1

        label_count = int(labels.labels.sum().item())
        per_window_label_counts.append(label_count)
        selected_positions = torch.nonzero(labels.labels > 0.5, as_tuple=False).view(-1).tolist()
        selected_gaps.extend(
            int(right - left)
            for left, right in zip(selected_positions, selected_positions[1:])
        )

        for decision_order in range(int(risk.numel())):
            label = int(labels.labels[decision_order].item() > 0.5)
            advantage_only_label = int(
                labels.advantage_only_labels[decision_order].item() > 0.5
            )
            risk_pct = float(labels.risk_percentile[decision_order].item())
            advantage_pct = float(labels.advantage_percentile[decision_order].item())
            advantage_value = float(advantage[decision_order].item())
            risk_value = float(risk[decision_order].item())
            detail_rows.append({
                "split": "train",
                "window_id": window_id,
                "start_index": start_index,
                "decision_order": decision_order,
                "risk": risk_value,
                "advantage": advantage_value,
                "risk_percentile": risk_pct,
                "advantage_percentile": advantage_pct,
                "priority": float(labels.priority[decision_order].item()),
                "label": label,
                "advantage_only_label": advantage_only_label,
            })
            if label:
                selected_count += 1
                overlap_count += advantage_only_label
                selected_advantages.append(advantage_value)
                selected_risk_percentiles.append(risk_pct)
                selected_advantage_percentiles.append(advantage_pct)
                if risk_pct > advantage_pct:
                    risk_dominant += 1
                elif advantage_pct > risk_pct:
                    advantage_dominant += 1
                else:
                    tied_dominant += 1
            else:
                hold_advantages.append(advantage_value)

    all_labels = torch.tensor(
        [float(row["label"]) for row in detail_rows],
        dtype=torch.float32,
    )
    if all_labels.numel() > 0:
        zero_logits = torch.zeros_like(all_labels)
        zero_logit_loss = float(
            balanced_guidance_bce(zero_logits, all_labels).item()
        )
        positive_rate = float(all_labels.mean().item())
        clipped_rate = min(max(positive_rate, 1e-6), 1.0 - 1e-6)
        prior_logit = math.log(clipped_rate / (1.0 - clipped_rate))
        prior_logits = torch.full_like(all_labels, prior_logit)
        prior_loss = float(
            balanced_guidance_bce(prior_logits, all_labels).item()
        )
    else:
        positive_rate = 0.0
        zero_logit_loss = 0.0
        prior_loss = 0.0

    summary = {
        "split": "train",
        "window_count": len(window_list),
        "decision_count": len(detail_rows),
        "positive_advantage_count": sum(
            int(math.isfinite(row["advantage"]) and row["advantage"] > 0.0)
            for row in detail_rows
        ),
        "switch_label_count": selected_count,
        "switch_label_rate": positive_rate,
        "zero_positive_window_count": zero_positive_windows,
        "max_switch_labels_per_window": max(per_window_label_counts, default=0),
        "mean_switch_labels_per_window": _safe_mean(per_window_label_counts),
        "selected_advantage_mean": _safe_mean(selected_advantages),
        "selected_advantage_median": _safe_median(selected_advantages),
        "hold_advantage_mean": _safe_mean(hold_advantages),
        "hold_advantage_median": _safe_median(hold_advantages),
        "selected_risk_percentile_mean": _safe_mean(selected_risk_percentiles),
        "selected_advantage_percentile_mean": _safe_mean(
            selected_advantage_percentiles
        ),
        "risk_dominant_count": risk_dominant,
        "advantage_dominant_count": advantage_dominant,
        "tied_dominant_count": tied_dominant,
        "selected_gap_mean": _safe_mean(selected_gaps),
        "selected_gap_median": _safe_median(selected_gaps),
        "selected_adjacent_gap_rate": (
            sum(gap == 1 for gap in selected_gaps) / len(selected_gaps)
            if selected_gaps else 0.0
        ),
        "label_overlap_count": overlap_count,
        "label_overlap_rate": (
            float(overlap_count / selected_count) if selected_count else 0.0
        ),
        "balanced_bce_zero_logit": zero_logit_loss,
        "balanced_bce_empirical_prior": prior_loss,
    }
    return detail_rows, summary


def render_guidance_report(
        summary: Dict,
        *,
        topk: int,
        rollout_len: int,
) -> str:
    return f"""# Controller Top-{int(topk)} 标签离线检测

## 检测范围

- 数据范围：仅使用训练集（train split），不读取验证集或测试集。
- Rollout 长度：{int(rollout_len)} 个交易日。
- 标签规则：风险与成本调整优势分别做窗口内百分位排名，取两者最大值；
  仅从 advantage 大于 0 的日期中选择 Top-{int(topk)}。

## 标签分布

| 指标 | 数值 |
|---|---:|
| 历史窗口数 | {int(summary['window_count'])} |
| 有效决策数 | {int(summary['decision_count'])} |
| 正优势日期数 | {int(summary['positive_advantage_count'])} |
| Switch 标签数 | {int(summary['switch_label_count'])} |
| Switch 标签比例 | {100.0 * float(summary['switch_label_rate']):.2f}% |
| 单窗口最大 Switch 标签数 | {int(summary['max_switch_labels_per_window'])} |
| 无正优势窗口数 | {int(summary['zero_positive_window_count'])} |

## 信号强度

| 指标 | 数值 |
|---|---:|
| Switch 平均 advantage | {100.0 * float(summary['selected_advantage_mean']):.4f}% |
| Switch 中位 advantage | {100.0 * float(summary['selected_advantage_median']):.4f}% |
| Hold 平均 advantage | {100.0 * float(summary['hold_advantage_mean']):.4f}% |
| Switch 平均风险百分位 | {float(summary['selected_risk_percentile_mean']):.4f} |
| Switch 平均优势百分位 | {float(summary['selected_advantage_percentile_mean']):.4f} |
| 风险主导标签数 | {int(summary['risk_dominant_count'])} |
| 优势主导标签数 | {int(summary['advantage_dominant_count'])} |
| 双信号并列标签数 | {int(summary['tied_dominant_count'])} |

## 聚集与对照

| 指标 | 数值 |
|---|---:|
| 相邻 Switch 标签平均间隔 | {float(summary['selected_gap_mean']):.2f} 个决策 |
| 相邻 Switch 标签中位间隔 | {float(summary['selected_gap_median']):.2f} 个决策 |
| 连续日期标签比例（间隔=1） | {100.0 * float(summary['selected_adjacent_gap_rate']):.2f}% |
| 与 advantage-only Top-{int(topk)} 重合率 | {100.0 * float(summary['label_overlap_rate']):.2f}% |

## 类别平衡 BCE 基线

| 常数预测 | Loss |
|---|---:|
| logit=0（概率0.5） | {float(summary['balanced_bce_zero_logit']):.6f} |
| 预测原始正类比例 | {float(summary['balanced_bce_empirical_prior']):.6f} |

类别平衡 BCE 对 Switch 与 Hold 两类分别求均值后再等权平均，因此正样本较少
不会使最终融合层退化为全 Hold。该报告只检查标签结构；是否提升策略收益仍需
在独立验证集和测试集比较 Outer-only 与 Outer + Controller。
"""
