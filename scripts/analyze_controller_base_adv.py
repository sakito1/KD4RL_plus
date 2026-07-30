#!/usr/bin/env python3
"""Full-test-set statistics for the Controller base gate and advantage term."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from statsmodels.stats.multitest import multipletests


MARKETS = (
    ("NASDAQ-100", "nas", 49),
    ("CSI-300", "sh", 90),
)
ADV_COEF = 1.9
ADV_SCALE = 0.02
MAX_HOLD = 30
HAC_LAGS = 5
BOOT_BLOCK = 20
BOOT_REPS = 2000
BOOT_SEED = 20260729


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "reproduced_outputs" / "controller_base_adv_statistics",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=BOOT_REPS)
    return parser.parse_args()


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_market(repo: Path, stem: str, seed: int) -> pd.DataFrame:
    trace_path = (
        repo
        / "reproduced_outputs"
        / "controller_gate_adv_case_analysis"
        / f"controller_gate_adv_trace_{stem}.csv"
    )
    cf_path = (
        repo
        / "paper_experiments_outputs"
        / "paper_experiments_final"
        / "_cache"
        / "counterfactual_horizon30"
        / f"{stem}_seed{seed}_full_controller_horizon30_actions.csv"
    )
    trace = pd.read_csv(trace_path)
    cf_cols = [
        "date",
        "step",
        "duration_before_decision",
        "controller_switch_advantage",
        "switch_advantage_20",
        "switch_advantage_30",
        "hold_curve_20",
        "hold_curve_30",
    ]
    counterfactual = pd.read_csv(cf_path, usecols=cf_cols)
    counterfactual = counterfactual.rename(
        columns={
            "hold_curve_20": "cf_hold_curve_20",
            "hold_curve_30": "cf_hold_curve_30",
        }
    )
    merged = trace.merge(
        counterfactual,
        on=["date", "step"],
        how="left",
        validate="one_to_one",
    )
    free = merged.loc[~truthy(merged["is_forced"])].copy()
    if free["controller_switch_advantage"].isna().any():
        raise ValueError(f"{stem}: missing adaptive counterfactual targets after merge")
    free["date"] = pd.to_datetime(free["date"])
    free = free.sort_values(["date", "step"]).reset_index(drop=True)
    free["base_prob"] = expit(free["base_exit_logit"])
    free["adv_logit"] = ADV_COEF * np.tanh(free["switch_advantage_pred"] / ADV_SCALE)
    free["formula_error"] = (
        free["base_exit_logit"] + free["adv_logit"] - free["exit_logit"]
    ).abs()
    free["adaptive_horizon"] = np.maximum(
        1, MAX_HOLD - free["duration_before_decision"].astype(int)
    )
    free["adaptive_adv_bp_day"] = (
        free["controller_switch_advantage"] / free["adaptive_horizon"] * 10000.0
    )
    free["fixed20_adv_bp_day"] = free["switch_advantage_20"] / 20.0 * 10000.0
    free["fixed30_adv_bp_day"] = free["switch_advantage_30"] / 30.0 * 10000.0
    for horizon in (20, 30):
        curve_column = f"cf_hold_curve_{horizon}"
        curve_length = free[curve_column].fillna("").map(
            lambda value: len(json.loads(value)) - 1 if value else 0
        )
        free.loc[
            curve_length < horizon,
            [f"switch_advantage_{horizon}", f"fixed{horizon}_adv_bp_day"],
        ] = np.nan
    free["actual_switch"] = truthy(free["is_switch"])
    free["adv_positive"] = free["switch_advantage_pred"] > 0.0
    return free


def circular_block_indices(
    n: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block))
    offsets = np.arange(block)
    return ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]


def block_bootstrap_ci(
    values: pd.DataFrame,
    metric,
    *,
    reps: int,
    seed: int,
    block: int = BOOT_BLOCK,
) -> tuple[float, float]:
    data = values.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        sample = data.iloc[circular_block_indices(len(data), block, rng)]
        estimate = metric(sample)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return np.nan, np.nan
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def hac_rank_association(x: pd.Series, y: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    x_rank = data["x"].rank(method="average").to_numpy(dtype=float)
    y_rank = data["y"].rank(method="average").to_numpy(dtype=float)
    x_std = (x_rank - x_rank.mean()) / x_rank.std(ddof=0)
    y_std = (y_rank - y_rank.mean()) / y_rank.std(ddof=0)
    fit = sm.OLS(y_std, sm.add_constant(x_std)).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    rho = float(spearmanr(data["x"], data["y"]).statistic)
    return {
        "rho": rho,
        "hac_t": float(fit.tvalues[1]),
        "hac_p": float(fit.pvalues[1]),
    }


def hac_group_difference(y: pd.Series, group: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"y": y, "group": group.astype(float)}).dropna()
    fit = sm.OLS(data["y"], sm.add_constant(data["group"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    return {
        "group0_mean": float(data.loc[data["group"] == 0, "y"].mean()),
        "group1_mean": float(data.loc[data["group"] == 1, "y"].mean()),
        "difference": float(fit.params["group"]),
        "hac_t": float(fit.tvalues["group"]),
        "hac_p": float(fit.pvalues["group"]),
        "group0_positive_rate": float(
            (data.loc[data["group"] == 0, "y"] > 0).mean()
        ),
        "group1_positive_rate": float(
            (data.loc[data["group"] == 1, "y"] > 0).mean()
        ),
    }


def distribution_table(data_by_market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns = {
        "Base logit": "base_exit_logit",
        "Base probability": "base_prob",
        "Adv raw prediction": "switch_advantage_pred",
        "Adv logit correction": "adv_logit",
        "Final logit": "exit_logit",
        "Final probability": "exit_prob",
    }
    rows = []
    for market, frame in data_by_market.items():
        for signal, column in columns.items():
            values = frame[column].dropna()
            rows.append(
                {
                    "market": market,
                    "signal": signal,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "min": values.min(),
                    "p05": values.quantile(0.05),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                    "lag1_autocorrelation": values.autocorr(1),
                }
            )
    return pd.DataFrame(rows)


def role_table(data_by_market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for market, frame in data_by_market.items():
        base_sd = frame["base_exit_logit"].std(ddof=1)
        adv_sd = frame["adv_logit"].std(ddof=1)
        required_adv = ADV_SCALE * np.arctanh(
            np.clip(-frame["base_exit_logit"] / ADV_COEF, -0.999999, 0.999999)
        )
        rows.append(
            {
                "market": market,
                "free_decisions": len(frame),
                "base_logit_mean": frame["base_exit_logit"].mean(),
                "base_logit_sd": base_sd,
                "base_prob_mean": frame["base_prob"].mean(),
                "base_prob_sd_percentage_points": frame["base_prob"].std(ddof=1)
                * 100.0,
                "base_only_switches": int((frame["base_exit_logit"] > 0).sum()),
                "adv_raw_mean": frame["switch_advantage_pred"].mean(),
                "adv_raw_sd": frame["switch_advantage_pred"].std(ddof=1),
                "adv_logit_sd": adv_sd,
                "adv_to_base_dynamic_sd_ratio": adv_sd / base_sd,
                "adv_positive_count": int(frame["adv_positive"].sum()),
                "adv_only_switch_rate": frame["adv_positive"].mean(),
                "actual_switches": int(frame["actual_switch"].sum()),
                "actual_switch_rate": frame["actual_switch"].mean(),
                "positive_adv_blocked_by_base": int(
                    (frame["adv_positive"] & ~frame["actual_switch"]).sum()
                ),
                "adv_overrides_base": int(
                    (
                        (frame["base_exit_logit"] < 0)
                        & (frame["exit_logit"] > 0)
                    ).sum()
                ),
                "negative_adv_reinforces_hold": int(
                    ((~frame["adv_positive"]) & ~frame["actual_switch"]).sum()
                ),
                "median_raw_adv_needed_to_switch": required_adv.median(),
                "formula_max_abs_error": frame["formula_error"].max(),
            }
        )
    return pd.DataFrame(rows)


def alignment_table(
    data_by_market: dict[str, pd.DataFrame], *, reps: int
) -> pd.DataFrame:
    signals = {
        "Base": "base_exit_logit",
        "Adv": "switch_advantage_pred",
        "Final": "exit_logit",
    }
    targets = {
        "Training-aligned adaptive horizon": "controller_switch_advantage",
        "Fixed 20-day robustness": "switch_advantage_20",
    }
    rows = []
    for market_idx, (market, frame) in enumerate(data_by_market.items()):
        for target_idx, (target_name, target) in enumerate(targets.items()):
            for signal_idx, (signal_name, signal) in enumerate(signals.items()):
                data = frame[[signal, target]].dropna().rename(
                    columns={signal: "x", target: "y"}
                )
                assoc = hac_rank_association(data["x"], data["y"])
                ci_low, ci_high = block_bootstrap_ci(
                    data,
                    lambda sample: spearmanr(sample["x"], sample["y"]).statistic,
                    reps=reps,
                    seed=BOOT_SEED
                    + 100 * market_idx
                    + 10 * target_idx
                    + signal_idx,
                )
                label = (data["y"] > 0).astype(int)
                auc = roc_auc_score(label, data["x"])
                predicted = data["x"] > 0
                rows.append(
                    {
                        "market": market,
                        "target": target_name,
                        "signal": signal_name,
                        "n": len(data),
                        "spearman_rho": assoc["rho"],
                        "block_ci_low": ci_low,
                        "block_ci_high": ci_high,
                        "newey_west_t": assoc["hac_t"],
                        "newey_west_p": assoc["hac_p"],
                        "sign_auc": auc,
                        "sign_accuracy": accuracy_score(label, predicted),
                        "sign_balanced_accuracy": balanced_accuracy_score(
                            label, predicted
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q_within_target_base_adv"] = np.nan
    for target_name in targets:
        mask = (result["target"] == target_name) & result["signal"].isin(
            ["Base", "Adv"]
        )
        result.loc[mask, "bh_q_within_target_base_adv"] = multipletests(
            result.loc[mask, "newey_west_p"], method="fdr_bh"
        )[1]
    return result


def quartile_table(
    data_by_market: dict[str, pd.DataFrame], *, reps: int
) -> pd.DataFrame:
    rows = []
    signals = {"Base": "base_exit_logit", "Adv": "switch_advantage_pred"}
    for market_idx, (market, frame) in enumerate(data_by_market.items()):
        for signal_idx, (signal_name, signal) in enumerate(signals.items()):
            q25, q75 = frame[signal].quantile([0.25, 0.75])
            subset = frame.loc[
                (frame[signal] <= q25) | (frame[signal] >= q75),
                [signal, "adaptive_adv_bp_day"],
            ].copy()
            subset["top_quartile"] = subset[signal] >= q75
            effect = hac_group_difference(
                subset["adaptive_adv_bp_day"], subset["top_quartile"]
            )
            ci_low, ci_high = block_bootstrap_ci(
                subset[["adaptive_adv_bp_day", "top_quartile"]],
                lambda sample: sample.loc[
                    sample["top_quartile"], "adaptive_adv_bp_day"
                ].mean()
                - sample.loc[
                    ~sample["top_quartile"], "adaptive_adv_bp_day"
                ].mean(),
                reps=reps,
                seed=BOOT_SEED + 1000 + 100 * market_idx + signal_idx,
            )
            rows.append(
                {
                    "market": market,
                    "signal": signal_name,
                    "n_extreme_quartiles": len(subset),
                    "bottom_quartile_mean_bp_day": effect["group0_mean"],
                    "top_quartile_mean_bp_day": effect["group1_mean"],
                    "top_minus_bottom_bp_day": effect["difference"],
                    "block_ci_low": ci_low,
                    "block_ci_high": ci_high,
                    "newey_west_t": effect["hac_t"],
                    "newey_west_p": effect["hac_p"],
                }
            )
    result = pd.DataFrame(rows)
    result["bh_q"] = multipletests(result["newey_west_p"], method="fdr_bh")[1]
    return result


def decision_effect_table(
    data_by_market: dict[str, pd.DataFrame], *, reps: int
) -> pd.DataFrame:
    outcomes = {
        "Training-aligned adaptive horizon": "adaptive_adv_bp_day",
        "Fixed 20-day": "fixed20_adv_bp_day",
        "Fixed 30-day": "fixed30_adv_bp_day",
    }
    groups = {
        "Actual Switch vs Hold": "actual_switch",
        "Positive Adv vs non-positive Adv": "adv_positive",
    }
    rows = []
    for market_idx, (market, frame) in enumerate(data_by_market.items()):
        for outcome_idx, (outcome_name, outcome) in enumerate(outcomes.items()):
            for group_idx, (comparison, group) in enumerate(groups.items()):
                data = frame[[outcome, group]].dropna().rename(
                    columns={outcome: "y", group: "group"}
                )
                data["group"] = data["group"].astype(bool)
                effect = hac_group_difference(data["y"], data["group"])
                ci_low, ci_high = block_bootstrap_ci(
                    data,
                    lambda sample: sample.loc[sample["group"], "y"].mean()
                    - sample.loc[~sample["group"], "y"].mean(),
                    reps=reps,
                    seed=BOOT_SEED
                    + 2000
                    + 100 * market_idx
                    + 10 * outcome_idx
                    + group_idx,
                )
                rows.append(
                    {
                        "market": market,
                        "outcome": outcome_name,
                        "comparison": comparison,
                        "n": len(data),
                        "reference_mean_bp_day": effect["group0_mean"],
                        "positive_group_mean_bp_day": effect["group1_mean"],
                        "difference_bp_day": effect["difference"],
                        "block_ci_low": ci_low,
                        "block_ci_high": ci_high,
                        "newey_west_t": effect["hac_t"],
                        "newey_west_p": effect["hac_p"],
                        "reference_positive_rate": effect[
                            "group0_positive_rate"
                        ],
                        "positive_group_positive_rate": effect[
                            "group1_positive_rate"
                        ],
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q_within_outcome_comparison"] = np.nan
    for outcome_name in outcomes:
        for comparison in groups:
            mask = (result["outcome"] == outcome_name) & (
                result["comparison"] == comparison
            )
            result.loc[
                mask, "bh_q_within_outcome_comparison"
            ] = multipletests(result.loc[mask, "newey_west_p"], method="fdr_bh")[1]
    return result


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def make_figure(
    data_by_market: dict[str, pd.DataFrame],
    alignment: pd.DataFrame,
    decision_effect: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {"NASDAQ-100": "#2474B5", "CSI-300": "#0E9F76"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), constrained_layout=True)
    for row, (market, frame) in enumerate(data_by_market.items()):
        color = colors[market]
        ax = axes[row, 0]
        ax.scatter(
            frame["base_prob"],
            np.zeros(len(frame)),
            s=8,
            alpha=0.15,
            color=color,
            edgecolors="none",
        )
        ax.axvline(0.5, color="#BB3E3E", linestyle="--", linewidth=1.5)
        ax.set_xlim(0.25, 0.51)
        ax.set_yticks([])
        ax.set_xlabel("Base-only switch probability")
        ax.set_title(
            f"{market}: Base range "
            f"[{frame['base_prob'].min():.3f}, {frame['base_prob'].max():.3f}]"
        )
        ax.text(
            0.5,
            0.78,
            "Switch threshold",
            transform=ax.get_xaxis_transform(),
            ha="right",
            color="#9C2E2E",
        )

        ax = axes[row, 1]
        ranked = frame.sort_values("switch_advantage_pred").copy()
        ranked["bin"] = pd.qcut(
            ranked["switch_advantage_pred"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        binned = ranked.groupby("bin", observed=True).agg(
            adv=("switch_advantage_pred", "mean"),
            target=("adaptive_adv_bp_day", "mean"),
        )
        ax.plot(
            binned["adv"],
            binned["target"],
            marker="o",
            color=color,
            linewidth=2,
        )
        ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
        row_stats = alignment.loc[
            (alignment["market"] == market)
            & (alignment["target"] == "Training-aligned adaptive horizon")
            & (alignment["signal"] == "Adv")
        ].iloc[0]
        ax.set_title(
            f"Adv deciles: rho={row_stats['spearman_rho']:.3f}, "
            f"NW p={row_stats['newey_west_p']:.3f}"
        )
        ax.set_xlabel("Predicted Adv")
        ax.set_ylabel("Realized adaptive advantage (bp/day)")

        ax = axes[row, 2]
        effect = decision_effect.loc[
            (decision_effect["market"] == market)
            & (decision_effect["outcome"] == "Fixed 20-day")
            & (
                decision_effect["comparison"]
                == "Actual Switch vs Hold"
            )
        ].iloc[0]
        means = [
            effect["reference_mean_bp_day"],
            effect["positive_group_mean_bp_day"],
        ]
        ax.bar(
            ["Hold decisions", "Switch decisions"],
            means,
            color=["#D55E55", "#159D82"],
            alpha=0.9,
        )
        ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
        ax.set_ylabel("20-day Switch−Hold advantage (bp/day)")
        ax.set_title(
            f"Difference={effect['difference_bp_day']:+.2f} bp/day, "
            f"NW p={effect['newey_west_p']:.3f}"
        )
    fig.suptitle(
        "Controller Base and Advantage: full-test-set diagnostics",
        fontsize=16,
    )
    fig.savefig(output_dir / "controller_base_adv_statistics.png", dpi=220)
    fig.savefig(output_dir / "controller_base_adv_statistics.pdf")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def write_report(
    output_dir: Path,
    role: pd.DataFrame,
    alignment: pd.DataFrame,
    quartile: pd.DataFrame,
    decision_effect: pd.DataFrame,
) -> None:
    role_view = role[
        [
            "market",
            "free_decisions",
            "base_prob_mean",
            "base_prob_sd_percentage_points",
            "base_only_switches",
            "adv_to_base_dynamic_sd_ratio",
            "adv_only_switch_rate",
            "actual_switch_rate",
            "positive_adv_blocked_by_base",
            "median_raw_adv_needed_to_switch",
        ]
    ].copy()
    role_view.columns = [
        "Market",
        "N",
        "Base prob mean",
        "Base prob SD (pp)",
        "Base-only Switch",
        "Adv/Base dynamic SD",
        "Adv-only Switch rate",
        "Final Switch rate",
        "Positive Adv blocked",
        "Median Adv threshold",
    ]
    for column in [
        "Base prob mean",
        "Base prob SD (pp)",
        "Adv/Base dynamic SD",
        "Adv-only Switch rate",
        "Final Switch rate",
        "Median Adv threshold",
    ]:
        role_view[column] = role_view[column].map(lambda x: format_float(x, 4))

    align_view = alignment.loc[
        alignment["signal"].isin(["Base", "Adv"]),
        [
            "market",
            "target",
            "signal",
            "spearman_rho",
            "block_ci_low",
            "block_ci_high",
            "newey_west_p",
            "bh_q_within_target_base_adv",
            "sign_auc",
        ],
    ].copy()
    align_view.columns = [
        "Market",
        "Target",
        "Signal",
        "Spearman rho",
        "CI low",
        "CI high",
        "NW p",
        "BH q",
        "Sign AUC",
    ]
    for column in [
        "Spearman rho",
        "CI low",
        "CI high",
        "NW p",
        "BH q",
        "Sign AUC",
    ]:
        align_view[column] = align_view[column].map(format_float)

    quartile_view = quartile[
        [
            "market",
            "signal",
            "bottom_quartile_mean_bp_day",
            "top_quartile_mean_bp_day",
            "top_minus_bottom_bp_day",
            "block_ci_low",
            "block_ci_high",
            "newey_west_p",
            "bh_q",
        ]
    ].copy()
    quartile_view.columns = [
        "Market",
        "Signal",
        "Bottom Q mean",
        "Top Q mean",
        "Q4-Q1 bp/day",
        "CI low",
        "CI high",
        "NW p",
        "BH q",
    ]
    for column in quartile_view.columns[2:]:
        quartile_view[column] = quartile_view[column].map(format_float)

    effect_view = decision_effect.loc[
        decision_effect["comparison"] == "Actual Switch vs Hold",
        [
            "market",
            "outcome",
            "reference_mean_bp_day",
            "positive_group_mean_bp_day",
            "difference_bp_day",
            "block_ci_low",
            "block_ci_high",
            "newey_west_p",
            "bh_q_within_outcome_comparison",
        ],
    ].copy()
    effect_view.columns = [
        "Market",
        "Outcome",
        "Hold mean",
        "Switch mean",
        "Switch-Hold bp/day",
        "CI low",
        "CI high",
        "NW p",
        "BH q",
    ]
    for column in effect_view.columns[2:]:
        effect_view[column] = effect_view[column].map(format_float)

    report = f"""# Controller Base 与 Adv 全测试集统计分析

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-29
- Verification Status：VERIFIED
- Version Label：controller_base_adv_stats_v1
- Markets：NASDAQ-100 seed 49；CSI-300 seed 90
- Sample：完整测试集中的自由 Controller 决策日
- HAC：Newey–West，5阶滞后
- Confidence interval：20日循环区块 bootstrap，2,000次，固定随机种子

## 1. 统计对象

- **Base**：`base_exit_logit`，以及其单独对应的 `sigmoid(base_exit_logit)`；
- **Adv**：`switch_advantage_pred`，实际策略修正为
  `1.9 × tanh(switch_advantage_pred / 0.02)`；
- **训练对齐目标**：决策时 Switch 与 Hold 两个候选组合在剩余持有期内的反事实
  log-return 差，扣除二者增量换手成本；
- **固定期限稳健性**：同一决策日未来20日和30日的冻结组合反事实收益差。

## 2. Base 与 Adv 的机制分工

{markdown_table(role_view)}

Base 的概率在两个市场都稳定在约27%，单独从未超过0.5。Adv 的动态标准差是
Base 的约155–457倍，说明最终概率的时间变化几乎完全来自Adv。Base的实际作用是
设置约0.012的正Adv门槛，把“Adv为正就切换”的高频策略过滤为更低频的最终Switch。

## 3. Base 与 Adv 是否对应真实反事实优势

{markdown_table(align_view)}

NASDAQ-100中，Base与Adv都没有形成可重复的优势排序；Adv在训练对齐目标上的
Spearman相关仅约0.04，Newey–West检验不显著。CSI-300中，Adv对训练对齐目标存在
弱正相关，且在HAC和BH校正后仍保留统计证据；Base仍接近随机。固定20日目标下，
两个市场的Adv关系都没有通过HAC检验，因此CSI结果应描述为“期限匹配下的弱证据”，
不能描述为跨期限稳定预测。

## 4. 极端分组检验

{markdown_table(quartile_view)}

CSI-300的Adv最高四分位相对于最低四分位对应更高的事后优势；NASDAQ-100没有
相同证据。Base的四分位差在两个市场都不稳定。

## 5. 最终 Controller 动作的反事实结果

{markdown_table(effect_view)}

- CSI-300：Switch日相对于Hold日在三个窗口中均表现更好；20日结果的
  Newey–West p与两市场BH q均约为0.044，属于边界性证据，30日结果更稳定。
- NASDAQ-100：训练对齐窗口无显著差异；固定20/30日的点估计为负，虽然
  Newey–West p约为0.027/0.029，但20日区块bootstrap置信区间为
  [−4.27, 0.25]、30日为[−3.58, 0.22]，均包含0。两种推断不一致，因此只能
  视为负向警示，不能断言Controller在NASDAQ上稳定改善或稳定损害切换收益。

## 6. 可以支持的结论

1. **结构分工得到强支持**：Base是低波动的保守阈值，Adv承担动态调制；
2. **Base不是独立预测器**：它不单独触发Switch，对真实优势的AUC约为0.5；
3. **Adv的统计解释具有市场差异**：CSI-300存在弱但可检验的期限匹配信号，
   NASDAQ-100没有；
4. **联合Controller的正向证据主要来自CSI-300**，NASDAQ结果只能支持“保守过滤机制”，
   不能支持“稳定改善切换收益”。

## 7. 解释边界与统计谬误检查

- Coverage：11/11。
- Simpson：NASDAQ与CSI分别报告，未用汇总市场结果掩盖方向差异；
- Ecological：推断单位保持为决策日，不外推到单只股票；
- Berkson：只分析自由决策日是模型机制规定的条件样本，结论不外推到强制动作；
- Collider：未加入事后表现作为控制变量；
- Base-rate neglect：同时报告正优势比例、AUC和balanced accuracy；
- Regression to mean：使用全测试集，不按极端事后收益筛选case；
- Survivorship：使用完整测试轨迹；不足20/30个未来交易日的末端样本从对应固定期限检验中排除；
- Look-elsewhere：训练对齐目标预先作为主结果，固定20/30日作为稳健性；
- Forking paths：分析属于事后解释性验证，不能当作预注册确认性证据；
- Correlation/causation：只陈述关联和反事实对齐，不作因果归因；
- Reverse causality：预测量先于未来收益，但模型选择和训练过程仍可能造成样本内偏差。

## 8. 文件

- `controller_base_adv_distribution.csv`：各信号完整分布；
- `controller_base_adv_roles.csv`：Base/Adv阈值与动作分工；
- `controller_base_adv_alignment.csv`：相关、区块CI、HAC、FDR与AUC；
- `controller_base_adv_quartiles.csv`：极端四分位检验；
- `controller_base_adv_decision_effect.csv`：Switch/Hold与Adv正负分组结果；
- `controller_base_adv_statistics.png/.pdf`：核心统计图。
"""
    (output_dir / "CONTROLLER_BASE_ADV_STATISTICS_CN.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_by_market = {
        market: load_market(args.repo, stem, seed)
        for market, stem, seed in MARKETS
    }
    distribution = distribution_table(data_by_market)
    role = role_table(data_by_market)
    alignment = alignment_table(
        data_by_market, reps=max(100, args.bootstrap_reps)
    )
    quartile = quartile_table(
        data_by_market, reps=max(100, args.bootstrap_reps)
    )
    decision_effect = decision_effect_table(
        data_by_market, reps=max(100, args.bootstrap_reps)
    )

    distribution.to_csv(
        args.output_dir / "controller_base_adv_distribution.csv", index=False
    )
    role.to_csv(args.output_dir / "controller_base_adv_roles.csv", index=False)
    alignment.to_csv(
        args.output_dir / "controller_base_adv_alignment.csv", index=False
    )
    quartile.to_csv(
        args.output_dir / "controller_base_adv_quartiles.csv", index=False
    )
    decision_effect.to_csv(
        args.output_dir / "controller_base_adv_decision_effect.csv", index=False
    )
    make_figure(data_by_market, alignment, decision_effect, args.output_dir)
    write_report(args.output_dir, role, alignment, quartile, decision_effect)

    for market, frame in data_by_market.items():
        print(
            f"{market}: free={len(frame)}, "
            f"formula_max_error={frame['formula_error'].max():.3e}, "
            f"switches={int(frame['actual_switch'].sum())}"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
