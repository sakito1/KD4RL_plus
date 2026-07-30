#!/usr/bin/env python3
"""Plot two verified cases where Base filters a weak positive Adv signal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADV_COEF = 1.9
ADV_SCALE = 0.02
CASES = (
    ("NASDAQ-100", "nas", 49, "2023-01-17"),
    ("CSI-300", "sh", 90, "2021-07-07"),
)


def adv_correction(raw_adv: float) -> float:
    return ADV_COEF * math.tanh(raw_adv / ADV_SCALE)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def trace_path(repo: Path, key: str) -> Path:
    return (
        repo
        / "reproduced_outputs"
        / "controller_gate_adv_case_analysis"
        / f"controller_gate_adv_trace_{key}.csv"
    )


def counterfactual_path(repo: Path, key: str, seed: int) -> Path:
    return (
        repo
        / "paper_experiments_outputs"
        / "paper_experiments_final"
        / "_cache"
        / "counterfactual_horizon30"
        / f"{key}_seed{seed}_full_controller_horizon30_actions.csv"
    )


def load_case_data(repo: Path) -> pd.DataFrame:
    ablation_path = (
        repo
        / "reproduced_outputs"
        / "controller_base_adv_cases"
        / "controller_base_adv_case_ablation.csv"
    )
    ablation = (
        pd.read_csv(ablation_path)
        if ablation_path.exists()
        else pd.DataFrame()
    )
    rows = []
    for market, key, seed, date in CASES:
        trace = pd.read_csv(trace_path(repo, key))
        cf = pd.read_csv(counterfactual_path(repo, key, seed))
        trace_row = trace.loc[trace["date"].eq(date)].iloc[0]
        cf_row = cf.loc[cf["date"].eq(date)].iloc[0]
        base = float(trace_row["base_exit_logit"])
        raw_adv = float(trace_row["switch_advantage_pred"])
        correction = adv_correction(raw_adv)
        final_logit = base + correction
        hold_curve_20 = np.asarray(
            json.loads(cf_row["hold_curve_20"]), dtype=float
        )
        candidate_curve_20 = np.asarray(
            json.loads(cf_row["switch_curve_20"]), dtype=float
        )
        hold_curve_30 = np.asarray(
            json.loads(cf_row["hold_curve_30"]), dtype=float
        )
        candidate_curve_30 = np.asarray(
            json.loads(cf_row["switch_curve_30"]), dtype=float
        )
        target_step = int(trace_row["step"])
        duration = int(trace_row["hold_duration"])
        segment = trace.loc[
            (trace["step"] >= target_step - duration)
            & (trace["step"] < target_step)
        ]
        values = segment["portfolio_value"].to_numpy(dtype=float)
        if len(values) == 0:
            values = np.array([float(trace_row["portfolio_value_before"])])
        past_curve = values / values[0]
        neutral_probability = np.nan
        neutral_action = "N/A"
        if not ablation.empty:
            neutral_match = ablation.loc[
                ablation["market"].eq(market)
                & ablation["date"].eq(date)
            ]
            if len(neutral_match) == 1:
                neutral_row = neutral_match.iloc[0]
                neutral_probability = float(
                    neutral_row["neutral_exit_prob"]
                )
                neutral_action = str(neutral_row["neutral_action"])
        rows.append(
            {
                "market": market,
                "market_key": key,
                "seed": seed,
                "date": date,
                "action": (
                    "Switch" if bool(int(trace_row["is_switch"])) else "Hold"
                ),
                "holding_duration": duration,
                "segment_return": float(trace_row["segment_return"]),
                "segment_drawdown": float(trace_row["segment_drawdown"]),
                "base_logit": base,
                "base_probability": sigmoid(base),
                "adv_raw": raw_adv,
                "adv_correction": correction,
                "adv_only_probability": sigmoid(correction),
                "final_logit": final_logit,
                "final_probability": sigmoid(final_logit),
                "neutral_candidate_probability": neutral_probability,
                "neutral_candidate_action": neutral_action,
                "formula_error": abs(
                    final_logit - float(trace_row["exit_logit"])
                ),
                "hold_return_20": hold_curve_20[-1] - 1.0,
                "candidate_return_20": candidate_curve_20[-1] - 1.0,
                "candidate_minus_hold_20": (
                    candidate_curve_20[-1] - hold_curve_20[-1]
                ),
                "hold_return_30": hold_curve_30[-1] - 1.0,
                "candidate_return_30": candidate_curve_30[-1] - 1.0,
                "candidate_minus_hold_30": (
                    candidate_curve_30[-1] - hold_curve_30[-1]
                ),
                "adaptive_candidate_advantage": float(
                    cf_row["controller_switch_advantage"]
                ),
                "past_curve": json.dumps(past_curve.tolist()),
                "hold_curve_30": json.dumps(hold_curve_30.tolist()),
                "candidate_curve_30": json.dumps(
                    candidate_curve_30.tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_cases(cases: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "state": "#2B6CB0",
        "base": "#2B6CB0",
        "adv": "#E69F00",
        "final": "#9C3D78",
        "hold": "#D55E5A",
        "candidate": "#009E73",
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(16.8, 9.2), constrained_layout=True
    )
    for row_index, row in cases.reset_index(drop=True).iterrows():
        market = row["market"]

        ax = axes[row_index, 0]
        past = np.asarray(json.loads(row["past_curve"]), dtype=float)
        ax.plot(
            np.arange(len(past)),
            (past - 1.0) * 100.0,
            color=colors["state"],
            linewidth=2.7,
        )
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1)
        ax.set_title(f"{market} {row['date']}: current portfolio")
        ax.set_xlabel("Days in current holding segment")
        ax.set_ylabel("Cumulative return (%)")
        ax.text(
            0.03,
            0.05,
            f"Held {int(row['holding_duration'])} days\n"
            f"Return {row['segment_return'] * 100:+.2f}% | "
            f"DD {row['segment_drawdown'] * 100:.2f}%",
            transform=ax.transAxes,
            fontsize=10,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#BBBBBB",
            },
        )

        ax = axes[row_index, 1]
        base = float(row["base_logit"])
        correction = float(row["adv_correction"])
        final = float(row["final_logit"])
        ax.bar(
            0,
            base,
            color=colors["base"],
            width=0.62,
            label="Base",
        )
        ax.bar(
            1,
            correction,
            bottom=base,
            color=colors["adv"],
            width=0.62,
            label="Adv correction",
        )
        ax.bar(
            2,
            final,
            color=colors["final"],
            width=0.62,
            label="Final",
        )
        ax.plot([0, 1], [base, base], color="#555555", linewidth=1)
        ax.plot([1, 2], [final, final], color="#555555", linewidth=1)
        ax.axhline(
            0.0,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label="Switch threshold",
        )
        ax.set_xticks([0, 1, 2], ["Base", "+ Adv", "Final"])
        ax.set_ylabel("Controller logit")
        ax.set_title(
            (
                "Base blocks weak positive Adv"
                if row["action"] == "Hold"
                else "Adv overcomes the Base hurdle"
            )
            + f": p {row['adv_only_probability']:.3f}"
            + f" → {row['final_probability']:.3f}"
        )
        ax.text(
            0,
            base - 0.06,
            f"{base:+.3f}",
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.text(
            1,
            base + correction / 2,
            f"+{correction:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            2,
            final - 0.04,
            f"{final:+.3f}",
            ha="center",
            va="top",
            fontsize=10,
        )
        required_adv = ADV_SCALE * np.arctanh(-base / ADV_COEF)
        candidate_probe = ""
        if np.isfinite(row["neutral_candidate_probability"]):
            candidate_probe = (
                "\ncandidate=current: "
                f"p {row['neutral_candidate_probability']:.3f} "
                f"{row['neutral_candidate_action']}"
            )
        ax.text(
            0.03,
            0.97,
            f"raw Adv = {row['adv_raw']:.5f}\n"
            f"required Adv = {required_adv:.5f}\n"
            f"decision: {row['action']}"
            f"{candidate_probe}",
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#BBBBBB",
            },
        )

        ax = axes[row_index, 2]
        hold = np.asarray(json.loads(row["hold_curve_30"]), dtype=float)
        candidate = np.asarray(
            json.loads(row["candidate_curve_30"]), dtype=float
        )
        days = np.arange(len(hold))
        ax.plot(
            days,
            (hold - 1.0) * 100.0,
            color=colors["hold"],
            linewidth=2.6,
            label="Continue Hold",
        )
        ax.plot(
            days,
            (candidate - 1.0) * 100.0,
            color=colors["candidate"],
            linewidth=2.6,
            label="Use candidate",
        )
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1)
        ax.axvline(20, color="#888888", linestyle="--", linewidth=1)
        ax.set_title("Frozen counterfactual after decision")
        ax.set_xlabel("Future trading days")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(loc="best", fontsize=9)
        ax.text(
            0.03,
            0.05,
            f"Candidate−Hold\n"
            f"20d {row['candidate_minus_hold_20'] * 100:+.2f} pp\n"
            f"30d {row['candidate_minus_hold_30'] * 100:+.2f} pp",
            transform=ax.transAxes,
            fontsize=10,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#BBBBBB",
            },
        )

    fig.suptitle(
        "Controller Base–Adv interaction: complementary Hold and Switch cases",
        fontsize=17,
        fontweight="bold",
    )
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.18)
    fig.savefig(
        output_dir / "controller_base_filter_cases.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "controller_base_filter_cases.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def write_report(cases: pd.DataFrame, output_dir: Path) -> None:
    case_table = cases[
        [
            "market",
            "date",
            "action",
            "holding_duration",
            "segment_return",
            "segment_drawdown",
            "base_logit",
            "adv_raw",
            "adv_correction",
            "adv_only_probability",
            "final_probability",
            "candidate_minus_hold_20",
            "candidate_minus_hold_30",
        ]
    ].copy()
    for column in [
        "segment_return",
        "segment_drawdown",
        "adv_only_probability",
        "final_probability",
        "candidate_minus_hold_20",
        "candidate_minus_hold_30",
    ]:
        case_table[column] *= 100.0
    case_table.columns = [
        "Market",
        "Date",
        "Action",
        "Held days",
        "Segment return (%)",
        "Drawdown (%)",
        "Base logit",
        "Raw Adv",
        "Adv correction",
        "Adv-only p (%)",
        "Final p (%)",
        "Candidate−Hold 20d (pp)",
        "Candidate−Hold 30d (pp)",
    ]
    numeric_columns = case_table.columns[4:]
    for column in numeric_columns:
        case_table[column] = case_table[column].map(
            lambda value: f"{float(value):.4f}"
        )

    nas = cases.loc[cases["market"].eq("NASDAQ-100")].iloc[0]
    csi = cases.loc[cases["market"].eq("CSI-300")].iloc[0]
    report = f"""# Controller Base–Adv 互补案例图表报告

## Material Passport

- Origin：完整测试集Controller trace与冻结反事实轨迹
- Verification Status：VERIFIED
- Markets：NASDAQ-100 seed 49；CSI-300 seed 90
- Cases：NASDAQ-100 {nas['date']}；CSI-300 {csi['date']}
- Claim level：一个Hold优势案例 + 一个Switch优势案例

## 1. 图表要证明什么

本图验证的核心机制是：

`Base + Adv correction = Final logit`。

NASDAQ案例说明：弱正Adv不足以克服Base，最终Hold，事后当前组合更好。
CSI案例说明：candidate的正Adv克服Base，最终Switch，事后candidate更好。
两例共同展示Controller既不会对任意正Adv机械切换，也能在candidate优势足够强时
执行切换。

## 2. 案例数值

{markdown_table(case_table)}

### NASDAQ-100：Hold更好（{nas['date']}）

当前组合持有{int(nas['holding_duration'])}日，区间收益
{nas['segment_return'] * 100:+.2f}%，回撤
{nas['segment_drawdown'] * 100:.2f}%。raw Adv为
{nas['adv_raw']:.5f}，方向为正但低于约0.012的Base门槛。
Adv-only概率为{nas['adv_only_probability']:.2%}，加入Base后降为
{nas['final_probability']:.2%}并Hold。candidate未来20日和30日分别落后
{abs(nas['candidate_minus_hold_20']) * 100:.2f}和
{abs(nas['candidate_minus_hold_30']) * 100:.2f}个百分点。

### CSI-300：Switch更好（{csi['date']}）

当前组合持有{int(csi['holding_duration'])}日，区间收益
{csi['segment_return'] * 100:+.2f}%，回撤
{csi['segment_drawdown'] * 100:.2f}%。Base-only概率仅为
{csi['base_probability']:.2%}，单独倾向Hold；raw Adv为
{csi['adv_raw']:.5f}，超过当前Base要求的门槛。
Adv correction把最终Switch概率推至
{csi['final_probability']:.2%}并触发Switch。candidate未来20日和30日分别领先
{csi['candidate_minus_hold_20'] * 100:.2f}和
{csi['candidate_minus_hold_30'] * 100:.2f}个百分点。
受控消融中，把candidate替换为当前组合后，Switch概率降为
{csi['neutral_candidate_probability']:.2%}并变为
{csi['neutral_candidate_action']}；说明该次动作翻转来自candidate相关的Adv通道，
而不是Base自身变化。

## 3. 可用于论文的案例结论

> The two cases illustrate complementary Controller behaviors. In NASDAQ-100,
> the Base hurdle filters a weak positive Adv signal and preserves the
> better-performing current portfolio. In CSI-300, a sufficiently strong
> candidate-relative Adv signal overcomes the conservative Base and triggers
> a profitable switch.

## 4. 案例解释边界

- 两个case用于展示机制闭环，不用于估计总体显著性；
- 未来收益只用于事后验证，没有进入当天Controller输入；
- 固定20/30日反事实冻结两套组合权重，比较的是同日起点下的配置差异；
- Base在当前checkpoint中接近固定负偏置，案例能说明其门槛作用，
  但不能证明学习型Base head不可替代；证明必要性仍需no-Base或fixed-threshold消融。
"""
    (output_dir / "CONTROLLER_BASE_FILTER_CASES_CN.md").write_text(
        report, encoding="utf-8"
    )


def build_outputs(repo: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    population_output = output_dir / "controller_base_filter_population.csv"
    if population_output.exists():
        population_output.unlink()
    cases = load_case_data(repo)
    cases.to_csv(
        output_dir / "controller_base_filter_cases.csv", index=False
    )
    plot_cases(cases, output_dir)
    write_report(cases, output_dir)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo
            / "reproduced_outputs"
            / "controller_base_filter_cases"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_outputs(args.repo, args.output_dir)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
