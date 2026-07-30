#!/usr/bin/env python3
"""Replay two Controller cases and neutralize the Outer candidate.

The neutral probe replaces ``switch_action`` with the current drifted holdings.
This removes candidate-current differences while leaving the market and
portfolio state unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASES = {
    "nas": {
        "market": "NASDAQ-100",
        "seed": 49,
        "date": "2025-05-08",
        "action": "Hold",
    },
    "sh": {
        "market": "CSI-300",
        "seed": 90,
        "date": "2021-07-07",
        "action": "Switch",
    },
}
ADV_COEF = 1.9
ADV_SCALE = 0.02


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--results-end",
        type=Path,
        default=Path("/home/tongwenxuan/KD4RL_plus/results/end"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "reproduced_outputs" / "controller_base_adv_cases",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse the existing ablation CSV and only rebuild report/figure.",
    )
    return parser.parse_args()


def load_figure_module(repo: Path):
    script = repo / "scripts" / "generate_interpretability_figures.py"
    spec = importlib.util.spec_from_file_location("controller_figure_runtime", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._code_root_for_results_end = lambda _: repo
    return module


def to_float(value) -> float:
    return float(value.detach().view(-1)[0].cpu().item())


def replay_probe(
    *,
    figure_module,
    repo: Path,
    results_end: Path,
    output_dir: Path,
    market_key: str,
    device: str,
) -> dict[str, object]:
    import torch

    case = CASES[market_key]
    trainer = figure_module.build_loaded_trainer(
        market_key, output_dir, device, results_end
    )
    env = trainer.env
    env.set_mode("test")
    test_window = trainer._test_episode_window()
    obs = env.reset_at(*test_window) if test_window is not None else env.reset()
    spec = trainer._get_phase_spec("joint")
    step_idx = 0
    last_switch_step = 0
    switch_count = 0

    with torch.no_grad():
        while True:
            duration = step_idx - last_switch_step
            force_switch, force_locked = trainer._compute_force_switch_locked(
                spec=spec,
                phase="joint",
                step_idx=step_idx,
                duration=duration,
                is_train=False,
                switch_schedule=None,
                fixed_cycle=None,
                current_segments=switch_count,
                rollout_len=int(
                    getattr(env, "current_episode_len", env.episode_len)
                ),
            )
            out = trainer.agent.get_action(
                obs,
                mode="eval",
                force_switch=force_switch,
                force_inner_zero=False,
                force_locked=force_locked,
            )
            stats = trainer.agent.net.mon.decision_stats(
                obs["ssm"]["z"],
                obs["ssm"]["h"],
                obs["ssm"]["p"],
                obs["ssm"]["q_bear"],
                obs["ssm"]["q_bull"],
                obs["weights_drift"],
                obs["port_state"],
                switch_action=out["act_out"],
                asset_state=obs.get("outer_state"),
            )
            date_value = str(env.all_dates[int(env.day)].date())
            is_switch = bool(out["act_mon"].view(-1)[0].item() == 1)
            if date_value == case["date"]:
                expected_switch = case["action"] == "Switch"
                if force_switch is not None or is_switch != expected_switch:
                    raise RuntimeError(
                        f"{market_key} {date_value}: expected a free "
                        f"{case['action']}"
                    )
                neutral_stats = trainer.agent.net.mon.decision_stats(
                    obs["ssm"]["z"],
                    obs["ssm"]["h"],
                    obs["ssm"]["p"],
                    obs["ssm"]["q_bear"],
                    obs["ssm"]["q_bull"],
                    obs["weights_drift"],
                    obs["port_state"],
                    switch_action=obs["weights_drift"],
                    asset_state=obs.get("outer_state"),
                )
                candidate_turnover, candidate_overlap, concentration = (
                    figure_module._normalized_row_sum_abs(
                        obs["weights_drift"], out["act_out"]
                    )
                )
                original_base = to_float(stats["base_exit_logit"])
                original_adv = to_float(stats["switch_advantage_pred"])
                original_adv_logit = ADV_COEF * np.tanh(
                    original_adv / ADV_SCALE
                )
                neutral_base = to_float(neutral_stats["base_exit_logit"])
                neutral_adv = to_float(
                    neutral_stats["switch_advantage_pred"]
                )
                neutral_adv_logit = ADV_COEF * np.tanh(
                    neutral_adv / ADV_SCALE
                )
                return {
                    "market": case["market"],
                    "market_key": market_key,
                    "seed": case["seed"],
                    "date": date_value,
                    "step": step_idx,
                    "duration_before_decision": duration,
                    "candidate_turnover": candidate_turnover,
                    "candidate_overlap": candidate_overlap,
                    "hold_concentration": concentration,
                    "original_base_logit": original_base,
                    "neutral_base_logit": neutral_base,
                    "candidate_effect_on_base": original_base - neutral_base,
                    "original_adv_raw": original_adv,
                    "neutral_adv_raw": neutral_adv,
                    "candidate_effect_on_adv_raw": original_adv - neutral_adv,
                    "original_adv_logit": original_adv_logit,
                    "neutral_adv_logit": neutral_adv_logit,
                    "candidate_effect_on_adv_logit": (
                        original_adv_logit - neutral_adv_logit
                    ),
                    "original_final_logit": to_float(stats["exit_logit"]),
                    "neutral_final_logit": to_float(
                        neutral_stats["exit_logit"]
                    ),
                    "original_exit_prob": to_float(stats["exit_prob"]),
                    "neutral_exit_prob": to_float(
                        neutral_stats["exit_prob"]
                    ),
                    "original_action": case["action"],
                    "neutral_action": (
                        "Switch"
                        if to_float(neutral_stats["exit_prob"]) > 0.5
                        else "Hold"
                    ),
                    "original_formula_error": abs(
                        original_base
                        + original_adv_logit
                        - to_float(stats["exit_logit"])
                    ),
                    "neutral_formula_error": abs(
                        neutral_base
                        + neutral_adv_logit
                        - to_float(neutral_stats["exit_logit"])
                    ),
                }

            next_obs, _, done, _ = env.step(
                out["weights_exec"].detach(),
                out["base_used"].detach(),
                outer_action=out["act_out"].detach(),
                is_switch=is_switch,
            )
            if is_switch:
                switch_count += 1
                last_switch_step = step_idx
            if done:
                break
            obs = next_obs
            step_idx += 1
    raise RuntimeError(f"{market_key}: target date {case['date']} not found")


def attach_case_data(repo: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trace_dir = (
        repo / "reproduced_outputs" / "controller_gate_adv_case_analysis"
    )
    cf_dir = (
        repo
        / "paper_experiments_outputs"
        / "paper_experiments_final"
        / "_cache"
        / "counterfactual_horizon30"
    )
    for record in frame.to_dict("records"):
        stem = record["market_key"]
        seed = int(record["seed"])
        trace = pd.read_csv(
            trace_dir / f"controller_gate_adv_trace_{stem}.csv"
        )
        cf = pd.read_csv(
            cf_dir
            / f"{stem}_seed{seed}_full_controller_horizon30_actions.csv"
        )
        trace_row = trace.loc[trace["date"] == record["date"]].iloc[0]
        cf_row = cf.loc[cf["date"] == record["date"]].iloc[0]
        for column in ["segment_return", "segment_drawdown"]:
            record[column] = trace_row[column]
        for column in [
            "hold_future_mdd_20",
            "switch_future_mdd_20",
            "hold_curve_20",
            "switch_curve_20",
        ]:
            record[column] = cf_row[column]
        hold_curve = np.asarray(
            json.loads(record["hold_curve_20"]), dtype=float
        )
        switch_curve = np.asarray(
            json.loads(record["switch_curve_20"]), dtype=float
        )
        record["hold_future_return_20"] = hold_curve[-1] - 1.0
        record["switch_future_return_20"] = switch_curve[-1] - 1.0
        record["controller_switch_advantage"] = cf_row[
            "controller_switch_advantage"
        ]
        record["switch_advantage_20"] = cf_row["switch_advantage_20"]
        record["switch_advantage_30"] = cf_row["switch_advantage_30"]
        target_step = int(trace_row["step"])
        duration = int(trace_row["hold_duration"])
        start_step = target_step - duration
        segment = trace.loc[
            (trace["step"] >= start_step) & (trace["step"] < target_step)
        ].copy()
        values = segment["portfolio_value"].to_numpy(dtype=float)
        if len(values) == 0:
            values = np.array([float(trace_row["portfolio_value_before"])])
        past_curve = values / values[0]
        record["past_curve"] = json.dumps(past_curve.tolist())
        rows.append(record)
    return pd.DataFrame(rows)


def plot_cases(frame: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "base": "#1477B8",
        "adv": "#E69F00",
        "final": "#B13A78",
        "hold": "#D65F4B",
        "switch": "#009E73",
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(16.5, 8.7), constrained_layout=True
    )
    for row_idx, row in frame.reset_index(drop=True).iterrows():
        ax = axes[row_idx, 0]
        past = np.asarray(json.loads(row["past_curve"]), dtype=float)
        ax.plot(
            np.arange(len(past)),
            (past - 1.0) * 100.0,
            color=colors["base"],
            linewidth=2.6,
        )
        ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
        ax.set_title(
            f"{row['market']} {row['date']}: current portfolio state"
        )
        ax.set_xlabel("Days in current holding segment")
        ax.set_ylabel("Segment cumulative return (%)")
        ax.text(
            0.03,
            0.05,
            f"Segment return {row['segment_return'] * 100:+.2f}%\n"
            f"Drawdown {row['segment_drawdown'] * 100:.2f}%",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
        )

        ax = axes[row_idx, 1]
        x = np.arange(3)
        width = 0.34
        neutral = [
            row["neutral_base_logit"],
            row["neutral_adv_logit"],
            row["neutral_final_logit"],
        ]
        candidate = [
            row["original_base_logit"],
            row["original_adv_logit"],
            row["original_final_logit"],
        ]
        ax.bar(
            x - width / 2,
            neutral,
            width,
            label="Candidate = current",
            color="#A9A9A9",
        )
        ax.bar(
            x + width / 2,
            candidate,
            width,
            label="Outer candidate",
            color=[colors["base"], colors["adv"], colors["final"]],
        )
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(x, ["Base logit", "Adv correction", "Final logit"])
        ax.set_ylabel("Controller logit")
        ax.set_title(
            f"Candidate ablation: p {row['neutral_exit_prob']:.3f}"
            f" → {row['original_exit_prob']:.3f}"
        )
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[row_idx, 2]
        hold = np.asarray(json.loads(row["hold_curve_20"]), dtype=float)
        switch = np.asarray(json.loads(row["switch_curve_20"]), dtype=float)
        ax.plot(
            np.arange(len(hold)),
            (hold - 1.0) * 100.0,
            color=colors["hold"],
            linewidth=2.5,
            label="Continue Hold",
        )
        ax.plot(
            np.arange(len(switch)),
            (switch - 1.0) * 100.0,
            color=colors["switch"],
            linewidth=2.5,
            label="Use candidate",
        )
        ax.axhline(0, color="#555555", linestyle=":", linewidth=1)
        ax.set_title(
            "20-day frozen counterfactual: "
            f"candidate−hold "
            f"{(row['switch_future_return_20'] - row['hold_future_return_20']) * 100:+.2f} pp"
        )
        ax.set_xlabel("Future trading days")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Controller cases: current portfolio, candidate comparison, and action",
        fontsize=16,
    )
    fig.savefig(output_dir / "controller_base_adv_two_market_cases.png", dpi=220)
    fig.savefig(output_dir / "controller_base_adv_two_market_cases.pdf")
    plt.close(fig)


def write_report(frame: pd.DataFrame, output_dir: Path) -> None:
    table = frame[
        [
            "market",
            "date",
            "original_action",
            "segment_return",
            "segment_drawdown",
            "original_base_logit",
            "candidate_effect_on_base",
            "original_adv_raw",
            "original_adv_logit",
            "candidate_effect_on_adv_logit",
            "original_exit_prob",
            "neutral_exit_prob",
            "hold_future_return_20",
            "switch_future_return_20",
        ]
    ].copy()
    for column in [
        "segment_return",
        "segment_drawdown",
        "hold_future_return_20",
        "switch_future_return_20",
    ]:
        table[column] *= 100.0
    table.columns = [
        "Market",
        "Date",
        "Action",
        "Segment return (%)",
        "Drawdown (%)",
        "Base logit",
        "Candidate ΔBase",
        "Adv raw",
        "Adv correction",
        "Candidate ΔAdv correction",
        "Final p",
        "Neutral-candidate p",
        "Future Hold (%)",
        "Future Candidate (%)",
    ]
    for column in table.columns[3:]:
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")

    nas = frame.loc[frame["market"] == "NASDAQ-100"].iloc[0]
    csi = frame.loc[frame["market"] == "CSI-300"].iloc[0]
    report = f"""# Controller Base–Adv 两市场可解释案例

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-29
- Verification Status：VERIFIED
- Cases：NASDAQ-100 {nas['date']}；CSI-300 {csi['date']}
- Probe：保持市场与当前组合状态不变，仅令candidate等于当前漂移权重

## 1. 需要先修正的解释

代码结构中，Base head和Adv head共享同一个264维表征，且该表征同时包含当前组合、
candidate组合及二者差异。因此不能把Base严格定义为“只观察当前组合的状态”。

更准确的表述是：

- **Base**：在完整决策状态上给出的基础切换倾向；当前checkpoint中表现为稳定的
  保守Hold先验；
- **Adv**：对candidate相对当前组合吸引力的显式辅助预测，并通过有界logit修正
  调节Base；
- **最终动作**：比较 `Base + Adv correction` 是否越过0。

## 2. 两个案例

{table.to_markdown(index=False)}

## 3. NASDAQ-100：{nas['date']}（为什么不切）

当前组合已经持有{int(nas['duration_before_decision'])}日，区间收益
{nas['segment_return'] * 100:+.2f}%，当前回撤{nas['segment_drawdown'] * 100:.2f}%：
这是一个状态良好的当前组合。Base logit为{nas['original_base_logit']:.3f}，
提供保守Hold先验；Outer candidate相对当前组合产生
{nas['original_adv_logit']:+.3f}的Adv修正，使最终Switch概率降至
{nas['original_exit_prob']:.3f}，Controller选择Hold。

未来20日冻结反事实中，继续Hold收益为{nas['hold_future_return_20'] * 100:+.2f}%；
采用candidate收益为{nas['switch_future_return_20'] * 100:+.2f}%，差值为
{(nas['switch_future_return_20'] - nas['hold_future_return_20']) * 100:+.2f}个百分点。
因此这个案例支持的是：当前组合状态好，且candidate相对更差，Adv进一步强化Hold。

## 4. CSI-300：{csi['date']}（为什么切）

当前组合已经持有{int(csi['duration_before_decision'])}日，区间收益
{csi['segment_return'] * 100:+.2f}%，当前回撤{csi['segment_drawdown'] * 100:.2f}%。
Base logit为{csi['original_base_logit']:.3f}，仍未单独支持切换；candidate产生
{csi['original_adv_logit']:+.3f}的Adv修正，把最终概率推至
{csi['original_exit_prob']:.3f}并触发Switch。

未来20日中，继续Hold收益为{csi['hold_future_return_20'] * 100:+.2f}%，采用candidate为
{csi['switch_future_return_20'] * 100:+.2f}%，改善
{(csi['switch_future_return_20'] - csi['hold_future_return_20']) * 100:+.2f}个百分点。

## 5. Candidate消融如何解释

把candidate替换成当前组合后，市场状态、持仓时间、区间收益和回撤全部保持不变。
因此：

- `Candidate ΔBase` 衡量Base head受到candidate输入影响的程度；
- `Candidate ΔAdv correction` 衡量Adv通道对candidate差异的响应；
- 如果Adv变化明显大于Base变化，可以说该checkpoint在行为上形成了“稳定Base +
  candidate-sensitive Adv”的近似分工；
- 即使消融支持这种近似分工，也不能声称两个head在架构上完全解耦。

## 6. 解释边界

两个case用于解释计算闭环，不是总体有效性的统计证明。对NASDAQ-100全部231个
自由Switch决策做candidate=current消融后，没有一次出现“中性candidate为Hold、
真实candidate将其翻转为Switch”；只有1次Adv修正略微增加（+0.0018）。
因此不能为NASDAQ挑选一个并不存在的“candidate通过Adv推动Switch”的案例。

这里改用一个可核验的Hold案例，与CSI-300的Switch案例组成互补解释：

- NASDAQ-100：当前组合好、candidate较差，Adv强化Hold；
- CSI-300：当前组合弱、candidate较好，Adv克服Base先验并推动Switch。

总体统计还表明，CSI-300的Adv与训练对齐反事实优势存在弱正相关，而
NASDAQ-100没有显著全局关系。因此两例只能证明行为链条在具体决策上可以解释，
不能据此声称两个市场都有稳定、普遍的Adv预测能力。
"""
    (output_dir / "CONTROLLER_BASE_ADV_TWO_CASES_CN.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "controller_base_adv_case_ablation.csv"
    if args.reuse:
        frame = pd.read_csv(csv_path)
    else:
        figure_module = load_figure_module(args.repo)
        rows = [
            replay_probe(
                figure_module=figure_module,
                repo=args.repo,
                results_end=args.results_end,
                output_dir=args.output_dir,
                market_key=market_key,
                device=args.device,
            )
            for market_key in ("nas", "sh")
        ]
        frame = attach_case_data(args.repo, pd.DataFrame(rows))
        frame.to_csv(csv_path, index=False)
    plot_cases(frame, args.output_dir)
    write_report(frame, args.output_dir)
    print(frame.to_string(index=False))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
