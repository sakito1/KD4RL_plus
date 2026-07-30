#!/usr/bin/env python3
"""Measure Controller Base sensitivity to current-portfolio state inputs.

For every free decision in the test trajectory, replay the checkpoint while
holding market features, holdings, and the Outer candidate fixed.  Then set one
portfolio-state field at a time to its neutral value:

* holding age -> 0 (new holding segment)
* drawdown -> 0
* segment return -> 0

The resulting Base-logit differences isolate behavioral sensitivity to each
explicit state field.  This is an input ablation, not a causal claim about
future portfolio performance.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


MARKETS = (
    ("NASDAQ-100", "nas", 49),
    ("CSI-300", "sh", 90),
)


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
        "--output",
        type=Path,
        default=(
            repo
            / "reproduced_outputs"
            / "controller_base_state"
            / "controller_base_state_ablation.csv"
        ),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_runtime(repo: Path):
    script = repo / "scripts" / "generate_interpretability_figures.py"
    spec = importlib.util.spec_from_file_location("controller_runtime", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._code_root_for_results_end = lambda _: repo
    return module


def scalar(value) -> float:
    return float(value.detach().view(-1)[0].cpu().item())


def replay_market(
    runtime,
    repo: Path,
    results_end: Path,
    output_dir: Path,
    market: str,
    key: str,
    seed: int,
    device: str,
) -> list[dict[str, object]]:
    import torch

    trainer = runtime.build_loaded_trainer(
        key, output_dir, device, results_end
    )
    env = trainer.env
    env.set_mode("test")
    test_window = trainer._test_episode_window()
    obs = env.reset_at(*test_window) if test_window is not None else env.reset()
    spec = trainer._get_phase_spec("joint")
    step_idx = 0
    last_switch_step = 0
    switch_count = 0
    rows: list[dict[str, object]] = []

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
            is_switch = bool(out["act_mon"].view(-1)[0].item() == 1)

            def base_with(port_state):
                stats = trainer.agent.net.mon.decision_stats(
                    obs["ssm"]["z"],
                    obs["ssm"]["h"],
                    obs["ssm"]["p"],
                    obs["ssm"]["q_bear"],
                    obs["ssm"]["q_bull"],
                    obs["weights_drift"],
                    port_state,
                    switch_action=out["act_out"],
                    asset_state=obs.get("outer_state"),
                )
                return scalar(stats["base_exit_logit"])

            if force_switch is None:
                state = obs["port_state"].clone()
                no_age = state.clone()
                no_age[:, 0] = 0.0
                no_drawdown = state.clone()
                no_drawdown[:, 1] = 0.0
                no_return = state.clone()
                no_return[:, 2] = 0.0
                neutral = state.clone()
                neutral[:, :3] = 0.0
                original = base_with(state)
                age_zero = base_with(no_age)
                drawdown_zero = base_with(no_drawdown)
                return_zero = base_with(no_return)
                all_zero = base_with(neutral)
                rows.append(
                    {
                        "market": market,
                        "market_key": key,
                        "seed": seed,
                        "date": str(env.all_dates[int(env.day)].date()),
                        "step": step_idx,
                        "holding_duration": duration,
                        "time_norm": scalar(state[:, 0]),
                        "segment_drawdown": scalar(state[:, 1]),
                        "segment_return": scalar(state[:, 2]),
                        "base_original": original,
                        "base_age_zero": age_zero,
                        "base_drawdown_zero": drawdown_zero,
                        "base_return_zero": return_zero,
                        "base_all_state_zero": all_zero,
                        "age_effect": original - age_zero,
                        "drawdown_effect": original - drawdown_zero,
                        "return_effect": original - return_zero,
                        "joint_state_effect": original - all_zero,
                    }
                )

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
    return rows


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(args.repo)
    rows = []
    for market, key, seed in MARKETS:
        rows.extend(
            replay_market(
                runtime,
                args.repo,
                args.results_end,
                args.output.parent / "_probe_runtime",
                market,
                key,
                seed,
                args.device,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    summary = frame.groupby("market")[
        ["age_effect", "drawdown_effect", "return_effect", "joint_state_effect"]
    ].agg(["mean", "std", "median", "max"])
    print(summary.to_string())
    print(f"Rows: {len(frame)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
