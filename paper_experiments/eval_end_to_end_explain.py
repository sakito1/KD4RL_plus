import argparse
import contextlib
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import pandas as pd

from .metrics import (
    compute_financial_metrics,
    compute_trading_metrics,
    summarize_all,
    summarize_controller_alignment,
    summarize_inner_alpha,
)
from .trace_utils import (
    RunInfo,
    discover_runs,
    ensure_output_dirs,
    normalize_portfolio_trace,
    parse_seed_specs,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIOS = {
    "fixed_hrl": {"fixed_cycle": "max_hold", "disable_inner": False, "use_controller": False},
    "fixed_hrl_no_inner": {"fixed_cycle": "max_hold", "disable_inner": True, "use_controller": False},
    "full_controller": {"fixed_cycle": None, "disable_inner": False, "use_controller": True},
}


STAGE_SPECS = [
    ("Fixed HRL checkpoint", "hrl_fixed_best", "fixed_hrl"),
    ("Controller-PG checkpoint", "controller_best", "full_controller"),
    ("Final E2E checkpoint", "best_model", "full_controller"),
]


@contextlib.contextmanager
def temporary_argv(argv: Sequence[str]):
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def code_root_for_results_root(results_root: Path) -> Path:
    results_root = Path(results_root).resolve()
    if results_root.name == "end" and results_root.parent.name == "results":
        return results_root.parent.parent
    return ROOT


def import_hrl_stack(code_root: Path):
    code_root = Path(code_root).resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    import torch
    import run_hrl_training as runner
    import utils.config as runtime_config
    from Train.PPO_train import HRL_Networks, HRL_Trainer
    from agent import HRL_Buffer, HRL_PPO_Agent
    from env import PPO_Env
    from utils.Log import create_logger

    return torch, runner, runtime_config, HRL_Networks, HRL_Trainer, HRL_Buffer, HRL_PPO_Agent, PPO_Env, create_logger


def args_from_command_json(command_json: Path, *, device: str, output_dir: Path, code_root: Path):
    _, runner, *_ = import_hrl_stack(code_root)
    payload = json.loads(Path(command_json).read_text(encoding="utf-8"))
    command = payload["command"]
    script_idx = next(i for i, token in enumerate(command) if str(token).endswith("run_hrl_training.py"))
    args_list = list(command[script_idx + 1 :])
    with temporary_argv(["run_hrl_training.py", *args_list]):
        args = runner.parse_args()
    runner.normalize_training_schedule(args)
    args.device = str(device)
    args.cpu = str(device).lower() == "cpu"
    args.run_root = str(output_dir / "_probe_runtime")
    args.skip_test = True
    args.test_only_checkpoint = None
    return args


def build_loaded_trainer(run: RunInfo, *, output_dir: Path, device: str, results_root: Path):
    code_root = code_root_for_results_root(results_root)
    torch, runner, runtime_config, HRL_Networks, HRL_Trainer, HRL_Buffer, HRL_PPO_Agent, PPO_Env, create_logger = import_hrl_stack(code_root)
    args = args_from_command_json(run.command_json, device=device, output_dir=output_dir, code_root=code_root)
    _, config_module = runner.MARKET_CONFIGS[run.market]
    runner.apply_market_config(config_module)
    market_root = output_dir / "_probe_runtime" / run.market / f"seed_{run.seed}"
    runner.set_runtime_training_args(args, market_root, int(run.seed))
    runtime_config.min_hold = int(args.min_hold)
    runtime_config.max_hold = int(args.max_hold)
    runtime_config.fixed_cycle = int(args.max_hold)
    runtime_config.rollout_segments_by_stage = {
        stage: int(getattr(args, f"{stage}_rollout_segments"))
        for stage in runner.TRAINING_STAGES
    }
    runtime_config.rollout_update_steps_by_stage = {
        stage: runtime_config.fixed_cycle * segments
        for stage, segments in runtime_config.rollout_segments_by_stage.items()
    }
    runtime_config.rollout_update_steps = runtime_config.rollout_update_steps_by_stage["warmup_outer"]

    logger = create_logger(str(output_dir / "logs" / f"{run.market}_seed{run.seed}_eval"))
    runner.set_seed(int(run.seed), logger)
    env = PPO_Env(
        logger=logger,
        episode_len=args.episode_len,
        outer_window=args.outer_window,
        max_hold=int(args.max_hold),
        min_hold=int(args.min_hold),
        train_episodes_per_epoch=args.train_episodes_per_epoch,
        train_start_stride_days=args.train_start_stride_days,
        train_episode_to_end=args.train_episode_to_end,
    )
    runtime_config.device = env.device
    networks = HRL_Networks(args.ssm_dim, env.num_stocks, runtime_config).to(runtime_config.device)
    agent = HRL_PPO_Agent(networks, runtime_config)
    buffer = HRL_Buffer(
        capacity=3000,
        device=runtime_config.device,
        outer_reward_scale=getattr(runtime_config, "reward_scale_outer", 1.0),
    )
    trainer = HRL_Trainer(agent, env, buffer, runtime_config, logger)
    return trainer, args, torch


def load_checkpoint_into_trainer(trainer, torch_module, checkpoint_path: Path) -> bool:
    if not Path(checkpoint_path).exists():
        warnings.warn(f"missing checkpoint: {checkpoint_path}", RuntimeWarning)
        return False
    checkpoint = torch_module.load(Path(checkpoint_path), map_location=trainer.cfg.device)
    state = checkpoint.get("agent_net", checkpoint)
    trainer.agent.net.load_state_dict(state)
    trainer.agent.net.eval()
    return True


def _tensor_to_float(value, default=np.nan) -> float:
    if value is None:
        return float(default)
    if hasattr(value, "detach"):
        arr = value.detach().view(-1)
        if arr.numel() <= 0:
            return float(default)
        return float(arr[0].cpu().item())
    try:
        return float(value)
    except Exception:
        return float(default)


def _tensor_to_numpy(value) -> np.ndarray:
    return value.detach().view(-1).cpu().numpy().astype("float64")


def _normalize_np(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.ones_like(arr) / max(1, arr.size)
    return arr / total


def _future_curve(env, weights, *, horizon=20, current_weights=None) -> np.ndarray:
    start_day = int(env.day)
    stop_day = min(start_day + int(horizon), int(env.ratio.shape[1]))
    if stop_day <= start_day:
        return np.ones(1, dtype="float64")
    ratio_matrix = env.ratio[:, start_day:stop_day].detach().cpu().numpy().astype("float64")
    weights_arr = _normalize_np(_tensor_to_numpy(weights))
    growth = np.cumprod(ratio_matrix, axis=1)
    values = np.sum(weights_arr[:, None] * growth, axis=0)
    if current_weights is not None:
        current = _normalize_np(_tensor_to_numpy(current_weights))
        turnover = float(np.sum(np.abs(weights_arr - current)))
        values = values * max(0.0, 1.0 - turnover * float(getattr(env, "transaction_cost_pct", 0.0)))
    return np.concatenate([np.ones(1, dtype="float64"), values])


def _curve_mdd(curve) -> float:
    arr = np.asarray(curve, dtype="float64")
    if arr.size <= 1:
        return 0.0
    peaks = np.maximum.accumulate(np.maximum(arr, 1e-12))
    return float(np.max((peaks - arr) / peaks))


def collect_eval_trace(
    trainer,
    *,
    scenario: str,
    fixed_cycle=None,
    disable_inner: bool = False,
    test_max_days: int = None,
    random_switch_steps: Sequence[int] = None,
) -> Dict[str, pd.DataFrame]:
    import torch

    env = trainer.env
    env.set_mode("test")
    test_window = trainer._test_episode_window()
    if test_window is None:
        test_indices = list(getattr(env, "idx_map", {}).get("test", []))
        test_window = (int(test_indices[0]), int(test_indices[-1])) if test_indices else None
    if test_window and test_max_days:
        test_window = (test_window[0], min(test_window[0] + int(test_max_days), test_window[1]))
    obs = env.reset_at(*test_window) if test_window is not None else env.reset()
    spec = trainer._get_phase_spec("joint")
    portfolio_rows = []
    action_rows = []
    step_idx = 0
    last_switch_step = 0
    switch_count = 0
    free_switch_count = 0
    forced_switch_count = 0
    forced_hold_count = 0
    random_switch_steps = set(int(x) for x in (random_switch_steps or []))
    with torch.no_grad():
        while True:
            duration = step_idx - last_switch_step
            if fixed_cycle is not None:
                force_switch = 1 if step_idx == 0 or duration >= int(fixed_cycle) else 0
                force_locked = True
            else:
                force_switch, force_locked = trainer._compute_force_switch_locked(
                    spec=spec,
                    phase="joint",
                    step_idx=step_idx,
                    duration=duration,
                    is_train=False,
                    switch_schedule=None,
                    fixed_cycle=None,
                    current_segments=switch_count,
                    rollout_len=int(getattr(env, "current_episode_len", env.episode_len)),
                )
            free_eligible = force_switch is None
            if random_switch_steps and free_eligible:
                force_switch = 1 if step_idx in random_switch_steps else 0
                force_locked = True
            force_inner_zero = bool(disable_inner)

            out = trainer.agent.get_action(
                obs,
                mode="eval",
                force_switch=force_switch,
                force_inner_zero=force_inner_zero,
                force_locked=force_locked,
            )
            stats = None
            if free_eligible and not random_switch_steps:
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
            is_switch = bool(out["act_mon"].view(-1)[0].item() == 1)
            is_forced_switch = bool(is_switch and force_switch == 1)
            is_forced_hold = bool((not is_switch) and force_switch == 0)
            is_free_switch = bool(is_switch and free_eligible and not random_switch_steps)
            date_value = str(env.all_dates[int(env.day)].date()) if hasattr(env, "all_dates") else str(int(env.day))
            value_before = float(env.portfolio_value.item())
            weights_drift = obs["weights_drift"].detach()
            base_used = out["base_used"].detach()
            weights_exec = out["weights_exec"].detach()
            r_future = env.ratio[:, int(env.day)]
            base_log_return = float(
                torch.log(torch.sum(base_used.view(-1) * r_future).clamp_min(1e-8)).cpu().item()
            )
            exec_log_return = np.nan
            inner_alpha = np.nan
            turnover = float(torch.sum(torch.abs(weights_exec.view(-1) - weights_drift.view(-1))).cpu().item())
            cost_rate = turnover * float(getattr(env, "transaction_cost_pct", 0.0))
            hold_future_return_20 = np.nan
            switch_future_return_20 = np.nan
            hold_future_mdd_20 = np.nan
            switch_future_mdd_20 = np.nan
            switch_advantage_20 = np.nan
            hold_curve_20 = ""
            switch_curve_20 = ""
            if free_eligible:
                hold_exec = trainer._deterministic_inner_exec(obs, obs["base_drift"].detach(), weights_drift)
                switch_exec = trainer._deterministic_inner_exec(obs, out["act_out"].detach(), weights_drift)
                hold_curve = _future_curve(env, hold_exec, horizon=20, current_weights=weights_drift)
                switch_curve = _future_curve(env, switch_exec, horizon=20, current_weights=weights_drift)
                hold_future_return_20 = float(np.log(max(hold_curve[-1], 1e-12)))
                switch_future_return_20 = float(np.log(max(switch_curve[-1], 1e-12)))
                hold_future_mdd_20 = _curve_mdd(hold_curve)
                switch_future_mdd_20 = _curve_mdd(switch_curve)
                switch_advantage_20 = switch_future_return_20 - hold_future_return_20
                hold_curve_20 = json.dumps([float(x) for x in hold_curve], ensure_ascii=False)
                switch_curve_20 = json.dumps([float(x) for x in switch_curve], ensure_ascii=False)
            next_obs, _, done, info = env.step(
                weights_exec,
                base_used,
                outer_action=out["act_out"].detach(),
                is_switch=is_switch,
            )
            value_after = float(info["portfolio_value"])
            daily_simple_return = value_after / max(value_before, 1e-12) - 1.0
            exec_log_return = float(np.log1p(daily_simple_return))
            inner_alpha = exec_log_return - base_log_return if np.isfinite(base_log_return) else np.nan
            if is_switch:
                switch_count += 1
                last_switch_step = step_idx
                if is_forced_switch:
                    forced_switch_count += 1
                if is_free_switch:
                    free_switch_count += 1
            elif is_forced_hold:
                forced_hold_count += 1
            controller_adv = (
                _tensor_to_float(info.get("controller_switch_advantage"), np.nan)
                if free_eligible
                else np.nan
            )
            exit_prob = _tensor_to_float(stats.get("exit_prob") if stats else None, np.nan)
            policy_logit = _tensor_to_float(stats.get("policy_logit") if stats else None, np.nan)
            portfolio_rows.append(
                {
                    "date": date_value,
                    "step": step_idx,
                    "portfolio_value": value_after,
                    "portfolio_value_before": value_before,
                    "daily_simple_return": daily_simple_return,
                    "daily_log_return": exec_log_return,
                    "base_log_return": base_log_return,
                    "inner_alpha": inner_alpha,
                    "turnover": turnover,
                    "cost_rate": cost_rate,
                    "is_switch": int(is_switch),
                    "is_free_switch": int(is_free_switch),
                    "is_forced_switch": int(is_forced_switch),
                    "is_forced_hold": int(is_forced_hold),
                    "holding_duration": duration,
                    "switch_count": switch_count,
                    "free_switch_count": free_switch_count,
                }
            )
            action_rows.append(
                {
                    "date": date_value,
                    "step": step_idx,
                    "duration_before_decision": duration,
                    "decision_type": "free_decision" if free_eligible else ("forced_switch" if force_switch == 1 else "forced_hold"),
                    "is_locked": int(not free_eligible),
                    "is_switch": int(is_switch),
                    "is_free_switch": int(is_free_switch),
                    "is_forced_switch": int(is_forced_switch),
                    "is_forced_hold": int(is_forced_hold),
                    "exit_prob": exit_prob,
                    "policy_logit": policy_logit,
                    "controller_switch_advantage": controller_adv,
                    "controller_hold_return_target": _tensor_to_float(info.get("controller_hold_return_target"), np.nan),
                    "controller_hold_mdd_target": _tensor_to_float(info.get("controller_hold_mdd_target"), np.nan),
                    "controller_switch_label": _tensor_to_float(info.get("controller_switch_label"), np.nan),
                    "controller_sup_weight": _tensor_to_float(info.get("controller_sup_weight"), np.nan),
                    "hold_future_return_20": hold_future_return_20,
                    "switch_future_return_20": switch_future_return_20,
                    "hold_future_mdd_20": hold_future_mdd_20,
                    "switch_future_mdd_20": switch_future_mdd_20,
                    "switch_advantage_20": switch_advantage_20,
                    "hold_curve_20": hold_curve_20,
                    "switch_curve_20": switch_curve_20,
                    "base_log_return": base_log_return,
                    "exec_log_return": exec_log_return,
                    "inner_alpha": inner_alpha,
                    "turnover": turnover,
                    "cost_rate": cost_rate,
                }
            )
            if done:
                break
            obs = next_obs
            step_idx += 1
    portfolio_df = normalize_portfolio_trace(pd.DataFrame(portfolio_rows))
    action_df = pd.DataFrame(action_rows)
    switch_events = build_switch_events(portfolio_df, action_df)
    return {"portfolio": portfolio_df, "actions": action_df, "switch_events": switch_events}


def build_switch_events(portfolio_df: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    if action_df.empty:
        return pd.DataFrame()
    events = action_df[pd.to_numeric(action_df.get("is_free_switch"), errors="coerce").fillna(0) == 1].copy()
    if events.empty:
        return events
    rows = []
    for event_id, (_, row) in enumerate(events.iterrows(), start=1):
        step = int(row["step"])
        pre = portfolio_df[(portfolio_df["step"] >= step - 20) & (portfolio_df["step"] <= step)]
        post = portfolio_df[(portfolio_df["step"] > step) & (portfolio_df["step"] <= step + 20)]
        pre_values = pre["portfolio_value"].to_numpy(dtype="float64")
        post_values = post["portfolio_value"].to_numpy(dtype="float64")
        pre_return = float(pre_values[-1] / max(pre_values[0], 1e-12) - 1.0) if len(pre_values) > 1 else np.nan
        post_return = float(post_values[-1] / max(pre_values[-1] if len(pre_values) else 1.0, 1e-12) - 1.0) if len(post_values) else np.nan
        rows.append(
            {
                "event_id": event_id,
                "date": row["date"],
                "step": step,
                "holding_duration": row.get("duration_before_decision", np.nan),
                "exit_prob": row.get("exit_prob", np.nan),
                "policy_logit": row.get("policy_logit", np.nan),
                "pre_return_20": pre_return,
                "pre_drawdown_20": compute_financial_metrics(pd.DataFrame({"portfolio_value": pre_values})).get("max_drawdown", np.nan),
                "post_actual_return_20": post_return,
                "post_hold_return_20": row.get("hold_future_return_20", np.nan),
                "post_switch_return_20": row.get("switch_future_return_20", np.nan),
                "post_actual_mdd_20": compute_financial_metrics(pd.DataFrame({"portfolio_value": post_values})).get("max_drawdown", np.nan),
                "post_hold_mdd_20": row.get("hold_future_mdd_20", np.nan),
                "post_switch_mdd_20": row.get("switch_future_mdd_20", np.nan),
                "avoided_loss_20": row.get("switch_advantage_20", np.nan),
                "controller_switch_advantage": row.get("controller_switch_advantage", np.nan),
            }
        )
    return pd.DataFrame(rows)


def scenario_output_name(market: str, seed: int, scenario: str) -> str:
    return f"{market}_seed{seed}_{scenario}"


def save_trace_bundle(paths: Dict[str, Path], run: RunInfo, scenario: str, bundle: Dict[str, pd.DataFrame], metrics: dict):
    prefix = scenario_output_name(run.market, run.seed, scenario)
    bundle["portfolio"].to_csv(paths["traces"] / f"{prefix}_portfolio.csv", index=False)
    bundle["actions"].to_csv(paths["traces"] / f"{prefix}_actions.csv", index=False)
    bundle["switch_events"].to_csv(paths["traces"] / f"{prefix}_switch_events.csv", index=False)
    write_json(paths["metrics"] / f"{prefix}_metrics.json", metrics)


def evaluate_scenario(
    trainer,
    run: RunInfo,
    *,
    scenario: str,
    test_max_days: int = None,
    random_switch_steps: Sequence[int] = None,
) -> Dict[str, pd.DataFrame]:
    spec = SCENARIOS[scenario]
    fixed_cycle = int(getattr(trainer.cfg, "max_hold", 30)) if spec["fixed_cycle"] == "max_hold" else None
    return collect_eval_trace(
        trainer,
        scenario=scenario,
        fixed_cycle=fixed_cycle,
        disable_inner=bool(spec["disable_inner"]),
        test_max_days=test_max_days,
        random_switch_steps=random_switch_steps,
    )


def trace_metrics(run: RunInfo, scenario: str, bundle: Dict[str, pd.DataFrame], extra: dict = None) -> dict:
    metrics = {
        "market": run.market,
        "seed": run.seed,
        "scenario": scenario,
        "status": "ok",
    }
    metrics.update(summarize_all(bundle["portfolio"]))
    if not bundle["actions"].empty:
        metrics.update(summarize_controller_alignment(bundle["actions"]))
    if extra:
        metrics.update(extra)
    return metrics


def random_switch_runs(
    trainer,
    run: RunInfo,
    full_bundle: Dict[str, pd.DataFrame],
    *,
    random_runs: int,
    test_max_days: int = None,
    seed: int = 1234,
) -> Sequence[dict]:
    actions = full_bundle["actions"]
    eligible = actions[actions["decision_type"] == "free_decision"]["step"].astype(int).tolist()
    free_count = int(pd.to_numeric(actions.get("is_free_switch"), errors="coerce").fillna(0).sum())
    rng = random.Random(int(seed) + int(run.seed))
    results = []
    for run_id in range(int(random_runs)):
        if eligible and free_count > 0:
            selected = rng.sample(eligible, k=min(free_count, len(eligible)))
        else:
            selected = []
        bundle = collect_eval_trace(
            trainer,
            scenario="random_switch_matched_count",
            fixed_cycle=None,
            disable_inner=False,
            test_max_days=test_max_days,
            random_switch_steps=selected,
        )
        metrics = trace_metrics(
            run,
            "random_switch_matched_count",
            bundle,
            {"random_run_id": run_id, "random_seed": int(seed) + int(run.seed) + run_id},
        )
        results.append({"bundle": bundle, "metrics": metrics})
    return results


def run_for_one_seed(run: RunInfo, *, args, paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    trainer, _, torch_module = build_loaded_trainer(run, output_dir=Path(args.output_dir), device=args.device, results_root=Path(args.results_root))
    all_metric_rows = []
    stage_rows = []
    scenario_bundles: Dict[str, Dict[str, pd.DataFrame]] = {}

    for stage_name, checkpoint_name, scenario in STAGE_SPECS:
        ckpt = run.checkpoints[checkpoint_name]
        if not ckpt.exists:
            row = {
                "market": run.market,
                "seed": run.seed,
                "stage": stage_name,
                "checkpoint_name": checkpoint_name,
                "scenario": scenario,
                "status": "missing",
            }
            stage_rows.append(row)
            all_metric_rows.append(row)
            continue
        load_checkpoint_into_trainer(trainer, torch_module, ckpt.path)
        bundle = evaluate_scenario(trainer, run, scenario=scenario, test_max_days=args.test_max_days)
        name = f"stage_{checkpoint_name}_{scenario}"
        save_trace_bundle(paths, run, name, bundle, trace_metrics(run, name, bundle, {"stage": stage_name, "checkpoint_name": checkpoint_name}))
        metrics = trace_metrics(run, scenario, bundle, {"stage": stage_name, "checkpoint_name": checkpoint_name})
        stage_rows.append(metrics)
        all_metric_rows.append(metrics)
        if checkpoint_name == "best_model" and scenario == "full_controller":
            scenario_bundles["full_controller"] = bundle

    best_ckpt = run.checkpoints["best_model"]
    if best_ckpt.exists:
        load_checkpoint_into_trainer(trainer, torch_module, best_ckpt.path)
        for scenario in ["fixed_hrl_no_inner", "fixed_hrl", "full_controller"]:
            bundle = evaluate_scenario(trainer, run, scenario=scenario, test_max_days=args.test_max_days)
            scenario_bundles[scenario] = bundle
            metrics = trace_metrics(run, scenario, bundle, {"checkpoint_name": "best_model"})
            save_trace_bundle(paths, run, scenario, bundle, metrics)
            all_metric_rows.append(metrics)
        if int(args.random_runs) > 0:
            random_results = random_switch_runs(
                trainer,
                run,
                scenario_bundles["full_controller"],
                random_runs=int(args.random_runs),
                test_max_days=args.test_max_days,
                seed=int(args.random_seed),
            )
            for result in random_results:
                rid = int(result["metrics"]["random_run_id"])
                scenario = f"random_switch_matched_count_{rid:03d}"
                save_trace_bundle(paths, run, scenario, result["bundle"], result["metrics"])
                all_metric_rows.append(result["metrics"])

    pd.DataFrame(stage_rows).to_csv(paths["metrics"] / f"{run.market}_seed{run.seed}_stage_progression.csv", index=False)
    pd.DataFrame(all_metric_rows).to_csv(paths["metrics"] / f"{run.market}_seed{run.seed}_all_metrics.csv", index=False)
    return scenario_bundles


def aggregate_outputs(paths: Dict[str, Path]):
    metric_files = sorted(paths["metrics"].glob("*_seed*_all_metrics.csv"))
    frames = [pd.read_csv(path) for path in metric_files if path.stat().st_size > 0]
    all_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_metrics.to_csv(paths["metrics"] / "all_metrics.csv", index=False)
    if all_metrics.empty:
        return
    stage = all_metrics[all_metrics.get("stage").notna()] if "stage" in all_metrics else pd.DataFrame()
    stage.to_csv(paths["metrics"] / "stage_progression.csv", index=False)
    non_stage = all_metrics[all_metrics.get("stage").isna()] if "stage" in all_metrics else all_metrics
    ablation = non_stage[non_stage["scenario"].isin(["fixed_hrl_no_inner", "fixed_hrl", "full_controller"])]
    ablation.to_csv(paths["metrics"] / "inference_ablation.csv", index=False)
    inner_rows = []
    for trace_path in paths["traces"].glob("*_portfolio.csv"):
        if "_stage_" in trace_path.name:
            continue
        if "_fixed_hrl" not in trace_path.name and "_full_controller" not in trace_path.name:
            continue
        df = pd.read_csv(trace_path)
        parts = trace_path.stem.replace("_portfolio", "").split("_")
        scenario = "_".join(parts[2:])
        row = {"market": parts[0], "seed": int(parts[1].replace("seed", "")), "scenario": scenario}
        row.update(summarize_inner_alpha(df))
        inner_rows.append(row)
    pd.DataFrame(inner_rows).to_csv(paths["metrics"] / "inner_alpha_summary.csv", index=False)
    align_rows = []
    event_frames = []
    for action_path in paths["traces"].glob("*_full_controller_actions.csv"):
        if "_stage_" in action_path.name:
            continue
        df = pd.read_csv(action_path)
        parts = action_path.stem.replace("_actions", "").split("_")
        row = {"market": parts[0], "seed": int(parts[1].replace("seed", "")), "scenario": "full_controller"}
        row.update(summarize_controller_alignment(df))
        align_rows.append(row)
    for event_path in paths["traces"].glob("*_full_controller_switch_events.csv"):
        if "_stage_" in event_path.name:
            continue
        df = pd.read_csv(event_path)
        if not df.empty:
            parts = event_path.stem.replace("_switch_events", "").split("_")
            df.insert(0, "market", parts[0])
            df.insert(1, "seed", int(parts[1].replace("seed", "")))
            event_frames.append(df)
    pd.DataFrame(align_rows).to_csv(paths["metrics"] / "switch_alignment_summary.csv", index=False)
    (pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()).to_csv(paths["metrics"] / "switch_events.csv", index=False)
    random_df = non_stage[non_stage["scenario"] == "random_switch_matched_count"].copy()
    random_df.to_csv(paths["metrics"] / "random_switch_summary.csv", index=False)
    random_comparison_rows = []
    for (market, seed), group in non_stage.groupby(["market", "seed"]):
        full = group[group["scenario"] == "full_controller"]
        rand = group[group["scenario"] == "random_switch_matched_count"]
        fixed = group[group["scenario"] == "fixed_hrl"]
        if full.empty or rand.empty:
            continue
        full_row = full.iloc[0]
        random_comparison_rows.append(
            {
                "market": market,
                "seed": seed,
                "full_total_return": full_row.get("total_return"),
                "random_mean_total_return": rand["total_return"].mean(),
                "random_std_total_return": rand["total_return"].std(),
                "full_sharpe": full_row.get("sharpe"),
                "random_mean_sharpe": rand["sharpe"].mean(),
                "random_std_sharpe": rand["sharpe"].std(),
                "full_mdd": full_row.get("max_drawdown"),
                "random_mean_mdd": rand["max_drawdown"].mean(),
                "random_std_mdd": rand["max_drawdown"].std(),
                "full_switch_count": full_row.get("switch_count"),
                "random_mean_switch_count": rand["switch_count"].mean(),
                "fixed_total_return": fixed.iloc[0].get("total_return") if not fixed.empty else np.nan,
                "full_percentile_by_return": float((rand["total_return"] <= full_row.get("total_return")).mean()),
                "full_percentile_by_sharpe": float((rand["sharpe"] <= full_row.get("sharpe")).mean()),
            }
        )
    pd.DataFrame(random_comparison_rows).to_csv(paths["metrics"] / "random_switch_comparison.csv", index=False)


def dry_run(args, runs: Sequence[RunInfo], paths: Dict[str, Path]) -> None:
    rows = []
    for run in runs:
        for checkpoint in run.checkpoints.values():
            rows.append(
                {
                    "market": run.market,
                    "seed": run.seed,
                    "run_dir": str(run.run_dir),
                    "command_json": str(run.command_json),
                    "checkpoint": checkpoint.name,
                    "checkpoint_path": str(checkpoint.path),
                    "exists": checkpoint.exists,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(paths["metrics"] / "dry_run_manifest.csv", index=False)
    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-only paper explanation experiments for end-to-end HRL/controller.")
    parser.add_argument("--results_root", default="/home/tongwenxuan/KD4RL_plus/results/end")
    parser.add_argument("--output_dir", default="paper_experiments_outputs/end_to_end_explain")
    parser.add_argument("--markets", nargs="+", default=["sh", "nas"])
    parser.add_argument("--seeds", nargs="+", default=["sh:90", "nas:49"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--test_max_days", type=int, default=None)
    parser.add_argument("--random_runs", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_output_dirs(Path(args.output_dir))
    seed_map = parse_seed_specs(args.seeds, args.markets)
    runs = discover_runs(Path(args.results_root), markets=args.markets, seed_map=seed_map)
    if args.dry_run:
        dry_run(args, runs, paths)
        return
    for run in runs:
        if not run.command_json.exists():
            warnings.warn(f"skip run with missing command json: {run.command_json}", RuntimeWarning)
            continue
        run_for_one_seed(run, args=args, paths=paths)
    aggregate_outputs(paths)
    try:
        from .plot_end_to_end_explain import main_from_paths as plot_main
        from .table_end_to_end_explain import main_from_paths as table_main

        plot_main(Path(args.output_dir), paths["figures"])
        table_main(Path(args.output_dir), paths["tables"])
    except Exception as exc:
        warnings.warn(f"plot/table generation failed: {exc}", RuntimeWarning)


if __name__ == "__main__":
    main()
