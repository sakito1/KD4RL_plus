"""Train EmbMonitor-ZH with DQN while reusing frozen KD4RL HRL policies."""

import argparse
import copy
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import utils.config as runtime_config
from agent import DQNMonitorAgent, DQNReplayBuffer
from env import PPO_Env
from Train.PPO_train import HRL_Networks, set_seed
from utils.Log import create_logger


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_BOOK = os.path.join(ROOT, "result.xlsx")
REFERENCE_BOOK = os.path.join(ROOT, "pretrained_assets", "kd4rl", "reference", "result.xlsx")

MARKETS = {
    "nas100": {
        "config": "utils.config_Nas",
        "source_seed": 47,
        "checkpoint": "pretrained_assets/kd4rl/nas100/ppo/seed_47/best_model.pth",
        "checkpoint_sha256": "dd7d9c65e6d50c66270f49b11c1a11b61cdec803ede3de1bd2796beea6422e6b",
        "ssm_path": "Dataset/Nas100数据/feature_ssm",
        "ssm_input_files": 78,
        "reference_rows": (150, 163),
        "sota_sharpe": 1.0987830427195604,
        "sota_return": 2.6439,
        "smoke_ranges": {
            "train": ("2016-01-04", "2017-12-29"),
            "val": ("2019-01-02", "2019-06-28"),
            "test": ("2020-04-23", "2020-08-31"),
        },
    },
    "sh": {
        "config": "utils.config_SH",
        "source_seed": 77,
        "checkpoint": "pretrained_assets/kd4rl/sh/ppo/seed_77/best_model.pth",
        "checkpoint_sha256": "e84aa880e4fc5b86f6fa234c692dc070847909ac6e4bb28acaa5a517d3ca7aa9",
        "ssm_path": "Dataset/沪深数据/feature_ssm",
        "ssm_input_files": 106,
        "reference_rows": (174, 187),
        "sota_sharpe": 1.03,
        "sota_return": 2.1281,
        "smoke_ranges": {
            "train": ("2016-01-04", "2017-12-28"),
            "val": ("2019-01-02", "2019-06-28"),
            "test": ("2020-01-02", "2020-06-30"),
        },
    },
}


def apply_market_config(market):
    module = __import__(MARKETS[market]["config"], fromlist=["dummy"])
    for name, value in vars(module).items():
        if not name.startswith("__"):
            setattr(runtime_config, name, copy.deepcopy(value))


def compute_metrics(history):
    values = pd.Series(history, dtype=float)
    daily = values.pct_change().fillna(0.0)
    annual_return = float(daily.mean() * 252)
    annual_vol = float(daily.std() * np.sqrt(252))
    max_dd = float(((values.cummax() - values) / values.cummax()).max())
    return {
        "total_ret": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "ann_ret": annual_return,
        "ann_vol": annual_vol,
        "sharpe": float(annual_return / (annual_vol + 1e-8)),
        "max_dd": max_dd,
        "cr": float(annual_return / (max_dd + 1e-8)),
        "final_value": float(values.iloc[-1]),
    }


def epsilon_at(step, start, end, decay_steps):
    ratio = min(float(step) / float(max(decay_steps, 1)), 1.0)
    return float(start + (end - start) * ratio)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_market_assets(market):
    settings = MARKETS[market]
    checkpoint = os.path.join(ROOT, settings["checkpoint"])
    actual_hash = sha256_file(checkpoint)
    if actual_hash != settings["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch for {market}: {checkpoint}")
    ssm_dir = Path(ROOT) / settings["ssm_path"]
    ssm_inputs = list(ssm_dir.glob("*.csv")) + list(ssm_dir.glob("*_ssm3_states.pt"))
    if len(ssm_inputs) != settings["ssm_input_files"]:
        raise RuntimeError(
            f"SSM input contract mismatch for {market}: "
            f"expected={settings['ssm_input_files']} found={len(ssm_inputs)}"
        )
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": actual_hash,
        "ssm_path": str(ssm_dir),
        "ssm_input_files": len(ssm_inputs),
    }


def build_environment(market, mode, logger):
    if mode == "smoke":
        ranges = MARKETS[market]["smoke_ranges"]
        return PPO_Env(
            logger=logger,
            episode_len=40,
            train_date_range=ranges["train"],
            val_date_range=ranges["val"],
            test_date_range=ranges["test"],
        )
    return PPO_Env(logger=logger)


def load_frozen_hrl(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("agent_net", checkpoint)
    reusable = {
        key: value for key, value in source.items()
        if key.startswith("outer.") or key.startswith("inner.")
    }
    missing, unexpected = model.load_state_dict(reusable, strict=False)
    unexpected = [key for key in unexpected if key.startswith(("outer.", "inner."))]
    missing_required = [
        key for key in missing if key.startswith(("outer.", "inner."))
    ]
    if missing_required or unexpected:
        raise RuntimeError(
            f"Frozen HRL mismatch missing={missing_required[:4]} unexpected={unexpected[:4]}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.outer.eval()
    model.inner.eval()


def long_hold_penalty(days, target_days):
    excess = max(float(days) - float(target_days), 0.0) / float(max(target_days, 1))
    return excess ** 2


def build_candidates(env, frozen_hrl, obs, lambda_cost=0.0, lambda_long=0.0):
    """Build hold and switch candidate executions without changing environment state."""
    with torch.inference_mode():
        weights = obs["weights_drift"]
        hold_base = obs["base_drift"]
        switch_base, _, _, _, _ = frozen_hrl.outer.pi(
            obs["outer_state"], weights, deterministic=True
        )
        value_hold = frozen_hrl.outer.value(obs["outer_state"], hold_base).squeeze(-1)
        value_switch = frozen_hrl.outer.value(obs["outer_state"], switch_base).squeeze(-1)
        action_args = {
            "alpha": float(getattr(runtime_config, "inner_max_boundary", 1.0)),
            "deterministic": True,
        }
        hold_exec, _, _, _, _ = frozen_hrl.inner.build_inner_action_simple(
            obs["inner_state"], hold_base, weights, **action_args
        )
        switch_exec, _, _, _, _ = frozen_hrl.inner.build_inner_action_simple(
            obs["inner_state"], switch_base, weights, **action_args
        )
        cost_hold = torch.sum(torch.abs(hold_exec - weights), dim=1) * env.transaction_cost_pct
        cost_switch = torch.sum(torch.abs(switch_exec - weights), dim=1) * env.transaction_cost_pct
        hold_penalty = long_hold_penalty(env.t_held + 1, env.max_hold)
        switch_penalty = long_hold_penalty(1, env.max_hold)
        utility_hold = (
            value_hold - float(lambda_cost) * cost_hold - float(lambda_long) * hold_penalty
        )
        utility_switch = (
            value_switch - float(lambda_cost) * cost_switch - float(lambda_long) * switch_penalty
        )
        candidate_costs = torch.stack(
            [cost_hold, cost_switch, cost_switch - cost_hold], dim=1
        )
    enriched_obs = dict(obs)
    enriched_obs["candidate_switch_base"] = switch_base
    enriched_obs["candidate_costs"] = candidate_costs
    candidate = {
        "hold_base": hold_base,
        "switch_base": switch_base,
        "hold_exec": hold_exec,
        "switch_exec": switch_exec,
        "value_hold": float(value_hold.item()),
        "value_switch": float(value_switch.item()),
        "cost_hold": float(cost_hold.item()),
        "cost_switch": float(cost_switch.item()),
        "long_penalty_hold": float(hold_penalty),
        "long_penalty_switch": float(switch_penalty),
        "utility_hold": float(utility_hold.item()),
        "utility_switch": float(utility_switch.item()),
    }
    return enriched_obs, candidate


def execute_step(env, candidate, action):
    base_used = candidate["switch_base"] if action == 1 else candidate["hold_base"]
    weights_exec = candidate["switch_exec"] if action == 1 else candidate["hold_exec"]
    reward = (
        candidate["utility_switch"] - candidate["utility_hold"]
        if action == 1 else candidate["utility_hold"] - candidate["utility_switch"]
    )
    with torch.inference_mode():
        next_obs, _, done, info = env.step(
            weights_exec,
            base_used,
            outer_action=candidate["switch_base"],
            is_switch=(action == 1),
            calculate_outer_reward=False,
        )
    return next_obs, reward, done, info, candidate["utility_switch"] - candidate["utility_hold"]


def run_episode(env, frozen_hrl, dqn=None, replay=None, train=False, fixed_cycle=None,
                oracle=False, cfg=None, utility_cfg=None):
    utility_cfg = utility_cfg or {"lambda_cost": 0.0, "lambda_long": 0.0}
    obs, candidate = build_candidates(env, frozen_hrl, env.reset(), **utility_cfg)
    history = [float(env.portfolio_value.item())]
    rewards, advantages, turnovers, costs = [], [], [], []
    estimated_cost_hold, estimated_cost_switch, long_penalties = [], [], []
    switches = 0
    hold_age = 0
    holding_lengths = []
    losses = []

    while True:
        if oracle:
            action = int(candidate["utility_switch"] > candidate["utility_hold"])
        elif fixed_cycle is None:
            epsilon = epsilon_at(
                cfg["env_steps"], cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay"]
            ) if train else 0.0
            action = dqn.select_action(obs, epsilon=epsilon)
        else:
            action = int(hold_age == 0 or hold_age >= fixed_cycle)
        next_raw_obs, reward, done, info, switch_adv = execute_step(env, candidate, action)
        next_obs, next_candidate = build_candidates(
            env, frozen_hrl, next_raw_obs, **utility_cfg
        )
        if train:
            replay.store(obs, action, reward * cfg["reward_scale"], next_obs, done)
            cfg["env_steps"] += 1
            if (
                len(replay) >= cfg["warmup"]
                and cfg["env_steps"] % cfg["train_frequency"] == 0
            ):
                loss = dqn.update(replay, cfg["batch_size"])
                if loss is not None:
                    losses.append(loss)

        if action == 1:
            switches += 1
            if hold_age > 0:
                holding_lengths.append(hold_age)
            hold_age = 1
        else:
            hold_age += 1
        rewards.append(reward)
        advantages.append(switch_adv)
        estimated_cost_hold.append(candidate["cost_hold"])
        estimated_cost_switch.append(candidate["cost_switch"])
        long_penalties.append(
            candidate["long_penalty_switch"] if action == 1 else candidate["long_penalty_hold"]
        )
        turnovers.append(float(info["rewards"]["turnover"]))
        costs.append(float(info["rewards"]["transaction_cost"]))
        history.append(float(info["portfolio_value"]))
        obs, candidate = next_obs, next_candidate
        if done:
            if hold_age > 0:
                holding_lengths.append(hold_age)
            break

    result = compute_metrics(history)
    result.update({
        "switch_count": int(switches),
        "avg_hold_days": float(np.mean(holding_lengths)) if holding_lengths else 0.0,
        "turnover": float(np.sum(turnovers)),
        "transaction_cost": float(np.sum(costs)),
        "mean_cf_reward": float(np.mean(rewards)),
        "mean_switch_advantage": float(np.mean(advantages)),
        "switch_advantage_positive_rate": float(np.mean(np.asarray(advantages) > 0)),
        "mean_estimated_cost_hold": float(np.mean(estimated_cost_hold)),
        "mean_estimated_cost_switch": float(np.mean(estimated_cost_switch)),
        "mean_long_hold_penalty": float(np.mean(long_penalties)),
        "dqn_loss": float(np.mean(losses)) if losses else None,
    })
    return result, history


def reference_frame():
    raw = pd.read_excel(REFERENCE_BOOK, header=None)
    frames = []
    for market, settings in MARKETS.items():
        start, end = settings["reference_rows"]
        frame = raw.iloc[start - 1:end, :7].copy()
        frame.columns = ["model", "return", "ar", "vol", "sharpe", "maxdd", "cr"]
        frame["market"] = market
        frames.append(frame[["market", "model", "return", "ar", "vol", "sharpe", "maxdd", "cr"]])
    return pd.concat(frames, ignore_index=True)


def update_workbook(run_row=None, final_rows=None, oracle_rows=None):
    reference = reference_frame()
    if os.path.exists(RESULT_BOOK):
        try:
            existing_runs = pd.read_excel(RESULT_BOOK, sheet_name="dqn_runs")
        except Exception:
            existing_runs = pd.DataFrame()
    else:
        existing_runs = pd.DataFrame()
    if run_row is not None:
        existing_runs = pd.concat([existing_runs, pd.DataFrame([run_row])], ignore_index=True)
    runs = existing_runs.drop_duplicates(subset=["run_id", "market"], keep="last")
    pretrained = pd.DataFrame([
        {"market": "nas100", "source_seed": 47, "fixed_return": 2.21303759765625,
         "fixed_sharpe": 1.0987830427195604},
        {"market": "sh", "source_seed": 77, "fixed_return": 1.0847807617187502,
         "fixed_sharpe": 0.7574401096169164},
    ])
    if os.path.exists(RESULT_BOOK):
        try:
            old_final = pd.read_excel(RESULT_BOOK, sheet_name="final_comparison")
        except Exception:
            old_final = pd.DataFrame()
    else:
        old_final = pd.DataFrame()
    final = pd.concat([old_final, pd.DataFrame(final_rows or [])], ignore_index=True)
    final = final.drop_duplicates(subset=["run_id", "market", "model"], keep="last")
    if os.path.exists(RESULT_BOOK):
        try:
            old_oracle = pd.read_excel(RESULT_BOOK, sheet_name="oracle_runs")
        except Exception:
            old_oracle = pd.DataFrame()
    else:
        old_oracle = pd.DataFrame()
    oracle = pd.concat([old_oracle, pd.DataFrame(oracle_rows or [])], ignore_index=True)
    if not oracle.empty:
        oracle = oracle.drop_duplicates(
            subset=["run_id", "market", "lambda_cost", "lambda_long"], keep="last"
        )
    with pd.ExcelWriter(RESULT_BOOK, engine="openpyxl") as writer:
        reference.to_excel(writer, sheet_name="reference_sota", index=False)
        pretrained.to_excel(writer, sheet_name="pretrained_hrl", index=False)
        runs.to_excel(writer, sheet_name="dqn_runs", index=False)
        oracle.to_excel(writer, sheet_name="oracle_runs", index=False)
        final.to_excel(writer, sheet_name="final_comparison", index=False)


def run_oracle_stage(env, frozen_hrl, args, market_cfg, provenance, out_dir):
    fixed_cycle = int(getattr(runtime_config, "max_hold", 60))
    env.set_mode("val")
    fixed_result, _ = run_episode(env, frozen_hrl, fixed_cycle=fixed_cycle)
    rows = []
    results = []
    quick_grid = [(0.0, 0.0), (5.0, 5e-4), (20.0, 1e-3)]
    full_grid = list(itertools.product(
        [0.0, 1.0, 5.0, 10.0, 20.0],
        [0.0, 1e-4, 5e-4, 1e-3],
    ))
    grid = quick_grid if args.oracle_grid == "quick" else full_grid
    for lambda_cost, lambda_long in grid:
        utility_cfg = {"lambda_cost": lambda_cost, "lambda_long": lambda_long}
        oracle_result, _ = run_episode(
            env, frozen_hrl, oracle=True, utility_cfg=utility_cfg
        )
        passed = (
            oracle_result["sharpe"] > fixed_result["sharpe"]
            and oracle_result["switch_count"] <= 2 * fixed_result["switch_count"]
        )
        record = {
            "run_id": args.run_id,
            "market": args.market,
            "mode": args.mode,
            "reward_source": "outer_value",
            "lambda_cost": lambda_cost,
            "lambda_long": lambda_long,
            "oracle_pass": bool(passed),
            "fixed_sharpe": fixed_result["sharpe"],
            "fixed_switch_count": fixed_result["switch_count"],
            "val_sharpe": oracle_result["sharpe"],
            "val_return": oracle_result["total_ret"],
            "max_dd": oracle_result["max_dd"],
            "switch_count": oracle_result["switch_count"],
            "turnover": oracle_result["turnover"],
            "transaction_cost": oracle_result["transaction_cost"],
        }
        rows.append(record)
        results.append({"config": utility_cfg, "passed": bool(passed), "metrics": oracle_result})
    passing = [item for item in results if item["passed"]]
    champion = max(passing or results, key=lambda item: item["metrics"]["sharpe"])
    summary = {
        "run_id": args.run_id,
        "market": args.market,
        "stage": "oracle",
        "mode": args.mode,
        "oracle_grid": args.oracle_grid,
        "fixed_cycle_validation": fixed_result,
        "oracle_pass": bool(passing),
        "champion": champion,
        "grid": results,
        "checkpoint": provenance["checkpoint"],
        "data_provenance": provenance,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    update_workbook(oracle_rows=rows)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    parser.add_argument("--stage", choices=["oracle", "dqn"], default="dqn")
    parser.add_argument("--oracle-grid", choices=["quick", "full"], default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--epsilon-decay", type=int, default=20000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--lambda-long", type=float, default=0.0)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.oracle_grid is None:
        args.oracle_grid = "quick" if args.mode == "smoke" else "full"
    if args.stage == "oracle" and args.mode == "full" and args.oracle_grid != "full":
        parser.error("Full oracle validation must use --oracle-grid full.")
    if args.train_frequency < 1 or args.validate_every < 1:
        parser.error("--train-frequency and --validate-every must be positive integers.")

    os.chdir(ROOT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DQN full experiment requires CUDA but torch cannot access the GPU.")
    device = torch.device(args.device)
    market_cfg = MARKETS[args.market]
    provenance = verify_market_assets(args.market)
    apply_market_config(args.market)
    runtime_config.device = device
    runtime_config.trade_num = 10
    runtime_config.seed = market_cfg["source_seed"]
    set_seed(runtime_config.seed, create_logger(os.path.join("results", "dqn_monitor", "setup")))

    run_id = args.run_id or (
        f"{args.stage}_{args.mode}"
        if args.stage == "oracle"
        else f"{args.mode}_ep{args.episodes or (2 if args.mode == 'smoke' else 20)}"
    )
    args.run_id = run_id
    out_dir = os.path.join("results", "dqn_monitor", args.market, run_id)
    os.makedirs(out_dir, exist_ok=True)
    logger = create_logger(out_dir)
    env = build_environment(args.market, args.mode, logger)
    frozen_hrl = HRL_Networks(16, env.num_stocks, runtime_config).to(device)
    checkpoint = provenance["checkpoint"]
    load_frozen_hrl(frozen_hrl, checkpoint, device)
    if args.stage == "oracle":
        run_oracle_stage(env, frozen_hrl, args, market_cfg, provenance, out_dir)
        return
    dqn = DQNMonitorAgent(
        device=device, hidden_dim=args.hidden_dim, fusion_hidden=args.fusion_hidden,
        lr=args.lr, gamma=args.gamma,
        target_update=args.target_update, grad_clip=1.0,
    )
    replay = DQNReplayBuffer(args.capacity)
    episodes = args.episodes or (2 if args.mode == "smoke" else 20)
    cfg = {
        "env_steps": 0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": args.epsilon_decay,
        "warmup": min(args.warmup, 32) if args.mode == "smoke" else args.warmup,
        "batch_size": min(args.batch_size, 32) if args.mode == "smoke" else args.batch_size,
        "reward_scale": args.reward_scale,
        "train_frequency": args.train_frequency,
    }
    utility_cfg = {"lambda_cost": args.lambda_cost, "lambda_long": args.lambda_long}

    best_sharpe = -float("inf")
    best_checkpoint = os.path.join(out_dir, "best_dqn_monitor.pth")
    env.set_mode("train")
    start_time = time.time()
    for episode in range(episodes):
        train_result, _ = run_episode(
            env, frozen_hrl, dqn, replay, train=True, cfg=cfg, utility_cfg=utility_cfg
        )
        evaluate = (
            episode == 0
            or (episode + 1) % args.validate_every == 0
            or episode + 1 == episodes
        )
        if evaluate:
            env.set_mode("val")
            val_result, _ = run_episode(env, frozen_hrl, dqn, utility_cfg=utility_cfg)
            logger.info(
                "DQN %s ep=%d/%d train_ret=%.2f%% val_ret=%.2f%% val_sharpe=%.4f "
                "switch=%d loss=%s",
                args.market, episode + 1, episodes, train_result["total_ret"] * 100,
                val_result["total_ret"] * 100, val_result["sharpe"], val_result["switch_count"],
                train_result["dqn_loss"],
            )
            if val_result["sharpe"] > best_sharpe:
                best_sharpe = val_result["sharpe"]
                dqn.save(best_checkpoint, {"episode": episode + 1, "val": val_result})
        else:
            logger.info(
                "DQN %s ep=%d/%d train_ret=%.2f%% validation=skipped loss=%s",
                args.market, episode + 1, episodes, train_result["total_ret"] * 100,
                train_result["dqn_loss"],
            )
        env.set_mode("train")

    best_meta = dqn.load(best_checkpoint)
    test_result = None
    fixed_result = None
    if not args.validation_only:
        env.set_mode("test")
        test_result, history = run_episode(env, frozen_hrl, dqn, utility_cfg=utility_cfg)
        fixed_result, fixed_history = run_episode(
            env, frozen_hrl, dqn, fixed_cycle=int(getattr(runtime_config, "max_hold", 60))
        )
    elapsed = time.time() - start_time
    comparable_to_sota = args.mode == "full" and not args.validation_only
    summary = {
        "run_id": run_id,
        "market": args.market,
        "source_seed": market_cfg["source_seed"],
        "mode": args.mode,
        "episodes": episodes,
        "best_validation": best_meta.get("val", {}),
        "dqn_test": test_result,
        "fixed_hrl_test": fixed_result,
        "target": {
            "comparable_to_sota": comparable_to_sota,
            "sota_sharpe": market_cfg["sota_sharpe"],
            "sota_return": market_cfg["sota_return"],
            "beats_sharpe": (
                test_result["sharpe"] > market_cfg["sota_sharpe"]
                if comparable_to_sota else None
            ),
            "beats_return": (
                test_result["total_ret"] > market_cfg["sota_return"]
                if comparable_to_sota else None
            ),
        },
        "elapsed_seconds": elapsed,
        "config": vars(args),
        "checkpoint": checkpoint,
        "data_provenance": provenance,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    if not args.validation_only:
        pd.DataFrame({"value": history}).to_csv(os.path.join(out_dir, "test_dqn.csv"), index=False)
        pd.DataFrame({"value": fixed_history}).to_csv(os.path.join(out_dir, "test_fixed_hrl.csv"), index=False)
    run_row = {
        "run_id": run_id, "market": args.market, "source_seed": market_cfg["source_seed"],
        "mode": args.mode, "episodes": episodes, "hidden_dim": args.hidden_dim, "lr": args.lr,
        "gamma": args.gamma, "reward_scale": args.reward_scale,
        "reward_source": "outer_value", "input_version": "candidate_state_v2",
        "lambda_cost": args.lambda_cost, "lambda_long": args.lambda_long,
        "train_frequency": args.train_frequency, "validate_every": args.validate_every,
        "validation_only": args.validation_only, "val_sharpe": best_sharpe,
        "test_sharpe": test_result["sharpe"] if test_result else np.nan,
        "test_return": test_result["total_ret"] if test_result else np.nan,
        "elapsed_seconds": elapsed,
    }
    rows = []
    if not args.validation_only:
        for model_name, result in (("dqn_monitor", test_result), ("fixed_hrl", fixed_result)):
            rows.append({"run_id": run_id, "market": args.market, "model": model_name, **result})
    update_workbook(run_row, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
