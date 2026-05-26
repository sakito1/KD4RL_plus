"""Train EmbMonitor-ZH with DQN while reusing frozen KD4RL HRL policies."""

import argparse
import copy
import hashlib
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
            max_hold=10,
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


def execute_step(env, frozen_hrl, obs, action):
    with torch.inference_mode():
        weights = obs["weights_drift"]
        outer_action, _, _, _, _ = frozen_hrl.outer.pi(
            obs["outer_state"], weights, deterministic=True
        )
        value_hold = frozen_hrl.outer.value(obs["outer_state"], obs["base_drift"]).squeeze(-1)
        value_switch = frozen_hrl.outer.value(obs["outer_state"], outer_action).squeeze(-1)
        base_used = outer_action if action == 1 else obs["base_drift"]
        weights_exec, _, _, _, _ = frozen_hrl.inner.build_inner_action_simple(
            obs["inner_state"],
            base_used,
            weights,
            alpha=float(getattr(runtime_config, "inner_max_boundary", 1.0)),
            deterministic=True,
        )
        reward = (value_switch - value_hold) if action == 1 else (value_hold - value_switch)
        next_obs, _, done, info = env.step(
            weights_exec,
            base_used,
            outer_action=outer_action,
            is_switch=(action == 1),
            calculate_outer_reward=False,
        )
    return next_obs, float(reward.item()), done, info, float((value_switch - value_hold).item())


def run_episode(env, frozen_hrl, dqn, replay=None, train=False, fixed_cycle=None, cfg=None):
    obs = env.reset()
    history = [float(env.portfolio_value.item())]
    rewards, advantages, turnovers, costs = [], [], [], []
    switches = 0
    hold_age = 0
    holding_lengths = []
    losses = []

    while True:
        if fixed_cycle is None:
            epsilon = epsilon_at(
                cfg["env_steps"], cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay"]
            ) if train else 0.0
            action = dqn.select_action(obs, epsilon=epsilon)
        else:
            action = int(hold_age == 0 or hold_age >= fixed_cycle)
        next_obs, reward, done, info, switch_adv = execute_step(env, frozen_hrl, obs, action)
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
        turnovers.append(float(info["rewards"]["turnover"]))
        costs.append(float(info["rewards"]["transaction_cost"]))
        history.append(float(info["portfolio_value"]))
        obs = next_obs
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


def update_workbook(run_row, final_rows):
    reference = reference_frame()
    if os.path.exists(RESULT_BOOK):
        try:
            existing_runs = pd.read_excel(RESULT_BOOK, sheet_name="dqn_runs")
        except Exception:
            existing_runs = pd.DataFrame()
    else:
        existing_runs = pd.DataFrame()
    runs = pd.concat([existing_runs, pd.DataFrame([run_row])], ignore_index=True)
    runs = runs.drop_duplicates(subset=["run_id", "market"], keep="last")
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
    final = pd.concat([old_final, pd.DataFrame(final_rows)], ignore_index=True)
    final = final.drop_duplicates(subset=["run_id", "market", "model"], keep="last")
    with pd.ExcelWriter(RESULT_BOOK, engine="openpyxl") as writer:
        reference.to_excel(writer, sheet_name="reference_sota", index=False)
        pretrained.to_excel(writer, sheet_name="pretrained_hrl", index=False)
        runs.to_excel(writer, sheet_name="dqn_runs", index=False)
        final.to_excel(writer, sheet_name="final_comparison", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--epsilon-decay", type=int, default=20000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
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

    run_id = args.run_id or f"{args.mode}_ep{args.episodes or (2 if args.mode == 'smoke' else 20)}"
    out_dir = os.path.join("results", "dqn_monitor", args.market, run_id)
    os.makedirs(out_dir, exist_ok=True)
    logger = create_logger(out_dir)
    env = build_environment(args.market, args.mode, logger)
    frozen_hrl = HRL_Networks(16, env.num_stocks, runtime_config).to(device)
    checkpoint = provenance["checkpoint"]
    load_frozen_hrl(frozen_hrl, checkpoint, device)
    dqn = DQNMonitorAgent(
        device=device, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma,
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

    best_sharpe = -float("inf")
    best_checkpoint = os.path.join(out_dir, "best_dqn_monitor.pth")
    env.set_mode("train")
    start_time = time.time()
    for episode in range(episodes):
        train_result, _ = run_episode(env, frozen_hrl, dqn, replay, train=True, cfg=cfg)
        evaluate = (
            episode == 0
            or (episode + 1) % args.validate_every == 0
            or episode + 1 == episodes
        )
        if evaluate:
            env.set_mode("val")
            val_result, _ = run_episode(env, frozen_hrl, dqn)
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
        test_result, history = run_episode(env, frozen_hrl, dqn)
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
