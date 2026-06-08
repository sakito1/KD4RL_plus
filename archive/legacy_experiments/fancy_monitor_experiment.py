"""Run EmbMonitor-ZH on top of frozen pretrained KD4RL Outer/Inner policies."""

import argparse
import copy
import json
import os

import pandas as pd
import torch

import utils.config as runtime_config
from agent import HRL_Buffer, HRL_PPO_Agent
from env import PPO_Env
from Train.PPO_train import HRL_Networks, HRL_Trainer, set_seed
from utils.Log import create_logger


ROOT = os.path.dirname(os.path.abspath(__file__))

MARKETS = {
    "nas100": {
        "config_module": "utils.config_Nas",
        "seed": 42,
        "pretrained": "pretrained_assets/kd4rl/nas100/ppo/seed_42/best_model.pth",
        "smoke_ranges": {
            "train": ("2016-01-04", "2017-12-29"),
            "val": ("2019-01-02", "2019-06-28"),
            "test": ("2020-04-23", "2020-08-31"),
        },
    },
    "sh": {
        "config_module": "utils.config_SH",
        "seed": 77,
        "pretrained": "pretrained_assets/kd4rl/sh/ppo/seed_77/best_model.pth",
        "smoke_ranges": {
            "train": ("2016-01-04", "2017-12-28"),
            "val": ("2019-01-02", "2019-06-28"),
            "test": ("2020-01-02", "2020-06-30"),
        },
    },
}


def apply_market_config(market):
    module = __import__(MARKETS[market]["config_module"], fromlist=["dummy"])
    for name, value in vars(module).items():
        if not name.startswith("__"):
            setattr(runtime_config, name, copy.deepcopy(value))


def make_env(market, mode, logger):
    if mode == "smoke":
        ranges = MARKETS[market]["smoke_ranges"]
        runtime_config.episode_len = 40
        runtime_config.max_hold = 10
        runtime_config.min_hold = 2
        return PPO_Env(
            logger=logger,
            episode_len=runtime_config.episode_len,
            max_hold=runtime_config.max_hold,
            train_date_range=ranges["train"],
            val_date_range=ranges["val"],
            test_date_range=ranges["test"],
        )
    if not hasattr(runtime_config, "min_hold"):
        runtime_config.min_hold = 5
    return PPO_Env(logger=logger)


def train_monitor_only(trainer, checkpoint, episodes):
    trainer.load_frozen_hrl_checkpoint(checkpoint)
    trainer.env.set_mode("train")
    best_sharpe = -float("inf")
    best_name = "best_embmonitor_zh.pth"

    for episode in range(episodes):
        train_stats = trainer.run_episode(trainer.env, mode="train", phase="warmup_monitor")
        losses = trainer.agent.update(trainer.buffer.get_batch(), phase="warmup_monitor")
        trainer.buffer.clear()
        metrics = trainer.evaluate("warmup_monitor")
        trainer.logger.info(
            "[EmbMonitor-ZH] episode %d/%d train_ret=%.2f%% val_ret=%.2f%% "
            "val_sharpe=%.4f mon_loss=%.4f",
            episode + 1,
            episodes,
            train_stats["total"] * 100.0,
            metrics["total_ret"] * 100.0,
            metrics["sharpe"],
            losses.get("mon_pi", 0.0),
        )
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            trainer.save_model(best_name)

    if not trainer._load_model(best_name):
        raise RuntimeError("EmbMonitor-ZH training did not create its best checkpoint.")
    trainer.agent.set_module_status("monitor")
    return best_name, best_sharpe


def evaluate_test(trainer, model_name):
    trainer._load_model(model_name)
    trainer.env.set_mode("test")
    trainer.agent.net.eval()

    emb_stats = trainer.run_episode(trainer.env, mode="eval", phase="warmup_monitor")
    fixed_stats = trainer.run_episode(
        trainer.env,
        mode="eval",
        phase="warmup_monitor",
        fixed_cycle=int(runtime_config.max_hold),
    )
    emb_metrics = trainer._compute_metrics(emb_stats["history"])
    fixed_metrics = trainer._compute_metrics(fixed_stats["history"])

    pd.DataFrame(emb_stats["history"], columns=["value"]).to_csv(
        os.path.join(trainer.run_dir, "test_embmonitor_zh.csv"), index=False
    )
    pd.DataFrame(fixed_stats["history"], columns=["value"]).to_csv(
        os.path.join(trainer.run_dir, "test_frozen_hrl_fixed.csv"), index=False
    )
    summary = {
        "embmonitor_zh": {
            **emb_metrics,
            "switch_count": emb_stats["switch_count"],
            "free_switch_count": emb_stats["switch_free_count"],
            "forced_switch_count": emb_stats["forced_switch_count"],
            "forced_hold_count": emb_stats["forced_hold_count"],
        },
        "frozen_hrl_fixed": {
            **fixed_metrics,
            "switch_count": fixed_stats["switch_count"],
            "free_switch_count": fixed_stats["switch_free_count"],
            "forced_switch_count": fixed_stats["forced_switch_count"],
            "forced_hold_count": fixed_stats["forced_hold_count"],
        },
    }
    with open(os.path.join(trainer.run_dir, "test_summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.chdir(ROOT)
    apply_market_config(args.market)
    runtime_config.seed = MARKETS[args.market]["seed"]
    runtime_config.ssm_dim = 16
    runtime_config.trade_num = 10
    runtime_config.lr_monitor = 1e-3
    runtime_config.lr_outer = 0.0
    runtime_config.lr_inner = 0.0
    runtime_config.ppo_epochs = 3
    runtime_config.val_interval = 1
    runtime_config.reward_scale_monitor = 1.0
    runtime_config.device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    runtime_config.cun_path = os.path.join("results", "fancy_monitor", args.mode, args.market, "ppo")

    episodes = args.episodes if args.episodes is not None else (2 if args.mode == "smoke" else 10)
    logger = create_logger(os.path.join("results", "fancy_monitor", args.mode, args.market))
    set_seed(runtime_config.seed, logger)
    env = make_env(args.market, args.mode, logger)
    networks = HRL_Networks(runtime_config.ssm_dim, env.num_stocks, runtime_config).to(runtime_config.device)
    agent = HRL_PPO_Agent(networks, runtime_config)
    buffer = HRL_Buffer(capacity=3000, device=runtime_config.device)
    trainer = HRL_Trainer(agent, env, buffer, runtime_config, logger)

    checkpoint = os.path.join(ROOT, MARKETS[args.market]["pretrained"])
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Missing migrated pretrained checkpoint: {checkpoint}")
    best_model, best_sharpe = train_monitor_only(trainer, checkpoint, episodes)
    summary = evaluate_test(trainer, best_model)
    logger.info("Best validation Sharpe: %.4f", best_sharpe)
    logger.info("Test summary: %s", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
