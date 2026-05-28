"""Train a policy-gradient hold/switch controller on realized episode Sharpe."""

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
from env import PPO_Env
from pg_controller import PGControllerNet, RunningObjectiveBaseline
from Train.PPO_train import HRL_Networks, set_seed
from utils.Log import create_logger


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_BOOK = os.path.join(ROOT, "result.xlsx")

MARKETS = {
    "nas100": {
        "config": "utils.config_Nas",
        "source_seed": 47,
        "checkpoint": "pretrained_assets/kd4rl/nas100/ppo/seed_47/best_model.pth",
        "checkpoint_sha256": "dd7d9c65e6d50c66270f49b11c1a11b61cdec803ede3de1bd2796beea6422e6b",
        "ssm_path": "Dataset/Nas100数据/feature_ssm",
        "ssm_input_files": 78,
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
    return {"checkpoint": checkpoint, "checkpoint_sha256": actual_hash,
            "ssm_path": str(ssm_dir), "ssm_input_files": len(ssm_inputs)}


def build_environment(market, mode, logger):
    if mode == "smoke":
        ranges = MARKETS[market]["smoke_ranges"]
        smoke_episode_len = max(40, int(getattr(runtime_config, "max_hold", 60)) + 20)
        return PPO_Env(
            logger=logger,
            episode_len=smoke_episode_len,
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
    missing_required = [key for key in missing if key.startswith(("outer.", "inner."))]
    if missing_required or unexpected:
        raise RuntimeError(
            f"Frozen HRL mismatch missing={missing_required[:4]} unexpected={unexpected[:4]}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.outer.eval()
    model.inner.eval()


def compute_metrics(history):
    values = pd.Series(history, dtype=float)
    daily = values.pct_change().fillna(0.0)
    ann_ret = float(daily.mean() * 252)
    ann_vol = float(daily.std() * np.sqrt(252))
    max_dd = float(((values.cummax() - values) / values.cummax()).max())
    return {
        "total_ret": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(ann_ret / (ann_vol + 1e-8)),
        "max_dd": max_dd,
        "cr": float(ann_ret / (max_dd + 1e-8)),
        "final_value": float(values.iloc[-1]),
    }


def violation_penalty(early_count, long_count, lambda_min, lambda_max):
    return float(lambda_min) * int(early_count) + float(lambda_max) * int(long_count)


def episode_objective(history, early_count, long_count, lambda_min, lambda_max):
    metrics = compute_metrics(history)
    penalty = violation_penalty(early_count, long_count, lambda_min, lambda_max)
    return float(metrics["sharpe"] - penalty), metrics, float(penalty)


def violates_max_hold_after_hold(hold_age, max_hold):
    return int(hold_age) + 1 > int(max_hold)


def build_candidates(env, frozen_hrl, obs):
    with torch.inference_mode():
        weights = obs["weights_drift"]
        hold_base = obs["base_drift"]
        switch_base, _, _, _, _ = frozen_hrl.outer.pi(
            obs["outer_state"], weights, deterministic=True
        )
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
        candidate_costs = torch.stack(
            [cost_hold, cost_switch, cost_switch - cost_hold], dim=1
        )
    def normal_tensor(value):
        return value.detach().clone()

    enriched_obs = dict(obs)
    enriched_obs["ssm"] = {
        "z": normal_tensor(obs["ssm"]["z"]),
        "h": normal_tensor(obs["ssm"]["h"]),
    }
    enriched_obs["weights_drift"] = normal_tensor(obs["weights_drift"])
    enriched_obs["base_drift"] = normal_tensor(obs["base_drift"])
    enriched_obs["port_state"] = normal_tensor(obs["port_state"])
    enriched_obs["held_p"] = normal_tensor(obs["held_p"])
    enriched_obs["candidate_switch_base"] = normal_tensor(switch_base)
    enriched_obs["candidate_costs"] = normal_tensor(candidate_costs)
    candidate = {
        "hold_base": hold_base,
        "switch_base": switch_base,
        "hold_exec": hold_exec,
        "switch_exec": switch_exec,
    }
    return enriched_obs, candidate


def execute_action(env, candidate, action):
    base = candidate["switch_base"] if action == 1 else candidate["hold_base"]
    weights = candidate["switch_exec"] if action == 1 else candidate["hold_exec"]
    with torch.inference_mode():
        next_obs, _, done, info = env.step(
            weights,
            base,
            outer_action=candidate["switch_base"],
            is_switch=(action == 1),
            calculate_outer_reward=False,
        )
    return next_obs, done, info


def run_pg_episode(env, frozen_hrl, controller, deterministic=False,
                   min_hold=5, max_hold=60, constraint_logit_bias=0.0):
    raw_obs = env.reset()
    obs, candidate = build_candidates(env, frozen_hrl, raw_obs)
    history = [float(env.portfolio_value.item())]
    log_probs, entropies = [], []
    min_constraint_probs, max_constraint_probs = [], []
    switch_count = 0
    hold_age = 0
    holding_lengths = []
    early_count = 0
    long_count = 0
    turnovers = []
    costs = []
    p_switch_all = []
    p_switch_pre_min = []
    p_switch_post_max = []
    p_hold_post_max = []

    while True:
        logits = controller(obs)
        if constraint_logit_bias:
            if hold_age < int(min_hold):
                logits[:, 1] = logits[:, 1] - float(constraint_logit_bias)
            if violates_max_hold_after_hold(hold_age, max_hold):
                logits[:, 0] = logits[:, 0] - float(constraint_logit_bias)
        dist = torch.distributions.Categorical(logits=logits)
        action_tensor = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        action = int(action_tensor.item())
        probs = dist.probs
        p_switch = float(probs[:, 1].detach().mean().item())
        p_hold = float(probs[:, 0].detach().mean().item())
        p_switch_all.append(p_switch)
        if hold_age < int(min_hold):
            p_switch_pre_min.append(p_switch)
        if violates_max_hold_after_hold(hold_age, max_hold):
            p_switch_post_max.append(p_switch)
            p_hold_post_max.append(p_hold)
        if not deterministic:
            log_probs.append(log_prob)
            entropies.append(entropy)
            if hold_age < int(min_hold):
                min_constraint_probs.append(probs[:, 1].mean())
            if violates_max_hold_after_hold(hold_age, max_hold):
                max_constraint_probs.append(probs[:, 0].mean())
        if action == 1 and hold_age < int(min_hold):
            early_count += 1
        if action == 0 and violates_max_hold_after_hold(hold_age, max_hold):
            long_count += 1

        next_raw_obs, done, info = execute_action(env, candidate, action)
        if action == 1:
            switch_count += 1
            if hold_age > 0:
                holding_lengths.append(hold_age)
            hold_age = 1
        else:
            hold_age += 1
        turnovers.append(float(info["rewards"]["turnover"]))
        costs.append(float(info["rewards"]["transaction_cost"]))
        history.append(float(info["portfolio_value"]))
        if done:
            if hold_age > 0:
                holding_lengths.append(hold_age)
            break
        obs, candidate = build_candidates(env, frozen_hrl, next_raw_obs)

    stats = {
        "switch_count": int(switch_count),
        "avg_hold_days": float(np.mean(holding_lengths)) if holding_lengths else 0.0,
        "early_violation_count": int(early_count),
        "long_violation_count": int(long_count),
        "early_violation_rate": float(early_count / max(len(history) - 1, 1)),
        "long_violation_rate": float(long_count / max(len(history) - 1, 1)),
        "turnover": float(np.sum(turnovers)),
        "transaction_cost": float(np.sum(costs)),
        "avg_p_switch": float(np.mean(p_switch_all)) if p_switch_all else 0.0,
        "avg_p_switch_pre_min": float(np.mean(p_switch_pre_min)) if p_switch_pre_min else 0.0,
        "avg_p_switch_post_max": float(np.mean(p_switch_post_max)) if p_switch_post_max else 0.0,
        "avg_p_hold_post_max": float(np.mean(p_hold_post_max)) if p_hold_post_max else 0.0,
    }
    constraints = {
        "min_probs": min_constraint_probs,
        "max_probs": max_constraint_probs,
    }
    return history, log_probs, entropies, constraints, stats


def update_workbook(run_row, final_rows=None):
    if os.path.exists(RESULT_BOOK):
        try:
            existing = pd.read_excel(RESULT_BOOK, sheet_name="pg_controller_runs")
        except Exception:
            existing = pd.DataFrame()
        try:
            final = pd.read_excel(RESULT_BOOK, sheet_name="final_comparison")
        except Exception:
            final = pd.DataFrame()
    else:
        existing = pd.DataFrame()
        final = pd.DataFrame()
    runs = pd.concat([existing, pd.DataFrame([run_row])], ignore_index=True)
    runs = runs.drop_duplicates(subset=["run_id", "market"], keep="last")
    if final_rows:
        final = pd.concat([final, pd.DataFrame(final_rows)], ignore_index=True)
        final = final.drop_duplicates(subset=["run_id", "market", "model"], keep="last")
    with pd.ExcelWriter(RESULT_BOOK, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        runs.to_excel(writer, sheet_name="pg_controller_runs", index=False)
        final.to_excel(writer, sheet_name="final_comparison", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--constraint-loss-scale", type=float, default=1.0)
    parser.add_argument("--constraint-logit-bias", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--use-rolling-baseline", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PG controller requires CUDA but torch cannot access the GPU.")
    device = torch.device(args.device)
    market_cfg = MARKETS[args.market]
    provenance = verify_market_assets(args.market)
    apply_market_config(args.market)
    runtime_config.device = device
    runtime_config.trade_num = 10
    runtime_config.seed = market_cfg["source_seed"]
    set_seed(runtime_config.seed, create_logger(os.path.join("results", "pg_controller", "setup")))

    run_id = args.run_id or f"pg_{args.mode}_ep{args.episodes or (2 if args.mode == 'smoke' else 20)}"
    out_dir = os.path.join("results", "pg_controller", args.market, run_id)
    os.makedirs(out_dir, exist_ok=True)
    logger = create_logger(out_dir)
    env = build_environment(args.market, args.mode, logger)
    frozen_hrl = HRL_Networks(16, env.num_stocks, runtime_config).to(device)
    load_frozen_hrl(frozen_hrl, provenance["checkpoint"], device)

    controller = PGControllerNet().to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    baseline = RunningObjectiveBaseline(momentum=args.baseline_momentum)
    min_hold = int(getattr(runtime_config, "min_hold", 5))
    max_hold = int(getattr(runtime_config, "max_hold", 60))
    episodes = args.episodes or (2 if args.mode == "smoke" else 20)
    best_objective = -float("inf")
    best_path = os.path.join(out_dir, "best_pg_controller.pth")
    records = []
    start = time.time()

    env.set_mode("train")
    for episode in range(episodes):
        history, log_probs, entropies, constraints, stats = run_pg_episode(
            env, frozen_hrl, controller, deterministic=False,
            min_hold=min_hold, max_hold=max_hold,
            constraint_logit_bias=args.constraint_logit_bias,
        )
        objective, metrics, penalty = episode_objective(
            history, stats["early_violation_count"], stats["long_violation_count"],
            args.lambda_min, args.lambda_max,
        )
        return_signal = (
            baseline.advantage(metrics["sharpe"])
            if args.use_rolling_baseline else metrics["sharpe"]
        )
        if log_probs:
            log_prob_sum = torch.stack(log_probs).sum()
            entropy_mean = torch.stack(entropies).mean()
            constraint_loss = torch.tensor(0.0, device=device)
            if constraints["min_probs"]:
                constraint_loss = constraint_loss + args.lambda_min * torch.stack(
                    constraints["min_probs"]
                ).sum()
            if constraints["max_probs"]:
                constraint_loss = constraint_loss + args.lambda_max * torch.stack(
                    constraints["max_probs"]
                ).sum()
            constraint_loss = args.constraint_loss_scale * constraint_loss
            loss = -log_prob_sum * float(return_signal)
            loss = loss + constraint_loss - args.ent_coef * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip)
            optimizer.step()
            loss_value = float(loss.item())
            constraint_loss_value = float(constraint_loss.detach().item())
        else:
            loss_value = None
            constraint_loss_value = None

        env.set_mode("val")
        with torch.inference_mode():
            val_history, _, _, _, val_stats = run_pg_episode(
                env, frozen_hrl, controller, deterministic=True,
                min_hold=min_hold, max_hold=max_hold,
                constraint_logit_bias=args.constraint_logit_bias,
            )
        val_metrics = compute_metrics(val_history)
        val_penalty = violation_penalty(
            val_stats["early_violation_count"], val_stats["long_violation_count"],
            args.lambda_min, args.lambda_max,
        )
        val_objective = float(val_metrics["sharpe"] - val_penalty)
        record = {
            "episode": episode + 1,
            "train_objective": objective,
            "train_sharpe": metrics["sharpe"],
            "train_penalty": penalty,
            "train_return_signal": return_signal,
            "train_constraint_loss": constraint_loss_value,
            "train_loss": loss_value,
            "val_objective": val_objective,
            "val_penalty": val_penalty,
            "val_sharpe": val_metrics["sharpe"],
            "val_return": val_metrics["total_ret"],
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        records.append(record)
        logger.info(
            "PG %s ep=%d/%d train_obj=%.4f train_sharpe=%.4f "
            "val_obj=%.4f val_sharpe=%.4f "
            "switch=%d early=%d long=%d",
            args.market, episode + 1, episodes, objective, metrics["sharpe"],
            val_objective, val_metrics["sharpe"], val_stats["switch_count"],
            val_stats["early_violation_count"], val_stats["long_violation_count"],
        )
        if val_objective > best_objective:
            best_objective = val_objective
            torch.save({
                "controller": controller.state_dict(),
                "episode": episode + 1,
                "val_objective": val_objective,
                "val_penalty": val_penalty,
                "val_metrics": val_metrics,
                "val_stats": val_stats,
            }, best_path)
        env.set_mode("train")

    best = torch.load(best_path, map_location=device)
    controller.load_state_dict(best["controller"])
    test_result = None
    test_stats = None
    final_rows = []
    if not args.validation_only:
        env.set_mode("test")
        with torch.inference_mode():
            test_history, _, _, _, test_stats = run_pg_episode(
                env, frozen_hrl, controller, deterministic=True,
                min_hold=min_hold, max_hold=max_hold,
                constraint_logit_bias=args.constraint_logit_bias,
            )
        test_result = compute_metrics(test_history)
        final_rows.append({
            "run_id": run_id, "market": args.market, "model": "pg_controller",
            **test_result, **test_stats,
        })

    elapsed = time.time() - start
    summary = {
        "run_id": run_id,
        "market": args.market,
        "mode": args.mode,
        "episodes": episodes,
        "best_validation_objective": best.get("val_objective"),
        "best_validation_penalty": best.get("val_penalty"),
        "best_validation": {**best["val_metrics"], **best["val_stats"]},
        "test_result": test_result,
        "test_stats": test_stats,
        "records": records,
        "config": vars(args),
        "min_hold": min_hold,
        "max_hold": max_hold,
        "checkpoint": best_path,
        "data_provenance": provenance,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    pd.DataFrame(records).to_csv(os.path.join(out_dir, "train_records.csv"), index=False)
    run_row = {
        "run_id": run_id,
        "market": args.market,
        "seed": market_cfg["source_seed"],
        "mode": args.mode,
        "episodes": episodes,
        "lr": args.lr,
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "constraint_loss_scale": args.constraint_loss_scale,
        "constraint_logit_bias": args.constraint_logit_bias,
        "grad_clip": args.grad_clip,
        "val_objective": best.get("val_objective"),
        "val_penalty": best.get("val_penalty"),
        "val_sharpe": best["val_metrics"]["sharpe"],
        "val_return": best["val_metrics"]["total_ret"],
        "val_maxdd": best["val_metrics"]["max_dd"],
        **best["val_stats"],
        "test_status": "skipped" if args.validation_only else "completed",
        "elapsed_seconds": elapsed,
    }
    update_workbook(run_row, final_rows=final_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
