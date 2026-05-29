"""Train a policy-gradient hold/switch controller on realized episode Sharpe."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


def compute_reward_to_go(history, gamma=1.0, standardize=True):
    values = np.asarray(history, dtype=float)
    step_rewards = np.diff(np.log(np.maximum(values, 1e-12)))
    returns = np.zeros_like(step_rewards)
    running = 0.0
    for index in range(len(step_rewards) - 1, -1, -1):
        running = step_rewards[index] + float(gamma) * running
        returns[index] = running
    if standardize and returns.size > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns.astype(np.float32)


def violation_penalty(early_count, long_count, lambda_min, lambda_max):
    return float(lambda_min) * int(early_count) + float(lambda_max) * int(long_count)


def episode_objective(history, early_count, long_count, lambda_min, lambda_max,
                      scheduled_switch_rate=0.0, schedule_penalty=0.0,
                      near_max_switch_rate=0.0, near_max_penalty=0.0,
                      min_boundary_switch_rate=0.0, min_boundary_penalty=0.0):
    metrics = compute_metrics(history)
    penalty = violation_penalty(early_count, long_count, lambda_min, lambda_max)
    penalty += float(schedule_penalty) * float(scheduled_switch_rate)
    penalty += float(near_max_penalty) * float(near_max_switch_rate)
    penalty += float(min_boundary_penalty) * float(min_boundary_switch_rate)
    return float(metrics["sharpe"] - penalty), metrics, float(penalty)


def violates_max_hold_after_hold(hold_age, max_hold):
    return int(hold_age) + 1 > int(max_hold)


def estimate_candidate_step_returns(env, weights_drift, hold_exec, switch_exec):
    """One-step counterfactual net log returns for the two executable candidates.

    This mirrors the environment's next-step accounting without mutating the
    environment. The values are used only as an auxiliary training label, never
    as controller input at validation/test time.
    """
    r_future = env.ratio[:, env.day]
    current_weights = weights_drift.reshape(-1).detach()

    def net_log_return(exec_weights):
        final_weights = exec_weights.reshape(-1).detach()
        turnover = torch.sum(torch.abs(final_weights - current_weights))
        cost_rate = turnover * env.transaction_cost_pct
        net_growth = torch.sum(final_weights * r_future) * torch.clamp(
            1.0 - cost_rate, min=1e-8
        )
        return torch.log(torch.clamp(net_growth, min=1e-12)), cost_rate

    hold_ret, hold_cost = net_log_return(hold_exec)
    switch_ret, switch_cost = net_log_return(switch_exec)
    candidate_returns = torch.stack(
        [hold_ret, switch_ret, switch_ret - hold_ret], dim=0
    ).unsqueeze(0)
    candidate_cost_rates = torch.stack(
        [hold_cost, switch_cost, switch_cost - hold_cost], dim=0
    ).unsqueeze(0)
    return candidate_returns, candidate_cost_rates


def counterfactual_advantage_loss(logits_list, advantages_list, margin=0.0,
                                  weight_clip=0.0):
    if not logits_list:
        return None
    logits = torch.cat(logits_list, dim=0)
    advantages = torch.cat(advantages_list, dim=0).reshape(-1)
    valid = torch.abs(advantages) > float(margin)
    if not bool(valid.any().item()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    switch_logit_margin = logits[:, 1] - logits[:, 0]
    target = (advantages > 0).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        switch_logit_margin[valid], target[valid], reduction="none"
    )
    if float(weight_clip) > 0:
        weight = torch.clamp(torch.abs(advantages[valid]) / float(weight_clip), max=1.0)
        loss = loss * weight
    return loss.mean()


def mask_controller_hold_age(obs):
    masked = dict(obs)
    port_state = obs["port_state"].clone()
    port_state[:, 0] = 0.0
    masked["port_state"] = port_state
    return masked


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
        candidate_returns, candidate_cost_rates = estimate_candidate_step_returns(
            env, weights, hold_exec, switch_exec
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
    enriched_obs["candidate_step_returns"] = normal_tensor(candidate_returns)
    enriched_obs["candidate_step_cost_rates"] = normal_tensor(candidate_cost_rates)
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


def run_supervised_pretrain(env, frozen_hrl, controller, optimizer, episodes,
                            min_hold=5, max_hold=60, margin=0.0, weight_clip=0.0,
                            grad_clip=10.0, mask_hold_age_feature=False,
                            logger=None, market=""):
    records = []
    env.set_mode("train")
    for episode in range(int(episodes)):
        raw_obs = env.reset()
        obs, candidate = build_candidates(env, frozen_hrl, raw_obs)
        hold_age = 0
        logits_list, advantages_list = [], []
        oracle_switch_count = 0
        oracle_positive_switch_count = 0
        oracle_early_count = 0
        oracle_long_count = 0
        while True:
            policy_obs = mask_controller_hold_age(obs) if mask_hold_age_feature else obs
            logits_list.append(controller(policy_obs))
            switch_advantage = obs["candidate_step_returns"][:, 2].reshape(-1).detach()
            advantages_list.append(switch_advantage)
            advantage_value = float(switch_advantage.mean().item())
            if hold_age < int(min_hold):
                action = 0
            elif violates_max_hold_after_hold(hold_age, max_hold):
                action = 1
            else:
                action = int(advantage_value > float(margin))
            if action == 1:
                oracle_switch_count += 1
                if advantage_value > 0:
                    oracle_positive_switch_count += 1
                if hold_age < int(min_hold):
                    oracle_early_count += 1
            if action == 0 and violates_max_hold_after_hold(hold_age, max_hold):
                oracle_long_count += 1

            next_raw_obs, done, _ = execute_action(env, candidate, action)
            hold_age = 1 if action == 1 else hold_age + 1
            if done:
                break
            obs, candidate = build_candidates(env, frozen_hrl, next_raw_obs)

        loss = counterfactual_advantage_loss(
            logits_list, advantages_list, margin=margin, weight_clip=weight_clip
        )
        if loss is None:
            loss = torch.tensor(0.0, device=next(controller.parameters()).device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
        optimizer.step()
        record = {
            "pretrain_episode": episode + 1,
            "pretrain_loss": float(loss.detach().item()),
            "pretrain_oracle_switch_count": int(oracle_switch_count),
            "pretrain_oracle_positive_switch_rate": float(
                oracle_positive_switch_count / max(oracle_switch_count, 1)
            ),
            "pretrain_oracle_early_count": int(oracle_early_count),
            "pretrain_oracle_long_count": int(oracle_long_count),
        }
        records.append(record)
        if logger:
            logger.info(
                "PG supervised %s ep=%d/%d loss=%.4f oracle_switch=%d pos_rate=%.2f",
                market, episode + 1, int(episodes), record["pretrain_loss"],
                oracle_switch_count, record["pretrain_oracle_positive_switch_rate"],
            )
    return records


def run_pg_episode(env, frozen_hrl, controller, deterministic=False,
                   min_hold=5, max_hold=60, constraint_logit_bias=0.0,
                   late_hold_start=0.75, late_hold_logit_bias=0.0,
                   mask_hold_age_feature=False, hard_boundary_mask=False):
    raw_obs = env.reset()
    obs, candidate = build_candidates(env, frozen_hrl, raw_obs)
    history = [float(env.portfolio_value.item())]
    log_probs, entropies = [], []
    min_constraint_probs, max_constraint_probs, late_hold_probs = [], [], []
    aux_logits, aux_advantages = [], []
    switch_count = 0
    hold_age = 0
    holding_lengths = []
    early_count = 0
    long_count = 0
    turnovers = []
    costs = []
    switch_ages = []
    p_switch_all = []
    p_switch_pre_min = []
    p_switch_post_max = []
    p_hold_post_max = []
    cf_advantages = []
    chosen_cf_advantages = []
    switch_positive_advantage = 0
    hold_negative_advantage = 0
    aligned_decisions = 0

    while True:
        policy_obs = mask_controller_hold_age(obs) if mask_hold_age_feature else obs
        logits = controller(policy_obs)
        late_start_day = int(np.floor(float(late_hold_start) * int(max_hold)))
        late_span = max(int(max_hold) - late_start_day, 1)
        in_late_window = late_start_day <= hold_age < int(max_hold)
        late_ramp = 0.0
        if in_late_window:
            late_ramp = float((hold_age - late_start_day + 1) / late_span)
            logits[:, 0] = logits[:, 0] - float(late_hold_logit_bias) * late_ramp
        if hard_boundary_mask:
            if hold_age < int(min_hold):
                logits[:, 1] = -1e9
            if violates_max_hold_after_hold(hold_age, max_hold):
                logits[:, 0] = -1e9
        elif constraint_logit_bias:
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
        switch_advantage = obs.get("candidate_step_returns")
        if switch_advantage is not None:
            switch_advantage = switch_advantage[:, 2].reshape(-1).detach()
            free_decision = (
                hold_age >= int(min_hold)
                and not violates_max_hold_after_hold(hold_age, max_hold)
            )
            if free_decision:
                aux_logits.append(logits)
                aux_advantages.append(switch_advantage)
            advantage_value = float(switch_advantage.mean().item())
            cf_advantages.append(advantage_value)
            chosen_advantage = advantage_value if action == 1 else -advantage_value
            chosen_cf_advantages.append(chosen_advantage)
            if action == 1 and advantage_value > 0:
                switch_positive_advantage += 1
            if action == 0 and advantage_value <= 0:
                hold_negative_advantage += 1
            if (action == 1 and advantage_value > 0) or (action == 0 and advantage_value <= 0):
                aligned_decisions += 1
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
            if in_late_window:
                late_hold_probs.append(late_ramp * probs[:, 0].mean())
            if violates_max_hold_after_hold(hold_age, max_hold):
                max_constraint_probs.append(probs[:, 0].mean())
        if action == 1 and hold_age < int(min_hold):
            early_count += 1
        if action == 0 and violates_max_hold_after_hold(hold_age, max_hold):
            long_count += 1

        next_raw_obs, done, info = execute_action(env, candidate, action)
        if action == 1:
            switch_count += 1
            switch_ages.append(int(hold_age))
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

    near_max_threshold = max(1, int(np.floor(0.9 * int(max_hold))))
    scheduled_switch_count = sum(age >= int(max_hold) for age in switch_ages)
    near_max_switch_count = sum(age >= near_max_threshold for age in switch_ages)
    min_hold_switch_count = sum(age <= int(min_hold) for age in switch_ages)
    stats = {
        "switch_count": int(switch_count),
        "avg_hold_days": float(np.mean(holding_lengths)) if holding_lengths else 0.0,
        "switch_age_mean": float(np.mean(switch_ages)) if switch_ages else 0.0,
        "switch_age_std": float(np.std(switch_ages)) if switch_ages else 0.0,
        "switch_age_min": int(min(switch_ages)) if switch_ages else 0,
        "switch_age_max": int(max(switch_ages)) if switch_ages else 0,
        "scheduled_switch_count": int(scheduled_switch_count),
        "scheduled_switch_rate": float(scheduled_switch_count / max(switch_count, 1)),
        "near_max_switch_count": int(near_max_switch_count),
        "near_max_switch_rate": float(near_max_switch_count / max(switch_count, 1)),
        "min_hold_switch_count": int(min_hold_switch_count),
        "min_hold_switch_rate": float(min_hold_switch_count / max(switch_count, 1)),
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
        "cf_switch_advantage_mean": float(np.mean(cf_advantages)) if cf_advantages else 0.0,
        "cf_switch_advantage_abs_mean": float(np.mean(np.abs(cf_advantages))) if cf_advantages else 0.0,
        "cf_switch_advantage_positive_rate": float(
            np.mean(np.asarray(cf_advantages) > 0)
        ) if cf_advantages else 0.0,
        "chosen_cf_advantage_mean": float(np.mean(chosen_cf_advantages)) if chosen_cf_advantages else 0.0,
        "decision_cf_alignment_rate": float(aligned_decisions / max(len(cf_advantages), 1)),
        "switch_positive_advantage_rate": float(switch_positive_advantage / max(switch_count, 1)),
        "hold_negative_advantage_rate": float(
            hold_negative_advantage / max((len(cf_advantages) - switch_count), 1)
        ),
    }
    constraints = {
        "min_probs": min_constraint_probs,
        "late_hold_probs": late_hold_probs,
        "max_probs": max_constraint_probs,
        "aux_logits": aux_logits if not deterministic else [],
        "aux_advantages": aux_advantages if not deterministic else [],
    }
    return history, log_probs, entropies, constraints, stats


def update_workbook(run_row, final_rows=None):
    lock_path = f"{RESULT_BOOK}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
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
        writer_kwargs = {"engine": "openpyxl"}
        if os.path.exists(RESULT_BOOK):
            writer_kwargs.update({"mode": "a", "if_sheet_exists": "replace"})
        with pd.ExcelWriter(RESULT_BOOK, **writer_kwargs) as writer:
            runs.to_excel(writer, sheet_name="pg_controller_runs", index=False)
            final.to_excel(writer, sheet_name="final_comparison", index=False)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--constraint-loss-scale", type=float, default=1.0)
    parser.add_argument("--constraint-logit-bias", type=float, default=10.0)
    parser.add_argument("--late-hold-start", type=float, default=0.75)
    parser.add_argument("--late-hold-logit-bias", type=float, default=0.0)
    parser.add_argument("--late-hold-loss-scale", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--schedule-penalty", type=float, default=0.5)
    parser.add_argument("--near-max-penalty", type=float, default=0.5)
    parser.add_argument("--min-boundary-penalty", type=float, default=0.0)
    parser.add_argument(
        "--pg-objective",
        choices=["episode_sharpe", "reward_to_go"],
        default="episode_sharpe",
    )
    parser.add_argument("--reward-gamma", type=float, default=1.0)
    parser.add_argument("--no-reward-standardize", action="store_true")
    parser.add_argument("--aux-advantage-loss-scale", type=float, default=0.0)
    parser.add_argument("--aux-advantage-margin", type=float, default=0.0)
    parser.add_argument("--aux-advantage-weight-clip", type=float, default=0.0)
    parser.add_argument("--mask-hold-age-feature", action="store_true")
    parser.add_argument("--hard-boundary-mask", action="store_true")
    parser.add_argument("--supervised-pretrain-episodes", type=int, default=0)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--use-rolling-baseline", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
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
    start = time.time()
    logger = create_logger(out_dir)
    env = build_environment(args.market, args.mode, logger)
    frozen_hrl = HRL_Networks(16, env.num_stocks, runtime_config).to(device)
    load_frozen_hrl(frozen_hrl, provenance["checkpoint"], device)

    controller = PGControllerNet(
        hidden_dim=args.hidden_dim,
        fusion_hidden=args.fusion_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    baseline = RunningObjectiveBaseline(momentum=args.baseline_momentum)
    min_hold = int(getattr(runtime_config, "min_hold", 5))
    max_hold = int(getattr(runtime_config, "max_hold", 60))
    episodes = args.episodes or (2 if args.mode == "smoke" else 20)
    if args.eval_checkpoint:
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        controller.load_state_dict(checkpoint.get("controller", checkpoint))
        env.set_mode(args.eval_split)
        with torch.inference_mode():
            eval_history, _, _, _, eval_stats = run_pg_episode(
                env, frozen_hrl, controller, deterministic=True,
                min_hold=min_hold, max_hold=max_hold,
                constraint_logit_bias=args.constraint_logit_bias,
                late_hold_start=args.late_hold_start,
                late_hold_logit_bias=args.late_hold_logit_bias,
                mask_hold_age_feature=args.mask_hold_age_feature,
                hard_boundary_mask=args.hard_boundary_mask,
            )
        eval_metrics = compute_metrics(eval_history)
        eval_penalty = violation_penalty(
            eval_stats["early_violation_count"], eval_stats["long_violation_count"],
            args.lambda_min, args.lambda_max,
        )
        eval_penalty += args.schedule_penalty * eval_stats["scheduled_switch_rate"]
        eval_penalty += args.near_max_penalty * eval_stats["near_max_switch_rate"]
        eval_penalty += args.min_boundary_penalty * eval_stats["min_hold_switch_rate"]
        eval_objective = float(eval_metrics["sharpe"] - eval_penalty)
        elapsed = time.time() - start
        summary = {
            "run_id": run_id,
            "market": args.market,
            "mode": args.mode,
            "eval_split": args.eval_split,
            "eval_checkpoint": args.eval_checkpoint,
            "eval_objective": eval_objective,
            "eval_penalty": eval_penalty,
            "eval_result": {**eval_metrics, **eval_stats},
            "config": vars(args),
            "min_hold": min_hold,
            "max_hold": max_hold,
            "data_provenance": provenance,
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as file:
            json.dump(summary, file, indent=2)
        run_row = {
            "run_id": run_id,
            "market": args.market,
            "seed": market_cfg["source_seed"],
            "mode": f"eval_{args.eval_split}",
            "episodes": 0,
            "hidden_dim": args.hidden_dim,
            "fusion_hidden": args.fusion_hidden,
            "lr": args.lr,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "constraint_loss_scale": args.constraint_loss_scale,
            "constraint_logit_bias": args.constraint_logit_bias,
            "late_hold_start": args.late_hold_start,
            "late_hold_logit_bias": args.late_hold_logit_bias,
            "late_hold_loss_scale": args.late_hold_loss_scale,
            "grad_clip": args.grad_clip,
            "schedule_penalty": args.schedule_penalty,
            "near_max_penalty": args.near_max_penalty,
            "min_boundary_penalty": args.min_boundary_penalty,
            "pg_objective": args.pg_objective,
            "reward_gamma": args.reward_gamma,
            "aux_advantage_loss_scale": args.aux_advantage_loss_scale,
            "aux_advantage_margin": args.aux_advantage_margin,
            "aux_advantage_weight_clip": args.aux_advantage_weight_clip,
            "mask_hold_age_feature": args.mask_hold_age_feature,
            "hard_boundary_mask": args.hard_boundary_mask,
            "supervised_pretrain_episodes": args.supervised_pretrain_episodes,
            "val_objective": eval_objective if args.eval_split == "val" else None,
            "val_penalty": eval_penalty if args.eval_split == "val" else None,
            "val_sharpe": eval_metrics["sharpe"] if args.eval_split == "val" else None,
            "val_return": eval_metrics["total_ret"] if args.eval_split == "val" else None,
            "val_maxdd": eval_metrics["max_dd"] if args.eval_split == "val" else None,
            **eval_stats,
            "test_status": "completed" if args.eval_split == "test" else "skipped",
            "elapsed_seconds": elapsed,
        }
        final_rows = []
        if args.eval_split == "test":
            final_rows.append({
                "run_id": run_id, "market": args.market, "model": "pg_controller",
                **eval_metrics, **eval_stats,
            })
        update_workbook(run_row, final_rows=final_rows)
        print(json.dumps(summary, indent=2))
        return
    best_objective = -float("inf")
    best_path = os.path.join(out_dir, "best_pg_controller.pth")
    records = []
    pretrain_records = []

    env.set_mode("train")
    if args.supervised_pretrain_episodes > 0:
        pretrain_records = run_supervised_pretrain(
            env, frozen_hrl, controller, optimizer,
            episodes=args.supervised_pretrain_episodes,
            min_hold=min_hold, max_hold=max_hold,
            margin=args.aux_advantage_margin,
            weight_clip=args.aux_advantage_weight_clip,
            grad_clip=args.grad_clip,
            mask_hold_age_feature=args.mask_hold_age_feature,
            logger=logger,
            market=args.market,
        )
        env.set_mode("train")
    for episode in range(episodes):
        history, log_probs, entropies, constraints, stats = run_pg_episode(
            env, frozen_hrl, controller, deterministic=False,
            min_hold=min_hold, max_hold=max_hold,
            constraint_logit_bias=args.constraint_logit_bias,
            late_hold_start=args.late_hold_start,
            late_hold_logit_bias=args.late_hold_logit_bias,
            mask_hold_age_feature=args.mask_hold_age_feature,
            hard_boundary_mask=args.hard_boundary_mask,
        )
        objective, metrics, penalty = episode_objective(
            history, stats["early_violation_count"], stats["long_violation_count"],
            args.lambda_min, args.lambda_max,
            stats["scheduled_switch_rate"], args.schedule_penalty,
            stats["near_max_switch_rate"], args.near_max_penalty,
            stats["min_hold_switch_rate"], args.min_boundary_penalty,
        )
        return_signal = (
            baseline.advantage(objective)
            if args.use_rolling_baseline else objective
        )
        if log_probs:
            log_probs_tensor = torch.stack(log_probs)
            log_prob_sum = log_probs_tensor.sum()
            entropy_mean = torch.stack(entropies).mean()
            constraint_loss = torch.tensor(0.0, device=device)
            if constraints["min_probs"]:
                constraint_loss = constraint_loss + args.lambda_min * torch.stack(
                    constraints["min_probs"]
                ).sum()
            if constraints["late_hold_probs"]:
                constraint_loss = constraint_loss + args.late_hold_loss_scale * torch.stack(
                    constraints["late_hold_probs"]
                ).sum()
            if constraints["max_probs"]:
                constraint_loss = constraint_loss + args.lambda_max * torch.stack(
                    constraints["max_probs"]
                ).sum()
            constraint_loss = args.constraint_loss_scale * constraint_loss
            aux_loss = counterfactual_advantage_loss(
                constraints.get("aux_logits", []),
                constraints.get("aux_advantages", []),
                margin=args.aux_advantage_margin,
                weight_clip=args.aux_advantage_weight_clip,
            )
            if aux_loss is None:
                aux_loss = torch.tensor(0.0, device=device)
            if args.pg_objective == "reward_to_go":
                weights = compute_reward_to_go(
                    history,
                    gamma=args.reward_gamma,
                    standardize=not args.no_reward_standardize,
                )
                weights = torch.as_tensor(weights, device=device, dtype=log_probs_tensor.dtype)
                policy_loss = -(log_probs_tensor.reshape(-1) * weights).sum()
            else:
                policy_loss = -log_prob_sum * float(return_signal)
            loss = policy_loss
            loss = loss + args.aux_advantage_loss_scale * aux_loss
            loss = loss + constraint_loss - args.ent_coef * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip)
            optimizer.step()
            loss_value = float(loss.item())
            policy_loss_value = float(policy_loss.detach().item())
            constraint_loss_value = float(constraint_loss.detach().item())
            aux_loss_value = float(aux_loss.detach().item())
        else:
            loss_value = None
            policy_loss_value = None
            constraint_loss_value = None
            aux_loss_value = None

        env.set_mode("val")
        with torch.inference_mode():
            val_history, _, _, _, val_stats = run_pg_episode(
                env, frozen_hrl, controller, deterministic=True,
                min_hold=min_hold, max_hold=max_hold,
                constraint_logit_bias=args.constraint_logit_bias,
                late_hold_start=args.late_hold_start,
                late_hold_logit_bias=args.late_hold_logit_bias,
                mask_hold_age_feature=args.mask_hold_age_feature,
                hard_boundary_mask=args.hard_boundary_mask,
            )
        val_metrics = compute_metrics(val_history)
        val_penalty = violation_penalty(
            val_stats["early_violation_count"], val_stats["long_violation_count"],
            args.lambda_min, args.lambda_max,
        )
        val_penalty += args.schedule_penalty * val_stats["scheduled_switch_rate"]
        val_penalty += args.near_max_penalty * val_stats["near_max_switch_rate"]
        val_penalty += args.min_boundary_penalty * val_stats["min_hold_switch_rate"]
        val_objective = float(val_metrics["sharpe"] - val_penalty)
        record = {
            "episode": episode + 1,
            "train_objective": objective,
            "train_sharpe": metrics["sharpe"],
            "train_penalty": penalty,
            "train_return_signal": return_signal,
            "train_policy_loss": policy_loss_value,
            "train_constraint_loss": constraint_loss_value,
            "train_aux_advantage_loss": aux_loss_value,
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
                late_hold_start=args.late_hold_start,
                late_hold_logit_bias=args.late_hold_logit_bias,
                mask_hold_age_feature=args.mask_hold_age_feature,
                hard_boundary_mask=args.hard_boundary_mask,
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
        "pretrain_records": pretrain_records,
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
        "hidden_dim": args.hidden_dim,
        "fusion_hidden": args.fusion_hidden,
        "lr": args.lr,
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "constraint_loss_scale": args.constraint_loss_scale,
        "constraint_logit_bias": args.constraint_logit_bias,
        "late_hold_start": args.late_hold_start,
        "late_hold_logit_bias": args.late_hold_logit_bias,
        "late_hold_loss_scale": args.late_hold_loss_scale,
        "grad_clip": args.grad_clip,
        "schedule_penalty": args.schedule_penalty,
        "near_max_penalty": args.near_max_penalty,
        "min_boundary_penalty": args.min_boundary_penalty,
        "pg_objective": args.pg_objective,
        "reward_gamma": args.reward_gamma,
        "aux_advantage_loss_scale": args.aux_advantage_loss_scale,
        "aux_advantage_margin": args.aux_advantage_margin,
        "aux_advantage_weight_clip": args.aux_advantage_weight_clip,
        "mask_hold_age_feature": args.mask_hold_age_feature,
        "hard_boundary_mask": args.hard_boundary_mask,
        "supervised_pretrain_episodes": args.supervised_pretrain_episodes,
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
