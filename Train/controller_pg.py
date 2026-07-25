from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class CounterfactualStats:
    log_return: float
    max_drawdown: float
    turnover: float
    free_switch_count: int
    segment_count: int
    trading_days: int = 252
    downside_loss: float = 0.0

    def calmar_ratio(self, eps: float = 1e-8) -> float:
        days = max(1, int(self.trading_days))
        annualized_return = float(self.log_return) / float(days) * 252.0
        return annualized_return / max(float(self.max_drawdown), float(eps))


def segment_budget_allows_switch(
        *,
        day_offset: int,
        rollout_len: int,
        current_segments: int,
        max_hold: int,
        max_segments: int,
) -> bool:
    """Return whether one extra free switch can still satisfy the segment cap."""
    remaining_days = max(0, int(rollout_len) - int(day_offset))
    needed_after_switch = int(math.ceil(remaining_days / max(float(max_hold), 1.0)))
    return int(current_segments) + 1 + needed_after_switch <= int(max_segments)


def segment_count_band_penalty(segment_count: int, *, count_min: int = 15, count_max: int = 25) -> float:
    """Distance outside the desired controller holding-segment count band."""
    lower = int(count_min)
    upper = int(count_max)
    if upper < lower:
        lower, upper = upper, lower
    count = int(segment_count)
    if count < lower:
        return float(lower - count)
    if count > upper:
        return float(count - upper)
    return 0.0


def max_switch_overflow_penalty(actual_switch_count: int, *, max_switch_count: int) -> float:
    """Squared distance above the allowed switch count; no penalty for too few switches."""
    excess = max(0, int(actual_switch_count) - int(max_switch_count))
    return float(excess * excess)


def controller_reward(
        baseline: CounterfactualStats,
        controlled: CounterfactualStats,
        *,
        reward_mode: str = "return_uplift",
        mdd_coef: float = 2.0,
        downside_coef: float = 0.0,
        return_coef: float = 0.5,
        cr_coef: float = 1.0,
        cr_eps: float = 1e-8,
        count_min: int = 15,
        count_max: int = 25,
        count_penalty_coef: float = 0.5,
        max_switch_count: int = None,
        max_switch_penalty_coef: float = None,
        switch_coef: float = 0.0,
        turnover_coef: float = 0.0,
) -> float:
    """Controller reward with opt-in risk-aware modes and max-switch overflow penalty.

    Legacy keyword arguments are accepted for compatibility, but the current
    default objective intentionally ignores MDD, turnover, and minimum-count
    penalties. ``relative_cr`` compares controlled and trained-HRL baseline
    Calmar ratios. ``relative_return_mdd`` uses relative return uplift as the
    main signal and relative MDD improvement as a smaller stabilizing signal.
    ``relative_downside_mdd`` additionally rewards reducing cumulative negative
    daily log returns, while leaving return uplift as the dominant policy term.
    Overflow is divided by ``max_switch_count ** 2`` so the switch term stays on
    a similar scale to log-return uplift across different rollout lengths.
    """
    mode = str(reward_mode)
    if mode in ("return_uplift", "return"):
        base_reward = float(return_coef) * (float(controlled.log_return) - float(baseline.log_return))
    elif mode in ("relative_cr", "cr", "calmar"):
        base_reward = float(cr_coef) * (
            controlled.calmar_ratio(cr_eps) - baseline.calmar_ratio(cr_eps)
        )
    elif mode in ("relative_return_mdd", "relative_ret_mdd", "relative_uplift"):
        relative_return_uplift = (
            float(controlled.log_return) - float(baseline.log_return)
        ) / max(abs(float(baseline.log_return)), float(cr_eps))
        relative_mdd_uplift = (
            float(baseline.max_drawdown) - float(controlled.max_drawdown)
        ) / max(abs(float(baseline.max_drawdown)), float(cr_eps))
        base_reward = (
            float(return_coef) * relative_return_uplift
            + float(mdd_coef) * relative_mdd_uplift
        )
    elif mode in ("relative_downside_mdd", "relative_downside", "downside_uplift"):
        relative_return_uplift = (
            float(controlled.log_return) - float(baseline.log_return)
        ) / max(abs(float(baseline.log_return)), float(cr_eps))
        relative_downside_uplift = (
            float(baseline.downside_loss) - float(controlled.downside_loss)
        ) / max(abs(float(baseline.downside_loss)), float(cr_eps))
        relative_mdd_uplift = (
            float(baseline.max_drawdown) - float(controlled.max_drawdown)
        ) / max(abs(float(baseline.max_drawdown)), float(cr_eps))
        base_reward = (
            float(return_coef) * relative_return_uplift
            + float(downside_coef) * relative_downside_uplift
            + float(mdd_coef) * relative_mdd_uplift
        )
    else:
        raise ValueError(f"Unknown controller reward mode: {reward_mode}")

    allowed = int(count_max if max_switch_count is None else max_switch_count)
    penalty_coef = float(count_penalty_coef if max_switch_penalty_coef is None else max_switch_penalty_coef)
    overflow_penalty = max_switch_overflow_penalty(
        controlled.segment_count,
        max_switch_count=allowed,
    )
    normalized_overflow_penalty = overflow_penalty / float(max(1, allowed) ** 2)
    return base_reward - penalty_coef * normalized_overflow_penalty


def controller_pg_loss(
        episode_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        entropies: torch.Tensor = None,
        *,
        entropy_coef: float = 0.01,
        values: torch.Tensor = None,
        value_coef: float = 0.0,
        normalize_value_advantage: bool = True,
        eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE/actor-critic loss for counterfactual controller rewards."""
    if episode_log_probs.numel() == 0:
        raise ValueError("controller_pg_loss requires at least one episode log-prob.")

    rewards = rewards.to(device=episode_log_probs.device, dtype=episode_log_probs.dtype).view(-1)
    episode_log_probs = episode_log_probs.view(-1)
    if rewards.shape[0] != episode_log_probs.shape[0]:
        raise ValueError("rewards and episode_log_probs must have the same length.")

    value_loss = episode_log_probs.new_tensor(0.0)
    if values is not None:
        values = values.to(device=episode_log_probs.device, dtype=episode_log_probs.dtype).view(-1)
        if values.shape[0] != episode_log_probs.shape[0]:
            raise ValueError("values and episode_log_probs must have the same length.")
        raw_advantage = rewards - values.detach()
        value_loss = torch.mean((values - rewards) ** 2)
        advantage = raw_advantage
    else:
        advantage = rewards

    policy_loss = -(advantage.detach() * episode_log_probs).mean()
    if entropies is None:
        entropy_term = episode_log_probs.new_tensor(0.0)
    else:
        entropy_term = entropies.to(device=episode_log_probs.device, dtype=episode_log_probs.dtype).view(-1).mean()
    entropy_loss = -float(entropy_coef) * entropy_term
    loss = policy_loss + entropy_loss + float(value_coef) * value_loss
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    diagnostics = {
        "loss": float(loss.detach().cpu().item()),
        "loss_abs": float(loss.detach().abs().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "policy_abs_loss": float(policy_loss.detach().abs().cpu().item()),
        "entropy": float(entropy_term.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "reward_mean": float(reward_mean.detach().cpu().item()),
        "reward_std": float(reward_std.detach().cpu().item()),
        "reward_min": float(rewards.min().detach().cpu().item()),
        "reward_max": float(rewards.max().detach().cpu().item()),
        "reward_abs_mean": float(rewards.abs().mean().detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "value_weighted_loss": float((float(value_coef) * value_loss).detach().cpu().item()),
    }
    return loss, diagnostics


def overflow_switch_policy_loss(
        switch_log_probs: torch.Tensor,
        overflow_orders: torch.Tensor,
        *,
        penalty_coef: float,
) -> torch.Tensor:
    """Policy-gradient cost for sampled switch actions beyond the max-switch threshold."""
    if switch_log_probs.numel() == 0 or float(penalty_coef) <= 0.0:
        return switch_log_probs.new_tensor(0.0)
    switch_log_probs = switch_log_probs.view(-1)
    overflow_orders = overflow_orders.to(
        device=switch_log_probs.device,
        dtype=switch_log_probs.dtype,
    ).view(-1)
    if switch_log_probs.shape[0] != overflow_orders.shape[0]:
        raise ValueError("switch_log_probs and overflow_orders must have the same length.")
    marginal_cost = float(penalty_coef) * (2.0 * overflow_orders.clamp_min(1.0) - 1.0)
    return (marginal_cost.detach() * switch_log_probs).mean()
