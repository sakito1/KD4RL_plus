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
        mdd_coef: float = 2.0,
        return_coef: float = 0.5,
        count_min: int = 15,
        count_max: int = 25,
        count_penalty_coef: float = 0.5,
        max_switch_count: int = None,
        max_switch_penalty_coef: float = None,
        switch_coef: float = 0.0,
        turnover_coef: float = 0.0,
) -> float:
    """Relative-return controller reward with a max-switch overflow penalty.

    Legacy keyword arguments are accepted for compatibility, but the current
    controller objective intentionally ignores MDD, turnover, and minimum-count
    penalties. Forced max-hold switches already provide the lower switch bound.
    """
    return_uplift = float(controlled.log_return) - float(baseline.log_return)
    allowed = int(count_max if max_switch_count is None else max_switch_count)
    penalty_coef = float(count_penalty_coef if max_switch_penalty_coef is None else max_switch_penalty_coef)
    overflow_penalty = max_switch_overflow_penalty(
        controlled.segment_count,
        max_switch_count=allowed,
    )
    return (
        float(return_coef) * return_uplift
        - penalty_coef * overflow_penalty
    )


def controller_pg_loss(
        episode_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        entropies: torch.Tensor = None,
        *,
        entropy_coef: float = 0.01,
        eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE loss with batch-normalized counterfactual rewards."""
    if episode_log_probs.numel() == 0:
        raise ValueError("controller_pg_loss requires at least one episode log-prob.")

    rewards = rewards.to(device=episode_log_probs.device, dtype=episode_log_probs.dtype).view(-1)
    episode_log_probs = episode_log_probs.view(-1)
    if rewards.shape[0] != episode_log_probs.shape[0]:
        raise ValueError("rewards and episode_log_probs must have the same length.")

    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    if rewards.numel() > 1:
        advantage = (rewards - reward_mean) / reward_std.clamp_min(eps)
    else:
        advantage = rewards - reward_mean

    policy_loss = -(advantage.detach() * episode_log_probs).mean()
    if entropies is None:
        entropy_term = episode_log_probs.new_tensor(0.0)
    else:
        entropy_term = entropies.to(device=episode_log_probs.device, dtype=episode_log_probs.dtype).view(-1).mean()
    entropy_loss = -float(entropy_coef) * entropy_term
    loss = policy_loss + entropy_loss
    diagnostics = {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "entropy": float(entropy_term.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "reward_mean": float(reward_mean.detach().cpu().item()),
        "reward_std": float(reward_std.detach().cpu().item()),
    }
    return loss, diagnostics
