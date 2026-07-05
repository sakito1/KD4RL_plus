# Sharpe/CR Risk Training Design

Date: 2026-07-05
Status: Approved design, pending implementation plan

## Goal

Improve KD4RL+ so the risk-aware HRL/controller variant can beat DeepAries and DeepTrader on as many headline metrics as possible, ideally total return, Sharpe, maximum drawdown, and Calmar ratio. The first implementation must not change the behavior of the existing production/reproduction flow:

`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`

The new training path will be isolated behind new reward-mode arguments and a new seed-sweep script. Existing defaults remain unchanged.

## Current Context

The existing end-to-end script trains with a return-oriented controller objective:

- Controller PG passes `--controller_return_coef` and hard-codes `--controller_mdd_coef 0.0`.
- `Train/controller_pg.py::controller_reward` currently accepts MDD-related arguments but intentionally ignores MDD, turnover, and minimum-count penalties. It rewards relative log-return uplift plus a normalized max-switch overflow penalty.
- `env/PPO_env.py::step` exposes `outer_step_reward` as the daily log return, so the outer actor is not directly optimized for Sharpe.
- Validation supports `return`, `mdd`, `sharpe`, and `risk_return`, but not CR/Calmar or all-metric rank scoring.

These observations mean a new risk-oriented experiment cannot be achieved by script-level coefficient tuning alone.

## Target Behavior

Add an opt-in risk training variant:

- Outer actor reward mode: Sharpe-based segment reward.
- Controller reward mode: relative CR uplift against the baseline rollout.
- Model selection: multi-metric rank score, with single-metric best checkpoints retained for analysis.
- Seed sweep: broader NASDAQ and CSI-300 seed ranges to search for models that dominate multiple metrics.

The original training script and default reward behavior remain unchanged.

## Approach

Use an isolated opt-in design with conservative defaults:

```text
--outer_reward_mode return|sharpe
--controller_reward_mode return_uplift|relative_cr
--model_selection_metric sharpe|return|mdd|cr|rank_score
--inner_selection_metric sharpe|return|mdd|cr|rank_score
--controller_selection_metric risk_return|return|mdd|sharpe|cr|rank_score
```

Default values preserve current behavior:

```text
outer_reward_mode=return
controller_reward_mode=return_uplift
model_selection_metric=sharpe
inner_selection_metric=return
controller_selection_metric=risk_return
```

## Reward Design

### Outer Actor

When `outer_reward_mode=return`, keep the current daily log-return behavior.

When `outer_reward_mode=sharpe`, compute the outer decision reward over the holding segment rather than as a raw daily return:

```text
segment_sharpe = mean(segment_daily_log_returns) / (std(segment_daily_log_returns) + eps) * sqrt(252)
outer_reward = outer_sharpe_coef * clip(segment_sharpe, -outer_sharpe_clip, outer_sharpe_clip)
             + outer_return_floor_coef * segment_log_return
```

The small return floor prevents the actor from preferring low-volatility, near-zero-return segments. The clipping prevents very short or near-constant segments from producing unstable Sharpe values.

Implementation detail: store daily outer raw returns in the buffer as today, but alter the outer segment aggregation in `HRL_Buffer.finish_episode` based on `outer_reward_mode`. This keeps the segment-level SMDP training structure intact.

### Controller

When `controller_reward_mode=return_uplift`, keep current behavior.

When `controller_reward_mode=relative_cr`, compute the reward from controlled and baseline counterfactual statistics:

```text
baseline_cr = baseline_annualized_return / max(baseline_max_drawdown, eps)
controlled_cr = controlled_annualized_return / max(controlled_max_drawdown, eps)

reward = controller_cr_coef * clip(controlled_cr - baseline_cr, -controller_cr_clip, controller_cr_clip)
       + controller_return_floor_coef * (controlled_log_return - baseline_log_return)
       - normalized_max_switch_overflow_penalty
```

Use CR uplift rather than absolute CR so the controller learns whether switching improves the same window relative to continuing the baseline policy.

The return floor should be small. It is a guard against models that achieve high CR only by suppressing both risk and return.

## Selection And Reporting

Extend validation metrics with CR/Calmar:

```text
cr = ann_ret / max(max_dd, eps)
```

Add `rank_score` selection:

```text
rank_score =
  rank(total_return, higher better)
+ rank(sharpe, higher better)
+ rank(cr, higher better)
+ rank(max_drawdown, lower better)
```

The trainer should track:

- best total return checkpoint
- best Sharpe checkpoint
- best MDD checkpoint
- best CR checkpoint
- best rank-score checkpoint

The main seed-sweep output should include one summary table per market and one combined leaderboard. The leaderboard should make it clear whether a model wins all metrics or only wins the aggregate rank score.

## Seed Sweep Script

Add a new script:

`train_sh/run_end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep.sh`

Default first-round seeds:

```text
NAS_SEEDS="41 42 43 44 45 46 47 48 49 50"
SH_SEEDS="82 83 84 85 86 87 88 89 90 91"
```

Optional broader sweep:

```text
NAS_SEEDS="40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59"
SH_SEEDS="80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99"
```

Default output root:

`results/end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep`

The script must keep the same protected-output checks as the existing end-to-end script and must refuse to write into archived good-model roots.

## Data Flow

1. `run_hrl_training.py` parses the new reward and selection parameters.
2. Runtime config stores reward modes, clipping values, and coefficients.
3. `PPO_Env.step` continues to emit daily log returns and portfolio values.
4. `HRL_Buffer.finish_episode` converts outer segment returns into either return reward or Sharpe reward.
5. Controller counterfactual rollout computes baseline and controlled stats.
6. `controller_reward` chooses either return uplift or relative CR uplift.
7. Validation computes total return, annualized return, Sharpe, MDD, and CR.
8. Trainer saves both single-metric best checkpoints and rank-score best checkpoints.
9. Seed-sweep script runs NASDAQ and CSI-300 with expanded seeds and isolated output paths.

## Testing

Add focused tests before implementation:

- `controller_reward` preserves old return-uplift behavior by default.
- `controller_reward` computes relative CR uplift when the new mode is enabled.
- CR handles zero or tiny drawdown with an epsilon and clipping.
- `_validation_score` supports `cr` and `rank_score`.
- Outer segment reward remains unchanged for `outer_reward_mode=return`.
- Outer segment reward uses clipped Sharpe for `outer_reward_mode=sharpe`.
- Existing `run_end_to_end_hrl_controller_joint_nas49_sh90.sh` echo tests still pass unchanged.
- New seed-sweep script echo test verifies seeds, reward modes, selection metrics, and safe output root.

## Rollout Plan

Implement in small steps:

1. Add metric utilities for annualized return, MDD, Sharpe, and CR where training already computes validation metrics.
2. Add controller reward mode with tests.
3. Add outer Sharpe segment reward mode with tests.
4. Add CR/rank-score validation selection.
5. Add isolated seed-sweep script.
6. Run unit tests and an echo dry-run of both old and new scripts.
7. Start with a 10-seed first-round sweep; expand to 20 seeds only after confirming training stability.

## Risks And Mitigations

Short-window Sharpe can be noisy. Mitigate with clipping, epsilon, and a small return floor.

CR can explode when drawdown is near zero. Mitigate with epsilon and clipped CR uplift.

All-metric best may not exist because return and drawdown objectives conflict. Mitigate by saving single-metric winners and a rank-score winner, then choose the reporting checkpoint based on the final leaderboard.

The existing good run must remain reproducible. Mitigate by keeping current defaults unchanged and testing the original script output.
