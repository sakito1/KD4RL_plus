import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from torch.distributions import Categorical

# === 导入自定义模块 ===
from env import PPO_Env
from agent import HRL_PPO_Agent, HRL_Buffer
# 导入模型定义
from Components.PPO_model import FullModel
try:
    from Train.controller_pg import (
        CounterfactualStats,
        controller_pg_loss,
        controller_reward,
        overflow_switch_policy_loss,
        segment_budget_allows_switch,
    )
    from Train.controller_guidance import (
        analyze_guidance_windows,
        build_economic_guidance_labels,
        render_guidance_report,
    )
except ModuleNotFoundError:
    from controller_pg import (
        CounterfactualStats,
        controller_pg_loss,
        controller_reward,
        overflow_switch_policy_loss,
        segment_budget_allows_switch,
    )
    from controller_guidance import (
        analyze_guidance_windows,
        build_economic_guidance_labels,
        render_guidance_report,
    )
import utils.config as config


# ==============================================================================
# 1. 种子设置工具
# ==============================================================================
def set_seed(seed, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global Random Seed set to: {seed}")


def sample_switch_schedule(T, low=30, high=60):
    s = [0] * T
    s[0] = 1
    t = 0
    while t < T - 1:
        L = random.randint(low, high)
        t = min(t + L, T - 1)
        s[t] = 1
    return s


@dataclass(frozen=True)
class PhaseSpec:
    """Single source of truth for per-phase behavior."""
    use_schedule: bool
    inner_always_zero: bool
    monitor_always_forced: bool
    mask_monitor_update: bool
    use_hold_constraints: bool


def _prepare_controller_joint_baseline(trainer, controller_best_ckpt, final_best_ckpt, controller_pg_result):
    """Seed final best with the controller-PG checkpoint before optional joint finetune."""
    controller_best_path = os.path.join(trainer.model_dir, controller_best_ckpt)
    final_best_path = os.path.join(trainer.model_dir, final_best_ckpt)
    baseline_score = -np.inf
    pg_updates = int((controller_pg_result or {}).get("updates", 0) or 0)
    if pg_updates > 0:
        try:
            baseline_score = float((controller_pg_result or {}).get("best_score", -np.inf))
        except (TypeError, ValueError):
            baseline_score = -np.inf

    loaded_controller_best = False
    if os.path.exists(controller_best_path):
        loaded_controller_best = bool(trainer._load_model(controller_best_ckpt))
        if getattr(trainer, "logger", None):
            trainer.logger.info("   ↺ Loaded best controller before end-to-end joint finetune.")
    elif os.path.exists(final_best_path):
        trainer._load_model(final_best_ckpt)
        if getattr(trainer, "logger", None):
            trainer.logger.info("   ↺ Loaded current best model before end-to-end joint finetune.")

    if loaded_controller_best and np.isfinite(baseline_score):
        trainer.save_model(final_best_ckpt)
        if getattr(trainer, "logger", None):
            trainer.logger.info(
                "       (Seeded final best with Controller-PG Best: %.4f)",
                baseline_score,
            )
    return baseline_score


# ==============================================================================
# 2. 网络包装器 (Model Wrapper)
# ==============================================================================
class HRL_Networks(nn.Module):
    def __init__(self, ssm_dim, num_stocks, cfg):
        super(HRL_Networks, self).__init__()
        raw_feature_dim = len(cfg.dataset['features_name']) if hasattr(cfg, 'dataset') else 102
        port_state_dim = 6
        HIDDEN_DIM = 32
        CONTROLLER_HIDDEN_DIM = int(getattr(cfg, "controller_hidden_dim", HIDDEN_DIM))

        self.model = FullModel(
            monitor_args=dict(
                z_dim=ssm_dim,
                h_dim=ssm_dim,
                port_state_dim=port_state_dim,
                hidden_dim=CONTROLLER_HIDDEN_DIM,
                action_dim=num_stocks,
                min_hold=getattr(cfg, "min_hold", 20),
                max_hold=getattr(cfg, "max_hold", 40),
                tau_min=getattr(cfg, "controller_tau_min", 0.5),
                tau_max=getattr(cfg, "controller_tau_max", 0.9),
                policy_temperature=getattr(cfg, "controller_policy_temperature", 10.0),
                ret_scale=getattr(cfg, "controller_state_return_scale", 0.05),
                drawdown_scale=getattr(cfg, "controller_state_drawdown_scale", 0.10),
                asset_in_dim=raw_feature_dim,
                controller_window=getattr(cfg, "controller_window", 15),
                weight_floor=getattr(cfg, "controller_weight_floor", 1e-6),
                eval_switch_threshold=getattr(cfg, "controller_eval_switch_threshold", 0.5),
                init_exit_bias=getattr(cfg, "controller_init_exit_bias", None),
                switch_adv_logit_coef=getattr(cfg, "controller_switch_adv_logit_coef", 0.0),
                switch_adv_logit_scale=getattr(cfg, "controller_switch_adv_logit_scale", 0.02),
                switch_adv_logit_detach=getattr(cfg, "controller_switch_adv_logit_detach", False),
            ),
            outer_actor_args=dict(
                lstm_dim=HIDDEN_DIM,
                caan_dim=HIDDEN_DIM,
                in_dim=raw_feature_dim,
                hidden_dim=HIDDEN_DIM,
                trade_num=cfg.trade_num,
            ),
            outer_critic_args=dict(
                num_nodes=num_stocks,
            ),
            inner_actor_args=dict(
                in_features=raw_feature_dim,
                hidden_dim=HIDDEN_DIM,
                max_boundary=getattr(cfg, 'inner_max_boundary', 0.5),
                dropout=0.1,
            ),
            inner_critic_args=dict(
                in_features=raw_feature_dim,
                hidden_dim=HIDDEN_DIM,
            ),
        )

        # [集成] 只暴露三个模块：outer / inner / mon
        self.outer = self.model.outer
        self.inner = self.model.inner
        self.mon = self.model.mon

    def forward(self, *args, **kwargs):
        raise RuntimeError("HRL_Networks is a wrapper; use net.outer/net.inner/net.mon.")


# ==============================================================================
# 3. 训练管理器 (Trainer)
# ==============================================================================
class HRL_Trainer:
    def __init__(self, agent, env, buffer, config, logger):
        self.agent = agent
        self.buffer = buffer
        self.cfg = config
        self.device = config.device
        self.logger = logger
        self.env = env

        # 路径管理
        self.run_dir = os.path.join(config.cun_path, f"seed_{config.seed}")
        self.model_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.model_dir, exist_ok=True)
        self._save_config()

    def _save_config(self):
        cfg_dict = {k: str(v) for k, v in vars(self.cfg).items() if not k.startswith('__')}
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(cfg_dict, f, indent=4)

    # 计算详细的金融指标
    def _compute_metrics(self, history):
        df = pd.DataFrame(history, columns=['value'])
        df['ret'] = df['value'].pct_change().fillna(0)

        if len(df) < 2:
            return {'sharpe': 0.0, 'total_ret': 0.0, 'ann_ret': 0.0, 'max_dd': 0.0, 'final_value': df['value'].iloc[-1]}

        total_ret = (df['value'].iloc[-1] - df['value'].iloc[0]) / df['value'].iloc[0]

        mean_ret = df['ret'].mean()
        std_ret = df['ret'].std()

        ann_ret = mean_ret * 252
        ann_vol = std_ret * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-8) if ann_vol > 1e-8 else 0.0

        roll_max = df['value'].cummax()
        drawdown = (roll_max - df['value']) / roll_max
        max_dd = drawdown.max()

        return {
            'sharpe': sharpe,
            'total_ret': total_ret,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'max_dd': max_dd,
            'final_value': df['value'].iloc[-1]
        }

    @staticmethod
    def _validation_score(metrics, cfg, phase: str = "joint") -> float:
        if phase == "controller":
            metric_name = str(getattr(cfg, "controller_selection_metric", "risk_return"))
        elif phase == "warmup_inner":
            metric_name = str(getattr(cfg, "inner_selection_metric", "return"))
        else:
            metric_name = str(getattr(cfg, "model_selection_metric", "sharpe"))

        if metric_name in ("return", "total_ret", "ret"):
            return float(metrics["total_ret"])
        if metric_name in ("mdd", "min_mdd"):
            return -float(metrics["max_dd"])
        if metric_name in ("risk_return", "controller_default"):
            mdd_coef = float(getattr(cfg, "controller_mdd_coef", 2.0))
            return_coef = float(getattr(cfg, "controller_return_coef", 0.5))
            return -mdd_coef * float(metrics["max_dd"]) + return_coef * float(metrics["total_ret"])
        if metric_name == "sharpe":
            return float(metrics["sharpe"])
        raise ValueError(f"Unknown validation selection metric: {metric_name}")

    # ==============================================================================
    # 核心 Episode 运行逻辑 (支持 Phase 和 强制换仓)
    # ==============================================================================
    def run_episode(self, env, *, mode='train', phase='warmup_outer', fixed_cycle=None, disable_inner=False,
                    use_rule_switch=False, rollout_update_steps=None, auto_update_phase=None,
                    train_monitor=None, rollout_buffer=None, explicit_episode_window=None):
        if explicit_episode_window is not None:
            obs = env.reset_at(*explicit_episode_window)
        else:
            obs = env.reset()
        active_buffer = rollout_buffer if rollout_buffer is not None else self.buffer
        is_train = (mode == 'train')
        if is_train:
            # 支持多 episode 累积更新：标记当前 episode 的起点
            if hasattr(active_buffer, 'mark_episode_start'):
                active_buffer.mark_episode_start()

        spec = self._get_phase_spec(phase)

        T = max(1, int(getattr(env, 'current_episode_len', env.episode_len)))
        m_hold = int(getattr(self.cfg, 'max_hold', 60))
        switch_schedule = sample_switch_schedule(T,m_hold, m_hold) if spec.use_schedule else None

        ret_stats = {
            'total': 0.0,
            'history': [env.portfolio_value.item()],
            'episode_start': int(getattr(env, 'current_episode_start', env.day)),
            'episode_end': int(getattr(env, 'current_episode_end', env.stop_step)),
        }
        if hasattr(env, 'all_dates'):
            ret_stats['start_date'] = str(env.all_dates[ret_stats['episode_start']].date())
            ret_stats['end_date'] = str(env.all_dates[ret_stats['episode_end']].date())
        step_idx = 0

        # 统计计数器
        switch_count = 0
        switch_free_count = 0
        forced_hold_count = 0
        forced_switch_count = 0
        forced_schedule_count = 0
        last_switch_step = 0

        # [调试用] 记录每天的P值
        debug_p_list = []
        controller_eval_probs = []
        controller_eval_diag_enabled = (
                (not is_train)
                and bool(getattr(self.cfg, "controller_eval_diagnostics", False))
        )
        controller_eval_diag_thresholds = getattr(self.cfg, "controller_eval_diag_thresholds", None)

        # [Rule] 初始化规则计数器
        rule_consecutive_low = 0
        RULE_THRESHOLD = getattr(self.cfg, 'rule_switch_threshold', 0.5)

        # [Fix] 从配置读取最小/最大持仓，确保规则模式也遵守
        CFG_MIN_HOLD = int(getattr(self.cfg, 'min_hold', 5))  # 默认5天
        CFG_MAX_HOLD = int(getattr(self.cfg, 'max_hold', 60))  # 默认60天
        rollout_update_steps = int(rollout_update_steps or 0)
        stored_since_update = 0
        loss_accum = {}
        update_count = 0

        def _merge_loss(loss_log):
            nonlocal update_count
            if not loss_log:
                return
            update_count += 1
            for k, v in loss_log.items():
                loss_accum.setdefault(k, []).append(float(v))

        def _bootstrap_values(next_obs, done_flag):
            if done_flag:
                return {'val_inn': 0.0, 'val_out': 0.0, 'val_mon': 0.0}
            with torch.no_grad():
                last_out = self.agent.get_action(
                    next_obs,
                    mode=mode,
                    force_switch=1,
                    force_inner_zero=True,
                    force_locked=True,
                )
            return {
                'val_inn': last_out['val_inn'].item(),
                'val_out': last_out['val_out'].item(),
                'val_mon': last_out['val_mon'].item(),
            }

        def _finish_rollout(next_obs, done_flag):
            if len(active_buffer.data.get('rew_mon', [])) <= 0:
                return
            active_buffer.finish_episode(_bootstrap_values(next_obs, done_flag))
            if auto_update_phase is None:
                return

            loss_log = self.agent.update(
                active_buffer.get_batch(),
                phase=auto_update_phase,
                train_monitor=train_monitor,
            )
            _merge_loss(loss_log)
            active_buffer.clear()
            if hasattr(active_buffer, 'mark_episode_start'):
                active_buffer.mark_episode_start()
            if (
                    bool(getattr(self.cfg, "clear_cuda_cache_on_update", False))
                    and getattr(self.device, "type", str(self.device)) == "cuda"
            ):
                torch.cuda.empty_cache()

        while True:
            duration = step_idx - last_switch_step

            # 1. 计算默认状态 (此处的计算结果在 rule 模式下会被覆盖，但用作参考)
            force_switch, force_locked = self._compute_force_switch_locked(
                spec=spec,
                phase=phase,
                step_idx=step_idx,
                duration=duration,
                is_train=is_train,
                switch_schedule=switch_schedule,
                fixed_cycle=fixed_cycle,
                current_segments=switch_count,
                rollout_len=T,
            )

            # 2. [规则接管逻辑]
            if use_rule_switch:
                # 获取当前持仓 p
                current_p = obs.get('held_p', torch.tensor([1.0], device=self.device)).item()

                # 记录调试信息
                debug_p_list.append({
                    'step': step_idx,
                    'held_p': current_p,
                    'duration': duration,
                    'is_locked': duration < CFG_MIN_HOLD,
                    'threshold': RULE_THRESHOLD
                })

                # 更新计数器
                if current_p < RULE_THRESHOLD:
                    rule_consecutive_low += 1
                else:
                    rule_consecutive_low = 0

                # === 规则决策 (优先级逻辑 - 修复版) ===

                # 优先级 A: 第 0 步必须 Switch (建仓)
                if step_idx == 0:
                    force_switch = 1
                    force_locked = True

                # 优先级 B: 持有期不足 min_hold -> 强制 Hold (保护期)
                elif duration < CFG_MIN_HOLD:
                    force_switch = 0
                    force_locked = True

                # 优先级 C: [关键修复] 超过 max_hold -> 强制 Switch (到期平仓)
                # 之前的代码漏掉了这一条，导致无限持仓
                elif duration >= CFG_MAX_HOLD:
                    force_switch = 1
                    force_locked = True

                # 优先级 D: 连续下跌触发止损 -> 强制 Switch
                elif rule_consecutive_low >= self.cfg.max_rule_consecutive_low:
                    force_switch = 1
                    force_locked = True

                # 优先级 E: 其他情况 -> 强制 Hold
                else:
                    force_switch = 0
                    force_locked = True

            # inner gating
            force_inner_zero = bool(spec.inner_always_zero)
            if disable_inner:
                force_inner_zero = True

            # monitor gating: if phase wants monitor always forced but we didn't compute a forced value
            if spec.monitor_always_forced and force_switch is None:
                # fall back to HOLD to avoid accidental monitor sampling
                force_switch = 0

            # mask monitor update when phase freezes monitor
            if spec.mask_monitor_update:
                force_locked = True

            # === Agent forward ===
            # 说明：
            # 1. 如果 use_rule_switch=True，force_switch 必然是 0 或 1，Monitor 网络会被跳过。
            # 2. 如果 use_rule_switch=False (且无其他强制)，force_switch 为 None，Monitor 网络会被调用。
            # 3. Outer 和 Inner 无论 force_switch 是什么，都会正常计算。
            with torch.no_grad():
                out = self.agent.get_action(
                    obs,
                    mode=mode,
                    force_switch=force_switch,
                    force_inner_zero=force_inner_zero,
                    force_locked=force_locked,
                )
                if controller_eval_diag_enabled and force_switch is None:
                    diag_stats = self.agent.net.mon.decision_stats(
                        obs['weights_drift'], obs['port_state'],
                        switch_action=out['act_out'],
                        asset_state=obs.get('outer_state'),
                        hold_exec_weights=out.get('controller_hold_exec'),
                        switch_exec_weights=out.get('controller_switch_exec'),
                        remaining_horizon=out.get('controller_remaining_horizon'),
                    )
                    controller_eval_probs.extend(
                        diag_stats["exit_prob"].detach().view(-1).cpu().tolist()
                    )

            base_used = out['base_used']
            weights_exec = out['weights_exec']
            outer_action = out['act_out']
            is_switch_action = (out['act_mon'].item() == 1)

            # Controller reward comes from env.step as daily portfolio log-return.
            # The old outer-critic counterfactual reward is deliberately disabled.
            cf_switch_advantage = torch.zeros((), device=self.device)

            # === Update Counters ===
            is_scheduled_active = (
                    spec.use_schedule and switch_schedule is not None and step_idx < len(switch_schedule))

            if is_switch_action:
                switch_count += 1
                last_switch_step = step_idx

                # [修改点 3] 发生切换后，重置规则计数器
                # 因为换了新仓位，之前的连续下跌记录失效，重新开始计算
                if use_rule_switch:
                    rule_consecutive_low = 0

                # Determine if this was a forced switch
                is_forced_by_schedule = (is_scheduled_active and bool(switch_schedule[step_idx]))
                # 无论是规则触发，还是 Schedule 触发，只要 force_switch=1 都算 forced
                is_forced_by_logic = (force_switch == 1)

                if is_forced_by_schedule:
                    forced_switch_count += 1
                    forced_schedule_count += 1
                elif is_forced_by_logic:
                    forced_switch_count += 1
                else:
                    switch_free_count += 1

            else:
                # It is a Hold
                is_forced_hold_schedule = (is_scheduled_active and not bool(switch_schedule[step_idx]))
                is_forced_hold_logic = (force_switch == 0)

                if is_forced_hold_schedule or is_forced_hold_logic:
                    forced_hold_count += 1

            # === hard invariants ===
            if force_inner_zero:
                if not torch.allclose(weights_exec, base_used, atol=1e-6, rtol=1e-5):
                    # Warning logic instead of error to keep running
                    pass

            next_obs, _, done, info = env.step(weights_exec, base_used, outer_action=outer_action,
                                               is_switch=is_switch_action)
            info['rewards']['monitor_cf_switch_advantage'] = cf_switch_advantage.item()

            if is_train:
                transition = {
                    'outer_state': obs['outer_state'],
                    'inner_state': out.get('inner_state_used', obs['inner_state']),
                    'inner_base_used': out.get('inner_base_used', base_used),
                    'inner_weights_drift': out.get('inner_weights_drift', obs['weights_drift']),
                    'inner_indices': out.get('inner_indices', torch.empty(0, device=self.device, dtype=torch.long)),
                    'weights_drift': obs['weights_drift'],
                    'base_drift': obs['base_drift'],
                    'port_state': obs['port_state'],

                    'act_mon': out['act_mon'],
                    'logp_mon': out['logp_mon'],
                    'val_mon': out['val_mon'],

                    'act_out_raw': out['act_out_raw'],
                    'act_out': out['act_out'],
                    'logp_out': out['logp_out'],
                    'val_out': out['val_out'],

                    'act_inn_raw': out['act_inn_raw'],
                    'logp_inn': out['logp_inn'],
                    'val_inn': out['val_inn'],

                    'base_used': base_used,
                    'weights_exec': weights_exec,

                    'is_switch': out['act_mon'],
                    'is_locked': out['is_locked'],
                    'dones': done,

                    'rew_mon': info['rewards']['monitor_reward'],
                    'rew_alpha': info['rewards']['inner_reward'],
                    'rew_outer_raw': info['rewards']['outer_step_reward'],
                    'outer_stock_return_target': info['outer_stock_return_target'],
                    'controller_hold_return_target': info['controller_hold_return_target'],
                    'controller_hold_mdd_target': info['controller_hold_mdd_target'],
                }
                if float(getattr(self.cfg, 'inner_pred_coef', 0.0)) > 0.0:
                    inner_indices = out.get('inner_indices')
                    inner_target_full = info.get(
                        'inner_stock_return_target',
                        info.get('inner_next_return_target'),
                    )
                    if inner_indices is not None and isinstance(inner_target_full, torch.Tensor):
                        target_full = inner_target_full.unsqueeze(0).expand(inner_indices.shape[0], -1)
                        inner_target_used = torch.gather(target_full, dim=1, index=inner_indices)
                    else:
                        inner_target_used = inner_target_full
                    transition['inner_stock_return_target'] = inner_target_used
                    transition['inner_next_return_target'] = inner_target_used

                if float(getattr(self.cfg, 'controller_sup_coef', getattr(self.cfg, 'monitor_sup_coef', 0.0))) > 0.0:
                    transition['controller_switch_label'] = info['controller_switch_label']
                    transition['controller_sup_weight'] = info['controller_sup_weight']

                active_buffer.store_daily(transition)
                stored_since_update += 1

                if (
                        auto_update_phase is not None
                        and rollout_update_steps > 0
                        and stored_since_update >= rollout_update_steps
                        and not done):
                    _finish_rollout(next_obs, done_flag=False)
                    stored_since_update = 0

            obs = next_obs
            ret_stats['history'].append(info['portfolio_value'])

            if done:
                # [调试输出] 保存P值记录
                if use_rule_switch and len(debug_p_list) > 0:
                    debug_df = pd.DataFrame(debug_p_list)
                    debug_filename = os.path.join(self.run_dir, f"debug_p_values_th{RULE_THRESHOLD}.csv")
                    debug_df.to_csv(debug_filename, index=False)

                    if step_idx > 0:
                        avg_p = debug_df['held_p'].mean()
                        low_ratio = (debug_df['held_p'] < RULE_THRESHOLD).mean()
                        self.logger.info(
                            f"   [DEBUG Analysis] Th={RULE_THRESHOLD} | Avg Held_P={avg_p:.4f} | Days < Th Ratio: {low_ratio * 100:.1f}%")

                if is_train:
                    _finish_rollout(obs, done_flag=True)
                break

            step_idx += 1

        ret_stats['total'] = (ret_stats['history'][-1] - ret_stats['history'][0]) / (ret_stats['history'][0] + 1e-8)
        ret_stats['switch_count'] = switch_count
        ret_stats['switch_free_count'] = switch_free_count
        ret_stats['forced_hold_count'] = forced_hold_count
        ret_stats['forced_switch_count'] = forced_switch_count
        ret_stats['forced_schedule_count'] = forced_schedule_count
        ret_stats['total_steps'] = step_idx
        ret_stats['update_count'] = update_count
        ret_stats['loss_log'] = {
            k: float(np.mean(v)) if len(v) > 0 else 0.0
            for k, v in loss_accum.items()
        }
        if controller_eval_probs:
            summary = self._controller_exit_prob_summary(
                controller_eval_probs,
                thresholds=controller_eval_diag_thresholds,
            )
            ret_stats['controller_exit_prob_summary'] = summary
            threshold_bits = []
            for key, value in summary.items():
                if key.startswith("gt_"):
                    threshold_bits.append(f"{key}={value}")
            self.logger.info(
                "Controller eval exit_prob: count=%s mean=%.6f p50=%.6f p75=%.6f p95=%.6f "
                "hard_switch_rate=%.4f max=%.6f %s",
                summary.get("count", 0),
                summary.get("mean", 0.0),
                summary.get("p50", 0.0),
                summary.get("p75", 0.0),
                summary.get("p95", 0.0),
                summary.get("hard_switch_rate", 0.0),
                summary.get("max", 0.0),
                " ".join(threshold_bits),
            )
        return ret_stats

    def _get_phase_spec(self, phase: str) -> PhaseSpec:
        """Map phase string to deterministic trainer behavior."""
        # Warmups
        if phase == 'warmup_outer':
            return PhaseSpec(use_schedule=True, inner_always_zero=True,
                             monitor_always_forced=True, mask_monitor_update=True,
                             use_hold_constraints=False)
        if phase == 'warmup_inner':
            return PhaseSpec(use_schedule=True, inner_always_zero=False,
                             monitor_always_forced=True, mask_monitor_update=True,
                             use_hold_constraints=False)
        if phase == 'warmup_monitor':
            return PhaseSpec(use_schedule=False, inner_always_zero=False,
                             monitor_always_forced=False, mask_monitor_update=False,
                             use_hold_constraints=True)

        # Round-robin phases
        if phase.startswith('round_'):
            # during round_* we still enforce constraints, and keep monitor free unless explicitly training others
            mod = phase.split('_', 1)[1]
            if mod == 'outer':
                return PhaseSpec(use_schedule=False, inner_always_zero=False,
                                 monitor_always_forced=False, mask_monitor_update=False,
                                 use_hold_constraints=True)
            if mod == 'inner':
                return PhaseSpec(use_schedule=False, inner_always_zero=False,
                                 monitor_always_forced=False, mask_monitor_update=False,
                                 use_hold_constraints=True)
            if mod == 'monitor':
                return PhaseSpec(use_schedule=False, inner_always_zero=False,
                                 monitor_always_forced=False, mask_monitor_update=False,
                                 use_hold_constraints=True)

        # Joint
        return PhaseSpec(use_schedule=False, inner_always_zero=False,
                         monitor_always_forced=False, mask_monitor_update=False,
                         use_hold_constraints=True)

    def _compute_force_switch_locked(
            self,
            *,
            spec: PhaseSpec,
            phase: str,
            step_idx: int,
            duration: int,
            is_train: bool,
            switch_schedule: Optional[list],
            fixed_cycle: Optional[int],
            current_segments: int = 0,
            rollout_len: Optional[int] = None,
    ):
        """Return (force_switch, force_locked) for this step."""
        force_switch = None
        force_locked = False

        # Priority 1: first step must switch to build initial position
        if step_idx == 0:
            return 1, True

        # Priority 2: optional fixed_cycle (eval/testing)
        if fixed_cycle is not None:
            if duration < fixed_cycle:
                return 0, True
            return 1, True

        if spec.use_hold_constraints and bool(getattr(self.cfg, "controller_no_hold_constraints", False)):
            hard_max_switches = int(getattr(self.cfg, "controller_hard_max_switches", 0) or 0)
            if hard_max_switches > 0 and int(current_segments) >= hard_max_switches:
                return 0, True
            max_hold = (
                self._controller_train_max_hold(fixed_cycle=fixed_cycle, rollout_len=rollout_len)
                if is_train
                else self._controller_eval_max_hold()
            )
            if max_hold > 0 and duration >= max_hold:
                return 1, True
            decision_mode = self._controller_decision_mode(is_train=is_train)
            if decision_mode == "stride":
                stride = self._controller_decision_stride(is_train=is_train)
                if int(duration) % max(1, stride) != 0:
                    return 0, True
                return None, False
            if decision_mode == "fixed_window":
                window = max(0, int(getattr(self.cfg, "controller_fixed_decision_window", 0)))
                if duration < max(1, max_hold - window):
                    return 0, True
                return None, False
            if (
                    not is_train
                    and bool(getattr(self.cfg, "controller_eval_force_max_hold", False))
                    and max_hold > 0
                    and duration >= max_hold
            ):
                return 1, True
            return None, False

        # Priority 3: warmup schedule
        if spec.use_schedule:
            if not switch_schedule:
                raise RuntimeError(f"Phase '{phase}' expects schedule but switch_schedule is None")
            forced = 1 if bool(switch_schedule[step_idx]) else 0
            # schedule forcing is not 'locked' for training mask purposes; mask handled by spec.mask_monitor_update
            return forced, False

        # Priority 4: holding constraints (30-60 or min_hold)
        if spec.use_hold_constraints:
            min_hold = int(getattr(self.cfg, 'min_hold', 1))
            max_hold = int(getattr(self.cfg, 'max_hold', 60))

            # minimum holding days after a switch
            if duration < min_hold:
                return 0, True
            # maximum holding days -> must switch
            if duration >= max_hold:
                return 1, True

            controller_check_stride = int(getattr(self.cfg, 'controller_check_stride_days', 1))
            controller_check_stride = max(1, controller_check_stride)
            if controller_check_stride > 1:
                free_elapsed = duration - min_hold
                if free_elapsed % controller_check_stride != 0:
                    return 0, True

            # free decision window
            force_switch = None
            force_locked = False

            # optional explore bias to generate more switch samples
            if is_train:
                p = float(getattr(self.cfg, 'monitor_switch_explore_prob', 0.0))
                if p > 0 and random.random() < p:
                    force_switch = 1
                    force_locked = False

            # warmup_monitor optional small probability to stabilize by forcing hold
            if phase == 'warmup_monitor' and is_train and force_switch is None:
                p_hold = float(getattr(self.cfg, 'warmup_monitor_force_hold_prob', 0.0))
                if p_hold > 0 and random.random() < p_hold:
                    force_switch = 0
                    force_locked = False

            return force_switch, force_locked

        # Default: free
        return None, False

    @staticmethod
    def _counterfactual_stats(history, turnover_sum=0.0, free_switch_count=0, segment_count=0):
        values = np.asarray(history, dtype=np.float64)
        values = np.maximum(values, 1e-12)
        peaks = np.maximum.accumulate(values)
        max_dd = float(np.max((peaks - values) / (peaks + 1e-12))) if values.size > 0 else 0.0
        log_return = float(np.log(values[-1] / values[0])) if values.size > 1 else 0.0
        daily_log_returns = np.diff(np.log(values)) if values.size > 1 else np.asarray([], dtype=np.float64)
        downside_loss = float(np.maximum(-daily_log_returns, 0.0).sum()) if daily_log_returns.size > 0 else 0.0
        trading_days = max(1, int(values.size) - 1)
        return CounterfactualStats(
            log_return=log_return,
            max_drawdown=max_dd,
            turnover=float(turnover_sum),
            free_switch_count=int(free_switch_count),
            segment_count=int(segment_count),
            trading_days=trading_days,
            downside_loss=downside_loss,
        )

    @staticmethod
    def _controller_exit_prob_summary(exit_probs, thresholds=None):
        values = np.asarray(list(exit_probs), dtype=np.float64)
        if values.size <= 0:
            return {"count": 0}
        thresholds = [0.5, 0.49, 0.45, 0.4] if thresholds is None else thresholds

        def _threshold_key(threshold):
            token = ("%g" % float(threshold)).replace("-", "m").replace(".", "p")
            return f"gt_{token}"

        summary = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "hard_switch_rate": float(np.mean(values > 0.5)),
        }
        for threshold in thresholds:
            threshold = float(threshold)
            summary[_threshold_key(threshold)] = int(np.sum(values > threshold))
        return summary

    @staticmethod
    def _controller_switch_advantage_summary(values, exit_probs=None):
        values = np.asarray(list(values), dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size <= 0:
            return {"count": 0}
        positive = values > 0.0
        summary = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "positive_count": int(np.sum(positive)),
            "positive_frac": float(np.mean(positive)),
        }
        if exit_probs is not None:
            probs = np.asarray(list(exit_probs), dtype=np.float64)
            n = min(values.size, probs.size)
            if n > 0:
                paired_values = values[:n]
                paired_probs = probs[:n]
                valid = np.isfinite(paired_values) & np.isfinite(paired_probs)
                if np.any(valid):
                    paired_values = paired_values[valid]
                    paired_probs = paired_probs[valid]
                    pos_probs = paired_probs[paired_values > 0.0]
                    neg_probs = paired_probs[paired_values <= 0.0]
                    pos_mean = float(np.mean(pos_probs)) if pos_probs.size > 0 else 0.0
                    neg_mean = float(np.mean(neg_probs)) if neg_probs.size > 0 else 0.0
                    summary.update({
                        "exit_prob_count": int(paired_probs.size),
                        "positive_exit_prob_count": int(pos_probs.size),
                        "negative_exit_prob_count": int(neg_probs.size),
                        "positive_exit_prob_mean": pos_mean,
                        "negative_exit_prob_mean": neg_mean,
                        "exit_prob_gap": float(pos_mean - neg_mean),
                    })
        return summary

    def _controller_uses_switch_advantage_targets(self) -> bool:
        return (
                bool(getattr(self.cfg, "controller_compute_switch_advantage", False))
                or float(getattr(self.cfg, "controller_aux_switch_adv_coef", 0.0)) > 0.0
                or float(getattr(self.cfg, "controller_local_adv_coef", 0.0)) > 0.0
        )

    def _controller_actual_holdings_switch_advantage(self, env, obs, switch_exec):
        actual_hold = obs["weights_drift"].detach().view(-1)
        if hasattr(env, "_normalize"):
            actual_hold = env._normalize(actual_hold)
            switch_weights = env._normalize(switch_exec.detach().view(-1))
        else:
            actual_hold = actual_hold / actual_hold.sum().clamp_min(1e-8)
            switch_weights = switch_exec.detach().view(-1)
            switch_weights = switch_weights / switch_weights.sum().clamp_min(1e-8)

        cfg_max_hold = int(getattr(getattr(self, "cfg", None), "max_hold", 60))
        max_hold = int(getattr(env, "max_hold", cfg_max_hold))
        t_held = int(getattr(env, "t_held", 0))
        horizon = max(1, max_hold - t_held)
        start_day = int(getattr(env, "day", 0))
        hold_return, _ = env._future_portfolio_return_and_max_drawdown(
            actual_hold.detach(),
            start_day,
            horizon,
        )
        switch_return, _ = env._future_portfolio_return_and_max_drawdown(
            switch_weights.detach(),
            start_day,
            horizon,
        )
        switch_turnover = torch.sum(torch.abs(switch_weights - actual_hold))
        transaction_cost = float(getattr(env, "transaction_cost_pct", 0.0))
        return switch_return - hold_return - switch_turnover * transaction_cost

    def _controller_decision_stride(self, *, is_train: bool, epoch: int = None) -> int:
        if is_train:
            schedule = getattr(self.cfg, "controller_decision_stride_schedule", None)
            if schedule:
                idx = 0 if epoch is None else min(max(0, int(epoch)), len(schedule) - 1)
                return max(1, int(schedule[idx]))
            return max(1, int(getattr(self.cfg, "controller_decision_stride", 1)))
        eval_stride = int(getattr(self.cfg, "controller_eval_decision_stride", 0) or 0)
        if eval_stride > 0:
            return max(1, eval_stride)
        return max(1, int(getattr(self.cfg, "controller_decision_stride", 1)))

    def _controller_decision_mode(self, *, is_train: bool) -> str:
        mode = str(getattr(self.cfg, "controller_decision_mode", "daily"))
        if not is_train:
            eval_mode = getattr(self.cfg, "controller_eval_decision_mode", None)
            if eval_mode:
                mode = str(eval_mode)
        return mode

    def _deterministic_inner_exec(self, obs, base_used, weights_drift):
        inner_state_used, inner_base_used, inner_weight_drift, inner_indices = self.agent._select_inner_inputs(
            obs, base_used, weights_drift
        )
        if hasattr(self.agent.net.inner, "build_inner_action_simple"):
            alpha = float(getattr(self.cfg, "inner_max_boundary", 1.0))
            w_new_sel, _, _, _, _ = self.agent.net.inner.build_inner_action_simple(
                inner_state_used,
                inner_base_used,
                inner_weight_drift,
                alpha=alpha,
                deterministic=True,
            )
            return self.agent._scatter_selected_weights(w_new_sel, inner_indices, base_used.shape[1])

        act_inn_raw, _, _ = self.agent.net.inner.pi(
            inner_state_used,
            inner_base_used,
            inner_weight_drift,
            deterministic=True,
        )
        action_for_calc = torch.clamp(act_inn_raw, -3.0, 3.0)
        adjusted = inner_base_used * torch.exp(action_for_calc * self.cfg.inner_max_boundary)
        w_new_sel = adjusted / (adjusted.sum(dim=1, keepdim=True) + 1e-12)
        return self.agent._scatter_selected_weights(w_new_sel, inner_indices, base_used.shape[1])

    def _controller_exec_weights(self, obs, base_used, weights_drift, *, disable_inner: bool = False):
        if bool(disable_inner):
            return base_used.detach()
        return self._deterministic_inner_exec(obs, base_used, weights_drift)

    def _run_fixed_hrl_window(self, env, start_idx: int, stop_idx: int, fixed_cycle: int, disable_inner: bool = False):
        obs = env.reset_at(start_idx, stop_idx)
        history = [env.portfolio_value.item()]
        turnover_sum = 0.0
        segment_count = 0
        last_switch_step = 0
        step_idx = 0

        self.agent.net.eval()
        with torch.no_grad():
            while True:
                duration = step_idx - last_switch_step
                force_switch = 1 if step_idx == 0 or duration >= int(fixed_cycle) else 0
                out = self.agent.get_action(
                    obs,
                    mode="eval",
                    force_switch=force_switch,
                    force_inner_zero=bool(disable_inner),
                    force_locked=True,
                )
                is_switch = bool(out["act_mon"].item() == 1)
                if is_switch:
                    segment_count += 1
                    last_switch_step = step_idx
                turnover_sum += float(torch.sum(torch.abs(out["weights_exec"] - obs["weights_drift"])).item())
                next_obs, _, done, info = env.step(
                    out["weights_exec"],
                    out["base_used"],
                    outer_action=out["act_out"],
                    is_switch=is_switch,
                )
                history.append(info["portfolio_value"])
                if done:
                    break
                obs = next_obs
                step_idx += 1

        return self._counterfactual_stats(
            history,
            turnover_sum=turnover_sum,
            free_switch_count=0,
            segment_count=segment_count,
        ), history

    def _finish_pg_segment(self, segment_logps, episode_segment_logps):
        if len(segment_logps) > 0:
            if isinstance(segment_logps[0], dict):
                episode_segment_logps.append(list(segment_logps))
            else:
                episode_segment_logps.append(torch.stack(segment_logps).mean())
            segment_logps.clear()

    @staticmethod
    def _detach_controller_record(obs, act_out, act_mon, target_return=None, target_mdd=None,
                                  sup_label=None, sup_weight=None, free_switch_index=0,
                                  switch_advantage=None, hold_exec_weights=None,
                                  switch_exec_weights=None, remaining_horizon=None):
        return {
            "weights_drift": obs["weights_drift"].detach(),
            "port_state": obs["port_state"].detach(),
            "asset_state": obs.get("outer_state").detach() if isinstance(obs.get("outer_state"), torch.Tensor) else obs.get("outer_state"),
            "switch_action": act_out.detach(),
            "action": act_mon.detach().view(-1),
            "free_switch_index": int(free_switch_index),
            "target_return": target_return.detach().view(-1) if isinstance(target_return, torch.Tensor) else None,
            "target_mdd": target_mdd.detach().view(-1) if isinstance(target_mdd, torch.Tensor) else None,
            "sup_label": sup_label.detach().view(-1) if isinstance(sup_label, torch.Tensor) else None,
            "sup_weight": sup_weight.detach().view(-1) if isinstance(sup_weight, torch.Tensor) else None,
            "switch_advantage": switch_advantage.detach().view(-1) if isinstance(switch_advantage, torch.Tensor) else None,
            "hold_exec_weights": (
                hold_exec_weights.detach()
                if isinstance(hold_exec_weights, torch.Tensor)
                else None
            ),
            "switch_exec_weights": (
                switch_exec_weights.detach()
                if isinstance(switch_exec_weights, torch.Tensor)
                else None
            ),
            "remaining_horizon": (
                remaining_horizon.detach().view(-1)
                if isinstance(remaining_horizon, torch.Tensor)
                else None
            ),
        }

    @staticmethod
    def _controller_remaining_horizon(env, *, dtype, device):
        max_hold = max(1, int(getattr(env, "max_hold", 1)))
        held = max(0, int(getattr(env, "t_held", 0)))
        remaining = max(1, max_hold - held)
        return torch.tensor(
            [float(remaining) / float(max_hold)],
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _controller_top_tail_rate_loss(
            policy_logits,
            *,
            min_rate,
            max_rate,
            margin=0.1,
    ):
        if isinstance(policy_logits, torch.Tensor):
            logits = policy_logits.view(-1)
        else:
            logits = torch.cat([logit.view(-1) for logit in policy_logits]).view(-1)
        if logits.numel() == 0:
            raise ValueError("policy_logits must contain at least one free-decision logit")
        min_rate = float(min_rate)
        max_rate = float(max_rate)
        if not 0.0 <= min_rate <= max_rate <= 1.0:
            raise ValueError("switch-rate bounds must satisfy 0 <= min_rate <= max_rate <= 1")
        margin = float(margin)
        if margin < 0.0:
            raise ValueError("switch-rate margin must be non-negative")

        sorted_logits = torch.sort(logits, descending=True).values
        count = int(sorted_logits.numel())
        k_min = min(count, int(math.ceil(min_rate * count))) if min_rate > 0.0 else 0
        k_max = min(count, int(math.floor(max_rate * count)))
        k_max = max(k_min, k_max)

        lower_loss = logits.sum() * 0.0
        if k_min > 0:
            lower_loss = torch.relu(
                logits.new_tensor(margin) - sorted_logits[:k_min]
            ).pow(2).mean()
        upper_loss = logits.sum() * 0.0
        if k_max < count:
            upper_loss = torch.relu(
                logits.new_tensor(margin) + sorted_logits[k_max:]
            ).pow(2).mean()
        hard_rate = (logits > 0.0).to(dtype=logits.dtype).mean()
        return lower_loss + upper_loss, hard_rate.detach()

    def _annotate_controller_guidance_segments(self, episode_segments):
        annotated_records = []
        labels_parts = []
        masks_parts = []
        risk_threshold = float(getattr(
            self.cfg,
            "controller_guidance_risk_threshold",
            0.05,
        ))
        advantage_threshold = float(getattr(
            self.cfg,
            "controller_guidance_advantage_threshold",
            0.05,
        ))
        risk_min_advantage_threshold = float(getattr(
            self.cfg,
            "controller_guidance_risk_min_advantage_threshold",
            0.0,
        ))
        for segment in episode_segments:
            valid_records = [
                record
                for record in segment
                if isinstance(record.get("target_mdd"), torch.Tensor)
                and isinstance(record.get("switch_advantage"), torch.Tensor)
            ]
            if not valid_records:
                continue
            risk = torch.cat([
                record["target_mdd"].detach().view(-1).cpu()
                for record in valid_records
            ])
            advantage = torch.cat([
                record["switch_advantage"].detach().view(-1).cpu()
                for record in valid_records
            ])
            guidance = build_economic_guidance_labels(
                risk,
                advantage,
                risk_threshold=risk_threshold,
                risk_min_advantage_threshold=risk_min_advantage_threshold,
                advantage_threshold=advantage_threshold,
            )
            if guidance.labels.numel() != len(valid_records):
                raise ValueError("Controller guidance currently expects one target per decision record.")
            annotated_records.extend(valid_records)
            labels_parts.append(guidance.labels)
            masks_parts.append(guidance.mask)

        if not annotated_records:
            return episode_segments

        labels = torch.cat(labels_parts)
        masks = torch.cat(masks_parts)
        # Preserve the naturally sparse economic label distribution.  Every
        # valid free decision contributes once to ordinary BCE.
        weights = masks
        for record, label, weight in zip(annotated_records, labels, weights):
            target = record["target_mdd"]
            record["sup_label"] = target.new_tensor([float(label)])
            record["sup_weight"] = target.new_tensor([float(weight)])
        return episode_segments

    def _controller_train_max_hold(self, fixed_cycle: int, rollout_len: int) -> int:
        override = int(getattr(self.cfg, "controller_train_max_hold", -1))
        if override == 0:
            fallback_len = getattr(self.cfg, "controller_rollout_len", getattr(self.cfg, "max_hold", 60))
            return max(1, int(rollout_len if rollout_len is not None else fallback_len) + 1)
        if override > 0:
            return override
        fallback = fixed_cycle if fixed_cycle is not None else getattr(self.cfg, "max_hold", 60)
        return max(1, int(getattr(self.cfg, "max_hold", fallback)))

    def _controller_eval_max_hold(self) -> int:
        override = int(getattr(self.cfg, "controller_eval_max_hold", -1))
        if override == 0:
            return 0
        if override > 0:
            return override
        return max(1, int(getattr(self.cfg, "max_hold", 60)))

    def _controller_should_record_train_decision(self, duration: int) -> bool:
        max_duration = int(getattr(self.cfg, "controller_train_record_max_duration", 0) or 0)
        if max_duration <= 0:
            return True
        return int(duration) < max_duration

    def _sample_controller_pg_action(self, stats, logits, *, start_idx: int, step_idx: int, epoch: int = None):
        dist = Categorical(logits=logits)
        if bool(getattr(self.cfg, "controller_deterministic_rollout_sampling", False)):
            exit_prob = stats["exit_prob"].detach()
            epoch_val = 0 if epoch is None else int(epoch)
            base_seed = int(getattr(self.cfg, "seed", 0) or 0)
            seed = (
                base_seed * 1_000_003
                + int(start_idx) * 10_007
                + int(step_idx) * 101
                + epoch_val * 1_009
            ) % (2 ** 63 - 1)
            generator = torch.Generator(device=exit_prob.device)
            generator.manual_seed(seed)
            draw = torch.rand(exit_prob.shape, device=exit_prob.device, generator=generator)
            act_mon = (draw < exit_prob).long()
        else:
            act_mon = dist.sample()
        return act_mon, dist.entropy().mean()

    def _run_controller_pg_window(self, env, start_idx: int, stop_idx: int, fixed_cycle: int, epoch: int = None):
        obs = env.reset_at(start_idx, stop_idx)
        history = [env.portfolio_value.item()]
        rollout_len = max(1, int(stop_idx) - int(start_idx))
        min_hold = int(getattr(self.cfg, "min_hold", 10))
        max_hold = self._controller_train_max_hold(fixed_cycle=fixed_cycle, rollout_len=rollout_len)
        max_segments = self._controller_max_allowed_switches(rollout_len, min_hold)
        disable_inner = bool(getattr(self.cfg, "controller_pg_disable_inner", False))

        turnover_sum = 0.0
        free_switch_count = 0
        segment_count = 0
        forced_switch_count = 0
        last_switch_step = 0
        step_idx = 0
        episode_segment_logps = []
        current_segment_logps = []
        entropies = []
        switch_advantage_values = []
        switch_advantage_exit_probs = []

        self.agent.net.outer.eval()
        self.agent.net.inner.eval()
        self.agent.net.mon.train()

        while True:
            duration = step_idx - last_switch_step
            forced = False
            free_decision = False

            if step_idx == 0:
                force_switch = 1
                forced = True
            elif int(getattr(self.cfg, "controller_hard_max_switches", 0) or 0) > 0 and (
                    segment_count >= int(getattr(self.cfg, "controller_hard_max_switches", 0) or 0)
            ):
                force_switch = 0
                forced = True
            elif duration >= max_hold:
                force_switch = 1
                forced = True
            elif bool(getattr(self.cfg, "controller_no_hold_constraints", False)):
                decision_mode = str(getattr(self.cfg, "controller_decision_mode", "daily"))
                if decision_mode == "stride":
                    stride = self._controller_decision_stride(is_train=True, epoch=epoch)
                    if int(duration) % max(1, stride) != 0:
                        force_switch = 0
                        forced = True
                    else:
                        force_switch = None
                        free_decision = True
                elif decision_mode == "fixed_window":
                    window = max(0, int(getattr(self.cfg, "controller_fixed_decision_window", 0)))
                    if duration < max(1, max_hold - window):
                        force_switch = 0
                        forced = True
                    else:
                        force_switch = None
                        free_decision = True
                else:
                    force_switch = None
                    free_decision = True
            elif duration < min_hold:
                force_switch = 0
                forced = True
            elif duration >= max_hold:
                force_switch = 1
                forced = True
            elif (
                    int(getattr(self.cfg, "controller_check_stride_days", 1)) > 1
                    and ((duration - min_hold) % int(getattr(self.cfg, "controller_check_stride_days", 1)) != 0)
            ):
                force_switch = 0
                forced = True
            elif not segment_budget_allows_switch(
                    day_offset=step_idx,
                    rollout_len=rollout_len,
                    current_segments=segment_count,
                    max_hold=max_hold,
                    max_segments=max_segments,
            ):
                force_switch = 0
                forced = True
            else:
                force_switch = None
                free_decision = True

            weights_drift = obs["weights_drift"]
            pending_record = None
            free_switch_index = 0
            precomputed_weights_exec = None
            actual_holdings_switch_advantage = None
            free_exit_probs = None
            if free_decision:
                with torch.no_grad():
                    act_out, _, _, _, _ = self.agent.net.outer.pi(
                        obs["outer_state"],
                        weights_drift,
                        deterministic=True,
                    )
                    actual_hold = weights_drift.detach()
                    if disable_inner:
                        inner_hold_exec = self._controller_exec_weights(
                            obs,
                            obs["base_drift"].detach(),
                            weights_drift,
                            disable_inner=True,
                        )
                        switch_exec = self._controller_exec_weights(
                            obs,
                            act_out.detach(),
                            weights_drift,
                            disable_inner=True,
                        )
                    else:
                        inner_hold_exec = self._deterministic_inner_exec(
                            obs,
                            obs["base_drift"].detach(),
                            weights_drift,
                        )
                        switch_exec = self._controller_exec_weights(
                            obs,
                            act_out.detach(),
                            weights_drift,
                            disable_inner=False,
                        )
                    remaining_horizon = self._controller_remaining_horizon(
                        env,
                        dtype=weights_drift.dtype,
                        device=weights_drift.device,
                    )
                    stats = self.agent.net.mon.decision_stats(
                        weights_drift, obs["port_state"],
                        switch_action=act_out,
                        asset_state=obs.get("outer_state"),
                        hold_exec_weights=actual_hold,
                        switch_exec_weights=switch_exec,
                        remaining_horizon=remaining_horizon,
                    )
                    zeros = torch.zeros_like(stats["policy_logit"])
                    logits = torch.stack([zeros, stats["policy_logit"]], dim=-1)
                    act_mon, entropy = self._sample_controller_pg_action(
                        stats,
                        logits,
                        start_idx=start_idx,
                        step_idx=step_idx,
                        epoch=epoch,
                    )
                    free_exit_probs = stats["exit_prob"].detach().view(-1).cpu().numpy().astype(float).tolist()
                is_switch = bool(act_mon.item() == 1)
                base_used = torch.where(act_mon.view(-1, 1).bool(), act_out.detach(), obs["base_drift"])
                record_train_decision = self._controller_should_record_train_decision(duration)
                if record_train_decision:
                    entropies.append(entropy.reshape(()))
                if is_switch:
                    free_switch_count += 1
                    free_switch_index = free_switch_count
                if record_train_decision and self._controller_uses_switch_advantage_targets():
                    precomputed_weights_exec = switch_exec if is_switch else inner_hold_exec
                    actual_holdings_switch_advantage = self._controller_actual_holdings_switch_advantage(
                        env,
                        obs,
                        switch_exec,
                    )
                if record_train_decision:
                    pending_record = (
                        obs,
                        act_out,
                        act_mon,
                        free_switch_index,
                        free_exit_probs,
                        actual_hold,
                        switch_exec,
                        remaining_horizon,
                    )
            else:
                with torch.no_grad():
                    out = self.agent.get_action(
                        obs,
                        mode="eval",
                        force_switch=force_switch,
                        force_inner_zero=disable_inner,
                        force_locked=True,
                    )
                    act_out = out["act_out"]
                    act_mon = out["act_mon"]
                    base_used = out["base_used"]
                    is_switch = bool(act_mon.item() == 1)

            with torch.no_grad():
                weights_exec = (
                    precomputed_weights_exec
                    if precomputed_weights_exec is not None
                    else self._controller_exec_weights(
                        obs,
                        base_used.detach(),
                        weights_drift,
                        disable_inner=disable_inner,
                    )
                )

            if is_switch:
                segment_count += 1
                last_switch_step = step_idx
                if forced and step_idx != 0:
                    forced_switch_count += 1

            turnover_sum += float(torch.sum(torch.abs(weights_exec - weights_drift)).item())
            next_obs, _, done, info = env.step(
                weights_exec.detach(),
                base_used.detach(),
                outer_action=act_out.detach(),
                is_switch=is_switch,
            )
            if free_decision and pending_record is not None:
                (
                    rec_obs,
                    rec_act_out,
                    rec_act_mon,
                    rec_free_switch_index,
                    rec_exit_probs,
                    rec_hold_exec,
                    rec_switch_exec,
                    rec_remaining_horizon,
                ) = pending_record
                switch_advantage = (
                    actual_holdings_switch_advantage
                    if actual_holdings_switch_advantage is not None
                    else info.get("controller_switch_advantage")
                )
                current_segment_logps.append(self._detach_controller_record(
                    rec_obs,
                    rec_act_out,
                    rec_act_mon,
                    target_return=info.get("controller_hold_return_target"),
                    target_mdd=info.get("controller_hold_mdd_target"),
                    sup_label=info.get("controller_switch_label"),
                    sup_weight=info.get("controller_sup_weight"),
                    free_switch_index=rec_free_switch_index,
                    switch_advantage=switch_advantage,
                    hold_exec_weights=rec_hold_exec,
                    switch_exec_weights=rec_switch_exec,
                    remaining_horizon=rec_remaining_horizon,
                ))
                if isinstance(switch_advantage, torch.Tensor):
                    switch_advantages = switch_advantage.detach().view(-1).cpu().numpy().astype(float).tolist()
                    switch_advantage_values.extend(switch_advantages)
                    if rec_exit_probs is not None:
                        exit_probs = list(rec_exit_probs)
                        if len(exit_probs) == 1 and len(switch_advantages) > 1:
                            exit_probs = exit_probs * len(switch_advantages)
                        switch_advantage_exit_probs.extend(exit_probs[:len(switch_advantages)])
            if is_switch and step_idx != 0:
                self._finish_pg_segment(current_segment_logps, episode_segment_logps)
            history.append(info["portfolio_value"])
            if done:
                self._finish_pg_segment(current_segment_logps, episode_segment_logps)
                break
            obs = next_obs
            step_idx += 1

        stats = self._counterfactual_stats(
            history,
            turnover_sum=turnover_sum,
            free_switch_count=free_switch_count,
            segment_count=segment_count,
        )
        if episode_segment_logps and not isinstance(episode_segment_logps[0], list):
            episode_logprob = torch.stack(episode_segment_logps).mean()
        else:
            episode_logprob = None
        entropy_mean = torch.stack(entropies).mean() if entropies else None
        self._annotate_controller_guidance_segments(episode_segment_logps)
        return {
            "stats": stats,
            "history": history,
            "episode_logprob": episode_logprob,
            "entropy": entropy_mean,
            "episode_segments": episode_segment_logps,
            "forced_switch_count": forced_switch_count,
            "switch_advantage_summary": self._controller_switch_advantage_summary(
                switch_advantage_values,
                exit_probs=switch_advantage_exit_probs,
            ),
        }

    def _controller_max_allowed_switches(self, rollout_len: int, min_hold: int) -> int:
        theoretical_max = max(1, int(rollout_len) // max(1, int(min_hold)))
        manual_max = int(getattr(self.cfg, "controller_max_switches", 0) or 0)
        if bool(getattr(self.cfg, "controller_no_hold_constraints", False)):
            return max(1, manual_max) if manual_max > 0 else theoretical_max
        if manual_max > 0:
            return max(1, min(manual_max, theoretical_max))
        return theoretical_max

    def _controller_train_starts(self, windows_per_epoch: int, rollout_len: int, epoch: int):
        raw_indices = self.env.idx_map["train"]
        train_start = int(raw_indices[0])
        train_end = int(raw_indices[-1])
        latest_start = max(train_start, train_end - int(rollout_len))
        stride = int(getattr(self.cfg, "controller_start_stride_days", self.cfg.max_hold))
        stride = max(1, stride)
        starts = list(range(train_start, latest_start + 1, stride))
        if not starts:
            starts = [train_start]
        out = []
        for i in range(max(1, int(windows_per_epoch))):
            out.append(starts[(epoch * max(1, int(windows_per_epoch)) + i) % len(starts)])
        return out

    def _controller_train_start_pool(self, rollout_len: int):
        raw_indices = self.env.idx_map["train"]
        starts = self.env._build_fixed_train_pool(
            raw_indices,
            total_days=self.env.total_days,
            episode_len=int(rollout_len),
            stride_days=int(getattr(self.cfg, "controller_start_stride_days", self.cfg.max_hold)),
            start_offsets=int(getattr(self.cfg, "controller_windows_per_epoch", self.cfg.train_episodes_per_epoch)),
        )
        if not starts:
            train_start = int(raw_indices[0])
            train_end = int(raw_indices[-1])
            latest_start = max(train_start, train_end - int(rollout_len))
            starts = [min(train_start, latest_start)]
        pool_limit = int(getattr(self.cfg, "controller_fixed_pool_limit", 0) or 0)
        if pool_limit > 0:
            if pool_limit < len(starts):
                sampled_indices = np.linspace(0, len(starts) - 1, pool_limit)
                starts = [starts[int(round(idx))] for idx in sampled_indices]
            else:
                starts = starts[:pool_limit]
        return [int(s) for s in starts]

    def _run_controller_pg_pair_worker(self, start_idx: int, stop_idx: int, fixed_cycle: int, epoch: int = None):
        controlled_env = copy.deepcopy(self.env)
        controlled_env.logger = None
        controlled_env.mode = "train"
        baseline_stats = self._get_controller_baseline_stats(start_idx, stop_idx, fixed_cycle)
        controlled = self._run_controller_pg_window(controlled_env, start_idx, stop_idx, fixed_cycle, epoch=epoch)
        return baseline_stats, controlled

    def _get_controller_baseline_stats(self, start_idx: int, stop_idx: int, fixed_cycle: int):
        use_cache = bool(getattr(self.cfg, "controller_cache_baseline_stats", True))
        disable_inner = bool(getattr(self.cfg, "controller_pg_disable_inner", False))
        key = (int(start_idx), int(stop_idx), int(fixed_cycle), bool(disable_inner))
        if use_cache:
            cache = getattr(self, "_controller_baseline_cache", None)
            if cache is None:
                cache = {}
                self._controller_baseline_cache = cache
            if key in cache:
                return cache[key]
        baseline_env = copy.deepcopy(self.env)
        baseline_env.logger = None
        baseline_env.mode = "train"
        baseline_stats, _ = self._run_fixed_hrl_window(
            baseline_env,
            start_idx,
            stop_idx,
            fixed_cycle,
            disable_inner=disable_inner,
        )
        if use_cache:
            self._controller_baseline_cache[key] = baseline_stats
        return baseline_stats

    def _run_controller_pg_pairs(self, windows, fixed_cycle: int, epoch: int = None):
        workers = max(1, int(getattr(self.cfg, "controller_episode_parallel_workers", 1)))
        workers = min(workers, max(1, len(windows)))
        if workers <= 1:
            return [
                self._run_controller_pg_pair_worker(start, stop, fixed_cycle, epoch=epoch)
                for start, stop in windows
            ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._run_controller_pg_pair_worker, start, stop, fixed_cycle, epoch)
                for start, stop in windows
            ]
            return [future.result() for future in futures]

    def _run_controller_pg_controlled_windows(self, windows, fixed_cycle: int, epoch: int = None):
        workers = max(1, int(getattr(self.cfg, "controller_episode_parallel_workers", 1)))
        workers = min(workers, max(1, len(windows)))

        def _worker(start_idx, stop_idx):
            controlled_env = copy.deepcopy(self.env)
            controlled_env.logger = None
            controlled_env.mode = "train"
            return self._run_controller_pg_window(controlled_env, start_idx, stop_idx, fixed_cycle, epoch=epoch)

        if workers <= 1:
            return [_worker(start, stop) for start, stop in windows]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, start, stop) for start, stop in windows]
            return [future.result() for future in futures]

    def _run_controller_aux_fixed_window(self, env, start_idx: int, stop_idx: int, fixed_cycle: int,
                                         epoch: int = None):
        obs = env.reset_at(start_idx, stop_idx)
        max_hold = int(getattr(self.cfg, "max_hold", fixed_cycle))
        last_switch_step = 0
        step_idx = 0
        free_switch_index = 0
        current_records = []
        episode_segments = []
        switch_advantage_values = []
        switch_advantage_exit_probs = []

        self.agent.net.outer.eval()
        self.agent.net.inner.eval()
        self.agent.net.mon.train()

        while True:
            duration = step_idx - last_switch_step
            force_switch = int(step_idx == 0 or duration >= max_hold)
            free_decision = False
            if not force_switch and bool(getattr(self.cfg, "controller_no_hold_constraints", False)):
                decision_mode = str(getattr(self.cfg, "controller_decision_mode", "daily"))
                if decision_mode == "stride":
                    stride = self._controller_decision_stride(is_train=True, epoch=epoch)
                    free_decision = int(duration) % max(1, stride) == 0
                elif decision_mode == "fixed_window":
                    window = max(0, int(getattr(self.cfg, "controller_fixed_decision_window", 0)))
                    free_decision = duration >= max(1, max_hold - window)
                else:
                    free_decision = True

            weights_drift = obs["weights_drift"]
            with torch.no_grad():
                act_out, _, _, _, _ = self.agent.net.outer.pi(
                    obs["outer_state"],
                    weights_drift,
                    deterministic=True,
                )
                disable_inner = bool(getattr(self.cfg, "controller_pg_disable_inner", False))
                hold_exec = self._controller_exec_weights(
                    obs,
                    obs["base_drift"].detach(),
                    weights_drift,
                    disable_inner=disable_inner,
                )
                switch_exec = self._controller_exec_weights(
                    obs,
                    act_out.detach(),
                    weights_drift,
                    disable_inner=disable_inner,
                )
                base_used = act_out.detach() if force_switch else obs["base_drift"].detach()
                weights_exec = switch_exec if force_switch else hold_exec
                action = torch.full(
                    (weights_drift.shape[0],),
                    int(force_switch),
                    dtype=torch.long,
                    device=self.device,
                )
                switch_advantage = None
                exit_probs = None
                if free_decision:
                    if self._controller_uses_switch_advantage_targets():
                        switch_advantage = self._controller_actual_holdings_switch_advantage(
                            env,
                            obs,
                            switch_exec,
                        )
                    remaining_horizon = self._controller_remaining_horizon(
                        env,
                        dtype=weights_drift.dtype,
                        device=weights_drift.device,
                    )
                    stats = self.agent.net.mon.decision_stats(
                        weights_drift, obs["port_state"],
                        switch_action=act_out,
                        asset_state=obs.get("outer_state"),
                        hold_exec_weights=weights_drift.detach(),
                        switch_exec_weights=switch_exec,
                        remaining_horizon=remaining_horizon,
                    )
                    exit_probs = stats["exit_prob"].detach().view(-1).cpu().numpy().astype(float).tolist()

            next_obs, _, done, info = env.step(
                weights_exec.detach(),
                base_used.detach(),
                outer_action=act_out.detach(),
                is_switch=bool(force_switch),
            )
            if force_switch:
                if current_records:
                    episode_segments.append(current_records)
                    current_records = []
                last_switch_step = step_idx
            elif free_decision:
                free_switch_index += 1
                switch_advantage = (
                    switch_advantage
                    if switch_advantage is not None
                    else info.get("controller_switch_advantage")
                )
                current_records.append(self._detach_controller_record(
                    obs,
                    act_out,
                    action,
                    target_return=info.get("controller_hold_return_target"),
                    target_mdd=info.get("controller_hold_mdd_target"),
                    sup_label=info.get("controller_switch_label"),
                    sup_weight=info.get("controller_sup_weight"),
                    free_switch_index=free_switch_index,
                    switch_advantage=switch_advantage,
                    hold_exec_weights=weights_drift.detach(),
                    switch_exec_weights=switch_exec,
                    remaining_horizon=remaining_horizon,
                ))
                if isinstance(switch_advantage, torch.Tensor):
                    switch_advantages = switch_advantage.detach().view(-1).cpu().numpy().astype(float).tolist()
                    switch_advantage_values.extend(switch_advantages)
                    if exit_probs is not None:
                        probs = list(exit_probs)
                        if len(probs) == 1 and len(switch_advantages) > 1:
                            probs = probs * len(switch_advantages)
                        switch_advantage_exit_probs.extend(probs[:len(switch_advantages)])

            if done:
                break
            obs = next_obs
            step_idx += 1

        if current_records:
            episode_segments.append(current_records)
        self._annotate_controller_guidance_segments(episode_segments)
        return {
            "episode_segments": episode_segments,
            "switch_advantage_summary": self._controller_switch_advantage_summary(
                switch_advantage_values,
                exit_probs=switch_advantage_exit_probs,
            ),
        }

    def _run_controller_aux_fixed_windows(self, windows, fixed_cycle: int, epoch: int = None):
        workers = max(1, int(getattr(self.cfg, "controller_episode_parallel_workers", 1)))
        workers = min(workers, max(1, len(windows)))

        def _worker(start_idx, stop_idx):
            fixed_env = copy.deepcopy(self.env)
            fixed_env.logger = None
            fixed_env.mode = "train"
            return self._run_controller_aux_fixed_window(fixed_env, start_idx, stop_idx, fixed_cycle, epoch=epoch)

        if workers <= 1:
            return [_worker(start, stop) for start, stop in windows]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, start, stop) for start, stop in windows]
            return [future.result() for future in futures]

    def _run_controller_aux_pretrain_windows(self, windows, fixed_cycle: int, epoch: int = None):
        if bool(getattr(self.cfg, "controller_aux_pretrain_offpolicy", False)):
            return self._run_controller_aux_fixed_windows(windows, fixed_cycle, epoch=epoch)
        return self._run_controller_pg_controlled_windows(windows, fixed_cycle, epoch=epoch)

    def analyze_controller_guidance_labels(
            self,
            *,
            fixed_cycle: int,
            rollout_len: int = 300,
            window_count: int = 12,
            topk: int = 20,
    ):
        rollout_len = max(2, int(rollout_len))
        window_count = max(1, int(window_count))
        topk = max(0, int(topk))
        starts = self._controller_train_start_pool(rollout_len)[:window_count]
        train_end = int(self.env.idx_map["train"][-1])
        windows = [
            (int(start), min(int(start) + rollout_len, train_end))
            for start in starts
            if min(int(start) + rollout_len, train_end) - int(start) >= max(2, int(fixed_cycle))
        ]
        if not windows:
            raise RuntimeError("No valid train windows are available for Controller guidance probe.")

        self.logger.info(
            "Controller guidance probe: split=train rollout_len=%s windows=%s topk=%s disable_inner=%s",
            rollout_len,
            len(windows),
            topk,
            bool(getattr(self.cfg, "controller_pg_disable_inner", False)),
        )
        controlled_results = self._run_controller_aux_fixed_windows(
            windows,
            fixed_cycle,
            epoch=0,
        )
        guidance_windows = []
        for window_id, ((start, _), controlled) in enumerate(zip(windows, controlled_results)):
            records = [
                record
                for segment in controlled.get("episode_segments", [])
                for record in segment
            ]
            risk_values = []
            advantage_values = []
            for record in records:
                risk = record.get("target_mdd")
                advantage = record.get("switch_advantage")
                if not isinstance(risk, torch.Tensor) or not isinstance(advantage, torch.Tensor):
                    continue
                risk_values.extend(risk.detach().view(-1).cpu().tolist())
                advantage_values.extend(advantage.detach().view(-1).cpu().tolist())
            if risk_values and len(risk_values) == len(advantage_values):
                guidance_windows.append({
                    "window_id": window_id,
                    "start_index": int(start),
                    "risk": torch.tensor(risk_values, dtype=torch.float32),
                    "advantage": torch.tensor(advantage_values, dtype=torch.float32),
                })
        if not guidance_windows:
            raise RuntimeError("Controller guidance probe collected no valid risk/advantage pairs.")

        detail_rows, summary = analyze_guidance_windows(
            guidance_windows,
            topk=topk,
        )
        csv_path = os.path.join(self.run_dir, "controller_guidance_probe.csv")
        report_path = os.path.join(self.run_dir, "controller_guidance_probe.md")
        pd.DataFrame(detail_rows).to_csv(csv_path, index=False)
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(render_guidance_report(
                summary,
                topk=topk,
                rollout_len=rollout_len,
            ))
        self.logger.info(
            "Controller guidance probe completed: decisions=%s switch_labels=%s rate=%.4f "
            "adjacent_rate=%.4f report=%s",
            summary["decision_count"],
            summary["switch_label_count"],
            summary["switch_label_rate"],
            summary["selected_adjacent_gap_rate"],
            report_path,
        )
        return {
            "guidance_probe_only": True,
            "report": report_path,
            "csv": csv_path,
            "summary": summary,
        }

    def _controller_switch_supervision_enabled(self) -> bool:
        return (
                bool(getattr(self.cfg, "controller_use_switch_supervision", False))
                and float(getattr(self.cfg, "controller_sup_coef", getattr(self.cfg, "monitor_sup_coef", 0.0))) > 0.0
        )

    def _should_validate_controller_pg(self, *, epoch: int, epochs: int, val_interval: int) -> bool:
        if bool(getattr(self.cfg, "controller_skip_val", False)):
            return False
        return int(epoch) % max(1, int(val_interval)) == 0 or int(epoch) == int(epochs)

    def _controller_episode_terms(self, episode_segments, *, return_policy_logits=False):
        entropies = []
        aux_return_losses = []
        aux_mdd_losses = []
        aux_switch_adv_losses = []
        sup_losses = []
        local_adv_losses = []
        local_adv_bce_weighted_losses = []
        local_adv_bce_weights = []
        local_adv_bce_pos_losses = []
        local_adv_bce_pos_weights = []
        local_adv_bce_neg_losses = []
        local_adv_bce_neg_weights = []
        episode_values = []
        expected_switch_probs = []
        policy_logits = []
        overflow_switch_logps = []
        overflow_switch_orders = []
        aux_return_coef = float(getattr(self.cfg, "controller_aux_return_coef", 0.0))
        aux_mdd_coef = float(getattr(self.cfg, "controller_aux_mdd_coef", 0.0))
        aux_switch_adv_coef = float(getattr(self.cfg, "controller_aux_switch_adv_coef", 0.0))
        local_adv_coef = float(getattr(self.cfg, "controller_local_adv_coef", 0.0))
        local_adv_scale = max(float(getattr(self.cfg, "controller_local_adv_scale", 0.05)), 1e-8)
        local_adv_clip = max(float(getattr(self.cfg, "controller_local_adv_clip", 10.0)), 0.0)
        local_adv_margin = float(getattr(self.cfg, "controller_local_adv_margin", 0.0) or 0.0)
        local_adv_loss_type = str(getattr(self.cfg, "controller_local_adv_loss_type", "linear")).lower()
        local_adv_balance_classes = bool(getattr(self.cfg, "controller_local_adv_balance_classes", False))
        expected_switch_penalty_coef = float(getattr(self.cfg, "controller_expected_switch_penalty_coef", 0.0))
        sup_coef = (
            float(getattr(self.cfg, "controller_sup_coef", getattr(self.cfg, "monitor_sup_coef", 0.0)))
            if self._controller_switch_supervision_enabled()
            else 0.0
        )
        logprob_reduction = str(getattr(self.cfg, "controller_pg_logprob_reduction", "mean")).lower()
        max_switches = int(getattr(self.cfg, "controller_max_switches", 0) or 0)
        max_free_switches = max(0, max_switches - 1) if max_switches > 0 else 0
        episode_record_logps = []
        for segment in episode_segments:
            for record in segment:
                asset_state = record.get("asset_state")
                if isinstance(asset_state, torch.Tensor):
                    asset_state = asset_state.to(self.device)
                hold_exec_weights = record.get("hold_exec_weights")
                if isinstance(hold_exec_weights, torch.Tensor):
                    hold_exec_weights = hold_exec_weights.to(self.device)
                switch_exec_weights = record.get("switch_exec_weights")
                if isinstance(switch_exec_weights, torch.Tensor):
                    switch_exec_weights = switch_exec_weights.to(self.device)
                remaining_horizon = record.get("remaining_horizon")
                if isinstance(remaining_horizon, torch.Tensor):
                    remaining_horizon = remaining_horizon.to(self.device)
                stats = self.agent.net.mon.decision_stats(
                    record["weights_drift"].to(self.device),
                    record["port_state"].to(self.device),
                    switch_action=record["switch_action"].to(self.device),
                    asset_state=asset_state,
                    hold_exec_weights=hold_exec_weights,
                    switch_exec_weights=switch_exec_weights,
                    remaining_horizon=remaining_horizon,
                )
                zeros = torch.zeros_like(stats["policy_logit"])
                logits = torch.stack([zeros, stats["policy_logit"]], dim=-1)
                dist = Categorical(logits=logits)
                action = record["action"].to(self.device).long().view(-1)
                log_prob = dist.log_prob(action).view(-1).mean()
                exit_prob = stats["exit_prob"].view(-1).mean()
                episode_record_logps.append(log_prob)
                expected_switch_probs.append(exit_prob)
                policy_logits.append(stats["policy_logit"].view(-1).mean())
                entropies.append(dist.entropy().mean())
                if stats.get("value") is not None:
                    episode_values.append(stats["value"].view(-1).mean())
                free_switch_index = int(record.get("free_switch_index", 0) or 0)
                if (
                        max_free_switches > 0
                        and int(action.item()) == 1
                        and free_switch_index > max_free_switches
                ):
                    overflow_switch_logps.append(log_prob)
                    overflow_switch_orders.append(float(free_switch_index - max_free_switches))
                if aux_return_coef > 0.0 and record.get("target_return") is not None:
                    target_return = record["target_return"].to(
                        device=self.device,
                        dtype=stats["hold_return_pred"].dtype,
                    )
                    target_return = target_return * float(getattr(
                        self.cfg,
                        "controller_aux_return_target_scale",
                        1.0,
                    ))
                    aux_return_losses.append(
                        F.smooth_l1_loss(stats["hold_return_pred"].view(-1), target_return.view(-1))
                    )
                if aux_mdd_coef > 0.0 and record.get("target_mdd") is not None:
                    target_mdd = record["target_mdd"].to(
                        device=self.device,
                        dtype=stats["hold_risk_pred"].dtype,
                    )
                    target_mdd = target_mdd * float(getattr(
                        self.cfg,
                        "controller_aux_mdd_target_scale",
                        1.0,
                    ))
                    aux_mdd_losses.append(
                        F.mse_loss(stats["hold_risk_pred"].view(-1), target_mdd.view(-1))
                    )
                if (
                        aux_switch_adv_coef > 0.0
                        and record.get("switch_advantage") is not None
                        and stats.get("switch_advantage_pred") is not None
                ):
                    target_switch_adv = record["switch_advantage"].to(
                        device=self.device,
                        dtype=stats["switch_advantage_pred"].dtype,
                    )
                    target_switch_adv = target_switch_adv * float(getattr(
                        self.cfg,
                        "controller_aux_switch_adv_target_scale",
                        1.0,
                    ))
                    pred_switch_adv = stats["switch_advantage_pred"].view(-1)
                    target_switch_adv = target_switch_adv.view(-1)
                    valid_switch_adv = torch.isfinite(target_switch_adv) & torch.isfinite(pred_switch_adv)
                    if torch.count_nonzero(valid_switch_adv).item() > 0:
                        pred_switch_adv = pred_switch_adv[valid_switch_adv]
                        target_switch_adv = target_switch_adv[valid_switch_adv]
                        aux_switch_adv_losses.append(
                            F.mse_loss(pred_switch_adv, target_switch_adv)
                        )
                if sup_coef > 0.0 and record.get("sup_label") is not None and record.get("sup_weight") is not None:
                    label = record["sup_label"].to(device=self.device, dtype=stats["policy_logit"].dtype).view(-1)
                    weight = record["sup_weight"].to(device=self.device, dtype=stats["policy_logit"].dtype).view(-1)
                    valid_sup = torch.isfinite(label) & torch.isfinite(weight) & (weight > 0)
                    if torch.count_nonzero(valid_sup).item() > 0:
                        raw_sup = F.binary_cross_entropy_with_logits(
                            stats["policy_logit"].view(-1)[valid_sup],
                            label[valid_sup],
                            reduction="none",
                        )
                        sup_losses.append((raw_sup * weight[valid_sup]).mean())
                if local_adv_coef > 0.0 and record.get("switch_advantage") is not None:
                    switch_advantage = record["switch_advantage"].to(
                        device=self.device,
                        dtype=stats["exit_prob"].dtype,
                    ).view(-1)
                    valid_adv = torch.isfinite(switch_advantage)
                    if torch.count_nonzero(valid_adv).item() > 0:
                        effective_adv = switch_advantage[valid_adv] - local_adv_margin
                        scaled_adv = (effective_adv / local_adv_scale).clamp(
                            -local_adv_clip,
                            local_adv_clip,
                        )
                        if local_adv_loss_type in {"weighted_bce", "bce"}:
                            labels = (scaled_adv > 0).to(dtype=stats["policy_logit"].dtype)
                            weights = torch.abs(scaled_adv).detach()
                            raw_bce = F.binary_cross_entropy_with_logits(
                                stats["policy_logit"].view(-1)[valid_adv],
                                labels,
                                reduction="none",
                            )
                            if local_adv_balance_classes:
                                pos_mask = labels > 0.5
                                neg_mask = ~pos_mask
                                if torch.count_nonzero(pos_mask).item() > 0:
                                    local_adv_bce_pos_losses.append((raw_bce[pos_mask] * weights[pos_mask]).sum())
                                    local_adv_bce_pos_weights.append(weights[pos_mask].sum())
                                if torch.count_nonzero(neg_mask).item() > 0:
                                    local_adv_bce_neg_losses.append((raw_bce[neg_mask] * weights[neg_mask]).sum())
                                    local_adv_bce_neg_weights.append(weights[neg_mask].sum())
                            else:
                                local_adv_bce_weighted_losses.append((raw_bce * weights).sum())
                                local_adv_bce_weights.append(weights.sum())
                        else:
                            local_adv_losses.append(
                                -(scaled_adv.detach() * stats["exit_prob"].view(-1)[valid_adv]).mean()
                            )
        if not episode_record_logps:
            empty_terms = (None, None, None, None, None, None, None, None, None, None)
            if return_policy_logits:
                return empty_terms, None
            return empty_terms
        if logprob_reduction == "sum":
            episode_logprob = torch.stack(episode_record_logps).sum()
        else:
            episode_logprob = torch.stack(episode_record_logps).mean()
        entropy = torch.stack(entropies).mean() if entropies else episode_logprob.new_tensor(0.0)
        episode_value = torch.stack(episode_values).mean() if episode_values else None
        aux_return_loss = torch.stack(aux_return_losses).mean() if aux_return_losses else None
        aux_mdd_loss = torch.stack(aux_mdd_losses).mean() if aux_mdd_losses else None
        aux_switch_adv_loss = torch.stack(aux_switch_adv_losses).mean() if aux_switch_adv_losses else None
        sup_loss = torch.stack(sup_losses).mean() if sup_losses else None
        if local_adv_loss_type in {"weighted_bce", "bce"}:
            if local_adv_balance_classes and (local_adv_bce_pos_losses or local_adv_bce_neg_losses):
                class_losses = []
                if local_adv_bce_pos_losses:
                    class_losses.append(
                        torch.stack(local_adv_bce_pos_losses).sum()
                        / torch.stack(local_adv_bce_pos_weights).sum().clamp_min(1e-8)
                    )
                if local_adv_bce_neg_losses:
                    class_losses.append(
                        torch.stack(local_adv_bce_neg_losses).sum()
                        / torch.stack(local_adv_bce_neg_weights).sum().clamp_min(1e-8)
                    )
                local_adv_loss = torch.stack(class_losses).mean()
            elif local_adv_bce_weighted_losses:
                local_adv_loss = (
                    torch.stack(local_adv_bce_weighted_losses).sum()
                    / torch.stack(local_adv_bce_weights).sum().clamp_min(1e-8)
                )
            else:
                local_adv_loss = None
        else:
            local_adv_loss = torch.stack(local_adv_losses).mean() if local_adv_losses else None
        expected_switch_loss = None
        if expected_switch_penalty_coef > 0.0 and max_free_switches > 0 and expected_switch_probs:
            expected_switch_count = torch.stack(expected_switch_probs).sum()
            expected_overflow = torch.relu(expected_switch_count - episode_logprob.new_tensor(float(max_free_switches)))
            expected_switch_loss = expected_switch_penalty_coef * expected_overflow.pow(2)
        overflow_action_penalty_coef = float(getattr(self.cfg, "controller_overflow_action_penalty_coef", 0.0))
        if overflow_switch_logps and overflow_action_penalty_coef > 0.0:
            overflow_action_loss = overflow_switch_policy_loss(
                torch.stack(overflow_switch_logps),
                torch.tensor(overflow_switch_orders, device=self.device, dtype=episode_logprob.dtype),
                penalty_coef=overflow_action_penalty_coef,
            )
        else:
            overflow_action_loss = episode_logprob.new_tensor(0.0)
        terms = (
            episode_logprob,
            entropy,
            episode_value,
            aux_return_loss,
            aux_mdd_loss,
            aux_switch_adv_loss,
            overflow_action_loss,
            sup_loss,
            local_adv_loss,
            expected_switch_loss,
        )
        if return_policy_logits:
            return terms, torch.stack(policy_logits)
        return terms

    def _update_controller_pg_batch(self, episode_logprobs, rewards, entropies,
                                    aux_return_losses=None, aux_mdd_losses=None,
                                    aux_switch_adv_losses=None, overflow_action_losses=None,
                                    sup_losses=None, local_adv_losses=None, expected_switch_losses=None,
                                    episode_values=None):
        if len(episode_logprobs) == 0:
            return {}
        self.agent.opt_mon.zero_grad(set_to_none=True)
        logprob_tensor = torch.stack(episode_logprobs)
        reward_tensor = torch.tensor(rewards, dtype=logprob_tensor.dtype, device=self.device)
        entropy_tensor = torch.stack(entropies) if entropies else torch.zeros_like(logprob_tensor)
        value_coef = float(getattr(self.cfg, "controller_value_coef", 0.0))
        value_tensor = None
        if value_coef > 0.0 and episode_values:
            value_tensor = torch.stack(episode_values)
        loss, diagnostics = controller_pg_loss(
            logprob_tensor,
            reward_tensor,
            entropy_tensor,
            entropy_coef=float(getattr(self.cfg, "controller_entropy_coef", 0.01)),
            values=value_tensor,
            value_coef=value_coef,
            normalize_value_advantage=bool(getattr(self.cfg, "controller_value_normalize_advantage", True)),
        )
        aux_return_coef = float(getattr(self.cfg, "controller_aux_return_coef", 0.0))
        aux_mdd_coef = float(getattr(self.cfg, "controller_aux_mdd_coef", 0.0))
        aux_switch_adv_coef = float(getattr(self.cfg, "controller_aux_switch_adv_coef", 0.0))
        aux_return_loss = logprob_tensor.new_tensor(0.0)
        aux_mdd_loss = logprob_tensor.new_tensor(0.0)
        aux_switch_adv_loss = logprob_tensor.new_tensor(0.0)
        overflow_action_loss = logprob_tensor.new_tensor(0.0)
        sup_loss = logprob_tensor.new_tensor(0.0)
        local_adv_loss = logprob_tensor.new_tensor(0.0)
        expected_switch_loss = logprob_tensor.new_tensor(0.0)
        if aux_return_coef > 0.0 and aux_return_losses:
            aux_return_loss = torch.stack(aux_return_losses).mean()
            loss = loss + aux_return_coef * aux_return_loss
        if aux_mdd_coef > 0.0 and aux_mdd_losses:
            aux_mdd_loss = torch.stack(aux_mdd_losses).mean()
            loss = loss + aux_mdd_coef * aux_mdd_loss
        if aux_switch_adv_coef > 0.0 and aux_switch_adv_losses:
            aux_switch_adv_loss = torch.stack(aux_switch_adv_losses).mean()
            loss = loss + aux_switch_adv_coef * aux_switch_adv_loss
        if overflow_action_losses:
            overflow_action_loss = torch.stack(overflow_action_losses).mean()
            loss = loss + overflow_action_loss
        sup_coef = (
            float(getattr(self.cfg, "controller_sup_coef", getattr(self.cfg, "monitor_sup_coef", 0.0)))
            if self._controller_switch_supervision_enabled()
            else 0.0
        )
        if sup_coef > 0.0 and sup_losses:
            sup_loss = torch.stack(sup_losses).mean()
            loss = loss + sup_coef * sup_loss
        local_adv_coef = float(getattr(self.cfg, "controller_local_adv_coef", 0.0))
        if local_adv_coef > 0.0 and local_adv_losses:
            local_adv_loss = torch.stack(local_adv_losses).mean()
            loss = loss + local_adv_coef * local_adv_loss
        if expected_switch_losses:
            expected_switch_loss = torch.stack(expected_switch_losses).mean()
            loss = loss + expected_switch_loss
        diagnostics["loss"] = float(loss.detach().cpu().item())
        diagnostics["loss_abs"] = float(loss.detach().abs().cpu().item())
        diagnostics["aux_return_loss"] = float(aux_return_loss.detach().cpu().item())
        diagnostics["aux_mdd_loss"] = float(aux_mdd_loss.detach().cpu().item())
        diagnostics["aux_switch_adv_loss"] = float(aux_switch_adv_loss.detach().cpu().item())
        diagnostics["aux_return_weighted_loss"] = float((aux_return_coef * aux_return_loss).detach().cpu().item())
        diagnostics["aux_mdd_weighted_loss"] = float((aux_mdd_coef * aux_mdd_loss).detach().cpu().item())
        diagnostics["aux_switch_adv_weighted_loss"] = float((aux_switch_adv_coef * aux_switch_adv_loss).detach().cpu().item())
        diagnostics["overflow_action_loss"] = float(overflow_action_loss.detach().cpu().item())
        diagnostics["sup_loss"] = float(sup_loss.detach().cpu().item())
        diagnostics["sup_weighted_loss"] = float((sup_coef * sup_loss).detach().cpu().item())
        diagnostics["local_adv_loss"] = float(local_adv_loss.detach().cpu().item())
        diagnostics["local_adv_weighted_loss"] = float((local_adv_coef * local_adv_loss).detach().cpu().item())
        diagnostics["expected_switch_loss"] = float(expected_switch_loss.detach().cpu().item())
        loss.backward()
        if self.agent.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.agent.net.mon.parameters()), self.agent.max_grad_norm)
        self.agent.opt_mon.step()
        return diagnostics

    def _controller_pg_advantage_tensor(self, rewards, detached_values=None, *, dtype=torch.float32):
        reward_tensor = torch.tensor(rewards, dtype=dtype, device=self.device).view(-1)
        value_coef = float(getattr(self.cfg, "controller_value_coef", 0.0))
        values = None
        if value_coef > 0.0 and detached_values:
            valid_values = [v for v in detached_values if v is not None]
            if len(valid_values) == len(rewards):
                values = torch.stack([
                    v.to(device=self.device, dtype=dtype).view(())
                    for v in valid_values
                ]).view(-1)

        if values is not None:
            raw_advantage = reward_tensor - values
            advantage = raw_advantage
        else:
            advantage = reward_tensor
        return reward_tensor, advantage

    def _update_controller_pg_segments_batch(self, episode_segments_batch, rewards):
        if not episode_segments_batch:
            return {}
        if len(episode_segments_batch) != len(rewards):
            raise ValueError("episode_segments_batch and rewards must have the same length.")

        value_coef = float(getattr(self.cfg, "controller_value_coef", 0.0))
        detached_values = []
        if value_coef > 0.0:
            with torch.no_grad():
                for episode_segments in episode_segments_batch:
                    terms = self._controller_episode_terms(episode_segments)
                    episode_value = terms[2] if terms is not None else None
                    if episode_value is None:
                        detached_values.append(None)
                    else:
                        detached_values.append(episode_value.detach().view(()).to(self.device))

        reward_tensor, advantage = self._controller_pg_advantage_tensor(
            rewards,
            detached_values=detached_values,
        )
        batch_size = max(1, len(episode_segments_batch))
        entropy_coef = float(getattr(self.cfg, "controller_entropy_coef", 0.01))
        aux_return_coef = float(getattr(self.cfg, "controller_aux_return_coef", 0.0))
        aux_mdd_coef = float(getattr(self.cfg, "controller_aux_mdd_coef", 0.0))
        aux_switch_adv_coef = float(getattr(self.cfg, "controller_aux_switch_adv_coef", 0.0))
        sup_coef = (
            float(getattr(self.cfg, "controller_sup_coef", getattr(self.cfg, "monitor_sup_coef", 0.0)))
            if self._controller_switch_supervision_enabled()
            else 0.0
        )
        local_adv_coef = float(getattr(self.cfg, "controller_local_adv_coef", 0.0))
        switch_rate_coef = float(getattr(
            self.cfg,
            "controller_switch_rate_penalty_coef",
            0.0,
        ))

        self.agent.opt_mon.zero_grad(set_to_none=True)
        diagnostics = {
            "loss": 0.0,
            "loss_abs": 0.0,
            "policy_loss": 0.0,
            "policy_abs_loss": 0.0,
            "entropy": 0.0,
            "entropy_loss": 0.0,
            "reward_mean": float(reward_tensor.mean().detach().cpu().item()),
            "reward_std": float(reward_tensor.std(unbiased=False).detach().cpu().item()),
            "reward_min": float(reward_tensor.min().detach().cpu().item()),
            "reward_max": float(reward_tensor.max().detach().cpu().item()),
            "reward_abs_mean": float(reward_tensor.abs().mean().detach().cpu().item()),
            "value_loss": 0.0,
            "value_weighted_loss": 0.0,
            "aux_return_loss": 0.0,
            "aux_mdd_loss": 0.0,
            "aux_switch_adv_loss": 0.0,
            "aux_return_weighted_loss": 0.0,
            "aux_mdd_weighted_loss": 0.0,
            "aux_switch_adv_weighted_loss": 0.0,
            "overflow_action_loss": 0.0,
            "sup_loss": 0.0,
            "sup_weighted_loss": 0.0,
            "local_adv_loss": 0.0,
            "local_adv_weighted_loss": 0.0,
            "expected_switch_loss": 0.0,
            "switch_rate_loss": 0.0,
            "switch_rate_weighted_loss": 0.0,
            "hard_switch_rate": 0.0,
            "episode_count": int(len(episode_segments_batch)),
        }
        valid_episodes = 0
        batch_loss = None
        batch_policy_logits = []

        for episode_idx, episode_segments in enumerate(episode_segments_batch):
            terms, episode_policy_logits = self._controller_episode_terms(
                episode_segments,
                return_policy_logits=True,
            )
            if terms is None:
                continue
            (
                episode_logprob,
                entropy,
                episode_value,
                aux_return_loss,
                aux_mdd_loss,
                aux_switch_adv_loss,
                overflow_action_loss,
                sup_loss,
                local_adv_loss,
                expected_switch_loss,
            ) = terms
            if episode_logprob is None:
                continue
            valid_episodes += 1
            if episode_policy_logits is not None:
                batch_policy_logits.append(episode_policy_logits)
            denom = float(batch_size)
            episode_reward = reward_tensor[episode_idx].to(
                device=episode_logprob.device,
                dtype=episode_logprob.dtype,
            )
            episode_advantage = advantage[episode_idx].to(
                device=episode_logprob.device,
                dtype=episode_logprob.dtype,
            ).detach()
            episode_policy_loss = -(episode_advantage * episode_logprob) / denom
            episode_loss = episode_policy_loss
            diagnostics["policy_loss"] += float(episode_policy_loss.detach().cpu().item())
            diagnostics["policy_abs_loss"] += float(episode_policy_loss.detach().abs().cpu().item())

            entropy_term = (
                entropy.to(device=episode_logprob.device, dtype=episode_logprob.dtype)
                if entropy is not None
                else episode_logprob.new_tensor(0.0)
            )
            entropy_loss = -entropy_coef * entropy_term / denom
            episode_loss = episode_loss + entropy_loss
            diagnostics["entropy"] += float((entropy_term / denom).detach().cpu().item())
            diagnostics["entropy_loss"] += float(entropy_loss.detach().cpu().item())

            if value_coef > 0.0 and episode_value is not None:
                value_loss = (episode_value.view(()) - episode_reward).pow(2)
                weighted = value_coef * value_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["value_loss"] += float((value_loss / denom).detach().cpu().item())
                diagnostics["value_weighted_loss"] += float(weighted.detach().cpu().item())
            if aux_return_coef > 0.0 and aux_return_loss is not None:
                weighted = aux_return_coef * aux_return_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["aux_return_loss"] += float((aux_return_loss / denom).detach().cpu().item())
                diagnostics["aux_return_weighted_loss"] += float(weighted.detach().cpu().item())
            if aux_mdd_coef > 0.0 and aux_mdd_loss is not None:
                weighted = aux_mdd_coef * aux_mdd_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["aux_mdd_loss"] += float((aux_mdd_loss / denom).detach().cpu().item())
                diagnostics["aux_mdd_weighted_loss"] += float(weighted.detach().cpu().item())
            if aux_switch_adv_coef > 0.0 and aux_switch_adv_loss is not None:
                weighted = aux_switch_adv_coef * aux_switch_adv_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["aux_switch_adv_loss"] += float((aux_switch_adv_loss / denom).detach().cpu().item())
                diagnostics["aux_switch_adv_weighted_loss"] += float(weighted.detach().cpu().item())
            if overflow_action_loss is not None:
                weighted = overflow_action_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["overflow_action_loss"] += float(weighted.detach().cpu().item())
            if sup_coef > 0.0 and sup_loss is not None:
                weighted = sup_coef * sup_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["sup_loss"] += float((sup_loss / denom).detach().cpu().item())
                diagnostics["sup_weighted_loss"] += float(weighted.detach().cpu().item())
            if local_adv_coef > 0.0 and local_adv_loss is not None:
                weighted = local_adv_coef * local_adv_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["local_adv_loss"] += float((local_adv_loss / denom).detach().cpu().item())
                diagnostics["local_adv_weighted_loss"] += float(weighted.detach().cpu().item())
            if expected_switch_loss is not None:
                weighted = expected_switch_loss / denom
                episode_loss = episode_loss + weighted
                diagnostics["expected_switch_loss"] += float(weighted.detach().cpu().item())

            batch_loss = episode_loss if batch_loss is None else batch_loss + episode_loss

        if valid_episodes == 0:
            self.agent.opt_mon.zero_grad(set_to_none=True)
            return {}
        if switch_rate_coef > 0.0 and batch_policy_logits:
            switch_rate_loss, hard_switch_rate = self._controller_top_tail_rate_loss(
                torch.cat(batch_policy_logits),
                min_rate=float(getattr(self.cfg, "controller_switch_rate_min", 0.05)),
                max_rate=float(getattr(self.cfg, "controller_switch_rate_max", 0.15)),
                margin=float(getattr(self.cfg, "controller_switch_rate_margin", 0.1)),
            )
            switch_rate_weighted_loss = switch_rate_coef * switch_rate_loss
            batch_loss = batch_loss + switch_rate_weighted_loss
            diagnostics["switch_rate_loss"] = float(switch_rate_loss.detach().cpu().item())
            diagnostics["switch_rate_weighted_loss"] = float(
                switch_rate_weighted_loss.detach().cpu().item()
            )
            diagnostics["hard_switch_rate"] = float(hard_switch_rate.cpu().item())
        diagnostics["loss"] = float(batch_loss.detach().cpu().item())
        diagnostics["loss_abs"] = float(batch_loss.detach().abs().cpu().item())
        batch_loss.backward()
        if self.agent.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.agent.net.mon.parameters()), self.agent.max_grad_norm)
        self.agent.opt_mon.step()
        diagnostics["episode_count"] = int(valid_episodes)
        return diagnostics

    def _update_controller_aux_batch(self, aux_return_losses=None, aux_mdd_losses=None,
                                     aux_switch_adv_losses=None, local_adv_losses=None,
                                     sup_losses=None, switch_rate_losses=None):
        aux_return_losses = aux_return_losses or []
        aux_mdd_losses = aux_mdd_losses or []
        aux_switch_adv_losses = aux_switch_adv_losses or []
        local_adv_losses = local_adv_losses or []
        sup_losses = sup_losses or []
        switch_rate_losses = switch_rate_losses or []
        aux_return_coef = float(getattr(self.cfg, "controller_aux_return_coef", 0.0))
        aux_mdd_coef = float(getattr(self.cfg, "controller_aux_mdd_coef", 0.0))
        aux_switch_adv_coef = float(getattr(self.cfg, "controller_aux_switch_adv_coef", 0.0))
        local_adv_coef = float(getattr(self.cfg, "controller_local_adv_coef", 0.0))
        use_guidance_pretrain = bool(getattr(
            self.cfg,
            "controller_use_switch_supervision",
            False,
        ))
        guidance_coef = (
            float(getattr(self.cfg, "controller_guidance_pretrain_coef", 1.0))
            if use_guidance_pretrain
            else 0.0
        )
        risk_coef = guidance_coef if use_guidance_pretrain else aux_mdd_coef
        advantage_coef = guidance_coef if use_guidance_pretrain else aux_switch_adv_coef
        rate_coef = guidance_coef
        if (
                (aux_return_coef <= 0.0 or not aux_return_losses)
                and (risk_coef <= 0.0 or not aux_mdd_losses)
                and (advantage_coef <= 0.0 or not aux_switch_adv_losses)
                and (local_adv_coef <= 0.0 or not local_adv_losses)
                and (guidance_coef <= 0.0 or not sup_losses)
                and (rate_coef <= 0.0 or not switch_rate_losses)
        ):
            return {}

        self.agent.opt_mon.zero_grad(set_to_none=True)
        ref = (
            aux_return_losses
            or aux_mdd_losses
            or aux_switch_adv_losses
            or local_adv_losses
            or sup_losses
            or switch_rate_losses
        )[0]
        loss = ref.new_tensor(0.0)
        aux_return_loss = ref.new_tensor(0.0)
        aux_mdd_loss = ref.new_tensor(0.0)
        aux_switch_adv_loss = ref.new_tensor(0.0)
        local_adv_loss = ref.new_tensor(0.0)
        sup_loss = ref.new_tensor(0.0)
        switch_rate_loss = ref.new_tensor(0.0)
        if aux_return_coef > 0.0 and aux_return_losses:
            aux_return_loss = torch.stack(aux_return_losses).mean()
            loss = loss + aux_return_coef * aux_return_loss
        if risk_coef > 0.0 and aux_mdd_losses:
            aux_mdd_loss = torch.stack(aux_mdd_losses).mean()
            loss = loss + risk_coef * aux_mdd_loss
        if advantage_coef > 0.0 and aux_switch_adv_losses:
            aux_switch_adv_loss = torch.stack(aux_switch_adv_losses).mean()
            loss = loss + advantage_coef * aux_switch_adv_loss
        if local_adv_coef > 0.0 and local_adv_losses:
            local_adv_loss = torch.stack(local_adv_losses).mean()
            loss = loss + local_adv_coef * local_adv_loss
        if guidance_coef > 0.0 and sup_losses:
            sup_loss = torch.stack(sup_losses).mean()
            loss = loss + guidance_coef * sup_loss
        if rate_coef > 0.0 and switch_rate_losses:
            switch_rate_loss = torch.stack(switch_rate_losses).mean()
            loss = loss + rate_coef * switch_rate_loss
        loss.backward()
        if self.agent.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.agent.net.mon.parameters()), self.agent.max_grad_norm)
        self.agent.opt_mon.step()
        return {
            "loss": float(loss.detach().cpu().item()),
            "aux_return_loss": float(aux_return_loss.detach().cpu().item()),
            "aux_mdd_loss": float(aux_mdd_loss.detach().cpu().item()),
            "aux_switch_adv_loss": float(aux_switch_adv_loss.detach().cpu().item()),
            "aux_return_weighted_loss": float((aux_return_coef * aux_return_loss).detach().cpu().item()),
            "aux_mdd_weighted_loss": float((risk_coef * aux_mdd_loss).detach().cpu().item()),
            "aux_switch_adv_weighted_loss": float((advantage_coef * aux_switch_adv_loss).detach().cpu().item()),
            "local_adv_loss": float(local_adv_loss.detach().cpu().item()),
            "local_adv_weighted_loss": float((local_adv_coef * local_adv_loss).detach().cpu().item()),
            "sup_loss": float(sup_loss.detach().cpu().item()),
            "sup_weighted_loss": float((guidance_coef * sup_loss).detach().cpu().item()),
            "switch_rate_loss": float(switch_rate_loss.detach().cpu().item()),
            "switch_rate_weighted_loss": float((rate_coef * switch_rate_loss).detach().cpu().item()),
        }

    def _controller_aux_losses_from_episode_segments_batch(self, episode_segments_batch):
        aux_return_losses = []
        aux_mdd_losses = []
        aux_switch_adv_losses = []
        local_adv_losses = []
        sup_losses = []
        switch_rate_losses = []
        combined_segments = [
            segment
            for episode_segments in episode_segments_batch
            for segment in episode_segments
        ]
        if not combined_segments:
            return (
                aux_return_losses,
                aux_mdd_losses,
                aux_switch_adv_losses,
                local_adv_losses,
                sup_losses,
                switch_rate_losses,
            )
        terms, policy_logits = self._controller_episode_terms(
            combined_segments,
            return_policy_logits=True,
        )
        (
            _,
            _,
            _,
            aux_return_loss,
            aux_mdd_loss,
            aux_switch_adv_loss,
            _,
            sup_loss,
            local_adv_loss,
            _,
        ) = terms
        if aux_return_loss is not None:
            aux_return_losses.append(aux_return_loss)
        if aux_mdd_loss is not None:
            aux_mdd_losses.append(aux_mdd_loss)
        if aux_switch_adv_loss is not None:
            aux_switch_adv_losses.append(aux_switch_adv_loss)
        if local_adv_loss is not None:
            local_adv_losses.append(local_adv_loss)
        if sup_loss is not None:
            sup_losses.append(sup_loss)
        if (
                float(getattr(self.cfg, "controller_switch_rate_penalty_coef", 0.0)) > 0.0
                and policy_logits is not None
        ):
            switch_rate_loss, _ = self._controller_top_tail_rate_loss(
                policy_logits,
                min_rate=float(getattr(self.cfg, "controller_switch_rate_min", 0.05)),
                max_rate=float(getattr(self.cfg, "controller_switch_rate_max", 0.15)),
                margin=float(getattr(self.cfg, "controller_switch_rate_margin", 0.1)),
            )
            switch_rate_losses.append(switch_rate_loss)
        return (
            aux_return_losses,
            aux_mdd_losses,
            aux_switch_adv_losses,
            local_adv_losses,
            sup_losses,
            switch_rate_losses,
        )

    def _update_controller_aux_replay_batch(self, episode_segments_batch, replay_epochs=None):
        if not episode_segments_batch:
            return []
        replay_epochs = (
            int(getattr(self.cfg, "controller_aux_replay_epochs", 1) or 1)
            if replay_epochs is None
            else int(replay_epochs)
        )
        diagnostics = []
        for _ in range(max(1, replay_epochs)):
            losses = self._controller_aux_losses_from_episode_segments_batch(episode_segments_batch)
            diag = self._update_controller_aux_batch(*losses)
            if diag:
                diagnostics.append(diag)
        return diagnostics

    def _update_controller_sup_batch(self, sup_losses):
        if not sup_losses:
            return {}
        if not self._controller_switch_supervision_enabled():
            return {}
        sup_coef = float(getattr(self.cfg, "controller_sup_coef", getattr(self.cfg, "monitor_sup_coef", 0.0)))
        if sup_coef <= 0.0:
            return {}
        self.agent.opt_mon.zero_grad(set_to_none=True)
        sup_loss = torch.stack(sup_losses).mean()
        loss = sup_coef * sup_loss
        loss.backward()
        if self.agent.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.agent.net.mon.parameters()), self.agent.max_grad_norm)
        self.agent.opt_mon.step()
        return {
            "loss": float(loss.detach().cpu().item()),
            "sup_loss": float(sup_loss.detach().cpu().item()),
            "sup_weighted_loss": float(loss.detach().cpu().item()),
        }

    def train_controller_counterfactual_pg(self, *, epochs: int, fixed_cycle: int,
                                           val_interval: int = 1, save_name: str = "best_model.pth"):
        self.env.set_mode("train")
        self.buffer.clear()
        self.agent.set_module_status("controller")
        self._controller_baseline_cache = {}

        rollout_len = int(getattr(self.cfg, "controller_rollout_len", int(fixed_cycle) * 10))
        windows_per_epoch = int(getattr(self.cfg, "controller_windows_per_epoch", self.cfg.train_episodes_per_epoch))
        batch_windows = int(getattr(self.cfg, "controller_pg_batch_windows", 4))
        batch_windows = max(1, batch_windows)
        best_score = -np.inf
        update_count = 0
        pretrain_update_count = 0

        pending_episode_segments = []
        pending_segment_rewards = []
        pending_logprobs = []
        pending_rewards = []
        pending_entropies = []
        pending_aux_return_losses = []
        pending_aux_mdd_losses = []
        pending_aux_switch_adv_losses = []
        pending_overflow_action_losses = []
        pending_sup_losses = []
        pending_local_adv_losses = []
        pending_expected_switch_losses = []
        pending_episode_values = []

        def _validate_controller(epoch_idx: int):
            self.env.set_mode("val")
            self.agent.net.eval()
            ret_stats = self.run_episode(
                self.env,
                mode="eval",
                phase="joint",
                fixed_cycle=None,
                disable_inner=bool(getattr(self.cfg, "controller_pg_disable_inner", False)),
            )
            metrics = self._compute_metrics(ret_stats["history"])
            score = self._validation_score(metrics, self.cfg, phase="controller")
            self.env.set_mode("train")
            self.agent.net.train()
            self.logger.info(
                "   >>> [VAL controller_pg ep=%s] select=%s score=%.4f Sharpe=%.4f Ret=%.2f%% MDD=%.2f%% switches=%s",
                epoch_idx,
                getattr(self.cfg, "controller_selection_metric", "risk_return"),
                score,
                metrics["sharpe"],
                metrics["total_ret"] * 100.0,
                metrics["max_dd"] * 100.0,
                ret_stats.get("switch_count", 0),
            )
            return score, metrics

        fixed_episode_pool = bool(getattr(self.cfg, "controller_train_fixed_episodes", False))
        controller_start_pool = self._controller_train_start_pool(rollout_len) if fixed_episode_pool else None
        controller_train_max_hold = self._controller_train_max_hold(
            fixed_cycle=fixed_cycle,
            rollout_len=rollout_len,
        )
        controller_episode_batch_size = max(
            1,
            int(getattr(self.cfg, "controller_episode_batch_size", batch_windows)),
        )
        if fixed_episode_pool:
            self.logger.info(
                "   [Controller Batch] fixed episode pool=%s, episode_batch_size=%s, parallel_workers=%s, "
                "episode_len=%s, start_stride_days=%s, train_max_hold=%s, "
                "record_max_duration=%s, switch_adv_mining=%s",
                len(controller_start_pool),
                controller_episode_batch_size,
                max(1, int(getattr(self.cfg, "controller_episode_parallel_workers", 1))),
                rollout_len,
                int(getattr(self.cfg, "controller_start_stride_days", self.cfg.max_hold)),
                controller_train_max_hold,
                int(getattr(self.cfg, "controller_train_record_max_duration", 0) or 0),
                bool(getattr(self.cfg, "controller_compute_switch_advantage", False)),
            )

        sup_pretrain_epochs = int(getattr(self.cfg, "controller_sup_pretrain_epochs", 0) or 0)
        pretrain_only = bool(getattr(self.cfg, "controller_pretrain_only", False))
        if pretrain_only and (sup_pretrain_epochs <= 0 or not fixed_episode_pool):
            raise ValueError(
                "controller_pretrain_only requires controller_sup_pretrain_epochs > 0 "
                "and controller_train_fixed_episodes."
            )
        if sup_pretrain_epochs > 0 and fixed_episode_pool:
            sup_rollout_len = int(getattr(self.cfg, "controller_sup_pretrain_rollout_len", 0) or 0)
            sup_rollout_len = rollout_len if sup_rollout_len <= 0 else max(2, sup_rollout_len)
            aux_replay_epochs = max(1, int(getattr(self.cfg, "controller_aux_replay_epochs", 1) or 1))
            self.logger.info(
                "   [Controller Aux] pretrain epochs=%s, replay_epochs=%s, "
                "pretrain_aux_coef=%s, episode_len=%s, "
                "top_tail_rate=[%.3f, %.3f], margin=%.3f; "
                "PG coefs: risk=%.3f advantage=%.3f label=%.3f rate=%.3f",
                sup_pretrain_epochs,
                aux_replay_epochs,
                float(getattr(self.cfg, "controller_guidance_pretrain_coef", 1.0)),
                sup_rollout_len,
                float(getattr(self.cfg, "controller_switch_rate_min", 0.05)),
                float(getattr(self.cfg, "controller_switch_rate_max", 0.15)),
                float(getattr(self.cfg, "controller_switch_rate_margin", 0.1)),
                float(getattr(self.cfg, "controller_aux_mdd_coef", 0.0)),
                float(getattr(self.cfg, "controller_aux_switch_adv_coef", 0.0)),
                float(getattr(self.cfg, "controller_sup_coef", 0.0)),
                float(getattr(self.cfg, "controller_switch_rate_penalty_coef", 0.0)),
            )
            for sup_epoch in range(sup_pretrain_epochs):
                starts = list(controller_start_pool)
                if sup_epoch > 0:
                    random.shuffle(starts)
                aux_pretrain_segments = []
                sup_updates = 0
                for batch_start in range(0, len(starts), controller_episode_batch_size):
                    batch_starts = starts[batch_start:batch_start + controller_episode_batch_size]
                    windows = []
                    for start in batch_starts:
                        stop = min(int(start) + sup_rollout_len, int(self.env.idx_map["train"][-1]))
                        if stop - int(start) >= max(2, fixed_cycle):
                            windows.append((int(start), int(stop)))
                    if not windows:
                        continue
                    controlled_results = self._run_controller_aux_pretrain_windows(
                        windows,
                        fixed_cycle,
                        epoch=sup_epoch,
                    )
                    for controlled in controlled_results:
                        if controlled.get("episode_segments"):
                            aux_pretrain_segments.append(controlled["episode_segments"])
                    if len(aux_pretrain_segments) >= controller_episode_batch_size:
                        diags = self._update_controller_aux_replay_batch(
                            aux_pretrain_segments,
                            replay_epochs=aux_replay_epochs,
                        )
                        for replay_idx, diag in enumerate(diags, start=1):
                            sup_updates += 1
                            pretrain_update_count += 1
                            self.logger.info(
                                "[CTRL-AUX] update=%s epoch=%s replay=%s/%s windows=%s/%s "
                                "loss=%.4f label=%.4f risk=%.4f advantage=%.4f "
                                "rate_w=%.4f aux_ret=%.4f local_adv=%.4f",
                                sup_updates,
                                sup_epoch + 1,
                                replay_idx,
                                aux_replay_epochs,
                                min(batch_start + len(batch_starts), len(starts)),
                                len(starts),
                                diag.get("loss", 0.0),
                                diag.get("sup_loss", 0.0),
                                diag.get("aux_mdd_loss", 0.0),
                                diag.get("aux_switch_adv_loss", 0.0),
                                diag.get("switch_rate_weighted_loss", 0.0),
                                diag.get("aux_return_loss", 0.0),
                                diag.get("local_adv_loss", 0.0),
                            )
                        aux_pretrain_segments = []
                if aux_pretrain_segments:
                    diags = self._update_controller_aux_replay_batch(
                        aux_pretrain_segments,
                        replay_epochs=aux_replay_epochs,
                    )
                    for replay_idx, diag in enumerate(diags, start=1):
                        sup_updates += 1
                        pretrain_update_count += 1
                        self.logger.info(
                            "[CTRL-AUX] update=%s epoch=%s final-batch replay=%s/%s "
                            "loss=%.4f label=%.4f risk=%.4f advantage=%.4f "
                            "rate_w=%.4f aux_ret=%.4f local_adv=%.4f",
                            sup_updates,
                            sup_epoch + 1,
                            replay_idx,
                            aux_replay_epochs,
                            diag.get("loss", 0.0),
                            diag.get("sup_loss", 0.0),
                            diag.get("aux_mdd_loss", 0.0),
                            diag.get("aux_switch_adv_loss", 0.0),
                            diag.get("switch_rate_weighted_loss", 0.0),
                            diag.get("aux_return_loss", 0.0),
                            diag.get("local_adv_loss", 0.0),
                        )
            pending_logprobs, pending_rewards, pending_entropies = [], [], []
            pending_aux_return_losses, pending_aux_mdd_losses = [], []
            pending_aux_switch_adv_losses = []
            pending_overflow_action_losses, pending_sup_losses = [], []
            pending_local_adv_losses = []
            pending_expected_switch_losses = []
            pending_episode_values = []
            pending_episode_segments, pending_segment_rewards = [], []

        if pretrain_only:
            self.save_model(save_name)
            self.logger.info(
                "   ↺ Controller supervised pretraining finished after %s updates; "
                "saved %s and skipped policy-gradient rollouts.",
                pretrain_update_count,
                save_name,
            )
            return {
                "best_score": 0.0,
                "updates": int(pretrain_update_count),
                "selection_metric": str(getattr(
                    self.cfg,
                    "controller_selection_metric",
                    "return",
                )),
                "pretrain_only": True,
            }

        def _record_controlled_result(baseline_stats, controlled, start, stop):
            ctrl_stats = controlled["stats"]
            window_len = max(1, int(stop) - int(start))
            max_allowed_switches = self._controller_max_allowed_switches(
                window_len,
                int(getattr(self.cfg, "min_hold", 10)),
            )
            reward = controller_reward(
                baseline_stats,
                ctrl_stats,
                reward_mode=str(getattr(self.cfg, "controller_reward_mode", "return_uplift")),
                return_coef=float(getattr(self.cfg, "controller_return_coef", 1.0)),
                downside_coef=float(getattr(self.cfg, "controller_downside_coef", 0.0)),
                mdd_coef=float(getattr(self.cfg, "controller_mdd_coef", 0.0)),
                max_switch_count=max_allowed_switches,
                max_switch_penalty_coef=float(getattr(
                    self.cfg,
                    "controller_max_switch_penalty_coef",
                    getattr(self.cfg, "controller_count_penalty_coef", 0.5),
                )),
            )
            if controlled["episode_logprob"] is not None:
                pending_logprobs.append(controlled["episode_logprob"])
                pending_rewards.append(float(reward))
                pending_entropies.append(
                    controlled["entropy"]
                    if controlled["entropy"] is not None
                    else controlled["episode_logprob"].new_tensor(0.0)
                )
                if controlled.get("aux_return_loss") is not None:
                    pending_aux_return_losses.append(controlled["aux_return_loss"])
                if controlled.get("aux_mdd_loss") is not None:
                    pending_aux_mdd_losses.append(controlled["aux_mdd_loss"])
                if controlled.get("aux_switch_adv_loss") is not None:
                    pending_aux_switch_adv_losses.append(controlled["aux_switch_adv_loss"])
                if controlled.get("episode_value") is not None:
                    pending_episode_values.append(controlled["episode_value"])
            elif controlled.get("episode_segments"):
                pending_episode_segments.append(controlled["episode_segments"])
                pending_segment_rewards.append(float(reward))
            return reward, ctrl_stats, max_allowed_switches

        def _clear_controller_pg_pending():
            pending_episode_segments.clear()
            pending_segment_rewards.clear()
            pending_logprobs.clear()
            pending_rewards.clear()
            pending_entropies.clear()
            pending_aux_return_losses.clear()
            pending_aux_mdd_losses.clear()
            pending_aux_switch_adv_losses.clear()
            pending_overflow_action_losses.clear()
            pending_sup_losses.clear()
            pending_local_adv_losses.clear()
            pending_expected_switch_losses.clear()
            pending_episode_values.clear()

        def _update_controller_pg_pending():
            if pending_episode_segments:
                diag = self._update_controller_pg_segments_batch(
                    pending_episode_segments,
                    pending_segment_rewards,
                )
            else:
                diag = self._update_controller_pg_batch(
                    pending_logprobs,
                    pending_rewards,
                    pending_entropies,
                    pending_aux_return_losses,
                    pending_aux_mdd_losses,
                    pending_aux_switch_adv_losses,
                    pending_overflow_action_losses,
                    pending_sup_losses,
                    pending_local_adv_losses,
                    pending_expected_switch_losses,
                    pending_episode_values,
                )
            _clear_controller_pg_pending()
            return diag

        for epoch in range(max(0, int(epochs))):
            if fixed_episode_pool:
                starts = list(controller_start_pool)
                if epoch > 0:
                    random.shuffle(starts)
            else:
                starts = self._controller_train_starts(windows_per_epoch, rollout_len, epoch)
            epoch_rewards = []
            epoch_mdd_uplifts = []
            epoch_return_uplifts = []
            epoch_segments = []
            epoch_free_switches = []
            epoch_switch_overflows = []
            epoch_max_switches = []
            epoch_switch_adv_count = 0
            epoch_switch_adv_sum = 0.0
            epoch_switch_adv_positive_count = 0
            epoch_switch_adv_pos_exit_sum = 0.0
            epoch_switch_adv_pos_exit_count = 0
            epoch_switch_adv_neg_exit_sum = 0.0
            epoch_switch_adv_neg_exit_count = 0

            for batch_start in range(0, len(starts), controller_episode_batch_size if fixed_episode_pool else 1):
                batch_starts = starts[batch_start:batch_start + (controller_episode_batch_size if fixed_episode_pool else 1)]
                windows = []
                for start in batch_starts:
                    stop = min(int(start) + rollout_len, int(self.env.idx_map["train"][-1]))
                    if stop - int(start) >= max(2, fixed_cycle):
                        windows.append((int(start), int(stop)))
                if not windows:
                    continue

                pair_results = (
                    self._run_controller_pg_pairs(windows, fixed_cycle, epoch=epoch)
                    if fixed_episode_pool
                    else [
                        (
                            self._run_fixed_hrl_window(
                                self.env,
                                windows[0][0],
                                windows[0][1],
                                fixed_cycle,
                                disable_inner=bool(getattr(self.cfg, "controller_pg_disable_inner", False)),
                            )[0],
                            self._run_controller_pg_window(
                                self.env,
                                windows[0][0],
                                windows[0][1],
                                fixed_cycle,
                                epoch=epoch,
                            ),
                        )
                    ]
                )

                for (start, stop), (baseline_stats, controlled) in zip(windows, pair_results):
                    reward, ctrl_stats, max_allowed_switches = _record_controlled_result(
                        baseline_stats,
                        controlled,
                        start,
                        stop,
                    )
                    epoch_rewards.append(reward)
                    epoch_mdd_uplifts.append(baseline_stats.max_drawdown - ctrl_stats.max_drawdown)
                    epoch_return_uplifts.append(ctrl_stats.log_return - baseline_stats.log_return)
                    epoch_segments.append(ctrl_stats.segment_count)
                    epoch_free_switches.append(ctrl_stats.free_switch_count)
                    epoch_switch_overflows.append(max(0, ctrl_stats.segment_count - max_allowed_switches))
                    epoch_max_switches.append(max_allowed_switches)
                    switch_adv_summary = controlled.get("switch_advantage_summary") or {}
                    switch_adv_count = int(switch_adv_summary.get("count", 0) or 0)
                    if switch_adv_count > 0:
                        epoch_switch_adv_count += switch_adv_count
                        epoch_switch_adv_sum += float(switch_adv_summary.get("mean", 0.0)) * switch_adv_count
                        epoch_switch_adv_positive_count += int(switch_adv_summary.get("positive_count", 0) or 0)
                        pos_exit_count = int(switch_adv_summary.get("positive_exit_prob_count", 0) or 0)
                        neg_exit_count = int(switch_adv_summary.get("negative_exit_prob_count", 0) or 0)
                        if pos_exit_count > 0:
                            epoch_switch_adv_pos_exit_sum += (
                                float(switch_adv_summary.get("positive_exit_prob_mean", 0.0))
                                * pos_exit_count
                            )
                            epoch_switch_adv_pos_exit_count += pos_exit_count
                        if neg_exit_count > 0:
                            epoch_switch_adv_neg_exit_sum += (
                                float(switch_adv_summary.get("negative_exit_prob_mean", 0.0))
                                * neg_exit_count
                            )
                            epoch_switch_adv_neg_exit_count += neg_exit_count

                pending_pg_count = len(pending_episode_segments) if pending_episode_segments else len(pending_logprobs)
                should_update = (
                    pending_pg_count >= controller_episode_batch_size
                    if fixed_episode_pool
                    else pending_pg_count >= batch_windows
                )
                if should_update:
                    diag = _update_controller_pg_pending()
                    update_count += 1
                    self.logger.info(
                        "[CTRL-PG] update=%s epoch=%s windows=%s/%s loss=%.4f loss_abs=%.4f "
                        "policy=%.4f policy_abs=%.4f "
                        "ent_loss=%.4f aux_ret_w=%.4f aux_mdd_w=%.4f aux_sw_w=%.4f overflow_act=%.4f "
                        "sup_w=%.4f local_adv_w=%.4f rate_w=%.4f hard_rate=%.3f value_w=%.4f "
                        "reward=%.4f rew_abs=%.4f rew_std=%.4f rew_min=%.4f rew_max=%.4f "
                        "entropy=%.4f aux_ret=%.4f aux_mdd=%.4f sup=%.4f "
                        "local_adv=%.4f value=%.4f",
                        update_count, epoch + 1, min(batch_start + len(batch_starts), len(starts)), len(starts),
                        diag.get("loss", 0.0), diag.get("loss_abs", 0.0),
                        diag.get("policy_loss", 0.0), diag.get("policy_abs_loss", 0.0),
                        diag.get("entropy_loss", 0.0),
                        diag.get("aux_return_weighted_loss", 0.0),
                        diag.get("aux_mdd_weighted_loss", 0.0),
                        diag.get("aux_switch_adv_weighted_loss", 0.0),
                        diag.get("overflow_action_loss", 0.0),
                        diag.get("sup_weighted_loss", 0.0),
                        diag.get("local_adv_weighted_loss", 0.0),
                        diag.get("switch_rate_weighted_loss", 0.0),
                        diag.get("hard_switch_rate", 0.0),
                        diag.get("value_weighted_loss", 0.0),
                        diag.get("reward_mean", 0.0),
                        diag.get("reward_abs_mean", 0.0),
                        diag.get("reward_std", 0.0),
                        diag.get("reward_min", 0.0),
                        diag.get("reward_max", 0.0),
                        diag.get("entropy", 0.0),
                        diag.get("aux_return_loss", 0.0), diag.get("aux_mdd_loss", 0.0),
                        diag.get("sup_loss", 0.0),
                        diag.get("local_adv_loss", 0.0),
                        diag.get("value_loss", 0.0),
                    )

            if pending_episode_segments or pending_logprobs:
                diag = _update_controller_pg_pending()
                update_count += 1
                self.logger.info(
                    "[CTRL-PG] update=%s epoch=%s final-batch loss=%.4f loss_abs=%.4f "
                    "policy=%.4f policy_abs=%.4f "
                    "ent_loss=%.4f aux_ret_w=%.4f aux_mdd_w=%.4f aux_sw_w=%.4f overflow_act=%.4f "
                    "sup_w=%.4f local_adv_w=%.4f rate_w=%.4f hard_rate=%.3f value_w=%.4f "
                    "reward=%.4f rew_abs=%.4f rew_std=%.4f rew_min=%.4f rew_max=%.4f "
                    "entropy=%.4f aux_ret=%.4f aux_mdd=%.4f sup=%.4f "
                    "local_adv=%.4f value=%.4f",
                    update_count, epoch + 1,
                    diag.get("loss", 0.0), diag.get("loss_abs", 0.0),
                    diag.get("policy_loss", 0.0), diag.get("policy_abs_loss", 0.0),
                    diag.get("entropy_loss", 0.0),
                    diag.get("aux_return_weighted_loss", 0.0),
                    diag.get("aux_mdd_weighted_loss", 0.0),
                    diag.get("aux_switch_adv_weighted_loss", 0.0),
                    diag.get("overflow_action_loss", 0.0),
                    diag.get("sup_weighted_loss", 0.0),
                    diag.get("local_adv_weighted_loss", 0.0),
                    diag.get("switch_rate_weighted_loss", 0.0),
                    diag.get("hard_switch_rate", 0.0),
                    diag.get("value_weighted_loss", 0.0),
                    diag.get("reward_mean", 0.0),
                    diag.get("reward_abs_mean", 0.0),
                    diag.get("reward_std", 0.0),
                    diag.get("reward_min", 0.0),
                    diag.get("reward_max", 0.0),
                    diag.get("entropy", 0.0),
                    diag.get("aux_return_loss", 0.0), diag.get("aux_mdd_loss", 0.0),
                    diag.get("sup_loss", 0.0),
                    diag.get("local_adv_loss", 0.0),
                    diag.get("value_loss", 0.0),
                )

            self.logger.info(
                "[CTRL-PG] epoch %s/%s | reward=%.4f mdd_uplift=%.4f ret_uplift=%.4f "
                "switches=%.2f max_switch=%.2f overflow=%.2f free_switch=%.2f "
                "sw_adv_mean=%.5f sw_adv_pos=%.3f sw_adv_n=%s "
                "exit_pos=%.5f exit_neg=%.5f exit_gap=%.5f updates=%s",
                epoch + 1, epochs,
                float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
                float(np.mean(epoch_mdd_uplifts)) if epoch_mdd_uplifts else 0.0,
                float(np.mean(epoch_return_uplifts)) if epoch_return_uplifts else 0.0,
                float(np.mean(epoch_segments)) if epoch_segments else 0.0,
                float(np.mean(epoch_max_switches)) if epoch_max_switches else 0.0,
                float(np.mean(epoch_switch_overflows)) if epoch_switch_overflows else 0.0,
                float(np.mean(epoch_free_switches)) if epoch_free_switches else 0.0,
                float(epoch_switch_adv_sum / epoch_switch_adv_count) if epoch_switch_adv_count > 0 else 0.0,
                float(epoch_switch_adv_positive_count / epoch_switch_adv_count) if epoch_switch_adv_count > 0 else 0.0,
                int(epoch_switch_adv_count),
                float(epoch_switch_adv_pos_exit_sum / epoch_switch_adv_pos_exit_count)
                if epoch_switch_adv_pos_exit_count > 0 else 0.0,
                float(epoch_switch_adv_neg_exit_sum / epoch_switch_adv_neg_exit_count)
                if epoch_switch_adv_neg_exit_count > 0 else 0.0,
                (
                    float(epoch_switch_adv_pos_exit_sum / epoch_switch_adv_pos_exit_count)
                    if epoch_switch_adv_pos_exit_count > 0 else 0.0
                ) - (
                    float(epoch_switch_adv_neg_exit_sum / epoch_switch_adv_neg_exit_count)
                    if epoch_switch_adv_neg_exit_count > 0 else 0.0
                ),
                update_count,
            )

            if self._should_validate_controller_pg(epoch=epoch + 1, epochs=int(epochs), val_interval=val_interval):
                score, _ = _validate_controller(epoch + 1)
                if score > best_score:
                    best_score = score
                    self.save_model(save_name)
                    self.logger.info("       (New Controller-PG Best: %.4f)", best_score)

        if os.path.exists(os.path.join(self.model_dir, save_name)):
            self._load_model(save_name)
            self.logger.info("   ↺ Controller-PG finished. Loaded best controller model.")
        return {
            "best_score": float(best_score),
            "updates": int(update_count),
            "selection_metric": str(getattr(self.cfg, "controller_selection_metric", "risk_return")),
        }

    # ==============================================================================
    # 评估与模型管理
    # ==============================================================================
    def evaluate(self, phase):
        """
        在验证集上评估，支持传入 phase 以保持逻辑一致性
        """
        self.env.set_mode('val')
        self.agent.net.eval()
        ret_stats = self.run_episode(self.env, mode='eval', phase=phase)
        metrics = self._compute_metrics(ret_stats['history'])
        self.env.set_mode('train')  # 切回训练模式
        self.agent.net.train()
        return metrics

    def save_model(self, name):
        state = {
            'agent_net': self.agent.net.state_dict(),
            'opt_mon': self.agent.opt_mon.state_dict(),
            'opt_out': self.agent.opt_out.state_dict(),
            'opt_inn': self.agent.opt_inn.state_dict(),
        }
        torch.save(state, os.path.join(self.model_dir, name))

    def _load_model(self, name):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.agent.net.load_state_dict(ckpt['agent_net'])
            # 加载优化器状态有助于在阶段切换时平滑过渡
            self.agent.opt_mon.load_state_dict(ckpt['opt_mon'])
            self.agent.opt_out.load_state_dict(ckpt['opt_out'])
            self.agent.opt_inn.load_state_dict(ckpt['opt_inn'])
            return True
        return False

    def _capture_train_episode_config(self):
        keys = (
            "train_episode_to_end",
            "train_episodes_per_epoch",
            "train_episode_count",
            "train_start_stride_days",
            "train_episode_start_stride",
            "episode_len",
            "stride",
            "train_ptr",
            "completed_train_epoch_count",
        )
        return {key: getattr(self.env, key) for key in keys if hasattr(self.env, key)}

    def _reset_train_episode_pool(self):
        if hasattr(self.env, "_train_pool_signature"):
            delattr(self.env, "_train_pool_signature")
        if hasattr(self.env, "train_ptr"):
            self.env.train_ptr = 0
        if hasattr(self.env, "completed_train_epoch_count"):
            self.env.completed_train_epoch_count = 0
        self.env.set_mode("train")

    def _apply_train_episode_config(self, *, train_episode_to_end, train_episodes_per_epoch,
                                    train_start_stride_days, episode_len):
        previous = self._capture_train_episode_config()
        train_episodes_per_epoch = max(1, int(train_episodes_per_epoch))
        train_start_stride_days = max(1, int(train_start_stride_days))

        self.env.train_episode_to_end = bool(train_episode_to_end)
        self.env.train_episodes_per_epoch = train_episodes_per_epoch
        self.env.train_episode_count = train_episodes_per_epoch
        self.env.train_start_stride_days = train_start_stride_days
        self.env.train_episode_start_stride = train_start_stride_days
        self.env.episode_len = max(1, int(episode_len))
        if hasattr(self.env, "stride"):
            self.env.stride = train_start_stride_days

        self._reset_train_episode_pool()
        return previous

    def _restore_train_episode_config(self, previous):
        for key, value in previous.items():
            setattr(self.env, key, value)
        self._reset_train_episode_pool()

    def _new_rollout_buffer(self):
        return HRL_Buffer(
            capacity=getattr(self.buffer, "capacity", 100000),
            device=self.device,
            gamma=getattr(self.buffer, "gamma", 0.99),
            gae_lambda=getattr(self.buffer, "gae_lambda", 0.95),
            outer_reward_scale=getattr(self.buffer, "outer_reward_scale", 1.0),
            outer_reward_mode=getattr(self.buffer, "outer_reward_mode", "return"),
        )

    @staticmethod
    def _merge_buffer_data(target_buffer, source_buffer):
        for key, values in source_buffer.data.items():
            if not values:
                continue
            target_buffer.data.setdefault(key, []).extend(values)

    def _reserve_train_episode_windows(self, episode_count: int):
        windows = []
        episode_count = max(1, int(episode_count))
        pool = getattr(self.env, "train_indices_pool", None)
        if not pool:
            raise RuntimeError("Cannot reserve parallel train episodes without train_indices_pool.")

        for _ in range(episode_count):
            if self.env.train_ptr >= len(self.env.train_indices_pool):
                self.env.train_ptr = 0
                self.env.completed_train_epoch_count = int(getattr(self.env, "completed_train_epoch_count", 0)) + 1
                if not bool(getattr(self.env, "train_episode_to_end", True)):
                    random.shuffle(self.env.train_indices_pool)
                    if getattr(self.env, "logger", None):
                        self.env.logger.info(
                            f"Completed train epoch {self.env.completed_train_epoch_count}; reshuffled pool."
                        )

            start_idx = int(self.env.train_indices_pool[self.env.train_ptr])
            self.env.train_ptr += 1
            if bool(getattr(self.env, "train_episode_to_end", True)):
                stop_idx = int(getattr(self.env, "train_episode_stop_idx"))
            else:
                stop_idx = start_idx + int(getattr(self.env, "episode_len", 1))
            windows.append((start_idx, stop_idx))

        return windows

    def _run_inner_episode_worker(self, start_idx: int, stop_idx: int, fixed_cycle: int, train_monitor: bool):
        local_env = copy.deepcopy(self.env)
        local_env.logger = None
        local_env.mode = "train"
        local_buffer = self._new_rollout_buffer()
        ret = self.run_episode(
            local_env,
            mode="train",
            phase="warmup_inner",
            fixed_cycle=fixed_cycle,
            use_rule_switch=False,
            rollout_update_steps=0,
            auto_update_phase=None,
            train_monitor=train_monitor,
            rollout_buffer=local_buffer,
            explicit_episode_window=(int(start_idx), int(stop_idx)),
        )
        ret["buffer"] = local_buffer
        ret["episodes"] = 1
        return ret

    def _run_inner_episode_batch(self, episode_count: int, *, fixed_cycle: int, train_monitor: bool = False):
        """Run several fixed-length Inner episodes, then update Inner once on the combined buffer."""
        episode_count = max(1, int(episode_count))
        histories = []
        total_steps = 0
        parallel_workers = min(
            episode_count,
            max(1, int(getattr(self.cfg, "inner_episode_parallel_workers", 1))),
        )

        if parallel_workers > 1:
            windows = self._reserve_train_episode_windows(episode_count)
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [
                    executor.submit(
                        self._run_inner_episode_worker,
                        start_idx,
                        stop_idx,
                        fixed_cycle,
                        train_monitor,
                    )
                    for start_idx, stop_idx in windows
                ]
                results = [future.result() for future in futures]

            for ret in results:
                histories.append(ret)
                total_steps += int(ret.get("total_steps", 0))
                source_buffer = ret.get("buffer")
                if source_buffer is not None:
                    self._merge_buffer_data(self.buffer, source_buffer)
        else:
            for _ in range(episode_count):
                ret = self.run_episode(
                    self.env,
                    mode="train",
                    phase="warmup_inner",
                    fixed_cycle=fixed_cycle,
                    use_rule_switch=False,
                    rollout_update_steps=0,
                    auto_update_phase=None,
                    train_monitor=train_monitor,
                )
                histories.append(ret)
                total_steps += int(ret.get("total_steps", 0))

        loss_log = {}
        update_count = 0
        if len(self.buffer.data.get("rew_mon", [])) > 0:
            loss_log = self.agent.update(
                self.buffer.get_batch(),
                phase="warmup_inner",
                train_monitor=train_monitor,
            )
            update_count = 1
            self.buffer.clear()
            if hasattr(self.buffer, "mark_episode_start"):
                self.buffer.mark_episode_start()
            if (
                    bool(getattr(self.cfg, "clear_cuda_cache_on_update", False))
                    and getattr(self.device, "type", str(self.device)) == "cuda"
            ):
                torch.cuda.empty_cache()

        return {
            "episodes": episode_count,
            "total_steps": total_steps,
            "history": histories[-1].get("history", []) if histories else [],
            "loss_log": loss_log,
            "update_count": update_count,
        }

    def _run_joint_full_train_episode(self, *, fixed_cycle: int, rollout_update_steps: int,
                                      train_monitor: bool = False):
        """Run one Joint episode from the first train start to train_end, then restore schedule."""
        previous = self._apply_train_episode_config(
            train_episode_to_end=True,
            train_episodes_per_epoch=1,
            train_start_stride_days=1,
            episode_len=getattr(self.env, "episode_len", max(1, int(fixed_cycle))),
        )
        try:
            return self.run_episode(
                self.env,
                mode="train",
                phase="joint",
                fixed_cycle=fixed_cycle,
                use_rule_switch=False,
                rollout_update_steps=int(rollout_update_steps),
                auto_update_phase="joint",
                train_monitor=train_monitor,
            )
        finally:
            self._restore_train_episode_config(previous)

    def load_frozen_hrl_checkpoint(self, path):
        """Load pretrained Outer/Inner parameters and leave EmbMonitor fresh."""
        checkpoint = torch.load(path, map_location=self.device)
        source_state = checkpoint.get('agent_net', checkpoint)
        reusable = {
            key: value for key, value in source_state.items()
            if key.startswith('outer.') or key.startswith('inner.')
        }
        expected = {
            key for key in self.agent.net.state_dict()
            if key.startswith('outer.') or key.startswith('inner.')
        }
        missing = sorted(expected.difference(reusable))
        if missing:
            raise RuntimeError(f"Pretrained HRL checkpoint misses reusable parameters: {missing[:5]}")
        self.agent.net.load_state_dict(reusable, strict=False)
        self.agent.set_module_status("monitor")
        self.logger.info(f"Loaded frozen Outer/Inner HRL checkpoint: {path}")
        return path

    # ==============================================================================
    # 核心：带验证与回滚的训练块 (Training Block with Validation & Rollback)
    # ==============================================================================
    def _run_block(self, log_name, episodes, phase, save_mode='last'):
        """
        运行一个训练阶段
        :param save_mode: 'last' (保留最后状态，用于预热/轮转) 或 'best' (回滚到最优，用于联合训练)
        """
        best_sharpe = -np.inf
        # 为每个阶段定义唯一的模型名
        best_ckpt = os.path.join(self.model_dir, f"best_{log_name}_{phase}.pth")
        last_ckpt = os.path.join(self.model_dir, f"last_{log_name}_{phase}.pth")

        # 确保该阶段开始时 buffer 为空（支持多 episode 累积）
        self.buffer.clear()

        for i in range(episodes):
            # 1) Rollout (多 episode 累积)
            ret = self.run_episode(self.env, mode='train', phase=phase)

            # 2) Update：攒够 rollout_episodes（或到阶段结束）再更新一次
            rollout_episodes = int(getattr(self.cfg, 'rollout_episodes', 1))
            do_update = ((i + 1) % rollout_episodes == 0) or ((i + 1) == episodes)
            if do_update:
                loss_log = self.agent.update(self.buffer.get_batch(), phase=phase)
                self.buffer.clear()
            else:
                loss_log = {}

            # 2. Log Training Stats
            real_ret = ret['total'] * 100.0
            log_str = (f"[{log_name[:3].upper()}] {i + 1}/{episodes} | "
                       f"Ret:{real_ret:5.2f}% | "
                       f"Sw:{ret['switch_count']} (free:{ret.get('switch_free_count', 0)}, "
                       f"forced_h:{ret.get('forced_hold_count', 0)}, forced_s:{ret.get('forced_switch_count', 0)})")
            if 'inn_pi' in loss_log: log_str += f" | L_In:{loss_log['inn_pi']:.3f}"
            if 'out_pi' in loss_log: log_str += f" | L_Out:{loss_log['out_pi']:.3f}"
            if 'out_pred' in loss_log: log_str += f" | L_OutPred:{loss_log['out_pred']:.3f}"
            if 'mon_pi' in loss_log: log_str += f" | L_Mon:{loss_log.get('mon_pi', 0):.3f}"
            self.logger.info(log_str)

            # 3. Validate & Save Best
            # 即使在 save_mode='last' 模式下，我们依然保存 best 以便观察潜力和手动恢复
            if (i + 1) % self.cfg.val_interval == 0 or i == episodes - 1:
                val_metrics = self.evaluate(phase)
                val_sharpe = val_metrics['sharpe']
                self.logger.info(
                    f"   >>> [VAL] Sharpe: {val_sharpe:.4f} | Ret: {val_metrics['total_ret'] * 100:.2f}% | DD: {val_metrics['max_dd'] * 100:.2f}%")

                if val_sharpe > best_sharpe:
                    best_sharpe = val_sharpe
                    self.save_model(best_ckpt)
                    self.logger.info(f"       (New Best Saved: {best_sharpe:.4f})")

        # 4. End of Block Decision
        # 首先保存当前的最后状态
        self.save_model(last_ckpt)

        if save_mode == 'best':
            # 联合训练阶段：我们想要最好的结果
            self.logger.info(f"   ↺ End of {phase}. Rolling back to BEST model: {best_ckpt}")
            if not self._load_model(best_ckpt):
                self.logger.warning("   ⚠️ Load failed, continuing with last model.")
            # 顺便更新全局最优模型 (用于 Test)
            self.save_model("final_best_model.pth")

        else:
            # 预热/轮转阶段：我们想要参数的连续性，继续往下训练
            self.logger.info(f"   End of {phase}. Keeping LAST model state for next phase: {last_ckpt}")
            # 不需要 load，因为当前内存里的就是 last，但为了保险可以 load 一次确保文件没坏
            # self._load_model(last_ckpt)

    # ==============================================================================
    # 主训练流程：混合策略 (Hybrid Strategy)
    # ==============================================================================
    def train_hybrid_strategy(self):
        total = self.cfg.train_episodes
        # 建议比例: 20% 预热, 50% 轮转, 30% 联合
        warmup = int(total * 0.2)
        round_robin = int(total * 0.5)
        joint = total - warmup - round_robin

        self.logger.info("==========================================")
        self.logger.info("🚀 START HYBRID TRAINING STRATEGY")
        self.logger.info(f"Plan: Warmup({warmup}) -> RoundRobin({round_robin}) -> Joint({joint})")
        self.logger.info("==========================================")

        # --- Phase 1: Ordered Warmup (Save Last) ---
        # 1.1 Outer
        p1_out = int(warmup * 0.4)
        self.logger.info(f"\n[Phase 1.1] Outer Warm-up ({p1_out} eps)")
        self.agent.set_module_status('outer')
        self._run_block('outer', p1_out, 'warmup_outer', save_mode='last')

        # 1.2 Inner
        p1_inn = int(warmup * 0.4)
        self.logger.info(f"\n[Phase 1.2] Inner Warm-up ({p1_inn} eps)")
        self.agent.set_module_status('inner')
        self._run_block('inner', p1_inn, 'warmup_inner', save_mode='last')

        # 1.3 Monitor
        p1_mon = max(1, warmup - p1_out - p1_inn)
        self.logger.info(f"\n[Phase 1.3] Monitor Warm-up ({p1_mon} eps)")
        self.agent.set_module_status('monitor')
        self._run_block('monitor', p1_mon, 'warmup_monitor', save_mode='last')

        # --- Phase 2: Round Robin (Save Last) ---
        self.logger.info("\n==========================================")
        self.logger.info(f"[Phase 2] Round-Robin Cycling ({round_robin} eps)")
        cycle = [('outer', 30), ('inner', 30), ('monitor', 30)]
        done = 0
        idx = 0
        while done < round_robin:
            mod, count = cycle[idx % 3]
            actual = min(count, round_robin - done)
            if actual <= 0: break

            self.logger.info(f"   >>> Cycle: Training {mod.upper()} ({actual} eps)")
            self.agent.set_module_status(mod)
            # 轮转阶段：保持参数连续性，使用 save_mode='last'
            self._run_block(mod, actual, f"round_{mod}", save_mode='last')

            done += actual
            idx += 1

        # --- Phase 3: Joint Fine-tuning (Save Best) ---
        self.logger.info("\n==========================================")
        self.logger.info(f"[Phase 3] Joint Fine-tuning ({joint} eps)")
        self.agent.set_module_status('all')
        # LR Decay (x0.1)
        for g in self.agent.opt_inn.param_groups: g['lr'] *= 0.1
        for g in self.agent.opt_out.param_groups: g['lr'] *= 0.1
        for g in self.agent.opt_mon.param_groups: g['lr'] *= 0.1
        self.logger.info("   >>> Learning Rates reduced to 10%")

        # 最终阶段：我们要选出表现最好的模型，使用 save_mode='best'
        self._run_block('joint', joint, 'joint', save_mode='best')

        self.logger.info("🏆 Training Finished!")

    # ==============================================================================
    # 报告与测试辅助工具
    # ==============================================================================
    def _print_report(self, title, ret_stats, metrics):
        total_switches = ret_stats['switch_count']
        total_days = ret_stats['total_steps']
        switch_rate = (total_switches / total_days) * 100 if total_days > 0 else 0

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"       TEST REPORT: {title}       ")
        self.logger.info("=" * 60)
        self.logger.info(f"Days       : {total_days}")
        self.logger.info(f"Switches   : {total_switches} ({switch_rate:.2f}%)")
        self.logger.info(
            "Switch detail: free=%s, forced_h=%s, forced_s=%s",
            ret_stats.get('switch_free_count', 0),
            ret_stats.get('forced_hold_count', 0),
            ret_stats.get('forced_switch_count', 0),
        )
        self.logger.info("-" * 60)
        self.logger.info(f"Total Ret  : {metrics['total_ret'] * 100:.2f}%")
        self.logger.info(f"Ann Ret    : {metrics['ann_ret'] * 100:.2f}%")
        self.logger.info(f"Ann Vol    : {metrics['ann_vol'] * 100:.2f}%")
        self.logger.info(f"Sharpe     : {metrics['sharpe']:.4f}")
        self.logger.info(f"Max DD     : {metrics['max_dd'] * 100:.2f}%")
        self.logger.info("=" * 60 + "\n")

    def _test_episode_window(self):
        max_days = max(0, int(getattr(self.cfg, "test_max_days", 0) or 0))
        if max_days <= 0:
            return None
        test_indices = list(getattr(self.env, "idx_map", {}).get("test", []))
        if not test_indices:
            return None
        start = int(test_indices[0])
        full_stop = int(test_indices[-1])
        stop = min(start + max_days, full_stop)
        if stop <= start:
            return None
        return start, stop

    def test(self, model_path):
        """测试函数：加载最终最优模型并输出3种情况的详细报表"""
        self.logger.info("========== Testing ==========")
        # 优先加载 Joint 阶段保存的 Best 模型，如果没有则尝试 best_model.pth

        if not os.path.exists(model_path):
            run_model_path = os.path.join(self.model_dir, model_path)
            if os.path.exists(run_model_path):
                model_path = run_model_path
            else:
                model_path = os.path.join(self.model_dir, "best_exp_nm_no_monitor_fixed60.pth")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.agent.net.load_state_dict(checkpoint['agent_net'])
            self.logger.info(f"Loaded Best Model: {model_path}")
        else:
            self.logger.warning("No best model found, testing with current weights.")

        self.env.set_mode('test')
        fix_cycle = getattr(self.cfg, 'max_hold', 60)
        self.agent.net.eval()
        test_episode_window = self._test_episode_window()
        if test_episode_window is not None:
            self.logger.info(
                "Limiting test episode to %s days: [%s, %s]",
                test_episode_window[1] - test_episode_window[0],
                test_episode_window[0],
                test_episode_window[1],
            )
        if not bool(getattr(self.cfg, "test_skip_fixed_scenarios", False)):
            # Scenario 1: 不使用 monitor，固定 60 天
            self.logger.info(f"Running Scenario 1: No Monitor, Fixed {fix_cycle} Days...")
            ret_stats_1 = self.run_episode(
                self.env,
                mode='eval',
                phase='joint',
                fixed_cycle=fix_cycle,
                explicit_episode_window=test_episode_window,
            )
            metrics_1 = self._compute_metrics(ret_stats_1['history'])
            self._print_report(f"Scenario 1 (No Monitor, {fix_cycle}d)", ret_stats_1, metrics_1)
            pd.DataFrame(ret_stats_1['history'], columns=['value']).to_csv(
                os.path.join(self.run_dir, f"test_s1_Fixed{fix_cycle}d.csv"), index=False
            )

            # Scenario 2: 不使用 monitor 和 inner actor，固定 60 天
            self.logger.info(f"Running Scenario 2: No Monitor/Inner, Fixed {fix_cycle} Days...")
            ret_stats_2 = self.run_episode(
                self.env,
                mode='eval',
                phase='joint',
                fixed_cycle=fix_cycle,
                disable_inner=True,
                explicit_episode_window=test_episode_window,
            )
            metrics_2 = self._compute_metrics(ret_stats_2['history'])
            self._print_report(f"Scenario 2 (No Mon/Inner, {fix_cycle}d)", ret_stats_2, metrics_2)
            pd.DataFrame(ret_stats_2['history'], columns=['value']).to_csv(
                os.path.join(self.run_dir, f"test_s2_Fixed{fix_cycle}d_NoInner.csv"), index=False
            )
        else:
            self.logger.info("Skipping Scenario 1/2 fixed-cycle tests by request.")

        # Scenario 3: 三个模块都用 (Standard)
        self.logger.info("Running Scenario 3: All Modules (Standard)...")
        ret_stats_3 = self.run_episode(
            self.env,
            mode='eval',
            phase='joint',
            explicit_episode_window=test_episode_window,
        )
        metrics_3 = self._compute_metrics(ret_stats_3['history'])
        self._print_report("Scenario 3 (All Modules)", ret_stats_3, metrics_3)
        pd.DataFrame(ret_stats_3['history'], columns=['value']).to_csv(
            os.path.join(self.run_dir, "test_s3_AllModules.csv"), index=False
        )

        if bool(getattr(self.cfg, "test_controller_no_inner_scenario", False)):
            self.logger.info("Running Scenario 5: Controller + Outer, Inner bypassed...")
            ret_stats_5 = self.run_episode(
                self.env,
                mode='eval',
                phase='joint',
                disable_inner=True,
                explicit_episode_window=test_episode_window,
            )
            metrics_5 = self._compute_metrics(ret_stats_5['history'])
            self._print_report("Scenario 5 (Controller No Inner)", ret_stats_5, metrics_5)
            pd.DataFrame(ret_stats_5['history'], columns=['value']).to_csv(
                os.path.join(self.run_dir, "test_s5_ControllerNoInner.csv"), index=False
            )

        thresholds = getattr(self.cfg, "controller_test_thresholds", None)
        if thresholds:
            original_threshold = getattr(self.agent.net.mon, "eval_switch_threshold", 0.5)
            for threshold in thresholds:
                threshold = min(1.0, max(0.0, float(threshold)))
                self.agent.net.mon.eval_switch_threshold = threshold
                self.logger.info("Running Scenario 3 sweep: All Modules threshold=%.4f...", threshold)
                ret_stats_t = self.run_episode(
                    self.env,
                    mode='eval',
                    phase='joint',
                    explicit_episode_window=test_episode_window,
                )
                metrics_t = self._compute_metrics(ret_stats_t['history'])
                self._print_report(f"Scenario 3 (All Modules, th={threshold:.4f})", ret_stats_t, metrics_t)
                tag = str(threshold).replace(".", "p")
                pd.DataFrame(ret_stats_t['history'], columns=['value']).to_csv(
                    os.path.join(self.run_dir, f"test_s3_AllModules_th{tag}.csv"), index=False
                )
            self.agent.net.mon.eval_switch_threshold = original_threshold

        if not bool(getattr(self.cfg, "run_rule_switch_test", False)):
            self.agent.net.train()
            return

        # Scenario 4: Rule Based Switch (Multi-threshold)
        test_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,0.65,0.7]
        original_th = getattr(self.cfg, 'rule_switch_threshold', 0.3)

        self.logger.info(f"Running Scenario 4: Rule Based Switch (Thresholds: {test_thresholds})...")

        for th in test_thresholds:
            # 动态调整阈值
            self.cfg.rule_switch_threshold = th

            ret_stats_4 = self.run_episode(self.env, mode='eval', phase='joint', use_rule_switch=True)
            metrics_4 = self._compute_metrics(ret_stats_4['history'])

            label_str = f"Scenario 4 (Rule th={th})"
            self._print_report(label_str, ret_stats_4, metrics_4)

            filename = f"test_s4_RuleSwitch_th{th}.csv"
            pd.DataFrame(ret_stats_4['history'], columns=['value']).to_csv(
                os.path.join(self.run_dir, filename), index=False
            )

            # Plotting comparison for current threshold
            try:
                plt.figure(figsize=(10, 6))

                # 1. Outer Only (Scenario 2)
                if 'ret_stats_2' in locals():
                    hist2 = ret_stats_2['history']
                    val2 = np.array(hist2)
                    ret2 = (val2 - val2[0]) / (val2[0] + 1e-8)
                    plt.plot(ret2, label='Outer Only (Fixed)', linestyle='--', alpha=0.7)

                # 2. Inner + Outer (Scenario 1)
                if 'ret_stats_1' in locals():
                    hist1 = ret_stats_1['history']
                    val1 = np.array(hist1)
                    ret1 = (val1 - val1[0]) / (val1[0] + 1e-8)
                    plt.plot(ret1, label='Inner & Outer (Fixed)', linestyle=':', alpha=0.8)

                # 4. Rule Based (Scenario 4)
                hist4 = ret_stats_4['history']
                val4 = np.array(hist4)
                ret4 = (val4 - val4[0]) / (val4[0] + 1e-8)
                plt.plot(ret4, label=f'Inner & Outer & Rule Monitor (th={th})', linewidth=2)

                plt.title(f'Yield Curve Comparison')
                plt.xlabel('Days')
                plt.ylabel('Cumulative Return')
                plt.legend()
                plt.grid(True)

                plot_path = os.path.join(self.run_dir, f"comparison_yield_curve_th{th}.png")
                plt.savefig(plot_path)
                plt.close()
                self.logger.info(f"Saved yield curve comparison to {plot_path}")

            except Exception as e:
                self.logger.warning(f"Plotting failed for th={th}: {e}")

        # 恢复原始配置
        self.cfg.rule_switch_threshold = original_th

        self.agent.net.train()


def train_warmup_then_joint_with_monitor(trainer,
                                         warmup_outer_episodes: int = 50,
                                         warmup_inner_episodes: int = 50,
                                         warmup_monitor_episodes: int = 20,
                                         joint_episodes: int = 100,
                                         fixed_cycle: int = 60,
                                         val_interval: int = None,
                                         train_monitor: bool = False,
                                         use_rule_switch_train: bool = True,
                                         save_prefix: str = "cf_monitor"):
    import numpy as np
    import os

    if val_interval is None:
        val_interval = int(getattr(trainer.cfg, "val_interval", 10))

    # 强制设定：前 10 轮不验证
    MIN_WARMUP_STEPS = 6
    train_episodes_per_epoch = int(
        getattr(trainer.cfg, "train_episodes_per_epoch", getattr(trainer.cfg, "train_episode_count", 1))
    )
    train_episodes_per_epoch = max(1, train_episodes_per_epoch)
    inner_train_fixed_episodes = bool(getattr(trainer.cfg, "inner_train_fixed_episodes", False))
    inner_train_episodes_per_epoch = max(
        1,
        int(getattr(trainer.cfg, "inner_train_episodes_per_epoch", train_episodes_per_epoch)),
    )
    inner_episode_batch_size = max(1, int(getattr(trainer.cfg, "inner_episode_batch_size", 1)))
    configured_inner_epochs = max(0, int(getattr(
        trainer.cfg,
        "warmup_inner_epochs",
        warmup_inner_episodes // max(1, inner_train_episodes_per_epoch),
    )))
    train_monitor = (
        bool(getattr(trainer.cfg, "train_monitor_enabled"))
        if hasattr(trainer.cfg, "train_monitor_enabled")
        else bool(train_monitor)
    )
    use_rule_switch_train = bool(getattr(trainer.cfg, "use_rule_switch_train", use_rule_switch_train))
    if not train_monitor:
        warmup_monitor_episodes = 0
    def _rollout_update_steps_for(stage_name: str) -> int:
        if stage_name == "warmup_inner":
            inner_steps = int(getattr(trainer.cfg, "inner_rollout_update_steps", 0) or 0)
            if inner_steps > 0:
                return inner_steps
        by_stage = getattr(trainer.cfg, "rollout_update_steps_by_stage", None)
        if isinstance(by_stage, dict) and stage_name in by_stage:
            return int(by_stage[stage_name])
        segments = getattr(trainer.cfg, f"{stage_name}_rollout_segments", None)
        if segments is not None:
            return int(fixed_cycle) * int(segments)
        return int(getattr(trainer.cfg, "rollout_update_steps", 0) or 0)

    def _episodes_per_epoch_for(stage_name: str) -> int:
        if stage_name == "warmup_inner" and inner_train_fixed_episodes:
            return inner_train_episodes_per_epoch
        return train_episodes_per_epoch

    for stage_name, episode_total in (
            ("warmup_outer", warmup_outer_episodes),
            ("warmup_inner", warmup_inner_episodes),
            ("warmup_monitor", warmup_monitor_episodes),
            ("joint", joint_episodes)):
        stage_episodes_per_epoch = _episodes_per_epoch_for(stage_name)
        if episode_total % stage_episodes_per_epoch != 0:
            raise ValueError(
                f"{stage_name} has {episode_total} train episodes, which is not a complete number of "
                f"train epochs with episodes_per_epoch={stage_episodes_per_epoch}."
            )

    def _epoch_progress(stage_name: str, ep: int) -> str:
        stage_episodes_per_epoch = _episodes_per_epoch_for(stage_name)
        return (
            f"epoch {ep // stage_episodes_per_epoch + 1}, "
            f"slot {ep % stage_episodes_per_epoch + 1}/{stage_episodes_per_epoch}"
        )

    # 文件名定义
    final_best_ckpt = f"best_model.pth"
    final_last_ckpt = f"last_model.pth"
    warmup_out_best_ckpt = f"temp_warmup_outer.pth"
    warmup_inn_best_ckpt = f"temp_warmup_inner.pth"
    hrl_fixed_best_ckpt = f"hrl_fixed_best.pth"
    controller_best_ckpt = f"controller_best.pth"

    def _do_validate(tag: str, *, use_controller: bool = False, disable_inner: bool = False):
        trainer.env.set_mode("val")
        trainer.agent.net.eval()
        val_ret = trainer.run_episode(
            trainer.env,
            mode="eval",
            phase="joint",
            fixed_cycle=None if use_controller else fixed_cycle,
            disable_inner=disable_inner,
            use_rule_switch=False,
        )
        val_metrics = trainer._compute_metrics(val_ret["history"])
        trainer.env.set_mode("train")
        trainer.agent.net.train()
        trainer.logger.info(
            f"   >>> [VAL {tag}] Sharpe:{val_metrics['sharpe']:.4f} | Ret:{val_metrics['total_ret'] * 100:.2f}%")
        return val_metrics

    # =========================================================
    # 1) Warmup Outer
    # =========================================================
    trainer.logger.info(
        f"### [1/4] Warmup OUTER: {warmup_outer_episodes // train_episodes_per_epoch} epochs "
        f"/ {warmup_outer_episodes} eps, update_every={_rollout_update_steps_for('warmup_outer')} steps "
        f"(Min Warmup: {MIN_WARMUP_STEPS}) ###")
    trainer.agent.set_module_status("outer")
    phase_best_sharpe = -np.inf

    for ep in range(warmup_outer_episodes):
        # 每个 Episode 更新一次
        ret = trainer.run_episode(
            trainer.env,
            mode="train",
            phase="warmup_outer",
            fixed_cycle=fixed_cycle,
            use_rule_switch=False,
            rollout_update_steps=_rollout_update_steps_for("warmup_outer"),
            auto_update_phase="warmup_outer",
            train_monitor=train_monitor,
        )
        loss_log = ret.get("loss_log", {})

        trainer.logger.info(
            f"[{save_prefix}] W-Out {ep + 1}/{warmup_outer_episodes} ({_epoch_progress('warmup_outer', ep)}) | "
            f"Updates:{ret.get('update_count', 0)} "
            f"L_out:{loss_log.get('out_pi', 0):.3f} "
            f"L_out_pred:{loss_log.get('out_pred', 0):.3f}")

        # 前 10 轮跳过，之后每 2 轮验证一次
        if (ep + 1) > MIN_WARMUP_STEPS and (ep + 1) % val_interval == 0:
            m = _do_validate("warmup_outer", disable_inner=True)
            score = trainer._validation_score(m, trainer.cfg, phase="warmup_outer")
            if score > phase_best_sharpe:
                phase_best_sharpe = score
                trainer.save_model(warmup_out_best_ckpt)
                trainer.logger.info(
                    "       (New Warmup-Outer Best by %s: %.4f)",
                    getattr(trainer.cfg, "model_selection_metric", "sharpe"),
                    phase_best_sharpe,
                )

    # 预热结束，回滚至该阶段最优，为 Inner 预热提供更好的基准
    if os.path.exists(os.path.join(trainer.model_dir, warmup_out_best_ckpt)):
        trainer._load_model(warmup_out_best_ckpt)
        trainer.logger.info(f"   ↺ Phase Outer finished. Loaded best warmup_outer model.")

    # =========================================================
    # 2) Warmup Inner
    # =========================================================
    inner_previous_train_config = None
    try:
        if inner_train_fixed_episodes:
            inner_previous_train_config = trainer._apply_train_episode_config(
                train_episode_to_end=False,
                train_episodes_per_epoch=inner_train_episodes_per_epoch,
                train_start_stride_days=int(getattr(trainer.cfg, "inner_train_start_stride_days", fixed_cycle * 3)),
                episode_len=int(getattr(trainer.cfg, "inner_episode_len", fixed_cycle * 10)),
            )
            trainer.logger.info(
                "   [Inner Schedule] fixed windows: episodes_per_epoch=%s, episode_len=%s, "
                "start_stride_days=%s, update_every=%s steps",
                trainer.env.train_episodes_per_epoch,
                trainer.env.episode_len,
                trainer.env.train_start_stride_days,
                _rollout_update_steps_for("warmup_inner"),
            )

        trainer.logger.info(
            f"### [2/4] Warmup INNER: {warmup_inner_episodes // _episodes_per_epoch_for('warmup_inner')} epochs "
            f"/ {warmup_inner_episodes} eps, update_every={_rollout_update_steps_for('warmup_inner')} steps "
            f"(Min Warmup: {MIN_WARMUP_STEPS}) ###")
        trainer.agent.set_module_status("inner")
        phase_best_sharpe = -np.inf

        if inner_train_fixed_episodes:
            pool_size = len(getattr(trainer.env, "train_indices_pool", []))
            if pool_size <= 0:
                raise RuntimeError("Inner episode-batch training requires a non-empty fixed train episode pool.")
            inner_epochs = max(1, configured_inner_epochs)
            trainer.logger.info(
                "   [Inner Batch] fixed episode pool=%s, episode_batch_size=%s, parallel_workers=%s, epochs=%s",
                pool_size,
                inner_episode_batch_size,
                max(1, int(getattr(trainer.cfg, "inner_episode_parallel_workers", 1))),
                inner_epochs,
            )
            for epoch in range(inner_epochs):
                if epoch > 0:
                    trainer.env.train_ptr = 0
                    np.random.shuffle(trainer.env.train_indices_pool)

                batch_id = 0
                epoch_loss = {}
                epoch_updates = 0
                epoch_episodes = 0
                for start in range(0, pool_size, inner_episode_batch_size):
                    episode_count = min(inner_episode_batch_size, pool_size - start)
                    ret = trainer._run_inner_episode_batch(
                        episode_count=episode_count,
                        fixed_cycle=fixed_cycle,
                        train_monitor=train_monitor,
                    )
                    batch_id += 1
                    epoch_updates += int(ret.get("update_count", 0))
                    epoch_episodes += int(ret.get("episodes", 0))
                    for k, v in ret.get("loss_log", {}).items():
                        epoch_loss.setdefault(k, []).append(float(v))
                    loss_log = {
                        k: float(np.mean(v)) if len(v) > 0 else 0.0
                        for k, v in epoch_loss.items()
                    }
                    trainer.logger.info(
                        f"[{save_prefix}] W-Inn epoch {epoch + 1}/{inner_epochs} "
                        f"batch {batch_id}/{int(np.ceil(pool_size / inner_episode_batch_size))} | "
                        f"episodes:{epoch_episodes}/{pool_size} updates:{epoch_updates} "
                        f"L_in:{loss_log.get('inn_pi', 0):.3f} "
                        f"L_in_pred:{loss_log.get('inn_pred', 0):.3f}")

                m = _do_validate("warmup_inner")
                score = trainer._validation_score(m, trainer.cfg, phase="warmup_inner")
                if score > phase_best_sharpe:
                    phase_best_sharpe = score
                    trainer.save_model(warmup_inn_best_ckpt)
                    trainer.logger.info(
                        "       (New Warmup-Inner Best by %s: %.4f)",
                        getattr(trainer.cfg, "inner_selection_metric", "return"),
                        phase_best_sharpe,
                    )
        else:
            for ep in range(warmup_inner_episodes):
                ret = trainer.run_episode(
                    trainer.env,
                    mode="train",
                    phase="warmup_inner",
                    fixed_cycle=fixed_cycle,
                    use_rule_switch=False,
                    rollout_update_steps=_rollout_update_steps_for("warmup_inner"),
                    auto_update_phase="warmup_inner",
                    train_monitor=train_monitor,
                )
                loss_log = ret.get("loss_log", {})

                trainer.logger.info(
                    f"[{save_prefix}] W-Inn {ep + 1}/{warmup_inner_episodes} ({_epoch_progress('warmup_inner', ep)}) | "
                    f"Updates:{ret.get('update_count', 0)} L_in:{loss_log.get('inn_pi', 0):.3f} "
                    f"L_in_pred:{loss_log.get('inn_pred', 0):.3f}")

                if (ep + 1) > MIN_WARMUP_STEPS and (ep + 1) % val_interval == 0:
                    m = _do_validate("warmup_inner")
                    score = trainer._validation_score(m, trainer.cfg, phase="warmup_inner")
                    if score > phase_best_sharpe:
                        phase_best_sharpe = score
                        trainer.save_model(warmup_inn_best_ckpt)
                        trainer.logger.info(
                            "       (New Warmup-Inner Best by %s: %.4f)",
                            getattr(trainer.cfg, "inner_selection_metric", "return"),
                            phase_best_sharpe,
                        )

        # 回滚至该阶段最优，进入 Monitor 预热
        if os.path.exists(os.path.join(trainer.model_dir, warmup_inn_best_ckpt)):
            trainer._load_model(warmup_inn_best_ckpt)
            trainer.logger.info(f"   ↺ Phase Inner finished. Loaded best warmup_inner model.")
    finally:
        if inner_previous_train_config is not None:
            trainer._restore_train_episode_config(inner_previous_train_config)
            trainer.logger.info("   [Inner Schedule] restored global train episode schedule.")

    # =========================================================
    # 3) Outer + Inner Joint Finetune (controller disabled)
    # =========================================================
    trainer.logger.info(
        f"### [3/4] JOINT OUTER+INNER: {joint_episodes // train_episodes_per_epoch} epochs "
        f"/ {joint_episodes} eps, update_every={_rollout_update_steps_for('joint')} steps ###")
    trainer.agent.set_module_status("outer_inner")
    joint_lr_mult = float(getattr(trainer.cfg, "joint_lr_mult", 1.0))
    if hasattr(trainer.agent, "set_lr_multiplier"):
        joint_lrs = trainer.agent.set_lr_multiplier(joint_lr_mult)
        trainer.logger.info(
            "Outer+Inner joint lr multiplier=%s -> monitor=%g outer=%g inner=%g",
            joint_lr_mult,
            joint_lrs.get("monitor", 0.0),
            joint_lrs.get("outer", 0.0),
            joint_lrs.get("inner", 0.0),
        )
    global_best_sharpe = -np.inf
    joint_single_full_episode = bool(getattr(trainer.cfg, "joint_single_full_episode", False))
    joint_loop_count = (
        max(1, int(getattr(trainer.cfg, "joint_epochs", joint_episodes // train_episodes_per_epoch)))
        if joint_single_full_episode
        else joint_episodes
    )
    if joint_single_full_episode:
        trainer.logger.info(
            "   [Joint Schedule] single full train-to-end episode per joint epoch, epochs=%s",
            joint_loop_count,
        )

    for ep in range(joint_loop_count):
        if joint_single_full_episode:
            ret = trainer._run_joint_full_train_episode(
                fixed_cycle=fixed_cycle,
                rollout_update_steps=_rollout_update_steps_for("joint"),
                train_monitor=False,
            )
        else:
            ret = trainer.run_episode(
                trainer.env,
                mode="train",
                phase="joint",
                fixed_cycle=fixed_cycle,
                use_rule_switch=False,
                rollout_update_steps=_rollout_update_steps_for("joint"),
                auto_update_phase="joint",
                train_monitor=False,
            )
        loss = ret.get("loss_log", {})

        trainer.logger.info(
            f"[{save_prefix}] OI-Joint {ep + 1}/{joint_loop_count} "
            f"({'full-train' if joint_single_full_episode else _epoch_progress('joint', ep)}) | "
            f"Updates:{ret.get('update_count', 0)} "
            f"L_out:{loss.get('out_pi', 0):.3f} L_in:{loss.get('inn_pi', 0):.3f} "
            f"L_out_pred:{loss.get('out_pred', 0):.3f} "
            f"L_in_pred:{loss.get('inn_pred', 0):.3f}")
        # 联合训练默认按 val_interval 验证，避免每个长 episode 都完整跑验证集。
        if (ep + 1) % val_interval == 0 or (ep + 1) == joint_loop_count:
            m = _do_validate("outer_inner_joint", use_controller=False)
            score = trainer._validation_score(m, trainer.cfg, phase="joint")
            if score > global_best_sharpe:
                global_best_sharpe = score
                trainer.save_model(hrl_fixed_best_ckpt)
                trainer.logger.info(
                    "       (New Fixed-HRL Best Saved by %s: %.4f)",
                    getattr(trainer.cfg, "model_selection_metric", "sharpe"),
                    global_best_sharpe,
                )
    if os.path.exists(os.path.join(trainer.model_dir, hrl_fixed_best_ckpt)):
        trainer._load_model(hrl_fixed_best_ckpt)
        trainer.logger.info("   ↺ Loaded best fixed HRL before controller training.")
    else:
        trainer.save_model(hrl_fixed_best_ckpt)

    if hasattr(trainer.agent, "set_lr_multiplier"):
        trainer.agent.set_lr_multiplier(1.0)

    # =========================================================
    # 4) Controller Counterfactual Policy Gradient
    # =========================================================
    controller_pg_result = {"best_score": 0.0, "updates": 0}
    controller_epochs = warmup_monitor_episodes // train_episodes_per_epoch if train_episodes_per_epoch > 0 else 0
    end_to_end_controller_joint = bool(getattr(trainer.cfg, "end_to_end_controller_joint", False))
    controller_joint_epochs = max(0, int(getattr(trainer.cfg, "controller_joint_epochs", 0) or 0))
    run_controller_joint = end_to_end_controller_joint and controller_joint_epochs > 0
    controller_pg_save_name = controller_best_ckpt if run_controller_joint else final_best_ckpt
    if train_monitor and controller_epochs > 0:
        trainer.logger.info(
            f"### [4/4] CONTROLLER Counterfactual PG: {controller_epochs} epochs, "
            f"rollout_len={getattr(trainer.cfg, 'controller_rollout_len', fixed_cycle * 10)}, "
            f"batch_windows={getattr(trainer.cfg, 'controller_pg_batch_windows', 4)}, "
            f"max_switches={trainer._controller_max_allowed_switches(int(getattr(trainer.cfg, 'controller_rollout_len', fixed_cycle * 10)), int(getattr(trainer.cfg, 'min_hold', 10)))} ###")
        controller_val_interval = int(getattr(trainer.cfg, "controller_val_interval_epochs", val_interval))
        controller_pg_result = trainer.train_controller_counterfactual_pg(
            epochs=controller_epochs,
            fixed_cycle=fixed_cycle,
            val_interval=max(1, controller_val_interval),
            save_name=controller_pg_save_name,
        )
    else:
        trainer.logger.info("### [4/4] CONTROLLER Counterfactual PG: skipped ###")
        trainer.save_model(final_best_ckpt)

    controller_joint_result = {
        "enabled": bool(run_controller_joint),
        "best_score": None,
        "episodes": 0,
        "controller_active": False,
        "selection_metric": str(getattr(trainer.cfg, "controller_selection_metric", "risk_return")),
    }
    if run_controller_joint:
        best_controller_joint_score = _prepare_controller_joint_baseline(
            trainer,
            controller_best_ckpt=controller_best_ckpt,
            final_best_ckpt=final_best_ckpt,
            controller_pg_result=controller_pg_result,
        )

        trainer.logger.info(
            "### [5/5] END-TO-END CONTROLLER+HRL JOINT FINETUNE: %s epochs, controller active, update_every=%s steps ###",
            controller_joint_epochs,
            _rollout_update_steps_for("joint"),
        )
        trainer.agent.set_module_status("all")
        joint_lr_mult = float(getattr(trainer.cfg, "joint_lr_mult", 1.0))
        if hasattr(trainer.agent, "set_lr_multiplier"):
            joint_lrs = trainer.agent.set_lr_multiplier(joint_lr_mult)
            trainer.logger.info(
                "End-to-end controller joint lr multiplier=%s -> monitor=%g outer=%g inner=%g",
                joint_lr_mult,
                joint_lrs.get("monitor", 0.0),
                joint_lrs.get("outer", 0.0),
                joint_lrs.get("inner", 0.0),
        )

        joint_update_steps = _rollout_update_steps_for("joint")
        disable_inner_joint = bool(getattr(trainer.cfg, "controller_pg_disable_inner", False))
        for ep in range(controller_joint_epochs):
            if joint_single_full_episode:
                previous = trainer._apply_train_episode_config(
                    train_episode_to_end=True,
                    train_episodes_per_epoch=1,
                    train_start_stride_days=1,
                    episode_len=getattr(trainer.env, "episode_len", max(1, int(fixed_cycle))),
                )
                try:
                    ret = trainer.run_episode(
                        trainer.env,
                        mode="train",
                        phase="joint",
                        fixed_cycle=None,
                        use_rule_switch=False,
                        rollout_update_steps=joint_update_steps,
                        auto_update_phase="joint",
                        train_monitor=train_monitor,
                        disable_inner=disable_inner_joint,
                    )
                finally:
                    trainer._restore_train_episode_config(previous)
            else:
                ret = trainer.run_episode(
                    trainer.env,
                    mode="train",
                    phase="joint",
                    fixed_cycle=None,
                    use_rule_switch=False,
                    rollout_update_steps=joint_update_steps,
                    auto_update_phase="joint",
                    train_monitor=train_monitor,
                    disable_inner=disable_inner_joint,
                )
            loss = ret.get("loss_log", {})
            trainer.logger.info(
                "[%s] E2E-Controller-Joint %s/%s | Updates:%s L_mon:%0.3f L_out:%0.3f L_in:%0.3f",
                save_prefix,
                ep + 1,
                controller_joint_epochs,
                ret.get("update_count", 0),
                loss.get("mon_pi", 0.0),
                loss.get("out_pi", 0.0),
                loss.get("inn_pi", 0.0),
            )

            if (ep + 1) % max(1, int(val_interval)) == 0 or (ep + 1) == controller_joint_epochs:
                trainer.env.set_mode("val")
                trainer.agent.net.eval()
                val_ret = trainer.run_episode(
                    trainer.env,
                    mode="eval",
                    phase="joint",
                    fixed_cycle=None,
                    use_rule_switch=False,
                    disable_inner=disable_inner_joint,
                )
                metrics = trainer._compute_metrics(val_ret["history"])
                score = trainer._validation_score(metrics, trainer.cfg, phase="controller")
                trainer.env.set_mode("train")
                trainer.agent.net.train()
                trainer.logger.info(
                    "   >>> [VAL e2e_controller_joint ep=%s] select=%s score=%.4f Sharpe=%.4f Ret=%.2f%% MDD=%.2f%% switches=%s",
                    ep + 1,
                    getattr(trainer.cfg, "controller_selection_metric", "risk_return"),
                    score,
                    metrics["sharpe"],
                    metrics["total_ret"] * 100.0,
                    metrics["max_dd"] * 100.0,
                    val_ret.get("switch_count", 0),
                )
                if score > best_controller_joint_score:
                    best_controller_joint_score = score
                    trainer.save_model(final_best_ckpt)
                    trainer.logger.info("       (New End-to-End Controller Joint Best: %.4f)", score)

        if os.path.exists(os.path.join(trainer.model_dir, final_best_ckpt)):
            trainer._load_model(final_best_ckpt)
            trainer.logger.info("   ↺ End-to-end controller joint finished. Loaded best final model.")
        else:
            trainer.save_model(final_best_ckpt)
        if hasattr(trainer.agent, "set_lr_multiplier"):
            trainer.agent.set_lr_multiplier(1.0)
        controller_joint_result = {
            "enabled": True,
            "best_score": float(best_controller_joint_score),
            "episodes": int(controller_joint_epochs),
            "controller_active": True,
            "selection_metric": str(getattr(trainer.cfg, "controller_selection_metric", "risk_return")),
        }

    trainer.save_model(final_last_ckpt)
    final_best_score = (
        controller_joint_result["best_score"]
        if controller_joint_result["best_score"] is not None
        else float(global_best_sharpe)
    )
    return {
        "fixed_cycle": fixed_cycle,
        "best_sharpe": float(global_best_sharpe),
        "best_selection_score": float(final_best_score),
        "model_selection_metric": str(
            getattr(
                trainer.cfg,
                "controller_selection_metric" if run_controller_joint else "model_selection_metric",
                "risk_return" if run_controller_joint else "sharpe",
            )
        ),
        "best_ckpt": final_best_ckpt,
        "hrl_fixed_ckpt": hrl_fixed_best_ckpt,
        "controller_best_ckpt": controller_best_ckpt if run_controller_joint else None,
        "controller_pg": controller_pg_result,
        "controller_joint_finetune": controller_joint_result,
    }


def train_controller_then_joint_finetune(trainer,
                                         controller_episodes: int,
                                         joint_episodes: int,
                                         fixed_cycle: int,
                                         val_interval: int = None,
                                         train_monitor: bool = True,
                                         save_prefix: str = "controller_first"):
    """Train controller on frozen Outer/Inner, then jointly finetune all modules.

    This schedule is intended for runs that call load_frozen_hrl_checkpoint()
    before entry. Joint finetune keeps fixed_cycle=None so controller decisions
    are active during the final phase.
    """
    import numpy as np
    import os

    if val_interval is None:
        val_interval = int(getattr(trainer.cfg, "val_interval", 10))
    train_episodes_per_epoch = max(
        1,
        int(getattr(trainer.cfg, "train_episodes_per_epoch", getattr(trainer.cfg, "train_episode_count", 1))),
    )
    controller_epochs = int(controller_episodes) // train_episodes_per_epoch
    if int(controller_episodes) % train_episodes_per_epoch != 0:
        raise ValueError(
            "controller_episodes must be a complete number of train epochs with "
            f"episodes_per_epoch={train_episodes_per_epoch}."
        )

    final_best_ckpt = "best_model.pth"
    final_last_ckpt = "last_model.pth"
    hrl_fixed_best_ckpt = "hrl_fixed_best.pth"
    controller_best_ckpt = "controller_best.pth"

    trainer.save_model(hrl_fixed_best_ckpt)
    trainer.logger.info("### [1/2] CONTROLLER-FIRST: saved loaded fixed HRL as %s ###", hrl_fixed_best_ckpt)

    controller_pg_result = {"best_score": 0.0, "updates": 0}
    if train_monitor and controller_epochs > 0:
        controller_rollout_len = int(getattr(trainer.cfg, "controller_rollout_len", fixed_cycle * 10))
        trainer.logger.info(
            "### [1/2] CONTROLLER-FIRST Counterfactual PG: %s epochs, rollout_len=%s, "
            "batch_windows=%s, max_switches=%s ###",
            controller_epochs,
            controller_rollout_len,
            getattr(trainer.cfg, "controller_pg_batch_windows", 4),
            trainer._controller_max_allowed_switches(
                controller_rollout_len,
                int(getattr(trainer.cfg, "min_hold", 10)),
            ),
        )
        controller_val_interval = int(getattr(trainer.cfg, "controller_val_interval_epochs", val_interval))
        controller_pg_result = trainer.train_controller_counterfactual_pg(
            epochs=controller_epochs,
            fixed_cycle=fixed_cycle,
            val_interval=max(1, controller_val_interval),
            save_name=controller_best_ckpt,
        )
        if os.path.exists(os.path.join(trainer.model_dir, controller_best_ckpt)):
            trainer._load_model(controller_best_ckpt)
            trainer.logger.info("   ↺ Loaded best controller before joint finetune.")
    else:
        trainer.logger.info("### [1/2] CONTROLLER-FIRST Counterfactual PG: skipped ###")

    trainer.logger.info(
        "### [2/2] CONTROLLER-FIRST JOINT FINETUNE: %s episodes, controller active, update_every=%s steps ###",
        joint_episodes,
        int(getattr(trainer.cfg, "rollout_update_steps_by_stage", {}).get(
            "joint",
            int(getattr(trainer.cfg, "rollout_update_steps", 0) or 0),
        )),
    )
    trainer.agent.set_module_status("all")
    joint_lr_mult = float(getattr(trainer.cfg, "joint_lr_mult", 1.0))
    if hasattr(trainer.agent, "set_lr_multiplier"):
        joint_lrs = trainer.agent.set_lr_multiplier(joint_lr_mult)
        trainer.logger.info(
            "Controller-first joint lr multiplier=%s -> monitor=%g outer=%g inner=%g",
            joint_lr_mult,
            joint_lrs.get("monitor", 0.0),
            joint_lrs.get("outer", 0.0),
            joint_lrs.get("inner", 0.0),
        )

    def _validate_joint(epoch_idx: int):
        trainer.env.set_mode("val")
        trainer.agent.net.eval()
        disable_inner = bool(getattr(trainer.cfg, "controller_pg_disable_inner", False))
        ret_stats = trainer.run_episode(
            trainer.env,
            mode="eval",
            phase="joint",
            fixed_cycle=None,
            use_rule_switch=False,
            disable_inner=disable_inner,
        )
        metrics = trainer._compute_metrics(ret_stats["history"])
        score = trainer._validation_score(metrics, trainer.cfg, phase="controller")
        trainer.env.set_mode("train")
        trainer.agent.net.train()
        trainer.logger.info(
            "   >>> [VAL controller_first_joint ep=%s] select=%s score=%.4f Sharpe=%.4f Ret=%.2f%% MDD=%.2f%% switches=%s",
            epoch_idx,
            getattr(trainer.cfg, "controller_selection_metric", "risk_return"),
            score,
            metrics["sharpe"],
            metrics["total_ret"] * 100.0,
            metrics["max_dd"] * 100.0,
            ret_stats.get("switch_count", 0),
        )
        return score, metrics

    joint_single_full_episode = bool(getattr(trainer.cfg, "joint_single_full_episode", False))
    joint_loop_count = (
        max(1, int(getattr(trainer.cfg, "joint_epochs", max(1, joint_episodes // train_episodes_per_epoch))))
        if joint_single_full_episode
        else int(joint_episodes)
    )
    joint_update_steps = int(getattr(trainer.cfg, "rollout_update_steps_by_stage", {}).get(
        "joint",
        int(getattr(trainer.cfg, "rollout_update_steps", 0) or 0),
    ))
    best_score = _prepare_controller_joint_baseline(
        trainer,
        controller_best_ckpt=controller_best_ckpt,
        final_best_ckpt=final_best_ckpt,
        controller_pg_result=controller_pg_result,
    )
    disable_inner_joint = bool(getattr(trainer.cfg, "controller_pg_disable_inner", False))

    for ep in range(max(0, joint_loop_count)):
        if joint_single_full_episode:
            previous = trainer._apply_train_episode_config(
                train_episode_to_end=True,
                train_episodes_per_epoch=1,
                train_start_stride_days=1,
                episode_len=getattr(trainer.env, "episode_len", max(1, int(fixed_cycle))),
            )
            try:
                ret = trainer.run_episode(
                    trainer.env,
                    mode="train",
                    phase="joint",
                    fixed_cycle=None,
                    use_rule_switch=False,
                    rollout_update_steps=joint_update_steps,
                    auto_update_phase="joint",
                    train_monitor=train_monitor,
                    disable_inner=disable_inner_joint,
                )
            finally:
                trainer._restore_train_episode_config(previous)
        else:
            ret = trainer.run_episode(
                trainer.env,
                mode="train",
                phase="joint",
                fixed_cycle=None,
                use_rule_switch=False,
                rollout_update_steps=joint_update_steps,
                auto_update_phase="joint",
                train_monitor=train_monitor,
                disable_inner=disable_inner_joint,
            )
        loss = ret.get("loss_log", {})
        trainer.logger.info(
            "[%s] Controller-Joint %s/%s | Updates:%s L_mon:%0.3f L_out:%0.3f L_in:%0.3f",
            save_prefix,
            ep + 1,
            joint_loop_count,
            ret.get("update_count", 0),
            loss.get("mon_pi", 0.0),
            loss.get("out_pi", 0.0),
            loss.get("inn_pi", 0.0),
        )

        if (ep + 1) % max(1, int(val_interval)) == 0 or (ep + 1) == joint_loop_count:
            score, _ = _validate_joint(ep + 1)
            if score > best_score:
                best_score = score
                trainer.save_model(final_best_ckpt)
                trainer.logger.info("       (New Controller-First Joint Best: %.4f)", best_score)

    if os.path.exists(os.path.join(trainer.model_dir, final_best_ckpt)):
        trainer._load_model(final_best_ckpt)
        trainer.logger.info("   ↺ Controller-first joint finished. Loaded best final model.")
    else:
        trainer.save_model(final_best_ckpt)
    trainer.save_model(final_last_ckpt)
    if hasattr(trainer.agent, "set_lr_multiplier"):
        trainer.agent.set_lr_multiplier(1.0)

    return {
        "fixed_cycle": fixed_cycle,
        "best_sharpe": float(best_score),
        "best_selection_score": float(best_score),
        "model_selection_metric": str(getattr(trainer.cfg, "controller_selection_metric", "risk_return")),
        "best_ckpt": final_best_ckpt,
        "hrl_fixed_ckpt": hrl_fixed_best_ckpt,
        "controller_best_ckpt": controller_best_ckpt,
        "controller_pg": controller_pg_result,
        "joint_finetune": {
            "best_score": float(best_score),
            "episodes": int(joint_loop_count),
            "controller_active": True,
            "selection_metric": str(getattr(trainer.cfg, "controller_selection_metric", "risk_return")),
        },
    }


def train_controller_only_from_frozen(trainer,
                                      controller_episodes: int,
                                      fixed_cycle: int,
                                      val_interval: int = None,
                                      train_monitor: bool = True,
                                      save_prefix: str = "controller_only"):
    """Train only the controller on top of an already-loaded frozen HRL model."""
    import os

    if val_interval is None:
        val_interval = int(getattr(trainer.cfg, "val_interval", 10))
    train_episodes_per_epoch = max(
        1,
        int(getattr(trainer.cfg, "train_episodes_per_epoch", getattr(trainer.cfg, "train_episode_count", 1))),
    )
    controller_epochs = int(controller_episodes) // train_episodes_per_epoch
    if int(controller_episodes) % train_episodes_per_epoch != 0:
        raise ValueError(
            "controller_episodes must be a complete number of train epochs with "
            f"episodes_per_epoch={train_episodes_per_epoch}."
        )

    final_best_ckpt = "best_model.pth"
    final_last_ckpt = "last_model.pth"
    hrl_fixed_best_ckpt = "hrl_fixed_best.pth"
    controller_best_ckpt = "controller_best.pth"

    trainer.save_model(hrl_fixed_best_ckpt)
    trainer.logger.info("### CONTROLLER-ONLY: saved loaded fixed HRL as %s ###", hrl_fixed_best_ckpt)

    controller_pg_result = {"best_score": 0.0, "updates": 0}
    if train_monitor and controller_epochs > 0:
        controller_rollout_len = int(getattr(trainer.cfg, "controller_rollout_len", fixed_cycle * 10))
        trainer.logger.info(
            "### CONTROLLER-ONLY Counterfactual PG: %s epochs, rollout_len=%s, "
            "batch_windows=%s, max_switches=%s ###",
            controller_epochs,
            controller_rollout_len,
            getattr(trainer.cfg, "controller_pg_batch_windows", 4),
            trainer._controller_max_allowed_switches(
                controller_rollout_len,
                int(getattr(trainer.cfg, "min_hold", 10)),
            ),
        )
        controller_val_interval = int(getattr(trainer.cfg, "controller_val_interval_epochs", val_interval))
        controller_pg_result = trainer.train_controller_counterfactual_pg(
            epochs=controller_epochs,
            fixed_cycle=fixed_cycle,
            val_interval=max(1, controller_val_interval),
            save_name=controller_best_ckpt,
        )
        if os.path.exists(os.path.join(trainer.model_dir, controller_best_ckpt)):
            trainer._load_model(controller_best_ckpt)
            trainer.logger.info("   ↺ Loaded best controller for controller-only final model.")
    else:
        trainer.logger.info("### CONTROLLER-ONLY Counterfactual PG: skipped ###")

    trainer.save_model(final_best_ckpt)
    trainer.save_model(final_last_ckpt)
    trainer.logger.info("   ↺ Controller-only finished. Saved final controller model as %s.", final_best_ckpt)

    return {
        "fixed_cycle": fixed_cycle,
        "best_sharpe": float(controller_pg_result.get("best_score", 0.0)),
        "best_selection_score": float(controller_pg_result.get("best_score", 0.0)),
        "model_selection_metric": str(getattr(trainer.cfg, "controller_selection_metric", "risk_return")),
        "best_ckpt": final_best_ckpt,
        "hrl_fixed_ckpt": hrl_fixed_best_ckpt,
        "controller_best_ckpt": controller_best_ckpt,
        "controller_pg": controller_pg_result,
        "controller_only": {
            "episodes": int(controller_epochs * train_episodes_per_epoch),
            "controller_active": True,
            "outer_inner_frozen": True,
            "selection_metric": str(getattr(trainer.cfg, "controller_selection_metric", "risk_return")),
        },
    }


def train_inner_only_from_frozen(trainer,
                                 inner_episodes: int,
                                 fixed_cycle: int,
                                 val_interval: int = None,
                                 train_monitor: bool = True,
                                 save_prefix: str = "inner_only"):
    """Train only Inner while using already-loaded frozen Outer and Controller."""
    import os
    import numpy as np

    if val_interval is None:
        val_interval = int(getattr(trainer.cfg, "val_interval", 10))
    train_episodes_per_epoch = max(
        1,
        int(getattr(trainer.cfg, "train_episodes_per_epoch", getattr(trainer.cfg, "train_episode_count", 1))),
    )
    if int(inner_episodes) % train_episodes_per_epoch != 0:
        raise ValueError(
            "inner_episodes must be a complete number of train epochs with "
            f"episodes_per_epoch={train_episodes_per_epoch}."
        )

    final_best_ckpt = "best_model.pth"
    final_last_ckpt = "last_model.pth"
    frozen_controller_ckpt = "controller_frozen_input.pth"
    trainer.save_model(frozen_controller_ckpt)
    trainer.agent.set_module_status("inner")

    def _rollout_update_steps_for_inner() -> int:
        inner_steps = int(getattr(trainer.cfg, "inner_rollout_update_steps", 0) or 0)
        if inner_steps > 0:
            return inner_steps
        by_stage = getattr(trainer.cfg, "rollout_update_steps_by_stage", None)
        if isinstance(by_stage, dict) and "warmup_inner" in by_stage:
            return int(by_stage["warmup_inner"])
        return int(getattr(trainer.cfg, "rollout_update_steps", 0) or 0)

    def _validate_inner(epoch_idx: int):
        trainer.env.set_mode("val")
        trainer.agent.net.eval()
        ret_stats = trainer.run_episode(
            trainer.env,
            mode="eval",
            phase="joint",
            fixed_cycle=None,
            use_rule_switch=False,
        )
        metrics = trainer._compute_metrics(ret_stats["history"])
        score = trainer._validation_score(metrics, trainer.cfg, phase="warmup_inner")
        trainer.env.set_mode("train")
        trainer.agent.net.train()
        trainer.logger.info(
            "   >>> [VAL inner_only ep=%s] select=%s score=%.4f Sharpe=%.4f Ret=%.2f%% MDD=%.2f%% switches=%s",
            epoch_idx,
            getattr(trainer.cfg, "inner_selection_metric", "return"),
            score,
            metrics["sharpe"],
            metrics["total_ret"] * 100.0,
            metrics["max_dd"] * 100.0,
            ret_stats.get("switch_count", 0),
        )
        return score

    best_score = -np.inf
    loop_count = max(0, int(inner_episodes))
    update_steps = _rollout_update_steps_for_inner()
    trainer.logger.info(
        "### INNER-ONLY FROM FROZEN OUTER+CONTROLLER: %s episodes, controller active, update_every=%s steps ###",
        loop_count,
        update_steps,
    )
    for ep in range(loop_count):
        ret = trainer.run_episode(
            trainer.env,
            mode="train",
            phase="round_inner",
            fixed_cycle=None,
            use_rule_switch=False,
            rollout_update_steps=update_steps,
            auto_update_phase="round_inner",
            train_monitor=train_monitor,
        )
        loss = ret.get("loss_log", {})
        trainer.logger.info(
            "[%s] Inner-Only %s/%s | Updates:%s L_in:%0.3f L_in_pred:%0.3f",
            save_prefix,
            ep + 1,
            loop_count,
            ret.get("update_count", 0),
            loss.get("inn_pi", 0.0),
            loss.get("inn_pred", 0.0),
        )
        if (ep + 1) % max(1, int(val_interval)) == 0 or (ep + 1) == loop_count:
            score = _validate_inner(ep + 1)
            if score > best_score:
                best_score = score
                trainer.save_model(final_best_ckpt)
                trainer.logger.info("       (New Inner-Only Best: %.4f)", best_score)

    if os.path.exists(os.path.join(trainer.model_dir, final_best_ckpt)):
        trainer._load_model(final_best_ckpt)
        trainer.logger.info("   ↺ Inner-only finished. Loaded best final model.")
    else:
        trainer.save_model(final_best_ckpt)
    trainer.save_model(final_last_ckpt)
    return {
        "fixed_cycle": fixed_cycle,
        "best_sharpe": float(best_score if np.isfinite(best_score) else 0.0),
        "best_selection_score": float(best_score if np.isfinite(best_score) else 0.0),
        "model_selection_metric": str(getattr(trainer.cfg, "inner_selection_metric", "return")),
        "best_ckpt": final_best_ckpt,
        "controller_frozen_input_ckpt": frozen_controller_ckpt,
        "inner_only": {
            "episodes": int(loop_count),
            "controller_active": True,
            "outer_controller_frozen": True,
            "selection_metric": str(getattr(trainer.cfg, "inner_selection_metric", "return")),
        },
    }


# ==============================================================================
# 4. 主程序入口
# ==============================================================================
def main(cun_path, logger_ignored, seed_list=None):
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_episodes', type=int, default=43)  # 建议增加总局数以配合分阶段
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--cun_path', type=str, default='./results')

    # 学习率
    parser.add_argument('--lr_monitor', type=float, default=1e-3)
    parser.add_argument('--lr_outer', type=float, default=1e-3)
    parser.add_argument('--lr_inner', type=float, default=1e-3)

    # PPO 参数
    parser.add_argument('--ppo_epochs', type=int, default=3)
    parser.add_argument('--inner_batch_size', type=int, default=64)
    parser.add_argument('--trade_num', type=int, default=5)
    parser.add_argument('--ssm_dim', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outer_pred_coef', type=float, default=0.1)
    parser.add_argument('--inner_pred_coef', type=float, default=0.0)
    parser.add_argument('--inner_gate_reg_coef', type=float, default=0.0)
    parser.add_argument('--train_episodes_per_epoch', type=int, default=5)
    parser.add_argument('--train_start_stride_days', type=int, default=5)
    parser.add_argument('--train_episode_to_end', type=bool, default=True)

    args = parser.parse_args(args=[]) # 避免并行时命令行参数干扰

    args.cun_path = os.path.join(cun_path, 'ppo')
    # 为当前批次创建独立日志 (如果 seed_list 存在)
    from utils.Log import create_logger
    batch_label = f"batch_{seed_list[0]}_{seed_list[-1]}"
    local_logger = create_logger(os.path.join(args.cun_path, 'logs', batch_label))

    for current_seed in seed_list:
        local_logger.info(f"开始种子 {current_seed} 的训练任务...")

        # 1. 设置随机种子
        set_seed(current_seed, local_logger)

        # 2. 配置初始化
        cfg = config
        for k, v in vars(args).items():
            setattr(cfg, k, v)
        cfg.seed = current_seed
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        cfg.device = device

        # Init
        env = PPO_Env(
            logger=local_logger,
            train_episodes_per_epoch=args.train_episodes_per_epoch,
            train_start_stride_days=args.train_start_stride_days,
            train_episode_to_end=args.train_episode_to_end,
        )
        networks = HRL_Networks(args.ssm_dim, env.num_stocks, cfg).to(device)
        agent = HRL_PPO_Agent(networks, cfg)
        buffer = HRL_Buffer(
            capacity=3000,
            device=device,
            outer_reward_scale=getattr(cfg, 'reward_scale_outer', 1.0),
            outer_reward_mode=getattr(cfg, 'outer_reward_mode', 'return'),
        )

        trainer = HRL_Trainer(agent, env, buffer, cfg, local_logger)
        env.set_mode('train')

        res = train_warmup_then_joint_with_monitor(
            trainer,
            warmup_outer_episodes=25,
            warmup_inner_episodes=25,
            warmup_monitor_episodes=10,
            joint_episodes=15,
            fixed_cycle=cfg.max_hold,
            save_prefix="exp_cf_mon",

        )

        # # Run Hybrid Strategy
        # trainer.train_hybrid_strategy()
        trainer.test(res["best_ckpt"])
        local_logger.info(f"种子 {current_seed} 训练测试完成。")

        # best_path = os.path.join(args.cun_path, f'seed_{current_seed}', 'checkpoints',
        #                  'best_model.pth')  #
        # trainer.test(best_path)



def run_single_seed_single_threshold(
        cun_path,
        seed: int,
        threshold: float,
        train_if_needed: bool = False,
        warmup_outer_episodes: int = 25,
        warmup_inner_episodes: int = 25,
        warmup_monitor_episodes: int = 10,
        joint_episodes: int = 15,
):
    """
    针对单个 seed + 单个 threshold 的定向补充实验入口。
    适合论文附录/补充实验：
        - 如果已有 best_model.pth，直接加载并测试
        - 如果没有且 train_if_needed=True，则先训练再测试
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--train_episodes', type=int, default=43)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--cun_path', type=str, default='./results')

    parser.add_argument('--lr_monitor', type=float, default=1e-3)
    parser.add_argument('--lr_outer', type=float, default=1e-3)
    parser.add_argument('--lr_inner', type=float, default=1e-3)

    parser.add_argument('--ppo_epochs', type=int, default=3)
    parser.add_argument('--inner_batch_size', type=int, default=64)
    parser.add_argument('--trade_num', type=int, default=5)
    parser.add_argument('--ssm_dim', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outer_pred_coef', type=float, default=0.1)
    parser.add_argument('--inner_pred_coef', type=float, default=0.0)
    parser.add_argument('--inner_gate_reg_coef', type=float, default=0.0)
    parser.add_argument('--train_episodes_per_epoch', type=int, default=5)
    parser.add_argument('--train_start_stride_days', type=int, default=5)
    parser.add_argument('--train_episode_to_end', type=bool, default=True)

    args = parser.parse_args(args=[])
    args.cun_path = os.path.join(cun_path, 'ppo')

    from utils.Log import create_logger
    logger = create_logger(os.path.join(args.cun_path, 'logs', f"single_seed_{seed}_th_{threshold}"))

    logger.info(f"开始单独补充实验: seed={seed}, threshold={threshold}")

    set_seed(seed, logger)

    cfg = config
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    cfg.seed = seed
    cfg.rule_switch_threshold = threshold

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg.device = device

    env = PPO_Env(
        logger=logger,
        train_episodes_per_epoch=args.train_episodes_per_epoch,
        train_start_stride_days=args.train_start_stride_days,
        train_episode_to_end=args.train_episode_to_end,
    )
    networks = HRL_Networks(args.ssm_dim, env.num_stocks, cfg).to(device)
    agent = HRL_PPO_Agent(networks, cfg)
    buffer = HRL_Buffer(
        capacity=3000,
        device=device,
        outer_reward_scale=getattr(cfg, 'reward_scale_outer', 1.0),
        outer_reward_mode=getattr(cfg, 'outer_reward_mode', 'return'),
    )

    trainer = HRL_Trainer(agent, env, buffer, cfg, logger)
    env.set_mode('train')

    best_model_path = os.path.join(trainer.model_dir, "best_model.pth")

    if (not os.path.exists(best_model_path)) and train_if_needed:
        logger.info("未发现 best_model.pth，先执行训练...")
        res = train_warmup_then_joint_with_monitor(
            trainer,
            warmup_outer_episodes=warmup_outer_episodes,
            warmup_inner_episodes=warmup_inner_episodes,
            warmup_monitor_episodes=warmup_monitor_episodes,
            joint_episodes=joint_episodes,
            fixed_cycle=cfg.max_hold,
            save_prefix=f"single_seed_{seed}",
        )
        best_model_path = os.path.join(trainer.model_dir, res["best_ckpt"])

    elif not os.path.exists(best_model_path):
        logger.warning(f"未找到模型: {best_model_path}")
        logger.warning("请先训练对应 seed，或者设置 train_if_needed=True")
        return None

    summary = trainer.run_single_rule_threshold_experiment(
        model_path=best_model_path,
        threshold=threshold,
        seed=seed,
        compare_with_fixed=True,
        fixed_cycle=cfg.max_hold,
        save_tag=f"seed{seed}_th{threshold}"
    )

    logger.info(f"补充实验完成: seed={seed}, threshold={threshold}")
    return summary
