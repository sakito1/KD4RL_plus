import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

# === 导入自定义模块 ===
from env import PPO_Env
from agent import HRL_PPO_Agent, HRL_Buffer
# 导入模型定义
from Components.PPO_model import FullModel
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


# ==============================================================================
# 2. 网络包装器 (Model Wrapper)
# ==============================================================================
class HRL_Networks(nn.Module):
    def __init__(self, ssm_dim, num_stocks, cfg):
        super(HRL_Networks, self).__init__()
        raw_feature_dim = len(cfg.dataset['features_name']) if hasattr(cfg, 'dataset') else 102
        port_state_dim = 6
        HIDDEN_DIM = 32

        self.model = FullModel(
            monitor_args=dict(
                z_dim=ssm_dim,
                h_dim=ssm_dim,
                port_state_dim=port_state_dim,
                hidden_dim=HIDDEN_DIM,
                action_dim=num_stocks,
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

    # ==============================================================================
    # 核心 Episode 运行逻辑 (支持 Phase 和 强制换仓)
    # ==============================================================================
    def run_episode(self, env, *, mode='train', phase='warmup_outer', fixed_cycle=None, disable_inner=False,
                    use_rule_switch=False):
        obs = env.reset()
        is_train = (mode == 'train')
        if is_train:
            # 支持多 episode 累积更新：标记当前 episode 的起点
            if hasattr(self.buffer, 'mark_episode_start'):
                self.buffer.mark_episode_start()

        spec = self._get_phase_spec(phase)

        T = env.episode_len
        m_hold = int(getattr(self.cfg, 'max_hold', 60))
        switch_schedule = sample_switch_schedule(T,m_hold, m_hold) if spec.use_schedule else None

        ret_stats = {'total': 0.0, 'history': [env.portfolio_value.item()]}
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

        # [Rule] 初始化规则计数器
        rule_consecutive_low = 0
        RULE_THRESHOLD = getattr(self.cfg, 'rule_switch_threshold', 0.5)

        # [Fix] 从配置读取最小/最大持仓，确保规则模式也遵守
        CFG_MIN_HOLD = int(getattr(self.cfg, 'min_hold', 5))  # 默认5天
        CFG_MAX_HOLD = int(getattr(self.cfg, 'max_hold', 60))  # 默认60天

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

            base_used = out['base_used']
            weights_exec = out['weights_exec']
            outer_action = out['act_out']
            is_switch_action = (out['act_mon'].item() == 1)

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

            if is_train:
                self.buffer.store_daily({
                    'ssm': obs['ssm'],
                    'outer_state': obs['outer_state'],
                    'inner_state': obs['inner_state'],
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
                })

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
                    with torch.no_grad():
                        last_out = self.agent.get_action(
                            obs,
                            mode=mode,
                            force_switch=0,
                            force_inner_zero=True,
                            force_locked=True,
                        )
                    self.buffer.finish_episode({
                        'val_inn': last_out['val_inn'].item(),
                        'val_out': last_out['val_out'].item(),
                        'val_mon': last_out['val_mon'].item(),
                    })
                break

            step_idx += 1

        ret_stats['total'] = (ret_stats['history'][-1] - ret_stats['history'][0]) / (ret_stats['history'][0] + 1e-8)
        ret_stats['switch_count'] = switch_count
        ret_stats['switch_free_count'] = switch_free_count
        ret_stats['forced_hold_count'] = forced_hold_count
        ret_stats['forced_switch_count'] = forced_switch_count
        ret_stats['forced_schedule_count'] = forced_schedule_count
        ret_stats['total_steps'] = step_idx
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
        self.logger.info("-" * 60)
        self.logger.info(f"Total Ret  : {metrics['total_ret'] * 100:.2f}%")
        self.logger.info(f"Ann Ret    : {metrics['ann_ret'] * 100:.2f}%")
        self.logger.info(f"Ann Vol    : {metrics['ann_vol'] * 100:.2f}%")
        self.logger.info(f"Sharpe     : {metrics['sharpe']:.4f}")
        self.logger.info(f"Max DD     : {metrics['max_dd'] * 100:.2f}%")
        self.logger.info("=" * 60 + "\n")

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
        # Scenario 1: 不使用 monitor，固定 60 天
        self.logger.info(f"Running Scenario 1: No Monitor, Fixed {fix_cycle} Days...")
        ret_stats_1 = self.run_episode(self.env, mode='eval', phase='joint', fixed_cycle=fix_cycle)
        metrics_1 = self._compute_metrics(ret_stats_1['history'])
        self._print_report(f"Scenario 1 (No Monitor, {fix_cycle}d)", ret_stats_1, metrics_1)
        pd.DataFrame(ret_stats_1['history'], columns=['value']).to_csv(
            os.path.join(self.run_dir, f"test_s1_Fixed{fix_cycle}d.csv"), index=False
        )

        # Scenario 2: 不使用 monitor 和 inner actor，固定 60 天
        self.logger.info(f"Running Scenario 2: No Monitor/Inner, Fixed {fix_cycle} Days...")
        ret_stats_2 = self.run_episode(self.env, mode='eval', phase='joint', fixed_cycle=fix_cycle, disable_inner=True)
        metrics_2 = self._compute_metrics(ret_stats_2['history'])
        self._print_report(f"Scenario 2 (No Mon/Inner, {fix_cycle}d)", ret_stats_2, metrics_2)
        pd.DataFrame(ret_stats_2['history'], columns=['value']).to_csv(
            os.path.join(self.run_dir, f"test_s2_Fixed{fix_cycle}d_NoInner.csv"), index=False
        )

        # Scenario 3: 三个模块都用 (Standard)
        self.logger.info("Running Scenario 3: All Modules (Standard)...")
        ret_stats_3 = self.run_episode(self.env, mode='eval', phase='joint')
        metrics_3 = self._compute_metrics(ret_stats_3['history'])
        self._print_report("Scenario 3 (All Modules)", ret_stats_3, metrics_3)
        pd.DataFrame(ret_stats_3['history'], columns=['value']).to_csv(
            os.path.join(self.run_dir, "test_s3_AllModules.csv"), index=False
        )

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


def train_warmup_then_joint_no_monitor(trainer,
                                       warmup_outer_episodes: int = 50,
                                       warmup_inner_episodes: int = 50,
                                       joint_episodes: int = 100,
                                       fixed_cycle: int = 60,
                                       val_interval: int = None,
                                       save_prefix: str = "nm60"):
    import numpy as np
    import os

    if val_interval is None:
        val_interval = int(getattr(trainer.cfg, "val_interval", 10))

    # 强制设定：前 10 轮不验证
    MIN_WARMUP_STEPS = 6

    # 文件名定义
    final_best_ckpt = f"best_model.pth"
    final_last_ckpt = f"last_model.pth"
    warmup_out_best_ckpt = f"temp_warmup_outer.pth"
    warmup_inn_best_ckpt = f"temp_warmup_inner.pth"

    def _do_validate(tag: str):
        trainer.env.set_mode("val")
        trainer.agent.net.eval()
        val_ret = trainer.run_episode(trainer.env, mode="eval", phase="round_outer",
                                      fixed_cycle=fixed_cycle, disable_inner=False)
        val_metrics = trainer._compute_metrics(val_ret["history"])
        trainer.env.set_mode("train")
        trainer.agent.net.train()
        trainer.logger.info(
            f"   >>> [VAL {tag}] Sharpe:{val_metrics['sharpe']:.4f} | Ret:{val_metrics['total_ret'] * 100:.2f}%")
        return val_metrics

    # =========================================================
    # 1) Warmup Outer
    # =========================================================
    trainer.logger.info(f"### [1/3] Warmup OUTER: {warmup_outer_episodes} eps (Min Warmup: {MIN_WARMUP_STEPS}) ###")
    trainer.agent.set_module_status("outer")
    phase_best_sharpe = -np.inf

    for ep in range(warmup_outer_episodes):
        # 每个 Episode 更新一次
        ret = trainer.run_episode(trainer.env, mode="train", phase="warmup_outer", fixed_cycle=fixed_cycle)
        loss_log = trainer.agent.update(trainer.buffer.get_batch(), phase="warmup_outer")
        trainer.buffer.clear()

        trainer.logger.info(
            f"[{save_prefix}] W-Out {ep + 1}/{warmup_outer_episodes} | L_out:{loss_log.get('out_pi', 0):.3f}")

        # 前 10 轮跳过，之后每 2 轮验证一次
        if (ep + 1) > MIN_WARMUP_STEPS and (ep + 1) % val_interval == 0:
            m = _do_validate("warmup_outer")
            if float(m["sharpe"]) > phase_best_sharpe:
                phase_best_sharpe = float(m["sharpe"])
                trainer.save_model(warmup_out_best_ckpt)
                trainer.logger.info(f"       (New Warmup-Outer Best: {phase_best_sharpe:.4f})")

    # 预热结束，回滚至该阶段最优，为 Inner 预热提供更好的基准
    if os.path.exists(os.path.join(trainer.model_dir, warmup_out_best_ckpt)):
        trainer._load_model(warmup_out_best_ckpt)
        trainer.logger.info(f"   ↺ Phase Outer finished. Loaded best warmup_outer model.")

    # =========================================================
    # 2) Warmup Inner
    # =========================================================
    trainer.logger.info(f"### [2/3] Warmup INNER: {warmup_inner_episodes} eps (Min Warmup: {MIN_WARMUP_STEPS}) ###")
    trainer.agent.set_module_status("inner")
    phase_best_sharpe = -np.inf

    for ep in range(warmup_inner_episodes):
        ret = trainer.run_episode(trainer.env, mode="train", phase="warmup_inner", fixed_cycle=fixed_cycle)
        loss_log = trainer.agent.update(trainer.buffer.get_batch(), phase="warmup_inner")
        trainer.buffer.clear()

        trainer.logger.info(
            f"[{save_prefix}] W-Inn {ep + 1}/{warmup_inner_episodes} | L_in:{loss_log.get('inn_pi', 0):.3f}")

        if (ep + 1) > MIN_WARMUP_STEPS and (ep + 1) % val_interval == 0:
            m = _do_validate("warmup_inner")
            if float(m["sharpe"]) > phase_best_sharpe:
                phase_best_sharpe = float(m["sharpe"])
                trainer.save_model(warmup_inn_best_ckpt)
                trainer.logger.info(f"       (New Warmup-Inner Best: {phase_best_sharpe:.4f})")

    # 回滚至该阶段最优，进入联合训练
    if os.path.exists(os.path.join(trainer.model_dir, warmup_inn_best_ckpt)):
        trainer._load_model(warmup_inn_best_ckpt)
        trainer.logger.info(f"   ↺ Phase Inner finished. Loaded best warmup_inner model.")

    # =========================================================
    # 3) Joint Finetune
    # =========================================================
    trainer.logger.info(f"### [3/3] JOINT (Outer+Inner): {joint_episodes} eps ###")
    trainer.agent.set_module_status("all")
    global_best_sharpe = -np.inf

    for ep in range(joint_episodes):
        ret = trainer.run_episode(trainer.env, mode="train", phase="round_outer", fixed_cycle=fixed_cycle)
        batch = trainer.buffer.get_batch()

        # 联合训练中，两个模块均在每个 Episode 后更新
        loss_out = trainer.agent.update(batch, phase="round_outer")
        loss_inn = trainer.agent.update(batch, phase="round_inner")
        trainer.buffer.clear()

        trainer.logger.info(
            f"[{save_prefix}] Joint {ep + 1}/{joint_episodes} | L_out:{loss_out.get('out_pi', 0):.3f} L_in:{loss_inn.get('inn_pi', 0):.3f}")

        # 联合训练阶段始终开启验证
        m = _do_validate("joint")
        if float(m["sharpe"]) > global_best_sharpe:
            global_best_sharpe = float(m["sharpe"])
            trainer.save_model(final_best_ckpt)
            trainer.logger.info(f"       (🏆 New Final Best Saved: {global_best_sharpe:.4f})")

    trainer.save_model(final_last_ckpt)
    return {
        "fixed_cycle": fixed_cycle,
        "best_sharpe": float(global_best_sharpe),
        "best_ckpt": final_best_ckpt
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
    parser.add_argument('--trade_num', type=int, default=10)
    parser.add_argument('--ssm_dim', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')

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
        env = PPO_Env(logger=local_logger)
        networks = HRL_Networks(args.ssm_dim, env.num_stocks, cfg).to(device)
        agent = HRL_PPO_Agent(networks, cfg)
        buffer = HRL_Buffer(capacity=3000, device=device)

        trainer = HRL_Trainer(agent, env, buffer, cfg, local_logger)
        env.set_mode('train')

        res = train_warmup_then_joint_no_monitor(
            trainer,
            warmup_outer_episodes=46,
            warmup_inner_episodes=46,
            joint_episodes=20,
            fixed_cycle=cfg.max_hold,
            save_prefix="exp_nm",

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
        warmup_outer_episodes: int = 46,
        warmup_inner_episodes: int = 46,
        joint_episodes: int = 20,
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
    parser.add_argument('--trade_num', type=int, default=10)
    parser.add_argument('--ssm_dim', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')

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

    env = PPO_Env(logger=logger)
    networks = HRL_Networks(args.ssm_dim, env.num_stocks, cfg).to(device)
    agent = HRL_PPO_Agent(networks, cfg)
    buffer = HRL_Buffer(capacity=3000, device=device)

    trainer = HRL_Trainer(agent, env, buffer, cfg, logger)
    env.set_mode('train')

    best_model_path = os.path.join(trainer.model_dir, "best_model.pth")

    if (not os.path.exists(best_model_path)) and train_if_needed:
        logger.info("未发现 best_model.pth，先执行训练...")
        res = train_warmup_then_joint_no_monitor(
            trainer,
            warmup_outer_episodes=warmup_outer_episodes,
            warmup_inner_episodes=warmup_inner_episodes,
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
