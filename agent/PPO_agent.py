import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


def actor_score_smooth_l1_loss(actor_mu, target, *, squash=False, target_scale=1.0):
    """Supervise actor trading scores directly, without using auxiliary pred heads."""
    if hasattr(actor_mu, "rsample"):
        actor_mu = actor_mu.rsample()
    score = torch.tanh(actor_mu) if squash else actor_mu
    target = target.to(device=score.device, dtype=score.dtype) * float(target_scale)
    return F.smooth_l1_loss(score, target)


class HRL_Buffer:
    """
    [Simplified] Hierarchical PPO Buffer for Single-Episode Updates

    Simplified logic:
    1. Stores data for exactly one episode.
    2. 'finish_episode' calculates GAE using the entire buffer.
    3. 'get_batch' returns the stacked data.
    4. Must be cleared after every update.
    """

    def __init__(
            self,
            capacity,
            device,
            gamma=0.99,
            gae_lambda=0.95,
            outer_reward_scale=1.0,
            outer_reward_mode="return",
    ):
        self.device = device
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.outer_reward_scale = float(outer_reward_scale)
        self.outer_reward_mode = str(outer_reward_mode)
        self.clear()

    def _outer_segment_reward(self, segment_returns):
        mode = str(getattr(self, "outer_reward_mode", "return"))
        if mode == "return":
            return torch.sum(segment_returns)
        if mode == "segment_sharpe":
            mean = segment_returns.mean()
            std = segment_returns.std(unbiased=False)
            return mean / (std + 1e-8) * (252.0 ** 0.5)
        raise ValueError(f"Unknown outer reward mode: {mode}")

    def clear(self):
        # Initialize storage lists
        self.data = {
            'outer_state': [], 'inner_state': [],
            'inner_base_used': [], 'inner_weights_drift': [], 'inner_indices': [],
            'port_state': [], 'weights_drift': [], 'base_drift': [],
            'base_used': [], 'weights_exec': [],
            'act_mon': [], 'logp_mon': [],
            'act_out_raw': [], 'act_out': [], 'logp_out': [],
            'act_inn_raw': [], 'logp_inn': [],
            'val_mon': [], 'val_out': [], 'val_inn': [],
            'rew_mon': [], 'rew_alpha': [], 'rew_outer_raw': [],
            'is_switch': [], 'is_locked': [], 'dones': [],
            'outer_stock_return_target': [],
            'inner_stock_return_target': [],
            'inner_next_return_target': [],
            'controller_hold_return_target': [],
            'controller_hold_mdd_target': [],
            'controller_switch_label': [],
            'controller_sup_weight': [],
            # GAE results will be stored here
            'adv_inn': [], 'ret_inn': [],
            'adv_mon': [], 'ret_mon': [],
            'adv_out': [], 'ret_out': []
        }

        # Multi-episode support
        self.cur_ep_start = 0

    def store_daily(self, transition):
        """Store a single step's data."""
        for k, v in transition.items():
            if k in self.data:
                val = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                self.data[k].append(val)

    def mark_episode_start(self):
        """在每个 episode 开始时调用，用于多 episode 累积更新。"""
        self.cur_ep_start = len(self.data['rew_mon'])

    def finish_episode(self, last_vals, start_idx=None):
        """
        结束一个 episode：对 [start_idx, end) 这段计算 GAE，并把结果 append 到 adv/ret 列表。
        - 兼容单 episode：不传 start_idx 时默认使用 self.cur_ep_start（初始为 0）。
        - 支持多 episode 累积：每个 episode 开始前调用 mark_episode_start()。
        """
        if start_idx is None:
            start_idx = int(getattr(self, "cur_ep_start", 0))

        end_idx = len(self.data['rew_mon'])
        length = end_idx - start_idx
        if length <= 0:
            return

        def get_tensor(key):
            return torch.stack(self.data[key][start_idx:end_idx]).view(-1)

        # done_mask: 1 means non-terminal, 0 means terminal
        dones = get_tensor('dones').float()
        done_mask = 1.0 - dones  # (T,)

        sw = get_tensor('is_switch').view(-1)

        # ==========================
        # 1) Inner GAE (discounted, with done_mask)
        # ==========================
        rew = get_tensor('rew_alpha')
        val = get_tensor('val_inn')
        last_val_inn = last_vals['val_inn']  # scalar/tensor

        adv = torch.zeros(length, device=self.device)
        last_gae = torch.zeros((), device=self.device)

        for t in reversed(range(length)):
            next_val = last_val_inn if t == length - 1 else val[t + 1]
            m = done_mask[t]
            delta = rew[t] + self.gamma * m * next_val - val[t]
            last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
            adv[t] = last_gae

        ret = adv + val
        self.data['adv_inn'].extend(list(adv.detach()))
        self.data['ret_inn'].extend(list(ret.detach()))

        # ==========================
        # 2) Monitor GAE (discounted, with done_mask)
        # ==========================
        rew_mon = get_tensor('rew_mon')
        val_mon = get_tensor('val_mon')
        last_v_mon = last_vals['val_mon']

        adv = torch.zeros(length, device=self.device)
        last_gae = torch.zeros((), device=self.device)

        for t in reversed(range(length)):
            next_val = last_v_mon if t == length - 1 else val_mon[t + 1]
            m = done_mask[t]
            delta = rew_mon[t] + self.gamma * m * next_val - val_mon[t]
            last_gae = delta + self.gamma * self.gae_lambda * m * last_gae
            adv[t] = last_gae

        ret = adv + val_mon
        self.data['adv_mon'].extend(list(adv.detach()))
        self.data['ret_mon'].extend(list(ret.detach()))

        # ==========================
        # 3) Outer (SMDP segment return + bootstrap, with done handling)
        # ==========================
        rew_out = get_tensor('rew_outer_raw')
        v_out = get_tensor('val_out')
        last_v_out = last_vals['val_out']

        adv = torch.zeros(length, device=self.device)
        ret = torch.zeros(length, device=self.device)
        switch_indices = torch.where(sw == 1)[0]
        if switch_indices.numel() == 0:
            # 没有外层决策点：也要对齐长度（全部 0）
            self.data['adv_out'].extend(list(adv.detach()))
            self.data['ret_out'].extend(list(ret.detach()))
            return

        boundaries = torch.cat([switch_indices, torch.tensor([length], device=self.device)])

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            segment_rew = self._outer_segment_reward(rew_out[start:end]) * self.outer_reward_scale

            last_step = end - 1
            if last_step >= 0 and dones[last_step] > 0.5:
                bootstrap_val = torch.zeros((), device=self.device)
            else:
                bootstrap_val = last_v_out if end == length else v_out[end]

            target = segment_rew + bootstrap_val * 0.90
            ret[start] = target
            adv[start] = target - v_out[start]

        self.data['adv_out'].extend(list(adv.detach()))
        self.data['ret_out'].extend(list(ret.detach()))

    def get_batch(self):
        """
        Stack all lists into tensors for the PPO update.
        [Optimized] Do NOT stack 'outer_state' to avoid OOM. Keep it as list.
        """
        res = {}
        for k, v in self.data.items():
            if len(v) == 0: continue

            if k == 'outer_state':
                # [FIX] Do NOT stack outer_state, keeps huge memory savings
                res[k] = v  # Keep as list of tensors (or list of individual tensors)

            elif isinstance(v, torch.Tensor):
                res[k] = v

            elif isinstance(v, list):
                # Stack list of tensors
                try:
                    stacked = torch.stack(v)
                    if stacked.dim() > 1 and stacked.shape[1] == 1:
                        stacked = stacked.squeeze(1)
                    res[k] = stacked
                except:
                    # Fallback if stacking fails (e.g. empty list or dim mismatch)
                    res[k] = v

        return res


class HRL_PPO_Agent:
    def __init__(self, models, config):
        self.net = models
        self.cfg = config
        self.device = config.device

        # PPO hyper-params
        self.clip_range = getattr(config, 'clip_range', 0.2)
        self.vf_coef = getattr(config, 'vf_coef', 0.1)
        self.ent_coef = getattr(config, 'ent_coef', 0.001)
        self.max_grad_norm = getattr(config, 'max_grad_norm', None)

        # 直接优化三个集成模块
        self.opt_mon = optim.Adam(self.net.mon.parameters(), lr=config.lr_monitor)
        self.opt_out = optim.Adam(self.net.outer.parameters(), lr=config.lr_outer)
        self.opt_inn = optim.Adam(self.net.inner.parameters(), lr=config.lr_inner)
        self.base_lrs = {
            "monitor": float(config.lr_monitor),
            "outer": float(config.lr_outer),
            "inner": float(config.lr_inner),
        }

        self.mse_loss = nn.MSELoss()
        self.min_clip = -5
        self.max_clip = 5

    @staticmethod
    def _set_optimizer_lr(optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

    def set_lr_multiplier(self, multiplier=1.0):
        multiplier = float(multiplier)
        self._set_optimizer_lr(self.opt_mon, self.base_lrs["monitor"] * multiplier)
        self._set_optimizer_lr(self.opt_out, self.base_lrs["outer"] * multiplier)
        self._set_optimizer_lr(self.opt_inn, self.base_lrs["inner"] * multiplier)
        return {
            "monitor": self.base_lrs["monitor"] * multiplier,
            "outer": self.base_lrs["outer"] * multiplier,
            "inner": self.base_lrs["inner"] * multiplier,
        }

    def set_module_status(self, mode='all'):
        """灵活冻结/解冻模块。"""
        for p in self.net.parameters():
            p.requires_grad = False

        if mode == 'all':
            for p in self.net.parameters():
                p.requires_grad = True
            return

        if mode == 'inner':
            for p in self.net.inner.parameters():
                p.requires_grad = True
        elif mode == 'outer':
            for p in self.net.outer.parameters():
                p.requires_grad = True
        elif mode == 'outer_inner':
            for p in self.net.outer.parameters():
                p.requires_grad = True
            for p in self.net.inner.parameters():
                p.requires_grad = True
        elif mode == 'monitor':
            for p in self.net.mon.parameters():
                p.requires_grad = True
        elif mode == 'controller':
            for p in self.net.mon.parameters():
                p.requires_grad = True

    @staticmethod
    def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        denom = x.sum(dim=1, keepdim=True)
        fallback = torch.ones_like(x) / float(max(x.shape[1], 1))
        return torch.where(denom.abs() > eps, x / (denom + eps), fallback)

    def _select_inner_inputs(self, obs, base_used, weight_drift):
        """Crop inner inputs to the currently tradable/topK assets."""
        inner_state = obs['inner_state']
        B, N = base_used.shape
        if not bool(getattr(self.cfg, 'inner_use_topk', False)):
            indices = torch.arange(N, device=base_used.device).view(1, N).expand(B, -1)
            return inner_state, base_used, weight_drift, indices

        K = min(int(getattr(self.cfg, 'trade_num', N)), N)
        indices = torch.topk(base_used, k=K, dim=1).indices

        _, _, T, Fdim = inner_state.shape
        state_idx = indices.view(B, K, 1, 1).expand(-1, -1, T, Fdim)
        inner_state_sel = torch.gather(inner_state, dim=1, index=state_idx)
        base_sel = torch.gather(base_used, dim=1, index=indices)
        drift_sel = torch.gather(weight_drift, dim=1, index=indices)
        base_sel = self._normalize_rows(base_sel)
        drift_sel = self._normalize_rows(drift_sel)
        return inner_state_sel, base_sel, drift_sel, indices

    @staticmethod
    def _scatter_selected_weights(selected_weights, indices, num_assets):
        full = selected_weights.new_zeros((selected_weights.shape[0], num_assets))
        return full.scatter(1, indices, selected_weights)

    def _preview_inner_exec(self, obs, base_used, weight_drift, *, force_inner_zero=False):
        if force_inner_zero:
            return base_used.detach()
        inner_state_used, inner_base_used, inner_weight_drift, inner_indices = self._select_inner_inputs(
            obs,
            base_used,
            weight_drift,
        )
        with torch.no_grad():
            if hasattr(self.net.inner, "build_inner_action_simple"):
                alpha = float(getattr(self.cfg, "inner_max_boundary", 1.0))
                selected, _, _, _, _ = self.net.inner.build_inner_action_simple(
                    inner_state_used,
                    inner_base_used,
                    inner_weight_drift,
                    alpha=alpha,
                    deterministic=True,
                )
            else:
                raw, _, _ = self.net.inner.pi(
                    inner_state_used,
                    inner_base_used,
                    inner_weight_drift,
                    deterministic=True,
                )
                adjusted = inner_base_used * torch.exp(
                    torch.clamp(raw, -3.0, 3.0) * self.cfg.inner_max_boundary
                )
                selected = adjusted / (adjusted.sum(dim=1, keepdim=True) + 1e-12)
        return self._scatter_selected_weights(
            selected,
            inner_indices,
            base_used.shape[1],
        ).detach()

    def _select_outer_state_batch(self, outer_state, idx_or_mask=None):
        if outer_state is None:
            return None
        if idx_or_mask is None:
            state = outer_state
        elif isinstance(outer_state, list):
            if isinstance(idx_or_mask, torch.Tensor) and idx_or_mask.dtype == torch.bool:
                indices = torch.where(idx_or_mask)[0]
            else:
                indices = idx_or_mask
            subset = [outer_state[int(i)] for i in indices.detach().cpu().tolist()]
            state = torch.stack(subset)
        else:
            state = outer_state[idx_or_mask]
        if state.dim() == 5 and state.shape[1] == 1:
            state = state.squeeze(1)
        return state.to(self.device)

    def get_action(self, obs, mode='train', force_switch=None, force_inner_zero=False, force_locked=False):
        is_train = (mode == 'train')

        weight_drift = obs['weights_drift']
        base_drift = obs['base_drift']

        # =========================
        # Outer
        # =========================
        # 固定持仓的 hold 日不需要产生新的外层组合，跳过最贵的 LSTM/CAAN。
        # switch 日和自由 monitor 决策日仍正常计算 outer actor/critic。
        needs_outer_action = (force_switch is None) or (int(force_switch) == 1)
        if needs_outer_action:
            act_out, act_out_raw, logp_out, _, _ = self.net.outer.pi(
                obs['outer_state'],
                weight_drift,
                deterministic=(not is_train)
            )
            val_out = self.net.outer.value(obs['outer_state'], weight_drift)
        else:
            B = weight_drift.shape[0]
            act_out = base_drift
            act_out_raw = torch.zeros_like(weight_drift)
            logp_out = torch.zeros((B,), device=self.device)
            val_out = torch.zeros((B, 1), device=self.device)

        # =========================
        # Monitor
        # =========================
        controller_hold_exec = None
        controller_switch_exec = None
        controller_remaining_horizon = None
        if force_switch is not None:
            B = weight_drift.shape[0]
            act_mon = torch.full((B,), force_switch, dtype=torch.long, device=self.device)
            logp_mon = torch.zeros((B,), device=self.device)
            val_mon = torch.zeros((B, 1), device=self.device)
            is_locked_val = 1 if force_locked else 0
            is_locked = torch.full((B,), is_locked_val, dtype=torch.long, device=self.device)
        else:
            controller_hold_exec = weight_drift.detach()
            controller_switch_exec = self._preview_inner_exec(
                obs,
                act_out,
                weight_drift,
                force_inner_zero=force_inner_zero,
            )
            controller_remaining_horizon = (
                1.0 - obs["port_state"][:, :1]
            ).clamp(0.0, 1.0)
            act_mon_samp, logp_mon_samp, _, _, val_mon = self.net.mon(
                weight_drift, obs['port_state'], switch_action=act_out,
                deterministic=(not is_train),
                asset_state=obs.get('outer_state'),
                hold_exec_weights=controller_hold_exec,
                switch_exec_weights=controller_switch_exec,
                remaining_horizon=controller_remaining_horizon,
            )
            act_mon = act_mon_samp
            logp_mon = logp_mon_samp
            is_locked = torch.zeros_like(act_mon)

        has_switched = (act_mon == 1)
        base_used = torch.where(has_switched.unsqueeze(-1).bool(), act_out, base_drift)

        # =========================
        # Inner
        # =========================
        inner_state_used, inner_base_used, inner_weight_drift, inner_indices = self._select_inner_inputs(
            obs, base_used, weight_drift
        )
        if force_inner_zero:
            B, N = weight_drift.shape
            K = inner_indices.shape[1]
            act_inn_raw = torch.zeros((B, K), device=self.device)
            logp_inn = torch.zeros((B,), device=self.device)
            val_inn = self.net.inner.value(inner_state_used, inner_base_used, inner_weight_drift)
            weights_exec = base_used
        else:
            # 直接生成可执行权重（weights_exec），同时返回 score_sample 作为 PPO 的“原始动作变量”
            # 这样：执行动作稳定可控，训练仍然符合 PPO（ratio 用 score_sample 的 logp）
            if hasattr(self.net.inner, "build_inner_action_simple"):
                alpha = float(getattr(self.cfg, "inner_max_boundary", 1.0))
                w_new_sel, score_sample, logp_inn, _, val_inn = self.net.inner.build_inner_action_simple(
                    inner_state_used,
                    inner_base_used,
                    inner_weight_drift,
                    alpha=alpha,
                    deterministic=(not is_train),
                )
                weights_exec = self._scatter_selected_weights(w_new_sel, inner_indices, base_used.shape[1])
                act_inn_raw = score_sample
            else:
                # 兼容旧实现：用 raw_signal 生成最终权重
                act_inn_raw, logp_inn, _ = self.net.inner.pi(
                    inner_state_used,
                    inner_base_used,
                    inner_weight_drift,
                    deterministic=(not is_train)
                )
                val_inn = self.net.inner.value(inner_state_used, inner_base_used, inner_weight_drift)

                action_for_calc = torch.clamp(act_inn_raw, -3.0, 3.0)
                adjusted = inner_base_used * torch.exp(action_for_calc * self.cfg.inner_max_boundary)
                w_new_sel = adjusted / (adjusted.sum(dim=1, keepdim=True) + 1e-12)
                weights_exec = self._scatter_selected_weights(w_new_sel, inner_indices, base_used.shape[1])

        return {
            'act_mon': act_mon,
            'logp_mon': logp_mon,
            'val_mon': val_mon,
            'is_locked': is_locked,
            'is_switch': act_mon,

            'act_out_raw': act_out_raw,
            'act_out': act_out,
            'logp_out': logp_out,
            'val_out': val_out,
            'base_used': base_used,

            'act_inn_raw': act_inn_raw,
            'logp_inn': logp_inn,
            'val_inn': val_inn,
            'weights_exec': weights_exec,
            'inner_state_used': inner_state_used,
            'inner_base_used': inner_base_used,
            'inner_weights_drift': inner_weight_drift,
            'inner_indices': inner_indices,
            'controller_hold_exec': controller_hold_exec,
            'controller_switch_exec': controller_switch_exec,
            'controller_remaining_horizon': controller_remaining_horizon,
        }

    def update(self, buffer_data, phase='joint', train_monitor=None):
        bs = int(len(buffer_data.get('rew_mon', [])))
        if bs <= 0:
            return {}

        ppo_epochs = int(getattr(self.cfg, 'ppo_epochs', 1))
        if phase == 'warmup_inner':
            ppo_epochs = int(getattr(self.cfg, 'inner_ppo_epochs', ppo_epochs))
        if train_monitor is None:
            train_monitor = bool(getattr(self.cfg, 'train_monitor_enabled', True))
        metrics = {}

        for _ in range(ppo_epochs):
            if phase in ['warmup_inner', 'joint', 'round_inner']:
                l_inn = self._update_inner(buffer_data)
                self._log_metric(metrics, 'inn', l_inn)

            if train_monitor and phase in ['warmup_monitor', 'joint', 'round_monitor']:
                mask_free = torch.as_tensor(buffer_data['is_locked'] == 0, device=self.device, dtype=torch.bool)
                if torch.count_nonzero(mask_free).item() > 0:
                    l_mon = self._update_monitor(buffer_data, mask_free)
                    self._log_metric(metrics, 'mon', l_mon)

            if phase in ['warmup_outer', 'joint', 'round_outer']:
                mask_out = torch.as_tensor(buffer_data['is_switch'] == 1, device=self.device, dtype=torch.bool)
                if torch.count_nonzero(mask_out).item() > 0:
                    l_out = self._update_outer(buffer_data, mask_out)
                    self._log_metric(metrics, 'out', l_out)

        return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}

    def _log_metric(self, metrics, prefix, loss):
        for k, v in loss.items():
            metrics.setdefault(f"{prefix}_{k}", []).append(v)

    def _normalize_adv(self, adv: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-8):
        """PPO 常用技巧：对 advantage 做标准化（可选 mask）。"""
        if mask is None:
            x = adv
            if x.numel() <= 1:
                return x
            mean = x.mean()
            std = x.std(unbiased=False).clamp_min(eps)
            return (x - mean) / std
        else:
            x = adv[mask]
            if x.numel() <= 1:
                return adv
            mean = x.mean()
            std = x.std(unbiased=False).clamp_min(eps)
            out = adv.clone()
            out[mask] = (out[mask] - mean) / std
            return out

    def _update_inner(self, data):
        self.opt_inn.zero_grad(set_to_none=True)

        def _param_stats(params):
            with torch.no_grad():
                flat = torch.cat([p.data.view(-1) for p in params if p.requires_grad])
                return flat.norm().item(), flat.abs().mean().item()

        def _param_delta(params, old):
            with torch.no_grad():
                flat = torch.cat([p.data.view(-1) for p in params if p.requires_grad])
                d = flat - old
                return d.norm().item(), d.abs().mean().item()

        debug_inner_update_stats = bool(getattr(self.cfg, "debug_inner_update_stats", False))
        if debug_inner_update_stats:
            params = [p for p in self.net.inner.parameters() if p.requires_grad]
            w_norm0, _ = _param_stats(params)
            w_flat0 = torch.cat([p.data.view(-1) for p in params]).clone()

        adv_all = self._normalize_adv(data['adv_inn'])
        total = int(adv_all.shape[0])
        batch_size = max(1, int(getattr(self.cfg, 'inner_batch_size', 256)))
        perm = torch.randperm(total, device=self.device)

        loss_pi_sum = 0.0
        loss_v_sum = 0.0
        loss_pred_sum = 0.0
        entropy_sum = 0.0
        pred_coef = float(getattr(self.cfg, 'inner_pred_coef', 0.0))
        use_pred_loss = pred_coef > 0.0
        target_key = None
        if use_pred_loss:
            target_key = 'inner_stock_return_target' if 'inner_stock_return_target' in data else 'inner_next_return_target'
            if target_key not in data:
                raise KeyError("inner_pred_coef > 0 requires inner stock return targets in the PPO buffer.")

        for start in range(0, total, batch_size):
            idx = perm[start:start + batch_size]
            weight = float(idx.numel()) / float(total)

            inner_state = data['inner_state'][idx]
            base_used = data.get('inner_base_used', data['base_used'])[idx]
            weights_drift = data.get('inner_weights_drift', data['weights_drift'])[idx]

            # 用“旧动作”计算 new_logp(a_old)，避免重新采样导致 PPO ratio 失真。
            # 同一批次只编码一次，actor/critic 复用特征，避免重复跑 TCN。
            feat = self.net.inner.encode(inner_state, base_used, weights_drift)
            dist = self.net.inner.get_dist(feat)
            new_logp = dist.log_prob(data['act_inn_raw'][idx]).sum(dim=1)
            entropy = dist.entropy().mean()
            ratio = torch.exp(torch.clamp(new_logp - data['logp_inn'][idx], min=self.min_clip, max=self.max_clip))

            adv = adv_all[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            loss_pi = -torch.min(surr1, surr2).mean()

            beta_rep = torch.sum(feat * base_used.unsqueeze(-1), dim=1)
            attn_scores = self.net.inner.alpha_query(feat).squeeze(-1)
            alpha = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
            alpha_rep = torch.sum(feat * alpha, dim=1)
            global_rep = torch.cat([beta_rep, alpha_rep], dim=-1)
            val = self.net.inner.value_head(global_rep)
            loss_v = self.mse_loss(val.squeeze(-1), data['ret_inn'][idx])

            if use_pred_loss:
                pred_return = self.net.inner.pred_head(feat).squeeze(-1)
                target_return = (
                    data[target_key][idx].to(device=pred_return.device, dtype=pred_return.dtype)
                    * float(getattr(self.cfg, 'inner_pred_target_scale', 1.0))
                )
                loss_pred = F.smooth_l1_loss(pred_return, target_return)
            else:
                loss_pred = feat.new_tensor(0.0)

            loss_total = (
                loss_pi
                + self.vf_coef * loss_v
                + pred_coef * loss_pred
                - self.ent_coef * entropy
            )
            (loss_total * weight).backward()

            loss_pi_sum += loss_pi.item() * idx.numel()
            loss_v_sum += loss_v.item() * idx.numel()
            loss_pred_sum += loss_pred.item() * idx.numel()
            entropy_sum += entropy.item() * idx.numel()

        self.opt_inn.step()

        if debug_inner_update_stats:
            w_norm1, _ = _param_stats(params)
            d_norm, d_mean = _param_delta(params, w_flat0)
            print(f"[INNER] w_norm {w_norm0:.4e}->{w_norm1:.4e} | d_norm {d_norm:.4e} d_mean {d_mean:.4e}")
        if (
                bool(getattr(self.cfg, "clear_cuda_cache_on_update", False))
                and getattr(self.device, "type", str(self.device)) == "cuda"
        ):
            torch.cuda.empty_cache()
        return {
            'pi': loss_pi_sum / total,
            'v': loss_v_sum / total,
            'pred': loss_pred_sum / total,
            'ent': entropy_sum / total,
        }

    def _update_outer(self, data, mask):
        self.opt_out.zero_grad(set_to_none=True)

        valid_indices = torch.where(mask)[0]
        total = int(valid_indices.numel())
        if total <= 0:
            return {'pi': 0.0, 'v': 0.0, 'pred': 0.0, 'ent': 0.0}

        adv_all = data['adv_out'][valid_indices]
        if adv_all.numel() > 1:
            adv_all = self._normalize_adv(adv_all)

        batch_size = max(1, int(getattr(self.cfg, 'outer_update_batch_size', 32)))
        perm = torch.randperm(total, device=self.device)
        pred_coef = float(getattr(self.cfg, 'outer_pred_coef', 0.1))

        loss_pi_sum = 0.0
        loss_v_sum = 0.0
        loss_pred_sum = 0.0
        entropy_sum = 0.0

        for start in range(0, total, batch_size):
            local_idx = perm[start:start + batch_size]
            idx = valid_indices[local_idx]
            weight = float(idx.numel()) / float(total)

            if isinstance(data['outer_state'], list):
                subset_list = [data['outer_state'][int(i)] for i in idx.detach().cpu().tolist()]
                state_subset = torch.stack(subset_list)
                if state_subset.dim() == 5 and state_subset.shape[1] == 1:
                    state_subset = state_subset.squeeze(1)
                state_subset = state_subset.to(self.device)
            else:
                state_subset = data['outer_state'][idx]

            weights_drift = data['weights_drift'][idx]

            # 同一批次只跑一次 LSTM/CAAN encoder，actor、critic、监督项复用 feat。
            feat = self.net.outer.encode(state_subset, weights_drift)
            dist = self.net.outer.get_dist(feat)
            old_raw = data['act_out_raw'][idx]
            new_logp = dist.log_prob(old_raw).sum(dim=1)
            entropy = dist.entropy().mean()
            ratio = torch.exp(torch.clamp(new_logp - data['logp_out'][idx], min=self.min_clip, max=self.max_clip))

            adv = adv_all[local_idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            loss_pi = -torch.min(surr1, surr2).mean()

            scores = torch.matmul(feat, self.net.outer.market_query.transpose(0, 1))
            attn = F.softmax(scores, dim=1)
            market_rep = torch.sum(feat * attn, dim=1)
            w_rep = self.net.outer.w_proj(weights_drift)
            val = self.net.outer.v_head(torch.cat([market_rep, w_rep], dim=-1))
            loss_v = self.mse_loss(val.squeeze(-1), data['ret_out'][idx])

            pred_return = self.net.outer.pred_head(feat).squeeze(-1)
            target_return = data['outer_stock_return_target'][idx].to(
                device=pred_return.device,
                dtype=pred_return.dtype,
            )
            loss_pred = F.smooth_l1_loss(pred_return, target_return)

            loss_total = loss_pi + self.vf_coef * loss_v + pred_coef * loss_pred - self.ent_coef * entropy
            (loss_total * weight).backward()

            loss_pi_sum += loss_pi.item() * idx.numel()
            loss_v_sum += loss_v.item() * idx.numel()
            loss_pred_sum += loss_pred.item() * idx.numel()
            entropy_sum += entropy.item() * idx.numel()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.net.outer.parameters()), self.max_grad_norm)

        self.opt_out.step()
        if (
                bool(getattr(self.cfg, "clear_cuda_cache_on_update", False))
                and getattr(self.device, "type", str(self.device)) == "cuda"
        ):
            torch.cuda.empty_cache()
        return {
            'pi': loss_pi_sum / total,
            'v': loss_v_sum / total,
            'pred': loss_pred_sum / total,
            'ent': entropy_sum / total,
        }

    def _update_monitor(self, data, mask):
        self.opt_mon.zero_grad()

        asset_state = self._select_outer_state_batch(data.get('outer_state'), mask)
        stats = self.net.mon.decision_stats(
            data['weights_drift'][mask],
            data['port_state'][mask],
            switch_action=data['act_out'][mask],
            asset_state=asset_state,
        )
        zeros = torch.zeros_like(stats["policy_logit"])
        logits = torch.stack([zeros, stats["policy_logit"]], dim=-1)
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()
        v = stats["value"]

        new_logp = dist.log_prob(data['act_mon'][mask])
        ratio = torch.exp(torch.clamp(new_logp - data['logp_mon'][mask], min=self.min_clip, max=self.max_clip))

        adv = data['adv_mon'][mask]
        if adv.numel() > 1:
            adv = self._normalize_adv(adv)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        loss_pi = -torch.min(surr1, surr2).mean()

        loss_v = self.mse_loss(v.squeeze(-1), data['ret_mon'][mask])

        loss_sup = logits.new_tensor(0.0)
        sup_coef = float(getattr(self.cfg, 'controller_sup_coef', getattr(self.cfg, 'monitor_sup_coef', 0.0)))
        if sup_coef > 0.0 and 'controller_switch_label' in data and 'controller_sup_weight' in data:
            labels = data['controller_switch_label'][mask].float().view(-1)
            weights = data['controller_sup_weight'][mask].float().view(-1)
            valid_sup = torch.isfinite(labels) & torch.isfinite(weights) & (weights > 0)
            if torch.count_nonzero(valid_sup).item() > 0:
                switch_logit = logits[:, 1] - logits[:, 0]
                raw_sup = F.binary_cross_entropy_with_logits(
                    switch_logit[valid_sup],
                    labels[valid_sup],
                    reduction='none',
                )
                loss_sup = (raw_sup * weights[valid_sup]).mean()

        loss_aux_return = logits.new_tensor(0.0)
        loss_aux_mdd = logits.new_tensor(0.0)
        aux_return_coef = float(getattr(self.cfg, 'controller_aux_return_coef', 0.0))
        aux_mdd_coef = float(getattr(self.cfg, 'controller_aux_mdd_coef', 0.0))
        if aux_return_coef > 0.0 and 'controller_hold_return_target' in data:
            target_return = data['controller_hold_return_target'][mask].to(
                device=logits.device,
                dtype=logits.dtype,
            ).view(-1)
            target_return = target_return * float(getattr(self.cfg, 'controller_aux_return_target_scale', 1.0))
            pred_return = stats["hold_return_pred"].view(-1)
            loss_aux_return = F.smooth_l1_loss(pred_return, target_return)
        if aux_mdd_coef > 0.0 and 'controller_hold_mdd_target' in data:
            target_mdd = data['controller_hold_mdd_target'][mask].to(
                device=logits.device,
                dtype=logits.dtype,
            ).view(-1)
            target_mdd = target_mdd * float(getattr(self.cfg, 'controller_aux_mdd_target_scale', 1.0))
            pred_mdd = stats["hold_risk_pred"].view(-1)
            loss_aux_mdd = F.smooth_l1_loss(pred_mdd, target_mdd)

        loss_total = (
            loss_pi
            + self.vf_coef * loss_v
            + sup_coef * loss_sup
            + aux_return_coef * loss_aux_return
            + aux_mdd_coef * loss_aux_mdd
            - self.ent_coef * entropy
        )
        loss_total.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.net.mon.parameters()), self.max_grad_norm)

        self.opt_mon.step()
        return {
            'pi': loss_pi.item(),
            'v': loss_v.item(),
            'sup': loss_sup.item(),
            'aux_return': loss_aux_return.item(),
            'aux_mdd': loss_aux_mdd.item(),
            'ent': entropy.item(),
        }
