import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class HRL_Buffer:
    """
    [Simplified] Hierarchical PPO Buffer for Single-Episode Updates

    Simplified logic:
    1. Stores data for exactly one episode.
    2. 'finish_episode' calculates GAE using the entire buffer.
    3. 'get_batch' returns the stacked data.
    4. Must be cleared after every update.
    """

    def __init__(self, capacity, device, gamma=0.99, gae_lambda=0.95):
        self.device = device
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        # Initialize storage lists
        self.data = {
            'ssm': [], 'outer_state': [], 'inner_state': [],
            'port_state': [], 'weights_drift': [], 'base_drift': [],
            'base_used': [], 'weights_exec': [],
            'act_mon': [], 'logp_mon': [],
            'act_out_raw': [], 'act_out': [], 'logp_out': [],
            'act_inn_raw': [], 'logp_inn': [],
            'val_mon': [], 'val_out': [], 'val_inn': [],
            'rew_mon': [], 'rew_alpha': [], 'rew_outer_raw': [],
            'is_switch': [], 'is_locked': [], 'dones': [],
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
                if k == 'ssm' and isinstance(v, dict):
                    # Store dict references (detached)
                    val = {sk: sv.detach() for sk, sv in v.items()}
                else:
                    # Store tensors (detached)
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

            segment_rew = torch.sum(rew_out[start:end])

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

            if k == 'ssm':
                # Reconstruct SSM dictionary of tensors
                # v is a list of dicts: [{z:.., h:..}, {z:.., h:..}]
                first_key = list(v[0].keys())[0]  # e.g. 'z'
                res['ssm'] = {}
                for sk in v[0].keys():
                    # stack (T, 1, D) -> squeeze -> (T, D)
                    res['ssm'][sk] = torch.stack([x[sk] for x in v]).squeeze(1)

            elif k == 'outer_state':
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

        self.mse_loss = nn.MSELoss()
        self.min_clip = -5
        self.max_clip = 5

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
        elif mode == 'monitor':
            for p in self.net.mon.parameters():
                p.requires_grad = True

    def get_action(self, obs, mode='train', force_switch=None, force_inner_zero=False, force_locked=False):
        is_train = (mode == 'train')

        weight_drift = obs['weights_drift']
        base_drift = obs['base_drift']

        # =========================
        # Outer
        # =========================
        act_out, act_out_raw, logp_out, _, _ = self.net.outer.pi(
            obs['outer_state'],
            weight_drift,
            deterministic=(not is_train)
        )
        val_out = self.net.outer.value(obs['outer_state'], weight_drift)

        # =========================
        # Monitor
        # =========================
        act_mon_samp, logp_mon_samp, _, _ = self.net.mon.pi(
            obs['ssm']['z'], obs['ssm']['h'], obs['ssm']['p'],
            obs['ssm']['q_bear'], obs['ssm']['q_bull'],
            weight_drift, obs['port_state'], deterministic=(not is_train)
        )
        val_mon = self.net.mon.value(
            obs['ssm']['z'], obs['ssm']['h'], obs['ssm']['p'],
            obs['ssm']['q_bear'], obs['ssm']['q_bull'],
            weight_drift, obs['port_state']
        )

        if force_switch is not None:
            B = weight_drift.shape[0]
            act_mon = torch.full((B,), force_switch, dtype=torch.long, device=self.device)
            logp_mon = logp_mon_samp if is_train else torch.zeros((B,), device=self.device)
            is_locked_val = 1 if force_locked else 0
            is_locked = torch.full((B,), is_locked_val, dtype=torch.long, device=self.device)
        else:
            act_mon = act_mon_samp
            logp_mon = logp_mon_samp
            is_locked = torch.zeros_like(act_mon)

        has_switched = (act_mon == 1)
        base_used = torch.where(has_switched.unsqueeze(-1).bool(), act_out, base_drift)

        # =========================
        # Inner
        # =========================
        if force_inner_zero:
            B, N = weight_drift.shape
            act_inn_raw = torch.zeros((B, N), device=self.device)
            logp_inn = torch.zeros((B,), device=self.device)
            val_inn = self.net.inner.value(obs['inner_state'], base_used, weight_drift)
            weights_exec = base_used
        else:
            # 直接生成可执行权重（weights_exec），同时返回 score_sample 作为 PPO 的“原始动作变量”
            # 这样：执行动作稳定可控，训练仍然符合 PPO（ratio 用 score_sample 的 logp）
            if hasattr(self.net.inner, "build_inner_action_simple"):
                alpha = float(getattr(self.cfg, "inner_max_boundary", 1.0))
                w_new, score_sample, logp_inn, _, val_inn = self.net.inner.build_inner_action_simple(
                    obs['inner_state'],
                    base_used,
                    weight_drift,
                    alpha=alpha,
                    deterministic=(not is_train),
                )
                weights_exec = w_new
                act_inn_raw = score_sample
            else:
                # 兼容旧实现：用 raw_signal 生成最终权重
                act_inn_raw, logp_inn, _ = self.net.inner.pi(
                    obs['inner_state'],
                    base_used,
                    weight_drift,
                    deterministic=(not is_train)
                )
                val_inn = self.net.inner.value(obs['inner_state'], base_used, weight_drift)

                action_for_calc = torch.clamp(act_inn_raw, -3.0, 3.0)
                adjusted = base_used * torch.exp(action_for_calc * self.cfg.inner_max_boundary)
                weights_exec = adjusted / (adjusted.sum(dim=1, keepdim=True) + 1e-12)

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
            'weights_exec': weights_exec
        }

    def update(self, buffer_data, phase='joint'):
        bs = int(len(buffer_data.get('rew_mon', [])))
        if bs <= 0:
            return {}

        ppo_epochs = int(getattr(self.cfg, 'ppo_epochs', 1))
        metrics = {}

        for _ in range(ppo_epochs):
            if phase in ['warmup_inner', 'joint', 'round_inner']:
                l_inn = self._update_inner(buffer_data)
                self._log_metric(metrics, 'inn', l_inn)

            if phase in ['warmup_monitor', 'joint', 'round_monitor']:
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
        self.opt_inn.zero_grad()

        # 用“旧动作”计算 new_logp(a_old)，避免重新采样导致 PPO ratio 失真
        feat = self.net.inner.encode(
            data['inner_state'],
            data['base_used'],
            data['weights_drift'],
        )
        dist = self.net.inner.get_dist(feat)
        new_logp = dist.log_prob(data['act_inn_raw']).sum(dim=1)  # [B]
        entropy = dist.entropy().mean()  # scalar
        ratio = torch.exp(torch.clamp(new_logp - data['logp_inn'], min=self.min_clip, max=self.max_clip))

        adv = data['adv_inn']
        adv = self._normalize_adv(adv)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        loss_pi = -torch.min(surr1, surr2).mean()

        val = self.net.inner.value(
            data['inner_state'],
            data['base_used'],
            data['weights_drift']
        )
        loss_v = self.mse_loss(val.squeeze(-1), data['ret_inn'])

        loss_total = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy

        def _param_stats(params):
            with torch.no_grad():
                flat = torch.cat([p.data.view(-1) for p in params if p.requires_grad])
                return flat.norm().item(), flat.abs().mean().item()

        def _param_delta(params, old):
            with torch.no_grad():
                flat = torch.cat([p.data.view(-1) for p in params if p.requires_grad])
                d = flat - old
                return d.norm().item(), d.abs().mean().item()

        params = [p for p in self.net.inner.parameters() if p.requires_grad]
        w_norm0, w_mean0 = _param_stats(params)
        w_flat0 = torch.cat([p.data.view(-1) for p in params]).clone()

        # backward + step ...
        loss_total.backward()
        self.opt_inn.step()

        w_norm1, w_mean1 = _param_stats(params)
        d_norm, d_mean = _param_delta(params, w_flat0)

        print(f"[INNER] w_norm {w_norm0:.4e}->{w_norm1:.4e} | Δnorm {d_norm:.4e} Δmean {d_mean:.4e}")
        return {'pi': loss_pi.item(), 'v': loss_v.item(), 'ent': entropy.item()}

    def _update_outer(self, data, mask):
        self.opt_out.zero_grad()

        if isinstance(data['outer_state'], list):
            valid_indices = torch.where(mask)[0].cpu().tolist()
            subset_list = [data['outer_state'][i] for i in valid_indices]
            state_subset = torch.stack(subset_list)
            if state_subset.dim() == 5 and state_subset.shape[1] == 1:
                state_subset = state_subset.squeeze(1)
            state_subset = state_subset.to(self.device)
        else:
            state_subset = data['outer_state'][mask]

        # 用“旧动作”计算 new_logp(a_old)，避免重新采样导致 PPO ratio 失真
        feat = self.net.outer.encode(state_subset, data['weights_drift'][mask])
        dist = self.net.outer.get_dist(feat)
        old_raw = data['act_out_raw'][mask]
        new_logp = dist.log_prob(old_raw).sum(dim=1)  # [B_mask]
        entropy = dist.entropy().mean()
        ratio = torch.exp(torch.clamp(new_logp - data['logp_out'][mask], min=self.min_clip, max=self.max_clip))

        adv = data['adv_out'][mask]
        # 只对有效外层决策点归一化（mask 后的向量）
        if adv.numel() > 1:
            adv = self._normalize_adv(adv)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        loss_pi = -torch.min(surr1, surr2).mean()

        val = self.net.outer.value(
            state_subset,
            data['weights_drift'][mask]
        )
        loss_v = self.mse_loss(val.squeeze(-1), data['ret_out'][mask])

        loss_total = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy
        loss_total.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.net.outer.parameters()), self.max_grad_norm)

        self.opt_out.step()
        return {'pi': loss_pi.item(), 'v': loss_v.item(), 'ent': entropy.item()}

    def _update_monitor(self, data, mask):
        self.opt_mon.zero_grad()

        ssm = {k: v[mask] for k, v in data['ssm'].items()}

        feat = self.net.mon.encode(
            ssm['z'], ssm['h'], ssm['p'], ssm['q_bear'], ssm['q_bull'],
            data['weights_drift'][mask],
            data['port_state'][mask],
        )
        logits = self.net.mon.actor_head(feat)
        dist = Categorical(logits=logits)

        new_logp = dist.log_prob(data['act_mon'][mask])
        entropy = dist.entropy().mean()
        ratio = torch.exp(torch.clamp(new_logp - data['logp_mon'][mask], min=self.min_clip, max=self.max_clip))

        adv = data['adv_mon'][mask]
        if adv.numel() > 1:
            adv = self._normalize_adv(adv)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        loss_pi = -torch.min(surr1, surr2).mean()

        v = self.net.mon.v_head(feat)
        loss_v = self.mse_loss(v.squeeze(-1), data['ret_mon'][mask])

        loss_total = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy
        loss_total.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.net.mon.parameters()), self.max_grad_norm)

        self.opt_mon.step()
        return {'pi': loss_pi.item(), 'v': loss_v.item(), 'ent': entropy.item()}
