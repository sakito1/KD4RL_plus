import numpy as np
import pandas as pd
import gym
import torch
import os
import random
import warnings
import utils.config as config
from utils.PriceMatrix import process_files

warnings.filterwarnings("ignore")


class PPO_Env(gym.Env):
    def __init__(self,
                 dataset: dict = None,
                 outer_window: int = None,
                 inner_window: int = None,
                 max_hold: int = None,
                 min_hold: int = None,
                 episode_len: int = None,
                 train_date_range: tuple = None,
                 val_date_range: tuple = None,
                 test_date_range: tuple = None,
                 initial_amount: int = None,
                 transaction_cost_pct: float = None,
                 risk_gamma: float = 5.0,
                 train_episodes_per_epoch: int = None,
                 train_start_stride_days: int = None,
                 train_episode_count: int = None,
                 train_episode_start_stride: int = None,
                 train_episode_to_end: bool = None,
                 cun_file: str = None,
                 logger=None):

        super(PPO_Env, self).__init__()

        dataset = config.dataset if dataset is None else dataset
        outer_window = config.outer_window if outer_window is None else outer_window
        inner_window = config.inner_window if inner_window is None else inner_window
        max_hold = config.max_hold if max_hold is None else max_hold
        min_hold = getattr(config, 'min_hold', 20) if min_hold is None else min_hold
        episode_len = config.episode_len if episode_len is None else episode_len
        train_date_range = (config.train_start_date, config.train_end_date) if train_date_range is None else train_date_range
        val_date_range = (config.valid_start_date, config.valid_end_date) if val_date_range is None else val_date_range
        test_date_range = (config.test_start_date, config.test_end_date) if test_date_range is None else test_date_range
        initial_amount = config.initial_amount if initial_amount is None else initial_amount
        transaction_cost_pct = config.TRANSACTION_COST_RATE if transaction_cost_pct is None else transaction_cost_pct
        if train_episodes_per_epoch is None:
            train_episodes_per_epoch = train_episode_count
        if train_start_stride_days is None:
            train_start_stride_days = train_episode_start_stride
        if train_episodes_per_epoch is None:
            train_episodes_per_epoch = getattr(config, 'train_episodes_per_epoch', getattr(config, 'train_episode_count', 5))
        if train_start_stride_days is None:
            train_start_stride_days = getattr(config, 'train_start_stride_days', getattr(config, 'train_episode_start_stride', 5))
        train_episode_to_end = getattr(config, 'train_episode_to_end', False) if train_episode_to_end is None else train_episode_to_end

        self.dataset = dataset
        self.outer_window = outer_window
        self.inner_window = inner_window
        self.max_hold = max_hold
        self.min_hold = int(min_hold)
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_gamma = risk_gamma
        self.logger = logger
        self.cun_path = cun_file if cun_file else "./results"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stride = 200
        self.episode_len = episode_len
        self.train_episodes_per_epoch = max(1, int(train_episodes_per_epoch))
        self.train_start_stride_days = max(1, int(train_start_stride_days))
        self.train_episode_count = self.train_episodes_per_epoch
        self.train_episode_start_stride = self.train_start_stride_days
        self.train_episode_to_end = bool(train_episode_to_end)
        self.train_episode_stop_idx = None
        self.train_ptr = 0
        self.completed_train_epoch_count = 0
        self.current_episode_start = 0
        self.current_episode_end = 0
        self.current_episode_len = int(episode_len)

        # Reward scaling (global)
        self.reward_scale_portfolio = float(getattr(config, 'reward_scale_portfolio', 10.0))
        self.reward_scale_base = float(getattr(config, 'reward_scale_base', 10.0))
        self.reward_scale_outer = float(getattr(config, 'reward_scale_outer', 10.0))
        self.reward_scale_inner = float(getattr(config, 'reward_scale_inner', 1000.0))
        self.reward_scale_monitor = float(getattr(config, 'reward_scale_monitor', 10.0))
        self.reward_scale_controller = float(getattr(config, 'reward_scale_controller', self.reward_scale_monitor))
        self.controller_sup_horizon = int(getattr(config, 'controller_sup_horizon', self.min_hold))
        self.controller_sup_scale = float(getattr(config, 'controller_sup_scale', 0.05))
        self.controller_sup_max_weight = float(getattr(config, 'controller_sup_max_weight', 5.0))
        self.controller_sup_coef = float(getattr(config, 'controller_sup_coef', getattr(config, 'monitor_sup_coef', 0.0)))
        self.controller_sup_enabled = self.controller_sup_coef > 0.0

        self.monitor_reward_mode = getattr(config, 'monitor_reward_mode', 'mean')

        self.inner_max_boundary = getattr(config, 'inner_max_boundary', 0.2)

        if not os.path.exists(self.cun_path):
            os.makedirs(self.cun_path)

        # =====================================================================
        # =====================================================================
        if self.logger: self.logger.info("正在加载全量数据以保证边界连续性...")

        loaded_data = process_files(
            self.dataset['ssm_data_path'],
            self._load_stock_list(self.dataset['stocks_path']),
            self.dataset['features_name']
        )

        self.all_dates = pd.to_datetime(loaded_data['dates'])
        self.total_days = len(self.all_dates)
        self.num_stocks = loaded_data['data'].shape[0]
        self.feat_names = self.dataset.get('features_name', [])
        self.ohlc_indices = [
            i for i, c in enumerate(self.feat_names)
            if any(x in c.lower() for x in ['adjopen', 'adjhigh', 'adjlow', 'adjclose'])
        ]
        self.amount_indices = [
            i for i, c in enumerate(self.feat_names)
            if 'amount' in c.lower()
        ]
        self.price_indices = [
            i for i, c in enumerate(self.feat_names)
            if any(x in c.lower() for x in ['adjopen', 'adjhigh', 'adjlow', 'adjclose', 'amount'])
        ]
        self.adjclose_index = next(
            (i for i, c in enumerate(self.feat_names) if c.lower() == 'adjclose'),
            self.ohlc_indices[-1] if len(self.ohlc_indices) > 0 else None,
        )
        scaled_indices = set(self.price_indices)
        self.other_indices = [i for i in range(len(self.feat_names)) if i not in scaled_indices]
        self.inner_norm_mode = str(getattr(config, 'inner_norm_mode', 'legacy')).lower()
        self.features = torch.tensor(loaded_data['data'], dtype=torch.float32, device=self.device)
        self.prices = torch.tensor(loaded_data['prices'], dtype=torch.float32, device=self.device)

        self.cdj_price = self.prices[:, :, 1]

        self.ratio = self.cdj_price[:, 1:] / (self.cdj_price[:, :-1] + 1e-8)

        ssm = loaded_data['ssm']
        self.h_tensor = torch.tensor(ssm['h'], dtype=torch.float32, device=self.device)
        self.z_tensor = torch.tensor(ssm['z'], dtype=torch.float32, device=self.device)
        self.p_tensor = torch.tensor(ssm['p'], dtype=torch.float32, device=self.device)
        self.q_bear_tensor = torch.tensor(ssm['q_bear'], dtype=torch.float32, device=self.device)
        self.q_bull_tensor = torch.tensor(ssm['q_bull'], dtype=torch.float32, device=self.device)

        self.idx_map = {
            'train': self._get_indices_range(train_date_range),
            'val': self._get_indices_range(val_date_range),
            'test': self._get_indices_range(test_date_range)
        }

        self._precompute_stats()

        self.mode = 'train'
        self.day = 0
        self.stop_step = 0

        self.portfolio_value = torch.tensor(self.initial_amount, dtype=torch.float32, device=self.device)
        self.t_held = 0
        self.peak_value = float(self.initial_amount)
        self.segment_init_value = float(self.initial_amount)
        self.cumulative_alpha = 0.0
        self.cumulative_risk = 0.0

        self.monitor_reward_mode = getattr(config, 'monitor_reward_mode', 'mean')

    @staticmethod
    def _normalize(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s = w.sum()
        if torch.isnan(s) or torch.isinf(s) or s.abs() < eps:
            n = w.numel()
            return torch.ones_like(w) / float(n)
        return w / (s + eps)

    @staticmethod
    def _load_stock_list(path):
        with open(path) as f:
            return [i.strip() for i in f.readlines()]

    def _episode_ratio_limit(self):
        return min(int(getattr(self, 'stop_step', self.total_days - 2)), self.ratio.shape[1])

    def _get_indices_range(self, date_range, use_stride=False):
        start_date, end_date = pd.to_datetime(date_range)
        start_idx = np.searchsorted(self.all_dates, start_date)
        end_idx = np.searchsorted(self.all_dates, end_date, side='right')

        min_start = max(self.outer_window, self.inner_window)
        start_idx = max(start_idx, min_start)
        end_idx = min(end_idx, self.total_days)

        if start_idx >= end_idx:
            return []
        return list(range(start_idx, end_idx))

    def _precompute_stats(self):
        num_stocks, total_days, num_feats = self.features.shape
        self.rolling_means = torch.zeros_like(self.features)
        self.rolling_stds = torch.zeros_like(self.features)

        for t in range(self.outer_window, self.total_days):
            start = t - self.outer_window
            end = t
            window_data = self.features[:, start+1:end+1, :]
            self.rolling_means[:, t, :] = window_data.mean(dim=1)
            self.rolling_stds[:, t, :] = window_data.std(dim=1) + 1e-8

    def set_mode(self, mode_name):
        if mode_name not in self.idx_map:
            raise ValueError(f"Unknown mode: {mode_name}")
        self.mode = mode_name

        if self.mode == 'train':
            raw_indices = self.idx_map['train']
            if not raw_indices: return
            pool_signature = (
                self.train_episode_to_end,
                self.train_episodes_per_epoch,
                self.train_start_stride_days,
                self.episode_len,
                raw_indices[0],
                raw_indices[-1],
            )
            if getattr(self, '_train_pool_signature', None) != pool_signature:
                if self.train_episode_to_end:
                    train_start = raw_indices[0]
                    train_end = raw_indices[-1]
                    candidate_starts = [
                        train_start + k * self.train_start_stride_days
                        for k in range(self.train_episodes_per_epoch)
                    ]
                    self.train_indices_pool = [t for t in candidate_starts if t < train_end]
                    if not self.train_indices_pool:
                        self.train_indices_pool = [train_start]
                    self.train_episode_stop_idx = train_end
                else:
                    self.train_indices_pool = self._build_fixed_train_pool(
                        raw_indices,
                        total_days=self.total_days,
                        episode_len=self.episode_len,
                        stride_days=self.train_start_stride_days,
                    )
                    random.shuffle(self.train_indices_pool)
                    self.train_episode_stop_idx = None
                self.train_ptr = 0
                self.completed_train_epoch_count = 0
                self._train_pool_signature = pool_signature
                if self.logger:
                    if self.train_episode_to_end:
                        start_dates = [str(self.all_dates[t].date()) for t in self.train_indices_pool]
                        self.logger.info(
                            "Mode TRAIN. DeepAries-style train epochs: "
                            f"{len(self.train_indices_pool)} starts/epoch, stride_days={self.train_start_stride_days}, "
                            f"end={self.all_dates[self.train_episode_stop_idx].date()}, starts={start_dates}"
                        )
                    else:
                        self.logger.info(f"Mode TRAIN. Pool Size: {len(self.train_indices_pool)}")
        else:
            if self.logger:
                self.logger.info(f"Mode {mode_name.upper()}. Sequential.")

    def reset(self):
        indices = self.idx_map[self.mode]
        if not indices: raise ValueError(f"No indices for mode {self.mode}")

        if self.mode == 'train':
            if self.train_ptr >= len(self.train_indices_pool):
                self.train_ptr = 0
                self.completed_train_epoch_count += 1
                if not self.train_episode_to_end:
                    random.shuffle(self.train_indices_pool)
                    if self.logger: self.logger.info(f"Completed train epoch {self.completed_train_epoch_count}; reshuffled pool.")
                elif self.logger:
                    self.logger.info(
                        f"Completed train epoch {self.completed_train_epoch_count}; restarting fixed train episode starts."
                    )
            self.day = self.train_indices_pool[self.train_ptr]
            self.train_ptr += 1
            if self.train_episode_to_end:
                self.stop_step = int(self.train_episode_stop_idx)
            else:
                self.stop_step = self.day + self.episode_len
        else:
            self.day = indices[0]
            self.stop_step = indices[-1]

        self.current_episode_start = int(self.day)
        self.current_episode_end = int(self.stop_step)
        self.current_episode_len = max(1, self.current_episode_end - self.current_episode_start)

        self.portfolio_value = torch.tensor(self.initial_amount, dtype=torch.float32, device=self.device)
        init_w = torch.ones(self.num_stocks, dtype=torch.float32, device=self.device) / self.num_stocks
        self.prev_weights = init_w.clone()
        self.prev_base_weight = init_w.clone()

        self.t_held = 0
        self.peak_value = self.initial_amount
        self.segment_init_value = self.initial_amount
        self.cumulative_alpha = 0.0
        self.cumulative_risk = 0.0

        return self._get_observation()

    @staticmethod
    def _build_fixed_train_pool(raw_indices, total_days: int, episode_len: int, stride_days: int):
        """Build fixed-length train episode starts with a configurable trading-day stride."""
        absolute_limit = int(total_days) - 2
        episode_len = max(1, int(episode_len))
        stride_days = max(1, int(stride_days))
        valid_starts = [int(t) for t in raw_indices if int(t) + episode_len <= absolute_limit]
        return valid_starts[::stride_days]

    def reset_at(self, start_idx: int, stop_idx: int = None):
        """Reset to an explicit absolute day window without advancing train pointers."""
        start_idx = int(start_idx)
        if stop_idx is None:
            stop_idx = start_idx + int(self.episode_len)
        stop_idx = int(stop_idx)
        if start_idx < self.outer_window:
            raise ValueError(f"start_idx={start_idx} is before outer_window={self.outer_window}")
        if start_idx >= self.total_days - 2:
            raise ValueError(f"start_idx={start_idx} is outside available data.")
        stop_idx = min(stop_idx, self.total_days - 2)
        if stop_idx <= start_idx:
            raise ValueError(f"stop_idx={stop_idx} must be after start_idx={start_idx}")

        self.day = start_idx
        self.stop_step = stop_idx
        self.current_episode_start = int(self.day)
        self.current_episode_end = int(self.stop_step)
        self.current_episode_len = max(1, self.current_episode_end - self.current_episode_start)

        self.portfolio_value = torch.tensor(self.initial_amount, dtype=torch.float32, device=self.device)
        init_w = torch.ones(self.num_stocks, dtype=torch.float32, device=self.device) / self.num_stocks
        self.prev_weights = init_w.clone()
        self.prev_base_weight = init_w.clone()

        self.t_held = 0
        self.peak_value = self.initial_amount
        self.segment_init_value = self.initial_amount
        self.cumulative_alpha = 0.0
        self.cumulative_risk = 0.0

        return self._get_observation()


    # =====================================================================
    # =====================================================================
    def _calc_future_sharpe(self, weights: torch.Tensor, start_day: int, horizon: int = 60) -> torch.Tensor:
        """计算给定权重在未来 horizon 天内 Buy & Hold 的年化夏普比率。

        逻辑：模拟实际资金曲线，而非每日再平衡。
        """
        max_h = min(horizon, self._episode_ratio_limit() - start_day)
        if max_h <= 1:
            return torch.tensor(0.0, device=self.device)

        # shape: [Num_Stocks, max_h]
        future_ratios = self.ratio[:, start_day: start_day + max_h]

        w_init = self._normalize(weights).view(-1, 1)

        asset_growth = torch.cumprod(future_ratios, dim=1)

        # shape: [max_h]
        # value_t = sum(w_i * growth_i_t)
        port_value_curve = torch.sum(w_init * asset_growth, dim=0)

        initial_val = torch.tensor([1.0], device=self.device)
        full_curve = torch.cat([initial_val, port_value_curve])

        # Ret_t = Val_t / Val_{t-1} - 1
        daily_rets = full_curve[1:] / (full_curve[:-1] + 1e-8) - 1.0

        mean_ret = daily_rets.mean()
        std_ret = daily_rets.std(unbiased=False)

        sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)

        return torch.clamp(sharpe, -10.0, 10.0)
    @staticmethod
    def _calc_outer_step_reward(portfolio_return: torch.Tensor, risk_gamma: float) -> torch.Tensor:
        """Outer step reward 与现有逻辑保持一致：portfolio_return - risk_gamma * (min(portfolio_return, 0)^2)."""
        downside = torch.clamp(portfolio_return, max=0.0)
        step_risk_penalty = risk_gamma * (downside ** 2)
        return portfolio_return - step_risk_penalty

    def _future_outer_sum(self, w0: torch.Tensor, start_day: int, horizon: int) -> torch.Tensor:
        """从 start_day 开始，按 buy-and-hold 持有 horizon 天的“纯收益率”(log return) 近似。

        monitor reward 专用：只比较收益率，不引入风险惩罚。
        """
        if horizon <= 0:
            return torch.tensor(0.0, device=self.device)

        max_h = min(horizon, self._episode_ratio_limit() - start_day)
        if max_h <= 0:
            return torch.tensor(0.0, device=self.device)

        w = self._normalize(w0)
        g = torch.prod(self.ratio[:, start_day:start_day + max_h], dim=1)
        port_g = torch.sum(w * g)
        return torch.log(port_g + 1e-12)

    def _future_stock_return_target(self, start_day: int, horizon: int) -> torch.Tensor:
        """DeepAries-style pred target: each stock's future cumulative return."""
        if horizon <= 0:
            return torch.zeros(self.num_stocks, dtype=torch.float32, device=self.device)

        max_h = min(horizon, self._episode_ratio_limit() - start_day)
        if max_h <= 0:
            return torch.zeros(self.num_stocks, dtype=torch.float32, device=self.device)

        growth = torch.prod(self.ratio[:, start_day:start_day + max_h], dim=1)
        return torch.log(growth.clamp_min(1e-8))

    def _future_portfolio_return_and_max_drawdown(self, weights, start_day: int, horizon: int):
        if horizon <= 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return zero, zero

        max_h = min(horizon, self._episode_ratio_limit() - start_day)
        if max_h <= 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return zero, zero

        w = self._normalize(weights)
        growth_path = torch.cumprod(self.ratio[:, start_day:start_day + max_h], dim=1)
        portfolio_path = torch.sum(w.unsqueeze(1) * growth_path, dim=0).clamp_min(1e-8)
        values = torch.cat([portfolio_path.new_ones(1), portfolio_path])
        running_peak = torch.cummax(values, dim=0).values.clamp_min(1e-8)
        drawdown = (running_peak - values) / running_peak
        return torch.log(values[-1].clamp_min(1e-8)), drawdown.max()

    # =====================================================================
    # =====================================================================
    def step(self, real_weight, base_weight=None, outer_action=None, is_switch: bool = False):
        """step 函数参数说明：

        - real_weight: Inner 调整后的最终权重（用于计算本步实际收益）
        - base_weight: Agent 决定使用的新基准（Switch 时为 Outer Action，Hold 时为 Drifted Base）
        - outer_action: Outer Actor 建议的候选动作（Monitor Reward 暂不使用，但保留接口）
        - is_switch: Monitor 是否执行了 Switch 动作
        """

        if self.day >= self.stop_step:
            return self._get_observation(), 0.0, True, {}

        r_past = self.ratio[:, self.day - 1]  # t-1 -> t
        r_future = self.ratio[:, self.day]  # t -> t+1

        # =====================================================================
        # =====================================================================

        current_holdings_drift = self._normalize(self.prev_base_weight * r_past)
        remaining_hold_horizon = max(1, int(self.max_hold) - int(self.t_held))
        controller_hold_return_target, controller_hold_mdd_target = self._future_portfolio_return_and_max_drawdown(
            current_holdings_drift.detach(),
            self.day,
            remaining_hold_horizon,
        )

        # =====================================================================
        # =====================================================================

        final_weights = real_weight.flatten().detach()
        new_base_weight = base_weight.flatten().detach()

        if is_switch:
            self.t_held = 1
            self.segment_init_value = self.portfolio_value.item()
            self.cumulative_risk = 0.0
            self.cumulative_alpha = 0.0
        else:
            self.t_held += 1

        # =====================================================================
        # =====================================================================
        weight_drifted = self.prev_weights * r_past
        val_t1_raw = self.portfolio_value * weight_drifted.sum()

        weight_drifted_norm = weight_drifted / (weight_drifted.sum() + 1e-8)
        turnover = torch.sum(torch.abs(final_weights - weight_drifted_norm))
        cost = val_t1_raw * turnover * self.transaction_cost_pct
        cost_rate = cost / (val_t1_raw + 1e-8)

        val_t1_net = val_t1_raw - cost

        # =====================================================================
        # =====================================================================
        portfolio_gross = (
            torch.sum(final_weights * r_future).clamp_min(1e-8)
            * (val_t1_net / (val_t1_raw + 1e-8)).clamp_min(1e-8)
        )
        portfolio_return = torch.log(portfolio_gross.clamp_min(1e-8))
        base_return_val = torch.log(torch.sum(new_base_weight * r_future).clamp_min(1e-8))
        outer_return_log = portfolio_return
        outer_stock_return_target = self._future_stock_return_target(self.day, self.max_hold)
        inner_stock_return_target = self._future_stock_return_target(self.day, 1)

        if self.controller_sup_enabled:
            candidate_switch_weight = (
                outer_action.flatten().detach()
                if outer_action is not None
                else new_base_weight
            )
            sup_h = max(1, int(self.controller_sup_horizon))
            hold_20 = self._future_outer_sum(weight_drifted_norm.detach(), self.day, sup_h)
            switch_20 = self._future_outer_sum(candidate_switch_weight.detach(), self.day, sup_h)
            switch_turnover = torch.sum(torch.abs(self._normalize(candidate_switch_weight) - weight_drifted_norm.detach()))
            switch_adv_20 = switch_20 - hold_20 - switch_turnover * self.transaction_cost_pct
            controller_switch_label = (switch_adv_20 > 0).float()
            controller_sup_weight = torch.clamp(
                torch.abs(switch_adv_20) / max(self.controller_sup_scale, 1e-8),
                0.0,
                self.controller_sup_max_weight,
            )
        else:
            hold_20 = portfolio_return.new_tensor(0.0)
            switch_20 = portfolio_return.new_tensor(0.0)
            switch_adv_20 = portfolio_return.new_tensor(0.0)
            controller_switch_label = portfolio_return.new_tensor(0.0)
            controller_sup_weight = portfolio_return.new_tensor(0.0)

        step_alpha = portfolio_return - base_return_val
        # =====================================================================
        # =====================================================================
        step_outer_raw = outer_return_log
        step_monitor = portfolio_return * self.reward_scale_controller

        # =====================================================================
        # =====================================================================
        self.peak_value = max(self.peak_value, val_t1_net.item())
        self.cumulative_alpha += step_alpha.item()

        downside = torch.clamp(portfolio_return, max=0.0)
        self.cumulative_risk += (self.risk_gamma * (downside ** 2)).item()

        self.portfolio_value = val_t1_net
        self.prev_weights = final_weights
        self.prev_base_weight = new_base_weight

        self.day += 1

        info = {
            'rewards': {
                'portfolio_return': portfolio_return.item() * self.reward_scale_portfolio,
                'outer_step_reward': step_outer_raw.item(),
                'outer_return_rate': outer_return_log.item(),
                # 'inner_reward': portfolio_return.item() * self.reward_scale_portfolio,
                'inner_reward': step_alpha.item() * self.reward_scale_inner,
                'monitor_reward': step_monitor.item(),
                'base_return': base_return_val.item() * self.reward_scale_base,
                'cost_rate': cost_rate.item(),
            },
            'portfolio_value': val_t1_net.item(),
            'outer_stock_return_target': outer_stock_return_target.detach(),
            'inner_stock_return_target': inner_stock_return_target.detach(),
            'inner_next_return_target': inner_stock_return_target.detach(),
            'controller_hold_return_target': controller_hold_return_target.detach(),
            'controller_hold_mdd_target': controller_hold_mdd_target.detach(),
            'controller_switch_label': controller_switch_label.detach(),
            'controller_sup_weight': controller_sup_weight.detach(),
            'controller_switch_adv_20': switch_adv_20.detach(),
            'controller_hold_20': hold_20.detach(),
            'controller_switch_20': switch_20.detach(),
            'date': self.all_dates[self.day]
        }

        done = (self.day >= self.stop_step) or (self.day >= self.total_days - 2)
        return self._get_observation(), 0.0, done, info

    def _get_observation(self):
        t = self.day
        if t >= self.p_tensor.shape[1]:
            t = self.p_tensor.shape[1] - 1

        ssm_dict = {
            'z': self.z_tensor[:, t, :].unsqueeze(0),
            'h': self.h_tensor[:, t, :].unsqueeze(0),
            'p': self.p_tensor[:, t].unsqueeze(0),
            'q_bear': self.q_bear_tensor[:, t].unsqueeze(0),
            'q_bull': self.q_bull_tensor[:, t].unsqueeze(0)
        }
        if self.day - 1 >= 0:
            r_past = self.ratio[:, self.day - 1]  # (t-1 -> t)
        else:
            r_past = torch.ones(self.num_stocks, dtype=torch.float32, device=self.device)

        weights_drift = self._normalize(self.prev_weights * r_past)  # w_t^drift
        base_drift = self._normalize(self.prev_base_weight * r_past)  # b_t^drift

        current_p = self.p_tensor[:, t]
        held_p = torch.sum(current_p * weights_drift)

        outer_state, _ = self.get_outer_state(t)
        inner_state, _ = self.get_inner_state(t)
        port_state = self._calc_port_state()

        return {
            'ssm': ssm_dict,
            'outer_state': outer_state.unsqueeze(0),
            'inner_state': inner_state.unsqueeze(0),
            'weights_drift': weights_drift.unsqueeze(0),
            'base_drift': base_drift.unsqueeze(0),
            'port_state': port_state,
            'held_p': held_p.unsqueeze(0)
        }

    def get_outer_state(self, t):
        start = t - self.outer_window+1
        end = t+1
        raw_feat = self.features[:, start:end, :]
        mean = self.rolling_means[:, t, :].unsqueeze(1)
        std = self.rolling_stds[:, t, :].unsqueeze(1)
        norm_state = (raw_feat - mean) / (std + 1e-8)
        return norm_state, self.h_tensor[:, start:end, :]

    def get_inner_state(self, t):
        start = t - self.inner_window + 1
        end = t + 1
        raw_feat = self.features[:, start:end, :]

        norm_state = torch.zeros_like(raw_feat)

        if self.inner_norm_mode == 'legacy':
            if len(self.price_indices) > 0:
                p_idx = self.price_indices
                last_val = raw_feat[:, -1, p_idx].unsqueeze(1)
                norm_state[:, :, p_idx] = raw_feat[:, :, p_idx] / (last_val + 1e-8)
            if len(self.other_indices) > 0:
                norm_state[:, :, self.other_indices] = raw_feat[:, :, self.other_indices]
            return norm_state, self.h_tensor[:, start:end, :]

        if len(self.ohlc_indices) > 0:
            p_idx = self.ohlc_indices
            if self.adjclose_index is not None:
                last_close = raw_feat[:, -1, self.adjclose_index].view(-1, 1, 1)
                norm_state[:, :, p_idx] = torch.log(
                    (raw_feat[:, :, p_idx] / (last_close + 1e-8)).clamp_min(1e-8)
                )
            else:
                last_val = raw_feat[:, -1, p_idx].unsqueeze(1)
                norm_state[:, :, p_idx] = torch.log(
                    (raw_feat[:, :, p_idx] / (last_val + 1e-8)).clamp_min(1e-8)
                )

        def _window_zscore(x):
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            return ((x - mean) / std).clamp(-5.0, 5.0)

        if len(self.amount_indices) > 0:
            a_idx = self.amount_indices
            amount = torch.log1p(raw_feat[:, :, a_idx].clamp_min(0.0))
            norm_state[:, :, a_idx] = _window_zscore(amount)

        if len(self.other_indices) > 0:
            o_idx = self.other_indices
            norm_state[:, :, o_idx] = _window_zscore(raw_feat[:, :, o_idx])

        return norm_state, self.h_tensor[:, start:end, :]

    def _calc_port_state(self):
        time_norm = self.t_held / float(self.max_hold)
        current_val = self.portfolio_value.item()

        peak = self.peak_value if self.peak_value > 1e-8 else 1.0
        drawdown = (peak - current_val) / peak

        init_val = self.segment_init_value if self.segment_init_value > 1e-8 else 1.0
        seg_return = (current_val - init_val) / init_val

        cost_feat = self.transaction_cost_pct * 100

        state = torch.tensor([
            time_norm, drawdown, seg_return,
            self.cumulative_alpha, self.cumulative_risk, cost_feat
        ], dtype=torch.float32, device=self.device)
        return state.unsqueeze(0)


