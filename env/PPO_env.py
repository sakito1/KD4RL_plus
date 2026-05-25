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
                 episode_len: int = None,
                 train_date_range: tuple = None,
                 val_date_range: tuple = None,
                 test_date_range: tuple = None,
                 initial_amount: int = None,
                 transaction_cost_pct: float = None,
                 risk_gamma: float = 5.0,
                 cun_file: str = None,
                 logger=None):

        super(PPO_Env, self).__init__()

        dataset = config.dataset if dataset is None else dataset
        outer_window = config.outer_window if outer_window is None else outer_window
        inner_window = config.inner_window if inner_window is None else inner_window
        max_hold = config.max_hold if max_hold is None else max_hold
        episode_len = config.episode_len if episode_len is None else episode_len
        train_date_range = (config.train_start_date, config.train_end_date) if train_date_range is None else train_date_range
        val_date_range = (config.valid_start_date, config.valid_end_date) if val_date_range is None else val_date_range
        test_date_range = (config.test_start_date, config.test_end_date) if test_date_range is None else test_date_range
        initial_amount = config.initial_amount if initial_amount is None else initial_amount
        transaction_cost_pct = config.TRANSACTION_COST_RATE if transaction_cost_pct is None else transaction_cost_pct

        self.dataset = dataset
        self.outer_window = outer_window
        self.inner_window = inner_window
        self.max_hold = max_hold
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_gamma = risk_gamma
        self.logger = logger
        self.cun_path = cun_file if cun_file else "./results"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stride = 200
        self.episode_len = episode_len

        # Reward scaling (global)
        self.reward_scale_portfolio = float(getattr(config, 'reward_scale_portfolio', 10.0))
        self.reward_scale_base = float(getattr(config, 'reward_scale_base', 10.0))
        self.reward_scale_outer = float(getattr(config, 'reward_scale_outer', 10.0))
        self.reward_scale_inner = float(getattr(config, 'reward_scale_inner', 1000.0))
        self.reward_scale_monitor = float(getattr(config, 'reward_scale_monitor', 10.0))

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
        self.price_indices = [
            i for i, c in enumerate(self.feat_names)
            if any(x in c.lower() for x in ['adjopen', 'adjhigh', 'adjlow', 'adjclose', 'amount'])
        ]
        self.other_indices = [i for i in range(len(self.feat_names)) if i not in self.price_indices]
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

    def _get_indices_range(self, date_range, use_stride=False):
        start_date, end_date = pd.to_datetime(date_range)
        start_idx = np.searchsorted(self.all_dates, start_date)
        end_idx = np.searchsorted(self.all_dates, end_date)

        min_start = max(self.outer_window, self.inner_window)
        start_idx = max(start_idx, min_start)
        max_end = self.total_days - 2
        end_idx = min(end_idx, max_end)

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
            if not raw_indices:
                raise ValueError("No indices for mode train")
            absolute_limit = self.total_days - 2
            train_end_exclusive = raw_indices[-1] + 1
            stop_limit = min(train_end_exclusive, absolute_limit)
            valid_starts = [t for t in raw_indices if t + self.episode_len <= stop_limit]
            if not valid_starts:
                raise ValueError(
                    "Train date range is too short for one full episode: "
                    f"available_steps={len(raw_indices)}, episode_len={self.episode_len}"
                )
            self.train_indices_pool = valid_starts[::self.stride]
            random.shuffle(self.train_indices_pool)
            self.train_ptr = 0
            if self.logger:
                self.logger.info(f"Mode TRAIN. Pool Size: {len(self.train_indices_pool)}")
        else:
            if self.logger:
                self.logger.info(f"Mode {mode_name.upper()}. Sequential.")

    def reset(self):
        indices = self.idx_map[self.mode]
        if not indices: raise ValueError(f"No indices for mode {self.mode}")

        if self.mode == 'train':
            if self.train_ptr >= len(self.train_indices_pool):
                random.shuffle(self.train_indices_pool)
                self.train_ptr = 0
                if self.logger: self.logger.info("Epoch reshuffle.")
            self.day = self.train_indices_pool[self.train_ptr]
            self.train_ptr += 1
            self.stop_step = self.day + self.episode_len
        else:
            self.day = indices[0]
            self.stop_step = indices[-1]

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
        max_h = min(horizon, self.ratio.shape[1] - start_day)
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

        max_h = min(horizon, self.ratio.shape[1] - start_day)
        if max_h <= 0:
            return torch.tensor(0.0, device=self.device)

        w = self._normalize(w0)
        g = torch.prod(self.ratio[:, start_day:start_day + max_h], dim=1)
        port_g = torch.sum(w * g)
        return torch.log(port_g + 1e-12)

    # =====================================================================
    # =====================================================================
    def step(self, real_weight, base_weight=None, outer_action=None, is_switch: bool = False):
        """step 函数参数说明：

        - real_weight: Inner 调整后的最终权重（用于计算本步实际收益）
        - base_weight: Agent 决定使用的新基准（Switch 时为 Outer Action，Hold 时为 Drifted Base）
        - outer_action: Outer Actor 建议的候选动作（保留接口用于记录/兼容）
        - is_switch: Monitor 是否执行了 Switch 动作

        Monitor reward is assigned by the trainer from the outer critic's
        hold-versus-switch counterfactual values.  The environment must not
        look into future prices to label today's monitor action.
        """

        if self.day >= self.stop_step:
            return self._get_observation(), 0.0, True, {}

        r_past = self.ratio[:, self.day - 1]  # t-1 -> t
        r_future = self.ratio[:, self.day]  # t -> t+1

        # =====================================================================
        # =====================================================================

        step_monitor = torch.tensor(0.0, dtype=torch.float32, device=self.device)

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
        portfolio_return = torch.log(torch.sum(final_weights * r_future) * val_t1_net / val_t1_raw)
        base_return_val = torch.log(torch.sum(new_base_weight * r_future))

        step_alpha = portfolio_return - base_return_val
        # =====================================================================
        # =====================================================================
        if is_switch:
            sharpe = self._calc_future_sharpe(new_base_weight, self.day, horizon=self.max_hold)
            benchmark_weights = torch.ones(self.num_stocks, device=self.device) / self.num_stocks
            sharpe_benchmark = self._calc_future_sharpe(benchmark_weights, self.day, horizon=self.max_hold)
            step_outer_raw = sharpe - sharpe_benchmark
        else:
            step_outer_raw = torch.tensor(0.0, device=self.device)
        # step_outer_raw = self._calc_outer_step_reward(base_return_val, self.risk_gamma)

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
                # 'outer_step_reward': portfolio_return.item() * self.reward_scale_outer,
                # 'inner_reward': portfolio_return.item() * self.reward_scale_portfolio,
                'inner_reward': step_alpha.item() * self.reward_scale_inner,
                'monitor_reward': step_monitor.item(),
                'base_return': base_return_val.item() * self.reward_scale_base,
                'cost_rate': cost_rate.item(),
            },
            'portfolio_value': val_t1_net.item(),
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

        if len(self.price_indices) > 0:
            p_idx = self.price_indices
            last_val = raw_feat[:, -1, p_idx].unsqueeze(1)
            norm_state[:, :, p_idx] = raw_feat[:, :, p_idx] / (last_val + 1e-8)
        norm_state[:, :, self.other_indices] = raw_feat[:, :, self.other_indices]

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

