import copy
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ppo, iTransformer
from utils.tools import EarlyStopping, adjust_learning_rate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings('ignore')


class TradingEnvironment:
    """
    Trading environment for dynamic portfolio management.
    Supports both simple and complex fee structures.
    """
    def __init__(self, args):
        self.args = args
        self.initial_amount = float(getattr(args, "initial_amount", 1.0))
        self.seq_len = self.args.seq_len
        self.fee_rate = self.args.fee_rate
        # For complex fee structure, sell fee (cs) is doubled.
        if args.complex_fee:
            self.cs = self.fee_rate * 2
            self.cp = self.fee_rate
        self.n_assets = self.args.num_stocks
        self.w_old = np.zeros(self.args.total_stocks)
        self.a0_old = 1.0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [self.w_old]
        self.transaction_cost_memory = []
        self.rollout = []
        self.rollout_len = args.rollout_len

    def reset(self):
        """Reset the environment to initial conditions."""
        self.rollout = []
        self.w_old = np.zeros(self.args.total_stocks)
        self.a0_old = 1.0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [self.w_old]
        self.transaction_cost_memory = []

    def normalization(self, actions):
        """Normalize an action vector so that its elements sum to 1."""
        s = np.sum(actions, axis=-1, keepdims=True)
        return actions / s

    def compute_mu_t(self, w_old, a_target, a0_old, cs, cp, max_iter=50, tol=1e-6):
        """
        Compute the scaling factor mu based on Jiang et al. (2017).
        Returns mu such that the new risk-asset allocation is mu * a_target.
        """
        mu = 1.0
        for _ in range(max_iter):
            sum_sell = 0.0
            for i in range(len(w_old)):
                diff = w_old[i] - mu * a_target[i]
                if diff > 0:
                    sum_sell += diff
            bracket = 1.0 - cp * a0_old - (cs + cp - cs * cp) * sum_sell
            denom = 1.0 - cp * a0_old
            mu_new = 0.0 if abs(denom) < 1e-12 else bracket / denom
            if abs(mu_new - mu) < tol:
                mu = mu_new
                break
            mu = mu_new
        return max(mu, 0.0)

    def get_sample(self, dataset, index, device):
        """
        Retrieve a sample from the dataset and convert it to torch tensors.
        """
        sample = dataset[index]
        (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) = sample
        return (torch.tensor(batch_x, dtype=torch.float).to(device),
                torch.tensor(batch_y, dtype=torch.float).to(device),
                torch.tensor(batch_x_mark, dtype=torch.float).to(device),
                torch.tensor(batch_y_mark, dtype=torch.float).to(device),
                torch.tensor(ground_true, dtype=torch.float).to(device))

    def step(self, weights, returns):
        """
        Execute one step in the environment using the provided weights and asset returns.
        For complex fee structure, adjust weights using mu-scaling; otherwise, update based on returns.
        Returns the one-step reward.
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        if self.args.complex_fee:
            mu = self.compute_mu_t(self.w_old, weights, self.a0_old, self.cs, self.cp, max_iter=100)
            w_new = mu * weights
            sum_wnew = np.sum(w_new)
            a0_new = max(1.0 - sum_wnew, 0.0)
            portfolio_return = np.sum(w_new * returns)
            new_pf_value = self.portfolio_value * (1 + portfolio_return)
            reward = (new_pf_value - self.portfolio_value) / self.portfolio_value
            self.portfolio_value = new_pf_value
            self.w_old = w_new
            self.a0_old = a0_new
            self.weights_memory.append((a0_new, w_new))
            self.portfolio_return_memory.append(portfolio_return)
            self.asset_memory.append(new_pf_value)
            return reward
        else:
            self.weights_memory.append(weights)
            portfolio_return = np.sum(weights * returns)
            change_ratio = returns + 1
            weights_new = self.normalization(weights * change_ratio)
            weights_old = self.weights_memory[-3] if len(self.weights_memory) >= 3 else weights
            diff_weights = np.sum(np.abs(weights_old - weights_new), axis=-1)
            transaction_fee = diff_weights * self.fee_rate * self.portfolio_value
            new_pf_value = (self.portfolio_value - transaction_fee) * (1 + portfolio_return)
            portfolio_return = (new_pf_value - self.portfolio_value) / self.portfolio_value
            reward = portfolio_return
            self.portfolio_value = new_pf_value
            self.portfolio_return_memory.append(portfolio_return)
            self.asset_memory.append(new_pf_value)
            return reward


class Exp_DeepAries(Exp_Basic):
    """
    Implements the DeepAries experiment using a PPO-based reinforcement learning framework
    for dynamic portfolio management.
    """
    def __init__(self, args, setting):
        super(Exp_DeepAries, self).__init__(args, setting)
        self.setting = setting
        self.old_model = ppo.ADA_PPO(args.model, args, setting, deterministic=False).to(self.device)
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.horizons = args.horizons
        self.temperature = args.temperature
        self.env = TradingEnvironment(self.args)
        self.buffer_size = 1
        self.ent_coef = 0.01
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.1
        self.beta = 0.1
        self.data = []
        self.max_clip = 5
        self.min_clip = -5

    def _build_model(self):
        model = ppo.ADA_PPO(self.args.model, self.args, self.setting, deterministic=False)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def put_data(self, item):
        self.data.append(item)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return optim.Adam(trainable_params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        train_dataset, _ = self._get_data(flag='train')
        n_data = len(train_dataset)
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.logger.info(f"[Train] Epoch {epoch + 1}/{self.args.train_epochs}")
            self.model.train()
            self.env.reset()
            i = 0
            train_steps = 0
            epoch_loss = []
            epoch_time = time.time()

            while i < n_data and (
                self.args.max_train_steps is None or train_steps < self.args.max_train_steps
            ):
                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = \
                    self.env.get_sample(train_dataset, i, self.device)
                dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
                scores, log_prob, entropy, selected_period_indices = self.old_model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                final_weights = torch.zeros_like(scores)
                final_weights[top_indices] = topk_weights
                returns = ground_true[:, selected_period_indices]

                simulated_returns = torch.tensor(
                    [(final_weights * ground_true[:, i]).sum() for i in range(len(self.horizons))],
                    device=ground_true.device
                )
                optimal_horizon_index = torch.argmax(simulated_returns)
                bonus_ratio = 0.2
                reward = self.env.step(final_weights, returns)
                if optimal_horizon_index.item() == selected_period_indices.item():
                    reward *= (1 + bonus_ratio)
                else:
                    reward *= (1 - bonus_ratio)

                chosen_horizon = self.horizons[selected_period_indices]
                next_i = i + chosen_horizon
                done = (next_i >= n_data - 1)
                if not done:
                    next_sample = self.env.get_sample(train_dataset, next_i, self.device)
                    next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, _ = next_sample
                else:
                    next_batch_x = torch.zeros_like(batch_x)
                    next_batch_y = torch.zeros_like(batch_y)
                    next_batch_x_mark = torch.zeros_like(batch_x_mark)
                    next_batch_y_mark = torch.zeros_like(batch_y_mark)

                transition = (
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                    scores.detach(), reward,
                    next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                    log_prob.detach(),
                    done,
                    returns,
                )
                self.env.rollout.append(transition)
                i = next_i
                train_steps += 1
                if len(self.env.rollout) >= self.env.rollout_len:
                    self.put_data(self.env.rollout)
                    self.env.rollout = []
                    loss_val = self.train_net(K_epoch=1, model_optim=model_optim)
                    self.old_model.load_state_dict(self.model.state_dict())
                    if loss_val is not None:
                        epoch_loss.append(loss_val)
                    torch.cuda.empty_cache()
            vali_loss, vali_reward, test_loss, test_reward = self.vali()
            avg_train_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            torch.cuda.empty_cache()
            self.logger.info(
                f"[Epoch {epoch + 1}] TrainLoss={avg_train_loss:.4f}, "
                f"ValiLoss={vali_loss:.4f}, TestLoss={test_loss:.4f}"
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopped!")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_path = os.path.join(path, 'checkpoint.pth')
        # Load checkpoint with device mapping (CPU-compatible)
        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.logger.info("Training completed, best model loaded.")

    def vali(self):
        vali_dataset, _ = self._get_data('val')
        v_loss, v_reward = self._evaluate_dataset(vali_dataset)
        test_dataset, _ = self._get_data('test')
        t_loss, t_reward = self._evaluate_dataset(test_dataset)
        return v_loss, v_reward, t_loss, t_reward

    def _evaluate_dataset(self, dataset):
        self.model.eval()
        env_temp = TradingEnvironment(self.args)
        env_temp.reset()
        total_loss, total_value_loss, total_pred_loss = 0.0, 0.0, 0.0
        n_data = len(dataset)
        i = 0
        total_reward = 0.0

        while i < n_data:
            batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = \
                self.env.get_sample(dataset, i, self.device)
            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
            with torch.no_grad():
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                final_weights = torch.zeros_like(scores)
                final_weights[top_indices] = topk_weights
                returns = ground_true[:, selected_period_indices]
                reward = env_temp.step(final_weights, returns)
                chosen_horizon = self.horizons[selected_period_indices]
                i += chosen_horizon
                total_reward += reward
                if i < n_data:
                    next_sample = self.env.get_sample(dataset, i, self.device)
                    next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, _ = next_sample
                else:
                    next_batch_x = torch.zeros_like(batch_x)
                    next_batch_y = torch.zeros_like(batch_y)
                    next_batch_x_mark = torch.zeros_like(batch_x_mark)
                    next_batch_y_mark = torch.zeros_like(batch_y_mark)
                value = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                next_value = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)
                td_target = reward + self.gamma * next_value * (0 if i >= n_data - 1 else 1)
                value_loss = F.smooth_l1_loss(value, td_target)
                pred_loss = F.smooth_l1_loss(scores, returns)
                loss = value_loss + self.beta * pred_loss

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_pred_loss += pred_loss.item()
        avg_loss = total_loss
        self.logger.info(
            f"[Validation] Total Reward={total_reward:.4f}, Total Loss={avg_loss:.4f}, "
            f"Value Loss={total_value_loss:.4f}, Pred Loss={total_pred_loss:.4f}"
        )
        self.model.train()
        return avg_loss, total_reward

    def backtest(self, setting, load=False):
        if load:
            best_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            
            # Check if checkpoint file exists
            if not os.path.exists(best_path):
                error_msg = (
                    f"Checkpoint file not found: {best_path}\n"
                    f"Please ensure:\n"
                    f"  1. The checkpoint directory '{setting}' exists in '{self.args.checkpoints}'\n"
                    f"  2. The checkpoint.pth file exists in that directory\n"
                    f"  3. Or specify the correct checkpoint directory using --checkpoint_dir argument"
                )
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            try:
                # Load checkpoint with device mapping (CPU-compatible)
                # If trained on GPU, automatically maps to current device (CPU or GPU)
                checkpoint = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.logger.info(f"Successfully loaded checkpoint from {best_path} on device: {self.device}")
            except Exception as e:
                error_msg = (
                    f"Failed to load checkpoint from {best_path}\n"
                    f"Error: {str(e)}\n"
                    f"This might be due to:\n"
                    f"  1. Model architecture mismatch (check enc_in, dec_in, d_model, etc.)\n"
                    f"  2. Corrupted checkpoint file\n"
                    f"  3. Incompatible PyTorch version"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        backtest_dataset, _ = self._get_data('backtest')
        n_data = len(backtest_dataset)
        self.model.eval()
        env_test = TradingEnvironment(self.args)
        env_test.reset()
        portfolio_values = [env_test.portfolio_value]
        portfolio_dates = [backtest_dataset.unique_dates[0]]
        decision_dates = []
        decision_end_dates = []
        horizon_memory = []
        rebalance_returns = []
        weight_rows = []
        i = 0
        while i < n_data:
            batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = \
                self.env.get_sample(backtest_dataset, i, self.device)
            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
            with torch.no_grad():
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
            # Retrieve current date from the dataset based on sequence length.
            current_pos = i + self.args.seq_len
            current_date = backtest_dataset.unique_dates[current_pos]
            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
            selected_scores = scores[top_indices]
            topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
            final_weights = torch.zeros_like(scores)
            final_weights[top_indices] = topk_weights
            returns = ground_true[:, selected_period_indices]
            selected_idx = int(selected_period_indices.item()) if hasattr(selected_period_indices, "item") else int(selected_period_indices)
            chosen_horizon = int(self.horizons[selected_idx])
            horizon_end_pos = min(current_pos + chosen_horizon, len(backtest_dataset.unique_dates) - 1)
            horizon_end_date = backtest_dataset.unique_dates[horizon_end_pos]
            reward = env_test.step(final_weights, returns)
            decision_dates.append(current_date)
            decision_end_dates.append(horizon_end_date)
            horizon_memory.append(int(chosen_horizon))
            rebalance_returns.append(float(reward))
            weight_rows.append(final_weights.detach().cpu().numpy())
            portfolio_values.append(env_test.portfolio_value)
            portfolio_dates.append(horizon_end_date)
            i += chosen_horizon

        final_pf = env_test.portfolio_value
        self.logger.info(f"[BackTest] Rebalance-horizon Portfolio Value = {final_pf:.4f}")

        tickers = backtest_dataset.df.index.get_level_values("tic").unique().tolist()
        if weight_rows:
            decision_weights_df = pd.DataFrame(
                np.vstack(weight_rows),
                index=pd.to_datetime(decision_dates),
                columns=tickers,
            )
        else:
            decision_weights_df = pd.DataFrame(columns=tickers)

        daily_returns = []
        daily_weight_rows = []
        daily_weight_dates = []
        daily_start_date = pd.to_datetime(decision_dates[0]) if decision_dates else pd.to_datetime(portfolio_dates[0])
        daily_portfolio_dates = [daily_start_date]
        daily_portfolio_values = [env_test.initial_amount]
        raw_path = os.path.join(self.args.root_path, self.args.data_path)
        close_panel = None
        if weight_rows and os.path.exists(raw_path):
            raw_df = pd.read_csv(raw_path)
            price_col = "adjclose" if "adjclose" in raw_df.columns else "close" if "close" in raw_df.columns else None
            if price_col is not None and {"date", "tic"}.issubset(raw_df.columns):
                raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.tz_localize(None)
                raw_df = raw_df[raw_df["tic"].isin(tickers)]
                close_panel = (
                    raw_df.pivot(index="date", columns="tic", values=price_col)
                    .sort_index()
                    .reindex(columns=tickers)
                    .ffill()
                    .dropna(how="any")
                )

        if close_panel is not None and not close_panel.empty:
            close_dates = close_panel.index
            for dec_date, end_date, weights in zip(pd.to_datetime(decision_dates), pd.to_datetime(decision_end_dates), weight_rows):
                start_loc = int(close_dates.searchsorted(dec_date, side="left"))
                end_loc = int(close_dates.searchsorted(end_date, side="right")) - 1
                if start_loc < 0 or end_loc <= start_loc or start_loc >= len(close_dates):
                    continue
                end_loc = min(end_loc, len(close_dates) - 1)
                weights = np.asarray(weights, dtype=np.float64)
                for loc in range(start_loc, end_loc):
                    current_close = close_panel.iloc[loc].to_numpy(dtype=np.float64)
                    next_close = close_panel.iloc[loc + 1].to_numpy(dtype=np.float64)
                    ret_vec = next_close / (current_close + 1e-12) - 1.0
                    day_ret = float(np.sum(weights * ret_vec))
                    daily_returns.append(day_ret)
                    daily_weight_dates.append(close_dates[loc])
                    daily_weight_rows.append(weights)
                    daily_portfolio_values.append(daily_portfolio_values[-1] * (1.0 + day_ret))
                    daily_portfolio_dates.append(close_dates[loc + 1])

        if daily_returns:
            portfolio_values = daily_portfolio_values
            portfolio_dates = daily_portfolio_dates
            final_pf = portfolio_values[-1]
            weights_df = pd.DataFrame(daily_weight_rows, index=pd.to_datetime(daily_weight_dates), columns=tickers)
            metric_returns = np.asarray(daily_returns, dtype=np.float64)
            metric_frequency = "daily_close_to_close"
        else:
            weights_df = decision_weights_df.copy()
            metric_returns = np.asarray(rebalance_returns, dtype=np.float64)
            metric_frequency = "rebalance_fallback"

        # Compute portfolio metrics
        initial_pf = portfolio_values[0]
        total_return = (final_pf - initial_pf) / initial_pf

        # Convert portfolio_dates to pandas Timestamps (if not already)
        dates = pd.to_datetime(portfolio_dates)
        days = (dates[-1] - dates[0]).days
        annualized_return = ((final_pf / initial_pf) ** (365 / days) - 1) if days > 0 else 0.0

        # Calculate maximum drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (np.array(portfolio_values) - cummax) / cummax
        max_drawdown = drawdowns.min()
        if len(metric_returns) > 1:
            periods_per_year = 252.0 if metric_frequency == "daily_close_to_close" else len(metric_returns) / max(days / 365.0, 1e-12)
            ann_return_mean = metric_returns.mean() * periods_per_year
            ann_vol = metric_returns.std() * np.sqrt(periods_per_year)
            sharpe = ann_return_mean / ann_vol if ann_vol > 0 else np.nan
        else:
            ann_return_mean = 0.0
            ann_vol = 0.0
            sharpe = np.nan

        # Log and print portfolio performance metrics
        self.logger.info(f"\n=========== 回测结果: {setting}===========")
        self.logger.info(f"初始组合价值: {initial_pf:.2f}")
        self.logger.info(f"最终组合价值: {final_pf:.2f}")
        self.logger.info("\n====== 测试区间绩效 ======")
        self.logger.info(f"累计收益: {total_return:.2%}")
        self.logger.info(f"年化收益: {ann_return_mean:.2%}")
        self.logger.info(f"年化收益(CAGR): {annualized_return:.2%}")
        self.logger.info(f"年化波动率: {ann_vol:.2%}")
        self.logger.info(f"夏普比率: {sharpe:.2f}")
        self.logger.info(f"最大回撤: {abs(max_drawdown):.2%}")
        self.logger.info("=================================\n")

        print(f"Final Portfolio Value: {final_pf:.4f}")
        print(f"累计收益: {total_return:.2%}")
        print(f"年化收益: {ann_return_mean:.2%}")
        print(f"年化收益(CAGR): {annualized_return:.2%}")
        print(f"年化波动率: {ann_vol:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"最大回撤: {abs(max_drawdown):.2%}")

        result_dir = os.path.join(self.args.results_root, setting)
        os.makedirs(result_dir, exist_ok=True)
        actions_dir = os.path.join(result_dir, "actions")
        os.makedirs(actions_dir, exist_ok=True)

        portfolio_df = pd.DataFrame({
            "date": pd.to_datetime(portfolio_dates),
            "portfolio_value": portfolio_values,
        })
        portfolio_df["period_return"] = portfolio_df["portfolio_value"].pct_change().fillna(0.0)
        portfolio_df.to_csv(os.path.join(result_dir, "portfolio_performance.csv"), index=False)

        decision_df = pd.DataFrame({
            "date": pd.to_datetime(decision_dates),
            "end_date": pd.to_datetime(decision_end_dates),
            "chosen_horizon": horizon_memory,
            "period_return": rebalance_returns,
            "portfolio_value": env_test.asset_memory[1:],
        })
        decision_df.to_csv(os.path.join(result_dir, "rebalance_decisions.csv"), index=False)

        decision_weights_df.to_csv(os.path.join(result_dir, "rebalance_weights.csv"))
        weights_df.to_csv(os.path.join(actions_dir, "test.csv"))
        weights_df.to_csv(os.path.join(result_dir, "daily_weights.csv"))

        metrics = {
            "initial_value": float(initial_pf),
            "final_value": float(final_pf),
            "cumulative_return": float(total_return),
            "annualized_return": float(ann_return_mean),
            "annualized_return_cagr": float(annualized_return),
            "annualized_return_mean": float(ann_return_mean),
            "annualized_volatility": float(ann_vol),
            "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
            "max_drawdown": float(max_drawdown),
            "num_rebalances": int(len(rebalance_returns)),
            "metric_frequency": metric_frequency,
            "start_date": str(pd.to_datetime(portfolio_dates[0]).date()),
            "end_date": str(pd.to_datetime(portfolio_dates[-1]).date()),
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(result_dir, "backtest_metrics.csv"), index=False)
        pd.Series(metrics).to_json(
            os.path.join(result_dir, "backtest_metrics.json"),
            force_ascii=False,
            indent=2,
        )

        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(portfolio_dates), portfolio_values, label="Cumulative Returns")
        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "cumulative_returns.png"), bbox_inches="tight")
        plt.close()

        if not weights_df.empty:
            plt.figure(figsize=(15, 6))
            weights_df.plot(kind="area", stacked=True, title="Asset Allocation Over Time", figsize=(15, 6))
            plt.xlabel("Date")
            plt.ylabel("Weight")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "asset_allocation.png"), bbox_inches="tight")
            plt.close()

        return final_pf

    def make_batch(self):
        batch_data = {
            "batch_x": [],
            "batch_y": [],
            "batch_x_mark": [],
            "batch_y_mark": [],
            "scores": [],
            "reward": [],
            "next_batch_x": [],
            "next_batch_y": [],
            "next_batch_x_mark": [],
            "next_batch_y_mark": [],
            "log_prob": [],
            "done": [],
            "return_data": []
        }
        for _ in range(self.buffer_size):
            rollout = self.data.pop(0)
            for transition in rollout:
                (batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward,
                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, log_prob, done, return_data) = transition
                batch_data["batch_x"].append(batch_x)
                batch_data["batch_y"].append(batch_y)
                batch_data["batch_x_mark"].append(batch_x_mark)
                batch_data["batch_y_mark"].append(batch_y_mark)
                batch_data["scores"].append(scores)
                batch_data["reward"].append([reward])
                batch_data["next_batch_x"].append(next_batch_x)
                batch_data["next_batch_y"].append(next_batch_y)
                batch_data["next_batch_x_mark"].append(next_batch_x_mark)
                batch_data["next_batch_y_mark"].append(next_batch_y_mark)
                batch_data["log_prob"].append(log_prob)
                done_mask = 0 if done else 1
                batch_data["done"].append([done_mask])
                batch_data["return_data"].append(return_data)
        for key in batch_data:
            try:
                if key in ["reward", "done"]:
                    batch_data[key] = torch.tensor(batch_data[key], dtype=torch.float).squeeze().to(self.device)
                else:
                    batch_data[key] = torch.stack(batch_data[key], dim=0).to(self.device)
            except Exception as e:
                print(f"Error processing key {key}: {e}")
        mini_batches = []
        for i in range(len(batch_data["batch_x"])):
            mini_batches.append((
                batch_data["batch_x"][i],
                batch_data["batch_y"][i],
                batch_data["batch_x_mark"][i],
                batch_data["batch_y_mark"][i],
                batch_data["scores"][i],
                batch_data["reward"][i],
                batch_data["next_batch_x"][i],
                batch_data["next_batch_y"][i],
                batch_data["next_batch_x_mark"][i],
                batch_data["next_batch_y_mark"][i],
                batch_data["done"][i],
                batch_data["log_prob"][i],
                batch_data["return_data"][i]
            ))
        return mini_batches

    def calc_advantage(self, data):
        data_with_adv = []
        td_target_lst = []
        delta_lst = []
        for i, transition in enumerate(data):
            batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward, next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, done_mask, old_log_prob, return_data = transition
            with torch.no_grad():
                v_s_next = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)
                v_s = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                td_target = reward + self.gamma * v_s_next * done_mask
                delta = td_target - v_s
            td_target_lst.append(td_target.item())
            delta_lst.append(delta.item())
        advantage_lst = []
        running_adv = 0.0
        for delta_t in reversed(delta_lst):
            running_adv = self.gamma * self.lmbda * running_adv + delta_t
            advantage_lst.append(running_adv)
        advantage_lst.reverse()
        advantage_tensor = torch.tensor(advantage_lst, dtype=torch.float, device=self.device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
        for i, transition in enumerate(data):
            (batch_x, batch_y, batch_x_mark, batch_y_mark,
             scores, reward,
             next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
             done_mask, old_log_prob, return_data) = transition
            td_target_tensor = torch.tensor(td_target_lst[i], dtype=torch.float, device=self.device)
            adv_val = advantage_tensor[i]
            data_with_adv.append((
                batch_x, batch_y, batch_x_mark, batch_y_mark,
                scores, reward,
                next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                done_mask, old_log_prob,
                td_target_tensor, adv_val,
                return_data
            ))
        return data_with_adv

    def train_net(self, K_epoch=10, model_optim=None):
        if len(self.data) == self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            losses = 0.0
            for _ in range(K_epoch):
                for mini_batch in data:
                    (batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward,
                     next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                     done_mask, old_log_prob, td_target, advantage, return_data) = mini_batch
                    scores, log_prob, total_entropy, _ = self.model.pi(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=self.min_clip, max=self.max_clip))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    entropy_mean = total_entropy
                    pred_loss = F.smooth_l1_loss(self.model.pred(batch_x, batch_x_mark, batch_y, batch_y_mark), return_data)
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.smooth_l1_loss(self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark), td_target)
                    loss = self.beta * pred_loss + policy_loss + value_loss - self.ent_coef * entropy_mean
                    losses += loss.mean().item()
                    model_optim.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
            return losses

    def _get_market_index_name(self):
        market_indices = {
            'kospi': '^KS11',
            'dj30': '^DJI',
            'sp500': '^GSPC',
            'nasdaq': '^IXIC',
            'csi300': '000300.SS',
            'ftse': '^FTSE',
        }
        index_name = market_indices.get(self.args.market)
        if not index_name:
            raise ValueError(f"Unsupported market: {self.args.market}")
        return index_name

    def simulate_episode(self):
        self.env.reset()
        done = False
        total_reward = 0.0
        dataset, _ = self._get_data('train')
        i = 0
        n_data = len(dataset)
        while not done and i < n_data:
            batch = self.env.get_sample(dataset, i, self.device)
            batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = batch
            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
            with torch.no_grad():
                scores, _, _, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
            selected_scores = scores[top_indices]
            topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
            final_weights = torch.zeros_like(scores)
            final_weights[top_indices] = topk_weights
            returns = ground_true[:, selected_period_indices]
            reward = self.env.step(final_weights, returns)
            total_reward += reward
            chosen_horizon = self.horizons[selected_period_indices]
            i += chosen_horizon
            if i >= n_data - 1:
                done = True
        return total_reward
