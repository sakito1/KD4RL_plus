import torch
import numpy as np
from typing import List, Any
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.draw_process import (load_prices, load_weights, compute_asset_contributions, plot_price_and_contrib)
import random
from utils import config
from utils.PriceMatrix import alphastock_files
import warnings
import os
from AlphaStock.AlphaStock import LSTMDRL
import math
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")


def apply_transaction_costs(gross_returns, turnovers, transaction_cost_rate=None):
    """Deduct proportional one-way transaction costs from portfolio returns."""
    rate = (
        config.TRANSACTION_COST_RATE
        if transaction_cost_rate is None
        else float(transaction_cost_rate)
    )
    gross = np.asarray(gross_returns, dtype=float)
    turnover = np.asarray(turnovers, dtype=float)
    if gross.shape != turnover.shape:
        raise ValueError("gross_returns and turnovers must have identical shapes")
    return gross - turnover * rate


def rebalance_turnovers(weights, prices, decision_offsets):
    """Return L1 turnover at each decision, measured from drifted holdings."""
    weights = np.asarray(weights, dtype=float)
    prices = np.asarray(prices, dtype=float)
    turnovers = np.zeros(weights.shape[0], dtype=float)
    for offset in decision_offsets:
        offset = int(offset)
        if offset < 0 or offset >= len(turnovers):
            continue
        target = weights[offset]
        if offset == 0:
            current = np.zeros_like(target)
        else:
            relatives = prices[:, offset] / prices[:, offset - 1]
            current = weights[offset - 1] * relatives
            current = current / (current.sum() + 1e-12)
        turnovers[offset] = np.abs(target - current).sum()
    return turnovers


class BasedDataLoader:
    def __init__(self, sample_list: List):
        self.original_samples = sample_list.copy()  # 保存原始列表
        self.num_samples = len(sample_list)

    def shuffle_dataset(self):
        """打乱当前 epoch 的样本顺序"""
        random.shuffle(self.original_samples)

    def generate_batches(self, batch_size: int, shuffle_batch: bool = True):
        """
        生成批次数据，每个 epoch 会先打乱整体顺序

        参数:
            batch_size: 每个批次的大小
            shuffle_batch: 是否在批次内部再次打乱元素

        返回:
            生成器，每次返回一个批次的数据
        """
        # 每次调用时都重新打乱样本顺序
        self.shuffle_dataset()

        if batch_size <= 0:
            raise ValueError("批次大小必须为正数")

        # 计算总批次数量
        num_batches = (self.num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.num_samples)
            batch = self.original_samples[start_idx:end_idx]
            yield batch


class alphastock_main:
    def __init__(self, cun_file, logger):
        self.cun_file = os.path.join(cun_file, 'AlphaStock')
        if not os.path.exists(self.cun_file):
            os.makedirs(self.cun_file)
        self.logger = logger
        """基于投资组合优化的强化学习环境（继承自OpenAI Gym）"""
        alphastock_param = config.alphastcok
        self.look_back = alphastock_param['look_back']
        self.step_size = alphastock_param['step_size']
        self.num_epoch = alphastock_param['num_epoch']
        self.batch_size = alphastock_param['batch_size']
        self.num_steps = alphastock_param['num_steps']
        self.model_param = alphastock_param['model_param']

        dataset = config.dataset  # 输入数据集对象

        # 生成训练集的部分
        self.start_train_date = config.train_start_date
        self.end_train_date = config.train_end_date
        self.initial_amount = config.initial_amount
        self.transaction_cost_pct = float(config.TRANSACTION_COST_RATE)
        # 回测的部分，这部分需要按照
        self.start_valid_date = config.valid_start_date
        self.end_valid_date = config.valid_end_date
        self.start_test_date = config.test_start_date
        self.end_test_date = config.test_end_date

        with open(dataset['stocks_path']) as f:
            self.stocks = [i.strip() for i in f.readlines()]
        stocks_info = ', '.join(self.stocks)
        logger.info(f"alphastock环境加载资产数据：资产数量 {len(self.stocks)}，资产列表 {stocks_info}")
        # 特征相关配置

        features_name = dataset['features_name']  # 特征列名

        self.model_param['in_features'] = len(features_name)
        # 加载关键数据，返回array格式的数据
        self.data_tensor, self.all_dates, _, _, self.prices = \
        alphastock_files(dataset['feature_path'],self.stocks,features_name)
        self.prices = torch.tensor(self.prices, dtype=torch.float32)
        self.adjc = self.prices[:, :, 1]
        self.features = self.data_tensor

        # 生成训练集的序列。（特征可能有点多了）
        self.sample_list = self.get_train_index(self.start_train_date, self.end_train_date)
        # 打印数据形状信息
        logger.info(f"环境加载资产数据：价格形状 {self.prices.shape}, 训练样本数 {len(self.sample_list)}")

        # ======== 验证集 step 列表（用于模型选择） ========
        if self.start_valid_date is not None and self.end_valid_date is not None:
            self.valid_step_list, self.valid_dates = self.get_test_index(self.start_valid_date, self.end_valid_date)
            logger.info(f"验证集时间区间: {self.start_valid_date} ~ {self.end_valid_date}, "
                        f"step 数量 {len(self.valid_step_list)}")
        else:
            self.valid_step_list, self.valid_dates = None, None
            logger.info("未配置 valid_start_date / valid_end_date，跳过验证集模型选择。")

        # 生成测试step列表
        self.test_step_list, self.test_dates = self.get_test_index(self.start_test_date, self.end_test_date)

        # 打印数据形状信息
        logger.info(f"环境加载资产数据：测试step {self.test_step_list}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMDRL(**self.model_param).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # ======== 验证集最优模型记录 ========
        self.best_val_sharpe = -np.inf
        self.best_epoch = -1

    def train(self):
        dataloader = BasedDataLoader(self.sample_list)
        for epoch in range(self.num_epoch):
            self.model.train()
            for i, batch in enumerate(dataloader.generate_batches(self.batch_size)):
                loss = 0
                # 每个idx表示投资起点，input_list表示从这个投资起点每个时间步的起点
                for idx in batch:
                    input_list = [idx + k * self.step_size for k in range(self.num_steps)]
                    # 随机打乱当前batch内的序列顺序
                    random.shuffle(input_list)
                    _input = self.generate_input(input_list).to(self.device)
                    actions = self.model(_input)
                    # 计算每个trajectory的夏普率损失
                    loss += self.generate_sharp(actions, input_list)

                # 对每个batch里的样本进行训练, 表示每个trajectory的夏普率平均水平
                loss = loss / len(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.logger.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

            # ======== 每个 epoch 结束以后：仅用验证集 Sharpe 选模型 ========
            val_sharpe = self.validate()
            self.logger.info(f"[Epoch {epoch}] 验证集 Sharpe: {val_sharpe:.4f}")
            if val_sharpe > self.best_val_sharpe:
                self.best_val_sharpe = val_sharpe
                self.best_epoch = epoch
                best_path = os.path.join(self.cun_file, 'best_model.pth')
                torch.save(self.model.state_dict(), best_path)
                self.logger.info(f"验证集 Sharpe 提升，保存当前最优模型到: {best_path}")
            # 没有验证集：按 epoch 保存，最后用最后一个模型做测试
            epoch_path = os.path.join(self.cun_file, f'model_{epoch}.pth')
            torch.save(self.model.state_dict(), epoch_path)


    # ====================== 验证：仅用于模型选择 ======================
    def validate(self):
        """
        仅用于模型选择：
        - 在验证集上跑一遍
        - 不写文件，不画图
        - 只计算一个 Sharpe 并返回
        """
        if self.valid_step_list is None or len(self.valid_step_list) == 0:
            return -np.inf

        self.model.eval()
        with torch.no_grad():
            # 1. 前向
            val_input = self.generate_input(self.valid_step_list).to(self.device)
            actions = self.model(val_input)  # [num_traj, num_stocks]

            # 2. 时间展开成逐日权重
            weights = torch.repeat_interleave(
                input=actions,
                repeats=self.step_size,
                dim=0
            ).detach().cpu().numpy()

            # 3. 对齐验证集价格
            adjc = self.adjc[:, self.valid_step_list[0]:
                                self.valid_step_list[0] + len(self.valid_dates)].numpy()
            # 权重长度截断为 L-1
            weights = weights[:adjc.shape[1] - 1, :]

            # 4. 按持仓区间进行权重“漂移”更新（和 generate_estimate 一致）
            start_day = self.valid_step_list[0]
            input_list2 = [day - start_day for day in self.valid_step_list]
            input_list2.append(adjc.shape[1] - 1)

            for i, day in enumerate(self.valid_step_list):
                prev_action = actions[i].unsqueeze(1).detach().cpu().numpy()
                for j in range(input_list2[i] + 1, input_list2[i + 1]):
                    adjc_ratio = adjc[:, j:j + 1] / adjc[:, j - 1:j]
                    prev_action = prev_action * adjc_ratio / np.sum(prev_action * adjc_ratio)
                    weights[j, :] = prev_action.squeeze(-1)

            # 5. 根据权重 + 价格计算每日收益率
            returns = np.sum(
                weights.T * (adjc[:, 1:] - adjc[:, :-1]) / adjc[:, :-1],
                axis=0
            )  # [L-1]
            decision_offsets = [
                day - self.valid_step_list[0] for day in self.valid_step_list
            ]
            turnovers = rebalance_turnovers(weights, adjc, decision_offsets)
            returns = apply_transaction_costs(
                returns,
                turnovers,
                transaction_cost_rate=self.transaction_cost_pct,
            )

            # 6. 只计算 Sharpe 作为模型选择指标
            mean_ret = returns.mean()
            vol_ret = returns.std()
            if vol_ret <= 0:
                sharpe = -np.inf
            else:
                ann_return = self.annualize_returns(mean_ret)
                ann_vol = self.annualize_volatility(vol_ret)
                sharpe = ann_return / ann_vol

        return sharpe

    def test(self):
        # ======== 训练结束：加载最优模型，在测试集回测 ========
        # if self.best_epoch >= 0:
        #     best_path = os.path.join(self.cun_file, 'best_model.pth')
        #     self.logger.info(f"训练结束，加载验证集最优模型（epoch={self.best_epoch}）进行测试: {best_path}")
        #     self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        # else:
        #     # 没有验证集的情况：使用最后一个 epoch 的权重
        #     last_path = os.path.join(self.cun_file, f'model_{self.num_epoch - 1}.pth')
        #     self.logger.info(f"未使用验证集，加载最后一个 epoch 模型进行测试: {last_path}")
        #     self.model.load_state_dict(torch.load(last_path, map_location=self.device))
        # # TODO: 单测
        model_path = os.path.join(self.cun_file, f'best_model.pth')
        self.logger.info(f"模型进行测试: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # self.test_step_list为测试的时间步开始列表
        test_input = self.generate_input(self.test_step_list).to(self.device)
        actions = self.model(test_input)
        self.generate_estimate(actions, self.test_step_list)


    def generate_input(self, input_list):
        states = []
        for i, day in enumerate(input_list):
            state = self.features[:, day-self.look_back+1 : day+1, :]
            state = self._normalize_window(state)
            state = torch.tensor(state, dtype=torch.float32)
            states.append(state)
        return torch.stack(states, dim=0)

    def _ann_sharpe_from_returns(self, rets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        rets: [T] 日收益率序列（torch），返回标量年化Sharpe
        """
        mu = rets.mean()
        sigma = rets.std(unbiased=False)
        # return (math.sqrt(252.0) * mu) / (sigma + eps)
        return mu/sigma

    def generate_sharp(self, actions: torch.Tensor, input_list):
        """
        训练用loss：
        mean over steps of  - (AnnSharpe(port) - AnnSharpe(BH))
        """
        total_loss = 0.0
        n_assets = self.adjc.shape[0]
        eps = 1e-8

        for i, t in enumerate(input_list):
            # 需要 close[t : t+step_size+1]
            end = t + self.step_size + 1
            adjc = self.adjc[:, t:end].to(self.device)  # [N, L]

            if adjc.shape[1] < 2:
                continue

            action = actions[i].unsqueeze(1)  # [N,1]
            # 强制非负归一（防止模型输出无约束）
            action = torch.clamp(action, min=0.0)
            action = action / (torch.sum(action, dim=0, keepdim=True) + eps)

            rel = adjc / (adjc[:, 0:1] + eps)  # [N, L]
            port_val = torch.sum(action * rel, dim=0)  # [L]
            port_ret = (port_val[1:] - port_val[:-1]) / (port_val[:-1] + eps)  # [L-1]
            port_ret = port_ret.clone()
            port_ret[0] = (
                port_ret[0]
                - torch.sum(torch.abs(action[:, 0])) * self.transaction_cost_pct
            )

            s_port = self._ann_sharpe_from_returns(port_ret, eps=eps)
            total_loss = total_loss - s_port

        return total_loss


    def generate_estimate(self, actions, input_list):
        # 得到投资组合在各个trajectory中的收益情况
        init_value = 1000
        weights = torch.repeat_interleave(
            input=actions,
            repeats=self.step_size,  # 每组重复的次数
            dim=0  # 在第0维（行）上重复
        ).detach().cpu().numpy()

        """获取当前所有股票的价格数据"""
        # 计算得到整个trajectory的夏普收益率，文中未考虑前一时间步的actions
        adjc = self.adjc[:, self.test_step_list[0]:self.test_step_list[0]+len(self.test_dates)].numpy()
        weights = weights[:adjc.shape[1]-1, :]
        start_day = input_list[0]
        input_list2 = [day-start_day for day in input_list]
        input_list2.append(adjc.shape[1]-1)
        for i, day in enumerate(input_list):
            prev_action=actions[i].unsqueeze(1).detach().cpu().numpy()
            for j in range(input_list2[i]+1, input_list2[i+1]):
                adjc_ratio = adjc[:,j:j+1]/adjc[:,j-1:j]
                prev_action = prev_action * adjc_ratio/sum(prev_action * adjc_ratio)
                weights[j,:] = prev_action.squeeze(-1)

        # 创建动作DataFrame（行为日期，列为资产名）
        actions_df = pd.DataFrame(
            data=weights,
            index=self.test_dates[:adjc.shape[1]-1],  # 使用日期作为索引
            columns=self.stocks
        )

        action_path = os.path.join(self.cun_file, 'actions')
        if not os.path.exists(action_path):
            os.makedirs(action_path)
            self.logger.info(f'创建action保存路径{action_path}')

        file_path = action_path + f"/test.csv"
        actions_df.to_csv(file_path)
        self.logger.info(f"动作数据已保存至：{file_path}")

        returns = np.sum(weights.T * (adjc[:, 1:] - adjc[:, :-1]) / adjc[:, :-1], axis=0)
        turnovers = rebalance_turnovers(weights, adjc, input_list2[:-1])
        returns = apply_transaction_costs(
            returns,
            turnovers,
            transaction_cost_rate=self.transaction_cost_pct,
        )

        wealth_index = (1 + returns).cumprod()
        # 计算累计最大值（替代 pandas 的 cummax()）
        cum_max_wealth = np.maximum.accumulate(wealth_index)  # 每个时间步的累计最大值
        # 绩效指标
        ann_return = self.annualize_returns(returns.mean())
        ann_vol = self.annualize_volatility(returns.std())
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        max_dd = (wealth_index / cum_max_wealth - 1).min()
        cum_return = wealth_index[-1] - 1

        # 打印性能指标
        self.logger.info(f"\n=========== 回测结果: {self.cun_file}===========")
        self.logger.info(f"初始组合价值: {init_value:.2f}")
        self.logger.info(f"最终组合价值: {init_value*wealth_index[-1]:.2f}")
        self.logger.info("\n====== 测试区间绩效 ======")
        self.logger.info(f"累计收益: {cum_return:.2%}")
        self.logger.info(f"年化收益: {ann_return:.2%}")
        self.logger.info(f"年化波动率: {ann_vol:.2%}")
        self.logger.info(f"夏普比率: {sharpe:.2f}")
        self.logger.info(f"最大回撤: {abs(max_dd):.2%}")
        self.logger.info("=================================\n")

        # # 绘制图形
        self._plot_actions(actions_df)
        self._plot_returns(wealth_index,self.test_dates[:adjc.shape[1]-1])

        raw_folder = config.dataset['feature_path']
        prices_df = load_prices(raw_folder, self.stocks)
        contrib_df = compute_asset_contributions(
            actions_df, prices_df,
            transaction_cost_pct=self.transaction_cost_pct,
            initial_value=1000
        )

        plot_price_and_contrib(
            actions_df, prices_df, contrib_df,
            save_path=f"{self.cun_file}/combined.png"
        )

    def get_train_index(self, start_date_str, end_date_str):
        """
        训练集索引生成：
        - 样本起点 day 需要有足够的回看窗口：day >= look_back
        - 样本内部要往后走 num_steps * step_size 天，不能越界
        """
        start = pd.to_datetime(start_date_str)
        end = pd.to_datetime(end_date_str)

        # -------- 1. 找日期对应的原始索引 --------
        try:
            start_index = self.all_dates.index(start)
        except ValueError:
            if start < self.all_dates[0]:
                self.logger.info(
                    f"[get_train_index] 起始日期 {start_date_str} < 数据开始 {self.all_dates[0]}，截断为数据开始。")
                start_index = 0
            elif start > self.all_dates[-1]:
                self.logger.info(
                    f"[get_train_index] 起始日期 {start_date_str} > 数据结束 {self.all_dates[-1]}，无法生成训练集。")
                return []
            else:
                self.logger.info(f"[get_train_index] 起始日期 {start_date_str} 在 all_dates 中找不到，检查日期格式。")
                return []

        try:
            end_index = self.all_dates.index(end)
        except ValueError:
            if end < self.all_dates[0]:
                self.logger.info(
                    f"[get_train_index] 结束日期 {end_date_str} < 数据开始 {self.all_dates[0]}，无法生成训练集。")
                return []
            elif end > self.all_dates[-1]:
                self.logger.info(
                    f"[get_train_index] 结束日期 {end_date_str} > 数据结束 {self.all_dates[-1]}，截断为数据结束。")
                end_index = len(self.all_dates) - 1
            else:
                self.logger.info(f"[get_train_index] 结束日期 {end_date_str} 在 all_dates 中找不到，检查日期格式。")
                return []

        # -------- 2. 考虑 look_back 和 trajectory 长度 --------
        # day 作为样本起点时，需要 day >= look_back 才能取到完整回看窗口
        effective_start = max(start_index, self.look_back)

        # 训练中，每个样本内部会从 day 开始，往后走 num_steps 个点，每次 step_size
        # 最远用到 day + step_size * num_steps
        max_traj_span = self.step_size * self.num_steps
        effective_end = min(end_index - max_traj_span, len(self.all_dates) - 1)

        if effective_end < effective_start:
            self.logger.info(
                f"[get_train_index] 区间 {start_date_str} ~ {end_date_str} 在 "
                f"look_back={self.look_back}, num_steps={self.num_steps}, "
                f"step_size={self.step_size} 约束下无有效训练样本。"
            )
            return []

        sample_list = list(range(effective_start, effective_end + 1))
        self.logger.info(
            f"[get_train_index] 训练样本起点索引范围: [{effective_start}, {effective_end}], "
            f"样本数={len(sample_list)}, 日期 {self.all_dates[effective_start]} ~ {self.all_dates[effective_end]}"
        )
        return sample_list

    # 生成验证 / 测试集每一个 step 的起始索引
    def get_test_index(self, start_date_str, end_date_str):
        """
        输入日期字符串，返回：
          - step_list: 用于 generate_input(...) 的起始 day 索引列表
          - dates:     对应价格/权重序列的日期索引（从第一个可用 day 开始）

        约束：
        - 用作 step 的 day 必须满足 day >= look_back，保证
          window = features[:, day - look_back : day, :] 存在。
        - step 之间间隔为 self.step_size。
        """
        start = pd.to_datetime(start_date_str)
        end = pd.to_datetime(end_date_str)

        # -------- 1. 找日期对应的原始索引 --------
        try:
            start_index = self.all_dates.index(start)
        except ValueError:
            if start < self.all_dates[0]:
                self.logger.info(
                    f"[get_test_index] 起始日期 {start_date_str} < 数据开始 {self.all_dates[0]}，截断为数据开始。")
                start_index = 0
            elif start > self.all_dates[-1]:
                self.logger.info(
                    f"[get_test_index] 起始日期 {start_date_str} > 数据结束 {self.all_dates[-1]}，无法生成 step_list。")
                return [], []
            else:
                self.logger.info(f"[get_test_index] 起始日期 {start_date_str} 在 all_dates 中找不到，检查日期格式。")
                return [], []

        try:
            end_index = self.all_dates.index(end)
        except ValueError:
            if end < self.all_dates[0]:
                self.logger.info(
                    f"[get_test_index] 结束日期 {end_date_str} < 数据开始 {self.all_dates[0]}，无法生成 step_list。")
                return [], []
            elif end > self.all_dates[-1]:
                self.logger.info(
                    f"[get_test_index] 结束日期 {end_date_str} > 数据结束 {self.all_dates[-1]}，截断为数据结束。")
                end_index = len(self.all_dates) - 1
            else:
                self.logger.info(f"[get_test_index] 结束日期 {end_date_str} 在 all_dates 中找不到，检查日期格式。")
                return [], []

        # -------- 2. 考虑 look_back：day 必须 >= look_back --------
        # 原始数据从 2000-04-07 开始时：
        #   - index(2000-04-07) = 0
        #   - 第一个满足 day >= look_back=240 的 day ≈ 2001 年某个交易日
        effective_start = max(start_index, self.look_back)
        effective_end = end_index-2

        if effective_end < effective_start:
            self.logger.info(
                f"[get_test_index] 区间 {start_date_str} ~ {end_date_str} 在 "
                f"look_back={self.look_back} 约束下无有效 step。"
            )
            return [], []

        # 按 step_size 生成 step_list
        step_list = list(range(effective_start, effective_end + 1, self.step_size))
        dates = self.all_dates[effective_start:effective_end + 1]

        self.logger.info(
            f"[get_test_index] step 索引范围: [{effective_start}, {effective_end}], "
            f"step 数={len(step_list)}, 日期 {dates[0]} ~ {dates[-1]}"
        )
        return step_list, dates

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        """
        对单个 (num_stocks, window_size, features) 窗口做 min–max 归一化。
        按资产维度分别对各自的 window_size×features 做归一到 [0,1]。
        """
        eps = 1e-8
        # 窗口 shape = (N_assets, N_days, N_feats)
        mean = window.mean(axis=1, keepdims=True)  # (N_assets, 1, N_feats)
        std = window.std(axis=1, keepdims=True)  # (N_assets, 1, N_feats)
        return (window - mean) / (std + eps)

    def _plot_returns(self, returns,dates):
        """绘制收益率曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(dates,returns, label="Cumulative Returns")
        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
        plt.savefig(self.cun_file + f"/cumulative_returns.png", bbox_inches="tight")
        plt.close()

    def _plot_actions(self, actions_df: pd.DataFrame):
        """绘制资产配置变化图"""
        plt.figure(figsize=(15, 6))
        actions_df.plot(kind='area', stacked=True, title="Asset Allocation Over Time")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.savefig(self.cun_file + f"/asset_allocation.png", bbox_inches="tight")
        plt.close()

    # —— 工具函数 ——
    def compute_log_returns(self, prices):
        return np.log(prices / prices.shift(1)).dropna()

    def compute_simple_returns(self, prices):
        return prices.pct_change()

    def annualize_returns(self, mean_daily, periods_per_year=252):
        return mean_daily * periods_per_year

    def annualize_volatility(self, daily_vol, periods_per_year=252):
        return daily_vol * np.sqrt(periods_per_year)



# 原函数接口保持不变
def Alpha_stock(cun_path, logger):
    rl_trading = alphastock_main(cun_path, logger)
    rl_trading.train()
    rl_trading.test()
