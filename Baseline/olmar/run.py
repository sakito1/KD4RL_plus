import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from utils import config as config
from utils.PriceMatrix import Datamatrix_adjclose, Datamatrix_adjopen


TRANSACTION_COST_RATE=config.TRANSACTION_COST_RATE
# —— 工具函数 ——
def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

#计算rt=(pt-pt-1)/pt-1
def compute_simple_returns(prices):
    return prices.pct_change()

def annualize_returns(mean_daily, periods_per_year=252):
    return mean_daily * periods_per_year

def annualize_volatility(daily_vol, periods_per_year=252):
    return daily_vol * np.sqrt(periods_per_year)

def calculate_transaction_costs(prev_weights, target_weights, prev_prices, current_prices,
                                transaction_cost_rate=TRANSACTION_COST_RATE):
    """计算交易成本，考虑因价格波动导致的权重偏离"""
    if prev_weights is None:
        # 首次交易，假设从全现金(权重为0)开始
        trades = target_weights
        # 计算交易成本
        return np.sum(trades) * transaction_cost_rate, trades
    elif target_weights is None:
        # 说明不做买卖交易，保持上一部分的持有情况，但是随着价格变动，权重会发生变化
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)  # 归一化
        # 计算交易成本
        return 0, current_weights
    else:
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)  # 归一化
        # 计算从当前实际权重调整到目标权重所需的交易
        trades = np.abs(target_weights - current_weights)
        # 计算交易成本
        return np.sum(trades) * transaction_cost_rate, target_weights


def simplex_projection(v: np.ndarray) -> np.ndarray:
    """
    Projects a vector v onto the probability simplex (elements sum to 1, non-negative).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def olmar(prices: pd.DataFrame, window: int = 5, eps: float = 10.0) -> pd.DataFrame:
    """
    Implements the On-Line Moving Average Reversion (OLMAR) algorithm.

    Parameters:
    - prices: DataFrame of historical prices (index = dates, columns = asset ticks)
    - window: look-back window size for moving average
    - eps: sensitivity parameter for the passive-aggressive update

    Returns:
    - DataFrame of portfolio weights over time (index aligned to prices.index[window:])
    """
    n_assets = prices.shape[1]
    weights = np.ones(n_assets) / n_assets
    weights_history = []

    # Iterate from the first day with enough history
    for t in range(window, len(prices)):
        ma = prices.iloc[t - window:t].mean(axis=0)
        x_pred = ma / prices.iloc[t]
        p = weights.dot(x_pred.values)
        loss = max(0.0, eps - p)

        if loss > 0:
            x_bar = x_pred.values.mean()
            var = np.linalg.norm(x_pred.values - x_bar) ** 2
            lam = loss / var
            weights = weights + lam * (x_pred.values - x_bar)
            weights = simplex_projection(weights)

        weights_history.append(weights.copy())

    result_idx = prices.index[window:]
    return pd.DataFrame(weights_history, index=result_idx, columns=prices.columns)

def run(cun_path, logger):
    start_train = config.train_start_date
    start_test  = config.test_start_date
    end_test    = config.test_end_date

    with open(config.dataset['stocks_path']) as f:
        stocks = [line.strip() for line in f]

    # C2C：只用 close
    prices_full = Datamatrix_adjclose(
        file_paths=config.dataset['feature_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )

    cun_path = os.path.join(cun_path, "olmar")
    os.makedirs(cun_path, exist_ok=True)

    full_index = prices_full.index
    end_test_dt = full_index[-1]
    start_pos = full_index.get_loc(start_test)
    end_pos = full_index.get_loc(end_test_dt)
    index_list = full_index.tolist()

    # 1) OLMAR 权重（基于 close 序列）
    weights_df = olmar(prices_full, window=config.olmar_window_size, eps=5)
    weights_df = weights_df.loc[start_test:end_test_dt]

    transaction_costs = []
    weight_records = []

    prev_weights = None
    prev_prices = prices_full.iloc[start_pos - 1, :] if start_pos - 1 >= 0 else prices_full.iloc[start_pos, :]

    for current_pos in range(start_pos, end_pos + 1):
        day = index_list[current_pos]
        current_prices = prices_full.loc[day, :]

        # 你原逻辑：每 30 天更新一次目标权重，其余天只漂移
        if (current_pos - start_pos) % 30 == 0:
            w = weights_df.loc[day]
        else:
            w = None

        cost, true_w = calculate_transaction_costs(prev_weights, w, prev_prices, current_prices)
        transaction_costs.append({'date': day, 'cost': float(cost)})

        weight_records.append(true_w.rename(day))
        prev_weights = true_w
        prev_prices = current_prices

    # 实际持仓权重序列（用于收益计算）
    wdf = pd.DataFrame(weight_records).sort_index()

    # 成本严格按 wdf 对齐（修复你原来的 reindex 错位）
    costs_df = pd.DataFrame(transaction_costs).set_index('date')
    costs_df = costs_df.reindex(wdf.index).fillna(0.0)

    # 2) C2C 收益：ret_t = (C_{t+1}/C_t - 1) 对齐到 t
    returns_c2c = compute_simple_returns(prices_full).shift(-1)
    returns_c2c = returns_c2c.reindex(wdf.index)

    portfolio_returns = (wdf * returns_c2c).sum(axis=1) - costs_df['cost']
    portfolio_returns = portfolio_returns.dropna()  # 最后一天没有 t+1 close

    wealth_index = (1 + portfolio_returns).cumprod()

    ann_return = annualize_returns(portfolio_returns.mean())
    ann_vol = annualize_volatility(portfolio_returns.std())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((wealth_index / wealth_index.cummax()) - 1).min()
    cum_return = wealth_index.iloc[-1] - 1

    logger.info("\n====== 测试区间绩效 ======")
    logger.info("基线策略：olmar (Close-to-Close)")
    logger.info(f"累计收益: {cum_return:.2%}")
    logger.info(f"年化收益: {ann_return:.2%}")
    logger.info(f"年化波动率: {ann_vol:.2%}")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {abs(max_dd):.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_index, label="olmar Portfolio")
    plt.title("Cumulative Return (Close-to-Close)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, 'value.png'))
    plt.close()

    plt.figure(figsize=(15, 6))
    wdf.loc[wealth_index.index].plot(kind='area', stacked=True, title="Asset Allocation Over Time", figsize=(15, 6))
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, 'asset_allocation.png'))
    plt.close()

    wdf.to_csv(os.path.join(cun_path, "daily_weights.csv"))
    pd.DataFrame({
        "portfolio_value": wealth_index,
        "daily_return": portfolio_returns
    }).to_csv(os.path.join(cun_path, "portfolio_performance.csv"))

