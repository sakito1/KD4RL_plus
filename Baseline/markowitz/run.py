import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from utils import config as config
from utils.PriceMatrix import Datamatrix_adjclose, Datamatrix_adjopen
from utils.config import train_end_date

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

def max_sharpe_long_only(mu, cov, rf=0.0):
    """长-only 最大夏普率组合，通过优化添加非负约束"""
    n = len(mu)
    args = (mu, cov, rf)
    # 目标：负Sharpe（因为我们最小化）
    def neg_sharpe(w, mu, cov, rf):
        ret = w.dot(mu)
        vol = np.sqrt(w.dot(cov).dot(w))
        return -(ret - rf) / vol
    # 约束：和为1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # 非负
    bounds = tuple((0, 1) for _ in range(n))
    # 初始猜测均匀分配
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, args=args, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else w0


def portfolio_metrics(prices, weights, rf=0.0):
    rets = compute_simple_returns(prices)
    port_rets = rets.dot(weights)
    cum_ret = (1 + port_rets).prod() - 1
    ann_ret = annualize_returns(port_rets.mean())
    ann_vol = annualize_volatility(port_rets.std())
    sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan
    wealth_index = (1 + port_rets).cumprod()
    running_max = wealth_index.cummax()
    drawdowns = (wealth_index - running_max) / running_max
    max_dd = drawdowns.min()
    return {
        'weights': pd.Series(weights, index=prices.columns),
        'cumulative_return': cum_ret,
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

def run(cun_path, logger):
    # 配置日期
    start_train = config.train_start_date
    end_train   = config.train_end_date
    start_test  = config.test_start_date
    end_test    = config.test_end_date

    with open(config.dataset['stocks_path']) as f:
        stocks = [i.strip() for i in f]

    # C2C：只需要 close
    prices_full = Datamatrix_adjclose(
        file_paths=config.dataset['feature_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )

    cun_path = os.path.join(cun_path, "Markowitz")
    os.makedirs(cun_path, exist_ok=True)

    full_index = prices_full.index
    end_test_dt = full_index[-1]  # 确保 end_test 在索引内
    start_pos = full_index.get_loc(start_test)
    end_pos = full_index.get_loc(end_test_dt)
    index_list = full_index.tolist()

    rf_rate = 0.01
    window_size = config.markowitz_window_size

    weight_records = []
    transaction_costs = []

    prev_weights = None
    prev_prices = prices_full.iloc[start_pos - 1, :] if start_pos - 1 >= 0 else prices_full.iloc[start_pos, :]

    for current_pos in range(start_pos, end_pos + 1):
        # 窗口不足直接跳过（避免负索引语义问题）
        if current_pos - window_size < 0:
            continue

        # hist 覆盖 [current_pos-window_size, current_pos) —— 不含 current_pos 当天
        hist = prices_full.iloc[current_pos - window_size: current_pos, :]
        if len(hist) < window_size:
            continue

        day = index_list[current_pos]

        # 用 hist 计算参数（用 log return 年化）
        log_rets = compute_log_returns(hist)
        mu = annualize_returns(log_rets.mean())
        cov = log_rets.cov() * 252

        current_prices = prices_full.iloc[current_pos, :]

        # 每 5 天（交易日）调一次仓，其余日期只漂移权重
        if (current_pos - start_pos) % 5 == 0:
            w = max_sharpe_long_only(mu.values, cov.values, rf=rf_rate)
        else:
            w = None

        cost, true_w = calculate_transaction_costs(prev_weights, w, prev_prices, current_prices)
        transaction_costs.append({'date': day, 'cost': float(cost)})

        prev_weights = true_w
        prev_prices = current_prices

        weights_series = pd.Series(true_w, index=hist.columns, name=day)
        weight_records.append(weights_series)

    # 整理权重 DataFrame
    weights_df = pd.DataFrame(weight_records).sort_index()

    # 交易成本对齐
    costs_df = pd.DataFrame(transaction_costs).set_index('date')
    costs_df = costs_df.reindex(weights_df.index).fillna(0.0)

    # C2C 收益：ret_t = (C_{t+1}/C_t - 1) 对齐到 t
    returns_c2c = compute_simple_returns(prices_full).shift(-1)
    returns_c2c = returns_c2c.reindex(weights_df.index)

    portfolio_returns = (weights_df * returns_c2c).sum(axis=1) - costs_df['cost']
    portfolio_returns = portfolio_returns.dropna()  # 去掉最后一天无 t+1 close

    wealth_index = (1 + portfolio_returns).cumprod()

    ann_return = annualize_returns(portfolio_returns.mean())
    ann_vol = annualize_volatility(portfolio_returns.std())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((wealth_index / wealth_index.cummax()) - 1).min()
    cum_return = wealth_index.iloc[-1] - 1

    logger.info("\n====== 测试区间绩效 ======")
    logger.info("基线策略：Markowitz (Close-to-Close)")
    logger.info(f"累计收益: {cum_return:.2%}")
    logger.info(f"年化收益: {ann_return:.2%}")
    logger.info(f"年化波动率: {ann_vol:.2%}")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {abs(max_dd):.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_index, label="Markowitz Portfolio")
    plt.title("Cumulative Return (Close-to-Close)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, 'value.png'))
    plt.close()

    plt.figure(figsize=(15, 6))
    weights_df.loc[wealth_index.index].plot(kind='area', stacked=True, title="Asset Allocation Over Time", figsize=(15, 6))
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, 'asset_allocation.png'))
    plt.close()

    weights_df.to_csv(os.path.join(cun_path, "daily_weights.csv"))
    pd.DataFrame({
        "portfolio_value": wealth_index,
        "daily_return": portfolio_returns
    }).to_csv(os.path.join(cun_path, "portfolio_performance.csv"))
