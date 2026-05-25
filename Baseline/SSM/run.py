import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from utils import config as config
from utils.PriceMatrix import Datamatrix_adjclose, Datamatrix_adjopen, Datamatrix_bull_label


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
    else:
        # 计算当前实际权重（考虑价格变动）
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)  # 归一化

        # 计算从当前实际权重调整到目标权重所需的交易
        trades = np.abs(target_weights - current_weights)
    # 计算交易成本
    return np.sum(trades) * transaction_cost_rate



def run(cun_path, logger):
    start_train = config.train_start_date
    start_test  = config.test_start_date
    end_test    = config.test_end_date

    with open(config.dataset['stocks_path']) as f:
        stocks = [line.strip() for line in f]

    # C2C：交易与收益都用 close
    prices_close = Datamatrix_adjclose(
        file_paths=config.dataset['feature_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )

    # bull label（保持你原来的保守对齐：shift(1) 避免未来信息）
    price_label = Datamatrix_bull_label(
        file_paths=config.dataset['ssm_data_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )


    cun_path = os.path.join(cun_path, "ssm_only")
    os.makedirs(cun_path, exist_ok=True)

    full_index = prices_close.index
    end_test_dt = full_index[-1]
    start_pos = full_index.get_loc(start_test)
    end_pos = full_index.get_loc(end_test_dt)
    index_list = full_index.tolist()

    # 1) label -> 权重（全 0 行改成全 1，再按行归一化）
    weights_df = price_label.astype(float)
    row_sums = weights_df.sum(axis=1)
    zero_mask = (row_sums == 0)
    if zero_mask.any():
        weights_df.loc[zero_mask, :] = 1.0
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)

    # 限制到测试区间（权重日期 = t）
    weights_df = weights_df.loc[start_test:end_test_dt]

    # 2) 交易成本：按 close 价格漂移权重并计算换手
    transaction_costs = []
    prev_weights = None
    prev_prices = prices_close.iloc[start_pos - 1, :] if start_pos - 1 >= 0 else prices_close.iloc[start_pos, :]

    for current_pos in range(start_pos, end_pos + 1):
        day = index_list[current_pos]
        current_prices = prices_close.loc[day, :]

        w = weights_df.loc[day]  # 每天都用 label 权重（SSM_only 就是每日信号）
        cost = calculate_transaction_costs(prev_weights, w, prev_prices, current_prices)
        transaction_costs.append({'date': day, 'cost': float(cost)})

        prev_weights = w
        prev_prices = current_prices

    costs_df = pd.DataFrame(transaction_costs).set_index('date')
    costs_df = costs_df.reindex(weights_df.index).fillna(0.0)

    # 3) C2C 收益：ret_t = (C_{t+1}/C_t - 1) 对齐到 t
    returns_c2c = compute_simple_returns(prices_close).shift(-1)
    returns_c2c = returns_c2c.reindex(weights_df.index)

    portfolio_returns = (weights_df * returns_c2c).sum(axis=1) - costs_df['cost']
    portfolio_returns = portfolio_returns.dropna()  # 最后一天无 t+1 close

    wealth_index = (1 + portfolio_returns).cumprod()

    ann_return = annualize_returns(portfolio_returns.mean())
    ann_vol = annualize_volatility(portfolio_returns.std())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((wealth_index / wealth_index.cummax()) - 1).min()
    cum_return = wealth_index.iloc[-1] - 1

    logger.info("\n====== 测试区间绩效 ======")
    logger.info("基线策略：SSM_only (Close-to-Close)")
    logger.info(f"累计收益: {cum_return:.2%}")
    logger.info(f"年化收益: {ann_return:.2%}")
    logger.info(f"年化波动率: {ann_vol:.2%}")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {abs(max_dd):.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_index, label="SSM_only Portfolio")
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
