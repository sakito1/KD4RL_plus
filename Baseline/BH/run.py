import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import config as config
from utils.PriceMatrix import Datamatrix_adjclose

TRANSACTION_COST_RATE = config.TRANSACTION_COST_RATE

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
        trades = target_weights
    else:
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)
        trades = np.abs(target_weights - current_weights)
    return float(np.sum(trades) * transaction_cost_rate)

def buy_and_hold_weights(prices: pd.DataFrame, initial_weights: np.ndarray = None) -> pd.DataFrame:
    n_assets = prices.shape[1]
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets
    else:
        initial_weights = np.array(initial_weights)
        initial_weights = initial_weights / initial_weights.sum()

    init_prices = prices.iloc[0].values
    holdings = initial_weights / init_prices

    asset_values = prices.multiply(holdings, axis=1)
    total_value = asset_values.sum(axis=1)
    weights_over_time = asset_values.div(total_value, axis=0)
    return weights_over_time

def run(cun_path, logger):
    start_train = config.train_start_date
    start_test = config.test_start_date
    end_test = config.test_end_date

    with open(config.dataset['stocks_path']) as f:
        stocks = [line.strip() for line in f]

    # C2C：只需要 close 价格
    prices_close = Datamatrix_adjclose(
        file_paths=config.dataset['feature_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )

    cun_path = os.path.join(cun_path, "BH")
    os.makedirs(cun_path, exist_ok=True)

    full_index = prices_close.index
    end_test_dt = full_index[-1]  # 保证在数据索引内
    start_pos = full_index.get_loc(start_test)
    end_pos = full_index.get_loc(end_test_dt)
    index_list = full_index.tolist()

    # Buy&Hold 权重路径（从 start_test 开始）
    weights_df = buy_and_hold_weights(prices_close.loc[start_test:end_test_dt, :])
    weights_df = weights_df.loc[start_test:end_test_dt]

    # 交易成本：仅首日建仓一次，之后不交易（权重随价格自然漂移）
    transaction_costs = []
    prev_prices = prices_close.iloc[start_pos - 1, :] if start_pos - 1 >= 0 else prices_close.iloc[start_pos, :]

    for current_pos in range(start_pos, end_pos + 1):
        day = index_list[current_pos]
        current_prices = prices_close.loc[day, :]

        if day == weights_df.index[0]:
            # 首日：从现金进入目标权重（用 weights_df 首日权重）
            w0 = weights_df.loc[day].values
            cost = float(np.sum(np.abs(w0)) * TRANSACTION_COST_RATE)
        else:
            cost = 0.0

        transaction_costs.append({"date": day, "cost": cost})
        prev_prices = current_prices

    costs_df = pd.DataFrame(transaction_costs).set_index("date")
    costs_df = costs_df.reindex(weights_df.index).fillna(0.0)

    # C2C 收益：ret_t = (C_{t+1}/C_t - 1)，对齐到 t
    returns_c2c = compute_simple_returns(prices_close).shift(-1)
    returns_c2c = returns_c2c.loc[weights_df.index]

    portfolio_returns = (weights_df * returns_c2c).sum(axis=1) - costs_df["cost"]
    portfolio_returns = portfolio_returns.dropna()  # 最后一天没有 t+1 close

    wealth_index = (1.0 + portfolio_returns).cumprod()

    ann_return = annualize_returns(portfolio_returns.mean())
    ann_vol = annualize_volatility(portfolio_returns.std())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((wealth_index / wealth_index.cummax()) - 1).min()
    cum_return = wealth_index.iloc[-1] - 1

    logger.info("\n====== 测试区间绩效 ======")
    logger.info("基线策略：Buy and Hold (Close-to-Close)")
    logger.info(f"累计收益: {cum_return:.2%}")
    logger.info(f"年化收益: {ann_return:.2%}")
    logger.info(f"年化波动率: {ann_vol:.2%}")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {abs(max_dd):.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_index, label="BH Portfolio")
    plt.title("Cumulative Return (Close-to-Close)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, "value.png"))
    plt.close()

    plt.figure(figsize=(15, 6))
    weights_df.loc[wealth_index.index].plot(kind="area", stacked=True, title="Asset Allocation Over Time", figsize=(15, 6))
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, "asset_allocation.png"))
    plt.close()

    weights_df.to_csv(os.path.join(cun_path, "daily_weights.csv"))
    pd.DataFrame({
        "portfolio_value": wealth_index,
        "daily_return": portfolio_returns
    }).to_csv(os.path.join(cun_path, "portfolio_performance.csv"))
