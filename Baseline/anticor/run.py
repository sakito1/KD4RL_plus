import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import config as config
from utils.PriceMatrix import Datamatrix_adjclose

TRANSACTION_COST_RATE = config.TRANSACTION_COST_RATE

def compute_simple_returns(prices: pd.DataFrame):
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
        return np.sum(trades) * transaction_cost_rate, trades
    elif target_weights is None:
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)
        return 0.0, current_weights
    else:
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)
        trades = np.abs(target_weights - current_weights)
        return np.sum(trades) * transaction_cost_rate, target_weights

def simplex_projection(v: np.ndarray) -> np.ndarray:
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def anticor(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    n, m = prices.shape
    weights = np.ones(m) / m
    W = []

    R = prices.values[1:] / prices.values[:-1]
    epsilon = 1e-10
    R += np.random.uniform(-epsilon, epsilon, R.shape)

    for t in range(window, n - 1):
        R1 = R[t - 2*window:t-window]
        R2 = R[t-window:t]

        mu1 = np.nanmean(R1, axis=0)
        mu2 = np.nanmean(R2, axis=0)

        C = np.cov(R1, rowvar=False)
        std1 = np.sqrt(np.diag(C))
        P = C / (std1[:, None] * std1[None, :])

        transfer = np.zeros((m, m))
        valid_pairs = (mu1 < mu1[:, None]) & (mu2 > mu2[:, None])
        transfer[valid_pairs] = np.maximum(0, P[valid_pairs])

        claim = transfer - transfer.T
        delta = claim.sum(axis=1)
        w_new = weights + delta
        w_new = simplex_projection(w_new)
        weights = w_new
        W.append(weights.copy())

    result_idx = prices.index[window + 1:]
    return pd.DataFrame(W, index=result_idx, columns=prices.columns)

def run(cun_path, logger):
    start_train = config.train_start_date
    end_test = config.test_end_date
    start_test = config.test_start_date

    with open(config.dataset['stocks_path']) as f:
        stocks = [line.strip() for line in f]

    # 统一使用收盘价（Close-to-Close）
    prices_close = Datamatrix_adjclose(
        file_paths=config.dataset['feature_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )

    cun_path = os.path.join(cun_path, "anticor")
    os.makedirs(cun_path, exist_ok=True)

    full_index = prices_close.index
    # end_test 可能是字符串/时间戳，确保取到数据里最后一个有效交易日
    end_test_dt = full_index[-1]
    start_pos = full_index.get_loc(start_test)
    end_pos = full_index.get_loc(end_test_dt)
    index_list = full_index.tolist()

    # 1) 生成权重（基于 close 序列）
    weights_df = anticor(prices_close, window=config.anticor_window_size)
    weights_df = weights_df.loc[start_test:end_test_dt]

    weight_records = []
    transaction_costs = []

    prev_weights = None
    # 处理 start_pos=0 的情况
    if start_pos - 1 >= 0:
        prev_prices = prices_close.iloc[start_pos - 1, :]
    else:
        prev_prices = prices_close.iloc[start_pos, :]

    for current_pos in range(start_pos, end_pos + 1):
        day = index_list[current_pos]
        current_prices = prices_close.loc[day, :]

        # 每 30 天（交易日）调一次仓，其余日期只漂移权重
        if (current_pos - start_pos) % 30 == 0:
            w = weights_df.loc[day]
        else:
            w = None

        cost, true_w = calculate_transaction_costs(prev_weights, w, prev_prices, current_prices)
        transaction_costs.append({'date': day, 'cost': float(cost)})

        weight_records.append(true_w.rename(day))
        prev_weights = true_w
        prev_prices = current_prices

    costs_df = pd.DataFrame(transaction_costs).set_index('date')
    wdf = pd.DataFrame(weight_records).sort_index()
    costs_df = costs_df.reindex(wdf.index).fillna(0.0)

    # 2) Close-to-Close 区间收益：ret_t = (C_{t+1}/C_t - 1)
    # 与你现有写法保持一致：pct_change()给的是 (C_t/C_{t-1}-1)，shift(-1) 变成 (C_{t+1}/C_t-1) 对齐到 t
    returns_c2c = compute_simple_returns(prices_close).shift(-1)
    returns_c2c = returns_c2c.loc[wdf.index]  # 与权重日期严格对齐

    portfolio_returns = (wdf * returns_c2c).sum(axis=1) - costs_df['cost']
    # 最后一天没有 t+1，会是 NaN，去掉
    portfolio_returns = portfolio_returns.dropna()

    wealth_index = (1.0 + portfolio_returns).cumprod()

    ann_return = annualize_returns(portfolio_returns.mean())
    ann_vol = annualize_volatility(portfolio_returns.std())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((wealth_index / wealth_index.cummax()) - 1).min()
    cum_return = wealth_index.iloc[-1] - 1

    logger.info("\n====== 测试区间绩效 ======")
    logger.info("基线策略：anticor (Close-to-Close)")
    logger.info(f"累计收益: {cum_return:.2%}")
    logger.info(f"年化收益: {ann_return:.2%}")
    logger.info(f"年化波动率: {ann_vol:.2%}")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {abs(max_dd):.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_index, label="anticor Portfolio")
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
