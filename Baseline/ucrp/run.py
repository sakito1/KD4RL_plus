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
    if prev_weights is None:
        trades = target_weights
    else:
        price_changes = current_prices / prev_prices
        current_weights = prev_weights * price_changes
        current_weights = current_weights / np.sum(current_weights)
        trades = np.abs(target_weights - current_weights)
    return float(np.sum(trades) * transaction_cost_rate)

class UCRP:
    def __init__(self):
        self.weights = None

    def fit(self, returns_df: pd.DataFrame):
        n_assets = returns_df.shape[1]
        self.weights = np.ones(n_assets) / n_assets
        return self

    def predict(self, prices_close: pd.DataFrame, returns_c2c: pd.DataFrame, start_pos: int, end_pos: int):
        """
        prices_close: (T, N) close 价格（用于成本/漂移）
        returns_c2c:  (T, N) 对齐到 t 的 close->close 收益：ret_t = C_{t+1}/C_t - 1
        """
        if self.weights is None:
            self.fit(returns_c2c)

        full_index = prices_close.index
        index_list = full_index.tolist()

        # 构造等权目标（每日再平衡语义：每天都调回等权）
        n_assets = prices_close.shape[1]
        target_w = self.weights.copy()

        transaction_costs = []
        prev_weights = None
        prev_prices = prices_close.iloc[start_pos - 1, :] if start_pos - 1 >= 0 else prices_close.iloc[start_pos, :]

        # 逐日计算成本
        for i in range(start_pos, end_pos + 1):
            day = index_list[i]
            current_prices = prices_close.iloc[i, :]
            cost = calculate_transaction_costs(prev_weights, target_w, prev_prices, current_prices)

            transaction_costs.append({'date': day, 'cost': cost})
            prev_weights = target_w
            prev_prices = current_prices

        costs_df = pd.DataFrame(transaction_costs).set_index('date')

        # 计算组合收益（按日期对齐）
        # 只取回测区间的收益；最后一天通常为 NaN（无 t+1）
        ret_slice = returns_c2c.iloc[start_pos:end_pos + 1, :]
        ret_slice = ret_slice.copy()
        ret_slice['__date__'] = prices_close.index[start_pos:end_pos + 1]
        ret_slice = ret_slice.set_index('__date__')

        costs_df = costs_df.reindex(ret_slice.index).fillna(0.0)

        # 每天组合收益：w · r_t - cost_t
        daily_returns = ret_slice.dot(target_w) - costs_df['cost']
        daily_returns = daily_returns.dropna()  # drop 最后一天 NaN

        cum_wealth = (1.0 + daily_returns).cumprod()
        return cum_wealth, daily_returns

def run(cun_path, logger):
    start_train = config.train_start_date
    start_test  = config.test_start_date
    end_test    = config.test_end_date

    with open(config.dataset['stocks_path']) as f:
        stocks = [i.strip() for i in f]

    prices_close = Datamatrix_adjclose(
        file_paths=config.dataset['feature_path'],
        stocks=stocks,
        start_date=start_train,
        end_date=end_test
    )

    cun_path = os.path.join(cun_path, "ucrp")
    os.makedirs(cun_path, exist_ok=True)

    full_index = prices_close.index
    end_test_dt = full_index[-1]
    start_pos = full_index.get_loc(start_test)
    end_pos = full_index.get_loc(end_test_dt)

    # C2C：ret_t = (C_{t+1}/C_t - 1) 对齐到 t
    returns_c2c = compute_simple_returns(prices_close).shift(-1)

    ucrp = UCRP()
    cum_wealth, daily_returns = ucrp.predict(prices_close, returns_c2c, start_pos, end_pos)

    wealth_index = (1.0 + daily_returns).cumprod()

    ann_return = annualize_returns(daily_returns.mean())
    ann_vol = annualize_volatility(daily_returns.std())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((wealth_index / wealth_index.cummax()) - 1).min()
    cum_return = wealth_index.iloc[-1] - 1

    logger.info("\n====== 测试区间绩效 ======")
    logger.info("基线策略：ucrp (Close-to-Close)")
    logger.info(f"累计收益: {cum_return:.2%}")
    logger.info(f"年化收益: {ann_return:.2%}")
    logger.info(f"年化波动率: {ann_vol:.2%}")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {abs(max_dd):.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_index, label="UCRP Portfolio")
    plt.title("Cumulative Return (Close-to-Close)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, 'value.png'))
    plt.close()

    weights_df = pd.DataFrame(
        np.tile(ucrp.weights, (len(daily_returns), 1)),
        index=daily_returns.index,
        columns=prices_close.columns
    )
    weights_df.to_csv(os.path.join(cun_path, "daily_weights.csv"))
    pd.DataFrame({
        "portfolio_value": wealth_index,
        "daily_return": daily_returns
    }).to_csv(os.path.join(cun_path, "portfolio_performance.csv"))
