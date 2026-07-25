import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import config
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

def load_weights(filename: str) -> pd.DataFrame:
    """加载 RL agent 输出的每日权重。"""
    return pd.read_csv(filename, index_col=0, parse_dates=True)

def load_prices(raw_folder: str, assets: list) -> pd.DataFrame:
    """加载多只资产的收盘价，返回 Date×Asset 的 DataFrame。"""
    dfs = []
    adj = []
    for a in assets:
        path = os.path.join(raw_folder, f"{a}.csv")
        tmp = pd.read_csv(path, usecols=["Date","adjopen"], parse_dates=["Date"])
        tmp = tmp.rename(columns={"adjopen": a}).set_index("Date")
        dfs.append(tmp)

    return pd.concat(dfs, axis=1).sort_index()

def compute_asset_contributions(weights: pd.DataFrame,
                                prices: pd.DataFrame,
                                transaction_cost_pct: float = None,
                                initial_value: float = 1.0) -> pd.DataFrame:
    """
    计算每个资产对组合的绝对累计贡献：
      daily_contrib_i[t] = prev_portfolio_value * weight_i[t-1] * pct_change_i[t]
    累积后得到 cum_contrib_i[t]；权重为0时贡献为0，累积保持不变。
    返回 DataFrame（index=日期，columns=资产名）。
    """
    transaction_cost_pct = (
        config.TRANSACTION_COST_RATE
        if transaction_cost_pct is None
        else float(transaction_cost_pct)
    )

    # 对齐日期并填充
    idx = weights.index.intersection(prices.index)
    w = weights.reindex(idx).fillna(method="ffill")
    p = prices.reindex(idx).ffill()

    assets = [c for c in w.columns if c.lower() != "cash"]
    n_steps, n_assets = len(idx), len(assets)

    # pct change of prices
    pct = p[assets].pct_change().fillna(0).values  # shape (n_steps, n_assets)
    w_arr = w[assets].values             # shape (n_steps, n_assets+1)

    contrib = np.zeros((n_steps, n_assets))
    port_val = np.zeros(n_steps)
    port_val[0] = initial_value

    for t in range(1, n_steps):
        prev_val = port_val[t-1]
        prev_w = w_arr[t-1]      #  w1, w2, ...]
        # 每资产当日贡献
        daily_contrib = prev_val * prev_w * pct[t]
        contrib[t] = contrib[t-1] + daily_contrib

        # 更新组合总价值（仅供检验）
        stock_ret = np.sum(prev_w * pct[t])
        new_val = prev_val * (1 + stock_ret)
        tc = new_val * transaction_cost_pct * np.sum(np.abs(prev_w - w_arr[t]))
        new_val -= tc
        port_val[t] = new_val

    return pd.DataFrame(contrib, index=idx, columns=assets)

def plot_price_and_contrib(weights: pd.DataFrame,
                           prices: pd.DataFrame,
                           contrib: pd.DataFrame,
                           save_path: str = None):
    """
    将价格折线与该资产累计贡献合并到同一子图：
      - 左轴：价格（灰虚线），在 weight>0 时以红点和红线高亮；
      - 右轴：累计贡献（实线+圆点）。
    """
    assets = contrib.columns
    idx = contrib.index
    w = weights.reindex(idx).fillna(method="ffill")
    p = prices.reindex(idx).ffill()

    fig, axes = plt.subplots(nrows=math.ceil(len(assets)/2), ncols=2, figsize=(14, 4*math.ceil(len(assets)/2)), sharex=True)
    axes = axes.flatten()

    for i, asset in enumerate(assets):
        ax = axes[i]
        ax2 = ax.twinx()

        price = p[asset]
        weight = w[asset]

        # 1. 画价格：灰色虚线
        ax.plot(idx, price, linestyle='--', color='lightgray',
                linewidth=1.2, label="Price")

        # 2. 高亮持仓区间：当天至下一天的红线和红点
        for t in range(len(idx) - 1):
            if weight.iloc[t] > config.red:
                ax.plot(idx[t:t+2], price.iloc[t:t+2],
                        '-', color='red', linewidth=2)
                ax.plot(idx[t], price.iloc[t],
                        'o', color='red', markersize=6)

        ax.set_ylabel("Price", fontsize=10)
        ax.set_title(asset, fontsize=12)
        ax.legend(loc="upper left", fontsize=8)

        # 3. 画累计贡献：蓝色实线+圆点
        ax2.plot(idx, contrib[asset],
                 '-o', markersize=5, linewidth=1.5,
                 color='blue', label="Cumulative Contribution")
        ax2.set_ylabel("Contribution", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8)

    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    print("Done")


