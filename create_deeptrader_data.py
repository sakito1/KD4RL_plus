import os
import numpy as np
import pandas as pd
import utils.config as config

def _to_timestamp(x):
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x
    return pd.to_datetime(x)


def _date_to_idx(dates, target, side="left"):
    """
    dates: list[pd.Timestamp] 已排序
    target: pd.Timestamp
    side: "left" 返回第一个 >= target 的位置；"right" 返回第一个 > target 的位置
    """
    if target is None:
        return None
    arr = pd.DatetimeIndex(dates)
    return int(arr.searchsorted(target, side=side))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def deeptrader_files(
    prices_cols=["adjopen", "adjclose"],
    return_type="simple",      # "simple" or "log"
    fillna_method=None,        # None / "ffill" / "bfill"
    dtype=np.float32
):
    """
    仅使用资产数据构造 DeepTrader 输入，并默认资产关系图为 Identity(N)。

    Returns
    -------
    data_tensor : (T, N, F)
    common_dates: list[pd.Timestamp] length T
    id_to_stocks: dict[int, str]
    stocks_to_id: dict[str, int]
    prices      : (T, N, P)  P=len(prices_cols)
    rets        : (T, N)
    relation    : (N, N) identity matrix
    """
    dataset = config.dataset
    file_paths = dataset.get('ssm_data_path', dataset['feature_path'])
    with open(dataset['stocks_path']) as f:
        stocks = [i.strip() for i in f.readlines()]
    feature_cols = dataset['features_name']
    cun_path = config.deeptrader_data_path
    _ensure_dir(cun_path)

    asset_data = {}
    common_dates = None
    # 日期切分配置
    train_date = _to_timestamp(getattr(config, "train_start_date", None))
    valid_date = _to_timestamp(getattr(config, "valid_start_date", None))
    test_date = _to_timestamp(getattr(config, "test_start_date", None))
    test_end_date = _to_timestamp(getattr(config, "test_end_date", None))

    # 遍历每个文件路径
    for stock in stocks:
        stock_path = stock + '.csv'
        # 读取CSV文件
        df = pd.read_csv(os.path.join(file_paths, stock_path), index_col='Date', parse_dates=['Date'])
        df['ret'] = (df['adjclose']/df['adjclose'].shift(1)-1.0).fillna(0)
        # 将日期转换为pandas的日期时间类型
        df.index = pd.to_datetime(df.index)
        asset_data[stock] = df
        # 计算日期交集
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates.intersection_update(df.index)

    # 如果没有共同日期，抛出异常
    if not common_dates:
        raise ValueError("没有找到所有资产共有的日期！")

    # 对共同日期进行排序
    common_dates = sorted(common_dates)
    print(f"找到{len(common_dates)}个所有资产共有的日期")

    # 存储每个资产对齐后的数据
    aligned_data = []
    aligned_prices = []
    aligned_rets = []
    # 存储id到股票名称的映射
    id_to_stocks = {}
    stocks_to_id = {}

    for i, (asset_name, df) in enumerate(asset_data.items()):
        # 将数据按照所有日期进行重新索引
        df = df.reindex(common_dates)
        df_features = df[feature_cols]
        df_price = df[prices_cols]
        # 将DataFrame转换为numpy数组
        aligned_data.append(df_features.values.T)
        aligned_prices.append(df_price.values.T)
        aligned_rets.append(df['ret'].values.T)  # (T,)
        # ---------------------------------------------

        # 存储id到股票名称的映射
        id_to_stocks[i] = asset_name
        stocks_to_id[asset_name] = i

    # 将所有资产的数据堆叠成一个三维Tensor, 将时间步放在第2维感觉会好一些
    data_tensor = np.stack(aligned_data, axis=0).transpose(0, 2, 1)
    prices = np.stack(aligned_prices, axis=0).transpose(0, 2, 1)

    # 你原来写的 rets 悬空，这里补齐为 (N, T)
    rets = np.stack(aligned_rets, axis=0)

    # 5) 资产关系图：默认对角线矩阵（不预设关系）
    relation = np.eye(len(stocks), dtype=dtype)

    # 3) 保存 npy（到 cun_path）
    np.save(os.path.join(cun_path, "features.npy"), data_tensor)
    np.save(os.path.join(cun_path, "prices.npy"), prices)
    np.save(os.path.join(cun_path, "rets.npy"), rets)
    np.save(os.path.join(cun_path, "relation.npy"), relation)

# 4) 保存 split idx（按 T 轴索引）
    if train_date is None:
        train_date = common_dates[0]
    if valid_date is None or test_date is None:
        raise ValueError("valid_start_date / test_start_date 不能为空（用于切分 idx）")
    if test_end_date is None:
        test_end_date = common_dates[-1]

    tr_start = _date_to_idx(common_dates, train_date, side="left")
    va_start = _date_to_idx(common_dates, valid_date, side="left")
    te_start = _date_to_idx(common_dates, test_date, side="left")
    te_end_excl = _date_to_idx(common_dates, test_end_date, side="right")  # 包含 end

    T = len(common_dates)
    if not (0 <= tr_start < va_start <= te_start <= te_end_excl <= T):
        raise ValueError(
            f"切分索引不合法：T={T}, tr={tr_start}, va={va_start}, te={te_start}, te_end_excl={te_end_excl}\n"
            f"all: {common_dates[0]} -> {common_dates[-1]}"
        )

    with open(os.path.join(cun_path, "split_idx.txt"), "w", encoding="utf-8") as f:
        f.write("# 0-based, python slicing (end_excl)\n")
        f.write(f"# data_tensor: (N,T,F)={data_tensor.shape}\n")
        f.write(f"# prices:      (N,T,P)={prices.shape}\n")
        f.write(f"# rets:        (N,T)  ={rets.shape}  (adjopen-based)\n\n")

        f.write("[train]\n")
        f.write(f"start={tr_start}\nend_excl={va_start}\n")
        f.write(f"date_start={common_dates[tr_start]}\n")
        f.write(f"date_end={common_dates[va_start-1]}\n\n")

        f.write("[valid]\n")
        f.write(f"start={va_start}\nend_excl={te_start}\n")
        f.write(f"date_start={common_dates[va_start]}\n")
        f.write(f"date_end={common_dates[te_start-1]}\n\n")

        f.write("[test]\n")
        f.write(f"start={te_start}\nend_excl={te_end_excl}\n")
        f.write(f"date_start={common_dates[te_start]}\n")
        f.write(f"date_end={common_dates[te_end_excl-1]}\n\n")

    # 5) 保存所有日期到 txt（可选但建议）
    with open(os.path.join(cun_path, "dates.txt"), "w", encoding="utf-8") as f:
        for d in common_dates:
            f.write(str(d) + "\n")

    print(f"已保存到 {cun_path}")
    print(f"- data_tensor.npy {data_tensor.shape}")
    print(f"- prices.npy      {prices.shape}")
    print(f"- rets.npy        {rets.shape}")
    print(f"- relation.npy    {relation.shape}")
    print(f"- split_idx.txt / dates.txt")

if __name__ == "__main__":
    deeptrader_files()
