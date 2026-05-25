import os
import pandas as pd
import utils.config as config
dataset = config.dataset
file_paths = dataset['feature_path']
with open(dataset['stocks_path']) as f:
    stocks = [i.strip() for i in f.readlines()]
data_dir = "DeepAries/data/sh"   # 存放各股票 df/csv 的目录
out_path = "sh_data.csv"

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

dfs = []
train_date = _to_timestamp(getattr(config, "train_start_date", None))
valid_date = _to_timestamp(getattr(config, "valid_start_date", None))
test_date = _to_timestamp(getattr(config, "test_start_date", None))
test_end_date = _to_timestamp(getattr(config, "test_end_date", None))
for stock in stocks:
    stock_path = stock + '.csv'
    # 读取CSV文件
    df = pd.read_csv(os.path.join(file_paths, stock_path))
    df["tic"] = stock                # 加股票名
    df = df.rename(columns={"Date": "date"})
    df['date'] = pd.to_datetime(df['date'])
    df = df[
        (df["date"] >= train_date) &
        (df["date"] <=test_end_date)
        ]
    dfs.append(df)

# 合并
all_df = pd.concat(dfs, ignore_index=True)

# 统一列顺序（可选）
cols = [
    "date", "tic",
    "adjopen", "adjclose", "adjhigh", "adjlow",
    "amount", "amp", "body"
]
all_df = all_df[cols]
all_df = all_df.sort_values(["date", "tic"]).reset_index(drop=True)
# 保存
all_df.to_csv(os.path.join(data_dir, out_path), index=False)

print("saved:", out_path)
