import shutil
'''
注意ret1和mov1是需要预测的label，不能作为features，需要特别注意
'''


import os
from glob import glob
import pathlib
import sys
import pandas as pd
import numpy as np
import warnings

def cal_feature_cs(df: pd.DataFrame):
    """
    生成横截面基础行情 + 当日K线形态：
      - 复权 OHLC
      - 成交额 amount
      - 当日振幅（amp）
      - 当日实体（body）
    """
    eps = 1e-12

    # ===== 1) 计算复权因子 =====
    if "AdjFactor" in df.columns and df["AdjFactor"].notna().any():
        adjfactor = df["AdjFactor"].astype(float)
    elif "adjclose" in df.columns and df["adjclose"].notna().any():
        adjfactor = df["adjclose"].astype(float) / (df["close"].astype(float) + eps)
    else:
        adjfactor = pd.Series(1.0, index=df.index)

    df["adjfactor"] = adjfactor

    # ===== 2) 复权 OHLC =====
    df["adjopen"]  = df["open"].astype(float)  * adjfactor
    df["adjhigh"]  = df["high"].astype(float)  * adjfactor
    df["adjlow"]   = df["low"].astype(float)   * adjfactor
    df["adjclose"] = (
        df["adjclose"].astype(float)
        if "adjclose" in df.columns
        else df["close"].astype(float) * adjfactor
    )

    # ===== 3) 成交额 =====
    if "amount" not in df.columns:
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)

    # ===== 4) 当日K线形态（无量纲） =====
    day_range = df["adjhigh"] - df["adjlow"] + eps
    df["amp"]  = day_range / (df["adjopen"] + eps)                 # 振幅
    df["body"] = (df["adjclose"] - df["adjopen"]) / day_range      # 实体方向

    # ===== 5) 最终保留字段 =====
    keep_cols = [
        "adjopen",
        "adjhigh",
        "adjlow",
        "adjclose",
        "amount",
        "amp",
        "body",
    ]
    return df[keep_cols].copy()


def main(inpath, outpath):
    features = [
        "adjopen",
        "adjhigh",
        "adjlow",
        "adjclose",
        "amount",
        "amp",
        "body",
    ]
    with open("utils/NAS100_pool.txt", "r") as f:
        stocks = [line.strip() for line in f.readlines()]

    pathlib.Path(outpath).mkdir(parents=True,exist_ok=True)
    paths = [os.path.join(inpath, f"{stock}.csv") for stock in stocks if os.path.exists(os.path.join(inpath, f"{stock}.csv"))]
    for path in paths:
        name = os.path.basename(path)

        df = pd.read_csv(path, index_col=0)
        df.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adjclose',
                   'Volume': 'volume'}, axis=1, inplace=True)
        df = cal_feature_cs(df)
        df = df.dropna()
        df.to_csv(os.path.join(outpath, name))

if __name__ == '__main__':
    main("Dataset/Nas100数据/raw",
         "Dataset/Nas100数据/feature")