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

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_feature(df):
    # intermediate values
    df['adjclose'] = df['close'] * df['adjfactor']
    df['adjopen'] = df['open'] * df['adjfactor']
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1
    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    df["weekday"] = pd.to_datetime(df.index).weekday
    df["day"] = pd.to_datetime(df.index).day
    df["month"] = pd.to_datetime(df.index).month

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1'], inplace=True)

def cal_target(df):
    df['ret_open'] = df['adjopen'].pct_change()
    df['ret_close'] = df['adjclose'].pct_change()
    # 这边有很明显的注明
    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)

def main(inpath, outpath):
    features = [
        'adjfactor',
        'kmid2',
        'kup2',
        'klow',
        'klow2',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
    ]
    with open("utils/NAS100_pool.txt", "r") as f:
        stocks = [line.strip() for line in f.readlines()]
    # paths = glob(os.path.join(inpath, "*.csv"))
    # paths.sort()  # 按字母顺序排序
    pathlib.Path(outpath).mkdir(parents=True,exist_ok=True)
    paths = [os.path.join(inpath, f"{stock}.csv") for stock in stocks if os.path.exists(os.path.join(inpath, f"{stock}.csv"))]
    for path in paths:
        name = os.path.basename(path)
        print(name)
        df = pd.read_csv(path, index_col=0)
        df.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'AdjFactor': 'adjfactor'}, axis=1, inplace=True)
        cal_feature(df)
        cal_target(df)

        df = df.iloc[60:-1].reset_index()
        df = df.reset_index(drop=True)
        df = df.dropna()
        df.to_csv(os.path.join(outpath, name))

if __name__ == '__main__':
    main("Dataset/Nas100数据/raw",
         "Dataset/Nas100数据/ssm_feature")