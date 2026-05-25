import utils.config as config
from Baseline import(
    baseline_anticor,
    baseline_BH,
    baseline_markowitz,
    baseline_olmar,
    baseline_ucrp,
    baseline_wmamr,
    baseline_ssm,
)
import matplotlib.pyplot as plt
import os
from glob import glob
import pandas as pd
from Train.deep_baseline import deep_baseline

def baseline(cun_path, logger, include_deep=True, deep_smoke=False):
    baseline_ssm(cun_path, logger)
    baseline_anticor(cun_path, logger)
    baseline_BH(cun_path, logger)
    baseline_markowitz(cun_path, logger)
    baseline_olmar(cun_path, logger)
    baseline_ucrp(cun_path, logger)
    baseline_wmamr(cun_path, logger)
    if include_deep:
        deep_baseline(cun_path, logger, smoke=deep_smoke)
    raw_path = config.dataset['feature_path']
    start_train = config.train_start_date
    end_train = config.train_end_date
    end_train_dt = pd.to_datetime(end_train)


    logger.info('------Begin Baseline------')
    csv_files=[]

    with open(config.dataset['stocks_path']) as f:
        stocks = f.readlines()
        for stock in stocks:
            stock = stock.strip()
            csv_files.append(os.path.join(raw_path, f"{stock}.csv"))
    stocks_info = ', '.join(stocks)
    logger.info('stocks_info:{}'.format(stocks_info))
    n = len(csv_files)
    fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n), sharex=True)
    logger.info('begin plot stocks trend...'
                'start_train: {}, end_train: {}'.format(start_train, end_train))

    for i, file in enumerate(csv_files):
        # 读取CSV时解析日期并确保价格列为数值类型
        df = pd.read_csv(file)
        # 转换价格列为数值类型（处理可能的非数字值）
        df['Date'] = pd.to_datetime(df['Date'])
        # 开盘价
        ax=axes[i]
        ax.plot(df['Date'], df['adjopen'], label='Open Price adj', color='blue')
        ax.plot(df['Date'], df['adjclose'], label='Close Price adj', color='orange')
        ax.axvline(x=end_train_dt, color='red', linestyle='--', label='Train End')
        ax.set_title(f"{stocks[i]} Open and Close Price")
        ax.legend()
        ax.set_ylabel("Price")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(cun_path, "all_stocks_open_close_subplots.png"))
    plt.close()




    logger.info('------End Baseline------')
