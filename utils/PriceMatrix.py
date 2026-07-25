import os
import pandas as pd
import numpy as np


def process_files(file_paths, stocks, feature_cols):
    """
    从 feature 目录一次性加载并对齐价格与特征。
    """
    asset_data = {}
    common_dates = None

    # 1. 扫描所有 CSV 获取公共日期
    for stock in stocks:
        full_path = os.path.join(file_paths, f"{stock}.csv")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"CSV not found: {full_path}")

        df = pd.read_csv(full_path, index_col='Date', parse_dates=['Date'])
        asset_data[stock] = df

        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates.intersection_update(df.index)

    if not common_dates:
        raise ValueError("No common dates found across assets!")

    common_dates = sorted(list(common_dates))
    print(f"[PriceMatrix] Aligned {len(stocks)} stocks on {len(common_dates)} days.")

    # 2. 准备容器
    aligned_feats = []
    aligned_prices = []

    prices_name = ['adjopen', 'adjclose']

    # 3. 遍历提取 CSV 数据并对齐
    for stock in stocks:
        df = asset_data[stock].reindex(common_dates)

        # A. 提取特征 (Features)
        # 填充缺失值防止 NaN
        feat_val = df[feature_cols].ffill().fillna(0.0).values
        aligned_feats.append(feat_val)

        # B. 提取价格 (Prices)
        price_val = df[prices_name].ffill().fillna(0.0).values
        aligned_prices.append(price_val)

    # 4. 堆叠为 Numpy Array
    # (N, T, F)
    data_tensor = np.stack(aligned_feats, axis=0).astype(np.float32)
    # (N, T, 2)
    price_tensor = np.stack(aligned_prices, axis=0).astype(np.float32)

    return {
        'data': data_tensor,
        'prices': price_tensor,
        'dates': common_dates,
        'id2stock': {i: s for i, s in enumerate(stocks)},
        'stock2id': {s: i for i, s in enumerate(stocks)}
    }


# 以下代码获取每个资产的时序，及每个时间步的特征
def alphastock_files(file_paths,stocks,feature_cols):
    # 存储每个资产的数据
    asset_data = {}
    # 存储所有出现的时间点
    prices_name = ['adjopen','adjclose']
    # 存储所有资产共有的时间点（交集）
    common_dates = None

    # 遍历每个文件路径
    for stock in stocks:
        stock_path=stock+'.csv'
        # 读取CSV文件
        df = pd.read_csv(os.path.join(file_paths, stock_path),index_col='Date',parse_dates=['Date'])
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
    aligned_prices=[]
    # 存储id到股票名称的映射
    id_to_stocks = {}
    stocks_to_id={}

    for i, (asset_name, df) in enumerate(asset_data.items()):
        # 将数据按照所有日期进行重新索引
        df = df.reindex(common_dates)

        df_features=df[feature_cols]

        df_price = df[prices_name]


        # 将DataFrame转换为numpy数组
        aligned_data.append(df_features.values.T)
        aligned_prices.append(df_price.values.T)


        # 存储id到股票名称的映射
        id_to_stocks[i] = asset_name
        stocks_to_id[asset_name] = i

    # 将所有资产的数据堆叠成一个三维Tensor, 将时间步放在第2维感觉会好一些
    data_tensor = np.stack(aligned_data,  axis=0).transpose(0, 2, 1)
    prices = np.stack(aligned_prices, axis=0).transpose(0, 2, 1)




    return data_tensor, common_dates, id_to_stocks, stocks_to_id, prices

def Datamatrix_adjclose(file_paths,stocks,start_date,end_date):
    price_frames=[]
    for stock in stocks:
        stock_path=stock+'.csv'
        # 读取CSV文件
        df = pd.read_csv(os.path.join(file_paths, stock_path),parse_dates=['Date'], usecols=['Date','adjclose'])
        df = df.rename(columns={'adjclose': stock})
        df.set_index('Date', inplace=True)
        price_frames.append(df)
    # 按日期做外连接，自动对齐
    price_matrix = pd.concat(price_frames, axis=1, join='inner').sort_index()
    if start_date is None and end_date is None:
        return price_matrix.copy()
    return price_matrix.loc[start_date:end_date]

def Datamatrix_adjopen(file_paths,stocks,start_date,end_date):
    price_frames=[]
    for stock in stocks:
        stock_path=stock+'.csv'
        # 读取CSV文件
        df = pd.read_csv(os.path.join(file_paths, stock_path),parse_dates=['Date'], usecols=['Date','adjopen'])
        df = df.rename(columns={'adjopen': stock})
        df.set_index('Date', inplace=True)
        price_frames.append(df)
    # 按日期做外连接，自动对齐
    price_matrix = pd.concat(price_frames, axis=1, join='inner').sort_index()
    if start_date is None and end_date is None:
        return price_matrix.copy()
    return price_matrix.loc[start_date:end_date]

def Datamatrix_bull_label(file_paths,stocks,start_date,end_date):
    price_frames=[]
    for stock in stocks:
        stock_path=stock+'.csv'
        # 读取CSV文件
        df = pd.read_csv(os.path.join(file_paths, stock_path),parse_dates=['Date'], usecols=['Date','ssm3_pred'])
        df = df.rename(columns={'ssm3_pred': stock})
        df.set_index('Date', inplace=True)
        price_frames.append(df)
    # 按日期做外连接，自动对齐
    price_matrix = pd.concat(price_frames, axis=1, join='inner').sort_index()
    # 大于0.5的元素变为1，反之变为0
    if start_date is None and end_date is None:
        return price_matrix.copy()
    return price_matrix.loc[start_date:end_date]

# def Datamatrix_bull(file_paths,stocks,start_date=None,end_date=None):
#     price_frames=[]
#
#     for stock in stocks:
#         stock_path=stock+'.csv'
#         # 读取CSV文件
#         df = pd.read_csv(os.path.join(file_paths, stock_path),parse_dates=['Date'], usecols=['Date','Bull'])
#         df = df.rename(columns={'Bull': stock})
#         df.set_index('Date', inplace=True)
#         price_frames.append(df)
#     # 按日期做外连接，自动对齐
#     price_matrix = pd.concat(price_frames, axis=1, join='inner').sort_index()
#     if start_date is None and end_date is None:
#         return price_matrix.copy()
#     return price_matrix.loc[start_date:end_date]

def Datamatrix_bull(file_paths,stocks,start_date=None,end_date=None):
    price_frames=[]

    for stock in stocks:
        stock_path=stock+'.csv'
        # 读取CSV文件
        df = pd.read_csv(os.path.join(file_paths, stock_path),parse_dates=['Date'], usecols=['Date','Regime'])
        df['Regime'] = 1 - df['Regime']  # 假设 'Bull' 概率是 'Regime' 的补数
        df = df.rename(columns={'Regime': stock})
        df.set_index('Date', inplace=True)
        price_frames.append(df)
    # 按日期做外连接，自动对齐
    price_matrix = pd.concat(price_frames, axis=1, join='inner').sort_index()
    if start_date is None and end_date is None:
        return price_matrix.copy()
    return price_matrix.loc[start_date:end_date]

