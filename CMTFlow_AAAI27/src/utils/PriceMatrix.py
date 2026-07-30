import os
import pandas as pd
import numpy as np
import torch


def _standardize_feature_columns(df, feature_cols):
    df = df.copy()
    eps = 1e-12

    if 'adjfactor' in df.columns:
        adjfactor = pd.to_numeric(df['adjfactor'], errors='coerce').fillna(1.0)
    elif {'adjclose', 'close'}.issubset(df.columns):
        close = pd.to_numeric(df['close'], errors='coerce')
        adjclose = pd.to_numeric(df['adjclose'], errors='coerce')
        adjfactor = (adjclose / (close + eps)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        adjfactor = pd.Series(1.0, index=df.index)

    derived_specs = {
        'adjopen': ('open', adjfactor),
        'adjclose': ('close', adjfactor),
        'adjhigh': ('high', adjfactor),
        'adjlow': ('low', adjfactor),
    }
    for target, (source, factor) in derived_specs.items():
        if target not in df.columns and source in df.columns:
            df[target] = pd.to_numeric(df[source], errors='coerce') * factor

    if 'amount' not in df.columns:
        if {'close', 'volume'}.issubset(df.columns):
            df['amount'] = (
                pd.to_numeric(df['close'], errors='coerce')
                * pd.to_numeric(df['volume'], errors='coerce')
            )
        elif 'volume' in df.columns:
            df['amount'] = pd.to_numeric(df['volume'], errors='coerce')

    if 'amp' not in df.columns or 'body' not in df.columns:
        if {'adjopen', 'adjclose', 'adjhigh', 'adjlow'}.issubset(df.columns):
            day_range = (
                pd.to_numeric(df['adjhigh'], errors='coerce')
                - pd.to_numeric(df['adjlow'], errors='coerce')
                + eps
            )
            if 'amp' not in df.columns:
                df['amp'] = day_range / (pd.to_numeric(df['adjopen'], errors='coerce') + eps)
            if 'body' not in df.columns:
                df['body'] = (
                    pd.to_numeric(df['adjclose'], errors='coerce')
                    - pd.to_numeric(df['adjopen'], errors='coerce')
                ) / day_range

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns after derivation: {missing}")

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def Datamatrix_ssm_hidden(hidden_dir, stocks, all_dates):
    """
    专门加载 .pt 文件中的 h 和 z，并对齐到 all_dates
    """
    all_dates = pd.to_datetime(all_dates)

    h_frames = []
    z_frames = []

    H_dim = None
    Z_dim = None

    for code in stocks:
        path = os.path.join(hidden_dir, f"{code}_ssm3_states.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[SSM] hidden file not found: {path}")

        # 加载数据
        ck = torch.load(path, map_location="cpu")
        h = ck["h"].numpy()  # (T_i, H)
        z = ck["z"].numpy()  # (T_i, Z)
        date_idx = pd.to_datetime(ck["date_idx"])

        # 记录维度以便检查
        if H_dim is None: H_dim = h.shape[1]
        if Z_dim is None: Z_dim = z.shape[1]

        # 构造 DataFrame 进行对齐
        df_h = pd.DataFrame(h, index=date_idx)
        df_z = pd.DataFrame(z, index=date_idx)

        # Reindex 到公共时间轴，缺失填 0
        h_aligned = df_h.reindex(all_dates).fillna(0.0).values
        z_aligned = df_z.reindex(all_dates).fillna(0.0).values

        h_frames.append(h_aligned)
        z_frames.append(z_aligned)

    # 堆叠 -> (N, T, Dim)
    h_tensor = np.stack(h_frames, axis=0).astype(np.float32)
    z_tensor = np.stack(z_frames, axis=0).astype(np.float32)

    return h_tensor, z_tensor


def process_files(file_paths, stocks, feature_cols):
    """
    [最终版] 一次性加载所有数据

    CSV 包含: 价格, features, 以及 ssm3_p, ssm3_q_bear, ssm3_q_bull
    PT  包含: h, z
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

    # SSM 概率信号容器 (从 CSV 读取)
    aligned_p = []
    aligned_q_bear = []
    aligned_q_bull = []

    # 定义需要从 CSV 读取的特殊列
    ssm_cols = ['ssm3_p', 'ssm3_q_bear', 'ssm3_q_bull']
    prices_name = ['adjopen', 'adjclose']

    # 3. 遍历提取 CSV 数据并对齐
    for stock in stocks:
        df = asset_data[stock].reindex(common_dates)
        df = _standardize_feature_columns(df, feature_cols)

        # A. 提取特征 (Features)
        # 填充缺失值防止 NaN
        feat_val = df[feature_cols].fillna(method='ffill').fillna(0.0).values
        aligned_feats.append(feat_val)

        # B. 提取价格 (Prices)
        price_val = df[prices_name].fillna(method='ffill').fillna(0.0).values
        aligned_prices.append(price_val)

        # C. 提取 SSM 概率 (Probabilities from CSV)
        # 确保列存在，如果不存在则填 0.5 或 0.1
        if set(ssm_cols).issubset(df.columns):
            p_val = df['ssm3_p'].fillna(0.5).values
            q_bear_val = df['ssm3_q_bear'].fillna(0.1).values
            q_bull_val = df['ssm3_q_bull'].fillna(0.1).values
        else:
            print(f"Warning: SSM columns missing in {stock}.csv, using defaults.")
            p_val = np.zeros(len(common_dates)) + 0.5
            q_bear_val = np.zeros(len(common_dates)) + 0.1
            q_bull_val = np.zeros(len(common_dates)) + 0.1

        aligned_p.append(p_val)
        aligned_q_bear.append(q_bear_val)
        aligned_q_bull.append(q_bull_val)

    # 4. 堆叠为 Numpy Array
    # (N, T, F)
    data_tensor = np.stack(aligned_feats, axis=0).astype(np.float32)
    # (N, T, 2)
    price_tensor = np.stack(aligned_prices, axis=0).astype(np.float32)

    # (N, T) -> 概率信号
    p_tensor = np.stack(aligned_p, axis=0).astype(np.float32)
    q_bear_tensor = np.stack(aligned_q_bear, axis=0).astype(np.float32)
    q_bull_tensor = np.stack(aligned_q_bull, axis=0).astype(np.float32)

    # 5. 加载 .pt 文件中的 h 和 z
    if file_paths is not None:
        h_tensor, z_tensor = Datamatrix_ssm_hidden(file_paths, stocks, common_dates)
    else:
        h_tensor = None
        z_tensor = None

    # 6. 打包返回
    # 构造 ssm 字典
    ssm_dict = {
        'h': h_tensor,  # [N, T, H_dim] (from PT)
        'z': z_tensor,  # [N, T, Z_dim] (from PT)
        'p': p_tensor,  # [N, T] (from CSV)
        'q_bear': q_bear_tensor,  # [N, T] (from CSV)
        'q_bull': q_bull_tensor  # [N, T] (from CSV)
    }

    return {
        'data': data_tensor,
        'prices': price_tensor,
        'ssm': ssm_dict,
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

