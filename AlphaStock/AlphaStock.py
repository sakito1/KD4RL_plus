import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HiddenAttn(nn.Module):
    """
    针对LSTM输出的hn（所有时间步隐藏态）设计注意力模块，完全遵循AlphaStock论文3.2节思想：
    1. 以hn的「最后时间步隐藏态」为查询（Q），捕捉时序特征的最终压缩信息；
    2. 以hn的「所有时间步隐藏态」为键值（K/V），覆盖完整时序信息；
    3. 通过注意力加权融合全局依赖，输出优于单一隐藏态的资产表示。
    """
    def __init__(self, hidden_dim, dropout=0.2):
        super(HiddenAttn, self).__init__()
        # 注意力参数（严格对应论文公式13-14：α_k = w^T·tanh(W1·h_k + W2·h_K)）
        self.w_attn = nn.Parameter(torch.randn(hidden_dim))  # 注意力权重向量
        self.W1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # 对所有时间步隐藏态h_k的投影
        self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # 对最后时间步隐藏态h_K的投影
        self.dropout = nn.Dropout(dropout)  # 防止过拟合，参考论文实验设置（论文5.1节dropout=0.2）
        self.hidden_dim = hidden_dim

    def forward(self, hn, batch_size, num_nodes):
        """
        输入：
            hn: (window_len, batch*num_nodes, hidden_dim) → LSTM最后一层所有时间步的隐藏态（对应论文3.2节的h_1~h_K）
            batch_size: 批次大小 → 用于重塑维度
            num_nodes: 资产数量 → 用于重塑维度（对应论文中的「股票数量I」）
        输出：
            attn_feat: (batch_size, num_nodes, hidden_dim) → 融合注意力的资产时序表示（对应论文3.2节的r^(i)）
        """
        # -------------------------- Step 2：确定注意力的Q（查询）与K/V（键值） --------------------------
        # Q：最后时间步的隐藏态（论文3.2节的h_K，代表时序特征的最终压缩结果）
        query = hn[:, -1, :]  # (B*N, H) → 取每个资产的最后时间步隐藏态
        query = self.dropout(query)  # 防止过拟合

        # K/V：所有时间步的隐藏态（论文3.2节的h_1~h_K，代表完整的时序信息）
        key = hn  # (B*N, T, H)
        value = hn
        key = self.dropout(key)
        value = self.dropout(value)

        # -------------------------- Step 3：计算注意力权重（论文公式13-14） --------------------------
        # 1. 投影计算：W1·key（对所有h_k投影）、W2·query（对h_K投影）
        # key投影：(B*N, T, H) → (B*N, T, H)
        key_proj = torch.matmul(key, self.W1)
        # query投影：(B*N, H) → (B*N, 1, H)（广播适配时间步维度T）
        query_proj = torch.matmul(query, self.W2).unsqueeze(1)

        # 2. 注意力分数α_k：α_k = w_attn^T · tanh(key_proj + query_proj)（论文公式14）
        # tanh激活：融合key与query的交互信息，避免梯度消失
        tanh_out = torch.tanh(key_proj + query_proj)  # (B*N, T, H)
        # 点积计算分数：(B*N, T, H) · (H,) → (B*N, T)
        alpha = torch.matmul(tanh_out, self.w_attn.unsqueeze(-1)).squeeze(-1)

        # 3. softmax归一化：得到注意力权重（论文公式13：ATT(h_K, h_k) = exp(α_k)/Σexp(α_k)）
        attn_weights = F.softmax(alpha, dim=-1).unsqueeze(-1)  # (B*N, T, 1)

        # -------------------------- Step 4：注意力加权融合（论文公式12） --------------------------
        # 加权求和：value * 注意力权重 → 融合所有时间步的关键信息
        attn_feat = torch.sum(value * attn_weights, dim=1)  # (B*N, H)

        # -------------------------- Step 5：恢复原始维度（适配资产分组） --------------------------
        # 从 (B*N, H) 重塑为 (B, N, H)，与AssetLSTM输出格式一致，便于后续策略网络使用
        attn_feat = attn_feat.reshape(batch_size, num_nodes, self.hidden_dim)

        return attn_feat

class AssetLSTMATTN(nn.Module):
    """处理每个资产时序的LSTM模块"""

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2):
        super(AssetLSTMATTN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        self.ATTN = HiddenAttn(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        # x: (batch, num_nodes, window_len, in_features)
        batch_size, num_nodes, window_len, in_features = x.shape

        # 重塑为 (batch * num_nodes, window_len, in_features) 以便并行处理所有资产
        x = x.reshape(-1, window_len, in_features)

        # LSTM处理
        hn, (_, _) = self.lstm(x)  # hn: ( batch*num_nodes, window_len, hidden_dim)
        # 注意力融合LSTM所有时间步隐藏态，生成资产时序表示
        attn_feat = self.ATTN(hn, batch_size, num_nodes)  # (batch, num_nodes, hidden_dim)
        return attn_feat


class CAAN(nn.Module):
    """论文3.3：Cross-Asset Attention Network（跨资产注意力网络）
    输入：LSTM-HA输出的股票表示r (stock_num, hidden_size2)、价格排名c (stock_num,)
    输出：Winner Score s (stock_num,) —— 表示股票成为“赢家”的概率
    """

    def __init__(self, hidden_dim, dropout_p=0.5):
        """
        Args:
            hidden_dim: 输入资产表示r的维度（即LSTM-HA的输出维度，论文中对应r(i)的维度）
            dropout_p: Dropout丢弃概率（用于防止过拟合，训练时生效）
        """
        super().__init__()
        self.hidden_dim = hidden_dim  # 对应论文中的Dk（Key向量维度）

        # 1. Query/Key/Value投影层（论文公式14）
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)  # W(Q)：r(i) → q(i)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)  # W(K)：r(i) → k(i)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)  # W(V)：r(i) → v(i)

        # 2. Winner Score全连接层（论文公式18）
        self.W_s = nn.Linear(hidden_dim, 1)  # w(s)：a(i) → 线性输出
        self.b_s = nn.Parameter(torch.zeros(1))  # e(s)：偏置项

        # Dropout层（可选，用于正则化）
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, r):
        """
        前向传播：基于自注意力建模资产间相关性，生成Winner Score
        Args:
            r: 资产表示矩阵，shape=(stock_num, hidden_dim)，stock_num为资产总数
        Returns:
            s: Winner Score向量，shape=(stock_num,)
        """
        stock_num = r.size(0)

        # Step 1：生成Query/Key/Value向量（论文公式14）
        q = self.W_Q(r)  # (stock_num, hidden_dim) —— 每个资产的Query
        k = self.W_K(r)  # (stock_num, hidden_dim) —— 每个资产的Key
        v = self.W_V(r)  # (stock_num, hidden_dim) —— 每个资产的Value

        # 可选：Dropout正则化（对投影后向量进行随机丢弃）
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Step 2：计算资产间原始相关性βij（论文公式15）
        # q与k的点积：shape=(stock_num, stock_num)，其中β[i][j] = q(i)⊤ · k(j)
        beta = torch.matmul(q, k.transpose(1, 2))
        # 缩放：除以√Dk（Dk=hidden_dim），避免点积结果过大
        beta = beta / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=beta.device))

        # Step 3：注意力权重归一化（论文公式17）
        # Softmax沿j维度（对每个i，归一化所有j的βij）
        attn_weights = F.softmax(beta, dim=1)  # (stock_num, stock_num)
        # 可选：对注意力权重进行Dropout（进一步正则化）
        attn_weights = self.dropout(attn_weights)

        # Step 4：注意力加权融合Value（论文公式16）
        # a(i) = Σj [SATT(q(i),k(j)) · v(j)]，shape=(stock_num, hidden_dim)
        a = torch.matmul(attn_weights, v)

        # Step 5：生成Winner Score（论文公式18）
        # s(i) = sigmoid(w(s)⊤ · a(i) + e(s))
        s = self.W_s(a).squeeze(-1)  # 线性变换：(stock_num, hidden_dim) → (stock_num,)
        s = s + self.b_s  # 加偏置项
        s = torch.sigmoid(s)  # 激活到0-1区间

        return s

class LSTMDRL(nn.Module):
    def __init__(self,hidden_dim1,hidden_dim2,in_features,trade_num):
        super(LSTMDRL, self).__init__()

        self.asset_lstm = AssetLSTMATTN(input_dim=in_features,
                                        hidden_dim=hidden_dim1,
                                        num_layers=2,
                                        dropout=0.2)
        self.caan = CAAN(
            hidden_dim=hidden_dim2,
            dropout_p= 0.2
        )
        self.trade_num = trade_num

    def forward(self, inputs):
        """
        前向传播
        inputs: 输入数据列表，每个元素形状为[batch_size, num_stocks, window_size, num_features]
        trade_num: 每个时间步的交易股票数量
        """
        asset_features = self.asset_lstm(inputs) # 输出形状: [batch_size, num_stocks, hidden_dim1]
        score = self.caan(asset_features)

        # 计算买卖目标
        top_val, top_indices = torch.topk(score, k=self.trade_num)

        # 计算买卖比例（使用softmax）
        buy_por = F.softmax(top_val, dim=1)

        # 创建一个与原始分数相同维度的零张量
        actions = torch.zeros_like(score)
        # 生成批次索引 [0,1,...,11]，并扩展为 [12,1] 以匹配top_indices的维度
        batch_idx = torch.arange(actions.size(0), device=actions.device).unsqueeze(1)
        # 将buy_por填充到对应的topk索引位置
        actions[batch_idx, top_indices] = buy_por

        return actions


class alphastock(nn.Module):
    def __init__(self,hidden_dim1,hidden_dim2,in_features,trade_num):
        super(alphastock, self).__init__()

        self.asset_lstm = AssetLSTMATTN(input_dim=in_features,
                                        hidden_dim=hidden_dim1,
                                        num_layers=2,
                                        dropout=0.2)
        self.caan = CAAN(
            hidden_dim=hidden_dim2,
            dropout_p= 0.2
        )
        self.trade_num = trade_num

    def forward(self, inputs):
        """
        前向传播
        inputs: 输入数据列表，每个元素形状为[batch_size, num_stocks, window_size, num_features]
        trade_num: 每个时间步的交易股票数量
        """
        asset_features = self.asset_lstm(inputs) # 输出形状: [batch_size, num_stocks, hidden_dim1]
        score = self.caan(asset_features)

        # 计算买卖目标
        top_val, top_indices = torch.topk(score, k=self.trade_num)

        # 计算买卖比例（使用softmax）
        buy_por = F.softmax(top_val, dim=1)

        # 创建一个与原始分数相同维度的零张量
        actions = torch.zeros_like(score)
        # 生成批次索引 [0,1,...,11]，并扩展为 [12,1] 以匹配top_indices的维度
        batch_idx = torch.arange(actions.size(0), device=actions.device).unsqueeze(1)
        # 将buy_por填充到对应的topk索引位置
        actions[batch_idx, top_indices] = buy_por

        return actions, top_indices