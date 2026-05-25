import math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from create_soft_regime_label import build_ensemble_soft_label, build_fullseries_segment_labels_k2, build_fullseries_prob_labels, build_smooth_ensemble_label

try:
    import utils.config as config

    FEATURE_NAMES = config.dataset["features_name"]
except Exception:
    config = None
    FEATURE_NAMES = None

# 定义 Label 列名
COL_LABEL_MAIN = "label_ensemble_soft"  # 主目标 (Soft)
COL_LABEL_AUX = "label_micro_raw"  # 辅助目标 (为了 Encoder 敏感度)


def ensure_labels_exist(df, csv_path):
    """
    检查 df 中是否存在所需 label 列。
    如果不存在，生成并保存回 csv_path。
    返回带有 label 的 df。
    """
    is_dirty = False

    # 1. 检查/生成 主目标 (Ensemble Soft Label)
    if COL_LABEL_MAIN not in df.columns:
        print(f"[Label Gen] {COL_LABEL_MAIN} missing for {os.path.basename(csv_path)}, generating...")
        # 调用集成生成器 (参数可调)
        soft_label, std_label = build_smooth_ensemble_label(
            df,
            price_col="adjclose",
            spans=[i for i in range(10, 30, 5)],  # 覆盖从周线到季线的趋势
            slope_tols=[0.0002, 0.0005, 0.0001],  # 覆盖对“平盘”的不同定义
            min_lens=[5, 10, 15, 21],  # 覆盖不同的抗噪级别
            sensitivity=4,  # Sigmoid 较平缓
            final_smooth_sigma=2.0  # 最后加一个 sigma=3 的高斯模糊
        )
        df[COL_LABEL_MAIN] = soft_label
        is_dirty = True
    else:
        print(f"[Label Gen] Found {COL_LABEL_MAIN}, skipping generation.")

    # 2. 检查/生成 辅助目标 (Micro Label - 用于 Encoder 训练)
    # 辅助目标通常需要非常敏感，帮助 Encoder 捕捉短期波动
    if COL_LABEL_AUX not in df.columns:
        print(f"[Label Gen] {COL_LABEL_AUX} missing, generating...")
        # 使用 K=2 生成一个比较敏感的 label
        _, _, labels_micro, _, _, _ = build_fullseries_prob_labels(
            df,
            price_col="adjclose",
            smooth_span=3,  # 短周期
            slope_tol=0.0001,  # 高敏感
            min_len=3,
            sensitivity=8.0
        )
        df[COL_LABEL_AUX] = labels_micro
        is_dirty = True

    # 3. 如果有更新，保存回 CSV
    if is_dirty:
        print(f"[Label Gen] Saving updates to {csv_path} ...")
        df.to_csv(csv_path)

    return df

import numpy as np
import pandas as pd


def causal_rolling_normalize(df, feat_cols, trend_macro, trend_micro, W=63):
    """
    有选择的因果滚动归一化 (Selective Causal Normalization)
    策略：
    1. 价格类 (OHLC): Log-Diff 模式 (相对于窗口末端的收盘价) -> 保留波动率尺度
    2. 量能类 (Vol/Amt): Log1p + Rolling Z-Score -> 防止地量导致的除法爆炸
    3. 形态类 (Amp/Body): 原样保留 (Pass-through) -> 本身就是比例，无需处理
    """
    # -------------------------------------------------------------------------
    # 1. 特征分组 (Feature Grouping)
    # -------------------------------------------------------------------------
    # 价格类：包含 adj, open, high, low, close 的列
    price_cols = [c for c in feat_cols if any(x in c for x in ['adjopen', 'adjhigh', 'adjlow', 'adjclose', 'amount'])]
    # 形态/比例类：amp, body 等已经是 Ratio 的列
    ratio_cols = [c for c in feat_cols if c in ['amp', 'body']]

    # -------------------------------------------------------------------------
    # 2. 数据准备与窗口切分
    # -------------------------------------------------------------------------
    # 提取原始数据矩阵
    X_raw = df[feat_cols].values.astype(np.float32)
    idx_all = df.index
    cp = df["adjclose"].to_numpy(dtype=np.float32)  # 用于后续跟踪

    T_all, D = X_raw.shape

    # 定义有效时间步
    tau_min = W - 1
    tau_max = T_all - 6  # 预留 label 计算空间

    if T_all == 0 or tau_max < tau_min:
        return (np.zeros((0, W, D), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.float32), idx_all[:0], np.zeros((0,), np.float32))

    taus = np.arange(tau_min, tau_max + 1, dtype=np.int64)
    T_valid = len(taus)

    # 预分配最终的归一化矩阵
    X_out = np.zeros((T_valid, W, D), dtype=np.float32)

    # 为了加速，先切分出原始窗口数据 (T_valid, W, D)
    # 注意：这里会消耗内存，显存不足时可改为循环内处理
    X_wins = np.zeros((T_valid, W, D), dtype=np.float32)
    for i, tau in enumerate(taus):
        X_wins[i] = X_raw[tau - W + 1: tau + 1]

    # -------------------------------------------------------------------------
    # 3. 分组归一化处理 (The Core Logic)
    # -------------------------------------------------------------------------

    # === A 组：价格类 (Div-Last / Log-Diff) ===
    # 公式：Log(P_t) - Log(P_last_close)
    # 效果：将窗口内所有价格转换为“相对于当前收盘价的涨跌幅”
    if price_cols:
        p_idxs = [feat_cols.index(c) for c in price_cols]
        P_wins = X_wins[:, :, p_idxs]  # (T, W, N_p)

        # 寻找基准：使用 adjclose 作为锚点 (Anchor)
        try:
            anchor_idx = feat_cols.index('adjclose')
            # 取每个窗口最后一个时间步的 close 作为基准
            # shape: (T, 1, 1) -> 广播到整个窗口
            P_anchor = X_wins[:, -1:, anchor_idx:anchor_idx + 1]
        except ValueError:
            # 如果没有 adjclose，退而求其次用最后一列价格
            P_anchor = P_wins[:, -1:, -1:]

        # Log-Diff 计算 (加 1e-8 防止 log 0)
        # 结果含义：0.05 代表比当前收盘价高 5%，-0.1 代表低 10%
        # 在高波动时期，这些数值的绝对值会变大；低波动时期会趋近于 0
        X_out[:, :, p_idxs] = np.log(P_wins + 1e-8) - np.log(P_anchor + 1e-8)

    # === C 组：形态/比例类 (Pass-Through) ===
    # 公式：保持原样 (或简单缩放)
    # 原因：Amp/Body 已经是归一化后的 Ratio，再做处理反而破坏分布
    if ratio_cols:
        r_idxs = [feat_cols.index(c) for c in ratio_cols]
        # 直接赋值
        X_out[:, :, r_idxs] = X_wins[:, :, r_idxs]

    # -------------------------------------------------------------------------
    # 4. 标签与辅助数据对齐
    # -------------------------------------------------------------------------
    y_target_macro = trend_macro[taus].astype(np.float32)
    y_target_micro = trend_micro[taus].astype(np.float32)
    idx_out = idx_all[taus]
    close_tau = cp[taus].astype(np.float32)

    return X_out, y_target_macro, y_target_micro, idx_out, close_tau

class MultiAssetWindows(Dataset):
    """
    现在 Dataset 会返回双重标签：(y_macro, y_micro)
    """

    def __init__(self, all_X, all_y_macro, all_y_micro, seq_len, stride=1):
        assert len(all_X) == len(all_y_macro) == len(all_y_micro)
        self.seq_len = seq_len
        self.stride = stride
        self.X_list = [torch.tensor(X, dtype=torch.float32) for X in all_X]
        self.y_macro_list = [torch.tensor(y, dtype=torch.float32) for y in all_y_macro]
        self.y_micro_list = [torch.tensor(y, dtype=torch.float32) for y in all_y_micro]

        self.samples = []
        for a, y_a in enumerate(self.y_macro_list):
            N = len(y_a)
            if N <= 0:
                continue
            if N < seq_len:
                self.samples.append((a, 0))
            else:
                s = 0
                while s + seq_len <= N:
                    self.samples.append((a, s))
                    s += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        a, s = self.samples[i]

        X_a = self.X_list[a]
        y_mac_a = self.y_macro_list[a]
        y_mic_a = self.y_micro_list[a]

        actual_len = min(self.seq_len, len(X_a) - s)

        return (X_a[s:s + actual_len],
                y_mac_a[s:s + actual_len],
                y_mic_a[s:s + actual_len],
                a)


# ---------------- Models (ResTCN / Encoder / SSM) ----------------
# 特征注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # # [保留] SEBlock 极小代价提升特征纯度
        # self.se = SEBlock(out_channels)
        self.act2 = nn.GELU()  # [新增] Residual 后的激活通常也需要
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.pad(x, (self.left_pad, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = F.pad(out, (self.left_pad, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        #
        # out = self.se(out)  # 关键：通道筛选

        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return F.relu(out + residual)


class MultiScale_Gated_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim=16, hidden_size=32, dropout=0.1, **kwargs):
        super().__init__()
        self.start_conv = nn.Conv1d(in_dim, hidden_size, 1)
        self.layers = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16]
        for d in dilations:
            self.layers.append(ResTCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=d, dropout=dropout))
        self.fusion_dim = hidden_size * len(dilations)
        self.fusion_ln = nn.LayerNorm(self.fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2), nn.GELU(),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim), nn.Sigmoid()
        )
        self.out_proj = nn.Linear(self.fusion_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x_win):
        x = x_win.transpose(1, 2)
        x = self.start_conv(x)
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        final_steps = [o[:, :, -1] for o in layer_outputs]
        cat_feat = torch.cat(final_steps, dim=1)
        normed_feat = self.fusion_ln(cat_feat)
        weights = self.gate(normed_feat)
        weighted_feat = cat_feat * weights
        z = self.out_proj(weighted_feat)
        z = self.ln(z)
        return z


def tv_loss(p, weight=1.0):
    if p.dim() == 3: p = p.squeeze(-1)
    if p.size(1) <= 1: return torch.tensor(0.0, device=p.device)
    diff = torch.abs(p[:, 1:] - p[:, :-1])
    return diff.mean() * weight


class HazardFusionDSSM(nn.Module):
    # (保持之前的 HazardFusionDSSM 不变，直接复用)
    def __init__(self, x_dim, h_dim=32, num_assets=1, asset_emb_dim=8, beta_init=1.0, learn_beta=True,
                 use_asset_stickiness=True, tau_emit=1.0):
        super().__init__()
        self.h_dim = int(h_dim)
        self.num_assets = int(num_assets)
        self.asset_emb_dim = int(asset_emb_dim)
        self.learn_beta = bool(learn_beta)
        self.use_asset_stickiness = bool(use_asset_stickiness)
        self.tau_emit = float(tau_emit)

        self.encoder = MultiScale_Gated_Encoder(in_dim=x_dim, out_dim=h_dim, hidden_size=h_dim)
        self.asset_emb = nn.Embedding(num_assets, asset_emb_dim)
        self.gru_cell = nn.GRUCell(input_size=h_dim + asset_emb_dim, hidden_size=h_dim)
        # Fusion Layer
        self.emit_head = nn.Sequential(nn.Linear(h_dim + asset_emb_dim, 32),
                                       nn.LayerNorm(32),  # [新增] 隔离 Encoder，防止 Aux Loss 冲击过大
                                       nn.GELU(),
                                       nn.Linear(32, 1))
        self.macro_head = nn.Sequential(
            nn.Linear(h_dim, 32),
            nn.LayerNorm(32),  # 隔离 h_t，防止 Loss 回传破坏 GRU 状态
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.trans_head = nn.Sequential(
            nn.Linear(h_dim * 2 + asset_emb_dim, 32),
            nn.LayerNorm(32),  # Pre-Activation Norm
            nn.GELU(),  # [优化] 使用 GELU 更平滑
            nn.Linear(32, 16),
            nn.LayerNorm(16),  # [新增] 内部层也加 Norm，防止梯度消失/爆炸
            nn.GELU(),  # [优化]
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

        self.raw_beta_emb = nn.Embedding(num_assets, 1)
        nn.init.constant_(self.raw_beta_emb.weight, float(beta_init))

    def beta(self,asset_id):
        raw = self.raw_beta_emb(asset_id)
        return F.softplus(raw) + 1e-6

    @staticmethod
    def _safe_logit(p, eps=1e-6):
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def forward(self, X, asset_id):
        B, T, W, D = X.shape
        X_flat = X.reshape(B * T, W, D)

        # 1. Encoder 提取特征
        z_flat = self.encoder(X_flat)
        z_seq = z_flat.view(B, T, -1)

        e_a = self.asset_emb(asset_id)
        e_seq = e_a.unsqueeze(1).expand(-1, T, -1)

        # Emission (Micro) Logits 计算
        # 这一步包含了整个序列的 micro prediction
        emit_in = torch.cat([z_seq, e_seq], dim=-1)
        emit_logits = self.emit_head(emit_in).squeeze(-1) / self.tau_emit

        beta_vals = self.beta(asset_id).unsqueeze(1)  # (B, 1, 1)

        # 列表容器
        p_list = []
        q_bear_list = []
        q_bull_list = []
        fused_logits_list = []
        macro_logits_list = []
        h_prev = torch.zeros(B, self.h_dim, device=X.device, dtype=X.dtype)
        p_prev = torch.sigmoid(emit_logits[:, 0])

        for t in range(T):
            z_t = z_seq[:, t, :]

            # --- [关键步骤] 宏观锚定预测 ---
            # 在 h_prev 还没有看到 z_t 之前，就让它预测当前的 macro trend
            # 这相当于要求：h_{t-1} -> y_t (预测未来)
            macro_logit_t = self.macro_head(h_prev).squeeze(-1)
            macro_logits_list.append(macro_logit_t)

            # A. 惯性判断 (Hazard)
            trans_in = torch.cat([h_prev, z_t, e_a], dim=-1)
            qs = self.trans_head(trans_in)
            q_bear_t = qs[:, 0]
            q_bull_t = qs[:, 1]
            q_bear_list.append(q_bear_t)
            q_bull_list.append(q_bull_t)

            # B. 贝叶斯更新 (SSM)
            p_prior = p_prev * (1.0 - q_bear_t) + (1.0 - p_prev) * q_bull_t
            fused_logit_t = self._safe_logit(p_prior) + beta_vals.squeeze() * emit_logits[:, t]
            p_t = torch.sigmoid(fused_logit_t)

            p_list.append(p_t)
            fused_logits_list.append(fused_logit_t)

            # C. 状态更新 (GRU)
            inp_t = torch.cat([z_t, e_a], dim=-1)
            h_prev = self.gru_cell(inp_t, h_prev)  # h_prev 变为 h_t
            p_prev = p_t

        return {
            "p": torch.stack(p_list, dim=1),
            "q_bear": torch.stack(q_bear_list, dim=1),
            "q_bull": torch.stack(q_bull_list, dim=1),
            "emit_logits": emit_logits,
            "fused_logits": torch.stack(fused_logits_list, dim=1),
            "macro_logits": torch.stack(macro_logits_list, dim=1)  # (B, T)
        }

    @torch.no_grad()
    def print_model_params(self):
        avg_beta = self.raw_beta_emb.weight.data.mean().item()
        real_beta = F.softplus(torch.tensor(avg_beta)).item()
        print(f"\n[SSM] h={self.h_dim}, emb={self.asset_emb_dim}, avg_beta={real_beta:.3f}")

# =========================================================================
#  3. 结构化约束 Loss (防抖 + 状态稀疏)
# =========================================================================
def tv_loss(p, weight=1.0):
    """
    [防抖动核心] Total Variation Loss。
    强迫预测概率 p 的曲线是平滑的，禁止高频锯齿状波动。
    对于 Soft Label 拟合，这是防止 Overfitting 噪声的关键。
    """
    if p.dim() == 3: p = p.squeeze(-1)
    if p.size(1) <= 1: return torch.tensor(0.0, device=p.device)

    # 惩罚相邻时间步的差异
    diff = torch.abs(p[:, 1:] - p[:, :-1])
    return diff.mean() * weight

# =========================================================================
#  Loss 组件定义
# =========================================================================

class ContinuousFocalLoss(nn.Module):
    """
    [数值逼近] 负责让 p 在数值上接近 y。
    保留这个组件，因为它能很好地处理 Soft Label 的难易样本挖掘。
    """

    def __init__(self, gamma=2.0, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 距离越远，权重越大。这很适合 Soft Label。
        distance = torch.abs(targets - probs)
        focal_weight = torch.pow(distance, self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        return loss


class DerivativeMatchingLoss(nn.Module):
    """
    [速度/形态逼近] 核心改进组件。
    替代传统的 TV Loss。

    逻辑：预测曲线的"斜率"应该等于真实标签的"斜率"。
    Loss = | (p_t - p_{t-1}) - (y_t - y_{t-1}) |

    优势：
    1. 平稳区：y差分≈0，等价于 TV Loss -> 防抖。
    2. 转折区：y差分很大，强迫 p 产生同样的差分 -> 防滞后。
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, p, y):
        if p.dim() == 3: p = p.squeeze(-1)

        # 计算 p 的一阶差分
        p_diff = p[:, 1:] - p[:, :-1]

        # 计算 y 的一阶差分 (Soft Label 的变化率)
        with torch.no_grad():
            y_diff = y[:, 1:] - y[:, :-1]

        # 让两者的变化率尽可能一致
        # 使用 L1 Loss 保持梯度的稀疏性和鲁棒性
        loss = torch.abs(p_diff - y_diff).mean()

        return loss * self.weight


def calc_refractory_loss(q, window=10):
    """
    [稀疏约束] 保持不变。SSM 内部状态依然需要稀疏性。
    """
    loss = 0.0
    for k in range(1, window):
        loss += (q[:, :-k] * q[:, k:]).mean()
    return loss


# =========================================================================
#  辅助计算：动态权重 (保留用于 Focal Loss)
# =========================================================================
def get_trend_emphasis_weights(y_target, kernel_size=5, scale=5.0):
    """
    即使有了导数匹配，我们依然希望在 Focal Loss 层面给予转折点更高的关注度。
    """
    with torch.no_grad():
        # 计算梯度幅度
        diff = torch.abs(y_target[:, 1:] - y_target[:, :-1])
        diff = F.pad(diff, (1, 0), "constant", 0)

        # 膨胀
        mask = diff.float().unsqueeze(1)
        dilated = F.max_pool1d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).squeeze(1)

        # 权重：趋势变化越剧烈，Focal Loss 权重越大
        weights = 1.0 + dilated * scale
    return weights


def shifted_bce_loss(logits, targets):
    """
    [新增] 平移预测 Loss (Forecasting Loss)

    Logits: h_t 产生的预测，意图预测 y_{t+1}
    Targets: y_t

    逻辑：
    - Logits[:, t] 应该匹配 Targets[:, t+1]
    - 我们将 Targets 向左平移一位。
    - 缺失的最后一位用倒数第二位填充 (Forward Fill)，保持形状一致。
    """
    # 1. 目标左移：Target[:, 1:]
    # 2. 填充最后一位：Target[:, -1:]
    target_shifted = torch.cat([targets[:, 1:], targets[:, -1:]], dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, target_shifted)
    return loss

# =========================================================================
#  训练循环
# =========================================================================

def train_loop(model, loader, opt_enc, opt_ssm, device, epoch,
               warmup_enc_epochs=5,  # 阶段1: 练眼睛 (Encoder)
               warmup_gru_epochs=10,
               lam_fuse=10.0,  # 主任务权重 (数值逼近)
               lam_emit=1.0,  # 辅助任务权重
               lam_deriv=10.0,  # [核心] 导数匹配权重 (替代 TV Loss)
               lam_macro_aux=5.0,
               lam_refra=20.0,
               lam_tv=1.0,# 状态稀疏权重
               **kwargs):
    model.train()
    criterion_focal = ContinuousFocalLoss(gamma=2.0, reduction='none').to(device)
    criterion_deriv = DerivativeMatchingLoss(weight=1.0).to(device)  # 内部weight设1，外部控制
    criterion_aux = nn.BCEWithLogitsLoss().to(device)
    criterion_macro = nn.BCEWithLogitsLoss().to(device)
    # --- 判定当前阶段 ---
    # 注意：这里假设 epoch 是从 1 开始传入的
    if epoch <= warmup_enc_epochs:
        phase = "PHASE_1_ENC_ONLY"
    elif epoch <= (warmup_enc_epochs + warmup_gru_epochs):
        phase = "PHASE_2_GRU_ONLY"
    else:
        phase = "PHASE_3_JOINT"
    # 日志累加
    log_loss_enc = 0.0
    log_loss_ssm = 0.0
    ns = 0
    for X, y_target, y_aux, asset_id in loader:
        X, asset_id = X.to(device), asset_id.to(device)
        y_target = y_target.to(device)  # Soft Label
        y_aux = y_aux.to(device)
        out = model(X, asset_id)

        p = out["p"]
        q_bear = out["q_bear"]
        q_bull = out["q_bull"]
        fused_logits = out["fused_logits"]
        emit_logits = out["emit_logits"]
        macro_logits = out["macro_logits"]
        # 微观训练
        loss_emit = criterion_aux(emit_logits, y_aux)
        if phase == "PHASE_1_ENC_ONLY":
            # === 阶段 A: 热身期 (Warmup) ===
            # 策略：只练眼睛 (Encoder)，不练脑子 (SSM)
            # 目的：让 z_t 具备基本的物理意义，避免 GRU 拟合随机噪音

            # 在热身期，我们临时把权重设为 1.0，让它学快点
            loss_total_enc = 10.0 * loss_emit

            opt_enc.zero_grad()
            opt_ssm.zero_grad()

            loss_total_enc.backward()  # 梯度只回传给 Encoder

            opt_enc.step()
            # 注意：opt_ssm 不动！

            # 记录日志
            log_loss_enc += loss_emit.item()
        elif phase == "PHASE_2_GRU_ONLY":
            # -----------------------------------------------
            # 阶段 2: 锁死 Encoder，只练 GRU
            # -----------------------------------------------
            # 目标：让 GRU 适应现有的 z_t，建立宏观逻辑
            # 防止 GRU 初期的巨大梯度破坏 Encoder
            # --- 1. Value Matching (主任务) ---
            w_trend = get_trend_emphasis_weights(y_target, scale=5.0)
            raw_focal = criterion_focal(fused_logits, y_target)
            loss_fuse = (raw_focal * w_trend).mean()
            # --- 2. Velocity/Shape Matching (速度逼近) ---
            loss_deriv_val = criterion_deriv(p, y_target)
            # 宏观辅助训练
            loss_macro_val = criterion_macro(macro_logits, y_target)
            # --- 3. Auxiliary & Regularization ---
            loss_refra_val = calc_refractory_loss(q_bear) + calc_refractory_loss(q_bull)
            # 1. TV Loss: 压制 p 的高频噪声 (核心防抖)
            loss_tv_val = tv_loss(p, weight=lam_tv)
            # Total Loss
            loss = (lam_fuse * loss_fuse) + \
                   (lam_emit * loss_emit) + \
                   (lam_macro_aux * loss_macro_val) + \
                   (lam_deriv * loss_deriv_val) + \
                   (lam_refra * loss_refra_val) + \
                   (loss_tv_val)
            opt_enc.zero_grad()
            opt_ssm.zero_grad()

            loss.backward()

            # [关键] 只更新 SSM，Encoder 保持静止
            torch.nn.utils.clip_grad_norm_(model.gru_cell.parameters(), 1.0)
            opt_ssm.step()

            log_loss_enc += loss_emit.item()
            log_loss_ssm += loss.item()
        else:
            # --- 1. Value Matching (主任务) ---
            w_trend = get_trend_emphasis_weights(y_target, scale=5.0)
            raw_focal = criterion_focal(fused_logits, y_target)
            loss_fuse = (raw_focal * w_trend).mean()
            # --- 2. Velocity/Shape Matching (速度逼近) ---
            loss_deriv_val = criterion_deriv(p, y_target)
            # 宏观辅助训练
            loss_macro_val = criterion_macro(macro_logits, y_target)
            # --- 3. Auxiliary & Regularization ---
            loss_refra_val = calc_refractory_loss(q_bear) + calc_refractory_loss(q_bull)
            # 1. TV Loss: 压制 p 的高频噪声 (核心防抖)
            loss_tv_val = tv_loss(p, weight=lam_tv)
            # Total Loss
            loss = (lam_fuse * loss_fuse) + \
                   (lam_emit * loss_emit) + \
                   (lam_macro_aux * loss_macro_val) +\
                   (lam_deriv * loss_deriv_val) + \
                   (lam_refra * loss_refra_val)+ \
                   (loss_tv_val)

            opt_enc.zero_grad()
            opt_ssm.zero_grad()

            loss.backward()  # 梯度流贯通全网

            # C. 梯度裁剪 (保护机制)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # D. 双管齐下
            opt_enc.step()  # lr=1e-4 (微调模式)
            opt_ssm.step()  # lr=1e-3 (主训模式)

            # 记录日志
            log_loss_enc += loss_emit.item()
            log_loss_ssm += loss.item()

        ns += 1

        # 格式化输出 Phase 状态，方便监控
        status_map = {
            "PHASE_1_ENC_ONLY": "[P1:Enc]",
            "PHASE_2_GRU_ONLY": "[P2:GRU]",
            "PHASE_3_JOINT": "[P3:All]"
        }
    return log_loss_enc / max(ns, 1), log_loss_ssm / max(ns, 1), status_map[phase]


@torch.no_grad()
def eval_loop(model, loader, device,
              lam_fuse=10.0, lam_emit=1.0, lam_deriv=10.0,
              lam_macro_aux=5.0,lam_refra=20.0,lam_tv=1.0,**kwargs):
    model.eval()
    criterion_focal = ContinuousFocalLoss(gamma=2.0, reduction='none').to(device)
    criterion_deriv = DerivativeMatchingLoss(weight=1.0).to(device)
    criterion_aux = nn.BCEWithLogitsLoss().to(device)
    criterion_macro = nn.BCEWithLogitsLoss().to(device)
    total, ns = 0.0, 0
    log_loss_enc = 0.0
    for X, y_target, y_aux, asset_id in loader:
        X, asset_id = X.to(device), asset_id.to(device)
        y_target, y_aux = y_target.to(device), y_aux.to(device)

        out = model(X, asset_id)

        w_trend = get_trend_emphasis_weights(y_target, scale=5.0)
        raw_focal = criterion_focal(out["fused_logits"], y_target)
        loss_fuse = (raw_focal * w_trend).mean()

        loss_deriv_val = criterion_deriv(out["p"], y_target)
        loss_macro_val = criterion_macro(out["macro_logits"], y_target)
        loss_emit = criterion_aux(out["emit_logits"], y_aux)
        log_loss_enc += loss_emit.item()
        loss_refra_val = calc_refractory_loss(out["q_bear"]) + calc_refractory_loss(out["q_bull"])

        # 1. TV Loss: 压制 p 的高频噪声 (核心防抖)
        loss_tv_val = tv_loss(out['p'], weight=lam_tv)
        # Total Loss
        loss = (lam_fuse * loss_fuse) + \
               (lam_emit * loss_emit) + \
               (lam_deriv * loss_deriv_val) + \
               (lam_macro_aux * loss_macro_val)+\
               (lam_refra * loss_refra_val) + \
               (loss_tv_val)
        total += float(loss.item())
        ns += 1
    return total / max(ns, 1), log_loss_enc/ max(ns, 1)


def get_reg_defaults():
    """
    推荐参数配置：
    lam_fuse=10.0:  保证数值拟合是第一优先级。
    lam_deriv=10.0: [关键] 强力约束 p 的变化率必须紧跟 y。
                    这比 TV Loss 更安全，因为它是双向约束（既防抖又防滞后）。
    lam_refra=20.0: 保持状态稀疏，这对于 SSM 的可解释性很重要。
    """
    return dict(
        lam_fuse=10.0,
        lam_emit=5.0,
        lam_deriv=10.0,
        lam_refra=20.0,
        lam_macro_aux=1.0,
        lam_tv=1.0,
        lam_entropy=0.0,
        lam_q=0.01,
        lam_switch=1.0,
        pos_weight_switch=50
    )
@torch.no_grad()
def infer_prob_asset(model, X_np, y_macro_np, asset_id, device):
    X = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.tensor([asset_id], dtype=torch.long, device=device)
    out = model(X, a)
    p = out["p"].squeeze(0).cpu().numpy()
    q_bear = out["q_bear"].squeeze(0).cpu().numpy()
    q_bull = out["q_bull"].squeeze(0).cpu().numpy()
    pred = (p > 0.5).astype(np.int64)
    true = (y_macro_np > 0.5).astype(np.int64)
    return p, q_bear, q_bull, pred, true


def build_dual_optimizers(model, lr_enc=1e-4, lr_ssm=1e-3, weight_decay=1e-6):
    """
    [新增] 构建双优化器
    - opt_enc: 负责 Encoder 和 Emit Head (微观视觉)
    - opt_ssm: 负责 GRU, Trans Head, Macro Head (宏观逻辑)
    """
    # 1. 筛选 Encoder 相关的参数
    # 包括特征提取器本身，以及用于监督它的 emit_head
    enc_params = list(model.encoder.parameters()) + list(model.emit_head.parameters())

    # 2. 筛选 SSM 逻辑相关的参数
    # 技巧：用 id() 排除掉已经放进 enc_params 的参数
    enc_param_ids = set(map(id, enc_params))
    ssm_params = [p for p in model.parameters() if id(p) not in enc_param_ids]

    # 3. 创建两个优化器
    # Encoder 通常需要更细腻的学习率 (低)
    opt_enc = torch.optim.AdamW(enc_params, lr=lr_enc, weight_decay=weight_decay)

    # SSM 逻辑部分通常需要更强的动力 (高)
    opt_ssm = torch.optim.AdamW(ssm_params, lr=lr_ssm, weight_decay=weight_decay)

    return opt_enc, opt_ssm


def plot_price_true_pred(dates, price, true_soft, pred_prob, title, outpath, pred_inertial=None):
    """
    绘制价格与趋势标签对比图。
    新增功能：如果传入 pred_inertial，则绘制 3 行子图 (True / Raw / Inertial)。
    """
    # 1. 数据预处理
    true01 = (np.asarray(true_soft) > 0.5).astype(int)
    pred01 = (np.asarray(pred_prob) > 0.5).astype(int)

    if len(price) == 0: return
    price = np.asarray(price, dtype=float)

    # 定义背景色：0=下跌(红), 1=上涨(绿)
    colors = {0: (0.9, 0.1, 0.1, 0.15), 1: (0.1, 0.7, 0.1, 0.15)}

    def shade(ax, labs):
        if labs is None or len(labs) == 0: return
        cur = int(labs[0])
        start = 0
        for i in range(1, len(labs)):
            if int(labs[i]) != cur:
                ax.axvspan(dates[start], dates[i - 1], color=colors[cur], linewidth=0)
                start = i
                cur = int(labs[i])
        ax.axvspan(dates[start], dates[-1], color=colors[cur], linewidth=0)

    # 2. 动态决定子图数量 (2行还是3行)
    has_inertial = pred_inertial is not None
    rows = 3 if has_inertial else 2
    fig_height = 10 if has_inertial else 7  # 3行时把图拉高一点

    fig, axes = plt.subplots(rows, 1, figsize=(14, fig_height), sharex=True)

    # 3. 绘制第一行：真实标签 (Target)
    ax1 = axes[0]
    ax1.plot(dates, price, lw=1.3, color="black")
    shade(ax1, true01)
    ax1.set_title(title + " (Target Label > 0.5)")
    ax1.set_ylabel("Price")

    # 4. 绘制第二行：原始预测 (Raw Prediction)
    ax2 = axes[1]
    ax2.plot(dates, price, lw=1.3, color="black")
    shade(ax2, pred01)
    ax2.set_title(title + " (SSM Raw Pred > 0.5)")
    ax2.set_ylabel("Price")

    # 5. [新增] 绘制第三行：惯性监控信号 (Inertial Monitor)
    if has_inertial:
        ax3 = axes[2]
        # 惯性信号通常已经是 0/1 整数，不需要 threshold
        monitor_signal = np.asarray(pred_inertial).astype(int)

        ax3.plot(dates, price, lw=1.3, color="black")
        shade(ax3, monitor_signal)
        # 标题高亮显示，方便区分
        ax3.set_title(title + " (Inertial Monitor Signal)", color="darkblue", fontweight="bold")
        ax3.set_ylabel("Price")

    plt.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=150)
        plt.close()
    else:
        plt.close()





