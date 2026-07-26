import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


# ==============================================================================
# 0. 轻量时序模块: 因果卷积块 (Causal TCN)
# ==============================================================================

class CausalConv1dBlock(nn.Module):
    """轻量因果卷积块：只做左侧 padding，保证不泄露未来。输入输出 shape: [B, C, T]"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)

        # 如果通道数不一致，用 1x1 做残差投影（保持轻量）
        self.res_proj = None
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]
        if self.pad_len > 0:
            x_pad = F.pad(x, (self.pad_len, 0))
        else:
            x_pad = x

        y = self.conv(x_pad)
        y = self.act(y)

        # 残差（对齐到相同的 T）
        res = x
        if self.res_proj is not None:
            res = self.res_proj(res)
        res = res[:, :, -y.shape[-1]:]

        y = y + res

        # LayerNorm over channel dim -> [B, T, C]
        y = y.permute(0, 2, 1)
        y = self.norm(y)
        y = y.permute(0, 2, 1)
        return y


# ==============================================================================
# 1. 基础组件 (Attention & LSTM & CAAN)
# ==============================================================================

class HiddenAttn(nn.Module):
    """
    针对LSTM输出的hn（所有时间步隐藏态）设计注意力模块
    完全遵循 AlphaStock 论文 3.2 节思想
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super(HiddenAttn, self).__init__()
        self.w_attn = nn.Parameter(torch.randn(hidden_dim))
        self.W1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, hn, batch_size, num_nodes):
        # Q: 最后时间步 (B*N, H)
        query = self.dropout(hn[:, -1, :])
        # K, V: 所有时间步 (B*N, T, H)
        key = self.dropout(hn)
        value = self.dropout(hn)

        # Attention Score Calculation
        key_proj = torch.matmul(key, self.W1)
        query_proj = torch.matmul(query, self.W2).unsqueeze(1)

        tanh_out = torch.tanh(key_proj + query_proj)
        alpha = torch.matmul(tanh_out, self.w_attn.unsqueeze(-1)).squeeze(-1)

        attn_weights = F.softmax(alpha, dim=-1).unsqueeze(-1)
        attn_feat = torch.sum(value * attn_weights, dim=1)

        return attn_feat.reshape(batch_size, num_nodes, self.hidden_dim)


class AssetLSTMATTN(nn.Module):
    """处理每个资产时序的LSTM模块"""

    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
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
        # x: [B, N, T, F]
        batch_size, num_nodes, window_len, in_features = x.shape
        x = x.reshape(-1, window_len, in_features)

        hn, _ = self.lstm(x)
        attn_feat = self.ATTN(hn, batch_size, num_nodes)  # [B, N, H]
        return attn_feat


class CAAN(nn.Module):
    """Cross-Asset Attention Network (跨资产注意力网络)"""

    def __init__(self, hidden_dim, dropout_p=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, r):
        # r: [B, N, H] -> 这里的 batch 处理需要注意，通常 CAAN 是针对单个 Batch 内的 Assets
        # 假设 r 是 [B, N, H]
        q = self.W_Q(r)
        k = self.W_K(r)
        v = self.W_V(r)

        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Attention: [B, N, N]
        beta = torch.matmul(q, k.transpose(-2, -1))
        beta = beta / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=beta.device))

        attn_weights = F.softmax(beta, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output: [B, N, H]
        a = torch.matmul(attn_weights, v)
        return a

class OuterAC(nn.Module):
    """
    Outer 层：共享 encoder（AssetLSTMATTN + CAAN + Fusion），actor/critic 头分离。
    提供 pi/value/forward。
    """

    def __init__(self, lstm_dim, caan_dim, in_dim, hidden_dim, trade_num, num_nodes):
        super().__init__()
        self.trade_num = trade_num
        self.hidden_dim = hidden_dim

        self.asset_lstm = AssetLSTMATTN(input_dim=in_dim, hidden_dim=lstm_dim, num_layers=2, dropout=0.1)
        self.caan = CAAN(hidden_dim=caan_dim, dropout_p=0.1)

        # 共享特征：CAAN(caan_dim) + last_day_feat(in_dim) + weights_drift(1)
        fusion_in = caan_dim + in_dim + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
        )

        # actor head
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

        # critic head：market pooling + weights embedding
        self.market_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.w_proj = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # DeepAries-style auxiliary head: predict each stock's future return
        # from its own encoded representation, instead of predicting the
        # realized return of the already selected portfolio.
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, outer_input, weights_drift):
        asset_features = self.asset_lstm(outer_input)           # [B, N, lstm_dim]
        processed = self.caan(asset_features)                   # [B, N, caan_dim]
        last_day_feat = outer_input[:, :, -1, :]                # [B, N, in_dim]
        w_expanded = weights_drift.unsqueeze(-1)                # [B, N, 1]
        combined = torch.cat([processed, last_day_feat, w_expanded], dim=-1)
        feat = self.fusion(combined)                            # [B, N, hidden_dim]
        return feat

    def get_dist(self, feat):
        mu = self.mean(feat).squeeze(-1)                        # [B, N]
        log_std = torch.clamp(self.log_std(feat).squeeze(-1), -5, 2)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def pi(self, outer_input, weights_drift, deterministic=False):
        feat = self.encode(outer_input, weights_drift)
        dist = self.get_dist(feat)

        raw_action = dist.mean if deterministic else dist.rsample()  # [B, N]
        scores = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(torch.clamp(1 - scores.pow(2), min=1e-5))
        log_prob = log_prob.sum(dim=1)

        top_val, top_indices = torch.topk(scores, k=self.trade_num, dim=1)
        buy_por = F.softmax(top_val, dim=1)
        actions = torch.zeros_like(raw_action).scatter(1, top_indices, buy_por)

        entropy = dist.entropy().mean()                     # [B]
        return actions, raw_action, log_prob, entropy, scores

    def value(self, outer_input, weights_drift):
        feat = self.encode(outer_input, weights_drift)          # [B, N, hidden_dim]
        scores = torch.matmul(feat, self.market_query.transpose(0, 1))  # [B, N, 1]
        attn = F.softmax(scores, dim=1)
        market_rep = torch.sum(feat * attn, dim=1)              # [B, hidden_dim]

        w_rep = self.w_proj(weights_drift)                      # [B, hidden_dim//2]
        v_in = torch.cat([market_rep, w_rep], dim=-1)
        return self.v_head(v_in)                                # [B, 1]

    def pred_stock_return(self, outer_input, weights_drift):
        feat = self.encode(outer_input, weights_drift)           # [B, N, hidden_dim]
        return self.pred_head(feat).squeeze(-1)                  # [B, N]

    def pred_return(self, outer_input, weights_drift, action_weights=None):
        return self.pred_stock_return(outer_input, weights_drift)

    def forward(self, outer_input, weights_drift, deterministic=False):
        actions, raw_action, log_prob, entropy, scores = self.pi(
            outer_input, weights_drift, deterministic=deterministic
        )
        v = self.value(outer_input, weights_drift)
        return actions, raw_action, log_prob, entropy, scores, v


class InnerAC(nn.Module):
    """
    Inner 层：共享 encoder + actor/critic heads。

    [更新]
    - 使用两层 LSTM 编码每个资产的短窗序列。
    - 使用两层 query attention：每层都用最后一个时间步表征作为 query，
      从整段序列里提取关键时序信息，生成每只股票最终 embedding。

    输入：inner_input: [B, N, T, F]
    """

    def __init__(self, in_features, hidden_dim, max_boundary, dropout=0.2,
                 tcn_kernel_size: int = 3):
        super().__init__()
        self.max_boundary = max_boundary
        self.hidden_dim = hidden_dim

        self.inner_lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_attn2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_norm2 = nn.LayerNorm(hidden_dim)
        self.last_temporal_attn1 = None
        self.last_temporal_attn2 = None

        # shared fusion: (self H) + base(1) + drift(1)
        fusion_dim = hidden_dim + 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
        )

        # actor head
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

        # critic aggregation head（沿用原 InnerCritic 的聚合思路）
        self.alpha_query = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_lstm_attn(self, inner_input: torch.Tensor) -> torch.Tensor:
        """LSTM + 两层时序注意力：将每资产窗口序列编码为 [B, N, H]。"""
        B, N, T, M = inner_input.shape
        x = inner_input.reshape(B * N, T, M)
        seq, _ = self.inner_lstm(x)                               # [B*N, T, H]

        query1 = seq[:, -1:, :]                                    # [B*N, 1, H]
        attn1, weights1 = self.temporal_attn1(
            query=query1,
            key=seq,
            value=seq,
            need_weights=True,
            average_attn_weights=True,
        )
        query2 = self.temporal_norm1(query1 + attn1)
        attn2, weights2 = self.temporal_attn2(
            query=query2,
            key=seq,
            value=seq,
            need_weights=True,
            average_attn_weights=True,
        )
        feat = self.temporal_norm2(query2 + attn2).squeeze(1)      # [B*N, H]

        self.last_temporal_attn1 = weights1.detach().reshape(B, N, T)
        self.last_temporal_attn2 = weights2.detach().reshape(B, N, T)
        return feat.reshape(B, N, self.hidden_dim)

    def encode(self, inner_input, base_used_t, weight_drift):
        # 1) per-asset short-window encoding
        node_feat = self._encode_lstm_attn(inner_input)            # [B, N, H]

        # 2) fuse with current portfolio state
        combined = torch.cat(
            [node_feat, base_used_t.unsqueeze(-1), weight_drift.unsqueeze(-1)],
            dim=-1,
        )
        feat = self.fusion(combined)                              # [B, N, H]
        return feat

    def get_dist(self, feat):
        mu = self.mu_head(feat).squeeze(-1)                # [B, N]
        log_std = torch.clamp(self.log_std_head(feat).squeeze(-1), -5, 0)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def pi(self, inner_input, base_used_t, weight_drift, deterministic=False):
        feat = self.encode(inner_input, base_used_t, weight_drift)
        dist = self.get_dist(feat)
        raw_signal = dist.mean if deterministic else dist.rsample()  # [B, N]

        log_prob = dist.log_prob(raw_signal).sum(dim=1)             # [B]
        entropy = dist.entropy().mean()                      # [B]
        return raw_signal, log_prob, entropy

    def value(self, inner_input, base_used_t, weight_drift):
        feat = self.encode(inner_input, base_used_t, weight_drift)  # [B, N, H]

        beta_rep = torch.sum(feat * base_used_t.unsqueeze(-1), dim=1)  # [B, H]
        attn_scores = self.alpha_query(feat).squeeze(-1)               # [B, N]
        alpha = F.softmax(attn_scores, dim=-1).unsqueeze(-1)           # [B, N, 1]
        alpha_rep = torch.sum(feat * alpha, dim=1)                     # [B, H]
        global_rep = torch.cat([beta_rep, alpha_rep], dim=-1)
        return self.value_head(global_rep)                             # [B, 1]

    def pred_next_return(self, inner_input, base_used_t, weight_drift):
        feat = self.encode(inner_input, base_used_t, weight_drift)      # [B, N, H]
        return self.pred_head(feat).squeeze(-1)                         # [B, N]

    def forward(self, inner_input, base_used_t, weight_drift, deterministic=False):
        raw_signal, log_prob, entropy = self.pi(
            inner_input, base_used_t, weight_drift, deterministic=deterministic
        )
        v = self.value(inner_input, base_used_t, weight_drift)
        return raw_signal, log_prob, entropy, v

    def build_inner_action_simple(
            self,
            inner_input,
            base_used_t,
            weight_drift,
            alpha: float,
            deterministic: bool = False,
            eps: float = 1e-12,
    ):
        """
        最简：
          score_sample ~ Normal(mu,std)
          target = softmax(score_sample on mask)
          w_new = (1-alpha)*base + alpha*target
        返回：w_new, score_sample, logp, entropy, v
        """
        feat = self.encode(inner_input, base_used_t, weight_drift)
        dist = self.get_dist(feat)  # 你原来的 get_dist，不改它

        score_sample = dist.mean if deterministic else dist.rsample()  # [B,N]
        logp = dist.log_prob(score_sample).sum(dim=1)  # [B]
        ent = dist.entropy().mean()  # [B]

        # 1) mask 内 softmax 得到 target
        mask = (base_used_t > 0).float()  # [B,N]
        score_masked = score_sample.masked_fill(mask < 0.5, float("-inf"))
        target = F.softmax(score_masked, dim=1) * mask
        target = target / (target.sum(dim=1, keepdim=True) + eps)  # 必要：保证 target sum=1

        # 2) 凸组合得到最终执行权重
        a = float(alpha)
        w_new = (1.0 - a) * base_used_t + a * target

        # 3) value
        v = self.value(inner_input, base_used_t, weight_drift)  # [B,1]
        return w_new, score_sample, logp, ent, v


class MonitorAC(nn.Module):
    """
    Hold-exit controller.

    The controller uses the last controller_window days of the normalized
    feature window in asset_state=[B,N,T,F].
    """

    def __init__(
            self,
            z_dim,
            h_dim,
            port_state_dim,
            hidden_dim=32,
            action_dim=None,
            min_hold=20,
            max_hold=40,
            tau_min=0.5,
            tau_max=0.9,
            policy_temperature=10.0,
            ret_scale=0.05,
            drawdown_scale=0.10,
            asset_in_dim=None,
            controller_window=15,
            weight_floor=1e-6,
            eval_switch_threshold=0.5,
            init_exit_bias=None,
            switch_adv_logit_coef=0.0,
            switch_adv_logit_scale=0.02,
            switch_adv_logit_detach=False,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.h_dim = int(h_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_hold = int(min_hold)
        self.max_hold = int(max_hold)
        self.ret_scale = max(float(ret_scale), 1e-6)
        self.drawdown_scale = max(float(drawdown_scale), 1e-6)
        self.asset_in_dim = int(asset_in_dim) if asset_in_dim is not None else None
        self.controller_window = max(1, int(controller_window))
        self.weight_floor = max(float(weight_floor), 0.0)
        self.eval_switch_threshold = min(1.0, max(0.0, float(eval_switch_threshold)))
        self.switch_adv_logit_coef = float(switch_adv_logit_coef)
        self.switch_adv_logit_scale = max(float(switch_adv_logit_scale), 1e-8)
        self.switch_adv_logit_detach = bool(switch_adv_logit_detach)

        input_dim = self.asset_in_dim if self.asset_in_dim is not None else self.z_dim
        self.asset_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fallback_projection = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_dim = 5
        self.action_state_dim = 3
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim + self.state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attn1_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn1_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn1_v = nn.Linear(hidden_dim, hidden_dim)
        self.query2_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attn2_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn2_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn2_v = nn.Linear(hidden_dim, hidden_dim)

        # The two branches deliberately receive different information.
        # Risk estimates whether the current holding is becoming unsafe, while
        # advantage compares the Manager candidate with that current holding.
        risk_in = hidden_dim * 2 + self.state_dim
        advantage_in = hidden_dim * 4 + self.action_state_dim
        self.risk_mlp = nn.Sequential(
            nn.Linear(risk_in, hidden_dim),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.advantage_mlp = nn.Sequential(
            nn.Linear(advantage_in, hidden_dim),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        fusion_in = hidden_dim * 2
        self.switch_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.exit_head = nn.Linear(hidden_dim, 1)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.switch_adv_head = nn.Linear(hidden_dim, 1)
        if init_exit_bias is not None:
            nn.init.constant_(self.exit_head.bias, float(init_exit_bias))

    @staticmethod
    def _normalize_weights(weights):
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    def _soft_hold_weights(self, weights):
        w = weights.clamp_min(0.0) + self.weight_floor
        return self._normalize_weights(w)

    def _state_features(self, port_state, hold_weights):
        time_norm = port_state[:, :1].clamp(0.0, 1.0)
        remaining_norm = (1.0 - time_norm).clamp(0.0, 1.0)
        drawdown_norm = torch.tanh(port_state[:, 1:2].clamp_min(0.0) / self.drawdown_scale)
        seg_return_norm = torch.tanh(port_state[:, 2:3] / self.ret_scale)
        concentration = torch.sum(hold_weights.pow(2), dim=1, keepdim=True)
        return torch.cat([time_norm, remaining_norm, seg_return_norm, drawdown_norm, concentration], dim=-1)

    def _switch_action_features(self, hold_weights, switch_weights):
        turnover = torch.sum(torch.abs(switch_weights - hold_weights), dim=1, keepdim=True)
        switch_concentration = torch.sum(switch_weights.pow(2), dim=1, keepdim=True)
        overlap = torch.sum(torch.minimum(hold_weights, switch_weights), dim=1, keepdim=True)
        return torch.cat([turnover, switch_concentration, overlap], dim=-1)

    def _encode_asset_sequence(self, asset_state):
        x = asset_state
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.shape[2] > self.controller_window:
            x = x[:, :, -self.controller_window:, :]
        if self.asset_in_dim is not None and x.shape[-1] != self.asset_in_dim:
            raise ValueError(
                f"asset_state feature dim={x.shape[-1]} does not match asset_in_dim={self.asset_in_dim}"
            )
        bsz, num_assets, seq_len, feat_dim = x.shape
        seq, _ = self.asset_lstm(x.reshape(bsz * num_assets, seq_len, feat_dim))
        return seq.reshape(bsz, num_assets, seq_len, self.hidden_dim)

    def _temporal_attention_global(self, query, seq, q_proj, k_proj, v_proj):
        q = q_proj(query).unsqueeze(1).unsqueeze(2)
        k = k_proj(seq)
        v = v_proj(seq)
        logits = torch.sum(q * k, dim=-1) / (self.hidden_dim ** 0.5)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        return torch.sum(weights * v, dim=2)

    def _temporal_attention_per_asset(self, query, seq, q_proj, k_proj, v_proj):
        q = q_proj(query).unsqueeze(2)
        k = k_proj(seq)
        v = v_proj(seq)
        logits = torch.sum(q * k, dim=-1) / (self.hidden_dim ** 0.5)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        return torch.sum(weights * v, dim=2)

    def decision_stats(
            self,
            weights_drift,
            port_state,
            switch_action=None,
            asset_state=None,
            hold_exec_weights=None,
            switch_exec_weights=None,
            remaining_horizon=None,
    ):
        asset_seq = self._encode_asset_sequence(asset_state)
        hold_weights = self._soft_hold_weights(weights_drift)
        switch_weights = self._soft_hold_weights(switch_action) if switch_action is not None else hold_weights
        advantage_hold_weights = (
            self._soft_hold_weights(hold_exec_weights)
            if hold_exec_weights is not None
            else hold_weights
        )
        advantage_switch_weights = (
            self._soft_hold_weights(switch_exec_weights)
            if switch_exec_weights is not None
            else switch_weights
        )
        if remaining_horizon is None:
            horizon_gate = torch.ones(
                (hold_weights.shape[0], 1),
                dtype=hold_weights.dtype,
                device=hold_weights.device,
            )
        else:
            horizon_gate = torch.as_tensor(
                remaining_horizon,
                dtype=hold_weights.dtype,
                device=hold_weights.device,
            ).reshape(hold_weights.shape[0], -1)[:, :1].clamp(0.0, 1.0)

        last_emb = asset_seq[:, :, -1, :]
        portfolio_last = torch.sum(last_emb * hold_weights.unsqueeze(-1), dim=1)
        advantage_hold_last = torch.sum(
            last_emb * advantage_hold_weights.unsqueeze(-1),
            dim=1,
        )
        advantage_switch_last = torch.sum(
            last_emb * advantage_switch_weights.unsqueeze(-1),
            dim=1,
        )
        state5 = self._state_features(port_state, hold_weights)
        action_state = self._switch_action_features(
            advantage_hold_weights,
            advantage_switch_weights,
        )

        query = self.query_mlp(torch.cat([portfolio_last, state5], dim=-1))
        asset_ctx1 = self._temporal_attention_global(
            query, asset_seq, self.attn1_q, self.attn1_k, self.attn1_v
        )
        query2 = self.query2_mlp(torch.cat([
            asset_ctx1,
            query.unsqueeze(1).expand(-1, asset_ctx1.shape[1], -1),
        ], dim=-1))
        asset_ctx2 = self._temporal_attention_per_asset(
            query2, asset_seq, self.attn2_q, self.attn2_k, self.attn2_v
        )
        portfolio_ctx = torch.sum(asset_ctx2 * hold_weights.unsqueeze(-1), dim=1)
        advantage_hold_ctx = torch.sum(
            asset_ctx2 * advantage_hold_weights.unsqueeze(-1),
            dim=1,
        )
        advantage_switch_ctx = torch.sum(
            asset_ctx2 * advantage_switch_weights.unsqueeze(-1),
            dim=1,
        )

        risk_feat = torch.cat([
            portfolio_last,
            portfolio_ctx,
            state5,
        ], dim=-1)
        advantage_feat = torch.cat([
            advantage_hold_last,
            advantage_hold_ctx,
            (advantage_switch_last - advantage_hold_last) * horizon_gate,
            (advantage_switch_ctx - advantage_hold_ctx) * horizon_gate,
            action_state,
        ], dim=-1)
        risk_embedding = self.risk_mlp(risk_feat)
        advantage_embedding = self.advantage_mlp(advantage_feat)
        value_feat = torch.cat([risk_embedding, advantage_embedding], dim=-1)
        switch_embedding = self.switch_mlp(value_feat)

        # Final switching is learned from both embeddings.  In particular,
        # switch_advantage_pred is not injected through a fixed coefficient or
        # detached path; policy gradients therefore reach both branches.
        exit_logit = self.exit_head(switch_embedding).squeeze(-1)
        base_exit_logit = exit_logit
        switch_advantage_pred = self.switch_adv_head(advantage_embedding).squeeze(-1)
        exit_prob = torch.sigmoid(exit_logit).clamp(1e-6, 1.0 - 1e-6)
        value = self.value_mlp(value_feat)
        return {
            "base_exit_logit": base_exit_logit,
            "exit_logit": exit_logit,
            "exit_prob": exit_prob,
            "policy_logit": exit_logit,
            "pi_switch": exit_prob,
            "p_adv": exit_prob,
            "tau": torch.full_like(exit_prob, 0.5),
            "hold_return_pred": self.return_head(risk_embedding).squeeze(-1),
            "hold_risk_pred": self.risk_head(risk_embedding).squeeze(-1),
            "switch_advantage_pred": switch_advantage_pred,
            "risk_embedding": risk_embedding,
            "advantage_embedding": advantage_embedding,
            "value": value,
            "value_feat": value_feat,
        }

    def encode(self, weights_drift, port_state,
               switch_action=None, asset_state=None, hold_exec_weights=None,
               switch_exec_weights=None, remaining_horizon=None):
        return self.decision_stats(
            weights_drift, port_state,
            switch_action=switch_action, asset_state=asset_state,
            hold_exec_weights=hold_exec_weights,
            switch_exec_weights=switch_exec_weights,
            remaining_horizon=remaining_horizon,
        )["value_feat"]

    def pi(self, weights_drift, port_state, switch_action=None,
           deterministic=False, asset_state=None, hold_exec_weights=None,
           switch_exec_weights=None, remaining_horizon=None):
        stats = self.decision_stats(
            weights_drift, port_state,
            switch_action=switch_action, asset_state=asset_state,
            hold_exec_weights=hold_exec_weights,
            switch_exec_weights=switch_exec_weights,
            remaining_horizon=remaining_horizon,
        )
        zeros = torch.zeros_like(stats["policy_logit"])
        logits = torch.stack([zeros, stats["policy_logit"]], dim=-1)
        dist = Categorical(logits=logits)
        action = (stats["exit_prob"] > self.eval_switch_threshold).long() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy, logits

    def value(self, weights_drift, port_state,
              switch_action=None, asset_state=None, hold_exec_weights=None,
              switch_exec_weights=None, remaining_horizon=None):
        return self.decision_stats(
            weights_drift, port_state,
            switch_action=switch_action, asset_state=asset_state,
            hold_exec_weights=hold_exec_weights,
            switch_exec_weights=switch_exec_weights,
            remaining_horizon=remaining_horizon,
        )["value"]

    def forward(self, weights_drift, port_state, switch_action=None,
                deterministic=False, asset_state=None, hold_exec_weights=None,
                switch_exec_weights=None, remaining_horizon=None):
        stats = self.decision_stats(
            weights_drift, port_state,
            switch_action=switch_action, asset_state=asset_state,
            hold_exec_weights=hold_exec_weights,
            switch_exec_weights=switch_exec_weights,
            remaining_horizon=remaining_horizon,
        )
        zeros = torch.zeros_like(stats["policy_logit"])
        logits = torch.stack([zeros, stats["policy_logit"]], dim=-1)
        dist = Categorical(logits=logits)
        action = (stats["exit_prob"] > self.eval_switch_threshold).long() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        v = stats["value"]
        return action, log_prob, entropy, logits, v


class FullModel(nn.Module):
    def __init__(self,
                 monitor_args,
                 outer_actor_args,
                 outer_critic_args,
                 inner_actor_args,
                 inner_critic_args):
        super().__init__()

        # 从旧参数里取必要字段，兼容你已有构造入参风格
        # Outer: 复用 outer_actor_args 为主，同时补齐 num_nodes（来自 outer_critic_args）
        outer_args = dict(outer_actor_args)
        outer_args["num_nodes"] = outer_critic_args["num_nodes"]
        self.outer = OuterAC(**outer_args)

        # Inner: 只需要 in_features/hidden_dim/max_boundary（从 inner_actor_args 取）
        inner_args = {
            "in_features": inner_actor_args["in_features"],
            "hidden_dim": inner_actor_args["hidden_dim"],
            "max_boundary": inner_actor_args["max_boundary"],
            "dropout": inner_actor_args.get("dropout", 0.2),
        }
        self.inner = InnerAC(**inner_args)

        # Controller/Monitor: 直接复用 monitor_args
        self.mon = MonitorAC(**monitor_args)
