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
    - 使用轻量因果 TCN（两层 dilation=1,2）编码每个资产的短窗 window=5。
    - 去掉跨资产 attention（该部分在股票数较多时非常耗时）。

    输入：inner_input: [B, N, T(=5), F]
    """

    def __init__(self, in_features, hidden_dim, max_boundary, dropout=0.2,
                 tcn_kernel_size: int = 3):
        super().__init__()
        self.max_boundary = max_boundary
        self.hidden_dim = hidden_dim

        # ---- Causal TCN encoder (very small) ----
        # 两层因果卷积：dilation=1,2 覆盖 window=5 的有效感受野
        self.tcn1 = CausalConv1dBlock(
            in_channels=in_features,
            out_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
            dilation=1,
        )
        self.tcn2 = CausalConv1dBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
            dilation=2,
        )

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

    def _encode_tcn(self, inner_input: torch.Tensor) -> torch.Tensor:
        """TCN 编码：将每资产窗口序列编码为 [B, N, H]（取最后一个时间步）。"""
        B, N, T, M = inner_input.shape
        # [B, N, T, M] -> [B*N, M, T]
        x = inner_input.reshape(B * N, T, M).permute(0, 2, 1)
        x = self.tcn1(x)
        x = self.tcn2(x)  # [B*N, H, T]
        feat = x[:, :, -1].reshape(B, N, self.hidden_dim)
        return feat

    def encode(self, inner_input, base_used_t, weight_drift):
        # 1) per-asset short-window encoding
        node_feat = self._encode_tcn(inner_input)                 # [B, N, H]

        # 2) fuse (no cross-asset attention)
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
    Controller for hold/switch decisions.

    It compares the current live portfolio against the latest outer candidate
    using stock-level z embeddings, weighted p signals, turnover, and the
    holding-time gate from port_state.
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
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_hold = int(min_hold)
        self.max_hold = int(max_hold)

        self.asset_projection = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        controller_dim = hidden_dim * 4 + 6
        self.controller_mlp = nn.Sequential(
            nn.Linear(controller_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.actor_head = nn.Linear(hidden_dim, 2)
        self.v_head = nn.Linear(hidden_dim, 1)

    def encode(self, z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=None):
        asset_emb = self.asset_projection(z)
        weights = weights_drift / (weights_drift.sum(dim=1, keepdim=True) + 1e-8)
        if switch_action is None:
            candidate_weights = weights
        else:
            candidate_weights = switch_action / (switch_action.sum(dim=1, keepdim=True) + 1e-8)
        weighted_emb = torch.sum(asset_emb * weights.unsqueeze(-1), dim=1)
        candidate_emb = torch.sum(asset_emb * candidate_weights.unsqueeze(-1), dim=1)
        mean_emb = torch.mean(asset_emb, dim=1)
        diff_emb = candidate_emb - weighted_emb

        hold_ratio = port_state[:, :1].clamp(0.0, 10.0)
        max_hold = max(float(self.max_hold), 1.0)
        free_den = max(float(self.max_hold - self.min_hold), 1.0)
        hold_days = hold_ratio * max_hold
        free_ratio = torch.clamp((hold_days - float(self.min_hold)) / free_den, 0.0, 1.0)
        turnover = torch.sum(torch.abs(candidate_weights - weights), dim=1, keepdim=True)
        p_cur = torch.sum(weights * p, dim=1, keepdim=True)
        p_cand = torch.sum(candidate_weights * p, dim=1, keepdim=True)
        p_delta = p_cand - p_cur

        controller_state = torch.cat(
            [
                weighted_emb,
                candidate_emb,
                diff_emb,
                mean_emb,
                hold_ratio,
                free_ratio,
                turnover,
                p_cur,
                p_cand,
                p_delta,
            ],
            dim=-1,
        )
        return self.controller_mlp(controller_state)

    def pi(self, z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=None,
           deterministic=False):
        feat = self.encode(z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=switch_action)
        logits = self.actor_head(feat)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy, logits

    def value(self, z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=None):
        feat = self.encode(z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=switch_action)
        return self.v_head(feat)

    def forward(self, z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=None,
                deterministic=False):
        feat = self.encode(
            z, h, p, q_bear, q_bull, weights_drift, port_state, switch_action=switch_action
        )
        logits = self.actor_head(feat)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        v = self.v_head(feat)
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
