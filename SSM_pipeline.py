import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 依赖 SSM3
import SSM3
from create_soft_regime_label import build_ensemble_soft_label, build_fullseries_segment_labels_k2, build_smooth_ensemble_label, build_fullseries_prob_labels
try:
    import utils.config as config
except Exception:
    config = None


def _select_market_config(market: str):
    market = str(market or "nas").lower()
    if market in {"nas", "nas100", "us"}:
        import utils.config_Nas as selected_config
    elif market in {"sh", "a", "ashare", "a-share"}:
        import utils.config_SH as selected_config
    else:
        raise ValueError(f"unsupported market: {market}")
    return selected_config


class _PrintLogger:
    def info(self, msg):
        print(msg)

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
    if COL_LABEL_MAIN not in df.columns:
        print(f"[Label Gen] {COL_LABEL_MAIN} missing for {os.path.basename(csv_path)}, generating...")
        soft_label, _ = build_smooth_ensemble_label(
            df,
            price_col="adjclose",
            spans=[i for i in range(10, 30, 5)],
            slope_tols=[0.0002, 0.0005, 0.001],
            min_lens=[5, 10, 15, 21],
            sensitivity=3,
            final_smooth_sigma=3.0
        )
        df[COL_LABEL_MAIN] = soft_label
        is_dirty = True

    if COL_LABEL_AUX not in df.columns:
        print(f"[Label Gen] {COL_LABEL_AUX} missing for {os.path.basename(csv_path)}, generating...")
        df[COL_LABEL_AUX] = build_fullseries_prob_labels(
            df,
            price_col="adjclose",
            smooth_span=3,
            slope_tol=0.0002,
            min_len=3,
            sensitivity=1.5
        )
        is_dirty = True

    # 3. 如果有更新，保存回 CSV
    if is_dirty:
        print(f"[Label Gen] Saving updates to {csv_path} ...")
        df.to_csv(csv_path)

    return df

# 画图：adjclose + 背景 label
def plot_adjclose_with_labels(dates, adjclose, labels, outpath, title="SSM3 trend"):
    dates = pd.to_datetime(dates)
    adjclose = np.asarray(adjclose).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    if len(labels) == 0: return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, adjclose, lw=1.5, color="black", label="adjclose(t)")
    colors = {0: (0.9, 0.1, 0.1, 0.15), 1: (0.1, 0.7, 0.1, 0.15)}
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if cur in colors:
                ax.axvspan(dates[start], dates[i - 1], color=colors[cur], linewidth=0)
            start = i
            cur = labels[i]
    if cur in colors:
        ax.axvspan(dates[start], dates[-1], color=colors[cur], linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("adjclose")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


@torch.no_grad()
def infer_states_ssm3(model, X_np, y_macro_np, asset_id, device):
    if len(X_np) == 0:
        return dict(p=np.zeros((0,)), q=np.zeros((0,)), pred=np.zeros((0,)), true=np.zeros((0,)),
                    z=np.zeros((0, model.h_dim)), h=np.zeros((0, model.h_dim)))

    X = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)
    y_np = np.asarray(y_macro_np)
    a = torch.tensor([asset_id], dtype=torch.long, device=device)

    B, T, W, D = X.shape
    X_flat = X.view(B * T, W, D)
    z_flat = model.encoder(X_flat)
    z_seq = z_flat.view(B, T, -1)

    e_a = model.asset_emb(a)
    e_seq = e_a.unsqueeze(1).expand(-1, T, -1)

    emit_in = torch.cat([z_seq, e_seq], dim=-1)
    emit_logits = model.emit_head(emit_in).squeeze(-1) / model.tau_emit
    beta = model.beta(a)

    # Output containers
    q_bear_out = torch.zeros(B, T, device=device)
    q_bull_out = torch.zeros(B, T, device=device)
    p_out = torch.zeros(B, T, device=device)

    h_dim = model.h_dim
    h_prev = torch.zeros(B, h_dim, device=device, dtype=X.dtype)
    p_prev = torch.sigmoid(emit_logits[:, 0])
    h_seq = torch.zeros(B, T, h_dim, device=device, dtype=X.dtype)

    for t in range(T):
        z_t = z_seq[:, t, :]  # (B,h_dim)

        # 1. Hazard / Transition (Now using trans_head for dual prob)
        trans_in = torch.cat([h_prev, z_t, e_a], dim=-1)
        qs = model.trans_head(trans_in)
        q_bear_t = qs[:, 0]
        q_bull_t = qs[:, 1]

        q_bear_out[:, t] = q_bear_t
        q_bull_out[:, t] = q_bull_t

        # 2. Prior
        p_prior = p_prev * (1.0 - q_bear_t) + (1.0 - p_prev) * q_bull_t

        # 3. Posterior / Fusion
        fused_logits_t = model._safe_logit(p_prior) + beta.squeeze() * emit_logits[:, t]
        p_t = torch.sigmoid(fused_logits_t)
        p_out[:, t] = p_t

        # 4. Recurrence with Gradient Blocking
        inp_t = torch.cat([z_t, e_a], dim=-1)
        h_prev = model.gru_cell(inp_t, h_prev)

        h_seq[:, t, :] = h_prev
        p_prev = p_t

    p_np = p_out.squeeze(0).cpu().numpy()
    q_bear_np = q_bear_out.squeeze(0).cpu().numpy()
    q_bull_np = q_bull_out.squeeze(0).cpu().numpy()

    z_np = z_seq.squeeze(0).cpu().numpy()
    h_np = h_seq.squeeze(0).cpu().numpy()

    true = (y_np > 0.5).astype(np.int64)
    pred = (p_np > 0.5).astype(np.int64)

    return dict(p=p_np, q_bear=q_bear_np, q_bull=q_bull_np,
                pred=pred, true=true, z=z_np, h=h_np)


def build_inertial_label(probs, high=0.7, low=0.3, confirm_days=0):
    """
    基于迟滞（Hysteresis）和惯性逻辑生成稳健的趋势标签。

    参数:
        probs (np.array): 模型输出的概率序列 p_t (0~1)
        high (float): 上确界，高于此值确认为上涨 (State 1)
        low (float): 下确界，低于此值确认为下跌 (State 0)
        confirm_days (int): 防抖动参数。必须连续 N 天满足条件才切换状态。
                            设为 0 表示只要突破阈值立刻切换（仅保留空间迟滞）。

    返回:
        np.array: 生成的惯性标签序列 (0或1)
    """
    n = len(probs)
    labels = np.zeros(n, dtype=int)

    # 初始化：第0天如果处于模糊区，简单四舍五入，或者默认设为0
    # 这里采用更稳健的做法：先按简单阈值初始化第一个点
    current_state = 1 if probs[0] > 0.5 else 0
    labels[0] = current_state

    # 计数器（用于时间确认）
    consecutive_high = 0
    consecutive_low = 0

    for t in range(1, n):
        p = probs[t]

        # 1. 计数逻辑
        if p > high:
            consecutive_high += 1
            consecutive_low = 0
        elif p < low:
            consecutive_low += 1
            consecutive_high = 0
        else:
            # 进入中间模糊区 (low <= p <= high)
            # 计数器归零 (激进模式) 或者 保持 (保守模式)
            # 这里建议归零，意味着必须是"连续且强烈"的突破才算数
            consecutive_high = 0
            consecutive_low = 0

        # 2. 状态切换逻辑
        if current_state == 0:
            # 如果当前是下跌，必须强势突破高点，且维持一定天数
            if p > high and consecutive_high >= confirm_days:
                current_state = 1
        elif current_state == 1:
            # 如果当前是上涨，必须强势跌破低点，且维持一定天数
            if p < low and consecutive_low >= confirm_days:
                current_state = 0

        # 3. 惯性维持：如果没触发切换，current_state 保持不变
        labels[t] = current_state

    return labels


class MacroMicroWindowDataset(torch.utils.data.Dataset):
    def __init__(self, assets, split):
        self.assets = assets
        self.samples = []
        for ai, asset in enumerate(assets):
            n = len(asset[f"y_macro_{split}"])
            for wi in range(n):
                self.samples.append((ai, wi))
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ai, wi = self.samples[index]
        asset = self.assets[ai]
        split = self.split
        y_macro_cls = asset.get(
            f"y_macro_cls_{split}",
            (asset[f"y_macro_{split}"] > 0.5).astype(np.float32),
        )
        if wi > 0:
            x_prev = asset[f"X_{split}"][wi - 1]
            prev_same_state = float(y_macro_cls[wi] > 0.5) == float(y_macro_cls[wi - 1] > 0.5)
            persist_weight = 1.0 if prev_same_state else 0.0
        else:
            x_prev = np.zeros_like(asset[f"X_{split}"][wi], dtype=np.float32)
            persist_weight = 0.0
        return (
            torch.tensor(asset[f"X_{split}"][wi], dtype=torch.float32),
            torch.tensor(asset[f"y_macro_{split}"][wi], dtype=torch.float32),
            torch.tensor(y_macro_cls[wi], dtype=torch.float32),
            torch.tensor(asset[f"y_micro_{split}"][wi], dtype=torch.float32),
            torch.tensor(x_prev, dtype=torch.float32),
            torch.tensor(persist_weight, dtype=torch.float32),
            torch.tensor(asset["asset_id"], dtype=torch.long),
        )


class MacroMicroAttentionLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim=16,
        lstm_hidden_dim=32,
        lstm_layers=1,
        dropout=0.1,
        gate_strength=0.5,
        gate_hidden_dim=0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.gate_strength = float(gate_strength)
        gate_hidden_dim = int(gate_hidden_dim) if int(gate_hidden_dim) > 0 else max(self.input_dim * 2, 16)
        self.feature_gate = nn.Sequential(
            nn.Linear(self.input_dim * 4, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_hidden_dim, self.input_dim),
        )
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)
        lstm_dropout = float(dropout) if self.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        attn_mid = max(self.lstm_hidden_dim // 2, 4)
        self.attn_pool = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, attn_mid),
            nn.Tanh(),
            nn.Linear(attn_mid, 1),
        )
        self.pool_mix = nn.Parameter(torch.tensor(0.5))
        self.emb_proj = nn.Linear(self.lstm_hidden_dim, self.emb_dim)
        self.emb_ln = nn.LayerNorm(self.emb_dim)
        self.state_prototypes = nn.Parameter(torch.randn(2, self.emb_dim) * 0.02)
        self.macro_head = nn.Linear(self.emb_dim, 1)
        self.macro_cls_head = nn.Linear(self.emb_dim, 1)
        self.micro_head = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        last_x = x[:, -1, :]
        gate_stats = torch.cat(
            [
                last_x,
                x.mean(dim=1),
                x.std(dim=1, unbiased=False),
                last_x - x[:, 0, :],
            ],
            dim=-1,
        )
        feature_gate = torch.sigmoid(self.feature_gate(gate_stats))
        gate_scale = 1.0 + self.gate_strength * (2.0 * feature_gate - 1.0)
        h_time, _ = self.lstm(x * gate_scale.unsqueeze(1))
        last = h_time[:, -1, :]
        attn = torch.softmax(self.attn_pool(h_time).squeeze(-1), dim=-1)
        pooled = torch.sum(h_time * attn.unsqueeze(-1), dim=1)
        mix = torch.sigmoid(self.pool_mix)
        embedding = self.emb_ln(self.emb_proj(mix * last + (1.0 - mix) * pooled))
        macro_logits = self.macro_head(embedding).squeeze(-1)
        macro_cls_logits = self.macro_cls_head(embedding).squeeze(-1)
        micro_logits = self.micro_head(embedding).squeeze(-1)
        return {
            "embedding": embedding,
            "macro_logits": macro_logits,
            "macro_cls_logits": macro_cls_logits,
            "micro_logits": micro_logits,
            "macro_prob": torch.sigmoid(macro_logits),
            "macro_cls_prob": torch.sigmoid(macro_cls_logits),
            "micro_prob": torch.sigmoid(micro_logits),
            "feature_gate": feature_gate,
        }


def compute_macro_micro_loss(
    outputs,
    y_macro,
    y_macro_cls,
    y_micro,
    lambda_macro=1.0,
    lambda_macro_cls=0.5,
    lambda_micro=1.0,
):
    macro_loss = F.binary_cross_entropy_with_logits(outputs["macro_logits"], y_macro)
    macro_cls_loss = F.binary_cross_entropy_with_logits(outputs["macro_cls_logits"], y_macro_cls)
    if float(lambda_micro) > 0.0 and y_micro is not None:
        micro_loss = F.binary_cross_entropy_with_logits(outputs["micro_logits"], y_micro)
    else:
        micro_loss = outputs["macro_logits"].new_tensor(0.0)
    total = (
        float(lambda_macro) * macro_loss
        + float(lambda_macro_cls) * macro_cls_loss
        + float(lambda_micro) * micro_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "macro_bce": float(macro_loss.detach().cpu()),
        "macro_cls_bce": float(macro_cls_loss.detach().cpu()),
        "micro_bce": float(micro_loss.detach().cpu()),
    }


def compute_state_center_loss(embedding, y_macro, y_macro_cls, prototypes, margin=1.0):
    z = F.normalize(embedding, dim=-1)
    proto = F.normalize(prototypes, dim=-1)
    target = y_macro_cls.long().clamp(0, 1)
    target_proto = proto[target]
    confidence = torch.clamp(torch.abs(y_macro - 0.5) * 2.0, min=0.0, max=1.0)
    center = 1.0 - torch.sum(z * target_proto, dim=-1)
    center = torch.sum(center * confidence) / torch.clamp(confidence.sum(), min=1.0)
    sep = torch.relu(float(margin) - torch.norm(proto[0] - proto[1], p=2)).pow(2)
    return center + sep


def compute_embedding_persist_loss(embedding, prev_embedding, persist_weight):
    if persist_weight is None:
        return embedding.new_tensor(0.0)
    weight = persist_weight.float()
    if torch.sum(weight) <= 0:
        return embedding.new_tensor(0.0)
    dist = F.smooth_l1_loss(embedding, prev_embedding, reduction="none").mean(dim=-1)
    return torch.sum(dist * weight) / torch.clamp(weight.sum(), min=1.0)


def _make_macro_micro_windows(values, window):
    values = np.asarray(values, dtype=np.float32)
    n_time = values.shape[0]
    if n_time < int(window):
        return np.zeros((0, int(window), values.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    endpoints = np.arange(int(window) - 1, n_time, dtype=np.int64)
    out = np.empty((len(endpoints), int(window), values.shape[1]), dtype=np.float32)
    for i, end in enumerate(endpoints):
        out[i] = values[end - int(window) + 1 : end + 1]
    return out, endpoints


def _clean_soft_label(values):
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _macro_hard_label_from_soft(values, high=0.6, low=0.4):
    if not float(low) < float(high):
        raise ValueError(f"macro hard-label low must be < high, got low={low}, high={high}")
    y = _clean_soft_label(values)
    if len(y) == 0:
        return np.zeros((0,), dtype=np.float32)
    return build_inertial_label(y, high=float(high), low=float(low), confirm_days=0).astype(np.float32)


def _delta_proxy_q(macro_prob):
    p = np.asarray(macro_prob, dtype=np.float32).reshape(-1)
    if len(p) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    delta = np.diff(p, prepend=p[0]).astype(np.float32)
    q_bear = np.clip(-delta, 0.0, 1.0).astype(np.float32)
    q_bull = np.clip(delta, 0.0, 1.0).astype(np.float32)
    return q_bear, q_bull


def _binary_metrics(prob, y):
    prob = np.asarray(prob, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {"bce": float("nan"), "acc": float("nan")}
    p = np.clip(prob, 1e-8, 1.0 - 1e-8)
    bce = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    return {"bce": bce, "acc": acc}


@torch.no_grad()
def infer_macro_micro(model, X_np, device, batch_size=512):
    model.eval()
    embeddings, macro_probs, macro_cls_probs, micro_probs = [], [], [], []
    for start in range(0, len(X_np), int(batch_size)):
        xb = torch.tensor(X_np[start : start + int(batch_size)], dtype=torch.float32, device=device)
        out = model(xb)
        embeddings.append(out["embedding"].cpu())
        macro_probs.append(out["macro_prob"].cpu())
        macro_cls_probs.append(out["macro_cls_prob"].cpu())
        micro_probs.append(out["micro_prob"].cpu())
    emb = torch.cat(embeddings, dim=0) if embeddings else torch.zeros((0, model.emb_dim), dtype=torch.float32)
    macro = torch.cat(macro_probs, dim=0).numpy() if macro_probs else np.zeros((0,), dtype=np.float32)
    macro_cls = torch.cat(macro_cls_probs, dim=0).numpy() if macro_cls_probs else np.zeros((0,), dtype=np.float32)
    micro = torch.cat(micro_probs, dim=0).numpy() if micro_probs else np.zeros((0,), dtype=np.float32)
    return emb.numpy(), macro, macro_cls, micro


def evaluate_macro_micro_split(model, assets, split, device, batch_size=512, use_micro_label=True):
    macro_probs, macro_cls_probs, micro_probs = [], [], []
    macro_true, macro_cls_true, micro_true = [], [], []
    pred_switches, true_switches, turning_correct, turning_total = 0, 0, 0, 0
    for asset in assets:
        X = asset[f"X_{split}"]
        y_macro = asset[f"y_macro_{split}"]
        y_macro_cls = asset.get(f"y_macro_cls_{split}", (y_macro > 0.5).astype(np.float32))
        y_micro = asset[f"y_micro_{split}"]
        if len(y_macro) == 0:
            continue
        _, macro_prob, macro_cls_prob, micro_prob = infer_macro_micro(model, X, device, batch_size=batch_size)
        macro_probs.append(macro_prob)
        macro_cls_probs.append(macro_cls_prob)
        if use_micro_label:
            micro_probs.append(micro_prob)
        macro_true.append(y_macro)
        macro_cls_true.append(y_macro_cls)
        if use_micro_label:
            micro_true.append(y_micro)

        pred = (macro_cls_prob > 0.5).astype(np.int64)
        true = (y_macro_cls > 0.5).astype(np.int64)
        if len(pred) > 1:
            pred_switches += int(np.sum(pred[1:] != pred[:-1]))
            true_switches += int(np.sum(true[1:] != true[:-1]))
            switch_idx = np.where(true[1:] != true[:-1])[0] + 1
            turning_total += int(len(switch_idx))
            if len(switch_idx):
                turning_correct += int(np.sum(pred[switch_idx] == true[switch_idx]))

    if not macro_true:
        return {
            "num_samples": 0,
            "macro_bce": float("nan"),
            "macro_cls_bce": float("nan"),
            "micro_bce": float("nan"),
            "macro_acc": float("nan"),
            "macro_cls_acc": float("nan"),
            "micro_acc": float("nan"),
            "pred_switches": 0,
            "true_switches": 0,
            "turning_acc": float("nan"),
        }
    macro_prob = np.concatenate(macro_probs)
    macro_cls_prob = np.concatenate(macro_cls_probs)
    micro_prob = np.concatenate(micro_probs) if use_micro_label else np.zeros((0,), dtype=np.float32)
    y_macro = np.concatenate(macro_true)
    y_macro_cls = np.concatenate(macro_cls_true)
    y_micro = np.concatenate(micro_true) if use_micro_label else np.zeros((0,), dtype=np.float32)
    macro = _binary_metrics(macro_prob, y_macro)
    macro_cls = _binary_metrics(macro_cls_prob, y_macro_cls)
    micro = _binary_metrics(micro_prob, y_micro) if use_micro_label else {"bce": float("nan"), "acc": float("nan")}
    return {
        "num_samples": int(len(y_macro)),
        "macro_bce": macro["bce"],
        "macro_cls_bce": macro_cls["bce"],
        "micro_bce": micro["bce"],
        "macro_acc": macro["acc"],
        "macro_cls_acc": macro_cls["acc"],
        "micro_acc": micro["acc"],
        "pred_switches": int(pred_switches),
        "true_switches": int(true_switches),
        "turning_acc": float(turning_correct / turning_total) if turning_total else float("nan"),
    }


def build_macro_micro_assets(args, logger):
    from risk_tpsm_lite import build_risk_tpsm_features

    feature_preset = args.feature_preset
    if feature_preset == "hybrid_lite" and int(args.target_feature_count) == 25:
        feature_preset = "risk_only"
        logger.info(
            "[MacroMicro] using risk_only features for stable 25-column input across all assets."
        )

    stock_file = config.dataset["stocks_path"]
    feature_dir = config.dataset["feature_path"]
    with open(stock_file, "r") as f:
        codes = [line.strip() for line in f if line.strip()]
    if int(getattr(args, "max_stocks", 0) or 0) > 0:
        codes = codes[: int(args.max_stocks)]

    assets = []
    feature_names = None
    train_start, train_end = pd.Timestamp(config.train_start_date), pd.Timestamp(config.train_end_date)
    val_start, val_end = pd.Timestamp(config.valid_start_date), pd.Timestamp(config.valid_end_date)
    test_start, test_end = pd.Timestamp(config.test_start_date), pd.Timestamp(config.test_end_date)

    for asset_id, code in enumerate(codes):
        csv_path = os.path.join(feature_dir, f"{code}.csv")
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
        df = ensure_labels_exist(df, csv_path)
        if int(getattr(args, "max_rows", 0) or 0) > 0:
            df = df.iloc[: int(args.max_rows)].copy()
        try:
            from utils.PriceMatrix import _standardize_feature_columns

            df = _standardize_feature_columns(
                df,
                ["adjopen", "adjhigh", "adjlow", "adjclose", "amount", "amp", "body"],
            )
        except KeyError:
            pass
        feat_df = build_risk_tpsm_features(
            df,
            window=args.window,
            normalization_lookback=args.normalization_lookback,
            normalization=args.normalization,
            clip=args.feature_clip,
            feature_preset=feature_preset,
            original_feature_names=args.original_features,
            original_feature_limit=args.original_feature_limit,
            selected_feature_names=getattr(args, "feature_names", None),
            target_feature_count=args.target_feature_count,
        )
        if feature_names is None:
            feature_names = list(feat_df.columns)
        elif feature_names != list(feat_df.columns):
            raise ValueError(f"Feature mismatch for {code}.")

        X, endpoints = _make_macro_micro_windows(feat_df.values, args.window)
        if len(endpoints) == 0:
            logger.info(f"[MacroMicro] skip {code}: not enough rows for window={args.window}")
            continue

        dates = pd.to_datetime(df.index)
        idx_all = dates[endpoints]
        y_macro_full = _clean_soft_label(df[COL_LABEL_MAIN].values)
        y_macro_cls_full = _macro_hard_label_from_soft(
            y_macro_full,
            high=getattr(args, "macro_cls_high", 0.6),
            low=getattr(args, "macro_cls_low", 0.4),
        )
        y_micro_full = _clean_soft_label(df[COL_LABEL_AUX].values)
        y_macro = y_macro_full[endpoints]
        y_macro_cls = y_macro_cls_full[endpoints]
        y_micro = y_micro_full[endpoints]
        close = df["adjclose"].to_numpy(dtype=np.float32)[endpoints]

        masks = {
            "train": (idx_all >= train_start) & (idx_all <= train_end),
            "val": (idx_all >= val_start) & (idx_all <= val_end),
            "test": (idx_all >= test_start) & (idx_all <= test_end),
        }
        asset = {
            "asset_id": asset_id,
            "code": code,
            "csv_path": csv_path,
            "df": df,
            "X_all": X,
            "y_macro_all": y_macro,
            "y_macro_cls_all": y_macro_cls,
            "y_micro_all": y_micro,
            "idx_all": idx_all,
            "close_all": close,
        }
        for split, mask in masks.items():
            asset[f"X_{split}"] = X[mask]
            asset[f"y_macro_{split}"] = y_macro[mask]
            asset[f"y_macro_cls_{split}"] = y_macro_cls[mask]
            asset[f"y_micro_{split}"] = y_micro[mask]
            asset[f"idx_{split}"] = idx_all[mask]
            asset[f"close_{split}"] = close[mask]
        assets.append(asset)

    if not assets:
        raise ValueError("No assets available for Macro/Micro Lite training.")
    logger.info(
        f"[MacroMicro] assets={len(assets)} features={len(feature_names)} "
        f"window={args.window} normalization={args.normalization} "
        f"macro_cls=({getattr(args, 'macro_cls_low', 0.4):.2f},{getattr(args, 'macro_cls_high', 0.6):.2f})"
    )
    return assets, feature_names or []


def train_macro_micro_one_epoch(model, loader, optimizer, device, args):
    model.train()
    logs = []
    use_micro_label = not getattr(args, "disable_micro_label", False) and float(args.lambda_micro) > 0.0
    for X, y_macro, y_macro_cls, y_micro, X_prev, persist_weight, _ in loader:
        X = X.to(device)
        y_macro = y_macro.to(device)
        y_macro_cls = y_macro_cls.to(device)
        y_micro = y_micro.to(device) if use_micro_label else None
        X_prev = X_prev.to(device)
        persist_weight = persist_weight.to(device)
        out = model(X)
        loss, parts = compute_macro_micro_loss(
            out,
            y_macro,
            y_macro_cls,
            y_micro,
            lambda_macro=args.lambda_macro,
            lambda_macro_cls=args.lambda_macro_cls,
            lambda_micro=args.lambda_micro,
        )
        if float(getattr(args, "lambda_state_center", 0.0)) > 0.0:
            state_center = compute_state_center_loss(
                out["embedding"],
                y_macro,
                y_macro_cls,
                model.state_prototypes,
                margin=getattr(args, "state_center_margin", 1.0),
            )
            loss = loss + float(args.lambda_state_center) * state_center
            parts["state_center"] = float(state_center.detach().cpu())
            parts["loss"] = float(loss.detach().cpu())
        else:
            parts["state_center"] = 0.0

        if float(getattr(args, "lambda_embedding_persist", 0.0)) > 0.0:
            with torch.no_grad():
                prev_embedding = model(X_prev)["embedding"]
            persist = compute_embedding_persist_loss(out["embedding"], prev_embedding, persist_weight)
            loss = loss + float(args.lambda_embedding_persist) * persist
            parts["embedding_persist"] = float(persist.detach().cpu())
            parts["loss"] = float(loss.detach().cpu())
        else:
            parts["embedding_persist"] = 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logs.append(parts)
    return {k: float(np.mean([x[k] for x in logs])) for k in logs[0].keys()} if logs else {}


def export_macro_micro_outputs(model, assets, args, out_dir, checkpoint_path, feature_names, device, logger):
    os.makedirs(out_dir, exist_ok=True)
    use_micro_label = not getattr(args, "disable_micro_label", False) and float(getattr(args, "lambda_micro", 1.0)) > 0.0
    feature_preset = getattr(args, "feature_preset", "risk_only")
    export_feature_preset = (
        "risk_only"
        if feature_preset == "hybrid_lite" and int(args.target_feature_count) == 25
        else feature_preset
    )
    manifest = {
        "model": "macro_micro_lite",
        "checkpoint": checkpoint_path,
        "feature_names": list(feature_names),
        "window": int(args.window),
        "normalization": args.normalization,
        "feature_preset": export_feature_preset,
        "target_feature_count": int(args.target_feature_count),
        "lambda_macro": float(args.lambda_macro),
        "lambda_macro_cls": float(getattr(args, "lambda_macro_cls", 0.5)),
        "lambda_micro": float(args.lambda_micro),
        "disable_micro_label": bool(getattr(args, "disable_micro_label", False)),
        "lambda_state_center": float(getattr(args, "lambda_state_center", 0.0)),
        "lambda_embedding_persist": float(getattr(args, "lambda_embedding_persist", 0.0)),
        "state_center_margin": float(getattr(args, "state_center_margin", 1.0)),
        "macro_cls_high": float(getattr(args, "macro_cls_high", 0.6)),
        "macro_cls_low": float(getattr(args, "macro_cls_low", 0.4)),
        "gate_strength": float(getattr(args, "gate_strength", 0.5)),
        "gate_hidden_dim": int(getattr(args, "gate_hidden_dim", 0)),
        "export_semantics": "ssm3_p is macro label probability; q_bear/q_bull are delta proxy columns.",
    }
    with open(os.path.join(out_dir, "macro_micro_lite_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    for asset in assets:
        code = asset["code"]
        emb, macro_prob, macro_cls_prob, micro_prob = infer_macro_micro(
            model,
            asset["X_all"],
            device,
            batch_size=args.infer_batch_size,
        )
        if not use_micro_label:
            micro_prob = np.full_like(macro_prob, np.nan, dtype=np.float32)
        q_bear, q_bull = _delta_proxy_q(macro_prob)
        pred_inertial = build_inertial_label(
            macro_prob,
            high=getattr(args, "macro_cls_high", 0.6),
            low=getattr(args, "macro_cls_low", 0.4),
            confirm_days=0,
        )
        pred = pred_inertial.astype(np.int64)
        true = (asset.get("y_macro_cls_all", asset["y_macro_all"] > 0.5) > 0.5).astype(np.int64)

        out_df = asset["df"].copy()
        idx_tgt = asset["idx_all"]
        out_df.loc[idx_tgt, "ssm3_p"] = macro_prob
        out_df.loc[idx_tgt, "ssm3_q_bear"] = q_bear
        out_df.loc[idx_tgt, "ssm3_q_bull"] = q_bull
        out_df.loc[idx_tgt, "ssm3_pred"] = pred
        out_df.loc[idx_tgt, "ssm3_true"] = true
        out_df.loc[idx_tgt, "ssm3_pred_inertial"] = pred_inertial
        out_df.loc[idx_tgt, "ssm3_true_micro"] = asset["y_micro_all"]
        out_df.loc[idx_tgt, "macro_micro_prob"] = macro_prob
        out_df.loc[idx_tgt, "macro_cls_prob"] = macro_cls_prob
        out_df.loc[idx_tgt, "macro_cls_true"] = true
        out_df.loc[idx_tgt, "macro_soft_true"] = asset["y_macro_all"]
        out_df.loc[idx_tgt, "micro_prob"] = micro_prob
        out_df.to_csv(os.path.join(out_dir, f"{code}.csv"), date_format="%Y-%m-%d")

        state = {
            "h": torch.tensor(emb, dtype=torch.float32),
            "z": torch.tensor(emb, dtype=torch.float32),
            "date_idx": np.asarray(idx_tgt),
            "macro_prob": torch.tensor(macro_prob, dtype=torch.float32),
            "macro_cls_prob": torch.tensor(macro_cls_prob, dtype=torch.float32),
            "micro_prob": torch.tensor(micro_prob, dtype=torch.float32),
        }
        torch.save(state, os.path.join(out_dir, f"{code}_ssm3_states.pt"))
        logger.info(f"[MacroMicro Export] {code}: {len(idx_tgt)} rows")


def run_macro_micro_lite_pipeline(args, logger):
    if config is None:
        raise RuntimeError("未找到 utils.config")
    device = torch.device(args.device if getattr(args, "device", "") else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_micro_label = not getattr(args, "disable_micro_label", False) and float(args.lambda_micro) > 0.0
    assets, feature_names = build_macro_micro_assets(args, logger)

    train_ds = MacroMicroWindowDataset(assets, "train")
    val_ds = MacroMicroWindowDataset(assets, "val")
    test_ds = MacroMicroWindowDataset(assets, "test")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty Macro/Micro split: train={len(train_ds)} val={len(val_ds)}")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.macro_micro_batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MacroMicroAttentionLSTM(
        input_dim=len(feature_names),
        emb_dim=args.emb_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        gate_strength=args.gate_strength,
        gate_hidden_dim=args.gate_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir_model = os.path.join(args.kd_path, "MacroMicroLite")
    os.makedirs(out_dir_model, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(out_dir_model, "best_macro_micro_lite.pt")

    if args.mode in ("train", "train_export"):
        best_val = float("inf")
        metrics_rows = []
        for ep in range(1, int(args.macro_micro_epochs) + 1):
            train_log = train_macro_micro_one_epoch(model, train_loader, optimizer, device, args)
            val_metrics = evaluate_macro_micro_split(
                model,
                assets,
                "val",
                device,
                batch_size=args.infer_batch_size,
                use_micro_label=use_micro_label,
            )
            val_loss = (
                float(args.lambda_macro) * val_metrics["macro_bce"]
                + float(args.lambda_macro_cls) * val_metrics["macro_cls_bce"]
            )
            if use_micro_label:
                val_loss += float(args.lambda_micro) * val_metrics["micro_bce"]
            row = {"epoch": ep, **{f"train_{k}": v for k, v in train_log.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            metrics_rows.append(row)
            pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir_model, "macro_micro_lite_metrics.csv"), index=False)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": {
                            "input_dim": len(feature_names),
                            "emb_dim": int(args.emb_dim),
                            "lstm_hidden_dim": int(args.lstm_hidden_dim),
                            "lstm_layers": int(args.lstm_layers),
                            "dropout": float(args.dropout),
                            "gate_strength": float(args.gate_strength),
                            "gate_hidden_dim": int(args.gate_hidden_dim),
                            "window": int(args.window),
                            "feature_names": list(feature_names),
                            "normalization": args.normalization,
                            "feature_preset": "risk_only"
                            if args.feature_preset == "hybrid_lite" and int(args.target_feature_count) == 25
                            else args.feature_preset,
                            "target_feature_count": int(args.target_feature_count),
                            "lambda_macro": float(args.lambda_macro),
                            "lambda_macro_cls": float(args.lambda_macro_cls),
                            "lambda_micro": float(args.lambda_micro),
                            "disable_micro_label": bool(getattr(args, "disable_micro_label", False)),
                            "lambda_state_center": float(args.lambda_state_center),
                            "lambda_embedding_persist": float(args.lambda_embedding_persist),
                            "state_center_margin": float(args.state_center_margin),
                            "macro_cls_high": float(args.macro_cls_high),
                            "macro_cls_low": float(args.macro_cls_low),
                        },
                        "epoch": ep,
                        "valid_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
            logger.info(
                f"[MacroMicro] ep={ep:03d}/{int(args.macro_micro_epochs)} "
                f"train_macro={train_log.get('macro_bce', float('nan')):.5f} "
                f"train_cls={train_log.get('macro_cls_bce', float('nan')):.5f} "
                f"train_micro={train_log.get('micro_bce', float('nan')):.5f} "
                f"val_macro={val_metrics['macro_bce']:.5f} val_cls={val_metrics['macro_cls_bce']:.5f} "
                f"val_micro={val_metrics['micro_bce']:.5f} "
                f"val_acc={val_metrics['macro_cls_acc']:.3f} switches={val_metrics['pred_switches']}/{val_metrics['true_switches']} "
                f"best={best_val:.5f}"
            )
    elif args.mode == "export":
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing Macro/Micro checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"unsupported mode for macro_micro_lite: {args.mode}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    final_metrics = {
        split: evaluate_macro_micro_split(
            model,
            assets,
            split,
            device,
            batch_size=args.infer_batch_size,
            use_micro_label=use_micro_label,
        )
        for split in ("train", "val", "test")
    }
    with open(os.path.join(out_dir_model, "macro_micro_lite_final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"[MacroMicro] final metrics: {json.dumps(final_metrics, ensure_ascii=False)}")

    if args.mode in ("train_export", "export") or getattr(args, "export_after_train", False):
        out_dir = getattr(args, "macro_micro_output_dir", "") or config.dataset["ssm_data_path"]
        export_macro_micro_outputs(model, assets, args, out_dir, checkpoint_path, feature_names, device, logger)
    return checkpoint_path, final_metrics


class MacroTransitionSeqDataset(torch.utils.data.Dataset):
    def __init__(self, assets, split, seq_len=720, stride=63):
        self.assets = assets
        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.samples = []
        for ai, asset in enumerate(assets):
            n = len(asset[f"y_macro_{split}"])
            if n < self.seq_len:
                continue
            s = 0
            while s + self.seq_len <= n:
                self.samples.append((ai, s))
                s += self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ai, s = self.samples[index]
        asset = self.assets[ai]
        split = self.split
        e = s + self.seq_len
        return (
            torch.tensor(asset[f"X_{split}"][s:e], dtype=torch.float32),
            torch.tensor(asset[f"y_macro_{split}"][s:e], dtype=torch.float32),
            torch.tensor(asset["asset_id"], dtype=torch.long),
        )


class TimeWiseGatedAttentionEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=16, lstm_hidden_dim=32, lstm_layers=1, dropout=0.1, gate_strength=0.3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.gate_strength = float(gate_strength)
        gate_hidden = max(self.input_dim * 2, 16)
        self.feature_gate = nn.Sequential(
            nn.Linear(self.input_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_hidden, self.input_dim),
        )
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)
        lstm_dropout = float(dropout) if int(lstm_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=lstm_dropout,
        )
        attn_mid = max(self.lstm_hidden_dim // 2, 4)
        self.attn_pool = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, attn_mid),
            nn.Tanh(),
            nn.Linear(attn_mid, 1),
        )
        self.pool_mix = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(self.lstm_hidden_dim, self.emb_dim)
        self.ln = nn.LayerNorm(self.emb_dim)

    def forward(self, x):
        feature_gate = torch.sigmoid(self.feature_gate(x))
        gate_scale = 1.0 + self.gate_strength * (2.0 * feature_gate - 1.0)
        h_time, _ = self.lstm(x * gate_scale)
        last = h_time[:, -1, :]
        attn = torch.softmax(self.attn_pool(h_time).squeeze(-1), dim=-1)
        pooled = torch.sum(h_time * attn.unsqueeze(-1), dim=1)
        mix = torch.sigmoid(self.pool_mix)
        z = self.ln(self.out_proj(mix * last + (1.0 - mix) * pooled))
        return z, feature_gate


class MacroTransitionGateSSM(nn.Module):
    def __init__(
        self,
        input_dim,
        num_assets,
        emb_dim=16,
        h_dim=32,
        asset_emb_dim=8,
        lstm_layers=1,
        dropout=0.1,
        gate_strength=0.3,
        q_max=0.5,
        beta_init=0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)
        self.h_dim = int(h_dim)
        self.num_assets = int(num_assets)
        self.asset_emb_dim = int(asset_emb_dim)
        self.q_max = float(q_max)
        self.encoder = TimeWiseGatedAttentionEncoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            lstm_hidden_dim=h_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            gate_strength=gate_strength,
        )
        self.asset_emb = nn.Embedding(num_assets, asset_emb_dim)
        self.gru_cell = nn.GRUCell(input_size=emb_dim + asset_emb_dim, hidden_size=h_dim)
        self.emit_head = nn.Sequential(
            nn.Linear(emb_dim + asset_emb_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )
        self.macro_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )
        trans_dim = h_dim + emb_dim + emb_dim + asset_emb_dim
        self.turn_gate_head = nn.Sequential(
            nn.Linear(trans_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )
        self.trans_dir_head = nn.Sequential(
            nn.Linear(trans_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2),
        )
        self.raw_beta = nn.Parameter(torch.tensor(float(beta_init)))

    @staticmethod
    def _safe_logit(p, eps=1e-6):
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def beta(self):
        return F.softplus(self.raw_beta) + 1e-6

    def forward(self, X, asset_id, detach_encoder=False):
        B, T, W, D = X.shape
        X_flat = X.reshape(B * T, W, D)
        z_flat, feature_gate_flat = self.encoder(X_flat)
        if detach_encoder:
            z_flat = z_flat.detach()
        z_seq = z_flat.view(B, T, -1)
        feature_gate = feature_gate_flat.view(B, T, W, D)

        e_a = self.asset_emb(asset_id)
        e_seq = e_a.unsqueeze(1).expand(-1, T, -1)
        emit_logits = self.emit_head(torch.cat([z_seq, e_seq], dim=-1)).squeeze(-1)

        p_list, q_bear_list, q_bull_list = [], [], []
        gate_list, gate_logit_list, fused_logits_list, macro_logits_list, h_list = [], [], [], [], []
        h_prev = torch.zeros(B, self.h_dim, dtype=X.dtype, device=X.device)
        p_prev = torch.sigmoid(emit_logits[:, 0])
        beta = self.beta()

        for t in range(T):
            z_t = z_seq[:, t, :]
            macro_logits_list.append(self.macro_head(h_prev).squeeze(-1))
            trans_in = torch.cat([h_prev, z_t, torch.abs(z_t - h_prev[:, : self.emb_dim]), e_a], dim=-1)
            turn_logit = self.turn_gate_head(trans_in).squeeze(-1)
            turn_gate = torch.sigmoid(turn_logit)
            q_dir = torch.sigmoid(self.trans_dir_head(trans_in))
            q_bear_t = self.q_max * turn_gate * q_dir[:, 0]
            q_bull_t = self.q_max * turn_gate * q_dir[:, 1]
            p_prior = p_prev * (1.0 - q_bear_t) + (1.0 - p_prev) * q_bull_t
            fused_logit_t = self._safe_logit(p_prior) + beta * emit_logits[:, t]
            p_t = torch.sigmoid(fused_logit_t)

            p_list.append(p_t)
            q_bear_list.append(q_bear_t)
            q_bull_list.append(q_bull_t)
            gate_list.append(turn_gate)
            gate_logit_list.append(turn_logit)
            fused_logits_list.append(fused_logit_t)
            inp_t = torch.cat([z_t, e_a], dim=-1)
            h_prev = self.gru_cell(inp_t, h_prev)
            h_list.append(h_prev)
            p_prev = p_t

        return {
            "p": torch.stack(p_list, dim=1),
            "q_bear": torch.stack(q_bear_list, dim=1),
            "q_bull": torch.stack(q_bull_list, dim=1),
            "turn_gate": torch.stack(gate_list, dim=1),
            "turn_gate_logits": torch.stack(gate_logit_list, dim=1),
            "emit_logits": emit_logits,
            "fused_logits": torch.stack(fused_logits_list, dim=1),
            "macro_logits": torch.stack(macro_logits_list, dim=1),
            "z": z_seq,
            "h": torch.stack(h_list, dim=1),
            "feature_gate": feature_gate,
        }


def _macro_turn_targets(y, scale=5.0):
    diff = y[:, 1:] - y[:, :-1]
    turn = torch.clamp(torch.abs(diff) * float(scale), 0.0, 1.0)
    q_bull = torch.clamp(torch.relu(diff) * float(scale), 0.0, 1.0)
    q_bear = torch.clamp(torch.relu(-diff) * float(scale), 0.0, 1.0)
    pad = torch.zeros(y.size(0), 1, dtype=y.dtype, device=y.device)
    return torch.cat([pad, turn], dim=1), torch.cat([pad, q_bear], dim=1), torch.cat([pad, q_bull], dim=1)


def compute_macro_transition_loss(out, y, args, phase="joint"):
    bce_fused = F.binary_cross_entropy_with_logits(out["fused_logits"], y)
    bce_emit = F.binary_cross_entropy_with_logits(out["emit_logits"], y)
    bce_macro = F.binary_cross_entropy_with_logits(out["macro_logits"], y)
    deriv = SSM3.DerivativeMatchingLoss(weight=1.0)(out["p"], y)
    tv = SSM3.tv_loss(out["p"], weight=1.0)
    refractory = SSM3.calc_refractory_loss(out["q_bear"]) + SSM3.calc_refractory_loss(out["q_bull"])
    turn_target, q_bear_target, q_bull_target = _macro_turn_targets(y, scale=args.turn_target_scale)
    gate_turn = F.binary_cross_entropy_with_logits(out["turn_gate_logits"], turn_target)
    q_target = (
        F.binary_cross_entropy(torch.clamp(out["q_bear"] / max(float(args.q_max), 1e-6), 1e-6, 1.0 - 1e-6), q_bear_target)
        + F.binary_cross_entropy(torch.clamp(out["q_bull"] / max(float(args.q_max), 1e-6), 1e-6, 1.0 - 1e-6), q_bull_target)
    )
    gate_sparse = out["turn_gate"].mean()

    if phase == "enc":
        total = bce_emit
    else:
        total = (
            float(args.lambda_fuse) * bce_fused
            + float(args.lambda_emit) * bce_emit
            + float(args.lambda_macro_aux) * bce_macro
            + float(args.lambda_deriv) * deriv
            + float(args.transition_lambda_tv) * tv
            + float(args.lambda_refra) * refractory
            + float(args.lambda_turn_gate) * gate_turn
            + float(args.lambda_q_target) * q_target
            + float(args.lambda_gate_sparse) * gate_sparse
        )
    return total, {
        "loss": float(total.detach().cpu()),
        "fused_bce": float(bce_fused.detach().cpu()),
        "emit_bce": float(bce_emit.detach().cpu()),
        "macro_bce": float(bce_macro.detach().cpu()),
        "deriv": float(deriv.detach().cpu()),
        "tv": float(tv.detach().cpu()),
        "refractory": float(refractory.detach().cpu()),
        "gate_turn": float(gate_turn.detach().cpu()),
        "q_target": float(q_target.detach().cpu()),
        "gate_mean": float(gate_sparse.detach().cpu()),
        "q_mean": float(((out["q_bear"].mean() + out["q_bull"].mean()) * 0.5).detach().cpu()),
    }


def build_macro_transition_optimizers(model, lr_enc=1e-3, lr_ssm=1e-3, weight_decay=1e-6):
    enc_params = list(model.encoder.parameters()) + list(model.emit_head.parameters())
    enc_ids = set(map(id, enc_params))
    ssm_params = [p for p in model.parameters() if id(p) not in enc_ids]
    return (
        torch.optim.AdamW(enc_params, lr=lr_enc, weight_decay=weight_decay),
        torch.optim.AdamW(ssm_params, lr=lr_ssm, weight_decay=weight_decay),
    )


def train_macro_transition_one_epoch(model, loader, opt_enc, opt_ssm, device, args, epoch):
    model.train()
    if epoch <= int(args.transition_warmup_enc_epochs):
        phase = "enc"
    elif epoch <= int(args.transition_warmup_enc_epochs) + int(args.transition_warmup_ssm_epochs):
        phase = "ssm"
    else:
        phase = "joint"
    rows = []
    for X, y, asset_id in loader:
        X = X.to(device)
        y = y.to(device)
        asset_id = asset_id.to(device)
        out = model(X, asset_id, detach_encoder=(phase == "ssm"))
        loss, parts = compute_macro_transition_loss(out, y, args, phase=phase)
        opt_enc.zero_grad()
        opt_ssm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if phase in ("enc", "joint"):
            opt_enc.step()
        if phase in ("ssm", "joint"):
            opt_ssm.step()
        rows.append(parts)
    log = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()} if rows else {
        "loss": 0.0,
        "macro_bce": 0.0,
        "obs_bce": 0.0,
        "alpha_bce": 0.0,
        "alpha_mean": 1.0,
        "update_mean": 0.0,
    }
    log["phase"] = phase
    return log


@torch.no_grad()
def infer_macro_transition_asset(model, X_np, asset_id, device, batch_size=None):
    model.eval()
    X = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.tensor([asset_id], dtype=torch.long, device=device)
    out = model(X, a)
    return {
        "p": out["p"].squeeze(0).cpu().numpy(),
        "q_bear": out["q_bear"].squeeze(0).cpu().numpy(),
        "q_bull": out["q_bull"].squeeze(0).cpu().numpy(),
        "turn_gate": out["turn_gate"].squeeze(0).cpu().numpy(),
        "emit_prob": torch.sigmoid(out["emit_logits"]).squeeze(0).cpu().numpy(),
        "z": out["z"].squeeze(0).cpu().numpy(),
        "h": out["h"].squeeze(0).cpu().numpy(),
    }


def evaluate_macro_transition_split(model, assets, split, device, args):
    probs, emit_probs, y_all = [], [], []
    pred_switches = true_switches = inertial_switches = turning_correct = turning_total = 0
    q_means, gate_means, deriv_maes, tvs = [], [], [], []
    for asset in assets:
        X = asset[f"X_{split}"]
        y = asset[f"y_macro_{split}"]
        if len(y) == 0:
            continue
        info = infer_macro_transition_asset(model, X, asset["asset_id"], device)
        p = info["p"]
        probs.append(p)
        emit_probs.append(info["emit_prob"])
        y_all.append(y)
        pred = (p > 0.5).astype(np.int64)
        true = (y > 0.5).astype(np.int64)
        pred_inertial = build_inertial_label(p, high=args.macro_cls_high, low=args.macro_cls_low, confirm_days=0)
        if len(pred) > 1:
            pred_switches += int(np.sum(pred[1:] != pred[:-1]))
            inertial_switches += int(np.sum(pred_inertial[1:] != pred_inertial[:-1]))
            true_switches += int(np.sum(true[1:] != true[:-1]))
            switch_idx = np.where(true[1:] != true[:-1])[0] + 1
            turning_total += int(len(switch_idx))
            if len(switch_idx):
                turning_correct += int(np.sum(pred_inertial[switch_idx] == true[switch_idx]))
            deriv_maes.append(float(np.mean(np.abs(np.diff(p) - np.diff(y)))))
            tvs.append(float(np.mean(np.abs(np.diff(p)))))
        q_means.append(float((np.mean(info["q_bear"]) + np.mean(info["q_bull"])) * 0.5))
        gate_means.append(float(np.mean(info["turn_gate"])))

    if not probs:
        return {"num_samples": 0}
    p = np.concatenate(probs)
    emit_p = np.concatenate(emit_probs)
    y = np.concatenate(y_all)
    p_clip = np.clip(p, 1e-8, 1.0 - 1e-8)
    e_clip = np.clip(emit_p, 1e-8, 1.0 - 1e-8)
    macro_bce = float(np.mean(-(y * np.log(p_clip) + (1.0 - y) * np.log1p(-p_clip))))
    emit_bce = float(np.mean(-(y * np.log(e_clip) + (1.0 - y) * np.log1p(-e_clip))))
    return {
        "num_samples": int(len(y)),
        "macro_bce": macro_bce,
        "emit_bce": emit_bce,
        "macro_acc": float(((p > 0.5) == (y > 0.5)).mean()),
        "pred_switches": int(pred_switches),
        "inertial_switches": int(inertial_switches),
        "true_switches": int(true_switches),
        "turning_acc": float(turning_correct / turning_total) if turning_total else float("nan"),
        "q_mean": float(np.mean(q_means)),
        "gate_mean": float(np.mean(gate_means)),
        "deriv_mae": float(np.mean(deriv_maes)) if deriv_maes else float("nan"),
        "tv": float(np.mean(tvs)) if tvs else float("nan"),
    }


def export_macro_transition_outputs(model, assets, args, out_dir, checkpoint_path, feature_names, device, logger):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "model": "macro_transition_gate_lite",
        "checkpoint": checkpoint_path,
        "feature_names": list(feature_names),
        "window": int(args.window),
        "normalization": args.normalization,
        "target_feature_count": int(args.target_feature_count),
        "micro_label": "disabled",
        "q_max": float(args.q_max),
        "gate_strength": float(args.gate_strength),
        "export_semantics": "ssm3_p is recurrent macro posterior; q_bear/q_bull are gated transition probabilities.",
    }
    with open(os.path.join(out_dir, "macro_transition_gate_lite_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    for asset in assets:
        info = infer_macro_transition_asset(model, asset["X_all"], asset["asset_id"], device)
        p, q_bear, q_bull = info["p"], info["q_bear"], info["q_bull"]
        pred_inertial = build_inertial_label(p, high=args.macro_cls_high, low=args.macro_cls_low, confirm_days=0)
        true = (asset["y_macro_all"] > 0.5).astype(np.int64)
        out_df = asset["df"].copy()
        idx_tgt = asset["idx_all"]
        out_df.loc[idx_tgt, "ssm3_p"] = p
        out_df.loc[idx_tgt, "ssm3_q_bear"] = q_bear
        out_df.loc[idx_tgt, "ssm3_q_bull"] = q_bull
        out_df.loc[idx_tgt, "ssm3_pred"] = pred_inertial.astype(np.int64)
        out_df.loc[idx_tgt, "ssm3_true"] = true
        out_df.loc[idx_tgt, "ssm3_pred_raw"] = (p > 0.5).astype(np.int64)
        out_df.loc[idx_tgt, "ssm3_pred_inertial"] = pred_inertial
        out_df.loc[idx_tgt, "turn_gate"] = info["turn_gate"]
        out_df.loc[idx_tgt, "macro_emit_prob"] = info["emit_prob"]
        out_df.loc[idx_tgt, "macro_soft_true"] = asset["y_macro_all"]
        out_df.to_csv(os.path.join(out_dir, f"{asset['code']}.csv"), date_format="%Y-%m-%d")
        state = {
            "h": torch.tensor(info["h"], dtype=torch.float32),
            "z": torch.tensor(info["z"], dtype=torch.float32),
            "date_idx": np.asarray(idx_tgt),
            "macro_prob": torch.tensor(p, dtype=torch.float32),
            "turn_gate": torch.tensor(info["turn_gate"], dtype=torch.float32),
        }
        torch.save(state, os.path.join(out_dir, f"{asset['code']}_ssm3_states.pt"))
        logger.info(f"[MacroTransition Export] {asset['code']}: {len(idx_tgt)} rows")


def run_macro_transition_gate_lite_pipeline(args, logger):
    if config is None:
        raise RuntimeError("未找到 utils.config")
    device = torch.device(args.device if getattr(args, "device", "") else ("cuda" if torch.cuda.is_available() else "cpu"))
    assets, feature_names = build_macro_micro_assets(args, logger)
    train_ds = MacroTransitionSeqDataset(
        assets,
        "train",
        seq_len=args.macro_transition_seq_len,
        stride=args.macro_transition_stride,
    )
    if len(train_ds) == 0:
        raise ValueError("No macro transition training sequences.")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.macro_transition_batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = MacroTransitionGateSSM(
        input_dim=len(feature_names),
        num_assets=len(assets),
        emb_dim=args.emb_dim,
        h_dim=args.lstm_hidden_dim,
        asset_emb_dim=args.asset_emb_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        gate_strength=args.gate_strength,
        q_max=args.q_max,
        beta_init=args.transition_beta_init,
    ).to(device)
    opt_enc, opt_ssm = build_macro_transition_optimizers(
        model,
        lr_enc=args.lr,
        lr_ssm=args.transition_lr_ssm,
        weight_decay=args.weight_decay,
    )
    out_dir_model = os.path.join(args.kd_path, "MacroTransitionGateLite")
    os.makedirs(out_dir_model, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(out_dir_model, "best_macro_transition_gate_lite.pt")
    if args.mode in ("train", "train_export"):
        best_val = float("inf")
        rows = []
        for ep in range(1, int(args.macro_transition_epochs) + 1):
            train_log = train_macro_transition_one_epoch(model, train_loader, opt_enc, opt_ssm, device, args, ep)
            val_metrics = evaluate_macro_transition_split(model, assets, "val", device, args)
            val_obj = val_metrics["macro_bce"] + float(args.val_deriv_weight) * val_metrics["deriv_mae"] + float(args.val_tv_weight) * val_metrics["tv"]
            row = {"epoch": ep, **{f"train_{k}": v for k, v in train_log.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, "val_obj": val_obj}
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(out_dir_model, "macro_transition_gate_lite_metrics.csv"), index=False)
            if val_obj < best_val:
                best_val = val_obj
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "input_dim": len(feature_names),
                            "num_assets": len(assets),
                            "emb_dim": int(args.emb_dim),
                            "h_dim": int(args.lstm_hidden_dim),
                            "asset_emb_dim": int(args.asset_emb_dim),
                            "lstm_layers": int(args.lstm_layers),
                            "dropout": float(args.dropout),
                            "gate_strength": float(args.gate_strength),
                            "q_max": float(args.q_max),
                            "transition_beta_init": float(args.transition_beta_init),
                            "window": int(args.window),
                            "feature_names": list(feature_names),
                            "normalization": args.normalization,
                            "target_feature_count": int(args.target_feature_count),
                        },
                        "epoch": ep,
                        "valid_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
            logger.info(
                f"[MacroTransition] ep={ep:03d}/{int(args.macro_transition_epochs)} "
                f"{train_log.get('phase', '')} train_fused={train_log.get('fused_bce', float('nan')):.5f} "
                f"val_bce={val_metrics['macro_bce']:.5f} val_deriv={val_metrics['deriv_mae']:.5f} "
                f"acc={val_metrics['macro_acc']:.3f} sw={val_metrics['pred_switches']}/{val_metrics['inertial_switches']}/{val_metrics['true_switches']} "
                f"gate={val_metrics['gate_mean']:.3f} q={val_metrics['q_mean']:.3f} best={best_val:.5f}"
            )
    elif args.mode == "export":
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing MacroTransition checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"unsupported mode for macro_transition_gate_lite: {args.mode}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    final_splits = ("train", "val", "test") if getattr(args, "eval_train_final", False) else ("val", "test")
    final_metrics = {
        split: evaluate_macro_transition_split(model, assets, split, device, args)
        for split in final_splits
    }
    with open(os.path.join(out_dir_model, "macro_transition_gate_lite_final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"[MacroTransition] final metrics: {json.dumps(final_metrics, ensure_ascii=False)}")
    if args.mode in ("train_export", "export") or getattr(args, "export_after_train", False):
        out_dir = getattr(args, "macro_micro_output_dir", "") or config.dataset["ssm_data_path"]
        export_macro_transition_outputs(model, assets, args, out_dir, checkpoint_path, feature_names, device, logger)
    return checkpoint_path, final_metrics


class AlphaStateSeqDataset(torch.utils.data.Dataset):
    def __init__(self, assets, split, seq_len=720, stride=63):
        self.assets = assets
        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.samples = []
        for ai, asset in enumerate(assets):
            n = len(asset[f"y_macro_{split}"])
            if n < self.seq_len:
                continue
            s = 0
            while s + self.seq_len <= n:
                self.samples.append((ai, s, self.seq_len))
                s += self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ai, s, length = self.samples[index]
        asset = self.assets[ai]
        split = self.split
        e = s + length
        x = asset[f"X_{split}"][s:e, -1, :]
        y = asset[f"y_macro_{split}"][s:e]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(asset["asset_id"], dtype=torch.long),
        )


class AlphaStateMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_assets,
        emb_dim=16,
        hidden_dim=64,
        asset_emb_dim=4,
        dropout=0.1,
        alpha_init_bias=2.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_assets = int(num_assets)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.asset_emb_dim = int(asset_emb_dim)
        self.asset_emb = nn.Embedding(self.num_assets, self.asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.asset_emb_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
        )
        self.alpha_lstm = nn.LSTM(
            input_size=self.input_dim + self.asset_emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.alpha_attn_score = nn.Sequential(
            nn.Linear(self.hidden_dim, max(self.hidden_dim // 2, 4)),
            nn.Tanh(),
            nn.Linear(max(self.hidden_dim // 2, 4), 1),
        )
        alpha_in = self.hidden_dim * 2 + self.emb_dim
        self.alpha_gate = nn.Sequential(
            nn.Linear(alpha_in, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.constant_(self.alpha_gate[-1].bias, float(alpha_init_bias))
        self.macro_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.emb_dim, 1),
        )

    def encode_obs(self, x, asset_id):
        if x.dim() == 2:
            asset_emb = self.asset_emb(asset_id)
            return self.obs_encoder(torch.cat([x, asset_emb], dim=-1))
        if x.dim() != 3:
            raise ValueError(f"AlphaStateMLP expects x dim 2 or 3, got {x.shape}")
        B, T, _ = x.shape
        asset_emb = self.asset_emb(asset_id)
        asset_seq = asset_emb.unsqueeze(1).expand(-1, T, -1)
        obs = self.obs_encoder(torch.cat([x, asset_seq], dim=-1).reshape(B * T, -1))
        return obs.view(B, T, self.emb_dim)

    def encode_alpha(self, x, asset_id, obs_seq):
        if x.dim() != 3:
            raise ValueError(f"AlphaStateMLP alpha path expects x [B,T,F], got {x.shape}")
        B, T, _ = x.shape
        asset_emb = self.asset_emb(asset_id)
        asset_seq = asset_emb.unsqueeze(1).expand(-1, T, -1)
        alpha_input = torch.cat([x, asset_seq], dim=-1)
        h_time, _ = self.alpha_lstm(alpha_input)

        # Efficient causal attention: context_t is the softmax-weighted average
        # of LSTM states from 0..t, so alpha_t never sees future timesteps.
        score = torch.clamp(self.alpha_attn_score(h_time).squeeze(-1), -20.0, 20.0)
        exp_score = torch.exp(score)
        num = torch.cumsum(exp_score.unsqueeze(-1) * h_time, dim=1)
        den = torch.cumsum(exp_score, dim=1).unsqueeze(-1).clamp_min(1e-8)
        attn_context = num / den

        gate_in = torch.cat([h_time, attn_context, obs_seq], dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(gate_in.reshape(B * T, -1))).view(B, T)
        if T > 0:
            alpha = alpha.clone()
            alpha[:, 0] = 1.0
        return alpha

    def forward(self, x, asset_id):
        obs_seq = self.encode_obs(x, asset_id)
        B, T, _ = obs_seq.shape
        alpha_values = self.encode_alpha(x, asset_id, obs_seq)
        states = []
        alphas = []
        state = obs_seq[:, 0, :]
        states.append(state)
        alphas.append(torch.ones(B, 1, dtype=x.dtype, device=x.device))
        for t in range(1, T):
            obs_t = obs_seq[:, t, :]
            alpha_t = alpha_values[:, t].unsqueeze(-1)
            state = alpha_t * state + (1.0 - alpha_t) * obs_t
            states.append(state)
            alphas.append(alpha_t)
        state_seq = torch.stack(states, dim=1)
        alpha_seq = torch.stack(alphas, dim=1).squeeze(-1)
        macro_logits = self.macro_head(state_seq.reshape(B * T, -1)).view(B, T)
        obs_logits = self.macro_head(obs_seq.reshape(B * T, -1)).view(B, T)
        return {
            "obs_emb": obs_seq,
            "state_emb": state_seq,
            "alpha": alpha_seq,
            "update_rate": 1.0 - alpha_seq,
            "macro_logits": macro_logits,
            "obs_logits": obs_logits,
            "macro_prob": torch.sigmoid(macro_logits),
            "obs_prob": torch.sigmoid(obs_logits),
        }


def train_alpha_state_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    rows = []
    for X, y, asset_id in loader:
        X = X.to(device)
        y = y.to(device)
        asset_id = asset_id.to(device)
        out = model(X, asset_id)
        if y.size(1) <= 1:
            continue
        target_next = y[:, 1:]
        macro_logits = out["macro_logits"][:, :-1]
        obs_logits = out["obs_logits"][:, :-1]
        macro_bce = F.binary_cross_entropy_with_logits(macro_logits, target_next)
        obs_bce = F.binary_cross_entropy_with_logits(obs_logits, target_next)
        alpha_loss = macro_logits.new_tensor(0.0)
        if y.size(1) > 2 and float(getattr(args, "alpha_trend_weight", 0.0)) > 0.0:
            alpha_target = 1.0 - torch.abs(y[:, 1:] - y[:, :-1])
            alpha_pred = torch.clamp(out["alpha"][:, 1:-1], 1e-6, 1.0 - 1e-6)
            alpha_loss = F.binary_cross_entropy(alpha_pred, alpha_target[:, 1:])
        loss = (
            macro_bce
            + float(args.alpha_obs_aux_weight) * obs_bce
            + float(getattr(args, "alpha_trend_weight", 0.0)) * alpha_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows.append({
            "loss": float(loss.detach().cpu()),
            "macro_bce": float(macro_bce.detach().cpu()),
            "obs_bce": float(obs_bce.detach().cpu()),
            "alpha_bce": float(alpha_loss.detach().cpu()),
            "alpha_mean": float(out["alpha"][:, 1:].mean().detach().cpu()) if out["alpha"].size(1) > 1 else 1.0,
            "update_mean": float(out["update_rate"][:, 1:].mean().detach().cpu()) if out["alpha"].size(1) > 1 else 0.0,
        })
    log = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()} if rows else {}
    log["phase"] = "state"
    return log


@torch.no_grad()
def infer_alpha_state_asset(model, X_np, asset_id, device):
    model.eval()
    x = np.asarray(X_np, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, -1, :]
    X = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.tensor([int(asset_id)], dtype=torch.long, device=device)
    out = model(X, a)
    return {
        "obs_emb": out["obs_emb"].squeeze(0).cpu().numpy(),
        "state_emb": out["state_emb"].squeeze(0).cpu().numpy(),
        "alpha": out["alpha"].squeeze(0).cpu().numpy(),
        "update_rate": out["update_rate"].squeeze(0).cpu().numpy(),
        "macro_prob": out["macro_prob"].squeeze(0).cpu().numpy(),
        "obs_prob": out["obs_prob"].squeeze(0).cpu().numpy(),
    }


def _auc_rank_score(score, y):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    auc = float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return max(auc, 1.0 - auc)


def _embedding_centroid_auc(z, y):
    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y) > 0.5
    if z.size == 0 or y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    c0 = z[~y].mean(axis=0)
    c1 = z[y].mean(axis=0)
    direction = c1 - c0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return float("nan")
    return _auc_rank_score(z @ (direction / norm), y)


def evaluate_alpha_state_split(model, assets, split, device, args):
    probs, obs_probs, y_all, states, obs_states = [], [], [], [], []
    pred_switches = true_switches = inertial_switches = turning_correct = turning_total = 0
    alpha_same, alpha_switch, update_same, update_switch = [], [], [], []
    for asset in assets:
        X = asset[f"X_{split}"]
        y = asset[f"y_macro_{split}"]
        if len(y) == 0:
            continue
        info = infer_alpha_state_asset(model, X, asset["asset_id"], device)
        p = info["macro_prob"]
        if len(y) <= 1:
            continue
        p_eval = p[:-1]
        obs_eval = info["obs_prob"][:-1]
        y_eval = y[1:]
        probs.append(p_eval)
        obs_probs.append(obs_eval)
        y_all.append(y_eval)
        states.append(info["state_emb"][:-1])
        obs_states.append(info["obs_emb"][:-1])
        pred = (p_eval > 0.5).astype(np.int64)
        true = (y_eval > 0.5).astype(np.int64)
        pred_inertial = build_inertial_label(p_eval, high=args.macro_cls_high, low=args.macro_cls_low, confirm_days=0)
        if len(pred) > 1:
            pred_switches += int(np.sum(pred[1:] != pred[:-1]))
            inertial_switches += int(np.sum(pred_inertial[1:] != pred_inertial[:-1]))
            true_switches += int(np.sum(true[1:] != true[:-1]))
            switch_mask = true[1:] != true[:-1]
            if np.any(~switch_mask):
                alpha_same.extend(info["alpha"][1:len(true)][~switch_mask].tolist())
                update_same.extend(info["update_rate"][1:len(true)][~switch_mask].tolist())
            if np.any(switch_mask):
                alpha_switch.extend(info["alpha"][1:len(true)][switch_mask].tolist())
                update_switch.extend(info["update_rate"][1:len(true)][switch_mask].tolist())
            switch_idx = np.where(switch_mask)[0] + 1
            turning_total += int(len(switch_idx))
            if len(switch_idx):
                turning_correct += int(np.sum(pred_inertial[switch_idx] == true[switch_idx]))

    if not probs:
        return {"num_samples": 0}
    p = np.concatenate(probs)
    obs_p = np.concatenate(obs_probs)
    y = np.concatenate(y_all)
    state_z = np.concatenate(states)
    obs_z = np.concatenate(obs_states)
    p_clip = np.clip(p, 1e-8, 1.0 - 1e-8)
    obs_clip = np.clip(obs_p, 1e-8, 1.0 - 1e-8)
    macro_bce = float(np.mean(-(y * np.log(p_clip) + (1.0 - y) * np.log1p(-p_clip))))
    obs_bce = float(np.mean(-(y * np.log(obs_clip) + (1.0 - y) * np.log1p(-obs_clip))))
    return {
        "num_samples": int(len(y)),
        "macro_bce": macro_bce,
        "obs_bce": obs_bce,
        "macro_acc": float(((p > 0.5) == (y > 0.5)).mean()),
        "pred_switches": int(pred_switches),
        "inertial_switches": int(inertial_switches),
        "true_switches": int(true_switches),
        "turning_acc": float(turning_correct / turning_total) if turning_total else float("nan"),
        "alpha_mean": float(np.mean(alpha_same + alpha_switch)) if (alpha_same or alpha_switch) else float("nan"),
        "alpha_same": float(np.mean(alpha_same)) if alpha_same else float("nan"),
        "alpha_switch": float(np.mean(alpha_switch)) if alpha_switch else float("nan"),
        "update_same": float(np.mean(update_same)) if update_same else float("nan"),
        "update_switch": float(np.mean(update_switch)) if update_switch else float("nan"),
        "state_emb_auc": _embedding_centroid_auc(state_z, y),
        "obs_emb_auc": _embedding_centroid_auc(obs_z, y),
    }


def export_alpha_state_outputs(model, assets, args, out_dir, checkpoint_path, feature_names, device, logger):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "model": "alpha_state_lite",
        "checkpoint": checkpoint_path,
        "feature_names": list(feature_names),
        "window": int(args.window),
        "normalization": args.normalization,
        "target_feature_count": int(args.target_feature_count),
        "asset_emb_dim": int(args.alpha_asset_emb_dim),
        "emb_dim": int(args.emb_dim),
        "alpha_init_bias": float(args.alpha_init_bias),
        "alpha_trend_weight": float(getattr(args, "alpha_trend_weight", 0.0)),
        "export_semantics": (
            "z/h are recursive final state_emb. obs_emb is current observation embedding. "
            "alpha is previous-state carry ratio. ssm3_p/macro_prob/p_next are predictions for t+1."
        ),
    }
    with open(os.path.join(out_dir, "alpha_state_lite_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    for asset in assets:
        info = infer_alpha_state_asset(model, asset["X_all"], asset["asset_id"], device)
        p = info["macro_prob"]
        q_bear, q_bull = _delta_proxy_q(p)
        pred_inertial = build_inertial_label(p, high=args.macro_cls_high, low=args.macro_cls_low, confirm_days=0)
        true = (asset["y_macro_all"] > 0.5).astype(np.int64)
        out_df = asset["df"].copy()
        idx_tgt = asset["idx_all"]
        out_df.loc[idx_tgt, "ssm3_p"] = p
        out_df.loc[idx_tgt, "p_next"] = p
        out_df.loc[idx_tgt, "ssm3_q_bear"] = q_bear
        out_df.loc[idx_tgt, "ssm3_q_bull"] = q_bull
        out_df.loc[idx_tgt, "ssm3_pred"] = pred_inertial.astype(np.int64)
        out_df.loc[idx_tgt, "ssm3_true"] = true
        out_df.loc[idx_tgt, "ssm3_pred_raw"] = (p > 0.5).astype(np.int64)
        out_df.loc[idx_tgt, "ssm3_pred_inertial"] = pred_inertial
        out_df.loc[idx_tgt, "alpha"] = info["alpha"]
        out_df.loc[idx_tgt, "update_rate"] = info["update_rate"]
        out_df.loc[idx_tgt, "obs_prob"] = info["obs_prob"]
        out_df.loc[idx_tgt, "macro_soft_true"] = asset["y_macro_all"]
        out_df.to_csv(os.path.join(out_dir, f"{asset['code']}.csv"), date_format="%Y-%m-%d")
        state = {
            "h": torch.tensor(info["state_emb"], dtype=torch.float32),
            "z": torch.tensor(info["state_emb"], dtype=torch.float32),
            "obs_emb": torch.tensor(info["obs_emb"], dtype=torch.float32),
            "state_emb": torch.tensor(info["state_emb"], dtype=torch.float32),
            "date_idx": np.asarray(idx_tgt),
            "macro_prob": torch.tensor(p, dtype=torch.float32),
            "p_next": torch.tensor(p, dtype=torch.float32),
            "obs_prob": torch.tensor(info["obs_prob"], dtype=torch.float32),
            "alpha": torch.tensor(info["alpha"], dtype=torch.float32),
            "update_rate": torch.tensor(info["update_rate"], dtype=torch.float32),
        }
        torch.save(state, os.path.join(out_dir, f"{asset['code']}_ssm3_states.pt"))
        logger.info(f"[AlphaState Export] {asset['code']}: {len(idx_tgt)} rows")


def run_alpha_state_lite_pipeline(args, logger):
    if config is None:
        raise RuntimeError("未找到 utils.config")
    device = torch.device(args.device if getattr(args, "device", "") else ("cuda" if torch.cuda.is_available() else "cpu"))
    assets, feature_names = build_macro_micro_assets(args, logger)
    train_ds = AlphaStateSeqDataset(
        assets,
        "train",
        seq_len=args.alpha_seq_len,
        stride=args.alpha_stride,
    )
    if len(train_ds) == 0:
        raise ValueError("No alpha-state training sequences.")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.alpha_batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = AlphaStateMLP(
        input_dim=len(feature_names),
        num_assets=len(assets),
        emb_dim=args.emb_dim,
        hidden_dim=args.alpha_hidden_dim,
        asset_emb_dim=args.alpha_asset_emb_dim,
        dropout=args.dropout,
        alpha_init_bias=args.alpha_init_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir_model = os.path.join(args.kd_path, "AlphaStateLite")
    os.makedirs(out_dir_model, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(out_dir_model, "best_alpha_state_lite.pt")
    if args.mode in ("train", "train_export"):
        best_val = float("inf")
        rows = []
        for ep in range(1, int(args.alpha_epochs) + 1):
            train_log = train_alpha_state_one_epoch(model, train_loader, optimizer, device, args, ep)
            val_metrics = evaluate_alpha_state_split(model, assets, "val", device, args)
            val_obj = val_metrics["macro_bce"]
            row = {"epoch": ep, **{f"train_{k}": v for k, v in train_log.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, "val_obj": val_obj}
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(out_dir_model, "alpha_state_lite_metrics.csv"), index=False)
            if val_obj < best_val:
                best_val = val_obj
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "input_dim": len(feature_names),
                            "num_assets": len(assets),
                            "emb_dim": int(args.emb_dim),
                            "hidden_dim": int(args.alpha_hidden_dim),
                            "asset_emb_dim": int(args.alpha_asset_emb_dim),
                            "dropout": float(args.dropout),
                            "alpha_init_bias": float(args.alpha_init_bias),
                            "alpha_trend_weight": float(getattr(args, "alpha_trend_weight", 0.0)),
                            "window": int(args.window),
                            "feature_names": list(feature_names),
                            "normalization": args.normalization,
                            "target_feature_count": int(args.target_feature_count),
                        },
                        "epoch": ep,
                        "valid_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
            logger.info(
                f"[AlphaState] ep={ep:03d}/{int(args.alpha_epochs)} {train_log.get('phase', '')} "
                f"train_macro={train_log.get('macro_bce', float('nan')):.5f} "
                f"train_obs={train_log.get('obs_bce', float('nan')):.5f} "
                f"val_macro={val_metrics['macro_bce']:.5f} val_obs={val_metrics['obs_bce']:.5f} "
                f"acc={val_metrics['macro_acc']:.3f} sw={val_metrics['pred_switches']}/{val_metrics['inertial_switches']}/{val_metrics['true_switches']} "
                f"alpha={val_metrics['alpha_mean']:.3f} emb_auc={val_metrics['state_emb_auc']:.3f} best={best_val:.5f}"
            )
    elif args.mode == "export":
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing AlphaState checkpoint: {checkpoint_path}")
    else:
        raise ValueError(f"unsupported mode for alpha_state_lite: {args.mode}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    final_splits = ("train", "val", "test") if getattr(args, "eval_train_final", False) else ("val", "test")
    final_metrics = {
        split: evaluate_alpha_state_split(model, assets, split, device, args)
        for split in final_splits
    }
    with open(os.path.join(out_dir_model, "alpha_state_lite_final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"[AlphaState] final metrics: {json.dumps(final_metrics, ensure_ascii=False)}")
    if args.mode in ("train_export", "export") or getattr(args, "export_after_train", False):
        out_dir = getattr(args, "macro_micro_output_dir", "") or config.dataset["ssm_data_path"]
        export_alpha_state_outputs(model, assets, args, out_dir, checkpoint_path, feature_names, device, logger)
    return checkpoint_path, final_metrics

def full_pipeline(KD_path: str, logger, K: int = 2, horizon: int = 1, do_train: bool = True,
                  hyperparam_grid: dict | None = None, max_map_steps: int = 50000,
                  seq_len: int = 720, stride: int = 63, batch_size: int = 256,
                  epochs: int = 200, warmup_enc_epochs: int = 60,
                  warmup_gru_epochs: int = 100, train_drop_last: bool = True):
    if config is None: raise RuntimeError("未找到 utils.config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stock_file = config.dataset["stocks_path"]
    feature_dir = config.dataset["feature_path"]
    out_dir_out = config.dataset["ssm_data_path"]
    os.makedirs(out_dir_out, exist_ok=True)
    out_dir_model = os.path.join(KD_path, "SSM3")
    os.makedirs(out_dir_model, exist_ok=True)
    ckpt_path = os.path.join(out_dir_model, "best_trendlabel_model.pth")

    with open(stock_file, "r") as f:
        codes = [line.strip() for line in f if line.strip()]

    csv_paths = [os.path.join(feature_dir, f"{c}.csv") for c in codes]

    # 训练/验证/测试数据容器（包含 Macro 和 Micro）
    all_X_tr, all_y_mac_tr, all_y_mic_tr, all_idx_tr, all_price_tr = [], [], [], [], []
    all_X_va, all_y_mac_va, all_y_mic_va, all_idx_va, all_price_va = [], [], [], [], []
    all_X_te, all_y_mac_te, all_y_mic_te, all_idx_te, all_price_te = [], [], [], [], []

    X_all_list, y_mac_all_list, y_mic_all_list, idx_all_list, price_all_list = [], [], [], [], []

    train_start, train_end = config.train_start_date, config.train_end_date
    val_start, val_end = config.valid_start_date, config.valid_end_date
    test_start, test_end = config.test_start_date, config.test_end_date

    print(f"[Pipeline] 构建双重标签数据集...")

    for a, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
        # === 核心修改：检查并生成 Label，保存回文件 ===
        df = ensure_labels_exist(df, csv_path)

        if SSM3.FEATURE_NAMES is None:
            feat_cols = [c for c in df.columns if c != "adjclose"]
        else:
            feat_cols = SSM3.FEATURE_NAMES

        # 获取 Label Series
        trend_macro = df[COL_LABEL_MAIN].values
        trend_micro = df[COL_LABEL_AUX].values

        # 归一化并对齐
        X_all, y_macro, y_micro, idx_all, price_all = SSM3.causal_rolling_normalize(
            df, feat_cols, trend_macro, trend_micro, W=config.ssm_encoder_window
        )

        X_all_list.append(X_all)
        y_mac_all_list.append(y_macro)
        y_mic_all_list.append(y_micro)
        idx_all_list.append(idx_all)
        price_all_list.append(price_all)

        mask_tr = (idx_all >= train_start) & (idx_all <= train_end)
        mask_va = (idx_all >= val_start) & (idx_all <= val_end)
        mask_te = (idx_all >= test_start) & (idx_all <= test_end)

        all_X_tr.append(X_all[mask_tr]);
        all_y_mac_tr.append(y_macro[mask_tr]);
        all_y_mic_tr.append(y_micro[mask_tr]);
        all_idx_tr.append(idx_all[mask_tr]);
        all_price_tr.append(price_all[mask_tr])
        all_X_va.append(X_all[mask_va]);
        all_y_mac_va.append(y_macro[mask_va]);
        all_y_mic_va.append(y_micro[mask_va]);
        all_idx_va.append(idx_all[mask_va]);
        all_price_va.append(price_all[mask_va])
        all_X_te.append(X_all[mask_te]);
        all_y_mac_te.append(y_macro[mask_te]);
        all_y_mic_te.append(y_micro[mask_te]);
        all_idx_te.append(idx_all[mask_te]);
        all_price_te.append(price_all[mask_te])

    # DataLoader
    hidden_dim, asset_emb_dim = 16, 8

    ds_tr = SSM3.MultiAssetWindows(all_X_tr, all_y_mac_tr, all_y_mic_tr, seq_len=seq_len, stride=stride)
    ds_va = SSM3.MultiAssetWindows(all_X_va, all_y_mac_va, all_y_mic_va, seq_len=seq_len, stride=seq_len)
    ds_te = SSM3.MultiAssetWindows(all_X_te, all_y_mac_te, all_y_mic_te, seq_len=seq_len, stride=seq_len)

    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=train_drop_last)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    x_dim = all_X_tr[0].shape[2]
    num_assets = len(csv_paths)
    logger.info(f"[SSM3] Init HazardFusionDSSM x_dim={x_dim}, num_assets={num_assets}")

    model = SSM3.HazardFusionDSSM(
        x_dim=x_dim, h_dim=hidden_dim, num_assets=num_assets, asset_emb_dim=asset_emb_dim,
        beta_init=1.0, learn_beta=True, use_asset_stickiness=True, tau_emit=1.0
    ).to(device)

    opt_enc, opt_ssm = SSM3.build_dual_optimizers(model, lr_enc=1e-3, lr_ssm=1e-3)
    REG = SSM3.get_reg_defaults()

    if do_train:
        best_last = float("inf")
        for ep in range(1, epochs + 1):
            # 传入 epoch 和 warmup_epochs
            loss_enc, loss_ssm, status = SSM3.train_loop(
                model, dl_tr, opt_enc, opt_ssm, device,
                epoch=ep,
                warmup_enc_epochs=warmup_enc_epochs,
                warmup_gru_epochs=warmup_gru_epochs,
                **REG
            )

            va_loss, val_enc_loss = SSM3.eval_loop(model, dl_va, device, **REG)
            if va_loss < best_last:
                best_last = va_loss
                torch.save({"model_state": model.state_dict()}, ckpt_path)
            logger.info(f"Ep {ep:03d} {status} | EncLoss {loss_enc:.5f} | SSMLoss {loss_ssm:.4f} | Val {va_loss:.4f} Val_EncLoss {val_enc_loss:.5f}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # 推断与保存
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    for aid, code in enumerate(codes):
        # 对该资产全时段数据进行推断
        X_tgt, y_tgt, idx_tgt = X_all_list[aid], y_mac_all_list[aid], idx_all_list[aid]
        if len(y_tgt) == 0: continue

        info = infer_states_ssm3(model, X_tgt, y_tgt, asset_id=aid, device=device)
        p, q_bear, q_bull = info["p"], info["q_bear"], info["q_bull"]
        pred, true, z, h = info["pred"], info["true"], info["z"], info["h"]

        pred_inertial = build_inertial_label(p, high=0.7, low=0.3, confirm_days=0)

        # Monitor Stats
        acc_inertial = float((pred_inertial == true).mean()) if len(pred_inertial) > 0 else 0.0
        switches_inertial = int(np.sum(pred_inertial[1:] != pred_inertial[:-1])) if len(pred_inertial) > 1 else 0
        switches = int(np.sum(pred[1:] != pred[:-1])) if len(pred) > 1 else 0
        # [NEW] Calculate True Switches for "All" phase
        switches_true = int(np.sum(true[1:] != true[:-1])) if len(true) > 1 else 0
        acc = float((pred == true).mean()) if len(pred) > 0 else float("nan")

        q_avg = (np.mean(q_bear) + np.mean(q_bull)) / 2.0
        # [NEW] Added TrueSw to the log
        logger.info(
            f"[all Raw Acc={acc:.3f} PredSw={switches} TrueSw={switches_true} | Monitor Acc={acc_inertial:.3f} Sw={switches_inertial} | q_avg={q_avg:.3f}")

        feature_csv = os.path.join(feature_dir, f"{code}.csv")
        if os.path.exists(feature_csv):
            out_df = pd.read_csv(feature_csv, parse_dates=["Date"], index_col="Date").sort_index()
            # 保存推断结果
            out_df.loc[idx_tgt, "ssm3_p"] = p
            out_df.loc[idx_tgt, "ssm3_q_bear"] = q_bear
            out_df.loc[idx_tgt, "ssm3_q_bull"] = q_bull
            out_df.loc[idx_tgt, "ssm3_pred"] = pred
            out_df.loc[idx_tgt, "ssm3_true"] = true  # 这里的 true 是宏观 trend
            out_df.loc[idx_tgt, "ssm3_pred_inertial"] = pred_inertial
            out_df.loc[idx_tgt, "ssm3_true_micro"] = y_mic_all_list[aid]

            save_csv = os.path.join(out_dir_out, f"{code}.csv")
            out_df.to_csv(save_csv, date_format="%Y-%m-%d")

            hidden_path = os.path.join(out_dir_out, f"{code}_ssm3_states.pt")
            torch.save({"h": torch.tensor(h), "z": torch.tensor(z), "date_idx": np.asarray(idx_tgt)}, hidden_path)

            fig_path = os.path.join(out_dir_out, f"{code}_ssm3_trend.png")
            plot_adjclose_with_labels(idx_tgt, out_df.loc[idx_tgt, "adjclose"].values, pred, fig_path,
                                     f"{code} - SSM3 Pred")
            logger.info(f"[SSM3 Done] {code}")

            def run_split(X, y_mac, idx, price, tag):
                if len(y_mac) == 0:
                    return
                # Use SSM3 helper for quick metrics (it doesn't return Z/H)
                p, q_bear, q_bull, pred, true = SSM3.infer_prob_asset(model, X, y_mac, aid, device)
                pred_inertial = build_inertial_label(p, high=0.7, low=0.3, confirm_days=0)
                acc_inertial = float((pred_inertial == true).mean()) if len(pred_inertial) > 0 else 0.0
                switches_inertial = int(np.sum(pred_inertial[1:] != pred_inertial[:-1])) if len(
                    pred_inertial) > 1 else 0

                # 关键：检查转折点（Turning Points）的准确率
                switch_indices = np.where(true[1:] != true[:-1])[0] + 1
                acc_at_switch = (pred[switch_indices] == true[switch_indices]).mean() if len(
                    switch_indices) > 0 else 0.0
                logger.info(f"Acc at Turning Points: {acc_at_switch:.4f}")

                switches = int(np.sum(pred[1:] != pred[:-1])) if len(pred) > 1 else 0
                # [NEW] Calculate True Switches for split phase
                switches_true = int(np.sum(true[1:] != true[:-1])) if len(true) > 1 else 0
                acc = float((pred == true).mean()) if len(pred) > 0 else float("nan")
                q_mean = (q_bear.mean() + q_bull.mean()) / 2

                # [NEW] Added TrueSw to the log
                logger.info(
                    f"[{tag}] Raw Acc={acc:.3f} PredSw={switches} TrueSw={switches_true} | Monitor Acc={acc_inertial:.3f} Sw={switches_inertial} | q_mean={q_mean:.3f}")

                outpath = os.path.join(out_dir_out, f"{code}_{tag}_trendlabel.png")
                SSM3.plot_price_true_pred(idx, price, true, pred,
                                          title=f"{tag} - {code}",
                                          outpath=outpath,
                                          pred_inertial=pred_inertial)
                logger.info(f"[{tag}] plot saved: {outpath}")

            run_split(all_X_tr[aid], all_y_mac_tr[aid], all_idx_tr[aid], all_price_tr[aid], "train")
            run_split(all_X_va[aid], all_y_mac_va[aid], all_idx_va[aid], all_price_va[aid], "val")
            run_split(all_X_te[aid], all_y_mac_te[aid], all_idx_te[aid], all_price_te[aid], "test")


def build_cli_parser():
    parser = argparse.ArgumentParser(description="SSM/TPSM training and export pipeline.")
    parser.add_argument(
        "--market",
        choices=["nas", "sh"],
        default="nas",
        help="Market config used by lightweight SSM pipelines. Default: nas.",
    )
    parser.add_argument(
        "--ssm_model_type",
        choices=["original", "risk_lite", "macro_micro_lite", "macro_transition_gate_lite", "alpha_state_lite"],
        default="original",
    )
    parser.add_argument("--mode", choices=["train", "export", "train_export"], default="train")
    parser.add_argument("--kd_path", default="checkpoints/risk_tpsm_lite")
    parser.add_argument("--do_train", action="store_true", help="Train original SSM3 before exporting.")
    parser.add_argument("--original_epochs", type=int, default=200)
    parser.add_argument("--original_seq_len", type=int, default=720)
    parser.add_argument("--original_stride", type=int, default=63)
    parser.add_argument("--original_batch_size", type=int, default=256)
    parser.add_argument("--lambda_macro", type=float, default=1.0)
    parser.add_argument("--lambda_macro_cls", type=float, default=0.1)
    parser.add_argument("--lambda_micro", type=float, default=1.0)
    parser.add_argument("--disable_micro_label", action="store_true")
    parser.add_argument("--lambda_state_center", type=float, default=0.0)
    parser.add_argument("--lambda_embedding_persist", type=float, default=0.0)
    parser.add_argument("--state_center_margin", type=float, default=1.0)
    parser.add_argument("--macro_cls_high", type=float, default=0.6)
    parser.add_argument("--macro_cls_low", type=float, default=0.4)
    parser.add_argument("--gate_strength", type=float, default=0.3)
    parser.add_argument("--gate_hidden_dim", type=int, default=0)
    parser.add_argument("--macro_micro_epochs", type=int, default=200)
    parser.add_argument("--macro_micro_batch_size", type=int, default=256)
    parser.add_argument("--alpha_epochs", type=int, default=30)
    parser.add_argument("--alpha_seq_len", type=int, default=720)
    parser.add_argument("--alpha_stride", type=int, default=63)
    parser.add_argument("--alpha_batch_size", type=int, default=128)
    parser.add_argument("--alpha_hidden_dim", type=int, default=64)
    parser.add_argument("--alpha_asset_emb_dim", type=int, default=4)
    parser.add_argument("--alpha_init_bias", type=float, default=2.0)
    parser.add_argument("--alpha_warmup_epochs", type=int, default=0)
    parser.add_argument("--alpha_obs_aux_weight", type=float, default=0.0)
    parser.add_argument("--alpha_trend_weight", type=float, default=0.05)
    parser.add_argument("--macro_transition_epochs", type=int, default=30)
    parser.add_argument("--macro_transition_seq_len", type=int, default=720)
    parser.add_argument("--macro_transition_stride", type=int, default=63)
    parser.add_argument("--macro_transition_batch_size", type=int, default=64)
    parser.add_argument("--transition_warmup_enc_epochs", type=int, default=3)
    parser.add_argument("--transition_warmup_ssm_epochs", type=int, default=7)
    parser.add_argument("--transition_lr_ssm", type=float, default=1e-3)
    parser.add_argument("--transition_beta_init", type=float, default=0.5)
    parser.add_argument("--q_max", type=float, default=0.5)
    parser.add_argument("--lambda_fuse", type=float, default=1.0)
    parser.add_argument("--lambda_emit", type=float, default=0.2)
    parser.add_argument("--lambda_macro_aux", type=float, default=0.2)
    parser.add_argument("--lambda_deriv", type=float, default=2.0)
    parser.add_argument("--transition_lambda_tv", type=float, default=0.5)
    parser.add_argument("--lambda_refra", type=float, default=5.0)
    parser.add_argument("--lambda_turn_gate", type=float, default=0.5)
    parser.add_argument("--lambda_q_target", type=float, default=0.2)
    parser.add_argument("--lambda_gate_sparse", type=float, default=0.05)
    parser.add_argument("--turn_target_scale", type=float, default=5.0)
    parser.add_argument("--val_deriv_weight", type=float, default=1.0)
    parser.add_argument("--val_tv_weight", type=float, default=0.2)
    parser.add_argument("--eval_train_final", action="store_true")
    parser.add_argument(
        "--macro_micro_output_dir",
        default="",
        help="Optional export directory for macro_micro_lite. Defaults to config.dataset['ssm_data_path'].",
    )

    from risk_tpsm_lite import add_risk_tpsm_args

    add_risk_tpsm_args(parser)
    return parser


def main():
    global config
    parser = build_cli_parser()
    args = parser.parse_args()
    config = _select_market_config(args.market)
    if args.ssm_model_type == "risk_lite":
        from risk_tpsm_lite import run_risk_tpsm_cli

        return run_risk_tpsm_cli(args)
    if args.ssm_model_type == "macro_micro_lite":
        logger = _PrintLogger()
        return run_macro_micro_lite_pipeline(args, logger)
    if args.ssm_model_type == "macro_transition_gate_lite":
        logger = _PrintLogger()
        return run_macro_transition_gate_lite_pipeline(args, logger)
    if args.ssm_model_type == "alpha_state_lite":
        logger = _PrintLogger()
        return run_alpha_state_lite_pipeline(args, logger)

    logger = _PrintLogger()
    return full_pipeline(
        args.kd_path,
        logger,
        do_train=args.do_train,
        epochs=args.original_epochs,
        seq_len=args.original_seq_len,
        stride=args.original_stride,
        batch_size=args.original_batch_size,
    )


if __name__ == "__main__":
    main()
