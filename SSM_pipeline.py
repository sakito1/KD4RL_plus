import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 依赖 SSM3
import SSM3
from create_soft_regime_label import build_ensemble_soft_label, build_fullseries_segment_labels_k2, build_smooth_ensemble_label, build_fullseries_prob_labels
try:
    import utils.config as config
except Exception:
    config = None


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
    parser.add_argument("--ssm_model_type", choices=["original", "risk_lite"], default="original")
    parser.add_argument("--mode", choices=["train", "export", "train_export"], default="train")
    parser.add_argument("--kd_path", default="checkpoints/risk_tpsm_lite")
    parser.add_argument("--do_train", action="store_true", help="Train original SSM3 before exporting.")
    parser.add_argument("--original_epochs", type=int, default=200)
    parser.add_argument("--original_seq_len", type=int, default=720)
    parser.add_argument("--original_stride", type=int, default=63)
    parser.add_argument("--original_batch_size", type=int, default=256)

    from risk_tpsm_lite import add_risk_tpsm_args

    add_risk_tpsm_args(parser)
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()
    if args.ssm_model_type == "risk_lite":
        from risk_tpsm_lite import run_risk_tpsm_cli

        return run_risk_tpsm_cli(args)

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
