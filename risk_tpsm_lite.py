from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import utils.config as runtime_config
except Exception:
    runtime_config = None


EPS = 1e-8
DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_ORIGINAL_FEATURE_CANDIDATES = (
    # Always prefer the compact HRL/raw columns available in both markets.
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "amount",
    "volume",
    "amp",
    "body",
    # If the richer original SSM feature table is used, add a few diverse
    # technical factors without letting the feature count explode.
    "kmid2",
    "kup2",
    "klow",
    "ksft2",
    "roc_5",
    "roc_20",
    "std_20",
    "rank_20",
    "qtld_20",
    "cntn_20",
    "sumn_20",
)
DEFAULT_RISK_LITE_FEATURES = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "open_close_ret",
    "high_low_range",
    "vol_10d",
    "downside_vol_10d",
    "loss_ratio_10d",
    "vol_20d",
    "downside_vol_20d",
    "drawdown_20d",
    "drawdown_60d",
    "volume_ratio_20d",
)
LOG_SCALE_ORIGINAL_FEATURES = {
    "open",
    "high",
    "low",
    "close",
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "amount",
    "volume",
    "vol",
}


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _date(value):
    if value is None:
        return None
    return pd.Timestamp(value)


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_col = {c.lower(): c for c in df.columns}
    for name in candidates:
        col = lower_to_col.get(name.lower())
        if col is not None:
            return col
    return None


def _feature_key(name):
    return str(name).strip().lower()


def _prefixed_original_name(name):
    return f"orig_{_feature_key(name)}"


def _select_original_feature_columns(df, original_feature_names=None, limit=12):
    lower_to_col = {c.lower(): c for c in df.columns}
    candidates = original_feature_names or DEFAULT_ORIGINAL_FEATURE_CANDIDATES
    selected = []
    for name in candidates:
        col = lower_to_col.get(str(name).lower())
        if col is None:
            continue
        low = col.lower()
        if low.startswith(("label_", "ssm3_", "risk_", "regime_", "dd_pred")):
            continue
        if low in {"date", "weekday", "day", "month", "unnamed: 0"}:
            continue
        if col not in selected:
            selected.append(col)
        if int(limit) > 0 and len(selected) >= int(limit):
            break
    return selected


def _safe_np(x, fill=0.0):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=fill, posinf=fill, neginf=fill)


def causal_rolling_zscore(x, lookback=252, min_periods=5, clip=5.0):
    """Causal rolling z-score that only uses current and past observations."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    mean = s.rolling(int(lookback), min_periods=int(min_periods)).mean()
    std = s.rolling(int(lookback), min_periods=int(min_periods)).std(ddof=0)
    z = (s - mean) / (std + EPS)
    z = np.nan_to_num(z.to_numpy(dtype=np.float64), nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip).astype(np.float32)


def causal_rolling_minmax(x, lookback=252, min_periods=5):
    """Causal rolling min-max scaling that only uses current and past values."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    roll_min = s.rolling(int(lookback), min_periods=int(min_periods)).min()
    roll_max = s.rolling(int(lookback), min_periods=int(min_periods)).max()
    denom = roll_max - roll_min
    scaled = (s - roll_min) / (denom + EPS)
    scaled = scaled.to_numpy(dtype=np.float64)
    flat = denom.to_numpy(dtype=np.float64) <= EPS
    scaled[flat] = 0.5
    scaled = np.nan_to_num(scaled, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _normalization_type(value):
    if isinstance(value, dict):
        value = value.get("type", "zscore")
    value = str(value or "minmax").lower()
    aliases = {
        "causal_rolling_zscore": "zscore",
        "rolling_zscore": "zscore",
        "causal_rolling_minmax": "minmax",
        "rolling_minmax": "minmax",
        "maxmin": "minmax",
        "max-min": "minmax",
    }
    value = aliases.get(value, value)
    if value not in {"zscore", "minmax"}:
        raise ValueError(f"Unsupported normalization: {value}")
    return value


def normalize_feature_series(values, normalization="minmax", lookback=252, min_periods=5, clip=5.0):
    norm = _normalization_type(normalization)
    if norm == "zscore":
        return causal_rolling_zscore(values, lookback=lookback, min_periods=min_periods, clip=clip)
    return causal_rolling_minmax(values, lookback=lookback, min_periods=min_periods)


def compute_rolling_drawdown(close, lookback):
    close = pd.Series(_safe_np(close, fill=np.nan)).replace([np.inf, -np.inf], np.nan)
    roll_max = close.rolling(int(lookback), min_periods=1).max()
    dd = 1.0 - close / (roll_max + EPS)
    return np.clip(np.nan_to_num(dd.to_numpy(dtype=np.float64), nan=0.0), 0.0, None).astype(np.float32)


def compute_downside_volatility(return_1d, lookback):
    r = np.minimum(_safe_np(return_1d), 0.0)
    return (
        pd.Series(r)
        .rolling(int(lookback), min_periods=2)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def build_risk_tpsm_features(
    df: pd.DataFrame,
    window: int = 63,
    normalization_lookback: int = 252,
    normalization: str = "minmax",
    clip: float = 5.0,
    use_cross_sectional: bool = False,
    feature_preset: str = "hybrid_lite",
    original_feature_names=None,
    original_feature_limit: int = 12,
    selected_feature_names=None,
    target_feature_count: int = 0,
) -> pd.DataFrame:
    """Build causal stock-level hybrid features for Risk-TPSM-Lite.

    The default `hybrid_lite` preset combines a compact set of the original
    stock features with lightweight downside-risk features. Every selected
    column is causal-normalized by rolling min-max or z-score. No full-sample
    statistics are used.
    Cross-sectional features are intentionally a no-op here; the hook is kept so
    market-relative features can be added without changing the training entry.
    """
    del window, use_cross_sectional
    df = df.sort_index().copy()
    close_col = _find_column(df, ["adjclose", "close"])
    if close_col is None:
        raise ValueError("Risk-TPSM-Lite requires an adjclose or close column.")
    open_col = _find_column(df, ["adjopen", "open"])
    high_col = _find_column(df, ["adjhigh", "high"])
    low_col = _find_column(df, ["adjlow", "low"])
    vol_col = _find_column(df, ["volume", "amount", "vol"])

    close = np.maximum(_safe_np(df[close_col], fill=np.nan), EPS)
    open_ = np.maximum(_safe_np(df[open_col], fill=np.nan), EPS) if open_col else None
    high = np.maximum(_safe_np(df[high_col], fill=np.nan), EPS) if high_col else None
    low = np.maximum(_safe_np(df[low_col], fill=np.nan), EPS) if low_col else None
    volume = np.maximum(_safe_np(df[vol_col], fill=np.nan), EPS) if vol_col else None

    log_close = np.log(close)
    risk_raw = {}

    r_1d = np.zeros_like(close, dtype=np.float64)
    r_1d[1:] = log_close[1:] - log_close[:-1]
    risk_raw["ret_1d"] = r_1d

    for length in (5, 10, 20, 30):
        ret = np.zeros_like(close, dtype=np.float64)
        if len(close) > length:
            ret[length:] = log_close[length:] - log_close[:-length]
        risk_raw[f"ret_{length}d"] = ret

    if open_ is not None:
        risk_raw["open_close_ret"] = np.log(close / open_)
    if high is not None and low is not None:
        risk_raw["high_low_range"] = np.log(high / low)

    for length in (5, 10, 20, 30):
        risk_raw[f"vol_{length}d"] = (
            pd.Series(r_1d)
            .rolling(length, min_periods=2)
            .std(ddof=0)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        risk_raw[f"downside_vol_{length}d"] = compute_downside_volatility(r_1d, length)
        risk_raw[f"loss_ratio_{length}d"] = (
            pd.Series((r_1d < 0.0).astype(np.float64))
            .rolling(length, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    risk_raw["drawdown_20d"] = compute_rolling_drawdown(close, 20)
    risk_raw["drawdown_60d"] = compute_rolling_drawdown(close, 60)

    if volume is not None:
        log_volume = np.log(volume)
        for length in (5, 20):
            ma = (
                pd.Series(volume)
                .rolling(length, min_periods=1)
                .mean()
                .ffill()
                .fillna(1.0)
                .to_numpy(dtype=np.float64)
            )
            risk_raw[f"volume_ratio_{length}d"] = np.log(volume / (ma + EPS))
        risk_raw["volume_log"] = log_volume
        risk_raw["volume_vol_20d"] = (
            pd.Series(log_volume)
            .rolling(20, min_periods=2)
            .std(ddof=0)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    raw = {}
    if str(feature_preset) != "risk_only":
        for col in _select_original_feature_columns(
            df, original_feature_names=original_feature_names, limit=original_feature_limit
        ):
            values = _safe_np(df[col], fill=np.nan)
            if col.lower() in LOG_SCALE_ORIGINAL_FEATURES:
                values = np.log(np.maximum(values, EPS))
            raw[_prefixed_original_name(col)] = values
    raw.update(risk_raw)

    if selected_feature_names:
        feature_order = [str(name) for name in selected_feature_names]
    elif str(feature_preset) == "hybrid_lite":
        original_names = [
            _prefixed_original_name(col)
            for col in _select_original_feature_columns(
                df, original_feature_names=original_feature_names, limit=original_feature_limit
            )
        ]
        feature_order = original_names + [name for name in DEFAULT_RISK_LITE_FEATURES if name in raw]
    elif str(feature_preset) in ("risk_only", "hybrid_full"):
        feature_order = list(raw.keys())
    else:
        raise ValueError(f"Unsupported feature_preset: {feature_preset}")

    missing = [name for name in feature_order if name not in raw]
    if missing:
        raise ValueError(f"Requested RiskTPSM features are missing from dataframe: {missing[:8]}")
    if int(target_feature_count) > 0 and len(feature_order) != int(target_feature_count):
        raise ValueError(
            f"RiskTPSM feature count mismatch: got {len(feature_order)}, "
            f"expected {int(target_feature_count)}. "
            "Adjust --original_feature_limit/--original_features or set --target_feature_count 0."
        )

    features = {}
    for name in feature_order:
        features[name] = normalize_feature_series(
            raw[name],
            normalization=normalization,
            lookback=normalization_lookback,
            min_periods=5,
            clip=clip,
        )
    out = pd.DataFrame(features, index=df.index)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def _to_2d_close(close):
    arr = np.asarray(close, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("close must be 1D [T] or 2D [N, T].")
    return np.maximum(np.nan_to_num(arr, nan=np.nan), EPS)


def _rolling_std_2d(values, lookback):
    out = np.zeros_like(values, dtype=np.float64)
    for i in range(values.shape[0]):
        out[i] = (
            pd.Series(values[i])
            .rolling(int(lookback), min_periods=2)
            .std(ddof=0)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    return out


def _future_drawdown_arrays(close, horizons, vol_lookback):
    close = _to_2d_close(close)
    n_assets, n_time = close.shape
    horizons = [int(h) for h in horizons]
    log_close = np.log(close)
    ret_1d = np.zeros_like(close, dtype=np.float64)
    ret_1d[:, 1:] = log_close[:, 1:] - log_close[:, :-1]
    sigma = _rolling_std_2d(ret_1d, vol_lookback)

    dd_raw = np.full((n_assets, n_time, len(horizons)), np.nan, dtype=np.float64)
    dd_norm = np.full_like(dd_raw, np.nan)
    mask = np.zeros_like(dd_raw, dtype=bool)

    for hi, h in enumerate(horizons):
        dd = np.zeros((n_assets, n_time), dtype=np.float64)
        valid = np.zeros((n_assets, n_time), dtype=bool)
        for u in range(1, h + 1):
            if u >= n_time:
                break
            rel = log_close[:, u:] - log_close[:, :-u]
            drop = np.maximum(-rel, 0.0)
            dd[:, : n_time - u] = np.maximum(dd[:, : n_time - u], drop)
        if h < n_time:
            valid[:, : n_time - h] = True
        denom = sigma * math.sqrt(float(h)) + EPS
        dd_raw[:, :, hi] = dd
        dd_norm[:, :, hi] = dd / denom
        mask[:, :, hi] = valid & np.isfinite(dd_norm[:, :, hi])
    return dd_norm.astype(np.float32), dd_raw.astype(np.float32), mask


def _broadcast_time_mask(train_mask, n_assets, n_time):
    if train_mask is None:
        return np.ones((n_assets, n_time), dtype=bool)
    arr = np.asarray(train_mask, dtype=bool)
    if arr.ndim == 1:
        if arr.shape[0] != n_time:
            raise ValueError("1D train_mask length must match time length.")
        return np.broadcast_to(arr[None, :], (n_assets, n_time))
    if arr.shape != (n_assets, n_time):
        raise ValueError("2D train_mask must have shape [N, T].")
    return arr


def build_future_drawdown_risk_labels(
    close,
    horizons=DEFAULT_HORIZONS,
    vol_lookback: int = 20,
    threshold_quantile: float = 0.7,
    tau: float | list[float] = 0.25,
    train_mask=None,
    thresholds=None,
):
    """Build multi-horizon future path drawdown risk labels.

    Thresholds are estimated only on train_mask when not explicitly supplied.
    """
    close_2d = _to_2d_close(close)
    n_assets, n_time = close_2d.shape
    horizons = [int(h) for h in horizons]
    dd_norm, dd_raw, label_mask = _future_drawdown_arrays(close_2d, horizons, vol_lookback)
    train_mask_2d = _broadcast_time_mask(train_mask, n_assets, n_time)

    if thresholds is None:
        thresholds = []
        for hi in range(len(horizons)):
            valid = label_mask[:, :, hi] & train_mask_2d & np.isfinite(dd_norm[:, :, hi])
            vals = dd_norm[:, :, hi][valid]
            if vals.size == 0:
                thresholds.append(1.0)
            else:
                thresholds.append(float(np.quantile(vals, threshold_quantile)))
    thresholds = np.asarray(thresholds, dtype=np.float32)
    tau_arr = np.asarray(_as_list(tau), dtype=np.float32)
    if tau_arr.size == 1:
        tau_arr = np.repeat(tau_arr, len(horizons))

    y_risk = np.zeros_like(dd_norm, dtype=np.float32)
    for hi in range(len(horizons)):
        logits = (dd_norm[:, :, hi] - thresholds[hi]) / (float(tau_arr[hi]) + EPS)
        y = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        y_risk[:, :, hi] = np.where(label_mask[:, :, hi], y, 0.0)

    if np.asarray(close).ndim == 1:
        return y_risk[0], dd_norm[0], dd_raw[0], label_mask[0], thresholds
    return y_risk, dd_norm, dd_raw, label_mask, thresholds


def build_soft_regime_labels(
    close,
    horizons=DEFAULT_HORIZONS,
    vol_lookback: int = 20,
    b_up: float = 0.25,
    b_down: float = 0.25,
    b_flat: float = 0.5,
    tau_reg: float = 0.5,
):
    """Build soft up/flat/down regime labels. State order: up, flat, down."""
    original_ndim = np.asarray(close).ndim
    close = _to_2d_close(close)
    n_assets, n_time = close.shape
    horizons = [int(h) for h in horizons]
    log_close = np.log(close)
    ret_1d = np.zeros_like(close, dtype=np.float64)
    ret_1d[:, 1:] = log_close[:, 1:] - log_close[:, :-1]
    sigma = _rolling_std_2d(ret_1d, vol_lookback)

    y_regime = np.zeros((n_assets, n_time, len(horizons), 3), dtype=np.float32)
    mask = np.zeros((n_assets, n_time, len(horizons)), dtype=bool)
    for hi, h in enumerate(horizons):
        z = np.zeros((n_assets, n_time), dtype=np.float64)
        valid = np.zeros((n_assets, n_time), dtype=bool)
        if h < n_time:
            ret_h = log_close[:, h:] - log_close[:, :-h]
            denom = sigma[:, :-h] * math.sqrt(float(h)) + EPS
            z[:, :-h] = ret_h / denom
            valid[:, :-h] = np.isfinite(z[:, :-h])
        s_up = (z - b_up) / (tau_reg + EPS)
        s_down = (-z - b_down) / (tau_reg + EPS)
        s_flat = -np.abs(z) / (b_flat + EPS)
        logits = np.stack([s_up, s_flat, s_down], axis=-1)
        logits = logits - np.nanmax(logits, axis=-1, keepdims=True)
        probs = np.exp(np.clip(logits, -60.0, 60.0))
        probs = probs / (probs.sum(axis=-1, keepdims=True) + EPS)
        y_regime[:, :, hi, :] = np.where(valid[:, :, None], probs, 0.0).astype(np.float32)
        mask[:, :, hi] = valid

    if original_ndim == 1:
        return y_regime[0], mask[0]
    return y_regime, mask


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.left_pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)


class RiskTPSMLite(nn.Module):
    """Light temporal encoder for stock-level downside-risk embeddings."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 16,
        num_horizons: int = 4,
        tcn_channels: int = 32,
        dropout: float = 0.1,
        use_severity_head: bool = False,
        use_asset_conditioning: bool = False,
        num_assets: int = 0,
        asset_emb_dim: int = 8,
        use_attention_pooling: bool = False,
        encoder_type: str = "attention_lstm",
        lstm_hidden_dim: int | None = None,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.emb_dim = int(emb_dim)
        self.num_horizons = int(num_horizons)
        self.use_severity_head = bool(use_severity_head)
        self.use_asset_conditioning = bool(use_asset_conditioning)
        self.num_assets = int(num_assets)
        self.asset_emb_dim = int(asset_emb_dim)
        self.encoder_type = str(encoder_type)
        self.tcn_channels = int(tcn_channels)
        self.lstm_hidden_dim = int(lstm_hidden_dim or tcn_channels)
        self.lstm_layers = int(lstm_layers)
        self.use_attention_pooling = bool(use_attention_pooling) or self.encoder_type == "attention_lstm"

        if self.encoder_type == "tcn":
            encoder_dim = int(tcn_channels)
            self.encoder = nn.Sequential(
                CausalConvBlock(self.in_dim, encoder_dim, kernel_size=3, dilation=1, dropout=dropout),
                CausalConvBlock(encoder_dim, encoder_dim, kernel_size=3, dilation=2, dropout=dropout),
                CausalConvBlock(encoder_dim, encoder_dim, kernel_size=3, dilation=4, dropout=0.0),
            )
        elif self.encoder_type == "attention_lstm":
            encoder_dim = self.lstm_hidden_dim
            lstm_dropout = float(dropout) if self.lstm_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=self.in_dim,
                hidden_size=encoder_dim,
                num_layers=self.lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        if self.use_attention_pooling:
            self.attn_pool = nn.Sequential(
                nn.Linear(encoder_dim, max(encoder_dim // 2, 4)),
                nn.Tanh(),
                nn.Linear(max(encoder_dim // 2, 4), 1),
            )
            self.pool_mix = nn.Parameter(torch.tensor(0.5))
        else:
            self.attn_pool = None
            self.pool_mix = None
        if self.use_asset_conditioning:
            if self.num_assets <= 0:
                raise ValueError("num_assets must be positive when use_asset_conditioning=True.")
            self.asset_emb = nn.Embedding(self.num_assets, self.asset_emb_dim)
            self.asset_film = nn.Sequential(
                nn.Linear(self.asset_emb_dim, encoder_dim * 2),
                nn.Tanh(),
            )
            self.asset_risk_bias = nn.Embedding(self.num_assets, self.num_horizons)
            nn.init.zeros_(self.asset_risk_bias.weight)
        else:
            self.asset_emb = None
            self.asset_film = None
            self.asset_risk_bias = None
        self.emb_proj = nn.Linear(encoder_dim, self.emb_dim)
        self.emb_ln = nn.LayerNorm(self.emb_dim)
        self.risk_head = nn.Linear(self.emb_dim, self.num_horizons)
        self.regime_head = nn.Linear(self.emb_dim, self.num_horizons * 3)
        self.severity_head = nn.Linear(self.emb_dim, self.num_horizons) if self.use_severity_head else None

    def forward(self, x, asset_id=None):
        assert x.dim() == 3, f"expected x [B, W, F], got {tuple(x.shape)}"
        assert x.size(-1) == self.in_dim, f"expected F={self.in_dim}, got {x.size(-1)}"
        if self.encoder_type == "tcn":
            h = self.encoder(x.transpose(1, 2))
            last = h[:, :, -1]
            if self.use_attention_pooling:
                h_time = h.transpose(1, 2)
                attn = torch.softmax(self.attn_pool(h_time).squeeze(-1), dim=-1)
                pooled = torch.sum(h_time * attn.unsqueeze(-1), dim=1)
                mix = torch.sigmoid(self.pool_mix)
                last = mix * last + (1.0 - mix) * pooled
        else:
            h_time, _ = self.encoder(x)
            last = h_time[:, -1, :]
            attn = torch.softmax(self.attn_pool(h_time).squeeze(-1), dim=-1)
            pooled = torch.sum(h_time * attn.unsqueeze(-1), dim=1)
            mix = torch.sigmoid(self.pool_mix)
            last = mix * last + (1.0 - mix) * pooled
        if self.use_asset_conditioning:
            if asset_id is None:
                raise ValueError("asset_id is required when use_asset_conditioning=True.")
            asset_id = asset_id.to(device=x.device, dtype=torch.long).view(-1)
            asset_context = self.asset_emb(asset_id)
            gamma, beta = self.asset_film(asset_context).chunk(2, dim=-1)
            last = last * (1.0 + 0.1 * gamma) + 0.1 * beta
        embedding = self.emb_ln(self.emb_proj(last))
        risk_logits = self.risk_head(embedding)
        if self.use_asset_conditioning:
            risk_logits = risk_logits + 0.1 * self.asset_risk_bias(asset_id)
        q_risk = torch.sigmoid(risk_logits)
        regime_logits = self.regime_head(embedding).view(-1, self.num_horizons, 3)
        regime_probs = torch.softmax(regime_logits, dim=-1)
        out = {
            "embedding": embedding,
            "risk_logits": risk_logits,
            "q_risk": q_risk,
            "regime_logits": regime_logits,
            "regime_probs": regime_probs,
        }
        if self.severity_head is not None:
            out["severity"] = F.softplus(self.severity_head(embedding))
        return out


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def pairwise_ranking_loss(risk_logits, y_risk, mask, label_margin=0.15, max_pairs=4096):
    """RankNet-style pairwise loss to improve cross-sample risk ordering."""
    losses = []
    num_horizons = risk_logits.size(1)
    for hi in range(num_horizons):
        valid = mask[:, hi].bool()
        if int(valid.sum().item()) < 2:
            continue
        scores = risk_logits[valid, hi]
        labels = y_risk[valid, hi]
        label_diff = labels[:, None] - labels[None, :]
        pair_mask = label_diff.abs() >= float(label_margin)
        if not bool(pair_mask.any().item()):
            continue
        score_diff = scores[:, None] - scores[None, :]
        signed_diff = torch.sign(label_diff[pair_mask]) * score_diff[pair_mask]
        if int(max_pairs) > 0 and signed_diff.numel() > int(max_pairs):
            perm = torch.randperm(signed_diff.numel(), device=signed_diff.device)[: int(max_pairs)]
            signed_diff = signed_diff[perm]
        losses.append(F.softplus(-signed_diff).mean())
    if not losses:
        return risk_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_risk_tpsm_loss(
    outputs,
    batch,
    stage: int = 3,
    pos_weight=None,
    lambda_regime: float = 0.3,
    lambda_brier: float = 0.1,
    lambda_tv: float = 1e-3,
    lambda_severity: float = 0.0,
    lambda_rank: float = 0.0,
    rank_label_margin: float = 0.15,
    rank_max_pairs: int = 4096,
    use_weighted_bce: bool = True,
    risk_loss_type: str = "bce",
    focal_gamma: float = 2.0,
):
    q_risk = outputs["q_risk"]
    regime_probs = outputs["regime_probs"]
    y_risk = batch["y_risk"]
    y_regime = batch["y_regime"]
    mask = batch["mask"].bool()

    assert q_risk.shape == y_risk.shape
    assert regime_probs.shape == y_regime.shape
    assert mask.shape == y_risk.shape

    q_safe = torch.clamp(q_risk, EPS, 1.0 - EPS)
    if use_weighted_bce and pos_weight is not None:
        pw = pos_weight.to(q_risk.device).view(1, -1)
        bce = -(pw * y_risk * torch.log(q_safe) + (1.0 - y_risk) * torch.log1p(-q_safe))
    else:
        bce = F.binary_cross_entropy(q_safe, y_risk, reduction="none")
    if risk_loss_type == "focal":
        bce = bce * torch.abs(y_risk - q_risk).pow(float(focal_gamma))
    loss_risk = _masked_mean(bce, mask)
    loss_brier = _masked_mean((q_risk - y_risk).pow(2), mask)

    log_regime = torch.log(torch.clamp(regime_probs, EPS, 1.0))
    regime_ce = -(y_regime * log_regime).sum(dim=-1)
    loss_regime = _masked_mean(regime_ce, mask)

    loss_tv = q_risk.new_tensor(0.0)
    if stage >= 3 and "q_risk_prev" in outputs:
        prev_valid = batch["prev_valid"].bool().view(-1, 1) & mask
        loss_tv = _masked_mean(torch.abs(q_risk - outputs["q_risk_prev"]), prev_valid)

    loss_severity = q_risk.new_tensor(0.0)
    if "severity" in outputs and lambda_severity > 0.0:
        severity = outputs["severity"]
        loss_severity = _masked_mean(F.huber_loss(severity, batch["dd_norm"], reduction="none"), mask)

    loss_rank = q_risk.new_tensor(0.0)
    if lambda_rank > 0.0:
        loss_rank = pairwise_ranking_loss(
            outputs["risk_logits"],
            y_risk,
            mask,
            label_margin=rank_label_margin,
            max_pairs=rank_max_pairs,
        )

    total = loss_risk + lambda_brier * loss_brier
    if stage >= 2:
        total = total + lambda_regime * loss_regime
    if stage >= 3:
        total = total + lambda_tv * loss_tv
    if lambda_severity > 0.0:
        total = total + lambda_severity * loss_severity
    if lambda_rank > 0.0:
        total = total + lambda_rank * loss_rank

    return total, {
        "loss": float(total.detach().cpu()),
        "risk_bce": float(loss_risk.detach().cpu()),
        "brier": float(loss_brier.detach().cpu()),
        "regime_ce": float(loss_regime.detach().cpu()),
        "tv": float(loss_tv.detach().cpu()),
        "severity": float(loss_severity.detach().cpu()),
        "rank": float(loss_rank.detach().cpu()),
    }


def _make_windows(values, window):
    values = np.asarray(values, dtype=np.float32)
    n_time = values.shape[0]
    if n_time < window:
        return np.zeros((0, window, values.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    endpoints = np.arange(window - 1, n_time, dtype=np.int64)
    out = np.empty((len(endpoints), window, values.shape[1]), dtype=np.float32)
    for i, end in enumerate(endpoints):
        out[i] = values[end - window + 1 : end + 1]
    return out, endpoints


class RiskTPSMWindowDataset(Dataset):
    def __init__(self, assets, split: str, train_start=None, train_end=None, valid_start=None, valid_end=None):
        self.assets = assets
        self.samples = []
        if split == "train":
            start, end = _date(train_start), _date(train_end)
        elif split == "valid":
            start, end = _date(valid_start), _date(valid_end)
        elif split == "all":
            start, end = None, None
        else:
            raise ValueError(f"unknown split: {split}")
        for ai, asset in enumerate(assets):
            dates = pd.to_datetime(asset["window_dates"])
            if start is None:
                date_mask = np.ones(len(dates), dtype=bool)
            else:
                date_mask = (dates >= start) & (dates <= end)
            valid_label = asset["mask"].any(axis=1)
            for wi in np.where(date_mask & valid_label)[0]:
                self.samples.append((ai, int(wi)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ai, wi = self.samples[index]
        asset = self.assets[ai]
        prev_wi = max(wi - 1, 0)
        return {
            "x": torch.tensor(asset["X"][wi], dtype=torch.float32),
            "x_prev": torch.tensor(asset["X"][prev_wi], dtype=torch.float32),
            "asset_id": torch.tensor(ai, dtype=torch.long),
            "prev_valid": torch.tensor(wi > 0, dtype=torch.bool),
            "y_risk": torch.tensor(asset["y_risk"][wi], dtype=torch.float32),
            "y_regime": torch.tensor(asset["y_regime"][wi], dtype=torch.float32),
            "dd_norm": torch.tensor(asset["dd_norm"][wi], dtype=torch.float32),
            "mask": torch.tensor(asset["mask"][wi], dtype=torch.bool),
        }


def _read_stock_codes(data_dir, stock_file=None, max_stocks=None):
    data_dir = Path(data_dir)
    if stock_file:
        with open(stock_file, "r") as f:
            codes = [line.strip() for line in f if line.strip()]
    else:
        codes = sorted(p.stem for p in data_dir.glob("*.csv") if not p.name.endswith("_states.csv"))
    codes = [c for c in codes if (data_dir / f"{c}.csv").exists()]
    if max_stocks:
        codes = codes[: int(max_stocks)]
    return codes


def _read_stock_frame(path, max_rows=None):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.index.name = "Date"
    if max_rows:
        df = df.iloc[: int(max_rows)].copy()
    return df


def prepare_risk_tpsm_assets(args, thresholds=None, need_labels=True):
    horizons = [int(h) for h in args.horizons]
    codes = _read_stock_codes(args.data_dir, getattr(args, "stock_file", None), getattr(args, "max_stocks", None))
    if not codes:
        raise FileNotFoundError(f"No stock CSVs found under {args.data_dir}")

    raw_assets = []
    train_dd_values = [[] for _ in horizons]
    for asset_id, code in enumerate(codes):
        path = Path(args.data_dir) / f"{code}.csv"
        df = _read_stock_frame(path, max_rows=getattr(args, "max_rows", None))
        feat_df = build_risk_tpsm_features(
            df,
            window=args.window,
            normalization_lookback=args.normalization_lookback,
            normalization=getattr(args, "normalization", "minmax"),
            clip=args.feature_clip,
            feature_preset=getattr(args, "feature_preset", "hybrid_lite"),
            original_feature_names=getattr(args, "original_features", None),
            original_feature_limit=getattr(args, "original_feature_limit", 12),
            selected_feature_names=getattr(args, "feature_names", None),
            target_feature_count=getattr(args, "target_feature_count", 0),
        )
        close_col = _find_column(df, ["adjclose", "close"])
        close = np.maximum(_safe_np(df[close_col], fill=np.nan), EPS)
        X, endpoints = _make_windows(feat_df.values, args.window)
        if len(endpoints) == 0:
            continue
        dates = pd.to_datetime(df.index)
        window_dates = dates[endpoints]
        asset = {
            "code": code,
            "asset_id": asset_id,
            "df": df,
            "feature_names": list(feat_df.columns),
            "X": X,
            "endpoints": endpoints,
            "window_dates": window_dates,
            "close": close,
        }
        if need_labels:
            _, dd_norm, dd_raw, mask, _ = build_future_drawdown_risk_labels(
                close,
                horizons=horizons,
                vol_lookback=args.vol_lookback,
                threshold_quantile=args.threshold_quantile,
                tau=args.tau_risk,
                thresholds=np.zeros(len(horizons), dtype=np.float32),
            )
            train_mask = np.asarray((dates >= _date(args.train_start)) & (dates <= _date(args.train_end)))
            for hi in range(len(horizons)):
                valid = mask[:, hi] & train_mask
                vals = dd_norm[:, hi][valid & np.isfinite(dd_norm[:, hi])]
                if vals.size:
                    train_dd_values[hi].append(vals)
            asset.update({"dd_norm_full": dd_norm, "dd_raw_full": dd_raw, "label_mask_full": mask})
        raw_assets.append(asset)

    if not raw_assets:
        raise ValueError("No assets have enough rows for the requested window.")

    feature_names = raw_assets[0]["feature_names"]
    for asset in raw_assets:
        if asset["feature_names"] != feature_names:
            raise ValueError("Feature name mismatch across assets.")

    if need_labels:
        if thresholds is None:
            thresholds = []
            for vals in train_dd_values:
                merged = np.concatenate(vals) if vals else np.asarray([], dtype=np.float32)
                thresholds.append(float(np.quantile(merged, args.threshold_quantile)) if merged.size else 1.0)
        thresholds = np.asarray(thresholds, dtype=np.float32)
        for asset in raw_assets:
            dd_norm = asset["dd_norm_full"]
            mask = asset["label_mask_full"]
            y_risk = np.zeros_like(dd_norm, dtype=np.float32)
            for hi, _ in enumerate(horizons):
                logits = (dd_norm[:, hi] - thresholds[hi]) / (float(args.tau_risk) + EPS)
                y = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
                y_risk[:, hi] = np.where(mask[:, hi], y, 0.0)
            y_regime, regime_mask = build_soft_regime_labels(
                asset["close"],
                horizons=horizons,
                vol_lookback=args.vol_lookback,
                b_up=args.b_up,
                b_down=args.b_down,
                b_flat=args.b_flat,
                tau_reg=args.tau_reg,
            )
            endpoints = asset["endpoints"]
            asset["y_risk"] = y_risk[endpoints]
            asset["y_regime"] = y_regime[endpoints]
            asset["dd_norm"] = dd_norm[endpoints]
            asset["mask"] = (mask[endpoints] & regime_mask[endpoints]).astype(bool)
        return raw_assets, feature_names, thresholds

    return raw_assets, feature_names, thresholds


def compute_pos_weight_from_dataset(dataset, num_horizons):
    pos = np.zeros(num_horizons, dtype=np.float64)
    neg = np.zeros(num_horizons, dtype=np.float64)
    for ai, wi in dataset.samples:
        asset = dataset.assets[ai]
        mask = asset["mask"][wi].astype(bool)
        y = asset["y_risk"][wi]
        pos += np.where(mask, y, 0.0)
        neg += np.where(mask, 1.0 - y, 0.0)
    return torch.tensor(neg / (pos + EPS), dtype=torch.float32).clamp(0.1, 20.0)


def _move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _binary_auc(y_true, y_score):
    y = np.asarray(y_true) >= 0.5
    s = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def evaluate_risk_tpsm(model, loader, device, args, pos_weight=None, stage=3):
    model.eval()
    losses = []
    q_all, y_all, mask_all = [], [], []
    regime_pred, regime_true = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["x"], batch.get("asset_id"))
        if stage >= 3:
            out["q_risk_prev"] = model(batch["x_prev"], batch.get("asset_id"))["q_risk"]
        _, loss_parts = compute_risk_tpsm_loss(
            out,
            batch,
            stage=stage,
            pos_weight=pos_weight,
            lambda_regime=args.lambda_regime,
            lambda_brier=args.lambda_brier,
            lambda_tv=args.lambda_tv,
            lambda_severity=args.lambda_severity if args.use_severity_head else 0.0,
            lambda_rank=args.lambda_rank,
            rank_label_margin=args.rank_label_margin,
            rank_max_pairs=args.rank_max_pairs,
            use_weighted_bce=args.use_weighted_bce,
            risk_loss_type=args.risk_loss_type,
            focal_gamma=args.focal_gamma,
        )
        losses.append(loss_parts)
        q_all.append(out["q_risk"].cpu().numpy())
        y_all.append(batch["y_risk"].cpu().numpy())
        mask_all.append(batch["mask"].cpu().numpy())
        regime_pred.append(out["regime_probs"].cpu().numpy())
        regime_true.append(batch["y_regime"].cpu().numpy())

    if not q_all:
        return {"risk_bce_mean": float("inf"), "num_samples": 0}

    q = np.concatenate(q_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    m = np.concatenate(mask_all, axis=0).astype(bool)
    rp = np.concatenate(regime_pred, axis=0)
    rt = np.concatenate(regime_true, axis=0)
    metrics = {"num_samples": int(q.shape[0])}
    for key in losses[0].keys():
        metrics[key] = float(np.mean([x[key] for x in losses]))

    bce_vals, brier_vals = [], []
    for hi, h in enumerate(args.horizons):
        valid = m[:, hi]
        qh = np.clip(q[:, hi][valid], EPS, 1.0 - EPS)
        yh = y[:, hi][valid]
        if yh.size == 0:
            bce = brier = mean_pred = pos_rate = auc = float("nan")
        else:
            bce = float(np.mean(-(yh * np.log(qh) + (1.0 - yh) * np.log1p(-qh))))
            brier = float(np.mean((qh - yh) ** 2))
            mean_pred = float(np.mean(qh))
            pos_rate = float(np.mean(yh))
            auc = _binary_auc(yh, qh)
        metrics[f"risk_bce_h{h}"] = bce
        metrics[f"brier_h{h}"] = brier
        metrics[f"mean_q_h{h}"] = mean_pred
        metrics[f"label_rate_h{h}"] = pos_rate
        metrics[f"auc_h{h}"] = auc
        bce_vals.append(bce)
        brier_vals.append(brier)

    pred_cls = np.argmax(rp, axis=-1)
    true_cls = np.argmax(rt, axis=-1)
    valid_reg = m
    metrics["regime_hard_acc"] = float(np.mean(pred_cls[valid_reg] == true_cls[valid_reg])) if valid_reg.any() else float("nan")
    flat_down = valid_reg & ((true_cls == 1) | (true_cls == 2))
    metrics["flat_down_confusion"] = float(np.mean(pred_cls[flat_down] != true_cls[flat_down])) if flat_down.any() else float("nan")
    metrics["risk_bce_mean"] = float(np.nanmean(bce_vals))
    metrics["brier_mean"] = float(np.nanmean(brier_vals))
    return metrics


def _train_one_epoch(model, loader, optimizer, device, args, pos_weight, stage):
    model.train()
    logs = []
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["x"], batch.get("asset_id"))
        if stage >= 3:
            out["q_risk_prev"] = model(batch["x_prev"], batch.get("asset_id"))["q_risk"].detach()
        loss, loss_parts = compute_risk_tpsm_loss(
            out,
            batch,
            stage=stage,
            pos_weight=pos_weight,
            lambda_regime=args.lambda_regime,
            lambda_brier=args.lambda_brier,
            lambda_tv=args.lambda_tv,
            lambda_severity=args.lambda_severity if args.use_severity_head else 0.0,
            lambda_rank=args.lambda_rank,
            rank_label_margin=args.rank_label_margin,
            rank_max_pairs=args.rank_max_pairs,
            use_weighted_bce=args.use_weighted_bce,
            risk_loss_type=args.risk_loss_type,
            focal_gamma=args.focal_gamma,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logs.append(loss_parts)
    return {k: float(np.mean([x[k] for x in logs])) for k in logs[0].keys()} if logs else {}


def _serializable_args(args, feature_dim=None, feature_names=None, thresholds=None):
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    data.setdefault("encoder_type", getattr(args, "encoder_type", "attention_lstm"))
    data.setdefault(
        "lstm_hidden_dim",
        int(getattr(args, "lstm_hidden_dim", getattr(args, "tcn_channels", 32))),
    )
    data.setdefault("lstm_layers", int(getattr(args, "lstm_layers", 1)))
    data["horizons"] = [int(h) for h in data["horizons"]]
    if feature_dim is not None:
        data["feature_dim"] = int(feature_dim)
    if feature_names is not None:
        data["feature_names"] = list(feature_names)
    if thresholds is not None:
        data["label_thresholds"] = [float(x) for x in thresholds]
    norm_type = _normalization_type(getattr(args, "normalization", "minmax"))
    data["normalization"] = {
        "type": norm_type,
        "lookback": int(args.normalization_lookback),
    }
    if norm_type == "zscore":
        data["normalization"]["clip"] = float(args.feature_clip)
    else:
        data["normalization"]["feature_range"] = [0.0, 1.0]
    return data


def selection_score(metrics, selection_metric="risk_bce_mean", bce_weight=0.25):
    auc_keys = [key for key in metrics if key.startswith("auc_h")]
    auc_mean = float(np.nanmean([metrics[key] for key in auc_keys])) if auc_keys else float("nan")
    bce = float(metrics.get("risk_bce_mean", float("inf")))
    brier = float(metrics.get("brier_mean", float("inf")))
    metric = str(selection_metric)
    if metric == "risk_bce_mean":
        return -bce
    if metric == "auc_mean":
        return auc_mean
    if metric == "brier_mean":
        return -brier
    if metric == "bce_auc_combo":
        return auc_mean - float(bce_weight) * bce
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def selected_checkpoint_path(checkpoint_policy, best_path, final_path):
    policy = str(checkpoint_policy)
    if policy == "best":
        return Path(best_path)
    if policy in ("final", "best_and_final"):
        return Path(final_path)
    raise ValueError(f"Unsupported checkpoint_policy: {checkpoint_policy}")


def train_risk_tpsm(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    assets, feature_names, thresholds = prepare_risk_tpsm_assets(args, need_labels=True)
    train_ds = RiskTPSMWindowDataset(
        assets,
        "train",
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )
    valid_ds = RiskTPSMWindowDataset(
        assets,
        "valid",
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise ValueError(f"Empty train/valid set: train={len(train_ds)} valid={len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    pos_weight = compute_pos_weight_from_dataset(train_ds, len(args.horizons)) if args.use_weighted_bce else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    model = RiskTPSMLite(
        in_dim=len(feature_names),
        emb_dim=args.emb_dim,
        num_horizons=len(args.horizons),
        tcn_channels=args.tcn_channels,
        dropout=args.dropout,
        use_severity_head=args.use_severity_head,
        use_asset_conditioning=args.use_asset_conditioning,
        num_assets=len(assets),
        asset_emb_dim=args.asset_emb_dim,
        use_attention_pooling=args.use_attention_pooling,
        encoder_type=getattr(args, "encoder_type", "attention_lstm"),
        lstm_hidden_dim=getattr(args, "lstm_hidden_dim", getattr(args, "tcn_channels", 32)),
        lstm_layers=getattr(args, "lstm_layers", 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cfg = _serializable_args(args, feature_dim=len(feature_names), feature_names=feature_names, thresholds=thresholds)
    cfg["num_assets"] = len(assets)
    cfg["asset_codes"] = [asset["code"] for asset in assets]
    best_score = -float("inf")
    best_metric = float("inf")
    best_path = Path(args.checkpoint_dir) / "risk_tpsm_lite_best.pt"
    final_path = Path(args.checkpoint_dir) / "risk_tpsm_lite_final.pt"
    metrics_path = Path(args.output_dir) / "risk_tpsm_lite_metrics.jsonl"
    metrics_csv_path = Path(args.output_dir) / "risk_tpsm_lite_metrics.csv"
    stage_plan = [(1, args.stage1_epochs), (2, args.stage2_epochs), (3, args.stage3_epochs)]
    global_epoch = 0
    flat_rows = []
    last_valid_metrics = None
    last_selection_score = None

    def flatten_metrics(row):
        out = {
            "epoch": row["epoch"],
            "stage": row["stage"],
            "best_valid_risk_bce": best_metric,
            "best_selection_score": best_score,
        }
        for prefix in ("train", "valid"):
            for key, value in row[prefix].items():
                if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                    out[f"{prefix}_{key}"] = float(value)
        return out

    with open(metrics_path, "w") as mf:
        for stage, epochs in stage_plan:
            for _ in range(int(epochs)):
                global_epoch += 1
                train_log = _train_one_epoch(model, train_loader, optimizer, device, args, pos_weight, stage)
                valid_metrics = evaluate_risk_tpsm(model, valid_loader, device, args, pos_weight=pos_weight, stage=stage)
                row = {
                    "epoch": global_epoch,
                    "stage": stage,
                    "train": train_log,
                    "valid": valid_metrics,
                    "thresholds": [float(x) for x in thresholds],
                    "pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else None,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                mf.flush()
                flat_rows.append(flatten_metrics(row))
                pd.DataFrame(flat_rows).to_csv(metrics_csv_path, index=False)
                auc_text = " ".join(
                    f"auc_h{h}={valid_metrics.get(f'auc_h{h}', float('nan')):.3f}"
                    for h in args.horizons
                )
                gpu_text = ""
                if device.type == "cuda":
                    gpu_text = f" gpu_mem={torch.cuda.memory_allocated(device) / (1024 ** 2):.0f}MB"
                current_score = selection_score(
                    valid_metrics,
                    selection_metric=args.selection_metric,
                    bce_weight=args.selection_bce_weight,
                )
                last_valid_metrics = valid_metrics
                last_selection_score = current_score
                print(
                    f"[RiskTPSM] ep={global_epoch:03d} stage={stage} "
                    f"train_bce={train_log.get('risk_bce', float('nan')):.5f} "
                    f"val_bce={valid_metrics.get('risk_bce_mean', float('nan')):.5f} "
                    f"val_brier={valid_metrics.get('brier_mean', float('nan')):.5f} "
                    f"sel={current_score:.5f} {auc_text}{gpu_text}",
                    flush=True,
                )
                if current_score > best_score:
                    best_score = current_score
                    best_metric = valid_metrics["risk_bce_mean"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": cfg,
                            "horizons": [int(h) for h in args.horizons],
                            "label_thresholds": thresholds,
                            "pos_weight": pos_weight.detach().cpu() if pos_weight is not None else None,
                            "epoch": global_epoch,
                            "valid_metrics": valid_metrics,
                            "selection_metric": args.selection_metric,
                            "selection_score": best_score,
                            "checkpoint_policy": "best",
                        },
                        best_path,
                    )

    if last_valid_metrics is None:
        raise ValueError("No training epochs were run. Increase stage epoch counts before saving a checkpoint.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "horizons": [int(h) for h in args.horizons],
            "label_thresholds": thresholds,
            "pos_weight": pos_weight.detach().cpu() if pos_weight is not None else None,
            "epoch": global_epoch,
            "valid_metrics": last_valid_metrics,
            "selection_metric": args.selection_metric,
            "selection_score": last_selection_score,
            "checkpoint_policy": "final",
        },
        final_path,
    )
    selected_path = selected_checkpoint_path(args.checkpoint_policy, best_path, final_path)

    with open(Path(args.output_dir) / "risk_tpsm_lite_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(
        f"[RiskTPSM] best checkpoint: {best_path} "
        f"selection={args.selection_metric} score={best_score:.6f} valid_risk_bce={best_metric:.6f}"
    )
    print(
        f"[RiskTPSM] final checkpoint: {final_path} "
        f"epoch={global_epoch} selection={args.selection_metric} "
        f"score={last_selection_score:.6f} valid_risk_bce={last_valid_metrics.get('risk_bce_mean', float('nan')):.6f}"
    )
    print(f"[RiskTPSM] selected checkpoint ({args.checkpoint_policy}): {selected_path}")
    if args.export_after_train:
        export_args = SimpleNamespace(**vars(args))
        export_args.checkpoint = str(selected_path)
        export_risk_tpsm_outputs(export_args)
    return selected_path


def map_risk_outputs_to_legacy(q_risk, horizons):
    q = np.asarray(q_risk, dtype=np.float32)
    horizons = [int(h) for h in horizons]
    aggregate = np.mean(q, axis=-1)
    short_idx = int(np.argmin(np.abs(np.asarray(horizons) - 5)))
    mid_target = 20 if 20 in horizons else 30
    mid_idx = int(np.argmin(np.abs(np.asarray(horizons) - mid_target)))
    q_bear = q[..., short_idx]
    q_bull = q[..., mid_idx]
    return aggregate.astype(np.float32), q_bear.astype(np.float32), q_bull.astype(np.float32)


@torch.no_grad()
def _infer_asset(model, X, device, batch_size=512, asset_id=None):
    model.eval()
    embeddings, risks, regimes, severities = [], [], [], []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        aid = None
        if model.use_asset_conditioning:
            if asset_id is None:
                raise ValueError("asset_id is required for asset-conditioned export.")
            aid = torch.full((xb.size(0),), int(asset_id), dtype=torch.long, device=device)
        out = model(xb, aid)
        embeddings.append(out["embedding"].cpu())
        risks.append(out["q_risk"].cpu())
        regimes.append(out["regime_probs"].cpu())
        if "severity" in out:
            severities.append(out["severity"].cpu())
    emb = torch.cat(embeddings, dim=0) if embeddings else torch.zeros(0, model.emb_dim)
    q = torch.cat(risks, dim=0) if risks else torch.zeros(0, model.num_horizons)
    reg = torch.cat(regimes, dim=0) if regimes else torch.zeros(0, model.num_horizons, 3)
    sev = torch.cat(severities, dim=0) if severities else None
    return emb, q, reg, sev


def _merge_checkpoint_config(args, ckpt_config):
    merged = dict(ckpt_config)
    if "normalization" not in merged:
        merged["normalization"] = {
            "type": "zscore",
            "lookback": int(merged.get("normalization_lookback", 252)),
            "clip": float(merged.get("feature_clip", 5.0)),
        }
    if "encoder_type" not in merged:
        merged["encoder_type"] = "tcn"
    for key in ("data_dir", "output_dir", "stock_file", "checkpoint", "device", "max_stocks", "max_rows"):
        value = getattr(args, key, None)
        if value not in (None, ""):
            merged[key] = value
    return SimpleNamespace(**merged)


def export_risk_tpsm_outputs(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = _merge_checkpoint_config(args, ckpt.get("config", {}))
    cfg.horizons = [int(h) for h in getattr(cfg, "horizons", ckpt.get("horizons", DEFAULT_HORIZONS))]
    device = torch.device(getattr(cfg, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
    assets, feature_names, _ = prepare_risk_tpsm_assets(cfg, thresholds=ckpt.get("label_thresholds"), need_labels=False)
    in_dim = int(getattr(cfg, "feature_dim", len(feature_names)))
    model = RiskTPSMLite(
        in_dim=in_dim,
        emb_dim=int(cfg.emb_dim),
        num_horizons=len(cfg.horizons),
        tcn_channels=int(cfg.tcn_channels),
        dropout=float(cfg.dropout),
        use_severity_head=bool(getattr(cfg, "use_severity_head", False)),
        use_asset_conditioning=bool(getattr(cfg, "use_asset_conditioning", False)),
        num_assets=int(getattr(cfg, "num_assets", len(assets))),
        asset_emb_dim=int(getattr(cfg, "asset_emb_dim", 8)),
        use_attention_pooling=bool(getattr(cfg, "use_attention_pooling", False)),
        encoder_type=str(getattr(cfg, "encoder_type", "tcn")),
        lstm_hidden_dim=int(getattr(cfg, "lstm_hidden_dim", getattr(cfg, "tcn_channels", 32))),
        lstm_layers=int(getattr(cfg, "lstm_layers", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_manifest = {
        "checkpoint": str(args.checkpoint),
        "horizons": cfg.horizons,
        "feature_names": feature_names,
        "normalization": getattr(cfg, "normalization", "minmax"),
        "encoder_type": getattr(cfg, "encoder_type", "tcn"),
        "legacy_semantics": (
            "In risk_lite mode, ssm3_p/ssm3_q_bear/ssm3_q_bull are downside-risk "
            "proxies, not original uptrend or transition probabilities."
        ),
    }

    for asset in assets:
        code = asset["code"]
        emb, q_risk, regime, severity = _infer_asset(
            model,
            asset["X"],
            device,
            batch_size=getattr(cfg, "infer_batch_size", 512),
            asset_id=asset.get("asset_id"),
        )
        q_np = q_risk.numpy()
        reg_np = regime.numpy()
        aggregate, q_bear, q_bull = map_risk_outputs_to_legacy(q_np, cfg.horizons)

        out_df = asset["df"].copy()
        idx_tgt = pd.to_datetime(asset["window_dates"])
        for hi, h in enumerate(cfg.horizons):
            out_df.loc[idx_tgt, f"risk_h{h}"] = q_np[:, hi]
            out_df.loc[idx_tgt, f"regime_up_h{h}"] = reg_np[:, hi, 0]
            out_df.loc[idx_tgt, f"regime_flat_h{h}"] = reg_np[:, hi, 1]
            out_df.loc[idx_tgt, f"regime_down_h{h}"] = reg_np[:, hi, 2]
            if severity is not None:
                out_df.loc[idx_tgt, f"dd_pred_h{h}"] = severity.numpy()[:, hi]
        out_df.loc[idx_tgt, "ssm3_p"] = aggregate
        out_df.loc[idx_tgt, "ssm3_q_bear"] = q_bear
        out_df.loc[idx_tgt, "ssm3_q_bull"] = q_bull
        out_df.loc[idx_tgt, "ssm3_pred"] = (aggregate > 0.5).astype(np.int64)
        out_df.loc[idx_tgt, "ssm3_pred_inertial"] = (aggregate > 0.5).astype(np.int64)

        csv_path = out_dir / f"{code}.csv"
        out_df.to_csv(csv_path, date_format="%Y-%m-%d")
        state = {
            "embedding": emb,
            "q_risk": q_risk,
            "regime": regime,
            "severity": severity,
            "date_idx": np.asarray(idx_tgt),
            "z": emb,
            "h": emb,
            "horizons": cfg.horizons,
            "legacy_semantics": export_manifest["legacy_semantics"],
        }
        torch.save(state, out_dir / f"{code}_risk_tpsm_states.pt")
        torch.save(state, out_dir / f"{code}_ssm3_states.pt")
        print(f"[RiskTPSM Export] {code}: {csv_path} embedding={tuple(emb.shape)}")

    with open(out_dir / "risk_tpsm_export_manifest.json", "w") as f:
        json.dump(export_manifest, f, indent=2, ensure_ascii=False)
    return out_dir


def add_risk_tpsm_args(parser: argparse.ArgumentParser):
    dataset = getattr(runtime_config, "dataset", {}) if runtime_config is not None else {}
    parser.add_argument("--data_dir", default=dataset.get("ssm_data_path", "Dataset/Nas100数据/feature_ssm"))
    parser.add_argument("--stock_file", default=dataset.get("stocks_path", None))
    parser.add_argument("--output_dir", default="results/risk_tpsm_lite")
    parser.add_argument("--checkpoint_dir", default="checkpoints/risk_tpsm_lite")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--checkpoint_policy",
        choices=["best", "final", "best_and_final"],
        default="final",
        help=(
            "Checkpoint used for train_export/export_after_train. "
            "final trains through all configured epochs and exports the converged last epoch."
        ),
    )
    parser.add_argument("--export_after_train", action="store_true")
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument(
        "--normalization",
        choices=["minmax", "zscore"],
        default="minmax",
        help="Causal rolling feature normalization. minmax uses current/past rolling min and max.",
    )
    parser.add_argument("--normalization_lookback", type=int, default=252)
    parser.add_argument("--feature_clip", type=float, default=5.0)
    parser.add_argument(
        "--feature_preset",
        choices=["hybrid_lite", "hybrid_full", "risk_only"],
        default="hybrid_lite",
        help=(
            "hybrid_lite uses compact original stock features plus compact risk features; "
            "risk_only preserves the old pure risk-feature path."
        ),
    )
    parser.add_argument(
        "--original_features",
        nargs="*",
        default=None,
        help="Optional ordered original feature names to include before risk features.",
    )
    parser.add_argument(
        "--feature_names",
        nargs="*",
        default=None,
        help=(
            "Optional exact normalized feature names used by build_risk_tpsm_features, "
            "for example orig_adjopen orig_adjhigh ..."
        ),
    )
    parser.add_argument("--original_feature_limit", type=int, default=12)
    parser.add_argument(
        "--target_feature_count",
        type=int,
        default=25,
        help="Require this exact input feature count after feature selection. Use 0 to disable.",
    )
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument(
        "--encoder_type",
        choices=["attention_lstm", "tcn"],
        default="attention_lstm",
        help="Temporal encoder used to learn the risk embedding.",
    )
    parser.add_argument("--tcn_channels", type=int, default=32)
    parser.add_argument("--lstm_hidden_dim", type=int, default=32)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_asset_conditioning", action="store_true")
    parser.add_argument("--asset_emb_dim", type=int, default=8)
    parser.add_argument("--use_attention_pooling", action="store_true")
    parser.add_argument("--threshold_quantile", type=float, default=0.7)
    parser.add_argument("--tau_risk", type=float, default=0.25)
    parser.add_argument("--b_up", type=float, default=0.25)
    parser.add_argument("--b_down", type=float, default=0.25)
    parser.add_argument("--b_flat", type=float, default=0.5)
    parser.add_argument("--tau_reg", type=float, default=0.5)
    parser.add_argument("--lambda_regime", type=float, default=0.3)
    parser.add_argument("--lambda_brier", type=float, default=0.1)
    parser.add_argument("--lambda_tv", type=float, default=1e-3)
    parser.add_argument("--lambda_severity", type=float, default=0.0)
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--rank_label_margin", type=float, default=0.15)
    parser.add_argument("--rank_max_pairs", type=int, default=4096)
    parser.add_argument("--risk_loss_type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--selection_metric",
        choices=["risk_bce_mean", "auc_mean", "brier_mean", "bce_auc_combo"],
        default="risk_bce_mean",
    )
    parser.add_argument("--selection_bce_weight", type=float, default=0.25)
    parser.add_argument("--use_severity_head", action="store_true")
    parser.add_argument("--no_weighted_bce", dest="use_weighted_bce", action="store_false")
    parser.set_defaults(use_weighted_bce=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage3_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--infer_batch_size", type=int, default=512)
    parser.add_argument("--max_stocks", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--train_start", default=getattr(runtime_config, "train_start_date", "2000-04-07"))
    parser.add_argument("--train_end", default=getattr(runtime_config, "train_end_date", "2017-12-29"))
    parser.add_argument("--valid_start", default=getattr(runtime_config, "valid_start_date", "2018-01-02"))
    parser.add_argument("--valid_end", default=getattr(runtime_config, "valid_end_date", "2020-04-22"))
    return parser


def run_risk_tpsm_cli(args):
    if args.mode == "train":
        return train_risk_tpsm(args)
    if args.mode == "export":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode export")
        return export_risk_tpsm_outputs(args)
    if args.mode == "train_export":
        args.export_after_train = True
        return train_risk_tpsm(args)
    raise ValueError(f"unsupported mode: {args.mode}")
