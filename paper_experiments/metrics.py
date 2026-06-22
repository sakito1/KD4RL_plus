import warnings
from typing import Dict

import numpy as np
import pandas as pd


def _finite_series(values) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def max_drawdown(values) -> float:
    series = _finite_series(values)
    if len(series) < 2:
        return 0.0
    peaks = series.cummax().clip(lower=1e-12)
    return float(((peaks - series) / peaks).max())


def compute_financial_metrics(portfolio_trace: pd.DataFrame) -> Dict[str, float]:
    if portfolio_trace is None or portfolio_trace.empty or "portfolio_value" not in portfolio_trace:
        warnings.warn("empty portfolio trace; financial metrics are NaN", RuntimeWarning)
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "final_value": np.nan,
            "daily_win_rate": np.nan,
        }
    values = _finite_series(portfolio_trace["portfolio_value"])
    if len(values) < 2:
        warnings.warn("portfolio trace too short; financial metrics are mostly NaN", RuntimeWarning)
        final = float(values.iloc[-1]) if len(values) else np.nan
        return {
            "total_return": 0.0,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": 0.0,
            "calmar": np.nan,
            "final_value": final,
            "daily_win_rate": np.nan,
        }
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / max(values.iloc[0], 1e-12) - 1.0)
    ann_return = float(returns.mean() * 252.0) if len(returns) else np.nan
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else np.nan
    downside = returns[returns < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(252.0)) if len(downside) > 1 else np.nan
    sharpe = float(ann_return / ann_vol) if ann_vol and np.isfinite(ann_vol) and ann_vol > 1e-12 else np.nan
    sortino = (
        float(ann_return / downside_vol)
        if downside_vol and np.isfinite(downside_vol) and downside_vol > 1e-12
        else np.nan
    )
    mdd = max_drawdown(values)
    calmar = float(ann_return / mdd) if mdd > 1e-12 and np.isfinite(ann_return) else np.nan
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "final_value": float(values.iloc[-1]),
        "daily_win_rate": float((returns > 0.0).mean()) if len(returns) else np.nan,
    }


def compute_trading_metrics(portfolio_trace: pd.DataFrame) -> Dict[str, float]:
    if portfolio_trace is None or portfolio_trace.empty:
        return {
            "switch_count": 0,
            "free_switch_count": 0,
            "forced_switch_count": 0,
            "forced_hold_count": 0,
            "avg_holding_duration": np.nan,
            "median_holding_duration": np.nan,
            "holding_duration_std": np.nan,
            "turnover_sum": 0.0,
            "turnover_mean": np.nan,
            "transaction_cost_sum": 0.0,
            "over_trading_index": 0.0,
        }
    df = portfolio_trace
    switch_count = int(pd.to_numeric(df.get("is_switch", 0), errors="coerce").fillna(0).sum())
    free_switch_count = int(pd.to_numeric(df.get("is_free_switch", 0), errors="coerce").fillna(0).sum())
    forced_switch_count = int(pd.to_numeric(df.get("is_forced_switch", 0), errors="coerce").fillna(0).sum())
    forced_hold_count = int(pd.to_numeric(df.get("is_forced_hold", 0), errors="coerce").fillna(0).sum())
    duration = _finite_series(df.get("holding_duration", pd.Series(dtype=float)))
    turnover = pd.to_numeric(df.get("turnover", pd.Series(dtype=float)), errors="coerce")
    cost = pd.to_numeric(df.get("cost_rate", pd.Series(dtype=float)), errors="coerce")
    return {
        "switch_count": switch_count,
        "free_switch_count": free_switch_count,
        "forced_switch_count": forced_switch_count,
        "forced_hold_count": forced_hold_count,
        "avg_holding_duration": float(duration.mean()) if len(duration) else np.nan,
        "median_holding_duration": float(duration.median()) if len(duration) else np.nan,
        "holding_duration_std": float(duration.std(ddof=1)) if len(duration) > 1 else np.nan,
        "turnover_sum": float(turnover.fillna(0.0).sum()),
        "turnover_mean": float(turnover.mean()) if turnover.notna().any() else np.nan,
        "transaction_cost_sum": float(cost.fillna(0.0).sum()),
        "over_trading_index": float(free_switch_count / max(1, switch_count)),
    }


def summarize_inner_alpha(portfolio_trace: pd.DataFrame) -> Dict[str, float]:
    if portfolio_trace is None or portfolio_trace.empty or "inner_alpha" not in portfolio_trace:
        warnings.warn("missing inner_alpha trace", RuntimeWarning)
        return {
            "cumulative_inner_alpha": np.nan,
            "mean_inner_alpha": np.nan,
            "inner_alpha_std": np.nan,
            "positive_inner_alpha_ratio": np.nan,
            "inner_alpha_sharpe": np.nan,
            "mean_abs_inner_alpha": np.nan,
            "mean_turnover": np.nan,
            "inner_alpha_per_turnover": np.nan,
        }
    alpha = _finite_series(portfolio_trace["inner_alpha"])
    turnover = _finite_series(portfolio_trace.get("turnover", pd.Series(dtype=float)))
    mean_alpha = float(alpha.mean()) if len(alpha) else np.nan
    std_alpha = float(alpha.std(ddof=1)) if len(alpha) > 1 else np.nan
    alpha_sharpe = (
        float(mean_alpha / std_alpha * np.sqrt(252.0))
        if std_alpha and np.isfinite(std_alpha) and std_alpha > 1e-12
        else np.nan
    )
    mean_turnover = float(turnover.mean()) if len(turnover) else np.nan
    return {
        "cumulative_inner_alpha": float(alpha.sum()) if len(alpha) else np.nan,
        "mean_inner_alpha": mean_alpha,
        "inner_alpha_std": std_alpha,
        "positive_inner_alpha_ratio": float((alpha > 0.0).mean()) if len(alpha) else np.nan,
        "inner_alpha_sharpe": alpha_sharpe,
        "mean_abs_inner_alpha": float(alpha.abs().mean()) if len(alpha) else np.nan,
        "mean_turnover": mean_turnover,
        "inner_alpha_per_turnover": (
            float(alpha.sum() / max(turnover.sum(), 1e-12)) if len(alpha) and len(turnover) else np.nan
        ),
    }


def summarize_controller_alignment(action_trace: pd.DataFrame) -> Dict[str, float]:
    if action_trace is None or action_trace.empty:
        return {}
    df = action_trace.copy()
    free = df[df.get("decision_type", "") == "free_decision"].copy()
    if free.empty and "is_locked" in df:
        free = df[pd.to_numeric(df["is_locked"], errors="coerce").fillna(1) == 0].copy()
    if free.empty:
        warnings.warn("no free controller decisions in action trace", RuntimeWarning)
        return {
            "n_free_decisions": 0,
            "n_free_switches": 0,
            "switch_advantage_mean": np.nan,
            "switch_advantage_positive_ratio": np.nan,
            "exit_prob_mean": np.nan,
            "exit_prob_switch_mean": np.nan,
            "exit_prob_hold_mean": np.nan,
            "exit_prob_pos_adv_mean": np.nan,
            "exit_prob_neg_adv_mean": np.nan,
            "exit_gap": np.nan,
            "switch_precision": np.nan,
            "switch_recall": np.nan,
            "corr_exit_prob_switch_advantage": np.nan,
            "corr_policy_logit_switch_advantage": np.nan,
        }
    adv = pd.to_numeric(free.get("controller_switch_advantage"), errors="coerce")
    exit_prob = pd.to_numeric(free.get("exit_prob"), errors="coerce")
    logit = pd.to_numeric(free.get("policy_logit"), errors="coerce")
    switched = pd.to_numeric(free.get("is_switch"), errors="coerce").fillna(0).astype(int) == 1
    pos = adv > 0.0
    def corr(x, y):
        data = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(data) < 2 or data["x"].std() <= 1e-12 or data["y"].std() <= 1e-12:
            return np.nan
        return float(data["x"].corr(data["y"]))
    pos_mean = float(exit_prob[pos].mean()) if pos.any() else np.nan
    neg_mean = float(exit_prob[~pos].mean()) if (~pos).any() else np.nan
    return {
        "n_free_decisions": int(len(free)),
        "n_free_switches": int((switched & (pd.to_numeric(free.get("is_free_switch"), errors="coerce").fillna(0) == 1)).sum()),
        "switch_advantage_mean": float(adv.mean()) if adv.notna().any() else np.nan,
        "switch_advantage_positive_ratio": float(pos.mean()) if len(pos) else np.nan,
        "exit_prob_mean": float(exit_prob.mean()) if exit_prob.notna().any() else np.nan,
        "exit_prob_switch_mean": float(exit_prob[switched].mean()) if switched.any() else np.nan,
        "exit_prob_hold_mean": float(exit_prob[~switched].mean()) if (~switched).any() else np.nan,
        "exit_prob_pos_adv_mean": pos_mean,
        "exit_prob_neg_adv_mean": neg_mean,
        "exit_gap": float(pos_mean - neg_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan,
        "switch_precision": float(pos[switched].mean()) if switched.any() else np.nan,
        "switch_recall": float(switched[pos].mean()) if pos.any() else np.nan,
        "corr_exit_prob_switch_advantage": corr(exit_prob, adv),
        "corr_policy_logit_switch_advantage": corr(logit, adv),
    }


def summarize_all(portfolio_trace: pd.DataFrame) -> Dict[str, float]:
    metrics = {}
    metrics.update(compute_financial_metrics(portfolio_trace))
    metrics.update(compute_trading_metrics(portfolio_trace))
    metrics.update(summarize_inner_alpha(portfolio_trace))
    return metrics

